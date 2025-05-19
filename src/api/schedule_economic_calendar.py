# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/api/schedule_economic_calendar.py
# Collecte des événements macroéconomiques (ex. : FOMC, NFP) pour enrichir les features contextuelles
# et stockage dans data/macro_events.csv avec volatilité calculée.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Récupère les événements macro via l’API Investing.com (~50 USD/mois, premium),
#        applique une clusterisation (PCA + K-means, méthode 7) pour stocker les patterns
#        dans market_memory.db, calcule la volatilité des prix ES, et planifie les mises à jour.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, requests>=2.28.0,<3.0.0, sklearn>=1.2.0,<2.0.0, psutil>=5.9.8,<6.0.0,
#   pyyaml>=6.0.0,<7.0.0, sqlite3, boto3>=1.26.0,<2.0.0, schedule>=1.2.0,<2.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
# - src/utils/telegram_alert.py
# - src/model/utils/db_setup.py
# - src/api/data_provider.py
#
# Inputs :
# - config/credentials.yaml (clés API Investing.com)
# - config/es_config.yaml
# - data/iqfeed/iqfeed_data.csv (pour volatilité)
#
# Outputs :
# - data/macro_events.csv
# - data/event_volatility_history.csv
# - data/economic_calendar_snapshots/*.json.gz
# - data/logs/economic_calendar_performance.csv
# - data/logs/economic_calendar.log
# - data/checkpoints/economic_calendar_*.json.gz
# - market_memory.db (table clusters)
#
# Notes :
# - Conforme à structure.txt (version 2.1.3, 2025-05-13).
# - Utilise exclusivement l’API Investing.com (https://www.investing.com/economic-calendar/, ~50 USD/mois pour premium).
# - Implémente retries (max 3, délai 2^attempt), logs psutil, alertes Telegram, snapshots compressés.
# - Intègre mémoire contextuelle (Phase 7) via PCA (10 dimensions, 95% variance) et K-means (10 clusters).
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des données.
# - Intègre validation SHAP (Phase 17) pour les features utilisées (ex. : event_impact, event_type).
# - Calcule la volatilité des prix ES autour des événements (fenêtre 30 min, IQFeed).
# - Planifie les mises à jour (quotidienne, horaire, ou toutes les 15 min, configurable).
# - Phases intégrées : Phase 1 (collecte via API), Phase 7 (mémoire contextuelle), Phase 8 (auto-conscience),
#                     Phase 17 (interprétabilité SHAP).
# - Sauvegardes incrémentielles toutes les 5 min dans data/checkpoints/ avec versionnage (5 versions).
# - Sauvegardes distribuées toutes les 15 min vers S3 (configurable).
# - Tests unitaires dans tests/test_schedule_economic_calendar.py.
# - Validation complète prévue pour juin 2025.
# - Évolution future : Intégration d’autres sources d’événements (juin 2025).

import gzip
import json
import logging
import os
import signal
import sqlite3
import time
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import boto3
import pandas as pd
import psutil
import requests
import schedule
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.api.data_provider import get_data_provider
from src.model.utils.alert_manager import send_alert
from src.model.utils.config_manager import get_config
from src.model.utils.db_setup import setup_database
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.utils.telegram_alert import send_telegram_alert

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
EVENTS_CSV_PATH = os.path.join(BASE_DIR, "data", "macro_events.csv")
VOLATILITY_CSV_PATH = os.path.join(BASE_DIR, "data", "event_volatility_history.csv")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "economic_calendar_snapshots")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "data", "checkpoints")
CSV_LOG_PATH = os.path.join(
    BASE_DIR, "data", "logs", "economic_calendar_performance.csv"
)
DB_PATH = os.path.join(BASE_DIR, "data", "market_memory.db")
FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR, "data", "features", "feature_importance.csv"
)

# Configuration du logging
os.makedirs(os.path.join(BASE_DIR, "data", "logs"), exist_ok=True)
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "data", "logs", "economic_calendar.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Variable pour gérer l'arrêt propre
RUNNING = True


class EconomicCalendar:
    """
    Collecte et stocke les événements macroéconomiques (ex. : FOMC, NFP) via l’API Investing.com.
    """

    def __init__(
        self,
        config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml"),
        credentials_path: str = os.path.join(BASE_DIR, "config", "credentials.yaml"),
    ):
        """
        Initialise le collecteur d’événements macro.

        Args:
            config_path (str): Chemin vers le fichier de configuration.
            credentials_path (str): Chemin vers le fichier des identifiants.
        """
        self.log_buffer = []
        self.cache = {}
        self.buffer_size = 100
        self.max_cache_size = 1000
        self.checkpoint_versions = []
        try:
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            self.config = self.load_config_with_validation(config_path)
            self.credentials = self.load_config_with_validation(credentials_path)
            self.api_key = self.credentials.get("investing_com_api_key", "")
            if not self.api_key:
                raise ValueError(
                    "Clé API Investing.com manquante dans credentials.yaml"
                )
            self.data_provider = get_data_provider("IQFeedProvider")
            setup_database()  # Initialise market_memory.db
            signal.signal(signal.SIGINT, self.signal_handler)
            success_msg = "EconomicCalendar initialisé"
            miya_speak(
                success_msg, tag="ECONOMIC_CALENDAR", voice_profile="calm", priority=2
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("init", 0, success=True)
            self.save_snapshot(
                "init", {"config": self.config, "api_key": "masked"}, compress=True
            )
        except Exception as e:
            error_msg = f"Erreur initialisation EconomicCalendar: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="ECONOMIC_CALENDAR", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            self.config = {
                "api_url": "https://www.investing.com/economic-calendar/",
                "s3_bucket": None,
                "s3_prefix": "economic_calendar/",
                "frequency": "daily",
                "scheduled_time": "00:00",
                "cache_hours": 24,
                "volatility_window_minutes": 30,
            }
            self.credentials = {"investing_com_api_key": ""}
            self.data_provider = None

    def load_config_with_validation(self, config_path: str) -> Dict:
        """
        Charge et valide la configuration.

        Args:
            config_path (str): Chemin vers le fichier de configuration.

        Returns:
            Dict: Configuration validée.
        """
        start_time = datetime.now()
        try:
            config = get_config(os.path.basename(config_path))
            section = (
                "economic_calendar" if "es_config" in config_path else "credentials"
            )
            if section not in config:
                config[section] = {}
            required_keys = (
                [
                    "api_url",
                    "frequency",
                    "scheduled_time",
                    "cache_hours",
                    "volatility_window_minutes",
                ]
                if section == "economic_calendar"
                else ["investing_com_api_key"]
            )
            missing_keys = [key for key in required_keys if key not in config[section]]
            if missing_keys:
                raise ValueError(
                    f"Clés de configuration manquantes dans {section}: {missing_keys}"
                )
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Configuration {section} chargée"
            miya_speak(
                success_msg, tag="ECONOMIC_CALENDAR", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "load_config_with_validation",
                latency,
                success=True,
                config_section=section,
            )
            return config[section]
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur chargement configuration {config_path}: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="ECONOMIC_CALENDAR", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "load_config_with_validation", latency, success=False, error=str(e)
            )
            return (
                {
                    "api_url": "https://www.investing.com/economic-calendar/",
                    "frequency": "daily",
                    "scheduled_time": "00:00",
                    "cache_hours": 24,
                    "volatility_window_minutes": 30,
                }
                if "es_config" in config_path
                else {"investing_com_api_key": ""}
            )

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ) -> None:
        """
        Journalise les performances des opérations critiques.

        Args:
            operation (str): Nom de l’opération (ex. : get_events).
            latency (float): Latence en secondes.
            success (bool): Succès de l’opération.
            error (str, optional): Message d’erreur si échec.
            **kwargs: Paramètres supplémentaires (ex. : num_events, confidence_drop_rate).
        """
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_usage = psutil.cpu_percent()
            if memory_usage > 1024:
                alert_msg = (
                    f"ALERTE: Utilisation mémoire élevée ({memory_usage:.2f} MB)"
                )
                miya_alerts(
                    alert_msg,
                    tag="ECONOMIC_CALENDAR",
                    voice_profile="urgent",
                    priority=5,
                )
                send_alert(alert_msg, priority=4)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "latency": latency,
                "success": success,
                "error": error,
                "memory_usage_mb": memory_usage,
                "cpu_usage_percent": cpu_usage,
                **kwargs,
            }
            self.log_buffer.append(log_entry)
            if len(self.log_buffer) >= self.buffer_size:
                log_df = pd.DataFrame(self.log_buffer)
                os.makedirs(os.path.dirname(CSV_LOG_PATH), exist_ok=True)

                def write_log():
                    log_df.to_csv(
                        CSV_LOG_PATH,
                        mode="a" if os.path.exists(CSV_LOG_PATH) else "w",
                        header=not os.path.exists(CSV_LOG_PATH),
                        index=False,
                        encoding="utf-8",
                    )

                self.with_retries(write_log)
                self.log_buffer = []
            logger.info(f"Performance journalisée pour {operation}. CPU: {cpu_usage}%")
        except Exception as e:
            error_msg = (
                f"Erreur journalisation performance: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, tag="ECONOMIC_CALENDAR", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def save_snapshot(
        self, snapshot_type: str, data: Dict, compress: bool = True
    ) -> None:
        """
        Sauvegarde un instantané des résultats.

        Args:
            snapshot_type (str): Type de snapshot (ex. : get_events).
            data (Dict): Données à sauvegarder.
            compress (bool): Compresser avec gzip (défaut : True).
        """
        try:
            start_time = datetime.now()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = {
                "timestamp": timestamp,
                "type": snapshot_type,
                "data": data,
                "buffer_size": len(self.log_buffer),
            }
            snapshot_path = os.path.join(
                SNAPSHOT_DIR, f"snapshot_{snapshot_type}_{timestamp}.json"
            )
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)

            def write_snapshot():
                if compress:
                    with gzip.open(f"{snapshot_path}.gz", "wt", encoding="utf-8") as f:
                        json.dump(snapshot, f, indent=4)
                else:
                    with open(snapshot_path, "w", encoding="utf-8") as f:
                        json.dump(snapshot, f, indent=4)

            self.with_retries(write_snapshot)
            save_path = f"{snapshot_path}.gz" if compress else snapshot_path
            file_size = os.path.getsize(save_path) / 1024 / 1024
            if file_size > 1.0:
                alert_msg = f"Snapshot size {file_size:.2f} MB exceeds 1 MB"
                miya_alerts(
                    alert_msg,
                    tag="ECONOMIC_CALENDAR",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Snapshot {snapshot_type} sauvegardé: {save_path}"
            miya_speak(
                success_msg, tag="ECONOMIC_CALENDAR", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "save_snapshot",
                latency,
                success=True,
                snapshot_type=snapshot_type,
                file_size_mb=file_size,
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}\n{traceback.format_exc()}"
            self.log_performance("save_snapshot", 0, success=False, error=str(e))
            miya_alerts(
                error_msg, tag="ECONOMIC_CALENDAR", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def checkpoint(self, events: pd.DataFrame) -> None:
        """
        Sauvegarde incrémentielle des événements toutes les 5 minutes avec versionnage (5 versions).

        Args:
            events (pd.DataFrame): Événements à sauvegarder.
        """
        try:
            start_time = datetime.now()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_data = {
                "timestamp": timestamp,
                "events": events.to_dict(orient="records"),
                "num_events": len(events),
            }
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR, f"economic_calendar_{timestamp}.json.gz"
            )
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)

            def write_checkpoint():
                with gzip.open(checkpoint_path, "wt", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, indent=4)

            self.with_retries(write_checkpoint)
            self.checkpoint_versions.append(checkpoint_path)
            if len(self.checkpoint_versions) > 5:
                oldest = self.checkpoint_versions.pop(0)
                if os.path.exists(oldest):
                    os.remove(oldest)
            file_size = os.path.getsize(checkpoint_path) / 1024 / 1024
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Checkpoint sauvegardé: {checkpoint_path}"
            miya_speak(
                success_msg, tag="ECONOMIC_CALENDAR", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "checkpoint",
                latency,
                success=True,
                file_size_mb=file_size,
                num_events=len(events),
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde checkpoint: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance("checkpoint", 0, success=False, error=str(e))
            miya_alerts(
                error_msg, tag="ECONOMIC_CALENDAR", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def cloud_backup(self, events: pd.DataFrame) -> None:
        """
        Sauvegarde distribuée des événements vers AWS S3 toutes les 15 minutes (configurable).

        Args:
            events (pd.DataFrame): Événements à sauvegarder.
        """
        try:
            start_time = datetime.now()
            if not self.config.get("s3_bucket"):
                warning_msg = "S3 bucket non configuré, sauvegarde cloud ignorée"
                miya_speak(
                    warning_msg,
                    tag="ECONOMIC_CALENDAR",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_data = {
                "timestamp": timestamp,
                "events": events.to_dict(orient="records"),
                "num_events": len(events),
            }
            backup_path = (
                f"{self.config['s3_prefix']}economic_calendar_{timestamp}.json.gz"
            )
            temp_path = os.path.join(CHECKPOINT_DIR, f"temp_s3_{timestamp}.json.gz")

            def write_temp():
                with gzip.open(temp_path, "wt", encoding="utf-8") as f:
                    json.dump(backup_data, f, indent=4)

            self.with_retries(write_temp)
            s3_client = boto3.client("s3")

            def upload_s3():
                s3_client.upload_file(temp_path, self.config["s3_bucket"], backup_path)

            self.with_retries(upload_s3)
            os.remove(temp_path)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Sauvegarde cloud S3 effectuée: {backup_path}"
            miya_speak(
                success_msg, tag="ECONOMIC_CALENDAR", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "cloud_backup", latency, success=True, num_events=len(events)
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde cloud S3: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance("cloud_backup", 0, success=False, error=str(e))
            miya_alerts(
                error_msg, tag="ECONOMIC_CALENDAR", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def validate_shap_features(self, features: List[str]) -> bool:
        """
        Valide que les features sont dans les top 150 SHAP (Phase 17).

        Args:
            features (List[str]): Liste des features à valider.

        Returns:
            bool: True si toutes les features sont valides, False sinon.
        """
        try:
            start_time = datetime.now()
            if not os.path.exists(FEATURE_IMPORTANCE_PATH):
                error_msg = "Fichier SHAP manquant"
                miya_alerts(
                    error_msg,
                    tag="ECONOMIC_CALENDAR",
                    voice_profile="urgent",
                    priority=4,
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return False
            shap_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
            if len(shap_df) < 150:
                error_msg = f"Nombre insuffisant de SHAP features: {len(shap_df)} < 150"
                miya_alerts(
                    error_msg,
                    tag="ECONOMIC_CALENDAR",
                    voice_profile="urgent",
                    priority=4,
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return False
            valid_features = set(shap_df["feature"].head(150))
            missing = [f for f in features if f not in valid_features]
            if missing:
                warning_msg = f"Features non incluses dans top 150 SHAP: {missing}"
                miya_alerts(
                    warning_msg,
                    tag="ECONOMIC_CALENDAR",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "SHAP features validées"
            miya_speak(
                success_msg, tag="ECONOMIC_CALENDAR", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "validate_shap_features",
                latency,
                success=True,
                num_features=len(features),
            )
            self.save_snapshot(
                "validate_shap_features",
                {"num_features": len(features), "missing": missing},
                compress=True,
            )
            return len(missing) == 0
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur validation SHAP features: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, tag="ECONOMIC_CALENDAR", voice_profile="urgent", priority=4
            )
            send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "validate_shap_features", latency, success=False, error=str(e)
            )
            return False

    def with_retries(
        self, func: callable, max_attempts: int = 3, delay_base: float = 2.0
    ) -> Any:
        """
        Exécute une fonction avec retries (max 3, délai exponentiel).

        Args:
            func (callable): Fonction à exécuter.
            max_attempts (int): Nombre maximum de tentatives (défaut : 3).
            delay_base (float): Base du délai exponentiel (défaut : 2.0).

        Returns:
            Any: Résultat de la fonction.

        Raises:
            Exception: Si toutes les tentatives échouent.
        """
        start_time = datetime.now()
        for attempt in range(max_attempts):
            try:
                result = func()
                latency = (datetime.now() - start_time).total_seconds()
                self.log_performance(
                    f"retry_attempt_{attempt+1}",
                    latency,
                    success=True,
                    attempt_number=attempt + 1,
                )
                return result
            except Exception as e:
                if attempt == max_attempts - 1:
                    error_msg = f"Échec après {max_attempts} tentatives: {str(e)}\n{traceback.format_exc()}"
                    self.log_performance(
                        f"retry_attempt_{attempt+1}",
                        0,
                        success=False,
                        error=str(e),
                        attempt_number=attempt + 1,
                    )
                    miya_alerts(
                        error_msg,
                        tag="ECONOMIC_CALENDAR",
                        voice_profile="urgent",
                        priority=4,
                    )
                    send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    logger.error(error_msg)
                    raise
                delay = delay_base * (2**attempt)
                warning_msg = f"Tentative {attempt+1} échouée, retry après {delay}s"
                miya_speak(
                    warning_msg,
                    tag="ECONOMIC_CALENDAR",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
                time.sleep(delay)

    def calculate_volatility_from_es(
        self, event_time: pd.Timestamp, window_minutes: int = 30
    ) -> float:
        """
        Calcule la volatilité à partir des prix ES autour de l’événement.

        Args:
            event_time (pd.Timestamp): Timestamp de l’événement.
            window_minutes (int): Fenêtre temporelle en minutes autour de l’événement.

        Returns:
            float: Volatilité calculée (variation relative des prix).
        """
        try:
            start_time = datetime.now()
            if not self.data_provider:
                error_msg = "Fournisseur de données IQFeed non disponible"
                miya_alerts(
                    error_msg,
                    tag="ECONOMIC_CALENDAR",
                    voice_profile="urgent",
                    priority=4,
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                self.log_performance(
                    "calculate_volatility_from_es", 0, success=False, error=error_msg
                )
                return 0.0

            es_data = self.data_provider.fetch_ohlc(
                symbol="ES",
                start_time=event_time - timedelta(days=1),
                end_time=event_time + timedelta(days=1),
            )
            if es_data.empty or "close" not in es_data.columns:
                error_msg = "Aucune donnée ES disponible pour calculer la volatilité"
                miya_alerts(
                    error_msg,
                    tag="ECONOMIC_CALENDAR",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                logger.warning(error_msg)
                self.log_performance(
                    "calculate_volatility_from_es", 0, success=False, error=error_msg
                )
                return 0.0

            es_data["timestamp"] = pd.to_datetime(es_data["timestamp"])
            start_time_window = event_time - timedelta(minutes=window_minutes)
            end_time_window = event_time + timedelta(minutes=window_minutes)
            window_data = es_data[
                (es_data["timestamp"] >= start_time_window)
                & (es_data["timestamp"] <= end_time_window)
            ]

            if window_data.empty:
                error_msg = f"Aucune donnée ES dans la fenêtre {start_time_window} à {end_time_window}"
                miya_speak(
                    error_msg,
                    tag="ECONOMIC_CALENDAR",
                    voice_profile="warning",
                    priority=2,
                )
                send_alert(error_msg, priority=2)
                send_telegram_alert(error_msg)
                logger.warning(error_msg)
                self.log_performance(
                    "calculate_volatility_from_es", 0, success=False, error=error_msg
                )
                return 0.0

            price_max = window_data["close"].max()
            price_min = window_data["close"].min()
            price_mean = window_data["close"].mean()
            if price_mean == 0 or pd.isna(price_mean):
                error_msg = "Prix moyen invalide dans la fenêtre"
                miya_speak(
                    error_msg,
                    tag="ECONOMIC_CALENDAR",
                    voice_profile="warning",
                    priority=2,
                )
                send_alert(error_msg, priority=2)
                send_telegram_alert(error_msg)
                logger.warning(error_msg)
                self.log_performance(
                    "calculate_volatility_from_es", 0, success=False, error=error_msg
                )
                return 0.0

            volatility = (price_max - price_min) / price_mean
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = (
                f"Volatilité calculée: {volatility:.4f} pour événement à {event_time}"
            )
            miya_speak(
                success_msg, tag="ECONOMIC_CALENDAR", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "calculate_volatility_from_es",
                latency,
                success=True,
                volatility=volatility,
            )
            return volatility
        except Exception as e:
            error_msg = f"Erreur calcul volatilité ES pour événement à {event_time}: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                "calculate_volatility_from_es", 0, success=False, error=str(e)
            )
            miya_alerts(
                error_msg, tag="ECONOMIC_CALENDAR", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            return 0.0

    def store_event_patterns(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Applique PCA et K-means pour clusteriser les événements et les stocker dans market_memory.db.

        Args:
            events (pd.DataFrame): Événements macroéconomiques.

        Returns:
            pd.DataFrame: Événements avec colonne cluster_id.
        """
        try:
            start_time = datetime.now()
            if events.empty:
                warning_msg = "Aucun événement à clusteriser"
                miya_speak(
                    warning_msg,
                    tag="ECONOMIC_CALENDAR",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
                return events

            # Préparer les features pour PCA
            features = events[["impact", "volatility"]].copy()
            features["event_type_encoded"] = pd.Categorical(events["event_type"]).codes
            features.fillna(0, inplace=True)

            # Appliquer PCA (10 dimensions, 95% variance)
            pca = PCA(n_components=10, random_state=42)
            pca_features = pca.fit_transform(features)
            explained_variance = sum(pca.explained_variance_ratio_)
            if explained_variance < 0.95:
                warning_msg = f"Variance expliquée PCA insuffisante: {explained_variance:.2f} < 0.95"
                miya_speak(
                    warning_msg,
                    tag="ECONOMIC_CALENDAR",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)

            # Appliquer K-means (10 clusters)
            kmeans = KMeans(n_clusters=10, random_state=42)
            cluster_ids = kmeans.fit_predict(pca_features)
            events["cluster_id"] = cluster_ids

            # Stocker dans market_memory.db
            conn = sqlite3.connect(DB_PATH)
            events[
                [
                    "event_id",
                    "start_time",
                    "event_type",
                    "impact",
                    "description",
                    "volatility",
                    "cluster_id",
                ]
            ].to_sql("clusters", conn, if_exists="append", index=False)
            conn.close()

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Patterns d’événements clusterisés et stockés: {len(events)} événements"
            miya_speak(
                success_msg, tag="ECONOMIC_CALENDAR", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "store_event_patterns",
                latency,
                success=True,
                num_events=len(events),
                explained_variance=explained_variance,
            )
            self.save_snapshot(
                "store_event_patterns",
                {"num_events": len(events), "explained_variance": explained_variance},
                compress=True,
            )
            return events
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur clusterisation événements: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance("store_event_patterns", 0, success=False, error=str(e))
            miya_alerts(
                error_msg, tag="ECONOMIC_CALENDAR", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            return events

    def update_calendar(self) -> Optional[pd.DataFrame]:
        """
        Met à jour le calendrier économique via l’API Investing.com.

        Returns:
            Optional[pd.DataFrame]: Événements mis à jour ou None si échec.
        """
        try:
            start_time = datetime.now()
            cache_hours = self.config.get("cache_hours", 24)
            if os.path.exists(EVENTS_CSV_PATH):
                last_update = pd.to_datetime(
                    pd.read_csv(EVENTS_CSV_PATH)["start_time"].max()
                )
                if (
                    last_update is not None
                    and (datetime.now() - last_update).total_seconds() / 3600
                    < cache_hours
                ):
                    events = pd.read_csv(EVENTS_CSV_PATH)
                    success_msg = (
                        f"Données récentes ({last_update}), mise à jour ignorée"
                    )
                    miya_speak(
                        success_msg,
                        tag="ECONOMIC_CALENDAR",
                        voice_profile="calm",
                        priority=2,
                    )
                    send_alert(success_msg, priority=2)
                    send_telegram_alert(success_msg)
                    logger.info(success_msg)
                    self.log_performance("update_calendar_cache_hit", 0, success=True)
                    return events

            events = self.get_events()
            if events is not None and not events.empty:
                latency = (datetime.now() - start_time).total_seconds()
                success_msg = f"Calendrier mis à jour: {len(events)} événements"
                miya_speak(
                    success_msg,
                    tag="ECONOMIC_CALENDAR",
                    voice_profile="calm",
                    priority=2,
                )
                send_alert(success_msg, priority=2)
                send_telegram_alert(success_msg)
                logger.info(success_msg)
                self.log_performance(
                    "update_calendar", latency, success=True, num_events=len(events)
                )
                self.save_snapshot(
                    "update_calendar", {"num_events": len(events)}, compress=True
                )
                return events
            return None
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur mise à jour calendrier: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, tag="ECONOMIC_CALENDAR", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "update_calendar", latency, success=False, error=str(e)
            )
            return None

    def get_events(self) -> Optional[pd.DataFrame]:
        """
        Récupère les événements macro via l’API Investing.com, calcule la volatilité, et les stocke.

        Returns:
            Optional[pd.DataFrame]: Événements macroéconomiques ou None si échec.
        """
        try:
            start_time = datetime.now()
            required_features = ["event_impact", "event_type", "event_volatility"]
            if not self.validate_shap_features(required_features):
                warning_msg = (
                    "Certaines features d’événements ne sont pas dans les top 150 SHAP"
                )
                miya_speak(
                    warning_msg,
                    tag="ECONOMIC_CALENDAR",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)

            def fetch_events():
                headers = {"Authorization": f"Bearer {self.api_key}"}
                response = requests.get(
                    self.config["api_url"], headers=headers, timeout=10
                )
                response.raise_for_status()
                data = response.json()
                events = pd.DataFrame(
                    data,
                    columns=[
                        "event_id",
                        "start_time",
                        "event_type",
                        "impact",
                        "description",
                    ],
                )
                events["start_time"] = pd.to_datetime(events["start_time"])
                events["impact"] = (
                    events["impact"]
                    .replace({"High": 3, "Medium": 2, "Low": 1})
                    .fillna(1)
                    .astype(int)
                )
                return events

            events = self.with_retries(fetch_events)
            if events is None or events.empty:
                error_msg = "Aucun événement récupéré de l’API Investing.com"
                miya_alerts(
                    error_msg,
                    tag="ECONOMIC_CALENDAR",
                    voice_profile="urgent",
                    priority=4,
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return None

            # Calculer confidence_drop_rate
            required_cols = [
                "event_id",
                "start_time",
                "event_type",
                "impact",
                "description",
            ]
            missing_cols = [col for col in required_cols if col not in events.columns]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(events.columns)} colonnes)"
                miya_alerts(
                    alert_msg,
                    tag="ECONOMIC_CALENDAR",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            # Calculer volatilité
            events["volatility"] = 0.0
            for idx, row in events.iterrows():
                volatility = self.calculate_volatility_from_es(
                    row["start_time"], self.config["volatility_window_minutes"]
                )
                events.at[idx, "volatility"] = volatility
                if volatility > 0.01:
                    alert_msg = f"Volatilité élevée détectée: {volatility:.4f} pour {row['event_type']} ({row['start_time']})"
                    miya_speak(
                        alert_msg,
                        tag="ECONOMIC_CALENDAR",
                        voice_profile="urgent",
                        priority=3,
                    )
                    send_alert(alert_msg, priority=3)
                    send_telegram_alert(alert_msg)
                    logger.info(alert_msg)

            # Clusteriser et stocker dans market_memory.db
            events = self.store_event_patterns(events)

            # Sauvegarder dans macro_events.csv
            os.makedirs(os.path.dirname(EVENTS_CSV_PATH), exist_ok=True)

            def save_events():
                events.to_csv(
                    EVENTS_CSV_PATH,
                    mode="a" if os.path.exists(EVENTS_CSV_PATH) else "w",
                    index=False,
                    encoding="utf-8",
                )

            self.with_retries(save_events)

            # Sauvegarder volatilité dans event_volatility_history.csv
            os.makedirs(os.path.dirname(VOLATILITY_CSV_PATH), exist_ok=True)

            def save_volatility():
                volatility_data = events[
                    [
                        "event_id",
                        "start_time",
                        "event_type",
                        "impact",
                        "volatility",
                        "cluster_id",
                    ]
                ]
                volatility_data.to_csv(
                    VOLATILITY_CSV_PATH,
                    mode="a" if os.path.exists(VOLATILITY_CSV_PATH) else "w",
                    index=False,
                    encoding="utf-8",
                )

            self.with_retries(save_volatility)

            # Sauvegardes
            self.checkpoint(events)
            self.cloud_backup(events)

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Événements macro récupérés et stockés: {len(events)} événements, Confidence drop rate: {confidence_drop_rate:.2f}"
            miya_speak(
                success_msg, tag="ECONOMIC_CALENDAR", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "get_events",
                latency,
                success=True,
                num_events=len(events),
                confidence_drop_rate=confidence_drop_rate,
            )
            self.save_snapshot(
                "get_events",
                {
                    "num_events": len(events),
                    "confidence_drop_rate": confidence_drop_rate,
                },
                compress=True,
            )
            return events
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur récupération événements: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance("get_events", 0, success=False, error=str(e))
            miya_alerts(
                error_msg, tag="ECONOMIC_CALENDAR", voice_profile="urgent", priority=4
            )
            send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.save_snapshot("error_get_events", {"error": str(e)}, compress=True)
            return None

    def schedule_calendar_updates(
        self, frequency: str = None, timeout_hours: int = 24
    ) -> None:
        """
        Planifie la mise à jour automatique du calendrier économique.

        Args:
            frequency (str): Fréquence de mise à jour ('daily', 'hourly', '15min').
            timeout_hours (int): Durée maximale d’exécution en heures.
        """
        try:
            start_time = datetime.now()
            frequency = frequency or self.config.get("frequency", "daily")
            scheduled_time = self.config.get("scheduled_time", "00:00")

            def job():
                self.update_calendar()

            valid_frequencies = {
                "daily": lambda: schedule.every().day.at(scheduled_time).do(job),
                "hourly": lambda: schedule.every().hour.do(job),
                "15min": lambda: schedule.every(15).minutes.do(job),
            }
            if frequency not in valid_frequencies:
                error_msg = f"Fréquence non reconnue: {frequency}, utilisation 'daily'"
                miya_alerts(
                    error_msg,
                    tag="ECONOMIC_CALENDAR",
                    voice_profile="urgent",
                    priority=4,
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                frequency = "daily"

            valid_frequencies[frequency]()
            success_msg = f"Mise à jour {frequency} planifiée à {scheduled_time if frequency == 'daily' else 'chaque heure/minute'}"
            miya_speak(
                success_msg, tag="ECONOMIC_CALENDAR", voice_profile="calm", priority=2
            )
            send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            logger.info(success_msg)

            timeout = datetime.now() + timedelta(hours=timeout_hours)
            while RUNNING and datetime.now() < timeout:
                schedule.run_pending()
                time.sleep(10)

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Planificateur terminé ou timeout atteint"
            miya_speak(
                success_msg, tag="ECONOMIC_CALENDAR", voice_profile="calm", priority=2
            )
            send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("schedule_calendar_updates", latency, success=True)
        except KeyboardInterrupt:
            self.signal_handler(signal.SIGINT, None)
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur planification: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="ECONOMIC_CALENDAR", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "schedule_calendar_updates", latency, success=False, error=str(e)
            )

    def signal_handler(self, sig, frame) -> None:
        """
        Gère l'arrêt propre du service (Ctrl+C).

        Args:
            sig: Signal reçu.
            frame: Frame actuel.
        """
        global RUNNING
        start_time = datetime.now()
        try:
            RUNNING = False
            success_msg = "Arrêt du collecteur d’événements en cours..."
            miya_speak(
                success_msg, tag="ECONOMIC_CALENDAR", voice_profile="calm", priority=2
            )
            send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            logger.info("Arrêt du collecteur initié")

            events = self.update_calendar()
            if events is not None and not events.empty:
                self.checkpoint(events)
                self.save_snapshot(
                    "shutdown",
                    {
                        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                        "num_events": len(events),
                    },
                    compress=True,
                )

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Collecteur d’événements arrêté proprement"
            miya_speak(
                success_msg, tag="ECONOMIC_CALENDAR", voice_profile="calm", priority=2
            )
            send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("signal_handler", latency, success=True)
            exit(0)
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur arrêt collecteur: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="ECONOMIC_CALENDAR", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("signal_handler", latency, success=False, error=str(e))
            exit(1)


if __name__ == "__main__":
    calendar = EconomicCalendar()
    calendar.schedule_calendar_updates()
