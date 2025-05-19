# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/api/context_aware_filter.py
# Gère les métriques contextuelles liées aux nouvelles, événements macro, et cyclicité temporelle.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Calcule 19 métriques contextuelles (16 nouvelles + 3 existantes comme event_volatility_impact),
#        utilise IQFeed pour les nouvelles et futures, intègre SHAP (méthode 17), confidence_drop_rate (méthode 8),
#        et enregistre logs psutil, snapshots, et sauvegardes.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.23.0,<2.0.0, psutil>=5.9.8,<6.0.0, matplotlib>=3.7.0,<4.0.0,
#   pyyaml>=6.0.0,<7.0.0, boto3>=1.26.0,<2.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/es_config.yaml
# - data/iqfeed/news_data.csv
# - data/iqfeed/futures_data.csv
# - data/events/macro_events.csv
# - data/events/event_volatility_history.csv
# - data/features/feature_importance.csv
#
# Outputs :
# - data/features/cache/context_aware/
# - data/logs/context_aware_performance.csv
# - data/logs/context_aware.log
# - data/context_aware_snapshots/*.json.gz
# - data/checkpoints/context_aware_*.json.gz
# - data/figures/context_aware/
#
# Notes :
# - Conforme à structure.txt (version 2.1.3, 2025-05-13).
# - Utilise exclusivement IQFeed (dxFeed supprimé).
# - Implémente retries (max 3, délai 2^attempt), logs psutil, alertes Telegram, snapshots compressés.
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des données.
# - Intègre validation SHAP (Phase 17) pour les top 150 features en production.
# - Phases intégrées : Phase 1 (collecte IQFeed), Phase 8 (auto-conscience), Phase 17 (interprétabilité SHAP).
# - Sauvegardes incrémentielles toutes les 5 min dans data/checkpoints/ avec versionnage (5 versions).
# - Sauvegardes distribuées toutes les 15 min vers S3 (configurable).
# - Tests unitaires dans tests/test_context_aware_filter.py.
# - Validation complète prévue pour juin 2025.
# - Évolution future : Intégration avec feature_pipeline.py pour top 150 SHAP, migration API Investing.com (juin 2025).

import gzip
import hashlib
import json
import logging
import os
import signal
import time
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

from src.model.utils.alert_manager import send_alert
from src.model.utils.config_manager import get_config
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.utils.telegram_alert import send_telegram_alert

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "es_config.yaml")
CACHE_DIR = os.path.join(BASE_DIR, "data", "features", "cache", "context_aware")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "context_aware_snapshots")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "data", "checkpoints")
CSV_LOG_PATH = os.path.join(BASE_DIR, "data", "logs", "context_aware_performance.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures", "context_aware")
FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR, "data", "features", "feature_importance.csv"
)

# Création des dossiers
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data", "logs"), exist_ok=True)

# Configuration du logging
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "data", "logs", "context_aware.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Variable pour gérer l'arrêt propre
RUNNING = True


class ContextAwareFilter:

    def __init__(self, config_path: str = CONFIG_PATH):
        """
        Initialise le filtre contextuel.

        Args:
            config_path (str): Chemin vers le fichier de configuration.
        """
        self.log_buffer = []
        self.cache = {}
        self.checkpoint_versions = []
        try:
            self.config = self.load_config_with_validation(config_path)
            self.buffer_size = self.config.get("buffer_size", 100)
            self.max_cache_size = self.config.get("max_cache_size", 1000)
            signal.signal(signal.SIGINT, self.handle_sigint)
            success_msg = "ContextAwareFilter initialisé"
            miya_speak(
                success_msg, tag="CONTEXT_AWARE", voice_profile="calm", priority=2
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config_path": config_path})
        except Exception as e:
            error_msg = f"Erreur initialisation ContextAwareFilter: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            self.config = {
                "buffer_size": 100,
                "max_cache_size": 1000,
                "cache_hours": 24,
                "s3_bucket": None,
                "s3_prefix": "context_aware/",
            }

    def load_config_with_validation(self, config_path: str) -> Dict[str, Any]:
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
            if "context_aware_filter" not in config:
                config["context_aware_filter"] = {}
            required_keys = [
                "buffer_size",
                "max_cache_size",
                "cache_hours",
                "s3_bucket",
                "s3_prefix",
            ]
            missing_keys = [
                key
                for key in required_keys
                if key not in config["context_aware_filter"]
            ]
            if missing_keys:
                raise ValueError(
                    f"Clés de configuration manquantes dans context_aware_filter: {missing_keys}"
                )
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Configuration {config_path} chargée"
            miya_speak(
                success_msg, tag="CONTEXT_AWARE", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("load_config_with_validation", latency, success=True)
            return config["context_aware_filter"]
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur chargement configuration {config_path}: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "load_config_with_validation", latency, success=False, error=str(e)
            )
            return {
                "buffer_size": 100,
                "max_cache_size": 1000,
                "cache_hours": 24,
                "s3_bucket": None,
                "s3_prefix": "context_aware/",
            }

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ) -> None:
        """
        Journalise les performances des opérations critiques.

        Args:
            operation (str): Nom de l’opération (ex. : compute_contextual_metrics).
            latency (float): Latence en secondes.
            success (bool): Succès de l’opération.
            error (str, optional): Message d’erreur si échec.
            **kwargs: Paramètres supplémentaires (ex. : num_rows, num_metrics, confidence_drop_rate).
        """
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_usage = psutil.cpu_percent()
            if memory_usage > 1024:
                alert_msg = (
                    f"ALERTE: Utilisation mémoire élevée ({memory_usage:.2f} MB)"
                )
                miya_alerts(
                    alert_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=5
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
                error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=3
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
            snapshot_type (str): Type de snapshot (ex. : compute_contextual_metrics).
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
                    alert_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=3
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Snapshot {snapshot_type} sauvegardé: {save_path}"
            miya_speak(
                success_msg, tag="CONTEXT_AWARE", voice_profile="calm", priority=1
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
                error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def checkpoint(self, metrics: pd.DataFrame) -> None:
        """
        Sauvegarde incrémentielle des métriques toutes les 5 minutes avec versionnage (5 versions).

        Args:
            metrics (pd.DataFrame): Métriques à sauvegarder.
        """
        try:
            start_time = datetime.now()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_data = {
                "timestamp": timestamp,
                "num_rows": len(metrics),
                "columns": list(metrics.columns),
            }
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR, f"context_aware_{timestamp}.json.gz"
            )
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)

            def write_checkpoint():
                with gzip.open(checkpoint_path, "wt", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, indent=4)
                metrics.to_csv(
                    checkpoint_path.replace(".json.gz", ".csv"),
                    index=False,
                    encoding="utf-8",
                )

            self.with_retries(write_checkpoint)
            self.checkpoint_versions.append(checkpoint_path)
            if len(self.checkpoint_versions) > 5:
                oldest = self.checkpoint_versions.pop(0)
                if os.path.exists(oldest):
                    os.remove(oldest)
                csv_oldest = oldest.replace(".json.gz", ".csv")
                if os.path.exists(csv_oldest):
                    os.remove(csv_oldest)
            file_size = os.path.getsize(checkpoint_path) / 1024 / 1024
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Checkpoint sauvegardé: {checkpoint_path}"
            miya_speak(
                success_msg, tag="CONTEXT_AWARE", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "checkpoint",
                latency,
                success=True,
                file_size_mb=file_size,
                num_rows=len(metrics),
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde checkpoint: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance("checkpoint", 0, success=False, error=str(e))
            miya_alerts(
                error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def cloud_backup(self, metrics: pd.DataFrame) -> None:
        """
        Sauvegarde distribuée des métriques vers AWS S3 toutes les 15 minutes (configurable).

        Args:
            metrics (pd.DataFrame): Métriques à sauvegarder.
        """
        try:
            start_time = datetime.now()
            if not self.config.get("s3_bucket"):
                warning_msg = "S3 bucket non configuré, sauvegarde cloud ignorée"
                miya_speak(
                    warning_msg,
                    tag="CONTEXT_AWARE",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.config['s3_prefix']}context_aware_{timestamp}.csv.gz"
            temp_path = os.path.join(CHECKPOINT_DIR, f"temp_s3_{timestamp}.csv.gz")

            def write_temp():
                metrics.to_csv(
                    temp_path, compression="gzip", index=False, encoding="utf-8"
                )

            self.with_retries(write_temp)
            s3_client = boto3.client("s3")

            def upload_s3():
                s3_client.upload_file(temp_path, self.config["s3_bucket"], backup_path)

            self.with_retries(upload_s3)
            os.remove(temp_path)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Sauvegarde cloud S3 effectuée: {backup_path}"
            miya_speak(
                success_msg, tag="CONTEXT_AWARE", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "cloud_backup", latency, success=True, num_rows=len(metrics)
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde cloud S3: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance("cloud_backup", 0, success=False, error=str(e))
            miya_alerts(
                error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

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
                        tag="CONTEXT_AWARE",
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
                    tag="CONTEXT_AWARE",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
                time.sleep(delay)

    def handle_sigint(self, signal: int, frame: Any) -> None:
        """
        Gère l'arrêt propre du service (Ctrl+C).

        Args:
            signal: Signal reçu.
            frame: Frame actuel.
        """
        global RUNNING
        datetime.now()
        try:
            RUNNING = False
            snapshot_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "sigint",
                "log_buffer": self.log_buffer,
                "cache_size": len(self.cache),
            }
            self.save_snapshot("sigint", snapshot_data)
            success_msg = "SIGINT reçu, snapshot sauvegardé"
            miya_speak(
                success_msg, tag="CONTEXT_AWARE", voice_profile="calm", priority=2
            )
            send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("handle_sigint", 0, success=True)
            exit(0)
        except Exception as e:
            error_msg = f"Erreur gestion SIGINT: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("handle_sigint", 0, success=False, error=str(e))
            exit(1)

    def parse_news_data(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyse les données de nouvelles pour extraire les métriques nécessaires.

        Args:
            news_data (pd.DataFrame): Données de nouvelles avec sentiment_score et volume.

        Returns:
            pd.DataFrame: Données analysées.
        """
        try:
            start_time = datetime.now()
            required_cols = ["sentiment_score", "timestamp"]
            missing_cols = [
                col for col in required_cols if col not in news_data.columns
            ]
            if missing_cols:
                error_msg = f"Colonnes manquantes dans news_data: {missing_cols}"
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return pd.DataFrame()
            news_data = news_data.copy()
            news_data["timestamp"] = pd.to_datetime(
                news_data["timestamp"], errors="coerce"
            )
            news_data["sentiment_score"] = pd.to_numeric(
                news_data["sentiment_score"], errors="coerce"
            ).fillna(0.0)
            news_data["volume"] = news_data.get(
                "volume", pd.Series(1, index=news_data.index)
            )
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Données de nouvelles analysées: {len(news_data)} lignes"
            miya_speak(
                success_msg, tag="CONTEXT_AWARE", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "parse_news_data", latency, success=True, num_rows=len(news_data)
            )
            return news_data
        except Exception as e:
            error_msg = (
                f"Erreur dans parse_news_data: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance("parse_news_data", 0, success=False, error=str(e))
            miya_alerts(
                error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            return pd.DataFrame()

    def calculate_event_volatility_impact(
        self,
        events: pd.DataFrame,
        volatility_history: pd.DataFrame,
        timestamp: pd.Timestamp,
    ) -> float:
        """
        Calcule l'impact de volatilité des événements macro.

        Args:
            events (pd.DataFrame): Données des événements macro.
            volatility_history (pd.DataFrame): Historique de volatilité.
            timestamp (pd.Timestamp): Timestamp de référence.

        Returns:
            float: Impact de volatilité.
        """
        try:
            start_time = datetime.now()
            events["time_diff"] = np.abs(
                (events["timestamp"] - timestamp).dt.total_seconds()
            )
            nearest_event = events.loc[events["time_diff"].idxmin()]
            event_type = nearest_event["event_type"]
            impact_score = nearest_event["event_impact_score"]
            historical_impact = volatility_history[
                volatility_history["event_type"] == event_type
            ]["volatility_impact"].mean()
            if pd.isna(historical_impact):
                historical_impact = 0.0
            volatility_impact = historical_impact * impact_score
            result = volatility_impact if not np.isnan(volatility_impact) else 0.0
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Impact de volatilité calculé: {result:.4f}"
            miya_speak(
                success_msg, tag="CONTEXT_AWARE", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "calculate_event_volatility_impact",
                latency,
                success=True,
                volatility_impact=result,
            )
            return result
        except Exception as e:
            error_msg = f"Erreur dans calculate_event_volatility_impact: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                "calculate_event_volatility_impact", 0, success=False, error=str(e)
            )
            miya_alerts(
                error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            return 0.0

    def calculate_event_timing_proximity(
        self, events: pd.DataFrame, timestamp: pd.Timestamp
    ) -> float:
        """
        Calcule la proximité temporelle de l'événement macro le plus proche.

        Args:
            events (pd.DataFrame): Données des événements macro.
            timestamp (pd.Timestamp): Timestamp de référence.

        Returns:
            float: Proximité temporelle en minutes.
        """
        try:
            start_time = datetime.now()
            events["time_diff"] = (
                events["timestamp"] - timestamp
            ).dt.total_seconds() / 60.0
            closest_event_diff = events["time_diff"].iloc[
                events["time_diff"].abs().argmin()
            ]
            result = closest_event_diff if not np.isnan(closest_event_diff) else 0.0
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Proximité temporelle calculée: {result:.2f} minutes"
            miya_speak(
                success_msg, tag="CONTEXT_AWARE", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "calculate_event_timing_proximity",
                latency,
                success=True,
                proximity=result,
            )
            return result
        except Exception as e:
            error_msg = f"Erreur dans calculate_event_timing_proximity: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                "calculate_event_timing_proximity", 0, success=False, error=str(e)
            )
            miya_alerts(
                error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            return 0.0

    def calculate_event_frequency_24h(
        self, events: pd.DataFrame, timestamp: pd.Timestamp
    ) -> int:
        """
        Calcule la fréquence des événements macro dans les dernières 24 heures.

        Args:
            events (pd.DataFrame): Données des événements macro.
            timestamp (pd.Timestamp): Timestamp de référence.

        Returns:
            int: Nombre d'événements dans les dernières 24 heures.
        """
        try:
            start_time = datetime.now()
            time_window = timestamp - timedelta(hours=24)
            recent_events = events[
                (events["timestamp"] >= time_window)
                & (events["timestamp"] <= timestamp)
            ]
            result = len(recent_events)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Fréquence des événements calculée: {result} événements"
            miya_speak(
                success_msg, tag="CONTEXT_AWARE", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "calculate_event_frequency_24h", latency, success=True, frequency=result
            )
            return result
        except Exception as e:
            error_msg = f"Erreur dans calculate_event_frequency_24h: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                "calculate_event_frequency_24h", 0, success=False, error=str(e)
            )
            miya_alerts(
                error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            return 0

    def compute_news_sentiment_momentum(
        self, news_data: pd.DataFrame, window: str = "1h"
    ) -> pd.Series:
        """
        Calcule la dynamique du sentiment des nouvelles.

        Args:
            news_data (pd.DataFrame): Données de nouvelles.
            window (str): Fenêtre temporelle (par défaut : '1h').

        Returns:
            pd.Series: Dynamique du sentiment.
        """
        try:
            start_time = datetime.now()
            news_data = self.parse_news_data(news_data)
            if news_data.empty:
                return pd.Series(0.0, index=news_data.index)
            momentum = (
                news_data["sentiment_score"]
                .groupby(pd.Grouper(key="timestamp", freq=window))
                .mean()
                .pct_change()
                .fillna(0.0)
            )
            result = momentum.reindex(news_data.index, method="ffill").fillna(0.0)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Dynamique du sentiment calculée: {len(result)} valeurs"
            miya_speak(
                success_msg, tag="CONTEXT_AWARE", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "compute_news_sentiment_momentum",
                latency,
                success=True,
                num_values=len(result),
            )
            return result
        except Exception as e:
            error_msg = f"Erreur dans compute_news_sentiment_momentum: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                "compute_news_sentiment_momentum", 0, success=False, error=str(e)
            )
            miya_alerts(
                error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            return pd.Series(0.0, index=news_data.index)

    def compute_news_event_proximity(
        self, news_data: pd.DataFrame, window: str = "1h"
    ) -> pd.Series:
        """
        Calcule la proximité des événements de nouvelles.

        Args:
            news_data (pd.DataFrame): Données de nouvelles.
            window (str): Fenêtre temporelle (par défaut : '1h').

        Returns:
            pd.Series: Proximité des événements.
        """
        try:
            start_time = datetime.now()
            news_data = self.parse_news_data(news_data)
            if news_data.empty:
                return pd.Series(0.0, index=news_data.index)
            proximity = (
                news_data["timestamp"]
                .groupby(pd.Grouper(key="timestamp", freq=window))
                .count()
            )
            result = proximity.reindex(news_data.index, method="ffill").fillna(0.0)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Proximité des événements calculée: {len(result)} valeurs"
            miya_speak(
                success_msg, tag="CONTEXT_AWARE", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "compute_news_event_proximity",
                latency,
                success=True,
                num_values=len(result),
            )
            return result
        except Exception as e:
            error_msg = f"Erreur dans compute_news_event_proximity: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                "compute_news_event_proximity", 0, success=False, error=str(e)
            )
            miya_alerts(
                error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            return pd.Series(0.0, index=news_data.index)

    def compute_macro_event_severity(self, calendar_data: pd.DataFrame) -> pd.Series:
        """
        Calcule la sévérité des événements macro.

        Args:
            calendar_data (pd.DataFrame): Données du calendrier économique.

        Returns:
            pd.Series: Sévérité des événements.
        """
        try:
            start_time = datetime.now()
            required_cols = ["severity", "timestamp"]
            missing_cols = [
                col for col in required_cols if col not in calendar_data.columns
            ]
            if missing_cols:
                error_msg = f"Colonnes manquantes dans calendar_data: {missing_cols}"
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return pd.Series(0.0, index=calendar_data.index)
            calendar_data = calendar_data.copy()
            calendar_data["timestamp"] = pd.to_datetime(
                calendar_data["timestamp"], errors="coerce"
            )
            result = calendar_data["severity"].fillna(0.0)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Sévérité des événements calculée: {len(result)} valeurs"
            miya_speak(
                success_msg, tag="CONTEXT_AWARE", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "compute_macro_event_severity",
                latency,
                success=True,
                num_values=len(result),
            )
            return result
        except Exception as e:
            error_msg = f"Erreur dans compute_macro_event_severity: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                "compute_macro_event_severity", 0, success=False, error=str(e)
            )
            miya_alerts(
                error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            return pd.Series(0.0, index=calendar_data.index)

    def compute_time_to_expiry_proximity(
        self, data: pd.DataFrame, expiry_dates: pd.Series
    ) -> pd.Series:
        """
        Calcule la proximité temporelle aux dates d'expiration.

        Args:
            data (pd.DataFrame): Données principales.
            expiry_dates (pd.Series): Dates d'expiration.

        Returns:
            pd.Series: Proximité aux dates d'expiration en jours.
        """
        try:
            start_time = datetime.now()
            if "timestamp" not in data.columns:
                error_msg = "Colonne timestamp manquante dans data"
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return pd.Series(0.0, index=data.index)
            data = data.copy()
            data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
            expiry_dates = pd.to_datetime(expiry_dates, errors="coerce")
            proximity = (expiry_dates - data["timestamp"]).dt.total_seconds() / (
                24 * 3600
            )
            result = proximity.fillna(0.0)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Proximité aux expirations calculée: {len(result)} valeurs"
            miya_speak(
                success_msg, tag="CONTEXT_AWARE", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "compute_time_to_expiry_proximity",
                latency,
                success=True,
                num_values=len(result),
            )
            return result
        except Exception as e:
            error_msg = f"Erreur dans compute_time_to_expiry_proximity: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                "compute_time_to_expiry_proximity", 0, success=False, error=str(e)
            )
            miya_alerts(
                error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            return pd.Series(0.0, index=data.index)

    def compute_economic_calendar_weight(
        self, calendar_data: pd.DataFrame
    ) -> pd.Series:
        """
        Calcule le poids des événements du calendrier économique.

        Args:
            calendar_data (pd.DataFrame): Données du calendrier économique.

        Returns:
            pd.Series: Poids des événements.
        """
        try:
            start_time = datetime.now()
            required_cols = ["weight", "timestamp"]
            missing_cols = [
                col for col in required_cols if col not in calendar_data.columns
            ]
            if missing_cols:
                error_msg = f"Colonnes manquantes dans calendar_data: {missing_cols}"
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return pd.Series(0.0, index=calendar_data.index)
            calendar_data = calendar_data.copy()
            calendar_data["timestamp"] = pd.to_datetime(
                calendar_data["timestamp"], errors="coerce"
            )
            result = calendar_data["weight"].fillna(0.0)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Poids du calendrier calculé: {len(result)} valeurs"
            miya_speak(
                success_msg, tag="CONTEXT_AWARE", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "compute_economic_calendar_weight",
                latency,
                success=True,
                num_values=len(result),
            )
            return result
        except Exception as e:
            error_msg = f"Erreur dans compute_economic_calendar_weight: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                "compute_economic_calendar_weight", 0, success=False, error=str(e)
            )
            miya_alerts(
                error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            return pd.Series(0.0, index=calendar_data.index)

    def compute_news_volume_spike(
        self, news_data: pd.DataFrame, window: str = "5min"
    ) -> pd.Series:
        """
        Calcule les pics de volume des nouvelles.

        Args:
            news_data (pd.DataFrame): Données de nouvelles.
            window (str): Fenêtre temporelle (par défaut : '5min').

        Returns:
            pd.Series: Pics de volume.
        """
        try:
            start_time = datetime.now()
            news_data = self.parse_news_data(news_data)
            if news_data.empty:
                return pd.Series(0.0, index=news_data.index)
            volume = (
                news_data["volume"]
                .groupby(pd.Grouper(key="timestamp", freq=window))
                .sum()
            )
            mean_volume = volume.rolling(window="1h", min_periods=1).mean()
            spike = (volume - mean_volume) / mean_volume.replace(0, 1e-6)
            result = spike.reindex(news_data.index, method="ffill").fillna(0.0)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Pics de volume calculés: {len(result)} valeurs"
            miya_speak(
                success_msg, tag="CONTEXT_AWARE", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "compute_news_volume_spike",
                latency,
                success=True,
                num_values=len(result),
            )
            return result
        except Exception as e:
            error_msg = f"Erreur dans compute_news_volume_spike: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                "compute_news_volume_spike", 0, success=False, error=str(e)
            )
            miya_alerts(
                error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            return pd.Series(0.0, index=news_data.index)

    def compute_news_sentiment_acceleration(
        self, news_data: pd.DataFrame, window: str = "1h"
    ) -> pd.Series:
        """
        Calcule l'accélération du sentiment des nouvelles.

        Args:
            news_data (pd.DataFrame): Données de nouvelles.
            window (str): Fenêtre temporelle (par défaut : '1h').

        Returns:
            pd.Series: Accélération du sentiment.
        """
        try:
            start_time = datetime.now()
            news_data = self.parse_news_data(news_data)
            if news_data.empty:
                return pd.Series(0.0, index=news_data.index)
            momentum = (
                news_data["sentiment_score"]
                .groupby(pd.Grouper(key="timestamp", freq=window))
                .mean()
                .pct_change()
            )
            acceleration = momentum.diff().fillna(0.0)
            result = acceleration.reindex(news_data.index, method="ffill").fillna(0.0)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Accélération du sentiment calculée: {len(result)} valeurs"
            miya_speak(
                success_msg, tag="CONTEXT_AWARE", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "compute_news_sentiment_acceleration",
                latency,
                success=True,
                num_values=len(result),
            )
            return result
        except Exception as e:
            error_msg = f"Erreur dans compute_news_sentiment_acceleration: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                "compute_news_sentiment_acceleration", 0, success=False, error=str(e)
            )
            miya_alerts(
                error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            return pd.Series(0.0, index=news_data.index)

    def compute_macro_event_momentum(
        self, calendar_data: pd.DataFrame, window: str = "1d"
    ) -> pd.Series:
        """
        Calcule la dynamique des événements macro.

        Args:
            calendar_data (pd.DataFrame): Données du calendrier économique.
            window (str): Fenêtre temporelle (par défaut : '1d').

        Returns:
            pd.Series: Dynamique des événements.
        """
        try:
            start_time = datetime.now()
            required_cols = ["severity", "timestamp"]
            missing_cols = [
                col for col in required_cols if col not in calendar_data.columns
            ]
            if missing_cols:
                error_msg = f"Colonnes manquantes dans calendar_data: {missing_cols}"
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return pd.Series(0.0, index=calendar_data.index)
            calendar_data = calendar_data.copy()
            calendar_data["timestamp"] = pd.to_datetime(
                calendar_data["timestamp"], errors="coerce"
            )
            momentum = (
                calendar_data["severity"]
                .groupby(pd.Grouper(key="timestamp", freq=window))
                .mean()
                .pct_change()
                .fillna(0.0)
            )
            result = momentum.reindex(calendar_data.index, method="ffill").fillna(0.0)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Dynamique des événements calculée: {len(result)} valeurs"
            miya_speak(
                success_msg, tag="CONTEXT_AWARE", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "compute_macro_event_momentum",
                latency,
                success=True,
                num_values=len(result),
            )
            return result
        except Exception as e:
            error_msg = f"Erreur dans compute_macro_event_momentum: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                "compute_macro_event_momentum", 0, success=False, error=str(e)
            )
            miya_alerts(
                error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            return pd.Series(0.0, index=calendar_data.index)

    def compute_cyclic_feature(
        self, data: pd.DataFrame, period: str = "month", func: str = "sin"
    ) -> pd.Series:
        """
        Calcule les features cycliques basées sur le temps.

        Args:
            data (pd.DataFrame): Données principales.
            period (str): Période cyclique ('month' ou 'week').
            func (str): Fonction cyclique ('sin' ou 'cos').

        Returns:
            pd.Series: Feature cyclique.
        """
        try:
            start_time = datetime.now()
            if "timestamp" not in data.columns:
                error_msg = "Colonne timestamp manquante dans data"
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return pd.Series(0.0, index=data.index)
            data = data.copy()
            data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
            if period == "month":
                value = data["timestamp"].dt.month / 12.0
            elif period == "week":
                value = data["timestamp"].dt.isocalendar().week / 5.0
            else:
                error_msg = f"Période invalide: {period}"
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return pd.Series(0.0, index=data.index)
            if func == "sin":
                result = np.sin(2 * np.pi * value)
            elif func == "cos":
                result = np.cos(2 * np.pi * value)
            else:
                error_msg = f"Fonction invalide: {func}"
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return pd.Series(0.0, index=data.index)
            result = pd.Series(result, index=data.index).fillna(0.0)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = (
                f"Feature cyclique {period}_{func} calculée: {len(result)} valeurs"
            )
            miya_speak(
                success_msg, tag="CONTEXT_AWARE", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                f"compute_cyclic_feature_{period}_{func}",
                latency,
                success=True,
                num_values=len(result),
            )
            return result
        except Exception as e:
            error_msg = f"Erreur dans compute_cyclic_feature: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                f"compute_cyclic_feature_{period}_{func}",
                0,
                success=False,
                error=str(e),
            )
            miya_alerts(
                error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            return pd.Series(0.0, index=data.index)

    def compute_roll_yield_curve(self, futures_data: pd.DataFrame) -> pd.Series:
        """
        Calcule la courbe de rendement des futures.

        Args:
            futures_data (pd.DataFrame): Données des futures.

        Returns:
            pd.Series: Courbe de rendement.
        """
        try:
            start_time = datetime.now()
            required_cols = ["near_price", "far_price"]
            missing_cols = [
                col for col in required_cols if col not in futures_data.columns
            ]
            if missing_cols:
                error_msg = f"Colonnes manquantes dans futures_data: {missing_cols}"
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return pd.Series(0.0, index=futures_data.index)
            futures_data = futures_data.copy()
            futures_data["near_price"] = pd.to_numeric(
                futures_data["near_price"], errors="coerce"
            )
            futures_data["far_price"] = pd.to_numeric(
                futures_data["far_price"], errors="coerce"
            )
            yield_curve = (
                futures_data["far_price"] - futures_data["near_price"]
            ) / futures_data["near_price"].replace(0, 1e-6)
            result = yield_curve.fillna(0.0)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Courbe de rendement calculée: {len(result)} valeurs"
            miya_speak(
                success_msg, tag="CONTEXT_AWARE", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "compute_roll_yield_curve",
                latency,
                success=True,
                num_values=len(result),
            )
            return result
        except Exception as e:
            error_msg = f"Erreur dans compute_roll_yield_curve: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                "compute_roll_yield_curve", 0, success=False, error=str(e)
            )
            miya_alerts(
                error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            return pd.Series(0.0, index=futures_data.index)

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
                    error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return False
            shap_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
            if len(shap_df) < 150:
                error_msg = f"Nombre insuffisant de SHAP features: {len(shap_df)} < 150"
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=4
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
                    tag="CONTEXT_AWARE",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "SHAP features validées"
            miya_speak(
                success_msg, tag="CONTEXT_AWARE", voice_profile="calm", priority=1
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
            )
            return len(missing) == 0
        except Exception as e:
            error_msg = (
                f"Erreur validation SHAP features: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance(
                "validate_shap_features", 0, success=False, error=str(e)
            )
            miya_alerts(
                error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=4
            )
            send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            return False

    def cache_metrics(self, metrics: pd.DataFrame, cache_key: str) -> None:
        """
        Met en cache les métriques calculées.

        Args:
            metrics (pd.DataFrame): Métriques à mettre en cache.
            cache_key (str): Clé de cache.
        """
        try:
            start_time = datetime.now()
            cache_path = os.path.join(CACHE_DIR, f"{cache_key}.csv")
            os.makedirs(CACHE_DIR, exist_ok=True)

            def save_cache():
                metrics.to_csv(cache_path, index=False, encoding="utf-8")

            self.with_retries(save_cache)
            self.cache[cache_key] = {"timestamp": datetime.now(), "path": cache_path}
            current_time = datetime.now()
            expired_keys = [
                k
                for k, v in self.cache.items()
                if (current_time - v["timestamp"]).total_seconds()
                > self.config.get("cache_hours", 24) * 3600
            ]
            for k in expired_keys:
                try:
                    os.remove(self.cache[k]["path"])
                except BaseException:
                    pass
                self.cache.pop(k)
            if len(self.cache) > self.max_cache_size:
                oldest_key = min(self.cache, key=lambda k: self.cache[k]["timestamp"])
                try:
                    os.remove(self.cache[oldest_key]["path"])
                except BaseException:
                    pass
                self.cache.pop(oldest_key)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Métriques mises en cache: {cache_path}"
            miya_speak(
                success_msg, tag="CONTEXT_AWARE", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "cache_metrics", latency, success=True, cache_size=len(self.cache)
            )
        except Exception as e:
            error_msg = (
                f"Erreur mise en cache métriques: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance("cache_metrics", 0, success=False, error=str(e))
            miya_alerts(
                error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def plot_metrics(self, metrics: pd.DataFrame, timestamp: str) -> None:
        """
        Génère des visualisations des métriques contextuelles.

        Args:
            metrics (pd.DataFrame): Métriques à visualiser.
            timestamp (str): Timestamp pour le nom du fichier.
        """
        try:
            start_time = datetime.now()
            timestamp_safe = timestamp.replace(":", "-")
            plt.figure(figsize=(12, 6))
            plt.plot(
                metrics["timestamp"],
                metrics["news_sentiment_momentum"],
                label="News Sentiment Momentum",
                color="blue",
            )
            plt.plot(
                metrics["timestamp"],
                metrics["news_volume_spike"],
                label="News Volume Spike",
                color="orange",
            )
            plt.plot(
                metrics["timestamp"],
                metrics["event_volatility_impact"],
                label="Event Volatility Impact",
                color="green",
            )
            plt.title(f"Contextual Metrics - {timestamp}")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(
                FIGURES_DIR, f"context_aware_metrics_{timestamp_safe}.png"
            )
            plt.savefig(plot_path)
            plt.close()
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Visualisations générées: {plot_path}"
            miya_speak(
                success_msg, tag="CONTEXT_AWARE", voice_profile="calm", priority=2
            )
            send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "plot_metrics", latency, success=True, plot_path=plot_path
            )
        except Exception as e:
            error_msg = (
                f"Erreur génération visualisations: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance("plot_metrics", 0, success=False, error=str(e))
            miya_alerts(
                error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def compute_contextual_metrics(
        self,
        data: pd.DataFrame,
        news_data: pd.DataFrame,
        calendar_data: pd.DataFrame,
        futures_data: pd.DataFrame,
        expiry_dates: pd.Series,
        macro_events_path: str = os.path.join(
            BASE_DIR, "data", "events", "macro_events.csv"
        ),
        volatility_history_path: str = os.path.join(
            BASE_DIR, "data", "events", "event_volatility_history.csv"
        ),
    ) -> pd.DataFrame:
        """
        Calcule les métriques contextuelles basées sur les événements macro, nouvelles IQFeed, et cyclicité temporelle.

        Args:
            data (pd.DataFrame): Données principales avec timestamp.
            news_data (pd.DataFrame): Données de nouvelles avec sentiment_score et volume.
            calendar_data (pd.DataFrame): Données du calendrier économique avec severity et weight.
            futures_data (pd.DataFrame): Données des futures avec near_price et far_price.
            expiry_dates (pd.Series): Dates d'expiration des contrats.
            macro_events_path (str): Chemin vers macro_events.csv.
            volatility_history_path (str): Chemin vers event_volatility_history.csv.

        Returns:
            pd.DataFrame: Données enrichies avec 19 métriques contextuelles.
        """
        try:
            start_time = datetime.now()
            if data.empty:
                error_msg = "DataFrame principal vide"
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=5
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                raise ValueError(error_msg)

            cache_key = hashlib.sha256(data.to_json().encode()).hexdigest()
            if cache_key in self.cache:
                cached_data = pd.read_csv(
                    self.cache[cache_key]["path"], encoding="utf-8"
                )
                if (
                    datetime.now() - self.cache[cache_key]["timestamp"]
                ).total_seconds() < self.config.get("cache_hours", 24) * 3600:
                    success_msg = "Features contextuelles récupérées du cache"
                    miya_speak(
                        success_msg,
                        tag="CONTEXT_AWARE",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert(success_msg, priority=1)
                    send_telegram_alert(success_msg)
                    logger.info(success_msg)
                    self.log_performance(
                        "compute_contextual_metrics_cache_hit",
                        0,
                        success=True,
                        num_rows=len(cached_data),
                    )
                    return cached_data

            data = data.copy()
            if "timestamp" not in data.columns:
                error_msg = "Colonne timestamp manquante, création par défaut"
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="warning", priority=3
                )
                send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                logger.warning(error_msg)
                default_start = pd.Timestamp.now()
                data["timestamp"] = pd.date_range(
                    start=default_start, periods=len(data), freq="1min"
                )

            data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
            if data["timestamp"].isna().any():
                error_msg = "NaN dans timestamp, imputés avec la première date valide"
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="warning", priority=3
                )
                send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                logger.warning(error_msg)
                first_valid_time = (
                    data["timestamp"].dropna().iloc[0]
                    if not data["timestamp"].dropna().empty
                    else pd.Timestamp.now()
                )
                data["timestamp"] = data["timestamp"].fillna(first_valid_time)

            # Charger les fichiers d'événements et d'historique de volatilité
            try:
                macro_events = pd.read_csv(macro_events_path)
                volatility_history = pd.read_csv(volatility_history_path)
            except FileNotFoundError as e:
                error_msg = f"Fichier introuvable: {e}"
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                raise
            except Exception as e:
                error_msg = f"Erreur lors du chargement des fichiers: {e}"
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                raise

            # Vérification des colonnes requises dans macro_events
            required_event_cols = ["timestamp", "event_type", "event_impact_score"]
            missing_event_cols = [
                col for col in required_event_cols if col not in macro_events.columns
            ]
            for col in missing_event_cols:
                macro_events[col] = 0
                error_msg = (
                    f"Colonne IQFeed '{col}' manquante dans macro_events, imputée à 0"
                )
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="warning", priority=3
                )
                send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                logger.warning(error_msg)

            # Vérification des colonnes requises dans volatility_history
            required_history_cols = ["event_type", "volatility_impact"]
            missing_history_cols = [
                col
                for col in required_history_cols
                if col not in volatility_history.columns
            ]
            for col in missing_history_cols:
                volatility_history[col] = 0
                error_msg = f"Colonne IQFeed '{col}' manquante dans volatility_history, imputée à 0"
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="warning", priority=3
                )
                send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                logger.warning(error_msg)

            # Conversion des timestamps
            macro_events["timestamp"] = pd.to_datetime(
                macro_events["timestamp"], errors="coerce"
            )
            volatility_history["timestamp"] = pd.to_datetime(
                volatility_history["timestamp"], errors="coerce"
            )

            # Calcul des métriques contextuelles
            metrics = [
                "event_volatility_impact",
                "event_timing_proximity",
                "event_frequency_24h",
                "news_sentiment_momentum",
                "news_event_proximity",
                "macro_event_severity",
                "time_to_expiry_proximity",
                "economic_calendar_weight",
                "news_volume_spike",
                "news_volume_1h",
                "news_volume_1d",
                "news_sentiment_acceleration",
                "macro_event_momentum",
                "month_of_year_sin",
                "month_of_year_cos",
                "week_of_month_sin",
                "week_of_month_cos",
                "roll_yield_curve",
            ]

            # Calculer confidence_drop_rate
            missing_metrics = [m for m in metrics if m not in data.columns]
            confidence_drop_rate = 1.0 - min(
                (len(metrics) - len(missing_metrics)) / len(metrics), 1.0
            )
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(metrics) - len(missing_metrics)} métriques présentes)"
                miya_alerts(
                    alert_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=3
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            data["event_volatility_impact"] = data["timestamp"].apply(
                lambda x: self.calculate_event_volatility_impact(
                    macro_events, volatility_history, x
                )
            )
            data["event_timing_proximity"] = data["timestamp"].apply(
                lambda x: self.calculate_event_timing_proximity(macro_events, x)
            )
            data["event_frequency_24h"] = data["timestamp"].apply(
                lambda x: self.calculate_event_frequency_24h(macro_events, x)
            )
            data["news_sentiment_momentum"] = self.compute_news_sentiment_momentum(
                news_data
            )
            data["news_event_proximity"] = self.compute_news_event_proximity(news_data)
            data["macro_event_severity"] = self.compute_macro_event_severity(
                calendar_data
            )
            data["time_to_expiry_proximity"] = self.compute_time_to_expiry_proximity(
                data, expiry_dates
            )
            data["economic_calendar_weight"] = self.compute_economic_calendar_weight(
                calendar_data
            )
            data["news_volume_spike"] = self.compute_news_volume_spike(
                news_data, window="5min"
            )
            data["news_volume_1h"] = self.compute_news_volume_spike(
                news_data, window="1h"
            )
            data["news_volume_1d"] = self.compute_news_volume_spike(
                news_data, window="1d"
            )
            data["news_sentiment_acceleration"] = (
                self.compute_news_sentiment_acceleration(news_data)
            )
            data["macro_event_momentum"] = self.compute_macro_event_momentum(
                calendar_data
            )
            data["month_of_year_sin"] = self.compute_cyclic_feature(
                data, period="month", func="sin"
            )
            data["month_of_year_cos"] = self.compute_cyclic_feature(
                data, period="month", func="cos"
            )
            data["week_of_month_sin"] = self.compute_cyclic_feature(
                data, period="week", func="sin"
            )
            data["week_of_month_cos"] = self.compute_cyclic_feature(
                data, period="week", func="cos"
            )
            data["roll_yield_curve"] = self.compute_roll_yield_curve(futures_data)

            # Validation SHAP
            self.validate_shap_features(metrics)

            # Mise en cache et sauvegardes
            self.cache_metrics(data, cache_key)
            self.checkpoint(data)
            self.cloud_backup(data)
            self.plot_metrics(data, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Métriques contextuelles calculées: {len(data)} lignes, {len(metrics)} métriques, Confidence drop rate: {confidence_drop_rate:.2f}"
            miya_speak(
                success_msg, tag="CONTEXT_AWARE", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "compute_contextual_metrics",
                latency,
                success=True,
                num_rows=len(data),
                num_metrics=len(metrics),
                confidence_drop_rate=confidence_drop_rate,
            )
            self.save_snapshot(
                "compute_contextual_metrics",
                {
                    "num_rows": len(data),
                    "metrics": metrics,
                    "confidence_drop_rate": confidence_drop_rate,
                },
            )
            return data
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur dans compute_contextual_metrics: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "compute_contextual_metrics", latency, success=False, error=str(e)
            )
            self.save_snapshot("compute_contextual_metrics", {"error": str(e)})
            data["timestamp"] = pd.date_range(
                start=pd.Timestamp.now(), periods=len(data), freq="1min"
            )
            for col in metrics:
                data[col] = 0.0
            return data


if __name__ == "__main__":
    try:
        calculator = ContextAwareFilter()
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "close": np.random.normal(5100, 10, 100),
            }
        )
        news_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "sentiment_score": np.random.uniform(-1, 1, 100),
                "volume": np.random.randint(1, 10, 100),
            }
        )
        calendar_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "severity": np.random.uniform(0, 1, 100),
                "weight": np.random.uniform(0, 1, 100),
            }
        )
        futures_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "near_price": np.random.normal(5100, 10, 100),
                "far_price": np.random.normal(5120, 10, 100),
            }
        )
        expiry_dates = pd.Series(pd.date_range("2025-04-15", periods=100, freq="1d"))
        macro_events = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2025-04-14 08:00",
                        "2025-04-14 09:10",
                        "2025-04-14 10:00",
                        "2025-04-13 09:00",
                        "2025-04-15 09:00",
                    ]
                ),
                "event_type": [
                    "CPI Release",
                    "FOMC Meeting",
                    "Earnings Report",
                    "GDP Release",
                    "Jobs Report",
                ],
                "event_impact_score": [0.8, 0.9, 0.6, 0.7, 0.85],
            }
        )
        event_volatility_history = pd.DataFrame(
            {
                "event_type": [
                    "CPI Release",
                    "FOMC Meeting",
                    "Earnings Report",
                    "GDP Release",
                    "Jobs Report",
                ],
                "volatility_impact": [0.15, 0.20, 0.10, 0.12, 0.18],
                "timestamp": pd.to_datetime(["2025-04-01"] * 5),
            }
        )
        os.makedirs(os.path.join(BASE_DIR, "data", "events"), exist_ok=True)
        macro_events.to_csv(
            os.path.join(BASE_DIR, "data", "events", "macro_events.csv"), index=False
        )
        event_volatility_history.to_csv(
            os.path.join(BASE_DIR, "data", "events", "event_volatility_history.csv"),
            index=False,
        )
        result = calculator.compute_contextual_metrics(
            data, news_data, calendar_data, futures_data, expiry_dates
        )
        print(
            result[
                [
                    "timestamp",
                    "event_volatility_impact",
                    "news_sentiment_momentum",
                    "month_of_year_sin",
                ]
            ].head()
        )
        success_msg = "Test compute_contextual_metrics terminé"
        miya_speak(success_msg, tag="CONTEXT_AWARE", voice_profile="calm", priority=1)
        send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
        logger.info(success_msg)
    except Exception as e:
        error_msg = f"Erreur test: {str(e)}\n{traceback.format_exc()}"
        miya_alerts(error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=3)
        send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)
        logger.error(error_msg)
        raise
