# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/api/option_chain_fetch.py
# Récupère les données de chaîne d’options via IQFeed pour spotgamma_recalculator.py et options_levels_service.py.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Collecte les métriques d’options (call_iv_atm, put_iv_atm, option_volume, oi_concentration, option_skew)
#        via IQFeedProvider (data_provider.py), avec cache horaire, snapshots, et sauvegardes.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, requests>=2.28.0,<3.0.0, psutil>=5.9.8,<6.0.0, pyyaml>=6.0.0,<7.0.0,
#   boto3>=1.26.0,<2.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
# - src/utils/telegram_alert.py
# - src/api/data_provider.py
#
# Inputs :
# - config/iqfeed_config.yaml
# - config/credentials.yaml
# - config/es_config.yaml
#
# Outputs :
# - data/iqfeed/option_chain.csv
# - data/iqfeed/cache/options/hourly_YYYYMMDD_HH.csv
# - data/option_chain_snapshots/*.json.gz
# - data/logs/option_chain_performance.csv
# - data/logs/option_chain_fetch.log
# - data/checkpoints/option_chain_*.json.gz
# - data/option_chain_dashboard.json
#
# Notes :
# - Conforme à structure.txt (version 2.1.3, 2025-05-13).
# - Utilise exclusivement IQFeed via data_provider.py (dxFeed supprimé).
# - Implémente retries (max 3, délai 2^attempt), logs psutil, alertes Telegram, snapshots compressés.
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des données.
# - Intègre validation SHAP (Phase 17) pour les métriques d’options (ex. : call_iv_atm, put_iv_atm).
# - Phases intégrées : Phase 1 (collecte IQFeed), Phase 8 (auto-conscience), Phase 17 (interprétabilité SHAP).
# - Sauvegardes incrémentielles toutes les 5 min dans data/checkpoints/ avec versionnage (5 versions).
# - Sauvegardes distribuées toutes les 15 min vers S3 (configurable).
# - Tests unitaires dans tests/test_option_chain_fetch.py.
# - Validation complète prévue pour juin 2025.
# - Évolution future : Optimisation des appels API IQFeed (juin 2025).

import gzip
import json
import logging
import os
import signal
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional

import boto3
import pandas as pd
import psutil

from src.api.data_provider import get_data_provider
from src.model.utils.alert_manager import send_alert
from src.model.utils.config_manager import get_config
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.utils.telegram_alert import send_telegram_alert

# Configuration du logging
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
os.makedirs(os.path.join(BASE_DIR, "data", "logs"), exist_ok=True)
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "data", "logs", "option_chain_fetch.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Chemins et constantes
CONFIG_PATH = os.path.join(BASE_DIR, "config", "iqfeed_config.yaml")
CREDENTIALS_PATH = os.path.join(BASE_DIR, "config", "credentials.yaml")
ES_CONFIG_PATH = os.path.join(BASE_DIR, "config", "es_config.yaml")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "iqfeed", "option_chain.csv")
CACHE_DIR = os.path.join(BASE_DIR, "data", "iqfeed", "cache", "options")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "option_chain_snapshots")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "data", "checkpoints")
PERFORMANCE_LOG = os.path.join(BASE_DIR, "data", "logs", "option_chain_performance.csv")
DASHBOARD_PATH = os.path.join(BASE_DIR, "data", "option_chain_dashboard.json")
DEFAULT_CONFIG = {
    "connection": {
        "endpoint": "https://api.iqfeed.net",
        "retry_attempts": 3,
        "retry_delay_base": 2.0,
        "timeout": 30,
        "cache_hours": 1,
    },
    "symbols": ["ES"],
    "data_types": {"options": {"enabled": True}},
}

# Variable pour gérer l'arrêt propre
RUNNING = True


class OptionChainFetcher:
    """Classe pour collecter les données de chaîne d'options via IQFeed."""

    def __init__(
        self, config_path: str = CONFIG_PATH, credentials_path: str = CREDENTIALS_PATH
    ):
        """
        Initialise le fetcher de chaîne d'options.

        Args:
            config_path (str): Chemin vers le fichier de configuration.
            credentials_path (str): Chemin vers le fichier des identifiants.
        """
        self.log_buffer = []
        self.buffer_size = 100
        self.checkpoint_versions = []
        self.dashboard_data = {"last_update": None, "options": {}}
        try:
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            os.makedirs(CACHE_DIR, exist_ok=True)
            self.config = self.load_config_with_validation(config_path)
            self.credentials = self.load_config_with_validation(credentials_path)
            self.api_key = self.credentials.get("iqfeed_api_key", "")
            if not self.api_key:
                raise ValueError("Clé API IQFeed manquante dans credentials.yaml")
            self.data_provider = get_data_provider("IQFeedProvider")
            self.es_config = self.load_es_config()
            signal.signal(signal.SIGINT, self.signal_handler)
            success_msg = "OptionChainFetcher initialisé"
            miya_speak(
                success_msg, tag="IQFEED_OPTIONS", voice_profile="calm", priority=2
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("init", 0, success=True)
            self.save_snapshot(
                "init", {"config": self.config, "api_key": "masked"}, compress=True
            )
        except Exception as e:
            error_msg = f"Erreur initialisation OptionChainFetcher: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="IQFEED_OPTIONS", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            self.config = DEFAULT_CONFIG
            self.credentials = {"iqfeed_api_key": ""}
            self.data_provider = None
            self.es_config = {
                "retry_attempts": 3,
                "retry_delay_base": 2,
                "timeout_seconds": 1800,
                "max_data_age_seconds": 300,
            }

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
            section = "iqfeed" if "iqfeed_config" in config_path else "credentials"
            if section not in config:
                config[section] = {}
            required_keys = (
                [
                    "endpoint",
                    "retry_attempts",
                    "retry_delay_base",
                    "timeout",
                    "cache_hours",
                    "symbols",
                    "data_types",
                ]
                if section == "iqfeed"
                else ["iqfeed_api_key"]
            )
            missing_keys = [key for key in required_keys if key not in config[section]]
            if missing_keys:
                raise ValueError(
                    f"Clés de configuration manquantes dans {section}: {missing_keys}"
                )
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Configuration {section} chargée"
            miya_speak(
                success_msg, tag="IQFEED_OPTIONS", voice_profile="calm", priority=1
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
                error_msg, tag="IQFEED_OPTIONS", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "load_config_with_validation", latency, success=False, error=str(e)
            )
            return (
                DEFAULT_CONFIG
                if "iqfeed_config" in config_path
                else {"iqfeed_api_key": ""}
            )

    def load_es_config(self) -> Dict:
        """
        Charge les paramètres de prétraitement via config_manager.py.

        Returns:
            Dict: Configuration de prétraitement.
        """
        try:
            start_time = datetime.now()
            config = get_config(os.path.basename(ES_CONFIG_PATH))
            es_config = config.get("preprocessing", {}) | config.get(
                "spotgamma_recalculator", {}
            )
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Configuration es_config chargée"
            miya_speak(
                success_msg, tag="IQFEED_OPTIONS", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("load_es_config", latency, success=True)
            return es_config
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur chargement es_config: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, tag="IQFEED_OPTIONS", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("load_es_config", latency, success=False, error=str(e))
            return {
                "retry_attempts": 3,
                "retry_delay_base": 2,
                "timeout_seconds": 1800,
                "max_data_age_seconds": 300,
            }

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ) -> None:
        """
        Journalise les performances des opérations critiques.

        Args:
            operation (str): Nom de l’opération (ex. : fetch_option_chain).
            latency (float): Latence en secondes.
            success (bool): Succès de l’opération.
            error (str, optional): Message d’erreur si échec.
            **kwargs: Paramètres supplémentaires (ex. : symbol, num_rows, confidence_drop_rate).
        """
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_usage = psutil.cpu_percent()
            if memory_usage > 1024:
                alert_msg = (
                    f"ALERTE: Utilisation mémoire élevée ({memory_usage:.2f} MB)"
                )
                miya_alerts(
                    alert_msg, tag="IQFEED_OPTIONS", voice_profile="urgent", priority=5
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
                os.makedirs(os.path.dirname(PERFORMANCE_LOG), exist_ok=True)

                def write_log():
                    log_df.to_csv(
                        PERFORMANCE_LOG,
                        mode="a" if os.path.exists(PERFORMANCE_LOG) else "w",
                        header=not os.path.exists(PERFORMANCE_LOG),
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
                error_msg, tag="IQFEED_OPTIONS", voice_profile="urgent", priority=3
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
            snapshot_type (str): Type de snapshot (ex. : fetch_option_chain).
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
                    alert_msg, tag="IQFEED_OPTIONS", voice_profile="urgent", priority=3
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Snapshot {snapshot_type} sauvegardé: {save_path}"
            miya_speak(
                success_msg, tag="IQFEED_OPTIONS", voice_profile="calm", priority=1
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
                error_msg, tag="IQFEED_OPTIONS", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def checkpoint(self, data: pd.DataFrame, symbol: str) -> None:
        """
        Sauvegarde incrémentielle des données toutes les 5 minutes avec versionnage (5 versions).

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            symbol (str): Symbole des options.
        """
        try:
            start_time = datetime.now()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_data = {
                "timestamp": timestamp,
                "num_rows": len(data),
                "symbol": symbol,
                "columns": list(data.columns),
            }
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR, f"option_chain_{timestamp}.json.gz"
            )
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)

            def write_checkpoint():
                with gzip.open(checkpoint_path, "wt", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, indent=4)
                data.to_csv(
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
                success_msg, tag="IQFEED_OPTIONS", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "checkpoint",
                latency,
                success=True,
                file_size_mb=file_size,
                num_rows=len(data),
                symbol=symbol,
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde checkpoint: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance("checkpoint", 0, success=False, error=str(e))
            miya_alerts(
                error_msg, tag="IQFEED_OPTIONS", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def cloud_backup(self, data: pd.DataFrame, symbol: str) -> None:
        """
        Sauvegarde distribuée des données vers AWS S3 toutes les 15 minutes (configurable).

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            symbol (str): Symbole des options.
        """
        try:
            start_time = datetime.now()
            if not self.config.get("s3_bucket"):
                warning_msg = "S3 bucket non configuré, sauvegarde cloud ignorée"
                miya_speak(
                    warning_msg,
                    tag="IQFEED_OPTIONS",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.config['s3_prefix']}option_chain_{timestamp}.csv.gz"
            temp_path = os.path.join(CHECKPOINT_DIR, f"temp_s3_{timestamp}.csv.gz")

            def write_temp():
                data.to_csv(
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
                success_msg, tag="IQFEED_OPTIONS", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "cloud_backup", latency, success=True, num_rows=len(data), symbol=symbol
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde cloud S3: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance("cloud_backup", 0, success=False, error=str(e))
            miya_alerts(
                error_msg, tag="IQFEED_OPTIONS", voice_profile="urgent", priority=3
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
                    error_msg, tag="IQFEED_OPTIONS", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return False
            shap_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
            if len(shap_df) < 150:
                error_msg = f"Nombre insuffisant de SHAP features: {len(shap_df)} < 150"
                miya_alerts(
                    error_msg, tag="IQFEED_OPTIONS", voice_profile="urgent", priority=4
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
                    tag="IQFEED_OPTIONS",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "SHAP features validées"
            miya_speak(
                success_msg, tag="IQFEED_OPTIONS", voice_profile="calm", priority=1
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
                error_msg, tag="IQFEED_OPTIONS", voice_profile="urgent", priority=4
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
                        tag="IQFEED_OPTIONS",
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
                    tag="IQFEED_OPTIONS",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
                time.sleep(delay)

    def check_data_freshness(self, data: pd.DataFrame, max_age_seconds: int) -> bool:
        """
        Vérifie si les données sont fraîches (timestamp < max_age_seconds).

        Args:
            data (pd.DataFrame): Données à vérifier.
            max_age_seconds (int): Âge maximum des données en secondes.

        Returns:
            bool: True si les données sont fraîches, False sinon.
        """
        try:
            start_time = datetime.now()
            if "timestamp" not in data.columns:
                error_msg = "Colonne timestamp manquante"
                miya_alerts(
                    error_msg, tag="IQFEED_OPTIONS", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                self.log_performance(
                    "check_data_freshness", 0, success=False, error=error_msg
                )
                return False
            latest_timestamp = pd.to_datetime(data["timestamp"].max())
            age_seconds = (datetime.now() - latest_timestamp).total_seconds()
            if age_seconds > max_age_seconds:
                error_msg = f"Données trop anciennes: {age_seconds} secondes"
                miya_alerts(
                    error_msg, tag="IQFEED_OPTIONS", voice_profile="urgent", priority=3
                )
                send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                logger.warning(error_msg)
                self.log_performance(
                    "check_data_freshness",
                    0,
                    success=False,
                    error=error_msg,
                    age_seconds=age_seconds,
                )
                return False
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "check_data_freshness", latency, success=True, age_seconds=age_seconds
            )
            return True
        except Exception as e:
            error_msg = (
                f"Erreur vérification fraîcheur: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, tag="IQFEED_OPTIONS", voice_profile="urgent", priority=4
            )
            send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("check_data_freshness", 0, success=False, error=str(e))
            return False

    def validate_data(self, data: Dict) -> bool:
        """
        Valide les données d’options (types, plages, métriques spécifiques).

        Args:
            data (Dict): Données à valider.

        Returns:
            bool: True si les données sont valides, False sinon.
        """
        try:
            start_time = datetime.now()
            critical_cols = [
                "open_interest",
                "call_iv_atm",
                "put_iv_atm",
                "option_volume",
                "oi_concentration",
                "option_skew",
                "gamma",
                "delta",
                "vega",
                "price",
            ]
            for col in critical_cols:
                if col in data:
                    value = data[col]
                    if not isinstance(value, (int, float)) or pd.isna(value):
                        error_msg = f"Colonne {col} non scalaire ou invalide: {value}"
                        miya_alerts(
                            error_msg,
                            tag="IQFEED_OPTIONS",
                            voice_profile="urgent",
                            priority=4,
                        )
                        send_alert(error_msg, priority=4)
                        send_telegram_alert(error_msg)
                        logger.error(error_msg)
                        self.log_performance(
                            "validate_data", 0, success=False, error=error_msg
                        )
                        return False
                    if (
                        col in ["call_iv_atm", "put_iv_atm", "option_skew"]
                        and value < 0
                    ):
                        error_msg = f"Valeur négative pour {col}: {value}"
                        miya_alerts(
                            error_msg,
                            tag="IQFEED_OPTIONS",
                            voice_profile="urgent",
                            priority=4,
                        )
                        send_alert(error_msg, priority=4)
                        send_telegram_alert(error_msg)
                        logger.error(error_msg)
                        self.log_performance(
                            "validate_data", 0, success=False, error=error_msg
                        )
                        return False
                    if col == "delta" and (value < -1 or value > 1):
                        error_msg = f"Delta hors plage [-1, 1]: {value}"
                        miya_alerts(
                            error_msg,
                            tag="IQFEED_OPTIONS",
                            voice_profile="urgent",
                            priority=4,
                        )
                        send_alert(error_msg, priority=4)
                        send_telegram_alert(error_msg)
                        logger.error(error_msg)
                        self.log_performance(
                            "validate_data", 0, success=False, error=error_msg
                        )
                        return False
                    if col in ["open_interest", "option_volume", "price"] and value < 0:
                        error_msg = f"Valeur négative pour {col}: {value}"
                        miya_alerts(
                            error_msg,
                            tag="IQFEED_OPTIONS",
                            voice_profile="urgent",
                            priority=4,
                        )
                        send_alert(error_msg, priority=4)
                        send_telegram_alert(error_msg)
                        logger.error(error_msg)
                        self.log_performance(
                            "validate_data", 0, success=False, error=error_msg
                        )
                        return False
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("validate_data", latency, success=True)
            return True
        except Exception as e:
            error_msg = f"Erreur validation données: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="IQFEED_OPTIONS", voice_profile="urgent", priority=4
            )
            send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("validate_data", 0, success=False, error=str(e))
            return False

    def load_from_cache(
        self, symbol: str, max_age_seconds: int
    ) -> Optional[pd.DataFrame]:
        """
        Charge les données depuis le cache si fraîches.

        Args:
            symbol (str): Symbole des options.
            max_age_seconds (int): Âge maximum des données en secondes.

        Returns:
            Optional[pd.DataFrame]: Données en cache ou None.
        """
        try:
            start_time = datetime.now()
            cache_path = os.path.join(
                CACHE_DIR,
                f"options_{symbol}_{datetime.now().strftime('%Y%m%d_%H')}.csv",
            )
            if not os.path.exists(cache_path):
                return None
            df = pd.read_csv(cache_path, encoding="utf-8")
            if self.check_data_freshness(df, max_age_seconds):
                success_msg = f"Données chargées depuis cache: {cache_path}"
                miya_speak(
                    success_msg, tag="IQFEED_OPTIONS", voice_profile="calm", priority=1
                )
                send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)
                logger.info(success_msg)
                latency = (datetime.now() - start_time).total_seconds()
                self.log_performance(
                    "load_from_cache", latency, success=True, symbol=symbol
                )
                return df
            return None
        except Exception as e:
            error_msg = f"Erreur chargement cache: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="IQFEED_OPTIONS", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("load_from_cache", 0, success=False, error=str(e))
            return None

    def save_to_cache(self, data: pd.DataFrame, symbol: str) -> None:
        """
        Sauvegarde les données dans le cache horaire.

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            symbol (str): Symbole des options.
        """
        try:
            start_time = datetime.now()
            cache_path = os.path.join(
                CACHE_DIR,
                f"options_{symbol}_{datetime.now().strftime('%Y%m%d_%H')}.csv",
            )
            os.makedirs(CACHE_DIR, exist_ok=True)

            def write_cache():
                data.to_csv(cache_path, index=False, encoding="utf-8")

            self.with_retries(write_cache)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Cache horaire sauvegardé: {cache_path}"
            miya_speak(
                success_msg, tag="IQFEED_OPTIONS", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "save_to_cache",
                latency,
                success=True,
                symbol=symbol,
                num_rows=len(data),
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde cache: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="IQFEED_OPTIONS", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("save_to_cache", 0, success=False, error=str(e))

    def update_dashboard(self, data: pd.DataFrame, symbol: str) -> None:
        """
        Met à jour le JSON pour mia_dashboard.py.

        Args:
            data (pd.DataFrame): Données à inclure dans le dashboard.
            symbol (str): Symbole des options.
        """
        try:
            start_time = datetime.now()
            if data.empty:
                return
            self.dashboard_data["last_update"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            self.dashboard_data["options"][symbol] = {
                "num_rows": len(data),
                "latest": data.iloc[-1][
                    [
                        "strike",
                        "option_type",
                        "open_interest",
                        "call_iv_atm",
                        "put_iv_atm",
                    ]
                ].to_dict(),
            }
            os.makedirs(os.path.dirname(DASHBOARD_PATH), exist_ok=True)

            def save_json():
                with open(DASHBOARD_PATH, "w", encoding="utf-8") as f:
                    json.dump(self.dashboard_data, f, indent=4)

            self.with_retries(save_json)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Mise à jour dashboard JSON"
            miya_speak(
                success_msg, tag="IQFEED_OPTIONS", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "update_dashboard", latency, success=True, symbol=symbol
            )
        except Exception as e:
            error_msg = (
                f"Erreur mise à jour dashboard: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, tag="IQFEED_OPTIONS", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("update_dashboard", 0, success=False, error=str(e))

    def fetch_option_chain(self, symbol: str = "ES") -> pd.DataFrame:
        """
        Récupère les données de chaîne d’options IQFeed avec cache et retries.

        Args:
            symbol (str): Symbole des options (par défaut : "ES").

        Returns:
            pd.DataFrame: Données de la chaîne d’options.
        """
        try:
            start_time = datetime.now()
            max_age_seconds = self.es_config.get("max_data_age_seconds", 300)
            required_features = [
                "call_iv_atm",
                "put_iv_atm",
                "option_volume",
                "oi_concentration",
                "option_skew",
                "gamma",
                "delta",
                "vega",
                "price",
            ]
            if not self.validate_shap_features(required_features):
                warning_msg = "Certaines features ne sont pas dans les top 150 SHAP"
                miya_speak(
                    warning_msg,
                    tag="IQFEED_OPTIONS",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)

            # Vérifier le cache
            cached_data = self.load_from_cache(symbol, max_age_seconds)
            if cached_data is not None:
                self.update_dashboard(cached_data, symbol)
                latency = (datetime.now() - start_time).total_seconds()
                self.log_performance(
                    "fetch_option_chain",
                    latency,
                    success=True,
                    symbol=symbol,
                    num_rows=len(cached_data),
                )
                return cached_data

            # Récupérer les données via IQFeed
            def fetch():
                if not self.data_provider:
                    raise ValueError("Fournisseur de données IQFeed non disponible")
                df = self.data_provider.fetch_options(symbol=symbol)
                if df.empty:
                    raise ValueError("Aucune donnée reçue de IQFeed")
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                return df

            df = self.with_retries(
                fetch,
                max_attempts=self.config["retry_attempts"],
                delay_base=self.config["retry_delay_base"],
            )
            if df is None or df.empty:
                error_msg = f"Aucune donnée valide récupérée pour {symbol}"
                miya_alerts(
                    error_msg, tag="IQFEED_OPTIONS", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                self.log_performance(
                    "fetch_option_chain",
                    0,
                    success=False,
                    error=error_msg,
                    symbol=symbol,
                )
                return pd.DataFrame()

            # Validation des colonnes
            required_cols = [
                "timestamp",
                "strike",
                "option_type",
                "open_interest",
                "call_iv_atm",
                "put_iv_atm",
                "option_volume",
                "oi_concentration",
                "option_skew",
                "gamma",
                "delta",
                "vega",
                "price",
            ]
            missing_cols = [col for col in required_cols if col not in df.columns]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(df.columns)} colonnes)"
                miya_alerts(
                    alert_msg, tag="IQFEED_OPTIONS", voice_profile="urgent", priority=3
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            # Validation des données
            for _, row in df.iterrows():
                if not self.validate_data(row.to_dict()):
                    error_msg = f"Données invalides pour {symbol}"
                    miya_alerts(
                        error_msg,
                        tag="IQFEED_OPTIONS",
                        voice_profile="urgent",
                        priority=4,
                    )
                    send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    logger.error(error_msg)
                    return pd.DataFrame()

            # Vérification de la fraîcheur
            if not self.check_data_freshness(df, max_age_seconds):
                error_msg = f"Données non fraîches pour {symbol}"
                miya_alerts(
                    error_msg, tag="IQFEED_OPTIONS", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return pd.DataFrame()

            # Sauvegarde
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

            def save_output():
                mode = "a" if os.path.exists(OUTPUT_PATH) else "w"
                df.to_csv(
                    OUTPUT_PATH,
                    mode=mode,
                    index=False,
                    header=not os.path.exists(OUTPUT_PATH),
                    encoding="utf-8",
                )

            self.with_retries(save_output)
            self.save_to_cache(df, symbol)
            self.update_dashboard(df, symbol)
            self.checkpoint(df, symbol)
            self.cloud_backup(df, symbol)

            # Alertes pour forte volatilité
            for _, row in df.iterrows():
                if row["call_iv_atm"] > 0.25 or row["put_iv_atm"] > 0.25:
                    alert_msg = f"Forte volatilité: IV={row['call_iv_atm']:.2f}/{row['put_iv_atm']:.2f} pour {symbol}"
                    miya_alerts(
                        alert_msg,
                        tag="IQFEED_OPTIONS",
                        voice_profile="urgent",
                        priority=3,
                    )
                    send_alert(alert_msg, priority=3)
                    send_telegram_alert(alert_msg)
                    logger.info(alert_msg)

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Données récupérées pour {symbol}: {len(df)} lignes, Confidence drop rate: {confidence_drop_rate:.2f}"
            miya_speak(
                success_msg, tag="IQFEED_OPTIONS", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "fetch_option_chain",
                latency,
                success=True,
                symbol=symbol,
                num_rows=len(df),
                confidence_drop_rate=confidence_drop_rate,
            )
            self.save_snapshot(
                "fetch_option_chain",
                {
                    "num_rows": len(df),
                    "symbol": symbol,
                    "confidence_drop_rate": confidence_drop_rate,
                },
                compress=True,
            )
            return df
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur fetch_option_chain pour {symbol}: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="IQFEED_OPTIONS", voice_profile="urgent", priority=5
            )
            send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "fetch_option_chain",
                latency,
                success=False,
                error=str(e),
                symbol=symbol,
            )
            self.save_snapshot(
                "error_fetch_option_chain",
                {"error": str(e), "symbol": symbol},
                compress=True,
            )
            return pd.DataFrame()

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
            success_msg = "Arrêt du fetcher de chaîne d’options en cours..."
            miya_speak(
                success_msg, tag="IQFEED_OPTIONS", voice_profile="calm", priority=2
            )
            send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            logger.info("Arrêt du fetcher initié")

            # Sauvegarder le cache actuel si disponible
            if os.path.exists(OUTPUT_PATH):
                df = pd.read_csv(OUTPUT_PATH, encoding="utf-8")
                if not df.empty:
                    self.checkpoint(df, "ES")
                    self.save_snapshot(
                        "shutdown",
                        {
                            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                            "num_rows": len(df),
                        },
                        compress=True,
                    )

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Fetcher de chaîne d’options arrêté proprement"
            miya_speak(
                success_msg, tag="IQFEED_OPTIONS", voice_profile="calm", priority=2
            )
            send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("signal_handler", latency, success=True)
            exit(0)
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur arrêt fetcher: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="IQFEED_OPTIONS", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("signal_handler", latency, success=False, error=str(e))
            exit(1)


if __name__ == "__main__":
    try:
        fetcher = OptionChainFetcher()
        df = fetcher.fetch_option_chain(symbol="ES")
        if not df.empty:
            print(f"Données récupérées:\n{df.head()}")
        else:
            print("Aucune donnée récupérée")
        success_msg = "Test fetch_option_chain terminé"
        miya_speak(success_msg, tag="IQFEED_OPTIONS", voice_profile="calm", priority=1)
        send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
        logger.info(success_msg)
    except Exception as e:
        error_msg = f"Erreur test principal: {str(e)}\n{traceback.format_exc()}"
        miya_alerts(error_msg, tag="IQFEED_OPTIONS", voice_profile="urgent", priority=5)
        send_alert(error_msg, priority=5)
        send_telegram_alert(error_msg)
        logger.error(error_msg)
