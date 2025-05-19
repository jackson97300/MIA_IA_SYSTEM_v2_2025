# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/api/merge_data_sources.py
# Fusionne les données IQFeed (OHLC, options, volatilité, nouvelles) dans une source consolidée pour les 350/150 SHAP features.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Charge les données via IQFeedProvider (data_provider.py), intègre la volatilité (vix_es_correlation, atr_14, méthode 1),
#        les données d'options (call_iv_atm, put_iv_atm, option_volume, oi_concentration, option_skew, méthode 2),
#        et les nouvelles pour news_impact_score. Génère merged_data.csv avec validation robuste, logs psutil,
#        snapshots, et sauvegardes incrémentielles/distribuées.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.23.0,<2.0.0, psutil>=5.9.8,<6.0.0, pyyaml>=6.0.0,<7.0.0,
#   boto3>=1.26.0,<2.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
# - src/utils/telegram_alert.py
# - src/api/data_provider.py
# - src/api/news_scraper.py
#
# Inputs :
# - config/iqfeed_config.yaml
# - config/credentials.yaml
#
# Outputs :
# - data/features/merged_data.csv
# - data/merge_snapshots/*.json.gz
# - data/logs/merge_performance.csv
# - data/logs/merge_data_sources.log
# - data/checkpoints/merge_data_*.json.gz
#
# Notes :
# - Conforme à structure.txt (version 2.1.3, 2025-05-13).
# - Utilise exclusivement IQFeed via data_provider.py (dxFeed supprimé).
# - Implémente retries (max 3, délai 2^attempt), logs psutil, alertes Telegram, snapshots compressés.
# - Intègre volatilité (méthode 1) : vix_es_correlation, atr_14.
# - Intègre données d’options (méthode 2) : call_iv_atm, put_iv_atm, option_volume, oi_concentration, option_skew.
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des données.
# - Intègre validation SHAP (Phase 17) pour les top 150 features en production.
# - Phases intégrées : Phase 1 (collecte IQFeed), Phase 8 (auto-conscience), Phase 17 (interprétabilité SHAP).
# - Sauvegardes incrémentielles toutes les 5 min dans data/checkpoints/ avec versionnage (5 versions).
# - Sauvegardes distribuées toutes les 15 min vers S3 (configurable).
# - Tests unitaires dans tests/test_merge_data_sources.py.
# - Validation complète prévue pour juin 2025.
# - Évolution future : Intégration de nouvelles sources de données (juin 2025).

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
import numpy as np
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
    filename=os.path.join(BASE_DIR, "data", "logs", "merge_data_sources.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Chemins et constantes
CONFIG_PATH = os.path.join(BASE_DIR, "config", "iqfeed_config.yaml")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "features", "merged_data.csv")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "merge_snapshots")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "data", "checkpoints")
PERFORMANCE_LOG = os.path.join(BASE_DIR, "data", "logs", "merge_performance.csv")
FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR, "data", "features", "feature_importance.csv"
)

# Dtypes stricts pour les colonnes critiques
IQFEED_DTYPES = {
    "timestamp": str,
    "open": float,
    "high": float,
    "low": float,
    "close": float,
    "volume": float,
    "bid_size_level_1": float,
    "ask_size_level_1": float,
    "bid_price_level_1": float,
    "ask_price_level_1": float,
    "total_bid_volume": float,
    "total_ask_volume": float,
    "delta_volume": float,
    "atr_14": float,
    "adx_14": float,
    "rsi_14": float,
    "momentum_10": float,
    "oi_call_strike_1": float,
    "oi_put_strike_1": float,
    "gamma_strike_1": float,
    "delta_strike_1": float,
    "iv_strike_1": float,
    "theta_strike_1": float,
    "strike_1": float,
    "tick_count": float,
    "trade_price": float,
    "trade_size": float,
    "vix_es_correlation": float,
    "call_iv_atm": float,
    "put_iv_atm": float,
    "option_volume": float,
    "oi_concentration": float,
    "option_skew": float,
    "news_impact_score": float,
}

# Variable pour gérer l'arrêt propre
RUNNING = True


class DataMerger:
    """Classe pour gérer la fusion des données IQFeed."""

    def __init__(self, config_path: str = CONFIG_PATH):
        """
        Initialise le merger de données.

        Args:
            config_path (str): Chemin vers le fichier de configuration.
        """
        self.log_buffer = []
        self.buffer_size = 100
        self.checkpoint_versions = []
        try:
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            self.config = self.load_config_with_validation(config_path)
            self.data_provider = get_data_provider("IQFeedProvider")
            signal.signal(signal.SIGINT, self.signal_handler)
            success_msg = "DataMerger initialisé"
            miya_speak(success_msg, tag="MERGE", voice_profile="calm", priority=2)
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config": self.config}, compress=True)
        except Exception as e:
            error_msg = (
                f"Erreur initialisation DataMerger: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="MERGE", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            self.config = {
                "chunk_size": 10000,
                "cache_hours": 24,
                "time_tolerance": "10s",
                "s3_bucket": None,
                "s3_prefix": "merge_data/",
            }
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
            if "data_paths" not in config:
                config["data_paths"] = {}
            required_keys = [
                "chunk_size",
                "cache_hours",
                "time_tolerance",
                "s3_bucket",
                "s3_prefix",
            ]
            missing_keys = [
                key for key in required_keys if key not in config["data_paths"]
            ]
            if missing_keys:
                raise ValueError(f"Clés de configuration manquantes: {missing_keys}")
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Configuration data_paths chargée"
            miya_speak(success_msg, tag="MERGE", voice_profile="calm", priority=1)
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("load_config_with_validation", latency, success=True)
            return config["data_paths"]
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur chargement configuration: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="MERGE", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "load_config_with_validation", latency, success=False, error=str(e)
            )
            return {
                "chunk_size": 10000,
                "cache_hours": 24,
                "time_tolerance": "10s",
                "s3_bucket": None,
                "s3_prefix": "merge_data/",
            }

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ) -> None:
        """
        Journalise les performances des opérations critiques.

        Args:
            operation (str): Nom de l’opération (ex. : merge_data).
            latency (float): Latence en secondes.
            success (bool): Succès de l’opération.
            error (str, optional): Message d’erreur si échec.
            **kwargs: Paramètres supplémentaires (ex. : num_rows, confidence_drop_rate).
        """
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_usage = psutil.cpu_percent()
            if memory_usage > 1024:
                alert_msg = (
                    f"ALERTE: Utilisation mémoire élevée ({memory_usage:.2f} MB)"
                )
                miya_alerts(alert_msg, tag="MERGE", voice_profile="urgent", priority=5)
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
            miya_alerts(error_msg, tag="MERGE", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def save_snapshot(
        self, snapshot_type: str, data: Dict, compress: bool = True
    ) -> None:
        """
        Sauvegarde un instantané des résultats.

        Args:
            snapshot_type (str): Type de snapshot (ex. : merge_data).
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
                miya_alerts(alert_msg, tag="MERGE", voice_profile="urgent", priority=3)
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Snapshot {snapshot_type} sauvegardé: {save_path}"
            miya_speak(success_msg, tag="MERGE", voice_profile="calm", priority=1)
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
            miya_alerts(error_msg, tag="MERGE", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def checkpoint(self, data: pd.DataFrame) -> None:
        """
        Sauvegarde incrémentielle des données fusionnées toutes les 5 minutes avec versionnage (5 versions).

        Args:
            data (pd.DataFrame): Données à sauvegarder.
        """
        try:
            start_time = datetime.now()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_data = {
                "timestamp": timestamp,
                "num_rows": len(data),
                "columns": list(data.columns),
            }
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR, f"merge_data_{timestamp}.json.gz"
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
            miya_speak(success_msg, tag="MERGE", voice_profile="calm", priority=1)
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "checkpoint",
                latency,
                success=True,
                file_size_mb=file_size,
                num_rows=len(data),
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde checkpoint: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance("checkpoint", 0, success=False, error=str(e))
            miya_alerts(error_msg, tag="MERGE", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def cloud_backup(self, data: pd.DataFrame) -> None:
        """
        Sauvegarde distribuée des données fusionnées vers AWS S3 toutes les 15 minutes (configurable).

        Args:
            data (pd.DataFrame): Données à sauvegarder.
        """
        try:
            start_time = datetime.now()
            if not self.config.get("s3_bucket"):
                warning_msg = "S3 bucket non configuré, sauvegarde cloud ignorée"
                miya_speak(
                    warning_msg, tag="MERGE", voice_profile="warning", priority=3
                )
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_data = {
                "timestamp": timestamp,
                "num_rows": len(data),
                "columns": list(data.columns),
            }
            backup_path = f"{self.config['s3_prefix']}merge_data_{timestamp}.csv.gz"
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
            miya_speak(success_msg, tag="MERGE", voice_profile="calm", priority=1)
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "cloud_backup", latency, success=True, num_rows=len(data)
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde cloud S3: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance("cloud_backup", 0, success=False, error=str(e))
            miya_alerts(error_msg, tag="MERGE", voice_profile="urgent", priority=3)
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
                miya_alerts(error_msg, tag="MERGE", voice_profile="urgent", priority=4)
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return False
            shap_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
            if len(shap_df) < 150:
                error_msg = f"Nombre insuffisant de SHAP features: {len(shap_df)} < 150"
                miya_alerts(error_msg, tag="MERGE", voice_profile="urgent", priority=4)
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return False
            valid_features = set(shap_df["feature"].head(150))
            missing = [f for f in features if f not in valid_features]
            if missing:
                warning_msg = f"Features non incluses dans top 150 SHAP: {missing}"
                miya_alerts(
                    warning_msg, tag="MERGE", voice_profile="warning", priority=3
                )
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "SHAP features validées"
            miya_speak(success_msg, tag="MERGE", voice_profile="calm", priority=1)
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
            miya_alerts(error_msg, tag="MERGE", voice_profile="urgent", priority=4)
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
                        error_msg, tag="MERGE", voice_profile="urgent", priority=4
                    )
                    send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    logger.error(error_msg)
                    raise
                delay = delay_base * (2**attempt)
                warning_msg = f"Tentative {attempt+1} échouée, retry après {delay}s"
                miya_speak(
                    warning_msg, tag="MERGE", voice_profile="warning", priority=3
                )
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
                time.sleep(delay)

    def merge_data(
        self,
        ohlc_data: pd.DataFrame,
        options_data: pd.DataFrame,
        news_data: pd.DataFrame,
    ) -> Optional[pd.DataFrame]:
        """
        Fusionne les données OHLC, options et nouvelles via IQFeedProvider (méthode 1, méthode 2).

        Args:
            ohlc_data (pd.DataFrame): Données OHLC avec volatilité (vix_es_correlation, atr_14).
            options_data (pd.DataFrame): Données d'options (call_iv_atm, put_iv_atm, etc.).
            news_data (pd.DataFrame): Données de nouvelles (news_impact_score).

        Returns:
            Optional[pd.DataFrame]: Données fusionnées ou None si échec.
        """
        try:
            start_time = datetime.now()
            required_cols = [
                "timestamp",
                "vix_es_correlation",
                "atr_14",
                "call_iv_atm",
                "put_iv_atm",
                "option_volume",
                "oi_concentration",
                "option_skew",
                "news_impact_score",
            ]

            # Validation des entrées
            for df, source in [
                (ohlc_data, "ohlc"),
                (options_data, "options"),
                (news_data, "news"),
            ]:
                if df.empty:
                    error_msg = f"Données {source} vides"
                    miya_alerts(
                        error_msg, tag="MERGE", voice_profile="urgent", priority=4
                    )
                    send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    logger.error(error_msg)
                    self.log_performance(
                        "merge_data", 0, success=False, error=error_msg
                    )
                    return None
                if "timestamp" not in df.columns:
                    error_msg = f"Colonne 'timestamp' manquante dans {source}"
                    miya_alerts(
                        error_msg, tag="MERGE", voice_profile="urgent", priority=4
                    )
                    send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    logger.error(error_msg)
                    self.log_performance(
                        "merge_data", 0, success=False, error=error_msg
                    )
                    return None

            # Calculer confidence_drop_rate
            all_cols = (
                set(ohlc_data.columns)
                | set(options_data.columns)
                | set(news_data.columns)
            )
            missing_cols = [col for col in required_cols if col not in all_cols]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(all_cols)} colonnes)"
                miya_alerts(alert_msg, tag="MERGE", voice_profile="urgent", priority=3)
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            # Validation SHAP
            if not self.validate_shap_features(required_cols[1:]):  # Exclure timestamp
                warning_msg = "Certaines features ne sont pas dans les top 150 SHAP"
                miya_speak(
                    warning_msg, tag="MERGE", voice_profile="warning", priority=3
                )
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)

            # Fusion des données
            df = pd.concat(
                [
                    ohlc_data.set_index("timestamp"),
                    options_data.set_index("timestamp"),
                    news_data.set_index("timestamp"),
                ],
                axis=1,
                join="outer",
            ).reset_index()
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])

            # Vérifier les colonnes requises
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                error_msg = f"Colonnes manquantes dans fusion: {missing_cols}"
                miya_alerts(error_msg, tag="MERGE", voice_profile="urgent", priority=4)
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                self.log_performance("merge_data", 0, success=False, error=error_msg)
                return None

            # Gestion des NaN et inf
            numeric_cols = [
                col
                for col in df.columns
                if col != "timestamp" and df[col].dtype in ["float64", "int64"]
            ]
            df[numeric_cols] = (
                df[numeric_cols]
                .replace([float("inf"), -float("inf")], np.nan)
                .fillna(0)
            )
            string_cols = [
                col
                for col in df.columns
                if col not in numeric_cols and col != "timestamp"
            ]
            df[string_cols] = df[string_cols].fillna("")

            # Regroupement temporel
            time_tolerance = self.config.get("time_tolerance", "10s")
            pre_group_lines = len(df)
            df = (
                df.groupby(pd.Grouper(key="timestamp", freq=time_tolerance))
                .last()
                .reset_index()
            )
            post_group_lines = len(df)
            success_msg = f"Après regroupement temporel (tolérance {time_tolerance}): {pre_group_lines} -> {post_group_lines} lignes"
            miya_speak(success_msg, tag="MERGE", voice_profile="calm", priority=1)
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)

            # Supprimer doublons
            pre_duplicate_lines = len(df)
            df = df.drop_duplicates(subset=["timestamp"], keep="last")
            post_duplicate_lines = len(df)
            success_msg = f"Après suppression des doublons temporels: {pre_duplicate_lines} -> {post_duplicate_lines} lignes"
            miya_speak(success_msg, tag="MERGE", voice_profile="calm", priority=1)
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)

            # Sauvegarde
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

            def save_data():
                df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

            self.with_retries(save_data)

            # Sauvegardes
            self.checkpoint(df)
            self.cloud_backup(df)

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Fusion IQFeed sauvegardée: {len(df)} lignes, Confidence drop rate: {confidence_drop_rate:.2f}"
            miya_speak(success_msg, tag="MERGE", voice_profile="calm", priority=1)
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "merge_data",
                latency,
                success=True,
                num_rows=len(df),
                confidence_drop_rate=confidence_drop_rate,
            )
            self.save_snapshot(
                "merge_data",
                {"num_rows": len(df), "columns": list(df.columns)},
                compress=True,
            )
            return df
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur fusion données: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="MERGE", voice_profile="urgent", priority=4)
            send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("merge_data", latency, success=False, error=str(e))
            self.save_snapshot("error_merge_data", {"error": str(e)}, compress=True)
            return None

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
            success_msg = "Arrêt du merger de données en cours..."
            miya_speak(success_msg, tag="MERGE", voice_profile="calm", priority=2)
            send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            logger.info("Arrêt du merger initié")

            # Sauvegarder le cache actuel si disponible
            if os.path.exists(OUTPUT_PATH):
                df = pd.read_csv(OUTPUT_PATH, encoding="utf-8")
                if not df.empty:
                    self.checkpoint(df)
                    self.save_snapshot(
                        "shutdown",
                        {
                            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                            "num_rows": len(df),
                        },
                        compress=True,
                    )

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Merger de données arrêté proprement"
            miya_speak(success_msg, tag="MERGE", voice_profile="calm", priority=2)
            send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("signal_handler", latency, success=True)
            exit(0)
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur arrêt merger: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="MERGE", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("signal_handler", latency, success=False, error=str(e))
            exit(1)


if __name__ == "__main__":
    try:
        merger = DataMerger()
        ohlc_data = merger.with_retries(
            lambda: merger.data_provider.fetch_ohlc(symbol="ES")
        )
        options_data = merger.with_retries(
            lambda: merger.data_provider.fetch_options(symbol="ES")
        )
        news_data = merger.with_retries(lambda: merger.data_provider.fetch_news())
        merged_data = merger.merge_data(ohlc_data, options_data, news_data)
        if merged_data is not None:
            print(f"Données fusionnées: {len(merged_data)} lignes")
            success_msg = "Fusion des données terminée"
            miya_speak(success_msg, tag="MERGE", voice_profile="calm", priority=1)
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
    except Exception as e:
        error_msg = f"Erreur principale: {str(e)}\n{traceback.format_exc()}"
        miya_alerts(error_msg, tag="MERGE", voice_profile="urgent", priority=5)
        send_alert(error_msg, priority=5)
        send_telegram_alert(error_msg)
        logger.error(error_msg)
