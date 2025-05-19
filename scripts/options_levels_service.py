# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/scripts/options_levels_service.py
# Service en arrière-plan pour recalculer les niveaux critiques d’options toutes les 15 minutes
# (ou 5 minutes si vix_es_correlation > 25.0) à partir des données IQFeed.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Exécute des recalculs périodiques des niveaux d’options en utilisant spotgamma_recalculator.py,
#        ajuste les recalculs avec predicted_vix via LSTM (méthode 12), et gère une boucle asynchrone
#        avec retries. Intègre des sauvegardes incrémentielles/versionnées toutes les 5 minutes et
#        distribuées (S3) toutes les 15 minutes.
#
# Dépendances :
# - schedule>=1.2.0, pandas>=2.0.0,<3.0.0, psutil>=5.9.8,<6.0.0, asyncio, json, gzip, boto3>=1.26.0,<2.0.0,
#   pyyaml>=6.0.0,<7.0.0
# - src/features/spotgamma_recalculator.py
# - src/api/iqfeed_fetch.py
# - src/model/utils/miya_console.py
# - src/utils/telegram_alert.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/neural_pipeline.py
#
# Inputs :
# - config/es_config.yaml
# - data/iqfeed/option_chain.csv
#
# Outputs :
# - data/options_snapshots/*.json.gz
# - data/logs/options_levels_service_performance.csv
# - data/logs/options_levels_service.log
# - data/options_levels_dashboard.json
# - data/figures/options/*.png
# - data/checkpoints/options_levels_*.json.gz
#
# Notes :
# - Conforme à structure.txt (version 2.1.3, 2025-05-13).
# - Utilise exclusivement IQFeed via iqfeed_fetch.py.
# - Implémente retries (max 3, délai 2^attempt), logs psutil, alertes Telegram, snapshots compressés.
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des données.
# - Intègre validation SHAP (Phase 17) pour les top 150 features en production.
# - Phases intégrées : Phase 1 (collecte IQFeed), Phase 8 (auto-conscience), Phase 16 (ensemble learning via LSTM),
#                     Phase 17 (interprétabilité SHAP).
# - Sauvegardes incrémentielles toutes les 5 min dans data/checkpoints/ avec versionnage (5 versions).
# - Sauvegardes distribuées toutes les 15 min vers S3 (configurable).
# - Tests unitaires dans tests/test_options_levels_service.py.
# - Validation complète prévue pour juin 2025.
# - Évolution future : Migration API Investing.com (juin 2025), intégration complète LSTM.

import asyncio
import gzip
import hashlib
import json
import logging
import os
import signal
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List

import boto3
import matplotlib.pyplot as plt
import pandas as pd
import psutil
import schedule

from src.api.iqfeed_fetch import fetch_option_chain
from src.features.spotgamma_recalculator import SpotGammaRecalculator
from src.model.neural_pipeline import predict_vix
from src.model.utils.alert_manager import send_alert
from src.model.utils.config_manager import get_config
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.utils.telegram_alert import send_telegram_alert

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "options_snapshots")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "data", "checkpoints")
CSV_LOG_PATH = os.path.join(
    BASE_DIR, "data", "logs", "options_levels_service_performance.csv"
)
DASHBOARD_PATH = os.path.join(BASE_DIR, "data", "options_levels_dashboard.json")
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures", "options")
FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR, "data", "features", "feature_importance.csv"
)

# Configuration du logging
os.makedirs(os.path.join(BASE_DIR, "data", "logs"), exist_ok=True)
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "data", "logs", "options_levels_service.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Variable pour gérer l'arrêt propre
RUNNING = True


class OptionsLevelsService:
    """
    Service pour recalculer les niveaux critiques d’options en arrière-plan.
    """

    def __init__(
        self, config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml")
    ):
        """
        Initialise le service.

        Args:
            config_path (str): Chemin vers le fichier de configuration.
        """
        self.log_buffer = []
        self.cache = {}
        self.buffer_size = 100
        self.max_cache_size = 1000
        self.checkpoint_versions = []
        try:
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            os.makedirs(FIGURES_DIR, exist_ok=True)
            self.config = self.load_config_with_validation(config_path)
            self.recalculator = SpotGammaRecalculator(config_path)
            signal.signal(signal.SIGINT, self.signal_handler)
            miya_speak(
                "OptionsLevelsService initialisé",
                tag="LEVELS_SERVICE",
                voice_profile="calm",
                priority=2,
            )
            send_alert("OptionsLevelsService initialisé", priority=1)
            send_telegram_alert("OptionsLevelsService initialisé")
            logger.info("OptionsLevelsService initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config": self.config}, compress=True)
        except Exception as e:
            error_msg = f"Erreur initialisation OptionsLevelsService: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="LEVELS_SERVICE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            self.config = {
                "max_data_age_seconds": 300,
                "s3_bucket": None,
                "s3_prefix": "options_levels/",
            }
            self.recalculator = SpotGammaRecalculator(config_path)

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
            config = get_config("es_config.yaml")
            if "options_levels_service" not in config:
                config["options_levels_service"] = {}
            required_keys = ["max_data_age_seconds", "s3_bucket", "s3_prefix"]
            missing_keys = [
                key
                for key in required_keys
                if key not in config["options_levels_service"]
            ]
            if missing_keys:
                raise ValueError(f"Clés de configuration manquantes: {missing_keys}")
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Configuration options_levels_service chargée"
            miya_speak(
                success_msg, tag="LEVELS_SERVICE", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("load_config_with_validation", latency, success=True)
            return config["options_levels_service"]
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur chargement configuration: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, tag="LEVELS_SERVICE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "load_config_with_validation", latency, success=False, error=str(e)
            )
            return {
                "max_data_age_seconds": 300,
                "s3_bucket": None,
                "s3_prefix": "options_levels/",
            }

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ) -> None:
        """
        Journalise les performances des opérations critiques.

        Args:
            operation (str): Nom de l’opération (ex. : update_levels).
            latency (float): Latence en secondes.
            success (bool): Succès de l’opération.
            error (str, optional): Message d’erreur si échec.
            **kwargs: Paramètres supplémentaires (ex. : num_levels, vix_es_correlation).
        """
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_usage = psutil.cpu_percent()
            if memory_usage > 1024:
                alert_msg = (
                    f"ALERTE: Utilisation mémoire élevée ({memory_usage:.2f} MB)"
                )
                miya_alerts(
                    alert_msg, tag="LEVELS_SERVICE", voice_profile="urgent", priority=5
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

                self.with_retries_sync(write_log)
                self.log_buffer = []
            logger.info(f"Performance journalisée pour {operation}. CPU: {cpu_usage}%")
        except Exception as e:
            error_msg = (
                f"Erreur journalisation performance: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, tag="LEVELS_SERVICE", voice_profile="urgent", priority=3
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
            snapshot_type (str): Type de snapshot (ex. : update_levels).
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

            self.with_retries_sync(write_snapshot)
            save_path = f"{snapshot_path}.gz" if compress else snapshot_path
            file_size = os.path.getsize(save_path) / 1024 / 1024
            if file_size > 1.0:
                alert_msg = f"Snapshot size {file_size:.2f} MB exceeds 1 MB"
                miya_alerts(
                    alert_msg, tag="LEVELS_SERVICE", voice_profile="urgent", priority=3
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Snapshot {snapshot_type} sauvegardé: {save_path}"
            miya_speak(
                success_msg, tag="LEVELS_SERVICE", voice_profile="calm", priority=1
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
                error_msg, tag="LEVELS_SERVICE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def checkpoint(self, levels: Dict) -> None:
        """
        Sauvegarde incrémentielle des niveaux toutes les 5 minutes avec versionnage (5 versions).

        Args:
            levels (Dict): Niveaux recalculés à sauvegarder.
        """
        try:
            start_time = datetime.now()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_data = {
                "timestamp": timestamp,
                "levels": levels,
                "vix_es_correlation": levels.get("vix_es_correlation", 0.0),
                "predicted_vix": levels.get("predicted_vix", 0.0),
            }
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR, f"options_levels_{timestamp}.json.gz"
            )
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)

            def write_checkpoint():
                with gzip.open(checkpoint_path, "wt", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, indent=4)

            self.with_retries_sync(write_checkpoint)
            self.checkpoint_versions.append(checkpoint_path)
            if len(self.checkpoint_versions) > 5:
                oldest = self.checkpoint_versions.pop(0)
                if os.path.exists(oldest):
                    os.remove(oldest)
            file_size = os.path.getsize(checkpoint_path) / 1024 / 1024
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Checkpoint sauvegardé: {checkpoint_path}"
            miya_speak(
                success_msg, tag="LEVELS_SERVICE", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "checkpoint", latency, success=True, file_size_mb=file_size
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde checkpoint: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance("checkpoint", 0, success=False, error=str(e))
            miya_alerts(
                error_msg, tag="LEVELS_SERVICE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def cloud_backup(self, levels: Dict) -> None:
        """
        Sauvegarde distribuée des niveaux vers AWS S3 toutes les 15 minutes (configurable).

        Args:
            levels (Dict): Niveaux recalculés à sauvegarder.
        """
        try:
            start_time = datetime.now()
            if not self.config.get("s3_bucket"):
                warning_msg = "S3 bucket non configuré, sauvegarde cloud ignorée"
                miya_speak(
                    warning_msg,
                    tag="LEVELS_SERVICE",
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
                "levels": levels,
                "vix_es_correlation": levels.get("vix_es_correlation", 0.0),
                "predicted_vix": levels.get("predicted_vix", 0.0),
            }
            backup_path = (
                f"{self.config['s3_prefix']}options_levels_{timestamp}.json.gz"
            )
            temp_path = os.path.join(CHECKPOINT_DIR, f"temp_s3_{timestamp}.json.gz")

            def write_temp():
                with gzip.open(temp_path, "wt", encoding="utf-8") as f:
                    json.dump(backup_data, f, indent=4)

            self.with_retries_sync(write_temp)
            s3_client = boto3.client("s3")

            def upload_s3():
                s3_client.upload_file(temp_path, self.config["s3_bucket"], backup_path)

            self.with_retries_sync(upload_s3)
            os.remove(temp_path)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Sauvegarde cloud S3 effectuée: {backup_path}"
            miya_speak(
                success_msg, tag="LEVELS_SERVICE", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("cloud_backup", latency, success=True)
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde cloud S3: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance("cloud_backup", 0, success=False, error=str(e))
            miya_alerts(
                error_msg, tag="LEVELS_SERVICE", voice_profile="urgent", priority=3
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
                    error_msg, tag="LEVELS_SERVICE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return False
            shap_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
            if len(shap_df) < 150:
                error_msg = f"Nombre insuffisant de SHAP features: {len(shap_df)} < 150"
                miya_alerts(
                    error_msg, tag="LEVELS_SERVICE", voice_profile="urgent", priority=4
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
                    tag="LEVELS_SERVICE",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "SHAP features validées"
            miya_speak(
                success_msg, tag="LEVELS_SERVICE", voice_profile="calm", priority=1
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
                error_msg, tag="LEVELS_SERVICE", voice_profile="urgent", priority=4
            )
            send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "validate_shap_features", latency, success=False, error=str(e)
            )
            return False

    def save_dashboard_status(
        self, status: Dict, status_file: str = DASHBOARD_PATH
    ) -> None:
        """
        Sauvegarde l'état du service pour mia_dashboard.py.

        Args:
            status (Dict): État du service.
            status_file (str): Chemin du fichier JSON.
        """
        try:
            start_time = datetime.now()
            os.makedirs(os.path.dirname(status_file), exist_ok=True)

            def write_status():
                with open(status_file, "w", encoding="utf-8") as f:
                    json.dump(status, f, indent=4)

            self.with_retries_sync(write_status)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"État sauvegardé dans {status_file}"
            miya_speak(
                success_msg, tag="LEVELS_SERVICE", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("save_dashboard_status", latency, success=True)
            self.save_snapshot(
                "save_dashboard_status", {"status_file": status_file}, compress=True
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde dashboard: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance(
                "save_dashboard_status", 0, success=False, error=str(e)
            )
            miya_alerts(
                error_msg, tag="LEVELS_SERVICE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def plot_levels(self, levels: Dict, timestamp: str) -> None:
        """
        Génère des visualisations pour les niveaux recalculés.

        Args:
            levels (Dict): Niveaux recalculés.
            timestamp (str): Horodatage pour le nom du fichier.
        """
        start_time = datetime.now()
        try:
            if not levels or not isinstance(levels, dict):
                warning_msg = "Aucun niveau à visualiser"
                miya_speak(
                    warning_msg,
                    tag="LEVELS_SERVICE",
                    voice_profile="warning",
                    priority=2,
                )
                send_alert(warning_msg, priority=2)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
                return

            timestamp_safe = timestamp.replace(":", "-")
            os.makedirs(FIGURES_DIR, exist_ok=True)

            plt.figure(figsize=(10, 6))
            strikes = [
                levels.get("put_wall", 0),
                levels.get("call_wall", 0),
                levels.get("zero_gamma", 0),
            ]
            labels = ["Put Wall", "Call Wall", "Zero Gamma"]
            plt.bar(labels, strikes, color=["red", "green", "purple"])
            plt.title(f"Niveaux Critiques d’Options - {timestamp}")
            plt.ylabel("Strike")
            plt.grid(True)
            plot_path = os.path.join(FIGURES_DIR, f"levels_{timestamp_safe}.png")
            plt.savefig(plot_path)
            plt.close()

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Visualisations générées: {plot_path}"
            miya_speak(
                success_msg, tag="LEVELS_SERVICE", voice_profile="calm", priority=2
            )
            send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("plot_levels", latency, success=True)
            self.save_snapshot("plot_levels", {"plot_path": plot_path}, compress=True)
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur génération visualisations: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, tag="LEVELS_SERVICE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("plot_levels", latency, success=False, error=str(e))

    def check_iqfeed_freshness(
        self, option_chain: pd.DataFrame, max_age_seconds: int = 300
    ) -> bool:
        """
        Vérifie si les données IQFeed sont fraîches (timestamp < max_age_seconds).

        Args:
            option_chain (pd.DataFrame): Données de la chaîne d’options.
            max_age_seconds (int): Âge maximum des données en secondes.

        Returns:
            bool: True si les données sont fraîches, False sinon.
        """
        start_time = datetime.now()
        try:
            if "timestamp" not in option_chain.columns:
                error_msg = "Colonne timestamp manquante dans option_chain"
                miya_alerts(
                    error_msg, tag="LEVELS_SERVICE", voice_profile="warning", priority=3
                )
                send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                logger.warning(error_msg)
                self.log_performance(
                    "check_iqfeed_freshness",
                    0,
                    success=False,
                    error="Colonne timestamp manquante",
                )
                return False
            latest_timestamp = pd.to_datetime(option_chain["timestamp"].max())
            current_time = datetime.now()
            age_seconds = (current_time - latest_timestamp).total_seconds()
            if age_seconds > max_age_seconds:
                error_msg = f"Données IQFeed trop anciennes: {age_seconds} secondes"
                miya_alerts(
                    error_msg, tag="LEVELS_SERVICE", voice_profile="warning", priority=3
                )
                send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                logger.warning(error_msg)
                self.log_performance(
                    "check_iqfeed_freshness",
                    0,
                    success=False,
                    error=f"Données trop anciennes: {age_seconds}s",
                )
                return False
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "check_iqfeed_freshness", latency, success=True, age_seconds=age_seconds
            )
            return True
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur vérification fraîcheur IQFeed: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="LEVELS_SERVICE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "check_iqfeed_freshness", latency, success=False, error=str(e)
            )
            return False

    async def fetch_predicted_vix(self, option_chain: pd.DataFrame) -> float:
        """
        Récupère predicted_vix via neural_pipeline.py (LSTM, méthode 12).

        Args:
            option_chain (pd.DataFrame): Données de la chaîne d’options.

        Returns:
            float: Valeur prédite de VIX.
        """
        try:
            start_time = datetime.now()
            required_cols = [
                "vix_es_correlation",
                "iv_atm",
                "key_strikes_1",
                "max_pain_strike",
                "net_gamma",
                "zero_gamma",
                "dealer_zones_count",
                "vol_trigger",
                "ref_px",
                "data_release",
            ]
            missing_cols = [
                col for col in required_cols if col not in option_chain.columns
            ]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(option_chain.columns)} features)"
                miya_alerts(
                    alert_msg, tag="LEVELS_SERVICE", voice_profile="urgent", priority=3
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)
            if not self.validate_shap_features(required_cols):
                warning_msg = "Certaines features VIX ne sont pas dans les top 150 SHAP"
                miya_speak(
                    warning_msg,
                    tag="LEVELS_SERVICE",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
            predicted_vix = (
                predict_vix(option_chain) if not option_chain.empty else 20.0
            )
            if not isinstance(predicted_vix, (int, float)):
                raise ValueError(
                    f"predicted_vix doit être numérique, reçu: {type(predicted_vix)}"
                )
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Predicted VIX: {predicted_vix:.2f}, Confidence drop rate: {confidence_drop_rate:.2f}"
            miya_speak(
                success_msg, tag="LEVELS_SERVICE", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "fetch_predicted_vix",
                latency,
                success=True,
                predicted_vix=predicted_vix,
                confidence_drop_rate=confidence_drop_rate,
            )
            return float(predicted_vix)
        except Exception as e:
            error_msg = (
                f"Erreur récupération predicted VIX: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance("fetch_predicted_vix", 0, success=False, error=str(e))
            miya_alerts(
                error_msg, tag="LEVELS_SERVICE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            return 20.0

    def get_recalculation_interval(self, vix_es_correlation: float) -> int:
        """
        Détermine l’intervalle de recalcul basé sur vix_es_correlation.

        Args:
            vix_es_correlation (float): Corrélation VIX/ES.

        Returns:
            int: Intervalle en secondes (300 si > 25.0, sinon 900).
        """
        try:
            start_time = datetime.now()
            interval = 300 if vix_es_correlation > 25.0 else 900
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Intervalle de recalcul: {interval}s (vix_es_correlation={vix_es_correlation:.2f})"
            miya_speak(
                success_msg, tag="LEVELS_SERVICE", voice_profile="calm", priority=1
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "get_recalculation_interval",
                latency,
                success=True,
                vix_es_correlation=vix_es_correlation,
            )
            return interval
        except Exception as e:
            error_msg = f"Erreur détermination intervalle recalcul: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                "get_recalculation_interval", 0, success=False, error=str(e)
            )
            miya_alerts(
                error_msg, tag="LEVELS_SERVICE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            return 900

    def with_retries_sync(
        self, func: callable, max_attempts: int = 3, delay_base: float = 2.0
    ) -> Any:
        """
        Exécute une fonction synchrone avec retries (max 3, délai exponentiel).

        Args:
            func (callable): Fonction synchrone à exécuter.
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
                        tag="LEVELS_SERVICE",
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
                    tag="LEVELS_SERVICE",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
                time.sleep(delay)

    async def with_retries_async(
        self, func: callable, max_attempts: int = 3, delay_base: float = 2.0
    ) -> Any:
        """
        Exécute une fonction asynchrone avec retries (max 3, délai exponentiel).

        Args:
            func (callable): Fonction asynchrone à exécuter.
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
                result = await func()
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
                        tag="LEVELS_SERVICE",
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
                    tag="LEVELS_SERVICE",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
                await asyncio.sleep(delay)

    async def update_levels_async(self) -> Dict:
        """
        Recalcule et sauvegarde les niveaux critiques d’options à partir des données IQFeed (async).

        Returns:
            Dict: Niveaux recalculés.
        """

        async def compute_levels():
            max_age_seconds = self.config.get("max_data_age_seconds", 300)
            option_chain = await asyncio.get_event_loop().run_in_executor(
                None, fetch_option_chain
            )
            if option_chain.empty:
                raise ValueError("Aucune donnée reçue de fetch_option_chain")

            required_cols = [
                "vix_es_correlation",
                "iv_atm",
                "key_strikes_1",
                "max_pain_strike",
                "net_gamma",
                "zero_gamma",
                "dealer_zones_count",
                "vol_trigger",
                "ref_px",
                "data_release",
            ]
            missing_cols = [
                col for col in required_cols if col not in option_chain.columns
            ]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(option_chain.columns)} features)"
                miya_alerts(
                    alert_msg, tag="LEVELS_SERVICE", voice_profile="urgent", priority=3
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            vix_es_correlation = option_chain.get(
                "vix_es_correlation", pd.Series([0.0])
            ).iloc[0]
            predicted_vix = await self.fetch_predicted_vix(option_chain)

            cache_key = hashlib.sha256(option_chain.to_json().encode()).hexdigest()
            if (
                cache_key in self.cache
                and (
                    datetime.now() - self.cache[cache_key]["timestamp"]
                ).total_seconds()
                < 3600
            ):
                levels = self.cache[cache_key]["levels"]
                success_msg = f"Niveaux récupérés du cache pour {cache_key}"
                miya_speak(
                    success_msg, tag="LEVELS_SERVICE", voice_profile="calm", priority=1
                )
                send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)
                self.log_performance(
                    "update_levels_cache_hit",
                    0,
                    success=True,
                    confidence_drop_rate=confidence_drop_rate,
                )
                return levels

            if not self.check_iqfeed_freshness(option_chain, max_age_seconds):
                error_msg = "Données IQFeed non fraîches, recalcul annulé"
                miya_alerts(
                    error_msg, tag="LEVELS_SERVICE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.warning(error_msg)
                raise ValueError(error_msg)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            levels = self.recalculator.recalculate_levels(
                option_chain, timestamp, mode="live", predicted_vix=predicted_vix
            )
            levels["vix_es_correlation"] = vix_es_correlation
            levels["predicted_vix"] = predicted_vix
            self.recalculator.save_levels(levels)

            self.cache[cache_key] = {"timestamp": datetime.now(), "levels": levels}
            if len(self.cache) > self.max_cache_size:
                self.cache.pop(next(iter(self.cache)))

            self.plot_levels(levels, timestamp)
            self.save_snapshot(
                "update_levels",
                {
                    "timestamp": timestamp,
                    "num_levels": len(levels),
                    "vix_es_correlation": vix_es_correlation,
                    "predicted_vix": predicted_vix,
                    "confidence_drop_rate": confidence_drop_rate,
                },
            )
            self.save_dashboard_status(
                {
                    "last_run": timestamp,
                    "success": True,
                    "num_levels": len(levels),
                    "recent_errors": len(
                        [log for log in self.log_buffer if not log["success"]]
                    ),
                    "average_latency": (
                        sum(log["latency"] for log in self.log_buffer)
                        / len(self.log_buffer)
                        if self.log_buffer
                        else 0
                    ),
                    "vix_es_correlation": vix_es_correlation,
                    "predicted_vix": predicted_vix,
                    "confidence_drop_rate": confidence_drop_rate,
                }
            )

            return levels

        try:
            start_time = datetime.now()
            levels = await self.with_retries_async(compute_levels)
            latency = (datetime.now() - start_time).total_seconds()
            confidence_drop_rate = levels.get("confidence_drop_rate", 0.0)
            success_msg = f"Niveaux recalculés pour {levels['timestamp']}, Confidence drop rate: {confidence_drop_rate:.2f}"
            miya_speak(
                success_msg, tag="LEVELS_SERVICE", voice_profile="calm", priority=2
            )
            send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "update_levels_async",
                latency,
                success=True,
                num_levels=len(levels),
                vix_es_correlation=levels.get("vix_es_correlation", 0.0),
                confidence_drop_rate=confidence_drop_rate,
            )
            return levels
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur mise à jour niveaux async: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, tag="LEVELS_SERVICE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "update_levels_async", latency, success=False, error=str(e)
            )
            return {}

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
            miya_speak(
                "Arrêt du service en cours...",
                tag="LEVELS_SERVICE",
                voice_profile="calm",
                priority=2,
            )
            send_alert("Arrêt du service en cours...", priority=2)
            send_telegram_alert("Arrêt du service en cours...")
            logger.info("Arrêt du service initié")

            def save_final_levels():
                option_chain = fetch_option_chain()
                if self.check_iqfeed_freshness(
                    option_chain, self.config.get("max_data_age_seconds", 300)
                ):
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    levels = self.recalculator.recalculate_levels(
                        option_chain, timestamp, mode="live"
                    )
                    self.recalculator.save_levels(levels)
                    self.checkpoint(levels)
                    self.save_snapshot(
                        "shutdown",
                        {"timestamp": timestamp, "num_levels": len(levels)},
                        compress=True,
                    )

            self.with_retries_sync(save_final_levels)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Service arrêté proprement"
            miya_speak(
                success_msg, tag="LEVELS_SERVICE", voice_profile="calm", priority=2
            )
            send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("signal_handler", latency, success=True)
            exit(0)
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur arrêt service: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="LEVELS_SERVICE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("signal_handler", latency, success=False, error=str(e))
            exit(1)

    def run(self) -> None:
        """
        Exécute le service de recalcul des niveaux (synchrone, conservé pour compatibilité).
        """
        start_time = datetime.now()
        try:
            miya_speak(
                "Service de mise à jour des niveaux démarré (synchrone)",
                tag="LEVELS_SERVICE",
                voice_profile="calm",
                priority=2,
            )
            send_alert(
                "Service de mise à jour des niveaux démarré (synchrone)", priority=2
            )
            send_telegram_alert(
                "Service de mise à jour des niveaux démarré (synchrone)"
            )
            logger.info("Service démarré (synchrone)")

            schedule.every(15).minutes.do(self.update_levels_sync)

            while RUNNING:
                schedule.run_pending()
                time.sleep(1)

            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("run", latency, success=True)
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur service niveaux: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="LEVELS_SERVICE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("run", latency, success=False, error=str(e))
            raise

    def update_levels_sync(self) -> None:
        """
        Version synchrone de update_levels pour compatibilité.
        """
        loop = asyncio.get_event_loop()
        levels = loop.run_until_complete(self.update_levels_async())
        if levels:
            self.checkpoint(levels)

    async def run_async(self) -> None:
        """
        Exécute le service de recalcul des niveaux (asynchrone, préféré).
        """
        global RUNNING
        start_time = datetime.now()
        last_checkpoint = start_time
        last_cloud_backup = start_time
        try:
            miya_speak(
                "Service de mise à jour des niveaux démarré (asynchrone)",
                tag="LEVELS_SERVICE",
                voice_profile="calm",
                priority=2,
            )
            send_alert(
                "Service de mise à jour des niveaux démarré (asynchrone)", priority=2
            )
            send_telegram_alert(
                "Service de mise à jour des niveaux démarré (asynchrone)"
            )
            logger.info("Service démarré (asynchrone)")

            async def update_loop():
                nonlocal last_checkpoint, last_cloud_backup
                option_chain = await asyncio.get_event_loop().run_in_executor(
                    None, fetch_option_chain
                )
                vix_es_correlation = option_chain.get(
                    "vix_es_correlation", pd.Series([0.0])
                ).iloc[0]
                interval = self.get_recalculation_interval(vix_es_correlation)
                levels = await self.update_levels_async()
                if levels:
                    now = datetime.now()
                    if (now - last_checkpoint).total_seconds() >= 300:  # 5 minutes
                        self.checkpoint(levels)
                        last_checkpoint = now
                    if (now - last_cloud_backup).total_seconds() >= 900:  # 15 minutes
                        self.cloud_backup(levels)
                        last_cloud_backup = now
                await asyncio.sleep(interval)

            while RUNNING:
                await self.with_retries_async(update_loop)

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Service asynchrone terminé"
            miya_speak(
                success_msg, tag="LEVELS_SERVICE", voice_profile="calm", priority=2
            )
            send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("run_async", latency, success=True)
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur service niveaux async: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, tag="LEVELS_SERVICE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("run_async", latency, success=False, error=str(e))
            raise


if __name__ == "__main__":
    service = OptionsLevelsService()
    asyncio.run(service.run_async())
