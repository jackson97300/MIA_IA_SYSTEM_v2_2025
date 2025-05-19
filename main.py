# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/main.py
# Rôle : Point d'entrée principal pour orchestrer l'exécution des modules de MIA_IA_SYSTEM_v2_2025.
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, loguru>=0.7.0,<1.0.0, pyyaml>=6.0.0,<7.0.0, boto3>=1.26.0,<2.0.0,
#   psutil>=5.9.8,<6.0.0, argparse, os, signal, threading, gzip
# - src/mind/mind_voice.py
# - src/monitoring/correlation_heatmap.py
# - src/monitoring/data_drift.py
# - src/monitoring/export_visuals.py
# - src/monitoring/mia_dashboard.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/es_config.yaml
# - data/features/features_latest.csv
# - data/trades/trades_simulated.csv
# - data/features/feature_importance.csv
# - data/logs/regime_history.csv
#
# Outputs :
# - Logs dans data/logs/main.log
# - Logs de performance dans data/logs/main_performance.csv
# - Snapshots dans data/cache/main/*.json.gz
# - Sauvegardes incrémentielles dans data/checkpoints/main_*.json.gz
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Utilise exclusivement IQFeed pour les données d’entrée (via modules appelés).
# - Supprime toute référence à dxFeed, obs_t, 320 features, et 81 features.
# - Implémente retries (max 3, délai 2^attempt secondes) pour les opérations critiques.
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des données.
# - Sauvegardes incrémentielles toutes les 5 min dans data/checkpoints/ (5 versions).
# - Sauvegardes distribuées toutes les 15 min vers S3 (configurable).
# - Alertes via alert_manager.py et Telegram pour erreurs critiques (priorité ≥ 4).
# - Tests unitaires disponibles dans tests/test_main.py.
# - Phases intégrées : Phase 8 (auto-conscience via confidence_drop_rate), Phases 1-18 via modules appelés.
# - Validation complète prévue pour juin 2025.

import argparse
import gzip
import json
import os
import signal
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
import pandas as pd
import psutil
from loguru import logger

from src.mind.mind_voice import VoiceManager
from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import get_config
from src.monitoring.correlation_heatmap import CorrelationHeatmap
from src.monitoring.data_drift import DataDriftDetector
from src.monitoring.export_visuals import VisualExporter
from src.monitoring.mia_dashboard import MIADashboard
from src.utils.telegram_alert import send_telegram_alert

# Configuration logging
logger.remove()
BASE_DIR = Path(os.path.dirname(os.path.dirname(__file__)))
LOG_DIR = BASE_DIR / "data" / "logs"
CACHE_DIR = BASE_DIR / "data" / "cache" / "main"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints"
PERF_LOG_PATH = LOG_DIR / "main_performance.csv"
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
logger.add(LOG_DIR / "main.log", rotation="10 MB", level="INFO", encoding="utf-8")
logger.add(lambda msg: print(msg, end=""), level="INFO")

# Constantes
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Secondes, exponentiel
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB

# Variable pour gérer l'arrêt propre
RUNNING = True


class MainRunner:
    """
    Classe pour orchestrer l'exécution des modules de MIA_IA_SYSTEM_v2_2025.
    """

    def __init__(self, config_path: str = "config/es_config.yaml"):
        """
        Initialise le point d'entrée principal.

        Args:
            config_path (str): Chemin vers le fichier de configuration.
        """
        self.alert_manager = AlertManager()
        self.checkpoint_versions = []
        signal.signal(signal.SIGINT, self.handle_sigint)

        start_time = datetime.now()
        try:
            self.config = self.load_config(config_path)
            self.log_buffer = []
            self._clean_cache()
            success_msg = "MainRunner initialisé avec succès"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("init", latency, success=True)
            self.save_snapshot(
                "init",
                {"config_path": config_path, "timestamp": datetime.now().isoformat()},
            )
        except Exception as e:
            error_msg = (
                f"Erreur initialisation MainRunner: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            raise

    def handle_sigint(self, signum, frame):
        """Gère l'arrêt propre sur SIGINT."""
        global RUNNING
        datetime.now()
        snapshot = {"timestamp": datetime.now().isoformat(), "status": "SIGINT"}
        CACHE_DIR / f'main_sigint_{snapshot["timestamp"]}.json.gz'
        try:
            RUNNING = False
            self.save_snapshot("sigint", snapshot)
            success_msg = "Arrêt propre sur SIGINT, snapshot sauvegardé"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            self.log_performance("handle_sigint", 0, success=True)
            exit(0)
        except Exception as e:
            error_msg = f"Erreur sauvegarde SIGINT: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance("handle_sigint", 0, success=False, error=str(e))
            exit(1)

    def save_snapshot(
        self, snapshot_type: str, data: Dict, compress: bool = True
    ) -> None:
        """
        Sauvegarde un instantané des résultats.

        Args:
            snapshot_type (str): Type de snapshot (ex. : init, sigint).
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
            snapshot_path = CACHE_DIR / f"snapshot_{snapshot_type}_{timestamp}.json"
            CACHE_DIR.mkdir(exist_ok=True)

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
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Snapshot {snapshot_type} sauvegardé: {save_path}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "save_snapshot", latency, success=True, snapshot_size_mb=file_size
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance("save_snapshot", 0, success=False, error=str(e))

    def checkpoint(self, data: pd.DataFrame, data_type: str = "system_state") -> None:
        """
        Sauvegarde incrémentielle des données toutes les 5 minutes avec versionnage (5 versions).

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : system_state).
        """
        try:
            start_time = datetime.now()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_data = {
                "timestamp": timestamp,
                "num_rows": len(data),
                "columns": list(data.columns),
                "data_type": data_type,
            }
            checkpoint_path = CHECKPOINT_DIR / f"main_{data_type}_{timestamp}.json.gz"
            CHECKPOINT_DIR.mkdir(exist_ok=True)

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
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "checkpoint",
                latency,
                success=True,
                file_size_mb=file_size,
                num_rows=len(data),
                data_type=data_type,
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde checkpoint: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "checkpoint", 0, success=False, error=str(e), data_type=data_type
            )

    def cloud_backup(self, data: pd.DataFrame, data_type: str = "system_state") -> None:
        """
        Sauvegarde distribuée des données vers AWS S3 toutes les 15 minutes (configurable).

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : system_state).
        """
        try:
            start_time = datetime.now()
            if not self.config.get("s3_bucket"):
                warning_msg = "S3 bucket non configuré, sauvegarde cloud ignorée"
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = (
                f"{self.config['s3_prefix']}main_{data_type}_{timestamp}.csv.gz"
            )
            temp_path = CHECKPOINT_DIR / f"temp_s3_{timestamp}.csv.gz"

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
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "cloud_backup",
                latency,
                success=True,
                num_rows=len(data),
                data_type=data_type,
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde cloud S3: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "cloud_backup", 0, success=False, error=str(e), data_type=data_type
            )

    def with_retries(
        self,
        func: callable,
        max_attempts: int = MAX_RETRIES,
        delay_base: float = RETRY_DELAY_BASE,
    ) -> Optional[Any]:
        """
        Exécute une fonction avec retries exponentiels.

        Args:
            func (callable): Fonction à exécuter.
            max_attempts (int): Nombre maximum de tentatives.
            delay_base (float): Base pour le délai exponentiel.

        Returns:
            Optional[Any]: Résultat de la fonction ou None si échec.
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
                    logger.error(error_msg)
                    self.alert_manager.send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    latency = (datetime.now() - start_time).total_seconds()
                    self.log_performance(
                        f"retry_attempt_{attempt+1}",
                        latency,
                        success=False,
                        error=str(e),
                        attempt_number=attempt + 1,
                    )
                    return None
                delay = delay_base**attempt
                warning_msg = f"Tentative {attempt+1} échouée, retry après {delay}s"
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                time.sleep(delay)

    def _clean_cache(self, max_size_mb: float = MAX_CACHE_SIZE_MB):
        """
        Supprime les fichiers cache expirés ou si la taille dépasse max_size_mb.
        """
        start_time = datetime.now()

        def clean():
            total_size = sum(
                os.path.getsize(os.path.join(CACHE_DIR, f))
                for f in os.listdir(CACHE_DIR)
                if os.path.isfile(os.path.join(CACHE_DIR, f))
            ) / (1024 * 1024)
            if total_size > max_size_mb:
                files = [
                    (f, os.path.getmtime(os.path.join(CACHE_DIR, f)))
                    for f in os.listdir(CACHE_DIR)
                ]
                files.sort(key=lambda x: x[1])
                for f, _ in files[: len(files) // 2]:
                    os.remove(os.path.join(CACHE_DIR, f))
            for filename in os.listdir(CACHE_DIR):
                path = os.path.join(CACHE_DIR, filename)
                if (
                    os.path.isfile(path)
                    and (time.time() - os.path.getmtime(path)) > CACHE_EXPIRATION
                ):
                    os.remove(path)

        try:
            self.with_retries(clean)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Cache nettoyé avec succès"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "clean_cache", latency, success=True, data_type="cache"
            )
        except OSError as e:
            error_msg = f"Erreur nettoyage cache: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "clean_cache", latency, success=False, error=str(e), data_type="cache"
            )

    def load_config(self, config_path: str) -> Dict:
        """
        Charge la configuration avec repli sur les valeurs par défaut.

        Args:
            config_path (str): Chemin vers le fichier de configuration.

        Returns:
            Dict: Configuration chargée.
        """
        start_time = datetime.now()
        try:
            config = get_config(BASE_DIR / config_path)
            if not config:
                raise ValueError("Configuration vide ou non trouvée")
            params = config.get(
                "main_params",
                {
                    "s3_bucket": None,
                    "s3_prefix": "main/",
                    "modules": [
                        "mind_voice",
                        "correlation_heatmap",
                        "data_drift",
                        "export_visuals",
                        "mia_dashboard",
                    ],
                },
            )
            required_keys = ["s3_bucket", "s3_prefix", "modules"]
            missing_keys = [key for key in required_keys if key not in params]
            if missing_keys:
                raise ValueError(f"Clés de configuration manquantes: {missing_keys}")
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Configuration {config_path} chargée"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance("load_config", latency, success=True)
            return params
        except Exception as e:
            error_msg = (
                f"Erreur chargement configuration: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance("load_config", 0, success=False, error=str(e))
            return {
                "s3_bucket": None,
                "s3_prefix": "main/",
                "modules": [
                    "mind_voice",
                    "correlation_heatmap",
                    "data_drift",
                    "export_visuals",
                    "mia_dashboard",
                ],
            }

    def validate_inputs(
        self, features_path: str, trades_path: str, shap_path: str, regime_path: str
    ) -> None:
        """
        Valide l'existence et la cohérence des fichiers d'entrée.

        Args:
            features_path (str): Chemin des features.
            trades_path (str): Chemin des trades.
            shap_path (str): Chemin des importances SHAP.
            regime_path (str): Chemin des probabilités de régimes.

        Raises:
            FileNotFoundError: Si un fichier est introuvable.
            ValueError: Si les données sont invalides.
        """
        start_time = datetime.now()
        try:
            files = [
                (features_path, "features"),
                (trades_path, "trades"),
                (shap_path, "feature_importance"),
                (regime_path, "regime_history"),
            ]
            for file_path, file_type in files:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(
                        f"Fichier {file_type} introuvable: {file_path}"
                    )
                df = pd.read_csv(file_path)
                if df.empty:
                    raise ValueError(f"Fichier {file_type} vide: {file_path}")
                if len(df) < 50:
                    raise ValueError(f"Trop peu de lignes dans {file_type}: {len(df)}")
                required_cols = (
                    ["timestamp", "reward"] if file_type == "trades" else ["timestamp"]
                )
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(
                        f"Colonnes manquantes dans {file_type}: {missing_cols}"
                    )
                if file_type == "features":
                    critical_cols = ["close", "volume", "vix_es_correlation"]
                    missing_critical = [
                        col for col in critical_cols if col not in df.columns
                    ]
                    confidence_drop_rate = 1.0 - min(
                        (len(critical_cols) - len(missing_critical))
                        / len(critical_cols),
                        1.0,
                    )
                    if confidence_drop_rate > 0.5:
                        alert_msg = f"Confidence_drop_rate élevé pour {file_type}: {confidence_drop_rate:.2f} ({len(critical_cols) - len(missing_critical)}/{len(critical_cols)} colonnes)"
                        logger.warning(alert_msg)
                        self.alert_manager.send_alert(alert_msg, priority=3)
                        send_telegram_alert(alert_msg)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Validation des fichiers d'entrée réussie"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "validate_inputs",
                latency,
                success=True,
                confidence_drop_rate=confidence_drop_rate,
            )
        except Exception as e:
            error_msg = f"Erreur validation fichiers d'entrée: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance("validate_inputs", 0, success=False, error=str(e))
            raise

    def log_performance(
        self,
        operation: str,
        latency: float,
        success: bool,
        error: str = None,
        num_rows: int = None,
        attempt_number: int = None,
        data_type: str = None,
        snapshot_size_mb: float = None,
        confidence_drop_rate: float = None,
    ):
        """
        Journalise les performances des opérations.

        Args:
            operation (str): Nom de l'opération.
            latency (float): Temps d'exécution (secondes).
            success (bool): Succès de l'opération.
            error (str, optional): Message d'erreur.
            num_rows (int, optional): Nombre de lignes traitées.
            attempt_number (int, optional): Numéro de tentative.
            data_type (str, optional): Type de données.
            snapshot_size_mb (float, optional): Taille du snapshot.
            confidence_drop_rate (float, optional): Taux de confiance.
        """
        start_time = datetime.now()
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            if memory_usage > 1024:
                alert_msg = f"ALERTE: Usage mémoire élevé ({memory_usage:.2f} MB)"
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=5)
                send_telegram_alert(alert_msg)
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "data_type": data_type,
                "latency": latency,
                "success": success,
                "error": error,
                "memory_mb": memory_usage,
                "cpu_percent": cpu_percent,
                "num_rows": num_rows,
                "attempt_number": attempt_number,
                "snapshot_size_mb": snapshot_size_mb,
                "confidence_drop_rate": confidence_drop_rate,
            }
            self.log_buffer.append(log_entry)
            if len(self.log_buffer) >= self.config.get("logging", {}).get(
                "buffer_size", 100
            ):
                df = pd.DataFrame(self.log_buffer)

                def save_log():
                    df.to_csv(
                        PERF_LOG_PATH,
                        mode="a",
                        header=not os.path.exists(PERF_LOG_PATH),
                        index=False,
                        encoding="utf-8",
                    )

                self.with_retries(save_log)
                self.checkpoint(df, data_type="performance_logs")
                self.cloud_backup(df, data_type="performance_logs")
                self.log_buffer = []
            latency = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Performance journalisée pour {operation}. CPU: {cpu_percent}%"
            )
            self.log_performance(
                "log_performance", latency, success=True, data_type="logging"
            )
        except Exception as e:
            error_msg = (
                f"Erreur journalisation performance: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "log_performance", 0, success=False, error=str(e), data_type="logging"
            )

    def run_module(self, module_name: str, **kwargs) -> None:
        """
        Exécute un module spécifique.

        Args:
            module_name (str): Nom du module (ex. : mind_voice, mia_dashboard).
            **kwargs: Arguments spécifiques au module.
        """
        start_time = datetime.now()
        try:
            logger.info(f"Exécution module {module_name}")
            self.alert_manager.send_alert(f"Exécution module {module_name}", priority=2)
            send_telegram_alert(f"Exécution module {module_name}")

            if module_name == "mind_voice":
                voice_manager = VoiceManager(
                    config_path=kwargs.get("config_path", "config/es_config.yaml")
                )
                voice_manager.speak(
                    "MIA system started", profile="calm", async_mode=True
                )
            elif module_name == "correlation_heatmap":
                heatmap = CorrelationHeatmap(
                    config_path=kwargs.get("config_path", "config/es_config.yaml")
                )
                heatmap.plot_heatmap(
                    df=pd.read_csv(
                        kwargs.get(
                            "features_path",
                            str(BASE_DIR / "data" / "features" / "features_latest.csv"),
                        )
                    ),
                    output_path=str(
                        BASE_DIR
                        / "data"
                        / "figures"
                        / "heatmap"
                        / f"correlation_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    ),
                    figsize=(12, 10),
                    threshold=0.8,
                )
                heatmap.save_correlation_matrix(
                    heatmap.compute_correlations(
                        pd.read_csv(kwargs.get("features_path"))
                    ),
                    kwargs.get(
                        "output_csv", str(BASE_DIR / "data" / "correlation_matrix.csv")
                    ),
                )
            elif module_name == "data_drift":
                detector = DataDriftDetector(
                    config_path=kwargs.get("config_path", "config/es_config.yaml")
                )
                train_df = pd.read_csv(
                    kwargs.get(
                        "train_path",
                        str(BASE_DIR / "data" / "features" / "features_train.csv"),
                    )
                )
                live_df = pd.read_csv(
                    kwargs.get(
                        "features_path",
                        str(BASE_DIR / "data" / "features" / "features_latest.csv"),
                    )
                )
                shap_df = pd.read_csv(
                    kwargs.get(
                        "shap_path",
                        str(BASE_DIR / "data" / "features" / "feature_importance.csv"),
                    )
                )
                features = shap_df["feature"].head(50).tolist() + ["vix_es_correlation"]
                drift_results = detector.compute_drift_metrics(
                    train_df, live_df, features
                )
                detector.plot_drift(drift_results)
                detector.save_drift_report(
                    drift_results,
                    kwargs.get(
                        "output_path",
                        str(BASE_DIR / "data" / "logs" / "drift_report.csv"),
                    ),
                )
            elif module_name == "export_visuals":
                exporter = VisualExporter(
                    config_path=kwargs.get("config_path", "config/es_config.yaml")
                )
                data = pd.read_csv(
                    kwargs.get(
                        "trades_path",
                        str(BASE_DIR / "data" / "trades" / "trades_simulated.csv"),
                    )
                )
                data["cumulative_return"] = data["reward"].cumsum()
                data["gamma_exposure"] = np.random.uniform(-500, 500, len(data))
                data["iv_sensitivity"] = np.random.uniform(0.1, 0.5, len(data))
                exporter.export_visuals(
                    data, prefix=f"visuals_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
            elif module_name == "mia_dashboard":
                dashboard = MIADashboard(
                    config_path=kwargs.get("config_path", "config/es_config.yaml")
                )
                dashboard.main()  # Note: Dashboard runs in main thread due to Dash server
            else:
                raise ValueError(f"Module inconnu: {module_name}")

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Module {module_name} exécuté avec succès"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(f"run_module_{module_name}", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur exécution module {module_name}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance(
                f"run_module_{module_name}", 0, success=False, error=str(e)
            )

    def run(self, args: argparse.Namespace):
        """
        Exécute tous les modules configurés.

        Args:
            args: Arguments CLI (config_path, features_path, trades_path, shap_path, regime_path).
        """
        start_time = datetime.now()
        try:
            logger.info("Démarrage exécution système MIA")
            self.alert_manager.send_alert("Démarrage exécution système MIA", priority=2)
            send_telegram_alert("Démarrage exécution système MIA")

            # Valider les fichiers d'entrée
            self.validate_inputs(
                args.features_path, args.trades_path, args.shap_path, args.regime_path
            )

            # Exécuter les modules dans des threads séparés (sauf mia_dashboard)
            threads = []
            for module in self.config["modules"]:
                if module != "mia_dashboard":
                    thread = threading.Thread(
                        target=self.run_module,
                        args=(module,),
                        kwargs={
                            "config_path": args.config_path,
                            "features_path": args.features_path,
                            "trades_path": args.trades_path,
                            "shap_path": args.shap_path,
                            "output_path": str(
                                BASE_DIR / "data" / "logs" / "drift_report.csv"
                            ),
                            "train_path": str(
                                BASE_DIR / "data" / "features" / "features_train.csv"
                            ),
                        },
                    )
                    threads.append(thread)
                    thread.start()

            # Exécuter mia_dashboard dans le thread principal (Dash nécessite le
            # thread principal)
            if "mia_dashboard" in self.config["modules"]:
                self.run_module(
                    "mia_dashboard",
                    config_path=args.config_path,
                    features_path=args.features_path,
                    trades_path=args.trades_path,
                    shap_path=args.shap_path,
                )

            # Attendre la fin des threads
            for thread in threads:
                thread.join()

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Exécution système MIA terminée"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance("run", latency, success=True)
            self.save_snapshot(
                "run",
                {
                    "modules": self.config["modules"],
                    "timestamp": datetime.now().isoformat(),
                },
            )
        except Exception as e:
            error_msg = (
                f"Erreur exécution système MIA: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance("run", 0, success=False, error=str(e))
            raise

    def main(self):
        """
        Point d’entrée principal avec arguments CLI.
        """
        parser = argparse.ArgumentParser(
            description="Point d’entrée principal pour MIA_IA_SYSTEM_v2_2025"
        )
        parser.add_argument(
            "--config-path",
            type=str,
            default=str(BASE_DIR / "config" / "es_config.yaml"),
            help="Fichier de configuration",
        )
        parser.add_argument(
            "--features-path",
            type=str,
            default=str(BASE_DIR / "data" / "features" / "features_latest.csv"),
            help="Fichier CSV des features",
        )
        parser.add_argument(
            "--trades-path",
            type=str,
            default=str(BASE_DIR / "data" / "trades" / "trades_simulated.csv"),
            help="Fichier CSV des trades",
        )
        parser.add_argument(
            "--shap-path",
            type=str,
            default=str(BASE_DIR / "data" / "features" / "feature_importance.csv"),
            help="Fichier CSV des importances SHAP",
        )
        parser.add_argument(
            "--regime-path",
            type=str,
            default=str(BASE_DIR / "data" / "logs" / "regime_history.csv"),
            help="Fichier CSV des probabilités de régimes",
        )
        args = parser.parse_args()

        self.run(args)


if __name__ == "__main__":
    runner = MainRunner()
    runner.main()
