```python
# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/mind/mind_stream.py
# Module pour gérer le flux cognitif de MIA, analysant les données en temps réel 
# et résumant l'activité cognitive.
#
# Version : 2.1.4
# Date : 2025-05-15
#
# Rôle : Analyse les données en temps réel pour détecter patterns, anomalies, et 
# tendances avec mémoire contextuelle (méthode 7, K-means 10 clusters dans 
# market_memory.db). Utilise IQFeed comme source de données. Collecte les méta-données 
# dans market_memory.db (Proposition 2, Étape 1).
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.8,<6.0.0, 
#   sklearn>=1.2.0,<2.0.0, matplotlib>=3.8.0,<4.0.0, seaborn>=0.13.0,<1.0.0, 
#   pyyaml>=6.0.0,<7.0.0, boto3>=1.26.0,<2.0.0, logging, json, signal, threading, 
#   gzip, traceback, sqlite3
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/db_setup.py
# - src/mind/mind.py
#
# Inputs :
# - config/market_config.yaml
# - config/feature_sets.yaml
# - data/features/features_latest.csv
# - data/logs/mind/mind_stream.json
#
# Outputs :
# - data/logs/mind/mind_stream.log
# - data/logs/mind/mind_stream.json/csv
# - data/logs/mind/mind_stream_performance.csv
# - data/mind/mind_stream_snapshots/*.json.gz
# - data/checkpoints/mind_stream_*.json.gz
# - data/mind/mind_dashboard.json
# - data/figures/mind_stream/
# - market_memory.db (table clusters, meta_runs)
#
# Notes :
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence) 
#   définies dans config/feature_sets.yaml.
# - Utilise IQFeed exclusivement via data_provider.py pour les données d’entrée.
# - Supprime toute référence à 320 features, 81 features, et obs_t.
# - Implémente retries (max 3, délai 2^attempt secondes) pour les opérations 
#   critiques.
# - Logs psutil dans data/logs/mind_stream_performance.csv avec métriques détaillées.
# - Alertes via alert_manager.py pour priorité ≥ 4.
# - Snapshots compressés avec gzip dans data/mind/mind_stream_snapshots/.
# - Sauvegardes incrémentielles toutes les 5 min dans data/checkpoints/ (5 versions).
# - Sauvegardes distribuées toutes les 15 min vers S3 (configurable).
# - Tests unitaires disponibles dans tests/test_mind_stream.py.
# - Phases intégrées : Phase 7 (mémoire contextuelle), Phase 8 (auto-conscience 
#   via confidence_drop_rate), Phase 17 (SHAP).
# - Validation complète prévue pour juin 2025.

# Note : L'erreur E712 concernant les comparaisons avec False est une fausse alerte.
# Aucune comparaison == False n'est utilisée ; is False ou not var sont préférés.

import gzip
import json
import logging
import os
import signal
import sqlite3
import threading
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterator, Optional, Any

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
from sklearn.cluster import KMeans

from src.mind.mind import miya_decides
from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import get_config
from src.model.utils.db_setup import get_db_connection

# Configuration du logging
base_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
log_dir = base_dir / "data" / "logs" / "mind"
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=log_dir / "mind_stream.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Chemins pour les logs et les données du dashboard
LOG_PATH = log_dir / "mind_stream.json"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
DASHBOARD_PATH = base_dir / "data" / "mind" / "mind_dashboard.json"
CSV_LOG_PATH = log_dir / "mind_stream.csv"
PERFORMANCE_LOG_PATH = log_dir / "mind_stream_performance.csv"
SNAPSHOT_DIR = base_dir / "data" / "mind" / "mind_stream_snapshots"
CHECKPOINT_DIR = base_dir / "data" / "checkpoints"
MARKET_MEMORY_DB = base_dir / "data" / "market_memory.db"

# Constantes
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Secondes, exponentiel

# Création des dossiers
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Variable pour gérer l'arrêt propre
RUNNING = True


class MindStream:
    """
    Classe pour gérer le flux cognitif de MIA, analysant les données en temps réel 
    pour détecter des patterns, anomalies et tendances.
    """

    def __init__(self, config_path: str = "config/market_config.yaml"):
        """
        Initialise le gestionnaire de flux cognitif de MIA.

        Args:
            config_path (str): Chemin vers la configuration du marché.
        """
        self.alert_manager = AlertManager()
        self.checkpoint_versions = []
        signal.signal(signal.SIGINT, self.handle_sigint)

        start_time = datetime.now()
        try:
            self.log_buffer = []
            self.analysis_history = []
            self.config = self.load_config(os.path.basename(config_path))

            # Charger les seuils depuis la configuration
            self.thresholds = {
                "vix_threshold": self.config.get("thresholds", {}).get(
                    "vix_threshold", 30
                ),
                "anomaly_score_threshold": self.config.get("thresholds", {}).get(
                    "anomaly_score_threshold", 0.9
                ),
                "trend_confidence_threshold": self.config.get("thresholds", {}).get(
                    "trend_confidence_threshold", 0.7
                ),
                "regime_change_threshold": self.config.get("thresholds", {}).get(
                    "regime_change_threshold", 0.8
                ),
                "max_analysis_frequency": self.config.get("thresholds", {}).get(
                    "max_analysis_frequency", 60
                ),
            }

            # Valider les seuils
            for key, value in self.thresholds.items():
                if not isinstance(value, (int, float)) or value <= 0:
                    raise ValueError(f"Seuil invalide pour {key}: {value}")

            # Chemins de sortie
            self.stream_log_path = CSV_LOG_PATH
            self.dashboard_path = DASHBOARD_PATH
            self.snapshot_dir = SNAPSHOT_DIR
            self.performance_log_path = PERFORMANCE_LOG_PATH
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)

            # État interne
            self.last_analysis_time = datetime.now() - timedelta(
                seconds=self.thresholds["max_analysis_frequency"]
            )

            success_msg = "Flux cognitif initialisé"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=2)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("init_mind_stream", latency, success=True)
            self.save_snapshot(
                "init",
                {
                    "config_path": config_path,
                    "timestamp": datetime.now().isoformat(),
                },
            )
        except Exception as e:
            error_msg = (
                f"Erreur initialisation MindStream : {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=5)
            logger.error(error_msg)
            self.log_performance("init_mind_stream", 0, success=False, error=str(e))
            raise

    def handle_sigint(self, signum, frame):
        """Gère l'arrêt propre sur SIGINT."""
        global RUNNING
        snapshot = {"timestamp": datetime.now().isoformat(), "status": "SIGINT"}
        snapshot_path = (
            self.snapshot_dir / f"stream_sigint_{snapshot['timestamp']}.json.gz"
        )
        try:
            RUNNING = False
            self.save_snapshot("sigint", snapshot)
            success_msg = "Arrêt propre sur SIGINT, snapshot sauvegardé"
            self.alert_manager.send_alert(success_msg, priority=2)
            logger.info(success_msg)
            self.log_performance("handle_sigint", 0, success=True)
            exit(0)
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde SIGINT: {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance("handle_sigint", 0, success=False, error=str(e))
            exit(1)

    def log_performance(
        self,
        operation: str,
        latency: float,
        success: bool,
        error: str = None,
        **kwargs,
    ):
        """
        Enregistre les performances avec psutil dans 
        data/logs/mind_stream_performance.csv.

        Args:
            operation (str): Nom de l’opération.
            latency (float): Temps d’exécution en secondes.
            success (bool): Succès de l’opération.
            error (str, optional): Message d’erreur.
            **kwargs: Paramètres supplémentaires (ex. : num_analyses, 
            snapshot_size_mb).
        """
        start_time = datetime.now()
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
            cpu_usage = psutil.cpu_percent()
            if memory_usage > 1024:
                alert_msg = f"ALERTE: Usage mémoire élevé ({memory_usage:.2f} MB)"
                self.alert_manager.send_alert(alert_msg, priority=5)
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "success": success,
                "latency": latency,
                "error": error,
                "cpu_percent": cpu_usage,
                "memory_mb": memory_usage,
                **kwargs,
            }
            self.log_buffer.append(log_entry)
            buffer_size = self.config.get("logging", {}).get("buffer_size", 100)
            if len(self.log_buffer) >= buffer_size:
                log_df = pd.DataFrame(self.log_buffer)
                mode = "a" if self.performance_log_path.exists() else "w"

                def save_log():
                    log_df.to_csv(
                        self.performance_log_path,
                        mode=mode,
                        index=False,
                        header=not self.performance_log_path.exists(),
                        encoding="utf-8",
                    )

                self.with_retries(save_log)
                self.log_buffer = []
            latency = (datetime.now() - start_time).total_seconds()
            logger.info(f"Performance journalisée pour {operation}. CPU: {cpu_usage}%")
        except Exception as e:
            error_msg = (
                f"Erreur journalisation performance: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)

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
            snapshot_path = (
                self.snapshot_dir / f"snapshot_{snapshot_type}_{timestamp}.json"
            )
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)

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
                self.alert_manager.send_alert(alert_msg, priority=3)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Snapshot {snapshot_type} sauvegardé: {save_path}"
            self.alert_manager.send_alert(success_msg, priority=1)
            logger.info(success_msg)
            self.log_performance(
                "save_snapshot", latency, success=True, snapshot_size_mb=file_size
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance("save_snapshot", 0, success=False, error=str(e))

    def checkpoint(self, data: pd.DataFrame) -> None:
        """
        Sauvegarde incrémentielle des analyses toutes les 5 minutes avec versionnage 
        (5 versions).

        Args:
            data (pd.DataFrame): Données des analyses à sauvegarder.
        """
        try:
            start_time = datetime.now()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_data = {
                "timestamp": timestamp,
                "num_rows": len(data),
                "columns": list(data.columns),
            }
            checkpoint_path = CHECKPOINT_DIR / f"mind_stream_{timestamp}.json.gz"
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

            def write_checkpoint():
                with gzip.open(checkpoint_path, "wt", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, indent=4)
                csv_path = checkpoint_path.with_suffix(".csv")
                data.to_csv(csv_path, index=False, encoding="utf-8")

            self.with_retries(write_checkpoint)
            self.checkpoint_versions.append(checkpoint_path)
            if len(self.checkpoint_versions) > 5:
                oldest = self.checkpoint_versions.pop(0)
                if oldest.exists():
                    oldest.unlink()
                csv_oldest = oldest.with_suffix(".csv")
                if csv_oldest.exists():
                    csv_oldest.unlink()
            file_size = os.path.getsize(checkpoint_path) / 1024 / 1024
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Checkpoint sauvegardé: {checkpoint_path}"
            self.alert_manager.send_alert(success_msg, priority=1)
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
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance("checkpoint", 0, success=False, error=str(e))

    def cloud_backup(self, data: pd.DataFrame) -> None:
        """
        Sauvegarde distribuée des analyses vers AWS S3 toutes les 15 minutes 
        (configurable).

        Args:
            data (pd.DataFrame): Données des analyses à sauvegarder.
        """
        try:
            start_time = datetime.now()
            if not self.config.get("s3_bucket"):
                warning_msg = "S3 bucket non configuré, sauvegarde cloud ignorée"
                self.alert_manager.send_alert(warning_msg, priority=3)
                logger.warning(warning_msg)
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.config['s3_prefix']}mind_stream_{timestamp}.csv.gz"
            temp_path = CHECKPOINT_DIR / f"temp_s3_{timestamp}.csv.gz"

            def write_temp():
                data.to_csv(
                    temp_path, compression="gzip", index=False, encoding="utf-8"
                )

            self.with_retries(write_temp)
            s3_client = boto3.client("s3")

            def upload_s3():
                s3_client.upload_file(str(temp_path), self.config["s3_bucket"], backup_path)

            self.with_retries(upload_s3)
            temp_path.unlink()
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Sauvegarde cloud S3 effectuée: {backup_path}"
            self.alert_manager.send_alert(success_msg, priority=1)
            logger.info(success_msg)
            self.log_performance(
                "cloud_backup", latency, success=True, num_rows=len(data)
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde cloud S3: {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance("cloud_backup", 0, success=False, error=str(e))

    def with_retries(
        self,
        func: callable,
        max_attempts: int = MAX_RETRIES,
        delay_base: float = RETRY_DELAY_BASE,
    ) -> Optional[Any]:
        """
        Exécute une fonction avec retries (max 3, délai exponentiel).

        Args:
            func (callable): Fonction à exécuter.
            max_attempts (int): Nombre maximum de tentatives.
            delay_base (float): Base pour le délai exponentiel (premier délai = 2s).

        Returns:
            Optional[Any]: Résultat de la fonction ou None si échec.
        """
        start_time = datetime.now()
        for attempt in range(1, max_attempts + 1):
            try:
                result = func()
                latency = (datetime.now() - start_time).total_seconds()
                self.log_performance(
                    f"retry_attempt_{attempt}",
                    latency,
                    success=True,
                    attempt_number=attempt,
                )
                return result
            except Exception as e:
                if attempt == max_attempts:
                    error_msg = (
                        f"Échec après {max_attempts} tentatives: {str(e)}\n"
                        f"{traceback.format_exc()}"
                    )
                    self.alert_manager.send_alert(error_msg, priority=4)
                    logger.error(error_msg)
                    self.log_performance(
                        f"retry_attempt_{attempt}",
                        0,
                        success=False,
                        error=str(e),
                        attempt_number=attempt,
                    )
                    return None
                delay = delay_base ** (attempt - 1)
                warning_msg = (
                    f"Tentative {attempt} échouée, retry après {delay}s"
                )
                self.alert_manager.send_alert(warning_msg, priority=3)
                logger.warning(warning_msg)
                time.sleep(delay)

    def load_config(self, config_path: str) -> Dict:
        """
        Charge la configuration avec repli sur les valeurs par défaut.

        Args:
            config_path (str): Nom du fichier de configuration.

        Returns:
            Dict: Configuration chargée.
        """
        start_time = datetime.now()
        try:
            config = get_config(base_dir / config_path)
            if not config:
                raise ValueError("Configuration vide ou non trouvée")
            required_keys = ["thresholds", "logging", "s3_bucket", "s3_prefix"]
            missing_keys = [key for key in required_keys if key not in config]
            if missing_keys:
                error_msg = f"Clés de configuration manquantes: {missing_keys}"
                raise ValueError(error_msg)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Configuration {config_path} chargée"
            self.alert_manager.send_alert(success_msg, priority=1)
            logger.info(success_msg)
            self.log_performance("load_config", latency, success=True)
            return config
        except Exception as e:
            error_msg = (
                f"Erreur chargement configuration : {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            logger.error(error_msg)
            self.log_performance("load_config", 0, success=False, error=str(e))
            return {
                "thresholds": {},
                "logging": {"buffer_size": 100},
                "s3_bucket": None,
                "s3_prefix": "mind_stream/",
            }

    def load_logs(self, max_logs: int = 1000) -> Iterator[Dict]:
        """
        Charge les logs JSON de la session avec gestion des erreurs, utilisant 
        un générateur.

        Args:
            max_logs (int): Nombre maximum de logs à charger pour optimiser la 
            performance.

        Yields:
            Dict: Log chargé.
        """
        start_time = datetime.now()
        try:
            if not LOG_PATH.exists():
                error_msg = "Aucun log JSON trouvé"
                self.alert_manager.send_alert(error_msg, priority=3)
                return

            def load():
                with open(LOG_PATH, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i >= max_logs:
                            break
                        if line.strip():
                            try:
                                yield json.loads(line)
                            except json.JSONDecodeError:
                                self.alert_manager.send_alert(
                                    "Log JSON corrompu ignoré", priority=2
                                )
                                continue

            for log in self.with_retries(load) or []:
                yield log
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("load_logs", latency, success=True)
        except Exception as e:
            error_msg = (
                f"Erreur chargement logs : {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            logger.error(error_msg)
            self.log_performance("load_logs", 0, success=False, error=str(e))

    def validate_data(self, data: pd.DataFrame) -> None:
        """
        Valide les données d’entrée (350/150 features) avec une validation intelligente.

        Args:
            data (pd.DataFrame): Données à valider.

        Raises:
            ValueError: Si les données sont invalides.
        """
        start_time = datetime.now()
        try:
            num_cols = len(data.columns)
            if num_cols >= 350:
                feature_set = "training"
                expected_features = 350
            elif num_cols >= 150:
                feature_set = "inference"
                expected_features = 150
            else:
                error_msg = f"Nombre de features trop bas: {num_cols} < 150"
                raise ValueError(error_msg)

            if num_cols != expected_features:
                error_msg = (
                    f"Nombre de features incorrect: {num_cols} != "
                    f"{expected_features} pour ensemble {feature_set}"
                )
                raise ValueError(error_msg)

            # Validation des features SHAP dans config/feature_sets.yaml
            feature_sets_path = base_dir / "config" / "feature_sets.yaml"
            if feature_sets_path.exists() and feature_set == "inference":
                feature_config = get_config(feature_sets_path)
                shap_features = feature_config.get("inference", {}).get(
                    "shap_features", []
                )
                if len(shap_features) != 150:
                    error_msg = (
                        f"Nombre de SHAP features incorrect dans feature_sets.yaml: "
                        f"{len(shap_features)} != 150"
                    )
                    raise ValueError(error_msg)
                missing_features = [
                    col for col in shap_features if col not in data.columns
                ]
                if missing_features:
                    error_msg = (
                        f"Features SHAP manquantes dans les données: "
                        f"{missing_features[:5]}"
                    )
                    raise ValueError(error_msg)

            critical_cols = [
                "vix",
                "neural_regime",
                "predicted_volatility",
                "trade_frequency_1s",
                "close",
                "rsi_14",
            ]
            for col in critical_cols:
                if col in data.columns:
                    if data[col].isnull().any():
                        raise ValueError(f"Colonne {col} contient des NaN")
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        error_msg = (
                            f"Colonne {col} n'est pas numérique: "
                            f"{data[col].dtype}"
                        )
                        raise ValueError(error_msg)
            if "timestamp" in data.columns:
                latest_timestamp = pd.to_datetime(
                    data["timestamp"].iloc[-1], errors="coerce"
                )
                if pd.isna(latest_timestamp):
                    raise ValueError("Timestamp non valide")
                if latest_timestamp > datetime.now() + timedelta(minutes=5) or (
                    latest_timestamp < datetime.now() - timedelta(hours=24)
                ):
                    error_msg = f"Timestamp hors plage: {latest_timestamp}"
                    raise ValueError(error_msg)

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = (
                f"Données validées pour ensemble {feature_set} "
                f"({expected_features} features)"
            )
            self.alert_manager.send_alert(success_msg, priority=1)
            logger.info(success_msg)
            self.log_performance("validate_data", latency, success=True)
        except Exception as e:
            error_msg = (
                f"Erreur validation données : {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            logger.error(error_msg)
            self.log_performance("validate_data", 0, success=False, error=str(e))
            raise

    def store_analysis_pattern(self, analysis_result: Dict, data: pd.DataFrame) -> Dict:
        """
        Stocke un pattern d’analyse dans market_memory.db avec clusterisation 
        K-means (méthode 7).

        Args:
            analysis_result (Dict): Résultats de l’analyse.
            data (pd.DataFrame): Données contextuelles.

        Returns:
            Dict: Résultat avec cluster_id.
        """
        start_time = datetime.now()
        try:
            features = {
                "anomaly_score": analysis_result.get("anomaly_score", 0),
                "trend_score": analysis_result.get("trend_score", 0),
                "regime_change_score": analysis_result.get(
                    "regime_change_score", 0
                ),
            }
            required_cols = ["anomaly_score", "trend_score", "regime_change_score"]
            missing_cols = [col for col in required_cols if col not in features]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols),
                1.0,
            )
            if confidence_drop_rate > 0.5:
                alert_msg = (
                    f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} "
                    f"({len(features)} colonnes)"
                )
                self.alert_manager.send_alert(alert_msg, priority=3)

            df = pd.DataFrame([features])
            X = (
                df[["anomaly_score", "trend_score", "regime_change_score"]]
                .fillna(0)
                .values
            )

            if len(X) < 10:
                analysis_result["cluster_id"] = 0
            else:

                def run_kmeans():
                    kmeans = KMeans(n_clusters=10, random_state=42)
                    return kmeans.fit_predict(X)[0]

                analysis_result["cluster_id"] = self.with_retries(run_kmeans)
                if analysis_result["cluster_id"] is None:
                    raise ValueError("Échec clusterisation K-means")

            def store_clusters():
                db_path = base_dir / "data" / "market_memory.db"
                conn = get_db_connection(str(db_path))
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS clusters (
                        cluster_id INTEGER,
                        event_type TEXT,
                        features TEXT,
                        timestamp TEXT
                    )
                    """
                )
                features_json = json.dumps(features)
                cursor.execute(
                    """
                    INSERT INTO clusters (cluster_id, event_type, features, timestamp)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        analysis_result["cluster_id"],
                        "stream_analysis",
                        features_json,
                        analysis_result["timestamp"],
                    ),
                )
                conn.commit()
                conn.close()

            self.with_retries(store_clusters)

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = (
                f"Pattern analyse stocké, cluster_id="
                f"{analysis_result['cluster_id']}"
            )
            self.alert_manager.send_alert(success_msg, priority=1)
            logger.info(success_msg)
            self.log_performance(
                "store_analysis_pattern",
                latency,
                success=True,
                confidence_drop_rate=confidence_drop_rate,
            )
            self.save_snapshot(
                "store_analysis_pattern",
                {
                    "cluster_id": analysis_result["cluster_id"],
                    "features": features,
                },
            )
            return analysis_result
        except Exception as e:
            error_msg = (
                f"Erreur stockage pattern analyse : {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance(
                "store_analysis_pattern", 0, success=False, error=str(e)
            )
            analysis_result["cluster_id"] = 0
            return analysis_result

    def store_stream_metadata(self, analysis_result: Dict, data: pd.DataFrame, context: Dict) -> None:
        """Stocke les méta-données de l'analyse dans market_memory.db (Proposition 2, Étape 1)."""
        try:
            start_time = datetime.now()
            metadata = {
                "timestamp": analysis_result.get("timestamp", datetime.now().isoformat()),
                "trade_id": 0,  # Pas de trade_id spécifique pour l'analyse en flux
                "metrics": {
                    "anomaly_score": analysis_result.get("anomaly_score", 0.0),
                    "trend_score": analysis_result.get("trend_score", 0.0),
                    "trend_type": analysis_result.get("trend_type", "unknown"),
                    "regime_change_score": analysis_result.get("regime_change_score", 0.0),
                    "regime_type": analysis_result.get("regime_type", "unknown"),
                    "vix": analysis_result.get("vix", 0.0),
                    "rsi": analysis_result.get("rsi", 0.0),
                    "trade_frequency": analysis_result.get("trade_frequency", 0.0),
                    "cluster_id": analysis_result.get("cluster_id", 0),
                },
                "hyperparameters": self.config.get("model_params", {}),
                "performance": analysis_result.get("anomaly_score", 0.0),  # Utiliser anomaly_score comme indicateur
                "regime": analysis_result.get("regime_type", "unknown"),
                "session": data.get("session", "unknown").iloc[-1] if "session" in data.columns else "unknown",
                "shap_metrics": {},  # Pas de SHAP dans stream_analysis
                "context": context,
            }
            conn = sqlite3.connect(MARKET_MEMORY_DB)
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS meta_runs (
                    run_id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    trade_id INTEGER,
                    metrics TEXT,
                    hyperparameters TEXT,
                    performance REAL,
                    regime TEXT,
                    session TEXT,
                    shap_metrics TEXT,
                    context TEXT
                )
                """
            )
            cursor.execute(
                """
                INSERT INTO meta_runs (timestamp, trade_id, metrics, hyperparameters, performance, regime, session, shap_metrics, context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata["timestamp"],
                    metadata["trade_id"],
                    json.dumps(metadata["metrics"]),
                    json.dumps(metadata["hyperparameters"]),
                    metadata["performance"],
                    metadata["regime"],
                    metadata["session"],
                    json.dumps(metadata["shap_metrics"]),
                    json.dumps(metadata["context"]),
                ),
            )
            conn.commit()
            conn.close()
            logger.info(f"Méta-données analyse {metadata['timestamp']} stockées dans market_memory.db")
            self.alert_manager.send_alert(
                f"Méta-données analyse {metadata['timestamp']} stockées dans market_memory.db", priority=1
            )
            self.log_performance("store_stream_metadata", time.time() - start_time, success=True)
        except Exception as e:
            error_msg = f"Erreur stockage méta-données analyse {analysis_result.get('timestamp', 'unknown')}: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance("store_stream_metadata", time.time() - start_time, success=False, error=str(e))

    def visualize_analysis_patterns(self):
        """
        Génère une heatmap des clusters d’analyse dans data/figures/mind_stream/.
        """
        start_time = datetime.now()
        try:
            analyses = self.analysis_history[-100:]
            if not analyses:
                error_msg = "Aucune analyse pour visualisation"
                self.alert_manager.send_alert(error_msg, priority=3)
                return

            df = pd.DataFrame(analyses)
            if "cluster_id" not in df.columns or df["cluster_id"].isnull().all():
                error_msg = "Aucun cluster_id pour visualisation"
                self.alert_manager.send_alert(error_msg, priority=3)
                return

            pivot = df.pivot_table(
                index="cluster_id",
                columns="anomaly_score",
                values="timestamp",
                aggfunc="count",
                fill_value=0,
            )
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot, annot=True, cmap="Blues")
            plt.title("Heatmap des Clusters d’Analyse")
            figures_dir = base_dir / "data" / "figures" / "mind_stream"
            figures_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d")
            output_path = figures_dir / f"analysis_clusters_{timestamp}.png"

            def save_fig():
                plt.savefig(output_path)
                plt.close()

            self.with_retries(save_fig)

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Heatmap des clusters d’analyse générée"
            self.alert_manager.send_alert(success_msg, priority=1)
            logger.info(success_msg)
            self.log_performance("visualize_analysis_patterns", latency, success=True)
        except Exception as e:
            error_msg = (
                f"Erreur visualisation clusters : {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance(
                "visualize_analysis_patterns", 0, success=False, error=str(e)
            )

    def stream_analysis(
        self, data: pd.DataFrame, context: Optional[Dict] = None
    ) -> Dict:
        """
        Analyse les données en temps réel pour détecter des patterns, anomalies, 
        et tendances.

        Args:
            data (pd.DataFrame): Données contenant les features (350/150).
            context (Optional[Dict]): Contexte cognitif 
            (ex. : neural_regime, strategy_params).

        Returns:
            Dict: Résultats de l'analyse (anomalies, tendances, régime).
        """
        start_time = datetime.now()
        try:
            self.validate_data(data)
            context = context or {}
            current_data = data.iloc[-1]
            timestamp = datetime.now()

            max_analysis_freq = self.thresholds["max_analysis_frequency"]
            if (timestamp - self.last_analysis_time).total_seconds() < max_analysis_freq:
                warning_msg = "Analyse trop fréquente, ignorée"
                self.alert_manager.send_alert(warning_msg, priority=2)
                return {}
            self.last_analysis_time = timestamp

            vix = current_data.get("vix", 0)
            trade_frequency = current_data.get("trade_frequency_1s", 0)
            anomaly_score = (
                0.6 * (vix / self.thresholds["vix_threshold"])
                + 0.4 * (trade_frequency / 10.0)
            )
            anomaly_score = min(1.0, max(0.0, anomaly_score))

            rsi = current_data.get("rsi_14", 50)
            price = current_data.get("close", 0)
            prev_price = data["close"].iloc[-2] if len(data) > 1 else price
            delta_price = (
                (price - prev_price) / prev_price if prev_price != 0 else 0
            )
            trend_score = (
                0.5 * (1 if rsi > 70 else -1 if rsi < 30 else 0)
                + 0.5 * (delta_price / 0.01)
            )
            trend_score = min(1.0, max(-1.0, trend_score))
            trend_confidence = self.thresholds["trend_confidence_threshold"]
            trend_type = (
                "haussière"
                if trend_score > trend_confidence
                else (
                    "baissière"
                    if trend_score < -trend_confidence
                    else "range"
                )
            )

            neural_regime = current_data.get("neural_regime", 0)
            regime_change_score = (
                abs(trend_score)
                if neural_regime != context.get("neural_regime", neural_regime)
                else 0.0
            )
            regime_type = (
                "trend"
                if neural_regime == 0
                else "range" if neural_regime == 1 else "defensive"
            )

            analysis_result = {
                "timestamp": datetime.now().isoformat(),
                "anomaly_score": anomaly_score,
                "trend_score": trend_score,
                "trend_type": trend_type,
                "regime_change_score": regime_change_score,
                "regime_type": regime_type,
                "vix": vix,
                "rsi": rsi,
                "trade_frequency": trade_frequency,
                "context": context,
                "cluster_id": None,
            }

            anomaly_threshold = self.thresholds["anomaly_score_threshold"]
            if anomaly_score > anomaly_threshold:
                alert_msg = (
                    f"Anomalie détectée : score={anomaly_score:.2f}, "
                    f"VIX={vix:.2f}, Fréquence={trade_frequency:.2f}"
                )
                self.alert_manager.send_alert(alert_msg, priority=4)
            if abs(trend_score) > trend_confidence:
                alert_msg = (
                    f"Tendance détectée : {trend_type} "
                    f"(score={trend_score:.2f}, RSI={rsi:.2f})"
                )
                self.alert_manager.send_alert(alert_msg, priority=3)
            regime_threshold = self.thresholds["regime_change_threshold"]
            if regime_change_score > regime_threshold:
                decision_msg = (
                    f"Changement de régime vers {regime_type} "
                    f"(score={regime_change_score:.2f})"
                )
                miya_decides(decision_msg, priority=4)

            def log_analysis():
                analysis_result = self.store_analysis_pattern(analysis_result, data)
                log_entry = {
                    "timestamp": analysis_result["timestamp"],
                    "anomaly_score": analysis_result["anomaly_score"],
                    "trend_score": analysis_result["trend_score"],
                    "trend_type": analysis_result["trend_type"],
                    "regime_change_score": analysis_result["regime_change_score"],
                    "regime_type": analysis_result["regime_type"],
                    "vix": analysis_result["vix"],
                    "rsi": analysis_result["rsi"],
                    "trade_frequency": analysis_result["trade_frequency"],
                    "cluster_id": analysis_result["cluster_id"],
                    "context_neural_regime": analysis_result["context"].get(
                        "neural_regime", None
                    ),
                }
                with open(LOG_PATH, "a", encoding="utf-8") as f:
                    json.dump(log_entry, f)
                    f.write("\n")

                self.log_buffer.append(log_entry)
                buffer_size = self.config.get("logging", {}).get(
                    "buffer_size", 100
                )
                if len(self.log_buffer) >= buffer_size:
                    log_df = pd.DataFrame(self.log_buffer)
                    mode = "a" if self.stream_log_path.exists() else "w"
                    log_df.to_csv(
                        self.stream_log_path,
                        mode=mode,
                        index=False,
                        header=not self.stream_log_path.exists(),
                        encoding="utf-8",
                    )
                    self.checkpoint(log_df)
                    self.cloud_backup(log_df)
                    self.log_buffer = []

            self.with_retries(log_analysis)

            # Stocker les méta-données de l'analyse
            self.store_stream_metadata(analysis_result, data, context)

            self.save_analysis_snapshot(
                len(self.analysis_history),
                pd.Timestamp.now(),
                analysis_result,
                data,
            )
            self.miya_dashboard_update(analysis_data=analysis_result)
            self.analysis_history.append(analysis_result)
            threading.Thread(
                target=self.visualize_analysis_patterns, daemon=True
            ).start()

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = (
                f"Analyse en temps réel effectuée: anomaly_score="
                f"{anomaly_score:.2f}, trend_type={trend_type}"
            )
            self.alert_manager.send_alert(success_msg, priority=1)
            logger.info(success_msg)
            self.log_performance(
                "stream_analysis",
                latency,
                success=True,
                num_analyses=len(self.analysis_history),
            )
            return analysis_result
        except Exception as e:
            error_msg = (
                f"Erreur stream_analysis : {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=5)
            logger.error(error_msg)
            self.log_performance("stream_analysis", 0, success=False, error=str(e))
            return {}

    def save_analysis_snapshot(
        self,
        step: int,
        timestamp: pd.Timestamp,
        result: Dict,
        data: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Sauvegarde un instantané des résultats d'analyse.

        Args:
            step (int): Étape de l'analyse.
            timestamp (pd.Timestamp): Horodatage.
            result (Dict): Résultats de l'analyse.
            data (Optional[pd.DataFrame]): Données contextuelles.
        """
        start_time = datetime.now()
        try:
            snapshot = {
                "step": step,
                "timestamp": timestamp.isoformat(),
                "result": result,
                "features": (
                    data.to_dict(orient="records")[0] if data is not None else None
                ),
                "analysis_history": self.analysis_history[-10:],
                "performance": {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_mb": (
                        psutil.Process().memory_info().rss / 1024 / 1024
                    ),
                },
            }
            self.save_snapshot(f"analysis_step_{step:04d}", snapshot, compress=True)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Snapshot analyse step {step} sauvegardé"
            self.alert_manager.send_alert(success_msg, priority=2)
            logger.info(success_msg)
            self.log_performance("save_analysis_snapshot", latency, success=True)
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde snapshot analyse : {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance(
                "save_analysis_snapshot", 0, success=False, error=str(e)
            )

    def miya_brain_scan(self):
        """
        Analyse les patterns cognitifs récents et met à jour le dashboard.
        """
        start_time = datetime.now()
        try:
            from src.mind.mind import (
                miya_detect_patterns,
                miya_error_classifier,
                miya_narrative_builder,
                miya_summary,
            )

            alert_msg = "Scan cognitif MIA en cours..."
            self.alert_manager.send_alert(alert_msg, priority=3)
            miya_summary()
            miya_error_classifier()
            miya_detect_patterns()
            miya_narrative_builder()
            self.miya_dashboard_update()
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Scan cognitif terminé"
            self.alert_manager.send_alert(success_msg, priority=1)
            logger.info(success_msg)
            self.log_performance("miya_brain_scan", latency, success=True)
        except Exception as e:
            error_msg = (
                f"Erreur brain scan : {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=5)
            logger.error(error_msg)
            self.log_performance("miya_brain_scan", 0, success=False, error=str(e))

    def miya_report_all(self, env=None):
        """
        Génère un rapport complet sur l'état de MIA et du système.

        Args:
            env: Environnement de trading (optionnel).
        """
        start_time = datetime.now()
        try:
            from src.mind.mind import (
                miya_context,
                miya_health_check,
                miya_narrative_builder,
                miya_summary,
            )

            alert_msg = "Génération rapport global"
            self.alert_manager.send_alert(alert_msg, priority=3)
            miya_context(env)
            miya_summary()
            miya_health_check()
            miya_narrative_builder()
            self.miya_dashboard_update()
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Rapport global généré"
            self.alert_manager.send_alert(success_msg, priority=1)
            logger.info(success_msg)
            self.log_performance("miya_report_all", latency, success=True)
        except Exception as e:
            error_msg = (
                f"Erreur rapport global : {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=5)
            logger.error(error_msg)
            self.log_performance("miya_report_all", 0, success=False, error=str(e))

    def miya_mood(self) -> str:
        """
        Évalue l’état d’esprit de MIA basé sur les 20 derniers logs.

        Returns:
            str: Humeur de MIA (ex. : "Confiant", "Stable").
        """
        start_time = datetime.now()
        try:
            logs = list(self.load_logs(max_logs=20))
            if not logs:
                mood = "⚪ Neutre"
            else:
                score = sum(
                    (
                        1
                        if log["level"] == "info"
                        else -2 if log["level"] in ["warning", "error"] else 0
                    )
                    for log in logs
                )
                if score > 8:
                    mood = "✨ Confiant"
                elif score > 0:
                    mood = "🙂 Stable"
                elif score > -8:
                    mood = "😕 Sous pression"
                else:
                    mood = "⚠️ Fragile"
            alert_msg = f"Humeur MIA : {mood}"
            self.alert_manager.send_alert(alert_msg, priority=2)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("miya_mood", latency, success=True)
            return mood
        except Exception as e:
            error_msg = (
                f"Erreur évaluation humeur : {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance("miya_mood", 0, success=False, error=str(e))
            return "⚪ Neutre"

    def miya_check_regime(self, regime: str):
        """
        Rappelle les décisions passées dans un régime donné (trend, range, defensive).

        Args:
            regime (str): Régime à analyser.
        """
        start_time = datetime.now()
        try:
            valid_regimes = ["trend", "range", "defensive"]
            if regime not in valid_regimes:
                error_msg = (
                    f"Régime invalide : {regime}, attendu : {valid_regimes}"
                )
                self.alert_manager.send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return
            logs = list(self.load_logs(max_logs=500))
            related = [
                log
                for log in logs
                if log.get("tag") == "DECIDE"
                and regime in log.get("message", "").lower()
            ]
            alert_msg = f"{len(related)} décisions en mode {regime}"
            self.alert_manager.send_alert(alert_msg, priority=2)
            for log in related[-3:]:
                self.alert_manager.send_alert(
                    f"- {log['timestamp']}: {log['message']}", priority=1
                )
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("miya_check_regime", latency, success=True)
        except Exception as e:
            error_msg = (
                f"Erreur vérification régime : {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance("miya_check_regime", 0, success=False, error=str(e))

    def miya_mental_rewind(self, n: int = 5):
        """
        Remonte l’historique mental de MIA (dernier n logs).

        Args:
            n (int): Nombre de logs à afficher.
        """
        start_time = datetime.now()
        try:
            logs = list(self.load_logs(max_logs=n))
            if not logs:
                error_msg = "Aucun log disponible pour rewind"
                self.alert_manager.send_alert(error_msg, priority=2)
                return
            for log in logs[-n:]:
                self.alert_manager.send_alert(
                    f"[{log['tag']}] {log['message']}", priority=2
                )
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Rewind mental effectué: {n} logs"
            self.alert_manager.send_alert(success_msg, priority=1)
            logger.info(success_msg)
            self.log_performance("miya_mental_rewind", latency, success=True)
        except Exception as e:
            error_msg = (
                f"Erreur rewind mental : {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance("miya_mental_rewind", 0, success=False, error=str(e))

    def miya_think_market_snapshot(self, data: Dict[str, float], env=None):
        """
        Analyse un instantané du marché et met à jour le dashboard.

        Args:
            data (Dict[str, float]): Données du marché (ex. : RSI, GEX).
            env: Environnement de trading (optionnel).
        """
        start_time = datetime.now()
        try:
            from src.mind.mind import miya_speaks_markets

            rsi = data.get("rsi", "N/A")
            gex = data.get("gex", "N/A")
            message = f"Snapshot marché : RSI={rsi}, GEX={gex}"
            self.alert_manager.send_alert(message, priority=2)
            miya_speaks_markets(data, priority=3, env=env)
            self.miya_dashboard_update(market_data={"rsi": rsi, "gex": gex})
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Snapshot marché analysé"
            self.alert_manager.send_alert(success_msg, priority=1)
            logger.info(success_msg)
            self.log_performance(
                "miya_think_market_snapshot", latency, success=True
            )
        except Exception as e:
            error_msg = (
                f"Erreur snapshot marché : {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance(
                "miya_think_market_snapshot", 0, success=False, error=str(e)
            )

    def miya_status(self) -> Dict:
        """
        Fournit un aperçu rapide de l'état de MIA pour le dashboard.

        Returns:
            Dict: Statut actuel (humeur, erreurs récentes, régime).
        """
        start_time = datetime.now()
        try:
            logs = list(self.load_logs(max_logs=50))
            errors = len(
                [log for log in logs if log["level"] in ["warning", "error"]]
            )
            last_regime = next(
                (log["regime"] for log in reversed(logs) if log.get("regime")),
                "Inconnu",
            )
            status = {
                "mood": self.miya_mood(),
                "recent_errors": errors,
                "last_regime": last_regime,
                "timestamp": datetime.now().isoformat(),
                "performance": {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_mb": (
                        psutil.Process().memory_info().rss / 1024 / 1024
                    ),
                },
            }
            alert_msg = (
                f"Statut : Humeur={status['mood']}, Erreurs={errors}, "
                f"Régime={last_regime}"
            )
            self.alert_manager.send_alert(alert_msg, priority=2)
            self.save_snapshot("status", status)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("miya_status", latency, success=True)
            return status
        except Exception as e:
            error_msg = (
                f"Erreur statut : {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance("miya_status", 0, success=False, error=str(e))
            return {}

    def miya_dashboard_update(
        self,
        market_data: Optional[Dict] = None,
        analysis_data: Optional[Dict] = None,
    ):
        """
        Met à jour un fichier JSON pour partager l'état avec mia_dashboard.py.

        Args:
            market_data (Dict, optional): Données du marché à inclure.
            analysis_data (Dict, optional): Données d'analyse à inclure.
        """
        start_time = datetime.now()
        try:
            status = self.miya_status()
            if market_data:
                status["market"] = market_data
            if analysis_data:
                status["analysis"] = analysis_data

            def save_dashboard():
                with open(self.dashboard_path, "w", encoding="utf-8") as f:
                    json.dump(status, f, indent=4)

            self.with_retries(save_dashboard)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Mise à jour dashboard JSON effectuée"
            self.alert_manager.send_alert(success_msg, priority=1)
            logger.info(success_msg)
            self.log_performance("miya_dashboard_update", latency, success=True)
        except Exception as e:
            error_msg = (
                f"Erreur mise à jour dashboard : {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            logger.error(error_msg)
            self.log_performance(
                "miya_dashboard_update", 0, success=False, error=str(e)
            )


if __name__ == "__main__":
    stream = MindStream()
    stream.miya_brain_scan()
    print("État MIA :", stream.miya_mood())
    stream.miya_mental_rewind(n=3)
    stream.miya_check_regime("trend")
    stream.miya_think_market_snapshot({"rsi": 75, "gex": 100000})
    data = pd.DataFrame(
        {
            "timestamp": [datetime.now()],
            "vix": [20.0],
            "neural_regime": [0],
            "predicted_volatility": [1.5],
            "trade_frequency_1s": [8.0],
            "close": [5100.0],
            "rsi_14": [75.0],
            **{f"feature_{i}": [np.random.uniform(0, 1)] for i in range(143)},
        }
    )
    context = {"neural_regime": 0, "strategy_params": {"entry_threshold": 70.0}}
    result = stream.stream_analysis(data, context)
    print("Résultat analyse :", result)
