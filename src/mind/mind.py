# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/mind/mind.py
# Moteur cognitif central de MIA pour logs, communication, et analyse, optimisé pour MIA_IA_SYSTEM_v2_2025.
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Gère les logs, la communication (console, vocal, alertes, Telegram), et l’analyse des événements avec mémoire contextuelle
#        (méthode 7, K-means 10 clusters dans market_memory.db). Utilise IQFeed comme source de données principale.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, sklearn>=1.2.0,<2.0.0, psutil>=5.9.8,<6.0.0, matplotlib>=3.8.0,<4.0.0, seaborn>=0.13.0,<1.0.0,
#   pyyaml>=6.0.0,<7.0.0, boto3>=1.26.0,<2.0.0, logging, threading, queue, hashlib, traceback, json, csv, os, signal
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/db_setup.py
# - src/envs/trading_env.py
# - src/mind/mind_voice.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/es_config.yaml
# - data/features/features_latest.csv
# - config/feature_sets.yaml
#
# Outputs :
# - data/logs/mind_stream.log
# - data/logs/mind_stream.json/csv
# - data/logs/mind_performance.csv
# - data/logs/brain_state.json
# - data/logs/mind_snapshots/*.json.gz
# - data/checkpoints/mind_*.json.gz
# - data/logs/mind_dashboard.json
# - data/figures/mind/
# - market_memory.db (table clusters)
#
# Notes :
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence) définies dans config/feature_sets.yaml.
# - Utilise IQFeed exclusivement via data_provider.py pour les données d’entrée.
# - Supprime toute référence à 320 features, 81 features, et obs_t.
# - Implémente retries (max 3, délai 2^attempt secondes) pour les opérations critiques.
# - Logs psutil dans data/logs/mind_performance.csv avec métriques détaillées.
# - Alertes via alert_manager.py et Telegram pour priorité ≥ 4.
# - Snapshots compressés avec gzip dans data/logs/mind_snapshots/.
# - Sauvegardes incrémentielles toutes les 5 min dans data/checkpoints/ (5 versions).
# - Sauvegardes distribuées toutes les 15 min vers S3 (configurable).
# - Tests unitaires disponibles dans tests/test_mind.py.
# - Phases intégrées : Phase 7 (mémoire contextuelle), Phase 8 (auto-conscience via confidence_drop_rate), Phase 17 (SHAP).
# - Validation complète prévue pour juin 2025.

import csv
import gzip
import hashlib
import json
import logging
import os
import queue
import signal
import threading
import time
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Callable, Dict, Iterator, List, Optional

import boto3
import matplotlib.pyplot as plt
import pandas as pd
import psutil
import seaborn as sns
from sklearn.cluster import KMeans

from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import get_config
from src.model.utils.db_setup import get_db_connection
from src.utils.telegram_alert import send_telegram_alert

try:
    from src.envs.trading_env import TradingEnv
except ImportError:
    TradingEnv = None  # Permet tests sans TradingEnv
from pathlib import Path

# Configuration du logging
logger = logging.getLogger("MiyaMind")

# Configuration
CONFIG_PATH = "config/es_config.yaml"
DEFAULT_CONFIG = {
    "mia": {
        "language": "fr",
        "vocal_enabled": True,
        "vocal_async": True,
        "log_dir": "data/logs",
        "enable_csv_log": False,
        "log_rotation_mb": 10,
        "verbosity": "normal",
        "voice_profile": "calm",
        "max_logs": 1000,
        "buffer_size": 100,
        "s3_bucket": None,
        "s3_prefix": "mind/",
    }
}

# Limite du buffer de logs
MAX_LOG_BUFFER = 10000
KEEP_LOG_BUFFER = 5000

# Chemins
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
MARKET_MEMORY_DB = BASE_DIR / "data" / "market_memory.db"
LOG_DIR = BASE_DIR / "data" / "logs"
SNAPSHOT_DIR = LOG_DIR / "mind_snapshots"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints"
FIGURES_DIR = BASE_DIR / "data" / "figures" / "mind"

# Constantes
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Secondes, exponentiel

# Création des dossiers
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Variable pour gérer l'arrêt propre
RUNNING = True


class MindEngine:
    """
    Classe pour gérer le moteur cognitif central de MIA, incluant les logs, la communication et l’analyse.
    """

    def __init__(self, config_path: str = CONFIG_PATH):
        """
        Initialise le moteur cognitif de MIA.

        Args:
            config_path (str): Chemin vers la configuration.
        """
        self.alert_manager = AlertManager()
        self.checkpoint_versions = []
        signal.signal(signal.SIGINT, self.handle_sigint)

        start_time = time.time()
        try:
            self.config = self.load_config(os.path.basename(config_path))
            self.log_dir = BASE_DIR / self.config["log_dir"]
            self.log_dir.mkdir(exist_ok=True)
            self.brain_state_file = self.log_dir / "brain_state.json"
            self.csv_log_file = None
            self.mind_stack = []
            self.log_queue = queue.PriorityQueue()
            self.agents = []
            self.log_buffer = []  # Buffer pour logs psutil

            # Configurer le logger
            handler = RotatingFileHandler(
                self.log_dir / "mind_stream.log",
                maxBytes=self.config["log_rotation_mb"] * 1024 * 1024,
                backupCount=5,
            )
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - [Miya] %(message)s")
            )
            logger.handlers = [handler]
            logger.setLevel(logging.INFO)

            # Lancer le thread de journalisation
            threading.Thread(target=self.log_worker, daemon=True).start()

            # Support multilingue
            self.translations = {
                "fr": {
                    "thinking": "Je réfléchis...",
                    "decision": "Décision :",
                    "reason": "Raison :",
                    "summary": "Résumé :",
                    "context": "Contexte :",
                    "learn": "Apprentissage :",
                },
                "en": {
                    "thinking": "I'm thinking...",
                    "decision": "Decision:",
                    "reason": "Reason:",
                    "summary": "Summary:",
                    "context": "Context:",
                    "learn": "Learning:",
                },
            }

            # Gestionnaires d’événements
            self.event_hooks = {
                "ALERT": lambda msg: self.miya_speak(
                    f"Urgence : {msg}", tag="NOTIFY", priority=4, voice_profile="urgent"
                ),
                "DECIDE": lambda msg: self.miya_speak(
                    f"Confirmation : {msg}", tag="CONFIRM", priority=3
                ),
            }

            success_msg = "Moteur cognitif initialisé"
            self.alert_manager.send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            latency = time.time() - start_time
            self.log_performance("init_mind_engine", latency, success=True)
            self.save_snapshot(
                "init",
                {"config_path": config_path, "timestamp": datetime.now().isoformat()},
            )
        except Exception as e:
            error_msg = (
                f"Erreur initialisation MindEngine : {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init_mind_engine", 0, success=False, error=str(e))
            raise

    def handle_sigint(self, signum, frame):
        """Gère l'arrêt propre sur SIGINT."""
        global RUNNING
        start_time = time.time()
        snapshot = {"timestamp": datetime.now().isoformat(), "status": "SIGINT"}
        self.log_dir / f'sigint_{snapshot["timestamp"]}.json.gz'
        try:
            RUNNING = False
            self.save_brain_state()
            self.save_snapshot("sigint", snapshot)
            success_msg = "Arrêt propre sur SIGINT, logs et snapshot sauvegardés"
            self.alert_manager.send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "handle_sigint", time.time() - start_time, success=True
            )
            exit(0)
        except Exception as e:
            error_msg = f"Erreur sauvegarde SIGINT: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("handle_sigint", 0, success=False, error=str(e))
            exit(1)

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ):
        """
        Enregistre les performances avec psutil dans data/logs/mind_performance.csv.

        Args:
            operation (str): Nom de l’opération.
            latency (float): Temps d’exécution en secondes.
            success (bool): Succès de l’opération.
            error (str, optional): Message d’erreur.
            **kwargs: Paramètres supplémentaires (ex. : num_logs, snapshot_size_mb).
        """
        start_time = time.time()
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
            cpu_usage = psutil.cpu_percent()
            if memory_usage > 1024:
                alert_msg = f"ALERTE: Usage mémoire élevé ({memory_usage:.2f} MB)"
                self.alert_manager.send_alert(alert_msg, priority=5)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)
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
            if len(self.log_buffer) >= self.config.get("buffer_size", 100):
                log_df = pd.DataFrame(self.log_buffer)
                performance_log = self.log_dir / "mind_performance.csv"
                mode = "a" if performance_log.exists() else "w"

                def save_log():
                    log_df.to_csv(
                        performance_log,
                        mode=mode,
                        index=False,
                        header=not performance_log.exists(),
                        encoding="utf-8",
                    )

                self.with_retries(save_log)
                self.log_buffer = []
            logger.info(f"Performance journalisée pour {operation}. CPU: {cpu_usage}%")
            self.log_performance(
                "log_performance", time.time() - start_time, success=True
            )
        except Exception as e:
            error_msg = (
                f"Erreur journalisation performance: {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("log_performance", 0, success=False, error=str(e))

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
            snapshot_path = SNAPSHOT_DIR / f"snapshot_{snapshot_type}_{timestamp}.json"
            SNAPSHOT_DIR.mkdir(exist_ok=True)

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
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Snapshot {snapshot_type} sauvegardé: {save_path}"
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "save_snapshot", latency, success=True, snapshot_size_mb=file_size
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("save_snapshot", 0, success=False, error=str(e))

    def checkpoint(self, data: pd.DataFrame) -> None:
        """
        Sauvegarde incrémentielle des logs toutes les 5 minutes avec versionnage (5 versions).

        Args:
            data (pd.DataFrame): Données des logs à sauvegarder.
        """
        try:
            start_time = datetime.now()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_data = {
                "timestamp": timestamp,
                "num_rows": len(data),
                "columns": list(data.columns),
            }
            checkpoint_path = CHECKPOINT_DIR / f"mind_{timestamp}.json.gz"
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
            self.alert_manager.send_alert(success_msg, priority=1)
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
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("checkpoint", 0, success=False, error=str(e))

    def cloud_backup(self, data: pd.DataFrame) -> None:
        """
        Sauvegarde distribuée des logs vers AWS S3 toutes les 15 minutes (configurable).

        Args:
            data (pd.DataFrame): Données des logs à sauvegarder.
        """
        try:
            start_time = datetime.now()
            if not self.config.get("s3_bucket"):
                warning_msg = "S3 bucket non configuré, sauvegarde cloud ignorée"
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.config['s3_prefix']}mind_{timestamp}.csv.gz"
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
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "cloud_backup", latency, success=True, num_rows=len(data)
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde cloud S3: {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("cloud_backup", 0, success=False, error=str(e))

    def with_retries(
        self,
        func: callable,
        max_attempts: int = MAX_RETRIES,
        delay_base: float = RETRY_DELAY_BASE,
    ) -> Optional[any]:
        """
        Exécute une fonction avec retries (max 3, délai exponentiel).

        Args:
            func (callable): Fonction à exécuter.
            max_attempts (int): Nombre maximum de tentatives.
            delay_base (float): Base pour le délai exponentiel.

        Returns:
            Optional[any]: Résultat de la fonction ou None si échec.
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
                    self.alert_manager.send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    logger.error(error_msg)
                    self.log_performance(
                        f"retry_attempt_{attempt+1}",
                        0,
                        success=False,
                        error=str(e),
                        attempt_number=attempt + 1,
                    )
                    return None
                delay = delay_base**attempt
                warning_msg = f"Tentative {attempt+1} échouée, retry après {delay}s"
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
                time.sleep(delay)

    def load_config(self, config_path: str) -> Dict:
        """
        Charge la configuration avec repli sur les valeurs par défaut.

        Args:
            config_path (str): Nom du fichier de configuration.

        Returns:
            Dict: Configuration MIA.
        """
        start_time = datetime.now()
        try:
            config = get_config(BASE_DIR / config_path)
            if not config or "mia" not in config:
                raise ValueError("Configuration vide ou non trouvée")
            result = config["mia"]
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Configuration {config_path} chargée"
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("load_config", latency, success=True)
            return result
        except Exception as e:
            error_msg = f"Erreur lecture config : {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("load_config", 0, success=False, error=str(e))
            return DEFAULT_CONFIG["mia"]

    def init_csv_log(self) -> None:
        """
        Initialise le fichier CSV pour les logs.
        """
        start_time = datetime.now()
        try:
            if self.csv_log_file is None:
                self.csv_log_file = self.log_dir / "mind_stream.csv"
            if not self.csv_log_file.exists() and self.config.get(
                "enable_csv_log", False
            ):

                def init_csv():
                    with open(
                        self.csv_log_file, "w", newline="", encoding="utf-8"
                    ) as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                "timestamp",
                                "level",
                                "tag",
                                "priority",
                                "category",
                                "message",
                                "hash",
                                "balance",
                                "position",
                                "regime",
                                "cluster_id",
                            ]
                        )

                self.with_retries(init_csv)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Fichier CSV de logs initialisé"
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("init_csv_log", latency, success=True)
        except Exception as e:
            error_msg = (
                f"Erreur initialisation CSV log : {str(e)}\n{traceback.format_exc()}"
            )
            self.miya_alerts(
                error_msg, tag="ENGINE", priority=3, voice_profile="urgent"
            )
            self.log_performance("init_csv_log", 0, success=False, error=str(e))

    def analyze_log_patterns(self) -> pd.DataFrame:
        """
        Analyse les patterns de logs avec K-means (méthode 7) et stocke dans market_memory.db.

        Returns:
            pd.DataFrame: Logs avec cluster_id.
        """
        start_time = datetime.now()
        try:
            logs = list(self.load_logs(max_logs=100))
            if not logs:
                error_msg = "Aucun log pour clusterisation"
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                logger.warning(error_msg)
                return pd.DataFrame()

            df = pd.DataFrame(logs)
            required_cols = ["priority", "timestamp"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(df.columns)} colonnes)"
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            features = ["priority"]
            X = df[features].fillna(0).values
            if len(X) < 10:
                df["cluster_id"] = 0
                warning_msg = "Trop peu de logs pour clusterisation, cluster_id=0"
                self.alert_manager.send_alert(warning_msg, priority=2)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
                self.checkpoint(df)
                return df

            def run_kmeans():
                kmeans = KMeans(n_clusters=10, random_state=42)
                return kmeans.fit_predict(X)

            df["cluster_id"] = self.with_retries(run_kmeans)
            if df["cluster_id"].isnull().any():
                raise ValueError("Échec clusterisation K-means")

            # Stocker dans market_memory.db
            def store_clusters():
                conn = get_db_connection(str(MARKET_MEMORY_DB))
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
                for _, row in df.iterrows():
                    features_json = json.dumps({"priority": row["priority"]})
                    cursor.execute(
                        """
                        INSERT INTO clusters (cluster_id, event_type, features, timestamp)
                        VALUES (?, ?, ?, ?)
                    """,
                        (
                            row["cluster_id"],
                            row.get("tag", "UNKNOWN"),
                            features_json,
                            row["timestamp"],
                        ),
                    )
                conn.commit()
                conn.close()

            self.with_retries(store_clusters)

            # Sauvegarde incrémentielle
            self.checkpoint(df)
            self.cloud_backup(df)

            # Générer visualisation
            self.visualize_log_patterns(df)

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Clusterisation K-means terminée: {len(df)} logs, Confidence drop rate: {confidence_drop_rate:.2f}"
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "analyze_log_patterns",
                latency,
                success=True,
                num_logs=len(df),
                confidence_drop_rate=confidence_drop_rate,
            )
            self.save_snapshot(
                "analyze_log_patterns",
                {"num_logs": len(df), "confidence_drop_rate": confidence_drop_rate},
            )
            return df
        except Exception as e:
            error_msg = (
                f"Erreur clusterisation logs: {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("analyze_log_patterns", 0, success=False, error=str(e))
            return pd.DataFrame()

    def visualize_log_patterns(self, df: pd.DataFrame):
        """
        Génère une heatmap des clusters de logs dans data/figures/mind/.

        Args:
            df (pd.DataFrame): Données des logs avec cluster_id.
        """
        start_time = datetime.now()
        try:
            if df.empty or "cluster_id" not in df.columns:
                error_msg = "Aucune donnée pour visualisation"
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                logger.warning(error_msg)
                return

            pivot = df.pivot_table(
                index="cluster_id",
                columns="priority",
                values="timestamp",
                aggfunc="count",
                fill_value=0,
            )
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot, annot=True, cmap="Blues")
            plt.title("Heatmap des Clusters de Logs")
            output_path = (
                FIGURES_DIR / f"log_clusters_{datetime.now().strftime('%Y%m%d')}.png"
            )
            FIGURES_DIR.mkdir(exist_ok=True)

            def save_fig():
                plt.savefig(output_path)
                plt.close()

            self.with_retries(save_fig)

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Heatmap des clusters générée"
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("visualize_log_patterns", latency, success=True)
        except Exception as e:
            error_msg = (
                f"Erreur visualisation clusters: {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "visualize_log_patterns", 0, success=False, error=str(e)
            )

    def t(self, key: str) -> str:
        """
        Traduit une clé selon la langue configurée.

        Args:
            key (str): Clé à traduire.

        Returns:
            str: Traduction.
        """
        lang = self.config["language"]
        return self.translations.get(lang, self.translations["fr"]).get(key, key)

    def save_brain_state(self) -> None:
        """
        Sauvegarde l’état mental de MIA.
        """
        start_time = datetime.now()
        try:

            def save():
                with open(self.brain_state_file, "w", encoding="utf-8") as f:
                    json.dump(self.mind_stack[-100:], f, indent=4)  # Limite à 100

            self.with_retries(save)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "État mental sauvegardé"
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("save_brain_state", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur sauvegarde état : {str(e)}\n{traceback.format_exc()}"
            self.miya_alerts(error_msg, tag="BRAIN", priority=4, voice_profile="urgent")
            self.log_performance("save_brain_state", 0, success=False, error=str(e))

    def save_brain_snapshot(self, step: int) -> None:
        """
        Sauvegarde un instantané de l’état mental.

        Args:
            step (int): Étape de l’instantané.
        """
        start_time = datetime.now()
        try:
            snapshot = {
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "mind_stack": self.mind_stack[-10:],  # Derniers 10 logs
                "agent_count": len(self.agents),
            }
            self.save_snapshot("brain_snapshot", snapshot, compress=True)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Snapshot état mental step {step} sauvegardé"
            self.miya_speak(success_msg, tag="BRAIN", voice_profile="calm", priority=1)
            self.log_performance("save_brain_snapshot", latency, success=True)
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde snapshot état : {str(e)}\n{traceback.format_exc()}"
            )
            self.miya_alerts(error_msg, tag="BRAIN", priority=3, voice_profile="urgent")
            self.log_performance("save_brain_snapshot", 0, success=False, error=str(e))

    def load_brain_state(self) -> None:
        """
        Charge l’état mental sauvegardé.
        """
        start_time = datetime.now()
        try:

            def load():
                if self.brain_state_file.exists():
                    with open(self.brain_state_file, "r", encoding="utf-8") as f:
                        self.mind_stack = json.load(f)

            self.with_retries(load)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "État mental chargé"
            self.alert_manager.send_alert(success_msg, page=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("load_brain_state", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur chargement état : {str(e)}\n{traceback.format_exc()}"
            self.miya_alerts(error_msg, tag="BRAIN", level="warning", priority=3)
            self.log_performance("load_brain_state", 0, success=False, error=str(e))

    def register_agent(
        self,
        fn: Callable,
        agent_priority: int = 1,
        condition: Optional[Callable] = None,
    ) -> None:
        """
        Enregistre un agent cognitif.

        Args:
            fn (Callable): Fonction de l’agent.
            agent_priority (int): Priorité de l’agent.
            condition (Callable, optional): Condition d’exécution.
        """
        start_time = datetime.now()
        try:
            self.agents.append(
                {"fn": fn, "priority": agent_priority, "condition": condition}
            )
            success_msg = f"Agent enregistré : {fn.__name__}"
            self.miya_speak(success_msg, tag="AGENT", priority=2)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("register_agent", latency, success=True)
        except Exception as e:
            error_msg = (
                f"Erreur enregistrement agent : {str(e)}\n{traceback.format_exc()}"
            )
            self.miya_alerts(error_msg, tag="AGENT", priority=3, voice_profile="urgent")
            self.log_performance("register_agent", 0, success=False, error=str(e))

    def adjust_priority(self, message: str, priority: int) -> int:
        """
        Ajuste la priorité du message en fonction de son contenu.

        Args:
            message (str): Message à évaluer.
            priority (int): Priorité initiale.

        Returns:
            int: Priorité ajustée.
        """
        if any(
            keyword in message.lower()
            for keyword in ["erreur", "anomalie", "urgent", "alerte"]
        ):
            return priority + 1
        return priority

    def miya_speak(
        self,
        message: str,
        tag: str = "SYS",
        level: str = "info",
        vocal: bool = True,
        priority: int = 1,
        category: str = "generic",
        env: Optional[TradingEnv] = None,
        voice_profile: str = "default",
    ) -> None:
        """
        Dialogue/log cognitif central de MIA.

        Args:
            message (str): Message à afficher ou vocaliser.
            tag (str): Étiquette pour catégoriser le message.
            level (str): Niveau de gravité (info, warning, error).
            vocal (bool): Activer la synthèse vocale.
            priority (int): Priorité du message (1 à 5).
            category (str): Catégorie du message.
            env (TradingEnv, optional): Environnement de trading.
            voice_profile (str): Profil vocal (calm, urgent, educative).
        """
        start_time = datetime.now()
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            full_message = f"{message}"
            valid_levels = ["info", "warning", "error", "critical"]
            if level not in valid_levels:
                level = "info"

            # Affichage en console
            try:
                verbosity_levels = {"minimal": 4, "normal": 2, "verbose": 1}
                should_display = priority >= verbosity_levels.get(
                    self.config.get("verbosity", "normal"), 2
                )
                if should_display:
                    color_map = {
                        "info": "\033[92m",
                        "warning": "\033[93m",
                        "error": "\033[91m",
                        "critical": "\033[91m",
                    }
                    color = color_map.get(level, "\033[92m")
                    reset = "\033[0m"
                    print(f"{color}[MIA - {tag}] {full_message}{reset}")
            except Exception as e:
                print(f"[MIA] Erreur affichage message : {e}")

            # Log asynchrone
            adjusted_priority = self.adjust_priority(full_message, priority)
            self.log_queue.put(
                (adjusted_priority, (level, full_message, tag, priority))
            )

            # Log structuré avec cluster_id
            log_entry = {
                "timestamp": timestamp,
                "level": level,
                "tag": tag,
                "priority": priority,
                "category": category,
                "message": message,
                "hash": (
                    hashlib.sha256(full_message.encode("utf-8")).hexdigest()
                    if tag in ["ALERT", "FATAL"]
                    else None
                ),
                "balance": env.balance if env and hasattr(env, "balance") else None,
                "position": env.position if env and hasattr(env, "position") else None,
                "regime": env.mode if env and hasattr(env, "mode") else None,
                "cluster_id": None,  # Rempli après clusterisation
            }

            # Clusterisation asynchrone
            threading.Thread(target=self.analyze_log_patterns, daemon=True).start()

            def write_logs():
                # Écriture JSON
                json_path = self.log_dir / "mind_stream.json"
                if (
                    json_path.exists()
                    and json_path.stat().st_size
                    > self.config["log_rotation_mb"] * 1024 * 1024
                ):
                    os.rename(json_path, f"{json_path}.{timestamp}.bak")
                with open(json_path, "a", encoding="utf-8") as f:
                    json.dump(log_entry, f)
                    f.write("\n")

                # Log CSV
                if self.config.get("enable_csv_log", False):
                    if self.csv_log_file is None:
                        self.init_csv_log()
                    with open(
                        self.csv_log_file, "a", newline="", encoding="utf-8"
                    ) as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                timestamp,
                                level,
                                tag,
                                priority,
                                category,
                                message,
                                log_entry["hash"],
                                log_entry["balance"],
                                log_entry["position"],
                                log_entry["regime"],
                                log_entry["cluster_id"],
                            ]
                        )

            self.with_retries(write_logs)

            # Pile mentale
            self.mind_stack.append(log_entry)
            if len(self.mind_stack) > MAX_LOG_BUFFER:
                self.mind_stack = self.mind_stack[
                    -KEEP_LOG_BUFFER:
                ]  # Conserver les plus récents
            self.save_brain_state()
            self.save_brain_snapshot(len(self.mind_stack))

            # Événements
            if tag in self.event_hooks:
                self.event_hooks[tag](full_message)

            # Alertes Telegram pour priorité élevée
            if priority >= 4:
                send_telegram_alert(f"[MIA - {tag}] {full_message}")

            # Voix
            if vocal and self.config.get("vocal_enabled", True):
                try:
                    from src.mind.mind_voice import VoiceManager

                    def speak():
                        voice_manager = VoiceManager()
                        voice_manager.speak(
                            full_message,
                            voice_profile or self.config.get("voice_profile", "calm"),
                            self.config.get("vocal_async", True),
                        )

                    self.with_retries(speak)
                except Exception as e:
                    error_msg = f"Erreur vocalisation : {str(e)}, repli sur texte"
                    self.miya_alerts(
                        error_msg, tag="VOICE", level="warning", priority=3
                    )
                    send_telegram_alert(error_msg)

            # Agents
            for agent in sorted(self.agents, key=lambda x: x["priority"], reverse=True):
                if not agent["condition"] or agent["condition"](log_entry):
                    agent["fn"](env=env, log_entry=log_entry)

            # Notification dashboard
            if tag in ["ALERT", "DECIDE", "SUMMARY"]:
                self.miya_dashboard_notify(
                    {
                        "tag": tag,
                        "message": message,
                        "level": level,
                        "timestamp": timestamp,
                    }
                )

            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("miya_speak", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur miya_speak: {str(e)}\n{traceback.format_exc()}"
            self.miya_traceback_simplifier(e)
            send_telegram_alert(error_msg)
            self.log_performance("miya_speak", 0, success=False, error=str(e))

    def log_worker(self) -> None:
        """
        Thread asynchrone pour écrire les logs.
        """
        start_time = datetime.now()
        while RUNNING:
            try:
                priority, (level, message, tag, orig_priority) = self.log_queue.get(
                    timeout=10
                )
                log_func = getattr(logger, level.lower(), logger.info)
                log_func(f"[{tag}] {message} [P{orig_priority}]")
                self.log_queue.task_done()
                latency = (datetime.now() - start_time).total_seconds()
                self.log_performance("log_worker", latency, success=True)
            except queue.Empty:
                continue
            except Exception as e:
                error_msg = f"Erreur journalisation asynchrone : {str(e)}\n{traceback.format_exc()}"
                self.miya_alerts(
                    error_msg, tag="ENGINE", priority=5, voice_profile="urgent"
                )
                send_telegram_alert(error_msg)
                self.log_performance("log_worker", 0, success=False, error=str(e))

    def miya_dashboard_notify(self, data: Dict) -> None:
        """
        Envoie une notification au dashboard via un fichier JSON.

        Args:
            data (Dict): Données à partager (tag, message, level, timestamp).
        """
        start_time = datetime.now()
        try:

            def notify():
                json_path = self.log_dir / "mind_dashboard.json"
                notifications = []
                if json_path.exists():
                    with open(json_path, "r", encoding="utf-8") as f:
                        notifications = json.load(f)
                notifications.append(data)
                notifications = notifications[-10:]  # Limite à 10 dernières
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(notifications, f, indent=4)

            self.with_retries(notify)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Notification dashboard envoyée"
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("miya_dashboard_notify", latency, success=True)
        except Exception as e:
            error_msg = (
                f"Erreur notification dashboard : {str(e)}\n{traceback.format_exc()}"
            )
            self.miya_alerts(error_msg, tag="DASHBOARD", level="warning", priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "miya_dashboard_notify", 0, success=False, error=str(e)
            )

    def miya_thinks(
        self,
        thought: str,
        category: str = "analyse",
        priority: int = 1,
        env: Optional[TradingEnv] = None,
    ) -> None:
        """Enregistre une réflexion de MIA."""
        self.miya_speak(
            f"{self.t('thinking')} {thought}",
            tag="THINK",
            priority=priority,
            category=category,
            env=env,
        )

    def miya_decides(
        self, decision: str, priority: int = 3, env: Optional[TradingEnv] = None
    ) -> None:
        """Enregistre une décision de MIA."""
        self.miya_speak(
            f"{self.t('decision')} {decision}",
            tag="DECIDE",
            priority=priority,
            category="decision",
            env=env,
        )

    def miya_explains(
        self, reason: str, priority: int = 2, env: Optional[TradingEnv] = None
    ) -> None:
        """Explique une décision ou un événement."""
        self.miya_speak(
            f"{self.t('reason')} {reason}",
            tag="EXPLAIN",
            priority=priority,
            category="explanation",
            env=env,
        )

    def miya_alerts(
        self,
        warning: str,
        priority: int = 4,
        env: Optional[TradingEnv] = None,
        voice_profile: str = "urgent",
    ) -> None:
        """Envoie une alerte critique."""
        self.miya_speak(
            f"⚠️ Alerte : {warning}",
            tag="ALERT",
            level="warning",
            priority=priority,
            category="alert",
            env=env,
            voice_profile=voice_profile,
        )
        self.alert_manager.send_alert(f"Alerte MIA : {warning}", priority=priority)
        if priority >= 4:
            send_telegram_alert(f"Alerte MIA : {warning}")

    def miya_context(self, env: Optional[TradingEnv] = None, priority: int = 2) -> Dict:
        """Résume le contexte actuel."""
        start_time = datetime.now()
        try:
            context = {
                "balance": env.balance if env and hasattr(env, "balance") else None,
                "position": env.position if env and hasattr(env, "position") else None,
                "regime": env.mode if env and hasattr(env, "mode") else "inconnu",
                "log_count": len(self.mind_stack),
                "agent_count": len(self.agents),
            }
            message = (
                f"{self.t('context')} Balance={context['balance']}, Position={context['position']}, "
                f"Régime={context['regime']}, Logs={context['log_count']}, Agents={context['agent_count']}"
            )
            self.miya_speak(
                message, tag="CONTEXT", priority=priority, category="status"
            )
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("miya_context", latency, success=True)
            return context
        except Exception as e:
            error_msg = f"Erreur contexte : {str(e)}\n{traceback.format_exc()}"
            self.miya_alerts(
                error_msg, tag="CONTEXT", priority=4, voice_profile="urgent"
            )
            send_telegram_alert(error_msg)
            self.log_performance("miya_context", 0, success=False, error=str(e))
            return {}

    def miya_summary(self, priority: int = 2) -> None:
        """Résume l’activité récente."""
        start_time = datetime.now()
        try:
            stats = self.analyze_logs()
            message = (
                f"{self.t('summary')} {stats['total_logs']} logs, {stats['errors']} erreurs, "
                f"{stats.get('by_tag', {}).get('DECIDE', 0)} décisions"
            )
            self.miya_speak(
                message, tag="SUMMARY", priority=priority, category="status"
            )
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("miya_summary", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur résumé : {str(e)}\n{traceback.format_exc()}"
            self.miya_alerts(
                error_msg, tag="SUMMARY", priority=4, voice_profile="urgent"
            )
            send_telegram_alert(error_msg)
            self.log_performance("miya_summary", 0, success=False, error=str(e))

    def miya_narrative_builder(self, priority: int = 2) -> str:
        """Construit un récit des événements récents."""
        start_time = datetime.now()
        try:
            logs = list(self.load_logs(max_logs=50))
            narrative = "Historique récent :\n"
            for log in logs[-5:]:
                narrative += f"- {log['timestamp']} [{log['tag']}]: {log['message']}\n"
            self.miya_speak(
                narrative, tag="NARRATIVE", priority=priority, category="history"
            )
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("miya_narrative_builder", latency, success=True)
            return narrative
        except Exception as e:
            error_msg = f"Erreur narrative : {str(e)}\n{traceback.format_exc()}"
            self.miya_alerts(error_msg, tag="ALERT", voice_profile="urgent", priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "miya_narrative_builder", 0, success=False, error=str(e)
            )
            return ""

    def miya_feedback(
        self, trade_result: Dict, priority: int = 3, env: Optional[TradingEnv] = None
    ) -> None:
        """Analyse un résultat de trade."""
        start_time = datetime.now()
        try:
            profit = trade_result.get("profit", 0)
            action = trade_result.get("action", "inconnu")
            message = f"Feedback trade : Action={action}, Profit={profit}, Balance={env.balance if env else 'N/A'}"
            self.miya_speak(
                message, tag="FEEDBACK", priority=priority, category="trade", env=env
            )
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("miya_feedback", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur feedback : {str(e)}\n{traceback.format_exc()}"
            self.miya_alerts(
                error_msg, tag="FEEDBACK", priority=4, voice_profile="urgent"
            )
            send_telegram_alert(error_msg)
            self.log_performance("miya_feedback", 0, success=False, error=str(e))

    def miya_error_classifier(self, priority: int = 3) -> Dict:
        """Classe les erreurs récentes."""
        start_time = datetime.now()
        try:
            logs = list(self.load_logs(max_logs=100))
            errors = [log for log in logs if log.get("level") in ["warning", "error"]]
            classified = {
                "connexion": sum(
                    1 for log in errors if "connexion" in log["message"].lower()
                ),
                "data": sum(1 for log in errors if "data" in log["message"].lower()),
                "stratégie": sum(
                    1 for log in errors if "trade" in log["message"].lower()
                ),
                "agent": sum(1 for log in errors if "agent" in log["message"].lower()),
            }
            message = (
                f"Erreurs : Connexion={classified['connexion']}, Data={classified['data']}, "
                f"Stratégie={classified['stratégie']}, Agent={classified['agent']}"
            )
            self.miya_speak(
                message, tag="ERROR_CLASS", priority=priority, category="analysis"
            )
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("miya_error_classifier", latency, success=True)
            return classified
        except Exception as e:
            error_msg = f"Erreur classification : {str(e)}\n{traceback.format_exc()}"
            self.miya_alerts(error_msg, tag="ALERT", voice_profile="urgent", priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "miya_error_classifier", 0, success=False, error=str(e)
            )
            return {}

    def miya_detect_patterns(self, priority: int = 3) -> List[str]:
        """Détecte des séquences d’événements récurrents."""
        start_time = datetime.now()
        try:
            logs = list(self.load_logs(max_logs=100))
            patterns = []
            for i in range(len(logs) - 2):
                if (
                    logs[i]["tag"] == "CONTEXT"
                    and logs[i + 1]["level"] == "warning"
                    and logs[i + 2]["tag"] == "DECIDE"
                ):
                    patterns.append(
                        f"Contexte {logs[i]['message']} → Erreur {logs[i+1]['message']} → Décision {logs[i+2]['message']}"
                    )
            message = f"Patterns : {len(patterns)} séquences trouvées"
            self.miya_speak(
                message, tag="PATTERN", priority=priority, category="analysis"
            )
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("miya_detect_patterns", latency, success=True)
            return patterns
        except Exception as e:
            error_msg = f"Erreur patterns : {str(e)}\n{traceback.format_exc()}"
            self.miya_alerts(error_msg, tag="ALERT", voice_profile="urgent", priority=4)
            send_telegram_alert(error_msg)
            self.log_performance("miya_detect_patterns", 0, success=False, error=str(e))
            return []

    def miya_speaks_markets(
        self, market_state: Dict, priority: int = 2, env: Optional[TradingEnv] = None
    ) -> None:
        """Commente l’état du marché."""
        start_time = datetime.now()
        try:
            rsi = market_state.get("rsi", "inconnu")
            gex = market_state.get("gex", "inconnu")
            sentiment = (
                "haussier"
                if rsi == "inconnu" or rsi > 70
                else "baissier" if rsi < 30 else "neutre"
            )
            message = f"Marché : RSI={rsi}, GEX={gex}, Sentiment={sentiment}"
            self.miya_speak(
                message,
                tag="MARKET",
                priority=priority,
                category="market",
                env=env,
                voice_profile="educative",
            )
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("miya_speaks_markets", latency, success=True)
        except Exception as e:
            error_msg = (
                f"Erreur commentaire marché : {str(e)}\n{traceback.format_exc()}"
            )
            self.miya_alerts(
                error_msg, tag="MARKET", priority=4, voice_profile="urgent"
            )
            send_telegram_alert(error_msg)
            self.log_performance("miya_speaks_markets", 0, success=False, error=str(e))

    def miya_learns(self, priority: int = 3) -> List[str]:
        """Propose des optimisations basées sur les logs."""
        start_time = datetime.now()
        try:
            stats = self.analyze_logs()
            suggestions = []
            if stats["errors"] > 10:
                suggestions.append("Vérifier connexion IQFeed : trop d’erreurs réseau")
            if stats.get("by_tag", {}).get("AGENT", 0) > stats.get("by_tag", {}).get(
                "DECIDE", 0
            ):
                suggestions.append("Réduire fréquence agents pour éviter surcharge")
            message = f"{self.t('learn')} {len(suggestions)} suggestions : {', '.join(suggestions) if suggestions else 'aucune'}"
            self.miya_speak(
                message, tag="LEARN", priority=priority, category="learning"
            )
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("miya_learns", latency, success=True)
            return suggestions
        except Exception as e:
            error_msg = f"Erreur apprentissage : {str(e)}\n{traceback.format_exc()}"
            self.miya_alerts(error_msg, tag="LEARN", priority=4, voice_profile="urgent")
            send_telegram_alert(error_msg)
            self.log_performance("miya_learns", 0, success=False, error=str(e))
            return []

    def miya_env_watcher(
        self,
        env: Optional[TradingEnv] = None,
        priority: int = 2,
        log_entry: Optional[Dict] = None,
    ) -> None:
        """Observe l’environnement de trading."""
        start_time = datetime.now()
        try:
            if env:
                rsi = env.rsi if hasattr(env, "rsi") else "N/A"
                message = f"Observation : RSI={rsi}, Position={env.position if hasattr(env, 'position') else 'N/A'}"
                self.miya_speak(
                    message,
                    tag="WATCHER",
                    priority=priority,
                    category="environment",
                    voice_profile="educative",
                )
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("miya_env_watcher", latency, success=True)
        except Exception as e:
            error_msg = (
                f"Erreur observation environnement : {str(e)}\n{traceback.format_exc()}"
            )
            self.miya_alerts(
                error_msg, tag="WATCHER", priority=4, voice_profile="urgent"
            )
            send_telegram_alert(error_msg)
            self.log_performance("miya_env_watcher", 0, success=False, error=str(e))

    def analyze_logs(self) -> Dict:
        """Analyse détaillée des logs."""
        start_time = datetime.now()
        try:
            logs = list(self.load_logs(max_logs=self.config["max_logs"]))
            stats = {
                "total_logs": len(logs),
                "by_tag": {
                    tag: sum(1 for log in logs if log.get("tag") == tag)
                    for tag in ["THINK", "ALERT", "DECIDE", "SUMMARY", "AGENT"]
                },
                "errors": sum(
                    1
                    for log in logs
                    if log.get("level") in ["warning", "error", "critical"]
                ),
                "frequent_messages": pd.Series([log["message"] for log in logs])
                .value_counts()
                .head()
                .to_dict(),
            }
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "analyze_logs", latency, success=True, num_logs=len(logs)
            )
            return stats
        except Exception as e:
            error_msg = f"Erreur analyse logs : {str(e)}\n{traceback.format_exc()}"
            self.miya_alerts(error_msg, tag="ALERT", voice_profile="urgent", priority=4)
            send_telegram_alert(error_msg)
            self.log_performance("analyze_logs", 0, success=False, error=str(e))
            return {"total_logs": 0, "by_tag": {}, "errors": 0, "frequent_messages": {}}

    def miya_health_check(self, priority: int = 2) -> Dict:
        """Vérifie l’état du système."""
        start_time = datetime.now()
        try:
            critical_paths = [
                BASE_DIR / "data" / "features" / "features_latest.csv",
                BASE_DIR / "config" / "es_config.yaml",
                BASE_DIR / "config" / "feature_sets.yaml",
            ]
            checks = {"files": {}, "logs": {}, "agents": len(self.agents)}
            self.miya_speak("Contrôle santé système...", tag="CHECK", priority=priority)
            for path in critical_paths:
                checks["files"][str(path)] = path.exists()
                status = "✅" if checks["files"][str(path)] else "❌"
                size_kb = (
                    round(path.stat().st_size / 1024, 2)
                    if checks["files"][str(path)]
                    else 0
                )
                self.miya_speak(
                    f"{status} {path} ({size_kb} Ko)", tag="HEALTH", priority=priority
                )
                # Validation des features pour features_latest.csv
                if path.name == "features_latest.csv" and checks["files"][str(path)]:
                    try:
                        df = pd.read_csv(path)
                        num_features = len(df.columns)
                        if num_features >= 350:
                            feature_set = "training"
                            expected_features = 350
                        elif num_features >= 150:
                            feature_set = "inference"
                            expected_features = 150
                        else:
                            self.miya_alerts(
                                f"Nombre de features insuffisant dans {path} : {num_features} < 150",
                                tag="HEALTH",
                                priority=4,
                            )
                            continue
                        if num_features != expected_features:
                            self.miya_alerts(
                                f"Nombre de features incorrect dans {path} : {num_features} au lieu de {expected_features} pour {feature_set}",
                                tag="HEALTH",
                                priority=4,
                            )
                        # Validation des types scalaires
                        critical_cols = [
                            "bid_size_level_1",
                            "ask_size_level_1",
                            "trade_frequency_1s",
                        ]
                        for col in critical_cols:
                            if col in df.columns:
                                if not pd.api.types.is_numeric_dtype(df[col]):
                                    self.miya_alerts(
                                        f"Colonne {col} n’est pas numérique : {df[col].dtype}",
                                        tag="HEALTH",
                                        priority=4,
                                    )
                                non_scalar = [
                                    val
                                    for val in df[col]
                                    if isinstance(val, (list, dict, tuple))
                                ]
                                if non_scalar:
                                    self.miya_alerts(
                                        f"Colonne {col} contient des valeurs non scalaires : {non_scalar[:5]}",
                                        tag="HEALTH",
                                        priority=4,
                                    )
                    except Exception as e:
                        self.miya_alerts(
                            f"Erreur validation {path} : {str(e)}",
                            tag="HEALTH",
                            priority=4,
                        )
                # Validation des features SHAP dans feature_sets.yaml
                if path.name == "feature_sets.yaml" and checks["files"][str(path)]:
                    try:
                        feature_config = get_config(path)
                        shap_features = feature_config.get("inference", {}).get(
                            "shap_features", []
                        )
                        if len(shap_features) != 150:
                            self.miya_alerts(
                                f"Nombre de SHAP features incorrect dans {path} : {len(shap_features)} au lieu de 150",
                                tag="HEALTH",
                                priority=4,
                            )
                    except Exception as e:
                        self.miya_alerts(
                            f"Erreur validation {path} : {str(e)}",
                            tag="HEALTH",
                            priority=4,
                        )
            checks["logs"]["size"] = (
                round(
                    (self.log_dir / "mind_stream.log").stat().st_size / 1024 / 1024, 2
                )
                if (self.log_dir / "mind_stream.log").exists()
                else 0
            )
            checks["logs"]["json_count"] = len(list(self.load_logs(max_logs=100)))
            self.miya_speak(
                f"Logs : {checks['logs']['size']} Mo, {checks['logs']['json_count']} JSON",
                tag="HEALTH",
                priority=priority,
            )
            self.miya_speak(
                f"Agents actifs : {checks['agents']}", tag="HEALTH", priority=priority
            )

            # Sauvegarde snapshot
            snapshot = {"timestamp": datetime.now().isoformat(), "checks": checks}
            self.save_snapshot("health_check", snapshot)

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Contrôle santé système terminé"
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("miya_health_check", latency, success=True)
            return checks
        except Exception as e:
            error_msg = f"Erreur santé système : {str(e)}\n{traceback.format_exc()}"
            self.miya_alerts(
                error_msg, tag="HEALTH", priority=4, voice_profile="urgent"
            )
            send_telegram_alert(error_msg)
            self.log_performance("miya_health_check", 0, success=False, error=str(e))
            return {}

    def miya_debug_mode(self, enable: bool = True) -> None:
        """Active/désactive le mode debug."""
        start_time = datetime.now()
        try:
            level = logging.DEBUG if enable else logging.INFO
            logger.setLevel(level)
            success_msg = f"Mode debug {'activé' if enable else 'désactivé'}"
            self.miya_speak(success_msg, tag="DEBUG", priority=2)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("miya_debug_mode", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur mode debug : {str(e)}\n{traceback.format_exc()}"
            self.miya_alerts(error_msg, tag="DEBUG", priority=4, voice_profile="urgent")
            send_telegram_alert(error_msg)
            self.log_performance("miya_debug_mode", 0, success=False, error=str(e))

    def miya_traceback_simplifier(self, exc) -> None:
        """Simplifie une traceback pour les logs."""
        start_time = datetime.now()
        try:
            simplified = traceback.format_exception(type(exc), exc, exc.__traceback__)
            important = [
                line for line in simplified if "File" in line or "Error" in line
            ][:3]
            message = "\n".join(important)
            self.miya_speak(
                f"Erreur détectée :\n{message}", tag="ERROR", level="error", priority=4
            )
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("miya_traceback_simplifier", latency, success=True)
        except Exception as e:
            error_msg = (
                f"Erreur simplification traceback : {str(e)}\n{traceback.format_exc()}"
            )
            self.miya_alerts(error_msg, tag="ERROR", priority=4, voice_profile="urgent")
            send_telegram_alert(error_msg)
            self.log_performance(
                "miya_traceback_simplifier", 0, success=False, error=str(e)
            )

    def miya_cli(self) -> None:
        """Interface CLI interactive pour MIA."""
        start_time = datetime.now()
        try:
            self.miya_speak(
                "Terminal MIA : santé, debug, normal, contexte, résumé, narrative, apprendre, exit",
                tag="CLI",
                priority=2,
            )
            while True:
                try:
                    cmd = input("🧠 MIA > ").strip().lower()
                    if cmd in ("exit", "quit", "stop"):
                        success_msg = "Session terminée"
                        self.miya_speak(success_msg, tag="CLI", priority=2)
                        self.alert_manager.send_alert(success_msg, priority=1)
                        send_telegram_alert(success_msg)
                        break
                    elif cmd in ("santé", "check"):
                        self.miya_health_check()
                    elif cmd == "debug":
                        self.miya_debug_mode(True)
                    elif cmd in ("normal", "standard"):
                        self.miya_debug_mode(False)
                    elif cmd == "contexte":
                        self.miya_context()
                    elif cmd == "résumé":
                        self.miya_summary()
                    elif cmd == "narrative":
                        self.miya_narrative_builder()
                    elif cmd == "apprendre":
                        self.miya_learns()
                    else:
                        self.miya_speak(
                            f"Commande inconnue : {cmd}",
                            tag="CLI",
                            level="warning",
                            priority=3,
                        )
                except KeyboardInterrupt:
                    success_msg = "Sortie CLI"
                    self.miya_speak(success_msg, tag="CLI", priority=2)
                    self.alert_manager.send_alert(success_msg, priority=1)
                    send_telegram_alert(success_msg)
                    break
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("miya_cli", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur CLI : {str(e)}\n{traceback.format_exc()}"
            self.miya_alerts(error_msg, tag="CLI", priority=4, voice_profile="urgent")
            send_telegram_alert(error_msg)
            self.log_performance("miya_cli", 0, success=False, error=str(e))

    def load_logs(self, max_logs: int = 1000) -> Iterator[Dict]:
        """
        Charge les logs JSON depuis mind_stream.json avec un générateur.

        Args:
            max_logs (int): Nombre maximum de logs à charger.

        Yields:
            Dict: Log chargé.
        """
        start_time = datetime.now()
        try:
            json_path = self.log_dir / "mind_stream.json"
            if not json_path.exists():
                error_msg = "Aucun log JSON trouvé"
                self.miya_alerts(error_msg, tag="ENGINE", level="warning", priority=3)
                send_telegram_alert(error_msg)
                return

            def load():
                with open(json_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i >= max_logs:
                            break
                        if line.strip():
                            try:
                                yield json.loads(line)
                            except json.JSONDecodeError:
                                self.miya_alerts(
                                    "Log JSON corrompu ignoré",
                                    tag="ENGINE",
                                    level="warning",
                                    priority=2,
                                )
                                send_telegram_alert("Log JSON corrompu ignoré")
                                continue

            for log in self.with_retries(load) or []:
                yield log
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("load_logs", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur chargement logs : {str(e)}\n{traceback.format_exc()}"
            self.miya_alerts(error_msg, tag="ALERT", voice_profile="urgent", priority=4)
            send_telegram_alert(error_msg)
            self.log_performance("load_logs", 0, success=False, error=str(e))


if __name__ == "__main__":
    engine = MindEngine()
    engine.load_brain_state()
    engine.init_csv_log()
    env = TradingEnv("config/es_config.yaml") if TradingEnv else None
    if env:
        env.balance = 10000.0
        env.position = 1
        env.mode = "trend"
        env.rsi = 75
    engine.register_agent(engine.miya_env_watcher, agent_priority=2)
    engine.miya_thinks("Analyse volatilité", category="observation")
    engine.miya_decides("Acheter ES", priority=3, env=env)
    engine.miya_explains("RSI > 70")
    engine.miya_alerts("Risque drawdown", priority=4)
    engine.miya_context(env=env)
    engine.miya_summary()
    engine.miya_narrative_builder()
    engine.miya_feedback({"action": "buy", "profit": 100})
    engine.miya_error_classifier()
    engine.miya_detect_patterns()
    engine.miya_speaks_markets({"rsi": 75, "gex": 100000})
    engine.miya_learns()
    engine.miya_health_check()
    engine.miya_cli()
