# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/model/adaptive_learning.py
# Gère l'apprentissage adaptatif en stockant les patterns de marché dans market_memory.db, utilise K-means pour le clustering,
# et réentraîne/fine-tune les modèles SAC/PPO/DDPG pour MIA_IA_SYSTEM_v2_2025.
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle :
# Stocke les patterns de marché (features, action, récompense, régime, probabilités, SHAP, volatilité, options, LSTM),
# effectue un clustering K-means, et déclenche l’entraînement via train_sac_auto.py. Supporte 350 features (entraînement)
# ou 150 SHAP features (inférence).
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, sqlite3, psutil>=5.9.8,<6.0.0, matplotlib>=3.7.0,<4.0.0,
#   sklearn>=1.5.0,<2.0.0, stable-baselines3>=2.0.0,<3.0.0, boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0,
#   pyyaml>=6.0.0,<7.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/model_validator.py
# - src/model/utils/algo_performance_logger.py
# - src/model/train_sac.py
# - src/model/train_sac_auto.py
# - src/envs/trading_env.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/feature_sets.yaml, config/algo_config.yaml via config_manager
# - Données brutes (DataFrame avec 350 features en entraînement ou 150 SHAP features en inférence)
#
# Outputs :
# - Patterns/clusters dans data/market_memory.db
# - Modèles SAC/PPO/DDPG dans model/sac_models/<market>/
# - Logs dans data/logs/adaptive_learning_performance.csv
# - Snapshots compressés dans data/cache/adaptive_learning/<market>/*.json.gz
# - Sauvegardes incrémentielles dans data/checkpoints/adaptive_learning/<market>/*.json.gz
# - Figures dans data/figures/adaptive_learning/<market>/
#
# Lien avec SAC :
# Fournit les patterns et déclenche l'entraînement pour train_sac_auto.py et live_trading.py
#
# Notes :
# - Utilise exclusivement IQFeed comme source de données via TradingEnv.
# - Suppression de toute référence à dxFeed, obs_t, 320/81 features pour conformité avec MIA_IA_SYSTEM_v2_2025.
# - Intègre les Phases 1-18 :
#   - Phase 1 (news_scraper.py, news_analyzer.py) : Analyse de sentiment (news_sentiment_score).
#   - Phase 13 (orderflow_indicators.py, options_metrics.py) : Métriques comme vix_es_correlation, atr_14, call_iv_atm, option_skew.
#   - Phase 15 (microstructure_guard.py, spotgamma_recalculator.py) : Métriques de microstructure (spoofing_score, volume_anomaly) et options (net_gamma, call_wall).
#   - Phase 18 : Métriques avancées de microstructure (trade_velocity, hft_activity_score).
# - Intègre régimes hybrides (méthode 11), volatilité (méthode 1), données d’options (méthode 2), SHAP (méthode 17),
#   et prédictions LSTM (méthode 12).
# - Implémente retries (max 3, délai 2^attempt secondes) pour les opérations critiques.
# - Sauvegardes incrémentielles toutes les 5 min dans data/checkpoints/adaptive_learning/ (5 versions).
# - Sauvegardes distribuées toutes les 15 min vers S3 (configurable).
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des données.
# - Tests unitaires disponibles dans tests/test_adaptive_learning.py.

import asyncio
import gzip
import json
import os
import signal
import sqlite3
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.envs.trading_env import TradingEnv
from src.model.train_sac import SACTrainer
from src.model.train_sac_auto import TrainSACAuto
from src.model.utils.alert_manager import AlertManager
from src.model.utils.algo_performance_logger import AlgoPerformanceLogger
from src.model.utils.config_manager import get_config
from src.model.utils.model_validator import ModelValidator
from src.utils.telegram_alert import send_telegram_alert

# Configuration logging
logger.remove()
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOG_DIR = BASE_DIR / "data" / "logs"
CACHE_DIR = BASE_DIR / "data" / "cache" / "adaptive_learning"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints" / "adaptive_learning"
FIGURE_DIR = BASE_DIR / "data" / "figures" / "adaptive_learning"
PERF_LOG_PATH = LOG_DIR / "adaptive_learning_performance.csv"
SNAPSHOT_DIR = CACHE_DIR
DB_PATH = BASE_DIR / "data" / "market_memory.db"
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "adaptive_learning.log", rotation="10 MB", level="INFO", encoding="utf-8"
)
logger.add(lambda msg: print(msg, end=""), level="INFO")

# Constantes
MAX_RETRIES = 3
RETRY_DELAY = 2.0
MAX_CACHE_SIZE = 1000
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB

# Variable pour gérer l'arrêt propre
RUNNING = True


class AdaptiveLearning:
    """
    Gère l'apprentissage adaptatif et la mémoire de l'IA pour MIA_IA_SYSTEM_v2_2025.

    Attributes:
        config (Dict): Configuration chargée depuis algo_config.yaml.
        alert_manager (AlertManager): Gestionnaire d’alertes.
        validator (ModelValidator): Validateur de modèles.
        performance_logger (AlgoPerformanceLogger): Enregistreur de performances.
        scaler (StandardScaler): Normalisateur des données.
        snapshot_dir (Path): Dossier pour snapshots JSON.
        perf_log (Path): Fichier pour logs de performance.
        figure_dir (Path): Dossier pour visualisations.
        training_mode (bool): Si True, utilise 350 features; sinon, 150 SHAP features.
        feature_cols (List[str]): Liste des colonnes de features.
        num_features (int): Nombre de features (350 ou 150).
        cluster_cache (OrderedDict): Cache LRU pour les calculs de clustering.
        checkpoint_versions (List): Liste des versions de checkpoints.
        log_buffer (List): Buffer pour les logs de performance.
    """

    def __init__(self, training_mode: bool = True):
        """
        Initialise l'apprenant adaptatif.

        Args:
            training_mode (bool): Si True, utilise 350 features; sinon, 150 SHAP features.
        """
        start_time = datetime.now()
        try:
            self.alert_manager = AlertManager()
            self.config = get_config(BASE_DIR / "config/algo_config.yaml")
            self.validator = ModelValidator()
            self.performance_logger = AlgoPerformanceLogger()
            self.scaler = StandardScaler()
            self.training_mode = training_mode
            self.num_features = 350 if training_mode else 150
            feature_sets_path = BASE_DIR / "config/feature_sets.yaml"
            self.features = get_config(feature_sets_path)
            self.feature_cols = (
                self.features.get("training", {}).get("features", [])[:350]
                if training_mode
                else self.features.get("inference", {}).get("shap_features", [])[:150]
            )
            if len(self.feature_cols) != self.num_features:
                raise ValueError(
                    f"Attendu {self.num_features} features, trouvé {len(self.feature_cols)}"
                )

            self.snapshot_dir = SNAPSHOT_DIR
            self.perf_log = PERF_LOG_PATH
            self.figure_dir = FIGURE_DIR
            self.cluster_cache = OrderedDict()
            self.max_cache_size = MAX_CACHE_SIZE
            self.checkpoint_versions = []
            self.log_buffer = []

            self.initialize_database()
            success_msg = f"AdaptiveLearning initialisé avec {self.num_features} features, training_mode={training_mode}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "init", latency, success=True, num_features=self.num_features
            )
            signal.signal(signal.SIGINT, self.handle_sigint)
        except Exception as e:
            error_msg = f"Erreur initialisation AdaptiveLearning: {str(e)}\n{traceback.format_exc()}"
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
        try:
            RUNNING = False
            self.save_snapshot("sigint", snapshot, market="ES")
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

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ) -> None:
        """
        Journalise les performances dans adaptive_learning_performance.csv.

        Args:
            operation (str): Nom de l’opération.
            latency (float): Temps d’exécution en secondes.
            success (bool): Indique si l’opération a réussi.
            error (str): Message d’erreur (si applicable).
            **kwargs: Paramètres supplémentaires (ex. : num_features, snapshot_size_mb).
        """
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
                "latency": latency,
                "success": success,
                "error": error,
                "memory_usage_mb": memory_usage,
                "cpu_percent": cpu_percent,
                **kwargs,
            }
            self.log_buffer.append(log_entry)
            if len(self.log_buffer) >= self.config.get("logging", {}).get(
                "buffer_size", 100
            ):
                log_df = pd.DataFrame(self.log_buffer)

                def save_log():
                    if not PERF_LOG_PATH.exists():
                        log_df.to_csv(PERF_LOG_PATH, index=False, encoding="utf-8")
                    else:
                        log_df.to_csv(
                            PERF_LOG_PATH,
                            mode="a",
                            header=False,
                            index=False,
                            encoding="utf-8",
                        )

                self.with_retries(save_log)
                self.checkpoint(log_df, data_type="performance_logs", market="ES")
                self.cloud_backup(log_df, data_type="performance_logs", market="ES")
                self.log_buffer = []
            logger.info(
                f"Performance journalisée pour {operation}. CPU: {cpu_percent}%"
            )
        except Exception as e:
            error_msg = (
                f"Erreur journalisation performance: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)

    def with_retries(
        self,
        func: callable,
        max_attempts: int = MAX_RETRIES,
        delay_base: float = RETRY_DELAY,
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

    def save_snapshot(
        self, snapshot_type: str, data: Dict, market: str = "ES", compress: bool = True
    ) -> None:
        """
        Sauvegarde un instantané des résultats, compressé avec gzip.

        Args:
            snapshot_type (str): Type de snapshot (ex. : store_pattern).
            data (Dict): Données à sauvegarder.
            market (str): Marché (ex. : ES, MNQ).
            compress (bool): Compresser avec gzip (défaut : True).
        """
        try:
            start_time = datetime.now()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = {
                "timestamp": timestamp,
                "type": snapshot_type,
                "market": market,
                "data": data,
                "buffer_size": len(self.log_buffer),
            }
            snapshot_dir = SNAPSHOT_DIR / market
            snapshot_dir.mkdir(exist_ok=True)
            snapshot_path = snapshot_dir / f"snapshot_{snapshot_type}_{timestamp}.json"

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
                alert_msg = (
                    f"Snapshot size {file_size:.2f} MB exceeds 1 MB pour {market}"
                )
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = (
                f"Snapshot {snapshot_type} sauvegardé pour {market}: {save_path}"
            )
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "save_snapshot",
                latency,
                success=True,
                snapshot_size_mb=file_size,
                market=market,
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot {snapshot_type} pour {market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "save_snapshot", 0, success=False, error=str(e), market=market
            )

    def checkpoint(
        self,
        data: pd.DataFrame,
        data_type: str = "adaptive_learning_state",
        market: str = "ES",
    ) -> None:
        """
        Sauvegarde incrémentielle des données toutes les 5 minutes avec versionnage (5 versions).

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : adaptive_learning_state).
            market (str): Marché (ex. : ES, MNQ).
        """
        try:
            start_time = datetime.now()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_data = {
                "timestamp": timestamp,
                "num_rows": len(data),
                "columns": list(data.columns),
                "data_type": data_type,
                "market": market,
            }
            checkpoint_dir = CHECKPOINT_DIR / market
            checkpoint_dir.mkdir(exist_ok=True)
            checkpoint_path = (
                checkpoint_dir / f"adaptive_learning_{data_type}_{timestamp}.json.gz"
            )

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
            success_msg = f"Checkpoint sauvegardé pour {market}: {checkpoint_path}"
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
                market=market,
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde checkpoint pour {market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "checkpoint",
                0,
                success=False,
                error=str(e),
                data_type=data_type,
                market=market,
            )

    def cloud_backup(
        self,
        data: pd.DataFrame,
        data_type: str = "adaptive_learning_state",
        market: str = "ES",
    ) -> None:
        """
        Sauvegarde distribuée des données vers AWS S3 toutes les 15 minutes (configurable).

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : adaptive_learning_state).
            market (str): Marché (ex. : ES, MNQ).
        """
        try:
            start_time = datetime.now()
            if not self.config.get("s3_bucket"):
                warning_msg = (
                    f"S3 bucket non configuré, sauvegarde cloud ignorée pour {market}"
                )
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.config['s3_prefix']}adaptive_learning_{data_type}_{market}_{timestamp}.csv.gz"
            temp_path = CHECKPOINT_DIR / market / f"temp_s3_{timestamp}.csv.gz"
            temp_path.parent.mkdir(exist_ok=True)

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
            success_msg = f"Sauvegarde cloud S3 effectuée pour {market}: {backup_path}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "cloud_backup",
                latency,
                success=True,
                num_rows=len(data),
                data_type=data_type,
                market=market,
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde cloud S3 pour {market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "cloud_backup",
                0,
                success=False,
                error=str(e),
                data_type=data_type,
                market=market,
            )

    def initialize_database(self, db_path: str = str(DB_PATH)) -> None:
        """
        Initialise la base de données SQLite pour stocker les patterns et clusters.

        Args:
            db_path (str): Chemin vers la base de données SQLite.
        """
        start_time = datetime.now()
        try:

            def init_db():
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        market TEXT,
                        features TEXT,  -- JSON des 350/150 features
                        action REAL,
                        reward REAL,
                        neural_regime INTEGER,
                        confidence REAL,
                        regime_probs TEXT,  -- JSON des probabilités de régimes
                        shap_values TEXT,  -- JSON des valeurs SHAP
                        vix_es_correlation REAL,
                        atr_14 REAL,
                        call_iv_atm REAL,
                        option_skew REAL,
                        predicted_vix REAL,
                        news_sentiment_score REAL,  -- Phase 1
                        spoofing_score REAL,  -- Phase 15
                        volume_anomaly REAL,  -- Phase 15
                        net_gamma REAL,  -- Phase 15
                        call_wall REAL,  -- Phase 15
                        trade_velocity REAL,  -- Phase 18
                        hft_activity_score REAL  -- Phase 18
                    )
                """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS clusters (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        market TEXT,
                        centroid TEXT,  -- JSON du centroïde (350/150 features)
                        cluster_size INTEGER
                    )
                """
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_patterns_timestamp ON patterns (timestamp)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_patterns_market ON patterns (market)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_clusters_timestamp ON clusters (timestamp)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_clusters_market ON clusters (market)"
                )
                conn.commit()
                conn.close()
                logger.info(f"Base de données initialisée: {db_path}")

            self.with_retries(init_db)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("initialize_database", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur initialisation base de données: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance("initialize_database", 0, success=False, error=str(e))
            raise

    def verify_figures(self, market: str = "ES") -> bool:
        """
        Vérifie la présence et la validité des figures dans data/figures/adaptive_learning/<market>/.

        Args:
            market (str): Marché (ex. : ES, MNQ).

        Returns:
            bool: True si au moins une figure valide est trouvée, False sinon.
        """
        start_time = datetime.now()
        try:
            figure_dir = self.figure_dir / market
            valid_figures = False
            for file in figure_dir.glob("*.png"):
                if file.is_file() and file.stat().st_size > 0:
                    valid_figures = True
                    logger.info(f"Figure valide trouvée pour {market}: {file}")
            if not valid_figures:
                warning_msg = f"Aucune figure valide trouvée dans {figure_dir}"
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "verify_figures", latency, success=valid_figures, market=market
            )
            return valid_figures
        except Exception as e:
            error_msg = f"Erreur vérification figures pour {market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "verify_figures", 0, success=False, error=str(e), market=market
            )
            return False

    def validate_data(self, data: pd.DataFrame, market: str = "ES") -> bool:
        """
        Valide que les données contiennent les features attendues.

        Args:
            data (pd.DataFrame): Données à valider.
            market (str): Marché (ex. : ES, MNQ).

        Returns:
            bool: True si valide, False sinon.
        """
        start_time = datetime.now()
        try:
            missing_cols = [col for col in self.feature_cols if col not in data.columns]
            if missing_cols:
                error_msg = f"Colonnes manquantes pour {market}: {missing_cols}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                return False
            null_count = data[self.feature_cols].isnull().sum().sum()
            confidence_drop_rate = (
                null_count / (len(data) * len(self.feature_cols))
                if (len(data) * len(self.feature_cols)) > 0
                else 0.0
            )
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé pour {market}: {confidence_drop_rate:.2f} ({null_count} valeurs nulles)"
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
            if null_count > 0:
                error_msg = f"Valeurs nulles détectées dans les features pour {market}: {null_count}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                return False
            if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
                error_msg = (
                    f"Colonne 'timestamp' doit être de type datetime pour {market}"
                )
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                return False
            for col in self.feature_cols:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    error_msg = f"Colonne {col} n'est pas numérique pour {market}: {data[col].dtype}"
                    logger.error(error_msg)
                    self.alert_manager.send_alert(error_msg, priority=3)
                    send_telegram_alert(error_msg)
                    return False
                if data[col].apply(lambda x: isinstance(x, (list, dict, tuple))).any():
                    error_msg = f"Colonne {col} contient des valeurs non scalaires pour {market}"
                    logger.error(error_msg)
                    self.alert_manager.send_alert(error_msg, priority=3)
                    send_telegram_alert(error_msg)
                    return False
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "validate_data",
                latency,
                success=True,
                confidence_drop_rate=confidence_drop_rate,
                market=market,
            )
            return True
        except Exception as e:
            error_msg = f"Erreur validation données pour {market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "validate_data", 0, success=False, error=str(e), market=market
            )
            return False

    async def store_pattern(
        self,
        data: pd.DataFrame,
        action: float,
        reward: float,
        neural_regime: Optional[int] = None,
        confidence: float = 0.7,
        regime_probs: Optional[Dict[str, float]] = None,
        shap_values: Optional[Dict[str, float]] = None,
        vix_es_correlation: float = 0.0,
        atr_14: float = 0.0,
        call_iv_atm: float = 0.0,
        option_skew: float = 0.0,
        predicted_vix: float = 0.0,
        news_sentiment_score: float = 0.0,  # Phase 1
        spoofing_score: float = 0.0,  # Phase 15
        volume_anomaly: float = 0.0,  # Phase 15
        net_gamma: float = 0.0,  # Phase 15
        call_wall: float = 0.0,  # Phase 15
        trade_velocity: float = 0.0,  # Phase 18
        hft_activity_score: float = 0.0,  # Phase 18
        market: str = "ES",
        db_path: str = str(DB_PATH),
    ) -> None:
        """
        Stocke un pattern de marché dans market_memory.db.

        Args:
            data (pd.DataFrame): Données avec 350/150 features (une ligne).
            action (float): Action prise (-1 pour vendre, 1 pour acheter).
            reward (float): Récompense obtenue.
            neural_regime (Optional[int]): Régime (0: trend, 1: range, 2: défensif).
            confidence (float): Confiance dans l’action.
            regime_probs (Optional[Dict[str, float]]): Probabilités des régimes.
            shap_values (Optional[Dict[str, float]]): Valeurs SHAP des features.
            vix_es_correlation (float): Corrélation VIX-ES.
            atr_14 (float): ATR sur 14 périodes.
            call_iv_atm (float): Implied volatility ATM des calls.
            option_skew (float): Skew des options.
            predicted_vix (float): VIX prédit par LSTM.
            news_sentiment_score (float): Score de sentiment des nouvelles (Phase 1).
            spoofing_score (float): Score de détection de spoofing (Phase 15).
            volume_anomaly (float): Score d'anomalie de volume (Phase 15).
            net_gamma (float): Gamma net (Phase 15).
            call_wall (float): Mur de calls (Phase 15).
            trade_velocity (float): Vélocité des trades (Phase 18).
            hft_activity_score (float): Score d'activité HFT (Phase 18).
            market (str): Marché (ex. : ES, MNQ).
            db_path (str): Chemin vers la base de données SQLite.
        """
        start_time = datetime.now()
        try:
            if not isinstance(data, pd.DataFrame) or len(data) != 1:
                raise ValueError(
                    f"Les données doivent être un DataFrame avec une seule ligne pour {market}"
                )
            if not self.validate_data(data, market=market):
                error_msg = f"Données invalides pour stockage pattern ({market})"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                return

            features = data[self.feature_cols].iloc[0].to_dict()
            features_json = pd.Series(features).to_json(orient="columns")
            timestamp = (
                data["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M:%S")
                if "timestamp" in data.columns
                else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            neural_regime = neural_regime if neural_regime is not None else -1
            regime_probs = regime_probs or {
                "range": 0.33,
                "trend": 0.33,
                "defensive": 0.34,
            }
            regime_probs_json = json.dumps(regime_probs)
            shap_values = shap_values or {}
            shap_values_json = json.dumps(shap_values)

            def store():
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO patterns (timestamp, market, features, action, reward, neural_regime, confidence, regime_probs,
                                         shap_values, vix_es_correlation, atr_14, call_iv_atm, option_skew, predicted_vix,
                                         news_sentiment_score, spoofing_score, volume_anomaly, net_gamma, call_wall,
                                         trade_velocity, hft_activity_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp,
                        market,
                        features_json,
                        action,
                        reward,
                        neural_regime,
                        confidence,
                        regime_probs_json,
                        shap_values_json,
                        vix_es_correlation,
                        atr_14,
                        call_iv_atm,
                        option_skew,
                        predicted_vix,
                        news_sentiment_score,
                        spoofing_score,
                        volume_anomaly,
                        net_gamma,
                        call_wall,
                        trade_velocity,
                        hft_activity_score,
                    ),
                )
                conn.commit()
                conn.close()

            self.with_retries(store)

            snapshot = {
                "timestamp": timestamp,
                "market": market,
                "action": action,
                "reward": reward,
                "neural_regime": neural_regime,
                "confidence": confidence,
                "regime_probs": regime_probs,
                "shap_values": shap_values,
                "vix_es_correlation": vix_es_correlation,
                "atr_14": atr_14,
                "call_iv_atm": call_iv_atm,
                "option_skew": option_skew,
                "predicted_vix": predicted_vix,
                "news_sentiment_score": news_sentiment_score,
                "spoofing_score": spoofing_score,
                "volume_anomaly": volume_anomaly,
                "net_gamma": net_gamma,
                "call_wall": call_wall,
                "trade_velocity": trade_velocity,
                "hft_activity_score": hft_activity_score,
            }
            self.save_snapshot("store_pattern", snapshot, market=market)

            success_msg = f"Pattern stocké pour {market}: timestamp={timestamp}, action={action}, reward={reward}, regime_probs={regime_probs}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "store_pattern", latency, success=True, reward=reward, market=market
            )

            pattern_df = pd.DataFrame(
                [
                    {
                        "timestamp": timestamp,
                        "market": market,
                        "action": action,
                        "reward": reward,
                        **{f"feature_{k}": v for k, v in features.items()},
                        **{f"regime_prob_{k}": v for k, v in regime_probs.items()},
                        **{f"shap_{k}": v for k, v in shap_values.items()},
                    }
                ]
            )
            self.checkpoint(pattern_df, data_type="pattern", market=market)
            self.cloud_backup(pattern_df, data_type="pattern", market=market)
        except Exception as e:
            error_msg = f"Erreur stockage pattern pour {market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "store_pattern", 0, success=False, error=str(e), market=market
            )

    async def last_3_trades_success_rate(
        self, market: str = "ES", db_path: str = str(DB_PATH)
    ) -> float:
        """
        Calcule le taux de succès des 3 derniers trades (proportion de trades avec reward > 0).

        Args:
            market (str): Marché (ex. : ES, MNQ).
            db_path (str): Chemin vers la base de données SQLite.

        Returns:
            float: Taux de succès (0 à 1), ou 0 si moins de 3 trades.
        """
        start_time = datetime.now()
        try:

            def get_rewards():
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT reward FROM patterns WHERE market = ? ORDER BY timestamp DESC LIMIT 3",
                    (market,),
                )
                rewards = [row[0] for row in cursor.fetchall()]
                conn.close()
                return rewards

            rewards = self.with_retries(get_rewards)
            if rewards is None or len(rewards) < 3:
                error_msg = f"Moins de 3 trades disponibles pour {market} ({len(rewards) if rewards else 0})"
                logger.warning(error_msg)
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                return 0.0

            success_rate = sum(1 for r in rewards if r > 0) / len(rewards)
            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "market": market,
                "success_rate": success_rate,
                "rewards": rewards,
            }
            self.save_snapshot("success_rate", snapshot, market=market)

            success_msg = f"Taux de succès des 3 derniers trades pour {market}: {success_rate:.2f}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "success_rate",
                latency,
                success=True,
                success_rate=success_rate,
                market=market,
            )
            return success_rate
        except Exception as e:
            error_msg = f"Erreur calcul taux de succès pour {market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "success_rate", 0, success=False, error=str(e), market=market
            )
            return 0.0


async def cluster_score_distance(
    self,
    data: pd.DataFrame,
    n_clusters: int = 5,
    market: str = "ES",
    db_path: str = str(DB_PATH),
    figure_name: Optional[str] = None,
) -> float:
    """
    Calcule la distance entre le pattern actuel et le centroïde du cluster le plus proche.

    Args:
        data (pd.DataFrame): Données avec 350/150 features (une ligne).
        n_clusters (int): Nombre de clusters pour K-means.
        market (str): Marché (ex. : ES, MNQ).
        db_path (str): Chemin vers la base de données SQLite.
        figure_name (Optional[str]): Nom statique pour la figure, sinon dynamique.

    Returns:
        float: Distance au centroïde le plus proche (0 si pas assez de données).
    """
    start_time = datetime.now()
    try:
        if not isinstance(data, pd.DataFrame) or len(data) != 1:
            raise ValueError(
                f"Les données doivent être un DataFrame avec une seule ligne pour {market}"
            )
        if not self.validate_data(data, market=market):
            error_msg = f"Données invalides pour clustering ({market})"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            return 0.0

        cache_key = f"{market}_{data['timestamp'].iloc[0].isoformat()}_{n_clusters}"
        if cache_key in self.cluster_cache:
            min_distance = self.cluster_cache[cache_key]
            self.cluster_cache.move_to_end(cache_key)
            return min_distance
        while len(self.cluster_cache) > self.max_cache_size:
            self.cluster_cache.popitem(last=False)

        def load_patterns():
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT features, regime_probs, shap_values, vix_es_correlation, atr_14, call_iv_atm,
                       option_skew, predicted_vix, news_sentiment_score, spoofing_score, volume_anomaly,
                       net_gamma, call_wall, trade_velocity, hft_activity_score
                FROM patterns WHERE market = ?
                """,
                (market,),
            )
            patterns = [
                {
                    "features": pd.read_json(row[0], typ="series"),
                    "regime_probs": json.loads(row[1]),
                    "shap_values": json.loads(row[2]),
                    "vix_es_correlation": row[3],
                    "atr_14": row[4],
                    "call_iv_atm": row[5],
                    "option_skew": row[6],
                    "predicted_vix": row[7],
                    "news_sentiment_score": row[8],
                    "spoofing_score": row[9],
                    "volume_anomaly": row[10],
                    "net_gamma": row[11],
                    "call_wall": row[12],
                    "trade_velocity": row[13],
                    "hft_activity_score": row[14],
                }
                for row in cursor.fetchall()
            ]
            conn.close()
            return patterns

        patterns = self.with_retries(load_patterns)
        if patterns is None or len(patterns) < n_clusters:
            error_msg = f"Pas assez de patterns pour clustering ({len(patterns) if patterns else 0} < {n_clusters}) pour {market}"
            logger.warning(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            return 0.0

        pattern_data = pd.DataFrame(
            [p["features"] for p in patterns], columns=self.feature_cols
        )
        extra_cols = [
            "vix_es_correlation",
            "atr_14",
            "call_iv_atm",
            "option_skew",
            "predicted_vix",
            "news_sentiment_score",
            "spoofing_score",
            "volume_anomaly",
            "net_gamma",
            "call_wall",
            "trade_velocity",
            "hft_activity_score",
        ]
        for col in extra_cols:
            pattern_data[col] = [p[col] for p in patterns]
        current_pattern = data[self.feature_cols].iloc[0].values
        current_extra = (
            data[extra_cols].iloc[0].values
            if all(col in data.columns for col in extra_cols)
            else [0.0] * len(extra_cols)
        )
        current_data = np.concatenate([current_pattern, current_extra]).reshape(1, -1)

        pattern_scaled = self.scaler.fit_transform(pattern_data)
        current_scaled = self.scaler.transform(current_data)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(pattern_scaled)
        distances = kmeans.transform(current_scaled)
        min_distance = distances.min()

        def store_clusters():
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            for i, centroid in enumerate(kmeans.cluster_centers_):
                centroid_json = pd.Series(
                    self.scaler.inverse_transform(centroid)[: self.num_features],
                    index=self.feature_cols,
                ).to_json()
                cluster_size = sum(kmeans.labels_ == i)
                cursor.execute(
                    """
                    INSERT INTO clusters (timestamp, market, centroid, cluster_size)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        market,
                        centroid_json,
                        cluster_size,
                    ),
                )
            conn.commit()
            conn.close()

        self.with_retries(store_clusters)

        plt.figure(figsize=(10, 8))
        plt.scatter(
            pattern_scaled[:, 0],
            pattern_scaled[:, 1],
            c=kmeans.labels_,
            cmap="viridis",
            alpha=0.5,
        )
        plt.scatter(
            current_scaled[:, 0],
            current_scaled[:, 1],
            c="red",
            marker="x",
            s=200,
            label="Current Pattern",
        )
        plt.title(
            f"Clustering pour {market}: Distance au plus proche {min_distance:.4f}\n"
            f"VIX: {current_extra[0]:.2f}, IV: {current_extra[2]:.2f}, Skew: {current_extra[3]:.2f}, "
            f"Sentiment: {current_extra[5]:.2f}, Spoofing: {current_extra[6]:.2f}"
        )
        plt.xlabel("Feature 1 (scaled)")
        plt.ylabel("Feature 2 (scaled)")
        plt.legend()
        figure_dir = self.figure_dir / market
        figure_dir.mkdir(exist_ok=True)
        figure_filename = (
            figure_name or f"cluster_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        figure_path = figure_dir / figure_filename
        plt.savefig(figure_path)
        plt.close()

        self.verify_figures(market=market)

        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "market": market,
            "min_distance": min_distance,
            "n_clusters": n_clusters,
            "vix_es_correlation": current_extra[0],
            "atr_14": current_extra[1],
            "call_iv_atm": current_extra[2],
            "option_skew": current_extra[3],
            "predicted_vix": current_extra[4],
            "news_sentiment_score": current_extra[5],
            "spoofing_score": current_extra[6],
            "volume_anomaly": current_extra[7],
            "net_gamma": current_extra[8],
            "call_wall": current_extra[9],
            "trade_velocity": current_extra[10],
            "hft_activity_score": current_extra[11],
        }
        self.save_snapshot("cluster_distance", snapshot, market=market)

        self.cluster_cache[cache_key] = min_distance

        success_msg = (
            f"Distance au cluster le plus proche pour {market}: {min_distance:.4f}"
        )
        logger.info(success_msg)
        self.alert_manager.send_alert(success_msg, priority=2)
        send_telegram_alert(success_msg)
        latency = (datetime.now() - start_time).total_seconds()
        self.log_performance(
            "cluster_distance",
            latency,
            success=True,
            min_distance=min_distance,
            market=market,
        )
        return min_distance
    except Exception as e:
        error_msg = f"Erreur calcul distance cluster pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        self.alert_manager.send_alert(error_msg, priority=4)
        send_telegram_alert(error_msg)
        self.log_performance(
            "cluster_distance", 0, success=False, error=str(e), market=market
        )
        return 0.0

    async def clean_database(
        self,
        max_age_days: int = 30,
        max_patterns: int = 10000,
        market: str = "ES",
        db_path: str = str(DB_PATH),
    ) -> None:
        """
        Nettoie la base de données en supprimant les patterns anciens ou excédentaires.

        Args:
            max_age_days (int): Âge maximum des patterns en jours.
            max_patterns (int): Nombre maximum de patterns à conserver.
            market (str): Marché (ex. : ES, MNQ).
            db_path (str): Chemin vers la base de données SQLite.
        """
        start_time = datetime.now()
        try:

            def clean():
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cutoff_date = (datetime.now() - timedelta(days=max_age_days)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                cursor.execute(
                    "DELETE FROM patterns WHERE market = ? AND timestamp < ?",
                    (market, cutoff_date),
                )
                deleted_old = cursor.rowcount
                cursor.execute(
                    "SELECT COUNT(*) FROM patterns WHERE market = ?", (market,)
                )
                total_patterns = cursor.fetchone()[0]
                deleted_excess = 0
                if total_patterns > max_patterns:
                    cursor.execute(
                        """
                        DELETE FROM patterns WHERE id IN (
                            SELECT id FROM patterns WHERE market = ? ORDER BY timestamp ASC LIMIT ?
                        )
                        """,
                        (market, total_patterns - max_patterns),
                    )
                    deleted_excess = cursor.rowcount
                conn.commit()
                conn.close()
                return deleted_old, deleted_excess, total_patterns

            result = self.with_retries(clean)
            if result is None:
                return
            deleted_old, deleted_excess, total_patterns = result

            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "market": market,
                "deleted_old": deleted_old,
                "deleted_excess": deleted_excess,
                "total_patterns": total_patterns,
            }
            self.save_snapshot("clean_database", snapshot, market=market)

            success_msg = f"Nettoyage base de données pour {market}: {deleted_old} anciens, {deleted_excess} excédentaires"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "clean_database",
                latency,
                success=True,
                deleted_count=deleted_old + deleted_excess,
                market=market,
            )
        except Exception as e:
            error_msg = f"Erreur nettoyage base de données pour {market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "clean_database", 0, success=False, error=str(e), market=market
            )

    async def retrain_model(
        self,
        env: TradingEnv,
        mode: str = "trend",
        algo_type: str = "sac",
        policy_type: str = "mlp",
        timesteps: int = 10000,
        min_trades: int = 100,
        market: str = "ES",
        db_path: str = str(DB_PATH),
    ) -> Optional[Any]:
        """
        Réentraîne un modèle SAC/PPO/DDPG à partir de zéro avec les trades récents.

        Args:
            env (TradingEnv): Environnement de trading.
            mode (str): Mode du modèle ("trend", "range", "defensive").
            algo_type (str): Type d’algorithme ("sac", "ppo", "ddpg").
            policy_type (str): Type de politique ("mlp", "transformer").
            timesteps (int): Nombre de timesteps pour l’entraînement.
            min_trades (int): Nombre minimum de trades requis.
            market (str): Marché (ex. : ES, MNQ).
            db_path (str): Chemin vers la base de données SQLite.

        Returns:
            Optional[Any]: Modèle réentraîné (SAC, PPO, ou DDPG), ou None en cas d’échec.
        """
        start_time = datetime.now()
        try:
            valid_modes = ["trend", "range", "defensive"]
            valid_algo_types = ["sac", "ppo", "ddpg"]
            valid_policies = ["mlp", "transformer"]
            if mode not in valid_modes:
                raise ValueError(
                    f"Mode non supporté pour {market}: {mode}. Options: {valid_modes}"
                )
            if algo_type not in valid_algo_types:
                raise ValueError(
                    f"Type d’algorithme non supporté pour {market}: {algo_type}. Options: {valid_algo_types}"
                )
            if policy_type not in valid_policies:
                raise ValueError(
                    f"Type de politique non supporté pour {market}: {policy_type}. Options: {valid_policies}"
                )

            def load_trades():
                conn = sqlite3.connect(db_path)
                trades = pd.read_sql_query(
                    "SELECT * FROM patterns WHERE market = ? ORDER BY timestamp DESC",
                    conn,
                    params=(market,),
                )
                conn.close()
                return trades

            trades = self.with_retries(load_trades)
            if trades is None or len(trades) < min_trades:
                error_msg = f"Trades insuffisants pour {market}: {len(trades) if trades is not None else 0} < {min_trades}"
                logger.warning(error_msg)
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                return None

            features_df = pd.json_normalize(trades["features"])
            extra_cols = [
                "vix_es_correlation",
                "atr_14",
                "call_iv_atm",
                "option_skew",
                "predicted_vix",
                "news_sentiment_score",
                "spoofing_score",
                "volume_anomaly",
                "net_gamma",
                "call_wall",
                "trade_velocity",
                "hft_activity_score",
            ]
            trades = pd.concat(
                [
                    trades[
                        [
                            "timestamp",
                            "action",
                            "reward",
                            "neural_regime",
                            "confidence",
                            "regime_probs",
                        ]
                    ],
                    pd.json_normalize(trades["regime_probs"]).rename(
                        columns={
                            k: f"regime_prob_{k}"
                            for k in ["range", "trend", "defensive"]
                        }
                    ),
                    pd.json_normalize(trades["shap_values"]),
                    trades[extra_cols],
                    features_df,
                ],
                axis=1,
            )

            trainer = SACTrainer(env, policy_type)
            await trainer.train_multi_market(
                trades, total_timesteps=timesteps, mode=mode, market=market
            )
            model = trainer.models.get(algo_type, {}).get(mode)
            if not model:
                error_msg = f"Modèle {algo_type} ({mode}) non disponible après réentraînement pour {market}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                return None

            validation = self.validator.validate_model(
                model, algo_type, mode, trades[self.feature_cols].values
            )
            if not validation["valid"]:
                error_msg = f"Modèle {algo_type} ({mode}) invalide après réentraînement pour {market}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                return None

            save_dir = Path("model/sac_models") / market / mode / policy_type
            save_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = save_dir / f"{algo_type}_{mode}_{policy_type}_{timestamp}.zip"
            model.save(save_path)

            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "market": market,
                "algo_type": algo_type,
                "mode": mode,
                "save_path": str(save_path),
                "trades_count": len(trades),
                "mean_reward": validation.get("mean_reward", 0.0),
            }
            self.save_snapshot(f"retrain_{algo_type}_{mode}", snapshot, market=market)

            success_msg = f"Réentraînement {algo_type} réussi pour {market}: mode={mode}, policy_type={policy_type}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "retrain_model",
                latency,
                success=True,
                algo_type=algo_type,
                mode=mode,
                mean_reward=validation.get("mean_reward", 0.0),
                market=market,
            )
            return model
        except Exception as e:
            error_msg = f"Erreur réentraînement {algo_type} pour {market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "retrain_model", 0, success=False, error=str(e), market=market
            )
            return None

    async def fine_tune_model(
        self,
        model: Any,
        env: TradingEnv,
        mode: str = "trend",
        algo_type: str = "sac",
        policy_type: str = "mlp",
        timesteps: int = 5000,
        min_trades: int = 50,
        min_confidence: float = 0.7,
        batch_size: int = 64,
        market: str = "ES",
        db_path: str = str(DB_PATH),
    ) -> Optional[Any]:
        """
        Ajuste finement un modèle SAC/PPO/DDPG existant avec les trades récents.

        Args:
            model (Any): Modèle à ajuster (SAC, PPO, DDPG).
            env (TradingEnv): Environnement de trading.
            mode (str): Mode du modèle ("trend", "range", "defensive").
            algo_type (str): Type d’algorithme ("sac", "ppo", "ddpg").
            policy_type (str): Type de politique ("mlp", "transformer").
            timesteps (int): Nombre de timesteps pour l’ajustement.
            min_trades (int): Nombre minimum de trades requis.
            min_confidence (float): Seuil de confiance pour inclure un trade.
            batch_size (int): Taille du batch pour l’apprentissage.
            market (str): Marché (ex. : ES, MNQ).
            db_path (str): Chemin vers la base de données SQLite.

        Returns:
            Optional[Any]: Modèle ajusté, ou None en cas d’échec.
        """
        start_time = datetime.now()
        try:
            valid_modes = ["trend", "range", "defensive"]
            valid_algo_types = ["sac", "ppo", "ddpg"]
            valid_policies = ["mlp", "transformer"]
            if mode not in valid_modes:
                raise ValueError(
                    f"Mode non supporté pour {market}: {mode}. Options: {valid_modes}"
                )
            if algo_type not in valid_algo_types:
                raise ValueError(
                    f"Type d’algorithme non supporté pour {market}: {algo_type}. Options: {valid_algo_types}"
                )
            if policy_type not in valid_policies:
                raise ValueError(
                    f"Type de politique non supporté pour {market}: {policy_type}. Options: {valid_policies}"
                )

            def load_trades():
                conn = sqlite3.connect(db_path)
                trades = pd.read_sql_query(
                    "SELECT * FROM patterns WHERE market = ? AND confidence >= ? ORDER BY timestamp DESC",
                    conn,
                    params=(market, min_confidence),
                )
                conn.close()
                return trades

            trades = self.with_retries(load_trades)
            if trades is None or len(trades) < min_trades:
                error_msg = f"Trades filtrés insuffisants pour {market}: {len(trades) if trades is not None else 0} < {min_trades}"
                logger.warning(error_msg)
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                return model

            features_df = pd.json_normalize(trades["features"])
            extra_cols = [
                "vix_es_correlation",
                "atr_14",
                "call_iv_atm",
                "option_skew",
                "predicted_vix",
                "news_sentiment_score",
                "spoofing_score",
                "volume_anomaly",
                "net_gamma",
                "call_wall",
                "trade_velocity",
                "hft_activity_score",
            ]
            trades = pd.concat(
                [
                    trades[
                        [
                            "timestamp",
                            "action",
                            "reward",
                            "neural_regime",
                            "confidence",
                            "regime_probs",
                        ]
                    ],
                    pd.json_normalize(trades["regime_probs"]).rename(
                        columns={
                            k: f"regime_prob_{k}"
                            for k in ["range", "trend", "defensive"]
                        }
                    ),
                    pd.json_normalize(trades["shap_values"]),
                    trades[extra_cols],
                    features_df,
                ],
                axis=1,
            )

            env.data = trades
            env.mode = mode
            env.policy_type = policy_type

            model.set_env(env)
            model.learn(total_timesteps=timesteps, reset_num_timesteps=False)

            validation = self.validator.validate_model(
                model, algo_type, mode, trades[self.feature_cols].values
            )
            if not validation["valid"]:
                error_msg = f"Modèle {algo_type} ({mode}) invalide après fine-tuning pour {market}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                return model

            save_dir = Path("model/sac_models") / market / mode / policy_type
            save_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = (
                save_dir / f"{algo_type}_{mode}_{policy_type}_finetuned_{timestamp}.zip"
            )
            model.save(save_path)

            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "market": market,
                "algo_type": algo_type,
                "mode": mode,
                "save_path": str(save_path),
                "trades_count": len(trades),
                "mean_reward": validation.get("mean_reward", 0.0),
            }
            self.save_snapshot(f"fine_tune_{algo_type}_{mode}", snapshot, market=market)

            success_msg = f"Ajustement fin {algo_type} réussi pour {market}: mode={mode}, policy_type={policy_type}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "fine_tune_model",
                latency,
                success=True,
                algo_type=algo_type,
                mode=mode,
                mean_reward=validation.get("mean_reward", 0.0),
                market=market,
            )
            return model
        except Exception as e:
            error_msg = f"Erreur ajustement fin {algo_type} pour {market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "fine_tune_model", 0, success=False, error=str(e), market=market
            )
            return None

    async def trigger_training(
        self,
        data: pd.DataFrame,
        modes: List[str] = None,
        algo_types: List[str] = None,
        policy_types: List[str] = None,
        epochs: int = 100000,
        market: str = "ES",
        db_path: str = str(DB_PATH),
    ) -> Dict[str, bool]:
        """
        Déclenche l'entraînement automatique via train_sac_auto.py.

        Args:
            data (pd.DataFrame): Données avec 350/150 features.
            modes (List[str]): Modes à entraîner (trend, range, defensive).
            algo_types (List[str]): Algorithmes à entraîner (sac, ppo, ddpg).
            policy_types (List[str]): Politiques (mlp, transformer).
            epochs (int): Nombre de timesteps.
            market (str): Marché (ex. : ES, MNQ).
            db_path (str): Chemin vers la base de données SQLite.

        Returns:
            Dict[str, bool]: Résultats de l'entraînement.
        """
        start_time = datetime.now()
        try:
            if not self.validate_data(data, market=market):
                error_msg = (
                    f"Données invalides pour déclencher l'entraînement ({market})"
                )
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                return {}

            modes = modes or ["trend", "range", "defensive"]
            algo_types = algo_types or ["sac", "ppo", "ddpg"]
            policy_types = policy_types or ["mlp", "transformer"]

            trainer = TrainSACAuto()
            results = await trainer.auto_train(
                data=data,
                modes=modes,
                algo_types=algo_types,
                policy_types=policy_types,
                epochs=epochs,
                market=market,
            )

            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "market": market,
                "results": results,
                "data_size": len(data),
                "modes": modes,
                "algo_types": algo_types,
                "policy_types": policy_types,
            }
            self.save_snapshot("trigger_training", snapshot, market=market)

            success_msg = f"Entraînement déclenché pour {market}: {results}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "trigger_training",
                latency,
                success=True,
                results_count=len(results),
                market=market,
            )
            return results
        except Exception as e:
            error_msg = f"Erreur déclenchement entraînement pour {market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "trigger_training", 0, success=False, error=str(e), market=market
            )
            return {}


if __name__ == "__main__":

    async def main():
        try:
            learner = AdaptiveLearning(training_mode=True)
            learner.initialize_database()

            env = TradingEnv(
                config_path=str(BASE_DIR / "config/trading_env_config.yaml")
            )
            feature_cols = learner.feature_cols
            data = pd.DataFrame(
                {
                    "timestamp": [datetime.now()],
                    **{col: [np.random.uniform(0, 1)] for col in feature_cols},
                    "vix_es_correlation": [20.0],
                    "atr_14": [0.5],
                    "call_iv_atm": [0.1],
                    "option_skew": [0.01],
                    "predicted_vix": [25.0],
                    "news_sentiment_score": [0.5],
                    "spoofing_score": [0.2],
                    "volume_anomaly": [0.3],
                    "net_gamma": [0.4],
                    "call_wall": [4000.0],
                    "trade_velocity": [100.0],
                    "hft_activity_score": [0.6],
                }
            )

            await learner.store_pattern(
                data,
                action=1.0,
                reward=10.0,
                neural_regime=0,
                confidence=0.8,
                regime_probs={"range": 0.2, "trend": 0.7, "defensive": 0.1},
                shap_values={col: np.random.uniform(0, 1) for col in feature_cols[:50]},
                vix_es_correlation=20.0,
                atr_14=0.5,
                call_iv_atm=0.1,
                option_skew=0.01,
                predicted_vix=25.0,
                news_sentiment_score=0.5,
                spoofing_score=0.2,
                volume_anomaly=0.3,
                net_gamma=0.4,
                call_wall=4000.0,
                trade_velocity=100.0,
                hft_activity_score=0.6,
                market="ES",
            )

            await learner.clean_database(
                max_age_days=30, max_patterns=10000, market="ES"
            )

            success_rate = await learner.last_3_trades_success_rate(market="ES")
            cluster_distance = await learner.cluster_score_distance(data, market="ES")
            print(f"Taux de succès des 3 derniers trades pour ES: {success_rate:.2f}")
            print(f"Distance au cluster le plus proche pour ES: {cluster_distance:.4f}")

            results = await learner.trigger_training(
                data,
                modes=["trend"],
                algo_types=["sac"],
                policy_types=["mlp"],
                market="ES",
            )
            print(f"Entraînement déclenché pour ES: {results}")
        except Exception as e:
            logger.error(f"Erreur test: {str(e)}\n{traceback.format_exc()}")
            raise

    asyncio.run(main())
