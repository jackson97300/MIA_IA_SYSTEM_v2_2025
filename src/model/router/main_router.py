# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/model/router/main_router.py
# Orchestre les régimes de trading (range, trend, défensif, ultra-défensif) en détectant le régime via detect_regime.py
# et en sélectionnant le modèle SAC/PPO/DDPG via train_sac.py.
#
# Version : 2.1.5
# Date : 2025-05-14
#
# Rôle :
# Coordonne la détection des régimes de marché et la sélection des modèles SAC/PPO/DDPG pour prédire des actions de trading.
# Intègre volatilité (méthode 1), données d’options (méthode 2), régimes hybrides (méthode 11), SHAP (méthode 17), et
# prédictions LSTM (méthode 12) via detect_regime.py.
# Intègre SignalResolver pour résoudre les conflits de signaux avant l’appel aux modèles RL, avec propagation du run_id
# dans les snapshots/cache et un fallback explicite en cas d’échec.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.8,<6.0.0, matplotlib>=3.7.0,<4.0.0, asyncio,
#   sqlite3, sklearn>=1.5.0,<2.0.0, boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0, pyyaml>=6.0.0,<7.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/train_sac.py
# - src/model/router/detect_regime.py
# - src/model/utils/mind_dialogue.py
# - src/model/utils/prediction_aggregator.py
# - src/model/utils/model_validator.py
# - src/model/utils/algo_performance_logger.py
# - src/utils/telegram_alert.py
# - src/model/utils/signal_resolver.py
#
# Inputs :
# - config/router_config.yaml, config/feature_sets.yaml, config/algo_config.yaml, config/es_config.yaml via config_manager
# - Données brutes (DataFrame avec 350 features en entraînement ou 150 SHAP features en inférence)
#
# Outputs :
# - Action prédite (float), détails (dict avec régime, probabilités, métriques, signal resolution metadata, etc.)
# - Logs dans data/logs/main_router.log
# - Logs de performance dans data/logs/main_router_performance.csv
# - Snapshots compressés dans data/cache/main_router/*.json.gz
# - Sauvegardes incrémentielles dans data/checkpoints/main_router_*.json.gz
# - Visualisations dans data/figures/main_router/
# - Métriques dans data/market_memory.db
#
# Lien avec SAC :
# Coordonne les modèles SAC/PPO/DDPG de train_sac.py pour live_trading.py
#
# Notes :
# - Gère 350 features (entraînement) ou 150 SHAP features (inférence) via config_manager.
# - Intègre volatilité, données d’options, régimes hybrides, SHAP, prédictions LSTM, et résolution des conflits de signaux.
# - Utilise IQFeed comme source de données via TradingEnv.
# - Implémente retries (max 3, délai 2^attempt secondes) pour les opérations critiques.
# - Sauvegardes incrémentielles toutes les 5 min dans data/checkpoints/ (5 versions).
# - Sauvegardes distribuées toutes les 15 min vers S3 (configurable).
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des données.
# - Tests unitaires disponibles dans tests/test_main_router.py.
# - Validation complète prévue pour juin 2025.
# - SignalResolver intégré avec run_id propagé dans snapshots/cache, fallback explicite en cas d’échec, et score_type
# utilisé comme "intermediate" (extensible à "final", "precheck" pour des
# passes multiples pré-trade/post-trade).

import asyncio
import gzip
import json
import os
import signal
import sqlite3
import time
import traceback
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from loguru import logger
from sklearn.preprocessing import MinMaxScaler

from src.model.router.detect_regime import detect_market_regime_vectorized
from src.model.train_sac import SACTrainer
from src.model.utils.alert_manager import AlertManager
from src.model.utils.algo_performance_logger import AlgoPerformanceLogger
from src.model.utils.config_manager import get_config
from src.model.utils.mind_dialogue import DialogueManager
from src.model.utils.model_validator import ModelValidator
from src.model.utils.prediction_aggregator import PredictionAggregator
from src.model.utils.signal_resolver import SignalResolver
from src.utils.telegram_alert import send_telegram_alert

# Configuration logging
logger.remove()
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOG_DIR = BASE_DIR / "data" / "logs"
CACHE_DIR = BASE_DIR / "data" / "cache" / "main_router"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints"
FIGURE_DIR = BASE_DIR / "data" / "figures" / "main_router"
PERF_LOG_PATH = LOG_DIR / "main_router_performance.csv"
SNAPSHOT_DIR = CACHE_DIR
DB_PATH = BASE_DIR / "data" / "market_memory.db"
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "main_router.log", rotation="10 MB", level="INFO", encoding="utf-8"
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


class MainRouter:
    """
    Orchestre les régimes de trading et sélectionne les modèles SAC/PPO/DDPG.

    Attributes:
        env: Environnement de trading (ex. : TradingEnv).
        policy_type (str): Type de politique ("mlp" ou "transformer").
        training_mode (bool): Si True, utilise 350 features; sinon, 150 SHAP features.
        feature_cols (List[str]): Liste des colonnes de features.
        num_features (int): Nombre de features (350 ou 150).
        trainer: Instance de SACTrainer pour gérer les modèles.
        scaler: Scaler pour normaliser les observations.
        alert_manager: Gestionnaire d’alertes.
        dialogue_manager: Gestionnaire de commandes vocales.
        aggregator: Agrégateur de prédictions.
        validator: Validateur de modèles.
        performance_logger: Logger de performances.
        signal_resolver: Résolveur de conflits de signaux.
        prediction_cache: Cache LRU pour les prédictions.
    """

    def __init__(self, env, policy_type: str = "mlp", training_mode: bool = True):
        """
        Initialise le routeur principal.

        Args:
            env: Environnement de trading.
            policy_type (str): Type de politique ("mlp" ou "transformer").
            training_mode (bool): Si True, utilise 350 features; sinon, 150 SHAP features.
        """
        start_time = datetime.now()
        try:
            self.alert_manager = AlertManager()
            self.env = env
            self.policy_type = policy_type
            self.training_mode = training_mode
            self.num_features = 350 if training_mode else 150
            self.config = get_config(BASE_DIR / "config/router_config.yaml")
            self.algo_config = get_config(BASE_DIR / "config/algo_config.yaml")
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
            self.db_path = DB_PATH
            self.checkpoint_versions = []
            self.log_buffer = []
            self.prediction_cache = OrderedDict()
            self.max_cache_size = MAX_CACHE_SIZE

            self.trainer = SACTrainer(env, policy_type)
            self.scaler = MinMaxScaler()
            self.dialogue_manager = DialogueManager()
            self.aggregator = PredictionAggregator()
            self.validator = ModelValidator()
            self.performance_logger = AlgoPerformanceLogger()
            self.signal_resolver = SignalResolver(
                config_path=str(BASE_DIR / "config/es_config.yaml"), market=env.market
            )

            self.init_db()
            success_msg = (
                f"MainRouter initialisé avec {self.num_features} features, "
                f"policy_type={policy_type}, training_mode={training_mode}"
            )
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "init", latency, success=True, num_features=self.num_features
            )
            self.save_snapshot(
                "init",
                {
                    "config_path": str(BASE_DIR / "config/router_config.yaml"),
                    "timestamp": datetime.now().isoformat(),
                },
            )
            signal.signal(signal.SIGINT, self.handle_sigint)
        except Exception as e:
            error_msg = (
                f"Erreur initialisation MainRouter: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            raise

    def init_db(self):
        """Initialise market_memory.db."""
        try:
            start_time = datetime.now()
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS metrics (
                        timestamp TEXT,
                        regime TEXT,
                        action REAL,
                        reward REAL,
                        sharpe REAL,
                        drawdown REAL,
                        profit_factor REAL,
                        vix_es_correlation REAL,
                        call_iv_atm REAL,
                        option_skew REAL,
                        predicted_vix REAL
                    )
                """
                )
                conn.commit()
            success_msg = "Base de données market_memory.db initialisée"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("init_db", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur initialisation market_memory.db: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance("init_db", 0, success=False, error=str(e))
            raise

    def handle_sigint(self, signum, frame):
        """Gère l'arrêt propre sur SIGINT."""
        global RUNNING
        datetime.now()
        snapshot = {"timestamp": datetime.now().isoformat(), "status": "SIGINT"}
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
        Sauvegarde un instantané des résultats, compressé avec gzip.

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

    def checkpoint(self, data: pd.DataFrame, data_type: str = "router_state") -> None:
        """
        Sauvegarde incrémentielle des données toutes les 5 minutes avec versionnage (5 versions).

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : router_state).
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
            checkpoint_path = (
                CHECKPOINT_DIR / f"main_router_{data_type}_{timestamp}.json.gz"
            )
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

    def cloud_backup(self, data: pd.DataFrame, data_type: str = "router_state") -> None:
        """
        Sauvegarde distribuée des données vers AWS S3 toutes les 15 minutes (configurable).

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : router_state).
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
                f"{self.config['s3_prefix']}main_router_{data_type}_{timestamp}.csv.gz"
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

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ) -> None:
        """
        Journalise les performances dans main_router_performance.csv.

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
                self.checkpoint(log_df, data_type="performance_logs")
                self.cloud_backup(log_df, data_type="performance_logs")
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
    ) -> Any:
        """
        Exécute une fonction avec retries exponentiels.

        Args:
            func (callable): Fonction à exécuter.
            max_attempts (int): Nombre maximum de tentatives.
            delay_base (float): Base pour le délai exponentiel.

        Returns:
            Any: Résultat de la fonction.
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
                    raise
                delay = delay_base**attempt
                warning_msg = f"Tentative {attempt+1} échouée, retry après {delay}s"
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                time.sleep(delay)

    def prune_cache(self):
        """Supprime les entrées anciennes du cache."""
        while len(self.prediction_cache) > self.max_cache_size:
            self.prediction_cache.popitem(last=False)
        logger.debug(f"Cache prédictions purgé, taille: {len(self.prediction_cache)}")

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Valide le DataFrame d’entrée.

        Args:
            df (pd.DataFrame): Données d’entrée.

        Returns:
            bool: True si les données sont valides, False sinon.
        """
        try:
            start_time = datetime.now()
            missing_cols = [col for col in self.feature_cols if col not in df.columns]
            if missing_cols:
                error_msg = f"Colonnes manquantes: {missing_cols}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                return False
            null_count = df[self.feature_cols].isnull().sum().sum()
            confidence_drop_rate = (
                null_count / (len(df) * len(self.feature_cols))
                if (len(df) * len(self.feature_cols)) > 0
                else 0.0
            )
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({null_count} valeurs nulles)"
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
            if null_count > 0:
                error_msg = f"Valeurs nulles détectées dans les features: {null_count}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                return False
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                error_msg = "Colonne 'timestamp' doit être de type datetime"
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
            )
            return True
        except Exception as e:
            error_msg = f"Erreur validation données: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance("validate_data", 0, success=False, error=str(e))
            return False

    def calculate_metrics(
        self, df: pd.DataFrame, action: float, rewards: List[float], current_step: int
    ) -> Dict[str, float]:
        """
        Calcule les métriques sharpe, drawdown, profit_factor.

        Args:
            df (pd.DataFrame): Données d’entrée.
            action (float): Action prédite.
            rewards (List[float]): Récompenses des modèles.
            current_step (int): Étape actuelle.

        Returns:
            Dict[str, float]: Métriques calculées.
        """
        try:
            start_time = datetime.now()
            window_size = min(100, current_step + 1)
            if window_size < 2:
                return {"sharpe": 0.0, "drawdown": 0.0, "profit_factor": 1.0}

            returns = (
                df["close"]
                .iloc[max(0, current_step - window_size + 1) : current_step + 1]
                .pct_change()
                .dropna()
            )
            if len(returns) == 0:
                return {"sharpe": 0.0, "drawdown": 0.0, "profit_factor": 1.0}

            mean_return = returns.mean() * 252
            std_return = returns.std() * np.sqrt(252)
            sharpe = mean_return / std_return if std_return != 0 else 0.0

            prices = df["close"].iloc[
                max(0, current_step - window_size + 1) : current_step + 1
            ]
            cumulative_returns = (1 + returns).cumprod()
            peak = cumulative_returns.cummax()
            drawdown = ((peak - cumulative_returns) / peak).max()

            positive_rewards = [r for r in rewards if r > 0]
            negative_rewards = [abs(r) for r in rewards if r < 0]
            sum_positive = sum(positive_rewards) if positive_rewards else 1e-6
            sum_negative = sum(negative_rewards) if negative_rewards else 1e-6
            profit_factor = sum_positive / sum_negative if sum_negative != 0 else 1.0

            metrics = {
                "sharpe": sharpe,
                "drawdown": drawdown,
                "profit_factor": profit_factor,
            }
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("calculate_metrics", latency, success=True)
            return metrics
        except Exception as e:
            error_msg = f"Erreur calcul métriques: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance("calculate_metrics", 0, success=False, error=str(e))
            return {"sharpe": 0.0, "drawdown": 0.0, "profit_factor": 1.0}

    def resolve_signals(
        self, df: pd.DataFrame, details: Dict, regime: str, current_step: int
    ) -> Dict:
        """
        Résout les conflits entre signaux en utilisant SignalResolver.

        Args:
            df (pd.DataFrame): Données d’entrée.
            details (Dict): Détails du régime (ex. : probabilités, métriques).
            regime (str): Régime de marché détecté.
            current_step (int): Étape actuelle.

        Returns:
            Dict: Métadonnées de la résolution des signaux (score, normalized_score, entropy, conflict_coefficient, etc.).
        """
        try:
            start_time = datetime.now()
            run_id = str(uuid.uuid4())
            # Extraction des signaux bruts
            signals = {
                "regime_trend": 1.0 if regime == "trend" else 0.0,
                "regime_range": 1.0 if regime == "range" else 0.0,
                "regime_defensive": 1.0 if regime == "defensive" else 0.0,
                "microstructure_bullish": (
                    df.get("bid_ask_imbalance", pd.Series([0.0])).iloc[current_step]
                    if "bid_ask_imbalance" in df.columns
                    else 0.0
                ),
                "news_score_positive": (
                    df.get("news_impact_score", pd.Series([0.0])).iloc[current_step]
                    if "news_impact_score" in df.columns
                    else 0.0
                ),
                "qr_dqn_positive": details.get("qr_dqn_quantile_mean", 0.0),
            }
            # Appel à SignalResolver
            score, metadata = self.signal_resolver.resolve_conflict(
                signals=signals,
                normalize=True,
                persist_to_db=True,
                export_to_csv=True,
                run_id=run_id,
                score_type="intermediate",
                mode_debug=True,
            )
            logger.info(
                f"Résolution des signaux (score_type=intermediate): Score={score:.2f}, "
                f"Conflict Coefficient={metadata['conflict_coefficient']:.2f}, "
                f"Entropy={metadata['entropy']:.2f}, Run ID={metadata['run_id']}"
            )
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "resolve_signals",
                latency,
                success=True,
                score=score,
                conflict_coefficient=metadata["conflict_coefficient"],
                run_id=run_id,
            )
            return metadata
        except Exception as e:
            run_id = str(uuid.uuid4())
            error_msg = (
                f"Erreur résolution des signaux: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            latency = (datetime.now() - start_time).total_seconds()
            fallback_metadata = {
                "score": 0.0,
                "normalized_score": None,
                "entropy": 0.0,
                "conflict_coefficient": 0.0,
                "score_type": "intermediate",
                "contributions": {},
                "run_id": run_id,
                "error": str(e),
            }
            logger.info(
                f"Fallback SignalResolver: Score=0.0, Conflict Coefficient=0.0, Run ID={run_id}"
            )
            self.log_performance(
                "resolve_signals",
                latency,
                success=False,
                error=str(e),
                score=0.0,
                conflict_coefficient=0.0,
                run_id=run_id,
            )
            return fallback_metadata

    async def route(
        self, df: pd.DataFrame, current_step: int
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Route la prédiction vers le modèle approprié, en intégrant la résolution des conflits de signaux.

        Args:
            df (pd.DataFrame): Données d’entrée.
            current_step (int): Étape actuelle.

        Returns:
            Tuple[float, Dict[str, Any]]: Action prédite et détails (incluant les métadonnées de signal resolution).
        """
        start_time = datetime.now()
        try:

            async def route_logic():
                if not self.validate_data(df):
                    error_msg = "Données invalides, retour action neutre"
                    logger.error(error_msg)
                    self.alert_manager.send_alert(error_msg, priority=3)
                    send_telegram_alert(error_msg)
                    return 0.0, {"error": "Données invalides"}

                regime, details = await detect_market_regime_vectorized(
                    df, current_step
                )

                # Résolution des conflits de signaux
                signal_metadata = self.resolve_signals(
                    df, details, regime, current_step
                )
                run_id = signal_metadata.get("run_id", str(uuid.uuid4()))

                obs = self.scaler.fit_transform(
                    df[self.feature_cols].iloc[current_step].values.reshape(1, -1)
                )
                obs = obs.flatten()

                cache_key = f"{current_step}_{regime}_{obs.sum()}"
                if cache_key in self.prediction_cache:
                    final_action, cached_details = self.prediction_cache[cache_key]
                    cached_details["signal_metadata"] = signal_metadata
                    self.prediction_cache.move_to_end(cache_key)
                    return final_action, cached_details
                self.prune_cache()

                command = self.dialogue_manager.process_command()
                if command == "switch_to_live_trading":
                    success_msg = "Passage en mode trading live"
                    logger.info(success_msg)
                    self.alert_manager.send_alert(success_msg, priority=2)
                    send_telegram_alert(success_msg)
                elif command == "stop_trading":
                    error_msg = "Arrêt du trading"
                    logger.info(error_msg)
                    self.alert_manager.send_alert(error_msg, priority=3)
                    send_telegram_alert(error_msg)
                    return 0.0, {"error": "Arrêt commandé"}

                actions = []
                rewards = []
                algo_types = ["sac", "ppo", "ddpg"]
                regime_probs = details.get(
                    "regime_probs", {"range": 0.33, "trend": 0.33, "defensive": 0.34}
                )
                for algo_type in algo_types:
                    model = self.trainer.models.get(algo_type, {}).get(regime.lower())
                    if not model:
                        warning_msg = f"Modèle {algo_type} ({regime}) non disponible"
                        logger.warning(warning_msg)
                        self.alert_manager.send_alert(warning_msg, priority=3)
                        send_telegram_alert(warning_msg)
                        continue
                    validation = self.validator.validate_model(
                        model, algo_type, regime, obs
                    )
                    if not validation["valid"]:
                        warning_msg = f"Modèle {algo_type} ({regime}) invalide"
                        logger.warning(warning_msg)
                        self.alert_manager.send_alert(warning_msg, priority=3)
                        send_telegram_alert(warning_msg)
                        continue
                    action, state = model.predict(obs, deterministic=True)
                    reward = model.calculate_reward(
                        obs,
                        action,
                        details.get("profit", 0),
                        details.get("risk", 0),
                        details.get("timing", 0),
                    )
                    actions.append(
                        float(action) * regime_probs.get(regime.lower(), 1.0)
                    )
                    rewards.append(reward)
                    self.performance_logger.log_performance(
                        algo_type,
                        regime,
                        reward,
                        time.time() - start_time,
                        psutil.Process().memory_info().rss / 1024 / 1024,
                    )

                final_action, agg_details = self.aggregator.aggregate_predictions(
                    actions, rewards, regime
                )

                vix = details.get("vix_es_correlation", 0.0)
                spread = details.get("spread", 0.0)
                call_iv_atm = details.get("call_iv_atm", 0.0)
                option_skew = details.get("option_skew", 0.0)
                predicted_vix = details.get("predicted_vix", 0.0)
                thresholds = self.config.get("thresholds", {})
                if (
                    regime == "ultra-defensive"
                    or (
                        vix > thresholds.get("vix_peak_threshold", 30.0)
                        and spread > thresholds.get("spread_explosion_threshold", 0.05)
                    )
                    or call_iv_atm > thresholds.get("iv_high_threshold", 0.25)
                    or abs(option_skew) > thresholds.get("skew_extreme_threshold", 0.1)
                    or predicted_vix > thresholds.get("vix_high_threshold", 25.0)
                ):
                    alert_msg = (
                        f"Mode ultra-defensive activé: VIX={vix:.2f}, spread={spread:.4f}, "
                        f"IV={call_iv_atm:.2f}, Skew={option_skew:.2f}, Predicted VIX={predicted_vix:.2f}"
                    )
                    logger.info(alert_msg)
                    self.alert_manager.send_alert(alert_msg, priority=4)
                    send_telegram_alert(alert_msg)
                    final_action = 0.0

                metrics = self.calculate_metrics(
                    df, final_action, rewards, current_step
                )
                sharpe = metrics["sharpe"]
                drawdown = metrics["drawdown"]
                profit_factor = metrics["profit_factor"]

                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO metrics (timestamp, regime, action, reward, sharpe, drawdown, profit_factor,
                                            vix_es_correlation, call_iv_atm, option_skew, predicted_vix)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            datetime.now().isoformat(),
                            regime,
                            float(final_action),
                            np.mean(rewards) if rewards else 0.0,
                            sharpe,
                            drawdown,
                            profit_factor,
                            vix,
                            call_iv_atm,
                            option_skew,
                            predicted_vix,
                        ),
                    )
                    conn.commit()

                snapshot = {
                    "timestamp": datetime.now().isoformat(),
                    "regime": regime,
                    "regime_probs": regime_probs,
                    "action": float(final_action),
                    "rewards": rewards,
                    "sharpe": sharpe,
                    "drawdown": drawdown,
                    "profit_factor": profit_factor,
                    "vix_es_correlation": vix,
                    "call_iv_atm": call_iv_atm,
                    "option_skew": option_skew,
                    "predicted_vix": predicted_vix,
                    "shap_values": details.get("shap_values", {}),
                    "agg_details": agg_details,
                    "signal_metadata": signal_metadata,
                    "run_id": run_id,
                    "details": details,
                    "num_features": self.num_features,
                }
                self.save_snapshot(f"route_{regime}", snapshot)

                # Sauvegarde incrémentielle
                perf_data = pd.DataFrame(
                    [
                        {
                            "timestamp": datetime.now().isoformat(),
                            "regime": regime,
                            "action": float(final_action),
                            "reward": np.mean(rewards) if rewards else 0.0,
                            "sharpe": sharpe,
                            "drawdown": drawdown,
                            "profit_factor": profit_factor,
                            "vix_es_correlation": vix,
                            "call_iv_atm": call_iv_atm,
                        }
                    ]
                )
                self.checkpoint(perf_data, data_type="route_metrics")
                self.cloud_backup(perf_data, data_type="route_metrics")

                plt.figure(figsize=(10, 6))
                plt.plot(df["close"].iloc[-100:], label="Close", color="blue")
                plt.plot(
                    df["vix_es_correlation"].iloc[-100:],
                    label="VIX-ES Correlation",
                    color="orange",
                )
                plt.axvline(
                    x=current_step % 100,
                    color="red",
                    linestyle="--",
                    label="Current Step",
                )
                plt.title(
                    f"Régime: {regime}, Action: {final_action:.2f}, Sharpe: {sharpe:.2f}\n"
                    f"Probs: Range={regime_probs.get('range', 0.0):.2f}, Trend={regime_probs.get('trend', 0.0):.2f}, "
                    f"Def={regime_probs.get('defensive', 0.0):.2f}\n"
                    f"Signal Score: {signal_metadata.get('normalized_score', 0.0):.2f}, "
                    f"Conflict: {signal_metadata.get('conflict_coefficient', 0.0):.2f}, Run ID: {run_id}"
                )
                plt.xlabel("Time")
                plt.ylabel("Value")
                plt.legend()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plt.savefig(self.figure_dir / f"route_{regime}_{timestamp}.png")
                plt.close()

                self.prediction_cache[cache_key] = (
                    final_action,
                    {
                        "regime": regime,
                        "regime_probs": regime_probs,
                        "rewards": rewards,
                        "sharpe": sharpe,
                        "drawdown": drawdown,
                        "profit_factor": profit_factor,
                        "vix_es_correlation": vix,
                        "call_iv_atm": call_iv_atm,
                        "option_skew": option_skew,
                        "predicted_vix": predicted_vix,
                        "shap_values": details.get("shap_values", {}),
                        "agg_details": agg_details,
                        "signal_metadata": signal_metadata,
                        "run_id": run_id,
                        "details": details,
                    },
                )
                self.prune_cache()

                return final_action, {
                    "regime": regime,
                    "regime_probs": regime_probs,
                    "rewards": rewards,
                    "sharpe": sharpe,
                    "drawdown": drawdown,
                    "profit_factor": profit_factor,
                    "vix_es_correlation": vix,
                    "call_iv_atm": call_iv_atm,
                    "option_skew": option_skew,
                    "predicted_vix": predicted_vix,
                    "shap_values": details.get("shap_values", {}),
                    "agg_details": agg_details,
                    "signal_metadata": signal_metadata,
                    "run_id": run_id,
                    "details": details,
                }

            final_action, details = await route_logic()
            run_id = details.get("run_id", str(uuid.uuid4()))
            latency = (datetime.now() - start_time).total_seconds()
            self.performance_logger.log_performance(
                "router",
                details.get("regime", "unknown"),
                np.mean(details.get("rewards", [0.0])),
                latency,
                psutil.Process().memory_info().rss / 1024 / 1024,
            )
            self.log_performance(
                "route",
                latency,
                success=True,
                regime=details.get("regime", "unknown"),
                num_features=self.num_features,
                run_id=run_id,
            )
            return final_action, details
        except Exception as e:
            run_id = str(uuid.uuid4())
            error_msg = f"Erreur routage: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.performance_logger.log_performance(
                "router",
                "error",
                0.0,
                latency,
                psutil.Process().memory_info().rss / 1024 / 1024,
                error=str(e),
            )
            self.log_performance(
                "route",
                latency,
                success=False,
                error=str(e),
                num_features=self.num_features,
                run_id=run_id,
            )
            return 0.0, {"error": str(e), "run_id": run_id}


async def main():
    from src.envs.trading_env import TradingEnv

    env = TradingEnv(config_path=str(BASE_DIR / "config/trading_env_config.yaml"))
    router = MainRouter(env, policy_type="mlp", training_mode=True)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=100, freq="1min"),
            **{f"feature_{i}": np.random.uniform(0, 1, 100) for i in range(350)},
            "close": np.random.uniform(4000, 5000, 100),
            "vix_es_correlation": [20.0] * 50 + [30.0] * 50,
            "call_iv_atm": [0.1] * 50 + [0.2] * 50,
            "option_skew": [0.01] * 50 + [0.05] * 50,
            "ask_size_level_1": np.random.randint(100, 500, 100),
            "bid_size_level_1": np.random.randint(100, 500, 100),
            "bid_ask_imbalance": np.random.uniform(-0.5, 0.5, 100),
            "news_impact_score": np.random.uniform(0.0, 1.0, 100),
        }
    )
    action, details = await router.route(df, 50)
    print(f"Action routée: {action}, Détails: {details}")


if __name__ == "__main__":
    asyncio.run(main())
