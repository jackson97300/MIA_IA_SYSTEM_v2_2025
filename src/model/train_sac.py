# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/model/train_sac.py
# Entraîne les modèles SAC, PPO, DDPG par régime (range, trend, défensif) avec des méthodes avancées.
#
# Version  : 2.1.5
# Date : 2025-05-16
#
# Rôle :
# Entraîne les modèles SAC, PPO, DDPG pour les régimes de marché (range, trend, défensif) en intégrant des méthodes avancées :
# volatilité/poids SAC (méthode 4), récompenses adaptatives (méthode 5), exploration basée sur la volatilité (méthode 6),
# mémoire contextuelle (méthode 7), fine-tuning (méthode 8), apprentissage en ligne (méthode 10), régimes hybrides
# (méthode 11), LSTM (méthode 12), exploration adaptative (méthode 13), régularisation dynamique (méthode 14),
# curriculum progressif (méthode 15), transfer learning (méthode 16), SHAP (méthode 17), et meta-learning (méthode 18).
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.8,<6.0.0, stable-baselines3>=2.0.0,<3.0.0,
#   pyyaml>=6.0.0,<7.0.0, torch>=2.0.0,<3.0.0, sklearn>=1.5.0,<2.0.0, matplotlib>=3.7.0,<4.0.0,
#   boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0, asyncio, sqlite3
# - src/features/neural_pipeline.py
# - src/model/router/detect_regime.py
# - src/model/adaptive_learner.py
# - src/features/feature_pipeline.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/algo_performance_logger.py
# - src/model/utils/finetune_utils.py
# - src/model/utils/maml_utils.py
# - src/utils/telegram_alert.py
# - data/market_memory.db (table clusters)
#
# Inputs :
# - config/algo_config.yaml (paramètres d’entraînement)
# - config/feature_sets.yaml (350 features et 150 SHAP features)
# - data/features/feature_importance.csv (SHAP features)
# - data/features/feature_importance_cache.csv (cache SHAP)
# - data/market_memory.db (table clusters pour mémoire contextuelle)
# - Données brutes (pd.DataFrame avec 350 features)
#
# Outputs :
# - Modèles entraînés dans model/sac_models/<market>/<mode>/<policy_type>/<algo>_<mode>_<timestamp>.zip
# - Checkpoints dans data/checkpoints/train_sac/<market>/<mode>/<policy_type>/<algo>_<mode>_<timestamp>.zip
# - Logs dans data/logs/train_sac.log
# - Logs de performance dans data/logs/train_sac_performance.csv
# - Snapshots compressés dans data/cache/train_sac/<market>/*.json.gz
# - Visualisations dans data/figures/train_sac/<market>/<mode>/
#
# Notes :
# - Intègre 13 méthodes avancées (4-18) pour un entraînement robuste et adaptatif.
# - Valide 350 features pour l’entraînement et 150 SHAP features pour l’inférence avec fallback (cache ou liste statique).
# - Utilise IQFeed comme source de données via TradingEnv.
# - Supporte le trading multi-marchés (ex. : ES, MNQ) with features spécifiques (ex. : gex_es, gex_mnq).
# - Implémente sauvegardes incrémentielles (5 min), distribuées (15 min, AWS S3), et versionnées (5 versions).
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des données.
# - Génère des visualisations matplotlib pour les rewards SAC, feature importance, et clusters.
# - Configure PyTorch pour l’A100 avec parallélisation des calculs LSTM et SHAP.
# - Modularise fine-tuning et meta-learning dans finetune_utils.py et maml_utils.py.
# - Tests unitaires disponibles dans tests/test_train_sac.py.
# - TODO : Intégration future avec Bloomberg ou autres fournisseurs.
# - Mise à jour (2025-05-16) : Validation renforcée avec validate_obs_t, optimisation mémoire (float32/int8),
#   parallélisation asynchrone des prédictions LSTM, intégration iv_skew/iv_term_structure/bid_ask_imbalance/trade_aggressiveness,
#   métriques Prometheus pour reward/regime_probs/shap_weights, validation schéma DB, réduction clustering.

import asyncio
import gzip
import json
import os
import sqlite3
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
from loguru import logger
from sklearn.cluster import KMeans
from stable_baselines3 import DDPG, PPO, SAC

from src.envs.trading_env import TradingEnv
from src.features.neural_pipeline import NeuralPipeline
from src.model.router.detect_regime import MarketRegimeDetector
from src.model.utils.alert_manager import AlertManager
from src.model.utils.algo_performance_logger import AlgoPerformanceLogger
from src.model.utils.config_manager import get_config
from src.model.utils.finetune_utils import finetune_model, online_learning
from src.model.utils.maml_utils import apply_prototypical_networks
from src.utils.telegram_alert import send_telegram_alert

# Configuration logging
logger.remove()
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOG_DIR = BASE_DIR / "data" / "logs"
CACHE_DIR = BASE_DIR / "data" / "cache" / "train_sac"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints" / "train_sac"
FIGURE_DIR = BASE_DIR / "data" / "figures" / "train_sac"
PERF_LOG_PATH = LOG_DIR / "train_sac_performance.csv"
SNAPSHOT_DIR = CACHE_DIR
MODEL_DIR = BASE_DIR / "model" / "sac_models"
DB_PATH = BASE_DIR / "data" / "market_memory.db"
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)
logger.add(LOG_DIR / "train_sac.log", rotation="10 MB", level="INFO", encoding="utf-8")
logger.add(lambda msg: print(msg, end=""), level="INFO")

# Constantes
MAX_RETRIES = 3
RETRY_DELAY = 2.0
MAX_CACHE_SIZE = 1000
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB

# Variable pour gérer l'arrêt propre
RUNNING = True

# Ajout : Métriques Prometheus
from src.monitoring.prometheus_metrics import Gauge
reward_metric = Gauge(
    name="sac_reward",
    description="Récompense moyenne SAC",
    labelnames=["market", "mode"],
)
regime_probs_metric = Gauge(
    name="regime_probs",
    description="Probabilités des régimes de marché",
    labelnames=["market", "mode"],
)
shap_weights_metric = Gauge(
    name="shap_weights",
    description="Poids SHAP des features",
    labelnames=["market", "mode", "feature"],
)

# Ajout : Fonction de validation des observations
def validate_obs_t(df: pd.DataFrame, context: str = "train_sac") -> bool:
    """Valide les observations selon les plages définies."""
    feature_ranges = {
        "bid_ask_imbalance": (-1, 1),
        "trade_aggressiveness": (-1, 1),
        "iv_skew": (-0.5, 0.5),
        "iv_term_structure": (0, 0.05),
        "option_skew": (-1, 1),
        "news_impact_score": (-1, 1),
        "vix_es_correlation": (-1, 1),
        "call_iv_atm": (0, 0.5),
    }
    for feature, (min_val, max_val) in feature_ranges.items():
        if feature in df.columns:
            if not df[feature].between(min_val, max_val).all():
                logger.warning(f"Feature {feature} hors plage [{min_val}, {max_val}] dans {context}")
                return False
    return True

class SACTrainer:
    """Entraîne les modèles SAC, PPO, DDPG par régime avec méthodes avancées."""

    def __init__(self, env: TradingEnv, policy_type: str = "mlp"):
        """
        Initialise le formateur SAC/PPO/DDPG.

        Args:
            env (TradingEnv): Environnement de trading.
            policy_type (str): Type de politique ("mlp" ou "transformer").
        """
        start_time = datetime.now()
        try:
            self.alert_manager = AlertManager()
            self.env = env
            self.policy_type = policy_type
            self.config = get_config(BASE_DIR / "config/algo_config.yaml")
            self.performance_logger = AlgoPerformanceLogger()
            self.models = {"sac": {}, "ppo": {}, "ddpg": {}}
            self.neural_pipeline = NeuralPipeline(
                window_size=self.config.get("neural_pipeline", {}).get("window_size", 50),
                base_features=350,
                config_path=str(BASE_DIR / "config/algo_config.yaml"),
            )
            self.lstm_cache = OrderedDict()
            self.max_cache_size = MAX_CACHE_SIZE
            self.checkpoint_versions = []
            self.log_buffer = []
            success_msg = f"SACTrainer initialisé avec policy_type={policy_type}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("init", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur initialisation SACTrainer: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            raise

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ) -> None:
        """
        Journalise les performances dans train_sac_performance.csv.

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
            if memory_usage > 800:  # Modification : Seuil réduit de 1024 à 800 MB
                alert_msg = f"ALERTE: Usage mémoire élevé ({memory_usage:.2f} MB)"
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=5)
                send_telegram_alert(alert_msg)
                # Ajout : Réduction dynamique du cache
                self.max_cache_size = max(100, self.max_cache_size // 2)
                while len(self.lstm_cache) > self.max_cache_size:
                    self.lstm_cache.popitem(last=False)
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
            if len(self.log_buffer) >= self.config.get("logging", {}).get("buffer_size", 100):
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
                self.checkpoint(log_df, data_type="performance_logs", market=kwargs.get("market", "ES"))
                self.cloud_backup(log_df, data_type="performance_logs", market=kwargs.get("market", "ES"))
                self.log_buffer = []
            logger.info(f"Performance journalisée pour {operation}. CPU: {cpu_percent}%")
        except Exception as e:
            error_msg = f"Erreur journalisation performance: {str(e)}\n{traceback.format_exc()}"
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
            snapshot_type (str): Type de snapshot (ex. : init, train_sac).
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
                alert_msg = f"Snapshot size {file_size:.2f} MB exceeds 1 MB pour {market}"
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Snapshot {snapshot_type} sauvegardé pour {market}: {save_path}"
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
        self, data: pd.DataFrame, data_type: str = "train_sac_state", market: str = "ES"
    ) -> None:
        """
        Sauvegarde incrémentielle des données toutes les 5 minutes avec versionnage (5 versions).

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : train_sac_state).
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
            checkpoint_path = checkpoint_dir / f"train_sac_{data_type}_{timestamp}.json.gz"

            def write_checkpoint():
                with gzip.open(checkpoint_path, "wt", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, indent=4)
                data.to_csv(
                    str(checkpoint_path).replace(".json.gz", ".csv"),
                    index=False,
                    encoding="utf-8",
                )

            self.with_retries(write_checkpoint)
            self.checkpoint_versions.append(str(checkpoint_path))
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
        self, data: pd.DataFrame, data_type: str = "train_sac_state", market: str = "ES"
    ) -> None:
        """
        Sauvegarde distribuée des données vers AWS S3 toutes les 15 minutes (configurable).

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : train_sac_state).
            market (str): Marché (ex. : ES, MNQ).
        """
        try:
            start_time = datetime.now()
            if not self.config.get("s3_bucket"):
                warning_msg = f"S3 bucket non configuré, sauvegarde cloud ignorée pour {market}"
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.config['s3_prefix']}train_sac_{data_type}_{market}_{timestamp}.csv.gz"
            temp_path = CHECKPOINT_DIR / market / f"temp_s3_{timestamp}.csv.gz"
            temp_path.parent.mkdir(exist_ok=True)

            def write_temp():
                data.to_csv(temp_path, compression="gzip", index=False, encoding="utf-8")

            self.with_retries(write_temp)
            s3_client = boto3.client("s3")

            def upload_s3():
                s3_client.upload_file(str(temp_path), self.config["s3_bucket"], backup_path)

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

    def validate_data(
        self, data: pd.DataFrame, shap_features: bool = False, market: str = "ES"
    ) -> bool:
        """
        Valide que les données contiennent 350 features (entraînement) ou 150 SHAP features (inférence).

        Args:
            data (pd.DataFrame): Données à valider.
            shap_features (bool): Si True, valide les 150 SHAP features; sinon, valide les 350 features.
            market (str): Marché (ex. : ES, MNQ).

        Returns:
            bool: True si les données sont valides, False sinon.
        """
        start_time = datetime.now()
        try:
            feature_sets_path = BASE_DIR / "config/feature_sets.yaml"
            features_config = get_config(feature_sets_path)
            expected_cols = (
                features_config.get("training", {}).get("features", [])[:350]
                if not shap_features
                else features_config.get("inference", {}).get("shap_features", [])[:150]
            )
            expected_len = 150 if shap_features else 350
            if len(expected_cols) != expected_len:
                error_msg = f"Attendu {expected_len} features pour {market}, trouvé {len(expected_cols)}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                return False

            # Modification : Conversion en float32
            data = data.astype({col: np.float32 for col in data.columns if col != "timestamp"})

            missing_cols = [col for col in expected_cols if col not in data.columns]
            if missing_cols:
                warning_msg = f"Colonnes manquantes pour {market} ({'SHAP' if shap_features else 'full'}): {missing_cols}"
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                for col in missing_cols:
                    data[col] = 0.0

            critical_cols = [
                "bid_size_level_1",
                "ask_size_level_1",
                "trade_frequency_1s",
                "spread_avg_1min",
                "close",
            ]
            null_count = 0
            for col in critical_cols:
                if col in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        error_msg = f"Colonne {col} n'est pas numérique pour {market}: {data[col].dtype}"
                        logger.error(error_msg)
                        self.alert_manager.send_alert(error_msg, priority=3)
                        send_telegram_alert(error_msg)
                        return False
                    null_count += data[col].isna().sum()
                    if data[col].isna().any():
                        data[col] = (
                            data[col]
                            .interpolate(method="linear", limit_direction="both")
                            .fillna(0.0)
                        )
                    if col in [
                        "bid_size_level_1",
                        "ask_size_level_1",
                        "trade_frequency_1s",
                        "spread_avg_1min",
                    ]:
                        if (data[col] <= 0).any():
                            data[col] = data[col].clip(lower=1e-6)
                            logger.warning(
                                f"Valeurs non positives dans {col} pour {market}, corrigées à 1e-6"
                            )
            # Ajout : Validation des nouvelles features
            feature_ranges = {
                "bid_ask_imbalance": (-1, 1),
                "trade_aggressiveness": (-1, 1),
                "iv_skew": (-0.5, 0.5),
                "iv_term_structure": (0, 0.05),
                "option_skew": (-1, 1),
                "news_impact_score": (-1, 1),
            }
            for feature, (min_val, max_val) in feature_ranges.items():
                if feature in data.columns:
                    if not data[feature].between(min_val, max_val).all():
                        warning_msg = f"Feature {feature} hors plage [{min_val}, {max_val}] pour {market}"
                        logger.warning(warning_msg)
                        self.alert_manager.send_alert(warning_msg, priority=2)
                        send_telegram_alert(warning_msg)
                        return False
            # Ajout : Intégration de validate_obs_t
            if not validate_obs_t(data, context="train_sac"):
                warning_msg = f"Échec validation obs_t pour {market}"
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=2)
                send_telegram_alert(warning_msg)
                return False
            confidence_drop_rate = (
                null_count / (len(data) * len(critical_cols))
                if (len(data) * len(critical_cols)) > 0
                else 0.0
            )
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé pour {market}: {confidence_drop_rate:.2f} ({null_count} valeurs nulles)"
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
            if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
                error_msg = f"Colonne 'timestamp' doit être de type datetime pour {market}"
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

    def load_clusters(
        self, db_path: str = str(DB_PATH), market: str = "ES"
    ) -> np.ndarray:
        """
        Charge les clusters depuis market_memory.db (table clusters) pour un marché spécifique.

        Args:
            db_path (str): Chemin vers market_memory.db.
            market (str): Marché (ex. : ES, MNQ).

        Returns:
            np.ndarray: Labels des clusters.
        """
        start_time = datetime.now()
        try:
            def fetch_clusters():
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                # Ajout : Validation du schéma
                cursor.execute("PRAGMA table_info(clusters)")
                columns = {col[1]: col[2] for col in cursor.fetchall()}
                if not all(col in columns for col in ["cluster", "market"]) or columns["cluster"] != "INTEGER":
                    raise ValueError(f"Schéma invalide dans market_memory.db (table clusters) pour {market}")
                # Modification : Limiter aux 1000 derniers points
                query = "SELECT cluster FROM clusters WHERE market = ? ORDER BY timestamp DESC LIMIT 1000"
                clusters = pd.read_sql(query, conn, params=(market,))["cluster"].values
                conn.close()
                if len(clusters) == 0:
                    raise ValueError(f"Aucun cluster trouvé pour {market}")
                return clusters

            clusters = self.with_retries(fetch_clusters)
            if clusters is None or len(clusters) == 0:
                warning_msg = f"Aucun cluster trouvé pour {market}, retour à un tableau vide"
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                return np.zeros(1, dtype=np.int8)  # Modification : int8
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("load_clusters", latency, success=True, market=market)
            return clusters.astype(np.int8)  # Modification : int8
        except Exception as e:
            error_msg = f"Erreur chargement clusters pour {market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "load_clusters", 0, success=False, error=str(e), market=market
            )
            return np.zeros(1, dtype=np.int8)  # Modification : int8

    def configure_gpu(self):
        """
        Configure PyTorch pour utiliser l’A100 pour les calculs parallèles.
        """
        start_time = datetime.now()
        try:
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
                torch.backends.cudnn.benchmark = True
                success_msg = "GPU A100 configuré pour PyTorch"
                logger.info(success_msg)
                self.alert_manager.send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)
            else:
                warning_msg = "Aucun GPU disponible, utilisation du CPU"
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("configure_gpu", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur configuration GPU: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance("configure_gpu", 0, success=False, error=str(e))

    def load_shap_fallback(
        self, shap_file: str = str(BASE_DIR / "data/features/feature_importance.csv")
    ) -> List[str]:
        """
        Charge les 150 SHAP features avec fallback (cache ou liste statique).

        Args:
            shap_file (str): Chemin vers feature_importance.csv.

        Returns:
            List[str]: Liste des 150 SHAP features.
        """
        start_time = datetime.now()
        try:
            if os.path.exists(shap_file):
                shap_df = pd.read_csv(shap_file)
                features = shap_df["feature"].head(150).tolist()
                success_msg = f"SHAP features chargées depuis {shap_file}: {len(features)} features"
                logger.info(success_msg)
                self.alert_manager.send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)
                return features
            cache_file = BASE_DIR / "data/features/feature_importance_cache.csv"
            if os.path.exists(cache_file):
                shap_df = pd.read_csv(cache_file)
                features = shap_df["feature"].head(150).tolist()
                success_msg = f"SHAP features chargées depuis cache {cache_file}: {len(features)} features"
                logger.info(success_msg)
                self.alert_manager.send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)
                return features
            feature_sets_path = BASE_DIR / "config/feature_sets.yaml"
            features_config = get_config(feature_sets_path)
            features = features_config.get("inference", {}).get("shap_features", [])[:150]
            warning_msg = f"SHAP features non disponibles, utilisation de la liste statique: {len(features)} features"
            logger.warning(warning_msg)
            self.alert_manager.send_alert(warning_msg, priority=3)
            send_telegram_alert(warning_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("load_shap_fallback", latency, success=True)
            return features
        except Exception as e:
            error_msg = f"Erreur chargement SHAP fallback: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance("load_shap_fallback", 0, success=False, error=str(e))
            return []

    def initialize_model(
        self, algo_type: str, mode: str, market: str = "ES"
    ) -> Optional[Any]:
        """
        Initialise un modèle SAC, PPO, ou DDPG pour un marché spécifique.

        Args:
            algo_type (str): Type d’algorithme ("sac", "ppo", "ddpg").
            mode (str): Mode du modèle ("trend", "range", "defensive").
            market (str): Marché (ex. : ES, MNQ).

        Returns:
            Optional[Any]: Modèle initialisé ou None si échec.
        """
        start_time = datetime.now()
        try:
            self.configure_gpu()
            model_class = {"sac": SAC, "ppo": PPO, "ddpg": DDPG}[algo_type]
            model = model_class(
                policy="MlpPolicy" if self.policy_type == "mlp" else "CnnPolicy",
                env=self.env,
                learning_rate=self.config.get(algo_type, {}).get("learning_rate", 0.0001),
                verbose=0,
            )
            success_msg = f"Modèle {algo_type} initialisé pour {market}/{mode}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "initialize_model", latency, success=True, market=market, mode=mode
            )
            return model
        except Exception as e:
            error_msg = f"Erreur initialisation modèle {algo_type} ({market}/{mode}): {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "initialize_model",
                0,
                success=False,
                error=str(e),
                market=market,
                mode=mode,
            )
            return None

    async def integrate_lstm_predictions(self, data: pd.DataFrame, market: str = "ES") -> pd.DataFrame:
        """
        Intègre des prédictions LSTM pour les features dynamiques (méthode 12).

        Args:
            data (pd.DataFrame): Données d’entraînement.
            market (str): Marché (ex. : ES, MNQ).

        Returns:
            pd.DataFrame: Données enrichies avec prédictions LSTM.
        """
        start_time = datetime.now()
        try:
            # Modification : Conversion en float32
            data = data.astype({col: np.float32 for col in data.columns if col != "timestamp"})
            cache_key = f"{market}_{data['timestamp'].iloc[-1].isoformat()}_{len(data)}"
            if cache_key in self.lstm_cache:
                lstm_features = self.lstm_cache[cache_key]
                self.lstm_cache.move_to_end(cache_key)
            else:
                # Modification : Ajout de nouvelles features dans les prédictions LSTM
                lstm_features = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.neural_pipeline.generate_lstm_predictions(
                        data,
                        features=[
                            f"gex_{market.lower()}",
                            "vix_es_correlation",
                            "atr_14",
                            "iv_skew",
                            "iv_term_structure",
                            "bid_ask_imbalance",
                            "trade_aggressiveness",
                        ],
                    ),
                )
                self.lstm_cache[cache_key] = lstm_features
                while len(self.lstm_cache) > self.max_cache_size:
                    self.lstm_cache.popitem(last=False)
            data["neural_dynamic_feature_1"] = lstm_features.get("predicted_vix", np.zeros(len(data), dtype=np.float32))
            # Ajout : Feature supplémentaire pour iv_skew
            data["neural_dynamic_feature_2"] = lstm_features.get("predicted_iv_skew", np.zeros(len(data), dtype=np.float32))
            success_msg = f"Prédictions LSTM intégrées pour neural_dynamic_feature_1 et _2 ({market})"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "integrate_lstm_predictions", latency, success=True, market=market
            )
            return data
        except Exception as e:
            error_msg = f"Erreur intégration LSTM pour {market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "integrate_lstm_predictions",
                0,
                success=False,
                error=str(e),
                market=market,
            )
            return data

    async def save_checkpoint(
        self,
        model: Any,
        mode: str,
        policy_type: str,
        market: str = "ES",
        version_count: int = 5,
    ) -> None:
        """
        Sauvegarde incrémentielle des poids du modèle.

        Args:
            model (Any): Modèle à sauvegarder (SAC, PPO, DDPG).
            mode (str): Mode du modèle ("trend", "range", "defensive").
            policy_type (str): Type de politique ("mlp", "transformer").
            market (str): Marché (ex. : ES, MNQ).
            version_count (int): Nombre de versions à conserver.
        """
        start_time = datetime.now()
        try:
            checkpoint_dir = CHECKPOINT_DIR / market / mode / policy_type
            checkpoint_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = checkpoint_dir / f"sac_{mode}_{policy_type}_{timestamp}.zip"
            model.save(str(checkpoint_path))

            checkpoints = sorted(checkpoint_dir.glob("*.zip"), key=os.path.getmtime)
            while len(checkpoints) > version_count:
                os.remove(checkpoints.pop(0))

            snapshot = {
                "market": market,
                "mode": mode,
                "policy_type": policy_type,
                "checkpoint_path": str(checkpoint_path),
                "timestamp": timestamp,
            }
            self.save_snapshot(f"checkpoint_{market}_{mode}", snapshot, market=market)

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Checkpoint sauvegardé pour {market}/{mode}: {checkpoint_path}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "save_checkpoint", latency, success=True, market=market, mode=mode
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde checkpoint pour {market}/{mode}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "save_checkpoint",
                0,
                success=False,
                error=str(e),
                market=market,
                mode=mode,
            )

    def generate_visualizations(
        self, data: pd.DataFrame, rewards: List[float], mode: str, market: str = "ES"
    ) -> None:
        """
        Génère des visualisations matplotlib pour les rewards SAC et feature importance.

        Args:
            data (pd.DataFrame): Données d’entraînement.
            rewards (List[float]): Récompenses observées.
            mode (str): Mode du modèle ("trend", "range", "defensive").
            market (str): Marché (ex. : ES, MNQ).
        """
        start_time = datetime.now()
        try:
            # Modification : Conversion des rewards en float32
            rewards = np.array(rewards, dtype=np.float32)
            fig_dir = FIGURE_DIR / market / mode
            fig_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            plt.figure(figsize=(10, 6))
            plt.plot(rewards, label="Rewards")
            plt.title(f"Rewards SAC pour {market}/{mode}")
            plt.xlabel("Étape")
            plt.ylabel("Reward")
            plt.legend()
            plt.savefig(fig_dir / f"rewards_{mode}_{timestamp}.png")
            plt.close()

            shap_file = BASE_DIR / "data/features/feature_importance.csv"
            if os.path.exists(shap_file):
                shap_df = pd.read_csv(shap_file).head(10)
                plt.figure(figsize=(10, 6))
                plt.bar(shap_df["feature"], shap_df["importance"])
                plt.title(f"Top 10 SHAP Features pour {market}/{mode}")
                plt.xlabel("Feature")
                plt.ylabel("Importance")
                plt.xticks(rotation=45)
                plt.savefig(fig_dir / f"shap_features_{mode}_{timestamp}.png")
                plt.close()

            # Ajout : Visualisation des nouvelles features
            if "bid_ask_imbalance" in data.columns:
                plt.figure(figsize=(10, 6))
                plt.hist(data["bid_ask_imbalance"], bins=50, alpha=0.7, label="Bid-Ask Imbalance")
                plt.title(f"Distribution de bid_ask_imbalance pour {market}/{mode}")
                plt.xlabel("Valeur")
                plt.ylabel("Fréquence")
                plt.legend()
                plt.savefig(fig_dir / f"bid_ask_imbalance_{mode}_{timestamp}.png")
                plt.close()

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Visualisations générées pour {market}/{mode}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "generate_visualizations",
                latency,
                success=True,
                market=market,
                mode=mode,
            )
        except Exception as e:
            error_msg = f"Erreur génération visualisations pour {market}/{mode}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "generate_visualizations",
                0,
                success=False,
                error=str(e),
                market=market,
                mode=mode,
            )

    async def train(
        self, data: pd.DataFrame, total_timesteps: int = 100000, mode: str = "trend"
    ) -> Optional[Any]:
        """
        Entraîne un modèle SAC, PPO, ou DDPG avec méthodes avancées.

        Args:
            data (pd.DataFrame): Données d’entraînement.
            total_timesteps (int): Nombre total de timesteps.
            mode (str): Mode du modèle ("trend", "range", "defensive").

        Returns:
            Optional[Any]: Modèle entraîné ou None si échec.
        """
        return await self.train_multi_market(data, total_timesteps, mode, market="ES")

    async def train_multi_market(
        self,
        data: pd.DataFrame,
        total_timesteps: int = 100000,
        mode: str = "trend",
        market: str = "ES",
    ) -> Optional[Any]:
        """
        Entraîne un modèle SAC, PPO, ou DDPG pour un marché spécifique avec méthodes avancées.

        Args:
            data (pd.DataFrame): Données d’entraînement.
            total_timesteps (int): Nombre total de timesteps.
            mode (str): Mode du modèle ("trend", "range", "defensive").
            market (str): Marché (ex. : ES, MNQ).

        Returns:
            Optional[Any]: Modèle entraîné ou None si échec.
        """
        start_time = datetime.now()
        try:
            if not self.validate_data(
                data, shap_features=(self.policy_type == "transformer"), market=market
            ):
                error_msg = f"Données invalides pour entraînement ({market}/{mode})"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                return None

            # Modification : Conversion en float32
            data = data.astype({col: np.float32 for col in data.columns if col != "timestamp"})

            # Ajout : Pré-calcul des features stables
            stable_features = {}
            for feature in ["vix_es_correlation", f"gex_{market.lower()}"]:
                if feature in data.columns:
                    stable_features[feature] = data[feature].mean()

            data = await self.integrate_lstm_predictions(data, market)

            clusters = self.load_clusters(market=market)
            if len(clusters) > 1:
                data["cluster"] = KMeans(n_clusters=10).fit_predict(
                    data[[f"gex_{market.lower()}", "vix_es_correlation"]]
                ).astype(np.int8)  # Modification : int8
            else:
                data["cluster"] = np.zeros(len(data), dtype=np.int8)  # Modification : int8

            model = self.initialize_model("sac", mode, market)
            if model is None:
                return None

            detector = MarketRegimeDetector(
                training_mode=(self.policy_type != "transformer")
            )
            regime_probs = (await detector.detect(data, 0))[1].get(
                "regime_probs", {"trend": 0.4, "range": 0.4, "defensive": 0.2}
            )

            shap_features = self.load_shap_fallback()
            shap_weights = {f: 1.0 for f in shap_features}
            shap_file = BASE_DIR / "data/features/feature_importance.csv"
            if os.path.exists(shap_file):
                shap_df = pd.read_csv(shap_file)
                shap_weights.update(
                    {
                        row["feature"]: row.get("importance", 1.0)
                        for _, row in shap_df.iterrows()
                    }
                )

            shared_weights = None
            for other_mode in ["trend", "range", "defensive"]:
                if other_mode != mode and other_mode in self.models["sac"]:
                    shared_weights = self.models["sac"][other_mode].policy.state_dict()
                    break
            if shared_weights:
                model.policy.load_state_dict(shared_weights)
                success_msg = f"Transfer learning appliqué depuis {other_mode} pour {market}/{mode}"
                logger.info(success_msg)
                self.alert_manager.send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)

            for complexity in range(1, 4):
                timesteps = total_timesteps * complexity // 3
                volatility = stable_features.get("vix_es_correlation", data["vix_es_correlation"].mean())
                # Ajout : Ajustement basé sur bid_ask_imbalance
                imbalance = data["bid_ask_imbalance"].mean() if "bid_ask_imbalance" in data.columns else 0.0
                ent_coef = (
                    0.1 * (1.0 if mode == "range" else 0.5) * (1 + volatility / 20) * (1 + abs(imbalance))
                )
                l2_reg = 0.01 * (1 + volatility / 10) * (1 + abs(imbalance) / 2)
                dropout = 0.2 * (1 + volatility / 15) * (1 + abs(imbalance) / 2)

                data["reward"] = (
                    data.get(f"profit_{market.lower()}", data.get("profit", 0))
                    + data.get("news_impact_score", 0)
                    + data.get("neural_dynamic_feature_1", data.get("predicted_vix", 0))
                    + data.get("neural_dynamic_feature_2", 0)  # Ajout : Feature iv_skew
                )

                model = finetune_model(
                    data,
                    ent_coef=ent_coef,
                    config_path=str(BASE_DIR / "config/algo_config.yaml"),
                    mode=mode,
                    policy_type=self.policy_type,
                    batch_size=100,
                    total_timesteps=timesteps // 3,
                )
                if model is None:
                    error_msg = f"Échec fine-tuning pour {market}/{mode} (complexity={complexity})"
                    logger.error(error_msg)
                    self.alert_manager.send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    return None

                model = apply_prototypical_networks(
                    model,
                    data,
                    config_path=str(BASE_DIR / "config/algo_config.yaml"),
                    mode=mode,
                    policy_type=self.policy_type,
                    batch_size=100,
                )
                if model is None:
                    error_msg = f"Échec meta-learning pour {market}/{mode} (complexity={complexity})"
                    logger.error(error_msg)
                    self.alert_manager.send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    return None

                def train_with_complexity():
                    model.set_parameters(
                        {"l2_factor": l2_reg, "dropout_rate": min(dropout, 0.5)}
                    )
                    model.learn(total_timesteps=timesteps)
                    return model

                model = self.with_retries(train_with_complexity)
                if model is None:
                    error_msg = f"Échec entraînement curriculum (complexity={complexity}) pour {market}/{mode}"
                    logger.error(error_msg)
                    self.alert_manager.send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    return None

                model = online_learning(
                    data,
                    model,
                    self.env,
                    batch_size=32,
                    total_timesteps=timesteps // 10,
                    learning_rate=self.config.get("sac", {}).get("learning_rate", 0.0001),
                )
                if model is None:
                    error_msg = f"Échec apprentissage en ligne pour {market}/{mode} (complexity={complexity})"
                    logger.error(error_msg)
                    self.alert_manager.send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    return None

                await self.save_checkpoint(model, mode, self.policy_type, market)

            model_dir = MODEL_DIR / market / mode / self.policy_type
            model_dir.mkdir(exist_ok=True)
            model_path = (
                model_dir
                / f"sac_{mode}_{self.policy_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            )
            model.save(str(model_path))
            success_msg = f"Modèle SAC sauvegardé pour {market}/{mode}: {model_path}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)

            self.generate_visualizations(data, data["reward"].tolist(), mode, market)

            # Ajout : Métriques Prometheus
            reward_metric.labels(market=market, mode=mode).set(data["reward"].mean())
            for regime, prob in regime_probs.items():
                regime_probs_metric.labels(market=market, mode=regime).set(prob)
            for feature, weight in shap_weights.items():
                shap_weights_metric.labels(market=market, mode=mode, feature=feature).set(weight)

            self.models["sac"][mode] = model
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "train_sac_multi_market",
                latency,
                success=True,
                market=market,
                mode=mode,
                policy_type=self.policy_type,
                timesteps=total_timesteps,
                num_rows=len(data),
            )
            self.save_snapshot(
                f"train_sac_multi_market_{market}",
                {
                    "market": market,
                    "mode": mode,
                    "policy_type": self.policy_type,
                    "timesteps": total_timesteps,
                    "num_rows": len(data),
                    "model_path": str(model_path),
                },
                market=market,
            )
            return model

        except Exception as e:
            error_msg = f"Erreur dans train_sac_multi_market pour {market}/{mode}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "train_sac_multi_market",
                latency,
                success=False,
                error=str(e),
                market=market,
                mode=mode,
                policy_type=self.policy_type,
            )
            return None

async def train_sac(data: pd.DataFrame, total_timesteps: int = 100000) -> Optional[SAC]:
    """
    Entraîne un modèle SAC avec clustering et méthodes avancées.

    Args:
        data (pd.DataFrame): Données d’entraînement.
        total_timesteps (int): Nombre total de timesteps.

    Returns:
        Optional[SAC]: Modèle SAC entraîné ou None si échec.
    """
    start_time = datetime.now()
    try:
        env = TradingEnv(config_path=str(BASE_DIR / "config/trading_env_config.yaml"))
        trainer = SACTrainer(env, policy_type="mlp")
        model = await trainer.train_multi_market(
            data, total_timesteps=total_timesteps, mode="trend", market="ES"
        )
        if model is None:
            error_msg = "Échec entraînement SAC pour trend/ES"
            logger.error(error_msg)
            trainer.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            return None
        latency = (datetime.now() - start_time).total_seconds()
        success_msg = "Entraînement SAC terminé pour trend/ES"
        logger.info(success_msg)
        trainer.alert_manager.send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
        trainer.log_performance("train_sac", latency, success=True, num_rows=len(data))
        return model
    except Exception as e:
        error_msg = f"Erreur dans train_sac: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        trainer.alert_manager.send_alert(error_msg, priority=5)
        send_telegram_alert(error_msg)
        latency = (datetime.now() - start_time).total_seconds()
        trainer.log_performance("train_sac", latency, success=False, error=str(e))
        return None

if __name__ == "__main__":
    async def main():
        try:
            env = TradingEnv(config_path=str(BASE_DIR / "config/trading_env_config.yaml"))
            trainer = SACTrainer(env, policy_type="mlp")
            data = pd.DataFrame(
                {
                    "timestamp": pd.date_range("2025-05-13 09:00", periods=100, freq="1min"),
                    **{f"feature_{i}": np.random.uniform(0, 1, 100) for i in range(350)},
                    "close": np.random.uniform(4000, 5000, 100),
                    "vix_es_correlation": [20.0] * 50 + [30.0] * 50,
                    "news_impact_score": np.random.uniform(-1, 1, 100),
                    "predicted_vix": np.random.uniform(15, 25, 100),
                    "gex_es": np.random.uniform(-1000, 1000, 100),
                    "gex_mnq": np.random.uniform(-1000, 1000, 100),
                    "profit_es": np.random.uniform(-100, 100, 100),
                    "profit_mnq": np.random.uniform(-100, 100, 100),
                    "bid_size_level_1": np.random.randint(100, 500, 100),
                    "ask_size_level_1": np.random.randint(100, 500, 100),
                    "trade_frequency_1s": np.random.uniform(0.1, 10, 100),
                    "spread_avg_1min": np.random.uniform(0.01, 0.05, 100),
                    # Ajout : Nouvelles features pour le test
                    "bid_ask_imbalance": np.random.uniform(-1, 1, 100),
                    "trade_aggressiveness": np.random.uniform(-1, 1, 100),
                    "iv_skew": np.random.uniform(-0.5, 0.5, 100),
                    "iv_term_structure": np.random.uniform(0, 0.05, 100),
                    "option_skew": np.random.uniform(-1, 1, 100),
                }
            )
            model = await trainer.train_multi_market(
                data, total_timesteps=1000, mode="trend", market="ES"
            )
            print(f"Modèle entraîné pour ES/trend: {model}")
        except Exception as e:
            logger.error(f"Erreur test: {str(e)}\n{traceback.format_exc()}")
            raise

    asyncio.run(main())