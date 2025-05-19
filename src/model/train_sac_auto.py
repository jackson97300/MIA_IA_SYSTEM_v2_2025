# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/model/train_sac_auto.py
# Automatise l'entraînement SAC, PPO, DDPG pour plusieurs modes (trend, range, defensive), déclenché par adaptive_learner.py.
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle :
# Automatise l’entraînement des modèles SAC, PPO, DDPG avec fine-tuning (méthode 8), apprentissage en ligne (méthode 10),
# curriculum progressif (méthode 15), et meta-learning (méthode 18). Intègre des mécanismes robustes (retries, validation,
# snapshots compressés, alertes Telegram) pour une fiabilité optimale.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, psutil>=5.9.8,<6.0.0, matplotlib>=3.7.0,<4.0.0, stable-baselines3>=2.0.0,<3.0.0,
#   pyyaml>=6.0.0,<7.0.0, boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0, argparse, asyncio, signal
# - src/model/train_sac.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/model_validator.py
# - src/model/utils/algo_performance_logger.py
# - src/model/utils/finetune_utils.py
# - src/model/utils/maml_utils.py
# - src/envs/trading_env.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/algo_config.yaml (paramètres d’entraînement)
# - config/feature_sets.yaml (350 features et 150 SHAP features)
# - Données brutes (pd.DataFrame avec 350 features)
#
# Outputs :
# - Modèles entraînés dans model/sac_models/<market>/<mode>/<policy_type>/<algo>_<mode>_<timestamp>.zip
# - Logs dans data/logs/train_sac_auto.log
# - Logs de performance dans data/logs/train_sac_auto_performance.csv
# - Snapshots compressés dans data/cache/train_sac_auto/<market>/*.json.gz
# - Sauvegardes incrémentielles dans data/checkpoints/train_sac_auto/<market>/*.json.gz
# - Figures dans data/figures/train_sac_auto/<market>/
# - Résultats dans model/sac_models/<market>/train_sac_auto_results_<timestamp>.csv
#
# Notes :
# - Intègre fine-tuning (méthode 8), apprentissage en ligne (méthode 10), curriculum progressif (méthode 15), et
#   meta-learning (méthode 18) pour SAC.
# - Valide 350 features pour l’entraînement et 150 SHAP features pour l’inférence.
# - Utilise IQFeed comme source de données via TradingEnv.
# - Suppression de toute référence à dxFeed, obs_t, 320/81 features.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des données.
# - Tests unitaires disponibles dans tests/test_train_sac_auto.py.

import argparse
import asyncio
import gzip
import json
import signal
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from gymnasium import spaces
from loguru import logger
from stable_baselines3 import SAC

from src.envs.trading_env import TradingEnv
from src.model.train_sac import SACTrainer
from src.model.utils.alert_manager import AlertManager
from src.model.utils.algo_performance_logger import AlgoPerformanceLogger
from src.model.utils.config_manager import get_config
from src.model.utils.finetune_utils import finetune_model, online_learning
from src.model.utils.maml_utils import apply_prototypical_networks
from src.model.utils.model_validator import ModelValidator
from src.utils.telegram_alert import send_telegram_alert

# Configuration logging
logger.remove()
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOG_DIR = BASE_DIR / "data" / "logs"
CACHE_DIR = BASE_DIR / "data" / "cache" / "train_sac_auto"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints" / "train_sac_auto"
FIGURE_DIR = BASE_DIR / "data" / "figures" / "train_sac_auto"
RESULTS_DIR = BASE_DIR / "model" / "sac_models"
PERF_LOG_PATH = LOG_DIR / "train_sac_auto_performance.csv"
SNAPSHOT_DIR = CACHE_DIR
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "train_sac_auto.log", rotation="10 MB", level="INFO", encoding="utf-8"
)
logger.add(lambda msg: print(msg, end=""), level="INFO")

# Constantes
MAX_RETRIES = 3
RETRY_DELAY = 2.0
MAX_CACHE_SIZE = 1000
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB

# Cache global pour les résultats d'entraînement
train_cache = OrderedDict()


def log_performance(
    operation: str,
    latency: float,
    success: bool = True,
    error: str = None,
    market: str = "ES",
    **kwargs,
) -> None:
    """
    Enregistre les performances (CPU, mémoire, latence) dans train_sac_auto_performance.csv.

    Args:
        operation (str): Nom de l’opération.
        latency (float): Temps d’exécution en secondes.
        success (bool): Indique si l’opération a réussi.
        error (str): Message d’erreur (si applicable).
        market (str): Marché (ex. : ES, MNQ).
    """
    try:
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
        cpu_percent = psutil.cpu_percent()
        if memory_usage > 1024:
            alert_msg = (
                f"ALERTE: Usage mémoire élevé ({memory_usage:.2f} MB) pour {market}"
            )
            logger.warning(alert_msg)
            AlertManager().send_alert(alert_msg, priority=5)
            send_telegram_alert(alert_msg)
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "latency": latency,
            "success": success,
            "error": error,
            "memory_usage_mb": memory_usage,
            "cpu_percent": cpu_percent,
            "market": market,
            **kwargs,
        }
        log_df = pd.DataFrame([log_entry])

        def save_log():
            if not PERF_LOG_PATH.exists():
                log_df.to_csv(PERF_LOG_PATH, index=False, encoding="utf-8")
            else:
                log_df.to_csv(
                    PERF_LOG_PATH, mode="a", header=False, index=False, encoding="utf-8"
                )

        with_retries(save_log, market=market)
        logger.info(f"Performance journalisée pour {operation}. CPU: {cpu_percent}%")
    except Exception as e:
        error_msg = f"Erreur journalisation performance pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        AlertManager().send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)


def save_snapshot(
    snapshot_type: str, data: Dict, market: str = "ES", compress: bool = True
) -> None:
    """
    Sauvegarde un instantané JSON des résultats, compressé avec gzip.

    Args:
        snapshot_type (str): Type de snapshot (ex. : auto_train, auto_train_sac).
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

        with_retries(write_snapshot, market=market)
        save_path = f"{snapshot_path}.gz" if compress else snapshot_path
        file_size = os.path.getsize(save_path) / 1024 / 1024
        if file_size > 1.0:
            alert_msg = f"Snapshot size {file_size:.2f} MB exceeds 1 MB pour {market}"
            logger.warning(alert_msg)
            AlertManager().send_alert(alert_msg, priority=3)
            send_telegram_alert(alert_msg)
        latency = (datetime.now() - start_time).total_seconds()
        success_msg = f"Snapshot {snapshot_type} sauvegardé pour {market}: {save_path}"
        logger.info(success_msg)
        AlertManager().send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
        log_performance(
            "save_snapshot",
            latency,
            success=True,
            snapshot_size_mb=file_size,
            market=market,
        )
    except Exception as e:
        error_msg = f"Erreur sauvegarde snapshot {snapshot_type} pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        AlertManager().send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)
        log_performance("save_snapshot", 0, success=False, error=str(e), market=market)


def checkpoint(
    data: pd.DataFrame, data_type: str = "train_sac_auto_state", market: str = "ES"
) -> None:
    """
    Sauvegarde incrémentielle des données toutes les 5 minutes avec versionnage (5 versions).

    Args:
        data (pd.DataFrame): Données à sauvegarder.
        data_type (str): Type de données (ex. : train_sac_auto_state).
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
            checkpoint_dir / f"train_sac_auto_{data_type}_{timestamp}.json.gz"
        )
        checkpoint_versions = []

        def write_checkpoint():
            with gzip.open(checkpoint_path, "wt", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=4)
            data.to_csv(
                checkpoint_path.replace(".json.gz", ".csv"),
                index=False,
                encoding="utf-8",
            )

        with_retries(write_checkpoint, market=market)
        checkpoint_versions.append(checkpoint_path)
        if len(checkpoint_versions) > 5:
            oldest = checkpoint_versions.pop(0)
            if os.path.exists(oldest):
                os.remove(oldest)
            csv_oldest = oldest.replace(".json.gz", ".csv")
            if os.path.exists(csv_oldest):
                os.remove(csv_oldest)
        file_size = os.path.getsize(checkpoint_path) / 1024 / 1024
        latency = (datetime.now() - start_time).total_seconds()
        success_msg = f"Checkpoint sauvegardé pour {market}: {checkpoint_path}"
        logger.info(success_msg)
        AlertManager().send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
        log_performance(
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
        AlertManager().send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)
        log_performance(
            "checkpoint",
            0,
            success=False,
            error=str(e),
            data_type=data_type,
            market=market,
        )


def cloud_backup(
    data: pd.DataFrame, data_type: str = "train_sac_auto_state", market: str = "ES"
) -> None:
    """
    Sauvegarde distribuée des données vers AWS S3 toutes les 15 minutes.

    Args:
        data (pd.DataFrame): Données à sauvegarder.
        data_type (str): Type de données (ex. : train_sac_auto_state).
        market (str): Marché (ex. : ES, MNQ).
    """
    try:
        start_time = datetime.now()
        config = get_config(BASE_DIR / "config/algo_config.yaml")
        if not config.get("s3_bucket"):
            warning_msg = (
                f"S3 bucket non configuré, sauvegarde cloud ignorée pour {market}"
            )
            logger.warning(warning_msg)
            AlertManager().send_alert(warning_msg, priority=3)
            send_telegram_alert(warning_msg)
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{config['s3_prefix']}train_sac_auto_{data_type}_{market}_{timestamp}.csv.gz"
        temp_path = CHECKPOINT_DIR / market / f"temp_s3_{timestamp}.csv.gz"
        temp_path.parent.mkdir(exist_ok=True)

        def write_temp():
            data.to_csv(temp_path, compression="gzip", index=False, encoding="utf-8")

        with_retries(write_temp, market=market)
        s3_client = boto3.client("s3")

        def upload_s3():
            s3_client.upload_file(temp_path, config["s3_bucket"], backup_path)

        with_retries(upload_s3, market=market)
        os.remove(temp_path)
        latency = (datetime.now() - start_time).total_seconds()
        success_msg = f"Sauvegarde cloud S3 effectuée pour {market}: {backup_path}"
        logger.info(success_msg)
        AlertManager().send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
        log_performance(
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
        AlertManager().send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)
        log_performance(
            "cloud_backup",
            0,
            success=False,
            error=str(e),
            data_type=data_type,
            market=market,
        )


def with_retries(
    func: callable,
    max_attempts: int = MAX_RETRIES,
    delay_base: float = RETRY_DELAY,
    market: str = "ES",
) -> Optional[Any]:
    """
    Exécute une fonction avec retries exponentiels.

    Args:
        func (callable): Fonction à exécuter.
        max_attempts (int): Nombre maximum de tentatives.
        delay_base (float): Base pour le délai exponentiel.
        market (str): Marché (ex. : ES, MNQ).

    Returns:
        Optional[Any]: Résultat de la fonction ou None si échec.
    """
    start_time = datetime.now()
    for attempt in range(max_attempts):
        try:
            result = func()
            latency = (datetime.now() - start_time).total_seconds()
            log_performance(
                f"retry_attempt_{attempt+1}",
                latency,
                success=True,
                attempt_number=attempt + 1,
                market=market,
            )
            return result
        except Exception as e:
            if attempt == max_attempts - 1:
                error_msg = f"Échec après {max_attempts} tentatives pour {market}: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                AlertManager().send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                latency = (datetime.now() - start_time).total_seconds()
                log_performance(
                    f"retry_attempt_{attempt+1}",
                    latency,
                    success=False,
                    error=str(e),
                    attempt_number=attempt + 1,
                    market=market,
                )
                return None
            delay = delay_base**attempt
            warning_msg = (
                f"Tentative {attempt+1} échouée pour {market}, retry après {delay}s"
            )
            logger.warning(warning_msg)
            AlertManager().send_alert(warning_msg, priority=3)
            send_telegram_alert(warning_msg)
            time.sleep(delay)


def validate_data(
    data: pd.DataFrame, shap_features: bool = False, market: str = "ES"
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
            raise ValueError(
                f"Attendu {expected_len} features pour {market}, trouvé {len(expected_cols)}"
            )

        missing_cols = [col for col in expected_cols if col not in data.columns]
        null_count = data[expected_cols].isnull().sum().sum()
        confidence_drop_rate = (
            null_count / (len(data) * len(expected_cols))
            if (len(data) * len(expected_cols)) > 0
            else 0.0
        )
        if confidence_drop_rate > 0.5:
            alert_msg = f"Confidence_drop_rate élevé pour {market}: {confidence_drop_rate:.2f} ({null_count} valeurs nulles)"
            logger.warning(alert_msg)
            AlertManager().send_alert(alert_msg, priority=3)
            send_telegram_alert(alert_msg)
        if missing_cols:
            warning_msg = f"Colonnes manquantes pour {market} ({'SHAP' if shap_features else 'full'}): {missing_cols}"
            logger.warning(warning_msg)
            AlertManager().send_alert(warning_msg, priority=2)
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
        for col in critical_cols:
            if col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    raise ValueError(
                        f"Colonne {col} n'est pas numérique pour {market}: {data[col].dtype}"
                    )
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
        if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
            raise ValueError(
                f"Colonne 'timestamp' doit être de type datetime pour {market}"
            )
        latency = (datetime.now() - start_time).total_seconds()
        log_performance(
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
        AlertManager().send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)
        latency = (datetime.now() - start_time).total_seconds()
        log_performance(
            "validate_data", latency, success=False, error=str(e), market=market
        )
        return False


class TrainSACAuto:
    """Automatise l'entraînement SAC, PPO, DDPG pour plusieurs modes."""

    def __init__(self):
        self.alert_manager = AlertManager()
        config_path = BASE_DIR / "config/algo_config.yaml"
        self.config = get_config(config_path)
        self.validator = ModelValidator()
        self.performance_logger = AlgoPerformanceLogger()
        SNAPSHOT_DIR.mkdir(exist_ok=True)
        PERF_LOG_PATH.parent.mkdir(exist_ok=True)
        FIGURE_DIR.mkdir(exist_ok=True)
        RESULTS_DIR.mkdir(exist_ok=True)
        success_msg = "TrainSACAuto initialisé"
        logger.info(success_msg)
        self.alert_manager.send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
        signal.signal(signal.SIGINT, self.handle_sigint)

    def handle_sigint(self, signum, frame):
        """Gère l'arrêt propre sur SIGINT."""
        snapshot = {"timestamp": datetime.now().isoformat(), "status": "SIGINT"}
        save_snapshot("sigint", snapshot, market="ES")
        success_msg = "Arrêt propre sur SIGINT, snapshot sauvegardé"
        logger.info(success_msg)
        self.alert_manager.send_alert(success_msg, priority=2)
        send_telegram_alert(success_msg)
        exit(0)

    def auto_train_sac(
        self,
        data: pd.DataFrame,
        config_path: str = str(BASE_DIR / "config/trading_env_config.yaml"),
        mode: str = "trend",
        policy_type: str = "mlp",
        epochs: int = 1000,
        market: str = "ES",
    ) -> Optional[SAC]:
        """
        Entraîne un modèle SAC avec fine-tuning (méthode 8), apprentissage en ligne (méthode 10),
        curriculum progressif (méthode 15), et meta-learning (méthode 18).

        Args:
            data (pd.DataFrame): Données contenant 350 features (entraînement) ou 150 SHAP features (inférence).
            config_path (str): Chemin vers la configuration.
            mode (str): Mode du modèle ("trend", "range", "defensive").
            policy_type (str): Type de politique ("mlp", "transformer").
            epochs (int): Nombre de timesteps de base pour l’entraînement.
            market (str): Marché (ex. : ES, MNQ).

        Returns:
            Optional[SAC]: Modèle SAC entraîné ou None si échec.
        """
        start_time = datetime.now()
        try:
            # Validation des données
            if not validate_data(
                data, shap_features=(policy_type == "transformer"), market=market
            ):
                error_msg = f"Données invalides pour entraînement SAC pour {market}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                return None

            # Configurer l’environnement
            env = TradingEnv(config_path=config_path)
            env.mode = mode
            env.policy_type = policy_type
            feature_sets_path = BASE_DIR / "config/feature_sets.yaml"
            features_config = get_config(feature_sets_path)
            env.obs_t = (
                features_config.get("training", {}).get("features", [])[:350]
                if policy_type == "mlp"
                else features_config.get("inference", {}).get("shap_features", [])[:150]
            )
            env.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    (env.sequence_length, len(env.obs_t))
                    if policy_type == "transformer"
                    else (len(env.obs_t),)
                ),
                dtype=np.float32,
            )
            env.data = data

            # Fine-tuning (méthode 8)
            model = finetune_model(
                data,
                ent_coef=0.1,
                config_path=config_path,
                mode=mode,
                policy_type=policy_type,
                batch_size=100,
                total_timesteps=100,
            )
            if model is None:
                error_msg = f"Échec du fine-tuning pour mode={mode}, policy_type={policy_type} pour {market}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                return None
            logger.info(
                f"Fine-tuning (méthode 8) terminé pour mode={mode}, policy_type={policy_type} pour {market}"
            )

            # Meta-learning (méthode 18)
            model = apply_prototypical_networks(
                model,
                data,
                config_path=config_path,
                mode=mode,
                policy_type=policy_type,
                batch_size=100,
            )
            if model is None:
                error_msg = f"Échec du meta-learning pour mode={mode}, policy_type={policy_type} pour {market}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                return None
            logger.info(
                f"Meta-learning (méthode 18) terminé pour mode={mode}, policy_type={policy_type} pour {market}"
            )

            # Curriculum progressif (méthode 15)
            for complexity in range(1, 4):
                timesteps = epochs * complexity

                def train_with_complexity():
                    model.learn(total_timesteps=timesteps)
                    return model

                model = with_retries(train_with_complexity, market=market)
                if model is None:
                    error_msg = f"Échec de l’entraînement curriculum (complexity={complexity}) pour mode={mode}, policy_type={policy_type} pour {market}"
                    logger.error(error_msg)
                    self.alert_manager.send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    return None
                logger.info(
                    f"Curriculum progressif (méthode 15) terminé pour complexity={complexity}, mode={mode}, policy_type={policy_type} pour {market}"
                )

            # Apprentissage en ligne (méthode 10)
            model = online_learning(
                data,
                model,
                env,
                batch_size=32,
                total_timesteps=50,
                learning_rate=0.0001,
                config_path=config_path,
            )
            if model is None:
                error_msg = f"Échec de l’apprentissage en ligne pour mode={mode}, policy_type={policy_type} pour {market}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                return None
            logger.info(
                f"Apprentissage en ligne (méthode 10) terminé pour mode={mode}, policy_type={policy_type} pour {market}"
            )

            # Sauvegarder le modèle
            model_dir = RESULTS_DIR / market / mode / policy_type
            model_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = model_dir / f"sac_{mode}_{policy_type}_{timestamp}.zip"
            model.save(model_path)
            logger.info(f"Modèle SAC sauvegardé pour {market}: {model_path}")

            # Sauvegarder un snapshot
            snapshot_data = {
                "mode": mode,
                "policy_type": policy_type,
                "epochs": epochs,
                "num_rows": len(data),
                "model_path": str(model_path),
                "market": market,
            }
            save_snapshot("auto_train_sac", snapshot_data, market=market)

            # Sauvegarder un checkpoint
            checkpoint_data = pd.DataFrame(
                [
                    {
                        "timestamp": datetime.now().isoformat(),
                        "mode": mode,
                        "policy_type": policy_type,
                        "model_path": str(model_path),
                        "market": market,
                    }
                ]
            )
            checkpoint(checkpoint_data, data_type="model_state", market=market)
            cloud_backup(checkpoint_data, data_type="model_state", market=market)

            latency = (datetime.now() - start_time).total_seconds()
            log_performance(
                "auto_train_sac",
                latency,
                success=True,
                mode=mode,
                policy_type=policy_type,
                epochs=epochs,
                num_rows=len(data),
                market=market,
            )
            success_msg = f"Entraînement auto SAC terminé pour mode={mode}, policy_type={policy_type} pour {market}. CPU: {psutil.cpu_percent()}%"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            return model

        except Exception as e:
            error_msg = f"Erreur dans auto_train_sac pour {market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            latency = (datetime.now() - start_time).total_seconds()
            log_performance(
                "auto_train_sac",
                latency,
                success=False,
                error=str(e),
                mode=mode,
                policy_type=policy_type,
                market=market,
            )
            return None

    async def auto_train(
        self,
        data: pd.DataFrame,
        config_path: str = str(BASE_DIR / "config/trading_env_config.yaml"),
        modes: Optional[List[str]] = None,
        algo_types: Optional[List[str]] = None,
        policy_types: Optional[List[str]] = None,
        epochs: int = 100000,
        market: str = "ES",
    ) -> Dict[str, bool]:
        """
        Automatise l'entraînement SAC, PPO, DDPG pour plusieurs modes et politiques.

        Args:
            data (pd.DataFrame): Données contenant 350 features.
            config_path (str): Chemin vers la configuration.
            modes (List[str], optional): Modes à entraîner ("trend", "range", "defensive").
            algo_types (List[str], optional): Algorithmes à entraîner ("sac", "ppo", "ddpg").
            policy_types (List[str], optional): Types de politiques ("mlp", "transformer").
            epochs (int): Nombre total de timesteps par entraînement.
            market (str): Marché (ex. : ES, MNQ).

        Returns:
            Dict[str, bool]: Résultats de l’entraînement par mode, algo, et politique.
        """
        start_time = datetime.now()
        try:
            # Validation des paramètres
            valid_modes = ["trend", "range", "defensive"]
            valid_algo_types = ["sac", "ppo", "ddpg"]
            valid_policies = ["mlp", "transformer"]
            modes = modes if modes is not None else valid_modes
            algo_types = algo_types if algo_types is not None else valid_algo_types
            policy_types = policy_types if policy_types is not None else ["mlp"]
            for mode in modes:
                if mode not in valid_modes:
                    raise ValueError(
                        f"Mode non supporté pour {market}: {mode}. Options: {valid_modes}"
                    )
            for algo_type in algo_types:
                if algo_type not in valid_algo_types:
                    raise ValueError(
                        f"Type d’algorithme non supporté pour {market}: {algo_type}. Options: {valid_algo_types}"
                    )
            for policy_type in policy_types:
                if policy_type not in valid_policies:
                    raise ValueError(
                        f"Type de politique non supporté pour {market}: {policy_type}. Options: {valid_policies}"
                    )

            # Validation des données
            if not isinstance(data, pd.DataFrame):
                raise ValueError(
                    f"Les données doivent être un DataFrame pandas pour {market}"
                )
            if not validate_data(data, market=market):
                error_msg = f"Données invalides pour entraînement pour {market}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                return {}
            logger.info(
                f"Données validées pour entraînement automatique: {len(data)} lignes pour {market}"
            )

            # Vérifier le cache
            cache_key = f"{market}_{hash(str(data))}_{','.join(modes)}_{','.join(algo_types)}_{','.join(policy_types)}_{epochs}"
            if cache_key in train_cache:
                results = train_cache[cache_key]
                train_cache.move_to_end(cache_key)
                return results
            while len(train_cache) > MAX_CACHE_SIZE:
                train_cache.popitem(last=False)

            # Initialisation des résultats
            results = {}
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Boucle sur les modes, algorithmes, et politiques
            for mode in modes:
                for algo_type in algo_types:
                    for policy_type in policy_types:
                        key = f"{mode}_{algo_type}_{policy_type}"
                        logger.info(
                            f"Début entraînement: mode={mode}, algo_type={algo_type}, policy_type={policy_type}, epochs={epochs} pour {market}"
                        )
                        try:
                            if algo_type == "sac":
                                # Utiliser auto_train_sac pour SAC
                                model = self.auto_train_sac(
                                    data,
                                    config_path,
                                    mode,
                                    policy_type,
                                    epochs,
                                    market=market,
                                )
                                validation = (
                                    self.validator.validate_model(
                                        model,
                                        algo_type,
                                        mode,
                                        np.zeros(
                                            (1, 350 if policy_type == "mlp" else 150)
                                        ),
                                    )
                                    if model
                                    else {"valid": False}
                                )
                            else:
                                # Entraînement standard pour PPO/DDPG
                                env = TradingEnv(config_path=config_path)
                                trainer = SACTrainer(env, policy_type)
                                await trainer.train_multi_market(
                                    data,
                                    total_timesteps=epochs,
                                    mode=mode,
                                    market=market,
                                )
                                validation = self.validator.validate_model(
                                    trainer.models[algo_type][mode].model,
                                    algo_type,
                                    mode,
                                    np.zeros((1, 350 if policy_type == "mlp" else 150)),
                                )

                            if not validation["valid"]:
                                error_msg = f"Modèle {algo_type} ({mode}, {policy_type}) invalide après entraînement pour {market}"
                                logger.error(error_msg)
                                self.alert_manager.send_alert(error_msg, priority=3)
                                send_telegram_alert(error_msg)
                                results[key] = False
                                continue

                            results[key] = True
                            self.performance_logger.log_performance(
                                algo_type,
                                mode,
                                validation["mean_reward"],
                                time.time() - start_time,
                                psutil.Process().memory_info().rss / 1024 / 1024,
                            )
                            logger.info(
                                f"Entraînement réussi: mode={mode}, algo_type={algo_type}, policy_type={policy_type} pour {market}"
                            )
                        except Exception as e:
                            error_msg = f"Erreur entraînement mode={mode}, algo_type={algo_type}, policy_type={policy_type} pour {market}: {str(e)}\n{traceback.format_exc()}"
                            logger.error(error_msg)
                            self.alert_manager.send_alert(error_msg, priority=4)
                            send_telegram_alert(error_msg)
                            results[key] = False

            # Exportation des résultats
            results_log = [
                {"mode_algo_policy": k, "success": v} for k, v in results.items()
            ]
            results_dir = RESULTS_DIR / market
            results_dir.mkdir(exist_ok=True)
            results_file = results_dir / f"train_sac_auto_results_{timestamp}.csv"
            results_df = pd.DataFrame(results_log)
            results_df.to_csv(results_file, index=False)
            logger.info(f"Résultats exportés pour {market}: {results_file}")

            # Sauvegarder un checkpoint
            checkpoint(results_df, data_type="results", market=market)
            cloud_backup(results_df, data_type="results", market=market)

            # Visualisation
            fig, ax = plt.subplots()
            for key, success in results.items():
                ax.bar(key, 1 if success else 0, alpha=0.4, label=key)
            ax.set_title(
                f"Résultats Entraînement Automatique pour {market}: {timestamp}"
            )
            ax.legend()
            figure_dir = FIGURE_DIR / market
            figure_dir.mkdir(exist_ok=True)
            plt.savefig(figure_dir / f"results_{timestamp}.png")
            plt.close()

            # Sauvegarder un snapshot
            snapshot_data = {
                "timestamp": timestamp,
                "modes": modes,
                "algo_types": algo_types,
                "policy_types": policy_types,
                "epochs": epochs,
                "num_rows": len(data),
                "results": results,
                "market": market,
            }
            save_snapshot("auto_train", snapshot_data, market=market)

            # Mettre à jour le cache
            train_cache[cache_key] = results

            latency = (datetime.now() - start_time).total_seconds()
            log_performance(
                "auto_train", latency, success=True, num_rows=len(data), market=market
            )
            success_msg = f"Entraînement automatique terminé pour {market}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            return results

        except Exception as e:
            error_msg = f"Erreur dans auto_train pour {market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            latency = (datetime.now() - start_time).total_seconds()
            log_performance(
                "auto_train", latency, success=False, error=str(e), market=market
            )
            return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automatise l'entraînement SAC/PPO/DDPG pour plusieurs modes."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(BASE_DIR / "data/features/features_latest_filtered.csv"),
        help="Chemin vers les données",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=str(BASE_DIR / "config/trading_env_config.yaml"),
        help="Chemin vers le fichier de configuration",
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="*",
        default=None,
        help="Modes à entraîner (trend, range, defensive)",
    )
    parser.add_argument(
        "--algo_types",
        type=str,
        nargs="*",
        default=None,
        help="Algorithmes à entraîner (sac, ppo, ddpg)",
    )
    parser.add_argument(
        "--policy_types",
        type=str,
        nargs="*",
        default=["mlp"],
        help="Types de politiques (mlp, transformer)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100000,
        help="Nombre total de timesteps par entraînement",
    )
    parser.add_argument(
        "--market", type=str, default="ES", help="Marché (ex. : ES, MNQ)"
    )

    args = parser.parse_args()

    try:
        data = pd.read_csv(args.data_path)
        trainer = TrainSACAuto()
        results = asyncio.run(
            trainer.auto_train(
                data=data,
                config_path=args.config_path,
                modes=args.modes,
                algo_types=args.algo_types,
                policy_types=args.policy_types,
                epochs=args.epochs,
                market=args.market,
            )
        )
        for key, success in results.items():
            print(
                f"Entraînement {key} pour {args.market}: {'Succès' if success else 'Échec'}"
            )
    except Exception as e:
        error_msg = f"Erreur test principal pour {args.market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        AlertManager().send_alert(error_msg, priority=5)
        send_telegram_alert(error_msg)
        raise
