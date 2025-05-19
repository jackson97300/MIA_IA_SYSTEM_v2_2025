# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/model/utils_model.py
# Fonctions utilitaires pour early stopping, gestion mémoire, export logs, optimisées pour MIA_IA_SYSTEM_v2_2025.
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle :
# Fournit des fonctions utilitaires pour la validation des features, la gestion de la configuration, l’early stopping,
# l’audit des features, l’exportation des logs, le calcul des métriques, et la gestion de la mémoire. Intègre des
# mécanismes robustes (retries, logs psutil, snapshots compressés, alertes Telegram) pour une fiabilité optimale.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.8,<6.0.0, pyyaml>=6.0.0,<7.0.0,
#   stable-baselines3>=2.0.0,<3.0.0, boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0
# - src/features/neural_pipeline.py
# - src/envs/trading_env.py
# - src/model/utils/config_manager.py
# - src/model/utils/miya_console.py
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/model_params.yaml (paramètres utilitaires)
# - config/feature_sets.yaml (350 features et 150 SHAP features)
# - Données (pd.DataFrame avec 350 features ou 150 SHAP features)
#
# Outputs :
# - Logs dans data/logs/utils_model.log
# - Logs de performance dans data/logs/utils_model_performance.csv
# - Snapshots compressés dans data/cache/utils_model/<market>/*.json.gz
# - Sauvegardes incrémentielles dans data/checkpoints/utils_model/<market>/*.json.gz
#
# Notes :
# - Valide 350 features pour l’entraînement et 150 SHAP features pour l’inférence via config_manager.
# - Utilise IQFeed comme source de données via TradingEnv.
# - Suppression de toute référence à dxFeed, obs_t, 320/81 features.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des données.
# - Tests unitaires disponibles dans tests/test_utils_model.py.

import gzip
import json
import os
import time
from collections import OrderedDict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import boto3
import numpy as np
import pandas as pd
import psutil
import yaml
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback

from src.envs.trading_env import TradingEnv
from src.features.neural_pipeline import NeuralPipeline
from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import get_config
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.utils.telegram_alert import send_telegram_alert

# Configuration logging
logger.remove()
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOG_DIR = BASE_DIR / "data" / "logs"
CACHE_DIR = BASE_DIR / "data" / "cache" / "utils_model"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints" / "utils_model"
PERF_LOG_PATH = LOG_DIR / "utils_model_performance.csv"
SNAPSHOT_DIR = CACHE_DIR
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "utils_model.log", rotation="10 MB", level="INFO", encoding="utf-8"
)
logger.add(lambda msg: print(msg, end=""), level="INFO")

# Constantes
MAX_RETRIES = 3
RETRY_DELAY = 2.0
MAX_CACHE_SIZE = 1000
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB

# Seuils de performance globaux
PERFORMANCE_THRESHOLDS = {
    "min_mean_reward": -100.0,
    "max_volatility": 2.0,
    "min_balance": 9000.0,
    "max_drawdown": -1000.0,
    "min_reward": -100.0,
    "min_sharpe": 0.0,
}

# Buffer global pour les logs de trading
trade_log_buffer = deque(maxlen=10000)


def log_performance(
    operation: str,
    latency: float,
    success: bool = True,
    error: str = None,
    market: str = "ES",
    **kwargs,
) -> None:
    """
    Enregistre les performances (CPU, mémoire, latence) dans utils_model_performance.csv.

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

        with_retries(save_log)
        logger.info(f"Performance journalisée pour {operation}. CPU: {cpu_percent}%")
    except Exception as e:
        error_msg = f"Erreur journalisation performance pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        miya_alerts(error_msg, tag="UTILS_MODEL", priority=3, voice_profile="urgent")
        AlertManager().send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)


def save_snapshot(
    snapshot_type: str, data: Dict, market: str = "ES", compress: bool = True
) -> None:
    """
    Sauvegarde un instantané JSON des résultats, compressé avec gzip.

    Args:
        snapshot_type (str): Type de snapshot (ex. : validate_features).
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

        with_retries(write_snapshot)
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
        miya_speak(success_msg, tag="UTILS_MODEL", voice_profile="calm")
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
        miya_alerts(error_msg, tag="UTILS_MODEL", priority=3, voice_profile="urgent")
        AlertManager().send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)
        log_performance("save_snapshot", 0, success=False, error=str(e), market=market)


def checkpoint(
    data: pd.DataFrame, data_type: str = "utils_model_state", market: str = "ES"
) -> None:
    """
    Sauvegarde incrémentielle des données toutes les 5 minutes avec versionnage (5 versions).

    Args:
        data (pd.DataFrame): Données à sauvegarder.
        data_type (str): Type de données (ex. : utils_model_state).
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
            checkpoint_dir / f"utils_model_{data_type}_{timestamp}.json.gz"
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

        with_retries(write_checkpoint)
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
        miya_alerts(error_msg, tag="UTILS_MODEL", priority=3, voice_profile="urgent")
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
    data: pd.DataFrame, data_type: str = "utils_model_state", market: str = "ES"
) -> None:
    """
    Sauvegarde distribuée des données vers AWS S3 toutes les 15 minutes.

    Args:
        data (pd.DataFrame): Données à sauvegarder.
        data_type (str): Type de données (ex. : utils_model_state).
        market (str): Marché (ex. : ES, MNQ).
    """
    try:
        start_time = datetime.now()
        config = get_config(BASE_DIR / "config/model_params.yaml")
        if not config.get("s3_bucket"):
            warning_msg = (
                f"S3 bucket non configuré, sauvegarde cloud ignorée pour {market}"
            )
            logger.warning(warning_msg)
            AlertManager().send_alert(warning_msg, priority=3)
            send_telegram_alert(warning_msg)
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = (
            f"{config['s3_prefix']}utils_model_{data_type}_{market}_{timestamp}.csv.gz"
        )
        temp_path = CHECKPOINT_DIR / market / f"temp_s3_{timestamp}.csv.gz"
        temp_path.parent.mkdir(exist_ok=True)

        def write_temp():
            data.to_csv(temp_path, compression="gzip", index=False, encoding="utf-8")

        with_retries(write_temp)
        s3_client = boto3.client("s3")

        def upload_s3():
            s3_client.upload_file(temp_path, config["s3_bucket"], backup_path)

        with_retries(upload_s3)
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
        miya_alerts(error_msg, tag="UTILS_MODEL", priority=3, voice_profile="urgent")
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
                miya_alerts(
                    error_msg, tag="UTILS_MODEL", priority=4, voice_profile="urgent"
                )
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
            miya_speak(warning_msg, tag="UTILS_MODEL", level="warning")
            AlertManager().send_alert(warning_msg, priority=3)
            send_telegram_alert(warning_msg)
            time.sleep(delay)


def validate_features(
    data: pd.DataFrame, step: int = 0, shap_features: bool = False, market: str = "ES"
) -> None:
    """
    Valide la cohérence des features avec 350 (entraînement) ou 150 SHAP features (inférence).

    Args:
        data (pd.DataFrame): Données à valider.
        step (int): Étape actuelle pour logging.
        shap_features (bool): Si True, valide les 150 SHAP features; sinon, valide les 350 features.
        market (str): Marché (ex. : ES, MNQ).

    Raises:
        ValueError: Si les colonnes critiques sont manquantes ou non numériques.
    """
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
            warning_msg = f"Colonnes manquantes au step {step} pour {market} ({'SHAP' if shap_features else 'full'}): {missing_cols}"
            logger.warning(warning_msg)
            miya_speak(warning_msg, tag="UTILS_MODEL", level="warning")
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
                non_scalar = data[col].apply(
                    lambda x: isinstance(x, (list, dict, tuple))
                )
                if non_scalar.any():
                    raise ValueError(
                        f"Colonne {col} contient des valeurs non scalaires pour {market}: {data[col][non_scalar].head().tolist()}"
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

        save_snapshot(
            "validate_features",
            {
                "step": step,
                "num_columns": len(data.columns),
                "missing_columns": missing_cols,
                "shap_features": shap_features,
                "confidence_drop_rate": confidence_drop_rate,
                "market": market,
            },
            market=market,
        )
    except Exception as e:
        error_msg = f"Erreur validation features au step {step} pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        miya_alerts(error_msg, tag="UTILS_MODEL", priority=5, voice_profile="urgent")
        AlertManager().send_alert(error_msg, priority=5)
        send_telegram_alert(error_msg)
        raise


def load_config(
    config_path: str = str(BASE_DIR / "config/model_params.yaml"),
    policy_type: str = "mlp",
    market: str = "ES",
) -> Dict[str, Any]:
    """
    Charge les paramètres de configuration depuis model_params.yaml avec retries.

    Args:
        config_path (str): Chemin vers le fichier de configuration.
        policy_type (str): Type de politique ("mlp", "transformer").
        market (str): Marché (ex. : ES, MNQ).

    Returns:
        Dict[str, Any]: Paramètres de configuration.
    """

    def load_yaml():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    try:
        config = with_retries(load_yaml, market=market)
        if config is None:
            raise FileNotFoundError(
                f"Fichier {config_path} non trouvé ou vide pour {market}"
            )
        success_msg = f"Configuration chargée depuis {config_path}, policy_type={policy_type} pour {market}"
        logger.info(success_msg)
        miya_speak(success_msg, tag="UTILS_MODEL", voice_profile="calm")
        AlertManager().send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
        return config.get("utils_model", {})
    except Exception as e:
        error_msg = f"Erreur chargement {config_path} pour {market}: {str(e)}, utilisation valeurs par défaut\n{traceback.format_exc()}"
        logger.warning(error_msg)
        miya_alerts(error_msg, tag="UTILS_MODEL", priority=4, voice_profile="urgent")
        AlertManager().send_alert(error_msg, priority=4)
        send_telegram_alert(error_msg)
        return {
            "check_freq": 1000,
            "patience": 5000,
            "min_reward_improvement": 0.01,
            "max_log_size": 10000,
        }


class EarlyStoppingCallback(BaseCallback):
    """
    Callback pour arrêter l’entraînement si aucune amélioration n’est observée.

    Args:
        check_freq (int): Fréquence de vérification des performances.
        patience (int): Nombre de steps sans amélioration avant arrêt.
        min_reward_improvement (float): Amélioration minimale de la récompense.
        policy_type (str): Type de politique ("mlp", "transformer").
        market (str): Marché (ex. : ES, MNQ).
    """

    def __init__(
        self,
        check_freq: int = 1000,
        patience: int = 5000,
        min_reward_improvement: float = 0.01,
        policy_type: str = "mlp",
        market: str = "ES",
    ):
        super().__init__()
        config = load_config(policy_type=policy_type, market=market)
        self.check_freq = config.get("check_freq", check_freq)
        self.patience = config.get("patience", patience)
        self.min_reward_improvement = config.get(
            "min_reward_improvement", min_reward_improvement
        )
        self.policy_type = policy_type
        self.market = market
        self.best_mean_reward = -float("inf")
        self.steps_since_improvement = 0
        self.best_neural_metrics = {
            "predicted_volatility": None,
            "neural_regime": None,
            "cnn_pressure": None,
        }
        success_msg = f"EarlyStoppingCallback initialisé pour {market}: check_freq={self.check_freq}, patience={self.patience}, policy_type={policy_type}"
        logger.info(success_msg)
        miya_speak(success_msg, tag="UTILS_MODEL", voice_profile="calm")
        AlertManager().send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)

    def _on_step(self) -> bool:
        """
        Vérifie les performances à chaque étape et décide de continuer ou d’arrêter.

        Returns:
            bool: True pour continuer, False pour arrêter.
        """
        start_time = datetime.now()
        try:
            if self.n_calls % self.check_freq == 0:
                episode_rewards = self.locals.get("rewards", []) or self.locals.get(
                    "episode_rewards", []
                )
                mean_reward = (
                    float(np.mean(episode_rewards[-100:]))
                    if episode_rewards
                    else -float("inf")
                )

                env = self.model.env
                current_step = getattr(env, "current_step", 0)
                predicted_volatility = None
                neural_regime = None
                cnn_pressure = None
                sequence_length = getattr(env, "sequence_length", 50)

                if (
                    hasattr(env, "data")
                    and isinstance(env.data, pd.DataFrame)
                    and current_step < len(env.data)
                ):
                    validate_features(
                        env.data,
                        current_step,
                        shap_features=(self.policy_type == "transformer"),
                        market=self.market,
                    )
                    if "predicted_volatility" in env.data.columns:
                        predicted_volatility = float(
                            env.data["predicted_volatility"].iloc[current_step]
                        )
                    if "neural_regime" in env.data.columns:
                        neural_regime = int(
                            env.data["neural_regime"].iloc[current_step]
                        )
                    if "cnn_pressure" in env.data.columns:
                        cnn_pressure = float(
                            env.data["cnn_pressure"].iloc[current_step]
                        )

                    if (
                        predicted_volatility is None
                        or neural_regime is None
                        or cnn_pressure is None
                    ):

                        def compute_neural_metrics():
                            pipeline = NeuralPipeline(
                                window_size=sequence_length,
                                base_features=350,
                                config_path=str(BASE_DIR / "config/model_params.yaml"),
                            )
                            pipeline.load_models()
                            window_start = max(0, current_step - sequence_length + 1)
                            window_end = current_step + 1
                            raw_data = (
                                env.data[
                                    [
                                        "timestamp",
                                        "open",
                                        "high",
                                        "low",
                                        "close",
                                        "volume",
                                        f"atr_14_{self.market.lower()}",
                                        "adx_14",
                                    ]
                                ]
                                .iloc[window_start:window_end]
                                .fillna(0)
                            )
                            options_data = (
                                env.data[
                                    [
                                        "timestamp",
                                        f"gex_{self.market.lower()}",
                                        "oi_peak_call_near",
                                        "gamma_wall_call",
                                        "gamma_wall_put",
                                    ]
                                ]
                                .iloc[window_start:window_end]
                                .fillna(0)
                            )
                            orderflow_data = (
                                env.data[
                                    [
                                        "timestamp",
                                        "bid_size_level_1",
                                        "ask_size_level_1",
                                    ]
                                ]
                                .iloc[window_start:window_end]
                                .fillna(0)
                            )
                            return pipeline.run(raw_data, options_data, orderflow_data)

                        neural_result = with_retries(
                            compute_neural_metrics, market=self.market
                        )
                        if neural_result:
                            predicted_volatility = (
                                float(neural_result["volatility"][-1])
                                if predicted_volatility is None
                                else predicted_volatility
                            )
                            neural_regime = (
                                int(neural_result["regime"][-1])
                                if neural_regime is None
                                else neural_regime
                            )
                            cnn_pressure = (
                                float(neural_result["features"][-1, -1])
                                if cnn_pressure is None
                                else cnn_pressure
                            )
                            success_msg = f"Prédictions neuronales recalculées au step {self.n_calls} pour {self.market}: Vol={predicted_volatility:.2f}, Regime={neural_regime}, Pressure={cnn_pressure:.2f}"
                            logger.debug(success_msg)
                            miya_speak(
                                success_msg, tag="UTILS_MODEL", voice_profile="calm"
                            )
                            AlertManager().send_alert(success_msg, priority=1)
                            send_telegram_alert(success_msg)

                log_msg = f"Step {self.n_calls} pour {self.market} - Mean reward: {mean_reward:.2f}, policy_type={self.policy_type}"
                if predicted_volatility is not None:
                    log_msg += f", Predicted Volatility: {predicted_volatility:.2f}"
                if neural_regime is not None:
                    log_msg += f", Neural Regime: {neural_regime}"
                if cnn_pressure is not None:
                    log_msg += f", CNN Pressure: {cnn_pressure:.2f}"
                logger.info(log_msg)
                miya_speak(log_msg, tag="UTILS_MODEL", voice_profile="calm")

                if mean_reward > self.best_mean_reward + self.min_reward_improvement:
                    self.best_mean_reward = mean_reward
                    self.steps_since_improvement = 0
                    self.best_neural_metrics.update(
                        {
                            "predicted_volatility": predicted_volatility,
                            "neural_regime": neural_regime,
                            "cnn_pressure": cnn_pressure,
                        }
                    )
                else:
                    self.steps_since_improvement += self.check_freq

                if self.steps_since_improvement >= self.patience:
                    stop_msg = f"Early stopping déclenché après {self.steps_since_improvement} steps sans amélioration pour {self.market}. Best Mean Reward: {self.best_mean_reward:.2f}"
                    logger.info(stop_msg)
                    miya_speak(stop_msg, tag="UTILS_MODEL", voice_profile="warning")
                    AlertManager().send_alert(stop_msg, priority=3)
                    send_telegram_alert(stop_msg)
                    return False

                save_snapshot(
                    "early_stopping_check",
                    {
                        "step": self.n_calls,
                        "mean_reward": mean_reward,
                        "best_mean_reward": self.best_mean_reward,
                        "steps_since_improvement": self.steps_since_improvement,
                        "market": self.market,
                    },
                    market=self.market,
                )

            latency = (datetime.now() - start_time).total_seconds()
            log_performance(
                "early_stopping_step",
                latency,
                success=True,
                step=self.n_calls,
                market=self.market,
            )
            return True
        except Exception as e:
            error_msg = f"Erreur dans EarlyStoppingCallback._on_step pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            miya_alerts(
                error_msg, tag="UTILS_MODEL", priority=5, voice_profile="urgent"
            )
            AlertManager().send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            latency = (datetime.now() - start_time).total_seconds()
            log_performance(
                "early_stopping_step",
                latency,
                success=False,
                error=str(e),
                step=self.n_calls,
                market=self.market,
            )
            return True  # Continue par défaut


def audit_features(
    df: pd.DataFrame,
    shap_features: bool = False,
    policy_type: str = "mlp",
    market: str = "ES",
) -> bool:
    """
    Vérifie que le DataFrame contient toutes les colonnes attendues.

    Args:
        df (pd.DataFrame): DataFrame à auditer.
        shap_features (bool): Si True, valide les 150 SHAP features; sinon, valide les 350 features.
        policy_type (str): Type de politique ("mlp", "transformer").
        market (str): Marché (ex. : ES, MNQ).

    Returns:
        bool: True si toutes les colonnes sont présentes, False sinon.
    """
    try:
        feature_sets_path = BASE_DIR / "config/feature_sets.yaml"
        features_config = get_config(feature_sets_path)
        expected_cols = (
            features_config.get("training", {}).get("features", [])[:350]
            if not shap_features
            else features_config.get("inference", {}).get("shap_features", [])[:150]
        )
        missing = [col for col in expected_cols if col not in df.columns]
        if missing:
            warning_msg = f"Colonnes manquantes dans le DataFrame pour {market}: {missing}, policy_type={policy_type}"
            logger.warning(warning_msg)
            miya_speak(warning_msg, tag="UTILS_MODEL", level="warning")
            AlertManager().send_alert(warning_msg, priority=2)
            send_telegram_alert(warning_msg)
            save_snapshot(
                "audit_features",
                {
                    "missing_columns": missing,
                    "policy_type": policy_type,
                    "market": market,
                },
                market=market,
            )
            return False
        success_msg = f"Toutes les {len(expected_cols)} colonnes attendues sont présentes pour {market}, policy_type={policy_type}"
        logger.info(success_msg)
        miya_speak(success_msg, tag="UTILS_MODEL", voice_profile="calm")
        AlertManager().send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
        save_snapshot(
            "audit_features",
            {
                "num_columns": len(expected_cols),
                "policy_type": policy_type,
                "market": market,
            },
            market=market,
        )
        return True
    except Exception as e:
        error_msg = f"Erreur audit features pour {market}: {str(e)}, policy_type={policy_type}\n{traceback.format_exc()}"
        logger.error(error_msg)
        miya_alerts(error_msg, tag="UTILS_MODEL", priority=5, voice_profile="urgent")
        AlertManager().send_alert(error_msg, priority=5)
        send_telegram_alert(error_msg)
        return False


def export_logs(
    trade_log: List[Dict],
    output_dir: str,
    filename: str,
    policy_type: str = "mlp",
    market: str = "ES",
) -> None:
    """
    Exporte les logs de trading au format CSV avec retries.

    Args:
        trade_log (List[Dict]): Liste des logs de trading.
        output_dir (str): Répertoire de sortie pour les logs.
        filename (str): Nom du fichier de sortie.
        policy_type (str): Type de politique ("mlp", "transformer").
        market (str): Marché (ex. : ES, MNQ).
    """

    def save_logs():
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        log_df = pd.DataFrame(trade_log)
        required_cols = ["step", "action", "reward"]
        for col in required_cols:
            if col not in log_df.columns:
                warning_msg = f"Colonne requise '{col}' manquante pour {market}, ajoutée avec valeur par défaut"
                logger.warning(warning_msg)
                miya_speak(warning_msg, tag="UTILS_MODEL", level="warning")
                AlertManager().send_alert(warning_msg, priority=2)
                send_telegram_alert(warning_msg)
                log_df[col] = 0
        log_df.to_csv(output_path, index=False, chunksize=10000)
        return output_path

    start_time = datetime.now()
    try:
        output_path = with_retries(save_logs, market=market)
        if output_path is None:
            raise OSError(
                f"Échec de l’exportation des logs après retries pour {market}"
            )
        success_msg = f"Logs exportés sous {output_path} pour {market}: {len(trade_log)} lignes, policy_type={policy_type}"
        logger.info(success_msg)
        miya_speak(success_msg, tag="UTILS_MODEL", voice_profile="calm")
        AlertManager().send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
        save_snapshot(
            "export_logs",
            {
                "output_path": output_path,
                "num_rows": len(trade_log),
                "policy_type": policy_type,
                "market": market,
            },
            market=market,
        )
        log_df = pd.DataFrame(trade_log)
        checkpoint(log_df, data_type="trade_log", market=market)
        cloud_backup(log_df, data_type="trade_log", market=market)
    except Exception as e:
        error_msg = (
            f"Erreur export logs pour {market}: {str(e)}\n{traceback.format_exc()}"
        )
        logger.error(error_msg)
        miya_alerts(error_msg, tag="UTILS_MODEL", priority=5, voice_profile="urgent")
        AlertManager().send_alert(error_msg, priority=5)
        send_telegram_alert(error_msg)
        latency = (datetime.now() - start_time).total_seconds()
        log_performance(
            "export_logs",
            latency,
            success=False,
            error=str(e),
            policy_type=policy_type,
            market=market,
        )
        raise


def compute_metrics(
    df: Union[pd.DataFrame, List[float]], policy_type: str = "mlp", market: str = "ES"
) -> Dict[str, float]:
    """
    Calcule les métriques de performance à partir des données avec cache LRU.

    Args:
        df (Union[pd.DataFrame, List[float]]): DataFrame ou liste contenant au moins les récompenses.
        policy_type (str): Type de politique ("mlp", "transformer").
        market (str): Marché (ex. : ES, MNQ).

    Returns:
        Dict[str, float]: Métriques calculées (total_reward, sharpe_ratio, max_drawdown, etc.).
    """
    start_time = datetime.now()
    try:
        cache_key = f"{market}_{policy_type}_{hash(str(df))}"
        if cache_key in metric_cache:
            metrics = metric_cache[cache_key]
            metric_cache.move_to_end(cache_key)
            return metrics
        while len(metric_cache) > MAX_CACHE_SIZE:
            metric_cache.popitem(last=False)

        if isinstance(df, list):
            df = pd.DataFrame({"reward": df})
        validate_features(
            df, shap_features=(policy_type == "transformer"), market=market
        )
        metrics = {}
        if "reward" in df.columns:
            rewards = df["reward"].astype(float).dropna()
            metrics["total_reward"] = float(rewards.sum())
            metrics["sharpe_ratio"] = (
                float(rewards.mean() / rewards.std()) if rewards.std() != 0 else 0.0
            )
            metrics["max_drawdown"] = float(
                (rewards.cumsum() - rewards.cumsum().cummax()).min()
            )
            metrics["min_reward"] = float(rewards.min())
        if f"predicted_volatility_{market.lower()}" in df.columns:
            metrics["volatility_mean"] = float(
                df[f"predicted_volatility_{market.lower()}"].mean()
            )
        if "neural_regime" in df.columns:
            metrics["regime_distribution"] = (
                df["neural_regime"].value_counts(normalize=True).to_dict()
            )
        if "cnn_pressure" in df.columns:
            metrics["pressure_mean"] = float(df["cnn_pressure"].mean())

        for metric, value in metrics.items():
            if (
                metric == "total_reward"
                and value < PERFORMANCE_THRESHOLDS["min_mean_reward"]
            ):
                warning_msg = f"Seuil non atteint pour {market}: {metric} ({value:.2f}) < {PERFORMANCE_THRESHOLDS['min_mean_reward']}"
                logger.warning(warning_msg)
                miya_speak(warning_msg, tag="UTILS_MODEL", level="warning")
                AlertManager().send_alert(warning_msg, priority=2)
                send_telegram_alert(warning_msg)
            if (
                metric == "sharpe_ratio"
                and value < PERFORMANCE_THRESHOLDS["min_sharpe"]
            ):
                warning_msg = f"Seuil non atteint pour {market}: {metric} ({value:.2f}) < {PERFORMANCE_THRESHOLDS['min_sharpe']}"
                logger.warning(warning_msg)
                miya_speak(warning_msg, tag="UTILS_MODEL", level="warning")
                AlertManager().send_alert(warning_msg, priority=2)
                send_telegram_alert(warning_msg)
            if metric == "min_reward" and value < PERFORMANCE_THRESHOLDS["min_reward"]:
                warning_msg = f"Seuil non atteint pour {market}: {metric} ({value:.2f}) < {PERFORMANCE_THRESHOLDS['min_reward']}"
                logger.warning(warning_msg)
                miya_speak(warning_msg, tag="UTILS_MODEL", level="warning")
                AlertManager().send_alert(warning_msg, priority=2)
                send_telegram_alert(warning_msg)

        metric_cache[cache_key] = metrics
        success_msg = (
            f"Métriques calculées pour {market}: {metrics}, policy_type={policy_type}"
        )
        logger.info(success_msg)
        miya_speak(success_msg, tag="UTILS_MODEL", voice_profile="calm")
        AlertManager().send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
        save_snapshot(
            "compute_metrics",
            {"metrics": metrics, "policy_type": policy_type, "market": market},
            market=market,
        )
        latency = (datetime.now() - start_time).total_seconds()
        log_performance(
            "compute_metrics",
            latency,
            success=True,
            policy_type=policy_type,
            market=market,
        )
        return metrics
    except Exception as e:
        error_msg = (
            f"Erreur calcul métriques pour {market}: {str(e)}\n{traceback.format_exc()}"
        )
        logger.error(error_msg)
        miya_alerts(error_msg, tag="UTILS_MODEL", priority=5, voice_profile="urgent")
        AlertManager().send_alert(error_msg, priority=5)
        send_telegram_alert(error_msg)
        latency = (datetime.now() - start_time).total_seconds()
        log_performance(
            "compute_metrics",
            latency,
            success=False,
            error=str(e),
            policy_type=policy_type,
            market=market,
        )
        return {}


def clear_memory(
    max_log_size: Optional[int] = None, policy_type: str = "mlp", market: str = "ES"
) -> None:
    """
    Nettoie la mémoire en limitant la taille des logs.

    Args:
        max_log_size (Optional[int]): Taille maximale du buffer de logs.
        policy_type (str): Type de politique ("mlp", "transformer").
        market (str): Marché (ex. : ES, MNQ).
    """
    start_time = datetime.now()
    try:
        config = load_config(policy_type=policy_type, market=market)
        max_size = (
            max_log_size
            if max_log_size is not None
            else config.get("max_log_size", 10000)
        )
        global trade_log_buffer
        if len(trade_log_buffer) >= max_size:
            success_msg = f"Réduction buffer logs à {max_size} éléments pour {market}, policy_type={policy_type}"
            logger.info(success_msg)
            miya_speak(success_msg, tag="UTILS_MODEL", voice_profile="calm")
            AlertManager().send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            trade_log_buffer = deque(trade_log_buffer, maxlen=max_size)
        save_snapshot(
            "clear_memory",
            {"max_size": max_size, "policy_type": policy_type, "market": market},
            market=market,
        )
        latency = (datetime.now() - start_time).total_seconds()
        log_performance(
            "clear_memory",
            latency,
            success=True,
            policy_type=policy_type,
            market=market,
        )
    except Exception as e:
        error_msg = f"Erreur nettoyage mémoire pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        miya_alerts(error_msg, tag="UTILS_MODEL", priority=5, voice_profile="urgent")
        AlertManager().send_alert(error_msg, priority=5)
        send_telegram_alert(error_msg)
        latency = (datetime.now() - start_time).total_seconds()
        log_performance(
            "clear_memory",
            latency,
            success=False,
            error=str(e),
            policy_type=policy_type,
            market=market,
        )


# Cache global pour les métriques
metric_cache = OrderedDict()

if __name__ == "__main__":
    try:
        # Simulation de données pour test avec 350 features
        features = [f"feature_{i}" for i in range(350)]
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-05-13 09:00", periods=100, freq="1min"
                ),
                "reward": np.random.randn(100),
                "predicted_volatility_es": np.random.uniform(0.5, 2, 100),
                "neural_regime": np.random.choice([0, 1, 2], 100),
                "cnn_pressure": np.random.randn(100),
                "step": range(100),
                "action": np.random.uniform(-1, 1, 100),
                "bid_size_level_1": np.random.randint(100, 500, 100),
                "ask_size_level_1": np.random.randint(100, 500, 100),
                "trade_frequency_1s": np.random.uniform(0.1, 10, 100),
                "spread_avg_1min": np.random.uniform(0.01, 0.05, 100),
                "close": np.random.uniform(4000, 5000, 100),
                "open": np.random.uniform(4000, 5000, 100),
                "high": np.random.uniform(4000, 5000, 100),
                "low": np.random.uniform(4000, 5000, 100),
                "volume": np.random.randint(1000, 5000, 100),
                "atr_14_es": np.random.uniform(0.1, 1.0, 100),
                "adx_14": np.random.uniform(10, 50, 100),
                "gex_es": np.random.uniform(-1000, 1000, 100),
                "oi_peak_call_near": np.random.uniform(100, 1000, 100),
                "gamma_wall_call": np.random.uniform(100, 1000, 100),
                "gamma_wall_put": np.random.uniform(100, 1000, 100),
                **{f: np.random.uniform(0, 1, 100) for f in features},
            }
        )

        # Test de validate_features
        validate_features(data, step=0, market="ES")
        logger.info("✅ Test validate_features réussi")
        miya_speak("Test validate_features réussi", tag="TEST")

        # Test de EarlyStoppingCallback
        for policy_type in ["mlp", "transformer"]:
            callback = EarlyStoppingCallback(policy_type=policy_type, market="ES")

            class MockModel:
                def __init__(self):
                    self.env = TradingEnv(str(BASE_DIR / "config/algo_config.yaml"))
                    self.env.data = data
                    self.env.policy_type = policy_type

            callback.model = MockModel()
            for step in range(10):
                callback.n_calls = step * 1000
                callback.locals = {"rewards": data["reward"][: step + 1].tolist()}
                continue_training = callback._on_step()
                logger.info(
                    f"Step {callback.n_calls}, Continue: {continue_training}, Policy: {policy_type}"
                )
                miya_speak(
                    f"Step {callback.n_calls}, Continue: {continue_training}, Policy: {policy_type}",
                    tag="TEST",
                )

        # Test de export_logs
        trade_log = [
            {
                "step": i,
                "action": float(data["action"][i]),
                "reward": float(data["reward"][i]),
            }
            for i in range(5)
        ]
        export_logs(
            trade_log, str(BASE_DIR / "data/logs"), "test_export.csv", market="ES"
        )
        logger.info("✅ Test export_logs réussi")
        miya_speak("Test export_logs réussi", tag="TEST")

        # Test de compute_metrics
        metrics = compute_metrics(data, market="ES")
        logger.info(f"Métriques: {metrics}")
        miya_speak(f"Métriques: {metrics}", tag="TEST")

        # Test de clear_memory
        global trade_log_buffer
        trade_log_buffer = deque([{"step": i} for i in range(15000)])
        clear_memory(market="ES")
        logger.info(f"Taille buffer après nettoyage: {len(trade_log_buffer)}")
        miya_speak(
            f"Taille buffer après nettoyage: {len(trade_log_buffer)}", tag="TEST"
        )

    except Exception as e:
        error_msg = f"Erreur test principal: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        miya_alerts(error_msg, tag="UTILS_MODEL", priority=5, voice_profile="urgent")
        AlertManager().send_alert(error_msg, priority=5)
        send_telegram_alert(error_msg)
