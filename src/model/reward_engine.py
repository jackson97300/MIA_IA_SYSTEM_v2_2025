# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/model/reward_engine.py
# Calcule les récompenses pour l'entraînement SAC/PPO/DDPG, intégrant des récompenses adaptatives (méthode 5).
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle :
# Calcule les récompenses pour l’entraînement SAC/PPO/DDPG, intégrant profit, risque, drawdown, risque de crash de
# liquidité, et récompenses adaptatives basées sur news_impact_score et predicted_vix (méthode 5).
#
# Dépendances :
# - numpy>=1.26.4,<2.0.0, pandas>=2.0.0,<3.0.0, psutil>=5.9.8,<6.0.0, pyyaml>=6.0.0,<7.0.0,
#   boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0
# - src/features/neural_pipeline.py
# - src/envs/trading_env.py
# - src/model/utils/miya_console.py
# - src/model/utils/config_manager.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/feature_sets.yaml (pour les 350 features et 150 SHAP features)
# - Données de l’environnement (via TradingEnv.data)
#
# Outputs :
# - Logs dans data/logs/reward_engine.log
# - Logs de performance dans data/logs/reward_engine_performance.csv
# - Snapshots compressés dans data/cache/reward_engine/<market>/*.json.gz
# - Sauvegardes incrémentielles dans data/checkpoints/reward_engine/<market>/*.json.gz
#
# Notes :
# - Intègre récompenses adaptatives (méthode 5) avec news_impact_score et predicted_vix.
# - Valide 350 features pour l’entraînement et 150 SHAP features pour l’inférence.
# - Utilise IQFeed comme source de données via TradingEnv.
# - Suppression de toute référence à dxFeed, obs_t, 320/81 features.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des données.
# - Tests unitaires disponibles dans tests/test_reward_engine.py.

import gzip
import json
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import boto3
import numpy as np
import pandas as pd
import psutil
from loguru import logger

from src.envs.trading_env import TradingEnv
from src.features.neural_pipeline import NeuralPipeline
from src.model.utils.config_manager import get_config
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.utils.telegram_alert import send_telegram_alert

# Configuration logging
logger.remove()
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOG_DIR = BASE_DIR / "data" / "logs"
CACHE_DIR = BASE_DIR / "data" / "cache" / "reward_engine"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints" / "reward_engine"
PERF_LOG_PATH = LOG_DIR / "reward_engine_performance.csv"
SNAPSHOT_DIR = CACHE_DIR
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "reward_engine.log", rotation="10 MB", level="INFO", encoding="utf-8"
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
    "min_reward": -100.0,  # Récompense minimale acceptable
    "max_volatility": 2.0,  # Volatilité prédite maximale
    "min_balance": 9000.0,  # Solde minimum du portefeuille
    "max_drawdown": -1000.0,  # Drawdown maximal toléré
    "max_spread": 0.5,  # Spread moyen maximal (indicateur de liquidité)
    "min_trade_frequency": 5.0,  # Fréquence minimale des trades par seconde
}

# Cache global pour les récompenses
reward_cache = OrderedDict()


def log_performance(
    operation: str,
    latency: float,
    success: bool = True,
    error: str = None,
    market: str = "ES",
    **kwargs,
) -> None:
    """
    Enregistre les performances (CPU, mémoire, latence) dans reward_engine_performance.csv.

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
            miya_alerts(alert_msg, tag="REWARD", priority=5)
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
        miya_alerts(error_msg, tag="REWARD", priority=3)
        AlertManager().send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)


def save_snapshot(
    snapshot_type: str, data: Dict, market: str = "ES", compress: bool = True
) -> None:
    """
    Sauvegarde un instantané JSON des résultats, compressé avec gzip.

    Args:
        snapshot_type (str): Type de snapshot (ex. : calculate_reward).
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
            miya_alerts(alert_msg, tag="REWARD", priority=3)
            AlertManager().send_alert(alert_msg, priority=3)
            send_telegram_alert(alert_msg)
        latency = (datetime.now() - start_time).total_seconds()
        success_msg = f"Snapshot {snapshot_type} sauvegardé pour {market}: {save_path}"
        logger.info(success_msg)
        miya_speak(success_msg, tag="REWARD")
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
        miya_alerts(error_msg, tag="REWARD", priority=3)
        AlertManager().send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)
        log_performance("save_snapshot", 0, success=False, error=str(e), market=market)


def checkpoint(
    data: pd.DataFrame, data_type: str = "reward_engine_state", market: str = "ES"
) -> None:
    """
    Sauvegarde incrémentielle des données toutes les 5 minutes avec versionnage (5 versions).

    Args:
        data (pd.DataFrame): Données à sauvegarder.
        data_type (str): Type de données (ex. : reward_engine_state).
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
            checkpoint_dir / f"reward_engine_{data_type}_{timestamp}.json.gz"
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
        miya_speak(success_msg, tag="REWARD")
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
        miya_alerts(error_msg, tag="REWARD", priority=3)
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
    data: pd.DataFrame, data_type: str = "reward_engine_state", market: str = "ES"
) -> None:
    """
    Sauvegarde distribuée des données vers AWS S3 toutes les 15 minutes.

    Args:
        data (pd.DataFrame): Données à sauvegarder.
        data_type (str): Type de données (ex. : reward_engine_state).
        market (str): Marché (ex. : ES, MNQ).
    """
    try:
        start_time = datetime.now()
        config = get_config(BASE_DIR / "config/reward_engine_config.yaml")
        if not config.get("s3_bucket"):
            warning_msg = (
                f"S3 bucket non configuré, sauvegarde cloud ignorée pour {market}"
            )
            logger.warning(warning_msg)
            miya_alerts(warning_msg, tag="REWARD", priority=3)
            AlertManager().send_alert(warning_msg, priority=3)
            send_telegram_alert(warning_msg)
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{config['s3_prefix']}reward_engine_{data_type}_{market}_{timestamp}.csv.gz"
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
        miya_speak(success_msg, tag="REWARD")
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
        miya_alerts(error_msg, tag="REWARD", priority=3)
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
                miya_alerts(error_msg, tag="REWARD", priority=4)
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
            miya_speak(warning_msg, tag="REWARD", level="warning")
            AlertManager().send_alert(warning_msg, priority=3)
            send_telegram_alert(warning_msg)
            time.sleep(delay)


def validate_features(
    data: pd.DataFrame,
    current_step: int,
    shap_features: bool = False,
    market: str = "ES",
) -> None:
    """
    Valide la présence des 350 features (entraînement) ou 150 SHAP features (inférence) et impute les valeurs manquantes.

    Args:
        data (pd.DataFrame): Données de l’environnement.
        current_step (int): Étape actuelle.
        shap_features (bool): Si True, valide les 150 SHAP features; sinon, valide les 350 features.
        market (str): Marché (ex. : ES, MNQ).

    Raises:
        ValueError: Si des colonnes critiques sont manquantes ou non numériques.
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
                f"Nombre de features incorrect pour {market}: {len(expected_cols)} au lieu de {expected_len}"
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
            miya_alerts(alert_msg, tag="REWARD", priority=3)
            AlertManager().send_alert(alert_msg, priority=3)
            send_telegram_alert(alert_msg)
        if missing_cols:
            warning_msg = f"Colonnes manquantes au step {current_step} pour {market} ({'SHAP' if shap_features else 'full'}): {missing_cols}"
            logger.warning(warning_msg)
            miya_speak(warning_msg, tag="REWARD", level="warning")
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
                "step": current_step,
                "num_columns": len(data.columns),
                "missing_columns": missing_cols,
                "shap_features": shap_features,
                "confidence_drop_rate": confidence_drop_rate,
                "market": market,
            },
            market=market,
        )
    except Exception as e:
        error_msg = f"Erreur validation features au step {current_step} pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        miya_alerts(error_msg, tag="REWARD", priority=5)
        AlertManager().send_alert(error_msg, priority=5)
        send_telegram_alert(error_msg)
        raise


def calculate_liquidity_crash_risk(
    data: pd.DataFrame, current_step: int, market: str = "ES"
) -> float:
    """
    Calcule le risque de crash de liquidité basé sur le spread et la fréquence des trades.

    Args:
        data (pd.DataFrame): Données de l’environnement.
        current_step (int): Étape actuelle.
        market (str): Marché (ex. : ES, MNQ).

    Returns:
        float: Risque de crash de liquidité (0.0 à 1.0, où 1.0 est un risque élevé).
    """
    try:
        spread = (
            float(data["spread_avg_1min"].iloc[current_step])
            if "spread_avg_1min" in data.columns
            else 0.25
        )
        trade_freq = (
            float(data["trade_frequency_1s"].iloc[current_step])
            if "trade_frequency_1s" in data.columns
            else 10.0
        )

        spread_risk = min(spread / PERFORMANCE_THRESHOLDS["max_spread"], 1.0)
        trade_freq_risk = max(
            0.0, 1.0 - trade_freq / PERFORMANCE_THRESHOLDS["min_trade_frequency"]
        )
        liquidity_risk = 0.7 * spread_risk + 0.3 * trade_freq_risk
        result = min(max(liquidity_risk, 0.0), 1.0)
        save_snapshot(
            "calculate_liquidity_crash_risk",
            {
                "step": current_step,
                "spread": spread,
                "trade_freq": trade_freq,
                "liquidity_risk": result,
                "market": market,
            },
            market=market,
        )
        return result
    except Exception as e:
        error_msg = f"Erreur calcul risque liquidité au step {current_step} pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.warning(error_msg)
        miya_alerts(error_msg, tag="REWARD", priority=3)
        AlertManager().send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)
        return 0.5


def calculate_reward(
    state: Dict,
    env: Optional[TradingEnv] = None,
    mode: str = "trend",
    policy_type: str = "mlp",
    market: str = "ES",
) -> float:
    """
    Calcule la récompense pour SAC/PPO/DDPG avec récompenses adaptatives (méthode 5).

    Args:
        state (Dict): État contenant 'profit', 'risk', 'news_impact_score', 'predicted_vix'.
        env (TradingEnv, optional): Instance de TradingEnv pour les données et l’état.
        mode (str): Mode du modèle ("trend", "range", "defensive"). Par défaut "trend".
        policy_type (str): Type de politique ("mlp", "transformer"). Par défaut "mlp".
        market (str): Marché (ex. : ES, MNQ).

    Returns:
        float: Récompense calculée.

    Raises:
        KeyError: Si les clés nécessaires dans state ou env.data sont absentes.
        ValueError: Si les validations échouent.
    """
    start_time = datetime.now()
    try:
        # Vérifier le cache
        cache_key = f"{market}_{mode}_{policy_type}_{hash(str(state))}"
        if cache_key in reward_cache:
            reward = reward_cache[cache_key]
            reward_cache.move_to_end(cache_key)
            return reward
        while len(reward_cache) > MAX_CACHE_SIZE:
            reward_cache.popitem(last=False)

        # Récompense adaptative (méthode 5)
        required_keys = ["profit", "risk", "news_impact_score", "predicted_vix"]
        missing_keys = [key for key in required_keys if key not in state]
        if missing_keys:
            raise KeyError(f"Clés manquantes dans state pour {market}: {missing_keys}")

        profit = float(state["profit"])
        risk = float(state["risk"])
        news_impact_score = float(state["news_impact_score"])
        predicted_vix = float(state["predicted_vix"])
        reward = profit - risk + news_impact_score + predicted_vix

        # Si env est fourni, appliquer les ajustements
        if env is not None:
            current_step = env.current_step
            validate_features(
                env.data,
                current_step,
                shap_features=(policy_type == "transformer"),
                market=market,
            )

            # Paramètres du marché
            point_value = env.config.get("market_params", {}).get("point_value", 50.0)
            trade_duration = int(state.get("duration", 1))
            float(state.get("entry_price", env.data["close"].iloc[current_step]))
            exit_price = float(
                state.get("exit_price", env.data["close"].iloc[current_step])
            )

            # Récupérer les prédictions neuronales
            volatility_col = f"predicted_volatility_{market.lower()}"
            predicted_volatility = (
                float(env.data[volatility_col].iloc[current_step])
                if volatility_col in env.data.columns
                else 0.0
            )
            neural_regime = (
                int(env.data["neural_regime"].iloc[current_step])
                if "neural_regime" in env.data.columns
                else None
            )
            cnn_pressure = (
                float(env.data["cnn_pressure"].iloc[current_step])
                if "cnn_pressure" in env.data.columns
                else 0.0
            )

            # Recalcul avec NeuralPipeline si nécessaire
            if neural_regime is None or predicted_volatility == 0.0:

                def compute_neural_metrics():
                    neural_pipeline = NeuralPipeline(
                        window_size=50,
                        base_features=350,
                        config_path=str(BASE_DIR / "config/model_params.yaml"),
                    )
                    neural_pipeline.load_models()
                    window_start = max(0, current_step - 49)
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
                                f"atr_14_{market.lower()}",
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
                                f"gex_{market.lower()}",
                                "oi_peak_call_near",
                                "gamma_wall_call",
                                "gamma_wall_put",
                            ]
                        ]
                        .iloc[window_start:window_end]
                        .fillna(0)
                    )
                    orderflow_data = (
                        env.data[["timestamp", "bid_size_level_1", "ask_size_level_1"]]
                        .iloc[window_start:window_end]
                        .fillna(0)
                    )
                    return neural_pipeline.run(raw_data, options_data, orderflow_data)

                neural_result = with_retries(compute_neural_metrics, market=market)
                if neural_result:
                    predicted_volatility = float(neural_result["volatility"][-1])
                    neural_regime = int(neural_result["regime"][-1])
                    cnn_pressure = float(neural_result["features"][-1, -1])
                    success_msg = f"Prédictions neuronales calculées au step {current_step} pour {market}: Volatility={predicted_volatility:.2f}, Regime={neural_regime}, Pressure={cnn_pressure:.2f}"
                    logger.debug(success_msg)
                    miya_speak(success_msg, tag="REWARD")
                    AlertManager().send_alert(success_msg, priority=1)
                    send_telegram_alert(success_msg)
                else:
                    predicted_volatility = 0.0
                    neural_regime = None
                    cnn_pressure = 0.0

            # Ajustements basés sur la volatilité et la pression
            volatility_adjustment = (
                1 / (1 + predicted_volatility) if predicted_volatility > 0 else 1.0
            )
            pressure_adjustment = 1 + (cnn_pressure * 0.1) if cnn_pressure != 0 else 1.0

            # Calcul du risque de crash de liquidité
            liquidity_crash_risk = calculate_liquidity_crash_risk(
                env.data, current_step, market=market
            )
            liquidity_adjustment = 1.0 - liquidity_crash_risk

            # Ajustements par mode
            if mode == "trend":
                reward = profit * 1.5 if profit > 0 else profit
                if trade_duration > 10 and profit <= 0:
                    reward -= 0.1 * trade_duration * point_value
                if neural_regime == 0:
                    reward *= 1.2
                elif neural_regime is not None and neural_regime != 0:
                    reward *= 0.8
                reward *= (
                    volatility_adjustment * pressure_adjustment * liquidity_adjustment
                )

            elif mode == "range":
                vwap = (
                    float(env.data[f"vwap_{market.lower()}"].iloc[current_step])
                    if f"vwap_{market.lower()}" in env.data.columns
                    else exit_price
                )
                price_diff = abs(exit_price - vwap)
                reward += (
                    10 * point_value
                    if price_diff
                    < env.config.get("thresholds", {}).get("vwap_slope_max_range", 0.01)
                    else -5 * point_value
                )
                if neural_regime == 1:
                    reward *= 1.2
                elif neural_regime is not None and neural_regime != 1:
                    reward *= 0.8
                reward *= (
                    volatility_adjustment * pressure_adjustment * liquidity_adjustment
                )

            elif mode == "defensive":
                reward = profit if profit > 0 else profit * 2
                if env.max_drawdown < env.config.get("thresholds", {}).get(
                    "max_drawdown", -0.10
                ):
                    reward -= abs(env.max_drawdown) * point_value
                if neural_regime == 2:
                    reward *= 1.2
                elif neural_regime is not None and neural_regime != 2:
                    reward *= 0.8
                reward *= (
                    volatility_adjustment * pressure_adjustment * liquidity_adjustment
                )

            # Pénalité pour drawdown excessif
            if env.max_drawdown < PERFORMANCE_THRESHOLDS["max_drawdown"]:
                penalty = (
                    abs(env.max_drawdown - PERFORMANCE_THRESHOLDS["max_drawdown"])
                    * point_value
                    * 0.1
                )
                reward -= penalty
                warning_msg = f"Pénalité drawdown appliquée pour {market}: -{penalty:.2f} au step {current_step}"
                logger.warning(warning_msg)
                miya_alerts(warning_msg, tag="REWARD", priority=3)
                AlertManager().send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)

            # Validation des seuils de performance
            if reward < PERFORMANCE_THRESHOLDS["min_reward"]:
                warning_msg = f"Récompense ({reward:.2f}) < {PERFORMANCE_THRESHOLDS['min_reward']} au step {current_step} pour {market}"
                logger.warning(warning_msg)
                miya_alerts(warning_msg, tag="REWARD", priority=3)
                AlertManager().send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
            if predicted_volatility > PERFORMANCE_THRESHOLDS["max_volatility"]:
                warning_msg = f"Volatilité prédite ({predicted_volatility:.2f}) > {PERFORMANCE_THRESHOLDS['max_volatility']} au step {current_step} pour {market}"
                logger.warning(warning_msg)
                miya_alerts(warning_msg, tag="REWARD", priority=3)
                AlertManager().send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
            if env.balance < PERFORMANCE_THRESHOLDS["min_balance"]:
                warning_msg = f"Balance ({env.balance:.2f}) < {PERFORMANCE_THRESHOLDS['min_balance']} au step {current_step} pour {market}"
                logger.warning(warning_msg)
                miya_alerts(warning_msg, tag="REWARD", priority=3)
                AlertManager().send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
            if env.max_drawdown < PERFORMANCE_THRESHOLDS["max_drawdown"]:
                warning_msg = f"Max Drawdown ({env.max_drawdown:.2f}) < {PERFORMANCE_THRESHOLDS['max_drawdown']} au step {current_step} pour {market}"
                logger.warning(warning_msg)
                miya_alerts(warning_msg, tag="REWARD", priority=3)
                AlertManager().send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
            if liquidity_crash_risk > 0.5:
                warning_msg = f"Risque de crash de liquidité élevé ({liquidity_crash_risk:.2f}) au step {current_step} pour {market}"
                logger.warning(warning_msg)
                miya_alerts(warning_msg, tag="REWARD", priority=3)
                AlertManager().send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)

        # Mettre à jour le cache
        reward_cache[cache_key] = reward

        # Sauvegarder un snapshot
        snapshot_data = {
            "reward": reward,
            "profit": profit,
            "risk": risk,
            "news_impact_score": news_impact_score,
            "predicted_vix": predicted_vix,
            "mode": mode,
            "policy_type": policy_type,
            "market": market,
        }
        save_snapshot("calculate_reward", snapshot_data, market=market)

        # Sauvegarder un checkpoint
        checkpoint_data = pd.DataFrame(
            [
                {
                    "timestamp": datetime.now().isoformat(),
                    "reward": reward,
                    "profit": profit,
                    "news_impact_score": news_impact_score,
                    "predicted_vix": predicted_vix,
                    "mode": mode,
                    "policy_type": policy_type,
                    "market": market,
                }
            ]
        )
        checkpoint(checkpoint_data, data_type="reward", market=market)
        cloud_backup(checkpoint_data, data_type="reward", market=market)

        # Log final
        latency = (datetime.now() - start_time).total_seconds()
        log_performance(
            "calculate_reward",
            latency,
            success=True,
            mode=mode,
            policy_type=policy_type,
            reward=reward,
            profit=profit,
            news_impact_score=news_impact_score,
            predicted_vix=predicted_vix,
            market=market,
        )
        success_msg = f"Récompense calculée pour mode {mode}, policy_type={policy_type} pour {market}: {reward:.2f}, Profit: {profit:.2f}, News Impact: {news_impact_score:.2f}, Predicted VIX: {predicted_vix:.2f}"
        logger.info(success_msg)
        miya_speak(success_msg, tag="REWARD")
        AlertManager().send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
        return float(reward)

    except Exception as e:
        error_msg = f"Erreur dans calculate_reward, mode={mode}, step={getattr(env, 'current_step', 'N/A')} pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        miya_alerts(error_msg, tag="REWARD", priority=5)
        AlertManager().send_alert(error_msg, priority=5)
        send_telegram_alert(error_msg)
        latency = (datetime.now() - start_time).total_seconds()
        log_performance(
            "calculate_reward",
            latency,
            success=False,
            error=str(e),
            mode=mode,
            policy_type=policy_type,
            market=market,
        )
        return 0.0


if __name__ == "__main__":

    class MockEnv:
        def __init__(self):
            self.config = {
                "market_params": {"point_value": 50},
                "thresholds": {"max_drawdown": -1000.0, "vwap_slope_max_range": 0.01},
            }
            self.current_step = 0
            self.max_drawdown = -500.0
            self.balance = 10000.0
            feature_sets_path = BASE_DIR / "config/feature_sets.yaml"
            features_config = get_config(feature_sets_path)
            feature_cols = features_config.get("training", {}).get("features", [])[:350]
            self.data = pd.DataFrame(
                {
                    "timestamp": pd.date_range(
                        "2025-05-13 09:00", periods=100, freq="1min"
                    ),
                    "open": [5100] * 100,
                    "high": [5105] * 100,
                    "low": [5095] * 100,
                    "close": [5100] * 100,
                    "volume": [1000] * 100,
                    "atr_14_es": [1.0] * 100,
                    "adx_14": [25] * 100,
                    "gex_es": [500] * 100,
                    "oi_peak_call_near": [10000] * 100,
                    "gamma_wall_call": [0.02] * 100,
                    "gamma_wall_put": [0.02] * 100,
                    "bid_size_level_1": [100] * 100,
                    "ask_size_level_1": [120] * 100,
                    "vwap_es": [5100] * 100,
                    "spread_avg_1min": [0.3] * 100,
                    "predicted_volatility_es": [1.5] * 100,
                    "neural_regime": [0] * 100,
                    "cnn_pressure": [0.5] * 100,
                    "trade_frequency_1s": [8] * 100,
                    **{
                        f"neural_feature_{i}": [np.random.randn()] * 100
                        for i in range(8)
                    },
                    **{
                        col: [np.random.uniform(0, 1)] * 100
                        for col in feature_cols
                        if col
                        not in [
                            "timestamp",
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                            "atr_14_es",
                            "adx_14",
                            "gex_es",
                            "oi_peak_call_near",
                            "gamma_wall_call",
                            "gamma_wall_put",
                            "bid_size_level_1",
                            "ask_size_level_1",
                            "vwap_es",
                            "spread_avg_1min",
                            "predicted_volatility_es",
                            "neural_regime",
                            "cnn_pressure",
                            "trade_frequency_1s",
                        ]
                    },
                }
            )

    env = MockEnv()
    state = {
        "profit": 100.0,
        "risk": 10.0,
        "news_impact_score": 0.5,
        "predicted_vix": 20.0,
        "duration": 5,
        "entry_price": 5100.0,
        "exit_price": 5105.0,
    }
    for market in ["ES", "MNQ"]:
        for mode in ["trend", "range", "defensive"]:
            for policy_type in ["mlp", "transformer"]:
                reward = calculate_reward(
                    state, env, mode, policy_type=policy_type, market=market
                )
                success_msg = f"Mode: {mode}, Policy: {policy_type}, Market: {market}, Récompense: {reward:.2f}"
                logger.info(success_msg)
                miya_speak(success_msg, tag="TEST")
                print(success_msg)
