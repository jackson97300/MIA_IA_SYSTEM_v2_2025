# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/backtest/backtest_sac.py
# Backtest un modèle SAC avec parallélisation et Monte Carlo, optimisé pour MIA_IA_SYSTEM_v2_2025 avec neural_pipeline.
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle :
# Simule des stratégies de trading basées sur un modèle SAC (Soft Actor-Critic) avec parallélisation
# et Monte Carlo. Intègre des récompenses adaptatives (méthode 5) basées sur news_impact_score
# et trade_success_prob, optimisé pour 350 features et 150 SHAP features pour l’inférence/fallback.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, pyyaml>=6.0.0,<7.0.0, joblib>=1.3.0,<2.0.0
# - stable_baselines3>=2.0.0,<3.0.0, gymnasium>=1.0.0,<2.0.0, boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0
# - src/features/neural_pipeline.py
# - src/model/utils/miya_console.py
# - src/features/context_aware_filter.py (implicite via feature_pipeline)
# - src/features/cross_asset_processor.py (implicite via feature_pipeline)
# - src/features/feature_pipeline.py (implicite pour les 350 features)
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/es_config.yaml
# - data/features/features_latest_filtered.csv (contenant timestamp, close, 350 features)
# - data/features/spotgamma_metrics.csv (pour features comme call_wall, put_wall)
# - data/iqfeed/news_data.csv (pour features contextuelles comme news_volume_spike_1m)
# - model/sac_models/<market> (modèles SAC pré-entraînés)
#
# Outputs :
# - Logs dans data/logs/backtest/backtest_<mode>_<timestamp>.csv
# - Logs de rejets dans data/logs/backtest/backtest_<mode>_rejections_<timestamp>.csv
# - Logs globaux dans data/logs/backtest_sac.log
# - Snapshots compressés dans data/cache/backtest/<market>/*.json.gz
# - Sauvegardes incrémentielles dans data/checkpoints/backtest/<market>/*.json.gz
#
# Notes :
# - Intègre 350 features pour l’entraînement et 150 SHAP features pour l’inférence/fallback.
# - Utilise exclusivement IQFeed comme source de données.
# - Suppression de toute référence à dxFeed, obs_t, 320/81 features.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des données.
# - Tests unitaires disponibles dans tests/test_backtest_sac.py.

import gzip
import json
import os
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import boto3
import numpy as np
import pandas as pd
import yaml
from gymnasium import spaces
from joblib import Parallel, delayed
from loguru import logger
from stable_baselines3 import SAC

from src.features.neural_pipeline import NeuralPipeline
from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import get_config
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.utils.telegram_alert import send_telegram_alert

# Configuration logging
logger.remove()
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOG_DIR = BASE_DIR / "data" / "logs" / "backtest"
CACHE_DIR = BASE_DIR / "data" / "cache" / "backtest"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints" / "backtest"
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "backtest_sac.log", rotation="10 MB", level="INFO", encoding="utf-8"
)
logger.add(lambda msg: print(msg, end=""), level="INFO")

# Constantes
MAX_RETRIES = 3
RETRY_DELAY = 2.0
MAX_CACHE_SIZE = 1000
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB

# Seuils de performance configurables
DEFAULT_PERFORMANCE_THRESHOLDS = {
    "min_profit_factor": 1.2,
    "min_sharpe": 0.5,
    "max_drawdown": -0.2,  # -20%
    "min_balance": 9000.0,
    "min_reward": -100.0,
}

# Cache global pour les résultats de backtest
backtest_cache = OrderedDict()


def load_config(
    config_path: str = str(BASE_DIR / "config/es_config.yaml"), market: str = "ES"
) -> Dict[str, Any]:
    """
    Charge la configuration depuis es_config.yaml avec retries.

    Args:
        config_path (str): Chemin vers le fichier de configuration.
        market (str): Marché (ex. : ES, MNQ).

    Returns:
        Dict[str, Any]: Configuration chargée.
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
        success_msg = f"Configuration {config_path} chargée pour {market}"
        logger.info(success_msg)
        miya_speak(success_msg, tag="BACKTEST_SAC", voice_profile="calm")
        AlertManager().send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
        return config.get("backtest_sac", {})
    except Exception as e:
        error_msg = f"Erreur chargement configuration {config_path} pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        miya_alerts(error_msg, tag="BACKTEST_SAC", priority=4, voice_profile="urgent")
        AlertManager().send_alert(error_msg, priority=4)
        send_telegram_alert(error_msg)
        return {
            "data_path": str(BASE_DIR / "data/features/features_latest_filtered.csv"),
            "model_dir": str(BASE_DIR / "model/sac_models"),
            "log_dir": str(BASE_DIR / "data/logs/backtest"),
            "sequence_length": 50,
            "n_simulations": 100,
            "min_rows": 100,
            "performance_thresholds": DEFAULT_PERFORMANCE_THRESHOLDS,
        }


def with_retries(
    func: callable,
    max_attempts: int = MAX_RETRIES,
    delay_base: float = RETRY_DELAY,
    market: str = "ES",
) -> Any:
    """
    Exécute une fonction avec retries exponentiels.

    Args:
        func (callable): Fonction à exécuter.
        max_attempts (int): Nombre maximum de tentatives.
        delay_base (float): Base pour le délai exponentiel.
        market (str): Marché (ex. : ES, MNQ).

    Returns:
        Any: Résultat de la fonction ou None si échec.
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
                    error_msg, tag="BACKTEST_SAC", priority=4, voice_profile="urgent"
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
            miya_speak(warning_msg, tag="BACKTEST_SAC", level="warning")
            AlertManager().send_alert(warning_msg, priority=3)
            send_telegram_alert(warning_msg)
            time.sleep(delay)


def log_performance(
    operation: str,
    latency: float,
    success: bool = True,
    error: str = None,
    market: str = "ES",
    **kwargs,
) -> None:
    """
    Enregistre les performances (CPU, mémoire, latence) dans backtest_performance.csv.

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
            miya_alerts(alert_msg, tag="BACKTEST_SAC", priority=5)
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
        perf_log_path = LOG_DIR / "backtest_performance.csv"

        def save_log():
            if not perf_log_path.exists():
                log_df.to_csv(perf_log_path, index=False, encoding="utf-8")
            else:
                log_df.to_csv(
                    perf_log_path, mode="a", header=False, index=False, encoding="utf-8"
                )

        with_retries(save_log, market=market)
        logger.info(f"Performance journalisée pour {operation}. CPU: {cpu_percent}%")
    except Exception as e:
        error_msg = f"Erreur journalisation performance pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        miya_alerts(error_msg, tag="BACKTEST_SAC", priority=3)
        AlertManager().send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)


def save_snapshot(
    snapshot_type: str, data: Dict, market: str = "ES", compress: bool = True
) -> None:
    """
    Sauvegarde un instantané JSON des résultats, compressé avec gzip.

    Args:
        snapshot_type (str): Type de snapshot (ex. : backtest_sac).
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
        snapshot_dir = CACHE_DIR / market
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
            miya_alerts(alert_msg, tag="BACKTEST_SAC", priority=3)
            AlertManager().send_alert(alert_msg, priority=3)
            send_telegram_alert(alert_msg)
        latency = (datetime.now() - start_time).total_seconds()
        success_msg = f"Snapshot {snapshot_type} sauvegardé pour {market}: {save_path}"
        logger.info(success_msg)
        miya_speak(success_msg, tag="BACKTEST_SAC", voice_profile="calm")
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
        miya_alerts(error_msg, tag="BACKTEST_SAC", priority=3)
        AlertManager().send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)
        log_performance("save_snapshot", 0, success=False, error=str(e), market=market)


def checkpoint(
    data: pd.DataFrame, data_type: str = "backtest_state", market: str = "ES"
) -> None:
    """
    Sauvegarde incrémentielle des données toutes les 5 minutes avec versionnage (5 versions).

    Args:
        data (pd.DataFrame): Données à sauvegarder.
        data_type (str): Type de données (ex. : backtest_state).
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
        checkpoint_path = checkpoint_dir / f"backtest_{data_type}_{timestamp}.json.gz"
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
        miya_speak(success_msg, tag="BACKTEST_SAC", voice_profile="calm")
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
        miya_alerts(error_msg, tag="BACKTEST_SAC", priority=3)
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
    data: pd.DataFrame, data_type: str = "backtest_state", market: str = "ES"
) -> None:
    """
    Sauvegarde distribuée des données vers AWS S3 toutes les 15 minutes.

    Args:
        data (pd.DataFrame): Données à sauvegarder.
        data_type (str): Type de données (ex. : backtest_state).
        market (str): Marché (ex. : ES, MNQ).
    """
    try:
        start_time = datetime.now()
        config = get_config(BASE_DIR / "config/es_config.yaml")
        if not config.get("s3_bucket"):
            warning_msg = (
                f"S3 bucket non configuré, sauvegarde cloud ignorée pour {market}"
            )
            logger.warning(warning_msg)
            miya_alerts(warning_msg, tag="BACKTEST_SAC", priority=3)
            AlertManager().send_alert(warning_msg, priority=3)
            send_telegram_alert(warning_msg)
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = (
            f"{config['s3_prefix']}backtest_{data_type}_{market}_{timestamp}.csv.gz"
        )
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
        miya_speak(success_msg, tag="BACKTEST_SAC", voice_profile="calm")
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
        miya_alerts(error_msg, tag="BACKTEST_SAC", priority=3)
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


class TradingEnv:
    """
    Environnement de trading simulé pour le backtest SAC.
    """

    def __init__(self, config_path: str, market: str = "ES"):
        self.config = load_config(config_path, market=market)
        self.data = None
        self.current_step = 0
        self.mode = "defensive"
        self.policy_type = "mlp"
        self.sequence_length = self.config.get("sequence_length", 50)
        self.balance = 10000.0
        self.position = 0.0
        self.entry_price = 0.0
        self.observation_space = None
        self.features = self.load_features()
        self.market = market

    def load_features(self) -> List[str]:
        """Charge les 350 features depuis feature_sets.yaml."""
        try:
            feature_sets_path = BASE_DIR / "config/feature_sets.yaml"
            features_config = get_config(feature_sets_path)
            features = features_config.get("training", {}).get("features", [])[:350]
            if len(features) != 350:
                raise ValueError(
                    f"Attendu 350 features pour {self.market}, trouvé {len(features)}"
                )
            return features
        except Exception as e:
            error_msg = f"Erreur chargement features pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            miya_alerts(error_msg, tag="BACKTEST_SAC", priority=4)
            raise

    def reset(self):
        self.current_step = 0
        self.balance = 10000.0
        self.position = 0.0
        self.entry_price = 0.0
        obs = self.data[self.features].iloc[0].values.astype(np.float32)
        return obs, {}

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.data):
            return None, 0.0, True, False, {}
        obs = self.data[self.features].iloc[self.current_step].values.astype(np.float32)
        price = self.data["close"].iloc[self.current_step]
        news_impact = self.data.get("news_impact_score", pd.Series([0.5])).iloc[
            self.current_step
        ]
        trade_success_prob = self.data.get("trade_success_prob", pd.Series([0.5])).iloc[
            self.current_step
        ]

        # Calcul de la récompense
        action_value = action[
            0
        ]  # Action entre [-1, 1] : -1=vendre, 0=neutre, 1=acheter
        reward = 0.0
        if action_value > 0 and self.position == 0:  # Achat
            self.position = self.balance * 0.1 / price  # 10% du capital
            self.entry_price = price
            self.balance -= self.position * price
        elif action_value < 0 and self.position > 0:  # Vente
            profit = (price - self.entry_price) * self.position
            reward = profit * (
                1 + news_impact * trade_success_prob
            )  # Récompense adaptative
            self.balance += self.position * price
            self.position = 0
            self.entry_price = 0
        elif self.position > 0:  # Maintien de la position
            unrealized_profit = (price - self.entry_price) * self.position
            reward = (
                unrealized_profit * 0.01 * trade_success_prob
            )  # Récompense partielle

        done = self.current_step >= len(self.data) - 1
        info = {"balance": self.balance, "position": self.position}
        return obs, reward, done, False, info


class MarketRegimeDetector:
    """
    Détecteur de régime de marché simulé.
    """

    def __init__(self, config_path: str, market: str = "ES"):
        self.config = load_config(config_path, market=market)
        self.market = market

    def detect(self, data: pd.DataFrame, step: int) -> Tuple[str, Dict]:
        try:
            neural_regime = data.get("neural_regime", pd.Series([1])).iloc[step]
            regime_map = {0: "trend", 1: "range", 2: "defensive"}
            regime = regime_map.get(int(neural_regime), "range")
            details = {"neural_regime": int(neural_regime)}
            return regime, details
        except Exception as e:
            error_msg = f"Erreur détection régime pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            miya_alerts(error_msg, tag="BACKTEST_SAC", priority=4)
            return "range", {"neural_regime": 1}


def validate_features(
    data: pd.DataFrame, step: int, shap_features: bool = False, market: str = "ES"
) -> None:
    """
    Valide la présence des 350 features (entraînement) ou 150 SHAP features (inférence) et impute les valeurs manquantes.

    Args:
        data (pd.DataFrame): Données à valider.
        step (int): Étape actuelle pour logging.
        shap_features (bool): Si True, valide les 150 SHAP features; sinon, valide les 350 features.
        market (str): Marché (ex. : ES, MNQ).
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
            miya_alerts(alert_msg, tag="BACKTEST_SAC", priority=3)
            AlertManager().send_alert(alert_msg, priority=3)
            send_telegram_alert(alert_msg)
        if missing_cols:
            warning_msg = f"Colonnes manquantes au step {step} pour {market} ({'SHAP' if shap_features else 'full'}): {missing_cols}"
            logger.warning(warning_msg)
            miya_speak(warning_msg, tag="BACKTEST_SAC", level="warning")
            AlertManager().send_alert(warning_msg, priority=2)
            send_telegram_alert(warning_msg)
            for col in missing_cols:
                data[col] = 0.0

        critical_cols = [
            "bid_size_level_1",
            "ask_size_level_1",
            "trade_frequency_1s",
            "close",
            "volume",
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
                        .fillna(data[col].median())
                    )
                if col in [
                    "bid_size_level_1",
                    "ask_size_level_1",
                    "trade_frequency_1s",
                    "volume",
                ]:
                    if (data[col] <= 0).any():
                        warning_msg = f"Valeurs non positives dans {col} pour {market}, corrigées à 1e-6"
                        logger.warning(warning_msg)
                        miya_alerts(warning_msg, tag="BACKTEST_SAC", priority=4)
                        data[col] = data[col].clip(lower=1e-6)
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
        miya_alerts(error_msg, tag="BACKTEST_SAC", priority=5)
        AlertManager().send_alert(error_msg, priority=5)
        send_telegram_alert(error_msg)
        raise


def validate_data(
    data: pd.DataFrame, shap_features: bool = False, market: str = "ES"
) -> None:
    """
    Valide que les données contiennent 350 features (entraînement) ou 150 SHAP features (inférence/fallback).

    Args:
        data (pd.DataFrame): Données à valider.
        shap_features (bool): Si True, valide les 150 SHAP features; sinon, valide les 350 features.
        market (str): Marché (ex. : ES, MNQ).
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

        missing_cols = [
            col
            for col in expected_cols
            + ["close", "news_impact_score", "trade_success_prob"]
            if col not in data.columns
        ]
        null_count = data[expected_cols].isnull().sum().sum()
        confidence_drop_rate = (
            null_count / (len(data) * len(expected_cols))
            if (len(data) * len(expected_cols)) > 0
            else 0.0
        )
        if confidence_drop_rate > 0.5:
            alert_msg = f"Confidence_drop_rate élevé pour {market}: {confidence_drop_rate:.2f} ({null_count} valeurs nulles)"
            logger.warning(alert_msg)
            miya_alerts(alert_msg, tag="BACKTEST_SAC", priority=3)
            AlertManager().send_alert(alert_msg, priority=3)
            send_telegram_alert(alert_msg)
        if missing_cols:
            raise ValueError(f"Colonnes manquantes pour {market}: {missing_cols}")
        critical_cols = [
            "bid_size_level_1",
            "ask_size_level_1",
            "trade_frequency_1s",
            "close",
            "volume",
            "news_impact_score",
            "trade_success_prob",
        ]
        for col in critical_cols:
            if col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    raise ValueError(
                        f"Colonne {col} non numérique pour {market}: {data[col].dtype}"
                    )
                if (
                    col not in ["news_impact_score", "trade_success_prob"]
                    and (data[col] <= 0).any()
                ):
                    raise ValueError(
                        f"Colonne {col} contient des valeurs non positives pour {market}"
                    )
        save_snapshot(
            "validate_data",
            {
                "num_columns": len(data.columns),
                "missing_columns": missing_cols,
                "shap_features": shap_features,
                "confidence_drop_rate": confidence_drop_rate,
                "market": market,
            },
            market=market,
        )
    except Exception as e:
        error_msg = f"Erreur validation données pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        miya_alerts(error_msg, tag="BACKTEST_SAC", priority=5)
        AlertManager().send_alert(error_msg, priority=5)
        send_telegram_alert(error_msg)
        raise


def export_logs(
    log_data: List[Dict], log_dir: str, log_file: str, market: str = "ES"
) -> None:
    """
    Exporte les logs dans un fichier CSV.

    Args:
        log_data (List[Dict]): Données de log à exporter.
        log_dir (str): Répertoire de destination.
        log_file (str): Nom du fichier.
        market (str): Marché (ex. : ES, MNQ).
    """
    try:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)
        log_df = pd.DataFrame(log_data)

        def save_log():
            log_df.to_csv(log_path, index=False, encoding="utf-8")

        with_retries(save_log, market=market)
        success_msg = f"Logs exportés pour {market}: {log_file}"
        logger.info(success_msg)
        miya_speak(success_msg, tag="BACKTEST_SAC", voice_profile="calm")
        AlertManager().send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
        checkpoint(log_df, data_type="logs", market=market)
        cloud_backup(log_df, data_type="logs", market=market)
    except Exception as e:
        error_msg = (
            f"Erreur export logs pour {market}: {str(e)}\n{traceback.format_exc()}"
        )
        logger.error(error_msg)
        miya_alerts(error_msg, tag="BACKTEST_SAC", priority=3)
        AlertManager().send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)


def eval_sac_segment(
    model: SAC,
    data_segment: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    env: TradingEnv,
    detector: MarketRegimeDetector,
    neural_pipeline: NeuralPipeline,
    sequence_length: int = 50,
    policy_type: str = "mlp",
    market: str = "ES",
) -> Tuple[float, float, List[float], List[Dict], List[int], List[Dict]]:
    """
    Évalue un segment de données pour le backtest SAC.

    Args:
        model (SAC): Modèle SAC entraîné.
        data_segment (pd.DataFrame): Segment de données.
        start_idx (int): Indice de début.
        end_idx (int): Indice de fin.
        env (TradingEnv): Environnement de trading.
        detector (MarketRegimeDetector): Détecteur de régime.
        neural_pipeline (NeuralPipeline): Pipeline neuronal.
        sequence_length (int): Longueur de la séquence.
        policy_type (str): Type de politique.
        market (str): Marché (ex. : ES, MNQ).

    Returns:
        Tuple[float, float, List[float], List[Dict], List[int], List[Dict]]: Métriques et logs.
    """
    try:
        env.data = data_segment
        env.current_step = 0
        obs, _ = env.reset()
        rewards = []
        trade_log = []
        rejected_trades = [0]
        rejected_stats = []

        for i in range(len(data_segment)):
            validate_features(
                env.data,
                env.current_step,
                shap_features=(policy_type == "transformer"),
                market=market,
            )
            row = env.data.iloc[env.current_step]
            float(row.get(f"atr_14_{market.lower()}", 1.0))
            regime, details = detector.detect(env.data, env.current_step)
            neural_regime = details.get("neural_regime", None)

            if neural_regime is None:
                window_start = max(0, env.current_step - sequence_length + 1)
                window_end = env.current_step + 1
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
                            "call_wall",
                            "put_wall",
                            "zero_gamma",
                            "dealer_position_bias",
                            "key_strikes_1",
                            "max_pain_strike",
                            "net_gamma",
                            "dealer_zones_count",
                            "vol_trigger",
                            "ref_px",
                            "data_release",
                            "oi_sweep_count",
                            "iv_acceleration",
                            "theta_exposure",
                            "iv_atm",
                            "option_skew",
                            "gex_slope",
                            "vomma_exposure",
                            "speed_exposure",
                            "ultima_exposure",
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
                            "bid_size_level_4",
                            "ask_size_level_4",
                            "bid_size_level_5",
                            "ask_size_level_5",
                        ]
                    ]
                    .iloc[window_start:window_end]
                    .fillna(0)
                )
                neural_result = neural_pipeline.run(
                    raw_data, options_data, orderflow_data
                )
                env.data.loc[
                    env.current_step, f"predicted_volatility_{market.lower()}"
                ] = neural_result["volatility"][-1]
                env.data.loc[env.current_step, "neural_regime"] = neural_result[
                    "regime"
                ][-1]
                feature_cols = [f"neural_feature_{i}" for i in range(8)] + [
                    "cnn_pressure"
                ]
                for j, col in enumerate(feature_cols):
                    env.data.loc[env.current_step, col] = neural_result["features"][
                        -1, j
                    ]
                neural_regime = int(neural_result["regime"][-1])
                success_msg = f"Régime neuronal calculé au step {env.current_step} pour {market}: {neural_regime}"
                logger.debug(success_msg)
                miya_speak(success_msg, tag="BACKTEST_SAC", voice_profile="calm")
                AlertManager().send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)

            regime_match = regime == env.mode or (
                neural_regime == {"trend": 0, "range": 1, "defensive": 2}.get(env.mode)
            )
            if regime_match:
                if policy_type == "transformer":
                    obs_seq = (
                        env.data[env.features]
                        .iloc[
                            max(
                                0, env.current_step - sequence_length + 1
                            ) : env.current_step
                            + 1
                        ]
                        .values
                    )
                    if len(obs_seq) < sequence_length:
                        padding = np.zeros(
                            (sequence_length - len(obs_seq), len(env.features))
                        )
                        obs_seq = np.vstack([padding, obs_seq])
                    obs = obs_seq.astype(np.float32)
                else:
                    obs = (
                        env.data[env.features]
                        .iloc[env.current_step]
                        .values.astype(np.float32)
                    )

                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                rewards.append(float(reward))
                trade_log.append(
                    {
                        "step": start_idx + i,
                        "action": float(action[0]),
                        "reward": float(reward),
                        "regime": regime,
                        "neural_regime": neural_regime,
                        "info": info,
                    }
                )
            else:
                rejected_trades[0] += 1
                rejected_stats.append(
                    {
                        "step": start_idx + i,
                        "reason": f"Regime mismatch (regime={regime}, neural={neural_regime})",
                    }
                )
                obs, reward, done, _, info = env.step(np.array([0.0]))
                rewards.append(float(reward))

            if done:
                break

        profit_factor = (
            sum(r for r in rewards if r > 0) / abs(sum(r for r in rewards if r < 0))
            if any(r < 0 for r in rewards)
            else float("inf")
        )
        sharpe = np.mean(rewards) / np.std(rewards) if np.std(rewards) > 0 else 0.0
        save_snapshot(
            "eval_sac_segment",
            {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "profit_factor": profit_factor,
                "sharpe": sharpe,
                "num_trades": len(trade_log),
                "rejected_trades": rejected_trades[0],
                "market": market,
            },
            market=market,
        )
        return (
            profit_factor,
            sharpe,
            rewards,
            trade_log,
            rejected_trades,
            rejected_stats,
        )

    except Exception as e:
        error_msg = f"Erreur dans eval_sac_segment {start_idx}-{end_idx} pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        miya_alerts(error_msg, tag="BACKTEST_SAC", priority=5)
        AlertManager().send_alert(error_msg, priority=5)
        send_telegram_alert(error_msg)
        return 0.0, 0.0, [], [], [0], [{"step": start_idx, "reason": str(e)}]


def monte_carlo_backtest(
    model: SAC,
    data: pd.DataFrame,
    env: TradingEnv,
    detector: MarketRegimeDetector,
    neural_pipeline: NeuralPipeline,
    n_simulations: int = 100,
    sequence_length: int = 50,
    policy_type: str = "mlp",
    market: str = "ES",
) -> Dict[str, float]:
    """
    Effectue un backtest Monte Carlo avec des simulations bruitées.

    Args:
        model (SAC): Modèle SAC.
        data (pd.DataFrame): Données de base.
        env (TradingEnv): Environnement de trading.
        detector (MarketRegimeDetector): Détecteur de régime.
        neural_pipeline (NeuralPipeline): Pipeline neuronal.
        n_simulations (int): Nombre de simulations.
        sequence_length (int): Longueur de la séquence.
        policy_type (str): Type de politique.
        market (str): Marché (ex. : ES, MNQ).

    Returns:
        Dict[str, float]: Statistiques Monte Carlo.
    """
    try:
        cache_key = f"{market}_{hash(str(data))}_{policy_type}_{n_simulations}"
        if cache_key in backtest_cache:
            results = backtest_cache[cache_key]
            backtest_cache.move_to_end(cache_key)
            return results
        while len(backtest_cache) > MAX_CACHE_SIZE:
            backtest_cache.popitem(last=False)

        results = []
        for sim in range(n_simulations):
            noisy_data = data.copy()
            for col in ["close", f"atr_14_{market.lower()}", f"gex_{market.lower()}"]:
                if col in noisy_data.columns:
                    noisy_data[col] += np.random.normal(
                        0, noisy_data[col].std() * 0.05, len(data)
                    )
            validate_data(noisy_data, market=market)
            validate_features(noisy_data, 0, market=market)
            profit_factor, sharpe, _, _, _, _ = eval_sac_segment(
                model,
                noisy_data,
                0,
                len(noisy_data),
                env,
                detector,
                neural_pipeline,
                sequence_length,
                policy_type,
                market=market,
            )
            results.append({"profit_factor": profit_factor, "sharpe": sharpe})

        results_df = pd.DataFrame(results)
        mc_stats = {
            "profit_factor_mean": float(results_df["profit_factor"].mean()),
            "profit_factor_p5": float(results_df["profit_factor"].quantile(0.05)),
            "profit_factor_p95": float(results_df["profit_factor"].quantile(0.95)),
            "sharpe_mean": float(results_df["sharpe"].mean()),
            "sharpe_p5": float(results_df["sharpe"].quantile(0.05)),
            "sharpe_p95": float(results_df["sharpe"].quantile(0.95)),
        }
        backtest_cache[cache_key] = mc_stats
        success_msg = f"Monte Carlo terminé pour {n_simulations} simulations pour {market}: {mc_stats}"
        logger.info(success_msg)
        miya_speak(success_msg, tag="BACKTEST_SAC", voice_profile="calm")
        AlertManager().send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
        save_snapshot(
            "monte_carlo_backtest",
            {"n_simulations": n_simulations, "stats": mc_stats, "market": market},
            market=market,
        )
        checkpoint(results_df, data_type="monte_carlo_results", market=market)
        cloud_backup(results_df, data_type="monte_carlo_results", market=market)
        return mc_stats
    except Exception as e:
        error_msg = f"Erreur dans monte_carlo_backtest pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        miya_alerts(error_msg, tag="BACKTEST_SAC", priority=5)
        AlertManager().send_alert(error_msg, priority=5)
        send_telegram_alert(error_msg)
        return {
            "profit_factor_mean": 0.0,
            "profit_factor_p5": 0.0,
            "profit_factor_p95": 0.0,
            "sharpe_mean": 0.0,
            "sharpe_p5": 0.0,
            "sharpe_p95": 0.0,
        }


def backtest_sac(
    config_path: str = str(BASE_DIR / "config/es_config.yaml"),
    mode: str = "defensive",
    n_jobs: int = -1,
    policy_type: str = "mlp",
    market: str = "ES",
) -> Tuple[float, float, float, Dict[str, float]]:
    """
    Backtest un modèle SAC avec parallélisation et Monte Carlo.

    Args:
        config_path (str): Chemin vers la configuration.
        mode (str): Mode du modèle ("defensive", "trend", "range").
        n_jobs (int): Nombre de jobs pour parallélisation.
        policy_type (str): Type de politique ("mlp", "transformer").
        market (str): Marché (ex. : ES, MNQ).

    Returns:
        Tuple[float, float, float, Dict[str, float]]: Profit factor, Sharpe, max drawdown, stats Monte Carlo.
    """
    try:
        config = load_config(config_path, market=market)
        data_path = config.get(
            "data_path", str(BASE_DIR / "data/features/features_latest_filtered.csv")
        )
        model_dir = config.get("model_dir", str(BASE_DIR / "model/sac_models" / market))
        log_dir = config.get("log_dir", str(BASE_DIR / "data/logs/backtest"))
        sequence_length = config.get("sequence_length", 50)
        n_simulations = config.get("n_simulations", 100)
        min_rows = config.get("min_rows", 100)
        performance_thresholds = config.get(
            "performance_thresholds", DEFAULT_PERFORMANCE_THRESHOLDS
        )

        # Chargement des données
        data = pd.read_csv(data_path)
        if len(data) < min_rows:
            error_msg = f"Données insuffisantes pour {market}: {len(data)} < {min_rows}"
            logger.error(error_msg)
            miya_alerts(error_msg, tag="BACKTEST_SAC", priority=5)
            raise ValueError(error_msg)
        validate_data(data, market=market)
        validate_features(data, 0, market=market)

        # Initialisation de l’environnement
        env = TradingEnv(config_path, market=market)
        env.mode = mode
        env.policy_type = policy_type
        if policy_type == "transformer":
            env.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(sequence_length, len(env.features)),
                dtype=np.float32,
            )
        else:
            env.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(len(env.features),), dtype=np.float32
            )

        # Chargement du modèle
        model_files = [
            f
            for f in os.listdir(model_dir)
            if f.startswith(f"sac_{mode}_") and f.endswith(".zip")
        ]
        if not model_files:
            error_msg = (
                f"Aucun modèle trouvé pour mode {mode} dans {model_dir} pour {market}"
            )
            logger.error(error_msg)
            miya_alerts(error_msg, tag="BACKTEST_SAC", priority=5)
            raise FileNotFoundError(error_msg)
        model_path = os.path.join(
            model_dir,
            max(
                model_files, key=lambda x: os.path.getctime(os.path.join(model_dir, x))
            ),
        )
        model = SAC.load(model_path, env=env)
        model.policy_alias = policy_type

        # Initialisation du détecteur et pipeline neuronal
        detector = MarketRegimeDetector(config_path, market=market)
        neural_pipeline = NeuralPipeline(
            window_size=sequence_length,
            base_features=350,
            config_path=str(BASE_DIR / "config/model_params.yaml"),
        )
        neural_pipeline.load_models()

        # Parallélisation des segments
        n_segments = n_jobs if n_jobs > 0 else os.cpu_count()
        segment_size = max(len(data) // n_segments, sequence_length)
        segments = [
            (data.iloc[i : i + segment_size], i, i + segment_size)
            for i in range(0, len(data), segment_size)
        ]

        results = Parallel(n_jobs=n_jobs)(
            delayed(eval_sac_segment)(
                model,
                segment,
                start_idx,
                end_idx,
                env,
                detector,
                neural_pipeline,
                sequence_length,
                policy_type,
                market=market,
            )
            for segment, start_idx, end_idx in segments
        )

        (
            profit_factors,
            sharpes,
            all_rewards,
            all_trade_logs,
            all_rejected_trades,
            all_rejected_stats,
        ) = zip(*results)
        total_rewards = [r for segment_rewards in all_rewards for r in segment_rewards]
        total_trade_log = [t for segment_log in all_trade_logs for t in segment_log]
        total_rejected_trades = sum(r[0] for r in all_rejected_trades)
        total_rejected_stats = [
            s for segment_stats in all_rejected_stats for s in segment_stats
        ]

        profit_factor = float(np.mean([pf for pf in profit_factors if np.isfinite(pf)]))
        sharpe = float(np.mean([s for s in sharpes if np.isfinite(s)]))
        max_drawdown = float(
            np.min(
                np.cumsum(total_rewards)
                - np.maximum.accumulate(np.cumsum(total_rewards))
            )
            / 10000
        )

        # Validation des seuils
        for metric, threshold in performance_thresholds.items():
            value = locals().get(metric.replace("min_", "").replace("max_", ""), 0)
            if "min_" in metric and value < threshold:
                warning_msg = f"Seuil non atteint pour {market}: {metric} ({value:.2f}) < {threshold}"
                logger.warning(warning_msg)
                miya_speak(warning_msg, tag="BACKTEST_SAC", level="warning")
                AlertManager().send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
            elif "max_" in metric and value > threshold:
                warning_msg = f"Seuil non atteint pour {market}: {metric} ({value:.2f}) > {threshold}"
                logger.warning(warning_msg)
                miya_speak(warning_msg, tag="BACKTEST_SAC", level="warning")
                AlertManager().send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)

        # Monte Carlo
        mc_stats = monte_carlo_backtest(
            model,
            data,
            env,
            detector,
            neural_pipeline,
            n_simulations,
            sequence_length,
            policy_type,
            market=market,
        )

        # Export des logs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"backtest_{mode}_{timestamp}.csv"
        rejection_log_file = f"backtest_{mode}_rejections_{timestamp}.csv"
        export_logs(total_trade_log, log_dir, log_file, market=market)
        export_logs(total_rejected_stats, log_dir, rejection_log_file, market=market)

        success_msg = f"Backtest terminé pour {market}: mode={mode}, PF: {profit_factor:.2f}, Sharpe: {sharpe:.2f}, Max DD: {max_drawdown:.2f}"
        logger.info(success_msg)
        miya_speak(success_msg, tag="BACKTEST_SAC", voice_profile="calm")
        AlertManager().send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
        save_snapshot(
            "backtest_sac",
            {
                "mode": mode,
                "profit_factor": profit_factor,
                "sharpe": sharpe,
                "max_drawdown": max_drawdown,
                "mc_stats": mc_stats,
                "num_trades": len(total_trade_log),
                "rejected_trades": total_rejected_trades,
                "market": market,
            },
            market=market,
        )
        checkpoint(pd.DataFrame(total_trade_log), data_type="trade_log", market=market)
        cloud_backup(
            pd.DataFrame(total_trade_log), data_type="trade_log", market=market
        )
        return profit_factor, sharpe, max_drawdown, mc_stats

    except Exception as e:
        error_msg = f"Erreur dans backtest_sac, mode={mode} pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        miya_alerts(error_msg, tag="BACKTEST_SAC", priority=5)
        AlertManager().send_alert(error_msg, priority=5)
        send_telegram_alert(error_msg)
        return (
            0.0,
            0.0,
            0.0,
            {
                "profit_factor_mean": 0.0,
                "profit_factor_p5": 0.0,
                "profit_factor_p95": 0.0,
                "sharpe_mean": 0.0,
                "sharpe_p5": 0.0,
                "sharpe_p95": 0.0,
            },
        )


def incremental_backtest_sac(
    row: pd.Series,
    buffer: pd.DataFrame,
    state: Dict[str, Any],
    model: SAC,
    env: TradingEnv,
    detector: MarketRegimeDetector,
    neural_pipeline: NeuralPipeline,
    config_path: str = str(BASE_DIR / "config/es_config.yaml"),
    market: str = "ES",
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Exécute un backtest incrémental pour une seule ligne en temps réel.

    Args:
        row (pd.Series): Ligne contenant timestamp, close, et 350 features.
        buffer (pd.DataFrame): Buffer des données précédentes.
        state (Dict[str, Any]): État du backtest.
        model (SAC): Modèle SAC.
        env (TradingEnv): Environnement de trading.
        detector (MarketRegimeDetector): Détecteur de régime.
        neural_pipeline (NeuralPipeline): Pipeline neuronal.
        config_path (str): Chemin vers la configuration.
        market (str): Marché (ex. : ES, MNQ).

    Returns:
        Tuple[Dict[str, float], Dict[str, Any]]: Métriques et état mis à jour.
    """
    try:
        config = load_config(config_path, market=market)
        sequence_length = config.get("sequence_length", 50)

        # Mise à jour du buffer
        buffer = pd.concat([buffer, row.to_frame().T], ignore_index=True)
        if len(buffer) < sequence_length:
            return {"profit_factor": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}, state

        # Validation
        validate_features(
            buffer,
            len(buffer) - 1,
            shap_features=(env.policy_type == "transformer"),
            market=market,
        )
        row = buffer.iloc[-1]
        regime, details = detector.detect(buffer, len(buffer) - 1)
        neural_regime = details.get("neural_regime", None)

        if neural_regime is None:
            raw_data = (
                buffer[
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
                .tail(sequence_length)
                .fillna(0)
            )
            options_data = (
                buffer[
                    [
                        "timestamp",
                        f"gex_{market.lower()}",
                        "oi_peak_call_near",
                        "gamma_wall_call",
                        "gamma_wall_put",
                        "call_wall",
                        "put_wall",
                        "zero_gamma",
                        "dealer_position_bias",
                        "key_strikes_1",
                        "max_pain_strike",
                        "net_gamma",
                        "dealer_zones_count",
                        "vol_trigger",
                        "ref_px",
                        "data_release",
                        "oi_sweep_count",
                        "iv_acceleration",
                        "theta_exposure",
                        "iv_atm",
                        "option_skew",
                        "gex_slope",
                        "vomma_exposure",
                        "speed_exposure",
                        "ultima_exposure",
                    ]
                ]
                .tail(sequence_length)
                .fillna(0)
            )
            orderflow_data = (
                buffer[
                    [
                        "timestamp",
                        "bid_size_level_1",
                        "ask_size_level_1",
                        "bid_size_level_4",
                        "ask_size_level_4",
                        "bid_size_level_5",
                        "ask_size_level_5",
                    ]
                ]
                .tail(sequence_length)
                .fillna(0)
            )
            neural_result = neural_pipeline.run(raw_data, options_data, orderflow_data)
            buffer.loc[len(buffer) - 1, f"predicted_volatility_{market.lower()}"] = (
                neural_result["volatility"][-1]
            )
            buffer.loc[len(buffer) - 1, "neural_regime"] = neural_result["regime"][-1]
            feature_cols = [f"neural_feature_{i}" for i in range(8)] + ["cnn_pressure"]
            for j, col in enumerate(feature_cols):
                buffer.loc[len(buffer) - 1, col] = neural_result["features"][-1, j]
            neural_regime = int(neural_result["regime"][-1])
            success_msg = f"Régime neuronal calculé au step {len(buffer) - 1} pour {market}: {neural_regime}"
            logger.debug(success_msg)
            miya_speak(success_msg, tag="BACKTEST_SAC", voice_profile="calm")
            AlertManager().send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)

        # Initialisation de l’état
        if not state:
            state = {
                "rewards": [],
                "trade_log": [],
                "rejected_trades": 0,
                "rejected_stats": [],
                "balance": env.balance,
            }

        regime_match = regime == env.mode or (
            neural_regime == {"trend": 0, "range": 1, "defensive": 2}.get(env.mode)
        )
        env.data = buffer.tail(sequence_length)
        env.current_step = sequence_length - 1

        if regime_match:
            if env.policy_type == "transformer":
                obs = (
                    buffer[env.features].tail(sequence_length).values.astype(np.float32)
                )
            else:
                obs = buffer[env.features].iloc[-1].values.astype(np.float32)

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            state["rewards"].append(float(reward))
            state["trade_log"].append(
                {
                    "step": len(buffer) - 1,
                    "action": float(action[0]),
                    "reward": float(reward),
                    "regime": regime,
                    "neural_regime": neural_regime,
                    "info": info,
                }
            )
        else:
            state["rejected_trades"] += 1
            state["rejected_stats"].append(
                {"step": len(buffer) - 1, "reason": "Regime mismatch"}
            )
            obs, reward, done, _, info = env.step(np.array([0.0]))
            state["rewards"].append(float(reward))

        state["balance"] = env.balance
        profit_factor = (
            sum(r for r in state["rewards"] if r > 0)
            / abs(sum(r for r in state["rewards"] if r < 0))
            if any(r < 0 for r in state["rewards"])
            else float("inf")
        )
        sharpe = (
            np.mean(state["rewards"]) / np.std(state["rewards"])
            if np.std(state["rewards"]) > 0
            else 0.0
        )
        max_drawdown = (
            np.min(
                np.cumsum(state["rewards"])
                - np.maximum.accumulate(np.cumsum(state["rewards"]))
            )
            / 10000
            if state["rewards"]
            else 0.0
        )

        metrics = {
            "profit_factor": profit_factor,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
        }
        save_snapshot(
            "incremental_backtest_sac",
            {
                "step": len(buffer) - 1,
                "metrics": metrics,
                "num_trades": len(state["trade_log"]),
                "rejected_trades": state["rejected_trades"],
                "market": market,
            },
            market=market,
        )
        checkpoint(
            pd.DataFrame(state["trade_log"]),
            data_type="incremental_trade_log",
            market=market,
        )
        cloud_backup(
            pd.DataFrame(state["trade_log"]),
            data_type="incremental_trade_log",
            market=market,
        )
        return metrics, state

    except Exception as e:
        error_msg = f"Erreur dans incremental_backtest_sac pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        miya_alerts(error_msg, tag="BACKTEST_SAC", priority=5)
        AlertManager().send_alert(error_msg, priority=5)
        send_telegram_alert(error_msg)
        return {"profit_factor": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}, state


if __name__ == "__main__":
    try:
        # Test avec données simulées
        feature_sets_path = BASE_DIR / "config/feature_sets.yaml"
        features_config = get_config(feature_sets_path)
        feature_cols = features_config.get("training", {}).get("features", [])[:350]
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-05-13 09:00", periods=1000, freq="1min"
                ),
                "open": np.random.normal(5100, 10, 1000),
                "high": np.random.normal(5105, 10, 1000),
                "low": np.random.normal(5095, 10, 1000),
                "close": np.random.normal(5100, 10, 1000),
                "volume": np.random.randint(100, 1000, 1000),
                "bid_size_level_1": np.random.randint(100, 1000, 1000),
                "ask_size_level_1": np.random.randint(100, 1000, 1000),
                "trade_frequency_1s": np.random.randint(0, 50, 1000),
                "atr_14_es": np.random.uniform(0.5, 2.0, 1000),
                "adx_14": np.random.uniform(10, 50, 1000),
                "gex_es": np.random.uniform(-1000, 1000, 1000),
                "oi_peak_call_near": np.random.randint(100, 1000, 1000),
                "gamma_wall_call": np.random.uniform(0, 1000, 1000),
                "gamma_wall_put": np.random.uniform(0, 1000, 1000),
                "news_impact_score": np.random.uniform(0, 1, 1000),
                "trade_success_prob": np.random.uniform(0, 1, 1000),
                **{
                    col: np.random.normal(0, 1, 1000)
                    for col in feature_cols
                    if col
                    not in [
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "bid_size_level_1",
                        "ask_size_level_1",
                        "trade_frequency_1s",
                        "atr_14_es",
                        "adx_14",
                        "gex_es",
                        "oi_peak_call_near",
                        "gamma_wall_call",
                        "gamma_wall_put",
                        "news_impact_score",
                        "trade_success_prob",
                    ]
                },
            }
        )
        data.to_csv(
            BASE_DIR / "data/features/features_latest_filtered.csv", index=False
        )

        for market in ["ES", "MNQ"]:
            for mode in ["trend", "defensive"]:
                for policy_type in ["mlp"]:
                    profit_factor, sharpe, max_drawdown, mc_stats = backtest_sac(
                        config_path=str(BASE_DIR / "config/es_config.yaml"),
                        mode=mode,
                        n_jobs=2,
                        policy_type=policy_type,
                        market=market,
                    )
                    print(
                        f"Market: {market}, Mode: {mode}, Policy: {policy_type}, Profit Factor: {profit_factor:.2f}, Sharpe: {sharpe:.2f}, Max Drawdown: {max_drawdown:.2f}"
                    )
                    print(f"Monte Carlo Stats: {mc_stats}")
        success_msg = "Test backtest_sac terminé pour ES et MNQ"
        logger.info(success_msg)
        miya_speak(success_msg, tag="BACKTEST_SAC", voice_profile="calm")
        AlertManager().send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
    except Exception as e:
        error_msg = f"Erreur test principal: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        miya_alerts(error_msg, tag="BACKTEST_SAC", priority=5)
        AlertManager().send_alert(error_msg, priority=5)
        send_telegram_alert(error_msg)
        raise
