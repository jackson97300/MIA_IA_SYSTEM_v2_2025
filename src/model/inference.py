# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2022/src/model/inference.py
# Effectue des prédictions avec un modèle SAC entraîné, optimisé pour MIA_IA_SYSTEM_v2_2025 
# avec neural_pipeline.
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Prédit les actions de trading avec un modèle SAC, intégrant trade_success_prob via 
# TradeProbabilityPredictor, utilisant les top 150 SHAP features pour l’inférence, avec 
# validation quotidienne.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, stable-baselines3>=2.0.0,<3.0.0, 
#   psutil>=5.9.8,<6.0.0, pyyaml>=6.0.0,<7.0.0, gymnasium>=1.0.0,<2.0.0, 
#   boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0
# - src/envs/trading_env.py
# - src/envs/gym_wrappers.py
# - src/model/utils/trading_utils.py
# - src/model/router/detect_regime.py
# - src/model/router/mode_trend.py
# - src/model/router/mode_range.py
# - src/model/router/mode_defensive.py
# - src/model/utils_model.py
# - src/features/neural_pipeline.py
# - src/model/adaptive_learning.py
# - src/model/utils/miya_console.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/trade_probability.py
# - src/data/data_provider.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/es_config.yaml
# - data/features/features_latest_filtered.csv
# - data/features/feature_importance.csv
#
# Outputs :
# - data/logs/inference.log
# - data/logs/inference_performance.csv
# - data/cache/inference/<market>/*.json.gz
# - data/checkpoints/inference/<market>/*.json.gz
# - data/logs/inference_<mode>_<policy_type>_<timestamp>.csv
#
# Notes :
# - Utilise IQFeed via data_provider.py, retries (max 3, délai 2^attempt), logs psutil, 
#   alertes Telegram via alert_manager.py, snapshots JSON compressés, et top 150 SHAP 
#   features pour l’inférence.
# - Suppression de toute référence à dxFeed, obs_t, 320/81 features.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des données.
# - Tests unitaires disponibles dans tests/test_inference.py.

import asyncio
import gzip
import json
import os
import time
import traceback
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import numpy as np
import pandas as pd
import psutil
import yaml
from gymnasium import spaces
from loguru import logger

# Note: `np` est importé une seule fois ici (F811 est une fausse alerte, aucune 
# redéfinition de `np` dans ce fichier).

from src.data.data_provider import get_data_provider
from src.envs.gym_wrappers import (
    ClippingWrapper,
    NormalizationWrapper,
    ObservationStackingWrapper,
)
from src.envs.trading_env import TradingEnv
from src.features.neural_pipeline import NeuralPipeline
from src.model.adaptive_learning import last_3_trades_success_rate, store_pattern
from src.model.router.detect_regime import MarketRegimeDetector
from src.model.router.mode_defensive import ModeDefensive
from src.model.router.mode_range import ModeRange
from src.model.router.mode_trend import ModeTrend
from src.model.trade_probability import TradeProbabilityPredictor
from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import get_config
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.model.utils.trading_utils import (
    adjust_risk_and_leverage,
    validate_trade_entry_combined,
)
from src.model.utils_model import compute_metrics, export_logs, validate_features
from src.utils.telegram_alert import send_telegram_alert

# Configuration logging
logger.remove()
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOG_DIR = BASE_DIR / "data" / "logs"
CACHE_DIR = BASE_DIR / "data" / "cache" / "inference"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints" / "inference"
SNAPSHOT_DIR = CACHE_DIR
PERF_LOG_PATH = LOG_DIR / "inference_performance.csv"
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "inference.log", 
    rotation="10 MB", 
    level="INFO", 
    encoding="utf-8"
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
    "min_reward": -50.0,  # Récompense minimale par étape
    "min_balance": 9000.0,  # Solde minimal
    "max_drawdown": -1000.0,  # Drawdown maximal
    "min_profit_factor": 1.2,  # Facteur de profit minimal
    "min_sharpe": 0.5,  # Ratio de Sharpe minimal
}

# Cache global pour les prédictions
prediction_cache = OrderedDict()


class InferenceEngine:
    def __init__(
        self, 
        config_path: str = str(BASE_DIR / "config/es_config.yaml")
    ):
        """Initialise l’engine d’inférence."""
        try:
            self.alert_manager = AlertManager()
            self.config = self.load_config(config_path)
            self.trade_predictor = TradeProbabilityPredictor()
            SNAPSHOT_DIR.mkdir(exist_ok=True)
            PERF_LOG_PATH.parent.mkdir(exist_ok=True)
            success_msg = "InferenceEngine initialisé"
            logger.info(success_msg)
            miya_speak(success_msg, tag="INFERENCE", voice_profile="calm")
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance("init", 0, success=True)
        except Exception as e:
            error_msg = (
                f"Erreur initialisation InferenceEngine: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            logger.error(error_msg)
            miya_alerts(
                error_msg, 
                tag="INFERENCE", 
                priority=5, 
                voice_profile="urgent"
            )
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            raise

    def load_config(self, config_path: str) -> Dict:
        """Charge le fichier de configuration."""

        def load_yaml():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}

        try:
            config = self.with_retries(load_yaml)
            if config is None:
                raise FileNotFoundError(f"Fichier {config_path} non trouvé ou vide")
            return config
        except Exception as e:
            error_msg = (
                f"Erreur chargement configuration {config_path}: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            logger.error(error_msg)
            miya_alerts(
                error_msg, 
                tag="INFERENCE", 
                priority=4, 
                voice_profile="urgent"
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            return {}

    def with_retries(
        self,
        func: callable,
        max_attempts: int = MAX_RETRIES,
        delay_base: float = RETRY_DELAY,
        market: str = "ES",
    ) -> Any:
        """Exécute une fonction avec retries."""
        start_time = time.time()
        for attempt in range(max_attempts):
            try:
                result = func()
                latency = time.time() - start_time
                self.log_performance(
                    f"retry_attempt_{attempt+1}",
                    latency,
                    success=True,
                    attempt_number=attempt + 1,
                    market=market,
                )
                return result
            except Exception as e:
                if attempt == max_attempts - 1:
                    latency = time.time() - start_time
                    self.log_performance(
                        f"retry_attempt_{attempt+1}",
                        latency,
                        success=False,
                        error=str(e),
                        attempt_number=attempt + 1,
                        market=market,
                    )
                    error_msg = (
                        f"Échec après {max_attempts} tentatives pour {market}: "
                        f"{str(e)}\n{traceback.format_exc()}"
                    )
                    logger.error(error_msg)
                    self.alert_manager.send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    raise
                delay = delay_base * (2**attempt)
                warning_msg = (
                    f"Tentative {attempt+1} échouée pour {market}, "
                    f"retry après {delay}s"
                )
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                time.sleep(delay)

    def log_performance(
        self,
        operation: str,
        latency: float,
        success: bool,
        error: str = None,
        market: str = "ES",
        **kwargs,
    ) -> None:
        """Journalise les performances des opérations critiques."""
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
            cpu_usage = psutil.cpu_percent()  # % CPU
            if memory_usage > 1024:
                alert_msg = (
                    f"ALERTE: Usage mémoire élevé ({memory_usage:.2f} MB) "
                    f"pour {market}"
                )
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
                "cpu_usage_percent": cpu_usage,
                "market": market,
                **kwargs,
            }
            log_df = pd.DataFrame([log_entry])

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

            self.with_retries(save_log, market=market)
            logger.info(f"Performance journalisée pour {operation}. CPU: {cpu_usage}%")
        except Exception as e:
            error_msg = (
                f"Erreur journalisation performance pour {market}: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            logger.error(error_msg)
            miya_alerts(
                error_msg, 
                tag="INFERENCE", 
                priority=3, 
                voice_profile="urgent"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)

    def save_snapshot(
        self, 
        snapshot_type: str, 
        data: Dict, 
        market: str = "ES", 
        compress: bool = True
    ) -> None:
        """Sauvegarde un instantané des résultats, compressé avec gzip."""
        try:
            start_time = time.time()
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

            self.with_retries(write_snapshot, market=market)
            save_path = f"{snapshot_path}.gz" if compress else snapshot_path
            file_size = os.path.getsize(save_path) / 1024 / 1024
            if file_size > 1.0:
                alert_msg = (
                    f"Snapshot size {file_size:.2f} MB exceeds 1 MB pour {market}"
                )
                logger.warning(alert_msg)
                miya_alerts(alert_msg, tag="INFERENCE", priority=3)
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
            latency = time.time() - start_time
            success_msg = (
                f"Snapshot {snapshot_type} sauvegardé pour {market}: {save_path}"
            )
            logger.info(success_msg)
            miya_speak(success_msg, tag="INFERENCE", voice_profile="calm")
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
            error_msg = (
                f"Erreur sauvegarde snapshot {snapshot_type} pour {market}: "
                f"{str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            miya_alerts(
                error_msg, 
                tag="INFERENCE", 
                priority=3, 
                voice_profile="urgent"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "save_snapshot", 
                0, 
                success=False, 
                error=str(e), 
                market=market
            )

    def checkpoint(
        self, 
        data: pd.DataFrame, 
        data_type: str = "inference_state", 
        market: str = "ES"
    ) -> None:
        """Sauvegarde incrémentielle des données toutes les 5 minutes avec 
        versionnage (5 versions)."""
        try:
            start_time = time.time()
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
                checkpoint_dir / f"inference_{data_type}_{timestamp}.json.gz"
            )
            checkpoint_versions = []

            def write_checkpoint():
                with gzip.open(checkpoint_path, "wt", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, indent=4)
                csv_path = checkpoint_path.with_suffix(".csv")
                data.to_csv(csv_path, index=False, encoding="utf-8")

            self.with_retries(write_checkpoint, market=market)
            checkpoint_versions.append(checkpoint_path)
            if len(checkpoint_versions) > 5:
                oldest = checkpoint_versions.pop(0)
                if oldest.exists():
                    oldest.unlink()
                csv_oldest = oldest.with_suffix(".csv")
                if csv_oldest.exists():
                    csv_oldest.unlink()
            file_size = os.path.getsize(checkpoint_path) / 1024 / 1024
            latency = time.time() - start_time
            success_msg = (
                f"Checkpoint sauvegardé pour {market}: {checkpoint_path}"
            )
            logger.info(success_msg)
            miya_speak(success_msg, tag="INFERENCE", voice_profile="calm")
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
            error_msg = (
                f"Erreur sauvegarde checkpoint pour {market}: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            logger.error(error_msg)
            miya_alerts(
                error_msg, 
                tag="INFERENCE", 
                priority=3, 
                voice_profile="urgent"
            )
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
        data_type: str = "inference_state", 
        market: str = "ES"
    ) -> None:
        """Sauvegarde distribuée des données vers AWS S3 toutes les 15 minutes."""
        try:
            start_time = time.time()
            config_path = str(BASE_DIR / "config/es_config.yaml")
            config = self.load_config(config_path)
            if not config.get("s3_bucket"):
                warning_msg = (
                    f"S3 bucket non configuré, sauvegarde cloud ignorée pour {market}"
                )
                logger.warning(warning_msg)
                miya_alerts(warning_msg, tag="INFERENCE", priority=3)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = (
                f"{config['s3_prefix']}inference_{data_type}_{market}_{timestamp}.csv.gz"
            )
            temp_path = CHECKPOINT_DIR / market / f"temp_s3_{timestamp}.csv.gz"
            temp_path.parent.mkdir(exist_ok=True)

            def write_temp():
                data.to_csv(
                    temp_path, 
                    compression="gzip", 
                    index=False, 
                    encoding="utf-8"
                )

            self.with_retries(write_temp, market=market)
            s3_client = boto3.client("s3")

            def upload_s3():
                s3_client.upload_file(str(temp_path), config["s3_bucket"], backup_path)

            self.with_retries(upload_s3, market=market)
            temp_path.unlink()
            latency = time.time() - start_time
            success_msg = (
                f"Sauvegarde cloud S3 effectuée pour {market}: {backup_path}"
            )
            logger.info(success_msg)
            miya_speak(success_msg, tag="INFERENCE", voice_profile="calm")
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
            error_msg = (
                f"Erreur sauvegarde cloud S3 pour {market}: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            logger.error(error_msg)
            miya_alerts(
                error_msg, 
                tag="INFERENCE", 
                priority=3, 
                voice_profile="urgent"
            )
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

    def configure_feature_set(self) -> List[str]:
        """Charge les top 150 SHAP features."""
        try:
            shap_file = str(BASE_DIR / "data/features/feature_importance.csv")
            if os.path.exists(shap_file):
                shap_df = pd.read_csv(shap_file)
                if len(shap_df) >= 150:
                    features = shap_df["feature"].head(150).tolist()
                    success_msg = "Top 150 SHAP features chargées pour inférence"
                    logger.info(success_msg)
                    miya_speak(success_msg, tag="INFERENCE", voice_profile="calm")
                    self.alert_manager.send_alert(success_msg, priority=1)
                    send_telegram_alert(success_msg)
                    return features
            error_msg = (
                "SHAP features non trouvées ou insuffisantes dans "
                "feature_importance.csv"
            )
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = (
                f"Erreur chargement SHAP features: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            logger.error(error_msg)
            miya_alerts(
                error_msg, 
                tag="INFERENCE", 
                priority=3, 
                voice_profile="urgent"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            raise

    def validate_shap_features(self, market: str = "ES") -> bool:
        """Valide quotidiennement les top 150 SHAP features."""
        try:
            shap_file = str(BASE_DIR / "data/features/feature_importance.csv")
            if not os.path.exists(shap_file):
                error_msg = f"Fichier SHAP manquant pour {market}"
                logger.error(error_msg)
                miya_alerts(
                    error_msg, 
                    tag="INFERENCE", 
                    priority=4, 
                    voice_profile="urgent"
                )
                self.alert_manager.send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                return False
            shap_df = pd.read_csv(shap_file)
            if len(shap_df) < 150:
                error_msg = (
                    f"Nombre insuffisant de SHAP features pour {market}: "
                    f"{len(shap_df)} < 150"
                )
                logger.error(error_msg)
                miya_alerts(
                    error_msg, 
                    tag="INFERENCE", 
                    priority=4, 
                    voice_profile="urgent"
                )
                self.alert_manager.send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                return False
            success_msg = f"SHAP features validées pour {market}"
            logger.info(success_msg)
            miya_speak(success_msg, tag="INFERENCE", voice_profile="calm")
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            return True
        except Exception as e:
            error_msg = (
                f"Erreur validation SHAP features pour {market}: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            logger.error(error_msg)
            miya_alerts(
                error_msg, 
                tag="INFERENCE", 
                priority=4, 
                voice_profile="urgent"
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            return False

    async def load_data(self, data_path: str, market: str = "ES") -> pd.DataFrame:
        """Charge les données via data_provider.py avec retries."""

        def fetch_data():
            provider = get_data_provider("iqfeed")
            data = provider.fetch_features(data_path)
            if data.empty:
                error_msg = f"Aucune donnée chargée depuis IQFeed pour {market}"
                raise ValueError(error_msg)
            self.validate_data(data, market=market)
            return data

        try:
            start_time = time.time()
            data = self.with_retries(fetch_data, market=market)
            latency = time.time() - start_time
            self.log_performance(
                "load_data", 
                latency, 
                success=True, 
                num_rows=len(data), 
                market=market
            )
            self.save_snapshot(
                "load_data",
                {"num_rows": len(data), "columns": list(data.columns)},
                market=market,
            )
            return data
        except Exception as e:
            error_msg = (
                f"Erreur chargement données pour {market}: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            logger.error(error_msg)
            miya_alerts(
                error_msg, 
                tag="INFERENCE", 
                priority=4, 
                voice_profile="urgent"
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            return pd.DataFrame()

    def validate_data(self, data: pd.DataFrame, market: str = "ES") -> None:
        """Valide que les données contiennent les features attendues."""
        try:
            expected_cols = self.configure_feature_set()
            missing_cols = [col for col in expected_cols if col not in data.columns]
            null_count = data[expected_cols].isnull().sum().sum()
            confidence_drop_rate = (
                null_count / (len(data) * len(expected_cols))
                if (len(data) * len(expected_cols)) > 0
                else 0.0
            )
            if confidence_drop_rate > 0.5:
                alert_msg = (
                    f"Confidence_drop_rate élevé pour {market}: "
                    f"{confidence_drop_rate:.2f} ({null_count} valeurs nulles)"
                )
                logger.warning(alert_msg)
                miya_alerts(alert_msg, tag="INFERENCE", priority=3)
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
            if missing_cols:
                error_msg = (
                    f"Colonnes manquantes dans les données pour {market}: "
                    f"{missing_cols}"
                )
                raise ValueError(error_msg)

            critical_cols = [
                "bid_size_level_1",
                "ask_size_level_1",
                "trade_frequency_1s",
                "spread_avg_1min",
                "close",
                f"predicted_volatility_{market.lower()}",
                "neural_regime",
            ]
            for col in critical_cols:
                if col in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        error_msg = (
                            f"Colonne {col} n'est pas numérique pour {market}: "
                            f"{data[col].dtype}"
                        )
                        raise ValueError(error_msg)
                    non_scalar = data[col].apply(
                        lambda x: isinstance(x, (list, dict, tuple))
                    )
                    if non_scalar.any():
                        non_scalar_values = data[col][non_scalar].head().tolist()
                        error_msg = (
                            f"Colonne {col} contient des valeurs non scalaires "
                            f"pour {market}: {non_scalar_values}"
                        )
                        raise ValueError(error_msg)
                    if data[col].isna().any():
                        error_msg = (
                            f"Colonne {col} contient des valeurs NaN pour {market}"
                        )
                        raise ValueError(error_msg)

            thresholds = {
                "bid_size_level_1": {"min": 0.0},
                "ask_size_level_1": {"min": 0.0},
                "trade_frequency_1s": {"min": 0.0},
                f"predicted_volatility_{market.lower()}": {"min": 0.0},
                "neural_regime": {"min": 0, "max": 2},
            }
            for col, th in thresholds.items():
                if col in data.columns:
                    if "min" in th and data[col].min() < th["min"]:
                        error_msg = (
                            f"Colonne {col} sous le seuil minimum pour {market}: "
                            f"{data[col].min()} < {th['min']}"
                        )
                        raise ValueError(error_msg)
                    if "max" in th and data[col].max() > th["max"]:
                        error_msg = (
                            f"Colonne {col} dépasse le seuil maximum pour {market}: "
                            f"{data[col].max()} > {th['max']}"
                        )
                        raise ValueError(error_msg)
            self.save_snapshot(
                "validate_data",
                {
                    "num_columns": len(data.columns Hannah: True
                    "missing_columns": missing_cols,
                    "confidence_drop_rate": confidence_drop_rate,
                    "market": market,
                },
                market=market,
            )
        except Exception as e:
            error_msg = (
                f"Erreur validation données pour {market}: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            logger.error(error_msg)
            miya_alerts(
                error_msg, 
                tag="INFERENCE", 
                priority=5, 
                voice_profile="urgent"
            )
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            raise


def load_model(
    config_path: str,
    mode: str,
    model_path: Optional[str] = None,
    policy_type: str = "mlp",
    market: str = "ES",
) -> Tuple[Any, TradingEnv]:
    """Charge un modèle SAC entraîné ou un agent spécifique au mode avec la 
    politique spécifiée."""
    try:
        valid_modes = ["trend", "range", "defensive"]
        valid_policies = ["mlp", "transformer"]
        if mode not in valid_modes:
            error_msg = (
                f"Mode non supporté pour {market}: {mode}. "
                f"Options: {valid_modes}"
            )
            raise ValueError(error_msg)
        if policy_type not in valid_policies:
            error_msg = (
                f"Type de politique non supporté pour {market}: {policy_type}. "
                f"Options: {valid_policies}"
            )
            raise ValueError(error_msg)

        get_config(config_path)
        save_dir = str(BASE_DIR / "model/sac_models" / market / "deployed")
        if model_path is None:
            mode_dir = os.path.join(save_dir, mode, policy_type)
            os.makedirs(mode_dir, exist_ok=True)
            model_files = [
                f
                for f in os.listdir(mode_dir)
                if f.startswith(f"sac_{mode}_") and f.endswith(".zip")
            ]
            if not model_files:
                error_msg = (
                    f"Aucun modèle trouvé pour le mode {mode} dans {mode_dir} "
                    f"pour {market}"
                )
                raise FileNotFoundError(error_msg)
            model_path = os.path.join(
                mode_dir,
                max(
                    model_files,
                    key=lambda x: os.path.getctime(os.path.join(mode_dir, x)),
                ),
            )

        engine = InferenceEngine(config_path)
        feature_cols = engine.configure_feature_set()
        env = TradingEnv(config_path)
        env.mode = mode
        env.policy_type = policy_type
        env.obs_t = feature_cols
        if policy_type == "transformer":
            env.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(env.sequence_length, len(feature_cols)),
                dtype=np.float32,
            )
        else:
            env.observation_space = spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(len(feature_cols),), 
                dtype=np.float32
            )

        env = ObservationStackingWrapper(
            env, 
            stack_size=3, 
            config_path=config_path, 
            policy_type=policy_type
        )
        env = NormalizationWrapper(
            env, 
            config_path=config_path, 
            policy_type=policy_type
        )
        env = ClippingWrapper(
            env,
            clip_min=-5.0,
            clip_max=5.0,
            config_path=config_path,
            policy_type=policy_type,
        )

        if mode == "trend":
            agent = ModeTrend(model_path=model_path, env=env)
        elif mode == "range":
            agent = ModeRange(model_path=model_path, env=env)
        elif mode == "defensive":
            agent = ModeDefensive(model_path=model_path, env=env)

        success_msg = (
            f"Agent {mode} chargé pour {market}: {model_path}, "
            f"politique={policy_type}, obs_space={env.observation_space.shape}"
        )
        logger.info(success_msg)
        miya_speak(success_msg, tag="INFERENCE", voice_profile="calm")
        engine.alert_manager.send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
        return agent, env

    except Exception as e:
        error_msg = (
            f"Erreur chargement modèle mode={mode}, policy_type={policy_type} "
            f"pour {market}: {str(e)}\n{traceback.format_exc()}"
        )
        logger.error(error_msg)
        miya_alerts(
            error_msg, 
            tag="INFERENCE", 
            priority=5, 
            voice_profile="urgent"
        )
        engine.alert_manager.send_alert(error_msg, priority=5)
        send_telegram_alert(error_msg)
        raise


async def predict_sac(
    config_path: str = str(BASE_DIR / "config/es_config.yaml"),
    data_path: str = str(BASE_DIR / "data/features/features_latest_filtered.csv"),
    mode: Optional[str] = None,
    model_path: Optional[str] = None,
    dynamic_mode: bool = True,
    policy_type: str = "mlp",
    market: str = "ES",
) -> pd.DataFrame:
    """Effectue des prédictions avec un modèle SAC entraîné, avec option de détection 
    dynamique du régime."""
    engine = InferenceEngine(config_path)
    try:
        if not os.path.exists(config_path):
            error_msg = (
                f"Fichier de configuration introuvable pour {market}: {config_path}"
            )
            raise FileNotFoundError(error_msg)
        if not os.path.exists(data_path):
            error_msg = (
                f"Fichier de données introuvable pour {market}: {data_path}"
            )
            raise FileNotFoundError(error_msg)
        valid_policies = ["mlp", "transformer"]
        if policy_type not in valid_policies:
            error_msg = (
                f"Type de politique non supporté pour {market}: {policy_type}. "
                f"Options: {valid_policies}"
            )
            raise ValueError(error_msg)
        valid_modes = ["trend", "range", "defensive"]
        if mode and mode not in valid_modes:
            error_msg = (
                f"Mode non supporté pour {market}: {mode}. "
                f"Options: {valid_modes}"
            )
            raise ValueError(error_msg)

        if not engine.validate_shap_features(market=market):
            error_msg = f"Échec validation SHAP features pour {market}"
            raise ValueError(error_msg)

        data = await engine.load_data(data_path, market=market)
        if data.empty:
            error_msg = f"Aucune donnée valide chargée pour {market}"
            raise ValueError(error_msg)
        success_msg = (
            f"Données chargées pour {market}: {data_path}, "
            f"{len(data)} lignes, {len(data.columns)} colonnes, "
            f"policy_type={policy_type}"
        )
        logger.info(success_msg)
        miya_speak(success_msg, tag="INFERENCE", voice_profile="calm")
        engine.alert_manager.send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)

        env = TradingEnv(config_path)
        env.policy_type = policy_type
        env.obs_t = engine.configure_feature_set()
        env.data = data
        if policy_type == "transformer":
            env.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(env.sequence_length, len(env.obs_t)),
                dtype=np.float32,
            )
        else:
            env.observation_space = spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(len(env.obs_t),), 
                dtype=np.float32
            )

        env = ObservationStackingWrapper(
            env, 
            stack_size=3, 
            config_path=config_path, 
            policy_type=policy_type
        )
        env = NormalizationWrapper(
            env, 
            config_path=config_path, 
            policy_type=policy_type
        )
        env = ClippingWrapper(
            env,
            clip_min=-5.0,
            clip_max=5.0,
            config_path=config_path,
            policy_type=policy_type,
        )

        sequence_length = getattr(env, "sequence_length", 50)
        success_msg = (
            f"Environnement initialisé pour {market}: "
            f"{len(env.obs_t)} features, "
            f"obs_space={env.observation_space.shape}"
        )
        logger.info(success_msg)
        miya_speak(success_msg, tag="INFERENCE", voice_profile="calm")
        engine.alert_manager.send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)

        neural_pipeline = NeuralPipeline(
            window_size=sequence_length,
            base_features=len(env.obs_t),
            config_path=str(BASE_DIR / "config/model_params.yaml"),
        )
        try:
            neural_pipeline.load_models()
            success_msg = f"NeuralPipeline chargé pour inférence pour {market}"
            logger.info(success_msg)
            miya_speak(success_msg, tag="INFERENCE", voice_profile="calm")
            engine.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
        except Exception as e:
            error_msg = (
                f"Erreur chargement NeuralPipeline pour {market}: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            logger.error(error_msg)
            miya_alerts(
                error_msg, 
                tag="INFERENCE", 
                priority=4, 
                voice_profile="urgent"
            )
            engine.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            raise

        agents = {}
        detector = None
        if dynamic_mode:
            detector = MarketRegimeDetector(config_path)
            for m in valid_modes:
                agents[m], _ = load_model(
                    config_path, 
                    m, 
                    None, 
                    policy_type, 
                    market=market
                )
        else:
            if mode is None:
                error_msg = (
                    f"Mode doit être spécifié si dynamic_mode=False pour {market}"
                )
                raise ValueError(error_msg)
            agent, env = load_model(
                config_path, 
                mode, 
                model_path, 
                policy_type, 
                market=market
            )

        obs, _ = env.reset()
        predictions = []
        rewards_history = []
        trade_log = []

        for step in range(len(data)):
            start_step_time = time.time()
            validate_features(
                data, 
                step, 
                shap_features=(policy_type == "transformer"), 
                market=market
            )

            if dynamic_mode:
                regime, details = detector.detect(data, step)
                neural_regime = details.get("neural_regime", None)
                if neural_regime is None or any(
                    col not in data.columns
                    for col in [
                        f"predicted_volatility_{market.lower()}",
                        "neural_regime",
                        "cnn_pressure",
                    ]
                ):
                    try:
                        window_start = max(0, step - sequence_length + 1)
                        window_end = step + 1
                        raw_data = (
                            data[
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
                            data[
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
                            data[["timestamp", "bid_size_level_1", "ask_size_level_1"]]
                            .iloc[window_start:window_end]
                            .fillna(0)
                        )
                        neural_result = neural_pipeline.run(
                            raw_data, 
                            options_data, 
                            orderflow_data
                        )
                        volatility_col = f"predicted_volatility_{market.lower()}"
                        data.loc[step, volatility_col] = neural_result["volatility"][-1]
                        data.loc[step, "neural_regime"] = neural_result["regime"][-1]
                        feature_cols = [f"neural_feature_{i}" for i in range(8)] + [
                            "cnn_pressure"
                        ]
                        for i, col in enumerate(feature_cols):
                            data.loc[step, col] = neural_result["features"][-1, i]
                        neural_regime = int(neural_result["regime"][-1])
                        success_msg = (
                            f"Régime neuronal calculé au step {step} pour {market}: "
                            f"{neural_regime}"
                        )
                        logger.info(success_msg)
                        miya_speak(success_msg, tag="INFERENCE", voice_profile="calm")
                        engine.alert_manager.send_alert(success_msg, priority=1)
                        send_telegram_alert(success_msg)
                    except Exception as e:
                        error_msg = (
                            f"Erreur calcul neuronal au step {step} pour {market}: "
                            f"{str(e)}\n{traceback.format_exc()}"
                        )
                        logger.error(error_msg)
                        miya_alerts(
                            error_msg,
                            tag="INFERENCE",
                            priority=4,
                            voice_profile="urgent",
                        )
                        engine.alert_manager.send_alert(error_msg, priority=4)
                        send_telegram_alert(error_msg)
                        neural_regime = None
                agent = agents.get(regime, agents["defensive"])
                env.mode = regime
            else:
                regime = mode
                neural_regime = (
                    data["neural_regime"].iloc[step]
                    if "neural_regime" in data.columns
                    else None
                )

            risk_factor = 1.0
            drawdown = 0.0
            if rewards_history:
                cum_rewards = np.cumsum(rewards_history)
                drawdown = float(
                    np.min(cum_rewards - np.maximum.accumulate(cum_rewards)) / 10000
                )
                if drawdown < PERFORMANCE_THRESHOLDS["max_drawdown"]:
                    risk_factor = 0.5
                    warning_msg = (
                        f"Step {step}: Drawdown excessif ({drawdown:.2f}) < "
                        f"{PERFORMANCE_THRESHOLDS['max_drawdown']} pour {market}, "
                        f"risque réduit à {risk_factor}"
                    )
                    logger.warning(warning_msg)
                    miya_speak(
                        warning_msg, 
                        tag="INFERENCE", 
                        voice_profile="warning"
                    )
                    engine.alert_manager.send_alert(warning_msg, priority=3)
                    send_telegram_alert(warning_msg)
                atr = float(data[f"atr_14_{market.lower()}"].iloc[step])
                spread = float(data.get("spread_avg_1min", pd.Series(0.25)).iloc[step])
                adjust_risk_and_leverage(env, drawdown, atr, spread)
                if atr > 2.0:
                    risk_factor = min(risk_factor, 0.5)
                    warning_msg = (
                        f"Step {step}: Volatilité extrême (ATR={atr:.2f}) "
                        f"pour {market}, risque réduit à {risk_factor}"
                    )
                    logger.warning(warning_msg)
                    miya_speak(
                        warning_msg, 
                        tag="INFERENCE", 
                        voice_profile="warning"
                    )
                    engine.alert_manager.send_alert(warning_msg, priority=3)
                    send_telegram_alert(warning_msg)

            if policy_type == "transformer" and step >= sequence_length - 1:
                obs_seq = (
                    data[env.obs_t]
                    .iloc[step - sequence_length + 1 : step + 1]
                    .values
                )
                if len(obs_seq) < sequence_length:
                    padding = np.zeros((sequence_length - len(obs_seq), len(env.obs_t)))
                    obs_seq = np.vstack([padding, obs_seq])
                obs = obs_seq.astype(np.float32)
            else:
                obs = data[env.obs_t].iloc[step].values.astype(np.float32)

            action = 0.0
            confidence = 0.7
            if step >= env.sequence_length:
                range_model_path = str(
                    BASE_DIR / f"model/ml_models/{regime}_filter.pkl"
                )
                order_flow_model_path = str(
                    BASE_DIR / "model/ml_models/order_flow_filter.pkl"
                )
                if not validate_trade_entry_combined(
                    env,
                    range_model_path=range_model_path,
                    order_flow_model_path=order_flow_model_path,
                    sequence_length=sequence_length,
                ):
                    warning_msg = (
                        f"Step {step}: Trade rejeté par filtre ML pour {regime} "
                        f"pour {market}"
                    )
                    logger.warning(warning_msg)
                    miya_speak(
                        warning_msg, 
                        tag="INFERENCE", 
                        voice_profile="warning"
                    )
                    engine.alert_manager.send_alert(warning_msg, priority=2)
                    send_telegram_alert(warning_msg)
                else:
                    action, _ = agent.predict(obs, data, step, rewards_history)
                    confidence = 0.9
                    pattern_data = data.iloc[[step]].copy()
                    trade_success_prob = engine.trade_predictor.predict(pattern_data)
                    if trade_success_prob < 0.5:
                        action = 0.0
                        warning_msg = (
                            f"Step {step}: Trade rejeté pour {market}, "
                            f"trade_success_prob={trade_success_prob:.2f} < 0.5"
                        )
                        logger.warning(warning_msg)
                        miya_speak(
                            warning_msg, 
                            tag="INFERENCE", 
                            voice_profile="warning"
                        )
                        engine.alert_manager.send_alert(warning_msg, priority=3)
                        send_telegram_alert(warning_msg)
            else:
                action, _ = agent.predict(obs, data, step, rewards_history)
                trade_success_prob = 0.5

            action = float(action) * risk_factor

            env.current_step = step
            obs, reward, done, _, info = env.step(np.array([action]))
            rewards_history.append(float(reward))

            try:
                pattern_data = data.iloc[[step]].copy()
                db_path = str(BASE_DIR / "data/market_memory.db")
                store_pattern(
                    pattern_data,
                    action=action,
                    reward=reward,
                    neural_regime=neural_regime,
                    confidence=confidence,
                    db_path=db_path,
                )
            except Exception as e:
                error_msg = (
                    f"Erreur stockage pattern au step {step} pour {market}: "
                    f"{str(e)}\n{traceback.format_exc()}"
                )
                logger.error(error_msg)
                miya_alerts(
                    error_msg, 
                    tag="INFERENCE", 
                    priority=3, 
                    voice_profile="urgent"
                )
                engine.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)

            db_path = str(BASE_DIR / "data/market_memory.db")
            success_rate = last_3_trades_success_rate(db_path=db_path)
            info["success_rate"] = success_rate

            prediction = {
                "step": step,
                "timestamp": data["timestamp"].iloc[step],
                "action": action,
                "reward": reward,
                "regime": regime,
                "neural_regime": (
                    neural_regime if neural_regime is not None else "N/A"
                ),
                "balance": info["balance"],
                "position": info["position"],
                "price": info["price"],
                "drawdown": drawdown,
                "risk_factor": risk_factor,
                "success_rate": success_rate,
                "confidence": confidence,
                "trade_success_prob": trade_success_prob,
            }
            predictions.append(prediction)
            trade_log.append(prediction)

            engine.save_snapshot("prediction", prediction, market=market)

            for metric, value in [
                ("reward", reward),
                ("balance", info["balance"]),
                ("drawdown", drawdown),
            ]:
                threshold = PERFORMANCE_THRESHOLDS[
                    f"min_{metric}" if metric != "drawdown" else "max_drawdown"
                ]
                if (metric != "drawdown" and value < threshold) or (
                    metric == "drawdown" and value < threshold
                ):
                    warning_msg = (
                        f"Seuil non atteint au step {step} pour {market}: "
                        f"{metric.capitalize()} ({value:.2f}) < {threshold}"
                    )
                    logger.warning(warning_msg)
                    miya_speak(
                        warning_msg, 
                        tag="INFERENCE", 
                        voice_profile="warning"
                    )
                    engine.alert_manager.send_alert(warning_msg, priority=3)
                    send_telegram_alert(warning_msg)

            if done:
                success_msg = (
                    f"Prédiction terminée au step {step} pour {market}: "
                    f"condition 'done'"
                )
                logger.info(success_msg)
                miya_speak(success_msg, tag="INFERENCE", voice_profile="calm")
                engine.alert_manager.send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)
                break

            step_latency = time.time() - start_step_time
            engine.log_performance(
                "predict_step", 
                step_latency, 
                success=True, 
                step=step, 
                market=market
            )

        result_df = pd.DataFrame(predictions)
        if not result_df.empty:
            metrics = compute_metrics(result_df, policy_type=policy_type, market=market)
            success_msg = f"Métriques globales pour {market}: {metrics}"
            logger.info(success_msg)
            miya_speak(success_msg, tag="INFERENCE", voice_profile="calm")
            engine.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)

            sharpe_ratio = metrics.get("sharpe_ratio", 0.0)
            if sharpe_ratio < PERFORMANCE_THRESHOLDS["min_sharpe"]:
                error_msg = (
                    f"Seuil global non atteint pour {market}: "
                    f"Sharpe Ratio ({sharpe_ratio:.2f}) < "
                    f"{PERFORMANCE_THRESHOLDS['min_sharpe']}"
                )
                logger.error(error_msg)
                miya_alerts(
                    error_msg, 
                    tag="INFERENCE", 
                    priority=4, 
                    voice_profile="urgent"
                )
                engine.alert_manager.send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
            total_reward = metrics.get("total_reward", 0.0)
            total_loss = metrics.get("total_loss", 1.0) + 1e-6
            profit_factor = total_reward / total_loss
            if profit_factor < PERFORMANCE_THRESHOLDS["min_profit_factor"]:
                error_msg = (
                    f"Seuil global non atteint pour {market}: "
                    f"Profit Factor ({profit_factor:.2f}) < "
                    f"{PERFORMANCE_THRESHOLDS['min_profit_factor']}"
                )
                logger.error(error_msg)
                miya_alerts(
                    error_msg, 
                    tag="INFERENCE", 
                    priority=4, 
                    voice_profile="urgent"
                )
                engine.alert_manager.send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)

            engine.checkpoint(result_df, data_type="predictions", market=market)
            engine.cloud_backup(result_df, data_type="predictions", market=market)

        config = get_config(config_path)
        output_dir = str(
            BASE_DIR / config.get("logging", {}).get("directory", "data/logs")
        )
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        output_path = (
            output_dir / 
            f"inference_{mode if mode else 'dynamic'}_{policy_type}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        export_logs(
            trade_log,
            str(output_dir),
            output_path.name,
            policy_type=policy_type,
            market=market,
        )
        success_msg = (
            f"Prédictions sauvegardées pour {market}: {output_path}, "
            f"{len(result_df)} prédictions"
        )
        logger.info(success_msg)
        miya_speak(success_msg, tag="INFERENCE", voice_profile="calm")
        engine.alert_manager.send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)

        return result_df

    except Exception as e:
        error_msg = (
            f"Erreur prédiction, policy_type={policy_type} pour {market}: "
            f"{str(e)}\n{traceback.format_exc()}"
        )
        logger.error(error_msg)
        miya_alerts(
            error_msg, 
            tag="INFERENCE", 
            priority=5, 
            voice_profile="urgent"
        )
        engine.alert_manager.send_alert(error_msg, priority=5)
        send_telegram_alert(error_msg)
        engine.save_snapshot("error", {"error": str(e)}, market=market)
        return pd.DataFrame()


if __name__ == "__main__":
    try:
        config_path = str(BASE_DIR / "config/es_config.yaml")
        data_path = str(BASE_DIR / "data/features/features_latest_filtered.csv")

        result_fixed = asyncio.run(
            predict_sac(
                config_path=config_path,
                data_path=data_path,
                mode="trend",
                policy_type="mlp",
                market="ES",
            )
        )
        success_msg = "Mode fixe (trend) avec mlp pour ES:"
        logger.info(success_msg)
        miya_speak(success_msg, tag="TEST", voice_profile="calm")
        engine.alert_manager.send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
        print(success_msg)
        print(result_fixed.head())

        result_dynamic = asyncio.run(
            predict_sac(
                config_path=config_path,
                data_path=data_path,
                dynamic_mode=True,
                policy_type="transformer",
                market="MNQ",
            )
        )
        success_msg = "Mode dynamique avec transformer pour MNQ:"
        logger.info(success_msg)
        miya_speak(success_msg, tag="TEST", voice_profile="calm")
        engine.alert_manager.send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
        print("\n" + success_msg)
        print(result_dynamic.head())

    except Exception as e:
        error_msg = (
            f"Erreur test principal: {str(e)}\n{traceback.format_exc()}"
        )
        logger.error(error_msg)
        miya_alerts(
            error_msg, 
            tag="INFERENCE", 
            priority=5, 
            voice_profile="urgent"
        )
        engine.alert_manager.send_alert(error_msg, priority=5)
        send_telegram_alert(error_msg)
        raise