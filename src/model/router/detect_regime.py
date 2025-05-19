# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/model/router/detect_regime.py
# Détecte les régimes de marché (trend, range, défensif, ultra-défensif) pour activer le modèle SAC correspondant.
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle :
# Détecte les régimes de marché en utilisant 350 features (entraînement) ou 150 SHAP features (inférence), intégrant volatilité (méthode 1),
# données d’options (méthode 2), régimes hybrides (méthode 11), SHAP (méthode 17), et prédictions LSTM (méthode 12) via NeuralPipeline.
# Fournit des probabilités de régimes et des analyses d’importance des features.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, ta>=0.10.0,<0.11.0, pydantic>=2.0.0,<3.0.0, psutil>=5.9.8,<6.0.0,
#   matplotlib>=3.7.0,<4.0.0, sklearn>=1.5.0,<2.0.0, shap>=0.44.0,<0.45.0, boto3>=1.26.0,<2.0.0,
#   loguru>=0.7.0,<1.0.0, pyyaml>=6.0.0,<7.0.0, asyncio
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/features/neural_pipeline.py
# - src/features/feature_pipeline.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/feature_sets.yaml (350 features et 150 SHAP features)
# - config/router_config.yaml, config/model_params.yaml via config_manager
# - Données brutes (DataFrame avec 350/150 features)
#
# Outputs :
# - Régime détecté (str), détails (dict avec probabilités et importance des features)
# - Logs dans data/logs/regime_detection.log
# - Logs de performance dans data/logs/regime_performance.csv
# - Snapshots compressés dans data/cache/regime/*.json.gz
# - Sauvegardes incrémentielles dans data/checkpoints/regime_*.json.gz
# - Visualisations dans data/figures/regime/
# - Importance des features dans data/feature_importance.csv
#
# Lien avec SAC :
# Fournit le régime et probabilités pour activer SACRange, SACTendance, SACDefensif dans train_sac.py
#
# Notes :
# - Intègre volatilité, données d’options, régimes hybrides, SHAP, et prédictions LSTM.
# - Valide 350 features (entraînement) ou 150 SHAP features (inférence) via config_manager.
# - Utilise IQFeed comme source de données via TradingEnv.
# - Implémente retries (max 3, délai 2^attempt secondes) pour les opérations critiques.
# - Sauvegardes incrémentielles toutes les 5 min dans data/checkpoints/ (5 versions).
# - Sauvegardes distribuées toutes les 15 min vers S3 (configurable).
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des données.
# - Tests unitaires disponibles dans tests/test_detect_regime.py.
# - Validation complète prévue pour juin 2025.

import asyncio
import signal
from collections import OrderedDict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import shap
import ta
from loguru import logger
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler

from src.features.feature_pipeline import FeaturePipeline
from src.features.neural_pipeline import NeuralPipeline
from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import get_config
from src.utils.telegram_alert import send_telegram_alert

# Configuration logging
logger.remove()
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOG_DIR = BASE_DIR / "data" / "logs"
CACHE_DIR = BASE_DIR / "data" / "cache" / "regime"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints"
FIGURE_DIR = BASE_DIR / "data" / "figures" / "regime"
PERF_LOG_PATH = LOG_DIR / "regime_performance.csv"
SNAPSHOT_DIR = CACHE_DIR
FEATURE_IMPORTANCE_PATH = BASE_DIR / "data" / "feature_importance.csv"
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "regime_detection.log", rotation="10 MB", level="INFO", encoding="utf-8"
)
logger.add(lambda msg: print(msg, end=""), level="INFO")

# Constantes
MAX_CACHE_SIZE = 1000
PERFORMANCE_WINDOW_SIZE = 100
RECENT_REGIMES_SIZE = 3
MAX_RETRIES = 3
RETRY_DELAY = 2.0
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB

# Variable pour gérer l'arrêt propre
RUNNING = True


class InputData(BaseModel):
    df: pd.DataFrame
    current_step: int

    class Config:
        arbitrary_types_allowed = True


class MarketRegimeDetector:
    def __init__(
        self,
        env: str = "prod",
        log_history: bool = True,
        policy_type: str = "mlp",
        training_mode: bool = True,
    ):
        """
        Initialise le détecteur de régime avec 350 features (entraînement) ou 150 SHAP features (inférence).

        Args:
            env (str): Environnement ("prod" ou "test").
            log_history (bool): Si True, enregistre l’historique des régimes.
            policy_type (str): Type de politique ("mlp" ou "transformer").
            training_mode (bool): Si True, utilise 350 features; sinon, 150 SHAP features.
        """
        start_time = datetime.now()
        try:
            self.alert_manager = AlertManager()
            self.env = env
            self.log_history = log_history
            self.policy_type = policy_type
            self.training_mode = training_mode
            self.num_features = 350 if training_mode else 150
            self.snapshot_dir = SNAPSHOT_DIR
            self.perf_log = PERF_LOG_PATH
            self.figure_dir = FIGURE_DIR
            self.feature_importance_path = FEATURE_IMPORTANCE_PATH
            self.checkpoint_versions = []
            self.log_buffer = []

            # Charger configurations
            self.config = get_config(BASE_DIR / "config/router_config.yaml")
            self.model_config = get_config(BASE_DIR / "config/model_params.yaml")
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

            # Historique
            self.history_file = BASE_DIR / "data" / "logs" / "regime_history.csv"
            if log_history and not self.history_file.exists():
                pd.DataFrame(
                    columns=[
                        "step",
                        "regime",
                        "regime_probs",
                        "feature_importance",
                        "adx_14",
                        "atr_14",
                        "gex",
                        "gex_change",
                        "bid_ask_ratio",
                        "bid_ask_ratio_level_2",
                        "vwap_deviation",
                        "rsi_14",
                        "ofi_score",
                        "vix_es_correlation",
                        "neural_regime",
                        "confidence_score",
                        "spread",
                        "call_iv_atm",
                        "option_skew",
                        "option_volume",
                        "oi_concentration",
                        "predicted_vix",
                        "timestamp",
                        "error",
                    ]
                ).to_csv(self.history_file, index=False)

            # Paramètres
            self.impute_nan = self.config.get("impute_nan", False)
            self.use_optimized_calculations = self.config.get(
                "use_optimized_calculations", True
            )
            self.fast_mode = self.config.get("fast_mode", False)
            self.enable_heartbeat = self.config.get("enable_heartbeat", True)
            self.critical_times = self.config.get("critical_times", ["14:00", "15:30"])

            # Cache et contexte
            self.neural_cache = OrderedDict()
            self.last_neural_regime = "N/A"
            self.performance_window = deque(maxlen=PERFORMANCE_WINDOW_SIZE)
            self.recent_regimes = deque(maxlen=RECENT_REGIMES_SIZE)

            # NeuralPipeline et FeaturePipeline
            self.neural_pipeline = NeuralPipeline(
                window_size=self.model_config.get("neural_pipeline", {}).get(
                    "window_size", 50
                ),
                base_features=self.num_features,
                config_path=str(BASE_DIR / "config/model_params.yaml"),
            )
            self.feature_pipeline = FeaturePipeline(
                window_size=self.model_config.get("neural_pipeline", {}).get(
                    "window_size", 50
                ),
                base_features=self.num_features,
                config_path=str(BASE_DIR / "config/model_params.yaml"),
            )
            self.neural_pipeline.load_models()
            self.feature_pipeline.load_pipeline()
            success_msg = "NeuralPipeline et FeaturePipeline initialisés avec succès"
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
            error_msg = f"Erreur initialisation MarketRegimeDetector: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            raise

    def handle_sigint(self, signum, frame):
        """Gère l'arrêt propre sur SIGINT."""
        global RUNNING
        datetime.now()
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "status": "SIGINT",
            "regime": self.last_neural_regime,
        }
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

    def checkpoint(self, data: pd.DataFrame, data_type: str = "regime_state") -> None:
        """
        Sauvegarde incrémentielle des données toutes les 5 minutes avec versionnage (5 versions).

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : regime_state).
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
            checkpoint_path = CHECKPOINT_DIR / f"regime_{data_type}_{timestamp}.json.gz"
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

    def cloud_backup(self, data: pd.DataFrame, data_type: str = "regime_state") -> None:
        """
        Sauvegarde distribuée des données vers AWS S3 toutes les 15 minutes (configurable).

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : regime_state).
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
                f"{self.config['s3_prefix']}regime_{data_type}_{timestamp}.csv.gz"
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
        Journalise les performances dans regime_performance.csv.

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

    async def check_neural_pipeline_health(self) -> bool:
        """
        Vérifie la santé de NeuralPipeline de manière asynchrone.

        Returns:
            bool: True si la pipeline est saine, False sinon.
        """
        try:
            start_time = datetime.now()
            test_data = pd.DataFrame(
                {
                    "timestamp": pd.date_range(
                        "2025-05-13 09:00", periods=50, freq="1min"
                    ),
                    **{col: np.random.uniform(0, 1, 50) for col in self.feature_cols},
                    "open": np.random.uniform(5000, 5100, 50),
                    "high": np.random.uniform(5100, 5200, 50),
                    "low": np.random.uniform(4900, 5000, 50),
                    "close": np.random.uniform(5000, 5100, 50),
                    "volume": np.random.randint(1000, 10000, 50),
                    "bid_size_level_1": np.random.randint(100, 500, 50),
                    "ask_size_level_1": np.random.randint(100, 500, 50),
                    "gex": np.random.uniform(-1000, 1000, 50),
                    "oi_peak_call_near": np.random.randint(1000, 5000, 50),
                    "gamma_wall_call": np.random.uniform(5000, 5200, 50),
                    "call_iv_atm": np.random.uniform(0.1, 0.3, 50),
                    "option_skew": np.random.uniform(-0.05, 0.05, 50),
                    "option_volume": np.random.randint(1000, 5000, 50),
                    "oi_concentration": np.random.uniform(0.1, 0.9, 50),
                }
            )
            raw_data = test_data[
                [
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "atr_14",
                    "adx_14",
                ]
            ].fillna(0)
            options_data = test_data[
                [
                    "timestamp",
                    "gex",
                    "oi_peak_call_near",
                    "gamma_wall_call",
                    "call_iv_atm",
                    "option_skew",
                    "option_volume",
                    "oi_concentration",
                ]
            ].fillna(0)
            orderflow_data = test_data[
                ["timestamp", "bid_size_level_1", "ask_size_level_1"]
            ].fillna(0)
            neural_result = self.neural_pipeline.run(
                raw_data, options_data, orderflow_data
            )
            if (
                not isinstance(neural_result, dict)
                or "features" not in neural_result
                or len(neural_result["features"][-1]) != self.num_features
            ):
                raise ValueError("Sortie NeuralPipeline non conforme")
            success_msg = "Heartbeat check NeuralPipeline réussi"
            logger.debug(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("check_neural_pipeline_health", latency, success=True)
            return True
        except Exception as e:
            error_msg = (
                f"Échec heartbeat NeuralPipeline: {str(e)}\n{traceback.format_exc()}"
            )
            logger.warning(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "check_neural_pipeline_health", latency, success=False, error=str(e)
            )
            return False

    def prune_cache(self):
        """Supprime les entrées anciennes du cache."""
        while len(self.neural_cache) > MAX_CACHE_SIZE:
            self.neural_cache.popitem(last=False)
        logger.debug(f"Cache NeuralPipeline purgé, taille: {len(self.neural_cache)}")

    def update_thresholds(
        self,
        adx: float,
        atr: float,
        predicted_volatility: float,
        vix: float,
        atr_normalized: float,
        call_iv_atm: float,
    ):
        """
        Met à jour dynamiquement les seuils.

        Args:
            adx (float): Valeur ADX.
            atr (float): Valeur ATR.
            predicted_volatility (float): Volatilité prédite.
            vix (float): VIX.
            atr_normalized (float): ATR normalisé.
            call_iv_atm (float): Implied volatility ATM des calls.
        """
        try:
            start_time = datetime.now()
            thresholds = self.config.get("thresholds", {})
            if (
                vix > thresholds.get("vix_peak_threshold", 30.0)
                or atr_normalized > thresholds.get("volatility_spike_threshold", 1.0)
                or call_iv_atm > thresholds.get("iv_high_threshold", 0.25)
            ):
                self.performance_window.clear()
                alert_msg = f"Reset performance_window: VIX={vix:.2f}, ATR_normalized={atr_normalized:.2f}, IV={call_iv_atm:.2f}"
                logger.info(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
            self.performance_window.append(
                {"adx": adx, "atr": atr, "predicted_volatility": predicted_volatility}
            )
            if len(self.performance_window) >= 10:
                window_df = pd.DataFrame(self.performance_window)
                thresholds["min_adx"] = window_df["adx"].mean() - window_df["adx"].std()
                thresholds["min_atr"] = window_df["atr"].mean() - window_df["atr"].std()
                thresholds["min_predicted_volatility"] = (
                    window_df["predicted_volatility"].mean()
                    - window_df["predicted_volatility"].std()
                )
                logger.debug(f"Seuils mis à jour: {thresholds}")
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("update_thresholds", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur mise à jour seuils: {str(e)}\n{traceback.format_exc()}"
            logger.warning(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance("update_thresholds", 0, success=False, error=str(e))

    def check_transition_validity(self, new_regime: str) -> bool:
        """
        Vérifie la validité de la transition vers un nouveau régime.

        Args:
            new_regime (str): Nouveau régime détecté.

        Returns:
            bool: True si la transition est valide, False sinon.
        """
        try:
            if not self.recent_regimes:
                return True
            return (
                new_regime in self.recent_regimes or len(set(self.recent_regimes)) > 1
            )
        except Exception as e:
            error_msg = (
                f"Erreur vérification transition: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            return False

    def precompute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les indicateurs techniques et d'options pour enrichir les données.

        Args:
            df (pd.DataFrame): Données d’entrée.

        Returns:
            pd.DataFrame: Données enrichies avec indicateurs.
        """
        try:
            start_time = datetime.now()
            df = df.copy()
            # Indicateurs techniques
            df["atr_14"] = ta.volatility.AverageTrueRange(
                df["high"], df["low"], df["close"], window=14
            ).average_true_range()
            df["adx_14"] = ta.trend.ADXIndicator(
                df["high"], df["low"], df["close"], window=14
            ).adx()
            df["vwap"] = ta.volume.VolumeWeightedAveragePrice(
                df["high"], df["low"], df["close"], df["volume"]
            ).volume_weighted_average_price()
            df["rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
            df["bollinger_width"] = ta.volatility.BollingerBands(
                df["close"]
            ).bollinger_wband()
            df["momentum"] = ta.momentum.ROCIndicator(df["close"], window=14).roc()
            # Indicateurs d'options
            df["iv_atm"] = (
                (df["call_iv_atm"] + df["put_iv_atm"]) / 2
                if "call_iv_atm" in df and "put_iv_atm" in df
                else 0.0
            )
            df["option_skew"] = (
                df["call_iv_atm"] - df["put_iv_atm"]
                if "call_iv_atm" in df and "put_iv_atm" in df
                else 0.0
            )
            df["option_volume"] = (
                df["call_volume"] + df["put_volume"]
                if "call_volume" in df and "put_volume" in df
                else 0.0
            )
            df["oi_concentration"] = (
                df["oi_peak_call_near"]
                / (df["oi_peak_call_near"] + df["oi_peak_put_near"] + 1e-6)
                if "oi_peak_call_near" in df and "oi_peak_put_near" in df
                else 0.0
            )
            # Imputation des NaN
            if self.impute_nan:
                df.fillna(0, inplace=True)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "precompute_indicators", latency, success=True, num_rows=len(df)
            )
            return df
        except Exception as e:
            error_msg = f"Erreur calcul indicateurs: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "precompute_indicators", 0, success=False, error=str(e)
            )
            raise

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Valide que les données contiennent les features attendues.

        Args:
            df (pd.DataFrame): Données d’entrée.

        Returns:
            bool: True si les données sont valides, False sinon.
        """
        try:
            start_time = datetime.now()
            missing_cols = [col for col in self.feature_cols if col not in df.columns]
            if missing_cols:
                alert_msg = f"Colonnes manquantes: {missing_cols}"
                logger.error(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
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
                alert_msg = f"Valeurs nulles détectées dans les features: {null_count}"
                logger.error(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                return False
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                alert_msg = "Colonne 'timestamp' doit être de type datetime"
                logger.error(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
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

    def calculate_shap_values(self, df: pd.DataFrame, step: int) -> Dict[str, float]:
        """
        Calcule les valeurs SHAP pour l’importance des features (limité à 50 features).

        Args:
            df (pd.DataFrame): Données d’entrée.
            step (int): Étape actuelle.

        Returns:
            Dict[str, float]: Importance des 50 features les plus influentes.
        """
        try:
            start_time = datetime.now()
            from sklearn.ensemble import RandomForestClassifier

            X = df[self.feature_cols].iloc[max(0, step - 50) : step + 1].fillna(0)
            y = np.random.choice(
                [0, 1, 2], size=len(X)
            )  # Placeholder pour les labels (régimes)
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X.iloc[-1:])
            shap_dict = {
                col: abs(float(shap_values[0][-1, i]))
                for i, col in enumerate(self.feature_cols)
            }
            sorted_shap = dict(
                sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)[:50]
            )
            total = sum(sorted_shap.values()) + 1e-6
            sorted_shap = {
                k: v / total for k, v in sorted_shap.items()
            }  # Normalisation
            shap_df = pd.DataFrame(
                [{"feature": k, "importance": v} for k, v in sorted_shap.items()]
            )

            def save_shap():
                if not self.feature_importance_path.exists():
                    shap_df.to_csv(
                        self.feature_importance_path, index=False, encoding="utf-8"
                    )
                else:
                    shap_df.to_csv(
                        self.feature_importance_path,
                        mode="a",
                        header=False,
                        index=False,
                        encoding="utf-8",
                    )

            self.with_retries(save_shap)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "calculate_shap_values",
                latency,
                success=True,
                num_features=len(sorted_shap),
            )
            return sorted_shap
        except Exception as e:
            error_msg = f"Erreur calcul SHAP: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "calculate_shap_values", 0, success=False, error=str(e)
            )
            return {}

    async def detect(
        self, df: pd.DataFrame, current_step: int
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Détecte le régime de marché avec volatilité, données d’options, régimes hybrides, SHAP, et prédictions LSTM.

        Args:
            df (pd.DataFrame): Données d’entrée.
            current_step (int): Étape actuelle.

        Returns:
            Tuple[str, Dict[str, Any]]: Régime détecté et détails (probabilités, importance des features, etc.).
        """
        start_time = datetime.now()
        try:
            InputData(df=df, current_step=current_step)
            if not self.validate_data(df):
                error_details = {
                    "step": current_step,
                    "regime": "defensive",
                    "regime_probs": {"range": 0.0, "trend": 0.0, "defensive": 1.0},
                    "shap_values": {},
                    "neural_regime": "N/A",
                    "confidence_score": 0.0,
                    "error": "Invalid data",
                    "spread": 0.0,
                    "vix_es_correlation": 0.0,
                    "call_iv_atm": 0.0,
                    "option_skew": 0.0,
                    "option_volume": 0.0,
                    "oi_concentration": 0.0,
                    "predicted_vix": 0.0,
                }
                self.log_performance(
                    "detect",
                    0,
                    success=False,
                    error="Invalid data",
                    num_features=self.num_features,
                )
                return "defensive", error_details

            df_enriched = self.precompute_indicators(df)
            scaler = MinMaxScaler()
            df_enriched[self.feature_cols] = scaler.fit_transform(
                df_enriched[self.feature_cols]
            )

            # Extraction des métriques
            atr = (
                float(df_enriched["atr_14"].iloc[current_step])
                if "atr_14" in df_enriched
                else 0.0
            )
            adx = (
                float(df_enriched["adx_14"].iloc[current_step])
                if "adx_14" in df_enriched
                else 0.0
            )
            (
                float(df_enriched["vwap"].iloc[current_step])
                if "vwap" in df_enriched
                else 0.0
            )
            (
                float(df_enriched["close"].iloc[current_step])
                if "close" in df_enriched
                else 0.0
            )
            rsi = (
                float(df_enriched["rsi_14"].iloc[current_step])
                if "rsi_14" in df_enriched
                else 50.0
            )
            bid_ask_ratio = (
                float(df_enriched["bid_ask_ratio"].iloc[current_step])
                if "bid_ask_ratio" in df_enriched
                else 0.5
            )
            bid_ask_ratio_level_2 = (
                float(df_enriched["bid_ask_ratio_level_2"].iloc[current_step])
                if "bid_ask_ratio_level_2" in df_enriched
                else 0.5
            )
            gex = (
                float(df_enriched["gex"].iloc[current_step])
                if "gex" in df_enriched
                else 0.0
            )
            gex_change = (
                float(df_enriched["gex_change"].iloc[current_step])
                if "gex_change" in df_enriched
                else 0.0
            )
            vwap_deviation = (
                float(df_enriched["vwap_deviation"].iloc[current_step])
                if "vwap_deviation" in df_enriched
                else 0.0
            )
            atr_normalized = (
                float(df_enriched["atr_normalized"].iloc[current_step])
                if "atr_normalized" in df_enriched
                else 0.0
            )
            vix_es_correlation = (
                float(df_enriched["vix_es_correlation"].iloc[current_step])
                if "vix_es_correlation" in df_enriched
                else 0.0
            )
            spread = (
                (
                    df_enriched["ask_size_level_1"].iloc[current_step]
                    - df_enriched["bid_size_level_1"].iloc[current_step]
                )
                / df_enriched["close"].iloc[current_step]
                if "ask_size_level_1" in df_enriched
                and df_enriched["close"].iloc[current_step] != 0
                else 0.0
            )
            call_iv_atm = (
                float(df_enriched["call_iv_atm"].iloc[current_step])
                if "call_iv_atm" in df_enriched
                else 0.0
            )
            option_skew = (
                float(df_enriched["option_skew"].iloc[current_step])
                if "option_skew" in df_enriched
                else 0.0
            )
            option_volume = (
                float(df_enriched["option_volume"].iloc[current_step])
                if "option_volume" in df_enriched
                else 0.0
            )
            oi_concentration = (
                float(df_enriched["oi_concentration"].iloc[current_step])
                if "oi_concentration" in df_enriched
                else 0.0
            )

            # Calcul NeuralPipeline pour predicted_vix
            predicted_vix = 0.0
            neural_regime = None
            confidence_score = 0.0
            cache_key = f"{current_step}_{id(df_enriched)}"
            if cache_key in self.neural_cache:
                neural_result = self.neural_cache[cache_key]
                self.neural_cache.move_to_end(cache_key)
            else:

                def run_neural_pipeline():
                    window_start = max(
                        0, current_step - self.neural_pipeline.window_size + 1
                    )
                    window_end = current_step + 1
                    raw_data = (
                        df_enriched[
                            [
                                "timestamp",
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                                "atr_14",
                                "adx_14",
                            ]
                        ]
                        .iloc[window_start:window_end]
                        .fillna(0)
                    )
                    options_data = (
                        df_enriched[
                            [
                                "timestamp",
                                "gex",
                                "oi_peak_call_near",
                                "gamma_wall_call",
                                "call_iv_atm",
                                "option_skew",
                                "option_volume",
                                "oi_concentration",
                            ]
                        ]
                        .iloc[window_start:window_end]
                        .fillna(0)
                    )
                    orderflow_data = (
                        df_enriched[
                            ["timestamp", "bid_size_level_1", "ask_size_level_1"]
                        ]
                        .iloc[window_start:window_end]
                        .fillna(0)
                    )
                    return self.neural_pipeline.run(
                        raw_data, options_data, orderflow_data
                    )

                neural_result = self.with_retries(run_neural_pipeline)
                if neural_result is None:
                    neural_result = {
                        "regime": [self.last_neural_regime],
                        "volatility": [0.0],
                        "confidence": 0.0,
                        "features": np.zeros((1, self.num_features)),
                        "raw_scores": np.array([0.0, 0.0, 0.0]),
                    }
                self.neural_cache[cache_key] = neural_result
                self.prune_cache()

            neural_regime = (
                int(neural_result["regime"][-1]) if neural_result["regime"] else None
            )
            predicted_vix = (
                float(neural_result["volatility"][-1])
                if neural_result["volatility"]
                else vix_es_correlation
            )
            confidence_score = float(neural_result.get("confidence", 0.0))
            df_enriched.loc[current_step, "predicted_vix"] = predicted_vix
            df_enriched.loc[current_step, "neural_regime"] = (
                regime_map.get(neural_regime, "N/A")
                if neural_regime is not None
                else "N/A"
            )

            # Vérification NeuralPipeline
            neural_pipeline_healthy = (
                await self.check_neural_pipeline_health()
                if not self.fast_mode and self.enable_heartbeat
                else False
            )
            thresholds = self.config.get("thresholds", {})
            if (
                not neural_pipeline_healthy
                and spread > thresholds.get("spread_explosion_threshold", 0.05)
                and vix_es_correlation > thresholds.get("vix_peak_threshold", 30.0)
            ):
                alert_msg = f"Mode Ultra-Defensive: spread={spread:.4f}, VIX={vix_es_correlation:.2f}"
                logger.info(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=4)
                send_telegram_alert(alert_msg)
                details = {
                    "step": current_step,
                    "regime": "ultra-defensive",
                    "regime_probs": {"range": 0.0, "trend": 0.0, "defensive": 1.0},
                    "shap_values": {},
                    "neural_regime": self.last_neural_regime,
                    "confidence_score": 0.0,
                    "spread": spread,
                    "vix_es_correlation": vix_es_correlation,
                    "call_iv_atm": call_iv_atm,
                    "option_skew": option_skew,
                    "option_volume": option_volume,
                    "oi_concentration": oi_concentration,
                    "predicted_vix": predicted_vix,
                    "error": None,
                }
                self.log_performance(
                    "detect",
                    (datetime.now() - start_time).total_seconds(),
                    success=True,
                    num_features=self.num_features,
                )
                return "ultra-defensive", details

            # Conditions de volatilité (méthode 1)
            volatility_conditions = {
                "range": (
                    vix_es_correlation < thresholds.get("vix_low_threshold", 15.0)
                    and atr_normalized < thresholds.get("atr_low_threshold", 0.5)
                    and df_enriched["bollinger_width"].iloc[current_step]
                    < thresholds.get("bollinger_width_low", 0.01)
                ),
                "trend": (
                    thresholds.get("vix_low_threshold", 15.0)
                    <= vix_es_correlation
                    <= thresholds.get("vix_high_threshold", 25.0)
                    and adx > thresholds.get("adx_threshold", 25.0)
                    and df_enriched["momentum"].iloc[current_step]
                    > thresholds.get("momentum_threshold", 0.0)
                ),
                "defensive": (
                    vix_es_correlation > thresholds.get("vix_high_threshold", 25.0)
                    or atr_normalized > thresholds.get("atr_high_threshold", 1.0)
                    or predicted_vix > thresholds.get("vix_high_threshold", 25.0)
                ),
            }

            # Conditions d'options (méthode 2)
            option_conditions = {
                "range": (
                    call_iv_atm < thresholds.get("iv_low_threshold", 0.15)
                    and abs(option_skew) < thresholds.get("skew_low_threshold", 0.02)
                    and option_volume < thresholds.get("volume_low_threshold", 1000)
                ),
                "trend": (
                    thresholds.get("iv_low_threshold", 0.15)
                    <= call_iv_atm
                    <= thresholds.get("iv_high_threshold", 0.25)
                    and abs(option_skew) > thresholds.get("skew_high_threshold", 0.05)
                    and option_volume > thresholds.get("volume_high_threshold", 5000)
                ),
                "defensive": (
                    call_iv_atm > thresholds.get("iv_high_threshold", 0.25)
                    or abs(option_skew) > thresholds.get("skew_extreme_threshold", 0.1)
                    or oi_concentration
                    > thresholds.get("oi_concentration_threshold", 0.7)
                ),
            }

            # Régimes hybrides (méthode 11)
            regime_probs = {"range": 0.0, "trend": 0.0, "defensive": 0.0}
            if neural_regime is not None and confidence_score >= thresholds.get(
                "confidence_threshold", 0.7
            ):
                raw_scores = neural_result.get("raw_scores", np.array([0.0, 0.0, 0.0]))
                softmax_scores = np.exp(raw_scores) / np.sum(np.exp(raw_scores))
                regime_probs["trend"] = softmax_scores[0]
                regime_probs["range"] = softmax_scores[1]
                regime_probs["defensive"] = softmax_scores[2]
            else:
                regime_probs["range"] = (
                    0.7
                    if volatility_conditions["range"] and option_conditions["range"]
                    else 0.2
                )
                regime_probs["trend"] = (
                    0.7
                    if volatility_conditions["trend"] and option_conditions["trend"]
                    else 0.2
                )
                regime_probs["defensive"] = (
                    0.7
                    if volatility_conditions["defensive"]
                    or option_conditions["defensive"]
                    else 0.2
                )
                total = sum(regime_probs.values())
                regime_probs = {k: v / total for k, v in regime_probs.items()}

            # Importance des features (méthode 17)
            shap_values = self.calculate_shap_values(df_enriched, current_step)

            # Mise à jour seuils
            self.last_neural_regime = (
                regime_map.get(neural_regime, "defensive")
                if neural_regime is not None
                else self.last_neural_regime
            )
            self.update_thresholds(
                adx, atr, predicted_vix, vix_es_correlation, atr_normalized, call_iv_atm
            )

            # Conditions heuristiques
            atr_threshold = self.config.get("atr_threshold", 1.5)
            adx_threshold = self.config.get("adx_threshold", 25)
            vwap_deviation_threshold = self.config.get("vwap_deviation_threshold", 0.01)
            gex_threshold = self.config.get("gex_threshold", 1000)
            vix_correlation_threshold = self.config.get(
                "vix_correlation_threshold", 0.5
            )
            gex_change_threshold = self.config.get("gex_change_threshold", 0.5)

            trend_conditions = (
                adx > adx_threshold
                and atr > atr_threshold
                and rsi > 50
                and abs(gex) < gex_threshold
                and abs(gex_change) < gex_change_threshold
                and abs(vix_es_correlation) < vix_correlation_threshold
                and abs(bid_ask_ratio_level_2 - 0.5) > 0.1
            )
            range_conditions = (
                abs(vwap_deviation) < vwap_deviation_threshold
                and adx < 20
                and atr_normalized < self.config.get("atr_normalized_threshold", 0.8)
                and abs(gex) > gex_threshold
                and abs(gex_change) > gex_change_threshold
                and bid_ask_ratio > 0.4
                and bid_ask_ratio < 0.6
                and abs(vix_es_correlation) < vix_correlation_threshold
                and abs(bid_ask_ratio_level_2 - 0.5) < 0.1
            )

            # Détection du régime
            regime_map = {0: "trend", 1: "range", 2: "defensive"}
            regime = "defensive"
            if volatility_conditions["range"] and option_conditions["range"]:
                regime = "range"
                regime_probs["range"] = max(regime_probs["range"], 0.7)
            elif volatility_conditions["trend"] and option_conditions["trend"]:
                regime = "trend"
                regime_probs["trend"] = max(regime_probs["trend"], 0.7)
            elif volatility_conditions["defensive"] or option_conditions["defensive"]:
                regime = "defensive"
                regime_probs["defensive"] = max(regime_probs["defensive"], 0.7)
            elif (
                neural_regime is not None
                and self.config.get("use_neural_regime", True)
                and confidence_score >= thresholds.get("confidence_threshold", 0.7)
                and not self.fast_mode
            ):
                regime = regime_map.get(neural_regime, "defensive")
                logger.info(
                    f"Régime neuronal au step {current_step}: {regime}, confidence={confidence_score:.2f}"
                )
            else:
                regime = (
                    "trend"
                    if trend_conditions
                    else "range" if range_conditions else "defensive"
                )
                logger.info(
                    f"Régime heuristique au step {current_step}: {regime}, confidence={confidence_score:.2f}"
                )

            if not self.check_transition_validity(regime):
                alert_msg = f"Transition non valide au step {current_step}: retour à 'defensive'"
                logger.info(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                regime = "defensive"
                regime_probs = {"range": 0.0, "trend": 0.0, "defensive": 1.0}

            self.recent_regimes.append(regime)

            # Détails pour les logs
            details_dict = {
                "step": current_step,
                "regime": regime,
                "regime_probs": regime_probs,
                "shap_values": shap_values,
                "adx_14": adx,
                "atr_14": atr,
                "gex": gex,
                "gex_change": gex_change,
                "bid_ask_ratio": bid_ask_ratio,
                "bid_ask_ratio_level_2": bid_ask_ratio_level_2,
                "vwap_deviation": vwap_deviation,
                "rsi_14": rsi,
                "vix_es_correlation": vix_es_correlation,
                "neural_regime": (
                    regime_map.get(neural_regime, "N/A")
                    if neural_regime is not None
                    else "N/A"
                ),
                "confidence_score": confidence_score,
                "spread": spread,
                "call_iv_atm": call_iv_atm,
                "option_skew": option_skew,
                "option_volume": option_volume,
                "oi_concentration": oi_concentration,
                "predicted_vix": predicted_vix,
                "timestamp": str(datetime.now()),
                "error": None,
            }
            if self.log_history:
                history_df = pd.DataFrame([details_dict])

                def save_history():
                    if not self.history_file.exists():
                        history_df.to_csv(
                            self.history_file, index=False, encoding="utf-8"
                        )
                    else:
                        history_df.to_csv(
                            self.history_file,
                            mode="a",
                            header=False,
                            index=False,
                            encoding="utf-8",
                        )

                self.with_retries(save_history)

            # Snapshot JSON
            snapshot_data = {
                "step": current_step,
                "regime": regime,
                "regime_probs": regime_probs,
                "shap_values": shap_values,
                "predicted_vix": predicted_vix,
                "timestamp": details_dict["timestamp"],
            }
            self.save_snapshot("regime", snapshot_data)

            # Sauvegarde incrémentielle
            perf_data = pd.DataFrame(
                [
                    {
                        "step": current_step,
                        "regime": regime,
                        "confidence_score": confidence_score,
                        "vix_es_correlation": vix_es_correlation,
                        "call_iv_atm": call_iv_atm,
                        "timestamp": details_dict["timestamp"],
                    }
                ]
            )
            self.checkpoint(perf_data, data_type="regime_metrics")
            self.cloud_backup(perf_data, data_type="regime_metrics")

            # Visualisation
            plt.figure(figsize=(8, 6))
            plt.plot(
                df_enriched["vix_es_correlation"],
                label="VIX-ES Correlation",
                color="blue",
            )
            plt.axhline(
                y=thresholds.get("vix_high_threshold", 25.0),
                color="red",
                linestyle="--",
                label="VIX High Threshold",
            )
            plt.title(
                f"Régime: {regime} (Probs: Range={regime_probs['range']:.2f}, Trend={regime_probs['trend']:.2f}, "
                f"Def={regime_probs['defensive']:.2f}, VIX: {vix_es_correlation:.2f}, IV: {call_iv_atm:.2f})"
            )
            plt.xlabel("Time")
            plt.ylabel("VIX-ES Correlation")
            plt.legend()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(self.figure_dir / f"vix_{timestamp}.png")
            plt.close()

            success_msg = (
                f"Régime déterminé au step {current_step}: {regime}, VIX={vix_es_correlation:.2f}, "
                f"IV={call_iv_atm:.2f}, Skew={option_skew:.2f}, Volume={option_volume:.0f}, Probs={regime_probs}"
            )
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "detect", latency, success=True, num_features=self.num_features
            )
            return regime, details_dict
        except Exception as e:
            error_msg = f"Erreur detect: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            error_details = {
                "step": current_step,
                "regime": "defensive",
                "regime_probs": {"range": 0.0, "trend": 0.0, "defensive": 1.0},
                "shap_values": {},
                "neural_regime": self.last_neural_regime,
                "confidence_score": 0.0,
                "spread": spread,
                "vix_es_correlation": vix_es_correlation,
                "call_iv_atm": call_iv_atm,
                "option_skew": option_skew,
                "option_volume": option_volume,
                "oi_concentration": oi_concentration,
                "predicted_vix": predicted_vix,
                "error": str(e),
            }
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "detect",
                latency,
                success=False,
                error=str(e),
                num_features=self.num_features,
            )
            return "defensive", error_details

    async def detect_market_regime_vectorized(
        self, df: pd.DataFrame, current_step: int
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Détection vectorisée pour compatibilité externe.

        Args:
            df (pd.DataFrame): Données d’entrée.
            current_step (int): Étape actuelle.

        Returns:
            Tuple[str, Dict[str, Any]]: Régime détecté et détails.
        """
        return await self.detect(df, current_step)


if __name__ == "__main__":

    async def main():
        try:
            data = pd.DataFrame(
                {
                    "timestamp": pd.date_range(
                        "2025-05-13 09:00", periods=100, freq="1min"
                    ),
                    **{
                        f"feature_{i}": np.random.uniform(0, 1, 100) for i in range(350)
                    },
                    "open": np.random.uniform(5000, 5100, 100),
                    "high": np.random.uniform(5100, 5200, 100),
                    "low": np.random.uniform(4900, 5000, 100),
                    "close": np.random.uniform(5000, 5100, 100),
                    "volume": np.random.randint(1000, 10000, 100),
                    "call_iv_atm": [0.1] * 50 + [0.2] * 50,
                    "put_iv_atm": [0.1] * 50 + [0.15] * 50,
                    "call_volume": [250] * 50 + [1000] * 50,
                    "put_volume": [250] * 50 + [1000] * 50,
                    "oi_peak_call_near": [1000] * 50 + [5000] * 50,
                    "oi_peak_put_near": [1000] * 50 + [1000] * 50,
                    "bid_size_level_1": np.random.randint(100, 500, 100),
                    "ask_size_level_1": np.random.randint(100, 500, 100),
                    "gex": np.random.uniform(-1000, 1000, 100),
                    "gex_change": np.random.uniform(-0.5, 0.5, 100),
                    "vwap_deviation": np.random.uniform(-0.01, 0.01, 100),
                    "atr_normalized": np.random.uniform(0.1, 0.9, 100),
                    "vix_es_correlation": np.random.uniform(-0.5, 0.5, 100),
                    "bid_ask_ratio": np.random.uniform(0.4, 0.6, 100),
                    "bid_ask_ratio_level_2": np.random.uniform(0.4, 0.6, 100),
                }
            )
            detector = MarketRegimeDetector(policy_type="mlp", training_mode=True)
            regime, details = await detector.detect(data, 50)
            print(f"Régime détecté au step 50: {regime}, détails: {details}")
        except Exception as e:
            logger.error(f"Erreur test: {str(e)}\n{traceback.format_exc()}")
            raise

    asyncio.run(main())
