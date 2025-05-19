# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/model/utils/prediction_aggregator.py
# Rôle : Agrège les prédictions de SAC, PPO, DDPG avec une logique avancée d’ensemble learning pour MIA_IA_SYSTEM_v2_2025.
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, matplotlib>=3.7.0,<4.0.0, psutil>=5.9.8,<6.0.0,
#   boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0, pyyaml>=6.0.0,<7.0.0, signal
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - Prédictions (actions, rewards) de SAC/PPO/DDPG
# - Données brutes (pd.DataFrame avec 350 features pour l’entraînement ou 150 SHAP features pour l’inférence)
# - Configuration via algo_config.yaml
#
# Outputs :
# - Action finale agrégée
# - Logs dans data/logs/prediction_aggregator.log
# - Logs de performance dans data/logs/prediction_aggregator_performance.csv
# - Snapshots JSON compressés dans data/cache/prediction_aggregator/<market>/*.json.gz
# - Sauvegardes incrémentielles dans data/checkpoints/prediction_aggregator/<market>/*.json.gz
# - Visualisations dans data/figures/prediction_aggregator/<market>/
#
# Notes :
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence) définies dans feature_sets.yaml.
# - Utilise IQFeed exclusivement via data_provider.py pour les données d’entrée.
# - Supprime toute référence à 320 features, 81 features, et obs_t.
# - Implémente retries (max 3, délai 2^attempt secondes) pour les calculs critiques.
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des données.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.
# - Tests unitaires disponibles dans tests/test_prediction_aggregator.py.

import gzip
import json
import signal
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from loguru import logger

from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import get_config
from src.utils.telegram_alert import send_telegram_alert

# Configuration logging
logger.remove()
BASE_DIR = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
LOG_DIR = BASE_DIR / "data" / "logs"
CACHE_DIR = BASE_DIR / "data" / "cache" / "prediction_aggregator"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints" / "prediction_aggregator"
FIGURE_DIR = BASE_DIR / "data" / "figures" / "prediction_aggregator"
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "prediction_aggregator.log",
    rotation="10 MB",
    level="INFO",
    encoding="utf-8",
)
logger.add(lambda msg: print(msg, end=""), level="INFO")

# Constantes
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Secondes, exponentiel
PERF_LOG_PATH = LOG_DIR / "prediction_aggregator_performance.csv"
MAX_CACHE_SIZE = 1000
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB

# Cache global pour les résultats de aggregate_predictions
prediction_cache = OrderedDict()


class PredictionAggregator:
    """Agrège les prédictions de SAC, PPO, DDPG avec une logique d’ensemble learning."""

    def __init__(
        self,
        config_path: str = str(BASE_DIR / "config" / "algo_config.yaml"),
        market: str = "ES",
    ):
        """
        Initialise l’agrégateur de prédictions.

        Args:
            config_path (str): Chemin vers le fichier de configuration.
            market (str): Marché (ex. : ES, MNQ).
        """
        self.market = market
        self.alert_manager = AlertManager()
        self.config = get_config(config_path).get("prediction_aggregator", {})
        self.perf_log = PERF_LOG_PATH
        self.snapshot_dir = CACHE_DIR / market
        self.figure_dir = FIGURE_DIR / market
        self.checkpoint_dir = CHECKPOINT_DIR / market
        self.snapshot_dir.mkdir(exist_ok=True)
        self.figure_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        signal.signal(signal.SIGINT, self.handle_sigint)
        logger.info(f"PredictionAggregator initialisé pour {market}")
        self.log_performance("init", 0, success=True)

    def handle_sigint(self, signum, frame):
        """Gère l'arrêt propre sur SIGINT."""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "status": "SIGINT",
            "market": self.market,
        }
        snapshot_path = self.snapshot_dir / f'sigint_{snapshot["timestamp"]}.json.gz'
        try:
            with gzip.open(snapshot_path, "wt", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=4)
            success_msg = (
                f"Arrêt propre sur SIGINT, snapshot sauvegardé pour {self.market}"
            )
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot SIGINT pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
        exit(0)

    def log_performance(
        self,
        operation: str,
        latency: float,
        success: bool = True,
        error: str = None,
        **kwargs,
    ) -> None:
        """
        Enregistre les performances (latence, mémoire, CPU) dans prediction_aggregator_performance.csv.

        Args:
            operation (str): Nom de l’opération (ex. : aggregate_predictions).
            latency (float): Temps d’exécution en secondes.
            success (bool): Indique si l’opération a réussi.
            error (str, optional): Message d’erreur si applicable.
            **kwargs: Paramètres supplémentaires.
        """
        cache_key = f"{self.market}_{operation}_{hash(str(latency))}_{hash(str(error))}"
        if cache_key in prediction_cache:
            return
        while len(prediction_cache) > MAX_CACHE_SIZE:
            prediction_cache.popitem(last=False)

        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
            cpu_percent = psutil.cpu_percent()
            confidence_drop_rate = 1.0 if success else 0.0  # Simplifié pour Phase 8
            if memory_usage > 1024:
                alert_msg = f"ALERTE: Usage mémoire élevé ({memory_usage:.2f} MB) pour {self.market}"
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
                "confidence_drop_rate": confidence_drop_rate,
                "market": self.market,
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

            self.with_retries(save_log)
            logger.info(
                f"Performance journalisée pour {operation}. CPU: {cpu_percent}%"
            )
            self.save_snapshot("log_performance", log_entry)
            prediction_cache[cache_key] = True
        except Exception as e:
            error_msg = f"Erreur journalisation performance pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)

    def save_snapshot(
        self, snapshot_type: str, data: Dict, compress: bool = True
    ) -> None:
        """
        Sauvegarde un instantané JSON des résultats, compressé avec gzip.

        Args:
            snapshot_type (str): Type de snapshot (ex. : aggregate_predictions).
            data (Dict): Données à sauvegarder.
            compress (bool): Compresser avec gzip (défaut : True).
        """
        start_time = time.time()
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = {
                "timestamp": timestamp,
                "type": snapshot_type,
                "market": self.market,
                "data": data,
            }
            snapshot_path = (
                self.snapshot_dir / f"snapshot_{snapshot_type}_{timestamp}.json"
            )

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
                    f"Snapshot size {file_size:.2f} MB exceeds 1 MB pour {self.market}"
                )
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
            latency = time.time() - start_time
            success_msg = (
                f"Snapshot {snapshot_type} sauvegardé pour {self.market}: {save_path}"
            )
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "save_snapshot", latency, success=True, snapshot_size_mb=file_size
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot {snapshot_type} pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance("save_snapshot", 0, success=False, error=str(e))

    def checkpoint(
        self, data: pd.DataFrame, data_type: str = "prediction_aggregator_state"
    ) -> None:
        """
        Sauvegarde incrémentielle des données toutes les 5 minutes avec versionnage (5 versions).

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : prediction_aggregator_state).
        """
        try:
            start_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_data = {
                "timestamp": timestamp,
                "num_rows": len(data),
                "columns": list(data.columns),
                "data_type": data_type,
                "market": self.market,
            }
            checkpoint_path = (
                self.checkpoint_dir
                / f"prediction_aggregator_{data_type}_{timestamp}.json.gz"
            )
            checkpoint_versions = []

            def write_checkpoint():
                with gzip.open(checkpoint_path, "wt", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, indent=4)
                data.to_csv(
                    checkpoint_path.with_suffix(".csv"), index=False, encoding="utf-8"
                )

            self.with_retries(write_checkpoint)
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
            success_msg = f"Checkpoint sauvegardé pour {self.market}: {checkpoint_path}"
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
            error_msg = f"Erreur sauvegarde checkpoint pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "checkpoint", 0, success=False, error=str(e), data_type=data_type
            )

    def cloud_backup(
        self, data: pd.DataFrame, data_type: str = "prediction_aggregator_state"
    ) -> None:
        """
        Sauvegarde distribuée des données vers AWS S3 toutes les 15 minutes.

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : prediction_aggregator_state).
        """
        try:
            start_time = time.time()
            config = get_config(str(BASE_DIR / "config/es_config.yaml"))
            if not config.get("s3_bucket"):
                warning_msg = f"S3 bucket non configuré, sauvegarde cloud ignorée pour {self.market}"
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{config['s3_prefix']}prediction_aggregator_{data_type}_{self.market}_{timestamp}.csv.gz"
            temp_path = self.checkpoint_dir / f"temp_s3_{timestamp}.csv.gz"
            temp_path.parent.mkdir(exist_ok=True)

            def write_temp():
                data.to_csv(
                    temp_path, compression="gzip", index=False, encoding="utf-8"
                )

            self.with_retries(write_temp)
            s3_client = boto3.client("s3")

            def upload_s3():
                s3_client.upload_file(str(temp_path), config["s3_bucket"], backup_path)

            self.with_retries(upload_s3)
            temp_path.unlink()
            latency = time.time() - start_time
            success_msg = (
                f"Sauvegarde cloud S3 effectuée pour {self.market}: {backup_path}"
            )
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
            error_msg = f"Erreur sauvegarde cloud S3 pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "cloud_backup", 0, success=False, error=str(e), data_type=data_type
            )

    def with_retries(
        self,
        func: callable,
        max_attempts: int = MAX_RETRIES,
        delay_base: float = RETRY_DELAY_BASE,
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
                )
                return result
            except Exception as e:
                if attempt == max_attempts - 1:
                    error_msg = f"Échec après {max_attempts} tentatives pour {self.market}: {str(e)}\n{traceback.format_exc()}"
                    logger.error(error_msg)
                    self.alert_manager.send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    self.log_performance(
                        f"retry_attempt_{attempt+1}",
                        time.time() - start_time,
                        success=False,
                        error=str(e),
                        attempt_number=attempt + 1,
                    )
                    return None
                delay = delay_base**attempt
                warning_msg = f"Tentative {attempt+1} échouée pour {self.market}, retry après {delay}s"
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                time.sleep(delay)

    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Valide les données d’entrée avec confidence_drop_rate.

        Args:
            data (pd.DataFrame): Données à valider.

        Raises:
            ValueError: Si les données sont invalides.
        """
        start_time = time.time()
        try:
            feature_sets_path = BASE_DIR / "config/feature_sets.yaml"
            features_config = get_config(feature_sets_path)
            expected_cols = (
                features_config.get("training", {}).get("features", [])[:350]
                if data.shape[1] >= 350
                else features_config.get("inference", {}).get("shap_features", [])[:150]
            )
            expected_len = 350 if data.shape[1] >= 350 else 150
            if len(expected_cols) != expected_len:
                raise ValueError(
                    f"Nombre de features incorrect pour {self.market}: {len(expected_cols)} au lieu de {expected_len}"
                )

            missing_cols = [col for col in expected_cols if col not in data.columns]
            null_count = data[expected_cols].isnull().sum().sum()
            confidence_drop_rate = (
                null_count / (len(data) * len(expected_cols))
                if (len(data) * len(expected_cols)) > 0
                else 0.0
            )
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé pour {self.market}: {confidence_drop_rate:.2f} ({null_count} valeurs nulles)"
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
            if missing_cols:
                warning_msg = f"Colonnes manquantes pour {self.market} ({'SHAP' if expected_len == 150 else 'full'}): {missing_cols}"
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=2)
                send_telegram_alert(warning_msg)
                for col in missing_cols:
                    data[col] = 0.0

            critical_cols = [
                "bid_size_level_1",
                "ask_size_level_1",
                "trade_frequency_1s",
                f"spread_avg_1min_{self.market.lower()}",
                "close",
            ]
            for col in critical_cols:
                if col in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        raise ValueError(
                            f"Colonne {col} n'est pas numérique pour {self.market}: {data[col].dtype}"
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
                        f"spread_avg_1min_{self.market.lower()}",
                    ]:
                        if (data[col] <= 0).any():
                            warning_msg = f"Valeurs non positives dans {col} pour {self.market}, corrigées à 1e-6"
                            logger.warning(warning_msg)
                            self.alert_manager.send_alert(warning_msg, priority=4)
                            data[col] = data[col].clip(lower=1e-6)

            self.save_snapshot(
                "validate_data",
                {
                    "num_columns": len(data.columns),
                    "missing_columns": missing_cols,
                    "confidence_drop_rate": confidence_drop_rate,
                },
            )
            self.log_performance(
                "validate_data", time.time() - start_time, success=True
            )
        except Exception as e:
            error_msg = f"Erreur validation données pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance(
                "validate_data", time.time() - start_time, success=False, error=str(e)
            )
            raise

    def calculate_weights(self, rewards: List[float], regime: str) -> List[float]:
        """
        Calcule les poids pour l’ensemble learning basé sur les récompenses.

        Args:
            rewards (List[float]): Récompenses des modèles SAC, PPO, DDPG.
            regime (str): Régime de marché ("trend", "range", "defensive").

        Returns:
            List[float]: Poids pour chaque modèle.
        """
        start_time = time.time()
        try:
            valid_regimes = {"trend", "range", "defensive"}
            if regime.lower() not in valid_regimes:
                error_msg = f"Régime invalide pour {self.market}: {regime}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                self.log_performance(
                    "calculate_weights",
                    time.time() - start_time,
                    success=False,
                    error="Régime invalide",
                )
                return [1 / 3] * 3
            if (
                not isinstance(rewards, list)
                or len(rewards) != 3
                or not all(isinstance(r, (int, float)) for r in rewards)
            ):
                error_msg = (
                    f"Nombre ou type incorrect de récompenses pour {self.market}"
                )
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                self.log_performance(
                    "calculate_weights",
                    time.time() - start_time,
                    success=False,
                    error="Récompenses invalides",
                )
                return [1 / 3] * 3

            def compute_weights():
                rewards_array = np.array(rewards)
                reward_sum = np.sum(np.abs(rewards_array))
                weights = (
                    rewards_array / reward_sum
                    if reward_sum != 0
                    else np.array([1 / 3] * 3)
                )
                return weights

            weights = self.with_retries(compute_weights)
            if weights is None:
                error_msg = f"Échec calcul des poids pour {self.market}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                self.log_performance(
                    "calculate_weights",
                    time.time() - start_time,
                    success=False,
                    error="Échec calcul poids",
                )
                return [1 / 3] * 3

            regime_weights = {
                "range": [0.4, 0.3, 0.3],  # SAC privilégié pour range
                "trend": [0.3, 0.4, 0.3],  # PPO privilégié pour trend
                "defensive": [0.3, 0.3, 0.4],  # DDPG privilégié pour défensif
            }
            weights = weights * regime_weights.get(regime.lower(), [1 / 3] * 3)
            weights = (
                weights / np.sum(weights)
                if np.sum(weights) != 0
                else np.array([1 / 3] * 3)
            )
            weights = weights.tolist()

            self.save_snapshot(
                "calculate_weights",
                {"rewards": rewards, "regime": regime, "weights": weights},
            )
            self.log_performance(
                "calculate_weights",
                time.time() - start_time,
                success=True,
                regime=regime,
            )
            return weights

        except Exception as e:
            error_msg = f"Erreur calcul poids pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "calculate_weights",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            return [1 / 3] * 3

    def aggregate_predictions(
        self,
        actions: List[float],
        rewards: List[float],
        regime: str,
        raw_data: Optional[pd.DataFrame] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Agrège les prédictions avec pondération.

        Args:
            actions (List[float]): Actions prédites par SAC, PPO, DDPG.
            rewards (List[float]): Récompenses associées aux prédictions.
            regime (str): Régime de marché ("trend", "range", "defensive").
            raw_data (Optional[pd.DataFrame]): Données brutes pour validation (350 features pour entraînement, 150 pour inférence).

        Returns:
            Tuple[float, Dict[str, Any]]: Action finale agrégée et détails.
        """
        start_time = time.time()
        try:
            cache_key = (
                f"{self.market}_{hash(str(actions))}_{hash(str(rewards))}_{regime}"
            )
            if cache_key in prediction_cache:
                result = prediction_cache[cache_key]
                prediction_cache.move_to_end(cache_key)
                return result
            while len(prediction_cache) > MAX_CACHE_SIZE:
                prediction_cache.popitem(last=False)

            valid_regimes = {"trend", "range", "defensive"}
            if regime.lower() not in valid_regimes:
                error_msg = f"Régime invalide pour {self.market}: {regime}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                self.log_performance(
                    "aggregate_predictions",
                    time.time() - start_time,
                    success=False,
                    error="Régime invalide",
                )
                return 0.0, {"error": f"Régime invalide: {regime}"}
            if (
                not isinstance(actions, list)
                or len(actions) != 3
                or not all(isinstance(a, (int, float)) for a in actions)
            ):
                error_msg = f"Nombre ou type incorrect d’actions pour {self.market}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                self.log_performance(
                    "aggregate_predictions",
                    time.time() - start_time,
                    success=False,
                    error="Actions invalides",
                )
                return 0.0, {"error": "Nombre ou type incorrect d’actions"}
            if (
                not isinstance(rewards, list)
                or len(rewards) != 3
                or not all(isinstance(r, (int, float)) for r in rewards)
            ):
                error_msg = (
                    f"Nombre ou type incorrect de récompenses pour {self.market}"
                )
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                self.log_performance(
                    "aggregate_predictions",
                    time.time() - start_time,
                    success=False,
                    error="Récompenses invalides",
                )
                return 0.0, {"error": "Nombre ou type incorrect de récompenses"}

            if raw_data is not None:
                self._validate_data(raw_data)

            weights = self.calculate_weights(rewards, regime)
            final_action = float(np.sum(np.array(actions) * np.array(weights)))

            max_action = self.config.get("max_action", 1.0)
            if abs(final_action) > max_action:
                warning_msg = f"Action finale hors plage pour {self.market}: {final_action:.2f}, limitée à [-{max_action}, {max_action}]"
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                final_action = np.clip(final_action, -max_action, max_action)

            action_variance = float(np.var(actions)) if actions else 0.0
            max_variance = self.config.get("max_action_variance", 1.0)
            confidence = (
                1.0 - min(action_variance / max_variance, 1.0)
                if max_variance > 0
                else 1.0
            )

            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "regime": regime,
                "actions": actions,
                "rewards": rewards,
                "weights": weights,
                "final_action": final_action,
                "action_variance": action_variance,
                "confidence": confidence,
                "data_dim": raw_data.shape[1] if raw_data is not None else None,
            }
            self.save_snapshot(f"aggregate_{regime}", snapshot)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(["SAC", "PPO", "DDPG"], actions, alpha=0.4, label="Actions")
            ax.axhline(
                final_action,
                color="r",
                linestyle="--",
                label=f"Final Action: {final_action:.2f}",
            )
            ax.set_title(
                f"Agrégation: Régime {regime}, Variance: {action_variance:.2f}, Confiance: {confidence:.2f}"
            )
            ax.set_ylabel("Action")
            ax.legend()
            ax.grid(True)
            plt.savefig(
                self.figure_dir / f'aggregate_{regime}_{snapshot["timestamp"]}.png',
                bbox_inches="tight",
                optimize=True,
            )
            plt.close()

            result = {
                "weights": weights,
                "actions": actions,
                "action_variance": action_variance,
                "confidence": confidence,
            }
            self.checkpoint(pd.DataFrame([snapshot]), data_type="aggregate_predictions")
            self.cloud_backup(
                pd.DataFrame([snapshot]), data_type="aggregate_predictions"
            )

            latency = time.time() - start_time
            success_msg = f"Agrégation réussie pour {self.market}: régime={regime}, action={final_action:.2f}, weights={weights}, variance={action_variance:.2f}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "aggregate_predictions",
                latency,
                success=True,
                final_action=final_action,
                action_variance=action_variance,
            )
            prediction_cache[cache_key] = (final_action, result)
            return final_action, result

        except Exception as e:
            error_msg = f"Erreur agrégation pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "aggregate_predictions",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            return 0.0, {"error": str(e)}


if __name__ == "__main__":
    aggregator = PredictionAggregator()
    actions = [0.5, -0.3, 0.2]
    rewards = [1.0, 0.8, 0.9]
    regime = "trend"
    feature_cols = [f"feature_{i}" for i in range(350)]
    data = pd.DataFrame(
        {
            "bid_size_level_1": [100],
            "ask_size_level_1": [120],
            "trade_frequency_1s": [8],
            "spread_avg_1min_es": [0.3],
            "close": [5100],
            **{col: [np.random.uniform(0, 1)] for col in feature_cols},
        }
    )
    final_action, details = aggregator.aggregate_predictions(
        actions, rewards, regime, data
    )
    print(f"Action finale: {final_action}, Détails: {details}")
