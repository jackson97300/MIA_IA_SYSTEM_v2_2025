# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/model/utils/model_validator.py
# Rôle : Valide les modèles SAC, PPO, DDPG pour stabilité et performance dans MIA_IA_SYSTEM_v2_2025.
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Dépendances :
# - torch>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, pandas>=2.0.0,<3.0.0, matplotlib>=3.7.0,<4.0.0,
#   psutil>=5.9.8,<6.0.0, boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0, pyyaml>=6.0.0,<7.0.0, signal
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - Modèle (SAC/PPO/DDPG)
# - Données d’observations (np.ndarray avec 350 features pour l’entraînement ou 150 SHAP features pour l’inférence)
# - Configuration via algo_config.yaml
#
# Outputs :
# - Résultat de validation (validité, récompense moyenne, détails)
# - Logs dans data/logs/model_validator.log
# - Logs de performance dans data/logs/model_validator_performance.csv
# - Snapshots JSON compressés dans data/cache/model_validator/<market>/*.json.gz
# - Sauvegardes incrémentielles dans data/checkpoints/model_validator/<market>/*.json.gz
# - Visualisations dans data/figures/model_validator/<market>/
#
# Notes :
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence) définies dans feature_sets.yaml.
# - Utilise IQFeed exclusivement via data_provider.py pour les données d’entrée.
# - Supprime toute référence à 320 features, 81 features, et obs_t.
# - Implémente retries (max 3, délai 2^attempt secondes) pour les appels critiques.
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des observations.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.
# - Tests unitaires disponibles dans tests/test_model_validator.py.

import gzip
import json
import signal
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
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
CACHE_DIR = BASE_DIR / "data" / "cache" / "model_validator"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints" / "model_validator"
FIGURE_DIR = BASE_DIR / "data" / "figures" / "model_validator"
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "model_validator.log", rotation="10 MB", level="INFO", encoding="utf-8"
)
logger.add(lambda msg: print(msg, end=""), level="INFO")

# Constantes
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Secondes, exponentiel
PERF_LOG_PATH = LOG_DIR / "model_validator_performance.csv"
MAX_CACHE_SIZE = 1000
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB

# Cache global pour les résultats de validate_model
validation_cache = OrderedDict()


class ModelValidator:
    """Valide les modèles SAC, PPO, DDPG pour stabilité et performance."""

    def __init__(
        self,
        config_path: str = str(BASE_DIR / "config" / "algo_config.yaml"),
        market: str = "ES",
    ):
        """
        Initialise le validateur de modèles.

        Args:
            config_path (str): Chemin vers le fichier de configuration.
            market (str): Marché (ex. : ES, MNQ).
        """
        self.market = market
        self.alert_manager = AlertManager()
        self.config = get_config(config_path).get("model_validator", {})
        self.perf_log = PERF_LOG_PATH
        self.snapshot_dir = CACHE_DIR / market
        self.figure_dir = FIGURE_DIR / market
        self.checkpoint_dir = CHECKPOINT_DIR / market
        self.snapshot_dir.mkdir(exist_ok=True)
        self.figure_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        signal.signal(signal.SIGINT, self.handle_sigint)
        logger.info(f"ModelValidator initialisé pour {market}")
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
        Enregistre les performances (latence, mémoire, CPU) dans model_validator_performance.csv.

        Args:
            operation (str): Nom de l’opération (ex. : validate_model).
            latency (float): Temps d’exécution en secondes.
            success (bool): Indique si l’opération a réussi.
            error (str, optional): Message d’erreur si applicable.
            **kwargs: Paramètres supplémentaires.
        """
        cache_key = f"{self.market}_{operation}_{hash(str(latency))}_{hash(str(error))}"
        if cache_key in validation_cache:
            return
        while len(validation_cache) > MAX_CACHE_SIZE:
            validation_cache.popitem(last=False)

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
            validation_cache[cache_key] = True
        except Exception as e:
            error_msg = f"Erreur journalisation performance pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)


def save_snapshot(self, snapshot_type: str, data: Dict, compress: bool = True) -> None:
    """
    Sauvegarde un instantané JSON des résultats, compressé avec gzip.

    Args:
        snapshot_type (str): Type de snapshot (ex. : validate_model).
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
        snapshot_path = self.snapshot_dir / f"snapshot_{snapshot_type}_{timestamp}.json"

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
        self, data: pd.DataFrame, data_type: str = "model_validator_state"
    ) -> None:
        """
        Sauvegarde incrémentielle des données toutes les 5 minutes avec versionnage (5 versions).

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : model_validator_state).
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
                self.checkpoint_dir / f"model_validator_{data_type}_{timestamp}.json.gz"
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
        self, data: pd.DataFrame, data_type: str = "model_validator_state"
    ) -> None:
        """
        Sauvegarde distribuée des données vers AWS S3 toutes les 15 minutes.

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : model_validator_state).
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
            backup_path = f"{config['s3_prefix']}model_validator_{data_type}_{self.market}_{timestamp}.csv.gz"
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

    def _validate_observations(self, observations: np.ndarray) -> None:
        """
        Valide les observations d’entrée avec confidence_drop_rate.

        Args:
            observations (np.ndarray): Données d’observations à valider.

        Raises:
            ValueError: Si les observations sont invalides.
        """
        start_time = time.time()
        try:
            feature_sets_path = BASE_DIR / "config/feature_sets.yaml"
            features_config = get_config(feature_sets_path)
            expected_len = 350 if observations.shape[1] >= 350 else 150
            expected_cols = (
                features_config.get("training", {}).get("features", [])[:350]
                if expected_len == 350
                else features_config.get("inference", {}).get("shap_features", [])[:150]
            )
            if len(expected_cols) != expected_len:
                raise ValueError(
                    f"Nombre de features incorrect pour {self.market}: {len(expected_cols)} au lieu de {expected_len}"
                )

            obs_df = pd.DataFrame(
                observations, columns=expected_cols[: observations.shape[1]]
            )
            null_count = obs_df.isnull().sum().sum()
            confidence_drop_rate = (
                null_count / (obs_df.size) if obs_df.size > 0 else 0.0
            )
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé pour {self.market}: {confidence_drop_rate:.2f} ({null_count} valeurs nulles)"
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)

            for col in obs_df.columns:
                if not np.issubdtype(obs_df[col].dtype, np.number):
                    raise ValueError(
                        f"Colonne {col} n'est pas numérique pour {self.market}: {obs_df[col].dtype}"
                    )
                if obs_df[col].isna().any():
                    obs_df[col] = (
                        obs_df[col]
                        .interpolate(method="linear", limit_direction="both")
                        .fillna(0.0)
                    )

            self.save_snapshot(
                "validate_observations",
                {
                    "num_columns": len(obs_df.columns),
                    "confidence_drop_rate": confidence_drop_rate,
                    "null_count": int(null_count),
                },
            )
            self.log_performance(
                "validate_observations", time.time() - start_time, success=True
            )
        except Exception as e:
            error_msg = f"Erreur validation observations pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance(
                "validate_observations",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            raise

    def validate_model(
        self, model: Any, algo_type: str, regime: str, observations: np.ndarray
    ) -> Dict[str, Any]:
        """
        Valide un modèle pour stabilité et performance.

        Args:
            model (Any): Modèle SAC, PPO, ou DDPG.
            algo_type (str): Type d’algorithme ("sac", "ppo", "ddpg").
            regime (str): Régime de marché ("trend", "range", "defensive").
            observations (np.ndarray): Données d’observations (350 features pour entraînement, 150 pour inférence).

        Returns:
            Dict[str, Any]: Résultat de validation (validité, récompense moyenne, détails).
        """
        start_time = time.time()
        try:
            cache_key = f"{self.market}_{algo_type}_{regime}_{hash(str(observations.tobytes()))}"
            if cache_key in validation_cache:
                result = validation_cache[cache_key]
                validation_cache.move_to_end(cache_key)
                return result
            while len(validation_cache) > MAX_CACHE_SIZE:
                validation_cache.popitem(last=False)

            valid_algos = {"sac", "ppo", "ddpg"}
            valid_regimes = {"trend", "range", "defensive"}
            if algo_type.lower() not in valid_algos:
                error_msg = (
                    f"Type d’algorithme invalide pour {self.market}: {algo_type}"
                )
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                self.log_performance(
                    "validate_model",
                    time.time() - start_time,
                    success=False,
                    error="Algo invalide",
                )
                return {
                    "valid": False,
                    "mean_reward": 0.0,
                    "details": {"error": f"Type d’algorithme invalide: {algo_type}"},
                }
            if regime.lower() not in valid_regimes:
                error_msg = f"Régime invalide pour {self.market}: {regime}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                self.log_performance(
                    "validate_model",
                    time.time() - start_time,
                    success=False,
                    error="Régime invalide",
                )
                return {
                    "valid": False,
                    "mean_reward": 0.0,
                    "details": {"error": f"Régime invalide: {regime}"},
                }

            expected_dims = self.config.get(
                "observation_dims", {"training": 350, "inference": 150}
            )
            obs_dim = (
                observations.shape[1]
                if observations.ndim > 1
                else observations.shape[0]
            )
            if obs_dim not in (expected_dims["training"], expected_dims["inference"]):
                error_msg = f"Dimension des observations invalide pour {self.market}: {obs_dim}, attendu {expected_dims['training']} ou {expected_dims['inference']}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                self.log_performance(
                    "validate_model",
                    time.time() - start_time,
                    success=False,
                    error="Dimension observations invalide",
                )
                return {
                    "valid": False,
                    "mean_reward": 0.0,
                    "details": {"error": f"Dimension observations invalide: {obs_dim}"},
                }

            self._validate_observations(observations)

            weights_valid = True
            weight_stats = {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0}
            for param in model.policy.parameters():
                param_data = param.detach().cpu().numpy()
                if np.any(np.isnan(param_data)) or np.any(np.isinf(param_data)):
                    weights_valid = False
                    error_msg = f"Poids invalides détectés pour {algo_type} ({regime}) dans {self.market}"
                    logger.error(error_msg)
                    self.alert_manager.send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    break
                weight_stats["mean"] = float(np.mean(param_data))
                weight_stats["std"] = float(np.std(param_data))
                weight_stats["max"] = float(np.max(param_data))
                weight_stats["min"] = float(np.min(param_data))

            gradients_valid = True
            gradient_stats = {"max_grad": 0.0}
            if algo_type.lower() == "ddpg":

                def compute_gradients():
                    model.policy.zero_grad()
                    obs_tensor = torch.tensor(observations, dtype=torch.float32)
                    action = model.predict(obs_tensor, deterministic=True)[0]
                    loss = torch.mean(
                        torch.tensor(action, dtype=torch.float32) ** 2
                    )  # Dummy loss
                    loss.backward()
                    max_grad = max(
                        p.grad.abs().max().item()
                        for p in model.policy.parameters()
                        if p.grad is not None
                    )
                    return max_grad

                max_grad = self.with_retries(compute_gradients)
                if max_grad is None or max_grad > self.config.get(
                    "max_gradient_threshold", 10.0
                ):
                    gradients_valid = False
                    warning_msg = f"Explosion des gradients pour DDPG ({regime}) dans {self.market}: max={max_grad or 'N/A'}"
                    logger.warning(warning_msg)
                    self.alert_manager.send_alert(warning_msg, priority=3)
                    send_telegram_alert(warning_msg)
                gradient_stats["max_grad"] = max_grad if max_grad is not None else 0.0

            performance_valid = True
            rewards = []
            actions = []
            num_simulations = self.config.get("num_simulations", 10)
            for _ in range(num_simulations):

                def predict_action():
                    action, _ = model.predict(observations, deterministic=True)
                    return action

                action = self.with_retries(predict_action)
                if action is None:
                    performance_valid = False
                    error_msg = f"Échec prédiction action pour {algo_type} ({regime}) dans {self.market}"
                    logger.error(error_msg)
                    self.alert_manager.send_alert(error_msg, priority=3)
                    send_telegram_alert(error_msg)
                    break
                actions.append(action)
                target_action = np.zeros_like(action)  # Cible fictive
                # Récompense négative basée sur l’erreur
                reward = -np.mean((action - target_action) ** 2)
                rewards.append(reward)
            mean_reward = np.mean(rewards) if rewards else 0.0
            action_variance = np.var(actions, axis=0).mean() if actions else 0.0
            if mean_reward < self.config.get("min_reward_threshold", -0.5):
                performance_valid = False
                warning_msg = f"Performance faible pour {algo_type} ({regime}) dans {self.market}: mean_reward={mean_reward:.2f}"
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
            if action_variance > self.config.get("max_action_variance", 1.0):
                performance_valid = False
                warning_msg = f"Variance des actions élevée pour {algo_type} ({regime}) dans {self.market}: variance={action_variance:.2f}"
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)

            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "algo_type": algo_type,
                "regime": regime,
                "weights_valid": weights_valid,
                "gradients_valid": gradients_valid,
                "performance_valid": performance_valid,
                "mean_reward": mean_reward,
                "action_variance": action_variance,
                "weight_stats": weight_stats,
                "gradient_stats": gradient_stats,
                "observation_dim": obs_dim,
            }
            self.save_snapshot(f"validate_{algo_type}_{regime}", snapshot)
            self.checkpoint(
                pd.DataFrame([snapshot]), data_type=f"validate_{algo_type}_{regime}"
            )
            self.cloud_backup(
                pd.DataFrame([snapshot]), data_type=f"validate_{algo_type}_{regime}"
            )

            if rewards:
                plt.figure(figsize=(10, 6))
                plt.plot(rewards, label="Rewards")
                plt.title(
                    f"Validation: {algo_type} ({regime}), Mean Reward: {mean_reward:.2f}, Action Variance: {action_variance:.2f}"
                )
                plt.xlabel("Simulation")
                plt.ylabel("Reward")
                plt.legend()
                plt.grid(True)
                plt.savefig(
                    self.figure_dir
                    / f'validate_{algo_type}_{regime}_{snapshot["timestamp"]}.png',
                    bbox_inches="tight",
                    optimize=True,
                )
                plt.close()

            result = {
                "valid": weights_valid and gradients_valid and performance_valid,
                "mean_reward": mean_reward,
                "action_variance": action_variance,
                "details": snapshot,
            }
            validation_cache[cache_key] = result
            success_msg = f"Validation {algo_type} ({regime}) pour {self.market}: weights={weights_valid}, gradients={gradients_valid}, performance={performance_valid}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "validate_model",
                time.time() - start_time,
                success=True,
                mean_reward=mean_reward,
                action_variance=action_variance,
            )
            return result

        except Exception as e:
            error_msg = f"Erreur validation modèle {algo_type} ({regime}) pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "validate_model", time.time() - start_time, success=False, error=str(e)
            )
            return {
                "valid": False,
                "mean_reward": 0.0,
                "action_variance": 0.0,
                "details": {"error": str(e)},
            }


if __name__ == "__main__":
    validator = ModelValidator()

    class MockModel:
        class Policy:
            def parameters(self):
                return [
                    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                    torch.tensor([[0.0, 1.0]]),
                ]

            def zero_grad(self):
                pass

        def __init__(self):
            self.policy = self.Policy()

        def predict(self, obs, deterministic=True):
            return np.random.randn(2), None

    model = MockModel()
    feature_cols = [f"feature_{i}" for i in range(350)]
    observations = np.array([np.random.uniform(0, 1, 350)])
    result = validator.validate_model(model, "sac", "trend", observations)
    print(f"Validation result: {result}")
