# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/model/router/policies/transformer_policy.py
# Politique expérimentale basée sur un Transformer pour SAC/PPO/DDPG, avec attention contextuelle et régularisation dynamique.
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle :
# Implémente une politique Transformer pour SAC/PPO/DDPG, gérant 350 features en entraînement et 150 SHAP features en production.
# Intègre l’attention contextuelle (méthode 9) basée sur regime_probs et la régularisation dynamique (méthode 14) via dropout
# ajusté selon vix_es_correlation. Fournit des visualisations, snapshots compressés, et sauvegardes incrémentielles/distribuées.
#
# Dépendances :
# - torch>=2.0.0,<3.0.0, pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, stable-baselines3>=2.0.0,<3.0.0,
#   psutil>=5.9.8,<6.0.0, matplotlib>=3.7.0,<4.0.0, sklearn>=1.5.0,<2.0.0, boto3>=1.26.0,<2.0.0,
#   loguru>=0.7.0,<1.0.0, pyyaml>=6.0.0,<7.0.0, os, signal, gzip
# - src/model/router/detect_regime.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/envs/trading_env.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/router_config.yaml (paramètres de routage)
# - config/feature_sets.yaml (350 features et 150 SHAP features)
# - Observations (batch_size, sequence_length, 350/150 features)
#
# Outputs :
# - Actions prédites, valeurs, log probs
# - Logs dans data/logs/transformer_policy.log
# - Logs de performance dans data/logs/transformer_performance.csv
# - Snapshots compressés dans data/cache/transformer/*.json.gz
# - Sauvegardes incrémentielles dans data/checkpoints/transformer_*.json.gz
# - Visualisations dans data/figures/transformer/
#
# Notes :
# - Intègre attention contextuelle (méthode 9) et régularisation dynamique (méthode 14).
# - Valide 350 features pour l’entraînement et 150 SHAP features pour l’inférence via config_manager.
# - Utilise IQFeed comme source de données via TradingEnv.
# - Implémente retries (max 3, délai 2^attempt secondes) pour les opérations critiques.
# - Sauvegardes incrémentielles toutes les 5 min dans data/checkpoints/ (5 versions).
# - Sauvegardes distribuées toutes les 15 min vers S3 (configurable).
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des observations.
# - Tests unitaires disponibles dans tests/test_transformer_policy.py.
# - Validation complète prévue pour juin 2025.

from collections import OrderedDict, deque
from datetime import datetime
from pathlib import Path
import gzip
import json
import os
import signal
import time
from typing import Callable, Dict, Optional, Tuple, Type

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
from gymnasium import spaces
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import get_config
from src.utils.telegram_alert import send_telegram_alert

# Configuration logging
logger.remove()
BASE_DIR = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
LOG_DIR = BASE_DIR / "data" / "logs"
CACHE_DIR = BASE_DIR / "data" / "cache" / "transformer"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints"
FIGURE_DIR = BASE_DIR / "data" / "figures" / "transformer"
PERF_LOG_PATH = LOG_DIR / "transformer_performance.csv"
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "transformer_policy.log", rotation="10 MB", level="INFO", encoding="utf-8"
)
logger.add(lambda msg: print(msg, end=""), level="INFO")

# Constantes
WINDOW_SIZE = 100
RECENT_ACTIONS_SIZE = 3
MAX_RETRIES = 3
RETRY_DELAY = 2.0
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB

# Variable pour gérer l'arrêt propre
RUNNING = True


class TransformerPolicy(ActorCriticPolicy):
    """
    Politique basée sur un Transformer pour SAC/PPO/DDPG, avec attention contextuelle et régularisation dynamique.

    Attributes:
        observation_space (spaces.Box): Espace d’observation (sequence_length, 350/150).
        action_space (spaces.Box): Espace d’action.
        lr_schedule (Callable): Programme d’apprentissage.
        sequence_length (int): Longueur de la séquence temporelle.
        d_model (int): Dimension interne du Transformer.
        nhead (int): Nombre de têtes d’attention.
        num_layers (int): Nombre de couches Transformer.
        dim_feedforward (int): Dimension des couches feedforward.
        dropout (float): Taux de dropout de base.
        activation_fn (Type[nn.Module]): Fonction d’activation.
        fast_mode (bool): Si True, utilise une architecture légère.
        training_mode (bool): Si True, utilise 350 features; sinon, 150 SHAP features.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        lr_schedule: Callable[[float], float],
        sequence_length: int = 50,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation_fn: Type[nn.Module] = nn.ReLU,
        fast_mode: bool = False,
        training_mode: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialise la politique Transformer.

        Args:
            observation_space (spaces.Box): Espace d’observation (sequence_length, 350/150).
            action_space (spaces.Box): Espace d’action.
            lr_schedule (Callable): Fonction pour ajuster le taux d’apprentissage.
            sequence_length (int): Longueur de la séquence temporelle.
            d_model (int): Dimension interne du Transformer.
            nhead (int): Nombre de têtes d’attention.
            num_layers (int): Nombre de couches Transformer.
            dim_feedforward (int): Dimension des couches feedforward.
            dropout (float): Taux de dropout de base.
            activation_fn (Type[nn.Module]): Fonction d’activation.
            fast_mode (bool): Si True, utilise une architecture légère.
            training_mode (bool): Si True, utilise 350 features; sinon, 150 SHAP features.
            **kwargs: Arguments supplémentaires pour ActorCriticPolicy.
        """
        start_time = datetime.now()
        try:
            self.alert_manager = AlertManager()
            self.config = get_config(BASE_DIR / "config/router_config.yaml")
            self.fast_mode = fast_mode or self.config.get("fast_mode", False)
            self.sequence_length = sequence_length
            self.training_mode = training_mode
            self.num_features = 350 if training_mode else 150

            # Validation des features SHAP dans config/feature_sets.yaml
            feature_sets_path = BASE_DIR / "config/feature_sets.yaml"
            feature_config = get_config(feature_sets_path)
            expected_features = feature_config.get(
                "training" if training_mode else "inference", {}
            ).get("features" if training_mode else "shap_features", [])
            if len(expected_features) != self.num_features:
                raise ValueError(
                    f"Nombre de features incorrect dans feature_sets.yaml: {len(expected_features)} != {self.num_features}"
                )

            if (
                len(observation_space.shape) != 2
                or observation_space.shape[1] != self.num_features
                or observation_space.shape[0] != sequence_length
            ):
                raise ValueError(
                    f"Observation space doit être de forme ({sequence_length}, {self.num_features}), trouvé {observation_space.shape}"
                )
            if d_model % nhead != 0:
                raise ValueError(
                    f"d_model ({d_model}) doit être divisible par nhead ({nhead})"
                )

            # Réduire les paramètres en fast_mode
            self.d_model = 32 if self.fast_mode else d_model
            self.num_layers = 1 if self.fast_mode else num_layers
            self.dim_feedforward = 128 if self.fast_mode else dim_feedforward

            super().__init__(
                observation_space=observation_space,
                action_space=action_space,
                lr_schedule=lr_schedule,
                **kwargs,
            )
            self.nhead = nhead
            self.dropout = dropout
            self.activation_fn = activation_fn

            # Normalisation avec MinMaxScaler
            self.scaler = MinMaxScaler()
            self.obs_window = deque(maxlen=WINDOW_SIZE)

            # Seuils de performance
            self.performance_window = deque(maxlen=WINDOW_SIZE)
            self.performance_thresholds = {
                "min_action_mean": -1.0,
                "max_action_mean": 1.0,
                "min_value_mean": -1000.0,
                "max_value_mean": 1000.0,
                "vix_peak_threshold": self.config.get("thresholds", {}).get(
                    "vix_peak_threshold", 30.0
                ),
                "spread_explosion_threshold": self.config.get("thresholds", {}).get(
                    "spread_explosion_threshold", 0.05
                ),
                "confidence_threshold": self.config.get("thresholds", {}).get(
                    "confidence_threshold", 0.7
                ),
                "net_gamma_threshold": self.config.get("thresholds", {}).get(
                    "net_gamma_threshold", 1.0
                ),
                "vol_trigger_threshold": self.config.get("thresholds", {}).get(
                    "vol_trigger_threshold", 1.0
                ),
                "dealer_zones_count_threshold": self.config.get("thresholds", {}).get(
                    "dealer_zones_count_threshold", 5
                ),
            }

            # Contexte roulant pour actions récentes
            self.recent_actions = deque(maxlen=RECENT_ACTIONS_SIZE)

            # Cache LRU pour prédictions
            self.prediction_cache = OrderedDict()
            self.max_cache_size = 1000
            self.checkpoint_versions = []
            self.log_buffer = []

            # Projection des features vers d_model
            self.input_projection = nn.Linear(self.num_features, self.d_model)

            # Couche d’attention contextuelle pour regime_probs (méthode 9)
            self.regime_attention = nn.MultiheadAttention(
                embed_dim=self.d_model, num_heads=nhead, dropout=dropout
            )

            # Définition du Transformer
            encoder_layer = TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=dropout,
                activation="relu" if activation_fn == nn.ReLU else "gelu",
            )
            self.transformer_encoder = TransformerEncoder(
                encoder_layer, num_layers=self.num_layers
            )

            # Couches finales pour politique et valeur
            self.policy_net = nn.Linear(
                self.d_model, action_space.shape[0] * 2
            )  # Moyenne et log_std
            self.value_net = nn.Linear(self.d_model, 1)  # Valeur unique

            self.action_dist = SquashedDiagGaussianDistribution(action_space.shape[0])

            success_msg = (
                f"TransformerPolicy initialisée avec sequence_length={sequence_length}, d_model={self.d_model}, "
                f"nhead={nhead}, num_layers={self.num_layers}, fast_mode={self.fast_mode}, "
                f"training_mode={self.training_mode}"
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
            error_msg = f"Erreur initialisation TransformerPolicy: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
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
            snapshot_path = CACHE_DIR / f"snapshot_{snapshot_type}_{timestamp}.json"
            CACHE_DIR.mkdir(exist_ok=True)

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

    def checkpoint(self, data: pd.DataFrame, data_type: str = "policy_state") -> None:
        """
        Sauvegarde incrémentielle des données toutes les 5 minutes avec versionnage (5 versions).

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : policy_state).
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
                CHECKPOINT_DIR / f"transformer_{data_type}_{timestamp}.json.gz"
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

    def cloud_backup(self, data: pd.DataFrame, data_type: str = "policy_state") -> None:
        """
        Sauvegarde distribuée des données vers AWS S3 toutes les 15 minutes (configurable).

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : policy_state).
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
                f"{self.config['s3_prefix']}transformer_{data_type}_{timestamp}.csv.gz"
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
        Journalise les performances dans transformer_performance.csv.

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

    def _validate_observations(self, obs: torch.Tensor) -> None:
        """
        Valide que les observations sont numériques, scalaires et finies.

        Args:
            obs (torch.Tensor): Tenseur d’observations (batch_size, sequence_length, 350/150).
        """
        start_time = datetime.now()
        try:

            def validate():
                if (
                    len(obs.shape) != 3
                    or obs.shape[1] != self.sequence_length
                    or obs.shape[2] != self.num_features
                ):
                    raise ValueError(
                        f"Observation doit être de forme (batch_size, {self.sequence_length}, {self.num_features}), trouvé {obs.shape}"
                    )
                if not torch.is_floating_point(obs):
                    raise ValueError(
                        f"Observations doivent être de type flottant, trouvé {obs.dtype}"
                    )
                nan_count = torch.isnan(obs).sum().item()
                inf_count = torch.isinf(obs).sum().item()
                confidence_drop_rate = (
                    (nan_count + inf_count)
                    / (obs.shape[0] * obs.shape[1] * obs.shape[2])
                    if (obs.shape[0] * obs.shape[1] * obs.shape[2]) > 0
                    else 0.0
                )
                if confidence_drop_rate > 0.5:
                    alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({nan_count} NaN, {inf_count} Inf)"
                    logger.warning(alert_msg)
                    self.alert_manager.send_alert(alert_msg, priority=3)
                    send_telegram_alert(alert_msg)
                if nan_count > 0 or inf_count > 0:
                    raise ValueError(
                        f"Observations contiennent {nan_count} NaN et {inf_count} valeurs infinies"
                    )
                return confidence_drop_rate

            confidence_drop_rate = self.with_retries(validate)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "validate_observations",
                latency,
                success=True,
                confidence_drop_rate=confidence_drop_rate,
            )
        except Exception as e:
            error_msg = (
                f"Erreur validation observations: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "validate_observations", 0, success=False, error=str(e)
            )
            raise

    def update_normalization_stats(self, obs: torch.Tensor) -> None:
        """
        Met à jour les statistiques de normalisation avec fenêtre glissante.

        Args:
            obs (torch.Tensor): Tenseur d’observations (batch_size, sequence_length, 350/150).
        """
        start_time = datetime.now()
        try:
            if (
                len(obs.shape) != 3
                or obs.shape[1] != self.sequence_length
                or obs.shape[2] != self.num_features
            ):
                raise ValueError(
                    f"Observation doit être de forme (batch_size, {self.sequence_length}, {self.num_features}), trouvé {obs.shape}"
                )
            obs_flat = obs.view(-1, obs.shape[-1]).cpu().numpy()
            self.obs_window.extend(obs_flat)
            window_array = np.array(self.obs_window)
            if len(window_array) > 0:
                self.scaler.fit(window_array)
            success_msg = f"Statistiques de normalisation mises à jour, window_size={len(self.obs_window)}"
            logger.debug(success_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "update_normalization_stats",
                latency,
                success=True,
                num_features=obs.shape[-1],
            )
        except Exception as e:
            error_msg = f"Erreur dans update_normalization_stats: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "update_normalization_stats", 0, success=False, error=str(e)
            )
            raise

    def update_performance_thresholds(
        self,
        actions: torch.Tensor,
        values: torch.Tensor,
        context: Optional[Dict] = None,
    ):
        """
        Met à jour dynamiquement les seuils de performance.

        Args:
            actions (torch.Tensor): Actions prédites.
            values (torch.Tensor): Valeurs prédites.
            context (Optional[Dict]): Contexte avec métriques (ex. : vix_es_correlation, spread).
        """
        start_time = datetime.now()
        try:
            vix = context.get("vix_es_correlation", 0.0) if context else 0.0
            spread = context.get("spread", 0.0) if context else 0.0
            net_gamma = context.get("net_gamma", 0.0) if context else 0.0
            vol_trigger = context.get("vol_trigger", 0.0) if context else 0.0
            dealer_zones_count = context.get("dealer_zones_count", 0) if context else 0
            if (
                vix > self.performance_thresholds["vix_peak_threshold"]
                or spread > self.performance_thresholds["spread_explosion_threshold"]
                or abs(net_gamma) > self.performance_thresholds["net_gamma_threshold"]
                or abs(vol_trigger)
                > self.performance_thresholds["vol_trigger_threshold"]
                or dealer_zones_count
                > self.performance_thresholds["dealer_zones_count_threshold"]
            ):
                self.performance_window.clear()
                alert_msg = (
                    f"Reset performance_window: VIX={vix:.2f}, spread={spread:.4f}, "
                    f"net_gamma={net_gamma:.2f}, vol_trigger={vol_trigger:.2f}, "
                    f"dealer_zones_count={dealer_zones_count}"
                )
                logger.info(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=4)
                send_telegram_alert(alert_msg)
            self.performance_window.append(
                {
                    "action_mean": actions.mean().item(),
                    "value_mean": values.mean().item(),
                }
            )
            if len(self.performance_window) >= 10:
                window_df = pd.DataFrame(self.performance_window)
                self.performance_thresholds["min_action_mean"] = (
                    window_df["action_mean"].mean() - window_df["action_mean"].std()
                )
                self.performance_thresholds["max_action_mean"] = (
                    window_df["action_mean"].mean() + window_df["action_mean"].std()
                )
                self.performance_thresholds["min_value_mean"] = (
                    window_df["value_mean"].mean() - window_df["value_mean"].std()
                )
                self.performance_thresholds["max_value_mean"] = (
                    window_df["value_mean"].mean() + window_df["value_mean"].std()
                )
                logger.debug(f"Seuils mis à jour: {self.performance_thresholds}")
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("update_performance_thresholds", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur mise à jour seuils: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "update_performance_thresholds", 0, success=False, error=str(e)
            )
            raise

    def check_action_validity(self, action: torch.Tensor) -> bool:
        """
        Vérifie si l’action prédite est cohérente avec les actions récentes.

        Args:
            action (torch.Tensor): Actions prédites.

        Returns:
            bool: True si les actions sont cohérentes, False sinon.
        """
        try:
            if not self.recent_actions:
                return True
            action_mean = action.mean().item()
            recent_means = [a.mean().item() for a in self.recent_actions]
            recent_std = np.std(recent_means) if recent_means else 0.0
            return abs(action_mean - np.mean(recent_means)) <= 2 * recent_std + 1e-6
        except Exception as e:
            error_msg = f"Erreur vérification cohérence actions: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            return False

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        context: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Effectue un passage avant avec le Transformer, intégrant attention contextuelle (méthode 9) et régularisation dynamique (méthode 14).

        Args:
            obs (torch.Tensor): Tenseur d’observations (batch_size, sequence_length, 350/150).
            deterministic (bool): Si True, utilise des actions déterministes.
            context (Optional[Dict]): Contexte avec métriques (ex. : vix_es_correlation, regime_probs).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Actions, valeur, log probabilité.
        """
        start_time = datetime.now()
        try:

            def compute_forward():
                self._validate_observations(obs)

                # Mode ultra-défensif
                ultra_defensive = False
                vix = context.get("vix_es_correlation", 0.0) if context else 0.0
                spread = context.get("spread", 0.0) if context else 0.0
                net_gamma = context.get("net_gamma", 0.0) if context else 0.0
                vol_trigger = context.get("vol_trigger", 0.0) if context else 0.0
                dealer_zones_count = (
                    context.get("dealer_zones_count", 0) if context else 0
                )
                regime_probs = (
                    context.get(
                        "regime_probs",
                        {"trend": 0.33, "range": 0.34, "defensive": 0.33},
                    )
                    if context
                    else {"trend": 0.33, "range": 0.34, "defensive": 0.33}
                )
                regime = max(regime_probs, key=regime_probs.get)

                if (
                    vix > self.performance_thresholds["vix_peak_threshold"]
                    or spread
                    > self.performance_thresholds["spread_explosion_threshold"]
                    or abs(net_gamma)
                    > self.performance_thresholds["net_gamma_threshold"]
                    or abs(vol_trigger)
                    > self.performance_thresholds["vol_trigger_threshold"]
                    or dealer_zones_count
                    > self.performance_thresholds["dealer_zones_count_threshold"]
                ):
                    ultra_defensive = True
                    alert_msg = (
                        f"Mode Ultra-Defensive activé: VIX={vix:.2f}, spread={spread:.4f}, "
                        f"net_gamma={net_gamma:.2f}, vol_trigger={vol_trigger:.2f}, "
                        f"dealer_zones_count={dealer_zones_count}"
                    )
                    logger.info(alert_msg)
                    self.alert_manager.send_alert(alert_msg, priority=4)
                    send_telegram_alert(alert_msg)

                # Normalisation
                self.update_normalization_stats(obs)
                if self.obs_window:
                    obs_np = obs.cpu().numpy().reshape(-1, obs.shape[-1])
                    obs_normalized = self.scaler.transform(obs_np)
                    obs = torch.tensor(
                        obs_normalized.reshape(obs.shape),
                        dtype=torch.float32,
                        device=obs.device,
                    )

                if ultra_defensive:
                    actions = torch.zeros_like(
                        torch.tensor(self.action_space.sample(), device=obs.device),
                        dtype=torch.float32,
                    )
                    values = torch.zeros(
                        (obs.shape[0], 1), dtype=torch.float32, device=obs.device
                    )
                    log_probs = torch.zeros(
                        (obs.shape[0],), dtype=torch.float32, device=obs.device
                    )
                    confidence_score = 0.0
                    attention_weights = torch.zeros(
                        (obs.shape[1], obs.shape[1]), device=obs.device
                    )
                else:
                    # Cache
                    cache_key = f"{obs.sum().item()}_{deterministic}_{regime}_{vix}"
                    if cache_key in self.prediction_cache:
                        (
                            actions,
                            values,
                            log_probs,
                            confidence_score,
                            attention_weights,
                        ) = self.prediction_cache[cache_key]
                        self.prediction_cache.move_to_end(cache_key)
                    else:
                        # Régularisation dynamique (méthode 14)
                        dropout = min(0.3, 0.1 + vix / 25.0)
                        for layer in self.transformer_encoder.layers:
                            layer.dropout = dropout
                            layer.self_attn.dropout = dropout
                        self.regime_attention.dropout = dropout

                        # Projection des features
                        obs_proj = self.input_projection(
                            obs
                        )  # (batch_size, sequence_length, d_model)
                        obs_proj = obs_proj.permute(
                            1, 0, 2
                        )  # (sequence_length, batch_size, d_model)

                        # Attention contextuelle (méthode 9)
                        regime_vector = (
                            torch.tensor(
                                [
                                    regime_probs["trend"],
                                    regime_probs["range"],
                                    regime_probs["defensive"],
                                ],
                                dtype=torch.float32,
                                device=obs.device,
                            )
                            .unsqueeze(0)
                            .unsqueeze(0)
                        )
                        # (sequence_length, batch_size, 3)
                        regime_vector = regime_vector.expand(
                            obs.shape[1], obs.shape[0], -1
                        )
                        regime_vector = nn.Linear(3, self.d_model).to(obs.device)(
                            regime_vector
                        )  # (sequence_length, batch_size, d_model)
                        attention_output, attention_weights = self.regime_attention(
                            obs_proj, regime_vector, regime_vector
                        )

                        # Transformer
                        # (sequence_length, batch_size, d_model)
                        transformer_out = self.transformer_encoder(attention_output)
                        features = transformer_out[-1]  # (batch_size, d_model)

                        # Calcul des paramètres
                        action_params = self.policy_net(features)
                        mean_actions, log_std = action_params.chunk(2, dim=-1)
                        log_std = torch.clamp(log_std, -20, 2)
                        action_dist = self.action_dist.proba_distribution(
                            mean_actions, log_std
                        )
                        actions = action_dist.get_actions(deterministic=deterministic)

                        values = self.value_net(features)
                        log_probs = self.action_dist.log_prob(actions)

                        # Score de confiance
                        confidence_score = (
                            1.0 / (actions.std().item() + 1e-6)
                            if actions.std().item() > 0
                            else 1.0
                        )
                        confidence_score = min(confidence_score, 1.0)

                        # Vérification des actions
                        if not self.check_action_validity(actions):
                            warning_msg = (
                                f"Action incohérente: {actions.mean().item():.2f}"
                            )
                            logger.warning(warning_msg)
                            self.alert_manager.send_alert(warning_msg, priority=4)
                            send_telegram_alert(warning_msg)
                            actions = torch.zeros_like(actions, device=obs.device)

                        # Mise à jour cache
                        self.prediction_cache[cache_key] = (
                            actions,
                            values,
                            log_probs,
                            confidence_score,
                            attention_weights,
                        )
                        self.prune_cache()

                    # Mise à jour actions récentes
                    self.recent_actions.append(actions.clone().detach())

                # Validation seuils
                if not ultra_defensive:
                    if (
                        actions.mean().item()
                        < self.performance_thresholds["min_action_mean"]
                        or actions.mean().item()
                        > self.performance_thresholds["max_action_mean"]
                    ):
                        logger.warning(
                            f"Action mean ({actions.mean().item():.2f}) hors plage"
                        )
                    if (
                        values.mean().item()
                        < self.performance_thresholds["min_value_mean"]
                        or values.mean().item()
                        > self.performance_thresholds["max_value_mean"]
                    ):
                        logger.warning(
                            f"Value mean ({values.mean().item():.2f}) hors plage"
                        )
                    if (
                        confidence_score
                        < self.performance_thresholds["confidence_threshold"]
                    ):
                        logger.warning(f"Confiance faible: {confidence_score:.2f}")

                # Mise à jour seuils
                self.update_performance_thresholds(actions, values, context)

                # Visualisation des poids d’attention
                plt.figure(figsize=(10, 8))
                plt.imshow(attention_weights.cpu().numpy(), cmap="viridis")
                plt.colorbar(label="Attention Weight")
                plt.title(
                    f"Attention Weights - Regime {regime}, Dropout: {dropout:.2f}"
                )
                plt.xlabel("Key")
                plt.ylabel("Query")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plt.savefig(FIGURE_DIR / f"attention_{timestamp}.png")
                plt.close()

                # Visualisation des observations
                plt.figure(figsize=(10, 6))
                plt.plot(
                    obs.mean(dim=(0, 2)).cpu().numpy(),
                    label="Observation Mean",
                    color="blue",
                )
                if context and "net_gamma" in context:
                    plt.axhline(
                        y=context["net_gamma"],
                        color="orange",
                        linestyle="--",
                        label="Net Gamma",
                    )
                plt.title(
                    f"Transformer Policy: Regime {regime}, Action Mean: {actions.mean().item():.2f}, Dropout: {dropout:.2f}"
                )
                plt.xlabel("Feature Index")
                plt.ylabel("Value")
                plt.legend()
                plt.savefig(FIGURE_DIR / f"forward_{timestamp}.png")
                plt.close()

                # Snapshot JSON
                snapshot_data = {
                    "timestamp": timestamp,
                    "actions_mean": actions.mean().item(),
                    "values_mean": values.mean().item(),
                    "confidence_score": confidence_score,
                    "ultra_defensive": ultra_defensive,
                    "regime": regime,
                    "dropout": dropout,
                    "num_features": self.num_features,
                    "attention_weights_shape": list(attention_weights.shape),
                }
                self.save_snapshot("forward", snapshot_data)

                # Sauvegarde incrémentielle
                perf_data = pd.DataFrame(
                    [
                        {
                            "timestamp": timestamp,
                            "actions_mean": actions.mean().item(),
                            "values_mean": values.mean().item(),
                            "confidence_score": confidence_score,
                            "regime": regime,
                            "dropout": dropout,
                        }
                    ]
                )
                self.checkpoint(perf_data, data_type="forward_metrics")
                self.cloud_backup(perf_data, data_type="forward_metrics")

                logger.debug(
                    f"Passage avant: actions={actions.mean().item():.2f}, values={values.mean().item():.2f}, "
                    f"confidence={confidence_score:.2f}, ultra_defensive={ultra_defensive}, regime={regime}"
                )
                return actions, values, log_probs

            actions, values, log_probs = self.with_retries(compute_forward)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "forward",
                latency,
                success=True,
                regime=regime,
                num_features=self.num_features,
            )
            return actions, values, log_probs
        except Exception as e:
            error_msg = f"Erreur dans forward: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "forward",
                latency,
                success=False,
                error=str(e),
                regime=context.get("regime", "range") if context else "range",
                num_features=self.num_features,
            )
            raise

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        """
        Retourne les paramètres pour la reconstruction du modèle.

        Returns:
            Dict[str, Any]: Paramètres de construction.
        """
        start_time = datetime.now()
        try:
            data = super()._get_constructor_parameters()
            data.update(
                {
                    "sequence_length": self.sequence_length,
                    "d_model": self.d_model,
                    "nhead": self.nhead,
                    "num_layers": self.num_layers,
                    "dim_feedforward": self.dim_feedforward,
                    "dropout": self.dropout,
                    "activation_fn": self.activation_fn,
                    "fast_mode": self.fast_mode,
                    "training_mode": self.training_mode,
                }
            )
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("get_constructor_parameters", latency, success=True)
            return data
        except Exception as e:
            error_msg = f"Erreur dans _get_constructor_parameters: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "get_constructor_parameters", 0, success=False, error=str(e)
            )
            raise