# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/envs/gym_wrappers.py
# Wrappers Gym pour environnements de trading ES, adaptés à MIA_IA_SYSTEM_v2_2025.
# Contient des wrappers pour empilement, normalisation, clipping, et régimes spécifiques
# (range, trend, défensif) avec régimes hybrides (méthode 11) basés sur regime_probs.
#
# Version : 2.1.5
# Date : 2025-05-15
#
# Rôle : Ajuste les observations et récompenses selon les régimes, intègre les niveaux critiques
#        d’options (Put Wall, Call Wall, etc.) et les nouvelles fonctionnalités (bid_ask_imbalance,
#        iv_skew, iv_term_structure, trade_aggressiveness, option_skew, news_impact_score).
#        Conforme à la Phase 8 (auto-conscience via alertes), Phase 12 (simulation de trading),
#        et Phase 16 (ensemble learning). Optimisé pour HFT avec validation dynamique et gestion mémoire.
#
# Dépendances :
# - gymnasium>=0.26.0,<1.0.0
# - pandas>=2.0.0,<3.0.0
# - numpy>=1.26.4,<2.0.0
# - psutil>=5.9.0,<6.0.0
# - pyyaml>=6.0.0,<7.0.0
# - logging, signal, datetime, time, json, gzip
# - src.features.neural_pipeline (version 2.1.5)
# - src.model.utils.config_manager (version 2.1.5)
# - src.model.utils.alert_manager (version 2.1.5)
# - src.model.utils.obs_template (version 2.1.5)
# - src.utils.telegram_alert (version 2.1.5)
# - src.features.detect_regime (version 2.1.5)
#
# Inputs :
# - config/trading_env_config.yaml
# - config/feature_sets.yaml
# - config/model_params.yaml
#
# Outputs :
# - data/logs/gym_wrappers.log
# - data/logs/gym_wrappers_performance.csv
# - data/logs/normalization/norm_mean.npy
# - data/logs/normalization/norm_std.npy
# - data/logs/wrapper_sigint_*.json.gz
#
# Notes :
# - Utilise IQFeed exclusivement comme source de données.
# - Compatible avec 350 features (entraînement) ou 150 SHAP features (inférence) définies dans config/feature_sets.yaml.
# - Implémente retries (max 3, délai adaptatif 2^attempt secondes) pour les opérations critiques.
# - Logs psutil dans data/logs/gym_wrappers_performance.csv avec seuil mémoire réduit à 800 MB.
# - Alertes via alert_manager.py et telegram_alert.py.
# - Tests unitaires disponibles dans tests/test_gym_wrappers.py (couvre bid_ask_imbalance, iv_skew, etc.).
# - Intègre validation obs_t et ajustements dynamiques des poids de régime.

import gzip
import json
import logging
import os
import signal
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
from gymnasium import Env, ObservationWrapper, Wrapper, spaces

from src.features.detect_regime import detect_regime
from src.features.neural_pipeline import NeuralPipeline
from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import config_manager
from src.model.utils.obs_template import validate_obs_t
from src.utils.telegram_alert import send_telegram_alert

# Chemins relatifs
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Configuration du logging global
log_dir = BASE_DIR / "data" / "logs"
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_dir / "gym_wrappers.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Chemin pour les logs de performance
PERFORMANCE_LOG = log_dir / "gym_wrappers_performance.csv"

# Constantes
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Secondes, exponentiel
MEMORY_THRESHOLD = 800  # MB, seuil pour alerte mémoire


class BaseEnvWrapper(Wrapper):
    """Classe de base pour les wrappers avec logs psutil."""

    def __init__(
        self,
        env: Env,
        config_path: str = str(BASE_DIR / "config" / "trading_env_config.yaml"),
    ):
        super().__init__(env)
        self.alert_manager = AlertManager()
        signal.signal(signal.SIGINT, self.handle_sigint)
        self.log_buffer = []
        start_time = time.time()
        try:
            self.config = self.with_retries(
                lambda: config_manager.get_config(
                    os.path.relpath(BASE_DIR / config_path, BASE_DIR)
                )
            )
            if not self.config:
                raise ValueError("Configuration vide ou non trouvée")
            self.buffer_size = self.config.get("logging", {}).get("buffer_size", 50)  # Réduit pour HFT
            self.alert_manager.send_alert(
                f"BaseEnvWrapper configuré avec buffer_size={self.buffer_size}",
                priority=1,
            )
            send_telegram_alert(
                f"BaseEnvWrapper configuré avec buffer_size={self.buffer_size}"
            )
            logger.info(f"BaseEnvWrapper configuré avec buffer_size={self.buffer_size}")
        except Exception as e:
            error_msg = f"Erreur chargement config: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.config = {"logging": {"buffer_size": 50}}
            self.buffer_size = 50
            self.log_performance("init_base_wrapper", 0, success=False, error=str(e))

        self.log_performance(
            "init_base_wrapper", time.time() - start_time, success=True
        )

    def handle_sigint(self, signum, frame):
        """Gère l'arrêt propre sur SIGINT."""
        snapshot = {"timestamp": datetime.now().isoformat(), "status": "SIGINT"}
        try:
            self.save_snapshot("sigint", snapshot)
            self.alert_manager.send_alert(
                "Arrêt propre sur SIGINT, snapshot sauvegardé", priority=2
            )
            send_telegram_alert("Arrêt propre sur SIGINT, snapshot sauvegardé")
        except Exception as e:
            error_msg = f"Erreur sauvegarde SIGINT: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
        exit(0)

    def save_snapshot(self, snapshot_type: str, data: Dict) -> None:
        """Sauvegarde un instantané des résultats avec compression gzip."""
        try:
            start_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = {
                "timestamp": timestamp,
                "type": snapshot_type,
                "data": data,
                "buffer_size": len(self.log_buffer),
            }
            path = log_dir / f"wrapper_{snapshot_type}_{timestamp}.json"
            os.makedirs(path.parent, exist_ok=True)
            with gzip.open(f"{path}.gz", "wt", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=4)
            latency = time.time() - start_time
            self.alert_manager.send_alert(
                f"Snapshot {snapshot_type} sauvegardé: {path}.gz", priority=1
            )
            send_telegram_alert(f"Snapshot {snapshot_type} sauvegardé: {path}.gz")
            logger.info(f"Snapshot {snapshot_type} sauvegardé: {path}.gz")
            self.log_performance("save_snapshot", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("save_snapshot", 0, success=False, error=str(e))

    def with_retries(
        self,
        func: callable,
        max_attempts: int = MAX_RETRIES,
        delay_base: float = RETRY_DELAY_BASE,
    ) -> Optional[Any]:
        """
        Exécute une fonction avec retries (max 3, délai exponentiel).

        Args:
            func (callable): Fonction à exécuter.
            max_attempts (int): Nombre maximum de tentatives.
            delay_base (float): Base pour le délai exponentiel.

        Returns:
            Optional[any]: Résultat de la fonction ou None si échec.
        """
        start_time = time.time()
        for attempt in range(max_attempts):
            try:
                result = func()
                latency = time.time() - start_time
                self.log_performance(
                    f"retry_attempt_{attempt+1}", latency, success=True
                )
                return result
            except Exception as e:
                if attempt == max_attempts - 1:
                    error_msg = f"Échec après {max_attempts} tentatives: {str(e)}\n{traceback.format_exc()}"
                    self.alert_manager.send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    logger.error(error_msg)
                    self.log_performance(
                        f"retry_attempt_{attempt+1}", 0, success=False, error=str(e)
                    )
                    return None
                delay = delay_base ** attempt
                self.alert_manager.send_alert(
                    f"Tentative {attempt+1} échouée, retry après {delay}s", priority=3
                )
                send_telegram_alert(
                    f"Tentative {attempt+1} échouée, retry après {delay}s"
                )
                logger.warning(f"Tentative {attempt+1} échouée, retry après {delay}s")
                time.sleep(delay)

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None
    ):
        """Enregistre les performances avec psutil dans data/logs/gym_wrappers_performance.csv."""
        start_time = time.time()
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
            cpu_usage = psutil.cpu_percent()
            if memory_usage > MEMORY_THRESHOLD:
                error_msg = f"ALERTE: Usage mémoire élevé ({memory_usage:.2f} MB)"
                self.alert_manager.send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "success": success,
                "latency": latency,
                "error": error,
                "cpu_percent": cpu_usage,
                "memory_mb": memory_usage,
            }
            self.log_buffer.append(log_entry)
            if len(self.log_buffer) >= self.buffer_size:
                log_df = pd.DataFrame(self.log_buffer)
                mode = "a" if PERFORMANCE_LOG.exists() else "w"

                def save_log():
                    log_df.to_csv(
                        PERFORMANCE_LOG,
                        mode=mode,
                        index=False,
                        header=not PERFORMANCE_LOG.exists(),
                        encoding="utf-8",
                    )

                self.with_retries(save_log)
                self.log_buffer = []
            self.log_performance(
                "log_performance", time.time() - start_time, success=True
            )
        except Exception as e:
            error_msg = (
                f"Erreur journalisation performance: {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("log_performance", 0, success=False, error=str(e))


class ObservationStackingWrapper(BaseEnvWrapper, ObservationWrapper):
    """Wrapper pour empiler plusieurs observations consécutives."""

    def __init__(
        self,
        env: Env,
        stack_size: int = None,
        config_path: str = str(BASE_DIR / "config" / "trading_env_config.yaml"),
        policy_type: str = "mlp",
    ):
        super().__init__(env, config_path)
        self.policy_type = policy_type
        start_time = time.time()
        try:
            self.stack_size = (
                stack_size
                if stack_size is not None
                else self.config.get("stacking", {}).get("stack_size", 3)
            )
            feature_sets = self.with_retries(
                lambda: config_manager.get_config(
                    os.path.relpath(BASE_DIR / "config" / "feature_sets.yaml", BASE_DIR)
                )
            )
            self.training_mode = self.config.get("environment", {}).get(
                "training_mode", True
            )
            self.base_features = 350 if self.training_mode else 150
            if self.stack_size <= 0:
                raise ValueError(
                    f"stack_size doit être positif, obtenu : {self.stack_size}"
                )
            self.alert_manager.send_alert(
                f"ObservationStackingWrapper configuré: stack_size={self.stack_size}, base_features={self.base_features}",
                priority=1,
            )
            send_telegram_alert(
                f"ObservationStackingWrapper configuré: stack_size={self.stack_size}, base_features={self.base_features}"
            )
            logger.info(
                f"ObservationStackingWrapper configuré: stack_size={self.stack_size}, base_features={self.base_features}"
            )
        except Exception as e:
            error_msg = f"Erreur configuration: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.warning(error_msg)
            self.stack_size = 3
            self.base_features = 350
            self.log_performance(
                "init_observation_stacking", 0, success=False, error=str(e)
            )

        self.neural_pipeline = NeuralPipeline(
            window_size=self.config.get("observation", {}).get("sequence_length", 50),
            base_features=self.base_features,
            config_path=str(BASE_DIR / "config" / "model_params.yaml"),
        )
        try:
            self.with_retries(lambda: self.neural_pipeline.load_models())
            self.alert_manager.send_alert(
                f"NeuralPipeline chargé, policy_type={self.policy_type}", priority=1
            )
            send_telegram_alert(
                f"NeuralPipeline chargé, policy_type={self.policy_type}"
            )
            logger.info(f"NeuralPipeline chargé, policy_type={self.policy_type}")
        except Exception as e:
            error_msg = (
                f"Erreur chargement NeuralPipeline: {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.warning(error_msg)
            self.log_performance("init_neural_pipeline", 0, success=False, error=str(e))

        self.obs_stack = deque(maxlen=self.stack_size)

        if self.policy_type == "transformer":
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.stack_size, self.base_features),
                dtype=np.float32,
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.base_features * self.stack_size,),
                dtype=np.float32,
            )
        self._stacked_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.alert_manager.send_alert(
            f"ObservationStackingWrapper initialisé: stack_size={self.stack_size}, shape={self.observation_space.shape}",
            priority=1,
        )
        send_telegram_alert(
            f"ObservationStackingWrapper initialisé: stack_size={self.stack_size}, shape={self.observation_space.shape}"
        )
        logger.info(
            f"ObservationStackingWrapper initialisé: stack_size={self.stack_size}, shape={self.observation_space.shape}"
        )
        self.log_performance(
            "init_observation_stacking", time.time() - start_time, success=True
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Réinitialise l'environnement et la pile."""
        start_time = time.time()
        try:
            obs, info = self.env.reset(**kwargs)
            self.obs_stack.clear()
            zero_obs = np.zeros(self.base_features, dtype=np.float32)
            for _ in range(self.stack_size - 1):
                self.obs_stack.append(zero_obs)
            self.obs_stack.append(obs)
            stacked_obs = self._get_stacked_observation()
            self.log_performance(
                "reset_observation_stacking", time.time() - start_time, success=True
            )
            self.alert_manager.send_alert(
                "ObservationStackingWrapper réinitialisé", priority=1
            )
            send_telegram_alert("ObservationStackingWrapper réinitialisé")
            return stacked_obs, info
        except Exception as e:
            error_msg = f"Erreur reset: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "reset_observation_stacking", 0, success=False, error=str(e)
            )
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Ajoute l'observation à la pile et retourne la pile complète."""
        start_time = time.time()
        try:
            if not np.all(np.isfinite(observation)):
                error_msg = "Observation contient NaN/inf, remplacement par 0"
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                logger.warning(error_msg)
                observation = np.nan_to_num(
                    observation, nan=0.0, posinf=0.0, neginf=0.0
                )
            if len(observation) != self.base_features:
                error_msg = f"Taille observation incorrecte: {len(observation)} au lieu de {self.base_features}"
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)
            self.obs_stack.append(observation)
            result = self._get_stacked_observation()
            self.log_performance(
                "observation_stacking", time.time() - start_time, success=True
            )
            return result
        except Exception as e:
            error_msg = f"Erreur observation: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("observation_stacking", 0, success=False, error=str(e))
            return np.zeros(self.observation_space.shape, dtype=np.float32)

    def incremental_observation(self, observation: np.ndarray) -> np.ndarray:
        """Ajoute une observation unique à la pile pour le mode incrémental."""
        start_time = time.time()
        try:
            if not np.all(np.isfinite(observation)):
                error_msg = (
                    "Observation incrémentale contient NaN/inf, remplacement par 0"
                )
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                logger.warning(error_msg)
                observation = np.nan_to_num(
                    observation, nan=0.0, posinf=0.0, neginf=0.0
                )
            if len(observation) != self.base_features:
                error_msg = f"Taille observation incrémentale incorrecte: {len(observation)} au lieu de {self.base_features}"
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)
            self.obs_stack.append(observation)
            result = self._get_stacked_observation()
            self.log_performance(
                "incremental_observation_stacking",
                time.time() - start_time,
                success=True,
            )
            return result
        except Exception as e:
            error_msg = (
                f"Erreur incremental_observation: {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "incremental_observation_stacking", 0, success=False, error=str(e)
            )
            return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _get_stacked_observation(self) -> np.ndarray:
        """Retourne les observations empilées selon policy_type."""
        try:
            if self.policy_type == "transformer":
                return np.array(self.obs_stack, dtype=np.float32)
            else:
                for i, obs in enumerate(self.obs_stack):
                    start = i * self.base_features
                    end = (i + 1) * self.base_features
                    self._stacked_obs[start:end] = obs
                return self._stacked_obs.copy()
        except Exception as e:
            error_msg = (
                f"Erreur _get_stacked_observation: {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            return np.zeros(self.observation_space.shape, dtype=np.float32)


class NormalizationWrapper(BaseEnvWrapper, ObservationWrapper):
    """Wrapper pour normaliser les observations."""

    def __init__(
        self,
        env: Env,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        config_path: str = str(BASE_DIR / "config" / "trading_env_config.yaml"),
        policy_type: str = "mlp",
    ):
        super().__init__(env, config_path)
        self.policy_type = policy_type
        start_time = time.time()
        try:
            self.min_std = self.config.get("normalization", {}).get("min_std", 1e-8)
            self.alpha = self.config.get("normalization", {}).get("alpha", 0.01)
            self.save_dir = (
                BASE_DIR
                / self.config.get("logging", {}).get("directory", "data/logs")
                / "normalization"
            )
            self.save_dir.mkdir(exist_ok=True)
            self.alert_manager.send_alert(
                f"NormalizationWrapper configuré: min_std={self.min_std}, alpha={self.alpha}",
                priority=1,
            )
            send_telegram_alert(
                f"NormalizationWrapper configuré: min_std={self.min_std}, alpha={self.alpha}"
            )
            logger.info(
                f"NormalizationWrapper configuré: min_std={self.min_std}, alpha={self.alpha}"
            )
        except Exception as e:
            error_msg = f"Erreur configuration: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.warning(error_msg)
            self.min_std = 1e-8
            self.alpha = 0.01
            self.save_dir = BASE_DIR / "data" / "logs" / "normalization"
            self.save_dir.mkdir(exist_ok=True)
            self.log_performance("init_normalization", 0, success=False, error=str(e))

        self.neural_pipeline = NeuralPipeline(
            window_size=self.config.get("observation", {}).get("sequence_length", 50),
            base_features=(
                350
                if self.config.get("environment", {}).get("training_mode", True)
                else 150
            ),
            config_path=str(BASE_DIR / "config" / "model_params.yaml"),
        )
        try:
            self.with_retries(lambda: self.neural_pipeline.load_models())
            self.alert_manager.send_alert(
                f"NeuralPipeline chargé, policy_type={self.policy_type}", priority=1
            )
            send_telegram_alert(
                f"NeuralPipeline chargé, policy_type={self.policy_type}"
            )
            logger.info(f"NeuralPipeline chargé, policy_type={self.policy_type}")
        except Exception as e:
            error_msg = (
                f"Erreur chargement NeuralPipeline: {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.warning(error_msg)
            self.log_performance("init_neural_pipeline", 0, success=False, error=str(e))

        self.mean = (
            mean
            if mean is not None
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )
        self.std = (
            std
            if std is not None
            else np.ones(self.observation_space.shape, dtype=np.float32)
        )
        mean_path = self.save_dir / "norm_mean.npy"
        std_path = self.save_dir / "norm_std.npy"
        if mean_path.exists() and std_path.exists():
            try:
                def load_stats():
                    self.mean = np.load(mean_path)
                    self.std = np.load(std_path)

                self.with_retries(load_stats)
                self.alert_manager.send_alert(
                    f"Statistiques normalisation chargées: mean_shape={self.mean.shape}",
                    priority=1,
                )
                send_telegram_alert(
                    f"Statistiques normalisation chargées: mean_shape={self.mean.shape}"
                )
                logger.info(
                    f"Statistiques normalisation chargées: mean_shape={self.mean.shape}"
                )
            except Exception as e:
                error_msg = (
                    f"Erreur chargement stats: {str(e)}\n{traceback.format_exc()}"
                )
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                logger.warning(error_msg)
                self.log_performance(
                    "load_normalization_stats", 0, success=False, error=str(e)
                )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.env.observation_space.shape,
            dtype=np.float32,
        )
        self.alert_manager.send_alert(
            f"NormalizationWrapper initialisé: shape={self.observation_space.shape}",
            priority=1,
        )
        send_telegram_alert(
            f"NormalizationWrapper initialisé: shape={self.observation_space.shape}"
        )
        logger.info(
            f"NormalizationWrapper initialisé: shape={self.observation_space.shape}"
        )
        self.log_performance(
            "init_normalization", time.time() - start_time, success=True
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Normalise l'observation avec la moyenne et l'écart-type."""
        start_time = time.time()
        try:
            if not np.all(np.isfinite(observation)):
                error_msg = "Observation contient NaN/inf, remplacement par 0"
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                logger.warning(error_msg)
                observation = np.nan_to_num(
                    observation, nan=0.0, posinf=0.0, neginf=0.0
                )

            normalized_obs = (observation - self.mean) / np.maximum(
                self.std, self.min_std
            )
            self.log_performance(
                "observation_normalization", time.time() - start_time, success=True
            )
            return normalized_obs
        except Exception as e:
            error_msg = f"Erreur observation: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "observation_normalization", 0, success=False, error=str(e)
            )
            return np.zeros(self.observation_space.shape, dtype=np.float32)

    def update_statistics(self, data: pd.DataFrame) -> None:
        """Met à jour les statistiques de normalisation à partir des données."""
        start_time = time.time()
        try:
            if data is None or data.empty:
                error_msg = "Données vides, stats non mises à jour"
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                logger.warning(error_msg)
                return

            # Valider les données via obs_t
            if not validate_obs_t(data, context="gym_wrappers"):
                error_msg = "Échec de la validation obs_t pour les données"
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            # Valider les colonnes critiques via feature_sets.yaml
            feature_sets = self.with_retries(
                lambda: config_manager.get_config(
                    os.path.relpath(BASE_DIR / "config" / "feature_sets.yaml", BASE_DIR)
                )
            )
            critical_cols = [
                "iv_rank_30d",
                "call_wall",
                "put_wall",
                "zero_gamma",
                "dealer_position_bias",
                "bid_ask_imbalance",
                "iv_skew",
                "iv_term_structure",
                "trade_aggressiveness",
                "option_skew",
                "news_impact_score",
            ]
            expected_cols = feature_sets.get(
                (
                    "training_features"
                    if self.config.get("environment", {}).get("training_mode", True)
                    else "shap_features"
                ),
                [],
            )
            missing_cols = [
                col
                for col in critical_cols
                if col in expected_cols and col not in data.columns
            ]
            if missing_cols:
                error_msg = (
                    f"Colonnes critiques manquantes dans les données: {missing_cols}"
                )
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            # Imputation et clipping des colonnes critiques
            for col in critical_cols:
                if col in data.columns:
                    if col == "iv_rank_30d":
                        data[col] = data[col].clip(0, 100).fillna(50)
                    elif col in ["call_wall", "put_wall", "zero_gamma"]:
                        data[col] = (
                            data[col].clip(lower=0).fillna(data["close"].median())
                        )
                    elif col == "dealer_position_bias":
                        data[col] = data[col].clip(-1, 1).fillna(0)
                    elif col in ["bid_ask_imbalance", "trade_aggressiveness", "option_skew"]:
                        data[col] = data[col].clip(-1, 1).fillna(0)
                    elif col == "iv_skew":
                        data[col] = data[col].clip(-0.5, 0.5).fillna(0)
                    elif col == "iv_term_structure":
                        data[col] = data[col].clip(0, 0.05).fillna(0)
                    elif col == "news_impact_score":
                        data[col] = data[col].clip(-1, 1).fillna(0)

            obs_cols = [col for col in data.columns if col in expected_cols][
                : self.observation_space.shape[-1]
            ]
            if data[obs_cols].isna().any().any():
                error_msg = "NaN détectés, imputation par moyenne"
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                logger.warning(error_msg)
                data[obs_cols] = data[obs_cols].fillna(data[obs_cols].mean())

            # Pré-calcul des statistiques pour les features stables
            stable_cols = ["iv_rank_30d", "call_wall", "put_wall"]
            if all(col in obs_cols for col in stable_cols):
                stable_data = data[stable_cols]
                stable_mean = stable_data.mean().values.astype(np.float32)
                stable_std = np.maximum(
                    stable_data.std().values.astype(np.float32), self.min_std
                )
                for i, col in enumerate(stable_cols):
                    idx = obs_cols.index(col)
                    self.mean[idx] = stable_mean[i]
                    self.std[idx] = stable_std[i]

            def update_stats():
                self.mean = data[obs_cols].mean().values.astype(np.float32)
                self.std = np.maximum(
                    data[obs_cols].std().values.astype(np.float32), self.min_std
                )
                np.save(self.save_dir / "norm_mean.npy", self.mean)
                np.save(self.save_dir / "norm_std.npy", self.std)

            self.with_retries(update_stats)
            self.log_performance(
                "update_statistics_normalization",
                time.time() - start_time,
                success=True,
            )
            self.alert_manager.send_alert(
                f"Statistiques normalisation mises à jour: shape={self.mean.shape}",
                priority=1,
            )
            send_telegram_alert(
                f"Statistiques normalisation mises à jour: shape={self.mean.shape}"
            )
        except Exception as e:
            error_msg = f"Erreur update_statistics: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "update_statistics_normalization", 0, success=False, error=str(e)
            )

    def update_incremental(self, observation: np.ndarray, alpha: float = None) -> None:
        """Met à jour les statistiques de manière incrémentale."""
        start_time = time.time()
        try:
            alpha = alpha if alpha is not None else self.alpha
            if not np.all(np.isfinite(observation)):
                error_msg = "Observation incrémentale NaN/inf, ignorée"
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                logger.warning(error_msg)
                return
            if observation.shape != self.observation_space.shape:
                error_msg = f"Taille observation incrémentale incorrecte: {observation.shape} au lieu de {self.observation_space.shape}"
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)
            delta = observation - self.mean
            self.mean += alpha * delta
            self.std = np.sqrt(self.std**2 + alpha * (delta**2 - self.std**2))
            self.std = np.maximum(self.std, self.min_std)
            self.log_performance(
                "update_incremental_normalization",
                time.time() - start_time,
                success=True,
            )
            self.alert_manager.send_alert(
                "Statistiques incrémentales mises à jour", priority=1
            )
            send_telegram_alert("Statistiques incrémentales mises à jour")
        except Exception as e:
            error_msg = f"Erreur update_incremental: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "update_incremental_normalization", 0, success=False, error=str(e)
            )


class ClippingWrapper(BaseEnvWrapper, ObservationWrapper):
    """Wrapper pour limiter les valeurs extrêmes des observations."""

    def __init__(
        self,
        env: Env,
        clip_min: float = None,
        clip_max: float = None,
        config_path: str = str(BASE_DIR / "config" / "trading_env_config.yaml"),
        policy_type: str = "mlp",
    ):
        super().__init__(env, config_path)
        self.policy_type = policy_type
        start_time = time.time()
        try:
            self.clip_min = (
                clip_min
                if clip_min is not None
                else self.config.get("clipping", {}).get("clip_min", -5.0)
            )
            self.clip_max = (
                clip_max
                if clip_max is not None
                else self.config.get("clipping", {}).get("clip_max", 5.0)
            )
            self.option_columns = self.config.get("clipping", {}).get(
                "option_columns",
                {
                    "iv_rank_30d": {"min": 0, "max": 100},
                    "call_wall": {"min": 0, "max": float("inf")},
                    "put_wall": {"min": 0, "max": float("inf")},
                    "zero_gamma": {"min": 0, "max": float("inf")},
                    "dealer_position_bias": {"min": -1, "max": 1},
                    "bid_ask_imbalance": {"min": -1, "max": 1},
                    "iv_skew": {"min": -0.5, "max": 0.5},
                    "iv_term_structure": {"min": 0, "max": 0.05},
                    "trade_aggressiveness": {"min": -1, "max": 1},
                    "option_skew": {"min": -1, "max": 1},
                    "news_impact_score": {"min": -1, "max": 1},
                },
            )
            if self.clip_min >= self.clip_max:
                raise ValueError(
                    f"clip_min ({self.clip_min}) doit être inférieur à clip_max ({self.clip_max})"
                )
            self.alert_manager.send_alert(
                f"ClippingWrapper configuré: min={self.clip_min}, max={self.clip_max}",
                priority=1,
            )
            send_telegram_alert(
                f"ClippingWrapper configuré: min={self.clip_min}, max={self.clip_max}"
            )
            logger.info(
                f"ClippingWrapper configuré: min={self.clip_min}, max={self.clip_max}, option_columns={self.option_columns}"
            )
        except Exception as e:
            error_msg = f"Erreur configuration: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.warning(error_msg)
            self.clip_min = clip_min if clip_min is not None else -5.0
            self.clip_max = clip_max if clip_max is not None else 5.0
            self.option_columns = {
                "iv_rank_30d": {"min": 0, "max": 100},
                "call_wall": {"min": 0, "max": float("inf")},
                "put_wall": {"min": 0, "max": float("inf")},
                "zero_gamma": {"min": 0, "max": float("inf")},
                "dealer_position_bias": {"min": -1, "max": 1},
                "bid_ask_imbalance": {"min": -1, "max": 1},
                "iv_skew": {"min": -0.5, "max": 0.5},
                "iv_term_structure": {"min": 0, "max": 0.05},
                "trade_aggressiveness": {"min": -1, "max": 1},
                "option_skew": {"min": -1, "max": 1},
                "news_impact_score": {"min": -1, "max": 1},
            }
            self.log_performance("init_clipping", 0, success=False, error=str(e))

        self.observation_space = spaces.Box(
            low=self.clip_min,
            high=self.clip_max,
            shape=self.env.observation_space.shape,
            dtype=np.float32,
        )
        self.alert_manager.send_alert(
            f"ClippingWrapper initialisé: shape={self.observation_space.shape}",
            priority=1,
        )
        send_telegram_alert(
            f"ClippingWrapper initialisé: shape={self.observation_space.shape}"
        )
        logger.info(
            f"ClippingWrapper initialisé: min={self.clip_min}, max={self.clip_max}, shape={self.observation_space.shape}"
        )
        self.log_performance("init_clipping", time.time() - start_time, success=True)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Limite les valeurs de l'observation entre clip_min et clip_max."""
        start_time = time.time()
        try:
            if not np.all(np.isfinite(observation)):
                error_msg = "Observation contient NaN/inf, remplacement par clip_min"
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                logger.warning(error_msg)
                observation = np.nan_to_num(
                    observation,
                    nan=self.clip_min,
                    posinf=self.clip_max,
                    neginf=self.clip_min,
                )

            clipped_obs = np.clip(observation, self.clip_min, self.clip_max)
            self.log_performance(
                "observation_clipping", time.time() - start_time, success=True
            )
            return clipped_obs
        except Exception as e:
            error_msg = f"Erreur observation: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("observation_clipping", 0, success=False, error=str(e))
            return np.zeros(self.observation_space.shape, dtype=np.float32)


class RangeEnvWrapper(BaseEnvWrapper):
    """Wrapper pour le régime Range, ajuste observations et récompenses."""

    def __init__(
        self,
        env: Env,
        regime_probs: Dict[str, float],
        config_path: str = str(BASE_DIR / "config" / "trading_env_config.yaml"),
    ):
        super().__init__(env, config_path)
        start_time = time.time()
        try:
            self.weights = regime_probs.get("range", 0.5)
            if not 0 <= self.weights <= 1:
                raise ValueError(f"Poids du régime range invalide: {self.weights}")
            self.alert_manager.send_alert(
                f"RangeEnvWrapper initialisé: weights={self.weights}, CPU: {psutil.cpu_percent()}%",
                priority=1,
            )
            send_telegram_alert(
                f"RangeEnvWrapper initialisé: weights={self.weights}, CPU: {psutil.cpu_percent()}%"
            )
            logger.info(
                f"RangeEnvWrapper initialisé: weights={self.weights}, CPU: {psutil.cpu_percent()}%"
            )
        except Exception as e:
            error_msg = f"Erreur initialisation RangeEnvWrapper: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.weights = 0.5
            self.log_performance("init_range_wrapper", 0, success=False, error=str(e))
        self.log_performance(
            "init_range_wrapper", time.time() - start_time, success=True
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Ajuste les récompenses selon le régime Range et regime_probs (méthode 11)."""
        start_time = time.time()
        try:
            obs, reward, done, truncated, info = self.env.step(action)
            # Ajustement dynamique des poids
            self.weights = min(max(self.weights * info.get("regime_probs", {}).get("range", 0.5), 0.1), 1.0)
            reward *= self.weights
            if (
                info.get("call_wall", float("inf"))
                > info.get("price", 0)
                > info.get("put_wall", 0)
            ):
                reward *= 1.5 * self.weights
            # Ajustements basés sur les nouvelles fonctionnalités
            if "bid_ask_imbalance" in info and info["bid_ask_imbalance"] > 0.3:
                reward *= 1.2
            if "trade_aggressiveness" in info and info["trade_aggressiveness"] > 0.3:
                reward *= 1.1
            if "news_impact_score" in info and info["news_impact_score"] > 0.5 and reward > 0:
                reward *= 1.3
            self.log_performance(
                "step_range_wrapper", time.time() - start_time, success=True
            )
            return obs, reward, done, truncated, info
        except Exception as e:
            error_msg = (
                f"Erreur step RangeEnvWrapper: {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("step_range_wrapper", 0, success=False, error=str(e))
            return obs, 0.0, True, False, info


class TrendEnvWrapper(BaseEnvWrapper):
    """Wrapper pour le régime Trend, ajuste observations et récompenses."""

    def __init__(
        self,
        env: Env,
        regime_probs: Dict[str, float],
        config_path: str = str(BASE_DIR / "config" / "trading_env_config.yaml"),
    ):
        super().__init__(env, config_path)
        start_time = time.time()
        try:
            self.weights = regime_probs.get("trend", 0.5)
            if not 0 <= self.weights <= 1:
                raise ValueError(f"Poids du régime trend invalide: {self.weights}")
            self.alert_manager.send_alert(
                f"TrendEnvWrapper initialisé: weights={self.weights}, CPU: {psutil.cpu_percent()}%",
                priority=1,
            )
            send_telegram_alert(
                f"TrendEnvWrapper initialisé: weights={self.weights}, CPU: {psutil.cpu_percent()}%"
            )
            logger.info(
                f"TrendEnvWrapper initialisé: weights={self.weights}, CPU: {psutil.cpu_percent()}%"
            )
        except Exception as e:
            error_msg = f"Erreur initialisation TrendEnvWrapper: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.weights = 0.5
            self.log_performance("init_trend_wrapper", 0, success=False, error=str(e))
        self.log_performance(
            "init_trend_wrapper", time.time() - start_time, success=True
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Ajuste les récompenses selon le régime Trend et regime_probs (méthode 11)."""
        start_time = time.time()
        try:
            obs, reward, done, truncated, info = self.env.step(action)
            # Ajustement dynamique des poids
            self.weights = min(max(self.weights * info.get("regime_probs", {}).get("trend", 0.5), 0.1), 1.0)
            reward *= self.weights
            if info.get("profit", 0) > 0:
                reward *= 1.5 * self.weights
            else:
                reward *= 0.5
            # Ajustements basés sur les nouvelles fonctionnalités
            if "bid_ask_imbalance" in info and info["bid_ask_imbalance"] > 0.3:
                reward *= 1.2
            if "trade_aggressiveness" in info and info["trade_aggressiveness"] > 0.3:
                reward *= 1.1
            if "news_impact_score" in info and info["news_impact_score"] > 0.5 and reward > 0:
                reward *= 1.3
            self.log_performance(
                "step_trend_wrapper", time.time() - start_time, success=True
            )
            return obs, reward, done, truncated, info
        except Exception as e:
            error_msg = (
                f"Erreur step TrendEnvWrapper: {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("step_trend_wrapper", 0, success=False, error=str(e))
            return obs, 0.0, True, False, info


class DefensiveEnvWrapper(BaseEnvWrapper):
    """Wrapper pour le régime Défensif, ajuste observations et récompenses."""

    def __init__(
        self,
        env: Env,
        regime_probs: Dict[str, float],
        config_path: str = str(BASE_DIR / "config" / "trading_env_config.yaml"),
    ):
        super().__init__(env, config_path)
        start_time = time.time()
        try:
            self.weights = regime_probs.get("defensive", 0.5)
            if not 0 <= self.weights <= 1:
                raise ValueError(f"Poids du régime défensif invalide: {self.weights}")
            self.alert_manager.send_alert(
                f"DefensiveEnvWrapper initialisé: weights={self.weights}, CPU: {psutil.cpu_percent()}%",
                priority=1,
            )
            send_telegram_alert(
                f"DefensiveEnvWrapper initialisé: weights={self.weights}, CPU: {psutil.cpu_percent()}%"
            )
            logger.info(
                f"DefensiveEnvWrapper initialisé: weights={self.weights}, CPU: {psutil.cpu_percent()}%"
            )
        except Exception as e:
            error_msg = f"Erreur initialisation DefensiveEnvWrapper: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.weights = 0.5
            self.log_performance(
                "init_defensive_wrapper", 0, success=False, error=str(e)
            )
        self.log_performance(
            "init_defensive_wrapper", time.time() - start_time, success=True
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Ajuste les récompenses selon le régime Défensif et regime_probs (méthode 11)."""
        start_time = time.time()
        try:
            obs, reward, done, truncated, info = self.env.step(action)
            # Ajustement dynamique des poids
            self.weights = min(max(self.weights * info.get("regime_probs", {}).get("defensive", 0.5), 0.1), 1.0)
            reward *= self.weights
            if info.get("profit", 0) < 0:
                reward *= 0.5
            else:
                reward *= 1.0 * self.weights
            # Ajustements basés sur les nouvelles fonctionnalités
            if "bid_ask_imbalance" in info and info["bid_ask_imbalance"] > 0.3:
                reward *= 1.2
            if "trade_aggressiveness" in info and info["trade_aggressiveness"] > 0.3:
                reward *= 1.1
            if "news_impact_score" in info and info["news_impact_score"] > 0.5 and reward > 0:
                reward *= 1.3
            self.log_performance(
                "step_defensive_wrapper", time.time() - start_time, success=True
            )
            return obs, reward, done, truncated, info
        except Exception as e:
            error_msg = (
                f"Erreur step DefensiveEnvWrapper: {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "step_defensive_wrapper", 0, success=False, error=str(e)
            )
            return obs, 0.0, True, False, info


if __name__ == "__main__":
    import pandas as pd

    from src.envs.trading_env import TradingEnv

    try:
        env = TradingEnv()
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-05-15 09:00", periods=100, freq="1min"
                ),
                "close": np.random.normal(5100, 10, 100),
                "open": np.random.normal(5100, 10, 100),
                "high": np.random.normal(5105, 10, 100),
                "low": np.random.normal(5095, 10, 100),
                "volume": np.random.randint(100, 1000, 100),
                "atr_14": np.random.uniform(0.5, 2.0, 100),
                "adx_14": np.random.uniform(10, 40, 100),
                "gex": np.random.uniform(-1000, 1000, 100),
                "oi_peak_call_near": np.random.randint(5000, 15000, 100),
                "gamma_wall": np.random.uniform(5090, 5110, 100),
                "iv_rank_30d": np.random.uniform(50, 80, 100),
                "call_wall": np.random.uniform(5150, 5200, 100),
                "put_wall": np.random.uniform(5050, 5100, 100),
                "zero_gamma": np.random.uniform(5095, 5105, 100),
                "dealer_position_bias": np.random.uniform(-0.2, 0.2, 100),
                "predicted_volatility": np.random.uniform(0.1, 0.5, 100),
                "predicted_vix": np.random.uniform(15, 25, 100),
                "news_impact_score": np.random.uniform(-1, 1, 100),
                "bid_size_level_1": np.random.randint(50, 500, 100),
                "ask_size_level_1": np.random.randint(50, 500, 100),
                "spread_avg_1min": np.random.uniform(1.0, 3.0, 100),
                "bid_ask_imbalance": np.random.uniform(-1, 1, 100),
                "iv_skew": np.random.uniform(-0.5, 0.5, 100),
                "iv_term_structure": np.random.uniform(0, 0.05, 100),
                "trade_aggressiveness": np.random.uniform(-1, 1, 100),
                "option_skew": np.random.uniform(-1, 1, 100),
                **{
                    f"feature_{i}": np.random.uniform(0, 1, 100) for i in range(324)
                },  # Total 350 features
            }
        )
        env.data = data
        env.mode = "range"
        alert_manager = AlertManager()
        alert_manager.send_alert("Données simulées chargées pour test", priority=1)
        send_telegram_alert("Données simulées chargées pour test")
        logger.info("Données simulées chargées pour test")

        regime_probs = detect_regime(
            data, model_path=str(BASE_DIR / "data" / "models" / "regime_model.pkl")
        )

        for policy in ["mlp", "transformer"]:
            wrapped_env = ObservationStackingWrapper(env, policy_type=policy)
            wrapped_env = NormalizationWrapper(wrapped_env, policy_type=policy)
            wrapped_env.update_statistics(data)
            wrapped_env = ClippingWrapper(wrapped_env, policy_type=policy)
            wrapped_env = RangeEnvWrapper(wrapped_env, regime_probs)
            wrapped_env = TrendEnvWrapper(wrapped_env, regime_probs)
            wrapped_env = DefensiveEnvWrapper(wrapped_env, regime_probs)
            obs, info = wrapped_env.reset()
            alert_manager.send_alert(
                f"Policy: {policy}, Observation shape: {obs.shape}", priority=1
            )
            send_telegram_alert(f"Policy: {policy}, Observation shape: {obs.shape}")
            print(f"Policy: {policy}, Observation shape: {obs.shape}")
            action = np.array([0.6])
            obs, reward, done, _, info = wrapped_env.step(action)
            alert_manager.send_alert(
                f"Policy: {policy}, Récompense: {reward}, Info: {info}", priority=1
            )
            send_telegram_alert(f"Policy: {policy}, Récompense: {reward}, Info: {info}")
            print(f"Policy: {policy}, Récompense: {reward}, Info: {info}")
    except Exception as e:
        error_msg = f"Erreur test: {str(e)}\n{traceback.format_exc()}"
        AlertManager().send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)
        logger.error(error_msg)
        raise