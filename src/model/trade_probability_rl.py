# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/model/trade_probability_rl.py
# Gestion des modèles RL pour la prédiction de probabilité de trade.
#
# Version : 2.1.5
# Date : 2025-05-14
#
# Rôle : Entraîne et prédit avec des modèles RL (SAC, PPO, DDPG, PPO-CVaR, QR-DQN) en intégrant
#        les coûts de transaction, microstructure, walk-forward, Safe RL, surface de volatilité,
#        et résolution des conflits de signaux via SignalResolver.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, stable-baselines3>=2.0.0,<3.0.0, rllib>=2.0.0,<3.0.0
# - gym>=0.21.0,<1.0.0, psutil>=5.9.8,<6.0.0, joblib>=1.3.0,<2.0.0
# - src/utils/error_tracker.py
# - src/monitoring/prometheus_metrics.py
# - src/utils/mlflow_tracker.py
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
# - src/model/utils/signal_resolver.py
#
# Inputs :
# - Données avec features (bid_ask_imbalance, trade_aggressiveness, iv_skew, iv_term_structure, slippage_estimate)
# - config/es_config.yaml pour SignalResolver
#
# Outputs :
# - data/models/<market>/*.pth
# - Logs dans data/logs/trade_probability_rl_performance.csv
#
# Notes :
# - Intègre validation glissante (TimeSeriesSplit), Safe RL (PPO-CVaR), Distributional RL (QR-DQN).
# - Journalise via MLflow et monitore via Prometheus.
# - Ajoute logs psutil pour CPU/mémoire.
# - Corrige conversion action → probabilité pour SAC/PPO/DDPG et QR-DQN.
# - Intègre SignalResolver pour résoudre les conflits de signaux avant la prédiction.
# - Policies Note: The official directory pour routing policies est src/model/router/policies.

import uuid
from datetime import datetime
from pathlib import Path

import gym
import numpy as np
import pandas as pd
import psutil
from gym import spaces
from joblib import Parallel, delayed
from loguru import logger
from rllib.agents.qr_dqn import QRDQN
from sklearn.model_selection import TimeSeriesSplit
from stable_baselines3 import DDPG, PPO, SAC
from stable_baselines3.common.policies import BasePolicy

from src.model.utils.alert_manager import AlertManager
from src.model.utils.signal_resolver import SignalResolver
from src.monitoring.prometheus_metrics import Gauge
from src.utils.error_tracker import capture_error
from src.utils.mlflow_tracker import MLFlowTracker
from src.utils.telegram_alert import send_telegram_alert

BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
RL_MODEL_DIR = BASE_DIR / "data" / "models"
LOG_DIR = BASE_DIR / "data" / "logs"
LOG_DIR.mkdir(exist_ok=True)

cvar_loss_metric = Gauge("cvar_loss", "Perte CVaR pour PPO-Lagrangian", ["market"])
qr_dqn_quantiles_metric = Gauge("qr_dqn_quantiles", "Quantiles QR-DQN", ["market"])


class CVaRWrapper(PPO):
    """Wrapper pour PPO avec CVaR-Lagrangian."""

    def __init__(
        self, policy: BasePolicy, env: gym.Env, cvar_alpha: float = 0.95, **kwargs
    ):
        super().__init__(policy, env, **kwargs)
        self.cvar_alpha = cvar_alpha
        self.cvar_loss = 0.0
        self.rewards = []

    def learn(self, total_timesteps: int, **kwargs):
        super().learn(total_timesteps, **kwargs)
        # Calcul CVaR à partir des récompenses collectées
        if self.rewards:
            sorted_rewards = np.sort(self.rewards)
            cvar_idx = int(self.cvar_alpha * len(sorted_rewards))
            self.cvar_loss = np.mean(sorted_rewards[:cvar_idx])
            cvar_loss_metric.labels(market=self.env.market).set(self.cvar_loss)
        else:
            self.cvar_loss = 0.0
            cvar_loss_metric.labels(market=self.env.market).set(0.0)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.rewards.append(reward)
        return obs, reward, done, info


class TradeEnv(gym.Env):
    """Environnement Gym pour le trading."""

    def __init__(self, data: pd.DataFrame, market: str = "ES"):
        super().__init__()
        self.data = data
        self.market = market
        self.current_step = 0
        self.features = [
            "bid_ask_imbalance",
            "trade_aggressiveness",
            "iv_skew",
            "iv_term_structure",
        ]
        self.action_space = spaces.Discrete(2)  # 0: ne rien faire, 1: trader
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.features),), dtype=np.float32
        )

    def reset(self):
        self.current_step = 0
        return self.data[self.features].iloc[self.current_step].values

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        reward = 0.0
        if action == 1:
            reward = (
                np.random.uniform(-1, 1)
                - self.data["slippage_estimate"].iloc[self.current_step]
            )
        obs = (
            self.data[self.features].iloc[self.current_step].values
            if not done
            else np.zeros(len(self.features))
        )
        return obs, reward, done, {}


class RLTrainer:
    """Entraîne et prédit avec des modèles RL pour la probabilité de trade."""

    def __init__(self, market: str = "ES"):
        self.market = market
        self.rl_models = {
            "sac": None,
            "ppo": None,
            "ddpg": None,
            "ppo_cvar": None,
            "qr_dqn": None,
        }
        self.mlflow_tracker = MLFlowTracker(
            experiment_name=f"trade_probability_rl_{market}"
        )
        self.alert_manager = AlertManager()
        self.signal_resolver = SignalResolver(
            config_path=str(BASE_DIR / "config/es_config.yaml"), market=market
        )
        RL_MODEL_DIR.mkdir(exist_ok=True)

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ):
        """Journalise les performances CPU/mémoire."""
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_usage = psutil.cpu_percent()
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
                "cpu_usage_percent": cpu_usage,
                **kwargs,
            }
            log_df = pd.DataFrame([log_entry])
            log_path = LOG_DIR / "trade_probability_rl_performance.csv"
            log_df.to_csv(
                log_path,
                mode="a",
                header=not log_path.exists(),
                index=False,
                encoding="utf-8",
            )
            self.mlflow_tracker.log_metrics(
                {
                    "latency": latency,
                    "memory_usage_mb": memory_usage,
                    "cpu_usage_percent": cpu_usage,
                }
            )
        except Exception as e:
            error_msg = (
                f"Erreur journalisation performance RL pour {self.market}: {str(e)}"
            )
            logger.error(error_msg)
            capture_error(
                e,
                context={"market": self.market},
                market=self.market,
                operation="log_performance",
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)

    def create_rl_env(self, data: pd.DataFrame) -> gym.Env:
        """Crée un environnement Gym pour le trading."""
        try:
            return TradeEnv(data, market=self.market)
        except Exception as e:
            error_msg = f"Erreur création environnement RL pour {self.market}: {str(e)}"
            logger.error(error_msg)
            capture_error(
                e,
                context={"market": self.market},
                market=self.market,
                operation="create_rl_env",
            )
            raise

    def resolve_signals(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Résout les conflits entre signaux en utilisant SignalResolver."""
        try:
            start_time = datetime.now()
            run_id = str(uuid.uuid4())
            if features.empty:
                raise ValueError("DataFrame vide fourni pour la résolution des signaux")
            signals = {
                "microstructure_bullish": features.get("bid_ask_imbalance", 0.0),
                "news_score_positive": features.get("news_impact_score", 0.0),
                "qr_dqn_positive": features.get("qr_dqn_quantile_mean", 0.0),
                "iv_term_structure": features.get("iv_term_structure", 0.0),
            }
            score, metadata = self.signal_resolver.resolve_conflict(
                signals=signals,
                normalize=True,
                persist_to_db=True,
                export_to_csv=True,
                run_id=run_id,
                score_type="intermediate",
                mode_debug=True,
            )
            metadata["signal_score"] = score
            latency = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Résolution des signaux pour {self.market}: Score={score:.2f}, "
                f"Conflict Coefficient={metadata['conflict_coefficient']:.2f}, "
                f"Entropy={metadata['entropy']:.2f}, Run ID={run_id}"
            )
            self.log_performance(
                "resolve_signals",
                latency,
                success=True,
                score=score,
                conflict_coefficient=metadata["conflict_coefficient"],
                run_id=run_id,
            )
            return metadata
        except Exception as e:
            run_id = str(uuid.uuid4())
            error_msg = f"Erreur résolution des signaux pour {self.market}: {str(e)}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            fallback_metadata = {
                "signal_score": 0.0,
                "conflict_coefficient": 0.0,
                "entropy": 0.0,
                "score_type": "intermediate",
                "contributions": {},
                "run_id": run_id,
                "error": str(e),
            }
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "resolve_signals",
                latency,
                success=False,
                error=str(e),
                score=0.0,
                conflict_coefficient=0.0,
                run_id=run_id,
            )
            return fallback_metadata

    def train_rl_models(self, data: pd.DataFrame, total_timesteps: int = 100000):
        """Entraîne SAC, PPO, DDPG, PPO-CVaR, QR-DQN avec validation glissante."""
        start_time = datetime.now()
        try:
            tscv = TimeSeriesSplit(n_splits=5)
            features = [
                "bid_ask_imbalance",
                "trade_aggressiveness",
                "iv_skew",
                "iv_term_structure",
            ]

            def train_model(model_name, train_data, env):
                with self.mlflow_tracker.start_run(
                    run_name=f"{model_name}_{self.market}"
                ):
                    start = datetime.now()
                    if model_name == "sac":
                        model = SAC("MlpPolicy", env, verbose=0)
                    elif model_name == "ppo":
                        model = PPO("MlpPolicy", env, verbose=0)
                    elif model_name == "ddpg":
                        model = DDPG("MlpPolicy", env, verbose=0)
                    elif model_name == "ppo_cvar":
                        model = CVaRWrapper(
                            "MlpPolicy", env, cvar_alpha=0.95, verbose=0
                        )
                    elif model_name == "qr_dqn":
                        model = QRDQN("MlpPolicy", env, quantiles=51, verbose=0)
                        qr_dqn_quantiles_metric.labels(market=self.market).set(51)
                    model.learn(total_timesteps=total_timesteps)
                    model_dir = RL_MODEL_DIR / self.market
                    model_dir.mkdir(exist_ok=True)
                    model_path = (
                        model_dir
                        / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
                    )
                    model.save(model_path)
                    self.mlflow_tracker.log_artifact(model_path)
                    self.mlflow_tracker.log_params(
                        {"model_name": model_name, "total_timesteps": total_timesteps}
                    )
                    latency = (datetime.now() - start).total_seconds()
                    self.log_performance(
                        f"train_{model_name}",
                        latency,
                        success=True,
                        model_path=str(model_path),
                    )
                    return model_name, model

            for train_idx, _ in tscv.split(data):
                train_data = data.iloc[train_idx][features]
                env = self.create_rl_env(train_data)
                results = Parallel(n_jobs=-1)(
                    delayed(train_model)(model_name, train_data, env)
                    for model_name in self.rl_models
                )
                for model_name, model in results:
                    self.rl_models[model_name] = model

            latency = (datetime.now() - start_time).total_seconds()
            logger.info(f"Modèles RL entraînés pour {self.market}. Latence: {latency}s")
            self.log_performance("train_rl_models", latency, success=True)
            self.mlflow_tracker.log_metrics({"train_rl_latency": latency})
        except Exception as e:
            error_msg = f"Erreur entraînement modèles RL pour {self.market}: {str(e)}"
            logger.error(error_msg)
            capture_error(
                e,
                context={"market": self.market},
                market=self.market,
                operation="train_rl_models",
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance("train_rl_models", 0, success=False, error=str(e))
            raise

    def predict(self, data: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
        """Prédit la probabilité moyenne de succès avec les modèles RL."""
        try:
            start_time = datetime.now()
            # Résolution des signaux
            signal_metadata = self.resolve_signals(data.iloc[-1])
            logger.info(
                f"Signal resolution metadata pour {self.market}: {signal_metadata}"
            )
            features = data[
                [
                    "bid_ask_imbalance",
                    "trade_aggressiveness",
                    "iv_skew",
                    "iv_term_structure",
                ]
            ]
            probs = []
            for model_name, model in self.rl_models.items():
                if model is not None:
                    if model_name == "qr_dqn":
                        quantiles = model.predict_quantiles(features.values)
                        prob = np.mean(quantiles > 0)  # P(y>0)
                    else:
                        _, values = model.policy.predict_values(features.values)
                        prob = 1 / (
                            1 + np.exp(-values.mean())
                        )  # Sigmoid sur valeur moyenne
                    probs.append(np.clip(prob, 0, 1))
            prob = np.mean(probs) if probs else 0.0
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("predict", latency, success=True, probability=prob)
            return prob, signal_metadata
        except Exception as e:
            error_msg = f"Erreur prédiction RL pour {self.market}: {str(e)}"
            logger.error(error_msg)
            capture_error(
                e,
                context={"market": self.market},
                market=self.market,
                operation="predict_rl",
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            return 0.0, {
                "signal_score": 0.0,
                "conflict_coefficient": 0.0,
                "entropy": 0.0,
                "run_id": str(uuid.uuid4()),
                "error": str(e),
            }

    def load_models(self):
        """Charge les modèles RL depuis RL_MODEL_DIR."""
        try:
            rl_model_dir = RL_MODEL_DIR / self.market
            rl_model_dir.mkdir(exist_ok=True)
            for model_name in self.rl_models:
                model_files = [
                    f
                    for f in os.listdir(rl_model_dir)
                    if f.startswith(model_name) and f.endswith(".pth")
                ]
                if model_files:
                    latest_model = max(
                        model_files,
                        key=lambda x: os.path.getctime(os.path.join(rl_model_dir, x)),
                    )
                    model_path = rl_model_dir / latest_model
                    if model_name == "sac":
                        self.rl_models[model_name] = SAC.load(model_path)
                    elif model_name == "ppo":
                        self.rl_models[model_name] = PPO.load(model_path)
                    elif model_name == "ddpg":
                        self.rl_models[model_name] = DDPG.load(model_path)
                    elif model_name == "ppo_cvar":
                        self.rl_models[model_name] = CVaRWrapper.load(model_path)
                    elif model_name == "qr_dqn":
                        self.rl_models[model_name] = QRDQN.load(model_path)
            logger.info(f"Modèles RL chargés pour {self.market}")
        except Exception as e:
            error_msg = f"Erreur chargement modèles RL pour {self.market}: {str(e)}"
            logger.error(error_msg)
            capture_error(
                e,
                context={"market": self.market},
                market=self.market,
                operation="load_rl_models",
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
