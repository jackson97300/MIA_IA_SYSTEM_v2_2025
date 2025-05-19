# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/model/utils/hyperparam_optimizer.py
# Optimisation des hyperparamètres avec Optuna pour MIA_IA_SYSTEM_v2_2025
#
# Version: 2.1.5
# Date: 2025-05-14
#
# Rôle: Optimise les hyperparamètres des modèles SAC, PPO, DDPG, PPO-Lagrangian, QR-DQN, et ensembles de politiques
#       avec Optuna pour maximiser les performances. Journalise les meilleurs hyperparamètres dans market_memory.db.
#
# Utilisé par: train_pipeline.py, trade_probability.py.
#
# Notes:
# - Conforme à structure.txt (version 2.1.5, 2025-05-14).
# - Supporte les suggestions 3 (simulation configurable), 7 (Safe RL/CVaR-PPO), 8 (Distributional RL/QR-DQN),
#   9 (réentraînement), 10 (Ensembles de politiques).
# - Intègre avec mlflow_tracker.py, prometheus_metrics.py, market_memory.db.
# - Remplace scikit-optimize pour une optimisation avancée.
# - Intègre logs psutil dans data/logs/hyperparam_optimizer_performance.csv.
# - Utilise error_tracker.py pour capturer les erreurs et alert_manager.py pour les alertes.
# - Journalise les meilleurs hyperparamètres dans la table best_hyperparams de market_memory.db.
# - Valide les paramètres avant d’appeler quick_train_and_eval pour éviter les erreurs.
# - Pas de références à dxFeed, obs_t, 320 features, ou 81 features.
# - Policies Note: The official directory for routing policies is src/model/router/policies.
# The src/model/policies directory is a residual and should be verified
# for removal to avoid import conflicts.

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import optuna
import pandas as pd
import psutil
from loguru import logger

from src.model.utils.alert_manager import AlertManager
from src.model.utils.mlflow_tracker import MLflowTracker
from src.monitoring.prometheus_metrics import Gauge
from src.utils.error_tracker import capture_error

# Configuration du logging Loguru
logger.remove()
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOG_DIR = BASE_DIR / "data" / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "hyperparam_optimizer.log",
    rotation="10 MB",
    level="INFO",
    encoding="utf-8",
)

# Configuration du logging standard pour compatibilité
logging.basicConfig(
    filename=str(LOG_DIR / "hyperparam_optimizer.log"),
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)

# Métrique Prometheus pour la charge d’optimisation
optimization_load = Gauge(
    "optimization_load_percent",
    "Charge système pendant l’optimisation des hyperparamètres",
    ["market"],
)


class HyperparamOptimizer:
    """Optimise les hyperparamètres avec Optuna."""

    def __init__(self, market: str = "ES", storage: str = None):
        """Initialise l’optimiseur Optuna.

        Args:
            market: Marché cible (défaut: 'ES').
            storage: URI de stockage SQLite (défaut: None, utilise mémoire).

        Raises:
            ValueError: Si l’initialisation d’Optuna échoue.
        """
        start_time = datetime.now()
        self.market = market
        self.alert_manager = AlertManager()
        try:
            storage = storage or f"sqlite:///{BASE_DIR}/data/market_memory.db"
            self.study = optuna.create_study(
                study_name=f"online_hpo_{market}",
                storage=storage,
                directions=["maximize", "minimize"],  # Sharpe, drawdown
                load_if_exists=True,
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
                sampler=optuna.samplers.TPESampler(n_startup_trials=10),
            )
            self._load_previous_best()
            logger.info(f"Étude Optuna initialisée pour {market}")
            logging.info(f"Étude Optuna initialisée pour {market}")
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("init_optuna", latency, success=True)
        except Exception as e:
            error_msg = (
                f"Erreur lors de l’initialisation d’Optuna pour {market}: {str(e)}"
            )
            logger.error(error_msg)
            logging.error(error_msg)
            capture_error(
                e, context={"market": market}, market=market, operation="init_optuna"
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("init_optuna", latency, success=False, error=str(e))
            raise ValueError(f"Échec de l’initialisation d’Optuna: {str(e)}")

    def _load_previous_best(self) -> None:
        """Charge les meilleurs hyperparamètres précédents depuis market_memory.db."""
        start_time = datetime.now()
        try:
            db_path = BASE_DIR / "data" / "market_memory.db"
            with sqlite3.connect(db_path) as conn:
                result = conn.execute(
                    "SELECT parameters FROM mlflow_runs WHERE metrics->>'sharpe' IS NOT NULL ORDER BY metrics->>'sharpe' DESC LIMIT 1"
                ).fetchone()
                if result:
                    best_params = json.loads(result[0])
                    self.study.enqueue_trial(best_params)
                    logger.debug(
                        f"Meilleurs hyperparamètres précédents chargés: {best_params}"
                    )
                    latency = (datetime.now() - start_time).total_seconds()
                    self.log_performance("load_previous_best", latency, success=True)
        except sqlite3.Error as e:
            error_msg = (
                f"Erreur lors du chargement des hyperparamètres précédents: {str(e)}"
            )
            logger.warning(error_msg)
            logging.warning(error_msg)
            capture_error(
                e,
                context={"market": self.market},
                market=self.market,
                operation="load_previous_best",
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "load_previous_best", latency, success=False, error=str(e)
            )

    def _validate_params(self, params: Dict, model_name: str) -> None:
        """Valide les paramètres avant d’appeler quick_train_and_eval.

        Args:
            params: Dictionnaire des paramètres.
            model_name: Nom du modèle.

        Raises:
            ValueError: Si des paramètres sont manquants ou invalides.
        """
        expected_params = {
            "ppo_cvar": {"cvar_alpha", "learning_rate", "gamma", "batch_size"},
            "qr_dqn": {"quantiles", "learning_rate", "gamma", "batch_size"},
            "ensemble": {"sac_weight", "ppo_weight", "ddpg_weight", "learning_rate"},
            "sac": {
                "learning_rate",
                "gamma",
                "batch_size",
                "n_layers",
                "hidden_dim",
                "tau",
                "entropy_coef",
                "actor_lr",
                "critic_lr",
                "n_steps",
            },
            "ppo": {
                "learning_rate",
                "gamma",
                "batch_size",
                "n_layers",
                "hidden_dim",
                "clip_ratio",
                "entropy_coef",
                "actor_lr",
                "critic_lr",
                "n_steps",
            },
            "ddpg": {
                "learning_rate",
                "gamma",
                "batch_size",
                "n_layers",
                "hidden_dim",
                "tau",
                "actor_lr",
                "critic_lr",
                "n_steps",
            },
        }
        required = expected_params.get(model_name, expected_params["sac"])
        missing = required - set(params.keys())
        if missing:
            raise ValueError(f"Paramètres manquants pour {model_name}: {missing}")
        if model_name == "ensemble":
            total_weight = (
                params["sac_weight"] + params["ppo_weight"] + params["ddpg_weight"]
            )
            if not abs(total_weight - 1.0) < 1e-6:
                raise ValueError(
                    f"Les poids bayésiens doivent sommer à 1.0, obtenu: {total_weight}"
                )

    def objective(self, trial: optuna.Trial, model_name: str) -> Tuple[float, float]:
        """Définit l’objectif d’optimisation.

        Args:
            trial: Essai Optuna.
            model_name: Nom du modèle (ex. : 'sac', 'ppo', 'ddpg', 'ppo_cvar', 'qr_dqn', 'ensemble').

        Returns:
            Tuple[float, float]: Sharpe ratio (maximiser), drawdown (minimiser).
        """
        start_time = datetime.now()
        try:
            # Espace de recherche par modèle
            if model_name == "ppo_cvar":  # Suggestion 7
                params = {
                    "cvar_alpha": trial.suggest_float("cvar_alpha", 0.9, 0.99),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 1e-5, 1e-3, log=True
                    ),
                    "gamma": trial.suggest_float("gamma", 0.90, 0.999),
                    "batch_size": trial.suggest_int("batch_size", 64, 512, step=64),
                }
            elif model_name == "qr_dqn":  # Suggestion 8
                params = {
                    "quantiles": trial.suggest_int("quantiles", 10, 100),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 1e-5, 1e-3, log=True
                    ),
                    "gamma": trial.suggest_float("gamma", 0.90, 0.999),
                    "batch_size": trial.suggest_int("batch_size", 32, 256, step=32),
                }
            elif model_name == "ensemble":  # Suggestion 10
                sac_weight = trial.suggest_float("sac_weight", 0.1, 0.8)
                ppo_weight = trial.suggest_float("ppo_weight", 0.1, 0.9 - sac_weight)
                ddpg_weight = 1.0 - sac_weight - ppo_weight
                params = {
                    "sac_weight": sac_weight,
                    "ppo_weight": ppo_weight,
                    "ddpg_weight": ddpg_weight,
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 1e-5, 1e-3, log=True
                    ),
                }
            else:  # SAC, PPO, DDPG
                params = {
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 1e-5, 1e-2, log=True
                    ),
                    "gamma": trial.suggest_float("gamma", 0.90, 0.999),
                    "batch_size": trial.suggest_int("batch_size", 64, 512, step=64),
                    "n_layers": trial.suggest_int("n_layers", 1, 4),
                    "hidden_dim": trial.suggest_int("hidden_dim", 32, 512, step=32),
                    "tau": trial.suggest_float("tau", 0.005, 0.01),
                    "clip_ratio": trial.suggest_float("clip_ratio", 0.1, 0.3),  # PPO
                    "entropy_coef": trial.suggest_float("entropy_coef", 0.0, 0.01),
                    "actor_lr": trial.suggest_float("actor_lr", 1e-5, 1e-2, log=True),
                    "critic_lr": trial.suggest_float("critic_lr", 1e-5, 1e-2, log=True),
                    "n_steps": trial.suggest_int("n_steps", 100, 1000, step=100),
                }

            # Valider les paramètres
            self._validate_params(params, model_name)

            # Simuler l’entraînement rapide
            from src.model.trade_probability import quick_train_and_eval

            metrics = quick_train_and_eval(
                **params, market=self.market, model_name=model_name
            )

            # Journaliser l’essai avec MLflow
            MLflowTracker().log_run(model_name, metrics, params, [], market=self.market)

            # Monitorer la charge système
            optimization_load.labels(market=self.market).set(
                metrics.get("cpu_usage", 0)
            )

            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "objective", latency, success=True, model_name=model_name
            )
            return metrics["sharpe"], metrics["drawdown"]
        except Exception as e:
            error_msg = f"Erreur dans l’essai Optuna pour {model_name}: {str(e)}"
            logger.error(error_msg)
            logging.error(error_msg)
            capture_error(
                e,
                context={"market": self.market, "model_name": model_name},
                market=self.market,
                operation="optuna_trial",
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "objective", latency, success=False, error=str(e), model_name=model_name
            )
            return 0.0, float("inf")  # Valeurs par défaut en cas d’échec

    def _log_best_to_db(self, model_name: str, best_params: Dict) -> None:
        """Enregistre les meilleurs hyperparamètres dans la table best_hyperparams de market_memory.db.

        Args:
            model_name: Nom du modèle.
            best_params: Dictionnaire des meilleurs hyperparamètres.
        """
        start_time = datetime.now()
        try:
            db_path = BASE_DIR / "data" / "market_memory.db"
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS best_hyperparams (
                        model_name TEXT,
                        parameters TEXT,
                        sharpe REAL,
                        drawdown REAL,
                        timestamp TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO best_hyperparams (model_name, parameters, sharpe, drawdown, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        model_name,
                        json.dumps(best_params),
                        best_params.get("sharpe", 0.0),
                        best_params.get("drawdown", 0.0),
                        datetime.now().isoformat(),
                    ),
                )
                conn.commit()
                logger.debug(
                    f"Meilleurs hyperparamètres enregistrés dans best_hyperparams pour {model_name}"
                )
                latency = (datetime.now() - start_time).total_seconds()
                self.log_performance(
                    "log_best_to_db", latency, success=True, model_name=model_name
                )
        except sqlite3.Error as e:
            error_msg = f"Erreur lors de l’enregistrement dans best_hyperparams pour {model_name}: {str(e)}"
            logger.error(error_msg)
            logging.error(error_msg)
            capture_error(
                e,
                context={"model_name": model_name},
                market=self.market,
                operation="log_best_to_db",
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "log_best_to_db",
                latency,
                success=False,
                error=str(e),
                model_name=model_name,
            )

    def optimize(
        self, model_name: str, n_trials: int = 30, timeout: float = 600
    ) -> Dict:
        """Exécute l’optimisation des hyperparamètres.

        Args:
            model_name: Nom du modèle (ex. : 'sac', 'ppo', 'ddpg', 'ppo_cvar', 'qr_dqn', 'ensemble').
            n_trials: Nombre d’essais (défaut: 30).
            timeout: Délai maximal en secondes (défaut: 600).

        Returns:
            Dict: Meilleurs hyperparamètres et valeurs des métriques.
        """
        start_time = datetime.now()
        try:
            self.study.optimize(
                lambda trial: self.objective(trial, model_name),
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=4,
            )
            best = self.study.best_params
            best["sharpe"] = self.study.best_values[0]
            best["drawdown"] = self.study.best_values[1]
            self._log_best_to_db(model_name, best)
            logger.info(
                f"Meilleurs hyperparamètres pour {model_name} sur {self.market}: {best}"
            )
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "optimize", latency, success=True, model_name=model_name
            )
            self.alert_manager.send_alert(
                f"Hyperparamètres optimisés pour {model_name} sur {self.market}",
                priority=2,
            )
            return best
        except Exception as e:
            error_msg = f"Erreur lors de l’optimisation pour {model_name} sur {self.market}: {str(e)}"
            logger.error(error_msg)
            logging.error(error_msg)
            capture_error(
                e,
                context={"market": self.market, "model_name": model_name},
                market=self.market,
                operation="optuna_optimize",
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "optimize", latency, success=False, error=str(e), model_name=model_name
            )
            return {}

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ) -> None:
        """Journalise les performances CPU/mémoire dans hyperparam_optimizer_performance.csv."""
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
            cpu_percent = psutil.cpu_percent()
            if memory_usage > 512:
                alert_msg = f"ALERTE: Utilisation mémoire élevée ({memory_usage:.2f} MB) pour {operation}"
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=3)
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "market": self.market,
                "latency": latency,
                "success": success,
                "error": error,
                "memory_usage_mb": memory_usage,
                "cpu_usage_percent": cpu_percent,
                **kwargs,
            }
            log_df = pd.DataFrame([log_entry])
            log_path = LOG_DIR / "hyperparam_optimizer_performance.csv"
            log_df.to_csv(
                log_path,
                mode="a",
                header=not log_path.exists(),
                index=False,
                encoding="utf-8",
            )
            logger.info(
                f"Performance journalisée pour {operation}. CPU: {cpu_percent}%"
            )
        except Exception as e:
            error_msg = f"Erreur journalisation performance: {str(e)}"
            logger.error(error_msg)
            capture_error(
                e,
                context={"operation": operation},
                market=self.market,
                operation="log_performance",
            )
            self.alert_manager.send_alert(error_msg, priority=4)


# Exemple d’utilisation (à supprimer avant production)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    try:
        optimizer = HyperparamOptimizer(market="ES")
        best_params = optimizer.optimize("sac", n_trials=10, timeout=300)
        print(f"Meilleurs hyperparamètres: {best_params}")
    except ValueError as e:
        print(f"Erreur: {e}")
