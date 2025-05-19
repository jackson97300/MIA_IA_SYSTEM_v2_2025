# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/model/utils/mlflow_tracker.py
# Tracking des runs MLflow pour MIA_IA_SYSTEM_v2_2025
#
# Version: 2.1.5
# Date: 2025-05-14
#
# Rôle: Journalise les paramètres, métriques, et artefacts des réentraînements avec MLflow.
# Utilisé par: trade_probability.py, dags/train_pipeline.py.
#
# Notes:
# - Conforme à structure.txt (version 2.1.5, 2025-05-14).
# - Supporte les suggestions 5 (Walk-forward), 7 (Safe RL/CVaR-PPO), 8 (Distributional RL/QR-DQN),
#   9 (réentraînement périodique), 10 (Ensembles de politiques).
# - Stocke les métadonnées dans data/market_memory.db (table mlflow_runs).
# - Intègre logs psutil dans data/logs/mlflow_tracker_performance.csv.
# - Utilise error_tracker.py pour capturer les erreurs et alert_manager.py pour les alertes.
# - Ajoute des tags MLflow pour market et suggestion_id, un nom de run explicite, et un artefact run_summary.json.
# - Pas de références à dxFeed, obs_t, 320 features, ou 81 features.
# - Policies Note: The official directory for routing policies is src/model/router/policies.
# The src/model/policies directory is a residual and should be verified
# for removal to avoid import conflicts.

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import mlflow
import pandas as pd
import psutil
from loguru import logger
from prometheus_client import Counter

from src.model.utils.alert_manager import AlertManager
from src.utils.error_tracker import capture_error

# Configuration du logging Loguru
logger.remove()
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOG_DIR = BASE_DIR / "data" / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "mlflow_tracker.log", rotation="10 MB", level="INFO", encoding="utf-8"
)

# Configuration du logging standard pour compatibilité
logging.basicConfig(
    filename=str(LOG_DIR / "mlflow_tracker.log"),
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)

# Compteur Prometheus pour les runs MLflow
mlflow_runs = Counter(
    "mlflow_runs_total",
    "Nombre total de runs MLflow enregistrés",
    ["market", "model_name"],
)


class MLflowTracker:
    """Trace les réentraînements avec MLflow et stocke les métadonnées localement."""

    def __init__(self, tracking_uri: str = "http://mlflow:5000", market: str = "ES"):
        """Initialise le tracker MLflow.

        Args:
            tracking_uri: URI du serveur MLflow (défaut: http://mlflow:5000).
            market: Marché cible (défaut: 'ES').

        Raises:
            ValueError: Si le tracking URI est invalide.
        """
        start_time = datetime.now()
        self.market = market
        self.alert_manager = AlertManager()
        try:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(
                f"Tracker MLflow initialisé avec URI: {tracking_uri} pour {market}"
            )
            logging.info(
                f"Tracker MLflow initialisé avec URI: {tracking_uri} pour {market}"
            )
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("init_mlflow_tracker", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur lors de l’initialisation de MLflow: {str(e)}"
            logger.error(error_msg)
            logging.error(error_msg)
            capture_error(
                e,
                context={"tracking_uri": tracking_uri},
                market=self.market,
                operation="init_mlflow_tracker",
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "init_mlflow_tracker", latency, success=False, error=str(e)
            )
            raise ValueError(f"Échec de l’initialisation de MLflow: {str(e)}")

    def log_run(
        self,
        model_name: str,
        metrics: Dict,
        parameters: Dict,
        artifacts: List[str] = None,
    ) -> None:
        """Journalise un run avec MLflow et stocke les métadonnées.

        Args:
            model_name: Nom du modèle (ex. : 'sac', 'ppo', 'ddpg', 'ppo_cvar', 'qr_dqn', 'walk_forward_sac').
            metrics: Dictionnaire des métriques (ex. : {'sharpe_ratio': 1.5, 'cvar_loss': 0.1}).
            parameters: Dictionnaire des hyperparamètres (ex. : {'learning_rate': 0.0003, 'cvar_alpha': 0.95}).
            artifacts: Liste des chemins d’artefacts (ex. : ['data/features/ES/feature_importance.csv']).

        Raises:
            ValueError: Si le run échoue ou si les paramètres sont invalides.
        """
        start_time = datetime.now()
        artifacts = artifacts or []
        try:
            # Définir le suggestion_id en fonction du modèle
            suggestion_map = {
                "walk_forward_sac": "5",
                "walk_forward_ppo": "5",
                "walk_forward_ddpg": "5",
                "ppo_cvar": "7",
                "qr_dqn": "8",
                "sac": "10",
                "ppo": "10",
                "ddpg": "10",
            }
            suggestion_id = suggestion_map.get(model_name, "unknown")

            # Définir le nom du run
            run_name = (
                f"{self.market}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

            with mlflow.start_run(run_name=run_name):
                # Ajouter des tags
                mlflow.set_tag("market", self.market)
                mlflow.set_tag("suggestion", suggestion_id)

                # Journaliser les paramètres
                mlflow.log_params(parameters)
                logger.debug(f"Paramètres journalisés pour {model_name}: {parameters}")

                # Journaliser les métriques
                mlflow.log_metrics(metrics)
                logger.debug(f"Métriques journalisées pour {model_name}: {metrics}")

                # Journaliser les métriques spécifiques
                if model_name in ["sac", "ppo", "ddpg"]:
                    mlflow.log_metric(
                        "ensemble_weight", metrics.get("ensemble_weight", 1.0)
                    )  # Suggestion 10
                elif model_name == "ppo_cvar":
                    mlflow.log_metric(
                        "cvar_loss", metrics.get("cvar_loss", 0.0)
                    )  # Suggestion 7
                    mlflow.log_param("cvar_alpha", parameters.get("cvar_alpha", 0.95))
                elif model_name == "qr_dqn":
                    mlflow.log_metric(
                        "quantiles", metrics.get("quantiles", 51)
                    )  # Suggestion 8
                elif "walk_forward" in model_name:
                    mlflow.log_metric(
                        "sharpe_ratio", metrics.get("sharpe_ratio", 0.0)
                    )  # Suggestion 5
                    mlflow.log_metric("max_drawdown", metrics.get("max_drawdown", 0.0))

                # Journaliser les artefacts
                for artifact in artifacts:
                    if os.path.exists(artifact):
                        mlflow.log_artifact(artifact)
                        logger.debug(
                            f"Artefact journalisé pour {model_name}: {artifact}"
                        )
                    else:
                        logger.warning(
                            f"Artefact non trouvé pour {model_name}: {artifact}"
                        )

                # Créer et journaliser run_summary.json
                summary = {
                    "model_name": model_name,
                    "parameters": parameters,
                    "metrics": metrics,
                    "timestamp": datetime.now().isoformat(),
                }
                summary_path = "run_summary.json"
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=4)
                mlflow.log_artifact(summary_path)
                os.remove(summary_path)
                logger.debug(f"Artefact run_summary.json journalisé pour {model_name}")

                # Enregistrer dans la base de données
                run_id = mlflow.active_run().info.run_id
                self._log_to_db(run_id, parameters, metrics, model_name)

                # Incrémenter le compteur Prometheus
                mlflow_runs.labels(market=self.market, model_name=model_name).inc()
                logger.info(
                    f"Run MLflow {run_id} journalisé pour {model_name} sur {self.market}"
                )
                latency = (datetime.now() - start_time).total_seconds()
                self.log_performance(
                    "log_run",
                    latency,
                    success=True,
                    model_name=model_name,
                    suggestion_id=suggestion_id,
                )
                self.alert_manager.send_alert(
                    f"Run journalisé pour {model_name} sur {self.market}", priority=2
                )

        except Exception as e:
            error_msg = f"Erreur lors du journalisation du run MLflow pour {model_name}: {str(e)}"
            logger.error(error_msg)
            logging.error(error_msg)
            capture_error(
                e,
                context={
                    "model_name": model_name,
                    "metrics": metrics,
                    "parameters": parameters,
                },
                market=self.market,
                operation="log_run",
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "log_run",
                latency,
                success=False,
                error=str(e),
                model_name=model_name,
                suggestion_id=suggestion_id,
            )
            raise ValueError(f"Échec du run MLflow: {str(e)}")

    def _log_to_db(
        self, run_id: str, parameters: Dict, metrics: Dict, model_name: str
    ) -> None:
        """Enregistre les métadonnées du run dans data/market_memory.db.

        Args:
            run_id: Identifiant du run MLflow.
            parameters: Dictionnaire des hyperparamètres.
            metrics: Dictionnaire des métriques.
            model_name: Nom du modèle.
        """
        start_time = datetime.now()
        try:
            db_path = BASE_DIR / "data" / "market_memory.db"
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO mlflow_runs (run_id, model_name, parameters, metrics, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        model_name,
                        json.dumps(parameters),
                        json.dumps(metrics),
                        datetime.now().isoformat(),
                    ),
                )
                conn.commit()
                logger.debug(
                    f"Métadonnées du run {run_id} enregistrées dans market_memory.db pour {model_name}"
                )
                latency = (datetime.now() - start_time).total_seconds()
                self.log_performance(
                    "log_to_db", latency, success=True, model_name=model_name
                )
        except sqlite3.Error as e:
            error_msg = f"Erreur lors de l’enregistrement dans market_memory.db pour {model_name}: {str(e)}"
            logger.error(error_msg)
            logging.error(error_msg)
            capture_error(
                e,
                context={"run_id": run_id, "model_name": model_name},
                market=self.market,
                operation="log_to_db",
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "log_to_db", latency, success=False, error=str(e), model_name=model_name
            )
            raise ValueError(
                f"Échec de l’enregistrement dans market_memory.db: {str(e)}"
            )

    def get_model_weights(self, models: List[str]) -> Dict:
        """Récupère les poids bayésiens pour l'ensemble.

        Args:
            models: Liste des noms de modèles (ex. : ['sac', 'ppo', 'ddpg']).

        Returns:
            Dict: Dictionnaire des poids bayésiens (ex. : {'sac': 0.33, 'ppo': 0.33, 'ddpg': 0.33}).

        Raises:
            ValueError: Si la récupération des poids échoue.
        """
        start_time = datetime.now()
        try:
            weights = {
                model: 1.0 / len(models) for model in models
            }  # Placeholder pour suggestion 10
            logger.debug(f"Poids bayésiens récupérés: {weights}")
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "get_model_weights", latency, success=True, models=",".join(models)
            )
            self.alert_manager.send_alert(
                f"Poids bayésiens récupérés pour {','.join(models)}", priority=2
            )
            return weights
        except Exception as e:
            error_msg = f"Erreur lors de la récupération des poids bayésiens: {str(e)}"
            logger.error(error_msg)
            logging.error(error_msg)
            capture_error(
                e,
                context={"models": models},
                market=self.market,
                operation="get_model_weights",
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "get_model_weights",
                latency,
                success=False,
                error=str(e),
                models=",".join(models),
            )
            return {model: 1.0 / len(models) for model in models}

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ) -> None:
        """Journalise les performances CPU/mémoire dans mlflow_tracker_performance.csv."""
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
            log_path = LOG_DIR / "mlflow_tracker_performance.csv"
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
        tracker = MLflowTracker()
        parameters = {"learning_rate": 0.0003, "evaluation_steps": 100}
        metrics = {"sharpe_ratio": 1.5, "accuracy": 0.85}
        artifacts = [
            str(BASE_DIR / "data" / "features" / "ES" / "feature_importance.csv")
        ]
        tracker.log_run("sac", metrics, parameters, artifacts)
        weights = tracker.get_model_weights(["sac", "ppo", "ddpg"])
        print(f"Run MLflow journalisé avec succès. Poids: {weights}")
    except ValueError as e:
        print(f"Erreur: {e}")
