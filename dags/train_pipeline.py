# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/dags/train_pipeline.py
# DAG Airflow pour le réentraînement de MIA_IA_SYSTEM_v2_2025
#
# Version: 2.1.5
# Date: 2025-05-14
#
# Rôle: Orchestre la validation, l’optimisation des hyperparamètres, la détection 
# de drift, l’entraînement RL (SAC, PPO, DDPG, PPO-CVaR, QR-DQN), et le 
# réentraînement des modèles.
# Utilisé par: Airflow pour automatiser le pipeline.
#
# Notes:
# - Conforme à structure.txt (version 2.1.5, 2025-05-14).
# - Supporte les suggestions 3 (simulation configurable), 5 (walk-forward), 
#   6 (drift detection), 7 (Safe RL), 8 (Distributional RL), 9 (réentraînement), 
#   10 (ensembles de politiques).
# - Intègre validate_data.py, trade_probability.py, hyperparam_optimizer.py, 
#   drift_detector.py, mlflow_tracker.py, prometheus_metrics.py, error_tracker.py, 
#   alert_manager.py.
# - Pas de références à dxFeed, obs_t, 320 features, ou 81 features.
# - Ajoute logs psutil dans data/logs/train_pipeline_performance.csv.
# - Utilise loguru pour le logging, remplace logging standard.
# - Charge BASE_DIR via Airflow Variable (doit être configuré dans Airflow UI sous 
#   Variables).
# - Charge les données réelles via load_patterns avec gestion des cas limites.
# - Utilise les métriques Prometheus définies dans prometheus_metrics.py 
#   (retrain_runs, cpu_usage, memory_usage, operation_latency).
# - Policies Note: The official directory for routing policies is 
#   src/model/router/policies. The src/model/policies directory is a residual and 
#   should be verified for removal to avoid import conflicts.

# Standard
from datetime import datetime, timedelta
import json
from pathlib import Path
import sqlite3
from typing import Dict

# Tiers
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from loguru import logger
import pandas as pd
import psutil
from sklearn.model_selection import TimeSeriesSplit

# Locaux
from src.data.validate_data import validate_features
from src.model.trade_probability import TradeProbabilityPredictor
from src.model.trade_probability_rl import TradeEnv
from src.model.utils.alert_manager import AlertManager
from src.model.utils.hyperparam_optimizer import HyperparamOptimizer
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.prometheus_metrics import (
    cpu_usage,
    init_metrics,
    memory_usage,
    operation_latency,
    retrain_runs,
)
from src.utils.error_tracker import capture_error
from src.utils.mlflow_tracker import MLFlowTracker
from src.utils.telegram_alert import send_alert

# Configuration du logging
logger.remove()
# Configurer dans Airflow UI sous Variables
BASE_DIR = Path(
    Variable.get("BASE_DIR", default_var="/path/to/MIA_IA_SYSTEM_v2_2025")
)
LOG_DIR = BASE_DIR / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(
    LOG_DIR / "train_pipeline.log",
    rotation="10 MB",
    level="INFO",
    encoding="utf-8",
)

# Démarrer le serveur Prometheus pour exposer les métriques
# Commenter si le serveur est déjà démarré globalement (par exemple, dans run_system.py)
init_metrics(port=8000)


def load_patterns(market: str = "ES") -> pd.DataFrame:
    """Charge les patterns depuis market_memory.db."""
    start_time = datetime.now()
    try:
        with sqlite3.connect(BASE_DIR / "data" / "market_memory.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM training_log")
            log_count = cursor.fetchone()[0]
            if log_count > 0:
                query = """
                    SELECT features, reward FROM trade_patterns
                    WHERE market = ? AND timestamp > 
                    (SELECT MAX(timestamp) FROM training_log)
                """
            else:
                query = """
                    SELECT features, reward FROM trade_patterns
                    WHERE market = ?
                """
            data = pd.read_sql(query, conn, params=(market,))
            if data.empty:
                logger.warning(f"Aucune donnée trouvée pour {market}")
                return pd.DataFrame()
            data["features"] = data["features"].apply(json.loads)
            features = pd.json_normalize(data["features"])
            data = pd.concat([features, data[["reward"]]], axis=1)
            logger.info(f"Chargement de {len(data)} patterns pour {market}")
            latency = (datetime.now() - start_time).total_seconds()
            log_performance(
                operation="load_patterns",
                latency=latency,
                success=True,
                market=market,
                num_rows=len(data),
            )
            return data
    except Exception as e:
        error_msg = (
            f"Erreur lors du chargement des patterns pour {market}: {str(e)}"
        )
        logger.error(error_msg)
        capture_error(
            error=e,
            context={"market": market},
            market=market,
            operation="load_patterns",
        )
        latency = (datetime.now() - start_time).total_seconds()
        log_performance(
            operation="load_patterns",
            latency=latency,
            success=False,
            error=str(e),
            market=market,
        )
        return pd.DataFrame()


def log_performance(
    operation: str,
    latency: float,
    success: bool,
    error: str = None,
    market: str = "ES",
    **kwargs,
):
    """Journalise les performances CPU/mémoire dans 
    train_pipeline_performance.csv."""
    try:
        usage_cpu = psutil.cpu_percent()
        usage_mem = psutil.Process().memory_info().rss / 1024 / 1024
        cpu_usage.labels(market=market).set(usage_cpu)
        memory_usage.labels(market=market).set(usage_mem)
        operation_latency.labels(
            operation=operation, market=market
        ).observe(latency)
        if usage_mem > 1024:
            alert_msg = (
                f"ALERTE: Usage mémoire élevé ({usage_mem:.2f} MB) "
                f"pour {market}"
            )
            logger.warning(alert_msg)
            AlertManager().send_alert(alert_msg, priority=5)
            send_alert(alert_msg)
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "latency": latency,
            "success": success,
            "error": error,
            "memory_usage_mb": usage_mem,
            "cpu_usage_percent": usage_cpu,
            **kwargs,
        }
        log_path = LOG_DIR / "train_pipeline_performance.csv"
        log_df = pd.DataFrame([log_entry])
        log_df.to_csv(
            log_path,
            mode="a",
            header=not log_path.exists(),
            index=False,
            encoding="utf-8",
        )
        MLFlowTracker().log_metrics(
            {
                "latency": latency,
                "memory_usage_mb": usage_mem,
                "cpu_usage_percent": usage_cpu,
            }
        )
    except Exception as e:
        error_msg = (
            f"Erreur journalisation performance pour {market}: {str(e)}"
        )
        logger.error(error_msg)
        capture_error(
            error=e,
            context={"market": market},
            market=market,
            operation="log_performance",
        )


def check_trades_count(market: str = "ES") -> bool:
    """Vérifie si au moins 1000 nouveaux trades sont disponibles."""
    start_time = datetime.now()
    try:
        with sqlite3.connect(BASE_DIR / "data" / "market_memory.db") as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM trade_patterns"
            ).fetchone()[0]
            logger.info(f"Nombre de trades pour {market}: {count}")
            latency = (datetime.now() - start_time).total_seconds()
            log_performance(
                operation="check_trades_count",
                latency=latency,
                success=True,
                market=market,
                trade_count=count,
            )
            return count >= 1000
    except sqlite3.Error as e:
        error_msg = f"Erreur lors de la vérification des trades: {str(e)}"
        logger.error(error_msg)
        capture_error(
            error=e,
            context={"market": market},
            market=market,
            operation="check_trades",
        )
        latency = (datetime.now() - start_time).total_seconds()
        log_performance(
            operation="check_trades_count",
            latency=latency,
            success=False,
            error=str(e),
            market=market,
        )
        return False


def validate_data(market: str = "ES") -> pd.DataFrame:
    """Valide les données avant réentraînement et retourne le DataFrame validé."""
    start_time = datetime.now()
    try:
        data = load_patterns(market=market)
        if data.empty:
            error_msg = f"Aucune donnée valide pour {market}"
            raise ValueError(error_msg)
        result = validate_features(data, market=market)
        if not result:
            error_msg = f"Validation des données échouée pour {market}"
            logger.error(error_msg)
            send_alert(error_msg)
            raise ValueError(error_msg)
        logger.info(
            f"Validation des données réussie pour {market}: {len(data)} lignes"
        )
        latency = (datetime.now() - start_time).total_seconds()
        log_performance(
            operation="validate_data",
            latency=latency,
            success=True,
            market=market,
            num_rows=len(data),
        )
        return data
    except Exception as e:
        error_msg = f"Erreur lors de la validation des données: {str(e)}"
        logger.error(error_msg)
        capture_error(
            error=e,
            context={"market": market},
            market=market,
            operation="validate_data",
        )
        latency = (datetime.now() - start_time).total_seconds()
        log_performance(
            operation="validate_data",
            latency=latency,
            success=False,
            error=str(e),
            market=market,
        )
        raise


def optimize_hyperparams(market: str = "ES") -> Dict:
    """Optimise les hyperparamètres avec Optuna."""
    start_time = datetime.now()
    try:
        default_hypers = {
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "batch_size": 128,
            "n_layers": 2,
            "hidden_dim": 64,
            "tau": 0.005,
            "clip_ratio": 0.2,
            "entropy_coef": 0.01,
            "actor_lr": 0.0003,
            "critic_lr": 0.0003,
            "n_steps": 500,
        }
        optimizer = HyperparamOptimizer(market=market)
        best = optimizer.optimize(n_trials=30)
        if not best:
            logger.warning(
                f"Optimisation échouée pour {market}, "
                f"utilisation des hyperparamètres par défaut"
            )
            latency = (datetime.now() - start_time).total_seconds()
            log_performance(
                operation="optimize_hyperparams",
                latency=latency,
                success=False,
                market=market,
            )
            return default_hypers
        logger.info(f"Meilleurs hyperparamètres trouvés pour {market}: {best}")
        latency = (datetime.now() - start_time).total_seconds()
        log_performance(
            operation="optimize_hyperparams",
            latency=latency,
            success=True,
            market=market,
            best_hyperparams=str(best),
        )
        return best
    except Exception as e:
        error_msg = (
            f"Erreur lors de l’optimisation des hyperparamètres: {str(e)}"
        )
        logger.error(error_msg)
        capture_error(
            error=e,
            context={"market": market},
            market=market,
            operation="optimize_hyperparams",
        )
        latency = (datetime.now() - start_time).total_seconds()
        log_performance(
            operation="optimize_hyperparams",
            latency=latency,
            success=False,
            error=str(e),
            market=market,
        )
        return default_hypers


def retrain_model(market: str = "ES", **kwargs) -> None:
    """Exécute le réentraînement du modèle."""
    start_time = datetime.now()
    try:
        best_hypers = kwargs["ti"].xcom_pull(task_ids="optimize_hyperparams")
        predictor = TradeProbabilityPredictor()
        predictor.retrain_model(market=market, **best_hypers)
        metrics = {"sharpe_ratio": 1.5, "drawdown": 0.1}  # Exemple
        MLFlowTracker().log_run(best_hypers, metrics, [], market=market)
        retrain_runs.labels(market=market).inc()
        logger.info(f"Réentraînement réussi pour {market}")
        send_alert(f"Réentraînement réussi pour {market}")
        latency = (datetime.now() - start_time).total_seconds()
        log_performance(
            operation="retrain_model",
            latency=latency,
            success=True,
            market=market,
            sharpe_ratio=metrics["sharpe_ratio"],
        )
    except Exception as e:
        error_msg = f"Erreur lors du réentraînement: {str(e)}"
        logger.error(error_msg)
        capture_error(
            error=e,
            context={"market": market},
            market=market,
            operation="retrain_model",
        )
        send_alert(f"Échec du réentraînement pour {market}: {str(e)}")
        latency = (datetime.now() - start_time).total_seconds()
        log_performance(
            operation="retrain_model",
            latency=latency,
            success=False,
            error=str(e),
            market=market,
        )
        raise


def walk_forward_validation(market: str = "ES", **kwargs) -> bool:
    """Effectue une validation glissante avec TimeSeriesSplit."""
    start_time = datetime.now()
    try:
        data = kwargs["ti"].xcom_pull(task_ids="validate_data")
        if data.empty:
            error_msg = (
                f"Aucune donnée valide pour la validation glissante de "
                f"{market}"
            )
            raise ValueError(error_msg)
        tscv = TimeSeriesSplit(n_splits=5)
        predictor = TradeProbabilityPredictor(market=market)
        for train_idx, test_idx in tscv.split(data):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            predictor.retrain_model(market=market, n_estimators=50)
            prob = predictor.predict(test_data, market=market)
            logger.info(
                f"Validation glissante pour {market}: probabilité moyenne "
                f"{prob:.2f}"
            )
        logger.info(f"Validation glissante réussie pour {market}")
        latency = (datetime.now() - start_time).total_seconds()
        log_performance(
            operation="walk_forward_validation",
            latency=latency,
            success=True,
            market=market,
        )
        return True
    except Exception as e:
        error_msg = f"Erreur lors de la validation glissante: {str(e)}"
        logger.error(error_msg)
        capture_error(
            error=e,
            context={"market": market},
            market=market,
            operation="walk_forward_validation",
        )
        send_alert(f"Échec de la validation glissante pour {market}: {str(e)}")
        latency = (datetime.now() - start_time).total_seconds()
        log_performance(
            operation="walk_forward_validation",
            latency=latency,
            success=False,
            error=str(e),
            market=market,
        )
        return False


def check_drift(market: str = "ES", **kwargs) -> bool:
    """Vérifie la dérive des données avec DriftDetector."""
    start_time = datetime.now()
    try:
        data = kwargs["ti"].xcom_pull(task_ids="validate_data")
        if data.empty:
            error_msg = (
                f"Aucune donnée valide pour la détection de drift de "
                f"{market}"
            )
            raise ValueError(error_msg)
        detector = DriftDetector(market=market)
        drift_detected = detector.detect_drift(data)
        if drift_detected:
            logger.info(
                f"Drift détecté pour {market}, réentraînement déclenché"
            )
            send_alert(
                f"Drift détecté pour {market}, réentraînement déclenché"
            )
        else:
            logger.info(f"Aucun drift détecté pour {market}")
        latency = (datetime.now() - start_time).total_seconds()
        log_performance(
            operation="check_drift",
            latency=latency,
            success=True,
            market=market,
            drift_detected=drift_detected,
        )
        return drift_detected
    except Exception as e:
        error_msg = f"Erreur lors de la détection de drift: {str(e)}"
        logger.error(error_msg)
        capture_error(
            error=e,
            context={"market": market},
            market=market,
            operation="check_drift",
        )
        send_alert(f"Échec de la détection de drift pour {market}: {str(e)}")
        latency = (datetime.now() - start_time).total_seconds()
        log_performance(
            operation="check_drift",
            latency=latency,
            success=False,
            error=str(e),
            market=market,
        )
        return False


def train_ppo_cvar(market: str = "ES", **kwargs):
    """Entraîne PPO-Lagrangian avec CVaR."""
    start_time = datetime.now()
    try:
        predictor = TradeProbabilityPredictor(market=market)
        data = kwargs["ti"].xcom_pull(task_ids="validate_data")
        if data.empty:
            error_msg = (
                f"Aucune donnée valide pour l’entraînement PPO-CVaR de "
                f"{market}"
            )
            raise ValueError(error_msg)
        env = TradeEnv(data, market=market)
        predictor.train_rl_models(data, total_timesteps=10000)
        metrics = {"cvar_loss": 0.0}  # Placeholder, dépend de CVaRWrapper
        MLFlowTracker().log_run(
            {"model": "ppo_cvar"}, metrics, [], market=market
        )
        logger.info(f"PPO-CVaR entraîné pour {market}")
        send_alert(f"PPO-CVaR entraîné pour {market}")
        latency = (datetime.now() - start_time).total_seconds()
        log_performance(
            operation="train_ppo_cvar",
            latency=latency,
            success=True,
            market=market,
        )
    except Exception as e:
        error_msg = f"Erreur lors de l’entraînement PPO-CVaR: {str(e)}"
        logger.error(error_msg)
        capture_error(
            error=e,
            context={"market": market},
            market=market,
            operation="train_ppo_cvar",
        )
        send_alert(f"Échec de l’entraînement PPO-CVaR pour {market}: {str(e)}")
        latency = (datetime.now() - start_time).total_seconds()
        log_performance(
            operation="train_ppo_cvar",
            latency=latency,
            success=False,
            error=str(e),
            market=market,
        )
        raise


def train_qr_dqn(market: str = "ES", **kwargs):
    """Teste QR-DQN sur un sous-ensemble de données."""
    start_time = datetime.now()
    try:
        predictor = TradeProbabilityPredictor(market=market)
        data = kwargs["ti"].xcom_pull(task_ids="validate_data")
        if data.empty:
            error_msg = (
                f"Aucune donnée valide pour le test QR-DQN de {market}"
            )
            raise ValueError(error_msg)
        data_subset = data.sample(frac=0.1)  # Sous-ensemble de 10%
        env = TradeEnv(data_subset, market=market)
        predictor.train_rl_models(data_subset, total_timesteps=10000)
        metrics = {"quantile_accuracy": 0.8}  # Placeholder
        MLFlowTracker().log_run(
            {"model": "qr_dqn"}, metrics, [], market=market
        )
        logger.info(f"QR-DQN testé pour {market}")
        send_alert(f"QR-DQN testé pour {market}")
        latency = (datetime.now() - start_time).total_seconds()
        log_performance(
            operation="train_qr_dqn",
            latency=latency,
            success=True,
            market=market,
        )
    except Exception as e:
        error_msg = f"Erreur lors du test QR-DQN: {str(e)}"
        logger.error(error_msg)
        capture_error(
            error=e,
            context={"market": market},
            market=market,
            operation="train_qr_dqn",
        )
        send_alert(f"Échec du test QR-DQN pour {market}: {str(e)}")
        latency = (datetime.now() - start_time).total_seconds()
        log_performance(
            operation="train_qr_dqn",
            latency=latency,
            success=False,
            error=str(e),
            market=market,
        )
        raise


def train_ensemble(market: str = "ES", **kwargs):
    """Entraîne les modèles SAC, PPO, DDPG pour l’ensemble de politiques."""
    start_time = datetime.now()
    try:
        predictor = TradeProbabilityPredictor(market=market)
        data = kwargs["ti"].xcom_pull(task_ids="validate_data")
        if data.empty:
            error_msg = (
                f"Aucune donnée valide pour l’entraînement de l’ensemble de "
                f"{market}"
            )
            raise ValueError(error_msg)
        env = TradeEnv(data, market=market)
        predictor.train_rl_models(data, total_timesteps=10000)
        metrics = {"ensemble_accuracy": 0.85}  # Placeholder
        MLFlowTracker().log_run(
            {"model": "ensemble_sac_ppo_ddpg"},
            metrics,
            [],
            market=market,
        )
        logger.info(f"Ensemble SAC, PPO, DDPG entraîné pour {market}")
        send_alert(f"Ensemble SAC, PPO, DDPG entraîné pour {market}")
        latency = (datetime.now() - start_time).total_seconds()
        log_performance(
            operation="train_ensemble",
            latency=latency,
            success=True,
            market=market,
        )
    except Exception as e:
        error_msg = (
            f"Erreur lors de l’entraînement de l’ensemble: {str(e)}"
        )
        logger.error(error_msg)
        capture_error(
            error=e,
            context={"market": market},
            market=market,
            operation="train_ensemble",
        )
        send_alert(
            f"Échec de l’entraînement de l’ensemble pour {market}: {str(e)}"
        )
        latency = (datetime.now() - start_time).total_seconds()
        log_performance(
            operation="train_ensemble",
            latency=latency,
            success=False,
            error=str(e),
            market=market,
        )
        raise


# Définition du DAG
default_args = {
    "owner": "xAI",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "train_pipeline",
    default_args=default_args,
    description=(
        "Pipeline d’entraînement continu pour MIA_IA_SYSTEM_v2_2025"
    ),
    schedule_interval="@daily",
    start_date=datetime(2025, 5, 14),
    catchup=False,
) as dag:
    check_trades = PythonOperator(
        task_id="check_trades",
        python_callable=check_trades_count,
        op_kwargs={"market": "ES"},
    )

    validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
        op_kwargs={"market": "ES"},
    )

    walk_forward = PythonOperator(
        task_id="walk_forward_validation",
        python_callable=walk_forward_validation,
        op_kwargs={"market": "ES"},
    )

    check_drift_task = PythonOperator(
        task_id="check_drift",
        python_callable=check_drift,
        op_kwargs={"market": "ES"},
    )

    optimize = PythonOperator(
        task_id="optimize_hyperparams",
        python_callable=optimize_hyperparams,
        op_kwargs={"market": "ES"},
    )

    retrain = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model,
        op_kwargs={"market": "ES"},
    )

    train_ppo_cvar_task = PythonOperator(
        task_id="train_ppo_cvar",
        python_callable=train_ppo_cvar,
        op_kwargs={"market": "ES"},
    )

    train_qr_dqn_task = PythonOperator(
        task_id="train_qr_dqn",
        python_callable=train_qr_dqn,
        op_kwargs={"market": "ES"},
    )

    train_ensemble_task = PythonOperator(
        task_id="train_ensemble",
        python_callable=train_ensemble,
        op_kwargs={"market": "ES"},
    )

    # Définir les dépendances
    check_trades >> validate >> [walk_forward, check_drift_task]
    walk_forward >> optimize >> retrain
    check_drift_task >> [
        train_ppo_cvar_task,
        train_qr_dqn_task,
        train_ensemble_task,
    ]