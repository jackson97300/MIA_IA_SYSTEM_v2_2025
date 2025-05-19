# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_train_pipeline.py
# Tests unitaires pour dags/train_pipeline.py
#
# Version : 2.1.5
# Date : 2025-05-14
#
# Rôle : Valide le chargement du DAG Airflow, l’exécution des tâches existantes (check_trades, validate_data,
#        optimize_hyperparams, retrain_model), et des nouvelles tâches (walk_forward_validation, check_drift,
#        train_ppo_cvar, train_qr_dqn, train_ensemble), ainsi que la journalisation psutil et MLflow.
#        Vérifie l’intégration des métriques Prometheus (retrain_runs, cpu_usage, memory_usage, operation_latency).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, airflow>=2.6.0,<3.0.0, pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0
# - dags/train_pipeline.py
# - src/data/validate_data.py
# - src/model/trade_probability.py
# - src/model/utils/hyperparam_optimizer.py
# - src/utils/mlflow_tracker.py
# - src/utils/error_tracker.py
# - src/utils/telegram_alert.py
# - src/monitoring/drift_detector.py
# - src/monitoring/prometheus_metrics.py
#
# Inputs :
# - DAG factice (train_pipeline)
# - Données factices pour les tâches
#
# Outputs :
# - Tests unitaires validant le DAG et ses tâches.
#
# Notes :
# - Conforme à structure.txt (version 2.1.5, 2025-05-14).
# - Tests les suggestions 5 (walk-forward), 6 (drift detection), 7 (Safe RL), 8 (Distributional RL),
#   9 (réentraînement), 10 (ensembles de politiques).
# - Vérifie les logs psutil et la journalisation MLflow.
# - Utilise un LOG_DIR configurable pour les tests.
# - Policies Note: The official directory for routing policies is src/model/router/policies.
# The src/model/policies directory is a residual and should be verified
# for removal to avoid import conflicts.

import json
import os
import sqlite3
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from airflow.models import DagBag
from airflow.utils.dag_cycle_tester import test_cycle

from dags.train_pipeline import (
    check_drift,
    check_trades_count,
    log_performance,
    optimize_hyperparams,
    retrain_model,
    train_ensemble,
    train_ppo_cvar,
    train_qr_dqn,
    validate_data,
    walk_forward_validation,
)


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs et la base de données."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    dags_dir = base_dir / "dags"
    dags_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    # Copier train_pipeline.py dans dags_dir pour le test
    with open(dags_dir / "train_pipeline.py", "w") as f:
        f.write(open("dags/train_pipeline.py").read())
    return {
        "base_dir": base_dir,
        "dags_dir": dags_dir,
        "db_path": data_dir / "market_memory.db",
        "logs_dir": logs_dir,
        "perf_log_path": logs_dir / "train_pipeline_performance.csv",
    }


@pytest.fixture
def dag_bag(tmp_dirs):
    """Charge le DAG train_pipeline."""
    dag_bag = DagBag(dag_folder=str(tmp_dirs["dags_dir"]), include_examples=False)
    assert (
        dag_bag.get_dag("train_pipeline") is not None
    ), "DAG train_pipeline non chargé"
    return dag_bag


@pytest.fixture
def mock_db(tmp_dirs):
    """Crée une base de données SQLite factice."""
    conn = sqlite3.connect(str(tmp_dirs["db_path"]))
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE trade_patterns (market TEXT, timestamp TEXT, features TEXT, reward REAL)"
    )
    cursor.execute("CREATE TABLE training_log (timestamp TEXT)")
    cursor.executemany(
        "INSERT INTO trade_patterns (market, timestamp, features, reward) VALUES (?, ?, ?, ?)",
        [
            (
                "ES",
                datetime.now().isoformat(),
                json.dumps(
                    {
                        "rsi_14": np.random.normal(50, 10),
                        "obi_score": np.random.normal(0, 0.5),
                        "bid_ask_imbalance": np.random.normal(0, 0.1),
                        "trade_aggressiveness": np.random.normal(0, 0.2),
                        "iv_skew": np.random.normal(0.01, 0.005),
                        "iv_term_structure": np.random.normal(0.02, 0.005),
                        "slippage_estimate": np.random.uniform(0.01, 0.1),
                    }
                ),
                np.random.uniform(-1, 1),
            )
            for _ in range(1000)
        ],
    )
    cursor.execute(
        "INSERT INTO training_log (timestamp) VALUES (?)",
        ((datetime.now() - pd.Timedelta(days=1)).isoformat(),),
    )
    conn.commit()
    conn.close()
    return tmp_dirs["db_path"]


def test_dag_load(dag_bag):
    """Teste le chargement du DAG sans erreurs."""
    with patch("src.monitoring.prometheus_metrics.init_metrics"):
        dag = dag_bag.get_dag("train_pipeline")
    assert dag is not None, "DAG train_pipeline non chargé"
    assert (
        len(dag.tasks) == 9
    ), f"Nombre de tâches incorrect: attendu 9, obtenu {len(dag.tasks)}"
    expected_tasks = [
        "check_trades",
        "validate_data",
        "walk_forward_validation",
        "check_drift",
        "optimize_hyperparams",
        "retrain_model",
        "train_ppo_cvar",
        "train_qr_dqn",
        "train_ensemble",
    ]
    assert all(
        task_id in [task.task_id for task in dag.tasks] for task_id in expected_tasks
    ), "Tâches manquantes"
    test_cycle(dag)  # Vérifie l'absence de cycles


def test_check_trades_count(tmp_dirs, mock_db):
    """Teste la vérification du nombre de trades."""
    with patch("dags.train_pipeline.MLFlowTracker.log_metrics") as mock_mlflow, patch(
        "src.monitoring.prometheus_metrics.cpu_usage.labels"
    ) as mock_cpu, patch(
        "src.monitoring.prometheus_metrics.memory_usage.labels"
    ) as mock_memory, patch(
        "src.monitoring.prometheus_metrics.operation_latency.labels"
    ) as mock_latency, patch(
        "src.utils.telegram_alert.send_alert"
    ):
        with patch("dags.train_pipeline.BASE_DIR", tmp_dirs["base_dir"]):
            result = check_trades_count(market="ES")
        assert result is True, "Vérification des trades échouée"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "check_trades_count"
        ), "Log check_trades_count manquant"
        mock_mlflow.assert_called()
        mock_cpu.assert_called_with(market="ES")
        mock_memory.assert_called_with(market="ES")
        mock_latency.assert_called_with(operation="check_trades_count", market="ES")


def test_validate_data(tmp_dirs, mock_db):
    """Teste la validation des données."""
    with patch("dags.train_pipeline.MLFlowTracker.log_metrics") as mock_mlflow, patch(
        "src.monitoring.prometheus_metrics.cpu_usage.labels"
    ) as mock_cpu, patch(
        "src.monitoring.prometheus_metrics.memory_usage.labels"
    ) as mock_memory, patch(
        "src.monitoring.prometheus_metrics.operation_latency.labels"
    ) as mock_latency, patch(
        "src.utils.telegram_alert.send_alert"
    ):
        with patch("dags.train_pipeline.BASE_DIR", tmp_dirs["base_dir"]):
            result = validate_data(market="ES")
        assert not result.empty, "Validation des données n’a pas retourné de DataFrame"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df["operation"] == "validate_data"), "Log validate_data manquant"
        mock_mlflow.assert_called()
        mock_cpu.assert_called_with(market="ES")
        mock_memory.assert_called_with(market="ES")
        mock_latency.assert_called_with(operation="validate_data", market="ES")


def test_optimize_hyperparams(tmp_dirs):
    """Teste l’optimisation des hyperparamètres."""
    with patch(
        "src.model.utils.hyperparam_optimizer.HyperparamOptimizer.optimize",
        return_value={"n_estimators": 100},
    ), patch("dags.train_pipeline.MLFlowTracker.log_metrics") as mock_mlflow, patch(
        "src.monitoring.prometheus_metrics.cpu_usage.labels"
    ) as mock_cpu, patch(
        "src.monitoring.prometheus_metrics.memory_usage.labels"
    ) as mock_memory, patch(
        "src.monitoring.prometheus_metrics.operation_latency.labels"
    ) as mock_latency, patch(
        "src.utils.telegram_alert.send_alert"
    ):
        with patch("dags.train_pipeline.BASE_DIR", tmp_dirs["base_dir"]):
            result = optimize_hyperparams(market="ES")
        assert "n_estimators" in result, "Hyperparamètres non retournés"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "optimize_hyperparams"
        ), "Log optimize_hyperparams manquant"
        mock_mlflow.assert_called()
        mock_cpu.assert_called_with(market="ES")
        mock_memory.assert_called_with(market="ES")
        mock_latency.assert_called_with(operation="optimize_hyperparams", market="ES")


def test_retrain_model(tmp_dirs, mock_db):
    """Teste le réentraînement du modèle."""
    with patch(
        "src.model.trade_probability.TradeProbabilityPredictor.retrain_model"
    ) as mock_retrain, patch(
        "dags.train_pipeline.MLFlowTracker.log_run"
    ) as mock_mlflow, patch(
        "src.monitoring.prometheus_metrics.retrain_runs.labels"
    ) as mock_counter, patch(
        "src.monitoring.prometheus_metrics.cpu_usage.labels"
    ) as mock_cpu, patch(
        "src.monitoring.prometheus_metrics.memory_usage.labels"
    ) as mock_memory, patch(
        "src.monitoring.prometheus_metrics.operation_latency.labels"
    ) as mock_latency, patch(
        "src.utils.telegram_alert.send_alert"
    ):
        with patch("dags.train_pipeline.BASE_DIR", tmp_dirs["base_dir"]):
            ti = MagicMock()
            ti.xcom_pull.return_value = {"n_estimators": 100}
            retrain_model(market="ES", ti=ti)
        mock_retrain.assert_called_with(market="ES", n_estimators=100)
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df["operation"] == "retrain_model"), "Log retrain_model manquant"
        mock_mlflow.assert_called()
        mock_counter.assert_called_with(market="ES")
        mock_counter.return_value.inc.assert_called()
        mock_cpu.assert_called_with(market="ES")
        mock_memory.assert_called_with(market="ES")
        mock_latency.assert_called_with(operation="retrain_model", market="ES")


def test_walk_forward_validation(tmp_dirs, mock_db):
    """Teste la validation glissante."""
    with patch(
        "src.model.trade_probability.TradeProbabilityPredictor.retrain_model"
    ) as mock_retrain, patch(
        "src.model.trade_probability.TradeProbabilityPredictor.predict",
        return_value=0.7,
    ), patch(
        "dags.train_pipeline.MLFlowTracker.log_metrics"
    ) as mock_mlflow, patch(
        "src.monitoring.prometheus_metrics.cpu_usage.labels"
    ) as mock_cpu, patch(
        "src.monitoring.prometheus_metrics.memory_usage.labels"
    ) as mock_memory, patch(
        "src.monitoring.prometheus_metrics.operation_latency.labels"
    ) as mock_latency, patch(
        "src.utils.telegram_alert.send_alert"
    ):
        with patch("dags.train_pipeline.BASE_DIR", tmp_dirs["base_dir"]):
            ti = MagicMock()
            ti.xcom_pull.return_value = pd.DataFrame(
                {
                    "rsi_14": np.random.normal(50, 10, 1000),
                    "obi_score": np.random.normal(0, 0.5, 1000),
                    "bid_ask_imbalance": np.random.normal(0, 0.1, 1000),
                    "trade_aggressiveness": np.random.normal(0, 0.2, 1000),
                    "iv_skew": np.random.normal(0.01, 0.005, 1000),
                    "iv_term_structure": np.random.normal(0.02, 0.005, 1000),
                    "slippage_estimate": np.random.uniform(0.01, 0.1, 1000),
                }
            )
            result = walk_forward_validation(market="ES", ti=ti)
        assert result is True, "Validation glissante échouée"
        mock_retrain.assert_called()
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "walk_forward_validation"
        ), "Log walk_forward_validation manquant"
        mock_mlflow.assert_called()
        mock_cpu.assert_called_with(market="ES")
        mock_memory.assert_called_with(market="ES")
        mock_latency.assert_called_with(
            operation="walk_forward_validation", market="ES"
        )


def test_check_drift(tmp_dirs, mock_db):
    """Teste la détection de drift."""
    with patch(
        "src.monitoring.drift_detector.DriftDetector.detect_drift", return_value=True
    ), patch("dags.train_pipeline.MLFlowTracker.log_metrics") as mock_mlflow, patch(
        "src.monitoring.prometheus_metrics.cpu_usage.labels"
    ) as mock_cpu, patch(
        "src.monitoring.prometheus_metrics.memory_usage.labels"
    ) as mock_memory, patch(
        "src.monitoring.prometheus_metrics.operation_latency.labels"
    ) as mock_latency, patch(
        "src.utils.telegram_alert.send_alert"
    ):
        with patch("dags.train_pipeline.BASE_DIR", tmp_dirs["base_dir"]):
            ti = MagicMock()
            ti.xcom_pull.return_value = pd.DataFrame(
                {
                    "rsi_14": np.random.normal(50, 10, 1000),
                    "obi_score": np.random.normal(0, 0.5, 1000),
                    "bid_ask_imbalance": np.random.normal(0, 0.1, 1000),
                    "trade_aggressiveness": np.random.normal(0, 0.2, 1000),
                    "iv_skew": np.random.normal(0.01, 0.005, 1000),
                    "iv_term_structure": np.random.normal(0.02, 0.005, 1000),
                    "slippage_estimate": np.random.uniform(0.01, 0.1, 1000),
                }
            )
            result = check_drift(market="ES", ti=ti)
        assert result is True, "Détection de drift échouée"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df["operation"] == "check_drift"), "Log check_drift manquant"
        mock_mlflow.assert_called()
        mock_cpu.assert_called_with(market="ES")
        mock_memory.assert_called_with(market="ES")
        mock_latency.assert_called_with(operation="check_drift", market="ES")


def test_train_ppo_cvar(tmp_dirs, mock_db):
    """Teste l’entraînement PPO-CVaR."""
    with patch(
        "src.model.trade_probability.TradeProbabilityPredictor.train_rl_models"
    ) as mock_train, patch(
        "dags.train_pipeline.MLFlowTracker.log_run"
    ) as mock_mlflow, patch(
        "src.monitoring.prometheus_metrics.cpu_usage.labels"
    ) as mock_cpu, patch(
        "src.monitoring.prometheus_metrics.memory_usage.labels"
    ) as mock_memory, patch(
        "src.monitoring.prometheus_metrics.operation_latency.labels"
    ) as mock_latency, patch(
        "src.utils.telegram_alert.send_alert"
    ):
        with patch("dags.train_pipeline.BASE_DIR", tmp_dirs["base_dir"]):
            ti = MagicMock()
            ti.xcom_pull.return_value = pd.DataFrame(
                {
                    "rsi_14": np.random.normal(50, 10, 1000),
                    "obi_score": np.random.normal(0, 0.5, 1000),
                    "bid_ask_imbalance": np.random.normal(0, 0.1, 1000),
                    "trade_aggressiveness": np.random.normal(0, 0.2, 1000),
                    "iv_skew": np.random.normal(0.01, 0.005, 1000),
                    "iv_term_structure": np.random.normal(0.02, 0.005, 1000),
                    "slippage_estimate": np.random.uniform(0.01, 0.1, 1000),
                }
            )
            train_ppo_cvar(market="ES", ti=ti)
        mock_train.assert_called()
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df["operation"] == "train_ppo_cvar"), "Log train_ppo_cvar manquant"
        mock_mlflow.assert_called()
        mock_cpu.assert_called_with(market="ES")
        mock_memory.assert_called_with(market="ES")
        mock_latency.assert_called_with(operation="train_ppo_cvar", market="ES")


def test_train_qr_dqn(tmp_dirs, mock_db):
    """Teste le test QR-DQN."""
    with patch(
        "src.model.trade_probability.TradeProbabilityPredictor.train_rl_models"
    ) as mock_train, patch(
        "dags.train_pipeline.MLFlowTracker.log_run"
    ) as mock_mlflow, patch(
        "src.monitoring.prometheus_metrics.cpu_usage.labels"
    ) as mock_cpu, patch(
        "src.monitoring.prometheus_metrics.memory_usage.labels"
    ) as mock_memory, patch(
        "src.monitoring.prometheus_metrics.operation_latency.labels"
    ) as mock_latency, patch(
        "src.utils.telegram_alert.send_alert"
    ):
        with patch("dags.train_pipeline.BASE_DIR", tmp_dirs["base_dir"]):
            ti = MagicMock()
            ti.xcom_pull.return_value = pd.DataFrame(
                {
                    "rsi_14": np.random.normal(50, 10, 1000),
                    "obi_score": np.random.normal(0, 0.5, 1000),
                    "bid_ask_imbalance": np.random.normal(0, 0.1, 1000),
                    "trade_aggressiveness": np.random.normal(0, 0.2, 1000),
                    "iv_skew": np.random.normal(0.01, 0.005, 1000),
                    "iv_term_structure": np.random.normal(0.02, 0.005, 1000),
                    "slippage_estimate": np.random.uniform(0.01, 0.1, 1000),
                }
            )
            train_qr_dqn(market="ES", ti=ti)
        mock_train.assert_called()
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df["operation"] == "train_qr_dqn"), "Log train_qr_dqn manquant"
        assert any(
            df["operation"] == "train_qr_dqn" and df["success"]
        ), "Succès de train_qr_dqn non journalisé"
        mock_mlflow.assert_called()
        mock_cpu.assert_called_with(market="ES")
        mock_memory.assert_called_with(market="ES")
        mock_latency.assert_called_with(operation="train_qr_dqn", market="ES")


def test_train_ensemble(tmp_dirs, mock_db):
    """Teste l’entraînement de l’ensemble SAC, PPO, DDPG."""
    with patch(
        "src.model.trade_probability.TradeProbabilityPredictor.train_rl_models"
    ) as mock_train, patch(
        "dags.train_pipeline.MLFlowTracker.log_run"
    ) as mock_mlflow, patch(
        "src.monitoring.prometheus_metrics.cpu_usage.labels"
    ) as mock_cpu, patch(
        "src.monitoring.prometheus_metrics.memory_usage.labels"
    ) as mock_memory, patch(
        "src.monitoring.prometheus_metrics.operation_latency.labels"
    ) as mock_latency, patch(
        "src.utils.telegram_alert.send_alert"
    ):
        with patch("dags.train_pipeline.BASE_DIR", tmp_dirs["base_dir"]):
            ti = MagicMock()
            ti.xcom_pull.return_value = pd.DataFrame(
                {
                    "rsi_14": np.random.normal(50, 10, 1000),
                    "obi_score": np.random.normal(0, 0.5, 1000),
                    "bid_ask_imbalance": np.random.normal(0, 0.1, 1000),
                    "trade_aggressiveness": np.random.normal(0, 0.2, 1000),
                    "iv_skew": np.random.normal(0.01, 0.005, 1000),
                    "iv_term_structure": np.random.normal(0.02, 0.005, 1000),
                    "slippage_estimate": np.random.uniform(0.01, 0.1, 1000),
                }
            )
            train_ensemble(market="ES", ti=ti)
        mock_train.assert_called()
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df["operation"] == "train_ensemble"), "Log train_ensemble manquant"
        assert any(
            df["operation"] == "train_ensemble" and df["success"]
        ), "Succès de train_ensemble non journalisé"
        mock_mlflow.assert_called()
        mock_cpu.assert_called_with(market="ES")
        mock_memory.assert_called_with(market="ES")
        mock_latency.assert_called_with(operation="train_ensemble", market="ES")


def test_log_performance_metrics(tmp_dirs, mock_db):
    """Teste la journalisation des métriques Prometheus dans log_performance."""
    with patch("dags.train_pipeline.MLFlowTracker.log_metrics") as mock_mlflow, patch(
        "src.monitoring.prometheus_metrics.cpu_usage.labels"
    ) as mock_cpu, patch(
        "src.monitoring.prometheus_metrics.memory_usage.labels"
    ) as mock_memory, patch(
        "src.monitoring.prometheus_metrics.operation_latency.labels"
    ) as mock_latency, patch(
        "src.utils.telegram_alert.send_alert"
    ):
        with patch("dags.train_pipeline.BASE_DIR", tmp_dirs["base_dir"]):
            log_performance("test_op", 0.1, True, market="ES")
        assert os.path.exists(tmp_dirs["perf_log_path"]), "Fichier de logs non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df["operation"] == "test_op"), "Log test_op manquant"
        mock_mlflow.assert_called()
        mock_cpu.assert_called_with(market="ES")
        mock_memory.assert_called_with(market="ES")
        mock_latency.assert_called_with(operation="test_op", market="ES")
        mock_cpu.return_value.set.assert_called()
        mock_memory.return_value.set.assert_called()
        mock_latency.return_value.observe.assert_called_with(0.1)
