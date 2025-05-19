# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_mlflow_tracker.py
# Tests unitaires pour src/model/utils/mlflow_tracker.py
#
# Version : 2.1.5
# Date : 2025-05-14
#
# Rôle : Valide la journalisation des runs MLflow pour les modèles SAC, PPO, DDPG, PPO-Lagrangian, QR-DQN,
#        et la validation glissante (Walk-forward). Teste les fonctionnalités : logs psutil,
#        gestion des erreurs avec error_tracker.py, alertes via alert_manager.py, poids bayésiens,
#        tags MLflow (market, suggestion), nom de run explicite, et artefact run_summary.json.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, psutil>=5.9.8,<6.0.0, sqlite3, json
# - src/model/utils/mlflow_tracker.py
# - src/utils/error_tracker.py
# - src/model/utils/alert_manager.py
#
# Inputs :
# - Données factices pour les runs MLflow.
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de mlflow_tracker.py.
#
# Notes :
# - Conforme à structure.txt (version 2.1.5, 2025-05-14).
# - Supporte les suggestions 5 (Walk-forward), 7 (Safe RL/CVaR-PPO), 8 (Distributional RL/QR-DQN),
#   9 (réentraînement périodique), 10 (Ensembles de politiques).
# - Vérifie les logs psutil dans data/logs/mlflow_tracker_performance.csv.
# - Vérifie les tags MLflow, le nom du run, et l'artefact run_summary.json.
# - Policies Note: The official directory for routing policies is src/model/router/policies.
# The src/model/policies directory is a residual and should be verified
# for removal to avoid import conflicts.

import json
import os
import sqlite3
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.model.utils.mlflow_tracker import MLflowTracker


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs et la base de données."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    logs_dir = base_dir / "data" / "logs"
    logs_dir.mkdir(parents=True)
    data_dir = base_dir / "data"
    db_path = data_dir / "market_memory.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE mlflow_runs (
                run_id TEXT,
                model_name TEXT,
                parameters TEXT,
                metrics TEXT,
                timestamp TEXT
            )
            """
        )
    return {
        "base_dir": str(base_dir),
        "logs_dir": str(logs_dir),
        "perf_log_path": str(logs_dir / "mlflow_tracker_performance.csv"),
        "log_file_path": str(logs_dir / "mlflow_tracker.log"),
        "db_path": str(db_path),
        "artifact_path": str(data_dir / "features" / "ES" / "feature_importance.csv"),
    }


@pytest.fixture
def mock_mlflow():
    """Mock pour MLflow."""
    with patch("mlflow.start_run") as mock_start_run, patch(
        "mlflow.log_params"
    ) as mock_log_params, patch("mlflow.log_metrics") as mock_log_metrics, patch(
        "mlflow.log_artifact"
    ) as mock_log_artifact, patch(
        "mlflow.set_tag"
    ) as mock_set_tag:
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_start_run.return_value.__enter__.return_value = mock_run
        yield {
            "start_run": mock_start_run,
            "log_params": mock_log_params,
            "log_metrics": mock_log_metrics,
            "log_artifact": mock_log_artifact,
            "set_tag": mock_set_tag,
        }


def test_mlflow_tracker_init_success(tmp_dirs):
    """Teste l'initialisation réussie de MLflowTracker."""
    with patch("mlflow.set_tracking_uri") as mock_mlflow, patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ) as mock_alert:
        tracker = MLflowTracker(tracking_uri="http://test:5000", market="ES")
        assert tracker.market == "ES", "Marché incorrect"
        mock_mlflow.assert_called_with("http://test:5000")
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "init_mlflow_tracker"
        ), "Log init_mlflow_tracker manquant"
        assert all(
            col in df.columns for col in ["memory_usage_mb", "cpu_usage_percent"]
        ), "Colonnes psutil manquantes"
        mock_alert.assert_called_with(
            "Secret récupéré: init_mlflow_tracker...", priority=2
        )


def test_mlflow_tracker_init_failure(tmp_dirs):
    """Teste l'échec de l'initialisation de MLflowTracker."""
    with patch(
        "mlflow.set_tracking_uri", side_effect=Exception("MLflow init failed")
    ), patch("src.utils.error_tracker.capture_error") as mock_capture_error, patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ) as mock_alert:
        with pytest.raises(ValueError, match="Échec de l’initialisation de MLflow"):
            MLflowTracker(tracking_uri="http://invalid:5000", market="ES")
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "init_mlflow_tracker" and df["success"] == False
        ), "Échec init_mlflow_tracker non logué"
        mock_capture_error.assert_called()
        mock_alert.assert_called_with(pytest.any(str), priority=4)


def test_log_run_basic(tmp_dirs, mock_mlflow):
    """Teste la journalisation de base d'un run (basé sur le test proposé)."""
    with patch("sqlite3.connect") as mock_connect, patch(
        "src.model.utils.mlflow_tracker.mlflow_runs"
    ) as mock_counter, patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ) as mock_alert, patch(
        "os.path.exists", return_value=True
    ):
        tracker = MLflowTracker(market="ES")
        parameters = {"learning_rate": 0.0003}
        metrics = {"sharpe_ratio": 1.5}
        artifacts = [tmp_dirs["artifact_path"]]
        tracker.log_run("sac", metrics, parameters, artifacts)
        mock_mlflow["start_run"].assert_called()
        mock_mlflow["log_params"].assert_called_with(parameters)
        mock_mlflow["log_metrics"].assert_called_with(metrics)
        mock_mlflow["log_artifact"].assert_called_with("run_summary.json")
        mock_connect.return_value.execute.assert_called()
        mock_counter.labels.assert_called_with(market="ES", model_name="sac")
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "log_run" and df["model_name"] == "sac"
        ), "Log log_run (sac) manquant"
        assert all(
            col in df.columns for col in ["memory_usage_mb", "cpu_usage_percent"]
        ), "Colonnes psutil manquantes"
        mock_alert.assert_called_with("Run journalisé pour sac sur ES", priority=2)


def test_log_run_invalid_artifact(tmp_dirs, mock_mlflow):
    """Teste la gestion des artefacts manquants (basé sur le test proposé)."""
    with patch("sqlite3.connect") as mock_connect, patch(
        "src.model.utils.mlflow_tracker.mlflow_runs"
    ) as mock_counter, patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ) as mock_alert, patch(
        "os.path.exists", return_value=False
    ):
        tracker = MLflowTracker(market="ES")
        parameters = {"learning_rate": 0.0003}
        metrics = {"sharpe_ratio": 1.5}
        artifacts = ["non_existent.csv"]
        with pytest.raises(ValueError, match="Échec du run MLflow"):
            tracker.log_run("sac", metrics, parameters, artifacts)
        mock_mlflow["start_run"].assert_called()
        mock_connect.return_value.execute.assert_called()
        mock_counter.labels.assert_called_with(market="ES", model_name="sac")
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "log_run" and df["success"] == False
        ), "Échec log_run non logué"
        mock_alert.assert_called_with(pytest.any(str), priority=4)


def test_log_run_new_models(tmp_dirs, mock_mlflow):
    """Teste la journalisation des runs pour les nouveaux modèles."""
    with patch("sqlite3.connect") as mock_connect, patch(
        "src.model.utils.mlflow_tracker.mlflow_runs"
    ) as mock_counter, patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ) as mock_alert, patch(
        "os.path.exists", return_value=True
    ):
        mock_connect.return_value.execute.return_value = None
        tracker = MLflowTracker(market="ES")
        scenarios = [
            (
                "walk_forward_sac",
                {"sharpe_ratio": 1.5, "max_drawdown": 0.2},
                {"n_splits": 5},
                "5",
            ),  # Suggestion 5
            (
                "ppo_cvar",
                {"cvar_loss": 0.1, "sharpe_ratio": 1.2},
                {"cvar_alpha": 0.95},
                "7",
            ),  # Suggestion 7
            (
                "qr_dqn",
                {"quantiles": 51, "sharpe_ratio": 1.3},
                {"num_quantiles": 51},
                "8",
            ),  # Suggestion 8
            (
                "sac",
                {"ensemble_weight": 0.4, "sharpe_ratio": 1.4},
                {"learning_rate": 0.0003},
                "10",
            ),  # Suggestion 10
        ]
        for model_name, metrics, parameters, suggestion_id in scenarios:
            tracker.log_run(
                model_name, metrics, parameters, [tmp_dirs["artifact_path"]]
            )
            mock_mlflow["start_run"].assert_called()
            mock_mlflow["log_params"].assert_called_with(parameters)
            mock_mlflow["log_metrics"].assert_any_call(metrics)
            mock_mlflow["set_tag"].assert_any_call("market", "ES")
            mock_mlflow["set_tag"].assert_any_call("suggestion", suggestion_id)
            mock_mlflow["log_artifact"].assert_called_with("run_summary.json")
            mock_counter.labels.assert_any_call(market="ES", model_name=model_name)
            with sqlite3.connect(tmp_dirs["db_path"]) as conn:
                df_db = pd.read_sql("SELECT * FROM mlflow_runs", conn)
                assert any(
                    df_db["model_name"] == model_name
                ), f"Run {model_name} non enregistré dans market_memory.db"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df["operation"] == "log_run"), "Log log_run manquant"
        mock_alert.assert_called_with(pytest.any(str), priority=2)


def test_log_run_run_summary(tmp_dirs, mock_mlflow):
    """Teste la journalisation de l'artefact run_summary.json."""
    with patch("sqlite3.connect") as mock_connect, patch(
        "src.model.utils.mlflow_tracker.mlflow_runs"
    ) as mock_counter, patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ) as mock_alert, patch(
        "os.path.exists", return_value=True
    ), patch(
        "json.dump"
    ) as mock_json_dump:
        tracker = MLflowTracker(market="ES")
        parameters = {"learning_rate": 0.0003}
        metrics = {"sharpe_ratio": 1.5}
        artifacts = [tmp_dirs["artifact_path"]]
        tracker.log_run("sac", metrics, parameters, artifacts)
        mock_mlflow["log_artifact"].assert_called_with("run_summary.json")
        mock_json_dump.assert_called()
        with open("run_summary.json", "r") as f:
            summary = json.load(f)
            assert (
                summary["model_name"] == "sac"
            ), "Model_name incorrect dans run_summary.json"
            assert (
                summary["parameters"] == parameters
            ), "Parameters incorrects dans run_summary.json"
            assert (
                summary["metrics"] == metrics
            ), "Metrics incorrectes dans run_summary.json"
            assert "timestamp" in summary, "Timestamp manquant dans run_summary.json"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "log_run" and df["model_name"] == "sac"
        ), "Log log_run (sac) manquant"
        mock_alert.assert_called_with("Run journalisé pour sac sur ES", priority=2)


def test_log_run_failure(tmp_dirs, mock_mlflow):
    """Teste l'échec de la journalisation d'un run."""
    with patch("mlflow.start_run", side_effect=Exception("MLflow run failed")), patch(
        "src.utils.error_tracker.capture_error"
    ) as mock_capture_error, patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ) as mock_alert:
        tracker = MLflowTracker(market="ES")
        with pytest.raises(ValueError, match="Échec du run MLflow"):
            tracker.log_run("ppo_cvar", {"cvar_loss": 0.1}, {"cvar_alpha": 0.95}, [])
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "log_run" and df["success"] == False
        ), "Échec log_run non logué"
        mock_capture_error.assert_called()
        mock_alert.assert_called_with(pytest.any(str), priority=4)


def test_get_model_weights_success(tmp_dirs):
    """Teste la récupération réussie des poids bayésiens."""
    with patch("src.model.utils.alert_manager.AlertManager.send_alert") as mock_alert:
        tracker = MLflowTracker(market="ES")
        models = ["sac", "ppo", "ddpg"]
        weights = tracker.get_model_weights(models)
        assert weights == {
            "sac": 1 / 3,
            "ppo": 1 / 3,
            "ddpg": 1 / 3,
        }, "Poids bayésiens incorrects"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "get_model_weights"
        ), "Log get_model_weights manquant"
        mock_alert.assert_called_with(
            "Poids bayésiens récupérés pour sac,ppo,ddpg", priority=2
        )


def test_log_performance_metrics(tmp_dirs):
    """Teste la journalisation des métriques psutil dans log_performance."""
    with patch("src.model.utils.alert_manager.AlertManager.send_alert") as mock_alert:
        tracker = MLflowTracker(market="ES")
        tracker.log_performance("test_op", 0.1, True, model_name="sac")
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df["operation"] == "test_op"), "Log test_op manquant"
        assert all(
            col in df.columns for col in ["memory_usage_mb", "cpu_usage_percent"]
        ), "Colonnes psutil manquantes"
        mock_alert.assert_not_called()
