# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_hyperparam_optimizer.py
# Tests unitaires pour src/model/utils/hyperparam_optimizer.py
#
# Version : 2.1.5
# Date : 2025-05-14
#
# Rôle : Valide l'optimisation des hyperparamètres pour les modèles SAC, PPO, DDPG, PPO-Lagrangian, QR-DQN,
#        et ensembles de politiques avec Optuna. Teste les fonctionnalités : logs psutil,
#        gestion des erreurs avec error_tracker.py, alertes via alert_manager.py, espaces de recherche spécifiques,
#        validation des paramètres, et journalisation dans la table best_hyperparams.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, psutil>=5.9.8,<6.0.0, sqlite3, json, optuna
# - src/model/utils/hyperparam_optimizer.py
# - src/utils/error_tracker.py
# - src/model/utils/alert_manager.py
#
# Inputs :
# - Données factices pour les essais Optuna.
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de hyperparam_optimizer.py.
#
# Notes :
# - Conforme à structure.txt (version 2.1.5, 2025-05-14).
# - Supporte les suggestions 3 (simulation configurable), 7 (Safe RL/CVaR-PPO), 8 (Distributional RL/QR-DQN),
#   9 (réentraînement), 10 (Ensembles de politiques).
# - Vérifie les logs psutil dans data/logs/hyperparam_optimizer_performance.csv.
# - Vérifie la validation des poids bayésiens, des paramètres, et la journalisation dans best_hyperparams.
# - Policies Note: The official directory for routing policies is src/model/router/policies.
# The src/model/policies directory is a residual and should be verified
# for removal to avoid import conflicts.

import os
import sqlite3
from unittest.mock import MagicMock, patch

import optuna
import pandas as pd
import pytest

from src.model.utils.hyperparam_optimizer import HyperparamOptimizer


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
                parameters TEXT,
                metrics TEXT,
                timestamp TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE best_hyperparams (
                model_name TEXT,
                parameters TEXT,
                sharpe REAL,
                drawdown REAL,
                timestamp TEXT
            )
            """
        )
    return {
        "base_dir": str(base_dir),
        "logs_dir": str(logs_dir),
        "perf_log_path": str(logs_dir / "hyperparam_optimizer_performance.csv"),
        "log_file_path": str(logs_dir / "hyperparam_optimizer.log"),
        "db_path": str(db_path),
    }


@pytest.fixture
def mock_optuna():
    """Mock pour Optuna."""
    with patch("optuna.create_study") as mock_create_study:
        mock_study = MagicMock()
        mock_study.best_params = {"learning_rate": 0.0003}
        mock_study.best_values = [1.5, 0.2]
        mock_create_study.return_value = mock_study
        yield mock_study


def test_hyperparam_optimizer_init_success(tmp_dirs, mock_optuna):
    """Teste l'initialisation réussie de HyperparamOptimizer (basé sur test_init)."""
    with patch("sqlite3.connect") as mock_connect, patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ) as mock_alert:
        optimizer = HyperparamOptimizer(market="ES")
        assert optimizer.market == "ES", "Marché incorrect"
        mock_optuna.assert_called_with(
            study_name="online_hpo_ES",
            storage=f"sqlite:///{tmp_dirs['base_dir']}/data/market_memory.db",
            directions=["maximize", "minimize"],
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            sampler=optuna.samplers.TPESampler(n_startup_trials=10),
        )
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df["operation"] == "init_optuna"), "Log init_optuna manquant"
        assert all(
            col in df.columns for col in ["memory_usage_mb", "cpu_usage_percent"]
        ), "Colonnes psutil manquantes"
        mock_alert.assert_called_with("Secret récupéré: init_optuna...", priority=2)


def test_hyperparam_optimizer_init_failure(tmp_dirs):
    """Teste l'échec de l'initialisation de HyperparamOptimizer."""
    with patch(
        "optuna.create_study", side_effect=Exception("Optuna init failed")
    ), patch("src.utils.error_tracker.capture_error") as mock_capture_error, patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ) as mock_alert:
        with pytest.raises(ValueError, match="Échec de l’initialisation d’Optuna"):
            HyperparamOptimizer(market="ES")
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "init_optuna" and df["success"] == False
        ), "Échec init_optuna non logué"
        mock_capture_error.assert_called()
        mock_alert.assert_called_with(pytest.any(str), priority=4)


def test_objective_sac(tmp_dirs, mock_optuna):
    """Teste la fonction objective pour SAC (basé sur test_objective)."""
    with patch(
        "src.model.trade_probability.quick_train_and_eval",
        return_value={"sharpe": 1.5, "drawdown": 0.1, "cpu_usage": 75},
    ), patch(
        "src.model.utils.mlflow_tracker.MLflowTracker.log_run"
    ) as mock_log_run, patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ) as mock_alert:
        optimizer = HyperparamOptimizer(market="ES")
        trial = optuna.trial.FixedTrial(
            {
                "learning_rate": 0.0001,
                "gamma": 0.99,
                "batch_size": 128,
                "n_layers": 2,
                "hidden_dim": 64,
                "tau": 0.005,
                "clip_ratio": 0.2,
                "entropy_coef": 0.01,
                "actor_lr": 0.0001,
                "critic_lr": 0.0001,
                "n_steps": 500,
            }
        )
        sharpe, drawdown = optimizer.objective(trial, "sac")
        assert sharpe == 1.5, "Sharpe ratio incorrect"
        assert drawdown == 0.1, "Drawdown incorrect"
        mock_log_run.assert_called()
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "objective" and df["model_name"] == "sac"
        ), "Log objective (sac) manquant"
        mock_alert.assert_not_called()


def test_optimize_success(tmp_dirs, mock_optuna):
    """Teste l'optimisation réussie des hyperparamètres (basé sur test_optimize)."""
    with patch(
        "src.model.trade_probability.quick_train_and_eval",
        return_value={"sharpe": 1.5, "drawdown": 0.2, "cpu_usage": 50},
    ), patch(
        "src.model.utils.mlflow_tracker.MLflowTracker.log_run"
    ) as mock_log_run, patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ) as mock_alert, patch(
        "sqlite3.connect"
    ) as mock_connect:
        optimizer = HyperparamOptimizer(market="ES")
        mock_optuna.best_params = {
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "batch_size": 128,
        }
        result = optimizer.optimize("sac", n_trials=5, timeout=600)
        assert isinstance(result, dict), "Résultat n'est pas un dictionnaire"
        assert result["sharpe"] == 1.5, "Sharpe ratio incorrect"
        assert result["drawdown"] == 0.2, "Drawdown incorrect"
        assert "learning_rate" in result, "learning_rate manquant"
        mock_optuna.optimize.assert_called_with(
            pytest.any, n_trials=5, timeout=600, n_jobs=4
        )
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "optimize" and df["model_name"] == "sac"
        ), "Log optimize (sac) manquant"
        assert all(
            col in df.columns for col in ["memory_usage_mb", "cpu_usage_percent"]
        ), "Colonnes psutil manquantes"
        with sqlite3.connect(tmp_dirs["db_path"]) as conn:
            df_db = pd.read_sql("SELECT * FROM best_hyperparams", conn)
            assert any(
                df_db["model_name"] == "sac"
            ), "Meilleurs hyperparamètres non enregistrés dans best_hyperparams"
        mock_alert.assert_called_with(
            "Hyperparamètres optimisés pour sac sur ES", priority=2
        )


def test_optimize_new_models(tmp_dirs, mock_optuna):
    """Teste l'optimisation pour les nouveaux modèles."""
    with patch(
        "src.model.trade_probability.quick_train_and_eval",
        return_value={"sharpe": 1.5, "drawdown": 0.2, "cpu_usage": 50},
    ), patch(
        "src.model.utils.mlflow_tracker.MLflowTracker.log_run"
    ) as mock_log_run, patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ) as mock_alert, patch(
        "sqlite3.connect"
    ) as mock_connect:
        optimizer = HyperparamOptimizer(market="ES")
        scenarios = [
            (
                "ppo_cvar",
                {"cvar_alpha": 0.95, "learning_rate": 0.0003},
                "cvar_alpha",
            ),  # Suggestion 7
            (
                "qr_dqn",
                {"quantiles": 51, "learning_rate": 0.0003},
                "quantiles",
            ),  # Suggestion 8
            (
                "ensemble",
                {"sac_weight": 0.4, "ppo_weight": 0.3, "ddpg_weight": 0.3},
                "sac_weight",
            ),  # Suggestion 10
        ]
        for model_name, expected_params, key_param in scenarios:
            mock_optuna.best_params = expected_params
            result = optimizer.optimize(model_name, n_trials=1)
            assert isinstance(
                result, dict
            ), f"Résultat pour {model_name} n'est pas un dictionnaire"
            assert (
                key_param in result
            ), f"Paramètre {key_param} manquant pour {model_name}"
            mock_optuna.optimize.assert_called()
            assert os.path.exists(
                tmp_dirs["perf_log_path"]
            ), "Fichier de logs de performance non créé"
            df = pd.read_csv(tmp_dirs["perf_log_path"])
            assert any(
                df["operation"] == "optimize" and df["model_name"] == model_name
            ), f"Log optimize ({model_name}) manquant"
            with sqlite3.connect(tmp_dirs["db_path"]) as conn:
                df_db = pd.read_sql("SELECT * FROM best_hyperparams", conn)
                assert any(
                    df_db["model_name"] == model_name
                ), f"Meilleurs hyperparamètres non enregistrés pour {model_name}"
            mock_alert.assert_called_with(
                f"Hyperparamètres optimisés pour {model_name} sur ES", priority=2
            )


def test_optimize_failure(tmp_dirs, mock_optuna):
    """Teste l'échec de l'optimisation des hyperparamètres."""
    with patch(
        "optuna.Study.optimize", side_effect=Exception("Optuna optimize failed")
    ), patch("src.utils.error_tracker.capture_error") as mock_capture_error, patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ) as mock_alert:
        optimizer = HyperparamOptimizer(market="ES")
        result = optimizer.optimize("ppo_cvar", n_trials=1)
        assert result == {}, "Résultat devrait être vide en cas d'échec"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "optimize" and df["success"] == False
        ), "Échec optimize non logué"
        mock_capture_error.assert_called()
        mock_alert.assert_called_with(pytest.any(str), priority=4)


def test_validate_params_ensemble(tmp_dirs):
    """Teste la validation des poids bayésiens pour ensemble."""
    optimizer = HyperparamOptimizer(market="ES")
    valid_params = {
        "sac_weight": 0.4,
        "ppo_weight": 0.3,
        "ddpg_weight": 0.3,
        "learning_rate": 0.0003,
    }
    optimizer._validate_params(valid_params, "ensemble")
    invalid_params = {
        "sac_weight": 0.5,
        "ppo_weight": 0.4,
        "ddpg_weight": 0.2,
        "learning_rate": 0.0003,
    }
    with pytest.raises(ValueError, match="Les poids bayésiens doivent sommer à 1.0"):
        optimizer._validate_params(invalid_params, "ensemble")
    missing_params = {"sac_weight": 0.4, "ppo_weight": 0.3}
    with pytest.raises(ValueError, match="Paramètres manquants pour ensemble"):
        optimizer._validate_params(missing_params, "ensemble")


def test_log_best_to_db(tmp_dirs):
    """Teste la journalisation dans best_hyperparams."""
    with patch("src.model.utils.alert_manager.AlertManager.send_alert") as mock_alert:
        optimizer = HyperparamOptimizer(market="ES")
        best_params = {"learning_rate": 0.0003, "sharpe": 1.5, "drawdown": 0.2}
        optimizer._log_best_to_db("sac", best_params)
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "log_best_to_db" and df["model_name"] == "sac"
        ), "Log log_best_to_db (sac) manquant"
        with sqlite3.connect(tmp_dirs["db_path"]) as conn:
            df_db = pd.read_sql("SELECT * FROM best_hyperparams", conn)
            assert any(
                df_db["model_name"] == "sac"
            ), "Meilleurs hyperparamètres non enregistrés dans best_hyperparams"
        mock_alert.assert_not_called()


def test_log_performance_metrics(tmp_dirs):
    """Teste la journalisation des métriques psutil dans log_performance."""
    with patch("src.model.utils.alert_manager.AlertManager.send_alert") as mock_alert:
        optimizer = HyperparamOptimizer(market="ES")
        optimizer.log_performance("test_op", 0.1, True, model_name="sac")
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df["operation"] == "test_op"), "Log test_op manquant"
        assert all(
            col in df.columns for col in ["memory_usage_mb", "cpu_usage_percent"]
        ), "Colonnes psutil manquantes"
        mock_alert.assert_not_called()
