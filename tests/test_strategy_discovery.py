# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_strategy_discovery.py
# Tests unitaires pour src/model/strategy_discovery.py
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Valide la classe StrategyDiscovery pour la découverte et l’adaptation des stratégies de trading,
#        incluant le clustering avec PCA (Phase 7), l’optimisation CMA-ES, la validation des données (350/150 SHAP features),
#        les snapshots compressés, les sauvegardes, les graphiques matplotlib, les logs psutil, et les alertes (Phase 8).
#        Compatible avec la simulation de trading (Phase 12) et l’ensemble learning (Phase 16).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pandas>=2.0.0,<3.0.0
# - numpy>=1.26.4,<2.0.0
# - psutil>=5.9.8,<6.0.0
# - pyyaml>=6.0.0,<7.0.0
# - sklearn>=1.2.0,<2.0.0
# - cma>=3.2.0,<4.0.0
# - matplotlib>=3.7.0,<4.0.0
# - src.model.strategy_discovery
# - src.model.utils.miya_console
# - src.model.utils.config_manager
# - src.model.utils.alert_manager
# - src.utils.telegram_alert
# - src.envs.trading_env
#
# Inputs :
# - config/es_config.yaml (simulé)
# - config/feature_sets.yaml (simulé)
# - Données de trading simulées
#
# Outputs :
# - Assertions sur l’état du module
# - data/logs/strategy_discovery/strategy_discovery_performance.csv (simulé)
# - data/strategy_discovery/snapshots/*.json.gz (simulé)
# - data/strategies/*.json (simulé)
# - data/clusters.csv (simulé)
# - data/strategy_discovery_dashboard.json (simulé)
# - data/checkpoints/strategy_discovery/*.json.gz (simulé)
# - data/figures/strategy_discovery/*.png (simulé)
#
# Notes :
# - Utilise des mocks pour simuler TradingEnv, SQLite, et CMA-ES.
# - Vérifie l’absence de références à dxFeed, obs_t, 320/81 features.

import gzip
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.model.strategy_discovery import StrategyDiscovery

BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, snapshots, checkpoints, figures, et configurations."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs" / "strategy_discovery"
    logs_dir.mkdir(parents=True)
    snapshots_dir = data_dir / "strategy_discovery" / "snapshots"
    snapshots_dir.mkdir(parents=True)
    strategies_dir = data_dir / "strategies"
    strategies_dir.mkdir()
    checkpoints_dir = data_dir / "checkpoints" / "strategy_discovery"
    checkpoints_dir.mkdir(parents=True)
    figures_dir = data_dir / "figures" / "strategy_discovery"
    figures_dir.mkdir(parents=True)

    # Créer es_config.yaml
    config_path = config_dir / "es_config.yaml"
    config_content = {
        "clustering": {"n_clusters": 10, "random_state": 42},
        "optimization": {
            "max_iterations": 100,
            "entry_threshold_bounds": [20.0, 80.0],
            "exit_threshold_bounds": [20.0, 80.0],
            "position_size_bounds": [0.1, 1.0],
            "stop_loss_bounds": [0.5, 5.0],
        },
        "features": {"observation_dims": {"training": 350, "inference": 150}},
        "logging": {"buffer_size": 100},
        "cache": {"max_cache_size": 1000},
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    # Créer feature_sets.yaml
    feature_sets_path = config_dir / "feature_sets.yaml"
    feature_sets_content = {
        "training_features": [
            "timestamp",
            "bid_size_level_1",
            "ask_size_level_1",
            "trade_frequency_1s",
            "spread_avg_1min",
            "close",
            "predicted_volatility",
            "neural_regime",
            "rsi_14",
            "cnn_pressure",
        ]
        + [f"feat_{i}" for i in range(340)],
        "shap_features": [
            "bid_size_level_1",
            "ask_size_level_1",
            "trade_frequency_1s",
            "spread_avg_1min",
            "close",
            "predicted_volatility",
            "neural_regime",
            "rsi_14",
            "cnn_pressure",
        ]
        + [f"feat_{i}" for i in range(141)],
    }
    with open(feature_sets_path, "w", encoding="utf-8") as f:
        yaml.dump(feature_sets_content, f)

    return {
        "config_path": str(config_path),
        "feature_sets_path": str(feature_sets_path),
        "logs_dir": str(logs_dir),
        "snapshots_dir": str(snapshots_dir),
        "strategies_dir": str(strategies_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "figures_dir": str(figures_dir),
        "perf_log_path": str(logs_dir / "strategy_discovery_performance.csv"),
        "dashboard_path": str(data_dir / "strategy_discovery_dashboard.json"),
        "clusters_path": str(data_dir / "clusters.csv"),
    }


@pytest.fixture
def mock_data():
    """Crée des données simulées pour les tests."""
    feature_sets = {
        "training_features": [
            "timestamp",
            "bid_size_level_1",
            "ask_size_level_1",
            "trade_frequency_1s",
            "spread_avg_1min",
            "close",
            "predicted_volatility",
            "neural_regime",
            "rsi_14",
            "cnn_pressure",
        ]
        + [f"feat_{i}" for i in range(340)],
        "shap_features": [
            "bid_size_level_1",
            "ask_size_level_1",
            "trade_frequency_1s",
            "spread_avg_1min",
            "close",
            "predicted_volatility",
            "neural_regime",
            "rsi_14",
            "cnn_pressure",
        ]
        + [f"feat_{i}" for i in range(141)],
    }
    training_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=100, freq="1min"),
            **{
                col: (
                    np.random.uniform(0, 1, 100)
                    if col
                    not in [
                        "timestamp",
                        "neural_regime",
                        "predicted_volatility",
                        "cnn_pressure",
                        "rsi_14",
                    ]
                    else (
                        np.random.randint(0, 3, 100)
                        if col == "neural_regime"
                        else (
                            np.random.uniform(0, 2, 100)
                            if col == "predicted_volatility"
                            else (
                                np.random.uniform(-5, 5, 100)
                                if col == "cnn_pressure"
                                else (
                                    np.random.uniform(20, 80, 100)
                                    if col == "rsi_14"
                                    else np.random.uniform(0, 1, 100)
                                )
                            )
                        )
                    )
                )
                for col in feature_sets["training_features"]
            },
        }
    )
    inference_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=100, freq="1min"),
            **{
                col: (
                    np.random.uniform(0, 1, 100)
                    if col
                    not in [
                        "timestamp",
                        "neural_regime",
                        "predicted_volatility",
                        "cnn_pressure",
                        "rsi_14",
                    ]
                    else (
                        np.random.randint(0, 3, 100)
                        if col == "neural_regime"
                        else (
                            np.random.uniform(0, 2, 100)
                            if col == "predicted_volatility"
                            else (
                                np.random.uniform(-5, 5, 100)
                                if col == "cnn_pressure"
                                else (
                                    np.random.uniform(20, 80, 100)
                                    if col == "rsi_14"
                                    else np.random.uniform(0, 1, 100)
                                )
                            )
                        )
                    )
                )
                for col in feature_sets["shap_features"]
            },
        }
    )
    return {"training": training_data, "inference": inference_data}


@pytest.fixture
def discovery(tmp_dirs, monkeypatch):
    """Initialise StrategyDiscovery avec des mocks."""
    monkeypatch.setattr(
        "src.model.strategy_discovery.config_manager.get_config",
        lambda x: (
            {
                "clustering": {"n_clusters": 10, "random_state": 42},
                "optimization": {
                    "max_iterations": 100,
                    "entry_threshold_bounds": [20.0, 80.0],
                    "exit_threshold_bounds": [20.0, 80.0],
                    "position_size_bounds": [0.1, 1.0],
                    "stop_loss_bounds": [0.5, 5.0],
                },
                "features": {"observation_dims": {"training": 350, "inference": 150}},
                "logging": {"buffer_size": 100},
                "cache": {"max_cache_size": 1000},
            }
            if "es_config.yaml" in str(x)
            else {
                "training_features": [
                    "timestamp",
                    "bid_size_level_1",
                    "ask_size_level_1",
                    "trade_frequency_1s",
                    "spread_avg_1min",
                    "close",
                    "predicted_volatility",
                    "neural_regime",
                    "rsi_14",
                    "cnn_pressure",
                ]
                + [f"feat_{i}" for i in range(340)],
                "shap_features": [
                    "bid_size_level_1",
                    "ask_size_level_1",
                    "trade_frequency_1s",
                    "spread_avg_1min",
                    "close",
                    "predicted_volatility",
                    "neural_regime",
                    "rsi_14",
                    "cnn_pressure",
                ]
                + [f"feat_{i}" for i in range(141)],
            }
        ),
    )
    with patch("src.envs.trading_env.TradingEnv.__init__", return_value=None), patch(
        "src.envs.trading_env.TradingEnv.reset", return_value=(np.zeros(350), {})
    ), patch(
        "src.envs.trading_env.TradingEnv.step",
        return_value=(np.zeros(350), 1.0, False, False, {"price": 5100.0}),
    ), patch(
        "sqlite3.connect", return_value=MagicMock()
    ), patch(
        "cma.CMAEvolutionStrategy.optimize",
        return_value=MagicMock(xbest=[70.0, 30.0, 0.5, 2.0], fbest=-2.0),
    ), patch(
        "src.model.utils.alert_manager.AlertManager.send_alert", return_value=None
    ), patch(
        "src.utils.telegram_alert.send_telegram_alert", return_value=None
    ):
        discovery = StrategyDiscovery(config_path=tmp_dirs["config_path"])
    return discovery


def test_strategy_discovery_init(tmp_dirs, discovery):
    """Teste l’initialisation de StrategyDiscovery."""
    assert discovery.config["clustering"]["n_clusters"] == 10
    assert discovery.config["features"]["observation_dims"]["training"] == 350
    assert Path(tmp_dirs["perf_log_path"]).exists()
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert "cpu_percent" in df.columns
    assert "memory_usage_mb" in df.columns
    assert Path(tmp_dirs["checkpoints_dir"]).exists()
    checkpoints = list(Path(tmp_dirs["checkpoints_dir"]).glob("checkpoint_*.json.gz"))
    assert len(checkpoints) >= 1


def test_validate_data_training(tmp_dirs, discovery, mock_data):
    """Teste la validation des données en mode entraînement (350 features)."""
    discovery.validate_data(mock_data["training"], training_mode=True)
    assert Path(tmp_dirs["snapshots_dir"]).exists()
    snapshots = list(
        Path(tmp_dirs["snapshots_dir"]).glob("snapshot_validate_data_*.json.gz")
    )
    assert len(snapshots) >= 1
    assert Path(tmp_dirs["perf_log_path"]).exists()


def test_validate_data_inference(tmp_dirs, discovery, mock_data):
    """Teste la validation des données en mode inférence (150 SHAP features)."""
    discovery.validate_data(mock_data["inference"], training_mode=False)
    assert Path(tmp_dirs["snapshots_dir"]).exists()
    snapshots = list(
        Path(tmp_dirs["snapshots_dir"]).glob("snapshot_validate_data_*.json.gz")
    )
    assert len(snapshots) >= 1
    assert Path(tmp_dirs["perf_log_path"]).exists()


def test_validate_data_invalid_columns(tmp_dirs, discovery, mock_data):
    """Teste la validation avec des colonnes manquantes."""
    invalid_data = mock_data["training"].drop(columns=["bid_size_level_1"])
    with pytest.raises(ValueError, match="Colonnes manquantes dans les données"):
        discovery.validate_data(invalid_data, training_mode=True)
    snapshots = list(
        Path(tmp_dirs["snapshots_dir"]).glob("snapshot_validate_data_*.json.gz")
    )
    assert len(snapshots) >= 1


def test_generate_clusters(tmp_dirs, discovery, mock_data):
    """Teste la génération des clusters avec PCA et K-means."""
    clusters_df = discovery.generate_clusters(
        mock_data["training"], n_clusters=10, training_mode=True
    )
    assert len(clusters_df) == 10
    assert "cluster_id" in clusters_df.columns
    assert Path(tmp_dirs["clusters_path"]).exists()
    assert Path(tmp_dirs["snapshots_dir"]).exists()
    snapshots = list(
        Path(tmp_dirs["snapshots_dir"]).glob("snapshot_clusters_*.json.gz")
    )
    assert len(snapshots) >= 1
    assert Path(tmp_dirs["figures_dir"]).exists()
    figures = list(Path(tmp_dirs["figures_dir"]).glob("clusters_*.png"))
    assert len(figures) >= 1
    assert Path(tmp_dirs["perf_log_path"]).exists()


def test_optimize_strategy(tmp_dirs, discovery, mock_data):
    """Teste l’optimisation des paramètres de stratégie avec CMA-ES."""
    initial_params = {
        "entry_threshold": 70.0,
        "exit_threshold": 30.0,
        "position_size": 0.5,
        "stop_loss": 2.0,
    }
    env = MagicMock()
    optimized_params = discovery.optimize_strategy(
        initial_params, mock_data["training"], env, training_mode=True
    )
    assert "entry_threshold" in optimized_params
    assert Path(tmp_dirs["strategies_dir"]).exists()
    strategies = list(Path(tmp_dirs["strategies_dir"]).glob("strategy_params_*.json"))
    assert len(strategies) >= 1
    assert Path(tmp_dirs["snapshots_dir"]).exists()
    snapshots = list(
        Path(tmp_dirs["snapshots_dir"]).glob("snapshot_strategy_params_*.json.gz")
    )
    assert len(snapshots) >= 1
    assert Path(tmp_dirs["perf_log_path"]).exists()


def test_adapt_strategy(tmp_dirs, discovery, mock_data):
    """Teste l’adaptation d’une stratégie en fonction du contexte."""
    context = {"neural_regime": 0, "predicted_volatility": 1.5}
    env = MagicMock()
    discovery.last_checkpoint_time = datetime.now() - timedelta(
        seconds=400
    )  # Permettre la sauvegarde
    adapted_params = discovery.adapt_strategy(
        mock_data["training"], context, env, training_mode=True
    )
    assert "entry_threshold" in adapted_params
    assert Path(tmp_dirs["strategies_dir"]).exists()
    strategies = list(Path(tmp_dirs["strategies_dir"]).glob("strategy_params_*.json"))
    assert len(strategies) >= 1
    assert Path(tmp_dirs["snapshots_dir"]).exists()
    snapshots = list(
        Path(tmp_dirs["snapshots_dir"]).glob("snapshot_adapted_strategy_*.json.gz")
    )
    assert len(snapshots) >= 1
    assert Path(tmp_dirs["checkpoints_dir"]).exists()
    checkpoints = list(Path(tmp_dirs["checkpoints_dir"]).glob("checkpoint_*.json.gz"))
    assert len(checkpoints) >= 1
    assert Path(tmp_dirs["dashboard_path"]).exists()


def test_handle_sigint(tmp_dirs, discovery):
    """Teste la gestion de SIGINT."""
    with patch("sys.exit") as mock_exit:
        discovery.handle_sigint(None, None)
    snapshots = list(Path(tmp_dirs["snapshots_dir"]).glob("snapshot_sigint_*.json.gz"))
    assert len(snapshots) == 1
    checkpoints = list(Path(tmp_dirs["checkpoints_dir"]).glob("checkpoint_*.json.gz"))
    assert len(checkpoints) >= 1
    with gzip.open(snapshots[0], "rt") as f:
        snapshot = json.load(f)
    assert snapshot["data"]["status"] == "SIGINT"
    mock_exit.assert_called_with(0)


def test_alerts(tmp_dirs, discovery, mock_data):
    """Teste les alertes via alert_manager et telegram_alert."""
    with patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ) as mock_alert, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        discovery.validate_data(mock_data["training"], training_mode=True)
    mock_alert.assert_called()
    mock_telegram.assert_called()


def test_no_obsolete_references(tmp_dirs):
    """Vérifie l'absence de références à dxFeed, obs_t, 320/81 features."""
    for file_path in [tmp_dirs["config_path"], tmp_dirs["feature_sets_path"]]:
        with open(file_path, "r") as f:
            content = f.read()
        assert "dxFeed" not in content, f"Référence à dxFeed trouvée dans {file_path}"
        assert "obs_t" not in content, f"Référence à obs_t trouvée dans {file_path}"
        assert (
            "320 features" not in content
        ), f"Référence à 320 features trouvée dans {file_path}"
        assert (
            "81 features" not in content
        ), f"Référence à 81 features trouvée dans {file_path}"
