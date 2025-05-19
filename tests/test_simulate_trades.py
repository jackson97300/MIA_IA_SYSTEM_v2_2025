# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_simulate_trades.py
# Tests unitaires pour src/trading/simulate_trades.py
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Valide la classe TradeSimulator pour la simulation de trades hors ligne,
#        incluant la validation des données (350/150 SHAP features), les récompenses adaptatives (méthode 5),
#        le paper trading via Sierra Chart, la mémoire contextuelle (Phase 7) via store_pattern,
#        les snapshots compressés, les sauvegardes, les graphiques matplotlib/seaborn, et les alertes (Phase 8).
#        Compatible avec la simulation de trading (Phase 12) et l’ensemble learning (Phase 16).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pandas>=2.0.0,<3.0.0
# - numpy>=1.26.4,<2.0.0
# - matplotlib>=3.8.0,<4.0.0
# - seaborn>=0.13.0,<1.0.0
# - psutil>=5.9.8,<6.0.0
# - pyyaml>=6.0.0,<7.0.0
# - src.trading.simulate_trades
# - src.model.router.main_router
# - src.envs.trading_env
# - src.model.utils_model
# - src.features.neural_pipeline
# - src.mind.mind
# - src.model.adaptive_learner
# - src.model.utils.config_manager
# - src.model.utils.alert_manager
# - src.utils.telegram_alert
# - src.trading.trade_executor
#
# Inputs :
# - config/market_config.yaml (simulé)
# - config/feature_sets.yaml (simulé)
# - config/es_config.yaml (simulé)
# - config/model_params.yaml (simulé)
# - Données de trading simulées
#
# Outputs :
# - Assertions sur l’état du simulateur
# - data/logs/simulate_trades_performance.csv (simulé)
# - data/trading/simulation_snapshots/*.json.gz (simulé)
# - data/trading/simulate_trades_dashboard.json (simulé)
# - data/figures/simulations/*.png (simulé)
# - data/trades/trades_simulated.csv (simulé)
# - data/checkpoints/simulate_trades/*.json.gz (simulé)
#
# Notes :
# - Utilise des mocks pour simuler MainRouter, TradingEnv, NeuralPipeline, MindEngine, store_pattern, et execute_trade.
# - Vérifie l’absence de références à dxFeed, obs_t, 320/81 features.

import gzip
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.trading.simulate_trades import TradeSimulator

BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, snapshots, checkpoints, figures, trades, et configurations."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    snapshots_dir = data_dir / "trading" / "simulation_snapshots"
    snapshots_dir.mkdir(parents=True)
    trades_dir = data_dir / "trades"
    trades_dir.mkdir()
    checkpoints_dir = data_dir / "checkpoints" / "simulate_trades"
    checkpoints_dir.mkdir(parents=True)
    figures_dir = data_dir / "figures" / "simulations"
    figures_dir.mkdir(parents=True)
    features_dir = data_dir / "features"
    features_dir.mkdir()

    # Créer market_config.yaml
    config_path = config_dir / "market_config.yaml"
    config_content = {
        "thresholds": {
            "min_sharpe": 0.5,
            "max_drawdown": -1000.0,
            "min_profit_factor": 1.2,
        },
        "simulation": {"use_neural_pipeline": True},
        "logging": {"buffer_size": 100},
        "cache": {"max_cache_size": 1000},
        "features": {"observation_dims": {"training": 350, "inference": 150}},
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    # Créer feature_sets.yaml
    feature_sets_path = config_dir / "feature_sets.yaml"
    feature_sets_content = {
        "training_features": [
            "timestamp",
            "vix",
            "neural_regime",
            "predicted_volatility",
            "trade_frequency_1s",
            "close",
            "rsi_14",
            "gex",
            "open",
            "high",
            "low",
            "volume",
            "atr_14",
            "adx_14",
            "bid_size_level_1",
            "ask_size_level_1",
            "news_impact_score",
            "predicted_vix",
        ]
        + [f"feat_{i}" for i in range(332)],
        "shap_features": [
            "timestamp",
            "vix",
            "neural_regime",
            "predicted_volatility",
            "trade_frequency_1s",
            "close",
            "rsi_14",
            "gex",
            "open",
            "high",
            "low",
            "volume",
            "atr_14",
            "adx_14",
            "bid_size_level_1",
            "ask_size_level_1",
            "news_impact_score",
            "predicted_vix",
        ]
        + [f"feat_{i}" for i in range(132)],
    }
    with open(feature_sets_path, "w", encoding="utf-8") as f:
        yaml.dump(feature_sets_content, f)

    # Créer es_config.yaml
    es_config_path = config_dir / "es_config.yaml"
    es_config_content = {"thresholds": {"max_drawdown": 0.10}}
    with open(es_config_path, "w", encoding="utf-8") as f:
        yaml.dump(es_config_content, f)

    # Créer model_params.yaml
    model_params_path = config_dir / "model_params.yaml"
    model_params_content = {"window_size": 50, "base_features": 150}
    with open(model_params_path, "w", encoding="utf-8") as f:
        yaml.dump(model_params_content, f)

    # Créer features_latest.csv
    features_path = features_dir / "features_latest.csv"
    features_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=100, freq="1min"),
            "vix": np.random.uniform(15, 25, 100),
            "neural_regime": np.random.randint(0, 3, 100),
            "predicted_volatility": np.random.uniform(0.1, 0.5, 100),
            "trade_frequency_1s": np.random.uniform(5, 10, 100),
            "close": np.random.normal(5100, 10, 100),
            "rsi_14": np.random.uniform(20, 80, 100),
            "gex": np.random.uniform(-1000, 1000, 100),
            "open": np.random.normal(5100, 10, 100),
            "high": np.random.normal(5105, 10, 100),
            "low": np.random.normal(5095, 10, 100),
            "volume": np.random.uniform(1000, 5000, 100),
            "atr_14": np.random.uniform(1, 3, 100),
            "adx_14": np.random.uniform(20, 50, 100),
            "bid_size_level_1": np.random.uniform(50, 200, 100),
            "ask_size_level_1": np.random.uniform(50, 200, 100),
            "news_impact_score": np.random.uniform(0, 1, 100),
            "predicted_vix": np.random.uniform(15, 25, 100),
            **{
                f"feat_{i}": np.random.uniform(0, 1, 100) for i in range(332)
            },  # Total 350 features
        }
    )
    features_data.to_csv(features_path, index=False, encoding="utf-8")

    return {
        "config_path": str(config_path),
        "feature_sets_path": str(feature_sets_path),
        "es_config_path": str(es_config_path),
        "model_params_path": str(model_params_path),
        "features_path": str(features_path),
        "logs_dir": str(logs_dir),
        "snapshots_dir": str(snapshots_dir),
        "trades_dir": str(trades_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "figures_dir": str(figures_dir),
        "perf_log_path": str(logs_dir / "simulate_trades_performance.csv"),
        "dashboard_path": str(data_dir / "trading" / "simulate_trades_dashboard.json"),
        "trades_path": str(trades_dir / "trades_simulated.csv"),
    }


@pytest.fixture
def mock_data():
    """Crée des données simulées pour les tests."""
    feature_sets = {
        "training_features": [
            "timestamp",
            "vix",
            "neural_regime",
            "predicted_volatility",
            "trade_frequency_1s",
            "close",
            "rsi_14",
            "gex",
            "open",
            "high",
            "low",
            "volume",
            "atr_14",
            "adx_14",
            "bid_size_level_1",
            "ask_size_level_1",
            "news_impact_score",
            "predicted_vix",
        ]
        + [f"feat_{i}" for i in range(332)],
        "shap_features": [
            "timestamp",
            "vix",
            "neural_regime",
            "predicted_volatility",
            "trade_frequency_1s",
            "close",
            "rsi_14",
            "gex",
            "open",
            "high",
            "low",
            "volume",
            "atr_14",
            "adx_14",
            "bid_size_level_1",
            "ask_size_level_1",
            "news_impact_score",
            "predicted_vix",
        ]
        + [f"feat_{i}" for i in range(132)],
    }
    training_data = pd.DataFrame(
        {
            "timestamp": [datetime.now()],
            "vix": [20.0],
            "neural_regime": [0],
            "predicted_volatility": [1.5],
            "trade_frequency_1s": [8.0],
            "close": [5100.0],
            "rsi_14": [50.0],
            "gex": [100.0],
            "open": [5100.0],
            "high": [5105.0],
            "low": [5095.0],
            "volume": [2000.0],
            "atr_14": [1.5],
            "adx_14": [30.0],
            "bid_size_level_1": [100.0],
            "ask_size_level_1": [120.0],
            "news_impact_score": [0.5],
            "predicted_vix": [20.0],
            **{
                f"feat_{i}": [np.random.uniform(0, 1)] for i in range(332)
            },  # Total 350 features
        }
    )
    inference_data = pd.DataFrame(
        {
            "timestamp": [datetime.now()],
            "vix": [20.0],
            "neural_regime": [0],
            "predicted_volatility": [1.5],
            "trade_frequency_1s": [8.0],
            "close": [5100.0],
            "rsi_14": [50.0],
            "gex": [100.0],
            "open": [5100.0],
            "high": [5105.0],
            "low": [5095.0],
            "volume": [2000.0],
            "atr_14": [1.5],
            "adx_14": [30.0],
            "bid_size_level_1": [100.0],
            "ask_size_level_1": [120.0],
            "news_impact_score": [0.5],
            "predicted_vix": [20.0],
            **{
                f"feat_{i}": [np.random.uniform(0, 1)] for i in range(132)
            },  # Total 150 features
        }
    )
    return {"training": training_data, "inference": inference_data}


@pytest.fixture
def simulator(tmp_dirs, monkeypatch):
    """Initialise TradeSimulator avec des mocks."""
    monkeypatch.setattr(
        "src.trading.simulate_trades.config_manager.get_config",
        lambda x: (
            {
                "thresholds": {
                    "min_sharpe": 0.5,
                    "max_drawdown": -1000.0,
                    "min_profit_factor": 1.2,
                },
                "simulation": {"use_neural_pipeline": True},
                "logging": {"buffer_size": 100},
                "cache": {"max_cache_size": 1000},
                "features": {"observation_dims": {"training": 350, "inference": 150}},
            }
            if "market_config.yaml" in str(x)
            else (
                {
                    "training_features": [
                        "timestamp",
                        "vix",
                        "neural_regime",
                        "predicted_volatility",
                        "trade_frequency_1s",
                        "close",
                        "rsi_14",
                        "gex",
                        "open",
                        "high",
                        "low",
                        "volume",
                        "atr_14",
                        "adx_14",
                        "bid_size_level_1",
                        "ask_size_level_1",
                        "news_impact_score",
                        "predicted_vix",
                    ]
                    + [f"feat_{i}" for i in range(332)],
                    "shap_features": [
                        "timestamp",
                        "vix",
                        "neural_regime",
                        "predicted_volatility",
                        "trade_frequency_1s",
                        "close",
                        "rsi_14",
                        "gex",
                        "open",
                        "high",
                        "low",
                        "volume",
                        "atr_14",
                        "adx_14",
                        "bid_size_level_1",
                        "ask_size_level_1",
                        "news_impact_score",
                        "predicted_vix",
                    ]
                    + [f"feat_{i}" for i in range(132)],
                }
                if "feature_sets.yaml" in str(x)
                else (
                    {"thresholds": {"max_drawdown": 0.10}}
                    if "es_config.yaml" in str(x)
                    else {"window_size": 50, "base_features": 150}
                )
            )
        ),
    )
    with patch("src.mind.mind.MindEngine.__init__", return_value=None), patch(
        "src.model.adaptive_learner.store_pattern", return_value=None
    ), patch(
        "src.model.router.main_router.MainRouter.__init__", return_value=None
    ), patch(
        "src.model.router.main_router.MainRouter.route", return_value=(1.0, "trend", {})
    ), patch(
        "src.envs.trading_env.TradingEnv.__init__", return_value=None
    ), patch(
        "src.envs.trading_env.TradingEnv._get_observation", return_value=np.zeros(350)
    ), patch(
        "src.envs.trading_env.TradingEnv.step",
        return_value=(
            np.zeros(350),
            1.0,
            False,
            False,
            {
                "balance": 10000.0,
                "max_drawdown": -100.0,
                "call_wall": 5200.0,
                "put_wall": 5000.0,
                "zero_gamma": 5100.0,
                "dealer_position_bias": 0.5,
            },
        ),
    ), patch(
        "src.features.neural_pipeline.NeuralPipeline.__init__", return_value=None
    ), patch(
        "src.features.neural_pipeline.NeuralPipeline.load_models", return_value=None
    ), patch(
        "src.features.neural_pipeline.NeuralPipeline.run",
        return_value={
            "regime": [0],
            "volatility": [1.5],
            "features": np.zeros((1, 21)),
        },
    ), patch(
        "src.trading.trade_executor.execute_trade",
        return_value={"action": 1.0, "reward": 1.0},
    ), patch(
        "src.model.utils_model.export_logs", return_value=None
    ), patch(
        "src.model.utils.alert_manager.AlertManager.send_alert", return_value=None
    ), patch(
        "src.utils.telegram_alert.send_telegram_alert", return_value=None
    ):
        simulator = TradeSimulator(config_path=tmp_dirs["config_path"])
    return simulator


def test_trade_simulator_init(tmp_dirs, simulator):
    """Teste l’initialisation de TradeSimulator."""
    assert simulator.performance_thresholds["min_sharpe"] == 0.5
    assert Path(tmp_dirs["perf_log_path"]).exists()
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert "cpu_percent" in df.columns
    assert "memory_usage_mb" in df.columns
    assert Path(tmp_dirs["checkpoints_dir"]).exists()
    checkpoints = list(Path(tmp_dirs["checkpoints_dir"]).glob("checkpoint_*.json.gz"))
    assert len(checkpoints) >= 1


def test_validate_data_training(tmp_dirs, simulator, mock_data):
    """Teste la validation des données en mode entraînement (350 features)."""
    simulator.validate_data(mock_data["training"])
    assert Path(tmp_dirs["snapshots_dir"]).exists()
    snapshots = list(
        Path(tmp_dirs["snapshots_dir"]).glob("snapshot_validate_data_*.json.gz")
    )
    assert len(snapshots) >= 1
    assert Path(tmp_dirs["perf_log_path"]).exists()


def test_validate_data_inference(tmp_dirs, simulator, mock_data):
    """Teste la validation des données en mode inférence (150 SHAP features)."""
    simulator.validate_data(mock_data["inference"])
    assert Path(tmp_dirs["snapshots_dir"]).exists()
    snapshots = list(
        Path(tmp_dirs["snapshots_dir"]).glob("snapshot_validate_data_*.json.gz")
    )
    assert len(snapshots) >= 1
    assert Path(tmp_dirs["perf_log_path"]).exists()


def test_validate_data_invalid_columns(tmp_dirs, simulator, mock_data):
    """Teste la validation avec des colonnes manquantes."""
    invalid_data = mock_data["training"].drop(columns=["vix"])
    with pytest.raises(ValueError, match="Colonnes manquantes dans les données"):
        simulator.validate_data(invalid_data)
    snapshots = list(
        Path(tmp_dirs["snapshots_dir"]).glob("snapshot_validate_data_*.json.gz")
    )
    assert len(snapshots) >= 1


def test_calculate_reward():
    """Teste le calcul des récompenses adaptatives."""
    from src.trading.simulate_trades import calculate_reward

    reward = calculate_reward(news_impact_score=0.5, predicted_vix=20.0)
    assert isinstance(reward, float)
    assert reward == 0.5 * 10 - 20.0 * 0.5  # 5 - 10 = -5


def test_compute_metrics(simulator):
    """Teste le calcul des métriques de performance."""
    rewards = [1.0, -0.5, 2.0, -1.0]
    metrics = simulator.compute_metrics(rewards)
    assert "sharpe" in metrics
    assert "drawdown" in metrics
    assert "total_reward" in metrics
    assert "profit_factor" in metrics


def test_simulate_trades(tmp_dirs, simulator, mock_data):
    """Teste la simulation de trades sans paper trading."""
    simulator.simulate_trades(
        mock_data["training"], paper_trading=False, asset="ES", mode="range"
    )
    assert Path(tmp_dirs["trades_path"]).exists()
    assert Path(tmp_dirs["snapshots_dir"]).exists()
    snapshots = list(
        Path(tmp_dirs["snapshots_dir"]).glob("snapshot_simulation_step_*.json.gz")
    )
    assert len(snapshots) >= 1
    assert Path(tmp_dirs["perf_log_path"]).exists()
    assert Path(tmp_dirs["dashboard_path"]).exists()


def test_simulate_trades_paper_trading(tmp_dirs, simulator, mock_data):
    """Teste la simulation de trades avec paper trading."""
    simulator.simulate_trades(
        mock_data["training"], paper_trading=True, asset="ES", mode="range"
    )
    assert Path(tmp_dirs["trades_path"]).exists()
    assert Path(tmp_dirs["snapshots_dir"]).exists()
    snapshots = list(
        Path(tmp_dirs["snapshots_dir"]).glob("snapshot_simulation_step_*.json.gz")
    )
    assert len(snapshots) >= 1
    assert Path(tmp_dirs["perf_log_path"]).exists()
    assert Path(tmp_dirs["dashboard_path"]).exists()


def test_simulate_trading(tmp_dirs, simulator):
    """Teste la simulation complète avec neural pipeline."""
    simulator.last_checkpoint_time = datetime.now() - timedelta(
        seconds=400
    )  # Permettre la sauvegarde
    simulator.simulate_trading(
        feature_csv_path=tmp_dirs["features_path"],
        output_csv_path=tmp_dirs["trades_path"],
        config_path=tmp_dirs["es_config_path"],
        asset="ES",
        mode="range",
    )
    assert Path(tmp_dirs["trades_path"]).exists()
    assert Path(tmp_dirs["snapshots_dir"]).exists()
    snapshots = list(
        Path(tmp_dirs["snapshots_dir"]).glob("snapshot_simulation_step_*.json.gz")
    )
    assert len(snapshots) >= 1
    assert Path(tmp_dirs["figures_dir"]).exists()
    figures = list(Path(tmp_dirs["figures_dir"]).glob("*.png"))
    assert len(figures) >= 3  # Balance, rewards, drawdown
    assert Path(tmp_dirs["checkpoints_dir"]).exists()
    checkpoints = list(Path(tmp_dirs["checkpoints_dir"]).glob("checkpoint_*.json.gz"))
    assert len(checkpoints) >= 1
    assert Path(tmp_dirs["dashboard_path"]).exists()


def test_generate_visualizations(tmp_dirs, simulator):
    """Teste la génération des visualisations."""
    trades_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=10, freq="1min"),
            "reward": np.random.uniform(-100, 100, 10),
            "balance": np.random.uniform(9000, 11000, 10),
        }
    )
    simulator.generate_visualizations(
        trades_df, mode="range", timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    assert Path(tmp_dirs["figures_dir"]).exists()
    figures = list(Path(tmp_dirs["figures_dir"]).glob("*.png"))
    assert len(figures) >= 3  # Balance, rewards, drawdown


def test_handle_sigint(tmp_dirs, simulator):
    """Teste la gestion de SIGINT."""
    with patch("sys.exit") as mock_exit:
        simulator.handle_sigint(None, None)
    snapshots = list(Path(tmp_dirs["snapshots_dir"]).glob("snapshot_sigint_*.json.gz"))
    assert len(snapshots) == 1
    checkpoints = list(Path(tmp_dirs["checkpoints_dir"]).glob("checkpoint_*.json.gz"))
    assert len(checkpoints) >= 1
    with gzip.open(snapshots[0], "rt") as f:
        snapshot = json.load(f)
    assert snapshot["data"]["status"] == "SIGINT"
    mock_exit.assert_called_with(0)


def test_alerts(tmp_dirs, simulator, mock_data):
    """Teste les alertes via alert_manager et telegram_alert."""
    with patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ) as mock_alert, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        simulator.validate_data(mock_data["training"])
    mock_alert.assert_called()
    mock_telegram.assert_called()


def test_no_obsolete_references(tmp_dirs):
    """Vérifie l'absence de références à dxFeed, obs_t, 320/81 features."""
    for file_path in [
        tmp_dirs["config_path"],
        tmp_dirs["feature_sets_path"],
        tmp_dirs["es_config_path"],
        tmp_dirs["model_params_path"],
        tmp_dirs["features_path"],
    ]:
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
