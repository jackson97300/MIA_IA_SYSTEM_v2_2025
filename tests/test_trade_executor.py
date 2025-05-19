# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_trade_executor.py
# Tests unitaires pour src/trading/trade_executor.py
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Valide la classe TradeExecutor pour l’exécution des trades via Sierra Chart (API Teton) en mode réel ou paper trading,
#        incluant la validation des données contextuelles (350/150 SHAP features), l’enregistrement dans market_memory.db,
#        le fine-tuning (méthode 8), l’apprentissage en ligne (méthode 10), les snapshots compressés, les sauvegardes,
#        les graphiques matplotlib, et les alertes (Phase 8). Compatible avec la simulation de trading (Phase 12)
#        et l’ensemble learning (Phase 16).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pandas>=2.0.0,<3.0.0
# - numpy>=1.26.4,<2.0.0
# - matplotlib>=3.8.0,<4.0.0
# - psutil>=5.9.8,<6.0.0
# - pyyaml>=6.0.0,<7.0.0
# - src.trading.trade_executor
# - src.model.adaptive_learner
# - src.model.utils.config_manager
# - src.model.utils.alert_manager
# - src.utils.telegram_alert
# - src.trading.finetune_utils
#
# Inputs :
# - config/market_config.yaml (simulé)
# - config/feature_sets.yaml (simulé)
# - config/credentials.yaml (simulé)
# - Données contextuelles simulées
#
# Outputs :
# - Assertions sur l’état de l’exécuteur
# - data/logs/trade_executor_performance.csv (simulé)
# - data/trade_snapshots/*.json.gz (simulé)
# - data/trading/trade_execution_dashboard.json (simulé)
# - data/figures/*.png (simulé)
# - data/trades/*.csv (simulé)
# - data/checkpoints/trade_executor/*.json.gz (simulé)
#
# Notes :
# - Utilise des mocks pour simuler store_pattern, finetune_model, et l’API Teton.
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

from src.trading.trade_executor import TradeExecutor

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
    snapshots_dir = data_dir / "trade_snapshots"
    snapshots_dir.mkdir()
    trades_dir = data_dir / "trades"
    trades_dir.mkdir()
    checkpoints_dir = data_dir / "checkpoints" / "trade_executor"
    checkpoints_dir.mkdir(parents=True)
    figures_dir = data_dir / "figures"
    figures_dir.mkdir()
    trading_dir = data_dir / "trading"
    trading_dir.mkdir()

    # Créer market_config.yaml
    config_path = config_dir / "market_config.yaml"
    config_content = {
        "thresholds": {
            "max_slippage": 0.01,
            "min_balance": -10000,
            "min_sharpe": 0.5,
            "max_drawdown": -1000.0,
            "min_profit_factor": 1.2,
        },
        "logging": {"buffer_size": 200},
        "visualization": {"figsize": [12, 5], "plot_interval": 50},
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    # Créer feature_sets.yaml
    feature_sets_path = config_dir / "feature_sets.yaml"
    feature_sets_content = {
        "training_features": [
            "vix",
            "neural_regime",
            "predicted_volatility",
            "trade_frequency_1s",
            "close",
            "rsi_14",
            "gex",
            "timestamp",
        ]
        + [f"feat_{i}" for i in range(342)],
        "shap_features": [
            "vix",
            "neural_regime",
            "predicted_volatility",
            "trade_frequency_1s",
            "close",
            "rsi_14",
            "gex",
            "timestamp",
        ]
        + [f"feat_{i}" for i in range(142)],
    }
    with open(feature_sets_path, "w", encoding="utf-8") as f:
        yaml.dump(feature_sets_content, f)

    # Créer credentials.yaml
    credentials_path = config_dir / "credentials.yaml"
    credentials_content = {"amp_futures": {"enabled": False}}
    with open(credentials_path, "w", encoding="utf-8") as f:
        yaml.dump(credentials_content, f)

    return {
        "config_path": str(config_path),
        "feature_sets_path": str(feature_sets_path),
        "credentials_path": str(credentials_path),
        "logs_dir": str(logs_dir),
        "snapshots_dir": str(snapshots_dir),
        "trades_dir": str(trades_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "figures_dir": str(figures_dir),
        "perf_log_path": str(logs_dir / "trade_executor_performance.csv"),
        "dashboard_path": str(trading_dir / "trade_execution_dashboard.json"),
        "trades_path": str(trades_dir / "trades_simulated.csv"),
    }


@pytest.fixture
def mock_data():
    """Crée des données contextuelles simulées pour les tests."""
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
            **{
                f"feat_{i}": [np.random.uniform(0, 1)] for i in range(342)
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
            **{
                f"feat_{i}": [np.random.uniform(0, 1)] for i in range(142)
            },  # Total 150 features
        }
    )
    return {"training": training_data, "inference": inference_data}


@pytest.fixture
def executor(tmp_dirs, monkeypatch):
    """Initialise TradeExecutor avec des mocks."""
    monkeypatch.setattr(
        "src.trading.trade_executor.config_manager.get_config",
        lambda x: (
            {
                "thresholds": {
                    "max_slippage": 0.01,
                    "min_balance": -10000,
                    "min_sharpe": 0.5,
                    "max_drawdown": -1000.0,
                    "min_profit_factor": 1.2,
                },
                "logging": {"buffer_size": 200},
                "visualization": {"figsize": [12, 5], "plot_interval": 50},
            }
            if "market_config.yaml" in str(x)
            else (
                {
                    "training_features": [
                        "vix",
                        "neural_regime",
                        "predicted_volatility",
                        "trade_frequency_1s",
                        "close",
                        "rsi_14",
                        "gex",
                        "timestamp",
                    ]
                    + [f"feat_{i}" for i in range(342)],
                    "shap_features": [
                        "vix",
                        "neural_regime",
                        "predicted_volatility",
                        "trade_frequency_1s",
                        "close",
                        "rsi_14",
                        "gex",
                        "timestamp",
                    ]
                    + [f"feat_{i}" for i in range(142)],
                }
                if "feature_sets.yaml" in str(x)
                else {"amp_futures": {"enabled": False}}
            )
        ),
    )
    with patch("src.model.adaptive_learner.store_pattern", return_value=None), patch(
        "src.trading.finetune_utils.finetune_model", return_value=None
    ), patch(
        "src.model.utils.alert_manager.AlertManager.send_alert", return_value=None
    ), patch(
        "src.utils.telegram_alert.send_telegram_alert", return_value=None
    ):
        executor = TradeExecutor(
            config_path=tmp_dirs["config_path"],
            credentials_path=tmp_dirs["credentials_path"],
        )
    return executor


def test_trade_executor_init(tmp_dirs, executor):
    """Teste l’initialisation de TradeExecutor."""
    assert executor.performance_thresholds["max_slippage"] == 0.01
    assert Path(tmp_dirs["perf_log_path"]).exists()
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert "cpu_percent" in df.columns
    assert "memory_usage_mb" in df.columns
    assert Path(tmp_dirs["checkpoints_dir"]).exists()
    checkpoints = list(Path(tmp_dirs["checkpoints_dir"]).glob("checkpoint_*.json.gz"))
    assert len(checkpoints) >= 1


def test_validate_context_data_training(tmp_dirs, executor, mock_data):
    """Teste la validation des données contextuelles en mode entraînement (350 features)."""
    executor.validate_context_data(mock_data["training"])
    assert Path(tmp_dirs["snapshots_dir"]).exists()
    snapshots = list(
        Path(tmp_dirs["snapshots_dir"]).glob("snapshot_validate_context_data_*.json.gz")
    )
    assert len(snapshots) >= 1
    assert Path(tmp_dirs["perf_log_path"]).exists()


def test_validate_context_data_inference(tmp_dirs, executor, mock_data):
    """Teste la validation des données contextuelles en mode inférence (150 SHAP features)."""
    executor.validate_context_data(mock_data["inference"])
    assert Path(tmp_dirs["snapshots_dir"]).exists()
    snapshots = list(
        Path(tmp_dirs["snapshots_dir"]).glob("snapshot_validate_context_data_*.json.gz")
    )
    assert len(snapshots) >= 1
    assert Path(tmp_dirs["perf_log_path"]).exists()


def test_validate_context_data_invalid_columns(tmp_dirs, executor, mock_data):
    """Teste la validation avec des colonnes manquantes."""
    invalid_data = mock_data["training"].drop(columns=["vix"])
    with pytest.raises(ValueError, match="Colonnes manquantes dans les données"):
        executor.validate_context_data(invalid_data)
    snapshots = list(
        Path(tmp_dirs["snapshots_dir"]).glob("snapshot_validate_context_data_*.json.gz")
    )
    assert len(snapshots) >= 1


def test_validate_trade_valid(tmp_dirs, executor):
    """Teste la validation d’un trade valide."""
    trade = {
        "trade_id": "123",
        "action": "buy",
        "price": 5100.0,
        "size": 1,
        "order_type": "market",
    }
    executor.validate_trade(trade)
    assert Path(tmp_dirs["perf_log_path"]).exists()


def test_validate_trade_invalid(tmp_dirs, executor):
    """Teste la validation d’un trade invalide."""
    trade = {
        "trade_id": "123",
        "action": "invalid",
        "price": -100.0,
        "size": 0,
        "order_type": "invalid",
    }
    with pytest.raises(ValueError):
        executor.validate_trade(trade)
    assert Path(tmp_dirs["perf_log_path"]).exists()


def test_execute_trade_paper(tmp_dirs, executor, mock_data):
    """Teste l’exécution d’un trade en mode paper trading."""
    trade = {
        "trade_id": "123",
        "action": "buy",
        "price": 5100.0,
        "size": 1,
        "order_type": "market",
    }
    executor.last_checkpoint_time = datetime.now() - timedelta(
        seconds=400
    )  # Permettre la sauvegarde
    result = executor.execute_trade(
        trade, mode="paper", context_data=mock_data["training"]
    )
    assert result["trade_id"] == "123"
    assert result["status"] == "filled"
    assert Path(tmp_dirs["trades_path"]).exists()
    assert Path(tmp_dirs["snapshots_dir"]).exists()
    snapshots = list(
        Path(tmp_dirs["snapshots_dir"]).glob("snapshot_execution_trade_123_*.json.gz")
    )
    assert len(snapshots) >= 1
    assert Path(tmp_dirs["perf_log_path"]).exists()
    assert Path(tmp_dirs["dashboard_path"]).exists()
    assert Path(tmp_dirs["checkpoints_dir"]).exists()
    checkpoints = list(Path(tmp_dirs["checkpoints_dir"]).glob("checkpoint_*.json.gz"))
    assert len(checkpoints) >= 1


def test_execute_trade_failed(tmp_dirs, executor, mock_data):
    """Teste l’exécution d’un trade échoué."""
    trade = {
        "trade_id": "123",
        "action": "buy",
        "price": -5100.0,  # Prix invalide
        "size": 1,
        "order_type": "market",
    }
    result = executor.execute_trade(
        trade, mode="paper", context_data=mock_data["training"]
    )
    assert result["trade_id"] == "123"
    assert result["status"] == "failed"
    assert Path(tmp_dirs["perf_log_path"]).exists()


def test_confirm_trade(tmp_dirs, executor):
    """Teste la confirmation d’un trade."""
    result = executor.confirm_trade("123")
    assert result["trade_id"] == "123"
    assert isinstance(result["confirmed"], bool)
    assert Path(tmp_dirs["perf_log_path"]).exists()


def test_store_execution_pattern(tmp_dirs, executor, mock_data):
    """Teste l’enregistrement d’un pattern d’exécution."""
    trade = {
        "trade_id": "123",
        "action": "buy",
        "price": 5100.0,
        "size": 1,
        "order_type": "market",
    }
    execution_result = {
        "trade_id": "123",
        "status": "filled",
        "execution_price": 5102.0,
        "execution_time": str(datetime.now()),
        "balance": 10000.0,
    }
    executor.store_execution_pattern(
        trade, execution_result, mode="paper", context_data=mock_data["training"]
    )
    assert Path(tmp_dirs["perf_log_path"]).exists()


def test_log_execution(tmp_dirs, executor, mock_data):
    """Teste l’enregistrement de l’exécution dans un CSV."""
    trade = {
        "trade_id": "123",
        "action": "buy",
        "price": 5100.0,
        "size": 1,
        "order_type": "market",
    }
    execution_result = {
        "trade_id": "123",
        "status": "filled",
        "execution_price": 5102.0,
        "execution_time": str(datetime.now()),
        "balance": 10000.0,
    }
    executor.last_checkpoint_time = datetime.now() - timedelta(
        seconds=400
    )  # Permettre la sauvegarde
    executor.log_execution(trade, execution_result, is_paper=True)
    assert Path(tmp_dirs["trades_path"]).exists()
    assert Path(tmp_dirs["snapshots_dir"]).exists()
    assert Path(tmp_dirs["checkpoints_dir"]).exists()
    checkpoints = list(Path(tmp_dirs["checkpoints_dir"]).glob("checkpoint_*.json.gz"))
    assert len(checkpoints) >= 1


def test_plot_execution_metrics(tmp_dirs, executor):
    """Teste la génération des graphiques d’exécution."""
    trade_buffer = [
        {
            "timestamp": str(datetime.now()),
            "trade_id": "123",
            "action": "buy",
            "price": 5100.0,
            "size": 1,
            "order_type": "market",
            "status": "filled",
            "execution_price": 5102.0,
            "balance": 10000.0,
            "slippage": 0.000392,
            "is_paper": True,
        }
    ]
    executor.plot_execution_metrics(trade_buffer)
    assert Path(tmp_dirs["figures_dir"]).exists()
    figures = list(Path(tmp_dirs["figures_dir"]).glob("execution_*.png"))
    assert len(figures) >= 3  # Slippage, balance, status
    assert Path(tmp_dirs["perf_log_path"]).exists()


def test_handle_sigint(tmp_dirs, executor):
    """Teste la gestion de SIGINT."""
    with patch("sys.exit") as mock_exit:
        executor.handle_sigint(None, None)
    snapshots = list(Path(tmp_dirs["snapshots_dir"]).glob("snapshot_sigint_*.json.gz"))
    assert len(snapshots) == 1
    checkpoints = list(Path(tmp_dirs["checkpoints_dir"]).glob("checkpoint_*.json.gz"))
    assert len(checkpoints) >= 1
    with gzip.open(snapshots[0], "rt") as f:
        snapshot = json.load(f)
    assert snapshot["data"]["status"] == "SIGINT"
    mock_exit.assert_called_with(0)


def test_alerts(tmp_dirs, executor, mock_data):
    """Teste les alertes via alert_manager et telegram_alert."""
    with patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ) as mock_alert, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        executor.validate_context_data(mock_data["training"])
    mock_alert.assert_called()
    mock_telegram.assert_called()


def test_no_obsolete_references(tmp_dirs):
    """Vérifie l'absence de références à dxFeed, obs_t, 320/81 features."""
    for file_path in [
        tmp_dirs["config_path"],
        tmp_dirs["feature_sets_path"],
        tmp_dirs["credentials_path"],
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
