# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_analyse_trades.py
# Tests unitaires pour src/trading/analyse_trades.py
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Valide la classe TradeDetailAnalyzer pour l’analyse des trades individuels,
#        incluant la validation des données (350/150 SHAP features), le calcul des métriques,
#        la mémoire contextuelle via clusters (Phase 7), les snapshots compressés, les sauvegardes,
#        les graphiques matplotlib, les logs psutil, et les alertes (Phase 8).
#        Compatible avec la simulation de trading (Phase 12) et l’ensemble learning (Phase 16).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pandas>=2.0.0,<3.0.0
# - numpy>=1.26.4,<2.0.0
# - psutil>=5.9.8,<6.0.0
# - pyyaml>=6.0.0,<7.0.0
# - matplotlib>=3.7.0,<4.0.0
# - src.trading.analyse_trades
# - src.mind.mind
# - src.model.adaptive_learner
# - src.model.utils.config_manager
# - src.model.utils.alert_manager
# - src.utils.telegram_alert
#
# Inputs :
# - config/market_config.yaml (simulé)
# - config/feature_sets.yaml (simulé)
# - Données de trading simulées
#
# Outputs :
# - Assertions sur l’état de l’analyseur
# - data/logs/analyse_trades.csv (simulé)
# - data/trade_snapshots/*.json.gz (simulé)
# - data/trades/*.csv (simulé)
# - data/checkpoints/analyse_trades/*.json.gz (simulé)
# - data/figures/trading/*.png (simulé)
#
# Notes :
# - Utilise des mocks pour simuler MindEngine, SQLite, et store_pattern.
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

from src.trading.analyse_trades import TradeDetailAnalyzer

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
    checkpoints_dir = data_dir / "checkpoints" / "analyse_trades"
    checkpoints_dir.mkdir(parents=True)
    figures_dir = data_dir / "figures" / "trading"
    figures_dir.mkdir(parents=True)

    # Créer market_config.yaml
    config_path = config_dir / "market_config.yaml"
    config_content = {
        "thresholds": {
            "min_profit_factor": 1.2,
            "max_slippage": 0.01,
            "min_trade_duration": 60,
        },
        "logging": {"buffer_size": 100},
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
            "reward",
            "step",
            "regime",
        ]
        + [f"feat_{i}" for i in range(339)],
        "shap_features": [
            "vix",
            "neural_regime",
            "predicted_volatility",
            "trade_frequency_1s",
            "close",
            "rsi_14",
            "gex",
            "reward",
            "step",
            "regime",
        ]
        + [f"feat_{i}" for i in range(140)],
    }
    with open(feature_sets_path, "w", encoding="utf-8") as f:
        yaml.dump(feature_sets_content, f)

    # Créer trades_simulated.csv
    trades_path = trades_dir / "trades_simulated.csv"
    trades_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=10, freq="1min"),
            "vix": np.random.uniform(15, 25, 10),
            "neural_regime": np.random.randint(0, 3, 10),
            "predicted_volatility": np.random.uniform(0.1, 0.5, 10),
            "trade_frequency_1s": np.random.uniform(5, 10, 10),
            "close": np.random.normal(5100, 10, 10),
            "rsi_14": np.random.uniform(20, 80, 10),
            "gex": np.random.uniform(-1000, 1000, 10),
            "reward": np.random.uniform(-100, 100, 10),
            "step": range(10),
            "regime": np.random.choice(["trend", "range", "defensive"], 10),
            "entry_price": np.random.normal(5100, 10, 10),
            "execution_price": np.random.normal(5100, 10, 10),
            "entry_time": pd.date_range("2025-05-13 09:00", periods=10, freq="1min"),
            "exit_time": pd.date_range("2025-05-13 09:01", periods=10, freq="1min"),
            **{
                f"feat_{i}": np.random.uniform(0, 1, 10) for i in range(339)
            },  # Total 350 features
        }
    )
    trades_data.to_csv(trades_path, index=False, encoding="utf-8")

    return {
        "config_path": str(config_path),
        "feature_sets_path": str(feature_sets_path),
        "trades_path": str(trades_path),
        "logs_dir": str(logs_dir),
        "snapshots_dir": str(snapshots_dir),
        "trades_dir": str(trades_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "figures_dir": str(figures_dir),
        "perf_log_path": str(logs_dir / "analyse_trades.csv"),
        "dashboard_path": str(data_dir / "trade_summary_ES.csv"),
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
            "reward",
            "step",
            "regime",
            "entry_price",
            "execution_price",
            "entry_time",
            "exit_time",
        ]
        + [f"feat_{i}" for i in range(335)],
        "shap_features": [
            "vix",
            "neural_regime",
            "predicted_volatility",
            "trade_frequency_1s",
            "close",
            "rsi_14",
            "gex",
            "reward",
            "step",
            "regime",
            "entry_price",
            "execution_price",
            "entry_time",
            "exit_time",
        ]
        + [f"feat_{i}" for i in range(136)],
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
            "reward": [50.0],
            "step": [1],
            "regime": ["trend"],
            "entry_price": [5100.0],
            "execution_price": [5102.0],
            "entry_time": [datetime.now()],
            "exit_time": [datetime.now() + timedelta(minutes=1)],
            **{
                f"feat_{i}": [np.random.uniform(0, 1)] for i in range(335)
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
            "reward": [50.0],
            "step": [1],
            "regime": ["trend"],
            "entry_price": [5100.0],
            "execution_price": [5102.0],
            "entry_time": [datetime.now()],
            "exit_time": [datetime.now() + timedelta(minutes=1)],
            **{
                f"feat_{i}": [np.random.uniform(0, 1)] for i in range(136)
            },  # Total 150 features
        }
    )
    return {"training": training_data, "inference": inference_data}


@pytest.fixture
def analyzer(tmp_dirs, monkeypatch):
    """Initialise TradeDetailAnalyzer avec des mocks."""
    monkeypatch.setattr(
        "src.trading.analyse_trades.config_manager.get_config",
        lambda x: (
            {
                "thresholds": {
                    "min_profit_factor": 1.2,
                    "max_slippage": 0.01,
                    "min_trade_duration": 60,
                },
                "logging": {"buffer_size": 100},
            }
            if "market_config.yaml" in str(x)
            else {
                "training_features": [
                    "timestamp",
                    "vix",
                    "neural_regime",
                    "predicted_volatility",
                    "trade_frequency_1s",
                    "close",
                    "rsi_14",
                    "gex",
                    "reward",
                    "step",
                    "regime",
                ]
                + [f"feat_{i}" for i in range(339)],
                "shap_features": [
                    "vix",
                    "neural_regime",
                    "predicted_volatility",
                    "trade_frequency_1s",
                    "close",
                    "rsi_14",
                    "gex",
                    "reward",
                    "step",
                    "regime",
                ]
                + [f"feat_{i}" for i in range(140)],
            }
        ),
    )
    with patch("src.mind.mind.MindEngine.__init__", return_value=None), patch(
        "src.model.adaptive_learner.store_pattern", return_value=None
    ), patch("sqlite3.connect", return_value=MagicMock()), patch(
        "src.model.utils.alert_manager.AlertManager.send_alert", return_value=None
    ), patch(
        "src.utils.telegram_alert.send_telegram_alert", return_value=None
    ):
        analyzer = TradeDetailAnalyzer(config_path=tmp_dirs["config_path"])
    return analyzer


def test_trade_detail_analyzer_init(tmp_dirs, analyzer):
    """Teste l’initialisation de TradeDetailAnalyzer."""
    assert analyzer.performance_thresholds["min_profit_factor"] == 1.2
    assert Path(tmp_dirs["perf_log_path"]).exists()
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert "cpu_percent" in df.columns
    assert "memory_usage_mb" in df.columns
    assert Path(tmp_dirs["checkpoints_dir"]).exists()
    checkpoints = list(Path(tmp_dirs["checkpoints_dir"]).glob("checkpoint_*.json.gz"))
    assert len(checkpoints) >= 1


def test_validate_data_training(tmp_dirs, analyzer, mock_data):
    """Teste la validation des données en mode entraînement (350 features)."""
    analyzer.validate_data(mock_data["training"])
    assert Path(tmp_dirs["snapshots_dir"]).exists()
    snapshots = list(
        Path(tmp_dirs["snapshots_dir"]).glob("snapshot_validation_*.json.gz")
    )
    assert len(snapshots) >= 1
    assert Path(tmp_dirs["perf_log_path"]).exists()


def test_validate_data_inference(tmp_dirs, analyzer, mock_data):
    """Teste la validation des données en mode inférence (150 SHAP features)."""
    analyzer.validate_data(mock_data["inference"])
    assert Path(tmp_dirs["snapshots_dir"]).exists()
    snapshots = list(
        Path(tmp_dirs["snapshots_dir"]).glob("snapshot_validation_*.json.gz")
    )
    assert len(snapshots) >= 1
    assert Path(tmp_dirs["perf_log_path"]).exists()


def test_validate_data_invalid_columns(tmp_dirs, analyzer, mock_data):
    """Teste la validation avec des colonnes manquantes."""
    invalid_data = mock_data["training"].drop(columns=["vix"])
    with pytest.raises(ValueError, match="Nombre de features insuffisant"):
        analyzer.validate_data(invalid_data)
    snapshots = list(
        Path(tmp_dirs["snapshots_dir"]).glob("snapshot_validation_*.json.gz")
    )
    assert len(snapshots) >= 1


def test_load_trades(tmp_dirs, analyzer):
    """Teste le chargement des trades depuis un fichier CSV."""
    df = analyzer.load_trades(tmp_dirs["trades_path"])
    assert len(df) == 10
    assert Path(tmp_dirs["snapshots_dir"]).exists()
    snapshots = list(
        Path(tmp_dirs["snapshots_dir"]).glob("snapshot_validation_*.json.gz")
    )
    assert len(snapshots) >= 1
    assert Path(tmp_dirs["perf_log_path"]).exists()


def test_compute_metrics(tmp_dirs, analyzer, tmp_dirs):
    """Teste le calcul des métriques de performance."""
    df = pd.read_csv(tmp_dirs["trades_path"])
    stats = analyzer.compute_metrics(df)
    assert "total_trades" in stats
    assert "by_regime" in stats
    assert Path(tmp_dirs["perf_log_path"]).exists()


def test_analyse_trade(tmp_dirs, analyzer, mock_data):
    """Teste l’analyse d’un trade individuel."""
    trade_metrics = analyzer.analyse_trade(1, mock_data["training"])
    assert "trade_id" in trade_metrics
    assert Path(tmp_dirs["snapshots_dir"]).exists()
    snapshots = list(Path(tmp_dirs["snapshots_dir"]).glob("snapshot_trade_1_*.json.gz"))
    assert len(snapshots) >= 1
    assert Path(tmp_dirs["perf_log_path"]).exists()


def test_save_summary(tmp_dirs, analyzer, tmp_dirs):
    """Teste la sauvegarde du résumé des métriques."""
    df = pd.read_csv(tmp_dirs["trades_path"])
    stats = analyzer.compute_metrics(df)
    analyzer.last_checkpoint_time = datetime.now() - timedelta(
        seconds=400
    )  # Permettre la sauvegarde
    analyzer.save_summary(stats, "ES", tmp_dirs["trades_dir"])
    assert Path(tmp_dirs["trades_dir"]).exists()
    summaries = list(Path(tmp_dirs["trades_dir"]).glob("trade_summary_ES_*.csv"))
    assert len(summaries) >= 1
    assert Path(tmp_dirs["snapshots_dir"]).exists()
    snapshots = list(
        Path(tmp_dirs["snapshots_dir"]).glob("snapshot_summary_ES_*.json.gz")
    )
    assert len(snapshots) >= 1
    assert Path(tmp_dirs["checkpoints_dir"]).exists()
    checkpoints = list(Path(tmp_dirs["checkpoints_dir"]).glob("checkpoint_*.json.gz"))
    assert len(checkpoints) >= 1
    assert Path(tmp_dirs["figures_dir"]).exists()
    figures = list(Path(tmp_dirs["figures_dir"]).glob("performance_ES_*.png"))
    assert len(figures) >= 1


def test_handle_sigint(tmp_dirs, analyzer):
    """Teste la gestion de SIGINT."""
    with patch("sys.exit") as mock_exit:
        analyzer.handle_sigint(None, None)
    snapshots = list(Path(tmp_dirs["snapshots_dir"]).glob("snapshot_sigint_*.json.gz"))
    assert len(snapshots) == 1
    checkpoints = list(Path(tmp_dirs["checkpoints_dir"]).glob("checkpoint_*.json.gz"))
    assert len(checkpoints) >= 1
    with gzip.open(snapshots[0], "rt") as f:
        snapshot = json.load(f)
    assert snapshot["data"]["status"] == "SIGINT"
    mock_exit.assert_called_with(0)


def test_alerts(tmp_dirs, analyzer, mock_data):
    """Teste les alertes via alert_manager et telegram_alert."""
    with patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ) as mock_alert, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        analyzer.validate_data(mock_data["training"])
    mock_alert.assert_called()
    mock_telegram.assert_called()


def test_no_obsolete_references(tmp_dirs):
    """Vérifie l'absence de références à dxFeed, obs_t, 320/81 features."""
    for file_path in [
        tmp_dirs["config_path"],
        tmp_dirs["feature_sets_path"],
        tmp_dirs["trades_path"],
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
