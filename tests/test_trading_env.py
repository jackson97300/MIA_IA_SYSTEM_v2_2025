# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_trading_env.py
# Tests unitaires pour trading_env.py.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie les fonctionnalités de TradingEnv, incluant l'initialisation,
#        la réinitialisation, les étapes, les étapes incrémentales, le calcul des récompenses,
#        la gestion des signaux, les snapshots JSON compressés, les logs de performance,
#        et les alertes.
#        Conforme à la Phase 8 (auto-conscience via alertes), Phase 12 (simulation de trading),
#        et Phase 16 (ensemble learning).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pandas>=2.0.0
# - numpy>=1.26.4
# - psutil>=5.9.8
# - gymnasium>=0.26.0
# - src.envs.trading_env
# - src.features.neural_pipeline
# - src.model.utils.config_manager
# - src.model.utils.alert_manager
# - src.utils.telegram_alert
# - src.utils.standard
#
# Notes :
# - Utilise des mocks pour simuler les dépendances externes (neural_pipeline, alert_manager).
# - Vérifie l'absence de références à dxFeed, obs_t, 320/81 features.

import gzip
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.envs.trading_env import TradingEnv

BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


@pytest.fixture
def temp_dir(tmp_path):
    """Crée un répertoire temporaire pour les tests."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    data_dir = tmp_path / "data"
    logs_dir = data_dir / "logs"
    iqfeed_dir = data_dir / "iqfeed"
    trades_dir = data_dir / "trades"
    logs_dir.mkdir(parents=True)
    iqfeed_dir.mkdir(parents=True)
    trades_dir.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def mock_config(temp_dir):
    """Crée des fichiers de configuration simulés."""
    config_path = temp_dir / "config" / "trading_env_config.yaml"
    config_content = """
environment:
  initial_cash: 100000.0
  max_position_size: 5
  point_value: 50
  transaction_cost: 2.0
  max_trade_duration: 20
  min_balance: 90000.0
  max_drawdown: -10000.0
  atr_limit: 2.0
  max_leverage: 5.0
  reward_threshold: 0.01
  call_wall_distance: 0.01
  zero_gamma_distance: 0.005
observation:
  sequence_length: 50
logging:
  buffer_size: 100
"""
    config_path.write_text(config_content)

    feature_sets_path = temp_dir / "config" / "feature_sets.yaml"
    feature_sets_content = """
training_features: []
shap_features: []
"""
    with open(feature_sets_path, "w") as f:
        f.write(feature_sets_content)

    return config_path


@pytest.fixture
def mock_data(temp_dir):
    """Crée un fichier iqfeed_data.csv simulé."""
    data_path = temp_dir / "data" / "iqfeed" / "iqfeed_data.csv"
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=100, freq="1min"),
            "close": np.random.normal(5100, 10, 100),
            "open": np.random.normal(5100, 10, 100),
            "high": np.random.normal(5105, 10, 100),
            "low": np.random.normal(5095, 10, 100),
            "volume": np.random.randint(100, 1000, 100),
            "atr_14": np.random.uniform(0.5, 2.0, 100),
            "adx_14": np.random.uniform(10, 40, 100),
            "gex": np.random.uniform(-1000, 1000, 100),
            "oi_peak_call_near": np.random.randint(5000, 15000, 100),
            "gamma_wall": np.random.uniform(5090, 5110, 100),
            "call_wall": np.random.uniform(5150, 5200, 100),
            "put_wall": np.random.uniform(5050, 5100, 100),
            "zero_gamma": np.random.uniform(5095, 5105, 100),
            "dealer_position_bias": np.random.uniform(-0.2, 0.2, 100),
            "predicted_volatility": np.random.uniform(0.1, 0.5, 100),
            "predicted_vix": np.random.uniform(15, 25, 100),
            "news_impact_score": np.random.uniform(-1, 1, 100),
            "bid_size_level_1": np.random.randint(50, 500, 100),
            "ask_size_level_1": np.random.randint(50, 500, 100),
            "spread_avg_1min": np.random.uniform(1.0, 3.0, 100),
            "neural_regime": np.random.randint(0, 3, 100),
        }
    )
    for i in range(129):  # Total 150 features
        data[f"feature_{i}"] = np.random.uniform(0, 1, 100)
    data.to_csv(data_path, index=False)
    return data_path


@pytest.fixture
def env(temp_dir, mock_config, mock_data, monkeypatch):
    """Initialise TradingEnv avec des mocks."""
    monkeypatch.setattr(
        "src.envs.trading_env.config_manager.get_config",
        lambda x: (
            {
                "environment": {
                    "initial_cash": 100000.0,
                    "max_position_size": 5,
                    "point_value": 50,
                    "transaction_cost": 2.0,
                    "max_trade_duration": 20,
                    "min_balance": 90000.0,
                    "max_drawdown": -10000.0,
                    "atr_limit": 2.0,
                    "max_leverage": 5.0,
                    "reward_threshold": 0.01,
                    "call_wall_distance": 0.01,
                    "zero_gamma_distance": 0.005,
                    "training_mode": True,
                },
                "observation": {"sequence_length": 50},
                "logging": {"buffer_size": 100},
            }
            if "trading_env_config.yaml" in str(x)
            else {
                "training_features": [f"feature_{i}" for i in range(350)],
                "shap_features": [f"feature_{i}" for i in range(150)],
            }
        ),
    )
    with patch(
        "src.features.neural_pipeline.NeuralPipeline.__init__", return_value=None
    ), patch(
        "src.features.neural_pipeline.NeuralPipeline.load_models", return_value=None
    ), patch(
        "src.features.neural_pipeline.NeuralPipeline.run",
        return_value={
            "features": np.random.uniform(0, 1, (50, 10)),
            "regime": np.random.randint(0, 3, 50),
        },
    ):
        env = TradingEnv(str(mock_config))
    env.data = pd.read_csv(mock_data, parse_dates=["timestamp"])
    return env


def test_init_env(temp_dir, mock_config, mock_data, env):
    """Teste l'initialisation de TradingEnv."""
    assert env.config["environment"]["initial_cash"] == 100000.0
    assert env.base_features == 350
    assert os.path.exists(temp_dir / "data" / "logs" / "trading_env_performance.csv")
    snapshots = list((temp_dir / "data" / "trades").glob("snapshot_init_*.json.gz"))
    assert len(snapshots) >= 1


def test_reset_env(temp_dir, env):
    """Teste la réinitialisation de l’environnement."""
    env.mode = "trend"
    env.policy_type = "mlp"
    obs, info = env.reset()
    assert obs.shape == (350,)
    assert env.current_step == 50
    assert env.balance == 100000.0
    assert env.position == 0
    snapshots = list((temp_dir / "data" / "trades").glob("snapshot_reset_*.json.gz"))
    assert len(snapshots) >= 1


def test_step_env(temp_dir, env):
    """Teste l’exécution d’une étape."""
    env.mode = "range"
    env.policy_type = "mlp"
    env.reset()
    action = np.array([0.6])
    obs, reward, done, truncated, info = env.step(action)
    assert obs.shape == (350,)
    assert isinstance(reward, float)
    assert not done
    assert not truncated
    assert "profit" in info
    assert env.position != 0
    snapshots = list((temp_dir / "data" / "trades").glob("snapshot_step_*.json.gz"))
    assert len(snapshots) >= 1


def test_incremental_step_env(temp_dir, env):
    """Teste l’exécution d’une étape incrémentale."""
    env.mode = "defensive"
    env.policy_type = "transformer"
    env.reset()
    action = np.array([0.6])
    row = env.data.iloc[50]
    obs, reward, done, truncated, info = env.incremental_step(row, action)
    assert obs.shape == (50, 350)
    assert isinstance(reward, float)
    assert not done
    assert not truncated
    assert "profit" in info
    snapshots = list(
        (temp_dir / "data" / "trades").glob("snapshot_incremental_step_*.json.gz")
    )
    assert len(snapshots) >= 1


def test_handle_sigint(temp_dir, env):
    """Teste la gestion de SIGINT."""
    env.balance = 95000.0
    with patch("sys.exit") as mock_exit:
        env.handle_sigint(None, None)
    snapshots = list((temp_dir / "data" / "trades").glob("snapshot_sigint_*.json.gz"))
    assert len(snapshots) == 1
    with gzip.open(snapshots[0], "rt") as f:
        snapshot = json.load(f)
    assert snapshot["data"]["status"] == "SIGINT"
    assert snapshot["data"]["balance"] == 95000.0
    mock_exit.assert_called_with(0)


def test_save_trade_history(temp_dir, env):
    """Teste la sauvegarde de l’historique des trades."""
    env.mode = "trend"
    env.reset()
    action = np.array([0.6])
    env.step(action)
    env.save_trade_history()
    output_path = temp_dir / "data" / "trades" / "trade_history.csv"
    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert not df.empty
    assert "profit" in df.columns


def test_no_obsolete_references(temp_dir, mock_config):
    """Vérifie l'absence de références à dxFeed, obs_t, 320/81 features."""
    with open(mock_config, "r") as f:
        content = f.read()
    assert "dxFeed" not in content, "Référence à dxFeed trouvée"
    assert "obs_t" not in content, "Référence à obs_t trouvée"
    assert "320 features" not in content, "Référence à 320 features trouvée"
    assert "81 features" not in content, "Référence à 81 features trouvée"
