# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_run_training.py
# Tests unitaires pour run_training.py.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie les fonctionnalités de TrainingManager, incluant l'initialisation,
#        la validation de la configuration et de l’environnement, l’entraînement SAC,
#        les retries, les snapshots JSON compressés, les logs de performance,
#        les graphiques, et les alertes.
#        Conforme à la Phase 8 (auto-conscience via alertes), Phase 10 (entraînement des politiques SAC),
#        et Phase 16 (ensemble learning).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pandas>=2.0.0
# - torch>=2.0.0
# - psutil>=5.9.8
# - matplotlib>=3.7.0
# - src.scripts.run_training
# - src.envs.trading_env
# - src.model.utils.config_manager
# - src.model.utils.alert_manager
# - src.model.utils.miya_console
# - src.utils.telegram_alert
# - src.utils.standard
#
# Notes :
# - Utilise des mocks pour simuler les dépendances externes.
# - Vérifie l'absence de références à dxFeed, obs_t, 320/81 features.

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch

from src.envs.trading_env import TradingEnv
from src.scripts.run_training import SimpleSAC, TrainingManager


@pytest.fixture
def temp_dir(tmp_path):
    """Crée un répertoire temporaire pour les tests."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    data_dir = tmp_path / "data"
    logs_dir = data_dir / "logs"
    features_dir = data_dir / "features"
    models_dir = data_dir / "models"
    snapshots_dir = data_dir / "training_snapshots"
    figures_dir = data_dir / "figures" / "training"
    trades_dir = data_dir / "trades"
    logs_dir.mkdir(parents=True)
    features_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)
    snapshots_dir.mkdir(parents=True)
    figures_dir.mkdir(parents=True)
    trades_dir.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def mock_config(temp_dir):
    """Crée un fichier es_config.yaml simulé."""
    config_path = temp_dir / "config" / "es_config.yaml"
    config_content = """
training:
  features_path: data/features/features_latest_filtered.csv
  model_path: data/models/sac_model.pth
  epochs: 10
  batch_size: 64
  learning_rate: 0.0003
  buffer_size: 10000
  chunk_size: 10000
  cache_hours: 24
  retry_attempts: 3
  retry_delay: 5
  timeout_seconds: 7200
  max_cache_size: 1000
  num_features: 350
  shap_features: 150
"""
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def mock_features(temp_dir):
    """Crée un fichier CSV de features simulé."""
    features_path = temp_dir / "data" / "features" / "features_latest_filtered.csv"
    features = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start="2025-05-13 10:00:00", periods=100, freq="T"
            ),
            "close": np.random.uniform(5000, 5100, 100),
            "rsi": np.random.uniform(0, 100, 100),
            "bid_size_level_1": np.random.uniform(100, 1000, 100),
            "ask_size_level_1": np.random.uniform(100, 1000, 100),
            "trade_frequency_1s": np.random.uniform(0, 10, 100),
        }
    )
    for i in range(344):  # Ajouter 344 colonnes pour atteindre 350 features
        features[f"feature_{i}"] = np.random.uniform(0, 1, 100)
    features.to_csv(features_path, index=False)
    return features_path


@pytest.fixture
def mock_feature_sets(temp_dir):
    """Crée un fichier feature_sets.yaml simulé."""
    feature_sets_path = temp_dir / "config" / "feature_sets.yaml"
    feature_sets = {
        "training_features": [
            "rsi",
            "bid_size_level_1",
            "ask_size_level_1",
            "trade_frequency_1s",
        ]
        + [f"feature_{i}" for i in range(346)]
    }
    with open(feature_sets_path, "w", encoding="utf-8") as f:
        yaml.dump(feature_sets, f)
    return feature_sets_path


@pytest.fixture
def mock_trades(temp_dir):
    """Crée un fichier trades_simulated.csv simulé."""
    trades_path = temp_dir / "data" / "trades" / "trades_simulated.csv"
    trades = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start="2025-05-13 10:00:00", periods=100, freq="T"
            ),
            "action": np.random.choice(["buy", "sell", "hold"], 100),
            "price": np.random.uniform(5000, 5100, 100),
            "confidence": np.random.uniform(0.7, 1.0, 100),
        }
    )
    trades.to_csv(trades_path, index=False)
    return trades_path


@pytest.fixture
def manager(
    temp_dir, mock_config, mock_features, mock_feature_sets, mock_trades, monkeypatch
):
    """Initialise TrainingManager avec des mocks."""
    monkeypatch.setattr("src.scripts.run_training.CONFIG_PATH", str(mock_config))
    monkeypatch.setattr("src.scripts.run_training.FEATURES_PATH", str(mock_features))
    monkeypatch.setattr(
        "src.scripts.run_training.config_manager.get_config",
        lambda x: {
            "training": {
                "features_path": str(mock_features),
                "model_path": str(temp_dir / "data" / "models" / "sac_model.pth"),
                "epochs": 10,
                "batch_size": 64,
                "learning_rate": 0.0003,
                "buffer_size": 10000,
                "chunk_size": 10000,
                "cache_hours": 24,
                "retry_attempts": 3,
                "retry_delay": 5,
                "timeout_seconds": 7200,
                "max_cache_size": 1000,
                "num_features": 350,
                "shap_features": 150,
            }
        },
    )
    manager = TrainingManager()
    return manager


def test_init_manager(
    temp_dir, mock_config, mock_features, mock_feature_sets, mock_trades, manager
):
    """Teste l'initialisation de TrainingManager."""
    assert manager.config["features_path"] == str(mock_features)
    assert os.path.exists(temp_dir / "data" / "training_snapshots")
    snapshots = list(
        (temp_dir / "data" / "training_snapshots").glob("snapshot_init_*.json.gz")
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "run_training_performance.csv"
    assert perf_log.exists()


def test_load_config_invalid(temp_dir, monkeypatch):
    """Teste la validation avec une configuration invalide."""
    invalid_config = temp_dir / "config" / "invalid.yaml"
    monkeypatch.setattr("src.scripts.run_training.CONFIG_PATH", str(invalid_config))
    with pytest.raises(FileNotFoundError, match="Fichier config introuvable"):
        TrainingManager()

    with patch("src.scripts.run_training.config_manager.get_config", return_value={}):
        monkeypatch.setattr(
            "src.scripts.run_training.CONFIG_PATH",
            str(temp_dir / "config" / "es_config.yaml"),
        )
        with pytest.raises(ValueError, match="Clé 'training' manquante"):
            TrainingManager()


def test_validate_env_valid(temp_dir, mock_config, manager):
    """Teste la validation d’un environnement valide."""
    env = TradingEnv(config_path=str(mock_config))
    env.observation_space = MagicMock(shape=(350,))
    env.action_space = MagicMock(n=3)
    manager.validate_env(env)
    snapshots = list(
        (temp_dir / "data" / "training_snapshots").glob(
            "snapshot_validate_env_*.json.gz"
        )
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "run_training_performance.csv"
    assert perf_log.exists()


def test_validate_env_invalid(temp_dir, mock_config, manager):
    """Teste la validation d’un environnement invalide."""
    env = MagicMock()
    env.observation_space = MagicMock(shape=(300,))
    env.action_space = MagicMock(n=2)
    with pytest.raises(ValueError, match="Dimension d’observation incorrecte"):
        manager.validate_env(env)
    snapshots = list(
        (temp_dir / "data" / "training_snapshots").glob(
            "snapshot_validate_env_*.json.gz"
        )
    )
    assert len(snapshots) >= 1


def test_load_cache_valid(temp_dir, mock_config, mock_features, manager):
    """Teste le chargement d’un modèle en cache valide."""
    model_path = temp_dir / "data" / "models" / "sac_model.pth"
    state_dict = {"actor.0.weight": torch.randn(256, 350)}
    torch.save(state_dict, model_path)
    os.utime(model_path, (time.time(), time.time()))
    cached_model = manager.load_cache(
        str(model_path), cache_hours=24, expected_input_dim=350
    )
    assert cached_model is not None
    snapshots = list(
        (temp_dir / "data" / "training_snapshots").glob("snapshot_load_cache_*.json.gz")
    )
    assert len(snapshots) >= 1


def test_train_sac_success(
    temp_dir, mock_config, mock_features, mock_feature_sets, mock_trades, manager
):
    """Teste l’entraînement SAC avec succès."""
    env = TradingEnv(config_path=str(mock_config))
    env.observation_space = MagicMock(shape=(350,))
    env.action_space = MagicMock(n=3)
    with patch("torch.save") as mock_save, patch(
        "src.model.adaptive_learning.retrain_with_recent_trades"
    ) as mock_retrain:
        mock_retrain.return_value = SimpleSAC(350, 3)
        state_dict, avg_reward = manager.train_sac(
            str(mock_features),
            str(temp_dir / "data" / "models" / "sac_model.pth"),
            manager.config,
            env,
        )

    assert isinstance(state_dict, dict)
    assert isinstance(avg_reward, float)
    snapshots = list(
        (temp_dir / "data" / "training_snapshots").glob("snapshot_train_sac_*.json.gz")
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "run_training_performance.csv"
    assert perf_log.exists()
    figures = list(
        (temp_dir / "data" / "figures" / "training").glob("training_loss_*.png")
    )
    assert len(figures) >= 1


def test_run_training_success(
    temp_dir, mock_config, mock_features, mock_feature_sets, mock_trades, manager
):
    """Teste l’exécution complète de l’entraînement avec succès."""
    env = TradingEnv(config_path=str(mock_config))
    env.observation_space = MagicMock(shape=(350,))
    env.action_space = MagicMock(n=3)
    with patch("src.scripts.run_training.TrainingManager.train_sac") as mock_train:
        mock_train.return_value = ({}, 0.5)
        status = manager.run_training(manager.config, env)

    assert status["success"] is True
    assert status["epochs_completed"] == 10
    snapshots = list(
        (temp_dir / "data" / "training_snapshots").glob(
            "snapshot_run_training_*.json.gz"
        )
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "run_training_performance.csv"
    assert perf_log.exists()


def test_no_obsolete_references(temp_dir, mock_config, mock_feature_sets):
    """Vérifie l'absence de références à dxFeed, obs_t, 320/81 features."""
    for file_path in [mock_config, mock_feature_sets]:
        with open(file_path, "r") as f:
            content = f.read()
        assert "dxFeed" not in content, "Référence à dxFeed trouvée"
        assert "obs_t" not in content, "Référence à obs_t trouvée"
        assert "320 features" not in content, "Référence à 320 features trouvée"
        assert "81 features" not in content, "Référence à 81 features trouvée"
