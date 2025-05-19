# Teste run_simulate.py.
# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_run_simulate.py
# Tests unitaires pour run_simulate.py.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie les fonctionnalités de SimulationManager, incluant l'initialisation,
#        la validation des données d’entrée, la simulation de trading, les retries,
#        les snapshots JSON compressés, les logs de performance, les graphiques,
#        et les alertes.
#        Conforme à la Phase 8 (auto-conscience via alertes), Phase 12 (simulation de trading),
#        et Phase 16 (ensemble learning).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pandas>=2.0.0
# - psutil>=5.9.8
# - matplotlib>=3.7.0
# - src.scripts.run_simulate
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
from unittest.mock import patch

import pandas as pd
import pytest

from src.scripts.run_simulate import SimulationManager


@pytest.fixture
def temp_dir(tmp_path):
    """Crée un répertoire temporaire pour les tests."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    data_dir = tmp_path / "data"
    logs_dir = data_dir / "logs"
    features_dir = data_dir / "features"
    trades_dir = data_dir / "trades"
    snapshots_dir = data_dir / "simulation_snapshots"
    figures_dir = data_dir / "figures" / "simulation"
    logs_dir.mkdir(parents=True)
    features_dir.mkdir(parents=True)
    trades_dir.mkdir(parents=True)
    snapshots_dir.mkdir(parents=True)
    figures_dir.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def mock_config(temp_dir):
    """Crée un fichier es_config.yaml simulé."""
    config_path = temp_dir / "config" / "es_config.yaml"
    config_content = """
simulation:
  features_path: data/features/features_latest_filtered.csv
  output_path: data/trades/trades_simulated.csv
  capital: 10000.0
  trade_size: 0.1
  chunk_size: 10000
  cache_hours: 24
  retry_attempts: 3
  retry_delay: 5
  timeout_seconds: 3600
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
            "neural_regime": np.random.choice(["trend", "range", "defensive"], 100),
            "rsi": np.random.uniform(0, 100, 100),
            "vix": np.random.uniform(10, 40, 100),
            "bid_size_level_1": np.random.uniform(100, 1000, 100),
            "ask_size_level_1": np.random.uniform(100, 1000, 100),
            "trade_frequency_1s": np.random.uniform(0, 10, 100),
        }
    )
    for i in range(342):  # Ajouter 342 colonnes pour atteindre 350 features
        features[f"feature_{i}"] = np.random.uniform(0, 1, 100)
    features.to_csv(features_path, index=False)
    return features_path


@pytest.fixture
def mock_feature_sets(temp_dir):
    """Crée un fichier feature_sets.yaml simulé."""
    feature_sets_path = temp_dir / "config" / "feature_sets.yaml"
    feature_sets = {
        "training_features": [
            "neural_regime",
            "rsi",
            "vix",
            "bid_size_level_1",
            "ask_size_level_1",
            "trade_frequency_1s",
        ]
        + [f"feature_{i}" for i in range(344)]
    }
    with open(feature_sets_path, "w", encoding="utf-8") as f:
        yaml.dump(feature_sets, f)
    return feature_sets_path


@pytest.fixture
def manager(temp_dir, mock_config, mock_features, mock_feature_sets, monkeypatch):
    """Initialise SimulationManager avec des mocks."""
    monkeypatch.setattr("src.scripts.run_simulate.CONFIG_PATH", str(mock_config))
    monkeypatch.setattr("src.scripts.run_simulate.FEATURES_PATH", str(mock_features))
    monkeypatch.setattr(
        "src.scripts.run_simulate.config_manager.get_config",
        lambda x: {
            "simulation": {
                "features_path": str(mock_features),
                "output_path": str(
                    temp_dir / "data" / "trades" / "trades_simulated.csv"
                ),
                "capital": 10000.0,
                "trade_size": 0.1,
                "chunk_size": 10000,
                "cache_hours": 24,
                "retry_attempts": 3,
                "retry_delay": 5,
                "timeout_seconds": 3600,
                "max_cache_size": 1000,
                "num_features": 350,
                "shap_features": 150,
            }
        },
    )
    manager = SimulationManager()
    return manager


def test_init_manager(temp_dir, mock_config, mock_features, mock_feature_sets, manager):
    """Teste l'initialisation de SimulationManager."""
    assert manager.config["features_path"] == str(mock_features)
    assert os.path.exists(temp_dir / "data" / "simulation_snapshots")
    snapshots = list(
        (temp_dir / "data" / "simulation_snapshots").glob("snapshot_init_*.json.gz")
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "run_simulate_performance.csv"
    assert perf_log.exists()


def test_load_config_invalid(temp_dir, monkeypatch):
    """Teste la validation avec une configuration invalide."""
    invalid_config = temp_dir / "config" / "invalid.yaml"
    monkeypatch.setattr("src.scripts.run_simulate.CONFIG_PATH", str(invalid_config))
    with pytest.raises(FileNotFoundError, match="Fichier config introuvable"):
        SimulationManager()

    with patch("src.scripts.run_simulate.config_manager.get_config", return_value={}):
        monkeypatch.setattr(
            "src.scripts.run_simulate.CONFIG_PATH",
            str(temp_dir / "config" / "es_config.yaml"),
        )
        with pytest.raises(ValueError, match="Clé 'simulation' manquante"):
            SimulationManager()


def test_validate_inputs_valid(temp_dir, mock_features, manager):
    """Teste la validation des paramètres d’entrée valides."""
    manager.validate_inputs(
        str(mock_features),
        str(temp_dir / "data" / "trades" / "trades_simulated.csv"),
        10000.0,
        0.1,
        10000,
    )
    snapshots = list(
        (temp_dir / "data" / "simulation_snapshots").glob(
            "snapshot_validate_inputs_*.json.gz"
        )
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "run_simulate_performance.csv"
    assert perf_log.exists()


def test_validate_inputs_invalid(temp_dir, manager):
    """Teste la validation des paramètres d’entrée invalides."""
    with pytest.raises(ValueError, match="Fichier features introuvable"):
        manager.validate_inputs(
            str(temp_dir / "invalid.csv"),
            str(temp_dir / "data" / "trades" / "trades_simulated.csv"),
            10000.0,
            0.1,
            10000,
        )

    with pytest.raises(ValueError, match="Capital invalide"):
        manager.validate_inputs(
            str(temp_dir / "data" / "features" / "features_latest_filtered.csv"),
            str(temp_dir / "data" / "trades" / "trades_simulated.csv"),
            -1000.0,
            0.1,
            10000,
        )

    snapshots = list(
        (temp_dir / "data" / "simulation_snapshots").glob(
            "snapshot_validate_inputs_*.json.gz"
        )
    )
    assert len(snapshots) >= 2


def test_load_cache_valid(temp_dir, mock_config, mock_features, manager):
    """Teste le chargement d’un cache valide."""
    output_path = temp_dir / "data" / "trades" / "trades_simulated.csv"
    trades = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start="2025-05-13 10:00:00", periods=100, freq="T"
            ),
            "action": np.random.choice(["buy", "sell", "close_short"], 100),
            "price": np.random.uniform(5000, 5100, 100),
            "reward": np.random.uniform(-100, 100, 100),
            "balance": np.random.uniform(9000, 11000, 100),
        }
    )
    trades.to_csv(output_path, index=False)
    os.utime(output_path, (time.time(), time.time()))
    cached_df = manager.load_cache(str(output_path), cache_hours=24)
    assert cached_df is not None
    assert len(cached_df) == 100
    snapshots = list(
        (temp_dir / "data" / "simulation_snapshots").glob(
            "snapshot_load_cache_*.json.gz"
        )
    )
    assert len(snapshots) >= 1


def test_simulate_trading_success(
    temp_dir, mock_config, mock_features, mock_feature_sets, manager
):
    """Teste la simulation de trading avec succès."""
    df_trades = manager.simulate_trading(
        str(mock_features),
        str(temp_dir / "data" / "trades" / "trades_simulated.csv"),
        capital=10000.0,
        trade_size=0.1,
        chunk_size=10000,
    )
    assert not df_trades.empty
    assert all(
        col in df_trades.columns for col in ["timestamp", "action", "reward", "balance"]
    )
    snapshots = list(
        (temp_dir / "data" / "simulation_snapshots").glob(
            "snapshot_simulate_trading_*.json.gz"
        )
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "run_simulate_performance.csv"
    assert perf_log.exists()
    figures = list(
        (temp_dir / "data" / "figures" / "simulation").glob("simulation_balance_*.png")
    )
    assert len(figures) >= 1


def test_run_simulation_success(
    temp_dir, mock_config, mock_features, mock_feature_sets, manager
):
    """Teste l’exécution complète de la simulation avec succès."""
    with patch(
        "src.scripts.run_simulate.SimulationManager.simulate_trading"
    ) as mock_simulate:
        mock_simulate.return_value = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start="2025-05-13 10:00:00", periods=100, freq="T"
                ),
                "action": ["buy"] * 100,
                "price": [5000] * 100,
                "reward": [0] * 100,
                "balance": [10000] * 100,
            }
        )
        status = manager.run_simulation(manager.config)

    assert status["success"] is True
    assert status["trade_count"] == 100
    snapshots = list(
        (temp_dir / "data" / "simulation_snapshots").glob(
            "snapshot_run_simulation_*.json.gz"
        )
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "run_simulate_performance.csv"
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
