# Teste run_preprocessing.py.
# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_run_preprocessing.py
# Tests unitaires pour run_preprocessing.py.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie les fonctionnalités de PreprocessingRunner, incluant l'initialisation,
#        la validation des données d’entrée, le preprocessing des features, les retries,
#        les snapshots JSON compressés, les logs de performance, les graphiques,
#        et les alertes.
#        Conforme à la Phase 7 (génération des features), Phase 8 (auto-conscience via alertes),
#        et Phase 16 (ensemble learning).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pandas>=2.0.0
# - numpy>=1.23.0
# - psutil>=5.9.8
# - matplotlib>=3.7.0
# - ta
# - src.scripts.run_preprocessing
# - src.api.merge_data_sources
# - src.features.feature_pipeline
# - src.features.neural_pipeline
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

import numpy as np
import pandas as pd
import pytest
import yaml

from src.scripts.run_preprocessing import PreprocessingRunner


@pytest.fixture
def temp_dir(tmp_path):
    """Crée un répertoire temporaire pour les tests."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    data_dir = tmp_path / "data"
    logs_dir = data_dir / "logs"
    ibkr_dir = data_dir / "ibkr"
    news_dir = data_dir / "news"
    features_dir = data_dir / "features"
    snapshots_dir = data_dir / "preprocessing_snapshots"
    figures_dir = data_dir / "figures" / "preprocessing"
    logs_dir.mkdir(parents=True)
    ibkr_dir.mkdir(parents=True)
    news_dir.mkdir(parents=True)
    features_dir.mkdir(parents=True)
    snapshots_dir.mkdir(parents=True)
    figures_dir.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def mock_config(temp_dir):
    """Crée un fichier es_config.yaml simulé."""
    config_path = temp_dir / "config" / "es_config.yaml"
    config_content = """
preprocessing:
  input_path: data/ibkr/ibkr_data.csv
  output_path: data/features/features_latest_filtered.csv
  news_path: data/news/news_data.csv
  chunk_size: 10000
  cache_hours: 24
  retry_attempts: 3
  retry_delay: 10
  timeout_seconds: 1800
  max_timestamp: "2025-05-13 11:39:00"
  buffer_size: 100
  max_cache_size: 1000
  num_features: 350
  shap_features: 150
"""
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def mock_ibkr_data(temp_dir):
    """Crée un fichier ibkr_data.csv simulé."""
    ibkr_path = temp_dir / "data" / "ibkr" / "ibkr_data.csv"
    ibkr_data = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start="2025-05-13 10:00:00", periods=100, freq="T"
            ),
            "close": np.random.uniform(5000, 5100, 100),
            "bid_size_level_1": np.random.uniform(100, 1000, 100),
            "ask_size_level_1": np.random.uniform(100, 1000, 100),
            "trade_frequency_1s": np.random.uniform(0, 10, 100),
        }
    )
    ibkr_data.to_csv(ibkr_path, index=False)
    return ibkr_path


@pytest.fixture
def mock_news_data(temp_dir):
    """Crée un fichier news_data.csv simulé."""
    news_path = temp_dir / "data" / "news" / "news_data.csv"
    news_data = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start="2025-05-13 10:00:00", periods=100, freq="T"
            ),
            "sentiment_label": np.random.choice(
                ["positive", "negative", "neutral"], 100
            ),
        }
    )
    news_data.to_csv(news_path, index=False)
    return news_path


@pytest.fixture
def mock_feature_sets(temp_dir):
    """Crée un fichier feature_sets.yaml simulé."""
    feature_sets_path = temp_dir / "config" / "feature_sets.yaml"
    feature_sets = {
        "training_features": [
            "rsi",
            "rsi_14",
            "rsi_21",
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
def runner(
    temp_dir,
    mock_config,
    mock_ibkr_data,
    mock_news_data,
    mock_feature_sets,
    monkeypatch,
):
    """Initialise PreprocessingRunner avec des mocks."""
    monkeypatch.setattr("src.scripts.run_preprocessing.CONFIG_PATH", str(mock_config))
    monkeypatch.setattr(
        "src.scripts.run_preprocessing.config_manager.get_config",
        lambda x: {
            "preprocessing": {
                "input_path": str(mock_ibkr_data),
                "output_path": str(
                    temp_dir / "data" / "features" / "features_latest_filtered.csv"
                ),
                "news_path": str(mock_news_data),
                "chunk_size": 10000,
                "cache_hours": 24,
                "retry_attempts": 3,
                "retry_delay": 10,
                "timeout_seconds": 1800,
                "max_timestamp": "2025-05-13 11:39:00",
                "buffer_size": 100,
                "max_cache_size": 1000,
                "num_features": 350,
                "shap_features": 150,
            }
        },
    )
    runner = PreprocessingRunner()
    return runner


def test_init_runner(
    temp_dir, mock_config, mock_ibkr_data, mock_news_data, mock_feature_sets, runner
):
    """Teste l'initialisation de PreprocessingRunner."""
    assert runner.config["input_path"] == str(mock_ibkr_data)
    assert os.path.exists(temp_dir / "data" / "preprocessing_snapshots")
    snapshots = list(
        (temp_dir / "data" / "preprocessing_snapshots").glob("snapshot_init_*.json.gz")
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "run_preprocessing_performance.csv"
    assert perf_log.exists()


def test_load_config_invalid(temp_dir, monkeypatch):
    """Teste la validation avec une configuration invalide."""
    invalid_config = temp_dir / "config" / "invalid.yaml"
    monkeypatch.setattr(
        "src.scripts.run_preprocessing.CONFIG_PATH", str(invalid_config)
    )
    with pytest.raises(FileNotFoundError, match="Fichier config introuvable"):
        PreprocessingRunner()

    with patch(
        "src.scripts.run_preprocessing.config_manager.get_config", return_value={}
    ):
        monkeypatch.setattr(
            "src.scripts.run_preprocessing.CONFIG_PATH",
            str(temp_dir / "config" / "es_config.yaml"),
        )
        with pytest.raises(ValueError, match="Clé 'preprocessing' manquante"):
            PreprocessingRunner()


def test_validate_inputs_valid(temp_dir, mock_ibkr_data, mock_news_data, runner):
    """Teste la validation des paramètres d’entrée valides."""
    runner.validate_inputs(
        str(mock_ibkr_data),
        str(temp_dir / "data" / "features" / "features_latest_filtered.csv"),
        str(mock_news_data),
        10000,
    )
    snapshots = list(
        (temp_dir / "data" / "preprocessing_snapshots").glob(
            "snapshot_validate_inputs_*.json.gz"
        )
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "run_preprocessing_performance.csv"
    assert perf_log.exists()


def test_validate_inputs_invalid(temp_dir, runner):
    """Teste la validation des paramètres d’entrée invalides."""
    with pytest.raises(ValueError, match="Fichier IBKR introuvable"):
        runner.validate_inputs(
            str(temp_dir / "invalid.csv"),
            str(temp_dir / "data" / "features" / "features_latest_filtered.csv"),
            str(temp_dir / "data" / "news" / "news_data.csv"),
            10000,
        )

    with pytest.raises(ValueError, match="Fichier de news introuvable"):
        runner.validate_inputs(
            str(temp_dir / "data" / "ibkr" / "ibkr_data.csv"),
            str(temp_dir / "data" / "features" / "features_latest_filtered.csv"),
            str(temp_dir / "invalid.csv"),
            10000,
        )

    snapshots = list(
        (temp_dir / "data" / "preprocessing_snapshots").glob(
            "snapshot_validate_inputs_*.json.gz"
        )
    )
    assert len(snapshots) >= 2


def test_load_cache_valid(
    temp_dir, mock_config, mock_ibkr_data, mock_news_data, runner
):
    """Teste le chargement d’un cache valide."""
    output_path = temp_dir / "data" / "features" / "features_latest_filtered.csv"
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
    features.to_csv(output_path, index=False)
    os.utime(output_path, (time.time(), time.time()))
    cached_df = runner.load_cache(str(output_path), cache_hours=24)
    assert cached_df is not None
    assert len(cached_df) == 100
    snapshots = list(
        (temp_dir / "data" / "preprocessing_snapshots").glob(
            "snapshot_load_cache_*.json.gz"
        )
    )
    assert len(snapshots) >= 1


def test_preprocess_features_success(
    temp_dir, mock_config, mock_ibkr_data, mock_news_data, mock_feature_sets, runner
):
    """Teste le preprocessing des features avec succès."""
    with patch(
        "src.api.merge_data_sources.merge_data_sources",
        return_value=pd.DataFrame(
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
        ),
    ), patch(
        "src.features.feature_pipeline.feature_pipeline",
        return_value=pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start="2025-05-13 10:00:00", periods=100, freq="T"
                ),
                "close": np.random.uniform(5000, 5100, 100),
                "rsi": np.random.uniform(0, 100, 100),
                "rsi_14": np.random.uniform(0, 100, 100),
            }
        ),
    ), patch(
        "src.features.feature_pipeline.calculate_technical_features",
        return_value=pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start="2025-05-13 10:00:00", periods=100, freq="T"
                ),
                "close": np.random.uniform(5000, 5100, 100),
                "rsi": np.random.uniform(0, 100, 100),
                "rsi_14": np.random.uniform(0, 100, 100),
                "rsi_7": np.random.uniform(0, 100, 100),
            }
        ),
    ), patch(
        "src.features.neural_pipeline.generate_neural_features",
        return_value=pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start="2025-05-13 10:00:00", periods=100, freq="T"
                ),
                "close": np.random.uniform(5000, 5100, 100),
                "rsi": np.random.uniform(0, 100, 100),
                "rsi_14": np.random.uniform(0, 100, 100),
                "rsi_7": np.random.uniform(0, 100, 100),
                "neural_regime": np.random.choice(["trend", "range", "defensive"], 100),
            }
        ),
    ):
        df = runner.preprocess_features(
            str(mock_ibkr_data),
            str(temp_dir / "data" / "features" / "features_latest_filtered.csv"),
            10000,
            runner.config,
        )

    assert not df.empty
    assert "timestamp" in df.columns
    assert "close" in df.columns
    snapshots = list(
        (temp_dir / "data" / "preprocessing_snapshots").glob(
            "snapshot_preprocess_features_*.json.gz"
        )
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "run_preprocessing_performance.csv"
    assert perf_log.exists()
    figures = list(
        (temp_dir / "data" / "figures" / "preprocessing").glob(
            "preprocessing_status_*.png"
        )
    )
    assert len(figures) >= 1


def test_run_preprocessing_success(
    temp_dir, mock_config, mock_ibkr_data, mock_news_data, mock_feature_sets, runner
):
    """Teste l’exécution complète du preprocessing avec succès."""
    with patch(
        "src.scripts.run_preprocessing.PreprocessingRunner.preprocess_features",
        return_value=pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start="2025-05-13 10:00:00", periods=100, freq="T"
                ),
                "close": np.random.uniform(5000, 5100, 100),
                "rsi": np.random.uniform(0, 100, 100),
            }
        ),
    ):
        status = runner.run_preprocessing()

    assert status["success"] is True
    assert status["feature_count"] > 0
    snapshots = list(
        (temp_dir / "data" / "preprocessing_snapshots").glob(
            "snapshot_run_preprocessing_*.json.gz"
        )
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "run_preprocessing_performance.csv"
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
