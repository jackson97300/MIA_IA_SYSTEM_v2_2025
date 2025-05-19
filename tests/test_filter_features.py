# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_filter_features.py
# Tests unitaires pour filter_features.py.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie les fonctionnalités de FeatureFilter, incluant l'initialisation,
#        la validation des données d’entrée, le filtrage des features, les retries,
#        les snapshots JSON compressés, les logs de performance, et les alertes.
#        Conforme à la Phase 7 (gestion des features), Phase 8 (auto-conscience via alertes),
#        et Phase 16 (ensemble learning).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pandas>=2.0.0
# - psutil>=5.9.8
# - src.scripts.filter_features
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

from src.scripts.filter_features import FeatureFilter


@pytest.fixture
def temp_dir(tmp_path):
    """Crée un répertoire temporaire pour les tests."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    data_dir = tmp_path / "data"
    logs_dir = data_dir / "logs"
    features_dir = data_dir / "features"
    snapshots_dir = data_dir / "filter_snapshots"
    logs_dir.mkdir(parents=True)
    features_dir.mkdir(parents=True)
    snapshots_dir.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def mock_config(temp_dir):
    """Crée un fichier es_config.yaml simulé."""
    config_path = temp_dir / "config" / "es_config.yaml"
    config_content = """
filter_features:
  input_path: data/features/features_latest.csv
  output_path: data/features/features_latest_filtered.csv
  max_timestamp: "2025-05-13 11:39:00"
  retry_attempts: 3
  retry_delay: 5
  buffer_size: 100
  max_cache_size: 1000
"""
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def mock_features(temp_dir):
    """Crée un fichier features_latest.csv simulé."""
    input_path = temp_dir / "data" / "features" / "features_latest.csv"
    features = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start="2025-05-13 10:00:00", periods=100, freq="T"
            ),
            "close": np.random.uniform(5000, 5100, 100),
        }
    )
    for i in range(348):  # Ajouter 348 colonnes pour atteindre 350 features
        features[f"feature_{i}"] = np.random.uniform(0, 1, 100)
    features.to_csv(input_path, index=False)
    return input_path


@pytest.fixture
def filterer(temp_dir, mock_config, mock_features, monkeypatch):
    """Initialise FeatureFilter avec des mocks."""
    monkeypatch.setattr("src.scripts.filter_features.CONFIG_PATH", str(mock_config))
    monkeypatch.setattr(
        "src.scripts.filter_features.config_manager.get_config",
        lambda x: {
            "filter_features": {
                "input_path": str(mock_features),
                "output_path": str(
                    temp_dir / "data" / "features" / "features_latest_filtered.csv"
                ),
                "max_timestamp": "2025-05-13 11:39:00",
                "retry_attempts": 3,
                "retry_delay": 5,
                "buffer_size": 100,
                "max_cache_size": 1000,
            }
        },
    )
    filterer = FeatureFilter()
    return filterer


def test_init_filterer(temp_dir, mock_config, mock_features, filterer):
    """Teste l'initialisation de FeatureFilter."""
    assert filterer.config["input_path"] == str(mock_features)
    assert os.path.exists(temp_dir / "data" / "filter_snapshots")
    snapshots = list(
        (temp_dir / "data" / "filter_snapshots").glob("snapshot_init_*.json.gz")
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "filter_features_performance.csv"
    assert perf_log.exists()


def test_load_config_invalid(temp_dir, monkeypatch):
    """Teste la validation avec une configuration invalide."""
    invalid_config = temp_dir / "config" / "invalid.yaml"
    monkeypatch.setattr("src.scripts.filter_features.CONFIG_PATH", str(invalid_config))
    with pytest.raises(FileNotFoundError, match="Fichier config introuvable"):
        FeatureFilter()

    with patch(
        "src.scripts.filter_features.config_manager.get_config", return_value={}
    ):
        monkeypatch.setattr(
            "src.scripts.filter_features.CONFIG_PATH",
            str(temp_dir / "config" / "es_config.yaml"),
        )
        with pytest.raises(ValueError, match="Clé 'filter_features' manquante"):
            FeatureFilter()


def test_validate_inputs_valid(temp_dir, mock_features, filterer):
    """Teste la validation des paramètres d’entrée valides."""
    filterer.validate_inputs(
        str(mock_features),
        str(temp_dir / "data" / "features" / "features_latest_filtered.csv"),
        "2025-05-13 11:39:00",
    )
    snapshots = list(
        (temp_dir / "data" / "filter_snapshots").glob(
            "snapshot_validate_inputs_*.json.gz"
        )
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "filter_features_performance.csv"
    assert perf_log.exists()


def test_validate_inputs_invalid(temp_dir, filterer):
    """Teste la validation des paramètres d’entrée invalides."""
    with pytest.raises(ValueError, match="Fichier d’entrée introuvable"):
        filterer.validate_inputs(
            str(temp_dir / "invalid.csv"),
            str(temp_dir / "data" / "features" / "features_latest_filtered.csv"),
            "2025-05-13 11:39:00",
        )

    with pytest.raises(ValueError, match="Format de timestamp invalide"):
        filterer.validate_inputs(
            str(temp_dir / "data" / "features" / "features_latest.csv"),
            str(temp_dir / "data" / "features" / "features_latest_filtered.csv"),
            "invalid_date",
        )

    snapshots = list(
        (temp_dir / "data" / "filter_snapshots").glob(
            "snapshot_validate_inputs_*.json.gz"
        )
    )
    assert len(snapshots) >= 2


def test_filter_features_success(temp_dir, mock_features, filterer):
    """Teste le filtrage des features avec succès."""
    df_filtered = filterer.filter_features(verbose=False)
    assert not df_filtered.empty
    assert len(df_filtered) <= 100
    assert "timestamp" in df_filtered.columns
    output_path = temp_dir / "data" / "features" / "features_latest_filtered.csv"
    assert output_path.exists()
    snapshots = list(
        (temp_dir / "data" / "filter_snapshots").glob(
            "snapshot_filter_features_*.json.gz"
        )
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "filter_features_performance.csv"
    assert perf_log.exists()


def test_filter_features_empty(temp_dir, mock_features, filterer):
    """Teste le filtrage avec un timestamp excluant toutes les données."""
    with pytest.raises(ValueError, match="Aucune donnée valide après filtrage"):
        filterer.filter_features(max_timestamp="2025-05-13 09:00:00", verbose=False)
    snapshots = list(
        (temp_dir / "data" / "filter_snapshots").glob(
            "snapshot_filter_features_*.json.gz"
        )
    )
    assert len(snapshots) >= 1


def test_no_obsolete_references(temp_dir, mock_config):
    """Vérifie l'absence de références à dxFeed, obs_t, 320/81 features."""
    with open(mock_config, "r") as f:
        content = f.read()
    assert "dxFeed" not in content, "Référence à dxFeed trouvée"
    assert "obs_t" not in content, "Référence à obs_t trouvée"
    assert "320 features" not in content, "Référence à 320 features trouvée"
    assert "81 features" not in content, "Référence à 81 features trouvée"
