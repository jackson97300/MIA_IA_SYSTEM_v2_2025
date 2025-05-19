# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_meta_features.py
# Tests unitaires pour src/features/meta_features.py
# Version : 2.1.3
# Date : 2025-05-13
# Rôle : Valide le calcul des métriques d’auto-analyse (ex. : confidence_drop_rate, sgc_entropy) avec validation SHAP (Phase 17),
#        confidence_drop_rate (Phase 8), et snapshots avec alertes Telegram.

import gzip
import json
import os
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.features.meta_features import MetaFeatures


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, snapshots, cache, et données."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    snapshots_dir = data_dir / "meta_features_snapshots"
    snapshots_dir.mkdir()
    features_dir = data_dir / "features"
    features_dir.mkdir()
    cache_dir = features_dir / "cache" / "meta_features"
    cache_dir.mkdir(parents=True)

    # Créer es_config.yaml
    config_path = config_dir / "es_config.yaml"
    config_content = {
        "meta_features": {
            "confidence_window": 10,
            "error_window": 20,
            "entropy_window": 20,
            "buffer_size": 100,
            "max_cache_size": 1000,
            "cache_hours": 24,
        }
    }
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)

    return {
        "base_dir": str(base_dir),
        "config_path": str(config_path),
        "logs_dir": str(logs_dir),
        "snapshots_dir": str(snapshots_dir),
        "features_dir": str(features_dir),
        "cache_dir": str(cache_dir),
        "perf_log_path": str(logs_dir / "meta_features_performance.csv"),
        "feature_importance_path": str(features_dir / "feature_importance.csv"),
    }


@pytest.fixture
def mock_data():
    """Crée des données factices pour tester."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=100, freq="1min"),
            "close": np.random.normal(5100, 10, 100),
            "predicted_price": np.random.normal(5100, 12, 100),
            "confidence": np.random.uniform(0.4, 0.9, 100),
        }
    )


@pytest.fixture
def mock_feature_importance(tmp_dirs):
    """Crée un fichier feature_importance.csv factice."""
    shap_data = pd.DataFrame(
        {
            "feature": [
                "confidence_drop_rate",
                "error_rolling_std",
                "sgc_entropy",
                "prediction_bias",
                "error_trend",
                "confidence_volatility",
                "signal_stability",
                "memory_retention_score",
            ]
            * 20,
            "importance": [0.1] * 160,
            "regime": ["range"] * 160,
        }
    )
    shap_data.to_csv(tmp_dirs["feature_importance_path"], index=False, encoding="utf-8")
    return shap_data


def test_meta_features_init(tmp_dirs, mock_feature_importance):
    """Teste l’initialisation de MetaFeatures."""
    calculator = MetaFeatures(config_path=tmp_dirs["config_path"])
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    assert os.path.exists(tmp_dirs["snapshots_dir"]), "Dossier de snapshots non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert all(
        col in df.columns
        for col in ["timestamp", "operation", "latency", "cpu_usage_percent"]
    ), "Colonnes de performance manquantes"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_init" in f and f.endswith(".json") for f in snapshot_files
    ), "Snapshot non créé"


def test_calculate_meta_features_valid(tmp_dirs, mock_data, mock_feature_importance):
    """Teste le calcul des métriques avec des données valides."""
    calculator = MetaFeatures(config_path=tmp_dirs["config_path"])
    enriched_data = calculator.calculate_meta_features(mock_data)
    expected_features = [
        "confidence_drop_rate",
        "error_rolling_std",
        "sgc_entropy",
        "prediction_bias",
        "error_trend",
        "confidence_volatility",
        "signal_stability",
        "memory_retention_score",
    ]
    assert all(
        f in enriched_data.columns for f in expected_features
    ), "Features manquantes dans les données enrichies"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_meta_features" in f and f.endswith(".json") for f in snapshot_files
    ), "Snapshot non créé"
    with open(os.path.join(tmp_dirs["snapshots_dir"], snapshot_files[-1]), "r") as f:
        snapshot = json.load(f)
    assert (
        "confidence_drop_rate" in snapshot["data"]
    ), "confidence_drop_rate absent du snapshot"


def test_calculate_meta_features_invalid_data(tmp_dirs, mock_feature_importance):
    """Teste le calcul des métriques avec des données invalides."""
    calculator = MetaFeatures(config_path=tmp_dirs["config_path"])
    invalid_data = pd.DataFrame({"timestamp": [datetime.now()]})  # Données vides
    enriched_data = calculator.calculate_meta_features(invalid_data)
    expected_features = [
        "confidence_drop_rate",
        "error_rolling_std",
        "sgc_entropy",
        "prediction_bias",
    ]
    assert all(
        f in enriched_data.columns for f in expected_features
    ), "Features manquantes dans les données enrichies"
    assert (
        enriched_data[expected_features].eq(0.0).all().all()
    ), "Features non nulles pour données invalides"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "DataFrame vide" in str(e) for e in df["error"].dropna()
    ), "Erreur DataFrame vide non loguée"


def test_calculate_incremental_meta_features(
    tmp_dirs, mock_data, mock_feature_importance
):
    """Teste le calcul incrémental des métriques."""
    calculator = MetaFeatures(config_path=tmp_dirs["config_path"])
    buffer = mock_data.iloc[:-1]
    row = mock_data.iloc[-1]
    result = calculator.calculate_incremental_meta_features(row, buffer)
    expected_features = [
        "confidence_drop_rate",
        "error_rolling_std",
        "sgc_entropy",
        "prediction_bias",
        "error_trend",
        "confidence_volatility",
        "signal_stability",
        "memory_retention_score",
    ]
    assert all(
        f in result.index for f in expected_features
    ), "Features manquantes dans le résultat incrémental"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_incremental_meta_features" in f and f.endswith(".json")
        for f in snapshot_files
    ), "Snapshot non créé"


def test_validate_shap_features(tmp_dirs, mock_feature_importance):
    """Teste la validation des top 150 SHAP features."""
    calculator = MetaFeatures(config_path=tmp_dirs["config_path"])
    features = ["confidence_drop_rate", "error_rolling_std", "sgc_entropy"]
    result = calculator.validate_shap_features(features)
    assert result, "Validation SHAP échouée pour features valides"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_validate_shap_features" in f for f in snapshot_files
    ), "Snapshot non créé"


def test_save_snapshot_compressed(tmp_dirs):
    """Teste la sauvegarde d’un snapshot compressé avec gzip."""
    calculator = MetaFeatures(config_path=tmp_dirs["config_path"])
    snapshot_data = {"test": "compressed_snapshot"}
    calculator.save_snapshot("test_compressed", snapshot_data, compress=True)
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_test_compressed" in f and f.endswith(".json.gz")
        for f in snapshot_files
    ), "Snapshot compressé non créé"
    with gzip.open(
        os.path.join(tmp_dirs["snapshots_dir"], snapshot_files[-1]),
        "rt",
        encoding="utf-8",
    ) as f:
        snapshot = json.load(f)
    assert snapshot["data"] == snapshot_data, "Contenu du snapshot compressé incorrect"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"


def test_critical_alerts(tmp_dirs, mock_feature_importance):
    """Teste les alertes Telegram pour les erreurs critiques."""
    with patch("src.features.meta_features.send_telegram_alert") as mock_telegram:
        calculator = MetaFeatures(config_path=tmp_dirs["config_path"])
        invalid_data = pd.DataFrame(
            {"timestamp": [datetime.now()]}
        )  # Données invalides
        calculator.calculate_meta_features(invalid_data)
        mock_telegram.assert_called_with(pytest.any(str))
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "DataFrame vide" in str(e) for e in df["error"].dropna()
        ), "Erreur critique non loguée"
