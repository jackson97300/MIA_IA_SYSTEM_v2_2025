# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_microstructure_guard.py
# Tests unitaires pour src/features/microstructure_guard.py
# Version : 2.1.3
# Date : 2025-05-13
# Rôle : Valide le calcul des scores iceberg et spoofing avec validation SHAP (Phase 17),
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

from src.features.microstructure_guard import MicrostructureGuard


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, snapshots, figures, cache, et données."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    snapshots_dir = data_dir / "microstructure_snapshots"
    snapshots_dir.mkdir()
    figures_dir = data_dir / "figures" / "microstructure"
    figures_dir.mkdir(parents=True)
    features_dir = data_dir / "features"
    features_dir.mkdir()
    cache_dir = features_dir / "cache" / "microstructure"
    cache_dir.mkdir(parents=True)

    # Créer es_config.yaml
    config_path = config_dir / "es_config.yaml"
    config_content = {
        "microstructure_guard": {
            "window_size": 20,
            "iceberg_threshold": 0.8,
            "spoofing_spread_threshold": 0.05,
            "min_rows": 10,
            "buffer_size": 100,
            "max_cache_size": 1000,
            "buffer_maxlen": 1000,
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
        "figures_dir": str(figures_dir),
        "features_dir": str(features_dir),
        "cache_dir": str(cache_dir),
        "perf_log_path": str(logs_dir / "microstructure_guard_performance.csv"),
        "feature_importance_path": str(features_dir / "feature_importance.csv"),
    }


@pytest.fixture
def mock_data():
    """Crée des données factices pour tester."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=100, freq="1min"),
            "bid_size_level_1": np.random.randint(100, 1000, 100),
            "ask_size_level_1": np.random.randint(100, 1000, 100),
            "bid_price_level_1": np.random.uniform(5090, 5110, 100),
            "ask_price_level_1": np.random.uniform(5090, 5110, 100),
            "bid_size_level_2": np.random.randint(50, 500, 100),
            "ask_size_level_2": np.random.randint(50, 500, 100),
        }
    )


@pytest.fixture
def mock_feature_importance(tmp_dirs):
    """Crée un fichier feature_importance.csv factice."""
    shap_data = pd.DataFrame(
        {
            "feature": ["iceberg_order_score", "spoofing_score"] * 75,
            "importance": [0.1] * 150,
            "regime": ["range"] * 150,
        }
    )
    shap_data.to_csv(tmp_dirs["feature_importance_path"], index=False, encoding="utf-8")
    return shap_data


def test_microstructure_guard_init(tmp_dirs, mock_feature_importance):
    """Teste l’initialisation de MicrostructureGuard."""
    guard = MicrostructureGuard(config_path=tmp_dirs["config_path"])
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


def test_calculate_iceberg_order_score_valid(
    tmp_dirs, mock_data, mock_feature_importance
):
    """Teste le calcul du score iceberg avec des données valides."""
    guard = MicrostructureGuard(config_path=tmp_dirs["config_path"])
    iceberg_score = guard.calculate_iceberg_order_score(mock_data)
    assert len(iceberg_score) == len(mock_data), "Longueur du score iceberg incorrecte"
    assert (iceberg_score >= 0).all() and (
        iceberg_score <= 1
    ).all(), "Score iceberg hors des bornes [0, 1]"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_calculate_iceberg_order_score" in f and f.endswith(".json")
        for f in snapshot_files
    ), "Snapshot non créé"
    with open(os.path.join(tmp_dirs["snapshots_dir"], snapshot_files[-1]), "r") as f:
        snapshot = json.load(f)
    assert (
        "confidence_drop_rate" in snapshot["data"]
    ), "confidence_drop_rate absent du snapshot"


def test_calculate_spoofing_score_valid(tmp_dirs, mock_data, mock_feature_importance):
    """Teste le calcul du score spoofing avec des données valides."""
    guard = MicrostructureGuard(config_path=tmp_dirs["config_path"])
    spoofing_score = guard.calculate_spoofing_score(mock_data)
    assert len(spoofing_score) == len(
        mock_data
    ), "Longueur du score spoofing incorrecte"
    assert (spoofing_score >= 0).all() and (
        spoofing_score <= 1
    ).all(), "Score spoofing hors des bornes [0, 1]"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_calculate_spoofing_score" in f and f.endswith(".json")
        for f in snapshot_files
    ), "Snapshot non créé"


def test_calculate_iceberg_order_score_invalid_data(tmp_dirs, mock_feature_importance):
    """Teste le calcul du score iceberg avec des données invalides."""
    guard = MicrostructureGuard(config_path=tmp_dirs["config_path"])
    invalid_data = pd.DataFrame({"timestamp": [datetime.now()]})  # Données vides
    with pytest.raises(ValueError, match="DataFrame vide ou insuffisant"):
        guard.calculate_iceberg_order_score(invalid_data)
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "DataFrame vide" in str(e) for e in df["error"].dropna()
    ), "Erreur DataFrame vide non loguée"


def test_calculate_incremental_iceberg_order_score(
    tmp_dirs, mock_data, mock_feature_importance
):
    """Teste le calcul incrémental du score iceberg."""
    guard = MicrostructureGuard(config_path=tmp_dirs["config_path"])
    for i in range(10):  # Remplir le buffer
        guard.buffer.append(mock_data.iloc[i].to_frame().T)
    row = mock_data.iloc[-1]
    score = guard.calculate_incremental_iceberg_order_score(row)
    assert isinstance(score, float), "Score incrémental iceberg n’est pas un float"
    assert 0 <= score <= 1, "Score incrémental iceberg hors des bornes [0, 1]"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_calculate_incremental_iceberg_order_score" in f
        and f.endswith(".json")
        for f in snapshot_files
    ), "Snapshot non créé"


def test_calculate_incremental_spoofing_score(
    tmp_dirs, mock_data, mock_feature_importance
):
    """Teste le calcul incrémental du score spoofing."""
    guard = MicrostructureGuard(config_path=tmp_dirs["config_path"])
    for i in range(10):  # Remplir le buffer
        guard.buffer.append(mock_data.iloc[i].to_frame().T)
    row = mock_data.iloc[-1]
    score = guard.calculate_incremental_spoofing_score(row)
    assert isinstance(score, float), "Score incrémental spoofing n’est pas un float"
    assert 0 <= score <= 1, "Score incrémental spoofing hors des bornes [0, 1]"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_calculate_incremental_spoofing_score" in f and f.endswith(".json")
        for f in snapshot_files
    ), "Snapshot non créé"


def test_validate_shap_features(tmp_dirs, mock_feature_importance):
    """Teste la validation des top 150 SHAP features."""
    guard = MicrostructureGuard(config_path=tmp_dirs["config_path"])
    features = ["iceberg_order_score", "spoofing_score"]
    result = guard.validate_shap_features(features)
    assert result, "Validation SHAP échouée pour features valides"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_validate_shap_features" in f for f in snapshot_files
    ), "Snapshot non créé"


def test_save_snapshot_compressed(tmp_dirs):
    """Teste la sauvegarde d’un snapshot compressé avec gzip."""
    guard = MicrostructureGuard(config_path=tmp_dirs["config_path"])
    snapshot_data = {"test": "compressed_snapshot"}
    guard.save_snapshot("test_compressed", snapshot_data, compress=True)
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


def test_plot_scores(tmp_dirs, mock_data, mock_feature_importance):
    """Teste la génération des visualisations des scores."""
    guard = MicrostructureGuard(config_path=tmp_dirs["config_path"])
    iceberg_score = guard.calculate_iceberg_order_score(mock_data)
    spoofing_score = guard.calculate_spoofing_score(mock_data)
    timestamp = mock_data["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")
    guard.plot_scores(iceberg_score, spoofing_score, timestamp)
    timestamp_safe = timestamp.replace(":", "-")
    assert os.path.exists(
        os.path.join(tmp_dirs["figures_dir"], f"scores_temporal_{timestamp_safe}.png")
    ), "Visualisation temporelle non générée"
    assert os.path.exists(
        os.path.join(
            tmp_dirs["figures_dir"], f"scores_distribution_{timestamp_safe}.png"
        )
    ), "Distribution non générée"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"


def test_critical_alerts(tmp_dirs, mock_feature_importance):
    """Teste les alertes Telegram pour les erreurs critiques."""
    with patch(
        "src.features.microstructure_guard.send_telegram_alert"
    ) as mock_telegram:
        guard = MicrostructureGuard(config_path=tmp_dirs["config_path"])
        invalid_data = pd.DataFrame(
            {"timestamp": [datetime.now()]}
        )  # Données invalides
        with pytest.raises(ValueError):
            guard.calculate_iceberg_order_score(invalid_data)
        mock_telegram.assert_called_with(pytest.any(str))
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "DataFrame vide" in str(e) for e in df["error"].dropna()
        ), "Erreur critique non loguée"
