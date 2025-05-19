# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_pca_orderflow.py
# Tests unitaires pour src/features/pca_orderflow.py
# Version : 2.1.3
# Date : 2025-05-13
# Rôle : Valide l’application PCA sur les features d’order flow avec pondération par régime (méthode 3),
#        validation SHAP (Phase 17), confidence_drop_rate (Phase 8), et snapshots.

import gzip
import json
import os
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.features.pca_orderflow import PCAOrderFlow


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, snapshots, figures, modèles, et données."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    snapshots_dir = data_dir / "pca_snapshots"
    snapshots_dir.mkdir()
    figures_dir = data_dir / "figures" / "pca"
    figures_dir.mkdir(parents=True)
    features_dir = data_dir / "features"
    features_dir.mkdir()
    models_dir = data_dir / "models"
    models_dir.mkdir()

    # Créer es_config.yaml
    config_path = config_dir / "es_config.yaml"
    config_content = {
        "pca_orderflow": {
            "n_components": 2,
            "output_csv": str(features_dir / "pca_orderflow.csv"),
            "window_size": 100,
            "model_path": str(models_dir / "pca_orderflow.pkl"),
            "scaler_path": str(models_dir / "scaler_orderflow.pkl"),
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
        "models_dir": str(models_dir),
        "perf_log_path": str(logs_dir / "pca_orderflow_performance.csv"),
        "feature_importance_path": str(features_dir / "feature_importance.csv"),
        "output_csv": str(features_dir / "pca_orderflow.csv"),
    }


@pytest.fixture
def mock_data():
    """Crée des données factices pour tester."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=100, freq="1min"),
            "delta_volume": np.random.uniform(-500, 500, 100),
            "obi_score": np.random.uniform(-1, 1, 100),
            "vds": np.random.uniform(-200, 200, 100),
            "absorption_strength": np.random.uniform(0, 100, 100),
            "bid_size_level_1": np.random.randint(100, 1000, 100),
            "ask_size_level_1": np.random.randint(100, 1000, 100),
            "trade_frequency_1s": np.random.randint(0, 50, 100),
            "close": np.random.normal(5100, 10, 100),
        }
    )


@pytest.fixture
def mock_feature_importance(tmp_dirs):
    """Crée un fichier feature_importance.csv factice."""
    shap_data = pd.DataFrame(
        {
            "feature": ["pca_orderflow_1", "pca_orderflow_2"] * 75,
            "importance": [0.1] * 150,
            "regime": ["range"] * 150,
        }
    )
    shap_data.to_csv(tmp_dirs["feature_importance_path"], index=False, encoding="utf-8")
    return shap_data


def test_pca_orderflow_init(tmp_dirs, mock_feature_importance):
    """Teste l’initialisation de PCAOrderFlow."""
    calculator = PCAOrderFlow(config_path=tmp_dirs["config_path"])
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


def test_apply_pca_orderflow_valid(tmp_dirs, mock_data, mock_feature_importance):
    """Teste l’application PCA avec des données valides."""
    calculator = PCAOrderFlow(config_path=tmp_dirs["config_path"])
    enriched_data = calculator.apply_pca_orderflow(mock_data, regime="trend")
    assert "pca_orderflow_1" in enriched_data.columns, "pca_orderflow_1 manquant"
    assert "pca_orderflow_2" in enriched_data.columns, "pca_orderflow_2 manquant"
    assert os.path.exists(tmp_dirs["output_csv"]), "Fichier pca_orderflow.csv non créé"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_apply_pca_orderflow" in f and f.endswith(".json")
        for f in snapshot_files
    ), "Snapshot non créé"
    with open(os.path.join(tmp_dirs["snapshots_dir"], snapshot_files[-1]), "r") as f:
        snapshot = json.load(f)
    assert (
        "confidence_drop_rate" in snapshot["data"]
    ), "confidence_drop_rate absent du snapshot"
    timestamp_safe = mock_data["timestamp"].iloc[-1].strftime("%Y-%m-%d %H-%M-%S")
    assert os.path.exists(
        os.path.join(tmp_dirs["figures_dir"], f"pca_scatter_{timestamp_safe}.png")
    ), "Visualisation scatter non générée"


def test_apply_pca_orderflow_invalid_data(tmp_dirs, mock_feature_importance):
    """Teste l’application PCA avec des données invalides."""
    calculator = PCAOrderFlow(config_path=tmp_dirs["config_path"])
    invalid_data = pd.DataFrame({"timestamp": [datetime.now()]})  # Données vides
    enriched_data = calculator.apply_pca_orderflow(invalid_data, regime="trend")
    assert (
        enriched_data["pca_orderflow_1"].eq(0.0).all()
    ), "pca_orderflow_1 non nul pour données invalides"
    assert (
        enriched_data["pca_orderflow_2"].eq(0.0).all()
    ), "pca_orderflow_2 non nul pour données invalides"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "DataFrame vide" in str(e) for e in df["error"].dropna()
    ), "Erreur DataFrame vide non loguée"


def test_apply_incremental_pca_orderflow(tmp_dirs, mock_data, mock_feature_importance):
    """Teste l’application incrémentale PCA."""
    calculator = PCAOrderFlow(config_path=tmp_dirs["config_path"])
    calculator.apply_pca_orderflow(
        mock_data, regime="trend"
    )  # Initialiser scaler et PCA
    row = mock_data.iloc[-1]
    for i in range(100):  # Remplir le buffer
        calculator.buffer.append(mock_data.iloc[i].to_frame().T)
    result = calculator.apply_incremental_pca_orderflow(row, regime="trend")
    assert "pca_orderflow_1" in result.index, "pca_orderflow_1 manquant"
    assert "pca_orderflow_2" in result.index, "pca_orderflow_2 manquant"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_apply_incremental_pca_orderflow" in f and f.endswith(".json")
        for f in snapshot_files
    ), "Snapshot non créé"


def test_validate_shap_features(tmp_dirs, mock_feature_importance):
    """Teste la validation des top 150 SHAP features."""
    calculator = PCAOrderFlow(config_path=tmp_dirs["config_path"])
    features = ["pca_orderflow_1", "pca_orderflow_2"]
    result = calculator.validate_shap_features(features)
    assert result, "Validation SHAP échouée pour features valides"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_validate_shap_features" in f for f in snapshot_files
    ), "Snapshot non créé"


def test_save_snapshot_compressed(tmp_dirs):
    """Teste la sauvegarde d’un snapshot compressé avec gzip."""
    calculator = PCAOrderFlow(config_path=tmp_dirs["config_path"])
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


def test_plot_pca_results(tmp_dirs, mock_data, mock_feature_importance):
    """Teste la génération des visualisations PCA."""
    calculator = PCAOrderFlow(config_path=tmp_dirs["config_path"])
    enriched_data = calculator.apply_pca_orderflow(mock_data, regime="trend")
    pca_result = enriched_data[["pca_orderflow_1", "pca_orderflow_2"]].tail(100).values
    explained_variance = np.array([0.6, 0.3])
    timestamp = mock_data["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")
    calculator.plot_pca_results(pca_result, timestamp, explained_variance)
    timestamp_safe = timestamp.replace(":", "-")
    assert os.path.exists(
        os.path.join(tmp_dirs["figures_dir"], f"pca_scatter_{timestamp_safe}.png")
    ), "Visualisation scatter non générée"
    assert os.path.exists(
        os.path.join(tmp_dirs["figures_dir"], f"pca_scree_{timestamp_safe}.png")
    ), "Visualisation scree non générée"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"


def test_critical_alerts(tmp_dirs, mock_feature_importance):
    """Teste les alertes Telegram pour les erreurs critiques."""
    with patch("src.features.pca_orderflow.send_telegram_alert") as mock_telegram:
        calculator = PCAOrderFlow(config_path=tmp_dirs["config_path"])
        invalid_data = pd.DataFrame(
            {"timestamp": [datetime.now()]}
        )  # Données invalides
        calculator.apply_pca_orderflow(invalid_data, regime="trend")
        mock_telegram.assert_called_with(pytest.any(str))
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "DataFrame vide" in str(e) for e in df["error"].dropna()
        ), "Erreur critique non loguée"
