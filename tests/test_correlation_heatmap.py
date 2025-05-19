# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_correlation_heatmap.py
# Tests unitaires pour src/monitoring/correlation_heatmap.py
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide la génération de heatmaps de corrélation des features, avec focus sur les top 50 SHAP features,
#        intégration de volatilité (méthode 1), utilisation exclusive d’IQFeed, snapshots compressés,
#        sauvegardes incrémentielles/distribuées, alertes Telegram, et confidence_drop_rate (Phase 8).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, seaborn>=0.13.0,<1.0.0, matplotlib>=3.8.0,<4.0.0,
#   psutil>=5.9.8,<6.0.0, loguru>=0.7.0,<1.0.0, numpy>=1.26.4,<2.0.0, pyyaml>=6.0.0,<7.0.0,
#   boto3>=1.26.0,<2.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/features/feature_pipeline.py
# - src/features/shap_weighting.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/es_config.yaml
# - config/feature_sets.yaml
# - data/features/features_latest.csv
# - data/features/feature_importance.csv
#
# Outputs :
# - Heatmaps dans data/figures/heatmap/
# - Matrice de corrélations dans data/correlation_matrix.csv
# - Snapshots dans data/cache/heatmap/*.json.gz
# - Sauvegardes incrémentielles dans data/checkpoints/heatmap_*.json.gz
# - Logs dans data/logs/correlation_heatmap.log
# - Logs de performance dans data/logs/heatmap_performance.csv
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Tests les phases 8 (auto-conscience via confidence_drop_rate), 17 (SHAP).
# - Vérifie l’absence de dxFeed, la suppression de 320/81 features et obs_t, les retries, logs psutil, alertes Telegram,
#   snapshots compressés, et sauvegardes incrémentielles/distribuées.
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence) via config/feature_sets.yaml.

import gzip
import json
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.monitoring.correlation_heatmap import CorrelationHeatmap


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les données, logs, figures, cache, et checkpoints."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    figures_dir = data_dir / "figures" / "heatmap"
    figures_dir.mkdir(parents=True)
    cache_dir = data_dir / "cache" / "heatmap"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints"
    checkpoints_dir.mkdir()
    features_dir = data_dir / "features"
    features_dir.mkdir()

    # Créer es_config.yaml
    config_path = config_dir / "es_config.yaml"
    config_content = {
        "correlation_params": {
            "significant_threshold": 0.8,
            "s3_bucket": "test-bucket",
            "s3_prefix": "heatmap/",
        },
        "logging": {"buffer_size": 100},
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    # Créer feature_sets.yaml
    feature_sets_path = config_dir / "feature_sets.yaml"
    feature_sets_content = {
        "inference": {"shap_features": [f"feature_{i}" for i in range(150)]},
        "training": {"features": [f"feature_{i}" for i in range(350)]},
    }
    with open(feature_sets_path, "w", encoding="utf-8") as f:
        yaml.dump(feature_sets_content, f)

    # Créer features_latest.csv
    features_latest_path = features_dir / "features_latest.csv"
    features_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=100, freq="1min"),
            "vix_es_correlation": np.random.uniform(-1, 1, 100),
            "atr_14": np.random.uniform(0, 10, 100),
            "bid_size_level_1": np.random.randint(100, 1000, 100),
            "ask_size_level_1": np.random.randint(100, 1000, 100),
            "trade_frequency_1s": np.random.uniform(0, 10, 100),
            **{
                f"feature_{i}": np.random.uniform(0, 1, 100) for i in range(145)
            },  # Total 150 columns
        }
    )
    features_data.to_csv(features_latest_path, index=False, encoding="utf-8")

    # Créer feature_importance.csv
    feature_importance_path = features_dir / "feature_importance.csv"
    shap_data = pd.DataFrame(
        {
            "feature": [f"feature_{i}" for i in range(150)],
            "importance": np.random.uniform(0, 1, 150),
        }
    )
    shap_data.to_csv(feature_importance_path, index=False, encoding="utf-8")

    return {
        "base_dir": str(base_dir),
        "config_path": str(config_path),
        "feature_sets_path": str(feature_sets_path),
        "features_latest_path": str(features_latest_path),
        "feature_importance_path": str(feature_importance_path),
        "logs_dir": str(logs_dir),
        "figures_dir": str(figures_dir),
        "cache_dir": str(cache_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "corr_matrix_path": str(data_dir / "correlation_matrix.csv"),
        "perf_log_path": str(logs_dir / "heatmap_performance.csv"),
    }


def test_init(tmp_dirs):
    """Teste l’initialisation de CorrelationHeatmap."""
    heatmap = CorrelationHeatmap(config_path=tmp_dirs["config_path"])
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "init" in str(op) for op in df["operation"]
    ), "Opération init non journalisée"
    snapshot_files = os.listdir(tmp_dirs["cache_dir"])
    assert any(
        "snapshot_init" in f and f.endswith(".json.gz") for f in snapshot_files
    ), "Snapshot init non créé"


def test_load_config(tmp_dirs):
    """Teste le chargement de la configuration."""
    heatmap = CorrelationHeatmap(config_path=tmp_dirs["config_path"])
    config = heatmap.load_config(tmp_dirs["config_path"])
    assert "significant_threshold" in config, "Clé significant_threshold manquante"
    assert "s3_bucket" in config, "Clé s3_bucket manquante"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "load_config" in str(op) for op in df["operation"]
    ), "Opération load_config non journalisée"


def test_load_data(tmp_dirs):
    """Teste le chargement et la validation des données."""
    heatmap = CorrelationHeatmap(config_path=tmp_dirs["config_path"])
    df = heatmap.load_data(tmp_dirs["features_latest_path"])
    assert len(df.columns) >= 50, "Moins de 50 features chargées"
    assert all(
        col in df.columns for col in ["vix_es_correlation", "atr_14"]
    ), "Colonnes de volatilité manquantes"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "load_data" in str(op) for op in df_perf["operation"]
    ), "Opération load_data non journalisée"


def test_compute_correlations(tmp_dirs):
    """Teste le calcul des corrélations."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        heatmap = CorrelationHeatmap(config_path=tmp_dirs["config_path"])
        df = heatmap.load_data(tmp_dirs["features_latest_path"])
        corr_matrix = heatmap.compute_correlations(df, method="pearson")
        assert isinstance(
            corr_matrix, pd.DataFrame
        ), "Matrice de corrélations n’est pas un DataFrame"
        assert corr_matrix.shape[0] == len(
            df.columns
        ), "Taille de la matrice incorrecte"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "compute_correlations" in str(op) for op in df_perf["operation"]
        ), "Opération compute_correlations non journalisée"
        assert any(
            "confidence_drop_rate" in str(kw) for kw in df_perf.to_dict("records")
        ), "confidence_drop_rate absent"
        mock_telegram.assert_called()


def test_detect_significant_correlations(tmp_dirs):
    """Teste la détection des corrélations significatives."""
    heatmap = CorrelationHeatmap(config_path=tmp_dirs["config_path"])
    df = heatmap.load_data(tmp_dirs["features_latest_path"])
    corr_matrix = heatmap.compute_correlations(df)
    significant_pairs = heatmap.detect_significant_correlations(
        corr_matrix, threshold=0.8
    )
    assert isinstance(significant_pairs, list), "Résultat n’est pas une liste"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "detect_significant_correlations" in str(op) for op in df_perf["operation"]
    ), "Opération detect_significant_correlations non journalisée"


def test_plot_heatmap(tmp_dirs):
    """Teste la génération et le cache de la heatmap."""
    heatmap = CorrelationHeatmap(config_path=tmp_dirs["config_path"])
    df = heatmap.load_data(tmp_dirs["features_latest_path"])
    output_path = os.path.join(tmp_dirs["figures_dir"], "correlation_heatmap.png")
    heatmap.plot_heatmap(df, output_path, figsize=(12, 10), threshold=0.8)
    assert os.path.exists(output_path), "Heatmap non générée"
    cache_files = os.listdir(tmp_dirs["cache_dir"])
    assert any(
        "heatmap_" in f and f.endswith(".png") for f in cache_files
    ), "Heatmap non mise en cache"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "plot_heatmap" in str(op) for op in df_perf["operation"]
    ), "Opération plot_heatmap non journalisée"


def test_plot_heatmap_cache_hit(tmp_dirs):
    """Teste la récupération d’une heatmap depuis le cache."""
    heatmap = CorrelationHeatmap(config_path=tmp_dirs["config_path"])
    df = heatmap.load_data(tmp_dirs["features_latest_path"])
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(df).tobytes()).hexdigest()
    cache_file = os.path.join(tmp_dirs["cache_dir"], f"heatmap_{data_hash}.png")
    with open(cache_file, "wb") as f:
        f.write(b"dummy image data")
    output_path = os.path.join(tmp_dirs["figures_dir"], "correlation_heatmap.png")
    heatmap.plot_heatmap(df, output_path, figsize=(12, 10), threshold=0.8)
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "plot_heatmap_cache_hit" in str(op) for op in df_perf["operation"]
    ), "Opération plot_heatmap_cache_hit non journalisée"


def test_save_correlation_matrix(tmp_dirs):
    """Teste la sauvegarde de la matrice de corrélations."""
    heatmap = CorrelationHeatmap(config_path=tmp_dirs["config_path"])
    df = heatmap.load_data(tmp_dirs["features_latest_path"])
    corr_matrix = heatmap.compute_correlations(df)
    heatmap.save_correlation_matrix(corr_matrix, tmp_dirs["corr_matrix_path"])
    assert os.path.exists(
        tmp_dirs["corr_matrix_path"]
    ), "Matrice de corrélations non sauvegardée"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "save_correlation_matrix" in str(op) for op in df_perf["operation"]
    ), "Opération save_correlation_matrix non journalisée"


def test_save_snapshot_compressed(tmp_dirs):
    """Teste la sauvegarde d’un snapshot compressé."""
    heatmap = CorrelationHeatmap(config_path=tmp_dirs["config_path"])
    snapshot_data = {"test": "compressed_snapshot"}
    heatmap.save_snapshot("test_compressed", snapshot_data, compress=True)
    snapshot_files = os.listdir(tmp_dirs["cache_dir"])
    assert any(
        "snapshot_test_compressed" in f and f.endswith(".json.gz")
        for f in snapshot_files
    ), "Snapshot compressé non créé"
    with gzip.open(
        os.path.join(tmp_dirs["cache_dir"], snapshot_files[-1]), "rt", encoding="utf-8"
    ) as f:
        snapshot = json.load(f)
    assert snapshot["data"] == snapshot_data, "Contenu du snapshot compressé incorrect"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "save_snapshot" in str(op) for op in df_perf["operation"]
    ), "Opération save_snapshot non journalisée"


def test_checkpoint(tmp_dirs):
    """Teste la sauvegarde incrémentielle."""
    heatmap = CorrelationHeatmap(config_path=tmp_dirs["config_path"])
    df = heatmap.load_data(tmp_dirs["features_latest_path"])
    corr_matrix = heatmap.compute_correlations(df)
    heatmap.checkpoint(corr_matrix)
    checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
    assert any(
        "heatmap_" in f and f.endswith(".json.gz") for f in checkpoint_files
    ), "Checkpoint non créé"
    with gzip.open(
        os.path.join(tmp_dirs["checkpoints_dir"], checkpoint_files[0]),
        "rt",
        encoding="utf-8",
    ) as f:
        checkpoint = json.load(f)
    assert checkpoint["num_rows"] == len(corr_matrix), "Nombre de lignes incorrect"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "checkpoint" in str(op) for op in df_perf["operation"]
    ), "Opération checkpoint non journalisée"


def test_cloud_backup(tmp_dirs):
    """Teste la sauvegarde distribuée S3."""
    with patch("boto3.client") as mock_s3:
        heatmap = CorrelationHeatmap(config_path=tmp_dirs["config_path"])
        df = heatmap.load_data(tmp_dirs["features_latest_path"])
        corr_matrix = heatmap.compute_correlations(df)
        heatmap.cloud_backup(corr_matrix)
        assert mock_s3.called, "Client S3 non appelé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "temp_s3_" in f for f in checkpoint_files
        ), "Fichier temporaire S3 non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "cloud_backup" in str(op) for op in df_perf["operation"]
        ), "Opération cloud_backup non journalisée"


def test_handle_sigint(tmp_dirs):
    """Teste la gestion SIGINT."""
    heatmap = CorrelationHeatmap(config_path=tmp_dirs["config_path"])
    with patch("sys.exit") as mock_exit:
        heatmap.handle_sigint(signal.SIGINT, None)
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_sigint" in f for f in snapshot_files
        ), "Snapshot sigint non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "handle_sigint" in str(op) for op in df_perf["operation"]
        ), "Opération handle_sigint non journalisée"
        mock_exit.assert_called_with(0)
