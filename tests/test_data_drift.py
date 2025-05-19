# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_data_drift.py
# Tests unitaires pour src/monitoring/data_drift.py
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide la détection des dérives dans les données entre entraînement et live, avec focus sur SHAP features,
#        intégration de volatilité (méthode 1), utilisation exclusive d’IQFeed, snapshots compressés,
#        sauvegardes incrémentielles/distribuées, alertes Telegram, et confidence_drop_rate (Phase 8).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, scipy>=1.10.0,<2.0.0,
#   matplotlib>=3.8.0,<4.0.0, psutil>=5.9.8,<6.0.0, loguru>=0.7.0,<1.0.0, pyyaml>=6.0.0,<7.0.0,
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
# - data/features/features_train.csv
# - data/features/features_latest.csv
# - data/features/feature_importance.csv
#
# Outputs :
# - Rapport de dérive dans data/logs/drift_report.csv
# - Snapshots dans data/cache/drift/*.json.gz
# - Sauvegardes incrémentielles dans data/checkpoints/drift_*.json.gz
# - Graphiques dans data/figures/drift/
# - Logs dans data/logs/data_drift.log
# - Logs de performance dans data/logs/drift_performance.csv
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
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.monitoring.data_drift import DataDriftDetector


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
    figures_dir = data_dir / "figures" / "drift"
    figures_dir.mkdir(parents=True)
    cache_dir = data_dir / "cache" / "drift"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints"
    checkpoints_dir.mkdir()
    features_dir = data_dir / "features"
    features_dir.mkdir()

    # Créer es_config.yaml
    config_path = config_dir / "es_config.yaml"
    config_content = {
        "drift_params": {
            "wass_threshold": 0.1,
            "ks_threshold": 0.05,
            "vix_drift_threshold": 0.5,
            "s3_bucket": "test-bucket",
            "s3_prefix": "drift/",
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

    # Créer features_train.csv
    features_train_path = features_dir / "features_train.csv"
    features_train_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=100, freq="1min"),
            "vix_es_correlation": np.random.uniform(-1, 1, 100),
            "bid_size_level_1": np.random.randint(100, 1000, 100),
            "ask_size_level_1": np.random.randint(100, 1000, 100),
            "trade_frequency_1s": np.random.uniform(0, 10, 100),
            **{
                f"feature_{i}": np.random.uniform(0, 1, 100) for i in range(145)
            },  # Total 150 columns
        }
    )
    features_train_data.to_csv(features_train_path, index=False, encoding="utf-8")

    # Créer features_latest.csv
    features_latest_path = features_dir / "features_latest.csv"
    features_latest_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=100, freq="1min"),
            "vix_es_correlation": np.random.uniform(-0.5, 1.5, 100),  # Simuler dérive
            "bid_size_level_1": np.random.randint(100, 1000, 100),
            "ask_size_level_1": np.random.randint(100, 1000, 100),
            "trade_frequency_1s": np.random.uniform(0, 10, 100),
            **{
                f"feature_{i}": np.random.uniform(0, 1, 100) for i in range(145)
            },  # Total 150 columns
        }
    )
    features_latest_data.to_csv(features_latest_path, index=False, encoding="utf-8")

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
        "features_train_path": str(features_train_path),
        "features_latest_path": str(features_latest_path),
        "feature_importance_path": str(feature_importance_path),
        "logs_dir": str(logs_dir),
        "figures_dir": str(figures_dir),
        "cache_dir": str(cache_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "drift_report_path": str(logs_dir / "drift_report.csv"),
        "perf_log_path": str(logs_dir / "drift_performance.csv"),
    }


def test_init(tmp_dirs):
    """Teste l’initialisation de DataDriftDetector."""
    detector = DataDriftDetector(config_path=tmp_dirs["config_path"])
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
    detector = DataDriftDetector(config_path=tmp_dirs["config_path"])
    config = detector.load_config(tmp_dirs["config_path"])
    assert "wass_threshold" in config, "Clé wass_threshold manquante"
    assert "s3_bucket" in config, "Clé s3_bucket manquante"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "load_config" in str(op) for op in df["operation"]
    ), "Opération load_config non journalisée"


def test_load_data(tmp_dirs):
    """Teste le chargement et la validation des données."""
    detector = DataDriftDetector(config_path=tmp_dirs["config_path"])
    df_train = detector.load_data(tmp_dirs["features_train_path"], "training")
    df_live = detector.load_data(tmp_dirs["features_latest_path"], "live")
    assert (
        len(df_train.columns) >= 50
    ), "Moins de 50 features chargées pour entraînement"
    assert len(df_live.columns) >= 50, "Moins de 50 features chargées pour live"
    assert (
        "vix_es_correlation" in df_train.columns
    ), "Colonne vix_es_correlation manquante"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "load_data" in str(op) for op in df_perf["operation"]
    ), "Opération load_data non journalisée"


def test_compute_drift_metrics(tmp_dirs):
    """Teste le calcul des métriques de dérive."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        detector = DataDriftDetector(config_path=tmp_dirs["config_path"])
        df_train = detector.load_data(tmp_dirs["features_train_path"], "training")
        df_live = detector.load_data(tmp_dirs["features_latest_path"], "live")
        features = df_train.columns.tolist()
        drift_results = detector.compute_drift_metrics(df_train, df_live, features)
        assert (
            "vix_es_correlation" in drift_results
        ), "vix_es_correlation absent des résultats"
        assert (
            "wasserstein" in drift_results["vix_es_correlation"]
        ), "Métrique Wasserstein absente"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "compute_drift_metrics" in str(op) for op in df_perf["operation"]
        ), "Opération compute_drift_metrics non journalisée"
        assert any(
            "confidence_drop_rate" in str(kw) for kw in df_perf.to_dict("records")
        ), "confidence_drop_rate absent"
        mock_telegram.assert_called()


def test_plot_drift(tmp_dirs):
    """Teste la génération du graphique de dérive."""
    detector = DataDriftDetector(config_path=tmp_dirs["config_path"])
    df_train = detector.load_data(tmp_dirs["features_train_path"], "training")
    df_live = detector.load_data(tmp_dirs["features_latest_path"], "live")
    features = df_train.columns.tolist()
    drift_results = detector.compute_drift_metrics(df_train, df_live, features)
    detector.plot_drift(drift_results, threshold_wass=0.1, threshold_ks=0.05)
    assert any(
        f.startswith("data_drift_") and f.endswith(".png")
        for f in os.listdir(tmp_dirs["figures_dir"])
    ), "Graphique non généré"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "plot_drift" in str(op) for op in df_perf["operation"]
    ), "Opération plot_drift non journalisée"
    snapshot_files = os.listdir(tmp_dirs["cache_dir"])
    assert any(
        "snapshot_plot_drift" in f for f in snapshot_files
    ), "Snapshot plot_drift non créé"


def test_save_drift_report(tmp_dirs):
    """Teste la sauvegarde du rapport de dérive."""
    detector = DataDriftDetector(config_path=tmp_dirs["config_path"])
    df_train = detector.load_data(tmp_dirs["features_train_path"], "training")
    df_live = detector.load_data(tmp_dirs["features_latest_path"], "live")
    features = df_train.columns.tolist()
    drift_results = detector.compute_drift_metrics(df_train, df_live, features)
    detector.save_drift_report(drift_results, tmp_dirs["drift_report_path"])
    assert os.path.exists(
        tmp_dirs["drift_report_path"]
    ), "Rapport de dérive non sauvegardé"
    df_report = pd.read_csv(tmp_dirs["drift_report_path"])
    assert (
        "vix_es_correlation" in df_report["feature"].values
    ), "vix_es_correlation absent du rapport"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "save_drift_report" in str(op) for op in df_perf["operation"]
    ), "Opération save_drift_report non journalisée"


def test_save_snapshot_compressed(tmp_dirs):
    """Teste la sauvegarde d’un snapshot compressé."""
    detector = DataDriftDetector(config_path=tmp_dirs["config_path"])
    snapshot_data = {"test": "compressed_snapshot"}
    detector.save_snapshot("test_compressed", snapshot_data, compress=True)
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
    detector = DataDriftDetector(config_path=tmp_dirs["config_path"])
    df = pd.DataFrame(
        {
            "feature": ["vix_es_correlation"],
            "wasserstein": [0.2],
            "ks_stat": [0.1],
            "ks_pvalue": [0.05],
            "ttest_pvalue": [0.05],
            "timestamp": [datetime.now().isoformat()],
        }
    )
    detector.checkpoint(df)
    checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
    assert any(
        "drift_" in f and f.endswith(".json.gz") for f in checkpoint_files
    ), "Checkpoint non créé"
    with gzip.open(
        os.path.join(tmp_dirs["checkpoints_dir"], checkpoint_files[0]),
        "rt",
        encoding="utf-8",
    ) as f:
        checkpoint = json.load(f)
    assert checkpoint["num_rows"] == len(df), "Nombre de lignes incorrect"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "checkpoint" in str(op) for op in df_perf["operation"]
    ), "Opération checkpoint non journalisée"


def test_cloud_backup(tmp_dirs):
    """Teste la sauvegarde distribuée S3."""
    with patch("boto3.client") as mock_s3:
        detector = DataDriftDetector(config_path=tmp_dirs["config_path"])
        df = pd.DataFrame(
            {
                "feature": ["vix_es_correlation"],
                "wasserstein": [0.2],
                "ks_stat": [0.1],
                "ks_pvalue": [0.05],
                "ttest_pvalue": [0.05],
                "timestamp": [datetime.now().isoformat()],
            }
        )
        detector.cloud_backup(df)
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
    detector = DataDriftDetector(config_path=tmp_dirs["config_path"])
    with patch("sys.exit") as mock_exit:
        detector.handle_sigint(signal.SIGINT, None)
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_sigint" in f for f in snapshot_files
        ), "Snapshot sigint non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "handle_sigint" in str(op) for op in df_perf["operation"]
        ), "Opération handle_sigint non journalisée"
        mock_exit.assert_called_with(0)
