# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_merge_data_sources.py
# Tests unitaires pour src/api/merge_data_sources.py
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Valide la fusion des données IQFeed (OHLC, options, nouvelles) dans merged_data.csv,
#        incluant la volatilité (méthode 1), les données d’options (méthode 2), la validation SHAP (Phase 17),
#        les snapshots, les sauvegardes, et les alertes Telegram.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, numpy>=1.23.0,<2.0.0, psutil>=5.9.8,<6.0.0,
#   pyyaml>=6.0.0,<7.0.0, boto3>=1.26.0,<2.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
# - src/utils/telegram_alert.py
# - src/api/data_provider.py
#
# Inputs :
# - config/iqfeed_config.yaml
# - config/credentials.yaml
#
# Outputs :
# - data/features/merged_data.csv
# - data/merge_snapshots/*.json.gz
# - data/logs/merge_performance.csv
# - data/logs/merge_data_sources.log
# - data/checkpoints/merge_data_*.json.gz
#
# Notes :
# - Conforme à structure.txt (version 2.1.3, 2025-05-13).
# - Tests les phases 1 (collecte IQFeed), 8 (auto-conscience), 17 (interprétabilité SHAP).
# - Vérifie l’absence de dxFeed, la présence de retries, logs psutil, alertes Telegram,
#   snapshots, et sauvegardes incrémentielles/distribuées.

import gzip
import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.api.merge_data_sources import DataMerger


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les données, logs, snapshots, et checkpoints."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    snapshots_dir = data_dir / "merge_snapshots"
    snapshots_dir.mkdir()
    checkpoints_dir = data_dir / "checkpoints"
    checkpoints_dir.mkdir()
    features_dir = data_dir / "features"
    features_dir.mkdir()

    # Créer iqfeed_config.yaml
    config_path = config_dir / "iqfeed_config.yaml"
    config_content = {
        "data_paths": {
            "chunk_size": 10000,
            "cache_hours": 24,
            "time_tolerance": "10s",
            "s3_bucket": "test-bucket",
            "s3_prefix": "merge_data/",
        }
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    # Créer credentials.yaml
    credentials_path = config_dir / "credentials.yaml"
    credentials_content = {"credentials": {"iqfeed_api_key": "test-api-key"}}
    with open(credentials_path, "w", encoding="utf-8") as f:
        yaml.dump(credentials_content, f)

    return {
        "base_dir": str(base_dir),
        "config_path": str(config_path),
        "credentials_path": str(credentials_path),
        "logs_dir": str(logs_dir),
        "snapshots_dir": str(snapshots_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "features_dir": str(features_dir),
        "output_path": str(features_dir / "merged_data.csv"),
        "perf_log_path": str(logs_dir / "merge_performance.csv"),
        "feature_importance_path": str(features_dir / "feature_importance.csv"),
    }


@pytest.fixture
def mock_ohlc_data():
    """Crée des données OHLC factices."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=10, freq="1min"),
            "open": np.random.normal(5100, 10, 10),
            "high": np.random.normal(5105, 10, 10),
            "low": np.random.normal(5095, 10, 10),
            "close": np.random.normal(5100, 10, 10),
            "volume": np.random.randint(1000, 5000, 10),
            "vix_es_correlation": np.random.uniform(-0.5, 0.5, 10),
            "atr_14": np.random.uniform(0.5, 2.0, 10),
        }
    )


@pytest.fixture
def mock_options_data():
    """Crée des données d’options factices."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=10, freq="1min"),
            "call_iv_atm": np.random.uniform(0.1, 0.2, 10),
            "put_iv_atm": np.random.uniform(0.1, 0.2, 10),
            "option_volume": np.random.randint(100, 1000, 10),
            "oi_concentration": np.random.uniform(0, 1, 10),
            "option_skew": np.random.uniform(0, 0.2, 10),
        }
    )


@pytest.fixture
def mock_news_data():
    """Crée des données de nouvelles factices."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=10, freq="1min"),
            "news_impact_score": np.random.uniform(0, 1, 10),
            "event": ["CPI Release"] * 5 + ["FOMC Meeting"] * 5,
            "description": ["Economic data release"] * 5
            + ["Federal Reserve decision"] * 5,
        }
    )


@pytest.fixture
def mock_feature_importance(tmp_dirs):
    """Crée un fichier feature_importance.csv factice pour la validation SHAP."""
    features = [
        "vix_es_correlation",
        "atr_14",
        "call_iv_atm",
        "put_iv_atm",
        "option_volume",
        "oi_concentration",
        "option_skew",
        "news_impact_score",
    ] + [f"feature_{i}" for i in range(142)]
    shap_data = pd.DataFrame(
        {"feature": features, "importance": [0.1] * 150, "regime": ["range"] * 150}
    )
    shap_data.to_csv(tmp_dirs["feature_importance_path"], index=False, encoding="utf-8")
    return shap_data


def test_init(tmp_dirs, mock_feature_importance):
    """Teste l’initialisation de DataMerger."""
    merger = DataMerger(config_path=tmp_dirs["config_path"])
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
        "snapshot_init" in f and f.endswith(".json.gz") for f in snapshot_files
    ), "Snapshot non créé"


def test_load_config_with_validation(tmp_dirs, mock_feature_importance):
    """Teste le chargement de la configuration."""
    merger = DataMerger(config_path=tmp_dirs["config_path"])
    config = merger.load_config_with_validation(tmp_dirs["config_path"])
    assert "chunk_size" in config, "Clé chunk_size manquante"
    assert "s3_bucket" in config, "Clé s3_bucket manquante"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "load_config_with_validation" in str(op) for op in df["operation"]
    ), "Opération non journalisée"


def test_merge_data(
    tmp_dirs, mock_ohlc_data, mock_options_data, mock_news_data, mock_feature_importance
):
    """Teste la fusion des données."""
    with patch(
        "src.api.data_provider.get_data_provider", return_value=MagicMock()
    ) as mock_provider:
        merger = DataMerger(config_path=tmp_dirs["config_path"])
        merged_data = merger.merge_data(
            mock_ohlc_data, mock_options_data, mock_news_data
        )
        assert merged_data is not None, "Fusion échouée"
        assert all(
            col in merged_data.columns
            for col in [
                "vix_es_correlation",
                "atr_14",
                "call_iv_atm",
                "put_iv_atm",
                "option_volume",
                "oi_concentration",
                "option_skew",
                "news_impact_score",
            ]
        ), "Colonnes requises manquantes"
        assert os.path.exists(
            tmp_dirs["output_path"]
        ), "Fichier merged_data.csv non créé"
        snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
        assert any(
            "snapshot_merge_data" in f for f in snapshot_files
        ), "Snapshot non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "merge_data" in str(op) for op in df["operation"]
        ), "Opération non journalisée"
        assert "confidence_drop_rate" in df.columns or any(
            "confidence_drop_rate" in str(kw) for kw in df.to_dict("records")
        ), "confidence_drop_rate absent"


def test_validate_shap_features(tmp_dirs, mock_feature_importance):
    """Teste la validation des features SHAP."""
    merger = DataMerger(config_path=tmp_dirs["config_path"])
    result = merger.validate_shap_features(
        [
            "vix_es_correlation",
            "atr_14",
            "call_iv_atm",
            "put_iv_atm",
            "option_volume",
            "oi_concentration",
            "option_skew",
            "news_impact_score",
        ]
    )
    assert result, "Validation SHAP échouée pour features valides"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_validate_shap_features" in f for f in snapshot_files
    ), "Snapshot non créé"


def test_save_snapshot_compressed(tmp_dirs, mock_feature_importance):
    """Teste la sauvegarde d’un snapshot compressé."""
    merger = DataMerger(config_path=tmp_dirs["config_path"])
    snapshot_data = {"test": "compressed_snapshot"}
    merger.save_snapshot("test_compressed", snapshot_data, compress=True)
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


def test_checkpoint(tmp_dirs, mock_ohlc_data, mock_feature_importance):
    """Teste la sauvegarde incrémentielle."""
    merger = DataMerger(config_path=tmp_dirs["config_path"])
    merger.checkpoint(mock_ohlc_data)
    checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
    assert any(
        "merge_data_" in f and f.endswith(".json.gz") for f in checkpoint_files
    ), "Checkpoint non créé"
    with gzip.open(
        os.path.join(tmp_dirs["checkpoints_dir"], checkpoint_files[0]),
        "rt",
        encoding="utf-8",
    ) as f:
        checkpoint = json.load(f)
    assert checkpoint["num_rows"] == len(mock_ohlc_data), "Nombre de lignes incorrect"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"


def test_cloud_backup(tmp_dirs, mock_ohlc_data, mock_feature_importance):
    """Teste la sauvegarde distribuée S3."""
    with patch("boto3.client") as mock_s3:
        merger = DataMerger(config_path=tmp_dirs["config_path"])
        merger.cloud_backup(mock_ohlc_data)
        assert mock_s3.called, "Client S3 non appelé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "temp_s3_" in f for f in checkpoint_files
        ), "Fichier temporaire S3 non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "cloud_backup" in str(op) for op in df["operation"]
        ), "Opération non journalisée"


def test_signal_handler(tmp_dirs, mock_ohlc_data, mock_feature_importance):
    """Teste la gestion SIGINT."""
    with patch("pandas.read_csv", return_value=mock_ohlc_data) as mock_read:
        merger = DataMerger(config_path=tmp_dirs["config_path"])
        merger.signal_handler(signal.SIGINT, None)
        snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
        assert any(
            "snapshot_shutdown" in f for f in snapshot_files
        ), "Snapshot shutdown non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "signal_handler" in str(op) for op in df["operation"]
        ), "Opération non journalisée"


def test_critical_alerts(tmp_dirs, mock_ohlc_data, mock_feature_importance):
    """Teste les alertes Telegram pour les erreurs critiques."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        merger = DataMerger(config_path=tmp_dirs["config_path"])
        with patch.object(
            merger.data_provider, "fetch_ohlc", side_effect=Exception("Erreur réseau")
        ):
            merged_data = merger.merge_data(
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            )
            assert merged_data is None, "Données fusionnées malgré erreur"
            mock_telegram.assert_called_with(pytest.any(str))
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "merge_data" in str(op) and not success
            for success, op in zip(df["success"], df["operation"])
        ), "Erreur critique non journalisée"
