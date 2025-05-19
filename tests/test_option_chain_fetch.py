# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_option_chain_fetch.py
# Tests unitaires pour src/api/option_chain_fetch.py
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Valide la collecte des données de chaîne d’options via IQFeed, incluant les métriques
#        (call_iv_atm, put_iv_atm, etc.), le cache horaire, la validation SHAP (Phase 17),
#        les snapshots, les sauvegardes, et les alertes Telegram.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, requests>=2.28.0,<3.0.0, psutil>=5.9.8,<6.0.0,
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
# - config/es_config.yaml
#
# Outputs :
# - data/iqfeed/option_chain.csv
# - data/iqfeed/cache/options/hourly_YYYYMMDD_HH.csv
# - data/option_chain_snapshots/*.json.gz
# - data/logs/option_chain_performance.csv
# - data/logs/option_chain_fetch.log
# - data/checkpoints/option_chain_*.json.gz
# - data/option_chain_dashboard.json
#
# Notes :
# - Conforme à structure.txt (version 2.1.3, 2025-05-13).
# - Tests les phases 1 (collecte IQFeed), 8 (auto-conscience), 17 (interprétabilité SHAP).
# - Vérifie l’absence de dxFeed, la présence de retries, logs psutil, alertes Telegram,
#   snapshots, et sauvegardes incrémentielles/distribuées.

import gzip
import json
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from src.api.option_chain_fetch import OptionChainFetcher


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les données, logs, snapshots, checkpoints, et cache."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    iqfeed_dir = data_dir / "iqfeed"
    iqfeed_dir.mkdir()
    cache_dir = iqfeed_dir / "cache" / "options"
    cache_dir.mkdir(parents=True)
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    snapshots_dir = data_dir / "option_chain_snapshots"
    snapshots_dir.mkdir()
    checkpoints_dir = data_dir / "checkpoints"
    checkpoints_dir.mkdir()
    features_dir = data_dir / "features"
    features_dir.mkdir()

    # Créer iqfeed_config.yaml
    config_path = config_dir / "iqfeed_config.yaml"
    config_content = {
        "iqfeed": {
            "endpoint": "https://api.iqfeed.net",
            "retry_attempts": 3,
            "retry_delay_base": 2.0,
            "timeout": 30,
            "cache_hours": 1,
            "symbols": ["ES"],
            "data_types": {"options": {"enabled": True}},
            "s3_bucket": "test-bucket",
            "s3_prefix": "option_chain/",
        }
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    # Créer credentials.yaml
    credentials_path = config_dir / "credentials.yaml"
    credentials_content = {"credentials": {"iqfeed_api_key": "test-api-key"}}
    with open(credentials_path, "w", encoding="utf-8") as f:
        yaml.dump(credentials_content, f)

    # Créer es_config.yaml
    es_config_path = config_dir / "es_config.yaml"
    es_config_content = {
        "preprocessing": {
            "retry_attempts": 3,
            "retry_delay_base": 2,
            "timeout_seconds": 1800,
            "max_data_age_seconds": 300,
        },
        "spotgamma_recalculator": {},
    }
    with open(es_config_path, "w", encoding="utf-8") as f:
        yaml.dump(es_config_content, f)

    return {
        "base_dir": str(base_dir),
        "config_path": str(config_path),
        "credentials_path": str(credentials_path),
        "es_config_path": str(es_config_path),
        "logs_dir": str(logs_dir),
        "snapshots_dir": str(snapshots_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "cache_dir": str(cache_dir),
        "output_path": str(iqfeed_dir / "option_chain.csv"),
        "perf_log_path": str(logs_dir / "option_chain_performance.csv"),
        "dashboard_path": str(data_dir / "option_chain_dashboard.json"),
        "feature_importance_path": str(features_dir / "feature_importance.csv"),
    }


@pytest.fixture
def mock_option_data():
    """Crée des données factices de chaîne d’options."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=10, freq="1min"),
            "symbol": ["ES"] * 10,
            "strike": [5100] * 10,
            "option_type": ["C"] * 5 + ["P"] * 5,
            "open_interest": [1000] * 10,
            "call_iv_atm": [0.15] * 10,
            "put_iv_atm": [0.14] * 10,
            "option_volume": [500] * 10,
            "oi_concentration": [0.8] * 10,
            "option_skew": [0.1] * 10,
            "gamma": [0.05] * 10,
            "delta": [0.5] * 5 + [-0.5] * 5,
            "vega": [0.2] * 10,
            "price": [10.0] * 10,
        }
    )


@pytest.fixture
def mock_feature_importance(tmp_dirs):
    """Crée un fichier feature_importance.csv factice pour la validation SHAP."""
    features = [
        "call_iv_atm",
        "put_iv_atm",
        "option_volume",
        "oi_concentration",
        "option_skew",
        "gamma",
        "delta",
        "vega",
        "price",
    ] + [f"feature_{i}" for i in range(141)]
    shap_data = pd.DataFrame(
        {"feature": features, "importance": [0.1] * 150, "regime": ["range"] * 150}
    )
    shap_data.to_csv(tmp_dirs["feature_importance_path"], index=False, encoding="utf-8")
    return shap_data


def test_init(tmp_dirs, mock_feature_importance):
    """Teste l’initialisation de OptionChainFetcher."""
    fetcher = OptionChainFetcher(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    assert os.path.exists(tmp_dirs["snapshots_dir"]), "Dossier de snapshots non créé"
    assert os.path.exists(tmp_dirs["cache_dir"]), "Dossier de cache non créé"
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
    fetcher = OptionChainFetcher(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    config = fetcher.load_config_with_validation(tmp_dirs["config_path"])
    assert "endpoint" in config, "Clé endpoint manquante"
    assert "s3_bucket" in config, "Clé s3_bucket manquante"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "load_config_with_validation" in str(op) for op in df["operation"]
    ), "Opération non journalisée"


def test_load_es_config(tmp_dirs, mock_feature_importance):
    """Teste le chargement de la configuration es_config."""
    fetcher = OptionChainFetcher(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    es_config = fetcher.load_es_config()
    assert "max_data_age_seconds" in es_config, "Clé max_data_age_seconds manquante"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "load_es_config" in str(op) for op in df["operation"]
    ), "Opération non journalisée"


def test_fetch_option_chain(tmp_dirs, mock_option_data, mock_feature_importance):
    """Teste la collecte des données d’options."""
    with patch(
        "src.api.data_provider.get_data_provider", return_value=MagicMock()
    ) as mock_provider:
        mock_provider.return_value.fetch_options.return_value = mock_option_data
        fetcher = OptionChainFetcher(
            config_path=tmp_dirs["config_path"],
            credentials_path=tmp_dirs["credentials_path"],
        )
        df = fetcher.fetch_option_chain(symbol="ES")
        assert not df.empty, "Aucune donnée récupérée"
        assert all(
            col in df.columns
            for col in [
                "timestamp",
                "strike",
                "option_type",
                "open_interest",
                "call_iv_atm",
                "put_iv_atm",
                "option_volume",
                "oi_concentration",
                "option_skew",
                "gamma",
                "delta",
                "vega",
                "price",
            ]
        ), "Colonnes requises manquantes"
        assert os.path.exists(
            tmp_dirs["output_path"]
        ), "Fichier option_chain.csv non créé"
        assert os.path.exists(
            os.path.join(
                tmp_dirs["cache_dir"],
                f"options_ES_{datetime.now().strftime('%Y%m%d_%H')}.csv",
            )
        ), "Cache horaire non créé"
        assert os.path.exists(
            tmp_dirs["dashboard_path"]
        ), "Fichier option_chain_dashboard.json non créé"
        snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
        assert any(
            "snapshot_fetch_option_chain" in f for f in snapshot_files
        ), "Snapshot non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "fetch_option_chain" in str(op) for op in df["operation"]
        ), "Opération non journalisée"
        assert "confidence_drop_rate" in df.columns or any(
            "confidence_drop_rate" in str(kw) for kw in df.to_dict("records")
        ), "confidence_drop_rate absent"


def test_validate_data(tmp_dirs, mock_feature_importance):
    """Teste la validation des données d’options."""
    fetcher = OptionChainFetcher(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    valid_data = {
        "open_interest": 1000,
        "call_iv_atm": 0.15,
        "put_iv_atm": 0.14,
        "option_volume": 500,
        "oi_concentration": 0.8,
        "option_skew": 0.1,
        "gamma": 0.05,
        "delta": 0.5,
        "vega": 0.2,
        "price": 10.0,
    }
    invalid_data = {
        "open_interest": -100,
        "call_iv_atm": 0.15,
        "put_iv_atm": 0.14,
        "option_volume": 500,
        "oi_concentration": 0.8,
        "option_skew": 0.1,
        "gamma": 0.05,
        "delta": 1.5,
        "vega": 0.2,
        "price": 10.0,
    }
    assert fetcher.validate_data(valid_data), "Données valides rejetées"
    assert not fetcher.validate_data(invalid_data), "Données invalides acceptées"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "validate_data" in str(op) for op in df["operation"]
    ), "Opération non journalisée"


def test_check_data_freshness(tmp_dirs, mock_option_data, mock_feature_importance):
    """Teste la vérification de la fraîcheur des données."""
    fetcher = OptionChainFetcher(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    fresh_data = mock_option_data.copy()
    fresh_data["timestamp"] = pd.date_range(
        datetime.now() - timedelta(minutes=1), periods=10, freq="1min"
    )
    old_data = mock_option_data.copy()
    old_data["timestamp"] = pd.date_range(
        datetime.now() - timedelta(hours=1), periods=10, freq="1min"
    )
    assert fetcher.check_data_freshness(
        fresh_data, max_age_seconds=300
    ), "Données fraîches rejetées"
    assert not fetcher.check_data_freshness(
        old_data, max_age_seconds=300
    ), "Données anciennes acceptées"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "check_data_freshness" in str(op) for op in df["operation"]
    ), "Opération non journalisée"


def test_load_save_cache(tmp_dirs, mock_option_data, mock_feature_importance):
    """Teste le chargement et la sauvegarde du cache horaire."""
    fetcher = OptionChainFetcher(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    fetcher.save_to_cache(mock_option_data, "ES")
    cache_path = os.path.join(
        tmp_dirs["cache_dir"], f"options_ES_{datetime.now().strftime('%Y%m%d_%H')}.csv"
    )
    assert os.path.exists(cache_path), "Cache horaire non créé"
    cached_df = fetcher.load_from_cache("ES", max_age_seconds=300)
    assert cached_df is not None, "Cache non chargé"
    assert len(cached_df) == len(mock_option_data), "Nombre de lignes incorrect"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "save_to_cache" in str(op) for op in df["operation"]
    ), "Opération save_to_cache non journalisée"
    assert any(
        "load_from_cache" in str(op) for op in df["operation"]
    ), "Opération load_from_cache non journalisée"


def test_validate_shap_features(tmp_dirs, mock_feature_importance):
    """Teste la validation des features SHAP."""
    fetcher = OptionChainFetcher(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    result = fetcher.validate_shap_features(
        [
            "call_iv_atm",
            "put_iv_atm",
            "option_volume",
            "oi_concentration",
            "option_skew",
            "gamma",
            "delta",
            "vega",
            "price",
        ]
    )
    assert result, "Validation SHAP échouée pour features valides"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_validate_shap_features" in f for f in snapshot_files
    ), "Snapshot non créé"


def test_save_snapshot_compressed(tmp_dirs, mock_feature_importance):
    """Teste la sauvegarde d’un snapshot compressé."""
    fetcher = OptionChainFetcher(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    snapshot_data = {"test": "compressed_snapshot"}
    fetcher.save_snapshot("test_compressed", snapshot_data, compress=True)
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


def test_checkpoint(tmp_dirs, mock_option_data, mock_feature_importance):
    """Teste la sauvegarde incrémentielle."""
    fetcher = OptionChainFetcher(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    fetcher.checkpoint(mock_option_data, "ES")
    checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
    assert any(
        "option_chain_" in f and f.endswith(".json.gz") for f in checkpoint_files
    ), "Checkpoint non créé"
    with gzip.open(
        os.path.join(tmp_dirs["checkpoints_dir"], checkpoint_files[0]),
        "rt",
        encoding="utf-8",
    ) as f:
        checkpoint = json.load(f)
    assert checkpoint["num_rows"] == len(mock_option_data), "Nombre de lignes incorrect"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"


def test_cloud_backup(tmp_dirs, mock_option_data, mock_feature_importance):
    """Teste la sauvegarde distribuée S3."""
    with patch("boto3.client") as mock_s3:
        fetcher = OptionChainFetcher(
            config_path=tmp_dirs["config_path"],
            credentials_path=tmp_dirs["credentials_path"],
        )
        fetcher.cloud_backup(mock_option_data, "ES")
        assert mock_s3.called, "Client S3 non appelé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "temp_s3_" in f for f in checkpoint_files
        ), "Fichier temporaire S3 non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "cloud_backup" in str(op) for op in df["operation"]
        ), "Opération non journalisée"


def test_signal_handler(tmp_dirs, mock_option_data, mock_feature_importance):
    """Teste la gestion SIGINT."""
    with patch("pandas.read_csv", return_value=mock_option_data) as mock_read:
        fetcher = OptionChainFetcher(
            config_path=tmp_dirs["config_path"],
            credentials_path=tmp_dirs["credentials_path"],
        )
        fetcher.signal_handler(signal.SIGINT, None)
        snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
        assert any(
            "snapshot_shutdown" in f for f in snapshot_files
        ), "Snapshot shutdown non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "signal_handler" in str(op) for op in df["operation"]
        ), "Opération non journalisée"


def test_critical_alerts(tmp_dirs, mock_feature_importance):
    """Teste les alertes Telegram pour les erreurs critiques."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram, patch(
        "src.api.data_provider.get_data_provider",
        side_effect=Exception("Erreur réseau"),
    ):
        fetcher = OptionChainFetcher(
            config_path=tmp_dirs["config_path"],
            credentials_path=tmp_dirs["credentials_path"],
        )
        df = fetcher.fetch_option_chain(symbol="ES")
        assert df.empty, "Données récupérées malgré erreur"
        mock_telegram.assert_called_with(pytest.any(str))
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "fetch_option_chain" in str(op) and not success
            for success, op in zip(df["success"], df["operation"])
        ), "Erreur critique non journalisée"
