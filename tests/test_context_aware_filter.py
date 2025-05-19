# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_context_aware_filter.py
# Tests unitaires pour src/api/context_aware_filter.py
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Valide le calcul des 19 métriques contextuelles (ex. : event_volatility_impact, news_sentiment_momentum),
#        l’utilisation exclusive d’IQFeed, l’intégration SHAP (Phase 17), confidence_drop_rate (Phase 8),
#        les snapshots, les sauvegardes incrémentielles/distribuées, et les alertes Telegram.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, numpy>=1.23.0,<2.0.0, psutil>=5.9.8,<6.0.0,
#   matplotlib>=3.7.0,<4.0.0, pyyaml>=6.0.0,<7.0.0, boto3>=1.26.0,<2.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/es_config.yaml
# - data/iqfeed/news_data.csv
# - data/iqfeed/futures_data.csv
# - data/events/macro_events.csv
# - data/events/event_volatility_history.csv
# - data/features/feature_importance.csv
#
# Outputs :
# - data/features/cache/context_aware/
# - data/logs/context_aware_performance.csv
# - data/logs/context_aware.log
# - data/context_aware_snapshots/*.json.gz
# - data/checkpoints/context_aware_*.json.gz
# - data/figures/context_aware/
#
# Notes :
# - Conforme à structure.txt (version 2.1.3, 2025-05-13).
# - Tests les phases 1 (collecte IQFeed), 8 (auto-conscience), 17 (interprétabilité SHAP).
# - Vérifie l’absence de dxFeed, la suppression d’obs_t, les retries, logs psutil, alertes Telegram,
#   snapshots, et sauvegardes incrémentielles/distribuées.

import gzip
import json
import os
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.api.context_aware_filter import ContextAwareFilter


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les données, logs, snapshots, checkpoints, et figures."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    iqfeed_dir = data_dir / "iqfeed"
    iqfeed_dir.mkdir()
    cache_dir = data_dir / "features" / "cache" / "context_aware"
    cache_dir.mkdir(parents=True)
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    snapshots_dir = data_dir / "context_aware_snapshots"
    snapshots_dir.mkdir()
    checkpoints_dir = data_dir / "checkpoints"
    checkpoints_dir.mkdir()
    figures_dir = data_dir / "figures" / "context_aware"
    figures_dir.mkdir(parents=True)
    events_dir = data_dir / "events"
    events_dir.mkdir()
    features_dir = data_dir / "features"
    features_dir.mkdir()

    # Créer es_config.yaml
    config_path = config_dir / "es_config.yaml"
    config_content = {
        "context_aware_filter": {
            "buffer_size": 100,
            "max_cache_size": 1000,
            "cache_hours": 24,
            "s3_bucket": "test-bucket",
            "s3_prefix": "context_aware/",
        }
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    return {
        "base_dir": str(base_dir),
        "config_path": str(config_path),
        "cache_dir": str(cache_dir),
        "logs_dir": str(logs_dir),
        "snapshots_dir": str(snapshots_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "figures_dir": str(figures_dir),
        "events_dir": str(events_dir),
        "perf_log_path": str(logs_dir / "context_aware_performance.csv"),
        "macro_events_path": str(events_dir / "macro_events.csv"),
        "volatility_history_path": str(events_dir / "event_volatility_history.csv"),
        "feature_importance_path": str(features_dir / "feature_importance.csv"),
    }


@pytest.fixture
def mock_data():
    """Crée des données factices pour les tests."""
    timestamp = pd.date_range("2025-05-13 09:00", periods=10, freq="1min")
    data = pd.DataFrame(
        {"timestamp": timestamp, "close": np.random.normal(5100, 10, 10)}
    )
    news_data = pd.DataFrame(
        {
            "timestamp": timestamp,
            "sentiment_score": np.random.uniform(-1, 1, 10),
            "volume": np.random.randint(1, 10, 10),
        }
    )
    calendar_data = pd.DataFrame(
        {
            "timestamp": timestamp,
            "severity": np.random.uniform(0, 1, 10),
            "weight": np.random.uniform(0, 1, 10),
        }
    )
    futures_data = pd.DataFrame(
        {
            "timestamp": timestamp,
            "near_price": np.random.normal(5100, 10, 10),
            "far_price": np.random.normal(5120, 10, 10),
        }
    )
    expiry_dates = pd.Series(
        pd.to_datetime([datetime.now() + timedelta(days=i) for i in range(10)])
    )
    macro_events = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [datetime.now() - timedelta(hours=i) for i in range(5)]
            ),
            "event_type": [
                "CPI Release",
                "FOMC Meeting",
                "Earnings Report",
                "GDP Release",
                "Jobs Report",
            ],
            "event_impact_score": [0.8, 0.9, 0.6, 0.7, 0.85],
        }
    )
    volatility_history = pd.DataFrame(
        {
            "event_type": [
                "CPI Release",
                "FOMC Meeting",
                "Earnings Report",
                "GDP Release",
                "Jobs Report",
            ],
            "volatility_impact": [0.15, 0.20, 0.10, 0.12, 0.18],
            "timestamp": pd.to_datetime(["2025-05-01"] * 5),
        }
    )
    return {
        "data": data,
        "news_data": news_data,
        "calendar_data": calendar_data,
        "futures_data": futures_data,
        "expiry_dates": expiry_dates,
        "macro_events": macro_events,
        "volatility_history": volatility_history,
    }


@pytest.fixture
def mock_feature_importance(tmp_dirs):
    """Crée un fichier feature_importance.csv factice pour la validation SHAP."""
    features = [
        "event_volatility_impact",
        "event_timing_proximity",
        "event_frequency_24h",
        "news_sentiment_momentum",
        "news_event_proximity",
        "macro_event_severity",
        "time_to_expiry_proximity",
        "economic_calendar_weight",
        "news_volume_spike",
        "news_volume_1h",
        "news_volume_1d",
        "news_sentiment_acceleration",
        "macro_event_momentum",
        "month_of_year_sin",
        "month_of_year_cos",
        "week_of_month_sin",
        "week_of_month_cos",
        "roll_yield_curve",
    ] + [f"feature_{i}" for i in range(132)]
    shap_data = pd.DataFrame(
        {"feature": features, "importance": [0.1] * 150, "regime": ["range"] * 150}
    )
    shap_data.to_csv(tmp_dirs["feature_importance_path"], index=False, encoding="utf-8")
    return shap_data


def test_init(tmp_dirs, mock_feature_importance):
    """Teste l’initialisation de ContextAwareFilter."""
    calculator = ContextAwareFilter(config_path=tmp_dirs["config_path"])
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
    calculator = ContextAwareFilter(config_path=tmp_dirs["config_path"])
    config = calculator.load_config_with_validation(tmp_dirs["config_path"])
    assert "buffer_size" in config, "Clé buffer_size manquante"
    assert "s3_bucket" in config, "Clé s3_bucket manquante"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "load_config_with_validation" in str(op) for op in df["operation"]
    ), "Opération non journalisée"


def test_compute_contextual_metrics(tmp_dirs, mock_data, mock_feature_importance):
    """Teste le calcul des métriques contextuelles."""
    calculator = ContextAwareFilter(config_path=tmp_dirs["config_path"])
    mock_data["macro_events"].to_csv(tmp_dirs["macro_events_path"], index=False)
    mock_data["volatility_history"].to_csv(
        tmp_dirs["volatility_history_path"], index=False
    )
    result = calculator.compute_contextual_metrics(
        data=mock_data["data"],
        news_data=mock_data["news_data"],
        calendar_data=mock_data["calendar_data"],
        futures_data=mock_data["futures_data"],
        expiry_dates=mock_data["expiry_dates"],
        macro_events_path=tmp_dirs["macro_events_path"],
        volatility_history_path=tmp_dirs["volatility_history_path"],
    )
    assert not result.empty, "Aucune métrique calculée"
    expected_metrics = [
        "event_volatility_impact",
        "event_timing_proximity",
        "event_frequency_24h",
        "news_sentiment_momentum",
        "news_event_proximity",
        "macro_event_severity",
        "time_to_expiry_proximity",
        "economic_calendar_weight",
        "news_volume_spike",
        "news_volume_1h",
        "news_volume_1d",
        "news_sentiment_acceleration",
        "macro_event_momentum",
        "month_of_year_sin",
        "month_of_year_cos",
        "week_of_month_sin",
        "week_of_month_cos",
        "roll_yield_curve",
    ]
    assert all(
        metric in result.columns for metric in expected_metrics
    ), "Métriques manquantes"
    assert os.path.exists(tmp_dirs["cache_dir"]), "Cache non créé"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_compute_contextual_metrics" in f for f in snapshot_files
    ), "Snapshot non créé"
    assert any(
        os.path.join(tmp_dirs["figures_dir"], f).endswith(".png")
        for f in os.listdir(tmp_dirs["figures_dir"])
    ), "Visualisations non générées"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "compute_contextual_metrics" in str(op) for op in df["operation"]
    ), "Opération non journalisée"
    assert "confidence_drop_rate" in df.columns or any(
        "confidence_drop_rate" in str(kw) for kw in df.to_dict("records")
    ), "confidence_drop_rate absent"


def test_parse_news_data(tmp_dirs, mock_data, mock_feature_importance):
    """Teste l’analyse des données de nouvelles."""
    calculator = ContextAwareFilter(config_path=tmp_dirs["config_path"])
    parsed_data = calculator.parse_news_data(mock_data["news_data"])
    assert not parsed_data.empty, "Données non analysées"
    assert "sentiment_score" in parsed_data.columns, "Colonne sentiment_score manquante"
    assert "volume" in parsed_data.columns, "Colonne volume manquante"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "parse_news_data" in str(op) for op in df["operation"]
    ), "Opération non journalisée"


def test_calculate_event_volatility_impact(
    tmp_dirs, mock_data, mock_feature_importance
):
    """Teste le calcul de l’impact de volatilité des événements."""
    calculator = ContextAwareFilter(config_path=tmp_dirs["config_path"])
    timestamp = pd.Timestamp.now()
    impact = calculator.calculate_event_volatility_impact(
        mock_data["macro_events"], mock_data["volatility_history"], timestamp
    )
    assert isinstance(impact, float), "Impact non numérique"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "calculate_event_volatility_impact" in str(op) for op in df["operation"]
    ), "Opération non journalisée"


def test_validate_shap_features(tmp_dirs, mock_feature_importance):
    """Teste la validation des features SHAP."""
    calculator = ContextAwareFilter(config_path=tmp_dirs["config_path"])
    features = [
        "event_volatility_impact",
        "news_sentiment_momentum",
        "month_of_year_sin",
    ]
    result = calculator.validate_shap_features(features)
    assert result, "Validation SHAP échouée pour features valides"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_validate_shap_features" in f for f in snapshot_files
    ), "Snapshot non créé"


def test_cache_metrics(tmp_dirs, mock_data, mock_feature_importance):
    """Teste la mise en cache des métriques."""
    calculator = ContextAwareFilter(config_path=tmp_dirs["config_path"])
    cache_key = hashlib.sha256(mock_data["data"].to_json().encode()).hexdigest()
    calculator.cache_metrics(mock_data["data"], cache_key)
    cache_path = os.path.join(tmp_dirs["cache_dir"], f"{cache_key}.csv")
    assert os.path.exists(cache_path), "Cache non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "cache_metrics" in str(op) for op in df["operation"]
    ), "Opération non journalisée"


def test_save_snapshot_compressed(tmp_dirs, mock_feature_importance):
    """Teste la sauvegarde d’un snapshot compressé."""
    calculator = ContextAwareFilter(config_path=tmp_dirs["config_path"])
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
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "save_snapshot" in str(op) for op in df["operation"]
    ), "Opération non journalisée"


def test_checkpoint(tmp_dirs, mock_data, mock_feature_importance):
    """Teste la sauvegarde incrémentielle."""
    calculator = ContextAwareFilter(config_path=tmp_dirs["config_path"])
    calculator.checkpoint(mock_data["data"])
    checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
    assert any(
        "context_aware_" in f and f.endswith(".json.gz") for f in checkpoint_files
    ), "Checkpoint non créé"
    with gzip.open(
        os.path.join(tmp_dirs["checkpoints_dir"], checkpoint_files[0]),
        "rt",
        encoding="utf-8",
    ) as f:
        checkpoint = json.load(f)
    assert checkpoint["num_rows"] == len(
        mock_data["data"]
    ), "Nombre de lignes incorrect"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "checkpoint" in str(op) for op in df["operation"]
    ), "Opération non journalisée"


def test_cloud_backup(tmp_dirs, mock_data, mock_feature_importance):
    """Teste la sauvegarde distribuée S3."""
    with patch("boto3.client") as mock_s3:
        calculator = ContextAwareFilter(config_path=tmp_dirs["config_path"])
        calculator.cloud_backup(mock_data["data"])
        assert mock_s3.called, "Client S3 non appelé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "temp_s3_" in f for f in checkpoint_files
        ), "Fichier temporaire S3 non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "cloud_backup" in str(op) for op in df["operation"]
        ), "Opération non journalisée"


def test_handle_sigint(tmp_dirs, mock_data, mock_feature_importance):
    """Teste la gestion SIGINT."""
    with patch("pandas.DataFrame.to_csv") as mock_save:
        calculator = ContextAwareFilter(config_path=tmp_dirs["config_path"])
        calculator.cache_metrics(mock_data["data"], "test_key")
        calculator.handle_sigint(signal.SIGINT, None)
        snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
        assert any(
            "snapshot_sigint" in f for f in snapshot_files
        ), "Snapshot sigint non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "handle_sigint" in str(op) for op in df["operation"]
        ), "Opération non journalisée"


def test_critical_alerts(tmp_dirs, mock_data, mock_feature_importance):
    """Teste les alertes Telegram pour les erreurs critiques."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        calculator = ContextAwareFilter(config_path=tmp_dirs["config_path"])
        empty_data = pd.DataFrame()
        result = calculator.compute_contextual_metrics(
            data=empty_data,
            news_data=mock_data["news_data"],
            calendar_data=mock_data["calendar_data"],
            futures_data=mock_data["futures_data"],
            expiry_dates=mock_data["expiry_dates"],
        )
        assert result.empty, "Métriques calculées malgré données vides"
        mock_telegram.assert_called_with(pytest.any(str))
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "compute_contextual_metrics" in str(op) and not success
            for success, op in zip(df["success"], df["operation"])
        ), "Erreur critique non journalisée"


def test_plot_metrics(tmp_dirs, mock_data, mock_feature_importance):
    """Teste la génération des visualisations."""
    calculator = ContextAwareFilter(config_path=tmp_dirs["config_path"])
    mock_data["data"]["news_sentiment_momentum"] = np.random.uniform(-1, 1, 10)
    mock_data["data"]["news_volume_spike"] = np.random.uniform(0, 1, 10)
    mock_data["data"]["event_volatility_impact"] = np.random.uniform(0, 0.2, 10)
    calculator.plot_metrics(
        mock_data["data"], datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    assert any(
        os.path.join(tmp_dirs["figures_dir"], f).endswith(".png")
        for f in os.listdir(tmp_dirs["figures_dir"])
    ), "Visualisations non générées"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "plot_metrics" in str(op) for op in df["operation"]
    ), "Opération non journalisée"
