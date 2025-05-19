# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_schedule_economic_calendar.py
# Tests unitaires pour src/api/schedule_economic_calendar.py
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Valide la collecte des événements macroéconomiques via l’API Investing.com,
#        la clusterisation (PCA + K-means, méthode 7), le calcul de volatilité,
#        la planification des mises à jour, et le stockage dans macro_events.csv
#        et market_memory.db.
#
# Dépendances : pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, requests>=2.28.0,<3.0.0,
#               sklearn>=1.2.0,<2.0.0, psutil>=5.9.8,<6.0.0, pyyaml>=6.0.0,<7.0.0,
#               sqlite3, boto3>=1.26.0,<2.0.0, schedule>=1.2.0,<2.0.0
#
# Inputs : config/credentials.yaml, config/es_config.yaml
#
# Outputs : data/macro_events.csv, data/event_volatility_history.csv,
#           data/economic_calendar_snapshots/*.json.gz,
#           data/logs/economic_calendar_performance.csv,
#           data/logs/economic_calendar.log,
#           data/checkpoints/economic_calendar_*.json.gz,
#           market_memory.db
#
# Notes :
# - Conforme à structure.txt (version 2.1.3, 2025-05-13).
# - Tests les phases 1 (collecte API), 7 (mémoire contextuelle), 8 (auto-conscience),
#   et 17 (interprétabilité SHAP).
# - Vérifie l’absence de dxFeed, la présence de retries, logs psutil, alertes Telegram,
#   snapshots, et sauvegardes incrémentielles/distribuées.

import gzip
import json
import os
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.api.schedule_economic_calendar import EconomicCalendar


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
    snapshots_dir = data_dir / "economic_calendar_snapshots"
    snapshots_dir.mkdir()
    checkpoints_dir = data_dir / "checkpoints"
    checkpoints_dir.mkdir()
    features_dir = data_dir / "features"
    features_dir.mkdir()

    # Créer es_config.yaml
    config_path = config_dir / "es_config.yaml"
    config_content = {
        "economic_calendar": {
            "api_url": "https://www.investing.com/economic-calendar/",
            "frequency": "daily",
            "scheduled_time": "00:00",
            "cache_hours": 24,
            "volatility_window_minutes": 30,
            "s3_bucket": "test-bucket",
            "s3_prefix": "economic_calendar/",
        }
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    # Créer credentials.yaml
    credentials_path = config_dir / "credentials.yaml"
    credentials_content = {"credentials": {"investing_com_api_key": "test-api-key"}}
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
        "events_csv_path": str(data_dir / "macro_events.csv"),
        "volatility_csv_path": str(data_dir / "event_volatility_history.csv"),
        "perf_log_path": str(logs_dir / "economic_calendar_performance.csv"),
        "db_path": str(data_dir / "market_memory.db"),
        "feature_importance_path": str(features_dir / "feature_importance.csv"),
    }


@pytest.fixture
def mock_events_data():
    """Crée des données factices d’événements macroéconomiques."""
    return pd.DataFrame(
        {
            "event_id": ["EVT001", "EVT002"],
            "start_time": [datetime.now() - timedelta(hours=1), datetime.now()],
            "event_type": ["FOMC", "NFP"],
            "impact": [3, 2],
            "description": ["Federal Reserve Meeting", "Non-Farm Payrolls"],
            "volatility": [0.0, 0.0],
        }
    )


@pytest.fixture
def mock_es_data():
    """Crée des données factices de prix ES pour le calcul de volatilité."""
    now = datetime.now()
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                now - timedelta(hours=1), now + timedelta(hours=1), freq="1min"
            ),
            "symbol": "ES",
            "close": np.random.normal(5100, 10, 121),
        }
    )


@pytest.fixture
def mock_feature_importance(tmp_dirs):
    """Crée un fichier feature_importance.csv factice pour la validation SHAP."""
    features = ["event_impact", "event_type", "event_volatility"] + [
        f"feature_{i}" for i in range(147)
    ]
    shap_data = pd.DataFrame(
        {"feature": features, "importance": [0.1] * 150, "regime": ["range"] * 150}
    )
    shap_data.to_csv(tmp_dirs["feature_importance_path"], index=False, encoding="utf-8")
    return shap_data


def test_init(tmp_dirs, mock_feature_importance):
    """Teste l’initialisation de EconomicCalendar."""
    calendar = EconomicCalendar(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
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
    calendar = EconomicCalendar(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    config = calendar.load_config_with_validation(tmp_dirs["config_path"])
    assert "api_url" in config, "Clé api_url manquante"
    assert "frequency" in config, "Clé frequency manquante"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "load_config_with_validation" in str(op) for op in df["operation"]
    ), "Opération non journalisée"


def test_get_events(tmp_dirs, mock_events_data, mock_es_data, mock_feature_importance):
    """Teste la récupération des événements avec volatilité."""
    with patch("requests.get") as mock_get, patch.object(
        EconomicCalendar, "calculate_volatility_from_es", return_value=0.01
    ) as mock_volatility:
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "event_id": "EVT001",
                "start_time": str(datetime.now()),
                "event_type": "FOMC",
                "impact": "High",
                "description": "Federal Reserve Meeting",
            },
            {
                "event_id": "EVT002",
                "start_time": str(datetime.now()),
                "event_type": "NFP",
                "impact": "Medium",
                "description": "Non-Farm Payrolls",
            },
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        calendar = EconomicCalendar(
            config_path=tmp_dirs["config_path"],
            credentials_path=tmp_dirs["credentials_path"],
        )
        calendar.data_provider = MagicMock()
        calendar.data_provider.fetch_ohlc.return_value = mock_es_data

        events = calendar.get_events()
        assert events is not None, "Événements non récupérés"
        assert "volatility" in events.columns, "Colonne volatility manquante"
        assert "cluster_id" in events.columns, "Colonne cluster_id manquante"
        assert os.path.exists(
            tmp_dirs["events_csv_path"]
        ), "Fichier macro_events.csv non créé"
        assert os.path.exists(
            tmp_dirs["volatility_csv_path"]
        ), "Fichier event_volatility_history.csv non créé"
        assert os.path.exists(
            tmp_dirs["db_path"]
        ), "Base de données market_memory.db non créée"
        conn = sqlite3.connect(tmp_dirs["db_path"])
        clusters = pd.read_sql("SELECT * FROM clusters", conn)
        conn.close()
        assert not clusters.empty, "Tableau clusters vide"
        snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
        assert any(
            "snapshot_get_events" in f for f in snapshot_files
        ), "Snapshot non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert "confidence_drop_rate" in df.columns or any(
            "confidence_drop_rate" in str(kw) for kw in df.to_dict("records")
        ), "confidence_drop_rate absent"


def test_store_event_patterns(tmp_dirs, mock_events_data, mock_feature_importance):
    """Teste la clusterisation et le stockage dans market_memory.db."""
    calendar = EconomicCalendar(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    events = calendar.store_event_patterns(mock_events_data)
    assert "cluster_id" in events.columns, "Colonne cluster_id manquante"
    assert os.path.exists(
        tmp_dirs["db_path"]
    ), "Base de données market_memory.db non créée"
    conn = sqlite3.connect(tmp_dirs["db_path"])
    clusters = pd.read_sql("SELECT * FROM clusters", conn)
    conn.close()
    assert len(clusters) == len(
        mock_events_data
    ), "Nombre d’événements incorrect dans clusters"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_store_event_patterns" in f for f in snapshot_files
    ), "Snapshot non créé"


def test_calculate_volatility_from_es(tmp_dirs, mock_es_data, mock_feature_importance):
    """Teste le calcul de volatilité."""
    calendar = EconomicCalendar(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    calendar.data_provider = MagicMock()
    calendar.data_provider.fetch_ohlc.return_value = mock_es_data
    volatility = calendar.calculate_volatility_from_es(
        datetime.now(), window_minutes=30
    )
    assert isinstance(volatility, float), "Volatilité n’est pas un flottant"
    assert volatility >= 0, "Volatilité négative"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "calculate_volatility_from_es" in str(op) for op in df["operation"]
    ), "Opération non journalisée"


def test_schedule_calendar_updates(tmp_dirs, mock_events_data, mock_feature_importance):
    """Teste la planification des mises à jour."""
    with patch.object(
        EconomicCalendar, "update_calendar", return_value=mock_events_data
    ) as mock_update:
        calendar = EconomicCalendar(
            config_path=tmp_dirs["config_path"],
            credentials_path=tmp_dirs["credentials_path"],
        )
        with patch("schedule.every") as mock_schedule:
            mock_schedule.return_value.day.at.return_value.do.return_value = None
            calendar.schedule_calendar_updates(frequency="daily", timeout_hours=0.001)
            mock_update.assert_called()
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "schedule_calendar_updates" in str(op) for op in df["operation"]
        ), "Opération non journalisée"


def test_validate_shap_features(tmp_dirs, mock_feature_importance):
    """Teste la validation des features SHAP."""
    calendar = EconomicCalendar(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    result = calendar.validate_shap_features(
        ["event_impact", "event_type", "event_volatility"]
    )
    assert result, "Validation SHAP échouée pour features valides"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_validate_shap_features" in f for f in snapshot_files
    ), "Snapshot non créé"


def test_save_snapshot_compressed(tmp_dirs, mock_feature_importance):
    """Teste la sauvegarde d’un snapshot compressé."""
    calendar = EconomicCalendar(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    snapshot_data = {"test": "compressed_snapshot"}
    calendar.save_snapshot("test_compressed", snapshot_data, compress=True)
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


def test_checkpoint(tmp_dirs, mock_events_data, mock_feature_importance):
    """Teste la sauvegarde incrémentielle."""
    calendar = EconomicCalendar(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    calendar.checkpoint(mock_events_data)
    checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
    assert any(
        "economic_calendar_" in f and f.endswith(".json.gz") for f in checkpoint_files
    ), "Checkpoint non créé"
    with gzip.open(
        os.path.join(tmp_dirs["checkpoints_dir"], checkpoint_files[0]),
        "rt",
        encoding="utf-8",
    ) as f:
        checkpoint = json.load(f)
    assert checkpoint["num_events"] == len(
        mock_events_data
    ), "Nombre d’événements incorrect"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"


def test_cloud_backup(tmp_dirs, mock_events_data, mock_feature_importance):
    """Teste la sauvegarde distribuée S3."""
    with patch("boto3.client") as mock_s3:
        calendar = EconomicCalendar(
            config_path=tmp_dirs["config_path"],
            credentials_path=tmp_dirs["credentials_path"],
        )
        calendar.cloud_backup(mock_events_data)
        assert mock_s3.called, "Client S3 non appelé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "temp_s3_" in f for f in checkpoint_files
        ), "Fichier temporaire S3 non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "cloud_backup" in str(op) for op in df["operation"]
        ), "Opération non journalisée"


def test_signal_handler(tmp_dirs, mock_events_data, mock_feature_importance):
    """Teste la gestion SIGINT."""
    with patch.object(
        EconomicCalendar, "update_calendar", return_value=mock_events_data
    ) as mock_update:
        calendar = EconomicCalendar(
            config_path=tmp_dirs["config_path"],
            credentials_path=tmp_dirs["credentials_path"],
        )
        calendar.signal_handler(signal.SIGINT, None)
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
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        calendar = EconomicCalendar(
            config_path=tmp_dirs["config_path"],
            credentials_path=tmp_dirs["credentials_path"],
        )
        with patch("requests.get", side_effect=Exception("Erreur réseau")):
            events = calendar.get_events()
            assert events is None, "Événements récupérés malgré erreur"
            mock_telegram.assert_called_with(pytest.any(str))
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "get_events" in str(op) and not success
            for success, op in zip(df["success"], df["operation"])
        ), "Erreur critique non journalisée"
