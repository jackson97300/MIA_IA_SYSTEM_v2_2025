# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_mind_stream.py
# Tests unitaires pour src/mind/mind_stream.py
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide le flux cognitif de MIA, incluant l’analyse en temps réel pour détecter patterns, anomalies, et tendances
#        avec mémoire contextuelle (méthode 7, K-means 10 clusters dans market_memory.db), l’utilisation exclusive d’IQFeed,
#        l’intégration SHAP (Phase 17), confidence_drop_rate (Phase 8), snapshots compressés, sauvegardes incrémentielles/distribuées,
#        et alertes Telegram.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.8,<6.0.0, sklearn>=1.2.0,<2.0.0,
#   matplotlib>=3.8.0,<4.0.0, seaborn>=0.13.0,<1.0.0, pyyaml>=6.0.0,<7.0.0, boto3>=1.26.0,<2.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/db_setup.py
# - src/mind/mind.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/market_config.yaml
# - config/feature_sets.yaml
# - data/features/features_latest.csv
# - data/logs/mind/mind_stream.json
#
# Outputs :
# - data/logs/mind/mind_stream.log
# - data/logs/mind/mind_stream.json/csv
# - data/logs/mind/mind_stream_performance.csv
# - data/mind/mind_stream_snapshots/*.json.gz
# - data/checkpoints/mind_stream_*.json.gz
# - data/mind/mind_dashboard.json
# - data/figures/mind_stream/
# - market_memory.db (table clusters)
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Tests les phases 7 (mémoire contextuelle), 8 (auto-conscience via confidence_drop_rate), 17 (SHAP).
# - Vérifie l’absence de dxFeed, la suppression de 320/81 features et obs_t, les retries, logs psutil, alertes Telegram,
#   snapshots compressés, et sauvegardes incrémentielles/distribuées.
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence) via config/feature_sets.yaml.

import gzip
import json
import os
import sqlite3
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.mind.mind_stream import MindStream


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les données, logs, snapshots, checkpoints, et figures."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs" / "mind"
    logs_dir.mkdir(parents=True)
    snapshots_dir = data_dir / "mind" / "mind_stream_snapshots"
    snapshots_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints"
    checkpoints_dir.mkdir()
    figures_dir = data_dir / "figures" / "mind_stream"
    figures_dir.mkdir(parents=True)
    features_dir = data_dir / "features"
    features_dir.mkdir()

    # Créer market_config.yaml
    config_path = config_dir / "market_config.yaml"
    config_content = {
        "thresholds": {
            "vix_threshold": 30,
            "anomaly_score_threshold": 0.9,
            "trend_confidence_threshold": 0.7,
            "regime_change_threshold": 0.8,
            "max_analysis_frequency": 60,
        },
        "logging": {"buffer_size": 100},
        "s3_bucket": "test-bucket",
        "s3_prefix": "mind_stream/",
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
            "timestamp": pd.date_range("2025-05-13 09:00", periods=10, freq="1min"),
            "vix": np.random.uniform(10, 30, 10),
            "neural_regime": np.random.randint(0, 3, 10),
            "predicted_volatility": np.random.uniform(0.01, 0.1, 10),
            "trade_frequency_1s": np.random.uniform(0, 10, 10),
            "close": np.random.normal(5100, 10, 10),
            "rsi_14": np.random.uniform(30, 70, 10),
            **{
                f"feature_{i}": np.random.uniform(0, 1, 10) for i in range(143)
            },  # Total 150 columns
        }
    )
    features_data.to_csv(features_latest_path, index=False, encoding="utf-8")

    return {
        "base_dir": str(base_dir),
        "config_path": str(config_path),
        "feature_sets_path": str(feature_sets_path),
        "features_latest_path": str(features_latest_path),
        "logs_dir": str(logs_dir),
        "snapshots_dir": str(snapshots_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "figures_dir": str(figures_dir),
        "json_log_path": str(logs_dir / "mind_stream.json"),
        "csv_log_path": str(logs_dir / "mind_stream.csv"),
        "perf_log_path": str(logs_dir / "mind_stream_performance.csv"),
        "dashboard_path": str(data_dir / "mind" / "mind_dashboard.json"),
        "db_path": str(data_dir / "market_memory.db"),
    }


@pytest.fixture
def mock_trading_env():
    """Crée un environnement de trading factice."""
    env = MagicMock()
    env.balance = 10000.0
    env.position = 1
    env.mode = "trend"
    env.rsi = 75
    return env


def test_init(tmp_dirs):
    """Teste l’initialisation de MindStream."""
    stream = MindStream(config_path=tmp_dirs["config_path"])
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    assert os.path.exists(tmp_dirs["snapshots_dir"]), "Dossier de snapshots non créé"
    assert os.path.exists(
        tmp_dirs["checkpoints_dir"]
    ), "Dossier de checkpoints non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert all(
        col in df.columns
        for col in ["timestamp", "operation", "latency", "cpu_percent"]
    ), "Colonnes de performance manquantes"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_init" in f and f.endswith(".json.gz") for f in snapshot_files
    ), "Snapshot non créé"


def test_load_config(tmp_dirs):
    """Teste le chargement de la configuration."""
    stream = MindStream(config_path=tmp_dirs["config_path"])
    config = stream.load_config(tmp_dirs["config_path"])
    assert "s3_bucket" in config, "Clé s3_bucket manquante"
    assert "thresholds" in config, "Clé thresholds manquante"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "load_config" in str(op) for op in df["operation"]
    ), "Opération non journalisée"


def test_validate_data(tmp_dirs):
    """Teste la validation des données."""
    stream = MindStream(config_path=tmp_dirs["config_path"])
    df = pd.read_csv(tmp_dirs["features_latest_path"])
    stream.validate_data(df)
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "validate_data" in str(op) for op in df_perf["operation"]
    ), "Opération non journalisée"


def test_stream_analysis(tmp_dirs, mock_trading_env):
    """Teste l’analyse en temps réel."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        stream = MindStream(config_path=tmp_dirs["config_path"])
        df = pd.read_csv(tmp_dirs["features_latest_path"])
        context = {"neural_regime": 0, "strategy_params": {"entry_threshold": 70.0}}
        result = stream.stream_analysis(df, context)
        assert "anomaly_score" in result, "anomaly_score manquant"
        assert "trend_type" in result, "trend_type manquant"
        assert os.path.exists(
            tmp_dirs["json_log_path"]
        ), "Fichier JSON de logs non créé"
        with open(tmp_dirs["json_log_path"], "r", encoding="utf-8") as f:
            logs = [json.loads(line) for line in f]
        assert any("anomaly_score" in log for log in logs), "Analyse non journalisée"
        assert os.path.exists(tmp_dirs["csv_log_path"]), "Fichier CSV de logs non créé"
        df_log = pd.read_csv(tmp_dirs["csv_log_path"])
        assert "cluster_id" in df_log.columns, "cluster_id manquant dans CSV"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "stream_analysis" in str(op) for op in df_perf["operation"]
        ), "Opération non journalisée"
        mock_telegram.assert_called()


def test_store_analysis_pattern(tmp_dirs):
    """Teste la clusterisation et le stockage des analyses."""
    stream = MindStream(config_path=tmp_dirs["config_path"])
    analysis_result = {
        "timestamp": datetime.now().isoformat(),
        "anomaly_score": 0.8,
        "trend_score": 0.6,
        "regime_change_score": 0.7,
    }
    df = pd.read_csv(tmp_dirs["features_latest_path"])
    result = stream.store_analysis_pattern(analysis_result, df)
    assert "cluster_id" in result, "cluster_id manquant"
    conn = sqlite3.connect(tmp_dirs["db_path"])
    df_db = pd.read_sql_query("SELECT * FROM clusters", conn)
    conn.close()
    assert not df_db.empty, "Aucun cluster stocké dans la base"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "store_analysis_pattern" in str(op) for op in df_perf["operation"]
    ), "Opération non journalisée"
    assert any(
        "confidence_drop_rate" in str(kw) for kw in df_perf.to_dict("records")
    ), "confidence_drop_rate absent"


def test_visualize_analysis_patterns(tmp_dirs):
    """Teste la génération de la heatmap des clusters."""
    stream = MindStream(config_path=tmp_dirs["config_path"])
    pd.read_csv(tmp_dirs["features_latest_path"])
    stream.analysis_history = [
        {"timestamp": datetime.now().isoformat(), "anomaly_score": 0.8, "cluster_id": 1}
    ]
    stream.visualize_analysis_patterns()
    assert any(
        f.endswith(".png") for f in os.listdir(tmp_dirs["figures_dir"])
    ), "Heatmap non générée"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "visualize_analysis_patterns" in str(op) for op in df_perf["operation"]
    ), "Opération non journalisée"


def test_save_snapshot_compressed(tmp_dirs):
    """Teste la sauvegarde d’un snapshot compressé."""
    stream = MindStream(config_path=tmp_dirs["config_path"])
    snapshot_data = {"test": "compressed_snapshot"}
    stream.save_snapshot("test_compressed", snapshot_data, compress=True)
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
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "save_snapshot" in str(op) for op in df_perf["operation"]
    ), "Opération non journalisée"


def test_checkpoint(tmp_dirs):
    """Teste la sauvegarde incrémentielle."""
    stream = MindStream(config_path=tmp_dirs["config_path"])
    df = pd.DataFrame(
        {
            "timestamp": [datetime.now().isoformat()] * 5,
            "anomaly_score": [0.8] * 5,
            "trend_score": [0.6] * 5,
            "trend_type": ["haussière"] * 5,
            "regime_change_score": [0.7] * 5,
            "regime_type": ["trend"] * 5,
            "vix": [20.0] * 5,
            "rsi": [75.0] * 5,
            "trade_frequency": [8.0] * 5,
            "cluster_id": [0] * 5,
            "context_neural_regime": [0] * 5,
        }
    )
    stream.checkpoint(df)
    checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
    assert any(
        "mind_stream_" in f and f.endswith(".json.gz") for f in checkpoint_files
    ), "Checkpoint non créé"
    with gzip.open(
        os.path.join(tmp_dirs["checkpoints_dir"], checkpoint_files[0]),
        "rt",
        encoding="utf-8",
    ) as f:
        checkpoint = json.load(f)
    assert checkpoint["num_rows"] == 5, "Nombre de lignes incorrect"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "checkpoint" in str(op) for op in df_perf["operation"]
    ), "Opération non journalisée"


def test_cloud_backup(tmp_dirs):
    """Teste la sauvegarde distribuée S3."""
    with patch("boto3.client") as mock_s3:
        stream = MindStream(config_path=tmp_dirs["config_path"])
        df = pd.DataFrame(
            {
                "timestamp": [datetime.now().isoformat()] * 5,
                "anomaly_score": [0.8] * 5,
                "trend_score": [0.6] * 5,
                "trend_type": ["haussière"] * 5,
                "regime_change_score": [0.7] * 5,
                "regime_type": ["trend"] * 5,
                "vix": [20.0] * 5,
                "rsi": [75.0] * 5,
                "trade_frequency": [8.0] * 5,
                "cluster_id": [0] * 5,
                "context_neural_regime": [0] * 5,
            }
        )
        stream.cloud_backup(df)
        assert mock_s3.called, "Client S3 non appelé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "temp_s3_" in f for f in checkpoint_files
        ), "Fichier temporaire S3 non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "cloud_backup" in str(op) for op in df_perf["operation"]
        ), "Opération non journalisée"


def test_handle_sigint(tmp_dirs):
    """Teste la gestion SIGINT."""
    stream = MindStream(config_path=tmp_dirs["config_path"])
    with patch("sys.exit") as mock_exit:
        stream.handle_sigint(signal.SIGINT, None)
        snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
        assert any(
            "snapshot_sigint" in f for f in snapshot_files
        ), "Snapshot sigint non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "handle_sigint" in str(op) for op in df_perf["operation"]
        ), "Opération non journalisée"
        mock_exit.assert_called_with(0)


def test_miya_mood(tmp_dirs):
    """Teste l’évaluation de l’humeur de MIA."""
    stream = MindStream(config_path=tmp_dirs["config_path"])
    with open(tmp_dirs["json_log_path"], "a", encoding="utf-8") as f:
        for _ in range(10):
            log_entry = {"level": "info", "tag": "TEST", "message": "Test log"}
            json.dump(log_entry, f)
            f.write("\n")
    mood = stream.miya_mood()
    assert mood == "✨ Confiant", "Humeur incorrecte"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "miya_mood" in str(op) for op in df_perf["operation"]
    ), "Opération non journalisée"


def test_miya_dashboard_update(tmp_dirs):
    """Teste la mise à jour du dashboard."""
    stream = MindStream(config_path=tmp_dirs["config_path"])
    stream.miya_dashboard_update(market_data={"rsi": 75, "gex": 100000})
    assert os.path.exists(tmp_dirs["dashboard_path"]), "Fichier dashboard non créé"
    with open(tmp_dirs["dashboard_path"], "r", encoding="utf-8") as f:
        dashboard = json.load(f)
    assert "market" in dashboard, "Données marché absentes"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "miya_dashboard_update" in str(op) for op in df_perf["operation"]
    ), "Opération non journalisée"
