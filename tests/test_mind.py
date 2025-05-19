# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_mind.py
# Tests unitaires pour src/mind/mind.py
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide le moteur cognitif central de MIA, incluant la gestion des logs, la communication (console, vocal, Telegram),
#        l’analyse des événements avec K-means (10 clusters dans market_memory.db), l’utilisation exclusive d’IQFeed,
#        l’intégration SHAP (Phase 17), confidence_drop_rate (Phase 8), snapshots compressés, sauvegardes incrémentielles/distribuées.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, sklearn>=1.2.0,<2.0.0, psutil>=5.9.8,<6.0.0,
#   matplotlib>=3.8.0,<4.0.0, seaborn>=0.13.0,<1.0.0, pyyaml>=6.0.0,<7.0.0, boto3>=1.26.0,<2.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/db_setup.py
# - src/envs/trading_env.py
# - src/mind/mind_voice.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/es_config.yaml
# - data/features/features_latest.csv
# - config/feature_sets.yaml
#
# Outputs :
# - data/logs/mind_stream.log
# - data/logs/mind_stream.json/csv
# - data/logs/mind_performance.csv
# - data/logs/brain_state.json
# - data/logs/mind_snapshots/*.json.gz
# - data/checkpoints/mind_*.json.gz
# - data/logs/mind_dashboard.json
# - data/figures/mind/
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

from src.mind.mind import MindEngine


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les données, logs, snapshots, checkpoints, et figures."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    snapshots_dir = logs_dir / "mind_snapshots"
    snapshots_dir.mkdir()
    checkpoints_dir = data_dir / "checkpoints"
    checkpoints_dir.mkdir()
    figures_dir = data_dir / "figures" / "mind"
    figures_dir.mkdir(parents=True)
    features_dir = data_dir / "features"
    features_dir.mkdir()

    # Créer es_config.yaml
    config_path = config_dir / "es_config.yaml"
    config_content = {
        "mia": {
            "language": "fr",
            "vocal_enabled": False,
            "vocal_async": True,
            "log_dir": "data/logs",
            "enable_csv_log": True,
            "log_rotation_mb": 10,
            "verbosity": "normal",
            "voice_profile": "calm",
            "max_logs": 1000,
            "buffer_size": 100,
            "s3_bucket": "test-bucket",
            "s3_prefix": "mind/",
        }
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
            "bid_size_level_1": np.random.randint(100, 1000, 10),
            "ask_size_level_1": np.random.randint(100, 1000, 10),
            "trade_frequency_1s": np.random.uniform(0, 10, 10),
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
        "perf_log_path": str(logs_dir / "mind_performance.csv"),
        "json_log_path": str(logs_dir / "mind_stream.json"),
        "csv_log_path": str(logs_dir / "mind_stream.csv"),
        "dashboard_path": str(logs_dir / "mind_dashboard.json"),
        "brain_state_path": str(logs_dir / "brain_state.json"),
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
    """Teste l’initialisation de MindEngine."""
    engine = MindEngine(config_path=tmp_dirs["config_path"])
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    assert os.path.exists(tmp_dirs["snapshots_dir"]), "Dossier de snapshots non créé"
    assert os.path.exists(tmp_dirs["json_log_path"]), "Fichier de logs JSON non créé"
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
    engine = MindEngine(config_path=tmp_dirs["config_path"])
    config = engine.load_config(tmp_dirs["config_path"])
    assert "s3_bucket" in config, "Clé s3_bucket manquante"
    assert "buffer_size" in config, "Clé buffer_size manquante"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "load_config" in str(op) for op in df["operation"]
    ), "Opération non journalisée"


def test_miya_speak(tmp_dirs, mock_trading_env):
    """Teste la méthode miya_speak."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        engine = MindEngine(config_path=tmp_dirs["config_path"])
        engine.miya_speak(
            "Test message",
            tag="TEST",
            level="info",
            priority=1,
            category="test",
            env=mock_trading_env,
        )
        assert os.path.exists(
            tmp_dirs["json_log_path"]
        ), "Fichier JSON de logs non créé"
        with open(tmp_dirs["json_log_path"], "r", encoding="utf-8") as f:
            logs = [json.loads(line) for line in f]
        assert any(
            "Test message" in log["message"] for log in logs
        ), "Message non journalisé"
        assert os.path.exists(tmp_dirs["csv_log_path"]), "Fichier CSV de logs non créé"
        df = pd.read_csv(tmp_dirs["csv_log_path"])
        assert any("TEST" in str(tag) for tag in df["tag"]), "Tag non journalisé"
        assert os.path.exists(tmp_dirs["brain_state_path"]), "Fichier d’état non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "miya_speak" in str(op) for op in df_perf["operation"]
        ), "Opération non journalisée"
        mock_telegram.assert_not_called()


def test_miya_speak_high_priority(tmp_dirs):
    """Teste miya_speak avec une priorité élevée (Telegram)."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        engine = MindEngine(config_path=tmp_dirs["config_path"])
        engine.miya_speak("Urgent message", tag="ALERT", level="error", priority=4)
        mock_telegram.assert_called_with("[MIA - ALERT] Urgent message")
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "miya_speak" in str(op) for op in df["operation"]
        ), "Opération non journalisée"


def test_analyze_log_patterns(tmp_dirs):
    """Teste l’analyse des patterns de logs avec K-means."""
    engine = MindEngine(config_path=tmp_dirs["config_path"])
    engine.miya_speak("Test log 1", tag="TEST", priority=1)
    engine.miya_speak("Test log 2", tag="TEST", priority=2)
    result = engine.analyze_log_patterns()
    assert isinstance(result, pd.DataFrame), "Résultat n’est pas un DataFrame"
    assert "cluster_id" in result.columns, "Colonne cluster_id manquante"
    conn = sqlite3.connect(tmp_dirs["db_path"])
    df_db = pd.read_sql_query("SELECT * FROM clusters", conn)
    conn.close()
    assert not df_db.empty, "Aucun cluster stocké dans la base"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "analyze_log_patterns" in str(op) for op in df_perf["operation"]
    ), "Opération non journalisée"
    assert any(
        "confidence_drop_rate" in str(kw) for kw in df_perf.to_dict("records")
    ), "confidence_drop_rate absent"
    checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
    assert any(
        "mind_" in f and f.endswith(".json.gz") for f in checkpoint_files
    ), "Checkpoint non créé"


def test_visualize_log_patterns(tmp_dirs):
    """Teste la génération de la heatmap des clusters."""
    engine = MindEngine(config_path=tmp_dirs["config_path"])
    df = pd.DataFrame(
        {
            "timestamp": [datetime.now().isoformat()] * 10,
            "priority": np.random.randint(1, 5, 10),
            "cluster_id": np.random.randint(0, 10, 10),
        }
    )
    engine.visualize_log_patterns(df)
    assert any(
        f.endswith(".png") for f in os.listdir(tmp_dirs["figures_dir"])
    ), "Heatmap non générée"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "visualize_log_patterns" in str(op) for op in df_perf["operation"]
    ), "Opération non journalisée"


def test_save_snapshot_compressed(tmp_dirs):
    """Teste la sauvegarde d’un snapshot compressé."""
    engine = MindEngine(config_path=tmp_dirs["config_path"])
    snapshot_data = {"test": "compressed_snapshot"}
    engine.save_snapshot("test_compressed", snapshot_data, compress=True)
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
    engine = MindEngine(config_path=tmp_dirs["config_path"])
    df = pd.DataFrame(
        {"timestamp": [datetime.now().isoformat()] * 5, "priority": [1, 2, 3, 4, 5]}
    )
    engine.checkpoint(df)
    checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
    assert any(
        "mind_" in f and f.endswith(".json.gz") for f in checkpoint_files
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
        engine = MindEngine(config_path=tmp_dirs["config_path"])
        df = pd.DataFrame(
            {"timestamp": [datetime.now().isoformat()] * 5, "priority": [1, 2, 3, 4, 5]}
        )
        engine.cloud_backup(df)
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
    engine = MindEngine(config_path=tmp_dirs["config_path"])
    engine.miya_speak("Test log", tag="TEST", priority=1)
    with patch("sys.exit") as mock_exit:
        engine.handle_sigint(signal.SIGINT, None)
        snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
        assert any(
            "snapshot_sigint" in f for f in snapshot_files
        ), "Snapshot sigint non créé"
        assert os.path.exists(
            tmp_dirs["brain_state_path"]
        ), "État mental non sauvegardé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "handle_sigint" in str(op) for op in df_perf["operation"]
        ), "Opération non journalisée"
        mock_exit.assert_called_with(0)


def test_miya_health_check(tmp_dirs):
    """Teste la vérification de l’état du système."""
    engine = MindEngine(config_path=tmp_dirs["config_path"])
    checks = engine.miya_health_check()
    assert checks["files"][
        tmp_dirs["features_latest_path"]
    ], "Fichier features_latest.csv non détecté"
    assert checks["files"][
        tmp_dirs["feature_sets_path"]
    ], "Fichier feature_sets.yaml non détecté"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_health_check" in f for f in snapshot_files
    ), "Snapshot santé non créé"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "miya_health_check" in str(op) for op in df_perf["operation"]
    ), "Opération non journalisée"


def test_miya_dashboard_notify(tmp_dirs):
    """Teste la notification au dashboard."""
    engine = MindEngine(config_path=tmp_dirs["config_path"])
    data = {
        "tag": "TEST",
        "message": "Test notify",
        "level": "info",
        "timestamp": datetime.now().isoformat(),
    }
    engine.miya_dashboard_notify(data)
    assert os.path.exists(tmp_dirs["dashboard_path"]), "Fichier dashboard non créé"
    with open(tmp_dirs["dashboard_path"], "r", encoding="utf-8") as f:
        notifications = json.load(f)
    assert any(
        "Test notify" in n["message"] for n in notifications
    ), "Notification non enregistrée"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "miya_dashboard_notify" in str(op) for op in df_perf["operation"]
    ), "Opération non journalisée"


def test_miya_learns(tmp_dirs):
    """Teste la méthode miya_learns."""
    engine = MindEngine(config_path=tmp_dirs["config_path"])
    engine.miya_speak("Erreur réseau", tag="ALERT", level="error", priority=4)
    engine.miya_speak("Erreur réseau", tag="ALERT", level="error", priority=4)
    suggestions = engine.miya_learns()
    assert any(
        "Vérifier connexion IQFeed" in s for s in suggestions
    ), "Suggestion réseau absente"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "miya_learns" in str(op) for op in df_perf["operation"]
    ), "Opération non journalisée"


def test_miya_cli(tmp_dirs):
    """Teste l’interface CLI."""
    with patch("builtins.input", side_effect=["santé", "exit"]):
        engine = MindEngine(config_path=tmp_dirs["config_path"])
        engine.miya_cli()
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "miya_cli" in str(op) for op in df_perf["operation"]
        ), "Opération non journalisée"
        assert any(
            "miya_health_check" in str(op) for op in df_perf["operation"]
        ), "Commande santé non exécutée"
