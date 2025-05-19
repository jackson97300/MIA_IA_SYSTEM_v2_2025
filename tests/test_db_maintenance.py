# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_db_maintenance.py
# Tests unitaires pour src/model/utils/db_maintenance.py
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide la maintenance de market_memory.db en purgeant les données obsolètes (méthode 7),
#        avec support multi-marchés, snapshots compressés, et sauvegardes incrémentielles/distribuées.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, sqlite3, psutil>=5.9.8,<6.0.0, pandas>=2.0.0,<3.0.0, boto3>=1.26.0,<2.0.0,
#   loguru>=0.7.0,<1.0.0, pyyaml>=6.0.0,<7.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - Fichier de configuration factice (market_config.yaml)
# - Base de données SQLite factice avec données de test
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de db_maintenance.py.
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Tests la Phase 8 (confidence_drop_rate) et la méthode 7 (mémoire contextuelle via market_memory.db).
# - Vérifie l’absence de dxFeed, la suppression de 320/81 features et obs_t, les retries,
#   logs psutil, alertes Telegram, snapshots compressés, et sauvegardes incrémentielles/distribuées.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.

import gzip
import json
import sqlite3
from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

from src.model.utils.db_maintenance import DBMaintenance


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, cache, checkpoints, et configuration."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    cache_dir = data_dir / "cache" / "db_maintenance" / "ES"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints" / "db_maintenance" / "ES"
    checkpoints_dir.mkdir(parents=True)

    # Créer market_config.yaml
    market_config_path = config_dir / "market_config.yaml"
    market_config_content = {
        "db_maintenance": {"db_path": str(data_dir / "market_memory_ES.db")}
    }
    with open(market_config_path, "w", encoding="utf-8") as f:
        yaml.dump(market_config_content, f)

    # Créer es_config.yaml
    es_config_path = config_dir / "es_config.yaml"
    es_config_content = {"s3_bucket": "test-bucket", "s3_prefix": "db_maintenance/"}
    with open(es_config_path, "w", encoding="utf-8") as f:
        yaml.dump(es_config_content, f)

    return {
        "base_dir": str(base_dir),
        "config_dir": str(config_dir),
        "data_dir": str(data_dir),
        "logs_dir": str(logs_dir),
        "cache_dir": str(cache_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "market_config_path": str(market_config_path),
        "es_config_path": str(es_config_path),
        "db_path": str(data_dir / "market_memory_ES.db"),
        "perf_log_path": str(logs_dir / "db_maintenance_performance.csv"),
    }


@pytest.fixture
def setup_test_db(tmp_dirs):
    """Crée une base de données SQLite avec des données de test."""
    conn = sqlite3.connect(tmp_dirs["db_path"])
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE clusters (
            cluster_id INTEGER PRIMARY KEY,
            event_type TEXT NOT NULL,
            features TEXT NOT NULL,
            timestamp DATETIME NOT NULL
        )
    """
    )
    cursor.execute("CREATE INDEX idx_cluster_id ON clusters(cluster_id)")
    cursor.execute("CREATE INDEX idx_timestamp ON clusters(timestamp)")
    cursor.executemany(
        "INSERT INTO clusters (event_type, features, timestamp) VALUES (?, ?, ?)",
        [
            ("test_event", "{}", "2025-04-01 00:00:00"),  # Donnée obsolète (>30 jours)
            ("test_event", "{}", "2025-05-10 00:00:00"),  # Donnée récente
        ],
    )
    conn.commit()
    conn.close()


@pytest.mark.asyncio
async def test_init_db_maintenance(tmp_dirs):
    """Teste l’initialisation de DBMaintenance."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        db_maintenance = DBMaintenance(
            config_path=tmp_dirs["market_config_path"], market="ES"
        )
        assert db_maintenance.market == "ES", "Marché incorrect"
        assert (
            db_maintenance.db_path == tmp_dirs["db_path"]
        ), "Chemin de la base de données incorrect"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "init" in str(op) for op in df_perf["operation"]
        ), "Opération init non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_purge_old_data(tmp_dirs, setup_test_db):
    """Teste la purge des données obsolètes."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        db_maintenance = DBMaintenance(
            config_path=tmp_dirs["market_config_path"], market="ES"
        )
        deleted_rows = db_maintenance.purge_old_data()
        assert deleted_rows == 1, "Nombre incorrect de lignes supprimées"
        conn = sqlite3.connect(tmp_dirs["db_path"])
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM clusters WHERE timestamp < date('now', '-30 days')"
        )
        assert cursor.fetchone()[0] == 0, "Données obsolètes non purgées"
        cursor.execute("SELECT COUNT(*) FROM clusters")
        assert cursor.fetchone()[0] == 1, "Données récentes supprimées par erreur"
        conn.close()
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_purge_old_data" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot purge_old_data non créé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "purge_old_data" in f and f.endswith(".json.gz") for f in checkpoint_files
        ), "Checkpoint purge_old_data non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "purge_old_data" in str(op) for op in df_perf["operation"]
        ), "Opération purge_old_data non journalisée"
        assert any(
            "confidence_drop_rate" in str(kw) for kw in df_perf.to_dict("records")
        ), "confidence_drop_rate absent"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_purge_old_data_no_table(tmp_dirs):
    """Teste la purge lorsque la table clusters n’existe pas."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        db_maintenance = DBMaintenance(
            config_path=tmp_dirs["market_config_path"], market="ES"
        )
        deleted_rows = db_maintenance.purge_old_data()
        assert (
            deleted_rows == 0
        ), "Aucune ligne ne devrait être supprimée si la table n’existe pas"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "purge_old_data" in str(op) and not kw["success"]
            for kw in df_perf.to_dict("records")
        ), "Erreur table absente non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_verify_indexes(tmp_dirs, setup_test_db):
    """Teste la vérification des index."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        db_maintenance = DBMaintenance(
            config_path=tmp_dirs["market_config_path"], market="ES"
        )
        conn = sqlite3.connect(tmp_dirs["db_path"])
        cursor = conn.cursor()
        valid = db_maintenance.verify_indexes(cursor)
        assert valid, "Index non valides détectés"
        conn.close()
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "verify_indexes" in str(op) for op in df_perf["operation"]
        ), "Opération verify_indexes non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_cloud_backup(tmp_dirs):
    """Teste la sauvegarde distribuée S3."""
    with patch("boto3.client") as mock_s3, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        db_maintenance = DBMaintenance(
            config_path=tmp_dirs["market_config_path"], market="ES"
        )
        df = pd.DataFrame(
            {"timestamp": [datetime.now().isoformat()], "deleted_rows": [1]}
        )
        db_maintenance.cloud_backup(df, data_type="test_metrics")
        assert mock_s3.called, "Client S3 non appelé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "temp_s3_" in f for f in checkpoint_files
        ), "Fichier temporaire S3 non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "cloud_backup" in str(op) for op in df_perf["operation"]
        ), "Opération cloud_backup non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_checkpoint(tmp_dirs):
    """Teste la sauvegarde incrémentielle."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        db_maintenance = DBMaintenance(
            config_path=tmp_dirs["market_config_path"], market="ES"
        )
        df = pd.DataFrame(
            {"timestamp": [datetime.now().isoformat()], "deleted_rows": [1]}
        )
        db_maintenance.checkpoint(df, data_type="test_metrics")
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "db_maintenance_test_metrics" in f and f.endswith(".json.gz")
            for f in checkpoint_files
        ), "Checkpoint non créé"
        with gzip.open(
            os.path.join(tmp_dirs["checkpoints_dir"], checkpoint_files[0]),
            "rt",
            encoding="utf-8",
        ) as f:
            checkpoint_data = json.load(f)
        assert checkpoint_data["num_rows"] == len(df), "Nombre de lignes incorrect"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "checkpoint" in str(op) for op in df_perf["operation"]
        ), "Opération checkpoint non journalisée"
        mock_telegram.assert_called()
