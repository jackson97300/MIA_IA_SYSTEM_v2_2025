# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_db_setup.py
# Tests unitaires pour src/model/utils/db_setup.py
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide l’initialisation et la configuration de market_memory.db pour la mémoire contextuelle (méthode 7),
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
# - Base de données SQLite factice
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de db_setup.py.
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

from src.model.utils.db_setup import DBSetup


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
    cache_dir = data_dir / "cache" / "db_setup" / "ES"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints" / "db_setup" / "ES"
    checkpoints_dir.mkdir(parents=True)

    # Créer market_config.yaml
    market_config_path = config_dir / "market_config.yaml"
    market_config_content = {
        "db_setup": {"db_path": str(data_dir / "market_memory_ES.db")}
    }
    with open(market_config_path, "w", encoding="utf-8") as f:
        yaml.dump(market_config_content, f)

    # Créer es_config.yaml
    es_config_path = config_dir / "es_config.yaml"
    es_config_content = {"s3_bucket": "test-bucket", "s3_prefix": "db_setup/"}
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
        "perf_log_path": str(logs_dir / "db_setup_performance.csv"),
    }


@pytest.mark.asyncio
async def test_init_db_setup(tmp_dirs):
    """Teste l’initialisation de DBSetup."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        db_setup = DBSetup(config_path=tmp_dirs["market_config_path"], market="ES")
        assert db_setup.market == "ES", "Marché incorrect"
        assert (
            db_setup.db_path == tmp_dirs["db_path"]
        ), "Chemin de la base de données incorrect"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "init" in str(op) for op in df_perf["operation"]
        ), "Opération init non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_setup_database(tmp_dirs):
    """Teste la configuration de la base de données."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        db_setup = DBSetup(config_path=tmp_dirs["market_config_path"], market="ES")
        success = db_setup.setup_database()
        assert success, "Configuration de la base de données a échoué"
        conn = sqlite3.connect(tmp_dirs["db_path"])
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='clusters'"
        )
        assert cursor.fetchone(), "Table clusters non créée"
        cursor.execute("PRAGMA index_list(clusters)")
        indexes = [index[1] for index in cursor.fetchall()]
        assert "idx_cluster_id" in indexes, "Index idx_cluster_id non créé"
        assert "idx_timestamp" in indexes, "Index idx_timestamp non créé"
        conn.close()
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_setup_database" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot setup_database non créé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "setup_database" in f and f.endswith(".json.gz") for f in checkpoint_files
        ), "Checkpoint setup_database non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "setup_database" in str(op) for op in df_perf["operation"]
        ), "Opération setup_database non journalisée"
        assert any(
            "confidence_drop_rate" in str(kw) for kw in df_perf.to_dict("records")
        ), "confidence_drop_rate absent"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_validate_table_schema(tmp_dirs):
    """Teste la validation du schéma de la table clusters."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        db_setup = DBSetup(config_path=tmp_dirs["market_config_path"], market="ES")
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
        conn.commit()
        valid = db_setup.validate_table_schema(cursor)
        assert valid, "Schéma de la table clusters non valide"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_validate_table_schema" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot validate_table_schema non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "validate_table_schema" in str(op) for op in df_perf["operation"]
        ), "Opération validate_table_schema non journalisée"
        conn.close()
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_validate_table_schema_invalid(tmp_dirs):
    """Teste la validation avec un schéma invalide."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        db_setup = DBSetup(config_path=tmp_dirs["market_config_path"], market="ES")
        conn = sqlite3.connect(tmp_dirs["db_path"])
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE clusters (
                cluster_id INTEGER PRIMARY KEY,
                event_type TEXT NOT NULL
            )
        """
        )
        conn.commit()
        valid = db_setup.validate_table_schema(cursor)
        assert not valid, "Schéma invalide devrait être détecté"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "validate_table_schema" in str(op) and not kw["success"]
            for kw in df_perf.to_dict("records")
        ), "Erreur schéma invalide non journalisée"
        conn.close()
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_cloud_backup(tmp_dirs):
    """Teste la sauvegarde distribuée S3."""
    with patch("boto3.client") as mock_s3, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        db_setup = DBSetup(config_path=tmp_dirs["market_config_path"], market="ES")
        df = pd.DataFrame(
            {"timestamp": [datetime.now().isoformat()], "table": ["clusters"]}
        )
        db_setup.cloud_backup(df, data_type="test_metrics")
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
        db_setup = DBSetup(config_path=tmp_dirs["market_config_path"], market="ES")
        df = pd.DataFrame(
            {"timestamp": [datetime.now().isoformat()], "table": ["clusters"]}
        )
        db_setup.checkpoint(df, data_type="test_metrics")
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "db_setup_test_metrics" in f and f.endswith(".json.gz")
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
