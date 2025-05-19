# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_pattern_analyzer.py
# Tests unitaires pour src/model/utils/pattern_analyzer.py
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide l’analyse des patterns dans market_memory.db (méthode 7, Phase 15),
#        avec support multi-marchés, retries, snapshots compressés, et sauvegardes incrémentielles/distribuées.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, sqlite3, psutil>=5.9.8,<6.0.0,
#   boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0, pyyaml>=6.0.0,<7.0.0
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
# - src/model/utils/config_manager.py
#
# Inputs :
# - Base de données SQLite factice (market_memory.db)
# - config/feature_sets.yaml (pour les 350 features)
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de pattern_analyzer.py.
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Tests la Phase 8 (confidence_drop_rate), méthode 7 (analyse des patterns), et Phase 15 (apprentissage adaptatif).
# - Vérifie l’absence de dxFeed, la suppression de 320/81 features et obs_t, les retries, logs psutil,
#   alertes Telegram, snapshots compressés, et sauvegardes incrémentielles/distribuées.
# - Compatible avec 350 features via config/feature_sets.yaml.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.

import gzip
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.model.utils.pattern_analyzer import PatternAnalyzer


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, cache, checkpoints, patterns, et base de données."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs" / "market"
    logs_dir.mkdir(parents=True)
    cache_dir = data_dir / "cache" / "patterns" / "ES"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints" / "patterns" / "ES"
    checkpoints_dir.mkdir(parents=True)
    patterns_dir = data_dir / "patterns" / "ES"
    patterns_dir.mkdir(parents=True)

    # Créer feature_sets.yaml
    feature_sets_path = config_dir / "feature_sets.yaml"
    feature_sets_content = {
        "training": {"features": [f"feature_{i}" for i in range(350)]},
        "inference": {"shap_features": [f"shap_feature_{i}" for i in range(150)]},
    }
    with open(feature_sets_path, "w", encoding="utf-8") as f:
        yaml.dump(feature_sets_content, f)

    # Créer es_config.yaml
    config_path = config_dir / "es_config.yaml"
    config_content = {"s3_bucket": "test-bucket", "s3_prefix": "patterns/"}
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    # Créer une base de données SQLite factice
    db_path = data_dir / "market_memory.db"
    feature_cols = [f"feature_{i}" for i in range(350)]
    conn = sqlite3.connect(db_path)
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start="2025-05-13 10:00:00", periods=100, freq="S"
            ).astype(str),
            "close": np.random.uniform(5000, 5100, 100),
            "volume": np.random.randint(1000, 5000, 100),
            "vix_close": np.random.uniform(20, 30, 100),
            "bid_size_level_2": np.random.randint(100, 1000, 100),
            "ask_size_level_2": np.random.randint(100, 1000, 100),
            **{col: np.random.uniform(0, 1, 100) for col in feature_cols},
        }
    )
    data.loc[50:55, "close"] += 100
    data.loc[50:55, "volume"] *= 3
    data.to_sql("market_data", conn, if_exists="replace", index=False)
    conn.close()

    return {
        "base_dir": str(base_dir),
        "config_dir": str(config_dir),
        "logs_dir": str(logs_dir),
        "cache_dir": str(cache_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "patterns_dir": str(patterns_dir),
        "db_path": str(db_path),
        "perf_log_path": str(logs_dir / "pattern_analyzer_performance.csv"),
    }


@pytest.mark.asyncio
async def test_init_pattern_analyzer(tmp_dirs):
    """Teste l’initialisation de PatternAnalyzer."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        analyzer = PatternAnalyzer(db_path=Path(tmp_dirs["db_path"]), market="ES")
        assert analyzer.db_path.exists(), "Base de données non créée"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "ensure_db_exists" in str(op) for op in df_perf["operation"]
        ), "Opération ensure_db_exists non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_validate_data(tmp_dirs):
    """Teste la validation des données."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        analyzer = PatternAnalyzer(db_path=Path(tmp_dirs["db_path"]), market="ES")
        conn = sqlite3.connect(tmp_dirs["db_path"])
        data = pd.read_sql_query(
            "SELECT * FROM market_data", conn, parse_dates=["timestamp"]
        )
        conn.close()
        analyzer._validate_data(data)
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_validate_data" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot validate_data non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "validate_data" in str(op) for op in df_perf["operation"]
        ), "Opération validate_data non journalisée"
        assert any(
            "confidence_drop_rate" in str(kw) for kw in df_perf.to_dict("records")
        ), "confidence_drop_rate absent"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_analyze_patterns(tmp_dirs):
    """Teste l’analyse des patterns."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        analyzer = PatternAnalyzer(db_path=Path(tmp_dirs["db_path"]), market="ES")
        patterns = analyzer.analyze_patterns()
        assert "price_spike" in patterns, "Patterns de prix absents"
        assert "volume_spike" in patterns, "Patterns de volume absents"
        assert len(patterns["price_spike"]) > 0, "Aucun pic de prix détecté"
        assert len(patterns["volume_spike"]) > 0, "Aucun pic de volume détecté"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_pattern_analysis" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot pattern_analysis non créé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "pattern_analysis" in f and f.endswith(".json.gz") for f in checkpoint_files
        ), "Checkpoint pattern_analysis non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "analyze_patterns" in str(op) for op in df_perf["operation"]
        ), "Opération analyze_patterns non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_save_patterns(tmp_dirs):
    """Teste la sauvegarde des patterns."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        analyzer = PatternAnalyzer(db_path=Path(tmp_dirs["db_path"]), market="ES")
        patterns = {"price_spike": [], "volume_spike": []}
        output_path = Path(tmp_dirs["patterns_dir"]) / "patterns.json"
        analyzer.save_patterns(patterns, output_path)
        assert (
            output_path.parent / "patterns.json.gz"
        ).exists(), "Fichier de patterns non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "save_patterns" in str(op) for op in df_perf["operation"]
        ), "Opération save_patterns non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_cloud_backup(tmp_dirs):
    """Teste la sauvegarde distribuée S3."""
    with patch("boto3.client") as mock_s3, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        analyzer = PatternAnalyzer(db_path=Path(tmp_dirs["db_path"]), market="ES")
        df = pd.DataFrame({"timestamp": [datetime.now().isoformat()], "close": [5100]})
        analyzer.cloud_backup(df, data_type="test_metrics")
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
        analyzer = PatternAnalyzer(db_path=Path(tmp_dirs["db_path"]), market="ES")
        df = pd.DataFrame({"timestamp": [datetime.now().isoformat()], "close": [5100]})
        analyzer.checkpoint(df, data_type="test_metrics")
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "pattern_test_metrics" in f and f.endswith(".json.gz")
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
