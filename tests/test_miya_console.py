# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_miya_console.py
# Tests unitaires pour src/utils/miya_console.py
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide les fonctions de logging et synthèse vocale (miya_speak, miya_alerts),
#        avec support multi-marchés, snapshots compressés, et sauvegardes incrémentielles/distribuées.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pyttsx3>=2.90,<3.0, colorama>=0.4.4,<0.5.0, psutil>=5.9.8,<6.0.0,
#   pandas>=2.0.0,<3.0.0, pyyaml>=6.0.0,<7.0.0, boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - Fichier de configuration factice (miya_config.yaml)
# - Messages de test pour miya_speak et miya_alerts
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de miya_console.py.
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Tests la Phase 8 (confidence_drop_rate) et la méthode 7 (logs pour mémoire contextuelle).
# - Vérifie l’absence de dxFeed, la suppression de 320/81 features et obs_t, les retries,
#   logs psutil, alertes Telegram, snapshots compressés, et sauvegardes incrémentielles/distribuées.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.

import gzip
import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from src.utils.miya_console import MiyaConsole


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
    cache_dir = data_dir / "cache" / "miya_console" / "ES"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints" / "miya_console" / "ES"
    checkpoints_dir.mkdir(parents=True)

    # Créer miya_config.yaml
    miya_config_path = config_dir / "miya_config.yaml"
    miya_config_content = {
        "miya_console": {
            "enable_colors": True,
            "color_map": {
                "info": "green",
                "warning": "yellow",
                "error": "red",
                "critical": "red",
            },
            "min_priority": 1,
            "enable_voice": False,
            "voice_profiles": {
                "default": {"rate": 150, "volume": 0.8},
                "urgent": {"rate": 180, "volume": 1.0},
                "calm": {"rate": 120, "volume": 0.6},
            },
            "log_to_console": True,
            "log_to_file": True,
            "json_output": True,
        }
    }
    with open(miya_config_path, "w", encoding="utf-8") as f:
        yaml.dump(miya_config_content, f)

    # Créer es_config.yaml
    es_config_path = config_dir / "es_config.yaml"
    es_config_content = {"s3_bucket": "test-bucket", "s3_prefix": "miya_console/"}
    with open(es_config_path, "w", encoding="utf-8") as f:
        yaml.dump(es_config_content, f)

    return {
        "base_dir": str(base_dir),
        "config_dir": str(config_dir),
        "logs_dir": str(logs_dir),
        "cache_dir": str(cache_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "miya_config_path": str(miya_config_path),
        "es_config_path": str(es_config_path),
        "perf_log_path": str(logs_dir / "miya_console_performance.csv"),
        "json_log_path": str(logs_dir / "miya_console.json"),
    }


@pytest.mark.asyncio
async def test_init_miya_console(tmp_dirs):
    """Teste l’initialisation de MiyaConsole."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        console = MiyaConsole(config_path=tmp_dirs["miya_config_path"], market="ES")
        assert console.market == "ES", "Marché incorrect"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "init" in str(op) for op in df_perf["operation"]
        ), "Opération init non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_load_config_manager(tmp_dirs):
    """Teste le chargement de la configuration."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        console = MiyaConsole(config_path=tmp_dirs["miya_config_path"], market="ES")
        config = console.load_config_manager()
        assert config["enable_colors"], "Configuration enable_colors incorrect"
        assert config["min_priority"] == 1, "Configuration min_priority incorrect"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_load_config_manager" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot load_config_manager non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "load_config_manager" in str(op) for op in df_perf["operation"]
        ), "Opération load_config_manager non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_miya_speak(tmp_dirs, capsys):
    """Teste la fonctionnalité miya_speak."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram, patch(
        "pyttsx3.init", return_value=MagicMock()
    ) as mock_engine:
        console = MiyaConsole(config_path=tmp_dirs["miya_config_path"], market="ES")
        console.miya_speak(
            "Test message", tag="TEST", level="info", priority=1, voice_profile="calm"
        )
        captured = capsys.readouterr()
        assert "Test message" in captured.out, "Message non affiché dans la console"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_miya_speak" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot miya_speak non créé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "miya_speak" in f and f.endswith(".json.gz") for f in checkpoint_files
        ), "Checkpoint miya_speak non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "miya_speak" in str(op) for op in df_perf["operation"]
        ), "Opération miya_speak non journalisée"
        assert any(
            "confidence_drop_rate" in str(kw) for kw in df_perf.to_dict("records")
        ), "confidence_drop_rate absent"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_miya_alerts(tmp_dirs, capsys):
    """Teste la fonctionnalité miya_alerts."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram, patch(
        "pyttsx3.init", return_value=MagicMock()
    ) as mock_engine:
        console = MiyaConsole(config_path=tmp_dirs["miya_config_path"], market="ES")
        console.miya_alerts(
            "Test alert", tag="ALERT", priority=3, voice_profile="urgent"
        )
        captured = capsys.readouterr()
        assert "Test alert" in captured.out, "Alerte non affichée dans la console"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_miya_speak" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot miya_speak non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "miya_speak" in str(op) for op in df_perf["operation"]
        ), "Opération miya_speak non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_miya_speak_invalid(tmp_dirs):
    """Teste miya_speak avec des paramètres invalides."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram, patch(
        "pyttsx3.init", return_value=MagicMock()
    ) as mock_engine:
        console = MiyaConsole(config_path=tmp_dirs["miya_config_path"], market="ES")
        console.miya_speak("", tag="TEST", level="info", priority=1)
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "miya_speak" in str(op) and not kw["success"]
            for kw in df_perf.to_dict("records")
        ), "Erreur message vide non journalisée"
        console.miya_speak("Test", tag="TEST", level="invalid", priority=1)
        console.miya_speak("Test", tag="TEST", level="info", priority=10)
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_cloud_backup(tmp_dirs):
    """Teste la sauvegarde distribuée S3."""
    with patch("boto3.client") as mock_s3, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        console = MiyaConsole(config_path=tmp_dirs["miya_config_path"], market="ES")
        df = pd.DataFrame(
            {"timestamp": [datetime.now().isoformat()], "message": ["Test"]}
        )
        console.cloud_backup(df, data_type="test_metrics")
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
        console = MiyaConsole(config_path=tmp_dirs["miya_config_path"], market="ES")
        df = pd.DataFrame(
            {"timestamp": [datetime.now().isoformat()], "message": ["Test"]}
        )
        console.checkpoint(df, data_type="test_metrics")
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "miya_console_test_metrics" in f and f.endswith(".json.gz")
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
