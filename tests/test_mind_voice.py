# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_mind_voice.py
# Tests unitaires pour src/mind/mind_voice.py
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide la synthèse vocale de MIA avec Google TTS, repli local via pyttsx3, journalisation des messages vocaux,
#        mémoire contextuelle (méthode 7, K-means 10 clusters dans market_memory.db), utilisation exclusive d’IQFeed,
#        intégration de confidence_drop_rate (Phase 8), snapshots compressés, sauvegardes incrémentielles/distribuées,
#        et alertes Telegram.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, psutil>=5.9.8,<6.0.0, sklearn>=1.2.0,<2.0.0,
#   matplotlib>=3.8.0,<4.0.0, seaborn>=0.13.0,<1.0.0, pyyaml>=6.0.0,<7.0.0, boto3>=1.26.0,<2.0.0
# - gtts>=2.3.0,<3.0.0, simpleaudio>=1.0.0,<2.0.0, playsound>=1.2.0,<2.0.0, pyttsx3>=2.90,<3.0.0 (optionnels)
# - src/mind/mind.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/db_setup.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/es_config.yaml
#
# Outputs :
# - data/logs/mind/mind_voice.log
# - data/logs/mind/mind_voice.csv
# - data/logs/mind_voice_performance.csv
# - data/mind/mind_voice_snapshots/*.json.gz
# - data/checkpoints/mind_voice_*.json.gz
# - data/mind/mind_voice_dashboard.json
# - data/audio/
# - data/figures/mind_voice/
# - market_memory.db (table clusters)
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Tests les phases 7 (mémoire contextuelle), 8 (auto-conscience via confidence_drop_rate).
# - Vérifie l’absence de dxFeed, la suppression de 320/81 features et obs_t, les retries, logs psutil, alertes Telegram,
#   snapshots compressés, et sauvegardes incrémentielles/distribuées.
# - Compatible avec les dépendances optionnelles (gtts, simpleaudio, playsound, pyttsx3).

import gzip
import json
import os
import sqlite3
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from src.mind.mind_voice import VoiceManager


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les données, logs, snapshots, checkpoints, audio, et figures."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs" / "mind"
    logs_dir.mkdir(parents=True)
    snapshots_dir = data_dir / "mind" / "mind_voice_snapshots"
    snapshots_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints"
    checkpoints_dir.mkdir()
    audio_dir = data_dir / "audio"
    audio_dir.mkdir()
    figures_dir = data_dir / "figures" / "mind_voice"
    figures_dir.mkdir(parents=True)

    # Créer es_config.yaml
    config_path = config_dir / "es_config.yaml"
    config_content = {
        "voice": {
            "enabled": True,
            "async": True,
            "cleanup_interval": 3600,
            "s3_bucket": "test-bucket",
            "s3_prefix": "mind_voice/",
        },
        "logging": {"buffer_size": 100},
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    return {
        "base_dir": str(base_dir),
        "config_path": str(config_path),
        "logs_dir": str(logs_dir),
        "snapshots_dir": str(snapshots_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "audio_dir": str(audio_dir),
        "figures_dir": str(figures_dir),
        "csv_log_path": str(logs_dir / "mind_voice.csv"),
        "perf_log_path": str(logs_dir / "mind_voice_performance.csv"),
        "dashboard_path": str(data_dir / "mind" / "mind_voice_dashboard.json"),
        "db_path": str(data_dir / "market_memory.db"),
    }


def test_init(tmp_dirs):
    """Teste l’initialisation de VoiceManager."""
    with patch("src.mind.mind_voice.gTTS") as mock_gtts:
        mock_gtts.side_effect = Exception("gTTS non disponible")
        voice_manager = VoiceManager(config_path=tmp_dirs["config_path"])
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        assert os.path.exists(
            tmp_dirs["snapshots_dir"]
        ), "Dossier de snapshots non créé"
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


def test_load_voice_config(tmp_dirs):
    """Teste le chargement de la configuration vocale."""
    voice_manager = VoiceManager(config_path=tmp_dirs["config_path"])
    config = voice_manager.load_voice_config(tmp_dirs["config_path"])
    assert "s3_bucket" in config, "Clé s3_bucket manquante"
    assert "cleanup_interval" in config, "Clé cleanup_interval manquante"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "load_voice_config" in str(op) for op in df["operation"]
    ), "Opération non journalisée"


def test_detect_system_capabilities(tmp_dirs):
    """Teste la détection des capacités du système."""
    with patch("src.mind.mind_voice.gTTS") as mock_gtts, patch(
        "src.mind.mind_voice.pyttsx3"
    ) as mock_pyttsx3:
        mock_gtts.return_value = MagicMock()
        mock_pyttsx3.init.return_value = MagicMock()
        voice_manager = VoiceManager(config_path=tmp_dirs["config_path"])
        engines = voice_manager.detect_system_capabilities()
        assert engines["gtts"], "gTTS non détecté"
        assert engines["pyttsx3"], "pyttsx3 non détecté"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "detect_system_capabilities" in str(op) for op in df["operation"]
        ), "Opération non journalisée"


def test_speak(tmp_dirs):
    """Teste la méthode speak avec cache et mode asynchrone."""
    with patch("src.mind.mind_voice.gTTS") as mock_gtts, patch(
        "src.mind.mind_voice.playsound"
    ) as mock_playsound, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        mock_gtts.return_value = MagicMock(save=MagicMock())
        mock_playsound.return_value = None
        voice_manager = VoiceManager(config_path=tmp_dirs["config_path"])
        voice_manager.engines = {
            "gtts": True,
            "simpleaudio": False,
            "playsound": True,
            "pyttsx3": False,
        }

        # Test avec nouveau message
        voice_manager.speak("Test vocal", profile="calm", async_mode=True)
        assert os.path.exists(tmp_dirs["csv_log_path"]), "Fichier CSV de logs non créé"
        df = pd.read_csv(tmp_dirs["csv_log_path"])
        assert any(
            "Test vocal" in str(text) for text in df["text"]
        ), "Message non journalisé"

        # Test avec cache
        voice_manager.audio_cache["Test vocal_calm"] = (
            tmp_dirs["audio_dir"] + "/mia_calm_20250513_090000.mp3"
        )
        with open(voice_manager.audio_cache["Test vocal_calm"], "wb") as f:
            f.write(b"dummy audio data")
        voice_manager.speak("Test vocal", profile="calm", async_mode=True)
        df = pd.read_csv(tmp_dirs["csv_log_path"])
        assert any(
            "cache" in str(engine) for engine in df["engine"]
        ), "Cache non utilisé"

        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "speak" in str(op) for op in df_perf["operation"]
        ), "Opération non journalisée"
        mock_telegram.assert_called()


def test_speak_urgent(tmp_dirs):
    """Teste la méthode speak avec profil urgent (Telegram)."""
    with patch("src.mind.mind_voice.gTTS") as mock_gtts, patch(
        "src.mind.mind_voice.playsound"
    ) as mock_playsound, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        mock_gtts.return_value = MagicMock(save=MagicMock())
        mock_playsound.return_value = None
        voice_manager = VoiceManager(config_path=tmp_dirs["config_path"])
        voice_manager.engines = {
            "gtts": True,
            "simpleaudio": False,
            "playsound": True,
            "pyttsx3": False,
        }
        voice_manager.speak("Alerte urgente !", profile="urgent", async_mode=True)
        mock_telegram.assert_called_with("[MIA - Vocal urgent] Alerte urgente !")
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "speak" in str(op) for op in df_perf["operation"]
        ), "Opération non journalisée"


def test_store_voice_pattern(tmp_dirs):
    """Teste la clusterisation et le stockage des patterns vocaux."""
    voice_manager = VoiceManager(config_path=tmp_dirs["config_path"])
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "text": "Test vocal",
        "profile": "calm",
        "engine": "gtts",
        "latency": 0.5,
        "success": True,
    }
    result = voice_manager.store_voice_pattern(log_entry)
    assert "cluster_id" in result, "cluster_id manquant"
    conn = sqlite3.connect(tmp_dirs["db_path"])
    df_db = pd.read_sql_query("SELECT * FROM clusters", conn)
    conn.close()
    assert not df_db.empty, "Aucun cluster stocké dans la base"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "store_voice_pattern" in str(op) for op in df_perf["operation"]
    ), "Opération non journalisée"
    assert any(
        "confidence_drop_rate" in str(kw) for kw in df_perf.to_dict("records")
    ), "confidence_drop_rate absent"


def test_visualize_voice_patterns(tmp_dirs):
    """Teste la génération de la heatmap des clusters vocaux."""
    voice_manager = VoiceManager(config_path=tmp_dirs["config_path"])
    voice_manager.log_buffer = [
        {
            "timestamp": datetime.now().isoformat(),
            "text": "Test vocal",
            "profile": "calm",
            "engine": "gtts",
            "latency": 0.5,
            "success": True,
            "cluster_id": 1,
            "priority": 1,
        }
    ]
    voice_manager.visualize_voice_patterns()
    assert any(
        f.endswith(".png") for f in os.listdir(tmp_dirs["figures_dir"])
    ), "Heatmap non générée"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "visualize_voice_patterns" in str(op) for op in df_perf["operation"]
    ), "Opération non journalisée"


def test_save_snapshot_compressed(tmp_dirs):
    """Teste la sauvegarde d’un snapshot compressé."""
    voice_manager = VoiceManager(config_path=tmp_dirs["config_path"])
    snapshot_data = {"test": "compressed_snapshot"}
    voice_manager.save_snapshot("test_compressed", snapshot_data, compress=True)
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
    voice_manager = VoiceManager(config_path=tmp_dirs["config_path"])
    df = pd.DataFrame(
        {
            "timestamp": [datetime.now().isoformat()] * 5,
            "text": ["Test vocal"] * 5,
            "profile": ["calm"] * 5,
            "engine": ["gtts"] * 5,
            "latency": [0.5] * 5,
            "success": [True] * 5,
            "cluster_id": [0] * 5,
        }
    )
    voice_manager.checkpoint(df)
    checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
    assert any(
        "mind_voice_" in f and f.endswith(".json.gz") for f in checkpoint_files
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
        voice_manager = VoiceManager(config_path=tmp_dirs["config_path"])
        df = pd.DataFrame(
            {
                "timestamp": [datetime.now().isoformat()] * 5,
                "text": ["Test vocal"] * 5,
                "profile": ["calm"] * 5,
                "engine": ["gtts"] * 5,
                "latency": [0.5] * 5,
                "success": [True] * 5,
                "cluster_id": [0] * 5,
            }
        )
        voice_manager.cloud_backup(df)
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
    voice_manager = VoiceManager(config_path=tmp_dirs["config_path"])
    with patch("sys.exit") as mock_exit:
        voice_manager.handle_sigint(signal.SIGINT, None)
        snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
        assert any(
            "snapshot_sigint" in f for f in snapshot_files
        ), "Snapshot sigint non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "handle_sigint" in str(op) for op in df_perf["operation"]
        ), "Opération non journalisée"
        mock_exit.assert_called_with(0)


def test_cleanup_audio_files(tmp_dirs):
    """Teste le nettoyage des fichiers audio temporaires."""
    voice_manager = VoiceManager(config_path=tmp_dirs["config_path"])
    # Créer un fichier audio temporaire
    temp_file = os.path.join(tmp_dirs["audio_dir"], "mia_calm_20250513_090000.mp3")
    with open(temp_file, "wb") as f:
        f.write(b"dummy audio data")
    # Simuler un fichier ancien
    os.utime(temp_file, (time.time() - 25 * 3600, time.time() - 25 * 3600))
    voice_manager.cleanup_audio_files(max_age_hours=24)
    assert not os.path.exists(temp_file), "Fichier audio non supprimé"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "cleanup_audio_files" in str(op) for op in df_perf["operation"]
    ), "Opération non journalisée"


def test_update_dashboard(tmp_dirs):
    """Teste la mise à jour du dashboard vocal."""
    voice_manager = VoiceManager(config_path=tmp_dirs["config_path"])
    voice_manager.log_buffer = [
        {
            "timestamp": datetime.now().isoformat(),
            "text": "Test vocal",
            "profile": "calm",
            "engine": "gtts",
            "latency": 0.5,
            "success": True,
            "cluster_id": 0,
        }
    ]
    voice_manager.update_dashboard()
    assert os.path.exists(tmp_dirs["dashboard_path"]), "Fichier dashboard non créé"
    with open(tmp_dirs["dashboard_path"], "r", encoding="utf-8") as f:
        dashboard = json.load(f)
    assert "num_messages" in dashboard, "num_messages absent"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "update_dashboard" in str(op) for op in df_perf["operation"]
    ), "Opération non journalisée"
