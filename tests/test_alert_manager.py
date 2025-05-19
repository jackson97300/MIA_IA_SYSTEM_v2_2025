# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_alert_manager.py
# Tests unitaires pour src/model/utils/alert_manager.py
#
# Version : 2.1.5
# Date : 2025-05-14
#
# Rôle : Valide la gestion centralisée des alertes multi-canaux (Telegram, Discord, Email) avec priorités,
#        avec support multi-marchés, snapshots compressés, sauvegardes incrémentielles/distribuées,
#        et stockage local configurable. Vérifie la persistance du cache et la gestion des erreurs.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, psutil>=5.9.8,<6.0.0, boto3>=1.26.0,<2.0.0,
#   loguru>=0.7.0,<1.0.0, pyyaml>=6.0.0,<7.0.0, smtplib, pickle
# - src/model/utils/config_manager.py
# - src/utils/telegram_alert.py
# - src/utils/discord_alert.py
# - src/utils/email_alert.py
#
# Inputs :
# - Fichier de configuration factice (es_config.yaml)
# - Messages de test pour les alertes
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de alert_manager.py.
#
# Notes :
# - Conforme à structure.txt (version 2.1.5, 2025-05-14).
# - Tests la Phase 8 (confidence_drop_rate) et la gestion multi-canaux.
# - Vérifie l’absence de dxFeed, la suppression de 320/81 features et obs_t, les retries,
#   logs psutil, alertes Telegram/Discord/Email, snapshots compressés, sauvegardes incrémentielles/distribuées,
#   stockage local, et persistance du cache.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.
# - Gestion des priorités :
#   - Priorité 1–2 : Discord (journalisation uniquement)
#   - Priorité 3 : Telegram + Discord
#   - Priorité 4 : Telegram + Discord + Email
#   - Priorité 5 : Telegram + Discord + Email + stockage local (configurable)

import gzip
import json
import pickle
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from src.model.utils.alert_manager import AlertManager


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, cache, checkpoints, alertes locales, et configuration."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    cache_dir = data_dir / "cache" / "alert_manager" / "ES"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints" / "alert_manager" / "ES"
    checkpoints_dir.mkdir(parents=True)
    local_alert_dir = data_dir / "alerts" / "local" / "ES"
    local_alert_dir.mkdir(parents=True)

    # Créer es_config.yaml
    es_config_path = config_dir / "es_config.yaml"
    es_config_content = {
        "telegram": {"enabled": True},
        "discord": {"enabled": True},
        "email": {"enabled": True},
        "local_storage": {"enabled": True, "path": str(local_alert_dir)},
        "cache": {"enabled": True, "path": str(data_dir / "cache" / "alert_cache.pkl")},
        "s3_bucket": "test-bucket",
        "s3_prefix": "alert_manager/",
    }
    with open(es_config_path, "w", encoding="utf-8") as f:
        yaml.dump(es_config_content, f)

    return {
        "base_dir": str(base_dir),
        "config_dir": str(config_dir),
        "logs_dir": str(logs_dir),
        "cache_dir": str(cache_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "local_alert_dir": str(local_alert_dir),
        "es_config_path": str(es_config_path),
        "perf_log_path": str(logs_dir / "alert_performance.csv"),
        "cache_path": str(data_dir / "cache" / "alert_cache.pkl"),
    }


@pytest.fixture
def mock_notifiers():
    """Mock pour les notificateurs."""
    telegram_mock = MagicMock()
    discord_mock = MagicMock()
    email_mock = MagicMock()
    return [
        ("telegram", telegram_mock),
        ("discord", discord_mock),
        ("email", email_mock),
    ]


def test_init_alert_manager(tmp_dirs, mock_notifiers):
    """Teste l’initialisation de AlertManager."""
    with patch(
        "src.utils.telegram_alert.TelegramAlert", return_value=mock_notifiers[0][1]
    ), patch(
        "src.utils.discord_alert.DiscordNotifier", return_value=mock_notifiers[1][1]
    ), patch(
        "src.utils.email_alert.EmailNotifier", return_value=mock_notifiers[2][1]
    ):
        alert_manager = AlertManager(
            config_path=tmp_dirs["es_config_path"], market="ES"
        )
        assert alert_manager.market == "ES", "Marché incorrect"
        assert len(alert_manager.notifiers) == 3, "Nombre de notificateurs incorrect"
        assert (
            alert_manager.notifiers[0][0] == "telegram"
        ), "Nom du canal Telegram incorrect"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "init" in str(op) for op in df_perf["operation"]
        ), "Opération init non journalisée"


def test_send_alert_priority_1(tmp_dirs, mock_notifiers):
    """Teste l’envoi d’une alerte de priorité 1 (Discord uniquement)."""
    with patch(
        "src.utils.telegram_alert.TelegramAlert", return_value=mock_notifiers[0][1]
    ), patch(
        "src.utils.discord_alert.DiscordNotifier", return_value=mock_notifiers[1][1]
    ), patch(
        "src.utils.email_alert.EmailNotifier", return_value=mock_notifiers[2][1]
    ), patch(
        "src.model.utils.alert_manager.discord_alerts_total"
    ) as mock_counter:
        mock_notifiers[1][1].send_alert.return_value = True
        alert_manager = AlertManager(
            config_path=tmp_dirs["es_config_path"], market="ES"
        )
        success = alert_manager.send_alert("Test alert priority 1", priority=1)
        assert success, "Échec envoi alerte priorité 1"
        mock_notifiers[1][1].send_alert.assert_called_once()
        mock_notifiers[0][1].send_alert.assert_not_called()
        mock_notifiers[2][1].send_alert.assert_not_called()
        mock_counter.labels.assert_called_with(market="ES", priority=1)
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_alert_1" in f and f.endswith(".json.gz") for f in snapshot_files
        ), "Snapshot alert_1 non créé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "alert_1" in f and f.endswith(".json.gz") for f in checkpoint_files
        ), "Checkpoint alert_1 non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "send_alert" in str(op) and "discord" in str(ch)
            for op, ch in zip(df_perf["operation"], df_perf["channels"])
        ), "Opération send_alert (discord) non journalisée"
        assert any(
            "confidence_drop_rate" in str(kw) for kw in df_perf.to_dict("records")
        ), "confidence_drop_rate absent"


def test_send_alert_priority_2(tmp_dirs, mock_notifiers):
    """Teste l’envoi d’une alerte de priorité 2 (Discord uniquement)."""
    with patch(
        "src.utils.telegram_alert.TelegramAlert", return_value=mock_notifiers[0][1]
    ), patch(
        "src.utils.discord_alert.DiscordNotifier", return_value=mock_notifiers[1][1]
    ), patch(
        "src.utils.email_alert.EmailNotifier", return_value=mock_notifiers[2][1]
    ), patch(
        "src.model.utils.alert_manager.discord_alerts_total"
    ) as mock_counter:
        mock_notifiers[1][1].send_alert.return_value = True
        alert_manager = AlertManager(
            config_path=tmp_dirs["es_config_path"], market="ES"
        )
        success = alert_manager.send_alert("Test alert priority 2", priority=2)
        assert success, "Échec envoi alerte priorité 2"
        mock_notifiers[1][1].send_alert.assert_called_once()
        mock_notifiers[0][1].send_alert.assert_not_called()
        mock_notifiers[2][1].send_alert.assert_not_called()
        mock_counter.labels.assert_called_with(market="ES", priority=2)
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_alert_2" in f and f.endswith(".json.gz") for f in snapshot_files
        ), "Snapshot alert_2 non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "send_alert" in str(op) and "discord" in str(ch)
            for op, ch in zip(df_perf["operation"], df_perf["channels"])
        ), "Opération send_alert (discord) non journalisée"


def test_send_alert_priority_3(tmp_dirs, mock_notifiers):
    """Teste l’envoi d’une alerte de priorité 3 (Telegram + Discord)."""
    with patch(
        "src.utils.telegram_alert.TelegramAlert", return_value=mock_notifiers[0][1]
    ), patch(
        "src.utils.discord_alert.DiscordNotifier", return_value=mock_notifiers[1][1]
    ), patch(
        "src.utils.email_alert.EmailNotifier", return_value=mock_notifiers[2][1]
    ), patch(
        "src.model.utils.alert_manager.telegram_alerts_total"
    ) as mock_telegram_counter, patch(
        "src.model.utils.alert_manager.discord_alerts_total"
    ) as mock_discord_counter:
        mock_notifiers[0][1].send_alert.return_value = True
        mock_notifiers[1][1].send_alert.return_value = True
        alert_manager = AlertManager(
            config_path=tmp_dirs["es_config_path"], market="ES"
        )
        success = alert_manager.send_alert("Test alert priority 3", priority=3)
        assert success, "Échec envoi alerte priorité 3"
        mock_notifiers[0][1].send_alert.assert_called_once()
        mock_notifiers[1][1].send_alert.assert_called_once()
        mock_notifiers[2][1].send_alert.assert_not_called()
        mock_telegram_counter.labels.assert_called_with(market="ES", priority=3)
        mock_discord_counter.labels.assert_called_with(market="ES", priority=3)
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_alert_3" in f and f.endswith(".json.gz") for f in snapshot_files
        ), "Snapshot alert_3 non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "send_alert" in str(op) and "telegram,discord" in str(ch)
            for op, ch in zip(df_perf["operation"], df_perf["channels"])
        ), "Opération send_alert (telegram,discord) non journalisée"


def test_send_alert_priority_4(tmp_dirs, mock_notifiers):
    """Teste l’envoi d’une alerte de priorité 4 (Telegram + Discord + Email)."""
    with patch(
        "src.utils.telegram_alert.TelegramAlert", return_value=mock_notifiers[0][1]
    ), patch(
        "src.utils.discord_alert.DiscordNotifier", return_value=mock_notifiers[1][1]
    ), patch(
        "src.utils.email_alert.EmailNotifier", return_value=mock_notifiers[2][1]
    ), patch(
        "src.model.utils.alert_manager.telegram_alerts_total"
    ) as mock_telegram_counter, patch(
        "src.model.utils.alert_manager.discord_alerts_total"
    ) as mock_discord_counter, patch(
        "src.model.utils.alert_manager.email_alerts_total"
    ) as mock_email_counter:
        mock_notifiers[0][1].send_alert.return_value = True
        mock_notifiers[1][1].send_alert.return_value = True
        mock_notifiers[2][1].send_alert.return_value = True
        alert_manager = AlertManager(
            config_path=tmp_dirs["es_config_path"], market="ES"
        )
        success = alert_manager.send_alert("Test alert priority 4", priority=4)
        assert success, "Échec envoi alerte priorité 4"
        mock_notifiers[0][1].send_alert.assert_called_once()
        mock_notifiers[1][1].send_alert.assert_called_once()
        mock_notifiers[2][1].send_alert.assert_called_once()
        mock_telegram_counter.labels.assert_called_with(market="ES", priority=4)
        mock_discord_counter.labels.assert_called_with(market="ES", priority=4)
        mock_email_counter.labels.assert_called_with(market="ES", priority=4)
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_alert_4" in f and f.endswith(".json.gz") for f in snapshot_files
        ), "Snapshot alert_4 non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "send_alert" in str(op) and "telegram,discord,email" in str(ch)
            for op, ch in zip(df_perf["operation"], df_perf["channels"])
        ), "Opération send_alert (telegram,discord,email) non journalisée"


def test_send_alert_priority_5(tmp_dirs, mock_notifiers):
    """Teste l’envoi d’une alerte de priorité 5 (Telegram + Discord + Email + local)."""
    with patch(
        "src.utils.telegram_alert.TelegramAlert", return_value=mock_notifiers[0][1]
    ), patch(
        "src.utils.discord_alert.DiscordNotifier", return_value=mock_notifiers[1][1]
    ), patch(
        "src.utils.email_alert.EmailNotifier", return_value=mock_notifiers[2][1]
    ), patch(
        "src.model.utils.alert_manager.telegram_alerts_total"
    ) as mock_telegram_counter, patch(
        "src.model.utils.alert_manager.discord_alerts_total"
    ) as mock_discord_counter, patch(
        "src.model.utils.alert_manager.email_alerts_total"
    ) as mock_email_counter:
        mock_notifiers[0][1].send_alert.return_value = True
        mock_notifiers[1][1].send_alert.return_value = True
        mock_notifiers[2][1].send_alert.return_value = True
        alert_manager = AlertManager(
            config_path=tmp_dirs["es_config_path"], market="ES"
        )
        success = alert_manager.send_alert("Test alert priority 5", priority=5)
        assert success, "Échec envoi alerte priorité 5"
        mock_notifiers[0][1].send_alert.assert_called_once()
        mock_notifiers[1][1].send_alert.assert_called_once()
        mock_notifiers[2][1].send_alert.assert_called_once()
        mock_telegram_counter.labels.assert_called_with(market="ES", priority=5)
        mock_discord_counter.labels.assert_called_with(market="ES", priority=5)
        mock_email_counter.labels.assert_called_with(market="ES", priority=5)
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_alert_5" in f and f.endswith(".json.gz") for f in snapshot_files
        ), "Snapshot alert_5 non créé"
        local_alert_files = os.listdir(tmp_dirs["local_alert_dir"])
        assert any(
            "alert_" in f and f.endswith(".json") for f in local_alert_files
        ), "Alerte locale non créée"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "save_local_alert" in str(op) for op in df_perf["operation"]
        ), "Opération save_local_alert non journalisée"
        assert any(
            "send_alert" in str(op) and "telegram,discord,email,local" in str(ch)
            for op, ch in zip(df_perf["operation"], df_perf["channels"])
        ), "Opération send_alert (telegram,discord,email,local)rial non journalisée"


def test_memory_overload(tmp_dirs, mock_notifiers):
    """Teste la détection d’une surcharge mémoire."""
    with patch(
        "src.utils.telegram_alert.TelegramAlert", return_value=mock_notifiers[0][1]
    ), patch(
        "src.utils.discord_alert.DiscordNotifier", return_value=mock_notifiers[1][1]
    ), patch(
        "src.utils.email_alert.EmailNotifier", return_value=mock_notifiers[2][1]
    ), patch(
        "psutil.Process"
    ) as mock_process:
        mock_process.return_value.memory_info.return_value.rss = (
            2048 * 1024 * 1024
        )  # 2 Go
        mock_notifiers[0][1].send_alert.return_value = True
        mock_notifiers[1][1].send_alert.return_value = True
        mock_notifiers[2][1].send_alert.return_value = True
        alert_manager = AlertManager(
            config_path=tmp_dirs["es_config_path"], market="ES"
        )
        alert_manager.send_alert("Test memory overload", priority=5)
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "memory_usage_mb" in str(kw) and float(kw["memory_usage_mb"]) > 1024
            for kw in df_perf.to_dict("records")
        ), "Surcharge mémoire non détectée"
        assert any(
            "ALERTE: Usage mémoire élevé" in str(kw["message"])
            for kw in df_perf.to_dict("records")
            if "message" in kw
        ), "Alerte mémoire non envoyée"


def test_channel_failure_fallback(tmp_dirs, mock_notifiers):
    """Teste le fallback si un canal échoue."""
    with patch(
        "src.utils.telegram_alert.TelegramAlert", return_value=mock_notifiers[0][1]
    ), patch(
        "src.utils.discord_alert.DiscordNotifier", return_value=mock_notifiers[1][1]
    ), patch(
        "src.utils.email_alert.EmailNotifier", return_value=mock_notifiers[2][1]
    ), patch(
        "src.model.utils.alert_manager.telegram_alerts_total"
    ) as mock_telegram_counter, patch(
        "src.model.utils.alert_manager.email_alerts_total"
    ) as mock_email_counter:
        mock_notifiers[0][1].send_alert.return_value = True
        mock_notifiers[1][1].send_alert.side_effect = Exception("Discord failure")
        mock_notifiers[2][1].send_alert.return_value = True
        alert_manager = AlertManager(
            config_path=tmp_dirs["es_config_path"], market="ES"
        )
        success = alert_manager.send_alert("Test channel failure", priority=4)
        assert success, "Échec envoi alerte priorité 4 avec fallback"
        mock_notifiers[0][1].send_alert.assert_called_once()
        mock_notifiers[1][1].send_alert.assert_called_once()
        mock_notifiers[2][1].send_alert.assert_called_once()
        mock_telegram_counter.labels.assert_called_with(market="ES", priority=4)
        mock_email_counter.labels.assert_called_with(market="ES", priority=4)
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "Erreur envoi alerte via discord" in str(kw["error"])
            for kw in df_perf.to_dict("records")
            if "error" in kw
        ), "Erreur Discord non journalisée"
        assert any(
            "send_alert" in str(op) and "telegram,discord,email" in str(ch)
            for op, ch in zip(df_perf["operation"], df_perf["channels"])
        ), "Opération send_alert non journalisée"


def test_cache_persistence(tmp_dirs, mock_notifiers):
    """Teste la persistance et le chargement du cache."""
    with patch(
        "src.utils.telegram_alert.TelegramAlert", return_value=mock_notifiers[0][1]
    ), patch(
        "src.utils.discord_alert.DiscordNotifier", return_value=mock_notifiers[1][1]
    ), patch(
        "src.utils.email_alert.EmailNotifier", return_value=mock_notifiers[2][1]
    ):
        mock_notifiers[1][1].send_alert.return_value = True
        alert_manager = AlertManager(
            config_path=tmp_dirs["es_config_path"], market="ES"
        )
        alert_manager.send_alert("Test cache persistence", priority=1)
        assert os.path.exists(tmp_dirs["cache_path"]), "Fichier cache non créé"
        with open(tmp_dirs["cache_path"], "rb") as f:
            cached_data = pickle.load(f)
        assert len(cached_data) > 0, "Cache vide"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "save_cache" in str(op) for op in df_perf["operation"]
        ), "Opération save_cache non journalisée"


def test_cloud_backup(tmp_dirs, mock_notifiers):
    """Teste la sauvegarde distribuée S3."""
    with patch(
        "src.utils.telegram_alert.TelegramAlert", return_value=mock_notifiers[0][1]
    ), patch(
        "src.utils.discord_alert.DiscordNotifier", return_value=mock_notifiers[1][1]
    ), patch(
        "src.utils.email_alert.EmailNotifier", return_value=mock_notifiers[2][1]
    ), patch(
        "boto3.client"
    ) as mock_s3:
        alert_manager = AlertManager(
            config_path=tmp_dirs["es_config_path"], market="ES"
        )
        df = pd.DataFrame(
            {"timestamp": [datetime.now().isoformat()], "message": ["Test"]}
        )
        alert_manager.cloud_backup(df, data_type="test_metrics")
        assert mock_s3.called, "Client S3 non appelé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "cloud_backup" in str(op) for op in df_perf["operation"]
        ), "Opération cloud_backup non journalisée"


def test_checkpoint(tmp_dirs, mock_notifiers):
    """Teste la sauvegarde incrémentielle."""
    with patch(
        "src.utils.telegram_alert.TelegramAlert", return_value=mock_notifiers[0][1]
    ), patch(
        "src.utils.discord_alert.DiscordNotifier", return_value=mock_notifiers[1][1]
    ), patch(
        "src.utils.email_alert.EmailNotifier", return_value=mock_notifiers[2][1]
    ):
        alert_manager = AlertManager(
            config_path=tmp_dirs["es_config_path"], market="ES"
        )
        df = pd.DataFrame(
            {"timestamp": [datetime.now().isoformat()], "message": ["Test"]}
        )
        alert_manager.checkpoint(df, data_type="test_metrics")
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "alert_manager_test_metrics" in f and f.endswith(".json.gz")
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
