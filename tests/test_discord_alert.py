# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_discord_alert.py
# Tests unitaires pour src/utils/discord_alert.py
#
# Version : 2.1.5
# Date : 2025-05-14
#
# Rôle : Valide l'envoi d'alertes critiques via Discord, incluant l'initialisation,
#        l'envoi d'alertes, et la journalisation des performances avec psutil.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, psutil>=5.9.8,<6.0.0, requests>=2.28.0,<3.0.0
# - src/utils/discord_alert.py
# - src/model/utils/alert_manager.py
# - src/utils/secret_manager.py
#
# Inputs :
# - Données factices pour les alertes Discord et les identifiants.
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de discord_alert.py.
#
# Notes :
# - Conforme à structure.txt (version 2.1.5, 2025-05-14).
# - Supporte les suggestions 1 (position sizing dynamique), 4 (HMM/changepoint detection),
#   6 (drift detection) via alert_manager.py.
# - Vérifie les logs psutil dans data/logs/discord_alert_performance.csv.

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from requests.exceptions import RequestException

from src.utils.discord_alert import DiscordNotifier


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    logs_dir = base_dir / "data" / "logs"
    logs_dir.mkdir(parents=True)
    return {
        "base_dir": str(base_dir),
        "logs_dir": str(logs_dir),
        "perf_log_path": str(logs_dir / "discord_alert_performance.csv"),
        "log_file_path": str(logs_dir / "discord_alert.log"),
    }


@pytest.fixture
def mock_secrets():
    """Mock pour les identifiants Discord."""
    return {"webhook_url": "https://discord.com/api/webhooks/mock"}


def test_discord_notifier_init_success(tmp_dirs, mock_secrets):
    """Teste l'initialisation réussie de DiscordNotifier."""
    with patch(
        "src.utils.secret_manager.SecretManager.get_secret", return_value=mock_secrets
    ), patch("src.model.utils.alert_manager.AlertManager.send_alert") as mock_alert:
        notifier = DiscordNotifier()
        assert (
            notifier.webhook_url == mock_secrets["webhook_url"]
        ), "Webhook URL incorrect"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "init_discord_notifier"
        ), "Log init_discord_notifier manquant"
        assert all(
            col in df.columns for col in ["memory_usage_mb", "cpu_usage_percent"]
        ), "Colonnes psutil manquantes"
        mock_alert.assert_not_called()


def test_discord_notifier_init_failure(tmp_dirs):
    """Teste l'échec de l'initialisation de DiscordNotifier."""
    with patch(
        "src.utils.secret_manager.SecretManager.get_secret", return_value={}
    ), patch("src.model.utils.alert_manager.AlertManager.send_alert") as mock_alert:
        with pytest.raises(ValueError, match="Échec de l’initialisation Discord"):
            DiscordNotifier()
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "init_discord_notifier" and df["success"] == False
        ), "Échec init_discord_notifier non logué"
        mock_alert.assert_called_with(pytest.any(str), priority=4)


def test_send_alert_success(tmp_dirs, mock_secrets):
    """Teste l'envoi réussi d'une alerte Discord."""
    with patch(
        "src.utils.secret_manager.SecretManager.get_secret", return_value=mock_secrets
    ), patch("requests.post") as mock_post, patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ) as mock_alert:
        mock_post.return_value = MagicMock(status_code=204)
        notifier = DiscordNotifier()
        result = notifier.send_alert("Test alert Discord", priority=3)
        assert result is True, "L'envoi de l'alerte aurait dû réussir"
        mock_post.assert_called_with(
            mock_secrets["webhook_url"],
            json={"content": "🚨 MIA_IA_SYSTEM Alert (Priority 3): Test alert Discord"},
            timeout=10,
        )
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "send_alert" and df["priority"] == 3
        ), "Log send_alert (priorité 3) manquant"
        assert all(
            col in df.columns for col in ["memory_usage_mb", "cpu_usage_percent"]
        ), "Colonnes psutil manquantes"
        mock_alert.assert_not_called()


def test_send_alert_request_failure(tmp_dirs, mock_secrets):
    """Teste l'échec de l'envoi d'une alerte Discord."""
    with patch(
        "src.utils.secret_manager.SecretManager.get_secret", return_value=mock_secrets
    ), patch("requests.post", side_effect=RequestException("Network error")), patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ) as mock_alert:
        notifier = DiscordNotifier()
        result = notifier.send_alert("Test alert Discord", priority=3)
        assert result is False, "L'envoi de l'alerte aurait dû échouer"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "send_alert" and df["success"] == False
        ), "Échec send_alert non logué"
        mock_alert.assert_called_with(pytest.any(str), priority=4)


def test_log_performance_metrics(tmp_dirs, mock_secrets):
    """Teste la journalisation des métriques psutil dans log_performance."""
    with patch(
        "src.utils.secret_manager.SecretManager.get_secret", return_value=mock_secrets
    ):
        notifier = DiscordNotifier()
        notifier.log_performance(
            "test_op", 0.1, True, message="Test performance", priority=3
        )
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df["operation"] == "test_op"), "Log test_op manquant"
        assert all(
            col in df.columns for col in ["memory_usage_mb", "cpu_usage_percent"]
        ), "Colonnes psutil manquantes"
