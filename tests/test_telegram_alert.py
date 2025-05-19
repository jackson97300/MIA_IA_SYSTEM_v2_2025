# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_telegram_alert.py
# Tests unitaires pour src/utils/telegram_alert.py
#
# Version : 2.1.5
# Date : 2025-05-14
#
# Rôle : Valide l'envoi d'alertes critiques via Telegram, incluant les alertes pour les tailles de position
#        excessives (risk_manager.py), les erreurs/transitions HMM (regime_detector.py), et les détections
#        de sharpe_drift (drift_detector.py). Teste les nouvelles fonctionnalités : logs psutil, intégration
#        avec secret_manager.py pour les identifiants.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, psutil>=5.9.0,<6.0.0, requests>=2.28.0,<3.0.0
# - src/utils/telegram_alert.py
# - src/model/utils/alert_manager.py
# - src/utils/secret_manager.py
#
# Inputs :
# - Données factices pour les alertes Telegram et les identifiants.
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de telegram_alert.py.
#
# Notes :
# - Conforme à structure.txt (version 2.1.5, 2025-05-14).
# - Supporte les suggestions 1 (position sizing dynamique), 4 (HMM/changepoint detection),
#   6 (drift detection).
# - Vérifie les logs psutil dans data/logs/telegram_alert_performance.csv.
# - Policies Note: The official directory for routing policies is src/model/router/policies.
# The src/model/policies directory is a residual and should be verified
# for removal to avoid import conflicts.

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from requests.exceptions import RequestException

from src.utils.telegram_alert import TelegramAlert


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
        "perf_log_path": str(logs_dir / "telegram_alert_performance.csv"),
        "log_file_path": str(logs_dir / "telegram_alert.log"),
    }


@pytest.fixture
def mock_secrets():
    """Mock pour les identifiants Telegram."""
    return {"bot_token": "mock_bot_token", "chat_id": "mock_chat_id"}


def test_telegram_alert_init_success(tmp_dirs, mock_secrets):
    """Teste l'initialisation réussie de TelegramAlert."""
    with patch(
        "src.utils.secret_manager.SecretManager.get_secret", return_value=mock_secrets
    ), patch("src.model.utils.alert_manager.AlertManager.send_alert") as mock_alert:
        telegram_alert = TelegramAlert()
        assert telegram_alert.bot_token == "mock_bot_token", "Bot token incorrect"
        assert telegram_alert.chat_id == "mock_chat_id", "Chat ID incorrect"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "init_telegram_alert"
        ), "Log init_telegram_alert manquant"
        assert all(
            col in df.columns for col in ["memory_usage_mb", "cpu_usage_percent"]
        ), "Colonnes psutil manquantes"
        mock_alert.assert_not_called()


def test_telegram_alert_init_failure(tmp_dirs):
    """Teste l'échec de l'initialisation de TelegramAlert."""
    with patch(
        "src.utils.secret_manager.SecretManager.get_secret", return_value={}
    ), patch("src.model.utils.alert_manager.AlertManager.send_alert") as mock_alert:
        with pytest.raises(ValueError, match="Échec de l’initialisation Telegram"):
            TelegramAlert()
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "init_telegram_alert" and df["success"] == False
        ), "Échec init_telegram_alert non logué"
        mock_alert.assert_called_with(pytest.any(str), priority=4)


def test_send_alert_critical(tmp_dirs, mock_secrets):
    """Teste l'envoi d'une alerte critique (priorité >= 3)."""
    with patch(
        "src.utils.secret_manager.SecretManager.get_secret", return_value=mock_secrets
    ), patch("requests.post") as mock_post, patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ) as mock_alert:
        mock_post.return_value = MagicMock(status_code=200)
        telegram_alert = TelegramAlert()
        telegram_alert.send_alert("Test alert critical", priority=4)
        mock_post.assert_called_with(
            f"https://api.telegram.org/bot{mock_secrets['bot_token']}/sendMessage",
            json={"chat_id": mock_secrets["chat_id"], "text": "Test alert critical"},
            timeout=10,
        )
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "send_alert" and df["priority"] == 4
        ), "Log send_alert (priorité 4) manquant"
        assert all(
            col in df.columns for col in ["memory_usage_mb", "cpu_usage_percent"]
        ), "Colonnes psutil manquantes"
        mock_alert.assert_not_called()


def test_send_alert_non_critical(tmp_dirs, mock_secrets):
    """Teste l'absence d'envoi pour une alerte non critique (priorité < 3)."""
    with patch(
        "src.utils.secret_manager.SecretManager.get_secret", return_value=mock_secrets
    ), patch("requests.post") as mock_post:
        telegram_alert = TelegramAlert()
        telegram_alert.send_alert("Test alert non-critical", priority=2)
        mock_post.assert_not_called()
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "send_alert" and df["priority"] == 2
        ), "Log send_alert (priorité 2) manquant"


def test_send_alert_new_scenarios(tmp_dirs, mock_secrets):
    """Teste les alertes pour les nouveaux scénarios."""
    with patch(
        "src.utils.secret_manager.SecretManager.get_secret", return_value=mock_secrets
    ), patch("requests.post") as mock_post, patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ) as mock_alert:
        mock_post.return_value = MagicMock(status_code=200)
        telegram_alert = TelegramAlert()
        scenarios = [
            ("Position size exceeds 0.1 * capital", 4, "Suggestion 1: risk_manager.py"),
            ("HMM training failed", 4, "Suggestion 4: regime_detector.py"),
            ("HMM regime transition detected", 3, "Suggestion 4: regime_detector.py"),
            (
                "Sharpe drift detected (value: 0.5)",
                3,
                "Suggestion 6: drift_detector.py",
            ),
        ]
        for message, priority, description in scenarios:
            telegram_alert.send_alert(f"{message} - {description}", priority=priority)
            if priority >= 3:
                mock_post.assert_called_with(
                    f"https://api.telegram.org/bot{mock_secrets['bot_token']}/sendMessage",
                    json={
                        "chat_id": mock_secrets["chat_id"],
                        "text": f"{message} - {description}",
                    },
                    timeout=10,
                )
            else:
                mock_post.assert_not_called()
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df["operation"] == "send_alert"), "Log send_alert manquant"
        mock_alert.assert_not_called()


def test_send_alert_request_failure(tmp_dirs, mock_secrets):
    """Teste l'échec de l'envoi d'une alerte."""
    with patch(
        "src.utils.secret_manager.SecretManager.get_secret", return_value=mock_secrets
    ), patch("requests.post", side_effect=RequestException("Network error")), patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ) as mock_alert:
        telegram_alert = TelegramAlert()
        telegram_alert.send_alert("Test alert critical", priority=4)
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
        telegram_alert = TelegramAlert()
        telegram_alert.log_performance(
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
