# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_email_alert.py
# Tests unitaires pour src/utils/email_alert.py
#
# Version : 2.1.5
# Date : 2025-05-14
#
# Rôle : Valide l'envoi d'alertes critiques via Email, incluant l'initialisation,
#        l'envoi d'alertes, et la journalisation des performances avec psutil.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, psutil>=5.9.8,<6.0.0
# - src/utils/email_alert.py
# - src/model/utils/alert_manager.py
# - src/utils/secret_manager.py
#
# Inputs :
# - Données factices pour les alertes Email et les identifiants SMTP.
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de email_alert.py.
#
# Notes :
# - Conforme à structure.txt (version 2.1.5, 2025-05-14).
# - Supporte les suggestions 1 (position sizing dynamique), 4 (HMM/changepoint detection),
#   6 (drift detection) via alert_manager.py.
# - Vérifie les logs psutil dans data/logs/email_alert_performance.csv.

import os
from smtplib import SMTPException
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.utils.email_alert import EmailNotifier


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
        "perf_log_path": str(logs_dir / "email_alert_performance.csv"),
        "log_file_path": str(logs_dir / "email_alert.log"),
    }


@pytest.fixture
def mock_secrets():
    """Mock pour les identifiants SMTP."""
    return {
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "sender_email": "test@example.com",
        "sender_password": "password",
        "receiver_email": "receiver@example.com",
    }


def test_email_notifier_init_success(tmp_dirs, mock_secrets):
    """Teste l'initialisation réussie de EmailNotifier."""
    with patch(
        "src.utils.secret_manager.SecretManager.get_secret", return_value=mock_secrets
    ), patch("src.model.utils.alert_manager.AlertManager.send_alert") as mock_alert:
        notifier = EmailNotifier()
        assert (
            notifier.smtp_server == mock_secrets["smtp_server"]
        ), "SMTP server incorrect"
        assert notifier.smtp_port == mock_secrets["smtp_port"], "SMTP port incorrect"
        assert (
            notifier.sender_email == mock_secrets["sender_email"]
        ), "Sender email incorrect"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "init_email_notifier"
        ), "Log init_email_notifier manquant"
        assert all(
            col in df.columns for col in ["memory_usage_mb", "cpu_usage_percent"]
        ), "Colonnes psutil manquantes"
        mock_alert.assert_not_called()


def test_email_notifier_init_failure(tmp_dirs):
    """Teste l'échec de l'initialisation de EmailNotifier."""
    with patch(
        "src.utils.secret_manager.SecretManager.get_secret", return_value={}
    ), patch("src.model.utils.alert_manager.AlertManager.send_alert") as mock_alert:
        with pytest.raises(ValueError, match="Échec de l’initialisation Email"):
            EmailNotifier()
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "init_email_notifier" and df["success"] == False
        ), "Échec init_email_notifier non logué"
        mock_alert.assert_called_with(pytest.any(str), priority=4)


def test_send_alert_success(tmp_dirs, mock_secrets):
    """Teste l'envoi réussi d'une alerte Email."""
    with patch(
        "src.utils.secret_manager.SecretManager.get_secret", return_value=mock_secrets
    ), patch("smtplib.SMTP") as mock_smtp, patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ) as mock_alert:
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        notifier = EmailNotifier()
        result = notifier.send_alert("Test alert Email", priority=4)
        assert result is True, "L'envoi de l'alerte aurait dû réussir"
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with(
            mock_secrets["sender_email"], mock_secrets["sender_password"]
        )
        mock_server.sendmail.assert_called_once()
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


def test_send_alert_smtp_failure(tmp_dirs, mock_secrets):
    """Teste l'échec de l'envoi d'une alerte Email."""
    with patch(
        "src.utils.secret_manager.SecretManager.get_secret", return_value=mock_secrets
    ), patch("smtplib.SMTP", side_effect=SMTPException("SMTP error")), patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ) as mock_alert:
        notifier = EmailNotifier()
        result = notifier.send_alert("Test alert Email", priority=4)
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
        notifier = EmailNotifier()
        notifier.log_performance(
            "test_op", 0.1, True, message="Test performance", priority=4
        )
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df["operation"] == "test_op"), "Log test_op manquant"
        assert all(
            col in df.columns for col in ["memory_usage_mb", "cpu_usage_percent"]
        ), "Colonnes psutil manquantes"
