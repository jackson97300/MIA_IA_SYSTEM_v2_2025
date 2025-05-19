# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_secret_manager.py
# Tests unitaires pour src/utils/secret_manager.py
#
# Version : 2.1.5
# Date : 2025-05-14
#
# Rôle : Valide la récupération des secrets chiffrés via AWS KMS, incluant les identifiants pour IQFeed, AWS, Sentry,
#        Telegram, Discord, Email, risk_manager.py, regime_detector.py, et trade_probability.py.
#        Teste les fonctionnalités : logs psutil, gestion des erreurs avec error_tracker.py, alertes via alert_manager.py.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, psutil>=5.9.8,<6.0.0
# - src/utils/secret_manager.py
# - src/utils/error_tracker.py
# - src/model/utils/alert_manager.py
#
# Inputs :
# - Données factices pour les secrets KMS.
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de secret_manager.py.
#
# Notes :
# - Conforme à structure.txt (version 2.1.5, 2025-05-14).
# - Supporte les suggestions 1 (position sizing dynamique), 4 (HMM/changepoint detection),
#   7 (Safe RL/CVaR-PPO), 8 (Distributional RL/QR-DQN).
# - Vérifie les logs psutil dans data/logs/secret_manager_performance.csv.
# - Tests adaptés pour inclure les nouveaux identifiants (telegram_credentials, discord_credentials, email_credentials).
# - Policies Note: The official directory for routing policies is src/model/router/policies.
# The src/model/policies directory is a residual and should be verified
# for removal to avoid import conflicts.

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from src.utils.secret_manager import SecretManager


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
        "perf_log_path": str(logs_dir / "secret_manager_performance.csv"),
        "log_file_path": str(logs_dir / "secret_manager.log"),
    }


def test_get_secret_generic(tmp_dirs):
    """Teste la récupération d'un secret générique."""
    with patch("boto3.client") as mock_kms, patch(
        "src.utils.secret_manager.AlertManager.send_alert"
    ) as mock_alert:
        mock_kms.return_value.decrypt.return_value = {"Plaintext": b"test-secret"}
        secret_manager = SecretManager()
        result = secret_manager.get_secret("AQICAH...")
        assert isinstance(result, dict), "Résultat n'est pas un dictionnaire"
        assert result == {"secret": "test-secret"}, "Secret générique incorrect"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df["operation"] == "get_secret"), "Log get_secret manquant"
        assert all(
            col in df.columns for col in ["memory_usage_mb", "cpu_usage_percent"]
        ), "Colonnes psutil manquantes"
        mock_alert.assert_called_with("Secret récupéré: AQICAH...", priority=2)


def test_secret_manager_init_success(tmp_dirs):
    """Teste l'initialisation réussie de SecretManager."""
    with patch("boto3.client") as mock_kms, patch(
        "src.utils.secret_manager.AlertManager.send_alert"
    ) as mock_alert:
        mock_kms.return_value = MagicMock()
        secret_manager = SecretManager()
        assert secret_manager.kms is not None, "Client KMS non initialisé"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "init_secret_manager"
        ), "Log init_secret_manager manquant"
        assert all(
            col in df.columns for col in ["memory_usage_mb", "cpu_usage_percent"]
        ), "Colonnes psutil manquantes"
        mock_alert.assert_called_with(
            "Secret récupéré: init_secret_manager...", priority=2
        )


def test_secret_manager_init_failure(tmp_dirs):
    """Teste l'échec de l'initialisation de SecretManager."""
    with patch(
        "boto3.client",
        side_effect=ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access denied"}}, "kms"
        ),
    ), patch("src.utils.error_tracker.capture_error") as mock_capture_error, patch(
        "src.utils.secret_manager.AlertManager.send_alert"
    ) as mock_alert:
        with pytest.raises(ValueError, match="Échec de l’initialisation KMS"):
            SecretManager()
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "init_secret_manager" and df["success"] == False
        ), "Échec init_secret_manager non logué"
        mock_capture_error.assert_called()
        mock_alert.assert_called_with(pytest.any(str), priority=4)


def test_get_secret_success(tmp_dirs):
    """Teste la récupération réussie d'un secret spécifique."""
    with patch("boto3.client") as mock_kms, patch(
        "src.utils.secret_manager.AlertManager.send_alert"
    ) as mock_alert:
        mock_kms.return_value.decrypt.return_value = {"Plaintext": b"test-secret"}
        secret_manager = SecretManager()
        result = secret_manager.get_secret("aws_credentials")
        assert isinstance(result, dict), "Résultat n'est pas un dictionnaire"
        assert result == {
            "access_key": "xxx",
            "secret_key": "yyy",
        }, "Secret aws_credentials incorrect"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df["operation"] == "get_secret"), "Log get_secret manquant"
        assert all(
            col in df.columns for col in ["memory_usage_mb", "cpu_usage_percent"]
        ), "Colonnes psutil manquantes"
        mock_alert.assert_called_with("Secret récupéré: aws_credent...", priority=2)


def test_get_secret_new_identifiers(tmp_dirs):
    """Teste la récupération des nouveaux identifiants."""
    with patch("boto3.client") as mock_kms, patch(
        "src.utils.secret_manager.AlertManager.send_alert"
    ) as mock_alert:
        mock_kms.return_value.decrypt.return_value = {"Plaintext": b"test-secret"}
        secret_manager = SecretManager()
        identifiers = [
            "telegram_credentials",
            "discord_credentials",
            "email_credentials",
            "risk_manager_api_key",
            "regime_detector_data_key",
            "gpu_cluster_key",
            "ray_cluster_key",
        ]
        for secret_id in identifiers:
            result = secret_manager.get_secret(secret_id)
            assert isinstance(result, dict) or isinstance(
                result, str
            ), f"Résultat pour {secret_id} n'est pas valide"
            assert len(result) > 0, f"Secret {secret_id} vide"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df["operation"] == "get_secret"), "Log get_secret manquant"
        mock_alert.assert_called_with(pytest.any(str), priority=2)


def test_get_secret_client_error(tmp_dirs):
    """Teste l'échec de récupération d'un secret dû à une erreur KMS."""
    with patch("boto3.client") as mock_kms, patch(
        "src.utils.error_tracker.capture_error"
    ) as mock_capture_error, patch(
        "src.utils.secret_manager.AlertManager.send_alert"
    ) as mock_alert:
        mock_kms.return_value.decrypt.side_effect = ClientError(
            {
                "Error": {
                    "Code": "InvalidCiphertextException",
                    "Message": "Invalid ciphertext",
                }
            },
            "decrypt",
        )
        secret_manager = SecretManager()
        with pytest.raises(ValueError, match="Erreur KMS"):
            secret_manager.get_secret("invalid_secret")
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "get_secret" and df["success"] == False
        ), "Échec get_secret non logué"
        mock_capture_error.assert_called()
        mock_alert.assert_called_with(pytest.any(str), priority=4)


def test_get_secret_invalid_format(tmp_dirs):
    """Teste l'échec de récupération d'un secret dû à un format invalide."""
    with patch("boto3.client") as mock_kms, patch(
        "src.utils.error_tracker.capture_error"
    ) as mock_capture_error, patch(
        "src.utils.secret_manager.AlertManager.send_alert"
    ) as mock_alert:
        mock_kms.return_value.decrypt.side_effect = TypeError("Invalid format")
        secret_manager = SecretManager()
        with pytest.raises(ValueError, match="secret_id invalide"):
            secret_manager.get_secret("invalid_format")
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "get_secret" and df["success"] == False
        ), "Échec get_secret non logué"
        mock_capture_error.assert_called()
        mock_alert.assert_called_with(pytest.any(str), priority=4)


def test_log_performance_metrics(tmp_dirs):
    """Teste la journalisation des métriques psutil dans log_performance."""
    secret_manager = SecretManager()
    secret_manager.log_performance("test_op", 0.1, True, market="ES")
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(df["operation"] == "test_op"), "Log test_op manquant"
    assert all(
        col in df.columns for col in ["memory_usage_mb", "cpu_usage_percent"]
    ), "Colonnes psutil manquantes"
