# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_error_tracker.py
# Tests unitaires pour src/utils/error_tracker.py
#
# Version : 2.1.5
# Date : 2025-05-14
#
# Rôle : Valide la capture des erreurs avec Sentry, incluant les erreurs des modules risk_manager.py,
#        regime_detector.py, trade_probability.py. Teste les nouvelles fonctionnalités :
#        logs psutil, alertes via alert_manager.py.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, psutil>=5.9.0,<6.0.0
# - src/utils/error_tracker.py
# - src/model/utils/alert_manager.py
# - src/utils/secret_manager.py
#
# Inputs :
# - Données factices pour les erreurs Sentry.
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de error_tracker.py.
#
# Notes :
# - Conforme à structure.txt (version 2.1.5, 2025-05-14).
# - Supporte les suggestions 1 (position sizing dynamique), 2 (loggers pour observabilité),
#   4 (HMM/changepoint detection), 7 (Safe RL/CVaR-PPO), 8 (Distributional RL/QR-DQN).
# - Vérifie les logs psutil dans data/logs/error_tracker_performance.csv.
# - Policies Note: The official directory for routing policies is src/model/router/policies.
# The src/model/policies directory is a residual and should be verified
# for removal to avoid import conflicts.

import os
from unittest.mock import patch

import pandas as pd
import pytest

from src.utils.error_tracker import capture_error, init_sentry, log_performance


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
        "perf_log_path": str(logs_dir / "error_tracker_performance.csv"),
        "log_file_path": str(logs_dir / "error_tracker.log"),
    }


def test_init_sentry_success(tmp_dirs):
    """Teste l'initialisation réussie de Sentry."""
    with patch("sentry_sdk.init") as mock_sentry_init, patch(
        "src.utils.secret_manager.SecretManager.get_secret",
        return_value="https://example@sentry.io/123",
    ):
        init_sentry(environment="development")
        mock_sentry_init.assert_called_with(
            dsn="https://example@sentry.io/123",
            environment="development",
            traces_sample_rate=1.0,
            release="MIA_IA_SYSTEM_v2_2025@2.1.5",
        )
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df["operation"] == "init_sentry"), "Log init_sentry manquant"
        assert all(
            col in df.columns for col in ["memory_usage_mb", "cpu_usage_percent"]
        ), "Colonnes psutil manquantes"


def test_init_sentry_failure(tmp_dirs):
    """Teste l'échec de l'initialisation de Sentry."""
    with patch("sentry_sdk.init", side_effect=Exception("Sentry init failed")), patch(
        "src.utils.secret_manager.SecretManager.get_secret",
        return_value="https://example@sentry.io/123",
    ):
        with pytest.raises(ValueError, match="Échec de l’initialisation de Sentry"):
            init_sentry(environment="development")
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "init_sentry" and df["success"] == False
        ), "Échec init_sentry non logué"


def test_capture_error(tmp_dirs):
    """Teste la capture réussie d'une erreur."""
    with patch("sentry_sdk.capture_exception") as mock_sentry, patch(
        "sentry_sdk.set_tag"
    ) as mock_set_tag, patch("sentry_sdk.set_context") as mock_set_context, patch(
        "src.utils.error_tracker.sentry_errors.labels"
    ) as mock_counter, patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ) as mock_alert:
        context = {"market": "ES", "operation": "test_op"}
        try:
            raise ValueError("Test error")
        except ValueError as e:
            capture_error(e, context=context, market="ES", operation="test_op")
            mock_sentry.assert_called()
            mock_set_tag.assert_any_call("market", "ES")
            mock_set_tag.assert_any_call("operation", "test_op")
            mock_set_context.assert_called_with("market", "ES")
            mock_counter.return_value.inc.assert_called()
            mock_alert.assert_called_with(
                "Erreur capturée pour ES (test_op): Test error", priority=4
            )
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df["operation"] == "capture_error"), "Log capture_error manquant"
        assert all(
            col in df.columns for col in ["memory_usage_mb", "cpu_usage_percent"]
        ), "Colonnes psutil manquantes"


def test_capture_error_new_modules(tmp_dirs):
    """Teste la capture des erreurs pour les nouveaux modules."""
    with patch("sentry_sdk.capture_exception") as mock_sentry, patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ) as mock_alert:
        error_scenarios = [
            (
                ValueError("Invalid position sizing"),
                "risk_manager_calc",
                "Suggestion 1: risk_manager.py",
            ),
            (
                RuntimeError("HMM training failed"),
                "hmm_training",
                "Suggestion 4: regime_detector.py",
            ),
            (
                ValueError("PPO training failed"),
                "ppo_training",
                "Suggestion 7: trade_probability.py",
            ),
            (
                RuntimeError("QR-DQN training failed"),
                "qr_dqn_training",
                "Suggestion 8: trade_probability.py",
            ),
        ]
        for exception, operation, description in error_scenarios:
            capture_error(
                exception,
                context={"module": description},
                market="ES",
                operation=operation,
            )
            mock_sentry.assert_called()
            mock_alert.assert_called_with(
                f"Erreur capturée pour ES ({operation}): {str(exception)}", priority=4
            )
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df["operation"] == "capture_error"), "Log capture_error manquant"


def test_capture_error_sentry_failure(tmp_dirs):
    """Teste l'échec de la capture d'erreur par Sentry."""
    with patch(
        "sentry_sdk.capture_exception", side_effect=Exception("Sentry capture failed")
    ), patch("src.model.utils.alert_manager.AlertManager.send_alert") as mock_alert:
        context = {"market": "ES", "operation": "test_op"}
        try:
            raise ValueError("Test error")
        except ValueError as e:
            capture_error(e, context=context, market="ES", operation="test_op")
            mock_alert.assert_called_with(
                "Erreur lors de la capture par Sentry: Sentry capture failed",
                priority=4,
            )
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "capture_error" and df["success"] == False
        ), "Échec capture_error non logué"


def test_log_performance_metrics(tmp_dirs):
    """Teste la journalisation des métriques psutil dans log_performance."""
    log_performance("test_op", 0.1, True, market="ES")
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(df["operation"] == "test_op"), "Log test_op manquant"
    assert all(
        col in df.columns for col in ["memory_usage_mb", "cpu_usage_percent"]
    ), "Colonnes psutil manquantes"
