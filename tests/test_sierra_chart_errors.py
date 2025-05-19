# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_sierra_chart_errors.py
# Tests unitaires pour src/risk/sierra_chart_errors.py
# Version : 2.1.3
# Date : 2025-05-13
# Rôle : Valide la gestion des erreurs de l’API Teton, incluant l’enregistrement dans sierra_errors.csv,
#        confidence_drop_rate (Phase 8), snapshots, et alertes Telegram.

import gzip
import json
import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.risk.sierra_chart_errors import SierraChartErrorManager


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs et snapshots."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs" / "trading"
    logs_dir.mkdir(parents=True)
    snapshot_dir = data_dir / "risk_snapshots"
    snapshot_dir.mkdir()

    return {
        "logs_dir": str(logs_dir),
        "snapshot_dir": str(snapshot_dir),
        "error_log_path": str(logs_dir / "sierra_errors.csv"),
        "perf_log_path": str(logs_dir / "sierra_errors_performance.csv"),
    }


@pytest.fixture
def error_manager(tmp_dirs):
    """Crée une instance de SierraChartErrorManager avec un chemin temporaire."""
    return SierraChartErrorManager(log_csv_path=Path(tmp_dirs["error_log_path"]))


def test_sierra_chart_error_manager_init(tmp_dirs, error_manager):
    """Teste l’initialisation de SierraChartErrorManager."""
    assert os.path.exists(tmp_dirs["error_log_path"]), "Fichier de log CSV non créé"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de log de performance non créé"
    assert os.path.exists(tmp_dirs["snapshot_dir"]), "Dossier de snapshots non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert all(
        col in df.columns
        for col in ["timestamp", "operation", "latency", "cpu_percent"]
    ), "Colonnes de performance manquantes"
    df = pd.read_csv(tmp_dirs["error_log_path"])
    assert all(
        col in df.columns
        for col in [
            "timestamp",
            "error_code",
            "message",
            "trade_id",
            "severity",
            "confidence_drop_rate",
        ]
    ), "Colonnes CSV incorrectes"


def test_log_error_valid(tmp_dirs, error_manager):
    """Teste l’enregistrement d’une erreur valide avec confidence_drop_rate."""
    with patch(
        "src.risk.sierra_chart_errors.AlertManager.send_alert"
    ) as mock_alert, patch(
        "src.risk.sierra_chart_errors.send_telegram_alert"
    ) as mock_telegram:
        error_manager.log_error(
            error_code="TETON_1001",
            message="Connexion à l'API Teton refusée",
            trade_id="TRADE_123",
            severity="CRITICAL",
        )
        assert os.path.exists(tmp_dirs["error_log_path"]), "Fichier CSV non créé"
        df = pd.read_csv(tmp_dirs["error_log_path"])
        assert df.iloc[0]["error_code"] == "TETON_1001", "error_code incorrect"
        assert df.iloc[0]["trade_id"] == "TRADE_123", "trade_id incorrect"
        assert df.iloc[0]["severity"] == "CRITICAL", "severity incorrect"
        assert (
            df.iloc[0]["confidence_drop_rate"] == 0.1
        ), "confidence_drop_rate incorrect"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de log de performance non créé"
        snapshot_files = os.listdir(tmp_dirs["snapshot_dir"])
        assert any(
            "snapshot_sierra_error" in f and f.endswith(".json") for f in snapshot_files
        ), "Snapshot non créé ou compressé"
        with open(os.path.join(tmp_dirs["snapshot_dir"], snapshot_files[-1]), "r") as f:
            snapshot = json.load(f)
        assert (
            snapshot["data"]["confidence_drop_rate"] == 0.1
        ), "confidence_drop_rate absent du snapshot"
        mock_alert.assert_called_with(pytest.any(str), priority=5)
        mock_telegram.assert_called_with(pytest.any(str))


def test_log_error_invalid(tmp_dirs, error_manager):
    """Teste l’enregistrement avec des paramètres invalides."""
    with pytest.raises(
        ValueError, match="Le code d'erreur doit être une chaîne non vide"
    ):
        error_manager.log_error(error_code="", message="Erreur test", severity="INFO")
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de log de performance non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "Le code d'erreur doit être une chaîne non vide" in str(e)
        for e in df["error"].dropna()
    ), "Erreur paramètre non loguée"


def test_critical_alerts(tmp_dirs, error_manager):
    """Teste les alertes Telegram pour les erreurs critiques."""
    with patch(
        "src.risk.sierra_chart_errors.AlertManager.send_alert"
    ) as mock_alert, patch(
        "src.risk.sierra_chart_errors.send_telegram_alert"
    ) as mock_telegram:
        # Simuler plusieurs erreurs critiques pour augmenter confidence_drop_rate
        for i in range(5):
            error_manager.log_error(
                error_code=f"TETON_100{i+1}",
                message=f"Erreur critique {i+1}",
                trade_id=f"TRADE_12{i+1}",
                severity="CRITICAL",
            )
        assert os.path.exists(tmp_dirs["error_log_path"]), "Fichier CSV non créé"
        df = pd.read_csv(tmp_dirs["error_log_path"])
        assert (
            df.iloc[-1]["confidence_drop_rate"] == 0.5
        ), "confidence_drop_rate incorrect"
        mock_alert.assert_called_with(pytest.any(str), priority=5)
        mock_telegram.assert_called_with(pytest.any(str))
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de log de performance non créé"
        snapshot_files = os.listdir(tmp_dirs["snapshot_dir"])
        assert (
            len([f for f in snapshot_files if "snapshot_sierra_error" in f]) == 5
        ), "Snapshots non créés pour erreurs critiques"


def test_save_snapshot_compressed(tmp_dirs, error_manager):
    """Teste la sauvegarde d’un snapshot compressé avec gzip."""
    snapshot_data = {"test": "compressed_snapshot"}
    error_manager.save_snapshot("test_compressed", snapshot_data, compress=True)
    snapshot_files = os.listdir(tmp_dirs["snapshot_dir"])
    assert any(
        "snapshot_test_compressed" in f and f.endswith(".json.gz")
        for f in snapshot_files
    ), "Snapshot compressé non créé"
    with gzip.open(
        os.path.join(tmp_dirs["snapshot_dir"], snapshot_files[-1]),
        "rt",
        encoding="utf-8",
    ) as f:
        snapshot = json.load(f)
    assert snapshot["data"] == snapshot_data, "Contenu du snapshot compressé incorrect"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de log de performance non créé"


def test_log_performance(tmp_dirs, error_manager):
    """Teste l’enregistrement des performances."""
    error_manager.log_error(
        error_code="TETON_3003", message="Délai d'exécution dépassé", severity="INFO"
    )
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de log de performance non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert all(
        col in df.columns
        for col in ["timestamp", "operation", "latency", "cpu_percent"]
    ), "Colonnes de performance manquantes"
    assert any(df["operation"] == "log_error"), "Opération log_error non journalisée"
    assert any(df["confidence_drop_rate"] >= 0), "confidence_drop_rate non journalisé"


def test_sigint_handling(tmp_dirs, error_manager):
    """Teste la gestion des interruptions SIGINT."""
    with patch("src.risk.sierra_chart_errors.sys.exit") as mock_exit, patch(
        "src.risk.sierra_chart_errors.AlertManager.send_alert"
    ) as mock_alert, patch(
        "src.risk.sierra_chart_errors.send_telegram_alert"
    ) as mock_telegram:
        error_manager.handle_sigint(signal.SIGINT, None)
        snapshot_files = os.listdir(tmp_dirs["snapshot_dir"])
        assert any(
            "snapshot_sigint" in f and f.endswith(".json") for f in snapshot_files
        ), "Snapshot SIGINT non créé"
        with open(os.path.join(tmp_dirs["snapshot_dir"], snapshot_files[-1]), "r") as f:
            snapshot = json.load(f)
        assert (
            snapshot["data"]["status"] == "SIGINT"
        ), "Contenu snapshot SIGINT incorrect"
        mock_alert.assert_called_with(pytest.any(str), priority=2)
        mock_telegram.assert_called_with(pytest.any(str))
        mock_exit.assert_called_with(0)
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de log de performance non créé"
