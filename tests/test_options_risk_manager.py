# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_options_risk_manager.py
# Tests unitaires pour src/risk/options_risk_manager.py
# Version : 2.1.3
# Date : 2025-05-13
# Rôle : Valide le calcul des métriques de risque des options (gamma_exposure, iv_sensitivity, risk_alert),
# confidence_drop_rate (Phase 8), analyse SHAP (Phase 17), et la
# sauvegarde des snapshots/CSV.

import gzip
import json
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.risk.options_risk_manager import OptionsRiskManager


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, snapshots, et configuration."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs" / "trading"
    logs_dir.mkdir(parents=True)
    snapshot_dir = data_dir / "risk_snapshots"
    snapshot_dir.mkdir()

    return {
        "logs_dir": str(logs_dir),
        "snapshot_dir": str(snapshot_dir),
        "perf_log_path": str(logs_dir / "options_risk_performance.csv"),
        "metrics_path": str(data_dir / "risk_snapshots" / "options_risk_metrics.csv"),
    }


@pytest.fixture
def mock_data():
    """Crée des données factices pour tester."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start="2025-05-13 10:00:00", periods=5, freq="H"
            ),
            "strike": [5100.0] * 5,
            "option_type": ["call"] * 3 + ["put"] * 2,
            "implied_volatility": np.array([0.2, 0.25, 0.3, 0.15, 0.28]),
            "gamma": np.array([0.02, 0.03, 0.04, 0.01, 0.05]),
            "delta": np.array([0.4, -0.3, 0.5, -0.2, 0.6]),
            "position_size": np.array([50, -100, 75, -25, 80]),
        }
    )


@pytest.fixture
def mock_shap():
    """Mock pour calculate_shap."""

    def mock_calculate_shap(data, target, max_features):
        columns = data.columns[:max_features]
        return pd.DataFrame(
            {col: [0.1] * len(data) for col in columns}, index=data.index
        )

    return mock_calculate_shap


def test_options_risk_manager_init(tmp_dirs):
    """Teste l’initialisation de OptionsRiskManager."""
    OptionsRiskManager()
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    assert os.path.exists(tmp_dirs["snapshot_dir"]), "Dossier de snapshots non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert all(
        col in df.columns
        for col in ["timestamp", "operation", "latency", "cpu_percent"]
    ), "Colonnes de performance manquantes"


def test_calculate_options_risk_valid(tmp_dirs, mock_data, mock_shap):
    """Teste le calcul des métriques de risque avec des données valides."""
    with patch("src.risk.options_risk_manager.calculate_shap", mock_shap):
        risk_manager = OptionsRiskManager(
            risk_thresholds={
                "gamma_exposure": 1000.0,
                "iv_sensitivity": 0.5,
                "min_confidence": 0.7,
            }
        )
        metrics = risk_manager.calculate_options_risk(mock_data)
        assert set(metrics.keys()) == {
            "gamma_exposure",
            "iv_sensitivity",
            "risk_alert",
            "confidence_drop_rate",
            "shap_metrics",
        }, "Métriques manquantes"
        assert isinstance(
            metrics["gamma_exposure"], pd.Series
        ), "gamma_exposure n’est pas une Series"
        assert isinstance(
            metrics["iv_sensitivity"], pd.Series
        ), "iv_sensitivity n’est pas une Series"
        assert isinstance(
            metrics["risk_alert"], pd.Series
        ), "risk_alert n’est pas une Series"
        assert isinstance(
            metrics["confidence_drop_rate"], pd.Series
        ), "confidence_drop_rate n’est pas une Series"
        assert (
            metrics["risk_alert"].isin([0, 1]).all()
        ), "risk_alert contient des valeurs invalides"
        assert (
            metrics["confidence_drop_rate"].ge(0).all()
        ), "confidence_drop_rate contient des valeurs négatives"
        assert len(metrics["shap_metrics"]) <= 50, "Trop de features SHAP"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        snapshot_files = os.listdir(tmp_dirs["snapshot_dir"])
        assert any(
            "snapshot_options_risk" in f and f.endswith(".json") for f in snapshot_files
        ), "Snapshot non créé ou compressé"
        with open(os.path.join(tmp_dirs["snapshot_dir"], snapshot_files[-1]), "r") as f:
            snapshot = json.load(f)
        assert (
            "confidence_drop_rate" in snapshot["data"]
        ), "confidence_drop_rate absent du snapshot"
        assert "shap_metrics" in snapshot["data"], "shap_metrics absent du snapshot"


def test_calculate_options_risk_invalid_data(tmp_dirs):
    """Teste le calcul avec des données invalides."""
    risk_manager = OptionsRiskManager()
    data = pd.DataFrame(
        {"timestamp": [datetime.now()], "strike": [5100.0]}
    )  # Colonnes manquantes
    with pytest.raises(ValueError, match="Colonnes manquantes"):
        risk_manager.calculate_options_risk(data)
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "Colonnes manquantes" in str(e) for e in df["error"].dropna()
    ), "Erreur colonnes non loguée"


def test_shap_failure(tmp_dirs, mock_data):
    """Teste le calcul lorsque calculate_shap échoue."""
    with patch("src.risk.options_risk_manager.calculate_shap", return_value=None):
        risk_manager = OptionsRiskManager()
        metrics = risk_manager.calculate_options_risk(mock_data)
        assert "shap_metrics" in metrics, "shap_metrics manquant"
        assert not metrics["shap_metrics"], "shap_metrics non vide malgré échec"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        snapshot_files = os.listdir(tmp_dirs["snapshot_dir"])
        assert any(
            "snapshot_options_risk" in f and f.endswith(".json") for f in snapshot_files
        ), "Snapshot non créé"
        with open(os.path.join(tmp_dirs["snapshot_dir"], snapshot_files[-1]), "r") as f:
            snapshot = json.load(f)
        assert (
            snapshot["data"]["shap_metrics"] == {}
        ), "shap_metrics non vide dans snapshot"


def test_save_snapshot_compressed(tmp_dirs, mock_data):
    """Teste la sauvegarde d’un snapshot compressé avec gzip."""
    risk_manager = OptionsRiskManager()
    snapshot_data = {"test": "compressed_snapshot"}
    risk_manager.save_snapshot("test_compressed", snapshot_data, compress=True)
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
    ), "Fichier de logs de performance non créé"


def test_save_risk_metrics(tmp_dirs, mock_data, mock_shap):
    """Teste la sauvegarde des métriques dans un CSV."""
    with patch("src.risk.options_risk_manager.calculate_shap", mock_shap):
        risk_manager = OptionsRiskManager()
        metrics = risk_manager.calculate_options_risk(mock_data)
        risk_manager.save_risk_metrics(metrics, Path(tmp_dirs["metrics_path"]))
        assert os.path.exists(tmp_dirs["metrics_path"]), "Fichier CSV non créé"
        df = pd.read_csv(tmp_dirs["metrics_path"])
        assert all(
            col in df.columns
            for col in [
                "gamma_exposure",
                "iv_sensitivity",
                "risk_alert",
                "confidence_drop_rate",
            ]
        ), "Colonnes CSV incorrectes"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"


def test_critical_alerts(tmp_dirs, mock_data, mock_shap):
    """Teste les alertes pour dépassements de seuils de risque."""
    with patch("src.risk.options_risk_manager.calculate_shap", mock_shap), patch(
        "src.risk.options_risk_manager.AlertManager.send_alert"
    ) as mock_alert, patch(
        "src.risk.options_risk_manager.send_telegram_alert"
    ) as mock_telegram:
        risk_manager = OptionsRiskManager(
            risk_thresholds={
                "gamma_exposure": 0.1,
                "iv_sensitivity": 0.1,
                "min_confidence": 0.7,
            }
        )
        metrics = risk_manager.calculate_options_risk(mock_data)
        mock_alert.assert_called_with(pytest.any(str), priority=4)
        mock_telegram.assert_called_with(pytest.any(str))
        assert metrics["risk_alert"].any(), "risk_alert devrait être activé"
        assert (
            metrics["confidence_drop_rate"].mean() > 0
        ), "confidence_drop_rate devrait être non nul"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
