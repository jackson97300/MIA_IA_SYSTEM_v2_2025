# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_decision_log.py
# Tests unitaires pour src/risk/decision_log.py
# Version : 2.1.3
# Date : 2025-05-13
# Rôle : Valide l’enregistrement des décisions de trading et la structure du CSV decision_log.csv, incluant
# trade_success_prob, confidence_drop_rate (Phase 8), et les métriques
# SHAP (Phase 17).

import json
import os
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.risk.decision_log import DecisionLog


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, snapshots, dashboard, et configuration."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    trading_logs_dir = logs_dir / "trading"
    trading_logs_dir.mkdir()
    snapshot_dir = data_dir / "risk" / "decision_snapshots"
    snapshot_dir.mkdir()
    risk_dir = data_dir / "risk"
    risk_dir.mkdir()

    # Créer market_config.yaml
    config_path = config_dir / "market_config.yaml"
    config_content = {
        "decision_log": {
            "log_level": "detailed",
            "alert_threshold": 0.9,
            "context_features": ["event_volatility_impact", "vix"],
            "max_log_size": 10 * 1024 * 1024,
            "snapshot_count": 10,
            "buffer_size": 100,
            "observation_dims": {"training": 350, "inference": 150},
            "thresholds": {"min_trade_success_prob": 0.7},
        }
    }
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)

    return {
        "config_path": str(config_path),
        "logs_dir": str(logs_dir),
        "trading_logs_dir": str(trading_logs_dir),
        "snapshot_dir": str(snapshot_dir),
        "risk_dir": str(risk_dir),
        "perf_log_path": str(logs_dir / "decision_log_performance.csv"),
        "decision_log_path": str(trading_logs_dir / "decision_log.csv"),
        "dashboard_path": str(risk_dir / "decision_log_dashboard.json"),
    }


@pytest.fixture
def mock_trade_predictor():
    """Mock pour TradeProbabilityPredictor."""

    class MockTradeProbabilityPredictor:
        def predict(self, data):
            return 0.75

    return MockTradeProbabilityPredictor()


@pytest.fixture
def mock_shap():
    """Mock pour calculate_shap."""

    def mock_calculate_shap(data, target, max_features):
        columns = data.columns[:max_features]
        return pd.DataFrame({col: [0.1] for col in columns}, index=data.index)

    return mock_calculate_shap


def test_decision_log_init(tmp_dirs):
    """Teste l’initialisation de DecisionLog."""
    decision_log = DecisionLog(config_path=tmp_dirs["config_path"])
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    assert os.path.exists(tmp_dirs["snapshot_dir"]), "Dossier de snapshots non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert all(
        col in df.columns
        for col in ["timestamp", "operation", "latency", "cpu_percent"]
    ), "Colonnes de performance manquantes"


def test_log_decision_valid(tmp_dirs, mock_trade_predictor, mock_shap):
    """Teste l’enregistrement d’une décision valide avec trade_success_prob, confidence_drop_rate, et SHAP."""
    with patch("src.risk.decision_log.calculate_shap", mock_shap):
        decision_log = DecisionLog(config_path=tmp_dirs["config_path"])
        decision_log.trade_predictor = mock_trade_predictor
        data = pd.DataFrame(
            {
                "timestamp": [datetime.now()],
                "event_volatility_impact": [0.3],
                "vix": [20.0],
                **{
                    f"feat_{i}": [np.random.uniform(0, 1)] for i in range(348)
                },  # 350 features
            }
        )
        decision_log.log_decision(
            trade_id=1,
            decision="execute",
            score=0.85,
            reason="Trade exécuté avec confiance élevée",
            data=data,
            outcome=0.02,
            regime_probs=[0.7, 0.2, 0.1],
        )
        assert os.path.exists(decision_log.decision_log_path), "Fichier CSV non créé"
        df = pd.read_csv(decision_log.decision_log_path)
        expected_columns = [
            "trade_id",
            "timestamp",
            "decision",
            "signal_score",
            "reason",
            "outcome",
            "regime_probs",
            "trade_success_prob",
            "confidence_drop_rate",
            "event_volatility_impact",
            "vix",
        ] + [
            f"shap_{col}" for col in data.columns[:50]
        ]  # SHAP limité à 50 features
        assert (
            list(df.columns)[: len(expected_columns)]
            == expected_columns[: len(df.columns)]
        ), "Colonnes CSV incorrectes"
        assert df.iloc[0]["trade_id"] == 1, "trade_id incorrect"
        assert df.iloc[0]["decision"] == "execute", "decision incorrecte"
        assert df.iloc[0]["signal_score"] == 0.85, "signal_score incorrect"
        assert json.loads(df.iloc[0]["regime_probs"]) == [
            0.7,
            0.2,
            0.1,
        ], "regime_probs incorrect"
        assert df.iloc[0]["trade_success_prob"] == 0.75, "trade_success_prob incorrect"
        assert df.iloc[0]["confidence_drop_rate"] == pytest.approx(
            1.0 - 0.75 / 0.7, 0.01
        ), "confidence_drop_rate incorrect"
        assert all(
            df[f"shap_{col}"].iloc[0] == 0.1 for col in data.columns[:50]
        ), "Métriques SHAP incorrectes"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        snapshot_files = os.listdir(tmp_dirs["snapshot_dir"])
        assert any(
            "snapshot_log_decision" in f and f.endswith(".json") for f in snapshot_files
        ), "Snapshot non créé ou compressé"


def test_log_decision_invalid_data(tmp_dirs):
    """Teste l’enregistrement avec des données invalides."""
    decision_log = DecisionLog(config_path=tmp_dirs["config_path"])
    data = pd.DataFrame(
        {"timestamp": [datetime.now()], "event_volatility_impact": [0.3], "vix": [20.0]}
    )  # Dimension incorrecte
    with pytest.raises(ValueError, match="Nombre de features incorrect"):
        decision_log.log_decision(
            trade_id=1,
            decision="execute",
            score=0.85,
            reason="Trade exécuté",
            data=data,
        )
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "Nombre de features incorrect" in str(e) for e in df["error"].dropna()
    ), "Erreur dimension non loguée"


def test_log_decision_shap_failure(tmp_dirs, mock_trade_predictor):
    """Teste l’enregistrement lorsque calculate_shap échoue."""
    with patch("src.risk.decision_log.calculate_shap", return_value=None):
        decision_log = DecisionLog(config_path=tmp_dirs["config_path"])
        decision_log.trade_predictor = mock_trade_predictor
        data = pd.DataFrame(
            {
                "timestamp": [datetime.now()],
                "event_volatility_impact": [0.3],
                "vix": [20.0],
                **{
                    f"feat_{i}": [np.random.uniform(0, 1)] for i in range(348)
                },  # 350 features
            }
        )
        decision_log.log_decision(
            trade_id=1,
            decision="execute",
            score=0.85,
            reason="Trade exécuté avec confiance élevée",
            data=data,
            outcome=0.02,
            regime_probs=[0.7, 0.2, 0.1],
        )
        assert os.path.exists(decision_log.decision_log_path), "Fichier CSV non créé"
        df = pd.read_csv(decision_log.decision_log_path)
        assert "trade_success_prob" in df.columns, "trade_success_prob manquant"
        assert "confidence_drop_rate" in df.columns, "confidence_drop_rate manquant"
        assert not any(
            col.startswith("shap_") for col in df.columns
        ), "Métriques SHAP présentes malgré échec"
        assert df.iloc[0]["confidence_drop_rate"] == pytest.approx(
            1.0 - 0.75 / 0.7, 0.01
        ), "confidence_drop_rate incorrect"
        snapshot_files = os.listdir(tmp_dirs["snapshot_dir"])
        assert any(
            "snapshot_log_decision" in f and f.endswith(".json") for f in snapshot_files
        ), "Snapshot non créé"
        with open(os.path.join(tmp_dirs["snapshot_dir"], snapshot_files[0]), "r") as f:
            snapshot = json.load(f)
        assert (
            snapshot["data"]["trade_success_prob"] == 0.75
        ), "trade_success_prob absent du snapshot"
        assert snapshot["data"]["confidence_drop_rate"] == pytest.approx(
            1.0 - 0.75 / 0.7, 0.01
        ), "confidence_drop_rate absent du snapshot"
        assert not any(
            k.startswith("shap_") for k in snapshot["data"]
        ), "Métriques SHAP présentes dans snapshot malgré échec"


def test_critical_alerts(tmp_dirs, mock_trade_predictor, mock_shap):
    """Teste les alertes pour décisions critiques (blocage critique, exécution risquée)."""
    with patch("src.risk.decision_log.calculate_shap", mock_shap), patch(
        "src.risk.decision_log.AlertManager.send_alert"
    ) as mock_alert, patch(
        "src.risk.decision_log.send_telegram_alert"
    ) as mock_telegram:
        decision_log = DecisionLog(config_path=tmp_dirs["config_path"])
        decision_log.trade_predictor = mock_trade_predictor
        data = pd.DataFrame(
            {
                "timestamp": [datetime.now()],
                "event_volatility_impact": [0.3],
                "vix": [20.0],
                **{
                    f"feat_{i}": [np.random.uniform(0, 1)] for i in range(348)
                },  # 350 features
            }
        )
        # Test blocage critique (score > 0.9)
        decision_log.log_decision(
            trade_id=1,
            decision="block",
            score=0.95,
            reason="Événement macro imminent",
            data=data,
            regime_probs=[0.1, 0.3, 0.6],
        )
        mock_alert.assert_called_with(pytest.any(str), priority=5)
        mock_telegram.assert_called_with(pytest.any(str))
        # Test exécution risquée (trade_success_prob < 0.5)
        decision_log.trade_predictor.predict = lambda x: 0.4
        decision_log.log_decision(
            trade_id=2,
            decision="execute",
            score=0.85,
            reason="Trade risqué",
            data=data,
            outcome=-0.15,
            regime_probs=[0.7, 0.2, 0.1],
        )
        mock_alert.assert_called_with(pytest.any(str), priority=4)
        mock_telegram.assert_called_with(pytest.any(str))
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"


def test_summarize_decisions_empty(tmp_dirs):
    """Teste le résumé avec aucune décision."""
    decision_log = DecisionLog(config_path=tmp_dirs["config_path"])
    summary = decision_log.summarize_decisions()
    assert summary["num_decisions"] == 0, "num_decisions incorrect"
    assert summary["success_rate"] == 0.0, "success_rate incorrect"
    assert (
        summary["average_trade_success_prob"] == 0.0
    ), "average_trade_success_prob incorrect"
    assert (
        summary["average_confidence_drop_rate"] == 0.0
    ), "average_confidence_drop_rate incorrect"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    snapshot_files = os.listdir(tmp_dirs["snapshot_dir"])
    assert any(
        "snapshot_summarize_decisions" in f and f.endswith(".json")
        for f in snapshot_files
    ), "Snapshot non créé ou compressé"


def test_export_decision_snapshot(tmp_dirs, mock_trade_predictor, mock_shap):
    """Teste l’exportation d’un snapshot de décision avec trade_success_prob, confidence_drop_rate, et SHAP."""
    with patch("src.risk.decision_log.calculate_shap", mock_shap):
        decision_log = DecisionLog(config_path=tmp_dirs["config_path"])
        decision_log.trade_predictor = mock_trade_predictor
        data = pd.DataFrame(
            {
                "timestamp": [datetime.now()],
                "event_volatility_impact": [0.3],
                "vix": [20.0],
                **{
                    f"feat_{i}": [np.random.uniform(0, 1)] for i in range(148)
                },  # 150 SHAP features
            }
        )
        decision_log.log_decision(
            trade_id=1,
            decision="execute",
            score=0.85,
            reason="Trade exécuté",
            data=data,
            regime_probs=[0.7, 0.2, 0.1],
        )
        decision_log.export_decision_snapshot(step=1, data=data)
        snapshot_files = os.listdir(tmp_dirs["snapshot_dir"])
        assert any(
            "decision_step_0001" in f and f.endswith(".json") for f in snapshot_files
        ), "Snapshot non créé ou compressé"
        with open(os.path.join(tmp_dirs["snapshot_dir"], snapshot_files[-1]), "r") as f:
            snapshot = json.load(f)
        assert (
            snapshot["data"]["decisions"][0]["trade_success_prob"] == 0.75
        ), "trade_success_prob absent du snapshot"
        assert snapshot["data"]["decisions"][0][
            "confidence_drop_rate"
        ] == pytest.approx(
            1.0 - 0.75 / 0.7, 0.01
        ), "confidence_drop_rate absent du snapshot"
        assert any(
            k.startswith("shap_") for k in snapshot["data"]["decisions"][0]
        ), "Métriques SHAP absentes du snapshot"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"


def test_save_dashboard_status(tmp_dirs, mock_trade_predictor, mock_shap):
    """Teste la sauvegarde du dashboard avec trade_success_prob et confidence_drop_rate."""
    with patch("src.risk.decision_log.calculate_shap", mock_shap):
        decision_log = DecisionLog(config_path=tmp_dirs["config_path"])
        decision_log.trade_predictor = mock_trade_predictor
        data = pd.DataFrame(
            {
                "timestamp": [datetime.now()],
                "event_volatility_impact": [0.3],
                "vix": [20.0],
                **{
                    f"feat_{i}": [np.random.uniform(0, 1)] for i in range(348)
                },  # 350 features
            }
        )
        decision_log.log_decision(
            trade_id=1,
            decision="execute",
            score=0.85,
            reason="Trade exécuté",
            data=data,
            regime_probs=[0.7, 0.2, 0.1],
        )
        decision_log.save_dashboard_status()
        assert os.path.exists(decision_log.dashboard_path), "Fichier dashboard non créé"
        with open(decision_log.dashboard_path, "r") as f:
            status = json.load(f)
        assert (
            status["last_decision"]["trade_success_prob"] == 0.75
        ), "trade_success_prob absent du dashboard"
        assert (
            status["average_trade_success_prob"] == 0.75
        ), "average_trade_success_prob incorrect"
        assert status["average_confidence_drop_rate"] == pytest.approx(
            1.0 - 0.75 / 0.7, 0.01
        ), "average_confidence_drop_rate incorrect"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"


def test_save_snapshot_compressed(tmp_dirs, mock_trade_predictor, mock_shap):
    """Teste la sauvegarde d’un snapshot compressé avec gzip."""
    with patch("src.risk.decision_log.calculate_shap", mock_shap):
        decision_log = DecisionLog(config_path=tmp_dirs["config_path"])
        decision_log.trade_predictor = mock_trade_predictor
        data = pd.DataFrame(
            {
                "timestamp": [datetime.now()],
                "event_volatility_impact": [0.3],
                "vix": [20.0],
                **{
                    f"feat_{i}": [np.random.uniform(0, 1)] for i in range(348)
                },  # 350 features
            }
        )
        decision_log.log_decision(
            trade_id=1,
            decision="execute",
            score=0.85,
            reason="Trade exécuté",
            data=data,
            regime_probs=[0.7, 0.2, 0.1],
        )
        snapshot_data = {"test": "compressed_snapshot"}
        decision_log.save_snapshot("test_compressed", snapshot_data, compress=True)
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
        assert (
            snapshot["data"] == snapshot_data
        ), "Contenu du snapshot compressé incorrect"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
