# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_risk_controller.py
# Tests unitaires pour src/risk/risk_controller.py
# Version : 2.1.3
# Date : 2025-05-13
# Rôle : Valide la gestion des risques de trading (stop trading, sizing, pénalités IA), incluant
#        market_liquidity_crash_risk, overtrade_risk_score, confidence_drop_rate (Phase 8),
#        et analyse SHAP (Phase 17).

import gzip
import json
import os
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.risk.risk_controller import RiskController


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, snapshots, dashboard, checkpoints, et configuration."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs" / "trading"
    logs_dir.mkdir(parents=True)
    snapshot_dir = data_dir / "risk_snapshots"
    snapshot_dir.mkdir()
    checkpoint_dir = data_dir / "checkpoints"
    checkpoint_dir.mkdir()
    risk_dir = data_dir / "risk"
    risk_dir.mkdir()

    # Créer market_config.yaml
    config_path = config_dir / "market_config.yaml"
    config_content = {
        "risk_controller": {
            "max_drawdown": -0.1,
            "vix_threshold": 30,
            "spread_threshold": 0.05,
            "event_impact_threshold": 0.5,
            "max_position_size": 5,
            "overtrade_risk_threshold": 0.8,
            "liquidity_crash_risk_threshold": 0.8,
            "max_consecutive_losses": 3,
            "penalty_threshold": 0.1,
            "risk_score_threshold": 0.5,
            "predicted_vix_threshold": 25.0,
            "min_confidence": 0.7,
            "spread_weight": 0.5,
            "trade_frequency_weight": 0.3,
            "vix_weight": 0.2,
            "cluster_risk_weight": 0.3,
            "options_risk_weight": 0.2,
            "news_impact_weight": 0.1,
            "buffer_size": 100,
            "observation_dims": {"training": 350, "inference": 150},
        }
    }
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)

    return {
        "config_path": str(config_path),
        "logs_dir": str(logs_dir),
        "snapshot_dir": str(snapshot_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "risk_dir": str(risk_dir),
        "perf_log_path": str(data_dir / "logs" / "risk_performance.csv"),
        "dashboard_path": str(risk_dir / "risk_dashboard.json"),
        "penalty_log_path": str(logs_dir / "penalty_log.csv"),
    }


@pytest.fixture
def mock_data():
    """Crée des données factices pour tester."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start="2025-05-13 10:00:00", periods=5, freq="T"
            ),
            "close": [5100.0] * 5,
            "bid_price_level_1": [5098.0] * 5,
            "ask_price_level_1": [5102.0] * 5,
            "vix": [20.0] * 5,
            "event_volatility_impact": [0.3] * 5,
            "spread_avg_1min": [0.04] * 5,
            "trade_frequency_1s": [8.0] * 5,
            "iv_atm": [0.15] * 5,
            "option_skew": [0.1] * 5,
            "news_impact_score": [0.2] * 5,
            "spoofing_score": [0.3] * 5,
            "volume_anomaly": [0.1] * 5,
            "option_type": ["call"] * 5,
            "neural_regime": ["range"] * 5,
            **{
                f"feat_{i}": [np.random.uniform(0, 1)] for i in range(336)
            },  # 350 features
        }
    )


@pytest.fixture
def mock_positions():
    """Crée des positions factices pour tester."""
    return [
        {
            "timestamp": datetime.now().isoformat(),
            "action": 1.0,
            "price": 5100.0,
            "size": 1,
        }
    ]


@pytest.fixture
def mock_shap():
    """Mock pour calculate_shap."""

    def mock_calculate_shap(data, target, max_features):
        columns = data.columns[:max_features]
        return pd.DataFrame(
            {col: [0.1] * len(data) for col in columns}, index=data.index
        )

    return mock_calculate_shap


@pytest.fixture
def mock_fetch_news():
    """Mock pour fetch_news."""
    return pd.DataFrame({"sentiment_score": [0.2]})


@pytest.fixture
def mock_options_risk():
    """Mock pour OptionsRiskManager.calculate_options_risk."""
    return {
        "gamma_exposure": pd.Series([100.0] * 5),
        "iv_sensitivity": pd.Series([0.3] * 5),
        "risk_alert": pd.Series([0] * 5),
        "confidence_drop_rate": pd.Series([0.0] * 5),
        "shap_metrics": {},
    }


def test_risk_controller_init(tmp_dirs):
    """Teste l’initialisation de RiskController."""
    risk_controller = RiskController(config_path=tmp_dirs["config_path"])
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    assert os.path.exists(tmp_dirs["snapshot_dir"]), "Dossier de snapshots non créé"
    assert os.path.exists(tmp_dirs["checkpoint_dir"]), "Dossier de checkpoints non créé"
    assert os.path.exists(tmp_dirs["risk_dir"]), "Dossier de dashboard non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert all(
        col in df.columns
        for col in ["timestamp", "operation", "latency", "cpu_percent"]
    ), "Colonnes de performance manquantes"


def test_calculate_risk_metrics_valid(
    tmp_dirs, mock_data, mock_positions, mock_shap, mock_fetch_news, mock_options_risk
):
    """Teste le calcul des métriques de risque avec des données valides."""
    with patch("src.risk.risk_controller.calculate_shap", mock_shap), patch(
        "src.risk.risk_controller.fetch_news", return_value=mock_fetch_news
    ), patch(
        "src.risk.risk_controller.OptionsRiskManager.calculate_options_risk",
        return_value=mock_options_risk,
    ):
        risk_controller = RiskController(config_path=tmp_dirs["config_path"])
        metrics = risk_controller.calculate_risk_metrics(mock_data, mock_positions)
        expected_keys = [
            "market_liquidity_crash_risk",
            "overtrade_risk_score",
            "confidence_drop_rate",
            "vix",
            "vix_mean_1h",
            "spread",
            "spread_mean_1h",
            "event_impact",
            "cluster_risk",
            "options_risk_score",
            "news_impact",
        ]
        assert all(key in metrics for key in expected_keys), "Métriques manquantes"
        assert metrics["confidence_drop_rate"] >= 0, "confidence_drop_rate négatif"
        assert any(
            key.startswith("shap_") for key in metrics
        ), "Métriques SHAP absentes"
        assert (
            len([k for k in metrics if k.startswith("shap_")]) <= 50
        ), "Trop de features SHAP"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        snapshot_files = os.listdir(tmp_dirs["snapshot_dir"])
        assert any(
            "snapshot_calculate_risk_metrics" in f and f.endswith(".json")
            for f in snapshot_files
        ), "Snapshot non créé ou compressé"
        with open(os.path.join(tmp_dirs["snapshot_dir"], snapshot_files[-1]), "r") as f:
            snapshot = json.load(f)
        assert (
            "confidence_drop_rate" in snapshot["data"]
        ), "confidence_drop_rate absent du snapshot"
        assert any(
            k.startswith("shap_") for k in snapshot["data"]
        ), "Métriques SHAP absentes du snapshot"


def test_calculate_risk_metrics_invalid_data(tmp_dirs):
    """Teste le calcul avec des données invalides."""
    risk_controller = RiskController(config_path=tmp_dirs["config_path"])
    data = pd.DataFrame(
        {"timestamp": [datetime.now()], "close": [5100.0]}
    )  # Features insuffisants
    positions = [
        {
            "timestamp": datetime.now().isoformat(),
            "action": 1.0,
            "price": 5100.0,
            "size": 1,
        }
    ]
    metrics = risk_controller.calculate_risk_metrics(data, positions)
    assert (
        metrics["market_liquidity_crash_risk"] == 0.0
    ), "market_liquidity_crash_risk non nul malgré erreur"
    assert (
        metrics["overtrade_risk_score"] == 0.0
    ), "overtrade_risk_score non nul malgré erreur"
    assert (
        metrics["confidence_drop_rate"] == 0.0
    ), "confidence_drop_rate non nul malgré erreur"
    assert not any(
        k.startswith("shap_") for k in metrics
    ), "Métriques SHAP présentes malgré erreur"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "Nombre de features incorrect" in str(e) for e in df["error"].dropna()
    ), "Erreur features non loguée"


def test_shap_failure(
    tmp_dirs, mock_data, mock_positions, mock_fetch_news, mock_options_risk
):
    """Teste le calcul lorsque calculate_shap échoue."""
    with patch("src.risk.risk_controller.calculate_shap", return_value=None), patch(
        "src.risk.risk_controller.fetch_news", return_value=mock_fetch_news
    ), patch(
        "src.risk.risk_controller.OptionsRiskManager.calculate_options_risk",
        return_value=mock_options_risk,
    ):
        risk_controller = RiskController(config_path=tmp_dirs["config_path"])
        metrics = risk_controller.calculate_risk_metrics(mock_data, mock_positions)
        assert "confidence_drop_rate" in metrics, "confidence_drop_rate manquant"
        assert not any(
            k.startswith("shap_") for k in metrics
        ), "Métriques SHAP présentes malgré échec"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        snapshot_files = os.listdir(tmp_dirs["snapshot_dir"])
        assert any(
            "snapshot_calculate_risk_metrics" in f and f.endswith(".json")
            for f in snapshot_files
        ), "Snapshot non créé"
        with open(os.path.join(tmp_dirs["snapshot_dir"], snapshot_files[-1]), "r") as f:
            snapshot = json.load(f)
        assert (
            "confidence_drop_rate" in snapshot["data"]
        ), "confidence_drop_rate absent du snapshot"
        assert not any(
            k.startswith("shap_") for k in snapshot["data"]
        ), "Métriques SHAP présentes dans snapshot malgré échec"


def test_stop_trading(
    tmp_dirs, mock_data, mock_positions, mock_shap, mock_fetch_news, mock_options_risk
):
    """Teste les conditions d’arrêt du trading."""
    with patch("src.risk.risk_controller.calculate_shap", mock_shap), patch(
        "src.risk.risk_controller.fetch_news", return_value=mock_fetch_news
    ), patch(
        "src.risk.risk_controller.OptionsRiskManager.calculate_options_risk",
        return_value=mock_options_risk,
    ):
        risk_controller = RiskController(config_path=tmp_dirs["config_path"])
        drawdown = -0.2  # Inférieur à max_drawdown
        should_stop = risk_controller.stop_trading(drawdown, mock_data, mock_positions)
        assert should_stop, "Trading n’a pas été arrêté malgré drawdown excessif"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        snapshot_files = os.listdir(tmp_dirs["snapshot_dir"])
        assert any(
            "snapshot_risk_step" in f and f.endswith(".json") for f in snapshot_files
        ), "Snapshot non créé"
        with open(os.path.join(tmp_dirs["snapshot_dir"], snapshot_files[-1]), "r") as f:
            snapshot = json.load(f)
        assert (
            "confidence_drop_rate" in snapshot["metrics"]
        ), "confidence_drop_rate absent du snapshot"
        assert any(
            k.startswith("shap_") for k in snapshot["metrics"]
        ), "Métriques SHAP absentes du snapshot"


def test_adjust_position_size(
    tmp_dirs, mock_data, mock_positions, mock_shap, mock_fetch_news, mock_options_risk
):
    """Teste l’ajustement de la taille de position."""
    with patch("src.risk.risk_controller.calculate_shap", mock_shap), patch(
        "src.risk.risk_controller.fetch_news", return_value=mock_fetch_news
    ), patch(
        "src.risk.risk_controller.OptionsRiskManager.calculate_options_risk",
        return_value=mock_options_risk,
    ):
        risk_controller = RiskController(config_path=tmp_dirs["config_path"])
        signal_score = 0.8
        size = risk_controller.adjust_position_size(
            signal_score, mock_data, mock_positions
        )
        assert 0 <= size <= 5, "Taille de position hors limites"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        snapshot_files = os.listdir(tmp_dirs["snapshot_dir"])
        assert any(
            "snapshot_adjust_position_size" in f and f.endswith(".json")
            for f in snapshot_files
        ), "Snapshot non créé"
        with open(os.path.join(tmp_dirs["snapshot_dir"], snapshot_files[-1]), "r") as f:
            snapshot = json.load(f)
        assert (
            "confidence_drop_rate" in snapshot["data"]
        ), "confidence_drop_rate absent du snapshot"


def test_penalize_ia(
    tmp_dirs, mock_data, mock_shap, mock_fetch_news, mock_options_risk
):
    """Teste l’application des pénalités IA."""
    with patch("src.risk.risk_controller.calculate_shap", mock_shap), patch(
        "src.risk.risk_controller.fetch_news", return_value=mock_fetch_news
    ), patch(
        "src.risk.risk_controller.OptionsRiskManager.calculate_options_risk",
        return_value=mock_options_risk,
    ):
        risk_controller = RiskController(config_path=tmp_dirs["config_path"])
        loss = -0.15  # Inférieur à penalty_threshold
        penalty = risk_controller.penalize_ia(loss, mock_data)
        assert penalty > 0, "Pénalité non appliquée malgré perte"
        assert os.path.exists(
            tmp_dirs["penalty_log_path"]
        ), "Fichier penalty_log.csv non créé"
        df = pd.read_csv(tmp_dirs["penalty_log_path"])
        assert df.iloc[0]["value"] == penalty, "Pénalité incorrecte dans le log"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        snapshot_files = os.listdir(tmp_dirs["snapshot_dir"])
        assert any(
            "snapshot_risk_step" in f and f.endswith(".json") for f in snapshot_files
        ), "Snapshot non créé"
        with open(os.path.join(tmp_dirs["snapshot_dir"], snapshot_files[-1]), "r") as f:
            snapshot = json.load(f)
        assert (
            "confidence_drop_rate" in snapshot["metrics"]
        ), "confidence_drop_rate absent du snapshot"


def test_save_snapshot_compressed(tmp_dirs, mock_data, mock_positions):
    """Teste la sauvegarde d’un snapshot compressé avec gzip."""
    risk_controller = RiskController(config_path=tmp_dirs["config_path"])
    snapshot_data = {"test": "compressed_snapshot"}
    risk_controller.save_snapshot("test_compressed", snapshot_data, compress=True)
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


def test_save_dashboard_status(
    tmp_dirs, mock_data, mock_positions, mock_shap, mock_fetch_news, mock_options_risk
):
    """Teste la sauvegarde du dashboard avec confidence_drop_rate."""
    with patch("src.risk.risk_controller.calculate_shap", mock_shap), patch(
        "src.risk.risk_controller.fetch_news", return_value=mock_fetch_news
    ), patch(
        "src.risk.risk_controller.OptionsRiskManager.calculate_options_risk",
        return_value=mock_options_risk,
    ):
        risk_controller = RiskController(config_path=tmp_dirs["config_path"])
        risk_controller.calculate_risk_metrics(mock_data, mock_positions)
        risk_controller.save_dashboard_status()
        assert os.path.exists(tmp_dirs["dashboard_path"]), "Fichier dashboard non créé"
        with open(tmp_dirs["dashboard_path"], "r") as f:
            status = json.load(f)
        assert (
            "confidence_drop_rate" in status["risk_metrics"]
        ), "confidence_drop_rate absent du dashboard"
        assert any(
            k.startswith("shap_") for k in status["risk_metrics"]
        ), "Métriques SHAP absentes du dashboard"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"


def test_critical_alerts(
    tmp_dirs, mock_data, mock_positions, mock_shap, mock_fetch_news, mock_options_risk
):
    """Teste les alertes Telegram pour les risques critiques."""
    with patch("src.risk.risk_controller.calculate_shap", mock_shap), patch(
        "src.risk.risk_controller.fetch_news", return_value=mock_fetch_news
    ), patch(
        "src.risk.risk_controller.OptionsRiskManager.calculate_options_risk",
        return_value=mock_options_risk,
    ), patch(
        "src.risk.risk_controller.send_telegram_alert"
    ) as mock_telegram:
        risk_controller = RiskController(config_path=tmp_dirs["config_path"])
        # Forcer un risque élevé en modifiant les données
        mock_data["vix"] = [40.0] * 5  # Supérieur à vix_threshold
        metrics = risk_controller.calculate_risk_metrics(mock_data, mock_positions)
        assert (
            metrics["confidence_drop_rate"] > 0
        ), "confidence_drop_rate devrait être non nul"
        mock_telegram.assert_called_with(pytest.any(str))
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        snapshot_files = os.listdir(tmp_dirs["snapshot_dir"])
        assert any(
            "snapshot_calculate_risk_metrics" in f and f.endswith(".json")
            for f in snapshot_files
        ), "Snapshot non créé"
        with open(os.path.join(tmp_dirs["snapshot_dir"], snapshot_files[-1]), "r") as f:
            snapshot = json.load(f)
        assert (
            "confidence_drop_rate" in snapshot["data"]
        ), "confidence_drop_rate absent du snapshot"
