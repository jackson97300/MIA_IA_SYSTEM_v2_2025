# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_trade_window_filter.py
# Tests unitaires pour src/risk/trade_window_filter.py
# Version : 2.1.3
# Date : 2025-05-13
# Rôle : Valide le filtrage des fenêtres de trading basé sur les conditions macro et de volatilité,
#        incluant confidence_drop_rate (Phase 8) et analyse SHAP (Phase 17).

import gzip
import json
import os
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.risk.trade_window_filter import TradeWindowFilter


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, snapshots, dashboard, et configuration."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs" / "risk"
    logs_dir.mkdir(parents=True)
    snapshot_dir = data_dir / "risk" / "trade_window_snapshots"
    snapshot_dir.mkdir()
    news_dir = data_dir / "news"
    news_dir.mkdir()
    risk_dir = data_dir / "risk"
    risk_dir.mkdir()

    # Créer market_config.yaml
    config_path = config_dir / "market_config.yaml"
    config_content = {
        "trade_window_filter": {
            "event_timing_threshold": 3600,
            "event_frequency_threshold": 3,
            "vix_threshold": 30,
            "vix_es_correlation_threshold": 25.0,
            "event_impact_threshold": 0.5,
            "max_trades_per_hour": 10,
            "window_score_threshold": 0.7,
            "macro_score_threshold": 0.8,
            "min_confidence": 0.7,
            "vix_weight": 0.3,
            "event_impact_weight": 0.3,
            "event_proximity_weight": 0.2,
            "event_frequency_weight": 0.1,
            "trade_frequency_weight": 0.1,
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
        "news_dir": str(news_dir),
        "risk_dir": str(risk_dir),
        "perf_log_path": str(logs_dir / "trade_window_performance.csv"),
        "dashboard_path": str(risk_dir / "trade_window_dashboard.json"),
        "macro_events_path": str(news_dir / "macro_events.csv"),
        "trade_window_log_path": str(data_dir / "risk" / "trade_window_log.csv"),
    }


@pytest.fixture
def mock_data():
    """Crée des données factices pour tester."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start="2025-05-13 10:00:00", periods=5, freq="T"
            ),
            "vix": [20.0] * 5,
            "vix_es_correlation": [22.0] * 5,
            "event_volatility_impact": [0.3] * 5,
            "event_timing_proximity": [1800.0] * 5,
            "event_frequency_24h": [2.0] * 5,
            "trade_frequency_1s": [8.0] * 5,
            **{
                f"feat_{i}": [np.random.uniform(0, 1)] for i in range(344)
            },  # 350 features
        }
    )


@pytest.fixture
def mock_macro_events(tmp_dirs):
    """Crée des événements macroéconomiques factices."""
    macro_events = pd.DataFrame(
        {
            "timestamp": [datetime.now() + timedelta(minutes=30)],
            "event_type": ["FOMC"],
            "impact_score": [0.8],
        }
    )
    macro_events.to_csv(tmp_dirs["macro_events_path"], index=False, encoding="utf-8")
    return macro_events


@pytest.fixture
def mock_shap():
    """Mock pour calculate_shap."""

    def mock_calculate_shap(data, target, max_features):
        columns = data.columns[:max_features]
        return pd.DataFrame(
            {col: [0.1] * len(data) for col in columns}, index=data.index
        )

    return mock_calculate_shap


def test_trade_window_filter_init(tmp_dirs, mock_macro_events):
    """Teste l’initialisation de TradeWindowFilter."""
    filter = TradeWindowFilter(
        config_path=tmp_dirs["config_path"],
        macro_events_path=tmp_dirs["macro_events_path"],
    )
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    assert os.path.exists(tmp_dirs["snapshot_dir"]), "Dossier de snapshots non créé"
    assert os.path.exists(
        tmp_dirs["macro_events_path"]
    ), "Fichier macro_events.csv non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert all(
        col in df.columns
        for col in ["timestamp", "operation", "latency", "cpu_percent"]
    ), "Colonnes de performance manquantes"


def test_calculate_window_score_valid(
    tmp_dirs, mock_data, mock_macro_events, mock_shap
):
    """Teste le calcul du window_score avec des données valides."""
    with patch("src.risk.trade_window_filter.calculate_shap", mock_shap):
        filter = TradeWindowFilter(
            config_path=tmp_dirs["config_path"],
            macro_events_path=tmp_dirs["macro_events_path"],
        )
        score = filter.calculate_window_score(mock_data)
        assert 0 <= score <= 1, "window_score hors limites"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        snapshot_files = os.listdir(tmp_dirs["snapshot_dir"])
        assert any(
            "snapshot_calculate_window_score" in f and f.endswith(".json")
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
        assert (
            len([k for k in snapshot["data"] if k.startswith("shap_")]) <= 50
        ), "Trop de features SHAP"


def test_calculate_window_score_invalid_data(tmp_dirs, mock_macro_events):
    """Teste le calcul avec des données invalides."""
    filter = TradeWindowFilter(
        config_path=tmp_dirs["config_path"],
        macro_events_path=tmp_dirs["macro_events_path"],
    )
    data = pd.DataFrame(
        {"timestamp": [datetime.now()], "vix": [20.0]}
    )  # Features insuffisants
    score = filter.calculate_window_score(data)
    assert score == 0.0, "window_score non nul malgré erreur"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "Nombre de features incorrect" in str(e) for e in df["error"].dropna()
    ), "Erreur features non loguée"


def test_shap_failure(tmp_dirs, mock_data, mock_macro_events):
    """Teste le calcul lorsque calculate_shap échoue."""
    with patch("src.risk.trade_window_filter.calculate_shap", return_value=None):
        filter = TradeWindowFilter(
            config_path=tmp_dirs["config_path"],
            macro_events_path=tmp_dirs["macro_events_path"],
        )
        score = filter.calculate_window_score(mock_data)
        assert 0 <= score <= 1, "window_score hors limites"
        snapshot_files = os.listdir(tmp_dirs["snapshot_dir"])
        assert any(
            "snapshot_calculate_window_score" in f and f.endswith(".json")
            for f in snapshot_files
        ), "Snapshot non créé"
        with open(os.path.join(tmp_dirs["snapshot_dir"], snapshot_files[-1]), "r") as f:
            snapshot = json.load(f)
        assert (
            "confidence_drop_rate" in snapshot["data"]
        ), "confidence_drop_rate absent du snapshot"
        assert not any(
            k.startswith("shap_") for k in snapshot["data"]
        ), "Métriques SHAP présentes malgré échec"


def test_block_trade(tmp_dirs, mock_data, mock_macro_events, mock_shap):
    """Teste les conditions de blocage des trades."""
    with patch("src.risk.trade_window_filter.calculate_shap", mock_shap):
        filter = TradeWindowFilter(
            config_path=tmp_dirs["config_path"],
            macro_events_path=tmp_dirs["macro_events_path"],
        )
        macro_score = 0.9  # Supérieur à macro_score_threshold
        step = 1
        is_blocked = filter.block_trade(macro_score, mock_data, step)
        assert is_blocked, "Trade n’a pas été bloqué malgré macro_score élevé"
        assert os.path.exists(
            tmp_dirs["trade_window_log_path"]
        ), "Fichier trade_window_log.csv non créé"
        df = pd.read_csv(tmp_dirs["trade_window_log_path"])
        assert df.iloc[0]["decision"] == "block", "Décision incorrecte dans le log"
        assert (
            df.iloc[0]["confidence_drop_rate"] >= 0
        ), "confidence_drop_rate absent ou négatif"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        snapshot_files = os.listdir(tmp_dirs["snapshot_dir"])
        assert any(
            "snapshot_window_step" in f and f.endswith(".json") for f in snapshot_files
        ), "Snapshot non créé"


def test_prevent_overtrading(tmp_dirs, mock_macro_events):
    """Teste la détection du surtrading."""
    filter = TradeWindowFilter(
        config_path=tmp_dirs["config_path"],
        macro_events_path=tmp_dirs["macro_events_path"],
    )
    filter.last_vix = 20.0  # VIX normal
    for _ in range(10):  # Simuler 10 trades dans l’heure
        filter.prevent_overtrading(datetime.now())
    is_blocked = filter.prevent_overtrading(datetime.now())
    assert is_blocked, "Surtrading n’a pas été détecté"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    snapshot_files = os.listdir(tmp_dirs["snapshot_dir"])
    assert any(
        "snapshot_window_step" in f and f.endswith(".json") for f in snapshot_files
    ), "Snapshot non créé"
    with open(os.path.join(tmp_dirs["snapshot_dir"], snapshot_files[-1]), "r") as f:
        snapshot = json.load(f)
    assert (
        "confidence_drop_rate" in snapshot["block_events"][-1]
    ), "confidence_drop_rate absent du snapshot"


def test_check_trade_window(tmp_dirs, mock_data, mock_macro_events, mock_shap):
    """Teste la combinaison de block_trade et prevent_overtrading."""
    with patch("src.risk.trade_window_filter.calculate_shap", mock_shap):
        filter = TradeWindowFilter(
            config_path=tmp_dirs["config_path"],
            macro_events_path=tmp_dirs["macro_events_path"],
        )
        macro_score = 0.9  # Supérieur à macro_score_threshold
        step = 1
        last_trade_time = datetime.now()
        is_blocked = filter.check_trade_window(
            macro_score, mock_data, step, last_trade_time
        )
        assert is_blocked, "Trade n’a pas été bloqué malgré macro_score élevé"
        assert os.path.exists(
            tmp_dirs["trade_window_log_path"]
        ), "Fichier trade_window_log.csv non créé"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"


def test_save_snapshot_compressed(tmp_dirs, mock_macro_events):
    """Teste la sauvegarde d’un snapshot compressé avec gzip."""
    filter = TradeWindowFilter(
        config_path=tmp_dirs["config_path"],
        macro_events_path=tmp_dirs["macro_events_path"],
    )
    snapshot_data = {"test": "compressed_snapshot"}
    filter.save_snapshot("test_compressed", snapshot_data, compress=True)
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


def test_save_dashboard_status(tmp_dirs, mock_data, mock_macro_events, mock_shap):
    """Teste la sauvegarde du dashboard avec confidence_drop_rate."""
    with patch("src.risk.trade_window_filter.calculate_shap", mock_shap):
        filter = TradeWindowFilter(
            config_path=tmp_dirs["config_path"],
            macro_events_path=tmp_dirs["macro_events_path"],
        )
        filter.calculate_window_score(mock_data)
        filter.save_dashboard_status()
        assert os.path.exists(tmp_dirs["dashboard_path"]), "Fichier dashboard non créé"
        with open(tmp_dirs["dashboard_path"], "r") as f:
            status = json.load(f)
        assert (
            "confidence_drop_rate" in status
        ), "confidence_drop_rate absent du dashboard"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"


def test_critical_alerts(tmp_dirs, mock_data, mock_macro_events, mock_shap):
    """Teste les alertes Telegram pour les blocages critiques."""
    with patch("src.risk.trade_window_filter.calculate_shap", mock_shap), patch(
        "src.risk.trade_window_filter.send_telegram_alert"
    ) as mock_telegram:
        filter = TradeWindowFilter(
            config_path=tmp_dirs["config_path"],
            macro_events_path=tmp_dirs["macro_events_path"],
        )
        mock_data["vix_es_correlation"] = [
            30.0
        ] * 5  # Supérieur à vix_es_correlation_threshold
        macro_score = 0.5
        step = 1
        is_blocked = filter.block_trade(macro_score, mock_data, step)
        assert is_blocked, "Trade n’a pas été bloqué malgré VIX élevé"
        mock_telegram.assert_called_with(pytest.any(str))
        assert os.path.exists(
            tmp_dirs["trade_window_log_path"]
        ), "Fichier trade_window_log.csv non créé"
        df = pd.read_csv(tmp_dirs["trade_window_log_path"])
        assert (
            df.iloc[0]["confidence_drop_rate"] > 0
        ), "confidence_drop_rate non positif malgré blocage"
