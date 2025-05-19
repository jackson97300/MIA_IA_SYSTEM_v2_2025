# Teste signal_selector.py.
# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_signal_selector.py
# Tests unitaires pour src/trading/signal_selector.py
# Version : 2.1.3
# Date : 2025-05-13
# Rôle : Valide le calcul du Score Global de Confiance (SGC) avec régimes hybrides (méthode 11),
#        validation SHAP (Phase 17), confidence_drop_rate (Phase 8), et données IQFeed.

import gzip
import json
import os
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.trading.signal_selector import SignalSelector


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, snapshots, figures, et données."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    snapshots_dir = data_dir / "signal_snapshots"
    snapshots_dir.mkdir()
    figures_dir = data_dir / "figures" / "signals"
    figures_dir.mkdir(parents=True)
    features_dir = data_dir / "features"
    features_dir.mkdir()
    iqfeed_dir = data_dir / "iqfeed"
    iqfeed_dir.mkdir()

    # Créer es_config.yaml
    config_path = config_dir / "es_config.yaml"
    config_content = {
        "signal_selector": {
            "feature_weights": {
                "rsi_14": 0.15,
                "atr_14": 0.10,
                "trend_strength": 0.10,
                "gex": 0.15,
                "iv_atm": 0.10,
                "gex_slope": 0.10,
                "pca_orderflow_1": 0.10,
                "pca_orderflow_2": 0.10,
                "confidence_drop_rate": 0.05,
                "spy_lead_return": 0.05,
                "sector_leader_correlation": 0.05,
            },
            "sgc_threshold": 0.7,
            "window_size": 20,
            "buffer_size": 100,
            "max_cache_size": 1000,
            "buffer_maxlen": 1000,
            "cache_hours": 24,
            "regime_weights": {
                "trend": {
                    "rsi_14": 1.5,
                    "trend_strength": 1.5,
                    "gex": 1.2,
                    "pca_orderflow_1": 1.2,
                },
                "range": {
                    "atr_14": 1.5,
                    "iv_atm": 1.5,
                    "gex_slope": 1.2,
                    "pca_orderflow_2": 1.2,
                },
                "defensive": {
                    "confidence_drop_rate": 1.5,
                    "spy_lead_return": 1.2,
                    "sector_leader_correlation": 1.2,
                },
            },
        }
    }
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)

    return {
        "base_dir": str(base_dir),
        "config_path": str(config_path),
        "logs_dir": str(logs_dir),
        "snapshots_dir": str(snapshots_dir),
        "figures_dir": str(figures_dir),
        "features_dir": str(features_dir),
        "iqfeed_dir": str(iqfeed_dir),
        "perf_log_path": str(logs_dir / "signal_selector_performance.csv"),
        "feature_importance_path": str(features_dir / "feature_importance.csv"),
    }


@pytest.fixture
def mock_data():
    """Crée des données factices pour tester."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=100, freq="1min"),
            "rsi_14": np.random.uniform(30, 70, 100),
            "atr_14": np.random.uniform(0.5, 2.0, 100),
            "trend_strength": np.random.uniform(10, 50, 100),
            "gex": np.random.uniform(-1000, 1000, 100),
            "iv_atm": np.random.uniform(0.1, 0.3, 100),
            "gex_slope": np.random.uniform(-0.05, 0.05, 100),
            "pca_orderflow_1": np.random.uniform(-1, 1, 100),
            "pca_orderflow_2": np.random.uniform(-1, 1, 100),
            "confidence_drop_rate": np.random.uniform(0, 1, 100),
            "spy_lead_return": np.random.uniform(-0.02, 0.02, 100),
            "sector_leader_correlation": np.random.uniform(-1, 1, 100),
        }
    )


@pytest.fixture
def mock_feature_importance(tmp_dirs):
    """Crée un fichier feature_importance.csv factice."""
    shap_data = pd.DataFrame(
        {
            "feature": [
                "rsi_14",
                "atr_14",
                "trend_strength",
                "gex",
                "iv_atm",
                "gex_slope",
                "pca_orderflow_1",
                "pca_orderflow_2",
                "confidence_drop_rate",
                "spy_lead_return",
                "sector_leader_correlation",
            ]
            * 15,
            "importance": [0.1] * 165,
            "regime": ["range"] * 165,
        }
    )
    shap_data.to_csv(tmp_dirs["feature_importance_path"], index=False, encoding="utf-8")
    return shap_data


def test_signal_selector_init(tmp_dirs, mock_feature_importance):
    """Teste l’initialisation de SignalSelector."""
    selector = SignalSelector(config_path=tmp_dirs["config_path"])
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    assert os.path.exists(tmp_dirs["snapshots_dir"]), "Dossier de snapshots non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert all(
        col in df.columns
        for col in ["timestamp", "operation", "latency", "cpu_usage_percent"]
    ), "Colonnes de performance manquantes"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_init" in f and f.endswith(".json") for f in snapshot_files
    ), "Snapshot non créé"


def test_calculate_sgc_valid(tmp_dirs, mock_data, mock_feature_importance):
    """Teste le calcul du SGC avec des données valides."""
    selector = SignalSelector(config_path=tmp_dirs["config_path"])
    regime_probs = {"trend": 0.4, "range": 0.5, "defensive": 0.1}
    sgc, secondary_metrics = selector.calculate_sgc(mock_data, regime_probs)
    assert len(sgc) == len(mock_data), "Longueur SGC incorrecte"
    assert all(
        col in secondary_metrics.columns
        for col in ["volatility_adjustment", "confidence_adjustment", "raw_sgc"]
    ), "Colonnes secondaires manquantes"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_calculate_sgc" in f and f.endswith(".json") for f in snapshot_files
    ), "Snapshot non créé"
    with open(os.path.join(tmp_dirs["snapshots_dir"], snapshot_files[-1]), "r") as f:
        snapshot = json.load(f)
    assert (
        "confidence_drop_rate" in snapshot["data"]
    ), "confidence_drop_rate absent du snapshot"
    assert os.path.exists(
        os.path.join(
            tmp_dirs["figures_dir"],
            f"sgc_temporal_{mock_data['timestamp'].iloc[-1].strftime('%Y-%m-%d %H-%M-%S')}.png",
        )
    ), "Visualisation temporelle non générée"


def test_calculate_sgc_invalid_data(tmp_dirs, mock_feature_importance):
    """Teste le calcul du SGC avec des données invalides."""
    selector = SignalSelector(config_path=tmp_dirs["config_path"])
    invalid_data = pd.DataFrame({"timestamp": [datetime.now()]})  # Données vides
    regime_probs = {"trend": 0.4, "range": 0.5, "defensive": 0.1}
    sgc, secondary_metrics = selector.calculate_sgc(invalid_data, regime_probs)
    assert sgc.eq(0.0).all(), "SGC non nul pour données invalides"
    assert (
        secondary_metrics.eq(0.0).all().all()
    ), "Métriques secondaires non nulles pour données invalides"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "DataFrame vide" in str(e) for e in df["error"].dropna()
    ), "Erreur DataFrame vide non loguée"


def test_calculate_incremental_sgc(tmp_dirs, mock_data, mock_feature_importance):
    """Teste le calcul incrémental du SGC."""
    selector = SignalSelector(config_path=tmp_dirs["config_path"])
    regime_probs = {"trend": 0.4, "range": 0.5, "defensive": 0.1}
    row = mock_data.iloc[-1]
    for i in range(20):  # Remplir le buffer
        selector.buffer.append(mock_data.iloc[i].to_frame().T)
    sgc, metrics = selector.calculate_incremental_sgc(row, regime_probs)
    assert isinstance(sgc, float), "SGC n’est pas un float"
    assert all(
        key in metrics
        for key in ["volatility_adjustment", "confidence_adjustment", "raw_sgc"]
    ), "Métriques secondaires manquantes"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_calculate_incremental_sgc" in f and f.endswith(".json")
        for f in snapshot_files
    ), "Snapshot non créé"


def test_validate_iqfeed_data(tmp_dirs, mock_data, mock_feature_importance):
    """Teste la validation des données IQFeed."""
    selector = SignalSelector(config_path=tmp_dirs["config_path"])
    result = selector.validate_iqfeed_data(mock_data)
    assert result, "Validation IQFeed échouée pour données valides"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"


def test_validate_shap_features(tmp_dirs, mock_feature_importance):
    """Teste la validation des top 150 SHAP features."""
    selector = SignalSelector(config_path=tmp_dirs["config_path"])
    features = ["rsi_14", "atr_14", "gex"]
    result = selector.validate_shap_features(features)
    assert result, "Validation SHAP échouée pour features valides"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_validate_shap_features" in f for f in snapshot_files
    ), "Snapshot non créé"


def test_save_snapshot_compressed(tmp_dirs):
    """Teste la sauvegarde d’un snapshot compressé avec gzip."""
    selector = SignalSelector(config_path=tmp_dirs["config_path"])
    snapshot_data = {"test": "compressed_snapshot"}
    selector.save_snapshot("test_compressed", snapshot_data, compress=True)
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_test_compressed" in f and f.endswith(".json.gz")
        for f in snapshot_files
    ), "Snapshot compressé non créé"
    with gzip.open(
        os.path.join(tmp_dirs["snapshots_dir"], snapshot_files[-1]),
        "rt",
        encoding="utf-8",
    ) as f:
        snapshot = json.load(f)
    assert snapshot["data"] == snapshot_data, "Contenu du snapshot compressé incorrect"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"


def test_plot_sgc(tmp_dirs, mock_data, mock_feature_importance):
    """Teste la génération des visualisations SGC."""
    selector = SignalSelector(config_path=tmp_dirs["config_path"])
    regime_probs = {"trend": 0.4, "range": 0.5, "defensive": 0.1}
    sgc, secondary_metrics = selector.calculate_sgc(mock_data, regime_probs)
    timestamp = mock_data["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")
    selector.plot_sgc(sgc, secondary_metrics, timestamp)
    timestamp_safe = timestamp.replace(":", "-")
    assert os.path.exists(
        os.path.join(tmp_dirs["figures_dir"], f"sgc_temporal_{timestamp_safe}.png")
    ), "Visualisation temporelle non générée"
    assert os.path.exists(
        os.path.join(tmp_dirs["figures_dir"], f"sgc_distribution_{timestamp_safe}.png")
    ), "Distribution non générée"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"


def test_critical_alerts(tmp_dirs, mock_feature_importance):
    """Teste les alertes Telegram pour les erreurs critiques."""
    with patch("src.trading.signal_selector.send_telegram_alert") as mock_telegram:
        selector = SignalSelector(config_path=tmp_dirs["config_path"])
        invalid_data = pd.DataFrame(
            {"timestamp": [datetime.now()]}
        )  # Données invalides
        regime_probs = {"trend": 0.4, "range": 0.5, "defensive": 0.1}
        selector.calculate_sgc(invalid_data, regime_probs)
        mock_telegram.assert_called_with(pytest.any(str))
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "DataFrame vide" in str(e) for e in df["error"].dropna()
        ), "Erreur critique non loguée"
