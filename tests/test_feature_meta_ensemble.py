# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_feature_meta_ensemble.py
# Tests unitaires pour src/features/feature_meta_ensemble.py
# Version : 2.1.3
# Date : 2025-05-13
# Rôle : Valide la réduction dynamique des features avec validation SHAP (Phase 17),
#        confidence_drop_rate (Phase 8), et snapshots avec alertes Telegram.

import gzip
import json
import os
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.features.feature_meta_ensemble import FeatureMetaEnsemble


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, snapshots, figures, cache, et données."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    snapshots_dir = data_dir / "ensemble_snapshots"
    snapshots_dir.mkdir()
    figures_dir = data_dir / "figures" / "ensemble"
    figures_dir.mkdir(parents=True)
    features_dir = data_dir / "features"
    features_dir.mkdir()
    cache_dir = features_dir / "cache" / "ensemble"
    cache_dir.mkdir(parents=True)

    # Créer es_config.yaml
    config_path = config_dir / "es_config.yaml"
    config_content = {
        "feature_meta_ensemble": {
            "target_dims": 81,
            "window_size": 20,
            "boost_factors": {"Trend": 1.5, "Range": 1.5, "Defensive": 1.5},
            "momentum_features": [
                "spy_lead_return",
                "sector_leader_momentum",
                "spy_momentum_diff",
                "order_flow_acceleration",
            ],
            "volatility_features": [
                "atr_14",
                "volatility_trend",
                "microstructure_volatility",
                "vix_es_correlation",
                "iv_atm",
                "option_skew",
            ],
            "risk_features": [
                "bond_equity_risk_spread",
                "vanna_cliff_slope",
                "vanna_exposure",
                "delta_exposure",
                "gex_slope",
            ],
            "buffer_size": 100,
            "max_cache_size": 1000,
            "buffer_maxlen": 1000,
            "cache_hours": 24,
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
        "cache_dir": str(cache_dir),
        "perf_log_path": str(logs_dir / "feature_meta_ensemble_performance.csv"),
        "feature_importance_path": str(features_dir / "feature_importance.csv"),
    }


@pytest.fixture
def mock_data():
    """Crée des données factices pour tester."""
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=100, freq="1min"),
            "close": np.random.normal(5100, 10, 100),
            "neural_regime": ["Trend"] * 20 + ["Range"] * 40 + ["Defensive"] * 40,
            "atr_14": np.random.uniform(0.5, 2.0, 100),
            "volatility_trend": np.random.uniform(-0.1, 0.1, 100),
            "spy_lead_return": np.random.uniform(-0.02, 0.02, 100),
            "sector_leader_momentum": np.random.uniform(-1, 1, 100),
            "order_flow_acceleration": np.random.uniform(-0.5, 0.5, 100),
            "microstructure_volatility": np.random.uniform(0, 0.1, 100),
            "vix_es_correlation": np.random.uniform(-1, 1, 100),
            "bond_equity_risk_spread": np.random.uniform(-1, 1, 100),
            "vanna_cliff_slope": np.random.uniform(-0.1, 0.1, 100),
            "vanna_exposure": np.random.uniform(-1000, 1000, 100),
            "delta_exposure": np.random.uniform(-1000, 1000, 100),
            "iv_atm": np.random.uniform(0.1, 0.3, 100),
            "option_skew": np.random.uniform(-0.5, 0.5, 100),
            "gex_slope": np.random.uniform(-0.1, 0.1, 100),
            "trade_success_prob": np.random.uniform(0, 1, 100),
        }
    )
    for i in range(350 - len(data.columns) + 1):
        data[f"feature_{i}"] = np.random.normal(0, 1, 100)
    return data


@pytest.fixture
def mock_feature_importance(tmp_dirs):
    """Crée un fichier feature_importance.csv factice."""
    features = [
        "atr_14",
        "volatility_trend",
        "spy_lead_return",
        "sector_leader_momentum",
        "order_flow_acceleration",
        "microstructure_volatility",
        "vix_es_correlation",
        "bond_equity_risk_spread",
        "vanna_cliff_slope",
        "vanna_exposure",
        "delta_exposure",
        "iv_atm",
        "option_skew",
        "gex_slope",
        "trade_success_prob",
    ] + [f"feature_{i}" for i in range(335)]
    shap_data = pd.DataFrame(
        {
            "feature": features[:150],
            "importance": [0.1] * 150,
            "regime": ["range"] * 150,
        }
    )
    shap_data.to_csv(tmp_dirs["feature_importance_path"], index=False, encoding="utf-8")
    return shap_data


def test_feature_meta_ensemble_init(tmp_dirs, mock_feature_importance):
    """Teste l’initialisation de FeatureMetaEnsemble."""
    ensemble = FeatureMetaEnsemble(config_path=tmp_dirs["config_path"])
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


def test_optimize_features_valid(tmp_dirs, mock_data, mock_feature_importance):
    """Teste la réduction des features avec des données valides."""
    ensemble = FeatureMetaEnsemble(config_path=tmp_dirs["config_path"])
    reduced_data = ensemble.optimize_features(mock_data)
    assert (
        len(reduced_data.columns) <= 82
    ), "Nombre de dimensions après réduction incorrect"  # 81 features + timestamp
    assert "timestamp" in reduced_data.columns, "Colonne timestamp manquante"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_optimize_features" in f and f.endswith(".json")
        for f in snapshot_files
    ), "Snapshot non créé"
    with open(os.path.join(tmp_dirs["snapshots_dir"], snapshot_files[-1]), "r") as f:
        snapshot = json.load(f)
    assert (
        "confidence_drop_rate" in snapshot["data"]
    ), "confidence_drop_rate absent du snapshot"


def test_optimize_features_invalid_data(tmp_dirs, mock_feature_importance):
    """Teste la réduction des features avec des données invalides."""
    ensemble = FeatureMetaEnsemble(config_path=tmp_dirs["config_path"])
    invalid_data = pd.DataFrame({"timestamp": [datetime.now()]})  # Données vides
    reduced_data = ensemble.optimize_features(invalid_data)
    assert (
        len(reduced_data.columns) <= 82
    ), "Nombre de dimensions après réduction incorrect"
    assert "timestamp" in reduced_data.columns, "Colonne timestamp manquante"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "DataFrame vide" in str(e) for e in df["error"].dropna()
    ), "Erreur DataFrame vide non loguée"


def test_reduce_incremental_features(tmp_dirs, mock_data, mock_feature_importance):
    """Teste la réduction incrémentale des features."""
    ensemble = FeatureMetaEnsemble(config_path=tmp_dirs["config_path"])
    for i in range(20):  # Remplir le buffer
        ensemble.buffer.append(mock_data.iloc[i].to_frame().T)
    row = mock_data.iloc[-1]
    result = ensemble.reduce_incremental_features(row, "Trend")
    assert (
        len(result) <= 82
    ), "Nombre de dimensions après réduction incrémentale incorrect"
    assert "timestamp" in result.index, "Colonne timestamp manquante"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_reduce_incremental_features" in f and f.endswith(".json")
        for f in snapshot_files
    ), "Snapshot non créé"


def test_validate_shap_features(tmp_dirs, mock_feature_importance):
    """Teste la validation des top 150 SHAP features."""
    ensemble = FeatureMetaEnsemble(config_path=tmp_dirs["config_path"])
    features = ["spy_lead_return", "atr_14", "bond_equity_risk_spread"]
    result = ensemble.validate_shap_features(features)
    assert result, "Validation SHAP échouée pour features valides"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_validate_shap_features" in f for f in snapshot_files
    ), "Snapshot non créé"


def test_save_snapshot_compressed(tmp_dirs):
    """Teste la sauvegarde d’un snapshot compressé avec gzip."""
    ensemble = FeatureMetaEnsemble(config_path=tmp_dirs["config_path"])
    snapshot_data = {"test": "compressed_snapshot"}
    ensemble.save_snapshot("test_compressed", snapshot_data, compress=True)
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


def test_plot_shap_importance(tmp_dirs, mock_data, mock_feature_importance):
    """Teste la génération des visualisations des importances SHAP."""
    ensemble = FeatureMetaEnsemble(config_path=tmp_dirs["config_path"])
    shap_importance = pd.Series(
        np.random.uniform(0, 1, 15),
        index=[
            "atr_14",
            "volatility_trend",
            "spy_lead_return",
            "sector_leader_momentum",
            "order_flow_acceleration",
            "microstructure_volatility",
            "vix_es_correlation",
            "bond_equity_risk_spread",
            "vanna_cliff_slope",
            "vanna_exposure",
            "delta_exposure",
            "iv_atm",
            "option_skew",
            "gex_slope",
            "trade_success_prob",
        ],
    )
    timestamp = mock_data["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")
    ensemble.plot_shap_importance(shap_importance, "Trend", timestamp)
    timestamp_safe = timestamp.replace(":", "-")
    assert os.path.exists(
        os.path.join(tmp_dirs["figures_dir"], f"shap_bar_Trend_{timestamp_safe}.png")
    ), "Visualisation SHAP non générée"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"


def test_critical_alerts(tmp_dirs, mock_feature_importance):
    """Teste les alertes Telegram pour les erreurs critiques."""
    with patch(
        "src.features.feature_meta_ensemble.send_telegram_alert"
    ) as mock_telegram:
        ensemble = FeatureMetaEnsemble(config_path=tmp_dirs["config_path"])
        invalid_data = pd.DataFrame(
            {"timestamp": [datetime.now()]}
        )  # Données invalides
        ensemble.optimize_features(invalid_data)
        mock_telegram.assert_called_with(pytest.any(str))
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "DataFrame vide" in str(e) for e in df["error"].dropna()
        ), "Erreur critique non loguée"
