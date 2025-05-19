# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_features_audit.py
# Tests unitaires pour src/features/features_audit.py
# Version : 2.1.3
# Date : 2025-05-13
# Rôle : Valide l’audit des 350 features, incluant l’analyse SHAP (Phase 17), confidence_drop_rate (Phase 8),
#        la validation des données IQFeed, et les snapshots avec alertes Telegram.

import gzip
import json
import os
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.features.features_audit import FeaturesAudit


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
    snapshots_dir = data_dir / "audit_snapshots"
    snapshots_dir.mkdir()
    figures_dir = data_dir / "figures"
    figures_dir.mkdir()
    features_dir = data_dir / "features"
    features_dir.mkdir()
    iqfeed_dir = data_dir / "iqfeed"
    iqfeed_dir.mkdir()

    # Créer feature_sets.yaml
    config_path = config_dir / "feature_sets.yaml"
    config_content = {
        "feature_sets": {
            "momentum": {
                "features": [
                    {"name": "spy_lead_return", "range": [-1, 1], "source": "IQFeed"}
                ]
            },
            "volatility": {
                "features": [{"name": "atr_14", "range": [0, None], "source": "IQFeed"}]
            },
            "risk": {
                "features": [
                    {"name": "vanna_exposure", "range": [-1, 1], "source": "IQFeed"}
                ]
            },
            "orderbook": {
                "features": [
                    {"name": "level_4_size_bid", "range": [0, None], "source": "IQFeed"}
                ]
            },
            "options": {
                "features": [
                    {
                        "name": "oi_concentration_ratio",
                        "range": [0, 1],
                        "source": "IQFeed",
                    }
                ]
            },
            "neural": {"features": [{"name": "neural_feature_0", "range": None}]},
        },
        "logging": {"buffer_size": 100},
        "cache": {"max_cache_size": 1000},
        "performance_thresholds": {
            "max_missing_features": 5,
            "max_nan_percentage": 0.1,
            "max_outliers_percentage": 0.05,
        },
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
        "perf_log_path": str(logs_dir / "features_audit_performance.csv"),
        "dashboard_path": str(data_dir / "features_audit_dashboard.json"),
        "audit_raw_path": str(logs_dir / "features_audit_raw.csv"),
        "audit_final_path": str(logs_dir / "features_audit_final.csv"),
        "feature_importance_path": str(features_dir / "feature_importance.csv"),
    }


@pytest.fixture
def mock_data():
    """Crée des données factices pour tester."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 10:00:00", periods=100, freq="1min"),
            "spy_lead_return": np.random.uniform(-1, 1, 100),
            "atr_14": np.random.uniform(0, 2, 100),
            "vanna_exposure": np.random.uniform(-1, 1, 100),
            "level_4_size_bid": np.random.randint(100, 1000, 100),
            "oi_concentration_ratio": np.random.uniform(0, 1, 100),
            "neural_feature_0": np.random.randn(100),
            "vix_term_1m": np.random.uniform(15, 25, 100),
            "iv_atm": np.random.uniform(0.1, 0.2, 100),
            "option_skew": np.random.uniform(0, 0.3, 100),
            "gex_slope": np.random.uniform(0, 0.05, 100),
            "trade_success_prob": np.random.uniform(0, 1, 100),
        }
    )


@pytest.fixture
def mock_feature_importance(tmp_dirs):
    """Crée un fichier feature_importance.csv factice."""
    shap_data = pd.DataFrame(
        {
            "feature": [
                "spy_lead_return",
                "atr_14",
                "vanna_exposure",
                "level_4_size_bid",
                "oi_concentration_ratio",
                "neural_feature_0",
                "vix_term_1m",
                "iv_atm",
                "option_skew",
                "gex_slope",
                "trade_success_prob",
            ]
            * 15,
            "importance": [0.1] * 165,
            "regime": ["range"] * 165,
        }
    )
    shap_data.to_csv(tmp_dirs["feature_importance_path"], index=False, encoding="utf-8")
    return shap_data


def test_features_audit_init(tmp_dirs, mock_feature_importance):
    """Teste l’initialisation de FeaturesAudit."""
    FeaturesAudit()
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
        "snapshot_load_config_with_manager_new" in f for f in snapshot_files
    ), "Snapshot de config non créé"


def test_audit_features_valid(tmp_dirs, mock_data, mock_feature_importance):
    """Teste l’audit avec des données valides."""
    audit = FeaturesAudit()
    result = audit.audit_features(mock_data, stage="raw")
    assert result, "Audit échoué pour données valides"
    assert os.path.exists(tmp_dirs["audit_raw_path"]), "Fichier audit_raw.csv non créé"
    assert os.path.exists(tmp_dirs["dashboard_path"]), "Fichier dashboard non créé"
    with open(tmp_dirs["dashboard_path"], "r") as f:
        status = json.load(f)
    assert "confidence_drop_rate" in status, "confidence_drop_rate absent du dashboard"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_audit_features" in f and f.endswith(".json") for f in snapshot_files
    ), "Snapshot non créé"
    assert os.path.exists(
        os.path.join(
            tmp_dirs["figures_dir"],
            f"features_audit_heatmap_raw_{status['timestamp'].replace(':', '-')}.png",
        )
    ), "Heatmap non générée"


def test_audit_features_invalid_data(tmp_dirs, mock_feature_importance):
    """Teste l’audit avec des données invalides."""
    audit = FeaturesAudit()
    invalid_data = pd.DataFrame({"timestamp": [datetime.now()]})  # Données vides
    result = audit.audit_features(invalid_data, stage="raw")
    assert not result, "Audit réussi pour données invalides"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "DataFrame vide" in str(e) for e in df["error"].dropna()
    ), "Erreur DataFrame vide non loguée"


def test_identify_critical_features(tmp_dirs, mock_feature_importance):
    """Teste la sélection des 70 features critiques par régime."""
    audit = FeaturesAudit()
    critical_features = audit.identify_critical_features("Trend", max_features=70)
    assert len(critical_features) <= 70, "Trop de features critiques"
    assert "spy_lead_return" in critical_features, "Feature momentum absente"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_identify_critical_features" in f and f.endswith(".json")
        for f in snapshot_files
    ), "Snapshot non créé"
    with open(os.path.join(tmp_dirs["snapshots_dir"], snapshot_files[-1]), "r") as f:
        snapshot = json.load(f)
    assert "features" in snapshot["data"], "Features critiques absentes du snapshot"


def test_validate_iqfeed_data(tmp_dirs, mock_data, mock_feature_importance):
    """Teste la validation des données IQFeed."""
    audit = FeaturesAudit()
    expected_features = [
        {"name": "level_4_size_bid", "source": "IQFeed"},
        {"name": "oi_concentration_ratio", "source": "IQFeed"},
        {"name": "vix_term_1m", "source": "IQFeed"},
    ]
    result = audit.validate_iqfeed_data(mock_data, expected_features)
    assert result, "Validation IQFeed échouée pour données valides"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"


def test_validate_shap_features(tmp_dirs, mock_feature_importance):
    """Teste la validation des top 150 SHAP features."""
    audit = FeaturesAudit()
    features = ["spy_lead_return", "atr_14", "vanna_exposure"]
    result = audit.validate_shap_features(features)
    assert result, "Validation SHAP échouée pour features valides"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_validate_shap_features" in f for f in snapshot_files
    ), "Snapshot non créé"


def test_save_snapshot_compressed(tmp_dirs):
    """Teste la sauvegarde d’un snapshot compressé avec gzip."""
    audit = FeaturesAudit()
    snapshot_data = {"test": "compressed_snapshot"}
    audit.save_snapshot("test_compressed", snapshot_data, compress=True)
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


def test_save_dashboard_status(tmp_dirs, mock_data, mock_feature_importance):
    """Teste la sauvegarde du dashboard avec confidence_drop_rate."""
    audit = FeaturesAudit()
    audit.audit_features(mock_data, stage="raw")
    status = {"stage": "raw", "num_features": 11, "confidence_drop_rate": 0.1}
    audit.save_dashboard_status(status)
    assert os.path.exists(tmp_dirs["dashboard_path"]), "Fichier dashboard non créé"
    with open(tmp_dirs["dashboard_path"], "r") as f:
        saved_status = json.load(f)
    assert (
        "confidence_drop_rate" in saved_status
    ), "confidence_drop_rate absent du dashboard"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"


def test_critical_alerts(tmp_dirs, mock_data, mock_feature_importance):
    """Teste les alertes Telegram pour les erreurs critiques."""
    with patch("src.features.features_audit.send_telegram_alert") as mock_telegram:
        audit = FeaturesAudit()
        invalid_data = pd.DataFrame(
            {"timestamp": [datetime.now()]}
        )  # Données invalides
        audit.audit_features(invalid_data, stage="raw")
        mock_telegram.assert_called_with(pytest.any(str))
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "DataFrame vide" in str(e) for e in df["error"].dropna()
        ), "Erreur critique non loguée"
