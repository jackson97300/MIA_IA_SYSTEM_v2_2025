# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_spotgamma_recalculator.py
# Tests unitaires pour src/features/spotgamma_recalculator.py
# Version : 2.1.3
# Date : 2025-05-13
# Rôle : Valide le calcul des métriques d’options (call_wall, put_wall, zero_gamma, etc.) et
#        l’analyse SHAP (Phase 17, limité à 50 features) avec confidence_drop_rate (Phase 8),
#        snapshots avec compression optionnelle, et alertes Telegram.

import gzip
import json
import os
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.features.spotgamma_recalculator import SpotGammaRecalculator


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, snapshots, figures, features, et données."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    snapshots_dir = data_dir / "spotgamma_snapshots"
    snapshots_dir.mkdir()
    figures_dir = data_dir / "figures" / "spotgamma"
    figures_dir.mkdir(parents=True)
    features_dir = data_dir / "features"
    features_dir.mkdir()
    iqfeed_dir = data_dir / "iqfeed"
    iqfeed_dir.mkdir()

    # Créer es_config.yaml
    config_path = config_dir / "es_config.yaml"
    config_content = {
        "spotgamma_recalculator": {
            "buffer_size": 100,
            "max_cache_size": 1000,
            "cache_hours": 24,
            "option_metrics": [
                "call_wall",
                "put_wall",
                "zero_gamma",
                "dealer_position_bias",
                "iv_rank_30d",
                "key_strikes_1",
                "key_strikes_2",
                "key_strikes_3",
                "key_strikes_4",
                "key_strikes_5",
                "max_pain_strike",
                "net_gamma",
                "dealer_zones_count",
                "vol_trigger",
                "ref_px",
                "data_release",
            ],
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
        "perf_log_path": str(logs_dir / "spotgamma_performance.csv"),
        "metrics_path": str(features_dir / "spotgamma_metrics.csv"),
        "shap_path": str(features_dir / "spotgamma_shap.csv"),
        "option_chain_path": str(iqfeed_dir / "option_chain.csv"),
    }


@pytest.fixture
def mock_option_chain():
    """Crée des données factices pour option_chain."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=100, freq="1min"),
            "strike": np.random.uniform(5000, 5200, 100),
            "option_type": np.random.choice(["call", "put"], 100),
            "open_interest": np.random.randint(100, 1000, 100),
            "volume": np.random.randint(10, 100, 100),
            "gamma": np.random.uniform(0, 0.1, 100),
            "delta": np.random.uniform(-1, 1, 100),
            "vega": np.random.uniform(0, 10, 100),
            "price": np.random.uniform(0, 200, 100),
            "underlying_price": np.random.normal(5100, 10, 100),
            "iv_atm": np.random.uniform(0.1, 0.3, 100),
        }
    )


@pytest.fixture
def mock_shap_data(tmp_dirs):
    """Crée un fichier spotgamma_shap.csv factice."""
    shap_data = pd.DataFrame(
        {
            "feature": [
                "call_wall",
                "put_wall",
                "zero_gamma",
                "dealer_position_bias",
                "iv_rank_30d",
                "key_strikes_1",
                "key_strikes_2",
                "key_strikes_3",
                "key_strikes_4",
                "key_strikes_5",
                "max_pain_strike",
                "net_gamma",
                "dealer_zones_count",
                "vol_trigger",
                "ref_px",
                "data_release",
            ]
            * 4,
            "importance": [0.1] * 64,
            "regime": ["range"] * 64,
        }
    )
    shap_data = shap_data.head(50)
    shap_data.to_csv(tmp_dirs["shap_path"], index=False, encoding="utf-8")
    return shap_data


def test_spotgamma_recalculator_init(tmp_dirs, mock_shap_data):
    """Teste l’initialisation de SpotGammaRecalculator."""
    recalculator = SpotGammaRecalculator(config_path=tmp_dirs["config_path"])
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


def test_recalculate_levels_valid(tmp_dirs, mock_option_chain, mock_shap_data):
    """Teste le recalcul des niveaux avec des données valides."""
    recalculator = SpotGammaRecalculator(config_path=tmp_dirs["config_path"])
    timestamp = mock_option_chain["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")
    metrics = recalculator.recalculate_levels(mock_option_chain, timestamp)
    expected_metrics = [
        "call_wall",
        "put_wall",
        "zero_gamma",
        "dealer_position_bias",
        "iv_rank_30d",
        "key_strikes_1",
        "key_strikes_2",
        "key_strikes_3",
        "key_strikes_4",
        "key_strikes_5",
        "max_pain_strike",
        "net_gamma",
        "dealer_zones_count",
        "vol_trigger",
        "ref_px",
        "data_release",
    ]
    assert all(
        metric in metrics for metric in expected_metrics
    ), "Métriques manquantes dans le résultat"
    assert os.path.exists(
        tmp_dirs["metrics_path"]
    ), "Fichier spotgamma_metrics.csv non créé"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_recalculate_levels" in f and f.endswith(".json")
        for f in snapshot_files
    ), "Snapshot non créé"
    with open(os.path.join(tmp_dirs["snapshots_dir"], snapshot_files[-1]), "r") as f:
        snapshot = json.load(f)
    assert (
        "confidence_drop_rate" in snapshot["data"]
    ), "confidence_drop_rate absent du snapshot"
    assert os.path.exists(
        os.path.join(
            tmp_dirs["figures_dir"], f"metrics_{timestamp.replace(':', '-')}.png"
        )
    ), "Visualisation non générée"


def test_recalculate_levels_invalid_data(tmp_dirs, mock_shap_data):
    """Teste le recalcul des niveaux avec des données invalides."""
    recalculator = SpotGammaRecalculator(config_path=tmp_dirs["config_path"])
    invalid_data = pd.DataFrame({"timestamp": [datetime.now()]})  # Données vides
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics = recalculator.recalculate_levels(invalid_data, timestamp)
    assert not metrics, "Résultat non vide pour données invalides"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "DataFrame option_chain vide" in str(e) for e in df["error"].dropna()
    ), "Erreur DataFrame vide non loguée"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_error_recalculate_levels" in f for f in snapshot_files
    ), "Snapshot d’erreur non créé"


def test_calculate_shap_impact(tmp_dirs, mock_option_chain, mock_shap_data):
    """Teste le calcul de l’impact SHAP."""
    recalculator = SpotGammaRecalculator(config_path=tmp_dirs["config_path"])
    shap_importance = recalculator.calculate_shap_impact(mock_option_chain)
    assert len(shap_importance) <= 50, "Nombre de features SHAP dépasse 50"
    assert {"feature", "importance", "regime"}.issubset(
        shap_importance.columns
    ), "Colonnes manquantes dans shap_importance"
    assert os.path.exists(tmp_dirs["shap_path"]), "Fichier spotgamma_shap.csv non créé"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_calculate_shap_impact" in f for f in snapshot_files
    ), "Snapshot non créé"


def test_validate_shap_features(tmp_dirs, mock_shap_data):
    """Teste la validation des top 50 SHAP features."""
    recalculator = SpotGammaRecalculator(config_path=tmp_dirs["config_path"])
    features = ["call_wall", "put_wall", "zero_gamma", "dealer_position_bias"]
    result = recalculator.validate_shap_features(features)
    assert result, "Validation SHAP échouée pour features valides"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_validate_shap_features" in f for f in snapshot_files
    ), "Snapshot non créé"


def test_save_snapshot_compressed(tmp_dirs):
    """Teste la sauvegarde d’un snapshot compressé avec gzip."""
    recalculator = SpotGammaRecalculator(config_path=tmp_dirs["config_path"])
    snapshot_data = {"test": "compressed_snapshot"}
    recalculator.save_snapshot("test_compressed", snapshot_data, compress=True)
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


def test_critical_alerts(tmp_dirs, mock_shap_data):
    """Teste les alertes Telegram pour les erreurs critiques."""
    with patch(
        "src.features.spotgamma_recalculator.send_telegram_alert"
    ) as mock_telegram:
        recalculator = SpotGammaRecalculator(config_path=tmp_dirs["config_path"])
        invalid_data = pd.DataFrame(
            {"timestamp": [datetime.now()]}
        )  # Données invalides
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        recalculator.recalculate_levels(invalid_data, timestamp)
        mock_telegram.assert_called_with(pytest.any(str))
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "DataFrame option_chain vide" in str(e) for e in df["error"].dropna()
        ), "Erreur critique non loguée"
