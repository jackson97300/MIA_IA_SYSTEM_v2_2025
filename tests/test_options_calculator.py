# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_options_calculator.py
# Tests unitaires pour src/features/options_calculator.py
# Version : 2.1.3
# Date : 2025-05-13
# Rôle : Valide le calcul des métriques d’options (Greeks, IV, OI, skews, etc.) avec validation SHAP (Phase 17),
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

from src.features.options_calculator import OptionsCalculator


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
    snapshots_dir = data_dir / "options_snapshots"
    snapshots_dir.mkdir()
    figures_dir = data_dir / "figures" / "options"
    figures_dir.mkdir(parents=True)
    features_dir = data_dir / "features"
    features_dir.mkdir()
    cache_dir = features_dir / "cache" / "options"
    cache_dir.mkdir(parents=True)
    options_dir = data_dir / "options"
    options_dir.mkdir()

    # Créer es_config.yaml
    config_path = config_dir / "es_config.yaml"
    config_content = {
        "options_calculator": {
            "option_chain_path": str(options_dir / "option_chain.csv"),
            "time_tolerance": "10s",
            "min_strikes": 5,
            "expiration_window": "30d",
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
        "options_dir": str(options_dir),
        "perf_log_path": str(logs_dir / "options_calculator_performance.csv"),
        "feature_importance_path": str(features_dir / "feature_importance.csv"),
        "option_chain_path": str(options_dir / "option_chain.csv"),
    }


@pytest.fixture
def mock_data():
    """Crée des données factices pour tester."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=5, freq="1min"),
            "close": [5100, 5102, 5098, 5105, 5100],
        }
    )


@pytest.fixture
def mock_option_chain(tmp_dirs):
    """Crée un fichier option_chain.csv factice."""
    option_chain = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-05-13 09:00"] * 6
                + ["2025-05-13 09:01"] * 6
                + ["2025-05-13 09:02"] * 6
                + ["2025-05-13 09:03"] * 6
                + ["2025-05-13 09:04"] * 6
            ),
            "underlying_price": [5100] * 6
            + [5102] * 6
            + [5098] * 6
            + [5105] * 6
            + [5100] * 6,
            "strike": [5080, 5090, 5100, 5110, 5120, 5130] * 5,
            "option_type": ["call", "call", "call", "put", "put", "put"] * 5,
            "implied_volatility": np.random.uniform(0.1, 0.3, 30),
            "open_interest": np.random.randint(100, 1000, 30),
            "gamma": np.random.uniform(0.01, 0.05, 30),
            "delta": np.random.uniform(-0.5, 0.5, 30),
            "theta": np.random.uniform(-0.1, -0.01, 30),
            "vega": np.random.uniform(0.05, 0.15, 30),
            "rho": np.random.uniform(-0.02, 0.02, 30),
            "vomma": np.random.uniform(0.01, 0.03, 30),
            "speed": np.random.uniform(-0.01, 0.01, 30),
            "ultima": np.random.uniform(-0.005, 0.005, 30),
            "expiration_date": ["2025-05-20"] * 30,
            "price": np.random.uniform(1.0, 10.0, 30),
            "bid_price": np.random.uniform(0.8, 9.0, 30),
            "ask_price": np.random.uniform(1.0, 10.0, 30),
            "volume": np.random.randint(10, 100, 30),
        }
    )
    option_chain.to_csv(tmp_dirs["option_chain_path"], index=False, encoding="utf-8")
    return option_chain


@pytest.fixture
def mock_feature_importance(tmp_dirs):
    """Crée un fichier feature_importance.csv factice."""
    shap_data = pd.DataFrame(
        {
            "feature": [
                "gex",
                "iv_atm",
                "option_skew",
                "oi_concentration",
                "gex_slope",
                "oi_peak_call_near",
                "oi_peak_put_near",
                "gamma_wall",
                "vanna_strike_near",
                "delta_hedge_pressure",
                "iv_skew_10delta",
                "max_pain_strike",
                "gamma_velocity_near",
                "short_dated_iv_slope",
                "vol_term_structure_slope",
                "term_structure_skew",
                "risk_reversal_25",
                "butterfly_25",
                "oi_concentration_ratio",
                "gamma_bucket_exposure",
                "delta_exposure_ratio",
                "open_interest_change_1h",
                "gamma_trend_30m",
                "vega_bucket_ratio",
                "atm_straddle_cost",
                "spot_gamma_corridor",
                "skewed_gamma_exposure",
                "oi_sweep_count",
                "theta_exposure",
                "vega_exposure",
                "rho_exposure",
                "vomma_exposure",
                "speed_exposure",
                "ultima_exposure",
                "iv_slope_10_25",
                "iv_slope_25_50",
                "iv_slope_50_75",
                "vol_surface_curvature",
                "iv_acceleration",
                "oi_velocity",
                "call_put_volume_ratio",
                "option_spread_cost",
            ]
            * 4,
            "importance": [0.1] * 168,
            "regime": ["range"] * 168,
        }
    )
    shap_data.to_csv(tmp_dirs["feature_importance_path"], index=False, encoding="utf-8")
    return shap_data


def test_options_calculator_init(tmp_dirs, mock_feature_importance, mock_option_chain):
    """Teste l’initialisation de OptionsCalculator."""
    calculator = OptionsCalculator(config_path=tmp_dirs["config_path"])
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


def test_calculate_options_features_valid(
    tmp_dirs, mock_data, mock_option_chain, mock_feature_importance
):
    """Teste le calcul des features avec des données valides."""
    calculator = OptionsCalculator(config_path=tmp_dirs["config_path"])
    enriched_data = calculator.calculate_options_features(mock_data)
    expected_features = [
        "gex",
        "iv_atm",
        "option_skew",
        "oi_concentration",
        "gex_slope",
        "oi_peak_call_near",
        "oi_peak_put_near",
        "gamma_wall",
        "vanna_strike_near",
        "delta_hedge_pressure",
    ]
    assert all(
        f in enriched_data.columns for f in expected_features
    ), "Features manquantes dans les données enrichies"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_calculate_options_features" in f and f.endswith(".json")
        for f in snapshot_files
    ), "Snapshot non créé"
    with open(os.path.join(tmp_dirs["snapshots_dir"], snapshot_files[-1]), "r") as f:
        snapshot = json.load(f)
    assert (
        "confidence_drop_rate" in snapshot["data"]
    ), "confidence_drop_rate absent du snapshot"
    timestamp_safe = mock_data["timestamp"].iloc[-1].strftime("%Y-%m-%d %H-%M-%S")
    assert os.path.exists(
        os.path.join(
            tmp_dirs["figures_dir"], f"options_features_temporal_{timestamp_safe}.png"
        )
    ), "Visualisation temporelle non générée"


def test_calculate_options_features_invalid_data(
    tmp_dirs, mock_feature_importance, mock_option_chain
):
    """Teste le calcul des features avec des données invalides."""
    calculator = OptionsCalculator(config_path=tmp_dirs["config_path"])
    invalid_data = pd.DataFrame({"timestamp": [datetime.now()]})  # Données vides
    enriched_data = calculator.calculate_options_features(invalid_data)
    expected_features = [
        "gex",
        "iv_atm",
        "option_skew",
        "oi_concentration",
        "gex_slope",
    ]
    assert all(
        f in enriched_data.columns for f in expected_features
    ), "Features manquantes dans les données enrichies"
    assert (
        enriched_data[expected_features].eq(0.0).all().all()
    ), "Features non nulles pour données invalides"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "DataFrame vide" in str(e) for e in df["error"].dropna()
    ), "Erreur DataFrame vide non loguée"


def test_calculate_incremental_options_features(
    tmp_dirs, mock_data, mock_option_chain, mock_feature_importance
):
    """Teste le calcul incrémental des features."""
    calculator = OptionsCalculator(config_path=tmp_dirs["config_path"])
    row = mock_data.iloc[-1]
    result = calculator.calculate_incremental_options_features(row, mock_option_chain)
    expected_features = [
        "gex",
        "iv_atm",
        "option_skew",
        "oi_concentration",
        "gex_slope",
        "oi_peak_call_near",
        "oi_peak_put_near",
        "gamma_wall",
        "vanna_strike_near",
        "delta_hedge_pressure",
    ]
    assert all(
        f in result.index for f in expected_features
    ), "Features manquantes dans le résultat incrémental"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_calculate_incremental_options_features" in f and f.endswith(".json")
        for f in snapshot_files
    ), "Snapshot non créé"


def test_validate_shap_features(tmp_dirs, mock_feature_importance):
    """Teste la validation des top 150 SHAP features."""
    calculator = OptionsCalculator(config_path=tmp_dirs["config_path"])
    features = ["gex", "iv_atm", "option_skew"]
    result = calculator.validate_shap_features(features)
    assert result, "Validation SHAP échouée pour features valides"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_validate_shap_features" in f for f in snapshot_files
    ), "Snapshot non créé"


def test_save_snapshot_compressed(tmp_dirs):
    """Teste la sauvegarde d’un snapshot compressé avec gzip."""
    calculator = OptionsCalculator(config_path=tmp_dirs["config_path"])
    snapshot_data = {"test": "compressed_snapshot"}
    calculator.save_snapshot("test_compressed", snapshot_data, compress=True)
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


def test_plot_options_features(
    tmp_dirs, mock_data, mock_option_chain, mock_feature_importance
):
    """Teste la génération des visualisations des features."""
    calculator = OptionsCalculator(config_path=tmp_dirs["config_path"])
    enriched_data = calculator.calculate_options_features(mock_data)
    timestamp = mock_data["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")
    calculator.plot_options_features(enriched_data, timestamp)
    timestamp_safe = timestamp.replace(":", "-")
    assert os.path.exists(
        os.path.join(
            tmp_dirs["figures_dir"], f"options_features_temporal_{timestamp_safe}.png"
        )
    ), "Visualisation temporelle non générée"
    assert os.path.exists(
        os.path.join(
            tmp_dirs["figures_dir"],
            f"options_features_distribution_{timestamp_safe}.png",
        )
    ), "Distribution non générée"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"


def test_critical_alerts(tmp_dirs, mock_feature_importance):
    """Teste les alertes Telegram pour les erreurs critiques."""
    with patch("src.features.options_calculator.send_telegram_alert") as mock_telegram:
        calculator = OptionsCalculator(config_path=tmp_dirs["config_path"])
        invalid_data = pd.DataFrame(
            {"timestamp": [datetime.now()]}
        )  # Données invalides
        calculator.calculate_options_features(invalid_data)
        mock_telegram.assert_called_with(pytest.any(str))
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "DataFrame vide" in str(e) for e in df["error"].dropna()
        ), "Erreur critique non loguée"
