# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_advanced_feature_generator.py
# Tests unitaires pour src/features/advanced_feature_generator.py
# Version : 2.1.3
# Date : 2025-05-13
# Rôle : Valide le calcul des features avancées (latency_spread, iv_atm, option_skew, gex_slope, etc.)
# avec validation SHAP (Phase 17), confidence_drop_rate (Phase 8), et
# snapshots avec alertes Telegram.

import gzip
import json
import os
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.features.advanced_feature_generator import AdvancedFeatureGenerator


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, snapshots, cache, et données."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    snapshots_dir = data_dir / "advanced_snapshots"
    snapshots_dir.mkdir()
    iqfeed_dir = data_dir / "iqfeed"
    iqfeed_dir.mkdir()
    features_dir = data_dir / "features"
    features_dir.mkdir()
    cache_dir = features_dir / "cache" / "advanced"
    cache_dir.mkdir(parents=True)

    # Créer es_config.yaml
    config_path = config_dir / "es_config.yaml"
    config_content = {
        "advanced_feature_generator": {
            "window_size": 5,
            "min_option_rows": 10,
            "buffer_size": 100,
            "max_cache_size": 1000,
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
        "iqfeed_dir": str(iqfeed_dir),
        "features_dir": str(features_dir),
        "cache_dir": str(cache_dir),
        "perf_log_path": str(logs_dir / "advanced_features_performance.csv"),
        "feature_importance_path": str(features_dir / "feature_importance.csv"),
        "option_chain_path": str(iqfeed_dir / "option_chain.csv"),
    }


@pytest.fixture
def mock_data():
    """Crée des données factices pour tester."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=100, freq="1min"),
            "close": np.random.normal(5100, 10, 100),
            "bid_size_level_1": np.random.randint(400, 600, 100),
            "ask_size_level_1": np.random.randint(400, 600, 100),
            "bid_price_level_1": np.random.normal(5099, 5, 100),
            "ask_price_level_1": np.random.normal(5101, 5, 100),
            "bid_size_level_2": np.random.randint(200, 400, 100),
            "ask_size_level_2": np.random.randint(200, 400, 100),
        }
    )


@pytest.fixture
def mock_option_chain(tmp_dirs):
    """Crée des données factices pour option_chain.csv."""
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
            "vanna": np.random.uniform(0.01, 0.05, 30),
            "gamma": np.random.uniform(0.01, 0.05, 30),
            "expiration_date": ["2025-05-20"] * 30,
        }
    )
    option_chain.to_csv(tmp_dirs["option_chain_path"], index=False, encoding="utf-8")
    return option_chain


@pytest.fixture
def mock_feature_importance(tmp_dirs):
    """Crée un fichier feature_importance.csv factice."""
    features = [
        "latency_spread",
        "queue_position_score",
        "vanna_cliff_slope",
        "order_book_pressure_slope",
        "depth_imbalance",
        "vanna_exposure",
        "order_flow_acceleration",
        "vanna_skew",
        "microstructure_volatility",
        "liquidity_absorption_rate",
        "iv_atm",
        "option_skew",
        "gex_slope",
    ] * 12
    shap_data = pd.DataFrame(
        {
            "feature": features[:150],
            "importance": [0.1] * 150,
            "regime": ["range"] * 150,
        }
    )
    shap_data.to_csv(tmp_dirs["feature_importance_path"], index=False, encoding="utf-8")
    return shap_data


def test_advanced_feature_generator_init(tmp_dirs, mock_feature_importance):
    """Teste l’initialisation de AdvancedFeatureGenerator."""
    generator = AdvancedFeatureGenerator(config_path=tmp_dirs["config_path"])
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


def test_calculate_advanced_features_valid(
    tmp_dirs, mock_data, mock_option_chain, mock_feature_importance
):
    """Teste le calcul des features avec des données valides."""
    generator = AdvancedFeatureGenerator(config_path=tmp_dirs["config_path"])
    enriched_data = generator.calculate_advanced_features(
        mock_data, tmp_dirs["option_chain_path"]
    )
    expected_features = [
        "latency_spread",
        "queue_position_score",
        "vanna_cliff_slope",
        "order_book_pressure_slope",
        "depth_imbalance",
        "vanna_exposure",
        "order_flow_acceleration",
        "vanna_skew",
        "microstructure_volatility",
        "liquidity_absorption_rate",
        "iv_atm",
        "option_skew",
        "gex_slope",
    ]
    assert all(
        f in enriched_data.columns for f in expected_features
    ), "Features manquantes dans les données enrichies"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_advanced_features" in f and f.endswith(".json")
        for f in snapshot_files
    ), "Snapshot non créé"
    with open(os.path.join(tmp_dirs["snapshots_dir"], snapshot_files[-1]), "r") as f:
        snapshot = json.load(f)
    assert (
        "confidence_drop_rate" in snapshot["data"]
    ), "confidence_drop_rate absent du snapshot"


def test_calculate_advanced_features_invalid_data(tmp_dirs, mock_feature_importance):
    """Teste le calcul des features avec des données invalides."""
    generator = AdvancedFeatureGenerator(config_path=tmp_dirs["config_path"])
    invalid_data = pd.DataFrame({"timestamp": [datetime.now()]})  # Données vides
    enriched_data = generator.calculate_advanced_features(invalid_data)
    expected_features = [
        "latency_spread",
        "queue_position_score",
        "vanna_cliff_slope",
        "order_book_pressure_slope",
        "depth_imbalance",
        "vanna_exposure",
        "order_flow_acceleration",
        "vanna_skew",
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


def test_calculate_incremental_features(
    tmp_dirs, mock_data, mock_option_chain, mock_feature_importance
):
    """Teste le calcul incrémental des features."""
    generator = AdvancedFeatureGenerator(config_path=tmp_dirs["config_path"])
    row = mock_data.iloc[-1]
    result = generator.calculate_incremental_features(row, mock_option_chain)
    expected_features = [
        "latency_spread",
        "queue_position_score",
        "vanna_cliff_slope",
        "order_book_pressure_slope",
        "depth_imbalance",
        "vanna_exposure",
        "order_flow_acceleration",
        "vanna_skew",
        "microstructure_volatility",
        "liquidity_absorption_rate",
        "iv_atm",
        "option_skew",
        "gex_slope",
    ]
    assert all(
        f in result.index for f in expected_features
    ), "Features manquantes dans le résultat incrémental"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_calculate_incremental_features" in f and f.endswith(".json")
        for f in snapshot_files
    ), "Snapshot non créé"


def test_load_option_chain(tmp_dirs, mock_option_chain, mock_feature_importance):
    """Teste le chargement de la chaîne d’options."""
    generator = AdvancedFeatureGenerator(config_path=tmp_dirs["config_path"])
    option_chain = generator.load_option_chain(tmp_dirs["option_chain_path"])
    assert len(option_chain) == len(
        mock_option_chain
    ), "Nombre incorrect de lignes dans option_chain"
    assert {"timestamp", "underlying_price", "strike", "option_type"}.issubset(
        option_chain.columns
    ), "Colonnes manquantes dans option_chain"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_load_option_chain" in f for f in snapshot_files
    ), "Snapshot non créé"


def test_validate_shap_features(tmp_dirs, mock_feature_importance):
    """Teste la validation des top 150 SHAP features."""
    generator = AdvancedFeatureGenerator(config_path=tmp_dirs["config_path"])
    features = ["latency_spread", "iv_atm", "option_skew", "gex_slope"]
    result = generator.validate_shap_features(features)
    assert result, "Validation SHAP échouée pour features valides"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_validate_shap_features" in f for f in snapshot_files
    ), "Snapshot non créé"


def test_save_snapshot_compressed(tmp_dirs):
    """Teste la sauvegarde d’un snapshot compressé avec gzip."""
    generator = AdvancedFeatureGenerator(config_path=tmp_dirs["config_path"])
    snapshot_data = {"test": "compressed_snapshot"}
    generator.save_snapshot("test_compressed", snapshot_data, compress=True)
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


def test_critical_alerts(tmp_dirs, mock_feature_importance):
    """Teste les alertes Telegram pour les erreurs critiques."""
    with patch(
        "src.features.advanced_feature_generator.send_telegram_alert"
    ) as mock_telegram:
        generator = AdvancedFeatureGenerator(config_path=tmp_dirs["config_path"])
        invalid_data = pd.DataFrame(
            {"timestamp": [datetime.now()]}
        )  # Données invalides
        generator.calculate_advanced_features(invalid_data)
        mock_telegram.assert_called_with(pytest.any(str))
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "DataFrame vide" in str(e) for e in df["error"].dropna()
        ), "Erreur critique non loguée"
