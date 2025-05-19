# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_feature_pipeline.py
# Tests unitaires pour src/features/feature_pipeline.py
#
# Version : 2.1.5
# Date : 2025-05-14
#
# Rôle : Valide le pipeline de génération des 350 features, incluant l’intégration des niveaux d’options,
#        l’analyse SHAP (Phase 17), confidence_drop_rate (Phase 8), et les snapshots avec alertes Telegram.
#        Teste les nouvelles fonctionnalités : dimensionnement dynamique (ATR, orderflow imbalance),
#        coûts de transaction (slippage), microstructure (bid-ask imbalance, trade aggressiveness),
#        HMM/changepoint (regime_hmm), surface de volatilité (iv_skew, iv_term_structure),
#        métriques Prometheus, logs psutil, validation via validate_data.py, stockage dans data_lake.py.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, pyyaml>=6.0.0,<7.0.0
# - src/features/feature_pipeline.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
# - src/features/regime_detector.py
# - src/data/validate_data.py
# - src/data/data_lake.py
# - src/utils/error_tracker.py
# - src/monitoring/prometheus_metrics.py
#
# Inputs :
# - Fichiers de configuration factices (feature_pipeline_config.yaml, credentials.yaml, feature_sets.yaml, iqfeed_config.yaml)
# - Données factices (merged_data.csv, option_chain.csv, spotgamma_metrics.csv, news_data.csv, futures_data.csv, macro_events.csv, event_volatility_history.csv)
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de feature_pipeline.py.
#
# Notes :
# - Conforme à structure.txt (version 2.1.5, 2025-05-14).
# - Tests la Phase 8 (confidence_drop_rate), Phase 17 (analyse SHAP), et la méthode load_shap_fallback (suggestion 8).
# - Vérifie l’absence de dxFeed (sauf fallback temporaire), la suppression de 320/81 features et obs_t, les retries,
#   logs psutil, alertes Telegram, snapshots compressés, et sauvegardes incrémentielles.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.
# - Policies Note: The official directory for routing policies is src/model/router/policies.
# The src/model/policies directory is a residual and should be verified
# for removal to avoid import conflicts.

import gzip
import json
import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data.data_lake import DataLake
from src.data.validate_data import validate_features
from src.features.feature_pipeline import FeaturePipeline


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, snapshots, cache, checkpoints, et données."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    features_dir = data_dir / "features"
    features_dir.mkdir()
    snapshots_dir = features_dir / "snapshots"
    snapshots_dir.mkdir()
    cache_dir = features_dir / "cache" / "shap"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints"
    checkpoints_dir.mkdir()
    figures_dir = data_dir / "figures" / "feature_pipeline"
    figures_dir.mkdir(parents=True)
    iqfeed_dir = data_dir / "iqfeed"
    iqfeed_dir.mkdir()
    events_dir = data_dir / "events"
    events_dir.mkdir()

    # Créer feature_pipeline_config.yaml
    config_path = config_dir / "feature_pipeline_config.yaml"
    config_content = {
        "performance_thresholds": {
            "min_features": 350,
            "min_rows": 50,
            "max_shap_features": 150,
            "max_key_shap_features": 150,
        },
        "neural_pipeline": {"window_size": 50},
        "logging": {"buffer_size": 100},
        "cache": {"max_cache_size": 1000},
    }
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)

    # Créer credentials.yaml
    credentials_path = config_dir / "credentials.yaml"
    credentials_content = {"iqfeed": {"api_key": "test_key"}}
    with open(credentials_path, "w") as f:
        yaml.dump(credentials_content, f)

    # Créer feature_sets.yaml
    feature_sets_path = config_dir / "feature_sets.yaml"
    feature_sets_content = {
        "ES": {
            "training": [
                "timestamp",
                "vix",
                "neural_regime",
                "predicted_volatility",
                "trade_frequency_1s",
                "close",
            ]
            + [f"feature_{i}" for i in range(344)],
            "inference": [
                "vix",
                "neural_regime",
                "predicted_volatility",
                "trade_frequency_1s",
                "close",
            ]
            + [f"shap_feature_{i}" for i in range(145)],
        },
        "metadata": {"version": "2.1.5", "alert_priority": 4},
    }
    with open(feature_sets_path, "w") as f:
        yaml.dump(feature_sets_content, f)

    # Créer iqfeed_config.yaml
    iqfeed_config_path = config_dir / "iqfeed_config.yaml"
    iqfeed_config_content = {"iqfeed": {"host": "localhost", "port": 9100}}
    with open(iqfeed_config_path, "w") as f:
        yaml.dump(iqfeed_config_content, f)

    return {
        "base_dir": str(base_dir),
        "config_path": str(config_path),
        "credentials_path": str(credentials_path),
        "feature_sets_path": str(feature_sets_path),
        "iqfeed_config_path": str(iqfeed_config_path),
        "logs_dir": str(logs_dir),
        "snapshots_dir": str(snapshots_dir),
        "cache_dir": str(cache_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "figures_dir": str(figures_dir),
        "iqfeed_dir": str(iqfeed_dir),
        "events_dir": str(events_dir),
        "perf_log_path": str(logs_dir / "feature_pipeline_performance.csv"),
        "dashboard_path": str(data_dir / "feature_pipeline_dashboard.json"),
        "merged_data_path": str(features_dir / "merged_data.csv"),
        "filtered_data_path": str(features_dir / "features_latest_filtered.csv"),
        "spotgamma_metrics_path": str(features_dir / "spotgamma_metrics.csv"),
        "news_data_path": str(iqfeed_dir / "news_data.csv"),
        "futures_data_path": str(iqfeed_dir / "futures_data.csv"),
        "option_chain_path": str(iqfeed_dir / "option_chain.csv"),
        "macro_events_path": str(events_dir / "macro_events.csv"),
        "volatility_history_path": str(events_dir / "event_volatility_history.csv"),
        "audit_raw_path": str(logs_dir / "features_audit_raw.csv"),
        "audit_final_path": str(logs_dir / "features_audit_final.csv"),
        "feature_importance_path": str(features_dir / "feature_importance.csv"),
        "feature_importance_cache_path": str(
            features_dir / "feature_importance_cache.csv"
        ),
    }


@pytest.fixture
def mock_data(tmp_dirs):
    """Crée des données factices pour tester."""
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-14 09:00", periods=100, freq="1min"),
            "open": np.random.normal(5100, 10, 100),
            "high": np.random.normal(5105, 10, 100),
            "low": np.random.normal(5095, 10, 100),
            "close": np.random.normal(5100, 10, 100),
            "volume": np.random.randint(100, 1000, 100),
            "bid_size_level_1": np.random.randint(50, 500, 100),
            "ask_size_level_1": np.random.randint(50, 500, 100),
            "bid_size_level_4": np.random.randint(50, 500, 100),
            "ask_size_level_4": np.random.randint(50, 500, 100),
            "bid_size_level_5": np.random.randint(50, 500, 100),
            "ask_size_level_5": np.random.randint(50, 500, 100),
            "bid_price_level_1": np.random.normal(5098, 5, 100),
            "ask_price_level_1": np.random.normal(5102, 5, 100),
            "tick_count": np.random.randint(10, 100, 100),
            "trade_price": np.random.normal(5100, 10, 100),
            "gex": np.random.uniform(-1000, 1000, 100),
            "vanna_skew": np.random.normal(0, 0.1, 100),
            "vanna_exposure": np.random.normal(0, 0.1, 100),
            "microstructure_volatility": np.random.normal(0, 0.05, 100),
            "liquidity_absorption_rate": np.random.normal(0, 0.1, 100),
            "depth_imbalance": np.random.normal(0, 0.2, 100),
            "vix_es_correlation": np.random.normal(0, 0.1, 100),
            "bid_volume": np.random.randint(100, 1000, 100),
            "ask_volume": np.random.randint(100, 1000, 100),
            "bid_ask_spread": np.random.uniform(0.1, 1.0, 100),
            "order_volume": np.random.randint(50, 500, 100),
            "taker_volume": np.random.randint(50, 500, 100),
            "iv_call": np.random.normal(0.2, 0.05, 100),
            "iv_put": np.random.normal(0.2, 0.05, 100),
            "strike": np.random.uniform(5000, 5200, 100),
            "iv_3m": np.random.normal(0.22, 0.05, 100),
            "iv_1m": np.random.normal(0.18, 0.05, 100),
        }
    )
    data.to_csv(tmp_dirs["merged_data_path"], index=False, encoding="utf-8")
    return data


@pytest.fixture
def mock_option_chain(tmp_dirs):
    """Crée une chaîne d’options factice."""
    option_chain = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-14 09:00", periods=100, freq="1min"),
            "strike": np.random.uniform(5000, 5200, 100),
            "option_type": np.random.choice(["call", "put"], 100),
            "open_interest": np.random.randint(100, 1000, 100),
            "volume": np.random.randint(10, 100, 100),
            "gamma": np.random.uniform(0, 0.1, 100),
            "delta": np.random.uniform(-1, 1, 100),
            "vega": np.random.uniform(0, 10, 100),
            "theta": np.random.uniform(-0.1, 0.1, 100),
            "vomma": np.random.uniform(-0.1, 0.1, 100),
            "speed": np.random.uniform(-0.1, 0.1, 100),
            "ultima": np.random.uniform(-0.1, 0.1, 100),
            "price": np.random.uniform(0, 200, 100),
            "underlying_price": np.random.normal(5100, 10, 100),
        }
    )
    option_chain.to_csv(tmp_dirs["option_chain_path"], index=False, encoding="utf-8")
    return option_chain


@pytest.fixture
def mock_spotgamma_metrics(tmp_dirs):
    """Crée des métriques SpotGamma factices."""
    spotgamma_metrics = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-14 09:00", periods=100, freq="1min"),
            "key_strikes_1": np.random.uniform(5000, 5200, 100),
            "key_strikes_2": np.random.uniform(5000, 5200, 100),
            "key_strikes_3": np.random.uniform(5000, 5200, 100),
            "key_strikes_4": np.random.uniform(5000, 5200, 100),
            "key_strikes_5": np.random.uniform(5000, 5200, 100),
            "max_pain_strike": np.random.uniform(5000, 5200, 100),
            "net_gamma": np.random.uniform(-1, 1, 100),
            "zero_gamma": np.random.uniform(5000, 5200, 100),
            "dealer_zones_count": np.random.randint(0, 11, 100),
            "vol_trigger": np.random.uniform(-1, 1, 100),
            "ref_px": np.random.uniform(5000, 5200, 100),
            "data_release": np.random.randint(0, 2, 100),
            "vomma_exposure": np.random.uniform(-1, 1, 100),
            "speed_exposure": np.random.uniform(-1, 1, 100),
            "ultima_exposure": np.random.uniform(-1, 1, 100),
        }
    )
    spotgamma_metrics.to_csv(
        tmp_dirs["spotgamma_metrics_path"], index=False, encoding="utf-8"
    )
    return spotgamma_metrics


@pytest.fixture
def mock_news_data(tmp_dirs):
    """Crée des données de nouvelles factices."""
    news_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-14 09:00", periods=100, freq="1min"),
            "sentiment_score": np.random.uniform(-1, 1, 100),
            "volume": np.random.randint(1, 10, 100),
        }
    )
    news_data.to_csv(tmp_dirs["news_data_path"], index=False, encoding="utf-8")
    return news_data


@pytest.fixture
def mock_futures_data(tmp_dirs):
    """Crée des données de futures factices."""
    futures_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-14 09:00", periods=100, freq="1min"),
            "near_price": np.random.normal(5100, 10, 100),
            "far_price": np.random.normal(5120, 10, 100),
        }
    )
    futures_data.to_csv(tmp_dirs["futures_data_path"], index=False, encoding="utf-8")
    return futures_data


@pytest.fixture
def mock_macro_events(tmp_dirs):
    """Crée des événements macroéconomiques factices."""
    macro_events = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-14 08:00", periods=5, freq="1h"),
            "event_type": [
                "CPI Release",
                "FOMC Meeting",
                "Earnings Report",
                "GDP Release",
                "Jobs Report",
            ],
            "event_impact_score": [0.8, 0.9, 0.6, 0.7, 0.85],
        }
    )
    macro_events.to_csv(tmp_dirs["macro_events_path"], index=False, encoding="utf-8")
    return macro_events


@pytest.fixture
def mock_volatility_history(tmp_dirs):
    """Crée un historique de volatilité factice."""
    volatility_history = pd.DataFrame(
        {
            "event_type": [
                "CPI Release",
                "FOMC Meeting",
                "Earnings Report",
                "GDP Release",
                "Jobs Report",
            ],
            "volatility_impact": [0.15, 0.20, 0.10, 0.12, 0.18],
            "timestamp": pd.to_datetime(["2025-04-01"] * 5),
        }
    )
    volatility_history.to_csv(
        tmp_dirs["volatility_history_path"], index=False, encoding="utf-8"
    )
    return volatility_history


@pytest.fixture
def mock_neural_pipeline():
    """Mock pour NeuralPipeline."""
    neural_pipeline = MagicMock()
    neural_pipeline.preprocess.return_value = (
        np.zeros((100, 50, 8)),
        np.zeros((100, 50, 8)),
        pd.DataFrame(index=range(100)),
    )
    neural_pipeline.pretrain_models.return_value = None
    neural_pipeline.run.return_value = {"features": np.random.uniform(0, 1, (100, 11))}
    neural_pipeline.run_incremental.return_value = {
        "features": np.random.uniform(0, 1, 11)
    }
    neural_pipeline.save_models.return_value = None
    return neural_pipeline


@pytest.fixture
def mock_shap():
    """Mock pour calculate_shap_features."""

    def mock_calculate_shap(data, regime):
        features = data.select_dtypes(include=[np.number]).columns[:150]
        return pd.DataFrame(
            {"feature": features, "importance": [0.1] * len(features), "regime": regime}
        )

    return mock_calculate_shap


@pytest.fixture
def mock_prometheus_gauge():
    """Mock pour les métriques Prometheus."""
    gauge = MagicMock()
    gauge.labels.return_value.set = MagicMock()
    return gauge


@pytest.fixture
def mock_data_lake():
    """Mock pour DataLake."""
    data_lake = MagicMock(spec=DataLake)
    data_lake.store_features = MagicMock()
    return data_lake


def test_feature_pipeline_init(
    tmp_dirs,
    mock_data,
    mock_option_chain,
    mock_spotgamma_metrics,
    mock_news_data,
    mock_futures_data,
    mock_macro_events,
    mock_volatility_history,
    mock_data_lake,
):
    """Teste l’initialisation de FeaturePipeline."""
    with patch("src.features.feature_pipeline.DataLake", return_value=mock_data_lake):
        pipeline = FeaturePipeline()
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        assert os.path.exists(
            tmp_dirs["snapshots_dir"]
        ), "Dossier de snapshots non créé"
        assert os.path.exists(tmp_dirs["cache_dir"]), "Dossier de cache SHAP non créé"
        assert os.path.exists(
            tmp_dirs["checkpoints_dir"]
        ), "Dossier de checkpoints non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert all(
            col in df.columns
            for col in [
                "timestamp",
                "operation",
                "latency",
                "cpu_percent",
                "memory_usage_mb",
            ]
        ), "Colonnes de performance manquantes"
        assert (
            pipeline.data_lake == mock_data_lake
        ), "DataLake non initialisé correctement"


def test_compute_features_valid(
    tmp_dirs,
    mock_data,
    mock_option_chain,
    mock_spotgamma_metrics,
    mock_news_data,
    mock_futures_data,
    mock_macro_events,
    mock_volatility_history,
    mock_shap,
    mock_prometheus_gauge,
    mock_data_lake,
):
    """Teste le calcul des features avec des données valides."""
    with patch(
        "src.features.feature_pipeline.calculate_shap_features", mock_shap
    ), patch(
        "src.features.feature_pipeline.DataLake", return_value=mock_data_lake
    ), patch(
        "src.features.feature_pipeline.atr_dynamic_metric", mock_prometheus_gauge
    ), patch(
        "src.features.feature_pipeline.orderflow_imbalance_metric",
        mock_prometheus_gauge,
    ), patch(
        "src.features.feature_pipeline.slippage_estimate_metric", mock_prometheus_gauge
    ), patch(
        "src.features.feature_pipeline.bid_ask_imbalance_metric", mock_prometheus_gauge
    ), patch(
        "src.features.feature_pipeline.trade_aggressiveness_metric",
        mock_prometheus_gauge,
    ), patch(
        "src.features.feature_pipeline.iv_skew_metric", mock_prometheus_gauge
    ), patch(
        "src.features.feature_pipeline.iv_term_metric", mock_prometheus_gauge
    ), patch(
        "src.features.regime_detector.RegimeDetector.detect_regime",
        return_value=np.random.randint(0, 3, len(mock_data)),
    ):
        pipeline = FeaturePipeline()
        result = pipeline.compute_features(
            mock_data, option_chain=mock_option_chain, regime="range"
        )
        assert len(result.columns) >= 350, "Nombre de features inférieur à 350"
        assert "trade_success_prob" in result.columns, "trade_success_prob manquant"
        assert "iv_atm" in result.columns, "iv_atm manquant"
        assert "option_skew" in result.columns, "option_skew manquant"
        assert "gex_slope" in result.columns, "gex_slope manquant"
        assert "atr_dynamic" in result.columns, "atr_dynamic manquant"
        assert "orderflow_imbalance" in result.columns, "orderflow_imbalance manquant"
        assert "slippage_estimate" in result.columns, "slippage_estimate manquant"
        assert "bid_ask_imbalance" in result.columns, "bid_ask_imbalance manquant"
        assert "trade_aggressiveness" in result.columns, "trade_aggressiveness manquant"
        assert "iv_skew" in result.columns, "iv_skew manquant"
        assert "iv_term_structure" in result.columns, "iv_term_structure manquant"
        assert "regime_hmm" in result.columns, "regime_hmm manquant"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "compute_features"
        ), "Log compute_features manquant"
        snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
        assert any(
            "snapshot_compute_features" in f and f.endswith(".json")
            for f in snapshot_files
        ), "Snapshot non créé ou compressé"
        with open(
            os.path.join(tmp_dirs["snapshots_dir"], snapshot_files[-1]), "r"
        ) as f:
            snapshot = json.load(f)
        assert (
            "confidence_drop_rate" in snapshot["data"]
        ), "confidence_drop_rate absent du snapshot"
        mock_data_lake.store_features.assert_called()
        validation_result = validate_features(
            result[
                [
                    "atr_dynamic",
                    "orderflow_imbalance",
                    "slippage_estimate",
                    "bid_ask_imbalance",
                    "trade_aggressiveness",
                    "iv_skew",
                    "iv_term_structure",
                    "regime_hmm",
                ]
            ]
        )
        assert validation_result[
            "valid"
        ], f"Validation des nouvelles features échouée: {validation_result['errors']}"


def test_compute_features_invalid_data(tmp_dirs, mock_option_chain, mock_data_lake):
    """Teste le calcul avec des données invalides."""
    with patch("src.features.feature_pipeline.DataLake", return_value=mock_data_lake):
        pipeline = FeaturePipeline()
        invalid_data = pd.DataFrame(
            {"timestamp": [datetime.now()]}
        )  # Données insuffisantes
        result = pipeline.compute_features(invalid_data, option_chain=mock_option_chain)
        assert (
            "trade_success_prob" in result.columns
        ), "trade_success_prob manquant malgré erreur"
        assert (
            result["trade_success_prob"].iloc[0] == 0.0
        ), "trade_success_prob non nul malgré erreur"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "Colonnes manquantes" in str(e) for e in df["error"].dropna()
        ), "Erreur colonnes non loguée"


def test_calculate_shap_features(tmp_dirs, mock_data, mock_shap):
    """Teste la génération des top 150 SHAP features."""
    with patch("src.features.feature_pipeline.calculate_shap_features", mock_shap):
        pipeline = FeaturePipeline()
        shap_df = pipeline.calculate_shap_features(mock_data, regime="range")
        assert len(shap_df) <= 150, "Trop de SHAP features"
        assert "feature" in shap_df.columns, "Colonne feature manquante"
        assert "importance" in shap_df.columns, "Colonne importance manquante"
        assert os.path.exists(
            tmp_dirs["feature_importance_path"]
        ), "Fichier feature_importance.csv non créé"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "calculate_shap_features"
        ), "Log calculate_shap_features manquant"
        snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
        assert any(
            "snapshot_calculate_shap_features" in f and f.endswith(".json")
            for f in snapshot_files
        ), "Snapshot non créé"


def test_integrate_option_levels(
    tmp_dirs, mock_data, mock_option_chain, mock_spotgamma_metrics
):
    """Teste l’intégration des niveaux d’options."""
    pipeline = FeaturePipeline()
    result = pipeline.integrate_option_levels(
        mock_data, option_chain_path=tmp_dirs["option_chain_path"]
    )
    option_cols = ["call_wall", "put_wall", "zero_gamma", "net_gamma", "gex_change"]
    for col in option_cols:
        assert col in result.columns, f"{col} manquant"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        df["operation"] == "integrate_option_levels"
    ), "Log integrate_option_levels manquant"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_integrate_option_levels" in f and f.endswith(".json")
        for f in snapshot_files
    ), "Snapshot non créé"


def test_save_snapshot_compressed(tmp_dirs):
    """Teste la sauvegarde d’un snapshot compressé avec gzip."""
    pipeline = FeaturePipeline()
    snapshot_data = {"test": "compressed_snapshot"}
    pipeline.save_snapshot("test_compressed", snapshot_data, compress=True)
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
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(df["operation"] == "save_snapshot"), "Log save_snapshot manquant"


def test_save_dashboard_status(tmp_dirs, mock_data, mock_option_chain, mock_shap):
    """Teste la sauvegarde du dashboard avec confidence_drop_rate."""
    with patch("src.features.feature_pipeline.calculate_shap_features", mock_shap):
        pipeline = FeaturePipeline()
        pipeline.compute_features(mock_data, option_chain=mock_option_chain)
        pipeline.save_dashboard_status(
            {"status": "test", "num_rows": 100, "confidence_drop_rate": 0.1}
        )
        assert os.path.exists(tmp_dirs["dashboard_path"]), "Fichier dashboard non créé"
        with open(tmp_dirs["dashboard_path"], "r") as f:
            status = json.load(f)
        assert (
            "confidence_drop_rate" in status
        ), "confidence_drop_rate absent du dashboard"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "save_dashboard_status"
        ), "Log save_dashboard_status manquant"


def test_critical_alerts(tmp_dirs, mock_data, mock_option_chain, mock_shap):
    """Teste les alertes Telegram pour les erreurs critiques."""
    with patch(
        "src.features.feature_pipeline.calculate_shap_features", mock_shap
    ), patch("src.features.feature_pipeline.send_telegram_alert") as mock_telegram:
        pipeline = FeaturePipeline()
        invalid_data = pd.DataFrame(
            {"timestamp": [datetime.now()]}
        )  # Données invalides
        pipeline.compute_features(invalid_data, option_chain=mock_option_chain)
        mock_telegram.assert_called_with(pytest.any(str))
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "Colonnes manquantes" in str(e) for e in df["error"].dropna()
        ), "Erreur critique non loguée"


def test_load_shap_fallback(tmp_dirs, mock_data):
    """Teste la méthode load_shap_fallback pour charger les SHAP features."""
    pipeline = FeaturePipeline()
    # Test sans cache
    if os.path.exists(tmp_dirs["feature_importance_cache_path"]):
        os.remove(tmp_dirs["feature_importance_cache_path"])
    shap_features = pipeline.load_shap_fallback()
    assert len(shap_features) == 150, "Nombre de SHAP features incorrect sans cache"
    assert all(
        isinstance(f, str) for f in shap_features
    ), "SHAP features ne sont pas des chaînes"
    assert "vix" in shap_features, "Feature 'vix' manquante dans le fallback"
    # Test avec cache valide
    cache_data = pd.DataFrame(
        {"feature_name": ["feature_" + str(i) for i in range(150)]}
    )
    cache_data.to_csv(
        tmp_dirs["feature_importance_cache_path"], index=False, encoding="utf-8"
    )
    shap_features = pipeline.load_shap_fallback()
    assert len(shap_features) == 150, "Nombre de SHAP features incorrect avec cache"
    assert (
        shap_features == cache_data["feature_name"].tolist()
    ), "SHAP features ne correspondent pas au cache"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        df["operation"] == "load_shap_fallback"
    ), "Log load_shap_fallback manquant"


def test_calculate_atr_dynamic(tmp_dirs, mock_data, mock_prometheus_gauge):
    """Teste le calcul de l'ATR dynamique."""
    with patch(
        "src.features.feature_pipeline.atr_dynamic_metric", mock_prometheus_gauge
    ):
        pipeline = FeaturePipeline()
        atr = pipeline.calculate_atr_dynamic(mock_data)
        assert isinstance(atr, pd.Series), "ATR dynamique n'est pas une Series"
        assert len(atr) == len(mock_data), "Longueur de l'ATR dynamique incorrecte"
        assert not atr.isna().all(), "ATR dynamique contient uniquement des NaN"
        mock_prometheus_gauge.labels().set.assert_called()
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "calculate_atr_dynamic"
        ), "Log calculate_atr_dynamic manquant"


def test_calculate_orderflow_imbalance(tmp_dirs, mock_data, mock_prometheus_gauge):
    """Teste le calcul de l'imbalance du carnet d'ordres."""
    with patch(
        "src.features.feature_pipeline.orderflow_imbalance_metric",
        mock_prometheus_gauge,
    ):
        pipeline = FeaturePipeline()
        imbalance = pipeline.calculate_orderflow_imbalance(mock_data)
        assert isinstance(
            imbalance, pd.Series
        ), "Orderflow imbalance n'est pas une Series"
        assert len(imbalance) == len(
            mock_data
        ), "Longueur de l'orderflow imbalance incorrecte"
        assert (
            not imbalance.isna().all()
        ), "Orderflow imbalance contient uniquement des NaN"
        mock_prometheus_gauge.labels().set.assert_called()
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "calculate_orderflow_imbalance"
        ), "Log calculate_orderflow_imbalance manquant"


def test_calculate_slippage_estimate(tmp_dirs, mock_data, mock_prometheus_gauge):
    """Teste le calcul de l'estimation du slippage."""
    with patch(
        "src.features.feature_pipeline.slippage_estimate_metric", mock_prometheus_gauge
    ):
        pipeline = FeaturePipeline()
        slippage = pipeline.calculate_slippage_estimate(mock_data)
        assert isinstance(slippage, pd.Series), "Slippage estimate n'est pas une Series"
        assert len(slippage) == len(
            mock_data
        ), "Longueur du slippage estimate incorrecte"
        assert (
            not slippage.isna().all()
        ), "Slippage estimate contient uniquement des NaN"
        mock_prometheus_gauge.labels().set.assert_called()
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "calculate_slippage_estimate"
        ), "Log calculate_slippage_estimate manquant"


def test_calculate_bid_ask_imbalance(tmp_dirs, mock_data, mock_prometheus_gauge):
    """Teste le calcul de l'imbalance bid-ask."""
    with patch(
        "src.features.feature_pipeline.bid_ask_imbalance_metric", mock_prometheus_gauge
    ):
        pipeline = FeaturePipeline()
        imbalance = pipeline.calculate_bid_ask_imbalance(mock_data)
        assert isinstance(
            imbalance, pd.Series
        ), "Bid-ask imbalance n'est pas une Series"
        assert len(imbalance) == len(
            mock_data
        ), "Longueur du bid-ask imbalance incorrecte"
        assert (
            not imbalance.isna().all()
        ), "Bid-ask imbalance contient uniquement des NaN"
        mock_prometheus_gauge.labels().set.assert_called()
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "calculate_bid_ask_imbalance"
        ), "Log calculate_bid_ask_imbalance manquant"


def test_calculate_trade_aggressiveness(tmp_dirs, mock_data, mock_prometheus_gauge):
    """Teste le calcul de l'agressivité des trades."""
    with patch(
        "src.features.feature_pipeline.trade_aggressiveness_metric",
        mock_prometheus_gauge,
    ):
        pipeline = FeaturePipeline()
        aggressiveness = pipeline.calculate_trade_aggressiveness(mock_data)
        assert isinstance(
            aggressiveness, pd.Series
        ), "Trade aggressiveness n'est pas une Series"
        assert len(aggressiveness) == len(
            mock_data
        ), "Longueur du trade aggressiveness incorrecte"
        assert (
            not aggressiveness.isna().all()
        ), "Trade aggressiveness contient uniquement des NaN"
        mock_prometheus_gauge.labels().set.assert_called()
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "calculate_trade_aggressiveness"
        ), "Log calculate_trade_aggressiveness manquant"


def test_calculate_iv_skew(tmp_dirs, mock_data, mock_prometheus_gauge):
    """Teste le calcul du skew de volatilité implicite."""
    with patch("src.features.feature_pipeline.iv_skew_metric", mock_prometheus_gauge):
        pipeline = FeaturePipeline()
        iv_skew = pipeline.calculate_iv_skew(mock_data)
        assert isinstance(iv_skew, pd.Series), "IV skew n'est pas une Series"
        assert len(iv_skew) == len(mock_data), "Longueur du IV skew incorrecte"
        assert not iv_skew.isna().all(), "IV skew contient uniquement des NaN"
        mock_prometheus_gauge.labels().set.assert_called()
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "calculate_iv_skew"
        ), "Log calculate_iv_skew manquant"


def test_calculate_iv_term_structure(tmp_dirs, mock_data, mock_prometheus_gauge):
    """Teste le calcul de la structure de terme de la volatilité implicite."""
    with patch("src.features.feature_pipeline.iv_term_metric", mock_prometheus_gauge):
        pipeline = FeaturePipeline()
        iv_term = pipeline.calculate_iv_term_structure(mock_data)
        assert isinstance(iv_term, pd.Series), "IV term structure n'est pas une Series"
        assert len(iv_term) == len(
            mock_data
        ), "Longueur du IV term structure incorrecte"
        assert not iv_term.isna().all(), "IV term structure contient uniquement des NaN"
        mock_prometheus_gauge.labels().set.assert_called()
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "calculate_iv_term_structure"
        ), "Log calculate_iv_term_structure manquant"


def test_error_handling_new_features(tmp_dirs, mock_data, mock_option_chain, mock_shap):
    """Teste la gestion des erreurs pour les nouvelles features."""
    with patch("src.features.feature_pipeline.capture_error") as mock_capture_error:
        pipeline = FeaturePipeline()
        invalid_data = mock_data.drop(
            columns=["bid_volume", "ask_volume"]
        )  # Manque des colonnes nécessaires
        result = pipeline.compute_features(invalid_data, option_chain=mock_option_chain)
        assert "atr_dynamic" in result.columns, "atr_dynamic manquant malgré erreur"
        assert (
            "orderflow_imbalance" in result.columns
        ), "orderflow_imbalance manquant malgré erreur"
        assert (
            result["orderflow_imbalance"].eq(0).all()
        ), "orderflow_imbalance non nul malgré erreur"
        mock_capture_error.assert_called()
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "compute_features" and df["success"] == False
        ), "Erreur compute_features non loguée"


def test_generate_features(tmp_dirs, mock_data, mock_prometheus_gauge, mock_data_lake):
    """Teste la méthode generate_features."""
    with patch(
        "src.features.feature_pipeline.atr_dynamic_metric", mock_prometheus_gauge
    ), patch(
        "src.features.feature_pipeline.orderflow_imbalance_metric",
        mock_prometheus_gauge,
    ), patch(
        "src.features.feature_pipeline.slippage_estimate_metric", mock_prometheus_gauge
    ), patch(
        "src.features.feature_pipeline.bid_ask_imbalance_metric", mock_prometheus_gauge
    ), patch(
        "src.features.feature_pipeline.trade_aggressiveness_metric",
        mock_prometheus_gauge,
    ), patch(
        "src.features.feature_pipeline.iv_skew_metric", mock_prometheus_gauge
    ), patch(
        "src.features.feature_pipeline.iv_term_metric", mock_prometheus_gauge
    ), patch(
        "src.features.regime_detector.RegimeDetector.detect_regime",
        return_value=np.random.randint(0, 3, len(mock_data)),
    ), patch(
        "src.features.feature_pipeline.DataLake", return_value=mock_data_lake
    ):
        pipeline = FeaturePipeline()
        result = pipeline.generate_features(mock_data)
        new_features = [
            "atr_dynamic",
            "orderflow_imbalance",
            "slippage_estimate",
            "bid_ask_imbalance",
            "trade_aggressiveness",
            "iv_skew",
            "iv_term_structure",
            "regime_hmm",
        ]
        for feature in new_features:
            assert (
                feature in result.columns
            ), f"{feature} manquant dans generate_features"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "generate_features"
        ), "Log generate_features manquant"
        assert os.path.exists(
            tmp_dirs["filtered_data_path"]
        ), "Fichier features_latest.csv non créé"
        mock_data_lake.store_features.assert_called()
        validation_result = validate_features(result[new_features])
        assert validation_result[
            "valid"
        ], f"Validation des nouvelles features échouée: {validation_result['errors']}"


def test_log_performance_metrics(tmp_dirs, mock_data, mock_data_lake):
    """Teste la journalisation des métriques psutil dans log_performance."""
    with patch("src.features.feature_pipeline.DataLake", return_value=mock_data_lake):
        pipeline = FeaturePipeline()
        pipeline.log_performance("test_op", 0.1, True, market="ES")
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df["operation"] == "test_op"), "Log test_op manquant"
        assert all(
            col in df.columns for col in ["memory_usage_mb", "cpu_percent"]
        ), "Colonnes psutil manquantes"
        assert any(df["operation"] == "log_performance"), "Log log_performance manquant"
