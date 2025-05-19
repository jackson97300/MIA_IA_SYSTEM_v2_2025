# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_volatility_metrics.py
# Tests unitaires pour la classe VolatilityMetricsCalculator.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie les fonctionnalités de VolatilityMetricsCalculator,
# incluant le calcul des 18 métriques de volatilité
# (range_threshold, bollinger_width_20, vix_es_correlation, etc.), la
# gestion des erreurs, la normalisation des métriques, la validation
# SHAP, et l'intégration avec IQFeed (Phase 7).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pandas>=2.0.0,<3.0.0
# - numpy>=1.26.4,<2.0.0
# - ta>=0.11.0,<1.0.0
# - psutil>=5.9.0,<6.0.0
# - scikit-learn>=1.5.0,<2.0.0
# - src/features/extractors/volatility_metrics.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
#
# Notes :
# - Utilise exclusivement des données simulées compatibles avec IQFeed.
# - Suppression de toute référence à dxFeed, obs_t, 320 features, et 81
#   features.
# - Couvre la Phase 7 (volatility_metrics.py).

import hashlib
import os
import signal

import numpy as np
import pandas as pd
import pytest

from src.features.extractors.volatility_metrics import VolatilityMetricsCalculator

# Chemins relatifs
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(__file__))
)
CACHE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "features",
    "cache",
    "volatility",
)
FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "features",
    "feature_importance.csv",
)
SNAPSHOT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "features",
    "volatility_snapshots",
)


@pytest.fixture
def volatility_metrics_calculator(tmp_path):
    """Crée une instance de VolatilityMetricsCalculator avec un
    répertoire temporaire."""
    config_path = tmp_path / "es_config.yaml"
    config_path.write_text(
        """
        volatility_metrics_calculator:
          buffer_size: 10
          max_cache_size: 100
          cache_hours: 24
        """
    )
    return VolatilityMetricsCalculator(config_path=str(config_path))


@pytest.fixture
def test_data():
    """Crée des données de test pour VolatilityMetricsCalculator."""
    timestamps = pd.date_range(
        "2025-04-14 09:00",
        periods=100,
        freq="1min",
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "high": np.random.normal(5105, 10, 100),
            "low": np.random.normal(5095, 10, 100),
            "close": np.random.normal(5100, 10, 100),
            "atr_14": np.random.uniform(0.5, 2.0, 100),
            "vix": np.random.normal(20, 2, 100),
            "es": np.random.normal(5100, 10, 100),
            "iv_30d": np.random.uniform(0.15, 0.25, 100),
            "realized_vol_30d": np.random.uniform(0.1, 0.2, 100),
            "iv_atm": np.random.uniform(0.15, 0.25, 100),
            "vix_term_1m": np.random.normal(20, 2, 100),
            "vix_term_3m": np.random.normal(21, 2, 100),
            "vix_term_6m": np.random.normal(22, 2, 100),
        }
    )


def test_calculate_volatility_metrics(
    volatility_metrics_calculator,
    test_data,
):
    """Teste calculate_volatility_metrics."""
    result = volatility_metrics_calculator.calculate_volatility_metrics(
        test_data
    )
    assert not result.empty, "Le DataFrame résultat ne doit pas être vide"
    expected_columns = [
        "range_threshold",
        "volatility_trend",
        "bollinger_width_20",
        "volatility_regime_stability",
        "vix_es_correlation",
        "volatility_premium",
        "implied_move_ratio",
        "vol_term_slope",
        "vol_term_curvature",
        "realized_vol_5m",
        "realized_vol_15m",
        "realized_vol_30m",
        "volatility_skew_15m",
        "volatility_breakout_signal",
        "session_volatility_index",
        "seasonal_volatility_index",
        "market_open_volatility",
        "market_close_volatility",
        "range_threshold_normalized",
        "volatility_trend_normalized",
        "bollinger_width_20_normalized",
        "volatility_regime_stability_normalized",
        "vix_es_correlation_normalized",
        "volatility_premium_normalized",
        "implied_move_ratio_normalized",
        "vol_term_slope_normalized",
        "vol_term_curvature_normalized",
        "realized_vol_5m_normalized",
        "realized_vol_15m_normalized",
        "realized_vol_30m_normalized",
        "volatility_skew_15m_normalized",
        "volatility_breakout_signal_normalized",
        "session_volatility_index_normalized",
        "seasonal_volatility_index_normalized",
        "market_open_volatility_normalized",
        "market_close_volatility_normalized",
    ]
    assert all(
        col in result.columns for col in expected_columns
    ), "Colonnes attendues manquantes"
    assert not result[
        "range_threshold"
    ].isna().all(), (
        "range_threshold ne doit pas contenir uniquement des NaN"
    )
    assert not result[
        "bollinger_width_20"
    ].isna().all(), (
        "bollinger_width_20 ne doit pas contenir uniquement des NaN"
    )
    assert not result[
        "vix_es_correlation"
    ].isna().all(), (
        "vix_es_correlation ne doit pas contenir uniquement des NaN"
    )
    assert (
        result["range_threshold_normalized"].between(0, 1).all()
        or result["range_threshold_normalized"].eq(0).all()
    ), "Normalisation range_threshold hors plage [0,1]"
    assert (
        result["bollinger_width_20_normalized"].between(0, 1).all()
        or result["bollinger_width_20_normalized"].eq(0).all()
    ), "Normalisation bollinger_width_20 hors plage [0,1]"


def test_calculate_volatility_metrics_empty_data(
    volatility_metrics_calculator,
):
    """Teste calculate_volatility_metrics avec un DataFrame vide."""
    empty_data = pd.DataFrame()
    result = volatility_metrics_calculator.calculate_volatility_metrics(
        empty_data
    )
    expected_columns = [
        "range_threshold",
        "volatility_trend",
        "bollinger_width_20",
        "volatility_regime_stability",
        "vix_es_correlation",
        "volatility_premium",
        "implied_move_ratio",
        "vol_term_slope",
        "vol_term_curvature",
        "realized_vol_5m",
        "realized_vol_15m",
        "realized_vol_30m",
        "volatility_skew_15m",
        "volatility_breakout_signal",
        "session_volatility_index",
        "seasonal_volatility_index",
        "market_open_volatility",
        "market_close_volatility",
        "range_threshold_normalized",
        "volatility_trend_normalized",
        "bollinger_width_20_normalized",
        "volatility_regime_stability_normalized",
        "vix_es_correlation_normalized",
        "volatility_premium_normalized",
        "implied_move_ratio_normalized",
        "vol_term_slope_normalized",
        "vol_term_curvature_normalized",
        "realized_vol_5m_normalized",
        "realized_vol_15m_normalized",
        "realized_vol_30m_normalized",
        "volatility_skew_15m_normalized",
        "volatility_breakout_signal_normalized",
        "session_volatility_index_normalized",
        "seasonal_volatility_index_normalized",
        "market_open_volatility_normalized",
        "market_close_volatility_normalized",
    ]
    assert result.empty or all(
        result[col].eq(0).all() for col in expected_columns
    ), "Les métriques doivent être à 0 pour un DataFrame vide"


def test_calculate_volatility_metrics_missing_columns(
    volatility_metrics_calculator,
    test_data,
):
    """Teste calculate_volatility_metrics avec des colonnes manquantes."""
    incomplete_data = test_data.drop(columns=["high", "low"])
    result = volatility_metrics_calculator.calculate_volatility_metrics(
        incomplete_data
    )
    assert not result.empty, "Le DataFrame résultat ne doit pas être vide"
    assert not result[
        "range_threshold"
    ].eq(0).all(), (
        "range_threshold ne doit pas être 0 si atr_14 est présent"
    )
    assert not result[
        "bollinger_width_20"
    ].eq(0).all(), (
        "bollinger_width_20 ne doit pas être 0 si close est présent"
    )


def test_calculate_vix_es_correlation(
    volatility_metrics_calculator,
    test_data,
):
    """Teste calculate_vix_es_correlation."""
    result = volatility_metrics_calculator.calculate_vix_es_correlation(
        test_data
    )
    assert isinstance(
        result, pd.Series
    ), "Le résultat doit être une pd.Series"
    assert len(result) == len(
        test_data
    ), "La longueur du résultat doit correspondre aux données d'entrée"
    assert (
        result.name == "vix_es_correlation"
    ), "Nom de la série incorrect"
    assert not result.isna().all(), (
        "Le résultat ne doit pas contenir uniquement des NaN"
    )
    # Test sans vix/es
    incomplete_data = test_data.drop(columns=["vix", "es"])
    result_no_vix_es = volatility_metrics_calculator.calculate_vix_es_correlation(
        incomplete_data
    )
    assert result_no_vix_es.eq(
        0
    ).all(), (
        "vix_es_correlation doit être 0 si vix/es manquent"
    )


def test_compute_volatility_premium(
    volatility_metrics_calculator,
    test_data,
):
    """Teste compute_volatility_premium."""
    result = volatility_metrics_calculator.compute_volatility_premium(
        test_data
    )
    assert isinstance(
        result, pd.Series
    ), "Le résultat doit être une pd.Series"
    assert len(result) == len(
        test_data
    ), "La longueur du résultat doit correspondre aux données d'entrée"
    assert (
        result.name == "volatility_premium"
    ), "Nom de la série incorrect"
    assert not result.isna().all(), (
        "Le résultat ne doit pas contenir uniquement des NaN"
    )
    # Test sans iv_30d/realized_vol_30d
    incomplete_data = test_data.drop(
        columns=["iv_30d", "realized_vol_30d"]
    )
    result_no_data = volatility_metrics_calculator.compute_volatility_premium(
        incomplete_data
    )
    assert result_no_data.eq(
        0
    ).all(), (
        "volatility_premium doit être 0 si iv_30d/realized_vol_30d manquent"
    )


def test_cache_metrics(
    volatility_metrics_calculator,
    test_data,
    tmp_path,
):
    """Teste cache_metrics."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    global CACHE_DIR
    CACHE_DIR = str(cache_dir)  # Redéfinir temporairement CACHE_DIR
    result = volatility_metrics_calculator.calculate_volatility_metrics(
        test_data
    )
    cache_key = hashlib.sha256(result.to_json().encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.csv")
    assert os.path.exists(
        cache_path
    ), (
        "Le fichier cache doit exister après calculate_volatility_metrics"
    )
    cached_data = pd.read_csv(cache_path)
    assert len(cached_data) == len(
        result
    ), (
        "Les données mises en cache doivent correspondre aux données "
        "calculées"
    )
    CACHE_DIR = os.path.join(
        BASE_DIR,
        "data",
        "features",
        "cache",
        "volatility",
    )  # Restaurer CACHE_DIR


def test_validate_shap_features(
    volatility_metrics_calculator,
    tmp_path,
):
    """Teste validate_shap_features avec un fichier SHAP vide."""
    feature_importance_path = tmp_path / "feature_importance.csv"
    pd.DataFrame({"feature": ["dummy_feature"]}).to_csv(
        feature_importance_path
    )
    global FEATURE_IMPORTANCE_PATH
    FEATURE_IMPORTANCE_PATH = str(
        feature_importance_path
    )  # Redéfinir temporairement
    features = ["range_threshold", "bollinger_width_20"]
    assert not volatility_metrics_calculator.validate_shap_features(
        features
    ), (
        "La validation doit échouer avec un fichier SHAP insuffisant"
    )
    FEATURE_IMPORTANCE_PATH = os.path.join(
        BASE_DIR,
        "data",
        "features",
        "feature_importance.csv",
    )  # Restaurer


def test_normalize_metrics(
    volatility_metrics_calculator,
    test_data,
):
    """Teste normalize_metrics."""
    metrics = ["range_threshold", "bollinger_width_20"]
    test_data["range_threshold"] = np.random.normal(
        0,
        100,
        len(test_data),
    )
    test_data["bollinger_width_20"] = np.random.normal(
        0,
        1,
        len(test_data),
    )
    result = volatility_metrics_calculator.normalize_metrics(
        test_data,
        metrics,
    )
    assert (
        "range_threshold_normalized" in result.columns
    ), "Colonne normalisée manquante"
    assert (
        "bollinger_width_20_normalized" in result.columns
    ), "Colonne normalisée manquante"
    assert (
        result["range_threshold_normalized"].between(0, 1).all()
        or result["range_threshold_normalized"].eq(0).all()
    ), "Normalisation range_threshold hors plage [0,1]"
    assert (
        result["bollinger_width_20_normalized"].between(0, 1).all()
        or result["bollinger_width_20_normalized"].eq(0).all()
    ), "Normalisation bollinger_width_20 hors plage [0,1]"


def test_clean_cache(
    volatility_metrics_calculator,
    tmp_path,
):
    """Teste _clean_cache."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    global CACHE_DIR
    CACHE_DIR = str(cache_dir)  # Redéfinir temporairement
    # Créer un fichier cache fictif
    cache_path = os.path.join(CACHE_DIR, "test_cache.csv")
    pd.DataFrame({"dummy": [1, 2, 3]}).to_csv(cache_path)
    volatility_metrics_calculator._clean_cache(max_size_mb=0.0001)
    assert not os.path.exists(
        cache_path
    ), "Le fichier cache doit être supprimé"
    CACHE_DIR = os.path.join(
        BASE_DIR,
        "data",
        "features",
        "cache",
        "volatility",
    )  # Restaurer


def test_handle_sigint(
    volatility_metrics_calculator,
    tmp_path,
):
    """Teste handle_sigint."""
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    global SNAPSHOT_DIR
    SNAPSHOT_DIR = str(snapshot_dir)  # Redéfinir temporairement
    with pytest.raises(SystemExit, match="Terminated by SIGINT"):
        volatility_metrics_calculator.handle_sigint(
            signal.SIGINT,
            None,
        )
    snapshot_files = list(snapshot_dir.glob("snapshot_sigint_*.json.gz"))
    assert len(snapshot_files) > 0, "Un snapshot SIGINT doit être créé"
    SNAPSHOT_DIR = os.path.join(
        BASE_DIR,
        "data",
        "features",
        "volatility_snapshots",
    )  # Restaurer