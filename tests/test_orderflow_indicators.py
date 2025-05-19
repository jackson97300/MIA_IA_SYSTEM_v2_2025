# Placeholder pour test_orderflow_indicators.py
# Rôle : Teste orderflow_indicators.py (Phase 13).
# test_calculate_orderflow_indicators()
# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_orderflow_indicators.py
# Tests unitaires pour la classe OrderFlowIndicators.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie les fonctionnalités de OrderFlowIndicators, incluant le calcul des 16 métriques
#        d’order flow (ex. : delta_volume, obi_score, bid_ask_imbalance), la gestion des erreurs,
#        la pondération par régime, la normalisation, et l'intégration avec IQFeed (Phases 3, 13).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.0,<6.0.0, scikit-learn>=1.5.0,<2.0.0
# - src/features/extractors/orderflow_indicators.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
#
# Notes :
# - Utilise exclusivement des données simulées compatibles avec IQFeed.
# - Suppression de toute référence à dxFeed, obs_t, 320 features, et 81 features.
# - Couvre les Phases 3 (order_flow.py) et 13 (orderflow_indicators.py, options_metrics.py).

import hashlib
import os

import numpy as np
import pandas as pd
import pytest

from src.features.extractors.orderflow_indicators import OrderFlowIndicators

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data", "features", "cache", "orderflow")
FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR, "data", "features", "feature_importance.csv"
)


@pytest.fixture
def orderflow_indicators(tmp_path):
    """Crée une instance de OrderFlowIndicators avec un répertoire temporaire."""
    config_path = tmp_path / "es_config.yaml"
    config_path.write_text(
        """
        orderflow_indicators:
          buffer_size: 10
          max_cache_size: 100
          cache_hours: 24
          weights:
            trend: 1.5
            range: 1.0
            defensive: 0.5
        """
    )
    return OrderFlowIndicators(config_path=str(config_path))


@pytest.fixture
def test_data():
    """Crée des données de test pour OrderFlowIndicators."""
    timestamps = pd.date_range("2025-04-14 09:00", periods=100, freq="1min")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "bid_size_level_1": np.random.randint(50, 500, 100),
            "ask_size_level_1": np.random.randint(50, 500, 100),
            "bid_price_level_1": np.random.normal(5100, 5, 100),
            "ask_price_level_1": np.random.normal(5102, 5, 100),
            "bid_size_level_2": np.random.randint(30, 300, 100),
            "ask_size_level_2": np.random.randint(30, 300, 100),
            "bid_price_level_2": np.random.normal(5098, 5, 100),
            "ask_price_level_2": np.random.normal(5104, 5, 100),
            "bid_size_level_3": np.random.randint(20, 200, 100),
            "ask_size_level_3": np.random.randint(20, 200, 100),
            "bid_size_level_4": np.random.randint(10, 100, 100),
            "ask_size_level_4": np.random.randint(10, 100, 100),
            "bid_size_level_5": np.random.randint(5, 50, 100),
            "ask_size_level_5": np.random.randint(5, 50, 100),
            "trade_frequency_1s": np.random.randint(0, 50, 100),
            "trade_price": np.random.normal(5101, 5, 100),
            "trade_size": np.random.randint(1, 100, 100),
            "close": np.random.normal(5100, 10, 100),
        }
    )


def test_calculate_orderflow_indicators(orderflow_indicators, test_data):
    """Teste calculate_orderflow_indicators."""
    result = orderflow_indicators.calculate_orderflow_indicators(
        test_data, regime="trend"
    )
    assert not result.empty, "Le DataFrame résultat ne doit pas être vide"
    expected_columns = [
        "delta_volume",
        "obi_score",
        "vds",
        "absorption_strength",
        "bid_ask_imbalance",
        "effective_spread",
        "realized_spread",
        "trade_size_variance",
        "aggressive_trade_ratio",
        "hidden_liquidity_ratio",
        "order_imbalance_decay",
        "footprint_delta",
        "order_book_skew_10",
        "trade_flow_acceleration",
        "order_book_momentum",
        "trade_size_skew_1m",
        "delta_volume_weighted",
        "obi_score_weighted",
        "vds_weighted",
        "absorption_strength_weighted",
        "bid_ask_imbalance_weighted",
        "effective_spread_weighted",
        "realized_spread_weighted",
        "trade_size_variance_weighted",
        "aggressive_trade_ratio_weighted",
        "hidden_liquidity_ratio_weighted",
        "order_imbalance_decay_weighted",
        "footprint_delta_weighted",
        "order_book_skew_10_weighted",
        "trade_flow_acceleration_weighted",
        "order_book_momentum_weighted",
        "trade_size_skew_1m_weighted",
        "delta_volume_normalized",
        "obi_score_normalized",
        "vds_normalized",
        "absorption_strength_normalized",
        "bid_ask_imbalance_normalized",
        "effective_spread_normalized",
        "realized_spread_normalized",
        "trade_size_variance_normalized",
        "aggressive_trade_ratio_normalized",
        "hidden_liquidity_ratio_normalized",
        "order_imbalance_decay_normalized",
        "footprint_delta_normalized",
        "order_book_skew_10_normalized",
        "trade_flow_acceleration_normalized",
        "order_book_momentum_normalized",
        "trade_size_skew_1m_normalized",
    ]
    assert all(
        col in result.columns for col in expected_columns
    ), "Colonnes attendues manquantes"
    assert (
        not result["delta_volume"].isna().all()
    ), "delta_volume ne doit pas contenir uniquement des NaN"
    assert all(
        result[f"{col}_weighted"] == result[col] * 1.5 for col in expected_columns[:16]
    ), "Pondération incorrecte pour régime trend"
    assert all(
        result[f"{col}_normalized"].between(0, 1).all()
        or result[f"{col}_normalized"].eq(0).all()
        for col in expected_columns[:16]
    ), "Normalisation hors plage [0,1]"


def test_calculate_orderflow_indicators_empty_data(orderflow_indicators):
    """Teste calculate_orderflow_indicators avec un DataFrame vide."""
    empty_data = pd.DataFrame()
    result = orderflow_indicators.calculate_orderflow_indicators(
        empty_data, regime="range"
    )
    expected_columns = [
        "delta_volume",
        "obi_score",
        "vds",
        "absorption_strength",
        "bid_ask_imbalance",
        "effective_spread",
        "realized_spread",
        "trade_size_variance",
        "aggressive_trade_ratio",
        "hidden_liquidity_ratio",
        "order_imbalance_decay",
        "footprint_delta",
        "order_book_skew_10",
        "trade_flow_acceleration",
        "order_book_momentum",
        "trade_size_skew_1m",
        "delta_volume_weighted",
        "obi_score_weighted",
        "vds_weighted",
        "absorption_strength_weighted",
        "bid_ask_imbalance_weighted",
        "effective_spread_weighted",
        "realized_spread_weighted",
        "trade_size_variance_weighted",
        "aggressive_trade_ratio_weighted",
        "hidden_liquidity_ratio_weighted",
        "order_imbalance_decay_weighted",
        "footprint_delta_weighted",
        "order_book_skew_10_weighted",
        "trade_flow_acceleration_weighted",
        "order_book_momentum_weighted",
        "trade_size_skew_1m_weighted",
        "delta_volume_normalized",
        "obi_score_normalized",
        "vds_normalized",
        "absorption_strength_normalized",
        "bid_ask_imbalance_normalized",
        "effective_spread_normalized",
        "realized_spread_normalized",
        "trade_size_variance_normalized",
        "aggressive_trade_ratio_normalized",
        "hidden_liquidity_ratio_normalized",
        "order_imbalance_decay_normalized",
        "footprint_delta_normalized",
        "order_book_skew_10_normalized",
        "trade_flow_acceleration_normalized",
        "order_book_momentum_normalized",
        "trade_size_skew_1m_normalized",
    ]
    assert result.empty or all(
        result[col].eq(0).all() for col in expected_columns
    ), "Les métriques doivent être à 0 pour un DataFrame vide"


def test_calculate_orderflow_indicators_invalid_regime(orderflow_indicators, test_data):
    """Teste calculate_orderflow_indicators avec un régime invalide."""
    with pytest.raises(ValueError, match="Régime invalide"):
        orderflow_indicators.calculate_orderflow_indicators(test_data, regime="invalid")


def test_compute_bid_ask_imbalance(orderflow_indicators, test_data):
    """Teste compute_bid_ask_imbalance."""
    result = orderflow_indicators.compute_bid_ask_imbalance(test_data)
    assert isinstance(result, pd.Series), "Le résultat doit être une pd.Series"
    assert len(result) == len(
        test_data
    ), "La longueur du résultat doit correspondre aux données d'entrée"
    assert result.name == "bid_ask_imbalance", "Nom de la série incorrect"
    assert (
        not result.isna().all()
    ), "Le résultat ne doit pas contenir uniquement des NaN"


def test_compute_effective_spread(orderflow_indicators, test_data):
    """Teste compute_effective_spread."""
    result = orderflow_indicators.compute_effective_spread(test_data)
    assert isinstance(result, pd.Series), "Le résultat doit être une pd.Series"
    assert len(result) == len(
        test_data
    ), "La longueur du résultat doit correspondre aux données d'entrée"
    assert result.name == "effective_spread", "Nom de la série incorrect"
    assert (
        not result.isna().all()
    ), "Le résultat ne doit pas contenir uniquement des NaN"


def test_cache_indicators(orderflow_indicators, test_data, tmp_path):
    """Teste cache_indicators."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    global CACHE_DIR
    CACHE_DIR = str(cache_dir)  # Redéfinir temporairement CACHE_DIR pour les tests
    result = orderflow_indicators.calculate_orderflow_indicators(
        test_data, regime="range"
    )
    cache_key = hashlib.sha256(f"{result.to_json()}_range".encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.csv")
    assert os.path.exists(
        cache_path
    ), "Le fichier cache doit exister après calculate_orderflow_indicators"
    cached_data = pd.read_csv(cache_path)
    assert len(cached_data) == len(
        result
    ), "Les données mises en cache doivent correspondre aux données calculées"
    CACHE_DIR = os.path.join(
        BASE_DIR, "data", "features", "cache", "orderflow"
    )  # Restaurer CACHE_DIR


def test_validate_shap_features(orderflow_indicators, tmp_path):
    """Teste validate_shap_features avec un fichier SHAP vide."""
    feature_importance_path = tmp_path / "feature_importance.csv"
    pd.DataFrame({"feature": ["dummy_feature"]}).to_csv(feature_importance_path)
    global FEATURE_IMPORTANCE_PATH
    FEATURE_IMPORTANCE_PATH = str(feature_importance_path)  # Redéfinir temporairement
    features = ["delta_volume", "obi_score"]
    assert not orderflow_indicators.validate_shap_features(
        features
    ), "La validation doit échouer avec un fichier SHAP insuffisant"
    FEATURE_IMPORTANCE_PATH = os.path.join(
        BASE_DIR, "data", "features", "feature_importance.csv"
    )  # Restaurer


def test_normalize_indicators(orderflow_indicators, test_data):
    """Teste normalize_indicators."""
    indicators = ["delta_volume", "obi_score"]
    test_data["delta_volume"] = np.random.normal(0, 100, len(test_data))
    test_data["obi_score"] = np.random.normal(0, 1, len(test_data))
    result = orderflow_indicators.normalize_indicators(test_data, indicators)
    assert "delta_volume_normalized" in result.columns, "Colonne normalisée manquante"
    assert "obi_score_normalized" in result.columns, "Colonne normalisée manquante"
    assert (
        result["delta_volume_normalized"].between(0, 1).all()
        or result["delta_volume_normalized"].eq(0).all()
    ), "Normalisation delta_volume hors plage [0,1]"
    assert (
        result["obi_score_normalized"].between(0, 1).all()
        or result["obi_score_normalized"].eq(0).all()
    ), "Normalisation obi_score hors plage [0,1]"


def test_clean_cache(orderflow_indicators, tmp_path):
    """Teste _clean_cache."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    global CACHE_DIR
    CACHE_DIR = str(cache_dir)  # Redéfinir temporairement
    # Créer un fichier cache fictif
    cache_path = os.path.join(CACHE_DIR, "test_cache.csv")
    pd.DataFrame({"dummy": [1, 2, 3]}).to_csv(cache_path)
    orderflow_indicators._clean_cache(max_size_mb=0.0001)  # Forcer suppression
    assert not os.path.exists(cache_path), "Le fichier cache doit être supprimé"
    CACHE_DIR = os.path.join(
        BASE_DIR, "data", "features", "cache", "orderflow"
    )  # Restaurer
