# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_latent_factor_generator.py
# Tests unitaires pour la classe LatentFactorGenerator.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie les fonctionnalités de LatentFactorGenerator, incluant
# le calcul des 21 métriques latentes (ex. : latent_vol_regime_vec_*,
# latent_option_skew_vec, latent_hft_activity_vec), la gestion des
# erreurs, et l'intégration des Phases 1-18.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pandas>=2.0.0,<3.0.0
# - numpy>=1.26.4,<2.0.0
# - scikit-learn>=1.5.0,<2.0.0
# - src/features/extractors/latent_factor_generator.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
#
# Notes :
# - Utilise exclusivement des données simulées compatibles avec IQFeed.
# - Suppression de toute référence à dxFeed, obs_t, 320 features, et 81
#   features.
# - Couvre les Phases 1-18 :
#   - Phase 1 : Métriques topic_vector_news_*.
#   - Phase 13 : Métrique latent_option_skew_vec.
#   - Phase 15 : Métrique latent_microstructure_vec.
#   - Phase 18 : Métrique latent_hft_activity_vec.

import os

import numpy as np
import pandas as pd
import pytest

from src.features.extractors.latent_factor_generator import LatentFactorGenerator

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture
def latent_factor_generator(tmp_path):
    """Crée une instance de LatentFactorGenerator avec un répertoire
    temporaire."""
    config_path = tmp_path / "es_config.yaml"
    config_path.write_text(
        """
        latent_factor_generator:
          buffer_size: 10
          max_cache_size: 100
          cache_hours: 24
        """
    )
    return LatentFactorGenerator(config_path=str(config_path))


@pytest.fixture
def test_data():
    """Crée des données de test pour LatentFactorGenerator."""
    timestamps = pd.date_range("2025-04-14 09:00", periods=100, freq="1min")
    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "atr_14": np.random.normal(1.5, 0.2, 100),
            "vix_term_1m": np.random.normal(20, 1, 100),
            "volatility_trend": np.random.normal(0, 0.1, 100),
            "spy_lead_return": np.random.normal(0.01, 0.005, 100),
            "order_flow_acceleration": np.random.normal(0, 0.1, 100),
            "spy_momentum_diff": np.random.normal(0, 0.05, 100),
            "orderbook_imbalance": np.random.normal(0, 0.2, 100),
            "depth_imbalance": np.random.normal(0, 0.2, 100),
            "neural_regime": np.random.randint(0, 3, 100),
            "vix_es_correlation": np.random.normal(0, 0.1, 100),
        }
    )
    news_data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "topic_score_1": np.random.normal(0, 1, 100),
            "topic_score_2": np.random.normal(0, 1, 100),
            "topic_score_3": np.random.normal(0, 1, 100),
            "topic_score_4": np.random.normal(0, 1, 100),
        }
    )
    ohlc_data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": np.random.normal(5100, 10, 100),
            "high": np.random.normal(5110, 10, 100),
            "low": np.random.normal(5090, 10, 100),
            "close": np.random.normal(5100, 10, 100),
        }
    )
    options_data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "iv_atm": np.random.normal(0.15, 0.02, 100),
            "option_skew": np.random.normal(0.1, 0.01, 100),
            "vix_term_1m": np.random.normal(20, 1, 100),
        }
    )
    microstructure_data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "spoofing_score": np.random.uniform(0, 1, 100),
            "volume_anomaly": np.random.uniform(0, 1, 100),
            "orderbook_velocity": np.random.uniform(0, 1, 100),
        }
    )
    hft_data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "hft_activity_score": np.random.uniform(0, 1, 100),
            "trade_velocity": np.random.uniform(50, 150, 100),
        }
    )
    return {
        "data": data,
        "news_data": news_data,
        "ohlc_data": ohlc_data,
        "options_data": options_data,
        "microstructure_data": microstructure_data,
        "hft_data": hft_data,
    }


def test_compute_latent_features(latent_factor_generator, test_data):
    """Teste compute_latent_features."""
    result = latent_factor_generator.compute_latent_features(
        data=test_data["data"],
        news_data=test_data["news_data"],
        ohlc_data=test_data["ohlc_data"],
        options_data=test_data["options_data"],
        microstructure_data=test_data["microstructure_data"],
        hft_data=test_data["hft_data"],
    )
    assert not result.empty, "Le DataFrame résultat ne doit pas être vide"
    expected_columns = [
        "latent_vol_regime_vec_1",
        "latent_vol_regime_vec_2",
        "latent_vol_regime_vec_3",
        "latent_vol_regime_vec_4",
        "topic_vector_news_1",
        "topic_vector_news_2",
        "topic_vector_news_3",
        "topic_vector_news_4",
        "latent_market_momentum_vec",
        "latent_order_flow_vec_1",
        "latent_order_flow_vec_2",
        "latent_regime_stability_vec",
        "pca_price_1",
        "pca_price_2",
        "pca_price_3",
        "pca_iv_1",
        "pca_iv_2",
        "pca_iv_3",
        "latent_option_skew_vec",
        "latent_microstructure_vec",
        "latent_hft_activity_vec",
    ]
    assert all(
        col in result.columns for col in expected_columns
    ), "Colonnes attendues manquantes"


def test_compute_latent_features_empty_data(latent_factor_generator, test_data):
    """Teste compute_latent_features avec un DataFrame vide."""
    empty_data = pd.DataFrame()
    result = latent_factor_generator.compute_latent_features(
        data=empty_data,
        news_data=test_data["news_data"],
        ohlc_data=test_data["ohlc_data"],
        options_data=test_data["options_data"],
        microstructure_data=test_data["microstructure_data"],
        hft_data=test_data["hft_data"],
    )
    assert result.empty or all(
        result[col].eq(0).all() for col in result.columns if col != "timestamp"
    ), "Les métriques doivent être à 0 pour un DataFrame vide"


def test_compute_t_sne_vol_regime(latent_factor_generator, test_data):
    """Teste compute_t_sne_vol_regime."""
    result = latent_factor_generator.compute_t_sne_vol_regime(
        data=test_data["data"],
        component=3,
    )
    assert isinstance(
        result, pd.DataFrame
    ), "Le résultat doit être un pd.DataFrame"
    assert len(result) == len(
        test_data["data"]
    ), "La longueur du résultat doit correspondre aux données d'entrée"
    expected_columns = {
        "latent_vol_regime_vec_1",
        "latent_vol_regime_vec_2",
        "latent_vol_regime_vec_3",
    }
    assert set(result.columns) == expected_columns, "Colonnes t-SNE incorrectes"
    assert (
        not result.isna().all().all()
    ), "Le résultat ne doit pas contenir uniquement des NaN"


def test_compute_news_topic_vector(latent_factor_generator, test_data):
    """Teste compute_news_topic_vector (Phase 1)."""
    result = latent_factor_generator.compute_news_topic_vector(
        news_data=test_data["news_data"],
        component=3,
    )
    assert isinstance(
        result, pd.DataFrame
    ), "Le résultat doit être un pd.DataFrame"
    assert len(result) == len(
        test_data["news_data"]
    ), "La longueur du résultat doit correspondre aux données d'entrée"
    expected_columns = {
        "topic_vector_news_1",
        "topic_vector_news_2",
        "topic_vector_news_3",
    }
    assert set(result.columns) == expected_columns, "Colonnes t-SNE incorrectes"
    assert (
        not result.isna().all().all()
    ), "Le résultat ne doit pas contenir uniquement des NaN"


def test_compute_latent_option_skew_vec(latent_factor_generator, test_data):
    """Teste compute_latent_option_skew_vec (Phase 13)."""
    result = latent_factor_generator.compute_latent_option_skew_vec(
        options_data=test_data["options_data"]
    )
    assert isinstance(result, pd.Series), "Le résultat doit être une pd.Series"
    assert len(result) == len(
        test_data["options_data"]
    ), "La longueur du résultat doit correspondre aux données d'entrée"
    assert (
        result.name == "latent_option_skew_vec"
    ), "Nom de la série incorrect"
    assert (
        not result.isna().all()
    ), "Le résultat ne doit pas contenir uniquement des NaN"


def test_compute_latent_microstructure_vec(latent_factor_generator, test_data):
    """Teste compute_latent_microstructure_vec (Phase 15)."""
    result = latent_factor_generator.compute_latent_microstructure_vec(
        microstructure_data=test_data["microstructure_data"]
    )
    assert isinstance(result, pd.Series), "Le résultat doit être une pd.Series"
    assert len(result) == len(
        test_data["microstructure_data"]
    ), "La longueur du résultat doit correspondre aux données d'entrée"
    assert (
        result.name == "latent_microstructure_vec"
    ), "Nom de la série incorrect"
    assert (
        not result.isna().all()
    ), "Le résultat ne doit pas contenir uniquement des NaN"


def test_compute_latent_hft_activity_vec(latent_factor_generator, test_data):
    """Teste compute_latent_hft_activity_vec (Phase 18)."""
    result = latent_factor_generator.compute_latent_hft_activity_vec(
        hft_data=test_data["hft_data"]
    )
    assert isinstance(result, pd.Series), "Le résultat doit être une pd.Series"
    assert len(result) == len(
        test_data["hft_data"]
    ), "La longueur du résultat doit correspondre aux données d'entrée"
    assert (
        result.name == "latent_hft_activity_vec"
    ), "Nom de la série incorrect"
    assert (
        not result.isna().all()
    ), "Le résultat ne doit pas contenir uniquement des NaN"


def test_compute_pca_price(latent_factor_generator, test_data):
    """Teste compute_pca pour les prix."""
    result = latent_factor_generator.compute_pca(
        data=test_data["ohlc_data"],
        data_type="price",
        component=1,
    )
    assert isinstance(result, pd.Series), "Le résultat doit être une pd.Series"
    assert len(result) == len(
        test_data["ohlc_data"]
    ), "La longueur du résultat doit correspondre aux données d'entrée"
    assert result.name == "pca_price_1", "Nom de la série incorrect"
    assert (
        not result.isna().all()
    ), "Le résultat ne doit pas contenir uniquement des NaN"


def test_cache_metrics(latent_factor_generator, test_data, tmp_path):
    """Teste cache_metrics."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    global CACHE_DIR
    CACHE_DIR = str(cache_dir)  # Redéfinir temporairement CACHE_DIR pour les tests
    result = latent_factor_generator.compute_latent_features(
        data=test_data["data"],
        news_data=test_data["news_data"],
        ohlc_data=test_data["ohlc_data"],
        options_data=test_data["options_data"],
        microstructure_data=test_data["microstructure_data"],
        hft_data=test_data["hft_data"],
    )
    cache_key = hashlib.sha256(result.to_json().encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.csv")
    assert os.path.exists(
        cache_path
    ), "Le fichier cache doit exister après compute_latent_features"
    cached_data = pd.read_csv(cache_path)
    assert len(cached_data) == len(
        result
    ), "Les données mises en cache doivent correspondre aux données calculées"
    CACHE_DIR = os.path.join(
        BASE_DIR,
        "data",
        "features",
        "cache",
        "latent_factors",
    )  # Restaurer CACHE_DIR


def test_validate_shap_features(latent_factor_generator, tmp_path):
    """Teste validate_shap_features avec un fichier SHAP vide."""
    feature_importance_path = tmp_path / "feature_importance.csv"
    pd.DataFrame({"feature": ["dummy_feature"]}).to_csv(
        feature_importance_path
    )
    global FEATURE_IMPORTANCE_PATH
    FEATURE_IMPORTANCE_PATH = str(
        feature_importance_path
    )  # Redéfinir temporairement
    features = ["latent_vol_regime_vec_1", "latent_option_skew_vec"]
    assert not latent_factor_generator.validate_shap_features(
        features
    ), "La validation doit échouer avec un fichier SHAP insuffisant"
    FEATURE_IMPORTANCE_PATH = os.path.join(
        BASE_DIR,
        "data",
        "features",
        "feature_importance.csv",
    )  # Restaurer