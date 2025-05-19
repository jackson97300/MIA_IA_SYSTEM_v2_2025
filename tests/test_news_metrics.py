# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_news_metrics.py
# Tests unitaires pour la classe NewsMetricsGenerator.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie les fonctionnalités de NewsMetricsGenerator, incluant le calcul des métriques
#        news_impact_score, news_frequency_1h, news_frequency_1d (Phases 1, 14),
#        la gestion des erreurs, et l'intégration avec IQFeed.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.0,<6.0.0
# - src/features/extractors/news_metrics.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
#
# Notes :
# - Utilise exclusivement des données simulées compatibles avec IQFeed.
# - Suppression de toute référence à dxFeed, obs_t, 320 features, et 81 features.
# - Couvre les Phases 1 (news_analyzer.py) et 14.

import os

import numpy as np
import pandas as pd
import pytest

from src.features.extractors.news_metrics import NewsMetricsGenerator

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture
def news_metrics_generator(tmp_path):
    """Crée une instance de NewsMetricsGenerator avec un répertoire temporaire."""
    config_path = tmp_path / "es_config.yaml"
    config_path.write_text(
        """
        news_metrics_generator:
          buffer_size: 10
          max_cache_size: 100
          cache_hours: 24
        """
    )
    return NewsMetricsGenerator(config_path=str(config_path))


@pytest.fixture
def test_news_data():
    """Crée des données de test pour NewsMetricsGenerator."""
    timestamps = pd.date_range("2025-04-14 09:00", periods=100, freq="1min")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "sentiment_score": np.random.uniform(-1, 1, 100),
            "source": np.random.choice(["DTN", "Benzinga", "DowJones", "unknown"], 100),
        }
    )


def test_compute_news_metrics(news_metrics_generator, test_news_data):
    """Teste compute_news_metrics."""
    result = news_metrics_generator.compute_news_metrics(test_news_data)
    assert not result.empty, "Le DataFrame résultat ne doit pas être vide"
    expected_columns = ["news_impact_score", "news_frequency_1h", "news_frequency_1d"]
    assert all(
        col in result.columns for col in expected_columns
    ), "Colonnes attendues manquantes"
    assert (
        not result["news_impact_score"].isna().all()
    ), "news_impact_score ne doit pas contenir uniquement des NaN"


def test_compute_news_metrics_empty_data(news_metrics_generator):
    """Teste compute_news_metrics avec un DataFrame vide."""
    empty_data = pd.DataFrame()
    result = news_metrics_generator.compute_news_metrics(empty_data)
    assert result.empty or all(
        result[col].eq(0).all()
        for col in ["news_impact_score", "news_frequency_1h", "news_frequency_1d"]
    ), "Les métriques doivent être à 0 pour un DataFrame vide"


def test_compute_news_impact_score(news_metrics_generator, test_news_data):
    """Teste compute_news_impact_score."""
    result = news_metrics_generator.compute_news_impact_score(test_news_data)
    assert isinstance(result, pd.Series), "Le résultat doit être une pd.Series"
    assert len(result) == len(
        test_news_data
    ), "La longueur du résultat doit correspondre aux données d'entrée"
    assert result.name == "news_impact_score", "Nom de la série incorrect"
    assert (
        not result.isna().all()
    ), "Le résultat ne doit pas contenir uniquement des NaN"


def test_compute_news_frequency(news_metrics_generator, test_news_data):
    """Teste compute_news_frequency."""
    result = news_metrics_generator.compute_news_frequency(test_news_data, "1h")
    assert isinstance(result, pd.Series), "Le résultat doit être une pd.Series"
    assert len(result) == len(
        test_news_data
    ), "La longueur du résultat doit correspondre aux données d'entrée"
    assert result.name == "news_frequency_1h", "Nom de la série incorrect"
    assert (
        not result.isna().all()
    ), "Le résultat ne doit pas contenir uniquement des NaN"


def test_cache_metrics(news_metrics_generator, test_news_data, tmp_path):
    """Teste cache_metrics."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    global CACHE_DIR
    CACHE_DIR = str(cache_dir)  # Redéfinir temporairement CACHE_DIR pour les tests
    result = news_metrics_generator.compute_news_metrics(test_news_data)
    cache_key = hashlib.sha256(result.to_json().encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.csv")
    assert os.path.exists(
        cache_path
    ), "Le fichier cache doit exister après compute_news_metrics"
    cached_data = pd.read_csv(cache_path)
    assert len(cached_data) == len(
        result
    ), "Les données mises en cache doivent correspondre aux données calculées"
    CACHE_DIR = os.path.join(
        BASE_DIR, "data", "features", "cache", "news_metrics"
    )  # Restaurer CACHE_DIR


def test_validate_shap_features(news_metrics_generator, tmp_path):
    """Teste validate_shap_features avec un fichier SHAP vide."""
    feature_importance_path = tmp_path / "feature_importance.csv"
    pd.DataFrame({"feature": ["dummy_feature"]}).to_csv(feature_importance_path)
    global FEATURE_IMPORTANCE_PATH
    FEATURE_IMPORTANCE_PATH = str(feature_importance_path)  # Redéfinir temporairement
    features = ["news_impact_score", "news_frequency_1h"]
    assert not news_metrics_generator.validate_shap_features(
        features
    ), "La validation doit échouer avec un fichier SHAP insuffisant"
    FEATURE_IMPORTANCE_PATH = os.path.join(
        BASE_DIR, "data", "features", "feature_importance.csv"
    )  # Restaurer
