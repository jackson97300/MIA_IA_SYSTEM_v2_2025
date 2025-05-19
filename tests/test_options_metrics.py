# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_options_metrics.py
# Tests unitaires pour la classe OptionsMetricsGenerator.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie les fonctionnalités de OptionsMetricsGenerator, incluant le calcul des 18 métriques
#        d’options (ex. : iv_atm, gex_slope, gamma_peak_distance), la gestion des erreurs,
#        et l'intégration avec IQFeed (Phases 13, 14).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.0,<6.0.0, matplotlib>=3.7.0,<4.0.0
# - src/features/extractors/options_metrics.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
#
# Notes :
# - Utilise exclusivement des données simulées compatibles avec IQFeed.
# - Suppression de toute référence à dxFeed, obs_t, 320 features, et 81 features.
# - Couvre les Phases 13 (options_metrics.py) et 14.

import os

import numpy as np
import pandas as pd
import pytest

from src.features.extractors.options_metrics import OptionsMetricsGenerator

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture
def options_metrics_generator(tmp_path):
    """Crée une instance de OptionsMetricsGenerator avec un répertoire temporaire."""
    config_path = tmp_path / "es_config.yaml"
    config_path.write_text(
        """
        options_metrics_generator:
          buffer_size: 10
          max_cache_size: 100
          cache_hours: 24
        """
    )
    return OptionsMetricsGenerator(config_path=str(config_path))


@pytest.fixture
def test_data():
    """Crée des données de test pour OptionsMetricsGenerator."""
    timestamps = pd.date_range("2025-04-14 09:00", periods=5, freq="1min")
    data = pd.DataFrame(
        {"timestamp": timestamps, "close": [5100, 5102, 5098, 5105, 5100]}
    )
    option_chain = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [timestamps[0]] * 6
                + [timestamps[1]] * 6
                + [timestamps[2]] * 6
                + [timestamps[3]] * 6
                + [timestamps[4]] * 6
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
            "expiration_date": ["2025-04-21"] * 30,
        }
    )
    return data, option_chain


def test_compute_options_metrics(options_metrics_generator, test_data):
    """Teste compute_options_metrics."""
    data, option_chain = test_data
    result = options_metrics_generator.compute_options_metrics(data, option_chain)
    assert not result.empty, "Le DataFrame résultat ne doit pas être vide"
    expected_columns = [
        "iv_atm",
        "gex_slope",
        "gamma_peak_distance",
        "iv_skew",
        "gex_total",
        "oi_peak_call",
        "oi_peak_put",
        "gamma_wall",
        "delta_exposure",
        "theta_pressure",
        "iv_slope",
        "call_put_ratio",
        "iv_atm_change",
        "gex_stability",
        "strike_density",
        "time_to_expiry",
        "iv_atm_call",
        "iv_atm_put",
    ]
    assert all(
        col in result.columns for col in expected_columns
    ), "Colonnes attendues manquantes"
    assert (
        not result["iv_atm"].isna().all()
    ), "iv_atm ne doit pas contenir uniquement des NaN"


def test_compute_options_metrics_empty_data(options_metrics_generator, test_data):
    """Teste compute_options_metrics avec un DataFrame vide."""
    _, option_chain = test_data
    empty_data = pd.DataFrame()
    result = options_metrics_generator.compute_options_metrics(empty_data, option_chain)
    assert result.empty or all(
        result[col].eq(0).all()
        for col in result.columns
        if col not in ["timestamp", "close"]
    ), "Les métriques doivent être à 0 pour un DataFrame vide"


def test_calculate_iv_atm(options_metrics_generator, test_data):
    """Teste calculate_iv_atm."""
    _, option_chain = test_data
    underlying_price = 5100
    result = options_metrics_generator.calculate_iv_atm(option_chain, underlying_price)
    assert isinstance(result, float), "Le résultat doit être un float"
    assert result >= 0, "iv_atm doit être non négatif"


def test_calculate_gex_slope(options_metrics_generator, test_data):
    """Teste calculate_gex_slope."""
    _, option_chain = test_data
    underlying_price = 5100
    result = options_metrics_generator.calculate_gex_slope(
        option_chain, underlying_price
    )
    assert isinstance(result, float), "Le résultat doit être un float"


def test_calculate_gamma_peak_distance(options_metrics_generator, test_data):
    """Teste calculate_gamma_peak_distance."""
    _, option_chain = test_data
    underlying_price = 5100
    result = options_metrics_generator.calculate_gamma_peak_distance(
        option_chain, underlying_price
    )
    assert isinstance(result, float), "Le résultat doit être un float"


def test_cache_metrics(options_metrics_generator, test_data, tmp_path):
    """Teste cache_metrics."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    global CACHE_DIR
    CACHE_DIR = str(cache_dir)  # Redéfinir temporairement CACHE_DIR pour les tests
    data, option_chain = test_data
    result = options_metrics_generator.compute_options_metrics(data, option_chain)
    cache_key = hashlib.sha256(result.to_json().encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.csv")
    assert os.path.exists(
        cache_path
    ), "Le fichier cache doit exister après compute_options_metrics"
    cached_data = pd.read_csv(cache_path)
    assert len(cached_data) == len(
        result
    ), "Les données mises en cache doivent correspondre aux données calculées"
    CACHE_DIR = os.path.join(
        BASE_DIR, "data", "features", "cache", "options_metrics"
    )  # Restaurer CACHE_DIR


def test_validate_shap_features(options_metrics_generator, tmp_path):
    """Teste validate_shap_features avec un fichier SHAP vide."""
    feature_importance_path = tmp_path / "feature_importance.csv"
    pd.DataFrame({"feature": ["dummy_feature"]}).to_csv(feature_importance_path)
    global FEATURE_IMPORTANCE_PATH
    FEATURE_IMPORTANCE_PATH = str(feature_importance_path)  # Redéfinir temporairement
    features = ["iv_atm", "gex_slope"]
    assert not options_metrics_generator.validate_shap_features(
        features
    ), "La validation doit échouer avec un fichier SHAP insuffisant"
    FEATURE_IMPORTANCE_PATH = os.path.join(
        BASE_DIR, "data", "features", "feature_importance.csv"
    )  # Restaurer
