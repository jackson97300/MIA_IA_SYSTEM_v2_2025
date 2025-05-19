# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_cross_asset_processor.py
# Tests unitaires pour la classe CrossAssetProcessor.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie les fonctionnalités de CrossAssetProcessor, incluant le calcul des 9 métriques cross-asset
#        (ex. : gold_correl, option_cross_correl, hft_cross_correl), la gestion des erreurs,
#        et l'intégration des Phases 1-18.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0
# - src/features/extractors/cross_asset_processor.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
#
# Notes :
# - Utilise exclusivement des données simulées compatibles avec IQFeed.
# - Suppression de toute référence à dxFeed, obs_t, 320 features, et 81 features.
# - Couvre les Phases 1-18 :
#   - Phase 1 : Contexte pour l'analyse cross-asset.
#   - Phase 13 : Métrique option_cross_correl.
#   - Phase 15 : Métrique microstructure_cross_impact.
#   - Phase 18 : Métrique hft_cross_correl.

import os

import numpy as np
import pandas as pd
import pytest

from src.features.extractors.cross_asset_processor import CrossAssetProcessor

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture
def cross_asset_processor(tmp_path):
    """Crée une instance de CrossAssetProcessor avec un répertoire temporaire."""
    config_path = tmp_path / "es_config.yaml"
    config_path.write_text(
        """
        cross_asset_processor:
          buffer_size: 10
          max_cache_size: 100
          cache_hours: 24
        """
    )
    return CrossAssetProcessor(config_path=str(config_path))


@pytest.fixture
def test_data():
    """Crée des données de test pour CrossAssetProcessor."""
    timestamps = pd.date_range("2025-04-14 09:00", periods=100, freq="1min")
    es_data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": np.random.normal(5100, 10, 100),
            "volume": np.random.randint(100, 1000, 100),
        }
    )
    spy_data = pd.DataFrame(
        {"timestamp": timestamps, "volume": np.random.randint(100, 1000, 100)}
    )
    asset_data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": np.random.normal(5100, 10, 100),
            "gold_price": np.random.normal(2000, 5, 100),
            "oil_price": np.random.normal(80, 2, 100),
            "btc_price": np.random.normal(60000, 1000, 100),
        }
    )
    bond_data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "yield_10y": np.random.normal(3.5, 0.1, 100),
            "yield_2y": np.random.normal(2.5, 0.1, 100),
        }
    )
    options_data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": np.random.normal(5100, 10, 100),
            "option_skew": np.random.uniform(0, 0.5, 100),
        }
    )
    microstructure_data = pd.DataFrame(
        {"timestamp": timestamps, "spoofing_score": np.random.uniform(0, 1, 100)}
    )
    hft_data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": np.random.normal(5100, 10, 100),
            "hft_activity_score": np.random.uniform(0, 1, 100),
        }
    )
    return {
        "es_data": es_data,
        "spy_data": spy_data,
        "asset_data": asset_data,
        "bond_data": bond_data,
        "options_data": options_data,
        "microstructure_data": microstructure_data,
        "hft_data": hft_data,
    }


def test_compute_cross_asset_features(cross_asset_processor, test_data):
    """Teste compute_cross_asset_features."""
    result = cross_asset_processor.compute_cross_asset_features(
        test_data["es_data"],
        test_data["spy_data"],
        test_data["asset_data"],
        test_data["bond_data"],
        test_data["options_data"],
        test_data["microstructure_data"],
        test_data["hft_data"],
    )
    assert not result.empty, "Le DataFrame résultat ne doit pas être vide"
    expected_columns = [
        "gold_correl",
        "yield_curve_slope",
        "cross_asset_flow_ratio",
        "option_cross_correl",
        "microstructure_cross_impact",
        "hft_cross_correl",
    ]
    assert all(
        col in result.columns for col in expected_columns
    ), "Colonnes attendues manquantes"


def test_compute_cross_asset_features_empty_data(cross_asset_processor, test_data):
    """Teste compute_cross_asset_features avec un DataFrame vide."""
    empty_data = pd.DataFrame()
    result = cross_asset_processor.compute_cross_asset_features(
        empty_data,
        test_data["spy_data"],
        test_data["asset_data"],
        test_data["bond_data"],
        test_data["options_data"],
        test_data["microstructure_data"],
        test_data["hft_data"],
    )
    assert result.empty or all(
        result[col].eq(0).all() for col in result.columns if col != "timestamp"
    ), "Les métriques doivent être à 0 pour un DataFrame vide"


def test_compute_gold_correlation(cross_asset_processor, test_data):
    """Teste compute_correlation pour gold_price."""
    result = cross_asset_processor.compute_correlation(
        test_data["asset_data"], "gold_price"
    )
    assert isinstance(result, pd.Series), "Le résultat doit être une pd.Series"
    assert len(result) == len(
        test_data["asset_data"]
    ), "La longueur du résultat doit correspondre aux données d'entrée"
    assert (
        not result.isna().all()
    ), "Le résultat ne doit pas contenir uniquement des NaN"


def test_compute_option_cross_correl(cross_asset_processor, test_data):
    """Teste compute_option_cross_correl (Phase 13)."""
    result = cross_asset_processor.compute_option_cross_correl(
        test_data["options_data"]
    )
    assert isinstance(result, pd.Series), "Le résultat doit être une pd.Series"
    assert len(result) == len(
        test_data["options_data"]
    ), "La longueur du résultat doit correspondre aux données d'entrée"
    assert (
        not result.isna().all()
    ), "Le résultat ne doit pas contenir uniquement des NaN"


def test_compute_hft_cross_correl(cross_asset_processor, test_data):
    """Teste compute_hft_cross_correl (Phase 18)."""
    result = cross_asset_processor.compute_hft_cross_correl(test_data["hft_data"])
    assert isinstance(result, pd.Series), "Le résultat doit être une pd.Series"
    assert len(result) == len(
        test_data["hft_data"]
    ), "La longueur du résultat doit correspondre aux données d'entrée"
    assert (
        not result.isna().all()
    ), "Le résultat ne doit pas contenir uniquement des NaN"
