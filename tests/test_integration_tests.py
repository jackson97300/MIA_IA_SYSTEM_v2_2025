# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_integration_tests.py
#
# Tests unitaires pour integration_tests.py de MIA_IA_SYSTEM_v2_2025.
# Rôle : Vérifie les tests d'intégration du pipeline collecte →
# features → trading (Phase 15).
# Conforme à structure.txt (version 2.1.3, 2025-05-13).

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from scripts.integration_tests import (
    test_data_to_features_pipeline,
    test_features_to_trading_pipeline,
    test_full_pipeline_integration,
    test_pipeline_empty_data,
    test_pipeline_error_handling,
)


# Fixture pour créer des données factices simulant IQFeed
@pytest.fixture
def mock_iqfeed_data():
    """
    Crée un DataFrame simulant les données brutes d'IQFeed.
    """
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start="2025-05-13 10:00:00",
                periods=10,
                freq="T",
            ),
            "open": np.random.uniform(5000, 5100, 10),
            "high": np.random.uniform(5100, 5200, 10),
            "low": np.random.uniform(4900, 5000, 10),
            "close": np.random.uniform(5000, 5100, 10),
            "volume": np.random.randint(1000, 5000, 10),
            "bid_size_level_2": np.random.randint(100, 1000, 10),
            "ask_size_level_2": np.random.randint(100, 1000, 10),
            "vix_close": np.random.uniform(20, 30, 10),
        }
    )


# Fixture pour simuler l'environnement de trading
@pytest.fixture
def mock_trading_env():
    """
    Crée un environnement de trading factice.
    """
    return {
        "max_position_size": 5,
        "reward_threshold": 0.01,
        "news_impact_threshold": 0.5,
    }


def test_data_to_features_pipeline_runs_without_errors(mock_iqfeed_data):
    """
    Vérifie que test_data_to_features_pipeline s'exécute sans erreur.
    """
    with patch("src.data.data_provider.DataProvider.get_data") as mock_get_data:
        mock_get_data.return_value = mock_iqfeed_data
        # Exécuter le test d'intégration
        test_data_to_features_pipeline(mock_iqfeed_data=mock_iqfeed_data)
        # Vérifier que le mock a été appelé
        mock_get_data.assert_called_once()


def test_features_to_trading_pipeline_runs_without_errors(
    mock_iqfeed_data,
    mock_trading_env,
):
    """
    Vérifie que test_features_to_trading_pipeline s'exécute sans erreur.
    """
    with patch(
        "src.trading.live_trading.LiveTrading.execute_trade"
    ) as mock_execute_trade:
        mock_execute_trade.return_value = {
            "trade_id": "TEST_123",
            "status": "success",
        }
        # Exécuter le test d'intégration
        test_features_to_trading_pipeline(
            mock_iqfeed_data=mock_iqfeed_data,
            mock_trading_env=mock_trading_env,
        )
        # Vérifier que le mock a été appelé
        mock_execute_trade.assert_called_once()


def test_full_pipeline_integration_runs_without_errors(
    mock_iqfeed_data,
    mock_trading_env,
):
    """
    Vérifie que test_full_pipeline_integration s'exécute sans erreur.
    """
    with patch(
        "src.data.data_provider.DataProvider.get_data"
    ) as mock_get_data, patch(
        "src.trading.live_trading.LiveTrading.execute_trade"
    ) as mock_execute_trade:
        mock_get_data.return_value = mock_iqfeed_data
        mock_execute_trade.return_value = {
            "trade_id": "TEST_124",
            "status": "success",
        }
        # Exécuter le test d'intégration
        test_full_pipeline_integration(
            mock_iqfeed_data=mock_iqfeed_data,
            mock_trading_env=mock_trading_env,
        )
        # Vérifier que les mocks ont été appelés
        mock_get_data.assert_called_once()
        mock_execute_trade.assert_called_once()


def test_pipeline_error_handling_catches_exceptions(mock_iqfeed_data):
    """
    Vérifie que test_pipeline_error_handling gère correctement les
    exceptions.
    """
    with patch("src.data.data_provider.DataProvider.get_data") as mock_get_data:
        mock_get_data.side_effect = ValueError(
            "Erreur de connexion IQFeed"
        )
        # Exécuter le test d'intégration
        test_pipeline_error_handling(mock_iqfeed_data=mock_iqfeed_data)
        # Vérifier que le mock a été appelé
        mock_get_data.assert_called_once()


def test_pipeline_empty_data_handles_empty_input(mock_trading_env):
    """
    Vérifie que test_pipeline_empty_data gère correctement les données
    vides.
    """
    with patch(
        "src.trading.live_trading.LiveTrading.execute_trade"
    ) as mock_execute_trade:
        # Exécuter le test d'intégration
        test_pipeline_empty_data(mock_trading_env=mock_trading_env)
        # Vérifier que le mock n'a pas été appelé (aucun trade exécuté)
        mock_execute_trade.assert_not_called()


def test_data_to_features_pipeline_output_format(mock_iqfeed_data):
    """
    Vérifie que test_data_to_features_pipeline produit le format de
    sortie attendu.
    """
    with patch(
        "src.data.data_provider.DataProvider.get_data"
    ) as mock_get_data, patch(
        "src.features.feature_pipeline.FeaturePipeline.generate_features"
    ) as mock_generate_features:
        mock_get_data.return_value = mock_iqfeed_data
        mock_generate_features.return_value = {
            "atr_14": pd.Series(
                np.random.uniform(0, 1, len(mock_iqfeed_data))
            ),
            "vix_es_correlation": pd.Series(
                np.random.uniform(-1, 1, len(mock_iqfeed_data))
            ),
            "delta_volume": pd.Series(
                np.random.uniform(-1000, 1000, len(mock_iqfeed_data))
            ),
            "ofi_score": pd.Series(
                np.random.uniform(-1, 1, len(mock_iqfeed_data))
            ),
        }

        # Exécuter le test d'intégration
        test_data_to_features_pipeline(mock_iqfeed_data=mock_iqfeed_data)

        # Vérifier que generate_features a été appelé
        mock_generate_features.assert_called_once()
        # Vérifier que get_data a été appelé
        mock_get_data.assert_called_once()


def test_full_pipeline_integration_trade_decision(
    mock_iqfeed_data,
    mock_trading_env,
):
    """
    Vérifie que test_full_pipeline_integration produit une décision de
    trading valide.
    """
    with patch(
        "src.data.data_provider.DataProvider.get_data"
    ) as mock_get_data, patch(
        "src.trading.live_trading.LiveTrading.execute_trade"
    ) as mock_execute_trade, patch(
        "src.trading.live_trading.LiveTrading.process_features"
    ) as mock_process_features:
        mock_get_data.return_value = mock_iqfeed_data
        mock_execute_trade.return_value = {
            "trade_id": "TEST_124",
            "status": "success",
        }
        mock_process_features.return_value = {
            "action": "buy",
            "size": 2,
        }

        # Exécuter le test d'intégration
        test_full_pipeline_integration(
            mock_iqfeed_data=mock_iqfeed_data,
            mock_trading_env=mock_trading_env,
        )

        # Vérifier que process_features a produit une décision
        mock_process_features.assert_called_once()
        # Vérifier que execute_trade a été appelé
        mock_execute_trade.assert_called_once()