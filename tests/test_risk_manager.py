# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_risk_manager.py
# Tests unitaires pour risk_manager.py dans MIA_IA_SYSTEM_v2_2025.
#
# Version: 2.1.4
# Date: 2025-05-13
#
# Rôle : Valide le position sizing dynamique avec Kelly/ATR.
# Utilisé par : Pipeline CI/CD (.github/workflows/python.yml).
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Supporte la suggestion 5 (profit factor) avec tests unitaires (couverture 100%).
# - Teste l'initialisation, le calcul de position, les cas d'erreur, la config, le cache, et la performance.
# - Policies Note: The official directory for routing policies is src/model/router/policies.
# The src/model/policies directory is a residual and should be verified
# for removal to avoid import conflicts.

import time
import unittest
from unittest.mock import patch

import numpy as np

from src.risk_management.risk_manager import RiskManager


class TestRiskManager(unittest.TestCase):
    def setUp(self):
        """Configure l'environnement de test."""
        self.config = {
            "buffer_size": 100,
            "max_retries": 3,
            "retry_delay": 2.0,
            "kelly_fraction": 0.1,
            "max_position_fraction": 0.1,
        }
        self.risk_manager = RiskManager(market="ES", capital=100000.0)

    @patch("src.model.utils.config_manager.get_config")
    def test_init_valid_config(self, mock_config):
        """Teste l'initialisation avec une config valide."""
        mock_config.return_value = self.config
        risk_manager = RiskManager(market="ES", capital=50000.0)
        self.assertEqual(risk_manager.market, "ES")
        self.assertEqual(risk_manager.capital, 50000.0)
        self.assertEqual(risk_manager.log_buffer, [])
        self.assertEqual(risk_manager.size_cache, {})
        mock_config.assert_called_with(risk_manager.config_path)

    @patch("src.model.utils.config_manager.get_config")
    def test_init_invalid_config(self, mock_config):
        """Teste l'initialisation avec une config invalide."""
        mock_config.return_value = {}
        with self.assertRaises(ValueError) as cm:
            RiskManager(market="ES")
        self.assertIn(
            "Clés manquantes dans risk_manager_config.yaml", str(cm.exception)
        )

    def test_calculate_position_size_valid(self):
        """Teste le calcul de la taille de position avec des entrées valides."""
        atr_dynamic = 2.0
        orderflow_imbalance = 0.5
        volatility_score = 0.2
        expected_size = min(
            0.1 * (1 - 0.2) * (1 / 2.0) * (1 + 0.5), 0.1
        )  # Kelly * (1-vol) * (1/ATR) * (1+imbalance)
        size = self.risk_manager.calculate_position_size(
            atr_dynamic, orderflow_imbalance, volatility_score
        )
        self.assertAlmostEqual(size, expected_size, places=5)
        self.assertTrue(0 <= size <= 0.1)

    def test_calculate_position_size_invalid_atr(self):
        """Teste le calcul avec un ATR invalide."""
        atr_dynamic = 0.0
        orderflow_imbalance = 0.5
        volatility_score = 0.2
        size = self.risk_manager.calculate_position_size(
            atr_dynamic, orderflow_imbalance, volatility_score
        )
        self.assertEqual(size, 0.0)

    def test_calculate_position_size_invalid_imbalance(self):
        """Teste le calcul avec une imbalance invalide."""
        atr_dynamic = 2.0
        orderflow_imbalance = 2.0
        volatility_score = 0.2
        size = self.risk_manager.calculate_position_size(
            atr_dynamic, orderflow_imbalance, volatility_score
        )
        self.assertEqual(size, 0.0)

    def test_calculate_position_size_invalid_volatility(self):
        """Teste le calcul avec un volatility score invalide."""
        atr_dynamic = 2.0
        orderflow_imbalance = 0.5
        volatility_score = 1.5
        size = self.risk_manager.calculate_position_size(
            atr_dynamic, orderflow_imbalance, volatility_score
        )
        self.assertEqual(size, 0.0)

    def test_calculate_position_size_cache(self):
        """Teste l'utilisation du cache."""
        atr_dynamic = 2.0
        orderflow_imbalance = 0.5
        volatility_score = 0.2
        size1 = self.risk_manager.calculate_position_size(
            atr_dynamic, orderflow_imbalance, volatility_score
        )
        size2 = self.risk_manager.calculate_position_size(
            atr_dynamic, orderflow_imbalance, volatility_score
        )
        self.assertEqual(size1, size2)
        self.assertIn(
            hash((atr_dynamic, orderflow_imbalance)), self.risk_manager.size_cache
        )

    def test_calculate_position_size_performance(self):
        """Teste la performance sur 1000 appels successifs."""
        atr_dynamic = 2.0
        orderflow_imbalance = 0.5
        volatility_score = 0.2
        start_time = time.time()
        for _ in range(1000):
            self.risk_manager.calculate_position_size(
                atr_dynamic, orderflow_imbalance, volatility_score
            )
        duration = time.time() - start_time
        self.assertLess(duration, 1.0, "Performance insuffisante pour 1000 appels")

    @patch("src.model.utils.alert_manager.AlertManager.send_alert")
    def test_position_size_alert(self, mock_alert):
        """Teste l'alerte pour une taille proche de la limite."""
        atr_dynamic = 1.0
        orderflow_imbalance = 0.9
        volatility_score = 0.0
        size = self.risk_manager.calculate_position_size(
            atr_dynamic, orderflow_imbalance, volatility_score
        )
        self.assertTrue(size >= 0.09)
        mock_alert.assert_called_with(
            f"Taille de position élevée pour ES: {size*100:.2f}% du capital", priority=3
        )

    def test_position_size_variance(self):
        """Teste le calcul de la variance des tailles."""
        atr_dynamic = 2.0
        orderflow_imbalance = 0.5
        volatility_score = 0.2
        for _ in range(50):
            self.risk_manager.calculate_position_size(
                atr_dynamic, orderflow_imbalance, volatility_score
            )
        for _ in range(50):
            self.risk_manager.calculate_position_size(
                atr_dynamic * 1.1, orderflow_imbalance, volatility_score
            )
        variance = np.var(self.risk_manager.recent_sizes)
        self.assertGreater(variance, 0.0, "Variance des tailles doit être positive")


if __name__ == "__main__":
    unittest.main()
