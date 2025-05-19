# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_regime_detector.py
# Tests unitaires pour regime_detector.py dans MIA_IA_SYSTEM_v2_2025.
#
# Version: 2.1.4
# Date: 2025-05-14
#
# Rôle : Valide la détection des régimes de marché avec HMM.
# Utilisé par : Pipeline CI/CD (.github/workflows/python.yml).
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-14).
# - Supporte la suggestion 1 (features dynamiques) avec tests unitaires (couverture 100%).
# - Teste l'initialisation, l'entraînement, la détection, le cache, les scénarios limites, et la performance.

import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.features.regime_detector import RegimeDetector


class TestRegimeDetector(unittest.TestCase):
    def setUp(self):
        """Configure l'environnement de test."""
        self.config = {
            "buffer_size": 100,
            "critical_buffer_size": 10,
            "max_retries": 3,
            "retry_delay": 2.0,
            "n_iterations": 10,
            "convergence_threshold": 1e-3,
            "covariance_type": "diag",
            "n_components": 3,
            "window_size": 50,
            "window_size_adaptive": True,
            "random_state": 42,
            "use_random_state": True,
            "min_train_rows": 100,
            "min_state_duration": 5,
            "cache_ttl_seconds": 300,
            "cache_ttl_adaptive": True,
            "prometheus_labels": {"env": "prod", "team": "quant", "market": "ES"},
        }
        self.detector = RegimeDetector(market="ES")
        self.orderflow_data = pd.DataFrame(
            {
                "bid_ask_imbalance": np.random.uniform(-1, 1, 1000),
                "total_volume": np.random.uniform(100, 1000, 1000),
            }
        )
        self.vix_data = pd.DataFrame({"vix": np.random.uniform(10, 30, 1000)})

    @patch("src.model.utils.config_manager.get_config")
    def test_init_valid_config(self, mock_config):
        """Teste l'initialisation avec une config valide."""
        mock_config.return_value = self.config
        detector = RegimeDetector(market="ES")
        self.assertEqual(detector.market, "ES")
        self.assertEqual(detector.config["n_components"], 3)
        self.assertEqual(detector.log_buffer, [])
        self.assertEqual(detector.error_buffer, [])
        self.assertEqual(detector.data_buffer.maxlen, 50)
        self.assertIsNone(detector.hmm_model)
        mock_config.assert_called_with(detector.config_path)

    @patch("src.model.utils.config_manager.get_config")
    def test_init_invalid_config(self, mock_config):
        """Teste l'initialisation avec une config invalide."""
        mock_config.return_value = {"buffer_size": 100}
        with self.assertRaises(ValidationError):
            RegimeDetector(market="ES")

    def test_train_hmm_valid(self):
        """Teste l'entraînement du modèle HMM avec des données valides."""
        self.detector.train_hmm(self.orderflow_data)
        self.assertIsNotNone(self.detector.hmm_model)
        self.assertEqual(self.detector.hmm_model.n_components, 3)
        model_path = self.detector.model_dir / f"hmm_{self.detector.market}.pkl"
        self.assertTrue(model_path.exists())
        trans_path = self.detector.log_dir / "hmm_transitions.csv"
        self.assertTrue(trans_path.exists())

    def test_train_hmm_invalid_data(self):
        """Teste l'entraînement avec des données invalides."""
        invalid_data = pd.DataFrame({"other_col": [1, 2, 3]})
        with self.assertRaises(ValueError) as cm:
            self.detector.train_hmm(invalid_data)
        self.assertIn("Colonnes manquantes dans orderflow_data", str(cm.exception))

    def test_train_hmm_insufficient_data(self):
        """Teste l'entraînement avec des données insuffisantes."""
        small_data = self.orderflow_data.iloc[:50]
        with self.assertRaises(ValueError) as cm:
            self.detector.train_hmm(small_data)
        self.assertIn("Données insuffisantes pour entraînement HMM", str(cm.exception))

    def test_train_hmm_extreme_values(self):
        """Teste l'entraînement avec des valeurs extrêmes."""
        extreme_data = self.orderflow_data.copy()
        extreme_data["total_volume"] *= 1e6
        self.detector.train_hmm(extreme_data)
        self.assertIsNotNone(self.detector.hmm_model)

    def test_detect_regime_valid(self):
        """Teste la détection de régime avec des données valides."""
        self.detector.train_hmm(self.orderflow_data)
        regime = self.detector.detect_regime(
            self.orderflow_data.iloc[-10:], volatility_score=0.3
        )
        self.assertIn(regime, [0, 1, 2])

    def test_detect_regime_untrained(self):
        """Teste la détection sans modèle entraîné."""
        regime = self.detector.detect_regime(self.orderflow_data.iloc[-10:])
        self.assertEqual(regime, 2)  # Range par défaut

    def test_detect_regime_invalid_data(self):
        """Teste la détection avec des données invalides."""
        self.detector.train_hmm(self.orderflow_data)
        invalid_data = pd.DataFrame({"other_col": [1, 2, 3]})
        regime = self.detector.detect_regime(invalid_data)
        self.assertEqual(regime, 2)  # Range par défaut

    def test_detect_regime_cache(self):
        """Teste l'utilisation du cache."""
        self.detector.train_hmm(self.orderflow_data)
        data_subset = self.orderflow_data.iloc[-10:]
        regime1 = self.detector.detect_regime(data_subset, volatility_score=0.3)
        regime2 = self.detector.detect_regime(data_subset, volatility_score=0.3)
        self.assertEqual(regime1, regime2)
        cache_key = hash(
            tuple(data_subset[["bid_ask_imbalance", "total_volume"]].iloc[-1].values)
        )
        self.assertIn(cache_key, self.detector.feature_cache)

    def test_detect_regime_min_state_duration(self):
        """Teste la contrainte min_state_duration."""
        self.detector.train_hmm(self.orderflow_data)
        data_subset = self.orderflow_data.iloc[-10:]
        regime1 = self.detector.detect_regime(data_subset, volatility_score=0.3)
        # Simuler plusieurs appels pour vérifier la durée minimale
        for _ in range(self.config["min_state_duration"] - 1):
            regime2 = self.detector.detect_regime(data_subset, volatility_score=0.3)
            self.assertEqual(regime1, regime2, "Régime changé avant min_state_duration")
        # Après min_state_duration, un changement est possible
        self.detector.state_duration = self.config["min_state_duration"]
        regime3 = self.detector.detect_regime(data_subset, volatility_score=0.3)
        self.assertTrue(regime3 in [0, 1, 2])

    def test_detect_regime_adaptive(self):
        """Teste l'ajustement adaptatif du window_size et cache_ttl."""
        self.detector.train_hmm(self.orderflow_data)
        data_subset = self.orderflow_data.iloc[-10:]
        # Haute volatilité (réduit window_size et ttl)
        self.detector.detect_regime(data_subset, volatility_score=0.6)
        self.assertEqual(self.detector.data_buffer.maxlen, 25)  # 50 // 2
        self.assertEqual(self.detector.feature_cache._ttl, 60)  # 300 // 5
        # Basse volatilité (valeurs par défaut)
        self.detector.detect_regime(data_subset, volatility_score=0.2)
        self.assertEqual(self.detector.data_buffer.maxlen, 50)
        self.assertEqual(self.detector.feature_cache._ttl, 300)

    def test_detect_regime_state_distribution(self):
        """Teste la répartition des états."""
        self.detector.train_hmm(self.orderflow_data)
        data_subset = self.orderflow_data.iloc[-10:]
        for _ in range(100):
            self.detector.detect_regime(data_subset, volatility_score=0.3)
        total_counts = sum(self.detector.state_counts.values())
        self.assertGreater(total_counts, 0)
        for state, count in self.detector.state_counts.items():
            prob = count / total_counts
            self.assertTrue(0 <= prob <= 1)

    def test_generate_heatmap_valid(self):
        """Teste la génération du heatmap."""
        self.detector.train_hmm(self.orderflow_data)
        self.detector.generate_heatmap(self.vix_data)
        heatmap_path = (
            self.detector.figures_dir
            / f"regime_vs_vix_{datetime.now().strftime('%Y%m')}.png"
        )
        self.assertTrue(heatmap_path.exists())

    def test_generate_heatmap_invalid_data(self):
        """Teste la génération avec des données invalides."""
        invalid_vix = pd.DataFrame({"other_col": [1, 2, 3]})
        with self.assertRaises(ValueError) as cm:
            self.detector.generate_heatmap(invalid_vix)
        self.assertIn("Colonne vix manquante dans vix_data", str(cm.exception))

    def test_error_logging(self):
        """Teste la journalisation des erreurs critiques."""
        invalid_data = pd.DataFrame({"other_col": [1, 2, 3]})
        self.detector.detect_regime(invalid_data)
        self.assertEqual(len(self.detector.error_buffer), 1)
        self.detector.error_buffer = [{}] * self.config["critical_buffer_size"]
        self.detector.detect_regime(invalid_data)
        error_path = self.detector.log_dir / "regime_detector_errors.csv"
        self.assertTrue(error_path.exists())

    def test_performance(self, benchmark):
        """Teste la performance sur 1000 appels successifs."""
        self.detector.train_hmm(self.orderflow_data)
        data_subset = self.orderflow_data.iloc[-10:]
        benchmark(
            lambda: self.detector.detect_regime(data_subset, volatility_score=0.3)
        )
        start_time = time.time()
        for _ in range(1000):
            self.detector.detect_regime(data_subset, volatility_score=0.3)
        duration = time.time() - start_time
        self.assertLess(duration, 1.0, "Performance insuffisante pour 1000 appels")


if __name__ == "__main__":
    unittest.main()
