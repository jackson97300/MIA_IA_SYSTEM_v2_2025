# tests/test_validate_data.py
# Tests unitaires pour validate_data.py dans MIA_IA_SYSTEM_v2_2025
#
# Version: 2.1.6
# Date: 2025-05-14
#
# Rôle : Valide les schémas et plages des features avant leur utilisation dans le pipeline de trading.
# Utilisé par : Pipeline CI/CD (.github/workflows/python.yml).
#
# Notes :
# - Conforme à structure.txt (version 2.1.6, 2025-05-14).
# - Supporte les suggestions 1 (position sizing dynamique), 2 (coûts de transaction), 3 (microstructure),
#   4 (HMM/changepoint), 9 (surface de volatilité).
# - Teste les nouvelles règles de validation pour atr_dynamic, orderflow_imbalance, slippage_estimate,
#   bid_ask_imbalance, trade_aggressiveness, regime_hmm, iv_skew, iv_term_structure.
# - Vérifie les logs de performance (psutil), la capture d'erreurs (error_tracker.py), et les alertes (alert_manager.py).
# - Adapté pour utiliser validate_features au lieu de DataValidator, avec des données incluant toutes les features demandées.

import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.data.validate_data import validate_features, validation_errors

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOG_DIR = BASE_DIR / "data" / "logs"
FEATURE_LOG = LOG_DIR / "validate_data_performance.csv"
FEATURE_AUDIT = BASE_DIR / "data" / "features" / "features_audit_final.csv"
CONFIG_PATH = BASE_DIR / "config" / "feature_sets.yaml"


class TestValidateData(unittest.TestCase):
    def setUp(self):
        """Configure l'environnement de test."""
        self.market = "ES"
        self.valid_data = pd.DataFrame(
            {
                "atr_dynamic": [2.0, 1.5, 3.0],
                "orderflow_imbalance": [0.5, -0.2, 0.1],
                "slippage_estimate": [0.1, 0.05, 0.2],
                "bid_ask_imbalance": [-0.1, 0.3, 0.0],
                "trade_aggressiveness": [0.2, 0.5, 0.8],
                "regime_hmm": [0, 1, 2],
                "iv_skew": [0.01, 0.02, 0.03],
                "iv_term_structure": [0.1, -0.1, 0.0],
                "rsi_14": [50.0, 60.0, 70.0],  # Feature existante
                "obi_score": [0.5, -0.3, 0.8],  # Feature existante
            }
        )
        self.invalid_data = pd.DataFrame(
            {
                "atr_dynamic": [-1.0, 200.0, 50.0],  # Négatif et hors plage
                "orderflow_imbalance": [2.0, -2.0, 0.5],  # Hors plage [-1, 1]
                "slippage_estimate": [-0.1, 0.05, 0.2],  # Négatif
                "bid_ask_imbalance": [2.0, -2.0, 0.0],  # Hors plage [-1, 1]
                "trade_aggressiveness": [-0.1, 1.5, 0.8],  # Hors plage [0, 1]
                "regime_hmm": [3, -1, 2],  # Hors plage [0, 2]
                "iv_skew": [-0.01, 0.02, 0.03],  # Négatif
                "iv_term_structure": [2.0, -2.0, 0.0],  # Hors plage [-1, 1]
                "rsi_14": [150.0, 200.0, -10.0],  # Invalide
                "obi_score": [2.0, -2.0, 0.5],  # Invalide
            }
        )

    def test_validate_features(self):
        """Teste la validation de données valides (basé sur le test proposé)."""
        result = validate_features(
            self.valid_data, market=self.market, return_dict=True
        )
        self.assertTrue(result["success"])
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["errors"]), 0)
        self.assertTrue(FEATURE_AUDIT.exists())
        self.assertTrue(FEATURE_LOG.exists())

    def test_validate_features_invalid(self):
        """Teste la validation de données invalides."""
        with patch(
            "src.model.utils.alert_manager.AlertManager.send_alert"
        ) as mock_alert:
            result = validate_features(
                self.invalid_data, market=self.market, return_dict=True
            )
            self.assertFalse(result["success"])
            self.assertFalse(result["valid"])
            self.assertGreater(len(result["errors"]), 0)
            self.assertGreater(
                validation_errors.labels(market=self.market)._value.get(), 0
            )
            self.assertTrue(FEATURE_LOG.exists())
            self.assertTrue((LOG_DIR / "validation_errors.csv").exists())
            mock_alert.assert_called()

    def test_validate_features_new_rules(self):
        """Teste les nouvelles règles de validation."""
        data = pd.DataFrame(
            {
                "atr_dynamic": [2.0, -1.0, 3.0],  # Contient une valeur invalide
                "orderflow_imbalance": [0.5, -0.2, 1.5],  # Hors plage
                "slippage_estimate": [0.1, -0.1, 0.2],  # Négatif
                "bid_ask_imbalance": [-0.1, 2.0, 0.0],  # Hors plage
                "trade_aggressiveness": [0.2, -0.1, 0.8],  # Négatif
                "regime_hmm": [0, 3, 2],  # Hors plage
                "iv_skew": [0.01, -0.01, 0.03],  # Négatif
                "iv_term_structure": [0.1, 2.0, 0.0],  # Hors plage
            }
        )
        with patch(
            "src.model.utils.alert_manager.AlertManager.send_alert"
        ) as mock_alert:
            result = validate_features(data, market=self.market, return_dict=True)
            self.assertFalse(result["success"])
            self.assertFalse(result["valid"])
            self.assertGreater(len(result["errors"]), 0)
            self.assertTrue(any("atr_dynamic" in error for error in result["errors"]))
            self.assertTrue(
                any("orderflow_imbalance" in error for error in result["errors"])
            )
            self.assertTrue(
                any("slippage_estimate" in error for error in result["errors"])
            )
            self.assertTrue(
                any("bid_ask_imbalance" in error for error in result["errors"])
            )
            self.assertTrue(
                any("trade_aggressiveness" in error for error in result["errors"])
            )
            self.assertTrue(any("regime_hmm" in error for error in result["errors"]))
            self.assertTrue(any("iv_skew" in error for error in result["errors"]))
            self.assertTrue(
                any("iv_term_structure" in error for error in result["errors"])
            )
            mock_alert.assert_called()

    def test_validate_features_performance_logging(self):
        """Teste la journalisation des performances avec psutil."""
        validate_features(self.valid_data, market=self.market)
        self.assertTrue(FEATURE_LOG.exists())
        perf_df = pd.read_csv(FEATURE_LOG)
        self.assertIn("timestamp", perf_df.columns)
        self.assertIn("market", perf_df.columns)
        self.assertIn("cpu_percent", perf_df.columns)
        self.assertIn("memory_usage_mb", perf_df.columns)
        self.assertIn("latency_s", perf_df.columns)
        self.assertGreater(len(perf_df), 0)

    def test_validate_features_error_capture(self):
        """Teste la capture des erreurs via error_tracker."""
        with patch("src.utils.error_tracker.capture_error") as mock_capture_error:
            data = pd.DataFrame(
                {"invalid_column": [1, 2, 3]}
            )  # Données totalement invalides
            result = validate_features(data, market=self.market, return_dict=True)
            self.assertFalse(result["success"])
            self.assertTrue(mock_capture_error.called)
            self.assertTrue((LOG_DIR / "validation_errors.csv").exists())

    def test_validate_features_hmm_components(self):
        """Teste la validation de regime_hmm avec n_components."""
        with patch("src.data.validate_data.load_hmm_components", return_value=4):
            data = pd.DataFrame(
                {"regime_hmm": [0, 1, 2, 3]}
            )  # Valide pour n_components=4
            result = validate_features(data, market=self.market, return_dict=True)
            self.assertTrue(result["success"])
            data_invalid = pd.DataFrame(
                {"regime_hmm": [0, 1, 4]}
            )  # Invalide pour n_components=4
            result = validate_features(
                data_invalid, market=self.market, return_dict=True
            )
            self.assertFalse(result["success"])
            self.assertTrue(any("regime_hmm" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
