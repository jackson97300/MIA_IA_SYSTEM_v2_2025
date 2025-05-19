# tests/test_data_lake.py
# Tests unitaires pour data_lake.py dans MIA_IA_SYSTEM_v2_2025
#
# Version: 2.1.6
# Date: 2025-05-14
#
# Rôle : Valide le stockage des données brutes, traitées, et des résultats des modèles dans un data lake S3.
# Utilisé par : Pipeline CI/CD (.github/workflows/python.yml).
#
# Notes :
# - Conforme à structure.txt (version 2.1.6, 2025-05-14).
# - Supporte les suggestions 1 (position sizing dynamique), 2 (coûts de transaction), 3 (microstructure),
#   4 (HMM/changepoint), 9 (surface de volatilité), 10 (ensembles de politiques).
# - Teste le stockage des features (atr_dynamic, orderflow_imbalance, etc.) dans processed/features/.
# - Teste le stockage des résultats des modèles (SAC, PPO, DDPG) dans processed/models/.
# - Vérifie les logs de performance (psutil), la capture d'erreurs (error_tracker.py), et les alertes (alert_manager.py).
# - Couverture augmentée pour inclure toutes les nouvelles fonctionnalités tout en préservant le test proposé.

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.data.data_lake import DataLake

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOG_DIR = BASE_DIR / "data" / "logs"
PERFORMANCE_LOG = LOG_DIR / "data_lake_performance.csv"


class TestDataLake(unittest.TestCase):
    def setUp(self):
        """Configure l'environnement de test."""
        self.data_lake = DataLake(market="ES")
        self.feature_data = pd.DataFrame(
            {
                "atr_dynamic": [2.0, 1.5, 3.0],
                "orderflow_imbalance": [0.5, -0.2, 0.1],
                "slippage_estimate": [0.1, 0.05, 0.2],
                "bid_ask_imbalance": [-0.1, 0.3, 0.0],
                "trade_aggressiveness": [0.2, 0.5, 0.8],
                "regime_hmm": [0, 1, 2],
                "iv_skew": [0.01, 0.02, 0.03],
                "iv_term_structure": [0.1, -0.1, 0.0],
            }
        )
        self.model_results = {
            "SAC": {"reward": 0.85, "timestamp": "2025-05-14T10:00:00"},
            "PPO": {"reward": 0.90, "timestamp": "2025-05-14T10:00:00"},
            "DDPG": {"reward": 0.88, "timestamp": "2025-05-14T10:00:00"},
        }

    @patch("boto3.client")
    def test_store_features(self, mock_s3):
        """Teste le stockage des features dans S3 (basé sur le test proposé)."""
        mock_s3.return_value.put_object = MagicMock()
        self.data_lake.store_features(self.feature_data)
        self.assertTrue(mock_s3.return_value.put_object.called)
        call_args = mock_s3.return_value.put_object.call_args
        self.assertEqual(call_args[1]["Bucket"], "mia-ia-system-data-lake")
        self.assertTrue(
            call_args[1]["Key"].startswith("processed/features/ES/features_")
        )
        self.assertTrue(PERFORMANCE_LOG.exists())

    @patch("boto3.client")
    def test_store_model_results(self, mock_s3):
        """Teste le stockage des résultats des modèles dans S3."""
        mock_s3.return_value.put_object = MagicMock()
        self.data_lake.store_model_results(self.model_results)
        self.assertTrue(mock_s3.return_value.put_object.called)
        call_args = mock_s3.return_value.put_object.call_args
        self.assertEqual(call_args[1]["Bucket"], "mia-ia-system-data-lake")
        self.assertTrue(
            call_args[1]["Key"].startswith("processed/models/ES/model_results_")
        )
        self.assertEqual(json.loads(call_args[1]["Body"]), self.model_results)
        self.assertTrue(PERFORMANCE_LOG.exists())

    @patch("boto3.client")
    def test_store_features_invalid_data(self, mock_s3):
        """Teste le stockage avec des données invalides."""
        mock_s3.return_value.put_object.side_effect = Exception("S3 error")
        with patch(
            "src.utils.error_tracker.capture_error"
        ) as mock_capture_error, patch(
            "src.model.utils.alert_manager.AlertManager.send_alert"
        ) as mock_alert:
            self.data_lake.store_features(self.feature_data)
            self.assertTrue(mock_capture_error.called)
            self.assertTrue(mock_alert.called)
            self.assertTrue(PERFORMANCE_LOG.exists())

    @patch("boto3.client")
    def test_store_model_results_invalid(self, mock_s3):
        """Teste le stockage avec des résultats de modèles invalides."""
        mock_s3.return_value.put_object.side_effect = Exception("S3 error")
        with patch(
            "src.utils.error_tracker.capture_error"
        ) as mock_capture_error, patch(
            "src.model.utils.alert_manager.AlertManager.send_alert"
        ) as mock_alert:
            self.data_lake.store_model_results(self.model_results)
            self.assertTrue(mock_capture_error.called)
            self.assertTrue(mock_alert.called)
            self.assertTrue(PERFORMANCE_LOG.exists())

    @patch("boto3.client")
    def test_performance_logging(self, mock_s3):
        """Teste la journalisation des performances avec psutil."""
        mock_s3.return_value.put_object = MagicMock()
        self.data_lake.store_features(self.feature_data)
        self.assertTrue(PERFORMANCE_LOG.exists())
        perf_df = pd.read_csv(PERFORMANCE_LOG)
        self.assertIn("timestamp", perf_df.columns)
        self.assertIn("market", perf_df.columns)
        self.assertIn("cpu_percent", perf_df.columns)
        self.assertIn("memory_usage_mb", perf_df.columns)
        self.assertIn("latency_s", perf_df.columns)
        self.assertGreater(len(perf_df), 0)

    @patch("boto3.client")
    def test_alert_manager_integration(self, mock_s3):
        """Teste l'intégration avec alert_manager.py."""
        mock_s3.return_value.put_object = MagicMock()
        with patch(
            "src.model.utils.alert_manager.AlertManager.send_alert"
        ) as mock_alert:
            self.data_lake.store_features(self.feature_data)
            self.assertTrue(mock_alert.called)
            self.assertEqual(
                mock_alert.call_args[0][0],
                f"Features stockées pour {self.data_lake.market}",
            )


if __name__ == "__main__":
    unittest.main()
