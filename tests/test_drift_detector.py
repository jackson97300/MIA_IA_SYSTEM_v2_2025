# tests/test_drift_detector.py
# Tests unitaires pour drift_detector.py dans MIA_IA_SYSTEM_v2_2025
#
# Version: 2.2.2
# Date: 2025-05-14
#
# Rôle : Valide la détection de drift des features et du Sharpe ratio dans le pipeline de trading.
# Utilisé par : Pipeline CI/CD (.github/workflows/python.yml).
#
# Notes :
# - Conforme à structure.txt (version 2.2.2, 2025-05-14).
# - Supporte les suggestions 1 (features dynamiques), 4 (changepoint), 6 (drift Sharpe), 8 (fallback SHAP).
# - Teste la détection de drift des features (test KS), du Sharpe (ADWIN), les transitions HMM,
#   les métriques Prometheus, les logs psutil, et la gestion des erreurs/alertes.
# - Basé sur le test proposé, enrichi pour une couverture complète des fonctionnalités.

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.monitoring.drift_detector import DriftDetector

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOG_DIR = BASE_DIR / "data" / "logs"
PERF_LOG = LOG_DIR / "drift_detector_performance.csv"


class TestDriftDetector(unittest.TestCase):
    def setUp(self):
        """Configure l'environnement de test."""
        self.detector = DriftDetector(market="ES")
        self.current_data = pd.DataFrame(
            {
                "profit": [100, 110, 90, 120, 80],
                "regime_hmm": [0, 0, 1, 1, 2],
                "feature1": [1.0, 1.1, 1.2, 1.3, 1.4],
                "feature2": [2.0, 2.1, 2.2, 2.3, 2.4],
            }
        )
        self.reference_data = pd.DataFrame(
            {
                "feature1": [0.5, 0.6, 0.7, 0.8, 0.9],
                "feature2": [1.5, 1.6, 1.7, 1.8, 1.9],
            }
        )

    def test_detect_sharpe_drift(self):
        """Teste la détection de drift du Sharpe ratio avec ADWIN (basé sur le test proposé)."""
        with patch("river.drift.ADWIN.update") as mock_update, patch(
            "river.drift.ADWIN.drift_detected", new_callable=MagicMock
        ) as mock_drift:
            mock_drift.return_value = True
            result = self.detector.detect_sharpe_drift(self.current_data)
            self.assertTrue(isinstance(result, bool))
            self.assertTrue(result)
            self.assertTrue(mock_update.called)
            self.assertTrue(PERF_LOG.exists())
            self.detector.alert_manager.send_alert.assert_called()

    def test_detect_feature_drift_valid(self):
        """Teste la détection de drift des features avec le test KS."""
        with patch("scipy.stats.ks_2samp") as mock_ks:
            mock_ks.return_value = (0.5, 0.01)  # Simule un drift (p-value < 0.05)
            result = self.detector.detect_feature_drift(
                self.current_data[["feature1", "feature2"]], self.reference_data
            )
            self.assertTrue(result)
            self.assertTrue(PERF_LOG.exists())
            self.detector.alert_manager.send_alert.assert_called()

    def test_detect_feature_drift_invalid_columns(self):
        """Teste la détection de drift avec des colonnes incompatibles."""
        invalid_data = pd.DataFrame({"feature3": [1, 2, 3]})
        with self.assertRaises(ValueError) as cm:
            self.detector.detect_feature_drift(invalid_data, self.reference_data)
        self.assertIn("Colonnes incompatibles", str(cm.exception))
        self.assertTrue(PERF_LOG.exists())
        self.detector.alert_manager.send_alert.assert_called()

    def test_detect_sharpe_drift_missing_columns(self):
        """Teste la détection de drift avec des colonnes manquantes."""
        invalid_data = pd.DataFrame({"other_col": [1, 2, 3]})
        with self.assertRaises(ValueError) as cm:
            self.detector.detect_sharpe_drift(invalid_data)
        self.assertIn("Colonnes manquantes", str(cm.exception))
        self.assertTrue(PERF_LOG.exists())
        self.detector.alert_manager.send_alert.assert_called()

    def test_detect_sharpe_drift_hmm_anomaly(self):
        """Teste la détection d'anomalies dans les transitions HMM."""
        data = pd.DataFrame(
            {
                "profit": [100] * 20,
                "regime_hmm": [
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                ],  # 10 changements
            }
        )
        result = self.detector.detect_sharpe_drift(data)
        self.assertFalse(result)  # Pas de drift Sharpe, mais anomalie HMM
        self.detector.alert_manager.send_alert.assert_called_with(
            f"Anomalie dans les transitions HMM pour {self.detector.market}", priority=3
        )
        self.assertTrue(PERF_LOG.exists())

    def test_performance_logging(self):
        """Teste la journalisation des performances avec psutil."""
        self.detector.detect_sharpe_drift(self.current_data)
        self.assertTrue(PERF_LOG.exists())
        perf_df = pd.read_csv(PERF_LOG)
        self.assertIn("timestamp", perf_df.columns)
        self.assertIn("market", perf_df.columns)
        self.assertIn("category", perf_df.columns)
        self.assertIn("cpu_percent", perf_df.columns)
        self.assertIn("memory_usage_mb", perf_df.columns)
        self.assertIn("latency_s", perf_df.columns)
        self.assertGreater(len(perf_df), 0)

    def test_prometheus_metrics(self):
        """Teste la mise à jour des métriques Prometheus."""
        with patch("river.drift.ADWIN.update") as mock_update, patch(
            "river.drift.ADWIN.drift_detected", new_callable=MagicMock
        ) as mock_drift, patch(
            "src.monitoring.prometheus_metrics.Gauge.set"
        ) as mock_set:
            mock_drift.return_value = True
            self.detector.detect_sharpe_drift(self.current_data)
            self.assertTrue(mock_set.called)
            self.assertTrue(PERF_LOG.exists())

    def test_log_drift(self):
        """Teste l'enregistrement des drifts dans la base SQLite."""
        with patch("sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            self.detector._log_drift("feature1", 0.01)
            self.assertTrue(mock_conn.execute.called)
            self.assertTrue(PERF_LOG.exists())


if __name__ == "__main__":
    unittest.main()
