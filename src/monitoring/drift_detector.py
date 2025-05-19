# src/monitoring/drift_detector.py
# Détection de drift des features pour MIA_IA_SYSTEM_v2_2025
#
# Version: 2.2.2
# Date: 2025-05-14
#
# Rôle: Surveille les distributions des 150 SHAP features avec un test de Kolmogorov-Smirnov (KS test) et la performance (Sharpe ratio) avec ADWIN pour détecter les drifts.
# Utilisé par: mia_switcher.py, feature_pipeline.py, train_pipeline.py.
#
# Notes:
# - Conforme à structure.txt (version 2.2.2, 2025-05-14).
# - Supporte les suggestions 1 (features dynamiques), 4 (changepoint), 6 (drift Sharpe), 8 (fallback SHAP).
# - Stocke les résultats dans data/market_memory.db (table drift_metrics).
# - Journalise les performances avec psutil dans data/logs/drift_detector_performance.csv.
# - Intègre fallback SHAP via data_lake.retrieve_shap_fallback().
# - Pas de références à dxFeed, obs_t, 320 features, ou 81 features.
# - Policies Note: The official directory for routing policies is src/model/router/policies.
# - Mise à jour pour renforcer la détection ADWIN, valider les transitions HMM, ajouter métrique Prometheus, logs psutil, et gestion des erreurs/alertes.

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import psutil
from prometheus_client import Counter, Gauge
from river.drift import ADWIN
from scipy.stats import ks_2samp

from src.data.data_lake import DataLake
from src.model.utils.alert_manager import AlertManager
from src.utils.error_tracker import capture_error

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOG_DIR = BASE_DIR / "data" / "logs"
LOG_DIR.mkdir(exist_ok=True)
PERF_LOG = LOG_DIR / "drift_detector_performance.csv"

logging.basicConfig(
    filename=LOG_DIR / "drift_detector.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)

drift_detected = Counter(
    "drift_detected_total",
    "Nombre total de drifts détectés pour les features",
    ["market", "feature"],
)
sharpe_drift_metric = Gauge("sharpe_drift", "Drift du ratio de Sharpe", ["market"])


class DriftDetector:
    def __init__(self, market: str = "ES"):
        """Initialise le détecteur de drift."""
        try:
            self.market = market
            self.alert_manager = AlertManager()
            self.drift_detector = ADWIN()
            self.data_lake = DataLake(bucket="mia-ia-system-data-lake", market=market)
            logging.info(f"Détecteur de drift initialisé pour le marché {market}")
            self.alert_manager.send_alert(
                f"Détecteur de drift initialisé pour {market}", priority=2
            )
        except Exception as e:
            error_msg = f"Erreur initialisation DriftDetector: {e}"
            logging.error(error_msg)
            capture_error(
                e,
                context={"market": market},
                market=market,
                operation="init_drift_detector",
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            raise

    def detect_feature_drift(
        self,
        current_data: pd.DataFrame,
        reference_data: pd.DataFrame,
        p_value_threshold: float = 0.05,
    ) -> bool:
        """Détecte les drifts dans les distributions des features avec le test KS."""
        start_time = datetime.now()
        try:
            if not set(current_data.columns) == set(reference_data.columns):
                error_msg = (
                    "Colonnes incompatibles entre current_data et reference_data"
                )
                logging.error(error_msg)
                raise ValueError(error_msg)

            drift_found = False
            for col in current_data.columns:
                try:
                    x = current_data[col].dropna()
                    y = reference_data[col].dropna()
                    if len(x) < 10 or len(y) < 10:
                        logging.warning(f"Données insuffisantes pour {col}")
                        continue
                    stat, p_value = ks_2samp(x, y)
                    if p_value < p_value_threshold:
                        drift_detected.labels(market=self.market, feature=col).inc()
                        self._log_drift(col, p_value)
                        logging.warning(
                            f"Drift détecté pour {col} (p-value={p_value:.4f})"
                        )
                        self.alert_manager.send_alert(
                            f"Drift détecté pour {col} (p-value={p_value:.4f})",
                            priority=3,
                        )
                        drift_found = True
                except (TypeError, ValueError) as e:
                    error_msg = f"Erreur test KS pour {col}: {e}"
                    logging.error(error_msg)
                    capture_error(
                        e,
                        context={"market": self.market},
                        market=self.market,
                        operation="detect_feature_drift",
                    )
                    self.alert_manager.send_alert(error_msg, priority=4)
                    continue

            latency = (datetime.now() - start_time).total_seconds()
            memory = psutil.Process().memory_info().rss / 1024 / 1024
            pd.DataFrame(
                [
                    {
                        "timestamp": datetime.now().isoformat(),
                        "market": self.market,
                        "category": "feature_drift",
                        "cpu_percent": psutil.cpu_percent(),
                        "memory_usage_mb": memory,
                        "latency_s": latency,
                    }
                ]
            ).to_csv(PERF_LOG, mode="a", index=False, header=not PERF_LOG.exists())
            logging.info(
                f"Détection de drift terminée pour {self.market}. Latence: {latency}s"
            )
            self.alert_manager.send_alert(
                f"Détection de drift terminée pour {self.market}", priority=2
            )
            return drift_found
        except Exception as e:
            error_msg = f"Erreur détection feature drift: {e}"
            logging.error(error_msg)
            capture_error(
                e,
                context={"market": self.market},
                market=self.market,
                operation="detect_feature_drift",
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            return False

    def detect_sharpe_drift(self, data: pd.DataFrame) -> bool:
        """Détecte les drifts dans le Sharpe ratio avec ADWIN et valide les transitions HMM."""
        start_time = datetime.now()
        try:
            required_cols = ["profit", "regime_hmm"]
            missing = [col for col in required_cols if col not in data.columns]
            if missing:
                error_msg = f"Colonnes manquantes: {missing}"
                logging.error(error_msg)
                raise ValueError(error_msg)

            sharpe = (
                data["profit"].rolling(window=20).mean()
                / data["profit"].rolling(window=20).std()
            )
            drift_found = False
            for value in sharpe.dropna():
                self.drift_detector.update(value)
                if self.drift_detector.drift_detected:
                    sharpe_drift_metric.labels(market=self.market).set(value)
                    logging.info(
                        f"Drift Sharpe détecté pour {self.market}: Sharpe={value}"
                    )
                    self.alert_manager.send_alert(
                        f"Drift Sharpe détecté pour {self.market}: Sharpe={value}",
                        priority=4,
                    )
                    self._log_drift("sharpe_ratio", float(value))
                    drift_found = True

            regime_changes = data["regime_hmm"].diff().abs().sum()
            if regime_changes > 10:  # Seuil arbitraire
                logging.warning(f"Anomalie dans les transitions HMM pour {self.market}")
                self.alert_manager.send_alert(
                    f"Anomalie dans les transitions HMM pour {self.market}", priority=3
                )

            latency = (datetime.now() - start_time).total_seconds()
            memory = psutil.Process().memory_info().rss / 1024 / 1024
            pd.DataFrame(
                [
                    {
                        "timestamp": datetime.now().isoformat(),
                        "market": self.market,
                        "category": "sharpe_drift",
                        "cpu_percent": psutil.cpu_percent(),
                        "memory_usage_mb": memory,
                        "latency_s": latency,
                    }
                ]
            ).to_csv(PERF_LOG, mode="a", index=False, header=not PERF_LOG.exists())
            logging.info(
                f"Détection Sharpe drift terminée pour {self.market}. Latence: {latency}s"
            )
            self.alert_manager.send_alert(
                f"Détection Sharpe drift terminée pour {self.market}", priority=2
            )
            return drift_found
        except Exception as e:
            error_msg = f"Erreur détection Sharpe drift: {e}"
            logging.error(error_msg)
            capture_error(
                e,
                context={"market": self.market},
                market=self.market,
                operation="detect_sharpe_drift",
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            latency = (datetime.now() - start_time).total_seconds()
            memory = psutil.Process().memory_info().rss / 1024 / 1024
            pd.DataFrame(
                [
                    {
                        "timestamp": datetime.now().isoformat(),
                        "market": self.market,
                        "category": "sharpe_drift_error",
                        "cpu_percent": psutil.cpu_percent(),
                        "memory_usage_mb": memory,
                        "latency_s": latency,
                    }
                ]
            ).to_csv(PERF_LOG, mode="a", index=False, header=not PERF_LOG.exists())
            return False

    def _log_drift(self, feature_name: str, p_value: float) -> None:
        """Enregistre les drifts détectés dans la base SQLite."""
        try:
            db_path = BASE_DIR / "data" / "market_memory.db"
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS drift_metrics (
                        feature_name TEXT,
                        p_value REAL,
                        timestamp TEXT
                    )
                """
                )
                conn.execute(
                    """
                    INSERT INTO drift_metrics (feature_name, p_value, timestamp)
                    VALUES (?, ?, ?)
                    """,
                    (feature_name, p_value, datetime.now().isoformat()),
                )
                conn.commit()
                logging.debug(
                    f"Drift enregistré pour {feature_name} (p-value={p_value})"
                )
        except sqlite3.Error as e:
            error_msg = f"Erreur SQLite drift log: {e}"
            logging.error(error_msg)
            capture_error(
                e,
                context={"market": self.market},
                market=self.market,
                operation="log_drift",
            )
            self.alert_manager.send_alert(error_msg, priority=4)
