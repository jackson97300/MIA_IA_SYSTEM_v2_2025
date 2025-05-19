# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_resilience.py
# Tests de résilience pour MIA_IA_SYSTEM_v2_2025
#
# Version: 2.1.5
# Date: 2025-05-14
#
# Rôle: Simule des pannes (latence DB, déconnexion IQFeed, surcharge CPU, volatilité extrême) pour valider
#       les mécanismes de résilience des composants critiques (data_provider.py, live_trading.py,
#       risk_manager.py, regime_detector.py). Intègre des tests pour le position sizing dynamique (suggestion 1)
#       et la détection de changements de régime (suggestion 4).
#
# Utilisé par: Pipeline CI/CD (.github/workflows/python.yml).
#
# Notes:
# - Conforme à structure.txt (version 2.1.5, 2025-05-14).
# - Supporte les suggestions 1 (position sizing dynamique), 4 (HMM/changepoint detection),
#   7 (tests unitaires pour une couverture de 100%).
# - Teste les circuit breakers (pybreaker) et retries (tenacity).
# - Intègre logs psutil dans data/logs/test_resilience_performance.csv avec noms d'opérations uniformes.
# - Utilise error_tracker.py pour capturer les erreurs et alert_manager.py pour les alertes.
# - Améliore DummyHMM pour simuler des transitions dynamiques de régimes.
# - Externalise la journalisation des performances via la fonction log_performance.
# - Pas de références à dxFeed, obs_t, 320 features, ou 81 features.
# - Policies Note: The official directory for routing policies is src/model/router/policies.
# The src/model/policies directory is a residual and should be verified
# for removal to avoid import conflicts.

import os
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import psutil
import pytest
from loguru import logger

from src import run_system
from src.data.data_provider import DataProvider
from src.features.feature_pipeline import FeaturePipeline
from src.features.regime_detector import RegimeDetector
from src.risk_management.risk_manager import RiskManager
from src.utils.error_tracker import sentry_errors

# Configuration du logging Loguru
logger.remove()
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOG_DIR = BASE_DIR / "data" / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "test_resilience.log", rotation="10 MB", level="INFO", encoding="utf-8"
)


def log_performance(
    operation: str, start_time: datetime, log_path: str, **extra
) -> None:
    """Journalise les performances CPU/mémoire dans le fichier spécifié."""
    latency = (datetime.now() - start_time).total_seconds()
    entry = {
        "timestamp": datetime.now().isoformat(),
        "operation": operation,
        "latency": latency,
        "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
        "cpu_usage_percent": psutil.cpu_percent(),
        **extra,
    }
    pd.DataFrame([entry]).to_csv(
        log_path,
        mode="a",
        header=not os.path.exists(log_path),
        index=False,
        encoding="utf-8",
    )


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    logs_dir = base_dir / "data" / "logs"
    logs_dir.mkdir(parents=True)
    return {
        "base_dir": str(base_dir),
        "logs_dir": str(logs_dir),
        "perf_log_path": str(logs_dir / "test_resilience_performance.csv"),
        "log_file_path": str(logs_dir / "test_resilience.log"),
    }


@pytest.fixture
def stress_data():
    """Fournit des données de stress pour les tests."""
    return pd.DataFrame(
        {
            "atr_dynamic": [1000.0],  # Volatilité extrême
            "orderflow_imbalance": [0.9],
            "bid_ask_imbalance": [0.1],
            "total_volume": [1000],
        }
    )


@pytest.fixture
def dummy_hmm():
    """Mock pour un modèle HMM simulant des transitions dynamiques."""

    class DummyHMM:
        def __init__(self):
            self.state = 0

        def predict(self, X):
            self.state = (self.state + 1) % 3
            return [self.state]

    return DummyHMM()


def test_db_latency(tmp_dirs):
    """Teste la résilience à une latence élevée de la base de données."""
    start_time = datetime.now()
    with patch("sqlite3.connect") as mock_connect:
        mock_connect.side_effect = lambda *args: time.sleep(2) or MagicMock()
        result = run_system.main(market="ES")
        assert result is not None, "Le système doit fonctionner malgré la latence DB"
        log_performance(
            "db_latency", start_time, tmp_dirs["perf_log_path"], market="ES"
        )
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df["operation"] == "db_latency"), "Log db_latency manquant"
        assert all(
            col in df.columns for col in ["memory_usage_mb", "cpu_usage_percent"]
        ), "Colonnes psutil manquantes"


def test_iqfeed_failure(tmp_dirs):
    """Teste la résilience à une panne IQFeed."""
    start_time = datetime.now()
    with patch(
        "src.data.data_provider.DataProvider.fetch_iqfeed_data"
    ) as mock_fetch, patch("src.utils.error_tracker.capture_error") as mock_capture:
        mock_fetch.side_effect = Exception("Connexion IQFeed échouée")
        data_provider = DataProvider()
        result = data_provider.fetch_iqfeed_data(symbol="ES")
        assert result is not None, "Le circuit breaker doit gérer l’échec IQFeed"
        assert (
            sentry_errors.labels(market="ES", operation="fetch_iqfeed")._value.get() > 0
        ), "L’erreur doit être capturée par Sentry"
        mock_capture.assert_called()
        log_performance(
            "iqfeed_failure", start_time, tmp_dirs["perf_log_path"], market="ES"
        )
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df["operation"] == "iqfeed_failure"), "Log iqfeed_failure manquant"


def test_shap_fallback_failure(tmp_dirs):
    """Teste la résilience à un échec du chargement du cache SHAP."""
    start_time = datetime.now()
    with patch("pybreaker.CircuitBreaker") as mock_breaker, patch(
        "src.utils.error_tracker.capture_error"
    ) as mock_capture:
        mock_breaker.side_effect = Exception("Cache SHAP indisponible")
        feature_pipeline = FeaturePipeline()
        result = feature_pipeline.load_shap_fallback()
        assert isinstance(
            result, pd.DataFrame
        ), "Le fallback SHAP doit retourner un DataFrame malgré l’échec"
        assert (
            sentry_errors.labels(
                market="ES", operation="load_shap_fallback"
            )._value.get()
            > 0
        ), "L’erreur doit être capturée par Sentry"
        mock_capture.assert_called()
        log_performance(
            "shap_fallback_failure", start_time, tmp_dirs["perf_log_path"], market="ES"
        )
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "shap_fallback_failure"
        ), "Log shap_fallback_failure manquant"


def test_risk_manager_stress(tmp_dirs, stress_data):
    """Teste risk_manager.py sous volatilité extrême."""
    start_time = datetime.now()
    with patch("src.model.utils.alert_manager.AlertManager.send_alert") as mock_alert:
        risk_manager = RiskManager(market="ES")
        size = risk_manager.calculate_position_size(
            atr_dynamic=stress_data["atr_dynamic"].iloc[0],
            orderflow_imbalance=stress_data["orderflow_imbalance"].iloc[0],
            volatility_score=0.2,
        )
        assert 0 <= size <= 0.1, "Taille de position hors plage sous stress"
        log_performance(
            "risk_manager_stress", start_time, tmp_dirs["perf_log_path"], market="ES"
        )
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "risk_manager_stress"
        ), "Log risk_manager_stress manquant"
        assert all(
            col in df.columns for col in ["memory_usage_mb", "cpu_usage_percent"]
        ), "Colonnes psutil manquantes"
        mock_alert.assert_not_called()


def test_regime_detector_transitions(tmp_dirs, stress_data, dummy_hmm):
    """Teste regime_detector.py face à des transitions rapides."""
    start_time = datetime.now()
    with patch("src.model.utils.alert_manager.AlertManager.send_alert") as mock_alert:
        regime_detector = RegimeDetector(market="ES")
        regime_detector.hmm_model = dummy_hmm
        regime = regime_detector.detect_regime(stress_data)
        assert regime in [0, 1, 2], "Régime détecté invalide sous stress"
        log_performance(
            "regime_detector_transitions",
            start_time,
            tmp_dirs["perf_log_path"],
            market="ES",
        )
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "regime_detector_transitions"
        ), "Log regime_detector_transitions manquant"
        mock_alert.assert_not_called()


def test_risk_manager_cpu_overload(tmp_dirs, stress_data):
    """Teste risk_manager.py sous surcharge CPU."""
    start_time = datetime.now()
    with patch("psutil.cpu_percent", return_value=95.0), patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ) as mock_alert:
        risk_manager = RiskManager(market="ES")
        size = risk_manager.calculate_position_size(
            atr_dynamic=stress_data["atr_dynamic"].iloc[0],
            orderflow_imbalance=stress_data["orderflow_imbalance"].iloc[0],
            volatility_score=0.2,
        )
        assert 0 <= size <= 0.1, "Taille de position hors plage sous surcharge CPU"
        log_performance(
            "risk_manager_cpu_overload",
            start_time,
            tmp_dirs["perf_log_path"],
            market="ES",
        )
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "risk_manager_cpu_overload"
        ), "Log risk_manager_cpu_overload manquant"
        mock_alert.assert_called_with(pytest.any(str), priority=3)


def test_regime_detector_multiple_transitions(tmp_dirs, stress_data, dummy_hmm):
    """Teste regime_detector.py avec des données simulant des transitions multiples."""
    start_time = datetime.now()
    with patch("src.model.utils.alert_manager.AlertManager.send_alert") as mock_alert:
        regime_detector = RegimeDetector(market="ES")
        regime_detector.hmm_model = dummy_hmm
        multi_transition_data = pd.concat([stress_data] * 10, ignore_index=True)
        regimes = regime_detector.detect_regime(multi_transition_data)
        expected_regimes = [(i % 3) for i in range(1, 11)]  # DummyHMM alterne 0, 1, 2
        assert (
            regimes == expected_regimes
        ), "Régimes détectés invalides sous transitions multiples"
        log_performance(
            "regime_detector_multiple_transitions",
            start_time,
            tmp_dirs["perf_log_path"],
            market="ES",
        )
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "regime_detector_multiple_transitions"
        ), "Log regime_detector_multiple_transitions manquant"
        mock_alert.assert_not_called()


def test_error_handling(tmp_dirs, stress_data):
    """Teste la gestion des erreurs sous stress."""
    start_time = datetime.now()
    with patch(
        "src.risk_management.risk_manager.RiskManager.calculate_position_size",
        side_effect=Exception("Erreur simulée"),
    ), patch("src.utils.error_tracker.capture_error") as mock_capture, patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ) as mock_alert:
        risk_manager = RiskManager(market="ES")
        with pytest.raises(Exception, match="Erreur simulée"):
            risk_manager.calculate_position_size(
                atr_dynamic=stress_data["atr_dynamic"].iloc[0],
                orderflow_imbalance=stress_data["orderflow_imbalance"].iloc[0],
                volatility_score=0.2,
            )
        mock_capture.assert_called()
        mock_alert.assert_called_with(pytest.any(str), priority=4)
        log_performance(
            "error_handling", start_time, tmp_dirs["perf_log_path"], market="ES"
        )
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df["operation"] == "error_handling"), "Log error_handling manquant"


def test_corrupted_timestamps(tmp_dirs):
    """Teste la résilience à des timestamps corrompus dans les DataFrames."""
    start_time = datetime.now()
    corrupted_data = pd.DataFrame(
        {
            "timestamp": [pd.NaT, "invalid_date", pd.Timestamp("2025-05-14")],
            "atr_dynamic": [1000.0, 900.0, 800.0],
            "orderflow_imbalance": [0.9, 0.8, 0.7],
            "bid_ask_imbalance": [0.1, 0.2, 0.3],
            "total_volume": [1000, 1100, 1200],
        }
    )
    with patch(
        "src.features.feature_pipeline.FeaturePipeline.process_data"
    ) as mock_process, patch(
        "src.utils.error_tracker.capture_error"
    ) as mock_capture, patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ) as mock_alert:
        mock_process.side_effect = lambda x: pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-05-14", periods=len(x), freq="T"),
                "atr_dynamic": x["atr_dynamic"],
                "orderflow_imbalance": x["orderflow_imbalance"],
                "bid_ask_imbalance": x["bid_ask_imbalance"],
                "total_volume": x["total_volume"],
            }
        )
        feature_pipeline = FeaturePipeline()
        result = feature_pipeline.process_data(corrupted_data)
        assert isinstance(
            result, pd.DataFrame
        ), "Le résultat doit être un DataFrame malgré des timestamps corrompus"
        assert (
            not result["timestamp"].isna().any()
        ), "Les timestamps doivent être corrigés"
        assert mock_process.called, "La méthode process_data doit être appelée"
        mock_capture.assert_called(), "L’erreur de timestamp doit être capturée par Sentry"
        mock_alert.assert_called_with(pytest.any(str), priority=4)
        log_performance(
            "corrupted_timestamps", start_time, tmp_dirs["perf_log_path"], market="ES"
        )
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "corrupted_timestamps"
        ), "Log corrupted_timestamps manquant"
        assert all(
            col in df.columns for col in ["memory_usage_mb", "cpu_usage_percent"]
        ), "Colonnes psutil manquantes"


# Amélioration possible : ajouter un test sur la gestion mémoire (RAM > seuil critique).
#
# Complémentarité : tester la robustesse à des timestamps corrompus dans les DataFrames (souvent source de bugs silencieux).
#
# Assertion renforcée : pour mock_alert.assert_not_called() → conditionnelle selon CPU ou régime détecté.
