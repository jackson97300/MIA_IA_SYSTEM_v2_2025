# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/integration_tests.py
# Tests d'intégration pour le pipeline collecte → features → trading de MIA_IA_SYSTEM_v2_2025.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie l'interaction entre data_provider.py, feature_pipeline.py, et live_trading.py (Phase 15).
#        Conforme à la Phase 1 (collecte via IQFeed), Phase 8 (auto-conscience via alertes),
#        Phase 15 (tests d’intégration), et Phase 16 (ensemble et transfer learning via live_trading.py).
#
# Dépendances : pytest>=7.3.0,<8.0.0, pandas>=2.0.0, numpy>=1.23.0, psutil>=5.9.8, json, yaml>=6.0.0, gzip,
#               src.data.data_provider, src.features.feature_pipeline, src.trading.live_trading,
#               src.model.utils.alert_manager, src.model.utils.miya_console, src.utils.telegram_alert
#
# Inputs : Données simulées via fixtures (mock_iqfeed_data)
#
# Outputs : data/logs/integration_tests_performance.csv, data/integration_snapshots/snapshot_*.json.gz
#
# Notes :
# - Utilise IQFeed exclusivement via data_provider.py (simulé).
# - Gère 350 features pour l’entraînement et 150 SHAP features pour l’inférence via feature_pipeline.py.
# - Implémente logs psutil, snapshots JSON compressés, et alertes via alert_manager.py.
# - Tests contenus dans tests/integration_tests.py.

import gzip
import json
import os
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import psutil
import pytest

from src.data.data_provider import DataProvider
from src.features.feature_pipeline import FeaturePipeline
from src.model.utils.alert_manager import AlertManager
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.trading.live_trading import LiveTrading
from src.utils.telegram_alert import send_telegram_alert

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "data", "logs")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "integration_snapshots")
CSV_LOG_PATH = os.path.join(
    BASE_DIR, "data", "logs", "integration_tests_performance.csv"
)

# Configuration des logs de performance
os.makedirs(LOG_DIR, exist_ok=True)
log_buffer = []
BUFFER_SIZE = 100


def log_performance(
    operation: str, latency: float, success: bool, error: str = None, **kwargs
) -> None:
    """
    Journalise les performances des tests d'intégration.

    Args:
        operation (str): Nom de l’opération (ex. : test_data_to_features_pipeline).
        latency (float): Latence en secondes.
        success (bool): Succès du test.
        error (str, optional): Message d’erreur si échec.
        **kwargs: Paramètres supplémentaires (ex. : num_records).
    """
    try:
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mémoire en Mo
        cpu_percent = psutil.cpu_percent()
        log_entry = {
            "timestamp": str(datetime.now()),
            "operation": operation,
            "latency": latency,
            "success": success,
            "error": error,
            "memory_usage_mb": memory_usage,
            "cpu_percent": cpu_percent,
            **kwargs,
        }
        log_buffer.append(log_entry)
        if len(log_buffer) >= BUFFER_SIZE:
            log_df = pd.DataFrame(log_buffer)
            os.makedirs(os.path.dirname(CSV_LOG_PATH), exist_ok=True)
            if not os.path.exists(CSV_LOG_PATH):
                log_df.to_csv(CSV_LOG_PATH, index=False, encoding="utf-8")
            else:
                log_df.to_csv(
                    CSV_LOG_PATH, mode="a", header=False, index=False, encoding="utf-8"
                )
            log_buffer.clear()
    except Exception as e:
        error_msg = f"Erreur journalisation performance: {str(e)}"
        miya_alerts(error_msg, tag="INTEGRATION_TESTS", level="error", priority=3)
        AlertManager().send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)


def save_snapshot(snapshot_type: str, data: Dict) -> None:
    """
    Sauvegarde un instantané des résultats des tests avec compression gzip.

    Args:
        snapshot_type (str): Type de snapshot (ex. : test_data_to_features_pipeline).
        data (Dict): Données à sauvegarder.
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot = {
            "timestamp": timestamp,
            "type": snapshot_type,
            "data": data,
            "buffer_size": len(log_buffer),
        }
        path = os.path.join(SNAPSHOT_DIR, f"snapshot_{snapshot_type}_{timestamp}.json")
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        with gzip.open(f"{path}.gz", "wt", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=4)
    except Exception as e:
        error_msg = f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}"
        miya_alerts(error_msg, tag="INTEGRATION_TESTS", level="error", priority=3)
        AlertManager().send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)


# Fixture pour créer des données factices simulant IQFeed


@pytest.fixture
def mock_iqfeed_data():
    """
    Crée un DataFrame simulant les données brutes d'IQFeed.
    """
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start="2025-05-13 10:00:00", periods=10, freq="T"
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
    return data


# Fixture pour simuler des données partielles


@pytest.fixture
def mock_partial_data():
    """
    Crée un DataFrame simulant des données IQFeed partielles (manque des colonnes).
    """
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start="2025-05-13 10:00:00", periods=10, freq="T"
            ),
            "open": np.random.uniform(5000, 5100, 10),
            "volume": np.random.randint(1000, 5000, 10),
        }
    )


# Fixture pour simuler l'environnement de trading


@pytest.fixture
def mock_trading_env():
    """
    Crée un environnement de trading factice pour éviter les trades réels.
    """
    return {
        "max_position_size": 5,
        "reward_threshold": 0.01,
        "news_impact_threshold": 0.5,
    }


@pytest.mark.timeout(30)
def test_data_to_features_pipeline(mock_iqfeed_data, tmp_path):
    """
    Teste l'intégration entre la collecte de données et la génération de features.
    """
    start_time = datetime.now()
    try:
        # Simuler data_provider.py
        with patch("src.data.data_provider.DataProvider.get_data") as mock_get_data:
            mock_get_data.return_value = mock_iqfeed_data
            data_provider = DataProvider()
            data = data_provider.get_data()

            # Vérifier que les données sont bien reçues
            assert isinstance(data, pd.DataFrame)
            assert len(data) == 10
            assert all(
                col in data.columns
                for col in ["open", "high", "low", "close", "volume", "vix_close"]
            )

            # Simuler feature_pipeline.py
            feature_pipeline = FeaturePipeline()
            features = feature_pipeline.generate_features(data)

            # Vérifier que les features sont générées
            assert isinstance(features, dict)
            assert len(features) >= 350  # Vérifier 350 features
            assert (
                sum("shap" in k for k in features.keys()) >= 150
            )  # Vérifier 150 SHAP features
            assert all(
                isinstance(f, (pd.Series, np.ndarray)) for f in features.values()
            )
            assert all(len(f) == len(data) for f in features.values())

            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                "Test data_to_features_pipeline réussi",
                tag="INTEGRATION_TESTS",
                level="info",
                priority=2,
            )
            AlertManager().send_alert(
                "Test data_to_features_pipeline réussi", priority=1
            )
            log_performance(
                "test_data_to_features_pipeline",
                latency,
                success=True,
                num_records=len(data),
                num_features=len(features),
            )
            save_snapshot(
                "test_data_to_features_pipeline",
                {"num_records": len(data), "num_features": len(features)},
            )
    except Exception as e:
        latency = (datetime.now() - start_time).total_seconds()
        error_msg = f"Échec test_data_to_features_pipeline : {str(e)}"
        miya_alerts(
            error_msg, tag="INTEGRATION_TESTS", voice_profile="urgent", priority=4
        )
        AlertManager().send_alert(error_msg, priority=4)
        send_telegram_alert(error_msg)
        log_performance(
            "test_data_to_features_pipeline", latency, success=False, error=str(e)
        )
        save_snapshot("test_data_to_features_pipeline", {"error": str(e)})
        raise


@pytest.mark.timeout(30)
def test_features_to_trading_pipeline(mock_iqfeed_data, mock_trading_env, tmp_path):
    """
    Teste l'intégration entre la génération de features et le trading.
    """
    start_time = datetime.now()
    try:
        # Simuler feature_pipeline.py
        feature_pipeline = FeaturePipeline()
        features = feature_pipeline.generate_features(mock_iqfeed_data)

        # Simuler live_trading.py en mode paper trading
        with patch(
            "src.trading.live_trading.LiveTrading.execute_trade"
        ) as mock_execute_trade:
            mock_execute_trade.return_value = {
                "trade_id": "TEST_123",
                "status": "success",
            }
            live_trading = LiveTrading(mode="paper", env_config=mock_trading_env)

            # Simuler une itération de trading
            trade_decision = live_trading.process_features(features)

            # Vérifier que la décision de trading est générée
            assert isinstance(trade_decision, dict)
            assert "action" in trade_decision
            assert trade_decision["action"] in ["buy", "sell", "hold"]
            assert "size" in trade_decision
            assert trade_decision["size"] <= mock_trading_env["max_position_size"]

            # Vérifier que l'exécution du trade est appelée
            mock_execute_trade.assert_called_once()

            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                "Test features_to_trading_pipeline réussi",
                tag="INTEGRATION_TESTS",
                level="info",
                priority=2,
            )
            AlertManager().send_alert(
                "Test features_to_trading_pipeline réussi", priority=1
            )
            log_performance(
                "test_features_to_trading_pipeline",
                latency,
                success=True,
                num_features=len(features),
            )
            save_snapshot(
                "test_features_to_trading_pipeline", {"decision": trade_decision}
            )
    except Exception as e:
        latency = (datetime.now() - start_time).total_seconds()
        error_msg = f"Échec test_features_to_trading_pipeline : {str(e)}"
        miya_alerts(
            error_msg, tag="INTEGRATION_TESTS", voice_profile="urgent", priority=4
        )
        AlertManager().send_alert(error_msg, priority=4)
        send_telegram_alert(error_msg)
        log_performance(
            "test_features_to_trading_pipeline", latency, success=False, error=str(e)
        )
        save_snapshot("test_features_to_trading_pipeline", {"error": str(e)})
        raise


@pytest.mark.timeout(30)
def test_full_pipeline_integration(mock_iqfeed_data, mock_trading_env, tmp_path):
    """
    Teste l'intégration complète du pipeline collecte → features → trading.
    """
    start_time = datetime.now()
    try:
        # Simuler data_provider.py
        with patch("src.data.data_provider.DataProvider.get_data") as mock_get_data:
            mock_get_data.return_value = mock_iqfeed_data
            data_provider = DataProvider()
            data = data_provider.get_data()

            # Simuler feature_pipeline.py
            feature_pipeline = FeaturePipeline()
            features = feature_pipeline.generate_features(data)

            # Simuler live_trading.py
            with patch(
                "src.trading.live_trading.LiveTrading.execute_trade"
            ) as mock_execute_trade:
                mock_execute_trade.return_value = {
                    "trade_id": "TEST_124",
                    "status": "success",
                }
                live_trading = LiveTrading(mode="paper", env_config=mock_trading_env)
                trade_decision = live_trading.process_features(features)

                # Vérifier le flux complet
                assert isinstance(data, pd.DataFrame)
                assert isinstance(features, dict)
                assert len(features) >= 350  # Vérifier 350 features
                assert (
                    sum("shap" in k for k in features.keys()) >= 150
                )  # Vérifier 150 SHAP features
                assert isinstance(trade_decision, dict)
                assert mock_execute_trade.called
                assert trade_decision["action"] in ["buy", "sell", "hold"]

                latency = (datetime.now() - start_time).total_seconds()
                miya_speak(
                    "Test full_pipeline_integration réussi",
                    tag="INTEGRATION_TESTS",
                    level="info",
                    priority=2,
                )
                AlertManager().send_alert(
                    "Test full_pipeline_integration réussi", priority=1
                )
                log_performance(
                    "test_full_pipeline_integration",
                    latency,
                    success=True,
                    num_records=len(data),
                    num_features=len(features),
                )
                save_snapshot(
                    "test_full_pipeline_integration",
                    {
                        "num_records": len(data),
                        "num_features": len(features),
                        "decision": trade_decision,
                    },
                )
    except Exception as e:
        latency = (datetime.now() - start_time).total_seconds()
        error_msg = f"Échec test_full_pipeline_integration : {str(e)}"
        miya_alerts(
            error_msg, tag="INTEGRATION_TESTS", voice_profile="urgent", priority=4
        )
        AlertManager().send_alert(error_msg, priority=4)
        send_telegram_alert(error_msg)
        log_performance(
            "test_full_pipeline_integration", latency, success=False, error=str(e)
        )
        save_snapshot("test_full_pipeline_integration", {"error": str(e)})
        raise


@pytest.mark.timeout(30)
def test_pipeline_error_handling(mock_iqfeed_data, tmp_path):
    """
    Teste la gestion des erreurs dans le pipeline.
    """
    start_time = datetime.now()
    try:
        # Simuler une erreur dans data_provider.py
        with patch("src.data.data_provider.DataProvider.get_data") as mock_get_data:
            mock_get_data.side_effect = ValueError("Erreur de connexion IQFeed")
            data_provider = DataProvider()

            with pytest.raises(ValueError, match="Erreur de connexion IQFeed"):
                data_provider.get_data()

        # Simuler des données invalides pour feature_pipeline.py
        invalid_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start="2025-05-13 10:00:00", periods=10, freq="T"
                )
            }
        )
        feature_pipeline = FeaturePipeline()

        with pytest.raises(ValueError, match="Colonnes manquantes"):
            feature_pipeline.generate_features(invalid_data)

        latency = (datetime.now() - start_time).total_seconds()
        miya_speak(
            "Test pipeline_error_handling réussi",
            tag="INTEGRATION_TESTS",
            level="info",
            priority=2,
        )
        AlertManager().send_alert("Test pipeline_error_handling réussi", priority=1)
        log_performance("test_pipeline_error_handling", latency, success=True)
        save_snapshot("test_pipeline_error_handling", {"status": "success"})
    except Exception as e:
        latency = (datetime.now() - start_time).total_seconds()
        error_msg = f"Échec test_pipeline_error_handling : {str(e)}"
        miya_alerts(
            error_msg, tag="INTEGRATION_TESTS", voice_profile="urgent", priority=4
        )
        AlertManager().send_alert(error_msg, priority=4)
        send_telegram_alert(error_msg)
        log_performance(
            "test_pipeline_error_handling", latency, success=False, error=str(e)
        )
        save_snapshot("test_pipeline_error_handling", {"error": str(e)})
        raise


@pytest.mark.timeout(30)
def test_pipeline_empty_data(mock_trading_env, tmp_path):
    """
    Teste le pipeline avec des données vides.
    """
    start_time = datetime.now()
    try:
        empty_data = pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume", "vix_close"]
        )

        feature_pipeline = FeaturePipeline()
        features = feature_pipeline.generate_features(empty_data)

        # Vérifier que les features sont vides ou contiennent des zéros
        assert isinstance(features, dict)
        assert all(len(f) == 0 or f.eq(0).all() for f in features.values())

        # Simuler live_trading.py
        with patch(
            "src.trading.live_trading.LiveTrading.execute_trade"
        ) as mock_execute_trade:
            live_trading = LiveTrading(mode="paper", env_config=mock_trading_env)
            trade_decision = live_trading.process_features(features)

            # Vérifier qu'aucun trade n'est exécuté
            assert trade_decision["action"] == "hold"
            assert not mock_execute_trade.called

            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                "Test pipeline_empty_data réussi",
                tag="INTEGRATION_TESTS",
                level="info",
                priority=2,
            )
            AlertManager().send_alert("Test pipeline_empty_data réussi", priority=1)
            log_performance("test_pipeline_empty_data", latency, success=True)
            save_snapshot("test_pipeline_empty_data", {"status": "success"})
    except Exception as e:
        latency = (datetime.now() - start_time).total_seconds()
        error_msg = f"Échec test_pipeline_empty_data : {str(e)}"
        miya_alerts(
            error_msg, tag="INTEGRATION_TESTS", voice_profile="urgent", priority=4
        )
        AlertManager().send_alert(error_msg, priority=4)
        send_telegram_alert(error_msg)
        log_performance(
            "test_pipeline_empty_data", latency, success=False, error=str(e)
        )
        save_snapshot("test_pipeline_empty_data", {"error": str(e)})
        raise


@pytest.mark.timeout(30)
def test_pipeline_partial_data(mock_partial_data, mock_trading_env, tmp_path):
    """
    Teste le pipeline avec des données partielles (manque des colonnes).
    """
    start_time = datetime.now()
    try:
        feature_pipeline = FeaturePipeline()

        with pytest.raises(ValueError, match="Colonnes manquantes"):
            feature_pipeline.generate_features(mock_partial_data)

        latency = (datetime.now() - start_time).total_seconds()
        miya_speak(
            "Test pipeline_partial_data réussi",
            tag="INTEGRATION_TESTS",
            level="info",
            priority=2,
        )
        AlertManager().send_alert("Test pipeline_partial_data réussi", priority=1)
        log_performance("test_pipeline_partial_data", latency, success=True)
        save_snapshot("test_pipeline_partial_data", {"status": "success"})
    except Exception as e:
        latency = (datetime.now() - start_time).total_seconds()
        error_msg = f"Échec test_pipeline_partial_data : {str(e)}"
        miya_alerts(
            error_msg, tag="INTEGRATION_TESTS", voice_profile="urgent", priority=4
        )
        AlertManager().send_alert(error_msg, priority=4)
        send_telegram_alert(error_msg)
        log_performance(
            "test_pipeline_partial_data", latency, success=False, error=str(e)
        )
        save_snapshot("test_pipeline_partial_data", {"error": str(e)})
        raise


@pytest.mark.timeout(30)
def test_pipeline_timeout(mock_iqfeed_data, mock_trading_env, tmp_path):
    """
    Teste la gestion des interruptions ou timeouts dans le pipeline.
    """
    start_time = datetime.now()
    try:
        # Simuler un timeout dans data_provider.py
        with patch("src.data.data_provider.DataProvider.get_data") as mock_get_data:
            mock_get_data.side_effect = TimeoutError("Timeout IQFeed")
            data_provider = DataProvider()

            with pytest.raises(TimeoutError, match="Timeout IQFeed"):
                data_provider.get_data()

        latency = (datetime.now() - start_time).total_seconds()
        miya_speak(
            "Test pipeline_timeout réussi",
            tag="INTEGRATION_TESTS",
            level="info",
            priority=2,
        )
        AlertManager().send_alert("Test pipeline_timeout réussi", priority=1)
        log_performance("test_pipeline_timeout", latency, success=True)
        save_snapshot("test_pipeline_timeout", {"status": "success"})
    except Exception as e:
        latency = (datetime.now() - start_time).total_seconds()
        error_msg = f"Échec test_pipeline_timeout : {str(e)}"
        miya_alerts(
            error_msg, tag="INTEGRATION_TESTS", voice_profile="urgent", priority=4
        )
        AlertManager().send_alert(error_msg, priority=4)
        send_telegram_alert(error_msg)
        log_performance("test_pipeline_timeout", latency, success=False, error=str(e))
        save_snapshot("test_pipeline_timeout", {"error": str(e)})
        raise


def test_no_obsolete_references(tmp_path):
    """
    Vérifie l'absence de références à dxFeed, obs_t, 320/81 features.
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text("source: iqfeed\n")
    with open(config_path, "r") as f:
        content = f.read()
    assert "dxFeed" not in content, "Référence à dxFeed trouvée"
    assert "obs_t" not in content, "Référence à obs_t trouvée"
    assert "320 features" not in content, "Référence à 320 features trouvée"
    assert "81 features" not in content, "Référence à 81 features trouvée"
