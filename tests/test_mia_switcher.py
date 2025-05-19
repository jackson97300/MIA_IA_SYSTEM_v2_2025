# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_mia_switcher.py
# Tests unitaires pour src/strategy/mia_switcher.py
#
# Version : 2.1.5
# Date : 2025-05-14
#
# Rôle : Valide l’initialisation de MiaSwitcher, l’arbitrage entre modèles (SAC, PPO, DDPG, LSTM),
#        la journalisation des performances, et les nouvelles fonctionnalités (position sizing dynamique,
#        microstructure, HMM, vote bayésien, métriques Prometheus, logs psutil, gestion des erreurs/alertes).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0
# - src/strategy/mia_switcher.py
# - src/risk_management/risk_manager.py
# - src/utils/mlflow_tracker.py
# - src/monitoring/prometheus_metrics.py
# - src/utils/error_tracker.py
# - src/model/utils/alert_manager.py
#
# Inputs :
# - Données factices pour les tests
# - Répertoire temporaire pour les logs
#
# Outputs :
# - Tests unitaires validant les fonctionnalités du switcher
#
# Notes :
# - Conforme à structure.txt (version 2.1.5, 2025-05-14).
# - Supporte les suggestions 1 (position sizing), 3 (microstructure), 4 (HMM), 10 (ensembles de politiques).
# - Policies Note: The official directory for routing policies is src/model/router/policies.
# The src/model/policies directory is a residual and should be verified
# for removal to avoid import conflicts.

import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.strategy.mia_switcher import MiaSwitcher


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs et snapshots."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs" / "strategy"
    logs_dir.mkdir(parents=True)
    snapshots_dir = data_dir / "strategy" / "mia_switcher_snapshots"
    snapshots_dir.mkdir()
    return {
        "base_dir": base_dir,
        "logs_dir": logs_dir,
        "snapshots_dir": snapshots_dir,
        "perf_log_path": logs_dir / "mia_switcher_performance.csv",
        "switcher_log_path": logs_dir / "mia_switcher_log.csv",
        "dashboard_path": data_dir / "strategy" / "mia_switcher_dashboard.json",
    }


@pytest.fixture
def mock_data():
    """Crée des données factices pour les tests."""
    return pd.DataFrame(
        {
            "timestamp": [datetime.now()],
            "vix": [20.0],
            "neural_regime": [0],
            "predicted_volatility": [1.5],
            "trade_frequency_1s": [8.0],
            "close": [5100.0],
            "atr_dynamic": [2.0],
            "orderflow_imbalance": [0.5],
            "bid_ask_imbalance": [0.1],
            "trade_aggressiveness": [0.2],
            "regime_hmm": [0],
            **{
                f"feat_{i}": [np.random.uniform(0, 1)] for i in range(339)
            },  # 350 features
        }
    )


@pytest.fixture
def mock_predictions():
    """Crée des prédictions factices pour les tests."""
    return {"sac": 0.7, "ppo": 0.65, "ddpg": 0.6}


def test_mia_switcher_init(tmp_dirs):
    """Teste l’instanciation de MiaSwitcher."""
    with patch(
        "src.model.utils.config_manager.config_manager.get_config",
        return_value={
            "mia_switcher": {
                "thresholds": {
                    "min_sharpe": 0.5,
                    "max_drawdown": -0.1,
                    "min_profit_factor": 1.2,
                    "vix_threshold": 30,
                    "switch_confidence_threshold": 0.8,
                    "regime_switch_frequency": 300,
                    "max_consecutive_underperformance": 3,
                    "observation_dims": {"training": 350, "inference": 150},
                },
                "weights": {
                    "sharpe_weight": 0.5,
                    "drawdown_weight": 0.3,
                    "profit_factor_weight": 0.2,
                },
                "cache": {"max_cache_size": 1000},
                "logging": {"buffer_size": 100},
                "features": {"observation_dims": {"training": 350, "inference": 150}},
                "evaluation_steps": 100,
                "max_profit_factor": 10.0,
            }
        },
    ), patch(
        "src.model.utils.config_manager.config_manager.get_config",
        return_value={
            "models": {
                "sac": {
                    "model_type": "sac",
                    "model_path": "path/to/sac",
                    "policy_type": "mlp",
                },
                "ppo": {
                    "model_type": "ppo",
                    "model_path": "path/to/ppo",
                    "policy_type": "mlp",
                },
                "ddpg": {
                    "model_type": "ddpg",
                    "model_path": "path/to/ddpg",
                    "policy_type": "mlp",
                },
            }
        },
        side_effect=[MagicMock(), MagicMock()],
    ), patch(
        "src.envs.trading_env.TradingEnv.__init__", return_value=None
    ), patch(
        "src.model.router.detect_regime.MarketRegimeDetector.__init__",
        return_value=None,
    ), patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ), patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ), patch(
        "src.risk_management.risk_manager.RiskManager.__init__", return_value=None
    ):
        with patch("src.strategy.mia_switcher.BASE_DIR", tmp_dirs["base_dir"]):
            switcher = MiaSwitcher()
        assert switcher.current_model in [
            "sac",
            "ppo",
            "ddpg",
        ], "Modèle par défaut incorrect"
        assert switcher.risk_manager is not None, "RiskManager non initialisé"
        assert switcher.alert_manager is not None, "AlertManager non initialisé"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"


def test_switch_strategy(tmp_dirs, mock_data, mock_predictions):
    """Teste la méthode switch_strategy avec position sizing, microstructure, HMM, et vote bayésien."""
    with patch(
        "src.risk_management.risk_manager.RiskManager.calculate_position_size",
        return_value=0.05,
    ), patch(
        "src.utils.mlflow_tracker.MLFlowTracker.get_model_weights",
        return_value={"sac": 0.4, "ppo": 0.3, "ddpg": 0.3},
    ), patch(
        "src.monitoring.prometheus_metrics.Gauge.labels"
    ) as mock_gauge, patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ), patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ), patch(
        "src.utils.error_tracker.capture_error"
    ):
        with patch("src.strategy.mia_switcher.BASE_DIR", tmp_dirs["base_dir"]):
            switcher = MiaSwitcher()
            result = switcher.switch_strategy(mock_data, mock_predictions)
        assert "decision" in result, "Clé 'decision' manquante"
        assert "position_size" in result, "Clé 'position_size' manquante"
        assert isinstance(result["decision"], float), "Décision doit être un float"
        assert 0 <= result["position_size"] <= 0.1, "Taille de position hors plage"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df["operation"] == "switch_strategy"), "Log switch_strategy manquant"
        assert any(df["decision"].notnull()), "Décision non journalisée"
        assert any(df["position_size"].notnull()), "Position size non journalisé"
        assert mock_gauge.call_count >= 3, "Métriques Prometheus non mises à jour"


def test_switch_strategy_invalid_data(tmp_dirs, mock_predictions):
    """Teste switch_strategy avec des données invalides."""
    invalid_data = pd.DataFrame({"timestamp": [datetime.now()], "vix": [20.0]})
    with patch(
        "src.risk_management.risk_manager.RiskManager.calculate_position_size"
    ), patch(
        "src.utils.mlflow_tracker.MLFlowTracker.get_model_weights",
        return_value={"sac": 0.4, "ppo": 0.3, "ddpg": 0.3},
    ), patch(
        "src.monitoring.prometheus_metrics.Gauge.labels"
    ), patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ), patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ), patch(
        "src.utils.error_tracker.capture_error"
    ) as mock_capture_error:
        with patch("src.strategy.mia_switcher.BASE_DIR", tmp_dirs["base_dir"]):
            switcher = MiaSwitcher()
            result = switcher.switch_strategy(invalid_data, mock_predictions)
        assert result == {
            "decision": 0.0,
            "position_size": 0.0,
        }, "Résultat par défaut incorrect"
        assert mock_capture_error.called, "Erreur non capturée"
        assert os.path.exists(tmp_dirs["perf_log_path"]), "Fichier de logs non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df["operation"] == "switch_strategy"), "Log switch_strategy manquant"
        assert any(df["success"] == False), "Échec non journalisé"


def test_switch_mia(tmp_dirs, mock_data):
    """Teste la méthode switch_mia."""
    with patch(
        "src.model.utils.config_manager.config_manager.get_config",
        return_value={
            "mia_switcher": {
                "thresholds": {
                    "min_sharpe": 0.5,
                    "max_drawdown": -0.1,
                    "min_profit_factor": 1.2,
                    "vix_threshold": 30,
                    "switch_confidence_threshold": 0.8,
                    "regime_switch_frequency": 300,
                    "max_consecutive_underperformance": 3,
                    "observation_dims": {"training": 350, "inference": 150},
                },
                "weights": {
                    "sharpe_weight": 0.5,
                    "drawdown_weight": 0.3,
                    "profit_factor_weight": 0.2,
                },
                "cache": {"max_cache_size": 1000},
                "logging": {"buffer_size": 100},
                "features": {"observation_dims": {"training": 350, "inference": 150}},
                "evaluation_steps": 100,
                "max_profit_factor": 10.0,
            }
        },
    ), patch(
        "src.model.utils.config_manager.config_manager.get_config",
        return_value={
            "models": {
                "sac": {
                    "model_type": "sac",
                    "model_path": "path/to/sac",
                    "policy_type": "mlp",
                },
                "ppo": {
                    "model_type": "ppo",
                    "model_path": "path/to/ppo",
                    "policy_type": "mlp",
                },
                "ddpg": {
                    "model_type": "ddpg",
                    "model_path": "path/to/ddpg",
                    "policy_type": "mlp",
                },
            }
        },
        side_effect=[MagicMock(), MagicMock()],
    ), patch(
        "src.envs.trading_env.TradingEnv.__init__", return_value=None
    ), patch(
        "src.model.router.detect_regime.MarketRegimeDetector.detect_market_regime_vectorized",
        return_value=("trend", [0.8, 0.1, 0.1]),
    ), patch(
        "src.model.inference.predict", return_value={"action": 0.5}
    ), patch(
        "src.envs.trading_env.TradingEnv.reset", return_value=(np.zeros(350), None)
    ), patch(
        "src.envs.trading_env.TradingEnv.step",
        return_value=(np.zeros(350), 0.1, False, False, {}),
    ), patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ), patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ), patch(
        "src.utils.error_tracker.capture_error"
    ), patch(
        "src.risk_management.risk_manager.RiskManager.__init__", return_value=None
    ):
        with patch("src.strategy.mia_switcher.BASE_DIR", tmp_dirs["base_dir"]):
            switcher = MiaSwitcher()
            context = {"neural_regime": "trend", "predicted_volatility": 1.5}
            result = switcher.switch_mia(mock_data, step=1, context=context)
        assert "model_type" in result, "Clé 'model_type' manquante"
        assert "model_path" in result, "Clé 'model_path' manquante"
        assert "policy_type" in result, "Clé 'policy_type' manquante"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df["operation"] == "switch_mia"), "Log switch_mia manquant"


def test_log_performance(tmp_dirs):
    """Teste la journalisation des performances avec psutil."""
    with patch(
        "psutil.Process.memory_info", return_value=MagicMock(rss=1024 * 1024 * 100)
    ), patch("psutil.cpu_percent", return_value=50.0), patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ), patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ), patch(
        "src.utils.error_tracker.capture_error"
    ), patch(
        "src.model.utils.performance_logger.PerformanceLogger.log"
    ):
        with patch("src.strategy.mia_switcher.BASE_DIR", tmp_dirs["base_dir"]):
            switcher = MiaSwitcher()
            switcher.log_performance(
                "test_op", 0.1, success=True, test_key="test_value"
            )
        assert os.path.exists(tmp_dirs["perf_log_path"]), "Fichier de logs non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df["operation"] == "test_op"), "Log test_op manquant"
        assert any(df["latency"] == 0.1), "Latence incorrecte"
        assert any(df["success"]), "Succès incorrect"
        assert any(df["memory_usage_mb"] == 100.0), "Mémoire incorrecte"
        assert any(df["cpu_usage_percent"] == 50.0), "CPU incorrect"
        assert any(df["test_key"] == "test_value"), "Clé supplémentaire manquante"


def test_switch_strategy_hmm_bull_bear(tmp_dirs, mock_predictions):
    """Teste l'impact des régimes HMM (bull/bear) sur la décision."""
    bull_data = pd.DataFrame(
        {
            "timestamp": [datetime.now()],
            "atr_dynamic": [2.0],
            "orderflow_imbalance": [0.5],
            "bid_ask_imbalance": [0.1],
            "trade_aggressiveness": [0.2],
            "regime_hmm": [0],  # Bull
            "predicted_volatility": [1.5],
        }
    )
    bear_data = pd.DataFrame(
        {
            "timestamp": [datetime.now()],
            "atr_dynamic": [2.0],
            "orderflow_imbalance": [0.5],
            "bid_ask_imbalance": [0.1],
            "trade_aggressiveness": [0.2],
            "regime_hmm": [1],  # Bear
            "predicted_volatility": [1.5],
        }
    )
    with patch(
        "src.risk_management.risk_manager.RiskManager.calculate_position_size",
        return_value=0.05,
    ), patch(
        "src.utils.mlflow_tracker.MLFlowTracker.get_model_weights",
        return_value={"sac": 0.4, "ppo": 0.3, "ddpg": 0.3},
    ), patch(
        "src.monitoring.prometheus_metrics.Gauge.labels"
    ), patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ), patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ), patch(
        "src.utils.error_tracker.capture_error"
    ):
        with patch("src.strategy.mia_switcher.BASE_DIR", tmp_dirs["base_dir"]):
            switcher = MiaSwitcher()
            bull_result = switcher.switch_strategy(bull_data, mock_predictions)
            bear_result = switcher.switch_strategy(bear_data, mock_predictions)
        assert (
            bull_result["decision"] > bear_result["decision"]
        ), "Décision bull doit être supérieure à bear"
        assert (
            bull_result["position_size"] == bear_result["position_size"]
        ), "Taille de position doit être identique"
        assert os.path.exists(tmp_dirs["perf_log_path"]), "Fichier de logs non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df["operation"] == "switch_strategy"), "Log switch_strategy manquant"


def test_switch_strategy_prometheus_metrics(tmp_dirs, mock_data, mock_predictions):
    """Teste la mise à jour des métriques Prometheus."""
    with patch(
        "src.risk_management.risk_manager.RiskManager.calculate_position_size",
        return_value=0.05,
    ), patch(
        "src.utils.mlflow_tracker.MLFlowTracker.get_model_weights",
        return_value={"sac": 0.4, "ppo": 0.3, "ddpg": 0.3},
    ), patch(
        "src.monitoring.prometheus_metrics.Gauge.labels"
    ) as mock_gauge, patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ), patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ), patch(
        "src.utils.error_tracker.capture_error"
    ):
        with patch("src.strategy.mia_switcher.BASE_DIR", tmp_dirs["base_dir"]):
            switcher = MiaSwitcher()
            switcher.switch_strategy(mock_data, mock_predictions)
        assert mock_gauge.call_count >= 3, "Métriques Prometheus non mises à jour"
        calls = mock_gauge.call_args_list
        sac_call = any("ensemble_weight_sac" in str(call) for call in calls)
        ppo_call = any("ensemble_weight_ppo" in str(call) for call in calls)
        ddpg_call = any("ensemble_weight_ddpg" in str(call) for call in calls)
        assert sac_call and ppo_call and ddpg_call, "Métriques Prometheus manquantes"
        assert os.path.exists(tmp_dirs["perf_log_path"]), "Fichier de logs non créé"


def test_switch_strategy_error_handling(tmp_dirs, mock_data, mock_predictions):
    """Teste la gestion des erreurs dans switch_strategy."""
    with patch(
        "src.risk_management.risk_manager.RiskManager.calculate_position_size",
        side_effect=ValueError("Erreur position size"),
    ), patch(
        "src.utils.mlflow_tracker.MLFlowTracker.get_model_weights",
        return_value={"sac": 0.4, "ppo": 0.3, "ddpg": 0.3},
    ), patch(
        "src.monitoring.prometheus_metrics.Gauge.labels"
    ), patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ), patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ), patch(
        "src.utils.error_tracker.capture_error"
    ) as mock_capture_error:
        with patch("src.strategy.mia_switcher.BASE_DIR", tmp_dirs["base_dir"]):
            switcher = MiaSwitcher()
            result = switcher.switch_strategy(mock_data, mock_predictions)
        assert result == {
            "decision": 0.0,
            "position_size": 0.0,
        }, "Résultat par défaut incorrect"
        assert mock_capture_error.called, "Erreur non capturée"
        assert os.path.exists(tmp_dirs["perf_log_path"]), "Fichier de logs non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df["operation"] == "switch_strategy"), "Log switch_strategy manquant"
        assert any(df["success"] == False), "Échec non journalisé"
