# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_run_system.py
# Tests unitaires pour src/scripts/run_system.py.
#
# Version : 2.1.5
# Date : 2025-05-14
#
# Rôle : Vérifie les fonctionnalités de SystemRunner, incluant l'initialisation,
#        la validation de la configuration, les étapes du pipeline global (collecte, features, trading),
#        les retries, les snapshots JSON compressés, les logs de performance, les alertes,
#        l'intégration de TradeProbabilityPredictor pour trade_success_prob et signal_metadata,
#        la journalisation des métadonnées (normalized_score, conflict_coefficient, run_id),
#        les alertes visuelles pour conflits élevés, et la sauvegarde des contributions de SignalResolver.
#        Conforme à la Phase 1 (collecte via IQFeed), Phase 8 (auto-conscience via alertes),
#        et Phase 16 (ensemble et transfer learning).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pandas>=2.0.0
# - psutil>=5.9.8
# - src.scripts.run_system
# - src.data.data_provider
# - src.features.feature_pipeline
# - src.trading.live_trading
# - src.model.trade_probability
# - src.model.utils.config_manager
# - src.model.utils.alert_manager
# - src.model.utils.miya_console
# - src.utils.telegram_alert
# - src.utils.standard
#
# Inputs :
# - Fichier de configuration factice (es_config.yaml)
# - Répertoires temporaires pour logs et snapshots
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de run_system.py.
#
# Notes :
# - Utilise des mocks pour simuler les dépendances externes.
# - Vérifie l'absence de références à dxFeed, obs_t, 320/81 features.
# - Couvre l'intégration de SignalResolver (métadonnées, alertes pour conflits, contributions).
# - Policies Note: The official directory for routing policies is src/model/router/policies.

import gzip
import json
import os
from unittest.mock import patch

import pandas as pd
import pytest

from src.scripts.run_system import SystemRunner


@pytest.fixture
def temp_dir(tmp_path):
    """Crée un répertoire temporaire pour les tests."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    data_dir = tmp_path / "data"
    logs_dir = data_dir / "logs" / "system"
    snapshots_dir = data_dir / "system_snapshots"
    logs_dir.mkdir(parents=True)
    snapshots_dir.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def mock_config(temp_dir):
    """Crée un fichier es_config.yaml simulé."""
    config_path = temp_dir / "config" / "es_config.yaml"
    config_content = """
system:
  mode: "paper"
  data_provider:
    source: "iqfeed"
  feature_pipeline:
    num_features: 350
    shap_features: 150
    required_features:
      - bid_ask_imbalance
      - trade_aggressiveness
      - iv_skew
      - iv_term_structure
      - vix_es_correlation
      - call_iv_atm
      - option_skew
      - news_impact_score
  trading:
    max_position_size: 5
    reward_threshold: 0.01
    news_impact_threshold: 0.5
    trade_success_threshold: 0.6
  retry_attempts: 3
  retry_delay_base: 2.0
  buffer_size: 100
  signal_resolver:
    thresholds:
      conflict_coefficient_alert: 0.5
"""
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def runner(temp_dir, mock_config, monkeypatch):
    """Initialise SystemRunner avec des mocks."""
    monkeypatch.setattr("src.scripts.run_system.CONFIG_PATH", str(mock_config))
    monkeypatch.setattr(
        "src.scripts.run_system.config_manager.get_config",
        lambda x: {
            "system": {
                "mode": "paper",
                "data_provider": {"source": "iqfeed"},
                "feature_pipeline": {
                    "num_features": 350,
                    "shap_features": 150,
                    "required_features": [
                        "bid_ask_imbalance",
                        "trade_aggressiveness",
                        "iv_skew",
                        "iv_term_structure",
                        "vix_es_correlation",
                        "call_iv_atm",
                        "option_skew",
                        "news_impact_score",
                    ],
                },
                "trading": {
                    "max_position_size": 5,
                    "reward_threshold": 0.01,
                    "news_impact_threshold": 0.5,
                    "trade_success_threshold": 0.6,
                },
                "retry_attempts": 3,
                "retry_delay_base": 2.0,
                "buffer_size": 100,
                "signal_resolver": {"thresholds": {"conflict_coefficient_alert": 0.5}},
            }
        },
    )
    runner = SystemRunner(mode="paper", config_path=mock_config)
    return runner


def test_init_runner(temp_dir, mock_config, monkeypatch):
    """Teste l'initialisation de SystemRunner."""
    runner = SystemRunner(mode="paper", config_path=mock_config)
    assert runner.mode == "paper"
    assert runner.config["data_provider"]["source"] == "iqfeed"
    assert runner.config["feature_pipeline"]["required_features"] == [
        "bid_ask_imbalance",
        "trade_aggressiveness",
        "iv_skew",
        "iv_term_structure",
        "vix_es_correlation",
        "call_iv_atm",
        "option_skew",
        "news_impact_score",
    ]
    assert os.path.exists(temp_dir / "data" / "system_snapshots")
    snapshots = list(
        (temp_dir / "data" / "system_snapshots").glob("snapshot_init_*.json.gz")
    )
    assert len(snapshots) >= 1


def test_validate_config_invalid(temp_dir, monkeypatch):
    """Teste la validation avec une configuration invalide."""
    invalid_config = temp_dir / "config" / "invalid.yaml"
    with pytest.raises(FileNotFoundError, match="Fichier de configuration introuvable"):
        SystemRunner(mode="paper", config_path=invalid_config)

    with patch("src.scripts.run_system.config_manager.get_config", return_value={}):
        with pytest.raises(ValueError, match="Clé 'system' manquante"):
            SystemRunner(
                mode="paper", config_path=temp_dir / "config" / "es_config.yaml"
            )

    with pytest.raises(ValueError, match="Mode invalide"):
        SystemRunner(mode="invalid", config_path=temp_dir / "config" / "es_config.yaml")

    with patch(
        "src.scripts.run_system.config_manager.get_config",
        return_value={
            "system": {
                "mode": "paper",
                "data_provider": {"source": "iqfeed"},
                "feature_pipeline": {
                    "num_features": 350,
                    "shap_features": 150,
                    "required_features": [],
                },
                "trading": {"max_position_size": 5},
            }
        },
    ):
        with pytest.raises(ValueError, match="Features requises manquantes"):
            SystemRunner(
                mode="paper", config_path=temp_dir / "config" / "es_config.yaml"
            )


def test_collect_data(temp_dir, runner, monkeypatch):
    """Teste la collecte de données."""
    mock_data = pd.DataFrame(
        {
            "bid_ask_imbalance": [0.1, 0.2, 0.3],
            "trade_aggressiveness": [0.2, 0.3, 0.4],
            "iv_skew": [0.01, 0.02, 0.03],
            "iv_term_structure": [0.02, 0.03, 0.04],
            "vix_es_correlation": [20, 21, 22],
            "call_iv_atm": [0.15, 0.16, 0.17],
            "option_skew": [0.02, 0.03, 0.04],
            "news_impact_score": [0.5, 0.6, 0.7],
        }
    )
    with patch("src.data.data_provider.DataProvider.get_data", return_value=mock_data):
        data = runner.collect_data()
    assert len(data) == 3
    assert all(
        f in data.columns
        for f in [
            "bid_ask_imbalance",
            "trade_aggressiveness",
            "iv_skew",
            "iv_term_structure",
            "vix_es_correlation",
            "call_iv_atm",
            "option_skew",
            "news_impact_score",
        ]
    )
    snapshots = list(
        (temp_dir / "data" / "system_snapshots").glob("snapshot_collect_data_*.json.gz")
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "run_system_performance.csv"
    assert perf_log.exists()


def test_generate_features(temp_dir, runner, monkeypatch):
    """Teste la génération de features."""
    mock_data = pd.DataFrame(
        {
            "bid_ask_imbalance": [0.1, 0.2, 0.3],
            "trade_aggressiveness": [0.2, 0.3, 0.4],
            "iv_skew": [0.01, 0.02, 0.03],
            "iv_term_structure": [0.02, 0.03, 0.04],
            "vix_es_correlation": [20, 21, 22],
            "call_iv_atm": [0.15, 0.16, 0.17],
            "option_skew": [0.02, 0.03, 0.04],
            "news_impact_score": [0.5, 0.6, 0.7],
        }
    )
    mock_features = {
        "bid_ask_imbalance": 0.2,
        "trade_aggressiveness": 0.3,
        "iv_skew": 0.02,
        "iv_term_structure": 0.03,
        "vix_es_correlation": 21,
        "call_iv_atm": 0.16,
        "option_skew": 0.03,
        "news_impact_score": 0.6,
    }
    with patch(
        "src.features.feature_pipeline.FeaturePipeline.generate_features",
        return_value=mock_features,
    ):
        features = runner.generate_features(mock_data)
    assert len(features) == 8
    assert all(
        f in features
        for f in [
            "bid_ask_imbalance",
            "trade_aggressiveness",
            "iv_skew",
            "iv_term_structure",
            "vix_es_correlation",
            "call_iv_atm",
            "option_skew",
            "news_impact_score",
        ]
    )
    snapshots = list(
        (temp_dir / "data" / "system_snapshots").glob(
            "snapshot_generate_features_*.json.gz"
        )
    )
    assert len(snapshots) >= 1


def test_execute_trading(temp_dir, runner, monkeypatch):
    """Teste l'exécution du trading."""
    mock_features = {
        "bid_ask_imbalance": 0.2,
        "trade_aggressiveness": 0.3,
        "iv_skew": 0.02,
        "iv_term_structure": 0.03,
        "vix_es_correlation": 21,
        "call_iv_atm": 0.16,
        "option_skew": 0.03,
        "news_impact_score": 0.6,
    }
    mock_decision = {"action": "buy", "size": 1}
    with patch(
        "src.trading.live_trading.LiveTrading.process_features",
        return_value=mock_decision,
    ), patch(
        "src.model.trade_probability.TradeProbabilityPredictor.predict",
        return_value=0.75,
    ), patch(
        "src.model.trade_probability.TradeProbabilityPredictor.resolve_signals",
        return_value={
            "normalized_score": 0.6,
            "conflict_coefficient": 0.3,
            "entropy": 0.2,
            "run_id": "test_run_123",
            "contributions": {
                "vix_es_correlation": {"value": 1.0, "weight": 1.0, "contribution": 1.0}
            },
        },
    ):
        decision = runner.execute_trading(mock_features)
    assert decision["action"] == "buy"
    snapshots = list(
        (temp_dir / "data" / "system_snapshots").glob(
            "snapshot_execute_trading_*.json.gz"
        )
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "run_system_performance.csv"
    assert perf_log.exists()
    df = pd.read_csv(perf_log)
    assert any(df["operation"] == "execute_trading"), "Log execute_trading manquant"
    assert any(df["trade_success_prob"].notnull()), "trade_success_prob non journalisé"
    assert any(df["normalized_score"].notnull()), "normalized_score non journalisé"
    assert any(
        df["conflict_coefficient"].notnull()
    ), "conflict_coefficient non journalisé"


def test_run_pipeline_success(temp_dir, runner, monkeypatch):
    """Teste l'exécution complète du pipeline."""
    mock_data = pd.DataFrame(
        {
            "bid_ask_imbalance": [0.1, 0.2, 0.3],
            "trade_aggressiveness": [0.2, 0.3, 0.4],
            "iv_skew": [0.01, 0.02, 0.03],
            "iv_term_structure": [0.02, 0.03, 0.04],
            "vix_es_correlation": [20, 21, 22],
            "call_iv_atm": [0.15, 0.16, 0.17],
            "option_skew": [0.02, 0.03, 0.04],
            "news_impact_score": [0.5, 0.6, 0.7],
        }
    )
    mock_features = {
        "bid_ask_imbalance": 0.2,
        "trade_aggressiveness": 0.3,
        "iv_skew": 0.02,
        "iv_term_structure": 0.03,
        "vix_es_correlation": 21,
        "call_iv_atm": 0.16,
        "option_skew": 0.03,
        "news_impact_score": 0.6,
    }
    mock_decision = {"action": "buy", "size": 1}

    with patch(
        "src.data.data_provider.DataProvider.get_data", return_value=mock_data
    ), patch(
        "src.features.feature_pipeline.FeaturePipeline.generate_features",
        return_value=mock_features,
    ), patch(
        "src.trading.live_trading.LiveTrading.process_features",
        return_value=mock_decision,
    ), patch(
        "src.model.trade_probability.TradeProbabilityPredictor.predict",
        return_value=0.75,
    ), patch(
        "src.model.trade_probability.TradeProbabilityPredictor.resolve_signals",
        return_value={
            "normalized_score": 0.6,
            "conflict_coefficient": 0.3,
            "entropy": 0.2,
            "run_id": "test_run_123",
            "contributions": {
                "vix_es_correlation": {"value": 1.0, "weight": 1.0, "contribution": 1.0}
            },
        },
    ):
        result = runner.run_pipeline()

    assert result is True
    snapshots = list(
        (temp_dir / "data" / "system_snapshots").glob("snapshot_run_pipeline_*.json.gz")
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "run_system_performance.csv"
    assert perf_log.exists()


def test_run_pipeline_failure(temp_dir, runner, monkeypatch):
    """Teste l'exécution du pipeline avec une erreur."""
    with patch(
        "src.data.data_provider.DataProvider.get_data",
        side_effect=ValueError("Erreur collecte"),
    ):
        with pytest.raises(ValueError, match="Erreur collecte"):
            runner.run_pipeline()

    snapshots = list(
        (temp_dir / "data" / "system_snapshots").glob("snapshot_run_pipeline_*.json.gz")
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "run_system_performance.csv"
    assert perf_log.exists()


def test_no_obsolete_references(temp_dir, mock_config):
    """Vérifie l'absence de références à dxFeed, obs_t, 320/81 features."""
    with open(mock_config, "r") as f:
        content = f.read()
    assert "dxFeed" not in content, "Référence à dxFeed trouvée"
    assert "obs_t" not in content, "Référence à obs_t trouvée"
    assert "320 features" not in content, "Référence à 320 features trouvée"
    assert "81 features" not in content, "Référence à 81 features trouvée"


def test_execute_trading_high_conflict(temp_dir, runner, monkeypatch):
    """Teste les alertes visuelles pour un conflict_coefficient élevé."""
    mock_features = {
        "bid_ask_imbalance": 0.2,
        "trade_aggressiveness": 0.3,
        "iv_skew": 0.02,
        "iv_term_structure": 0.03,
        "vix_es_correlation": 21,
        "call_iv_atm": 0.16,
        "option_skew": 0.03,
        "news_impact_score": 0.6,
    }
    mock_decision = {"action": "buy", "size": 1}
    with patch(
        "src.trading.live_trading.LiveTrading.process_features",
        return_value=mock_decision,
    ), patch(
        "src.model.trade_probability.TradeProbabilityPredictor.predict",
        return_value=0.75,
    ), patch(
        "src.model.trade_probability.TradeProbabilityPredictor.resolve_signals",
        return_value={
            "normalized_score": 0.6,
            "conflict_coefficient": 0.7,
            "entropy": 0.2,
            "run_id": "test_run_123",
            "contributions": {
                "vix_es_correlation": {"value": 1.0, "weight": 1.0, "contribution": 1.0}
            },
        },
    ), patch(
        "src.model.utils.miya_console.miya_alerts"
    ) as mock_miya_alerts, patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ) as mock_alert_manager, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        decision = runner.execute_trading(mock_features)
    assert decision["action"] == "buy"
    mock_miya_alerts.assert_called_with(
        "Conflit de signaux détecté. Analyse requise.",
        tag="RUN_SYSTEM",
        voice_profile="urgent",
        priority=3,
    )
    mock_alert_manager.assert_called_with(
        "Conflit de signaux détecté. Analyse requise.", priority=3
    )
    mock_telegram.assert_called_with("Conflit de signaux détecté. Analyse requise.")
    perf_log = temp_dir / "data" / "logs" / "run_system_performance.csv"
    assert perf_log.exists()
    df = pd.read_csv(perf_log)
    assert any(
        df["conflict_coefficient"] >= 0.7
    ), "conflict_coefficient non journalisé correctement"


def test_execute_trading_snapshot_contributions(temp_dir, runner, monkeypatch):
    """Teste la sauvegarde des contributions dans les snapshots."""
    mock_features = {
        "bid_ask_imbalance": 0.2,
        "trade_aggressiveness": 0.3,
        "iv_skew": 0.02,
        "iv_term_structure": 0.03,
        "vix_es_correlation": 21,
        "call_iv_atm": 0.16,
        "option_skew": 0.03,
        "news_impact_score": 0.6,
    }
    mock_decision = {"action": "buy", "size": 1}
    mock_metadata = {
        "normalized_score": 0.6,
        "conflict_coefficient": 0.3,
        "entropy": 0.2,
        "run_id": "test_run_123",
        "contributions": {
            "vix_es_correlation": {"value": 1.0, "weight": 1.0, "contribution": 1.0}
        },
    }
    with patch(
        "src.trading.live_trading.LiveTrading.process_features",
        return_value=mock_decision,
    ), patch(
        "src.model.trade_probability.TradeProbabilityPredictor.predict",
        return_value=0.75,
    ), patch(
        "src.model.trade_probability.TradeProbabilityPredictor.resolve_signals",
        return_value=mock_metadata,
    ):
        decision = runner.execute_trading(mock_features)
    assert decision["action"] == "buy"
    snapshots = list(
        (temp_dir / "data" / "system_snapshots").glob(
            "snapshot_execute_trading_*.json.gz"
        )
    )
    assert len(snapshots) >= 1
    with gzip.open(snapshots[-1], "rt", encoding="utf-8") as f:
        snapshot = json.load(f)
    assert (
        "contributions" in snapshot["data"]
    ), "Contributions manquantes dans le snapshot"
    assert (
        snapshot["data"]["contributions"] == mock_metadata["contributions"]
    ), "Contributions incorrectes dans le snapshot"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
