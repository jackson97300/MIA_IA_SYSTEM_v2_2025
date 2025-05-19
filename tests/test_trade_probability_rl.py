# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_trade_probability_rl.py
# Tests unitaires pour src/model/trade_probability_rl.py
#
# Version : 2.1.5
# Date : 2025-05-14
#
# Rôle : Valide l’entraînement et la prédiction des modèles RL (SAC, PPO, DDPG, PPO-CVaR, QR-DQN) dans trade_probability_rl.py,
#        incluant les coûts de transaction, microstructure, surface de volatilité, walk-forward validation,
#        Safe RL (CVaR-PPO), Distributional RL (QR-DQN), ensembles de politiques (vote bayésien),
#        journalisation MLflow/Prometheus, logs psutil, et résolution des conflits de signaux via SignalResolver.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, stable-baselines3>=2.0.0,<3.0.0,
#   rllib>=2.0.0,<3.0.0, gym>=0.21.0,<1.0.0, psutil>=5.9.8,<6.0.0
# - src/model/trade_probability_rl.py
# - src/utils/error_tracker.py
# - src/monitoring/prometheus_metrics.py
# - src/utils/mlflow_tracker.py
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
# - src/model/utils/signal_resolver.py
#
# Inputs :
# - Données factices pour les tests
# - Répertoire temporaire pour les logs et modèles
# - config/es_config.yaml pour SignalResolver
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de trade_probability_rl.py.
#
# Notes :
# - Conforme à structure.txt (version 2.1.5, 2025-05-14).
# - Couvre les suggestions 2 (coûts), 3 (microstructure), 5 (walk-forward), 7 (Safe RL), 8 (Distributional RL),
#   9 (volatilité), 10 (ensembles de politiques).
# - Vérifie les logs psutil, journalisation MLflow, métriques Prometheus, alertes Telegram,
#   et l’intégration de SignalResolver (résolution des signaux, métadonnées, gestion des erreurs).
# - Policies Note: The official directory for routing policies is src/model/router/policies.
# The src/model/policies directory is a residual and should be verified
# for removal to avoid import conflicts.

import os
from unittest.mock import MagicMock, patch

import gym
import numpy as np
import pandas as pd
import pytest
import yaml

from src.model.trade_probability_rl import CVaRWrapper, RLTrainer, TradeEnv


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs et modèles."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    rl_model_dir = data_dir / "models" / "ES"
    rl_model_dir.mkdir(parents=True)

    # Créer es_config.yaml
    es_config_path = config_dir / "es_config.yaml"
    es_config_content = {
        "signal_resolver": {
            "default_weights": {
                "microstructure_bullish": 1.0,
                "news_score_positive": 0.5,
                "qr_dqn_positive": 1.5,
                "iv_term_structure": 1.0,
            },
            "thresholds": {"entropy_alert": 0.5, "conflict_coefficient_alert": 0.5},
        }
    }
    with open(es_config_path, "w", encoding="utf-8") as f:
        yaml.dump(es_config_content, f)

    return {
        "base_dir": str(base_dir),
        "es_config_path": str(es_config_path),
        "logs_dir": str(logs_dir),
        "rl_model_dir": str(rl_model_dir),
        "rl_perf_log_path": str(logs_dir / "trade_probability_rl_performance.csv"),
    }


@pytest.fixture
def mock_data():
    """Crée des données factices pour tester."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=100, freq="1min"),
            "bid_ask_imbalance": np.random.normal(0, 0.1, 100),
            "trade_aggressiveness": np.random.normal(0, 0.2, 100),
            "iv_skew": np.random.normal(0.01, 0.005, 100),
            "iv_term_structure": np.random.normal(0.02, 0.005, 100),
            "slippage_estimate": np.random.uniform(0.01, 0.1, 100),
            "news_impact_score": np.random.uniform(0.0, 1.0, 100),
            "qr_dqn_quantile_mean": np.random.uniform(-0.5, 0.5, 100),
        }
    )


@pytest.fixture
def mock_env(mock_data):
    """Crée un environnement Gym factice."""
    return TradeEnv(mock_data, market="ES")


@pytest.fixture
def mock_mlflow_tracker():
    """Mock pour MLFlowTracker."""
    tracker = MagicMock()
    tracker.start_run = MagicMock()
    tracker.log_metrics = MagicMock()
    tracker.log_params = MagicMock()
    tracker.log_artifact = MagicMock()
    tracker.tracking_uri = "http://localhost:5000"
    return tracker


def test_rl_trainer_init(tmp_dirs, mock_mlflow_tracker):
    """Teste l’initialisation de RLTrainer."""
    with patch(
        "src.model.trade_probability_rl.MLFlowTracker", return_value=mock_mlflow_tracker
    ), patch("src.model.trade_probability_rl.send_telegram_alert"), patch(
        "src.model.trade_probability_rl.SignalResolver"
    ) as mock_resolver:
        mock_resolver.return_value = MagicMock()
        trainer = RLTrainer(market="ES")
        assert trainer.market == "ES", "Marché incorrect"
        assert set(trainer.rl_models.keys()) == {
            "sac",
            "ppo",
            "ddpg",
            "ppo_cvar",
            "qr_dqn",
        }, "Modèles RL incorrects"
        assert isinstance(
            trainer.mlflow_tracker, MagicMock
        ), "MLFlowTracker non initialisé"
        assert isinstance(
            trainer.alert_manager, MagicMock
        ), "AlertManager non initialisé"
        assert isinstance(
            trainer.signal_resolver, MagicMock
        ), "SignalResolver non initialisé"
        assert os.path.exists(
            tmp_dirs["rl_model_dir"]
        ), "Répertoire des modèles RL non créé"


def test_trade_env_init_and_step(mock_data):
    """Teste l’initialisation et le step de TradeEnv."""
    env = TradeEnv(mock_data, market="ES")
    assert env.features == [
        "bid_ask_imbalance",
        "trade_aggressiveness",
        "iv_skew",
        "iv_term_structure",
    ], "Features incorrectes"
    assert isinstance(
        env.action_space, gym.spaces.Discrete
    ), "Espace d’actions incorrect"
    assert env.observation_space.shape == (4,), "Espace d’observations incorrect"
    obs = env.reset()
    assert np.array_equal(
        obs, mock_data[env.features].iloc[0].values
    ), "Reset incorrect"
    obs, reward, done, info = env.step(1)
    assert isinstance(reward, float), "Récompense non numérique"
    assert (
        reward <= 1.0 - mock_data["slippage_estimate"].iloc[1]
    ), "Récompense sans slippage incorrecte"
    assert not done, "Done incorrect pour step initial"
    assert np.array_equal(
        obs, mock_data[env.features].iloc[1].values
    ), "Observation incorrecte après step"


def test_cvar_wrapper(mock_env):
    """Teste le wrapper CVaRWrapper pour PPO-Lagrangian."""
    with patch(
        "src.model.trade_probability_rl.PPO.__init__", return_value=None
    ) as mock_ppo, patch(
        "src.model.trade_probability_rl.PPO.learn"
    ) as mock_learn, patch(
        "src.model.trade_probability_rl.PPO.step"
    ) as mock_step, patch(
        "src.monitoring.prometheus_metrics.Gauge.labels"
    ) as mock_gauge:
        wrapper = CVaRWrapper("MlpPolicy", mock_env, cvar_alpha=0.95, verbose=0)
        assert wrapper.cvar_alpha == 0.95, "Alpha CVaR incorrect"
        wrapper.rewards = [-1.0, -0.5, 0.0, 0.5, 1.0]
        wrapper.learn(total_timesteps=10)
        expected_cvar = np.mean([-1.0, -0.5])  # 95% quantile
        assert (
            abs(wrapper.cvar_loss - expected_cvar) < 1e-6
        ), f"CVaR incorrect: attendu {expected_cvar}, obtenu {wrapper.cvar_loss}"
        mock_gauge.assert_called_with(market="ES")
        mock_gauge.return_value.set.assert_called_with(wrapper.cvar_loss)


def test_train_rl_models(tmp_dirs, mock_data, mock_env, mock_mlflow_tracker):
    """Teste l’entraînement des modèles RL avec validation glissante."""
    with patch(
        "src.model.trade_probability_rl.MLFlowTracker", return_value=mock_mlflow_tracker
    ), patch("src.model.trade_probability_rl.SAC") as mock_sac, patch(
        "src.model.trade_probability_rl.PPO"
    ) as mock_ppo, patch(
        "src.model.trade_probability_rl.DDPG"
    ) as mock_ddpg, patch(
        "src.model.trade_probability_rl.CVaRWrapper"
    ) as mock_cvar, patch(
        "src.model.trade_probability_rl.QRDQN"
    ) as mock_qrdqn, patch(
        "src.model.trade_probability_rl.send_telegram_alert"
    ) as mock_telegram, patch(
        "src.monitoring.prometheus_metrics.Gauge.labels"
    ) as mock_gauge, patch(
        "joblib.Parallel"
    ) as mock_parallel, patch(
        "src.model.trade_probability_rl.SignalResolver"
    ) as mock_resolver:
        mock_model = MagicMock()
        mock_model.learn = MagicMock()
        mock_model.save = MagicMock()
        mock_sac.return_value = mock_model
        mock_ppo.return_value = mock_model
        mock_ddpg.return_value = mock_model
        mock_cvar.return_value = mock_model
        mock_qrdqn.return_value = mock_model
        mock_parallel.return_value = [
            ("sac", mock_model),
            ("ppo", mock_model),
            ("ddpg", mock_model),
            ("ppo_cvar", mock_model),
            ("qr_dqn", mock_model),
        ]
        mock_resolver.return_value = MagicMock()
        trainer = RLTrainer(market="ES")
        trainer.train_rl_models(mock_data, total_timesteps=10)
        assert mock_parallel.called, "Parallélisation non utilisée"
        assert (
            mock_model.learn.call_count == 5
        ), "Entraînement RL non appelé pour tous les modèles"
        assert (
            mock_model.save.call_count == 5
        ), "Sauvegarde RL non appelée pour tous les modèles"
        assert os.path.exists(
            tmp_dirs["rl_perf_log_path"]
        ), "Fichier de logs RL non créé"
        df = pd.read_csv(tmp_dirs["rl_perf_log_path"])
        assert any(
            df["operation"].str.contains("train_")
        ), "Logs d’entraînement RL manquants"
        assert mock_gauge.call_count >= 2, "Métriques Prometheus non mises à jour"
        mock_telegram.assert_called()


def test_predict_rl(tmp_dirs, mock_data, mock_mlflow_tracker):
    """Teste la prédiction RL avec vote bayésien."""
    with patch(
        "src.model.trade_probability_rl.MLFlowTracker", return_value=mock_mlflow_tracker
    ), patch("src.model.trade_probability_rl.SAC") as mock_sac, patch(
        "src.model.trade_probability_rl.PPO"
    ) as mock_ppo, patch(
        "src.model.trade_probability_rl.DDPG"
    ) as mock_ddpg, patch(
        "src.model.trade_probability_rl.CVaRWrapper"
    ) as mock_cvar, patch(
        "src.model.trade_probability_rl.QRDQN"
    ) as mock_qrdqn, patch(
        "src.model.trade_probability_rl.send_telegram_alert"
    ) as mock_telegram, patch(
        "src.model.trade_probability_rl.SignalResolver"
    ) as mock_resolver:
        mock_model = MagicMock()
        mock_model.policy.predict_values = MagicMock(
            return_value=(None, np.array([1.0]))
        )
        mock_qrdqn_model = MagicMock()
        mock_qrdqn_model.predict_quantiles = MagicMock(
            return_value=np.array([0.5, -0.5])
        )
        mock_sac.return_value = mock_model
        mock_ppo.return_value = mock_model
        mock_ddpg.return_value = mock_model
        mock_cvar.return_value = mock_model
        mock_qrdqn.return_value = mock_qrdqn_model
        mock_resolver.return_value.resolve_conflict.return_value = (
            0.5,
            {
                "signal_score": 0.5,
                "normalized_score": 0.5,
                "entropy": 0.3,
                "conflict_coefficient": 0.4,
                "score_type": "intermediate",
                "contributions": {
                    "microstructure_bullish": {
                        "value": 1.0,
                        "weight": 1.0,
                        "contribution": 1.0,
                    }
                },
                "run_id": "test_run_123",
            },
        )
        trainer = RLTrainer(market="ES")
        trainer.rl_models = {
            "sac": mock_model,
            "ppo": mock_model,
            "ddpg": mock_model,
            "ppo_cvar": mock_model,
            "qr_dqn": mock_qrdqn_model,
        }
        data = mock_data.tail(1)
        prob, signal_metadata = trainer.predict(data)
        assert 0 <= prob <= 1, "Probabilité RL hors intervalle [0, 1]"
        expected_prob = np.mean(
            [
                1 / (1 + np.exp(-1.0)),  # SAC
                1 / (1 + np.exp(-1.0)),  # PPO
                1 / (1 + np.exp(-1.0)),  # DDPG
                1 / (1 + np.exp(-1.0)),  # PPO-CVaR
                0.5,  # QR-DQN (mean(quantiles > 0))
            ]
        )
        assert (
            abs(prob - expected_prob) < 1e-6
        ), f"Probabilité avec vote bayésien incorrecte: attendu {expected_prob}, obtenu {prob}"
        assert "signal_score" in signal_metadata, "Métadonnées manquent signal_score"
        assert (
            "conflict_coefficient" in signal_metadata
        ), "Métadonnées manquent conflict_coefficient"
        assert "run_id" in signal_metadata, "Métadonnées manquent run_id"
        assert os.path.exists(
            tmp_dirs["rl_perf_log_path"]
        ), "Fichier de logs RL non créé"
        df = pd.read_csv(tmp_dirs["rl_perf_log_path"])
        assert any(
            df["operation"].str.contains("predict")
        ), "Logs de prédiction RL manquants"
        mock_telegram.assert_called()


def test_load_models(tmp_dirs, mock_mlflow_tracker):
    """Teste le chargement des modèles RL."""
    with patch(
        "src.model.trade_probability_rl.MLFlowTracker", return_value=mock_mlflow_tracker
    ), patch("src.model.trade_probability_rl.SAC.load") as mock_sac_load, patch(
        "src.model.trade_probability_rl.PPO.load"
    ) as mock_ppo_load, patch(
        "src.model.trade_probability_rl.DDPG.load"
    ) as mock_ddpg_load, patch(
        "src.model.trade_probability_rl.CVaRWrapper.load"
    ) as mock_cvar_load, patch(
        "src.model.trade_probability_rl.QRDQN.load"
    ) as mock_qrdqn_load, patch(
        "src.model.trade_probability_rl.send_telegram_alert"
    ) as mock_telegram, patch(
        "src.model.trade_probability_rl.SignalResolver"
    ) as mock_resolver, patch(
        "os.listdir",
        return_value=[
            "sac_20250514_090000.pth",
            "ppo_20250514_090000.pth",
            "ddpg_20250514_090000.pth",
            "ppo_cvar_20250514_090000.pth",
            "qr_dqn_20250514_090000.pth",
        ],
    ), patch(
        "os.path.getctime", return_value=1
    ):
        mock_model = MagicMock()
        mock_sac_load.return_value = mock_model
        mock_ppo_load.return_value = mock_model
        mock_ddpg_load.return_value = mock_model
        mock_cvar_load.return_value = mock_model
        mock_qrdqn_load.return_value = mock_model
        mock_resolver.return_value = MagicMock()
        trainer = RLTrainer(market="ES")
        trainer.load_models()
        assert trainer.rl_models["sac"] == mock_model, "Modèle SAC non chargé"
        assert trainer.rl_models["ppo"] == mock_model, "Modèle PPO non chargé"
        assert trainer.rl_models["ddpg"] == mock_model, "Modèle DDPG non chargé"
        assert trainer.rl_models["ppo_cvar"] == mock_model, "Modèle PPO-CVaR non chargé"
        assert trainer.rl_models["qr_dqn"] == mock_model, "Modèle QR-DQN non chargé"
        assert os.path.exists(
            tmp_dirs["rl_perf_log_path"]
        ), "Fichier de logs RL non créé"
        mock_telegram.assert_called()


def test_resolve_signals(tmp_dirs, mock_data, mock_mlflow_tracker):
    """Teste la méthode resolve_signals."""
    with patch(
        "src.model.trade_probability_rl.MLFlowTracker", return_value=mock_mlflow_tracker
    ), patch("src.model.trade_probability_rl.SignalResolver") as mock_resolver, patch(
        "src.model.trade_probability_rl.send_telegram_alert"
    ) as mock_telegram:
        mock_resolver.return_value.resolve_conflict.return_value = (
            0.5,
            {
                "signal_score": 0.5,
                "normalized_score": 0.5,
                "entropy": 0.3,
                "conflict_coefficient": 0.4,
                "score_type": "intermediate",
                "contributions": {
                    "microstructure_bullish": {
                        "value": 1.0,
                        "weight": 1.0,
                        "contribution": 1.0,
                    }
                },
                "run_id": "test_run_123",
            },
        )
        trainer = RLTrainer(market="ES")
        data = mock_data.tail(1)
        metadata = trainer.resolve_signals(data)
        assert "signal_score" in metadata, "Métadonnées manquent signal_score"
        assert "normalized_score" in metadata, "Métadonnées manquent normalized_score"
        assert "entropy" in metadata, "Métadonnées manquent entropy"
        assert (
            "conflict_coefficient" in metadata
        ), "Métadonnées manquent conflict_coefficient"
        assert "run_id" in metadata, "Métadonnées manquent run_id"
        assert metadata["signal_score"] == 0.5, "Score incorrect"
        assert metadata["conflict_coefficient"] == 0.4, "Conflict coefficient incorrect"
        assert os.path.exists(
            tmp_dirs["rl_perf_log_path"]
        ), "Fichier de logs RL non créé"
        df = pd.read_csv(tmp_dirs["rl_perf_log_path"])
        assert any(
            df["operation"].str.contains("resolve_signals")
        ), "Logs de résolution des signaux manquants"
        mock_telegram.assert_called()


def test_resolve_signals_error(tmp_dirs, mock_mlflow_tracker):
    """Teste la gestion des erreurs dans resolve_signals."""
    with patch(
        "src.model.trade_probability_rl.MLFlowTracker", return_value=mock_mlflow_tracker
    ), patch("src.model.trade_probability_rl.SignalResolver") as mock_resolver, patch(
        "src.model.trade_probability_rl.send_telegram_alert"
    ) as mock_telegram:
        mock_resolver.return_value.resolve_conflict.side_effect = ValueError(
            "Signal invalide"
        )
        trainer = RLTrainer(market="ES")
        data = pd.DataFrame()  # DataFrame vide pour simuler une erreur
        metadata = trainer.resolve_signals(data)
        assert metadata["signal_score"] == 0.0, "Score par défaut incorrect"
        assert (
            metadata["conflict_coefficient"] == 0.0
        ), "Conflict coefficient par défaut incorrect"
        assert "error" in metadata, "Métadonnées manquent erreur"
        assert "run_id" in metadata, "Métadonnées manquent run_id"
        assert os.path.exists(
            tmp_dirs["rl_perf_log_path"]
        ), "Fichier de logs RL non créé"
        df = pd.read_csv(tmp_dirs["rl_perf_log_path"])
        assert any(
            df["operation"].str.contains("resolve_signals")
        ), "Logs de résolution des signaux manquants"
        mock_telegram.assert_called()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
