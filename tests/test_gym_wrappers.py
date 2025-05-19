# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_gym_wrappers.py
# Tests unitaires pour src/envs/gym_wrappers.py
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Valide les wrappers Gym pour l’environnement de trading ES,
# incluant l’empilement, la normalisation, le clipping, les régimes
# (range, trend, défensif), les snapshots JSON compressés, les logs
# psutil, et les alertes.
# Conforme à la Phase 8 (auto-conscience via alertes), Phase 12
# (simulation de trading), et Phase 16 (ensemble learning).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pandas>=2.0.0,<3.0.0
# - numpy>=1.26.4,<2.0.0
# - psutil>=5.9.8,<6.0.0
# - pyyaml>=6.0.0,<7.0.0
# - gymnasium>=0.26.0,<1.0.0
# - src.envs.gym_wrappers
# - src.envs.trading_env
# - src.features.neural_pipeline
# - src.model.utils.config_manager
# - src.model.utils.alert_manager
# - src.utils.telegram_alert
# - src.features.detect_regime
#
# Inputs :
# - config/trading_env_config.yaml (simulé)
# - config/feature_sets.yaml (simulé)
# - config/model_params.yaml (simulé)
# - data/iqfeed/iqfeed_data.csv (simulé)
#
# Outputs :
# - Assertions sur l’état des wrappers
# - data/logs/gym_wrappers_performance.csv (simulé)
# - data/logs/normalization/norm_mean.npy (simulé)
# - data/logs/normalization/norm_std.npy (simulé)
# - data/logs/wrapper_sigint_*.json.gz (simulé)
#
# Notes :
# - Utilise des mocks pour simuler neural_pipeline, alert_manager,
#   telegram_alert, et detect_regime.
# - Vérifie l'absence de références à dxFeed, obs_t, 320/81 features.

from datetime import datetime
import gzip
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.envs.gym_wrappers import (
    BaseEnvWrapper,
    ClippingWrapper,
    DefensiveEnvWrapper,
    NormalizationWrapper,
    ObservationStackingWrapper,
    RangeEnvWrapper,
    TrendEnvWrapper,
)
from src.envs.trading_env import TradingEnv

BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, normalisation,
    données IQFeed, et configurations."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    norm_dir = logs_dir / "normalization"
    norm_dir.mkdir()
    iqfeed_dir = data_dir / "iqfeed"
    iqfeed_dir.mkdir()
    models_dir = data_dir / "models"
    models_dir.mkdir()

    # Créer trading_env_config.yaml
    config_path = config_dir / "trading_env_config.yaml"
    config_content = {
        "observation": {"sequence_length": 50},
        "environment": {
            "initial_cash": 100000.0,
            "max_position_size": 5,
            "point_value": 50,
            "transaction_cost": 2.0,
            "max_leverage": 5.0,
            "atr_limit": 2.0,
            "max_trade_duration": 20,
            "min_balance": 90000.0,
            "max_drawdown": -10000.0,
            "call_wall_distance": 0.01,
            "zero_gamma_distance": 0.005,
            "reward_threshold": 0.01,
            "training_mode": True,
        },
        "stacking": {"stack_size": 3},
        "normalization": {"min_std": 1e-8, "alpha": 0.01},
        "clipping": {
            "clip_min": -5.0,
            "clip_max": 5.0,
            "option_columns": {
                "iv_rank_30d": {"min": 0, "max": 100},
                "call_wall": {"min": 0, "max": float("inf")},
                "put_wall": {"min": 0, "max": float("inf")},
                "zero_gamma": {"min": 0, "max": float("inf")},
                "dealer_position_bias": {"min": -1, "max": 1},
            },
        },
        "logging": {"buffer_size": 100, "directory": "data/logs"},
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    # Créer feature_sets.yaml
    feature_sets_path = config_dir / "feature_sets.yaml"
    feature_sets_content = {
        "training_features": [
            "timestamp",
            "close",
            "open",
            "high",
            "low",
            "volume",
            "atr_14",
            "adx_14",
            "gex",
            "oi_peak_call_near",
            "gamma_wall",
            "iv_rank_30d",
            "call_wall",
            "put_wall",
            "zero_gamma",
            "dealer_position_bias",
            "predicted_volatility",
            "predicted_vix",
            "news_impact_score",
            "bid_size_level_1",
            "ask_size_level_1",
            "spread_avg_1min",
        ] + [f"feat_{i}" for i in range(328)],
        "shap_features": [
            "close",
            "atr_14",
            "adx_14",
            "iv_rank_30d",
            "call_wall",
            "put_wall",
            "zero_gamma",
            "dealer_position_bias",
            "predicted_volatility",
            "predicted_vix",
            "news_impact_score",
            "bid_size_level_1",
            "ask_size_level_1",
            "spread_avg_1min",
        ] + [f"feat_{i}" for i in range(136)],
    }
    with open(feature_sets_path, "w", encoding="utf-8") as f:
        yaml.dump(feature_sets_content, f)

    # Créer model_params.yaml
    model_params_path = config_dir / "model_params.yaml"
    model_params_content = {
        "lstm_units": 64,
        "cnn_filters": 32,
    }
    with open(model_params_path, "w", encoding="utf-8") as f:
        yaml.dump(model_params_content, f)

    # Créer iqfeed_data.csv
    data_path = iqfeed_dir / "iqfeed_data.csv"
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-05-13 09:00",
                periods=100,
                freq="1min",
            ),
            "close": np.random.normal(5100, 10, 100),
            "open": np.random.normal(5100, 10, 100),
            "high": np.random.normal(5105, 10, 100),
            "low": np.random.normal(5095, 10, 100),
            "volume": np.random.randint(100, 1000, 100),
            "atr_14": np.random.uniform(0.5, 2.0, 100),
            "adx_14": np.random.uniform(10, 40, 100),
            "gex": np.random.uniform(-1000, 1000, 100),
            "oi_peak_call_near": np.random.randint(
                5000,
                15000,
                100,
            ),
            "gamma_wall": np.random.uniform(5090, 5110, 100),
            "iv_rank_30d": np.random.uniform(50, 80, 100),
            "call_wall": np.random.uniform(5150, 5200, 100),
            "put_wall": np.random.uniform(5050, 5100, 100),
            "zero_gamma": np.random.uniform(5095, 5105, 100),
            "dealer_position_bias": np.random.uniform(
                -0.2,
                0.2,
                100,
            ),
            "predicted_volatility": np.random.uniform(0.1, 0.5, 100),
            "predicted_vix": np.random.uniform(15, 25, 100),
            "news_impact_score": np.random.uniform(-1, 1, 100),
            "bid_size_level_1": np.random.randint(50, 500, 100),
            "ask_size_level_1": np.random.randint(50, 500, 100),
            "spread_avg_1min": np.random.uniform(1.0, 3.0, 100),
            **{
                f"feat_{i}": np.random.uniform(0, 1, 100)
                for i in range(328)
            },
        }
    )
    data.to_csv(data_path, index=False, encoding="utf-8")

    # Créer regime_model.pkl (fichier fictif)
    regime_model_path = models_dir / "regime_model.pkl"
    with open(regime_model_path, "wb") as f:
        f.write(b"")

    return {
        "config_path": str(config_path),
        "feature_sets_path": str(feature_sets_path),
        "model_params_path": str(model_params_path),
        "data_path": str(data_path),
        "logs_dir": str(logs_dir),
        "norm_dir": str(norm_dir),
        "perf_log_path": str(
            logs_dir / "gym_wrappers_performance.csv"
        ),
        "regime_model_path": str(regime_model_path),
    }


@pytest.fixture
def env(tmp_dirs, monkeypatch):
    """Initialise TradingEnv avec des mocks."""
    def mock_get_config(x):
        if "trading_env_config.yaml" in str(x):
            return {
                "environment": {
                    "initial_cash": 100000.0,
                    "max_position_size": 5,
                    "point_value": 50,
                    "transaction_cost": 2.0,
                    "max_leverage": 5.0,
                    "atr_limit": 2.0,
                    "max_trade_duration": 20,
                    "min_balance": 90000.0,
                    "max_drawdown": -10000.0,
                    "call_wall_distance": 0.01,
                    "zero_gamma_distance": 0.005,
                    "reward_threshold": 0.01,
                    "training_mode": True,
                },
                "observation": {"sequence_length": 50},
                "logging": {"buffer_size": 100},
            }
        elif "feature_sets.yaml" in str(x):
            return {
                "training_features": [
                    "timestamp",
                    "close",
                    "open",
                    "high",
                    "low",
                    "volume",
                    "atr_14",
                    "adx_14",
                    "gex",
                    "oi_peak_call_near",
                    "gamma_wall",
                    "iv_rank_30d",
                    "call_wall",
                    "put_wall",
                    "zero_gamma",
                    "dealer_position_bias",
                    "predicted_volatility",
                    "predicted_vix",
                    "news_impact_score",
                    "bid_size_level_1",
                    "ask_size_level_1",
                    "spread_avg_1min",
                ] + [f"feat_{i}" for i in range(328)],
                "shap_features": [
                    "close",
                    "atr_14",
                    "adx_14",
                    "iv_rank_30d",
                    "call_wall",
                    "put_wall",
                    "zero_gamma",
                    "dealer_position_bias",
                    "predicted_volatility",
                    "predicted_vix",
                    "news_impact_score",
                    "bid_size_level_1",
                    "ask_size_level_1",
                    "spread_avg_1min",
                ] + [f"feat_{i}" for i in range(136)],
            }
        else:
            return {"lstm_units": 64, "cnn_filters": 32}

    monkeypatch.setattr(
        "src.envs.trading_env.config_manager.get_config",
        mock_get_config,
    )
    with patch(
        "src.features.neural_pipeline.NeuralPipeline.__init__",
        return_value=None,
    ), patch(
        "src.features.neural_pipeline.NeuralPipeline.load_models",
        return_value=None,
    ), patch(
        "src.features.neural_pipeline.NeuralPipeline.run",
        return_value={
            "features": np.random.uniform(0, 1, (50, 10)),
            "regime": np.random.randint(0, 3, 50),
        },
    ), patch(
        "src.model.utils.alert_manager.AlertManager.send_alert",
        return_value=None,
    ), patch(
        "src.utils.telegram_alert.send_telegram_alert",
        return_value=None,
    ):
        env = TradingEnv(str(tmp_dirs["config_path"]))
    env.data = pd.read_csv(
        tmp_dirs["data_path"],
        parse_dates=["timestamp"],
    )
    return env


def test_base_env_wrapper_init(tmp_dirs, env):
    """Teste l’initialisation de BaseEnvWrapper."""
    wrapper = BaseEnvWrapper(env, config_path=tmp_dirs["config_path"])
    assert wrapper.config["logging"]["buffer_size"] == 100
    assert Path(tmp_dirs["perf_log_path"]).exists()
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert "cpu_percent" in df.columns
    assert "memory_mb" in df.columns


def test_observation_stacking_wrapper_init(tmp_dirs, env):
    """Teste l’initialisation de ObservationStackingWrapper."""
    wrapper = ObservationStackingWrapper(
        env,
        stack_size=3,
        config_path=tmp_dirs["config_path"],
        policy_type="mlp",
    )
    assert wrapper.stack_size == 3
    assert wrapper.base_features == 350
    assert wrapper.observation_space.shape == (350 * 3,)
    wrapper = ObservationStackingWrapper(
        env,
        stack_size=3,
        config_path=tmp_dirs["config_path"],
        policy_type="transformer",
    )
    assert wrapper.observation_space.shape == (3, 350)
    assert Path(tmp_dirs["perf_log_path"]).exists()


def test_observation_stacking_wrapper_reset(tmp_dirs, env):
    """Teste la réinitialisation de ObservationStackingWrapper."""
    wrapper = ObservationStackingWrapper(
        env,
        stack_size=3,
        config_path=tmp_dirs["config_path"],
        policy_type="mlp",
    )
    obs, info = wrapper.reset(
        options={"data_path": tmp_dirs["data_path"]}
    )
    assert obs.shape == (350 * 3,)
    assert np.all(obs[: 350 * 2] == 0)  # Vérifie le padding
    assert Path(tmp_dirs["perf_log_path"]).exists()


def test_observation_stacking_wrapper_observation(tmp_dirs, env):
    """Teste le traitement des observations dans
    ObservationStackingWrapper."""
    wrapper = ObservationStackingWrapper(
        env,
        stack_size=3,
        config_path=tmp_dirs["config_path"],
        policy_type="mlp",
    )
    wrapper.reset(options={"data_path": tmp_dirs["data_path"]})
    obs = np.random.uniform(0, 1, 350)
    stacked_obs = wrapper.observation(obs)
    assert stacked_obs.shape == (350 * 3,)
    assert np.all(stacked_obs[350 * 2 :] == obs)


def test_normalization_wrapper_init(tmp_dirs, env):
    """Teste l’initialisation de NormalizationWrapper."""
    wrapper = NormalizationWrapper(
        env,
        config_path=tmp_dirs["config_path"],
        policy_type="mlp",
    )
    assert wrapper.min_std == 1e-8
    assert wrapper.alpha == 0.01
    assert wrapper.observation_space.shape == (350,)
    assert Path(tmp_dirs["perf_log_path"]).exists()


def test_normalization_wrapper_update_statistics(tmp_dirs, env):
    """Teste la mise à jour des statistiques dans NormalizationWrapper."""
    wrapper = NormalizationWrapper(
        env,
        config_path=tmp_dirs["config_path"],
        policy_type="mlp",
    )
    data = pd.read_csv(tmp_dirs["data_path"])
    wrapper.update_statistics(data)
    assert Path(tmp_dirs["norm_dir"] / "norm_mean.npy").exists()
    assert Path(tmp_dirs["norm_dir"] / "norm_std.npy").exists()
    mean = np.load(tmp_dirs["norm_dir"] / "norm_mean.npy")
    assert mean.shape == (350,)
    assert Path(tmp_dirs["perf_log_path"]).exists()


def test_normalization_wrapper_observation(tmp_dirs, env):
    """Teste la normalisation des observations dans
    NormalizationWrapper."""
    wrapper = NormalizationWrapper(
        env,
        config_path=tmp_dirs["config_path"],
        policy_type="mlp",
    )
    data = pd.read_csv(tmp_dirs["data_path"])
    wrapper.update_statistics(data)
    obs = data.iloc[0][wrapper.env.data.columns[:350]].values.astype(
        np.float32
    )
    norm_obs = wrapper.observation(obs)
    assert norm_obs.shape == (350,)
    assert np.all(np.isfinite(norm_obs))


def test_clipping_wrapper_init(tmp_dirs, env):
    """Teste l’initialisation de ClippingWrapper."""
    wrapper = ClippingWrapper(
        env,
        config_path=tmp_dirs["config_path"],
        policy_type="mlp",
    )
    assert wrapper.clip_min == -5.0
    assert wrapper.clip_max == 5.0
    assert wrapper.observation_space.shape == (350,)
    assert Path(tmp_dirs["perf_log_path"]).exists()


def test_clipping_wrapper_observation(tmp_dirs, env):
    """Teste le clipping des observations dans ClippingWrapper."""
    wrapper = ClippingWrapper(
        env,
        clip_min=-2.0,
        clip_max=2.0,
        config_path=tmp_dirs["config_path"],
        policy_type="mlp",
    )
    obs = np.array([10.0, -10.0, 1.0] + [0.0] * 347).astype(
        np.float32
    )
    clipped_obs = wrapper.observation(obs)
    assert clipped_obs.shape == (350,)
    assert np.all(clipped_obs <= 2.0)
    assert np.all(clipped_obs >= -2.0)


def test_regime_wrappers_init(tmp_dirs, env):
    """Teste l’initialisation des wrappers de régime."""
    regime_probs = {"range": 0.6, "trend": 0.3, "defensive": 0.1}
    range_wrapper = RangeEnvWrapper(
        env,
        regime_probs,
        config_path=tmp_dirs["config_path"],
    )
    trend_wrapper = TrendEnvWrapper(
        env,
        regime_probs,
        config_path=tmp_dirs["config_path"],
    )
    defensive_wrapper = DefensiveEnvWrapper(
        env,
        regime_probs,
        config_path=tmp_dirs["config_path"],
    )
    assert range_wrapper.weights == 0.6
    assert trend_wrapper.weights == 0.3
    assert defensive_wrapper.weights == 0.1
    assert Path(tmp_dirs["perf_log_path"]).exists()


def test_regime_wrappers_step(tmp_dirs, env):
    """Teste les ajustements des récompenses dans les wrappers de
    régime."""
    regime_probs = {"range": 0.6, "trend": 0.3, "defensive": 0.1}
    env.mode = "range"
    env.reset(options={"data_path": tmp_dirs["data_path"]})
    action = np.array([0.6])
    env.step(action)  # Créer un trade
    info = {
        "price": 5100.0,
        "call_wall": 5150.0,
        "put_wall": 5050.0,
        "profit": 100.0,
    }

    range_wrapper = RangeEnvWrapper(
        env,
        regime_probs,
        config_path=tmp_dirs["config_path"],
    )
    with patch.object(
        env,
        "step",
        return_value=(np.zeros(350), 1.0, False, False, info),
    ):
        obs, reward, done, truncated, info = range_wrapper.step(action)
    assert reward == 1.0 * 0.6 * 1.5  # Dans la plage call_wall/put_wall

    trend_wrapper = TrendEnvWrapper(
        env,
        regime_probs,
        config_path=tmp_dirs["config_path"],
    )
    with patch.object(
        env,
        "step",
        return_value=(np.zeros(350), 1.0, False, False, info),
    ):
        obs, reward, done, truncated, info = trend_wrapper.step(action)
    assert reward == 1.0 * 0.3 * 1.5  # Profit positif

    defensive_wrapper = DefensiveEnvWrapper(
        env,
        regime_probs,
        config_path=tmp_dirs["config_path"],
    )
    with patch.object(
        env,
        "step",
        return_value=(np.zeros(350), 1.0, False, False, {"profit": -100.0}),
    ):
        obs, reward, done, truncated, info = defensive_wrapper.step(
            action
        )
    assert reward == 1.0 * 0.1 * 0.5  # Profit négatif


def test_handle_sigint(tmp_dirs, env):
    """Teste la gestion de SIGINT dans BaseEnvWrapper."""
    wrapper = BaseEnvWrapper(env, config_path=tmp_dirs["config_path"])
    with patch("sys.exit") as mock_exit:
        wrapper.handle_sigint(None, None)
    snapshots = list(
        Path(tmp_dirs["logs_dir"]).glob("wrapper_sigint_*.json.gz")
    )
    assert len(snapshots) == 1
    with gzip.open(snapshots[0], "rt") as f:
        snapshot = json.load(f)
    assert snapshot["data"]["status"] == "SIGINT"
    mock_exit.assert_called_with(0)


def test_alerts(tmp_dirs, env):
    """Teste les alertes via alert_manager et telegram_alert."""
    wrapper = ObservationStackingWrapper(
        env,
        config_path=tmp_dirs["config_path"],
        policy_type="mlp",
    )
    with patch(
        "src.model.utils.alert_manager.AlertManager.send_alert"
    ) as mock_alert, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        wrapper.reset(options={"data_path": tmp_dirs["data_path"]})
    mock_alert.assert_called()
    mock_telegram.assert_called()


def test_no_obsolete_references(tmp_dirs):
    """Vérifie l'absence de références à dxFeed, obs_t, 320/81
    features."""
    for file_path in [
        tmp_dirs["config_path"],
        tmp_dirs["feature_sets_path"],
        tmp_dirs["model_params_path"],
    ]:
        with open(file_path, "r") as f:
            content = f.read()
        assert (
            "dxFeed" not in content
        ), f"Référence à dxFeed trouvée dans {file_path}"
        assert (
            "obs_t" not in content
        ), f"Référence à obs_t trouvée dans {file_path}"
        assert (
            "320 features" not in content
        ), f"Référence à 320 features trouvée dans {file_path}"
        assert (
            "81 features" not in content
        ), f"Référence à 81 features trouvée dans {file_path}"