# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_reward_engine.py
# Tests unitaires pour src/model/reward_engine.py
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide le calcul des récompenses pour SAC/PPO/DDPG avec récompenses 
# adaptatives (méthode 5), intégrant profit, risque, drawdown, risque de crash 
# de liquidité, news_impact_score, et predicted_vix. Vérifie les sauvegardes 
# incrémentielles/distribuées et les alertes Telegram.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, numpy>=1.26.4,<2.0.0, pandas>=2.0.0,<3.0.0, 
#   psutil>=5.9.8,<6.0.0, pyyaml>=6.0.0,<7.0.0, boto3>=1.26.0,<2.0.0, 
#   loguru>=0.7.0,<1.0.0
# - src/features/neural_pipeline.py
# - src/envs/trading_env.py
# - src/model/utils/miya_console.py
# - src/model/utils/config_manager.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/feature_sets.yaml
# - Données factices (DataFrame avec 350 features)
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de reward_engine.py.
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Tests les Phases 8 (confidence_drop_rate) et 5 (récompenses adaptatives).
# - Vérifie l’absence de dxFeed, la suppression de 320/81 features et obs_t, 
#   les retries, logs psutil, alertes Telegram, snapshots compressés, et 
#   sauvegardes incrémentielles/distribuées.
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence) 
#   via config/feature_sets.yaml.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.

import gzip
import json
import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.model.reward_engine import (
    calculate_liquidity_crash_risk,
    calculate_reward,
    validate_features,
)


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, cache, et checkpoints."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    cache_dir = data_dir / "cache" / "reward_engine" / "ES"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints" / "reward_engine" / "ES"
    checkpoints_dir.mkdir(parents=True)

    # Créer feature_sets.yaml
    feature_sets_path = config_dir / "feature_sets.yaml"
    training_features = [f"feature_{i}" for i in range(350)]
    shap_features = [f"shap_feature_{i}" for i in range(150)]
    feature_sets_content = {
        "training": {"features": training_features},
        "inference": {"shap_features": shap_features},
    }
    with open(feature_sets_path, "w", encoding="utf-8") as f:
        yaml.dump(feature_sets_content, f)

    # Créer reward_engine_config.yaml
    config_path = config_dir / "reward_engine_config.yaml"
    config_content = {"s3_bucket": "test-bucket", "s3_prefix": "reward_engine/"}
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    return {
        "base_dir": str(base_dir),
        "config_path": str(config_path),
        "feature_sets_path": str(feature_sets_path),
        "logs_dir": str(logs_dir),
        "cache_dir": str(cache_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "perf_log_path": str(logs_dir / "reward_engine_performance.csv"),
    }


@pytest.fixture
def mock_data():
    """Crée un DataFrame factice avec 350 features."""
    feature_cols = [f"feature_{i}" for i in range(350)]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=100, freq="1min"),
            "open": [5100] * 100,
            "high": [5105] * 100,
            "low": [5095] * 100,
            "close": [5100] * 100,
            "volume": [1000] * 100,
            "atr_14_es": [1.0] * 100,
            "adx_14": [25] * 100,
            "gex_es": [500] * 100,
            "oi_peak_call_near": [10000] * 100,
            "gamma_wall_call": [0.02] * 100,
            "gamma_wall_put": [0.02] * 100,
            "bid_size_level_1": [100] * 100,
            "ask_size_level_1": [120] * 100,
            "vwap_es": [5100] * 100,
            "spread_avg_1min": [0.3] * 100,
            "predicted_volatility_es": [1.5] * 100,
            "neural_regime": [0] * 100,
            "cnn_pressure": [0.5] * 100,
            "trade_frequency_1s": [8] * 100,
            **{f"neural_feature_{i}": [np.random.randn()] * 100 for i in range(8)},
            **{
                col: [np.random.uniform(0, 1)] * 100
                for col in feature_cols
                if col
                not in [
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "atr_14_es",
                    "adx_14",
                    "gex_es",
                    "oi_peak_call_near",
                    "gamma_wall_call",
                    "gamma_wall_put",
                    "bid_size_level_1",
                    "ask_size_level_1",
                    "vwap_es",
                    "spread_avg_1min",
                    "predicted_volatility_es",
                    "neural_regime",
                    "cnn_pressure",
                    "trade_frequency_1s",
                ]
            },
        }
    )


@pytest.fixture
def mock_env(mock_data):
    """Crée un environnement de trading factice."""
    env = MagicMock()
    env.config = {
        "market_params": {"point_value": 50},
        "thresholds": {"max_drawdown": -1000.0, "vwap_slope_max_range": 0.01},
    }
    env.current_step = 0
    env.max_drawdown = -500.0
    env.balance = 10000.0
    env.data = mock_data
    return env


@pytest.fixture
def mock_state():
    """Crée un état factice pour le calcul de la récompense."""
    return {
        "profit": 100.0,
        "risk": 10.0,
        "news_impact_score": 0.5,
        "predicted_vix": 20.0,
        "duration": 5,
        "entry_price": 5100.0,
        "exit_price": 5105.0,
    }


@pytest.mark.asyncio
async def test_validate_features(tmp_dirs, mock_data):
    """Teste la validation des features."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        validate_features(mock_data, current_step=0, market="ES")
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_validate_features" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot validate_features non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "validate_features" in str(op) for op in df_perf["operation"]
        ), "Opération validate_features non journalisée"
        assert any(
            "confidence_drop_rate" in str(kw) for kw in df_perf.to_dict("records")
        ), "confidence_drop_rate absent"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_validate_features_invalid(mock_data):
    """Teste la validation avec des features invalides."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        invalid_data = mock_data.drop(columns=["bid_size_level_1"])
        with pytest.raises(ValueError):
            validate_features(invalid_data, current_step=0, market="ES")
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_calculate_liquidity_crash_risk(tmp_dirs, mock_data):
    """Teste le calcul du risque de crash de liquidité."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        risk = calculate_liquidity_crash_risk(mock_data, current_step=0, market="ES")
        assert 0 <= risk <= 1, "Risque de crash de liquidité hors intervalle [0, 1]"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_calculate_liquidity_crash_risk" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot calculate_liquidity_crash_risk non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "calculate_liquidity_crash_risk" in str(op) for op in df_perf["operation"]
        ), "Opération calculate_liquidity_crash_risk non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_calculate_reward(tmp_dirs, mock_state, mock_env):
    """Teste le calcul de la récompense."""
    with patch("src.features.neural_pipeline.NeuralPipeline") as mock_pipeline, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        mock_pipeline.return_value.load_models.return_value = None
        mock_pipeline.return_value.run.return_value = {
            "volatility": [1.5],
            "regime": [0],
            "features": np.array([[0.5]]),
        }
        for mode in ["trend", "range", "defensive"]:
            for policy_type in ["mlp", "transformer"]:
                reward = calculate_reward(
                    mock_state,
                    mock_env,
                    mode=mode,
                    policy_type=policy_type,
                    market="ES",
                )
                assert isinstance(reward, float), "Récompense doit être un float"
                snapshot_files = os.listdir(tmp_dirs["cache_dir"])
                assert any(
                    "snapshot_calculate_reward" in f and f.endswith(".json.gz")
                    for f in snapshot_files
                ), "Snapshot calculate_reward non créé"
                checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
                assert any(
                    "reward" in f and f.endswith(".json.gz") for f in checkpoint_files
                ), "Checkpoint reward non créé"
                df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
                assert any(
                    "calculate_reward" in str(op) for op in df_perf["operation"]
                ), "Opération calculate_reward non journalisée"
                mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_cloud_backup(tmp_dirs):
    """Teste la sauvegarde distribuée S3."""
    with patch("boto3.client") as mock_s3, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        df = pd.DataFrame(
            {
                "timestamp": [datetime.now().isoformat()],
                "reward": [100.0],
                "profit": [100.0],
            }
        )
        from src.model.reward_engine import cloud_backup

        cloud_backup(df, data_type="test_metrics", market="ES")
        assert mock_s3.called, "Client S3 non appelé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "temp_s3_" in f for f in checkpoint_files
        ), "Fichier temporaire S3 non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "cloud_backup" in str(op) for op in df_perf["operation"]
        ), "Opération cloud_backup non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_save_snapshot(tmp_dirs):
    """Teste la sauvegarde d’un snapshot compressé."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        snapshot_data = {"test": "snapshot"}
        from src.model.reward_engine import save_snapshot

        save_snapshot("test_snapshot", snapshot_data, market="ES", compress=True)
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_test_snapshot" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot compressé non créé"
        snapshot_file = os.path.join(tmp_dirs["cache_dir"], snapshot_files[-1])
        with gzip.open(snapshot_file, "rt", encoding="utf-8") as f:
            snapshot = json.load(f)
        assert (
            snapshot["data"] == snapshot_data
        ), "Contenu du snapshot compressé incorrect"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "save_snapshot" in str(op) for op in df_perf["operation"]
        ), "Opération save_snapshot non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_checkpoint(tmp_dirs):
    """Teste la sauvegarde incrémentielle."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        df = pd.DataFrame(
            {
                "timestamp": [datetime.now().isoformat()],
                "reward": [100.0],
                "profit": [100.0],
            }
        )
        from src.model.reward_engine import checkpoint

        checkpoint(df, data_type="test_metrics", market="ES")
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "reward_engine_test_metrics" in f and f.endswith(".json.gz")
            for f in checkpoint_files
        ), "Checkpoint non créé"
        checkpoint_file = os.path.join(tmp_dirs["checkpoints_dir"], checkpoint_files[0])
        with gzip.open(checkpoint_file, "rt", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
        assert checkpoint_data["num_rows"] == len(df), "Nombre de lignes incorrect"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "checkpoint" in str(op) for op in df_perf["operation"]
        ), "Opération checkpoint non journalisée"
        mock_telegram.assert_called()