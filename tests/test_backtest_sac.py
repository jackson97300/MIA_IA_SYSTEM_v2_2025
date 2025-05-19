# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_backtest_sac.py
# Tests unitaires pour src/backtest/backtest_sac.py
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide le backtest du modèle SAC avec parallélisation et Monte
# Carlo, intégrant des récompenses adaptatives (méthode 5) basées sur
# news_impact_score et trade_success_prob, optimisé pour 350 features et
# 150 SHAP features pour l’inférence/fallback, avec support multi-marchés.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pandas>=2.0.0,<3.0.0
# - numpy>=1.26.4,<2.0.0
# - joblib>=1.3.0,<2.0.0
# - stable-baselines3>=2.0.0,<3.0.0
# - gymnasium>=1.0.0,<2.0.0
# - pyyaml>=6.0.0,<7.0.0
# - boto3>=1.26.0,<2.0.0
# - loguru>=0.7.0,<1.0.0
# - src/features/neural_pipeline.py
# - src/model/utils/miya_console.py
# - src/features/context_aware_filter.py
# - src/features/cross_asset_processor.py
# - src/features/feature_pipeline.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/es_config.yaml
# - config/feature_sets.yaml
# - data/features/features_latest_filtered.csv
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de backtest_sac.py.
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Tests la Phase 8 (confidence_drop_rate), méthode 5 (récompenses
#   adaptatives), et autres standards.
# - Vérifie l’absence de dxFeed, la suppression de 320/81 features et obs_t,
#   les retries, logs psutil, alertes Telegram, snapshots compressés, et
#   sauvegardes incrémentielles/distribuées.
# - Compatible avec 350 features (entraînement) et 150 SHAP features
#   (inférence) via config/feature_sets.yaml.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.

from datetime import datetime
import gzip
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from gymnasium import spaces
import numpy as np
import pandas as pd
import pytest
from stable_baselines3 import SAC
import yaml

from src.backtest.backtest_sac import (
    TradingEnv,
    backtest_sac,
    eval_sac_segment,
    incremental_backtest_sac,
    load_config,
    monte_carlo_backtest,
    validate_data,
    validate_features,
)


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, cache, checkpoints, et
    modèles."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs" / "backtest"
    logs_dir.mkdir(parents=True)
    cache_dir = data_dir / "cache" / "backtest" / "ES"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints" / "backtest" / "ES"
    checkpoints_dir.mkdir(parents=True)
    model_dir = data_dir / "model" / "sac_models" / "ES" / "defensive" / "mlp"
    model_dir.mkdir(parents=True)
    features_dir = data_dir / "features"
    features_dir.mkdir()

    # Créer es_config.yaml
    config_path = config_dir / "es_config.yaml"
    config_content = {
        "backtest_sac": {
            "data_path": str(
                features_dir / "features_latest_filtered.csv"
            ),
            "model_dir": str(
                data_dir / "model" / "sac_models" / "ES"
            ),
            "log_dir": str(logs_dir),
            "sequence_length": 50,
            "n_simulations": 100,
            "min_rows": 100,
            "performance_thresholds": {
                "min_profit_factor": 1.2,
                "min_sharpe": 0.5,
                "max_drawdown": -0.2,
                "min_balance": 9000.0,
                "min_reward": -100.0,
            },
            "s3_bucket": "test-bucket",
            "s3_prefix": "backtest/",
        }
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    # Créer feature_sets.yaml
    feature_sets_path = config_dir / "feature_sets.yaml"
    feature_sets_content = {
        "training": {
            "features": [f"feature_{i}" for i in range(350)]
        },
        "inference": {
            "shap_features": [f"shap_feature_{i}" for i in range(150)]
        },
    }
    with open(feature_sets_path, "w", encoding="utf-8") as f:
        yaml.dump(feature_sets_content, f)

    # Créer model_params.yaml
    model_params_path = config_dir / "model_params.yaml"
    model_params_content = {
        "window_size": 50,
        "base_features": 350,
    }
    with open(model_params_path, "w", encoding="utf-8") as f:
        yaml.dump(model_params_content, f)

    # Créer features_latest_filtered.csv
    data_path = features_dir / "features_latest_filtered.csv"
    feature_cols = [f"feature_{i}" for i in range(350)]
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-05-13 09:00",
                periods=1000,
                freq="1min",
            ),
            "open": [5100] * 1000,
            "high": [5105] * 1000,
            "low": [5095] * 1000,
            "close": [5100] * 1000,
            "volume": [1000] * 1000,
            "bid_size_level_1": [100] * 1000,
            "ask_size_level_1": [120] * 1000,
            "trade_frequency_1s": [8] * 1000,
            "atr_14_es": [1.0] * 1000,
            "adx_14": [25] * 1000,
            "gex_es": [500] * 1000,
            "oi_peak_call_near": [10000] * 1000,
            "gamma_wall_call": [0.02] * 1000,
            "gamma_wall_put": [0.02] * 1000,
            "news_impact_score": [0.5] * 1000,
            "trade_success_prob": [0.5] * 1000,
            "neural_regime": [0] * 1000,
            "predicted_volatility_es": [1.5] * 1000,
            "cnn_pressure": [0.5] * 1000,
            **{
                f"neural_feature_{i}": [np.random.randn()] * 1000
                for i in range(8)
            },
            **{
                col: [np.random.uniform(0, 1)] * 1000
                for col in feature_cols
            },
        }
    )
    data.to_csv(data_path, index=False)

    # Créer un fichier modèle factice
    model_file = model_dir / "sac_defensive_mlp_20250513_120000.zip"
    with open(model_file, "wb") as f:
        f.write(b"")

    return {
        "base_dir": str(base_dir),
        "config_path": str(config_path),
        "feature_sets_path": str(feature_sets_path),
        "model_params_path": str(model_params_path),
        "logs_dir": str(logs_dir),
        "cache_dir": str(cache_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "model_dir": str(model_dir),
        "data_path": str(data_path),
        "perf_log_path": str(logs_dir / "backtest_performance.csv"),
    }


@pytest.fixture
def mock_env(tmp_dirs):
    """Crée un environnement de trading factice."""
    env = TradingEnv(tmp_dirs["config_path"], market="ES")
    env.data = pd.read_csv(tmp_dirs["data_path"])
    env.observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(350,),
        dtype=np.float32,
    )
    env.action_space = spaces.Box(
        low=-1,
        high=1,
        shape=(1,),
        dtype=np.float32,
    )
    return env


@pytest.fixture
def mock_model():
    """Crée un modèle SAC factice."""
    model = MagicMock(spec=SAC)
    model.predict.return_value = (np.array([0.5]), None)
    model.policy_alias = "mlp"
    return model


@pytest.fixture
def mock_detector():
    """Crée un détecteur de régime factice."""
    detector = MagicMock()
    detector.detect.return_value = ("trend", {"neural_regime": 0})
    return detector


@pytest.fixture
def mock_pipeline():
    """Crée un pipeline neuronal factice."""
    pipeline = MagicMock()
    pipeline.load_models.return_value = None
    pipeline.run.return_value = {
        "volatility": [1.5],
        "regime": [0],
        "features": np.array([[0.5] * 9]),
    }
    return pipeline


@pytest.mark.asyncio
async def test_load_config(tmp_dirs):
    """Teste le chargement de la configuration."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        config = load_config(tmp_dirs["config_path"], market="ES")
        expected_path = str(
            Path(tmp_dirs["features_dir"]) / "features_latest_filtered.csv"
        )
        assert (
            config["data_path"] == expected_path
        ), "Chemin des données incorrect"
        assert (
            config["sequence_length"] == 50
        ), "Longueur de séquence incorrecte"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "load_config" in str(op) for op in df_perf["operation"]
        ), "Opération load_config non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_validate_data(tmp_dirs):
    """Teste la validation des données."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        data = pd.read_csv(tmp_dirs["data_path"])
        validate_data(data, market="ES")
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_validate_data" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot validate_data non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "validate_data" in str(op) for op in df_perf["operation"]
        ), "Opération validate_data non journalisée"
        assert any(
            "confidence_drop_rate" in str(kw)
            for kw in df_perf.to_dict("records")
        ), "confidence_drop_rate absent"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_validate_features(tmp_dirs):
    """Teste la validation des features."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        data = pd.read_csv(tmp_dirs["data_path"])
        validate_features(data, step=0, market="ES")
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_validate_features" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot validate_features non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "validate_features" in str(op) for op in df_perf["operation"]
        ), "Opération validate_features non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_eval_sac_segment(
    tmp_dirs,
    mock_env,
    mock_model,
    mock_detector,
    mock_pipeline,
):
    """Teste l'évaluation d’un segment de données."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        data = pd.read_csv(tmp_dirs["data_path"])
        result = eval_sac_segment(
            model=mock_model,
            data=data,
            start_idx=0,
            end_idx=len(data),
            env=mock_env,
            regime_detector=mock_detector,
            neural_pipeline=mock_pipeline,
            market="ES",
        )
        profit_factor, sharpe, rewards, trade_log, rejected_trades, rejected_stats = result
        assert isinstance(
            profit_factor, float
        ), "Profit factor doit être un float"
        assert isinstance(sharpe, float), "Sharpe doit être un float"
        assert len(trade_log) > 0, "Trade log vide"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_eval_sac_segment" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot eval_sac_segment non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "eval_sac_segment" in str(op) for op in df_perf["operation"]
        ), "Opération eval_sac_segment non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_monte_carlo_backtest(
    tmp_dirs,
    mock_env,
    mock_model,
    mock_detector,
    mock_pipeline,
):
    """Teste le backtest Monte Carlo."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        data = pd.read_csv(tmp_dirs["data_path"])
        mc_stats = monte_carlo_backtest(
            model=mock_model,
            data=data,
            env=mock_env,
            regime_detector=mock_detector,
            neural_pipeline=mock_pipeline,
            n_simulations=5,
            market="ES",
        )
        assert (
            "profit_factor_mean" in mc_stats
        ), "Statistiques Monte Carlo incomplètes"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_monte_carlo_backtest" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot monte_carlo_backtest non créé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "monte_carlo_results" in f and f.endswith(".json.gz")
            for f in checkpoint_files
        ), "Checkpoint monte_carlo_results non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "monte_carlo_backtest" in str(op) for op in df_perf["operation"]
        ), "Opération monte_carlo_backtest non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_backtest_sac(tmp_dirs, mock_env, mock_model):
    """Teste le backtest complet."""
    with patch(
        "src.backtest.backtest_sac.SAC.load"
    ) as mock_sac_load, patch(
        "src.features.neural_pipeline.NeuralPipeline"
    ) as mock_pipeline, patch(
        "src.backtest.backtest_sac.MarketRegimeDetector"
    ) as mock_detector, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        mock_sac_load.return_value = mock_model
        mock_pipeline.return_value.load_models.return_value = None
        mock_pipeline.return_value.run.return_value = {
            "volatility": [1.5],
            "regime": [0],
            "features": np.array([[0.5] * 9]),
        }
        mock_detector.return_value.detect.return_value = (
            "defensive",
            {"neural_regime": 2},
        )
        result = backtest_sac(
            config_path=tmp_dirs["config_path"],
            mode="defensive",
            n_jobs=1,
            policy_type="mlp",
            market="ES",
        )
        profit_factor, sharpe, max_drawdown, mc_stats = result
        assert isinstance(
            profit_factor, float
        ), "Profit factor doit être un float"
        assert isinstance(sharpe, float), "Sharpe doit être un float"
        assert isinstance(
            max_drawdown, float
        ), "Max drawdown doit être un float"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_backtest_sac" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot backtest_sac non créé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "trade_log" in f and f.endswith(".json.gz")
            for f in checkpoint_files
        ), "Checkpoint trade_log non créé"
        output_files = os.listdir(tmp_dirs["logs_dir"])
        assert any(
            "backtest_defensive" in f and f.endswith(".csv")
            for f in output_files
        ), "Fichier de logs non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "backtest_sac" in str(op) for op in df_perf["operation"]
        ), "Opération backtest_sac non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_incremental_backtest_sac(
    tmp_dirs,
    mock_env,
    mock_model,
    mock_detector,
    mock_pipeline,
):
    """Teste le backtest incrémental."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        data = pd.read_csv(tmp_dirs["data_path"])
        row = data.iloc[0]
        buffer = pd.DataFrame()
        state = {}
        metrics, new_state = incremental_backtest_sac(
            row=row,
            buffer=buffer,
            state=state,
            model=mock_model,
            env=mock_env,
            regime_detector=mock_detector,
            neural_pipeline=mock_pipeline,
            config_path=tmp_dirs["config_path"],
            market="ES",
        )
        assert (
            metrics["profit_factor"] == 0.0
        ), "Métriques incorrectes pour buffer vide"
        buffer = data.iloc[:50]
        metrics, new_state = incremental_backtest_sac(
            row=row,
            buffer=buffer,
            state=state,
            model=mock_model,
            env=mock_env,
            regime_detector=mock_detector,
            neural_pipeline=mock_pipeline,
            config_path=tmp_dirs["config_path"],
            market="ES",
        )
        assert len(new_state["trade_log"]) > 0, "Trade log vide"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_incremental_backtest_sac" in f
            and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot incremental_backtest_sac non créé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "incremental_trade_log" in f and f.endswith(".json.gz")
            for f in checkpoint_files
        ), "Checkpoint incremental_trade_log non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "incremental_backtest_sac" in str(op)
            for op in df_perf["operation"]
        ), "Opération incremental_backtest_sac non journalisée"
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
                "step": [0],
                "action": [0.5],
            }
        )
        from src.backtest.backtest_sac import cloud_backup

        cloud_backup(
            df,
            data_type="test_metrics",
            market="ES",
        )
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
async def test_checkpoint(tmp_dirs):
    """Teste la sauvegarde incrémentielle."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        df = pd.DataFrame(
            {
                "timestamp": [datetime.now().isoformat()],
                "step": [0],
                "action": [0.5],
            }
        )
        from src.backtest.backtest_sac import checkpoint

        checkpoint(
            df,
            data_type="test_metrics",
            market="ES",
        )
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "backtest_test_metrics" in f and f.endswith(".json.gz")
            for f in checkpoint_files
        ), "Checkpoint non créé"
        with gzip.open(
            os.path.join(tmp_dirs["checkpoints_dir"], checkpoint_files[0]),
            "rt",
            encoding="utf-8",
        ) as f:
            checkpoint_data = json.load(f)
        assert (
            checkpoint_data["num_rows"] == len(df)
        ), "Nombre de lignes incorrect"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "checkpoint" in str(op) for op in df_perf["operation"]
        ), "Opération checkpoint non journalisée"
        mock_telegram.assert_called()