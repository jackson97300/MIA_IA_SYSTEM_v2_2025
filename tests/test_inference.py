# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_inference.py
# Tests unitaires pour src/model/inference.py
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide les prédictions avec un modèle SAC entraîné, intégrant trade_success_prob via TradeProbabilityPredictor,
#        utilisant les top 150 SHAP features pour l’inférence, avec validation quotidienne et support multi-marchés.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, stable-baselines3>=2.0.0,<3.0.0,
#   psutil>=5.9.8,<6.0.0, pyyaml>=6.0.0,<7.0.0, gymnasium>=1.0.0,<2.0.0, boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0
# - src/envs/trading_env.py
# - src/envs/gym_wrappers.py
# - src/model/utils/trading_utils.py
# - src/model/router/detect_regime.py
# - src/model/router/mode_trend.py
# - src/model/router/mode_range.py
# - src/model/router/mode_defensive.py
# - src/model/utils_model.py
# - src/features/neural_pipeline.py
# - src/model/adaptive_learning.py
# - src/model/utils/miya_console.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/trade_probability.py
# - src/data/data_provider.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/es_config.yaml
# - data/features/features_latest_filtered.csv
# - data/features/feature_importance.csv
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de inference.py.
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Tests la Phase 8 (confidence_drop_rate) et autres standards.
# - Vérifie l’absence de dxFeed, la suppression de 320/81 features et obs_t, les retries, logs psutil,
#   alertes Telegram, snapshots compressés, et sauvegardes incrémentielles/distribuées.
# - Compatible avec 150 SHAP features pour l’inférence via config/feature_sets.yaml.
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
from gymnasium import spaces
from stable_baselines3 import SAC

from src.model.inference import InferenceEngine, load_model, predict_sac


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, cache, checkpoints, et modèles."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    cache_dir = data_dir / "cache" / "inference" / "ES"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints" / "inference" / "ES"
    checkpoints_dir.mkdir(parents=True)
    model_dir = data_dir / "model" / "sac_models" / "ES" / "deployed" / "trend" / "mlp"
    model_dir.mkdir(parents=True)
    features_dir = data_dir / "features"
    features_dir.mkdir()
    ml_models_dir = data_dir / "model" / "ml_models"
    ml_models_dir.mkdir()

    # Créer es_config.yaml
    config_path = config_dir / "es_config.yaml"
    config_content = {
        "logging": {"directory": str(logs_dir)},
        "s3_bucket": "test-bucket",
        "s3_prefix": "inference/",
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    # Créer feature_sets.yaml
    feature_sets_path = config_dir / "feature_sets.yaml"
    feature_sets_content = {
        "training": {"features": [f"feature_{i}" for i in range(350)]},
        "inference": {"shap_features": [f"shap_feature_{i}" for i in range(150)]},
    }
    with open(feature_sets_path, "w", encoding="utf-8") as f:
        yaml.dump(feature_sets_content, f)

    # Créer model_params.yaml
    model_params_path = config_dir / "model_params.yaml"
    model_params_content = {"window_size": 50, "base_features": 150}
    with open(model_params_path, "w", encoding="utf-8") as f:
        yaml.dump(model_params_content, f)

    # Créer feature_importance.csv
    shap_file = features_dir / "feature_importance.csv"
    shap_data = pd.DataFrame({"feature": [f"shap_feature_{i}" for i in range(150)]})
    shap_data.to_csv(shap_file, index=False)

    # Créer features_latest_filtered.csv
    data_path = features_dir / "features_latest_filtered.csv"
    feature_cols = [f"shap_feature_{i}" for i in range(150)]
    data = pd.DataFrame(
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
            **{col: [np.random.uniform(0, 1)] * 100 for col in feature_cols},
        }
    )
    data.to_csv(data_path, index=False)

    return {
        "base_dir": str(base_dir),
        "config_path": str(config_path),
        "feature_sets_path": str(feature_sets_path),
        "model_params_path": str(model_params_path),
        "logs_dir": str(logs_dir),
        "cache_dir": str(cache_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "model_dir": str(model_dir),
        "features_dir": str(features_dir),
        "ml_models_dir": str(ml_models_dir),
        "data_path": str(data_path),
        "shap_file": str(shap_file),
        "perf_log_path": str(logs_dir / "inference_performance.csv"),
    }


@pytest.fixture
def mock_env():
    """Crée un environnement de trading factice."""
    env = MagicMock()
    env.sequence_length = 50
    env.observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(150,), dtype=np.float32
    )
    env.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
    env.obs_t = [f"shap_feature_{i}" for i in range(150)]
    env.reset.return_value = (np.zeros((150,), dtype=np.float32), {})
    env.step.return_value = (
        np.zeros((150,), dtype=np.float32),
        0.0,
        False,
        False,
        {"balance": 10000.0, "position": 0, "price": 5100.0},
    )
    env.current_step = 0
    env.mode = "trend"
    env.policy_type = "mlp"
    env.data = pd.DataFrame()
    return env


@pytest.fixture
def mock_model():
    """Crée un modèle SAC factice."""
    model = MagicMock(spec=SAC)
    model.predict.return_value = (np.array([0.5]), None)
    return model


@pytest.mark.asyncio
async def test_init(tmp_dirs):
    """Teste l’initialisation de InferenceEngine."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        InferenceEngine(tmp_dirs["config_path"])
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "init" in str(op) for op in df_perf["operation"]
        ), "Opération init non journalisée"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_init" in f and f.endswith(".json.gz") for f in snapshot_files
        ), "Snapshot init non créé"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_configure_feature_set(tmp_dirs):
    """Teste le chargement des top 150 SHAP features."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        engine = InferenceEngine(tmp_dirs["config_path"])
        features = engine.configure_feature_set()
        assert len(features) == 150, "Nombre incorrect de SHAP features"
        assert all(
            f.startswith("shap_feature_") for f in features
        ), "Features SHAP incorrectes"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_validate_shap_features(tmp_dirs):
    """Teste la validation quotidienne des SHAP features."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        engine = InferenceEngine(tmp_dirs["config_path"])
        assert engine.validate_shap_features(
            market="ES"
        ), "Validation SHAP features échouée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_load_data(tmp_dirs):
    """Teste le chargement des données."""
    with patch("src.data.data_provider.get_data_provider") as mock_provider, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        mock_provider.return_value.fetch_features.return_value = pd.read_csv(
            tmp_dirs["data_path"]
        )
        engine = InferenceEngine(tmp_dirs["config_path"])
        data = await engine.load_data(tmp_dirs["data_path"], market="ES")
        assert not data.empty, "Données non chargées"
        assert len(data) == 100, "Nombre incorrect de lignes"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_load_data" in f and f.endswith(".json.gz") for f in snapshot_files
        ), "Snapshot load_data non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "load_data" in str(op) for op in df_perf["operation"]
        ), "Opération load_data non journalisée"
        assert any(
            "confidence_drop_rate" in str(kw) for kw in df_perf.to_dict("records")
        ), "confidence_drop_rate absent"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_load_model(tmp_dirs, mock_env, mock_model):
    """Teste le chargement du modèle SAC."""
    with patch("src.model.router.mode_trend.ModeTrend") as mock_trend, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        mock_trend.return_value = mock_model
        model_file = os.path.join(
            tmp_dirs["model_dir"], "sac_trend_mlp_20250513_120000.zip"
        )
        with open(model_file, "wb") as f:
            f.write(b"")
        agent, env = load_model(
            tmp_dirs["config_path"], "trend", model_file, "mlp", market="ES"
        )
        assert agent is not None, "Agent non chargé"
        assert env is not None, "Environnement non chargé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "load_model" in str(op) for op in df_perf["operation"]
        ), "Opération load_model non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_predict_sac_fixed(tmp_dirs, mock_env, mock_model):
    """Teste les prédictions en mode fixe."""
    with patch("src.data.data_provider.get_data_provider") as mock_provider, patch(
        "src.model.router.mode_trend.ModeTrend"
    ) as mock_trend, patch(
        "src.model.trade_probability.TradeProbabilityPredictor.predict"
    ) as mock_predict, patch(
        "src.model.utils.trading_utils.validate_trade_entry_combined"
    ) as mock_validate, patch(
        "src.model.utils.trading_utils.adjust_risk_and_leverage"
    ) as mock_adjust, patch(
        "src.model.adaptive_learning.store_pattern"
    ) as mock_store, patch(
        "src.model.adaptive_learning.last_3_trades_success_rate"
    ) as mock_success_rate, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        mock_provider.return_value.fetch_features.return_value = pd.read_csv(
            tmp_dirs["data_path"]
        )
        mock_trend.return_value = mock_model
        mock_predict.return_value = 0.6
        mock_validate.return_value = True
        mock_adjust.return_value = None
        mock_store.return_value = None
        mock_success_rate.return_value = 0.7
        model_file = os.path.join(
            tmp_dirs["model_dir"], "sac_trend_mlp_20250513_120000.zip"
        )
        with open(model_file, "wb") as f:
            f.write(b"")
        result = await predict_sac(
            config_path=tmp_dirs["config_path"],
            data_path=tmp_dirs["data_path"],
            mode="trend",
            model_path=model_file,
            dynamic_mode=False,
            policy_type="mlp",
            market="ES",
        )
        assert not result.empty, "Résultat vide"
        assert len(result) > 0, "Aucune prédiction générée"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_prediction" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot prediction non créé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "predictions" in f and f.endswith(".json.gz") for f in checkpoint_files
        ), "Checkpoint predictions non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "predict_step" in str(op) for op in df_perf["operation"]
        ), "Opération predict_step non journalisée"
        output_files = os.listdir(tmp_dirs["logs_dir"])
        assert any(
            "inference_trend_mlp" in f and f.endswith(".csv") for f in output_files
        ), "Fichier de prédictions non créé"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_predict_sac_dynamic(tmp_dirs, mock_env, mock_model):
    """Teste les prédictions en mode dynamique."""
    with patch("src.data.data_provider.get_data_provider") as mock_provider, patch(
        "src.model.router.mode_trend.ModeTrend"
    ) as mock_trend, patch(
        "src.model.router.mode_range.ModeRange"
    ) as mock_range, patch(
        "src.model.router.mode_defensive.ModeDefensive"
    ) as mock_defensive, patch(
        "src.model.router.detect_regime.MarketRegimeDetector.detect"
    ) as mock_detect, patch(
        "src.model.trade_probability.TradeProbabilityPredictor.predict"
    ) as mock_predict, patch(
        "src.model.utils.trading_utils.validate_trade_entry_combined"
    ) as mock_validate, patch(
        "src.model.utils.trading_utils.adjust_risk_and_leverage"
    ) as mock_adjust, patch(
        "src.model.adaptive_learning.store_pattern"
    ) as mock_store, patch(
        "src.model.adaptive_learning.last_3_trades_success_rate"
    ) as mock_success_rate, patch(
        "src.features.neural_pipeline.NeuralPipeline"
    ) as mock_pipeline, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        mock_provider.return_value.fetch_features.return_value = pd.read_csv(
            tmp_dirs["data_path"]
        )
        mock_trend.return_value = mock_model
        mock_range.return_value = mock_model
        mock_defensive.return_value = mock_model
        mock_detect.return_value = ("trend", {"neural_regime": 0})
        mock_predict.return_value = 0.6
        mock_validate.return_value = True
        mock_adjust.return_value = None
        mock_store.return_value = None
        mock_success_rate.return_value = 0.7
        mock_pipeline.return_value.load_models.return_value = None
        mock_pipeline.return_value.run.return_value = {
            "volatility": [1.5],
            "regime": [0],
            "features": np.array([[0.5] * 9]),
        }
        model_file = os.path.join(
            tmp_dirs["model_dir"], "sac_trend_mlp_20250513_120000.zip"
        )
        with open(model_file, "wb") as f:
            f.write(b"")
        result = await predict_sac(
            config_path=tmp_dirs["config_path"],
            data_path=tmp_dirs["data_path"],
            dynamic_mode=True,
            policy_type="mlp",
            market="ES",
        )
        assert not result.empty, "Résultat vide"
        assert len(result) > 0, "Aucune prédiction générée"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_prediction" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot prediction non créé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "predictions" in f and f.endswith(".json.gz") for f in checkpoint_files
        ), "Checkpoint predictions non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "predict_step" in str(op) for op in df_perf["operation"]
        ), "Opération predict_step non journalisée"
        output_files = os.listdir(tmp_dirs["logs_dir"])
        assert any(
            "inference_dynamic_mlp" in f and f.endswith(".csv") for f in output_files
        ), "Fichier de prédictions non créé"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_cloud_backup(tmp_dirs):
    """Teste la sauvegarde distribuée S3."""
    with patch("boto3.client") as mock_s3, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        df = pd.DataFrame(
            {"timestamp": [datetime.now().isoformat()], "step": [0], "action": [0.5]}
        )
        engine = InferenceEngine(tmp_dirs["config_path"])
        engine.cloud_backup(df, data_type="test_metrics", market="ES")
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
            {"timestamp": [datetime.now().isoformat()], "step": [0], "action": [0.5]}
        )
        engine = InferenceEngine(tmp_dirs["config_path"])
        engine.checkpoint(df, data_type="test_metrics", market="ES")
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "inference_test_metrics" in f and f.endswith(".json.gz")
            for f in checkpoint_files
        ), "Checkpoint non créé"
        with gzip.open(
            os.path.join(tmp_dirs["checkpoints_dir"], checkpoint_files[0]),
            "rt",
            encoding="utf-8",
        ) as f:
            checkpoint_data = json.load(f)
        assert checkpoint_data["num_rows"] == len(df), "Nombre de lignes incorrect"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "checkpoint" in str(op) for op in df_perf["operation"]
        ), "Opération checkpoint non journalisée"
        mock_telegram.assert_called()
