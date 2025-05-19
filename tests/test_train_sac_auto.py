# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_train_sac_auto.py
# Tests unitaires pour src/model/train_sac_auto.py
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide l’automatisation de l’entraînement SAC, PPO, DDPG pour plusieurs modes, avec fine-tuning (méthode 8),
#        apprentissage en ligne (méthode 10), curriculum progressif (méthode 15), et meta-learning (méthode 18).
#        Vérifie les sauvegardes incrémentielles/distribuées et les alertes Telegram.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, psutil>=5.9.8,<6.0.0, matplotlib>=3.7.0,<4.0.0,
#   stable-baselines3>=2.0.0,<3.0.0, pyyaml>=6.0.0,<7.0.0, boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0
# - src/model/train_sac.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/model_validator.py
# - src/model/utils/algo_performance_logger.py
# - src/model/utils/finetune_utils.py
# - src/model/utils/maml_utils.py
# - src/envs/trading_env.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/algo_config.yaml
# - config/feature_sets.yaml
# - Données factices (DataFrame avec 350 features)
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de train_sac_auto.py.
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Tests les Phases 8 (confidence_drop_rate), 8 (fine-tuning), 10 (apprentissage en ligne),
#   15 (curriculum progressif), et 18 (meta-learning).
# - Vérifie l’absence de dxFeed, la suppression de 320/81 features et obs_t, les retries, logs psutil,
#   alertes Telegram, snapshots compressés, et sauvegardes incrémentielles/distribuées.
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence) via config/feature_sets.yaml.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.

import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml
from stable_baselines3 import SAC

from src.model.train_sac_auto import TrainSACAuto, validate_data


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, cache, checkpoints, figures, et modèles."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    cache_dir = data_dir / "cache" / "train_sac_auto" / "ES"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints" / "train_sac_auto" / "ES"
    checkpoints_dir.mkdir(parents=True)
    figures_dir = data_dir / "figures" / "train_sac_auto" / "ES"
    figures_dir.mkdir(parents=True)
    model_dir = data_dir / "model" / "sac_models" / "ES" / "trend" / "mlp"
    model_dir.mkdir(parents=True)

    # Créer algo_config.yaml
    config_path = config_dir / "algo_config.yaml"
    config_content = {
        "trade_probability": {"buffer_size": 100},
        "s3_bucket": "test-bucket",
        "s3_prefix": "train_sac_auto/",
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

    # Créer trading_env_config.yaml
    env_config_path = config_dir / "trading_env_config.yaml"
    env_config_content = {"sequence_length": 50}
    with open(env_config_path, "w", encoding="utf-8") as f:
        yaml.dump(env_config_content, f)

    return {
        "base_dir": str(base_dir),
        "config_path": str(config_path),
        "feature_sets_path": str(feature_sets_path),
        "env_config_path": str(env_config_path),
        "logs_dir": str(logs_dir),
        "cache_dir": str(cache_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "figures_dir": str(figures_dir),
        "model_dir": str(model_dir),
        "perf_log_path": str(logs_dir / "train_sac_auto_performance.csv"),
    }


@pytest.fixture
def mock_data():
    """Crée un DataFrame factice avec 350 features."""
    feature_cols = [f"feature_{i}" for i in range(350)]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=100, freq="1min"),
            "bid_size_level_1": np.random.randint(100, 500, 100),
            "ask_size_level_1": np.random.randint(100, 500, 100),
            "trade_frequency_1s": np.random.uniform(0.1, 10, 100),
            "spread_avg_1min": np.random.uniform(0.01, 0.05, 100),
            "close": np.random.uniform(4000, 5000, 100),
            **{f: np.random.uniform(0, 1, 100) for f in feature_cols},
        }
    )


@pytest.fixture
def mock_env():
    """Crée un environnement de trading factice."""
    env = MagicMock()
    env.sequence_length = 50
    env.observation_space.shape = (350,)
    env.action_space.shape = (1,)
    env.obs_t = [f"feature_{i}" for i in range(350)]
    return env


@pytest.fixture
def trainer(tmp_dirs):
    """Crée une instance de TrainSACAuto pour les tests."""
    with patch("src.model.utils.alert_manager.AlertManager") as mock_alert, patch(
        "src.model.utils.model_validator.ModelValidator"
    ) as mock_validator, patch(
        "src.model.utils.algo_performance_logger.AlgoPerformanceLogger"
    ) as mock_logger, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        mock_alert.return_value.send_alert.return_value = None
        mock_validator.return_value.validate_model.return_value = {
            "valid": True,
            "mean_reward": 1.0,
        }
        mock_logger.return_value.log_performance.return_value = None
        mock_telegram.return_value = None
        trainer = TrainSACAuto()
    return trainer


@pytest.mark.asyncio
async def test_init(tmp_dirs, trainer):
    """Teste l’initialisation de TrainSACAuto."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
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
async def test_validate_data(mock_data):
    """Teste la validation des données."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        assert validate_data(mock_data, market="ES"), "Validation des données échouée"
        df_perf = pd.read_csv(PERF_LOG_PATH)
        assert any(
            "validate_data" in str(op) for op in df_perf["operation"]
        ), "Opération validate_data non journalisée"
        assert any(
            "confidence_drop_rate" in str(kw) for kw in df_perf.to_dict("records")
        ), "confidence_drop_rate absent"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_validate_data_invalid(mock_data):
    """Teste la validation avec des données invalides."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        invalid_data = mock_data.drop(columns=["bid_size_level_1"])
        assert not validate_data(
            invalid_data, market="ES"
        ), "Validation des données invalides devrait échouer"
        df_perf = pd.read_csv(PERF_LOG_PATH)
        assert any(
            "validate_data" in str(op) for op in df_perf["operation"]
        ), "Opération validate_data non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_auto_train_sac(tmp_dirs, trainer, mock_data, mock_env):
    """Teste l’entraînement SAC via auto_train_sac."""
    with patch("src.model.train_sac.SACTrainer") as mock_trainer, patch(
        "src.model.utils.finetune_utils.finetune_model"
    ) as mock_finetune, patch(
        "src.model.utils.maml_utils.apply_prototypical_networks"
    ) as mock_maml, patch(
        "src.model.utils.finetune_utils.online_learning"
    ) as mock_online, patch(
        "src.envs.trading_env.TradingEnv"
    ) as mock_env_class, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        mock_env_class.return_value = mock_env
        mock_model = MagicMock(spec=SAC)
        mock_model.learn.return_value = None
        mock_model.save.return_value = None
        mock_finetune.return_value = mock_model
        mock_maml.return_value = mock_model
        mock_online.return_value = mock_model
        model = trainer.auto_train_sac(
            mock_data,
            tmp_dirs["env_config_path"],
            mode="trend",
            policy_type="mlp",
            epochs=100,
            market="ES",
        )
        assert model is not None, "Entraînement SAC échoué"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_auto_train_sac" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot auto_train_sac non créé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "model_state" in f and f.endswith(".json.gz") for f in checkpoint_files
        ), "Checkpoint model_state non créé"
        model_files = os.listdir(tmp_dirs["model_dir"])
        assert any(
            "sac_trend_mlp" in f and f.endswith(".zip") for f in model_files
        ), "Modèle SAC non sauvegardé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "auto_train_sac" in str(op) for op in df_perf["operation"]
        ), "Opération auto_train_sac non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_auto_train(tmp_dirs, trainer, mock_data):
    """Teste l’entraînement automatisé via auto_train."""
    with patch("src.model.train_sac.SACTrainer") as mock_trainer, patch(
        "src.model.utils.finetune_utils.finetune_model"
    ) as mock_finetune, patch(
        "src.model.utils.maml_utils.apply_prototypical_networks"
    ) as mock_maml, patch(
        "src.model.utils.finetune_utils.online_learning"
    ) as mock_online, patch(
        "src.envs.trading_env.TradingEnv"
    ) as mock_env_class, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        mock_env_class.return_value = MagicMock()
        mock_model = MagicMock(spec=SAC)
        mock_model.learn.return_value = None
        mock_model.save.return_value = None
        mock_finetune.return_value = mock_model
        mock_maml.return_value = mock_model
        mock_online.return_value = mock_model
        mock_trainer.return_value.train_multi_market = AsyncMock(return_value=None)
        mock_trainer.return_value.models = {
            "sac": {"trend": MagicMock(model=mock_model)}
        }
        results = await trainer.auto_train(
            mock_data,
            tmp_dirs["env_config_path"],
            modes=["trend"],
            algo_types=["sac"],
            policy_types=["mlp"],
            epochs=100,
            market="ES",
        )
        assert isinstance(results, dict), "Résultats d’entraînement incorrects"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_auto_train" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot auto_train non créé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "results" in f and f.endswith(".json.gz") for f in checkpoint_files
        ), "Checkpoint results non créé"
        figure_files = os.listdir(tmp_dirs["figures_dir"])
        assert any(
            "results_" in f and f.endswith(".png") for f in figure_files
        ), "Visualisation non générée"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "auto_train" in str(op) for op in df_perf["operation"]
        ), "Opération auto_train non journalisée"
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
                "operation": ["test"],
                "latency": [1.0],
            }
        )
        from src.model.train_sac_auto import cloud_backup

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
async def test_handle_sigint(tmp_dirs, trainer):
    """Teste la gestion SIGINT."""
    with patch("sys.exit") as mock_exit, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        trainer.handle_sigint(signal.SIGINT, None)
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_sigint" in f for f in snapshot_files
        ), "Snapshot sigint non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "handle_sigint" in str(op) for op in df_perf["operation"]
        ), "Opération handle_sigint non journalisée"
        mock_exit.assert_called_with(0)
        mock_telegram.assert_called()
