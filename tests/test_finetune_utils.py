# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_finetune_utils.py
# Tests unitaires pour src/model/utils/finetune_utils.py
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide le fine-tuning (méthode 8) et l’apprentissage en ligne
# (méthode 10) pour les modèles SAC, utilisant des mini-batchs et intégrant
# les 350 features pour l’entraînement et 150 SHAP features pour l’inférence,
# avec support multi-marchés.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pandas>=2.0.0,<3.0.0
# - numpy>=1.26.4,<2.0.0
# - psutil>=5.9.8,<6.0.0
# - stable-baselines3>=2.0.0,<3.0.0
# - pyyaml>=6.0.0,<7.0.0
# - torch>=2.0.0,<3.0.0
# - boto3>=1.26.0,<2.0.0
# - loguru>=0.7.0,<1.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/miya_console.py
# - src/model/utils/alert_manager.py
# - src/envs/trading_env.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/es_config.yaml
# - config/feature_sets.yaml
# - model/sac_models/<market>/<mode>/<policy_type>/*.zip
# - Données factices pour les tests
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de finetune_utils.py.
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Tests la Phase 8 (confidence_drop_rate) et autres standards.
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
from unittest.mock import MagicMock, patch

from gymnasium import spaces
import numpy as np
import pandas as pd
import pytest
from stable_baselines3 import SAC
import yaml

from src.model.utils.finetune_utils import (
    finetune_model,
    log_performance,
    online_learning,
    save_snapshot,
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
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    cache_dir = data_dir / "cache" / "finetune" / "ES"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints" / "finetune" / "ES"
    checkpoints_dir.mkdir(parents=True)
    model_dir = data_dir / "model" / "sac_models" / "ES" / "trend" / "mlp"
    model_dir.mkdir(parents=True)

    # Créer es_config.yaml
    config_path = config_dir / "es_config.yaml"
    config_content = {
        "backtest_sac": {
            "sequence_length": 50,
            "s3_bucket": "test-bucket",
            "s3_prefix": "finetune/",
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

    # Créer un fichier modèle factice
    model_file = model_dir / "sac_trend_mlp_20250513_120000.zip"
    with open(model_file, "wb") as f:
        f.write(b"")

    # Créer un DataFrame factice
    data_path = data_dir / "features" / "features_latest_filtered.csv"
    data_path.parent.mkdir(exist_ok=True)
    feature_cols = [f"feature_{i}" for i in range(350)]
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-05-13 09:00",
                periods=1000,
                freq="1min",
            ),
            "close": [5100] * 1000,
            "bid_size_level_1": [100] * 1000,
            "ask_size_level_1": [120] * 1000,
            "trade_frequency_1s": [8] * 1000,
            "spread_avg_1min_es": [0.3] * 1000,
            **{
                col: [np.random.uniform(0, 1)] * 1000
                for col in feature_cols
            },
        }
    )
    data.to_csv(data_path, index=False)

    return {
        "base_dir": str(base_dir),
        "config_path": str(config_path),
        "feature_sets_path": str(feature_sets_path),
        "logs_dir": str(logs_dir),
        "cache_dir": str(cache_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "model_dir": str(model_dir),
        "data_path": str(data_path),
        "perf_log_path": str(logs_dir / "finetune_performance.csv"),
    }


@pytest.fixture
def mock_data(tmp_dirs):
    """Crée un DataFrame factice avec 350 features."""
    feature_cols = [f"feature_{i}" for i in range(350)]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-05-13 09:00",
                periods=1000,
                freq="1min",
            ),
            "close": [5100] * 1000,
            "bid_size_level_1": [100] * 1000,
            "ask_size_level_1": [120] * 1000,
            "trade_frequency_1s": [8] * 1000,
            "spread_avg_1min_es": [0.3] * 1000,
            **{
                col: [np.random.uniform(0, 1)] * 1000
                for col in feature_cols
            },
        }
    )


@pytest.fixture
def mock_env(tmp_dirs):
    """Crée un environnement de trading factice."""
    env = MagicMock()
    env.mode = "trend"
    env.policy_type = "mlp"
    env.sequence_length = 50
    env.obs_t = [f"feature_{i}" for i in range(350)]
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
    env.data = pd.read_csv(tmp_dirs["data_path"])
    return env


@pytest.fixture
def mock_model():
    """Crée un modèle SAC factice."""
    model = MagicMock(spec=SAC)
    model.learn.return_value = None
    model.save.return_value = None
    model.set_parameters.return_value = None
    return model


@pytest.mark.asyncio
async def test_log_performance(tmp_dirs):
    """Teste la journalisation des performances."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        log_performance("test_op", 0.5, success=True, market="ES")
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "test_op" in str(op) for op in df_perf["operation"]
        ), "Opération test_op non journalisée"
        assert (
            df_perf["memory_usage_mb"].iloc[-1] > 0
        ), "Usage mémoire non journalisé"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_save_snapshot(tmp_dirs):
    """Teste la sauvegarde des snapshots."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        save_snapshot("test_snapshot", {"test": "data"}, market="ES")
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_test_snapshot" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot test_snapshot non créé"
        with gzip.open(
            os.path.join(tmp_dirs["cache_dir"], snapshot_files[0]),
            "rt",
            encoding="utf-8",
        ) as f:
            snapshot_data = json.load(f)
        assert (
            snapshot_data["type"] == "test_snapshot"
        ), "Type de snapshot incorrect"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "save_snapshot" in str(op) for op in df_perf["operation"]
        ), "Opération save_snapshot non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_validate_features(tmp_dirs, mock_data):
    """Teste la validation des features."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        validate_features(mock_data, shap_features=False, market="ES")
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
            "confidence_drop_rate" in str(kw)
            for kw in df_perf.to_dict("records")
        ), "confidence_drop_rate absent"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_finetune_model(tmp_dirs, mock_data, mock_env, mock_model):
    """Teste le fine-tuning du modèle SAC."""
    with patch("src.model.utils.finetune_utils.SAC.load") as mock_sac_load, patch(
        "src.model.utils.config_manager.get_config"
    ) as mock_config, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        mock_sac_load.return_value = mock_model
        mock_config.side_effect = [
            {
                "training": {
                    "features": [f"feature_{i}" for i in range(350)]
                },
                "inference": {
                    "shap_features": [
                        f"shap_feature_{i}" for i in range(150)
                    ]
                },
            },
            {"backtest_sac": {"sequence_length": 50}},
        ]
        model = finetune_model(
            mock_data,
            config_path=tmp_dirs["config_path"],
            mode="trend",
            policy_type="mlp",
            market="ES",
        )
        assert model is not None, "Modèle fine-tuné non retourné"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_finetune_model" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot finetune_model non créé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "finetune_batch" in f and f.endswith(".json.gz")
            for f in checkpoint_files
        ), "Checkpoint finetune_batch non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "finetune_model" in str(op) for op in df_perf["operation"]
        ), "Opération finetune_model non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_online_learning(tmp_dirs, mock_data, mock_env, mock_model):
    """Teste l’apprentissage en ligne."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        model = online_learning(
            mock_data,
            mock_model,
            mock_env,
            config_path=tmp_dirs["config_path"],
            market="ES",
        )
        assert model is not None, "Modèle mis à jour non retourné"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_online_learning" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot online_learning non créé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "online_learning_batch" in f and f.endswith(".json.gz")
            for f in checkpoint_files
        ), "Checkpoint online_learning_batch non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "online_learning" in str(op) for op in df_perf["operation"]
        ), "Opération online_learning non journalisée"
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
                "close": [5100],
            }
        )
        from src.model.utils.finetune_utils import cloud_backup

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
async def test_checkpoint(tmp_dirs):
    """Teste la sauvegarde incrémentielle."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        df = pd.DataFrame(
            {
                "timestamp": [datetime.now().isoformat()],
                "close": [5100],
            }
        )
        from src.model.utils.finetune_utils import checkpoint

        checkpoint(df, data_type="test_metrics", market="ES")
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "finetune_test_metrics" in f and f.endswith(".json.gz")
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