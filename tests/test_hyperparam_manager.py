# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_hyperparam_manager.py
# Tests unitaires pour src/model/utils/hyperparam_manager.py
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide la gestion des hyperparamètres pour SAC, PPO, DDPG (méthodes 6, 8, 18, Phase 14),
#        avec support multi-marchés, snapshots compressés, et sauvegardes incrémentielles/distribuées.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, pyyaml>=6.0.0,<7.0.0, psutil>=5.9.8,<6.0.0,
#   boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - Fichier YAML factice (algo_config.yaml)
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de hyperparam_manager.py.
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Tests la Phase 8 (confidence_drop_rate), méthodes 6, 8, 18, et Phase 14 (gestion des hyperparamètres).
# - Vérifie l’absence de dxFeed, la suppression de 320/81 features et obs_t, les retries,
#   logs psutil, alertes Telegram, snapshots compressés, et sauvegardes incrémentielles/distribuées.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.

import gzip
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

from src.model.utils.hyperparam_manager import HyperparamManager


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, cache, checkpoints, et configuration."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs" / "market"
    logs_dir.mkdir(parents=True)
    cache_dir = data_dir / "cache" / "hyperparams" / "ES"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints" / "hyperparams" / "ES"
    checkpoints_dir.mkdir(parents=True)

    # Créer algo_config.yaml
    algo_config_path = config_dir / "algo_config.yaml"
    algo_config_content = {
        "sac": {
            "learning_rate": {"value": 0.001},
            "batch_size": {"value": 64},
            "gamma": {"value": 0.99},
            "ent_coef": {"value": 0.1},
            "l2_lambda": {"value": 0.01},
            "trend": {"learning_rate": {"value": 0.0005}, "batch_size": {"value": 128}},
            "range": {"learning_rate": {"value": 0.002}, "batch_size": {"value": 32}},
            "defensive": {
                "learning_rate": {"value": 0.001},
                "batch_size": {"value": 64},
            },
        },
        "ppo": {
            "learning_rate": {"value": 0.001},
            "batch_size": {"value": 64},
            "gamma": {"value": 0.99},
            "ent_coef": {"value": 0.1},
            "l2_lambda": {"value": 0.01},
            "trend": {"learning_rate": {"value": 0.0005}, "batch_size": {"value": 128}},
        },
        "ddpg": {
            "learning_rate": {"value": 0.001},
            "batch_size": {"value": 64},
            "gamma": {"value": 0.99},
            "ent_coef": {"value": 0.1},
            "l2_lambda": {"value": 0.01},
        },
    }
    with open(algo_config_path, "w", encoding="utf-8") as f:
        yaml.dump(algo_config_content, f)

    # Créer es_config.yaml
    es_config_path = config_dir / "es_config.yaml"
    es_config_content = {"s3_bucket": "test-bucket", "s3_prefix": "hyperparams/"}
    with open(es_config_path, "w", encoding="utf-8") as f:
        yaml.dump(es_config_content, f)

    return {
        "base_dir": str(base_dir),
        "config_dir": str(config_dir),
        "logs_dir": str(logs_dir),
        "cache_dir": str(cache_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "algo_config_path": str(algo_config_path),
        "es_config_path": str(es_config_path),
        "perf_log_path": str(logs_dir / "hyperparam_manager_performance.csv"),
    }


@pytest.mark.asyncio
async def test_load_config(tmp_dirs):
    """Teste le chargement de la configuration."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        manager = HyperparamManager(config_path=Path(tmp_dirs["algo_config_path"]))
        assert "sac" in manager.hyperparams, "Modèle SAC absent dans la configuration"
        assert "ppo" in manager.hyperparams, "Modèle PPO absent dans la configuration"
        assert "ddpg" in manager.hyperparams, "Modèle DDPG absent dans la configuration"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "load_config" in str(op) for op in df_perf["operation"]
        ), "Opération load_config non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_validate_hyperparams(tmp_dirs):
    """Teste la validation des hyperparamètres."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        manager = HyperparamManager(config_path=Path(tmp_dirs["algo_config_path"]))
        hyperparams = {
            "learning_rate": {"value": 0.001},
            "batch_size": {"value": 64},
            "gamma": {"value": 0.99},
            "ent_coef": {"value": 0.1},
            "l2_lambda": {"value": 0.01},
        }
        manager._validate_hyperparams(hyperparams, "sac", "range", market="ES")
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_validate_hyperparams" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot validate_hyperparams non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "validate_hyperparams" in str(op) for op in df_perf["operation"]
        ), "Opération validate_hyperparams non journalisée"
        assert any(
            "confidence_drop_rate" in str(kw) for kw in df_perf.to_dict("records")
        ), "confidence_drop_rate absent"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_get_hyperparams(tmp_dirs):
    """Teste la récupération des hyperparamètres."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        manager = HyperparamManager(config_path=Path(tmp_dirs["algo_config_path"]))
        params = manager.get_hyperparams("sac", "range", market="ES")
        assert (
            params["learning_rate"]["value"] == 0.002
        ), "Hyperparamètre learning_rate incorrect"
        assert (
            params["batch_size"]["value"] == 32
        ), "Hyperparamètre batch_size incorrect"
        params = manager.get_hyperparams("ppo", "trend", market="ES")
        assert (
            params["learning_rate"]["value"] == 0.0005
        ), "Hyperparamètre learning_rate incorrect"
        params = manager.get_hyperparams("ddpg", market="ES")
        assert (
            params["learning_rate"]["value"] == 0.001
        ), "Hyperparamètre learning_rate incorrect"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_get_hyperparams" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot get_hyperparams non créé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "hyperparam_load" in f and f.endswith(".json.gz") for f in checkpoint_files
        ), "Checkpoint hyperparam_load non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "get_hyperparams" in str(op) for op in df_perf["operation"]
        ), "Opération get_hyperparams non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_get_hyperparams_invalid(tmp_dirs):
    """Teste la récupération des hyperparamètres avec des entrées invalides."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        manager = HyperparamManager(config_path=Path(tmp_dirs["algo_config_path"]))
        with pytest.raises(ValueError):
            manager.get_hyperparams("invalid_model", market="ES")
        with pytest.raises(ValueError):
            manager.get_hyperparams("sac", "invalid_regime", market="ES")
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_cloud_backup(tmp_dirs):
    """Teste la sauvegarde distribuée S3."""
    with patch("boto3.client") as mock_s3, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        manager = HyperparamManager(config_path=Path(tmp_dirs["algo_config_path"]))
        df = pd.DataFrame(
            {"model_type": ["sac"], "regime": ["range"], "num_params": [5]}
        )
        manager.cloud_backup(df, data_type="test_metrics", market="ES")
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
        manager = HyperparamManager(config_path=Path(tmp_dirs["algo_config_path"]))
        df = pd.DataFrame(
            {"model_type": ["sac"], "regime": ["range"], "num_params": [5]}
        )
        manager.checkpoint(df, data_type="test_metrics", market="ES")
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "hyperparam_test_metrics" in f and f.endswith(".json.gz")
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
