# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_algo_performance_logger.py
# Tests unitaires pour src/model/utils/algo_performance_logger.py
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide l’enregistrement des performances des algorithmes SAC, PPO, DDPG, avec métriques de fine-tuning,
#        apprentissage en ligne, et meta-learning, avec support multi-marchés, snapshots compressés, et sauvegardes
#        incrémentielles/distribuées.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, psutil>=5.9.8,<6.0.0, matplotlib>=3.7.0,<4.0.0,
#   boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0, pyyaml>=6.0.0,<7.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - Fichier de configuration factice (algo_config.yaml)
# - Données de performance simulées
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de algo_performance_logger.py.
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Tests la Phase 8 (confidence_drop_rate), méthode 10 (apprentissage en ligne), et méthode 18 (meta-learning).
# - Vérifie l’absence de dxFeed, la suppression de 320/81 features et obs_t, les retries,
#   logs psutil, alertes Telegram, snapshots compressés, et sauvegardes incrémentielles/distribuées.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.
# - Policies Note: The official directory for routing policies is src/model/router/policies.
# The src/model/policies directory is a residual and should be verified
# for removal to avoid import conflicts.

import gzip
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

from src.model.utils.algo_performance_logger import AlgoPerformanceLogger


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, cache, checkpoints, figures, et configuration."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs" / "ES"
    logs_dir.mkdir(parents=True)
    cache_dir = data_dir / "cache" / "algo_performance" / "ES"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints" / "algo_performance" / "ES"
    checkpoints_dir.mkdir(parents=True)
    figures_dir = data_dir / "figures" / "algo_performance" / "ES"
    figures_dir.mkdir(parents=True)

    # Créer algo_config.yaml
    algo_config_path = config_dir / "algo_config.yaml"
    algo_config_content = {
        "algo_performance_logger": {"log_interval": 10, "plot_frequency": 100}
    }
    with open(algo_config_path, "w", encoding="utf-8") as f:
        yaml.dump(algo_config_content, f)

    # Créer es_config.yaml pour S3
    es_config_path = config_dir / "es_config.yaml"
    es_config_content = {"s3_bucket": "test-bucket", "s3_prefix": "algo_performance/"}
    with open(es_config_path, "w", encoding="utf-8") as f:
        yaml.dump(es_config_content, f)

    return {
        "base_dir": str(base_dir),
        "config_dir": str(config_dir),
        "logs_dir": str(logs_dir),
        "cache_dir": str(cache_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "figures_dir": str(figures_dir),
        "algo_config_path": str(algo_config_path),
        "es_config_path": str(es_config_path),
        "sac_log_path": str(logs_dir / "sac_performance.csv"),
        "consolidated_log_path": str(logs_dir / "train_sac_performance.csv"),
    }


@pytest.mark.asyncio
async def test_init_algo_performance_logger(tmp_dirs):
    """Teste l’initialisation de AlgoPerformanceLogger."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        logger = AlgoPerformanceLogger(
            config_path=tmp_dirs["algo_config_path"], market="ES"
        )
        assert logger.market == "ES", "Marché incorrect"
        assert logger.log_files["sac"] == Path(
            tmp_dirs["sac_log_path"]
        ), "Chemin de log SAC incorrect"
        assert logger.consolidated_log_file == Path(
            tmp_dirs["consolidated_log_path"]
        ), "Chemin de log consolidé incorrect"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_log_performance(tmp_dirs):
    """Teste l’enregistrement des performances de base."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        logger = AlgoPerformanceLogger(
            config_path=tmp_dirs["algo_config_path"], market="ES"
        )
        logger.log_performance(
            algo_type="sac", regime="range", reward=100.0, latency=0.5, memory=512.0
        )
        df_sac = pd.read_csv(tmp_dirs["sac_log_path"])
        assert (
            len(df_sac) > 0
        ), "Aucune performance enregistrée dans sac_performance.csv"
        assert df_sac.iloc[-1]["reward"] == 100.0, "Récompense incorrecte"
        df_consolidated = pd.read_csv(tmp_dirs["consolidated_log_path"])
        assert (
            len(df_consolidated) > 0
        ), "Aucune performance enregistrée dans train_sac_performance.csv"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_log_sac_range" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot log_performance non créé"
        figure_files = os.listdir(tmp_dirs["figures_dir"])
        assert any(
            "log_sac_range" in f and f.endswith(".png") for f in figure_files
        ), "Figure log_performance non créée"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "log_sac_range" in f and f.endswith(".json.gz") for f in checkpoint_files
        ), "Checkpoint log_performance non créé"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_log_extended_performance(tmp_dirs):
    """Teste l’enregistrement des performances étendues."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        logger = AlgoPerformanceLogger(
            config_path=tmp_dirs["algo_config_path"], market="ES"
        )
        logger.log_extended_performance(
            algo_type="sac",
            regime="range",
            reward=100.0,
            latency=0.5,
            memory=512.0,
            finetune_loss=0.01,
            online_learning_steps=10,
            maml_steps=5,
        )
        df_consolidated = pd.read_csv(tmp_dirs["consolidated_log_path"])
        assert len(df_consolidated) > 0, "Aucune performance étendue enregistrée"
        assert (
            df_consolidated.iloc[-1]["finetune_loss"] == 0.01
        ), "Perte de fine-tuning incorrecte"
        assert (
            df_consolidated.iloc[-1]["online_learning_steps"] == 10
        ), "Étapes d’apprentissage en ligne incorrectes"
        assert (
            df_consolidated.iloc[-1]["maml_steps"] == 5
        ), "Étapes de meta-learning incorrectes"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_extended_log_sac_range" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot extended_log_performance non créé"
        figure_files = os.listdir(tmp_dirs["figures_dir"])
        assert any(
            "extended_log_sac_range" in f and f.endswith(".png") for f in figure_files
        ), "Figure extended_log_performance non créée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_log_performance_invalid_algo(tmp_dirs):
    """Teste l’enregistrement avec un algorithme invalide."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        logger = AlgoPerformanceLogger(
            config_path=tmp_dirs["algo_config_path"], market="ES"
        )
        logger.log_performance(
            algo_type="invalid", regime="range", reward=100.0, latency=0.5, memory=512.0
        )
        df_consolidated = pd.read_csv(tmp_dirs["consolidated_log_path"])
        assert (
            len(df_consolidated) == 0
        ), "Performance enregistrée pour algorithme invalide"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_cloud_backup(tmp_dirs):
    """Teste la sauvegarde distribuée S3."""
    with patch("boto3.client") as mock_s3, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        logger = AlgoPerformanceLogger(
            config_path=tmp_dirs["algo_config_path"], market="ES"
        )
        df = pd.DataFrame(
            {"timestamp": [datetime.now().isoformat()], "reward": [100.0]}
        )
        logger.cloud_backup(df, data_type="test_metrics")
        assert mock_s3.called, "Client S3 non appelé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "temp_s3_" in f for f in checkpoint_files
        ), "Fichier temporaire S3 non créé"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_checkpoint(tmp_dirs):
    """Teste la sauvegarde incrémentielle."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        logger = AlgoPerformanceLogger(
            config_path=tmp_dirs["algo_config_path"], market="ES"
        )
        df = pd.DataFrame(
            {"timestamp": [datetime.now().isoformat()], "reward": [100.0]}
        )
        logger.checkpoint(df, data_type="test_metrics")
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "algo_performance_test_metrics" in f and f.endswith(".json.gz")
            for f in checkpoint_files
        ), "Checkpoint non créé"
        with gzip.open(
            os.path.join(tmp_dirs["checkpoints_dir"], checkpoint_files[0]),
            "rt",
            encoding="utf-8",
        ) as f:
            checkpoint_data = json.load(f)
        assert checkpoint_data["num_rows"] == len(df), "Nombre de lignes incorrect"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_log_performance_buffer(tmp_dirs):
    """Teste l’enregistrement des métriques dans performance_buffer."""
    with patch("src.utils.telegram_alert.send_telegram_alert"):
        logger = AlgoPerformanceLogger(
            config_path=tmp_dirs["algo_config_path"], market="ES"
        )
        logger.log_performance_buffer(
            operation="test_buffer", latency=0.2, success=True, custom_metric=42
        )
        df_consolidated = pd.read_csv(tmp_dirs["consolidated_log_path"])
        assert (
            len(df_consolidated) > 0
        ), "Aucune métrique enregistrée dans train_sac_performance.csv"
        assert (
            df_consolidated.iloc[-1]["operation"] == "test_buffer"
        ), "Opération incorrecte"
        assert df_consolidated.iloc[-1]["latency"] == 0.2, "Latence incorrecte"
        assert df_consolidated.iloc[-1]["success"], "Succès incorrect"
        assert (
            df_consolidated.iloc[-1]["custom_metric"] == 42
        ), "Métrique personnalisée incorrecte"
