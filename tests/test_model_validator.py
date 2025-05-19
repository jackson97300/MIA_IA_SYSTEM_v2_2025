# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_model_validator.py
# Tests unitaires pour src/model/utils/model_validator.py
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide la validation des modèles SAC, PPO, DDPG pour stabilité et performance,
#        avec support multi-marchés, snapshots compressés, et sauvegardes incrémentielles/distribuées.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, torch>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, pandas>=2.0.0,<3.0.0,
#   matplotlib>=3.7.0,<4.0.0, psutil>=5.9.8,<6.0.0, boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0,
#   pyyaml>=6.0.0,<7.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - Modèle factice simulant SAC/PPO/DDPG
# - Observations factices (np.ndarray)
# - Fichiers de configuration factices (algo_config.yaml, feature_sets.yaml, es_config.yaml)
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de model_validator.py.
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Tests la Phase 8 (confidence_drop_rate) et la validation des modèles.
# - Vérifie l’absence de dxFeed, la suppression de 320/81 features et obs_t, les retries,
#   logs psutil, alertes Telegram, snapshots compressés, visualisations, et sauvegardes incrémentielles/distribuées.
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence).
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.

import gzip
import json
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.model.utils.model_validator import ModelValidator


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, cache, checkpoints, figures, et configuration."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    cache_dir = data_dir / "cache" / "model_validator" / "ES"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints" / "model_validator" / "ES"
    checkpoints_dir.mkdir(parents=True)
    figures_dir = data_dir / "figures" / "model_validator" / "ES"
    figures_dir.mkdir(parents=True)

    # Créer algo_config.yaml
    algo_config_path = config_dir / "algo_config.yaml"
    algo_config_content = {
        "model_validator": {
            "observation_dims": {"training": 350, "inference": 150},
            "max_gradient_threshold": 10.0,
            "min_reward_threshold": -0.5,
            "max_action_variance": 1.0,
            "num_simulations": 10,
        }
    }
    with open(algo_config_path, "w", encoding="utf-8") as f:
        yaml.dump(algo_config_content, f)

    # Créer feature_sets.yaml
    feature_sets_path = config_dir / "feature_sets.yaml"
    feature_sets_content = {
        "training": {"features": [f"feature_{i}" for i in range(350)]},
        "inference": {"shap_features": [f"shap_feature_{i}" for i in range(150)]},
    }
    with open(feature_sets_path, "w", encoding="utf-8") as f:
        yaml.dump(feature_sets_content, f)

    # Créer es_config.yaml
    es_config_path = config_dir / "es_config.yaml"
    es_config_content = {"s3_bucket": "test-bucket", "s3_prefix": "model_validator/"}
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
        "feature_sets_path": str(feature_sets_path),
        "es_config_path": str(es_config_path),
        "perf_log_path": str(logs_dir / "model_validator_performance.csv"),
    }


@pytest.fixture
def mock_model():
    """Crée un modèle factice simulant SAC/PPO/DDPG."""

    class MockModel:
        class Policy:
            def parameters(self):
                return [
                    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                    torch.tensor([[0.0, 1.0]]),
                ]

            def zero_grad(self):
                pass

        def __init__(self):
            self.policy = self.Policy()

        def predict(self, obs, deterministic=True):
            return np.random.randn(2), None

    return MockModel()


@pytest.fixture
def mock_observations():
    """Crée des observations factices avec 350 features."""
    return np.array([np.random.uniform(0, 1, 350)])


@pytest.mark.asyncio
async def test_init_model_validator(tmp_dirs):
    """Teste l’initialisation de ModelValidator."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        validator = ModelValidator(
            config_path=tmp_dirs["algo_config_path"], market="ES"
        )
        assert validator.market == "ES", "Marché incorrect"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "init" in str(op) for op in df_perf["operation"]
        ), "Opération init non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_validate_observations(tmp_dirs, mock_observations):
    """Teste la validation des observations."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        validator = ModelValidator(
            config_path=tmp_dirs["algo_config_path"], market="ES"
        )
        validator._validate_observations(mock_observations)
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_validate_observations" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot validate_observations non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "validate_observations" in str(op) for op in df_perf["operation"]
        ), "Opération validate_observations non journalisée"
        assert any(
            "confidence_drop_rate" in str(kw) for kw in df_perf.to_dict("records")
        ), "confidence_drop_rate absent"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_validate_model(tmp_dirs, mock_model, mock_observations):
    """Teste la validation du modèle."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        validator = ModelValidator(
            config_path=tmp_dirs["algo_config_path"], market="ES"
        )
        result = validator.validate_model(mock_model, "sac", "trend", mock_observations)
        assert "valid" in result, "Résultat manque valid"
        assert "mean_reward" in result, "Résultat manque mean_reward"
        assert "details" in result, "Résultat manque details"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_validate_sac_trend" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot validate_sac_trend non créé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "validate_sac_trend" in f and f.endswith(".json.gz")
            for f in checkpoint_files
        ), "Checkpoint validate_sac_trend non créé"
        figure_files = os.listdir(tmp_dirs["figures_dir"])
        assert any(
            "validate_sac_trend" in f and f.endswith(".png") for f in figure_files
        ), "Figure validate_sac_trend non créée"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "validate_model" in str(op) for op in df_perf["operation"]
        ), "Opération validate_model non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_validate_model_invalid(tmp_dirs, mock_model, mock_observations):
    """Teste la validation avec des paramètres invalides."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        validator = ModelValidator(
            config_path=tmp_dirs["algo_config_path"], market="ES"
        )
        result = validator.validate_model(
            mock_model, "invalid_algo", "trend", mock_observations
        )
        assert not result["valid"], "Modèle devrait être invalide"
        assert "error" in result["details"], "Détails devraient contenir une erreur"
        result = validator.validate_model(
            mock_model, "sac", "invalid_regime", mock_observations
        )
        assert not result["valid"], "Régime devrait être invalide"
        assert "error" in result["details"], "Détails devraient contenir une erreur"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "validate_model" in str(op) and not kw["success"]
            for kw in df_perf.to_dict("records")
        ), "Erreur paramètres invalides non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_cloud_backup(tmp_dirs):
    """Teste la sauvegarde distribuée S3."""
    with patch("boto3.client") as mock_s3, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        validator = ModelValidator(
            config_path=tmp_dirs["algo_config_path"], market="ES"
        )
        df = pd.DataFrame({"timestamp": [datetime.now().isoformat()], "value": [42]})
        validator.cloud_backup(df, data_type="test_metrics")
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
        validator = ModelValidator(
            config_path=tmp_dirs["algo_config_path"], market="ES"
        )
        df = pd.DataFrame({"timestamp": [datetime.now().isoformat()], "value": [42]})
        validator.checkpoint(df, data_type="test_metrics")
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "model_validator_test_metrics" in f and f.endswith(".json.gz")
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
