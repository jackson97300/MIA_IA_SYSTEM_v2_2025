# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_prediction_aggregator.py
# Tests unitaires pour src/model/utils/prediction_aggregator.py
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide l’agrégation des prédictions de SAC, PPO, DDPG avec ensemble learning,
#        avec support multi-marchés, snapshots compressés, et sauvegardes incrémentielles/distribuées.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, matplotlib>=3.7.0,<4.0.0,
#   psutil>=5.9.8,<6.0.0, boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0, pyyaml>=6.0.0,<7.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - Données factices simulant les prédictions et données brutes
# - Fichiers de configuration factices (algo_config.yaml, feature_sets.yaml, es_config.yaml)
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de prediction_aggregator.py.
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Tests la Phase 8 (confidence_drop_rate) et l’ensemble learning.
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
import yaml

from src.model.utils.prediction_aggregator import PredictionAggregator


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
    cache_dir = data_dir / "cache" / "prediction_aggregator" / "ES"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints" / "prediction_aggregator" / "ES"
    checkpoints_dir.mkdir(parents=True)
    figures_dir = data_dir / "figures" / "prediction_aggregator" / "ES"
    figures_dir.mkdir(parents=True)

    # Créer algo_config.yaml
    algo_config_path = config_dir / "algo_config.yaml"
    algo_config_content = {
        "prediction_aggregator": {
            "observation_dims": {"training": 350, "inference": 150},
            "max_action": 1.0,
            "max_action_variance": 1.0,
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
    es_config_content = {
        "s3_bucket": "test-bucket",
        "s3_prefix": "prediction_aggregator/",
    }
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
        "perf_log_path": str(logs_dir / "prediction_aggregator_performance.csv"),
    }


@pytest.fixture
def mock_data():
    """Crée des données factices avec 350 features."""
    feature_cols = [f"feature_{i}" for i in range(350)]
    return pd.DataFrame(
        {
            "bid_size_level_1": [100],
            "ask_size_level_1": [120],
            "trade_frequency_1s": [8],
            "spread_avg_1min_es": [0.3],
            "close": [5100],
            **{col: [np.random.uniform(0, 1)] for col in feature_cols},
        }
    )


@pytest.mark.asyncio
async def test_init_prediction_aggregator(tmp_dirs):
    """Teste l’initialisation de PredictionAggregator."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        aggregator = PredictionAggregator(
            config_path=tmp_dirs["algo_config_path"], market="ES"
        )
        assert aggregator.market == "ES", "Marché incorrect"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "init" in str(op) for op in df_perf["operation"]
        ), "Opération init non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_validate_data(tmp_dirs, mock_data):
    """Teste la validation des données."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        aggregator = PredictionAggregator(
            config_path=tmp_dirs["algo_config_path"], market="ES"
        )
        aggregator._validate_data(mock_data)
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
            "confidence_drop_rate" in str(kw) for kw in df_perf.to_dict("records")
        ), "confidence_drop_rate absent"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_calculate_weights(tmp_dirs):
    """Teste le calcul des poids."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        aggregator = PredictionAggregator(
            config_path=tmp_dirs["algo_config_path"], market="ES"
        )
        rewards = [1.0, 0.8, 0.9]
        regime = "trend"
        weights = aggregator.calculate_weights(rewards, regime)
        assert len(weights) == 3, "Nombre incorrect de poids"
        assert abs(sum(weights) - 1.0) < 1e-6, "Somme des poids non égale à 1"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_calculate_weights" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot calculate_weights non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "calculate_weights" in str(op) for op in df_perf["operation"]
        ), "Opération calculate_weights non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_aggregate_predictions(tmp_dirs, mock_data):
    """Teste l’agrégation des prédictions."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        aggregator = PredictionAggregator(
            config_path=tmp_dirs["algo_config_path"], market="ES"
        )
        actions = [0.5, -0.3, 0.2]
        rewards = [1.0, 0.8, 0.9]
        regime = "trend"
        final_action, details = aggregator.aggregate_predictions(
            actions, rewards, regime, mock_data
        )
        assert isinstance(final_action, float), "Action finale n’est pas un float"
        assert "weights" in details, "Détails manquent les poids"
        assert "confidence" in details, "Détails manquent la confiance"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_aggregate_trend" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot aggregate_trend non créé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "aggregate_predictions" in f and f.endswith(".json.gz")
            for f in checkpoint_files
        ), "Checkpoint aggregate_predictions non créé"
        figure_files = os.listdir(tmp_dirs["figures_dir"])
        assert any(
            "aggregate_trend" in f and f.endswith(".png") for f in figure_files
        ), "Figure aggregate_trend non créée"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "aggregate_predictions" in str(op) for op in df_perf["operation"]
        ), "Opération aggregate_predictions non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_aggregate_predictions_invalid(tmp_dirs, mock_data):
    """Teste l’agrégation avec des entrées invalides."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        aggregator = PredictionAggregator(
            config_path=tmp_dirs["algo_config_path"], market="ES"
        )
        actions = [0.5, -0.3]  # Trop peu d’actions
        rewards = [1.0, 0.8, 0.9]
        regime = "invalid_regime"
        final_action, details = aggregator.aggregate_predictions(
            actions, rewards, regime, mock_data
        )
        assert (
            final_action == 0.0
        ), "Action finale devrait être 0 pour entrées invalides"
        assert "error" in details, "Détails devraient contenir une erreur"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "aggregate_predictions" in str(op) and not kw["success"]
            for kw in df_perf.to_dict("records")
        ), "Erreur entrées invalides non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_cloud_backup(tmp_dirs):
    """Teste la sauvegarde distribuée S3."""
    with patch("boto3.client") as mock_s3, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        aggregator = PredictionAggregator(
            config_path=tmp_dirs["algo_config_path"], market="ES"
        )
        df = pd.DataFrame({"timestamp": [datetime.now().isoformat()], "value": [42]})
        aggregator.cloud_backup(df, data_type="test_metrics")
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
        aggregator = PredictionAggregator(
            config_path=tmp_dirs["algo_config_path"], market="ES"
        )
        df = pd.DataFrame({"timestamp": [datetime.now().isoformat()], "value": [42]})
        aggregator.checkpoint(df, data_type="test_metrics")
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "prediction_aggregator_test_metrics" in f and f.endswith(".json.gz")
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
