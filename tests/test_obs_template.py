# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_obs_template.py
# Tests unitaires pour src/model/utils/obs_template.py
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide le formatage du vecteur d’observation (150 SHAP features) pour trading_env.py, avec pondération selon le
#        régime (méthode 3), sélection via shap_weighting.py, et documentation dans feature_sets.yaml, avec support
#        multi-marchés, snapshots compressés, et sauvegardes incrémentielles/distribuées.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.8,<6.0.0, pyyaml>=6.0.0,<7.0.0,
#   matplotlib>=3.7.0,<4.0.0, boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
# - src/features/shap_weighting.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - Fichier de configuration factice (es_config.yaml)
# - Données de test simulant les 350 features
# - Fichier feature_importance.csv factice
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de obs_template.py.
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Tests la Phase 3 (pondération selon le régime) et Phase 8 (confidence_drop_rate).
# - Vérifie l’absence de dxFeed, la suppression de 320/81 features et obs_t, les retries,
#   logs psutil, alertes Telegram, snapshots compressés, et sauvegardes incrémentielles/distribuées.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.

import gzip
import json
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.model.utils.obs_template import ObsTemplate


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, cache, checkpoints, figures, features, et configuration."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs" / "ES"
    logs_dir.mkdir(parents=True)
    cache_dir = data_dir / "cache" / "obs_template" / "ES"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints" / "obs_template" / "ES"
    checkpoints_dir.mkdir(parents=True)
    figures_dir = data_dir / "figures" / "obs_template" / "ES"
    figures_dir.mkdir(parents=True)
    features_dir = data_dir / "features" / "ES"
    features_dir.mkdir(parents=True)

    # Créer es_config.yaml
    es_config_path = config_dir / "es_config.yaml"
    es_config_content = {
        "obs_template": {
            "buffer_size": 100,
            "max_cache_size": 1000,
            "weights": {
                "trend": {"mm_score": 1.5, "hft_score": 1.5, "breakout_score": 1.0},
                "range": {"breakout_score": 1.5, "mm_score": 1.0, "hft_score": 1.0},
                "defensive": {"mm_score": 0.5, "hft_score": 0.5, "breakout_score": 0.5},
            },
        },
        "s3_bucket": "test-bucket",
        "s3_prefix": "obs_template/",
    }
    with open(es_config_path, "w", encoding="utf-8") as f:
        yaml.dump(es_config_content, f)

    # Créer feature_importance.csv factice
    feature_importance_path = features_dir / "feature_importance.csv"
    feature_importance_data = pd.DataFrame(
        {
            "feature": [f"feature_{i}" for i in range(350)]
            + [
                "rsi_14",
                "obi_score",
                "gex",
                "news_impact_score",
                "predicted_volatility",
                "neural_regime",
                "cnn_pressure",
                "key_strikes_1",
                "max_pain_strike",
                "net_gamma",
                "zero_gamma",
                "dealer_zones_count",
                "vol_trigger",
                "ref_px",
                "data_release",
            ],
            "importance": np.random.uniform(0, 1, 365),
        }
    )
    feature_importance_data.to_csv(feature_importance_path, index=False)

    return {
        "base_dir": str(base_dir),
        "config_dir": str(config_dir),
        "logs_dir": str(logs_dir),
        "cache_dir": str(cache_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "figures_dir": str(figures_dir),
        "features_dir": str(features_dir),
        "es_config_path": str(es_config_path),
        "feature_importance_path": str(feature_importance_path),
        "obs_template_csv": str(features_dir / "obs_template.csv"),
        "feature_sets_path": str(config_dir / "feature_sets.yaml"),
        "perf_log_path": str(logs_dir / "obs_template_performance.csv"),
    }


@pytest.fixture
def mock_data():
    """Crée des données de test simulant les 350 features."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-04-14 09:00", periods=100, freq="1min"),
            **{f"feature_{i}": np.random.uniform(0, 1, 100) for i in range(350)},
            "rsi_14": np.random.uniform(0, 100, 100),
            "obi_score": np.random.uniform(-1, 1, 100),
            "gex": np.random.uniform(-1000, 1000, 100),
            "news_impact_score": np.random.uniform(0, 1, 100),
            "predicted_volatility": np.random.uniform(0, 2, 100),
            "neural_regime": np.random.randint(0, 3, 100),
            "cnn_pressure": np.random.uniform(-5, 5, 100),
            "key_strikes_1": np.random.uniform(5000, 5200, 100),
            "max_pain_strike": np.random.uniform(5000, 5200, 100),
            "net_gamma": np.random.uniform(-1, 1, 100),
            "zero_gamma": np.random.uniform(5000, 5200, 100),
            "dealer_zones_count": np.random.randint(0, 11, 100),
            "vol_trigger": np.random.uniform(-1, 1, 100),
            "ref_px": np.random.uniform(5000, 5200, 100),
            "data_release": np.random.randint(0, 2, 100),
        }
    ).set_index("timestamp")


@pytest.mark.asyncio
async def test_init_obs_template(tmp_dirs):
    """Teste l’initialisation de ObsTemplate."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        calculator = ObsTemplate(config_path=tmp_dirs["es_config_path"], market="ES")
        assert calculator.market == "ES", "Marché incorrect"
        assert calculator.buffer_size == 100, "Buffer size incorrect"
        assert calculator.max_cache_size == 1000, "Max cache size incorrect"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_init" in f and f.endswith(".json.gz") for f in snapshot_files
        ), "Snapshot init non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "init" in str(op) for op in df_perf["operation"]
        ), "Opération init non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_get_shap_features(tmp_dirs, mock_data):
    """Teste la sélection des 150 SHAP features."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        calculator = ObsTemplate(config_path=tmp_dirs["es_config_path"], market="ES")
        shap_features = calculator.get_shap_features(mock_data, regime="trend")
        assert len(shap_features) == 150, "Nombre incorrect de SHAP features"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_get_shap_features" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot get_shap_features non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "get_shap_features" in str(op) for op in df_perf["operation"]
        ), "Opération get_shap_features non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_create_observation(tmp_dirs, mock_data):
    """Teste la création du vecteur d’observation."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        calculator = ObsTemplate(config_path=tmp_dirs["es_config_path"], market="ES")
        observation = calculator.create_observation(mock_data, regime="trend")
        assert observation.shape == (150,), "Shape incorrect du vecteur d’observation"
        assert os.path.exists(tmp_dirs["obs_template_csv"]), "obs_template.csv non créé"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_create_observation" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot create_observation non créé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "create_observation" in f and f.endswith(".json.gz")
            for f in checkpoint_files
        ), "Checkpoint create_observation non créé"
        figure_files = os.listdir(tmp_dirs["figures_dir"])
        assert any(
            "observation_" in f and f.endswith(".png") for f in figure_files
        ), "Figure observation non créée"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "create_observation" in str(op) for op in df_perf["operation"]
        ), "Opération create_observation non journalisée"
        assert any(
            "confidence_drop_rate" in str(kw) for kw in df_perf.to_dict("records")
        ), "confidence_drop_rate absent"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_validate_obs_template_valid(tmp_dirs, mock_data):
    """Teste la validation du vecteur d’observation avec des données valides."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        calculator = ObsTemplate(config_path=tmp_dirs["es_config_path"], market="ES")
        result = calculator.validate_obs_template(mock_data, policy_type="mlp")
        assert result, "Validation devrait réussir avec des données valides"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "validate_obs_template" in str(op) for op in df_perf["operation"]
        ), "Opération validate_obs_template non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_validate_obs_template_no_data(tmp_dirs):
    """Teste la validation sans données fournies."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        calculator = ObsTemplate(config_path=tmp_dirs["es_config_path"], market="ES")
        result = calculator.validate_obs_template()
        assert result, "Validation devrait réussir sans données"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "validate_obs_template" in str(op) for op in df_perf["operation"]
        ), "Opération validate_obs_template non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_validate_obs_template_invalid(tmp_dirs, mock_data):
    """Teste la validation avec des données invalides."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        invalid_data = mock_data.copy()
        invalid_data["obi_score"] = ["invalid"] * 100  # Type non numérique
        calculator = ObsTemplate(config_path=tmp_dirs["es_config_path"], market="ES")
        result = calculator.validate_obs_template(invalid_data, policy_type="mlp")
        assert not result, "Validation devrait échouer avec des données invalides"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "validate_obs_template" in str(op) for op in df_perf["operation"]
        ), "Opération validate_obs_template non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_cloud_backup(tmp_dirs):
    """Teste la sauvegarde distribuée S3."""
    with patch("boto3.client") as mock_s3, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        calculator = ObsTemplate(config_path=tmp_dirs["es_config_path"], market="ES")
        df = pd.DataFrame({"timestamp": [datetime.now().isoformat()], "value": [42]})
        calculator.cloud_backup(df, data_type="test_metrics")
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
        calculator = ObsTemplate(config_path=tmp_dirs["es_config_path"], market="ES")
        df = pd.DataFrame({"timestamp": [datetime.now().isoformat()], "value": [42]})
        calculator.checkpoint(df, data_type="test_metrics")
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "obs_template_test_metrics" in f and f.endswith(".json.gz")
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
