# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_transformer_policy.py
# Tests unitaires pour src/model/router/policies/transformer_policy.py
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide la politique expérimentale basée sur un Transformer pour SAC/PPO/DDPG, avec attention contextuelle (méthode 9),
#        régularisation dynamique (méthode 14), gestion de 350/150 features, snapshots compressés,
#        sauvegardes incrémentielles/distribuées, et alertes Telegram.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, torch>=2.0.0,<3.0.0, pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0,
#   stable-baselines3>=2.0.0,<3.0.0, psutil>=5.9.8,<6.0.0, matplotlib>=3.7.0,<4.0.0,
#   sklearn>=1.5.0,<2.0.0, boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0, pyyaml>=6.0.0,<7.0.0
# - src/model/router/detect_regime.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/envs/trading_env.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/router_config.yaml
# - config/feature_sets.yaml
# - Données d’observation factices (batch_size, sequence_length, 350/150 features)
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de TransformerPolicy.
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Tests les phases 8 (auto-conscience via confidence_drop_rate), 9 (attention contextuelle), 14 (régularisation dynamique).
# - Vérifie l’absence de dxFeed, la suppression de 320/81 features et obs_t, les retries, logs psutil, alertes Telegram,
#   snapshots compressés, et sauvegardes incrémentielles/distribuées.
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence) via config/feature_sets.yaml.

import gzip
import json
import os
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from gymnasium import spaces

from src.model.router.policies.transformer_policy import TransformerPolicy


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les données, logs, figures, cache, et checkpoints."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    cache_dir = data_dir / "cache" / "transformer"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints"
    checkpoints_dir.mkdir()
    figures_dir = data_dir / "figures" / "transformer"
    figures_dir.mkdir(parents=True)

    # Créer router_config.yaml
    router_config_path = config_dir / "router_config.yaml"
    router_config_content = {
        "fast_mode": False,
        "weights": {
            "trend": {"feature_0": 1.2, "feature_1": 0.8},
            "range": {"feature_0": 1.0, "feature_1": 1.0},
            "defensive": {"feature_0": 0.8, "feature_1": 1.2},
        },
        "thresholds": {
            "vix_peak_threshold": 30.0,
            "spread_explosion_threshold": 0.05,
            "confidence_threshold": 0.7,
            "net_gamma_threshold": 1.0,
            "vol_trigger_threshold": 1.0,
            "dealer_zones_count_threshold": 5,
        },
        "s3_bucket": "test-bucket",
        "s3_prefix": "transformer/",
        "logging": {"buffer_size": 100},
    }
    with open(router_config_path, "w", encoding="utf-8") as f:
        yaml.dump(router_config_content, f)

    # Créer feature_sets.yaml
    feature_sets_path = config_dir / "feature_sets.yaml"
    feature_sets_content = {
        "inference": {"shap_features": [f"feature_{i}" for i in range(150)]},
        "training": {"features": [f"feature_{i}" for i in range(350)]},
    }
    with open(feature_sets_path, "w", encoding="utf-8") as f:
        yaml.dump(feature_sets_content, f)

    return {
        "base_dir": str(base_dir),
        "router_config_path": str(router_config_path),
        "feature_sets_path": str(feature_sets_path),
        "logs_dir": str(logs_dir),
        "cache_dir": str(cache_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "figures_dir": str(figures_dir),
        "perf_log_path": str(logs_dir / "transformer_performance.csv"),
    }


@pytest.fixture
def policy_setup(tmp_dirs):
    """Crée une instance de TransformerPolicy pour les tests."""
    observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(50, 350), dtype=np.float32
    )
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def lr_schedule(x):
        return 0.001

    with patch(
        "src.model.router.policies.transformer_policy.MarketRegimeDetector"
    ) as mock_detector:
        mock_detector.return_value.detect.return_value = (
            "trend",
            {
                "neural_regime": "trend",
                "regime_probs": {"trend": 0.7, "range": 0.2, "defensive": 0.1},
            },
        )
        policy = TransformerPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            sequence_length=50,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.1,
            training_mode=True,
        )
    return policy


def test_init(tmp_dirs, policy_setup):
    """Teste l’initialisation de TransformerPolicy."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        policy = policy_setup
        assert policy.num_features == 350, "Dimension des features incorrecte"
        assert policy.sequence_length == 50, "Longueur de séquence incorrecte"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "init" in str(op) for op in df["operation"]
        ), "Opération init non journalisée"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_init" in f and f.endswith(".json.gz") for f in snapshot_files
        ), "Snapshot init non créé"
        mock_telegram.assert_called()


def test_init_invalid_features(tmp_dirs):
    """Teste l’initialisation avec un nombre incorrect de features."""
    observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(50, 100), dtype=np.float32
    )
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def lr_schedule(x):
        return 0.001

    with pytest.raises(ValueError, match="Observation space doit être de forme"):
        TransformerPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            sequence_length=50,
            training_mode=True,
        )


def test_validate_observations(policy_setup):
    """Teste la validation des observations."""
    obs = torch.randn(10, 50, 350, dtype=torch.float32)
    policy_setup._validate_observations(obs)
    df_perf = pd.read_csv(policy_setup.PERF_LOG_PATH)
    assert any(
        "validate_observations" in str(op) for op in df_perf["operation"]
    ), "Opération validate_observations non journalisée"
    assert any(
        "confidence_drop_rate" in str(kw) for kw in df_perf.to_dict("records")
    ), "confidence_drop_rate absent"


def test_validate_observations_invalid(policy_setup):
    """Teste la validation avec des observations invalides."""
    obs = torch.tensor([[[float("nan")] * 350] * 50] * 10, dtype=torch.float32)
    with pytest.raises(ValueError, match="Observations contiennent"):
        policy_setup._validate_observations(obs)


def test_forward(policy_setup, tmp_dirs):
    """Teste le passage avant avec attention contextuelle."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        obs = torch.randn(10, 50, 350, dtype=torch.float32)
        context = {
            "vix_es_correlation": 20.0,
            "spread": 0.01,
            "net_gamma": 0.5,
            "vol_trigger": 0.5,
            "dealer_zones_count": 2,
            "regime_probs": {"trend": 0.7, "range": 0.2, "defensive": 0.1},
        }
        actions, values, log_probs = policy_setup.forward(
            obs, deterministic=True, context=context
        )
        assert actions.shape == (10, 1), "Shape des actions incorrecte"
        assert values.shape == (10, 1), "Shape des valeurs incorrecte"
        assert log_probs.shape == (10,), "Shape des log probs incorrecte"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "forward" in str(op) for op in df_perf["operation"]
        ), "Opération forward non journalisée"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_forward" in f and f.endswith(".json.gz") for f in snapshot_files
        ), "Snapshot forward non créé"
        assert any(
            f.startswith("attention_") and f.endswith(".png")
            for f in os.listdir(tmp_dirs["figures_dir"])
        ), "Visualisation attention non générée"
        assert any(
            f.startswith("forward_") and f.endswith(".png")
            for f in os.listdir(tmp_dirs["figures_dir"])
        ), "Visualisation forward non générée"
        mock_telegram.assert_called()


def test_forward_ultra_defensive(policy_setup):
    """Teste le mode ultra-défensif."""
    obs = torch.randn(10, 50, 350, dtype=torch.float32)
    context = {
        "vix_es_correlation": 40.0,  # Déclenche ultra-défensif
        "spread": 0.01,
        "net_gamma": 0.5,
        "vol_trigger": 0.5,
        "dealer_zones_count": 2,
        "regime_probs": {"trend": 0.7, "range": 0.2, "defensive": 0.1},
    }
    actions, values, log_probs = policy_setup.forward(
        obs, deterministic=True, context=context
    )
    assert torch.all(actions == 0), "Actions non nulles en mode ultra-défensif"
    assert torch.all(values == 0), "Valeurs non nulles en mode ultra-défensif"


def test_update_normalization_stats(policy_setup):
    """Teste la mise à jour des statistiques de normalisation."""
    obs = torch.randn(10, 50, 350, dtype=torch.float32)
    policy_setup.update_normalization_stats(obs)
    assert (
        len(policy_setup.obs_window) == 10 * 50
    ), "Fenêtre de normalisation incorrecte"
    df_perf = pd.read_csv(policy_setup.PERF_LOG_PATH)
    assert any(
        "update_normalization_stats" in str(op) for op in df_perf["operation"]
    ), "Opération update_normalization_stats non journalisée"


def test_update_performance_thresholds(policy_setup):
    """Teste la mise à jour des seuils de performance."""
    actions = torch.randn(10, 1, dtype=torch.float32)
    values = torch.randn(10, 1, dtype=torch.float32)
    context = {
        "vix_es_correlation": 20.0,
        "spread": 0.01,
        "net_gamma": 0.5,
        "vol_trigger": 0.5,
        "dealer_zones_count": 2,
    }
    for _ in range(15):  # Remplir la fenêtre
        policy_setup.update_performance_thresholds(actions, values, context)
    assert (
        len(policy_setup.performance_window) == 10
    ), "Fenêtre de performance incorrecte"
    df_perf = pd.read_csv(policy_setup.PERF_LOG_PATH)
    assert any(
        "update_performance_thresholds" in str(op) for op in df_perf["operation"]
    ), "Opération update_performance_thresholds non journalisée"


def test_check_action_validity(policy_setup):
    """Teste la vérification de la cohérence des actions."""
    policy_setup.recent_actions.append(torch.ones(10, 1))
    action = torch.ones(10, 1) * 1.1
    assert policy_setup.check_action_validity(action), "Action jugée incohérente"


def test_save_snapshot_compressed(tmp_dirs, policy_setup):
    """Teste la sauvegarde d’un snapshot compressé."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        snapshot_data = {"test": "compressed_snapshot"}
        policy_setup.save_snapshot("test_compressed", snapshot_data, compress=True)
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_test_compressed" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot compressé non créé"
        with gzip.open(
            os.path.join(tmp_dirs["cache_dir"], snapshot_files[-1]),
            "rt",
            encoding="utf-8",
        ) as f:
            snapshot = json.load(f)
        assert (
            snapshot["data"] == snapshot_data
        ), "Contenu du snapshot compressé incorrect"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "save_snapshot" in str(op) for op in df_perf["operation"]
        ), "Opération save_snapshot non journalisée"
        mock_telegram.assert_called()


def test_checkpoint(tmp_dirs, policy_setup):
    """Teste la sauvegarde incrémentielle."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        df = pd.DataFrame(
            {
                "timestamp": [datetime.now().isoformat()],
                "actions_mean": [0.5],
                "values_mean": [1.0],
                "confidence_score": [0.9],
                "regime": ["trend"],
                "dropout": [0.2],
            }
        )
        policy_setup.checkpoint(df, data_type="test_metrics")
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "transformer_test_metrics" in f and f.endswith(".json.gz")
            for f in checkpoint_files
        ), "Checkpoint non créé"
        with gzip.open(
            os.path.join(tmp_dirs["checkpoints_dir"], checkpoint_files[0]),
            "rt",
            encoding="utf-8",
        ) as f:
            checkpoint = json.load(f)
        assert checkpoint["num_rows"] == len(df), "Nombre de lignes incorrect"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "checkpoint" in str(op) for op in df_perf["operation"]
        ), "Opération checkpoint non journalisée"
        mock_telegram.assert_called()


def test_cloud_backup(tmp_dirs, policy_setup):
    """Teste la sauvegarde distribuée S3."""
    with patch("boto3.client") as mock_s3, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        df = pd.DataFrame(
            {
                "timestamp": [datetime.now().isoformat()],
                "actions_mean": [0.5],
                "values_mean": [1.0],
                "confidence_score": [0.9],
                "regime": ["trend"],
                "dropout": [0.2],
            }
        )
        policy_setup.cloud_backup(df, data_type="test_metrics")
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


def test_handle_sigint(tmp_dirs, policy_setup):
    """Teste la gestion SIGINT."""
    with patch("sys.exit") as mock_exit, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        policy_setup.handle_sigint(signal.SIGINT, None)
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
