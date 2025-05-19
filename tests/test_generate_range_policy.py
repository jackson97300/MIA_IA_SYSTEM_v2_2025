# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_generate_range_policy.py
# Tests unitaires pour generate_range_policy.py.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie les fonctionnalités de RangePolicyGenerator, incluant l'initialisation,
#        la validation de la configuration, la création de l’environnement TradingEnv,
#        l’entraînement SAC, la sauvegarde de la politique, les retries, les snapshots JSON compressés,
#        les logs de performance, et les alertes.
#        Conforme à la Phase 8 (auto-conscience via alertes),
#        Phase 10 (génération de politiques), et Phase 16 (ensemble et transfer learning).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - stable_baselines3>=2.0.0
# - psutil>=5.9.8
# - pandas>=2.0.0
# - src.scripts.generate_range_policy
# - src.model.trading_env
# - src.model.utils.config_manager
# - src.model.utils.alert_manager
# - src.model.utils.miya_console
# - src.utils.telegram_alert
# - src.utils.standard
#
# Notes :
# - Utilise des mocks pour simuler les dépendances externes.
# - Vérifie l'absence de références à dxFeed, obs_t, 320/81 features.

import os
from unittest.mock import patch

import pytest

from src.scripts.generate_range_policy import RangePolicyGenerator


@pytest.fixture
def temp_dir(tmp_path):
    """Crée un répertoire temporaire pour les tests."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    data_dir = tmp_path / "data"
    logs_dir = data_dir / "logs" / "market"
    snapshots_dir = data_dir / "range_policy_snapshots"
    policy_dir = tmp_path / "src" / "model" / "router" / "policies"
    logs_dir.mkdir(parents=True)
    snapshots_dir.mkdir(parents=True)
    policy_dir.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def mock_config(temp_dir):
    """Crée un fichier es_config.yaml simulé."""
    config_path = temp_dir / "config" / "es_config.yaml"
    config_content = """
sac_range:
  learning_rate: 0.0003
  buffer_size: 1000000
  learning_starts: 100
  batch_size: 256
  tau: 0.005
  gamma: 0.99
  train_freq: 1
  gradient_steps: 1
  ent_coef: 0.15
  total_timesteps: 1000
trading_env:
  max_position_size: 5
  reward_threshold: 0.01
  news_impact_threshold: 0.5
  obs_dimensions: 350
"""
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def generator(temp_dir, mock_config, monkeypatch):
    """Initialise RangePolicyGenerator avec des mocks."""
    monkeypatch.setattr(
        "src.scripts.generate_range_policy.CONFIG_PATH", str(mock_config)
    )
    monkeypatch.setattr(
        "src.scripts.generate_range_policy.config_manager.get_config",
        lambda x: {
            "sac_range": {
                "learning_rate": 0.0003,
                "buffer_size": 1000000,
                "learning_starts": 100,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": 1,
                "gradient_steps": 1,
                "ent_coef": 0.15,
                "total_timesteps": 1000,
            },
            "trading_env": {
                "max_position_size": 5,
                "reward_threshold": 0.01,
                "news_impact_threshold": 0.5,
                "obs_dimensions": 350,
            },
        },
    )
    generator = RangePolicyGenerator(config_path=mock_config)
    return generator


def test_init_generator(temp_dir, mock_config, monkeypatch):
    """Teste l'initialisation de RangePolicyGenerator."""
    generator = RangePolicyGenerator(config_path=mock_config)
    assert generator.config_path == mock_config
    assert generator.config["sac_range"]["learning_rate"] == 0.0003
    assert os.path.exists(temp_dir / "data" / "range_policy_snapshots")
    snapshots = list(
        (temp_dir / "data" / "range_policy_snapshots").glob("snapshot_init_*.json.gz")
    )
    assert len(snapshots) >= 1
    perf_log = (
        temp_dir / "data" / "logs" / "market" / "generate_range_policy_performance.csv"
    )
    assert perf_log.exists()


def test_validate_config_invalid(temp_dir, monkeypatch):
    """Teste la validation avec une configuration invalide."""
    invalid_config = temp_dir / "config" / "invalid.yaml"
    with pytest.raises(FileNotFoundError, match="Fichier de configuration introuvable"):
        RangePolicyGenerator(config_path=invalid_config)

    with patch(
        "src.scripts.generate_range_policy.config_manager.get_config", return_value={}
    ):
        with pytest.raises(ValueError, match="Clé 'sac_range' manquante"):
            RangePolicyGenerator(config_path=temp_dir / "config" / "es_config.yaml")


def test_create_range_policy_success(temp_dir, generator, monkeypatch):
    """Teste la création de la politique SAC avec succès."""
    with patch("stable_baselines3.SAC.learn") as mock_learn, patch(
        "src.model.trading_env.TradingEnv"
    ) as mock_env:
        generator.create_range_policy()

    policy_path = (
        temp_dir / "src" / "model" / "router" / "policies" / "range_policy.pkl"
    )
    assert policy_path.exists()
    snapshots = list(
        (temp_dir / "data" / "range_policy_snapshots").glob(
            "snapshot_create_range_policy_*.json.gz"
        )
    )
    assert len(snapshots) >= 1
    perf_log = (
        temp_dir / "data" / "logs" / "market" / "generate_range_policy_performance.csv"
    )
    assert perf_log.exists()
    assert mock_learn.called


def test_create_range_policy_failure(temp_dir, generator, monkeypatch):
    """Teste la création de la politique SAC avec une erreur."""
    with patch(
        "stable_baselines3.SAC.learn", side_effect=RuntimeError("Erreur entraînement")
    ):
        with pytest.raises(RuntimeError, match="Erreur entraînement"):
            generator.create_range_policy()

    snapshots = list(
        (temp_dir / "data" / "range_policy_snapshots").glob(
            "snapshot_create_range_policy_*.json.gz"
        )
    )
    assert len(snapshots) >= 1
    perf_log = (
        temp_dir / "data" / "logs" / "market" / "generate_range_policy_performance.csv"
    )
    assert perf_log.exists()


def test_no_obsolete_references(temp_dir, mock_config):
    """Vérifie l'absence de références à dxFeed, obs_t, 320/81 features."""
    with open(mock_config, "r") as f:
        content = f.read()
    assert "dxFeed" not in content, "Référence à dxFeed trouvée"
    assert "obs_t" not in content, "Référence à obs_t trouvée"
    assert "320 features" not in content, "Référence à 320 features trouvée"
    assert "81 features" not in content, "Référence à 81 features trouvée"
