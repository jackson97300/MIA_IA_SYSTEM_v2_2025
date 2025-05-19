# Teste model_params.yaml.
# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_model_params.py
# Tests unitaires pour la configuration model_params.yaml.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie les paramètres des modèles SAC et du pipeline neuronal dans model_params.yaml,
#        incluant les métadonnées, les valeurs, les plages, et la cohérence pour default, neural_pipeline,
#        contextual_state_encoder, et modes (trend, range, defensive).
#        Conforme à la Phase 8 (auto-conscience), Phase 10 (SAC et pipeline neuronal),
#        et Phase 16 (ensemble et transfer learning).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pyyaml>=6.0.0,<7.0.0
# - src/model/utils/config_manager.py
#
# Notes :
# - Utilise des données simulées pour tester la configuration.
# - Suppression de toute référence à dxFeed, obs_t, 320 features, et 81 features.
# - Vérifie l'alignement avec 350 features pour l’entraînement et 150 SHAP pour l’inférence.

import pytest
import yaml

from src.model.utils.config_manager import ConfigManager


@pytest.fixture
def model_params_config(tmp_path):
    """Crée un fichier model_params.yaml temporaire pour les tests."""
    config_path = tmp_path / "model_params.yaml"
    config_content = """
metadata:
  version: "2.1.3"
  updated: "2025-05-13"
default:
  learning_rate:
    value: 0.0001
    range: [0.00001, 0.01]
  buffer_size:
    value: 100000
    range: [10000, 1000000]
  batch_size:
    value: 256
    range: [32, 1024]
  tau:
    value: 0.02
    range: [0.005, 0.1]
  gamma:
    value: 0.98
    range: [0.9, 0.999]
  verbose:
    value: 1
    range: [0, 2]
  tensorboard_log:
    value: "data/logs/tensorboard"
  train_freq:
    value: 1
    range: [1, 10]
  gradient_steps:
    value: 1
    range: [1, 10]
  ent_coef:
    value: "auto"
  target_entropy:
    value: "auto"
  log_interval:
    value: 10
    range: [1, 100]
  news_impact_threshold:
    value: 0.5
    range: [0.0, 1.0]
  vix_threshold:
    value: 20.0
    range: [10.0, 50.0]
neural_pipeline:
  window_size:
    value: 50
    range: [20, 200]
  base_features:
    value: 350
    range: [150, 400]
  lstm:
    units:
      value: 128
      range: [64, 256]
    hidden_layers:
      value: [64, 20]
    dropout:
      value: 0.2
      range: [0.0, 0.5]
    learning_rate:
      value: 0.001
      range: [0.0001, 0.01]
    optimizer:
      value: "adam"
  cnn:
    filters:
      value: 32
      range: [16, 128]
    kernel_size:
      value: 5
      range: [3, 10]
    hidden_layers:
      value: [16, 1]
    dropout:
      value: 0.1
      range: [0.0, 0.3]
    learning_rate:
      value: 0.001
      range: [0.0001, 0.01]
    optimizer:
      value: "adam"
  mlp_volatility:
    units:
      value: 128
      range: [64, 256]
    hidden_layers:
      value: [64, 1]
    activation:
      value: "relu"
    learning_rate:
      value: 0.001
      range: [0.0001, 0.01]
    optimizer:
      value: "adam"
  mlp_regime:
    units:
      value: 128
      range: [64, 256]
    hidden_layers:
      value: [64, 3]
    activation:
      value: "relu"
    learning_rate:
      value: 0.001
      range: [0.0001, 0.01]
    optimizer:
      value: "adam"
  normalization:
    value: true
  batch_size:
    value: 32
    range: [16, 128]
  pretrain_epochs:
    value: 5
    range: [1, 20]
  validation_split:
    value: 0.2
    range: [0.1, 0.3]
contextual_state_encoder:
  tsne_perplexity:
    value: 30
    range: [5, 50]
  tsne_learning_rate:
    value: 200
    range: [10, 1000]
  tsne_n_components:
    value: 2
    range: [1, 3]
  tsne_n_iter:
    value: 1000
    range: [250, 5000]
modes:
  trend:
    learning_rate:
      value: 0.0001
      range: [0.00001, 0.01]
    buffer_size:
      value: 100000
      range: [10000, 1000000]
    batch_size:
      value: 256
      range: [32, 1024]
    tau:
      value: 0.02
      range: [0.005, 0.1]
    gamma:
      value: 0.98
      range: [0.9, 0.999]
    verbose:
      value: 1
      range: [0, 2]
    tensorboard_log:
      value: "data/logs/tensorboard/trend"
    train_freq:
      value: 1
      range: [1, 10]
    gradient_steps:
      value: 1
      range: [1, 10]
    policy_kwargs:
      net_arch:
        value: [256, 256]
    ent_coef:
      value: "auto"
    target_entropy:
      value: "auto"
  range:
    learning_rate:
      value: 0.0001
      range: [0.00001, 0.01]
    buffer_size:
      value: 100000
      range: [10000, 1000000]
    batch_size:
      value: 256
      range: [32, 1024]
    tau:
      value: 0.02
      range: [0.005, 0.1]
    gamma:
      value: 0.98
      range: [0.9, 0.999]
    verbose:
      value: 1
      range: [0, 2]
    tensorboard_log:
      value: "data/logs/tensorboard/range"
    train_freq:
      value: 1
      range: [1, 10]
    gradient_steps:
      value: 1
      range: [1, 10]
    policy_kwargs:
      net_arch:
        value: [128, 128]
    ent_coef:
      value: "auto"
    target_entropy:
      value: "auto"
  defensive:
    learning_rate:
      value: 0.00005
      range: [0.00001, 0.001]
    buffer_size:
      value: 50000
      range: [10000, 200000]
    batch_size:
      value: 128
      range: [32, 512]
    tau:
      value: 0.03
      range: [0.01, 0.1]
    gamma:
      value: 0.99
      range: [0.95, 0.999]
    verbose:
      value: 1
      range: [0, 2]
    tensorboard_log:
      value: "data/logs/tensorboard/defensive"
    train_freq:
      value: 2
      range: [1, 10]
    gradient_steps:
      value: 2
      range: [1, 10]
    policy_kwargs:
      net_arch:
        value: [64, 64]
    ent_coef:
      value: 0.01
      range: [0.0, 0.1]
    target_entropy:
      value: -1.0
      range: [-10.0, 0.0]
"""
    config_path.write_text(config_content)
    return config_path


def test_metadata(model_params_config):
    """Teste les métadonnées de model_params.yaml."""
    config = yaml.safe_load(model_params_config.read_text())
    assert config["metadata"]["version"] == "2.1.3", "Version incorrecte"
    assert config["metadata"]["updated"] == "2025-05-13", "Date incorrecte"
    assert "Phase 8" in config["metadata"]["description"], "Phase 8 non mentionnée"
    assert "Phase 10" in config["metadata"]["description"], "Phase 10 non mentionnée"
    assert "Phase 16" in config["metadata"]["description"], "Phase 16 non mentionnée"
    assert (
        "350 features" in config["metadata"]["description"]
    ), "Alignement 350 features non mentionné"


def test_default_params(model_params_config):
    """Teste les paramètres globaux par défaut."""
    config = yaml.safe_load(model_params_config.read_text())
    default = config["default"]
    assert default["learning_rate"]["value"] == 0.0001, "learning_rate incorrect"
    assert default["buffer_size"]["value"] == 100000, "buffer_size incorrect"
    assert default["batch_size"]["value"] == 256, "batch_size incorrect"
    assert default["tau"]["value"] == 0.02, "tau incorrect"
    assert default["gamma"]["value"] == 0.98, "gamma incorrect"
    assert default["verbose"]["value"] == 1, "verbose incorrect"
    assert (
        default["tensorboard_log"]["value"] == "data/logs/tensorboard"
    ), "tensorboard_log incorrect"
    assert (
        default["news_impact_threshold"]["value"] == 0.5
    ), "news_impact_threshold incorrect"
    assert default["vix_threshold"]["value"] == 20.0, "vix_threshold incorrect"


def test_neural_pipeline(model_params_config):
    """Teste les paramètres du pipeline neuronal."""
    config = yaml.safe_load(model_params_config.read_text())
    neural = config["neural_pipeline"]
    assert neural["window_size"]["value"] == 50, "window_size incorrect"
    assert neural["base_features"]["value"] == 350, "base_features incorrect"
    assert neural["lstm"]["units"]["value"] == 128, "lstm.units incorrect"
    assert neural["lstm"]["hidden_layers"]["value"] == [
        64,
        20,
    ], "lstm.hidden_layers incorrect"
    assert neural["cnn"]["filters"]["value"] == 32, "cnn.filters incorrect"
    assert neural["cnn"]["kernel_size"]["value"] == 5, "cnn.kernel_size incorrect"
    assert (
        neural["mlp_volatility"]["units"]["value"] == 128
    ), "mlp_volatility.units incorrect"
    assert neural["mlp_regime"]["hidden_layers"]["value"] == [
        64,
        3,
    ], "mlp_regime.hidden_layers incorrect"
    assert neural["normalization"]["value"] is True, "normalization incorrect"
    assert neural["batch_size"]["value"] == 32, "batch_size incorrect"


def test_contextual_state_encoder(model_params_config):
    """Teste les paramètres de contextual_state_encoder."""
    config = yaml.safe_load(model_params_config.read_text())
    encoder = config["contextual_state_encoder"]
    assert encoder["tsne_perplexity"]["value"] == 30, "tsne_perplexity incorrect"
    assert encoder["tsne_learning_rate"]["value"] == 200, "tsne_learning_rate incorrect"
    assert encoder["tsne_n_components"]["value"] == 2, "tsne_n_components incorrect"
    assert encoder["tsne_n_iter"]["value"] == 1000, "tsne_n_iter incorrect"


def test_modes(model_params_config):
    """Teste les hyperparamètres SAC par régime."""
    config = yaml.safe_load(model_params_config.read_text())
    modes = config["modes"]
    # Trend
    trend = modes["trend"]
    assert trend["learning_rate"]["value"] == 0.0001, "trend.learning_rate incorrect"
    assert trend["buffer_size"]["value"] == 100000, "trend.buffer_size incorrect"
    assert trend["policy_kwargs"]["net_arch"]["value"] == [
        256,
        256,
    ], "trend.net_arch incorrect"
    # Range
    range_mode = modes["range"]
    assert range_mode["batch_size"]["value"] == 256, "range.batch_size incorrect"
    assert range_mode["policy_kwargs"]["net_arch"]["value"] == [
        128,
        128,
    ], "range.net_arch incorrect"
    # Defensive
    defensive = modes["defensive"]
    assert (
        defensive["learning_rate"]["value"] == 0.00005
    ), "defensive.learning_rate incorrect"
    assert defensive["ent_coef"]["value"] == 0.01, "defensive.ent_coef incorrect"
    assert (
        defensive["target_entropy"]["value"] == -1.0
    ), "defensive.target_entropy incorrect"


def test_no_obsolete_references(model_params_config):
    """Vérifie l'absence de références à dxFeed, obs_t, 320/81 features."""
    config_content = model_params_config.read_text()
    assert "dxFeed" not in config_content, "Référence à dxFeed trouvée"
    assert "obs_t" not in config_content, "Référence à obs_t trouvée"
    assert "320 features" not in config_content, "Référence à 320 features trouvée"
    assert "81 features" not in config_content, "Référence à 81 features trouvée"


def test_invalid_config(tmp_path):
    """Teste un fichier model_params.yaml invalide."""
    config_path = tmp_path / "invalid_model_params.yaml"
    config_content = """
metadata:
  version: "2.1.3"
default:
  learning_rate:
    value: 0.1  # Hors plage
neural_pipeline:
  base_features:
    value: 500  # Hors plage
modes:
  trend:
    batch_size:
      value: 2048  # Hors plage
"""
    config_path.write_text(config_content)
    with pytest.raises(ValueError, match="learning_rate hors plage"):
        ConfigManager()._validate_config(
            "model_params.yaml", yaml.safe_load(config_path.read_text())
        )
