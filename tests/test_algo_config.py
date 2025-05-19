# tests/test_algo_config.py
import pytest
import yaml  # -*- coding: utf-8 -*-

from src.model.utils.config_manager import ConfigManager, get_config

# MIA_IA_SYSTEM_v2_2025/tests/test_algo_config.py
# Tests unitaires pour la configuration algo_config.yaml.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie les hyperparamètres pour SAC, PPO, et DDPG dans algo_config.yaml,
#        incluant les métadonnées, les valeurs, les plages, et la cohérence pour chaque régime.
#        Conforme aux Phases 6 (pipeline neuronal), 8 (auto-conscience), 14 (régularisation dynamique),
#        16 (ensemble et transfer learning), et 18 (optimisation HFT).
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


@pytest.fixture
def algo_config(tmp_path):
    """Crée un fichier algo_config.yaml temporaire pour les tests."""
    config_path = tmp_path / "algo_config.yaml"
    config_content = """
metadata:
  version: "2.1.3"
  updated: "2025-05-13"
sac:
  trend:
    learning_rate:
      value: 0.0003
      range: [0.0001, 0.001]
    gamma:
      value: 0.99
      range: [0.9, 0.999]
    ent_coef:
      value: 0.1
      range: [0.01, 0.2]
    l2_lambda:
      value: 0.01
      range: [0.001, 0.1]
    batch_size:
      value: 256
      range: [64, 512]
  range:
    learning_rate:
      value: 0.0002
      range: [0.0001, 0.001]
    gamma:
      value: 0.98
      range: [0.9, 0.999]
    ent_coef:
      value: 0.15
      range: [0.01, 0.2]
    l2_lambda:
      value: 0.02
      range: [0.001, 0.1]
    batch_size:
      value: 128
      range: [64, 512]
  defensive:
    learning_rate:
      value: 0.0001
      range: [0.00005, 0.0005]
    gamma:
      value: 0.95
      range: [0.9, 0.999]
    ent_coef:
      value: 0.05
      range: [0.01, 0.1]
    l2_lambda:
      value: 0.03
      range: [0.001, 0.1]
    batch_size:
      value: 64
      range: [32, 256]
ppo:
  trend:
    learning_rate:
      value: 0.00025
      range: [0.0001, 0.001]
    clip_range:
      value: 0.2
      range: [0.1, 0.3]
    n_steps:
      value: 2048
      range: [512, 4096]
    ent_coef:
      value: 0.01
      range: [0.0, 0.1]
    l2_lambda:
      value: 0.01
      range: [0.001, 0.1]
  range:
    learning_rate:
      value: 0.0002
      range: [0.0001, 0.001]
    clip_range:
      value: 0.15
      range: [0.1, 0.3]
    n_steps:
      value: 1024
      range: [512, 2048]
    ent_coef:
      value: 0.02
      range: [0.0, 0.1]
    l2_lambda:
      value: 0.02
      range: [0.001, 0.1]
  defensive:
    learning_rate:
      value: 0.00015
      range: [0.00005, 0.0005]
    clip_range:
      value: 0.1
      range: [0.05, 0.2]
    n_steps:
      value: 512
      range: [256, 1024]
    ent_coef:
      value: 0.0
      range: [0.0, 0.05]
    l2_lambda:
      value: 0.03
      range: [0.001, 0.1]
ddpg:
  trend:
    learning_rate:
      value: 0.001
      range: [0.0005, 0.005]
    gamma:
      value: 0.99
      range: [0.9, 0.999]
    tau:
      value: 0.005
      range: [0.001, 0.01]
    l2_lambda:
      value: 0.01
      range: [0.001, 0.1]
    batch_size:
      value: 128
      range: [64, 256]
  range:
    learning_rate:
      value: 0.0005
      range: [0.0001, 0.002]
    gamma:
      value: 0.98
      range: [0.9, 0.999]
    tau:
      value: 0.01
      range: [0.001, 0.02]
    l2_lambda:
      value: 0.02
      range: [0.001, 0.1]
    batch_size:
      value: 64
      range: [32, 128]
  defensive:
    learning_rate:
      value: 0.0003
      range: [0.0001, 0.001]
    gamma:
      value: 0.95
      range: [0.9, 0.999]
    tau:
      value: 0.015
      range: [0.001, 0.03]
    l2_lambda:
      value: 0.03
      range: [0.001, 0.1]
    batch_size:
      value: 32
      range: [16, 64]
"""
    config_path.write_text(config_content)
    return config_path


def test_metadata(algo_config):
    """Teste les métadonnées de algo_config.yaml."""
    config = yaml.safe_load(algo_config.read_text())
    assert config["metadata"]["version"] == "2.1.3", "Version incorrecte"
    assert config["metadata"]["updated"] == "2025-05-13", "Date incorrecte"
    assert "Phase 6" in config["metadata"]["description"], "Phase 6 non mentionnée"
    assert "Phase 8" in config["metadata"]["description"], "Phase 8 non mentionnée"
    assert "Phase 14" in config["metadata"]["description"], "Phase 14 non mentionnée"
    assert "Phase 16" in config["metadata"]["description"], "Phase 16 non mentionnée"
    assert "Phase 18" in config["metadata"]["description"], "Phase 18 non mentionnée"
    assert (
        "350 features" in config["metadata"]["description"]
    ), "Alignement 350 features non mentionné"


def test_sac_hyperparams(algo_config):
    """Teste les hyperparamètres SAC pour chaque régime."""
    config = yaml.safe_load(algo_config.read_text())
    sac = config["sac"]
    # Trend
    assert (
        sac["trend"]["learning_rate"]["value"] == 0.0003
    ), "SAC trend.learning_rate incorrect"
    assert sac["trend"]["gamma"]["value"] == 0.99, "SAC trend.gamma incorrect"
    assert sac["trend"]["ent_coef"]["value"] == 0.1, "SAC trend.ent_coef incorrect"
    assert sac["trend"]["l2_lambda"]["value"] == 0.01, "SAC trend.l2_lambda incorrect"
    assert sac["trend"]["batch_size"]["value"] == 256, "SAC trend.batch_size incorrect"
    # Range
    assert (
        sac["range"]["learning_rate"]["value"] == 0.0002
    ), "SAC range.learning_rate incorrect"
    assert sac["range"]["ent_coef"]["value"] == 0.15, "SAC range.ent_coef incorrect"
    assert sac["range"]["l2_lambda"]["value"] == 0.02, "SAC range.l2_lambda incorrect"
    # Defensive
    assert (
        sac["defensive"]["learning_rate"]["value"] == 0.0001
    ), "SAC defensive.learning_rate incorrect"
    assert (
        sac["defensive"]["ent_coef"]["value"] == 0.05
    ), "SAC defensive.ent_coef incorrect"
    assert (
        sac["defensive"]["batch_size"]["value"] == 64
    ), "SAC defensive.batch_size incorrect"


def test_ppo_hyperparams(algo_config):
    """Teste les hyperparamètres PPO pour chaque régime."""
    config = yaml.safe_load(algo_config.read_text())
    ppo = config["ppo"]
    # Trend
    assert (
        ppo["trend"]["learning_rate"]["value"] == 0.00025
    ), "PPO trend.learning_rate incorrect"
    assert ppo["trend"]["clip_range"]["value"] == 0.2, "PPO trend.clip_range incorrect"
    assert ppo["trend"]["n_steps"]["value"] == 2048, "PPO trend.n_steps incorrect"
    assert ppo["trend"]["ent_coef"]["value"] == 0.01, "PPO trend.ent_coef incorrect"
    assert ppo["trend"]["l2_lambda"]["value"] == 0.01, "PPO trend.l2_lambda incorrect"
    # Range
    assert (
        ppo["range"]["learning_rate"]["value"] == 0.0002
    ), "PPO range.learning_rate incorrect"
    assert ppo["range"]["clip_range"]["value"] == 0.15, "PPO range.clip_range incorrect"
    assert ppo["range"]["n_steps"]["value"] == 1024, "PPO range.n_steps incorrect"
    # Defensive
    assert (
        ppo["defensive"]["learning_rate"]["value"] == 0.00015
    ), "PPO defensive.learning_rate incorrect"
    assert (
        ppo["defensive"]["ent_coef"]["value"] == 0.0
    ), "PPO defensive.ent_coef incorrect"
    assert (
        ppo["defensive"]["l2_lambda"]["value"] == 0.03
    ), "PPO defensive.l2_lambda incorrect"


def test_ddpg_hyperparams(algo_config):
    """Teste les hyperparamètres DDPG pour chaque régime."""
    config = yaml.safe_load(algo_config.read_text())
    ddpg = config["ddpg"]
    # Trend
    assert (
        ddpg["trend"]["learning_rate"]["value"] == 0.001
    ), "DDPG trend.learning_rate incorrect"
    assert ddpg["trend"]["gamma"]["value"] == 0.99, "DDPG trend.gamma incorrect"
    assert ddpg["trend"]["tau"]["value"] == 0.005, "DDPG trend.tau incorrect"
    assert ddpg["trend"]["l2_lambda"]["value"] == 0.01, "DDPG trend.l2_lambda incorrect"
    assert (
        ddpg["trend"]["batch_size"]["value"] == 128
    ), "DDPG trend.batch_size incorrect"
    # Range
    assert (
        ddpg["range"]["learning_rate"]["value"] == 0.0005
    ), "DDPG range.learning_rate incorrect"
    assert ddpg["range"]["tau"]["value"] == 0.01, "DDPG range.tau incorrect"
    assert ddpg["range"]["l2_lambda"]["value"] == 0.02, "DDPG range.l2_lambda incorrect"
    # Defensive
    assert (
        ddpg["defensive"]["learning_rate"]["value"] == 0.0003
    ), "DDPG defensive.learning_rate incorrect"
    assert ddpg["defensive"]["tau"]["value"] == 0.015, "DDPG defensive.tau incorrect"
    assert (
        ddpg["defensive"]["batch_size"]["value"] == 32
    ), "DDPG defensive.batch_size incorrect"


def test_no_obsolete_references(algo_config):
    """Vérifie l'absence de références à dxFeed, obs_t, 320/81 features."""
    config_content = algo_config.read_text()
    assert "dxFeed" not in config_content, "Référence à dxFeed trouvée"
    assert "obs_t" not in config_content, "Référence à obs_t trouvée"
    assert "320 features" not in config_content, "Référence à 320 features trouvée"
    assert "81 features" not in config_content, "Référence à 81 features trouvée"


def test_invalid_config(tmp_path):
    """Teste un fichier algo_config.yaml invalide."""
    config_path = tmp_path / "invalid_algo_config.yaml"
    config_content = """
metadata:
  version: "2.1.3"
sac:
  trend:
    learning_rate:
      value: 0.01  # Hors plage
ppo:
  range:
    clip_range:
      value: 0.5  # Hors plage
ddpg:
  defensive:
    tau:
      value: 0.05  # Hors plage
"""
    config_path.write_text(config_content)
    with pytest.raises(ValueError, match="learning_rate hors plage"):
        ConfigManager()._validate_config(
            "algo_config.yaml", yaml.safe_load(config_path.read_text())
        )


def test_load_config():
    """Vérifie que algo_config.yaml se charge correctement."""
    config = get_config("algo_config.yaml")
    assert config is not None, "Échec du chargement de algo_config.yaml"
    assert "sac" in config, "Section SAC manquante"
    assert "ppo" in config, "Section PPO manquante"
    assert "ddpg" in config, "Section DDPG manquante"


def test_sac_range_parameters():
    """Vérifie les paramètres SAC pour le régime range."""
    config = get_config("algo_config.yaml")
    sac_range = config["sac"]["range"]
    assert sac_range["ent_coef"] == 0.15, "ent_coef invalide pour SAC range"
    assert sac_range["l2_lambda_base"] == 0.01, "l2_lambda_base invalide pour SAC range"
    assert sac_range["learning_rate"] == 1e-4, "learning_rate invalide pour SAC range"
    assert sac_range["gamma"] == 0.98, "gamma invalide pour SAC range"
    assert sac_range["batch_size"] == 256, "batch_size invalide pour SAC range"


def test_sac_trend_defensive_parameters():
    """Vérifie l2_lambda_base pour SAC dans trend et defensive."""
    config = get_config("algo_config.yaml")
    assert (
        config["sac"]["trend"]["l2_lambda_base"] == 0.01
    ), "l2_lambda_base invalide pour SAC trend"
    assert (
        config["sac"]["defensive"]["l2_lambda_base"] == 0.01
    ), "l2_lambda_base invalide pour SAC defensive"


def test_ppo_parameters():
    """Vérifie les paramètres PPO pour tous les régimes."""
    config = get_config("algo_config.yaml")
    for regime in ["range", "trend", "defensive"]:
        ppo = config["ppo"][regime]
        assert ppo["learning_rate"] == 3e-4, f"learning_rate invalide pour PPO {regime}"
        assert ppo["gamma"] == 0.99, f"gamma invalide pour PPO {regime}"
        assert ppo["batch_size"] == 64, f"batch_size invalide pour PPO {regime}"


def test_ddpg_parameters():
    """Vérifie les paramètres DDPG pour tous les régimes."""
    config = get_config("algo_config.yaml")
    for regime in ["range", "trend", "defensive"]:
        ddpg = config["ddpg"][regime]
        assert (
            ddpg["learning_rate"] == 1e-3
        ), f"learning_rate invalide pour DDPG {regime}"
        assert ddpg["gamma"] == 0.98, f"gamma invalide pour DDPG {regime}"
        assert ddpg["batch_size"] == 128, f"batch_size invalide pour DDPG {regime}"


def test_reward_weights():
    """Vérifie que les poids des récompenses somment à 1 pour chaque algorithme et régime."""
    config = get_config("algo_config.yaml")
    for algo in ["sac", "ppo", "ddpg"]:
        for regime in ["range", "trend", "defensive"]:
            weights = config[algo][regime]["reward_weights"]
            total = sum(weights.values())
            assert (
                abs(total - 1.0) < 1e-6
            ), f"Somme des reward_weights != 1 pour {algo} {regime}"


def test_parameter_ranges():
    """Vérifie les plages des paramètres numériques."""
    config = get_config("algo_config.yaml")
    for algo in ["sac", "ppo", "ddpg"]:
        for regime in ["range", "trend", "defensive"]:
            params = config[algo][regime]
            assert (
                params["learning_rate"] > 0
            ), f"learning_rate doit être positif pour {algo} {regime}"
            assert (
                0 <= params["gamma"] <= 1
            ), f"gamma doit être dans [0, 1] pour {algo} {regime}"
            assert (
                params["ent_coef"] >= 0
            ), f"ent_coef doit être non négatif pour {algo} {regime}"
            assert (
                params["batch_size"] > 0
            ), f"batch_size doit être positif pour {algo} {regime}"
            if "l2_lambda_base" in params:
                assert (
                    params["l2_lambda_base"] >= 0
                ), f"l2_lambda_base doit être non négatif pour {algo} {regime}"
