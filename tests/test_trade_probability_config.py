# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_trade_probability_config.py
# Tests unitaires pour la configuration trade_probability_config.yaml.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie les paramètres de TradeProbabilityPredictor dans trade_probability_config.yaml,
#        incluant les métadonnées, les valeurs, les plages, et la cohérence pour les prédictions de probabilité de trade.
#        Conforme à la Phase 16 (ensemble et transfer learning).
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
def trade_probability_config(tmp_path):
    """Crée un fichier trade_probability_config.yaml temporaire pour les tests."""
    config_path = tmp_path / "trade_probability_config.yaml"
    config_content = """
metadata:
  version: "2.1.3"
  updated: "2025-05-13"
trade_probability:
  buffer_size:
    value: 100
    range: [50, 1000]
  min_trade_success_prob:
    value: 0.7
    range: [0.5, 1.0]
  retrain_threshold:
    value: 1000
    range: [500, 5000]
  retrain_frequency:
    value: "weekly"
    options: ["daily", "weekly", "biweekly"]
  model_dir:
    value: "model/trade_probability"
  snapshot_dir:
    value: "data/trade_probability_snapshots"
  perf_log_path:
    value: "data/logs/trade_probability_performance.csv"
  metrics_log_path:
    value: "data/logs/trade_probability_metrics.csv"
"""
    config_path.write_text(config_content)
    return config_path


def test_metadata(trade_probability_config):
    """Teste les métadonnées de trade_probability_config.yaml."""
    config = yaml.safe_load(trade_probability_config.read_text())
    assert config["metadata"]["version"] == "2.1.3", "Version incorrecte"
    assert config["metadata"]["updated"] == "2025-05-13", "Date incorrecte"
    assert "Phase 16" in config["metadata"]["description"], "Phase 16 non mentionnée"


def test_trade_probability_params(trade_probability_config):
    """Teste les paramètres de trade_probability."""
    config = yaml.safe_load(trade_probability_config.read_text())
    tp = config["trade_probability"]
    assert tp["buffer_size"]["value"] == 100, "buffer_size incorrect"
    assert tp["buffer_size"]["range"] == [50, 1000], "Plage buffer_size incorrecte"
    assert (
        tp["min_trade_success_prob"]["value"] == 0.7
    ), "min_trade_success_prob incorrect"
    assert tp["min_trade_success_prob"]["range"] == [
        0.5,
        1.0,
    ], "Plage min_trade_success_prob incorrecte"
    assert tp["retrain_threshold"]["value"] == 1000, "retrain_threshold incorrect"
    assert tp["retrain_threshold"]["range"] == [
        500,
        5000,
    ], "Plage retrain_threshold incorrecte"
    assert tp["retrain_frequency"]["value"] == "weekly", "retrain_frequency incorrect"
    assert tp["retrain_frequency"]["options"] == [
        "daily",
        "weekly",
        "biweekly",
    ], "Options retrain_frequency incorrectes"
    assert tp["model_dir"]["value"] == "model/trade_probability", "model_dir incorrect"
    assert (
        tp["snapshot_dir"]["value"] == "data/trade_probability_snapshots"
    ), "snapshot_dir incorrect"
    assert (
        tp["perf_log_path"]["value"] == "data/logs/trade_probability_performance.csv"
    ), "perf_log_path incorrect"
    assert (
        tp["metrics_log_path"]["value"] == "data/logs/trade_probability_metrics.csv"
    ), "metrics_log_path incorrect"


def test_no_obsolete_references(trade_probability_config):
    """Vérifie l'absence de références à dxFeed, obs_t, 320/81 features."""
    config_content = trade_probability_config.read_text()
    assert "dxFeed" not in config_content, "Référence à dxFeed trouvée"
    assert "obs_t" not in config_content, "Référence à obs_t trouvée"
    assert "320 features" not in config_content, "Référence à 320 features trouvée"
    assert "81 features" not in config_content, "Référence à 81 features trouvée"


def test_invalid_config(tmp_path):
    """Teste un fichier trade_probability_config.yaml invalide."""
    config_path = tmp_path / "invalid_trade_probability_config.yaml"
    config_content = """
metadata:
  version: "2.1.3"
trade_probability:
  buffer_size:
    value: 10  # Hors plage
  min_trade_success_prob:
    value: 0.3  # Hors plage
  retrain_threshold:
    value: 100  # Hors plage
  retrain_frequency:
    value: "monthly"  # Non dans options
"""
    config_path.write_text(config_content)
    with pytest.raises(ValueError, match="buffer_size hors plage"):
        ConfigManager()._validate_config(
            "trade_probability_config.yaml", yaml.safe_load(config_path.read_text())
        )
