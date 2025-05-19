# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_es_config.py
# Tests unitaires pour la configuration es_config.yaml.
#
# Version : 2.1.5
# Date : 2025-05-14
#
# Rôle : Vérifie les paramètres spécifiques au marché E-mini S&P 500 dans es_config.yaml,
#        incluant les métadonnées, les valeurs, les plages, et la cohérence pour le prétraitement,
#        les niveaux d’options, les fonctionnalités cognitives, et les canaux d’alertes.
#        Conforme à la Phase 1 (collecte de données), Phase 8 (auto-conscience), et Phase 16 (ensemble et transfer learning).
#        Supporte l’architecture multi-canaux pour les alertes (Telegram, Discord, Email) avec gestion intelligente des priorités :
#        - Priorité 1–2 : Discord (journalisation uniquement)
#        - Priorité 3 : Telegram + Discord
#        - Priorité 4 : Telegram + Discord + Email
#        - Priorité 5 : Telegram + Discord + Email + stockage local (configurable)
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
# - Tests adaptés pour inclure la nouvelle section alerts (Telegram, Discord, Email, local_storage, cache).

import pytest
import yaml

from src.model.utils.config_manager import ConfigManager


@pytest.fixture
def es_config(tmp_path):
    """Crée un fichier es_config.yaml temporaire pour les tests."""
    config_path = tmp_path / "es_config.yaml"
    config_content = """
metadata:
  version: "2.1.5"
  updated: "2025-05-14"
market:
  symbol:
    value: "ES"
mia:
  language: fr
  vocal_enabled: true
  vocal_async: true
  log_dir: data/logs
  enable_csv_log: false
  log_rotation_mb: 10
  verbosity: normal
  voice_profile: calm
  max_logs: 1000
preprocessing:
  input_path: data/iqfeed/iqfeed_data.csv
  output_path: data/features/features_latest_350.csv
  output_path_inference: data/features/features_latest_150.csv
  chunk_size: 20000
  cache_hours: 24
  max_retries: 3
  retry_delay_base: 2.0
  timeout_seconds: 1800
  use_cache: true
  clean_temp_files: true
  drop_duplicates: true
  log_each_step: true
  optimize_memory: true
  depth_level: 5
  transaction_cost: 2.0
spotgamma_recalculator:
  min_volume_threshold: 10
  dealer_zone_gap: 30
  vega_normalization_factor: 1.0
  price_proximity_range: 10
  net_delta_positive_threshold: 0.5
  net_delta_negative_threshold: -0.5
  max_data_age_seconds: 600
  oi_sweep_threshold: 0.1
  iv_acceleration_window: "1h"
  oi_velocity_window: "30min"
  shap_features:
    - iv_atm
    - option_skew
volatility:
  realized_vol_windows:
    - 5min
    - 15min
    - 30min
  volatility_breakout_threshold: 2.0
contextualization:
  news_volume_spike_threshold: 3.0
  macro_event_severity_levels:
    - 0.3
    - 0.6
    - 1.0
  news_impact_threshold: 0.5
  vix_threshold: 20.0
order_flow:
  order_imbalance_persistence_windows:
    - 15min
    - 60min
  hidden_liquidity_detection_threshold: 0.2
memory_contextual:
  num_clusters: 10
  pca_dimensions: 15
  variance_explained: 0.95
  max_cluster_age_seconds: 2592000
alerts:
  telegram:
    enabled: true
    priority: 3
  discord:
    enabled: true
    priority: 1
  email:
    enabled: true
    priority: 4
  local_storage:
    enabled: false
    path: data/alerts/local
    priority: 5
  cache:
    enabled: true
    path: data/cache/alert_cache.pkl
"""
    config_path.write_text(config_content)
    return config_path


def test_metadata(es_config):
    """Teste les métadonnées de es_config.yaml."""
    config = yaml.safe_load(es_config.read_text())
    assert config["metadata"]["version"] == "2.1.5", "Version incorrecte"
    assert config["metadata"]["updated"] == "2025-05-14", "Date incorrecte"
    assert "Phase 1" in config["metadata"]["description"], "Phase 1 non mentionnée"
    assert "Phase 8" in config["metadata"]["description"], "Phase 8 non mentionnée"
    assert "Phase 16" in config["metadata"]["description"], "Phase 16 non mentionnée"
    assert (
        "350 features" in config["metadata"]["description"]
    ), "Alignement 350 features non mentionné"
    assert (
        "multi-canaux" in config["metadata"]["description"]
    ), "Architecture multi-canaux non mentionnée"


def test_market(es_config):
    """Teste les paramètres du marché."""
    config = yaml.safe_load(es_config.read_text())
    market = config["market"]
    assert market["symbol"]["value"] == "ES", "symbol incorrect"
    assert "E-mini S&P 500" in market["description"], "Description market incorrecte"


def test_mia(es_config):
    """Teste les paramètres des fonctionnalités cognitives."""
    config = yaml.safe_load(es_config.read_text())
    mia = config["mia"]
    assert mia["language"] == "fr", "language incorrect"
    assert mia["vocal_enabled"] is True, "vocal_enabled incorrect"
    assert mia["vocal_async"] is True, "vocal_async incorrect"
    assert mia["log_dir"] == "data/logs", "log_dir incorrect"
    assert mia["enable_csv_log"] is False, "enable_csv_log incorrect"
    assert mia["log_rotation_mb"] == 10, "log_rotation_mb incorrect"
    assert mia["verbosity"] == "normal", "verbosity incorrect"
    assert mia["voice_profile"] == "calm", "voice_profile incorrect"
    assert mia["max_logs"] == 1000, "max_logs incorrect"


def test_preprocessing(es_config):
    """Teste les paramètres de prétraitement."""
    config = yaml.safe_load(es_config.read_text())
    preprocessing = config["preprocessing"]
    assert (
        preprocessing["input_path"] == "data/iqfeed/iqfeed_data.csv"
    ), "input_path incorrect"
    assert (
        preprocessing["output_path"] == "data/features/features_latest_350.csv"
    ), "output_path incorrect"
    assert (
        preprocessing["output_path_inference"]
        == "data/features/features_latest_150.csv"
    ), "output_path_inference incorrect"
    assert preprocessing["chunk_size"] == 20000, "chunk_size incorrect"
    assert preprocessing["cache_hours"] == 24, "cache_hours incorrect"
    assert preprocessing["max_retries"] == 3, "max_retries incorrect"
    assert preprocessing["retry_delay_base"] == 2.0, "retry_delay_base incorrect"
    assert preprocessing["depth_level"] == 5, "depth_level incorrect"
    assert preprocessing["transaction_cost"] == 2.0, "transaction_cost incorrect"


def test_spotgamma_recalculator(es_config):
    """Teste les paramètres de spotgamma_recalculator."""
    config = yaml.safe_load(es_config.read_text())
    spotgamma = config["spotgamma_recalculator"]
    assert spotgamma["min_volume_threshold"] == 10, "min_volume_threshold incorrect"
    assert spotgamma["dealer_zone_gap"] == 30, "dealer_zone_gap incorrect"
    assert (
        spotgamma["vega_normalization_factor"] == 1.0
    ), "vega_normalization_factor incorrect"
    assert spotgamma["oi_sweep_threshold"] == 0.1, "oi_sweep_threshold incorrect"
    assert spotgamma["shap_features"] == [
        "iv_atm",
        "option_skew",
    ], "shap_features incorrect"


def test_volatility_and_contextualization(es_config):
    """Teste les paramètres de volatilité et de contextualisation."""
    config = yaml.safe_load(es_config.read_text())
    volatility = config["volatility"]
    assert volatility["realized_vol_windows"] == [
        "5min",
        "15min",
        "30min",
    ], "realized_vol_windows incorrect"
    assert (
        volatility["volatility_breakout_threshold"] == 2.0
    ), "volatility_breakout_threshold incorrect"
    contextualization = config["contextualization"]
    assert (
        contextualization["news_volume_spike_threshold"] == 3.0
    ), "news_volume_spike_threshold incorrect"
    assert contextualization["macro_event_severity_levels"] == [
        0.3,
        0.6,
        1.0,
    ], "macro_event_severity_levels incorrect"
    assert (
        contextualization["news_impact_threshold"] == 0.5
    ), "news_impact_threshold incorrect"
    assert contextualization["vix_threshold"] == 20.0, "vix_threshold incorrect"


def test_order_flow_and_memory_contextual(es_config):
    """Teste les paramètres d'order flow et de mémoire contextuelle."""
    config = yaml.safe_load(es_config.read_text())
    order_flow = config["order_flow"]
    assert order_flow["order_imbalance_persistence_windows"] == [
        "15min",
        "60min",
    ], "order_imbalance_persistence_windows incorrect"
    assert (
        order_flow["hidden_liquidity_detection_threshold"] == 0.2
    ), "hidden_liquidity_detection_threshold incorrect"
    memory = config["memory_contextual"]
    assert memory["num_clusters"] == 10, "num_clusters incorrect"
    assert memory["pca_dimensions"] == 15, "pca_dimensions incorrect"
    assert memory["variance_explained"] == 0.95, "variance_explained incorrect"
    assert (
        memory["max_cluster_age_seconds"] == 2592000
    ), "max_cluster_age_seconds incorrect"


def test_alerts_config(es_config):
    """Teste les paramètres des alertes."""
    config = yaml.safe_load(es_config.read_text())
    alerts = config["alerts"]
    assert alerts["telegram"]["enabled"] is True, "alerts.telegram.enabled incorrect"
    assert alerts["telegram"]["priority"] == 3, "alerts.telegram.priority incorrect"
    assert (
        "priorités 3–5" in alerts["telegram"]["description"]
    ), "alerts.telegram.description incorrecte"
    assert alerts["discord"]["enabled"] is True, "alerts.discord.enabled incorrect"
    assert alerts["discord"]["priority"] == 1, "alerts.discord.priority incorrect"
    assert (
        "toutes priorités (1–5)" in alerts["discord"]["description"]
    ), "alerts.discord.description incorrecte"
    assert alerts["email"]["enabled"] is True, "alerts.email.enabled incorrect"
    assert alerts["email"]["priority"] == 4, "alerts.email.priority incorrect"
    assert (
        "priorités 4–5" in alerts["email"]["description"]
    ), "alerts.email.description incorrecte"
    assert (
        alerts["local_storage"]["enabled"] is False
    ), "alerts.local_storage.enabled incorrect"
    assert (
        alerts["local_storage"]["path"] == "data/alerts/local"
    ), "alerts.local_storage.path incorrect"
    assert (
        alerts["local_storage"]["priority"] == 5
    ), "alerts.local_storage.priority incorrect"
    assert (
        "priorité 5" in alerts["local_storage"]["description"]
    ), "alerts.local_storage.description incorrecte"
    assert alerts["cache"]["enabled"] is True, "alerts.cache.enabled incorrect"
    assert (
        alerts["cache"]["path"] == "data/cache/alert_cache.pkl"
    ), "alerts.cache.path incorrect"
    assert (
        "persistance du cache" in alerts["cache"]["description"]
    ), "alerts.cache.description incorrecte"


def test_no_obsolete_references(es_config):
    """Vérifie l'absence de références à dxFeed, obs_t, 320/81 features."""
    config_content = es_config.read_text()
    assert "dxFeed" not in config_content, "Référence à dxFeed trouvée"
    assert "obs_t" not in config_content, "Référence à obs_t trouvée"
    assert "320 features" not in config_content, "Référence à 320 features trouvée"
    assert "81 features" not in config_content, "Référence à 81 features trouvée"


def test_invalid_config(tmp_path):
    """Teste un fichier es_config.yaml invalide."""
    config_path = tmp_path / "invalid_es_config.yaml"
    config_content = """
metadata:
  version: "2.1.5"
market:
  symbol:
    value: "SPY"  # Incorrect
preprocessing:
  chunk_size: 50000  # Hors plage implicite
spotgamma_recalculator:
  oi_sweep_threshold: 0.5  # Hors plage implicite
alerts:
  telegram:
    enabled: true
    priority: 6  # Hors plage
  discord:
    enabled: true
    priority: 0  # Hors plage
"""
    config_path.write_text(config_content)
    with pytest.raises(
        ValueError, match="alerts.telegram.priority doit être entre 1 et 5"
    ):
        ConfigManager()._validate_config(
            "es_config.yaml", yaml.safe_load(config_path.read_text())
        )
