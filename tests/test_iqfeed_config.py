# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_iqfeed_config.py
# Tests unitaires pour la configuration iqfeed_config.yaml.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie les paramètres de connexion IQFeed dans iqfeed_config.yaml,
#        incluant les valeurs, les plages, et la cohérence pour la collecte des données.
#        Conforme à la Phase 1 (collecte de données), Phase 8 (auto-conscience),
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
def iqfeed_config(tmp_path):
    """Crée un fichier iqfeed_config.yaml temporaire pour les tests."""
    config_path = tmp_path / "iqfeed_config.yaml"
    config_content = """
metadata:
  version: "2.1.3"
  updated: "2025-05-13"
iqfeed:
  host:
    value: "127.0.0.1"
  port:
    value: 9100
  protocol:
    value: "6.2"
connection:
  level:
    value: 2
    options: [1, 2]
  retry_attempts:
    value: 3
    range: [1, 10]
  retry_delay:
    value: 5
    range: [1, 60]
  timeout:
    value: 30
    range: [10, 120]
symbols:
  value: ["ES", "SPY", "TLT", "VIX", "VIX1M", "VIX3M", "VIX6M", "VIX1Y", "VVIX", "DVVIX", "GC", "CL", "BTC", "US10Y", "US2Y"]
data_types:
  ohlc:
    enabled: true
  dom:
    enabled: true
    depth: 5
    range: [1, 10]
  options:
    enabled: true
    metrics:
      - iv: ["call_iv_atm", "put_iv_atm"]
      - oi: ["open_interest", "oi_concentration"]
      - greeks: ["gamma", "delta", "vega", "theta", "vomma", "speed", "ultima"]
  cross_market:
    enabled: true
    metrics:
      - gold: ["gold_correl"]
      - oil: ["oil_correl"]
      - btc: ["crypto_btc_correl"]
      - bonds: ["treasury_10y_yield", "yield_curve_slope"]
      - spy: ["cross_asset_flow_ratio"]
  news:
    enabled: true
    metrics:
      - volume: ["news_volume_1h", "news_volume_1d", "news_volume_spike"]
      - sentiment: ["news_sentiment_momentum", "news_sentiment_acceleration"]
      - content: ["topic_vector_news_3", "topic_vector_news_4"]
frequency:
  ohlc:
    value: "1min"
    options: ["1sec", "5sec", "1min", "5min", "15min", "1h"]
  dom:
    value: "1sec"
    options: ["1sec", "5sec", "10sec"]
  options:
    value: "5min"
    options: ["1min", "5min", "15min"]
  vix:
    value: "1min"
    options: ["1sec", "1min", "5min"]
  cross_market:
    value: "1min"
    options: ["1sec", "1min", "5min"]
  news:
    value: "5min"
    options: ["1min", "5min", "15min"]
cache:
  enabled:
    value: true
  directory:
    value: "data/cache/iqfeed"
  duration_hours:
    value: 24
    range: [1, 168]
error_handling:
  log_errors:
    value: true
  log_file:
    value: "data/logs/iqfeed_errors.log"
  max_log_size_mb:
    value: 10
    range: [1, 100]
dashboard:
  notifications_enabled:
    value: true
  status_file:
    value: "data/iqfeed_config_dashboard.json"
"""
    config_path.write_text(config_content)
    return config_path


def test_metadata(iqfeed_config):
    """Teste les métadonnées de iqfeed_config.yaml."""
    config = yaml.safe_load(iqfeed_config.read_text())
    assert config["metadata"]["version"] == "2.1.3", "Version incorrecte"
    assert config["metadata"]["updated"] == "2025-05-13", "Date incorrecte"
    assert "Phase 1" in config["metadata"]["description"], "Phase 1 non mentionnée"
    assert "Phase 8" in config["metadata"]["description"], "Phase 8 non mentionnée"
    assert "Phase 16" in config["metadata"]["description"], "Phase 16 non mentionnée"
    assert (
        "350 features" in config["metadata"]["description"]
    ), "Alignement 350 features non mentionné"


def test_iqfeed_params(iqfeed_config):
    """Teste les paramètres IQFeed."""
    config = yaml.safe_load(iqfeed_config.read_text())
    iqfeed = config["iqfeed"]
    assert iqfeed["host"]["value"] == "127.0.0.1", "host incorrect"
    assert iqfeed["port"]["value"] == 9100, "port incorrect"
    assert iqfeed["protocol"]["value"] == "6.2", "protocol incorrect"


def test_connection_params(iqfeed_config):
    """Teste les paramètres de connexion."""
    config = yaml.safe_load(iqfeed_config.read_text())
    connection = config["connection"]
    assert connection["level"]["value"] == 2, "level incorrect"
    assert connection["retry_attempts"]["value"] == 3, "retry_attempts incorrect"
    assert connection["retry_delay"]["value"] == 5, "retry_delay incorrect"
    assert connection["timeout"]["value"] == 30, "timeout incorrect"


def test_symbols(iqfeed_config):
    """Teste les symboles à collecter."""
    config = yaml.safe_load(iqfeed_config.read_text())
    symbols = config["symbols"]["value"]
    assert "ES" in symbols, "ES manquant dans symbols"
    assert len(symbols) == 15, "Nombre de symboles incorrect"
    assert "DVVIX" in symbols, "DVVIX manquant dans symbols"
    assert (
        "IQFeed" in config["symbols"]["description"]
    ), "Description symbols doit mentionner IQFeed"


def test_data_types(iqfeed_config):
    """Teste les types de données."""
    config = yaml.safe_load(iqfeed_config.read_text())
    data_types = config["data_types"]
    assert data_types["ohlc"]["enabled"], "ohlc non activé"
    assert data_types["dom"]["depth"] == 5, "dom.depth incorrect"
    assert (
        "call_iv_atm" in data_types["options"]["metrics"][0]["iv"]
    ), "call_iv_atm manquant"
    assert (
        "gold_correl" in data_types["cross_market"]["metrics"][0]["gold"]
    ), "gold_correl manquant"
    assert (
        "news_volume_1h" in data_types["news"]["metrics"][0]["volume"]
    ), "news_volume_1h manquant"


def test_frequency(iqfeed_config):
    """Teste les fréquences de collecte."""
    config = yaml.safe_load(iqfeed_config.read_text())
    frequency = config["frequency"]
    assert frequency["ohlc"]["value"] == "1min", "Fréquence ohlc incorrecte"
    assert frequency["dom"]["value"] == "1sec", "Fréquence dom incorrecte"
    assert frequency["options"]["value"] == "5min", "Fréquence options incorrecte"
    assert frequency["vix"]["value"] == "1min", "Fréquence vix incorrecte"
    assert (
        frequency["cross_market"]["value"] == "1min"
    ), "Fréquence cross_market incorrecte"
    assert frequency["news"]["value"] == "5min", "Fréquence news incorrecte"


def test_cache_and_error_handling(iqfeed_config):
    """Teste les paramètres de cache et gestion des erreurs."""
    config = yaml.safe_load(iqfeed_config.read_text())
    cache = config["cache"]
    assert cache["enabled"]["value"], "Cache non activé"
    assert (
        cache["directory"]["value"] == "data/cache/iqfeed"
    ), "Répertoire cache incorrect"
    assert cache["duration_hours"]["value"] == 24, "duration_hours incorrect"
    error_handling = config["error_handling"]
    assert error_handling["log_errors"]["value"], "log_errors non activé"
    assert (
        error_handling["log_file"]["value"] == "data/logs/iqfeed_errors.log"
    ), "log_file incorrect"
    assert error_handling["max_log_size_mb"]["value"] == 10, "max_log_size_mb incorrect"


def test_dashboard(iqfeed_config):
    """Teste les paramètres du tableau de bord."""
    config = yaml.safe_load(iqfeed_config.read_text())
    dashboard = config["dashboard"]
    assert dashboard["notifications_enabled"][
        "value"
    ], "notifications_enabled non activé"
    assert (
        dashboard["status_file"]["value"] == "data/iqfeed_config_dashboard.json"
    ), "status_file incorrect"


def test_no_obsolete_references(iqfeed_config):
    """Vérifie l'absence de références à dxFeed, obs_t, 320/81 features."""
    config_content = iqfeed_config.read_text()
    assert "dxFeed" not in config_content, "Référence à dxFeed trouvée"
    assert "obs_t" not in config_content, "Référence à obs_t trouvée"
    assert "320 features" not in config_content, "Référence à 320 features trouvée"
    assert "81 features" not in config_content, "Référence à 81 features trouvée"


def test_invalid_config(tmp_path):
    """Teste un fichier iqfeed_config.yaml invalide."""
    config_path = tmp_path / "invalid_iqfeed_config.yaml"
    config_content = """
metadata:
  version: "2.1.3"
iqfeed:
  host:
    value: "invalid_host"  # Incorrect
  port:
    value: 80  # Hors plage
connection:
  level:
    value: 3  # Non dans options
"""
    config_path.write_text(config_content)
    with pytest.raises(ValueError, match="iqfeed.host doit être '127.0.0.1'"):
        ConfigManager()._validate_config(
            "iqfeed_config.yaml", yaml.safe_load(config_path.read_text())
        )
