# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_market_config.py
# Tests unitaires pour la configuration market_config.yaml.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie les paramètres du marché E-mini S&P 500 dans market_config.yaml,
#        incluant les métadonnées, les sources de données, les chemins, les modes de trading,
#        les seuils, les risques, et la gestion des nouvelles. Conforme à la Phase 1 (collecte de données),
#        Phase 8 (auto-conscience), Phase 12 (gestion des risques), et Phase 16 (ensemble et transfer learning).
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
def market_config(tmp_path):
    """Crée un fichier market_config.yaml temporaire pour les tests."""
    config_path = tmp_path / "market_config.yaml"
    config_content = """
metadata:
  version: "2.1.3"
  updated: "2025-05-13"
data_sources:
  data_feed: "iqfeed"
  providers:
    iqfeed:
      host: "localhost"
      ohlc_port: 9100
      quote_port: 9200
      news_port: 9300
      api_key: "<key>"
  symbols:
    value: ["ES", "SPY", "TLT", "VIX", "VVIX", "DVVIX"]
market:
  symbol: "ES"
data:
  raw: "data/iqfeed/iqfeed_data.csv"
  processed: "data/iqfeed/merged_data.csv"
  features: "data/features/features_latest.csv"
  trades_simulated: "data/trades/trades_simulated.csv"
  trades_real: "data/trades/trades_real.csv"
  cache_duration:
    value: 24
    range: [1, 168]
market_params:
  tick_size:
    value: 0.25
    range: [0.01, 1.0]
  point_value:
    value: 50
    range: [10, 100]
  typical_price_range:
    value: [4000, 6000]
  trading_hours:
    start: "09:30"
    end: "16:00"
    timezone: "US/Eastern"
  cross_symbols:
    value: ["SPY", "TLT"]
modes:
  trend:
    enabled: true
    priority: 1
    priority_range: [1, 5]
  range:
    enabled: true
    priority: 2
    priority_range: [1, 5]
  defensive:
    enabled: true
    priority: 3
    priority_range: [1, 5]
  paper_trading_enabled:
    value: true
thresholds:
  reference: "router_config.yaml"
  max_drawdown:
    value: -0.10
    range: [-1.0, 0.0]
  vix_high_threshold:
    value: 25.0
    range: [10.0, 50.0]
  iv_high_threshold:
    value: 0.25
    range: [0.1, 0.5]
  news_impact_threshold:
    value: 0.5
    range: [0.0, 1.0]
  vix_threshold:
    value: 20.0
    range: [10.0, 50.0]
risk:
  default_rrr:
    value: 2.0
    range: [1.0, 5.0]
  min_rrr:
    value: 1.5
    range: [1.0, 3.0]
  max_loss_per_trade:
    value: -50
    range: [-1000, 0]
  target_profit_per_trade:
    value: 100
    range: [0, 1000]
news:
  enabled: true
  source: "data/iqfeed/news.csv"
  critical_events:
    value: ["FOMC", "CPI", "NFP"]
  pause_trading_window_minutes:
    value: 30
    range: [5, 120]
logging:
  directory: "data/logs/market"
  files:
    backtest: "backtest.csv"
    live_trading: "live_trading.log"
    regime_detection: "regime_detection.log"
    provider_performance: "provider_performance.csv"
    feature_pipeline_performance: "feature_pipeline_performance.csv"
    train_sac_performance: "train_sac_performance.csv"
    risk_performance: "risk_performance.csv"
    signal_selector_performance: "signal_selector_performance.csv"
  max_size_mb:
    value: 10
    range: [1, 100]
dashboard:
  notifications_enabled: true
  status_file: "data/market_config_dashboard.json"
"""
    config_path.write_text(config_content)
    return config_path


def test_metadata(market_config):
    """Teste les métadonnées de market_config.yaml."""
    config = yaml.safe_load(market_config.read_text())
    assert config["metadata"]["version"] == "2.1.3", "Version incorrecte"
    assert config["metadata"]["updated"] == "2025-05-13", "Date incorrecte"
    assert "Phase 1" in config["metadata"]["description"], "Phase 1 non mentionnée"
    assert "Phase 8" in config["metadata"]["description"], "Phase 8 non mentionnée"
    assert "Phase 12" in config["metadata"]["description"], "Phase 12 non mentionnée"
    assert "Phase 16" in config["metadata"]["description"], "Phase 16 non mentionnée"
    assert (
        "350 features" in config["metadata"]["description"]
    ), "Alignement 350 features non mentionné"


def test_data_sources(market_config):
    """Teste les paramètres des sources de données."""
    config = yaml.safe_load(market_config.read_text())
    data_sources = config["data_sources"]
    assert data_sources["data_feed"] == "iqfeed", "data_feed incorrect"
    assert "iqfeed" in data_sources["providers"], "iqfeed provider manquant"
    assert (
        "dxfeed" not in data_sources["providers"]
    ), "dxfeed provider présent, doit être supprimé"
    assert data_sources["symbols"]["value"] == [
        "ES",
        "SPY",
        "TLT",
        "VIX",
        "VVIX",
        "DVVIX",
    ], "symbols incorrects"


def test_data_paths(market_config):
    """Teste les chemins des données."""
    config = yaml.safe_load(market_config.read_text())
    data = config["data"]
    assert data["raw"] == "data/iqfeed/iqfeed_data.csv", "raw path incorrect"
    assert (
        data["processed"] == "data/iqfeed/merged_data.csv"
    ), "processed path incorrect"
    assert (
        data["features"] == "data/features/features_latest.csv"
    ), "features path incorrect"
    assert (
        "350 features" in data["features_description"]
    ), "features_description doit mentionner 350 features"
    assert (
        data["trades_simulated"] == "data/trades/trades_simulated.csv"
    ), "trades_simulated path incorrect"
    assert (
        data["trades_real"] == "data/trades/trades_real.csv"
    ), "trades_real path incorrect"
    assert data["cache_duration"]["value"] == 24, "cache_duration incorrect"
    assert data["cache_duration"]["range"] == [
        1,
        168,
    ], "Plage cache_duration incorrecte"


def test_market_params(market_config):
    """Teste les paramètres spécifiques au marché."""
    config = yaml.safe_load(market_config.read_text())
    market_params = config["market_params"]
    assert market_params["tick_size"]["value"] == 0.25, "tick_size incorrect"
    assert market_params["point_value"]["value"] == 50, "point_value incorrect"
    assert market_params["typical_price_range"]["value"] == [
        4000,
        6000,
    ], "typical_price_range incorrect"
    assert (
        market_params["trading_hours"]["start"] == "09:30"
    ), "trading_hours.start incorrect"
    assert (
        market_params["trading_hours"]["timezone"] == "US/Eastern"
    ), "trading_hours.timezone incorrect"
    assert market_params["cross_symbols"]["value"] == [
        "SPY",
        "TLT",
    ], "cross_symbols incorrect"


def test_modes_and_thresholds(market_config):
    """Teste les modes de trading et les seuils."""
    config = yaml.safe_load(market_config.read_text())
    modes = config["modes"]
    assert modes["trend"]["enabled"], "trend.enabled incorrect"
    assert modes["range"]["priority"] == 2, "range.priority incorrect"
    assert modes["defensive"]["enabled"], "defensive.enabled incorrect"
    assert modes["paper_trading_enabled"]["value"], "paper_trading_enabled incorrect"
    thresholds = config["thresholds"]
    assert thresholds["max_drawdown"]["value"] == -0.10, "max_drawdown incorrect"
    assert (
        thresholds["vix_high_threshold"]["value"] == 25.0
    ), "vix_high_threshold incorrect"
    assert (
        thresholds["news_impact_threshold"]["value"] == 0.5
    ), "news_impact_threshold incorrect"
    assert thresholds["vix_threshold"]["value"] == 20.0, "vix_threshold incorrect"


def test_risk_and_news(market_config):
    """Teste les paramètres de gestion des risques et des nouvelles."""
    config = yaml.safe_load(market_config.read_text())
    risk = config["risk"]
    assert risk["default_rrr"]["value"] == 2.0, "default_rrr incorrect"
    assert risk["min_rrr"]["value"] == 1.5, "min_rrr incorrect"
    assert risk["max_loss_per_trade"]["value"] == -50, "max_loss_per_trade incorrect"
    assert (
        risk["target_profit_per_trade"]["value"] == 100
    ), "target_profit_per_trade incorrect"
    news = config["news"]
    assert news["enabled"], "news.enabled incorrect"
    assert news["source"] == "data/iqfeed/news.csv", "news.source incorrect"
    assert news["critical_events"]["value"] == [
        "FOMC",
        "CPI",
        "NFP",
    ], "critical_events incorrect"
    assert (
        news["pause_trading_window_minutes"]["value"] == 30
    ), "pause_trading_window_minutes incorrect"


def test_logging_and_dashboard(market_config):
    """Teste les paramètres de journalisation et du tableau de bord."""
    config = yaml.safe_load(market_config.read_text())
    logging = config["logging"]
    assert logging["directory"] == "data/logs/market", "logging.directory incorrect"
    assert (
        "psutil" in logging["directory_description"]
    ), "logging.directory_description doit mentionner psutil"
    assert (
        logging["files"]["backtest"] == "backtest.csv"
    ), "logging.files.backtest incorrect"
    assert logging["max_size_mb"]["value"] == 10, "logging.max_size_mb incorrect"
    dashboard = config["dashboard"]
    assert dashboard[
        "notifications_enabled"
    ], "dashboard.notifications_enabled incorrect"
    assert (
        dashboard["status_file"] == "data/market_config_dashboard.json"
    ), "dashboard.status_file incorrect"


def test_no_obsolete_references(market_config):
    """Vérifie l'absence de références à dxFeed, obs_t, 320/81 features."""
    config_content = market_config.read_text()
    assert "dxFeed" not in config_content, "Référence à dxFeed trouvée"
    assert "obs_t" not in config_content, "Référence à obs_t trouvée"
    assert "320 features" not in config_content, "Référence à 320 features trouvée"
    assert "81 features" not in config_content, "Référence à 81 features trouvée"


def test_invalid_config(tmp_path):
    """Teste un fichier market_config.yaml invalide."""
    config_path = tmp_path / "invalid_market_config.yaml"
    config_content = """
metadata:
  version: "2.1.3"
data_sources:
  data_feed: "invalid_feed"  # Incorrect
market:
  symbol: "SPY"  # Incorrect
thresholds:
  max_drawdown:
    value: -2.0  # Hors plage
"""
    config_path.write_text(config_content)
    with pytest.raises(ValueError, match="data_feed doit être 'iqfeed'"):
        ConfigManager()._validate_config(
            "market_config.yaml", yaml.safe_load(config_path.read_text())
        )
