# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_router_config.py
# Tests unitaires pour la configuration router_config.yaml.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie les seuils et paramètres pour la détection des régimes dans router_config.yaml,
#        incluant les métadonnées, les valeurs, les plages, et la cohérence pour trend, range, defensive,
#        neural pipeline, détection, sécurité, dashboard, corrélation, et dérive.
#        Conforme à la Phase 8 (auto-conscience), Phase 11 (détection des régimes),
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
def router_config(tmp_path):
    """Crée un fichier router_config.yaml temporaire pour les tests."""
    config_path = tmp_path / "router_config.yaml"
    config_content = """
metadata:
  version: "2.1.3"
  updated: "2025-05-13"
trend:
  atr_threshold:
    value: 1.8
    range: [0.5, 5.0]
  adx_threshold:
    value: 25
    range: [10, 50]
  ofi_score_threshold:
    value: 0.3
    range: [0.1, 0.8]
  macro_score_threshold:
    value: 0.8
    range: [0.5, 1.0]
  vol_impact_threshold:
    value: 0.5
    range: [0.2, 1.0]
  news_impact_threshold:
    value: 0.5
    range: [0.0, 1.0]
  vix_threshold:
    value: 20.0
    range: [10.0, 50.0]
range:
  vwap_slope_threshold:
    value: 0.01
    range: [0.0, 0.05]
  atr_normalized_threshold:
    value: 0.8
    range: [0.2, 1.5]
  volume_atr_threshold:
    value: 50
    range: [20, 200]
  macro_score_threshold:
    value: 0.8
    range: [0.5, 1.0]
  vol_impact_threshold:
    value: 0.5
    range: [0.2, 1.0]
  news_impact_threshold:
    value: 0.5
    range: [0.0, 1.0]
  vix_threshold:
    value: 20.0
    range: [10.0, 50.0]
defensive:
  volatility_spike_threshold:
    value: 2.0
    range: [1.0, 5.0]
  regime_confidence_threshold:
    value: 0.5
    range: [0.0, 0.9]
  macro_score_threshold:
    value: 0.8
    range: [0.5, 1.0]
  vol_impact_threshold:
    value: 0.5
    range: [0.2, 1.0]
  news_impact_threshold:
    value: 0.5
    range: [0.0, 1.0]
  vix_threshold:
    value: 20.0
    range: [10.0, 50.0]
neural:
  neural_regime_confidence:
    value: 0.7
    range: [0.5, 0.95]
  predicted_volatility_threshold:
    value: 1.2
    range: [0.5, 3.0]
detection:
  impute_nan:
    value: false
  use_optimized_calculations:
    value: true
  compute_shap:
    value: false
safety:
  safe_mode:
    value: false
  critical_times:
    value: ["14:00", "15:30"]
dashboard:
  interval:
    value: 10000
    range: [5000, 30000]
  thresholds:
    vix_es_correlation:
      value: 25
      range: [10, 50]
  compute_shap:
    value: false
correlation:
  significant_threshold:
    value: 0.8
    range: [0.5, 0.95]
drift:
  wass_threshold:
    value: 0.1
    range: [0.01, 0.5]
  ks_threshold:
    value: 0.05
    range: [0.01, 0.1]
"""
    config_path.write_text(config_content)
    return config_path


def test_metadata(router_config):
    """Teste les métadonnées de router_config.yaml."""
    config = yaml.safe_load(router_config.read_text())
    assert config["metadata"]["version"] == "2.1.3", "Version incorrecte"
    assert config["metadata"]["updated"] == "2025-05-13", "Date incorrecte"
    assert "Phase 8" in config["metadata"]["description"], "Phase 8 non mentionnée"
    assert "Phase 11" in config["metadata"]["description"], "Phase 11 non mentionnée"
    assert "Phase 16" in config["metadata"]["description"], "Phase 16 non mentionnée"
    assert (
        "350 features" in config["metadata"]["description"]
    ), "Alignement 350 features non mentionné"


def test_trend_thresholds(router_config):
    """Teste les seuils pour le régime trend."""
    config = yaml.safe_load(router_config.read_text())
    trend = config["trend"]
    assert trend["atr_threshold"]["value"] == 1.8, "atr_threshold incorrect"
    assert trend["adx_threshold"]["value"] == 25, "adx_threshold incorrect"
    assert trend["ofi_score_threshold"]["value"] == 0.3, "ofi_score_threshold incorrect"
    assert (
        trend["macro_score_threshold"]["value"] == 0.8
    ), "macro_score_threshold incorrect"
    assert (
        trend["vol_impact_threshold"]["value"] == 0.5
    ), "vol_impact_threshold incorrect"
    assert (
        trend["news_impact_threshold"]["value"] == 0.5
    ), "news_impact_threshold incorrect"
    assert trend["vix_threshold"]["value"] == 20.0, "vix_threshold incorrect"


def test_range_thresholds(router_config):
    """Teste les seuils pour le régime range."""
    config = yaml.safe_load(router_config.read_text())
    range_regime = config["range"]
    assert (
        range_regime["vwap_slope_threshold"]["value"] == 0.01
    ), "vwap_slope_threshold incorrect"
    assert (
        range_regime["atr_normalized_threshold"]["value"] == 0.8
    ), "atr_normalized_threshold incorrect"
    assert (
        range_regime["volume_atr_threshold"]["value"] == 50
    ), "volume_atr_threshold incorrect"
    assert (
        range_regime["macro_score_threshold"]["value"] == 0.8
    ), "macro_score_threshold incorrect"
    assert (
        range_regime["vol_impact_threshold"]["value"] == 0.5
    ), "vol_impact_threshold incorrect"
    assert (
        range_regime["news_impact_threshold"]["value"] == 0.5
    ), "news_impact_threshold incorrect"
    assert range_regime["vix_threshold"]["value"] == 20.0, "vix_threshold incorrect"


def test_defensive_thresholds(router_config):
    """Teste les seuils pour le régime defensive."""
    config = yaml.safe_load(router_config.read_text())
    defensive = config["defensive"]
    assert (
        defensive["volatility_spike_threshold"]["value"] == 2.0
    ), "volatility_spike_threshold incorrect"
    assert (
        defensive["regime_confidence_threshold"]["value"] == 0.5
    ), "regime_confidence_threshold incorrect"
    assert (
        defensive["macro_score_threshold"]["value"] == 0.8
    ), "macro_score_threshold incorrect"
    assert (
        defensive["vol_impact_threshold"]["value"] == 0.5
    ), "vol_impact_threshold incorrect"
    assert (
        defensive["news_impact_threshold"]["value"] == 0.5
    ), "news_impact_threshold incorrect"
    assert defensive["vix_threshold"]["value"] == 20.0, "vix_threshold incorrect"


def test_neural_and_detection(router_config):
    """Teste les paramètres de la pipeline neurale et de détection."""
    config = yaml.safe_load(router_config.read_text())
    neural = config["neural"]
    assert (
        neural["neural_regime_confidence"]["value"] == 0.7
    ), "neural_regime_confidence incorrect"
    assert (
        neural["predicted_volatility_threshold"]["value"] == 1.2
    ), "predicted_volatility_threshold incorrect"
    detection = config["detection"]
    assert detection["impute_nan"]["value"] is False, "impute_nan incorrect"
    assert (
        detection["use_optimized_calculations"]["value"] is True
    ), "use_optimized_calculations incorrect"
    assert detection["compute_shap"]["value"] is False, "compute_shap incorrect"


def test_safety_and_dashboard(router_config):
    """Teste les paramètres de sécurité et du tableau de bord."""
    config = yaml.safe_load(router_config.read_text())
    safety = config["safety"]
    assert safety["safe_mode"]["value"] is False, "safe_mode incorrect"
    assert safety["critical_times"]["value"] == [
        "14:00",
        "15:30",
    ], "critical_times incorrect"
    dashboard = config["dashboard"]
    assert dashboard["interval"]["value"] == 10000, "interval incorrect"
    assert (
        dashboard["thresholds"]["vix_es_correlation"]["value"] == 25
    ), "vix_es_correlation incorrect"
    assert dashboard["compute_shap"]["value"] is False, "compute_shap incorrect"


def test_correlation_and_drift(router_config):
    """Teste les paramètres de corrélation et de dérive."""
    config = yaml.safe_load(router_config.read_text())
    correlation = config["correlation"]
    assert (
        correlation["significant_threshold"]["value"] == 0.8
    ), "significant_threshold incorrect"
    drift = config["drift"]
    assert drift["wass_threshold"]["value"] == 0.1, "wass_threshold incorrect"
    assert drift["ks_threshold"]["value"] == 0.05, "ks_threshold incorrect"


def test_no_obsolete_references(router_config):
    """Vérifie l'absence de références à dxFeed, obs_t, 320/81 features."""
    config_content = router_config.read_text()
    assert "dxFeed" not in config_content, "Référence à dxFeed trouvée"
    assert "obs_t" not in config_content, "Référence à obs_t trouvée"
    assert "320 features" not in config_content, "Référence à 320 features trouvée"
    assert "81 features" not in config_content, "Référence à 81 features trouvée"


def test_invalid_config(tmp_path):
    """Teste un fichier router_config.yaml invalide."""
    config_path = tmp_path / "invalid_router_config.yaml"
    config_content = """
metadata:
  version: "2.1.3"
trend:
  atr_threshold:
    value: 6.0  # Hors plage
range:
  vwap_slope_threshold:
    value: 0.1  # Hors plage
defensive:
  volatility_spike_threshold:
    value: 6.0  # Hors plage
"""
    config_path.write_text(config_content)
    with pytest.raises(ValueError, match="atr_threshold hors plage"):
        ConfigManager()._validate_config(
            "router_config.yaml", yaml.safe_load(config_path.read_text())
        )
