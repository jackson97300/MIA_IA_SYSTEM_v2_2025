# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_alert_config.py
# Tests unitaires pour la configuration alert_config.yaml.
#
# Version : 2.1.5
# Date : 2025-05-14
#
# Rôle : Vérifie les paramètres des canaux d'alertes et des seuils dans alert_config.yaml,
#        incluant les métadonnées, les priorités, les seuils, et la cohérence pour Telegram, Discord, et Email.
#        Conforme aux Phases 1 (collecte et traitement des nouvelles), 8 (auto-conscience), 13 (métriques d’options),
#        15 (garde-fou microstructure), et 16 (ensemble et transfer learning).
#        Supporte la nouvelle architecture multi-canaux avec gestion des priorités :
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
# - Utilise des clés fictives pour les tests.
# - Suppression de toute référence à dxFeed, obs_t, 320 features, et 81 features.
# - Vérifie l'alignement avec 350 features pour l’entraînement et 150 SHAP pour l’inférence.
# - Support SMS supprimé ; tests adaptés pour Telegram, Discord, Email, stockage local, et cache.

import pytest
import yaml

from src.model.utils.config_manager import ConfigManager


@pytest.fixture
def alert_config(tmp_path):
    """Crée un fichier alert_config.yaml temporaire pour les tests."""
    config_path = tmp_path / "alert_config.yaml"
    config_content = """
metadata:
  version: "2.1.5"
  updated: "2025-05-14"
telegram:
  enabled: true
  bot_token: "test_bot_token"
  chat_id: "test_chat_id"
  priority: 3
discord:
  enabled: true
  webhook_url: "test_discord_webhook_url"
  priority: 1
email:
  enabled: true
  smtp_server: "smtp.gmail.com"
  smtp_port: 587
  sender_email: "test_email@gmail.com"
  sender_password: "test_password"
  receiver_email: "test_receiver@example.com"
  priority: 4
local_storage:
  enabled: false
  path: "data/alerts/local"
  priority: 5
cache:
  enabled: true
  path: "data/cache/alert_cache.pkl"
alert_thresholds:
  volatility_spike_threshold:
    value: 2.0
    range: [1.0, 5.0]
  oi_sweep_alert_threshold:
    value: 0.15
    range: [0.1, 0.3]
  macro_event_severity_alert:
    value: 0.6
    range: [0.3, 1.0]
  news_impact_score_threshold:
    value: 0.5
    range: [0.0, 1.0]
  spoofing_score_threshold:
    value: 0.7
    range: [0.5, 1.0]
  volume_anomaly_threshold:
    value: 0.8
    range: [0.5, 1.0]
  options_risk_score_threshold:
    value: 0.6
    range: [0.3, 1.0]
  prediction_uncertainty_threshold:
    value: 0.4
    range: [0.2, 0.6]
"""
    config_path.write_text(config_content)
    return config_path


def test_metadata(alert_config):
    """Teste les métadonnées de alert_config.yaml."""
    config = yaml.safe_load(alert_config.read_text())
    assert config["metadata"]["version"] == "2.1.5", "Version incorrecte"
    assert config["metadata"]["updated"] == "2025-05-14", "Date incorrecte"
    assert "Phase 1" in config["metadata"]["description"], "Phase 1 non mentionnée"
    assert "Phase 8" in config["metadata"]["description"], "Phase 8 non mentionnée"
    assert "Phase 13" in config["metadata"]["description"], "Phase 13 non mentionnée"
    assert "Phase 15" in config["metadata"]["description"], "Phase 15 non mentionnée"
    assert "Phase 16" in config["metadata"]["description"], "Phase 16 non mentionnée"
    assert (
        "350 features" in config["metadata"]["description"]
    ), "Alignement 350 features non mentionné"
    assert (
        "multi-canaux" in config["metadata"]["description"]
    ), "Architecture multi-canaux non mentionnée"


def test_telegram_config(alert_config):
    """Teste la configuration Telegram."""
    config = yaml.safe_load(alert_config.read_text())
    telegram = config["telegram"]
    assert telegram["enabled"] is True, "Telegram.enabled incorrect"
    assert telegram["bot_token"] == "test_bot_token", "Telegram.bot_token incorrect"
    assert telegram["chat_id"] == "test_chat_id", "Telegram.chat_id incorrect"
    assert telegram["priority"] == 3, "Telegram.priority incorrect"
    assert "priority_description" in telegram, "Telegram.priority_description manquante"
    assert (
        "3=error" in telegram["priority_description"]
    ), "Telegram.priority_description incorrecte"
    assert (
        "priorités 3–5" in telegram["priority_description"]
    ), "Description des priorités Telegram incorrecte"


def test_discord_config(alert_config):
    """Teste la configuration Discord."""
    config = yaml.safe_load(alert_config.read_text())
    discord = config["discord"]
    assert discord["enabled"] is True, "Discord.enabled incorrect"
    assert (
        discord["webhook_url"] == "test_discord_webhook_url"
    ), "Discord.webhook_url incorrect"
    assert discord["priority"] == 1, "Discord.priority incorrect"
    assert "priority_description" in discord, "Discord.priority_description manquante"
    assert (
        "1=info" in discord["priority_description"]
    ), "Discord.priority_description incorrecte"
    assert (
        "toutes les priorités (1–5)" in discord["priority_description"]
    ), "Description des priorités Discord incorrecte"


def test_email_config(alert_config):
    """Teste la configuration Email."""
    config = yaml.safe_load(alert_config.read_text())
    email = config["email"]
    assert email["enabled"] is True, "Email.enabled incorrect"
    assert email["smtp_server"] == "smtp.gmail.com", "Email.smtp_server incorrect"
    assert email["smtp_port"] == 587, "Email.smtp_port incorrect"
    assert (
        email["sender_email"] == "test_email@gmail.com"
    ), "Email.sender_email incorrect"
    assert (
        email["sender_password"] == "test_password"
    ), "Email.sender_password incorrect"
    assert (
        email["receiver_email"] == "test_receiver@example.com"
    ), "Email.receiver_email incorrect"
    assert email["priority"] == 4, "Email.priority incorrect"
    assert "priority_description" in email, "Email.priority_description manquante"
    assert (
        "4=critical" in email["priority_description"]
    ), "Email.priority_description incorrecte"
    assert (
        "priorités 4–5" in email["priority_description"]
    ), "Description des priorités Email incorrecte"


def test_local_storage_config(alert_config):
    """Teste la configuration du stockage local."""
    config = yaml.safe_load(alert_config.read_text())
    local_storage = config["local_storage"]
    assert local_storage["enabled"] is False, "Local_storage.enabled incorrect"
    assert local_storage["path"] == "data/alerts/local", "Local_storage.path incorrect"
    assert local_storage["priority"] == 5, "Local_storage.priority incorrect"
    assert (
        "priority_description" in local_storage
    ), "Local_storage.priority_description manquante"
    assert (
        "priorité 5" in local_storage["priority_description"]
    ), "Local_storage.priority_description incorrecte"


def test_cache_config(alert_config):
    """Teste la configuration du cache."""
    config = yaml.safe_load(alert_config.read_text())
    cache = config["cache"]
    assert cache["enabled"] is True, "Cache.enabled incorrect"
    assert cache["path"] == "data/cache/alert_cache.pkl", "Cache.path incorrect"
    assert "description" in cache, "Cache.description manquante"
    assert (
        "persistance du cache" in cache["description"]
    ), "Cache.description incorrecte"


def test_alert_thresholds(alert_config):
    """Teste les seuils d'alerte."""
    config = yaml.safe_load(alert_config.read_text())
    thresholds = config["alert_thresholds"]
    assert (
        thresholds["volatility_spike_threshold"]["value"] == 2.0
    ), "volatility_spike_threshold incorrect"
    assert thresholds["volatility_spike_threshold"]["range"] == [
        1.0,
        5.0,
    ], "Plage volatility_spike_threshold incorrecte"
    assert (
        thresholds["oi_sweep_alert_threshold"]["value"] == 0.15
    ), "oi_sweep_alert_threshold incorrect"
    assert (
        thresholds["macro_event_severity_alert"]["value"] == 0.6
    ), "macro_event_severity_alert incorrect"
    assert (
        thresholds["news_impact_score_threshold"]["value"] == 0.5
    ), "news_impact_score_threshold incorrect"
    assert (
        thresholds["spoofing_score_threshold"]["value"] == 0.7
    ), "spoofing_score_threshold incorrect"
    assert (
        thresholds["volume_anomaly_threshold"]["value"] == 0.8
    ), "volume_anomaly_threshold incorrect"
    assert (
        thresholds["options_risk_score_threshold"]["value"] == 0.6
    ), "options_risk_score_threshold incorrect"
    assert (
        thresholds["prediction_uncertainty_threshold"]["value"] == 0.4
    ), "prediction_uncertainty_threshold incorrect"
    assert thresholds["prediction_uncertainty_threshold"]["range"] == [
        0.2,
        0.6,
    ], "Plage prediction_uncertainty_threshold incorrecte"
    assert (
        "description" in thresholds["prediction_uncertainty_threshold"]
    ), "prediction_uncertainty_threshold.description manquante"


def test_no_obsolete_references(alert_config):
    """Vérifie l'absence de références à dxFeed, obs_t, 320/81 features."""
    config_content = alert_config.read_text()
    assert "dxFeed" not in config_content, "Référence à dxFeed trouvée"
    assert "obs_t" not in config_content, "Référence à obs_t trouvée"
    assert "320 features" not in config_content, "Référence à 320 features trouvée"
    assert "81 features" not in config_content, "Référence à 81 features trouvée"


def test_invalid_config(tmp_path):
    """Teste un fichier alert_config.yaml invalide."""
    config_path = tmp_path / "invalid_alert_config.yaml"
    config_content = """
metadata:
  version: "2.1.5"
telegram:
  enabled: true
  bot_token: ""
  chat_id: ""
  priority: 6  # Hors plage
discord:
  enabled: true
  webhook_url: ""
  priority: 0  # Hors plage
alert_thresholds:
  volatility_spike_threshold:
    value: 6.0  # Hors plage
"""
    config_path.write_text(config_content)
    with pytest.raises(ValueError, match="telegram.priority doit être entre 1 et 5"):
        ConfigManager()._validate_config(
            "alert_config.yaml", yaml.safe_load(config_path.read_text())
        )
