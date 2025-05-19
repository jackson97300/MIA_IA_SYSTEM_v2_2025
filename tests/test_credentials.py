# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_credentials.py
# Tests unitaires pour la configuration credentials.yaml.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie les clés API dans credentials.yaml pour IQFeed, Investing.com, NewsData.io, et OpenAI,
#        incluant leur structure et leur validation. Conforme à la Phase 1 (collecte de données)
#        et Phase 8 (auto-conscience pour les fonctionnalités cognitives).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pyyaml>=6.0.0,<7.0.0
# - src/model/utils/config_manager.py
#
# Notes :
# - Utilise des clés fictives pour les tests.
# - Suppression de toute référence à dxFeed, obs_t, 320 features, et 81 features.
# - Vérifie l'alignement avec l’exclusivité IQFeed et 350/150 SHAP features.

import pytest
import yaml

from src.model.utils.config_manager import ConfigManager


@pytest.fixture
def credentials_config(tmp_path):
    """Crée un fichier credentials.yaml temporaire pour les tests."""
    config_path = tmp_path / "credentials.yaml"
    config_content = """
metadata:
  version: "2.1.3"
  updated: "2025-05-13"
iqfeed:
  api_key: "test_iqfeed_key"
investing_com:
  api_key: "test_investing_key"
newsdata_io:
  api_keys:
    - "test_newsdata_key1"
nlp:
  enabled: true
  provider: "openai"
  api_key: "test_openai_key"
"""
    config_path.write_text(config_content)
    return config_path


def test_metadata(credentials_config):
    """Teste les métadonnées de credentials.yaml."""
    config = yaml.safe_load(credentials_config.read_text())
    assert config["metadata"]["version"] == "2.1.3", "Version incorrecte"
    assert config["metadata"]["updated"] == "2025-05-13", "Date incorrecte"
    assert "Phase 1" in config["metadata"]["description"], "Phase 1 non mentionnée"
    assert "Phase 8" in config["metadata"]["description"], "Phase 8 non mentionnée"
    assert (
        "350 features" in config["metadata"]["description"]
    ), "Alignement 350 features non mentionné"


def test_iqfeed_credentials(credentials_config):
    """Teste les identifiants IQFeed."""
    config = yaml.safe_load(credentials_config.read_text())
    assert config["iqfeed"]["api_key"] == "test_iqfeed_key", "Clé IQFeed incorrecte"
    assert "description" in config["iqfeed"], "Description IQFeed manquante"
    assert (
        "alert_manager.py" in config["iqfeed"]["description"]
    ), "Description IQFeed doit mentionner alert_manager.py"


def test_investing_com_credentials(credentials_config):
    """Teste les identifiants Investing.com."""
    config = yaml.safe_load(credentials_config.read_text())
    assert (
        config["investing_com"]["api_key"] == "test_investing_key"
    ), "Clé Investing.com incorrecte"
    assert (
        "description" in config["investing_com"]
    ), "Description Investing.com manquante"
    assert (
        "alert_manager.py" in config["investing_com"]["description"]
    ), "Description Investing.com doit mentionner alert_manager.py"


def test_newsdata_io_credentials(credentials_config):
    """Teste les identifiants NewsData.io."""
    config = yaml.safe_load(credentials_config.read_text())
    assert (
        len(config["newsdata_io"]["api_keys"]) == 1
    ), "Nombre de clés NewsData.io incorrect"
    assert (
        config["newsdata_io"]["api_keys"][0] == "test_newsdata_key1"
    ), "Clé NewsData.io incorrecte"
    assert "description" in config["newsdata_io"], "Description NewsData.io manquante"
    assert (
        "alert_manager.py" in config["newsdata_io"]["description"]
    ), "Description NewsData.io doit mentionner alert_manager.py"


def test_nlp_credentials(credentials_config):
    """Teste les identifiants OpenAI."""
    config = yaml.safe_load(credentials_config.read_text())
    nlp = config["nlp"]
    assert nlp["enabled"], "NLP non activé"
    assert nlp["provider"] == "openai", "Provider NLP incorrect"
    assert nlp["api_key"] == "test_openai_key", "Clé OpenAI incorrecte"
    assert "api_key_description" in nlp, "Description clé OpenAI manquante"
    assert (
        "alert_manager.py" in nlp["api_key_description"]
    ), "Description clé OpenAI doit mentionner alert_manager.py"


def test_no_obsolete_references(credentials_config):
    """Vérifie l'absence de références à dxFeed, obs_t, 320/81 features."""
    config_content = credentials_config.read_text()
    assert "dxFeed" not in config_content, "Référence à dxFeed trouvée"
    assert "obs_t" not in config_content, "Référence à obs_t trouvée"
    assert "320 features" not in config_content, "Référence à 320 features trouvée"
    assert "81 features" not in config_content, "Référence à 81 features trouvée"


def test_invalid_config(tmp_path):
    """Teste un fichier credentials.yaml invalide."""
    config_path = tmp_path / "invalid_credentials.yaml"
    config_content = """
metadata:
  version: "2.1.3"
iqfeed:
  api_key: ""  # Vide
newsdata_io:
  api_keys: []  # Vide
"""
    config_path.write_text(config_content)
    with pytest.raises(ValueError, match="iqfeed.api_key non défini ou vide"):
        ConfigManager()._validate_config(
            "credentials.yaml", yaml.safe_load(config_path.read_text())
        )
