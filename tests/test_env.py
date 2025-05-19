# Placeholder pour test_env.py
# Rôle : Teste .env (Phase 14).
# test_load_env()
# tests/test_env.py
"""
Tests unitaires pour le fichier .env de MIA_IA_SYSTEM_v2_2025.
Rôle : Vérifie le chargement des variables d'environnement sécurisées (IQFEED_API_KEY, NEWS_API_KEY) (Phase 14).
Conforme à structure.txt (version 2.1.3, 2025-05-13).
"""

import os
import tempfile
from pathlib import Path

import pytest
from dotenv import find_dotenv, load_dotenv

# Fixture pour créer un fichier .env temporaire


@pytest.fixture
def temp_env_file():
    """
    Crée un fichier .env temporaire avec des variables d'environnement factices.
    """
    env_content = """
    IQFEED_API_KEY=test_iqfeed_key
    NEWS_API_KEY=test_news_key
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write(env_content)
        temp_path = Path(f.name)

    yield temp_path

    # Nettoyage
    if os.path.exists(temp_path):
        os.unlink(temp_path)


# Fixture pour charger l'environnement


@pytest.fixture
def setup_env(temp_env_file):
    """
    Charge le fichier .env temporaire et retourne les variables d'environnement.
    """
    load_dotenv(temp_env_file, override=True)
    return {
        "IQFEED_API_KEY": os.getenv("IQFEED_API_KEY"),
        "NEWS_API_KEY": os.getenv("NEWS_API_KEY"),
    }


def test_load_env_valid(setup_env):
    """
    Teste le chargement des variables d'environnement valides.
    """
    env_vars = setup_env

    # Vérifier que les variables sont présentes et correctes
    assert "IQFEED_API_KEY" in env_vars
    assert env_vars["IQFEED_API_KEY"] == "test_iqfeed_key"
    assert "NEWS_API_KEY" in env_vars
    assert env_vars["NEWS_API_KEY"] == "test_news_key"


def test_load_env_missing_file():
    """
    Teste le chargement lorsque le fichier .env est manquant.
    """
    # S'assurer qu'aucun fichier .env n'est chargé
    os.environ.pop("IQFEED_API_KEY", None)
    os.environ.pop("NEWS_API_KEY", None)

    # Simuler l'absence de fichier .env
    with pytest.raises(FileNotFoundError, match="Fichier .env introuvable"):
        load_dotenv("non_existent.env")

    # Vérifier que les variables ne sont pas définies
    assert os.getenv("IQFEED_API_KEY") is None
    assert os.getenv("NEWS_API_KEY") is None


def test_load_env_missing_variables(temp_env_file):
    """
    Teste le chargement lorsque des variables sont manquantes dans .env.
    """
    # Créer un fichier .env sans variables
    with open(temp_env_file, "w") as f:
        f.write("")  # Fichier vide

    load_dotenv(temp_env_file, override=True)

    # Vérifier que les variables ne sont pas définies
    assert os.getenv("IQFEED_API_KEY") is None
    assert os.getenv("NEWS_API_KEY") is None


def test_load_env_invalid_format(temp_env_file):
    """
    Teste le chargement d'un fichier .env avec un format invalide.
    """
    # Créer un fichier .env avec un format incorrect
    with open(temp_env_file, "w") as f:
        f.write("INVALID_LINE_WITHOUT_EQUALS")

    # Charger le fichier (ne devrait pas lever d'erreur, mais ignorer les
    # lignes invalides)
    load_dotenv(temp_env_file, override=True)

    # Vérifier que les variables ne sont pas définies
    assert os.getenv("IQFEED_API_KEY") is None
    assert os.getenv("NEWS_API_KEY") is None


def test_load_env_variable_validation(setup_env):
    """
    Teste la validation des variables d'environnement chargées.
    """
    env_vars = setup_env

    # Vérifier que les clés ne sont pas vides
    assert env_vars["IQFEED_API_KEY"] != ""
    assert env_vars["NEWS_API_KEY"] != ""

    # Vérifier que les clés sont des chaînes
    assert isinstance(env_vars["IQFEED_API_KEY"], str)
    assert isinstance(env_vars["NEWS_API_KEY"], str)


def test_load_env_with_find_dotenv():
    """
    Teste le chargement automatique avec find_dotenv.
    """
    # S'assurer que l'environnement est propre
    os.environ.pop("IQFEED_API_KEY", None)
    os.environ.pop("NEWS_API_KEY", None)

    # Créer un fichier .env dans le répertoire de test
    env_path = Path("D:/MIA_IA_SYSTEM_v2_2025/data/.env")
    if env_path.exists():
        # Simuler un fichier .env temporaire pour éviter de modifier le vrai
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False, dir="D:/MIA_IA_SYSTEM_v2_2025/data"
        ) as f:
            f.write("IQFEED_API_KEY=test_iqfeed_key\nNEWS_API_KEY=test_news_key")
            temp_path = Path(f.name)

        try:
            load_dotenv(find_dotenv())
            assert os.getenv("IQFEED_API_KEY") == "test_iqfeed_key"
            assert os.getenv("NEWS_API_KEY") == "test_news_key"
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    else:
        # Si .env n'existe pas, find_dotenv ne devrait rien charger
        load_dotenv(find_dotenv(raise_error_if_not_found=False))
        assert os.getenv("IQFEED_API_KEY") is None
        assert os.getenv("NEWS_API_KEY") is None
