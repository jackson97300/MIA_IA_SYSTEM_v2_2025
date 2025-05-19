# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_setup_env.py
# Tests unitaires pour setup_env.py.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie les fonctionnalités de EnvironmentSetup, incluant l'initialisation,
#        la vérification de la version de Python, la validation de requirements.txt,
#        l'installation des dépendances, la vérification des dépendances,
#        les snapshots JSON compressés, les logs de performance, et les alertes.
#        Conforme à la Phase 8 (auto-conscience via alertes) et Phase 15 (automatisation de l’environnement).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pandas>=2.0.0
# - psutil>=5.9.8
# - src.scripts.setup_env
# - src.model.utils.alert_manager
# - src.model.utils.miya_console
# - src.utils.telegram_alert
# - src.utils.standard
#
# Notes :
# - Utilise des mocks pour simuler les dépendances externes.
# - Vérifie l'absence de références à dxFeed, obs_t, 320/81 features.

import os
from unittest.mock import MagicMock, patch

import pkg_resources
import pytest

from src.scripts.setup_env import EnvironmentSetup


@pytest.fixture
def temp_dir(tmp_path):
    """Crée un répertoire temporaire pour les tests."""
    data_dir = tmp_path / "data"
    logs_dir = data_dir / "logs"
    snapshots_dir = data_dir / "setup_snapshots"
    logs_dir.mkdir(parents=True)
    snapshots_dir.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def mock_requirements(temp_dir):
    """Crée un fichier requirements.txt simulé."""
    req_path = temp_dir / "requirements.txt"
    req_content = """
pandas>=2.0.0
numpy>=1.23.0
pyyaml>=6.0.0
"""
    req_path.write_text(req_content)
    return req_path


def test_init_setup(temp_dir, mock_requirements, monkeypatch):
    """Teste l'initialisation de EnvironmentSetup."""
    monkeypatch.setattr(
        "src.scripts.setup_env.REQUIREMENTS_PATH", str(mock_requirements)
    )
    setup = EnvironmentSetup()
    assert setup.requirements_path == mock_requirements
    assert os.path.exists(temp_dir / "data" / "setup_snapshots")
    snapshots = list(
        (temp_dir / "data" / "setup_snapshots").glob("snapshot_init_*.json.gz")
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "setup_env_performance.csv"
    assert perf_log.exists()


def test_check_python_version_valid(temp_dir, mock_requirements, monkeypatch):
    """Teste la vérification d'une version Python valide."""
    monkeypatch.setattr(
        "src.scripts.setup_env.REQUIREMENTS_PATH", str(mock_requirements)
    )
    setup = EnvironmentSetup()
    with patch("sys.version_info", (3, 8)):
        assert setup.check_python_version() is True
    snapshots = list(
        (temp_dir / "data" / "setup_snapshots").glob(
            "snapshot_check_python_version_*.json.gz"
        )
    )
    assert len(snapshots) >= 1


def test_check_python_version_invalid(temp_dir, mock_requirements, monkeypatch):
    """Teste la vérification d'une version Python invalide."""
    monkeypatch.setattr(
        "src.scripts.setup_env.REQUIREMENTS_PATH", str(mock_requirements)
    )
    setup = EnvironmentSetup()
    with patch("sys.version_info", (3, 7)):
        assert setup.check_python_version() is False
    snapshots = list(
        (temp_dir / "data" / "setup_snapshots").glob(
            "snapshot_check_python_version_*.json.gz"
        )
    )
    assert len(snapshots) >= 1


def test_validate_requirements_file_valid(temp_dir, mock_requirements, monkeypatch):
    """Teste la validation d'un fichier requirements.txt valide."""
    monkeypatch.setattr(
        "src.scripts.setup_env.REQUIREMENTS_PATH", str(mock_requirements)
    )
    setup = EnvironmentSetup()
    assert setup.validate_requirements_file() is True
    snapshots = list(
        (temp_dir / "data" / "setup_snapshots").glob(
            "snapshot_validate_requirements_file_*.json.gz"
        )
    )
    assert len(snapshots) >= 1


def test_validate_requirements_file_invalid(temp_dir, monkeypatch):
    """Teste la validation d'un fichier requirements.txt invalide."""
    invalid_req = temp_dir / "invalid.txt"
    invalid_req.write_text("")  # Fichier vide
    monkeypatch.setattr("src.scripts.setup_env.REQUIREMENTS_PATH", str(invalid_req))
    setup = EnvironmentSetup()
    assert setup.validate_requirements_file() is False

    malformed_req = temp_dir / "malformed.txt"
    malformed_req.write_text("invalid_package\n")  # Ligne mal formée
    monkeypatch.setattr("src.scripts.setup_env.REQUIREMENTS_PATH", str(malformed_req))
    setup = EnvironmentSetup()
    assert setup.validate_requirements_file() is False
    snapshots = list(
        (temp_dir / "data" / "setup_snapshots").glob(
            "snapshot_validate_requirements_file_*.json.gz"
        )
    )
    assert len(snapshots) >= 2


def test_install_requirements_success(temp_dir, mock_requirements, monkeypatch):
    """Teste l'installation des dépendances avec succès."""
    monkeypatch.setattr(
        "src.scripts.setup_env.REQUIREMENTS_PATH", str(mock_requirements)
    )
    setup = EnvironmentSetup()
    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
        assert setup.install_requirements() is True
    snapshots = list(
        (temp_dir / "data" / "setup_snapshots").glob(
            "snapshot_install_requirements_*.json.gz"
        )
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "setup_env_performance.csv"
    assert perf_log.exists()


def test_install_requirements_failure(temp_dir, mock_requirements, monkeypatch):
    """Teste l'installation des dépendances avec échec."""
    monkeypatch.setattr(
        "src.scripts.setup_env.REQUIREMENTS_PATH", str(mock_requirements)
    )
    setup = EnvironmentSetup()
    with patch(
        "subprocess.run", return_value=MagicMock(returncode=1, stderr="Erreur pip")
    ):
        assert setup.install_requirements() is False
    snapshots = list(
        (temp_dir / "data" / "setup_snapshots").glob(
            "snapshot_install_requirements_*.json.gz"
        )
    )
    assert len(snapshots) >= 1


def test_verify_dependencies_valid(temp_dir, mock_requirements, monkeypatch):
    """Teste la vérification des dépendances valides."""
    monkeypatch.setattr(
        "src.scripts.setup_env.REQUIREMENTS_PATH", str(mock_requirements)
    )
    setup = EnvironmentSetup()
    required_packages = [("pandas", "2.0.0"), ("numpy", "1.23.0")]
    with patch(
        "pkg_resources.get_distribution",
        side_effect=lambda x: MagicMock(version="2.0.0" if x == "pandas" else "1.23.0"),
    ):
        assert setup.verify_dependencies(required_packages) is True
    snapshots = list(
        (temp_dir / "data" / "setup_snapshots").glob(
            "snapshot_verify_dependencies_*.json.gz"
        )
    )
    assert len(snapshots) >= 1


def test_verify_dependencies_invalid(temp_dir, mock_requirements, monkeypatch):
    """Teste la vérification des dépendances invalides."""
    monkeypatch.setattr(
        "src.scripts.setup_env.REQUIREMENTS_PATH", str(mock_requirements)
    )
    setup = EnvironmentSetup()
    required_packages = [("pandas", "2.0.0"), ("missing", "1.0.0")]
    with patch(
        "pkg_resources.get_distribution",
        side_effect=lambda x: (
            MagicMock(version="2.0.0")
            if x == "pandas"
            else pkg_resources.DistributionNotFound
        ),
    ):
        assert setup.verify_dependencies(required_packages) is False
    snapshots = list(
        (temp_dir / "data" / "setup_snapshots").glob(
            "snapshot_verify_dependencies_*.json.gz"
        )
    )
    assert len(snapshots) >= 1


def test_setup_environment_success(temp_dir, mock_requirements, monkeypatch):
    """Teste la configuration complète de l'environnement avec succès."""
    monkeypatch.setattr(
        "src.scripts.setup_env.REQUIREMENTS_PATH", str(mock_requirements)
    )
    setup = EnvironmentSetup()
    with patch("sys.version_info", (3, 8)), patch(
        "subprocess.run", return_value=MagicMock(returncode=0)
    ), patch("pkg_resources.get_distribution", return_value=MagicMock(version="2.0.0")):
        assert setup.setup_environment() is True
    snapshots = list(
        (temp_dir / "data" / "setup_snapshots").glob(
            "snapshot_setup_environment_*.json.gz"
        )
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "setup_env_performance.csv"
    assert perf_log.exists()


def test_no_obsolete_references(temp_dir, mock_requirements):
    """Vérifie l'absence de références à dxFeed, obs_t, 320/81 features."""
    with open(mock_requirements, "r") as f:
        content = f.read()
    assert "dxFeed" not in content, "Référence à dxFeed trouvée"
    assert "obs_t" not in content, "Référence à obs_t trouvée"
    assert "320 features" not in content, "Référence à 320 features trouvée"
    assert "81 features" not in content, "Référence à 81 features trouvée"
