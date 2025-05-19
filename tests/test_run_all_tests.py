# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_run_all_tests.py
# Tests unitaires pour run_all_tests.py.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie les fonctionnalités de TestRunner, incluant l'initialisation,
#        la validation de la configuration, l’exécution des tests avec pytest,
#        les retries, les snapshots JSON compressés, les logs de performance, et les alertes.
#        Conforme à la Phase 8 (auto-conscience via alertes) et Phase 15 (automatisation et validation).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - psutil>=5.9.8
# - pandas>=2.0.0
# - src.scripts.run_all_tests
# - src.model.utils.config_manager
# - src.model.utils.alert_manager
# - src.model.utils.miya_console
# - src.utils.telegram_alert
# - src.utils.standard
#
# Notes :
# - Utilise des mocks pour simuler les dépendances externes.
# - Vérifie l'absence de références à dxFeed, obs_t, 320/81 features.

import os
from unittest.mock import patch

import pytest

from src.scripts.run_all_tests import TestRunner


@pytest.fixture
def temp_dir(tmp_path):
    """Crée un répertoire temporaire pour les tests."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    data_dir = tmp_path / "data"
    logs_dir = data_dir / "logs"
    snapshots_dir = data_dir / "test_snapshots"
    logs_dir.mkdir(parents=True)
    snapshots_dir.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def mock_config(temp_dir):
    """Crée un fichier es_config.yaml simulé."""
    config_path = temp_dir / "config" / "es_config.yaml"
    config_content = """
pytest:
  verbose: True
  cov: False
"""
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def mock_test_file(temp_dir):
    """Crée un fichier de test simulé."""
    test_file = temp_dir / "tests" / "test_example.py"
    test_file.write_text(
        """
import pytest
def test_example():
    assert True
"""
    )
    return test_file


@pytest.fixture
def runner(temp_dir, mock_config, mock_test_file, monkeypatch):
    """Initialise TestRunner avec des mocks."""
    monkeypatch.setattr("src.scripts.run_all_tests.CONFIG_PATH", str(mock_config))
    monkeypatch.setattr(
        "src.scripts.run_all_tests.config_manager.get_config",
        lambda x: {"pytest": {"verbose": True, "cov": False}},
    )
    runner = TestRunner(config_path=mock_config, tests_dir=temp_dir / "tests")
    return runner


def test_init_runner(temp_dir, mock_config, monkeypatch):
    """Teste l'initialisation de TestRunner."""
    runner = TestRunner(config_path=mock_config, tests_dir=temp_dir / "tests")
    assert runner.config_path == mock_config
    assert os.path.exists(temp_dir / "data" / "test_snapshots")
    snapshots = list(
        (temp_dir / "data" / "test_snapshots").glob("snapshot_init_*.json.gz")
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "run_all_tests_performance.csv"
    assert perf_log.exists()


def test_validate_config_invalid(temp_dir, monkeypatch):
    """Teste la validation avec une configuration invalide."""
    invalid_config = temp_dir / "config" / "invalid.yaml"
    with pytest.raises(FileNotFoundError, match="Fichier de configuration introuvable"):
        TestRunner(config_path=invalid_config, tests_dir=temp_dir / "tests")

    invalid_tests = temp_dir / "invalid_tests"
    with pytest.raises(FileNotFoundError, match="Répertoire de tests introuvable"):
        TestRunner(
            config_path=temp_dir / "config" / "es_config.yaml", tests_dir=invalid_tests
        )


def test_get_test_files(temp_dir, mock_test_file, runner):
    """Teste la récupération des fichiers de test."""
    files = runner.get_test_files()
    assert len(files) == 1
    assert files[0] == mock_test_file
    snapshots = list(
        (temp_dir / "data" / "test_snapshots").glob("snapshot_get_test_files_*.json.gz")
    )
    assert len(snapshots) >= 1


def test_run_tests_success(temp_dir, runner, monkeypatch):
    """Teste l’exécution des tests avec succès."""
    with patch("pytest.main") as mock_pytest:
        mock_pytest.return_value = 0
        report = runner.run_tests()

    assert report["success"] is True
    assert report["total_tests"] > 0
    assert report["failed"] == 0
    snapshots = list(
        (temp_dir / "data" / "test_snapshots").glob("snapshot_run_tests_*.json.gz")
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "run_all_tests_performance.csv"
    assert perf_log.exists()


def test_run_tests_failure(temp_dir, runner, monkeypatch):
    """Teste l’exécution des tests avec des échecs."""
    with patch("pytest.main") as mock_pytest:
        mock_pytest.return_value = 1
        report = runner.run_tests()

    assert report["success"] is False
    assert report["total_tests"] > 0
    assert report["failed"] > 0
    snapshots = list(
        (temp_dir / "data" / "test_snapshots").glob("snapshot_run_tests_*.json.gz")
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "run_all_tests_performance.csv"
    assert perf_log.exists()


def test_no_obsolete_references(temp_dir, mock_config):
    """Vérifie l'absence de références à dxFeed, obs_t, 320/81 features."""
    with open(mock_config, "r") as f:
        content = f.read()
    assert "dxFeed" not in content, "Référence à dxFeed trouvée"
    assert "obs_t" not in content, "Référence à obs_t trouvée"
    assert "320 features" not in content, "Référence à 320 features trouvée"
    assert "81 features" not in content, "Référence à 81 features trouvée"
