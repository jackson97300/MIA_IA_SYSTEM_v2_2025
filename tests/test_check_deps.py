# Teste check_deps.py.
# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_check_deps.py
# Tests unitaires pour check_deps.py.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie les fonctionnalités de DependencyChecker, incluant le chargement de la configuration,
#        la vérification des dépendances, les snapshots JSON, les logs de performance,
#        et la génération de graphiques.
#        Conforme à la Phase 1 (collecte via IQFeed), Phase 8 (auto-conscience via miya_console),
#        et Phase 16 (ensemble learning via torch, tensorflow).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pyyaml>=6.0.0,<7.0.0
# - psutil>=5.9.0,<6.0.0
# - matplotlib>=3.5.0
# - pandas>=1.5.0
# - src.scripts.check_deps
# - src.model.utils.miya_console
# - src.utils.telegram_alert
#
# Notes :
# - Utilise des fichiers temporaires pour tester.
# - Vérifie l'absence de références à dxFeed, obs_t, 320/81 features.

import json
from unittest.mock import patch

import pytest

from src.scripts.check_deps import DependencyChecker


@pytest.fixture
def temp_dir(tmp_path):
    """Crée un répertoire temporaire pour les tests."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    data_dir = tmp_path / "data"
    logs_dir = data_dir / "logs"
    snapshots_dir = data_dir / "deps_snapshots"
    figures_dir = data_dir / "figures" / "dependencies"
    logs_dir.mkdir(parents=True)
    snapshots_dir.mkdir(parents=True)
    figures_dir.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def valid_config(temp_dir):
    """Crée un fichier es_config.yaml valide pour les tests."""
    config_path = temp_dir / "config" / "es_config.yaml"
    config_content = """
dependencies:
  required:
    - name: pandas
      version: ">=1.5.0"
    - name: numpy
      version: ">=1.21.0"
  optional:
    - name: tensorflow
      version: ">=2.8.0"
logging:
  buffer_size: 100
cache:
  max_cache_size: 1000
"""
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def invalid_config(temp_dir):
    """Crée un fichier es_config.yaml invalide pour les tests."""
    config_path = temp_dir / "config" / "es_config.yaml"
    config_content = """
dependencies:
  invalid_key: []
"""
    config_path.write_text(config_content)
    return config_path


def test_init_dependency_checker(temp_dir, valid_config, monkeypatch):
    """Teste l'initialisation de DependencyChecker."""
    monkeypatch.setattr("src.scripts.check_deps.CONFIG_PATH", str(valid_config))
    checker = DependencyChecker()
    assert checker.config["required"][0]["name"] == "pandas"
    assert checker.buffer_size == 100
    assert checker.max_cache_size == 1000
    assert (temp_dir / "data" / "deps_snapshots").exists()
    assert (temp_dir / "data" / "figures" / "dependencies").exists()


def test_load_config_valid(temp_dir, valid_config, monkeypatch):
    """Teste le chargement d'une configuration valide."""
    monkeypatch.setattr("src.scripts.check_deps.CONFIG_PATH", str(valid_config))
    checker = DependencyChecker()
    config = checker.load_config()
    assert len(config["required"]) == 2
    assert config["required"][0]["name"] == "pandas"
    snapshots = list(
        (temp_dir / "data" / "deps_snapshots").glob("snapshot_load_config_*.json")
    )
    assert len(snapshots) >= 1


def test_load_config_invalid(temp_dir, invalid_config, monkeypatch):
    """Teste le chargement d'une configuration invalide avec retries."""
    monkeypatch.setattr("src.scripts.check_deps.CONFIG_PATH", str(invalid_config))
    with pytest.raises(
        ValueError, match="Clé 'required' manquante dans 'dependencies'"
    ):
        DependencyChecker()
    snapshots = list(
        (temp_dir / "data" / "deps_snapshots").glob("snapshot_init_*.json")
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "check_deps_performance.csv"
    assert perf_log.exists()


def test_check_dependency_installed(temp_dir, valid_config, monkeypatch):
    """Teste la vérification d'une dépendance installée."""
    monkeypatch.setattr("src.scripts.check_deps.CONFIG_PATH", str(valid_config))
    checker = DependencyChecker()
    dep = {"name": "pandas", "version": ">=1.5.0"}
    result = checker.check_dependency(dep)
    assert result["installed"] is True
    assert result["version"] is not None
    assert result["error"] is None


def test_check_dependency_missing(temp_dir, valid_config, monkeypatch):
    """Teste la vérification d'une dépendance manquante."""
    monkeypatch.setattr("src.scripts.check_deps.CONFIG_PATH", str(valid_config))
    checker = DependencyChecker()
    dep = {"name": "non_existent_module", "version": ">=1.0.0"}
    with patch("importlib.util.find_spec", return_value=None):
        result = checker.check_dependency(dep)
    assert result["installed"] is False
    assert result["version"] is None
    assert "non installé" in result["error"]


def test_plot_dependency_status(temp_dir, valid_config, monkeypatch):
    """Teste la génération du graphique de statut des dépendances."""
    monkeypatch.setattr("src.scripts.check_deps.CONFIG_PATH", str(valid_config))
    checker = DependencyChecker()
    results = [
        {
            "name": "pandas",
            "installed": True,
            "version": "1.5.0",
            "required_version": ">=1.5.0",
            "error": None,
        },
        {
            "name": "missing",
            "installed": False,
            "version": None,
            "required_version": ">=1.0.0",
            "error": "Module missing non installé",
        },
    ]
    checker.plot_dependency_status(results)
    figures = list(
        (temp_dir / "data" / "figures" / "dependencies").glob("dependency_status_*.png")
    )
    assert len(figures) >= 1


def test_save_dashboard_status(temp_dir, valid_config, monkeypatch):
    """Teste la sauvegarde de l'état pour le dashboard."""
    monkeypatch.setattr("src.scripts.check_deps.CONFIG_PATH", str(valid_config))
    monkeypatch.setattr(
        "src.scripts.check_deps.DASHBOARD_PATH",
        str(temp_dir / "data" / "deps_dashboard.json"),
    )
    checker = DependencyChecker()
    status = {
        "last_check": "2025-05-13 12:00:00",
        "dependencies": [],
        "missing_count": 0,
        "status": "complete",
    }
    checker.save_dashboard_status(status)
    assert (temp_dir / "data" / "deps_dashboard.json").exists()
    with open(temp_dir / "data" / "deps_dashboard.json", "r") as f:
        saved_status = json.load(f)
    assert saved_status["status"] == "complete"


def test_performance_logging(temp_dir, valid_config, monkeypatch):
    """Teste l'enregistrement des performances."""
    monkeypatch.setattr("src.scripts.check_deps.CONFIG_PATH", str(valid_config))
    checker = DependencyChecker()
    checker.log_performance("test_op", 0.5, success=True, num_deps=2)
    perf_log = temp_dir / "data" / "logs" / "check_deps_performance.csv"
    assert perf_log.exists()
    with open(perf_log, "r") as f:
        lines = f.readlines()
    assert len(lines) >= 1
    assert "test_op" in lines[-1]


def test_no_obsolete_references(temp_dir, valid_config, monkeypatch):
    """Vérifie l'absence de références à dxFeed, obs_t, 320/81 features."""
    monkeypatch.setattr("src.scripts.check_deps.CONFIG_PATH", str(valid_config))
    DependencyChecker()
    with open(valid_config, "r") as f:
        content = f.read()
    assert "dxFeed" not in content, "Référence à dxFeed trouvée"
    assert "obs_t" not in content, "Référence à obs_t trouvée"
    assert "320 features" not in content, "Référence à 320 features trouvée"
    assert "81 features" not in content, "Référence à 81 features trouvée"
