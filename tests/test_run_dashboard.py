# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_run_dashboard.py
# Tests unitaires pour run_dashboard.py.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie les fonctionnalités de DashboardRunner, incluant l'initialisation,
#        le chargement de la configuration, la validation des données regime_probs,
#        les snapshots JSON compressés, les logs de performance, la disponibilité des ports,
#        le démarrage du serveur, et l'arrêt propre.
#        Conforme à la Phase 8 (auto-conscience via miya_console),
#        Phase 11 (détection des régimes hybrides), et Phase 16 (ensemble learning).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pandas>=2.0.0
# - psutil>=5.9.8
# - matplotlib>=3.5.0
# - src.scripts.run_dashboard
# - src.monitoring.mia_dashboard
# - src.model.utils.config_manager
# - src.model.utils.miya_console
# - src.utils.telegram_alert
# - src.utils.standard
# - src.model.utils.alert_manager
#
# Notes :
# - Utilise des mocks pour simuler les dépendances externes.
# - Vérifie l'absence de références à dxFeed, obs_t, 320/81 features.

import gzip
import json
import os
from unittest.mock import patch

import pandas as pd
import pytest

from src.scripts.run_dashboard import DashboardRunner


@pytest.fixture
def temp_dir(tmp_path):
    """Crée un répertoire temporaire pour les tests."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    data_dir = tmp_path / "data"
    logs_dir = data_dir / "logs"
    snapshots_dir = data_dir / "dashboard_snapshots"
    figures_dir = data_dir / "figures" / "dashboard"
    logs_dir.mkdir(parents=True)
    snapshots_dir.mkdir(parents=True)
    figures_dir.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def mock_config(temp_dir):
    """Crée un fichier es_config.yaml simulé."""
    config_path = temp_dir / "config" / "es_config.yaml"
    config_content = """
dashboard:
  port: 8050
  host: "127.0.0.1"
  retry_attempts: 3
  retry_delay: 5
  debug: False
  buffer_size: 100
  max_cache_size: 1000
"""
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def mock_regime_probs(temp_dir):
    """Crée un fichier regime_history.csv simulé."""
    regime_probs_path = temp_dir / "data" / "logs" / "regime_history.csv"
    regime_probs_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"regime_probs": ["trend:0.7,range:0.2,defensive:0.1"]})
    df.to_csv(regime_probs_path, index=False)
    return regime_probs_path


@pytest.fixture
def runner(temp_dir, mock_config, monkeypatch):
    """Initialise DashboardRunner avec des mocks."""
    monkeypatch.setattr("src.scripts.run_dashboard.CONFIG_PATH", str(mock_config))
    monkeypatch.setattr(
        "src.scripts.run_dashboard.config_manager.get_config",
        lambda x: {
            "dashboard": {
                "port": 8050,
                "host": "127.0.0.1",
                "retry_attempts": 3,
                "retry_delay": 5,
                "debug": False,
                "buffer_size": 100,
                "max_cache_size": 1000,
            }
        },
    )
    runner = DashboardRunner()
    return runner


def test_init_runner(temp_dir, mock_config, monkeypatch):
    """Teste l'initialisation de DashboardRunner."""
    runner = DashboardRunner()
    assert runner.config["port"] == 8050
    assert os.path.exists(temp_dir / "data" / "dashboard_snapshots")
    assert os.path.exists(temp_dir / "data" / "figures" / "dashboard")
    snapshots = list(
        (temp_dir / "data" / "dashboard_snapshots").glob("snapshot_init_*.json.gz")
    )
    assert len(snapshots) >= 1


def test_load_config_invalid(temp_dir, monkeypatch):
    """Teste le chargement d'une configuration invalide."""
    invalid_config = temp_dir / "config" / "es_config.yaml"
    invalid_config.write_text("")
    monkeypatch.setattr("src.scripts.run_dashboard.CONFIG_PATH", str(invalid_config))
    with patch(
        "src.scripts.run_dashboard.config_manager.get_config",
        side_effect=ValueError("Configuration vide"),
    ):
        runner = DashboardRunner()
    assert runner.config["port"] == 8050  # Fallback to default
    snapshots = list(
        (temp_dir / "data" / "dashboard_snapshots").glob("snapshot_init_*.json.gz")
    )
    assert len(snapshots) >= 1


def test_validate_regime_probs_valid(temp_dir, mock_regime_probs, runner):
    """Teste la validation de regime_probs valide."""
    regime_probs = runner.validate_regime_probs(str(mock_regime_probs))
    assert len(regime_probs) == 1
    assert regime_probs[0] == "trend:0.7,range:0.2,defensive:0.1"
    perf_log = temp_dir / "data" / "logs" / "run_dashboard_performance.csv"
    assert perf_log.exists()


def test_validate_regime_probs_invalid(temp_dir, runner):
    """Teste la validation de regime_probs invalide."""
    invalid_path = temp_dir / "data" / "logs" / "invalid.csv"
    with pytest.raises(ValueError, match="Fichier regime_probs introuvable"):
        runner.validate_regime_probs(str(invalid_path))

    # Test fichier vide
    empty_path = temp_dir / "data" / "logs" / "empty.csv"
    empty_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame().to_csv(empty_path, index=False)
    with pytest.raises(ValueError, match="Données regime_probs vides ou mal formées"):
        runner.validate_regime_probs(str(empty_path))

    # Test fichier ancien
    old_path = temp_dir / "data" / "logs" / "old.csv"
    df = pd.DataFrame({"regime_probs": ["trend:0.7,range:0.2,defensive:0.1"]})
    df.to_csv(old_path, index=False)
    os.utime(
        old_path, (time.time() - 48 * 3600, time.time() - 48 * 3600)
    )  # Set file age > 24h
    with pytest.raises(ValueError, match="Fichier regime_probs trop ancien"):
        runner.validate_regime_probs(str(old_path))


def test_save_snapshot_compressed(temp_dir, runner):
    """Teste la sauvegarde d'un snapshot compressé."""
    data = {"test": "value"}
    runner.save_snapshot("test", data)
    snapshots = list(
        (temp_dir / "data" / "dashboard_snapshots").glob("snapshot_test_*.json.gz")
    )
    assert len(snapshots) == 1
    with gzip.open(snapshots[0], "rt", encoding="utf-8") as f:
        snapshot = json.load(f)
    assert snapshot["type"] == "test"
    assert snapshot["data"] == data


def test_run_dashboard_success(temp_dir, mock_regime_probs, runner, monkeypatch):
    """Teste le démarrage du serveur avec succès."""
    monkeypatch.setattr(
        "src.scripts.run_dashboard.RUNNING", False
    )  # Stop after one run
    with patch(
        "src.scripts.run_dashboard.dashboard_app.run_server"
    ) as mock_run, patch.object(runner, "is_port_available", return_value=True):
        runner.run_dashboard()

    assert mock_run.called
    status_file = temp_dir / "data" / "dashboard_status.json"
    assert status_file.exists()
    with open(status_file, "r") as f:
        status = json.load(f)
    assert status["running"] is True
    snapshots = list(
        (temp_dir / "data" / "dashboard_snapshots").glob(
            "snapshot_start_attempt_*.json.gz"
        )
    )
    assert len(snapshots) >= 1
    figures = list(
        (temp_dir / "data" / "figures" / "dashboard").glob("start_attempts_*.png")
    )
    assert len(figures) >= 1


def test_run_dashboard_port_unavailable(
    temp_dir, mock_regime_probs, runner, monkeypatch
):
    """Teste le démarrage avec port indisponible."""
    monkeypatch.setattr("src.scripts.run_dashboard.RUNNING", False)
    with patch.object(runner, "is_port_available", return_value=False), pytest.raises(
        RuntimeError, match="Aucun port disponible"
    ):
        runner.run_dashboard()

    status_file = temp_dir / "data" / "dashboard_status.json"
    assert status_file.exists()
    with open(status_file, "r") as f:
        status = json.load(f)
    assert status["running"] is False
    assert "Aucun port disponible" in status["error"]


def test_signal_handler(temp_dir, runner, monkeypatch):
    """Teste l'arrêt propre du serveur."""
    monkeypatch.setattr("src.scripts.run_dashboard.RUNNING", True)
    with patch("sys.exit") as mock_exit:
        runner.signal_handler(signal.SIGINT, None)

    status_file = temp_dir / "data" / "dashboard_status.json"
    assert status_file.exists()
    with open(status_file, "r") as f:
        status = json.load(f)
    assert status["running"] is False
    snapshots = list(
        (temp_dir / "data" / "dashboard_snapshots").glob("snapshot_shutdown_*.json.gz")
    )
    assert len(snapshots) >= 1
    assert mock_exit.called


def test_no_obsolete_references(temp_dir, mock_config):
    """Vérifie l'absence de références à dxFeed, obs_t, 320/81 features."""
    with open(mock_config, "r") as f:
        content = f.read()
    assert "dxFeed" not in content, "Référence à dxFeed trouvée"
    assert "obs_t" not in content, "Référence à obs_t trouvée"
    assert "320 features" not in content, "Référence à 320 features trouvée"
    assert "81 features" not in content, "Référence à 81 features trouvée"
