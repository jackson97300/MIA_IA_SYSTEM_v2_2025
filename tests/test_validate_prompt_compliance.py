# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_validate_prompt_compliance.py
# Tests unitaires pour validate_prompt_compliance.py.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie les fonctionnalités de PromptComplianceValidator, incluant l'initialisation,
#        la validation des fichiers Python, l’analyse des conformités (version, date, dépendances,
#        retries, logs, snapshots, alertes, Phases, tests), les retries, les snapshots JSON compressés,
#        les logs de performance, et les alertes.
#        Conforme à la Phase 8 (auto-conscience via alertes) et Phase 15 (automatisation et validation).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - psutil>=5.9.8
# - pandas>=2.0.0
# - src.scripts.validate_prompt_compliance
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

import pytest

from src.scripts.validate_prompt_compliance import PromptComplianceValidator


@pytest.fixture
def temp_dir(tmp_path):
    """Crée un répertoire temporaire pour les tests."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    data_dir = tmp_path / "data"
    logs_dir = data_dir / "logs"
    snapshots_dir = data_dir / "validation_snapshots"
    logs_dir.mkdir(parents=True)
    snapshots_dir.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def mock_config(temp_dir):
    """Crée un fichier es_config.yaml simulé."""
    config_path = temp_dir / "config" / "es_config.yaml"
    config_path.write_text("validate:\n  enabled: true\n")
    return config_path


@pytest.fixture
def mock_files(temp_dir):
    """Crée des fichiers Python simulés pour les tests."""
    compliant_file = temp_dir / "src" / "compliant.py"
    compliant_file.write_text(
        """
# -*- coding: utf-8 -*-
# Version : 2.1.3
# Date : 2025-05-13
# Tests unitaires dans tests/test_compliant.py
# Phase 15

import psutil
import config_manager
import alert_manager
import miya_console
import telegram_alert
import standard
from src.utils.standard import with_retries

@with_retries(max_attempts=3)
def example():
    psutil.cpu_percent()
    miya_console.miya_speak("Test")
    alert_manager.AlertManager().send_alert("Test")
    with gzip.open("snapshot.json.gz", "wt") as f:
        json.dump({}, f)
"""
    )
    non_compliant_file = temp_dir / "src" / "non_compliant.py"
    non_compliant_file.write_text(
        """
# -*- coding: utf-8 -*-
# Version : 2.0.0
import dxFeed
import obs_t
"""
    )
    return [compliant_file, non_compliant_file]


@pytest.fixture
def validator(temp_dir, mock_config, mock_files, monkeypatch):
    """Initialise PromptComplianceValidator avec des mocks."""
    monkeypatch.setattr(
        "src.scripts.validate_prompt_compliance.CONFIG_PATH", str(mock_config)
    )
    monkeypatch.setattr(
        "src.scripts.validate_prompt_compliance.config_manager.get_config",
        lambda x: {"validate": {"enabled": True}},
    )
    validator = PromptComplianceValidator(
        config_path=mock_config, src_dir=temp_dir / "src", tests_dir=temp_dir / "tests"
    )
    return validator


def test_init_validator(temp_dir, mock_config, monkeypatch):
    """Teste l'initialisation de PromptComplianceValidator."""
    validator = PromptComplianceValidator(
        config_path=mock_config, src_dir=temp_dir / "src", tests_dir=temp_dir / "tests"
    )
    assert validator.config_path == mock_config
    assert os.path.exists(temp_dir / "data" / "validation_snapshots")
    snapshots = list(
        (temp_dir / "data" / "validation_snapshots").glob("snapshot_init_*.json.gz")
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "validate_prompt_compliance_performance.csv"
    assert perf_log.exists()


def test_validate_config_invalid(temp_dir, monkeypatch):
    """Teste la validation avec une configuration invalide."""
    invalid_config = temp_dir / "config" / "invalid.yaml"
    with pytest.raises(FileNotFoundError, match="Fichier de configuration introuvable"):
        PromptComplianceValidator(
            config_path=invalid_config,
            src_dir=temp_dir / "src",
            tests_dir=temp_dir / "tests",
        )

    invalid_src = temp_dir / "invalid_src"
    with pytest.raises(FileNotFoundError, match="Répertoire source introuvable"):
        PromptComplianceValidator(
            config_path=temp_dir / "config" / "es_config.yaml",
            src_dir=invalid_src,
            tests_dir=temp_dir / "tests",
        )


def test_get_python_files(temp_dir, mock_files, validator):
    """Teste la récupération des fichiers Python."""
    files = validator.get_python_files()
    assert len(files) == 2
    assert all(f in files for f in mock_files)
    snapshots = list(
        (temp_dir / "data" / "validation_snapshots").glob(
            "snapshot_get_python_files_*.json.gz"
        )
    )
    assert len(snapshots) >= 1


def test_validate_file_compliant(temp_dir, mock_files, validator):
    """Teste la validation d’un fichier conforme."""
    result = validator.validate_file(mock_files[0])
    assert result["compliant"] is True
    assert len(result["issues"]) == 0
    snapshots = list(
        (temp_dir / "data" / "validation_snapshots").glob(
            "snapshot_validate_file_*.json.gz"
        )
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "validate_prompt_compliance_performance.csv"
    assert perf_log.exists()


def test_validate_file_non_compliant(temp_dir, mock_files, validator):
    """Teste la validation d’un fichier non conforme."""
    result = validator.validate_file(mock_files[1])
    assert result["compliant"] is False
    assert len(result["issues"]) > 0
    assert any("Version ou date incorrecte" in issue for issue in result["issues"])
    assert any("Imports manquants" in issue for issue in result["issues"])
    assert any(
        "Référence interdite détectée : dxFeed" in issue for issue in result["issues"]
    )
    snapshots = list(
        (temp_dir / "data" / "validation_snapshots").glob(
            "snapshot_validate_file_*.json.gz"
        )
    )
    assert len(snapshots) >= 1


def test_validate_all_files(temp_dir, mock_files, validator):
    """Teste la validation de tous les fichiers."""
    report = validator.validate_all_files()
    assert report["total_files"] == 2
    assert report["compliant_files"] == 1
    assert len(report["non_compliant_files"]) == 1
    assert report["compliance_rate"] == 0.5
    snapshots = list(
        (temp_dir / "data" / "validation_snapshots").glob(
            "snapshot_validate_all_files_*.json.gz"
        )
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "validate_prompt_compliance_performance.csv"
    assert perf_log.exists()


def test_no_obsolete_references(temp_dir, mock_config):
    """Vérifie l'absence de références à dxFeed, obs_t, 320/81 features."""
    with open(mock_config, "r") as f:
        content = f.read()
    assert "dxFeed" not in content, "Référence à dxFeed trouvée"
    assert "obs_t" not in content, "Référence à obs_t trouvée"
    assert "320 features" not in content, "Référence à 320 features trouvée"
    assert "81 features" not in content, "Référence à 81 features trouvée"
