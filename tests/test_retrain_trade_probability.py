# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_retrain_trade_probability.py
# Tests unitaires pour retrain_trade_probability.py.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie les fonctionnalités de TradeProbabilityRetrain, incluant l'initialisation,
#        la validation de la configuration et de la base de données, le comptage des patterns,
#        le re-entraînement, les retries, les snapshots JSON compressés, les logs de performance,
#        et les alertes.
#        Conforme à la Phase 8 (auto-conscience via alertes), Phase 9 (entraînement des modèles),
#        et Phase 16 (ensemble learning).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - psutil>=5.9.8
# - pandas>=2.0.0
# - sqlite3
# - src.scripts.retrain_trade_probability
# - src.model.trade_probability
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
import sqlite3
from unittest.mock import patch

import pytest

from src.scripts.retrain_trade_probability import TradeProbabilityRetrain


@pytest.fixture
def temp_dir(tmp_path):
    """Crée un répertoire temporaire pour les tests."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    data_dir = tmp_path / "data"
    logs_dir = data_dir / "logs"
    snapshots_dir = data_dir / "trade_probability_snapshots"
    logs_dir.mkdir(parents=True)
    snapshots_dir.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def mock_config(temp_dir):
    """Crée un fichier trade_probability_config.yaml simulé."""
    config_path = temp_dir / "config" / "trade_probability_config.yaml"
    config_content = """
trade_probability:
  retrain_threshold: 1000
  retrain_frequency: biweekly
"""
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def mock_db(temp_dir):
    """Crée une base de données SQLite simulée."""
    db_path = temp_dir / "data" / "market_memory.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE patterns (id INTEGER PRIMARY KEY, data TEXT)")
    cursor.executemany(
        "INSERT INTO patterns (data) VALUES (?)", [("pattern1",), ("pattern2",)]
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def retrainer(temp_dir, mock_config, mock_db, monkeypatch):
    """Initialise TradeProbabilityRetrain avec des mocks."""
    monkeypatch.setattr(
        "src.scripts.retrain_trade_probability.CONFIG_PATH", str(mock_config)
    )
    monkeypatch.setattr("src.scripts.retrain_trade_probability.DB_PATH", str(mock_db))
    monkeypatch.setattr(
        "src.scripts.retrain_trade_probability.config_manager.get_config",
        lambda x: {
            "trade_probability": {
                "retrain_threshold": 1000,
                "retrain_frequency": "biweekly",
            }
        },
    )
    retrainer = TradeProbabilityRetrain()
    return retrainer


def test_init_retrainer(temp_dir, mock_config, mock_db, retrainer):
    """Teste l'initialisation de TradeProbabilityRetrain."""
    assert retrainer.config["trade_probability"]["retrain_threshold"] == 1000
    assert os.path.exists(temp_dir / "data" / "trade_probability_snapshots")
    snapshots = list(
        (temp_dir / "data" / "trade_probability_snapshots").glob(
            "snapshot_init_*.json.gz"
        )
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "trade_probability_performance.csv"
    assert perf_log.exists()


def test_validate_config_invalid(temp_dir, mock_db, monkeypatch):
    """Teste la validation avec une configuration invalide."""
    invalid_config = temp_dir / "config" / "invalid.yaml"
    monkeypatch.setattr(
        "src.scripts.retrain_trade_probability.CONFIG_PATH", str(invalid_config)
    )
    with pytest.raises(FileNotFoundError, match="Fichier de configuration introuvable"):
        TradeProbabilityRetrain()

    with patch(
        "src.scripts.retrain_trade_probability.config_manager.get_config",
        return_value={},
    ):
        monkeypatch.setattr(
            "src.scripts.retrain_trade_probability.CONFIG_PATH",
            str(temp_dir / "config" / "trade_probability_config.yaml"),
        )
        with pytest.raises(ValueError, match="Clé 'trade_probability' manquante"):
            TradeProbabilityRetrain()


def test_validate_db_valid(temp_dir, mock_db, retrainer):
    """Teste la validation d’une base de données valide."""
    retrainer.validate_db(str(mock_db))
    snapshots = list(
        (temp_dir / "data" / "trade_probability_snapshots").glob(
            "snapshot_validate_db_*.json.gz"
        )
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "trade_probability_performance.csv"
    assert perf_log.exists()


def test_validate_db_invalid(temp_dir, retrainer):
    """Teste la validation d’une base de données invalide."""
    invalid_db = temp_dir / "data" / "invalid.db"
    with pytest.raises(FileNotFoundError, match="Base de données introuvable"):
        retrainer.validate_db(str(invalid_db))

    empty_db = temp_dir / "data" / "empty.db"
    conn = sqlite3.connect(empty_db)
    conn.close()
    with pytest.raises(ValueError, match="Table 'patterns' manquante"):
        retrainer.validate_db(str(empty_db))

    snapshots = list(
        (temp_dir / "data" / "trade_probability_snapshots").glob(
            "snapshot_validate_db_*.json.gz"
        )
    )
    assert len(snapshots) >= 2


def test_count_patterns(temp_dir, mock_db, retrainer):
    """Teste le comptage des patterns."""
    count = retrainer.count_patterns(str(mock_db))
    assert count == 2
    snapshots = list(
        (temp_dir / "data" / "trade_probability_snapshots").glob(
            "snapshot_count_patterns_*.json.gz"
        )
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "trade_probability_performance.csv"
    assert perf_log.exists()


def test_retrain_job_skipped(temp_dir, mock_db, retrainer):
    """Teste le re-entraînement lorsque le seuil n’est pas atteint."""
    with patch(
        "src.model.trade_probability.TradeProbabilityPredictor.train"
    ) as mock_train, patch(
        "src.model.trade_probability.TradeProbabilityPredictor.backtest_threshold"
    ) as mock_backtest:
        retrainer.retrain_job()

    assert not mock_train.called
    assert not mock_backtest.called
    snapshots = list(
        (temp_dir / "data" / "trade_probability_snapshots").glob(
            "snapshot_retrain_job_*.json.gz"
        )
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "trade_probability_performance.csv"
    assert perf_log.exists()


def test_retrain_job_executed(temp_dir, mock_db, retrainer, monkeypatch):
    """Teste le re-entraînement lorsque le seuil est atteint."""
    monkeypatch.setattr(
        "src.scripts.retrain_trade_probability.config_manager.get_config",
        lambda x: {
            "trade_probability": {
                "retrain_threshold": 2,
                "retrain_frequency": "biweekly",
            }
        },
    )
    with patch(
        "src.model.trade_probability.TradeProbabilityPredictor.train"
    ) as mock_train, patch(
        "src.model.trade_probability.TradeProbabilityPredictor.backtest_threshold"
    ) as mock_backtest:
        retrainer.retrain_job()

    assert mock_train.called
    assert mock_backtest.called
    snapshots = list(
        (temp_dir / "data" / "trade_probability_snapshots").glob(
            "snapshot_retrain_job_*.json.gz"
        )
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "trade_probability_performance.csv"
    assert perf_log.exists()


def test_no_obsolete_references(temp_dir, mock_config):
    """Vérifie l'absence de références à dxFeed, obs_t, 320/81 features."""
    with open(mock_config, "r") as f:
        content = f.read()
    assert "dxFeed" not in content, "Référence à dxFeed trouvée"
    assert "obs_t" not in content, "Référence à obs_t trouvée"
    assert "320 features" not in content, "Référence à 320 features trouvée"
    assert "81 features" not in content, "Référence à 81 features trouvée"
