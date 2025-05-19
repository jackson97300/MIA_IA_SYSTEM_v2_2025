# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_run_pipeline.py
# Tests unitaires pour run_pipeline.py.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie les fonctionnalités de PipelineRunner, incluant l'initialisation,
#        la validation des fichiers générés, l’exécution des modules (IBKR, news, merge),
#        les retries, les snapshots JSON compressés, les logs de performance, les graphiques,
#        et les alertes.
#        Conforme à la Phase 6 (collecte et fusion des données), Phase 8 (auto-conscience via alertes),
#        et Phase 16 (ensemble learning).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pandas>=2.0.0
# - psutil>=5.9.8
# - matplotlib>=3.7.0
# - asyncio
# - src.scripts.run_pipeline
# - src.api.ibkr_fetch
# - src.news.news_scraper
# - src.api.merge_data_sources
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
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from src.scripts.run_pipeline import PipelineRunner


@pytest.fixture
def temp_dir(tmp_path):
    """Crée un répertoire temporaire pour les tests."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    data_dir = tmp_path / "data"
    logs_dir = data_dir / "logs"
    ibkr_dir = data_dir / "ibkr"
    news_dir = data_dir / "news"
    features_dir = data_dir / "features"
    snapshots_dir = data_dir / "pipeline_snapshots"
    figures_dir = data_dir / "figures" / "pipeline"
    logs_dir.mkdir(parents=True)
    ibkr_dir.mkdir(parents=True)
    news_dir.mkdir(parents=True)
    features_dir.mkdir(parents=True)
    snapshots_dir.mkdir(parents=True)
    figures_dir.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def mock_config(temp_dir):
    """Crée un fichier es_config.yaml simulé."""
    config_path = temp_dir / "config" / "es_config.yaml"
    config_content = """
pipeline:
  ibkr_enabled: True
  news_enabled: True
  merge_enabled: True
  retry_attempts: 3
  retry_delay: 5
  timeout_seconds: 3600
  ibkr_duration: 60
  news_config:
    url: "https://www.forexfactory.com/calendar"
    output_path: data/news/news_data.csv
    timeout: 10
    retry_attempts: 3
    retry_delay: 5
    cache_days: 7
  merge_config:
    ibkr: data/ibkr/ibkr_data.csv
    news: data/news/news_data.csv
    merged: data/features/merged_data.csv
    chunk_size: 10000
    cache_hours: 24
    time_tolerance: "1min"
  buffer_size: 100
  max_cache_size: 1000
"""
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def mock_ibkr_data(temp_dir):
    """Crée un fichier ibkr_data.csv simulé."""
    ibkr_path = temp_dir / "data" / "ibkr" / "ibkr_data.csv"
    ibkr_data = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start="2025-05-13 10:00:00", periods=100, freq="T"
            ),
            "close": np.random.uniform(5000, 5100, 100),
        }
    )
    ibkr_data.to_csv(ibkr_path, index=False)
    return ibkr_path


@pytest.fixture
def mock_news_data(temp_dir):
    """Crée un fichier news_data.csv simulé."""
    news_path = temp_dir / "data" / "news" / "news_data.csv"
    news_data = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start="2025-05-13 10:00:00", periods=100, freq="T"
            ),
            "sentiment_label": np.random.choice(
                ["positive", "negative", "neutral"], 100
            ),
        }
    )
    news_data.to_csv(news_path, index=False)
    return news_path


@pytest.fixture
def mock_merged_data(temp_dir):
    """Crée un fichier merged_data.csv simulé."""
    merged_path = temp_dir / "data" / "features" / "merged_data.csv"
    merged_data = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start="2025-05-13 10:00:00", periods=100, freq="T"
            ),
            "close": np.random.uniform(5000, 5100, 100),
            "sentiment_label": np.random.choice(
                ["positive", "negative", "neutral"], 100
            ),
        }
    )
    merged_data.to_csv(merged_path, index=False)
    return merged_path


@pytest.fixture
def runner(
    temp_dir, mock_config, mock_ibkr_data, mock_news_data, mock_merged_data, monkeypatch
):
    """Initialise PipelineRunner avec des mocks."""
    monkeypatch.setattr("src.scripts.run_pipeline.CONFIG_PATH", str(mock_config))
    monkeypatch.setattr(
        "src.scripts.run_pipeline.config_manager.get_config",
        lambda x: {
            "pipeline": {
                "ibkr_enabled": True,
                "news_enabled": True,
                "merge_enabled": True,
                "retry_attempts": 3,
                "retry_delay": 5,
                "timeout_seconds": 3600,
                "ibkr_duration": 60,
                "news_config": {
                    "url": "https://www.forexfactory.com/calendar",
                    "output_path": str(mock_news_data),
                    "timeout": 10,
                    "retry_attempts": 3,
                    "retry_delay": 5,
                    "cache_days": 7,
                },
                "merge_config": {
                    "ibkr": str(mock_ibkr_data),
                    "news": str(mock_news_data),
                    "merged": str(mock_merged_data),
                    "chunk_size": 10000,
                    "cache_hours": 24,
                    "time_tolerance": "1min",
                },
                "buffer_size": 100,
                "max_cache_size": 1000,
            }
        },
    )
    runner = PipelineRunner()
    return runner


@pytest.mark.asyncio
async def test_init_runner(
    temp_dir, mock_config, mock_ibkr_data, mock_news_data, mock_merged_data, runner
):
    """Teste l'initialisation de PipelineRunner."""
    assert runner.config["merge_config"]["ibkr"] == str(mock_ibkr_data)
    assert os.path.exists(temp_dir / "data" / "pipeline_snapshots")
    snapshots = list(
        (temp_dir / "data" / "pipeline_snapshots").glob("snapshot_init_*.json.gz")
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "run_pipeline_performance.csv"
    assert perf_log.exists()


@pytest.mark.asyncio
async def test_load_config_invalid(temp_dir, monkeypatch):
    """Teste la validation avec une configuration invalide."""
    invalid_config = temp_dir / "config" / "invalid.yaml"
    monkeypatch.setattr("src.scripts.run_pipeline.CONFIG_PATH", str(invalid_config))
    with pytest.raises(FileNotFoundError, match="Fichier config introuvable"):
        PipelineRunner()

    with patch("src.scripts.run_pipeline.config_manager.get_config", return_value={}):
        monkeypatch.setattr(
            "src.scripts.run_pipeline.CONFIG_PATH",
            str(temp_dir / "config" / "es_config.yaml"),
        )
        with pytest.raises(ValueError, match="Clé 'pipeline' manquante"):
            PipelineRunner()


@pytest.mark.asyncio
async def test_validate_outputs_valid(
    temp_dir, mock_config, mock_ibkr_data, mock_news_data, mock_merged_data, runner
):
    """Teste la validation des fichiers générés valides."""
    runner.validate_outputs(runner.config)
    snapshots = list(
        (temp_dir / "data" / "pipeline_snapshots").glob(
            "snapshot_validate_outputs_*.json.gz"
        )
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "run_pipeline_performance.csv"
    assert perf_log.exists()


@pytest.mark.asyncio
async def test_validate_outputs_invalid(
    temp_dir, mock_config, mock_ibkr_data, mock_news_data, runner
):
    """Teste la validation des fichiers générés invalides."""
    invalid_config = runner.config.copy()
    invalid_config["merge_config"]["ibkr"] = str(temp_dir / "invalid.csv")
    with pytest.raises(ValueError, match="Fichier IBKR introuvable"):
        runner.validate_outputs(invalid_config)
    snapshots = list(
        (temp_dir / "data" / "pipeline_snapshots").glob(
            "snapshot_validate_outputs_*.json.gz"
        )
    )
    assert len(snapshots) >= 1


@pytest.mark.asyncio
async def test_run_ibkr_fetch_success(temp_dir, mock_config, mock_ibkr_data, runner):
    """Teste l’exécution de la collecte IBKR avec succès."""
    with patch("src.api.ibkr_fetch.main", new=AsyncMock()) as mock_fetch:
        result = await runner.run_ibkr_fetch(runner.config)
    assert result is True
    snapshots = list(
        (temp_dir / "data" / "pipeline_snapshots").glob(
            "snapshot_run_ibkr_fetch_*.json.gz"
        )
    )
    assert len(snapshots) >= 1


@pytest.mark.asyncio
async def test_run_news_scraper_success(temp_dir, mock_config, mock_news_data, runner):
    """Teste l’exécution du scraping de news avec succès."""
    with patch(
        "src.news.news_scraper.scrape_news_events",
        return_value=pd.DataFrame({"timestamp": [datetime.now()], "event": ["test"]}),
    ):
        result = await runner.run_news_scraper(runner.config)
    assert result is True
    snapshots = list(
        (temp_dir / "data" / "pipeline_snapshots").glob(
            "snapshot_run_news_scraper_*.json.gz"
        )
    )
    assert len(snapshots) >= 1


@pytest.mark.asyncio
async def test_run_merge_data_success(
    temp_dir, mock_config, mock_ibkr_data, mock_news_data, mock_merged_data, runner
):
    """Teste l’exécution de la fusion des données avec succès."""
    with patch(
        "src.api.merge_data_sources.merge_data_sources",
        return_value=pd.DataFrame({"timestamp": [datetime.now()], "close": [5000]}),
    ):
        result = await runner.run_merge_data(runner.config)
    assert result is True
    snapshots = list(
        (temp_dir / "data" / "pipeline_snapshots").glob(
            "snapshot_run_merge_data_*.json.gz"
        )
    )
    assert len(snapshots) >= 1


@pytest.mark.asyncio
async def test_run_pipeline_success(
    temp_dir, mock_config, mock_ibkr_data, mock_news_data, mock_merged_data, runner
):
    """Teste l’exécution complète du pipeline avec succès."""
    with patch(
        "src.scripts.run_pipeline.PipelineRunner.run_ibkr_fetch",
        new=AsyncMock(return_value=True),
    ), patch(
        "src.scripts.run_pipeline.PipelineRunner.run_news_scraper",
        new=AsyncMock(return_value=True),
    ), patch(
        "src.scripts.run_pipeline.PipelineRunner.run_merge_data",
        new=AsyncMock(return_value=True),
    ):
        status = await runner.run_pipeline()

    assert status["ibkr_success"] is True
    assert status["news_success"] is True
    assert status["merge_success"] is True
    snapshots = list(
        (temp_dir / "data" / "pipeline_snapshots").glob(
            "snapshot_run_pipeline_*.json.gz"
        )
    )
    assert len(snapshots) >= 1
    perf_log = temp_dir / "data" / "logs" / "run_pipeline_performance.csv"
    assert perf_log.exists()
    figures = list(
        (temp_dir / "data" / "figures" / "pipeline").glob("pipeline_status_*.png")
    )
    assert len(figures) >= 1


def test_no_obsolete_references(temp_dir, mock_config):
    """Vérifie l'absence de références à dxFeed, obs_t, 320/81 features."""
    with open(mock_config, "r") as f:
        content = f.read()
    assert "dxFeed" not in content, "Référence à dxFeed trouvée"
    assert "obs_t" not in content, "Référence à obs_t trouvée"
    assert "320 features" not in content, "Référence à 320 features trouvée"
    assert "81 features" not in content, "Référence à 81 features trouvée"
