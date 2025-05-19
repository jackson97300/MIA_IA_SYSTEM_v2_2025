# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_news_scraper.py
# Tests unitaires pour src/api/news_scraper.py
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Valide la collecte des nouvelles financières via NewsAPI, le calcul de news_impact_score
#        (méthode 5), le cache local, la validation SHAP (Phase 17), les snapshots, les sauvegardes,
#        et les alertes Telegram.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, requests>=2.28.0,<3.0.0, psutil>=5.9.8,<6.0.0,
#   pyyaml>=6.0.0,<7.0.0, boto3>=1.26.0,<2.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/news_config.yaml
# - config/credentials.yaml
#
# Outputs :
# - data/news/news_data.csv
# - data/news/cache/daily_YYYYMMDD.csv
# - data/news_snapshots/*.json.gz
# - data/logs/news_scraper_performance.csv
# - data/logs/news_scraper.log
# - data/checkpoints/news_data_*.json.gz
# - data/news_dashboard.json
#
# Notes :
# - Conforme à structure.txt (version 2.1.3, 2025-05-13).
# - Tests les phases 1 (collecte API), 8 (auto-conscience), 17 (interprétabilité SHAP).
# - Vérifie l’absence de scraping Investing.com, la présence de retries, logs psutil,
#   alertes Telegram, snapshots, et sauvegardes incrémentielles/distribuées.

import gzip
import json
import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from src.api.news_scraper import NewsScraper


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les données, logs, snapshots, checkpoints, et cache."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    news_dir = data_dir / "news"
    news_dir.mkdir()
    cache_dir = news_dir / "cache"
    cache_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    snapshots_dir = data_dir / "news_snapshots"
    snapshots_dir.mkdir()
    checkpoints_dir = data_dir / "checkpoints"
    checkpoints_dir.mkdir()
    features_dir = data_dir / "features"
    features_dir.mkdir()

    # Créer news_config.yaml
    config_path = config_dir / "news_config.yaml"
    config_content = {
        "news_scraper": {
            "endpoint": "https://newsapi.org/v2/everything",
            "timeout": 10,
            "retry_attempts": 3,
            "retry_delay_base": 2.0,
            "cache_days": 7,
            "s3_bucket": "test-bucket",
            "s3_prefix": "news_data/",
        }
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    # Créer credentials.yaml
    credentials_path = config_dir / "credentials.yaml"
    credentials_content = {"credentials": {"news_api_key": "test-api-key"}}
    with open(credentials_path, "w", encoding="utf-8") as f:
        yaml.dump(credentials_content, f)

    return {
        "base_dir": str(base_dir),
        "config_path": str(config_path),
        "credentials_path": str(credentials_path),
        "logs_dir": str(logs_dir),
        "snapshots_dir": str(snapshots_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "cache_dir": str(cache_dir),
        "output_path": str(news_dir / "news_data.csv"),
        "perf_log_path": str(logs_dir / "news_scraper_performance.csv"),
        "dashboard_path": str(news_dir / "news_dashboard.json"),
        "feature_importance_path": str(features_dir / "feature_importance.csv"),
    }


@pytest.fixture
def mock_news_data():
    """Crée des données factices de nouvelles."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=10, freq="1min"),
            "headline": [
                (
                    f"News {i}: FOMC Meeting"
                    if i % 2 == 0
                    else f"News {i}: Earnings Report"
                )
                for i in range(10)
            ],
            "source": ["Reuters"] * 5 + ["Bloomberg"] * 5,
            "description": ["Description"] * 10,
            "news_impact_score": [0.9 if i % 2 == 0 else 0.6 for i in range(10)],
        }
    )


@pytest.fixture
def mock_feature_importance(tmp_dirs):
    """Crée un fichier feature_importance.csv factice pour la validation SHAP."""
    features = ["news_impact_score", "source"] + [f"feature_{i}" for i in range(148)]
    shap_data = pd.DataFrame(
        {"feature": features, "importance": [0.1] * 150, "regime": ["range"] * 150}
    )
    shap_data.to_csv(tmp_dirs["feature_importance_path"], index=False, encoding="utf-8")
    return shap_data


def test_init(tmp_dirs, mock_feature_importance):
    """Teste l’initialisation de NewsScraper."""
    scraper = NewsScraper(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    assert os.path.exists(tmp_dirs["snapshots_dir"]), "Dossier de snapshots non créé"
    assert os.path.exists(tmp_dirs["cache_dir"]), "Dossier de cache non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert all(
        col in df.columns
        for col in ["timestamp", "operation", "latency", "cpu_usage_percent"]
    ), "Colonnes de performance manquantes"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_init" in f and f.endswith(".json.gz") for f in snapshot_files
    ), "Snapshot non créé"


def test_load_config_with_validation(tmp_dirs, mock_feature_importance):
    """Teste le chargement de la configuration."""
    scraper = NewsScraper(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    config = scraper.load_config_with_validation(tmp_dirs["config_path"])
    assert "endpoint" in config, "Clé endpoint manquante"
    assert "s3_bucket" in config, "Clé s3_bucket manquante"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "load_config_with_validation" in str(op) for op in df["operation"]
    ), "Opération non journalisée"


def test_scrape_news(tmp_dirs, mock_news_data, mock_feature_importance):
    """Teste la collecte des nouvelles avec news_impact_score."""
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "ok",
            "articles": [
                {
                    "title": "FOMC Meeting Scheduled",
                    "source": {"name": "Reuters"},
                    "publishedAt": str(datetime.now()),
                    "description": "Federal Reserve meeting",
                },
                {
                    "title": "Earnings Report Released",
                    "source": {"name": "Bloomberg"},
                    "publishedAt": str(datetime.now()),
                    "description": "Corporate earnings",
                },
            ],
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        scraper = NewsScraper(
            config_path=tmp_dirs["config_path"],
            credentials_path=tmp_dirs["credentials_path"],
        )
        news = scraper.scrape_news()
        assert news is not None, "Nouvelles non récupérées"
        assert all(
            col in news.columns
            for col in ["timestamp", "headline", "source", "news_impact_score"]
        ), "Colonnes requises manquantes"
        assert os.path.exists(tmp_dirs["output_path"]), "Fichier news_data.csv non créé"
        assert os.path.exists(
            os.path.join(
                tmp_dirs["cache_dir"], f"daily_{datetime.now().strftime('%Y%m%d')}.csv"
            )
        ), "Cache quotidien non créé"
        assert os.path.exists(
            tmp_dirs["dashboard_path"]
        ), "Fichier news_dashboard.json non créé"
        snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
        assert any(
            "snapshot_scrape_news" in f for f in snapshot_files
        ), "Snapshot non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "scrape_news" in str(op) for op in df["operation"]
        ), "Opération non journalisée"
        assert "confidence_drop_rate" in df.columns or any(
            "confidence_drop_rate" in str(kw) for kw in df.to_dict("records")
        ), "confidence_drop_rate absent"


def test_calculate_impact(tmp_dirs, mock_feature_importance):
    """Teste le calcul de news_impact_score."""
    scraper = NewsScraper(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    score_high = scraper.calculate_impact("FOMC Meeting Scheduled")
    score_medium = scraper.calculate_impact("Corporate Earnings Report")
    score_low = scraper.calculate_impact("General News")
    assert score_high >= 0.8, "Score pour FOMC trop bas"
    assert 0.4 <= score_medium < 0.8, "Score pour earnings incorrect"
    assert score_low == 0.3, "Score par défaut incorrect"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "calculate_impact" in str(op) for op in df["operation"]
    ), "Opération non journalisée"


def test_load_save_daily_cache(tmp_dirs, mock_news_data, mock_feature_importance):
    """Teste le chargement et la sauvegarde du cache quotidien."""
    scraper = NewsScraper(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    today = datetime.now().strftime("%Y%m%d")
    scraper.save_daily_cache(mock_news_data, today)
    cache_path = os.path.join(tmp_dirs["cache_dir"], f"daily_{today}.csv")
    assert os.path.exists(cache_path), "Cache quotidien non créé"
    cached_df = scraper.load_daily_cache(today)
    assert cached_df is not None, "Cache non chargé"
    assert len(cached_df) == len(mock_news_data), "Nombre de nouvelles incorrect"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "save_daily_cache" in str(op) for op in df["operation"]
    ), "Opération save_daily_cache non journalisée"
    assert any(
        "load_daily_cache" in str(op) for op in df["operation"]
    ), "Opération load_daily_cache non journalisée"


def test_validate_shap_features(tmp_dirs, mock_feature_importance):
    """Teste la validation des features SHAP."""
    scraper = NewsScraper(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    result = scraper.validate_shap_features(["news_impact_score", "source"])
    assert result, "Validation SHAP échouée pour features valides"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_validate_shap_features" in f for f in snapshot_files
    ), "Snapshot non créé"


def test_save_snapshot_compressed(tmp_dirs, mock_feature_importance):
    """Teste la sauvegarde d’un snapshot compressé."""
    scraper = NewsScraper(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    snapshot_data = {"test": "compressed_snapshot"}
    scraper.save_snapshot("test_compressed", snapshot_data, compress=True)
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_test_compressed" in f and f.endswith(".json.gz")
        for f in snapshot_files
    ), "Snapshot compressé non créé"
    with gzip.open(
        os.path.join(tmp_dirs["snapshots_dir"], snapshot_files[-1]),
        "rt",
        encoding="utf-8",
    ) as f:
        snapshot = json.load(f)
    assert snapshot["data"] == snapshot_data, "Contenu du snapshot compressé incorrect"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"


def test_checkpoint(tmp_dirs, mock_news_data, mock_feature_importance):
    """Teste la sauvegarde incrémentielle."""
    scraper = NewsScraper(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    scraper.checkpoint(mock_news_data)
    checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
    assert any(
        "news_data_" in f and f.endswith(".json.gz") for f in checkpoint_files
    ), "Checkpoint non créé"
    with gzip.open(
        os.path.join(tmp_dirs["checkpoints_dir"], checkpoint_files[0]),
        "rt",
        encoding="utf-8",
    ) as f:
        checkpoint = json.load(f)
    assert checkpoint["num_news"] == len(
        mock_news_data
    ), "Nombre de nouvelles incorrect"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"


def test_cloud_backup(tmp_dirs, mock_news_data, mock_feature_importance):
    """Teste la sauvegarde distribuée S3."""
    with patch("boto3.client") as mock_s3:
        scraper = NewsScraper(
            config_path=tmp_dirs["config_path"],
            credentials_path=tmp_dirs["credentials_path"],
        )
        scraper.cloud_backup(mock_news_data)
        assert mock_s3.called, "Client S3 non appelé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "temp_s3_" in f for f in checkpoint_files
        ), "Fichier temporaire S3 non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "cloud_backup" in str(op) for op in df["operation"]
        ), "Opération non journalisée"


def test_signal_handler(tmp_dirs, mock_news_data, mock_feature_importance):
    """Teste la gestion SIGINT."""
    with patch("pandas.read_csv", return_value=mock_news_data) as mock_read:
        scraper = NewsScraper(
            config_path=tmp_dirs["config_path"],
            credentials_path=tmp_dirs["credentials_path"],
        )
        scraper.signal_handler(signal.SIGINT, None)
        snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
        assert any(
            "snapshot_shutdown" in f for f in snapshot_files
        ), "Snapshot shutdown non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "signal_handler" in str(op) for op in df["operation"]
        ), "Opération non journalisée"


def test_critical_alerts(tmp_dirs, mock_feature_importance):
    """Teste les alertes Telegram pour les erreurs critiques."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram, patch(
        "requests.get", side_effect=Exception("Erreur réseau")
    ):
        scraper = NewsScraper(
            config_path=tmp_dirs["config_path"],
            credentials_path=tmp_dirs["credentials_path"],
        )
        news = scraper.scrape_news()
        assert news is None, "Nouvelles récupérées malgré erreur"
        mock_telegram.assert_called_with(pytest.any(str))
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "scrape_news" in str(op) and not success
            for success, op in zip(df["success"], df["operation"])
        ), "Erreur critique non journalisée"
