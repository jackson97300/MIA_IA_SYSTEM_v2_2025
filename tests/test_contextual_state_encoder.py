# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_contextual_state_encoder.py
# Tests unitaires pour src/features/contextual_state_encoder.py
# Version : 2.1.3
# Date : 2025-05-13
# Rôle : Valide l'encodage des composantes latentes (UMAP, NLP) avec validation SHAP (Phase 17),
#        confidence_drop_rate (Phase 8), et snapshots avec alertes Telegram.

import gzip
import json
import os
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.features.contextual_state_encoder import ContextualStateEncoder


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, snapshots, cache, latent, news, et données."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    snapshots_dir = data_dir / "contextual_snapshots"
    snapshots_dir.mkdir()
    latent_dir = data_dir / "latent"
    latent_dir.mkdir()
    news_dir = data_dir / "news"
    news_dir.mkdir()
    features_dir = data_dir / "features"
    features_dir.mkdir()
    cache_dir = features_dir / "cache" / "contextual"
    cache_dir.mkdir(parents=True)

    # Créer es_config.yaml
    config_path = config_dir / "es_config.yaml"
    config_content = {
        "contextual_state_encoder": {
            "n_topics": 3,
            "window_size": 10,
            "min_news": 10,
            "buffer_size": 100,
            "max_cache_size": 1000,
            "cache_hours": 24,
        }
    }
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)

    return {
        "base_dir": str(base_dir),
        "config_path": str(config_path),
        "logs_dir": str(logs_dir),
        "snapshots_dir": str(snapshots_dir),
        "latent_dir": str(latent_dir),
        "news_dir": str(news_dir),
        "features_dir": str(features_dir),
        "cache_dir": str(cache_dir),
        "perf_log_path": str(logs_dir / "contextual_encoder_performance.csv"),
        "feature_importance_path": str(features_dir / "feature_importance.csv"),
        "latent_vectors_path": str(latent_dir / "latent_vectors.csv"),
        "news_topics_path": str(news_dir / "news_topics.csv"),
        "news_data_path": str(news_dir / "news_data.csv"),
    }


@pytest.fixture
def mock_data():
    """Crée des données factices pour tester."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=100, freq="1min"),
            "atr_14": np.random.uniform(0.5, 2.0, 100),
            "volatility_trend": np.random.uniform(-0.1, 0.1, 100),
            "close": np.random.normal(5100, 10, 100),
        }
    )


@pytest.fixture
def mock_news_data(tmp_dirs):
    """Crée des données factices pour news_data.csv."""
    news_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=100, freq="1min"),
            "news_content": ["Market up due to tech rally"] * 20
            + ["Inflation fears rise"] * 20
            + ["Earnings season starts"] * 20
            + ["Fed rate decision looms"] * 20
            + ["Geopolitical tensions increase"] * 20,
        }
    )
    news_data.to_csv(tmp_dirs["news_data_path"], index=False, encoding="utf-8")
    return news_data


@pytest.fixture
def mock_feature_importance(tmp_dirs):
    """Crée un fichier feature_importance.csv factice."""
    shap_data = pd.DataFrame(
        {
            "feature": [
                "latent_vol_regime_vec_0",
                "latent_vol_regime_vec_1",
                "vol_regime_transition_prob",
                "news_topic_stability",
                "topic_vector_news_0",
                "topic_vector_news_1",
                "topic_vector_news_2",
            ]
            * 22,
            "importance": [0.1] * 154,
            "regime": ["range"] * 154,
        }
    )
    shap_data.to_csv(tmp_dirs["feature_importance_path"], index=False, encoding="utf-8")
    return shap_data


@pytest.fixture
def mock_db(tmp_dirs):
    """Simule market_memory.db."""
    from src.utils.database_manager import DatabaseManager

    db_path = os.path.join(tmp_dirs["base_dir"], "data", "market_memory.db")
    db_manager = DatabaseManager(db_path)
    with db_manager.connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS clusters (
                cluster_id INTEGER,
                event_type TEXT,
                features TEXT,
                timestamp TEXT
            )
        """
        )
        conn.commit()
    return db_manager


def test_contextual_state_encoder_init(tmp_dirs, mock_feature_importance, mock_db):
    """Teste l’initialisation de ContextualStateEncoder."""
    encoder = ContextualStateEncoder(config_path=tmp_dirs["config_path"])
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    assert os.path.exists(tmp_dirs["snapshots_dir"]), "Dossier de snapshots non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert all(
        col in df.columns
        for col in ["timestamp", "operation", "latency", "cpu_usage_percent"]
    ), "Colonnes de performance manquantes"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_init" in f and f.endswith(".json") for f in snapshot_files
    ), "Snapshot non créé"


def test_encode_vol_regime_valid(tmp_dirs, mock_data, mock_feature_importance, mock_db):
    """Teste l’encodage des régimes de volatilité avec des données valides."""
    encoder = ContextualStateEncoder(config_path=tmp_dirs["config_path"])
    latent_df = encoder.encode_vol_regime(mock_data, tmp_dirs["latent_vectors_path"])
    assert len(latent_df) == len(mock_data), "Longueur du DataFrame latent incorrecte"
    assert {"timestamp", "latent_vol_regime_vec_0", "latent_vol_regime_vec_1"}.issubset(
        latent_df.columns
    ), "Colonnes manquantes dans latent_df"
    assert os.path.exists(
        tmp_dirs["latent_vectors_path"]
    ), "Fichier latent_vectors.csv non créé"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_encode_vol_regime" in f and f.endswith(".json")
        for f in snapshot_files
    ), "Snapshot non créé"
    with open(os.path.join(tmp_dirs["snapshots_dir"], snapshot_files[-1]), "r") as f:
        snapshot = json.load(f)
    assert (
        "confidence_drop_rate" in snapshot["data"]
    ), "confidence_drop_rate absent du snapshot"


def test_encode_news_topics_valid(
    tmp_dirs, mock_data, mock_news_data, mock_feature_importance, mock_db
):
    """Teste l’encodage des sujets des news avec des données valides."""
    encoder = ContextualStateEncoder(config_path=tmp_dirs["config_path"])
    topics_df = encoder.encode_news_topics(
        mock_data, tmp_dirs["news_data_path"], tmp_dirs["news_topics_path"], n_topics=3
    )
    assert len(topics_df) <= len(
        mock_news_data
    ), "Longueur du DataFrame topics incorrecte"
    assert {
        "timestamp",
        "topic_vector_news_0",
        "topic_vector_news_1",
        "topic_vector_news_2",
    }.issubset(topics_df.columns), "Colonnes manquantes dans topics_df"
    assert os.path.exists(
        tmp_dirs["news_topics_path"]
    ), "Fichier news_topics.csv non créé"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_encode_news_topics" in f and f.endswith(".json")
        for f in snapshot_files
    ), "Snapshot non créé"


def test_calculate_contextual_encodings_valid(
    tmp_dirs, mock_data, mock_news_data, mock_feature_importance, mock_db
):
    """Teste le calcul des encodages contextuels complets."""
    encoder = ContextualStateEncoder(config_path=tmp_dirs["config_path"])
    result = encoder.calculate_contextual_encodings(
        mock_data,
        tmp_dirs["news_data_path"],
        tmp_dirs["latent_vectors_path"],
        tmp_dirs["news_topics_path"],
        n_topics=3,
    )
    assert len(result) == len(mock_data), "Longueur du DataFrame résultat incorrecte"
    expected_cols = [
        "timestamp",
        "latent_vol_regime_vec_0",
        "latent_vol_regime_vec_1",
        "vol_regime_transition_prob",
        "news_topic_stability",
        "topic_vector_news_0",
    ]
    assert all(
        col in result.columns for col in expected_cols
    ), "Colonnes manquantes dans result"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_contextual_encodings" in f and f.endswith(".json")
        for f in snapshot_files
    ), "Snapshot non créé"


def test_calculate_contextual_encodings_invalid_data(
    tmp_dirs, mock_feature_importance, mock_db
):
    """Teste le calcul des encodages contextuels avec des données invalides."""
    encoder = ContextualStateEncoder(config_path=tmp_dirs["config_path"])
    invalid_data = pd.DataFrame({"timestamp": [datetime.now()]})  # Données vides
    result = encoder.calculate_contextual_encodings(invalid_data)
    expected_cols = [
        "timestamp",
        "latent_vol_regime_vec_0",
        "latent_vol_regime_vec_1",
        "vol_regime_transition_prob",
        "news_topic_stability",
    ]
    assert all(
        col in result.columns for col in expected_cols
    ), "Colonnes manquantes dans result"
    assert (
        result[expected_cols[1:]].eq(0.0).all().all()
    ), "Features non nulles pour données invalides"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "DataFrame vide" in str(e) for e in df["error"].dropna()
    ), "Erreur DataFrame vide non loguée"


def test_save_to_market_memory(tmp_dirs, mock_data, mock_db):
    """Teste la sauvegarde dans market_memory.db."""
    encoder = ContextualStateEncoder(config_path=tmp_dirs["config_path"])
    test_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=5, freq="1min"),
            "latent_vol_regime_vec_0": np.random.uniform(0, 1, 5),
            "latent_vol_regime_vec_1": np.random.uniform(0, 1, 5),
        }
    )
    encoder.save_to_market_memory(
        test_data, "test_event", ["latent_vol_regime_vec_0", "latent_vol_regime_vec_1"]
    )
    with encoder.db_manager.connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM clusters WHERE event_type = 'test_event'")
        rows = cursor.fetchall()
    assert len(rows) == 5, "Nombre incorrect de lignes insérées dans market_memory.db"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_market_memory" in f for f in snapshot_files
    ), "Snapshot non créé"


def test_validate_shap_features(tmp_dirs, mock_feature_importance):
    """Teste la validation des top 150 SHAP features."""
    encoder = ContextualStateEncoder(config_path=tmp_dirs["config_path"])
    features = [
        "latent_vol_regime_vec_0",
        "latent_vol_regime_vec_1",
        "topic_vector_news_0",
    ]
    result = encoder.validate_shap_features(features)
    assert result, "Validation SHAP échouée pour features valides"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_validate_shap_features" in f for f in snapshot_files
    ), "Snapshot non créé"


def test_save_snapshot_compressed(tmp_dirs):
    """Teste la sauvegarde d’un snapshot compressé avec gzip."""
    encoder = ContextualStateEncoder(config_path=tmp_dirs["config_path"])
    snapshot_data = {"test": "compressed_snapshot"}
    encoder.save_snapshot("test_compressed", snapshot_data, compress=True)
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


def test_critical_alerts(tmp_dirs, mock_feature_importance, mock_db):
    """Teste les alertes Telegram pour les erreurs critiques."""
    with patch(
        "src.features.contextual_state_encoder.send_telegram_alert"
    ) as mock_telegram:
        encoder = ContextualStateEncoder(config_path=tmp_dirs["config_path"])
        invalid_data = pd.DataFrame(
            {"timestamp": [datetime.now()]}
        )  # Données invalides
        encoder.calculate_contextual_encodings(invalid_data)
        mock_telegram.assert_called_with(pytest.any(str))
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "DataFrame vide" in str(e) for e in df["error"].dropna()
        ), "Erreur critique non loguée"
