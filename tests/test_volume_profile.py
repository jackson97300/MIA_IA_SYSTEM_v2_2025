# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_volume_profile.py
# Tests unitaires pour la classe VolumeProfileExtractor.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie les fonctionnalités de VolumeProfileExtractor, incluant le calcul des métriques
#        de profil de volume (POC, VAH, VAL), la normalisation, la clusterisation (méthode 7),
#        le stockage dans market_memory.db, la gestion du cache, la validation SHAP, et l'intégration
#        avec IQFeed (Phase 5).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, scikit-learn>=1.5.0,<2.0.0, psutil>=5.9.0,<6.0.0, sqlite3
# - src/features/extractors/volume_profile.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
# - src/model/utils/db_maintenance.py
#
# Notes :
# - Utilise exclusivement des données simulées compatibles avec IQFeed.
# - Suppression de toute référence à dxFeed, obs_t, 320 features, et 81 features.
# - Couvre la Phase 5 (volume_profile.py).

import hashlib
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.features.extractors.volume_profile import VolumeProfileExtractor

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data", "features", "cache", "volume_profile")
FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR, "data", "features", "feature_importance.csv"
)
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "features", "volume_profile_snapshots")
DB_PATH = os.path.join(BASE_DIR, "data", "market_memory.db")


@pytest.fixture
def volume_profile_extractor(tmp_path):
    """Crée une instance de VolumeProfileExtractor avec un répertoire temporaire."""
    config_path = tmp_path / "es_config.yaml"
    config_path.write_text(
        """
        volume_profile_extractor:
          buffer_size: 10
          max_cache_size: 100
          cache_hours: 24
          n_clusters: 5
        """
    )
    return VolumeProfileExtractor(config_path=str(config_path))


@pytest.fixture
def test_data():
    """Crée des données de test pour VolumeProfileExtractor."""
    timestamps = pd.date_range("2025-04-14 09:00", periods=100, freq="1min")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": np.random.normal(5100, 10, 100),
            "volume": np.random.randint(100, 1000, 100),
        }
    )


def test_extract_volume_profile(volume_profile_extractor, test_data):
    """Teste extract_volume_profile."""
    result = volume_profile_extractor.extract_volume_profile(test_data)
    assert not result.empty, "Le DataFrame résultat ne doit pas être vide"
    expected_columns = [
        "poc",
        "vah",
        "val",
        "poc_normalized",
        "vah_normalized",
        "val_normalized",
    ]
    assert all(
        col in result.columns for col in expected_columns
    ), "Colonnes attendues manquantes"
    assert not result["poc"].isna().all(), "POC ne doit pas contenir uniquement des NaN"
    assert not result["vah"].isna().all(), "VAH ne doit pas contenir uniquement des NaN"
    assert not result["val"].isna().all(), "VAL ne doit pas contenir uniquement des NaN"
    assert (
        result["poc_normalized"].between(0, 1).all()
        or result["poc_normalized"].eq(0).all()
    ), "Normalisation POC hors plage [0,1]"


def test_extract_volume_profile_empty_data(volume_profile_extractor):
    """Teste extract_volume_profile avec un DataFrame vide."""
    empty_data = pd.DataFrame()
    result = volume_profile_extractor.extract_volume_profile(empty_data)
    expected_columns = [
        "poc",
        "vah",
        "val",
        "poc_normalized",
        "vah_normalized",
        "val_normalized",
    ]
    assert result.empty or all(
        result[col].eq(0).all() for col in expected_columns
    ), "Les métriques doivent être à 0 pour un DataFrame vide"


def test_extract_volume_profile_missing_columns(volume_profile_extractor, test_data):
    """Teste extract_volume_profile avec des colonnes manquantes."""
    incomplete_data = test_data.drop(columns=["volume"])
    result = volume_profile_extractor.extract_volume_profile(incomplete_data)
    assert not result.empty, "Le DataFrame résultat ne doit pas être vide"
    assert result["poc"].eq(0).all(), "POC doit être 0 si volume est manquant"
    assert result["vah"].eq(0).all(), "VAH doit être 0 si volume est manquant"
    assert result["val"].eq(0).all(), "VAL doit être 0 si volume est manquant"


def test_cache_profiles(volume_profile_extractor, test_data, tmp_path):
    """Teste cache_profiles."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    global CACHE_DIR
    CACHE_DIR = str(cache_dir)  # Redéfinir temporairement CACHE_DIR pour les tests
    result = volume_profile_extractor.extract_volume_profile(test_data)
    cache_key = hashlib.sha256(result.to_json().encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.csv")
    assert os.path.exists(
        cache_path
    ), "Le fichier cache doit exister après extract_volume_profile"
    cached_data = pd.read_csv(cache_path)
    assert len(cached_data) == len(
        result
    ), "Les données mises en cache doivent correspondre aux données calculées"
    CACHE_DIR = os.path.join(
        BASE_DIR, "data", "features", "cache", "volume_profile"
    )  # Restaurer CACHE_DIR


def test_validate_shap_features(volume_profile_extractor, tmp_path):
    """Teste validate_shap_features avec un fichier SHAP vide."""
    feature_importance_path = tmp_path / "feature_importance.csv"
    pd.DataFrame({"feature": ["dummy_feature"]}).to_csv(feature_importance_path)
    global FEATURE_IMPORTANCE_PATH
    FEATURE_IMPORTANCE_PATH = str(feature_importance_path)  # Redéfinir temporairement
    features = ["poc", "vah", "val"]
    assert not volume_profile_extractor.validate_shap_features(
        features
    ), "La validation doit échouer avec un fichier SHAP insuffisant"
    FEATURE_IMPORTANCE_PATH = os.path.join(
        BASE_DIR, "data", "features", "feature_importance.csv"
    )  # Restaurer


def test_normalize_profiles(volume_profile_extractor, test_data):
    """Teste normalize_profiles."""
    profiles = ["poc", "vah", "val"]
    test_data["poc"] = np.random.normal(5100, 10, len(test_data))
    test_data["vah"] = np.random.normal(5110, 10, len(test_data))
    test_data["val"] = np.random.normal(5090, 10, len(test_data))
    result = volume_profile_extractor.normalize_profiles(test_data, profiles)
    assert "poc_normalized" in result.columns, "Colonne normalisée manquante"
    assert "vah_normalized" in result.columns, "Colonne normalisée manquante"
    assert "val_normalized" in result.columns, "Colonne normalisée manquante"
    assert (
        result["poc_normalized"].between(0, 1).all()
        or result["poc_normalized"].eq(0).all()
    ), "Normalisation POC hors plage [0,1]"


def test_cluster_profiles(volume_profile_extractor, test_data):
    """Teste cluster_profiles."""
    result = volume_profile_extractor.extract_volume_profile(test_data)
    clusters_df = volume_profile_extractor.cluster_profiles(result)
    assert not clusters_df.empty, "Le DataFrame des clusters ne doit pas être vide"
    assert set(clusters_df.columns) == {
        "timestamp",
        "cluster_id",
        "centroid",
        "cluster_size",
        "event_type",
    }, "Colonnes de clusters incorrectes"
    assert (
        clusters_df["event_type"].eq("VOLUME_PROFILE").all()
    ), "event_type doit être VOLUME_PROFILE"
    assert clusters_df["cluster_size"].sum() == len(
        result
    ), "La somme des tailles de clusters doit correspondre au nombre de profils"


def test_store_clusters(volume_profile_extractor, test_data, tmp_path):
    """Teste store_clusters."""
    db_path = tmp_path / "market_memory.db"
    global DB_PATH
    DB_PATH = str(db_path)  # Redéfinir temporairement DB_PATH pour les tests
    result = volume_profile_extractor.extract_volume_profile(test_data)
    clusters_df = volume_profile_extractor.cluster_profiles(result)
    volume_profile_extractor.store_clusters(clusters_df)
    conn = sqlite3.connect(DB_PATH)
    stored_clusters = pd.read_sql("SELECT * FROM clusters", conn)
    conn.close()
    assert len(stored_clusters) == len(
        clusters_df
    ), "Les clusters stockés doivent correspondre aux clusters générés"
    DB_PATH = os.path.join(BASE_DIR, "data", "market_memory.db")  # Restaurer DB_PATH


def test_clean_cache(volume_profile_extractor, tmp_path):
    """Teste _clean_cache."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    global CACHE_DIR
    CACHE_DIR = str(cache_dir)  # Redéfinir temporairement
    cache_path = os.path.join(CACHE_DIR, "test_cache.csv")
    pd.DataFrame({"dummy": [1, 2, 3]}).to_csv(cache_path)
    volume_profile_extractor._clean_cache(max_size_mb=0.0001)  # Forcer suppression
    assert not os.path.exists(cache_path), "Le fichier cache doit être supprimé"
    CACHE_DIR = os.path.join(
        BASE_DIR, "data", "features", "cache", "volume_profile"
    )  # Restaurer


def test_handle_sigint(volume_profile_extractor, tmp_path):
    """Teste handle_sigint."""
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    global SNAPSHOT_DIR
    SNAPSHOT_DIR = str(snapshot_dir)  # Redéfinir temporairement
    with pytest.raises(SystemExit, match="Terminated by SIGINT"):
        volume_profile_extractor.handle_sigint(signal.SIGINT, None)
    snapshot_files = list(snapshot_dir.glob("snapshot_sigint_*.json.gz"))
    assert len(snapshot_files) > 0, "Un snapshot SIGINT doit être créé"
    SNAPSHOT_DIR = os.path.join(
        BASE_DIR, "data", "features", "volume_profile_snapshots"
    )  # Restaurer
