# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_smart_scores.py
# Tests unitaires pour la classe SmartScoresCalculator.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie les fonctionnalités de SmartScoresCalculator, incluant le calcul des 3 scores
#        intelligents (breakout_score, mm_score, hft_score), la gestion des erreurs,
#        la pondération par régime, la validation SHAP, et l'intégration avec IQFeed.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.0,<6.0.0, scikit-learn>=1.5.0,<2.0.0, matplotlib>=3.7.0,<4.0.0
# - src/features/extractors/smart_scores.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
#
# Notes :
# - Utilise exclusivement des données simulées compatibles avec IQFeed.
# - Suppression de toute référence à dxFeed, obs_t, 320 features, et 81 features.
# - Conforme aux Phases 1-18, bien que non explicitement lié à une phase spécifique.

import hashlib
import os

import numpy as np
import pandas as pd
import pytest

from src.features.extractors.smart_scores import SmartScoresCalculator

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data", "features", "cache", "smart_scores")
FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR, "data", "features", "feature_importance.csv"
)


@pytest.fixture
def smart_scores_calculator(tmp_path):
    """Crée une instance de SmartScoresCalculator avec un répertoire temporaire."""
    config_path = tmp_path / "es_config.yaml"
    config_path.write_text(
        """
        smart_scores_calculator:
          buffer_size: 10
          max_cache_size: 100
          cache_hours: 24
          weights:
            trend:
              mm_score: 1.5
              hft_score: 1.5
              breakout_score: 1.0
            range:
              breakout_score: 1.5
              mm_score: 1.0
              hft_score: 1.0
            defensive:
              mm_score: 0.5
              hft_score: 0.5
              breakout_score: 0.5
        """
    )
    return SmartScoresCalculator(config_path=str(config_path))


@pytest.fixture
def test_data():
    """Crée des données de test pour SmartScoresCalculator."""
    timestamps = pd.date_range("2025-04-14 09:00", periods=100, freq="1min")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "atr_14": np.random.uniform(0.5, 2.0, 100),
            "volume": np.random.randint(100, 1000, 100),
            "absorption_strength": np.random.uniform(0, 100, 100),
            "delta_volume": np.random.uniform(-500, 500, 100),
            "close": np.random.normal(5100, 10, 100),
        }
    )


def test_calculate_smart_scores(smart_scores_calculator, test_data):
    """Teste calculate_smart_scores."""
    result = smart_scores_calculator.calculate_smart_scores(test_data, regime="trend")
    assert not result.empty, "Le DataFrame résultat ne doit pas être vide"
    expected_columns = [
        "breakout_score",
        "mm_score",
        "hft_score",
        "breakout_score_weighted",
        "mm_score_weighted",
        "hft_score_weighted",
    ]
    assert all(
        col in result.columns for col in expected_columns
    ), "Colonnes attendues manquantes"
    assert (
        not result["breakout_score"].isna().all()
    ), "breakout_score ne doit pas contenir uniquement des NaN"
    assert (
        not result["mm_score"].isna().all()
    ), "mm_score ne doit pas contenir uniquement des NaN"
    assert (
        not result["hft_score"].isna().all()
    ), "hft_score ne doit pas contenir uniquement des NaN"
    # Vérifier la pondération pour le régime 'trend'
    assert all(
        result["mm_score_weighted"] == result["mm_score"] * 1.5
    ), "Pondération mm_score incorrecte pour trend"
    assert all(
        result["hft_score_weighted"] == result["hft_score"] * 1.5
    ), "Pondération hft_score incorrecte pour trend"
    assert all(
        result["breakout_score_weighted"] == result["breakout_score"] * 1.0
    ), "Pondération breakout_score incorrecte pour trend"
    # Vérifier que les scores sont normalisés entre 0 et 1
    assert (
        result["breakout_score"].between(0, 1).all()
    ), "breakout_score doit être normalisé entre 0 et 1"
    assert (
        result["mm_score"].between(0, 1).all()
    ), "mm_score doit être normalisé entre 0 et 1"
    assert (
        result["hft_score"].between(0, 1).all()
    ), "hft_score doit être normalisé entre 0 et 1"


def test_calculate_smart_scores_empty_data(smart_scores_calculator):
    """Teste calculate_smart_scores avec un DataFrame vide."""
    empty_data = pd.DataFrame()
    result = smart_scores_calculator.calculate_smart_scores(empty_data, regime="range")
    expected_columns = [
        "breakout_score",
        "mm_score",
        "hft_score",
        "breakout_score_weighted",
        "mm_score_weighted",
        "hft_score_weighted",
    ]
    assert result.empty or all(
        result[col].eq(0).all() for col in expected_columns
    ), "Les scores doivent être à 0 pour un DataFrame vide"


def test_calculate_smart_scores_invalid_regime(smart_scores_calculator, test_data):
    """Teste calculate_smart_scores avec un régime invalide."""
    with pytest.raises(ValueError, match="Régime invalide"):
        smart_scores_calculator.calculate_smart_scores(test_data, regime="invalid")


def test_calculate_smart_scores_missing_columns(smart_scores_calculator, test_data):
    """Teste calculate_smart_scores avec des colonnes manquantes."""
    incomplete_data = test_data.drop(columns=["atr_14", "volume"])
    result = smart_scores_calculator.calculate_smart_scores(
        incomplete_data, regime="range"
    )
    assert not result.empty, "Le DataFrame résultat ne doit pas être vide"
    assert (
        result["breakout_score"].eq(0).all()
    ), "breakout_score doit être 0 si atr_14 ou volume manquent"
    assert (
        not result["mm_score"].eq(0).all()
    ), "mm_score ne doit pas être 0 si absorption_strength est présent"
    assert (
        not result["hft_score"].eq(0).all()
    ), "hft_score ne doit pas être 0 si delta_volume est présent"


def test_cache_scores(smart_scores_calculator, test_data, tmp_path):
    """Teste cache_scores."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    global CACHE_DIR
    CACHE_DIR = str(cache_dir)  # Redéfinir temporairement CACHE_DIR pour les tests
    result = smart_scores_calculator.calculate_smart_scores(test_data, regime="range")
    cache_key = hashlib.sha256(f"{result.to_json()}_range".encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.csv")
    assert os.path.exists(
        cache_path
    ), "Le fichier cache doit exister après calculate_smart_scores"
    cached_data = pd.read_csv(cache_path)
    assert len(cached_data) == len(
        result
    ), "Les données mises en cache doivent correspondre aux données calculées"
    CACHE_DIR = os.path.join(
        BASE_DIR, "data", "features", "cache", "smart_scores"
    )  # Restaurer CACHE_DIR


def test_validate_shap_features(smart_scores_calculator, tmp_path):
    """Teste validate_shap_features avec un fichier SHAP vide."""
    feature_importance_path = tmp_path / "feature_importance.csv"
    pd.DataFrame({"feature": ["dummy_feature"]}).to_csv(feature_importance_path)
    global FEATURE_IMPORTANCE_PATH
    FEATURE_IMPORTANCE_PATH = str(feature_importance_path)  # Redéfinir temporairement
    features = ["breakout_score", "mm_score", "hft_score"]
    assert not smart_scores_calculator.validate_shap_features(
        features
    ), "La validation doit échouer avec un fichier SHAP insuffisant"
    FEATURE_IMPORTANCE_PATH = os.path.join(
        BASE_DIR, "data", "features", "feature_importance.csv"
    )  # Restaurer


def test_clean_cache(smart_scores_calculator, tmp_path):
    """Teste _clean_cache."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    global CACHE_DIR
    CACHE_DIR = str(cache_dir)  # Redéfinir temporairement
    # Créer un fichier cache fictif
    cache_path = os.path.join(CACHE_DIR, "test_cache.csv")
    pd.DataFrame({"dummy": [1, 2, 3]}).to_csv(cache_path)
    smart_scores_calculator._clean_cache(max_size_mb=0.0001)  # Forcer suppression
    assert not os.path.exists(cache_path), "Le fichier cache doit être supprimé"
    CACHE_DIR = os.path.join(
        BASE_DIR, "data", "features", "cache", "smart_scores"
    )  # Restaurer


def test_load_config_with_manager_new(smart_scores_calculator, test_data):
    """Teste load_config_with_manager_new."""
    config = smart_scores_calculator.load_config_with_manager_new()
    assert isinstance(config, dict), "La configuration doit être un dictionnaire"
    assert "buffer_size" in config, "buffer_size doit être dans la configuration"
    assert "weights" in config, "weights doit être dans la configuration"
    assert (
        config["weights"]["trend"]["mm_score"] == 1.5
    ), "Poids mm_score pour trend incorrect"


def test_handle_sigint(smart_scores_calculator, tmp_path):
    """Teste handle_sigint."""
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    global SNAPSHOT_DIR
    SNAPSHOT_DIR = str(snapshot_dir)  # Redéfinir temporairement
    with pytest.raises(SystemExit, match="Terminated by SIGINT"):
        smart_scores_calculator.handle_sigint(signal.SIGINT, None)
    snapshot_files = list(snapshot_dir.glob("snapshot_sigint_*.json.gz"))
    assert len(snapshot_files) > 0, "Un snapshot SIGINT doit être créé"
    SNAPSHOT_DIR = os.path.join(
        BASE_DIR, "data", "features", "smart_scores_snapshots"
    )  # Restaurer
