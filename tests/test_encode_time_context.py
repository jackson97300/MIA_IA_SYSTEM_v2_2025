# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_encode_time_context.py
# Tests unitaires pour la classe TimeContextEncoder.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie les fonctionnalités de TimeContextEncoder, incluant le calcul des 8 métriques temporelles
#        (ex. : time_of_day_sin, day_of_week, option_expiry_proximity, cluster_id), la clusterisation,
#        la gestion des erreurs, et l'intégration des Phases 1-18.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0
# - src/features/extractors/encode_time_context.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
#
# Notes :
# - Utilise exclusivement des données simulées compatibles avec IQFeed.
# - Suppression de toute référence à dxFeed, obs_t, 320 features, et 81 features.
# - Couvre les Phases 1-18 :
#   - Phase 1 : Contexte pour l'analyse de sentiment (évolution future).
#   - Phase 13 : Métrique option_expiry_proximity.
#   - Phase 15 : Métrique microstructure_time_sensitivity.
#   - Phase 18 : Métrique hft_time_sensitivity.

import os

import numpy as np
import pandas as pd
import pytest

from src.features.extractors.encode_time_context import TimeContextEncoder

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "market_memory.db")


@pytest.fixture
def time_context_encoder(tmp_path):
    """Crée une instance de TimeContextEncoder avec un répertoire temporaire."""
    config_path = tmp_path / "es_config.yaml"
    config_path.write_text(
        """
        time_context:
          timezone: "America/New_York"
          trading_session_start: "09:30"
          trading_session_end: "16:00"
          n_clusters: 10
          buffer_size: 10
          max_cache_size: 100
          buffer_maxlen: 1000
          cache_hours: 24
        """
    )
    return TimeContextEncoder(config_path=str(config_path))


@pytest.fixture
def test_data():
    """Crée des données de test pour TimeContextEncoder."""
    timestamps = pd.date_range("2025-04-14 09:00", periods=100, freq="1min")
    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": np.random.normal(5100, 10, 100),
            "volume": np.random.randint(100, 1000, 100),
        }
    )
    options_data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "expiry_date": pd.date_range("2025-04-15", periods=100, freq="1d"),
        }
    )
    microstructure_data = pd.DataFrame(
        {"timestamp": timestamps, "spoofing_score": np.random.uniform(0, 1, 100)}
    )
    hft_data = pd.DataFrame(
        {"timestamp": timestamps, "hft_activity_score": np.random.uniform(0, 1, 100)}
    )
    return {
        "data": data,
        "options_data": options_data,
        "microstructure_data": microstructure_data,
        "hft_data": hft_data,
    }


def test_encode_time_context(time_context_encoder, test_data):
    """Teste encode_time_context."""
    result = time_context_encoder.encode_time_context(
        test_data["data"],
        test_data["options_data"],
        test_data["microstructure_data"],
        test_data["hft_data"],
    )
    assert not result.empty, "Le DataFrame résultat ne doit pas être vide"
    expected_columns = [
        "time_of_day_sin",
        "time_of_day_cos",
        "day_of_week",
        "is_trading_session",
        "is_expiry_day",
        "option_expiry_proximity",
        "microstructure_time_sensitivity",
        "hft_time_sensitivity",
        "cluster_id",
    ]
    assert all(
        col in result.columns for col in expected_columns
    ), "Colonnes attendues manquantes"
    assert (
        result["cluster_id"].nunique() <= 10
    ), "Nombre de clusters doit être inférieur ou égal à n_clusters"


def test_encode_time_context_empty_data(time_context_encoder, test_data):
    """Teste encode_time_context avec un DataFrame vide."""
    empty_data = pd.DataFrame()
    result = time_context_encoder.encode_time_context(
        empty_data,
        test_data["options_data"],
        test_data["microstructure_data"],
        test_data["hft_data"],
    )
    assert result.empty or all(
        result[col].eq(0).all() for col in result.columns if col != "timestamp"
    ), "Les métriques doivent être à 0 pour un DataFrame vide"


def test_compute_option_expiry_proximity(time_context_encoder, test_data):
    """Teste compute_option_expiry_proximity (Phase 13)."""
    result = time_context_encoder.compute_option_expiry_proximity(
        test_data["data"], test_data["options_data"]
    )
    assert isinstance(result, pd.Series), "Le résultat doit être une pd.Series"
    assert len(result) == len(
        test_data["data"]
    ), "La longueur du résultat doit correspondre aux données d'entrée"
    assert (
        not result.isna().all()
    ), "Le résultat ne doit pas contenir uniquement des NaN"


def test_compute_microstructure_time_sensitivity(time_context_encoder, test_data):
    """Teste compute_microstructure_time_sensitivity (Phase 15)."""
    result = time_context_encoder.compute_microstructure_time_sensitivity(
        test_data["data"], test_data["microstructure_data"]
    )
    assert isinstance(result, pd.Series), "Le résultat doit être une pd.Series"
    assert len(result) == len(
        test_data["data"]
    ), "La longueur du résultat doit correspondre aux données d'entrée"
    assert (
        not result.isna().all()
    ), "Le résultat ne doit pas contenir uniquement des NaN"


def test_compute_hft_time_sensitivity(time_context_encoder, test_data):
    """Teste compute_hft_time_sensitivity (Phase 18)."""
    result = time_context_encoder.compute_hft_time_sensitivity(
        test_data["data"], test_data["hft_data"]
    )
    assert isinstance(result, pd.Series), "Le résultat doit être une pd.Series"
    assert len(result) == len(
        test_data["data"]
    ), "La longueur du résultat doit correspondre aux données d'entrée"
    assert (
        not result.isna().all()
    ), "Le résultat ne doit pas contenir uniquement des NaN"


def test_cluster_time_features(time_context_encoder, test_data):
    """Teste cluster_time_features."""
    # Préparer un DataFrame avec toutes les métriques nécessaires
    data = test_data["data"].copy()
    data["time_of_day_sin"] = np.sin(2 * np.pi * data["timestamp"].dt.hour / 24)
    data["time_of_day_cos"] = np.cos(2 * np.pi * data["timestamp"].dt.hour / 24)
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    data["is_trading_session"] = (
        data["timestamp"]
        .dt.time.between(pd.Timestamp("09:30").time(), pd.Timestamp("16:00").time())
        .astype(int)
    )
    data["is_expiry_day"] = (
        (data["timestamp"].dt.day_name() == "Friday")
        & (data["timestamp"].dt.day >= 15)
        & (data["timestamp"].dt.day <= 21)
    ).astype(int)
    data["option_expiry_proximity"] = (
        time_context_encoder.compute_option_expiry_proximity(
            data, test_data["options_data"]
        )
    )
    data["microstructure_time_sensitivity"] = (
        time_context_encoder.compute_microstructure_time_sensitivity(
            data, test_data["microstructure_data"]
        )
    )
    data["hft_time_sensitivity"] = time_context_encoder.compute_hft_time_sensitivity(
        data, test_data["hft_data"]
    )

    result = time_context_encoder.cluster_time_features(data)
    assert isinstance(result, pd.Series), "Le résultat doit être une pd.Series"
    assert len(result) == len(
        data
    ), "La longueur du résultat doit correspondre aux données d'entrée"
    assert (
        result.nunique() <= 10
    ), "Nombre de clusters doit être inférieur ou égal à n_clusters"
    # Vérifier que la base de données a été mise à jour
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM clusters")
        count = cursor.fetchone()[0]
        assert (
            count > 0
        ), "La table clusters doit contenir des données après la clusterisation"
