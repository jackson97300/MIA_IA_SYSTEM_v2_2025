"""
test_signal_resolver.py - Tests unitaires pour le module signal_resolver.py dans MIA_IA_SYSTEM_v2_2025.

Version : 2.1.5
Date : 2025-05-14

Rôle : Valide les fonctionnalités du module signal_resolver.py, incluant la résolution des conflits de signaux, la normalisation, la persistance dans market_memory.db, l'exportation en CSV, le calcul de l'entropie, le coefficient de conflit, le mode debug, et la gestion des erreurs.

Dépendances :
- pytest : Exécution des tests unitaires.
- pandas, numpy : Manipulation des données de test.
- sqlite3 : Vérification des enregistrements dans market_memory.db.
- unittest.mock : Simulation d'AlertManager pour éviter les appels réels.
- loguru : Vérification des logs.

Conformité : Aucune référence à dxFeed, obs_t, 320 features, ou 81 features.
Note sur les policies : Le répertoire officiel pour les politiques de routage est src/model/router/policies.
"""

import json
import sqlite3
from unittest.mock import patch

import pandas as pd
import pytest

from src.model.utils.signal_resolver import SignalResolver


@pytest.fixture
def signal_resolver(tmp_path):
    """
    Fixture pour initialiser SignalResolver avec un répertoire temporaire.

    Args:
        tmp_path: Répertoire temporaire fourni par pytest.

    Yields:
        SignalResolver: Instance configurée pour les tests.
    """
    market = "ES"
    config_path = tmp_path / "es_config.yaml"
    config = {
        "signal_resolver": {
            "default_weights": {
                "regime_trend": 2.0,
                "qr_dqn_positive": 1.5,
                "microstructure_bullish": 1.0,
                "news_score_positive": 0.5,
            }
        }
    }
    with open(config_path, "w") as f:
        json.dump(config, f)

    resolver = SignalResolver(config_path=str(config_path), market=market)
    resolver.db_path = tmp_path / f"market_memory_{market}.db"
    resolver.log_path = tmp_path / "signal_resolver.log"
    resolver.csv_path = tmp_path / "conflict_scores.csv"
    resolver._initialize_db()

    yield resolver


@pytest.fixture
def signals():
    """
    Fixture pour fournir des signaux de test.
    """
    return {
        "regime_trend": 1.0,
        "microstructure_bullish": 0.0,
        "news_score_positive": 1.0,
        "qr_dqn_positive": 0.0,
    }


@pytest.fixture
def weights():
    """
    Fixture pour fournir des poids de test.
    """
    return {
        "regime_trend": 2.0,
        "qr_dqn_positive": 1.5,
        "microstructure_bullish": 1.0,
        "news_score_positive": 0.5,
    }


def test_resolve_conflict(signal_resolver, signals, weights):
    """
    Teste la résolution des conflits sans normalisation ni persistance.
    """
    score, metadata = signal_resolver.resolve_conflict(signals, weights)
    expected_score = 1.0 * 2.0 + 0.0 * 1.0 + 1.0 * 0.5 + 0.0 * 1.5  # 2.5
    assert abs(score - expected_score) < 1e-6
    assert metadata["score"] == score
    assert "contributions" in metadata
    assert metadata["contributions"]["regime_trend"]["contribution"] == 2.0
    assert metadata["contributions"]["news_score_positive"]["contribution"] == 0.5


def test_resolve_conflict_normalized(signal_resolver, signals, weights):
    """
    Teste la résolution des conflits avec normalisation.
    """
    score, metadata = signal_resolver.resolve_conflict(signals, weights, normalize=True)
    total_weight = 2.0 + 1.5 + 1.0 + 0.5  # 5.0
    expected_score = (
        1.0 * 2.0 + 0.0 * 1.0 + 1.0 * 0.5 + 0.0 * 1.5
    ) / total_weight  # 2.5 / 5.0 = 0.5
    assert abs(score - expected_score) < 1e-6
    assert metadata["normalized_score"] == score
    assert metadata["score"] == 2.5


def test_resolve_conflict_db_persistence(signal_resolver, signals, weights):
    """
    Teste la persistance des scores dans market_memory.db.
    """
    run_id = "test_run_123"
    score, metadata = signal_resolver.resolve_conflict(
        signals, weights, persist_to_db=True, run_id=run_id
    )

    with sqlite3.connect(signal_resolver.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM conflict_scores WHERE run_id = ?", (run_id,))
        result = cursor.fetchone()
        assert result is not None
        assert result[1] == signal_resolver.market
        assert abs(result[2] - score) < 1e-6
        assert result[4] == run_id
        contributions = json.loads(result[3])
        assert contributions == metadata["contributions"]


def test_resolve_conflict_csv_export(signal_resolver, signals, weights):
    """
    Teste l'exportation des scores en CSV.
    """
    run_id = "test_run_456"
    signal_resolver.csv_buffer_limit = 1  # Forcer l'écriture immédiate
    score, metadata = signal_resolver.resolve_conflict(
        signals, weights, export_to_csv=True, run_id=run_id
    )

    assert signal_resolver.csv_path.exists()
    df = pd.read_csv(signal_resolver.csv_path)
    assert len(df) == 1
    assert df.iloc[0]["market"] == signal_resolver.market
    assert abs(df.iloc[0]["score"] - score) < 1e-6
    assert df.iloc[0]["run_id"] == run_id
    contributions = json.loads(df.iloc[0]["contributions_json"])
    assert contributions == metadata["contributions"]


def test_resolve_conflict_entropy(signal_resolver, signals, weights):
    """
    Teste le calcul de l'entropie pour la cohérence des contributions.
    """
    score, metadata = signal_resolver.resolve_conflict(signals, weights)
    contributions = metadata["contributions"]
    total_weight = sum(weights.get(name, 1.0) for name in signals)
    contributions_normalized = [
        abs(c["contribution"]) / total_weight if total_weight > 0 else 0
        for c in contributions.values()
    ]
    contributions_normalized = [max(1e-10, c) for c in contributions_normalized]
    expected_entropy = (
        entropy(contributions_normalized, base=2) if contributions_normalized else 0.0
    )
    assert abs(metadata["entropy"] - expected_entropy) < 1e-6


@patch("src.model.utils.alert_manager.AlertManager.send_alert")
def test_resolve_conflict_alerts(mock_alert, signal_resolver, signals, weights):
    """
    Teste le déclenchement des alertes pour les scores négatifs ou faibles.
    """
    signals_negative = signals.copy()
    signals_negative["regime_trend"] = -1.0
    score, metadata = signal_resolver.resolve_conflict(signals_negative, weights)
    assert mock_alert.called
    assert "Score de conflit négatif" in mock_alert.call_args[0][0]

    signals_low = signals.copy()
    signals_low["regime_trend"] = 0.0
    signals_low["news_score_positive"] = 0.0
    score, metadata = signal_resolver.resolve_conflict(signals_low, weights)
    assert mock_alert.called
    assert "Score de conflit faible" in mock_alert.call_args[0][0]


def test_resolve_conflict_invalid_signals(signal_resolver, signals, weights):
    """
    Teste la gestion des erreurs pour les signaux invalides.
    """
    invalid_signals = signals.copy()
    invalid_signals["regime_trend"] = 2.0
    with pytest.raises(
        ValueError, match="Signal regime_trend doit être un nombre entre -1 et 1"
    ):
        signal_resolver.resolve_conflict(invalid_signals, weights)


def test_resolve_conflict_zero_weights(signal_resolver, signals, weights):
    """
    Teste la normalisation avec une somme de poids nulle.
    """
    zero_weights = {k: 0.0 for k in weights}
    score, metadata = signal_resolver.resolve_conflict(
        signals, zero_weights, normalize=True
    )
    assert score == 0.0  # Score brut, normalisation ignorée
    assert metadata["normalized_score"] is None


def test_resolve_conflict_with_score_type(signal_resolver, signals, weights):
    """
    Vérifie que score_type est correctement inclus dans les métadonnées.
    """
    score, metadata = signal_resolver.resolve_conflict(
        signals, weights, score_type="intermediate"
    )
    assert metadata["score_type"] == "intermediate"


def test_conflict_coefficient_calculation(signal_resolver, signals, weights):
    """
    Vérifie le calcul du coefficient de conflit.
    """
    score, metadata = signal_resolver.resolve_conflict(signals, weights)
    contribs = metadata["contributions"]
    total = sum(abs(c["contribution"]) for c in contribs.values())
    max_c = max(abs(c["contribution"]) for c in contribs.values())
    expected = 1.0 - (max_c / total) if total > 0 else 0.0
    assert abs(metadata["conflict_coefficient"] - expected) < 1e-6


def test_resolve_conflict_debug_mode(signal_resolver, signals, weights):
    """
    Vérifie que le mode debug s'exécute sans erreur.
    """
    score, metadata = signal_resolver.resolve_conflict(
        signals, weights, mode_debug=True
    )
    assert "conflict_coefficient" in metadata
    assert "entropy" in metadata


if __name__ == "__main__":
    pytest.main(["-v", __file__])
