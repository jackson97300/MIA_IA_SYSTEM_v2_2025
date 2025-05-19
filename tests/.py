# tests/test_adaptive_learner.py
import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.features.adaptive_learner import AdaptiveLearner


@pytest.fixture
def learner():
    """Initialise AdaptiveLearner pour les tests."""
    return AdaptiveLearner(training_mode=True)


@pytest.fixture
def sample_data():
    """Fournit des données d'exemple avec 350 features."""
    data = pd.DataFrame(
        {
            "timestamp": [datetime.now()],
            "vix_es_correlation": [20.0],
            "reward": [100.0],
            "neural_regime": [0],
        }
    )
    for i in range(349):  # 350 features incluant vix_es_correlation
        data[f"feature_{i}"] = np.random.normal(0, 1, 1)
    return data


def test_validate_data(learner, sample_data):
    """Vérifie la validation des données pour 350 features."""
    assert learner.validate_data(sample_data), "Validation des données échouée"
    assert len(sample_data.columns) >= 350, "Nombre de features incorrect"


def test_store_pattern(learner, sample_data, tmp_path):
    """Vérifie le stockage d'un pattern dans market_memory.db."""
    db_path = tmp_path / "market_memory.db"
    learner.initialize_database(db_path=str(db_path))
    asyncio.run(
        learner.store_pattern(sample_data, action=0.5, reward=100.0, neural_regime=0)
    )
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM patterns")
    count = cursor.fetchone()[0]
    conn.close()
    assert count == 1, "Pattern non stocké"


def test_cluster_patterns(learner, sample_data, tmp_path):
    """Vérifie le clustering des patterns et la visualisation."""
    db_path = tmp_path / "market_memory.db"
    learner.initialize_database(db_path=str(db_path))
    for _ in range(10):  # Ajouter plusieurs patterns
        asyncio.run(
            learner.store_pattern(
                sample_data,
                action=0.5,
                reward=100.0,
                neural_regime=0,
                db_path=str(db_path),
            )
        )
    clusters = asyncio.run(learner.cluster_patterns(n_clusters=5, db_path=str(db_path)))
    assert len(clusters) > 0, "Clustering échoué"
    assert os.path.exists(
        learner.figure_dir / f"cluster_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    ), "Visualisation non générée"


def test_retrain_model(learner, sample_data, tmp_path):
    """Vérifie le fine-tuning du modèle avec clustering."""
    model = asyncio.run(
        learner.retrain_model(sample_data, algo_type="sac", regime="range")
    )
    assert model is not None, "Fine-tuning échoué"
    assert os.path.exists(
        learner.figure_dir
        / f"cluster_sac_range_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    ), "Visualisation non générée"


def test_performance_logging(learner, sample_data, tmp_path):
    """Vérifie les logs de performance."""
    db_path = tmp_path / "market_memory.db"
    learner.initialize_database(db_path=str(db_path))
    asyncio.run(
        learner.store_pattern(
            sample_data, action=0.5, reward=100.0, neural_regime=0, db_path=str(db_path)
        )
    )
    assert os.path.exists(CSV_LOG_PATH), "Fichier de performance non créé"
    log_df = pd.read_csv(CSV_LOG_PATH)
    assert len(log_df) > 0, "Aucun log de performance"
    assert "operation" in log_df.columns, "Colonne operation manquante"
    assert "cpu_usage_percent" in log_df.columns, "Colonne cpu_usage_percent manquante"
