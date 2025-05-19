# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_shap_weighting.py
# Tests unitaires pour src/features/shap_weighting.py
# Version : 2.1.5
# Date : 2025-05-13
# Rôle : Valide le calcul des poids SHAP pour les top 150 features (méthode 17),
#        l'entraînement du modèle XGBoost, la gestion des nouvelles métriques de spotgamma_recalculator.py,
# avec confidence_drop_rate (Phase 8), snapshots, visualisations, et
# alertes Telegram.

import gzip
import json
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.features.shap_weighting import SHAPWeighting


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, snapshots, figures, features, et modèles."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    snapshots_dir = data_dir / "shap_snapshots"
    snapshots_dir.mkdir()
    figures_dir = data_dir / "figures" / "shap"
    figures_dir.mkdir(parents=True)
    features_dir = data_dir / "features"
    features_dir.mkdir()
    models_dir = data_dir / "models"
    models_dir.mkdir()

    # Créer es_config.yaml
    config_path = config_dir / "es_config.yaml"
    config_content = {
        "shap_weighting": {
            "model_path": str(models_dir / "shap_regime_detector.pkl"),
            "window_size": 100,
            "regime_mapping": {"Trend": 0, "Range": 1, "Defensive": 2},
            "buffer_size": 100,
            "max_cache_size": 1000,
            "buffer_maxlen": 1000,
            "cache_hours": 24,
            "expected_count": 320,
            "key_features_count": 150,
        }
    }
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)

    return {
        "base_dir": str(base_dir),
        "config_path": str(config_path),
        "logs_dir": str(logs_dir),
        "snapshots_dir": str(snapshots_dir),
        "figures_dir": str(figures_dir),
        "features_dir": str(features_dir),
        "models_dir": str(models_dir),
        "perf_log_path": str(logs_dir / "shap_performance.csv"),
        "feature_importance_path": str(features_dir / "feature_importance.csv"),
        "model_path": str(models_dir / "shap_regime_detector.pkl"),
    }


@pytest.fixture
def mock_data():
    """Crée des données factices pour tester."""
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=100, freq="1min"),
            "close": np.random.normal(5100, 10, 100),
            "neural_regime": ["Trend"] * 20 + ["Range"] * 40 + ["Defensive"] * 40,
            "atr_14": np.random.uniform(0.5, 2.0, 100),
            "volatility_trend": np.random.uniform(-0.1, 0.1, 100),
            "spy_lead_return": np.random.uniform(-0.02, 0.02, 100),
            "key_strikes_1": np.random.uniform(5000, 5200, 100),
            "key_strikes_2": np.random.uniform(5000, 5200, 100),
            "key_strikes_3": np.random.uniform(5000, 5200, 100),
            "key_strikes_4": np.random.uniform(5000, 5200, 100),
            "key_strikes_5": np.random.uniform(5000, 5200, 100),
            "max_pain_strike": np.random.uniform(5000, 5200, 100),
            "net_gamma": np.random.uniform(-1, 1, 100),
            "zero_gamma": np.random.uniform(5000, 5200, 100),
            "dealer_zones_count": np.random.randint(0, 11, 100),
            "vol_trigger": np.random.uniform(-1, 1, 100),
            "ref_px": np.random.uniform(5000, 5200, 100),
            "data_release": np.random.randint(0, 2, 100),
        }
    )
    for i in range(320 - len(data.columns) + 1):
        data[f"feature_{i}"] = np.random.normal(0, 1, 100)
    return data


@pytest.fixture
def mock_single_row_data(mock_data):
    """Crée une seule ligne de données factices pour tester."""
    return mock_data.iloc[-1]


@pytest.fixture
def mock_feature_importance(tmp_dirs):
    """Crée un fichier feature_importance.csv factice pour la validation SHAP."""
    features = [
        "key_strikes_1",
        "key_strikes_2",
        "key_strikes_3",
        "key_strikes_4",
        "key_strikes_5",
        "max_pain_strike",
        "net_gamma",
        "zero_gamma",
        "dealer_zones_count",
        "vol_trigger",
        "ref_px",
        "data_release",
        "atr_14",
        "volatility_trend",
        "spy_lead_return",
    ] + [f"feature_{i}" for i in range(135)]
    shap_data = pd.DataFrame(
        {
            "feature": features[:150],
            "importance": [0.1] * 150,
            "regime": ["range"] * 150,
        }
    )
    shap_data.to_csv(tmp_dirs["feature_importance_path"], index=False, encoding="utf-8")
    return shap_data


def test_shap_weighting_init(tmp_dirs, mock_feature_importance):
    """Teste l’initialisation de SHAPWeighting."""
    calculator = SHAPWeighting(config_path=tmp_dirs["config_path"])
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


def test_load_config_invalid(tmp_dirs):
    """Teste le chargement d’une configuration invalide."""
    invalid_config_path = os.path.join(tmp_dirs["config_dir"], "invalid_config.yaml")
    with open(invalid_config_path, "w") as f:
        yaml.dump({"wrong_key": {}}, f)
    with pytest.raises(ValueError, match="Clé 'shap_weighting' manquante"):
        calculator = SHAPWeighting(config_path=invalid_config_path)
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "Clé 'shap_weighting' manquante" in str(e) for e in df["error"].dropna()
    ), "Erreur non loguée"


def test_train_shap_regime_detector_valid(tmp_dirs, mock_data, mock_feature_importance):
    """Teste l’entraînement du modèle XGBoost avec des données valides."""
    calculator = SHAPWeighting(config_path=tmp_dirs["config_path"])
    calculator.train_shap_regime_detector(mock_data, tmp_dirs["model_path"])
    assert os.path.exists(tmp_dirs["model_path"]), "Modèle XGBoost non sauvegardé"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_train_shap_regime_detector" in f and f.endswith(".json")
        for f in snapshot_files
    ), "Snapshot non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "train_shap_regime_detector" in op for op in df["operation"]
    ), "Opération d’entraînement non loguée"
    assert "confidence_drop_rate" in df.columns or any(
        "confidence_drop_rate" in str(kw) for kw in df.to_dict("records")
    ), "confidence_drop_rate absent"


def test_train_shap_regime_detector_invalid(tmp_dirs, mock_data):
    """Teste l’entraînement avec des données invalides."""
    calculator = SHAPWeighting(config_path=tmp_dirs["config_path"])
    invalid_data = mock_data.drop(columns=["neural_regime"])
    with pytest.raises(ValueError, match="Colonne 'neural_regime' manquante"):
        calculator.train_shap_regime_detector(invalid_data, tmp_dirs["model_path"])
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "Colonne 'neural_regime' manquante" in str(e) for e in df["error"].dropna()
    ), "Erreur non loguée"


def test_calculate_shap_weights_valid(tmp_dirs, mock_data, mock_feature_importance):
    """Teste le calcul des poids SHAP avec des données valides."""
    calculator = SHAPWeighting(config_path=tmp_dirs["config_path"])
    with patch(
        "shap.TreeExplainer",
        return_value=MagicMock(shap_values=lambda x: np.random.uniform(-1, 1, len(x))),
    ) as mock_explainer:
        shap_weights = calculator.calculate_shap_weights(mock_data)
        assert len(shap_weights) <= 150, "Nombre de features dépasse 150"
        assert {"feature", "importance", "regime"}.issubset(
            shap_weights.columns
        ), "Colonnes manquantes dans shap_weights"
        assert os.path.exists(
            tmp_dirs["feature_importance_path"]
        ), "Fichier feature_importance.csv non créé"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
        assert any(
            "snapshot_calculate_shap_weights" in f and f.endswith(".json")
            for f in snapshot_files
        ), "Snapshot non créé"
        assert any(
            f.startswith("shap_bar_") and f.endswith(".png")
            for f in os.listdir(tmp_dirs["figures_dir"])
        ), "Visualisation non générée"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert "confidence_drop_rate" in df.columns or any(
            "confidence_drop_rate" in str(kw) for kw in df.to_dict("records")
        ), "confidence_drop_rate absent"


def test_calculate_shap_weights_empty(tmp_dirs, mock_feature_importance):
    """Teste le calcul des poids SHAP avec un DataFrame vide."""
    calculator = SHAPWeighting(config_path=tmp_dirs["config_path"])
    empty_data = pd.DataFrame()
    shap_weights = calculator.calculate_shap_weights(empty_data)
    assert shap_weights.empty, "Résultat non vide pour DataFrame vide"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "DataFrame vide" in str(e) for e in df["error"].dropna()
    ), "Erreur DataFrame vide non loguée"


def test_calculate_incremental_shap_weights_valid(
    tmp_dirs, mock_single_row_data, mock_feature_importance
):
    """Teste le calcul incrémental des poids SHAP."""
    calculator = SHAPWeighting(config_path=tmp_dirs["config_path"])
    with patch(
        "shap.TreeExplainer",
        return_value=MagicMock(shap_values=lambda x: np.random.uniform(-1, 1, len(x))),
    ) as mock_explainer:
        for _ in range(100):  # Remplir le buffer
            calculator.calculate_incremental_shap_weights(mock_single_row_data)
        shap_weights = calculator.calculate_incremental_shap_weights(
            mock_single_row_data
        )
        assert len(shap_weights) <= 150, "Nombre de features dépasse 150"
        assert {"feature", "importance", "regime"}.issubset(
            shap_weights.columns
        ), "Colonnes manquantes dans shap_weights"
        assert os.path.exists(
            tmp_dirs["feature_importance_path"]
        ), "Fichier feature_importance.csv non créé"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
        assert any(
            "snapshot_calculate_incremental_shap_weights" in f and f.endswith(".json")
            for f in snapshot_files
        ), "Snapshot non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert "confidence_drop_rate" in df.columns or any(
            "confidence_drop_rate" in str(kw) for kw in df.to_dict("records")
        ), "confidence_drop_rate absent"


def test_validate_shap_features(tmp_dirs, mock_feature_importance):
    """Teste la validation des top 150 SHAP features."""
    calculator = SHAPWeighting(config_path=tmp_dirs["config_path"])
    features = ["key_strikes_1", "net_gamma", "vol_trigger"]
    result = calculator.validate_shap_features(features)
    assert result, "Validation SHAP échouée pour features valides"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_validate_shap_features" in f for f in snapshot_files
    ), "Snapshot non créé"


def test_save_snapshot_compressed(tmp_dirs):
    """Teste la sauvegarde d’un snapshot compressé avec gzip."""
    calculator = SHAPWeighting(config_path=tmp_dirs["config_path"])
    snapshot_data = {"test": "compressed_snapshot"}
    calculator.save_snapshot("test_compressed", snapshot_data, compress=True)
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


def test_critical_alerts(tmp_dirs, mock_feature_importance):
    """Teste les alertes Telegram pour les erreurs critiques."""
    with patch("src.features.shap_weighting.send_telegram_alert") as mock_telegram:
        calculator = SHAPWeighting(config_path=tmp_dirs["config_path"])
        empty_data = pd.DataFrame()
        calculator.calculate_shap_weights(empty_data)
        mock_telegram.assert_called_with(pytest.any(str))
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "DataFrame vide" in str(e) for e in df["error"].dropna()
        ), "Erreur critique non loguée"
