# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_neural_pipeline.py
# Tests unitaires pour src/features/neural_pipeline.py.
#
# Version : 2.1.5
# Date : 2025-05-14
#
# Rôle : Valide le pipeline neuronal multi-architecture (LSTM, CNN, MLP volatilité, MLP classification)
#        pour la génération de cnn_pressure, neural_regime, predicted_vix avec LSTM dédié (méthode 12).
#        Couvre confidence_drop_rate (Phase 8), validation SHAP (Phase 17), snapshots, visualisations,
#        alertes Telegram, ScalerManager, typage précis, validation obs_t, cache LRU, et les nouvelles
#        features bid_ask_imbalance, trade_aggressiveness, iv_skew, iv_term_structure, option_skew,
#        news_impact_score. Conforme à la Phase 1 (collecte via IQFeed), Phase 8 (auto-conscience via alertes),
#        et Phase 16 (ensemble et transfer learning).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pandas>=2.0.0
# - numpy>=1.26.4
# - psutil>=5.9.8
# - src.features.neural_pipeline
# - src.model.utils.miya_console
# - src.model.utils.alert_manager
# - src.utils.telegram_alert
#
# Inputs :
# - Fichier de configuration factice (model_params.yaml)
# - Répertoires temporaires pour logs, snapshots, figures, features, et modèles
# - Données factices pour raw_data, options_data, orderflow_data
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de neural_pipeline.py.
#
# Notes :
# - Utilise des mocks pour simuler les dépendances externes (TensorFlow, alertes).
# - Vérifie l'absence de références à dxFeed, obs_t non validé, 320/81 features.
# - Couvre ScalerManager, cache LRU, validation obs_t, et nouvelles features.
# - Policies Note: The official directory for routing policies is src/model/router/policies.

import gzip
import json
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.features.neural_pipeline import NeuralPipeline, ScalerManager


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
    snapshots_dir = data_dir / "neural_snapshots"
    snapshots_dir.mkdir()
    figures_dir = data_dir / "figures" / "neural_pipeline"
    figures_dir.mkdir(parents=True)
    features_dir = data_dir / "features"
    features_dir.mkdir()
    models_dir = data_dir / "models"
    models_dir.mkdir()

    # Créer model_params.yaml
    config_path = config_dir / "model_params.yaml"
    config_content = {
        "neural_pipeline": {
            "lstm": {
                "units": 128,
                "dropout": 0.2,
                "hidden_layers": [64],
                "learning_rate": 0.001,
            },
            "cnn": {
                "filters": 32,
                "kernel_size": 5,
                "dropout": 0.1,
                "hidden_layers": [16],
                "learning_rate": 0.001,
            },
            "mlp_volatility": {
                "units": 128,
                "hidden_layers": [64],
                "learning_rate": 0.001,
            },
            "mlp_regime": {"units": 128, "hidden_layers": [64], "learning_rate": 0.001},
            "vix_lstm": {
                "units": 128,
                "dropout": 0.2,
                "hidden_layers": [64],
                "learning_rate": 0.001,
            },
            "batch_size": 32,
            "pretrain_epochs": 5,
            "validation_split": 0.2,
            "normalization": True,
            "save_dir": str(models_dir),
            "num_lstm_features": 8,
            "logging": {"buffer_size": 100},
            "cache": {"max_cache_size": 1000, "cache_hours": 24},
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
        "perf_log_path": str(logs_dir / "neural_pipeline_performance.csv"),
        "feature_importance_path": str(features_dir / "feature_importance.csv"),
        "dashboard_path": str(data_dir / "neural_pipeline_dashboard.json"),
    }


@pytest.fixture
def mock_data():
    """Crée des données factices pour tester, incluant les nouvelles features."""
    raw_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-14 09:00", periods=100, freq="1min"),
            "open": np.random.normal(5100, 10, 100),
            "high": np.random.normal(5105, 10, 100),
            "low": np.random.normal(5095, 10, 100),
            "close": np.random.normal(5100, 10, 100),
            "volume": np.random.randint(100, 1000, 100),
            "atr_14": np.random.uniform(0.5, 2.0, 100),
            "adx_14": np.random.uniform(10, 40, 100),
            "rsi_14": np.random.uniform(30, 70, 100),
            "vix_es_correlation": np.random.uniform(-0.5, 0.5, 100),
            "bid_ask_imbalance": np.random.normal(0, 0.1, 100),
            "trade_aggressiveness": np.random.normal(0, 0.2, 100),
            "iv_skew": np.random.normal(0.01, 0.005, 100),
            "iv_term_structure": np.random.normal(0.02, 0.005, 100),
        }
    )
    options_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-14 09:00", periods=100, freq="1min"),
            "gex": np.random.uniform(-1000, 1000, 100),
            "oi_peak_call_near": np.random.randint(5000, 15000, 100),
            "gamma_wall": np.random.uniform(5090, 5110, 100),
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
            "iv_atm": np.random.uniform(0.1, 0.3, 100),
            "option_skew": np.random.normal(0.02, 0.01, 100),
            "news_impact_score": np.random.uniform(0.0, 1.0, 100),
        }
    )
    orderflow_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-14 09:00", periods=100, freq="1min"),
            "bid_size_level_1": np.random.randint(50, 500, 100),
            "ask_size_level_1": np.random.randint(50, 500, 100),
            "ofi_score": np.random.uniform(-1, 1, 100),
        }
    )
    return raw_data, options_data, orderflow_data


@pytest.fixture
def mock_single_row_data(mock_data):
    """Crée une seule ligne de données factices pour tester."""
    raw_data, options_data, orderflow_data = mock_data
    return raw_data.iloc[-1], options_data.iloc[-1], orderflow_data.iloc[-1]


@pytest.fixture
def mock_feature_importance(tmp_dirs):
    """Crée un fichier feature_importance.csv factice pour la validation SHAP."""
    features = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "atr_14",
        "adx_14",
        "rsi_14",
        "vix_es_correlation",
        "gex",
        "oi_peak_call_near",
        "gamma_wall",
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
        "iv_atm",
        "bid_size_level_1",
        "ask_size_level_1",
        "ofi_score",
        "bid_ask_imbalance",
        "trade_aggressiveness",
        "iv_skew",
        "iv_term_structure",
        "option_skew",
        "news_impact_score",
    ] + [f"feature_{i}" for i in range(116)]
    shap_data = pd.DataFrame(
        {
            "feature": features[:150],
            "importance": [0.1] * 150,
            "regime": ["range"] * 150,
        }
    )
    shap_data.to_csv(tmp_dirs["feature_importance_path"], index=False, encoding="utf-8")
    return shap_data


def test_neural_pipeline_init(tmp_dirs, mock_feature_importance):
    """Teste l’initialisation de NeuralPipeline."""
    pipeline = NeuralPipeline(config_path=tmp_dirs["config_path"], training_mode=True)
    assert (
        pipeline.base_features == 350
    ), "Nombre de features incorrect (devrait être 350)"
    assert isinstance(
        pipeline.scaler_manager, ScalerManager
    ), "ScalerManager non initialisé"
    assert isinstance(
        pipeline.cache, OrderedDict
    ), "Cache non initialisé comme OrderedDict"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    assert os.path.exists(tmp_dirs["snapshots_dir"]), "Dossier de snapshots non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert all(
        col in df.columns
        for col in [
            "timestamp",
            "operation",
            "latency",
            "cpu_usage_percent",
            "memory_usage_mb",
        ]
    ), "Colonnes de performance manquantes"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_init" in f and f.endswith(".json") for f in snapshot_files
    ), "Snapshot non créé"


def test_preprocess_valid(tmp_dirs, mock_data, mock_feature_importance):
    """Teste le prétraitement avec des données valides."""
    pipeline = NeuralPipeline(config_path=tmp_dirs["config_path"], training_mode=True)
    raw_data, options_data, orderflow_data = mock_data
    lstm_input, cnn_input, full_data = pipeline.preprocess(
        raw_data, options_data, orderflow_data
    )
    assert lstm_input.shape == (
        51,
        50,
        16,
    ), f"Shape incorrect pour lstm_input: {lstm_input.shape}"
    assert cnn_input.shape == (
        51,
        50,
        7,
    ), f"Shape incorrect pour cnn_input: {cnn_input.shape}"
    assert (
        len(full_data) == 50
    ), f"Nombre de lignes incorrect pour full_data: {len(full_data)}"
    assert all(
        f in full_data.columns
        for f in [
            "bid_ask_imbalance",
            "trade_aggressiveness",
            "iv_skew",
            "iv_term_structure",
            "option_skew",
            "news_impact_score",
        ]
    ), "Nouvelles features absentes"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_preprocess" in f and f.endswith(".json") for f in snapshot_files
    ), "Snapshot non créé"
    with open(os.path.join(tmp_dirs["snapshots_dir"], snapshot_files[-1]), "r") as f:
        snapshot = json.load(f)
    assert "new_features" in snapshot["data"], "new_features absent du snapshot"
    assert set(snapshot["data"]["new_features"]) == {
        "bid_ask_imbalance",
        "trade_aggressiveness",
        "iv_skew",
        "iv_term_structure",
        "option_skew",
        "news_impact_score",
    }, "Nouvelles features incorrectes"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "confidence_drop_rate" in str(kw) for kw in df.to_dict("records")
    ), "confidence_drop_rate absent"


def test_preprocess_invalid(tmp_dirs, mock_feature_importance):
    """Teste le prétraitement avec des données insuffisantes."""
    pipeline = NeuralPipeline(config_path=tmp_dirs["config_path"], training_mode=True)
    raw_data = pd.DataFrame(
        {"timestamp": pd.date_range("2025-05-14 09:00", periods=10, freq="1min")}
    )
    options_data = pd.DataFrame(
        {"timestamp": pd.date_range("2025-05-14 09:00", periods=10, freq="1min")}
    )
    orderflow_data = pd.DataFrame(
        {"timestamp": pd.date_range("2025-05-14 09:00", periods=10, freq="1min")}
    )
    with pytest.raises(ValueError, match="Données insuffisantes"):
        pipeline.preprocess(raw_data, options_data, orderflow_data)
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "Données insuffisantes" in str(e) for e in df["error"].dropna()
    ), "Erreur non loguée"


def test_predict_vix_valid(tmp_dirs, mock_data, mock_feature_importance):
    """Teste la prédiction VIX avec des données valides."""
    pipeline = NeuralPipeline(config_path=tmp_dirs["config_path"], training_mode=True)
    _, _, full_data = mock_data
    with patch(
        "tensorflow.keras.models.Sequential.predict", return_value=np.array([[20.0]])
    ) as mock_predict:
        vix_value = pipeline.predict_vix(full_data)
        assert isinstance(vix_value, float), "VIX prédit n’est pas un flottant"
        assert vix_value == 20.0, "VIX prédit incorrect"
        assert mock_predict.called, "Méthode predict non appelée"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_predict_vix" in f and f.endswith(".json") for f in snapshot_files
    ), "Snapshot non créé"
    with open(os.path.join(tmp_dirs["snapshots_dir"], snapshot_files[-1]), "r") as f:
        snapshot = json.load(f)
    assert "new_features" in snapshot["data"], "new_features absent du snapshot"
    assert set(snapshot["data"]["new_features"]) == {
        "iv_skew",
        "iv_term_structure",
        "option_skew",
        "news_impact_score",
    }, "Nouvelles features incorrectes"
    assert any(
        f.startswith("vix_") and f.endswith(".png")
        for f in os.listdir(tmp_dirs["figures_dir"])
    ), "Visualisation non générée"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "confidence_drop_rate" in str(kw) for kw in df.to_dict("records")
    ), "confidence_drop_rate absent"
    assert any(
        "memory_usage_mb" in str(kw) for kw in df.to_dict("records")
    ), "memory_usage_mb absent"


def test_pretrain_models_valid(tmp_dirs, mock_data, mock_feature_importance):
    """Teste le pré-entraînement des modèles avec des données valides."""
    pipeline = NeuralPipeline(config_path=tmp_dirs["config_path"], training_mode=True)
    raw_data, options_data, orderflow_data = mock_data
    lstm_input, cnn_input, full_data = pipeline.preprocess(
        raw_data, options_data, orderflow_data
    )
    with patch("tensorflow.keras.models.Sequential.fit") as mock_fit, patch(
        "tensorflow.keras.models.Sequential.predict",
        return_value=np.zeros((len(full_data), pipeline.num_lstm_features)),
    ) as mock_predict:
        pipeline.pretrain_models(lstm_input, cnn_input, full_data)
        assert mock_fit.called, "Méthode fit non appelée"
        assert mock_predict.called, "Méthode predict non appelée"
    assert os.path.exists(
        os.path.join(tmp_dirs["models_dir"], "lstm_model.h5")
    ), "Modèle LSTM non sauvegardé"
    assert os.path.exists(
        os.path.join(tmp_dirs["models_dir"], "cnn_model.h5")
    ), "Modèle CNN non sauvegardé"
    assert os.path.exists(
        os.path.join(tmp_dirs["models_dir"], "vol_mlp_model.h5")
    ), "Modèle MLP volatilité non sauvegardé"
    assert os.path.exists(
        os.path.join(tmp_dirs["models_dir"], "regime_mlp_model.h5")
    ), "Modèle MLP régime non sauvegardé"
    assert os.path.exists(
        os.path.join(tmp_dirs["models_dir"], "lstm_vix_model.h5")
    ), "Modèle LSTM VIX non sauvegardé"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_pretrain_models" in f and f.endswith(".json") for f in snapshot_files
    ), "Snapshot non créé"
    with open(os.path.join(tmp_dirs["snapshots_dir"], snapshot_files[-1]), "r") as f:
        snapshot = json.load(f)
    assert "new_features" in snapshot["data"], "new_features absent du snapshot"
    assert os.path.exists(tmp_dirs["dashboard_path"]), "Fichier dashboard non créé"
    with open(tmp_dirs["dashboard_path"], "r") as f:
        dashboard = json.load(f)
    assert "new_features" in dashboard, "new_features absent du dashboard"
    assert (
        "confidence_drop_rate" in dashboard
    ), "confidence_drop_rate absent du dashboard"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "confidence_drop_rate" in str(kw) for kw in df.to_dict("records")
    ), "confidence_drop_rate absent"
    assert any(
        "memory_usage_mb" in str(kw) for kw in df.to_dict("records")
    ), "memory_usage_mb absent"


def test_run_valid(tmp_dirs, mock_data, mock_feature_importance):
    """Teste l’exécution du pipeline en mode batch avec des données valides."""
    pipeline = NeuralPipeline(config_path=tmp_dirs["config_path"], training_mode=True)
    raw_data, options_data, orderflow_data = mock_data
    with patch(
        "tensorflow.keras.models.Sequential.predict",
        side_effect=[
            np.zeros((51, pipeline.num_lstm_features)),  # lstm
            np.zeros((51, 1)),  # cnn
            np.zeros((50, 1)),  # vol_mlp
            np.zeros((50, 3)),  # regime_mlp
            np.array([[20.0]]),  # lstm_vix
        ],
    ) as mock_predict:
        result = pipeline.run(raw_data, options_data, orderflow_data)
        assert isinstance(result, dict), "Résultat n’est pas un dictionnaire"
        assert all(
            k in result for k in ["features", "volatility", "regime", "predicted_vix"]
        ), "Clés manquantes dans le résultat"
        assert isinstance(
            result["features"], np.ndarray
        ), "features n’est pas un np.ndarray"
        assert isinstance(
            result["volatility"], np.ndarray
        ), "volatility n’est pas un np.ndarray"
        assert isinstance(
            result["regime"], np.ndarray
        ), "regime n’est pas un np.ndarray"
        assert isinstance(
            result["predicted_vix"], float
        ), "predicted_vix n’est pas un flottant"
        assert result["features"].shape[0] == 51, "Shape incorrect pour features"
        assert mock_predict.call_count >= 5, "Méthode predict non appelée suffisamment"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_run" in f and f.endswith(".json") for f in snapshot_files
    ), "Snapshot non créé"
    with open(os.path.join(tmp_dirs["snapshots_dir"], snapshot_files[-1]), "r") as f:
        snapshot = json.load(f)
    assert "new_features" in snapshot["data"], "new_features absent du snapshot"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "confidence_drop_rate" in str(kw) for kw in df.to_dict("records")
    ), "confidence_drop_rate absent"
    assert any(
        "memory_usage_mb" in str(kw) for kw in df.to_dict("records")
    ), "memory_usage_mb absent"


def test_run_incremental_valid(
    tmp_dirs, mock_data, mock_single_row_data, mock_feature_importance
):
    """Teste l’exécution incrémentale du pipeline."""
    pipeline = NeuralPipeline(config_path=tmp_dirs["config_path"], training_mode=True)
    raw_data, options_data, orderflow_data = mock_data
    raw_row, options_row, orderflow_row = mock_single_row_data
    buffer = raw_data.iloc[-50:].copy()
    with patch(
        "tensorflow.keras.models.Sequential.predict",
        side_effect=[
            np.zeros((1, pipeline.num_lstm_features)),  # lstm
            np.zeros((1, 1)),  # cnn
            np.zeros((1, 1)),  # vol_mlp
            np.zeros((1, 3)),  # regime_mlp
            np.array([[20.0]]),  # lstm_vix
        ],
    ) as mock_predict:
        result = pipeline.run_incremental(raw_row, buffer, options_row, orderflow_row)
        assert isinstance(result, dict), "Résultat n’est pas un dictionnaire"
        assert all(
            k in result for k in ["features", "volatility", "regime", "predicted_vix"]
        ), "Clés manquantes dans le résultat"
        assert isinstance(
            result["features"], np.ndarray
        ), "features n’est pas un np.ndarray"
        assert isinstance(
            result["volatility"], np.ndarray
        ), "volatility n’est pas un np.ndarray"
        assert isinstance(
            result["regime"], np.ndarray
        ), "regime n’est pas un np.ndarray"
        assert isinstance(
            result["predicted_vix"], float
        ), "predicted_vix n’est pas un flottant"
        assert mock_predict.call_count >= 5, "Méthode predict non appelée suffisamment"
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_run_incremental" in f and f.endswith(".json") for f in snapshot_files
    ), "Snapshot non créé"
    with open(os.path.join(tmp_dirs["snapshots_dir"], snapshot_files[-1]), "r") as f:
        snapshot = json.load(f)
    assert "new_features" in snapshot["data"], "new_features absent du snapshot"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "confidence_drop_rate" in str(kw) for kw in df.to_dict("records")
    ), "confidence_drop_rate absent"
    assert any(
        "memory_usage_mb" in str(kw) for kw in df.to_dict("records")
    ), "memory_usage_mb absent"


def test_validate_shap_features(tmp_dirs, mock_feature_importance):
    """Teste la validation des top 150 SHAP features et obs_t."""
    pipeline = NeuralPipeline(config_path=tmp_dirs["config_path"], training_mode=False)
    features = [
        "key_strikes_1",
        "net_gamma",
        "vol_trigger",
        "bid_ask_imbalance",
        "iv_skew",
    ]
    result = pipeline.validate_shap_features(features)
    assert result, "Validation SHAP échouée pour features valides"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_validate_shap_features" in f for f in snapshot_files
    ), "Snapshot non créé"
    with open(os.path.join(tmp_dirs["snapshots_dir"], snapshot_files[-1]), "r") as f:
        snapshot = json.load(f)
    assert "missing_obs_t" in snapshot["data"], "missing_obs_t absent du snapshot"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any("validate_obs_t" in df["operation"]), "Validation obs_t non loguée"


def test_save_snapshot_compressed(tmp_dirs):
    """Teste la sauvegarde d’un snapshot compressé avec gzip."""
    pipeline = NeuralPipeline(config_path=tmp_dirs["config_path"], training_mode=True)
    snapshot_data = {"test": "compressed_snapshot"}
    pipeline.save_snapshot("test_compressed", snapshot_data, compress=True)
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
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any("save_snapshot" in df["operation"]), "Sauvegarde snapshot non loguée"


def test_critical_alerts(tmp_dirs, mock_feature_importance):
    """Teste les alertes Telegram pour les erreurs critiques."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        pipeline = NeuralPipeline(
            config_path=tmp_dirs["config_path"], training_mode=True
        )
        raw_data = pd.DataFrame(
            {"timestamp": pd.date_range("2025-05-14 09:00", periods=10, freq="1min")}
        )
        options_data = pd.DataFrame(
            {"timestamp": pd.date_range("2025-05-14 09:00", periods=10, freq="1min")}
        )
        orderflow_data = pd.DataFrame(
            {"timestamp": pd.date_range("2025-05-14 09:00", periods=10, freq="1min")}
        )
        with pytest.raises(ValueError):
            pipeline.preprocess(raw_data, options_data, orderflow_data)
        mock_telegram.assert_called_with(pytest.any(str))
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "Données insuffisantes" in str(e) for e in df["error"].dropna()
        ), "Erreur critique non loguée"


def test_scaler_manager(tmp_dirs):
    """Teste ScalerManager pour le chargement, la sauvegarde, et la transformation des scalers."""
    pipeline = NeuralPipeline(config_path=tmp_dirs["config_path"], training_mode=True)
    data = np.random.rand(100, 16)
    scaler = pipeline.scaler_manager.load_or_create(
        "test", expected_features=16, data=data
    )
    assert scaler.n_features_in_ == 16, "Dimensions du scaler incorrectes"
    transformed = pipeline.scaler_manager.transform("test", data)
    assert transformed.shape == data.shape, "Transformation du scaler incorrecte"
    assert os.path.exists(
        os.path.join(tmp_dirs["models_dir"], "scaler_test.pkl")
    ), "Scaler non sauvegardé"
    with pytest.raises(ValueError, match="Scaler test2 non initialisé"):
        pipeline.scaler_manager.transform("test2", data)


def test_cache_lru(tmp_dirs, mock_data):
    """Teste la gestion du cache LRU avec OrderedDict."""
    pipeline = NeuralPipeline(config_path=tmp_dirs["config_path"], training_mode=True)
    pipeline.cache_hours = 0.01  # Réduire pour tester
    pipeline.max_cache_size = 2  # Limiter pour tester l’éviction
    raw_data, options_data, orderflow_data = mock_data
    with patch(
        "tensorflow.keras.models.Sequential.predict",
        side_effect=[
            np.zeros((51, pipeline.num_lstm_features)),
            np.zeros((51, 1)),
            np.zeros((50, 1)),
            np.zeros((50, 3)),
            np.array([[20.0]]),
        ],
    ):
        pipeline.run(raw_data, options_data, orderflow_data)  # Cache 1
        pipeline.run(
            raw_data.iloc[:99], options_data.iloc[:99], orderflow_data.iloc[:99]
        )  # Cache 2
        pipeline.run(
            raw_data.iloc[:98], options_data.iloc[:98], orderflow_data.iloc[:98]
        )  # Cache 3, éviction du plus ancien
    assert len(pipeline.cache) <= 2, "Cache dépasse max_cache_size"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any("run_cache_hit" in df["operation"]), "Cache hit non logué"


def test_no_obsolete_references(tmp_dirs):
    """Vérifie l'absence de références à dxFeed, 320/81 features."""
    with open(tmp_dirs["config_path"], "r") as f:
        content = f.read()
    assert "dxFeed" not in content, "Référence à dxFeed trouvée"
    assert "320 features" not in content, "Référence à 320 features trouvée"
    assert "81 features" not in content, "Référence à 81 features trouvée"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
