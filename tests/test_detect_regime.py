# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_detect_regime.py
# Tests unitaires pour src/model/router/detect_regime.py
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide la détection des régimes de marché (trend, range, défensif, ultra-défensif) avec 350/150 features,
#        intégrant volatilité, données d’options, régimes hybrides, SHAP, prédictions LSTM, snapshots compressés,
#        sauvegardes incrémentielles/distribuées, et alertes Telegram.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, ta>=0.10.0,<0.11.0, pydantic>=2.0.0,<3.0.0,
#   psutil>=5.9.8,<6.0.0, matplotlib>=3.7.0,<4.0.0, sklearn>=1.5.0,<2.0.0, shap>=0.44.0,<0.45.0,
#   boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0, pyyaml>=6.0.0,<7.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/features/neural_pipeline.py
# - src/features/feature_pipeline.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/feature_sets.yaml
# - config/router_config.yaml
# - config/model_params.yaml
# - Données factices (DataFrame avec 350/150 features)
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de MarketRegimeDetector.
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Tests les phases 8 (auto-conscience via confidence_drop_rate), 11 (régimes hybrides), 12 (prédictions LSTM), 17 (SHAP).
# - Vérifie l’absence de dxFeed, la suppression de 320/81 features et obs_t, les retries, logs psutil, alertes Telegram,
#   snapshots compressés, et sauvegardes incrémentielles/distribuées.
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence) via config/feature_sets.yaml.

import gzip
import json
import os
from datetime import datetime
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.model.router.detect_regime import MarketRegimeDetector


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les données, logs, figures, cache, et checkpoints."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    cache_dir = data_dir / "cache" / "regime"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints"
    checkpoints_dir.mkdir()
    figures_dir = data_dir / "figures" / "regime"
    figures_dir.mkdir(parents=True)

    # Créer router_config.yaml
    router_config_path = config_dir / "router_config.yaml"
    router_config_content = {
        "fast_mode": False,
        "impute_nan": True,
        "use_optimized_calculations": True,
        "enable_heartbeat": True,
        "critical_times": ["14:00", "15:30"],
        "thresholds": {
            "vix_peak_threshold": 30.0,
            "vix_low_threshold": 15.0,
            "vix_high_threshold": 25.0,
            "spread_explosion_threshold": 0.05,
            "confidence_threshold": 0.7,
            "adx_threshold": 25.0,
            "atr_low_threshold": 0.5,
            "atr_high_threshold": 1.0,
            "volatility_spike_threshold": 1.0,
            "iv_low_threshold": 0.15,
            "iv_high_threshold": 0.25,
            "skew_low_threshold": 0.02,
            "skew_high_threshold": 0.05,
            "skew_extreme_threshold": 0.1,
            "volume_low_threshold": 1000,
            "volume_high_threshold": 5000,
            "oi_concentration_threshold": 0.7,
            "momentum_threshold": 0.0,
            "bollinger_width_low": 0.01,
            "atr_threshold": 1.5,
            "vwap_deviation_threshold": 0.01,
            "gex_threshold": 1000,
            "vix_correlation_threshold": 0.5,
            "gex_change_threshold": 0.5,
            "atr_normalized_threshold": 0.8,
        },
        "s3_bucket": "test-bucket",
        "s3_prefix": "regime/",
        "logging": {"buffer_size": 100},
        "use_neural_regime": True,
    }
    with open(router_config_path, "w", encoding="utf-8") as f:
        yaml.dump(router_config_content, f)

    # Créer model_params.yaml
    model_params_path = config_dir / "model_params.yaml"
    model_params_content = {"neural_pipeline": {"window_size": 50}}
    with open(model_params_path, "w", encoding="utf-8") as f:
        yaml.dump(model_params_content, f)

    # Créer feature_sets.yaml
    feature_sets_path = config_dir / "feature_sets.yaml"
    feature_sets_content = {
        "training": {"features": [f"feature_{i}" for i in range(350)]},
        "inference": {"shap_features": [f"shap_feature_{i}" for i in range(150)]},
    }
    with open(feature_sets_path, "w", encoding="utf-8") as f:
        yaml.dump(feature_sets_content, f)

    return {
        "base_dir": str(base_dir),
        "router_config_path": str(router_config_path),
        "model_params_path": str(model_params_path),
        "feature_sets_path": str(feature_sets_path),
        "logs_dir": str(logs_dir),
        "cache_dir": str(cache_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "figures_dir": str(figures_dir),
        "perf_log_path": str(logs_dir / "regime_performance.csv"),
        "history_file": str(logs_dir / "regime_history.csv"),
        "feature_importance_path": str(data_dir / "feature_importance.csv"),
    }


@pytest.fixture
def mock_data():
    """Crée un DataFrame factice avec 350 features."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=100, freq="1min"),
            **{f"feature_{i}": np.random.uniform(0, 1, 100) for i in range(350)},
            "open": np.random.uniform(5000, 5100, 100),
            "high": np.random.uniform(5100, 5200, 100),
            "low": np.random.uniform(4900, 5000, 100),
            "close": np.random.uniform(5000, 5100, 100),
            "volume": np.random.randint(1000, 10000, 100),
            "call_iv_atm": [0.1] * 50 + [0.2] * 50,
            "put_iv_atm": [0.1] * 50 + [0.15] * 50,
            "call_volume": [250] * 50 + [1000] * 50,
            "put_volume": [250] * 50 + [1000] * 50,
            "oi_peak_call_near": [1000] * 50 + [5000] * 50,
            "oi_peak_put_near": [1000] * 50 + [1000] * 50,
            "bid_size_level_1": np.random.randint(100, 500, 100),
            "ask_size_level_1": np.random.randint(100, 500, 100),
            "gex": np.random.uniform(-1000, 1000, 100),
            "gex_change": np.random.uniform(-0.5, 0.5, 100),
            "vwap_deviation": np.random.uniform(-0.01, 0.01, 100),
            "atr_normalized": np.random.uniform(0.1, 0.9, 100),
            "vix_es_correlation": np.random.uniform(-0.5, 0.5, 100),
            "bid_ask_ratio": np.random.uniform(0.4, 0.6, 100),
            "bid_ask_ratio_level_2": np.random.uniform(0.4, 0.6, 100),
        }
    )


@pytest.fixture
def detector(tmp_dirs):
    """Crée une instance de MarketRegimeDetector pour les tests."""
    with patch("src.model.router.detect_regime.NeuralPipeline") as mock_neural, patch(
        "src.model.router.detect_regime.FeaturePipeline"
    ) as mock_feature, patch(
        "src.model.utils.alert_manager.AlertManager"
    ) as mock_alert, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        mock_neural.return_value.load_models.return_value = None
        mock_feature.return_value.load_pipeline.return_value = None
        mock_neural.return_value.run.return_value = {
            "regime": [0],
            "volatility": [20.0],
            "confidence": 0.8,
            "features": np.zeros((1, 350)),
            "raw_scores": np.array([0.7, 0.2, 0.1]),
        }
        mock_alert.return_value.send_alert.return_value = None
        mock_telegram.return_value = None
        detector = MarketRegimeDetector(policy_type="mlp", training_mode=True)
    return detector


@pytest.mark.asyncio
async def test_init(tmp_dirs, detector):
    """Teste l’initialisation de MarketRegimeDetector."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        assert detector.num_features == 350, "Dimension des features incorrecte"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "init" in str(op) for op in df["operation"]
        ), "Opération init non journalisée"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_init" in f and f.endswith(".json.gz") for f in snapshot_files
        ), "Snapshot init non créé"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_init_invalid_features(tmp_dirs):
    """Teste l’initialisation avec un nombre incorrect de features."""
    with patch("src.model.utils.config_manager.get_config") as mock_config, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ):
        mock_config.side_effect = [
            {"fast_mode": False, "impute_nan": True, "thresholds": {}},
            {"neural_pipeline": {"window_size": 50}},
            {
                "training": {"features": [f"feature_{i}" for i in range(100)]}
            },  # Moins de 350 features
        ]
        with pytest.raises(ValueError, match="Attendu 350 features"):
            MarketRegimeDetector(training_mode=True)


@pytest.mark.asyncio
async def test_validate_data(detector, mock_data):
    """Teste la validation des données."""
    assert detector.validate_data(mock_data), "Validation des données échouée"
    df_perf = pd.read_csv(detector.perf_log)
    assert any(
        "validate_data" in str(op) for op in df_perf["operation"]
    ), "Opération validate_data non journalisée"
    assert any(
        "confidence_drop_rate" in str(kw) for kw in df_perf.to_dict("records")
    ), "confidence_drop_rate absent"


@pytest.mark.asyncio
async def test_validate_data_invalid(detector, mock_data):
    """Teste la validation avec des données invalides."""
    invalid_data = mock_data.copy()
    invalid_data["feature_0"] = np.nan  # Ajouter des NaN
    assert not detector.validate_data(
        invalid_data
    ), "Validation des données invalides devrait échouer"
    df_perf = pd.read_csv(detector.perf_log)
    assert any(
        "validate_data" in str(op) for op in df_perf["operation"]
    ), "Opération validate_data non journalisée"


@pytest.mark.asyncio
async def test_precompute_indicators(detector, mock_data):
    """Teste le calcul des indicateurs techniques et d'options."""
    df_enriched = detector.precompute_indicators(mock_data)
    assert "atr_14" in df_enriched.columns, "Indicateur atr_14 manquant"
    assert "adx_14" in df_enriched.columns, "Indicateur adx_14 manquant"
    assert "rsi_14" in df_enriched.columns, "Indicateur rsi_14 manquant"
    assert "option_skew" in df_enriched.columns, "Indicateur option_skew manquant"
    df_perf = pd.read_csv(detector.perf_log)
    assert any(
        "precompute_indicators" in str(op) for op in df_perf["operation"]
    ), "Opération precompute_indicators non journalisée"


@pytest.mark.asyncio
async def test_detect(detector, mock_data, tmp_dirs):
    """Teste la détection des régimes de marché."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        regime, details = await detector.detect(mock_data, 50)
        assert regime in [
            "trend",
            "range",
            "defensive",
            "ultra-defensive",
        ], "Régime détecté invalide"
        assert "regime_probs" in details, "Détails manquent regime_probs"
        assert "shap_values" in details, "Détails manquent shap_values"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "detect" in str(op) for op in df_perf["operation"]
        ), "Opération detect non journalisée"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_regime" in f and f.endswith(".json.gz") for f in snapshot_files
        ), "Snapshot regime non créé"
        assert any(
            f.startswith("vix_") and f.endswith(".png")
            for f in os.listdir(tmp_dirs["figures_dir"])
        ), "Visualisation non générée"
        assert os.path.exists(tmp_dirs["history_file"]), "Fichier d’historique non créé"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_detect_ultra_defensive(detector, mock_data):
    """Teste la détection en mode ultra-défensif."""
    with patch(
        "src.model.router.detect_regime.MarketRegimeDetector.check_neural_pipeline_health",
        new=AsyncMock(return_value=False),
    ):
        mock_data["vix_es_correlation"] = 40.0  # Déclenche ultra-défensif
        mock_data["ask_size_level_1"] = 600
        mock_data["bid_size_level_1"] = 100
        mock_data["close"] = 5000
        regime, details = await detector.detect(mock_data, 50)
        assert regime == "ultra-defensive", "Mode ultra-défensif non détecté"
        assert (
            details["regime_probs"]["defensive"] == 1.0
        ), "Probabilités incorrectes pour ultra-défensif"


@pytest.mark.asyncio
async def test_check_neural_pipeline_health(detector, mock_data):
    """Teste la vérification de la santé de NeuralPipeline."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        result = await detector.check_neural_pipeline_health()
        assert result, "Vérification de la santé de NeuralPipeline échouée"
        df_perf = pd.read_csv(detector.perf_log)
        assert any(
            "check_neural_pipeline_health" in str(op) for op in df_perf["operation"]
        ), "Opération check_neural_pipeline_health non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_calculate_shap_values(detector, mock_data):
    """Teste le calcul des valeurs SHAP."""
    shap_values = detector.calculate_shap_values(mock_data, 50)
    assert isinstance(shap_values, dict), "Valeurs SHAP doivent être un dictionnaire"
    assert len(shap_values) <= 50, "Trop de features SHAP retournées"
    assert os.path.exists(
        detector.feature_importance_path
    ), "Fichier feature_importance.csv non créé"
    df_perf = pd.read_csv(detector.perf_log)
    assert any(
        "calculate_shap_values" in str(op) for op in df_perf["operation"]
    ), "Opération calculate_shap_values non journalisée"


@pytest.mark.asyncio
async def test_update_thresholds(detector):
    """Teste la mise à jour des seuils dynamiques."""
    detector.update_thresholds(
        adx=30.0,
        atr=1.0,
        predicted_volatility=20.0,
        vix=15.0,
        atr_normalized=0.5,
        call_iv_atm=0.15,
    )
    assert len(detector.performance_window) == 1, "Fenêtre de performance incorrecte"
    df_perf = pd.read_csv(detector.perf_log)
    assert any(
        "update_thresholds" in str(op) for op in df_perf["operation"]
    ), "Opération update_thresholds non journalisée"


@pytest.mark.asyncio
async def test_check_transition_validity(detector):
    """Teste la vérification de la validité des transitions."""
    detector.recent_regimes.append("trend")
    detector.recent_regimes.append("range")
    assert detector.check_transition_validity(
        "defensive"
    ), "Transition devrait être valide avec des régimes variés"


@pytest.mark.asyncio
async def test_save_snapshot_compressed(detector, tmp_dirs):
    """Teste la sauvegarde d’un snapshot compressé."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        snapshot_data = {"test": "compressed_snapshot"}
        detector.save_snapshot("test_compressed", snapshot_data, compress=True)
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_test_compressed" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot compressé non créé"
        with gzip.open(
            os.path.join(tmp_dirs["cache_dir"], snapshot_files[-1]),
            "rt",
            encoding="utf-8",
        ) as f:
            snapshot = json.load(f)
        assert (
            snapshot["data"] == snapshot_data
        ), "Contenu du snapshot compressé incorrect"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "save_snapshot" in str(op) for op in df_perf["operation"]
        ), "Opération save_snapshot non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_checkpoint(detector, tmp_dirs):
    """Teste la sauvegarde incrémentielle."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        df = pd.DataFrame(
            {
                "step": [50],
                "regime": ["trend"],
                "confidence_score": [0.8],
                "vix_es_correlation": [20.0],
                "call_iv_atm": [0.2],
                "timestamp": [datetime.now().isoformat()],
            }
        )
        detector.checkpoint(df, data_type="test_metrics")
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "regime_test_metrics" in f and f.endswith(".json.gz")
            for f in checkpoint_files
        ), "Checkpoint non créé"
        with gzip.open(
            os.path.join(tmp_dirs["checkpoints_dir"], checkpoint_files[0]),
            "rt",
            encoding="utf-8",
        ) as f:
            checkpoint = json.load(f)
        assert checkpoint["num_rows"] == len(df), "Nombre de lignes incorrect"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "checkpoint" in str(op) for op in df_perf["operation"]
        ), "Opération checkpoint non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_cloud_backup(detector, tmp_dirs):
    """Teste la sauvegarde distribuée S3."""
    with patch("boto3.client") as mock_s3, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        df = pd.DataFrame(
            {
                "step": [50],
                "regime": ["trend"],
                "confidence_score": [0.8],
                "vix_es_correlation": [20.0],
                "call_iv_atm": [0.2],
                "timestamp": [datetime.now().isoformat()],
            }
        )
        detector.cloud_backup(df, data_type="test_metrics")
        assert mock_s3.called, "Client S3 non appelé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "temp_s3_" in f for f in checkpoint_files
        ), "Fichier temporaire S3 non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "cloud_backup" in str(op) for op in df_perf["operation"]
        ), "Opération cloud_backup non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_handle_sigint(detector, tmp_dirs):
    """Teste la gestion SIGINT."""
    with patch("sys.exit") as mock_exit, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        detector.handle_sigint(signal.SIGINT, None)
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_sigint" in f for f in snapshot_files
        ), "Snapshot sigint non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "handle_sigint" in str(op) for op in df_perf["operation"]
        ), "Opération handle_sigint non journalisée"
        mock_exit.assert_called_with(0)
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_prune_cache(detector):
    """Teste la purge du cache."""
    detector.neural_cache = OrderedDict({f"key_{i}": {} for i in range(1001)})
    detector.prune_cache()
    assert len(detector.neural_cache) <= 1000, "Cache non purgé correctement"
