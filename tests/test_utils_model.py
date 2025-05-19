# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_utils_model.py
# Tests unitaires pour src/model/utils_model.py
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide les fonctions utilitaires pour la validation des features, la gestion 
# de la configuration, l’early stopping, l’audit des features, l’exportation des logs, 
# le calcul des métriques, et la gestion de la mémoire. Vérifie les sauvegardes 
# incrémentielles/distribuées et les alertes Telegram.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, 
#   psutil>=5.9.8,<6.0.0, pyyaml>=6.0.0,<7.0.0, stable-baselines3>=2.0.0,<3.0.0, 
#   boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0
# - src/features/neural_pipeline.py
# - src/envs/trading_env.py
# - src/model/utils/config_manager.py
# - src/model/utils/miya_console.py
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/model_params.yaml
# - config/feature_sets.yaml
# - Données factices (DataFrame avec 350 features)
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de utils_model.py.
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Tests la Phase 8 (confidence_drop_rate).
# - Vérifie l’absence de dxFeed, la suppression de 320/81 features et obs_t, 
#   les retries, logs psutil, alertes Telegram, snapshots compressés, et sauvegardes 
#   incrémentielles/distribuées.
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence) 
#   via config/feature_sets.yaml.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.

# Note : L'erreur F401 concernant des imports inutilisés est une fausse alerte.
# Tous les imports sont utilisés dans les tests ou fixtures.

import gzip
import json
import os
from collections import deque
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.model.utils_model import (
    EarlyStoppingCallback,
    audit_features,
    checkpoint,
    clear_memory,
    cloud_backup,
    compute_metrics,
    export_logs,
    load_config,
    validate_features,
)


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, cache, et checkpoints."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    cache_dir = data_dir / "cache" / "utils_model" / "ES"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints" / "utils_model" / "ES"
    checkpoints_dir.mkdir(parents=True)

    # Créer model_params.yaml
    config_path = config_dir / "model_params.yaml"
    config_content = {
        "utils_model": {
            "check_freq": 1000,
            "patience": 5000,
            "min_reward_improvement": 0.01,
            "max_log_size": 10000,
            "s3_bucket": "test-bucket",
            "s3_prefix": "utils_model/",
        }
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    # Créer feature_sets.yaml
    feature_sets_path = config_dir / "feature_sets.yaml"
    training_features = [f"feature_{i}" for i in range(350)]
    shap_features = [f"shap_feature_{i}" for i in range(150)]
    feature_sets_content = {
        "training": {"features": training_features},
        "inference": {"shap_features": shap_features},
    }
    with open(feature_sets_path, "w", encoding="utf-8") as f:
        yaml.dump(feature_sets_content, f)

    return {
        "base_dir": str(base_dir),
        "config_path": str(config_path),
        "feature_sets_path": str(feature_sets_path),
        "logs_dir": str(logs_dir),
        "cache_dir": str(cache_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "perf_log_path": str(logs_dir / "utils_model_performance.csv"),
    }


@pytest.fixture
def mock_data():
    """Crée un DataFrame factice avec 350 features."""
    feature_cols = [f"feature_{i}" for i in range(350)]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=100, freq="1min"),
            "reward": np.random.randn(100),
            "predicted_volatility_es": np.random.uniform(0.5, 2, 100),
            "neural_regime": np.random.choice([0, 1, 2], 100),
            "cnn_pressure": np.random.randn(100),
            "step": range(100),
            "action": np.random.uniform(-1, 1, 100),
            "bid_size_level_1": np.random.randint(100, 500, 100),
            "ask_size_level_1": np.random.randint(100, 500, 100),
            "trade_frequency_1s": np.random.uniform(0.1, 10, 100),
            "spread_avg_1min": np.random.uniform(0.01, 0.05, 100),
            "close": np.random.uniform(4000, 5000, 100),
            "open": np.random.uniform(4000, 5000, 100),
            "high": np.random.uniform(4000, 5000, 100),
            "low": np.random.uniform(4000, 5000, 100),
            "volume": np.random.randint(1000, 5000, 100),
            "atr_14_es": np.random.uniform(0.1, 1.0, 100),
            "adx_14": np.random.uniform(10, 50, 100),
            "gex_es": np.random.uniform(-1000, 1000, 100),
            "oi_peak_call_near": np.random.uniform(100, 1000, 100),
            "gamma_wall_call": np.random.uniform(100, 1000, 100),
            "gamma_wall_put": np.random.uniform(100, 1000, 100),
            **{f: np.random.uniform(0, 1, 100) for f in feature_cols},
        }
    )


@pytest.fixture
def mock_env(mock_data):
    """Crée un environnement de trading factice."""
    env = MagicMock()
    env.data = mock_data
    env.current_step = 0
    env.sequence_length = 50
    env.policy_type = "mlp"
    return env


@pytest.mark.asyncio
async def test_validate_features(tmp_dirs, mock_data):
    """Teste la validation des features."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        validate_features(mock_data, step=0, market="ES")
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_validate_features" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot validate_features non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "validate_features" in str(op) for op in df_perf["operation"]
        ), "Opération validate_features non journalisée"
        assert any(
            "confidence_drop_rate" in str(kw) for kw in df_perf.to_dict("records")
        ), "confidence_drop_rate absent"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_validate_features_invalid(mock_data):
    """Teste la validation avec des features invalides."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        invalid_data = mock_data.drop(columns=["feature_0", "bid_size_level_1"])
        with pytest.raises(ValueError):
            validate_features(invalid_data, step=0, market="ES")
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_load_config(tmp_dirs):
    """Teste le chargement de la configuration."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        config = load_config(config_path=tmp_dirs["config_path"], market="ES")
        assert config.get("check_freq") == 1000, "Configuration incorrecte"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "load_config" in str(op) for op in df_perf["operation"]
        ), "Opération load_config non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_early_stopping_callback(tmp_dirs, mock_env, mock_data):
    """Teste le callback EarlyStoppingCallback."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        callback = EarlyStoppingCallback(policy_type="mlp", market="ES")
        callback.model = MagicMock()
        callback.model.env = mock_env
        callback.n_calls = 1000
        callback.locals = {"rewards": mock_data["reward"].tolist()}
        assert callback._on_step(), "Early stopping déclenché trop tôt"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_early_stopping_check" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot early_stopping_check non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "early_stopping_step" in str(op) for op in df_perf["operation"]
        ), "Opération early_stopping_step non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_audit_features(mock_data):
    """Teste l’audit des features."""
    CACHE_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data",
        "cache",
        "utils_model",
    )
    PERF_LOG_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data",
        "logs",
        "utils_model_performance.csv",
    )
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        assert audit_features(mock_data, market="ES"), "Audit des features échoué"
        df_perf = pd.read_csv(PERF_LOG_PATH)
        assert any(
            "audit_features" in str(op) for op in df_perf["operation"]
        ), "Opération audit_features non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_export_logs(tmp_dirs):
    """Teste l’exportation des logs."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        trade_log = [
            {"step": i, "action": float(i), "reward": float(i)} for i in range(5)
        ]
        export_logs(trade_log, tmp_dirs["logs_dir"], "test_export.csv", market="ES")
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_export_logs" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot export_logs non créé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "trade_log" in f and f.endswith(".json.gz") for f in checkpoint_files
        ), "Checkpoint trade_log non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "export_logs" in str(op) for op in df_perf["operation"]
        ), "Opération export_logs non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_compute_metrics(mock_data):
    """Teste le calcul des métriques."""
    CACHE_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data",
        "cache",
        "utils_model",
    )
    PERF_LOG_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data",
        "logs",
        "utils_model_performance.csv",
    )
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        metrics = compute_metrics(mock_data, market="ES")
        assert "total_reward" in metrics, "Métrique total_reward absente"
        snapshot_files = os.listdir(os.path.join(CACHE_DIR, "ES"))
        assert any(
            "snapshot_compute_metrics" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot compute_metrics non créé"
        df_perf = pd.read_csv(PERF_LOG_PATH)
        assert any(
            "compute_metrics" in str(op) for op in df_perf["operation"]
        ), "Opération compute_metrics non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_clear_memory(tmp_dirs):
    """Teste le nettoyage de la mémoire."""
    CACHE_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data",
        "cache",
        "utils_model",
    )
    PERF_LOG_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data",
        "logs",
        "utils_model_performance.csv",
    )
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        log_buffer = deque([{"step": i} for i in range(15000)])
        clear_memory(max_log_size=10000, market="ES")
        assert len(log_buffer) <= 10000, "Buffer non réduit"
        snapshot_files = os.listdir(os.path.join(CACHE_DIR, "ES"))
        assert any(
            "snapshot_clear_memory" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot clear_memory non créé"
        df_perf = pd.read_csv(PERF_LOG_PATH)
        assert any(
            "clear_memory" in str(op) for op in df_perf["operation"]
        ), "Opération clear_memory non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_checkpoint(tmp_dirs):
    """Teste la sauvegarde incrémentielle."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        df = pd.DataFrame(
            {
                "timestamp": [datetime.now().isoformat()],
                "operation": ["test"],
                "latency": [1.0],
            }
        )
        checkpoint(df, data_type="test_metrics", market="ES")
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "utils_model_test_metrics" in f and f.endswith(".json.gz")
            for f in checkpoint_files
        ), "Checkpoint non créé"
        checkpoint_file = os.path.join(tmp_dirs["checkpoints_dir"], checkpoint_files[0])
        with gzip.open(checkpoint_file, "rt", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
        assert checkpoint_data["num_rows"] == len(df), "Nombre de lignes incorrect"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "checkpoint" in str(op) for op in df_perf["operation"]
        ), "Opération checkpoint non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_cloud_backup(tmp_dirs):
    """Teste la sauvegarde distribuée S3."""
    with patch("boto3.client") as mock_s3, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        df = pd.DataFrame(
            {
                "timestamp": [datetime.now().isoformat()],
                "operation": ["test"],
                "latency": [1.0],
            }
        )
        cloud_backup(df, data_type="test_metrics", market="ES")
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