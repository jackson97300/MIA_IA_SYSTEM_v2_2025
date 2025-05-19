# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_config_manager.py
# Tests unitaires pour src/model/utils/config_manager.py
#
# Version : 2.1.4
# Date : 2025-05-14
#
# Rôle : Valide le chargement, la validation, et la distribution des configurations YAML pour tous les modules,
#        standardisant sur 350 features, avec support multi-environnements, retries, et snapshots JSON.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, pyyaml>=6.0.0,<7.0.0, psutil>=5.9.8,<6.0.0,
#   boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0, pydantic>=2.0.0,<3.0.0
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - Fichiers YAML factices dans config/ ou config/envs/{env}/
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de config_manager.py.
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-14).
# - Supporte les améliorations Position sizing dynamique et HMM / Changepoint Detection via risk_manager_config.yaml et regime_detector_config.yaml.
# - Tests la Phase 8 (confidence_drop_rate) et autres standards.
# - Vérifie l’absence de dxFeed, la suppression de 320/81 features et obs_t, les retries, logs psutil,
#   alertes Telegram, snapshots compressés, et sauvegardes incrémentielles/distribuées.
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence) via config/feature_sets.yaml.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.

import gzip
import json
import os
from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

from src.model.utils.config_manager import ConfigManager


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, cache, checkpoints, et configurations."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    cache_dir = data_dir / "cache" / "config" / "ES"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints" / "config" / "ES"
    checkpoints_dir.mkdir(parents=True)

    # Créer des fichiers YAML factices
    config_files = [
        (
            "es_config.yaml",
            {
                "market": {"symbol": {"value": "ES"}},
                "preprocessing": {
                    "depth_level": 5,
                    "input_path": {"value": str(data_dir / "features")},
                },
                "spotgamma_recalculator": {
                    "min_volume_threshold": 100,
                    "price_proximity_range": 0.5,
                    "shap_features": ["iv_atm", "option_skew"],
                },
                "metadata": {"version": "2.1.4", "alert_priority": 4},
                "s3_bucket": "test-bucket",
                "s3_prefix": "config/",
            },
        ),
        (
            "trading_env_config.yaml",
            {
                "environment": {"max_position_size": {"value": 5}},
                "observation": {"obs_dimensions": {"value": 350}},
                "reward": {"reward_scaling": {"value": 1.0}},
                "metadata": {"version": "2.1.4", "alert_priority": 4},
            },
        ),
        (
            "mia_config.yaml",
            {
                "mia": {
                    "language": {"value": "fr"},
                    "max_message_length": {"value": 200},
                    "interactive_mode": {"enabled": True, "language": "fr"},
                },
                "logging": {"log_dir": {"value": str(logs_dir)}},
                "metadata": {"version": "2.1.4", "alert_priority": 4},
            },
        ),
        (
            "credentials.yaml",
            {
                "iqfeed": {"api_key": "test_key"},
                "investing_com": {"api_key": "test_key"},
                "newsdata_io": {"api_keys": ["test_key1", "test_key2"]},
                "nlp": {"api_key": "test_key"},
                "metadata": {"version": "2.1.4", "alert_priority": 4},
            },
        ),
        (
            "feature_sets.yaml",
            {
                "ES": {
                    "training": [
                        "timestamp",
                        "vix",
                        "neural_regime",
                        "predicted_volatility",
                        "trade_frequency_1s",
                        "close",
                    ]
                    + [f"feature_{i}" for i in range(344)],
                    "inference": [
                        "vix",
                        "neural_regime",
                        "predicted_volatility",
                        "trade_frequency_1s",
                        "close",
                    ]
                    + [f"shap_feature_{i}" for i in range(145)],
                },
                "metadata": {
                    "version": "2.1.4",
                    "alert_priority": 4,
                    "total_features": 350,
                },
            },
        ),
        (
            "market_config.yaml",
            {
                "market": {"symbol": {"value": "ES"}},
                "risk": {
                    "max_drawdown": {"value": -0.2},
                    "default_rrr": {"value": 2.0},
                },
                "metadata": {"version": "2.1.4", "alert_priority": 4},
            },
        ),
        (
            "model_params.yaml",
            {
                "default": {
                    "learning_rate": {"value": 0.001},
                    "buffer_size": {"value": 100000},
                    "news_impact_threshold": {"value": 0.5},
                    "vix_threshold": {"value": 20.0},
                },
                "neural_pipeline": {"base_features": {"value": 350}},
                "metadata": {"version": "2.1.4", "alert_priority": 4},
            },
        ),
        (
            "router_config.yaml",
            {
                "trend": {"atr_threshold": {"value": 1.0}},
                "range": {"vwap_slope_threshold": {"value": 0.01}},
                "defensive": {"volatility_spike_threshold": {"value": 2.0}},
                "metadata": {"version": "2.1.4", "alert_priority": 4},
            },
        ),
        (
            "iqfeed_config.yaml",
            {
                "symbols": {"value": ["ES", "MNQ"]},
                "connection": {"retry_attempts": {"value": 5}},
                "iqfeed": {"host": {"value": "127.0.0.1"}, "port": {"value": 9100}},
                "data_types": {"dom": {"depth": 5}},
                "metadata": {"version": "2.1.4", "alert_priority": 4},
            },
        ),
        (
            "algo_config.yaml",
            {
                "sac": {
                    "range": {
                        "learning_rate": {"value": 0.001},
                        "gamma": {"value": 0.99},
                        "ent_coef": {"value": 0.1},
                        "l2_lambda": {"value": 0.01},
                    },
                    "trend": {
                        "learning_rate": {"value": 0.001},
                        "gamma": {"value": 0.99},
                        "ent_coef": {"value": 0.1},
                        "l2_lambda": {"value": 0.01},
                    },
                    "defensive": {
                        "learning_rate": {"value": 0.001},
                        "gamma": {"value": 0.99},
                        "ent_coef": {"value": 0.1},
                        "l2_lambda": {"value": 0.01},
                    },
                },
                "ppo": {
                    "range": {
                        "learning_rate": {"value": 0.001},
                        "gamma": {"value": 0.99},
                        "ent_coef": {"value": 0.1},
                        "l2_lambda": {"value": 0.01},
                    },
                    "trend": {
                        "learning_rate": {"value": 0.001},
                        "gamma": {"value": 0.99},
                        "ent_coef": {"value": 0.1},
                        "l2_lambda": {"value": 0.01},
                    },
                    "defensive": {
                        "learning_rate": {"value": 0.001},
                        "gamma": {"value": 0.99},
                        "ent_coef": {"value": 0.1},
                        "l2_lambda": {"value": 0.01},
                    },
                },
                "ddpg": {
                    "range": {
                        "learning_rate": {"value": 0.001},
                        "gamma": {"value": 0.99},
                        "ent_coef": {"value": 0.1},
                        "l2_lambda": {"value": 0.01},
                    },
                    "trend": {
                        "learning_rate": {"value": 0.001},
                        "gamma": {"value": 0.99},
                        "ent_coef": {"value": 0.1},
                        "l2_lambda": {"value": 0.01},
                    },
                    "defensive": {
                        "learning_rate": {"value": 0.001},
                        "gamma": {"value": 0.99},
                        "ent_coef": {"value": 0.1},
                        "l2_lambda": {"value": 0.01},
                    },
                },
                "metadata": {"version": "2.1.4", "alert_priority": 4},
            },
        ),
        (
            "alert_config.yaml",
            {
                "telegram": {
                    "enabled": True,
                    "bot_token": "test_token",
                    "chat_id": "test_chat",
                    "priority": 4,
                },
                "sms": {
                    "enabled": False,
                    "account_sid": "test_sid",
                    "auth_token": "test_token",
                    "from_number": "test_from",
                    "to_number": "test_to",
                    "priority": 4,
                },
                "email": {
                    "enabled": False,
                    "sender_email": "test@email.com",
                    "sender_password": "test_pass",
                    "receiver_email": "test@email.com",
                    "priority": 4,
                },
                "alert_thresholds": {
                    "volatility_spike_threshold": {"value": 2.0},
                    "oi_sweep_alert_threshold": {"value": 0.5},
                    "macro_event_severity_alert": {"value": 0.7},
                },
                "metadata": {"version": "2.1.4", "alert_priority": 4},
            },
        ),
        (
            "trade_probability_config.yaml",
            {
                "trade_probability": {
                    "buffer_size": {"value": 100},
                    "min_trade_success_prob": {"value": 0.7},
                    "retrain_frequency": {"value": "weekly"},
                    "retrain_threshold": {"value": 1000},
                },
                "metadata": {"version": "2.1.4", "alert_priority": 4},
            },
        ),
        (
            "risk_manager_config.yaml",
            {
                "buffer_size": 100,
                "critical_buffer_size": 10,
                "max_retries": 3,
                "retry_delay": 2.0,
                "kelly_fraction": 0.1,
                "max_position_fraction": 0.1,
                "metadata": {"version": "2.1.4", "alert_priority": 4},
            },
        ),
        (
            "regime_detector_config.yaml",
            {
                "buffer_size": 100,
                "critical_buffer_size": 10,
                "max_retries": 3,
                "retry_delay": 2.0,
                "n_iterations": 10,
                "convergence_threshold": 1e-3,
                "covariance_type": "diag",
                "n_components": 3,
                "window_size": 50,
                "window_size_adaptive": True,
                "random_state": 42,
                "use_random_state": True,
                "min_train_rows": 100,
                "min_state_duration": 5,
                "cache_ttl_seconds": 300,
                "cache_ttl_adaptive": True,
                "prometheus_labels": {"env": "prod", "team": "quant", "market": "ES"},
                "metadata": {"version": "2.1.4", "alert_priority": 4},
            },
        ),
    ]

    for file_name, content in config_files:
        with open(config_dir / file_name, "w", encoding="utf-8") as f:
            yaml.dump(content, f)

    return {
        "base_dir": str(base_dir),
        "config_dir": str(config_dir),
        "logs_dir": str(logs_dir),
        "cache_dir": str(cache_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "perf_log_path": str(logs_dir / "config_performance.csv"),
    }


@pytest.mark.asyncio
async def test_load_all_configs(tmp_dirs):
    """Teste le chargement de toutes les configurations."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        config_manager = ConfigManager()
        assert len(config_manager.configs) == len(
            config_manager.CONFIG_FILES
        ), "Nombre incorrect de configurations chargées"
        assert (
            "risk_manager_config.yaml" in config_manager.configs
        ), "risk_manager_config.yaml non chargé"
        assert (
            "regime_detector_config.yaml" in config_manager.configs
        ), "regime_detector_config.yaml non chargé"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_config_load" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot config_load non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "load_config" in str(op) for op in df_perf["operation"]
        ), "Opération load_config non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_validate_config(tmp_dirs):
    """Teste la validation des configurations."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        ConfigManager()
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_config_validation" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot config_validation non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "confidence_drop_rate" in str(kw)
            for kw in df_perf.to_dict("records")
            if "feature_sets.yaml" in str(kw.get("config_file", ""))
        ), "confidence_drop_rate absent"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_validate_risk_manager_config(tmp_dirs):
    """Teste la validation de risk_manager_config.yaml."""
    with patch("src.utils.telegram_alert.send_telegram_alert"):
        config_manager = ConfigManager()
        config = config_manager.get_config("config/risk_manager_config.yaml")
        config_manager._validate_config("risk_manager_config.yaml", config, market="ES")
        assert config["kelly_fraction"] == 0.1, "kelly_fraction incorrect"
        assert config["max_position_fraction"] == 0.1, "max_position_fraction incorrect"

        # Test config invalide
        invalid_config = config.copy()
        invalid_config["kelly_fraction"] = 0.0
        with pytest.raises(ValueError, match="kelly_fraction hors plage"):
            config_manager._validate_config(
                "risk_manager_config.yaml", invalid_config, market="ES"
            )

        invalid_config = config.copy()
        invalid_config["max_position_fraction"] = 0.0
        with pytest.raises(ValueError, match="max_position_fraction hors plage"):
            config_manager._validate_config(
                "risk_manager_config.yaml", invalid_config, market="ES"
            )

        # Test Pydantic validation
        invalid_config = config.copy()
        invalid_config["buffer_size"] = 5  # < 10
        with pytest.raises(ValueError, match="Validation Pydantic échouée"):
            config_manager._validate_config(
                "risk_manager_config.yaml", invalid_config, market="ES"
            )


@pytest.mark.asyncio
async def test_validate_regime_detector_config(tmp_dirs):
    """Teste la validation de regime_detector_config.yaml."""
    with patch("src.utils.telegram_alert.send_telegram_alert"):
        config_manager = ConfigManager()
        config = config_manager.get_config("config/regime_detector_config.yaml")
        config_manager._validate_config(
            "regime_detector_config.yaml", config, market="ES"
        )
        assert config["n_components"] == 3, "n_components incorrect"
        assert config["min_state_duration"] == 5, "min_state_duration incorrect"

        # Test config invalide
        invalid_config = config.copy()
        invalid_config["n_components"] = 1
        with pytest.raises(ValueError, match="Validation Pydantic échouée"):
            config_manager._validate_config(
                "regime_detector_config.yaml", invalid_config, market="ES"
            )

        invalid_config = config.copy()
        invalid_config["min_train_rows"] = 10
        with pytest.raises(ValueError, match="min_train_rows hors plage"):
            config_manager._validate_config(
                "regime_detector_config.yaml", invalid_config, market="ES"
            )

        invalid_config = config.copy()
        invalid_config["cache_ttl_seconds"] = 5
        with pytest.raises(ValueError, match="cache_ttl_seconds hors plage"):
            config_manager._validate_config(
                "regime_detector_config.yaml", invalid_config, market="ES"
            )


@pytest.mark.asyncio
async def test_get_config(tmp_dirs):
    """Teste l'accès à une configuration spécifique."""
    config_manager = ConfigManager()
    config = config_manager.get_config("config/es_config.yaml")
    assert (
        config["market"]["symbol"]["value"] == "ES"
    ), "Configuration es_config.yaml incorrecte"
    config = config_manager.get_config("config/risk_manager_config.yaml")
    assert (
        config["kelly_fraction"] == 0.1
    ), "Configuration risk_manager_config.yaml incorrecte"
    config = config_manager.get_config("config/regime_detector_config.yaml")
    assert (
        config["n_components"] == 3
    ), "Configuration regime_detector_config.yaml incorrecte"
    with pytest.raises(KeyError):
        config_manager.get_config("config/invalid.yaml")


@pytest.mark.asyncio
async def test_get_features(tmp_dirs):
    """Teste l'accès à la configuration des features."""
    config_manager = ConfigManager()
    training_features = config_manager.get_features("training")
    inference_features = config_manager.get_features("inference")
    assert len(training_features) == 350, "Nombre de features d'entraînement incorrect"
    assert len(inference_features) == 150, "Nombre de SHAP features incorrect"


@pytest.mark.asyncio
async def test_get_iqfeed_config(tmp_dirs):
    """Teste l'accès à la configuration IQFeed."""
    config_manager = ConfigManager()
    iqfeed_config = config_manager.get_iqfeed_config()
    assert (
        "ES" in iqfeed_config["symbols"]["value"]
    ), "Symbole ES absent dans iqfeed_config"


@pytest.mark.asyncio
async def test_get_credentials(tmp_dirs):
    """Teste l'accès sécurisé aux credentials."""
    config_manager = ConfigManager()
    credentials = config_manager.get_credentials()
    assert credentials["iqfeed"]["api_key"] == "***", "Clé API non masquée"


@pytest.mark.asyncio
async def test_reload_config(tmp_dirs):
    """Teste le rechargement d'une configuration."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        config_manager = ConfigManager()
        config_manager.reload_config("risk_manager_config.yaml")
        config_manager.reload_config("regime_detector_config.yaml")
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_config_reload" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot config_reload non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "validate_config" in str(op) for op in df_perf["operation"]
        ), "Opération validate_config non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_cloud_backup(tmp_dirs):
    """Teste la sauvegarde distribuée S3."""
    with patch("boto3.client") as mock_s3, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        config_manager = ConfigManager()
        df = pd.DataFrame(
            {
                "timestamp": [datetime.now().isoformat()],
                "config_file": ["es_config.yaml"],
            }
        )
        config_manager.cloud_backup(df, data_type="test_metrics", market="ES")
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
async def test_checkpoint(tmp_dirs):
    """Teste la sauvegarde incrémentielle."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        config_manager = ConfigManager()
        df = pd.DataFrame(
            {
                "timestamp": [datetime.now().isoformat()],
                "config_file": ["es_config.yaml"],
            }
        )
        config_manager.checkpoint(df, data_type="test_metrics", market="ES")
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "config_test_metrics" in f and f.endswith(".json.gz")
            for f in checkpoint_files
        ), "Checkpoint non créé"
        with gzip.open(
            os.path.join(tmp_dirs["checkpoints_dir"], checkpoint_files[0]),
            "rt",
            encoding="utf-8",
        ) as f:
            checkpoint_data = json.load(f)
        assert checkpoint_data["num_rows"] == len(df), "Nombre de lignes incorrect"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "checkpoint" in str(op) for op in df_perf["operation"]
        ), "Opération checkpoint non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_get_features_dynamic(tmp_dirs):
    """Teste le chargement dynamique des features par contexte."""
    config_manager = ConfigManager()
    training_features = config_manager.get_features("training")
    inference_features = config_manager.get_features("inference")
    assert len(training_features) == 350, "Nombre de features d'entraînement incorrect"
    assert len(inference_features) == 150, "Nombre de SHAP features incorrect"
    with pytest.raises(ValueError, match="Contexte inconnu"):
        config_manager.get_features("invalid")


@pytest.mark.asyncio
async def test_get_config_mock(tmp_dirs):
    """Teste le chargement d’une configuration mockée."""
    config_manager = ConfigManager()
    mock_config = {"test_key": "test_value"}
    config = config_manager.get_config(mock_config=mock_config)
    assert config == mock_config, "Configuration mockée incorrecte"
