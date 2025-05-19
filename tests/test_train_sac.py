# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_train_sac.py
# Tests unitaires pour src/model/train_sac.py
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide l’entraînement des modèles SAC, PPO, DDPG par régime avec 350/150 features, intégrant 13 méthodes avancées,
#        snapshots compressés, sauvegardes incrémentielles/distribuées, et alertes Telegram.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.8,<6.0.0,
#   stable-baselines3>=2.0.0,<3.0.0, pyyaml>=6.0.0,<7.0.0, torch>=2.0.0,<3.0.0, sklearn>=1.5.0,<2.0.0,
#   matplotlib>=3.7.0,<4.0.0, boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0, sqlite3
# - src/features/neural_pipeline.py
# - src/model/router/detect_regime.py
# - src/model/adaptive_learner.py
# - src/features/feature_pipeline.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/algo_performance_logger.py
# - src/model/utils/finetune_utils.py
# - src/model/utils/maml_utils.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/algo_config.yaml
# - config/feature_sets.yaml
# - Données factices (DataFrame avec 350 features)
# - Base de données SQLite factice
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de SACTrainer.
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Tests les phases 4-18.
# - Vérifie l’absence de dxFeed, la suppression de 320/81 features et obs_t, les retries, logs psutil, alertes Telegram,
#   snapshots compressés, et sauvegardes incrémentielles/distribuées.
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence) via config/feature_sets.yaml.

import gzip
import json
import os
import sqlite3
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.model.train_sac import SACTrainer


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
    cache_dir = data_dir / "cache" / "train_sac" / "ES"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints" / "train_sac" / "ES" / "trend" / "mlp"
    checkpoints_dir.mkdir(parents=True)
    figures_dir = data_dir / "figures" / "train_sac" / "ES" / "trend"
    figures_dir.mkdir(parents=True)
    model_dir = data_dir / "model" / "sac_models" / "ES" / "trend" / "mlp"
    model_dir.mkdir(parents=True)
    features_dir = data_dir / "features"
    features_dir.mkdir()

    # Créer algo_config.yaml
    algo_config_path = config_dir / "algo_config.yaml"
    algo_config_content = {
        "sac": {"learning_rate": 0.0001},
        "ppo": {"learning_rate": 0.0001},
        "ddpg": {"learning_rate": 0.0001},
        "neural_pipeline": {"window_size": 50},
        "logging": {"buffer_size": 100},
        "s3_bucket": "test-bucket",
        "s3_prefix": "train_sac/",
    }
    with open(algo_config_path, "w", encoding="utf-8") as f:
        yaml.dump(algo_config_content, f)

    # Créer feature_sets.yaml
    feature_sets_path = config_dir / "feature_sets.yaml"
    feature_sets_content = {
        "training": {"features": [f"feature_{i}" for i in range(350)]},
        "inference": {"shap_features": [f"shap_feature_{i}" for i in range(150)]},
    }
    with open(feature_sets_path, "w", encoding="utf-8") as f:
        yaml.dump(feature_sets_content, f)

    # Créer feature_importance.csv
    feature_importance_path = features_dir / "feature_importance.csv"
    pd.DataFrame(
        {"feature": [f"feature_{i}" for i in range(150)], "importance": [0.1] * 150}
    ).to_csv(feature_importance_path, index=False)

    # Créer market_memory.db
    db_path = data_dir / "market_memory.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE clusters (market TEXT, cluster INTEGER)")
        conn.execute("INSERT INTO clusters (market, cluster) VALUES (?, ?)", ("ES", 1))

    return {
        "base_dir": str(base_dir),
        "algo_config_path": str(algo_config_path),
        "feature_sets_path": str(feature_sets_path),
        "feature_importance_path": str(feature_importance_path),
        "logs_dir": str(logs_dir),
        "cache_dir": str(cache_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "figures_dir": str(figures_dir),
        "model_dir": str(model_dir),
        "db_path": str(db_path),
        "perf_log_path": str(logs_dir / "train_sac_performance.csv"),
    }


@pytest.fixture
def mock_data():
    """Crée un DataFrame factice avec 350 features."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=100, freq="1min"),
            **{f"feature_{i}": np.random.uniform(0, 1, 100) for i in range(350)},
            "close": np.random.uniform(4000, 5000, 100),
            "vix_es_correlation": [20.0] * 50 + [30.0] * 50,
            "news_impact_score": np.random.uniform(-1, 1, 100),
            "predicted_vix": np.random.uniform(15, 25, 100),
            "gex_es": np.random.uniform(-1000, 1000, 100),
            "gex_mnq": np.random.uniform(-1000, 1000, 100),
            "profit_es": np.random.uniform(-100, 100, 100),
            "profit_mnq": np.random.uniform(-100, 100, 100),
            "bid_size_level_1": np.random.randint(100, 500, 100),
            "ask_size_level_1": np.random.randint(100, 500, 100),
            "trade_frequency_1s": np.random.uniform(0.1, 10, 100),
            "spread_avg_1min": np.random.uniform(0.01, 0.05, 100),
            "atr_14": np.random.uniform(0.1, 1.0, 100),
        }
    )


@pytest.fixture
def mock_env():
    """Crée un environnement de trading factice."""
    env = MagicMock()
    env.observation_space.shape = (350,)
    env.action_space.shape = (1,)
    env.feature_cols = [f"feature_{i}" for i in range(350)]
    return env


@pytest.fixture
def trainer(tmp_dirs, mock_env):
    """Crée une instance de SACTrainer pour les tests."""
    with patch("src.model.train_sac.NeuralPipeline") as mock_neural, patch(
        "src.model.utils.alert_manager.AlertManager"
    ) as mock_alert, patch(
        "src.model.utils.algo_performance_logger.AlgoPerformanceLogger"
    ) as mock_logger, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        mock_neural.return_value.generate_lstm_predictions.return_value = {
            "predicted_vix": np.random.uniform(15, 25, 100)
        }
        mock_alert.return_value.send_alert.return_value = None
        mock_logger.return_value.log_performance.return_value = None
        mock_telegram.return_value = None
        trainer = SACTrainer(mock_env, policy_type="mlp")
    return trainer


@pytest.mark.asyncio
async def test_init(tmp_dirs, trainer):
    """Teste l’initialisation de SACTrainer."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        assert trainer.policy_type == "mlp", "Type de politique incorrect"
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
async def test_validate_data(trainer, mock_data):
    """Teste la validation des données."""
    assert trainer.validate_data(
        mock_data, market="ES"
    ), "Validation des données échouée"
    df_perf = pd.read_csv(trainer.perf_log_path)
    assert any(
        "validate_data" in str(op) for op in df_perf["operation"]
    ), "Opération validate_data non journalisée"
    assert any(
        "confidence_drop_rate" in str(kw) for kw in df_perf.to_dict("records")
    ), "confidence_drop_rate absent"


@pytest.mark.asyncio
async def test_validate_data_invalid(trainer, mock_data):
    """Teste la validation avec des données invalides."""
    invalid_data = mock_data.copy()
    invalid_data["bid_size_level_1"] = np.nan
    assert trainer.validate_data(
        invalid_data, market="ES"
    ), "Validation devrait réussir avec interpolation"
    df_perf = pd.read_csv(trainer.perf_log_path)
    assert any(
        "validate_data" in str(op) for op in df_perf["operation"]
    ), "Opération validate_data non journalisée"


@pytest.mark.asyncio
async def test_load_clusters(tmp_dirs, trainer):
    """Teste le chargement des clusters."""
    clusters = trainer.load_clusters(db_path=tmp_dirs["db_path"], market="ES")
    assert len(clusters) > 0, "Clusters non chargés"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "load_clusters" in str(op) for op in df_perf["operation"]
    ), "Opération load_clusters non journalisée"


@pytest.mark.asyncio
async def test_configure_gpu(trainer):
    """Teste la configuration GPU."""
    with patch("torch.cuda.is_available", return_value=True), patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        trainer.configure_gpu()
        df_perf = pd.read_csv(trainer.perf_log_path)
        assert any(
            "configure_gpu" in str(op) for op in df_perf["operation"]
        ), "Opération configure_gpu non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_load_shap_fallback(tmp_dirs, trainer):
    """Teste le chargement des SHAP features avec fallback."""
    features = trainer.load_shap_fallback(shap_file=tmp_dirs["feature_importance_path"])
    assert len(features) == 150, "Nombre incorrect de SHAP features"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "load_shap_fallback" in str(op) for op in df_perf["operation"]
    ), "Opération load_shap_fallback non journalisée"


@pytest.mark.asyncio
async def test_initialize_model(trainer):
    """Teste l’initialisation du modèle."""
    with patch("stable_baselines3.SAC.__init__", return_value=None) as mock_sac:
        model = trainer.initialize_model("sac", "trend", market="ES")
        assert model is not None, "Modèle non initialisé"
        df_perf = pd.read_csv(trainer.perf_log_path)
        assert any(
            "initialize_model" in str(op) for op in df_perf["operation"]
        ), "Opération initialize_model non journalisée"


@pytest.mark.asyncio
async def test_integrate_lstm_predictions(trainer, mock_data):
    """Teste l’intégration des prédictions LSTM."""
    data = trainer.integrate_lstm_predictions(mock_data, market="ES")
    assert "neural_dynamic_feature_1" in data.columns, "Feature LSTM non ajoutée"
    df_perf = pd.read_csv(trainer.perf_log_path)
    assert any(
        "integrate_lstm_predictions" in str(op) for op in df_perf["operation"]
    ), "Opération integrate_lstm_predictions non journalisée"


@pytest.mark.asyncio
async def test_save_checkpoint(tmp_dirs, trainer):
    """Teste la sauvegarde incrémentielle du modèle."""
    with patch("stable_baselines3.SAC.save") as mock_save, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        model = MagicMock()
        await trainer.save_checkpoint(model, "trend", "mlp", market="ES")
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_checkpoint_ES_trend" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot checkpoint non créé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "sac_trend_mlp" in f and f.endswith(".zip") for f in checkpoint_files
        ), "Checkpoint non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "save_checkpoint" in str(op) for op in df_perf["operation"]
        ), "Opération save_checkpoint non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_generate_visualizations(tmp_dirs, trainer, mock_data):
    """Teste la génération des visualisations."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        trainer.generate_visualizations(
            mock_data, [1.0, 0.5, -0.2], "trend", market="ES"
        )
        figure_files = os.listdir(tmp_dirs["figures_dir"])
        assert any(
            "rewards_trend" in f and f.endswith(".png") for f in figure_files
        ), "Visualisation des rewards non générée"
        assert any(
            "shap_features_trend" in f and f.endswith(".png") for f in figure_files
        ), "Visualisation SHAP non générée"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "generate_visualizations" in str(op) for op in df_perf["operation"]
        ), "Opération generate_visualizations non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_train_multi_market(tmp_dirs, trainer, mock_data):
    """Teste l’entraînement multi-marché."""
    with patch(
        "src.model.train_sac.MarketRegimeDetector.detect",
        new=AsyncMock(
            return_value=(
                "trend",
                {"regime_probs": {"trend": 0.7, "range": 0.2, "defensive": 0.1}},
            )
        ),
    ), patch("stable_baselines3.SAC.__init__", return_value=None), patch(
        "stable_baselines3.SAC.save"
    ), patch(
        "src.model.utils.finetune_utils.finetune_model", return_value=MagicMock()
    ), patch(
        "src.model.utils.maml_utils.apply_prototypical_networks",
        return_value=MagicMock(),
    ), patch(
        "src.model.train_sac.online_learning", return_value=MagicMock()
    ), patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        model = await trainer.train_multi_market(
            mock_data, total_timesteps=1000, mode="trend", market="ES"
        )
        assert model is not None, "Entraînement échoué"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_train_sac_multi_market_ES" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot train non créé"
        model_files = os.listdir(tmp_dirs["model_dir"])
        assert any(
            "sac_trend_mlp" in f and f.endswith(".zip") for f in model_files
        ), "Modèle non sauvegardé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "train_sac_multi_market" in str(op) for op in df_perf["operation"]
        ), "Opération train_sac_multi_market non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_cloud_backup(tmp_dirs, trainer, mock_data):
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
        trainer.cloud_backup(df, data_type="test_metrics", market="ES")
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
async def test_save_snapshot_compressed(tmp_dirs, trainer):
    """Teste la sauvegarde d’un snapshot compressé."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        snapshot_data = {"test": "compressed_snapshot"}
        trainer.save_snapshot(
            "test_compressed", snapshot_data, market="ES", compress=True
        )
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
async def test_checkpoint(tmp_dirs, trainer):
    """Teste la sauvegarde incrémentielle."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        df = pd.DataFrame(
            {
                "timestamp": [datetime.now().isoformat()],
                "operation": ["test"],
                "latency": [1.0],
            }
        )
        trainer.checkpoint(df, data_type="test_metrics", market="ES")
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "train_sac_test_metrics" in f and f.endswith(".json.gz")
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
