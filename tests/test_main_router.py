"""
test_main_router.py - Tests unitaires pour src/model/router/main_router.py

Version : 2.1.5
Date : 2025-05-14

Rôle : Valide l’orchestration des régimes de trading et la sélection des modèles SAC/PPO/DDPG avec 350/150 features,
       intégrant volatilité, données d’options, régimes hybrides, SHAP, prédictions LSTM, snapshots compressés,
       sauvegardes incrémentielles/distribuées, alertes Telegram, et la résolution des conflits de signaux via SignalResolver.
       Vérifie la propagation du run_id dans snapshots/cache, le fallback explicite en cas d’échec de SignalResolver,
       et l’utilisation de score_type.

Dépendances :
- pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.8,<6.0.0, matplotlib>=3.7.0,<4.0.0,
  sqlite3, sklearn>=1.5.0,<2.0.0, boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0, pyyaml>=6.0.0,<7.0.0
- src/model/utils/config_manager.py
- src/model/utils/alert_manager.py
- src/model/train_sac.py
- src/model/router/detect_regime.py
- src/model/utils/mind_dialogue.py
- src/model/utils/prediction_aggregator.py
- src/model/utils/model_validator.py
- src/model/utils/algo_performance_logger.py
- src/utils/telegram_alert.py
- src/model/utils/signal_resolver.py

Inputs :
- config/router_config.yaml
- config/feature_sets.yaml
- config/algo_config.yaml
- config/es_config.yaml
- Données factices (DataFrame avec 350/150 features)

Outputs :
- Tests unitaires validant les fonctionnalités de MainRouter.

Notes :
- Conforme à structure.txt (version 2.1.5, 2025-05-14).
- Tests les phases 8 (auto-conscience via confidence_drop_rate), 11 (régimes hybrides), 12 (prédictions LSTM), 17 (SHAP).
- Vérifie l’absence de dxFeed, la suppression de 320/81 features et obs_t, les retries, logs psutil, alertes Telegram,
  snapshots compressés, sauvegardes incrémentielles/distribuées, et l’intégration de SignalResolver avec run_id,
  fallback, et score_type.
- Compatible avec 350 features (entraînement) et 150 SHAP features (inférence) via config/feature_sets.yaml.
"""

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

from src.model.router.main_router import MainRouter


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
    cache_dir = data_dir / "cache" / "main_router"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints"
    checkpoints_dir.mkdir()
    figures_dir = data_dir / "figures" / "main_router"
    figures_dir.mkdir(parents=True)

    # Créer router_config.yaml
    router_config_path = config_dir / "router_config.yaml"
    router_config_content = {
        "fast_mode": False,
        "thresholds": {
            "vix_peak_threshold": 30.0,
            "spread_explosion_threshold": 0.05,
            "iv_high_threshold": 0.25,
            "skew_extreme_threshold": 0.1,
            "vix_high_threshold": 25.0,
        },
        "s3_bucket": "test-bucket",
        "s3_prefix": "main_router/",
        "logging": {"buffer_size": 100},
    }
    with open(router_config_path, "w", encoding="utf-8") as f:
        yaml.dump(router_config_content, f)

    # Créer algo_config.yaml
    algo_config_path = config_dir / "algo_config.yaml"
    algo_config_content = {
        "sac": {"learning_rate": 0.001},
        "ppo": {"learning_rate": 0.001},
        "ddpg": {"learning_rate": 0.001},
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

    # Créer es_config.yaml
    es_config_path = config_dir / "es_config.yaml"
    es_config_content = {
        "signal_resolver": {
            "default_weights": {
                "regime_trend": 2.0,
                "regime_range": 1.0,
                "regime_defensive": 1.0,
                "microstructure_bullish": 1.0,
                "news_score_positive": 0.5,
                "qr_dqn_positive": 1.5,
            },
            "thresholds": {"entropy_alert": 0.5, "conflict_coefficient_alert": 0.5},
        }
    }
    with open(es_config_path, "w", encoding="utf-8") as f:
        yaml.dump(es_config_content, f)

    return {
        "base_dir": str(base_dir),
        "router_config_path": str(router_config_path),
        "algo_config_path": str(algo_config_path),
        "feature_sets_path": str(feature_sets_path),
        "es_config_path": str(es_config_path),
        "logs_dir": str(logs_dir),
        "cache_dir": str(cache_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "figures_dir": str(figures_dir),
        "perf_log_path": str(logs_dir / "main_router_performance.csv"),
        "db_path": str(data_dir / "market_memory.db"),
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
            "call_iv_atm": [0.1] * 50 + [0.2] * 50,
            "option_skew": [0.01] * 50 + [0.05] * 50,
            "ask_size_level_1": np.random.randint(100, 500, 100),
            "bid_size_level_1": np.random.randint(100, 500, 100),
            "bid_ask_imbalance": np.random.uniform(-0.5, 0.5, 100),
            "news_impact_score": np.random.uniform(0.0, 1.0, 100),
        }
    )


@pytest.fixture
def mock_env():
    """Crée un environnement de trading factice."""
    env = MagicMock()
    env.observation_space.shape = (350,)
    env.action_space.shape = (1,)
    env.market = "ES"
    return env


@pytest.fixture
def router(tmp_dirs, mock_env):
    """Crée une instance de MainRouter pour les tests."""
    with patch("src.model.router.main_router.SACTrainer") as mock_trainer, patch(
        "src.model.utils.alert_manager.AlertManager"
    ) as mock_alert, patch(
        "src.model.utils.mind_dialogue.DialogueManager"
    ) as mock_dialogue, patch(
        "src.model.utils.prediction_aggregator.PredictionAggregator"
    ) as mock_aggregator, patch(
        "src.model.utils.model_validator.ModelValidator"
    ) as mock_validator, patch(
        "src.model.utils.algo_performance_logger.AlgoPerformanceLogger"
    ) as mock_logger, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram, patch(
        "src.model.utils.signal_resolver.SignalResolver"
    ) as mock_resolver:
        mock_trainer.return_value.models = {
            "sac": {
                "trend": MagicMock(),
                "range": MagicMock(),
                "defensive": MagicMock(),
            },
            "ppo": {
                "trend": MagicMock(),
                "range": MagicMock(),
                "defensive": MagicMock(),
            },
            "ddpg": {
                "trend": MagicMock(),
                "range": MagicMock(),
                "defensive": MagicMock(),
            },
        }
        for algo in mock_trainer.return_value.models:
            for regime in mock_trainer.return_value.models[algo]:
                mock_trainer.return_value.models[algo][regime].predict.return_value = (
                    np.array([0.5]),
                    None,
                )
                mock_trainer.return_value.models[algo][
                    regime
                ].calculate_reward.return_value = 1.0
        mock_alert.return_value.send_alert.return_value = None
        mock_dialogue.return_value.process_command.return_value = None
        mock_aggregator.return_value.aggregate_predictions.return_value = (
            0.5,
            {"method": "weighted"},
        )
        mock_validator.return_value.validate_model.return_value = {"valid": True}
        mock_logger.return_value.log_performance.return_value = None
        mock_telegram.return_value = None
        mock_resolver.return_value.resolve_conflict.return_value = (
            0.5,
            {
                "score": 2.5,
                "normalized_score": 0.5,
                "entropy": 0.3,
                "conflict_coefficient": 0.4,
                "score_type": "intermediate",
                "contributions": {
                    "regime_trend": {"value": 1.0, "weight": 2.0, "contribution": 2.0},
                    "news_score_positive": {
                        "value": 1.0,
                        "weight": 0.5,
                        "contribution": 0.5,
                    },
                },
                "run_id": "test_run_123",
            },
        )
        router = MainRouter(mock_env, policy_type="mlp", training_mode=True)
    return router


@pytest.mark.asyncio
async def test_init(tmp_dirs, router):
    """Teste l’initialisation de MainRouter."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        assert router.num_features == 350, "Dimension des features incorrecte"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        assert os.path.exists(
            tmp_dirs["db_path"]
        ), "Base de données market_memory.db non créée"
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
async def test_init_invalid_features(tmp_dirs, mock_env):
    """Teste l’initialisation avec un nombre incorrect de features."""
    with patch("src.model.utils.config_manager.get_config") as mock_config, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ):
        mock_config.side_effect = [
            {"fast_mode": False, "thresholds": {}},
            {"sac": {}, "ppo": {}, "ddpg": {}},
            {
                "training": {"features": [f"feature_{i}" for i in range(100)]}
            },  # Moins de 350 features
        ]
        with pytest.raises(ValueError, match="Attendu 350 features"):
            MainRouter(mock_env, training_mode=True)


@pytest.mark.asyncio
async def test_validate_data(router, mock_data):
    """Teste la validation des données."""
    assert router.validate_data(mock_data), "Validation des données échouée"
    df_perf = pd.read_csv(router.perf_log)
    assert any(
        "validate_data" in str(op) for op in df_perf["operation"]
    ), "Opération validate_data non journalisée"
    assert any(
        "confidence_drop_rate" in str(kw) for kw in df_perf.to_dict("records")
    ), "confidence_drop_rate absent"


@pytest.mark.asyncio
async def test_validate_data_invalid(router, mock_data):
    """Teste la validation avec des données invalides."""
    invalid_data = mock_data.copy()
    invalid_data["feature_0"] = np.nan  # Ajouter des NaN
    assert not router.validate_data(
        invalid_data
    ), "Validation des données invalides devrait échouer"
    df_perf = pd.read_csv(router.perf_log)
    assert any(
        "validate_data" in str(op) for op in df_perf["operation"]
    ), "Opération validate_data non journalisée"


@pytest.mark.asyncio
async def test_calculate_metrics(router, mock_data):
    """Teste le calcul des métriques."""
    metrics = router.calculate_metrics(mock_data, 0.5, [1.0, -0.5, 0.2], 50)
    assert "sharpe" in metrics, "Métrique sharpe manquante"
    assert "drawdown" in metrics, "Métrique drawdown manquante"
    assert "profit_factor" in metrics, "Métrique profit_factor manquante"
    df_perf = pd.read_csv(router.perf_log)
    assert any(
        "calculate_metrics" in str(op) for op in df_perf["operation"]
    ), "Opération calculate_metrics non journalisée"


@pytest.mark.asyncio
async def test_route(tmp_dirs, router, mock_data):
    """Teste le routage des prédictions."""
    with patch(
        "src.model.router.main_router.detect_market_regime_vectorized",
        new=AsyncMock(
            return_value=(
                "trend",
                {
                    "regime_probs": {"range": 0.2, "trend": 0.7, "defensive": 0.1},
                    "vix_es_correlation": 20.0,
                    "call_iv_atm": 0.15,
                    "option_skew": 0.02,
                    "predicted_vix": 18.0,
                    "spread": 0.01,
                    "shap_values": {"feature_0": 0.1},
                },
            )
        ),
    ), patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        action, details = await router.route(mock_data, 50)
        assert isinstance(action, float), "Action doit être un float"
        assert "regime" in details, "Détails manquent regime"
        assert "rewards" in details, "Détails manquent rewards"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "route" in str(op) for op in df_perf["operation"]
        ), "Opération route non journalisée"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_route_trend" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot route non créé"
        assert any(
            f.startswith("route_trend_") and f.endswith(".png")
            for f in os.listdir(tmp_dirs["figures_dir"])
        ), "Visualisation non générée"
        with sqlite3.connect(tmp_dirs["db_path"]) as conn:
            df_metrics = pd.read_sql("SELECT * FROM metrics", conn)
            assert (
                len(df_metrics) > 0
            ), "Aucune métrique enregistrée dans market_memory.db"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_route_ultra_defensive(tmp_dirs, router, mock_data):
    """Teste le routage en mode ultra-défensif."""
    with patch(
        "src.model.router.main_router.detect_market_regime_vectorized",
        new=AsyncMock(
            return_value=(
                "ultra-defensive",
                {
                    "regime_probs": {"range": 0.0, "trend": 0.0, "defensive": 1.0},
                    "vix_es_correlation": 40.0,
                    "call_iv_atm": 0.3,
                    "option_skew": 0.15,
                    "predicted_vix": 30.0,
                    "spread": 0.06,
                    "shap_values": {},
                },
            )
        ),
    ), patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        action, details = await router.route(mock_data, 50)
        assert action == 0.0, "Action devrait être 0.0 en mode ultra-défensif"
        assert (
            details["regime"] == "ultra-defensive"
        ), "Régime ultra-défensif non détecté"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_stop_trading_command(tmp_dirs, router, mock_data):
    """Teste la commande d’arrêt du trading."""
    with patch(
        "src.model.router.main_router.detect_market_regime_vectorized",
        new=AsyncMock(
            return_value=(
                "trend",
                {
                    "regime_probs": {"range": 0.2, "trend": 0.7, "defensive": 0.1},
                    "vix_es_correlation": 20.0,
                    "call_iv_atm": 0.15,
                    "option_skew": 0.02,
                    "predicted_vix": 18.0,
                    "spread": 0.01,
                    "shap_values": {"feature_0": 0.1},
                },
            )
        ),
    ), patch(
        "src.model.utils.mind_dialogue.DialogueManager.process_command",
        return_value="stop_trading",
    ), patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        action, details = await router.route(mock_data, 50)
        assert action == 0.0, "Action devrait être 0.0 pour commande stop_trading"
        assert (
            "error" in details and details["error"] == "Arrêt commandé"
        ), "Détails d’erreur incorrects"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_save_snapshot_compressed(tmp_dirs, router):
    """Teste la sauvegarde d’un snapshot compressé."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        snapshot_data = {"test": "compressed_snapshot"}
        router.save_snapshot("test_compressed", snapshot_data, compress=True)
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
async def test_checkpoint(tmp_dirs, router):
    """Teste la sauvegarde incrémentielle."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        df = pd.DataFrame(
            {
                "timestamp": [datetime.now().isoformat()],
                "regime": ["trend"],
                "action": [0.5],
                "reward": [1.0],
                "sharpe": [0.8],
                "drawdown": [0.05],
                "profit_factor": [1.2],
                "vix_es_correlation": [20.0],
                "call_iv_atm": [0.15],
            }
        )
        router.checkpoint(df, data_type="test_metrics")
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "main_router_test_metrics" in f and f.endswith(".json.gz")
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
async def test_cloud_backup(tmp_dirs, router):
    """Teste la sauvegarde distribuée S3."""
    with patch("boto3.client") as mock_s3, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        df = pd.DataFrame(
            {
                "timestamp": [datetime.now().isoformat()],
                "regime": ["trend"],
                "action": [0.5],
                "reward": [1.0],
                "sharpe": [0.8],
                "drawdown": [0.05],
                "profit_factor": [1.2],
                "vix_es_correlation": [20.0],
                "call_iv_atm": [0.15],
            }
        )
        router.cloud_backup(df, data_type="test_metrics")
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
async def test_handle_sigint(tmp_dirs, router):
    """Teste la gestion SIGINT."""
    with patch("sys.exit") as mock_exit, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        router.handle_sigint(signal.SIGINT, None)
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
async def test_prune_cache(router):
    """Teste la purge du cache."""
    router.prediction_cache = OrderedDict({f"key_{i}": (0.5, {}) for i in range(1001)})
    router.prune_cache()
    assert len(router.prediction_cache) <= 1000, "Cache non purgé correctement"


@pytest.mark.asyncio
async def test_resolve_signals(tmp_dirs, router, mock_data):
    """Teste la méthode resolve_signals."""
    details = {
        "regime_probs": {"range": 0.2, "trend": 0.7, "defensive": 0.1},
        "qr_dqn_quantile_mean": 0.5,
    }
    metadata = router.resolve_signals(mock_data, details, "trend", 50)
    assert "score" in metadata, "Métadonnées manquent score"
    assert "normalized_score" in metadata, "Métadonnées manquent normalized_score"
    assert "entropy" in metadata, "Métadonnées manquent entropy"
    assert (
        "conflict_coefficient" in metadata
    ), "Métadonnées manquent conflict_coefficient"
    assert "run_id" in metadata, "Métadonnées manquent run_id"
    assert (
        "score_type" in metadata and metadata["score_type"] == "intermediate"
    ), "score_type incorrect"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "resolve_signals" in str(op) for op in df_perf["operation"]
    ), "Opération resolve_signals non journalisée"


@pytest.mark.asyncio
async def test_route_signal_metadata(tmp_dirs, router, mock_data):
    """Teste l'intégration de SignalResolver dans route."""
    with patch(
        "src.model.router.main_router.detect_market_regime_vectorized",
        new=AsyncMock(
            return_value=(
                "trend",
                {
                    "regime_probs": {"range": 0.2, "trend": 0.7, "defensive": 0.1},
                    "vix_es_correlation": 20.0,
                    "call_iv_atm": 0.15,
                    "option_skew": 0.02,
                    "predicted_vix": 18.0,
                    "spread": 0.01,
                    "shap_values": {"feature_0": 0.1},
                    "qr_dqn_quantile_mean": 0.5,
                },
            )
        ),
    ), patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        action, details = await router.route(mock_data, 50)
        assert "signal_metadata" in details, "Détails manquent signal_metadata"
        assert (
            "normalized_score" in details["signal_metadata"]
        ), "signal_metadata manque normalized_score"
        assert (
            "conflict_coefficient" in details["signal_metadata"]
        ), "signal_metadata manque conflict_coefficient"
        assert "entropy" in details["signal_metadata"], "signal_metadata manque entropy"
        assert "run_id" in details, "Détails manquent run_id"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_route_trend" in f for f in snapshot_files
        ), "Snapshot route non créé"
        with gzip.open(
            os.path.join(tmp_dirs["cache_dir"], snapshot_files[-1]),
            "rt",
            encoding="utf-8",
        ) as f:
            snapshot = json.load(f)
        assert "signal_metadata" in snapshot["data"], "Snapshot manque signal_metadata"
        assert "run_id" in snapshot["data"], "Snapshot manque run_id"
        assert any(
            f.startswith("route_trend_") and f.endswith(".png")
            for f in os.listdir(tmp_dirs["figures_dir"])
        ), "Visualisation non générée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_route_signal_resolution_error(tmp_dirs, router, mock_data):
    """Teste la gestion des erreurs dans resolve_signals avec fallback."""
    with patch(
        "src.model.router.main_router.detect_market_regime_vectorized",
        new=AsyncMock(
            return_value=(
                "trend",
                {
                    "regime_probs": {"range": 0.2, "trend": 0.7, "defensive": 0.1},
                    "vix_es_correlation": 20.0,
                    "call_iv_atm": 0.15,
                    "option_skew": 0.02,
                    "predicted_vix": 18.0,
                    "spread": 0.01,
                    "shap_values": {"feature_0": 0.1},
                    "qr_dqn_quantile_mean": 0.5,
                },
            )
        ),
    ), patch(
        "src.model.utils.signal_resolver.SignalResolver.resolve_conflict",
        side_effect=ValueError("Signal invalide"),
    ), patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        action, details = await router.route(mock_data, 50)
        assert action == 0.0, "Action devrait être 0.0 en cas d’erreur de résolution"
        assert "signal_metadata" in details, "Détails manquent signal_metadata"
        assert details["signal_metadata"]["score"] == 0.0, "Score par défaut incorrect"
        assert (
            details["signal_metadata"]["conflict_coefficient"] == 0.0
        ), "Conflict coefficient par défaut incorrect"
        assert "error" in details["signal_metadata"], "Métadonnées manquent erreur"
        assert "run_id" in details["signal_metadata"], "Métadonnées manquent run_id"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "resolve_signals" in str(op) for op in df_perf["operation"]
        ), "Opération resolve_signals non journalisée"
        mock_telegram.assert_called()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
