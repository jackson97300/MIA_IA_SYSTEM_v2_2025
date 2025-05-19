# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_mia_dashboard.py
# Tests unitaires pour src/monitoring/mia_dashboard.py
#
# Version : 2.1.5
# Date : 2025-05-14
#
# Rôle : Valide le tableau de bord interactif Dash pour visualiser les performances de trading, régimes, SHAP features,
#        et prédictions LSTM, avec intégration de confidence_drop_rate (Phase 8), snapshots compressés,
#        sauvegardes incrémentielles/distribuées, alertes Telegram, nouvelles features bid_ask_imbalance,
#        trade_aggressiveness, iv_skew, iv_term_structure, option_skew, news_impact_score, cache LRU,
#        validation obs_t, et optimisation mémoire via chemins PNG.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, plotly>=5.0.0,<6.0.0, numpy>=1.26.4,<2.0.0, matplotlib>=3.8.0,<4.0.0,
#   psutil>=5.9.8,<6.0.0, loguru>=0.7.0,<1.0.0, pyyaml>=6.0.0,<7.0.0, boto3>=1.26.0,<2.0.0
# - src/model/router/detect_regime.py
# - src/features/neural_pipeline.py
# - src/features/shap_weighting.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/es_config.yaml
# - config/feature_sets.yaml
# - config/model_params.yaml
# - data/features/features_latest.csv
# - data/trades/trades_simulated.csv
# - data/features/feature_importance.csv
# - data/logs/regime_history.csv
# - data/neural_pipeline_dashboard.json
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de mia_dashboard.py.
#
# Notes :
# - Conforme à structure.txt (version 2.1.5, 2025-05-14).
# - Tests les phases 8 (auto-conscience via confidence_drop_rate), 17 (SHAP).
# - Vérifie l’absence de dxFeed, la suppression de 320/81 features et obs_t, les retries, logs psutil, alertes Telegram,
#   snapshots compressés, sauvegardes incrémentielles/distribuées, cache LRU, validation obs_t, et nouvelles features.
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence) via config/feature_sets.yaml.
# - Tests renforcés avec mocks pour psutil, boto3, telegram_alert, et intégration CI/CD recommandée (GitHub Actions).
# - Couvre la séparation MVC simulée (CORE, LAYOUT, CALLBACKS) et l’optimisation mémoire (chemins PNG au lieu de figures Plotly).

import gzip
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml
from dash import html

from src.monitoring.mia_dashboard import MIADashboard, app


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
    trades_dir = data_dir / "trades"
    trades_dir.mkdir()
    features_dir = data_dir / "features"
    features_dir.mkdir()
    figures_dir = data_dir / "figures" / "monitoring"
    figures_dir.mkdir(parents=True)
    cache_dir = data_dir / "cache" / "dashboard"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints"
    checkpoints_dir.mkdir()

    # Créer es_config.yaml
    config_path = config_dir / "es_config.yaml"
    config_content = {
        "dashboard_params": {
            "interval": 10000,
            "max_rows": 100,
            "s3_bucket": "test-bucket",
            "s3_prefix": "dashboard/",
            "logging": {"buffer_size": 100},
        }
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    # Créer model_params.yaml
    model_params_path = config_dir / "model_params.yaml"
    model_params_content = {
        "neural_pipeline": {
            "window_size": 50,
            "base_features": 150,
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
            "save_dir": str(data_dir / "models"),
            "num_lstm_features": 8,
            "logging": {"buffer_size": 100},
            "cache": {"max_cache_size": 1000, "cache_hours": 24},
        }
    }
    with open(model_params_path, "w", encoding="utf-8") as f:
        yaml.dump(model_params_content, f)

    # Créer feature_sets.yaml
    feature_sets_path = config_dir / "feature_sets.yaml"
    feature_sets_content = {
        "inference": {
            "shap_features": [
                "vix_es_correlation",
                "predicted_vix",
                "news_impact_score",
                "call_iv_atm",
                "option_skew",
                "spoofing_score",
                "volume_anomaly",
                "net_gamma",
                "call_wall",
                "trade_velocity",
                "hft_activity_score",
                "bid_ask_imbalance",
                "trade_aggressiveness",
                "iv_skew",
                "iv_term_structure",
                "close",
                "volume",
            ]
            + [f"feature_{i}" for i in range(133)]  # Total 150
        },
        "training": {"features": [f"feature_{i}" for i in range(350)]},
    }
    with open(feature_sets_path, "w", encoding="utf-8") as f:
        yaml.dump(feature_sets_content, f)

    # Créer neural_pipeline_dashboard.json
    dashboard_json_path = data_dir / "neural_pipeline_dashboard.json"
    dashboard_json_content = {
        "status": "pretrained",
        "num_rows": 100,
        "recent_errors": 0,
        "average_latency": 0.5,
        "confidence_drop_rate": 0.2,
        "new_features": [
            "bid_ask_imbalance",
            "trade_aggressiveness",
            "iv_skew",
            "iv_term_structure",
            "option_skew",
            "news_impact_score",
        ],
    }
    with open(dashboard_json_path, "w", encoding="utf-8") as f:
        json.dump(dashboard_json_content, f)

    # Créer features_latest.csv
    features_latest_path = features_dir / "features_latest.csv"
    features_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-14 09:00", periods=100, freq="1min"),
            "close": np.random.normal(5100, 10, 100),
            "volume": np.random.randint(1000, 10000, 100),
            "vix_es_correlation": np.random.uniform(-1, 1, 100),
            "predicted_vix": np.random.uniform(10, 30, 100),
            "news_impact_score": np.random.uniform(0, 1, 100),
            "call_iv_atm": np.random.uniform(0.1, 0.5, 100),
            "option_skew": np.random.uniform(-0.2, 0.2, 100),
            "spoofing_score": np.random.uniform(0, 1, 100),
            "volume_anomaly": np.random.uniform(0, 1, 100),
            "net_gamma": np.random.uniform(-1000, 1000, 100),
            "call_wall": np.random.uniform(5000, 5200, 100),
            "trade_velocity": np.random.uniform(0, 10, 100),
            "hft_activity_score": np.random.uniform(0, 1, 100),
            "bid_size_level_1": np.random.randint(100, 1000, 100),
            "ask_size_level_1": np.random.randint(100, 1000, 100),
            "trade_frequency_1s": np.random.uniform(0, 10, 100),
            "market_regime": np.random.choice(["trend", "range", "defensive"], 100),
            "bid_ask_imbalance": np.random.normal(0, 0.1, 100),
            "trade_aggressiveness": np.random.normal(0, 0.2, 100),
            "iv_skew": np.random.normal(0.01, 0.005, 100),
            "iv_term_structure": np.random.normal(0.02, 0.005, 100),
            **{
                f"feature_{i}": np.random.uniform(0, 1, 100) for i in range(128)
            },  # Total 150 columns
        }
    )
    features_data.to_csv(features_latest_path, index=False, encoding="utf-8")

    # Créer trades_simulated.csv
    trades_simulated_path = trades_dir / "trades_simulated.csv"
    trades_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-14 09:00", periods=100, freq="1min"),
            "reward": np.random.uniform(-100, 100, 100),
        }
    )
    trades_data.to_csv(trades_simulated_path, index=False, encoding="utf-8")

    # Créer feature_importance.csv
    feature_importance_path = features_dir / "feature_importance.csv"
    shap_data = pd.DataFrame(
        {
            "feature": [
                "vix_es_correlation",
                "predicted_vix",
                "news_impact_score",
                "call_iv_atm",
                "option_skew",
                "spoofing_score",
                "volume_anomaly",
                "net_gamma",
                "call_wall",
                "trade_velocity",
                "hft_activity_score",
                "bid_ask_imbalance",
                "trade_aggressiveness",
                "iv_skew",
                "iv_term_structure",
                "close",
                "volume",
            ]
            + [f"feature_{i}" for i in range(133)],
            "importance": np.random.uniform(0, 1, 150),
        }
    )
    shap_data.to_csv(feature_importance_path, index=False, encoding="utf-8")

    # Créer regime_history.csv
    regime_history_path = logs_dir / "regime_history.csv"
    regime_history_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-14 09:00", periods=100, freq="1min"),
            "regime": np.random.choice(["trend", "range", "defensive"], 100),
        }
    )
    regime_history_data.to_csv(regime_history_path, index=False, encoding="utf-8")

    return {
        "base_dir": str(base_dir),
        "config_path": str(config_path),
        "model_params_path": str(model_params_path),
        "feature_sets_path": str(feature_sets_path),
        "dashboard_json_path": str(dashboard_json_path),
        "features_latest_path": str(features_latest_path),
        "trades_simulated_path": str(trades_simulated_path),
        "feature_importance_path": str(feature_importance_path),
        "regime_history_path": str(regime_history_path),
        "logs_dir": str(logs_dir),
        "figures_dir": str(figures_dir),
        "cache_dir": str(cache_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "perf_log_path": str(logs_dir / "dashboard_performance.csv"),
    }


@pytest.fixture
def mock_dependencies():
    """Mock les dépendances critiques pour les tests."""
    with patch(
        "src.monitoring.mia_dashboard.MarketRegimeDetector"
    ) as mock_detector, patch(
        "src.monitoring.mia_dashboard.NeuralPipeline"
    ) as mock_pipeline:
        mock_detector.return_value.detect.return_value = (
            "trend",
            {
                "neural_regime": "trend",
                "regime_probs": {"trend": 0.7, "range": 0.2, "defensive": 0.1},
            },
        )
        mock_pipeline.return_value.run.return_value = {
            "features": np.zeros((51, 159)),
            "volatility": np.random.uniform(10, 30, 50),
            "regime": np.random.randint(0, 3, 50),
            "predicted_vix": 20.0,
        }
        mock_pipeline.return_value.validate_obs_t.return_value = []
        yield


@pytest.fixture
def mock_psutil():
    """Mock psutil pour contrôler les métriques mémoire et CPU."""
    with patch("psutil.Process") as mock_process, patch(
        "psutil.cpu_percent"
    ) as mock_cpu:
        mock_process.return_value.memory_info.return_value.rss = (
            100 * 1024 * 1024
        )  # 100 MB
        mock_cpu.return_value = 50.0  # 50% CPU
        yield


def test_init(tmp_dirs, mock_dependencies, mock_psutil):
    """Teste l’initialisation de MIADashboard."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        dashboard = MIADashboard(config_path=tmp_dirs["config_path"])
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        assert isinstance(
            dashboard.cache, OrderedDict
        ), "Cache non initialisé comme OrderedDict"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "init" in str(op) for op in df["operation"]
        ), "Opération init non journalisée"
        assert any(
            "memory_usage_mb" in str(kw) for kw in df.to_dict("records")
        ), "memory_usage_mb absent"
        assert any(
            "cpu_usage_percent" in str(kw) for kw in df.to_dict("records")
        ), "cpu_usage_percent absent"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_init" in f and f.endswith(".json.gz") for f in snapshot_files
        ), "Snapshot init non créé"
        mock_telegram.assert_called()


def test_load_config(tmp_dirs, mock_psutil):
    """Teste le chargement de la configuration."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        dashboard = MIADashboard(config_path=tmp_dirs["config_path"])
        config = dashboard.load_config(tmp_dirs["config_path"])
        assert "interval" in config, "Clé interval manquante"
        assert "s3_bucket" in config, "Clé s3_bucket manquante"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "load_config" in str(op) for op in df["operation"]
        ), "Opération load_config non journalisée"
        assert any(
            "memory_usage_mb" in str(kw) for kw in df.to_dict("records")
        ), "memory_usage_mb absent"
        mock_telegram.assert_called()


def test_load_data_features(tmp_dirs, mock_dependencies, mock_psutil):
    """Teste le chargement et la validation des données de features."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        dashboard = MIADashboard(config_path=tmp_dirs["config_path"])
        df = dashboard.load_data(tmp_dirs["features_latest_path"], "features")
        assert len(df) <= 100, "Nombre de lignes dépasse max_rows"
        assert all(
            f in df.columns
            for f in [
                "vix_es_correlation",
                "predicted_vix",
                "news_impact_score",
                "call_iv_atm",
                "option_skew",
                "bid_ask_imbalance",
                "trade_aggressiveness",
                "iv_skew",
                "iv_term_structure",
            ]
        ), "Nouvelles features manquantes"
        assert len(df.columns) >= 50, "Moins de 50 features chargées"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "load_data" in str(op) for op in df_perf["operation"]
        ), "Opération load_data non journalisée"
        assert any(
            "confidence_drop_rate" in str(kw) for kw in df_perf.to_dict("records")
        ), "confidence_drop_rate absent"
        assert any(
            "validate_obs_t" in str(op) for op in df_perf["operation"]
        ), "Validation obs_t non journalisée"
        mock_telegram.assert_called()


def test_load_data_trades(tmp_dirs, mock_dependencies, mock_psutil):
    """Teste le chargement et la validation des données de trades."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        dashboard = MIADashboard(config_path=tmp_dirs["config_path"])
        df = dashboard.load_data(tmp_dirs["trades_simulated_path"], "trades")
        assert "reward" in df.columns, "Colonne reward manquante"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "load_data" in str(op) for op in df_perf["operation"]
        ), "Opération load_data non journalisée"
        assert any(
            "memory_usage_mb" in str(kw) for kw in df_perf.to_dict("records")
        ), "memory_usage_mb absent"
        mock_telegram.assert_called()


def test_get_regime(tmp_dirs, mock_dependencies, mock_psutil):
    """Teste la détection des régimes hybrides."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        dashboard = MIADashboard(config_path=tmp_dirs["config_path"])
        df = dashboard.load_data(tmp_dirs["features_latest_path"], "features")
        regime, neural_regime, regime_probs = dashboard.get_regime(
            df, step=99, config_path=tmp_dirs["config_path"]
        )
        assert regime == "trend", "Régime incorrect"
        assert neural_regime == "trend", "Neural régime incorrect"
        assert isinstance(regime_probs, pd.Series), "regime_probs n’est pas une Series"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "get_regime" in str(op) for op in df_perf["operation"]
        ), "Opération get_regime non journalisée"
        assert any(
            "memory_usage_mb" in str(kw) for kw in df_perf.to_dict("records")
        ), "memory_usage_mb absent"
        mock_telegram.assert_called()


def test_create_regime_fig(tmp_dirs, mock_dependencies, mock_psutil):
    """Teste la génération du graphique des régimes."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram, patch(
        "plotly.io.write_image"
    ) as mock_write_image:
        dashboard = MIADashboard(config_path=tmp_dirs["config_path"])
        df = dashboard.load_data(tmp_dirs["features_latest_path"], "features")
        output_path = str(Path(tmp_dirs["figures_dir"]) / "regime_test.png")
        fig_path = dashboard.create_regime_fig(df, step=99, output_path=output_path)
        assert fig_path == output_path, "Chemin de la figure incorrect"
        assert os.path.exists(output_path), "Figure régime non générée"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "create_regime_fig" in str(op) for op in df_perf["operation"]
        ), "Opération create_regime_fig non journalisée"
        assert any(
            "memory_usage_mb" in str(kw) for kw in df_perf.to_dict("records")
        ), "memory_usage_mb absent"
        mock_write_image.assert_called()
        mock_telegram.assert_called()


def test_create_feature_fig(tmp_dirs, mock_dependencies, mock_psutil):
    """Teste la génération du graphique des features."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram, patch(
        "plotly.io.write_image"
    ) as mock_write_image:
        dashboard = MIADashboard(config_path=tmp_dirs["config_path"])
        df = dashboard.load_data(tmp_dirs["features_latest_path"], "features")
        output_path = str(Path(tmp_dirs["figures_dir"]) / "feature_test.png")
        fig_path = dashboard.create_feature_fig(
            df,
            selected_features=["close", "predicted_vix", "bid_ask_imbalance"],
            output_path=output_path,
        )
        assert fig_path == output_path, "Chemin de la figure incorrect"
        assert os.path.exists(output_path), "Figure features non générée"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "create_feature_fig" in str(op) for op in df_perf["operation"]
        ), "Opération create_feature_fig non journalisée"
        assert any(
            "memory_usage_mb" in str(kw) for kw in df_perf.to_dict("records")
        ), "memory_usage_mb absent"
        mock_write_image.assert_called()
        mock_telegram.assert_called()


def test_create_equity_fig(tmp_dirs, mock_dependencies, mock_psutil):
    """Teste la génération du graphique de la courbe d'equity."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram, patch(
        "plotly.io.write_image"
    ) as mock_write_image:
        dashboard = MIADashboard(config_path=tmp_dirs["config_path"])
        output_path = str(Path(tmp_dirs["figures_dir"]) / "equity_test.png")
        fig_path = dashboard.create_equity_fig(
            tmp_dirs["trades_simulated_path"], output_path=output_path
        )
        assert fig_path == output_path, "Chemin de la figure incorrect"
        assert os.path.exists(output_path), "Figure equity non générée"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "create_equity_fig" in str(op) for op in df_perf["operation"]
        ), "Opération create_equity_fig non journalisée"
        assert any(
            "memory_usage_mb" in str(kw) for kw in df_perf.to_dict("records")
        ), "memory_usage_mb absent"
        mock_write_image.assert_called()
        mock_telegram.assert_called()


def test_get_metrics_summary(tmp_dirs, mock_dependencies, mock_psutil):
    """Teste le calcul des métriques de trading."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        dashboard = MIADashboard(config_path=tmp_dirs["config_path"])
        summary = dashboard.get_metrics_summary(tmp_dirs["trades_simulated_path"])
        assert "Trades: 100" in summary, "Résumé incorrect"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "get_metrics_summary" in str(op) for op in df_perf["operation"]
        ), "Opération get_metrics_summary non journalisée"
        assert any(
            "memory_usage_mb" in str(kw) for kw in df_perf.to_dict("records")
        ), "memory_usage_mb absent"
        mock_telegram.assert_called()


def test_plot_regime_probs(tmp_dirs, mock_dependencies, mock_psutil):
    """Teste la génération du graphique des probabilités de régimes."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        dashboard = MIADashboard(config_path=tmp_dirs["config_path"])
        regime_probs = pd.Series({"trend": 0.7, "range": 0.2, "defensive": 0.1})
        output_path = os.path.join(tmp_dirs["figures_dir"], "regime_probs_test.png")
        dashboard.plot_regime_probs(regime_probs, output_path)
        assert os.path.exists(output_path), "Graphique regime_probs non généré"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "plot_regime_probs" in str(op) for op in df_perf["operation"]
        ), "Opération plot_regime_probs non journalisée"
        assert any(
            "memory_usage_mb" in str(kw) for kw in df_perf.to_dict("records")
        ), "memory_usage_mb absent"
        mock_telegram.assert_called()


def test_plot_shap_features(tmp_dirs, mock_dependencies, mock_psutil):
    """Teste la génération du graphique des importances SHAP."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        dashboard = MIADashboard(config_path=tmp_dirs["config_path"])
        shap_data = pd.read_csv(tmp_dirs["feature_importance_path"]).head(150)
        output_path = os.path.join(tmp_dirs["figures_dir"], "shap_test.png")
        dashboard.plot_shap_features(shap_data, output_path)
        assert os.path.exists(output_path), "Graphique SHAP non généré"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "plot_shap_features" in str(op) for op in df_perf["operation"]
        ), "Opération plot_shap_features non journalisée"
        assert any(
            "memory_usage_mb" in str(kw) for kw in df_perf.to_dict("records")
        ), "memory_usage_mb absent"
        mock_telegram.assert_called()


def test_update_dashboard_cache_hit(tmp_dirs, mock_dependencies, mock_psutil):
    """Teste la récupération des données depuis le cache."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        dashboard = MIADashboard(config_path=tmp_dirs["config_path"])
        cache_data = {
            "regime_text": "Régime: TREND | Neural: trend | Step: 99",
            "regime_fig_path": str(Path(tmp_dirs["figures_dir"]) / "regime_test.png"),
            "feature_fig_path": str(Path(tmp_dirs["figures_dir"]) / "feature_test.png"),
            "equity_fig_path": str(Path(tmp_dirs["figures_dir"]) / "equity_test.png"),
            "metrics_summary": "Trades: 100 | Taux de succès: 50.0% | Retour total: 0.00",
            "regime_probs_img": str(
                Path(tmp_dirs["figures_dir"]) / "regime_probs_test.png"
            ),
            "shap_img": str(Path(tmp_dirs["figures_dir"]) / "shap_test.png"),
            "confidence_text": "Confidence Drop Rate: 0.20 | Nouvelles Features: bid_ask_imbalance, trade_aggressiveness, iv_skew, iv_term_structure, option_skew, news_impact_score",
        }
        cache_key = f"dashboard_{hashlib.sha256(str([tmp_dirs['features_latest_path'], tmp_dirs['trades_simulated_path'], tmp_dirs['regime_history_path'], tmp_dirs['feature_importance_path'], ['close', 'predicted_vix']]).encode()).hexdigest()}"
        dashboard.cache[cache_key] = {"data": cache_data, "timestamp": time.time()}
        result = update_dashboard(
            n=1,
            selected_features=["close", "predicted_vix"],
            dashboard_instance=dashboard,
            features_path=tmp_dirs["features_latest_path"],
            trades_path=tmp_dirs["trades_simulated_path"],
            regime_path=tmp_dirs["regime_history_path"],
            shap_path=tmp_dirs["feature_importance_path"],
            config_path=tmp_dirs["config_path"],
            dashboard_json_path=tmp_dirs["dashboard_json_path"],
        )
        assert result[0] == cache_data["regime_text"], "Cache non utilisé"
        assert result[8] == cache_data["confidence_text"], "confidence_text incorrect"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "update_dashboard_cache_hit" in str(op) for op in df_perf["operation"]
        ), "Opération update_dashboard_cache_hit non journalisée"
        assert any(
            "memory_usage_mb" in str(kw) for kw in df_perf.to_dict("records")
        ), "memory_usage_mb absent"
        mock_telegram.assert_called()


def test_save_snapshot_compressed(tmp_dirs, mock_dependencies, mock_psutil):
    """Teste la sauvegarde d’un snapshot compressé."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        dashboard = MIADashboard(config_path=tmp_dirs["config_path"])
        snapshot_data = {"test": "compressed_snapshot"}
        dashboard.save_snapshot("test_compressed", snapshot_data, compress=True)
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
        assert any(
            "memory_usage_mb" in str(kw) for kw in df_perf.to_dict("records")
        ), "memory_usage_mb absent"
        mock_telegram.assert_called()


def test_checkpoint(tmp_dirs, mock_dependencies, mock_psutil):
    """Teste la sauvegarde incrémentielle."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        dashboard = MIADashboard(config_path=tmp_dirs["config_path"])
        df = pd.read_csv(tmp_dirs["features_latest_path"])
        dashboard.checkpoint(df, data_type="features")
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "dashboard_features" in f and f.endswith(".json.gz")
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
        assert any(
            "memory_usage_mb" in str(kw) for kw in df_perf.to_dict("records")
        ), "memory_usage_mb absent"
        mock_telegram.assert_called()


def test_cloud_backup(tmp_dirs, mock_dependencies, mock_psutil):
    """Teste la sauvegarde distribuée S3."""
    with patch("boto3.client") as mock_s3, patch("os.environ.get") as mock_env, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        mock_env.side_effect = lambda key: None  # Simuler absence de clés AWS
        dashboard = MIADashboard(config_path=tmp_dirs["config_path"])
        df = pd.read_csv(tmp_dirs["features_latest_path"])
        dashboard.cloud_backup(df, data_type="features")
        assert not mock_s3.called, "Client S3 appelé malgré l'absence de clés"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "cloud_backup" in str(op) for op in df_perf["operation"]
        ), "Opération cloud_backup non journalisée"
        assert any(
            "Clés AWS non trouvées" in str(e) for e in df_perf["error"].dropna()
        ), "Erreur clés AWS non loguée"
        mock_telegram.assert_called()


def test_handle_sigint(tmp_dirs, mock_dependencies, mock_psutil):
    """Teste la gestion SIGINT."""
    with patch("sys.exit") as mock_exit, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        dashboard = MIADashboard(config_path=tmp_dirs["config_path"])
        dashboard.handle_sigint(signal.SIGINT, None)
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_sigint" in f for f in snapshot_files
        ), "Snapshot sigint non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "handle_sigint" in str(op) for op in df_perf["operation"]
        ), "Opération handle_sigint non journalisée"
        assert any(
            "memory_usage_mb" in str(kw) for kw in df_perf.to_dict("records")
        ), "memory_usage_mb absent"
        mock_exit.assert_called_with(0)
        mock_telegram.assert_called()


def test_load_dashboard_json(tmp_dirs, mock_dependencies, mock_psutil):
    """Teste le chargement de neural_pipeline_dashboard.json avec cache."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        dashboard = MIADashboard(config_path=tmp_dirs["config_path"])
        data = dashboard.load_dashboard_json(tmp_dirs["dashboard_json_path"])
        assert data["confidence_drop_rate"] == 0.2, "confidence_drop_rate incorrect"
        assert set(data["new_features"]) == {
            "bid_ask_imbalance",
            "trade_aggressiveness",
            "iv_skew",
            "iv_term_structure",
            "option_skew",
            "news_impact_score",
        }, "new_features incorrect"
        # Tester le cache
        data_cached = dashboard.load_dashboard_json(tmp_dirs["dashboard_json_path"])
        assert data_cached == data, "Cache JSON non utilisé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "load_dashboard_json_cache_hit" in str(op) for op in df_perf["operation"]
        ), "Opération load_dashboard_json_cache_hit non journalisée"
        assert any(
            "memory_usage_mb" in str(kw) for kw in df_perf.to_dict("records")
        ), "memory_usage_mb absent"
        mock_telegram.assert_called()


def test_update_dashboard_full(tmp_dirs, mock_dependencies, mock_psutil):
    """Teste la mise à jour complète du tableau de bord."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram, patch(
        "plotly.io.write_image"
    ) as mock_write_image:
        dashboard = MIADashboard(config_path=tmp_dirs["config_path"])
        result = update_dashboard(
            n=1,
            selected_features=["close", "predicted_vix", "bid_ask_imbalance"],
            dashboard_instance=dashboard,
            features_path=tmp_dirs["features_latest_path"],
            trades_path=tmp_dirs["trades_simulated_path"],
            regime_path=tmp_dirs["regime_history_path"],
            shap_path=tmp_dirs["feature_importance_path"],
            config_path=tmp_dirs["config_path"],
            dashboard_json_path=tmp_dirs["dashboard_json_path"],
        )
        assert "Régime: TREND" in result[0], "regime_text incorrect"
        assert result[1].endswith(".png"), "regime_fig_path incorrect"
        assert result[2].endswith(".png"), "feature_fig_path incorrect"
        assert result[3].endswith(".png"), "equity_fig_path incorrect"
        assert "Trades: 100" in result[4], "metrics_summary incorrect"
        assert result[5] == "", "error_output non vide"
        assert result[6].endswith(".png"), "regime_probs_img incorrect"
        assert result[7].endswith(".png"), "shap_img incorrect"
        assert "Confidence Drop Rate: 0.20" in result[8], "confidence_text incorrect"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "update_dashboard" in str(op) for op in df_perf["operation"]
        ), "Opération update_dashboard non journalisée"
        assert any(
            "confidence_drop_rate" in str(kw) for kw in df_perf.to_dict("records")
        ), "confidence_drop_rate absent"
        assert any(
            "memory_usage_mb" in str(kw) for kw in df_perf.to_dict("records")
        ), "memory_usage_mb absent"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_update_dashboard" in f for f in snapshot_files
        ), "Snapshot update_dashboard non créé"
        mock_write_image.assert_called()
        mock_telegram.assert_called()


def test_layout_structure(tmp_dirs, mock_dependencies, mock_psutil):
    """Teste la structure du layout Dash."""
    dashboard = MIADashboard(config_path=tmp_dirs["config_path"])
    layout = app.layout
    assert isinstance(layout, html.Div), "Layout n’est pas un Div"
    children = layout.children
    assert any(isinstance(c, html.H1) for c in children), "H1 manquant"
    assert any(isinstance(c, dcc.Interval) for c in children), "Interval manquant"
    assert any(
        isinstance(c, html.Div) and "Régime du Marché" in str(c) for c in children
    ), "Section Régime manquante"
    assert any(
        isinstance(c, html.Div) and "Données de Trading" in str(c) for c in children
    ), "Section Features manquante"
    assert any(
        isinstance(c, html.Div) and "Performance du Trading" in str(c) for c in children
    ), "Section Performance manquante"
    assert any(
        isinstance(c, html.Div) and "Importance des Features SHAP" in str(c)
        for c in children
    ), "Section SHAP manquante"
    assert any(
        isinstance(c, html.Div) and "Métriques de Confiance" in str(c) for c in children
    ), "Section Confidence manquante"


def test_no_obsolete_references(tmp_dirs):
    """Vérifie l'absence de références à dxFeed, 320/81 features."""
    with open(tmp_dirs["config_path"], "r") as f:
        content = f.read()
    assert "dxFeed" not in content, "Référence à dxFeed trouvée"
    assert "320 features" not in content, "Référence à 320 features trouvée"
    assert "81 features" not in content, "Référence à 81 features trouvée"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
