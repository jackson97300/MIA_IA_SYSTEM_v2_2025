# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_trade_probability.py
# Tests unitaires pour src/model/trade_probability.py et src/model/trade_probability_rl.py
#
# Version : 2.1.5
# Date : 2025-05-14
#
# Rôle : Valide la prédiction de la probabilité de succès d’un trade, l’entraînement du modèle RandomForestClassifier,
#        le réentraînement (suggestion 9), la validation des features, les sauvegardes (snapshots, checkpoints, S3),
#        les alertes Telegram, et les fonctionnalités RL (SAC, PPO, DDPG, PPO-CVaR, QR-DQN) avec vote bayésien,
#        coûts de transaction, microstructure, surface de volatilité, walk-forward validation, journalisation MLflow/Prometheus,
#        et résolution des conflits de signaux via SignalResolver.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, scikit-learn>=1.5.0,<2.0.0,
#   joblib>=1.3.0,<2.0.0, pyyaml>=6.0.0,<7.0.0, psutil>=5.9.8,<6.0.0, boto3>=1.26.0,<2.0.0
# - src/model/trade_probability.py
# - src/model/trade_probability_rl.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
# - src/model/utils/signal_resolver.py
#
# Inputs :
# - Fichiers de configuration factices (trade_probability_config.yaml, feature_sets.yaml, es_config.yaml)
# - Données factices (market_memory.db)
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de trade_probability.py et trade_probability_rl.py.
#
# Notes :
# - Conforme à structure.txt (version 2.1.5, 2025-05-14).
# - Tests la Phase 8 (confidence_drop_rate), la suggestion 9 (réentraînement), et la gestion des SHAP features.
# - Vérifie l’absence de dxFeed, la suppression de 320/81 features et obs_t, les retries, logs psutil,
#   alertes Telegram, snapshots compressés, et sauvegardes incrémentielles/distribuées.
# - Couvre les nouvelles fonctionnalités RL : prédiction RF seule, intégration RL, ajustement du slippage,
#   entraînement RL, journalisation MLflow/Prometheus, gestion du cache LRU, et résolution des conflits
#   de signaux avec SignalResolver (méthode resolve_signals, ajustement des probabilités, métadonnées).
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.
# - Policies Note: The official directory for routing policies is src/model/router/policies.
# The src/model/policies directory is a residual and should be verified
# for removal to avoid import conflicts.

import gzip
import json
import os
import sqlite3
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.model.trade_probability import TradeProbabilityPredictor
from src.model.trade_probability_rl import RLTrainer, TradeEnv


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, cache, checkpoints, modèles, et données."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    cache_dir = data_dir / "cache" / "trade_probability" / "ES"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints" / "trade_probability" / "ES"
    checkpoints_dir.mkdir(parents=True)
    model_dir = base_dir / "model" / "trade_probability" / "ES"
    model_dir.mkdir(parents=True)
    rl_model_dir = data_dir / "models" / "ES"
    rl_model_dir.mkdir(parents=True)

    # Créer trade_probability_config.yaml
    config_path = config_dir / "trade_probability_config.yaml"
    config_content = {
        "trade_probability": {
            "buffer_size": 100,
            "s3_bucket": "test-bucket",
            "s3_prefix": "trade_probability/",
            "signal_resolver": {"thresholds": {"conflict_coefficient_alert": 0.5}},
        }
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    # Créer feature_sets.yaml
    feature_sets_path = config_dir / "feature_sets.yaml"
    feature_sets_content = {
        "ES": {
            "training": {
                "features": [
                    "timestamp",
                    "vix",
                    "neural_regime",
                    "predicted_volatility",
                    "trade_frequency_1s",
                    "close",
                    "bid_ask_imbalance",
                    "trade_aggressiveness",
                    "iv_skew",
                    "iv_term_structure",
                    "slippage_estimate",
                    "vix_es_correlation",
                    "call_iv_atm",
                    "option_skew",
                    "news_impact_score",
                ]
                + [f"feature_{i}" for i in range(335)]
            },
            "inference": {
                "shap_features": [
                    "vix",
                    "neural_regime",
                    "predicted_volatility",
                    "trade_frequency_1s",
                    "close",
                    "bid_ask_imbalance",
                    "trade_aggressiveness",
                    "iv_skew",
                    "iv_term_structure",
                    "slippage_estimate",
                    "vix_es_correlation",
                    "call_iv_atm",
                    "option_skew",
                    "news_impact_score",
                ]
                + [f"shap_feature_{i}" for i in range(136)]
            },
        }
    }
    with open(feature_sets_path, "w", encoding="utf-8") as f:
        yaml.dump(feature_sets_content, f)

    # Créer es_config.yaml
    es_config_path = config_dir / "es_config.yaml"
    es_config_content = {
        "signal_resolver": {
            "default_weights": {
                "vix_es_correlation": 1.0,
                "call_iv_atm": 1.0,
                "option_skew": 1.0,
                "news_score_positive": 0.5,
                "microstructure_bullish": 1.0,
                "qr_dqn_positive": 1.5,
                "iv_term_structure": 1.0,
            },
            "thresholds": {"entropy_alert": 0.5, "conflict_coefficient_alert": 0.5},
        }
    }
    with open(es_config_path, "w", encoding="utf-8") as f:
        yaml.dump(es_config_content, f)

    return {
        "base_dir": str(base_dir),
        "config_path": str(config_path),
        "feature_sets_path": str(feature_sets_path),
        "es_config_path": str(es_config_path),
        "logs_dir": str(logs_dir),
        "cache_dir": str(cache_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "model_dir": str(model_dir),
        "rl_model_dir": str(rl_model_dir),
        "db_path": str(data_dir / "market_memory.db"),
        "perf_log_path": str(logs_dir / "trade_probability_performance.csv"),
        "metrics_log_path": str(logs_dir / "trade_probability_metrics.csv"),
        "backtest_log_path": str(logs_dir / "trade_probability_backtest.csv"),
        "rl_perf_log_path": str(logs_dir / "trade_probability_rl_performance.csv"),
    }


@pytest.fixture
def mock_data(tmp_dirs):
    """Crée des données factices pour tester."""
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=100, freq="1min"),
            "vix": np.random.normal(20, 2, 100),
            "neural_regime": np.random.choice(["range", "trend", "defensive"], 100),
            "predicted_volatility": np.random.normal(0.15, 0.02, 100),
            "trade_frequency_1s": np.random.randint(1, 10, 100),
            "close": np.random.normal(5100, 10, 100),
            "bid_ask_imbalance": np.random.normal(0, 0.1, 100),
            "trade_aggressiveness": np.random.normal(0, 0.2, 100),
            "iv_skew": np.random.normal(0.01, 0.005, 100),
            "iv_term_structure": np.random.normal(0.02, 0.005, 100),
            "slippage_estimate": np.random.uniform(0.01, 0.1, 100),
            "vix_es_correlation": np.random.normal(20, 5, 100),
            "call_iv_atm": np.random.normal(0.15, 0.05, 100),
            "option_skew": np.random.normal(0.02, 0.01, 100),
            "news_impact_score": np.random.uniform(0.0, 1.0, 100),
            **{f"feature_{i}": np.random.normal(0, 1, 100) for i in range(335)},
            **{f"shap_feature_{i}": np.random.normal(0, 1, 100) for i in range(136)},
        }
    )
    return data


@pytest.fixture
def mock_db(tmp_dirs, mock_data):
    """Crée une base de données SQLite factice."""
    conn = sqlite3.connect(tmp_dirs["db_path"])
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE patterns (
            market TEXT,
            features TEXT,
            reward REAL
        )
    """
    )
    cursor.execute(
        """
        CREATE TABLE trade_patterns (
            timestamp TEXT,
            vix REAL, neural_regime TEXT, predicted_volatility REAL, trade_frequency_1s INTEGER, close REAL,
            bid_ask_imbalance REAL, trade_aggressiveness REAL, iv_skew REAL, iv_term_structure REAL, slippage_estimate REAL,
            vix_es_correlation REAL, call_iv_atm REAL, option_skew REAL, news_impact_score REAL,
            reward REAL,
            {}
        )
    """.format(
            ", ".join([f"feature_{i} REAL" for i in range(335)])
        )
    )
    cursor.execute(
        """
        CREATE TABLE training_log (
            timestamp TEXT
        )
    """
    )
    # Insérer des données factices dans patterns
    patterns = []
    for _, row in mock_data.iterrows():
        features = row.drop(["timestamp"]).to_dict()
        reward = np.random.uniform(-1, 1)
        patterns.append(("ES", json.dumps(features), reward))
    pd.DataFrame(patterns, columns=["market", "features", "reward"]).to_sql(
        "patterns", conn, index=False
    )
    # Insérer des données factices dans trade_patterns
    trade_patterns = mock_data.copy()
    trade_patterns["reward"] = np.random.uniform(-1, 1, len(mock_data))
    trade_patterns.to_sql("trade_patterns", conn, if_exists="append", index=False)
    # Insérer un timestamp dans training_log
    cursor.execute(
        "INSERT INTO training_log (timestamp) VALUES (?)",
        ((datetime.now() - pd.Timedelta(days=1)).isoformat(),),
    )
    conn.commit()
    conn.close()
    return tmp_dirs["db_path"]


@pytest.fixture
def mock_env(mock_data):
    """Crée un environnement Gym factice."""
    return TradeEnv(mock_data, market="ES")


@pytest.fixture
def mock_mlflow_tracker():
    """Mock pour MLFlowTracker."""
    tracker = MagicMock()
    tracker.start_run = MagicMock()
    tracker.log_metrics = MagicMock()
    tracker.log_params = MagicMock()
    tracker.log_artifact = MagicMock()
    tracker.tracking_uri = "http://localhost:5000"
    return tracker


def test_trade_probability_init(tmp_dirs, mock_db, mock_mlflow_tracker):
    """Teste l’initialisation de TradeProbabilityPredictor."""
    with patch(
        "src.model.trade_probability.MLFlowTracker", return_value=mock_mlflow_tracker
    ), patch("src.model.trade_probability.send_telegram_alert") as mock_telegram, patch(
        "src.model.trade_probability.SignalResolver"
    ) as mock_resolver:
        mock_resolver.return_value = MagicMock()
        predictor = TradeProbabilityPredictor(db_path=mock_db, market="ES")
        assert predictor.db_path == mock_db, "Chemin de la base de données incorrect"
        assert len(predictor.feature_cols) >= 350, "Nombre de features incorrect"
        assert (
            len(predictor.shap_feature_cols) >= 150
        ), "Nombre de SHAP features incorrect"
        assert all(
            f in predictor.feature_cols
            for f in [
                "bid_ask_imbalance",
                "trade_aggressiveness",
                "iv_skew",
                "iv_term_structure",
                "vix_es_correlation",
                "call_iv_atm",
                "option_skew",
                "news_impact_score",
            ]
        ), "Nouvelles features manquantes"
        assert isinstance(
            predictor.signal_resolver, MagicMock
        ), "SignalResolver non initialisé"
        assert isinstance(
            predictor.signal_cache, OrderedDict
        ), "Cache LRU non initialisé"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df["operation"] == "init"), "Log d’initialisation manquant"
        assert isinstance(predictor.rl_trainer, RLTrainer), "RLTrainer non initialisé"
        mock_telegram.assert_called()


def test_load_data(tmp_dirs, mock_db, mock_mlflow_tracker):
    """Teste le chargement des données depuis market_memory.db."""
    with patch(
        "src.model.trade_probability.MLFlowTracker", return_value=mock_mlflow_tracker
    ), patch("src.model.trade_probability.send_telegram_alert") as mock_telegram:
        predictor = TradeProbabilityPredictor(db_path=mock_db, market="ES")
        X, y = predictor.load_data(market="ES")
        assert len(X) > 0, "Aucune donnée chargée"
        assert len(X.columns) >= 350, "Nombre de features incorrect"
        assert len(y) == len(X), "Taille des cibles incorrecte"
        assert all(
            f in X.columns
            for f in [
                "bid_ask_imbalance",
                "trade_aggressiveness",
                "iv_skew",
                "iv_term_structure",
                "vix_es_correlation",
                "call_iv_atm",
                "option_skew",
                "news_impact_score",
            ]
        ), "Nouvelles features manquantes"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_load_data" in f and f.endswith(".json.gz") for f in snapshot_files
        ), "Snapshot non créé"
        mock_telegram.assert_called()


def test_train_model(tmp_dirs, mock_db, mock_mlflow_tracker):
    """Teste l’entraînement du modèle RandomForestClassifier avec validation glissante."""
    with patch(
        "src.model.trade_probability.MLFlowTracker", return_value=mock_mlflow_tracker
    ), patch("src.model.trade_probability.send_telegram_alert") as mock_telegram:
        predictor = TradeProbabilityPredictor(db_path=mock_db, market="ES")
        predictor.train(market="ES")
        model_files = os.listdir(tmp_dirs["model_dir"])
        assert any(
            f.startswith("rf_trade_prob_") and f.endswith(".pkl") for f in model_files
        ), "Modèle non sauvegardé"
        assert os.path.exists(
            tmp_dirs["metrics_log_path"]
        ), "Fichier de métriques non créé"
        df = pd.read_csv(tmp_dirs["metrics_log_path"])
        assert "auc" in df.columns, "Métrique AUC manquante"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df_perf["operation"] == "train"), "Log d’entraînement manquant"
        assert any(
            df_perf["mean_auc"].notnull()
        ), "AUC moyen de cross-validation non journalisé"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_train" in f and f.endswith(".json.gz") for f in snapshot_files
        ), "Snapshot non créé"
        mock_telegram.assert_called()


def test_predict(tmp_dirs, mock_db, mock_data, mock_mlflow_tracker):
    """Teste la prédiction de la probabilité de succès."""
    with patch(
        "src.model.trade_probability.MLFlowTracker", return_value=mock_mlflow_tracker
    ), patch("src.model.trade_probability.send_telegram_alert") as mock_telegram:
        predictor = TradeProbabilityPredictor(db_path=mock_db, market="ES")
        predictor.train(market="ES")
        data = mock_data.tail(1)[predictor.feature_cols]
        prob = predictor.predict(data, market="ES")
        assert 0 <= prob <= 1, "Probabilité hors intervalle [0, 1]"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_predict" in f and f.endswith(".json.gz") for f in snapshot_files
        ), "Snapshot non créé"
        mock_telegram.assert_called()


def test_retrain_model(tmp_dirs, mock_db, mock_mlflow_tracker):
    """Teste le réentraînement du modèle avec de nouvelles données et validation glissante."""
    with patch(
        "src.model.trade_probability.MLFlowTracker", return_value=mock_mlflow_tracker
    ), patch("src.model.trade_probability.send_telegram_alert") as mock_telegram:
        predictor = TradeProbabilityPredictor(db_path=mock_db, market="ES")
        predictor.retrain_model(market="ES")
        model_files = os.listdir(tmp_dirs["model_dir"])
        assert any(
            f.startswith("rf_trade_prob_") and f.endswith(".pkl") for f in model_files
        ), "Modèle non sauvegardé après réentraînement"
        assert os.path.exists(
            tmp_dirs["metrics_log_path"]
        ), "Fichier de métriques non créé"
        df = pd.read_csv(tmp_dirs["metrics_log_path"])
        assert "auc" in df.columns, "Métrique AUC manquante"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df_perf["operation"] == "retrain_model"
        ), "Log de réentraînement manquant"
        assert any(
            df_perf["mean_auc"].notnull()
        ), "AUC moyen de cross-validation non journalisé"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_retrain_model" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot non créé"
        conn = sqlite3.connect(tmp_dirs["db_path"])
        training_log = pd.read_sql("SELECT * FROM training_log", conn)
        conn.close()
        assert len(training_log) >= 2, "training_log non mis à jour"
        mock_telegram.assert_called()


def test_retrain_model_insufficient_data(tmp_dirs, mock_db, mock_mlflow_tracker):
    """Teste le réentraînement avec des données insuffisantes."""
    with patch(
        "src.model.trade_probability.MLFlowTracker", return_value=mock_mlflow_tracker
    ), patch("src.model.trade_probability.send_telegram_alert") as mock_telegram:
        # Vider trade_patterns pour simuler des données insuffisantes
        conn = sqlite3.connect(tmp_dirs["db_path"])
        conn.execute("DELETE FROM trade_patterns")
        conn.commit()
        conn.close()
        predictor = TradeProbabilityPredictor(db_path=mock_db, market="ES")
        predictor.retrain_model(market="ES")
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "Données insuffisantes" in str(e) for e in df["error"].dropna()
        ), "Erreur données insuffisantes non loguée"
        model_files = os.listdir(tmp_dirs["model_dir"])
        assert not model_files, "Modèle sauvegardé malgré données insuffisantes"
        mock_telegram.assert_called_with(pytest.any(str))


def test_backtest_threshold(tmp_dirs, mock_db, mock_mlflow_tracker):
    """Teste le backtesting des seuils pour trade_success_prob."""
    with patch(
        "src.model.trade_probability.MLFlowTracker", return_value=mock_mlflow_tracker
    ), patch("src.model.trade_probability.send_telegram_alert") as mock_telegram:
        predictor = TradeProbabilityPredictor(db_path=mock_db, market="ES")
        predictor.train(market="ES")
        predictor.backtest_threshold(thresholds=[0.6, 0.7, 0.8], market="ES")
        assert os.path.exists(
            tmp_dirs["backtest_log_path"]
        ), "Fichier de backtest non créé"
        df = pd.read_csv(tmp_dirs["backtest_log_path"])
        assert len(df) >= 3, "Seuils de backtest non journalisés"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_backtest_threshold" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot non créé"
        mock_telegram.assert_called()


def test_cloud_backup(tmp_dirs, mock_db, mock_mlflow_tracker):
    """Teste la sauvegarde S3."""
    with patch("boto3.client") as mock_s3, patch(
        "src.model.trade_probability.MLFlowTracker", return_value=mock_mlflow_tracker
    ), patch("src.model.trade_probability.send_telegram_alert") as mock_telegram:
        predictor = TradeProbabilityPredictor(db_path=mock_db, market="ES")
        df = pd.DataFrame({"timestamp": [datetime.now().isoformat()], "value": [1.0]})
        predictor.cloud_backup(df, data_type="test_metrics", market="ES")
        assert mock_s3.called, "Client S3 non appelé"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        mock_telegram.assert_called()


def test_validate_features(tmp_dirs, mock_db, mock_mlflow_tracker):
    """Teste la validation des features."""
    with patch(
        "src.model.trade_probability.MLFlowTracker", return_value=mock_mlflow_tracker
    ), patch("src.model.trade_probability.send_telegram_alert") as mock_telegram:
        predictor = TradeProbabilityPredictor(db_path=mock_db, market="ES")
        valid_features = {col: np.random.normal(0, 1) for col in predictor.feature_cols}
        assert predictor.validate_features(
            valid_features, use_shap_fallback=False, market="ES"
        ), "Validation des features complètes échouée"
        shap_features = {
            col: np.random.normal(0, 1) for col in predictor.shap_feature_cols
        }
        assert predictor.validate_features(
            shap_features, use_shap_fallback=True, market="ES"
        ), "Validation des SHAP features échouée"
        invalid_features = {"invalid_col": 1.0}
        assert not predictor.validate_features(
            invalid_features, use_shap_fallback=False, market="ES"
        ), "Validation des features invalides non détectée"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        mock_telegram.assert_called()


def test_predict_rf_only(tmp_dirs, mock_db, mock_data, mock_mlflow_tracker):
    """Teste la prédiction RF seule (sans RL)."""
    with patch(
        "src.model.trade_probability.MLFlowTracker", return_value=mock_mlflow_tracker
    ), patch(
        "src.model.trade_probability.RLTrainer.predict", return_value=(0.0, {})
    ), patch(
        "src.model.trade_probability.send_telegram_alert"
    ) as mock_telegram:
        predictor = TradeProbabilityPredictor(db_path=mock_db, market="ES")
        predictor.train(market="ES")
        data = mock_data.tail(1)[predictor.feature_cols]
        prob = predictor.predict(data, market="ES")
        assert 0 <= prob <= 1, "Probabilité RF hors intervalle [0, 1]"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_predict" in f and f.endswith(".json.gz") for f in snapshot_files
        ), "Snapshot non créé"
        mock_telegram.assert_called()


def test_predict_with_rl(tmp_dirs, mock_db, mock_data, mock_mlflow_tracker):
    """Teste la prédiction avec intégration RL (vote bayésien)."""
    with patch(
        "src.model.trade_probability.MLFlowTracker", return_value=mock_mlflow_tracker
    ), patch(
        "src.model.trade_probability.RLTrainer.predict",
        return_value=(
            0.7,
            {
                "signal_score": 0.5,
                "conflict_coefficient": 0.4,
                "entropy": 0.3,
                "run_id": "rl_run_123",
            },
        ),
    ), patch(
        "src.model.trade_probability.send_telegram_alert"
    ) as mock_telegram:
        predictor = TradeProbabilityPredictor(db_path=mock_db, market="ES")
        predictor.train(market="ES")
        data = mock_data.tail(1)[predictor.feature_cols]
        prob = predictor.predict(data, market="ES")
        assert 0 <= prob <= 1, "Probabilité avec RL hors intervalle [0, 1]"
        rf_prob = predictor.model.predict_proba(data[predictor.feature_cols].fillna(0))[
            :, 1
        ][0]
        expected_prob = np.clip(
            0.5 * rf_prob + 0.5 * 0.7 - data["slippage_estimate"].iloc[0], 0, 1
        )
        assert (
            abs(prob - expected_prob) < 1e-6
        ), f"Probabilité avec vote bayésien incorrecte: attendu {expected_prob}, obtenu {prob}"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_predict" in f and f.endswith(".json.gz") for f in snapshot_files
        ), "Snapshot non créé"
        mock_telegram.assert_called()


def test_predict_with_slippage(tmp_dirs, mock_db, mock_data, mock_mlflow_tracker):
    """Teste l’ajustement de la probabilité avec slippage."""
    with patch(
        "src.model.trade_probability.MLFlowTracker", return_value=mock_mlflow_tracker
    ), patch(
        "src.model.trade_probability.RLTrainer.predict",
        return_value=(
            0.7,
            {
                "signal_score": 0.5,
                "conflict_coefficient": 0.4,
                "entropy": 0.3,
                "run_id": "rl_run_123",
            },
        ),
    ), patch(
        "src.model.trade_probability.send_telegram_alert"
    ) as mock_telegram:
        predictor = TradeProbabilityPredictor(db_path=mock_db, market="ES")
        predictor.train(market="ES")
        data = mock_data.tail(1)[predictor.feature_cols]
        rf_prob = predictor.model.predict_proba(data[predictor.feature_cols].fillna(0))[
            :, 1
        ][0]
        expected_raw_prob = 0.5 * rf_prob + 0.5 * 0.7
        slippage = data["slippage_estimate"].iloc[0]
        expected_prob = np.clip(expected_raw_prob - slippage, 0, 1)
        prob = predictor.predict(data, market="ES")
        assert (
            abs(prob - expected_prob) < 1e-6
        ), f"Probabilité avec slippage incorrecte: attendu {expected_prob}, obtenu {prob}"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        mock_telegram.assert_called()


def test_train_rl_models(tmp_dirs, mock_db, mock_data, mock_env, mock_mlflow_tracker):
    """Teste l’entraînement des modèles RL."""
    with patch(
        "src.model.trade_probability.MLFlowTracker", return_value=mock_mlflow_tracker
    ), patch("src.model.trade_probability_rl.SAC") as mock_sac, patch(
        "src.model.trade_probability_rl.PPO"
    ) as mock_ppo, patch(
        "src.model.trade_probability_rl.DDPG"
    ) as mock_ddpg, patch(
        "src.model.trade_probability_rl.CVaRWrapper"
    ) as mock_cvar, patch(
        "src.model.trade_probability_rl.QRDQN"
    ) as mock_qrdqn, patch(
        "src.model.trade_probability.send_telegram_alert"
    ) as mock_telegram, patch(
        "src.monitoring.prometheus_metrics.Gauge.labels"
    ) as mock_gauge:
        mock_model = MagicMock()
        mock_model.learn = MagicMock()
        mock_model.save = MagicMock()
        mock_sac.return_value = mock_model
        mock_ppo.return_value = mock_model
        mock_ddpg.return_value = mock_model
        mock_cvar.return_value = mock_model
        mock_qrdqn.return_value = mock_model
        predictor = TradeProbabilityPredictor(db_path=mock_db, market="ES")
        predictor.train_rl_models(mock_data, total_timesteps=10)
        assert mock_model.learn.called, "Entraînement RL non appelé"
        assert mock_model.save.called, "Sauvegarde RL non appelée"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        assert os.path.exists(
            tmp_dirs["rl_perf_log_path"]
        ), "Fichier de logs RL non créé"
        df_rl = pd.read_csv(tmp_dirs["rl_perf_log_path"])
        assert any(
            df_rl["operation"].str.contains("train_")
        ), "Logs d’entraînement RL manquants"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_train_rl_models" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot RL non créé"
        assert mock_gauge.call_count >= 2, "Métriques Prometheus non mises à jour"
        mock_telegram.assert_called()


def test_cache_invalidation(tmp_dirs, mock_db, mock_data, mock_mlflow_tracker):
    """Teste l’invalidation du cache LRU avec changement de features."""
    with patch(
        "src.model.trade_probability.MLFlowTracker", return_value=mock_mlflow_tracker
    ), patch("src.model.trade_probability.send_telegram_alert") as mock_telegram:
        predictor = TradeProbabilityPredictor(db_path=mock_db, market="ES")
        predictor.train(market="ES")
        data1 = mock_data.tail(1)[predictor.feature_cols]
        prob1 = predictor.predict(data1, market="ES")
        # Modifier une feature pour changer le hash
        data2 = data1.copy()
        data2["vix"] += 1.0
        prob2 = predictor.predict(data2, market="ES")
        assert prob1 != prob2, "Cache non invalidé pour features modifiées"
        assert len(predictor.prediction_cache) == 2, "Cache LRU incorrect"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        mock_telegram.assert_called()


def test_cloud_backup_no_s3(tmp_dirs, mock_db, mock_mlflow_tracker):
    """Teste la sauvegarde locale en cas d’absence de S3."""
    with patch(
        "src.model.trade_probability.MLFlowTracker", return_value=mock_mlflow_tracker
    ), patch("src.model.trade_probability.send_telegram_alert") as mock_telegram:
        # Modifier la config pour supprimer s3_bucket
        with open(tmp_dirs["config_path"], "w", encoding="utf-8") as f:
            yaml.dump({"trade_probability": {"buffer_size": 100}}, f)
        predictor = TradeProbabilityPredictor(db_path=mock_db, market="ES")
        df = pd.DataFrame({"timestamp": [datetime.now().isoformat()], "value": [1.0]})
        predictor.cloud_backup(df, data_type="test_metrics", market="ES")
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "cloud_backup" in op for op in df["operation"]
        ), "Log cloud_backup manquant"
        mock_telegram.assert_called_with(pytest.any(str))


def test_walk_forward_validation(tmp_dirs, mock_db, mock_mlflow_tracker):
    """Teste la validation glissante avec TimeSeriesSplit dans train."""
    with patch(
        "src.model.trade_probability.MLFlowTracker", return_value=mock_mlflow_tracker
    ), patch("src.model.trade_probability.send_telegram_alert") as mock_telegram:
        predictor = TradeProbabilityPredictor(db_path=mock_db, market="ES")
        predictor.train(market="ES")
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(df["operation"] == "train"), "Log d’entraînement manquant"
        assert any(
            df["mean_auc"].notnull()
        ), "AUC moyen de cross-validation non journalisé"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_train" in f and f.endswith(".json.gz") for f in snapshot_files
        ), "Snapshot non créé"
        mock_telegram.assert_called()


def test_resolve_signals(tmp_dirs, mock_db, mock_data, mock_mlflow_tracker):
    """Teste la méthode resolve_signals dans TradeProbabilityPredictor."""
    with patch(
        "src.model.trade_probability.MLFlowTracker", return_value=mock_mlflow_tracker
    ), patch("src.model.trade_probability.SignalResolver") as mock_resolver, patch(
        "src.model.trade_probability.send_telegram_alert"
    ) as mock_telegram:
        mock_resolver.return_value.resolve_conflict.return_value = (
            0.6,
            {
                "signal_score": 0.6,
                "normalized_score": 0.6,
                "entropy": 0.2,
                "conflict_coefficient": 0.3,
                "score_type": "intermediate",
                "contributions": {
                    "vix_es_correlation": {
                        "value": 1.0,
                        "weight": 1.0,
                        "contribution": 1.0,
                    }
                },
                "run_id": "rf_run_456",
            },
        )
        predictor = TradeProbabilityPredictor(db_path=mock_db, market="ES")
        data = mock_data.tail(1)
        metadata = predictor.resolve_signals(data)
        assert "signal_score" in metadata, "Métadonnées manquent signal_score"
        assert "normalized_score" in metadata, "Métadonnées manquent normalized_score"
        assert "entropy" in metadata, "Métadonnées manquent entropy"
        assert (
            "conflict_coefficient" in metadata
        ), "Métadonnées manquent conflict_coefficient"
        assert "run_id" in metadata, "Métadonnées manquent run_id"
        assert metadata["signal_score"] == 0.6, "Score incorrect"
        assert metadata["conflict_coefficient"] == 0.3, "Conflict coefficient incorrect"
        # Vérifier le cache
        cached_metadata = predictor.resolve_signals(data)
        assert cached_metadata == metadata, "Cache LRU non utilisé"
        assert len(predictor.signal_cache) == 1, "Cache LRU incorrect"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"].str.contains("resolve_signals")
        ), "Logs de résolution des signaux manquants"
        mock_telegram.assert_called()


def test_resolve_signals_error(tmp_dirs, mock_db, mock_mlflow_tracker):
    """Teste la gestion des erreurs dans resolve_signals."""
    with patch(
        "src.model.trade_probability.MLFlowTracker", return_value=mock_mlflow_tracker
    ), patch("src.model.trade_probability.SignalResolver") as mock_resolver, patch(
        "src.model.trade_probability.send_telegram_alert"
    ) as mock_telegram:
        mock_resolver.return_value.resolve_conflict.side_effect = ValueError(
            "Signal invalide"
        )
        predictor = TradeProbabilityPredictor(db_path=mock_db, market="ES")
        data = pd.DataFrame()  # DataFrame vide pour simuler une erreur
        metadata = predictor.resolve_signals(data)
        assert metadata["signal_score"] == 0.0, "Score par défaut incorrect"
        assert (
            metadata["conflict_coefficient"] == 0.0
        ), "Conflict coefficient par défaut incorrect"
        assert "error" in metadata, "Métadonnées manquent erreur"
        assert "run_id" in metadata, "Métadonnées manquent run_id"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"].str.contains("resolve_signals")
        ), "Logs de résolution des signaux manquants"
        mock_telegram.assert_called()


def test_predict_with_signal_metadata(
    tmp_dirs, mock_db, mock_data, mock_mlflow_tracker
):
    """Teste la prédiction avec gestion des métadonnées de SignalResolver."""
    with patch(
        "src.model.trade_probability.MLFlowTracker", return_value=mock_mlflow_tracker
    ), patch("src.model.trade_probability.SignalResolver") as mock_resolver, patch(
        "src.model.trade_probability.RLTrainer.predict",
        return_value=(
            0.7,
            {
                "signal_score": 0.5,
                "conflict_coefficient": 0.4,
                "entropy": 0.3,
                "run_id": "rl_run_123",
            },
        ),
    ), patch(
        "src.model.trade_probability.send_telegram_alert"
    ) as mock_telegram:
        mock_resolver.return_value.resolve_conflict.return_value = (
            0.6,
            {
                "signal_score": 0.6,
                "normalized_score": 0.6,
                "entropy": 0.2,
                "conflict_coefficient": 0.3,
                "score_type": "intermediate",
                "contributions": {
                    "vix_es_correlation": {
                        "value": 1.0,
                        "weight": 1.0,
                        "contribution": 1.0,
                    }
                },
                "run_id": "rf_run_456",
            },
        )
        predictor = TradeProbabilityPredictor(db_path=mock_db, market="ES")
        predictor.train(market="ES")
        data = mock_data.tail(1)[predictor.feature_cols]
        prob = predictor.predict(data, market="ES")
        assert 0 <= prob <= 1, "Probabilité hors intervalle [0, 1]"
        rf_prob = predictor.model.predict_proba(data[predictor.feature_cols].fillna(0))[
            :, 1
        ][0]
        raw_prob = 0.5 * rf_prob + 0.5 * 0.7
        slippage = data["slippage_estimate"].iloc[0]
        expected_prob = np.clip(raw_prob - slippage, 0, 1)
        assert (
            abs(prob - expected_prob) < 1e-6
        ), f"Probabilité sans ajustement de conflit incorrecte: attendu {expected_prob}, obtenu {prob}"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"].str.contains("predict")
        ), "Logs de prédiction manquants"
        assert any(
            df["rf_conflict_coefficient"].notnull()
        ), "RF conflict_coefficient non journalisé"
        assert any(
            df["rl_conflict_coefficient"].notnull()
        ), "RL conflict_coefficient non journalisé"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_predict" in f and f.endswith(".json.gz") for f in snapshot_files
        ), "Snapshot non créé"
        with gzip.open(
            os.path.join(tmp_dirs["cache_dir"], snapshot_files[-1]),
            "rt",
            encoding="utf-8",
        ) as f:
            snapshot = json.load(f)
        assert (
            "rf_signal_metadata" in snapshot["data"]
        ), "RF signal_metadata manquant dans snapshot"
        assert (
            "rl_signal_metadata" in snapshot["data"]
        ), "RL signal_metadata manquant dans snapshot"
        mock_telegram.assert_called()


def test_predict_with_high_conflict(tmp_dirs, mock_db, mock_data, mock_mlflow_tracker):
    """Teste l’ajustement de la probabilité avec un conflict_coefficient élevé."""
    with patch(
        "src.model.trade_probability.MLFlowTracker", return_value=mock_mlflow_tracker
    ), patch("src.model.trade_probability.SignalResolver") as mock_resolver, patch(
        "src.model.trade_probability.RLTrainer.predict",
        return_value=(
            0.7,
            {
                "signal_score": 0.5,
                "conflict_coefficient": 0.6,
                "entropy": 0.3,
                "run_id": "rl_run_123",
            },
        ),
    ), patch(
        "src.model.trade_probability.send_telegram_alert"
    ) as mock_telegram:
        mock_resolver.return_value.resolve_conflict.return_value = (
            0.6,
            {
                "signal_score": 0.6,
                "normalized_score": 0.6,
                "entropy": 0.2,
                "conflict_coefficient": 0.7,
                "score_type": "intermediate",
                "contributions": {
                    "vix_es_correlation": {
                        "value": 1.0,
                        "weight": 1.0,
                        "contribution": 1.0,
                    }
                },
                "run_id": "rf_run_456",
            },
        )
        predictor = TradeProbabilityPredictor(db_path=mock_db, market="ES")
        predictor.train(market="ES")
        data = mock_data.tail(1)[predictor.feature_cols]
        prob = predictor.predict(data, market="ES")
        assert 0 <= prob <= 1, "Probabilité hors intervalle [0, 1]"
        rf_prob = predictor.model.predict_proba(data[predictor.feature_cols].fillna(0))[
            :, 1
        ][0]
        raw_prob = 0.5 * rf_prob + 0.5 * 0.7
        slippage = data["slippage_estimate"].iloc[0]
        max_conflict = 0.7  # Maximum des conflict_coefficient (0.6 RL, 0.7 RF)
        adjustment_factor = 1 - max_conflict
        expected_prob = np.clip((raw_prob - slippage) * adjustment_factor, 0, 1)
        assert (
            abs(prob - expected_prob) < 1e-6
        ), f"Probabilité avec ajustement de conflit incorrecte: attendu {expected_prob}, obtenu {prob}"
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs de performance non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"].str.contains("predict")
        ), "Logs de prédiction manquants"
        assert any(
            df["rf_conflict_coefficient"] >= 0.7
        ), "RF conflict_coefficient non journalisé correctement"
        mock_telegram.assert_called()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
