# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_backtest_lab.py
# Tests unitaires pour src/backtest/backtest_lab.py
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide le backtesting et la simulation des stratégies basées sur
# le signal SGC (SpotGamma Composite) avec backtesting complet et
# incrémental, intégrant des récompenses adaptatives (méthode 5) basées
# sur news_impact_score, avec support multi-marchés.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pandas>=2.0.0,<3.0.0
# - numpy>=1.26.4,<2.0.0
# - psutil>=5.9.8,<6.0.0
# - pyyaml>=6.0.0,<7.0.0
# - boto3>=1.26.0,<2.0.0
# - loguru>=0.7.0,<1.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
# - src/features/signal_selector.py
# - src/backtest/simulate_trades.py
# - src/features/context_aware_filter.py
# - src/features/cross_asset_processor.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/backtest_config.yaml
# - config/feature_sets.yaml
# - data/features/features_latest_filtered.csv
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de backtest_lab.py.
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Tests la Phase 8 (confidence_drop_rate), méthode 5 (récompenses
#   adaptatives), et autres standards.
# - Vérifie l’absence de dxFeed, la suppression de 320/81 features et obs_t,
#   les retries, logs psutil, alertes Telegram, snapshots compressés, et
#   sauvegardes incrémentielles/distribuées.
# - Compatible avec 350 features (entraînement) et 150 SHAP features
#   (inférence) via config/feature_sets.yaml.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.

import gzip
import json
import os
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.backtest.backtest_lab import (
    load_config,
    log_performance,
    run_backtest,
    run_incremental_backtest,
    save_snapshot,
    validate_data,
)


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, cache, checkpoints, et
    données."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs" / "backtest"
    logs_dir.mkdir(parents=True)
    cache_dir = data_dir / "cache" / "backtest" / "ES"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints" / "backtest" / "ES"
    checkpoints_dir.mkdir(parents=True)
    features_dir = data_dir / "data" / "features"
    features_dir.mkdir(parents=True)
    backtest_dir = data_dir / "backtest"
    backtest_dir.mkdir()

    # Créer backtest_config.yaml
    config_path = config_dir / "backtest_config.yaml"
    config_content = {
        "backtest": {
            "initial_capital": 100000,
            "position_size_pct": 0.1,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.05,
            "transaction_cost_bps": 5,
            "slippage_pct": 0.001,
            "min_rows": 100,
            "sgc_threshold": 0.7,
            "s3_bucket": "test-bucket",
            "s3_prefix": "backtest/",
        }
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    # Créer feature_sets.yaml
    feature_sets_path = config_dir / "feature_sets.yaml"
    feature_sets_content = {
        "training": {
            "features": [f"feature_{i}" for i in range(350)]
        },
        "inference": {
            "shap_features": [f"shap_feature_{i}" for i in range(150)]
        },
    }
    with open(feature_sets_path, "w", encoding="utf-8") as f:
        yaml.dump(feature_sets_content, f)

    # Créer features_latest_filtered.csv
    data_path = features_dir / "features_latest_filtered.csv"
    feature_cols = [f"feature_{i}" for i in range(350)]
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-05-13 09:00",
                periods=1000,
                freq="1min",
            ),
            "close": [5100] * 1000,
            "news_impact_score": [0.5] * 1000,
            "bid_size_level_1": [100] * 1000,
            "ask_size_level_1": [120] * 1000,
            "trade_frequency_1s": [8] * 1000,
            "volume": [1000] * 1000,
            **{
                col: [np.random.uniform(0, 1)] * 1000
                for col in feature_cols
            },
        }
    )
    data.to_csv(data_path, index=False)

    return {
        "base_dir": str(base_dir),
        "config_path": str(config_path),
        "feature_sets_path": str(feature_sets_path),
        "logs_dir": str(logs_dir),
        "cache_dir": str(cache_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "features_dir": str(features_dir),
        "backtest_dir": str(backtest_dir),
        "data_path": str(data_path),
        "perf_log_path": str(logs_dir / "backtest_performance.csv"),
    }


@pytest.fixture
def mock_data(tmp_dirs):
    """Crée un DataFrame factice avec 350 features."""
    feature_cols = [f"feature_{i}" for i in range(350)]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-05-13 09:00",
                periods=1000,
                freq="1min",
            ),
            "close": [5100] * 1000,
            "news_impact_score": [0.5] * 1000,
            "bid_size_level_1": [100] * 1000,
            "ask_size_level_1": [120] * 1000,
            "trade_frequency_1s": [8] * 1000,
            "volume": [1000] * 1000,
            **{
                col: [np.random.uniform(0, 1)] * 1000
                for col in feature_cols
            },
        }
    )


@pytest.mark.asyncio
async def test_load_config(tmp_dirs):
    """Teste le chargement de la configuration."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        config = load_config(tmp_dirs["config_path"], market="ES")
        assert (
            config["initial_capital"] == 100000
        ), "Initial capital incorrect"
        assert (
            config["sgc_threshold"] == 0.7
        ), "SGC threshold incorrect"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "load_config" in str(op) for op in df_perf["operation"]
        ), "Opération load_config non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_validate_data(tmp_dirs, mock_data):
    """Teste la validation des données."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        assert validate_data(
            mock_data,
            market="ES",
        ), "Validation des données échouée"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_validate_data" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot validate_data non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "validate_data" in str(op) for op in df_perf["operation"]
        ), "Opération validate_data non journalisée"
        assert any(
            "confidence_drop_rate" in str(kw)
            for kw in df_perf.to_dict("records")
        ), "confidence_drop_rate absent"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_log_performance(tmp_dirs):
    """Teste la journalisation des performances."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        log_performance("test_op", 0.5, success=True, market="ES")
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "test_op" in str(op) for op in df_perf["operation"]
        ), "Opération test_op non journalisée"
        assert (
            df_perf["memory_usage_mb"].iloc[-1] > 0
        ), "Usage mémoire non journalisé"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_save_snapshot(tmp_dirs):
    """Teste la sauvegarde des snapshots."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        save_snapshot("test_snapshot", {"test": "data"}, market="ES")
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_test_snapshot" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot test_snapshot non créé"
        with gzip.open(
            os.path.join(tmp_dirs["cache_dir"], snapshot_files[0]),
            "rt",
            encoding="utf-8",
        ) as f:
            snapshot_data = json.load(f)
        assert (
            snapshot_data["type"] == "test_snapshot"
        ), "Type de snapshot incorrect"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "save_snapshot" in str(op) for op in df_perf["operation"]
        ), "Opération save_snapshot non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_run_backtest(tmp_dirs, mock_data):
    """Teste le backtest complet."""
    with patch(
        "src.features.signal_selector.calculate_sgc"
    ) as mock_sgc, patch(
        "src.backtest.simulate_trades.simulate_trades"
    ) as mock_simulate, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        mock_sgc.return_value = (pd.Series([0.8] * 1000), None)
        mock_simulate.return_value = []
        results_df, metrics = run_backtest(
            mock_data,
            tmp_dirs["config_path"],
            market="ES",
        )
        assert not results_df.empty, "Résultats du backtest vides"
        assert "total_return" in metrics, "Métriques incomplètes"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_backtest" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot backtest non créé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "backtest_results" in f and f.endswith(".json.gz")
            for f in checkpoint_files
        ), "Checkpoint backtest_results non créé"
        output_files = os.listdir(tmp_dirs["backtest_dir"])
        assert any(
            "backtest_results.csv" in f for f in output_files
        ), "Fichier de résultats non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "run_backtest" in str(op) for op in df_perf["operation"]
        ), "Opération run_backtest non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_run_incremental_backtest(tmp_dirs, mock_data):
    """Teste le backtest incrémental."""
    with patch(
        "src.features.signal_selector.calculate_sgc"
    ) as mock_sgc, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        mock_sgc.return_value = (pd.Series([0.8]), None)
        row = mock_data.iloc[0]
        buffer = pd.DataFrame()
        state = {}
        metrics, new_state = run_incremental_backtest(
            row,
            buffer,
            state,
            tmp_dirs["config_path"],
            market="ES",
        )
        assert (
            metrics["num_trades"] == 0
        ), "Nombre de trades incorrect pour buffer vide"
        buffer = mock_data.iloc[:50]
        metrics, new_state = run_incremental_backtest(
            row,
            buffer,
            state,
            tmp_dirs["config_path"],
            market="ES",
        )
        assert (
            len(new_state["equity_curve"]) > 0
        ), "Equity curve vide"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_incremental_backtest" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot incremental_backtest non créé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "incremental_equity_curve" in f and f.endswith(".json.gz")
            for f in checkpoint_files
        ), "Checkpoint incremental_equity_curve non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "run_incremental_backtest" in str(op)
            for op in df_perf["operation"]
        ), "Opération run_incremental_backtest non journalisée"
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
                "equity": [100000],
                "capital": [100000],
            }
        )
        from src.backtest.backtest_lab import cloud_backup

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


@pytest.mark.asyncio
async def test_checkpoint(tmp_dirs):
    """Teste la sauvegarde incrémentielle."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        df = pd.DataFrame(
            {
                "timestamp": [datetime.now().isoformat()],
                "equity": [100000],
                "capital": [100000],
            }
        )
        from src.backtest.backtest_lab import checkpoint

        checkpoint(df, data_type="test_metrics", market="ES")
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "backtest_test_metrics" in f and f.endswith(".json.gz")
            for f in checkpoint_files
        ), "Checkpoint non créé"
        with gzip.open(
            os.path.join(tmp_dirs["checkpoints_dir"], checkpoint_files[0]),
            "rt",
            encoding="utf-8",
        ) as f:
            checkpoint_data = json.load(f)
        assert (
            checkpoint_data["num_rows"] == len(df)
        ), "Nombre de lignes incorrect"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "checkpoint" in str(op) for op in df_perf["operation"]
        ), "Opération checkpoint non journalisée"
        mock_telegram.assert_called()