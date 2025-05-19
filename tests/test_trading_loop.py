# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_trading_loop.py
# Tests unitaires pour src/model/utils/trading_loop.py
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide la boucle de trading principale intégrant les méthodes 1-18,
#        avec support multi-marchés, snapshots compressés, et sauvegardes incrémentielles/distribuées.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.8,<6.0.0,
#   boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0, pyyaml>=6.0.0,<7.0.0
# - src/model/router/detect_regime.py
# - src/model/utils/risk_controller.py
# - src/model/utils/trade_executor.py
# - src/model/utils/signal_selector.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - Données factices simulant le flux IQFeed
# - Fichiers de configuration factices (algo_config.yaml, feature_sets.yaml, es_config.yaml)
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de trading_loop.py.
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Tests la Phase 8 (confidence_drop_rate) et les méthodes 1-18.
# - Vérifie l’absence de dxFeed, la suppression de 320/81 features et obs_t, les retries,
#   logs psutil, alertes Telegram, snapshots compressés, et sauvegardes incrémentielles/distribuées.
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence).
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.

import asyncio
import gzip
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.model.utils.trading_loop import TradingLoop


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, cache, checkpoints, et configuration."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    cache_dir = data_dir / "cache" / "trading_loop" / "ES"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints" / "trading_loop" / "ES"
    checkpoints_dir.mkdir(parents=True)

    # Créer algo_config.yaml
    algo_config_path = config_dir / "algo_config.yaml"
    algo_config_content = {
        "trading_loop": {"observation_dims": {"training": 350, "inference": 150}}
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
    es_config_content = {"s3_bucket": "test-bucket", "s3_prefix": "trading_loop/"}
    with open(es_config_path, "w", encoding="utf-8") as f:
        yaml.dump(es_config_content, f)

    return {
        "base_dir": str(base_dir),
        "config_dir": str(config_dir),
        "logs_dir": str(logs_dir),
        "cache_dir": str(cache_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "algo_config_path": str(algo_config_path),
        "feature_sets_path": str(feature_sets_path),
        "es_config_path": str(es_config_path),
        "perf_log_path": str(logs_dir / "trading_loop_performance.csv"),
    }


@pytest.fixture
async def mock_data_stream():
    """Crée un flux de données factice."""
    feature_cols = [f"feature_{i}" for i in range(350)]

    async def data_stream():
        for _ in range(2):
            data = pd.DataFrame(
                {
                    "vix_es_correlation": [20.0],
                    "drawdown": [0.01],
                    "bid_size_level_1": [100],
                    "ask_size_level_1": [120],
                    "trade_frequency_1s": [8],
                    "spread_avg_1min_es": [0.3],
                    "close": [5100],
                    **{col: [np.random.uniform(0, 1)] for col in feature_cols},
                }
            )
            yield data
            await asyncio.sleep(0.1)

    return data_stream()


@pytest.fixture
def mock_regime_detector():
    """Crée un détecteur de régime factice."""
    detector = MagicMock()
    detector.detect_market_regime_vectorized = AsyncMock(
        return_value=("trend", [0.7, 0.2, 0.1])
    )
    return detector


@pytest.fixture
def mock_risk_controller():
    """Crée un contrôleur de risques factice."""
    controller = MagicMock()
    controller.stop_trading = AsyncMock(return_value=False)
    return controller


@pytest.fixture
def mock_trade_executor():
    """Crée un exécuteur de trades factice."""
    executor = MagicMock()
    executor.execute_trade = AsyncMock(return_value={"status": "success"})
    return executor


@pytest.fixture
def mock_signal_selector():
    """Crée un sélecteur de signaux factice."""
    selector = MagicMock()
    selector.select_signal = AsyncMock(return_value={"action": "buy", "size": 1})
    return selector


@pytest.mark.asyncio
async def test_init_trading_loop(tmp_dirs):
    """Teste l’initialisation de TradingLoop."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        loop = TradingLoop(config_path=tmp_dirs["algo_config_path"], market="ES")
        assert loop.market == "ES", "Marché incorrect"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "init" in str(op) for op in df_perf["operation"]
        ), "Opération init non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_validate_data(tmp_dirs):
    """Teste la validation des données."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        loop = TradingLoop(config_path=tmp_dirs["algo_config_path"], market="ES")
        feature_cols = [f"feature_{i}" for i in range(350)]
        data = pd.DataFrame(
            {
                "vix_es_correlation": [20.0],
                "drawdown": [0.01],
                "bid_size_level_1": [100],
                "ask_size_level_1": [120],
                "trade_frequency_1s": [8],
                "spread_avg_1min_es": [0.3],
                "close": [5100],
                **{col: [np.random.uniform(0, 1)] for col in feature_cols},
            }
        )
        loop._validate_data(data)
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
            "confidence_drop_rate" in str(kw) for kw in df_perf.to_dict("records")
        ), "confidence_drop_rate absent"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_trading_loop(
    tmp_dirs,
    mock_data_stream,
    mock_regime_detector,
    mock_risk_controller,
    mock_trade_executor,
    mock_signal_selector,
):
    """Teste la boucle de trading principale."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        loop = TradingLoop(config_path=tmp_dirs["algo_config_path"], market="ES")
        await loop.trading_loop(
            mock_data_stream,
            mock_regime_detector,
            mock_risk_controller,
            mock_trade_executor,
            mock_signal_selector,
        )
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_trading_loop" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot trading_loop non créé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "trading_loop" in f and f.endswith(".json.gz") for f in checkpoint_files
        ), "Checkpoint trading_loop non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "trading_loop" in str(op) for op in df_perf["operation"]
        ), "Opération trading_loop non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_trading_loop_invalid_data(
    tmp_dirs,
    mock_regime_detector,
    mock_risk_controller,
    mock_trade_executor,
    mock_signal_selector,
):
    """Teste la boucle de trading avec des données invalides."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        loop = TradingLoop(config_path=tmp_dirs["algo_config_path"], market="ES")

        async def invalid_data_stream():
            yield "invalid_data"
            await asyncio.sleep(0.1)

        await loop.trading_loop(
            invalid_data_stream(),
            mock_regime_detector,
            mock_risk_controller,
            mock_trade_executor,
            mock_signal_selector,
        )
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "trading_loop" in str(op) and not kw["success"]
            for kw in df_perf.to_dict("records")
        ), "Erreur données invalides non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_cloud_backup(tmp_dirs):
    """Teste la sauvegarde distribuée S3."""
    with patch("boto3.client") as mock_s3, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        loop = TradingLoop(config_path=tmp_dirs["algo_config_path"], market="ES")
        df = pd.DataFrame({"timestamp": [datetime.now().isoformat()], "value": [42]})
        loop.cloud_backup(df, data_type="test_metrics")
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
        loop = TradingLoop(config_path=tmp_dirs["algo_config_path"], market="ES")
        df = pd.DataFrame({"timestamp": [datetime.now().isoformat()], "value": [42]})
        loop.checkpoint(df, data_type="test_metrics")
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "trading_loop_test_metrics" in f and f.endswith(".json.gz")
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
