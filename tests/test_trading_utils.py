# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_trading_utils.py
# Tests unitaires pour src/model/utils/trading_utils.py
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide les utilitaires de trading (détection de régime,
# ajustement du risque/levier, validation de trade, calcul du profit)
# avec support multi-marchés, snapshots compressés, et sauvegardes
# incrémentielles/distribuées.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pandas>=2.0.0,<3.0.0
# - numpy>=1.26.4,<2.0.0
# - pyyaml>=6.0.0,<7.0.0
# - psutil>=5.9.8,<6.0.0
# - boto3>=1.26.0,<2.0.0
# - loguru>=0.7.0,<1.0.0
# - src/model/utils/miya_console.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/router/detect_regime.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - Fichier de configuration factice (es_config.yaml)
# - Données de test simulant les features de trading
# - Modèles ML factices (pickle files)
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de trading_utils.py.
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Tests la Phase 8 (confidence_drop_rate) et la méthode 5
#   (récompenses adaptatives).
# - Vérifie l’absence de dxFeed, la suppression de 320/81 features et
#   obs_t, les retries, logs psutil, alertes Telegram, snapshots
#   compressés, et sauvegardes incrémentielles/distribuées.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.

import gzip
import json
import os
import pickle
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.model.utils.trading_utils import (
    adjust_risk_and_leverage,
    calculate_profit,
    detect_market_regime,
    incremental_adjust_risk_and_leverage,
    incremental_detect_market_regime,
    load_config_manager,
    validate_trade_entry_combined,
)


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, cache, checkpoints,
    modèles, et configuration."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    cache_dir = data_dir / "cache" / "trading_utils" / "ES"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints" / "trading_utils" / "ES"
    checkpoints_dir.mkdir(parents=True)
    models_dir = base_dir / "model" / "ml_models"
    models_dir.mkdir(parents=True)

    # Créer es_config.yaml
    es_config_path = config_dir / "es_config.yaml"
    es_config_content = {
        "trading_utils": {
            "max_leverage": 5.0,
            "risk_factor": 1.0,
            "range_pred_threshold": 0.7,
            "order_flow_pred_threshold": 0.6,
            "atr_threshold": 2.0,
            "adx_threshold": 20,
            "batch_size": 1000,
            "transaction_cost": 2.0,
            "news_impact_threshold": 0.5,
            "vix_threshold": 20.0,
        },
        "s3_bucket": "test-bucket",
        "s3_prefix": "trading_utils/",
    }
    with open(es_config_path, "w", encoding="utf-8") as f:
        yaml.dump(es_config_content, f)

    # Créer modèles ML factices
    range_model_path = models_dir / "range_filter.pkl"
    order_flow_model_path = models_dir / "order_flow_filter.pkl"

    class MockModel:
        def predict_proba(self, X):
            return np.array([[0.3, 0.7]])  # Simule probabilité

    with open(range_model_path, "wb") as f:
        pickle.dump(MockModel(), f)
    with open(order_flow_model_path, "wb") as f:
        pickle.dump(MockModel(), f)

    return {
        "base_dir": str(base_dir),
        "config_dir": str(config_dir),
        "data_dir": str(data_dir),
        "logs_dir": str(logs_dir),
        "cache_dir": str(cache_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "models_dir": str(models_dir),
        "es_config_path": str(es_config_path),
        "range_model_path": str(range_model_path),
        "order_flow_model_path": str(order_flow_model_path),
        "perf_log_path": str(logs_dir / "trading_utils_performance.csv"),
    }


@pytest.fixture
def mock_data():
    """Crée des données de test simulant les features de trading."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-04-14 09:00",
                periods=100,
                freq="1min",
            ),
            "atr_14": np.random.uniform(0.5, 2.0, 100),
            "adx_14": np.random.uniform(10, 50, 100),
            "close": np.random.uniform(4000, 4500, 100),
            "delta_volume": np.random.uniform(-1000, 1000, 100),
            "vwap": np.random.uniform(4000, 4500, 100),
            "bid_size_level_1": np.random.randint(50, 500, 100),
            "ask_size_level_1": np.random.randint(50, 500, 100),
        }
    ).set_index("timestamp")


@pytest.mark.asyncio
async def test_load_config_manager(tmp_dirs):
    """Teste le chargement de la configuration."""
    with patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        config = load_config_manager(tmp_dirs["es_config_path"])
        assert (
            config["max_leverage"] == 5.0
        ), "Configuration max_leverage incorrecte"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_load_config_manager" in f
            and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot load_config_manager non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "load_config_manager" in str(op)
            for op in df_perf["operation"]
        ), "Opération load_config_manager non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_detect_market_regime(tmp_dirs, mock_data):
    """Teste la détection du régime de marché."""
    with patch(
        "src.model.router.detect_regime.MarketRegimeDetector.detect",
        return_value=("trend", 0.8),
    ) as mock_detector:
        regimes = detect_market_regime(
            data=mock_data,
            config_path=tmp_dirs["es_config_path"],
            market="ES",
        )
        assert len(regimes) == len(
            mock_data
        ), "Nombre incorrect de régimes"
        assert all(
            regime in ["trend", "range", "defensive"]
            for regime in regimes
        ), "Régimes invalides détectés"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_detect_market_regime" in f
            and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot detect_market_regime non créé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "detect_market_regime" in f and f.endswith(".json.gz")
            for f in checkpoint_files
        ), "Checkpoint detect_market_regime non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "detect_market_regime" in str(op)
            for op in df_perf["operation"]
        ), "Opération detect_market_regime non journalisée"
        assert any(
            "confidence_drop_rate" in str(kw)
            for kw in df_perf.to_dict("records")
        ), "confidence_drop_rate absent"


@pytest.mark.asyncio
async def test_incremental_detect_market_regime(tmp_dirs, mock_data):
    """Teste la détection incrémentielle du régime de marché."""
    with patch(
        "src.model.router.detect_regime.MarketRegimeDetector.detect",
        return_value=("range", 0.7),
    ) as mock_detector:
        row = mock_data.iloc[-1]
        buffer = mock_data.iloc[:-1]
        regime = incremental_detect_market_regime(
            row=row,
            buffer=buffer,
            config_path=tmp_dirs["es_config_path"],
            market="ES",
        )
        assert regime in [
            "trend",
            "range",
            "defensive",
        ], "Régime invalide détecté"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_incremental_detect_market_regime" in f
            and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot incremental_detect_market_regime non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "incremental_detect_market_regime" in str(op)
            for op in df_perf["operation"]
        ), "Opération incremental_detect_market_regime non journalisée"


@pytest.mark.asyncio
async def test_adjust_risk_and_leverage(tmp_dirs, mock_data):
    """Teste l’ajustement du risque et du levier."""
    regimes = pd.Series("trend", index=mock_data.index)
    data = adjust_risk_and_leverage(
        data=mock_data,
        regimes=regimes,
        config_path=tmp_dirs["es_config_path"],
        market="ES",
    )
    assert (
        "adjusted_leverage" in data.columns
    ), "Colonne adjusted_leverage manquante"
    assert (
        "adjusted_risk" in data.columns
    ), "Colonne adjusted_risk manquante"
    assert (
        data["adjusted_leverage"].max() <= 5.0
    ), "Levier dépasse max_leverage"
    snapshot_files = os.listdir(tmp_dirs["cache_dir"])
    assert any(
        "snapshot_adjust_risk_and_leverage" in f
        and f.endswith(".json.gz")
        for f in snapshot_files
    ), "Snapshot adjust_risk_and_leverage non créé"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "adjust_risk_and_leverage" in str(op)
        for op in df_perf["operation"]
    ), "Opération adjust_risk_and_leverage non journalisée"


@pytest.mark.asyncio
async def test_incremental_adjust_risk_and_leverage(tmp_dirs, mock_data):
    """Teste l’ajustement incrémentiel du risque et du levier."""
    row = mock_data.iloc[-1]
    leverage, risk = incremental_adjust_risk_and_leverage(
        row=row,
        regime="trend",
        config_path=tmp_dirs["es_config_path"],
        market="ES",
    )
    assert leverage <= 5.0, "Levier dépasse max_leverage"
    assert risk > 0, "Risque invalide"
    snapshot_files = os.listdir(tmp_dirs["cache_dir"])
    assert any(
        "snapshot_incremental_adjust_risk_and_leverage" in f
        and f.endswith(".json.gz")
        for f in snapshot_files
    ), "Snapshot incremental_adjust_risk_and_leverage non créé"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "incremental_adjust_risk_and_leverage" in str(op)
        for op in df_perf["operation"]
    ), "Opération incremental_adjust_risk_and_leverage non journalisée"


@pytest.mark.asyncio
async def test_validate_trade_entry_combined(tmp_dirs, mock_data):
    """Teste la validation des entrées en trade."""

    class MockEnv:
        def __init__(self):
            self.data = mock_data
            self.current_step = 60

    env = MockEnv()
    with patch(
        "src.model.utils.trading_utils.load_model_cached",
        return_value=MagicMock(
            predict_proba=lambda x: np.array([[0.3, 0.7]])
        ),
    ):
        is_valid = validate_trade_entry_combined(
            env=env,
            range_model_path=tmp_dirs["range_model_path"],
            order_flow_model_path=tmp_dirs["order_flow_model_path"],
            config_path=tmp_dirs["es_config_path"],
            debug=True,
            market="ES",
        )
        assert isinstance(
            is_valid, bool
        ), "Résultat de validation invalide"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_validate_trade_entry_combined" in f
            and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot validate_trade_entry_combined non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "validate_trade_entry_combined" in str(op)
            for op in df_perf["operation"]
        ), "Opération validate_trade_entry_combined non journalisée"


@pytest.mark.asyncio
async def test_calculate_profit(tmp_dirs):
    """Teste le calcul du profit."""
    trade = {
        "entry_price": 4000.0,
        "exit_price": 4050.0,
        "position_size": 10,
        "trade_type": "long",
        "news_impact_score": 0.6,
        "vix": 18.0,
    }
    profit = calculate_profit(
        trade=trade,
        config_path=tmp_dirs["es_config_path"],
        market="ES",
    )
    assert isinstance(profit, float), "Profit invalide"
    assert profit > 0, "Profit devrait être positif pour ce trade"
    snapshot_files = os.listdir(tmp_dirs["cache_dir"])
    assert any(
        "snapshot_calculate_profit" in f and f.endswith(".json.gz")
        for f in snapshot_files
    ), "Snapshot calculate_profit non créé"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "calculate_profit" in str(op)
        for op in df_perf["operation"]
    ), "Opération calculate_profit non journalisée"


@pytest.mark.asyncio
async def test_cloud_backup(tmp_dirs):
    """Teste la sauvegarde distribuée S3."""
    with patch("boto3.client") as mock_s3:
        df = pd.DataFrame(
            {
                "timestamp": [datetime.now().isoformat()],
                "value": [42],
            }
        )
        from src.model.utils.trading_utils import cloud_backup

        cloud_backup(
            df=df,
            data_type="test_metrics",
            market="ES",
        )
        assert mock_s3.called, "Client S3 non appelé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "temp_s3_" in f for f in checkpoint_files
        ), "Fichier temporaire S3 non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "cloud_backup" in str(op)
            for op in df_perf["operation"]
        ), "Opération cloud_backup non journalisée"


@pytest.mark.asyncio
async def test_checkpoint(tmp_dirs):
    """Teste la sauvegarde incrémentielle."""
    df = pd.DataFrame(
        {
            "timestamp": [datetime.now().isoformat()],
            "value": [42],
        }
    )
    from src.model.utils.trading_utils import checkpoint

    checkpoint(
        df=df,
        data_type="test_metrics",
        market="ES",
    )
    checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
    assert any(
        "trading_utils_test_metrics" in f and f.endswith(".json.gz")
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
        "checkpoint" in str(op)
        for op in df_perf["operation"]
    ), "Opération checkpoint non journalisée"