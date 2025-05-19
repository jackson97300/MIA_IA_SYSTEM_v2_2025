# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_live_trading.py
# Tests unitaires pour src/trading/live_trading.py
# Version : 2.1.3
# Date : 2025-05-09
# Rôle : Valide l’exécution des trades en live, la gestion des données, et
# les snapshots JSON.

import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.trading.live_trading import LiveTrader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, figures, trades, snapshots, et configurations."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    trading_logs_dir = logs_dir / "trading"
    trading_logs_dir.mkdir()
    figures_dir = data_dir / "figures"
    figures_dir.mkdir()
    trades_dir = data_dir / "trades"
    trades_dir.mkdir()
    snapshot_dir = data_dir / "trade_snapshots"
    snapshot_dir.mkdir()
    features_dir = data_dir / "features"
    features_dir.mkdir()

    # Créer market_config.yaml
    config_path = config_dir / "market_config.yaml"
    config_content = {
        "thresholds": {
            "max_position_size": 5,
            "min_confidence": 0.7,
            "max_drawdown": -0.1,
            "vix_threshold": 30,
            "spread_threshold": 0.05,
            "market_liquidity_crash_risk": 0.8,
            "event_impact_threshold": 0.5,
            "min_sharpe": 0.5,
            "min_profit_factor": 1.2,
            "min_balance": -10000,
        },
        "cache": {"max_prediction_cache_size": 1000},
        "data": {"cache_duration": 24},
        "logging": {"buffer_size": 200},
        "plotting": {
            "figsize": [12, 5],
            "colors": {"equity": "blue", "profit": "orange", "rewards": "green"},
        },
    }
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)

    # Créer credentials.yaml
    credentials_path = config_dir / "credentials.yaml"
    credentials_content = {"amp_futures": {"api_key": "to_be_defined"}}
    with open(credentials_path, "w") as f:
        yaml.dump(credentials_content, f)

    # Créer trade_probability_config.yaml
    trade_prob_config_path = config_dir / "trade_probability_config.yaml"
    trade_prob_content = {"trade_probability": {"min_trade_success_prob": 0.7}}
    with open(trade_prob_config_path, "w") as f:
        yaml.dump(trade_prob_content, f)

    # Créer feature_importance.csv
    feature_importance_path = features_dir / "feature_importance.csv"
    feature_importance_data = pd.DataFrame(
        {"feature": [f"feat_{i}" for i in range(200)], "importance": [0.1] * 200}
    )
    feature_importance_data.to_csv(
        feature_importance_path, index=False, encoding="utf-8"
    )

    # Créer features_latest_filtered.csv
    features_path = features_dir / "features_latest_filtered.csv"
    features_data = pd.DataFrame(
        {
            "timestamp": [datetime.now() + timedelta(minutes=i) for i in range(10)],
            "close": [5100.0] * 10,
            "bid_price_level_1": [5098.0] * 10,
            "ask_price_level_1": [5102.0] * 10,
            "vix": [20.0] * 10,
            "rsi_14": [50.0] * 10,
            "gex": [100.0] * 10,
            "volume": [1000.0] * 10,
            "atr_14": [1.5] * 10,
            "adx_14": [20.0] * 10,
            **{
                f"feat_{i}": np.random.uniform(0, 1, 10) for i in range(140)
            },  # 150 features
        }
    )
    features_data.to_csv(features_path, index=False, encoding="utf-8")

    return {
        "config_path": str(config_path),
        "credentials_path": str(credentials_path),
        "trade_prob_config_path": str(trade_prob_config_path),
        "logs_dir": str(logs_dir),
        "figures_dir": str(figures_dir),
        "trades_dir": str(trades_dir),
        "snapshot_dir": str(snapshot_dir),
        "features_path": str(features_path),
        "feature_importance_path": str(feature_importance_path),
        "dashboard_path": str(data_dir / "live_trading_dashboard.json"),
        "decision_log_path": str(trading_logs_dir / "decision_log.csv"),
    }


def test_live_trader_init(tmp_dirs):
    """Teste l’initialisation de LiveTrader."""
    global PERF_LOG_PATH, SNAPSHOT_DIR
    PERF_LOG_PATH = tmp_dirs["logs_dir"] / "live_trading.csv"
    SNAPSHOT_DIR = tmp_dirs["snapshot_dir"]

    trader = LiveTrader(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    assert os.path.exists(PERF_LOG_PATH), "Fichier de logs de performance non créé"
    assert os.path.exists(SNAPSHOT_DIR), "Dossier de snapshots non créé"
    df = pd.read_csv(PERF_LOG_PATH)
    assert "cpu_percent" in df.columns, "Colonne cpu_percent manquante"


def test_validate_data_valid(tmp_dirs):
    """Teste la validation de données valides."""
    global PERF_LOG_PATH, SNAPSHOT_DIR
    PERF_LOG_PATH = tmp_dirs["logs_dir"] / "live_trading.csv"
    SNAPSHOT_DIR = tmp_dirs["snapshot_dir"]

    trader = LiveTrader(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    data = pd.DataFrame(
        {
            "timestamp": [datetime.now()],
            "close": [5100.0],
            "bid_price_level_1": [5098.0],
            "ask_price_level_1": [5102.0],
            "vix": [20.0],
            "rsi_14": [50.0],
            "gex": [100.0],
            "volume": [1000.0],
            "atr_14": [1.5],
            "adx_14": [20.0],
            **{
                f"feat_{i}": [np.random.uniform(0, 1)] for i in range(140)
            },  # 150 features
        }
    )
    trader.validate_data(data)
    assert os.path.exists(PERF_LOG_PATH), "Fichier de logs de performance non créé"


def test_validate_data_invalid(tmp_dirs):
    """Teste la validation de données invalides."""
    global PERF_LOG_PATH, SNAPSHOT_DIR
    PERF_LOG_PATH = tmp_dirs["logs_dir"] / "live_trading.csv"
    SNAPSHOT_DIR = tmp_dirs["snapshot_dir"]

    trader = LiveTrader(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    data = pd.DataFrame(
        {"timestamp": [datetime.now()], "close": [5100.0]}
    )  # Dimension incorrecte
    with pytest.raises(ValueError, match="Nombre de features insuffisant"):
        trader.validate_data(data)
    assert os.path.exists(PERF_LOG_PATH), "Fichier de logs de performance non créé"


@pytest.mark.asyncio
async def test_load_data(tmp_dirs):
    """Teste le chargement des données via IQFeed."""
    global PERF_LOG_PATH, SNAPSHOT_DIR
    PERF_LOG_PATH = tmp_dirs["logs_dir"] / "live_trading.csv"
    SNAPSHOT_DIR = tmp_dirs["snapshot_dir"]

    trader = LiveTrader(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    with patch(
        "src.data.data_provider.get_data_provider",
        return_value=AsyncMock(fetch_features=lambda x: pd.read_csv(x)),
    ):
        data = await trader.load_data(tmp_dirs["features_path"])
    assert not data.empty, "Données non chargées"
    assert len(data.columns) >= 150, "Nombre de features incorrect"
    assert os.path.exists(PERF_LOG_PATH), "Fichier de logs de performance non créé"


def test_save_trade_snapshot(tmp_dirs):
    """Teste la sauvegarde des instantanés JSON."""
    global PERF_LOG_PATH, SNAPSHOT_DIR
    PERF_LOG_PATH = tmp_dirs["logs_dir"] / "live_trading.csv"
    SNAPSHOT_DIR = tmp_dirs["snapshot_dir"]

    trader = LiveTrader(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    df_row = pd.Series(
        {
            "timestamp": datetime.now(),
            "close": 5100.0,
            **{
                f"feat_{i}": np.random.uniform(0, 1) for i in range(149)
            },  # 150 features
        }
    )
    trader.save_trade_snapshot(
        step=1,
        timestamp=pd.Timestamp.now(),
        action=1.0,
        regime="trend",
        reward=100.0,
        confidence=0.8,
        trade_success_prob=0.75,
        df_row=df_row,
        risk_metrics={"market_liquidity_crash_risk": 0.2},
        decision={"decision": "execute", "score": 0.8},
    )
    assert any(SNAPSHOT_DIR.glob("trade_step_0001.json")), "Snapshot JSON non créé"
    assert os.path.exists(PERF_LOG_PATH), "Fichier de logs de performance non créé"


@pytest.mark.asyncio
async def test_handle_vocal_commands(tmp_dirs):
    """Teste la gestion des commandes vocales."""
    global PERF_LOG_PATH, SNAPSHOT_DIR
    PERF_LOG_PATH = tmp_dirs["logs_dir"] / "live_trading.csv"
    SNAPSHOT_DIR = tmp_dirs["snapshot_dir"]

    trader = LiveTrader(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    data = pd.DataFrame(
        {
            "timestamp": [datetime.now()],
            "close": [5100.0],
            "bid_price_level_1": [5098.0],
            "ask_price_level_1": [5102.0],
            "vix": [20.0],
            "rsi_14": [50.0],
            "gex": [100.0],
            "volume": [1000.0],
            "atr_14": [1.5],
            "adx_14": [20.0],
            **{
                f"feat_{i}": [np.random.uniform(0, 1)] for i in range(140)
            },  # 150 features
        }
    )
    with patch(
        "src.mind.mind_dialogue.DialogueManager.listen_command",
        return_value="stop trading",
    ):
        with patch("src.mind.mind_dialogue.DialogueManager.respond", return_value=None):
            result = await trader.handle_vocal_commands(
                current_data=data,
                current_step=1,
                balance=10000.0,
                positions=[],
                risk_controller=RiskController(),
                trade_window_filter=TradeWindowFilter(),
                decision_log=DecisionLog(),
                live_mode=False,
            )
    assert result is True, "Commande vocale 'stop trading' non gérée correctement"
    assert os.path.exists(PERF_LOG_PATH), "Fichier de logs de performance non créé"
