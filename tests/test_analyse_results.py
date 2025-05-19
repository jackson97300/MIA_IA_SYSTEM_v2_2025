# Teste analyse_results.py.
# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_analyse_results.py
# Tests unitaires pour src/trading/analyse_results.py
# Version : 2.1.3
# Date : 2025-05-09
# Rôle : Valide l’analyse des résultats de trading, l’intégration SHAP, et
# les visualisations.

import os
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.trading.analyse_results import TradeAnalyzer

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
    figures_dir = data_dir / "figures/trading"
    figures_dir.mkdir(parents=True)
    trades_dir = data_dir / "trades"
    trades_dir.mkdir()
    snapshot_dir = data_dir / "trading/analyse_snapshots"
    snapshot_dir.mkdir()

    # Créer market_config.yaml
    config_path = config_dir / "market_config.yaml"
    config_content = {
        "thresholds": {
            "min_sharpe": 0.5,
            "max_drawdown": -1000.0,
            "min_profit_factor": 1.2,
        },
        "analysis": {"use_neural_pipeline": False},
        "logging": {"buffer_size": 100},
    }
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)

    # Créer trades_simulated.csv
    trades_path = trades_dir / "trades_simulated.csv"
    trades_data = pd.DataFrame(
        {
            "timestamp": [datetime.now() + timedelta(minutes=i) for i in range(10)],
            "reward": [100, -50, 200, -30, 150, -20, 300, -10, 50, -40],
            "regime": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            "rsi_14": [50, 55, 60, 45, 70, 65, 40, 50, 55, 60],
            "vix": [20.0] * 10,
            "neural_regime": [0] * 10,
            "predicted_volatility": [1.5] * 10,
            "trade_frequency_1s": [8.0] * 10,
            "close": [5100.0] * 10,
            "gex": [100.0] * 10,
            "step": list(range(10)),
            **{
                f"feat_{i}": np.random.uniform(0, 1, 10) for i in range(340)
            },  # 350 features
        }
    )
    trades_data.to_csv(trades_path, index=False, encoding="utf-8")

    return {
        "config_path": str(config_path),
        "logs_dir": str(logs_dir),
        "figures_dir": str(figures_dir),
        "trades_dir": str(trades_dir),
        "trades_path": str(trades_path),
        "snapshot_dir": str(snapshot_dir),
        "dashboard_path": str(data_dir / "trading/analyse_results_dashboard.json"),
    }


def test_trade_analyzer_init(tmp_dirs):
    """Teste l’initialisation de TradeAnalyzer."""
    global PERF_LOG_PATH, SNAPSHOT_DIR
    PERF_LOG_PATH = tmp_dirs["logs_dir"] / "analyse_results.csv"
    SNAPSHOT_DIR = tmp_dirs["snapshot_dir"]

    analyzer = TradeAnalyzer(config_path=tmp_dirs["config_path"])
    assert os.path.exists(PERF_LOG_PATH), "Fichier de logs de performance non créé"
    assert os.path.exists(SNAPSHOT_DIR), "Dossier de snapshots non créé"
    df = pd.read_csv(PERF_LOG_PATH)
    assert "cpu_percent" in df.columns, "Colonne cpu_percent manquante"


def test_validate_data_valid(tmp_dirs):
    """Teste la validation de données valides."""
    global PERF_LOG_PATH, SNAPSHOT_DIR
    PERF_LOG_PATH = tmp_dirs["logs_dir"] / "analyse_results.csv"
    SNAPSHOT_DIR = tmp_dirs["snapshot_dir"]

    analyzer = TradeAnalyzer(config_path=tmp_dirs["config_path"])
    data = pd.DataFrame(
        {
            "timestamp": [datetime.now()],
            "reward": [100],
            "regime": [0],
            "rsi_14": [50],
            "vix": [20.0],
            "neural_regime": [0],
            "predicted_volatility": [1.5],
            "trade_frequency_1s": [8.0],
            "close": [5100.0],
            "gex": [100.0],
            "step": [0],
            **{
                f"feat_{i}": [np.random.uniform(0, 1)] for i in range(339)
            },  # 350 features
        }
    )
    analyzer.validate_data(data)
    assert os.path.exists(PERF_LOG_PATH), "Fichier de logs de performance non créé"


def test_validate_data_invalid(tmp_dirs):
    """Teste la validation de données invalides."""
    global PERF_LOG_PATH, SNAPSHOT_DIR
    PERF_LOG_PATH = tmp_dirs["logs_dir"] / "analyse_results.csv"
    SNAPSHOT_DIR = tmp_dirs["snapshot_dir"]

    analyzer = TradeAnalyzer(config_path=tmp_dirs["config_path"])
    data = pd.DataFrame(
        {"timestamp": [datetime.now()], "reward": [100]}
    )  # Dimension incorrecte
    with pytest.raises(ValueError, match="Nombre de features insuffisant"):
        analyzer.validate_data(data)
    assert os.path.exists(PERF_LOG_PATH), "Fichier de logs de performance non créé"


def test_load_trades(tmp_dirs):
    """Teste le chargement des trades."""
    global PERF_LOG_PATH, SNAPSHOT_DIR
    PERF_LOG_PATH = tmp_dirs["logs_dir"] / "analyse_results.csv"
    SNAPSHOT_DIR = tmp_dirs["snapshot_dir"]

    analyzer = TradeAnalyzer(config_path=tmp_dirs["config_path"])
    df = analyzer.load_trades(tmp_dirs["trades_path"])
    assert df is not None, "Échec chargement trades"
    assert len(df) == 10, "Nombre de trades incorrect"
    assert os.path.exists(PERF_LOG_PATH), "Fichier de logs de performance non créé"


def test_analyse_results(tmp_dirs):
    """Teste l’analyse des résultats avec SHAP et visualisations."""
    global PERF_LOG_PATH, SNAPSHOT_DIR
    PERF_LOG_PATH = tmp_dirs["logs_dir"] / "analyse_results.csv"
    SNAPSHOT_DIR = tmp_dirs["snapshot_dir"]

    analyzer = TradeAnalyzer(config_path=tmp_dirs["config_path"])
    trades = pd.read_csv(tmp_dirs["trades_path"])

    with patch(
        "src.trading.shap_weighting.calculate_shap",
        return_value=pd.DataFrame({"rsi_14": [0.1] * 10, "vix": [0.05] * 10}),
    ):
        result = analyzer.analyse_results(trades, symbol="ES")

    assert "profit" in result, "Clé 'profit' manquante"
    assert "shap_metrics" in result, "Clé 'shap_metrics' manquante"
    assert os.path.exists(
        os.path.join(
            tmp_dirs["figures_dir"],
            "ES_reward_vs_rsi_14_" + datetime.now().strftime("%Y%m%d") + "*.png",
        )
    ), "Graphique SHAP non créé"
    assert os.path.exists(PERF_LOG_PATH), "Fichier de logs de performance non créé"
    assert os.path.exists(tmp_dirs["dashboard_path"]), "Fichier dashboard non créé"


def test_compute_metrics(tmp_dirs):
    """Teste le calcul des métriques."""
    global PERF_LOG_PATH, SNAPSHOT_DIR
    PERF_LOG_PATH = tmp_dirs["logs_dir"] / "analyse_results.csv"
    SNAPSHOT_DIR = tmp_dirs["snapshot_dir"]

    analyzer = TradeAnalyzer(config_path=tmp_dirs["config_path"])
    trades = pd.read_csv(tmp_dirs["trades_path"])
    stats = analyzer.compute_metrics(trades)
    assert "total_trades" in stats, "Métrique total_trades manquante"
    assert "sharpe_ratio" in stats, "Métrique sharpe_ratio manquante"
    assert os.path.exists(PERF_LOG_PATH), "Fichier de logs de performance non créé"
