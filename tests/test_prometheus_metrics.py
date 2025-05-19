# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_prometheus_metrics.py
# Tests unitaires pour src/monitoring/prometheus_metrics.py
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide l’initialisation des métriques Prometheus,
# l’instanciation de PrometheusMetrics, la journalisation des
# performances, et l’exposition des métriques pour le monitoring
# (latence, trades, position sizing, coûts, microstructure, HMM,
# drift, RL, volatilité, ensembles de politiques).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pandas>=2.0.0,<3.0.0
# - src/monitoring/prometheus_metrics.py
# - src/utils/error_tracker.py
# - src/model/utils/alert_manager.py
#
# Inputs :
# - Métriques factices et répertoire temporaire pour les logs
#
# Outputs :
# - Tests unitaires validant les métriques et la journalisation
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Supporte les suggestions 1 (position sizing), 2 (coûts), 3
#   (microstructure), 4 (HMM), 6 (drift), 7 (Safe RL), 8
#   (Distributional RL), 9 (volatilité), 10 (ensembles de
#   politiques).
# - Policies Note: The official directory for routing policies is
#   src/model/router/policies. The src/model/policies directory is a
#   residual and should be verified for removal to avoid import
#   conflicts.

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.monitoring.prometheus_metrics import (
    PrometheusMetrics,
    atr_dynamic,
    bid_ask_imbalance,
    cpu_usage,
    cvar_loss,
    ensemble_weight_ddpg,
    ensemble_weight_ppo,
    ensemble_weight_sac,
    hmm_state_distribution,
    inference_latency,
    init_metrics,
    iv_skew,
    iv_term_structure,
    memory_usage,
    operation_latency,
    orderflow_imbalance,
    position_size_percent,
    position_size_variance,
    profit_factor,
    qr_dqn_quantiles,
    regime_hmm_state,
    regime_transition_rate,
    retrain_runs,
    sharpe_drift,
    slippage_estimate,
    trade_aggressiveness,
    trade_latency,
    trades_processed,
)


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    return {
        "base_dir": base_dir,
        "logs_dir": logs_dir,
        "perf_log_path": logs_dir / "prometheus_metrics_performance.csv",
    }


def test_prometheus_metrics_init(tmp_dirs):
    """Teste l’instanciation de PrometheusMetrics."""
    with patch(
        "src.monitoring.prometheus_metrics.BASE_DIR",
        tmp_dirs["base_dir"],
    ), patch("src.model.utils.alert_manager.AlertManager"):
        metrics = PrometheusMetrics(market="ES")
        assert metrics.market == "ES", "Marché incorrect"
        assert (
            metrics.alert_manager is not None
        ), "AlertManager non initialisé"


def test_init_metrics(tmp_dirs):
    """Teste l’initialisation du serveur Prometheus."""
    with patch(
        "prometheus_client.start_http_server"
    ) as mock_start, patch(
        "src.monitoring.prometheus_metrics.logger.info"
    ) as mock_info, patch(
        "src.monitoring.prometheus_metrics.logger.error"
    ) as mock_error:
        init_metrics(port=8000)
        mock_start.assert_called_with(8000)
        mock_info.assert_called_with(
            "Serveur Prometheus démarré sur le port 8000"
        )
        mock_error.assert_not_called()


def test_metrics_initialization():
    """Teste l’initialisation de toutes les métriques Prometheus."""
    metrics = [
        trades_processed,
        retrain_runs,
        trade_latency,
        inference_latency,
        cpu_usage,
        memory_usage,
        profit_factor,
        position_size_percent,
        position_size_variance,
        regime_hmm_state,
        regime_transition_rate,
        atr_dynamic,
        orderflow_imbalance,
        slippage_estimate,
        bid_ask_imbalance,
        trade_aggressiveness,
        hmm_state_distribution,
        sharpe_drift,
        cvar_loss,
        qr_dqn_quantiles,
        iv_skew,
        iv_term_structure,
        ensemble_weight_sac,
        ensemble_weight_ppo,
        ensemble_weight_ddpg,
        operation_latency,
    ]
    for metric in metrics:
        assert (
            metric is not None
        ), f"Métrique {metric._name} non initialisée"
    assert (
        trades_processed._name == "trades_processed_total"
    ), "Nom de trades_processed incorrect"
    assert (
        retrain_runs._name == "retrain_runs_total"
    ), "Nom de retrain_runs incorrect"
    assert (
        trade_latency._name == "trade_latency"
    ), "Nom de trade_latency incorrect"
    assert (
        hmm_state_distribution._labelnames == ("market", "state")
    ), "Labels de hmm_state_distribution incorrects"


def test_log_performance(tmp_dirs):
    """Teste la journalisation des performances."""
    with patch(
        "psutil.Process.memory_info",
        return_value=MagicMock(rss=1024 * 1024 * 100),
    ), patch(
        "psutil.cpu_percent",
        return_value=50.0,
    ), patch(
        "src.monitoring.prometheus_metrics.cpu_usage.labels"
    ) as mock_cpu, patch(
        "src.monitoring.prometheus_metrics.memory_usage.labels"
    ) as mock_memory, patch(
        "src.monitoring.prometheus_metrics.operation_latency.labels"
    ) as mock_latency:
        metrics = PrometheusMetrics(market="ES")
        with patch(
            "src.monitoring.prometheus_metrics.BASE_DIR",
            tmp_dirs["base_dir"],
        ):
            metrics.log_performance(
                operation="test_op",
                latency=0.1,
                success=True,
                test_key="test_value",
            )
        assert os.path.exists(
            tmp_dirs["perf_log_path"]
        ), "Fichier de logs non créé"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            df["operation"] == "test_op"
        ), "Log test_op manquant"
        assert any(
            df["latency"] == 0.1
        ), "Latence incorrecte"
        assert any(
            df["success"]
        ), "Succès incorrect"
        assert any(
            df["memory_usage_mb"] == 100.0
        ), "Mémoire incorrecte"
        assert any(
            df["cpu_usage_percent"] == 50.0
        ), "CPU incorrect"
        assert any(
            df["test_key"] == "test_value"
        ), "Clé supplémentaire manquante"
        mock_cpu.assert_called_with(market="ES")
        mock_memory.assert_called_with(market="ES")
        mock_latency.assert_called_with(operation="test_op", market="ES")
        mock_cpu.return_value.set.assert_called_with(50.0)
        mock_memory.return_value.set.assert_called_with(100.0)
        mock_latency.return_value.observe.assert_called_with(0.1)