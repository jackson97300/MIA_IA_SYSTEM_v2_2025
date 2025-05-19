# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/monitoring/prometheus_metrics.py
# Exposition des métriques Prometheus pour MIA_IA_SYSTEM_v2_2025
#
# Version: 2.1.4
# Date: 2025-05-13
#
# Rôle: Fournit des compteurs et jauges pour surveiller en temps réel la latence, les trades traités, l’utilisation CPU/mémoire,
#       le facteur de profit, le position sizing, les coûts de transaction, la microstructure, les états HMM, le drift,
#       les performances RL, la surface de volatilité, et les ensembles de politiques.
# Utilisé par: run_system.py, performance_logger.py, switch_logger.py, mia_switcher.py, train_pipeline.py.
#
# Notes:
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Supporte les suggestions 1 (position sizing), 2 (coûts de transaction), 3 (microstructure),
#   4 (HMM/changepoint), 5 (monitoring du profit factor), 6 (drift detection), 7 (Safe RL),
#   8 (Distributional RL), 9 (surface de volatilité), 10 (ensembles de politiques).
# - Pas de références à dxFeed, obs_t, 320 features, ou 81 features.
# - Intégré avec prometheus.yml pour le scraping et grafana.ini pour les dashboards.
# - Journalise les performances dans data/logs/prometheus_metrics_performance.csv.
# - Policies Note: The official directory for routing policies is src/model/router/policies.
# The src/model/policies directory is a residual and should be verified
# for removal to avoid import conflicts.

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import psutil
from loguru import logger
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from src.model.utils.alert_manager import AlertManager
from src.utils.error_tracker import capture_error

# Configuration du logging
logger.remove()
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOG_DIR = BASE_DIR / "data" / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "prometheus_metrics.log", rotation="10 MB", level="INFO", encoding="utf-8"
)


def init_metrics(port: int = 8000) -> None:
    """Démarre le serveur HTTP pour exposer les métriques Prometheus.

    Args:
        port: Port pour l’endpoint HTTP (défaut: 8000).

    Raises:
        ValueError: Si le port est invalide ou déjà utilisé.
    """
    try:
        start_http_server(port)
        logger.info(f"Serveur Prometheus démarré sur le port {port}")
    except ValueError as e:
        logger.error(f"Erreur lors du démarrage du serveur Prometheus: {e}")
        raise ValueError(f"Port {port} invalide ou déjà utilisé: {e}")
    except OSError as e:
        logger.error(f"Erreur réseau lors du démarrage du serveur: {e}")
        raise ValueError(f"Échec du démarrage du serveur: {e}")


# Compteur pour les trades traités
trades_processed = Counter(
    "trades_processed_total", "Nombre total de trades traités", ["market"]
)

# Compteur pour les réentraînements
retrain_runs = Counter(
    "retrain_runs_total", "Nombre total de réentraînements exécutés", ["market"]
)

# Jauge pour la latence des trades
trade_latency = Gauge("trade_latency", "Latence des trades", ["market"])

# Jauge pour la latence d’inférence
inference_latency = Gauge(
    "inference_latency_seconds", "Latence d’inférence en secondes", ["market"]
)

# Jauge pour l’utilisation CPU
cpu_usage = Gauge("cpu_usage_percent", "Utilisation CPU en pourcentage", ["market"])

# Jauge pour l’utilisation mémoire
memory_usage = Gauge("memory_usage_mb", "Utilisation mémoire en mégaoctets", ["market"])

# Jauge pour le facteur de profit
profit_factor = Gauge("profit_factor", "Facteur de profit calculé", ["market"])

# Jauge pour la taille de position
position_size_percent = Gauge(
    "position_size_percent", "Taille de position en pourcentage", ["market"]
)

# Jauge pour la variance des tailles de position
position_size_variance = Gauge(
    "position_size_variance", "Variance des tailles de position", ["market"]
)

# Jauge pour l’état HMM
regime_hmm_state = Gauge(
    "regime_hmm_state", "État HMM (0: bull, 1: bear, 2: range)", ["market"]
)

# Jauge pour le taux de transition HMM
regime_transition_rate = Gauge(
    "regime_transition_rate", "Taux de transition par heure", ["market"]
)

# Jauge pour l’ATR dynamique
atr_dynamic = Gauge("atr_dynamic", "ATR dynamique sur 1-5 min", ["market"])

# Jauge pour l’imbalance du carnet d’ordres
orderflow_imbalance = Gauge(
    "orderflow_imbalance", "Imbalance du carnet d’ordres", ["market"]
)

# Jauge pour l’estimation du slippage
slippage_estimate = Gauge("slippage_estimate", "Estimation du slippage", ["market"])

# Jauge pour l’imbalance bid-ask
bid_ask_imbalance = Gauge("bid_ask_imbalance", "Imbalance bid-ask", ["market"])

# Jauge pour l’agressivité des trades
trade_aggressiveness = Gauge(
    "trade_aggressiveness", "Agressivité des trades", ["market"]
)

# Jauge pour la répartition des états HMM
hmm_state_distribution = Gauge(
    "hmm_state_distribution", "Répartition des états HMM", ["market", "state"]
)

# Jauge pour le drift du ratio de Sharpe
sharpe_drift = Gauge("sharpe_drift", "Drift du ratio de Sharpe", ["market"])

# Jauge pour la perte CVaR (Safe RL)
cvar_loss = Gauge("cvar_loss", "Perte CVaR pour PPO-Lagrangian", ["market"])

# Jauge pour les quantiles QR-DQN
qr_dqn_quantiles = Gauge("qr_dqn_quantiles", "Quantiles QR-DQN", ["market"])

# Jauge pour le skew de volatilité implicite
iv_skew = Gauge("iv_skew", "Skew de volatilité implicite", ["market"])

# Jauge pour la structure de terme IV
iv_term_structure = Gauge(
    "iv_term_structure", "Structure de terme de volatilité implicite", ["market"]
)

# Jauges pour les poids des ensembles de politiques
ensemble_weight_sac = Gauge(
    "ensemble_weight_sac", "Poids SAC dans le vote bayésien", ["market"]
)
ensemble_weight_ppo = Gauge(
    "ensemble_weight_ppo", "Poids PPO dans le vote bayésien", ["market"]
)
ensemble_weight_ddpg = Gauge(
    "ensemble_weight_ddpg", "Poids DDPG dans le vote bayésien", ["market"]
)

# Histogramme pour la latence des opérations
operation_latency = Histogram(
    "operation_latency_seconds",
    "Latence des opérations critiques (ex. : feature calculation, model switch)",
    ["operation", "market"],
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, float("inf")),
)


class PrometheusMetrics:
    def __init__(self, market: str = "ES"):
        self.market = market
        self.alert_manager = AlertManager()
        LOG_DIR.mkdir(exist_ok=True)

    def log_performance(self, operation: str, latency: float, success: bool, **kwargs):
        """Journalise les performances des métriques."""
        try:
            mem_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "latency": latency,
                "success": success,
                "memory_usage_mb": mem_usage,
                "cpu_usage_percent": cpu_percent,
                **kwargs,
            }
            log_df = pd.DataFrame([log_entry])
            log_path = LOG_DIR / "prometheus_metrics_performance.csv"
            log_df.to_csv(
                log_path,
                mode="a",
                header=not log_path.exists(),
                index=False,
                encoding="utf-8",
            )
            logger.info(
                f"Performance journalisée pour {operation}. CPU: {cpu_percent}%"
            )
            cpu_usage.labels(market=self.market).set(cpu_percent)
            memory_usage.labels(market=self.market).set(mem_usage)
            operation_latency.labels(operation=operation, market=self.market).observe(
                latency
            )
        except Exception as e:
            error_msg = f"Erreur journalisation performance: {str(e)}"
            logger.error(error_msg)
            capture_error(
                e,
                context={"market": self.market},
                market=self.market,
                operation="log_performance",
            )
            self.alert_manager.send_alert(error_msg, priority=3)


"""
# Bloc de débogage commenté pour éviter une exécution en production
if __name__ == "__main__":
    try:
        init_metrics(port=8000)
        # Simuler des métriques
        trades_processed.labels(market="ES").inc()
        retrain_runs.labels(market="ES").inc()
        trade_latency.labels(market="ES").set(0.05)
        inference_latency.labels(market="ES").set(0.1)
        cpu_usage.labels(market="ES").set(75.0)
        memory_usage.labels(market="ES").set(512.0)
        profit_factor.labels(market="ES").set(2.5)
        position_size_percent.labels(market="ES").set(1.5)
        position_size_variance.labels(market="ES").set(0.2)
        regime_hmm_state.labels(market="ES").set(1)
        regime_transition_rate.labels(market="ES").set(0.05)
        atr_dynamic.labels(market="ES").set(10.0)
        orderflow_imbalance.labels(market="ES").set(0.3)
        slippage_estimate.labels(market="ES").set(0.02)
        bid_ask_imbalance.labels(market="ES").set(-0.1)
        trade_aggressiveness.labels(market="ES").set(0.4)
        hmm_state_distribution.labels(market="ES", state="bull").set(0.6)
        sharpe_drift.labels(market="ES").set(0.01)
        cvar_loss.labels(market="ES").set(0.05)
        qr_dqn_quantiles.labels(market="ES").set(51)
        iv_skew.labels(market="ES").set(0.01)
        iv_term_structure.labels(market="ES").set(0.02)
        ensemble_weight_sac.labels(market="ES").set(0.33)
        ensemble_weight_ppo.labels(market="ES").set(0.33)
        ensemble_weight_ddpg.labels(market="ES").set(0.34)
        operation_latency.labels(operation="feature_calculation", market="ES").observe(0.15)
        logger.info("Métriques initialisées. Vérifiez http://localhost:8000")
    except ValueError as e:
        logger.error(f"Erreur: {e}")
"""
