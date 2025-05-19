# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/trading/metrics_computer.py
# Rôle : Calcule les métriques de performance des trades globales, par régime, et par session.
#
# Version : 2.1.3
# Date : 2025-05-15
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0
# - numpy>=1.26.4,<2.0.0
#
# Inputs :
# - Données de trading (pd.DataFrame avec reward, regime, session, trade_duration, iv_skew, timestamp)
# - Configuration via config/market_config.yaml
#
# Outputs :
# - Dictionnaire de métriques (win_rate, profit_factor, sharpe_ratio, iv_skew, entry_freq, period, etc.)
#
# Notes :
# - Calcule les métriques globales, par régime, et par session, incluant consecutive_losses et avg_duration.
# - Ajoute iv_skew, entry_freq, et period pour la génération de rapports narratifs (Proposition 1).
# - Supporte les fenêtres glissantes (rolling metrics) pour les rapports périodiques.
# - Utilise l’encodage UTF-8 pour les logs.
# - Intègre les alertes via AlertDispatcher pour les seuils non atteints.

import itertools
import logging
import os
import time
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any

# Chemins relatifs
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Configurer logging
logging.basicConfig(
    filename=BASE_DIR / "data" / "logs" / "metrics_computer.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

class MetricsComputer:
    def __init__(self, config: Dict, alert_dispatcher: Optional[Any] = None):
        self.config = config
        self.alert_dispatcher = alert_dispatcher
        self.performance_thresholds = {
            "min_sharpe": self.config.get("thresholds", {}).get("min_sharpe", 0.5),
            "max_drawdown": self.config.get("thresholds", {}).get("max_drawdown", -1000.0),
            "min_profit_factor": self.config.get("thresholds", {}).get("min_profit_factor", 1.2),
        }

    def compute_metrics(self, df: pd.DataFrame, rolling: List[int] = None) -> Dict[str, Any]:
        """Calcule les métriques de performance des trades."""
        try:
            start_time = time.time()
            rewards = df["reward"].fillna(0).values
            total_trades = len(rewards)
            win_rate = (rewards > 0).mean() * 100
            profit_factor = rewards[rewards > 0].sum() / abs(rewards[rewards < 0].sum()) if (rewards < 0).sum() != 0 else np.inf
            sharpe = (rewards.mean() / rewards.std()) * np.sqrt(252) if rewards.std() > 0 else 0
            equity = np.cumsum(rewards)
            max_dd = np.min(equity - np.maximum.accumulate(equity)) if len(equity) > 0 else 0
            avg_win = rewards[rewards > 0].mean() if (rewards > 0).sum() > 0 else 0
            avg_loss = rewards[rewards < 0].mean() if (rewards < 0).sum() != 0 else 0
            risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
            consecutive_losses = max(len(list(g)) for k, g in itertools.groupby(rewards, lambda x: x < 0) if k) if len(rewards) > 0 else 0
            
            # Vérifications des colonnes
            if "trade_duration" not in df:
                logger.warning("Colonne 'trade_duration' manquante, utilisation de 0")
                avg_duration = 0
            else:
                avg_duration = df["trade_duration"].mean()
            
            if "iv_skew" not in df:
                logger.warning("Colonne 'iv_skew' manquante, utilisation de 0")
                iv_skew = 0.0
            else:
                iv_skew = df["iv_skew"].mean()
            
            if "timestamp" not in df:
                logger.warning("Colonne 'timestamp' manquante, période et fréquence non calculées")
                entry_freq = 0.0
                period = "N/A"
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                period_days = (df["timestamp"].max() - df["timestamp"].min()).days
                entry_freq = len(df) / period_days if period_days > 0 else 0.0
                period = f"{df['timestamp'].min().strftime('%Y-%m-%d')} → {df['timestamp'].max().strftime('%Y-%m-%d')}"

            stats = {
                "total_trades": total_trades,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_dd,
                "total_return": equity[-1] if len(equity) > 0 else 0,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "risk_reward_ratio": risk_reward,
                "consecutive_losses": consecutive_losses,
                "avg_duration": avg_duration,
                "iv_skew": iv_skew,
                "entry_freq": entry_freq,
                "period": period,
            }

            if rolling:
                for window in rolling:
                    rolling_rewards = df["reward"].rolling(window).sum().fillna(0)
                    stats[f"rolling_return_{window}d"] = rolling_rewards.mean()
                    stats[f"rolling_sharpe_{window}d"] = (rolling_rewards.mean() / rolling_rewards.std()) * np.sqrt(252) if rolling_rewards.std() > 0 else 0

            for group in ["regime", "session"]:
                if group in df.columns:
                    stats[f"by_{group}"] = {}
                    for value in df[group].unique():
                        group_rewards = df[df[group] == value]["reward"].fillna(0).values
                        stats[f"by_{group}"][value] = {
                            "total_trades": len(group_rewards),
                            "win_rate": (group_rewards > 0).mean() * 100 if len(group_rewards) > 0 else 0,
                            "total_return": group_rewards.sum(),
                        }

            for metric, threshold in self.performance_thresholds.items():
                value = stats.get(metric.replace("min_", "").replace("max_", ""), 0)
                if (metric.startswith("max_") and value > threshold) or (metric.startswith("min_") and value < threshold):
                    error_msg = f"Seuil non atteint pour {metric}: {value} vs {threshold}"
                    logger.warning(error_msg)
                    if self.alert_dispatcher:
                        asyncio.run(self.alert_dispatcher.send_alert(error_msg, priority=3))
                    else:
                        logger.warning("AlertDispatcher non fourni, alerte non envoyée")

            logger.info("Métriques calculées")
            return stats
        except Exception as e:
            error_msg = f"Erreur calcul métriques: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            if self.alert_dispatcher:
                asyncio.run(self.alert_dispatcher.send_alert(error_msg, priority=5))
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "total_return": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "risk_reward_ratio": 0.0,
                "consecutive_losses": 0,
                "avg_duration": 0.0,
                "iv_skew": 0.0,
                "entry_freq": 0.0,
                "period": "N/A",
            }