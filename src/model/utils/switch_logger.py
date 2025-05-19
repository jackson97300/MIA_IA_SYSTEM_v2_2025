# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/model/utils/switch_logger.py
# Rôle : Gère le switch_buffer pour journaliser les événements de switching.
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, loguru>=0.7.0,<1.0.0
#
# Outputs :
# - data/logs/trading/decision_log.csv
#
# Notes :
# - Compatible avec MIA_IA_SYSTEM_v2_2025 pour journaliser les décisions de bascule entre modèles.
# - Utilisé par mia_switcher.py pour séparer les journaux de switching des métriques de performance.
# - Policies Note: The official directory for routing policies is src/model/router/policies.
# The src/model/policies directory is a residual and should be verified
# for removal to avoid import conflicts.

import os
from datetime import datetime

import pandas as pd
from loguru import logger


class SwitchLogger:
    """Gère le switch_buffer pour les événements de switching."""

    def __init__(self, market: str = "ES"):
        """Initialise le logger pour un marché.

        Args:
            market (str): Marché (ex. : 'ES').
        """
        self.market = market
        logger.info(f"SwitchLogger initialisé pour {market}")

    def log(self, decision: str, reason: str, regime: str, **kwargs) -> None:
        """Journalise un événement de switching.

        Args:
            decision (str): Décision prise (ex. : 'switch_to_sac').
            reason (str): Raison du switch (ex. : 'trend_detected').
            regime (str): Régime de marché (trend, range, defensive).
            **kwargs: Données supplémentaires.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "decision": decision,
            "reason": reason,
            "regime": regime,
            **kwargs,
        }
        log_path = "data/logs/trading/decision_log.csv"
        pd.DataFrame([log_entry]).to_csv(
            log_path, mode="a", header=not os.path.exists(log_path), index=False
        )
        logger.info(f"Switch journalisé: {log_entry}")
