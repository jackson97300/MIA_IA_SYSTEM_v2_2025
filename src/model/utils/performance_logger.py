# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/model/utils/performance_logger.py
# Rôle : Gère le performance_buffer pour journaliser les métriques de performance.
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, loguru>=0.7.0,<1.0.0
#
# Outputs :
# - data/logs/<market>/*_performance.csv
#
# Notes :
# - Compatible avec MIA_IA_SYSTEM_v2_2025 pour journaliser les performances des opérations critiques.
# - Utilisé par mia_switcher.py et algo_performance_logger.py pour séparer les métriques de performance.
# - Policies Note: The official directory for routing policies is src/model/router/policies.
# The src/model/policies directory is a residual and should be verified
# for removal to avoid import conflicts.

import os
from datetime import datetime

import pandas as pd
from loguru import logger


class PerformanceLogger:
    """Gère le performance_buffer pour les métriques de performance."""

    def __init__(self, market: str = "ES"):
        """Initialise le logger pour un marché.

        Args:
            market (str): Marché (ex. : 'ES').
        """
        self.market = market
        logger.info(f"PerformanceLogger initialisé pour {market}")

    def log(
        self,
        operation: str,
        latency: float,
        success: bool = True,
        error: str = None,
        **kwargs,
    ) -> None:
        """Journalise une métrique de performance.

        Args:
            operation (str): Nom de l’opération.
            latency (float): Temps d’exécution (secondes).
            success (bool): Succès de l’opération.
            error (str, optional): Message d’erreur.
            **kwargs: Métriques supplémentaires.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "latency": latency,
            "success": success,
            "error": error,
            **kwargs,
        }
        log_path = f"data/logs/{self.market}/{operation}_performance.csv"
        pd.DataFrame([log_entry]).to_csv(
            log_path, mode="a", header=not os.path.exists(log_path), index=False
        )
        logger.info(f"Métrique journalisée pour {operation}: {log_entry}")
