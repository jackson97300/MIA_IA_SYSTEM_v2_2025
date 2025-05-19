# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/utils/error_tracker.py
# Capture des erreurs avec Sentry pour MIA_IA_SYSTEM_v2_2025
#
# Version: 2.1.5
# Date: 2025-05-14
#
# Rôle: Journalise les stack traces et contextes d’erreurs en production avec Sentry.
# Utilisé par: performance_logger.py, run_system.py, data_provider.py, risk_manager.py,
#             regime_detector.py, trade_probability.py.
#
# Notes:
# - Conforme à structure.txt (version 2.1.5, 2025-05-14).
# - Supporte les suggestions 1 (position sizing dynamique), 2 (loggers pour observabilité),
#   4 (HMM/changepoint detection), 7 (Safe RL/CVaR-PPO), 8 (Distributional RL/QR-DQN).
# - Intègre logs psutil dans data/logs/error_tracker_performance.csv.
# - Utilise alert_manager.py pour les alertes au lieu de telegram_alert.py.
# - Pas de références à dxFeed, obs_t, 320 features, ou 81 features.
# - Policies Note: The official directory for routing policies is src/model/router/policies.
# The src/model/policies directory is a residual and should be verified
# for removal to avoid import conflicts.

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
import psutil
import sentry_sdk
from loguru import logger
from prometheus_client import Counter

from src.model.utils.alert_manager import AlertManager
from src.utils.secret_manager import SecretManager

# Configuration du logging Loguru
logger.remove()
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOG_DIR = BASE_DIR / "data" / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "error_tracker.log", rotation="10 MB", level="INFO", encoding="utf-8"
)

# Configuration du logging standard pour compatibilité
logging.basicConfig(
    filename=str(LOG_DIR / "error_tracker.log"),
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)

# Compteur Prometheus pour les erreurs Sentry
sentry_errors = Counter(
    "sentry_errors_total",
    "Nombre total d’erreurs capturées par Sentry",
    ["market", "operation"],
)


def init_sentry(
    dsn: str = None, environment: str = "production", sample_rate: float = 1.0
) -> None:
    """Initialise Sentry pour la capture des erreurs.

    Args:
        dsn: DSN (Data Source Name) de Sentry pour la journalisation. Si None, utilise secret_manager.py.
        environment: Environnement d’exécution (ex. : 'production', 'development').
        sample_rate: Taux d’échantillonnage pour les traces (0.0 à 1.0).

    Raises:
        ValueError: Si le DSN est invalide ou si l’initialisation échoue.
    """
    start_time = datetime.now()
    try:
        if dsn is None:
            secret_manager = SecretManager()
            dsn = secret_manager.get_secret("sentry_dsn")
        sentry_sdk.init(
            dsn=dsn,
            environment=environment,
            traces_sample_rate=sample_rate,
            release="MIA_IA_SYSTEM_v2_2025@2.1.5",
        )
        logger.info(
            f"Sentry initialisé avec DSN: {dsn[:10]}... (environnement: {environment})"
        )
        logging.info(
            f"Sentry initialisé avec DSN: {dsn[:10]}... (environnement: {environment})"
        )
        latency = (datetime.now() - start_time).total_seconds()
        log_performance("init_sentry", latency, success=True)
    except Exception as e:
        error_msg = f"Erreur lors de l’initialisation de Sentry: {e}"
        logger.error(error_msg)
        logging.error(error_msg)
        latency = (datetime.now() - start_time).total_seconds()
        log_performance("init_sentry", latency, success=False, error=str(e))
        raise ValueError(f"Échec de l’initialisation de Sentry: {e}")


def capture_error(
    exception: Exception,
    context: Dict = None,
    market: str = "ES",
    operation: str = "unknown",
) -> None:
    """Capture une erreur avec contexte et l’envoie à Sentry.

    Args:
        exception: Exception à journaliser.
        context: Métadonnées supplémentaires (ex. : {'user': 'system', 'data_size': 1000}).
        market: Marché cible (défaut: 'ES').
        operation: Opération en cours (ex. : 'feature_calculation', 'model_switch').

    Returns:
        None
    """
    start_time = datetime.now()
    try:
        # Ajouter des tags pour le marché et l’opération
        sentry_sdk.set_tag("market", market)
        sentry_sdk.set_tag("operation", operation)

        # Ajouter le contexte
        if context:
            for key, value in context.items():
                sentry_sdk.set_context(key, value)

        # Capturer l’erreur
        sentry_sdk.capture_exception(exception)
        sentry_errors.labels(market=market, operation=operation).inc()
        alert_manager = AlertManager()
        error_msg = f"Erreur capturée pour {market} ({operation}): {str(exception)}"
        logger.error(f"Erreur capturée par Sentry: {exception}, contexte: {context}")
        logging.error(f"Erreur capturée par Sentry: {exception}, contexte: {context}")
        alert_manager.send_alert(error_msg, priority=4)

        # Journaliser les performances
        latency = (datetime.now() - start_time).total_seconds()
        log_performance(
            "capture_error", latency, success=True, market=market, operation=operation
        )
    except Exception as e:
        error_msg = f"Erreur lors de la capture par Sentry: {e}"
        logger.error(error_msg)
        logging.error(error_msg)
        alert_manager = AlertManager()
        alert_manager.send_alert(error_msg, priority=4)
        latency = (datetime.now() - start_time).total_seconds()
        log_performance(
            "capture_error",
            latency,
            success=False,
            error=str(e),
            market=market,
            operation=operation,
        )


def log_performance(
    operation: str, latency: float, success: bool, error: str = None, **kwargs
) -> None:
    """Journalise les performances CPU/mémoire dans error_tracker_performance.csv."""
    try:
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
        cpu_percent = psutil.cpu_percent()
        if memory_usage > 512:
            alert_msg = f"ALERTE: Utilisation mémoire élevée ({memory_usage:.2f} MB) pour {operation}"
            logger.warning(alert_msg)
            alert_manager = AlertManager()
            alert_manager.send_alert(alert_msg, priority=3)
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "latency": latency,
            "success": success,
            "error": error,
            "memory_usage_mb": memory_usage,
            "cpu_usage_percent": cpu_percent,
            **kwargs,
        }
        log_df = pd.DataFrame([log_entry])
        log_path = LOG_DIR / "error_tracker_performance.csv"
        log_df.to_csv(
            log_path,
            mode="a",
            header=not log_path.exists(),
            index=False,
            encoding="utf-8",
        )
        logger.info(f"Performance journalisée pour {operation}. CPU: {cpu_percent}%")
    except Exception as e:
        error_msg = f"Erreur journalisation performance: {str(e)}"
        logger.error(error_msg)
        alert_manager = AlertManager()
        alert_manager.send_alert(error_msg, priority=4)


# Exemple d’utilisation (à supprimer avant production)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    try:
        # Remplacer par un DSN Sentry valide
        init_sentry(dsn="https://example@sentry.io/123", environment="development")
        context = {"market": "ES", "operation": "test"}
        capture_error(
            ValueError("Erreur de test"), context=context, market="ES", operation="test"
        )
        print("Erreur capturée avec succès")
    except ValueError as e:
        print(f"Erreur: {e}")
