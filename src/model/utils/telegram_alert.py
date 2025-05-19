# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/utils/telegram_alert.py
# Envoi d'alertes critiques via Telegram pour MIA_IA_SYSTEM_v2_2025
#
# Version: 2.1.5
# Date: 2025-05-14
#
# Rôle: Envoie des alertes Telegram pour informer les équipes des erreurs critiques ou des événements clés.
# Utilisé par: error_tracker.py, alert_manager.py, risk_manager.py, regime_detector.py, drift_detector.py.
#
# Notes:
# - Conforme à structure.txt (version 2.1.5, 2025-05-14).
# - Supporte les suggestions 1 (position sizing dynamique), 4 (HMM/changepoint detection),
#   6 (drift detection).
# - Intègre logs psutil dans data/logs/telegram_alert_performance.csv.
# - Utilise secret_manager.py pour récupérer bot_token et chat_id.
# - Pas de références à dxFeed, obs_t, 320 features, ou 81 features.
# - Policies Note: The official directory for routing policies is src/model/router/policies.
# The src/model/policies directory is a residual and should be verified
# for removal to avoid import conflicts.

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import psutil
import requests
from loguru import logger

from src.model.utils.alert_manager import AlertManager
from src.utils.secret_manager import SecretManager

# Configuration du logging Loguru
logger.remove()
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOG_DIR = BASE_DIR / "data" / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "telegram_alert.log", rotation="10 MB", level="INFO", encoding="utf-8"
)

# Configuration du logging standard pour compatibilité
logging.basicConfig(
    filename=str(LOG_DIR / "telegram_alert.log"),
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)


class TelegramAlert:
    """Envoie des alertes critiques via Telegram."""

    def __init__(self):
        """Initialise le client Telegram avec les identifiants sécurisés."""
        start_time = datetime.now()
        try:
            self.alert_manager = AlertManager()
            secret_manager = SecretManager()
            telegram_secrets = secret_manager.get_secret("telegram_credentials")
            self.bot_token = telegram_secrets.get("bot_token", "")
            self.chat_id = telegram_secrets.get("chat_id", "")
            if not self.bot_token or not self.chat_id:
                raise ValueError("Identifiants Telegram manquants ou invalides")
            logger.info("Client Telegram initialisé avec succès.")
            logging.info("Client Telegram initialisé avec succès.")
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("init_telegram_alert", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur lors de l’initialisation du client Telegram: {e}"
            logger.error(error_msg)
            logging.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "init_telegram_alert", latency, success=False, error=str(e)
            )
            raise ValueError(f"Échec de l’initialisation Telegram: {e}")

    def send_alert(self, message: str, priority: int) -> None:
        """Envoie une alerte via Telegram.

        Args:
            message: Message de l'alerte.
            priority: Niveau de priorité (1 à 5, alertes envoyées si priority >= 3).

        Returns:
            None
        """
        start_time = datetime.now()
        try:
            if priority >= 3:  # Seulement pour les alertes critiques
                url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
                payload = {
                    "chat_id": self.chat_id,
                    "text": message[:4096],
                }  # Limite Telegram
                response = requests.post(url, json=payload, timeout=10)
                response.raise_for_status()
                logger.info(f"Alerte Telegram envoyée: {message[:50]}...")
                logging.info(f"Alerte Telegram envoyée: {message[:50]}...")
            else:
                logger.debug(
                    f"Alerte non envoyée (priorité insuffisante: {priority}): {message[:50]}..."
                )
                logging.debug(
                    f"Alerte non envoyée (priorité insuffisante: {priority}): {message[:50]}..."
                )
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "send_alert",
                latency,
                success=True,
                message=message[:50],
                priority=priority,
            )
        except Exception as e:
            error_msg = f"Erreur envoi alerte Telegram: {str(e)}"
            logger.error(error_msg)
            logging.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "send_alert",
                latency,
                success=False,
                error=str(e),
                message=message[:50],
                priority=priority,
            )

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ) -> None:
        """Journalise les performances CPU/mémoire dans telegram_alert_performance.csv."""
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
            cpu_percent = psutil.cpu_percent()
            if memory_usage > 512:
                alert_msg = f"ALERTE: Utilisation mémoire élevée ({memory_usage:.2f} MB) pour {operation}"
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=3)
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
            log_path = LOG_DIR / "telegram_alert_performance.csv"
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
        except Exception as e:
            error_msg = f"Erreur journalisation performance: {str(e)}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)


# Exemple d’utilisation (à supprimer avant production)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    try:
        alert = TelegramAlert()
        alert.send_alert("Test alerte critique", priority=4)
        alert.send_alert("Test alerte non critique", priority=2)
        print("Alertes envoyées avec succès")
    except ValueError as e:
        print(f"Erreur: {e}")
