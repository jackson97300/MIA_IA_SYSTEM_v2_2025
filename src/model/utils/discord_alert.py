# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/utils/discord_alert.py
# Envoi d'alertes critiques via Discord pour MIA_IA_SYSTEM_v2_2025
#
# Version: 2.1.5
# Date: 2025-05-14
#
# Rôle: Envoie des alertes via Discord en utilisant un webhook, avec journalisation des performances.
# Utilisé par: alert_manager.py
#
# Notes:
# - Conforme à structure.txt (version 2.1.5, 2025-05-14).
# - Supporte les suggestions 1 (position sizing dynamique), 4 (HMM/changepoint detection),
#   6 (drift detection) via alert_manager.py.
# - Intègre logs psutil dans data/logs/discord_alert_performance.csv.
# - Récupère l'URL du webhook via secret_manager.py.
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
    LOG_DIR / "discord_alert.log", rotation="10 MB", level="INFO", encoding="utf-8"
)

# Configuration du logging standard pour compatibilité
logging.basicConfig(
    filename=str(LOG_DIR / "discord_alert.log"),
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)


class DiscordNotifier:
    """Envoie des alertes critiques via Discord."""

    def __init__(self):
        """Initialise le notificateur Discord avec l'URL du webhook sécurisée."""
        start_time = datetime.now()
        try:
            self.alert_manager = AlertManager()
            secret_manager = SecretManager()
            discord_secrets = secret_manager.get_secret("discord_credentials")
            self.webhook_url = discord_secrets.get("webhook_url", "")
            if not self.webhook_url:
                raise ValueError("URL du webhook Discord manquante ou invalide")
            logger.info("DiscordNotifier initialisé avec succès.")
            logging.info("DiscordNotifier initialisé avec succès.")
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("init_discord_notifier", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur lors de l’initialisation de DiscordNotifier: {str(e)}"
            logger.error(error_msg)
            logging.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "init_discord_notifier", latency, success=False, error=str(e)
            )
            raise ValueError(f"Échec de l’initialisation Discord: {e}")

    def send_alert(self, message: str, priority: int) -> bool:
        """Envoie une alerte via Discord.

        Args:
            message: Message de l'alerte.
            priority: Niveau de priorité (1 à 5).

        Returns:
            bool: True si l'envoi réussit, False sinon.
        """
        start_time = datetime.now()
        try:
            payload = {
                # Limite Discord
                "content": f"🚨 MIA_IA_SYSTEM Alert (Priority {priority}): {message[:2000]}"
            }
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info(f"Alerte Discord envoyée: {message[:50]}...")
            logging.info(f"Alerte Discord envoyée: {message[:50]}...")
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "send_alert",
                latency,
                success=True,
                message=message[:50],
                priority=priority,
            )
            return True
        except Exception as e:
            error_msg = f"Erreur envoi alerte Discord: {str(e)}"
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
            return False

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ) -> None:
        """Journalise les performances CPU/mémoire dans discord_alert_performance.csv."""
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
            log_path = LOG_DIR / "discord_alert_performance.csv"
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
        notifier = DiscordNotifier()
        notifier.send_alert("Test alerte Discord", priority=3)
        print("Alerte envoyée avec succès")
    except ValueError as e:
        print(f"Erreur: {e}")
