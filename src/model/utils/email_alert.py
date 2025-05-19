# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/utils/email_alert.py
# Envoi d'alertes critiques via Email pour MIA_IA_SYSTEM_v2_2025
#
# Version: 2.1.5
# Date: 2025-05-14
#
# Rôle: Envoie des alertes via Email en utilisant SMTP, avec journalisation des performances.
# Utilisé par: alert_manager.py
#
# Notes:
# - Conforme à structure.txt (version 2.1.5, 2025-05-14).
# - Supporte les suggestions 1 (position sizing dynamique), 4 (HMM/changepoint detection),
#   6 (drift detection) via alert_manager.py.
# - Intègre logs psutil dans data/logs/email_alert_performance.csv.
# - Récupère les identifiants SMTP via secret_manager.py.
# - Pas de références à dxFeed, obs_t, 320 features, ou 81 features.
# - Policies Note: The official directory for routing policies is src/model/router/policies.
# The src/model/policies directory is a residual and should be verified
# for removal to avoid import conflicts.

import logging
import smtplib
from datetime import datetime
from pathlib import Path

import pandas as pd
import psutil
from loguru import logger

from src.model.utils.alert_manager import AlertManager
from src.utils.secret_manager import SecretManager

# Configuration du logging Loguru
logger.remove()
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOG_DIR = BASE_DIR / "data" / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "email_alert.log", rotation="10 MB", level="INFO", encoding="utf-8"
)

# Configuration du logging standard pour compatibilité
logging.basicConfig(
    filename=str(LOG_DIR / "email_alert.log"),
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)


class EmailNotifier:
    """Envoie des alertes critiques via Email."""

    def __init__(self):
        """Initialise le notificateur Email avec les identifiants SMTP sécurisés."""
        start_time = datetime.now()
        try:
            self.alert_manager = AlertManager()
            secret_manager = SecretManager()
            email_secrets = secret_manager.get_secret("email_credentials")
            self.smtp_server = email_secrets.get("smtp_server", "")
            self.smtp_port = email_secrets.get("smtp_port", 587)
            self.sender_email = email_secrets.get("sender_email", "")
            self.sender_password = email_secrets.get("sender_password", "")
            self.receiver_email = email_secrets.get("receiver_email", "")
            if not all(
                [
                    self.smtp_server,
                    self.sender_email,
                    self.sender_password,
                    self.receiver_email,
                ]
            ):
                raise ValueError("Identifiants SMTP manquants ou invalides")
            logger.info("EmailNotifier initialisé avec succès.")
            logging.info("EmailNotifier initialisé avec succès.")
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("init_email_notifier", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur lors de l’initialisation de EmailNotifier: {str(e)}"
            logger.error(error_msg)
            logging.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "init_email_notifier", latency, success=False, error=str(e)
            )
            raise ValueError(f"Échec de l’initialisation Email: {e}")

    def send_alert(self, message: str, priority: int) -> bool:
        """Envoie une alerte via Email.

        Args:
            message: Message de l'alerte.
            priority: Niveau de priorité (1 à 5).

        Returns:
            bool: True si l'envoi réussit, False sinon.
        """
        start_time = datetime.now()
        try:
            subject = f"MIA_IA_SYSTEM Alert (Priority {priority})"
            body = f"{subject}\n\n{message}"
            email_message = f"Subject: {subject}\nTo: {self.receiver_email}\n\n{body}"
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=10) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, self.receiver_email, email_message)
            logger.info(f"Alerte Email envoyée: {message[:50]}...")
            logging.info(f"Alerte Email envoyée: {message[:50]}...")
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
            error_msg = f"Erreur envoi alerte Email: {str(e)}"
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
        """Journalise les performances CPU/mémoire dans email_alert_performance.csv."""
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
            log_path = LOG_DIR / "email_alert_performance.csv"
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
        notifier = EmailNotifier()
        notifier.send_alert("Test alerte Email", priority=4)
        print("Alerte envoyée avec succès")
    except ValueError as e:
        print(f"Erreur: {e}")
