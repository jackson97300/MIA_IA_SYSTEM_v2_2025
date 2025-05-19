# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/utils/secret_manager.py
# Gestion des secrets avec AWS KMS pour MIA_IA_SYSTEM_v2_2025
#
# Version: 2.1.5
# Date: 2025-05-14
#
# Rôle: Récupère les secrets chiffrés (ex. : identifiants IQFeed, AWS, Telegram, Discord, Email, risk_manager, regime_detector, trade_probability)
#       via AWS Key Management Service (KMS).
# Utilisé par: data_provider.py, telegram_alert.py, discord_alert.py, email_alert.py, data_lake.py, risk_manager.py, regime_detector.py, trade_probability.py.
#
# Notes:
# - Conforme à structure.txt (version 2.1.5, 2025-05-14).
# - Supporte les suggestions 1 (position sizing dynamique), 4 (HMM/changepoint detection),
#   7 (Safe RL/CVaR-PPO), 8 (Distributional RL/QR-DQN).
# - Intègre logs psutil dans data/logs/secret_manager_performance.csv.
# - Utilise error_tracker.py pour capturer les erreurs et alert_manager.py pour les alertes.
# - Pas de références à dxFeed, obs_t, 320 features, ou 81 features.
# - Policies Note: The official directory for routing policies is src/model/router/policies.
# The src/model/policies directory is a residual and should be verified
# for removal to avoid import conflicts.

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

import boto3
import pandas as pd
import psutil
from botocore.exceptions import ClientError
from loguru import logger

from src.model.utils.alert_manager import AlertManager
from src.utils.error_tracker import capture_error

# Configuration du logging Loguru
logger.remove()
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOG_DIR = BASE_DIR / "data" / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "secret_manager.log", rotation="10 MB", level="INFO", encoding="utf-8"
)

# Configuration du logging standard pour compatibilité
logging.basicConfig(
    filename=str(LOG_DIR / "secret_manager.log"),
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)


class SecretManager:
    """Gère les secrets chiffrés avec AWS KMS."""

    def __init__(self):
        """Initialise le client KMS."""
        start_time = datetime.now()
        try:
            self.kms = boto3.client("kms")
            self.alert_manager = AlertManager()
            logger.info("Client KMS initialisé avec succès.")
            logging.info("Client KMS initialisé avec succès.")
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("init_secret_manager", latency, success=True)
        except ClientError as e:
            error_msg = f"Erreur lors de l’initialisation du client KMS: {e}"
            logger.error(error_msg)
            logging.error(error_msg)
            capture_error(
                e,
                context={"operation": "init_secret_manager"},
                operation="init_secret_manager",
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "init_secret_manager", latency, success=False, error=str(e)
            )
            raise ValueError(f"Échec de l’initialisation KMS: {e}")

    def get_secret(self, secret_id: str) -> Dict:
        """Récupère un secret chiffré via AWS KMS.

        Args:
            secret_id: Identifiant du secret chiffré (chaîne encodée en base64).

        Returns:
            Dict: Dictionnaire contenant les identifiants déchiffrés.

        Raises:
            ValueError: Si le déchiffrement échoue ou si le secret_id est invalide.
        """
        start_time = datetime.now()
        try:
            response = self.kms.decrypt(CiphertextBlob=secret_id.encode())
            secret = response["Plaintext"].decode()
            secret_dict = {
                "aws_credentials": {"access_key": "xxx", "secret_key": "yyy"},
                "iqfeed_api_key": "zzz",
                "sentry_dsn": "aaa",
                # Pour telegram_alert.py
                "telegram_credentials": {
                    "bot_token": "test_bot_token",
                    "chat_id": "test_chat_id",
                },
                "discord_credentials": {
                    "webhook_url": "test_discord_webhook_url"
                },  # Pour discord_alert.py
                "email_credentials": {
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "sender_email": "test_email@gmail.com",
                    "sender_password": "test_password",
                    "receiver_email": "test_receiver@example.com",
                },  # Pour email_alert.py
                "risk_manager_api_key": "bbb",  # Pour risk_manager.py (suggestion 1)
                # Pour regime_detector.py (suggestion 4)
                "regime_detector_data_key": "ccc",
                # Pour trade_probability.py (Safe RL, suggestion 7)
                "gpu_cluster_key": "ddd",
                # Pour trade_probability.py (QR-DQN, suggestion 8)
                "ray_cluster_key": "eee",
            }.get(secret_id, {"secret": secret})
            logger.debug(f"Secret récupéré pour {secret_id[:10]}...")
            logging.debug(f"Secret récupéré pour {secret_id[:10]}...")
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "get_secret", latency, success=True, secret_id=secret_id[:10]
            )
            self.alert_manager.send_alert(
                f"Secret récupéré: {secret_id[:10]}...", priority=2
            )
            return secret_dict
        except ClientError as e:
            error_msg = (
                f"Erreur lors du déchiffrement du secret {secret_id[:10]}...: {e}"
            )
            logger.error(error_msg)
            logging.error(error_msg)
            capture_error(
                e, context={"secret_id": secret_id[:10]}, operation="get_secret"
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "get_secret",
                latency,
                success=False,
                error=str(e),
                secret_id=secret_id[:10],
            )
            raise ValueError(f"Erreur KMS: {e}")
        except (TypeError, ValueError) as e:
            error_msg = f"Format de secret_id invalide: {e}"
            logger.error(error_msg)
            logging.error(error_msg)
            capture_error(
                e, context={"secret_id": secret_id[:10]}, operation="get_secret"
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "get_secret",
                latency,
                success=False,
                error=str(e),
                secret_id=secret_id[:10],
            )
            raise ValueError(f"secret_id invalide: {e}")

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ) -> None:
        """Journalise les performances CPU/mémoire dans secret_manager_performance.csv."""
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
                "cpu_percent": cpu_percent,
                **kwargs,
            }
            log_df = pd.DataFrame([log_entry])
            log_path = LOG_DIR / "secret_manager_performance.csv"
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
            capture_error(
                e, context={"operation": operation}, operation="log_performance"
            )
            self.alert_manager.send_alert(error_msg, priority=4)


# Exemple d’utilisation (à supprimer avant production)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    try:
        secret_manager = SecretManager()
        # Exemple avec un secret_id fictif
        secret = secret_manager.get_secret("AQICAH...")
        print(f"Secret récupéré: {secret}")
    except ValueError as e:
        print(f"Erreur: {e}")
