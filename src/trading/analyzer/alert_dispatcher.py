```python
# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/trading/alert_dispatcher.py
# Rôle : Gère l’envoi d’alertes multicanaux (Telegram, Discord, email) pour les événements critiques.
#
# Version : 2.1.4
# Date : 2025-05-15
#
# Dépendances :
# - python-telegram-bot>=20.0,<21.0
# - discord.py>=2.0.0,<3.0.0
# - pandas>=2.0.0,<3.0.0
#
# Inputs :
# - Configuration via config/market_config.yaml
# - Messages d’alerte avec priorité (1 à 5)
#
# Outputs :
# - Alertes envoyées via Telegram, Discord, email
# - Logs dans data/logs/alert_dispatcher.log
# - Méta-données des alertes dans market_memory.db (Proposition 2, Étape 1)
#
# Notes :
# - Priorité ≥ 4 déclenche Telegram, ≥ 3 Discord, ≥ 5 email.
# - Stocke les alertes dans market_memory.db pour analyse des méta-données.
# - Compatible avec les narratifs LLM (Proposition 1) pour signaler les échecs d’appel GPT.
# - Implémente des retries pour Telegram/Discord (max 3 tentatives).
# - Utilise l’encodage UTF-8 pour tous les fichiers.

import logging
import os
import json
import time
import sqlite3
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict

import telegram
import discord
from discord import Intents
import smtplib
from email.mime.text import MIMEText

# Chemins relatifs
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
MARKET_MEMORY_DB = BASE_DIR / "data" / "market_memory.db"
LOG_DIR = BASE_DIR / "data" / "logs"

# Configurer logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "alert_dispatcher.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

class AlertDispatcher:
    def __init__(self, config: Dict):
        self.config = config
        self.max_retries = 3
        self.retry_delay = 2.0  # secondes

        # Vérifier les clés de configuration
        required_configs = {
            "telegram": ["token", "chat_id"],
            "discord": ["channel_id"],
            "email": ["sender", "receiver", "smtp_server", "smtp_port", "username", "password"],
        }
        for section, keys in required_configs.items():
            if section not in config or not all(key in config[section] for key in keys):
                logger.warning(f"Configuration incomplète pour {section}: {keys}")
        
        # Initialiser les clients
        self.telegram_bot = telegram.Bot(token=config.get("telegram", {}).get("token", ""))
        self.discord_client = discord.Client(intents=Intents.default())
        self.email_config = config.get("email", {})

    async def send_telegram_alert(self, message: str, priority: int) -> bool:
        """Envoie une alerte via Telegram avec retries."""
        if priority < 4:
            return False
        for attempt in range(self.max_retries):
            try:
                chat_id = self.config.get("telegram", {}).get("chat_id", "")
                await self.telegram_bot.send_message(chat_id=chat_id, text=message)
                logger.info(f"Alerte Telegram envoyée: {message}")
                return True
            except Exception as e:
                error_msg = f"Erreur envoi Telegram (tentative {attempt+1}/{self.max_retries}): {str(e)}"
                logger.error(error_msg)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
        return False

    async def send_discord_alert(self, message: str, priority: int) -> bool:
        """Envoie une alerte via Discord avec retries."""
        if priority < 3:
            return False
        for attempt in range(self.max_retries):
            try:
                channel_id = self.config.get("discord", {}).get("channel_id", 0)
                channel = self.discord_client.get_channel(channel_id)
                if channel:
                    await channel.send(message)
                    logger.info(f"Alerte Discord envoyée: {message}")
                    return True
                else:
                    logger.error(f"Channel Discord {channel_id} non trouvé")
            except Exception as e:
                error_msg = f"Erreur envoi Discord (tentative {attempt+1}/{self.max_retries}): {str(e)}"
                logger.error(error_msg)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
        return False

    def send_email_alert(self, message: str, priority: int) -> bool:
        """Envoie une alerte via email."""
        if priority < 5:
            return False
        try:
            msg = MIMEText(message)
            msg["Subject"] = f"MIA Alert - Priority {priority}"
            msg["From"] = self.email_config.get("sender", "")
            msg["To"] = self.email_config.get("receiver", "")
            with smtplib.SMTP(self.email_config.get("smtp_server", ""), self.email_config.get("smtp_port", 587)) as server:
                server.starttls()
                server.login(self.email_config.get("username", ""), self.email_config.get("password", ""))
                server.send_message(msg)
            logger.info(f"Alerte email envoyée: {message}")
            return True
        except Exception as e:
            error_msg = f"Erreur envoi email: {str(e)}"
            logger.error(error_msg)
            return False

    def log_alert_metadata(self, message: str, priority: int) -> None:
        """Stocke les méta-données des alertes dans market_memory.db."""
        try:
            alert_metadata = {
                "timestamp": datetime.now().isoformat(),
                "message": message[:100],  # Limiter pour SQLite
                "priority": priority,
                "channels": ["telegram" if priority >= 4 else None, "discord" if priority >= 3 else None, "email" if priority >= 5 else None],
            }
            conn = sqlite3.connect(MARKET_MEMORY_DB)
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS alert_metadata (
                    timestamp TEXT,
                    event_type TEXT,
                    metadata TEXT
                )
                """
            )
            cursor.execute(
                """
                INSERT INTO alert_metadata (timestamp, event_type, metadata)
                VALUES (?, ?, ?)
                """,
                (
                    alert_metadata["timestamp"],
                    "alert",
                    json.dumps(alert_metadata),
                ),
            )
            conn.commit()
            conn.close()
            logger.info("Méta-données de l’alerte stockées dans market_memory.db")
        except Exception as e:
            error_msg = f"Erreur stockage méta-données alerte: {str(e)}"
            logger.error(error_msg)

    async def send_alert(self, message: str, priority: int = 1) -> None:
        """Envoie une alerte via les canaux configurés selon la priorité."""
        try:
            start_time = time.time()
            logger.info(f"Envoi alerte: {message} (priorité {priority})")

            # Envoyer les alertes
            await asyncio.gather(
                self.send_telegram_alert(message, priority),
                self.send_discord_alert(message, priority),
            )
            self.send_email_alert(message, priority)

            # Stocker les méta-données (Proposition 2, Étape 1)
            self.log_alert_metadata(message, priority)

            logger.info(f"Alerte envoyée en {time.time() - start_time:.2f}s")
        except Exception as e:
            error_msg = f"Erreur envoi alerte: {str(e)}"
            logger.error(error_msg)
