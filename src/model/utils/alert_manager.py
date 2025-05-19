# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/model/utils/alert_manager.py
# Rôle : Centralise la gestion des alertes multi-canaux (Telegram, Discord, Email) pour MIA_IA_SYSTEM_v2_2025,
#        avec support des priorités et journalisation détaillée.
#
# Version : 2.1.5
# Date : 2025-05-14
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, psutil>=5.9.8,<6.0.0, boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0,
#   pyyaml>=6.0.0,<7.0.0, signal, requests>=2.28.0,<3.0.0, smtplib, pickle
# - src/model/utils/config_manager.py
# - src/utils/telegram_alert.py
# - src/utils/discord_alert.py
# - src/utils/email_alert.py
#
# Inputs :
# - config/es_config.yaml via config_manager
# - Message à envoyer et priorité
#
# Outputs :
# - Alertes envoyées via Telegram/Discord/Email
# - Logs dans data/logs/alert_manager.log
# - Logs de performance dans data/logs/alert_performance.csv
# - Snapshots JSON compressés dans data/cache/alert_manager/<market>/*.json.gz
# - Sauvegardes incrémentielles dans data/checkpoints/alert_manager/<market>/*.json.gz
# - Alertes locales (priorité 5) dans data/alerts/local/<market>/*.json (configurable)
# - Cache persistant dans data/cache/alert_cache.pkl
#
# Notes :
# - Fournit des alertes pour detect_regime.py, train_sac.py, transformer_policy.py, custom_mlp_policy.py, main_router.py.
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence) via les modules appelants.
# - Utilise IQFeed exclusivement via data_provider.py (implicite via les modules appelants).
# - Implémente retries (max 3, délai 2^attempt secondes) pour les appels critiques.
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des opérations.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.
# - Tests unitaires disponibles dans tests/test_alert_manager.py.
# - Gestion des priorités :
#   - Priorité 1–2 : Discord (journalisation uniquement)
#   - Priorité 3 : Telegram + Discord
#   - Priorité 4 : Telegram + Discord + Email
#   - Priorité 5 : Telegram + Discord + Email + stockage local (configurable)

import gzip
import json
import pickle
import signal
import time
import traceback
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import pandas as pd
import psutil
from loguru import logger
from prometheus_client import Counter

from src.model.utils.config_manager import get_config
from src.utils.discord_alert import DiscordNotifier
from src.utils.email_alert import EmailNotifier
from src.utils.telegram_alert import TelegramAlert

# Configuration logging
logger.remove()
BASE_DIR = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
LOG_DIR = BASE_DIR / "data" / "logs"
CACHE_DIR = BASE_DIR / "data" / "cache" / "alert_manager"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints" / "alert_manager"
LOCAL_ALERT_DIR = BASE_DIR / "data" / "alerts" / "local"
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOCAL_ALERT_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "alert_manager.log", rotation="10 MB", level="INFO", encoding="utf-8"
)
logger.add(lambda msg: print(msg, end=""), level="INFO")

# Constantes
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Secondes, exponentiel
PERF_LOG_PATH = LOG_DIR / "alert_performance.csv"
CACHE_PATH = BASE_DIR / "data" / "cache" / "alert_cache.pkl"
MAX_CACHE_SIZE = 1000
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB

# Compteurs Prometheus
telegram_alerts_total = Counter(
    "telegram_alerts_total",
    "Nombre total d’alertes Telegram envoyées",
    ["market", "priority"],
)
discord_alerts_total = Counter(
    "discord_alerts_total",
    "Nombre total d’alertes Discord envoyées",
    ["market", "priority"],
)
email_alerts_total = Counter(
    "email_alerts_total",
    "Nombre total d’alertes Email envoyées",
    ["market", "priority"],
)

# Cache global pour les résultats de send_alert
alert_cache = OrderedDict()

# Configuration des canaux par priorité
PRIORITY_CHANNELS = {
    1: ["discord"],
    2: ["discord"],
    3: ["telegram", "discord"],
    4: ["telegram", "discord", "email"],
    5: ["telegram", "discord", "email", "local"],
}


class AlertManager:
    """
    Gère l'envoi d'alertes via Telegram, Discord, Email, avec priorités et journalisation détaillée.

    Attributes:
        config (Dict): Configuration chargée depuis es_config.yaml.
        snapshot_dir (Path): Dossier pour snapshots JSON.
        checkpoint_dir (Path): Dossier pour checkpoints.
        local_alert_dir (Path): Dossier pour stockage local des alertes (priorité 5).
        perf_log (Path): Fichier pour logs de performance.
        market (str): Marché (ex. : ES, MNQ).
        notifiers (List[Tuple[str, Any]]): Liste des notificateurs avec leurs noms de canal.
    """

    def __init__(
        self,
        config_path: str = str(BASE_DIR / "config" / "es_config.yaml"),
        market: str = "ES",
    ):
        """
        Initialise le gestionnaire d'alertes.

        Args:
            config_path (str): Chemin vers le fichier de configuration.
            market (str): Marché (ex. : ES, MNQ).
        """
        start_time = time.time()
        self.market = market
        try:
            self.config = get_config(config_path)
            self.snapshot_dir = CACHE_DIR / market
            self.checkpoint_dir = CHECKPOINT_DIR / market
            self.local_alert_dir = LOCAL_ALERT_DIR / market
            self.perf_log = PERF_LOG_PATH
            self.snapshot_dir.mkdir(exist_ok=True)
            self.checkpoint_dir.mkdir(exist_ok=True)
            self.local_alert_dir.mkdir(exist_ok=True)
            PERF_LOG_PATH.parent.mkdir(exist_ok=True)

            # Validation configuration
            required_keys = [
                "telegram",
                "discord",
                "email",
                "local_storage",
                "s3_bucket",
                "s3_prefix",
                "cache",
            ]
            missing_keys = [key for key in required_keys if key not in self.config]
            if missing_keys:
                raise ValueError(
                    f"Configuration es_config.yaml incomplète: clés manquantes {missing_keys}"
                )

            # Initialisation des notificateurs
            self.notifiers: List[Tuple[str, Any]] = [
                ("telegram", TelegramAlert()),
                ("discord", DiscordNotifier()),
                ("email", EmailNotifier()),
            ]
            notifier_names = [name for name, _ in self.notifiers]
            success_msg = f"AlertManager initialisé avec succès pour {market} avec notificateurs: {notifier_names}"

            # Charger le cache persistant
            global alert_cache
            if (
                self.config.get("cache", {}).get("enabled", True)
                and CACHE_PATH.exists()
            ):
                try:
                    with open(CACHE_PATH, "rb") as f:
                        alert_cache = pickle.load(f)
                    logger.info(f"Cache chargé depuis {CACHE_PATH}")
                except Exception as e:
                    logger.warning(
                        f"Erreur chargement cache depuis {CACHE_PATH}: {str(e)}"
                    )

            logger.info(success_msg)
            self.log_performance("init", time.time() - start_time, success=True)
            signal.signal(signal.SIGINT, self.handle_sigint)
        except Exception as e:
            error_msg = f"Erreur initialisation AlertManager pour {market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.log_performance(
                "init", time.time() - start_time, success=False, error=str(e)
            )
            raise

    def handle_sigint(self, signum, frame):
        """Gère l'arrêt propre sur SIGINT."""
        start_time = time.time()
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "status": "SIGINT",
            "market": self.market,
        }
        snapshot_path = self.snapshot_dir / f'sigint_{snapshot["timestamp"]}.json.gz'
        try:
            with gzip.open(snapshot_path, "wt", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=4)
            self.save_cache()
            success_msg = f"Arrêt propre sur SIGINT pour {self.market}"
            logger.info(success_msg)
            self.log_performance(
                "handle_sigint", time.time() - start_time, success=True
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde snapshot SIGINT pour {self.market}: {str(e)}"
            )
            logger.error(error_msg)
            self.log_performance(
                "handle_sigint", time.time() - start_time, success=False, error=str(e)
            )
        exit(0)

    def save_cache(self) -> None:
        """Persiste le cache alert_cache sur disque."""
        try:
            if self.config.get("cache", {}).get("enabled", True):
                with open(CACHE_PATH, "wb") as f:
                    pickle.dump(alert_cache, f)
                logger.info(f"Cache sauvegardé dans {CACHE_PATH}")
                self.log_performance(
                    "save_cache", 0, success=True, cache_size=len(alert_cache)
                )
        except Exception as e:
            error_msg = f"Erreur sauvegarde cache pour {self.market}: {str(e)}"
            logger.error(error_msg)
            self.send_alert(error_msg, priority=3)
            self.log_performance("save_cache", 0, success=False, error=str(e))

    def log_performance(
        self,
        operation: str,
        latency: float,
        success: bool = True,
        error: str = None,
        **kwargs,
    ) -> None:
        """
        Enregistre les performances (latence, mémoire, CPU) dans alert_performance.csv.

        Args:
            operation (str): Nom de l’opération (ex. : send_alert).
            latency (float): Temps d’exécution en secondes.
            success (bool): Indique si l’opération a réussi.
            error (str, optional): Message d’erreur si applicable.
            **kwargs: Paramètres supplémentaires (ex. : channel, priority).
        """
        cache_key = f"{self.market}_{operation}_{hash(str(latency))}_{hash(str(error))}"
        if cache_key in alert_cache:
            return
        while len(alert_cache) > MAX_CACHE_SIZE:
            alert_cache.popitem(last=False)

        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
            cpu_percent = psutil.cpu_percent()
            confidence_drop_rate = 1.0 if success else 0.0  # Simplifié pour Phase 8
            if memory_usage > 1024:
                warning_msg = f"ALERTE: Usage mémoire élevé ({memory_usage:.2f} MB) pour {self.market}"
                logger.warning(warning_msg)
                self.send_alert(warning_msg, priority=5)
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "latency": latency,
                "success": success,
                "error": error,
                "memory_usage_mb": memory_usage,
                "cpu_percent": cpu_percent,
                "confidence_drop_rate": confidence_drop_rate,
                "market": self.market,
                **kwargs,
            }
            log_df = pd.DataFrame([log_entry])

            def save_log():
                if not PERF_LOG_PATH.exists():
                    log_df.to_csv(PERF_LOG_PATH, index=False, encoding="utf-8")
                else:
                    log_df.to_csv(
                        PERF_LOG_PATH,
                        mode="a",
                        header=False,
                        index=False,
                        encoding="utf-8",
                    )

            self.with_retries(save_log)
            logger.info(
                f"Performance journalisée pour {operation}. CPU: {cpu_percent}%"
            )
            self.save_snapshot("log_performance", log_entry)
            alert_cache[cache_key] = True
            self.save_cache()
        except Exception as e:
            error_msg = (
                f"Erreur journalisation performance pour {self.market}: {str(e)}"
            )
            logger.error(error_msg)
            self.send_alert(error_msg, priority=3)

    def save_snapshot(
        self, snapshot_type: str, data: Dict, compress: bool = True
    ) -> None:
        """
        Sauvegarde un instantané JSON des résultats, compressé avec gzip.

        Args:
            snapshot_type (str): Type de snapshot (ex. : send_alert).
            data (Dict): Données à sauvegarder.
            compress (bool): Compresser avec gzip (défaut : True).
        """
        start_time = time.time()
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = {
                "timestamp": timestamp,
                "type": snapshot_type,
                "market": self.market,
                "data": data,
            }
            snapshot_path = (
                self.snapshot_dir / f"snapshot_{snapshot_type}_{timestamp}.json"
            )

            def write_snapshot():
                if compress:
                    with gzip.open(f"{snapshot_path}.gz", "wt", encoding="utf-8") as f:
                        json.dump(snapshot, f, indent=4)
                else:
                    with open(snapshot_path, "w", encoding="utf-8") as f:
                        json.dump(snapshot, f, indent=4)

            self.with_retries(write_snapshot)
            save_path = f"{snapshot_path}.gz" if compress else snapshot_path
            file_size = os.path.getsize(save_path) / 1024 / 1024
            if file_size > 1.0:
                alert_msg = (
                    f"Snapshot size {file_size:.2f} MB exceeds 1 MB pour {self.market}"
                )
                logger.warning(alert_msg)
                self.send_alert(alert_msg, priority=3)
            latency = time.time() - start_time
            logger.info(
                f"Snapshot {snapshot_type} sauvegardé pour {self.market}: {save_path}"
            )
            self.log_performance(
                "save_snapshot", latency, success=True, snapshot_size_mb=file_size
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot {snapshot_type} pour {self.market}: {str(e)}"
            logger.error(error_msg)
            self.send_alert(error_msg, priority=3)
            self.log_performance(
                "save_snapshot", time.time() - start_time, success=False, error=str(e)
            )

    def checkpoint(
        self, data: pd.DataFrame, data_type: str = "alert_manager_state"
    ) -> None:
        """
        Sauvegarde incrémentielle des données toutes les 5 minutes avec versionnage (5 versions).

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : alert_manager_state).
        """
        start_time = time.time()
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_data = {
                "timestamp": timestamp,
                "num_rows": len(data),
                "columns": list(data.columns),
                "data_type": data_type,
                "market": self.market,
            }
            checkpoint_path = (
                self.checkpoint_dir / f"alert_manager_{data_type}_{timestamp}.json.gz"
            )
            checkpoint_versions = []

            def write_checkpoint():
                with gzip.open(checkpoint_path, "wt", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, indent=4)
                data.to_csv(
                    checkpoint_path.with_suffix(".csv"), index=False, encoding="utf-8"
                )

            self.with_retries(write_checkpoint)
            checkpoint_versions.append(checkpoint_path)
            if len(checkpoint_versions) > 5:
                oldest = checkpoint_versions.pop(0)
                if oldest.exists():
                    oldest.unlink()
                csv_oldest = oldest.with_suffix(".csv")
                if csv_oldest.exists():
                    csv_oldest.unlink()
            file_size = os.path.getsize(checkpoint_path) / 1024 / 1024
            latency = time.time() - start_time
            logger.info(f"Checkpoint sauvegardé pour {self.market}: {checkpoint_path}")
            self.send_alert(
                f"Checkpoint sauvegardé pour {self.market}: {checkpoint_path}",
                priority=1,
            )
            self.log_performance(
                "checkpoint",
                latency,
                success=True,
                file_size_mb=file_size,
                num_rows=len(data),
                data_type=data_type,
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde checkpoint pour {self.market}: {str(e)}"
            logger.error(error_msg)
            self.send_alert(error_msg, priority=3)
            self.log_performance(
                "checkpoint",
                time.time() - start_time,
                success=False,
                error=str(e),
                data_type=data_type,
            )

    def cloud_backup(
        self, data: pd.DataFrame, data_type: str = "alert_manager_state"
    ) -> None:
        """
        Sauvegarde distribuée des données vers AWS S3 toutes les 15 minutes.

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : alert_manager_state).
        """
        start_time = time.time()
        try:
            if not self.config.get("s3_bucket"):
                warning_msg = f"S3 bucket non configuré, sauvegarde cloud ignorée pour {self.market}"
                logger.warning(warning_msg)
                self.send_alert(warning_msg, priority=3)
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.config['s3_prefix']}alert_manager_{data_type}_{self.market}_{timestamp}.csv.gz"
            temp_path = self.checkpoint_dir / f"temp_s3_{timestamp}.csv.gz"
            temp_path.parent.mkdir(exist_ok=True)

            def write_temp():
                data.to_csv(
                    temp_path, compression="gzip", index=False, encoding="utf-8"
                )

            self.with_retries(write_temp)
            s3_client = boto3.client("s3")

            def upload_s3():
                s3_client.upload_file(
                    str(temp_path), self.config["s3_bucket"], backup_path
                )

            self.with_retries(upload_s3)
            temp_path.unlink()
            latency = time.time() - start_time
            logger.info(
                f"Sauvegarde cloud S3 effectuée pour {self.market}: {backup_path}"
            )
            self.send_alert(
                f"Sauvegarde cloud S3 effectuée pour {self.market}: {backup_path}",
                priority=1,
            )
            self.log_performance(
                "cloud_backup",
                latency,
                success=True,
                num_rows=len(data),
                data_type=data_type,
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde cloud S3 pour {self.market}: {str(e)}"
            logger.error(error_msg)
            self.send_alert(error_msg, priority=3)
            self.log_performance(
                "cloud_backup",
                time.time() - start_time,
                success=False,
                error=str(e),
                data_type=data_type,
            )

    def with_retries(
        self,
        func: callable,
        max_attempts: int = MAX_RETRIES,
        delay_base: float = RETRY_DELAY_BASE,
    ) -> Optional[Any]:
        """
        Exécute une fonction avec retries exponentiels.

        Args:
            func (callable): Fonction à exécuter.
            max_attempts (int): Nombre maximum de tentatives.
            delay_base (float): Base pour le délai exponentiel.

        Returns:
            Optional[Any]: Résultat de la fonction ou None si échec.
        """
        start_time = time.time()
        for attempt in range(max_attempts):
            try:
                result = func()
                latency = time.time() - start_time
                self.log_performance(
                    f"retry_attempt_{attempt+1}",
                    latency,
                    success=True,
                    attempt_number=attempt + 1,
                )
                return result
            except Exception as e:
                if attempt == max_attempts - 1:
                    error_msg = f"Échec après {max_attempts} tentatives pour {self.market}: {str(e)}"
                    logger.warning(error_msg)
                    self.send_alert(error_msg, priority=3)
                    self.log_performance(
                        f"retry_attempt_{attempt+1}",
                        time.time() - start_time,
                        success=False,
                        error=str(e),
                        attempt_number=attempt + 1,
                    )
                    return None
                delay = delay_base**attempt
                warning_msg = f"Tentative {attempt+1} échouée pour {self.market}, retry après {delay}s"
                logger.warning(warning_msg)
                self.send_alert(warning_msg, priority=3)
                time.sleep(delay)

    def save_local_alert(self, message: str, priority: int) -> bool:
        """
        Sauvegarde une alerte localement pour priorité 5 (si activé).

        Args:
            message (str): Message de l'alerte.
            priority (int): Niveau de priorité.

        Returns:
            bool: True si sauvegarde réussie, False sinon.
        """
        start_time = time.time()
        try:
            if not self.config.get("local_storage", {}).get("enabled", False):
                logger.debug(f"Stockage local désactivé pour {self.market}")
                return False
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            alert_data = {
                "timestamp": timestamp,
                "message": message,
                "priority": priority,
                "market": self.market,
            }
            alert_path = self.local_alert_dir / f"alert_{timestamp}.json"

            def write_alert():
                with open(alert_path, "w", encoding="utf-8") as f:
                    json.dump(alert_data, f, indent=4)

            self.with_retries(write_alert)
            file_size = os.path.getsize(alert_path) / 1024 / 1024
            if file_size > 1.0:
                alert_msg = f"Alerte locale size {file_size:.2f} MB exceeds 1 MB pour {self.market}"
                logger.warning(alert_msg)
                self.send_alert(alert_msg, priority=3)
            latency = time.time() - start_time
            logger.info(f"Alerte locale sauvegardée pour {self.market}: {alert_path}")
            self.log_performance(
                "save_local_alert", latency, success=True, file_size_mb=file_size
            )
            return True
        except Exception as e:
            error_msg = f"Erreur sauvegarde alerte locale pour {self.market}: {str(e)}"
            logger.error(error_msg)
            self.send_alert(error_msg, priority=3)
            self.log_performance(
                "save_local_alert",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            return False

    def send_alert(self, message: str, priority: int = 1) -> bool:
        """
        Envoie une alerte via les canaux appropriés en fonction de la priorité.

        Args:
            message (str): Message à envoyer.
            priority (int): Niveau de priorité (1=info, 2=warning, 3=error, 4=critical, 5=urgent).

        Returns:
            bool: True si au moins un canal a réussi, False sinon.
        """
        start_time = time.time()
        try:
            cache_key = f"{self.market}_{hash(message)}_{priority}"
            if cache_key in alert_cache:
                return alert_cache[cache_key]
            while len(alert_cache) > MAX_CACHE_SIZE:
                alert_cache.popitem(last=False)

            if not isinstance(priority, int) or priority < 1 or priority > 5:
                warning_msg = f"Priorité invalide pour {self.market}: {priority}, utilisation de 1"
                logger.warning(warning_msg)
                priority = 1

            success = False
            channels = PRIORITY_CHANNELS.get(priority, [])
            for channel_name, notifier in self.notifiers:
                if channel_name in channels:
                    try:
                        if notifier.send_alert(
                            f"MIA_IA_SYSTEM Alert (Priority {priority}) pour {self.market}: {message}",
                            priority,
                        ):
                            success = True
                            if channel_name == "telegram":
                                telegram_alerts_total.labels(
                                    market=self.market, priority=priority
                                ).inc()
                            elif channel_name == "discord":
                                discord_alerts_total.labels(
                                    market=self.market, priority=priority
                                ).inc()
                            elif channel_name == "email":
                                email_alerts_total.labels(
                                    market=self.market, priority=priority
                                ).inc()
                    except Exception as e:
                        error_msg = f"Erreur envoi alerte via {channel_name} pour {self.market}: {str(e)}"
                        logger.error(error_msg)
                        self.log_performance(
                            "send_alert",
                            time.time() - start_time,
                            success=False,
                            error=str(e),
                            channel=channel_name,
                        )

            if "local" in channels:
                success |= self.save_local_alert(message, priority)

            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "message": message,
                "priority": priority,
                "success": success,
                "market": self.market,
                "channels": channels,
            }
            self.save_snapshot(f"alert_{priority}", snapshot)
            self.checkpoint(pd.DataFrame([snapshot]), data_type=f"alert_{priority}")
            self.cloud_backup(pd.DataFrame([snapshot]), data_type=f"alert_{priority}")

            success_msg = f"Alerte envoyée (priority={priority}) pour {self.market}: {message}, success={success}"
            logger.info(success_msg)
            self.log_performance(
                "send_alert",
                time.time() - start_time,
                success=success,
                priority=priority,
                channels=",".join(channels),
            )
            alert_cache[cache_key] = success
            self.save_cache()
            return success
        except Exception as e:
            error_msg = f"Erreur envoi alerte pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.log_performance(
                "send_alert", time.time() - start_time, success=False, error=str(e)
            )
            return False


if __name__ == "__main__":
    alert_manager = AlertManager()
    alert_manager.send_alert("Test alert", priority=1)
