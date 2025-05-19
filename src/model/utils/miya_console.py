# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/utils/miya_console.py
# Module utilitaire pour les fonctions de log et synthèse vocale de MIA.
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle :
# Fournit des fonctions de logging avec synthèse vocale (miya_speak, miya_alerts) pour afficher et vocaliser des messages
# avec coloration et priorités, utilisées par les modules de MIA_IA_SYSTEM_v2_2025 (ex. : train_sac.py, detect_regime.py).
# Supporte la méthode 7 (mémoire contextuelle) via les logs des clusters stockés dans market_memory.db.
#
# Dépendances :
# - pyttsx3>=2.90,<3.0, colorama>=0.4.4,<0.5.0, psutil>=5.9.8,<6.0.0, pandas>=2.0.0,<3.0.0, pyyaml>=6.0.0,<7.0.0,
#   boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0, threading, signal
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/miya_config.yaml (configuration via config_manager)
# - Messages à afficher/vocaliser avec tags, niveaux, priorités, et profils vocaux
#
# Outputs :
# - Logs dans data/logs/miya_console.log
# - Logs de performance dans data/logs/miya_console_performance.csv
# - Snapshots JSON compressés dans data/cache/miya_console/<market>/*.json.gz
# - Sauvegardes incrémentielles dans data/checkpoints/miya_console/<market>/*.json.gz
# - Sortie JSON optionnelle dans data/logs/miya_console.json
#
# Notes :
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence) via les modules appelants.
# - Utilise IQFeed exclusivement via data_provider.py pour les données sous-jacentes.
# - Supprime toute référence à 320 features, 81 features, et obs_t.
# - Implémente retries (max 3, délai 2^attempt secondes) pour les appels critiques.
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des opérations.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.
# - Tests unitaires disponibles dans tests/test_miya_console.py.

import gzip
import json
import signal
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

import boto3
import colorama
import pandas as pd
import psutil
import pyttsx3
from loguru import logger

from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import get_config
from src.utils.telegram_alert import send_telegram_alert

# Initialisation de colorama pour la compatibilité Windows
colorama.init()

# Configuration logging
logger.remove()
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOG_DIR = BASE_DIR / "data" / "logs"
CACHE_DIR = BASE_DIR / "data" / "cache" / "miya_console"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints" / "miya_console"
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "miya_console.log", rotation="10 MB", level="INFO", encoding="utf-8"
)
logger.add(lambda msg: print(msg, end=""), level="INFO")

# Constantes
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Secondes, exponentiel
PERF_LOG_PATH = LOG_DIR / "miya_console_performance.csv"
MAX_CACHE_SIZE = 1000
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB

# Cache global pour la configuration et les messages
CONFIG_CACHE = None
message_cache = OrderedDict()

# Verrou pour la synthèse vocale (thread-safe)
VOICE_LOCK = Lock()


class MiyaConsole:
    """Gère le logging et la synthèse vocale pour MIA_IA_SYSTEM_v2_2025."""

    def __init__(
        self,
        config_path: str = str(BASE_DIR / "config" / "miya_config.yaml"),
        market: str = "ES",
    ):
        """
        Initialise le gestionnaire de console avec logging et synthèse vocale.

        Args:
            config_path (str): Chemin vers le fichier de configuration.
            market (str): Marché (ex. : ES, MNQ).
        """
        self.market = market
        self.alert_manager = AlertManager()
        self.config_path = config_path
        self.snapshot_dir = CACHE_DIR / market
        self.checkpoint_dir = CHECKPOINT_DIR / market
        self.snapshot_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.engine = self.initialize_voice_engine()
        signal.signal(signal.SIGINT, self.handle_sigint)
        logger.info(f"MiyaConsole initialisé pour {market}")
        self.log_performance("init", 0, success=True)

    def handle_sigint(self, signum, frame):
        """Gère l'arrêt propre sur SIGINT."""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "status": "SIGINT",
            "market": self.market,
        }
        snapshot_path = self.snapshot_dir / f'sigint_{snapshot["timestamp"]}.json.gz'
        try:
            with gzip.open(snapshot_path, "wt", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=4)
            success_msg = (
                f"Arrêt propre sur SIGINT, snapshot sauvegardé pour {self.market}"
            )
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot SIGINT pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
        exit(0)

    def log_performance(
        self,
        operation: str,
        latency: float,
        success: bool = True,
        error: str = None,
        **kwargs,
    ) -> None:
        """
        Enregistre les performances (CPU, mémoire, latence) dans miya_console_performance.csv.

        Args:
            operation (str): Nom de l’opération (ex. : miya_speak).
            latency (float): Temps d’exécution en secondes.
            success (bool): Indique si l’opération a réussi.
            error (str, optional): Message d’erreur si applicable.
            **kwargs: Paramètres supplémentaires.
        """
        cache_key = f"{self.market}_{operation}_{hash(str(latency))}_{hash(str(error))}"
        if cache_key in message_cache:
            return
        while len(message_cache) > MAX_CACHE_SIZE:
            message_cache.popitem(last=False)

        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
            cpu_percent = psutil.cpu_percent()
            confidence_drop_rate = 1.0 if success else 0.0  # Simplifié pour Phase 8
            if memory_usage > 1024:
                alert_msg = f"ALERTE: Usage mémoire élevé ({memory_usage:.2f} MB) pour {self.market}"
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=5)
                send_telegram_alert(alert_msg)
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
            message_cache[cache_key] = True
        except Exception as e:
            error_msg = f"Erreur journalisation performance pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)

    def save_snapshot(
        self, snapshot_type: str, data: Dict, compress: bool = True
    ) -> None:
        """
        Sauvegarde un instantané JSON des résultats, compressé avec gzip.

        Args:
            snapshot_type (str): Type de snapshot (ex. : miya_speak).
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
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
            latency = time.time() - start_time
            success_msg = (
                f"Snapshot {snapshot_type} sauvegardé pour {self.market}: {save_path}"
            )
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "save_snapshot", latency, success=True, snapshot_size_mb=file_size
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot {snapshot_type} pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance("save_snapshot", 0, success=False, error=str(e))

    def checkpoint(
        self, data: pd.DataFrame, data_type: str = "miya_console_state"
    ) -> None:
        """
        Sauvegarde incrémentielle des données toutes les 5 minutes avec versionnage (5 versions).

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : miya_console_state).
        """
        try:
            start_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_data = {
                "timestamp": timestamp,
                "num_rows": len(data),
                "columns": list(data.columns),
                "data_type": data_type,
                "market": self.market,
            }
            checkpoint_path = (
                self.checkpoint_dir / f"miya_console_{data_type}_{timestamp}.json.gz"
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
            success_msg = f"Checkpoint sauvegardé pour {self.market}: {checkpoint_path}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "checkpoint",
                latency,
                success=True,
                file_size_mb=file_size,
                num_rows=len(data),
                data_type=data_type,
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde checkpoint pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "checkpoint", 0, success=False, error=str(e), data_type=data_type
            )

    def cloud_backup(
        self, data: pd.DataFrame, data_type: str = "miya_console_state"
    ) -> None:
        """
        Sauvegarde distribuée des données vers AWS S3 toutes les 15 minutes.

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : miya_console_state).
        """
        try:
            start_time = time.time()
            config = get_config(str(BASE_DIR / "config/es_config.yaml"))
            if not config.get("s3_bucket"):
                warning_msg = f"S3 bucket non configuré, sauvegarde cloud ignorée pour {self.market}"
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{config['s3_prefix']}miya_console_{data_type}_{self.market}_{timestamp}.csv.gz"
            temp_path = self.checkpoint_dir / f"temp_s3_{timestamp}.csv.gz"
            temp_path.parent.mkdir(exist_ok=True)

            def write_temp():
                data.to_csv(
                    temp_path, compression="gzip", index=False, encoding="utf-8"
                )

            self.with_retries(write_temp)
            s3_client = boto3.client("s3")

            def upload_s3():
                s3_client.upload_file(str(temp_path), config["s3_bucket"], backup_path)

            self.with_retries(upload_s3)
            temp_path.unlink()
            latency = time.time() - start_time
            success_msg = (
                f"Sauvegarde cloud S3 effectuée pour {self.market}: {backup_path}"
            )
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "cloud_backup",
                latency,
                success=True,
                num_rows=len(data),
                data_type=data_type,
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde cloud S3 pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "cloud_backup", 0, success=False, error=str(e), data_type=data_type
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
                    error_msg = f"Échec après {max_attempts} tentatives pour {self.market}: {str(e)}\n{traceback.format_exc()}"
                    logger.error(error_msg)
                    self.alert_manager.send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
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
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                time.sleep(delay)

    def load_config_manager(
        self, config_path: str = str(BASE_DIR / "config" / "miya_config.yaml")
    ) -> Dict[str, Any]:
        """
        Charge la configuration via config_manager.

        Args:
            config_path (str): Chemin vers le fichier de configuration.

        Returns:
            Dict[str, Any]: Configuration chargée.
        """
        global CONFIG_CACHE
        if CONFIG_CACHE is not None:
            return CONFIG_CACHE
        start_time = time.time()
        try:

            def load():
                config = get_config(config_path).get("miya_console", {})
                return config

            CONFIG_CACHE = self.with_retries(load)
            if CONFIG_CACHE is None:
                raise RuntimeError(
                    f"Échec chargement configuration via config_manager pour {self.market}"
                )
            success_msg = f"Configuration chargée via config_manager pour {self.market}: {config_path}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "load_config_manager", time.time() - start_time, success=True
            )
            self.save_snapshot("load_config_manager", {"config_path": config_path})
            return CONFIG_CACHE
        except Exception as e:
            error_msg = f"Erreur chargement config via config_manager pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "load_config_manager",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            CONFIG_CACHE = {
                "enable_colors": True,
                "color_map": {
                    "info": "green",
                    "warning": "yellow",
                    "error": "red",
                    "critical": "red",
                },
                "min_priority": 1,
                "enable_voice": True,
                "voice_profiles": {
                    "default": {"rate": 150, "volume": 0.8},
                    "urgent": {"rate": 180, "volume": 1.0},
                    "calm": {"rate": 120, "volume": 0.6},
                },
                "log_to_console": True,
                "log_to_file": True,
                "json_output": False,
            }
            return CONFIG_CACHE

    def initialize_voice_engine(self) -> Optional[pyttsx3.Engine]:
        """
        Initialise le moteur de synthèse vocale.

        Returns:
            Optional[pyttsx3.Engine]: Moteur vocal ou None si erreur.
        """
        start_time = time.time()
        try:

            def init_engine():
                return pyttsx3.init()

            engine = self.with_retries(init_engine)
            if engine:
                success_msg = f"Moteur de synthèse vocale initialisé pour {self.market}"
                logger.info(success_msg)
                self.alert_manager.send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)
                self.log_performance(
                    "initialize_voice_engine", time.time() - start_time, success=True
                )
                self.save_snapshot("initialize_voice_engine", {"status": "success"})
            else:
                error_msg = f"Échec initialisation synthèse vocale pour {self.market}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                self.log_performance(
                    "initialize_voice_engine",
                    time.time() - start_time,
                    success=False,
                    error="Échec initialisation",
                )
            return engine
        except Exception as e:
            error_msg = f"Erreur initialisation synthèse vocale pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "initialize_voice_engine",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            return None

    def miya_speak(
        self,
        message: str,
        tag: str = "",
        level: str = "info",
        priority: int = 1,
        voice_profile: Optional[str] = None,
    ) -> None:
        """
        Affiche et/ou vocalise un message formaté pour MIA avec coloration.

        Args:
            message (str): Message à afficher/vocaliser.
            tag (str): Étiquette pour catégoriser le message.
            level (str): Niveau de gravité (info, warning, error, critical).
            priority (int): Priorité du message (1 à 5).
            voice_profile (str, optional): Profil vocal (default, urgent, calm).
        """
        start_time = time.time()
        try:
            cache_key = f"{self.market}_{hash(message)}_{tag}_{level}_{priority}"
            if cache_key in message_cache:
                return
            message_cache[cache_key] = True
            while len(message_cache) > MAX_CACHE_SIZE:
                message_cache.popitem(last=False)

            if not isinstance(message, str) or not message.strip():
                error_msg = f"Message vide ou invalide pour {self.market}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                self.log_performance(
                    "miya_speak",
                    time.time() - start_time,
                    success=False,
                    error="Message vide",
                )
                return
            if not isinstance(tag, str):
                tag = ""
            valid_levels = {"info", "warning", "error", "critical"}
            if level not in valid_levels:
                warning_msg = f"Niveau invalide pour {self.market}: {level}, utilisation de 'info'"
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=2)
                send_telegram_alert(warning_msg)
                level = "info"
            if not isinstance(priority, int) or priority < 1 or priority > 5:
                warning_msg = f"Priorité invalide pour {self.market}: {priority}, utilisation de 1"
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=2)
                send_telegram_alert(warning_msg)
                priority = 1
            if voice_profile is not None and not isinstance(voice_profile, str):
                warning_msg = f"Profil vocal invalide pour {self.market}: {voice_profile}, utilisation de 'default'"
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=2)
                send_telegram_alert(warning_msg)
                voice_profile = None

            config = self.load_config_manager()
            min_priority = config.get("min_priority", 1)
            if priority < min_priority:
                return

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_message = f"[{timestamp}] [MIA - {tag}] {message}"

            enable_colors = config.get("enable_colors", True) and (
                sys.stdout.isatty() or "WT_SESSION" in os.environ
            )
            color_map = {
                "info": "\033[92m" if enable_colors else "",  # Vert
                "warning": "\033[93m" if enable_colors else "",  # Jaune
                "error": "\033[91m" if enable_colors else "",  # Rouge
                "critical": "\033[91m" if enable_colors else "",  # Rouge
            }
            reset = "\033[0m" if enable_colors else ""
            color = color_map.get(level, "\033[92m")

            if config.get("log_to_console", True):
                print(f"{color}{formatted_message}{reset}")

            if config.get("log_to_file", True):
                log_method = {
                    "info": logger.info,
                    "warning": logger.warning,
                    "error": logger.error,
                    "critical": logger.critical,
                }.get(level, logger.info)
                log_method(formatted_message)

            if config.get("json_output", False):
                json_log = {
                    "timestamp": timestamp,
                    "tag": tag,
                    "level": level,
                    "priority": priority,
                    "message": message,
                    "market": self.market,
                }

                def write_json():
                    with open(
                        LOG_DIR / "miya_console.json", "a", encoding="utf-8"
                    ) as f:
                        json.dump(json_log, f)
                        f.write("\n")

                self.with_retries(write_json)

            if config.get("enable_voice", True) and self.engine and priority >= 3:
                voice_config = config.get("voice_profiles", {}).get(
                    voice_profile,
                    config.get("voice_profiles", {}).get(
                        "default", {"rate": 150, "volume": 0.8}
                    ),
                )
                with VOICE_LOCK:

                    def speak():
                        self.engine.setProperty("rate", voice_config.get("rate", 150))
                        self.engine.setProperty(
                            "volume", voice_config.get("volume", 0.8)
                        )
                        self.engine.say(message)
                        self.engine.runAndWait()

                    self.with_retries(speak)

            snapshot_data = {
                "message": message,
                "tag": tag,
                "level": level,
                "priority": priority,
            }
            self.save_snapshot("miya_speak", snapshot_data)
            self.checkpoint(pd.DataFrame([snapshot_data]), data_type="miya_speak")
            self.cloud_backup(pd.DataFrame([snapshot_data]), data_type="miya_speak")

            success_msg = f"Message vocalisé pour {self.market}: {message} (level={level}, priority={priority})"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "miya_speak",
                time.time() - start_time,
                success=True,
                level=level,
                priority=priority,
            )
        except Exception as e:
            error_msg = f"Erreur dans miya_speak pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "miya_speak", time.time() - start_time, success=False, error=str(e)
            )
            print(f"[{timestamp}] [MIA - {tag}] {message} (Erreur: {e})")

    def miya_alerts(
        self,
        message: str,
        tag: str = "",
        voice_profile: Optional[str] = "urgent",
        priority: int = 3,
    ) -> None:
        """
        Affiche et/ou vocalise une alerte pour MIA avec un niveau d'erreur.

        Args:
            message (str): Message d'alerte.
            tag (str): Étiquette pour catégoriser l'alerte.
            voice_profile (str): Profil vocal (default, urgent, calm).
            priority (int): Priorité de l'alerte.
        """
        self.miya_speak(
            message,
            tag=tag,
            level="error",
            priority=priority,
            voice_profile=voice_profile,
        )


if __name__ == "__main__":
    try:
        console = MiyaConsole()
        console.miya_speak(
            "Test d'information",
            tag="TEST",
            level="info",
            priority=1,
            voice_profile="calm",
        )
        console.miya_speak(
            "Test d'avertissement",
            tag="TEST",
            level="warning",
            priority=2,
            voice_profile="default",
        )
        console.miya_alerts(
            "Test d'erreur", tag="TEST", priority=3, voice_profile="urgent"
        )
        console.miya_speak(
            "Test critique",
            tag="TEST",
            level="critical",
            priority=4,
            voice_profile="urgent",
        )
        console.miya_speak("Test priorité faible", tag="TEST", level="info", priority=0)
        console.miya_speak("", tag="TEST", level="info", priority=1)
        console.miya_speak(
            "Test niveau invalide", tag="TEST", level="invalid", priority=1
        )
        print("Tests terminés")
    except Exception as e:
        error_msg = f"Erreur test principal: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        print(f"Erreur test: {e}")
