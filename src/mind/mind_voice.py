# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/mind/mind_voice.py
# Module de synthèse vocale pour MIA avec Google TTS et repli local.
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Gère la synthèse vocale avec Google TTS, repli local via pyttsx3, et mémoire contextuelle
#        (méthode 7, K-means 10 clusters dans market_memory.db). Utilise IQFeed comme source de données système.
#
# Dépendances :
# - gtts>=2.3.0,<3.0.0, simpleaudio>=1.0.0,<2.0.0, playsound>=1.2.0,<2.0.0, pyttsx3>=2.90,<3.0.0 (optionnels),
#   pandas>=2.0.0,<3.0.0, psutil>=5.9.8,<6.0.0, sklearn>=1.2.0,<2.0.0, matplotlib>=3.8.0,<4.0.0,
#   seaborn>=0.13.0,<1.0.0, pyyaml>=6.0.0,<7.0.0, boto3>=1.26.0,<2.0.0, logging, threading, queue, json, signal, gzip
# - src/mind/mind.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/db_setup.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/es_config.yaml
#
# Outputs :
# - data/logs/mind/mind_voice.log
# - data/logs/mind/mind_voice.csv
# - data/logs/mind_voice_performance.csv
# - data/mind/mind_voice_snapshots/*.json.gz
# - data/checkpoints/mind_voice_*.json.gz
# - data/audio/
# - data/figures/mind_voice/
# - market_memory.db (table clusters)
#
# Notes :
# - Utilise IQFeed exclusivement via data_provider.py pour le contexte système.
# - Supprime toute référence à 320 features, 81 features, et obs_t.
# - Implémente retries (max 3, délai 2^attempt secondes) pour les opérations critiques.
# - Logs psutil dans data/logs/mind_voice_performance.csv avec métriques détaillées.
# - Alertes via alert_manager.py et Telegram pour priorité ≥ 4.
# - Snapshots compressés avec gzip dans data/mind/mind_voice_snapshots/.
# - Sauvegardes incrémentielles toutes les 5 min dans data/checkpoints/ (5 versions).
# - Sauvegardes distribuées toutes les 15 min vers S3 (configurable).
# - Tests unitaires disponibles dans tests/test_mind_voice.py.
# - Phases intégrées : Phase 7 (mémoire contextuelle), Phase 8 (auto-conscience via confidence_drop_rate).
# - Validation complète prévue pour juin 2025.

import gzip
import json
import logging
import os
import queue
import threading
import time
from datetime import datetime
from typing import Dict, Optional

from gtts import gTTS

try:
    from playsound import playsound
except ImportError:
    playsound = None
try:
    import simpleaudio as sa
except ImportError:
    sa = None
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None
import signal
from pathlib import Path

import boto3
import matplotlib.pyplot as plt
import pandas as pd
import psutil
import seaborn as sns
from sklearn.cluster import KMeans

from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import get_config
from src.model.utils.db_setup import get_db_connection
from src.utils.telegram_alert import send_telegram_alert

# Configuration du logging
base_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
log_dir = base_dir / "data" / "logs" / "mind"
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_dir / "mind_voice.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Chemins et configuration
TEMP_DIR = base_dir / "data" / "audio"
CONFIG_PATH = base_dir / "config" / "es_config.yaml"
CSV_LOG_PATH = log_dir / "mind_voice.csv"
PERFORMANCE_LOG_PATH = log_dir / "mind_voice_performance.csv"
DASHBOARD_PATH = base_dir / "data" / "mind" / "mind_voice_dashboard.json"
SNAPSHOT_DIR = base_dir / "data" / "mind" / "mind_voice_snapshots"
CHECKPOINT_DIR = base_dir / "data" / "checkpoints"
FIGURES_DIR = base_dir / "data" / "figures" / "mind_voice"
TEMP_DIR.mkdir(exist_ok=True)
SNAPSHOT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Profils vocaux configurables
VOICE_PROFILES = {
    "calm": {"lang": "fr", "slow": True, "volume": 0.8, "priority": 1},
    "urgent": {"lang": "fr", "slow": False, "volume": 1.0, "priority": 3},
    "educative": {"lang": "fr", "slow": False, "volume": 0.9, "priority": 2},
    "dashboard": {"lang": "fr", "slow": True, "volume": 0.85, "priority": 1},
}

# Limite du cache audio
MAX_AUDIO_CACHE = 500

# Constantes
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Secondes, exponentiel

# Variable pour gérer l'arrêt propre
RUNNING = True


class VoiceManager:
    """
    Classe pour gérer la synthèse vocale de MIA avec Google TTS et repli local.
    """

    def __init__(self, config_path: str = str(CONFIG_PATH)):
        """
        Initialise le gestionnaire de synthèse vocale.

        Args:
            config_path (str): Chemin vers la configuration vocale.
        """
        self.alert_manager = AlertManager()
        self.checkpoint_versions = []
        signal.signal(signal.SIGINT, self.handle_sigint)

        start_time = datetime.now()
        try:
            self.config = self.load_voice_config(os.path.basename(config_path))
            self.audio_cache = {}  # Cache pour réutiliser les fichiers audio
            self.log_buffer = []  # Buffer pour les écritures CSV
            self.speak_queue = queue.PriorityQueue()  # File pour les messages vocaux
            self.cleanup_interval = self.config.get(
                "cleanup_interval", 3600
            )  # 1 heure par défaut
            self.stop_event = threading.Event()  # Événement pour arrêter le nettoyage

            # Détecter les capacités du système
            self.engines = self.detect_system_capabilities()
            if not any(self.engines.values()):
                error_msg = "Aucun moteur vocal ou lecteur audio disponible"
                self.alert_manager.send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Lancer le thread de traitement des messages vocaux
            threading.Thread(target=self.speak_worker, daemon=True).start()

            # Lancer le thread de nettoyage périodique
            threading.Thread(target=self.cleanup_worker, daemon=True).start()

            success_msg = "Gestionnaire vocal initialisé"
            self.alert_manager.send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("init_voice_manager", latency, success=True)
            self.save_snapshot(
                "init",
                {"config_path": config_path, "timestamp": datetime.now().isoformat()},
            )
        except Exception as e:
            error_msg = f"Erreur initialisation VoiceManager : {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init_voice_manager", 0, success=False, error=str(e))
            raise

    def handle_sigint(self, signum, frame):
        """Gère l'arrêt propre sur SIGINT."""
        global RUNNING
        datetime.now()
        snapshot = {"timestamp": datetime.now().isoformat(), "status": "SIGINT"}
        SNAPSHOT_DIR / f'voice_sigint_{snapshot["timestamp"]}.json.gz'
        try:
            RUNNING = False
            self.stop_event.set()
            self.save_snapshot("sigint", snapshot)
            success_msg = "Arrêt propre sur SIGINT, snapshot sauvegardé"
            self.alert_manager.send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("handle_sigint", 0, success=True)
            exit(0)
        except Exception as e:
            error_msg = f"Erreur sauvegarde SIGINT: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("handle_sigint", 0, success=False, error=str(e))
            exit(1)

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ):
        """
        Enregistre les performances avec psutil dans data/logs/mind_voice_performance.csv.

        Args:
            operation (str): Nom de l’opération.
            latency (float): Temps d’exécution en secondes.
            success (bool): Succès de l’opération.
            error (str, optional): Message d’erreur.
            **kwargs: Paramètres supplémentaires (ex. : num_messages, snapshot_size_mb).
        """
        start_time = datetime.now()
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
            cpu_usage = psutil.cpu_percent()
            if memory_usage > 1024:
                alert_msg = f"ALERTE: Usage mémoire élevé ({memory_usage:.2f} MB)"
                self.alert_manager.send_alert(alert_msg, priority=5)
                send_telegram_alert(alert_msg)
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "success": success,
                "latency": latency,
                "error": error,
                "cpu_percent": cpu_usage,
                "memory_mb": memory_usage,
                **kwargs,
            }
            self.log_buffer.append(log_entry)
            if len(self.log_buffer) >= self.config.get("logging", {}).get(
                "buffer_size", 100
            ):
                log_df = pd.DataFrame(self.log_buffer)
                mode = "a" if PERFORMANCE_LOG_PATH.exists() else "w"

                def save_log():
                    log_df.to_csv(
                        PERFORMANCE_LOG_PATH,
                        mode=mode,
                        index=False,
                        header=not PERFORMANCE_LOG_PATH.exists(),
                        encoding="utf-8",
                    )

                self.with_retries(save_log)
                self.checkpoint(log_df)
                self.cloud_backup(log_df)
                self.log_buffer = []
            latency = (datetime.now() - start_time).total_seconds()
            logger.info(f"Performance journalisée pour {operation}. CPU: {cpu_usage}%")
            self.log_performance("log_performance", latency, success=True)
        except Exception as e:
            error_msg = (
                f"Erreur journalisation performance: {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("log_performance", 0, success=False, error=str(e))

    def save_snapshot(
        self, snapshot_type: str, data: Dict, compress: bool = True
    ) -> None:
        """
        Sauvegarde un instantané des résultats.

        Args:
            snapshot_type (str): Type de snapshot (ex. : init, sigint).
            data (Dict): Données à sauvegarder.
            compress (bool): Compresser avec gzip (défaut : True).
        """
        try:
            start_time = datetime.now()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = {
                "timestamp": timestamp,
                "type": snapshot_type,
                "data": data,
                "buffer_size": len(self.log_buffer),
            }
            snapshot_path = SNAPSHOT_DIR / f"snapshot_{snapshot_type}_{timestamp}.json"
            SNAPSHOT_DIR.mkdir(exist_ok=True)

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
                alert_msg = f"Snapshot size {file_size:.2f} MB exceeds 1 MB"
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Snapshot {snapshot_type} sauvegardé: {save_path}"
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "save_snapshot", latency, success=True, snapshot_size_mb=file_size
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("save_snapshot", 0, success=False, error=str(e))

    def checkpoint(self, data: pd.DataFrame) -> None:
        """
        Sauvegarde incrémentielle des logs vocaux toutes les 5 minutes avec versionnage (5 versions).

        Args:
            data (pd.DataFrame): Données des logs vocaux à sauvegarder.
        """
        try:
            start_time = datetime.now()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_data = {
                "timestamp": timestamp,
                "num_rows": len(data),
                "columns": list(data.columns),
            }
            checkpoint_path = CHECKPOINT_DIR / f"mind_voice_{timestamp}.json.gz"
            CHECKPOINT_DIR.mkdir(exist_ok=True)

            def write_checkpoint():
                with gzip.open(checkpoint_path, "wt", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, indent=4)
                data.to_csv(
                    checkpoint_path.replace(".json.gz", ".csv"),
                    index=False,
                    encoding="utf-8",
                )

            self.with_retries(write_checkpoint)
            self.checkpoint_versions.append(checkpoint_path)
            if len(self.checkpoint_versions) > 5:
                oldest = self.checkpoint_versions.pop(0)
                if os.path.exists(oldest):
                    os.remove(oldest)
                csv_oldest = oldest.replace(".json.gz", ".csv")
                if os.path.exists(csv_oldest):
                    os.remove(csv_oldest)
            file_size = os.path.getsize(checkpoint_path) / 1024 / 1024
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Checkpoint sauvegardé: {checkpoint_path}"
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "checkpoint",
                latency,
                success=True,
                file_size_mb=file_size,
                num_rows=len(data),
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde checkpoint: {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("checkpoint", 0, success=False, error=str(e))

    def cloud_backup(self, data: pd.DataFrame) -> None:
        """
        Sauvegarde distribuée des logs vocaux vers AWS S3 toutes les 15 minutes (configurable).

        Args:
            data (pd.DataFrame): Données des logs vocaux à sauvegarder.
        """
        try:
            start_time = datetime.now()
            if not self.config.get("s3_bucket"):
                warning_msg = "S3 bucket non configuré, sauvegarde cloud ignorée"
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.config['s3_prefix']}mind_voice_{timestamp}.csv.gz"
            temp_path = CHECKPOINT_DIR / f"temp_s3_{timestamp}.csv.gz"

            def write_temp():
                data.to_csv(
                    temp_path, compression="gzip", index=False, encoding="utf-8"
                )

            self.with_retries(write_temp)
            s3_client = boto3.client("s3")

            def upload_s3():
                s3_client.upload_file(temp_path, self.config["s3_bucket"], backup_path)

            self.with_retries(upload_s3)
            os.remove(temp_path)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Sauvegarde cloud S3 effectuée: {backup_path}"
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "cloud_backup", latency, success=True, num_rows=len(data)
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde cloud S3: {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("cloud_backup", 0, success=False, error=str(e))

    def with_retries(
        self,
        func: callable,
        max_attempts: int = MAX_RETRIES,
        delay_base: float = RETRY_DELAY_BASE,
    ) -> Optional[any]:
        """
        Exécute une fonction avec retries (max 3, délai exponentiel).

        Args:
            func (callable): Fonction à exécuter.
            max_attempts (int): Nombre maximum de tentatives.
            delay_base (float): Base pour le délai exponentiel.

        Returns:
            Optional[any]: Résultat de la fonction ou None si échec.
        """
        start_time = datetime.now()
        for attempt in range(max_attempts):
            try:
                result = func()
                latency = (datetime.now() - start_time).total_seconds()
                self.log_performance(
                    f"retry_attempt_{attempt+1}",
                    latency,
                    success=True,
                    attempt_number=attempt + 1,
                )
                return result
            except Exception as e:
                if attempt == max_attempts - 1:
                    error_msg = f"Échec après {max_attempts} tentatives: {str(e)}\n{traceback.format_exc()}"
                    self.alert_manager.send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    logger.error(error_msg)
                    self.log_performance(
                        f"retry_attempt_{attempt+1}",
                        0,
                        success=False,
                        error=str(e),
                        attempt_number=attempt + 1,
                    )
                    return None
                delay = delay_base**attempt
                warning_msg = f"Tentative {attempt+1} échouée, retry après {delay}s"
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
                time.sleep(delay)

    def load_voice_config(self, config_path: str) -> Dict:
        """
        Charge la configuration vocale depuis es_config.yaml.

        Args:
            config_path (str): Nom du fichier de configuration.

        Returns:
            Dict: Configuration vocale.
        """
        start_time = datetime.now()
        try:
            config = get_config(base_dir / config_path)
            voice_config = config.get(
                "voice",
                {
                    "enabled": True,
                    "async": True,
                    "cleanup_interval": 3600,
                    "s3_bucket": None,
                    "s3_prefix": "mind_voice/",
                },
            )
            if (
                not isinstance(voice_config.get("cleanup_interval", 3600), (int, float))
                or voice_config["cleanup_interval"] <= 0
            ):
                raise ValueError("Intervalle de nettoyage invalide")
            required_keys = [
                "enabled",
                "async",
                "cleanup_interval",
                "s3_bucket",
                "s3_prefix",
            ]
            missing_keys = [key for key in required_keys if key not in voice_config]
            if missing_keys:
                raise ValueError(f"Clés de configuration manquantes: {missing_keys}")
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Configuration vocale {config_path} chargée"
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("load_voice_config", latency, success=True)
            return voice_config
        except Exception as e:
            error_msg = (
                f"Erreur chargement config vocale : {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("load_voice_config", 0, success=False, error=str(e))
            return {
                "enabled": True,
                "async": True,
                "cleanup_interval": 3600,
                "s3_bucket": None,
                "s3_prefix": "mind_voice/",
            }

    def detect_system_capabilities(self) -> Dict[str, bool]:
        """
        Détecte les moteurs vocaux et lecteurs audio disponibles sur le système.

        Returns:
            Dict[str, bool]: Disponibilité des moteurs et lecteurs (gtts, pyttsx3, simpleaudio, playsound).
        """
        start_time = datetime.now()
        try:
            engines = {
                "gtts": False,
                "pyttsx3": False,
                "simpleaudio": False,
                "playsound": False,
            }

            def test_gtts():
                gTTS("test", lang="fr")
                engines["gtts"] = True

            self.with_retries(test_gtts)

            if pyttsx3:

                def test_pyttsx3():
                    engine = pyttsx3.init()
                    engine.stop()
                    engines["pyttsx3"] = True

                self.with_retries(test_pyttsx3)

            if sa:
                engines["simpleaudio"] = True
            if playsound:
                engines["playsound"] = True

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Capacités détectées : {engines}"
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("detect_system_capabilities", latency, success=True)
            return engines
        except Exception as e:
            error_msg = (
                f"Erreur détection capacités : {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "detect_system_capabilities", 0, success=False, error=str(e)
            )
            return {
                "gtts": False,
                "pyttsx3": False,
                "simpleaudio": False,
                "playsound": False,
            }

    def validate_audio_file(self, filename: str) -> bool:
        """
        Vérifie l’intégrité d’un fichier audio.

        Args:
            filename (str): Chemin du fichier audio.

        Returns:
            bool: True si le fichier est valide, False sinon.
        """
        start_time = datetime.now()
        try:
            filename = Path(filename)
            if not filename.exists():
                return False
            if filename.stat().st_size < 1000:  # Fichier trop petit (< 1 Ko)
                return False
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("validate_audio_file", latency, success=True)
            return True
        except Exception as e:
            error_msg = (
                f"Erreur validation fichier audio : {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("validate_audio_file", 0, success=False, error=str(e))
            return False

    def safe_play(self, filename: str) -> bool:
        """
        Joue un fichier audio avec simpleaudio ou repli sur playsound.

        Args:
            filename (str): Chemin du fichier audio.

        Returns:
            bool: Succès de l’opération.
        """
        start_time = datetime.now()
        try:
            filename = Path(filename)
            if not self.validate_audio_file(filename):
                raise ValueError(f"Fichier audio invalide : {filename}")

            def play_with_simpleaudio():
                wave_obj = sa.WaveObject.from_wave_file(str(filename))
                play_obj = wave_obj.play()
                play_obj.wait_done()
                return True

            def play_with_playsound():
                playsound(str(filename))
                return True

            if self.engines.get("simpleaudio", False):
                self.with_retries(play_with_simpleaudio)
                latency = (datetime.now() - start_time).total_seconds()
                self.log_performance("safe_play", latency, success=True)
                return True

            if self.engines.get("playsound", False):
                self.with_retries(play_with_playsound)
                latency = (datetime.now() - start_time).total_seconds()
                self.log_performance("safe_play", latency, success=True)
                return True

            raise ValueError("Aucun lecteur audio disponible")
        except Exception as e:
            error_msg = f"Erreur lecture audio : {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("safe_play", 0, success=False, error=str(e))
            return False

    def speak_gtts(self, text: str, profile: str, filename: str) -> bool:
        """
        Tente de générer et jouer un message avec Google TTS.

        Args:
            text (str): Texte à vocaliser.
            profile (str): Profil vocal.
            filename (str): Chemin du fichier audio temporaire.

        Returns:
            bool: Succès de l’opération.
        """
        start_time = datetime.now()
        try:
            profile_cfg = VOICE_PROFILES.get(profile, VOICE_PROFILES["calm"])

            def generate():
                tts = gTTS(
                    text=text, lang=profile_cfg["lang"], slow=profile_cfg["slow"]
                )
                tts.save(str(filename))
                return self.safe_play(filename)

            result = self.with_retries(generate)
            if result is None:
                raise ValueError("Échec génération gTTS")
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("speak_gtts", latency, success=True)
            return result
        except Exception as e:
            error_msg = f"Erreur gTTS : {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("speak_gtts", 0, success=False, error=str(e))
            return False

    def speak_pyttsx3(self, text: str, profile: str, volume: float) -> bool:
        """
        Tente de générer et jouer un message avec pyttsx3.

        Args:
            text (str): Texte à vocaliser.
            profile (str): Profil vocal.
            volume (float): Volume du son.

        Returns:
            bool: Succès de l’opération.
        """
        start_time = datetime.now()
        if not pyttsx3:
            error_msg = "pyttsx3 non disponible"
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("speak_pyttsx3", 0, success=False, error=error_msg)
            return False

        try:
            profile_cfg = VOICE_PROFILES.get(profile, VOICE_PROFILES["calm"])

            def generate():
                engine = pyttsx3.init()
                engine.setProperty("volume", volume)
                engine.setProperty("rate", 120 if profile_cfg["slow"] else 150)
                engine.say(text)
                engine.runAndWait()

            self.with_retries(generate)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Message vocal généré avec pyttsx3: {text[:50]}..."
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("speak_pyttsx3", latency, success=True)
            return True
        except Exception as e:
            error_msg = f"Erreur pyttsx3 : {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("speak_pyttsx3", 0, success=False, error=str(e))
            return False

    def adjust_priority(self, text: str, profile: str) -> int:
        """
        Ajuste la priorité du message vocal en fonction de son contenu.

        Args:
            text (str): Texte à vocaliser.
            profile (str): Profil vocal.

        Returns:
            int: Priorité ajustée.
        """
        base_priority = VOICE_PROFILES.get(profile, VOICE_PROFILES["calm"])["priority"]
        if any(
            keyword in text.lower()
            for keyword in ["erreur", "anomalie", "urgent", "alerte"]
        ):
            return base_priority + 1
        return base_priority

    def store_voice_pattern(self, log_entry: Dict) -> Dict:
        """
        Stocke un pattern vocal dans market_memory.db avec clusterisation K-means (méthode 7).

        Args:
            log_entry (Dict): Entrée de journalisation vocale.

        Returns:
            Dict: Entrée avec cluster_id.
        """
        start_time = datetime.now()
        try:
            features = {
                "priority": VOICE_PROFILES.get(
                    log_entry["profile"], VOICE_PROFILES["calm"]
                )["priority"],
                "text_length": len(log_entry["text"]),
                "latency": log_entry["latency"],
            }
            required_cols = ["priority", "text_length", "latency"]
            missing_cols = [col for col in required_cols if col not in features]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(features)} colonnes)"
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)

            df = pd.DataFrame([features])
            X = df[["priority", "text_length", "latency"]].fillna(0).values

            if len(X) < 10:
                log_entry["cluster_id"] = 0
            else:

                def run_kmeans():
                    kmeans = KMeans(n_clusters=10, random_state=42)
                    return kmeans.fit_predict(X)[0]

                log_entry["cluster_id"] = self.with_retries(run_kmeans)
                if log_entry["cluster_id"] is None:
                    raise ValueError("Échec clusterisation K-means")

            def store_clusters():
                conn = get_db_connection(str(base_dir / "data" / "market_memory.db"))
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS clusters (
                        cluster_id INTEGER,
                        event_type TEXT,
                        features TEXT,
                        timestamp TEXT
                    )
                """
                )
                features_json = json.dumps(features)
                cursor.execute(
                    """
                    INSERT INTO clusters (cluster_id, event_type, features, timestamp)
                    VALUES (?, ?, ?, ?)
                """,
                    (
                        log_entry["cluster_id"],
                        "voice",
                        features_json,
                        log_entry["timestamp"],
                    ),
                )
                conn.commit()
                conn.close()

            self.with_retries(store_clusters)

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Pattern vocal stocké, cluster_id={log_entry['cluster_id']}"
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "store_voice_pattern",
                latency,
                success=True,
                confidence_drop_rate=confidence_drop_rate,
            )
            self.save_snapshot(
                "store_voice_pattern",
                {"cluster_id": log_entry["cluster_id"], "features": features},
            )
            return log_entry
        except Exception as e:
            error_msg = (
                f"Erreur stockage pattern vocal : {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("store_voice_pattern", 0, success=False, error=str(e))
            log_entry["cluster_id"] = 0
            return log_entry

    def visualize_voice_patterns(self):
        """
        Génère une heatmap des clusters de messages vocaux dans data/figures/mind_voice/.
        """
        start_time = datetime.now()
        try:
            logs = self.log_buffer[-100:]
            if not logs:
                error_msg = "Aucun log vocal pour visualisation"
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                logger.warning(error_msg)
                return

            df = pd.DataFrame(logs)
            if "cluster_id" not in df.columns or df["cluster_id"].isnull().all():
                error_msg = "Aucun cluster_id pour visualisation"
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                logger.warning(error_msg)
                return

            pivot = df.pivot_table(
                index="cluster_id",
                columns="priority",
                values="timestamp",
                aggfunc="count",
                fill_value=0,
            )
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot, annot=True, cmap="Blues")
            plt.title("Heatmap des Clusters de Messages Vocaux")

            def save_fig():
                plt.savefig(
                    str(
                        FIGURES_DIR
                        / f"voice_clusters_{datetime.now().strftime('%Y%m%d')}.png"
                    )
                )
                plt.close()

            self.with_retries(save_fig)

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Heatmap des clusters vocaux générée"
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("visualize_voice_patterns", latency, success=True)
        except Exception as e:
            error_msg = (
                f"Erreur visualisation clusters : {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "visualize_voice_patterns", 0, success=False, error=str(e)
            )

    def speak(
        self,
        text: str,
        profile: str = "calm",
        async_mode: Optional[bool] = None,
        volume: Optional[float] = None,
    ) -> None:
        """
        Génère et joue un message vocal avec Google TTS ou repli local.

        Args:
            text (str): Texte à vocaliser.
            profile (str): Profil vocal (calm, urgent, educative, dashboard).
            async_mode (bool, optional): Mode asynchrone pour éviter le blocage.
            volume (float, optional): Volume personnalisé (0.0 à 1.0).
        """
        start_time = datetime.now()
        try:
            if not self.config["enabled"]:
                warning_msg = "Synthèse vocale désactivée"
                self.alert_manager.send_alert(warning_msg, priority=1)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
                return

            if "dxfeed" in text.lower():
                error_msg = "Message contenant référence dxFeed ignoré"
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return

            profile_cfg = VOICE_PROFILES.get(profile, VOICE_PROFILES["calm"])
            async_mode = async_mode if async_mode is not None else self.config["async"]
            volume = volume if volume is not None else profile_cfg["volume"]

            cache_key = f"{text}_{profile}"
            if cache_key in self.audio_cache and self.validate_audio_file(
                self.audio_cache[cache_key]
            ):
                priority = self.adjust_priority(text, profile)
                if async_mode:
                    self.speak_queue.put(
                        (priority, (self.audio_cache[cache_key], text, profile, volume))
                    )
                else:
                    self.safe_play(self.audio_cache[cache_key])
                success_msg = f"Message vocal utilisé depuis le cache : {text[:50]}..."
                self.alert_manager.send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)
                logger.info(success_msg)
                latency = (datetime.now() - start_time).total_seconds()
                self.log_performance(
                    "speak", latency, success=True, num_messages=len(self.audio_cache)
                )
                return

            priority = self.adjust_priority(text, profile)
            if priority >= 4:
                send_telegram_alert(f"[MIA - Vocal {profile}] {text}")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = TEMP_DIR / f"mia_{profile}_{timestamp}.mp3"
            self.speak_queue.put((priority, (text, profile, volume, filename)))

            if not async_mode:
                while not self.speak_queue.empty():
                    time.sleep(0.1)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Message vocal en file d'attente : {text[:50]}..."
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "speak", latency, success=True, num_messages=len(self.audio_cache)
            )
        except Exception as e:
            error_msg = f"Erreur vocalisation : {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("speak", 0, success=False, error=str(e))

    def speak_worker(self):
        """
        Thread pour traiter les messages vocaux de manière asynchrone.
        """
        while RUNNING:
            try:
                priority, item = self.speak_queue.get(timeout=10)
                start_time = datetime.now()

                if (
                    isinstance(item, tuple)
                    and len(item) == 4
                    and isinstance(item[0], str)
                ):  # Nouveau message
                    text, profile, volume, filename = item
                    ENGINE_MAP = [
                        ("gtts", lambda: self.speak_gtts(text, profile, filename)),
                        ("pyttsx3", lambda: self.speak_pyttsx3(text, profile, volume)),
                    ]

                    success = False
                    engine_name = "none"
                    for name, engine_func in ENGINE_MAP:
                        if self.engines.get(name, False):
                            success = engine_func()
                            if success:
                                engine_name = name
                                if name == "gtts":
                                    self.audio_cache[f"{text}_{profile}"] = filename
                                    if len(self.audio_cache) > MAX_AUDIO_CACHE:
                                        oldest_key = list(self.audio_cache.keys())[0]
                                        del self.audio_cache[oldest_key]
                                break

                    if not success:
                        error_msg = "Échec de tous les moteurs vocaux"
                        self.alert_manager.send_alert(error_msg, priority=5)
                        send_telegram_alert(error_msg)
                        logger.error(error_msg)

                elif (
                    isinstance(item, tuple) and len(item) == 4
                ):  # Message depuis le cache
                    filename, text, profile, volume = item
                    success = self.safe_play(filename)
                    engine_name = "cache"

                latency = (datetime.now() - start_time).total_seconds()
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "text": text,
                    "profile": profile,
                    "engine": engine_name,
                    "latency": latency,
                    "success": success,
                    "cluster_id": None,
                }
                log_entry = self.store_voice_pattern(log_entry)
                self.log_buffer.append(log_entry)
                if len(self.log_buffer) > 10000:
                    self.log_buffer = self.log_buffer[-5000:]

                def log_to_csv():
                    if len(self.log_buffer) >= self.config.get("logging", {}).get(
                        "buffer_size", 100
                    ):
                        log_df = pd.DataFrame(self.log_buffer)
                        mode = "a" if CSV_LOG_PATH.exists() else "w"
                        log_df.to_csv(
                            CSV_LOG_PATH,
                            mode=mode,
                            index=False,
                            header=not CSV_LOG_PATH.exists(),
                            encoding="utf-8",
                        )
                        self.checkpoint(log_df)
                        self.cloud_backup(log_df)
                        self.log_buffer = []

                self.with_retries(log_to_csv)

                self.save_voice_snapshot(
                    len(self.audio_cache), text, profile, latency, success
                )
                self.update_dashboard()
                threading.Thread(
                    target=self.visualize_voice_patterns, daemon=True
                ).start()

                self.speak_queue.task_done()
                success_msg = (
                    f"Message vocal traité : {text[:50]}... (engine={engine_name})"
                )
                self.alert_manager.send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)
                logger.info(success_msg)
                self.log_performance(
                    "speak_worker",
                    latency,
                    success=True,
                    num_messages=len(self.audio_cache),
                )
            except queue.Empty:
                continue
            except Exception as e:
                error_msg = f"Erreur traitement message vocal : {str(e)}\n{traceback.format_exc()}"
                self.alert_manager.send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                self.log_performance("speak_worker", 0, success=False, error=str(e))

    def cleanup_audio_file(self, filename: str) -> None:
        """
        Supprime un fichier audio temporaire.

        Args:
            filename (str): Chemin du fichier à supprimer.
        """
        start_time = datetime.now()
        try:
            filename = Path(filename)

            def cleanup():
                if filename.exists():
                    os.remove(filename)

            self.with_retries(cleanup)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Fichier audio supprimé : {filename}"
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("cleanup_audio_file", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur suppression audio : {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("cleanup_audio_file", 0, success=False, error=str(e))

    def cleanup_audio_files(self, max_age_hours: int = 24) -> None:
        """
        Supprime les fichiers audio plus vieux que max_age_hours.

        Args:
            max_age_hours (int): Âge maximum des fichiers en heures.
        """
        start_time = datetime.now()
        try:
            now = time.time()
            for file in TEMP_DIR.iterdir():
                if (
                    file.is_file()
                    and (now - file.stat().st_mtime) > max_age_hours * 3600
                ):

                    def cleanup():
                        os.remove(file)

                    self.with_retries(cleanup)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Nettoyage des fichiers audio effectué"
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("cleanup_audio_files", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur nettoyage audio : {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("cleanup_audio_files", 0, success=False, error=str(e))

    def cleanup_worker(self):
        """
        Thread pour nettoyer périodiquement les fichiers audio temporaires.
        """
        start_time = datetime.now()
        while not self.stop_event.wait(timeout=self.cleanup_interval):
            try:
                self.cleanup_audio_files(max_age_hours=24)
                latency = (datetime.now() - start_time).total_seconds()
                self.log_performance("cleanup_worker", latency, success=True)
            except Exception as e:
                error_msg = (
                    f"Erreur nettoyage périodique : {str(e)}\n{traceback.format_exc()}"
                )
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                self.log_performance("cleanup_worker", 0, success=False, error=str(e))

    def save_voice_snapshot(
        self, step: int, text: str, profile: str, latency: float, success: bool
    ) -> None:
        """
        Sauvegarde un instantané d’une synthèse vocale.

        Args:
            step (int): Étape de la synthèse.
            text (str): Texte vocalisé.
            profile (str): Profil vocal.
            latency (float): Latence de la synthèse.
            success (bool): Succès de l’opération.
        """
        start_time = datetime.now()
        try:
            snapshot = {
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "text": text,
                "profile": profile,
                "latency": latency,
                "success": success,
                "audio_cache_size": len(self.audio_cache),
                "performance": {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_mb": psutil.Process().memory_info().rss / 1024 / 1024,
                },
            }
            self.save_snapshot(f"voice_step_{step:04d}", snapshot, compress=True)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Snapshot vocal step {step} sauvegardé"
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("save_voice_snapshot", latency, success=True)
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde snapshot vocal : {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("save_voice_snapshot", 0, success=False, error=str(e))

    def update_dashboard(self) -> None:
        """
        Met à jour un fichier JSON pour partager l'état vocal avec mia_dashboard.py.
        """
        start_time = datetime.now()
        try:
            success_count = sum(1 for log in self.log_buffer if log["success"])
            total_count = len(self.log_buffer)
            status = {
                "timestamp": datetime.now().isoformat(),
                "num_messages": len(self.audio_cache),
                "last_message": (
                    self.log_buffer[-1]["text"] if self.log_buffer else None
                ),
                "last_latency": (
                    self.log_buffer[-1]["latency"] if self.log_buffer else 0
                ),
                "success_rate": success_count / total_count if total_count > 0 else 0,
                "engines": self.engines,
                "average_latency": (
                    sum(log["latency"] for log in self.log_buffer) / total_count
                    if total_count > 0
                    else 0
                ),
                "cache_size": len(self.audio_cache),
                "performance": {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_mb": psutil.Process().memory_info().rss / 1024 / 1024,
                },
            }

            def save_dashboard():
                with open(DASHBOARD_PATH, "w", encoding="utf-8") as f:
                    json.dump(status, f, indent=4)

            self.with_retries(save_dashboard)
            self.save_snapshot("dashboard", status)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Mise à jour dashboard vocal effectuée"
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "update_dashboard",
                latency,
                success=True,
                num_messages=len(self.audio_cache),
            )
        except Exception as e:
            error_msg = f"Erreur mise à jour dashboard vocal : {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("update_dashboard", 0, success=False, error=str(e))


if __name__ == "__main__":
    voice_manager = VoiceManager()
    voice_manager.speak("Test de synthèse vocale calme", profile="calm")
    voice_manager.speak("Alerte urgente !", profile="urgent", async_mode=False)
    voice_manager.speak("Explication éducative", profile="educative")
    voice_manager.cleanup_audio_files()
