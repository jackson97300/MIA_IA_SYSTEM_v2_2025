# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/features/extractors/encode_time_context.py
# Encode les features temporelles (heures, jours, session, expiration) pour contextualiser.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Calcule 8 features temporelles (ex. : time_of_day_sin, day_of_week, cluster_id, option_expiry_proximity)
#        avec mémoire contextuelle (méthode 7), cache intermédiaire, logs psutil, validation SHAP (méthode 17),
#        et utilise IQFeed comme source de données exclusive (remplace dxFeed).
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, scikit-learn>=1.5.0,<2.0.0, psutil>=5.9.0,<6.0.0, sqlite3
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
#
# Inputs :
# - config/es_config.yaml
# - data/features/feature_importance.csv
# - data/iqfeed/merged_data.csv
# - data/iqfeed/options_data.csv
# - data/iqfeed/microstructure_data.csv
# - data/iqfeed/hft_data.csv
#
# Outputs :
# - data/features/cache/time_context/latest.csv
# - data/logs/time_context_performance.csv
# - data/time_context_snapshots/
# - market_memory.db (table clusters)
#
# Notes :
# - Utilise exclusivement IQFeed pour les données (remplace dxFeed).
# - Suppression de toute référence à dxFeed, obs_t, 320 features, et 81 features pour conformité avec MIA_IA_SYSTEM_v2_2025.
# - Intègre les Phases 1-18 :
#   - Phase 1 (news_scraper.py, news_analyzer.py) : Évolution future pour l'analyse de sentiment.
#   - Phase 13 (orderflow_indicators.py, options_metrics.py) : Nouvelle métrique option_expiry_proximity.
#   - Phase 15 (microstructure_guard.py, spotgamma_recalculator.py) : Nouvelle métrique microstructure_time_sensitivity.
#   - Phase 18 : Nouvelle métrique hft_time_sensitivity.
# - Tests unitaires disponibles dans tests/test_encode_time_context.py.
# Évolutions futures : Intégrer analyse de sentiment des nouvelles via
# news_analyzer.py (juin 2025).

import gzip
import hashlib
import json
import logging
import os
import signal
import sqlite3
import time
import traceback
from collections import deque
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import psutil
from sklearn.cluster import KMeans

from src.model.utils.alert_manager import send_alert
from src.model.utils.config_manager import config_manager
from src.model.utils.miya_console import miya_alerts, miya_speak

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "time_context_snapshots")
CSV_LOG_PATH = os.path.join(BASE_DIR, "data", "logs", "time_context_performance.csv")
CACHE_DIR = os.path.join(BASE_DIR, "data", "features", "cache", "time_context")
FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR, "data", "features", "feature_importance.csv"
)
DB_PATH = os.path.join(BASE_DIR, "data", "market_memory.db")

# Création des dossiers
os.makedirs(os.path.join(BASE_DIR, "data", "logs"), exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Configuration du logging
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "data", "logs", "encode_time_context.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 2.0
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes


class TimeContextEncoder:

    def __init__(
        self, config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml")
    ):
        self.log_buffer = []
        self.cache = {}
        self.buffer = deque(maxlen=1000)
        try:
            self.config = self.load_config_with_manager(config_path)
            self.buffer_size = self.config.get("buffer_size", 100)
            self.max_cache_size = self.config.get("max_cache_size", 1000)
            self.buffer_maxlen = self.config.get("buffer_maxlen", 1000)
            self.buffer = deque(maxlen=self.buffer_maxlen)
            signal.signal(signal.SIGINT, self.handle_sigint)
            self._clean_cache()
            self.initialize_db()
            miya_speak(
                "TimeContextEncoder initialisé",
                tag="TIME_ENCODER",
                voice_profile="calm",
                priority=2,
            )
            send_alert("TimeContextEncoder initialisé", priority=2)
            logger.info("TimeContextEncoder initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot_gzip("init", {"config_path": config_path})
        except Exception as e:
            error_msg = f"Erreur initialisation TimeContextEncoder: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="TIME_ENCODER", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            self.config = {
                "timezone": "America/New_York",
                "trading_session_start": "09:30",
                "trading_session_end": "16:00",
                "n_clusters": 10,
                "buffer_size": 100,
                "max_cache_size": 1000,
                "buffer_maxlen": 1000,
                "cache_hours": 24,
            }
            self.buffer_size = 100
            self.max_cache_size = 1000
            self.buffer_maxlen = 1000
            self.buffer = deque(maxlen=self.buffer_maxlen)

    def _clean_cache(self, max_size_mb: float = MAX_CACHE_SIZE_MB):
        """
        Supprime les fichiers cache expirés ou si la taille dépasse max_size_mb.
        """
        start_time = time.time()

        def clean():
            total_size = sum(
                os.path.getsize(os.path.join(CACHE_DIR, f))
                for f in os.listdir(CACHE_DIR)
                if os.path.isfile(os.path.join(CACHE_DIR, f))
            ) / (1024 * 1024)
            if total_size > max_size_mb:
                files = [
                    (f, os.path.getmtime(os.path.join(CACHE_DIR, f)))
                    for f in os.listdir(CACHE_DIR)
                ]
                files.sort(key=lambda x: x[1])
                for f, _ in files[: len(files) // 2]:
                    os.remove(os.path.join(CACHE_DIR, f))
            for filename in os.listdir(CACHE_DIR):
                path = os.path.join(CACHE_DIR, filename)
                if (
                    os.path.isfile(path)
                    and (time.time() - os.path.getmtime(path)) > CACHE_EXPIRATION
                ):
                    os.remove(path)

        try:
            self.with_retries(clean)
            latency = time.time() - start_time
            self.log_performance(
                "clean_cache", latency, success=True, data_type="cache"
            )
        except OSError as e:
            send_alert(f"Erreur nettoyage cache: {str(e)}", priority=3)
            logger.error(f"Erreur nettoyage cache: {str(e)}")
            self.log_performance(
                "clean_cache", latency, success=False, error=str(e), data_type="cache"
            )

    def handle_sigint(self, signal: int, frame: Any) -> None:
        """
        Gère l'arrêt via SIGINT en sauvegardant un snapshot.
        """
        try:
            snapshot_data = {
                "timestamp": str(datetime.now()),
                "type": "sigint",
                "log_buffer": self.log_buffer,
                "cache_size": len(self.cache),
            }
            self.save_snapshot_gzip("sigint", snapshot_data)
            miya_speak(
                "SIGINT received, snapshot saved",
                tag="TIME_ENCODER",
                voice_profile="calm",
                priority=2,
            )
            send_alert("SIGINT received, snapshot saved", priority=2)
            logger.info("SIGINT received, snapshot saved")
        except Exception as e:
            miya_alerts(
                f"Erreur gestion SIGINT: {str(e)}",
                tag="TIME_ENCODER",
                voice_profile="urgent",
                priority=5,
            )
            send_alert(f"Erreur gestion SIGINT: {str(e)}", priority=4)
            logger.error(f"Erreur gestion SIGINT: {str(e)}")
        finally:
            raise SystemExit("Terminated by SIGINT")

    def load_config_with_manager(self, config_path: str) -> Dict[str, Any]:
        """
        Charge la configuration depuis le fichier YAML via config_manager.
        """

        def load_yaml():
            config = config_manager.get_config(os.path.basename(config_path))
            if "time_context" not in config:
                raise ValueError("Clé 'time_context' manquante dans la configuration")
            required_keys = [
                "timezone",
                "trading_session_start",
                "trading_session_end",
                "n_clusters",
            ]
            missing_keys = [
                key for key in required_keys if key not in config["time_context"]
            ]
            if missing_keys:
                raise ValueError(f"Clés manquantes dans 'time_context': {missing_keys}")
            return config["time_context"]

        try:
            start_time = time.time()
            config = self.with_retries(load_yaml)
            cache_key = hashlib.sha256(str(config).encode()).hexdigest()
            self.cache[cache_key] = config
            if len(self.cache) > self.max_cache_size:
                self.cache.pop(next(iter(self.cache)))
            latency = time.time() - start_time
            miya_speak(
                "Configuration time_context chargée via config_manager",
                tag="TIME_ENCODER",
                voice_profile="calm",
                priority=2,
            )
            send_alert(
                "Configuration time_context chargée via config_manager", priority=2
            )
            logger.info("Configuration time_context chargée via config_manager")
            self.log_performance("load_config_with_manager", latency, success=True)
            self.save_snapshot_gzip(
                "load_config_with_manager", {"config_path": config_path}
            )
            return config
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur chargement config via config_manager: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="TIME_ENCODER", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance(
                "load_config_with_manager", latency, success=False, error=str(e)
            )
            return {
                "timezone": "America/New_York",
                "trading_session_start": "09:30",
                "trading_session_end": "16:00",
                "n_clusters": 10,
                "buffer_size": 100,
                "max_cache_size": 1000,
                "buffer_maxlen": 1000,
                "cache_hours": 24,
            }

    def with_retries(
        self,
        func: callable,
        max_attempts: int = MAX_RETRIES,
        delay_base: float = RETRY_DELAY,
    ) -> Any:
        """
        Exécute une fonction avec retries exponentiels.
        """
        start_time = time.time()
        for attempt in range(max_attempts):
            try:
                result = func()
                latency = time.time() - start_time
                self.log_performance(
                    f"retry_attempt_{attempt+1}", latency, success=True
                )
                return result
            except Exception as e:
                if attempt == max_attempts - 1:
                    latency = time.time() - start_time
                    self.log_performance(
                        f"retry_attempt_{attempt+1}",
                        latency,
                        success=False,
                        error=str(e),
                    )
                    send_alert(
                        f"Échec après {max_attempts} tentatives: {str(e)}", priority=4
                    )
                    logger.error(f"Échec après {max_attempts} tentatives: {str(e)}")
                    raise
                delay = delay_base * (2**attempt)
                send_alert(
                    f"Tentative {attempt+1} échouée, retry après {delay}s", priority=3
                )
                logger.warning(f"Tentative {attempt+1} échouée, retry après {delay}s")
                time.sleep(delay)

    def initialize_db(self) -> None:
        """
        Initialise la table clusters dans market_memory.db.
        """

        def create_table():
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS clusters (
                        cluster_id INTEGER,
                        event_type TEXT,
                        features TEXT,
                        timestamp TEXT,
                        PRIMARY KEY (cluster_id, timestamp)
                    )
                """
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_cluster_id ON clusters (cluster_id)"
                )
                conn.commit()

        try:
            start_time = time.time()
            self.with_retries(create_table)
            latency = time.time() - start_time
            miya_speak(
                "Base de données market_memory.db initialisée",
                tag="TIME_ENCODER",
                voice_profile="calm",
                priority=1,
            )
            send_alert("Base de données market_memory.db initialisée", priority=1)
            logger.info("Base de données market_memory.db initialisée")
            self.log_performance("initialize_db", latency, success=True)
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur initialisation market_memory.db: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="TIME_ENCODER", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance("initialize_db", latency, success=False, error=str(e))

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ) -> None:
        """
        Journalise les performances avec psutil.
        """
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_usage = psutil.cpu_percent()
            if memory_usage > 1024:
                miya_alerts(
                    f"ALERT: High memory usage ({memory_usage:.2f} MB)",
                    tag="TIME_ENCODER",
                    voice_profile="urgent",
                    priority=5,
                )
                send_alert(
                    f"ALERT: High memory usage ({memory_usage:.2f} MB)", priority=5
                )
            log_entry = {
                "timestamp": str(datetime.now()),
                "operation": operation,
                "latency": latency,
                "success": success,
                "error": error,
                "memory_usage_mb": memory_usage,
                "cpu_usage_percent": cpu_usage,
                **kwargs,
            }
            self.log_buffer.append(log_entry)
            if len(self.log_buffer) >= self.buffer_size:
                log_df = pd.DataFrame(self.log_buffer)
                if not os.path.exists(CSV_LOG_PATH):
                    log_df.to_csv(CSV_LOG_PATH, index=False, encoding="utf-8")
                else:
                    log_df.to_csv(
                        CSV_LOG_PATH,
                        mode="a",
                        header=False,
                        index=False,
                        encoding="utf-8",
                    )
                self.log_buffer = []
        except Exception as e:
            miya_alerts(
                f"Erreur logging performance: {str(e)}",
                tag="TIME_ENCODER",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur logging performance: {str(e)}", priority=3)
            logger.error(f"Erreur logging performance: {str(e)}")

    def save_snapshot_gzip(self, snapshot_type: str, data: Dict) -> None:
        """
        Sauvegarde un snapshot JSON avec compression gzip.
        """

        def save_json():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = {
                "timestamp": timestamp,
                "type": snapshot_type,
                "data": data,
                "buffer_size": len(self.log_buffer),
            }
            path = os.path.join(
                SNAPSHOT_DIR, f"snapshot_{snapshot_type}_{timestamp}.json"
            )
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            with gzip.open(f"{path}.gz", "wt", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=4)
            return path

        try:
            start_time = time.time()
            path = self.with_retries(save_json)
            file_size = os.path.getsize(f"{path}.gz") / 1024 / 1024
            if file_size > 1.0:
                send_alert(f"Snapshot size {file_size:.2f} MB exceeds 1 MB", priority=3)
            latency = time.time() - start_time
            miya_speak(
                f"Snapshot gzip {snapshot_type} saved: {path}",
                tag="TIME_ENCODER",
                voice_profile="calm",
                priority=1,
            )
            send_alert(f"Snapshot gzip {snapshot_type} saved", priority=1)
            logger.info(f"Snapshot gzip {snapshot_type} saved: {path}")
            self.log_performance("save_snapshot_gzip", latency, success=True)
        except Exception as e:
            self.log_performance("save_snapshot_gzip", 0, success=False, error=str(e))
            miya_alerts(
                f"Erreur sauvegarde snapshot gzip {snapshot_type}: {str(e)}",
                tag="TIME_ENCODER",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur sauvegarde snapshot gzip {snapshot_type}: {str(e)}", priority=3
            )
            logger.error(f"Erreur sauvegarde snapshot gzip {snapshot_type}: {str(e)}")

    def validate_shap_features(self, features: List[str]) -> bool:
        """
        Valide que les features sont dans le top 150 SHAP.
        """
        try:
            if not os.path.exists(FEATURE_IMPORTANCE_PATH):
                miya_alerts(
                    "Fichier SHAP manquant",
                    tag="TIME_ENCODER",
                    voice_profile="urgent",
                    priority=4,
                )
                send_alert("Fichier SHAP manquant", priority=4)
                logger.error("Fichier SHAP manquant")
                return False
            shap_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
            if len(shap_df) < 150:
                miya_alerts(
                    f"Nombre insuffisant de SHAP features: {len(shap_df)} < 150",
                    tag="TIME_ENCODER",
                    voice_profile="urgent",
                    priority=4,
                )
                send_alert(
                    f"Nombre insuffisant de SHAP features: {len(shap_df)}", priority=4
                )
                logger.error(f"Nombre insuffisant de SHAP features: {len(shap_df)}")
                return False
            valid_features = set(shap_df["feature"].head(150))
            missing = [f for f in features if f not in valid_features]
            if missing:
                miya_alerts(
                    f"Features non incluses dans top 150 SHAP: {missing}",
                    tag="TIME_ENCODER",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(
                    f"Features non incluses dans top 150 SHAP: {missing}", priority=3
                )
                logger.warning(f"Features non incluses dans top 150 SHAP: {missing}")
            miya_speak(
                "SHAP features validées",
                tag="TIME_ENCODER",
                voice_profile="calm",
                priority=1,
            )
            send_alert("SHAP features validées", priority=1)
            logger.info("SHAP features validées")
            return True
        except Exception as e:
            miya_alerts(
                f"Erreur validation SHAP features: {str(e)}",
                tag="TIME_ENCODER",
                voice_profile="urgent",
                priority=4,
            )
            send_alert(f"Erreur validation SHAP features: {str(e)}", priority=4)
            logger.error(f"Erreur validation SHAP features: {str(e)}")
            return False

    def validate_iqfeed_data(self, data: pd.DataFrame) -> bool:
        """
        Valide les données IQFeed.
        """
        try:
            required_cols = ["timestamp"]
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                error_msg = f"Colonnes IQFeed manquantes: {missing_cols}"
                miya_alerts(
                    error_msg, tag="TIME_ENCODER", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return False
            if data["timestamp"].isna().any():
                error_msg = "Valeurs NaN dans timestamp IQFeed"
                miya_alerts(
                    error_msg, tag="TIME_ENCODER", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return False
            miya_speak(
                "Données IQFeed validées",
                tag="TIME_ENCODER",
                voice_profile="calm",
                priority=1,
            )
            send_alert("Données IQFeed validées", priority=1)
            logger.info("Données IQFeed validées")
            return True
        except Exception as e:
            error_msg = (
                f"Erreur validation données IQFeed: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, tag="TIME_ENCODER", voice_profile="urgent", priority=4
            )
            send_alert(error_msg, priority=4)
            logger.error(error_msg)
            return False

    def cache_features(self, features: pd.DataFrame) -> None:
        """
        Met en cache les features temporelles.
        """

        def save_cache():
            cache_path = os.path.join(CACHE_DIR, "latest.csv")
            os.makedirs(CACHE_DIR, exist_ok=True)
            features.to_csv(cache_path, index=False, encoding="utf-8")
            return cache_path

        try:
            start_time = time.time()
            path = self.with_retries(save_cache)
            file_size = os.path.getsize(path) / 1024 / 1024
            if file_size > 1.0:
                send_alert(f"Cache size {file_size:.2f} MB exceeds 1 MB", priority=3)
            latency = time.time() - start_time
            miya_speak(
                f"Features temporelles mises en cache: {path}",
                tag="TIME_ENCODER",
                voice_profile="calm",
                priority=1,
            )
            send_alert(f"Features temporelles mises en cache: {path}", priority=1)
            logger.info(f"Features temporelles mises en cache: {path}")
            self.log_performance("cache_features", latency, success=True)
        except Exception as e:
            self.log_performance("cache_features", 0, success=False, error=str(e))
            miya_alerts(
                f"Erreur mise en cache features: {str(e)}",
                tag="TIME_ENCODER",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur mise en cache features: {str(e)}", priority=3)
            logger.error(f"Erreur mise en cache features: {str(e)}")

    def compute_option_expiry_proximity(
        self, data: pd.DataFrame, options_data: pd.DataFrame
    ) -> pd.Series:
        """
        Calcule la proximité temporelle des expirations d'options (Phase 13).
        """
        try:
            start_time = time.time()
            if (
                "timestamp" not in data.columns
                or "expiry_date" not in options_data.columns
            ):
                error_msg = "Colonnes timestamp ou expiry_date manquantes"
                miya_alerts(
                    error_msg, tag="TIME_ENCODER", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=data.index)
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            options_data["expiry_date"] = pd.to_datetime(options_data["expiry_date"])
            proximity = (
                options_data["expiry_date"] - data["timestamp"]
            ).dt.total_seconds() / (24 * 3600)
            result = proximity.fillna(0.0)
            latency = time.time() - start_time
            self.log_performance(
                "compute_option_expiry_proximity", latency, success=True
            )
            return result
        except Exception as e:
            self.log_performance(
                "compute_option_expiry_proximity", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_option_expiry_proximity: {str(e)}",
                tag="TIME_ENCODER",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_option_expiry_proximity: {str(e)}", priority=4
            )
            logger.error(f"Erreur dans compute_option_expiry_proximity: {str(e)}")
            return pd.Series(0.0, index=data.index)

    def compute_microstructure_time_sensitivity(
        self, data: pd.DataFrame, microstructure_data: pd.DataFrame
    ) -> pd.Series:
        """
        Calcule la sensibilité temporelle des événements de microstructure (Phase 15).
        """
        try:
            start_time = time.time()
            if (
                "timestamp" not in data.columns
                or "spoofing_score" not in microstructure_data.columns
            ):
                error_msg = "Colonnes timestamp ou spoofing_score manquantes"
                miya_alerts(
                    error_msg, tag="TIME_ENCODER", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=data.index)
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            microstructure_data["timestamp"] = pd.to_datetime(
                microstructure_data["timestamp"]
            )
            result = microstructure_data["spoofing_score"].fillna(0.0)
            latency = time.time() - start_time
            self.log_performance(
                "compute_microstructure_time_sensitivity", latency, success=True
            )
            return result
        except Exception as e:
            self.log_performance(
                "compute_microstructure_time_sensitivity",
                0,
                success=False,
                error=str(e),
            )
            miya_alerts(
                f"Erreur dans compute_microstructure_time_sensitivity: {str(e)}",
                tag="TIME_ENCODER",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_microstructure_time_sensitivity: {str(e)}",
                priority=4,
            )
            logger.error(
                f"Erreur dans compute_microstructure_time_sensitivity: {str(e)}"
            )
            return pd.Series(0.0, index=data.index)

    def compute_hft_time_sensitivity(
        self, data: pd.DataFrame, hft_data: pd.DataFrame
    ) -> pd.Series:
        """
        Calcule la sensibilité temporelle de l'activité HFT (Phase 18).
        """
        try:
            start_time = time.time()
            if (
                "timestamp" not in data.columns
                or "hft_activity_score" not in hft_data.columns
            ):
                error_msg = "Colonnes timestamp ou hft_activity_score manquantes"
                miya_alerts(
                    error_msg, tag="TIME_ENCODER", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=data.index)
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            hft_data["timestamp"] = pd.to_datetime(hft_data["timestamp"])
            result = hft_data["hft_activity_score"].fillna(0.0)
            latency = time.time() - start_time
            self.log_performance("compute_hft_time_sensitivity", latency, success=True)
            return result
        except Exception as e:
            self.log_performance(
                "compute_hft_time_sensitivity", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_hft_time_sensitivity: {str(e)}",
                tag="TIME_ENCODER",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_hft_time_sensitivity: {str(e)}", priority=4
            )
            logger.error(f"Erreur dans compute_hft_time_sensitivity: {str(e)}")
            return pd.Series(0.0, index=data.index)

    def cluster_time_features(self, features: pd.DataFrame) -> pd.Series:
        """
        Clusterise les features temporelles avec KMeans.
        """
        try:
            start_time = time.time()
            time_features = [
                "time_of_day_sin",
                "time_of_day_cos",
                "day_of_week",
                "is_trading_session",
                "is_expiry_day",
                "option_expiry_proximity",
                "microstructure_time_sensitivity",
                "hft_time_sensitivity",
            ]
            if not all(col in features.columns for col in time_features):
                error_msg = f"Features temporelles manquantes: {[col for col in time_features if col not in features.columns]}"
                miya_alerts(
                    error_msg, tag="TIME_ENCODER", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0, index=features.index)

            data = features[time_features].copy()
            kmeans = KMeans(
                n_clusters=self.config.get("n_clusters", 10), random_state=42
            )
            cluster_ids = kmeans.fit_predict(data)
            cluster_series = pd.Series(
                cluster_ids, index=features.index, name="cluster_id"
            )

            def save_clusters():
                with sqlite3.connect(DB_PATH) as conn:
                    for idx, row in features.iterrows():
                        cluster_data = {
                            "cluster_id": int(cluster_series.loc[idx]),
                            "event_type": "time_context",
                            "features": json.dumps(row[time_features].to_dict()),
                            "timestamp": str(row["timestamp"]),
                        }
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO clusters (cluster_id, event_type, features, timestamp)
                            VALUES (?, ?, ?, ?)
                        """,
                            (
                                cluster_data["cluster_id"],
                                cluster_data["event_type"],
                                cluster_data["features"],
                                cluster_data["timestamp"],
                            ),
                        )
                    conn.commit()

            self.with_retries(save_clusters)
            latency = time.time() - start_time
            miya_speak(
                f"Features temporelles clusterisées, {len(set(cluster_ids))} clusters",
                tag="TIME_ENCODER",
                voice_profile="calm",
                priority=1,
            )
            send_alert(
                f"Features temporelles clusterisées, {len(set(cluster_ids))} clusters",
                priority=1,
            )
            logger.info(
                f"Features temporelles clusterisées, {len(set(cluster_ids))} clusters"
            )
            self.log_performance(
                "cluster_time_features",
                latency,
                success=True,
                num_clusters=len(set(cluster_ids)),
            )
            self.save_snapshot_gzip(
                "cluster_time_features",
                {"num_clusters": len(set(cluster_ids)), "num_rows": len(features)},
            )
            return cluster_series
        except Exception as e:
            self.log_performance(
                "cluster_time_features", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur clusterisation features: {str(e)}",
                tag="TIME_ENCODER",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur clusterisation features: {str(e)}", priority=3)
            logger.error(f"Erreur clusterisation features: {str(e)}")
            return pd.Series(0, index=features.index)

    def encode_time_context(
        self,
        data: pd.DataFrame,
        options_data: pd.DataFrame,
        microstructure_data: pd.DataFrame,
        hft_data: pd.DataFrame,
        timestamp_col: str = "timestamp",
        timezone: str = "America/New_York",
    ) -> pd.DataFrame:
        """
        Encode le temps (heures, jours, session, expiration) pour contextualiser les données IQFeed.

        Args:
            data (pd.DataFrame): Données à enrichir avec des features temporelles.
            options_data (pd.DataFrame): Données des options avec expiry_date.
            microstructure_data (pd.DataFrame): Données de microstructure avec spoofing_score.
            hft_data (pd.DataFrame): Données HFT avec hft_activity_score.
            timestamp_col (str, optional): Nom de la colonne temporelle (par défaut : 'timestamp').
            timezone (str, optional): Fuseau horaire pour ajuster les timestamps (par défaut : 'America/New_York').

        Returns:
            pd.DataFrame: Données avec encodages temporels (time_of_day_sin, day_of_week, etc.).
        """
        try:
            start_time = time.time()
            cache_key = hashlib.sha256(data.to_json().encode()).hexdigest()
            cache_path = os.path.join(CACHE_DIR, f"{cache_key}.csv")
            if os.path.exists(cache_path):
                cached_data = pd.read_csv(cache_path)
                if len(cached_data) == len(data) and cached_data["timestamp"].equals(
                    data[timestamp_col]
                ):
                    latency = time.time() - start_time
                    miya_speak(
                        "Features temporelles chargées depuis cache",
                        tag="TIME_ENCODER",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert("Features temporelles chargées depuis cache", priority=1)
                    self.log_performance(
                        "encode_time_context_cache_hit", latency, success=True
                    )
                    return cached_data

            if not self.validate_iqfeed_data(data):
                error_msg = "Échec validation données IQFeed"
                miya_alerts(
                    error_msg, tag="TIME_ENCODER", voice_profile="urgent", priority=5
                )
                send_alert(error_msg, priority=5)
                raise ValueError(error_msg)

            result = data.copy()
            if result.empty:
                error_msg = "DataFrame vide dans encode_time_context"
                miya_alerts(
                    error_msg, tag="TIME_ENCODER", voice_profile="urgent", priority=5
                )
                send_alert(error_msg, priority=5)
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Vérification et création d’un timestamp si absent
            if timestamp_col not in result.columns:
                miya_speak(
                    f"Colonne '{timestamp_col}' manquante, création d’un timestamp par défaut",
                    tag="TIME_ENCODER",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(
                    f"Colonne '{timestamp_col}' manquante, création par défaut",
                    priority=3,
                )
                logger.warning(
                    f"Colonne '{timestamp_col}' manquante, création par défaut"
                )
                default_start = (
                    pd.Timestamp("2025-04-14") if result.empty else pd.Timestamp.now()
                )
                result[timestamp_col] = pd.date_range(
                    start=default_start, periods=len(result), freq="1min"
                )
                miya_speak(
                    "Timestamp par défaut ajouté",
                    tag="TIME_ENCODER",
                    voice_profile="calm",
                    priority=1,
                )
                send_alert("Timestamp par défaut ajouté", priority=1)

            # Conversion en datetime avec gestion des erreurs
            result[timestamp_col] = pd.to_datetime(
                result[timestamp_col], errors="coerce"
            )
            if result[timestamp_col].isna().any():
                miya_speak(
                    f"Valeurs invalides dans '{timestamp_col}', remplacées par la première date valide",
                    tag="TIME_ENCODER",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(
                    f"Valeurs invalides dans '{timestamp_col}', imputation", priority=3
                )
                logger.warning(f"Valeurs invalides dans '{timestamp_col}', imputation")
                first_valid_time = (
                    result[timestamp_col].dropna().iloc[0]
                    if not result[timestamp_col].dropna().empty
                    else pd.Timestamp("2025-04-14")
                )
                result[timestamp_col] = result[timestamp_col].fillna(first_valid_time)

            # Convertir les timestamps dans le fuseau horaire spécifié
            result[timestamp_col] = (
                result[timestamp_col]
                .dt.tz_localize("UTC", ambiguous="raise")
                .dt.tz_convert(timezone)
            )

            # Encodages temporels
            result["time_of_day_sin"] = np.sin(
                2 * np.pi * result[timestamp_col].dt.hour / 24
            )
            result["time_of_day_cos"] = np.cos(
                2 * np.pi * result[timestamp_col].dt.hour / 24
            )
            result["day_of_week"] = result[timestamp_col].dt.dayofweek
            result["is_trading_session"] = (
                result[timestamp_col]
                .dt.time.between(
                    pd.Timestamp(
                        self.config.get("trading_session_start", "09:30")
                    ).time(),
                    pd.Timestamp(
                        self.config.get("trading_session_end", "16:00")
                    ).time(),
                )
                .astype(int)
            )
            result["is_expiry_day"] = (
                (result[timestamp_col].dt.day_name() == "Friday")
                & (result[timestamp_col].dt.day >= 15)
                & (result[timestamp_col].dt.day <= 21)
            ).astype(int)

            # Gestion des NaN dans les features temporelles
            for col in [
                "time_of_day_sin",
                "time_of_day_cos",
                "day_of_week",
                "is_trading_session",
                "is_expiry_day",
            ]:
                if result[col].isna().any():
                    if col in ["time_of_day_sin", "time_of_day_cos"]:
                        median_value = result[col].median()
                        result[col] = result[col].fillna(median_value)
                        miya_speak(
                            f"NaN détectés dans {col}, imputés à {median_value:.2f}",
                            tag="TIME_ENCODER",
                            voice_profile="warning",
                            priority=3,
                        )
                        send_alert(
                            f"NaN dans {col}, imputés à {median_value:.2f}", priority=3
                        )
                        logger.warning(f"NaN dans {col}, imputés à {median_value:.2f}")
                    else:
                        result[col] = result[col].fillna(0)
                        miya_speak(
                            f"NaN détectés dans {col}, imputés à 0",
                            tag="TIME_ENCODER",
                            voice_profile="warning",
                            priority=3,
                        )
                        send_alert(f"NaN dans {col}, imputés à 0", priority=3)
                        logger.warning(f"NaN dans {col}, imputés à 0")

            # Ajout des métriques des Phases 13, 15, 18
            result["option_expiry_proximity"] = self.compute_option_expiry_proximity(
                result, options_data
            )
            result["microstructure_time_sensitivity"] = (
                self.compute_microstructure_time_sensitivity(
                    result, microstructure_data
                )
            )
            result["hft_time_sensitivity"] = self.compute_hft_time_sensitivity(
                result, hft_data
            )

            # Validation SHAP
            features = [
                "time_of_day_sin",
                "time_of_day_cos",
                "day_of_week",
                "is_trading_session",
                "is_expiry_day",
                "option_expiry_proximity",
                "microstructure_time_sensitivity",
                "hft_time_sensitivity",
            ]
            self.validate_shap_features(features)

            # Clusterisation
            result["cluster_id"] = self.cluster_time_features(result)

            # Mise en cache
            self.cache_features(result)

            latency = time.time() - start_time
            miya_speak(
                "Features temporelles encodées et clusterisées",
                tag="TIME_ENCODER",
                voice_profile="calm",
                priority=1,
            )
            send_alert("Features temporelles encodées et clusterisées", priority=1)
            logger.info("Features temporelles encodées et clusterisées")
            self.log_performance(
                "encode_time_context", latency, success=True, num_rows=len(result)
            )
            self.save_snapshot_gzip(
                "encode_time_context",
                {"num_rows": len(result), "features": result.columns.tolist()},
            )
            return result
        except Exception as e:
            latency = time.time() - start_time
            error_msg = (
                f"Erreur dans encode_time_context: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, tag="TIME_ENCODER", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance(
                "encode_time_context", latency, success=False, error=str(e)
            )
            self.save_snapshot_gzip("encode_time_context", {"error": str(e)})
            result = data.copy()
            result["timestamp"] = pd.date_range(
                start=pd.Timestamp.now(), periods=len(result), freq="1min"
            )
            for col in [
                "time_of_day_sin",
                "time_of_day_cos",
                "day_of_week",
                "is_trading_session",
                "is_expiry_day",
                "option_expiry_proximity",
                "microstructure_time_sensitivity",
                "hft_time_sensitivity",
                "cluster_id",
            ]:
                result[col] = 0.0
            return result


if __name__ == "__main__":
    try:
        encoder = TimeContextEncoder()
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "close": np.random.normal(5100, 10, 100),
                "volume": np.random.randint(100, 1000, 100),
            }
        )
        options_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "expiry_date": pd.date_range("2025-04-15", periods=100, freq="1d"),
            }
        )
        microstructure_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "spoofing_score": np.random.uniform(0, 1, 100),
            }
        )
        hft_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "hft_activity_score": np.random.uniform(0, 1, 100),
            }
        )
        result = encoder.encode_time_context(
            data, options_data, microstructure_data, hft_data
        )
        print(
            result[
                [
                    "timestamp",
                    "time_of_day_sin",
                    "time_of_day_cos",
                    "day_of_week",
                    "is_trading_session",
                    "is_expiry_day",
                    "option_expiry_proximity",
                    "microstructure_time_sensitivity",
                    "hft_time_sensitivity",
                    "cluster_id",
                ]
            ].head()
        )
        miya_speak(
            "Test encode_time_context terminé",
            tag="TEST",
            voice_profile="calm",
            priority=1,
        )
        send_alert("Test encode_time_context terminé", priority=1)
    except Exception as e:
        miya_alerts(
            f"Erreur test: {str(e)}\n{traceback.format_exc()}",
            tag="ALERT",
            voice_profile="urgent",
            priority=3,
        )
        send_alert(f"Erreur test: {str(e)}", priority=3)
        logger.error(f"Erreur test: {str(e)}")
        raise
