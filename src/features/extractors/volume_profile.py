# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/features/extractors/volume_profile.py
# Extrait les informations du profil de volume (POC, VAH, VAL) à partir des données IQFeed.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Génère les profils de volume (POC, VAH, VAL), clusterise les profils (méthode 7), stocke dans market_memory.db,
#        utilise un cache intermédiaire, enregistre des logs psutil, valide SHAP (méthode 17),
#        et normalise les métriques pour l’inférence. Conforme à la Phase 5 (volume profile).
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, scikit-learn>=1.5.0,<2.0.0, psutil>=5.9.0,<6.0.0, sqlite3, matplotlib>=3.7.0,<4.0.0, json, gzip, hashlib, logging
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
# - src/model/utils/db_maintenance.py
#
# Inputs :
# - config/es_config.yaml
# - data/iqfeed/merged_data.csv
# - market_memory.db
# - data/features/feature_importance.csv
#
# Outputs :
# - data/features/cache/volume_profile/
# - data/logs/volume_profile_performance.csv
# - data/features/volume_profile_snapshots/
# - data/figures/volume_profile/
#
# Notes :
# - Utilise exclusivement IQFeed pour les données (remplace dxFeed).
# - Suppression de toute référence à dxFeed, obs_t, 320 features, et 81 features.
# - Intègre les Phases 1-18, spécifiquement Phase 5 (volume_profile.py).
# - Tests unitaires disponibles dans tests/test_volume_profile.py.
# - Clusters stockés dans market_memory.db (table clusters) avec index pour performance.
# Évolution future : Intégration avec feature_pipeline.py pour top 150
# SHAP, migration API Investing.com (juin 2025).

import gzip
import hashlib
import json
import logging
import os
import signal
import sqlite3
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from src.model.utils.alert_manager import send_alert
from src.model.utils.config_manager import config_manager
from src.model.utils.db_maintenance import ensure_table_and_index
from src.model.utils.miya_console import miya_alerts, miya_speak

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data", "features", "cache", "volume_profile")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "features", "volume_profile_snapshots")
CSV_LOG_PATH = os.path.join(BASE_DIR, "data", "logs", "volume_profile_performance.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures", "volume_profile")
DB_PATH = os.path.join(BASE_DIR, "data", "market_memory.db")
FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR, "data", "features", "feature_importance.csv"
)

# Création des dossiers
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data", "logs"), exist_ok=True)

# Configuration du logging
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "data", "logs", "volume_profile.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 2.0
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes
WINDOW_SIZE = 100  # Fenêtre glissante pour normalisation


class VolumeProfileExtractor:

    def __init__(
        self, config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml")
    ):
        self.log_buffer = []
        self.cache = {}
        self.scaler = MinMaxScaler()
        try:
            self.config = self.load_config_with_manager(config_path)
            self.buffer_size = self.config.get("buffer_size", 100)
            self.max_cache_size = self.config.get("max_cache_size", 1000)
            signal.signal(signal.SIGINT, self.handle_sigint)
            self._clean_cache()
            miya_speak(
                "VolumeProfileExtractor initialisé",
                tag="VOLUME_PROFILE",
                voice_profile="calm",
                priority=2,
            )
            send_alert("VolumeProfileExtractor initialisé", priority=1)
            logger.info("VolumeProfileExtractor initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config_path": config_path})
        except Exception as e:
            error_msg = f"Erreur initialisation VolumeProfileExtractor: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="VOLUME_PROFILE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            self.config = {
                "buffer_size": 100,
                "max_cache_size": 1000,
                "cache_hours": 24,
                "n_clusters": 10,
            }

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

    def load_config_with_manager(self, config_path: str) -> Dict[str, Any]:
        """
        Charge la configuration depuis le fichier YAML via config_manager.
        """

        def load_yaml():
            config = config_manager.get_config(os.path.basename(config_path))
            if "volume_profile_extractor" not in config:
                raise ValueError(
                    "Clé 'volume_profile_extractor' manquante dans la configuration"
                )
            required_keys = [
                "buffer_size",
                "max_cache_size",
                "cache_hours",
                "n_clusters",
            ]
            missing_keys = [
                key
                for key in required_keys
                if key not in config["volume_profile_extractor"]
            ]
            if missing_keys:
                raise ValueError(
                    f"Clés manquantes dans 'volume_profile_extractor': {missing_keys}"
                )
            return config["volume_profile_extractor"]

        try:
            start_time = time.time()
            config = self.with_retries(load_yaml)
            latency = time.time() - start_time
            miya_speak(
                "Configuration volume_profile_extractor chargée via config_manager",
                tag="VOLUME_PROFILE",
                voice_profile="calm",
                priority=2,
            )
            send_alert(
                "Configuration volume_profile_extractor chargée via config_manager",
                priority=1,
            )
            logger.info(
                "Configuration volume_profile_extractor chargée via config_manager"
            )
            self.log_performance("load_config_with_manager", latency, success=True)
            self.save_snapshot("load_config_with_manager", {"config_path": config_path})
            return config
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur chargement config via config_manager: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="VOLUME_PROFILE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance(
                "load_config_with_manager", latency, success=False, error=str(e)
            )
            return {
                "buffer_size": 100,
                "max_cache_size": 1000,
                "cache_hours": 24,
                "n_clusters": 10,
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
                    tag="VOLUME_PROFILE",
                    voice_profile="urgent",
                    priority=5,
                )
                send_alert(
                    f"ALERT: High memory usage ({memory_usage:.2f} MB)", priority=4
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
                os.makedirs(os.path.dirname(CSV_LOG_PATH), exist_ok=True)
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
                tag="VOLUME_PROFILE",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur logging performance: {str(e)}", priority=4)
            logger.error(f"Erreur logging performance: {str(e)}")

    def save_snapshot(self, snapshot_type: str, data: Dict) -> None:
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
                f"Snapshot {snapshot_type} saved: {path}.gz",
                tag="VOLUME_PROFILE",
                voice_profile="calm",
                priority=1,
            )
            send_alert(f"Snapshot {snapshot_type} saved: {path}.gz", priority=1)
            logger.info(f"Snapshot {snapshot_type} saved: {path}.gz")
            self.log_performance("save_snapshot", latency, success=True)
        except Exception as e:
            self.log_performance("save_snapshot", 0, success=False, error=str(e))
            miya_alerts(
                f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}",
                tag="VOLUME_PROFILE",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}", priority=4
            )
            logger.error(f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}")

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
            self.save_snapshot("sigint", snapshot_data)
            miya_speak(
                "SIGINT received, snapshot saved",
                tag="VOLUME_PROFILE",
                voice_profile="calm",
                priority=2,
            )
            send_alert("SIGINT received, snapshot saved", priority=2)
            logger.info("SIGINT received, snapshot saved")
        except Exception as e:
            miya_alerts(
                f"Erreur gestion SIGINT: {str(e)}",
                tag="VOLUME_PROFILE",
                voice_profile="urgent",
                priority=5,
            )
            send_alert(f"Erreur gestion SIGINT: {str(e)}", priority=4)
            logger.error(f"Erreur gestion SIGINT: {str(e)}")
        finally:
            raise SystemExit("Terminated by SIGINT")

    def validate_iqfeed_data(self, data: pd.DataFrame) -> bool:
        """
        Valide les données IQFeed pour les profils de volume.
        """
        try:
            required_cols = ["timestamp", "close", "volume"]
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                error_msg = f"Colonnes IQFeed manquantes: {missing_cols}"
                miya_alerts(
                    error_msg, tag="VOLUME_PROFILE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return False
            if data[required_cols].isna().any().any():
                error_msg = "Valeurs NaN dans les données IQFeed"
                miya_alerts(
                    error_msg, tag="VOLUME_PROFILE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return False
            if not np.isfinite(
                data[required_cols].select_dtypes(include=[np.number]).values
            ).all():
                error_msg = "Valeurs infinies dans les données IQFeed"
                miya_alerts(
                    error_msg, tag="VOLUME_PROFILE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return False
            miya_speak(
                "Données IQFeed validées",
                tag="VOLUME_PROFILE",
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
                error_msg, tag="VOLUME_PROFILE", voice_profile="urgent", priority=4
            )
            send_alert(error_msg, priority=4)
            logger.error(error_msg)
            return False

    def validate_shap_features(self, features: List[str]) -> bool:
        """
        Valide que les features sont dans le top 150 SHAP.
        """
        try:
            if not os.path.exists(FEATURE_IMPORTANCE_PATH):
                miya_alerts(
                    "Fichier SHAP manquant",
                    tag="VOLUME_PROFILE",
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
                    tag="VOLUME_PROFILE",
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
                    tag="VOLUME_PROFILE",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(
                    f"Features non incluses dans top 150 SHAP: {missing}", priority=3
                )
                logger.warning(f"Features non incluses dans top 150 SHAP: {missing}")
            miya_speak(
                "SHAP features validées",
                tag="VOLUME_PROFILE",
                voice_profile="calm",
                priority=1,
            )
            send_alert("SHAP features validées", priority=1)
            logger.info("SHAP features validées")
            return True
        except Exception as e:
            miya_alerts(
                f"Erreur validation SHAP features: {str(e)}",
                tag="VOLUME_PROFILE",
                voice_profile="urgent",
                priority=4,
            )
            send_alert(f"Erreur validation SHAP features: {str(e)}", priority=4)
            logger.error(f"Erreur validation SHAP features: {str(e)}")
            return False

    def cache_profiles(self, profiles: pd.DataFrame, cache_key: str) -> None:
        """
        Met en cache les profils de volume.
        """

        def save_cache():
            cache_path = os.path.join(CACHE_DIR, f"{cache_key}.csv")
            os.makedirs(CACHE_DIR, exist_ok=True)
            profiles.to_csv(cache_path, index=False, encoding="utf-8")
            return cache_path

        try:
            start_time = time.time()
            path = self.with_retries(save_cache)
            self.cache[cache_key] = {"timestamp": datetime.now(), "path": path}
            current_time = datetime.now()
            expired_keys = [
                k
                for k, v in self.cache.items()
                if (current_time - v["timestamp"]).total_seconds()
                > self.config.get("cache_hours", 24) * 3600
            ]
            for k in expired_keys:
                try:
                    os.remove(self.cache[k]["path"])
                except BaseException:
                    pass
                self.cache.pop(k)
            if len(self.cache) > self.max_cache_size:
                oldest_key = min(self.cache, key=lambda k: self.cache[k]["timestamp"])
                try:
                    os.remove(self.cache[oldest_key]["path"])
                except BaseException:
                    pass
                self.cache.pop(oldest_key)
            latency = time.time() - start_time
            miya_speak(
                f"Profils mis en cache: {path}",
                tag="VOLUME_PROFILE",
                voice_profile="calm",
                priority=1,
            )
            send_alert(f"Profils mis en cache: {path}", priority=1)
            logger.info(f"Profils mis en cache: {path}")
            self.log_performance(
                "cache_profiles", latency, success=True, cache_size=len(self.cache)
            )
        except Exception as e:
            self.log_performance("cache_profiles", 0, success=False, error=str(e))
            miya_alerts(
                f"Erreur mise en cache profils: {str(e)}",
                tag="VOLUME_PROFILE",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur mise en cache profils: {str(e)}", priority=4)
            logger.error(f"Erreur mise en cache profils: {str(e)}")

    def cluster_profiles(self, profiles: pd.DataFrame) -> pd.DataFrame:
        """
        Clusterise les profils de volume avec KMeans.
        """
        try:
            start_time = time.time()
            if profiles.empty:
                miya_speak(
                    "Profils vides, clusterisation ignorée",
                    tag="VOLUME_PROFILE",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert("Profils vides, clusterisation ignorée", priority=3)
                logger.warning("Profils vides, clusterisation ignorée")
                return pd.DataFrame()
            feature_cols = ["poc", "vah", "val"]
            data = profiles[feature_cols].copy()
            if len(data) < self.config.get("n_clusters", 10):
                miya_speak(
                    f"Nombre de profils ({len(data)}) insuffisant pour {self.config.get('n_clusters', 10)} clusters, clusterisation ignorée",
                    tag="VOLUME_PROFILE",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(
                    "Nombre de profils insuffisant pour clusterisation", priority=3
                )
                logger.warning(
                    f"Nombre de profils ({len(data)}) insuffisant pour {self.config.get('n_clusters', 10)} clusters"
                )
                return pd.DataFrame()
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            kmeans = KMeans(
                n_clusters=self.config.get("n_clusters", 10),
                random_state=self.config.get("random_state", 42),
            )
            data["cluster_id"] = kmeans.fit_predict(scaled_data)
            centroids = scaler.inverse_transform(kmeans.cluster_centers_)
            clusters_df = pd.DataFrame(
                {
                    "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                    * len(data),
                    "cluster_id": data["cluster_id"],
                    "centroid": [
                        json.dumps(centroid.tolist())
                        for centroid in centroids[data["cluster_id"]]
                    ],
                    "cluster_size": [
                        np.sum(data["cluster_id"] == i) for i in data["cluster_id"]
                    ],
                    "event_type": ["VOLUME_PROFILE"] * len(data),
                }
            )
            latency = time.time() - start_time
            miya_speak(
                f"Clustered {len(data)} profils dans {self.config.get('n_clusters', 10)} clusters",
                tag="VOLUME_PROFILE",
                voice_profile="calm",
                priority=1,
            )
            send_alert(f"Clustered {len(data)} profils", priority=1)
            logger.info(
                f"Clustered {len(data)} profils dans {self.config.get('n_clusters', 10)} clusters"
            )
            self.log_performance(
                "cluster_profiles", latency, success=True, num_clusters=len(data)
            )
            return clusters_df
        except Exception as e:
            self.log_performance("cluster_profiles", 0, success=False, error=str(e))
            miya_alerts(
                f"Erreur dans cluster_profiles: {str(e)}",
                tag="VOLUME_PROFILE",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans cluster_profiles: {str(e)}", priority=4)
            logger.error(f"Erreur dans cluster_profiles: {str(e)}")
            return pd.DataFrame()

    def store_clusters(self, clusters_df: pd.DataFrame) -> None:
        """
        Stocke les clusters dans market_memory.db.
        """

        def check_db_size():
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM clusters")
            count = cursor.fetchone()[0]
            conn.close()
            if count > 1_000_000:
                miya_alerts(
                    f"Table clusters dépasse 1M d'entrées ({count})",
                    tag="VOLUME_PROFILE",
                    voice_profile="urgent",
                    priority=4,
                )
                send_alert(f"Table clusters dépasse 1M d'entrées ({count})", priority=4)
                logger.warning(f"Table clusters dépasse 1M d'entrées ({count})")

        def execute_sql():
            check_db_size()
            conn = sqlite3.connect(DB_PATH)
            ensure_table_and_index(conn, "clusters", ["timestamp", "cluster_id"])
            clusters_df.to_sql("clusters", conn, if_exists="append", index=False)
            conn.commit()
            conn.close()

        try:
            start_time = time.time()
            self.with_retries(execute_sql)
            latency = time.time() - start_time
            miya_speak(
                f"Clusters stockés dans {DB_PATH}",
                tag="VOLUME_PROFILE",
                voice_profile="calm",
                priority=1,
            )
            send_alert(f"Clusters stockés dans {DB_PATH}", priority=1)
            logger.info(f"Clusters stockés dans {DB_PATH}")
            self.log_performance(
                "store_clusters", latency, success=True, num_clusters=len(clusters_df)
            )
        except Exception as e:
            self.log_performance("store_clusters", 0, success=False, error=str(e))
            miya_alerts(
                f"Erreur dans store_clusters: {str(e)}",
                tag="VOLUME_PROFILE",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans store_clusters: {str(e)}", priority=4)
            logger.error(f"Erreur dans store_clusters: {str(e)}")

    def plot_profiles(self, profiles: pd.DataFrame, timestamp: str) -> None:
        """
        Génère des visualisations des profils de volume.
        """
        try:
            start_time = time.time()
            timestamp_safe = timestamp.replace(":", "-")
            plt.figure(figsize=(12, 6))
            plt.plot(profiles["timestamp"], profiles["poc"], label="POC", color="blue")
            plt.plot(profiles["timestamp"], profiles["vah"], label="VAH", color="green")
            plt.plot(profiles["timestamp"], profiles["val"], label="VAL", color="red")
            plt.title(f"Volume Profile - {timestamp}")
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True)
            plt.savefig(
                os.path.join(FIGURES_DIR, f"volume_profile_{timestamp_safe}.png")
            )
            plt.close()
            latency = time.time() - start_time
            miya_speak(
                f"Visualisations générées: {FIGURES_DIR}",
                tag="VOLUME_PROFILE",
                voice_profile="calm",
                priority=2,
            )
            send_alert(f"Visualisations générées: {FIGURES_DIR}", priority=2)
            logger.info(f"Visualisations générées: {FIGURES_DIR}")
            self.log_performance("plot_profiles", latency, success=True)
        except Exception as e:
            self.log_performance("plot_profiles", 0, success=False, error=str(e))
            miya_alerts(
                f"Erreur dans plot_profiles: {str(e)}",
                tag="VOLUME_PROFILE",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans plot_profiles: {str(e)}", priority=4)
            logger.error(f"Erreur dans plot_profiles: {str(e)}")

    def normalize_profiles(
        self, data: pd.DataFrame, profiles: List[str]
    ) -> pd.DataFrame:
        """
        Normalise les métriques de profil de volume avec MinMaxScaler sur une fenêtre glissante.
        """
        try:
            start_time = time.time()
            for col in profiles:
                if col in data.columns:
                    scaled = []
                    for i in range(len(data)):
                        window = (
                            data[col]
                            .iloc[max(0, i - WINDOW_SIZE + 1) : i + 1]
                            .values.reshape(-1, 1)
                        )
                        if len(window) >= 2:  # Nécessite au moins 2 points pour scaler
                            self.scaler.fit(window)
                            scaled_value = self.scaler.transform([[data[col].iloc[i]]])[
                                0
                            ][0]
                        else:
                            scaled_value = 0.0
                        scaled.append(scaled_value)
                    data[f"{col}_normalized"] = scaled
                else:
                    data[f"{col}_normalized"] = 0.0
                    miya_alerts(
                        f"Colonne {col} manquante pour normalisation",
                        tag="VOLUME_PROFILE",
                        voice_profile="warning",
                        priority=3,
                    )
                    send_alert(
                        f"Colonne {col} manquante pour normalisation", priority=3
                    )
                    logger.warning(f"Colonne {col} manquante pour normalisation")
            latency = time.time() - start_time
            self.log_performance("normalize_profiles", latency, success=True)
            return data
        except Exception as e:
            self.log_performance("normalize_profiles", 0, success=False, error=str(e))
            miya_alerts(
                f"Erreur dans normalize_profiles: {str(e)}",
                tag="VOLUME_PROFILE",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans normalize_profiles: {str(e)}", priority=4)
            logger.error(f"Erreur dans normalize_profiles: {str(e)}")
            return data

    def extract_volume_profile(
        self,
        data: pd.DataFrame,
        time_col: str = "timestamp",
        price_col: str = "close",
        volume_col: str = "volume",
        tick_size: float = 0.01,
        vah_quantile: float = 0.75,
        val_quantile: float = 0.25,
    ) -> pd.DataFrame:
        """
        Extrait les informations du profil de volume : POC, VAH, VAL à partir des données IQFeed.

        Args:
            data (pd.DataFrame): Données contenant les prix et volumes (close, volume, timestamp).
            time_col (str, optional): Nom de la colonne temporelle (par défaut : 'timestamp').
            price_col (str, optional): Nom de la colonne de prix (par défaut : 'close').
            volume_col (str, optional): Nom de la colonne de volume (par défaut : 'volume').
            tick_size (float, optional): Taille du tick pour arrondir les prix (par défaut : 0.01).
            vah_quantile (float, optional): Quantile pour Value Area High (par défaut : 0.75).
            val_quantile (float, optional): Quantile pour Value Area Low (par défaut : 0.25).

        Returns:
            pd.DataFrame: Données enrichies avec POC, VAH, VAL et leurs versions normalisées.

        Raises:
            ValueError: Si les données sont vides ou mal formées.
        """
        try:
            start_time = time.time()
            config = self.load_config_with_manager()

            if data.empty:
                error_msg = "DataFrame vide dans extract_volume_profile"
                miya_alerts(
                    error_msg, tag="VOLUME_PROFILE", voice_profile="urgent", priority=5
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                raise ValueError(error_msg)

            if not self.validate_iqfeed_data(data):
                error_msg = "Échec validation données IQFeed"
                miya_alerts(
                    error_msg, tag="VOLUME_PROFILE", voice_profile="urgent", priority=5
                )
                send_alert(error_msg, priority=4)
                raise ValueError(error_msg)

            cache_key = hashlib.sha256(data.to_json().encode()).hexdigest()
            if cache_key in self.cache:
                cached_data = pd.read_csv(
                    self.cache[cache_key]["path"], encoding="utf-8"
                )
                if (
                    datetime.now() - self.cache[cache_key]["timestamp"]
                ).total_seconds() < config.get("cache_hours", 24) * 3600:
                    miya_speak(
                        "Profils récupérés du cache",
                        tag="VOLUME_PROFILE",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert("Profils récupérés du cache", priority=1)
                    self.log_performance(
                        "extract_volume_profile_cache_hit", 0, success=True
                    )
                    return cached_data

            data = data.copy()
            if time_col not in data.columns:
                miya_speak(
                    f"Colonne '{time_col}' manquante, création par défaut",
                    tag="VOLUME_PROFILE",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(f"Colonne '{time_col}' manquante", priority=3)
                logger.warning(f"Colonne '{time_col}' manquante, création par défaut")
                default_start = pd.Timestamp.now()
                data[time_col] = pd.date_range(
                    start=default_start, periods=len(data), freq="1min"
                )
                miya_speak(
                    "Timestamp par défaut ajouté",
                    tag="VOLUME_PROFILE",
                    voice_profile="calm",
                    priority=1,
                )
                send_alert("Timestamp par défaut ajouté", priority=1)

            data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
            if data[time_col].isna().any():
                miya_speak(
                    f"NaN dans '{time_col}', imputés avec la première date valide",
                    tag="VOLUME_PROFILE",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(f"NaN dans '{time_col}'", priority=3)
                logger.warning(f"NaN dans '{time_col}', imputation")
                first_valid_time = (
                    data[time_col].dropna().iloc[0]
                    if not data[time_col].dropna().empty
                    else pd.Timestamp.now()
                )
                data[time_col] = data[time_col].fillna(first_valid_time)

            required_cols = [price_col, volume_col]
            missing_cols = [col for col in required_cols if col not in data.columns]
            for col in missing_cols:
                if col == price_col:
                    data[col] = (
                        data[price_col].median() if price_col in data.columns else 0.0
                    )
                    miya_speak(
                        f"Colonne IQFeed '{col}' manquante, imputée à la médiane ou 0",
                        tag="VOLUME_PROFILE",
                        voice_profile="warning",
                        priority=3,
                    )
                    send_alert(f"Colonne IQFeed '{col}' manquante", priority=3)
                    logger.warning(
                        f"Colonne manquante: {col}, imputée à la médiane ou 0"
                    )
                else:
                    data[col] = 0.0
                    miya_speak(
                        f"Colonne IQFeed '{col}' manquante, imputée à 0",
                        tag="VOLUME_PROFILE",
                        voice_profile="warning",
                        priority=3,
                    )
                    send_alert(f"Colonne IQFeed '{col}' manquante", priority=3)
                    logger.warning(f"Colonne manquante: {col}")

            for col in [price_col, volume_col]:
                data[col] = pd.to_numeric(data[col], errors="coerce")
                if data[col].isna().any():
                    if col == price_col:
                        median_value = data[col].median()
                        data[col] = data[col].fillna(median_value)
                        miya_speak(
                            f"NaN dans {col}, imputés à la médiane ({median_value:.2f})",
                            tag="VOLUME_PROFILE",
                            voice_profile="warning",
                            priority=3,
                        )
                        send_alert(f"NaN dans {col}", priority=3)
                        logger.warning(
                            f"NaN dans {col}, imputés à la médiane ({median_value:.2f})"
                        )
                    else:
                        data[col] = data[col].fillna(0.0)
                        miya_speak(
                            f"NaN dans {col}, imputés à 0",
                            tag="VOLUME_PROFILE",
                            voice_profile="warning",
                            priority=3,
                        )
                        send_alert(f"NaN dans {col}", priority=3)
                        logger.warning(f"NaN dans {col}, imputés à 0")

            if data[price_col].notna().any() and data[volume_col].notna().any():
                price_rounded = (data[price_col] / tick_size).round() * tick_size
                price_volume = data.groupby(price_rounded)[volume_col].sum()
                if not price_volume.empty:
                    data["poc"] = price_volume.idxmax()
                    data["vah"] = data[price_col].quantile(vah_quantile)
                    data["val"] = data[price_col].quantile(val_quantile)
                else:
                    miya_speak(
                        "Profil de volume vide, POC/VAH/VAL imputés à 0",
                        tag="VOLUME_PROFILE",
                        voice_profile="warning",
                        priority=3,
                    )
                    send_alert("Profil de volume vide", priority=3)
                    logger.warning("Profil de volume vide")
                    data["poc"] = 0.0
                    data["vah"] = 0.0
                    data["val"] = 0.0
            else:
                miya_speak(
                    f"Données {price_col} ou {volume_col} invalides, POC/VAH/VAL imputés à 0",
                    tag="VOLUME_PROFILE",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(f"Données {price_col} ou {volume_col} invalides", priority=3)
                logger.warning(f"Données {price_col} ou {volume_col} invalides")
                data["poc"] = 0.0
                data["vah"] = 0.0
                data["val"] = 0.0

            for col in ["poc", "vah", "val"]:
                if data[col].isna().any():
                    data[col] = data[col].fillna(0.0)
                    miya_speak(
                        f"NaN dans {col}, imputés à 0",
                        tag="VOLUME_PROFILE",
                        voice_profile="warning",
                        priority=3,
                    )
                    send_alert(f"NaN dans {col}", priority=3)
                    logger.warning(f"NaN dans {col}, imputés à 0")

            # Normalisation pour l’inférence
            profiles = ["poc", "vah", "val"]
            data = self.normalize_profiles(data, profiles)

            self.validate_shap_features(profiles)
            self.cache_profiles(data, cache_key)
            clusters_df = self.cluster_profiles(data)
            if not clusters_df.empty:
                self.store_clusters(clusters_df)
            self.plot_profiles(data, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            latency = time.time() - start_time
            miya_speak(
                "Profil de volume extrait : poc, vah, val",
                tag="VOLUME_PROFILE",
                voice_profile="calm",
                priority=1,
            )
            send_alert("Profil de volume extrait", priority=1)
            logger.info("Profil de volume extrait")
            self.log_performance(
                "extract_volume_profile", latency, success=True, num_rows=len(data)
            )
            self.save_snapshot(
                "extract_volume_profile", {"num_rows": len(data), "profiles": profiles}
            )
            return data
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans extract_volume_profile: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="VOLUME_PROFILE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=4)
            logger.error(error_msg)
            self.log_performance(
                "extract_volume_profile", latency, success=False, error=str(e)
            )
            self.save_snapshot("extract_volume_profile", {"error": str(e)})
            data["timestamp"] = pd.date_range(
                start=pd.Timestamp.now(), periods=len(data), freq="1min"
            )
            for col in [
                "poc",
                "vah",
                "val",
                "poc_normalized",
                "vah_normalized",
                "val_normalized",
            ]:
                data[col] = 0.0
            return data


if __name__ == "__main__":
    try:
        extractor = VolumeProfileExtractor()
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "close": np.random.normal(5100, 10, 100),
                "volume": np.random.randint(100, 1000, 100),
            }
        )
        result = extractor.extract_volume_profile(data)
        print(result[["timestamp", "poc", "vah", "val", "poc_normalized"]].head())
        miya_speak(
            "Test extract_volume_profile terminé",
            tag="TEST",
            voice_profile="calm",
            priority=1,
        )
        send_alert("Test extract_volume_profile terminé", priority=1)
    except Exception as e:
        miya_alerts(
            f"Erreur test: {str(e)}\n{traceback.format_exc()}",
            tag="ALERT",
            voice_profile="urgent",
            priority=3,
        )
        send_alert(f"Erreur test: {str(e)}", priority=4)
        logger.error(f"Erreur test: {str(e)}")
        raise
