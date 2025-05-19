# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/features/extractors/smart_scores.py
# Calcule les scores intelligents (breakout_score, mm_score, hft_score) à partir des données IQFeed.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Génère 3 scores intelligents avec pondération selon le régime de marché (méthode 3),
#        utilise un cache intermédiaire, enregistre des logs psutil, valide SHAP (méthode 17),
#        normalise les scores pour l’inférence, et utilise exclusivement IQFeed comme source de données.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.0,<6.0.0, scikit-learn>=1.5.0,<2.0.0, matplotlib>=3.7.0,<4.0.0, json, gzip, hashlib, logging
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
#
# Inputs :
# - config/es_config.yaml
# - data/iqfeed/merged_data.csv
# - data/features/feature_importance.csv
#
# Outputs :
# - data/features/cache/smart_scores/
# - data/logs/smart_scores_performance.csv
# - data/features/smart_scores_snapshots/
# - data/figures/smart_scores/
#
# Notes :
# - Utilise exclusivement IQFeed pour les données (remplace dxFeed).
# - Suppression de toute référence à dxFeed, obs_t, 320 features, et 81 features.
# - Intègre les Phases 1-18 (non explicitement lié à une phase spécifique).
# - Tests unitaires disponibles dans tests/test_smart_scores.py.
# - Pondération des scores selon le régime (trend: 1.5 pour mm_score/hft_score, range: 1.5 pour breakout_score, defensive: 0.5).
# Évolution future : Intégration avec feature_pipeline.py pour top 150
# SHAP, migration API Investing.com (juin 2025).

import gzip
import hashlib
import json
import logging
import os
import signal
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from sklearn.preprocessing import MinMaxScaler

from src.model.utils.alert_manager import send_alert
from src.model.utils.config_manager import config_manager
from src.model.utils.miya_console import miya_alerts, miya_speak

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data", "features", "cache", "smart_scores")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "features", "smart_scores_snapshots")
CSV_LOG_PATH = os.path.join(BASE_DIR, "data", "logs", "smart_scores_performance.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures", "smart_scores")
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
    filename=os.path.join(BASE_DIR, "data", "logs", "smart_scores.log"),
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


class SmartScoresCalculator:

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
                "SmartScoresCalculator initialisé",
                tag="SMART_SCORES",
                voice_profile="calm",
                priority=2,
            )
            send_alert("SmartScoresCalculator initialisé", priority=1)
            logger.info("SmartScoresCalculator initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config_path": config_path})
        except Exception as e:
            error_msg = f"Erreur initialisation SmartScoresCalculator: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="SMART_SCORES", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            self.config = {
                "buffer_size": 100,
                "max_cache_size": 1000,
                "cache_hours": 24,
                "weights": {
                    "trend": {"mm_score": 1.5, "hft_score": 1.5, "breakout_score": 1.0},
                    "range": {"breakout_score": 1.5, "mm_score": 1.0, "hft_score": 1.0},
                    "defensive": {
                        "mm_score": 0.5,
                        "hft_score": 0.5,
                        "breakout_score": 0.5,
                    },
                },
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
            if "smart_scores_calculator" not in config:
                raise ValueError(
                    "Clé 'smart_scores_calculator' manquante dans la configuration"
                )
            required_keys = ["buffer_size", "max_cache_size", "cache_hours", "weights"]
            missing_keys = [
                key
                for key in required_keys
                if key not in config["smart_scores_calculator"]
            ]
            if missing_keys:
                raise ValueError(
                    f"Clés manquantes dans 'smart_scores_calculator': {missing_keys}"
                )
            return config["smart_scores_calculator"]

        try:
            start_time = time.time()
            config = self.with_retries(load_yaml)
            latency = time.time() - start_time
            miya_speak(
                "Configuration smart_scores_calculator chargée via config_manager",
                tag="SMART_SCORES",
                voice_profile="calm",
                priority=2,
            )
            send_alert(
                "Configuration smart_scores_calculator chargée via config_manager",
                priority=1,
            )
            logger.info(
                "Configuration smart_scores_calculator chargée via config_manager"
            )
            self.log_performance("load_config_with_manager", latency, success=True)
            self.save_snapshot("load_config_with_manager", {"config_path": config_path})
            return config
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur chargement config via config_manager: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="SMART_SCORES", voice_profile="urgent", priority=3
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
                "weights": {
                    "trend": {"mm_score": 1.5, "hft_score": 1.5, "breakout_score": 1.0},
                    "range": {"breakout_score": 1.5, "mm_score": 1.0, "hft_score": 1.0},
                    "defensive": {
                        "mm_score": 0.5,
                        "hft_score": 0.5,
                        "breakout_score": 0.5,
                    },
                },
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
                    tag="SMART_SCORES",
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
                tag="SMART_SCORES",
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
                tag="SMART_SCORES",
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
                tag="SMART_SCORES",
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
                tag="SMART_SCORES",
                voice_profile="calm",
                priority=2,
            )
            send_alert("SIGINT received, snapshot saved", priority=2)
            logger.info("SIGINT received, snapshot saved")
        except Exception as e:
            miya_alerts(
                f"Erreur gestion SIGINT: {str(e)}",
                tag="SMART_SCORES",
                voice_profile="urgent",
                priority=5,
            )
            send_alert(f"Erreur gestion SIGINT: {str(e)}", priority=4)
            logger.error(f"Erreur gestion SIGINT: {str(e)}")
        finally:
            raise SystemExit("Terminated by SIGINT")

    def validate_iqfeed_data(self, data: pd.DataFrame) -> bool:
        """
        Valide les données IQFeed pour les scores intelligents.
        """
        try:
            required_cols = [
                "timestamp",
                "atr_14",
                "volume",
                "absorption_strength",
                "delta_volume",
            ]
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                error_msg = f"Colonnes IQFeed manquantes: {missing_cols}"
                miya_alerts(
                    error_msg, tag="SMART_SCORES", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return False
            if data[required_cols].isna().any().any():
                error_msg = "Valeurs NaN dans les données IQFeed"
                miya_alerts(
                    error_msg, tag="SMART_SCORES", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return False
            if not np.isfinite(
                data[required_cols].select_dtypes(include=[np.number]).values
            ).all():
                error_msg = "Valeurs infinies dans les données IQFeed"
                miya_alerts(
                    error_msg, tag="SMART_SCORES", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return False
            miya_speak(
                "Données IQFeed validées",
                tag="SMART_SCORES",
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
                error_msg, tag="SMART_SCORES", voice_profile="urgent", priority=4
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
                    tag="SMART_SCORES",
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
                    tag="SMART_SCORES",
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
                    tag="SMART_SCORES",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(
                    f"Features non incluses dans top 150 SHAP: {missing}", priority=3
                )
                logger.warning(f"Features non incluses dans top 150 SHAP: {missing}")
            miya_speak(
                "SHAP features validées",
                tag="SMART_SCORES",
                voice_profile="calm",
                priority=1,
            )
            send_alert("SHAP features validées", priority=1)
            logger.info("SHAP features validées")
            return True
        except Exception as e:
            miya_alerts(
                f"Erreur validation SHAP features: {str(e)}",
                tag="SMART_SCORES",
                voice_profile="urgent",
                priority=4,
            )
            send_alert(f"Erreur validation SHAP features: {str(e)}", priority=4)
            logger.error(f"Erreur validation SHAP features: {str(e)}")
            return False

    def cache_scores(self, scores: pd.DataFrame, cache_key: str) -> None:
        """
        Met en cache les scores intelligents.
        """

        def save_cache():
            cache_path = os.path.join(CACHE_DIR, f"{cache_key}.csv")
            os.makedirs(CACHE_DIR, exist_ok=True)
            scores.to_csv(cache_path, index=False, encoding="utf-8")
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
                f"Scores mis en cache: {path}",
                tag="SMART_SCORES",
                voice_profile="calm",
                priority=1,
            )
            send_alert(f"Scores mis en cache: {path}", priority=1)
            logger.info(f"Scores mis en cache: {path}")
            self.log_performance(
                "cache_scores", latency, success=True, cache_size=len(self.cache)
            )
        except Exception as e:
            self.log_performance("cache_scores", 0, success=False, error=str(e))
            miya_alerts(
                f"Erreur mise en cache scores: {str(e)}",
                tag="SMART_SCORES",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur mise en cache scores: {str(e)}", priority=4)
            logger.error(f"Erreur mise en cache scores: {str(e)}")

    def plot_scores(self, scores: pd.DataFrame, timestamp: str) -> None:
        """
        Génère des visualisations des scores intelligents.
        """
        try:
            start_time = time.time()
            timestamp_safe = timestamp.replace(":", "-")
            plt.figure(figsize=(12, 6))
            plt.plot(
                scores["timestamp"],
                scores["breakout_score"],
                label="Breakout Score",
                color="blue",
            )
            plt.plot(
                scores["timestamp"],
                scores["mm_score"],
                label="MM Score",
                color="orange",
            )
            plt.plot(
                scores["timestamp"],
                scores["hft_score"],
                label="HFT Score",
                color="green",
            )
            plt.title(f"Smart Scores - {timestamp}")
            plt.xlabel("Time")
            plt.ylabel("Score")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(FIGURES_DIR, f"smart_scores_{timestamp_safe}.png"))
            plt.close()
            latency = time.time() - start_time
            miya_speak(
                f"Visualisations générées: {FIGURES_DIR}",
                tag="SMART_SCORES",
                voice_profile="calm",
                priority=2,
            )
            send_alert(f"Visualisations générées: {FIGURES_DIR}", priority=2)
            logger.info(f"Visualisations générées: {FIGURES_DIR}")
            self.log_performance("plot_scores", latency, success=True)
        except Exception as e:
            self.log_performance("plot_scores", 0, success=False, error=str(e))
            miya_alerts(
                f"Erreur génération visualisations: {str(e)}",
                tag="SMART_SCORES",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur génération visualisations: {str(e)}", priority=4)
            logger.error(f"Erreur génération visualisations: {str(e)}")

    def normalize_scores(self, data: pd.DataFrame, scores: List[str]) -> pd.DataFrame:
        """
        Normalise les scores avec MinMaxScaler sur une fenêtre glissante pour l’inférence.
        """
        try:
            start_time = time.time()
            for col in scores:
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
                        tag="SMART_SCORES",
                        voice_profile="warning",
                        priority=3,
                    )
                    send_alert(
                        f"Colonne {col} manquante pour normalisation", priority=3
                    )
                    logger.warning(f"Colonne {col} manquante pour normalisation")
            latency = time.time() - start_time
            self.log_performance("normalize_scores", latency, success=True)
            return data
        except Exception as e:
            self.log_performance("normalize_scores", 0, success=False, error=str(e))
            miya_alerts(
                f"Erreur dans normalize_scores: {str(e)}",
                tag="SMART_SCORES",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans normalize_scores: {str(e)}", priority=4)
            logger.error(f"Erreur dans normalize_scores: {str(e)}")
            return data

    def calculate_smart_scores(
        self,
        data: pd.DataFrame,
        time_col: str = "timestamp",
        breakout_window: int = 20,
        mm_window: int = 10,
        hft_window: int = 5,
        regime: str = "range",
    ) -> pd.DataFrame:
        """
        Calcule Breakout Score, Market Maker Score, HFT Score à partir des données IQFeed,
        avec pondération selon le régime de marché.

        Args:
            data (pd.DataFrame): Données contenant les features nécessaires (atr_14, volume, absorption_strength, delta_volume).
            time_col (str, optional): Nom de la colonne temporelle (par défaut : 'timestamp').
            breakout_window (int, optional): Fenêtre pour la moyenne mobile du breakout_score (par défaut : 20).
            mm_window (int, optional): Fenêtre pour la moyenne mobile du mm_score (par défaut : 10).
            hft_window (int, optional): Fenêtre pour la moyenne mobile du hft_score (par défaut : 5).
            regime (str, optional): Régime de marché (trend, range, defensive; défaut : 'range').

        Returns:
            pd.DataFrame: Données enrichies avec les scores intelligents, pondérés et normalisés.

        Raises:
            ValueError: Si les données sont vides, mal formées, ou si le régime est invalide.
        """
        try:
            start_time = time.time()
            config = self.load_config_with_manager()
            weights = config.get("weights", {}).get(
                regime, {"breakout_score": 1.0, "mm_score": 1.0, "hft_score": 1.0}
            )

            if data.empty:
                error_msg = "DataFrame vide dans calculate_smart_scores"
                miya_alerts(
                    error_msg, tag="SMART_SCORES", voice_profile="urgent", priority=5
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                raise ValueError(error_msg)

            if not self.validate_iqfeed_data(data):
                error_msg = "Échec validation données IQFeed"
                miya_alerts(
                    error_msg, tag="SMART_SCORES", voice_profile="urgent", priority=5
                )
                send_alert(error_msg, priority=4)
                raise ValueError(error_msg)

            valid_regimes = ["trend", "range", "defensive"]
            if regime not in valid_regimes:
                error_msg = f"Régime invalide: {regime}, attendu: {valid_regimes}"
                miya_alerts(
                    error_msg, tag="SMART_SCORES", voice_profile="urgent", priority=5
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                raise ValueError(error_msg)

            cache_key = hashlib.sha256(
                f"{data.to_json()}_{regime}".encode()
            ).hexdigest()
            if cache_key in self.cache:
                cached_data = pd.read_csv(
                    self.cache[cache_key]["path"], encoding="utf-8"
                )
                if (
                    datetime.now() - self.cache[cache_key]["timestamp"]
                ).total_seconds() < config.get("cache_hours", 24) * 3600:
                    miya_speak(
                        "Scores récupérés du cache",
                        tag="SMART_SCORES",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert("Scores récupérés du cache", priority=1)
                    self.log_performance(
                        "calculate_smart_scores_cache_hit", 0, success=True
                    )
                    return cached_data

            data = data.copy()
            if time_col not in data.columns:
                miya_speak(
                    f"Colonne '{time_col}' manquante, création par défaut",
                    tag="SMART_SCORES",
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
                    tag="SMART_SCORES",
                    voice_profile="calm",
                    priority=1,
                )
                send_alert("Timestamp par défaut ajouté", priority=1)

            data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
            if data[time_col].isna().any():
                miya_speak(
                    f"NaN dans '{time_col}', imputés avec la première date valide",
                    tag="SMART_SCORES",
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

            required_cols = ["atr_14", "volume", "absorption_strength", "delta_volume"]
            missing_cols = [col for col in required_cols if col not in data.columns]
            for col in missing_cols:
                data[col] = pd.Series(0.0, index=data.index)
                miya_speak(
                    f"Colonne '{col}' manquante, imputée comme Series à 0",
                    tag="SMART_SCORES",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(f"Colonne '{col}' manquante", priority=3)
                logger.warning(f"Colonne manquante: {col}")

            for col in required_cols:
                data[col] = pd.to_numeric(data[col], errors="coerce")
                if data[col].isna().any():
                    if col in ["atr_14", "absorption_strength", "delta_volume"]:
                        median_value = data[col].median()
                        data[col] = data[col].fillna(median_value)
                        miya_speak(
                            f"NaN dans {col}, imputés à la médiane ({median_value:.2f})",
                            tag="SMART_SCORES",
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
                            tag="SMART_SCORES",
                            voice_profile="warning",
                            priority=3,
                        )
                        send_alert(f"NaN dans {col}", priority=3)
                        logger.warning(f"NaN dans {col}, imputés à 0")

            atr_normalized = (
                data["atr_14"]
                / data["atr_14"].rolling(breakout_window, min_periods=1).mean()
            )
            volume_normalized = (
                data["volume"]
                / data["volume"].rolling(breakout_window, min_periods=1).mean()
            )
            data["breakout_score"] = atr_normalized * volume_normalized
            data["breakout_score"] = data["breakout_score"].fillna(0.0)

            data["mm_score"] = (
                data["absorption_strength"]
                / data["absorption_strength"].rolling(mm_window, min_periods=1).mean()
            )
            data["mm_score"] = data["mm_score"].fillna(0.0)

            data["hft_score"] = (
                data["delta_volume"]
                .diff()
                .abs()
                .rolling(hft_window, min_periods=1)
                .mean()
            )
            data["hft_score"] = data["hft_score"].fillna(0.0)

            # Normalisation globale
            scaler = MinMaxScaler()
            for col in ["breakout_score", "mm_score", "hft_score"]:
                data[col] = scaler.fit_transform(data[[col]].values)[:, 0]
                data[f"{col}_weighted"] = data[col] * weights.get(col, 1.0)
                if data[f"{col}_weighted"].isna().any():
                    data[f"{col}_weighted"] = data[f"{col}_weighted"].fillna(0.0)
                    miya_speak(
                        f"NaN dans {col}_weighted, imputés à 0",
                        tag="SMART_SCORES",
                        voice_profile="warning",
                        priority=3,
                    )
                    send_alert(f"NaN dans {col}_weighted", priority=3)
                    logger.warning(f"NaN dans {col}_weighted, imputés à 0")

            # Normalisation par fenêtre glissante pour l’inférence
            scores = ["breakout_score", "mm_score", "hft_score"]
            data = self.normalize_scores(data, scores)

            self.validate_shap_features(scores)
            self.cache_scores(data, cache_key)
            self.plot_scores(data, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            latency = time.time() - start_time
            miya_speak(
                "Scores intelligents calculés : breakout_score, mm_score, hft_score",
                tag="SMART_SCORES",
                voice_profile="calm",
                priority=1,
            )
            send_alert("Scores intelligents calculés", priority=1)
            logger.info("Scores intelligents calculés")
            self.log_performance(
                "calculate_smart_scores",
                latency,
                success=True,
                num_rows=len(data),
                regime=regime,
            )
            self.save_snapshot(
                "calculate_smart_scores",
                {"num_rows": len(data), "scores": scores, "regime": regime},
            )
            return data
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans calculate_smart_scores: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="SMART_SCORES", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=4)
            logger.error(error_msg)
            self.log_performance(
                "calculate_smart_scores", latency, success=False, error=str(e)
            )
            self.save_snapshot("calculate_smart_scores", {"error": str(e)})
            data["timestamp"] = pd.date_range(
                start=pd.Timestamp.now(), periods=len(data), freq="1min"
            )
            for col in [
                "breakout_score",
                "mm_score",
                "hft_score",
                "breakout_score_weighted",
                "mm_score_weighted",
                "hft_score_weighted",
                "breakout_score_normalized",
                "mm_score_normalized",
                "hft_score_normalized",
            ]:
                data[col] = 0.0
            return data


if __name__ == "__main__":
    try:
        calculator = SmartScoresCalculator()
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "atr_14": np.random.uniform(0.5, 2.0, 100),
                "volume": np.random.randint(100, 1000, 100),
                "absorption_strength": np.random.uniform(0, 100, 100),
                "delta_volume": np.random.uniform(-500, 500, 100),
                "close": np.random.normal(5100, 10, 100),
            }
        )
        result = calculator.calculate_smart_scores(data, regime="trend")
        print(
            result[
                [
                    "timestamp",
                    "breakout_score",
                    "mm_score",
                    "hft_score",
                    "mm_score_weighted",
                    "breakout_score_normalized",
                ]
            ].head()
        )
        miya_speak(
            "Test calculate_smart_scores terminé",
            tag="TEST",
            voice_profile="calm",
            priority=1,
        )
        send_alert("Test calculate_smart_scores terminé", priority=1)
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
