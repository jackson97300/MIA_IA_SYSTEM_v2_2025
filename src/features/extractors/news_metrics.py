# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/features/extractors/news_metrics.py
# Calcule les métriques de sentiment des nouvelles pour MIA_IA_SYSTEM_v2_2025.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Génère des métriques contextuelles basées sur les nouvelles IQFeed (ex. : news_impact_score, news_frequency_1h, news_frequency_1d),
#        intègre SHAP (méthode 17), et enregistre logs psutil. Conforme à la Phase 1 (news_analyzer.py) et Phase 14.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.0,<6.0.0, matplotlib>=3.7.0,<4.0.0, json, gzip, hashlib, logging
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
#
# Inputs :
# - config/es_config.yaml
# - data/iqfeed/news_data.csv
# - data/features/feature_importance.csv
#
# Outputs :
# - data/features/cache/news_metrics/
# - data/logs/news_metrics_performance.csv
# - data/features/news_metrics_snapshots/
# - data/figures/news_metrics/
#
# Notes :
# - Utilise exclusivement IQFeed pour les données (remplace NewsAPI, dxFeed).
# - Suppression de toute référence à dxFeed, obs_t, 320 features, et 81 features.
# - Intègre les Phases 1-18 :
#   - Phase 1 (news_scraper.py, news_analyzer.py) : Métriques news_impact_score, news_frequency_1h, news_frequency_1d.
#   - Phase 14 : Calcul des métriques de sentiment contextuelles.
# - Tests unitaires disponibles dans tests/test_news_metrics.py.
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

from src.model.utils.alert_manager import send_alert
from src.model.utils.config_manager import config_manager
from src.model.utils.miya_console import miya_alerts, miya_speak

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data", "features", "cache", "news_metrics")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "features", "news_metrics_snapshots")
CSV_LOG_PATH = os.path.join(BASE_DIR, "data", "logs", "news_metrics_performance.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures", "news_metrics")
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
    filename=os.path.join(BASE_DIR, "data", "logs", "news_metrics.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 2.0
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes


class NewsMetricsGenerator:

    def __init__(
        self, config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml")
    ):
        self.log_buffer = []
        self.cache = {}
        try:
            self.config = self.load_config_with_manager(config_path)
            self.buffer_size = self.config.get("buffer_size", 100)
            self.max_cache_size = self.config.get("max_cache_size", 1000)
            signal.signal(signal.SIGINT, self.handle_sigint)
            self._clean_cache()
            miya_speak(
                "NewsMetricsGenerator initialisé",
                tag="NEWS_METRICS",
                voice_profile="calm",
                priority=2,
            )
            send_alert("NewsMetricsGenerator initialisé", priority=1)
            logger.info("NewsMetricsGenerator initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config_path": config_path})
        except Exception as e:
            error_msg = f"Erreur initialisation NewsMetricsGenerator: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="NEWS_METRICS", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            self.config = {
                "buffer_size": 100,
                "max_cache_size": 1000,
                "cache_hours": 24,
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
            if "news_metrics_generator" not in config:
                raise ValueError(
                    "Clé 'news_metrics_generator' manquante dans la configuration"
                )
            required_keys = ["buffer_size", "max_cache_size", "cache_hours"]
            missing_keys = [
                key
                for key in required_keys
                if key not in config["news_metrics_generator"]
            ]
            if missing_keys:
                raise ValueError(
                    f"Clés manquantes dans 'news_metrics_generator': {missing_keys}"
                )
            return config["news_metrics_generator"]

        try:
            start_time = time.time()
            config = self.with_retries(load_yaml)
            latency = time.time() - start_time
            miya_speak(
                "Configuration news_metrics_generator chargée via config_manager",
                tag="NEWS_METRICS",
                voice_profile="calm",
                priority=2,
            )
            send_alert(
                "Configuration news_metrics_generator chargée via config_manager",
                priority=1,
            )
            logger.info(
                "Configuration news_metrics_generator chargée via config_manager"
            )
            self.log_performance("load_config_with_manager", latency, success=True)
            self.save_snapshot("load_config_with_manager", {"config_path": config_path})
            return config
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur chargement config via config_manager: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="NEWS_METRICS", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance(
                "load_config_with_manager", latency, success=False, error=str(e)
            )
            return {"buffer_size": 100, "max_cache_size": 1000, "cache_hours": 24}

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
                    tag="NEWS_METRICS",
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
                tag="NEWS_METRICS",
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
                tag="NEWS_METRICS",
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
                tag="NEWS_METRICS",
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
                tag="NEWS_METRICS",
                voice_profile="calm",
                priority=2,
            )
            send_alert("SIGINT received, snapshot saved", priority=2)
            logger.info("SIGINT received, snapshot saved")
        except Exception as e:
            miya_alerts(
                f"Erreur gestion SIGINT: {str(e)}",
                tag="NEWS_METRICS",
                voice_profile="urgent",
                priority=5,
            )
            send_alert(f"Erreur gestion SIGINT: {str(e)}", priority=4)
            logger.error(f"Erreur gestion SIGINT: {str(e)}")
        finally:
            raise SystemExit("Terminated by SIGINT")

    def validate_shap_features(self, features: List[str]) -> bool:
        """
        Valide que les features sont dans le top 150 SHAP.
        """
        try:
            if not os.path.exists(FEATURE_IMPORTANCE_PATH):
                miya_alerts(
                    "Fichier SHAP manquant",
                    tag="NEWS_METRICS",
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
                    tag="NEWS_METRICS",
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
                    tag="NEWS_METRICS",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(
                    f"Features non incluses dans top 150 SHAP: {missing}", priority=3
                )
                logger.warning(f"Features non incluses dans top 150 SHAP: {missing}")
            miya_speak(
                "SHAP features validées",
                tag="NEWS_METRICS",
                voice_profile="calm",
                priority=1,
            )
            send_alert("SHAP features validées", priority=1)
            logger.info("SHAP features validées")
            return True
        except Exception as e:
            miya_alerts(
                f"Erreur validation SHAP features: {str(e)}",
                tag="NEWS_METRICS",
                voice_profile="urgent",
                priority=4,
            )
            send_alert(f"Erreur validation SHAP features: {str(e)}", priority=4)
            logger.error(f"Erreur validation SHAP features: {str(e)}")
            return False

    def compute_news_impact_score(self, news_data: pd.DataFrame) -> pd.Series:
        """
        Calcule le score d'impact des nouvelles basé sur le sentiment (Phase 1, 14).

        Args:
            news_data (pd.DataFrame): Données avec timestamp, sentiment_score, source.

        Returns:
            pd.Series: Score d'impact des nouvelles.
        """
        try:
            start_time = time.time()
            if (
                "sentiment_score" not in news_data.columns
                or "source" not in news_data.columns
            ):
                error_msg = (
                    "Colonnes sentiment_score ou source manquantes dans news_data"
                )
                miya_alerts(
                    error_msg, tag="NEWS_METRICS", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=news_data.index, name="news_impact_score")

            news_data["sentiment_score"] = pd.to_numeric(
                news_data["sentiment_score"], errors="coerce"
            ).fillna(0.0)
            source_weights = {
                "dtn": 1.0,
                "benzinga": 0.9,
                "dowjones": 1.0,
                "unknown": 0.5,
            }
            news_data["source_weight"] = (
                news_data["source"].str.lower().map(source_weights).fillna(0.5)
            )
            result = news_data["sentiment_score"] * news_data["source_weight"]
            result = result.fillna(0.0)
            result.name = "news_impact_score"
            latency = time.time() - start_time
            self.log_performance("compute_news_impact_score", latency, success=True)
            return result
        except Exception as e:
            self.log_performance(
                "compute_news_impact_score", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_news_impact_score: {str(e)}",
                tag="NEWS_METRICS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_news_impact_score: {str(e)}", priority=4)
            logger.error(f"Erreur dans compute_news_impact_score: {str(e)}")
            return pd.Series(0.0, index=news_data.index, name="news_impact_score")

    def compute_news_frequency(self, news_data: pd.DataFrame, window: str) -> pd.Series:
        """
        Calcule la fréquence des nouvelles dans une fenêtre temporelle (Phase 1).

        Args:
            news_data (pd.DataFrame): Données avec timestamp.
            window (str): Fenêtre temporelle (ex. : '1h', '1d').

        Returns:
            pd.Series: Fréquence des nouvelles.
        """
        try:
            start_time = time.time()
            if "timestamp" not in news_data.columns:
                error_msg = "Colonne timestamp manquante dans news_data"
                miya_alerts(
                    error_msg, tag="NEWS_METRICS", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(
                    0.0, index=news_data.index, name=f"news_frequency_{window}"
                )

            news_data["timestamp"] = pd.to_datetime(
                news_data["timestamp"], errors="coerce"
            )
            frequency = news_data.groupby(
                pd.Grouper(key="timestamp", freq=window)
            ).size()
            result = frequency.reindex(news_data.index, method="ffill").fillna(0.0)
            result.name = f"news_frequency_{window}"
            latency = time.time() - start_time
            self.log_performance(
                f"compute_news_frequency_{window}", latency, success=True
            )
            return result
        except Exception as e:
            self.log_performance(
                f"compute_news_frequency_{window}", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_news_frequency_{window}: {str(e)}",
                tag="NEWS_METRICS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_news_frequency_{window}: {str(e)}", priority=4
            )
            logger.error(f"Erreur dans compute_news_frequency_{window}: {str(e)}")
            return pd.Series(
                0.0, index=news_data.index, name=f"news_frequency_{window}"
            )

    def compute_news_metrics(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les métriques de sentiment des nouvelles à partir des données IQFeed (Phase 1, 14).

        Args:
            news_data (pd.DataFrame): Données avec timestamp, sentiment_score, source.

        Returns:
            pd.DataFrame: Données enrichies avec news_impact_score, news_frequency_1h, news_frequency_1d.
        """
        try:
            start_time = time.time()
            config = self.load_config_with_manager()

            if news_data.empty:
                error_msg = "DataFrame news_data vide"
                miya_alerts(
                    error_msg, tag="NEWS_METRICS", voice_profile="urgent", priority=5
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                raise ValueError(error_msg)

            cache_key = hashlib.sha256(news_data.to_json().encode()).hexdigest()
            if cache_key in self.cache:
                cached_data = pd.read_csv(
                    self.cache[cache_key]["path"], encoding="utf-8"
                )
                if (
                    datetime.now() - self.cache[cache_key]["timestamp"]
                ).total_seconds() < config.get("cache_hours", 24) * 3600:
                    miya_speak(
                        "Métriques de nouvelles récupérées du cache",
                        tag="NEWS_METRICS",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert("Métriques de nouvelles récupérées du cache", priority=1)
                    self.log_performance(
                        "compute_news_metrics_cache_hit", 0, success=True
                    )
                    return cached_data

            news_data = news_data.copy()
            if "timestamp" not in news_data.columns:
                miya_speak(
                    "Colonne timestamp manquante, création par défaut",
                    tag="NEWS_METRICS",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert("Colonne timestamp manquante", priority=3)
                logger.warning("Colonne timestamp manquante, création par défaut")
                default_start = pd.Timestamp.now()
                news_data["timestamp"] = pd.date_range(
                    start=default_start, periods=len(news_data), freq="1min"
                )

            news_data["timestamp"] = pd.to_datetime(
                news_data["timestamp"], errors="coerce"
            )
            if news_data["timestamp"].isna().any():
                miya_speak(
                    "NaN dans timestamp, imputés avec la première date valide",
                    tag="NEWS_METRICS",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert("NaN dans timestamp", priority=3)
                logger.warning("NaN dans timestamp, imputation")
                first_valid_time = (
                    news_data["timestamp"].dropna().iloc[0]
                    if not news_data["timestamp"].dropna().empty
                    else pd.Timestamp.now()
                )
                news_data["timestamp"] = news_data["timestamp"].fillna(first_valid_time)

            # Calcul des métriques
            news_data["news_impact_score"] = self.compute_news_impact_score(news_data)
            news_data["news_frequency_1h"] = self.compute_news_frequency(
                news_data, "1h"
            )
            news_data["news_frequency_1d"] = self.compute_news_frequency(
                news_data, "1d"
            )

            metrics = ["news_impact_score", "news_frequency_1h", "news_frequency_1d"]
            self.validate_shap_features(metrics)
            self.cache_metrics(news_data, cache_key)
            self.plot_metrics(news_data, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            latency = time.time() - start_time
            miya_speak(
                "Métriques de nouvelles calculées",
                tag="NEWS_METRICS",
                voice_profile="calm",
                priority=1,
            )
            send_alert("Métriques de nouvelles calculées", priority=1)
            logger.info("Métriques de nouvelles calculées")
            self.log_performance(
                "compute_news_metrics",
                latency,
                success=True,
                num_rows=len(news_data),
                num_metrics=len(metrics),
            )
            self.save_snapshot(
                "compute_news_metrics", {"num_rows": len(news_data), "metrics": metrics}
            )
            return news_data
        except Exception as e:
            latency = time.time() - start_time
            error_msg = (
                f"Erreur dans compute_news_metrics: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, tag="NEWS_METRICS", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=4)
            logger.error(error_msg)
            self.log_performance(
                "compute_news_metrics", latency, success=False, error=str(e)
            )
            self.save_snapshot("compute_news_metrics", {"error": str(e)})
            news_data["news_impact_score"] = 0.0
            news_data["news_frequency_1h"] = 0.0
            news_data["news_frequency_1d"] = 0.0
            return news_data

    def cache_metrics(self, metrics: pd.DataFrame, cache_key: str) -> None:
        """
        Met en cache les métriques de nouvelles.
        """

        def save_cache():
            cache_path = os.path.join(CACHE_DIR, f"{cache_key}.csv")
            os.makedirs(CACHE_DIR, exist_ok=True)
            metrics.to_csv(cache_path, index=False, encoding="utf-8")
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
                f"Métriques mises en cache: {path}",
                tag="NEWS_METRICS",
                voice_profile="calm",
                priority=1,
            )
            send_alert(f"Métriques mises en cache: {path}", priority=1)
            logger.info(f"Métriques mises en cache: {path}")
            self.log_performance(
                "cache_metrics", latency, success=True, cache_size=len(self.cache)
            )
        except Exception as e:
            self.log_performance("cache_metrics", 0, success=False, error=str(e))
            miya_alerts(
                f"Erreur mise en cache métriques: {str(e)}",
                tag="NEWS_METRICS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur mise en cache métriques: {str(e)}", priority=4)
            logger.error(f"Erreur mise en cache métriques: {str(e)}")

    def plot_metrics(self, metrics: pd.DataFrame, timestamp: str) -> None:
        """
        Génère des visualisations des métriques de nouvelles.

        Args:
            metrics (pd.DataFrame): Données avec les métriques.
            timestamp (str): Horodatage pour nommer le fichier.
        """
        try:
            start_time = time.time()
            timestamp_safe = timestamp.replace(":", "-")
            plt.figure(figsize=(12, 6))
            plt.plot(
                metrics["timestamp"],
                metrics["news_impact_score"],
                label="News Impact Score",
                color="blue",
            )
            plt.plot(
                metrics["timestamp"],
                metrics["news_frequency_1h"],
                label="News Frequency 1h",
                color="orange",
            )
            plt.plot(
                metrics["timestamp"],
                metrics["news_frequency_1d"],
                label="News Frequency 1d",
                color="green",
            )
            plt.title(f"News Metrics - {timestamp}")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(FIGURES_DIR, f"news_metrics_{timestamp_safe}.png"))
            plt.close()
            latency = time.time() - start_time
            miya_speak(
                f"Visualisations générées: {FIGURES_DIR}",
                tag="NEWS_METRICS",
                voice_profile="calm",
                priority=2,
            )
            send_alert(f"Visualisations générées: {FIGURES_DIR}", priority=2)
            logger.info(f"Visualisations générées: {FIGURES_DIR}")
            self.log_performance("plot_metrics", latency, success=True)
        except Exception as e:
            self.log_performance("plot_metrics", 0, success=False, error=str(e))
            miya_alerts(
                f"Erreur génération visualisations: {str(e)}",
                tag="NEWS_METRICS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur génération visualisations: {str(e)}", priority=4)
            logger.error(f"Erreur génération visualisations: {str(e)}")


if __name__ == "__main__":
    try:
        generator = NewsMetricsGenerator()
        news_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "sentiment_score": np.random.uniform(-1, 1, 100),
                "source": np.random.choice(
                    ["DTN", "Benzinga", "DowJones", "unknown"], 100
                ),
            }
        )
        result = generator.compute_news_metrics(news_data)
        print(
            result[
                [
                    "timestamp",
                    "news_impact_score",
                    "news_frequency_1h",
                    "news_frequency_1d",
                ]
            ].head()
        )
        miya_speak(
            "Test compute_news_metrics terminé",
            tag="TEST",
            voice_profile="calm",
            priority=1,
        )
        send_alert("Test compute_news_metrics terminé", priority=1)
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
