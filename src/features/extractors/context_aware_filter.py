# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/api/context_aware_filter.py
# Gère les métriques contextuelles liées aux nouvelles, événements macro, et cyclicité temporelle.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Calcule 22 métriques contextuelles (16 nouvelles + 3 existantes comme event_volatility_impact + 3 nouvelles pour Phases 13, 15, 18),
#        utilise IQFeed pour les nouvelles et futures, intègre SHAP (méthode 17), et enregistre logs psutil.
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
# - data/iqfeed/futures_data.csv
# - data/events/macro_events.csv
# - data/events/event_volatility_history.csv
# - data/features/feature_importance.csv
#
# Outputs :
# - data/features/cache/context_aware/
# - data/logs/context_aware_performance.csv
# - data/features/context_aware_snapshots/
# - data/figures/context_aware/
#
# Notes :
# - Utilise exclusivement IQFeed pour les données (remplace dxFeed).
# - Suppression de toute référence à dxFeed, obs_t, 320 features, et 81 features pour conformité avec MIA_IA_SYSTEM_v2_2025.
# - Intègre les Phases 1-18 :
#   - Phase 1 (news_scraper.py, news_analyzer.py) : Métriques comme news_sentiment_momentum, news_sentiment_acceleration.
#   - Phase 13 (orderflow_indicators.py, options_metrics.py) : Nouvelle métrique option_skew_impact.
#   - Phase 15 (microstructure_guard.py, spotgamma_recalculator.py) : Nouvelle métrique microstructure_event_impact.
#   - Phase 18 : Nouvelles métriques hft_news_reaction, trade_velocity_impact.
# - Tests unitaires disponibles dans tests/test_context_aware_filter.py.
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
from datetime import datetime, timedelta
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
CACHE_DIR = os.path.join(BASE_DIR, "data", "features", "cache", "context_aware")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "features", "context_aware_snapshots")
CSV_LOG_PATH = os.path.join(BASE_DIR, "data", "logs", "context_aware_performance.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures", "context_aware")
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
    filename=os.path.join(BASE_DIR, "data", "logs", "context_aware.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 2.0
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes


class ContextAwareFilter:

    def __init__(
        self, config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml")
    ):
        self.log_buffer = []
        self.cache = {}
        try:
            self.config = self.load_config_with_manager_new(config_path)
            self.buffer_size = self.config.get("buffer_size", 100)
            self.max_cache_size = self.config.get("max_cache_size", 1000)
            signal.signal(signal.SIGINT, self.handle_sigint)
            self._clean_cache()
            miya_speak(
                "ContextAwareFilter initialisé",
                tag="CONTEXT_AWARE",
                voice_profile="calm",
                priority=2,
            )
            send_alert("ContextAwareFilter initialisé", priority=1)
            logger.info("ContextAwareFilter initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config_path": config_path})
        except Exception as e:
            error_msg = f"Erreur initialisation ContextAwareFilter: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=3
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
            self.alert_manager.send_alert(
                f"Erreur nettoyage cache: {str(e)}", priority=3
            )
            logger.error(f"Erreur nettoyage cache: {str(e)}")
            self.log_performance(
                "clean_cache", latency, success=False, error=str(e), data_type="cache"
            )

    def load_config_with_manager_new(self, config_path: str) -> Dict[str, Any]:
        def load_yaml():
            config = config_manager.get_config(os.path.basename(config_path))
            if "context_aware_filter" not in config:
                raise ValueError(
                    "Clé 'context_aware_filter' manquante dans la configuration"
                )
            required_keys = ["buffer_size", "max_cache_size", "cache_hours"]
            missing_keys = [
                key
                for key in required_keys
                if key not in config["context_aware_filter"]
            ]
            if missing_keys:
                raise ValueError(
                    f"Clés manquantes dans 'context_aware_filter': {missing_keys}"
                )
            return config["context_aware_filter"]

        try:
            start_time = time.time()
            config = self.with_retries(load_yaml)
            latency = time.time() - start_time
            miya_speak(
                "Configuration context_aware_filter chargée via config_manager",
                tag="CONTEXT_AWARE",
                voice_profile="calm",
                priority=2,
            )
            send_alert(
                "Configuration context_aware_filter chargée via config_manager",
                priority=1,
            )
            logger.info("Configuration context_aware_filter chargée via config_manager")
            self.log_performance("load_config_with_manager_new", latency, success=True)
            self.save_snapshot(
                "load_config_with_manager_new", {"config_path": config_path}
            )
            return config
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur chargement config via config_manager: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance(
                "load_config_with_manager_new", latency, success=False, error=str(e)
            )
            return {"buffer_size": 100, "max_cache_size": 1000, "cache_hours": 24}

    def with_retries(
        self,
        func: callable,
        max_attempts: int = MAX_RETRIES,
        delay_base: float = RETRY_DELAY,
    ) -> Any:
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
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_usage = psutil.cpu_percent()
            if memory_usage > 1024:
                miya_alerts(
                    f"ALERT: High memory usage ({memory_usage:.2f} MB)",
                    tag="CONTEXT_AWARE",
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
                tag="CONTEXT_AWARE",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur logging performance: {str(e)}", priority=4)
            logger.error(f"Erreur logging performance: {str(e)}")

    def save_snapshot(self, snapshot_type: str, data: Dict) -> None:
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
                tag="CONTEXT_AWARE",
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
                tag="CONTEXT_AWARE",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}", priority=4
            )
            logger.error(f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}")

    def handle_sigint(self, signal: int, frame: Any) -> None:
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
                tag="CONTEXT_AWARE",
                voice_profile="calm",
                priority=2,
            )
            send_alert("SIGINT received, snapshot saved", priority=2)
            logger.info("SIGINT received, snapshot saved")
        except Exception as e:
            miya_alerts(
                f"Erreur gestion SIGINT: {str(e)}",
                tag="CONTEXT_AWARE",
                voice_profile="urgent",
                priority=5,
            )
            send_alert(f"Erreur gestion SIGINT: {str(e)}", priority=4)
            logger.error(f"Erreur gestion SIGINT: {str(e)}")
        finally:
            raise SystemExit("Terminated by SIGINT")

    def parse_news_data(self, news_data: pd.DataFrame) -> pd.DataFrame:
        try:
            start_time = time.time()
            if (
                "sentiment_score" not in news_data.columns
                or "timestamp" not in news_data.columns
            ):
                error_msg = (
                    "Colonnes sentiment_score ou timestamp manquantes dans news_data"
                )
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.DataFrame()
            news_data["timestamp"] = pd.to_datetime(news_data["timestamp"])
            news_data["sentiment_score"] = pd.to_numeric(
                news_data["sentiment_score"], errors="coerce"
            ).fillna(0.0)
            news_data["volume"] = news_data.get(
                "volume", pd.Series(1, index=news_data.index)
            )
            latency = time.time() - start_time
            self.log_performance(
                "parse_news_data", latency, success=True, num_rows=len(news_data)
            )
            return news_data
        except Exception as e:
            self.log_performance("parse_news_data", 0, success=False, error=str(e))
            miya_alerts(
                f"Erreur dans parse_news_data: {str(e)}",
                tag="CONTEXT_AWARE",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans parse_news_data: {str(e)}", priority=4)
            logger.error(f"Erreur dans parse_news_data: {str(e)}")
            return pd.DataFrame()

    def calculate_event_volatility_impact(
        self,
        events: pd.DataFrame,
        volatility_history: pd.DataFrame,
        timestamp: pd.Timestamp,
    ) -> float:
        try:
            start_time = time.time()
            events["time_diff"] = np.abs(
                (events["timestamp"] - timestamp).dt.total_seconds()
            )
            nearest_event = events.loc[events["time_diff"].idxmin()]
            event_type = nearest_event["event_type"]
            impact_score = nearest_event["event_impact_score"]
            historical_impact = volatility_history[
                volatility_history["event_type"] == event_type
            ]["volatility_impact"].mean()
            if pd.isna(historical_impact):
                historical_impact = 0.0
            volatility_impact = historical_impact * impact_score
            result = volatility_impact if not np.isnan(volatility_impact) else 0.0
            latency = time.time() - start_time
            self.log_performance(
                "calculate_event_volatility_impact", latency, success=True
            )
            return result
        except Exception as e:
            self.log_performance(
                "calculate_event_volatility_impact", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans calculate_event_volatility_impact: {str(e)}",
                tag="CONTEXT_AWARE",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans calculate_event_volatility_impact: {str(e)}", priority=4
            )
            logger.warning(f"Erreur dans calculate_event_volatility_impact: {str(e)}")
            return 0.0

    def calculate_event_timing_proximity(
        self, events: pd.DataFrame, timestamp: pd.Timestamp
    ) -> float:
        try:
            start_time = time.time()
            events["time_diff"] = (
                events["timestamp"] - timestamp
            ).dt.total_seconds() / 60.0
            closest_event_diff = events["time_diff"].iloc[
                events["time_diff"].abs().argmin()
            ]
            result = closest_event_diff if not np.isnan(closest_event_diff) else 0.0
            latency = time.time() - start_time
            self.log_performance(
                "calculate_event_timing_proximity", latency, success=True
            )
            return result
        except Exception as e:
            self.log_performance(
                "calculate_event_timing_proximity", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans calculate_event_timing_proximity: {str(e)}",
                tag="CONTEXT_AWARE",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans calculate_event_timing_proximity: {str(e)}", priority=4
            )
            logger.warning(f"Erreur dans calculate_event_timing_proximity: {str(e)}")
            return 0.0

    def calculate_event_frequency_24h(
        self, events: pd.DataFrame, timestamp: pd.Timestamp
    ) -> int:
        try:
            start_time = time.time()
            time_window = timestamp - timedelta(hours=24)
            recent_events = events[
                (events["timestamp"] >= time_window)
                & (events["timestamp"] <= timestamp)
            ]
            result = len(recent_events)
            latency = time.time() - start_time
            self.log_performance("calculate_event_frequency_24h", latency, success=True)
            return result
        except Exception as e:
            self.log_performance(
                "calculate_event_frequency_24h", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans calculate_event_frequency_24h: {str(e)}",
                tag="CONTEXT_AWARE",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans calculate_event_frequency_24h: {str(e)}", priority=4
            )
            logger.warning(f"Erreur dans calculate_event_frequency_24h: {str(e)}")
            return 0

    def compute_news_sentiment_momentum(
        self, news_data: pd.DataFrame, window: str = "1h"
    ) -> pd.Series:
        try:
            start_time = time.time()
            news_data = self.parse_news_data(news_data)
            if news_data.empty:
                return pd.Series(0.0, index=news_data.index)
            momentum = (
                news_data["sentiment_score"]
                .groupby(pd.Grouper(key="timestamp", freq=window))
                .mean()
                .pct_change()
                .fillna(0.0)
            )
            result = momentum.reindex(news_data.index, method="ffill").fillna(0.0)
            latency = time.time() - start_time
            self.log_performance(
                "compute_news_sentiment_momentum", latency, success=True
            )
            return result
        except Exception as e:
            self.log_performance(
                "compute_news_sentiment_momentum", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_news_sentiment_momentum: {str(e)}",
                tag="CONTEXT_AWARE",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_news_sentiment_momentum: {str(e)}", priority=4
            )
            logger.error(f"Erreur dans compute_news_sentiment_momentum: {str(e)}")
            return pd.Series(0.0, index=news_data.index)

    def compute_news_event_proximity(
        self, news_data: pd.DataFrame, window: str = "1h"
    ) -> pd.Series:
        try:
            start_time = time.time()
            news_data = self.parse_news_data(news_data)
            if news_data.empty:
                return pd.Series(0.0, index=news_data.index)
            proximity = (
                news_data["timestamp"]
                .groupby(pd.Grouper(key="timestamp", freq=window))
                .count()
            )
            result = proximity.reindex(news_data.index, method="ffill").fillna(0.0)
            latency = time.time() - start_time
            self.log_performance("compute_news_event_proximity", latency, success=True)
            return result
        except Exception as e:
            self.log_performance(
                "compute_news_event_proximity", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_news_event_proximity: {str(e)}",
                tag="CONTEXT_AWARE",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_news_event_proximity: {str(e)}", priority=4
            )
            logger.error(f"Erreur dans compute_news_event_proximity: {str(e)}")
            return pd.Series(0.0, index=news_data.index)

    def compute_macro_event_severity(self, calendar_data: pd.DataFrame) -> pd.Series:
        try:
            start_time = time.time()
            if (
                "severity" not in calendar_data.columns
                or "timestamp" not in calendar_data.columns
            ):
                error_msg = (
                    "Colonnes severity ou timestamp manquantes dans calendar_data"
                )
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=calendar_data.index)
            calendar_data["timestamp"] = pd.to_datetime(calendar_data["timestamp"])
            result = calendar_data["severity"].fillna(0.0)
            latency = time.time() - start_time
            self.log_performance("compute_macro_event_severity", latency, success=True)
            return result
        except Exception as e:
            self.log_performance(
                "compute_macro_event_severity", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_macro_event_severity: {str(e)}",
                tag="CONTEXT_AWARE",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_macro_event_severity: {str(e)}", priority=4
            )
            logger.error(f"Erreur dans compute_macro_event_severity: {str(e)}")
            return pd.Series(0.0, index=calendar_data.index)

    def compute_time_to_expiry_proximity(
        self, data: pd.DataFrame, expiry_dates: pd.Series
    ) -> pd.Series:
        try:
            start_time = time.time()
            if "timestamp" not in data.columns:
                error_msg = "Colonne timestamp manquante dans data"
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=data.index)
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            expiry_dates = pd.to_datetime(expiry_dates)
            proximity = (expiry_dates - data["timestamp"]).dt.total_seconds() / (
                24 * 3600
            )
            result = proximity.fillna(0.0)
            latency = time.time() - start_time
            self.log_performance(
                "compute_time_to_expiry_proximity", latency, success=True
            )
            return result
        except Exception as e:
            self.log_performance(
                "compute_time_to_expiry_proximity", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_time_to_expiry_proximity: {str(e)}",
                tag="CONTEXT_AWARE",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_time_to_expiry_proximity: {str(e)}", priority=4
            )
            logger.error(f"Erreur dans compute_time_to_expiry_proximity: {str(e)}")
            return pd.Series(0.0, index=data.index)

    def compute_economic_calendar_weight(
        self, calendar_data: pd.DataFrame
    ) -> pd.Series:
        try:
            start_time = time.time()
            if (
                "weight" not in calendar_data.columns
                or "timestamp" not in calendar_data.columns
            ):
                error_msg = "Colonnes weight ou timestamp manquantes dans calendar_data"
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=calendar_data.index)
            calendar_data["timestamp"] = pd.to_datetime(calendar_data["timestamp"])
            result = calendar_data["weight"].fillna(0.0)
            latency = time.time() - start_time
            self.log_performance(
                "compute_economic_calendar_weight", latency, success=True
            )
            return result
        except Exception as e:
            self.log_performance(
                "compute_economic_calendar_weight", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_economic_calendar_weight: {str(e)}",
                tag="CONTEXT_AWARE",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_economic_calendar_weight: {str(e)}", priority=4
            )
            logger.error(f"Erreur dans compute_economic_calendar_weight: {str(e)}")
            return pd.Series(0.0, index=calendar_data.index)

    def compute_news_volume_spike(
        self, news_data: pd.DataFrame, window: str = "5min"
    ) -> pd.Series:
        try:
            start_time = time.time()
            news_data = self.parse_news_data(news_data)
            if news_data.empty:
                return pd.Series(0.0, index=news_data.index)
            volume = (
                news_data["volume"]
                .groupby(pd.Grouper(key="timestamp", freq=window))
                .sum()
            )
            mean_volume = volume.rolling(window="1h", min_periods=1).mean()
            spike = (volume - mean_volume) / mean_volume.replace(0, 1e-6)
            result = spike.reindex(news_data.index, method="ffill").fillna(0.0)
            latency = time.time() - start_time
            self.log_performance("compute_news_volume_spike", latency, success=True)
            return result
        except Exception as e:
            self.log_performance(
                "compute_news_volume_spike", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_news_volume_spike: {str(e)}",
                tag="CONTEXT_AWARE",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_news_volume_spike: {str(e)}", priority=4)
            logger.error(f"Erreur dans compute_news_volume_spike: {str(e)}")
            return pd.Series(0.0, index=news_data.index)

    def compute_news_sentiment_acceleration(
        self, news_data: pd.DataFrame, window: str = "1h"
    ) -> pd.Series:
        try:
            start_time = time.time()
            news_data = self.parse_news_data(news_data)
            if news_data.empty:
                return pd.Series(0.0, index=news_data.index)
            momentum = (
                news_data["sentiment_score"]
                .groupby(pd.Grouper(key="timestamp", freq=window))
                .mean()
                .pct_change()
            )
            acceleration = momentum.diff().fillna(0.0)
            result = acceleration.reindex(news_data.index, method="ffill").fillna(0.0)
            latency = time.time() - start_time
            self.log_performance(
                "compute_news_sentiment_acceleration", latency, success=True
            )
            return result
        except Exception as e:
            self.log_performance(
                "compute_news_sentiment_acceleration", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_news_sentiment_acceleration: {str(e)}",
                tag="CONTEXT_AWARE",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_news_sentiment_acceleration: {str(e)}", priority=4
            )
            logger.error(f"Erreur dans compute_news_sentiment_acceleration: {str(e)}")
            return pd.Series(0.0, index=news_data.index)

    def compute_macro_event_momentum(
        self, calendar_data: pd.DataFrame, window: str = "1d"
    ) -> pd.Series:
        try:
            start_time = time.time()
            if (
                "severity" not in calendar_data.columns
                or "timestamp" not in calendar_data.columns
            ):
                error_msg = (
                    "Colonnes severity ou timestamp manquantes dans calendar_data"
                )
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=calendar_data.index)
            calendar_data["timestamp"] = pd.to_datetime(calendar_data["timestamp"])
            momentum = (
                calendar_data["severity"]
                .groupby(pd.Grouper(key="timestamp", freq=window))
                .mean()
                .pct_change()
                .fillna(0.0)
            )
            result = momentum.reindex(calendar_data.index, method="ffill").fillna(0.0)
            latency = time.time() - start_time
            self.log_performance("compute_macro_event_momentum", latency, success=True)
            return result
        except Exception as e:
            self.log_performance(
                "compute_macro_event_momentum", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_macro_event_momentum: {str(e)}",
                tag="CONTEXT_AWARE",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_macro_event_momentum: {str(e)}", priority=4
            )
            logger.error(f"Erreur dans compute_macro_event_momentum: {str(e)}")
            return pd.Series(0.0, index=calendar_data.index)

    def compute_cyclic_feature(
        self, data: pd.DataFrame, period: str = "month", func: str = "sin"
    ) -> pd.Series:
        try:
            start_time = time.time()
            if "timestamp" not in data.columns:
                error_msg = "Colonne timestamp manquante dans data"
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=data.index)
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            if period == "month":
                value = data["timestamp"].dt.month / 12.0
            elif period == "week":
                value = data["timestamp"].dt.isocalendar().week / 5.0
            else:
                error_msg = f"Période invalide: {period}"
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=data.index)
            if func == "sin":
                result = np.sin(2 * np.pi * value)
            elif func == "cos":
                result = np.cos(2 * np.pi * value)
            else:
                error_msg = f"Fonction invalide: {func}"
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=data.index)
            result = pd.Series(result, index=data.index).fillna(0.0)
            latency = time.time() - start_time
            self.log_performance(
                f"compute_cyclic_feature_{period}_{func}", latency, success=True
            )
            return result
        except Exception as e:
            self.log_performance(
                f"compute_cyclic_feature_{period}_{func}",
                0,
                success=False,
                error=str(e),
            )
            miya_alerts(
                f"Erreur dans compute_cyclic_feature: {str(e)}",
                tag="CONTEXT_AWARE",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_cyclic_feature: {str(e)}", priority=4)
            logger.error(f"Erreur dans compute_cyclic_feature: {str(e)}")
            return pd.Series(0.0, index=data.index)

    def compute_roll_yield_curve(self, futures_data: pd.DataFrame) -> pd.Series:
        try:
            start_time = time.time()
            if (
                "near_price" not in futures_data.columns
                or "far_price" not in futures_data.columns
            ):
                error_msg = (
                    "Colonnes near_price ou far_price manquantes dans futures_data"
                )
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=futures_data.index)
            futures_data["near_price"] = pd.to_numeric(
                futures_data["near_price"], errors="coerce"
            )
            futures_data["far_price"] = pd.to_numeric(
                futures_data["far_price"], errors="coerce"
            )
            yield_curve = (
                futures_data["far_price"] - futures_data["near_price"]
            ) / futures_data["near_price"].replace(0, 1e-6)
            result = yield_curve.fillna(0.0)
            latency = time.time() - start_time
            self.log_performance("compute_roll_yield_curve", latency, success=True)
            return result
        except Exception as e:
            self.log_performance(
                "compute_roll_yield_curve", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_roll_yield_curve: {str(e)}",
                tag="CONTEXT_AWARE",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_roll_yield_curve: {str(e)}", priority=4)
            logger.error(f"Erreur dans compute_roll_yield_curve: {str(e)}")
            return pd.Series(0.0, index=futures_data.index)

    def compute_option_skew_impact(self, options_data: pd.DataFrame) -> pd.Series:
        """
        Calcule l'impact du skew des options (Phase 13).
        """
        try:
            start_time = time.time()
            if (
                "option_skew" not in options_data.columns
                or "timestamp" not in options_data.columns
            ):
                error_msg = (
                    "Colonnes option_skew ou timestamp manquantes dans options_data"
                )
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=options_data.index)
            options_data["timestamp"] = pd.to_datetime(options_data["timestamp"])
            result = options_data["option_skew"].fillna(0.0)
            latency = time.time() - start_time
            self.log_performance("compute_option_skew_impact", latency, success=True)
            return result
        except Exception as e:
            self.log_performance(
                "compute_option_skew_impact", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_option_skew_impact: {str(e)}",
                tag="CONTEXT_AWARE",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_option_skew_impact: {str(e)}", priority=4)
            logger.error(f"Erreur dans compute_option_skew_impact: {str(e)}")
            return pd.Series(0.0, index=options_data.index)

    def compute_microstructure_event_impact(
        self, microstructure_data: pd.DataFrame
    ) -> pd.Series:
        """
        Calcule l'impact des événements de microstructure (Phase 15).
        """
        try:
            start_time = time.time()
            if (
                "spoofing_score" not in microstructure_data.columns
                or "timestamp" not in microstructure_data.columns
            ):
                error_msg = "Colonnes spoofing_score ou timestamp manquantes dans microstructure_data"
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=microstructure_data.index)
            microstructure_data["timestamp"] = pd.to_datetime(
                microstructure_data["timestamp"]
            )
            result = microstructure_data["spoofing_score"].fillna(0.0)
            latency = time.time() - start_time
            self.log_performance(
                "compute_microstructure_event_impact", latency, success=True
            )
            return result
        except Exception as e:
            self.log_performance(
                "compute_microstructure_event_impact", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_microstructure_event_impact: {str(e)}",
                tag="CONTEXT_AWARE",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_microstructure_event_impact: {str(e)}", priority=4
            )
            logger.error(f"Erreur dans compute_microstructure_event_impact: {str(e)}")
            return pd.Series(0.0, index=microstructure_data.index)

    def compute_hft_news_reaction(
        self, news_data: pd.DataFrame, hft_data: pd.DataFrame, window: str = "5min"
    ) -> pd.Series:
        """
        Calcule la réaction des HFT aux nouvelles (Phase 18).
        """
        try:
            start_time = time.time()
            news_data = self.parse_news_data(news_data)
            if (
                news_data.empty
                or "hft_activity_score" not in hft_data.columns
                or "timestamp" not in hft_data.columns
            ):
                error_msg = "Données news_data vides ou colonnes hft_activity_score/timestamp manquantes dans hft_data"
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=news_data.index)
            hft_data["timestamp"] = pd.to_datetime(hft_data["timestamp"])
            news_volume = (
                news_data["volume"]
                .groupby(pd.Grouper(key="timestamp", freq=window))
                .sum()
            )
            hft_activity = (
                hft_data["hft_activity_score"]
                .groupby(pd.Grouper(key="timestamp", freq=window))
                .mean()
            )
            reaction = (hft_activity * news_volume).fillna(0.0)
            result = reaction.reindex(news_data.index, method="ffill").fillna(0.0)
            latency = time.time() - start_time
            self.log_performance("compute_hft_news_reaction", latency, success=True)
            return result
        except Exception as e:
            self.log_performance(
                "compute_hft_news_reaction", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_hft_news_reaction: {str(e)}",
                tag="CONTEXT_AWARE",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_hft_news_reaction: {str(e)}", priority=4)
            logger.error(f"Erreur dans compute_hft_news_reaction: {str(e)}")
            return pd.Series(0.0, index=news_data.index)

    def compute_trade_velocity_impact(self, hft_data: pd.DataFrame) -> pd.Series:
        """
        Calcule l'impact de la vélocité des trades (Phase 18).
        """
        try:
            start_time = time.time()
            if (
                "trade_velocity" not in hft_data.columns
                or "timestamp" not in hft_data.columns
            ):
                error_msg = (
                    "Colonnes trade_velocity ou timestamp manquantes dans hft_data"
                )
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=hft_data.index)
            hft_data["timestamp"] = pd.to_datetime(hft_data["timestamp"])
            result = hft_data["trade_velocity"].fillna(0.0)
            latency = time.time() - start_time
            self.log_performance("compute_trade_velocity_impact", latency, success=True)
            return result
        except Exception as e:
            self.log_performance(
                "compute_trade_velocity_impact", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_trade_velocity_impact: {str(e)}",
                tag="CONTEXT_AWARE",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_trade_velocity_impact: {str(e)}", priority=4
            )
            logger.error(f"Erreur dans compute_trade_velocity_impact: {str(e)}")
            return pd.Series(0.0, index=hft_data.index)

    def compute_contextual_metrics(
        self,
        data: pd.DataFrame,
        news_data: pd.DataFrame,
        calendar_data: pd.DataFrame,
        futures_data: pd.DataFrame,
        options_data: pd.DataFrame,
        microstructure_data: pd.DataFrame,
        hft_data: pd.DataFrame,
        expiry_dates: pd.Series,
        macro_events_path: str = os.path.join(
            BASE_DIR, "data", "events", "macro_events.csv"
        ),
        volatility_history_path: str = os.path.join(
            BASE_DIR, "data", "events", "event_volatility_history.csv"
        ),
    ) -> pd.DataFrame:
        """
        Calcule les métriques contextuelles basées sur les événements macro, nouvelles IQFeed, et cyclicité temporelle.

        Args:
            data (pd.DataFrame): Données principales avec timestamp.
            news_data (pd.DataFrame): Données de nouvelles avec sentiment_score et volume.
            calendar_data (pd.DataFrame): Données du calendrier économique avec severity et weight.
            futures_data (pd.DataFrame): Données des futures avec near_price et far_price.
            options_data (pd.DataFrame): Données des options avec option_skew.
            microstructure_data (pd.DataFrame): Données de microstructure avec spoofing_score.
            hft_data (pd.DataFrame): Données HFT avec hft_activity_score et trade_velocity.
            expiry_dates (pd.Series): Dates d'expiration des contrats.
            macro_events_path (str): Chemin vers macro_events.csv.
            volatility_history_path (str): Chemin vers event_volatility_history.csv.

        Returns:
            pd.DataFrame: Données enrichies avec 22 métriques contextuelles.
        """
        try:
            start_time = time.time()
            config = self.load_config_with_manager_new()

            if data.empty:
                error_msg = "DataFrame principal vide"
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=5
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
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
                        "Features contextuelles récupérées du cache",
                        tag="CONTEXT_AWARE",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert("Features contextuelles récupérées du cache", priority=1)
                    self.log_performance(
                        "compute_contextual_metrics_cache_hit", 0, success=True
                    )
                    return cached_data

            data = data.copy()
            if "timestamp" not in data.columns:
                miya_speak(
                    "Colonne timestamp manquante, création par défaut",
                    tag="CONTEXT_AWARE",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert("Colonne timestamp manquante", priority=3)
                logger.warning("Colonne timestamp manquante, création par défaut")
                default_start = pd.Timestamp.now()
                data["timestamp"] = pd.date_range(
                    start=default_start, periods=len(data), freq="1min"
                )

            data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
            if data["timestamp"].isna().any():
                miya_speak(
                    "NaN dans timestamp, imputés avec la première date valide",
                    tag="CONTEXT_AWARE",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert("NaN dans timestamp", priority=3)
                logger.warning("NaN dans timestamp, imputation")
                first_valid_time = (
                    data["timestamp"].dropna().iloc[0]
                    if not data["timestamp"].dropna().empty
                    else pd.Timestamp.now()
                )
                data["timestamp"] = data["timestamp"].fillna(first_valid_time)

            # Charger les fichiers d'événements et d'historique de volatilité
            try:
                macro_events = pd.read_csv(macro_events_path)
                volatility_history = pd.read_csv(volatility_history_path)
            except FileNotFoundError as e:
                error_msg = f"Fichier introuvable: {e}"
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                raise
            except Exception as e:
                error_msg = f"Erreur lors du chargement des fichiers: {e}"
                miya_alerts(
                    error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                raise

            # Vérification des colonnes requises dans macro_events
            required_event_cols = ["timestamp", "event_type", "event_impact_score"]
            missing_event_cols = [
                col for col in required_event_cols if col not in macro_events.columns
            ]
            for col in missing_event_cols:
                macro_events[col] = 0
                miya_speak(
                    f"Colonne IQFeed '{col}' manquante dans macro_events, imputée à 0",
                    tag="CONTEXT_AWARE",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(
                    f"Colonne IQFeed '{col}' manquante dans macro_events", priority=3
                )
                logger.warning(f"Colonne manquante dans macro_events: {col}")

            # Vérification des colonnes requises dans volatility_history
            required_history_cols = ["event_type", "volatility_impact"]
            missing_history_cols = [
                col
                for col in required_history_cols
                if col not in volatility_history.columns
            ]
            for col in missing_history_cols:
                volatility_history[col] = 0
                miya_speak(
                    f"Colonne IQFeed '{col}' manquante dans volatility_history, imputée à 0",
                    tag="CONTEXT_AWARE",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(
                    f"Colonne IQFeed '{col}' manquante dans volatility_history",
                    priority=3,
                )
                logger.warning(f"Colonne manquante dans volatility_history: {col}")

            # Conversion des timestamps
            macro_events["timestamp"] = pd.to_datetime(
                macro_events["timestamp"], errors="coerce"
            )
            volatility_history["timestamp"] = pd.to_datetime(
                volatility_history["timestamp"], errors="coerce"
            )

            # Calcul des métriques existantes
            data["event_volatility_impact"] = data["timestamp"].apply(
                lambda x: self.calculate_event_volatility_impact(
                    macro_events, volatility_history, x
                )
            )
            data["event_timing_proximity"] = data["timestamp"].apply(
                lambda x: self.calculate_event_timing_proximity(macro_events, x)
            )
            data["event_frequency_24h"] = data["timestamp"].apply(
                lambda x: self.calculate_event_frequency_24h(macro_events, x)
            )

            # Calcul des nouvelles métriques
            data["news_sentiment_momentum"] = self.compute_news_sentiment_momentum(
                news_data
            )
            data["news_event_proximity"] = self.compute_news_event_proximity(news_data)
            data["macro_event_severity"] = self.compute_macro_event_severity(
                calendar_data
            )
            data["time_to_expiry_proximity"] = self.compute_time_to_expiry_proximity(
                data, expiry_dates
            )
            data["economic_calendar_weight"] = self.compute_economic_calendar_weight(
                calendar_data
            )
            data["news_volume_spike"] = self.compute_news_volume_spike(
                news_data, window="5min"
            )
            data["news_volume_1h"] = self.compute_news_volume_spike(
                news_data, window="1h"
            )
            data["news_volume_1d"] = self.compute_news_volume_spike(
                news_data, window="1d"
            )
            data["news_sentiment_acceleration"] = (
                self.compute_news_sentiment_acceleration(news_data)
            )
            data["macro_event_momentum"] = self.compute_macro_event_momentum(
                calendar_data
            )
            data["month_of_year_sin"] = self.compute_cyclic_feature(
                data, period="month", func="sin"
            )
            data["month_of_year_cos"] = self.compute_cyclic_feature(
                data, period="month", func="cos"
            )
            data["week_of_month_sin"] = self.compute_cyclic_feature(
                data, period="week", func="sin"
            )
            data["week_of_month_cos"] = self.compute_cyclic_feature(
                data, period="week", func="cos"
            )
            data["roll_yield_curve"] = self.compute_roll_yield_curve(futures_data)
            data["option_skew_impact"] = self.compute_option_skew_impact(options_data)
            data["microstructure_event_impact"] = (
                self.compute_microstructure_event_impact(microstructure_data)
            )
            data["hft_news_reaction"] = self.compute_hft_news_reaction(
                news_data, hft_data
            )
            data["trade_velocity_impact"] = self.compute_trade_velocity_impact(hft_data)

            metrics = [
                "event_volatility_impact",
                "event_timing_proximity",
                "event_frequency_24h",
                "news_sentiment_momentum",
                "news_event_proximity",
                "macro_event_severity",
                "time_to_expiry_proximity",
                "economic_calendar_weight",
                "news_volume_spike",
                "news_volume_1h",
                "news_volume_1d",
                "news_sentiment_acceleration",
                "macro_event_momentum",
                "month_of_year_sin",
                "month_of_year_cos",
                "week_of_month_sin",
                "week_of_month_cos",
                "roll_yield_curve",
                "option_skew_impact",
                "microstructure_event_impact",
                "hft_news_reaction",
                "trade_velocity_impact",
            ]
            self.validate_shap_features(metrics)
            self.cache_metrics(data, cache_key)
            self.plot_metrics(data, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            latency = time.time() - start_time
            miya_speak(
                "Métriques contextuelles calculées",
                tag="CONTEXT_AWARE",
                voice_profile="calm",
                priority=1,
            )
            send_alert("Métriques contextuelles calculées", priority=1)
            logger.info("Métriques contextuelles calculées")
            self.log_performance(
                "compute_contextual_metrics",
                latency,
                success=True,
                num_rows=len(data),
                num_metrics=len(metrics),
            )
            self.save_snapshot(
                "compute_contextual_metrics",
                {"num_rows": len(data), "metrics": metrics},
            )
            return data
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans compute_contextual_metrics: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="CONTEXT_AWARE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=4)
            logger.error(error_msg)
            self.log_performance(
                "compute_contextual_metrics", latency, success=False, error=str(e)
            )
            self.save_snapshot("compute_contextual_metrics", {"error": str(e)})
            data["timestamp"] = pd.date_range(
                start=pd.Timestamp.now(), periods=len(data), freq="1min"
            )
            for col in [
                "event_volatility_impact",
                "event_timing_proximity",
                "event_frequency_24h",
                "news_sentiment_momentum",
                "news_event_proximity",
                "macro_event_severity",
                "time_to_expiry_proximity",
                "economic_calendar_weight",
                "news_volume_spike",
                "news_volume_1h",
                "news_volume_1d",
                "news_sentiment_acceleration",
                "macro_event_momentum",
                "month_of_year_sin",
                "month_of_year_cos",
                "week_of_month_sin",
                "week_of_month_cos",
                "roll_yield_curve",
                "option_skew_impact",
                "microstructure_event_impact",
                "hft_news_reaction",
                "trade_velocity_impact",
            ]:
                data[col] = 0.0
            return data

    def cache_metrics(self, metrics: pd.DataFrame, cache_key: str) -> None:
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
                tag="CONTEXT_AWARE",
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
                tag="CONTEXT_AWARE",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur mise en cache métriques: {str(e)}", priority=4)
            logger.error(f"Erreur mise en cache métriques: {str(e)}")

    def validate_shap_features(self, features: List[str]) -> bool:
        try:
            if not os.path.exists(FEATURE_IMPORTANCE_PATH):
                miya_alerts(
                    "Fichier SHAP manquant",
                    tag="CONTEXT_AWARE",
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
                    tag="CONTEXT_AWARE",
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
                    tag="CONTEXT_AWARE",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(
                    f"Features non incluses dans top 150 SHAP: {missing}", priority=3
                )
                logger.warning(f"Features non incluses dans top 150 SHAP: {missing}")
            miya_speak(
                "SHAP features validées",
                tag="CONTEXT_AWARE",
                voice_profile="calm",
                priority=1,
            )
            send_alert("SHAP features validées", priority=1)
            logger.info("SHAP features validées")
            return True
        except Exception as e:
            miya_alerts(
                f"Erreur validation SHAP features: {str(e)}",
                tag="CONTEXT_AWARE",
                voice_profile="urgent",
                priority=4,
            )
            send_alert(f"Erreur validation SHAP features: {str(e)}", priority=4)
            logger.error(f"Erreur validation SHAP features: {str(e)}")
            return False

    def plot_metrics(self, metrics: pd.DataFrame, timestamp: str) -> None:
        try:
            start_time = time.time()
            timestamp_safe = timestamp.replace(":", "-")
            plt.figure(figsize=(12, 6))
            plt.plot(
                metrics["timestamp"],
                metrics["news_sentiment_momentum"],
                label="News Sentiment Momentum",
                color="blue",
            )
            plt.plot(
                metrics["timestamp"],
                metrics["news_volume_spike"],
                label="News Volume Spike",
                color="orange",
            )
            plt.plot(
                metrics["timestamp"],
                metrics["event_volatility_impact"],
                label="Event Volatility Impact",
                color="green",
            )
            plt.plot(
                metrics["timestamp"],
                metrics["option_skew_impact"],
                label="Option Skew Impact",
                color="purple",
            )
            plt.plot(
                metrics["timestamp"],
                metrics["microstructure_event_impact"],
                label="Microstructure Event Impact",
                color="red",
            )
            plt.plot(
                metrics["timestamp"],
                metrics["hft_news_reaction"],
                label="HFT News Reaction",
                color="cyan",
            )
            plt.title(f"Contextual Metrics - {timestamp}")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.savefig(
                os.path.join(FIGURES_DIR, f"context_aware_metrics_{timestamp_safe}.png")
            )
            plt.close()
            latency = time.time() - start_time
            miya_speak(
                f"Visualisations générées: {FIGURES_DIR}",
                tag="CONTEXT_AWARE",
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
                tag="CONTEXT_AWARE",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur génération visualisations: {str(e)}", priority=4)
            logger.error(f"Erreur génération visualisations: {str(e)}")


if __name__ == "__main__":
    try:
        calculator = ContextAwareFilter()
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "close": np.random.normal(5100, 10, 100),
            }
        )
        news_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "sentiment_score": np.random.uniform(-1, 1, 100),
                "volume": np.random.randint(1, 10, 100),
            }
        )
        calendar_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "severity": np.random.uniform(0, 1, 100),
                "weight": np.random.uniform(0, 1, 100),
            }
        )
        futures_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "near_price": np.random.normal(5100, 10, 100),
                "far_price": np.random.normal(5120, 10, 100),
            }
        )
        options_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "option_skew": np.random.uniform(0, 0.5, 100),
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
                "trade_velocity": np.random.uniform(50, 150, 100),
            }
        )
        expiry_dates = pd.Series(pd.date_range("2025-04-15", periods=100, freq="1d"))
        macro_events = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2025-04-14 08:00",
                        "2025-04-14 09:10",
                        "2025-04-14 10:00",
                        "2025-04-13 09:00",
                        "2025-04-15 09:00",
                    ]
                ),
                "event_type": [
                    "CPI Release",
                    "FOMC Meeting",
                    "Earnings Report",
                    "GDP Release",
                    "Jobs Report",
                ],
                "event_impact_score": [0.8, 0.9, 0.6, 0.7, 0.85],
            }
        )
        event_volatility_history = pd.DataFrame(
            {
                "event_type": [
                    "CPI Release",
                    "FOMC Meeting",
                    "Earnings Report",
                    "GDP Release",
                    "Jobs Report",
                ],
                "volatility_impact": [0.15, 0.20, 0.10, 0.12, 0.18],
                "timestamp": pd.to_datetime(["2025-04-01"] * 5),
            }
        )
        os.makedirs(os.path.join(BASE_DIR, "data", "events"), exist_ok=True)
        macro_events.to_csv(
            os.path.join(BASE_DIR, "data", "events", "macro_events.csv"), index=False
        )
        event_volatility_history.to_csv(
            os.path.join(BASE_DIR, "data", "events", "event_volatility_history.csv"),
            index=False,
        )
        result = calculator.compute_contextual_metrics(
            data,
            news_data,
            calendar_data,
            futures_data,
            options_data,
            microstructure_data,
            hft_data,
            expiry_dates,
        )
        print(
            result[
                [
                    "timestamp",
                    "event_volatility_impact",
                    "news_sentiment_momentum",
                    "month_of_year_sin",
                    "option_skew_impact",
                    "microstructure_event_impact",
                    "hft_news_reaction",
                    "trade_velocity_impact",
                ]
            ].head()
        )
        miya_speak(
            "Test compute_contextual_metrics terminé",
            tag="TEST",
            voice_profile="calm",
            priority=1,
        )
        send_alert("Test compute_contextual_metrics terminé", priority=1)
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
