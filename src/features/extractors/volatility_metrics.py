# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/features/extractors/volatility_metrics.py
# Calcule les métriques de volatilité (range_threshold, volatility_trend, bollinger_width_20, volatility_regime_stability, vix_es_correlation, volatility_premium, etc.) à partir des données IQFeed.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Génère 18 métriques de volatilité avec vix_es_correlation (méthode 1), utilise un cache intermédiaire,
#        enregistre des logs psutil, valide SHAP (méthode 17), normalise les métriques pour l’inférence,
#        et intègre 12 métriques avancées de volatilité. Conforme à la Phase 7 (volatilité).
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, ta>=0.11.0,<1.0.0, psutil>=5.9.0,<6.0.0, scikit-learn>=1.5.0,<2.0.0, matplotlib>=3.7.0,<4.0.0, json, gzip, hashlib, logging
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
# - data/features/cache/volatility/
# - data/logs/volatility_metrics_performance.csv
# - data/features/volatility_snapshots/
# - data/figures/volatility/
#
# Notes :
# - Utilise exclusivement IQFeed pour les données (remplace dxFeed).
# - Suppression de toute référence à dxFeed, obs_t, 320 features, et 81 features.
# - Intègre les Phases 1-18, spécifiquement Phase 7 (volatility_metrics.py).
# - Tests unitaires disponibles dans tests/test_volatility_metrics.py.
# - vix_es_correlation simulé si données VIX/ES absentes, à intégrer via iqfeed_fetch.py.
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
import ta
from sklearn.preprocessing import MinMaxScaler

from src.model.utils.alert_manager import send_alert
from src.model.utils.config_manager import config_manager
from src.model.utils.miya_console import miya_alerts, miya_speak

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data", "features", "cache", "volatility")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "features", "volatility_snapshots")
CSV_LOG_PATH = os.path.join(
    BASE_DIR, "data", "logs", "volatility_metrics_performance.csv"
)
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures", "volatility")
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
    filename=os.path.join(BASE_DIR, "data", "logs", "volatility_metrics.log"),
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


class VolatilityMetricsCalculator:

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
                "VolatilityMetricsCalculator initialisé",
                tag="VOLATILITY",
                voice_profile="calm",
                priority=2,
            )
            send_alert("VolatilityMetricsCalculator initialisé", priority=1)
            logger.info("VolatilityMetricsCalculator initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config_path": config_path})
        except Exception as e:
            error_msg = f"Erreur initialisation VolatilityMetricsCalculator: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="VOLATILITY", voice_profile="urgent", priority=3)
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
            if "volatility_metrics_calculator" not in config:
                raise ValueError(
                    "Clé 'volatility_metrics_calculator' manquante dans la configuration"
                )
            required_keys = ["buffer_size", "max_cache_size", "cache_hours"]
            missing_keys = [
                key
                for key in required_keys
                if key not in config["volatility_metrics_calculator"]
            ]
            if missing_keys:
                raise ValueError(
                    f"Clés manquantes dans 'volatility_metrics_calculator': {missing_keys}"
                )
            return config["volatility_metrics_calculator"]

        try:
            start_time = time.time()
            config = self.with_retries(load_yaml)
            latency = time.time() - start_time
            miya_speak(
                "Configuration volatility_metrics_calculator chargée via config_manager",
                tag="VOLATILITY",
                voice_profile="calm",
                priority=2,
            )
            send_alert(
                "Configuration volatility_metrics_calculator chargée via config_manager",
                priority=1,
            )
            logger.info(
                "Configuration volatility_metrics_calculator chargée via config_manager"
            )
            self.log_performance("load_config_with_manager", latency, success=True)
            self.save_snapshot("load_config_with_manager", {"config_path": config_path})
            return config
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur chargement config via config_manager: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="VOLATILITY", voice_profile="urgent", priority=3)
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
                    tag="VOLATILITY",
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
                tag="VOLATILITY",
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
                tag="VOLATILITY",
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
                tag="VOLATILITY",
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
                tag="VOLATILITY",
                voice_profile="calm",
                priority=2,
            )
            send_alert("SIGINT received, snapshot saved", priority=2)
            logger.info("SIGINT received, snapshot saved")
        except Exception as e:
            miya_alerts(
                f"Erreur gestion SIGINT: {str(e)}",
                tag="VOLATILITY",
                voice_profile="urgent",
                priority=5,
            )
            send_alert(f"Erreur gestion SIGINT: {str(e)}", priority=4)
            logger.error(f"Erreur gestion SIGINT: {str(e)}")
        finally:
            raise SystemExit("Terminated by SIGINT")

    def validate_iqfeed_data(self, data: pd.DataFrame) -> bool:
        """
        Valide les données IQFeed pour les métriques de volatilité.
        """
        try:
            required_cols = ["timestamp", "high", "low", "close", "atr_14"]
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                error_msg = f"Colonnes IQFeed manquantes: {missing_cols}"
                miya_alerts(
                    error_msg, tag="VOLATILITY", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return False
            if data[required_cols].isna().any().any():
                error_msg = "Valeurs NaN dans les données IQFeed"
                miya_alerts(
                    error_msg, tag="VOLATILITY", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return False
            if not np.isfinite(
                data[required_cols].select_dtypes(include=[np.number]).values
            ).all():
                error_msg = "Valeurs infinies dans les données IQFeed"
                miya_alerts(
                    error_msg, tag="VOLATILITY", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return False
            miya_speak(
                "Données IQFeed validées",
                tag="VOLATILITY",
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
            miya_alerts(error_msg, tag="VOLATILITY", voice_profile="urgent", priority=4)
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
                    tag="VOLATILITY",
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
                    tag="VOLATILITY",
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
                    tag="VOLATILITY",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(
                    f"Features non incluses dans top 150 SHAP: {missing}", priority=3
                )
                logger.warning(f"Features non incluses dans top 150 SHAP: {missing}")
            miya_speak(
                "SHAP features validées",
                tag="VOLATILITY",
                voice_profile="calm",
                priority=1,
            )
            send_alert("SHAP features validées", priority=1)
            logger.info("SHAP features validées")
            return True
        except Exception as e:
            miya_alerts(
                f"Erreur validation SHAP features: {str(e)}",
                tag="VOLATILITY",
                voice_profile="urgent",
                priority=4,
            )
            send_alert(f"Erreur validation SHAP features: {str(e)}", priority=4)
            logger.error(f"Erreur validation SHAP features: {str(e)}")
            return False

    def cache_metrics(self, metrics: pd.DataFrame, cache_key: str) -> None:
        """
        Met en cache les métriques de volatilité.
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
                tag="VOLATILITY",
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
                tag="VOLATILITY",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur mise en cache métriques: {str(e)}", priority=4)
            logger.error(f"Erreur mise en cache métriques: {str(e)}")

    def calculate_vix_es_correlation(
        self, data: pd.DataFrame, window: int = 20
    ) -> pd.Series:
        """
        Calcule la corrélation entre VIX et ES sur une fenêtre glissante.
        """
        try:
            start_time = time.time()
            if "vix" in data.columns and "es" in data.columns:
                data["vix"] = pd.to_numeric(data["vix"], errors="coerce")
                data["es"] = pd.to_numeric(data["es"], errors="coerce")
                if data[["vix", "es"]].isna().any().any():
                    miya_speak(
                        "NaN dans vix/es, imputés à 0 pour corrélation",
                        tag="VOLATILITY",
                        voice_profile="warning",
                        priority=3,
                    )
                    send_alert("NaN dans vix/es", priority=3)
                    logger.warning("NaN dans vix/es, imputés à 0 pour corrélation")
                    data[["vix", "es"]] = data[["vix", "es"]].fillna(0.0)
                vix_es_corr = (
                    data[["vix", "es"]]
                    .rolling(window=window, min_periods=1)
                    .corr()
                    .unstack()
                    .iloc[:, 1]
                )
                vix_es_corr = vix_es_corr.fillna(0.0)
            else:
                miya_speak(
                    "Colonnes vix/es manquantes, vix_es_correlation mise à 0 (intégration via iqfeed_fetch.py requise)",
                    tag="VOLATILITY",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(
                    "Colonnes vix/es manquantes, vix_es_correlation mise à 0",
                    priority=3,
                )
                logger.warning(
                    "Colonnes vix/es manquantes, vix_es_correlation mise à 0"
                )
                vix_es_corr = pd.Series(
                    0.0, index=data.index, name="vix_es_correlation"
                )
            latency = time.time() - start_time
            self.log_performance(
                "calculate_vix_es_correlation",
                latency,
                success=True,
                num_rows=len(data),
            )
            return vix_es_corr
        except Exception as e:
            self.log_performance(
                "calculate_vix_es_correlation", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans calculate_vix_es_correlation: {str(e)}",
                tag="VOLATILITY",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans calculate_vix_es_correlation: {str(e)}", priority=4
            )
            logger.error(f"Erreur dans calculate_vix_es_correlation: {str(e)}")
            return pd.Series(0.0, index=data.index, name="vix_es_correlation")

    def compute_volatility_premium(self, data: pd.DataFrame) -> pd.Series:
        """
        Calcule la prime de volatilité (IV vs volatilité réalisée).
        """
        try:
            start_time = time.time()
            if "iv_30d" not in data.columns or "realized_vol_30d" not in data.columns:
                error_msg = "Colonnes iv_30d ou realized_vol_30d manquantes"
                miya_alerts(
                    error_msg, tag="VOLATILITY", voice_profile="warning", priority=3
                )
                send_alert(error_msg, priority=3)
                logger.warning(error_msg)
                return pd.Series(0.0, index=data.index, name="volatility_premium")
            premium = (data["iv_30d"] - data["realized_vol_30d"]) / data[
                "realized_vol_30d"
            ].replace(0, 1e-6)
            result = pd.Series(premium, index=data.index).fillna(0.0)
            result.name = "volatility_premium"
            latency = time.time() - start_time
            self.log_performance("compute_volatility_premium", latency, success=True)
            return result
        except Exception as e:
            self.log_performance(
                "compute_volatility_premium", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_volatility_premium: {str(e)}",
                tag="VOLATILITY",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_volatility_premium: {str(e)}", priority=4)
            logger.error(f"Erreur dans compute_volatility_premium: {str(e)}")
            return pd.Series(0.0, index=data.index, name="volatility_premium")

    def compute_implied_move_ratio(self, data: pd.DataFrame) -> pd.Series:
        """
        Calcule le ratio de mouvement implicite (IV ATM vs volatilité réalisée).
        """
        try:
            start_time = time.time()
            if "iv_atm" not in data.columns or "realized_vol_30d" not in data.columns:
                error_msg = "Colonnes iv_atm ou realized_vol_30d manquantes"
                miya_alerts(
                    error_msg, tag="VOLATILITY", voice_profile="warning", priority=3
                )
                send_alert(error_msg, priority=3)
                logger.warning(error_msg)
                return pd.Series(0.0, index=data.index, name="implied_move_ratio")
            ratio = data["iv_atm"] / data["realized_vol_30d"].replace(0, 1e-6)
            result = pd.Series(ratio, index=data.index).fillna(0.0)
            result.name = "implied_move_ratio"
            latency = time.time() - start_time
            self.log_performance("compute_implied_move_ratio", latency, success=True)
            return result
        except Exception as e:
            self.log_performance(
                "compute_implied_move_ratio", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_implied_move_ratio: {str(e)}",
                tag="VOLATILITY",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_implied_move_ratio: {str(e)}", priority=4)
            logger.error(f"Erreur dans compute_implied_move_ratio: {str(e)}")
            return pd.Series(0.0, index=data.index, name="implied_move_ratio")

    def compute_vol_term_slope(
        self, data: pd.DataFrame, months: List[int] = [1, 3]
    ) -> pd.Series:
        """
        Calcule la pente de la courbe de volatilité à terme.
        """
        try:
            start_time = time.time()
            if (
                f"vix_term_{months[0]}m" not in data.columns
                or f"vix_term_{months[1]}m" not in data.columns
            ):
                error_msg = f"Colonnes vix_term_{months[0]}m ou vix_term_{months[1]}m manquantes"
                miya_alerts(
                    error_msg, tag="VOLATILITY", voice_profile="warning", priority=3
                )
                send_alert(error_msg, priority=3)
                logger.warning(error_msg)
                return pd.Series(0.0, index=data.index, name="vol_term_slope")
            slope = (
                data[f"vix_term_{months[1]}m"] - data[f"vix_term_{months[0]}m"]
            ) / (months[1] - months[0])
            result = pd.Series(slope, index=data.index).fillna(0.0)
            result.name = "vol_term_slope"
            latency = time.time() - start_time
            self.log_performance("compute_vol_term_slope", latency, success=True)
            return result
        except Exception as e:
            self.log_performance(
                "compute_vol_term_slope", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_vol_term_slope: {str(e)}",
                tag="VOLATILITY",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_vol_term_slope: {str(e)}", priority=4)
            logger.error(f"Erreur dans compute_vol_term_slope: {str(e)}")
            return pd.Series(0.0, index=data.index, name="vol_term_slope")

    def compute_vol_term_curvature(
        self, data: pd.DataFrame, months: List[int] = [1, 3, 6]
    ) -> pd.Series:
        """
        Calcule la courbure de la courbe de volatilité à terme.
        """
        try:
            start_time = time.time()
            cols = [f"vix_term_{m}m" for m in months]
            if not all(col in data.columns for col in cols):
                error_msg = f"Colonnes {cols} manquantes"
                miya_alerts(
                    error_msg, tag="VOLATILITY", voice_profile="warning", priority=3
                )
                send_alert(error_msg, priority=3)
                logger.warning(error_msg)
                return pd.Series(0.0, index=data.index, name="vol_term_curvature")
            curvature = (
                2 * data[f"vix_term_{months[1]}m"]
                - data[f"vix_term_{months[0]}m"]
                - data[f"vix_term_{months[2]}m"]
            )
            result = pd.Series(curvature, index=data.index).fillna(0.0)
            result.name = "vol_term_curvature"
            latency = time.time() - start_time
            self.log_performance("compute_vol_term_curvature", latency, success=True)
            return result
        except Exception as e:
            self.log_performance(
                "compute_vol_term_curvature", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_vol_term_curvature: {str(e)}",
                tag="VOLATILITY",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_vol_term_curvature: {str(e)}", priority=4)
            logger.error(f"Erreur dans compute_vol_term_curvature: {str(e)}")
            return pd.Series(0.0, index=data.index, name="vol_term_curvature")

    def compute_realized_vol(
        self, data: pd.DataFrame, window: str = "5min"
    ) -> pd.Series:
        """
        Calcule la volatilité réalisée sur une fenêtre temporelle.
        """
        try:
            start_time = time.time()
            if "close" not in data.columns or "timestamp" not in data.columns:
                error_msg = "Colonnes close ou timestamp manquantes"
                miya_alerts(
                    error_msg, tag="VOLATILITY", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=data.index, name=f"realized_vol_{window}")
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            returns = data["close"].pct_change().fillna(0)
            vol = returns.groupby(
                pd.Grouper(key="timestamp", freq=window)
            ).std() * np.sqrt(252 * 12 * 60 / 5)
            vol = vol.reindex(data.index, method="ffill").fillna(0.0)
            result = pd.Series(vol, index=data.index)
            result.name = f"realized_vol_{window}"
            latency = time.time() - start_time
            self.log_performance(
                f"compute_realized_vol_{window}", latency, success=True
            )
            return result
        except Exception as e:
            self.log_performance(
                f"compute_realized_vol_{window}", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_realized_vol_{window}: {str(e)}",
                tag="VOLATILITY",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_realized_vol_{window}: {str(e)}", priority=4
            )
            logger.error(f"Erreur dans compute_realized_vol_{window}: {str(e)}")
            return pd.Series(0.0, index=data.index, name=f"realized_vol_{window}")

    def compute_volatility_skew(
        self, data: pd.DataFrame, window: str = "15min"
    ) -> pd.Series:
        """
        Calcule l’asymétrie (skew) de la volatilité implicite.
        """
        try:
            start_time = time.time()
            if "iv_atm" not in data.columns or "timestamp" not in data.columns:
                error_msg = "Colonnes iv_atm ou timestamp manquantes"
                miya_alerts(
                    error_msg, tag="VOLATILITY", voice_profile="warning", priority=3
                )
                send_alert(error_msg, priority=3)
                logger.warning(error_msg)
                return pd.Series(0.0, index=data.index, name="volatility_skew_15m")
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            skew = (
                data["iv_atm"]
                .groupby(pd.Grouper(key="timestamp", freq=window))
                .skew()
                .reindex(data.index, method="ffill")
                .fillna(0.0)
            )
            result = pd.Series(skew, index=data.index)
            result.name = "volatility_skew_15m"
            latency = time.time() - start_time
            self.log_performance("compute_volatility_skew", latency, success=True)
            return result
        except Exception as e:
            self.log_performance(
                "compute_volatility_skew", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_volatility_skew: {str(e)}",
                tag="VOLATILITY",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_volatility_skew: {str(e)}", priority=4)
            logger.error(f"Erreur dans compute_volatility_skew: {str(e)}")
            return pd.Series(0.0, index=data.index, name="volatility_skew_15m")

    def compute_volatility_breakout_signal(self, data: pd.DataFrame) -> pd.Series:
        """
        Génère un signal de breakout de volatilité basé sur ATR.
        """
        try:
            start_time = time.time()
            if "atr_14" not in data.columns:
                error_msg = "Colonne atr_14 manquante"
                miya_alerts(
                    error_msg, tag="VOLATILITY", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(
                    0.0, index=data.index, name="volatility_breakout_signal"
                )
            mean_atr = data["atr_14"].rolling(window=20, min_periods=1).mean()
            std_atr = data["atr_14"].rolling(window=20, min_periods=1).std()
            signal = (data["atr_14"] > (mean_atr + 2 * std_atr)).astype(int)
            result = pd.Series(signal, index=data.index).fillna(0.0)
            result.name = "volatility_breakout_signal"
            latency = time.time() - start_time
            self.log_performance(
                "compute_volatility_breakout_signal", latency, success=True
            )
            return result
        except Exception as e:
            self.log_performance(
                "compute_volatility_breakout_signal", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_volatility_breakout_signal: {str(e)}",
                tag="VOLATILITY",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_volatility_breakout_signal: {str(e)}", priority=4
            )
            logger.error(f"Erreur dans compute_volatility_breakout_signal: {str(e)}")
            return pd.Series(0.0, index=data.index, name="volatility_breakout_signal")

    def compute_session_volatility_index(self, data: pd.DataFrame) -> pd.Series:
        """
        Calcule l’indice de volatilité par session horaire.
        """
        try:
            start_time = time.time()
            if "atr_14" not in data.columns or "timestamp" not in data.columns:
                error_msg = "Colonnes atr_14 ou timestamp manquantes"
                miya_alerts(
                    error_msg, tag="VOLATILITY", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=data.index, name="session_volatility_index")
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            session_index = data["atr_14"].groupby(data["timestamp"].dt.hour).mean()
            session_index = session_index.reindex(data["timestamp"].dt.hour).fillna(0.0)
            result = pd.Series(session_index.values, index=data.index)
            result.name = "session_volatility_index"
            latency = time.time() - start_time
            self.log_performance(
                "compute_session_volatility_index", latency, success=True
            )
            return result
        except Exception as e:
            self.log_performance(
                "compute_session_volatility_index", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_session_volatility_index: {str(e)}",
                tag="VOLATILITY",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_session_volatility_index: {str(e)}", priority=4
            )
            logger.error(f"Erreur dans compute_session_volatility_index: {str(e)}")
            return pd.Series(0.0, index=data.index, name="session_volatility_index")

    def compute_seasonal_volatility_index(self, data: pd.DataFrame) -> pd.Series:
        """
        Calcule l’indice de volatilité saisonnier par mois.
        """
        try:
            start_time = time.time()
            if "atr_14" not in data.columns or "timestamp" not in data.columns:
                error_msg = "Colonnes atr_14 ou timestamp manquantes"
                miya_alerts(
                    error_msg, tag="VOLATILITY", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(
                    0.0, index=data.index, name="seasonal_volatility_index"
                )
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            seasonal_index = data["atr_14"].groupby(data["timestamp"].dt.month).mean()
            seasonal_index = seasonal_index.reindex(data["timestamp"].dt.month).fillna(
                0.0
            )
            result = pd.Series(seasonal_index.values, index=data.index)
            result.name = "seasonal_volatility_index"
            latency = time.time() - start_time
            self.log_performance(
                "compute_seasonal_volatility_index", latency, success=True
            )
            return result
        except Exception as e:
            self.log_performance(
                "compute_seasonal_volatility_index", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_seasonal_volatility_index: {str(e)}",
                tag="VOLATILITY",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_seasonal_volatility_index: {str(e)}", priority=4
            )
            logger.error(f"Erreur dans compute_seasonal_volatility_index: {str(e)}")
            return pd.Series(0.0, index=data.index, name="seasonal_volatility_index")

    def compute_session_volatility(
        self, data: pd.DataFrame, period: str = "open"
    ) -> pd.Series:
        """
        Calcule la volatilité pour une session spécifique (open/close).
        """
        try:
            start_time = time.time()
            if "close" not in data.columns or "timestamp" not in data.columns:
                error_msg = "Colonnes close ou timestamp manquantes"
                miya_alerts(
                    error_msg, tag="VOLATILITY", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(
                    0.0, index=data.index, name=f"market_{period}_volatility"
                )
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            if period == "open":
                session_data = data[
                    data["timestamp"].dt.time.between(
                        pd.Timestamp("09:30").time(), pd.Timestamp("10:30").time()
                    )
                ]
            elif period == "close":
                session_data = data[
                    data["timestamp"].dt.time.between(
                        pd.Timestamp("15:00").time(), pd.Timestamp("16:00").time()
                    )
                ]
            else:
                error_msg = f"Période invalide: {period}, attendu: 'open' ou 'close'"
                miya_alerts(
                    error_msg, tag="VOLATILITY", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(
                    0.0, index=data.index, name=f"market_{period}_volatility"
                )
            vol = (
                session_data["close"].pct_change().std() * np.sqrt(252 * 12 * 60 / 5)
                if not session_data.empty
                else 0.0
            )
            result = pd.Series(vol, index=data.index).fillna(0.0)
            result.name = f"market_{period}_volatility"
            latency = time.time() - start_time
            self.log_performance(
                f"compute_session_volatility_{period}", latency, success=True
            )
            return result
        except Exception as e:
            self.log_performance(
                f"compute_session_volatility_{period}", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_session_volatility_{period}: {str(e)}",
                tag="VOLATILITY",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_session_volatility_{period}: {str(e)}", priority=4
            )
            logger.error(f"Erreur dans compute_session_volatility_{period}: {str(e)}")
            return pd.Series(0.0, index=data.index, name=f"market_{period}_volatility")

    def normalize_metrics(self, data: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
        """
        Normalise les métriques avec MinMaxScaler sur une fenêtre glissante pour l’inférence.
        """
        try:
            start_time = time.time()
            for col in metrics:
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
                        tag="VOLATILITY",
                        voice_profile="warning",
                        priority=3,
                    )
                    send_alert(
                        f"Colonne {col} manquante pour normalisation", priority=3
                    )
                    logger.warning(f"Colonne {col} manquante pour normalisation")
            latency = time.time() - start_time
            self.log_performance("normalize_metrics", latency, success=True)
            return data
        except Exception as e:
            self.log_performance("normalize_metrics", 0, success=False, error=str(e))
            miya_alerts(
                f"Erreur dans normalize_metrics: {str(e)}",
                tag="VOLATILITY",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans normalize_metrics: {str(e)}", priority=4)
            logger.error(f"Erreur dans normalize_metrics: {str(e)}")
            return data

    def plot_metrics(self, metrics: pd.DataFrame, timestamp: str) -> None:
        """
        Génère des visualisations des métriques de volatilité.
        """
        try:
            start_time = time.time()
            timestamp_safe = timestamp.replace(":", "-")
            plt.figure(figsize=(12, 6))
            plt.plot(
                metrics["timestamp"],
                metrics["bollinger_width_20"],
                label="Bollinger Width",
                color="blue",
            )
            plt.plot(
                metrics["timestamp"],
                metrics["vix_es_correlation"],
                label="VIX/ES Correlation",
                color="orange",
            )
            plt.plot(
                metrics["timestamp"],
                metrics["volatility_premium"],
                label="Volatility Premium",
                color="green",
            )
            plt.plot(
                metrics["timestamp"],
                metrics["implied_move_ratio"],
                label="Implied Move Ratio",
                color="red",
            )
            plt.title(f"Volatility Metrics - {timestamp}")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.savefig(
                os.path.join(FIGURES_DIR, f"volatility_metrics_{timestamp_safe}.png")
            )
            plt.close()
            latency = time.time() - start_time
            miya_speak(
                f"Visualisations générées: {FIGURES_DIR}",
                tag="VOLATILITY",
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
                tag="VOLATILITY",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur génération visualisations: {str(e)}", priority=4)
            logger.error(f"Erreur génération visualisations: {str(e)}")

    def calculate_volatility_metrics(
        self,
        data: pd.DataFrame,
        time_col: str = "timestamp",
        atr_multiplier: float = 1.5,
        volatility_trend_window: int = 10,
        bollinger_window: int = 20,
        stability_window: int = 20,
    ) -> pd.DataFrame:
        """
        Calcule les métriques de volatilité à partir des données IQFeed.

        Args:
            data (pd.DataFrame): Données contenant les prix (high, low, close, atr_14) et timestamp.
            time_col (str, optional): Nom de la colonne temporelle (par défaut : 'timestamp').
            atr_multiplier (float, optional): Multiplicateur pour range_threshold (par défaut : 1.5).
            volatility_trend_window (int, optional): Fenêtre pour volatility_trend (par défaut : 10).
            bollinger_window (int, optional): Fenêtre pour les bandes de Bollinger (par défaut : 20).
            stability_window (int, optional): Fenêtre pour la stabilité du régime (par défaut : 20).

        Returns:
            pd.DataFrame: Données enrichies avec les métriques de volatilité et leurs versions normalisées.

        Raises:
            ValueError: Si les données sont vides ou mal formées.
        """
        try:
            start_time = time.time()
            config = self.load_config_with_manager()

            if data.empty:
                error_msg = "DataFrame vide dans calculate_volatility_metrics"
                miya_alerts(
                    error_msg, tag="VOLATILITY", voice_profile="urgent", priority=5
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                raise ValueError(error_msg)

            if not self.validate_iqfeed_data(data):
                error_msg = "Échec validation données IQFeed"
                miya_alerts(
                    error_msg, tag="VOLATILITY", voice_profile="urgent", priority=5
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
                        "Métriques récupérées du cache",
                        tag="VOLATILITY",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert("Métriques récupérées du cache", priority=1)
                    self.log_performance(
                        "calculate_volatility_metrics_cache_hit", 0, success=True
                    )
                    return cached_data

            data = data.copy()
            if time_col not in data.columns:
                miya_speak(
                    f"Colonne '{time_col}' manquante, création par défaut",
                    tag="VOLATILITY",
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
                    tag="VOLATILITY",
                    voice_profile="calm",
                    priority=1,
                )
                send_alert("Timestamp par défaut ajouté", priority=1)

            data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
            if data[time_col].isna().any():
                miya_speak(
                    f"NaN dans '{time_col}', imputés avec la première date valide",
                    tag="VOLATILITY",
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

            required_cols = ["high", "low", "close", "atr_14"]
            optional_cols = [
                "vix",
                "es",
                "iv_30d",
                "realized_vol_30d",
                "iv_atm",
                "vix_term_1m",
                "vix_term_3m",
                "vix_term_6m",
            ]
            missing_cols = [col for col in required_cols if col not in data.columns]
            for col in missing_cols:
                if col in ["high", "low", "close"]:
                    data[col] = (
                        data["close"].median() if "close" in data.columns else 0.0
                    )
                    miya_speak(
                        f"Colonne IQFeed '{col}' manquante, imputée à la médiane de close ou 0",
                        tag="VOLATILITY",
                        voice_profile="warning",
                        priority=3,
                    )
                    send_alert(f"Colonne IQFeed '{col}' manquante", priority=3)
                    logger.warning(
                        f"Colonne manquante: {col}, imputée à la médiane de close ou 0"
                    )
                else:
                    data[col] = pd.Series(0.0, index=data.index)
                    miya_speak(
                        f"Colonne IQFeed '{col}' manquante, imputée à 0",
                        tag="VOLATILITY",
                        voice_profile="warning",
                        priority=3,
                    )
                    send_alert(f"Colonne IQFeed '{col}' manquante", priority=3)
                    logger.warning(f"Colonne manquante: {col}")

            for col in optional_cols:
                if col not in data.columns:
                    data[col] = pd.Series(0.0, index=data.index)
                    miya_speak(
                        f"Colonne optionnelle IQFeed '{col}' manquante, imputée à 0",
                        tag="VOLATILITY",
                        voice_profile="warning",
                        priority=3,
                    )
                    send_alert(
                        f"Colonne optionnelle IQFeed '{col}' manquante", priority=3
                    )
                    logger.warning(f"Colonne manquante: {col}")

            for col in required_cols + optional_cols:
                data[col] = pd.to_numeric(data[col], errors="coerce")
                if data[col].isna().any():
                    data[col] = data[col].fillna(0.0)
                    miya_speak(
                        f"NaN dans {col}, imputés à 0",
                        tag="VOLATILITY",
                        voice_profile="warning",
                        priority=3,
                    )
                    send_alert(f"NaN dans {col}", priority=3)
                    logger.warning(f"NaN dans {col}, imputés à 0")

            if data["atr_14"].eq(0).all():
                data["atr_14"] = ta.volatility.AverageTrueRange(
                    high=data["high"],
                    low=data["low"],
                    close=data["close"],
                    window=14,
                    fillna=True,
                ).average_true_range()
                miya_speak(
                    "ATR_14 recalculé car absent ou nul",
                    tag="VOLATILITY",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert("ATR_14 recalculé", priority=3)
                logger.warning("ATR_14 recalculé car absent ou nul")

            data["range_threshold"] = data["atr_14"] * atr_multiplier
            data["range_threshold"] = data["range_threshold"].fillna(0.0)
            data["volatility_trend"] = (
                data["atr_14"]
                .pct_change()
                .rolling(volatility_trend_window, min_periods=1)
                .mean()
                .fillna(0.0)
            )
            bb = ta.volatility.BollingerBands(
                close=data["close"], window=bollinger_window, fillna=True
            )
            data["bollinger_upper_20"] = bb.bollinger_hband()
            data["bollinger_lower_20"] = bb.bollinger_lband()
            data["bollinger_width_20"] = (
                data["bollinger_upper_20"] - data["bollinger_lower_20"]
            ) / data["close"].replace(0, 1e-6)
            data["bollinger_width_20"] = data["bollinger_width_20"].fillna(0.0)
            data["volatility_regime_stability"] = (
                data["atr_14"]
                .rolling(stability_window, min_periods=1)
                .std()
                .fillna(0.0)
            )
            data["vix_es_correlation"] = self.calculate_vix_es_correlation(
                data, window=bollinger_window
            )

            data["volatility_premium"] = self.compute_volatility_premium(data)
            data["implied_move_ratio"] = self.compute_implied_move_ratio(data)
            data["vol_term_slope"] = self.compute_vol_term_slope(data, months=[1, 3])
            data["vol_term_curvature"] = self.compute_vol_term_curvature(
                data, months=[1, 3, 6]
            )
            data["realized_vol_5m"] = self.compute_realized_vol(data, window="5min")
            data["realized_vol_15m"] = self.compute_realized_vol(data, window="15min")
            data["realized_vol_30m"] = self.compute_realized_vol(data, window="30min")
            data["volatility_skew_15m"] = self.compute_volatility_skew(
                data, window="15min"
            )
            data["volatility_breakout_signal"] = (
                self.compute_volatility_breakout_signal(data)
            )
            data["session_volatility_index"] = self.compute_session_volatility_index(
                data
            )
            data["seasonal_volatility_index"] = self.compute_seasonal_volatility_index(
                data
            )
            data["market_open_volatility"] = self.compute_session_volatility(
                data, period="open"
            )
            data["market_close_volatility"] = self.compute_session_volatility(
                data, period="close"
            )

            metrics = [
                "range_threshold",
                "volatility_trend",
                "bollinger_width_20",
                "volatility_regime_stability",
                "vix_es_correlation",
                "volatility_premium",
                "implied_move_ratio",
                "vol_term_slope",
                "vol_term_curvature",
                "realized_vol_5m",
                "realized_vol_15m",
                "realized_vol_30m",
                "volatility_skew_15m",
                "volatility_breakout_signal",
                "session_volatility_index",
                "seasonal_volatility_index",
                "market_open_volatility",
                "market_close_volatility",
            ]
            self.validate_shap_features(metrics)
            data = self.normalize_metrics(data, metrics)
            data = data.drop(
                columns=["bollinger_upper_20", "bollinger_lower_20"], errors="ignore"
            )
            self.cache_metrics(data, cache_key)
            self.plot_metrics(data, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            latency = time.time() - start_time
            miya_speak(
                "Métriques de volatilité calculées",
                tag="VOLATILITY",
                voice_profile="calm",
                priority=1,
            )
            send_alert("Métriques de volatilité calculées", priority=1)
            logger.info("Métriques de volatilité calculées")
            self.log_performance(
                "calculate_volatility_metrics",
                latency,
                success=True,
                num_rows=len(data),
                num_metrics=len(metrics),
            )
            self.save_snapshot(
                "calculate_volatility_metrics",
                {"num_rows": len(data), "metrics": metrics},
            )
            return data
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans calculate_volatility_metrics: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="VOLATILITY", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=4)
            logger.error(error_msg)
            self.log_performance(
                "calculate_volatility_metrics", latency, success=False, error=str(e)
            )
            self.save_snapshot("calculate_volatility_metrics", {"error": str(e)})
            data["timestamp"] = pd.date_range(
                start=pd.Timestamp.now(), periods=len(data), freq="1min"
            )
            for col in [
                "range_threshold",
                "volatility_trend",
                "bollinger_width_20",
                "volatility_regime_stability",
                "vix_es_correlation",
                "volatility_premium",
                "implied_move_ratio",
                "vol_term_slope",
                "vol_term_curvature",
                "realized_vol_5m",
                "realized_vol_15m",
                "realized_vol_30m",
                "volatility_skew_15m",
                "volatility_breakout_signal",
                "session_volatility_index",
                "seasonal_volatility_index",
                "market_open_volatility",
                "market_close_volatility",
                "range_threshold_normalized",
                "volatility_trend_normalized",
                "bollinger_width_20_normalized",
                "volatility_regime_stability_normalized",
                "vix_es_correlation_normalized",
                "volatility_premium_normalized",
                "implied_move_ratio_normalized",
                "vol_term_slope_normalized",
                "vol_term_curvature_normalized",
                "realized_vol_5m_normalized",
                "realized_vol_15m_normalized",
                "realized_vol_30m_normalized",
                "volatility_skew_15m_normalized",
                "volatility_breakout_signal_normalized",
                "session_volatility_index_normalized",
                "seasonal_volatility_index_normalized",
                "market_open_volatility_normalized",
                "market_close_volatility_normalized",
            ]:
                data[col] = 0.0
            return data


if __name__ == "__main__":
    try:
        calculator = VolatilityMetricsCalculator()
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "high": np.random.normal(5105, 10, 100),
                "low": np.random.normal(5095, 10, 100),
                "close": np.random.normal(5100, 10, 100),
                "atr_14": np.random.uniform(0.5, 2.0, 100),
                "vix": np.random.normal(20, 2, 100),
                "es": np.random.normal(5100, 10, 100),
                "iv_30d": np.random.uniform(0.15, 0.25, 100),
                "realized_vol_30d": np.random.uniform(0.1, 0.2, 100),
                "iv_atm": np.random.uniform(0.15, 0.25, 100),
                "vix_term_1m": np.random.normal(20, 2, 100),
                "vix_term_3m": np.random.normal(21, 2, 100),
                "vix_term_6m": np.random.normal(22, 2, 100),
            }
        )
        result = calculator.calculate_volatility_metrics(data)
        print(
            result[
                [
                    "timestamp",
                    "range_threshold",
                    "bollinger_width_20",
                    "vix_es_correlation",
                    "volatility_premium",
                    "implied_move_ratio",
                    "range_threshold_normalized",
                ]
            ].head()
        )
        miya_speak(
            "Test calculate_volatility_metrics terminé",
            tag="TEST",
            voice_profile="calm",
            priority=1,
        )
        send_alert("Test calculate_volatility_metrics terminé", priority=1)
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
