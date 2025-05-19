# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/features/extractors/options_metrics.py
# Calcule les indicateurs d’options (iv_atm, gex_slope, gamma_peak_distance, etc.) à partir des données IQFeed.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Génère 18 métriques d’options (ex. : iv_atm, gex_slope, gamma_peak_distance) à partir des données IQFeed,
#        intègre SHAP (méthode 17), et enregistre logs psutil. Conforme aux Phases 13 (options_metrics.py) et 14.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.0,<6.0.0, matplotlib>=3.7.0,<4.0.0, json, gzip, hashlib, logging
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
#
# Inputs :
# - config/es_config.yaml
# - data/iqfeed/options_data.csv
# - data/features/feature_importance.csv
#
# Outputs :
# - data/features/cache/options_metrics/
# - data/logs/options_metrics_performance.csv
# - data/features/options_metrics_snapshots/
# - data/figures/options_metrics/
#
# Notes :
# - Utilise exclusivement IQFeed pour les données (remplace dxFeed).
# - Suppression de toute référence à dxFeed, obs_t, 320 features, et 81 features.
# - Intègre les Phases 1-18 :
#   - Phase 13 (orderflow_indicators.py, options_metrics.py) : Métriques iv_atm, gex_slope, etc.
#   - Phase 14 : Calcul des métriques d’options contextuelles.
# - Tests unitaires disponibles dans tests/test_options_metrics.py.
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
CACHE_DIR = os.path.join(BASE_DIR, "data", "features", "cache", "options_metrics")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "features", "options_metrics_snapshots")
CSV_LOG_PATH = os.path.join(BASE_DIR, "data", "logs", "options_metrics_performance.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures", "options_metrics")
FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR, "data", "features", "feature_importance.csv"
)
OPTIONS_DATA_PATH = os.path.join(BASE_DIR, "data", "iqfeed", "options_data.csv")

# Création des dossiers
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data", "logs"), exist_ok=True)

# Configuration du logging
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "data", "logs", "options_metrics.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 2.0
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes


class OptionsMetricsGenerator:

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
                "OptionsMetricsGenerator initialisé",
                tag="OPTIONS_METRICS",
                voice_profile="calm",
                priority=2,
            )
            send_alert("OptionsMetricsGenerator initialisé", priority=1)
            logger.info("OptionsMetricsGenerator initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config_path": config_path})
        except Exception as e:
            error_msg = f"Erreur initialisation OptionsMetricsGenerator: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="OPTIONS_METRICS", voice_profile="urgent", priority=3
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
            if "options_metrics_generator" not in config:
                raise ValueError(
                    "Clé 'options_metrics_generator' manquante dans la configuration"
                )
            required_keys = ["buffer_size", "max_cache_size", "cache_hours"]
            missing_keys = [
                key
                for key in required_keys
                if key not in config["options_metrics_generator"]
            ]
            if missing_keys:
                raise ValueError(
                    f"Clés manquantes dans 'options_metrics_generator': {missing_keys}"
                )
            return config["options_metrics_generator"]

        try:
            start_time = time.time()
            config = self.with_retries(load_yaml)
            latency = time.time() - start_time
            miya_speak(
                "Configuration options_metrics_generator chargée via config_manager",
                tag="OPTIONS_METRICS",
                voice_profile="calm",
                priority=2,
            )
            send_alert(
                "Configuration options_metrics_generator chargée via config_manager",
                priority=1,
            )
            logger.info(
                "Configuration options_metrics_generator chargée via config_manager"
            )
            self.log_performance("load_config_with_manager", latency, success=True)
            self.save_snapshot("load_config_with_manager", {"config_path": config_path})
            return config
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur chargement config via config_manager: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="OPTIONS_METRICS", voice_profile="urgent", priority=3
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
                    tag="OPTIONS_METRICS",
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
                tag="OPTIONS_METRICS",
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
                tag="OPTIONS_METRICS",
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
                tag="OPTIONS_METRICS",
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
                tag="OPTIONS_METRICS",
                voice_profile="calm",
                priority=2,
            )
            send_alert("SIGINT received, snapshot saved", priority=2)
            logger.info("SIGINT received, snapshot saved")
        except Exception as e:
            miya_alerts(
                f"Erreur gestion SIGINT: {str(e)}",
                tag="OPTIONS_METRICS",
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
                    tag="OPTIONS_METRICS",
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
                    tag="OPTIONS_METRICS",
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
                    tag="OPTIONS_METRICS",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(
                    f"Features non incluses dans top 150 SHAP: {missing}", priority=3
                )
                logger.warning(f"Features non incluses dans top 150 SHAP: {missing}")
            miya_speak(
                "SHAP features validées",
                tag="OPTIONS_METRICS",
                voice_profile="calm",
                priority=1,
            )
            send_alert("SHAP features validées", priority=1)
            logger.info("SHAP features validées")
            return True
        except Exception as e:
            miya_alerts(
                f"Erreur validation SHAP features: {str(e)}",
                tag="OPTIONS_METRICS",
                voice_profile="urgent",
                priority=4,
            )
            send_alert(f"Erreur validation SHAP features: {str(e)}", priority=4)
            logger.error(f"Erreur validation SHAP features: {str(e)}")
            return False

    def calculate_iv_atm(
        self, option_chain: pd.DataFrame, underlying_price: float
    ) -> float:
        """
        Calcule la volatilité implicite ATM (At-The-Money) à partir de la chaîne d'options.

        Args:
            option_chain (pd.DataFrame): Données de la chaîne d'options (strike, implied_volatility, option_type).
            underlying_price (float): Prix actuel du sous-jacent.

        Returns:
            float: Volatilité implicite moyenne des options ATM (call et put).
        """
        try:
            start_time = time.time()
            option_chain["strike_distance"] = np.abs(
                option_chain["strike"] - underlying_price
            )
            option_chain["strike_distance"].idxmin()
            atm_options = option_chain.loc[
                option_chain["strike_distance"] == option_chain["strike_distance"].min()
            ]
            iv_call = atm_options[atm_options["option_type"] == "call"][
                "implied_volatility"
            ].mean()
            iv_put = atm_options[atm_options["option_type"] == "put"][
                "implied_volatility"
            ].mean()
            iv_atm = np.nanmean([iv_call, iv_put])
            result = iv_atm if not np.isnan(iv_atm) else 0.0
            latency = time.time() - start_time
            self.log_performance("calculate_iv_atm", latency, success=True)
            return result
        except Exception as e:
            self.log_performance("calculate_iv_atm", 0, success=False, error=str(e))
            miya_alerts(
                f"Erreur dans calculate_iv_atm: {str(e)}",
                tag="OPTIONS_METRICS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans calculate_iv_atm: {str(e)}", priority=4)
            logger.error(f"Erreur dans calculate_iv_atm: {str(e)}")
            return 0.0

    def calculate_gex_slope(
        self, option_chain: pd.DataFrame, underlying_price: float
    ) -> float:
        """
        Calcule la pente de l'exposition gamma (GEX) par rapport au prix sous-jacent.

        Args:
            option_chain (pd.DataFrame): Données de la chaîne d'options (strike, gamma, open_interest).
            underlying_price (float): Prix actuel du sous-jacent.

        Returns:
            float: Pente de l'exposition gamma.
        """
        try:
            start_time = time.time()
            option_chain["gex"] = option_chain["gamma"] * option_chain["open_interest"]
            gex_by_strike = option_chain.groupby("strike")["gex"].sum().reset_index()
            if len(gex_by_strike) < 2:
                return 0.0
            gex_by_strike = gex_by_strike.sort_values("strike")
            delta_gex = gex_by_strike["gex"].diff()
            delta_strike = gex_by_strike["strike"].diff()
            slope = (delta_gex / delta_strike).mean()
            result = slope if not np.isnan(slope) else 0.0
            latency = time.time() - start_time
            self.log_performance("calculate_gex_slope", latency, success=True)
            return result
        except Exception as e:
            self.log_performance("calculate_gex_slope", 0, success=False, error=str(e))
            miya_alerts(
                f"Erreur dans calculate_gex_slope: {str(e)}",
                tag="OPTIONS_METRICS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans calculate_gex_slope: {str(e)}", priority=4)
            logger.error(f"Erreur dans calculate_gex_slope: {str(e)}")
            return 0.0

    def calculate_gamma_peak_distance(
        self, option_chain: pd.DataFrame, underlying_price: float
    ) -> float:
        """
        Calcule la distance entre le prix actuel et le strike avec la plus forte concentration de gamma.

        Args:
            option_chain (pd.DataFrame): Données de la chaîne d'options (strike, gamma, open_interest).
            underlying_price (float): Prix actuel du sous-jacent.

        Returns:
            float: Distance entre le prix actuel et le strike de gamma maximale.
        """
        try:
            start_time = time.time()
            option_chain["gex"] = option_chain["gamma"] * option_chain["open_interest"]
            gamma_peak = option_chain.groupby("strike")["gex"].sum().idxmax()
            distance = underlying_price - gamma_peak
            result = distance if not np.isnan(distance) else 0.0
            latency = time.time() - start_time
            self.log_performance("calculate_gamma_peak_distance", latency, success=True)
            return result
        except Exception as e:
            self.log_performance(
                "calculate_gamma_peak_distance", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans calculate_gamma_peak_distance: {str(e)}",
                tag="OPTIONS_METRICS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans calculate_gamma_peak_distance: {str(e)}", priority=4
            )
            logger.error(f"Erreur dans calculate_gamma_peak_distance: {str(e)}")
            return 0.0

    def compute_options_metrics(
        self, data: pd.DataFrame, option_chain: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calcule les indicateurs d’options à partir des données IQFeed et enrichit le DataFrame.

        Args:
            data (pd.DataFrame): Données contenant le prix sous-jacent (close) et timestamp.
            option_chain (pd.DataFrame): Données de la chaîne d'options avec timestamp, underlying_price, strike, etc.

        Returns:
            pd.DataFrame: Données enrichies avec 18 métriques d’options.
        """
        try:
            start_time = time.time()
            config = self.load_config_with_manager()

            if data.empty:
                error_msg = "DataFrame principal vide"
                miya_alerts(
                    error_msg, tag="OPTIONS_METRICS", voice_profile="urgent", priority=5
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
                        "Métriques d’options récupérées du cache",
                        tag="OPTIONS_METRICS",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert("Métriques d’options récupérées du cache", priority=1)
                    self.log_performance(
                        "compute_options_metrics_cache_hit", 0, success=True
                    )
                    return cached_data

            data = data.copy()
            required_data_cols = ["timestamp", "close"]
            missing_data_cols = [
                col for col in required_data_cols if col not in data.columns
            ]
            if missing_data_cols:
                error_msg = f"Colonnes manquantes dans data: {missing_data_cols}"
                miya_alerts(
                    error_msg, tag="OPTIONS_METRICS", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                raise ValueError(error_msg)

            data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
            if data["timestamp"].isna().any():
                miya_speak(
                    "NaN dans timestamp, imputés avec la première date valide",
                    tag="OPTIONS_METRICS",
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

            if option_chain.empty:
                error_msg = "DataFrame option_chain vide"
                miya_alerts(
                    error_msg, tag="OPTIONS_METRICS", voice_profile="urgent", priority=5
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                raise ValueError(error_msg)

            required_option_cols = [
                "timestamp",
                "underlying_price",
                "strike",
                "option_type",
                "implied_volatility",
                "open_interest",
                "gamma",
                "expiration_date",
            ]
            missing_option_cols = [
                col for col in required_option_cols if col not in option_chain.columns
            ]
            if missing_option_cols:
                error_msg = (
                    f"Colonnes manquantes dans option_chain: {missing_option_cols}"
                )
                miya_alerts(
                    error_msg, tag="OPTIONS_METRICS", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                raise ValueError(error_msg)

            option_chain["timestamp"] = pd.to_datetime(
                option_chain["timestamp"], errors="coerce"
            )
            if option_chain["timestamp"].isna().any():
                miya_speak(
                    "NaN dans timestamp d’option_chain, imputés avec la première date valide",
                    tag="OPTIONS_METRICS",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert("NaN dans timestamp d’option_chain", priority=3)
                logger.warning("NaN dans timestamp d’option_chain, imputation")
                first_valid_time = (
                    option_chain["timestamp"].dropna().iloc[0]
                    if not option_chain["timestamp"].dropna().empty
                    else pd.Timestamp.now()
                )
                option_chain["timestamp"] = option_chain["timestamp"].fillna(
                    first_valid_time
                )

            features = [
                "iv_atm",
                "gex_slope",
                "gamma_peak_distance",
                "iv_skew",
                "gex_total",
                "oi_peak_call",
                "oi_peak_put",
                "gamma_wall",
                "delta_exposure",
                "theta_pressure",
                "iv_slope",
                "call_put_ratio",
                "iv_atm_change",
                "gex_stability",
                "strike_density",
                "time_to_expiry",
                "iv_atm_call",
                "iv_atm_put",
            ]
            for feature in features:
                data[feature] = 0.0

            for idx, row in data.iterrows():
                timestamp = row["timestamp"]
                underlying_price = row["close"]
                current_chain = option_chain[option_chain["timestamp"] == timestamp]
                if current_chain.empty:
                    continue

                data.at[idx, "iv_atm"] = self.calculate_iv_atm(
                    current_chain, underlying_price
                )
                data.at[idx, "gex_slope"] = self.calculate_gex_slope(
                    current_chain, underlying_price
                )
                data.at[idx, "gamma_peak_distance"] = (
                    self.calculate_gamma_peak_distance(current_chain, underlying_price)
                )

                iv_calls = current_chain[current_chain["option_type"] == "call"][
                    "implied_volatility"
                ].mean()
                iv_puts = current_chain[current_chain["option_type"] == "put"][
                    "implied_volatility"
                ].mean()
                data.at[idx, "iv_skew"] = (
                    iv_calls - iv_puts if not np.isnan(iv_calls - iv_puts) else 0.0
                )

                data.at[idx, "gex_total"] = (
                    current_chain["gamma"] * current_chain["open_interest"]
                ).sum()

                calls = current_chain[current_chain["option_type"] == "call"]
                puts = current_chain[current_chain["option_type"] == "put"]
                data.at[idx, "oi_peak_call"] = (
                    calls.loc[calls["open_interest"].idxmax(), "strike"]
                    if not calls.empty
                    else 0
                )
                data.at[idx, "oi_peak_put"] = (
                    puts.loc[puts["open_interest"].idxmax(), "strike"]
                    if not puts.empty
                    else 0
                )

                gex_by_strike = current_chain.groupby("strike")["gex"].sum()
                data.at[idx, "gamma_wall"] = (
                    gex_by_strike.idxmax() if not gex_by_strike.empty else 0
                )

                current_chain["delta_exposure"] = (
                    current_chain["open_interest"]
                    * current_chain["gamma"]
                    * (current_chain["strike"] - underlying_price)
                )
                data.at[idx, "delta_exposure"] = current_chain["delta_exposure"].sum()

                current_chain["expiration_days"] = (
                    pd.to_datetime(current_chain["expiration_date"]) - timestamp
                ).dt.days
                current_chain["theta_pressure"] = current_chain["open_interest"] / (
                    current_chain["expiration_days"] + 1e-6
                )
                data.at[idx, "theta_pressure"] = current_chain["theta_pressure"].sum()

                iv_by_strike = (
                    current_chain.groupby("strike")["implied_volatility"]
                    .mean()
                    .reset_index()
                )
                if len(iv_by_strike) >= 2:
                    iv_by_strike = iv_by_strike.sort_values("strike")
                    delta_iv = iv_by_strike["implied_volatility"].diff()
                    delta_strike = iv_by_strike["strike"].diff()
                    data.at[idx, "iv_slope"] = (
                        (delta_iv / delta_strike).mean()
                        if not delta_strike.eq(0).all()
                        else 0.0
                    )

                oi_calls = calls["open_interest"].sum()
                oi_puts = puts["open_interest"].sum()
                data.at[idx, "call_put_ratio"] = (
                    oi_calls / (oi_puts + 1e-6) if oi_puts > 0 else 0.0
                )

                if idx > 0:
                    data.at[idx, "iv_atm_change"] = (
                        data.at[idx, "iv_atm"] - data.at[idx - 1, "iv_atm"]
                    )

                if idx >= 5:
                    recent_gex = data.loc[idx - 5 : idx, "gex_total"]
                    data.at[idx, "gex_stability"] = (
                        recent_gex.std() if not recent_gex.empty else 0.0
                    )

                strike_range = current_chain[
                    (current_chain["strike"] >= underlying_price - 50)
                    & (current_chain["strike"] <= underlying_price + 50)
                ]
                data.at[idx, "strike_density"] = len(strike_range["strike"].unique())

                if not current_chain["expiration_days"].empty:
                    data.at[idx, "time_to_expiry"] = current_chain[
                        "expiration_days"
                    ].mean()

                atm_options = current_chain.loc[
                    current_chain["strike_distance"]
                    == current_chain["strike_distance"].min()
                ]
                data.at[idx, "iv_atm_call"] = (
                    atm_options[atm_options["option_type"] == "call"][
                        "implied_volatility"
                    ].mean()
                    or 0.0
                )
                data.at[idx, "iv_atm_put"] = (
                    atm_options[atm_options["option_type"] == "put"][
                        "implied_volatility"
                    ].mean()
                    or 0.0
                )

            self.validate_shap_features(features)
            self.cache_metrics(data, cache_key)
            self.plot_metrics(data, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            latency = time.time() - start_time
            miya_speak(
                "Métriques d’options calculées",
                tag="OPTIONS_METRICS",
                voice_profile="calm",
                priority=1,
            )
            send_alert("Métriques d’options calculées", priority=1)
            logger.info("Métriques d’options calculées")
            self.log_performance(
                "compute_options_metrics",
                latency,
                success=True,
                num_rows=len(data),
                num_metrics=len(features),
            )
            self.save_snapshot(
                "compute_options_metrics", {"num_rows": len(data), "metrics": features}
            )
            return data
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans compute_options_metrics: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="OPTIONS_METRICS", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=4)
            logger.error(error_msg)
            self.log_performance(
                "compute_options_metrics", latency, success=False, error=str(e)
            )
            self.save_snapshot("compute_options_metrics", {"error": str(e)})
            for feature in features:
                data[feature] = 0.0
            return data

    def cache_metrics(self, metrics: pd.DataFrame, cache_key: str) -> None:
        """
        Met en cache les métriques d’options.
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
                tag="OPTIONS_METRICS",
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
                tag="OPTIONS_METRICS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur mise en cache métriques: {str(e)}", priority=4)
            logger.error(f"Erreur mise en cache métriques: {str(e)}")

    def plot_metrics(self, metrics: pd.DataFrame, timestamp: str) -> None:
        """
        Génère des visualisations des métriques d’options.

        Args:
            metrics (pd.DataFrame): Données avec les métriques.
            timestamp (str): Horodatage pour nommer le fichier.
        """
        try:
            start_time = time.time()
            timestamp_safe = timestamp.replace(":", "-")
            plt.figure(figsize=(12, 6))
            plt.plot(
                metrics["timestamp"], metrics["iv_atm"], label="IV ATM", color="blue"
            )
            plt.plot(
                metrics["timestamp"],
                metrics["gex_slope"],
                label="GEX Slope",
                color="orange",
            )
            plt.plot(
                metrics["timestamp"],
                metrics["gamma_peak_distance"],
                label="Gamma Peak Distance",
                color="green",
            )
            plt.title(f"Options Metrics - {timestamp}")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.savefig(
                os.path.join(FIGURES_DIR, f"options_metrics_{timestamp_safe}.png")
            )
            plt.close()
            latency = time.time() - start_time
            miya_speak(
                f"Visualisations générées: {FIGURES_DIR}",
                tag="OPTIONS_METRICS",
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
                tag="OPTIONS_METRICS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur génération visualisations: {str(e)}", priority=4)
            logger.error(f"Erreur génération visualisations: {str(e)}")


if __name__ == "__main__":
    try:
        generator = OptionsMetricsGenerator()
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-04-14 09:00", periods=5, freq="1min"),
                "close": [5100, 5102, 5098, 5105, 5100],
            }
        )
        option_chain = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2025-04-14 09:00"] * 6
                    + ["2025-04-14 09:01"] * 6
                    + ["2025-04-14 09:02"] * 6
                    + ["2025-04-14 09:03"] * 6
                    + ["2025-04-14 09:04"] * 6
                ),
                "underlying_price": [5100] * 6
                + [5102] * 6
                + [5098] * 6
                + [5105] * 6
                + [5100] * 6,
                "strike": [5080, 5090, 5100, 5110, 5120, 5130] * 5,
                "option_type": ["call", "call", "call", "put", "put", "put"] * 5,
                "implied_volatility": np.random.uniform(0.1, 0.3, 30),
                "open_interest": np.random.randint(100, 1000, 30),
                "gamma": np.random.uniform(0.01, 0.05, 30),
                "expiration_date": ["2025-04-21"] * 30,
            }
        )
        result = generator.compute_options_metrics(data, option_chain)
        print(
            result[["timestamp", "iv_atm", "gex_slope", "gamma_peak_distance"]].head()
        )
        miya_speak(
            "Test compute_options_metrics terminé",
            tag="TEST",
            voice_profile="calm",
            priority=1,
        )
        send_alert("Test compute_options_metrics terminé", priority=1)
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
