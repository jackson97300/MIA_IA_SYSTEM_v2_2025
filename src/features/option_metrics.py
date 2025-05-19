# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/features/option_metrics.py
# Calcule les indicateurs d’options (iv_atm, gex_slope, gamma_peak_distance, etc.) à partir des données IQFeed.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Calcule les métriques d’options, avec cache intermédiaire, logs psutil,
#        validation SHAP (méthode 17), et compatibilité top 150 SHAP features.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.0,<6.0.0, pyyaml>=6.0.0,<7.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
# - src/model/utils/obs_template.py
# - src/data/data_provider.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/es_config.yaml
# - data/iqfeed/merged_data.csv
# - data/options/option_chain.csv
# - data/features/feature_importance.csv
#
# Outputs :
# - data/features/cache/option_metrics/*.csv
# - data/logs/option_metrics_performance.csv
# - data/option_metrics_snapshots/*.json (option *.json.gz)
#
# Notes :
# - Conforme à structure.txt (version 2.1.3, 2025-05-13).
# - Utilise IQFeed via data_provider.py, avec retries (max 3, délai 2^attempt secondes).
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des métriques d’options.
# - Intègre validation SHAP (Phase 17) pour assurer la conformité avec les top 150 SHAP features.
# - Sauvegarde des snapshots JSON non compressés par défaut avec option de compression gzip.
# - Envoie des alertes via AlertManager et telegram_alert pour les erreurs critiques et succès.
# - Tests unitaires disponibles dans tests/test_option_metrics.py.
# - Conforme à la ligne rouge des top 150 SHAP features définie dans feature_pipeline.py.

import gzip
import hashlib
import json
import logging
import os
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import psutil
import yaml

from src.model.utils.alert_manager import send_alert
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.model.utils.obs_template import obs_t
from src.utils.telegram_alert import send_telegram_alert

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data", "features", "cache", "option_metrics")
PERF_LOG_PATH = os.path.join(BASE_DIR, "data", "logs", "option_metrics_performance.csv")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "option_metrics_snapshots")
FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR, "data", "features", "feature_importance.csv"
)

# Création des répertoires
os.makedirs(os.path.join(BASE_DIR, "data", "logs"), exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Configuration du logging
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "data", "logs", "option_metrics.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Constantes
MAX_RETRIES = 3
RETRY_DELAY = 2.0


class OptionMetrics:
    """Gère les métriques d’options avec cache, logs, et validation SHAP."""

    def __init__(
        self, config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml")
    ):
        """Initialise le générateur de métriques d’options."""
        self.log_buffer = []
        self.cache = {}
        try:
            self.config = self.load_config(config_path)
            os.makedirs(CACHE_DIR, exist_ok=True)
            os.makedirs(os.path.dirname(PERF_LOG_PATH), exist_ok=True)
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            miya_speak(
                "OptionMetrics initialisé",
                tag="OPTIONS",
                voice_profile="calm",
                priority=2,
            )
            send_alert("OptionMetrics initialisé", priority=2)
            send_telegram_alert("OptionMetrics initialisé")
            logger.info("OptionMetrics initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config_path": config_path}, compress=False)
        except Exception as e:
            error_msg = f"Erreur initialisation OptionMetrics: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="OPTIONS", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            self.config = {
                "window_size": 5,
                "min_option_rows": 10,
                "time_tolerance": "10s",
            }

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Charge la configuration depuis es_config.yaml."""

        def load_yaml():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            if "option_metrics" not in config:
                raise ValueError("Clé 'option_metrics' manquante dans la configuration")
            required_keys = ["window_size", "min_option_rows", "time_tolerance"]
            missing_keys = [
                key for key in required_keys if key not in config["option_metrics"]
            ]
            if missing_keys:
                raise ValueError(
                    f"Clés manquantes dans 'option_metrics': {missing_keys}"
                )
            return config["option_metrics"]

        try:
            start_time = time.time()
            config = self.with_retries(load_yaml)
            cache_key = hashlib.sha256(str(config).encode()).hexdigest()
            self.cache[cache_key] = config
            if len(self.cache) > self.config.get("max_cache_size", 1000):
                self.cache.pop(next(iter(self.cache)))
            latency = time.time() - start_time
            miya_speak(
                "Configuration option_metrics chargée",
                tag="OPTIONS",
                voice_profile="calm",
                priority=2,
            )
            send_alert("Configuration option_metrics chargée", priority=2)
            send_telegram_alert("Configuration option_metrics chargée")
            logger.info("Configuration option_metrics chargée")
            self.log_performance("load_config", latency, success=True)
            self.save_snapshot(
                "load_config", {"config_path": config_path}, compress=False
            )
            return config
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur chargement config: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="OPTIONS", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("load_config", latency, success=False, error=str(e))
            raise

    def with_retries(
        self,
        func: callable,
        max_attempts: int = MAX_RETRIES,
        delay_base: float = RETRY_DELAY,
    ) -> Any:
        """Exécute une fonction avec retries (max 3, délai exponentiel)."""
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
                    latency = time.time() - start_time
                    self.log_performance(
                        f"retry_attempt_{attempt+1}",
                        latency,
                        success=False,
                        error=str(e),
                        attempt_number=attempt + 1,
                    )
                    miya_alerts(
                        f"Échec après {max_attempts} tentatives: {str(e)}",
                        tag="OPTIONS",
                        voice_profile="urgent",
                        priority=4,
                    )
                    send_alert(
                        f"Échec après {max_attempts} tentatives: {str(e)}", priority=4
                    )
                    send_telegram_alert(
                        f"Échec après {max_attempts} tentatives: {str(e)}"
                    )
                    logger.error(f"Échec après {max_attempts} tentatives: {str(e)}")
                    raise
                delay = delay_base * (2**attempt)
                miya_speak(
                    f"Tentative {attempt+1} échouée, retry après {delay}s",
                    tag="OPTIONS",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(
                    f"Tentative {attempt+1} échouée, retry après {delay}s", priority=3
                )
                send_telegram_alert(
                    f"Tentative {attempt+1} échouée, retry après {delay}s"
                )
                logger.warning(f"Tentative {attempt+1} échouée, retry après {delay}s")
                time.sleep(delay)

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ) -> None:
        """Journalise les performances des opérations critiques avec psutil."""
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
            cpu_usage = psutil.cpu_percent()  # % CPU
            if memory_usage > 1024:
                miya_alerts(
                    f"ALERTE: Utilisation mémoire élevée ({memory_usage:.2f} MB)",
                    tag="OPTIONS",
                    voice_profile="urgent",
                    priority=5,
                )
                send_alert(
                    f"ALERTE: Utilisation mémoire élevée ({memory_usage:.2f} MB)",
                    priority=5,
                )
                send_telegram_alert(
                    f"ALERTE: Utilisation mémoire élevée ({memory_usage:.2f} MB)"
                )
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "latency": latency,
                "success": success,
                "error": error,
                "memory_usage_mb": memory_usage,
                "cpu_usage_percent": cpu_usage,
                **kwargs,
            }
            self.log_buffer.append(log_entry)
            if len(self.log_buffer) >= self.config.get("buffer_size", 100):
                log_df = pd.DataFrame(self.log_buffer)
                os.makedirs(os.path.dirname(PERF_LOG_PATH), exist_ok=True)

                def write_log():
                    if not os.path.exists(PERF_LOG_PATH):
                        log_df.to_csv(PERF_LOG_PATH, index=False, encoding="utf-8")
                    else:
                        log_df.to_csv(
                            PERF_LOG_PATH,
                            mode="a",
                            header=False,
                            index=False,
                            encoding="utf-8",
                        )

                self.with_retries(write_log)
                self.log_buffer = []
            logger.info(f"Performance journalisée pour {operation}. CPU: {cpu_usage}%")
        except Exception as e:
            miya_alerts(
                f"Erreur journalisation performance: {str(e)}",
                tag="OPTIONS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur journalisation performance: {str(e)}", priority=3)
            send_telegram_alert(f"Erreur journalisation performance: {str(e)}")
            logger.error(f"Erreur journalisation performance: {str(e)}")

    def save_snapshot(
        self, snapshot_type: str, data: Dict, compress: bool = False
    ) -> None:
        """Sauvegarde un instantané des résultats avec option de compression gzip."""
        try:
            start_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = {
                "timestamp": timestamp,
                "type": snapshot_type,
                "data": data,
                "buffer_size": len(self.log_buffer),
            }
            snapshot_path = os.path.join(
                SNAPSHOT_DIR, f"snapshot_{snapshot_type}_{timestamp}.json"
            )
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)

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
                miya_alerts(
                    f"Snapshot size {file_size:.2f} MB exceeds 1 MB",
                    tag="OPTIONS",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(f"Snapshot size {file_size:.2f} MB exceeds 1 MB", priority=3)
                send_telegram_alert(f"Snapshot size {file_size:.2f} MB exceeds 1 MB")
            latency = time.time() - start_time
            miya_speak(
                f"Snapshot {snapshot_type} sauvegardé: {save_path}",
                tag="OPTIONS",
                voice_profile="calm",
                priority=1,
            )
            send_alert(f"Snapshot {snapshot_type} sauvegardé", priority=1)
            send_telegram_alert(f"Snapshot {snapshot_type} sauvegardé")
            logger.info(f"Snapshot {snapshot_type} sauvegardé: {save_path}")
            self.log_performance(
                "save_snapshot",
                latency,
                success=True,
                snapshot_type=snapshot_type,
                file_size_mb=file_size,
            )
        except Exception as e:
            self.log_performance("save_snapshot", 0, success=False, error=str(e))
            miya_alerts(
                f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}",
                tag="OPTIONS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}", priority=3
            )
            send_telegram_alert(f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}")
            logger.error(f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}")

    def validate_shap_features(self, features: List[str]) -> bool:
        """Valide que les features sont dans les top 150 SHAP."""
        try:
            start_time = time.time()
            if not os.path.exists(FEATURE_IMPORTANCE_PATH):
                miya_alerts(
                    "Fichier SHAP manquant",
                    tag="OPTIONS",
                    voice_profile="urgent",
                    priority=4,
                )
                send_alert("Fichier SHAP manquant", priority=4)
                send_telegram_alert("Fichier SHAP manquant")
                logger.error("Fichier SHAP manquant")
                return False
            shap_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
            if len(shap_df) < 150:
                miya_alerts(
                    f"Nombre insuffisant de SHAP features: {len(shap_df)} < 150",
                    tag="OPTIONS",
                    voice_profile="urgent",
                    priority=4,
                )
                send_alert(
                    f"Nombre insuffisant de SHAP features: {len(shap_df)}", priority=4
                )
                send_telegram_alert(
                    f"Nombre insuffisant de SHAP features: {len(shap_df)}"
                )
                logger.error(f"Nombre insuffisant de SHAP features: {len(shap_df)}")
                return False
            valid_features = set(shap_df["feature"].head(150)).union(obs_t)
            missing = [f for f in features if f not in valid_features]
            if missing:
                miya_alerts(
                    f"Features non incluses dans top 150 SHAP ou obs_t: {missing}",
                    tag="OPTIONS",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(
                    f"Features non incluses dans top 150 SHAP ou obs_t: {missing}",
                    priority=3,
                )
                send_telegram_alert(
                    f"Features non incluses dans top 150 SHAP ou obs_t: {missing}"
                )
                logger.warning(
                    f"Features non incluses dans top 150 SHAP ou obs_t: {missing}"
                )
            latency = time.time() - start_time
            miya_speak(
                "SHAP features validées",
                tag="OPTIONS",
                voice_profile="calm",
                priority=1,
            )
            send_alert("SHAP features validées", priority=1)
            send_telegram_alert("SHAP features validées")
            logger.info("SHAP features validées")
            self.log_performance(
                "validate_shap_features",
                latency,
                success=True,
                num_features=len(features),
            )
            self.save_snapshot(
                "validate_shap_features",
                {"num_features": len(features), "missing": missing},
                compress=False,
            )
            return True
        except Exception as e:
            latency = time.time() - start_time
            miya_alerts(
                f"Erreur validation SHAP features: {str(e)}",
                tag="OPTIONS",
                voice_profile="urgent",
                priority=4,
            )
            send_alert(f"Erreur validation SHAP features: {str(e)}", priority=4)
            send_telegram_alert(f"Erreur validation SHAP features: {str(e)}")
            logger.error(f"Erreur validation SHAP features: {str(e)}")
            self.log_performance(
                "validate_shap_features", latency, success=False, error=str(e)
            )
            return False

    def calculate_iv_atm(
        self, option_chain: pd.DataFrame, underlying_price: float
    ) -> float:
        """Calcule la volatilité implicite ATM (At-The-Money)."""
        try:
            start_time = time.time()
            if option_chain["strike"].nunique() < self.config.get("min_strikes", 5):
                return 0.0
            option_chain["strike_distance"] = np.abs(
                option_chain["strike"] - underlying_price
            )
            atm_options = option_chain[
                option_chain["strike_distance"] == option_chain["strike_distance"].min()
            ]
            iv_call = atm_options[atm_options["option_type"] == "call"][
                "implied_volatility"
            ].mean()
            iv_put = atm_options[atm_options["option_type"] == "put"][
                "implied_volatility"
            ].mean()
            iv_atm = np.nanmean([iv_call, iv_put])
            latency = time.time() - start_time
            self.log_performance("calculate_iv_atm", latency, success=True)
            return iv_atm if not np.isnan(iv_atm) and iv_atm >= 0 else 0.0
        except Exception as e:
            self.log_performance("calculate_iv_atm", 0, success=False, error=str(e))
            miya_alerts(
                f"Erreur dans calculate_iv_atm: {str(e)}",
                tag="OPTIONS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans calculate_iv_atm: {str(e)}", priority=3)
            send_telegram_alert(f"Erreur dans calculate_iv_atm: {str(e)}")
            logger.error(f"Erreur dans calculate_iv_atm: {str(e)}")
            return 0.0

    def calculate_gex_slope(
        self, option_chain: pd.DataFrame, underlying_price: float
    ) -> float:
        """Calcule la pente de l'exposition gamma (GEX)."""
        try:
            start_time = time.time()
            if option_chain["strike"].nunique() < self.config.get("min_strikes", 5):
                return 0.0
            option_chain["gex"] = option_chain["gamma"] * option_chain["open_interest"]
            gex_by_strike = option_chain.groupby("strike")["gex"].sum().reset_index()
            if len(gex_by_strike) < 2:
                return 0.0
            gex_by_strike = gex_by_strike.sort_values("strike")
            delta_gex = gex_by_strike["gex"].diff()
            delta_strike = gex_by_strike["strike"].diff()
            slope = (delta_gex / delta_strike).mean()
            latency = time.time() - start_time
            self.log_performance("calculate_gex_slope", latency, success=True)
            return slope if not np.isnan(slope) else 0.0
        except Exception as e:
            self.log_performance("calculate_gex_slope", 0, success=False, error=str(e))
            miya_alerts(
                f"Erreur dans calculate_gex_slope: {str(e)}",
                tag="OPTIONS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans calculate_gex_slope: {str(e)}", priority=3)
            send_telegram_alert(f"Erreur dans calculate_gex_slope: {str(e)}")
            logger.error(f"Erreur dans calculate_gex_slope: {str(e)}")
            return 0.0

    def calculate_gamma_peak_distance(
        self, option_chain: pd.DataFrame, underlying_price: float
    ) -> float:
        """Calcule la distance entre le prix actuel et le strike avec la plus forte concentration de gamma."""
        try:
            start_time = time.time()
            option_chain["gex"] = option_chain["gamma"] * option_chain["open_interest"]
            gex_by_strike = option_chain.groupby("strike")["gex"].sum()
            if gex_by_strike.empty:
                return 0.0
            gamma_peak = gex_by_strike.idxmax()
            distance = underlying_price - gamma_peak
            latency = time.time() - start_time
            self.log_performance("calculate_gamma_peak_distance", latency, success=True)
            return distance if not np.isnan(distance) else 0.0
        except Exception as e:
            self.log_performance(
                "calculate_gamma_peak_distance", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans calculate_gamma_peak_distance: {str(e)}",
                tag="OPTIONS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans calculate_gamma_peak_distance: {str(e)}", priority=3
            )
            send_telegram_alert(f"Erreur dans calculate_gamma_peak_distance: {str(e)}")
            logger.error(f"Erreur dans calculate_gamma_peak_distance: {str(e)}")
            return 0.0

    def calculate_option_metrics(
        self,
        data: pd.DataFrame,
        option_chain_path: str = os.path.join(
            BASE_DIR, "data", "options", "option_chain.csv"
        ),
        config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml"),
    ) -> pd.DataFrame:
        """Calcule les indicateurs d’options et enrichit le DataFrame."""
        cache_path = os.path.join(
            CACHE_DIR,
            f"option_metrics_{hashlib.sha256(data.to_json().encode()).hexdigest()}.csv",
        )
        try:
            start_time = time.time()
            current_time = datetime.now()
            expired_keys = [
                k
                for k, v in self.cache.items()
                if (current_time - v["timestamp"]).total_seconds()
                > self.config.get("cache_hours", 24) * 3600
            ]
            for k in expired_keys:
                self.cache.pop(k)
            if os.path.exists(cache_path):
                cached_data = pd.read_csv(cache_path)
                if len(cached_data) == len(data) and cached_data["timestamp"].equals(
                    data["timestamp"]
                ):
                    latency = time.time() - start_time
                    self.log_performance(
                        "calculate_option_metrics_cache_hit", latency, success=True
                    )
                    miya_speak(
                        "Option metrics chargées depuis cache",
                        tag="OPTIONS",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert("Option metrics chargées depuis cache", priority=1)
                    send_telegram_alert("Option metrics chargées depuis cache")
                    return cached_data

            config = self.load_config(config_path)
            window_size = config.get("window_size", 5)
            min_option_rows = config.get("min_option_rows", 10)
            time_tolerance = config.get("time_tolerance", "10s")

            if data.empty:
                error_msg = "DataFrame vide"
                miya_alerts(
                    error_msg, tag="OPTIONS", voice_profile="urgent", priority=5
                )
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            data = data.copy()

            def load_option_chain():
                if not os.path.exists(option_chain_path):
                    raise FileNotFoundError(f"Fichier {option_chain_path} introuvable")
                option_chain = pd.read_csv(option_chain_path)
                if len(option_chain) < min_option_rows:
                    raise ValueError(
                        f"Moins de {min_option_rows} lignes dans {option_chain_path}"
                    )
                return option_chain

            option_chain = self.with_retries(load_option_chain)

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
            required_data_cols = ["timestamp", "close"]
            missing_option_cols = [
                col for col in required_option_cols if col not in option_chain.columns
            ]
            missing_data_cols = [
                col for col in required_data_cols if col not in data.columns
            ]

            for col in missing_option_cols:
                option_chain[col] = (
                    pd.to_datetime("2025-04-14") if col == "expiration_date" else 0
                )
                miya_speak(
                    f"Colonne '{col}' manquante dans option_chain, imputée",
                    tag="OPTIONS",
                    voice_profile="warning",
                    priority=2,
                )
                send_alert(
                    f"Colonne '{col}' manquante dans option_chain, imputée", priority=2
                )
                send_telegram_alert(
                    f"Colonne '{col}' manquante dans option_chain, imputée"
                )
            for col in missing_data_cols:
                if col == "timestamp":
                    data[col] = pd.date_range(
                        start=pd.Timestamp.now(), periods=len(data), freq="1min"
                    )
                    miya_speak(
                        "Colonne timestamp manquante, création par défaut",
                        tag="OPTIONS",
                        voice_profile="warning",
                        priority=2,
                    )
                    send_alert(
                        "Colonne timestamp manquante, création par défaut", priority=2
                    )
                    send_telegram_alert(
                        "Colonne timestamp manquante, création par défaut"
                    )
                else:
                    data[col] = (
                        data[col]
                        .interpolate(method="linear", limit_direction="both")
                        .fillna(data[col].median() if col in data.columns else 0)
                    )
                    miya_speak(
                        f"Colonne '{col}' manquante, imputée",
                        tag="OPTIONS",
                        voice_profile="warning",
                        priority=2,
                    )
                    send_alert(f"Colonne '{col}' manquante, imputée", priority=2)
                    send_telegram_alert(f"Colonne '{col}' manquante, imputée")

            option_chain["timestamp"] = pd.to_datetime(
                option_chain["timestamp"], errors="coerce"
            )
            data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
            if data["timestamp"].isna().any() or option_chain["timestamp"].isna().any():
                error_msg = "NaN dans les timestamps"
                miya_alerts(
                    error_msg, tag="OPTIONS", voice_profile="urgent", priority=5
                )
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)
            if not data["timestamp"].is_monotonic_increasing:
                error_msg = "Timestamps non croissants dans data"
                miya_alerts(
                    error_msg, tag="OPTIONS", voice_profile="urgent", priority=5
                )
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            option_chain["expiration_date"] = pd.to_datetime(
                option_chain["expiration_date"], errors="coerce"
            )
            option_chain["expiration_days"] = (
                option_chain["expiration_date"] - option_chain["timestamp"]
            ).dt.days

            # Calculer confidence_drop_rate (Phase 8)
            valid_cols = [
                col for col in required_option_cols if col in option_chain.columns
            ]
            confidence_drop_rate = 1.0 - min(
                len(valid_cols) / len(required_option_cols), 1.0
            )
            if option_chain["strike"].nunique() < self.config.get("min_strikes", 5):
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(valid_cols)}/{len(required_option_cols)} colonnes valides, {option_chain['strike'].nunique()} strikes)"
                miya_alerts(
                    alert_msg, tag="OPTIONS", voice_profile="urgent", priority=3
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

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
            ]
            missing_features = [f for f in features if f not in obs_t]
            if missing_features:
                miya_alerts(
                    f"Features manquantes dans obs_t: {missing_features}",
                    tag="OPTIONS",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(
                    f"Features manquantes dans obs_t: {missing_features}", priority=3
                )
                send_telegram_alert(
                    f"Features manquantes dans obs_t: {missing_features}"
                )
                logger.warning(f"Features manquantes dans obs_t: {missing_features}")
            self.validate_shap_features(features)

            for feature in features:
                data[feature] = 0.0

            option_chain["gex"] = option_chain["gamma"] * option_chain["open_interest"]
            option_chain["delta_exposure"] = (
                option_chain["open_interest"]
                * option_chain["gamma"]
                * (option_chain["strike"] - option_chain["underlying_price"])
            )
            option_chain["theta_pressure"] = option_chain["open_interest"] / (
                option_chain["expiration_days"] + 1e-6
            )

            merged_data = pd.merge_asof(
                data.sort_values("timestamp"),
                option_chain.sort_values("timestamp"),
                on="timestamp",
                tolerance=pd.Timedelta(time_tolerance),
                direction="nearest",
            )
            if len(merged_data) / len(data) < 0.5:
                error_msg = "Moins de 50% des timestamps alignés"
                miya_alerts(
                    error_msg, tag="OPTIONS", voice_profile="urgent", priority=5
                )
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            grouped = merged_data.groupby("timestamp")
            data["iv_atm"] = grouped.apply(
                lambda x: self.calculate_iv_atm(
                    x, x["close"].iloc[0] if not x.empty else 0
                )
            ).reindex(data.index, fill_value=0)
            data["gex_slope"] = grouped.apply(
                lambda x: self.calculate_gex_slope(
                    x, x["close"].iloc[0] if not x.empty else 0
                )
            ).reindex(data.index, fill_value=0)
            data["gamma_peak_distance"] = grouped.apply(
                lambda x: self.calculate_gamma_peak_distance(
                    x, x["close"].iloc[0] if not x.empty else 0
                )
            ).reindex(data.index, fill_value=0)
            data["iv_skew"] = grouped.apply(
                lambda x: (
                    x[x["option_type"] == "call"]["implied_volatility"].mean()
                    - x[x["option_type"] == "put"]["implied_volatility"].mean()
                    if not x.empty
                    else 0
                )
            ).reindex(data.index, fill_value=0)
            data["gex_total"] = grouped["gex"].sum().reindex(data.index, fill_value=0)
            data["oi_peak_call"] = grouped.apply(
                lambda x: (
                    x[x["option_type"] == "call"]["open_interest"].idxmax()
                    if not x[x["option_type"] == "call"].empty
                    else 0
                )
            ).reindex(data.index, fill_value=0)
            data["oi_peak_put"] = grouped.apply(
                lambda x: (
                    x[x["option_type"] == "put"]["open_interest"].idxmax()
                    if not x[x["option_type"] == "put"].empty
                    else 0
                )
            ).reindex(data.index, fill_value=0)
            data["gamma_wall"] = grouped.apply(
                lambda x: (
                    x.groupby("strike")["gex"].sum().idxmax()
                    if not x.empty and "gex" in x.columns
                    else 0
                )
            ).reindex(data.index, fill_value=0)
            data["delta_exposure"] = (
                grouped["delta_exposure"].sum().reindex(data.index, fill_value=0)
            )
            data["theta_pressure"] = (
                grouped["theta_pressure"].sum().reindex(data.index, fill_value=0)
            )
            data["iv_slope"] = grouped.apply(
                lambda x: (
                    (x["implied_volatility"].diff() / x["strike"].diff()).mean()
                    if len(x) >= 2
                    else 0
                )
            ).reindex(data.index, fill_value=0)
            data["call_put_ratio"] = grouped.apply(
                lambda x: (
                    x[x["option_type"] == "call"]["open_interest"].sum()
                    / (x[x["option_type"] == "put"]["open_interest"].sum() + 1e-6)
                    if not x.empty
                    else 0
                )
            ).reindex(data.index, fill_value=0)
            data["iv_atm_change"] = data["iv_atm"].diff().fillna(0)
            data["gex_stability"] = (
                data["gex_total"]
                .rolling(window=window_size, min_periods=1)
                .std()
                .fillna(0)
            )
            data["strike_density"] = (
                grouped["strike"].nunique().reindex(data.index, fill_value=0)
            )
            data["time_to_expiry"] = (
                grouped["expiration_days"].mean().reindex(data.index, fill_value=0)
            )

            for feature in features:
                if data[feature].isna().any():
                    data[feature] = data[feature].fillna(0)
                if feature in ["iv_atm", "time_to_expiry"] and data[feature].min() < 0:
                    miya_alerts(
                        f"Valeurs négatives dans {feature}",
                        tag="OPTIONS",
                        voice_profile="urgent",
                        priority=3,
                    )
                    send_alert(f"Valeurs négatives dans {feature}", priority=3)
                    send_telegram_alert(f"Valeurs négatives dans {feature}")
                    data[feature] = data[feature].clip(lower=0)

            os.makedirs(CACHE_DIR, exist_ok=True)

            def write_cache():
                data.to_csv(cache_path, index=False, encoding="utf-8")

            self.with_retries(write_cache)
            file_size = os.path.getsize(cache_path) / 1024 / 1024
            if file_size > 1.0:
                miya_alerts(
                    f"Cache option_metrics size {file_size:.2f} MB exceeds 1 MB",
                    tag="OPTIONS",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(
                    f"Cache option_metrics size {file_size:.2f} MB exceeds 1 MB",
                    priority=3,
                )
                send_telegram_alert(
                    f"Cache option_metrics size {file_size:.2f} MB exceeds 1 MB"
                )

            latency = time.time() - start_time
            self.log_performance(
                "calculate_option_metrics",
                latency,
                success=True,
                num_rows=len(data),
                num_features=len(features),
                confidence_drop_rate=confidence_drop_rate,
            )
            miya_speak(
                "Métriques d'options calculées",
                tag="OPTIONS",
                voice_profile="calm",
                priority=1,
            )
            send_alert("Métriques d'options calculées", priority=1)
            send_telegram_alert("Métriques d'options calculées")
            logger.info("Métriques d'options calculées")
            self.save_snapshot(
                "option_metrics",
                {
                    "num_rows": len(data),
                    "num_features": len(features),
                    "confidence_drop_rate": confidence_drop_rate,
                },
                compress=False,
            )
            return data

        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans calculate_option_metrics: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                "calculate_option_metrics", latency, success=False, error=str(e)
            )
            miya_alerts(error_msg, tag="OPTIONS", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.save_snapshot("option_metrics", {"error": str(e)}, compress=False)
            for feature in features:
                data[feature] = 0.0
            return data

    def calculate_incremental_metrics(
        self, row: pd.Series, option_chain: pd.DataFrame, window_size: int = 5
    ) -> pd.Series:
        """Calcule les métriques d’options pour une seule ligne en temps réel."""
        try:
            start_time = time.time()
            features = {
                f: 0.0
                for f in [
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
                ]
            }

            current_chain = option_chain[option_chain["timestamp"] == row["timestamp"]]
            if current_chain.empty:
                return pd.Series(features)

            # Calculer confidence_drop_rate (Phase 8)
            required_cols = [
                "timestamp",
                "underlying_price",
                "strike",
                "option_type",
                "implied_volatility",
                "open_interest",
                "gamma",
                "expiration_date",
            ]
            valid_cols = [col for col in required_cols if col in current_chain.columns]
            confidence_drop_rate = 1.0 - min(len(valid_cols) / len(required_cols), 1.0)
            if current_chain["strike"].nunique() < self.config.get("min_strikes", 5):
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(valid_cols)}/{len(required_cols)} colonnes valides, {current_chain['strike'].nunique()} strikes)"
                miya_alerts(
                    alert_msg, tag="OPTIONS", voice_profile="urgent", priority=3
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            underlying_price = row["close"]
            features["iv_atm"] = self.calculate_iv_atm(current_chain, underlying_price)
            features["gex_slope"] = self.calculate_gex_slope(
                current_chain, underlying_price
            )
            features["gamma_peak_distance"] = self.calculate_gamma_peak_distance(
                current_chain, underlying_price
            )
            calls = current_chain[current_chain["option_type"] == "call"]
            puts = current_chain[current_chain["option_type"] == "put"]
            features["iv_skew"] = (
                calls["implied_volatility"].mean() - puts["implied_volatility"].mean()
                if not (calls.empty or puts.empty)
                else 0
            )
            features["gex_total"] = (
                current_chain["gamma"] * current_chain["open_interest"]
            ).sum()
            features["oi_peak_call"] = (
                calls.loc[calls["open_interest"].idxmax(), "strike"]
                if not calls.empty
                else 0
            )
            features["oi_peak_put"] = (
                puts.loc[puts["open_interest"].idxmax(), "strike"]
                if not puts.empty
                else 0
            )
            features["gamma_wall"] = (
                current_chain.groupby("strike")["gex"].sum().idxmax()
                if not current_chain.empty and "gex" in current_chain.columns
                else 0
            )
            features["delta_exposure"] = (
                current_chain["open_interest"]
                * current_chain["gamma"]
                * (current_chain["strike"] - underlying_price)
            ).sum()
            features["theta_pressure"] = (
                current_chain["open_interest"]
                / (current_chain["expiration_days"] + 1e-6)
            ).sum()
            features["iv_slope"] = (
                (
                    current_chain["implied_volatility"].diff()
                    / current_chain["strike"].diff()
                ).mean()
                if len(current_chain) >= 2
                else 0
            )
            features["call_put_ratio"] = (
                calls["open_interest"].sum() / (puts["open_interest"].sum() + 1e-6)
                if not puts.empty
                else 0
            )
            features["strike_density"] = len(current_chain["strike"].unique())
            features["time_to_expiry"] = (
                current_chain["expiration_days"].mean()
                if not current_chain.empty
                else 0
            )

            result = pd.Series(features)
            latency = time.time() - start_time
            self.log_performance(
                "calculate_incremental_metrics",
                latency,
                success=True,
                num_features=len(features),
                confidence_drop_rate=confidence_drop_rate,
            )
            miya_speak(
                "Métriques d’options incrémentales calculées",
                tag="OPTIONS",
                voice_profile="calm",
                priority=1,
            )
            send_alert("Métriques d’options incrémentales calculées", priority=1)
            send_telegram_alert("Métriques d’options incrémentales calculées")
            logger.info("Métriques d’options incrémentales calculées")
            self.save_snapshot(
                "incremental_metrics",
                {
                    "features": {k: float(v) for k, v in features.items()},
                    "confidence_drop_rate": confidence_drop_rate,
                },
                compress=False,
            )
            return result

        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans calculate_incremental_metrics: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                "calculate_incremental_metrics", latency, success=False, error=str(e)
            )
            miya_alerts(error_msg, tag="OPTIONS", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.save_snapshot("incremental_metrics", {"error": str(e)}, compress=False)
            return pd.Series(features)


if __name__ == "__main__":
    try:
        calculator = OptionMetrics()
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
        os.makedirs(os.path.join(BASE_DIR, "data", "options"), exist_ok=True)
        option_chain.to_csv(
            os.path.join(BASE_DIR, "data", "options", "option_chain.csv"), index=False
        )
        result = calculator.calculate_option_metrics(data)
        print(result.head())
        miya_speak(
            "Test calculate_option_metrics terminé",
            tag="TEST",
            voice_profile="calm",
            priority=1,
        )
        send_alert("Test calculate_option_metrics terminé", priority=1)
        send_telegram_alert("Test calculate_option_metrics terminé")
        logger.info("Test calculate_option_metrics terminé")
    except Exception as e:
        miya_alerts(
            f"Erreur test: {str(e)}\n{traceback.format_exc()}",
            tag="ALERT",
            voice_profile="urgent",
            priority=3,
        )
        send_alert(f"Erreur test: {str(e)}", priority=3)
        send_telegram_alert(f"Erreur test: {str(e)}")
        logger.error(f"Erreur test: {str(e)}")
        raise
