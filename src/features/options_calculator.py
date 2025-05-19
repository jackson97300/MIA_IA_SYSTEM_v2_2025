# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/features/options_calculator.py
# Calcule les features liées aux options à partir des données IQFeed.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Calcule les métriques d'options (Greeks, IV, OI, skews, etc.) pour feature_pipeline.py,
#        avec cache, logs psutil, validation SHAP (méthode 17), et compatibilité top 150 SHAP.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, sklearn>=1.0.2,<1.1.0, psutil>=5.9.0,<6.0.0,
#   pyyaml>=6.0.0,<7.0.0, matplotlib>=3.7.0,<3.8.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
# - src/model/utils/obs_template.py
# - src/data/data_provider.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/es_config.yaml
# - data/iqfeed/option_chain.csv
# - data/features/feature_importance.csv
#
# Outputs :
# - data/features/cache/options/*.csv
# - data/logs/options_calculator_performance.csv
# - data/options_snapshots/*.json (option *.json.gz)
# - data/figures/options/
#
# Notes :
# - Conforme à structure.txt (version 2.1.3, 2025-05-13).
# - Utilise IQFeed via data_provider.py, avec retries (max 3, délai 2^attempt secondes).
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des métriques d’options.
# - Intègre validation SHAP (Phase 17) pour assurer la conformité avec les top 150 SHAP features.
# - Sauvegarde des snapshots JSON non compressés par défaut avec option de compression gzip.
# - Envoie des alertes via AlertManager et telegram_alert pour les erreurs critiques et succès.
# - Tests unitaires disponibles dans tests/test_options_calculator.py.
# - Conforme à la ligne rouge des top 150 SHAP features définie dans feature_pipeline.py.

import gzip
import hashlib
import json
import logging
import os
import time
import traceback
from collections import deque
from datetime import datetime
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import yaml
from sklearn.preprocessing import MinMaxScaler

from src.model.utils.alert_manager import send_alert
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.model.utils.obs_template import obs_t
from src.utils.telegram_alert import send_telegram_alert

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "options_snapshots")
CSV_LOG_PATH = os.path.join(
    BASE_DIR, "data", "logs", "options_calculator_performance.csv"
)
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures", "options")
CACHE_DIR = os.path.join(BASE_DIR, "data", "features", "cache", "options")
FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR, "data", "features", "feature_importance.csv"
)

# Création des répertoires
os.makedirs(os.path.join(BASE_DIR, "data", "logs"), exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data", "options"), exist_ok=True)

# Configuration du logging
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "data", "logs", "options_calculator.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Constantes
MAX_RETRIES = 3
RETRY_DELAY = 2.0


class OptionsCalculator:

    def __init__(
        self, config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml")
    ):
        self.log_buffer = []
        self.cache = {}
        self.buffer = deque(maxlen=1000)
        try:
            self.config = self.load_config(config_path)
            self.buffer_size = self.config.get("buffer_size", 100)
            self.max_cache_size = self.config.get("max_cache_size", 1000)
            self.buffer_maxlen = self.config.get("buffer_maxlen", 1000)
            self.buffer = deque(maxlen=self.buffer_maxlen)
            miya_speak(
                "OptionsCalculator initialisé",
                tag="OPTIONS",
                voice_profile="calm",
                priority=2,
            )
            send_alert("OptionsCalculator initialisé", priority=2)
            send_telegram_alert("OptionsCalculator initialisé")
            logger.info("OptionsCalculator initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config_path": config_path}, compress=False)
        except Exception as e:
            error_msg = f"Erreur initialisation OptionsCalculator: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="OPTIONS", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            self.config = {
                "option_chain_path": os.path.join(
                    BASE_DIR, "data", "options", "option_chain.csv"
                ),
                "time_tolerance": "10s",
                "min_strikes": 5,
                "expiration_window": "30d",
                "buffer_size": 100,
                "max_cache_size": 1000,
                "buffer_maxlen": 1000,
                "cache_hours": 24,
            }
            self.buffer_size = 100
            self.max_cache_size = 1000
            self.buffer_maxlen = 1000
            self.buffer = deque(maxlen=self.buffer_maxlen)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Charge la configuration via yaml."""

        def load_yaml():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            if "options_calculator" not in config:
                raise ValueError(
                    "Clé 'options_calculator' manquante dans la configuration"
                )
            required_keys = ["option_chain_path", "time_tolerance", "min_strikes"]
            missing_keys = [
                key for key in required_keys if key not in config["options_calculator"]
            ]
            if missing_keys:
                raise ValueError(
                    f"Clés manquantes dans 'options_calculator': {missing_keys}"
                )
            return config["options_calculator"]

        try:
            start_time = time.time()
            config = self.with_retries(load_yaml)
            cache_key = hashlib.sha256(str(config).encode()).hexdigest()
            self.cache[cache_key] = config
            if len(self.cache) > self.max_cache_size:
                self.cache.pop(next(iter(self.cache)))
            latency = time.time() - start_time
            miya_speak(
                "Configuration options_calculator chargée",
                tag="OPTIONS",
                voice_profile="calm",
                priority=2,
            )
            send_alert("Configuration options_calculator chargée", priority=2)
            send_telegram_alert("Configuration options_calculator chargée")
            logger.info("Configuration options_calculator chargée")
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
        """Journalise les performances des opérations critiques."""
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
            cpu_usage = psutil.cpu_percent()
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
            if len(self.log_buffer) >= self.buffer_size:
                log_df = pd.DataFrame(self.log_buffer)
                os.makedirs(os.path.dirname(CSV_LOG_PATH), exist_ok=True)

                def write_log():
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
        """Valide les features par rapport aux top 150 SHAP."""
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

    def plot_options_features(self, data: pd.DataFrame, timestamp: str) -> None:
        """Génère des visualisations des features d’options."""
        start_time = time.time()
        try:
            timestamp_safe = timestamp.replace(":", "-")
            os.makedirs(FIGURES_DIR, exist_ok=True)

            features = [
                "gex",
                "iv_atm",
                "option_skew",
                "oi_concentration",
                "gex_slope",
                "short_dated_iv_slope",
                "term_structure_skew",
                "risk_reversal_25",
                "butterfly_25",
                "gamma_bucket_exposure",
                "delta_exposure_ratio",
            ]
            available_features = [f for f in features if f in data.columns]

            plt.figure(figsize=(12, 6))
            for feature in available_features:
                plt.plot(data.index, data[feature], label=feature)
            plt.title(f"Features Options - {timestamp}")
            plt.xlabel("Index")
            plt.ylabel("Valeur")
            plt.legend()
            plt.grid(True)
            plt.savefig(
                os.path.join(
                    FIGURES_DIR, f"options_features_temporal_{timestamp_safe}.png"
                )
            )
            plt.close()

            plt.figure(figsize=(8, 6))
            for feature in available_features:
                plt.hist(data[feature], bins=20, alpha=0.5, label=feature)
            plt.title(f"Distribution des Features Options - {timestamp}")
            plt.xlabel("Valeur")
            plt.ylabel("Fréquence")
            plt.legend()
            plt.grid(True)
            plt.savefig(
                os.path.join(
                    FIGURES_DIR, f"options_features_distribution_{timestamp_safe}.png"
                )
            )
            plt.close()

            latency = time.time() - start_time
            miya_speak(
                f"Visualisations générées: {FIGURES_DIR}",
                tag="OPTIONS",
                voice_profile="calm",
                priority=2,
            )
            send_alert(f"Visualisations générées: {FIGURES_DIR}", priority=2)
            send_telegram_alert(f"Visualisations générées: {FIGURES_DIR}")
            logger.info(f"Visualisations générées: {FIGURES_DIR}")
            self.log_performance("plot_options_features", latency, success=True)
        except Exception as e:
            latency = time.time() - start_time
            error_msg = (
                f"Erreur génération visualisations: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="OPTIONS", voice_profile="urgent", priority=2)
            send_alert(error_msg, priority=2)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "plot_options_features", latency, success=False, error=str(e)
            )

    def calculate_gex(self, data: pd.DataFrame) -> pd.Series:
        """Calcule le Gamma Exposure (GEX)."""
        try:
            start_time = time.time()
            if not all(
                col in data.columns
                for col in ["open_interest", "gamma", "option_type", "close"]
            ):
                raise ValueError("Colonnes manquantes pour GEX")
            calls = data[data["option_type"] == "call"]
            puts = data[data["option_type"] == "put"]
            gex = (
                calls["gamma"] * calls["open_interest"] * 100 * calls["close"]
            ).sum() - (
                puts["gamma"] * puts["open_interest"] * 100 * puts["close"]
            ).sum()
            result = pd.Series(gex, index=data.index)
            latency = time.time() - start_time
            self.log_performance("calculate_gex", latency, success=True)
            return result
        except Exception as e:
            self.log_performance("calculate_gex", 0, success=False, error=str(e))
            miya_alerts(
                f"Erreur dans calculate_gex: {str(e)}",
                tag="OPTIONS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans calculate_gex: {str(e)}", priority=3)
            send_telegram_alert(f"Erreur dans calculate_gex: {str(e)}")
            logger.error(f"Erreur dans calculate_gex: {str(e)}")
            return pd.Series(0.0, index=data.index)

    def calculate_iv_atm(self, data: pd.DataFrame, underlying_price: float) -> float:
        """Calcule la volatilité implicite ATM."""
        try:
            start_time = time.time()
            if data["strike"].nunique() < self.config.get("min_strikes", 5):
                return 0.0
            data["strike_distance"] = np.abs(data["strike"] - underlying_price)
            atm_options = data[data["strike_distance"] == data["strike_distance"].min()]
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

    def calculate_option_skew(self, data: pd.DataFrame) -> float:
        """Calcule le skew de volatilité implicite."""
        try:
            start_time = time.time()
            calls = data[data["option_type"] == "call"]
            puts = data[data["option_type"] == "put"]
            iv_skew = (
                calls["implied_volatility"].mean() - puts["implied_volatility"].mean()
            )
            latency = time.time() - start_time
            self.log_performance("calculate_option_skew", latency, success=True)
            return iv_skew if not np.isnan(iv_skew) else 0.0
        except Exception as e:
            self.log_performance(
                "calculate_option_skew", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans calculate_option_skew: {str(e)}",
                tag="OPTIONS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans calculate_option_skew: {str(e)}", priority=3)
            send_telegram_alert(f"Erreur dans calculate_option_skew: {str(e)}")
            logger.error(f"Erreur dans calculate_option_skew: {str(e)}")
            return 0.0

    def calculate_oi_concentration(self, data: pd.DataFrame) -> float:
        """Calcule la concentration de l’open interest."""
        try:
            start_time = time.time()
            oi_total = data["open_interest"].sum()
            if oi_total == 0:
                return 0.0
            top_oi = data.groupby("strike")["open_interest"].sum().nlargest(5).sum()
            concentration = top_oi / oi_total
            latency = time.time() - start_time
            self.log_performance("calculate_oi_concentration", latency, success=True)
            return concentration if not np.isnan(concentration) else 0.0
        except Exception as e:
            self.log_performance(
                "calculate_oi_concentration", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans calculate_oi_concentration: {str(e)}",
                tag="OPTIONS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans calculate_oi_concentration: {str(e)}", priority=3)
            send_telegram_alert(f"Erreur dans calculate_oi_concentration: {str(e)}")
            logger.error(f"Erreur dans calculate_oi_concentration: {str(e)}")
            return 0.0

    def calculate_gex_slope(self, data: pd.DataFrame, underlying_price: float) -> float:
        """Calcule la pente du GEX par rapport aux strikes."""
        try:
            start_time = time.time()
            if data["strike"].nunique() < self.config.get("min_strikes", 5):
                return 0.0
            data["gex"] = data["gamma"] * data["open_interest"]
            gex_by_strike = data.groupby("strike")["gex"].sum().reset_index()
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

    def compute_iv_slope(
        self, data: pd.DataFrame, days: List[int] = [7, 30]
    ) -> Dict[str, float]:
        """Calcule la pente de la volatilité implicite pour différentes maturités."""
        try:
            start_time = time.time()
            results = {}
            for day in days:
                chain = data[data["expiration_days"] <= day]
                if chain["strike"].nunique() < 2:
                    results[f"iv_slope_{day}d"] = 0.0
                    continue
                iv_by_strike = chain.groupby("strike")["implied_volatility"].mean()
                iv_diff = iv_by_strike.diff()
                strike_diff = iv_by_strike.index.to_series().diff()
                slope = (iv_diff / strike_diff).mean()
                results[f"iv_slope_{day}d"] = slope if not np.isnan(slope) else 0.0
            latency = time.time() - start_time
            self.log_performance("compute_iv_slope", latency, success=True)
            return results
        except Exception as e:
            self.log_performance("compute_iv_slope", 0, success=False, error=str(e))
            miya_alerts(
                f"Erreur dans compute_iv_slope: {str(e)}",
                tag="OPTIONS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_iv_slope: {str(e)}", priority=3)
            send_telegram_alert(f"Erreur dans compute_iv_slope: {str(e)}")
            logger.error(f"Erreur dans compute_iv_slope: {str(e)}")
            return {f"iv_slope_{day}d": 0.0 for day in days}

    def compute_term_structure_skew(
        self, data: pd.DataFrame, days: List[int] = [30, 60]
    ) -> float:
        """Calcule le skew de la structure par terme."""
        try:
            start_time = time.time()
            short_term = data[data["expiration_days"] <= days[0]]
            long_term = data[
                (data["expiration_days"] > days[0])
                & (data["expiration_days"] <= days[1])
            ]
            if short_term.empty or long_term.empty:
                return 0.0
            short_iv_skew = (
                short_term[short_term["option_type"] == "call"][
                    "implied_volatility"
                ].mean()
                - short_term[short_term["option_type"] == "put"][
                    "implied_volatility"
                ].mean()
            )
            long_iv_skew = (
                long_term[long_term["option_type"] == "call"][
                    "implied_volatility"
                ].mean()
                - long_term[long_term["option_type"] == "put"][
                    "implied_volatility"
                ].mean()
            )
            skew = short_iv_skew - long_iv_skew
            latency = time.time() - start_time
            self.log_performance("compute_term_structure_skew", latency, success=True)
            return skew if not np.isnan(skew) else 0.0
        except Exception as e:
            self.log_performance(
                "compute_term_structure_skew", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_term_structure_skew: {str(e)}",
                tag="OPTIONS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_term_structure_skew: {str(e)}", priority=3)
            send_telegram_alert(f"Erreur dans compute_term_structure_skew: {str(e)}")
            logger.error(f"Erreur dans compute_term_structure_skew: {str(e)}")
            return 0.0

    def compute_risk_reversal(self, data: pd.DataFrame, delta: float = 25) -> float:
        """Calcule le risk reversal pour un delta donné."""
        try:
            start_time = time.time()
            calls = data[
                (data["option_type"] == "call") & (data["delta"].abs() >= delta / 100)
            ]
            puts = data[
                (data["option_type"] == "put") & (data["delta"].abs() >= delta / 100)
            ]
            if calls.empty or puts.empty:
                return 0.0
            rr = calls["implied_volatility"].mean() - puts["implied_volatility"].mean()
            latency = time.time() - start_time
            self.log_performance("compute_risk_reversal", latency, success=True)
            return rr if not np.isnan(rr) else 0.0
        except Exception as e:
            self.log_performance(
                "compute_risk_reversal", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_risk_reversal: {str(e)}",
                tag="OPTIONS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_risk_reversal: {str(e)}", priority=3)
            send_telegram_alert(f"Erreur dans compute_risk_reversal: {str(e)}")
            logger.error(f"Erreur dans compute_risk_reversal: {str(e)}")
            return 0.0

    def compute_butterfly(self, data: pd.DataFrame, delta: float = 25) -> float:
        """Calcule le butterfly spread pour un delta donné."""
        try:
            start_time = time.time()
            atm = data[data["delta"].abs() < 0.05]
            wings = data[
                (data["delta"].abs() >= delta / 100)
                & (data["delta"].abs() <= (delta + 5) / 100)
            ]
            if atm.empty or len(wings) < 2:
                return 0.0
            atm_iv = atm["implied_volatility"].mean()
            wings_iv = wings["implied_volatility"].mean()
            butterfly = 2 * atm_iv - wings_iv
            latency = time.time() - start_time
            self.log_performance("compute_butterfly", latency, success=True)
            return butterfly if not np.isnan(butterfly) else 0.0
        except Exception as e:
            self.log_performance("compute_butterfly", 0, success=False, error=str(e))
            miya_alerts(
                f"Erreur dans compute_butterfly: {str(e)}",
                tag="OPTIONS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_butterfly: {str(e)}", priority=3)
            send_telegram_alert(f"Erreur dans compute_butterfly: {str(e)}")
            logger.error(f"Erreur dans compute_butterfly: {str(e)}")
            return 0.0

    def compute_oi_concentration_ratio(self, data: pd.DataFrame) -> float:
        """Calcule le ratio de concentration de l’open interest."""
        try:
            start_time = time.time()
            oi_total = data["open_interest"].sum()
            if oi_total == 0:
                return 0.0
            top_oi = data.groupby("strike")["open_interest"].sum().nlargest(3).sum()
            ratio = top_oi / oi_total
            latency = time.time() - start_time
            self.log_performance(
                "compute_oi_concentration_ratio", latency, success=True
            )
            return ratio if not np.isnan(ratio) else 0.0
        except Exception as e:
            self.log_performance(
                "compute_oi_concentration_ratio", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_oi_concentration_ratio: {str(e)}",
                tag="OPTIONS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_oi_concentration_ratio: {str(e)}", priority=3
            )
            send_telegram_alert(f"Erreur dans compute_oi_concentration_ratio: {str(e)}")
            logger.error(f"Erreur dans compute_oi_concentration_ratio: {str(e)}")
            return 0.0


def compute_gamma_bucket_exposure(
    self, data: pd.DataFrame, range_pct: float = 10
) -> float:
    """Calcule l’exposition gamma dans une plage donnée."""
    try:
        start_time = time.time()
        underlying_price = data["underlying_price"].iloc[0]
        lower_bound = underlying_price * (1 - range_pct / 100)
        upper_bound = underlying_price * (1 + range_pct / 100)
        bucket = data[(data["strike"] >= lower_bound) & (data["strike"] <= upper_bound)]
        exposure = (bucket["gamma"] * bucket["open_interest"]).sum()
        latency = time.time() - start_time
        self.log_performance("compute_gamma_bucket_exposure", latency, success=True)
        return exposure if not np.isnan(exposure) else 0.0
    except Exception as e:
        latency = time.time() - start_time
        error_msg = f"Erreur dans compute_gamma_bucket_exposure: {str(e)}\n{traceback.format_exc()}"
        self.log_performance(
            "compute_gamma_bucket_exposure", latency, success=False, error=str(e)
        )
        miya_alerts(error_msg, tag="OPTIONS", voice_profile="urgent", priority=3)
        send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)
        logger.error(error_msg)
        return 0.0

    def compute_delta_exposure_ratio(self, data: pd.DataFrame) -> float:
        """Calcule le ratio d’exposition delta call/put."""
        try:
            start_time = time.time()
            calls = data[data["option_type"] == "call"]
            puts = data[data["option_type"] == "put"]
            delta_calls = (calls["delta"] * calls["open_interest"]).sum()
            delta_puts = (puts["delta"] * puts["open_interest"]).sum()
            if delta_puts == 0:
                return 0.0
            ratio = delta_calls / delta_puts
            latency = time.time() - start_time
            self.log_performance("compute_delta_exposure_ratio", latency, success=True)
            return ratio if not np.isnan(ratio) else 0.0
        except Exception as e:
            self.log_performance(
                "compute_delta_exposure_ratio", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_delta_exposure_ratio: {str(e)}",
                tag="OPTIONS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_delta_exposure_ratio: {str(e)}", priority=3
            )
            send_telegram_alert(f"Erreur dans compute_delta_exposure_ratio: {str(e)}")
            logger.error(f"Erreur dans compute_delta_exposure_ratio: {str(e)}")
            return 0.0

    def compute_oi_change(self, data: pd.DataFrame, window: str = "1h") -> float:
        """Calcule le changement d’open interest sur une fenêtre donnée."""
        try:
            start_time = time.time()
            if len(data) < 2:
                return 0.0
            data = data.sort_values("timestamp")
            oi_diff = (
                data["open_interest"]
                .diff()
                .tail(pd.Timedelta(window) // pd.Timedelta("1min"))
                .sum()
            )
            latency = time.time() - start_time
            self.log_performance("compute_oi_change", latency, success=True)
            return oi_diff if not np.isnan(oi_diff) else 0.0
        except Exception as e:
            self.log_performance("compute_oi_change", 0, success=False, error=str(e))
            miya_alerts(
                f"Erreur dans compute_oi_change: {str(e)}",
                tag="OPTIONS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_oi_change: {str(e)}", priority=3)
            send_telegram_alert(f"Erreur dans compute_oi_change: {str(e)}")
            logger.error(f"Erreur dans compute_oi_change: {str(e)}")
            return 0.0

    def compute_gamma_trend(self, data: pd.DataFrame, window: str = "30min") -> float:
        """Calcule la tendance du gamma sur une fenêtre donnée."""
        try:
            start_time = time.time()
            if len(data) < 2:
                return 0.0
            data["gex"] = data["gamma"] * data["open_interest"]
            data = data.sort_values("timestamp")
            gex_trend = (
                data["gex"]
                .diff()
                .tail(pd.Timedelta(window) // pd.Timedelta("1min"))
                .mean()
            )
            latency = time.time() - start_time
            self.log_performance("compute_gamma_trend", latency, success=True)
            return gex_trend if not np.isnan(gex_trend) else 0.0
        except Exception as e:
            self.log_performance("compute_gamma_trend", 0, success=False, error=str(e))
            miya_alerts(
                f"Erreur dans compute_gamma_trend: {str(e)}",
                tag="OPTIONS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_gamma_trend: {str(e)}", priority=3)
            send_telegram_alert(f"Erreur dans compute_gamma_trend: {str(e)}")
            logger.error(f"Erreur dans compute_gamma_trend: {str(e)}")
            return 0.0

    def compute_vega_bucket_ratio(self, data: pd.DataFrame) -> float:
        """Calcule le ratio de vega concentré."""
        try:
            start_time = time.time()
            vega_total = (data["vega"] * data["open_interest"]).sum()
            if vega_total == 0:
                return 0.0
            top_vega = data.groupby("strike")["vega"].sum().nlargest(3).sum()
            ratio = top_vega / vega_total
            latency = time.time() - start_time
            self.log_performance("compute_vega_bucket_ratio", latency, success=True)
            return ratio if not np.isnan(ratio) else 0.0
        except Exception as e:
            self.log_performance(
                "compute_vega_bucket_ratio", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_vega_bucket_ratio: {str(e)}",
                tag="OPTIONS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_vega_bucket_ratio: {str(e)}", priority=3)
            send_telegram_alert(f"Erreur dans compute_vega_bucket_ratio: {str(e)}")
            logger.error(f"Erreur dans compute_vega_bucket_ratio: {str(e)}")
            return 0.0

    def compute_atm_straddle_cost(
        self, data: pd.DataFrame, underlying_price: float
    ) -> float:
        """Calcule le coût du straddle ATM."""
        try:
            start_time = time.time()
            data["strike_distance"] = np.abs(data["strike"] - underlying_price)
            atm_options = data[data["strike_distance"] == data["strike_distance"].min()]
            call_price = atm_options[atm_options["option_type"] == "call"][
                "price"
            ].mean()
            put_price = atm_options[atm_options["option_type"] == "put"]["price"].mean()
            cost = call_price + put_price
            latency = time.time() - start_time
            self.log_performance("compute_atm_straddle_cost", latency, success=True)
            return cost if not np.isnan(cost) else 0.0
        except Exception as e:
            self.log_performance(
                "compute_atm_straddle_cost", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_atm_straddle_cost: {str(e)}",
                tag="OPTIONS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_atm_straddle_cost: {str(e)}", priority=3)
            send_telegram_alert(f"Erreur dans compute_atm_straddle_cost: {str(e)}")
            logger.error(f"Erreur dans compute_atm_straddle_cost: {str(e)}")
            return 0.0

    def compute_spot_gamma_corridor(
        self, data: pd.DataFrame, range_pct: float = 1
    ) -> float:
        """Calcule le corridor gamma autour du spot."""
        try:
            start_time = time.time()
            underlying_price = data["underlying_price"].iloc[0]
            lower_bound = underlying_price * (1 - range_pct / 100)
            upper_bound = underlying_price * (1 + range_pct / 100)
            corridor = data[
                (data["strike"] >= lower_bound) & (data["strike"] <= upper_bound)
            ]
            corridor_gamma = (corridor["gamma"] * corridor["open_interest"]).sum()
            latency = time.time() - start_time
            self.log_performance("compute_spot_gamma_corridor", latency, success=True)
            return corridor_gamma if not np.isnan(corridor_gamma) else 0.0
        except Exception as e:
            self.log_performance(
                "compute_spot_gamma_corridor", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_spot_gamma_corridor: {str(e)}",
                tag="OPTIONS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_spot_gamma_corridor: {str(e)}", priority=3)
            send_telegram_alert(f"Erreur dans compute_spot_gamma_corridor: {str(e)}")
            logger.error(f"Erreur dans compute_spot_gamma_corridor: {str(e)}")
            return 0.0

    def compute_skewed_gamma_exposure(self, data: pd.DataFrame) -> float:
        """Calcule l’exposition gamma biaisée call/put."""
        try:
            start_time = time.time()
            calls = data[data["option_type"] == "call"]
            puts = data[data["option_type"] == "put"]
            call_gex = (calls["gamma"] * calls["open_interest"]).sum()
            put_gex = (puts["gamma"] * puts["open_interest"]).sum()
            skew = call_gex - put_gex
            latency = time.time() - start_time
            self.log_performance("compute_skewed_gamma_exposure", latency, success=True)
            return skew if not np.isnan(skew) else 0.0
        except Exception as e:
            self.log_performance(
                "compute_skewed_gamma_exposure", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_skewed_gamma_exposure: {str(e)}",
                tag="OPTIONS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_skewed_gamma_exposure: {str(e)}", priority=3
            )
            send_telegram_alert(f"Erreur dans compute_skewed_gamma_exposure: {str(e)}")
            logger.error(f"Erreur dans compute_skewed_gamma_exposure: {str(e)}")
            return 0.0

    def compute_oi_sweep_count(
        self, data: pd.DataFrame, threshold: float = 0.1
    ) -> float:
        """Compte les sweeps d’open interest dépassant un seuil."""
        try:
            start_time = time.time()
            if len(data) < 2:
                return 0.0
            oi_changes = data["open_interest"].pct_change().abs()
            sweeps = (oi_changes > threshold).sum()
            latency = time.time() - start_time
            self.log_performance("compute_oi_sweep_count", latency, success=True)
            return sweeps if not np.isnan(sweeps) else 0.0
        except Exception as e:
            self.log_performance(
                "compute_oi_sweep_count", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_oi_sweep_count: {str(e)}",
                tag="OPTIONS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_oi_sweep_count: {str(e)}", priority=3)
            send_telegram_alert(f"Erreur dans compute_oi_sweep_count: {str(e)}")
            logger.error(f"Erreur dans compute_oi_sweep_count: {str(e)}")
            return 0.0

    def compute_greek_exposure(self, data: pd.DataFrame, greek: str = "theta") -> float:
        """Calcule l’exposition pour un Greek donné."""
        try:
            start_time = time.time()
            if greek not in data.columns:
                return 0.0
            exposure = (data[greek] * data["open_interest"]).sum()
            latency = time.time() - start_time
            self.log_performance(f"compute_{greek}_exposure", latency, success=True)
            return exposure if not np.isnan(exposure) else 0.0
        except Exception as e:
            self.log_performance(
                f"compute_{greek}_exposure", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_{greek}_exposure: {str(e)}",
                tag="OPTIONS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_{greek}_exposure: {str(e)}", priority=3)
            send_telegram_alert(f"Erreur dans compute_{greek}_exposure: {str(e)}")
            logger.error(f"Erreur dans compute_{greek}_exposure: {str(e)}")
            return 0.0

    def compute_iv_slope_deltas(
        self, data: pd.DataFrame, deltas: List[float] = [10, 25, 50]
    ) -> Dict[str, float]:
        """Calcule la pente de la volatilité implicite pour des plages de delta."""
        try:
            start_time = time.time()
            results = {}
            for i in range(len(deltas) - 1):
                delta_lower = deltas[i]
                delta_upper = deltas[i + 1]
                chain = data[
                    (data["delta"].abs() >= delta_lower / 100)
                    & (data["delta"].abs() <= delta_upper / 100)
                ]
                if chain["strike"].nunique() < 2:
                    results[f"iv_slope_{delta_lower}_{delta_upper}"] = 0.0
                    continue
                iv_by_strike = chain.groupby("strike")["implied_volatility"].mean()
                iv_diff = iv_by_strike.diff()
                strike_diff = iv_by_strike.index.to_series().diff()
                slope = (iv_diff / strike_diff).mean()
                results[f"iv_slope_{delta_lower}_{delta_upper}"] = (
                    slope if not np.isnan(slope) else 0.0
                )
            latency = time.time() - start_time
            self.log_performance("compute_iv_slope_deltas", latency, success=True)
            return results
        except Exception as e:
            self.log_performance(
                "compute_iv_slope_deltas", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_iv_slope_deltas: {str(e)}",
                tag="OPTIONS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_iv_slope_deltas: {str(e)}", priority=3)
            send_telegram_alert(f"Erreur dans compute_iv_slope_deltas: {str(e)}")
            logger.error(f"Erreur dans compute_iv_slope_deltas: {str(e)}")
            return {
                f"iv_slope_{deltas[i]}_{deltas[i+1]}": 0.0
                for i in range(len(deltas) - 1)
            }

    def compute_vol_surface_curvature(self, data: pd.DataFrame) -> float:
        """Calcule la courbure de la surface de volatilité."""
        try:
            start_time = time.time()
            if data["strike"].nunique() < 3:
                return 0.0
            iv_by_strike = data.groupby("strike")["implied_volatility"].mean()
            second_deriv = iv_by_strike.diff().diff().mean()
            latency = time.time() - start_time
            self.log_performance("compute_vol_surface_curvature", latency, success=True)
            return second_deriv if not np.isnan(second_deriv) else 0.0
        except Exception as e:
            self.log_performance(
                "compute_vol_surface_curvature", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_vol_surface_curvature: {str(e)}",
                tag="OPTIONS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_vol_surface_curvature: {str(e)}", priority=3
            )
            send_telegram_alert(f"Erreur dans compute_vol_surface_curvature: {str(e)}")
            logger.error(f"Erreur dans compute_vol_surface_curvature: {str(e)}")
            return 0.0

    def compute_iv_acceleration(self, data: pd.DataFrame, window: str = "1h") -> float:
        """Calcule l’accélération de la volatilité implicite."""
        try:
            start_time = time.time()
            if len(data) < 3:
                return 0.0
            data = data.sort_values("timestamp")
            iv_velocity = data["implied_volatility"].diff()
            iv_acceleration = (
                iv_velocity.diff()
                .tail(pd.Timedelta(window) // pd.Timedelta("1min"))
                .mean()
            )
            latency = time.time() - start_time
            self.log_performance("compute_iv_acceleration", latency, success=True)
            return iv_acceleration if not np.isnan(iv_acceleration) else 0.0
        except Exception as e:
            self.log_performance(
                "compute_iv_acceleration", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_iv_acceleration: {str(e)}",
                tag="OPTIONS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_iv_acceleration: {str(e)}", priority=3)
            send_telegram_alert(f"Erreur dans compute_iv_acceleration: {str(e)}")
            logger.error(f"Erreur dans compute_iv_acceleration: {str(e)}")
            return 0.0

    def compute_oi_velocity(self, data: pd.DataFrame, window: str = "30min") -> float:
        """Calcule la vitesse de changement de l’open interest."""
        try:
            start_time = time.time()
            if len(data) < 2:
                return 0.0
            data = data.sort_values("timestamp")
            oi_velocity = (
                data["open_interest"]
                .diff()
                .tail(pd.Timedelta(window) // pd.Timedelta("1min"))
                .mean()
            )
            latency = time.time() - start_time
            self.log_performance("compute_oi_velocity", latency, success=True)
            return oi_velocity if not np.isnan(oi_velocity) else 0.0
        except Exception as e:
            self.log_performance("compute_oi_velocity", 0, success=False, error=str(e))
            miya_alerts(
                f"Erreur dans compute_oi_velocity: {str(e)}",
                tag="OPTIONS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_oi_velocity: {str(e)}", priority=3)
            send_telegram_alert(f"Erreur dans compute_oi_velocity: {str(e)}")
            logger.error(f"Erreur dans compute_oi_velocity: {str(e)}")
            return 0.0

    def compute_call_put_volume_ratio(self, data: pd.DataFrame) -> float:
        """Calcule le ratio de volume call/put."""
        try:
            start_time = time.time()
            calls = data[data["option_type"] == "call"]
            puts = data[data["option_type"] == "put"]
            call_volume = calls["volume"].sum()
            put_volume = puts["volume"].sum()
            if put_volume == 0:
                return 0.0
            ratio = call_volume / put_volume
            latency = time.time() - start_time
            self.log_performance("compute_call_put_volume_ratio", latency, success=True)
            return ratio if not np.isnan(ratio) else 0.0
        except Exception as e:
            self.log_performance(
                "compute_call_put_volume_ratio", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_call_put_volume_ratio: {str(e)}",
                tag="OPTIONS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_call_put_volume_ratio: {str(e)}", priority=3
            )
            send_telegram_alert(f"Erreur dans compute_call_put_volume_ratio: {str(e)}")
            logger.error(f"Erreur dans compute_call_put_volume_ratio: {str(e)}")
            return 0.0

    def compute_option_spread_cost(self, data: pd.DataFrame) -> float:
        """Calcule le coût moyen du spread bid-ask."""
        try:
            start_time = time.time()
            spread = (data["ask_price"] - data["bid_price"]).mean()
            latency = time.time() - start_time
            self.log_performance("compute_option_spread_cost", latency, success=True)
            return spread if not np.isnan(spread) else 0.0
        except Exception as e:
            self.log_performance(
                "compute_option_spread_cost", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_option_spread_cost: {str(e)}",
                tag="OPTIONS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_option_spread_cost: {str(e)}", priority=3)
            send_telegram_alert(f"Erreur dans compute_option_spread_cost: {str(e)}")
            logger.error(f"Erreur dans compute_option_spread_cost: {str(e)}")
            return 0.0


def calculate_options_features(
    self,
    data: pd.DataFrame,
    config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml"),
) -> pd.DataFrame:
    """Calcule les features d’options pour l’ensemble des données."""
    cache_path = os.path.join(
        CACHE_DIR,
        f"options_features_{hashlib.sha256(data.to_json().encode()).hexdigest()}.csv",
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
                    "calculate_options_features_cache_hit", latency, success=True
                )
                miya_speak(
                    "Features options chargées depuis cache",
                    tag="OPTIONS",
                    voice_profile="calm",
                    priority=1,
                )
                send_alert("Features options chargées depuis cache", priority=1)
                send_telegram_alert("Features options chargées depuis cache")
                return cached_data

        config = self.load_config(config_path)
        option_chain_path = config.get(
            "option_chain_path",
            os.path.join(BASE_DIR, "data", "options", "option_chain.csv"),
        )
        time_tolerance = config.get("time_tolerance", "10s")
        min_strikes = config.get("min_strikes", 5)
        expiration_window = config.get("expiration_window", "30d")

        if data.empty:
            error_msg = "DataFrame vide"
            miya_alerts(error_msg, tag="OPTIONS", voice_profile="urgent", priority=5)
            send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            raise ValueError(error_msg)

        data = data.copy()

        def load_option_chain():
            if not os.path.exists(option_chain_path):
                raise FileNotFoundError(f"Fichier {option_chain_path} introuvable")
            option_chain = pd.read_csv(option_chain_path)
            if option_chain["strike"].nunique() < min_strikes:
                raise ValueError(
                    f"Moins de {min_strikes} strikes uniques dans {option_chain_path}"
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
            "delta",
            "theta",
            "vega",
            "rho",
            "vomma",
            "speed",
            "ultima",
            "expiration_date",
            "price",
            "bid_price",
            "ask_price",
            "volume",
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
            send_telegram_alert(f"Colonne '{col}' manquante dans option_chain, imputée")
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
                send_telegram_alert("Colonne timestamp manquante, création par défaut")
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
            miya_alerts(error_msg, tag="OPTIONS", voice_profile="urgent", priority=5)
            send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            raise ValueError(error_msg)
        if not data["timestamp"].is_monotonic_increasing:
            error_msg = "Timestamps non croissants dans data"
            miya_alerts(error_msg, tag="OPTIONS", voice_profile="urgent", priority=5)
            send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            raise ValueError(error_msg)

        option_chain["expiration_date"] = pd.to_datetime(
            option_chain["expiration_date"], errors="coerce"
        )
        option_chain["expiration_days"] = (
            option_chain["expiration_date"] - option_chain["timestamp"]
        ).dt.days
        max_date = option_chain["timestamp"].max() + pd.Timedelta(expiration_window)
        option_chain = option_chain[option_chain["expiration_date"] <= max_date]

        features = [
            "gex",
            "iv_atm",
            "option_skew",
            "oi_concentration",
            "gex_slope",
            "oi_peak_call_near",
            "oi_peak_put_near",
            "gamma_wall",
            "vanna_strike_near",
            "delta_hedge_pressure",
            "iv_skew_10delta",
            "max_pain_strike",
            "gamma_velocity_near",
            "short_dated_iv_slope",
            "vol_term_structure_slope",
            "term_structure_skew",
            "risk_reversal_25",
            "butterfly_25",
            "oi_concentration_ratio",
            "gamma_bucket_exposure",
            "delta_exposure_ratio",
            "open_interest_change_1h",
            "gamma_trend_30m",
            "vega_bucket_ratio",
            "atm_straddle_cost",
            "spot_gamma_corridor",
            "skewed_gamma_exposure",
            "oi_sweep_count",
            "theta_exposure",
            "vega_exposure",
            "rho_exposure",
            "vomma_exposure",
            "speed_exposure",
            "ultima_exposure",
            "iv_slope_10_25",
            "iv_slope_25_50",
            "iv_slope_50_75",
            "vol_surface_curvature",
            "iv_acceleration",
            "oi_velocity",
            "call_put_volume_ratio",
            "option_spread_cost",
        ]
        self.validate_shap_features(features)

        for feature in features:
            data[feature] = 0.0

        # Calculer confidence_drop_rate (Phase 8)
        valid_cols = [
            col for col in required_option_cols if col in option_chain.columns
        ]
        confidence_drop_rate = 1.0 - min(
            len(valid_cols) / len(required_option_cols), 1.0
        )
        if option_chain["strike"].nunique() < min_strikes:
            confidence_drop_rate = max(confidence_drop_rate, 0.5)
        if confidence_drop_rate > 0.5:
            alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(valid_cols)}/{len(required_option_cols)} colonnes valides, {option_chain['strike'].nunique()} strikes)"
            miya_alerts(alert_msg, tag="OPTIONS", voice_profile="urgent", priority=3)
            send_alert(alert_msg, priority=3)
            send_telegram_alert(alert_msg)
            logger.warning(alert_msg)

        data = data.set_index("timestamp").sort_index()
        option_chain = option_chain.set_index("timestamp").sort_index()
        merged_data = pd.merge_asof(
            data,
            option_chain,
            left_index=True,
            right_index=True,
            tolerance=pd.Timedelta(time_tolerance),
            direction="nearest",
        ).reset_index()
        if len(merged_data) / len(data) < 0.5:
            error_msg = "Moins de 50% des timestamps alignés"
            miya_alerts(error_msg, tag="OPTIONS", voice_profile="urgent", priority=5)
            send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            raise ValueError(error_msg)

        grouped = merged_data.groupby("timestamp")
        scaler = MinMaxScaler()
        data["gex"] = grouped.apply(lambda x: self.calculate_gex(x)).reindex(
            data.index, fill_value=0
        )
        data["iv_atm"] = grouped.apply(
            lambda x: self.calculate_iv_atm(x, x["close"].iloc[0] if not x.empty else 0)
        ).reindex(data.index, fill_value=0)
        data["option_skew"] = grouped.apply(
            lambda x: self.calculate_option_skew(x)
        ).reindex(data.index, fill_value=0)
        data["oi_concentration"] = grouped.apply(
            lambda x: self.calculate_oi_concentration(x)
        ).reindex(data.index, fill_value=0)
        data["gex_slope"] = grouped.apply(
            lambda x: self.calculate_gex_slope(
                x, x["close"].iloc[0] if not x.empty else 0
            )
        ).reindex(data.index, fill_value=0)
        data["oi_peak_call_near"] = grouped.apply(
            lambda x: (
                x[x["option_type"] == "call"]["open_interest"].max()
                if not x[x["option_type"] == "call"].empty
                else 0
            )
        ).reindex(data.index, fill_value=0)
        data["oi_peak_put_near"] = grouped.apply(
            lambda x: (
                x[x["option_type"] == "put"]["open_interest"].max()
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
        data["vanna_strike_near"] = grouped.apply(
            lambda x: (
                (x["delta"].diff() / (x["implied_volatility"].diff() + 1e-6)).mean()
                if not x[x["strike_distance"] == x["strike_distance"].min()].empty
                else 0
            )
        ).reindex(data.index, fill_value=0)
        data["vanna_strike_near"] = scaler.fit_transform(data[["vanna_strike_near"]])[
            :, 0
        ]
        data["delta_hedge_pressure"] = grouped.apply(
            lambda x: (x["delta"] * x["open_interest"]).sum()
        ).reindex(data.index, fill_value=0)
        data["iv_skew_10delta"] = grouped.apply(
            lambda x: (
                x.groupby("strike")["implied_volatility"].mean().iloc[-1]
                - x.groupby("strike")["implied_volatility"].mean().iloc[0]
                if len(x.groupby("strike")) >= 2
                else 0
            )
        ).reindex(data.index, fill_value=0)
        data["iv_skew_10delta"] = scaler.fit_transform(data[["iv_skew_10delta"]])[:, 0]
        data["max_pain_strike"] = grouped.apply(
            lambda x: (
                (
                    (np.maximum(x["strike"] - x["close"], 0) * x["open_interest"])
                    .where(x["option_type"] == "call")
                    .sum()
                    + (np.maximum(x["close"] - x["strike"], 0) * x["open_interest"])
                    .where(x["option_type"] == "put")
                    .sum()
                )
                .groupby(x["strike"])
                .sum()
                .idxmin()
                if not x.empty
                else 0
            )
        ).reindex(data.index, fill_value=0)
        data["gamma_velocity_near"] = data["gamma_wall"].diff().fillna(0)
        iv_slopes = grouped.apply(lambda x: self.compute_iv_slope(x)).reindex(
            data.index, fill_value=0
        )
        for key in ["iv_slope_7d", "iv_slope_30d"]:
            data[key] = iv_slopes.apply(lambda x: x.get(key, 0.0))
        data["term_structure_skew"] = grouped.apply(
            lambda x: self.compute_term_structure_skew(x)
        ).reindex(data.index, fill_value=0)
        data["risk_reversal_25"] = grouped.apply(
            lambda x: self.compute_risk_reversal(x)
        ).reindex(data.index, fill_value=0)
        data["butterfly_25"] = grouped.apply(
            lambda x: self.compute_butterfly(x)
        ).reindex(data.index, fill_value=0)
        data["oi_concentration_ratio"] = grouped.apply(
            lambda x: self.compute_oi_concentration_ratio(x)
        ).reindex(data.index, fill_value=0)
        data["gamma_bucket_exposure"] = grouped.apply(
            lambda x: self.compute_gamma_bucket_exposure(x)
        ).reindex(data.index, fill_value=0)
        data["delta_exposure_ratio"] = grouped.apply(
            lambda x: self.compute_delta_exposure_ratio(x)
        ).reindex(data.index, fill_value=0)
        data["open_interest_change_1h"] = grouped.apply(
            lambda x: self.compute_oi_change(x)
        ).reindex(data.index, fill_value=0)
        data["gamma_trend_30m"] = grouped.apply(
            lambda x: self.compute_gamma_trend(x)
        ).reindex(data.index, fill_value=0)
        data["vega_bucket_ratio"] = grouped.apply(
            lambda x: self.compute_vega_bucket_ratio(x)
        ).reindex(data.index, fill_value=0)
        data["atm_straddle_cost"] = grouped.apply(
            lambda x: self.compute_atm_straddle_cost(
                x, x["close"].iloc[0] if not x.empty else 0
            )
        ).reindex(data.index, fill_value=0)
        data["spot_gamma_corridor"] = grouped.apply(
            lambda x: self.compute_spot_gamma_corridor(x)
        ).reindex(data.index, fill_value=0)
        data["skewed_gamma_exposure"] = grouped.apply(
            lambda x: self.compute_skewed_gamma_exposure(x)
        ).reindex(data.index, fill_value=0)
        data["oi_sweep_count"] = grouped.apply(
            lambda x: self.compute_oi_sweep_count(x)
        ).reindex(data.index, fill_value=0)
        for greek in ["theta", "vega", "rho", "vomma", "speed", "ultima"]:
            data[f"{greek}_exposure"] = grouped.apply(
                lambda x: self.compute_greek_exposure(x, greek)
            ).reindex(data.index, fill_value=0)
        iv_slope_deltas = grouped.apply(
            lambda x: self.compute_iv_slope_deltas(x)
        ).reindex(data.index, fill_value=0)
        for key in ["iv_slope_10_25", "iv_slope_25_50", "iv_slope_50_75"]:
            data[key] = iv_slope_deltas.apply(lambda x: x.get(key, 0.0))
        data["vol_surface_curvature"] = grouped.apply(
            lambda x: self.compute_vol_surface_curvature(x)
        ).reindex(data.index, fill_value=0)
        data["iv_acceleration"] = grouped.apply(
            lambda x: self.compute_iv_acceleration(x)
        ).reindex(data.index, fill_value=0)
        data["oi_velocity"] = grouped.apply(
            lambda x: self.compute_oi_velocity(x)
        ).reindex(data.index, fill_value=0)
        data["call_put_volume_ratio"] = grouped.apply(
            lambda x: self.compute_call_put_volume_ratio(x)
        ).reindex(data.index, fill_value=0)
        data["option_spread_cost"] = grouped.apply(
            lambda x: self.compute_option_spread_cost(x)
        ).reindex(data.index, fill_value=0)

        for feature in features:
            if data[feature].isna().any():
                data[feature] = data[feature].fillna(0)

        os.makedirs(CACHE_DIR, exist_ok=True)

        def write_cache():
            data.to_csv(cache_path, index=False, encoding="utf-8")

        self.with_retries(write_cache)
        file_size = os.path.getsize(cache_path) / 1024 / 1024
        if file_size > 1.0:
            miya_alerts(
                f"Cache options_features size {file_size:.2f} MB exceeds 1 MB",
                tag="OPTIONS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Cache options_features size {file_size:.2f} MB exceeds 1 MB",
                priority=3,
            )
            send_telegram_alert(
                f"Cache options_features size {file_size:.2f} MB exceeds 1 MB"
            )

        latency = time.time() - start_time
        miya_speak(
            "Features options calculées",
            tag="OPTIONS",
            voice_profile="calm",
            priority=1,
        )
        send_alert("Features options calculées", priority=1)
        send_telegram_alert("Features options calculées")
        logger.info("Features options calculées")
        self.log_performance(
            "calculate_options_features",
            latency,
            success=True,
            num_rows=len(data),
            num_features=len(features),
            confidence_drop_rate=confidence_drop_rate,
        )
        self.save_snapshot(
            "calculate_options_features",
            {
                "num_rows": len(data),
                "num_features": len(features),
                "confidence_drop_rate": confidence_drop_rate,
            },
            compress=False,
        )
        self.plot_options_features(
            data.reset_index(), data.index[-1].strftime("%Y-%m-%d %H:%M:%S")
        )
        return data.reset_index()
    except Exception as e:
        latency = time.time() - start_time
        error_msg = f"Erreur dans calculate_options_features: {str(e)}\n{traceback.format_exc()}"
        miya_alerts(error_msg, tag="OPTIONS", voice_profile="urgent", priority=3)
        send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)
        logger.error(error_msg)
        self.log_performance(
            "calculate_options_features", latency, success=False, error=str(e)
        )
        self.save_snapshot(
            "calculate_options_features", {"error": str(e)}, compress=False
        )
        for feature in features:
            data[feature] = 0.0
        return data.reset_index()


def calculate_incremental_options_features(
    self,
    row: pd.Series,
    option_chain: pd.DataFrame,
    config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml"),
) -> pd.Series:
    """Calcule les features d’options pour une seule ligne."""
    try:
        start_time = time.time()
        config = self.load_config(config_path)
        min_strikes = config.get("min_strikes", 5)
        expiration_window = config.get("expiration_window", "30d")

        features = {
            f: 0.0
            for f in [
                "gex",
                "iv_atm",
                "option_skew",
                "oi_concentration",
                "gex_slope",
                "oi_peak_call_near",
                "oi_peak_put_near",
                "gamma_wall",
                "vanna_strike_near",
                "delta_hedge_pressure",
                "iv_skew_10delta",
                "max_pain_strike",
                "gamma_velocity_near",
                "short_dated_iv_slope",
                "vol_term_structure_slope",
                "term_structure_skew",
                "risk_reversal_25",
                "butterfly_25",
                "oi_concentration_ratio",
                "gamma_bucket_exposure",
                "delta_exposure_ratio",
                "open_interest_change_1h",
                "gamma_trend_30m",
                "vega_bucket_ratio",
                "atm_straddle_cost",
                "spot_gamma_corridor",
                "skewed_gamma_exposure",
                "oi_sweep_count",
                "theta_exposure",
                "vega_exposure",
                "rho_exposure",
                "vomma_exposure",
                "speed_exposure",
                "ultima_exposure",
                "iv_slope_10_25",
                "iv_slope_25_50",
                "iv_slope_50_75",
                "vol_surface_curvature",
                "iv_acceleration",
                "oi_velocity",
                "call_put_volume_ratio",
                "option_spread_cost",
            ]
        }

        row = row.copy()
        row["timestamp"] = pd.to_datetime(row["timestamp"], errors="coerce")
        if pd.isna(row["timestamp"]):
            error_msg = "Timestamp invalide dans la ligne"
            miya_alerts(error_msg, tag="OPTIONS", voice_profile="urgent", priority=5)
            send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            raise ValueError(error_msg)

        current_chain = option_chain[option_chain["timestamp"] == row["timestamp"]]
        if current_chain.empty or current_chain["strike"].nunique() < min_strikes:
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
            "delta",
            "theta",
            "vega",
            "rho",
            "vomma",
            "speed",
            "ultima",
            "expiration_date",
            "price",
            "bid_price",
            "ask_price",
            "volume",
        ]
        valid_cols = [col for col in required_cols if col in current_chain.columns]
        confidence_drop_rate = 1.0 - min(len(valid_cols) / len(required_cols), 1.0)
        if current_chain["strike"].nunique() < min_strikes:
            confidence_drop_rate = max(confidence_drop_rate, 0.5)
        if confidence_drop_rate > 0.5:
            alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(valid_cols)}/{len(required_cols)} colonnes valides, {current_chain['strike'].nunique()} strikes)"
            miya_alerts(alert_msg, tag="OPTIONS", voice_profile="urgent", priority=3)
            send_alert(alert_msg, priority=3)
            send_telegram_alert(alert_msg)
            logger.warning(alert_msg)

        current_chain["expiration_date"] = pd.to_datetime(
            current_chain["expiration_date"], errors="coerce"
        )
        current_chain["expiration_days"] = (
            current_chain["expiration_date"] - current_chain["timestamp"]
        ).dt.days
        max_date = row["timestamp"] + pd.Timedelta(expiration_window)
        current_chain = current_chain[current_chain["expiration_date"] <= max_date]

        underlying_price = row["close"]
        current_chain["gex"] = current_chain["gamma"] * current_chain["open_interest"]
        calls = current_chain[current_chain["option_type"] == "call"]
        puts = current_chain[current_chain["option_type"] == "put"]

        scaler = MinMaxScaler()
        features["gex"] = (
            calls["gamma"] * calls["open_interest"] * 100 * underlying_price
        ).sum() - (puts["gamma"] * puts["open_interest"] * 100 * underlying_price).sum()
        features["gex"] = (
            scaler.fit_transform([[features["gex"]]])[0, 0]
            if features["gex"] != 0
            else 0
        )
        features["iv_atm"] = self.calculate_iv_atm(current_chain, underlying_price)
        features["option_skew"] = self.calculate_option_skew(current_chain)
        features["oi_concentration"] = self.calculate_oi_concentration(current_chain)
        features["gex_slope"] = self.calculate_gex_slope(
            current_chain, underlying_price
        )
        features["oi_peak_call_near"] = (
            calls["open_interest"].max() if not calls.empty else 0
        )
        features["oi_peak_call_near"] = (
            scaler.fit_transform([[features["oi_peak_call_near"]]])[0, 0]
            if features["oi_peak_call_near"] != 0
            else 0
        )
        features["oi_peak_put_near"] = (
            puts["open_interest"].max() if not puts.empty else 0
        )
        features["oi_peak_put_near"] = (
            scaler.fit_transform([[features["oi_peak_put_near"]]])[0, 0]
            if features["oi_peak_put_near"] != 0
            else 0
        )
        features["gamma_wall"] = (
            current_chain.groupby("strike")["gex"].sum().idxmax()
            if not current_chain.empty and "gex" in current_chain.columns
            else 0
        )
        current_chain["strike_distance"] = np.abs(
            current_chain["strike"] - underlying_price
        )
        atm_options = current_chain.loc[
            current_chain["strike_distance"] == current_chain["strike_distance"].min()
        ]
        features["vanna_strike_near"] = (
            (
                atm_options["delta"].diff()
                / (atm_options["implied_volatility"].diff() + 1e-6)
            ).mean()
            if not atm_options.empty and len(atm_options) >= 2
            else 0
        )
        features["vanna_strike_near"] = scaler.fit_transform(
            [[features["vanna_strike_near"]]]
        )[0, 0]
        features["delta_hedge_pressure"] = (
            current_chain["delta"] * current_chain["open_interest"]
        ).sum()
        features["delta_hedge_pressure"] = (
            scaler.fit_transform([[features["delta_hedge_pressure"]]])[0, 0]
            if features["delta_hedge_pressure"] != 0
            else 0
        )
        iv_by_strike = current_chain.groupby("strike")["implied_volatility"].mean()
        features["iv_skew_10delta"] = (
            iv_by_strike.iloc[-1] - iv_by_strike.iloc[0]
            if len(iv_by_strike) >= 2
            else 0
        )
        features["iv_skew_10delta"] = scaler.fit_transform(
            [[features["iv_skew_10delta"]]]
        )[0, 0]
        total_loss = pd.Series(0, index=current_chain.index)
        for idx, opt in current_chain.iterrows():
            strike = opt["strike"]
            oi = opt["open_interest"]
            total_loss.loc[idx] = (
                np.maximum(strike - underlying_price, 0) * oi
                if opt["option_type"] == "call"
                else np.maximum(underlying_price - strike, 0) * oi
            )
        features["max_pain_strike"] = (
            total_loss.groupby(current_chain["strike"]).sum().idxmin()
            if not current_chain.empty
            else 0
        )
        self.buffer.append(
            pd.Series({"gamma_wall": features["gamma_wall"]}, name=row["timestamp"])
        )
        if len(self.buffer) > 1:
            features["gamma_velocity_near"] = (
                self.buffer[-1]["gamma_wall"] - self.buffer[-2]["gamma_wall"]
            )
        iv_slopes = self.compute_iv_slope(current_chain)
        for key in ["iv_slope_7d", "iv_slope_30d"]:
            features[key] = iv_slopes.get(key, 0.0)
        features["short_dated_iv_slope"] = features["iv_slope_7d"]
        features["vol_term_structure_slope"] = features["iv_slope_30d"]
        features["term_structure_skew"] = self.compute_term_structure_skew(
            current_chain
        )
        features["risk_reversal_25"] = self.compute_risk_reversal(current_chain)
        features["butterfly_25"] = self.compute_butterfly(current_chain)
        features["oi_concentration_ratio"] = self.compute_oi_concentration_ratio(
            current_chain
        )
        features["gamma_bucket_exposure"] = self.compute_gamma_bucket_exposure(
            current_chain
        )
        features["delta_exposure_ratio"] = self.compute_delta_exposure_ratio(
            current_chain
        )
        features["open_interest_change_1h"] = self.compute_oi_change(current_chain)
        features["gamma_trend_30m"] = self.compute_gamma_trend(current_chain)
        features["vega_bucket_ratio"] = self.compute_vega_bucket_ratio(current_chain)
        features["atm_straddle_cost"] = self.compute_atm_straddle_cost(
            current_chain, underlying_price
        )
        features["spot_gamma_corridor"] = self.compute_spot_gamma_corridor(
            current_chain
        )
        features["skewed_gamma_exposure"] = self.compute_skewed_gamma_exposure(
            current_chain
        )
        features["oi_sweep_count"] = self.compute_oi_sweep_count(current_chain)
        for greek in ["theta", "vega", "rho", "vomma", "speed", "ultima"]:
            features[f"{greek}_exposure"] = self.compute_greek_exposure(
                current_chain, greek
            )
        iv_slope_deltas = self.compute_iv_slope_deltas(current_chain)
        for key in ["iv_slope_10_25", "iv_slope_25_50", "iv_slope_50_75"]:
            features[key] = iv_slope_deltas.get(key, 0.0)
        features["vol_surface_curvature"] = self.compute_vol_surface_curvature(
            current_chain
        )
        features["iv_acceleration"] = self.compute_iv_acceleration(current_chain)
        features["oi_velocity"] = self.compute_oi_velocity(current_chain)
        features["call_put_volume_ratio"] = self.compute_call_put_volume_ratio(
            current_chain
        )
        features["option_spread_cost"] = self.compute_option_spread_cost(current_chain)

        latency = time.time() - start_time
        miya_speak(
            "Incremental features options calculées",
            tag="OPTIONS",
            voice_profile="calm",
            priority=1,
        )
        send_alert("Incremental features options calculées", priority=1)
        send_telegram_alert("Incremental features options calculées")
        logger.info("Incremental features options calculées")
        self.log_performance(
            "calculate_incremental_options_features",
            latency,
            success=True,
            num_rows=1,
            confidence_drop_rate=confidence_drop_rate,
        )
        self.save_snapshot(
            "calculate_incremental_options_features",
            {
                "features": {k: float(v) for k, v in features.items()},
                "confidence_drop_rate": confidence_drop_rate,
            },
            compress=False,
        )
        return pd.Series(
            {k: v if not np.isnan(v) else 0.0 for k, v in features.items()}
        )
    except Exception as e:
        latency = time.time() - start_time
        error_msg = f"Erreur dans calculate_incremental_options_features: {str(e)}\n{traceback.format_exc()}"
        miya_alerts(error_msg, tag="OPTIONS", voice_profile="urgent", priority=3)
        send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)
        logger.error(error_msg)
        self.log_performance(
            "calculate_incremental_options_features",
            latency,
            success=False,
            error=str(e),
        )
        self.save_snapshot(
            "calculate_incremental_options_features", {"error": str(e)}, compress=False
        )
        return pd.Series(features)


if __name__ == "__main__":
    try:
        calculator = OptionsCalculator()
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
                "delta": np.random.uniform(-0.5, 0.5, 30),
                "theta": np.random.uniform(-0.1, -0.01, 30),
                "vega": np.random.uniform(0.05, 0.15, 30),
                "rho": np.random.uniform(-0.02, 0.02, 30),
                "vomma": np.random.uniform(0.01, 0.03, 30),
                "speed": np.random.uniform(-0.01, 0.01, 30),
                "ultima": np.random.uniform(-0.005, 0.005, 30),
                "expiration_date": ["2025-04-21"] * 30,
                "price": np.random.uniform(1.0, 10.0, 30),
                "bid_price": np.random.uniform(0.8, 9.0, 30),
                "ask_price": np.random.uniform(1.0, 10.0, 30),
                "volume": np.random.randint(10, 100, 30),
            }
        )
        os.makedirs(os.path.join(BASE_DIR, "data", "options"), exist_ok=True)
        option_chain.to_csv(
            os.path.join(BASE_DIR, "data", "options", "option_chain.csv"), index=False
        )
        enriched_data = calculator.calculate_options_features(data)
        print(enriched_data.head())
        miya_speak(
            "Test calculate_options_features terminé",
            tag="TEST",
            voice_profile="calm",
            priority=1,
        )
        send_alert("Test calculate_options_features terminé", priority=1)
        send_telegram_alert("Test calculate_options_features terminé")
        logger.info("Test calculate_options_features terminé")
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
