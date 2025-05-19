# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/features/extractors/orderflow_indicators.py
# Calcule les indicateurs d'order flow (delta_volume, obi_score, vds, absorption_strength, bid_ask_imbalance, etc.) à partir des données IQFeed.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Génère 16 indicateurs d’order flow avec pondération selon le régime de marché (méthode 3),
#        utilise un cache intermédiaire, enregistre des logs psutil, valide SHAP (méthode 17),
#        normalise les métriques avec MinMaxScaler pour l’inférence, et inclut 11 métriques avancées
#        (ex. : bid_ask_imbalance, effective_spread).
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
# - data/features/cache/orderflow/
# - data/logs/orderflow_performance.csv
# - data/features/orderflow_snapshots/
# - data/figures/orderflow/
#
# Notes :
# - Utilise exclusivement IQFeed pour les données (remplace dxFeed).
# - Suppression de toute référence à dxFeed, obs_t, 320 features, et 81 features.
# - Intègre les Phases 1-18 :
#   - Phase 3 (order_flow.py) : Métriques delta_volume, obi_score, vds, absorption_strength, etc.
#   - Phase 13 (orderflow_indicators.py, options_metrics.py) : Métriques avancées comme bid_ask_imbalance, effective_spread.
# - Tests unitaires disponibles dans tests/test_orderflow_indicators.py.
# - Pondération des indicateurs selon le régime (trend: 1.5, range: 1.0, defensive: 0.5).
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
CACHE_DIR = os.path.join(BASE_DIR, "data", "features", "cache", "orderflow")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "features", "orderflow_snapshots")
CSV_LOG_PATH = os.path.join(BASE_DIR, "data", "logs", "orderflow_performance.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures", "orderflow")
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
    filename=os.path.join(BASE_DIR, "data", "logs", "orderflow_indicators.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 2.0
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes
WINDOW_SIZE = 100  # Fenêtre glissante pour MinMaxScaler


class OrderFlowIndicators:
    """
    Classe pour calculer les indicateurs d’order flow avec pondération, cache, normalisation, et validation SHAP.
    """

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
                "OrderFlowIndicators initialisé",
                tag="ORDERFLOW",
                voice_profile="calm",
                priority=2,
            )
            send_alert("OrderFlowIndicators initialisé", priority=1)
            logger.info("OrderFlowIndicators initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config_path": config_path})
        except Exception as e:
            error_msg = f"Erreur initialisation OrderFlowIndicators: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="ORDERFLOW", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            self.config = {
                "buffer_size": 100,
                "max_cache_size": 1000,
                "cache_hours": 24,
                "weights": {"trend": 1.5, "range": 1.0, "defensive": 0.5},
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
            if "orderflow_indicators" not in config:
                raise ValueError(
                    "Clé 'orderflow_indicators' manquante dans la configuration"
                )
            required_keys = ["buffer_size", "max_cache_size", "cache_hours", "weights"]
            missing_keys = [
                key
                for key in required_keys
                if key not in config["orderflow_indicators"]
            ]
            if missing_keys:
                raise ValueError(
                    f"Clés manquantes dans 'orderflow_indicators': {missing_keys}"
                )
            return config["orderflow_indicators"]

        try:
            start_time = time.time()
            config = self.with_retries(load_yaml)
            latency = time.time() - start_time
            miya_speak(
                "Configuration orderflow_indicators chargée via config_manager",
                tag="ORDERFLOW",
                voice_profile="calm",
                priority=2,
            )
            send_alert(
                "Configuration orderflow_indicators chargée via config_manager",
                priority=1,
            )
            logger.info("Configuration orderflow_indicators chargée via config_manager")
            self.log_performance("load_config_with_manager", latency, success=True)
            self.save_snapshot("load_config_with_manager", {"config_path": config_path})
            return config
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur chargement config via config_manager: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="ORDERFLOW", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance(
                "load_config_with_manager", latency, success=False, error=str(e)
            )
            return {
                "buffer_size": 100,
                "max_cache_size": 1000,
                "cache_hours": 24,
                "weights": {"trend": 1.5, "range": 1.0, "defensive": 0.5},
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
                    tag="ORDERFLOW",
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
                tag="ORDERFLOW",
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
                tag="ORDERFLOW",
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
                tag="ORDERFLOW",
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
                tag="ORDERFLOW",
                voice_profile="calm",
                priority=2,
            )
            send_alert("SIGINT received, snapshot saved", priority=2)
            logger.info("SIGINT received, snapshot saved")
        except Exception as e:
            miya_alerts(
                f"Erreur gestion SIGINT: {str(e)}",
                tag="ORDERFLOW",
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
                    tag="ORDERFLOW",
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
                    tag="ORDERFLOW",
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
                    tag="ORDERFLOW",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(
                    f"Features non incluses dans top 150 SHAP: {missing}", priority=3
                )
                logger.warning(f"Features non incluses dans top 150 SHAP: {missing}")
            miya_speak(
                "SHAP features validées",
                tag="ORDERFLOW",
                voice_profile="calm",
                priority=1,
            )
            send_alert("SHAP features validées", priority=1)
            logger.info("SHAP features validées")
            return True
        except Exception as e:
            miya_alerts(
                f"Erreur validation SHAP features: {str(e)}",
                tag="ORDERFLOW",
                voice_profile="urgent",
                priority=4,
            )
            send_alert(f"Erreur validation SHAP features: {str(e)}", priority=4)
            logger.error(f"Erreur validation SHAP features: {str(e)}")
            return False

    def validate_iqfeed_data(self, data: pd.DataFrame) -> bool:
        """
        Valide les données IQFeed pour les indicateurs d’order flow.
        """
        try:
            required_cols = [
                "timestamp",
                "bid_size_level_1",
                "ask_size_level_1",
                "bid_price_level_1",
                "ask_price_level_1",
                "bid_size_level_2",
                "ask_size_level_2",
                "bid_price_level_2",
                "ask_price_level_2",
                "trade_frequency_1s",
            ]
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                error_msg = f"Colonnes IQFeed manquantes: {missing_cols}"
                miya_alerts(
                    error_msg, tag="ORDERFLOW", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return False
            if data[required_cols].isna().any().any():
                error_msg = "Valeurs NaN dans les données IQFeed"
                miya_alerts(
                    error_msg, tag="ORDERFLOW", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return False
            if not np.isfinite(
                data[required_cols].select_dtypes(include=[np.number]).values
            ).all():
                error_msg = "Valeurs infinies dans les données IQFeed"
                miya_alerts(
                    error_msg, tag="ORDERFLOW", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return False
            miya_speak(
                "Données IQFeed validées",
                tag="ORDERFLOW",
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
            miya_alerts(error_msg, tag="ORDERFLOW", voice_profile="urgent", priority=4)
            send_alert(error_msg, priority=4)
            logger.error(error_msg)
            return False

    def cache_indicators(self, data: pd.DataFrame, cache_key: str) -> None:
        """
        Met en cache les indicateurs d’order flow.
        """
        try:
            start_time = time.time()
            cache_path = os.path.join(CACHE_DIR, f"{cache_key}.csv")
            data.to_csv(cache_path, index=False, encoding="utf-8")
            self.cache[cache_key] = {"timestamp": datetime.now(), "path": cache_path}
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
                f"Indicateurs mis en cache: {cache_path}",
                tag="ORDERFLOW",
                voice_profile="calm",
                priority=1,
            )
            send_alert(f"Indicateurs mis en cache: {cache_path}", priority=1)
            logger.info(f"Indicateurs mis en cache: {cache_path}")
            self.log_performance(
                "cache_indicators", latency, success=True, cache_size=len(self.cache)
            )
        except Exception as e:
            self.log_performance("cache_indicators", 0, success=False, error=str(e))
            miya_alerts(
                f"Erreur mise en cache indicateurs: {str(e)}",
                tag="ORDERFLOW",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur mise en cache indicateurs: {str(e)}", priority=4)
            logger.error(f"Erreur mise en cache indicateurs: {str(e)}")

    def plot_indicators(self, data: pd.DataFrame, timestamp: str) -> None:
        """
        Génère des visualisations des indicateurs d’order flow.
        """
        try:
            start_time = time.time()
            timestamp_safe = timestamp.replace(":", "-")
            plt.figure(figsize=(12, 6))
            plt.plot(
                data["timestamp"],
                data["delta_volume"],
                label="Delta Volume",
                color="blue",
            )
            plt.plot(
                data["timestamp"], data["obi_score"], label="OBI Score", color="orange"
            )
            plt.plot(
                data["timestamp"],
                data["bid_ask_imbalance"],
                label="Bid-Ask Imbalance",
                color="green",
            )
            plt.plot(
                data["timestamp"],
                data["effective_spread"],
                label="Effective Spread",
                color="red",
            )
            plt.title(f"Order Flow Indicators - {timestamp}")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.savefig(
                os.path.join(FIGURES_DIR, f"orderflow_indicators_{timestamp_safe}.png")
            )
            plt.close()
            latency = time.time() - start_time
            miya_speak(
                f"Visualisations générées: {FIGURES_DIR}",
                tag="ORDERFLOW",
                voice_profile="calm",
                priority=2,
            )
            send_alert(f"Visualisations générées: {FIGURES_DIR}", priority=2)
            logger.info(f"Visualisations générées: {FIGURES_DIR}")
            self.log_performance("plot_indicators", latency, success=True)
        except Exception as e:
            self.log_performance("plot_indicators", 0, success=False, error=str(e))
            miya_alerts(
                f"Erreur génération visualisations: {str(e)}",
                tag="ORDERFLOW",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur génération visualisations: {str(e)}", priority=4)
            logger.error(f"Erreur génération visualisations: {str(e)}")

    def compute_bid_ask_imbalance(self, data: pd.DataFrame) -> pd.Series:
        """
        Calcule l’imbalance bid-ask sur les niveaux 1-5 (Phase 3).
        """
        try:
            start_time = time.time()
            bid_cols = [
                f"bid_size_level_{i}"
                for i in range(1, 6)
                if f"bid_size_level_{i}" in data.columns
            ]
            ask_cols = [
                f"ask_size_level_{i}"
                for i in range(1, 6)
                if f"ask_size_level_{i}" in data.columns
            ]
            if not bid_cols or not ask_cols:
                error_msg = "Colonnes bid/ask manquantes pour niveaux 1-5"
                miya_alerts(
                    error_msg, tag="ORDERFLOW", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=data.index, name="bid_ask_imbalance")
            bid_sum = data[bid_cols].sum(axis=1)
            ask_sum = data[ask_cols].sum(axis=1)
            imbalance = (bid_sum - ask_sum) / (bid_sum + ask_sum + 1e-6)
            result = pd.Series(imbalance, index=data.index).fillna(0.0)
            result.name = "bid_ask_imbalance"
            latency = time.time() - start_time
            self.log_performance("compute_bid_ask_imbalance", latency, success=True)
            return result
        except Exception as e:
            self.log_performance(
                "compute_bid_ask_imbalance", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_bid_ask_imbalance: {str(e)}",
                tag="ORDERFLOW",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_bid_ask_imbalance: {str(e)}", priority=4)
            logger.error(f"Erreur dans compute_bid_ask_imbalance: {str(e)}")
            return pd.Series(0.0, index=data.index, name="bid_ask_imbalance")

    def compute_effective_spread(self, data: pd.DataFrame) -> pd.Series:
        """
        Calcule l’écart effectif (Phase 13).
        """
        try:
            start_time = time.time()
            if (
                "trade_price" not in data.columns
                or "bid_price_level_1" not in data.columns
                or "ask_price_level_1" not in data.columns
            ):
                error_msg = "Colonnes trade_price, bid_price_level_1 ou ask_price_level_1 manquantes"
                miya_alerts(
                    error_msg, tag="ORDERFLOW", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=data.index, name="effective_spread")
            mid_price = (data["bid_price_level_1"] + data["ask_price_level_1"]) / 2
            spread = 2 * (data["trade_price"] - mid_price).abs() / mid_price
            result = pd.Series(spread, index=data.index).fillna(0.0)
            result.name = "effective_spread"
            latency = time.time() - start_time
            self.log_performance("compute_effective_spread", latency, success=True)
            return result
        except Exception as e:
            self.log_performance(
                "compute_effective_spread", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_effective_spread: {str(e)}",
                tag="ORDERFLOW",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_effective_spread: {str(e)}", priority=4)
            logger.error(f"Erreur dans compute_effective_spread: {str(e)}")
            return pd.Series(0.0, index=data.index, name="effective_spread")

    def compute_realized_spread(self, data: pd.DataFrame) -> pd.Series:
        """
        Calcule l’écart réalisé (Phase 13).
        """
        try:
            start_time = time.time()
            if (
                "trade_price" not in data.columns
                or "bid_price_level_1" not in data.columns
                or "ask_price_level_1" not in data.columns
            ):
                error_msg = "Colonnes trade_price, bid_price_level_1 ou ask_price_level_1 manquantes"
                miya_alerts(
                    error_msg, tag="ORDERFLOW", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=data.index, name="realized_spread")
            mid_price = (data["bid_price_level_1"] + data["ask_price_level_1"]) / 2
            future_mid_price = mid_price.shift(-5).fillna(method="ffill")
            spread = 2 * (data["trade_price"] - future_mid_price).abs() / mid_price
            result = pd.Series(spread, index=data.index).fillna(0.0)
            result.name = "realized_spread"
            latency = time.time() - start_time
            self.log_performance("compute_realized_spread", latency, success=True)
            return result
        except Exception as e:
            self.log_performance(
                "compute_realized_spread", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_realized_spread: {str(e)}",
                tag="ORDERFLOW",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_realized_spread: {str(e)}", priority=4)
            logger.error(f"Erreur dans compute_realized_spread: {str(e)}")
            return pd.Series(0.0, index=data.index, name="realized_spread")

    def compute_trade_size_variance(self, data: pd.DataFrame) -> pd.Series:
        """
        Calcule la variance de la taille des trades (Phase 13).
        """
        try:
            start_time = time.time()
            if "trade_size" not in data.columns:
                error_msg = "Colonne trade_size manquante"
                miya_alerts(
                    error_msg, tag="ORDERFLOW", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=data.index, name="trade_size_variance")
            variance = data["trade_size"].rolling(window=10, min_periods=1).var()
            result = pd.Series(variance, index=data.index).fillna(0.0)
            result.name = "trade_size_variance"
            latency = time.time() - start_time
            self.log_performance("compute_trade_size_variance", latency, success=True)
            return result
        except Exception as e:
            self.log_performance(
                "compute_trade_size_variance", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_trade_size_variance: {str(e)}",
                tag="ORDERFLOW",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_trade_size_variance: {str(e)}", priority=4)
            logger.error(f"Erreur dans compute_trade_size_variance: {str(e)}")
            return pd.Series(0.0, index=data.index, name="trade_size_variance")

    def compute_aggressive_trade_ratio(self, data: pd.DataFrame) -> pd.Series:
        """
        Calcule le ratio de trades agressifs (Phase 13).
        """
        try:
            start_time = time.time()
            if (
                "trade_price" not in data.columns
                or "bid_price_level_1" not in data.columns
                or "ask_price_level_1" not in data.columns
            ):
                error_msg = "Colonnes trade_price, bid_price_level_1 ou ask_price_level_1 manquantes"
                miya_alerts(
                    error_msg, tag="ORDERFLOW", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=data.index, name="aggressive_trade_ratio")
            aggressive_trades = (
                (data["trade_price"] >= data["ask_price_level_1"])
                | (data["trade_price"] <= data["bid_price_level_1"])
            ).astype(int)
            ratio = aggressive_trades.rolling(window=10, min_periods=1).mean()
            result = pd.Series(ratio, index=data.index).fillna(0.0)
            result.name = "aggressive_trade_ratio"
            latency = time.time() - start_time
            self.log_performance(
                "compute_aggressive_trade_ratio", latency, success=True
            )
            return result
        except Exception as e:
            self.log_performance(
                "compute_aggressive_trade_ratio", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_aggressive_trade_ratio: {str(e)}",
                tag="ORDERFLOW",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_aggressive_trade_ratio: {str(e)}", priority=4
            )
            logger.error(f"Erreur dans compute_aggressive_trade_ratio: {str(e)}")
            return pd.Series(0.0, index=data.index, name="aggressive_trade_ratio")

    def compute_hidden_liquidity_ratio(self, data: pd.DataFrame) -> pd.Series:
        """
        Calcule le ratio de liquidité cachée (Phase 13).
        """
        try:
            start_time = time.time()
            if (
                "trade_size" not in data.columns
                or "bid_size_level_1" not in data.columns
                or "ask_size_level_1" not in data.columns
            ):
                error_msg = "Colonnes trade_size, bid_size_level_1 ou ask_size_level_1 manquantes"
                miya_alerts(
                    error_msg, tag="ORDERFLOW", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=data.index, name="hidden_liquidity_ratio")
            iceberg_trades = (
                data["trade_size"]
                > (data["bid_size_level_1"] + data["ask_size_level_1"])
            ).astype(int)
            ratio = iceberg_trades.rolling(window=10, min_periods=1).mean()
            result = pd.Series(ratio, index=data.index).fillna(0.0)
            result.name = "hidden_liquidity_ratio"
            latency = time.time() - start_time
            self.log_performance(
                "compute_hidden_liquidity_ratio", latency, success=True
            )
            return result
        except Exception as e:
            self.log_performance(
                "compute_hidden_liquidity_ratio", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_hidden_liquidity_ratio: {str(e)}",
                tag="ORDERFLOW",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_hidden_liquidity_ratio: {str(e)}", priority=4
            )
            logger.error(f"Erreur dans compute_hidden_liquidity_ratio: {str(e)}")
            return pd.Series(0.0, index=data.index, name="hidden_liquidity_ratio")

    def compute_order_imbalance_decay(self, data: pd.DataFrame) -> pd.Series:
        """
        Calcule la décroissance de l’imbalance des ordres (Phase 13).
        """
        try:
            start_time = time.time()
            if "delta_volume" not in data.columns:
                error_msg = "Colonne delta_volume manquante"
                miya_alerts(
                    error_msg, tag="ORDERFLOW", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=data.index, name="order_imbalance_decay")
            decay = (
                data["delta_volume"]
                .abs()
                .rolling(window=10, min_periods=1)
                .apply(lambda x: np.exp(-len(x) / 10))
            )
            result = pd.Series(decay, index=data.index).fillna(0.0)
            result.name = "order_imbalance_decay"
            latency = time.time() - start_time
            self.log_performance("compute_order_imbalance_decay", latency, success=True)
            return result
        except Exception as e:
            self.log_performance(
                "compute_order_imbalance_decay", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_order_imbalance_decay: {str(e)}",
                tag="ORDERFLOW",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_order_imbalance_decay: {str(e)}", priority=4
            )
            logger.error(f"Erreur dans compute_order_imbalance_decay: {str(e)}")
            return pd.Series(0.0, index=data.index, name="order_imbalance_decay")

    def compute_footprint_delta(self, data: pd.DataFrame) -> pd.Series:
        """
        Calcule le delta du footprint des trades (Phase 13).
        """
        try:
            start_time = time.time()
            if "trade_price" not in data.columns or "trade_size" not in data.columns:
                error_msg = "Colonnes trade_price ou trade_size manquantes"
                miya_alerts(
                    error_msg, tag="ORDERFLOW", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=data.index, name="footprint_delta")
            delta = data.groupby("trade_price")["trade_size"].sum().diff().fillna(0)
            result = pd.Series(delta, index=data.index).fillna(0.0)
            result.name = "footprint_delta"
            latency = time.time() - start_time
            self.log_performance("compute_footprint_delta", latency, success=True)
            return result
        except Exception as e:
            self.log_performance(
                "compute_footprint_delta", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_footprint_delta: {str(e)}",
                tag="ORDERFLOW",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_footprint_delta: {str(e)}", priority=4)
            logger.error(f"Erreur dans compute_footprint_delta: {str(e)}")
            return pd.Series(0.0, index=data.index, name="footprint_delta")

    def compute_order_book_skew(
        self, data: pd.DataFrame, levels: int = 10
    ) -> pd.Series:
        """
        Calcule le skew du carnet d’ordres sur plusieurs niveaux (Phase 13).
        """
        try:
            start_time = time.time()
            bid_cols = [
                f"bid_size_level_{i}"
                for i in range(1, levels + 1)
                if f"bid_size_level_{i}" in data.columns
            ]
            ask_cols = [
                f"ask_size_level_{i}"
                for i in range(1, levels + 1)
                if f"ask_size_level_{i}" in data.columns
            ]
            if not bid_cols or not ask_cols:
                error_msg = f"Colonnes bid/ask manquantes pour {levels} niveaux"
                miya_alerts(
                    error_msg, tag="ORDERFLOW", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=data.index, name="order_book_skew_10")
            bid_sum = data[bid_cols].sum(axis=1)
            ask_sum = data[ask_cols].sum(axis=1)
            skew = (bid_sum - ask_sum) / (bid_sum + ask_sum + 1e-6)
            result = pd.Series(skew, index=data.index).fillna(0.0)
            result.name = "order_book_skew_10"
            latency = time.time() - start_time
            self.log_performance("compute_order_book_skew", latency, success=True)
            return result
        except Exception as e:
            self.log_performance(
                "compute_order_book_skew", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_order_book_skew: {str(e)}",
                tag="ORDERFLOW",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_order_book_skew: {str(e)}", priority=4)
            logger.error(f"Erreur dans compute_order_book_skew: {str(e)}")
            return pd.Series(0.0, index=data.index, name="order_book_skew_10")

    def compute_trade_flow_acceleration(self, data: pd.DataFrame) -> pd.Series:
        """
        Calcule l’accélération du flux de trades (Phase 13).
        """
        try:
            start_time = time.time()
            if "trade_frequency_1s" not in data.columns:
                error_msg = "Colonne trade_frequency_1s manquante"
                miya_alerts(
                    error_msg, tag="ORDERFLOW", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=data.index, name="trade_flow_acceleration")
            velocity = data["trade_frequency_1s"].diff().fillna(0)
            acceleration = velocity.diff().fillna(0)
            result = pd.Series(acceleration, index=data.index).fillna(0.0)
            result.name = "trade_flow_acceleration"
            latency = time.time() - start_time
            self.log_performance(
                "compute_trade_flow_acceleration", latency, success=True
            )
            return result
        except Exception as e:
            self.log_performance(
                "compute_trade_flow_acceleration", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_trade_flow_acceleration: {str(e)}",
                tag="ORDERFLOW",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_trade_flow_acceleration: {str(e)}", priority=4
            )
            logger.error(f"Erreur dans compute_trade_flow_acceleration: {str(e)}")
            return pd.Series(0.0, index=data.index, name="trade_flow_acceleration")

    def compute_order_book_momentum(self, data: pd.DataFrame) -> pd.Series:
        """
        Calcule le momentum du carnet d’ordres (Phase 13).
        """
        try:
            start_time = time.time()
            bid_cols = [
                f"bid_size_level_{i}"
                for i in range(1, 6)
                if f"bid_size_level_{i}" in data.columns
            ]
            ask_cols = [
                f"ask_size_level_{i}"
                for i in range(1, 6)
                if f"ask_size_level_{i}" in data.columns
            ]
            if not bid_cols or not ask_cols:
                error_msg = "Colonnes bid/ask manquantes pour niveaux 1-5"
                miya_alerts(
                    error_msg, tag="ORDERFLOW", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=data.index, name="order_book_momentum")
            total_depth = data[bid_cols + ask_cols].sum(axis=1)
            momentum = total_depth.diff().fillna(0)
            result = pd.Series(momentum, index=data.index).fillna(0.0)
            result.name = "order_book_momentum"
            latency = time.time() - start_time
            self.log_performance("compute_order_book_momentum", latency, success=True)
            return result
        except Exception as e:
            self.log_performance(
                "compute_order_book_momentum", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_order_book_momentum: {str(e)}",
                tag="ORDERFLOW",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_order_book_momentum: {str(e)}", priority=4)
            logger.error(f"Erreur dans compute_order_book_momentum: {str(e)}")
            return pd.Series(0.0, index=data.index, name="order_book_momentum")

    def compute_trade_size_skew(
        self, data: pd.DataFrame, window: str = "1min"
    ) -> pd.Series:
        """
        Calcule le skew de la taille des trades (Phase 13).
        """
        try:
            start_time = time.time()
            if "trade_size" not in data.columns or "timestamp" not in data.columns:
                error_msg = "Colonnes trade_size ou timestamp manquantes"
                miya_alerts(
                    error_msg, tag="ORDERFLOW", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=data.index, name="trade_size_skew_1m")
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            grouped = data.groupby(pd.Grouper(key="timestamp", freq=window))
            skew = (
                grouped["trade_size"]
                .skew()
                .reindex(data.index, method="ffill")
                .fillna(0.0)
            )
            result = pd.Series(skew, index=data.index)
            result.name = "trade_size_skew_1m"
            latency = time.time() - start_time
            self.log_performance("compute_trade_size_skew", latency, success=True)
            return result
        except Exception as e:
            self.log_performance(
                "compute_trade_size_skew", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_trade_size_skew: {str(e)}",
                tag="ORDERFLOW",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_trade_size_skew: {str(e)}", priority=4)
            logger.error(f"Erreur dans compute_trade_size_skew: {str(e)}")
            return pd.Series(0.0, index=data.index, name="trade_size_skew_1m")

    def normalize_indicators(
        self, data: pd.DataFrame, indicators: List[str]
    ) -> pd.DataFrame:
        """
        Normalise les indicateurs avec MinMaxScaler sur une fenêtre glissante.
        """
        try:
            start_time = time.time()
            for col in indicators:
                if col in data.columns:
                    # Appliquer MinMaxScaler sur une fenêtre glissante
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
                        tag="ORDERFLOW",
                        voice_profile="warning",
                        priority=3,
                    )
                    send_alert(
                        f"Colonne {col} manquante pour normalisation", priority=3
                    )
                    logger.warning(f"Colonne {col} manquante pour normalisation")
            latency = time.time() - start_time
            self.log_performance("normalize_indicators", latency, success=True)
            return data
        except Exception as e:
            self.log_performance("normalize_indicators", 0, success=False, error=str(e))
            miya_alerts(
                f"Erreur dans normalize_indicators: {str(e)}",
                tag="ORDERFLOW",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans normalize_indicators: {str(e)}", priority=4)
            logger.error(f"Erreur dans normalize_indicators: {str(e)}")
            return data

    def calculate_orderflow_indicators(
        self, data: pd.DataFrame, time_col: str = "timestamp", regime: str = "range"
    ) -> pd.DataFrame:
        """
        Calcule les indicateurs d’order flow à partir des données IQFeed avec pondération et normalisation.

        Args:
            data (pd.DataFrame): Données IQFeed avec timestamp, bid/ask sizes, prices, trade data.
            time_col (str): Nom de la colonne timestamp (par défaut : 'timestamp').
            regime (str): Régime de marché ('trend', 'range', 'defensive').

        Returns:
            pd.DataFrame: Données enrichies avec 16 indicateurs et leurs versions pondérées/normalisées.
        """
        try:
            start_time = time.time()
            if data.empty:
                error_msg = "DataFrame vide dans calculate_orderflow_indicators"
                miya_alerts(
                    error_msg, tag="ORDERFLOW", voice_profile="urgent", priority=5
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                raise ValueError(error_msg)

            if not self.validate_iqfeed_data(data):
                error_msg = "Échec validation données IQFeed"
                miya_alerts(
                    error_msg, tag="ORDERFLOW", voice_profile="urgent", priority=5
                )
                send_alert(error_msg, priority=4)
                raise ValueError(error_msg)

            valid_regimes = ["trend", "range", "defensive"]
            if regime not in valid_regimes:
                error_msg = f"Régime invalide: {regime}, attendu: {valid_regimes}"
                miya_alerts(
                    error_msg, tag="ORDERFLOW", voice_profile="urgent", priority=5
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
                ).total_seconds() < self.config.get("cache_hours", 24) * 3600:
                    miya_speak(
                        "Indicateurs récupérés du cache",
                        tag="ORDERFLOW",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert("Indicateurs récupérés du cache", priority=1)
                    self.log_performance(
                        "calculate_orderflow_indicators_cache_hit", 0, success=True
                    )
                    return cached_data

            data = data.copy()
            if time_col not in data.columns:
                miya_speak(
                    f"Colonne '{time_col}' manquante, création par défaut",
                    tag="ORDERFLOW",
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
                    tag="ORDERFLOW",
                    voice_profile="calm",
                    priority=1,
                )
                send_alert("Timestamp par défaut ajouté", priority=1)

            data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
            if data[time_col].isna().any():
                miya_speak(
                    f"NaN dans '{time_col}', imputés avec la première date valide",
                    tag="ORDERFLOW",
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

            required_cols = [
                "bid_size_level_1",
                "ask_size_level_1",
                "bid_price_level_1",
                "ask_price_level_1",
                "bid_size_level_2",
                "ask_size_level_2",
                "bid_price_level_2",
                "ask_price_level_2",
                "trade_frequency_1s",
                "trade_price",
                "trade_size",
            ]
            missing_cols = [col for col in required_cols if col not in data.columns]
            for col in missing_cols:
                data[col] = 0.0
                miya_speak(
                    f"Colonne IQFeed '{col}' manquante, imputée à 0",
                    tag="ORDERFLOW",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(f"Colonne IQFeed '{col}' manquante", priority=3)
                logger.warning(f"Colonne manquante: {col}")

            data["delta_volume"] = (
                data["ask_size_level_1"] + data["ask_size_level_2"]
            ) - (data["bid_size_level_1"] + data["bid_size_level_2"])
            data["delta_volume"] = data["delta_volume"].fillna(0.0)
            total_volume = (
                data["ask_size_level_1"]
                + data["ask_size_level_2"]
                + data["bid_size_level_1"]
                + data["bid_size_level_2"]
                + 1e-6
            )
            data["obi_score"] = data["delta_volume"] / total_volume
            data["obi_score"] = data["obi_score"].fillna(0.0)
            data["vds"] = (
                data["delta_volume"].rolling(window=5, min_periods=1).mean().fillna(0.0)
            )
            total_depth = (
                data["bid_size_level_1"]
                + data["bid_size_level_2"]
                + data["ask_size_level_1"]
                + data["ask_size_level_2"]
            )
            data["absorption_strength"] = (
                (
                    data["delta_volume"].abs()
                    * data["trade_frequency_1s"]
                    / (total_depth + 1e-6)
                )
                .rolling(window=10, min_periods=1)
                .mean()
                .fillna(0.0)
            )

            data["bid_ask_imbalance"] = self.compute_bid_ask_imbalance(data)
            data["effective_spread"] = self.compute_effective_spread(data)
            data["realized_spread"] = self.compute_realized_spread(data)
            data["trade_size_variance"] = self.compute_trade_size_variance(data)
            data["aggressive_trade_ratio"] = self.compute_aggressive_trade_ratio(data)
            data["hidden_liquidity_ratio"] = self.compute_hidden_liquidity_ratio(data)
            data["order_imbalance_decay"] = self.compute_order_imbalance_decay(data)
            data["footprint_delta"] = self.compute_footprint_delta(data)
            data["order_book_skew_10"] = self.compute_order_book_skew(data, levels=10)
            data["trade_flow_acceleration"] = self.compute_trade_flow_acceleration(data)
            data["order_book_momentum"] = self.compute_order_book_momentum(data)
            data["trade_size_skew_1m"] = self.compute_trade_size_skew(
                data, window="1min"
            )

            indicators = [
                "delta_volume",
                "obi_score",
                "vds",
                "absorption_strength",
                "bid_ask_imbalance",
                "effective_spread",
                "realized_spread",
                "trade_size_variance",
                "aggressive_trade_ratio",
                "hidden_liquidity_ratio",
                "order_imbalance_decay",
                "footprint_delta",
                "order_book_skew_10",
                "trade_flow_acceleration",
                "order_book_momentum",
                "trade_size_skew_1m",
            ]
            self.validate_shap_features(indicators)
            weight = self.config.get("weights", {}).get(regime, 1.0)
            for col in indicators:
                data[f"{col}_weighted"] = data[col] * weight
                if data[f"{col}_weighted"].isna().any():
                    data[f"{col}_weighted"] = data[f"{col}_weighted"].fillna(0.0)
                    miya_speak(
                        f"NaN dans {col}_weighted, imputés à 0",
                        tag="ORDERFLOW",
                        voice_profile="warning",
                        priority=3,
                    )
                    send_alert(f"NaN dans {col}_weighted", priority=3)
                    logger.warning(f"NaN dans {col}_weighted, imputés à 0")

            # Normalisation pour l’inférence
            data = self.normalize_indicators(data, indicators)

            self.cache_indicators(data, cache_key)
            self.plot_indicators(data, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            latency = time.time() - start_time
            miya_speak(
                "Indicateurs d'order flow calculés",
                tag="ORDERFLOW",
                voice_profile="calm",
                priority=1,
            )
            send_alert("Indicateurs d'order flow calculés", priority=1)
            logger.info("Indicateurs d'order flow calculés")
            self.log_performance(
                "calculate_orderflow_indicators",
                latency,
                success=True,
                num_rows=len(data),
                regime=regime,
                num_indicators=len(indicators),
            )
            self.save_snapshot(
                "calculate_orderflow_indicators",
                {"num_rows": len(data), "indicators": indicators, "regime": regime},
            )
            return data
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans calculate_orderflow_indicators: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="ORDERFLOW", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=4)
            logger.error(error_msg)
            self.log_performance(
                "calculate_orderflow_indicators", latency, success=False, error=str(e)
            )
            self.save_snapshot("calculate_orderflow_indicators", {"error": str(e)})
            data["timestamp"] = pd.date_range(
                start=pd.Timestamp.now(), periods=len(data), freq="1min"
            )
            for col in [
                "delta_volume",
                "obi_score",
                "vds",
                "absorption_strength",
                "bid_ask_imbalance",
                "effective_spread",
                "realized_spread",
                "trade_size_variance",
                "aggressive_trade_ratio",
                "hidden_liquidity_ratio",
                "order_imbalance_decay",
                "footprint_delta",
                "order_book_skew_10",
                "trade_flow_acceleration",
                "order_book_momentum",
                "trade_size_skew_1m",
                "delta_volume_weighted",
                "obi_score_weighted",
                "vds_weighted",
                "absorption_strength_weighted",
                "bid_ask_imbalance_weighted",
                "effective_spread_weighted",
                "realized_spread_weighted",
                "trade_size_variance_weighted",
                "aggressive_trade_ratio_weighted",
                "hidden_liquidity_ratio_weighted",
                "order_imbalance_decay_weighted",
                "footprint_delta_weighted",
                "order_book_skew_10_weighted",
                "trade_flow_acceleration_weighted",
                "order_book_momentum_weighted",
                "trade_size_skew_1m_weighted",
                "delta_volume_normalized",
                "obi_score_normalized",
                "vds_normalized",
                "absorption_strength_normalized",
                "bid_ask_imbalance_normalized",
                "effective_spread_normalized",
                "realized_spread_normalized",
                "trade_size_variance_normalized",
                "aggressive_trade_ratio_normalized",
                "hidden_liquidity_ratio_normalized",
                "order_imbalance_decay_normalized",
                "footprint_delta_normalized",
                "order_book_skew_10_normalized",
                "trade_flow_acceleration_normalized",
                "order_book_momentum_normalized",
                "trade_size_skew_1m_normalized",
            ]:
                data[col] = 0.0
            return data


if __name__ == "__main__":
    try:
        encoder = OrderFlowIndicators()
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "bid_size_level_1": np.random.randint(50, 500, 100),
                "ask_size_level_1": np.random.randint(50, 500, 100),
                "bid_price_level_1": np.random.normal(5100, 5, 100),
                "ask_price_level_1": np.random.normal(5102, 5, 100),
                "bid_size_level_2": np.random.randint(30, 300, 100),
                "ask_size_level_2": np.random.randint(30, 300, 100),
                "bid_price_level_2": np.random.normal(5098, 5, 100),
                "ask_price_level_2": np.random.normal(5104, 5, 100),
                "bid_size_level_3": np.random.randint(20, 200, 100),
                "ask_size_level_3": np.random.randint(20, 200, 100),
                "bid_size_level_4": np.random.randint(10, 100, 100),
                "ask_size_level_4": np.random.randint(10, 100, 100),
                "bid_size_level_5": np.random.randint(5, 50, 100),
                "ask_size_level_5": np.random.randint(5, 50, 100),
                "trade_frequency_1s": np.random.randint(0, 50, 100),
                "trade_price": np.random.normal(5101, 5, 100),
                "trade_size": np.random.randint(1, 100, 100),
                "close": np.random.normal(5100, 10, 100),
            }
        )
        result = encoder.calculate_orderflow_indicators(data, regime="trend")
        print(
            result[
                [
                    "timestamp",
                    "delta_volume",
                    "obi_score",
                    "bid_ask_imbalance",
                    "effective_spread",
                ]
            ].head()
        )
        miya_speak(
            "Test calculate_orderflow_indicators terminé",
            tag="ORDERFLOW",
            voice_profile="calm",
            priority=1,
        )
        send_alert("Test calculate_orderflow_indicators terminé", priority=1)
    except Exception as e:
        miya_alerts(
            f"Erreur test: {str(e)}\n{traceback.format_exc()}",
            tag="ORDERFLOW",
            voice_profile="urgent",
            priority=3,
        )
        send_alert(f"Erreur test: {str(e)}", priority=4)
        logger.error(f"Erreur test: {str(e)}")
        raise
