# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/features/extractors/cross_asset_processor.py
# Calcule les corrélations et métriques cross-asset (or, pétrole, BTC, obligations).
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Calcule 9 métriques cross-asset (6 existantes comme gold_correl, yield_curve_slope + 3 nouvelles pour Phases 13, 15, 18),
#        utilise IQFeed pour les prix, intègre SHAP (méthode 17), et enregistre logs psutil.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.0,<6.0.0, matplotlib>=3.7.0,<4.0.0, json, gzip, hashlib, logging
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
#
# Inputs :
# - config/es_config.yaml
# - data/iqfeed/cross_asset_data.csv
# - data/iqfeed/bond_data.csv
# - data/iqfeed/spy_data.csv
# - data/iqfeed/options_data.csv
# - data/iqfeed/microstructure_data.csv
# - data/iqfeed/hft_data.csv
# - data/features/feature_importance.csv
#
# Outputs :
# - data/features/cache/cross_asset/
# - data/logs/cross_asset_performance.csv
# - data/features/cross_asset_snapshots/
# - data/figures/cross_asset/
#
# Notes :
# - Utilise exclusivement IQFeed pour les données (remplace dxFeed).
# - Suppression de toute référence à dxFeed, obs_t, 320 features, et 81 features pour conformité avec MIA_IA_SYSTEM_v2_2025.
# - Intègre les Phases 1-18 :
#   - Phase 1 (news_scraper.py, news_analyzer.py) : Contexte pour l'analyse cross-asset.
#   - Phase 13 (orderflow_indicators.py, options_metrics.py) : Nouvelle métrique option_cross_correl.
#   - Phase 15 (microstructure_guard.py, spotgamma_recalculator.py) : Nouvelle métrique microstructure_cross_impact.
#   - Phase 18 : Nouvelle métrique hft_cross_correl pour corrélation avec l'activité HFT.
# - Tests unitaires disponibles dans tests/test_cross_asset_processor.py.
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
CACHE_DIR = os.path.join(BASE_DIR, "data", "features", "cache", "cross_asset")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "features", "cross_asset_snapshots")
CSV_LOG_PATH = os.path.join(BASE_DIR, "data", "logs", "cross_asset_performance.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures", "cross_asset")
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
    filename=os.path.join(BASE_DIR, "data", "logs", "cross_asset.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 2.0
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes


class CrossAssetProcessor:

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
                "CrossAssetProcessor initialisé",
                tag="CROSS_ASSET",
                voice_profile="calm",
                priority=2,
            )
            send_alert("CrossAssetProcessor initialisé", priority=1)
            logger.info("CrossAssetProcessor initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config_path": config_path})
        except Exception as e:
            error_msg = f"Erreur initialisation CrossAssetProcessor: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="CROSS_ASSET", voice_profile="urgent", priority=3
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

    def load_config_with_manager_new(self, config_path: str) -> Dict[str, Any]:
        def load_yaml():
            config = config_manager.get_config(os.path.basename(config_path))
            if "cross_asset_processor" not in config:
                raise ValueError(
                    "Clé 'cross_asset_processor' manquante dans la configuration"
                )
            required_keys = ["buffer_size", "max_cache_size", "cache_hours"]
            missing_keys = [
                key
                for key in required_keys
                if key not in config["cross_asset_processor"]
            ]
            if missing_keys:
                raise ValueError(
                    f"Clés manquantes dans 'cross_asset_processor': {missing_keys}"
                )
            return config["cross_asset_processor"]

        try:
            start_time = time.time()
            config = self.with_retries(load_yaml)
            latency = time.time() - start_time
            miya_speak(
                "Configuration cross_asset_processor chargée via config_manager",
                tag="CROSS_ASSET",
                voice_profile="calm",
                priority=2,
            )
            send_alert(
                "Configuration cross_asset_processor chargée via config_manager",
                priority=1,
            )
            logger.info(
                "Configuration cross_asset_processor chargée via config_manager"
            )
            self.log_performance("load_config_with_manager_new", latency, success=True)
            self.save_snapshot(
                "load_config_with_manager_new", {"config_path": config_path}
            )
            return config
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur chargement config via config_manager: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="CROSS_ASSET", voice_profile="urgent", priority=3
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
                    tag="CROSS_ASSET",
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
                tag="CROSS_ASSET",
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
                tag="CROSS_ASSET",
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
                tag="CROSS_ASSET",
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
                tag="CROSS_ASSET",
                voice_profile="calm",
                priority=2,
            )
            send_alert("SIGINT received, snapshot saved", priority=2)
            logger.info("SIGINT received, snapshot saved")
        except Exception as e:
            miya_alerts(
                f"Erreur gestion SIGINT: {str(e)}",
                tag="CROSS_ASSET",
                voice_profile="urgent",
                priority=5,
            )
            send_alert(f"Erreur gestion SIGINT: {str(e)}", priority=4)
            logger.error(f"Erreur gestion SIGINT: {str(e)}")
        finally:
            raise SystemExit("Terminated by SIGINT")

    def validate_shap_features(self, features: List[str]) -> bool:
        try:
            if not os.path.exists(FEATURE_IMPORTANCE_PATH):
                miya_alerts(
                    "Fichier SHAP manquant",
                    tag="CROSS_ASSET",
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
                    tag="CROSS_ASSET",
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
                    tag="CROSS_ASSET",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(
                    f"Features non incluses dans top 150 SHAP: {missing}", priority=3
                )
                logger.warning(f"Features non incluses dans top 150 SHAP: {missing}")
            miya_speak(
                "SHAP features validées",
                tag="CROSS_ASSET",
                voice_profile="calm",
                priority=1,
            )
            send_alert("SHAP features validées", priority=1)
            logger.info("SHAP features validées")
            return True
        except Exception as e:
            miya_alerts(
                f"Erreur validation SHAP features: {str(e)}",
                tag="CROSS_ASSET",
                voice_profile="urgent",
                priority=4,
            )
            send_alert(f"Erreur validation SHAP features: {str(e)}", priority=4)
            logger.error(f"Erreur validation SHAP features: {str(e)}")
            return False

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
                tag="CROSS_ASSET",
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
                tag="CROSS_ASSET",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur mise en cache métriques: {str(e)}", priority=4)
            logger.error(f"Erreur mise en cache métriques: {str(e)}")

    def compute_correlation(
        self, asset_data: pd.DataFrame, cross_asset: str, window: int = 20
    ) -> pd.Series:
        """
        Calcule la corrélation entre le prix principal et un actif cross-asset (or, pétrole, BTC, options, HFT).

        Args:
            asset_data (pd.DataFrame): Données contenant close et cross_asset (gold_price, oil_price, btc_price, option_skew, hft_activity_score).
            cross_asset (str): Nom de la colonne de l'actif (ex. : gold_price).
            window (int): Fenêtre de calcul de la corrélation (par défaut : 20).

        Returns:
            pd.Series: Série de corrélations.
        """
        try:
            start_time = time.time()
            if (
                "close" not in asset_data.columns
                or cross_asset not in asset_data.columns
            ):
                error_msg = (
                    f"Colonnes close ou {cross_asset} manquantes dans asset_data"
                )
                miya_alerts(
                    error_msg, tag="CROSS_ASSET", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=asset_data.index)
            asset_data["close"] = pd.to_numeric(asset_data["close"], errors="coerce")
            asset_data[cross_asset] = pd.to_numeric(
                asset_data[cross_asset], errors="coerce"
            )
            correlation = (
                asset_data[["close", cross_asset]]
                .rolling(window=window, min_periods=1)
                .corr()
                .unstack()
                .iloc[:, 1]
            )
            result = correlation.fillna(0.0)
            latency = time.time() - start_time
            self.log_performance(
                f"compute_correlation_{cross_asset}", latency, success=True
            )
            return result
        except Exception as e:
            self.log_performance(
                f"compute_correlation_{cross_asset}", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_correlation_{cross_asset}: {str(e)}",
                tag="CROSS_ASSET",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_correlation_{cross_asset}: {str(e)}", priority=4
            )
            logger.error(f"Erreur dans compute_correlation_{cross_asset}: {str(e)}")
            return pd.Series(0.0, index=asset_data.index)

    def compute_yield_curve_slope(self, bond_data: pd.DataFrame) -> pd.Series:
        """
        Calcule la pente de la courbe des rendements (10y - 2y).

        Args:
            bond_data (pd.DataFrame): Données contenant yield_10y et yield_2y.

        Returns:
            pd.Series: Série de pentes de la courbe des rendements.
        """
        try:
            start_time = time.time()
            if (
                "yield_10y" not in bond_data.columns
                or "yield_2y" not in bond_data.columns
            ):
                error_msg = "Colonnes yield_10y ou yield_2y manquantes dans bond_data"
                miya_alerts(
                    error_msg, tag="CROSS_ASSET", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=bond_data.index)
            bond_data["yield_10y"] = pd.to_numeric(
                bond_data["yield_10y"], errors="coerce"
            )
            bond_data["yield_2y"] = pd.to_numeric(
                bond_data["yield_2y"], errors="coerce"
            )
            slope = bond_data["yield_10y"] - bond_data["yield_2y"]
            result = slope.fillna(0.0)
            latency = time.time() - start_time
            self.log_performance("compute_yield_curve_slope", latency, success=True)
            return result
        except Exception as e:
            self.log_performance(
                "compute_yield_curve_slope", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_yield_curve_slope: {str(e)}",
                tag="CROSS_ASSET",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_yield_curve_slope: {str(e)}", priority=4)
            logger.error(f"Erreur dans compute_yield_curve_slope: {str(e)}")
            return pd.Series(0.0, index=bond_data.index)

    def compute_flow_ratio(
        self, es_data: pd.DataFrame, spy_data: pd.DataFrame
    ) -> pd.Series:
        """
        Calcule le ratio de flux entre ES et SPY (volume ES / volume SPY).

        Args:
            es_data (pd.DataFrame): Données ES avec volume.
            spy_data (pd.DataFrame): Données SPY avec volume.

        Returns:
            pd.Series: Série de ratios de flux.
        """
        try:
            start_time = time.time()
            if "volume" not in es_data.columns or "volume" not in spy_data.columns:
                error_msg = "Colonne volume manquante dans es_data ou spy_data"
                miya_alerts(
                    error_msg, tag="CROSS_ASSET", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=es_data.index)
            es_data["volume"] = pd.to_numeric(es_data["volume"], errors="coerce")
            spy_data["volume"] = pd.to_numeric(spy_data["volume"], errors="coerce")
            ratio = es_data["volume"] / spy_data["volume"].replace(0, 1e-6)
            result = ratio.fillna(0.0)
            latency = time.time() - start_time
            self.log_performance("compute_flow_ratio", latency, success=True)
            return result
        except Exception as e:
            self.log_performance("compute_flow_ratio", 0, success=False, error=str(e))
            miya_alerts(
                f"Erreur dans compute_flow_ratio: {str(e)}",
                tag="CROSS_ASSET",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_flow_ratio: {str(e)}", priority=4)
            logger.error(f"Erreur dans compute_flow_ratio: {str(e)}")
            return pd.Series(0.0, index=es_data.index)

    def compute_option_cross_correl(
        self, options_data: pd.DataFrame, window: int = 20
    ) -> pd.Series:
        """
        Calcule la corrélation entre le prix principal et le skew des options (Phase 13).

        Args:
            options_data (pd.DataFrame): Données contenant close et option_skew.
            window (int): Fenêtre de calcul de la corrélation (par défaut : 20).

        Returns:
            pd.Series: Série de corrélations.
        """
        try:
            start_time = time.time()
            if (
                "close" not in options_data.columns
                or "option_skew" not in options_data.columns
            ):
                error_msg = "Colonnes close ou option_skew manquantes dans options_data"
                miya_alerts(
                    error_msg, tag="CROSS_ASSET", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=options_data.index)
            options_data["close"] = pd.to_numeric(
                options_data["close"], errors="coerce"
            )
            options_data["option_skew"] = pd.to_numeric(
                options_data["option_skew"], errors="coerce"
            )
            correlation = (
                options_data[["close", "option_skew"]]
                .rolling(window=window, min_periods=1)
                .corr()
                .unstack()
                .iloc[:, 1]
            )
            result = correlation.fillna(0.0)
            latency = time.time() - start_time
            self.log_performance("compute_option_cross_correl", latency, success=True)
            return result
        except Exception as e:
            self.log_performance(
                "compute_option_cross_correl", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_option_cross_correl: {str(e)}",
                tag="CROSS_ASSET",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_option_cross_correl: {str(e)}", priority=4)
            logger.error(f"Erreur dans compute_option_cross_correl: {str(e)}")
            return pd.Series(0.0, index=options_data.index)

    def compute_microstructure_cross_impact(
        self, microstructure_data: pd.DataFrame
    ) -> pd.Series:
        """
        Calcule l'impact des événements de microstructure sur les actifs cross-asset (Phase 15).

        Args:
            microstructure_data (pd.DataFrame): Données contenant spoofing_score.

        Returns:
            pd.Series: Série d'impacts de microstructure.
        """
        try:
            start_time = time.time()
            if "spoofing_score" not in microstructure_data.columns:
                error_msg = "Colonne spoofing_score manquante dans microstructure_data"
                miya_alerts(
                    error_msg, tag="CROSS_ASSET", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=microstructure_data.index)
            microstructure_data["spoofing_score"] = pd.to_numeric(
                microstructure_data["spoofing_score"], errors="coerce"
            )
            result = microstructure_data["spoofing_score"].fillna(0.0)
            latency = time.time() - start_time
            self.log_performance(
                "compute_microstructure_cross_impact", latency, success=True
            )
            return result
        except Exception as e:
            self.log_performance(
                "compute_microstructure_cross_impact", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_microstructure_cross_impact: {str(e)}",
                tag="CROSS_ASSET",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_microstructure_cross_impact: {str(e)}", priority=4
            )
            logger.error(f"Erreur dans compute_microstructure_cross_impact: {str(e)}")
            return pd.Series(0.0, index=microstructure_data.index)

    def compute_hft_cross_correl(
        self, hft_data: pd.DataFrame, window: int = 20
    ) -> pd.Series:
        """
        Calcule la corrélation entre le prix principal et l'activité HFT (Phase 18).

        Args:
            hft_data (pd.DataFrame): Données contenant close et hft_activity_score.
            window (int): Fenêtre de calcul de la corrélation (par défaut : 20).

        Returns:
            pd.Series: Série de corrélations.
        """
        try:
            start_time = time.time()
            if (
                "close" not in hft_data.columns
                or "hft_activity_score" not in hft_data.columns
            ):
                error_msg = (
                    "Colonnes close ou hft_activity_score manquantes dans hft_data"
                )
                miya_alerts(
                    error_msg, tag="CROSS_ASSET", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=hft_data.index)
            hft_data["close"] = pd.to_numeric(hft_data["close"], errors="coerce")
            hft_data["hft_activity_score"] = pd.to_numeric(
                hft_data["hft_activity_score"], errors="coerce"
            )
            correlation = (
                hft_data[["close", "hft_activity_score"]]
                .rolling(window=window, min_periods=1)
                .corr()
                .unstack()
                .iloc[:, 1]
            )
            result = correlation.fillna(0.0)
            latency = time.time() - start_time
            self.log_performance("compute_hft_cross_correl", latency, success=True)
            return result
        except Exception as e:
            self.log_performance(
                "compute_hft_cross_correl", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_hft_cross_correl: {str(e)}",
                tag="CROSS_ASSET",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_hft_cross_correl: {str(e)}", priority=4)
            logger.error(f"Erreur dans compute_hft_cross_correl: {str(e)}")
            return pd.Series(0.0, index=hft_data.index)

    def compute_cross_asset_features(
        self,
        es_data: pd.DataFrame,
        spy_data: pd.DataFrame,
        asset_data: pd.DataFrame,
        bond_data: pd.DataFrame,
        options_data: pd.DataFrame,
        microstructure_data: pd.DataFrame,
        hft_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calcule les 9 features cross-asset à partir des données IQFeed.

        Args:
            es_data (pd.DataFrame): Données ES avec timestamp, close, volume.
            spy_data (pd.DataFrame): Données SPY avec volume.
            asset_data (pd.DataFrame): Données des actifs (or, pétrole, BTC) avec close, gold_price, oil_price, btc_price.
            bond_data (pd.DataFrame): Données des obligations avec yield_10y, yield_2y.
            options_data (pd.DataFrame): Données des options avec close, option_skew.
            microstructure_data (pd.DataFrame): Données de microstructure avec spoofing_score.
            hft_data (pd.DataFrame): Données HFT avec close, hft_activity_score.

        Returns:
            pd.DataFrame: Données ES enrichies avec les 9 features cross-asset.
        """
        try:
            start_time = time.time()
            config = self.load_config_with_manager_new()

            if es_data.empty:
                error_msg = "DataFrame es_data vide"
                miya_alerts(
                    error_msg, tag="CROSS_ASSET", voice_profile="urgent", priority=5
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                raise ValueError(error_msg)

            cache_key = hashlib.sha256(es_data.to_json().encode()).hexdigest()
            if cache_key in self.cache:
                cached_data = pd.read_csv(
                    self.cache[cache_key]["path"], encoding="utf-8"
                )
                if (
                    datetime.now() - self.cache[cache_key]["timestamp"]
                ).total_seconds() < config.get("cache_hours", 24) * 3600:
                    miya_speak(
                        "Features cross-asset récupérées du cache",
                        tag="CROSS_ASSET",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert("Features cross-asset récupérées du cache", priority=1)
                    self.log_performance(
                        "compute_cross_asset_features_cache_hit", 0, success=True
                    )
                    return cached_data

            es_data = es_data.copy()
            if "timestamp" not in es_data.columns:
                miya_speak(
                    "Colonne timestamp manquante, création par défaut",
                    tag="CROSS_ASSET",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert("Colonne timestamp manquante", priority=3)
                logger.warning("Colonne timestamp manquante, création par défaut")
                default_start = pd.Timestamp.now()
                es_data["timestamp"] = pd.date_range(
                    start=default_start, periods=len(es_data), freq="1min"
                )

            es_data["timestamp"] = pd.to_datetime(es_data["timestamp"], errors="coerce")
            if es_data["timestamp"].isna().any():
                miya_speak(
                    "NaN dans timestamp, imputés avec la première date valide",
                    tag="CROSS_ASSET",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert("NaN dans timestamp", priority=3)
                logger.warning("NaN dans timestamp, imputation")
                first_valid_time = (
                    es_data["timestamp"].dropna().iloc[0]
                    if not es_data["timestamp"].dropna().empty
                    else pd.Timestamp.now()
                )
                es_data["timestamp"] = es_data["timestamp"].fillna(first_valid_time)

            es_data["gold_correl"] = self.compute_correlation(asset_data, "gold_price")
            es_data["oil_correl"] = self.compute_correlation(asset_data, "oil_price")
            es_data["crypto_btc_correl"] = self.compute_correlation(
                asset_data, "btc_price"
            )
            es_data["treasury_10y_yield"] = bond_data.get(
                "yield_10y", pd.Series(0.0, index=bond_data.index)
            ).fillna(0.0)
            es_data["yield_curve_slope"] = self.compute_yield_curve_slope(bond_data)
            es_data["cross_asset_flow_ratio"] = self.compute_flow_ratio(
                es_data, spy_data
            )
            es_data["option_cross_correl"] = self.compute_option_cross_correl(
                options_data
            )
            es_data["microstructure_cross_impact"] = (
                self.compute_microstructure_cross_impact(microstructure_data)
            )
            es_data["hft_cross_correl"] = self.compute_hft_cross_correl(hft_data)

            metrics = [
                "gold_correl",
                "oil_correl",
                "crypto_btc_correl",
                "treasury_10y_yield",
                "yield_curve_slope",
                "cross_asset_flow_ratio",
                "option_cross_correl",
                "microstructure_cross_impact",
                "hft_cross_correl",
            ]
            self.validate_shap_features(metrics)
            self.cache_metrics(es_data, cache_key)
            self.plot_metrics(es_data, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            latency = time.time() - start_time
            miya_speak(
                "Features cross-asset calculées",
                tag="CROSS_ASSET",
                voice_profile="calm",
                priority=1,
            )
            send_alert("Features cross-asset calculées", priority=1)
            logger.info("Features cross-asset calculées")
            self.log_performance(
                "compute_cross_asset_features",
                latency,
                success=True,
                num_rows=len(es_data),
                num_metrics=len(metrics),
            )
            self.save_snapshot(
                "compute_cross_asset_features",
                {"num_rows": len(es_data), "metrics": metrics},
            )
            return es_data
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans compute_cross_asset_features: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="CROSS_ASSET", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=4)
            logger.error(error_msg)
            self.log_performance(
                "compute_cross_asset_features", latency, success=False, error=str(e)
            )
            self.save_snapshot("compute_cross_asset_features", {"error": str(e)})
            es_data["timestamp"] = pd.date_range(
                start=pd.Timestamp.now(), periods=len(es_data), freq="1min"
            )
            for col in [
                "gold_correl",
                "oil_correl",
                "crypto_btc_correl",
                "treasury_10y_yield",
                "yield_curve_slope",
                "cross_asset_flow_ratio",
                "option_cross_correl",
                "microstructure_cross_impact",
                "hft_cross_correl",
            ]:
                es_data[col] = 0.0
            return es_data

    def plot_metrics(self, metrics: pd.DataFrame, timestamp: str) -> None:
        """
        Génère des visualisations des métriques cross-asset.

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
                metrics["gold_correl"],
                label="Gold Correlation",
                color="gold",
            )
            plt.plot(
                metrics["timestamp"],
                metrics["oil_correl"],
                label="Oil Correlation",
                color="black",
            )
            plt.plot(
                metrics["timestamp"],
                metrics["crypto_btc_correl"],
                label="BTC Correlation",
                color="orange",
            )
            plt.plot(
                metrics["timestamp"],
                metrics["yield_curve_slope"],
                label="Yield Curve Slope",
                color="blue",
            )
            plt.plot(
                metrics["timestamp"],
                metrics["option_cross_correl"],
                label="Option Skew Correlation",
                color="purple",
            )
            plt.plot(
                metrics["timestamp"],
                metrics["microstructure_cross_impact"],
                label="Microstructure Impact",
                color="red",
            )
            plt.plot(
                metrics["timestamp"],
                metrics["hft_cross_correl"],
                label="HFT Correlation",
                color="cyan",
            )
            plt.title(f"Cross-Asset Metrics - {timestamp}")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.savefig(
                os.path.join(FIGURES_DIR, f"cross_asset_metrics_{timestamp_safe}.png")
            )
            plt.close()
            latency = time.time() - start_time
            miya_speak(
                f"Visualisations générées: {FIGURES_DIR}",
                tag="CROSS_ASSET",
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
                tag="CROSS_ASSET",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur génération visualisations: {str(e)}", priority=4)
            logger.error(f"Erreur génération visualisations: {str(e)}")


if __name__ == "__main__":
    try:
        processor = CrossAssetProcessor()
        es_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "close": np.random.normal(5100, 10, 100),
                "volume": np.random.randint(100, 1000, 100),
            }
        )
        spy_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "volume": np.random.randint(100, 1000, 100),
            }
        )
        asset_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "close": np.random.normal(5100, 10, 100),
                "gold_price": np.random.normal(2000, 5, 100),
                "oil_price": np.random.normal(80, 2, 100),
                "btc_price": np.random.normal(60000, 1000, 100),
            }
        )
        bond_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "yield_10y": np.random.normal(3.5, 0.1, 100),
                "yield_2y": np.random.normal(2.5, 0.1, 100),
            }
        )
        options_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "close": np.random.normal(5100, 10, 100),
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
                "close": np.random.normal(5100, 10, 100),
                "hft_activity_score": np.random.uniform(0, 1, 100),
            }
        )
        result = processor.compute_cross_asset_features(
            es_data,
            spy_data,
            asset_data,
            bond_data,
            options_data,
            microstructure_data,
            hft_data,
        )
        print(
            result[
                [
                    "timestamp",
                    "gold_correl",
                    "yield_curve_slope",
                    "cross_asset_flow_ratio",
                    "option_cross_correl",
                    "microstructure_cross_impact",
                    "hft_cross_correl",
                ]
            ].head()
        )
        miya_speak(
            "Test compute_cross_asset_features terminé",
            tag="TEST",
            voice_profile="calm",
            priority=1,
        )
        send_alert("Test compute_cross_asset_features terminé", priority=1)
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
