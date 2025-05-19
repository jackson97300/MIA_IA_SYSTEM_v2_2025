# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/features/extractors/latent_factor_generator.py
# Génère des features latentes via t-SNE, PCA, ou clustering.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Calcule 21 features latentes (18 existantes + 3 nouvelles pour Phases 13, 15, 18),
#        utilise IQFeed pour les nouvelles, prix OHLC, IV options, microstructure, et HFT,
#        intègre SHAP (méthode 17), et enregistre logs psutil.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, scikit-learn>=1.5.0,<2.0.0, psutil>=5.9.0,<6.0.0, matplotlib>=3.7.0,<4.0.0, json, gzip, hashlib, logging
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
#
# Inputs :
# - config/es_config.yaml
# - data/iqfeed/news_data.csv
# - data/iqfeed/ohlc_data.csv
# - data/iqfeed/options_data.csv
# - data/iqfeed/microstructure_data.csv
# - data/iqfeed/hft_data.csv
# - data/features/feature_importance.csv
#
# Outputs :
# - data/features/cache/latent_factors/
# - data/logs/latent_factor_performance.csv
# - data/features/latent_factor_snapshots/
# - data/figures/latent_factors/
#
# Notes :
# - Utilise exclusivement IQFeed pour les données (remplace dxFeed).
# - Suppression de toute référence à dxFeed, obs_t, 320 features, et 81 features pour conformité avec MIA_IA_SYSTEM_v2_2025.
# - Intègre les Phases 1-18 :
#   - Phase 1 (news_scraper.py, news_analyzer.py) : Métriques topic_vector_news_*.
#   - Phase 13 (orderflow_indicators.py, options_metrics.py) : Nouvelle métrique latent_option_skew_vec.
#   - Phase 15 (microstructure_guard.py, spotgamma_recalculator.py) : Nouvelle métrique latent_microstructure_vec.
#   - Phase 18 : Nouvelle métrique latent_hft_activity_vec.
# - Tests unitaires disponibles dans tests/test_latent_factor_generator.py.
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.model.utils.alert_manager import send_alert
from src.model.utils.config_manager import config_manager
from src.model.utils.miya_console import miya_alerts, miya_speak

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data", "features", "cache", "latent_factors")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "features", "latent_factor_snapshots")
CSV_LOG_PATH = os.path.join(BASE_DIR, "data", "logs", "latent_factor_performance.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures", "latent_factors")
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
    filename=os.path.join(BASE_DIR, "data", "logs", "latent_factor.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 2.0
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes


class LatentFactorGenerator:

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
                "LatentFactorGenerator initialisé",
                tag="LATENT_FACTOR",
                voice_profile="calm",
                priority=2,
            )
            send_alert("LatentFactorGenerator initialisé", priority=1)
            logger.info("LatentFactorGenerator initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config_path": config_path})
        except Exception as e:
            error_msg = f"Erreur initialisation LatentFactorGenerator: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="LATENT_FACTOR", voice_profile="urgent", priority=3
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
        """
        Charge la configuration depuis le fichier YAML via config_manager.
        """

        def load_yaml():
            config = config_manager.get_config(os.path.basename(config_path))
            if "latent_factor_generator" not in config:
                raise ValueError(
                    "Clé 'latent_factor_generator' manquante dans la configuration"
                )
            required_keys = ["buffer_size", "max_cache_size", "cache_hours"]
            missing_keys = [
                key
                for key in required_keys
                if key not in config["latent_factor_generator"]
            ]
            if missing_keys:
                raise ValueError(
                    f"Clés manquantes dans 'latent_factor_generator': {missing_keys}"
                )
            return config["latent_factor_generator"]

        try:
            start_time = time.time()
            config = self.with_retries(load_yaml)
            latency = time.time() - start_time
            miya_speak(
                "Configuration latent_factor_generator chargée via config_manager",
                tag="LATENT_FACTOR",
                voice_profile="calm",
                priority=2,
            )
            send_alert(
                "Configuration latent_factor_generator chargée via config_manager",
                priority=1,
            )
            logger.info(
                "Configuration latent_factor_generator chargée via config_manager"
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
                error_msg, tag="LATENT_FACTOR", voice_profile="urgent", priority=3
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
                    tag="LATENT_FACTOR",
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
                tag="LATENT_FACTOR",
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
                tag="LATENT_FACTOR",
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
                tag="LATENT_FACTOR",
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
                tag="LATENT_FACTOR",
                voice_profile="calm",
                priority=2,
            )
            send_alert("SIGINT received, snapshot saved", priority=2)
            logger.info("SIGINT received, snapshot saved")
        except Exception as e:
            miya_alerts(
                f"Erreur gestion SIGINT: {str(e)}",
                tag="LATENT_FACTOR",
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
                    tag="LATENT_FACTOR",
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
                    tag="LATENT_FACTOR",
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
                    tag="LATENT_FACTOR",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(
                    f"Features non incluses dans top 150 SHAP: {missing}", priority=3
                )
                logger.warning(f"Features non incluses dans top 150 SHAP: {missing}")
            miya_speak(
                "SHAP features validées",
                tag="LATENT_FACTOR",
                voice_profile="calm",
                priority=1,
            )
            send_alert("SHAP features validées", priority=1)
            logger.info("SHAP features validées")
            return True
        except Exception as e:
            miya_alerts(
                f"Erreur validation SHAP features: {str(e)}",
                tag="LATENT_FACTOR",
                voice_profile="urgent",
                priority=4,
            )
            send_alert(f"Erreur validation SHAP features: {str(e)}", priority=4)
            logger.error(f"Erreur validation SHAP features: {str(e)}")
            return False

    def compute_t_sne_vol_regime(
        self, data: pd.DataFrame, component: int = 3
    ) -> pd.DataFrame:
        """
        Calcule les features t-SNE pour le régime de volatilité (latent_vol_regime_vec_3/4).

        Args:
            data (pd.DataFrame): Données avec features de volatilité (ex. : atr_14, vix_term_1m).
            component (int): Nombre de composantes t-SNE (3 ou 4).

        Returns:
            pd.DataFrame: Données avec les composantes t-SNE.
        """
        try:
            start_time = time.time()
            vol_features = ["atr_14", "vix_term_1m", "volatility_trend"]
            available_features = [f for f in vol_features if f in data.columns]
            if not available_features:
                error_msg = "Aucune feature de volatilité disponible pour t-SNE"
                miya_alerts(
                    error_msg, tag="LATENT_FACTOR", voice_profile="urgent", priority=4
                )
                send_alert(error_msgsuperiority=4)
                logger.error(error_msg)
                return pd.DataFrame(
                    {
                        f"latent_vol_regime_vec_{i+1}": [0.0] * len(data)
                        for i in range(component)
                    },
                    index=data.index,
                )

            X = data[available_features].fillna(0.0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            tsne = TSNE(
                n_components=component, random_state=42, perplexity=min(30, len(X) - 1)
            )
            tsne_result = tsne.fit_transform(X_scaled)
            result = pd.DataFrame(
                {
                    f"latent_vol_regime_vec_{i+1}": tsne_result[:, i]
                    for i in range(component)
                },
                index=data.index,
            )
            latency = time.time() - start_time
            self.log_performance(
                f"compute_t_sne_vol_regime_{component}", latency, success=True
            )
            return result
        except Exception as e:
            self.log_performance(
                f"compute_t_sne_vol_regime_{component}", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_t_sne_vol_regime: {str(e)}",
                tag="LATENT_FACTOR",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_t_sne_vol_regime: {str(e)}", priority=4)
            logger.error(f"Erreur dans compute_t_sne_vol_regime: {str(e)}")
            return pd.DataFrame(
                {
                    f"latent_vol_regime_vec_{i+1}": [0.0] * len(data)
                    for i in range(component)
                },
                index=data.index,
            )

    def compute_news_topic_vector(
        self, news_data: pd.DataFrame, component: int = 3
    ) -> pd.DataFrame:
        """
        Calcule les features t-SNE pour les topics de nouvelles (topic_vector_news_3/4).

        Args:
            news_data (pd.DataFrame): Données avec topic_score_1, topic_score_2, etc.
            component (int): Nombre de composantes t-SNE (3 ou 4).

        Returns:
            pd.DataFrame: Données avec les composantes t-SNE.
        """
        try:
            start_time = time.time()
            topic_features = [
                f"topic_score_{i}"
                for i in range(1, 5)
                if f"topic_score_{i}" in news_data.columns
            ]
            if not topic_features:
                error_msg = "Aucune feature de topic disponible pour t-SNE"
                miya_alerts(
                    error_msg, tag="LATENT_FACTOR", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.DataFrame(
                    {
                        f"topic_vector_news_{i+1}": [0.0] * len(news_data)
                        for i in range(component)
                    },
                    index=news_data.index,
                )

            X = news_data[topic_features].fillna(0.0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            tsne = TSNE(
                n_components=component, random_state=42, perplexity=min(30, len(X) - 1)
            )
            tsne_result = tsne.fit_transform(X_scaled)
            result = pd.DataFrame(
                {
                    f"topic_vector_news_{i+1}": tsne_result[:, i]
                    for i in range(component)
                },
                index=news_data.index,
            )
            latency = time.time() - start_time
            self.log_performance(
                f"compute_news_topic_vector_{component}", latency, success=True
            )
            return result
        except Exception as e:
            self.log_performance(
                f"compute_news_topic_vector_{component}", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_news_topic_vector: {str(e)}",
                tag="LATENT_FACTOR",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_news_topic_vector: {str(e)}", priority=4)
            logger.error(f"Erreur dans compute_news_topic_vector: {str(e)}")
            return pd.DataFrame(
                {
                    f"topic_vector_news_{i+1}": [0.0] * len(news_data)
                    for i in range(component)
                },
                index=news_data.index,
            )

    def compute_t_sne_momentum(self, data: pd.DataFrame) -> pd.Series:
        """
        Calcule la feature t-SNE pour le momentum de marché (latent_market_momentum_vec).

        Args:
            data (pd.DataFrame): Données avec features de momentum (ex. : spy_lead_return, order_flow_acceleration).

        Returns:
            pd.Series: Feature t-SNE pour le momentum.
        """
        try:
            start_time = time.time()
            momentum_features = [
                "spy_lead_return",
                "order_flow_acceleration",
                "spy_momentum_diff",
            ]
            available_features = [f for f in momentum_features if f in data.columns]
            if not available_features:
                error_msg = "Aucune feature de momentum disponible pour t-SNE"
                miya_alerts(
                    error_msg, tag="LATENT_FACTOR", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(
                    0.0, index=data.index, name="latent_market_momentum_vec"
                )

            X = data[available_features].fillna(0.0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            tsne = TSNE(n_components=1, random_state=42, perplexity=min(30, len(X) - 1))
            tsne_result = tsne.fit_transform(X_scaled)
            result = pd.Series(
                tsne_result[:, 0], index=data.index, name="latent_market_momentum_vec"
            )
            latency = time.time() - start_time
            self.log_performance("compute_t_sne_momentum", latency, success=True)
            return result
        except Exception as e:
            self.log_performance(
                "compute_t_sne_momentum", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_t_sne_momentum: {str(e)}",
                tag="LATENT_FACTOR",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_t_sne_momentum: {str(e)}", priority=4)
            logger.error(f"Erreur dans compute_t_sne_momentum: {str(e)}")
            return pd.Series(0.0, index=data.index, name="latent_market_momentum_vec")

    def compute_t_sne_order_flow(
        self, data: pd.DataFrame, component: int = 1
    ) -> pd.DataFrame:
        """
        Calcule les features t-SNE pour le flux d'ordres (latent_order_flow_vec_1/2).

        Args:
            data (pd.DataFrame): Données avec features de flux d'ordres (ex. : orderbook_imbalance).
            component (int): Nombre de composantes t-SNE (1 ou 2).

        Returns:
            pd.DataFrame: Données avec les composantes t-SNE.
        """
        try:
            start_time = time.time()
            order_flow_features = [
                "orderbook_imbalance",
                "depth_imbalance",
                "order_flow_acceleration",
            ]
            available_features = [f for f in order_flow_features if f in data.columns]
            if not available_features:
                error_msg = "Aucune feature de flux d'ordres disponible pour t-SNE"
                miya_alerts(
                    error_msg, tag="LATENT_FACTOR", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.DataFrame(
                    {
                        f"latent_order_flow_vec_{i+1}": [0.0] * len(data)
                        for i in range(component)
                    },
                    index=data.index,
                )

            X = data[available_features].fillna(0.0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            tsne = TSNE(
                n_components=component, random_state=42, perplexity=min(30, len(X) - 1)
            )
            tsne_result = tsne.fit_transform(X_scaled)
            result = pd.DataFrame(
                {
                    f"latent_order_flow_vec_{i+1}": tsne_result[:, i]
                    for i in range(component)
                },
                index=data.index,
            )
            latency = time.time() - start_time
            self.log_performance(
                f"compute_t_sne_order_flow_{component}", latency, success=True
            )
            return result
        except Exception as e:
            self.log_performance(
                f"compute_t_sne_order_flow_{component}", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_t_sne_order_flow: {str(e)}",
                tag="LATENT_FACTOR",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_t_sne_order_flow: {str(e)}", priority=4)
            logger.error(f"Erreur dans compute_t_sne_order_flow: {str(e)}")
            return pd.DataFrame(
                {
                    f"latent_order_flow_vec_{i+1}": [0.0] * len(data)
                    for i in range(component)
                },
                index=data.index,
            )

    def compute_t_sne_regime_stability(self, data: pd.DataFrame) -> pd.Series:
        """
        Calcule la feature t-SNE pour la stabilité du régime (latent_regime_stability_vec).

        Args:
            data (pd.DataFrame): Données avec features de régime (ex. : neural_regime, volatility_trend).

        Returns:
            pd.Series: Feature t-SNE pour la stabilité.
        """
        try:
            start_time = time.time()
            regime_features = [
                "neural_regime",
                "volatility_trend",
                "vix_es_correlation",
            ]
            available_features = [f for f in regime_features if f in data.columns]
            if not available_features:
                error_msg = "Aucune feature de régime disponible pour t-SNE"
                miya_alerts(
                    error_msg, tag="LATENT_FACTOR", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(
                    0.0, index=data.index, name="latent_regime_stability_vec"
                )

            X = data[available_features].fillna(0.0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            tsne = TSNE(n_components=1, random_state=42, perplexity=min(30, len(X) - 1))
            tsne_result = tsne.fit_transform(X_scaled)
            result = pd.Series(
                tsne_result[:, 0], index=data.index, name="latent_regime_stability_vec"
            )
            latency = time.time() - start_time
            self.log_performance(
                "compute_t_sne_regime_stability", latency, success=True
            )
            return result
        except Exception as e:
            self.log_performance(
                "compute_t_sne_regime_stability", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_t_sne_regime_stability: {str(e)}",
                tag="LATENT_FACTOR",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_t_sne_regime_stability: {str(e)}", priority=4
            )
            logger.error(f"Erreur dans compute_t_sne_regime_stability: {str(e)}")
            return pd.Series(0.0, index=data.index, name="latent_regime_stability_vec")

    def compute_pca(
        self, data: pd.DataFrame, feature_type: str, component: int = 1
    ) -> pd.Series:
        """
        Calcule les composantes PCA pour les prix ou IV (pca_price_1-3, pca_iv_1-3).

        Args:
            data (pd.DataFrame): Données avec features (ex. : open, high, low, close pour prix; iv_atm pour IV).
            feature_type (str): Type de feature ("price" ou "iv").
            component (int): Numéro de la composante PCA (1, 2, ou 3).

        Returns:
            pd.Series: Composante PCA spécifiée.
        """
        try:
            start_time = time.time()
            if feature_type == "price":
                features = ["open", "high", "low", "close"]
                feature_name = f"pca_price_{component}"
            elif feature_type == "iv":
                features = ["iv_atm", "option_skew", "vix_term_1m"]
                feature_name = f"pca_iv_{component}"
            else:
                error_msg = f"Type de feature invalide: {feature_type}"
                miya_alerts(
                    error_msg, tag="LATENT_FACTOR", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=data.index, name=feature_name)

            available_features = [f for f in features if f in data.columns]
            if not available_features:
                error_msg = f"Aucune feature {feature_type} disponible pour PCA"
                miya_alerts(
                    error_msg, tag="LATENT_FACTOR", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(0.0, index=data.index, name=feature_name)

            X = data[available_features].fillna(0.0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=3, random_state=42)
            pca_result = pca.fit_transform(X_scaled)
            result = pd.Series(
                pca_result[:, component - 1], index=data.index, name=feature_name
            )
            latency = time.time() - start_time
            self.log_performance(
                f"compute_pca_{feature_type}_{component}", latency, success=True
            )
            return result
        except Exception as e:
            self.log_performance(
                f"compute_pca_{feature_type}_{component}",
                0,
                success=False,
                error=str(e),
            )
            miya_alerts(
                f"Erreur dans compute_pca_{feature_type}: {str(e)}",
                tag="LATENT_FACTOR",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur dans compute_pca_{feature_type}: {str(e)}", priority=4)
            logger.error(f"Erreur dans compute_pca_{feature_type}: {str(e)}")
            return pd.Series(0.0, index=data.index, name=feature_name)

    def compute_latent_option_skew_vec(self, options_data: pd.DataFrame) -> pd.Series:
        """
        Calcule la feature t-SNE pour le skew des options (Phase 13).

        Args:
            options_data (pd.DataFrame): Données avec features d'options (ex. : option_skew, iv_atm).

        Returns:
            pd.Series: Feature t-SNE pour le skew des options.
        """
        try:
            start_time = time.time()
            option_features = ["option_skew", "iv_atm", "vix_term_1m"]
            available_features = [
                f for f in option_features if f in options_data.columns
            ]
            if not available_features:
                error_msg = "Aucune feature d'options disponible pour t-SNE"
                miya_alerts(
                    error_msg, tag="LATENT_FACTOR", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(
                    0.0, index=options_data.index, name="latent_option_skew_vec"
                )

            X = options_data[available_features].fillna(0.0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            tsne = TSNE(n_components=1, random_state=42, perplexity=min(30, len(X) - 1))
            tsne_result = tsne.fit_transform(X_scaled)
            result = pd.Series(
                tsne_result[:, 0],
                index=options_data.index,
                name="latent_option_skew_vec",
            )
            latency = time.time() - start_time
            self.log_performance(
                "compute_latent_option_skew_vec", latency, success=True
            )
            return result
        except Exception as e:
            self.log_performance(
                "compute_latent_option_skew_vec", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_latent_option_skew_vec: {str(e)}",
                tag="LATENT_FACTOR",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_latent_option_skew_vec: {str(e)}", priority=4
            )
            logger.error(f"Erreur dans compute_latent_option_skew_vec: {str(e)}")
            return pd.Series(
                0.0, index=options_data.index, name="latent_option_skew_vec"
            )

    def compute_latent_microstructure_vec(
        self, microstructure_data: pd.DataFrame
    ) -> pd.Series:
        """
        Calcule la feature t-SNE pour les événements de microstructure (Phase 15).

        Args:
            microstructure_data (pd.DataFrame): Données avec features de microstructure (ex. : spoofing_score).

        Returns:
            pd.Series: Feature t-SNE pour les événements de microstructure.
        """
        try:
            start_time = time.time()
            microstructure_features = [
                "spoofing_score",
                "volume_anomaly",
                "orderbook_velocity",
            ]
            available_features = [
                f for f in microstructure_features if f in microstructure_data.columns
            ]
            if not available_features:
                error_msg = "Aucune feature de microstructure disponible pour t-SNE"
                miya_alerts(
                    error_msg, tag="LATENT_FACTOR", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(
                    0.0,
                    index=microstructure_data.index,
                    name="latent_microstructure_vec",
                )

            X = microstructure_data[available_features].fillna(0.0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            tsne = TSNE(n_components=1, random_state=42, perplexity=min(30, len(X) - 1))
            tsne_result = tsne.fit_transform(X_scaled)
            result = pd.Series(
                tsne_result[:, 0],
                index=microstructure_data.index,
                name="latent_microstructure_vec",
            )
            latency = time.time() - start_time
            self.log_performance(
                "compute_latent_microstructure_vec", latency, success=True
            )
            return result
        except Exception as e:
            self.log_performance(
                "compute_latent_microstructure_vec", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_latent_microstructure_vec: {str(e)}",
                tag="LATENT_FACTOR",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_latent_microstructure_vec: {str(e)}", priority=4
            )
            logger.error(f"Erreur dans compute_latent_microstructure_vec: {str(e)}")
            return pd.Series(
                0.0, index=microstructure_data.index, name="latent_microstructure_vec"
            )

    def compute_latent_hft_activity_vec(self, hft_data: pd.DataFrame) -> pd.Series:
        """
        Calcule la feature t-SNE pour l'activité HFT (Phase 18).

        Args:
            hft_data (pd.DataFrame): Données avec features HFT (ex. : hft_activity_score, trade_velocity).

        Returns:
            pd.Series: Feature t-SNE pour l'activité HFT.
        """
        try:
            start_time = time.time()
            hft_features = ["hft_activity_score", "trade_velocity"]
            available_features = [f for f in hft_features if f in hft_data.columns]
            if not available_features:
                error_msg = "Aucune feature HFT disponible pour t-SNE"
                miya_alerts(
                    error_msg, tag="LATENT_FACTOR", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return pd.Series(
                    0.0, index=hft_data.index, name="latent_hft_activity_vec"
                )

            X = hft_data[available_features].fillna(0.0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            tsne = TSNE(n_components=1, random_state=42, perplexity=min(30, len(X) - 1))
            tsne_result = tsne.fit_transform(X_scaled)
            result = pd.Series(
                tsne_result[:, 0], index=hft_data.index, name="latent_hft_activity_vec"
            )
            latency = time.time() - start_time
            self.log_performance(
                "compute_latent_hft_activity_vec", latency, success=True
            )
            return result
        except Exception as e:
            self.log_performance(
                "compute_latent_hft_activity_vec", 0, success=False, error=str(e)
            )
            miya_alerts(
                f"Erreur dans compute_latent_hft_activity_vec: {str(e)}",
                tag="LATENT_FACTOR",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur dans compute_latent_hft_activity_vec: {str(e)}", priority=4
            )
            logger.error(f"Erreur dans compute_latent_hft_activity_vec: {str(e)}")
            return pd.Series(0.0, index=hft_data.index, name="latent_hft_activity_vec")

    def compute_latent_features(
        self,
        data: pd.DataFrame,
        news_data: pd.DataFrame,
        ohlc_data: pd.DataFrame,
        options_data: pd.DataFrame,
        microstructure_data: pd.DataFrame,
        hft_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calcule les 21 features latentes à partir des données IQFeed et existantes.

        Args:
            data (pd.DataFrame): Données avec features existantes (volatilité, momentum, etc.).
            news_data (pd.DataFrame): Données de nouvelles avec topic_score_1, topic_score_2, etc.
            ohlc_data (pd.DataFrame): Données OHLC avec open, high, low, close.
            options_data (pd.DataFrame): Données d'options avec iv_atm, option_skew, etc.
            microstructure_data (pd.DataFrame): Données de microstructure avec spoofing_score, etc.
            hft_data (pd.DataFrame): Données HFT avec hft_activity_score, trade_velocity.

        Returns:
            pd.DataFrame: Données enrichies avec les 21 features latentes.
        """
        try:
            start_time = time.time()
            config = self.load_config_with_manager_new()

            if data.empty:
                error_msg = "DataFrame principal vide"
                miya_alerts(
                    error_msg, tag="LATENT_FACTOR", voice_profile="urgent", priority=5
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
                        "Features latentes récupérées du cache",
                        tag="LATENT_FACTOR",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert("Features latentes récupérées du cache", priority=1)
                    self.log_performance(
                        "compute_latent_features_cache_hit", 0, success=True
                    )
                    return cached_data

            data = data.copy()
            if "timestamp" not in data.columns:
                miya_speak(
                    "Colonne timestamp manquante, création par défaut",
                    tag="LATENT_FACTOR",
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
                    tag="LATENT_FACTOR",
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

            # Calcul des features t-SNE existantes
            vol_regime_3 = self.compute_t_sne_vol_regime(data, component=3)
            vol_regime_4 = self.compute_t_sne_vol_regime(data, component=4)
            news_topic_3 = self.compute_news_topic_vector(news_data, component=3)
            news_topic_4 = self.compute_news_topic_vector(news_data, component=4)
            data["latent_market_momentum_vec"] = self.compute_t_sne_momentum(data)
            order_flow_1 = self.compute_t_sne_order_flow(data, component=1)
            order_flow_2 = self.compute_t_sne_order_flow(data, component=2)
            data["latent_regime_stability_vec"] = self.compute_t_sne_regime_stability(
                data
            )

            # Calcul des features PCA existantes
            data["pca_price_1"] = self.compute_pca(ohlc_data, "price", component=1)
            data["pca_price_2"] = self.compute_pca(ohlc_data, "price", component=2)
            data["pca_price_3"] = self.compute_pca(ohlc_data, "price", component=3)
            data["pca_iv_1"] = self.compute_pca(options_data, "iv", component=1)
            data["pca_iv_2"] = self.compute_pca(options_data, "iv", component=2)
            data["pca_iv_3"] = self.compute_pca(options_data, "iv", component=3)

            # Calcul des nouvelles features t-SNE (Phases 13, 15, 18)
            data["latent_option_skew_vec"] = self.compute_latent_option_skew_vec(
                options_data
            )
            data["latent_microstructure_vec"] = self.compute_latent_microstructure_vec(
                microstructure_data
            )
            data["latent_hft_activity_vec"] = self.compute_latent_hft_activity_vec(
                hft_data
            )

            # Fusion des résultats t-SNE
            data = pd.concat(
                [
                    data,
                    vol_regime_3,
                    vol_regime_4,
                    news_topic_3,
                    news_topic_4,
                    order_flow_1,
                    order_flow_2,
                ],
                axis=1,
            )

            metrics = [
                "latent_vol_regime_vec_1",
                "latent_vol_regime_vec_2",
                "latent_vol_regime_vec_3",
                "latent_vol_regime_vec_4",
                "topic_vector_news_1",
                "topic_vector_news_2",
                "topic_vector_news_3",
                "topic_vector_news_4",
                "latent_market_momentum_vec",
                "latent_order_flow_vec_1",
                "latent_order_flow_vec_2",
                "latent_regime_stability_vec",
                "pca_price_1",
                "pca_price_2",
                "pca_price_3",
                "pca_iv_1",
                "pca_iv_2",
                "pca_iv_3",
                "latent_option_skew_vec",
                "latent_microstructure_vec",
                "latent_hft_activity_vec",
            ]
            self.validate_shap_features(metrics)
            self.cache_metrics(data, cache_key)
            self.plot_metrics(data, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            latency = time.time() - start_time
            miya_speak(
                "Features latentes calculées",
                tag="LATENT_FACTOR",
                voice_profile="calm",
                priority=1,
            )
            send_alert("Features latentes calculées", priority=1)
            logger.info("Features latentes calculées")
            self.log_performance(
                "compute_latent_features",
                latency,
                success=True,
                num_rows=len(data),
                num_metrics=len(metrics),
            )
            self.save_snapshot(
                "compute_latent_features", {"num_rows": len(data), "metrics": metrics}
            )
            return data
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans compute_latent_features: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="LATENT_FACTOR", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=4)
            logger.error(error_msg)
            self.log_performance(
                "compute_latent_features", latency, success=False, error=str(e)
            )
            self.save_snapshot("compute_latent_features", {"error": str(e)})
            data["timestamp"] = pd.date_range(
                start=pd.Timestamp.now(), periods=len(data), freq="1min"
            )
            for col in [
                "latent_vol_regime_vec_1",
                "latent_vol_regime_vec_2",
                "latent_vol_regime_vec_3",
                "latent_vol_regime_vec_4",
                "topic_vector_news_1",
                "topic_vector_news_2",
                "topic_vector_news_3",
                "topic_vector_news_4",
                "latent_market_momentum_vec",
                "latent_order_flow_vec_1",
                "latent_order_flow_vec_2",
                "latent_regime_stability_vec",
                "pca_price_1",
                "pca_price_2",
                "pca_price_3",
                "pca_iv_1",
                "pca_iv_2",
                "pca_iv_3",
                "latent_option_skew_vec",
                "latent_microstructure_vec",
                "latent_hft_activity_vec",
            ]:
                data[col] = 0.0
            return data

    def cache_metrics(self, metrics: pd.DataFrame, cache_key: str) -> None:
        """
        Met en cache les features latentes.
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
                tag="LATENT_FACTOR",
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
                tag="LATENT_FACTOR",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur mise en cache métriques: {str(e)}", priority=4)
            logger.error(f"Erreur mise en cache métriques: {str(e)}")

    def plot_metrics(self, metrics: pd.DataFrame, timestamp: str) -> None:
        """
        Génère des visualisations des métriques latentes.

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
                metrics["latent_market_momentum_vec"],
                label="Market Momentum",
                color="blue",
            )
            plt.plot(
                metrics["timestamp"],
                metrics["latent_regime_stability_vec"],
                label="Regime Stability",
                color="green",
            )
            plt.plot(
                metrics["timestamp"],
                metrics["pca_price_1"],
                label="PCA Price 1",
                color="orange",
            )
            plt.plot(
                metrics["timestamp"],
                metrics["latent_option_skew_vec"],
                label="Option Skew",
                color="purple",
            )
            plt.plot(
                metrics["timestamp"],
                metrics["latent_microstructure_vec"],
                label="Microstructure",
                color="red",
            )
            plt.plot(
                metrics["timestamp"],
                metrics["latent_hft_activity_vec"],
                label="HFT Activity",
                color="cyan",
            )
            plt.title(f"Latent Factor Metrics - {timestamp}")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.savefig(
                os.path.join(FIGURES_DIR, f"latent_factor_metrics_{timestamp_safe}.png")
            )
            plt.close()
            latency = time.time() - start_time
            miya_speak(
                f"Visualisations générées: {FIGURES_DIR}",
                tag="LATENT_FACTOR",
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
                tag="LATENT_FACTOR",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur génération visualisations: {str(e)}", priority=4)
            logger.error(f"Erreur génération visualisations: {str(e)}")


if __name__ == "__main__":
    try:
        generator = LatentFactorGenerator()
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "atr_14": np.random.normal(1.5, 0.2, 100),
                "vix_term_1m": np.random.normal(20, 1, 100),
                "volatility_trend": np.random.normal(0, 0.1, 100),
                "spy_lead_return": np.random.normal(0.01, 0.005, 100),
                "order_flow_acceleration": np.random.normal(0, 0.1, 100),
                "spy_momentum_diff": np.random.normal(0, 0.05, 100),
                "orderbook_imbalance": np.random.normal(0, 0.2, 100),
                "depth_imbalance": np.random.normal(0, 0.2, 100),
                "neural_regime": np.random.randint(0, 3, 100),
                "vix_es_correlation": np.random.normal(0, 0.1, 100),
            }
        )
        news_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "topic_score_1": np.random.normal(0, 1, 100),
                "topic_score_2": np.random.normal(0, 1, 100),
                "topic_score_3": np.random.normal(0, 1, 100),
                "topic_score_4": np.random.normal(0, 1, 100),
            }
        )
        ohlc_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "open": np.random.normal(5100, 10, 100),
                "high": np.random.normal(5110, 10, 100),
                "low": np.random.normal(5090, 10, 100),
                "close": np.random.normal(5100, 10, 100),
            }
        )
        options_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "iv_atm": np.random.normal(0.15, 0.02, 100),
                "option_skew": np.random.normal(0.1, 0.01, 100),
                "vix_term_1m": np.random.normal(20, 1, 100),
            }
        )
        microstructure_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "spoofing_score": np.random.uniform(0, 1, 100),
                "volume_anomaly": np.random.uniform(0, 1, 100),
                "orderbook_velocity": np.random.uniform(0, 1, 100),
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
        result = generator.compute_latent_features(
            data, news_data, ohlc_data, options_data, microstructure_data, hft_data
        )
        print(
            result[
                [
                    "timestamp",
                    "latent_market_momentum_vec",
                    "pca_price_1",
                    "latent_regime_stability_vec",
                    "latent_option_skew_vec",
                    "latent_microstructure_vec",
                    "latent_hft_activity_vec",
                ]
            ].head()
        )
        miya_speak(
            "Test compute_latent_features terminé",
            tag="TEST",
            voice_profile="calm",
            priority=1,
        )
        send_alert("Test compute_latent_features terminé", priority=1)
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
