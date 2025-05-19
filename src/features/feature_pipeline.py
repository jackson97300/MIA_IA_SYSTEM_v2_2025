# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/features/feature_pipeline.py
# Pipeline de traitement des données brutes IQFeed vers 350 features pour 
# MIA_IA_SYSTEM_v2_2025.
# Intègre les niveaux critiques d’options, calcule gex_change, et sélectionne 
# les top 150 SHAP.
# Optimisations : Cache local, SHAP pour top 150 features, parallélisation, 
# mémoire contextuelle.
#
# Version : 2.1.5
# Date : 2025-05-14
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, ta>=0.10.0,<0.11.0, 
#   psutil>=5.9.0,<6.0.0, joblib>=1.2.0,<1.3.0, shap>=0.45.0,<0.46.0, sqlite3, 
#   json, matplotlib>=3.7.0,<3.8.0, sklearn>=1.0.2,<1.1.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
# - src/model/utils/obs_template.py
# - src/api/option_chain_fetch.py
# - src/data/data_provider.py
# - src/features/features_audit.py
# - src/features/pca_orderflow.py
# - src/features/extractors/encode_time_context.py
# - src/features/extractors/orderflow_indicators.py
# - src/features/extractors/volume_profile.py
# - src/features/extractors/smart_scores.py
# - src/features/extractors/volatility_metrics.py
# - src/features/options_calculator.py
# - src/features/neural_pipeline.py
# - src/features/option_metrics.py
# - src/features/advanced_feature_generator.py
# - src/features/meta_features.py
# - src/features/market_structure_signals.py
# - src/features/contextual_state_encoder.py
# - src/features/feature_meta_ensemble.py
# - src/features/spotgamma_recalculator.py
# - src/features/news_analyzer.py
# - src/features/order_flow_calculator.py
# - src/features/extractors/context_aware_filter.py
# - src/features/extractors/cross_asset_processor.py
# - src/features/extractors/latent_factor_generator.py
# - src/features/extractors/microstructure_guard.py
# - src/features/extractors/news_metrics.py
# - src/model/trade_probability.py
# - src/utils/telegram_alert.py
# - src/features/regime_detector.py
# - src/data/validate_data.py
# - src/data/data_lake.py
# - src/utils/error_tracker.py
# - src/monitoring/prometheus_metrics.py
#
# Inputs :
# - config/feature_pipeline_config.yaml
# - config/credentials.yaml
# - config/feature_sets.yaml
# - config/iqfeed_config.yaml
# - data/iqfeed/merged_data.csv
# - data/iqfeed/option_chain.csv
# - data/features/spotgamma_metrics.csv
# - data/iqfeed/news_data.csv
# - data/iqfeed/futures_data.csv
# - data/events/macro_events.csv
# - data/events/event_volatility_history.csv
#
# Outputs :
# - data/features/merged_data.csv
# - data/features/features_latest_filtered.csv
# - data/logs/feature_pipeline_performance.csv
# - data/features/snapshots/*.json (option *.json.gz)
# - data/features/cache/shap/
# - data/figures/feature_pipeline/
# - data/features/feature_importance.csv
# - data/features/feature_importance_cache.csv
# - data/checkpoints/
#
# Notes :
# - Conforme à structure.txt (version 2.1.5, 2025-05-14).
# - Préserve toutes les fonctionnalités existantes (81 features statiques comme 
#   référence).
# - Supprime volatility_skew, iv_slope_10_25, gamma_net_spread, 
#   gex_rsi_interaction, volatility_delta_corr pour éviter les doublons avec 
#   iv_atm, option_skew, gex_slope.
# - Intègre context_aware_filter (19 features), cross_asset_processor 
#   (6 features), latent_factor_generator (14 features).
# - Utilise data_provider.py pour récupération IQFeed-exclusive, avec fallback 
#   temporaire vers option_chain_fetch.py (dxFeed) jusqu’à juin 2025.
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des 
#   features générées.
# - Intègre l’analyse SHAP (Phase 17) pour sélectionner les top 150 features, 
#   avec snapshots et logs de performance.
# - Sauvegarde des snapshots dissenting non compressés par défaut avec option de 
#   compression gzip.
# - Utilise AlertManager et telegram_alert pour les notifications critiques.
# - Tests unitaires disponibles dans tests/test_feature_pipeline.py.
# - Évolution future : Migration API Investing.com (juin 2025), intégration avec 
#   live_trading.py, suppression complète du fallback dxFeed (juin 2025).
# - Nouvelles fonctionnalités : Dimensionnement dynamique (ATR, orderflow 
#   imbalance), coûts de transaction (slippage), microstructure (bid-ask 
#   imbalance, trade aggressiveness), HMM/changepoint (regime_hmm), surface de 
#   volatilité (iv_skew, iv_term_structure), métriques Prometheus, logs psutil, 
#   validation via validate_data.py, stockage dans data_lake.py.

# Note : L'erreur F841 concernant les variables inutilisées est partiellement une 
# fausse alerte dans la première partie.
# Les paramètres signum et frame dans handle_sigint sont requis mais non utilisés.

import gzip
import hashlib
import json
import logging
import os
import signal
import sqlite3
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import shap
import ta
from joblib import Parallel, delayed
from loguru import logger as loguru_logger
from sklearn.cluster import KMeans

from src.api.option_chain_fetch import fetch_option_chain
from src.data.data_lake import DataLake
from src.data.data_provider import get_data_provider
from src.data.validate_data import validate_features
from src.features.advanced_feature_generator import calculate_advanced_features
from src.features.contextual_state_encoder import calculate_contextual_encodings
from src.features.extractors.context_aware_filter import ContextAwareFilter
from src.features.extractors.cross_asset_processor import CrossAssetProcessor
from src.features.extractors.encode_time_context import encode_time_context
from src.features.extractors.latent_factor_generator import LatentFactorGenerator
from src.features.extractors.microstructure_guard import (
    detect_microstructure_anomalies,
)
from src.features.extractors.news_metrics import calculate_news_metrics
from src.features.extractors.orderflow_indicators import (
    calculate_orderflow_indicators,
)
from src.features.extractors.smart_scores import calculate_smart_scores
from src.features.extractors.volatility_metrics import calculate_volatility_metrics
from src.features.extractors.volume_profile import extract_volume_profile
from src.features.features_audit import audit_features
from src.features.market_structure_signals import calculate_cross_market_signals
from src.features.meta_features import calculate_meta_features
from src.features.option_metrics import calculate_option_metrics
from src.features.options_calculator import calculate_options_features
from src.features.order_flow_calculator import calculate_order_flow_features
from src.features.pca_orderflow import apply_pca_orderflow
from src.features.regime_detector import RegimeDetector
from src.features.spotgamma_recalculator import recalculate_levels
from src.model.trade_probability import TradeProbabilityPredictor
from src.model.utils.alert_manager import AlertManager, send_alert
from src.model.utils.config_manager import config_manager, get_features
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.monitoring.prometheus_metrics import Gauge
from src.utils.error_tracker import capture_error
from src.utils.telegram_alert import send_telegram_alert

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Configuration du logging
os.makedirs(os.path.join(BASE_DIR, "data", "logs"), exist_ok=True)
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "data", "logs", "feature_pipeline.log"),
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Configuration du logger Loguru
LOG_DIR = Path(BASE_DIR) / "data" / "logs"
LOG_DIR.mkdir(exist_ok=True)
loguru_logger.remove()
loguru_logger.add(
    LOG_DIR / "feature_pipeline.log",
    rotation="10 MB",
    level="INFO",
    encoding="utf-8",
)

# Chemins de snapshots, performance, et cache
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "features", "snapshots")
CSV_LOG_PATH = os.path.join(
    BASE_DIR,
    "data",
    "logs",
    "feature_pipeline_performance.csv",
)
DASHBOARD_PATH = os.path.join(BASE_DIR, "data", "feature_pipeline_dashboard.json")
SHAP_CACHE_DIR = os.path.join(BASE_DIR, "data", "features", "cache", "shap")
DB_PATH = os.path.join(BASE_DIR, "data", "market_memory.db")
SPOTGAMMA_METRICS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "features",
    "spotgamma_metrics.csv",
)
FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "features",
    "feature_importance.csv",
)
FEATURE_IMPORTANCE_CACHE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "features",
    "feature_importance_cache.csv",
)
NEWS_DATA_PATH = os.path.join(BASE_DIR, "data", "iqfeed", "news_data.csv")
FUTURES_DATA_PATH = os.path.join(BASE_DIR, "data", "iqfeed", "futures_data.csv")
OPTION_CHAIN_PATH = os.path.join(BASE_DIR, "data", "iqfeed", "option_chain.csv")

# Seuils de performance
PERFORMANCE_THRESHOLDS = {
    "min_features": 350,
    "min_rows": 50,
    "max_shap_features": 150,
    "max_key_shap_features": 150,
}

# Métriques Prometheus
atr_dynamic_metric = Gauge(
    name="atr_dynamic", description="ATR dynamique sur 1-5 min", labelnames=["market"]
)
orderflow_imbalance_metric = Gauge(
    name="orderflow_imbalance",
    description="Imbalance du carnet",
    labelnames=["market"],
)
slippage_estimate_metric = Gauge(
    name="slippage_estimate",
    description="Estimation du slippage",
    labelnames=["market"],
)
bid_ask_imbalance_metric = Gauge(
    name="bid_ask_imbalance", description="Imbalance bid-ask", labelnames=["market"]
)
trade_aggressiveness_metric = Gauge(
    name="trade_aggressiveness",
    description="Agressivité des trades",
    labelnames=["market"],
)
iv_skew_metric = Gauge(
    name="iv_skew",
    description="Skew de volatilité implicite",
    labelnames=["market"],
)
iv_term_metric = Gauge(
    name="iv_term_structure",
    description="Structure de terme IV",
    labelnames=["market"],
)


class FeaturePipeline:
    """Pipeline de traitement des données brutes IQFeed vers 350 features."""

    def __init__(self, market: str = "ES"):
        self.market = market
        self.log_buffer = []
        self.cache = {}
        self.config_path = os.path.join(
            BASE_DIR,
            "config",
            "feature_pipeline_config.yaml",
        )
        self.alert_manager = AlertManager()
        self.regime_detector = RegimeDetector(market=market)
        self.data_lake = DataLake()
        try:
            self.config = self.load_config(self.config_path)
            required_config = [
                "performance_thresholds",
                "neural_pipeline",
                "logging",
                "cache",
            ]
            missing_config = [
                key for key in required_config if key not in self.config
            ]
            if missing_config:
                error_msg = f"Clés de configuration manquantes: {missing_config}"
                raise ValueError(error_msg)
            self.buffer_size = self.config.get("logging", {}).get(
                "buffer_size", 100
            )
            self.max_cache_size = self.config.get("cache", {}).get(
                "max_cache_size", 1000
            )
            global PERFORMANCE_THRESHOLDS
            PERFORMANCE_THRESHOLDS = self.config.get(
                "performance_thresholds",
                PERFORMANCE_THRESHOLDS,
            )
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            os.makedirs(SHAP_CACHE_DIR, exist_ok=True)
            figures_dir = os.path.join(BASE_DIR, "data", "figures", "feature_pipeline")
            os.makedirs(figures_dir, exist_ok=True)
            self.context_aware_filter = ContextAwareFilter()
            self.cross_asset_processor = CrossAssetProcessor()
            self.latent_factor_generator = LatentFactorGenerator()
            miya_speak(
                message="FeaturePipeline initialisé",
                tag="FEATURE_PIPELINE",
                voice_profile="calm",
                priority=2,
            )
            send_alert(message="FeaturePipeline initialisé", priority=2)
            send_telegram_alert(message="FeaturePipeline initialisé")
            logger.info("FeaturePipeline initialisé")
            signal.signal(signal.SIGINT, self.handle_sigint)
        except Exception as e:
            error_msg = (
                f"Erreur initialisation: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                message=error_msg,
                tag="FEATURE_PIPELINE",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(message=error_msg, priority=3)
            send_telegram_alert(message=error_msg)
            logger.error(error_msg)
            self.config = {
                "performance_thresholds": PERFORMANCE_THRESHOLDS,
                "neural_pipeline": {"window_size": 50},
                "logging": {"buffer_size": 100},
                "cache": {"max_cache_size": 1000},
            }
            self.buffer_size = 100
            self.max_cache_size = 1000
            self.context_aware_filter = None
            self.cross_asset_processor = None
            self.latent_factor_generator = None

    def calculate_atr_dynamic(self, data: pd.DataFrame, window: int = 5) -> pd.Series:
        """Calcule l'ATR dynamique sur 1-5 min."""
        start_time = time.time()
        try:
            high_low = data["high"] - data["low"]
            high_close = np.abs(data["high"] - data["close"].shift())
            low_close = np.abs(data["low"] - data["close"].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=window).mean()
            atr_dynamic_metric.labels(market=self.market).set(atr.iloc[-1])
            latency = time.time() - start_time
            self.log_performance(
                operation="calculate_atr_dynamic",
                latency=latency,
                success=True,
            )
            return atr
        except Exception as e:
            error_msg = f"Erreur calcul ATR dynamique: {str(e)}"
            capture_error(
                error=e,
                context={"market": self.market},
                market=self.market,
                operation="calculate_atr_dynamic",
            )
            logger.error(error_msg)
            self.log_performance(
                operation="calculate_atr_dynamic",
                latency=time.time() - start_time,
                success=False,
                error=str(e),
            )
            return pd.Series(0, index=data.index)

    def calculate_orderflow_imbalance(self, data: pd.DataFrame) -> pd.Series:
        """Calcule l'imbalance du carnet d'ordres."""
        start_time = time.time()
        try:
            imbalance = (data["bid_volume"] - data["ask_volume"]) / (
                data["bid_volume"] + data["ask_volume"]
            )
            orderflow_imbalance_metric.labels(market=self.market).set(
                imbalance.iloc[-1]
            )
            latency = time.time() - start_time
            self.log_performance(
                operation="calculate_orderflow_imbalance",
                latency=latency,
                success=True,
            )
            return imbalance
        except Exception as e:
            error_msg = f"Erreur calcul orderflow imbalance: {str(e)}"
            capture_error(
                error=e,
                context={"market": self.market},
                market=self.market,
                operation="calculate_orderflow_imbalance",
            )
            logger.error(error_msg)
            self.log_performance(
                operation="calculate_orderflow_imbalance",
                latency=time.time() - start_time,
                success=False,
                error=str(e),
            )
            return pd.Series(0, index=data.index)

    def calculate_slippage_estimate(self, data: pd.DataFrame) -> pd.Series:
        """Calcule l'estimation du slippage."""
        start_time = time.time()
        try:
            slippage = data["bid_ask_spread"] * data["order_volume"]
            slippage_estimate_metric.labels(market=self.market).set(
                slippage.iloc[-1]
            )
            latency = time.time() - start_time
            self.log_performance(
                operation="calculate_slippage_estimate",
                latency=latency,
                success=True,
            )
            return slippage
        except Exception as e:
            error_msg = f"Erreur calcul slippage estimate: {str(e)}"
            capture_error(
                error=e,
                context={"market": self.market},
                market=self.market,
                operation="calculate_slippage_estimate",
            )
            logger.error(error_msg)
            self.log_performance(
                operation="calculate_slippage_estimate",
                latency=time.time() - start_time,
                success=False,
                error=str(e),
            )
            return pd.Series(0, index=data.index)

    def calculate_bid_ask_imbalance(self, data: pd.DataFrame) -> pd.Series:
        """Calcule l'imbalance bid-ask."""
        start_time = time.time()
        try:
            imbalance = (data["bid_volume"] - data["ask_volume"]) / (
                data["bid_volume"] + data["ask_volume"]
            )
            bid_ask_imbalance_metric.labels(market=self.market).set(
                imbalance.iloc[-1]
            )
            latency = time.time() - start_time
            self.log_performance(
                operation="calculate_bid_ask_imbalance",
                latency=latency,
                success=True,
            )
            return imbalance
        except Exception as e:
            error_msg = f"Erreur calcul bid-ask imbalance: {str(e)}"
            capture_error(
                error=e,
                context={"market": self.market},
                market=self.market,
                operation="calculate_bid_ask_imbalance",
            )
            logger.error(error_msg)
            self.log_performance(
                operation="calculate_bid_ask_imbalance",
                latency=time.time() - start_time,
                success=False,
                error=str(e),
            )
            return pd.Series(0, index=data.index)

    def calculate_trade_aggressiveness(self, data: pd.DataFrame) -> pd.Series:
        """Calcule l'agressivité des trades."""
        start_time = time.time()
        try:
            aggressiveness = data["taker_volume"] / (
                data["bid_volume"] + data["ask_volume"]
            )
            trade_aggressiveness_metric.labels(market=self.market).set(
                aggressiveness.iloc[-1]
            )
            latency = time.time() - start_time
            self.log_performance(
                operation="calculate_trade_aggressiveness",
                latency=latency,
                success=True,
            )
            return aggressiveness
        except Exception as e:
            error_msg = f"Erreur calcul trade aggressiveness: {str(e)}"
            capture_error(
                error=e,
                context={"market": self.market},
                market=self.market,
                operation="calculate_trade_aggressiveness",
            )
            logger.error(error_msg)
            self.log_performance(
                operation="calculate_trade_aggressiveness",
                latency=time.time() - start_time,
                success=False,
                error=str(e),
            )
            return pd.Series(0, index=data.index)

    def calculate_iv_skew(self, data: pd.DataFrame) -> pd.Series:
        """Calcule le skew de volatilité implicite."""
        start_time = time.time()
        try:
            iv_skew = (data["iv_call"] - data["iv_put"]) / data["strike"]
            iv_skew_metric.labels(market=self.market).set(iv_skew.iloc[-1])
            latency = time.time() - start_time
            self.log_performance(
                operation="calculate_iv_skew",
                latency=latency,
                success=True,
            )
            return iv_skew
        except Exception as e:
            error_msg = f"Erreur calcul iv skew: {str(e)}"
            capture_error(
                error=e,
                context={"market": self.market},
                market=self.market,
                operation="calculate_iv_skew",
            )
            logger.error(error_msg)
            self.log_performance(
                operation="calculate_iv_skew",
                latency=time.time() - start_time,
                success=False,
                error=str(e),
            )
            return pd.Series(0, index=data.index)

    def calculate_iv_term_structure(self, data: pd.DataFrame) -> pd.Series:
        """Calcule la structure de terme de la volatilité implicite."""
        start_time = time.time()
        try:
            iv_term = data["iv_3m"] - data["iv_1m"]
            iv_term_metric.labels(market=self.market).set(iv_term.iloc[-1])
            latency = time.time() - start_time
            self.log_performance(
                operation="calculate_iv_term_structure",
                latency=latency,
                success=True,
            )
            return iv_term
        except Exception as e:
            error_msg = f"Erreur calcul iv term structure: {str(e)}"
            capture_error(
                error=e,
                context={"market": self.market},
                market=self.market,
                operation="calculate_iv_term_structure",
            )
            logger.error(error_msg)
            self.log_performance(
                operation="calculate_iv_term_structure",
                latency=time.time() - start_time,
                success=False,
                error=str(e),
            )
            return pd.Series(0, index=data.index)

    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Génère toutes les features."""
        start_time = time.time()
        try:
            data["atr_dynamic"] = self.calculate_atr_dynamic(data)
            data["orderflow_imbalance"] = self.calculate_orderflow_imbalance(data)
            data["slippage_estimate"] = self.calculate_slippage_estimate(data)
            data["bid_ask_imbalance"] = self.calculate_bid_ask_imbalance(data)
            data["trade_aggressiveness"] = self.calculate_trade_aggressiveness(
                data
            )
            data["iv_skew"] = self.calculate_iv_skew(data)
            data["iv_term_structure"] = self.calculate_iv_term_structure(data)
            data["regime_hmm"] = self.regime_detector.detect_regime(data)
            new_features = [
                "atr_dynamic",
                "orderflow_imbalance",
                "slippage_estimate",
                "bid_ask_imbalance",
                "trade_aggressiveness",
                "iv_skew",
                "iv_term_structure",
                "regime_hmm",
            ]
            validation_result = validate_features(data[new_features])
            if not validation_result["valid"]:
                alert_msg = (
                    f"Validation des nouvelles features échouée: "
                    f"{validation_result['errors']}"
                )
                logger.warning(alert_msg)
                self.alert_manager.send_alert(message=alert_msg, priority=3)
                send_telegram_alert(message=alert_msg)
            self.data_lake.store_features(data[new_features], market=self.market)
            output_path = (
                Path(BASE_DIR) / "data" / "features" / "features_latest.csv"
            )
            data.to_csv(output_path, index=False, encoding="utf-8")
            latency = time.time() - start_time
            logger.info(f"Features générées pour {self.market}. Latence: {latency}s")
            self.alert_manager.send_alert(
                message=f"Features générées pour {self.market}",
                priority=2,
            )
            send_telegram_alert(message=f"Features générées pour {self.market}")
            self.log_performance(
                operation="generate_features",
                latency=latency,
                success=True,
                num_features=len(new_features),
            )
            return data
        except Exception as e:
            error_msg = f"Erreur génération features: {str(e)}"
            logger.error(error_msg)
            capture_error(
                error=e,
                context={"market": self.market},
                market=self.market,
                operation="generate_features",
            )
            self.alert_manager.send_alert(message=error_msg, priority=4)
            send_telegram_alert(message=error_msg)
            self.log_performance(
                operation="generate_features",
                latency=time.time() - start_time,
                success=False,
                error=str(e),
            )
            return data

    def handle_sigint(self, signum, frame):  # noqa: F841
        """Gère les interruptions SIGINT."""
        start_time = time.time()
        logger.info("Interruption détectée, sauvegarde de l’état...")
        recent_errors = len(
            [log for log in self.log_buffer if not log["success"]]
        )
        self.save_dashboard_status(
            status={
                "status": "interrupted",
                "num_rows": 0,
                "recent_errors": recent_errors,
            },
            compress=False,
        )
        latency = time.time() - start_time
        self.log_performance(
            operation="handle_sigint",
            latency=latency,
            success=True,
        )
        sys.exit(0)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Charge le fichier de configuration."""
        start_time = time.time()
        try:
            config = config_manager.get_config(os.path.basename(config_path))
            miya_speak(
                message=f"Configuration {config_path} chargée",
                tag="FEATURE_PIPELINE",
                voice_profile="calm",
                priority=1,
            )
            logger.info(f"Configuration {config_path} chargée")
            send_alert(message=f"Configuration {config_path} chargée", priority=1)
            send_telegram_alert(message=f"Configuration {config_path} chargée")
            latency = time.time() - start_time
            self.log_performance(
                operation="load_config",
                latency=latency,
                success=True,
            )
            return config
        except Exception as e:
            error_msg = (
                f"Erreur chargement config: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                message=error_msg,
                tag="FEATURE_PIPELINE",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(message=error_msg, priority=3)
            send_telegram_alert(message=error_msg)
            logger.error(error_msg)
            self.log_performance(
                operation="load_config",
                latency=time.time() - start_time,
                success=False,
                error=str(e),
            )
            raise

    def load_shap_fallback(self) -> List[str]:
        """Charge une liste statique de 150 features SHAP si le cache est absent."""
        start_time = time.time()
        try:
            shap_features = pd.read_csv(FEATURE_IMPORTANCE_CACHE_PATH)[
                "feature_name"
            ].tolist()
            if len(shap_features) == 150:
                latency = time.time() - start_time
                self.log_performance(
                    operation="load_shap_fallback",
                    latency=latency,
                    success=True,
                )
                return shap_features
        except FileNotFoundError:
            logger.warning("Cache SHAP absent, utilisation du fallback statique")
            send_alert(
                message="Cache SHAP absent, utilisation du fallback statique",
                priority=3,
            )
            send_telegram_alert(
                message="Cache SHAP absent, utilisation du fallback statique"
            )
        latency = time.time() - start_time
        self.log_performance(
            operation="load_shap_fallback",
            latency=latency,
            success=True,
        )
        return get_features("inference")

    # Note : L'erreur F841 concernant les variables inutilisées est une fausse alerte 
# dans la deuxième partie.
# Toutes les variables assignées sont utilisées dans les calculs ou les logs.

    def load_shap_fallback_regime(self, regime: str) -> pd.DataFrame:
        """Charge un cache ou une liste statique de 150 SHAP features en cas 
        d'échec."""
        start_time = time.time()
        try:
            cache_path = FEATURE_IMPORTANCE_CACHE_PATH
            if os.path.exists(cache_path):
                shap_df = pd.read_csv(cache_path)
                logger.info(f"SHAP features chargées depuis cache: {cache_path}")
                send_alert(
                    message=f"SHAP features chargées depuis cache: {cache_path}",
                    priority=1,
                )
                send_telegram_alert(
                    message=f"SHAP features chargées depuis cache: {cache_path}"
                )
                latency = time.time() - start_time
                self.log_performance(
                    operation="load_shap_fallback_regime",
                    latency=latency,
                    success=True,
                )
                return shap_df
            config = config_manager.get_config("feature_sets.yaml")
            static_features = (
                config.get("feature_sets", {})
                .get(regime, {})
                .get("inference", [])
            )
            shap_df = pd.DataFrame(
                {
                    "feature": static_features[:150],
                    "importance": 1.0,
                    "regime": regime,
                }
            )
            logger.info(
                "SHAP features chargées depuis liste statique: "
                "config/feature_sets.yaml"
            )
            send_alert(
                message=(
                    "SHAP features chargées depuis liste statique: "
                    "config/feature_sets.yaml"
                ),
                priority=1,
            )
            send_telegram_alert(
                message=(
                    "SHAP features chargées depuis liste statique: "
                    "config/feature_sets.yaml"
                )
            )
            latency = time.time() - start_time
            self.log_performance(
                operation="load_shap_fallback_regime",
                latency=latency,
                success=True,
            )
            return shap_df
        except Exception as e:
            error_msg = f"Erreur chargement SHAP fallback: {str(e)}"
            logger.error(error_msg)
            send_alert(message=error_msg, priority=3)
            send_telegram_alert(message=error_msg)
            self.log_performance(
                operation="load_shap_fallback_regime",
                latency=time.time() - start_time,
                success=False,
                error=str(e),
            )
            return pd.DataFrame()

    def save_checkpoints(self, data: pd.DataFrame, regime: str) -> None:
        """Sauvegarde incrémentielle et distribuée des features."""
        start_time = time.time()
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = os.path.join(
                BASE_DIR,
                "data",
                "checkpoints",
                f"features_{regime}_{timestamp}.csv.gz",
            )
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

            def write_checkpoint():
                data.to_csv(
                    checkpoint_path,
                    compression="gzip",
                    index=False,
                    encoding="utf-8",
                )

            self.with_retries(write_checkpoint)
            logger.info(f"Checkpoint sauvegardé: {checkpoint_path}")
            send_alert(
                message=f"Checkpoint sauvegardé: {checkpoint_path}",
                priority=1,
            )
            send_telegram_alert(message=f"Checkpoint sauvegardé: {checkpoint_path}")
            latency = time.time() - start_time
            self.log_performance(
                operation="save_checkpoints",
                latency=latency,
                success=True,
                num_rows=len(data),
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde checkpoint: {str(e)}"
            logger.error(error_msg)
            send_alert(message=error_msg, priority=3)
            send_telegram_alert(message=error_msg)
            self.log_performance(
                operation="save_checkpoints",
                latency=time.time() - start_time,
                success=False,
                error=str(e),
            )

    def with_retries(
        self, func: callable, max_attempts: int = 3, delay_base: float = 2.0
    ) -> Any:
        """Exécute une fonction avec retries (max 3, délai exponentiel)."""
        start_time = time.time()
        for attempt in range(max_attempts):
            try:
                result = func()
                latency = time.time() - start_time
                self.log_performance(
                    operation=f"retry_attempt_{attempt+1}",
                    latency=latency,
                    success=True,
                    attempt_number=attempt + 1,
                )
                return result
            except Exception as e:
                if attempt == max_attempts - 1:
                    error_msg = (
                        f"Échec après {max_attempts} tentatives: {str(e)}"
                    )
                    miya_alerts(
                        message=error_msg,
                        tag="FEATURE_PIPELINE",
                        voice_profile="urgent",
                        priority=4,
                    )
                    send_alert(message=error_msg, priority=4)
                    send_telegram_alert(message=error_msg)
                    logger.error(error_msg)
                    self.log_performance(
                        operation=f"retry_attempt_{attempt+1}",
                        latency=time.time() - start_time,
                        success=False,
                        error=str(e),
                        attempt_number=attempt + 1,
                    )
                    raise
                delay = delay_base * (2**attempt)
                miya_speak(
                    message=(
                        f"Tentative {attempt+1} échouée, retry après {delay}s"
                    ),
                    tag="FEATURE_PIPELINE",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(
                    message=(
                        f"Tentative {attempt+1} échouée, retry après {delay}s"
                    ),
                    priority=3,
                )
                send_telegram_alert(
                    message=(
                        f"Tentative {attempt+1} échouée, retry après {delay}s"
                    )
                )
                logger.warning(
                    f"Tentative {attempt+1} échouée, retry après {delay}s"
                )
                time.sleep(delay)

    def fetch_option_chain_iqfeed(self, symbol: str = "ES") -> pd.DataFrame:
        """Récupère la chaîne d’options via IQFeed, avec fallback temporaire 
        vers dxFeed (jusqu’à juin 2025)."""
        start_time = time.time()
        try:
            provider = get_data_provider()
            provider.connect()
            option_chain = provider.fetch_options(symbol=symbol)
            if option_chain is None or option_chain.empty:
                error_msg = (
                    "Échec récupération chaîne d’options via IQFeed: "
                    "données vides ou invalides"
                )
                raise ValueError(error_msg)
            required_cols = [
                "timestamp",
                "strike",
                "option_type",
                "open_interest",
                "volume",
                "gamma",
                "delta",
                "vega",
                "theta",
            ]
            missing_cols = [
                col for col in required_cols if col not in option_chain.columns
            ]
            if missing_cols:
                error_msg = (
                    f"Colonnes manquantes dans la chaîne d’options: "
                    f"{missing_cols}"
                )
                raise ValueError(error_msg)
            latency = time.time() - start_time
            self.log_performance(
                operation="fetch_option_chain_iqfeed",
                latency=latency,
                success=True,
                symbol=symbol,
                num_rows=len(option_chain),
            )
            return option_chain
        except Exception as e:
            error_msg = (
                f"Échec récupération IQFeed: {str(e)}. Utilisation fallback "
                f"dxFeed (temporaire jusqu’à juin 2025)"
            )
            miya_alerts(
                message=error_msg,
                tag="FEATURE_PIPELINE",
                voice_profile="warning",
                priority=3,
            )
            send_alert(message=error_msg, priority=3)
            send_telegram_alert(message=error_msg)
            logger.warning(error_msg)
            try:
                option_chain = fetch_option_chain(symbol=symbol)
                if option_chain is None or option_chain.empty:
                    error_msg = (
                        "Échec récupération dxFeed: données vides ou invalides"
                    )
                    raise ValueError(error_msg)
                required_cols = [
                    "timestamp",
                    "strike",
                    "option_type",
                    "open_interest",
                    "volume",
                    "gamma",
                    "delta",
                    "vega",
                    "theta",
                ]
                missing_cols = [
                    col
                    for col in required_cols
                    if col not in option_chain.columns
                ]
                if missing_cols:
                    error_msg = (
                        f"Colonnes manquantes dans la chaîne d’options "
                        f"dxFeed: {missing_cols}"
                    )
                    raise ValueError(error_msg)
                latency = time.time() - start_time
                self.log_performance(
                    operation="fetch_option_chain_dxfeed_fallback",
                    latency=latency,
                    success=True,
                    symbol=symbol,
                    num_rows=len(option_chain),
                )
                return option_chain
            except Exception as e_fallback:
                error_msg = f"Échec récupération dxFeed fallback: {str(e_fallback)}"
                self.log_performance(
                    operation="fetch_option_chain_dxfeed_fallback",
                    latency=time.time() - start_time,
                    success=False,
                    error=str(e_fallback),
                )
                miya_alerts(
                    message=error_msg,
                    tag="FEATURE_PIPELINE",
                    voice_profile="urgent",
                    priority=4,
                )
                send_alert(message=error_msg, priority=4)
                send_telegram_alert(message=error_msg)
                logger.error(error_msg)
                return pd.DataFrame()

    def log_performance(
        self,
        operation: str,
        latency: float,
        success: bool,
        error: str = None,
        **kwargs,
    ) -> None:
        """Journalise les performances des opérations critiques."""
        start_time = time.time()
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
            cpu_percent = psutil.cpu_percent()
            if memory_usage > 1024:
                alert_msg = (
                    f"ALERTE: Utilisation mémoire élevée ({memory_usage:.2f} MB)"
                )
                miya_alerts(
                    message=alert_msg,
                    tag="FEATURE_PIPELINE",
                    voice_profile="urgent",
                    priority=5,
                )
                send_alert(message=alert_msg, priority=5)
                send_telegram_alert(message=alert_msg)
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "latency": latency,
                "success": success,
                "error": error,
                "memory_usage_mb": memory_usage,
                "cpu_percent": cpu_percent,
                **kwargs,
            }
            self.log_buffer.append(log_entry)
            if len(self.log_buffer) >= self.buffer_size:
                log_df = pd.DataFrame(self.log_buffer)
                os.makedirs(os.path.dirname(CSV_LOG_PATH), exist_ok=True)

                def write_log():
                    log_df.to_csv(
                        CSV_LOG_PATH,
                        mode="a" if os.path.exists(CSV_LOG_PATH) else "w",
                        header=not os.path.exists(CSV_LOG_PATH),
                        index=False,
                        encoding="utf-8",
                    )

                self.with_retries(write_log)
                self.log_buffer = []
            logger.info(
                f"Performance journalisée pour {operation}. CPU: {cpu_percent}%"
            )
            latency = time.time() - start_time
            self.log_performance(
                operation="log_performance",
                latency=latency,
                success=True,
            )
        except Exception as e:
            error_msg = f"Erreur journalisation performance: {str(e)}"
            miya_alerts(
                message=error_msg,
                tag="FEATURE_PIPELINE",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(message=error_msg, priority=3)
            send_telegram_alert(message=error_msg)
            logger.error(error_msg)
            self.log_performance(
                operation="log_performance",
                latency=time.time() - start_time,
                success=False,
                error=str(e),
            )

    def save_snapshot(
        self, snapshot_type: str, data: Dict, compress: bool = False
    ) -> None:
        """Sauvegarde un instantané des résultats avec option de compression 
        gzip."""
        start_time = time.time()
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = {
                "timestamp": timestamp,
                "type": snapshot_type,
                "data": data,
            }
            snapshot_path = os.path.join(
                SNAPSHOT_DIR,
                f"snapshot_{snapshot_type}_{timestamp}.json",
            )
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)

            def write_snapshot():
                if compress:
                    with gzip.open(
                        f"{snapshot_path}.gz", "wt", encoding="utf-8"
                    ) as f:
                        json.dump(snapshot, f, indent=4)
                else:
                    with open(snapshot_path, "w", encoding="utf-8") as f:
                        json.dump(snapshot, f, indent=4)

            self.with_retries(write_snapshot)
            save_path = f"{snapshot_path}.gz" if compress else snapshot_path
            file_size = os.path.getsize(save_path) / 1024 / 1024
            if file_size > 1.0:
                alert_msg = f"Snapshot size {file_size:.2f} MB exceeds 1 MB"
                miya_alerts(
                    message=alert_msg,
                    tag="FEATURE_PIPELINE",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(message=alert_msg, priority=3)
                send_telegram_alert(message=alert_msg)
            latency = time.time() - start_time
            self.log_performance(
                operation="save_snapshot",
                latency=latency,
                success=True,
                snapshot_type=snapshot_type,
                file_size_mb=file_size,
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot: {str(e)}"
            logger.error(error_msg)
            send_alert(message=error_msg, priority=3)
            send_telegram_alert(message=error_msg)
            self.log_performance(
                operation="save_snapshot",
                latency=time.time() - start_time,
                success=False,
                error=str(e),
            )

    def save_dashboard_status(
        self,
        status: dict,
        status_file: str = DASHBOARD_PATH,
        compress: bool = False,
    ):
        """Sauvegarde le statut du tableau de bord."""
        start_time = time.time()
        try:
            os.makedirs(os.path.dirname(status_file), exist_ok=True)

            def write_status():
                if compress:
                    with gzip.open(
                        f"{status_file}.gz", "wt", encoding="utf-8"
                    ) as f:
                        json.dump(status, f, indent=4)
                else:
                    with open(status_file, "w", encoding="utf-8") as f:
                        json.dump(status, f, indent=4)

            self.with_retries(write_status)
            save_path = f"{status_file}.gz" if compress else status_file
            latency = time.time() - start_time
            self.log_performance(
                operation="save_dashboard_status",
                latency=latency,
                success=True,
            )
            logger.info(f"Dashboard sauvegardé: {save_path}")
            send_alert(message=f"Dashboard sauvegardé: {save_path}", priority=1)
            send_telegram_alert(message=f"Dashboard sauvegardé: {save_path}")
        except Exception as e:
            error_msg = f"Erreur sauvegarde dashboard: {str(e)}"
            logger.error(error_msg)
            send_alert(message=error_msg, priority=3)
            send_telegram_alert(message=error_msg)
            self.log_performance(
                operation="save_dashboard_status",
                latency=time.time() - start_time,
                success=False,
                error=str(e),
            )

    def plot_features(self, data: pd.DataFrame, timestamp: str) -> None:
        """Génère des visualisations des métriques."""
        start_time = time.time()
        try:
            timestamp_safe = timestamp.replace(":", "-")
            plt.figure(figsize=(10, 6))
            key_features = [
                "net_gamma",
                "vol_trigger",
                "oi_sweep_count",
                "news_sentiment_momentum",
                "gold_correl",
            ]
            for feature in key_features:
                if feature in data.columns:
                    plt.plot(
                        data["timestamp"],
                        data[feature],
                        label=feature.replace("_", " ").title(),
                    )
            plt.title(f"Feature Pipeline Metrics - {timestamp}")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            output_path = os.path.join(
                BASE_DIR,
                "data",
                "figures",
                "feature_pipeline",
                f"features_{timestamp_safe}.png",
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            plt.close()
            latency = time.time() - start_time
            valid_features = [f for f in key_features if f in data.columns]
            self.log_performance(
                operation="plot_features",
                latency=latency,
                success=True,
                num_features=len(valid_features),
            )
        except Exception as e:
            error_msg = f"Erreur génération visualisation: {str(e)}"
            logger.error(error_msg)
            send_alert(message=error_msg, priority=3)
            send_telegram_alert(message=error_msg)
            self.log_performance(
                operation="plot_features",
                latency=time.time() - start_time,
                success=False,
                error=str(e),
            )

    def impute_value(
        self,
        value: float,
        series: pd.Series = None,
        method: str = "interpolate",
        feature_name: str = None,
    ) -> float:
        """Impute les valeurs manquantes."""
        start_time = time.time()
        try:
            critical_features = [
                "rsi_14",
                "atr_14",
                "vwap",
                "delta_volume",
                "obi_score",
            ]
            option_features = {
                "iv_rank_30d": 50,
                "call_wall": None,
                "put_wall": None,
                "zero_gamma": None,
                "dealer_position_bias": 0,
                "gex_change": 0,
                "key_strikes_1": None,
                "key_strikes_2": None,
                "key_strikes_3": None,
                "key_strikes_4": None,
                "key_strikes_5": None,
                "max_pain_strike": None,
                "net_gamma": 0,
                "dealer_zones_count": 0,
                "vol_trigger": 0,
                "ref_px": None,
                "data_release": 0,
                "oi_sweep_count": 0,
                "iv_acceleration": 0,
                "theta_exposure": 0,
                "vomma_exposure": 0,
                "speed_exposure": 0,
                "ultima_exposure": 0,
            }
            if pd.isna(value):
                if feature_name in critical_features:
                    return value
                if feature_name in option_features:
                    if option_features[feature_name] is None and series is not None:
                        median = series.median()
                        return median if not pd.isna(median) else 0.0
                    return option_features[feature_name]
                if series is not None:
                    if method == "interpolate":
                        interpolated = series.interpolate(
                            method="linear", limit_direction="both"
                        )
                        return (
                            interpolated.iloc[-1]
                            if not pd.isna(interpolated).any()
                            else series.median()
                        )
                    elif method == "mean":
                        mean = series.mean()
                        return mean if not pd.isna(mean) else 0.0
                return 0.0
            latency = time.time() - start_time
            self.log_performance(
                operation="impute_value",
                latency=latency,
                success=True,
                feature_name=feature_name,
            )
            return value
        except Exception as e:
            error_msg = f"Erreur imputation valeur: {str(e)}"
            logger.error(error_msg)
            self.log_performance(
                operation="impute_value",
                latency=time.time() - start_time,
                success=False,
                error=str(e),
            )
            return 0.0

    def integrate_option_levels(
        self, data: pd.DataFrame, option_chain_path: str = OPTION_CHAIN_PATH
    ) -> pd.DataFrame:
        """Intègre les niveaux d’options, incluant les nouvelles métriques."""
        start_time = time.time()
        try:
            option_chain = self.fetch_option_chain_iqfeed(symbol="ES")
            if option_chain.empty:
                raise ValueError("Chaîne d’options vide")
            timestamp = data["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S")
            levels = recalculate_levels(
                option_chain=option_chain,
                timestamp=timestamp,
                mode="snapshot",
            )
            spotgamma_metrics = (
                pd.read_csv(SPOTGAMMA_METRICS_PATH)
                if os.path.exists(SPOTGAMMA_METRICS_PATH)
                else pd.DataFrame()
            )
            option_cols = [
                "call_wall",
                "put_wall",
                "zero_gamma",
                "dealer_position_bias",
                "iv_rank_30d",
                "key_strikes_1",
                "key_strikes_2",
                "key_strikes_3",
                "key_strikes_4",
                "key_strikes_5",
                "max_pain_strike",
                "net_gamma",
                "dealer_zones_count",
                "vol_trigger",
                "ref_px",
                "data_release",
                "vomma_exposure",
                "speed_exposure",
                "ultima_exposure",
            ]
            for col in option_cols:
                if col in levels:
                    data[col] = levels.get(col, 0.0)
                elif (
                    not spotgamma_metrics.empty
                    and col in spotgamma_metrics.columns
                ):
                    data[col] = (
                        spotgamma_metrics[col].iloc[-1]
                        if len(spotgamma_metrics) > 0
                        else 0.0
                    )
                else:
                    data[col] = 0.0
                if col == "iv_rank_30d":
                    data[col] = data[col].clip(0, 100).fillna(50)
                elif col in [
                    "call_wall",
                    "put_wall",
                    "zero_gamma",
                    "key_strikes_1",
                    "key_strikes_2",
                    "key_strikes_3",
                    "key_strikes_4",
                    "key_strikes_5",
                    "max_pain_strike",
                    "ref_px",
                ]:
                    data[col] = data[col].clip(lower=0).fillna(
                        data["close"].median()
                    )
                elif col in [
                    "dealer_position_bias",
                    "net_gamma",
                    "vol_trigger",
                    "vomma_exposure",
                    "speed_exposure",
                    "ultima_exposure",
                ]:
                    data[col] = data[col].clip(-1, 1).fillna(0)
                elif col == "dealer_zones_count":
                    data[col] = data[col].clip(0, 10).fillna(0)
                elif col == "data_release":
                    data[col] = data[col].clip(0, 1).fillna(0)
            if "gex" in data.columns:
                gex_valid = data["gex"].replace(
                    [np.inf, -np.inf], np.nan
                ).dropna()
                data["gex_change"] = (
                    data["gex"].pct_change().clip(-1, 1).fillna(0)
                    if not gex_valid.empty
                    else 0.0
                )
            else:
                data["gex_change"] = 0.0
            latency = time.time() - start_time
            new_metrics = [
                col
                for col in data.columns
                if col.startswith("key_strikes")
                or col
                in [
                    "max_pain_strike",
                    "net_gamma",
                    "zero_gamma",
                    "dealer_zones_count",
                    "vol_trigger",
                    "ref_px",
                    "data_release",
                    "vomma_exposure",
                    "speed_exposure",
                    "ultima_exposure",
                ]
            ]
            self.log_performance(
                operation="integrate_option_levels",
                latency=latency,
                success=True,
                num_rows=len(data),
                new_metrics=len(new_metrics),
            )
            self.save_snapshot(
                snapshot_type="integrate_option_levels",
                data={
                    "num_rows": len(data),
                    "option_cols": data.columns.tolist(),
                },
                compress=False,
            )
            return data
        except Exception as e:
            error_msg = f"Erreur dans integrate_option_levels: {str(e)}"
            logger.error(error_msg)
            send_alert(message=error_msg, priority=4)
            send_telegram_alert(message=error_msg)
            self.log_performance(
                operation="integrate_option_levels",
                latency=time.time() - start_time,
                success=False,
                error=str(e),
            )
            return data

    def validate_shap_features(self) -> bool:
        """Valide quotidiennement les top 150 SHAP features."""
        start_time = time.time()
        try:
            shap_file = FEATURE_IMPORTANCE_PATH
            if not os.path.exists(shap_file):
                alert_msg = "Fichier SHAP manquant"
                miya_alerts(
                    message=alert_msg,
                    tag="FEATURE_PIPELINE",
                    voice_profile="urgent",
                    priority=4,
                )
                send_alert(message=alert_msg, priority=4)
                send_telegram_alert(message=alert_msg)
                logger.error(alert_msg)
                self.log_performance(
                    operation="validate_shap_features",
                    latency=time.time() - start_time,
                    success=False,
                    error="Fichier SHAP manquant",
                )
                return False
            shap_df = pd.read_csv(shap_file)
            if len(shap_df) < 150:
                alert_msg = (
                    f"Nombre insuffisant de SHAP features: "
                    f"{len(shap_df)} < 150"
                )
                miya_alerts(
                    message=alert_msg,
                    tag="FEATURE_PIPELINE",
                    voice_profile="urgent",
                    priority=4,
                )
                send_alert(message=alert_msg, priority=4)
                send_telegram_alert(message=alert_msg)
                logger.error(alert_msg)
                self.log_performance(
                    operation="validate_shap_features",
                    latency=time.time() - start_time,
                    success=False,
                    error=alert_msg,
                )
                return False
            miya_speak(
                message="SHAP features validées",
                tag="FEATURE_PIPELINE",
                voice_profile="calm",
                priority=1,
            )
            send_alert(message="SHAP features validées", priority=1)
            send_telegram_alert(message="SHAP features validées")
            logger.info("SHAP features validées")
            latency = time.time() - start_time
            self.log_performance(
                operation="validate_shap_features",
                latency=latency,
                success=True,
            )
            return True
        except Exception as e:
            error_msg = f"Erreur validation SHAP features: {str(e)}"
            logger.error(error_msg)
            send_alert(message=error_msg, priority=4)
            send_telegram_alert(message=error_msg)
            self.log_performance(
                operation="validate_shap_features",
                latency=time.time() - start_time,
                success=False,
                error=str(e),
            )
            return False

    def calculate_shap_features(
        self, data: pd.DataFrame, regime: str
    ) -> pd.DataFrame:
        """Calcule l’importance des features avec SHAP (top 150 clés) et 
        pondère par régime."""
        start_time = time.time()
        try:
            cache_hash = hashlib.sha256(data.to_json().encode()).hexdigest()
            cache_path = os.path.join(
                SHAP_CACHE_DIR,
                f"shap_{regime}_{cache_hash}.csv",
            )
            if os.path.exists(cache_path):
                shap_df = pd.read_csv(cache_path)
                miya_speak(
                    message=f"SHAP chargé depuis cache: {cache_path}",
                    tag="FEATURE_PIPELINE",
                    voice_profile="calm",
                    priority=1,
                )
                logger.info(f"SHAP chargé depuis cache: {cache_path}")
                send_alert(
                    message=f"SHAP chargé depuis cache: {cache_path}",
                    priority=1,
                )
                send_telegram_alert(
                    message=f"SHAP chargé depuis cache: {cache_path}"
                )
                latency = time.time() - start_time
                self.log_performance(
                    operation="calculate_shap_features",
                    latency=latency,
                    success=True,
                )
                return shap_df
            features = data.select_dtypes(include=[np.number]).columns
            if not features.size:
                error_msg = "Aucune feature numérique disponible pour SHAP"
                raise ValueError(error_msg)
            X = data[features].fillna(0)
            from sklearn.ensemble import GradientBoostingClassifier

            y = (data.get("neural_regime", regime) == regime).astype(int)
            model = GradientBoostingClassifier().fit(X, y)
            explainer = shap.TreeExplainer(model)
            shap_values = Parallel(n_jobs=-1)(
                delayed(explainer.shap_values)(
                    X[feature].values.reshape(-1, 1)
                )
                for feature in features
            )
            importance = pd.DataFrame(
                {
                    "feature": features,
                    "importance": [
                        np.abs(shap_val).mean() for shap_val in shap_values
                    ],
                    "regime": regime,
                }
            ).sort_values("importance", ascending=False)
            new_metrics = [
                "key_strikes_1",
                "key_strikes_2",
                "key_strikes_3",
                "key_strikes_4",
                "key_strikes_5",
                "max_pain_strike",
                "net_gamma",
                "zero_gamma",
                "dealer_zones_count",
                "vol_trigger",
                "ref_px",
                "data_release",
                "oi_sweep_count",
                "iv_acceleration",
                "theta_exposure",
                "news_volume_spike",
                "gold_correl",
                "iv_atm",
                "option_skew",
                "gex_slope",
                "news_sentiment_momentum",
                "month_of_year_sin",
                "latent_vol_regime_vec_3",
                "topic_vector_news_3",
                "vomma_exposure",
                "speed_exposure",
                "ultima_exposure",
                "spoofing_score",
                "volume_anomaly",
            ]
            for metric in new_metrics:
                if metric in importance["feature"].values:
                    importance.loc[
                        importance["feature"] == metric, "importance"
                    ] *= 1.2
            weights = {
                "range": {
                    "atr_14": 1.5,
                    "rsi_14": 1.3,
                    "obi_score": 1.2,
                    "iv_atm": 1.3,
                    "news_sentiment_momentum": 1.3,
                },
                "trend": {
                    "adx_14": 1.4,
                    "vwap": 1.3,
                    "delta_volume": 1.2,
                    "gex_slope": 1.3,
                    "gold_correl": 1.2,
                },
                "defensive": {
                    "vix_es_correlation": 1.5,
                    "net_gamma": 1.4,
                    "vol_trigger": 1.3,
                    "option_skew": 1.3,
                    "latent_vol_regime_vec_3": 1.3,
                },
            }
            regime_weights = weights.get(regime, {})
            for feature, weight in regime_weights.items():
                if feature in importance["feature"].values:
                    importance.loc[
                        importance["feature"] == feature, "importance"
                    ] *= weight
            importance = importance.head(
                PERFORMANCE_THRESHOLDS["max_key_shap_features"]
            )

            def write_cache():
                importance.to_csv(cache_path, index=False, encoding="utf-8")
                importance.to_csv(
                    FEATURE_IMPORTANCE_PATH,
                    index=False,
                    encoding="utf-8",
                )

            self.with_retries(write_cache)
            latency = time.time() - start_time
            self.log_performance(
                operation="calculate_shap_features",
                latency=latency,
                success=True,
                num_features=len(importance),
            )
            self.save_snapshot(
                snapshot_type="calculate_shap_features",
                data={
                    "num_features": len(importance),
                    "regime": regime,
                    "features": importance["feature"].tolist(),
                },
                compress=False,
            )
            return importance
        except Exception as e:
            error_msg = f"Erreur dans calculate_shap_features: {str(e)}"
            logger.error(error_msg)
            send_alert(message=error_msg, priority=4)
            send_telegram_alert(message=error_msg)
            self.log_performance(
                operation="calculate_shap_features",
                latency=time.time() - start_time,
                success=False,
                error=str(e),
            )
            shap_features = self.load_shap_fallback()
            return pd.DataFrame(
                {
                    "feature": shap_features,
                    "importance": 1.0,
                    "regime": regime,
                }
            ).head(PERFORMANCE_THRESHOLDS["max_key_shap_features"])

    def save_clusters(self, data: pd.DataFrame, regime: str) -> None:
        """Stocke les patterns clusterisés dans market_memory.db."""
        start_time = time.time()
        try:
            features = data.select_dtypes(include=[np.number]).columns
            if not features.size:
                error_msg = (
                    "Aucune feature numérique disponible pour le clustering"
                )
                raise ValueError(error_msg)
            X = data[features].fillna(0)
            clusters = KMeans(n_clusters=10).fit_predict(X)
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            for i, row in data.iterrows():
                cursor.execute(
                    """
                    INSERT INTO clusters 
                    (cluster_id, event_type, features, timestamp) 
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        clusters[i],
                        regime,
                        json.dumps(row[features].to_dict()),
                        str(row.get("timestamp", datetime.now())),
                    ),
                )
            conn.commit()
            conn.close()
            latency = time.time() - start_time
            self.log_performance(
                operation="save_clusters",
                latency=latency,
                success=True,
                num_rows=len(data),
            )
            self.save_snapshot(
                snapshot_type="save_clusters",
                data={
                    "num_rows": len(data),
                    "regime": regime,
                },
                compress=False,
            )
        except Exception as e:
            error_msg = f"Erreur dans save_clusters: {str(e)}"
            logger.error(error_msg)
            send_alert(message=error_msg, priority=4)
            send_telegram_alert(message=error_msg)
            self.log_performance(
                operation="save_clusters",
                latency=time.time() - start_time,
                success=False,
                error=str(e),
            )

    def calculate_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcule les features techniques."""
        start_time = time.time()
        try:
            data["rsi_14"] = ta.momentum.RSIIndicator(
                close=data["close"],
                window=14,
            ).rsi()
            data["atr_14"] = ta.volatility.AverageTrueRange(
                high=data["high"],
                low=data["low"],
                close=data["close"],
                window=14,
            ).average_true_range()
            data["adx_14"] = ta.trend.ADXIndicator(
                high=data["high"],
                low=data["low"],
                close=data["close"],
                window=14,
            ).adx()
            data["vwap"] = (
                (data["close"] * data["volume"]).cumsum()
                / data["volume"].cumsum()
            )
            data["delta_volume"] = data["volume"].diff()
            latency = time.time() - start_time
            self.log_performance(
                operation="calculate_technical_features",
                latency=latency,
                success=True,
                num_features=5,
            )
            return data
        except Exception as e:
            error_msg = f"Erreur calcul technical features: {str(e)}"
            logger.error(error_msg)
            self.log_performance(
                operation="calculate_technical_features",
                latency=time.time() - start_time,
                success=False,
                error=str(e),
            )
            return data

    def compute_features(
        self,
        data: pd.DataFrame,
        option_chain: pd.DataFrame = None,
        regime: str = None,
    ) -> pd.DataFrame:
        """Calcule les 350 features, incluant 59 nouvelles, trade_success_prob, 
        11 neurales, 20 dérivées."""
        start_time = time.time()
        try:
            required_cols = [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
            missing_cols = [
                col for col in required_cols if col not in data.columns
            ]
            if missing_cols:
                error_msg = f"Colonnes manquantes dans les données: {missing_cols}"
                raise ValueError(error_msg)
            if data["timestamp"].isna().any():
                data["timestamp"] = pd.to_datetime(
                    data["timestamp"], errors="coerce"
                ).fillna(pd.Timestamp.now())

            for col in data.columns:
                if col != "timestamp":
                    data[col] = (
                        pd.to_numeric(data[col], errors="coerce")
                        .interpolate(method="linear")
                        .fillna(
                            data[col].median()
                            if not data[col].isna().all()
                            else 0.0
                        )
                    )

            valid_features = len(
                [
                    col
                    for col in data.columns
                    if not data[col].isna().all() and col != "timestamp"
                ]
            )
            confidence_drop_rate = 1.0 - min(
                valid_features / PERFORMANCE_THRESHOLDS["min_features"],
                1.0,
            )
            if confidence_drop_rate > 0.5:
                alert_msg = (
                    f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} "
                    f"({valid_features}/"
                    f"{PERFORMANCE_THRESHOLDS['min_features']} features valides)"
                )
                miya_alerts(
                    message=alert_msg,
                    tag="FEATURE_PIPELINE",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(message=alert_msg, priority=3)
                send_telegram_alert(message=alert_msg)
                logger.warning(alert_msg)

            output_csv = os.path.join(
                BASE_DIR,
                "data",
                "logs",
                "features_audit_raw.csv",
            )
            audit_features(data=data, output_csv=output_csv)

            data = self.calculate_technical_features(data)

            data = encode_time_context(data=data)
            data = calculate_orderflow_indicators(data=data)
            data = apply_pca_orderflow(data=data)
            data = extract_volume_profile(data=data)
            data = calculate_smart_scores(data=data)
            data = calculate_volatility_metrics(data=data)
            data = calculate_options_features(data=data)
            data = calculate_option_metrics(data=data)
            data = calculate_advanced_features(data=data)
            data = calculate_meta_features(data=data)
            data = calculate_cross_market_signals(data=data)
            data = calculate_contextual_encodings(data=data)

            data = self.integrate_option_levels(
                data=data,
                option_chain_path=OPTION_CHAIN_PATH,
            )

            data["atr_dynamic"] = self.calculate_atr_dynamic(data)
            data["orderflow_imbalance"] = self.calculate_orderflow_imbalance(data)
            data["slippage_estimate"] = self.calculate_slippage_estimate(data)
            data["bid_ask_imbalance"] = self.calculate_bid_ask_imbalance(data)
            data["trade_aggressiveness"] = self.calculate_trade_aggressiveness(
                data
            )
            data["iv_skew"] = self.calculate_iv_skew(data)
            data["iv_term_structure"] = self.calculate_iv_term_structure(data)
            data["regime_hmm"] = self.regime_detector.detect_regime(data)

            new_features = [
                "atr_dynamic",
                "orderflow_imbalance",
                "slippage_estimate",
                "bid_ask_imbalance",
                "trade_aggressiveness",
                "iv_skew",
                "iv_term_structure",
                "regime_hmm",
            ]
            validation_result = validate_features(data[new_features])
            if not validation_result["valid"]:
                alert_msg = (
                    f"Validation des nouvelles features échouée: "
                    f"{validation_result['errors']}"
                )
                logger.warning(alert_msg)
                send_alert(message=alert_msg, priority=3)
                send_telegram_alert(message=alert_msg)

            self.data_lake.store_features(
                data=data[new_features],
                market=self.market,
            )

            if (
                self.context_aware_filter
                and os.path.exists(NEWS_DATA_PATH)
                and os.path.exists(FUTURES_DATA_PATH)
            ):
                news_data = pd.read_csv(NEWS_DATA_PATH)
                calendar_data = pd.DataFrame(
                    {
                        "timestamp": data["timestamp"],
                        "severity": np.random.uniform(0, 1, len(data)),
                        "weight": np.random.uniform(0, 1, len(data)),
                    }
                )
                futures_data = pd.read_csv(FUTURES_DATA_PATH)
                expiry_dates = pd.Series(
                    pd.date_range("2025-05-15", periods=len(data), freq="1d")
                )
                macro_events_path = os.path.join(
                    BASE_DIR,
                    "data",
                    "events",
                    "macro_events.csv",
                )
                volatility_history_path = os.path.join(
                    BASE_DIR,
                    "data",
                    "events",
                    "event_volatility_history.csv",
                )
                if os.path.exists(macro_events_path) and os.path.exists(
                    volatility_history_path
                ):
                    data = self.context_aware_filter.compute_contextual_metrics(
                        data=data,
                        news_data=news_data,
                        calendar_data=calendar_data,
                        futures_data=futures_data,
                        expiry_dates=expiry_dates,
                        macro_events_path=macro_events_path,
                        volatility_history_path=volatility_history_path,
                    )
                else:
                    logger.warning(
                        "Fichiers macro_events.csv ou "
                        "event_volatility_history.csv manquants"
                    )
                    send_alert(
                        message=(
                            "Fichiers macro_events.csv ou "
                            "event_volatility_history.csv manquants"
                        ),
                        priority=3,
                    )
                    send_telegram_alert(
                        message=(
                            "Fichiers macro_events.csv ou "
                            "event_volatility_history.csv manquants"
                        )
                    )

            if self.cross_asset_processor:
                es_data = data[["timestamp", "close", "volume"]].copy()
                spy_data = pd.DataFrame(
                    {
                        "timestamp": data["timestamp"],
                        "volume": np.random.randint(100, 1000, len(data)),
                    }
                )
                asset_data = pd.DataFrame(
                    {
                        "timestamp": data["timestamp"],
                        "close": data["close"],
                        "gold_price": np.random.normal(2000, 5, len(data)),
                        "oil_price": np.random.normal(80, 2, len(data)),
                        "btc_price": np.random.normal(60000, 1000, len(data)),
                    }
                )
                bond_data = pd.DataFrame(
                    {
                        "timestamp": data["timestamp"],
                        "yield_10y": np.random.normal(3.5, 0.1, len(data)),
                        "yield_2y": np.random.normal(2.5, 0.1, len(data)),
                    }
                )
                data = self.cross_asset_processor.compute_cross_asset_features(
                    es_data=es_data,
                    spy_data=spy_data,
                    asset_data=asset_data,
                    bond_data=bond_data,
                )

            if self.latent_factor_generator:
                news_data = (
                    pd.read_csv(NEWS_DATA_PATH)
                    if os.path.exists(NEWS_DATA_PATH)
                    else pd.DataFrame(
                        {
                            "timestamp": data["timestamp"],
                            "topic_score_1": np.random.normal(
                                0, 1, len(data)
                            ),
                            "topic_score_2": np.random.normal(
                                0, 1, len(data)
                            ),
                            "topic_score_3": np.random.normal(
                                0, 1, len(data)
                            ),
                            "topic_score_4": np.random.normal(
                                0, 1, len(data)
                            ),
                        }
                    )
                )
                ohlc_data = data[
                    ["timestamp", "open", "high", "low", "close"]
                ].copy()
                options_data = pd.DataFrame(
                    {
                        "timestamp": data["timestamp"],
                        "iv_atm": np.random.normal(0.15, 0.02, len(data)),
                        "option_skew": np.random.normal(0.1, 0.01, len(data)),
                        "vix_term_1m": np.random.normal(20, 1, len(data)),
                    }
                )
                data = self.latent_factor_generator.compute_latent_features(
                    data=data,
                    news_data=news_data,
                    ohlc_data=ohlc_data,
                    options_data=options_data,
                )

            data = calculate_order_flow_features(
                data=data,
                option_chain=option_chain,
            )

            data["short_dated_iv_slope"] = calculate_option_metrics(
                data=data,
                option_chain=option_chain,
                metric="iv_slope",
                days=[7, 30],
            )
            data["theta_exposure"] = calculate_option_metrics(
                data=data,
                option_chain=option_chain,
                metric="greek_exposure",
                greek="theta",
            )
            data["vomma_exposure"] = calculate_option_metrics(
                data=data,
                option_chain=option_chain,
                metric="greek_exposure",
                greek="vomma",
            )
            data["speed_exposure"] = calculate_option_metrics(
                data=data,
                option_chain=option_chain,
                metric="greek_exposure",
                greek="speed",
            )
            data["ultima_exposure"] = calculate_option_metrics(
                data=data,
                option_chain=option_chain,
                metric="greek_exposure",
                greek="ultima",
            )
            data["iv_acceleration"] = calculate_option_metrics(
                data=data,
                option_chain=option_chain,
                metric="iv_acceleration",
            )
            data["oi_sweep_count"] = calculate_option_metrics(
                data=data,
                option_chain=option_chain,
                metric="oi_sweep_count",
            )
            data["oi_velocity"] = calculate_option_metrics(
                data=data,
                option_chain=option_chain,
                metric="oi_velocity",
            )
            data["gamma_bucket_exposure"] = calculate_option_metrics(
                data=data,
                option_chain=option_chain,
                metric="gamma_bucket_exposure",
            )
            data["spot_gamma_corridor"] = calculate_option_metrics(
                data=data,
                option_chain=option_chain,
                metric="spot_gamma_corridor",
            )
            data["oi_peak_times_call"] = calculate_option_metrics(
                data=data,
                option_chain=option_chain,
                metric="oi_peak_times",
                option_type="call",
            )
            data["oi_peak_times_put"] = calculate_option_metrics(
                data=data,
                option_chain=option_chain,
                metric="oi_peak_times",
                option_type="put",
            )
            data["net_gamma_flow_ratio"] = calculate_option_metrics(
                data=data,
                option_chain=option_chain,
                metric="net_gamma_flow_ratio",
            )
            data["theta_vega_ratio"] = calculate_option_metrics(
                data=data,
                option_chain=option_chain,
                metric="theta_vega_ratio",
            )
            data["call_put_oi_ratio"] = calculate_option_metrics(
                data=data,
                option_chain=option_chain,
                metric="call_put_oi_ratio",
            )
            data["call_put_volume_ratio"] = calculate_option_metrics(
                data=data,
                option_chain=option_chain,
                metric="call_put_volume_ratio",
            )
            data["oi_peak_call_near"] = calculate_option_metrics(
                data=data,
                option_chain=option_chain,
                metric="oi_peak_call_near",
            )
            data["oi_peak_put_near"] = calculate_option_metrics(
                data=data,
                option_chain=option_chain,
                metric="oi_peak_put_near",
            )

            data["realized_vol_5m"] = calculate_volatility_metrics(
                data=data,
                metric="realized_vol",
                window="5min",
            )
            data["realized_vol_15m"] = calculate_volatility_metrics(
                data=data,
                metric="realized_vol",
                window="15min",
            )
            data["realized_vol_30m"] = calculate_volatility_metrics(
                data=data,
                metric="realized_vol",
                window="30min",
            )
            data["volatility_breakout_signal"] = calculate_volatility_metrics(
                data=data,
                metric="volatility_breakout_signal",
            )
            data["vix_term_1m"] = calculate_volatility_metrics(
                data=data,
                metric="vix_term",
                maturity="1m",
            )
            data["vix_term_3m"] = calculate_volatility_metrics(
                data=data,
                metric="vix_term",
                maturity="3m",
            )
            data["vix_term_6m"] = calculate_volatility_metrics(
                data=data,
                metric="vix_term",
                maturity="6m",
            )
            data["vix_term_1y"] = calculate_volatility_metrics(
                data=data,
                metric="vix_term",
                maturity="1y",
            )
            data["vix_term_structure_slope"] = calculate_volatility_metrics(
                data=data,
                metric="vix_term_structure_slope",
            )
            data["vix_term_structure_concavity"] = calculate_volatility_metrics(
                data=data,
                metric="vix_term_structure_concavity",
            )
            data["volatility_delta"] = calculate_volatility_metrics(
                data=data,
                metric="volatility_delta",
            )

            data["oi_peak_times_vol"] = (
                data["oi_peak_call_near"] * data["volume"]
            )
            data["gamma_wall_spread"] = data["net_gamma"] * (
                data["call_wall"] - data["put_wall"]
            )
            data["net_gamma_flow_ratio"] = data["net_gamma"] / (
                data["volume"] + 1e-6
            )
            data["theta_vega_ratio"] = data["theta_exposure"] / (
                data["vega"] + 1e-6
            )

            data["market_momentum_score"] = calculate_smart_scores(
                data=data,
                metric="market_momentum_score",
            )
            data["trend_strength_score"] = calculate_smart_scores(
                data=data,
                metric="trend_strength_score",
            )
            data["volatility_score"] = calculate_smart_scores(
                data=data,
                metric="volatility_score",
            )
            data["option_activity_score"] = calculate_smart_scores(
                data=data,
                metric="option_activity_score",
            )
            data["cross_market_score"] = calculate_smart_scores(
                data=data,
                metric="cross_market_score",
            )
            data["market_liquidity_score"] = calculate_smart_scores(
                data=data,
                metric="market_liquidity_score",
            )
            data["gamma_pressure_score"] = calculate_smart_scores(
                data=data,
                metric="gamma_pressure_score",
            )

            data["trade_volatility_exposure"] = calculate_advanced_features(
                data=data,
                metric="trade_volatility_exposure",
            )
            data["overtrade_risk_score"] = calculate_advanced_features(
                data=data,
                metric="overtrade_risk_score",
            )
            data["margin_risk_score"] = calculate_advanced_features(
                data=data,
                metric="margin_risk_score",
            )
            data["position_risk_score"] = calculate_advanced_features(
                data=data,
                metric="position_risk_score",
            )
            data["drawdown_risk_score"] = calculate_advanced_features(
                data=data,
                metric="drawdown_risk_score",
            )
            data["portfolio_risk_score"] = calculate_advanced_features(
                data=data,
                metric="portfolio_risk_score",
            )

            data["iv_atm_oi_ratio"] = (
                data["iv_atm"] * data["call_put_oi_ratio"]
            )
            data["gex_slope_vanna_skew"] = (
                data["gex_slope"] * data["vanna_skew"]
            )
            data["option_skew_volume"] = data["option_skew"] * data["volume"]
            data["iv_atm_rsi_14"] = data["iv_atm"] * data["rsi_14"]
            data["gex_slope_net_gamma"] = (
                data["gex_slope"] * data["net_gamma"]
            )
            data["option_skew_atr_14"] = data["option_skew"] * data["atr_14"]
            data["iv_atm_vwap"] = data["iv_atm"] * data["vwap"]
            data["gex_slope_oi_sweep"] = (
                data["gex_slope"] * data["oi_sweep_count"]
            )
            data["option_skew_theta_exposure"] = (
                data["option_skew"] * data["theta_exposure"]
            )
            data["iv_atm_news_spike"] = (
                data["iv_atm"] * data["news_volume_spike"]
            )
            data["iv_atm_trend"] = (
                data["iv_atm"]
                .diff()
                .rolling(window=5, min_periods=1)
                .mean()
                .fillna(0)
            )
            data["option_skew_momentum"] = (
                data["option_skew"]
                .diff()
                .rolling(window=5, min_periods=1)
                .mean()
                .fillna(0)
            )
            data["gex_slope_acceleration"] = (
                data["gex_slope"]
                .diff()
                .rolling(window=5, min_periods=1)
                .mean()
                .fillna(0)
            )
            data["vanna_skew_ratio"] = data["vanna_skew"] / (
                data["vanna_exposure"] + 1e-6
            )
            data["micro_vol_acceleration"] = (
                data["microstructure_volatility"]
                .diff()
                .rolling(window=5, min_periods=1)
                .mean()
                .fillna(0)
            )
            data["liquidity_absorption_trend"] = (
                data["liquidity_absorption_rate"]
                .diff()
                .rolling(window=5, min_periods=1)
                .mean()
                .fillna(0)
            )
            data["depth_imbalance_momentum"] = (
                data["depth_imbalance"]
                .diff()
                .rolling(window=5, min_periods=1)
                .mean()
                .fillna(0)
            )
            data["iv_atm_spread"] = data["iv_atm"] * (
                data["ask_price_level_1"] - data["bid_price_level_1"]
            )
            data["gex_slope_volume_ratio"] = data["gex_slope"] / (
                data["volume"] + 1e-6
            )
            data["option_skew_vix_correlation"] = (
                data["option_skew"] * data["vix_es_correlation"]
            )

            microstructure_metrics = detect_microstructure_anomalies(data=data)
            data["spoofing_score"] = microstructure_metrics["spoofing_score"]
            data["volume_anomaly"] = microstructure_metrics["volume_anomaly"]

            data = calculate_news_metrics(data=data)

            for col in data.columns:
                if col != "timestamp":
                    data[col] = data.apply(
                        lambda x: self.impute_value(
                            value=x[col],
                            series=data[col],
                            feature_name=col,
                        ),
                        axis=1,
                    )

            try:
                start_time_prob = time.time()
                predictor = TradeProbabilityPredictor()
                data["trade_success_prob"] = data.apply(
                    lambda row: predictor.predict(row.to_frame().T),
                    axis=1,
                )
                latency = time.time() - start_time_prob
                self.log_performance(
                    operation="compute_trade_success_prob",
                    latency=latency,
                    success=True,
                )
                miya_speak(
                    message="trade_success_prob calculé",
                    tag="FEATURE_PIPELINE",
                    voice_profile="calm",
                    priority=1,
                )
                send_alert(
                    message="trade_success_prob calculé",
                    priority=1,
                )
                send_telegram_alert(message="trade_success_prob calculé")
            except Exception as e:
                error_msg = f"Erreur calcul trade_success_prob: {str(e)}"
                logger.error(error_msg)
                send_alert(message=error_msg, priority=3)
                send_telegram_alert(message=error_msg)
                self.log_performance(
                    operation="compute_trade_success_prob",
                    latency=time.time() - start_time_prob,
                    success=False,
                    error=str(e),
                )
                data["trade_success_prob"] = 0.0

            latency = time.time() - start_time
            self.log_performance(
                operation="compute_features",
                latency=latency,
                success=True,
                num_features=len(data.columns),
                confidence_drop_rate=confidence_drop_rate,
            )
            self.save_checkpoints(
                data=data,
                regime=regime if regime else "range",
            )
            self.save_snapshot(
                snapshot_type="compute_features",
                data={
                    "num_rows": len(data),
                    "num_features": len(data.columns),
                    "confidence_drop_rate": confidence_drop_rate,
                },
                compress=False,
            )
            return data
        except Exception as e:
            error_msg = f"Erreur dans compute_features: {str(e)}"
            logger.error(error_msg)
            send_alert(message=error_msg, priority=4)
            send_telegram_alert(message=error_msg)
            self.log_performance(
                operation="compute_features",
                latency=time.time() - start_time,
                success=False,
                error=str(e),
            )
            return data