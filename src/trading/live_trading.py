# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/trading/live_trading.py
# Rôle : Exécute les trades en live via Sierra Chart, utilisant trade_executor.py, signal_selector.py, risk_controller.py.
#
# Version : 2.1.5
# Date : 2025-05-14
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.0,<6.0.0, matplotlib>=3.8.0,<4.0.0, yaml,
#   asyncio, logging, argparse, json, hashlib, signal
# - src/model/inference.py
# - src/model/adaptive_learner.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/risk/risk_controller.py
# - src/risk/trade_window_filter.py
# - src/risk/decision_log.py
# - src/risk/options_risk_manager.py
# - src/risk/sierra_chart_errors.py
# - src/trading/trade_executor.py
# - src/strategy/mia_switcher.py
# - src/mind/mind.py
# - src/mind/mind_dialogue.py
# - src/analyze/analyze_results.py
# - src/analyze/analyze_trades.py
# - src/trading/utils/trading_loop.py
# - src/model/trade_probability.py
# - src/data/data_provider.py
# - src/api/news_scraper.py
# - src/trading/shap_weighting.py
#
# Inputs :
# - config/market_config.yaml
# - config/credentials.yaml
# - config/feature_sets.yaml
# - data/features/features_latest_filtered.csv
# - data/features/feature_importance_cache.csv
# - src/model/router/policies/defensive_policy.pkl
# - src/model/router/policies/range_policy.pkl
# - src/model/router/policies/trend_policy.pkl
#
# Outputs :
# - data/logs/trading/live_trading.log
# - data/logs/live_trading.csv
# - data/trade_snapshots/*.json (option *.json.gz)
# - data/figures/
# - data/live_trading_dashboard.json
# - data/checkpoints/
#
# Notes :
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence) définies dans config/feature_sets.yaml.
# - Utilise IQFeed exclusivement via data_provider.py pour les données d’entrée.
# - Supprime toute référence à 320 features, 81 features, et obs_t (remplacé par validate_obs_t).
# - Implémente retries (max 3, délai 2^attempt secondes) pour les opérations critiques.
# - Intègre trade_success_prob comme critère d’entrée, affiché sur le dashboard.
# - Valide quotidiennement les SHAP features via validate_shap_features avec validation obs_t.
# - Modularise via trading_loop.py, avec logs psutil, alertes via alert_manager et telegram_alert, et snapshots JSON.
# - Gère les interruptions SIGINT avec sauvegarde de snapshots via stop_trading.
# - Ajouts : Intégration des politiques SAC (defensive_policy.pkl, range_policy.pkl, trend_policy.pkl),
#   métriques de risque options (options_risk_manager.py), gestion des erreurs Sierra Chart (sierra_chart_errors.py),
#   données de nouvelles via news_scraper.py, sauvegardes incrémentielles dans data/checkpoints/.
# - Intègre confidence_drop_rate (Phase 8) pour l’auto-conscience.
# - Intègre l’analyse SHAP (Phase 17) dans analyze_real_time.
# - Nouvelles fonctionnalités : Validation des features bid_ask_imbalance, trade_aggressiveness, iv_skew, iv_term_structure,
#   option_skew, news_impact_score dans validate_data.
# - Encapsulation des configurations via ConfigContext dans config_manager.py.
# - Ajout de run() et stop_trading() pour gérer la boucle de trading et sa terminaison propre.
# - Correction de load_shap_fallback pour éviter les biais avec validation dynamique via validate_obs_t.
# - Tests unitaires à créer/mettre à jour dans tests/test_live_trading.py.

import asyncio
import json
import logging
import os
import pickle
import signal
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

from src.analyze.analyze_results import analyze_results
from src.analyze.analyze_trades import analyze_trades
from src.api.news_scraper import fetch_news
from src.data.data_provider import get_data_provider
from src.mind.mind import MindEngine
from src.mind.mind_dialogue import DialogueManager
from src.model.adaptive_learner import store_pattern
from src.model.inference import predict
from src.model.trade_probability import TradeProbabilityPredictor
from src.model.utils.alert_manager import AlertManager, send_alert
from src.model.utils.config_manager import config_manager
from src.risk.decision_log import DecisionLog
from src.risk.options_risk_manager import OptionsRiskManager
from src.risk.risk_controller import RiskController
from src.risk.sierra_chart_errors import SierraChartErrorManager
from src.risk.trade_window_filter import TradeWindowFilter
from src.strategy.mia_switcher import MIASwitcher
from src.trading.shap_weighting import calculate_shap
from src.trading.trade_executor import TradeExecutor

# Configuration logging
log_dir = Path("data/logs/trading")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_dir / "live_trading.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Chemins relatifs
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
SNAPSHOT_DIR = BASE_DIR / "data" / "trade_snapshots"
DASHBOARD_PATH = BASE_DIR / "data" / "live_trading_dashboard.json"
CSV_LOG_PATH = BASE_DIR / "data" / "logs" / "live_trading.csv"
DECISION_LOG_PATH = BASE_DIR / "data" / "logs" / "trading" / "decision_log.csv"
FIGURES_DIR = BASE_DIR / "data" / "figures"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints"

# Constantes
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Secondes, exponentiel
PERFORMANCE_THRESHOLDS = {
    "max_position_size": 5,
    "min_confidence": 0.7,
    "max_drawdown": -0.1,
    "vix_threshold": 30,
    "spread_threshold": 0.05,
    "market_liquidity_crash_risk": 0.8,
    "event_impact_threshold": 0.5,
}
SHAP_FEATURE_LIMIT = 50
CRITICAL_FEATURES = [
    "close",
    "bid_price_level_1",
    "ask_price_level_1",
    "timestamp",
    "vix",
    "rsi_14",
    "gex",
    "volume",
    "atr_14",
    "adx_14",
    "iv_atm",
    "option_skew",
    "news_impact_score",
    "spoofing_score",
    "volume_anomaly",
    "bid_ask_imbalance",
    "trade_aggressiveness",
    "iv_skew",
    "iv_term_structure",
]


class LiveTrader:
    """
    Classe pour exécuter les trades en live via Sierra Chart, intégrant gestion des risques et commandes vocales.
    """

    def __init__(
        self,
        config_path: str = "config/market_config.yaml",
        credentials_path: str = "config/credentials.yaml",
    ):
        """
        Initialise le trader en live.

        Args:
            config_path (str): Chemin vers la configuration du marché.
            credentials_path (str): Chemin vers les identifiants AMP Futures.
        """
        self.alert_manager = AlertManager()
        SNAPSHOT_DIR.mkdir(exist_ok=True)
        FIGURES_DIR.mkdir(exist_ok=True)
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        signal.signal(signal.SIGINT, self.handle_sigint)

        self.log_buffer = []
        self.trade_buffer = []
        self.prediction_cache = OrderedDict()
        self.running = False
        self.live_mode = False  # Ajout pour suivre le mode live/paper
        self.positions = []  # Ajout pour suivre les positions ouvertes
        try:
            self.config = self.load_config(config_path)
            self.credentials = self.load_credentials(credentials_path)
            self.mind_engine = MindEngine()
            self.trade_executor = TradeExecutor(config_path, credentials_path)
            self.dialogue_manager = DialogueManager()
            self.switcher = MIASwitcher()
            self.options_risk_manager = OptionsRiskManager()
            self.sierra_error_manager = SierraChartErrorManager()

            self.max_cache_size = self.config.get("cache", {}).get(
                "max_prediction_cache_size", 1000
            )
            self.cache_duration = (
                self.config.get("data", {}).get("cache_duration", 24) * 3600
            )
            self.vocal_command_failures = 0

            self.performance_thresholds = {
                "max_position_size": self.config.get("thresholds", {}).get(
                    "max_position_size", PERFORMANCE_THRESHOLDS["max_position_size"]
                ),
                "min_confidence": self.config.get("thresholds", {}).get(
                    "min_confidence", PERFORMANCE_THRESHOLDS["min_confidence"]
                ),
                "max_drawdown": self.config.get("thresholds", {}).get(
                    "max_drawdown", PERFORMANCE_THRESHOLDS["max_drawdown"]
                ),
                "vix_threshold": self.config.get("thresholds", {}).get(
                    "vix_threshold", PERFORMANCE_THRESHOLDS["vix_threshold"]
                ),
                "spread_threshold": self.config.get("thresholds", {}).get(
                    "spread_threshold", PERFORMANCE_THRESHOLDS["spread_threshold"]
                ),
                "market_liquidity_crash_risk": self.config.get("thresholds", {}).get(
                    "market_liquidity_crash_risk",
                    PERFORMANCE_THRESHOLDS["market_liquidity_crash_risk"],
                ),
                "event_impact_threshold": self.config.get("thresholds", {}).get(
                    "event_impact_threshold",
                    PERFORMANCE_THRESHOLDS["event_impact_threshold"],
                ),
                "min_sharpe": self.config.get("thresholds", {}).get("min_sharpe", 0.5),
                "min_profit_factor": self.config.get("thresholds", {}).get(
                    "min_profit_factor", 1.2
                ),
                "min_balance": self.config.get("thresholds", {}).get(
                    "min_balance", -10000
                ),
                "min_trade_success_prob": config_manager.get_config(
                    BASE_DIR / "config" / "trade_probability_config.yaml"
                )
                .get("trade_probability", {})
                .get("min_trade_success_prob", 0.7),
            }

            for key, value in self.performance_thresholds.items():
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Seuil invalide pour {key}: {value}")
                if (
                    key
                    in [
                        "max_position_size",
                        "min_confidence",
                        "vix_threshold",
                        "spread_threshold",
                        "market_liquidity_crash_risk",
                        "event_impact_threshold",
                        "min_sharpe",
                        "min_profit_factor",
                        "min_trade_success_prob",
                    ]
                    and value <= 0
                ):
                    raise ValueError(f"Seuil {key} doit être positif: {value}")
                if key == "max_drawdown" and value >= 0:
                    raise ValueError(f"Seuil {key} doit être négatif: {value}")

            logger.info("LiveTrader initialisé avec succès")
            send_alert("LiveTrader initialisé avec succès", priority=2)
            send_telegram_alert("LiveTrader initialisé avec succès")
            self.log_performance("init", 0, success=True)
        except Exception as e:
            send_alert(f"Erreur initialisation LiveTrader: {str(e)}", priority=5)
            send_telegram_alert(f"Erreur initialisation LiveTrader: {str(e)}")
            logger.error(f"Erreur initialisation LiveTrader: {str(e)}")
            self.log_performance("init", 0, success=False, error=str(e))
            raise

    def handle_sigint(self, signum, frame):
        """Gère l'arrêt propre sur SIGINT."""
        self.stop_trading()
        logger.info("Arrêt propre sur SIGINT")
        send_alert("Arrêt propre sur SIGINT", priority=2)
        send_telegram_alert("Arrêt propre sur SIGINT")
        exit(0)

    def save_checkpoint(
        self, trades: List[Dict], regime: str, compress: bool = True
    ) -> None:
        """Sauvegarde incrémentielle et distribuée des trades."""
        try:
            start_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = CHECKPOINT_DIR / f"trades_{regime}_{timestamp}.json"
            checkpoint_data = {
                "timestamp": timestamp,
                "regime": regime,
                "trades": trades,
            }
            os.makedirs(checkpoint_path.parent, exist_ok=True)
            if compress:
                if not (
                    os.environ.get("AWS_ACCESS_KEY_ID")
                    or os.path.exists(os.path.expanduser("~/.aws/credentials"))
                ):
                    logger.warning(
                        "Clés AWS non configurées, sauvegarde locale uniquement"
                    )
                with gzip.open(f"{checkpoint_path}.gz", "wt", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, indent=4)
                save_path = f"{checkpoint_path}.gz"
            else:
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, indent=4)
                save_path = checkpoint_path
            logger.info(f"Checkpoint sauvegardé : {save_path}")
            send_alert(f"Checkpoint sauvegardé : {save_path}", priority=1)
            send_telegram_alert(f"Checkpoint sauvegardé : {save_path}")
            self.log_performance(
                "save_checkpoint", time.time() - start_time, success=True
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde checkpoint : {str(e)}"
            logger.error(error_msg)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "save_checkpoint", time.time() - start_time, success=False, error=str(e)
            )

    def with_retries(
        self,
        func: callable,
        max_attempts: int = MAX_RETRIES,
        delay_base: float = RETRY_DELAY_BASE,
    ) -> Optional[Any]:
        """
        Exécute une fonction avec retries exponentiels.

        Args:
            func (callable): Fonction à exécuter.
            max_attempts (int): Nombre maximum de tentatives.
            delay_base (float): Base pour le délai exponentiel.

        Returns:
            Optional[Any]: Résultat de la fonction ou None si échec.
        """
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
                    send_alert(
                        f"Échec après {max_attempts} tentatives: {str(e)}", priority=3
                    )
                    send_telegram_alert(
                        f"Échec après {max_attempts} tentatives: {str(e)}"
                    )
                    logger.error(f"Échec après {max_attempts} tentatives: {str(e)}")
                    self.sierra_error_manager.log_error(
                        "RETRY_FAIL",
                        f"Échec après {max_attempts} tentatives: {str(e)}",
                        severity="ERROR",
                    )
                    return None
                delay = delay_base**attempt
                logger.warning(f"Tentative {attempt+1} échouée, retry après {delay}s")
                time.sleep(delay)

    async def with_retries_async(
        self,
        func: callable,
        max_attempts: int = MAX_RETRIES,
        delay_base: float = RETRY_DELAY_BASE,
    ) -> Optional[Any]:
        """
        Exécute une fonction asynchrone avec retries exponentiels.

        Args:
            func (callable): Fonction asynchrone à exécuter.
            max_attempts (int): Nombre maximum de tentatives.
            delay_base (float): Base pour le délai exponentiel.

        Returns:
            Optional[Any]: Résultat de la fonction ou None si échec.
        """
        start_time = time.time()
        for attempt in range(max_attempts):
            try:
                result = await func()
                latency = time.time() - start_time
                self.log_performance(
                    f"retry_async_attempt_{attempt+1}",
                    latency,
                    success=True,
                    attempt_number=attempt + 1,
                )
                return result
            except Exception as e:
                if attempt == max_attempts - 1:
                    latency = time.time() - start_time
                    self.log_performance(
                        f"retry_async_attempt_{attempt+1}",
                        latency,
                        success=False,
                        error=str(e),
                        attempt_number=attempt + 1,
                    )
                    send_alert(
                        f"Échec après {max_attempts} tentatives: {str(e)}", priority=3
                    )
                    send_telegram_alert(
                        f"Échec après {max_attempts} tentatives: {str(e)}"
                    )
                    logger.error(f"Échec après {max_attempts} tentatives: {str(e)}")
                    self.sierra_error_manager.log_error(
                        "RETRY_FAIL_ASYNC",
                        f"Échec après {max_attempts} tentatives: {str(e)}",
                        severity="ERROR",
                    )
                    return None
                delay = delay_base**attempt
                logger.warning(f"Tentative {attempt+1} échouée, retry après {delay}s")
                await asyncio.sleep(delay)

    def load_config(self, config_path: str) -> Dict:
        """
        Charge la configuration avec repli sur les valeurs par défaut.

        Args:
            config_path (str): Chemin vers le fichier de configuration.

        Returns:
            Dict: Configuration chargée.
        """
        start_time = time.time()
        try:
            config = config_manager.get_config(BASE_DIR / config_path)
            if not config:
                raise ValueError("Configuration vide ou non trouvée")
            self.log_performance("load_config", time.time() - start_time, success=True)
            logger.info(f"Configuration chargée: {config_path}")
            send_alert(f"Configuration chargée: {config_path}", priority=1)
            send_telegram_alert(f"Configuration chargée: {config_path}")
            return config
        except Exception as e:
            send_alert(f"Erreur chargement configuration: {str(e)}", priority=4)
            send_telegram_alert(f"Erreur chargement configuration: {str(e)}")
            logger.error(f"Erreur chargement configuration: {str(e)}")
            self.sierra_error_manager.log_error(
                "CONFIG_LOAD_FAIL",
                f"Erreur chargement configuration: {str(e)}",
                severity="CRITICAL",
            )
            self.log_performance(
                "load_config", time.time() - start_time, success=False, error=str(e)
            )
            return {
                "thresholds": PERFORMANCE_THRESHOLDS,
                "cache": {"max_prediction_cache_size": 1000},
                "data": {"cache_duration": 24},
                "plotting": {
                    "figsize": [12, 5],
                    "colors": {
                        "equity": "blue",
                        "profit": "orange",
                        "rewards": "green",
                    },
                },
            }

    def load_credentials(self, credentials_path: str) -> Dict:
        """
        Charge les identifiants AMP Futures.

        Args:
            credentials_path (str): Chemin vers le fichier d’identifiants.

        Returns:
            Dict: Identifiants chargés.
        """
        start_time = time.time()
        try:
            credentials = config_manager.get_config(BASE_DIR / credentials_path)
            if not credentials:
                raise ValueError("Identifiants vides ou non trouvés")
            self.log_performance(
                "load_credentials", time.time() - start_time, success=True
            )
            logger.info(f"Identifiants chargés: {credentials_path}")
            send_alert(f"Identifiants chargés: {credentials_path}", priority=1)
            send_telegram_alert(f"Identifiants chargés: {credentials_path}")
            return credentials
        except Exception as e:
            send_alert(f"Erreur chargement identifiants: {str(e)}", priority=4)
            send_telegram_alert(f"Erreur chargement identifiants: {str(e)}")
            logger.error(f"Erreur chargement identifiants: {str(e)}")
            self.sierra_error_manager.log_error(
                "CREDENTIALS_LOAD_FAIL",
                f"Erreur chargement identifiants: {str(e)}",
                severity="CRITICAL",
            )
            self.log_performance(
                "load_credentials",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            return {"amp_futures": {"api_key": "to_be_defined"}}

    def load_shap_fallback(self, regime: str) -> List[str]:
        """Charge un cache ou une liste statique de 150 SHAP features en cas d'échec, avec validation obs_t."""
        start_time = time.time()
        try:
            cache_path = BASE_DIR / "data" / "features" / "feature_importance_cache.csv"
            if cache_path.exists():
                shap_df = pd.read_csv(cache_path)
                features = shap_df["feature"].head(150).tolist()
                missing_features = config_manager.validate_obs_t(
                    features, context="inference"
                )
                if not missing_features:
                    logger.info(f"SHAP features chargées depuis cache : {cache_path}")
                    send_alert(
                        f"SHAP features chargées depuis cache : {cache_path}",
                        priority=1,
                    )
                    send_telegram_alert(
                        f"SHAP features chargées depuis cache : {cache_path}"
                    )
                    self.log_performance(
                        "load_shap_fallback",
                        time.time() - start_time,
                        success=True,
                        num_features=len(features),
                    )
                    return features
            config = config_manager.get_config(
                BASE_DIR / "config" / "feature_sets.yaml"
            )
            static_features = (
                config.get("feature_sets", {}).get(regime, {}).get("inference", [])
            )
            missing_features = config_manager.validate_obs_t(
                static_features, context="inference"
            )
            if not missing_features:
                logger.info(
                    "SHAP features chargées depuis liste statique : config/feature_sets.yaml"
                )
                send_alert(
                    "SHAP features chargées depuis liste statique : config/feature_sets.yaml",
                    priority=1,
                )
                send_telegram_alert(
                    "SHAP features chargées depuis liste statique : config/feature_sets.yaml"
                )
                self.log_performance(
                    "load_shap_fallback",
                    time.time() - start_time,
                    success=True,
                    num_features=len(static_features),
                )
                return static_features[:150]
            raise ValueError("SHAP features statiques invalides après validation obs_t")
        except Exception as e:
            error_msg = f"Erreur chargement SHAP fallback : {str(e)}"
            logger.error(error_msg)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.sierra_error_manager.log_error(
                "SHAP_FALLBACK_FAIL", error_msg, severity="ERROR"
            )
            self.log_performance(
                "load_shap_fallback",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            # Fallback sécurisé avec validation
            fallback_features = config.get("fallback_features", {}).get("features", [])[
                :150
            ]
            missing_features = config_manager.validate_obs_t(
                fallback_features, context="inference"
            )
            if missing_features:
                logger.warning(f"Fallback SHAP features incomplet : {missing_features}")
                send_alert(
                    f"Fallback SHAP features incomplet : {missing_features}", priority=3
                )
                send_telegram_alert(
                    f"Fallback SHAP features incomplet : {missing_features}"
                )
            return fallback_features

    def configure_feature_set(self) -> List[str]:
        """
        Configure les 150 SHAP features pour l’inférence.

        Returns:
            List[str]: Liste des noms des features.
        """
        start_time = time.time()
        try:
            shap_file = BASE_DIR / "data" / "features" / "feature_importance.csv"
            regime = "range"  # Par défaut, peut être ajusté dynamiquement
            if shap_file.exists():
                shap_df = pd.read_csv(shap_file)
                if len(shap_df) >= 150:
                    features = shap_df["feature"].head(150).tolist()
                    missing_features = config_manager.validate_obs_t(
                        features, context="inference"
                    )
                    if not missing_features:
                        self.log_performance(
                            "configure_feature_set",
                            time.time() - start_time,
                            success=True,
                            num_features=len(features),
                        )
                        logger.info(
                            f"SHAP features configurées: {len(features)} features"
                        )
                        send_alert(
                            f"SHAP features configurées: {len(features)} features",
                            priority=1,
                        )
                        send_telegram_alert(
                            f"SHAP features configurées: {len(features)} features"
                        )
                        return features
            send_alert(
                "SHAP features non trouvées, repli sur cache ou liste statique",
                priority=3,
            )
            send_telegram_alert(
                "SHAP features non trouvées, repli sur cache ou liste statique"
            )
            logger.warning(
                "SHAP features non trouvées, repli sur cache ou liste statique"
            )
            features = self.load_shap_fallback(regime)
            self.log_performance(
                "configure_feature_set",
                time.time() - start_time,
                success=True,
                num_features=len(features),
            )
            return features
        except Exception as e:
            send_alert(f"Erreur chargement SHAP features: {str(e)}", priority=3)
            send_telegram_alert(f"Erreur chargement SHAP features: {str(e)}")
            logger.error(f"Erreur chargement SHAP features: {str(e)}")
            self.sierra_error_manager.log_error(
                "SHAP_CONFIG_FAIL",
                f"Erreur chargement SHAP features: {str(e)}",
                severity="ERROR",
            )
            self.log_performance(
                "configure_feature_set",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            return self.load_shap_fallback("range")

    def validate_shap_features(self) -> bool:
        """
        Valide quotidiennement les SHAP features.

        Returns:
            bool: True si validé, False sinon.
        """
        start_time = time.time()
        try:
            shap_file = BASE_DIR / "data" / "features" / "feature_importance.csv"
            if not shap_file.exists():
                send_alert("Fichier SHAP features manquant", priority=4)
                send_telegram_alert("Fichier SHAP features manquant")
                logger.error("Fichier SHAP features manquant")
                self.sierra_error_manager.log_error(
                    "SHAP_FILE_MISSING",
                    "Fichier SHAP features manquant",
                    severity="CRITICAL",
                )
                self.log_performance(
                    "validate_shap_features",
                    time.time() - start_time,
                    success=False,
                    error="Fichier SHAP manquant",
                )
                return False
            shap_df = pd.read_csv(shap_file)
            if len(shap_df) < 150:
                send_alert(
                    f"SHAP features insuffisantes: {len(shap_df)} < 150", priority=4
                )
                send_telegram_alert(
                    f"SHAP features insuffisantes: {len(shap_df)} < 150"
                )
                logger.error(f"SHAP features insuffisantes: {len(shap_df)}")
                self.sierra_error_manager.log_error(
                    "SHAP_INSUFFICIENT",
                    f"SHAP features insuffisantes: {len(shap_df)}",
                    severity="CRITICAL",
                )
                self.log_performance(
                    "validate_shap_features",
                    time.time() - start_time,
                    success=False,
                    error="Features insuffisantes",
                )
                return False
            features = shap_df["feature"].head(150).tolist()
            missing_features = config_manager.validate_obs_t(
                features, context="inference"
            )
            if missing_features:
                send_alert(
                    f"SHAP features invalides: features critiques manquantes {missing_features}",
                    priority=4,
                )
                send_telegram_alert(
                    f"SHAP features invalides: features critiques manquantes {missing_features}"
                )
                logger.error(
                    f"SHAP features invalides: features critiques manquantes {missing_features}"
                )
                self.log_performance(
                    "validate_shap_features",
                    time.time() - start_time,
                    success=False,
                    error=f"Features manquantes: {missing_features}",
                )
                return False
            logger.info("SHAP features validées")
            send_alert("SHAP features validées", priority=1)
            send_telegram_alert("SHAP features validées")
            self.log_performance(
                "validate_shap_features",
                time.time() - start_time,
                success=True,
                num_features=len(shap_df),
            )
            return True
        except Exception as e:
            send_alert(f"Erreur validation SHAP features: {str(e)}", priority=4)
            send_telegram_alert(f"Erreur validation SHAP features: {str(e)}")
            logger.error(f"Erreur validation SHAP features: {str(e)}")
            self.sierra_error_manager.log_error(
                "SHAP_VALIDATE_FAIL",
                f"Erreur validation SHAP features: {str(e)}",
                severity="CRITICAL",
            )
            self.log_performance(
                "validate_shap_features",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            return False

    def detect_feature_set(self, data: pd.DataFrame) -> str:
        """
        Détecte le type d’ensemble de features basé sur le nombre de colonnes.

        Args:
            data (pd.DataFrame): Données à analyser.

        Returns:
            str: Type d’ensemble ('training', 'inference').
        """
        start_time = time.time()
        try:
            num_cols = len(data.columns)
            if num_cols >= 350:
                feature_set = "training"
            elif num_cols >= 150:
                feature_set = "inference"
            else:
                raise ValueError(
                    f"Nombre de features insuffisant: {num_cols}, attendu ≥150"
                )
            self.log_performance(
                "detect_feature_set",
                time.time() - start_time,
                success=True,
                num_columns=num_cols,
            )
            return feature_set
        except Exception as e:
            send_alert(f"Erreur détection feature set: {str(e)}", priority=3)
            send_telegram_alert(f"Erreur détection feature set: {str(e)}")
            logger.error(f"Erreur détection feature set: {str(e)}")
            self.sierra_error_manager.log_error(
                "FEATURE_SET_FAIL",
                f"Erreur détection feature set: {str(e)}",
                severity="ERROR",
            )
            self.log_performance(
                "detect_feature_set",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            return "inference"

    async def load_data(self, feature_file: str) -> pd.DataFrame:
        """
        Charge les données via IQFeed, incluant les données de nouvelles.

        Args:
            feature_file (str): Chemin du fichier de features.

        Returns:
            pd.DataFrame: Données chargées.
        """
        start_time = time.time()
        try:

            def fetch_data():
                provider = get_data_provider("iqfeed")
                market_data = provider.fetch_features(feature_file)
                news_data = fetch_news(
                    self.credentials.get("newsapi", {}).get("api_key", "")
                )
                combined_data = pd.concat([market_data, news_data], axis=1)
                self.validate_data(combined_data)
                return combined_data

            data = await self.with_retries_async(fetch_data)
            if data is None or data.empty:
                raise ValueError("Échec du chargement des données")
            self.log_performance(
                "load_data", time.time() - start_time, success=True, num_rows=len(data)
            )
            logger.info(f"Données chargées: {feature_file}")
            send_alert(f"Données chargées: {feature_file}", priority=1)
            send_telegram_alert(f"Données chargées: {feature_file}")
            return data
        except Exception as e:
            send_alert(f"Erreur chargement données depuis IQFeed: {str(e)}", priority=4)
            send_telegram_alert(f"Erreur chargement données depuis IQFeed: {str(e)}")
            logger.error(f"Erreur chargement données depuis IQFeed: {str(e)}")
            self.sierra_error_manager.log_error(
                "DATA_LOAD_FAIL",
                f"Erreur chargement données: {str(e)}",
                severity="CRITICAL",
            )
            self.log_performance(
                "load_data", time.time() - start_time, success=False, error=str(e)
            )
            return pd.DataFrame()

    def impute_timestamp(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Impute les timestamps invalides en utilisant le dernier timestamp valide.

        Args:
            data (pd.DataFrame): Données avec colonne 'timestamp'.

        Returns:
            pd.DataFrame: Données avec timestamps imputés.
        """
        start_time = time.time()
        try:
            if "timestamp" not in data.columns:
                send_alert("Colonne timestamp manquante", priority=5)
                send_telegram_alert("Colonne timestamp manquante")
                logger.error("Colonne timestamp manquante")
                self.sierra_error_manager.log_error(
                    "TIMESTAMP_MISSING",
                    "Colonne timestamp manquante",
                    severity="CRITICAL",
                )
                return data

            data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
            if data["timestamp"].isna().any():
                last_valid = (
                    data["timestamp"].dropna().iloc[-1]
                    if not data["timestamp"].dropna().empty
                    else pd.Timestamp.now()
                )
                data["timestamp"] = data["timestamp"].fillna(last_valid)
                send_alert(
                    f"Valeurs timestamp invalides imputées avec {last_valid}",
                    priority=2,
                )
                send_telegram_alert(
                    f"Valeurs timestamp invalides imputées avec {last_valid}"
                )
                logger.warning(
                    f"Valeurs timestamp invalides imputées avec {last_valid}"
                )
            self.log_performance(
                "impute_timestamp",
                time.time() - start_time,
                success=True,
                num_rows=len(data),
            )
            return data
        except Exception as e:
            send_alert(f"Erreur imputation timestamp: {str(e)}", priority=3)
            send_telegram_alert(f"Erreur imputation timestamp: {str(e)}")
            logger.error(f"Erreur imputation timestamp: {str(e)}")
            self.sierra_error_manager.log_error(
                "TIMESTAMP_IMPUTE_FAIL",
                f"Erreur imputation timestamp: {str(e)}",
                severity="ERROR",
            )
            self.log_performance(
                "impute_timestamp",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            return data

    def validate_obs_t(
        self, data: pd.DataFrame, context: str = "inference"
    ) -> List[str]:
        """
        Valide les données par rapport à un modèle obs_t standardisé.

        Args:
            data (pd.DataFrame): Données à valider.
            context (str): Contexte ('training' ou 'inference').

        Returns:
            List[str]: Liste des features manquantes.
        """
        start_time = time.time()
        try:
            features = data.columns.tolist()
            missing_features = config_manager.validate_obs_t(features, context=context)
            if missing_features:
                logger.warning(
                    f"Validation obs_t échouée pour {context}: features manquantes {missing_features}"
                )
                send_alert(
                    f"Validation obs_t échouée pour {context}: features manquantes {missing_features}",
                    priority=3,
                )
                send_telegram_alert(
                    f"Validation obs_t échouée pour {context}: features manquantes {missing_features}"
                )
            self.log_performance(
                "validate_obs_t",
                time.time() - start_time,
                success=not missing_features,
                error=str(missing_features) if missing_features else None,
            )
            return missing_features
        except Exception as e:
            send_alert(f"Erreur validation obs_t: {str(e)}", priority=3)
            send_telegram_alert(f"Erreur validation obs_t: {str(e)}")
            logger.error(f"Erreur validation obs_t: {str(e)}")
            self.log_performance(
                "validate_obs_t", time.time() - start_time, success=False, error=str(e)
            )
            return CRITICAL_FEATURES

    def validate_data(self, data: pd.DataFrame) -> None:
        """
        Valide les données d’entrée (350 features pour entraînement, 150 SHAP pour inférence).

        Args:
            data (pd.DataFrame): Données à valider.

        Raises:
            ValueError: Si les données sont invalides.
        """
        start_time = time.time()
        try:
            feature_set = self.detect_feature_set(data)
            expected_features = 350 if feature_set == "training" else 150

            if len(data.columns) < expected_features:
                raise ValueError(
                    f"Nombre de features insuffisant: {len(data.columns)} < {expected_features} pour ensemble {feature_set}"
                )

            expected_cols = self.configure_feature_set()
            missing_cols = [col for col in expected_cols if col not in data.columns]
            if missing_cols and feature_set == "inference":
                raise ValueError(f"Colonnes manquantes: {missing_cols}")

            missing_critical = self.validate_obs_t(data, context=feature_set)
            if missing_critical:
                raise ValueError(
                    f"Colonnes critiques manquantes pour {feature_set}: {missing_critical}"
                )

            for col in CRITICAL_FEATURES:
                if col in data.columns:
                    if col != "timestamp" and data[col].isnull().any():
                        raise ValueError(f"Colonne {col} contient des NaN")
                    if col != "timestamp" and not pd.api.types.is_numeric_dtype(
                        data[col]
                    ):
                        raise ValueError(
                            f"Colonne {col} n’est pas numérique: {data[col].dtype}"
                        )
                    if col != "timestamp" and data[col].le(0).any():
                        raise ValueError(
                            f"Colonne {col} contient des valeurs négatives ou zéro"
                        )

            data = self.impute_timestamp(data)
            latest_timestamp = data["timestamp"].iloc[-1]
            if not isinstance(latest_timestamp, pd.Timestamp):
                raise ValueError(f"Timestamp invalide: {latest_timestamp}")
            if latest_timestamp > datetime.now() + timedelta(
                minutes=5
            ) or latest_timestamp < datetime.now() - timedelta(hours=24):
                raise ValueError(f"Timestamp hors plage: {latest_timestamp}")

            logger.info(
                f"Données validées pour ensemble {feature_set} avec {len(data.columns)} features"
            )
            send_alert(f"Données validées pour ensemble {feature_set}", priority=1)
            send_telegram_alert(f"Données validées pour ensemble {feature_set}")
            self.log_performance(
                "validate_data",
                time.time() - start_time,
                success=True,
                num_features=len(data.columns),
            )
        except Exception as e:
            send_alert(f"Erreur validation données: {str(e)}", priority=5)
            send_telegram_alert(f"Erreur validation données: {str(e)}")
            logger.error(f"Erreur validation données: {str(e)}")
            self.sierra_error_manager.log_error(
                "DATA_VALIDATE_FAIL",
                f"Erreur validation données: {str(e)}",
                severity="CRITICAL",
            )
            self.log_performance(
                "validate_data", time.time() - start_time, success=False, error=str(e)
            )
            raise

    def validate_pattern_data(
        self,
        data: pd.DataFrame,
        action: float,
        reward: float,
        neural_regime: str,
        confidence: float,
    ) -> bool:
        """
        Valide les données pour le stockage dans market_memory.db.

        Args:
            data (pd.DataFrame): Données à valider.
            action (float): Action effectuée.
            reward (float): Récompense obtenue.
            neural_regime (str): Régime neuronal.
            confidence (float): Confiance de la prédiction.

        Returns:
            bool: True si validé, False sinon.
        """
        start_time = time.time()
        try:
            if data.isnull().any().any():
                send_alert("Données contiennent des NaN pour le stockage", priority=3)
                send_telegram_alert("Données contiennent des NaN pour le stockage")
                logger.error("Données contiennent des NaN pour le stockage")
                self.sierra_error_manager.log_error(
                    "PATTERN_DATA_NAN",
                    "Données contiennent des NaN pour le stockage",
                    severity="ERROR",
                )
                return False
            if (
                not isinstance(action, (int, float))
                or not isinstance(reward, (int, float))
                or not isinstance(confidence, (int, float))
            ):
                send_alert("Action, récompense ou confiance non-numérique", priority=3)
                send_telegram_alert("Action, récompense ou confiance non-numérique")
                logger.error("Action, récompense ou confiance non-numérique")
                self.sierra_error_manager.log_error(
                    "PATTERN_DATA_TYPE",
                    "Action, récompense ou confiance non-numérique",
                    severity="ERROR",
                )
                return False
            if not isinstance(neural_regime, (str, int, float)):
                send_alert("Régime neuronal invalide pour le stockage", priority=3)
                send_telegram_alert("Régime neuronal invalide pour le stockage")
                logger.error("Régime neuronal invalide pour le stockage")
                self.sierra_error_manager.log_error(
                    "PATTERN_REGIME_INVALID",
                    "Régime neuronal invalide pour le stockage",
                    severity="ERROR",
                )
                return False
            self.log_performance(
                "validate_pattern_data", time.time() - start_time, success=True
            )
            return True
        except Exception as e:
            send_alert(f"Erreur validation données pattern: {str(e)}", priority=3)
            send_telegram_alert(f"Erreur validation données pattern: {str(e)}")
            logger.error(f"Erreur validation données pattern: {str(e)}")
            self.sierra_error_manager.log_error(
                "PATTERN_VALIDATE_FAIL",
                f"Erreur validation données pattern: {str(e)}",
                severity="ERROR",
            )
            self.log_performance(
                "validate_pattern_data",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            return False

    @lru_cache(maxsize=100)
    def select_model(self, data: pd.DataFrame, current_step: int) -> Dict:
        """
        Sélectionne le modèle via MIASwitcher ou une politique SAC.

        Args:
            data (pd.DataFrame): Données actuelles.
            current_step (int): Étape actuelle.

        Returns:
            Dict: Informations sur le modèle sélectionné.
        """
        start_time = time.time()
        try:

            def select():
                regime = data.get("neural_regime", "range").iloc[-1]
                policy_paths = {
                    "defensive": BASE_DIR
                    / "src"
                    / "model"
                    / "router"
                    / "policies"
                    / "defensive_policy.pkl",
                    "range": BASE_DIR
                    / "src"
                    / "model"
                    / "router"
                    / "policies"
                    / "range_policy.pkl",
                    "trend": BASE_DIR
                    / "src"
                    / "model"
                    / "router"
                    / "policies"
                    / "trend_policy.pkl",
                }
                policy_path = policy_paths.get(regime, policy_paths["range"])
                if policy_path.exists():
                    with open(policy_path, "rb") as f:
                        policy = pickle.load(f)
                    return {
                        "type": f"sac_{regime}",
                        "path": str(policy_path),
                        "policy": policy,
                        "policy_type": "sac",
                    }
                return self.switcher.switch_mia(data, current_step)

            selected_model = self.with_retries(select)
            if selected_model is None:
                raise ValueError("Échec de la sélection du modèle")
            logger.info(
                f"Modèle sélectionné: {selected_model['type']} ({selected_model['path']})"
            )
            send_alert(f"Modèle sélectionné: {selected_model['type']}", priority=2)
            send_telegram_alert(f"Modèle sélectionné: {selected_model['type']}")
            self.log_performance("select_model", time.time() - start_time, success=True)
            return selected_model
        except Exception as e:
            send_alert(f"Erreur sélection modèle: {str(e)}, repli sur SAC", priority=3)
            send_telegram_alert(f"Erreur sélection modèle: {str(e)}, repli sur SAC")
            logger.error(f"Erreur sélection modèle: {str(e)}")
            self.sierra_error_manager.log_error(
                "MODEL_SELECT_FAIL",
                f"Erreur sélection modèle: {str(e)}",
                severity="ERROR",
            )
            self.log_performance(
                "select_model", time.time() - start_time, success=False, error=str(e)
            )
            return {
                "type": "sac",
                "path": str(BASE_DIR / "data" / "models" / "sac_model.pth"),
                "policy_type": "mlp",
            }

    async def handle_vocal_commands(
        self,
        current_data: pd.DataFrame,
        current_step: int,
        balance: float,
        positions: List,
        risk_controller: RiskController,
        trade_window_filter: TradeWindowFilter,
        decision_log: DecisionLog,
        live_mode: bool,
    ) -> bool:
        """
        Gère les commandes vocales pour le trading.

        Args:
            current_data (pd.DataFrame): Données actuelles.
            current_step (int): Étape actuelle.
            balance (float): Balance actuelle.
            positions (List): Positions ouvertes.
            risk_controller (RiskController): Contrôleur de risque.
            trade_window_filter (TradeWindowFilter): Filtre de fenêtre de trading.
            decision_log (DecisionLog): Journal des décisions.
            live_mode (bool): Mode live (True) ou paper (False).

        Returns:
            bool: True si arrêt demandé, False sinon.
        """
        start_time = time.time()
        try:
            if current_data.empty:
                send_alert(
                    "Aucune donnée disponible pour les commandes vocales", priority=3
                )
                send_telegram_alert(
                    "Aucune donnée disponible pour les commandes vocales"
                )
                self.sierra_error_manager.log_error(
                    "VOCAL_NO_DATA",
                    "Aucune donnée disponible pour les commandes vocales",
                    severity="ERROR",
                )
                self.log_performance(
                    "handle_vocal_commands",
                    time.time() - start_time,
                    success=False,
                    error="Données vides",
                )
                return False

            async def listen_command():
                return self.dialogue_manager.listen_command()

            command = await self.with_retries_async(listen_command)
            if not command:
                self.log_performance(
                    "handle_vocal_commands",
                    time.time() - start_time,
                    success=False,
                    error="Aucune commande reçue",
                )
                return False

            if command.lower() == "execute trade":
                if trade_window_filter.check_trade_window(
                    0.5, current_data, current_step
                ):
                    self.dialogue_manager.respond("Fenêtre de trading inappropriée")
                    decision_log.log_decision(
                        trade_id=f"vocal_{current_step}",
                        decision="block",
                        score=0.5,
                        reason="Fenêtre de trading inappropriée (vocal)",
                        data=current_data,
                    )
                    self.log_performance(
                        "handle_vocal_commands",
                        time.time() - start_time,
                        success=True,
                        command=command,
                    )
                    return False

                risk_metrics = risk_controller.calculate_risk_metrics(
                    current_data, positions
                )
                options_risk = self.options_risk_manager.calculate_options_risk(
                    current_data
                )
                if (
                    risk_metrics["market_liquidity_crash_risk"]
                    > self.performance_thresholds["market_liquidity_crash_risk"]
                    or options_risk["risk_alert"].any()
                ):
                    self.dialogue_manager.respond(
                        "Risque de liquidité ou options trop élevé"
                    )
                    decision_log.log_decision(
                        trade_id=f"vocal_{current_step}",
                        decision="block",
                        score=0.5,
                        reason="Risque de liquidité ou options élevé (vocal)",
                        data=current_data,
                    )
                    self.log_performance(
                        "handle_vocal_commands",
                        time.time() - start_time,
                        success=True,
                        command=command,
                    )
                    return False

                if balance < self.performance_thresholds["min_balance"]:
                    send_alert(f"Balance incohérente: {balance:.2f}", priority=3)
                    send_telegram_alert(f"Balance incohérente: {balance:.2f}")
                    self.dialogue_manager.respond(
                        "Balance insuffisante pour exécuter le trade"
                    )
                    decision_log.log_decision(
                        trade_id=f"vocal_{current_step}",
                        decision="block",
                        score=0.5,
                        reason="Balance incohérente (vocal)",
                        data=current_data,
                    )
                    self.log_performance(
                        "handle_vocal_commands",
                        time.time() - start_time,
                        success=True,
                        command=command,
                    )
                    return False

                trade = {
                    "trade_id": f"vocal_{current_step}",
                    "action": 1,
                    "price": current_data["close"].iloc[-1],
                    "size": 1,
                    "order_type": "market",
                }

                def execute_trade():
                    execution_result = self.trade_executor.execute_trade(
                        trade,
                        mode="real" if live_mode else "paper",
                        context_data=current_data,
                    )
                    if execution_result["status"] == "filled":
                        confirmation = self.trade_executor.confirm_trade(
                            trade["trade_id"]
                        )
                        if confirmation["confirmed"]:
                            self.trade_executor.log_execution(
                                trade, execution_result, is_paper=not live_mode
                            )
                            self.dialogue_manager.respond(
                                f"Trade {trade['trade_id']} exécuté avec succès"
                            )
                    return execution_result

                result = self.with_retries(execute_trade)
                if result is None:
                    self.dialogue_manager.respond("Échec de l’exécution du trade")
                    self.sierra_error_manager.log_error(
                        "TRADE_EXECUTE_FAIL",
                        "Échec de l’exécution du trade vocal",
                        severity="ERROR",
                    )
                    self.log_performance(
                        "handle_vocal_commands",
                        time.time() - start_time,
                        success=False,
                        command=command,
                        error="Échec exécution",
                    )
                    return False

                self.log_performance(
                    "handle_vocal_commands",
                    time.time() - start_time,
                    success=True,
                    command=command,
                )
                return False

            elif command.lower() == "stop trading":
                self.dialogue_manager.respond("Trading arrêté")
                send_alert("Commande vocale: Trading arrêté", priority=3)
                send_telegram_alert("Commande vocale: Trading arrêté")
                self.log_performance(
                    "handle_vocal_commands",
                    time.time() - start_time,
                    success=True,
                    command=command,
                )
                return True

            elif command.lower() == "view dashboard":
                metrics = self.compute_metrics([])
                self.dialogue_manager.respond(
                    f"Trades exécutés: {metrics.get('trades_executed', 0)}, Balance: {balance:.2f}, Sharpe: {metrics.get('sharpe', 0.0):.2f}"
                )
                self.log_performance(
                    "handle_vocal_commands",
                    time.time() - start_time,
                    success=True,
                    command=command,
                )
                return False

            elif command.lower().startswith("switch to live trading"):
                if live_mode:
                    self.dialogue_manager.respond("Déjà en mode trading live")
                else:
                    credentials = self.load_credentials(
                        str(BASE_DIR / "config" / "credentials.yaml")
                    )
                    if (
                        credentials.get("amp_futures", {}).get(
                            "api_key", "to_be_defined"
                        )
                        == "to_be_defined"
                    ):
                        self.dialogue_manager.respond(
                            "Impossible de passer en mode live: Clé API AMP Futures non configurée"
                        )
                    else:
                        live_mode = True
                        self.live_mode = True
                        send_alert("Passage en mode trading live", priority=3)
                        send_telegram_alert("Passage en mode trading live")
                        self.dialogue_manager.respond("Passage en mode trading live")
                self.log_performance(
                    "handle_vocal_commands",
                    time.time() - start_time,
                    success=True,
                    command=command,
                )
                return False

            elif command.lower().startswith("adjust position size"):
                try:
                    size = float(command.split()[-1])
                    if (
                        size <= 0
                        or size > self.performance_thresholds["max_position_size"]
                    ):
                        self.dialogue_manager.respond(
                            f"Taille de position invalide: {size}, doit être entre 0 et {self.performance_thresholds['max_position_size']}"
                        )
                    else:
                        send_alert(f"Taille de position ajustée à {size}", priority=2)
                        send_telegram_alert(f"Taille de position ajustée à {size}")
                        self.dialogue_manager.respond(
                            f"Taille de position ajustée à {size}"
                        )
                    self.log_performance(
                        "handle_vocal_commands",
                        time.time() - start_time,
                        success=True,
                        command=command,
                    )
                    return False
                except ValueError:
                    self.dialogue_manager.respond(
                        "Format de taille de position invalide"
                    )
                    self.sierra_error_manager.log_error(
                        "VOCAL_INVALID_SIZE",
                        "Format de taille de position invalide",
                        severity="ERROR",
                    )
                    self.log_performance(
                        "handle_vocal_commands",
                        time.time() - start_time,
                        success=False,
                        command=command,
                        error="Format invalide",
                    )
                    return False

            else:
                send_alert(f"Commande vocale non reconnue: {command}", priority=3)
                send_telegram_alert(f"Commande vocale non reconnue: {command}")
                self.dialogue_manager.respond("Commande non reconnue")
                self.vocal_command_failures += 1
                if self.vocal_command_failures >= 3:
                    send_alert(
                        "Trop d’échecs de commandes vocales, passage en mode texte",
                        priority=5,
                    )
                    send_telegram_alert(
                        "Trop d’échecs de commandes vocales, passage en mode texte"
                    )
                    self.vocal_command_failures = 0
                self.log_performance(
                    "handle_vocal_commands",
                    time.time() - start_time,
                    success=False,
                    command=command,
                    error="Commande non reconnue",
                )
                return False

        except Exception as e:
            send_alert(f"Erreur gestion commande vocale: {str(e)}", priority=3)
            send_telegram_alert(f"Erreur gestion commande vocale: {str(e)}")
            logger.error(f"Erreur gestion commande vocale: {str(e)}")
            self.sierra_error_manager.log_error(
                "VOCAL_COMMAND_FAIL",
                f"Erreur gestion commande vocale: {str(e)}",
                severity="ERROR",
            )
            self.log_performance(
                "handle_vocal_commands",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            return False

    def validate_plot_config(self, plot_config: Dict) -> Dict:
        """
        Valide la configuration des graphiques.

        Args:
            plot_config (Dict): Configuration des graphiques.

        Returns:
            Dict: Configuration validée.
        """
        start_time = time.time()
        try:
            default_config = {
                "figsize": [12, 5],
                "colors": {"equity": "blue", "profit": "orange", "rewards": "green"},
            }
            if (
                not isinstance(plot_config.get("figsize"), list)
                or len(plot_config.get("figsize")) != 2
            ):
                send_alert(
                    "Figsize invalide dans plot_config, utilisation par défaut",
                    priority=2,
                )
                send_telegram_alert(
                    "Figsize invalide dans plot_config, utilisation par défaut"
                )
                plot_config["figsize"] = default_config["figsize"]
            if not isinstance(plot_config.get("colors"), dict) or not all(
                k in plot_config["colors"] for k in ["equity", "profit", "rewards"]
            ):
                send_alert(
                    "Couleurs invalides dans plot_config, utilisation par défaut",
                    priority=2,
                )
                send_telegram_alert(
                    "Couleurs invalides dans plot_config, utilisation par défaut"
                )
                plot_config["colors"] = default_config["colors"]
            self.log_performance(
                "validate_plot_config", time.time() - start_time, success=True
            )
            return plot_config
        except Exception as e:
            send_alert(
                f"Erreur validation plot_config: {str(e)}, utilisation par défaut",
                priority=3,
            )
            send_telegram_alert(
                f"Erreur validation plot_config: {str(e)}, utilisation par défaut"
            )
            logger.error(f"Erreur validation plot_config: {str(e)}")
            self.sierra_error_manager.log_error(
                "PLOT_CONFIG_FAIL",
                f"Erreur validation plot_config: {str(e)}",
                severity="ERROR",
            )
            self.log_performance(
                "validate_plot_config",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            return default_config

    def plot_real_time_metrics(
        self,
        rewards_history: List[float],
        timestamps: List[pd.Timestamp],
        output_dir: str = str(FIGURES_DIR),
    ) -> None:
        """
        Génère des graphiques en temps réel pour les métriques.

        Args:
            rewards_history (List[float]): Historique des récompenses.
            timestamps (List[pd.Timestamp]): Horodatages correspondants.
            output_dir (str): Répertoire de sortie des graphiques.
        """
        start_time = time.time()
        try:
            logger.info("Génération des graphiques en temps réel")
            send_alert("Génération des graphiques en temps réel", priority=2)
            send_telegram_alert("Génération des graphiques en temps réel")

            if not os.access(output_dir, os.W_OK):
                raise PermissionError(f"Pas de permission d’écriture pour {output_dir}")

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_config = self.validate_plot_config(self.config.get("plotting", {}))

            equity = np.cumsum(rewards_history)
            plt.figure(figsize=plot_config["figsize"])
            plt.plot(
                timestamps,
                equity,
                label="Equity Curve",
                color=plot_config["colors"]["equity"],
            )
            plt.title("Courbe d’Équité en Temps Réel")
            plt.xlabel("Timestamp")
            plt.ylabel("Équité")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()
            equity_path = os.path.join(output_dir, f"live_equity_{timestamp_str}.png")

            def save_equity():
                plt.savefig(equity_path)
                plt.close()

            self.with_retries(save_equity)
            logger.info(f"Courbe d’équité sauvegardée: {equity_path}")
            send_alert(f"Courbe d’équité sauvegardée: {equity_path}", priority=2)
            send_telegram_alert(f"Courbe d’équité sauvegardée: {equity_path}")

            plt.figure(figsize=plot_config["figsize"])
            plt.plot(
                timestamps,
                equity,
                label="Profit Cumulé",
                color=plot_config["colors"]["profit"],
            )
            plt.title("Profit Cumulé en Temps Réel")
            plt.xlabel("Timestamp")
            plt.ylabel("Profit Cumulé")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()
            profit_path = os.path.join(output_dir, f"live_profit_{timestamp_str}.png")

            def save_profit():
                plt.savefig(profit_path)
                plt.close()

            self.with_retries(save_profit)
            logger.info(f"Profit cumulé sauvegardé: {profit_path}")
            send_alert(f"Profit cumulé sauvegardé: {profit_path}", priority=2)
            send_telegram_alert(f"Profit cumulé sauvegardé: {profit_path}")

            plt.figure(figsize=plot_config["figsize"])
            plt.hist(
                rewards_history,
                bins=30,
                color=plot_config["colors"]["rewards"],
                edgecolor="black",
                alpha=0.7,
            )
            plt.title("Distribution des Récompenses en Temps Réel")
            plt.xlabel("Récompense par trade")
            plt.ylabel("Nombre de trades")
            plt.grid(True, linestyle="--", alpha=0.7)
            dist_path = os.path.join(output_dir, f"live_rewards_{timestamp_str}.png")

            def save_dist():
                plt.savefig(dist_path)
                plt.close()

            self.with_retries(save_dist)
            logger.info(f"Distribution des récompenses sauvegardée: {dist_path}")
            send_alert(
                f"Distribution des récompenses sauvegardée: {dist_path}", priority=2
            )
            send_telegram_alert(
                f"Distribution des récompenses sauvegardée: {dist_path}"
            )

            latency = time.time() - start_time
            self.log_performance("plot_real_time_metrics", latency, success=True)
        except Exception as e:
            send_alert(
                f"Erreur génération graphiques en temps réel: {str(e)}", priority=3
            )
            send_telegram_alert(f"Erreur génération graphiques en temps réel: {str(e)}")
            logger.error(f"Erreur génération graphiques en temps réel: {str(e)}")
            self.sierra_error_manager.log_error(
                "PLOT_METRICS_FAIL",
                f"Erreur génération graphiques: {str(e)}",
                severity="ERROR",
            )
            self.log_performance(
                "plot_real_time_metrics",
                time.time() - start_time,
                success=False,
                error=str(e),
            )

    def analyze_real_time(self, trades: List[Dict], metrics: Dict) -> Dict:
        """
        Analyse les trades en temps réel, incluant l’analyse SHAP.

        Args:
            trades (List[Dict]): Liste des trades.
            metrics (Dict): Métriques actuelles.

        Returns:
            Dict: Seuils ajustés si nécessaire, incluant métriques SHAP.
        """
        start_time = time.time()
        try:
            logger.info("Analyse des trades en temps réel")
            send_alert("Analyse des trades en temps réel", priority=2)
            send_telegram_alert("Analyse des trades en temps réel")

            trades_df = pd.DataFrame(trades)
            result_analysis = analyze_results(trades_df)
            trade_analysis = analyze_trades(trades_df)

            # Analyse SHAP (Phase 17)
            shap_metrics = {}
            if not trades_df.empty and "reward" in trades_df.columns:

                def run_shap():
                    return calculate_shap(
                        trades_df, target="reward", max_features=SHAP_FEATURE_LIMIT
                    )

                shap_values = self.with_retries(run_shap)
                if shap_values is not None:
                    shap_metrics = {
                        f"shap_{col}": float(shap_values[col].mean())
                        for col in shap_values.columns
                    }
                    # Ajout des nouvelles features dans l'analyse SHAP
                    for feature in [
                        "bid_ask_imbalance",
                        "trade_aggressiveness",
                        "iv_skew",
                        "iv_term_structure",
                    ]:
                        if feature in shap_values.columns:
                            shap_metrics[
                                f"shap_{feature}"
                            ] *= 1.2  # Poids accru pour les nouvelles features
                    self.save_trade_snapshot(
                        step=len(trades),
                        timestamp=pd.Timestamp.now(),
                        action=0.0,
                        regime=result_analysis.get("regime", "range"),
                        reward=0.0,
                        confidence=0.5,
                        trade_success_prob=0.0,
                        df_row=pd.Series(shap_metrics),
                        risk_metrics={},
                        decision={"status": "shap_analysis"},
                        vocal_command=None,
                        compress=False,
                    )
                else:
                    error_msg = "Échec calcul SHAP, poursuite sans SHAP"
                    send_alert(error_msg, priority=3)
                    send_telegram_alert(error_msg)
                    logger.warning(error_msg)

            adjusted_thresholds = {}
            if (
                result_analysis.get("sharpe", 0.0)
                < self.performance_thresholds["min_sharpe"]
            ):
                adjusted_thresholds["min_sharpe"] = (
                    self.performance_thresholds["min_sharpe"] * 0.9
                )
                send_alert(
                    f"Ajustement min_sharpe à {adjusted_thresholds['min_sharpe']:.2f} en raison de faible performance",
                    priority=3,
                )
                send_telegram_alert(
                    f"Ajustement min_sharpe à {adjusted_thresholds['min_sharpe']:.2f} en raison de faible performance"
                )
                logger.warning(
                    f"Ajustement min_sharpe à {adjusted_thresholds['min_sharpe']:.2f}"
                )

            if (
                trade_analysis.get("profit_factor", 0.0)
                < self.performance_thresholds["min_profit_factor"]
            ):
                adjusted_thresholds["min_profit_factor"] = (
                    self.performance_thresholds["min_profit_factor"] * 0.9
                )
                send_alert(
                    f"Ajustement min_profit_factor à {adjusted_thresholds['min_profit_factor']:.2f} en raison de faible performance",
                    priority=3,
                )
                send_telegram_alert(
                    f"Ajustement min_profit_factor à {adjusted_thresholds['min_profit_factor']:.2f} en raison de faible performance"
                )
                logger.warning(
                    f"Ajustement min_profit_factor à {adjusted_thresholds['min_profit_factor']:.2f}"
                )

            adjusted_thresholds["shap_metrics"] = shap_metrics
            self.log_performance(
                "analyze_real_time", time.time() - start_time, success=True
            )
            self.save_checkpoint(
                trades, result_analysis.get("regime", "range"), compress=True
            )
            return adjusted_thresholds
        except Exception as e:
            send_alert(f"Erreur analyse en temps réel: {str(e)}", priority=3)
            send_telegram_alert(f"Erreur analyse en temps réel: {str(e)}")
            logger.error(f"Erreur analyse en temps réel: {str(e)}")
            self.sierra_error_manager.log_error(
                "ANALYZE_RT_FAIL",
                f"Erreur analyse en temps réel: {str(e)}",
                severity="ERROR",
            )
            self.log_performance(
                "analyze_real_time",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            return {"shap_metrics": {}}

    def compute_metrics(self, rewards: List[float]) -> Dict[str, float]:
        """
        Calcule les métriques en temps réel, incluant confidence_drop_rate.

        Args:
            rewards (List[float]): Liste des récompenses.

        Returns:
            Dict[str, float]: Métriques calculées.
        """
        start_time = time.time()
        try:
            logger.info("Calcul des métriques en temps réel")
            send_alert("Calcul des métriques en temps réel", priority=2)
            send_telegram_alert("Calcul des métriques en temps réel")

            if not rewards:
                return {
                    "sharpe": 0.0,
                    "drawdown": 0.0,
                    "total_reward": 0.0,
                    "profit_factor": 0.0,
                    "hourly_drawdown": 0.0,
                    "daily_sharpe": 0.0,
                    "trades_executed": 0,
                    "confidence_drop_rate": 0.0,
                }

            equity = np.cumsum(rewards)
            drawdown = float(np.min(equity - np.maximum.accumulate(equity)))
            sharpe = (
                float(np.mean(rewards) / np.std(rewards))
                if np.std(rewards) > 0
                else 0.0
            )
            positive_rewards = sum(r for r in rewards if r > 0)
            negative_rewards = abs(sum(r for r in rewards if r < 0))
            profit_factor = (
                positive_rewards / negative_rewards
                if negative_rewards > 0
                else float("inf")
            )
            confidence_drop_rate = 1.0 - min(
                profit_factor / self.performance_thresholds["min_profit_factor"], 1.0
            )  # Phase 8

            df_rewards = pd.DataFrame(
                {
                    "reward": rewards,
                    "timestamp": [
                        datetime.now() - timedelta(minutes=len(rewards) - i)
                        for i in range(len(rewards))
                    ],
                }
            )
            df_rewards["timestamp"] = pd.to_datetime(df_rewards["timestamp"])
            hourly_drawdown = (
                df_rewards.groupby(df_rewards["timestamp"].dt.floor("H"))["reward"]
                .sum()
                .min()
            )
            daily_sharpe = (
                df_rewards.groupby(df_rewards["timestamp"].dt.floor("D"))
                .apply(
                    lambda x: (
                        x["reward"].mean() / x["reward"].std()
                        if x["reward"].std() > 0
                        else 0.0
                    )
                )
                .mean()
            )

            metrics = {
                "sharpe": sharpe,
                "drawdown": drawdown,
                "total_reward": float(np.sum(rewards)),
                "profit_factor": profit_factor,
                "hourly_drawdown": (
                    float(hourly_drawdown) if not pd.isna(hourly_drawdown) else 0.0
                ),
                "daily_sharpe": (
                    float(daily_sharpe) if not pd.isna(daily_sharpe) else 0.0
                ),
                "trades_executed": len(rewards),
                "confidence_drop_rate": confidence_drop_rate,
            }

            for metric, threshold in self.performance_thresholds.items():
                if metric in metrics:
                    value = metrics.get(metric, 0)
                    if (metric == "max_drawdown" and value < threshold) or (
                        metric != "max_drawdown" and value < threshold
                    ):
                        send_alert(
                            f"Seuil non atteint pour {metric}: {value} < {threshold}",
                            priority=3,
                        )
                        send_telegram_alert(
                            f"Seuil non atteint pour {metric}: {value} < {threshold}"
                        )
                        logger.warning(
                            f"Seuil non atteint pour {metric}: {value} < {threshold}"
                        )

            self.log_performance(
                "compute_metrics",
                time.time() - start_time,
                success=True,
                num_trades=len(rewards),
            )
            return metrics
        except Exception as e:
            send_alert(f"Erreur calcul métriques: {str(e)}", priority=4)
            send_telegram_alert(f"Erreur calcul métriques: {str(e)}")
            logger.error(f"Erreur calcul métriques: {str(e)}")
            self.sierra_error_manager.log_error(
                "METRICS_FAIL", f"Erreur calcul métriques: {str(e)}", severity="ERROR"
            )
            self.log_performance(
                "compute_metrics", time.time() - start_time, success=False, error=str(e)
            )
            return {
                "sharpe": 0.0,
                "drawdown": 0.0,
                "total_reward": 0.0,
                "profit_factor": 0.0,
                "hourly_drawdown": 0.0,
                "daily_sharpe": 0.0,
                "trades_executed": 0,
                "confidence_drop_rate": 0.0,
            }

    def save_trade_snapshot(
        self,
        step: int,
        timestamp: pd.Timestamp,
        action: float,
        regime: str,
        reward: float,
        confidence: float,
        trade_success_prob: float,
        df_row: pd.Series,
        risk_metrics: Dict,
        decision: Dict,
        vocal_command: Optional[str] = None,
        output_dir: str = str(SNAPSHOT_DIR),
        compress: bool = False,
    ) -> None:
        """
        Sauvegarde un instantané JSON du trade, avec option de compression.

        Args:
            step (int): Étape du trading.
            timestamp (pd.Timestamp): Horodatage du trade.
            action (float): Action effectuée.
            regime (str): Régime détecté.
            reward (float): Récompense obtenue.
            confidence (float): Confiance de la prédiction.
            trade_success_prob (float): Probabilité de succès du trade.
            df_row (pd.Series): Ligne de données.
            risk_metrics (Dict): Métriques de risque.
            decision (Dict): Décision prise.
            vocal_command (Optional[str]): Commande vocale (si applicable).
            output_dir (str): Répertoire de sortie.
            compress (bool): Si True, compresse en gzip.
        """
        start_time = time.time()
        try:
            logger.info(f"Sauvegarde instantané trade pour étape {step}")
            send_alert(f"Sauvegarde instantané trade pour étape {step}", priority=2)
            send_telegram_alert(f"Sauvegarde instantané trade pour étape {step}")

            snapshot = {
                "step": step,
                "timestamp": str(timestamp),
                "action": float(action),
                "regime": regime,
                "reward": float(reward),
                "confidence": float(confidence),
                "trade_success_prob": float(trade_success_prob),
                "features": df_row.to_dict(),
                "risk_metrics": risk_metrics,
                "decision": decision,
                "vocal_command": vocal_command,
            }
            snapshot_path = Path(output_dir) / f"trade_step_{step:04d}.json"

            def write_snapshot():
                if compress:
                    with gzip.open(f"{snapshot_path}.gz", "wt", encoding="utf-8") as f:
                        json.dump(snapshot, f, indent=4)
                else:
                    with open(snapshot_path, "w", encoding="utf-8") as f:
                        json.dump(snapshot, f, indent=4)

            self.with_retries(write_snapshot)

            save_path = f"{snapshot_path}.gz" if compress else snapshot_path
            self.log_performance(
                "save_trade_snapshot", time.time() - start_time, success=True, step=step
            )
            logger.info(f"Instantané trade sauvegardé: {save_path}")
            send_alert(f"Instantané trade sauvegardé: {save_path}", priority=2)
            send_telegram_alert(f"Instantané trade sauvegardé: {save_path}")
        except Exception as e:
            send_alert(
                f"Erreur sauvegarde instantané trade étape {step}: {str(e)}", priority=3
            )
            send_telegram_alert(
                f"Erreur sauvegarde instantané trade étape {step}: {str(e)}"
            )
            logger.error(f"Erreur sauvegarde instantané trade étape {step}: {str(e)}")
            self.sierra_error_manager.log_error(
                "SNAPSHOT_FAIL",
                f"Erreur sauvegarde instantané trade étape {step}: {str(e)}",
                severity="ERROR",
            )
            self.log_performance(
                "save_trade_snapshot",
                time.time() - start_time,
                success=False,
                error=str(e),
            )

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ) -> None:
        """
        Journalise les performances des opérations critiques.

        Args:
            operation (str): Nom de l’opération.
            latency (float): Latence en secondes.
            success (bool): Succès de l’opération.
            error (str, optional): Message d’erreur si échec.
            **kwargs: Paramètres supplémentaires (ex. : num_trades, step).
        """
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
            cpu_percent = psutil.cpu_percent()
            if memory_usage > 1024:
                send_alert(
                    f"ALERTE: Usage mémoire élevé ({memory_usage:.2f} MB)", priority=5
                )
                send_telegram_alert(
                    f"ALERTE: Usage mémoire élevé ({memory_usage:.2f} MB)"
                )
                logger.warning(f"Usage mémoire élevé: {memory_usage:.2f} MB")
                self.sierra_error_manager.log_error(
                    "HIGH_MEMORY",
                    f"Usage mémoire élevé: {memory_usage:.2f} MB",
                    severity="WARNING",
                )
            log_entry = {
                "timestamp": str(datetime.now()),
                "operation": operation,
                "latency": latency,
                "success": success,
                "error": error,
                "memory_usage_mb": memory_usage,
                "cpu_percent": cpu_percent,
                **kwargs,
            }
            self.log_buffer.append(log_entry)
            if len(self.log_buffer) >= self.config.get("logging", {}).get(
                "buffer_size", 200
            ):
                log_df = pd.DataFrame(self.log_buffer)

                def save_log():
                    if not CSV_LOG_PATH.exists():
                        log_df.to_csv(CSV_LOG_PATH, index=False, encoding="utf-8")
                    else:
                        log_df.to_csv(
                            CSV_LOG_PATH,
                            mode="a",
                            header=False,
                            index=False,
                            encoding="utf-8",
                        )

                self.with_retries(save_log)
                self.log_buffer = []
            logger.info(
                f"Performance journalisée pour {operation}. CPU: {cpu_percent}%"
            )
        except Exception as e:
            send_alert(f"Erreur journalisation performance: {str(e)}", priority=3)
            send_telegram_alert(f"Erreur journalisation performance: {str(e)}")
            logger.error(f"Erreur journalisation performance: {str(e)}")
            self.sierra_error_manager.log_error(
                "LOG_PERF_FAIL",
                f"Erreur journalisation performance: {str(e)}",
                severity="ERROR",
            )

    async def run(
        self,
        feature_file: str = "data/features/features_latest_filtered.csv",
        live_mode: bool = True,
        max_steps: int = 1000,
    ) -> None:
        """
        Exécute la boucle principale de trading en live ou en mode paper.

        Args:
            feature_file (str): Chemin du fichier de features.
            live_mode (bool): Mode live (True) ou paper (False).
            max_steps (int): Nombre maximum d'étapes de trading.
        """
        start_time = time.time()
        try:
            self.running = True
            self.live_mode = live_mode
            logger.info(f"Démarrage boucle trading {'live' if live_mode else 'paper'}")
            send_alert(
                f"Démarrage boucle trading {'live' if live_mode else 'paper'}",
                priority=2,
            )
            send_telegram_alert(
                f"Démarrage boucle trading {'live' if live_mode else 'paper'}"
            )

            risk_controller = RiskController()
            trade_window_filter = TradeWindowFilter()
            decision_log = DecisionLog()
            trade_probability = TradeProbabilityPredictor()
            rewards_history = []
            timestamps = []
            trades = []
            balance = 0.0
            positions = []
            current_step = 0

            while self.running and current_step < max_steps:
                try:
                    # Charger les données
                    current_data = await self.load_data(feature_file)
                    if current_data.empty:
                        logger.warning("Données vides, passage à l'itération suivante")
                        send_alert(
                            "Données vides, passage à l'itération suivante", priority=3
                        )
                        send_telegram_alert(
                            "Données vides, passage à l'itération suivante"
                        )
                        continue

                    # Sélectionner le modèle
                    model_info = self.select_model(current_data, current_step)

                    # Prédire l'action
                    @lru_cache(maxsize=100)
                    def cached_predict(data_tuple):
                        return predict(pd.DataFrame([data_tuple]), model_info["policy"])

                    data_tuple = tuple(current_data.iloc[-1].to_dict().items())
                    prediction = cached_predict(data_tuple)
                    action = float(prediction.get("action", 0.0))
                    confidence = float(prediction.get("confidence", 0.5))
                    trade_success_prob = trade_probability.predict(current_data)

                    # Valider la fenêtre de trading et les risques
                    if trade_window_filter.check_trade_window(
                        trade_success_prob, current_data, current_step
                    ):
                        decision_log.log_decision(
                            trade_id=f"step_{current_step}",
                            decision="block",
                            score=trade_success_prob,
                            reason="Fenêtre de trading inappropriée",
                            data=current_data,
                        )
                        continue

                    risk_metrics = risk_controller.calculate_risk_metrics(
                        current_data, positions
                    )
                    options_risk = self.options_risk_manager.calculate_options_risk(
                        current_data
                    )
                    if (
                        risk_metrics["market_liquidity_crash_risk"]
                        > self.performance_thresholds["market_liquidity_crash_risk"]
                        or options_risk["risk_alert"].any()
                        or confidence < self.performance_thresholds["min_confidence"]
                    ):
                        decision_log.log_decision(
                            trade_id=f"step_{current_step}",
                            decision="block",
                            score=confidence,
                            reason="Risque élevé ou faible confiance",
                            data=current_data,
                        )
                        continue

                    # Exécuter le trade
                    trade = {
                        "trade_id": f"step_{current_step}",
                        "action": action,
                        "price": current_data["close"].iloc[-1],
                        "size": min(
                            risk_metrics["position_size"],
                            self.performance_thresholds["max_position_size"],
                        ),
                        "order_type": "market",
                    }

                    def execute_trade():
                        return self.trade_executor.execute_trade(
                            trade,
                            mode="real" if live_mode else "paper",
                            context_data=current_data,
                        )

                    execution_result = self.with_retries(execute_trade)
                    if (
                        execution_result is None
                        or execution_result["status"] != "filled"
                    ):
                        decision_log.log_decision(
                            trade_id=trade["trade_id"],
                            decision="fail",
                            score=confidence,
                            reason="Échec de l'exécution du trade",
                            data=current_data,
                        )
                        continue

                    # Confirmer et journaliser le trade
                    confirmation = self.trade_executor.confirm_trade(trade["trade_id"])
                    if confirmation["confirmed"]:
                        self.trade_executor.log_execution(
                            trade, execution_result, is_paper=not live_mode
                        )
                        reward = execution_result.get("profit", 0.0)
                        balance += reward
                        positions.append(trade)
                        self.positions = positions
                        rewards_history.append(reward)
                        timestamps.append(pd.Timestamp.now())
                        trades.append(
                            {
                                "trade_id": trade["trade_id"],
                                "action": action,
                                "reward": reward,
                                "confidence": confidence,
                                "trade_success_prob": trade_success_prob,
                                "timestamp": str(pd.Timestamp.now()),
                            }
                        )

                        # Sauvegarder le snapshot
                        self.save_trade_snapshot(
                            step=current_step,
                            timestamp=pd.Timestamp.now(),
                            action=action,
                            regime=current_data.get("neural_regime", "range").iloc[-1],
                            reward=reward,
                            confidence=confidence,
                            trade_success_prob=trade_success_prob,
                            df_row=current_data.iloc[-1],
                            risk_metrics=risk_metrics,
                            decision={
                                "status": "executed",
                                "trade_id": trade["trade_id"],
                            },
                            vocal_command=None,
                            compress=True,
                        )

                        # Stocker le pattern
                        if self.validate_pattern_data(
                            current_data,
                            action,
                            reward,
                            current_data.get("neural_regime", "range").iloc[-1],
                            confidence,
                        ):
                            store_pattern(
                                current_data,
                                action,
                                reward,
                                current_data.get("neural_regime", "range").iloc[-1],
                                confidence,
                            )

                    # Analyser les métriques
                    metrics = self.compute_metrics(rewards_history)
                    adjusted_thresholds = self.analyze_real_time(trades, metrics)
                    self.performance_thresholds.update(adjusted_thresholds)

                    # Générer les graphiques
                    self.plot_real_time_metrics(rewards_history, timestamps)

                    # Gérer les commandes vocales
                    stop_requested = await self.handle_vocal_commands(
                        current_data,
                        current_step,
                        balance,
                        positions,
                        risk_controller,
                        trade_window_filter,
                        decision_log,
                        live_mode,
                    )
                    if stop_requested:
                        self.stop_trading()
                        break

                    current_step += 1

                except Exception as e:
                    send_alert(
                        f"Erreur boucle trading étape {current_step}: {str(e)}",
                        priority=4,
                    )
                    send_telegram_alert(
                        f"Erreur boucle trading étape {current_step}: {str(e)}"
                    )
                    logger.error(
                        f"Erreur boucle trading étape {current_step}: {str(e)}"
                    )
                    self.sierra_error_manager.log_error(
                        "TRADING_LOOP_FAIL",
                        f"Erreur boucle trading: {str(e)}",
                        severity="ERROR",
                    )
                    continue

            self.stop_trading()
            logger.info(f"Boucle trading terminée après {current_step} étapes")
            send_alert(
                f"Boucle trading terminée après {current_step} étapes", priority=2
            )
            send_telegram_alert(f"Boucle trading terminée après {current_step} étapes")
            self.log_performance(
                "run", time.time() - start_time, success=True, num_steps=current_step
            )
        except Exception as e:
            send_alert(f"Erreur critique boucle trading: {str(e)}", priority=5)
            send_telegram_alert(f"Erreur critique boucle trading: {str(e)}")
            logger.error(f"Erreur critique boucle trading: {str(e)}")
            self.sierra_error_manager.log_error(
                "RUN_FAIL",
                f"Erreur critique boucle trading: {str(e)}",
                severity="CRITICAL",
            )
            self.log_performance(
                "run", time.time() - start_time, success=False, error=str(e)
            )
            self.stop_trading()

    def stop_trading(self) -> None:
        """
        Termine proprement la boucle de trading, sauvegarde les snapshots et ferme les positions ouvertes.
        """
        start_time = time.time()
        try:
            self.running = False
            logger.info("Arrêt du trading en cours")
            send_alert("Arrêt du trading en cours", priority=2)
            send_telegram_alert("Arrêt du trading en cours")

            # Sauvegarder un snapshot final
            self.save_trade_snapshot(
                step=9999,  # Étape spéciale pour l'arrêt
                timestamp=pd.Timestamp.now(),
                action=0.0,
                regime="stopped",
                reward=0.0,
                confidence=0.0,
                trade_success_prob=0.0,
                df_row=pd.Series({"status": "stopped"}),
                risk_metrics={},
                decision={"status": "stopped"},
                vocal_command=None,
                compress=True,
            )

            # Fermer les positions ouvertes
            for position in self.positions:
                trade = {
                    "trade_id": f"close_{position['trade_id']}",
                    "action": -position["action"],
                    "price": position["price"],
                    "size": position["size"],
                    "order_type": "market",
                }

                def close_position():
                    execution_result = self.trade_executor.execute_trade(
                        trade,
                        mode="real" if self.live_mode else "paper",
                        context_data=pd.DataFrame(),
                    )
                    if execution_result["status"] == "filled":
                        self.trade_executor.log_execution(
                            trade, execution_result, is_paper=not self.live_mode
                        )

                self.with_retries(close_position)
            self.positions = []

            # Vider le cache de prédictions
            self.prediction_cache.clear()
            self.select_model.cache_clear()

            logger.info("Trading arrêté proprement")
            send_alert("Trading arrêté proprement", priority=2)
            send_telegram_alert("Trading arrêté proprement")
            self.log_performance("stop_trading", time.time() - start_time, success=True)
        except Exception as e:
            send_alert(f"Erreur arrêt trading: {str(e)}", priority=3)
            send_telegram_alert(f"Erreur arrêt trading: {str(e)}")
            logger.error(f"Erreur arrêt trading: {str(e)}")
            self.sierra_error_manager.log_error(
                "STOP_TRADING_FAIL", f"Erreur arrêt trading: {str(e)}", severity="ERROR"
            )
            self.log_performance(
                "stop_trading", time.time() - start_time, success=False, error=str(e)
            )
