# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/strategy/mia_switcher.py
# Rôle : Arbitre entre plusieurs modèles MIA (ex. : SAC, PPO, DDPG, LSTM) pour
# sélectionner la stratégie optimale dans MIA_IA_SYSTEM_v2_2025.
#
# Version : 2.1.5
# Date : 2025-05-14
#
# Rôle : Sélectionne le modèle MIA optimal en fonction des conditions de marché,
# avec validation des données, journalisation des performances, snapshots
# compressés, et alertes.
# Intègre le position sizing dynamique, la microstructure, HMM, et un vote
# bayésien pour les ensembles de politiques.
# Conforme à la Phase 8 (auto-conscience via alertes), Phase 12 (simulation de
# trading), et Phase 16 (ensemble learning).
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0
# - numpy>=1.26.4,<2.0.0
# - psutil>=5.9.0,<6.0.0
# - pyyaml>=6.0.0,<7.0.0
# - logging, os, json, hashlib, datetime, time, signal, gzip
# - src.model.adaptive_learner
# - src.model.utils.config_manager
# - src.model.utils.alert_manager
# - src.utils.telegram_alert
# - src.envs.trading_env
# - src.model.inference
# - src.model.router.detect_regime
# - src.model.utils.performance_logger
# - src.model.utils.switch_logger
# - src.risk_management.risk_manager
# - src.utils.error_tracker
# - src.utils.mlflow_tracker
# - src.monitoring.prometheus_metrics
#
# Inputs :
# - Données de trading (pd.DataFrame avec 350 features pour entraînement ou 150
#   SHAP features pour inférence)
# - Configuration via config/market_config.yaml, config/models_config.yaml, et
#   config/router_config.yaml
# - config/feature_sets.yaml pour la validation des colonnes
#
# Outputs :
# - Logs dans data/logs/strategy/mia_switcher.log
# - Logs de performance dans data/logs/strategy/mia_switcher_performance.csv
# - Snapshots JSON compressés dans
#   data/strategy/mia_switcher_snapshots/*.json.gz
# - Dashboard JSON dans data/strategy/mia_switcher_dashboard.json
# - Logs de bascule dans data/logs/strategy/mia_switcher_log.csv
#
# Notes :
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence)
#   définies dans config/feature_sets.yaml.
# - Utilise IQFeed exclusivement pour les données d’entrée.
# - Supprime toute référence à 320 features, 81 features, et obs_t.
# - Implémente retries (max 3, délai 2^attempt secondes) pour les opérations
#   critiques.
# - Intègre le contexte de marché via detect_regime.py.
# - Tests unitaires disponibles dans tests/test_mia_switcher.py.
# - Policies Note: The official directory for routing policies is
#   src/model/router/policies.
# - The src/model/policies directory is a residual and should be verified for
#   removal to avoid import conflicts.
# - Mise à jour pour intégrer le position sizing dynamique, la microstructure,
#   HMM, vote bayésien, métriques Prometheus, logs psutil renforcés, et gestion
#   des erreurs/alertes.

import gzip
import hashlib
import json
import logging
import os
import signal
import sqlite3
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil

from src.envs.trading_env import TradingEnv
from src.model.adaptive_learner import store_pattern
from src.model.router.detect_regime import MarketRegimeDetector
from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import config_manager, get_features
from src.model.utils.performance_logger import PerformanceLogger
from src.model.utils.switch_logger import SwitchLogger
from src.monitoring.prometheus_metrics import Gauge
from src.risk_management.risk_manager import RiskManager
from src.utils.error_tracker import capture_error
from src.utils.mlflow_tracker import MLFlowTracker
from src.utils.telegram_alert import send_telegram_alert

# Chemins relatifs
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOG_DIR = BASE_DIR / "data" / "logs" / "strategy"
SNAPSHOT_DIR = BASE_DIR / "data" / "strategy" / "mia_switcher_snapshots"
PERF_LOG_PATH = LOG_DIR / "mia_switcher_performance.csv"

# Configuration du logging
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "mia_switcher.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Constantes
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Secondes, exponentiel

# Métriques Prometheus
ensemble_weight_sac = Gauge(
    name="ensemble_weight_sac",
    description="Poids SAC dans le vote bayésien",
    labelnames=["market"],
)
ensemble_weight_ppo = Gauge(
    name="ensemble_weight_ppo",
    description="Poids PPO dans le vote bayésien",
    labelnames=["market"],
)
ensemble_weight_ddpg = Gauge(
    name="ensemble_weight_ddpg",
    description="Poids DDPG dans le vote bayésien",
    labelnames=["market"],
)


class MiaSwitcher:
    """
    Classe pour arbitrer entre plusieurs modèles MIA (ex. : SAC, PPO, DDPG,
    LSTM) en fonction des conditions de marché et des stratégies.
    """

    def __init__(
        self,
        config_path: str = "config/market_config.yaml",
        models_config_path: str = "config/models_config.yaml",
        config: Optional[Dict[str, Any]] = None,
        config_manager_instance: Optional[Any] = None,
    ):
        """
        Initialise le switcher de modèles MIA.

        Args:
            config_path (str): Chemin vers la configuration du marché.
            models_config_path (str): Chemin vers la configuration des modèles.
            config (Dict[str, Any], optional): Configuration dictionnaire.
            config_manager_instance (Any, optional): Instance de ConfigManager.
        """
        start_time = datetime.now()
        try:
            self.alert_manager = AlertManager()
            SNAPSHOT_DIR.mkdir(exist_ok=True)
            PERF_LOG_PATH.parent.mkdir(exist_ok=True)
            signal.signal(signal.SIGINT, self.handle_sigint)

            # Initialiser RiskManager pour le position sizing
            self.risk_manager = RiskManager(market="ES")

            # Charger la configuration du marché
            if config_manager_instance:
                config = config_manager_instance.get_config(config_path)
            elif config:
                config = config
            else:
                config = config_manager.get_config(
                    os.path.join(BASE_DIR, config_path)
                ).get("mia_switcher", {})
            required_config = ["thresholds", "weights", "cache", "logging", "features"]
            missing_config = [
                key for key in required_config if key not in config
            ]
            if missing_config:
                error_msg = (
                    f"Clés de configuration manquantes dans "
                    f"market_config.yaml: {missing_config}"
                )
                raise ValueError(error_msg)

            self.thresholds = {
                "min_sharpe": config["thresholds"].get("min_sharpe", 0.5),
                "max_drawdown": config["thresholds"].get("max_drawdown", -0.1),
                "min_profit_factor": config["thresholds"].get(
                    "min_profit_factor", 1.2
                ),
                "vix_threshold": config["thresholds"].get("vix_threshold", 30),
                "switch_confidence_threshold": config["thresholds"].get(
                    "switch_confidence_threshold", 0.8
                ),
                "regime_switch_frequency": config["thresholds"].get(
                    "regime_switch_frequency", 300
                ),
                "max_consecutive_underperformance": config["thresholds"].get(
                    "max_consecutive_underperformance", 3
                ),
                "observation_dims": config["features"].get(
                    "observation_dims",
                    {"training": 350, "inference": 150},
                ),
            }
            self.score_weights = {
                "sharpe_weight": config["weights"].get("sharpe_weight", 0.5),
                "drawdown_weight": config["weights"].get("drawdown_weight", 0.3),
                "profit_factor_weight": config["weights"].get(
                    "profit_factor_weight", 0.2
                ),
            }
            self.max_cache_size = config.get("cache", {}).get(
                "max_cache_size", 1000
            )
            self.buffer_size = config.get("logging", {}).get("buffer_size", 100)
            self.evaluation_steps = config.get("evaluation_steps", 100)
            self.max_profit_factor = config.get("max_profit_factor", 10.0)

            # Valider les seuils
            for key, value in self.thresholds.items():
                if key != "observation_dims" and not isinstance(
                    value, (int, float)
                ):
                    raise ValueError(f"Seuil invalide pour {key}: {value}")
                if key == "max_drawdown" and value >= 0:
                    raise ValueError(f"max_drawdown doit être négatif: {value}")

            # Charger la configuration des modèles
            models_config = config_manager.get_config(
                os.path.join(BASE_DIR, models_config_path)
            ).get("models", {})
            self.models = models_config
            if not self.models:
                raise ValueError("Aucun modèle défini dans models_config.yaml")

            # Valider les modèles
            required_keys = [
                "model_type",
                "model_path",
                "policy_type",
            ]
            for model_name, model_config in self.models.items():
                missing_keys = [
                    key for key in required_keys if key not in model_config
                ]
                if missing_keys:
                    error_msg = (
                        f"Modèle {model_name} mal configuré: clés manquantes "
                        f"{missing_keys}"
                    )
                    raise ValueError(error_msg)

            # Chemins de sortie
            self.switcher_log_path = os.path.join(
                BASE_DIR,
                "data",
                "logs",
                "strategy",
                "mia_switcher_log.csv",
            )
            self.dashboard_path = os.path.join(
                BASE_DIR,
                "data",
                "strategy",
                "mia_switcher_dashboard.json",
            )

            # État interne
            self.current_model = list(self.models.keys())[0]
            self.last_switch_time = datetime.now() - timedelta(
                seconds=self.thresholds["regime_switch_frequency"]
            )
            self.model_performance = {
                name: {
                    "sharpe": 0.0,
                    "drawdown": 0.0,
                    "profit_factor": 0.0,
                    "rewards": [],
                    "underperformance_count": 0,
                }
                for name in self.models
            }
            self.switch_history = []
            self.log_buffer = []
            self.data_cache = {}
            self.prediction_cache = {}
            self.performance_buffer = PerformanceLogger(market="ES")
            self.switch_buffer = SwitchLogger(market="ES")

            # Initialiser TradingEnv
            self.env = TradingEnv(config_path=config_path)

            # Initialiser MarketRegimeDetector
            self.regime_detector = MarketRegimeDetector()

            latency = (datetime.now() - start_time).total_seconds()
            logger.info("MiaSwitcher initialisé avec succès")
            self.alert_manager.send_alert("MiaSwitcher initialisé", priority=2)
            send_telegram_alert("MiaSwitcher initialisé")
            self.log_performance(
                operation="init",
                latency=latency,
                success=True,
            )
        except Exception as e:
            error_msg = (
                f"Erreur initialisation MiaSwitcher: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                operation="init",
                latency=0,
                success=False,
                error=str(e),
            )
            capture_error(
                error=e,
                context={"market": "ES"},
                market="ES",
                operation="init",
            )
            raise

    def handle_sigint(self, signum, frame):
        """Gère l'arrêt propre sur SIGINT."""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "status": "SIGINT",
        }
        try:
            self.save_snapshot("sigint", snapshot)
            logger.info("Arrêt propre sur SIGINT, snapshot sauvegardé")
            self.alert_manager.send_alert(
                "Arrêt propre sur SIGINT, snapshot sauvegardé",
                priority=2,
            )
            send_telegram_alert("Arrêt propre sur SIGINT, snapshot sauvegardé")
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde snapshot SIGINT: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            capture_error(
                error=e,
                context={"market": "ES"},
                market="ES",
                operation="handle_sigint",
            )
        exit(0)

    def save_snapshot(self, snapshot_type: str, data: Dict) -> None:
        """Sauvegarde un instantané des résultats avec compression gzip."""
        start_time = time.time()
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = {
                "timestamp": timestamp,
                "type": snapshot_type,
                "data": data,
                "buffer_size": len(self.log_buffer),
            }
            path = SNAPSHOT_DIR / f"snapshot_{snapshot_type}_{timestamp}.json"
            os.makedirs(path.parent, exist_ok=True)
            snapshot_path = f"{path}.gz"
            with gzip.open(snapshot_path, "wt", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=4)
            latency = (datetime.now() - start_time).total_seconds()
            self.alert_manager.send_alert(
                message=f"Snapshot {snapshot_type} sauvegardé: {snapshot_path}",
                priority=1,
            )
            send_telegram_alert(
                message=f"Snapshot {snapshot_type} sauvegardé: {snapshot_path}"
            )
            logger.info(f"Snapshot {snapshot_type} sauvegardé: {snapshot_path}")
            self.log_performance(
                operation="save_snapshot",
                latency=latency,
                success=True,
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                operation="save_snapshot",
                latency=latency,
                success=False,
                error=str(e),
            )
            capture_error(
                error=e,
                context={"market": "ES"},
                market="ES",
                operation="save_snapshot",
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
                latency = (datetime.now() - start_time).total_seconds()
                self.log_performance(
                    operation=f"retry_attempt_{attempt+1}",
                    latency=latency,
                    success=True,
                )
                return result
            except Exception as e:
                if attempt == max_attempts - 1:
                    error_msg = (
                        f"Échec après {max_attempts} tentatives: {str(e)}\n"
                        f"{traceback.format_exc()}"
                    )
                    self.alert_manager.send_alert(error_msg, priority=3)
                    send_telegram_alert(error_msg)
                    logger.error(error_msg)
                    self.log_performance(
                        operation=f"retry_attempt_{attempt+1}",
                        latency=latency,
                        success=False,
                        error=str(e),
                    )
                    capture_error(
                        error=e,
                        context={"market": "ES"},
                        market="ES",
                        operation=f"retry_attempt_{attempt+1}",
                    )
                    return None
                delay = delay_base**attempt
                logger.warning(
                    f"Tentative {attempt+1} échouée, retry après {delay}s"
                )
                time.sleep(delay)

    def log_performance(
        self,
        operation: str,
        latency: float,
        success: bool,
        error: str = None,
        **kwargs,
    ) -> None:
        """
        Journalise les performances des opérations critiques avec psutil.

        Args:
            operation (str): Nom de l’opération.
            latency (float): Latence en secondes.
            success (bool): Succès de l’opération.
            error (str, optional): Message d’erreur si échec.
            **kwargs: Paramètres supplémentaires.
        """
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            if memory_usage > 1024:
                error_msg = (
                    f"ALERTE: Utilisation mémoire élevée "
                    f"({memory_usage:.2f} MB)"
                )
                self.alert_manager.send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                logger.warning(error_msg)
                capture_error(
                    error=Exception(error_msg),
                    context={"market": "ES"},
                    market="ES",
                    operation="log_performance",
                )
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "latency": latency,
                "success": success,
                "error": error,
                "memory_usage_mb": memory_usage,
                "cpu_usage_percent": cpu_percent,
                **kwargs,
            }
            os.makedirs(os.path.dirname(PERF_LOG_PATH), exist_ok=True)
            log_df = pd.DataFrame([log_entry])

            def write_log():
                log_df.to_csv(
                    PERF_LOG_PATH,
                    mode="a",
                    header=not PERF_LOG_PATH.exists(),
                    index=False,
                    encoding="utf-8",
                )

            self.with_retries(write_log)
            logger.info(
                f"Performance journalisée pour {operation}. "
                f"CPU: {cpu_percent}%"
            )
            self.performance_buffer.log(
                operation=operation,
                latency=latency,
                success=success,
                error=error,
                **kwargs,
            )
        except Exception as e:
            error_msg = (
                f"Erreur journalisation performance: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            capture_error(
                error=e,
                context={"market": "ES"},
                market="ES",
                operation="log_performance",
            )

    def validate_features(
        self,
        data: pd.DataFrame,
        context: str = "inference",
    ) -> bool:
        """Valide que les données contiennent les features attendues pour le
        contexte."""
        expected_features = get_features(context)
        missing_features = [
            f for f in expected_features if f not in data.columns
        ]
        if missing_features:
            logger.error(f"Features manquantes pour {context}: {missing_features}")
            return False
        return True

    def log_performance_metrics(
        self,
        operation: str,
        latency: float,
        success: bool = True,
        error: str = None,
        **kwargs,
    ) -> None:
        """Journalise les métriques de performance dans performance_buffer."""
        self.performance_buffer.log(
            operation=operation,
            latency=latency,
            success=success,
            error=error,
            **kwargs,
        )

    def log_switch_event(
        self,
        decision: str,
        reason: str,
        regime: str,
        **kwargs,
    ) -> None:
        """Journalise un événement de switching dans switch_buffer."""
        self.switch_buffer.log(
            decision=decision,
            reason=reason,
            regime=regime,
            **kwargs,
        )

    def generate_cache_key(
        self,
        market: str,
        timestamp: str,
        metrics: Dict[str, Any],
    ) -> str:
        """Génère une clé de cache légère."""
        critical_metrics = {
            k: v for k, v in metrics.items() if k in ["operation", "success"]
        }
        return f"{market}_{timestamp}_{hash(str(critical_metrics))}"

    def calculate_profit_factor(
        self,
        positive_rewards: float,
        negative_rewards: float,
        max_pf: float = 10.0,
    ) -> float:
        """Calcule le profit factor avec un cap pour éviter l’infini."""
        if negative_rewards == 0:
            return max_pf
        return min(positive_rewards / (negative_rewards + 1e-6), max_pf)

    def fallback_shap_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Charge une liste statique de 150 features SHAP si le cache est
        absent."""
        shap_features = get_features("inference")
        missing_features = [
            f for f in shap_features if f not in data.columns
        ]
        for feature in missing_features:
            data[feature] = 0.0
        return data

    def maybe_retrain_trade_probability(self) -> None:
        """Vérifie si ≥ 1000 trades sont enregistrés et déclenche le
        réentraînement."""
        start_time = datetime.now()
        try:
            with sqlite3.connect("data/market_memory_ES.db") as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM trade_patterns 
                    WHERE timestamp > (SELECT MAX(timestamp) FROM training_log)
                    """
                )
                trade_count = cursor.fetchone()[0]
            if trade_count >= 1000:
                from scripts.retrain_trade_probability import (
                    retrain_trade_probability,
                )

                retrain_trade_probability(market="ES")
                logger.info(
                    f"Réentraînement déclenché pour ES après "
                    f"{trade_count} trades"
                )
                self.alert_manager.send_alert(
                    message=(
                        f"Réentraînement déclenché pour ES après "
                        f"{trade_count} trades"
                    ),
                    priority=2,
                )
                send_telegram_alert(
                    message=(
                        f"Réentraînement déclenché pour ES après "
                        f"{trade_count} trades"
                    )
                )
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                operation="maybe_retrain_trade_probability",
                latency=latency,
                success=True,
                trade_count=trade_count,
            )
        except Exception as e:
            error_msg = (
                f"Erreur dans maybe_retrain_trade_probability: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                operation="maybe_retrain_trade_probability",
                latency=latency,
                success=False,
                error=str(e),
            )
            capture_error(
                error=e,
                context={"market": "ES"},
                market="ES",
                operation="maybe_retrain_trade_probability",
            )

    def validate_data(self, data: pd.DataFrame) -> None:
        """
        Valide les données d’entrée (350 features pour entraînement, 150 SHAP
        pour inférence).
        """
        start_time = time.time()
        try:
            cache_key = hashlib.sha256(data.to_json().encode()).hexdigest()
            if cache_key in self.data_cache:
                self.log_performance_metrics(
                    operation="validate_data_cache_hit",
                    latency=0,
                    success=True,
                )
                return

            # Validation dynamique des features
            data_dim = data.shape[1]
            context = "training" if data_dim >= 350 else "inference"
            if not self.validate_features(data, context=context):
                error_msg = (
                    f"Features invalides pour {context}: attendu "
                    f"{self.thresholds['observation_dims'][context]} features"
                )
                raise ValueError(error_msg)

            # Appliquer le fallback SHAP si nécessaire pour l’inférence
            if context == "inference" and data_dim < 150:
                data = self.fallback_shap_features(data)

            critical_cols = [
                "vix",
                "neural_regime",
                "predicted_volatility",
                "trade_frequency_1s",
                "close",
                "atr_dynamic",
                "orderflow_imbalance",
                "bid_ask_imbalance",
                "trade_aggressiveness",
                "regime_hmm",
            ]
            for col in critical_cols:
                if col in data.columns:
                    if data[col].isnull().any():
                        error_msg = f"Colonne {col} contient des NaN"
                        raise ValueError(error_msg)
                    if col != "neural_regime" and not pd.api.types.is_numeric_dtype(
                        data[col]
                    ):
                        error_msg = (
                            f"Colonne {col} n'est pas numérique: "
                            f"{data[col].dtype}"
                        )
                        raise ValueError(error_msg)

            if "timestamp" in data.columns:
                data["timestamp"] = pd.to_datetime(
                    data["timestamp"],
                    errors="coerce",
                )
                if data["timestamp"].isna().any():
                    last_valid = (
                        data["timestamp"].dropna().iloc[-1]
                        if not data["timestamp"].dropna().empty
                        else pd.Timestamp.now()
                    )
                    data["timestamp"] = data["timestamp"].fillna(last_valid)
                    self.alert_manager.send_alert(
                        message=(
                            f"Valeurs de timestamp invalides détectées, "
                            f"imputées avec {last_valid}"
                        ),
                        priority=2,
                    )
                    send_telegram_alert(
                        message=(
                            f"Valeurs de timestamp invalides détectées, "
                            f"imputées avec {last_valid}"
                        )
                    )
                latest_timestamp = data["timestamp"].iloc[-1]
                if not isinstance(latest_timestamp, pd.Timestamp):
                    error_msg = f"Timestamp non valide: {latest_timestamp}"
                    raise ValueError(error_msg)
                time_check = (
                    latest_timestamp > datetime.now() + timedelta(minutes=5)
                    or latest_timestamp < datetime.now() - timedelta(hours=24)
                )
                if time_check:
                    error_msg = f"Timestamp hors plage: {latest_timestamp}"
                    raise ValueError(error_msg)

            for col in data.columns:
                if col not in critical_cols and data[col].isnull().any():
                    data[col] = data[col].fillna(data[col].median())

            self.data_cache[cache_key] = True
            if len(self.data_cache) > self.max_cache_size:
                self.data_cache.pop(next(iter(self.data_cache)))

            latency = (datetime.now() - start_time).total_seconds()
            logger.debug("Données validées avec succès")
            self.alert_manager.send_alert(
                message="Données validées avec succès",
                priority=1,
            )
            send_telegram_alert(message="Données validées avec succès")
            self.log_performance(
                operation="validate_data",
                latency=latency,
                success=True,
                num_features=data_dim,
            )
            self.save_snapshot(
                snapshot_type="validate_data",
                data={"num_features": data_dim},
            )
        except Exception as e:
            error_msg = (
                f"Erreur validation données: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                operation="validate_data",
                latency=(datetime.now() - start_time).total_seconds(),
                success=False,
                error=str(e),
            )
            capture_error(
                error=e,
                context={"market": "ES"},
                market="ES",
                operation="validate_data",
            )
            raise

    def compute_model_metrics(
        self,
        model_name: str,
        rewards: List[float],
    ) -> Dict[str, float]:
        """
        Calcule les métriques de performance pour un modèle.
        """
        start_time = time.time()
        try:
            if not rewards:
                return {
                    "sharpe": 0.0,
                    "drawdown": 0.0,
                    "profit_factor": 0.0,
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
            profit_factor = self.calculate_profit_factor(
                positive_rewards=positive_rewards,
                negative_rewards=negative_rewards,
                max_pf=self.max_profit_factor,
            )

            metrics = {
                "sharpe": sharpe,
                "drawdown": drawdown,
                "profit_factor": profit_factor,
            }

            logger.debug(f"Métriques calculées pour {model_name}: {metrics}")
            self.alert_manager.send_alert(
                message=f"Métriques calculées pour {model_name}: {metrics}",
                priority=1,
            )
            send_telegram_alert(
                message=f"Métriques calculées pour {model_name}: {metrics}"
            )
            self.log_performance(
                operation="compute_model_metrics",
                latency=(datetime.now() - start_time).total_seconds(),
                success=True,
                model_name=model_name,
            )
            self.save_snapshot(
                snapshot_type="compute_model_metrics",
                data=metrics,
            )
            return metrics
        except Exception as e:
            error_msg = (
                f"Erreur calcul métriques pour {model_name}: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                operation="compute_model_metrics",
                latency=(datetime.now() - start_time).total_seconds(),
                success=False,
                error=str(e),
            )
            capture_error(
                error=e,
                context={"market": "ES"},
                market="ES",
                operation="compute_model_metrics",
            )
            return {
                "sharpe": 0.0,
                "drawdown": 0.0,
                "profit_factor": 0.0,
            }

    def evaluate_model(
        self,
        model_name: str,
        data: pd.DataFrame,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Évalue la performance d’un modèle sur les données fournies.
        """
        start_time = time.time()
        try:
            self.validate_data(data)
            model_config = self.models[model_name]

            # Détecter le régime de marché
            regime, regime_probs = (
                self.regime_detector.detect_market_regime_vectorized(data, 0)
            )
            valid_regimes = {"trend", "range", "defensive"}
            if regime not in valid_regimes:
                regime = "trend"

            # Utiliser TradingEnv pour simuler
            self.env.data = data
            obs, _ = self.env.reset()
            iteration_rewards = []
            max_steps = min(self.evaluation_steps, len(data))
            for step in range(max_steps):
                cache_key = self.generate_cache_key(
                    market="ES",
                    timestamp=datetime.now().isoformat(),
                    metrics={
                        "operation": "evaluate_model",
                        "success": True,
                    },
                )
                if cache_key in self.prediction_cache:
                    prediction = self.prediction_cache[cache_key]
                    self.log_performance_metrics(
                        operation="prediction_cache_hit",
                        latency=0,
                        success=True,
                    )
                else:
                    try:
                        from src.model.inference import predict

                        def predict_model():
                            return predict(
                                data=data.iloc[step : step + 1],
                                model_path=model_config["model_path"],
                                mode=regime,
                                model_type=model_config["model_type"],
                                policy_type=model_config["policy_type"],
                            )

                        prediction = self.with_retries(predict_model)
                        if prediction is None:
                            raise ValueError("Échec de la prédiction")
                        self.prediction_cache[cache_key] = prediction
                        if len(self.prediction_cache) > self.max_cache_size:
                            self.prediction_cache.pop(
                                next(iter(self.prediction_cache))
                            )
                        self.log_performance_metrics(
                            operation="prediction_cache_miss",
                            latency=0,
                            success=True,
                        )
                    except Exception as e:
                        error_msg = (
                            f"Erreur prédiction pour {model_name}: {str(e)}\n"
                            f"{traceback.format_exc()}"
                        )
                        self.alert_manager.send_alert(error_msg, priority=4)
                        send_telegram_alert(error_msg)
                        logger.error(error_msg)
                        self.log_performance(
                            operation="evaluate_model",
                            latency=(
                                datetime.now() - start_time
                            ).total_seconds(),
                            success=False,
                            error=str(e),
                        )
                        capture_error(
                            error=e,
                            context={"market": "ES"},
                            market="ES",
                            operation="evaluate_model",
                        )
                        return 0.0, {
                            "sharpe": 0.0,
                            "drawdown": 0.0,
                            "profit_factor": 0.0,
                        }

                action = prediction["action"]
                obs, reward, done, _, info = self.env.step(np.array([action]))
                iteration_rewards.append(reward)
                if done:
                    break

            # Mettre à jour les performances
            self.model_performance[model_name]["rewards"].extend(
                iteration_rewards
            )
            self.model_performance[model_name]["rewards"] = (
                self.model_performance[model_name]["rewards"][-1000:]
            )
            metrics = self.compute_model_metrics(
                model_name=model_name,
                rewards=self.model_performance[model_name]["rewards"],
            )

            if metrics["sharpe"] < self.thresholds["min_sharpe"]:
                self.model_performance[model_name]["underperformance_count"] += 1
            else:
                self.model_performance[model_name]["underperformance_count"] = 0

            score = (
                self.score_weights["sharpe_weight"]
                * (metrics["sharpe"] / self.thresholds["min_sharpe"])
                + self.score_weights["drawdown_weight"]
                * (
                    1 - min(
                        metrics["drawdown"] / self.thresholds["max_drawdown"],
                        1.0,
                    )
                )
                + self.score_weights["profit_factor_weight"]
                * (
                    metrics["profit_factor"]
                    / self.thresholds["min_profit_factor"]
                )
            )
            score = min(1.0, max(0.0, score))

            logger.info(
                f"Évalué {model_name}: score={score:.2f}, métriques={metrics}"
            )
            self.alert_manager.send_alert(
                message=(
                    f"Évalué {model_name}: score={score:.2f}, "
                    f"métriques={metrics}"
                ),
                priority=1,
            )
            send_telegram_alert(
                message=(
                    f"Évalué {model_name}: score={score:.2f}, "
                    f"métriques={metrics}"
                )
            )
            self.log_performance(
                operation="evaluate_model",
                latency=(datetime.now() - start_time).total_seconds(),
                success=True,
                model_name=model_name,
                score=score,
            )
            return score, metrics
        except Exception as e:
            error_msg = (
                f"Erreur évaluation modèle {model_name}: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                operation="evaluate_model",
                latency=(datetime.now() - start_time).total_seconds(),
                success=False,
                error=str(e),
            )
            capture_error(
                error=e,
                context={"market": "ES"},
                market="ES",
                operation="evaluate_model",
            )
            return 0.0, {
                "sharpe": 0.0,
                "drawdown": 0.0,
                "profit_factor": 0.0,
            }

   def switch_strategy(
        self,
        data: pd.DataFrame,
        model_predictions: Dict,
    ) -> Dict:
        """
        Arbitre les stratégies en fonction des features et prédictions, avec
        position sizing, microstructure, HMM, et vote bayésien.

        Args:
            data (pd.DataFrame): Données avec les features (atr_dynamic,
                orderflow_imbalance, bid_ask_imbalance, trade_aggressiveness,
                regime_hmm).
            model_predictions (Dict): Prédictions des modèles (sac, ppo, ddpg).

        Returns:
            Dict: Décision stratégique et taille de position.
        """
        start_time = datetime.now()
        try:
            # Valider les données
            required_features = [
                "atr_dynamic",
                "orderflow_imbalance",
                "bid_ask_imbalance",
                "trade_aggressiveness",
                "regime_hmm",
            ]
            missing_features = [
                f for f in required_features if f not in data.columns
            ]
            if missing_features:
                error_msg = (
                    f"Features manquantes pour switch_strategy: "
                    f"{missing_features}"
                )
                raise ValueError(error_msg)

            # Récupérer les features dynamiques
            bid_ask_imbalance = data["bid_ask_imbalance"].iloc[-1]
            trade_aggressiveness = data["trade_aggressiveness"].iloc[-1]
            regime_hmm = data["regime_hmm"].iloc[-1]

            # Calculer la taille de position
            position_size = self.risk_manager.calculate_position_size(
                atr_dynamic=data["atr_dynamic"].iloc[-1],
                orderflow_imbalance=data["orderflow_imbalance"].iloc[-1],
                volatility_score=data.get("predicted_volatility", 0.2),
            )

            # Vote bayésien pour les prédictions
            weights = MLFlowTracker().get_model_weights(
                model_names=["sac", "ppo", "ddpg"]
            )
            ensemble_weight_sac.labels(market="ES").set(
                weights.get("sac", 0.33)
            )
            ensemble_weight_ppo.labels(market="ES").set(
                weights.get("ppo", 0.33)
            )
            ensemble_weight_ddpg.labels(market="ES").set(
                weights.get("ddpg", 0.33)
            )
            ensemble_pred = sum(
                weights.get(model, 0.33) * pred
                for model, pred in model_predictions.items()
            )

            # Ajuster la décision en fonction des features de microstructure et
            # HMM
            decision = ensemble_pred * (
                1 + bid_ask_imbalance * 0.1 + trade_aggressiveness * 0.1
            )
            if regime_hmm == 0:
                decision *= 1.2
            elif regime_hmm == 1:
                decision *= 0.8

            result = {
                "decision": float(decision),
                "position_size": float(position_size),
            }
            latency = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Décision stratégique pour ES. Décision: {decision:.4f}, "
                f"Position: {position_size:.4f}, Latence: {latency}s"
            )
            self.alert_manager.send_alert(
                message=(
                    f"Décision stratégique pour ES: décision={decision:.4f}, "
                    f"position={position_size:.4f}"
                ),
                priority=2,
            )
            self.log_performance(
                operation="switch_strategy",
                latency=latency,
                success=True,
                decision=decision,
                position_size=position_size,
            )
            return result
        except Exception as e:
            error_msg = (
                f"Erreur arbitrage stratégie: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                operation="switch_strategy",
                latency=latency,
                success=False,
                error=str(e),
            )
            capture_error(
                error=e,
                context={"market": "ES"},
                market="ES",
                operation="switch_strategy",
            )
            return {"decision": 0.0, "position_size": 0.0}

    def switch_mia(
        self,
        data: pd.DataFrame,
        step: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """
        Sélectionne le modèle MIA optimal en fonction des conditions de marché
        et des stratégies.
        """
        start_time = time.time()
        try:
            self.validate_data(data)
            context = context or {}
            required_context = ["neural_regime", "predicted_volatility"]
            for key in required_context:
                if key not in context:
                    context[key] = (
                        "unknown" if key == "neural_regime" else 0.0
                    )
            current_time = datetime.now()
            current_data = data.iloc[-1]

            # Vérifier si un réentraînement est nécessaire
            self.maybe_retrain_trade_probability()

            if (
                current_time - self.last_switch_time
            ).total_seconds() < self.thresholds["regime_switch_frequency"]:
                logger.info(
                    f"Bascule trop récente, conservation du modèle: "
                    f"{self.current_model}"
                )
                self.alert_manager.send_alert(
                    message=(
                        f"Bascule trop récente, conservation du modèle: "
                        f"{self.current_model}"
                    ),
                    priority=1,
                )
                send_telegram_alert(
                    message=(
                        f"Bascule trop récente, conservation du modèle: "
                        f"{self.current_model}"
                    )
                )
                self.log_performance(
                    operation="switch_mia",
                    latency=(datetime.now() - start_time).total_seconds(),
                    success=True,
                    model_name=self.current_model,
                )
                self.log_switch_event(
                    decision="keep_model",
                    reason="recent_switch",
                    regime=context.get("neural_regime", "unknown"),
                )
                return self.models[self.current_model]

            model_scores = {}
            model_metrics = {}
            valid_models = []
            for model_name in self.models:
                score, metrics = self.evaluate_model(
                    model_name=model_name,
                    data=data,
                )
                model_scores[model_name] = score
                model_metrics[model_name] = metrics
                if score > 0.0:
                    valid_models.append(model_name)

            if not valid_models:
                error_msg = (
                    "Aucun modèle valide trouvé, retour au SAC par défaut"
                )
                self.alert_manager.send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                default_model = next(
                    (
                        name
                        for name in self.models
                        if self.models[name]["model_type"] == "sac"
                    ),
                    self.current_model,
                )
                self.current_model = default_model
                self.log_performance(
                    operation="switch_mia",
                    latency=(datetime.now() - start_time).total_seconds(),
                    success=True,
                    model_name=self.current_model,
                )
                self.log_switch_event(
                    decision="default_model",
                    reason="no_valid_models",
                    regime=context.get("neural_regime", "unknown"),
                )
                return self.models[self.current_model]

            best_model = max(model_scores, key=model_scores.get)
            best_score = model_scores[best_model]

            if (
                best_model != self.current_model
                and best_score > self.thresholds["switch_confidence_threshold"]
                and self.model_performance[self.current_model][
                    "underperformance_count"
                ]
                >= self.thresholds["max_consecutive_underperformance"]
            ):
                reason = (
                    f"Bascule vers {best_model} (score={best_score:.2f}, "
                    f"précédent={self.current_model}, sous-performance="
                    f"{self.model_performance[self.current_model]['underperformance_count']})"
                )
                self.current_model = best_model
                self.last_switch_time = current_time
                self.switch_history.append(
                    {
                        "timestamp": str(current_time),
                        "model": best_model,
                        "score": best_score,
                        "reason": reason,
                        "context": context,
                        "vix": float(current_data.get("vix", 0.0)),
                        "neural_regime": context.get(
                            "neural_regime",
                            "unknown",
                        ),
                        "predicted_volatility": float(
                            current_data.get("predicted_volatility", 0.0)
                        ),
                    }
                )
                logger.info(reason)
                self.alert_manager.send_alert(reason, priority=3)
                send_telegram_alert(reason)
                self.log_switch_event(
                    decision="switch_model",
                    reason=reason,
                    regime=context.get("neural_regime", "unknown"),
                )

                def store_switch():
                    store_pattern(
                        data=data,
                        action=0.0,
                        reward=0.0,
                        neural_regime=context.get("neural_regime", "unknown"),
                        confidence=best_score,
                        metadata={
                            "event": "model_switch",
                            "model": best_model,
                            "reason": reason,
                            "strategy_params": context.get(
                                "strategy_params",
                                {},
                            ),
                            "vix": float(current_data.get("vix", 0.0)),
                            "predicted_volatility": float(
                                current_data.get("predicted_volatility", 0.0)
                            ),
                        },
                    )

                self.with_retries(store_switch)

                log_entry = {
                    "step": step,
                    "timestamp": str(current_time),
                    "model": best_model,
                    "score": best_score,
                    "reason": reason,
                    "vix": float(current_data.get("vix", 0.0)),
                    "neural_regime": context.get("neural_regime", "unknown"),
                    "predicted_volatility": float(
                        current_data.get("predicted_volatility", 0.0)
                    ),
                }
                self.log_buffer.append(log_entry)
                if len(self.log_buffer) >= self.buffer_size:
                    log_df = pd.DataFrame(self.log_buffer)
                    os.makedirs(
                        os.path.dirname(self.switcher_log_path),
                        exist_ok=True,
                    )

                    def write_log():
                        if not os.path.exists(self.switcher_log_path):
                            log_df.to_csv(
                                self.switcher_log_path,
                                index=False,
                                encoding="utf-8",
                            )
                        else:
                            log_df.to_csv(
                                self.switcher_log_path,
                                mode="a",
                                header=False,
                                index=False,
                                encoding="utf-8",
                            )

                    self.with_retries(write_log)
                    self.log_buffer = []

                self.save_switch_snapshot(
                    step=step,
                    timestamp=pd.Timestamp.now(),
                    model=best_model,
                    score=best_score,
                    reason=reason,
                    data=data,
                )
            else:
                reason = (
                    f"Modèle {self.current_model} conservé: "
                    f"score={model_scores[self.current_model]:.2f}"
                )
                logger.info(reason)
                self.alert_manager.send_alert(reason, priority=1)
                send_telegram_alert(reason)
                self.log_switch_event(
                    decision="keep_model",
                    reason=reason,
                    regime=context.get("neural_regime", "unknown"),
                )

            self.save_dashboard_status()
            self.log_performance(
                operation="switch_mia",
                latency=(datetime.now() - start_time).total_seconds(),
                success=True,
                model_name=self.current_model,
            )
            return self.models[self.current_model]
        except Exception as e:
            error_msg = (
                f"Erreur dans switch_mia: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                operation="switch_mia",
                latency=(datetime.now() - start_time).total_seconds(),
                success=False,
                error=str(e),
            )
            capture_error(
                error=e,
                context={"market": "ES"},
                market="ES",
                operation="switch_mia",
            )
            default_model = next(
                (
                    name
                    for name in self.models
                    if self.models[name]["model_type"] == "sac"
                ),
                self.current_model,
            )
            self.current_model = default_model
            self.log_switch_event(
                decision="default_model",
                reason="error",
                regime=context.get("neural_regime", "unknown"),
            )
            return self.models[self.current_model]

    def save_switch_snapshot(
        self,
        step: int,
        timestamp: pd.Timestamp,
        model: str,
        score: float,
        reason: str,
        data: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Sauvegarde un instantané des décisions de bascule.
        """
        start_time = time.time()
        try:
            snapshot = {
                "step": step,
                "timestamp": str(timestamp),
                "model": model,
                "score": score,
                "reason": reason,
                "features": (
                    data.to_dict(orient="records")[0]
                    if data is not None
                    else None
                ),
                "switch_history": self.switch_history[-10:],
            }
            self.save_snapshot(
                snapshot_type=f"switch_step_{step:04d}",
                data=snapshot,
            )
            self.alert_manager.send_alert(
                message=f"Snapshot bascule step {step} sauvegardé",
                priority=1,
            )
            send_telegram_alert(
                message=f"Snapshot bascule step {step} sauvegardé"
            )
            self.log_performance(
                operation="save_switch_snapshot",
                latency=(datetime.now() - start_time).total_seconds(),
                success=True,
                step=step,
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde snapshot bascule: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                operation="save_switch_snapshot",
                latency=(datetime.now() - start_time).total_seconds(),
                success=False,
                error=str(e),
            )
            capture_error(
                error=e,
                context={"market": "ES"},
                market="ES",
                operation="save_switch_snapshot",
            )

    def save_dashboard_status(
        self,
        status_file: Optional[str] = None,
    ) -> None:
        """
        Sauvegarde l’état du switcher pour mia_dashboard.py.
        """
        start_time = time.time()
        try:
            status_file = status_file or self.dashboard_path
            error_counts = {}
            for entry in self.log_buffer:
                if not entry.get("success") and entry.get("error"):
                    error_type = str(entry["error"].split(":")[0])
                    error_counts[error_type] = error_counts.get(
                        error_type,
                        0,
                    ) + 1
            switch_counts = {}
            for entry in self.switch_history:
                model = entry["model"]
                switch_counts[model] = switch_counts.get(model, 0) + 1
            status = {
                "timestamp": datetime.now().isoformat(),
                "current_model": self.current_model,
                "num_switches": len(self.switch_history),
                "switch_counts": switch_counts,
                "model_performance": self.model_performance,
                "last_switch": (
                    self.switch_history[-1] if self.switch_history else None
                ),
                "recent_switches": self.switch_history[-10:],
                "error_counts": error_counts,
                "vix": (
                    self.switch_history[-1]["vix"]
                    if self.switch_history
                    else 0.0
                ),
                "neural_regime": (
                    self.switch_history[-1]["neural_regime"]
                    if self.switch_history
                    else "unknown"
                ),
            }
            os.makedirs(os.path.dirname(status_file), exist_ok=True)

            def write_status():
                with open(status_file, "w", encoding="utf-8") as f:
                    json.dump(status, f, indent=4)

            self.with_retries(write_status)
            logger.info(f"Dashboard switcher sauvegardé: {status_file}")
            self.alert_manager.send_alert(
                message="Dashboard switcher mis à jour",
                priority=1,
            )
            send_telegram_alert(message="Dashboard switcher mis à jour")
            self.log_performance(
                operation="save_dashboard_status",
                latency=(datetime.now() - start_time).total_seconds(),
                success=True,
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde dashboard: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                operation="save_dashboard_status",
                latency=(datetime.now() - start_time).total_seconds(),
                success=False,
                error=str(e),
            )
            capture_error(
                error=e,
                context={"market": "ES"},
                market="ES",
                operation="save_dashboard_status",
            )


if __name__ == "__main__":
    try:
        # Données simulées pour test
        data = pd.DataFrame(
            {
                "timestamp": [datetime.now()],
                "vix": [20.0],
                "neural_regime": [0],
                "predicted_volatility": [1.5],
                "trade_frequency_1s": [8.0],
                "close": [5100.0],
                "atr_dynamic": [2.0],
                "orderflow_imbalance": [0.5],
                "bid_ask_imbalance": [0.1],
                "trade_aggressiveness": [0.2],
                "regime_hmm": [0],
                **{
                    f"feat_{i}": [np.random.uniform(0, 1)]
                    for i in range(340)
                },
            }
        )

        # Contexte simulé
        context = {
            "neural_regime": "trend",
            "predicted_volatility": 1.5,
            "strategy_params": {
                "entry_threshold": 70.0,
                "exit_threshold": 30.0,
                "position_size": 0.5,
                "stop_loss": 2.0,
            },
        }

        # Créer une instance de MiaSwitcher et tester
        switcher = MiaSwitcher()
        model_predictions = {"sac": 0.7, "ppo": 0.65, "ddpg": 0.6}
        result = switcher.switch_strategy(
            data=data,
            model_predictions=model_predictions,
        )
        print(
            f"Décision: {result['decision']}, "
            f"Position size: {result['position_size']}"
        )

        # Tester switch_mia
        model_config = switcher.switch_mia(
            data=data,
            step=1,
            context=context,
        )
        print(f"Modèle sélectionné: {model_config['model_type']}")
    except Exception as e:
        error_msg = (
            f"Erreur test MiaSwitcher: {str(e)}\n{traceback.format_exc()}"
        )
        logger.error(error_msg)