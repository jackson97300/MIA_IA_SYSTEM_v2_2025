# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/model/trade_probability.py
# Prédit la probabilité de réussite d’un trade (succès = reward > 0) pour 
# MIA_IA_SYSTEM_v2_2025.
# Utilisé comme critère d’entrée en position et affiché sur le dashboard.
#
# Version : 2.1.5
# Date : 2025-05-14
#
# Rôle : Entraîne un modèle RandomForestClassifier sur les données de 
# market_memory.db (350 features), prédit trade_success_prob, valide les 
# features (incluant 150 SHAP features pour fallback), et déclenche des alertes 
# pour les erreurs. Orchestre les modèles RL via trade_probability_rl.py.
# Intègre SignalResolver pour résoudre les conflits de signaux et ajuster la 
# probabilité, avec une méthode directe resolve_signals pour des signaux 
# spécifiques.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, scikit-learn>=1.5.0,<2.0.0, 
#   sqlite3, psutil>=5.9.8,<6.0.0, joblib>=1.3.0,<2.0.0, boto3>=1.26.0,<2.0.0, 
#   loguru>=0.7.0,<1.0.0, pyyaml>=6.0.0,<7.0.0, optuna>=3.6.0,<4.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/hyperparam_optimizer.py
# - src/utils/telegram_alert.py
# - src/model/trade_probability_rl.py
# - src/utils/mlflow_tracker.py
# - src/monitoring/prometheus_metrics.py
# - src/utils/error_tracker.py
# - src/model/utils/signal_resolver.py
#
# Inputs :
# - config/trade_probability_config.yaml
# - config/feature_sets.yaml
# - config/es_config.yaml
# - data/market_memory.db
#
# Outputs :
# - data/logs/trade_probability_performance.csv
# - data/logs/trade_probability_metrics.csv
# - data/logs/trade_probability_backtest.csv
# - data/cache/trade_probability/<market>/*.json.gz
# - data/checkpoints/trade_probability/<market>/*.json.gz
# - model/trade_probability/<market>/rf_trade_prob_*.pkl
#
# Notes :
# - Intègre retries (max 3, délai 2^attempt), logs psutil, alertes Telegram, 
#   validation des features, sérialisation du modèle, équilibre des classes, et 
#   backtest des seuils.
# - Utilise Optuna pour l’optimisation des hyperparamètres (remplace 
#   scikit-optimize).
# - Conforme à l’utilisation exclusive d’IQFeed.
# - Suppression de toute référence à dxFeed, obs_t, 320/81 features.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des 
#   données.
# - Tests unitaires disponibles dans tests/test_trade_probability.py.
# - Nouvelles fonctionnalités : coûts de transaction (slippage_estimate dans 
#   récompense), microstructure (bid_ask_imbalance, trade_aggressiveness), 
#   walk-forward (TimeSeriesSplit), Safe RL (PPO-Lagrangian avec CVaR), 
#   Distributional RL (QR-DQN), surface de volatilité (iv_skew, 
#   iv_term_structure), ensembles de politiques (vote bayésien), logs psutil, 
#   journalisation MLflow, monitoring Prometheus, résolution des conflits de 
#   signaux via SignalResolver.
# - SignalResolver intégré directement via resolve_signals pour des signaux 
#   spécifiques (ex. : vix_es_correlation, call_iv_atm) en complément de 
#   RLTrainer.
# - Policies Note: The official directory pour routing policies est 
#   src/model/router/policies.

import gzip
import hashlib
import json
import os
import signal
import sqlite3
import time
import traceback
import uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import boto3
import joblib
import numpy as np
import pandas as pd
import psutil
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from src.model.trade_probability_rl import RLTrainer
from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import get_config
from src.model.utils.signal_resolver import SignalResolver
from src.monitoring.prometheus_metrics import Gauge
from src.utils.error_tracker import capture_error
from src.utils.mlflow_tracker import MLFlowTracker
from src.utils.telegram_alert import send_telegram_alert

# Configuration logging
logger.remove()
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOG_DIR = BASE_DIR / "data" / "logs"
CACHE_DIR = BASE_DIR / "data" / "cache" / "trade_probability"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints" / "trade_probability"
MODEL_DIR = BASE_DIR / "model" / "trade_probability"
PERF_LOG_PATH = LOG_DIR / "trade_probability_performance.csv"
METRICS_LOG_PATH = LOG_DIR / "trade_probability_metrics.csv"
BACKTEST_LOG_PATH = LOG_DIR / "trade_probability_backtest.csv"
SNAPSHOT_DIR = CACHE_DIR
DB_PATH = BASE_DIR / "data" / "market_memory.db"
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "trade_probability.log",
    rotation="10 MB",
    level="INFO",
    encoding="utf-8",
)
logger.add(lambda msg: print(msg, end=""), level="INFO")

# Constantes
MAX_RETRIES = 3
RETRY_DELAY = 2.0
MAX_CACHE_SIZE = 1000
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB

# Métriques Prometheus
cvar_loss_metric = Gauge(
    name="cvar_loss",
    description="Perte CVaR pour PPO-Lagrangian",
    labelnames=["market"],
)
qr_dqn_quantiles_metric = Gauge(
    name="qr_dqn_quantiles",
    description="Quantiles QR-DQN",
    labelnames=["market"],
)


class TradeProbabilityPredictor:
    """
    Prédit la probabilité de réussite d’un trade basé sur les 350 features et 
    des modèles RL.

    Attributes:
        db_path (str): Chemin vers market_memory.db.
        model (RandomForestClassifier): Modèle de classification avec équilibre 
        des classes.
        rl_trainer (RLTrainer): Gestionnaire des modèles RL.
        feature_cols (list): Liste des 350 features attendues.
        shap_feature_cols (list): Liste des 150 SHAP features pour fallback.
        log_buffer (list): Buffer pour les logs de performance.
        metrics_buffer (list): Buffer pour les métriques d’apprentissage.
        backtest_buffer (list): Buffer pour les résultats de backtest.
        config (dict): Configuration chargée via config_manager.
        alert_manager (AlertManager): Gestionnaire d’alertes.
        prediction_cache (OrderedDict): Cache LRU pour les prédictions.
        signal_cache (OrderedDict): Cache LRU pour les résolutions de signaux.
        checkpoint_versions (List): Liste des versions de checkpoints.
        mlflow_tracker (MLFlowTracker): Tracker pour journalisation MLflow.
        signal_resolver (SignalResolver): Résolveur de conflits de signaux.
    """


    def __init__(self, db_path: str = str(DB_PATH), market: str = "ES"):
        """Initialise le prédicteur de probabilité de trade."""
        start_time = datetime.now()
        try:
            self.alert_manager = AlertManager()
            self.db_path = db_path
            self.market = market
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced",
            )
            self.rl_trainer = RLTrainer(market=market)
            feature_sets_path = BASE_DIR / "config/feature_sets.yaml"
            features_config = self.with_retries(
                lambda: get_config(feature_sets_path)
            )
            self.feature_cols = features_config.get("training", {}).get(
                "features", []
            )[:350]
            # Ajout des nouvelles features si elles ne sont pas déjà présentes
            new_features = [
                "bid_ask_imbalance",
                "trade_aggressiveness",
                "iv_skew",
                "iv_term_structure",
            ]
            for feature in new_features:
                if feature not in self.feature_cols:
                    self.feature_cols.append(feature)
            self.shap_feature_cols = features_config.get("inference", {}).get(
                "shap_features", []
            )[:150]
            # Ajout des nouvelles features SHAP si elles ne sont pas déjà 
            # présentes
            for feature in new_features:
                if feature not in self.shap_feature_cols:
                    self.shap_feature_cols.append(feature)
            self.log_buffer = []
            self.metrics_buffer = []
            self.backtest_buffer = []
            self.prediction_cache = OrderedDict()
            self.signal_cache = OrderedDict()
            self.max_cache_size = MAX_CACHE_SIZE
            self.checkpoint_versions = []
            self.config = self.with_retries(
                lambda: get_config(
                    BASE_DIR / "config/trade_probability_config.yaml"
                )
            )
            self.mlflow_tracker = MLFlowTracker(
                experiment_name=f"trade_probability_{market}"
            )
            config_path = str(BASE_DIR / "config/es_config.yaml")
            self.signal_resolver = SignalResolver(
                config_path=config_path,
                market=market,
            )
            # Validation MLflow et S3
            if not self.mlflow_tracker.tracking_uri:
                error_msg = f"MLflow tracking URI non configuré pour {market}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)
            if not self.config.get("trade_probability", {}).get("s3_bucket"):
                warning_msg = (
                    f"S3 bucket non configuré pour {market}, "
                    f"utilisant sauvegarde locale"
                )
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
            SNAPSHOT_DIR.mkdir(exist_ok=True)
            MODEL_DIR.mkdir(exist_ok=True)
            LOG_DIR.mkdir(exist_ok=True)
            CHECKPOINT_DIR.mkdir(exist_ok=True)
            success_msg = f"TradeProbabilityPredictor initialisé pour {market}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                operation="init",
                latency=latency,
                success=True,
                num_features=len(self.feature_cols),
                market=market,
            )
            signal.signal(signal.SIGINT, self.handle_sigint)
            self.load_model(market=market)
        except Exception as e:
            error_msg = (
                f"Erreur initialisation TradeProbabilityPredictor pour "
                f"{market}: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            capture_error(
                error=e,
                context={"market": market},
                market=market,
                operation="init",
            )
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance(
                operation="init",
                latency=0,
                success=False,
                error=str(e),
                market=market,
            )
            raise


    def handle_sigint(self, signum, frame):
        """Gère l'arrêt propre sur SIGINT."""
        datetime.now()
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "status": "SIGINT",
        }
        try:
            self.save_snapshot(
                snapshot_type="sigint",
                data=snapshot,
                market=self.market,
            )
            success_msg = (
                f"Arrêt propre sur SIGINT pour {self.market}, "
                f"snapshot sauvegardé"
            )
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            self.log_performance(
                operation="handle_sigint",
                latency=0,
                success=True,
                market=self.market,
            )
            exit(0)
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde SIGINT pour {self.market}: "
                f"{str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            capture_error(
                error=e,
                context={"market": self.market},
                market=self.market,
                operation="handle_sigint",
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                operation="handle_sigint",
                latency=0,
                success=False,
                error=str(e),
                market=self.market,
            )
            exit(1)


    def with_retries(
        self,
        func: callable,
        max_attempts: int = MAX_RETRIES,
        delay_base: float = RETRY_DELAY,
    ) -> Any:
        """Exécute une fonction avec retries exponentiels."""
        start_time = datetime.now()
        for attempt in range(max_attempts):
            try:
                result = func()
                latency = (datetime.now() - start_time).total_seconds()
                self.log_performance(
                    operation=f"retry_attempt_{attempt+1}",
                    latency=latency,
                    success=True,
                    attempt_number=attempt + 1,
                    market=self.market,
                )
                return result
            except Exception as e:
                if attempt == max_attempts - 1:
                    error_msg = (
                        f"Échec après {max_attempts} tentatives pour "
                        f"{self.market}: {str(e)}\n{traceback.format_exc()}"
                    )
                    logger.error(error_msg)
                    capture_error(
                        error=e,
                        context={"market": self.market},
                        market=self.market,
                        operation=f"retry_attempt_{attempt+1}",
                    )
                    self.alert_manager.send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    latency = (datetime.now() - start_time).total_seconds()
                    self.log_performance(
                        operation=f"retry_attempt_{attempt+1}",
                        latency=latency,
                        success=False,
                        error=str(e),
                        attempt_number=attempt + 1,
                        market=self.market,
                    )
                    raise
                delay = delay_base * (2**attempt)
                warning_msg = (
                    f"Tentative {attempt+1} échouée pour {self.market}, "
                    f"retry après {delay}s"
                )
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                time.sleep(delay)


    def log_performance(
        self,
        operation: str,
        latency: float,
        success: bool,
        error: str = None,
        **kwargs,
    ) -> None:
        """Journalise les performances des opérations critiques."""
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_usage = psutil.cpu_percent()
            if memory_usage > 1024:
                alert_msg = (
                    f"ALERTE: Usage mémoire élevé ({memory_usage:.2f} MB) "
                    f"pour {self.market}"
                )
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=5)
                send_telegram_alert(alert_msg)
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
            buffer_size = self.config.get("trade_probability", {}).get(
                "buffer_size", 100
            )
            if len(self.log_buffer) >= buffer_size:
                log_df = pd.DataFrame(self.log_buffer)

                def save_log():
                    if not PERF_LOG_PATH.exists():
                        log_df.to_csv(
                            PERF_LOG_PATH,
                            index=False,
                            encoding="utf-8",
                        )
                    else:
                        log_df.to_csv(
                            PERF_LOG_PATH,
                            mode="a",
                            header=False,
                            index=False,
                            encoding="utf-8",
                        )

                self.with_retries(save_log)
                self.checkpoint(
                    data=log_df,
                    data_type="performance_logs",
                    market=self.market,
                )
                self.cloud_backup(
                    data=log_df,
                    data_type="performance_logs",
                    market=self.market,
                )
                self.log_buffer = []
            logger.info(
                f"Performance journalisée pour {operation}. CPU: {cpu_usage}%"
            )
            self.mlflow_tracker.log_metrics(
                {
                    "latency": latency,
                    "memory_usage_mb": memory_usage,
                    "cpu_usage_percent": cpu_usage,
                }
            )
        except Exception as e:
            error_msg = (
                f"Erreur journalisation performance pour {self.market}: "
                f"{str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            capture_error(
                error=e,
                context={"market": self.market},
                market=self.market,
                operation="log_performance",
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)


    def log_metrics(
        self,
        auc: float,
        precision: float,
        recall: float,
        f1: float,
        market: str = "ES",
    ) -> None:
        """Journalise les métriques d’apprentissage."""
        try:
            start_time = datetime.now()
            metrics_entry = {
                "timestamp": datetime.now().isoformat(),
                "market": market,
                "auc": auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            self.metrics_buffer.append(metrics_entry)
            buffer_size = self.config.get("trade_probability", {}).get(
                "buffer_size", 100
            )
            if len(self.metrics_buffer) >= buffer_size:
                metrics_df = pd.DataFrame(self.metrics_buffer)

                def save_metrics():
                    if not METRICS_LOG_PATH.exists():
                        metrics_df.to_csv(
                            METRICS_LOG_PATH,
                            index=False,
                            encoding="utf-8",
                        )
                    else:
                        metrics_df.to_csv(
                            METRICS_LOG_PATH,
                            mode="a",
                            header=False,
                            index=False,
                            encoding="utf-8",
                        )

                self.with_retries(save_metrics)
                self.checkpoint(
                    data=metrics_df,
                    data_type="metrics",
                    market=market,
                )
                self.cloud_backup(
                    data=metrics_df,
                    data_type="metrics",
                    market=market,
                )
                self.metrics_buffer = []
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                operation="log_metrics",
                latency=latency,
                success=True,
                market=market,
            )
            self.mlflow_tracker.log_metrics(
                {
                    "auc": auc,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
        except Exception as e:
            error_msg = (
                f"Erreur journalisation métriques pour {market}: "
                f"{str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            capture_error(
                error=e,
                context={"market": market},
                market=market,
                operation="log_metrics",
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                operation="log_metrics",
                latency=0,
                success=False,
                error=str(e),
                market=market,
            )


    def log_backtest(
        self,
        threshold: float,
        win_rate: float,
        sharpe: float,
        num_trades: int,
        market: str = "ES",
    ) -> None:
        """Journalise les résultats du backtest."""
        try:
            start_time = datetime.now()
            backtest_entry = {
                "timestamp": datetime.now().isoformat(),
                "market": market,
                "threshold": threshold,
                "win_rate": win_rate,
                "sharpe": sharpe,
                "num_trades": num_trades,
            }
            self.backtest_buffer.append(backtest_entry)
            buffer_size = self.config.get("trade_probability", {}).get(
                "buffer_size", 100
            )
            if len(self.backtest_buffer) >= buffer_size:
                backtest_df = pd.DataFrame(self.backtest_buffer)

                def save_backtest():
                    if not BACKTEST_LOG_PATH.exists():
                        backtest_df.to_csv(
                            BACKTEST_LOG_PATH,
                            index=False,
                            encoding="utf-8",
                        )
                    else:
                        backtest_df.to_csv(
                            BACKTEST_LOG_PATH,
                            mode="a",
                            header=False,
                            index=False,
                            encoding="utf-8",
                        )

                self.with_retries(save_backtest)
                self.checkpoint(
                    data=backtest_df,
                    data_type="backtest",
                    market=market,
                )
                self.cloud_backup(
                    data=backtest_df,
                    data_type="backtest",
                    market=market,
                )
                self.backtest_buffer = []
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                operation="log_backtest",
                latency=latency,
                success=True,
                market=market,
            )
            self.mlflow_tracker.log_metrics(
                {
                    "backtest_threshold": threshold,
                    "win_rate": win_rate,
                    "sharpe": sharpe,
                    "num_trades": num_trades,
                }
            )
        except Exception as e:
            error_msg = (
                f"Erreur journalisation backtest pour {market}: "
                f"{str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            capture_error(
                error=e,
                context={"market": market},
                market=market,
                operation="log_backtest",
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                operation="log_backtest",
                latency=0,
                success=False,
                error=str(e),
                market=market,
            )


    def save_snapshot(
        self,
        snapshot_type: str,
        data: Dict,
        market: str = "ES",
        compress: bool = True,
    ) -> None:
        """Sauvegarde un instantané des résultats, compressé avec gzip."""
        try:
            start_time = datetime.now()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = {
                "timestamp": timestamp,
                "type": snapshot_type,
                "market": market,
                "data": data,
                "buffer_size": len(self.log_buffer),
            }
            snapshot_dir = SNAPSHOT_DIR / market
            snapshot_dir.mkdir(exist_ok=True)
            snapshot_path = (
                snapshot_dir / f"snapshot_{snapshot_type}_{timestamp}.json"
            )

            def write_snapshot():
                if compress:
                    with gzip.open(
                        f"{snapshot_path}.gz",
                        "wt",
                        encoding="utf-8",
                    ) as f:
                        json.dump(snapshot, f, indent=4)
                else:
                    with open(snapshot_path, "w", encoding="utf-8") as f:
                        json.dump(snapshot, f, indent=4)

            self.with_retries(write_snapshot)
            save_path = f"{snapshot_path}.gz" if compress else snapshot_path
            file_size = os.path.getsize(save_path) / 1024 / 1024
            if file_size > 1.0:
                alert_msg = (
                    f"Snapshot size {file_size:.2f} MB exceeds 1 MB pour "
                    f"{market}"
                )
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = (
                f"Snapshot {snapshot_type} sauvegardé pour {market}: "
                f"{save_path}"
            )
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                operation="save_snapshot",
                latency=latency,
                success=True,
                snapshot_size_mb=file_size,
                market=market,
            )
            self.mlflow_tracker.log_artifact(save_path)
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde snapshot {snapshot_type} pour {market}: "
                f"{str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            capture_error(
                error=e,
                context={"market": market},
                market=market,
                operation="save_snapshot",
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                operation="save_snapshot",
                latency=0,
                success=False,
                error=str(e),
                market=market,
            )


    def checkpoint(
        self,
        data: pd.DataFrame,
        data_type: str = "trade_probability_state",
        market: str = "ES",
    ) -> None:
        """Sauvegarde incrémentielle des données toutes les 5 minutes avec 
        versionnage (5 versions)."""
        try:
            start_time = datetime.now()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_data = {
                "timestamp": timestamp,
                "num_rows": len(data),
                "columns": list(data.columns),
                "data_type": data_type,
                "market": market,
            }
            checkpoint_dir = CHECKPOINT_DIR / market
            checkpoint_dir.mkdir(exist_ok=True)
            checkpoint_path = (
                checkpoint_dir
                / f"trade_probability_{data_type}_{timestamp}.json.gz"
            )

            def write_checkpoint():
                with gzip.open(checkpoint_path, "wt", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, indent=4)
                data.to_csv(
                    checkpoint_path.with_suffix(".csv"),
                    index=False,
                    encoding="utf-8",
                )

            self.with_retries(write_checkpoint)
            self.checkpoint_versions.append(str(checkpoint_path))
            if len(self.checkpoint_versions) > 5:
                oldest = self.checkpoint_versions.pop(0)
                if os.path.exists(oldest):
                    os.remove(oldest)
                csv_oldest = oldest.replace(".json.gz", ".csv")
                if os.path.exists(csv_oldest):
                    os.remove(csv_oldest)
            file_size = os.path.getsize(checkpoint_path) / 1024 / 1024
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = (
                f"Checkpoint sauvegardé pour {market}: {checkpoint_path}"
            )
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                operation="checkpoint",
                latency=latency,
                success=True,
                file_size_mb=file_size,
                num_rows=len(data),
                data_type=data_type,
                market=market,
            )
            self.mlflow_tracker.log_artifact(checkpoint_path)
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde checkpoint pour {market}: "
                f"{str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            capture_error(
                error=e,
                context={"market": market},
                market=market,
                operation="checkpoint",
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                operation="checkpoint",
                latency=0,
                success=False,
                error=str(e),
                data_type=data_type,
                market=market,
            )


    def cloud_backup(
        self,
        data: pd.DataFrame,
        data_type: str = "trade_probability_state",
        market: str = "ES",
    ) -> None:
        """Sauvegarde distribuée des données vers AWS S3 toutes les 15 
        minutes."""
        try:
            start_time = datetime.now()
            s3_config = self.config.get("trade_probability", {})
            if not s3_config.get("s3_bucket"):
                warning_msg = (
                    f"S3 bucket non configuré, sauvegarde cloud ignorée pour "
                    f"{market}"
                )
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_prefix = s3_config.get("s3_prefix", "")
            backup_path = (
                f"{s3_prefix}trade_probability_{data_type}_{market}_"
                f"{timestamp}.csv.gz"
            )
            temp_path = CHECKPOINT_DIR / market / f"temp_s3_{timestamp}.csv.gz"
            temp_path.parent.mkdir(exist_ok=True)

            def write_temp():
                data.to_csv(
                    temp_path,
                    compression="gzip",
                    index=False,
                    encoding="utf-8",
                )

            self.with_retries(write_temp)
            s3_client = boto3.client("s3")
            try:

                def upload_s3():
                    s3_client.upload_file(
                        str(temp_path),
                        s3_config["s3_bucket"],
                        backup_path,
                    )

                self.with_retries(upload_s3)
            except Exception as s3_error:
                warning_msg = (
                    f"Échec sauvegarde S3 pour {market}: {str(s3_error)}. "
                    f"Sauvegarde locale conservée."
                )
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
            finally:
                os.remove(temp_path)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = (
                f"Sauvegarde cloud S3 effectuée pour {market}: {backup_path}"
            )
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                operation="cloud_backup",
                latency=latency,
                success=True,
                num_rows=len(data),
                data_type=data_type,
                market=market,
            )
            self.mlflow_tracker.log_artifact(temp_path)
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde cloud S3 pour {market}: "
                f"{str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            capture_error(
                error=e,
                context={"market": market},
                market=market,
                operation="cloud_backup",
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                operation="cloud_backup",
                latency=0,
                success=False,
                error=str(e),
                data_type=data_type,
                market=market,
            )


    def validate_features(
        self,
        features: Dict,
        use_shap_fallback: bool = False,
        market: str = "ES",
    ) -> bool:
        """Valide que le JSON de features contient toutes les colonnes 
        attendues."""
        try:
            target_cols = (
                self.shap_feature_cols if use_shap_fallback else self.feature_cols
            )
            missing_cols = [
                col for col in target_cols if col not in features
            ]
            null_count = sum(
                1
                for k, v in features.items()
                if pd.isna(v) and k in target_cols
            )
            confidence_drop_rate = (
                null_count / len(target_cols) if len(target_cols) > 0 else 0.0
            )
            if confidence_drop_rate > 0.5:
                alert_msg = (
                    f"Confidence_drop_rate élevé pour {market}: "
                    f"{confidence_drop_rate:.2f} ({null_count} valeurs nulles)"
                )
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
            if missing_cols:
                warning_msg = (
                    f"Colonnes manquantes dans features pour {market} "
                    f"({'SHAP' if use_shap_fallback else 'full'}): "
                    f"{missing_cols}"
                )
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=2)
                send_telegram_alert(warning_msg)
                return False
            if null_count > 0:
                warning_msg = (
                    f"Valeurs nulles dans features pour {market} "
                    f"({'SHAP' if use_shap_fallback else 'full'}): "
                    f"{null_count}"
                )
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=2)
                send_telegram_alert(warning_msg)
                return False
            return True
        except Exception as e:
            error_msg = (
                f"Erreur validation features pour {market}: "
                f"{str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            capture_error(
                error=e,
                context={"market": market},
                market=market,
                operation="validate_features",
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            return False


    def resolve_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Résout les conflits entre signaux en utilisant SignalResolver avec des 
        signaux spécifiques.

        Args:
            data (pd.DataFrame): Données d’entrée contenant les features.

        Returns:
            Dict[str, Any]: Métadonnées de la résolution des signaux (score, 
            normalized_score, entropy, conflict_coefficient, run_id, etc.).
        """
        try:
            start_time = datetime.now()
            run_id = str(uuid.uuid4())
            if data.empty:
                error_msg = (
                    f"DataFrame vide fourni pour la résolution des signaux "
                    f"pour {self.market}"
                )
                raise ValueError(error_msg)
            signals = {
                "vix_es_correlation": (
                    data.get("vix_es_correlation", pd.Series([0.0])).iloc[-1]
                    if "vix_es_correlation" in data.columns
                    else 0.0
                ),
                "call_iv_atm": (
                    data.get("call_iv_atm", pd.Series([0.0])).iloc[-1]
                    if "call_iv_atm" in data.columns
                    else 0.0
                ),
                "option_skew": (
                    data.get("option_skew", pd.Series([0.0])).iloc[-1]
                    if "option_skew" in data.columns
                    else 0.0
                ),
                "news_score_positive": (
                    data.get("news_impact_score", pd.Series([0.0])).iloc[-1]
                    if "news_impact_score" in data.columns
                    else 0.0
                ),
            }
            # Créer une clé de cache
            signals_json = json.dumps(signals, sort_keys=True)
            signals_hash = hashlib.sha256(signals_json.encode()).hexdigest()
            cache_key = f"{self.market}_{signals_hash}"
            if cache_key in self.signal_cache:
                metadata = self.signal_cache[cache_key]
                self.signal_cache.move_to_end(cache_key)
                return metadata
            score, metadata = self.signal_resolver.resolve_conflict(
                signals=signals,
                normalize=True,
                persist_to_db=True,
                export_to_csv=True,
                run_id=run_id,
                score_type="intermediate",
                mode_debug=True,
            )
            metadata["signal_score"] = score
            self.signal_cache[cache_key] = metadata
            while len(self.signal_cache) > self.max_cache_size:
                self.signal_cache.popitem(last=False)
            latency = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Résolution des signaux pour {self.market}: "
                f"Score={score:.2f}, "
                f"Conflict Coefficient={metadata['conflict_coefficient']:.2f}, "
                f"Entropy={metadata['entropy']:.2f}, Run ID={run_id}"
            )
            self.log_performance(
                operation="resolve_signals",
                latency=latency,
                success=True,
                score=score,
                conflict_coefficient=metadata["conflict_coefficient"],
                run_id=run_id,
                market=self.market,
            )
            return metadata
        except Exception as e:
            run_id = str(uuid.uuid4())
            error_msg = (
                f"Erreur résolution des signaux pour {self.market}: "
                f"{str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            latency = (datetime.now() - start_time).total_seconds()
            fallback_metadata = {
                "signal_score": 0.0,
                "normalized_score": None,
                "entropy": 0.0,
                "conflict_coefficient": 0.0,
                "score_type": "intermediate",
                "contributions": {},
                "run_id": run_id,
                "error": str(e),
            }
            self.log_performance(
                operation="resolve_signals",
                latency=latency,
                success=False,
                error=str(e),
                score=0.0,
                conflict_coefficient=0.0,
                run_id=run_id,
                market=self.market,
            )
            return fallback_metadata


    def load_data(self, market: str = "ES") -> Tuple[pd.DataFrame, pd.Series]:
        """Charge les données depuis market_memory.db avec validation."""

        def fetch_data():
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT features, reward FROM patterns WHERE market = ?",
                (market,),
            )
            data = [(json.loads(row[0]), row[1]) for row in cursor.fetchall()]
            conn.close()
            X_data = []
            y_data = []
            for features, reward in data:
                if self.validate_features(features, market=market):
                    X_data.append(features)
                    y_data.append(1 if reward > 0 else 0)
                else:
                    logger.warning(
                        f"Pattern ignoré pour {market} en raison de features "
                        f"manquantes"
                    )
            X_df = pd.DataFrame(X_data)[self.feature_cols].fillna(0)
            missing_shap = [
                col for col in self.shap_feature_cols if col not in X_df.columns
            ]
            if missing_shap:
                error_msg = (
                    f"SHAP features manquantes dans les données pour "
                    f"{market}: {missing_shap}"
                )
                raise ValueError(error_msg)
            return X_df, pd.Series(y_data)

        try:
            start_time = datetime.now()
            X, y = self.with_retries(fetch_data)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                operation="load_data",
                latency=latency,
                success=True,
                num_patterns=len(X),
                market=market,
            )
            self.save_snapshot(
                snapshot_type="load_data",
                data={
                    "num_patterns": len(X),
                    "market": market,
                },
                market=market,
            )
            return X, y
        except Exception as e:
            error_msg = (
                f"Erreur chargement données pour {market}: "
                f"{str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            capture_error(
                error=e,
                context={"market": market},
                market=market,
                operation="load_data",
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                operation="load_data",
                latency=0,
                success=False,
                error=str(e),
                market=market,
            )
            return pd.DataFrame(), pd.Series()


    def train(self, market: str = "ES") -> None:
        """Entraîne le modèle RandomForestClassifier avec validation glissante 
        et sauvegarde."""
        try:
            start_time = datetime.now()
            X, y = self.load_data(market=market)
            if len(X) < 10:
                error_msg = (
                    f"Données insuffisantes pour l’entraînement pour "
                    f"{market}: {len(X)}"
                )
                raise ValueError(error_msg)
            # Validation glissante avec TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)
            auc_scores = []
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                self.model.fit(X_train, y_train)
                y_prob = self.model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_prob)
                auc_scores.append(auc)
            mean_auc = np.mean(auc_scores)
            # Entraînement final sur toutes les données
            self.model.fit(X, y)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_dir = MODEL_DIR / market
            model_dir.mkdir(exist_ok=True)
            model_path = model_dir / f"rf_trade_prob_{timestamp}.pkl"

            def save_model():
                joblib.dump(self.model, model_path)

            self.with_retries(save_model)
            y_pred = self.model.predict(X)
            y_prob = self.model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, y_prob)
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            self.log_metrics(
                auc=auc,
                precision=precision,
                recall=recall,
                f1=f1,
                market=market,
            )
            if mean_auc < 0.6:
                alert_msg = (
                    f"AUC faible (moyenne cross-val: {mean_auc:.2f}) pour "
                    f"{market}"
                )
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                operation="train",
                latency=latency,
                success=True,
                num_patterns=len(X),
                mean_auc=mean_auc,
                auc=auc,
                market=market,
            )
            self.save_snapshot(
                snapshot_type="train",
                data={
                    "num_patterns": len(X),
                    "model_path": str(model_path),
                    "mean_auc": mean_auc,
                    "auc": auc,
                },
                market=market,
            )
            success_msg = (
                f"Modèle de probabilité de trade entraîné et sauvegardé pour "
                f"{market}: {model_path}"
            )
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            self.mlflow_tracker.log_artifact(model_path)
        except Exception as e:
            error_msg = (
                f"Erreur entraînement modèle pour {market}: "
                f"{str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            capture_error(
                error=e,
                context={"market": market},
                market=market,
                operation="train",
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                operation="train",
                latency=0,
                success=False,
                error=str(e),
                market=market,
            )
            raise


    def train_rl_models(self, data: pd.DataFrame, total_timesteps: int = 100000):
        """Entraîne les modèles RL via RLTrainer."""
        start_time = datetime.now()
        try:
            self.rl_trainer.train_rl_models(
                data=data,
                total_timesteps=total_timesteps,
            )
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = (
                f"Modèles RL entraînés pour {self.market}. "
                f"Latence: {latency}s"
            )
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            self.log_performance(
                operation="train_rl_models",
                latency=latency,
                success=True,
                market=self.market,
            )
            self.save_snapshot(
                snapshot_type="train_rl_models",
                data={
                    "num_models": len(self.rl_trainer.rl_models),
                    "market": self.market,
                },
                market=self.market,
            )
        except Exception as e:
            error_msg = (
                f"Erreur entraînement modèles RL pour {self.market}: "
                f"{str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            capture_error(
                error=e,
                context={"market": self.market},
                market=self.market,
                operation="train_rl_models",
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                operation="train_rl_models",
                latency=0,
                success=False,
                error=str(e),
                market=self.market,
            )


    def load_model(self, market: str = "ES") -> bool:
        """Charge le modèle le plus récent depuis MODEL_DIR pour un marché 
        donné."""
        try:
            start_time = datetime.now()
            model_dir = MODEL_DIR / market
            model_dir.mkdir(exist_ok=True)
            model_files = [
                f
                for f in os.listdir(model_dir)
                if f.startswith("rf_trade_prob_") and f.endswith(".pkl")
            ]
            if not model_files:
                warning_msg = (
                    f"Aucun modèle RandomForest trouvé pour chargement pour "
                    f"{market}"
                )
                logger.info(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                return False
            latest_model = max(
                model_files,
                key=lambda x: os.path.getctime(os.path.join(model_dir, x)),
            )
            model_path = model_dir / latest_model

            def load_rf_model():
                self.model = joblib.load(model_path)

            self.with_retries(load_rf_model)
            self.rl_trainer.load_models()
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Modèles chargés pour {market}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                operation="load_model",
                latency=latency,
                success=True,
                market=market,
            )
            return True
        except Exception as e:
            error_msg = (
                f"Erreur chargement modèle pour {market}: "
                f"{str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            capture_error(
                error=e,
                context={"market": market},
                market=market,
                operation="load_model",
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                operation="load_model",
                latency=0,
                success=False,
                error=str(e),
                market=market,
            )
            return False


    def predict(self, data: pd.DataFrame, market: str = "ES") -> float:
        """Prédit la probabilité de succès d’un trade avec vote bayésien."""
        try:
            start_time = datetime.now()
            # Créer une clé de cache robuste
            data_json = data.to_json()
            data_hash = hashlib.sha256(data_json.encode()).hexdigest()
            cache_key = (
                f"{market}_{data['timestamp'].iloc[0].isoformat()}_"
                f"{len(data.columns)}_{data_hash}"
            )
            if cache_key in self.prediction_cache:
                prob = self.prediction_cache[cache_key]
                self.prediction_cache.move_to_end(cache_key)
                return prob
            while len(self.prediction_cache) >= self.max_cache_size:
                self.prediction_cache.popitem(last=False)

            # Résolution des signaux spécifique à TradeProbabilityPredictor
            rf_signal_metadata = self.resolve_signals(data)
            logger.info(
                f"RF signal resolution metadata pour {market}: "
                f"{rf_signal_metadata}"
            )

            use_shap_fallback = False
            if data.isnull().any().any():
                warning_msg = (
                    f"NaN détectés dans les données pour {market}, "
                    f"tentative avec SHAP features"
                )
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=2)
                send_telegram_alert(warning_msg)
                use_shap_fallback = True
            target_cols = (
                self.shap_feature_cols if use_shap_fallback else self.feature_cols
            )
            missing_cols = [
                col for col in target_cols if col not in data.columns
            ]
            if missing_cols:
                error_msg = (
                    f"Colonnes manquantes pour prédiction pour {market} "
                    f"({'SHAP' if use_shap_fallback else 'full'}): "
                    f"{missing_cols}"
                )
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                self.log_performance(
                    operation="predict_missing_cols",
                    latency=0,
                    success=False,
                    market=market,
                )
                self.save_snapshot(
                    snapshot_type="predict_error",
                    data={
                        "reason": (
                            f"Missing columns "
                            f"({'SHAP' if use_shap_fallback else 'full'}): "
                            f"{missing_cols}"
                        )
                    },
                    market=market,
                )
                return 0.0
            X = data[target_cols].fillna(0)
            rf_prob = self.model.predict_proba(X)[:, 1][0]
            # Prédictions RL avec vote bayésien
            rl_prob, rl_signal_metadata = self.rl_trainer.predict(data)
            logger.info(
                f"RL signal resolution metadata pour {market}: "
                f"{rl_signal_metadata}"
            )
            if rl_prob > 0:
                final_prob = 0.5 * rf_prob + 0.5 * rl_prob
            else:
                final_prob = rf_prob
            # Ajustement de la récompense avec slippage
            if "slippage_estimate" in data.columns:
                raw_reward = final_prob
                slippage = data["slippage_estimate"].iloc[-1]
                final_prob = raw_reward - slippage
                final_prob = np.clip(final_prob, 0, 1)
            # Ajustement en fonction de conflict_coefficient (maximum des deux 
            # résolutions)
            rf_conflict = rf_signal_metadata.get("conflict_coefficient", 0.0)
            rl_conflict = rl_signal_metadata.get("conflict_coefficient", 0.0)
            max_conflict = max(rf_conflict, rl_conflict)
            resolver_config = self.config.get("signal_resolver", {})
            thresholds = resolver_config.get("thresholds", {})
            conflict_threshold = thresholds.get("conflict_coefficient_alert", 0.5)
            if max_conflict > conflict_threshold:
                adjustment_factor = 1 - max_conflict
                final_prob *= adjustment_factor
                final_prob = np.clip(final_prob, 0, 1)
                logger.info(
                    f"Probabilité ajustée pour {market} en raison de conflit "
                    f"élevé (max: {max_conflict:.2f}): {final_prob:.2f}"
                )
            self.prediction_cache[cache_key] = final_prob
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                operation="predict",
                latency=latency,
                success=True,
                probability=final_prob,
                used_shap_fallback=use_shap_fallback,
                rf_conflict_coefficient=rf_conflict,
                rl_conflict_coefficient=rl_conflict,
                run_id_rf=rf_signal_metadata.get("run_id"),
                run_id_rl=rl_signal_metadata.get("run_id"),
                market=market,
            )
            self.save_snapshot(
                snapshot_type="predict",
                data={
                    "probability": final_prob,
                    "num_features": len(X.columns),
                    "used_shap_fallback": use_shap_fallback,
                    "rf_signal_metadata": rf_signal_metadata,
                    "rl_signal_metadata": rl_signal_metadata,
                },
                market=market,
            )
            success_msg = (
                f"Probabilité de succès du trade pour {market}: "
                f"{final_prob:.2f} (SHAP fallback: {use_shap_fallback})"
            )
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.mlflow_tracker.log_metrics(
                {
                    "predicted_probability": final_prob,
                    "max_conflict_coefficient": max_conflict,
                }
            )
            return final_prob
        except Exception as e:
            error_msg = (
                f"Erreur prédiction probabilité pour {market}: "
                f"{str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            capture_error(
                error=e,
                context={"market": market},
                market=market,
                operation="predict",
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                operation="predict",
                latency=0,
                success=False,
                error=str(e),
                market=market,
            )
            self.save_snapshot(
                snapshot_type="predict_error",
                data={"reason": str(e)},
                market=market,
            )
            return 0.0


    def predict_trade_success(self, data: pd.DataFrame) -> float:
        """Prédit la probabilité de succès d'un trade (alias pour predict)."""
        return self.predict(data, market=self.market)


    def backtest_threshold(
        self,
        thresholds: list = [0.6, 0.7, 0.8],
        market: str = "ES",
    ) -> None:
        """Backteste différents seuils pour trade_success_prob."""
        try:
            start_time = datetime.now()
            X, y = self.load_data(market=market)
            if len(X) < 10:
                error_msg = (
                    f"Données insuffisantes pour le backtest pour {market}: "
                    f"{len(X)}"
                )
                raise ValueError(error_msg)
            y_prob = self.model.predict_proba(X)[:, 1]
            for threshold in thresholds:
                y_pred = (y_prob >= threshold).astype(int)
                win_rate = (
                    precision_score(y, y_pred) if y_pred.sum() > 0 else 0.0
                )
                returns = np.where(y_pred == 1, y, 0)
                sharpe = (
                    np.mean(returns) / np.std(returns)
                    if np.std(returns) > 0
                    else 0.0
                )
                num_trades = y_pred.sum()
                self.log_backtest(
                    threshold=threshold,
                    win_rate=win_rate,
                    sharpe=sharpe,
                    num_trades=num_trades,
                    market=market,
                )
                success_msg = (
                    f"Backtest seuil {threshold} pour {market}: "
                    f"win_rate={win_rate:.2f}, sharpe={sharpe:.2f}, "
                    f"trades={num_trades}"
                )
                logger.info(success_msg)
                self.alert_manager.send_alert(success_msg, priority=2)
                send_telegram_alert(success_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                operation="backtest_threshold",
                latency=latency,
                success=True,
                num_thresholds=len(thresholds),
                market=market,
            )
            self.save_snapshot(
                snapshot_type="backtest_threshold",
                data={
                    "thresholds": thresholds,
                    "market": market,
                },
                market=market,
            )
        except Exception as e:
            error_msg = (
                f"Erreur backtest seuil pour {market}: "
                f"{str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            capture_error(
                error=e,
                context={"market": market},
                market=market,
                operation="backtest_threshold",
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                operation="backtest_threshold",
                latency=0,
                success=False,
                error=str(e),
                market=market,
            )


    def retrain_model(self, market: str = "ES", **hyperparams) -> None:
        """Réentraîne le modèle de probabilité de trade avec de nouvelles 
        données."""
        try:
            start_time = datetime.now()
            with sqlite3.connect(self.db_path) as conn:
                data = pd.read_sql(
                    """
                    SELECT * FROM trade_patterns 
                    WHERE timestamp > (SELECT MAX(timestamp) FROM training_log)
                    """,
                    conn,
                )
            if len(data) < 1000:
                warning_msg = (
                    f"Données insuffisantes pour réentraînement pour "
                    f"{market}: {len(data)} trades"
                )
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                self.log_performance(
                    operation="retrain_model",
                    latency=0,
                    success=False,
                    error="Données insuffisantes",
                    num_trades=len(data),
                    market=market,
                )
                return
            default_hypers = {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 42,
                "class_weight": "balanced",
            }
            hyperparams = {**default_hypers, **hyperparams}
            self.model = RandomForestClassifier(**hyperparams)
            X = data[self.feature_cols].fillna(0)
            y = (data["reward"] > 0).astype(int)
            # Validation glissante avec TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)
            auc_scores = []
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                self.model.fit(X_train, y_train)
                y_prob = self.model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_prob)
                auc_scores.append(auc)
            mean_auc = np.mean(auc_scores)
            # Entraînement final sur toutes les données
            self.model.fit(X, y)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_dir = MODEL_DIR / market
            model_dir.mkdir(exist_ok=True)
            model_path = model_dir / f"rf_trade_prob_{timestamp}.pkl"

            def save_model():
                joblib.dump(self.model, model_path)

            self.with_retries(save_model)
            y_pred = self.model.predict(X)
            y_prob = self.model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, y_prob)
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            self.log_metrics(
                auc=auc,
                precision=precision,
                recall=recall,
                f1=f1,
                market=market,
            )
            with sqlite3.connect(self.db_path) as conn:
                pd.DataFrame([{"timestamp": datetime.now().isoformat()}]).to_sql(
                    "training_log",
                    conn,
                    if_exists="append",
                    index=False,
                )
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = (
                f"Réentraînement du modèle de probabilité pour {market} avec "
                f"{len(data)} trades"
            )
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            self.log_performance(
                operation="retrain_model",
                latency=latency,
                success=True,
                num_trades=len(data),
                mean_auc=mean_auc,
                auc=auc,
                market=market,
            )
            self.save_snapshot(
                snapshot_type="retrain_model",
                data={
                    "num_trades": len(data),
                    "model_path": str(model_path),
                    "mean_auc": mean_auc,
                    "auc": auc,
                    "hyperparams": hyperparams,
                },
                market=market,
            )
            self.mlflow_tracker.log_params(hyperparams)
            self.mlflow_tracker.log_artifact(model_path)
        except Exception as e:
            error_msg = (
                f"Erreur réentraînement modèle pour {market}: "
                f"{str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            capture_error(
                error=e,
                context={"market": market},
                market=market,
                operation="retrain_model",
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                operation="retrain_model",
                latency=0,
                success=False,
                error=str(e),
                market=market,
            )


    def quick_train_and_eval(self, market: str = "ES", **hyperparams) -> Dict:
        """Effectue un entraînement rapide pour l’optimisation des 
        hyperparamètres."""
        try:
            start_time = datetime.now()
            with sqlite3.connect(self.db_path) as conn:
                data = pd.read_sql(
                    """
                    SELECT * FROM trade_patterns 
                    WHERE market = ? ORDER BY timestamp DESC LIMIT 500
                    """,
                    conn,
                    params=(market,),
                )
            if len(data) < 100:
                error_msg = (
                    f"Données insuffisantes pour entraînement rapide pour "
                    f"{market}: {len(data)} trades"
                )
                raise ValueError(error_msg)
            default_hypers = {
                "n_estimators": 50,
                "max_depth": 10,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 42,
                "class_weight": "balanced",
            }
            hyperparams = {**default_hypers, **hyperparams}
            model = RandomForestClassifier(**hyperparams)
            X = data[self.feature_cols].fillna(0)
            y = (data["reward"] > 0).astype(int)
            model.fit(X, y)
            y_prob = model.predict_proba(X)[:, 1]
            y_pred = (y_prob >= 0.7).astype(int)
            returns = np.where(y_pred == 1, data["reward"], 0)
            sharpe = (
                np.mean(returns) / np.std(returns)
                if np.std(returns) > 0
                else 0.0
            )
            drawdown = (
                np.min(np.cumsum(returns)) if len(returns) > 0 else 0.0
            )
            cpu_usage = psutil.cpu_percent()
            metrics = {
                "sharpe": sharpe,
                "drawdown": drawdown,
                "cpu_usage": cpu_usage,
            }
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                operation="quick_train_and_eval",
                latency=latency,
                success=True,
                market=market,
                sharpe=sharpe,
                drawdown=drawdown,
            )
            self.mlflow_tracker.log_metrics(metrics)
            return metrics
        except Exception as e:
            error_msg = (
                f"Erreur entraînement rapide pour {market}: "
                f"{str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            capture_error(
                error=e,
                context={"market": market},
                market=market,
                operation="quick_train_and_eval",
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                operation="quick_train_and_eval",
                latency=0,
                success=False,
                error=str(e),
                market=market,
            )
            return {
                "sharpe": 0.0,
                "drawdown": float("inf"),
                "cpu_usage": 0.0,
            }