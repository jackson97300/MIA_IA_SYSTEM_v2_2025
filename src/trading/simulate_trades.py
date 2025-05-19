```python
# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/trading/simulate_trades.py
# Rôle : Simule des trades hors ligne pour MIA_IA_SYSTEM_v2_2025, incluant récompenses adaptatives et paper trading.
#
# Version : 2.1.6
# Date : 2025-05-15
#
# Rôle : Simule des trades hors ligne avec récompenses adaptatives (méthode 5) basées sur news_impact_score et predicted_vix,
#        supporte le paper trading via Sierra Chart (API Teton), utilise la mémoire contextuelle (Phase 7) via store_pattern,
#        et génère des snapshots JSON (option compressée), des sauvegardes S3, des graphiques matplotlib/seaborn, et des alertes
#        standardisées (Phase 8). Compatible avec la simulation de trading (Phase 12) et l’ensemble learning (Phase 16).
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0
# - numpy>=1.26.4,<2.0.0
# - matplotlib>=3.8.0,<4.0.0
# - seaborn>=0.13.0,<1.0.0
# - psutil>=5.9.8,<6.0.0
# - pyyaml>=6.0.0,<7.0.0
# - boto3>=1.26.0,<2.0.0
# - logging, os, json, hashlib, datetime, signal, time, gzip
# - src.model.router.main_router
# - src.envs.trading_env
# - src.model.utils_model
# - src.features.neural_pipeline
# - src.mind.mind
# - src.model.adaptive_learner
# - src.model.utils.config_manager
# - src.model.utils.alert_manager
# - src.utils.telegram_alert
# - src.trading.trade_executor
#
# Inputs :
# - Données de trading (pd.DataFrame avec 351 features pour entraînement ou 150 SHAP features pour inférence)
# - Configuration via config/market_config.yaml
# - config/feature_sets.yaml
#
# Outputs :
# - Trades simulés dans data/trades/trades_simulated.csv
# - Logs dans data/logs/simulate_trades.log
# - Logs de performance dans data/logs/trading/simulate_trades_performance.csv
# - Snapshots JSON dans data/trading/simulation_snapshots/*.json (option *.json.gz)
# - Dashboard JSON dans data/trading/simulate_trades_dashboard.json
# - Visualisations dans data/figures/simulations/*.png
# - Sauvegardes dans data/checkpoints/simulate_trades/*.json.gz
#
# Notes :
# - Compatible avec 351 features (entraînement, incluant iv_skew) et 150 SHAP features (inférence) définies dans config/feature_sets.yaml.
# - Utilise IQFeed exclusivement via data_provider.py pour les données d’entrée.
# - Implémente retries (max 3, délai 2^attempt secondes) pour les opérations critiques.
# - Intègre récompenses adaptatives (méthode 5) avec news_impact_score et predicted_vix.
# - Supporte paper trading via Sierra Chart (API Teton).
# - Sauvegardes incrémentielles (5 min), distribuées S3 (15 min), versionnées (5 versions).
# - Intègre confidence_drop_rate (Phase 8) calculé dynamiquement à chaque étape.
# - Intègre seuils entry_freq_max et entry_freq_min_interval de market_config.yaml.
# - Tests unitaires disponibles dans tests/test_simulate_trades.py.
# - Corrections appliquées :
#   - Mise à jour de la version à 2.1.6 (2025-05-15) pour aligner avec market_config.yaml et feature_sets.yaml.
#   - Utilisation de market_config.yaml par défaut dans simulate_trading.
#   - Ajustement à 351 features avec ajout de iv_skew, bid_ask_imbalance, trade_aggressiveness dans critical_cols.
#   - Uniformisation des imports avec get_config.
#   - Implémentation de la sauvegarde S3 dans save_checkpoint.
#   - Validation des colonnes pour neural_pipeline.run.
#   - Calcul dynamique de confidence_drop_rate à chaque étape.
#   - Gestion dynamique des neural_feature_{i}.

import gzip
import hashlib
import json
import logging
import os
import signal
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns

from src.envs.trading_env import TradingEnv
from src.features.neural_pipeline import NeuralPipeline
from src.mind.mind import MindEngine
from src.model.adaptive_learner import store_pattern
from src.model.router.main_router import MainRouter
from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import get_config
from src.model.utils_model import export_logs
from src.trading.trade_executor import TradeExecutor
from src.utils.telegram_alert import send_telegram_alert

# Chemins relatifs
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
SNAPSHOT_DIR = BASE_DIR / "data" / "trading" / "simulation_snapshots"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints" / "simulate_trades"
DASHBOARD_PATH = BASE_DIR / "data" / "trading" / "simulate_trades_dashboard.json"
CSV_LOG_PATH = (
    BASE_DIR / "data" / "logs" / "trading" / "simulate_trades_performance.csv"
)
OUTPUT_DIR = BASE_DIR / "data" / "trades"
FIGURE_DIR = BASE_DIR / "data" / "figures" / "simulations"

# Configuration logging
os.makedirs(BASE_DIR / "data" / "logs", exist_ok=True)
logging.basicConfig(
    filename=BASE_DIR / "data" / "logs" / "simulate_trades.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Constantes
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Secondes, exponentiel


def calculate_reward(news_impact_score: float, predicted_vix: float) -> float:
    """
    Calcule une récompense adaptative basée sur news_impact_score et predicted_vix (méthode 5).

    Args:
        news_impact_score (float): Score d’impact des nouvelles.
        predicted_vix (float): VIX prédit.

    Returns:
        float: Récompense calculée.
    """
    try:
        reward = news_impact_score * 10 - predicted_vix * 0.5  # Simulation
        return float(reward)
    except Exception as e:
        error_msg = f"Erreur calcul récompense: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        send_telegram_alert(error_msg)
        return 0.0


class TradeSimulator:
    """
    Classe pour simuler des trades hors ligne, incluant récompenses adaptatives et paper trading via Sierra Chart.
    """

    def __init__(self, config_path: str = str(BASE_DIR / "config" / "market_config.yaml")):
        """
        Initialise le simulateur de trades.

        Args:
            config_path (str): Chemin vers la configuration du marché.
        """
        self.alert_manager = AlertManager()
        SNAPSHOT_DIR.mkdir(exist_ok=True)
        OUTPUT_DIR.mkdir(exist_ok=True)
        FIGURE_DIR.mkdir(exist_ok=True)
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        CSV_LOG_PATH.parent.mkdir(exist_ok=True)
        signal.signal(signal.SIGINT, self.handle_sigint)
        self.last_checkpoint_time = datetime.now()
        self.last_distributed_checkpoint_time = datetime.now()
        self.last_trade_time = None
        self.trade_count_hour = 0
        self.trade_timestamps = []

        self.log_buffer = []
        self.trade_buffer = []
        self.neural_cache = {}
        self.simulation_cache = {}
        try:
            self.config = self.load_config(config_path)
            required_config = [
                "thresholds",
                "simulation",
                "logging",
                "cache",
                "features",
            ]
            missing_config = [key for key in required_config if key not in self.config]
            if missing_config:
                raise ValueError(f"Clés de configuration manquantes: {missing_config}")
            self.mind_engine = MindEngine()
            self.trade_executor = TradeExecutor(
                config_path=str(config_path)
            )  # Instance de TradeExecutor
            self.buffer_size = self.config.get("logging", {}).get("buffer_size", 100)
            self.max_cache_size = self.config.get("cache", {}).get(
                "max_cache_size", 1000
            )

            self.performance_thresholds = {
                "min_sharpe": self.config.get("thresholds", {}).get("min_sharpe", 0.5),
                "max_drawdown": self.config.get("thresholds", {}).get(
                    "max_drawdown", -1000.0
                ),
                "min_profit_factor": self.config.get("thresholds", {}).get(
                    "min_profit_factor", 1.2
                ),
                "entry_freq_max": self.config.get("thresholds", {}).get(
                    "entry_freq_max", 5
                ),
                "entry_freq_min_interval": self.config.get("thresholds", {}).get(
                    "entry_freq_min_interval", 5
                ),
            }

            for key, value in self.performance_thresholds.items():
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Seuil invalide pour {key}: {value}")
                if key == "min_sharpe" and value <= 0:
                    raise ValueError(f"Seuil {key} doit être positif: {value}")
                if key == "max_drawdown" and value >= 0:
                    raise ValueError(f"Seuil {key} doit être négatif: {value}")
                if key == "min_profit_factor" and value <= 0:
                    raise ValueError(f"Seuil {key} doit être positif: {value}")
                if key == "entry_freq_max" and value <= 0:
                    raise ValueError(f"Seuil {key} doit être positif: {value}")
                if key == "entry_freq_min_interval" and value <= 0:
                    raise ValueError(f"Seuil {key} doit être positif: {value}")

            self.use_neural_pipeline = self.config.get("simulation", {}).get(
                "use_neural_pipeline", True
            )

            logger.info("TradeSimulator initialisé avec succès")
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert("TradeSimulator initialisé", priority=2)
                send_telegram_alert("TradeSimulator initialisé")
            self.log_performance("init", 0, success=True)
            self.save_checkpoint(incremental=True)
        except Exception as e:
            error_msg = f"Erreur initialisation TradeSimulator: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            raise

    def handle_sigint(self, signum, frame):
        """Gère l'arrêt propre sur SIGINT."""
        snapshot = {"timestamp": datetime.now().isoformat(), "status": "SIGINT"}
        try:
            self.save_snapshot("sigint", snapshot, compress=False)
            self.save_checkpoint(incremental=True)
            logger.info("Arrêt propre sur SIGINT, snapshot et checkpoint sauvegardés")
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(
                    "Arrêt propre sur SIGINT, snapshot et checkpoint sauvegardés",
                    priority=2,
                )
                send_telegram_alert(
                    "Arrêt propre sur SIGINT, snapshot et checkpoint sauvegardés"
                )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde snapshot SIGINT: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
        exit(0)

    def save_snapshot(
        self, snapshot_type: str, data: Dict, compress: bool = False
    ) -> None:
        """Sauvegarde un instantané des résultats, avec option de compression gzip."""
        try:
            start_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = {
                "timestamp": timestamp,
                "type": snapshot_type,
                "data": data,
                "buffer_size": len(self.log_buffer),
            }
            path = SNAPSHOT_DIR / f"snapshot_{snapshot_type}_{timestamp}.json"
            os.makedirs(path.parent, exist_ok=True)
            if compress:
                with gzip.open(f"{path}.gz", "wt", encoding="utf-8") as f:
                    json.dump(snapshot, f, indent=4)
                save_path = f"{path}.gz"
            else:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(snapshot, f, indent=4)
                save_path = path
            latency = time.time() - start_time
            logger.info(f"Snapshot {snapshot_type} sauvegardé: {save_path}")
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(
                    f"Snapshot {snapshot_type} sauvegardé: {save_path}", priority=1
                )
                send_telegram_alert(f"Snapshot {snapshot_type} sauvegardé: {save_path}")
            self.log_performance("save_snapshot", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
            self.log_performance("save_snapshot", 0, success=False, error=str(e))

    def save_checkpoint(self, incremental: bool = True, distributed: bool = False):
        """Sauvegarde l’état du simulateur (incrémentiel, distribué, versionné)."""
        try:
            start_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint = {
                "timestamp": timestamp,
                "log_buffer": self.log_buffer[-100:],  # Limiter la taille
                "trade_buffer": self.trade_buffer[-100:],  # Limiter la taille
                "neural_cache": {k: True for k in self.neural_cache},  # Simplifié
                "simulation_cache": {
                    k: {"log_rows": v["log_rows"][-100:]}
                    for k, v in self.simulation_cache.items()
                },
            }
            checkpoint_path = CHECKPOINT_DIR / f"checkpoint_{timestamp}.json.gz"
            os.makedirs(checkpoint_path.parent, exist_ok=True)
            with gzip.open(checkpoint_path, "wt", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=4)

            # Gestion des versions (max 5)
            checkpoints = sorted(CHECKPOINT_DIR.glob("checkpoint_*.json.gz"))
            if len(checkpoints) > 5:
                for old_checkpoint in checkpoints[:-5]:
                    os.remove(old_checkpoint)

            # Sauvegarde distribuée vers AWS S3
            if distributed:
                if not self.config.get("s3_bucket"):
                    warning_msg = "S3 bucket non configuré, sauvegarde cloud ignorée."
                    logger.warning(warning_msg)
                    if self.config.get("notifications_enabled", True):
                        self.alert_manager.send_alert(warning_msg, priority=3)
                        send_telegram_alert(warning_msg)
                else:
                    if not (
                        os.environ.get("AWS_ACCESS_KEY_ID")
                        and os.environ.get("AWS_SECRET_ACCESS_KEY")
                    ) and not os.path.exists(os.path.expanduser("~/.aws/credentials")):
                        warning_msg = "Clés AWS non trouvées. Configurez ~/.aws/credentials ou AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY."
                        logger.warning(warning_msg)
                        if self.config.get("notifications_enabled", True):
                            self.alert_manager.send_alert(warning_msg, priority=3)
                            send_telegram_alert(warning_msg)
                    else:
                        s3_client = boto3.client("s3")
                        backup_path = (
                            f"{self.config['s3_prefix']}checkpoint_{timestamp}.json.gz"
                        )

                        def upload_s3():
                            s3_client.upload_file(
                                str(checkpoint_path),
                                self.config["s3_bucket"],
                                backup_path,
                            )

                        self.with_retries(upload_s3)
                        logger.info(f"Sauvegarde S3 effectuée: {backup_path}")

            latency = time.time() - start_time
            logger.info(
                f"Checkpoint {'incrémentiel' if incremental else 'distribué'} sauvegardé: {checkpoint_path}"
            )
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(
                    f"Checkpoint {'incrémentiel' if incremental else 'distribué'} sauvegardé: {checkpoint_path}",
                    priority=1,
                )
                send_telegram_alert(
                    f"Checkpoint {'incrémentiel' if incremental else 'distribué'} sauvegardé: {checkpoint_path}"
                )
            self.log_performance(
                "save_checkpoint",
                latency,
                success=True,
                incremental=incremental,
                distributed=distributed,
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde checkpoint: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
            self.log_performance("save_checkpoint", 0, success=False, error=str(e))

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
                    f"retry_attempt_{attempt+1}", latency, success=True
                )
                return result
            except Exception as e:
                if attempt == max_attempts - 1:
                    error_msg = f"Échec après {max_attempts} tentatives: {str(e)}\n{traceback.format_exc()}"
                    logger.error(error_msg)
                    if self.config.get("notifications_enabled", True):
                        self.alert_manager.send_alert(error_msg, priority=3)
                        send_telegram_alert(error_msg)
                    self.log_performance(
                        f"retry_attempt_{attempt+1}",
                        time.time() - start_time,
                        success=False,
                        error=str(e),
                    )
                    return None
                delay = delay_base**attempt
                logger.warning(f"Tentative {attempt+1} échouée, retry après {delay}s")
                time.sleep(delay)

    def load_config(self, config_path: Union[str, Path]) -> Dict:
        """
        Charge la configuration avec repli sur les valeurs par défaut.

        Args:
            config_path (Union[str, Path]): Chemin vers le fichier de configuration.

        Returns:
            Dict: Configuration chargée.
        """
        start_time = time.time()
        try:
            config = self.with_retries(lambda: get_config(config_path))
            if not config:
                raise ValueError("Configuration vide ou non trouvée")
            self.log_performance("load_config", time.time() - start_time, success=True)
            logger.info("Configuration chargée avec succès")
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(
                    "Configuration chargée avec succès", priority=1
                )
                send_telegram_alert("Configuration chargée avec succès")
            return config
        except Exception as e:
            error_msg = (
                f"Erreur chargement configuration: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
            self.log_performance(
                "load_config", time.time() - start_time, success=False, error=str(e)
            )
            return {
                "thresholds": {
                    "min_sharpe": 0.5,
                    "max_drawdown": -1000.0,
                    "min_profit_factor": 1.2,
                    "entry_freq_max": 5,
                    "entry_freq_min_interval": 5,
                },
                "simulation": {"use_neural_pipeline": True},
                "logging": {"buffer_size": 100},
                "cache": {"max_cache_size": 1000},
                "features": {"observation_dims": {"training": 351, "inference": 150}},
                "s3_bucket": None,
                "s3_prefix": "simulate_trades/",
                "notifications_enabled": True,
            }

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
            **kwargs: Paramètres supplémentaires.
        """
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
            cpu_percent = psutil.cpu_percent()
            if memory_usage > 1024:
                error_msg = (
                    f"ALERTE: Utilisation mémoire élevée ({memory_usage:.2f} MB)"
                )
                logger.warning(error_msg)
                if self.config.get("notifications_enabled", True):
                    self.alert_manager.send_alert(error_msg, priority=5)
                    send_telegram_alert(error_msg)
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

                def save_log():
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

                self.with_retries(save_log)
                self.log_buffer = []
            logger.info(
                f"Performance journalisée pour {operation}. CPU: {cpu_percent}%"
            )
        except Exception as e:
            error_msg = (
                f"Erreur journalisation performance: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)

    def save_dashboard_status(
        self, status: Dict, status_file: Union[str, Path] = DASHBOARD_PATH
    ) -> None:
        """
        Sauvegarde l'état de la simulation pour mia_dashboard.py.

        Args:
            status (Dict): État de la simulation.
            status_file (Union[str, Path]): Chemin du fichier JSON.
        """
        start_time = time.time()
        try:
            def write_status():
                with open(status_file, "w", encoding="utf-8") as f:
                    json.dump(status, f, indent=4)

            self.with_retries(write_status)
            self.log_performance(
                "save_dashboard_status", time.time() - start_time, success=True
            )
            logger.info(f"État sauvegardé dans {status_file}")
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(
                    f"État sauvegardé dans {status_file}", priority=1
                )
                send_telegram_alert(f"État sauvegardé dans {status_file}")
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde dashboard: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
            self.log_performance(
                "save_dashboard_status",
                time.time() - start_time,
                success=False,
                error=str(e),
            )

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
            if num_cols >= 351:
                feature_set = "training"
            elif num_cols >= 150:
                feature_set = "inference"
            else:
                error_msg = f"Nombre de features insuffisant: {num_cols}, attendu ≥150"
                raise ValueError(error_msg)
            self.log_performance(
                "detect_feature_set",
                time.time() - start_time,
                success=True,
                num_columns=num_cols,
            )
            logger.info(f"Feature set détecté: {feature_set}")
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(
                    f"Feature set détecté: {feature_set}", priority=1
                )
                send_telegram_alert(f"Feature set détecté: {feature_set}")
            return feature_set
        except Exception as e:
            error_msg = f"Erreur détection ensemble features: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
            self.log_performance(
                "detect_feature_set",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            return "inference"

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
                error_msg = "Colonne 'timestamp' manquante, imputation impossible"
                raise ValueError(error_msg)

            data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
            if data["timestamp"].isna().any():
                last_valid = (
                    data["timestamp"].dropna().iloc[-1]
                    if not data["timestamp"].dropna().empty
                    else pd.Timestamp.now()
                )
                data["timestamp"] = data["timestamp"].fillna(last_valid)
                logger.info(
                    f"Valeurs invalides dans 'timestamp', imputées avec {last_valid}"
                )
                if self.config.get("notifications_enabled", True):
                    self.alert_manager.send_alert(
                        f"Valeurs invalides dans 'timestamp', imputées avec {last_valid}",
                        priority=2,
                    )
                    send_telegram_alert(
                        f"Valeurs invalides dans 'timestamp', imputées avec {last_valid}"
                    )
            self.log_performance(
                "impute_timestamp",
                time.time() - start_time,
                success=True,
                num_rows=len(data),
            )
            return data
        except Exception as e:
            error_msg = (
                f"Erreur imputation timestamp: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
            self.log_performance(
                "impute_timestamp",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            return data

    def validate_data(self, data: pd.DataFrame) -> None:
        """
        Valide les données d’entrée (351 features pour entraînement, 150 SHAP pour inférence).

        Args:
            data (pd.DataFrame): Données à valider.

        Raises:
            ValueError: Si les données sont invalides.
        """
        start_time = time.time()
        try:
            feature_set = self.detect_feature_set(data)
            expected_features = 351 if feature_set == "training" else 150

            # Valider les colonnes via feature_sets.yaml
            feature_sets = self.with_retries(
                lambda: get_config(BASE_DIR / "config" / "feature_sets.yaml")
            )
            expected_cols = (
                feature_sets.get("feature_sets", {})
                .get("ES", {})
                .get("training" if feature_set == "training" else "inference", [])
            )
            if len(expected_cols) != expected_features:
                error_msg = f"Nombre de colonnes attendu incorrect dans feature_sets.yaml: {len(expected_cols)} au lieu de {expected_features}"
                raise ValueError(error_msg)
            missing_cols = [col for col in expected_cols if col not in data.columns]
            if missing_cols:
                error_msg = f"Colonnes manquantes dans les données: {missing_cols}"
                raise ValueError(error_msg)
            if len(data.columns) < expected_features:
                error_msg = f"Nombre de features insuffisant: {len(data.columns)} < {expected_features} pour ensemble {feature_set}"
                raise ValueError(error_msg)

            critical_cols = [
                "vix",
                "neural_regime",
                "predicted_volatility",
                "trade_frequency_1s",
                "close",
                "rsi_14",
                "gex",
                "timestamp",
                "open",
                "high",
                "low",
                "volume",
                "atr_14",
                "adx_14",
                "bid_size_level_1",
                "ask_size_level_1",
                "news_impact_score",
                "predicted_vix",
                "iv_skew",
                "bid_ask_imbalance",
                "trade_aggressiveness",
            ]
            for col in critical_cols:
                if col in data.columns:
                    if col != "timestamp" and data[col].isnull().any():
                        error_msg = f"Colonne {col} contient des NaN"
                        raise ValueError(error_msg)
                    if col not in [
                        "timestamp",
                        "neural_regime",
                    ] and not pd.api.types.is_numeric_dtype(data[col]):
                        error_msg = (
                            f"Colonne {col} n’est pas numérique: {data[col].dtype}"
                        )
                        raise ValueError(error_msg)

            data = self.impute_timestamp(data)

            self.log_performance(
                "validate_data",
                time.time() - start_time,
                success=True,
                num_columns=len(data.columns),
            )
            logger.info(f"Données validées pour ensemble {feature_set}")
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(
                    f"Données validées pour ensemble {feature_set}", priority=1
                )
                send_telegram_alert(f"Données validées pour ensemble {feature_set}")
            self.save_snapshot(
                "validate_data", {"num_features": len(data.columns)}, compress=False
            )
        except Exception as e:
            error_msg = f"Erreur validation données: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
            self.log_performance(
                "validate_data", time.time() - start_time, success=False, error=str(e)
            )
            raise

    def compute_metrics(self, rewards: List[float]) -> Dict[str, float]:
        """
        Calcule les métriques de performance.

        Args:
            rewards (List[float]): Liste des rewards des trades.

        Returns:
            Dict[str, float]: Métriques calculées.
        """
        start_time = time.time()
        try:
            logger.info("Calcul des métriques de performance")
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(
                    "Calcul des métriques de performance", priority=2
                )
                send_telegram_alert("Calcul des métriques de performance")

            equity = np.cumsum(rewards)
            drawdown = float(np.min(equity - np.maximum.accumulate(equity)))
            sharpe = (
                float(np.mean(rewards) / np.std(rewards))
                if np.std(rewards) > 0
                else 0.0
            )
            profit_factor = (
                float(
                    sum(r for r in rewards if r > 0)
                    / abs(sum(r for r in rewards if r < 0))
                )
                if any(r < 0 for r in rewards)
                else float("inf")
            )
            confidence_drop_rate = 1.0 - min(
                profit_factor / self.performance_thresholds["min_profit_factor"], 1.0
            )  # Phase 8

            metrics = {
                "sharpe": sharpe,
                "drawdown": drawdown,
                "total_reward": float(np.sum(rewards)),
                "profit_factor": profit_factor,
                "confidence_drop_rate": confidence_drop_rate,
            }

            for metric, threshold in self.performance_thresholds.items():
                if metric in metrics:
                    value = metrics[metric]
                    if (metric == "max_drawdown" and value < threshold) or (
                        metric != "max_drawdown" and value < threshold
                    ):
                        error_msg = (
                            f"Seuil non atteint pour {metric}: {value} < {threshold}"
                        )
                        logger.warning(error_msg)
                        if self.config.get("notifications_enabled", True):
                            self.alert_manager.send_alert(error_msg, priority=3)
                            send_telegram_alert(error_msg)

            self.log_performance(
                "compute_metrics",
                time.time() - start_time,
                success=True,
                num_trades=len(rewards),
            )
            logger.info(f"Métriques calculées: {metrics}")
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(f"Métriques calculées: {metrics}", priority=1)
                send_telegram_alert(f"Métriques calculées: {metrics}")
            return metrics
        except Exception as e:
            error_msg = f"Erreur compute_metrics: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
            self.log_performance(
                "compute_metrics", time.time() - start_time, success=False, error=str(e)
            )
            return {
                "sharpe": 0.0,
                "drawdown": 0.0,
                "total_reward": 0.0,
                "profit_factor": 0.0,
                "confidence_drop_rate": 0.0,
            }

    def generate_visualizations(
        self, trades_df: pd.DataFrame, mode: str, timestamp: str
    ) -> None:
        """
        Génère des visualisations des résultats de la simulation.

        Args:
            trades_df (pd.DataFrame): DataFrame des trades simulés.
            mode (str): Mode de trading.
            timestamp (str): Horodatage pour les fichiers.
        """
        start_time = time.time()
        try:
            if trades_df.empty:
                error_msg = "Aucun trade pour générer des visualisations"
                logger.warning(error_msg)
                if self.config.get("notifications_enabled", True):
                    self.alert_manager.send_alert(error_msg, priority=3)
                    send_telegram_alert(error_msg)
                return

            plt.figure(figsize=(12, 6))
            plt.plot(trades_df["timestamp"], trades_df["balance"], label="Balance")
            plt.title(f"Balance au fil du temps (Mode: {mode})")
            plt.xlabel("Timestamp")
            plt.ylabel("Balance")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()
            balance_path = FIGURE_DIR / f"balance_{mode}_{timestamp}.png"
            plt.savefig(balance_path)
            plt.close()

            plt.figure(figsize=(12, 6))
            sns.histplot(trades_df["reward"], kde=True, color="green")
            plt.title(f"Distribution des Rewards (Mode: {mode})")
            plt.xlabel("Reward")
            plt.ylabel("Fréquence")
            plt.grid(True, linestyle="--", alpha=0.7)
            dist_path = FIGURE_DIR / f"rewards_{mode}_{timestamp}.png"
            plt.savefig(dist_path)
            plt.close()

            equity = trades_df["balance"].cumsum()
            drawdown = equity - equity.cummax()
            plt.figure(figsize=(12, 6))
            plt.plot(trades_df["timestamp"], drawdown, label="Drawdown", color="red")
            plt.title(f"Drawdown au fil du temps (Mode: {mode})")
            plt.xlabel("Timestamp")
            plt.ylabel("Drawdown")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()
            drawdown_path = FIGURE_DIR / f"drawdown_{mode}_{timestamp}.png"
            plt.savefig(drawdown_path)
            plt.close()

            plt.figure(figsize=(12, 6))
            plt.hist(
                trades_df["confidence_drop_rate"],
                bins=30,
                color="purple",
                edgecolor="black",
                alpha=0.7,
            )
            plt.title(f"Distribution du Confidence Drop Rate (Mode: {mode})")
            plt.xlabel("Confidence Drop Rate")
            plt.ylabel("Nombre de Trades")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.savefig(FIGURE_DIR / f"confidence_drop_{mode}_{timestamp}.png")
            plt.close()

            self.log_performance(
                "generate_visualizations",
                time.time() - start_time,
                success=True,
                num_trades=len(trades_df),
            )
            logger.info(f"Visualisations générées sous {FIGURE_DIR}")
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(
                    f"Visualisations générées sous {FIGURE_DIR}", priority=2
                )
                send_telegram_alert(f"Visualisations générées sous {FIGURE_DIR}")
        except Exception as e:
            error_msg = (
                f"Erreur génération visualisations: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
            self.log_performance(
                "generate_visualizations",
                time.time() - start_time,
                success=False,
                error=str(e),
            )

    def check_trade_frequency(self, current_time: pd.Timestamp) -> bool:
        """
        Vérifie si un nouveau trade respecte les seuils de fréquence.

        Args:
            current_time (pd.Timestamp): Timestamp du trade actuel.

        Returns:
            bool: True si le trade est autorisé, False sinon.
        """
        try:
            # Vérifier l'intervalle minimum entre trades
            if self.last_trade_time is not None:
                time_diff = (current_time - self.last_trade_time).total_seconds() / 60
                if (
                    time_diff
                    < self.performance_thresholds["entry_freq_min_interval"]
                ):
                    logger.warning(
                        f"Trade refusé: intervalle trop court ({time_diff:.2f} min < {self.performance_thresholds['entry_freq_min_interval']} min)"
                    )
                    return False

            # Vérifier le nombre maximum de trades par heure
            self.trade_timestamps = [
                t
                for t in self.trade_timestamps
                if (current_time - t).total_seconds() <= 3600
            ]
            if len(self.trade_timestamps) >= self.performance_thresholds["entry_freq_max"]:
                logger.warning(
                    f"Trade refusé: limite de {self.performance_thresholds['entry_freq_max']} trades/heure atteinte"
                )
                return False

            return True
        except Exception as e:
            error_msg = f"Erreur vérification fréquence trades: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
            return False

    def simulate_trades(
        self,
        data: pd.DataFrame,
        paper_trading: bool = False,
        asset: str = "ES",
        mode: str = "range",
    ) -> None:
        """
        Simule des trades avec récompenses adaptatives et option de paper trading.

        Args:
            data (pd.DataFrame): Données des features.
            paper_trading (bool): Si True, utilise paper trading via Sierra Chart.
            asset (str): Symbole du marché (ex. ES, NQ).
            mode (str): Mode de trading ("trend", "range", "defensive").
        """
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv_path = OUTPUT_DIR / "trades_simulated.csv"
        try:
            self.validate_data(data)
            logger.info(
                f"Simulation trades pour {asset.upper()} (Mode: {mode}, Paper: {paper_trading})"
            )
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(
                    f"Simulation trades pour {asset.upper()} (Mode: {mode}, Paper: {paper_trading})",
                    priority=2,
                )
                send_telegram_alert(
                    f"Simulation trades pour {asset.upper()} (Mode: {mode}, Paper: {paper_trading})"
                )

            # Gérer les sauvegardes incrémentielles et distribuées
            if (
                datetime.now() - self.last_checkpoint_time
            ).total_seconds() >= 300:  # 5 minutes
                self.save_checkpoint(incremental=True)
                self.last_checkpoint_time = datetime.now()
            if (
                datetime.now() - self.last_distributed_checkpoint_time
            ).total_seconds() >= 900:  # 15 minutes
                self.save_checkpoint(incremental=False, distributed=True)
                self.last_distributed_checkpoint_time = datetime.now()

            trades = []
            confidence_drop_rate = 0.0  # Phase 8
            for i in range(len(data)):
                step_start = datetime.now()
                try:
                    current_time = pd.to_datetime(data.iloc[i]["timestamp"])
                    if not self.check_trade_frequency(current_time):
                        continue

                    reward = calculate_reward(
                        data.iloc[i]["news_impact_score"], data.iloc[i]["predicted_vix"]
                    )
                    trade_data = {
                        "trade_id": f"sim_{i}",
                        "action": "buy",
                        "price": float(data.iloc[i]["close"]),
                        "size": 1,
                        "order_type": "market",
                    }

                    def exec_trade():
                        return self.trade_executor.execute_trade(
                            trade_data,
                            mode="paper" if paper_trading else "sim",
                            context_data=data.iloc[[i]],
                        )

                    trade = self.with_retries(exec_trade)
                    if trade is None:
                        raise ValueError("Échec de l’exécution du trade")
                    trades.append(trade)

                    # Mettre à jour les compteurs de fréquence
                    self.trade_timestamps.append(current_time)
                    self.last_trade_time = current_time

                    # Calculer confidence_drop_rate dynamiquement
                    temp_rewards = [t.get("reward", 0.0) for t in trades]
                    temp_metrics = self.compute_metrics(temp_rewards)
                    confidence_drop_rate = temp_metrics["confidence_drop_rate"]

                    def store_in_memory():
                        store_pattern(
                            data.iloc[[i]],
                            action=trade.get("action", 0.0),
                            reward=reward,
                            neural_regime=float(data.iloc[i].get("neural_regime", 0)),
                            confidence=1.0 - confidence_drop_rate,
                            metadata={
                                "event": "trade_simulation",
                                "step": i,
                                "asset": asset,
                                "mode": mode,
                                "trade": trade,
                                "confidence_drop_rate": confidence_drop_rate,
                            },
                        )

                    self.with_retries(store_in_memory)

                    self.log_performance(
                        "simulate_step",
                        (datetime.now() - step_start).total_seconds(),
                        success=True,
                        step=i,
                        confidence_drop_rate=confidence_drop_rate,
                    )
                except Exception as e:
                    error_msg = f"Erreur étape {i}: {str(e)}\n{traceback.format_exc()}"
                    logger.error(error_msg)
                    if self.config.get("notifications_enabled", True):
                        self.alert_manager.send_alert(error_msg, priority=3)
                        send_telegram_alert(error_msg)
                    self.log_performance(
                        "simulate_step",
                        (datetime.now() - step_start).total_seconds(),
                        success=False,
                        error=str(e),
                        step=i,
                    )
                    continue

            trades_df = pd.DataFrame(trades)
            if not trades_df.empty:
                metrics = self.compute_metrics(trades_df["reward"].tolist())
                confidence_drop_rate = metrics["confidence_drop_rate"]
                trades_df["confidence_drop_rate"] = confidence_drop_rate  # Phase 8

                def save_trades():
                    trades_df.to_csv(output_csv_path, index=False, encoding="utf-8")

                self.with_retries(save_trades)
                self.generate_visualizations(trades_df, mode, timestamp)
                self.save_simulation_results(trades, output_csv_path)
                self.save_simulation_snapshot(len(trades), metrics, asset)
                self.save_dashboard_status(metrics, DASHBOARD_PATH)

            self.log_performance(
                "simulate_trades",
                time.time() - start_time,
                success=True,
                num_trades=len(trades),
                confidence_drop_rate=confidence_drop_rate,
            )
            logger.info(f"Simulation terminée. CPU: {psutil.cpu_percent()}%")
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(
                    f"Simulation terminée pour {asset.upper()}", priority=2
                )
                send_telegram_alert(f"Simulation terminée pour {asset.upper()}")
        except Exception as e:
            error_msg = f"Erreur simulate_trades: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
            self.log_performance(
                "simulate_trades", time.time() - start_time, success=False, error=str(e)
            )

    def simulate_trading(
        self,
        feature_csv_path: Union[str, Path] = str(
            BASE_DIR / "data" / "features" / "features_latest.csv"
        ),
        output_csv_path: Union[str, Path] = str(
            BASE_DIR / "data" / "trades" / "trades_simulated.csv"
        ),
        config_path: Union[str, Path] = str(BASE_DIR / "config" / "market_config.yaml"),
        asset: str = "ES",
        mode: str = "range",
    ) -> None:
        """
        Simule le trading offline avec gestion du risque, métriques et intégration de neural_pipeline.

        Args:
            feature_csv_path (Union[str, Path]): Chemin vers le fichier CSV des features.
            output_csv_path (Union[str, Path]): Chemin pour sauvegarder les trades simulés.
            config_path (Union[str, Path]): Chemin vers la configuration de l’environnement.
            asset (str): Symbole du marché (ex. ES, NQ).
            mode (str): Mode de trading ("trend", "range", "defensive").
        """
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            logger.info(
                f"Simulation trading offline lancée pour {asset.upper()} (Mode: {mode})"
            )
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(
                    f"Simulation trading offline lancée pour {asset.upper()} (Mode: {mode})",
                    priority=2,
                )
                send_telegram_alert(
                    f"Simulation trading offline lancée pour {asset.upper()} (Mode: {mode})"
                )

            # Gérer les sauvegardes incrémentielles et distribuées
            if (
                datetime.now() - self.last_checkpoint_time
            ).total_seconds() >= 300:  # 5 minutes
                self.save_checkpoint(incremental=True)
                self.last_checkpoint_time = datetime.now()
            if (
                datetime.now() - self.last_distributed_checkpoint_time
            ).total_seconds() >= 900:  # 15 minutes
                self.save_checkpoint(incremental=False, distributed=True)
                self.last_distributed_checkpoint_time = datetime.now()

            if not os.path.exists(feature_csv_path):
                error_msg = f"Fichier introuvable: {feature_csv_path}"
                raise FileNotFoundError(error_msg)

            def read_csv():
                return pd.read_csv(feature_csv_path)

            df = self.with_retries(read_csv)
            self.validate_data(df)

            router = MainRouter(
                config_path=config_path, feature_data_path=feature_csv_path
            )
            env = TradingEnv(config_path=config_path)
            env.data = df
            env.mode = mode

            neural_pipeline = None
            if self.use_neural_pipeline:
                neural_pipeline = NeuralPipeline(
                    window_size=50,
                    base_features=150,
                    config_path=str(BASE_DIR / "config" / "model_params.yaml"),
                )
                try:
                    def load_models():
                        neural_pipeline.load_models()

                    self.with_retries(load_models)
                    logger.info("Modèles neural_pipeline chargés")
                    if self.config.get("notifications_enabled", True):
                        self.alert_manager.send_alert(
                            "Modèles neural_pipeline chargés", priority=2
                        )
                        send_telegram_alert("Modèles neural_pipeline chargés")
                except Exception as e:
                    error_msg = f"Erreur chargement neural_pipeline: {e}, simulation sans neural_regime"
                    logger.warning(error_msg)
                    if self.config.get("notifications_enabled", True):
                        self.alert_manager.send_alert(error_msg, priority=3)
                        send_telegram_alert(error_msg)
                    neural_pipeline = None

            log_rows = []
            sequence_length = 50
            rewards_history = []
            confidence_drop_rate = 0.0  # Phase 8

            cache_key = hashlib.sha256(
                pd.concat([df, pd.Series({"mode": mode, "asset": asset})])
                .to_json()
                .encode()
            ).hexdigest()
            if cache_key in self.simulation_cache:
                self.log_performance("simulate_trading_cache_hit", 0, success=True)
                log_rows = self.simulation_cache[cache_key]["log_rows"]
                trades_df = pd.DataFrame(log_rows)
                metrics = self.compute_metrics(
                    trades_df["reward"].tolist() if log_rows else []
                )
                confidence_drop_rate = metrics["confidence_drop_rate"]  # Phase 8
                trades_df["confidence_drop_rate"] = confidence_drop_rate
                self.generate_visualizations(trades_df, mode, timestamp)
                self.save_simulation_results(log_rows, output_csv_path)
                self.save_simulation_snapshot(len(log_rows), metrics, asset)
                self.save_dashboard_status(metrics, DASHBOARD_PATH)
                return

            for step in range(sequence_length, len(df) - 1):
                step_start = datetime.now()
                try:
                    current_time = pd.to_datetime(df.iloc[step]["timestamp"])
                    if not self.check_trade_frequency(current_time):
                        continue

                    action, regime, details = router.route(current_step=step)
                    env.current_step = step
                    env._get_observation()

                    if neural_pipeline:
                        cache_key_neural = hashlib.sha256(
                            df.iloc[step : step + 1].to_json().encode()
                        ).hexdigest()
                        if cache_key_neural in self.neural_cache:
                            neural_result = self.neural_cache[cache_key_neural]
                        else:
                            required_cols = [
                                "timestamp",
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                                "atr_14",
                                "adx_14",
                                "gex",
                                "oi_peak_call_near",
                                "gamma_wall_call",
                                "bid_size_level_1",
                                "ask_size_level_1",
                            ]
                            missing_cols = [
                                col for col in required_cols if col not in df.columns
                            ]
                            if missing_cols:
                                raise ValueError(
                                    f"Colonnes manquantes pour neural_pipeline: {missing_cols}"
                                )

                            def run_pipeline():
                                raw_data = (
                                    df[
                                        [
                                            "timestamp",
                                            "open",
                                            "high",
                                            "low",
                                            "close",
                                            "volume",
                                            "atr_14",
                                            "adx_14",
                                        ]
                                    ]
                                    .iloc[step : step + 1]
                                    .fillna(0)
                                )
                                options_data = (
                                    df[
                                        [
                                            "timestamp",
                                            "gex",
                                            "oi_peak_call_near",
                                            "gamma_wall_call",
                                        ]
                                    ]
                                    .iloc[step : step + 1]
                                    .fillna(0)
                                )
                                orderflow_data = (
                                    df[
                                        [
                                            "timestamp",
                                            "bid_size_level_1",
                                            "ask_size_level_1",
                                        ]
                                    ]
                                    .iloc[step : step + 1]
                                    .fillna(0)
                                )
                                return neural_pipeline.run(
                                    raw_data, options_data, orderflow_data
                                )

                            neural_result = self.with_retries(run_pipeline)
                            if neural_result is None:
                                raise ValueError("Échec NeuralPipeline")
                            self.neural_cache[cache_key_neural] = neural_result
                            if len(self.neural_cache) > self.max_cache_size:
                                self.neural_cache.pop(next(iter(self.neural_cache)))
                        df.loc[step, "predicted_volatility"] = neural_result[
                            "volatility"
                        ][0]
                        df.loc[step, "neural_regime"] = neural_result["regime"][0]
                        num_features = neural_result["features"].shape[1] - 1  # Exclure cnn_pressure
                        for i in range(num_features):
                            df.loc[step, f"neural_feature_{i}"] = neural_result[
                                "features"
                            ][0, i]
                        df.loc[step, "cnn_pressure"] = neural_result["features"][
                            0, num_features
                        ]

                    drawdown = (
                        np.min(
                            np.cumsum(rewards_history)
                            - np.maximum.accumulate(np.cumsum(rewards_history))
                        )
                        / 10000
                        if rewards_history
                        else 0
                    )
                    atr = df["atr_14"].iloc[step]
                    risk_factor = 1.0
                    if drawdown > self.performance_thresholds["max_drawdown"]:
                        risk_factor = 0.5
                        error_msg = f"Step {step}: Drawdown excessif ({drawdown:.2%}), réduction risque à {risk_factor}"
                        logger.info(error_msg)
                        if self.config.get("notifications_enabled", True):
                            self.alert_manager.send_alert(error_msg, priority=3)
                            send_telegram_alert(error_msg)
                    if atr > 2.0:
                        risk_factor = min(risk_factor, 0.5)
                        error_msg = f"Step {step}: Volatilité extrême (ATR={atr:.2f}), réduction risque à {risk_factor}"
                        logger.info(error_msg)
                        if self.config.get("notifications_enabled", True):
                            self.alert_manager.send_alert(error_msg, priority=3)
                            send_telegram_alert(error_msg)

                    action *= risk_factor
                    _, reward, done, _, info = env.step(np.array([action]))
                    rewards_history.append(reward)

                    price = df["close"].iloc[step]
                    timestamp_step = df["timestamp"].iloc[step]
                    neural_regime = (
                        df["neural_regime"].iloc[step]
                        if "neural_regime" in df.columns
                        else "N/A"
                    )

                    # Calculer confidence_drop_rate dynamiquement
                    temp_rewards = rewards_history[-100:]  # Limiter aux 100 derniers
                    temp_metrics = self.compute_metrics(temp_rewards)
                    confidence_drop_rate = temp_metrics["confidence_drop_rate"]

                    row = {
                        "timestamp": str(timestamp_step),
                        "step": step,
                        "regime": regime,
                        "neural_regime": (
                            int(neural_regime) if neural_regime != "N/A" else "N/A"
                        ),
                        "action": action,
                        "reward": reward,
                        "price": price,
                        "volume": df["volume"].iloc[step],
                        "risk_factor": risk_factor,
                        "drawdown": drawdown,
                        "balance": info["balance"],
                        "max_drawdown": info["max_drawdown"],
                        "call_wall": info["call_wall"],
                        "put_wall": info["put_wall"],
                        "zero_gamma": info["zero_gamma"],
                        "dealer_position_bias": info["dealer_position_bias"],
                        "news_impact_score": float(df["news_impact_score"].iloc[step]),
                        "predicted_vix": float(df["predicted_vix"].iloc[step]),
                        "confidence_drop_rate": confidence_drop_rate,  # Phase 8
                    }

                    if row["balance"] < -10000:
                        error_msg = f"Step {step}: Balance incohérente détectée ({row['balance']:.2f})"
                        logger.warning(error_msg)
                        if self.config.get("notifications_enabled", True):
                            self.alert_manager.send_alert(error_msg, priority=3)
                            send_telegram_alert(error_msg)

                    log_rows.append(row)
                    self.trade_buffer.append(row)

                    def store_in_memory():
                        store_pattern(
                            df.iloc[[step]],
                            action=action,
                            reward=reward,
                            neural_regime=(
                                float(neural_regime) if neural_regime != "N/A" else 0
                            ),
                            confidence=1.0 - confidence_drop_rate,
                            metadata={
                                "event": "trade_simulation",
                                "step": step,
                                "regime": regime,
                                "risk_factor": risk_factor,
                                "asset": asset,
                                "confidence_drop_rate": confidence_drop_rate,
                            },
                        )

                    self.with_retries(store_in_memory)

                    self.log_performance(
                        "simulate_step",
                        (datetime.now() - step_start).total_seconds(),
                        success=True,
                        step=step,
                        confidence_drop_rate=confidence_drop_rate,
                    )
                    logger.info(
                        f"Step {step}: {regime.upper()}, neural_regime={row['neural_regime']}, "
                        f"action={action:.2f}, reward={reward:.2f}, risk_factor={risk_factor:.2f}, "
                        f"confidence_drop_rate={confidence_drop_rate:.2f}"
                    )

                    if done:
                        logger.info(f"Simulation terminée à l'étape {step} (done)")
                        break

                except Exception as e:
                    error_msg = (
                        f"Erreur étape {step}: {str(e)}\n{traceback.format_exc()}"
                    )
                    logger.error(error_msg)
                    if self.config.get("notifications_enabled", True):
                        self.alert_manager.send_alert(error_msg, priority=5)
                        send_telegram_alert(error_msg)
                    self.log_performance(
                        "simulate_step",
                        (datetime.now() - step_start).total_seconds(),
                        success=False,
                        error=str(e),
                        step=step,
                    )
                    continue

            trades_df = pd.DataFrame(log_rows)
            if not trades_df.empty:
                self.save_simulation_results(log_rows, output_csv_path)
                metrics = self.compute_metrics(trades_df["reward"].tolist())
                confidence_drop_rate = metrics["confidence_drop_rate"]  # Phase 8
                trades_df["confidence_drop_rate"] = confidence_drop_rate
                self.generate_visualizations(trades_df, mode, timestamp)
                self.save_simulation_snapshot(len(log_rows), metrics, asset)
                self.save_dashboard_status(metrics, DASHBOARD_PATH)
                self.simulation_cache[cache_key] = {
                    "log_rows": log_rows,
                    "metrics": metrics,
                }
                if len(self.simulation_cache) > self.max_cache_size:
                    self.simulation_cache.pop(next(iter(self.simulation_cache)))

                metrics_path = str(output_csv_path).replace(".csv", "_metrics.json")

                def save_metrics():
                    with open(metrics_path, "w", encoding="utf-8") as f:
                        json.dump(metrics, f, indent=4)

                self.with_retries(save_metrics)

            self.log_performance(
                "simulate_trading",
                time.time() - start_time,
                success=True,
                num_trades=len(log_rows),
                confidence_drop_rate=confidence_drop_rate,
            )
            logger.info(
                f"Simulation terminée. Résultats sous {output_csv_path}, Statistiques: {metrics}"
            )
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(
                    f"Simulation terminée. Résultats sous {output_csv_path}", priority=2
                )
                send_telegram_alert(
                    f"Simulation terminée. Résultats sous {output_csv_path}"
                )

        except Exception as e:
            error_msg = f"Erreur simulate_trading: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
            self.log_performance(
                "simulate_trading",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            raise

    def save_simulation_results(
        self, trades: List[Dict], output_path: Union[str, Path]
    ) -> None:
        """
        Sauvegarde les trades simulés dans un CSV avec buffering.

        Args:
            trades (List[Dict]): Liste des trades simulés.
            output_path (Union[str, Path]): Chemin pour sauvegarder le CSV.
        """
        start_time = time.time()
        try:
            logger.info(f"Sauvegarde des trades simulés sous {output_path}")
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(
                    f"Sauvegarde des trades simulés sous {output_path}", priority=2
                )
                send_telegram_alert(f"Sauvegarde des trades simulés sous {output_path}")

            trades_df = pd.DataFrame(trades)
            if not trades_df.empty:
                self.trade_buffer.extend(trades)
                if len(self.trade_buffer) >= self.buffer_size:
                    buffer_df = pd.DataFrame(self.trade_buffer)

                    def save_csv():
                        if not os.path.exists(output_path):
                            buffer_df.to_csv(output_path, index=False, encoding="utf-8")
                        else:
                            buffer_df.to_csv(
                                output_path,
                                mode="a",
                                header=False,
                                index=False,
                                encoding="utf-8",
                            )

                    self.with_retries(save_csv)
                    export_logs(
                        self.trade_buffer,
                        os.path.dirname(output_path),
                        os.path.basename(output_path),
                    )
                    self.trade_buffer = []

            self.log_performance(
                "save_simulation_results",
                time.time() - start_time,
                success=True,
                num_trades=len(trades),
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde trades: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
            self.log_performance(
                "save_simulation_results",
                time.time() - start_time,
                success=False,
                error=str(e),
            )

    def save_simulation_snapshot(
        self, step: int, metrics: Dict[str, float], asset: str
    ) -> None:
        """
        Sauvegarde un instantané des métriques de simulation.

        Args:
            step (int): Étape de la simulation.
            metrics (Dict[str, float]): Métriques calculées.
            asset (str): Symbole du marché.
        """
        start_time = time.time()
        try:
            snapshot = {
                "step": step,
                "timestamp": str(datetime.now()),
                "asset": asset,
                "metrics": metrics,
                "trade_buffer_size": len(self.trade_buffer),
            }
            self.save_snapshot(f"simulation_step_{step:04d}", snapshot, compress=False)
            logger.info(f"Snapshot simulation step {step} sauvegardé")
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(
                    f"Snapshot simulation step {step} sauvegardé", priority=1
                )
                send_telegram_alert(f"Snapshot simulation step {step} sauvegardé")
            self.log_performance(
                "save_simulation_snapshot",
                time.time() - start_time,
                success=True,
                step=step,
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot simulation: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
            self.log_performance(
                "save_simulation_snapshot",
                time.time() - start_time,
                success=False,
                error=str(e),
            )


if __name__ == "__main__":
    try:
        simulator = TradeSimulator()
        simulator.simulate_trading()
    except Exception as e:
        error_msg = f"Erreur exécution: {str(e)}\n{traceback.format_exc()}"
        alert_manager = AlertManager()
        logger.error(error_msg)
        alert_manager.send_alert(error_msg, priority=5)
        send_telegram_alert(error_msg)
        raise
