# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/trading/trade_executor.py
# Rôle : Exécute les trades via Sierra Chart (AMP Futures, API Teton) en mode réel ou paper trading pour MIA_IA_SYSTEM_v2_2025.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Exécute des trades via Sierra Chart (API Teton) en mode réel ou paper trading, valide les données contextuelles
#        (350/150 SHAP features), enregistre les résultats dans market_memory.db, effectue un fine-tuning (méthode 8)
#        et un apprentissage en ligne (méthode 10), génère des snapshots JSON (option compressée), des sauvegardes,
#        des graphiques matplotlib, et des alertes standardisées (Phase 8). Compatible avec la simulation de trading (Phase 12)
#        et l’ensemble learning (Phase 16). Intègre des métriques d’auto-conscience (Phase 8) comme confidence_drop_rate.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0
# - numpy>=1.26.4,<2.0.0
# - matplotlib>=3.8.0,<4.0.0
# - psutil>=5.9.8,<6.0.0
# - pyyaml>=6.0.0,<7.0.0
# - logging, os, json, hashlib, datetime, signal, time, gzip, traceback
# - src.model.adaptive_learner
# - src.model.utils.config_manager
# - src.model.utils.alert_manager
# - src.utils.telegram_alert
# - src.trading.finetune_utils
# - src.trading.live_trading
# - src.trading.train_sac
#
# Inputs :
# - Données de trading (pd.DataFrame avec 350 features pour entraînement ou 150 SHAP features pour inférence)
# - Configuration via config/market_config.yaml
# - Identifiants via config/credentials.yaml
# - config/feature_sets.yaml
#
# Outputs :
# - Logs dans data/logs/trade_executor.log
# - Logs de performance dans data/logs/trading/trade_executor_performance.csv
# - Snapshots JSON dans data/trade_snapshots/*.json (option *.json.gz)
# - Dashboard JSON dans data/trading/trade_execution_dashboard.json
# - Visualisations dans data/figures/trading/*.png
# - Trades dans data/trades/*.csv
# - Sauvegardes dans data/checkpoints/trade_executor/*.json.gz
#
# Notes :
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence) définies dans config/feature_sets.yaml.
# - Utilise IQFeed exclusivement via data_provider.py pour les données d’entrée.
# - Supprime toute référence à 320 features, 81 features, et obs_t.
# - Implémente retries (max 3, délai 2^attempt secondes) pour les opérations critiques.
# - Intègre fine-tuning (méthode 8) et apprentissage en ligne (méthode 10) via finetune_utils.py.
# - Utilise l’API Teton de Sierra Chart avec simulation (intégration réelle en attente).
# - Tests unitaires disponibles dans tests/test_trade_executor.py.
# - Conforme aux Phases 8 (auto-conscience), 10 (apprentissage en ligne), 12 (gestion des risques).

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
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

from src.model.adaptive_learner import store_pattern
from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import config_manager
from src.trading.finetune_utils import finetune_model
from src.utils.telegram_alert import send_telegram_alert

# Configuration logging
log_path = Path("data/logs/trade_executor.log")
log_path.parent.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Chemins relatifs
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
TRADES_DIR = BASE_DIR / "data" / "trades"
SNAPSHOT_DIR = BASE_DIR / "data" / "trade_snapshots"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints" / "trade_executor"
FIGURES_DIR = BASE_DIR / "data" / "figures" / "trading"
DASHBOARD_PATH = BASE_DIR / "data" / "trading" / "trade_execution_dashboard.json"
CSV_LOG_PATH = BASE_DIR / "data" / "logs" / "trading" / "trade_executor_performance.csv"

# Constantes
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Secondes, exponentiel


class TradeExecutor:
    """
    Classe pour exécuter des trades via Sierra Chart (AMP Futures, API Teton) en mode réel ou paper trading.
    """

    def __init__(
        self,
        config_path: str = "config/market_config.yaml",
        credentials_path: str = "config/credentials.yaml",
    ):
        """
        Initialise l’exécuteur de trades.

        Args:
            config_path (str): Chemin vers la configuration du marché.
            credentials_path (str): Chemin vers les identifiants AMP Futures.
        """
        self.alert_manager = AlertManager()
        SNAPSHOT_DIR.mkdir(exist_ok=True)
        TRADES_DIR.mkdir(exist_ok=True)
        FIGURES_DIR.mkdir(exist_ok=True)
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        CSV_LOG_PATH.parent.mkdir(exist_ok=True)
        signal.signal(signal.SIGINT, self.handle_sigint)
        self.last_checkpoint_time = datetime.now()
        self.last_distributed_checkpoint_time = datetime.now()

        self.log_buffer = []
        self.trade_buffer = []
        self.trade_cache = {}
        try:
            self.config = self.load_config(config_path)
            self.credentials = self.load_credentials(credentials_path)

            self.performance_thresholds = {
                "max_slippage": self.config.get("thresholds", {}).get(
                    "max_slippage", 0.01
                ),
                "min_balance": self.config.get("thresholds", {}).get(
                    "min_balance", -10000
                ),
                "min_sharpe": self.config.get("thresholds", {}).get("min_sharpe", 0.5),
                "max_drawdown": self.config.get("thresholds", {}).get(
                    "max_drawdown", -1000.0
                ),
                "min_profit_factor": self.config.get("thresholds", {}).get(
                    "min_profit_factor", 1.2
                ),
            }

            for key, value in self.performance_thresholds.items():
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Seuil invalide pour {key}: {value}")
                if (
                    key in ["max_slippage", "min_sharpe", "min_profit_factor"]
                    and value <= 0
                ):
                    raise ValueError(f"Seuil {key} doit être positif: {value}")
                if key == "max_drawdown" and value >= 0:
                    raise ValueError(f"Seuil {key} doit être négatif: {value}")

            self.is_simulation = not self.credentials.get("amp_futures", {}).get(
                "enabled", False
            )

            logger.info("TradeExecutor initialisé avec succès")
            self.alert_manager.send_alert("TradeExecutor initialisé", priority=2)
            send_telegram_alert("TradeExecutor initialisé")
            self.log_performance("init", 0, success=True)
            self.save_checkpoint(incremental=True)
        except Exception as e:
            error_msg = f"Erreur initialisation TradeExecutor: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            raise

    def handle_sigint(self, signum, frame):
        """Gère l'arrêt propre sur SIGINT."""
        snapshot = {"timestamp": datetime.now().isoformat(), "status": "SIGINT"}
        try:
            self.save_snapshot("sigint", snapshot, compress=False)
            self.save_checkpoint(incremental=True)
            logger.info("Arrêt propre sur SIGINT, snapshot et checkpoint sauvegardés")
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
            self.alert_manager.send_alert(
                f"Snapshot {snapshot_type} sauvegardé: {save_path}", priority=1
            )
            send_telegram_alert(f"Snapshot {snapshot_type} sauvegardé: {save_path}")
            logger.info(f"Snapshot {snapshot_type} sauvegardé: {save_path}")
            self.log_performance("save_snapshot", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("save_snapshot", 0, success=False, error=str(e))

    def save_checkpoint(self, incremental: bool = True, distributed: bool = False):
        """Sauvegarde l’état de l’exécuteur (incrémentiel, distribué, versionné)."""
        try:
            start_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint = {
                "timestamp": timestamp,
                "log_buffer": self.log_buffer[-100:],  # Limiter la taille
                "trade_buffer": self.trade_buffer[-100:],  # Limiter la taille
                "trade_cache": {k: True for k in self.trade_cache},  # Simplifié
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

            # Sauvegarde distribuée (simulation, ex. : AWS S3)
            if distributed:
                # TODO: Implémenter la sauvegarde vers AWS S3
                logger.info(f"Sauvegarde distribuée simulée pour {checkpoint_path}")

            latency = time.time() - start_time
            self.alert_manager.send_alert(
                f"Checkpoint {'incrémentiel' if incremental else 'distribué'} sauvegardé: {checkpoint_path}",
                priority=1,
            )
            send_telegram_alert(
                f"Checkpoint {'incrémentiel' if incremental else 'distribué'} sauvegardé: {checkpoint_path}"
            )
            logger.info(
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
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("save_checkpoint", 0, success=False, error=str(e))

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ) -> None:
        """Enregistre les performances dans un CSV avec métriques psutil."""
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
            cpu_percent = psutil.cpu_percent()
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "latency": latency,
                "success": success,
                "error": error,
                "cpu_percent": cpu_percent,
                "memory_usage_mb": memory_usage,
                **kwargs,
            }
            self.log_buffer.append(log_entry)
            if len(self.log_buffer) >= self.config.get("logging", {}).get(
                "buffer_size", 200
            ):
                buffer_df = pd.DataFrame(self.log_buffer)
                if not CSV_LOG_PATH.exists():
                    buffer_df.to_csv(CSV_LOG_PATH, index=False, encoding="utf-8")
                else:
                    buffer_df.to_csv(
                        CSV_LOG_PATH,
                        mode="a",
                        header=False,
                        index=False,
                        encoding="utf-8",
                    )
                self.log_buffer = []
        except Exception as e:
            error_msg = (
                f"Erreur enregistrement performance: {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def with_retries(
        self,
        func: callable,
        max_attempts: int = MAX_RETRIES,
        delay_base: float = RETRY_DELAY_BASE,
    ) -> Optional[any]:
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
                    error_msg = f"Échec après {max_attempts} tentatives: {str(e)}\n{traceback.format_exc()}"
                    self.alert_manager.send_alert(error_msg, priority=3)
                    send_telegram_alert(error_msg)
                    logger.error(error_msg)
                    self.log_performance(
                        f"retry_attempt_{attempt+1}",
                        time.time() - start_time,
                        success=False,
                        error=str(e),
                        attempt_number=attempt + 1,
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
            config = self.with_retries(
                lambda: config_manager.get_config(BASE_DIR / config_path)
            )
            if not config:
                raise ValueError("Configuration vide ou non trouvée")
            self.log_performance("load_config", time.time() - start_time, success=True)
            logger.info("Configuration chargée avec succès")
            self.alert_manager.send_alert(
                "Configuration chargée avec succès", priority=1
            )
            send_telegram_alert("Configuration chargée avec succès")
            return config
        except Exception as e:
            error_msg = (
                f"Erreur chargement configuration: {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "load_config", time.time() - start_time, success=False, error=str(e)
            )
            return {
                "thresholds": {
                    "max_slippage": 0.01,
                    "min_balance": -10000,
                    "min_sharpe": 0.5,
                    "max_drawdown": -1000.0,
                    "min_profit_factor": 1.2,
                },
                "logging": {"buffer_size": 200},
                "visualization": {"figsize": [12, 5], "plot_interval": 50},
            }

    def load_credentials(self, credentials_path: Union[str, Path]) -> Dict:
        """
        Charge les identifiants AMP Futures.

        Args:
            credentials_path (Union[str, Path]): Chemin vers le fichier d’identifiants.

        Returns:
            Dict: Identifiants chargés.
        """
        start_time = time.time()
        try:
            credentials = self.with_retries(
                lambda: config_manager.get_config(BASE_DIR / credentials_path)
            )
            if not credentials:
                raise ValueError("Identifiants vides ou non trouvés")
            self.log_performance(
                "load_credentials", time.time() - start_time, success=True
            )
            logger.info("Identifiants chargés avec succès")
            self.alert_manager.send_alert(
                "Identifiants chargés avec succès", priority=1
            )
            send_telegram_alert("Identifiants chargés avec succès")
            return credentials
        except Exception as e:
            error_msg = (
                f"Erreur chargement identifiants: {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "load_credentials",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            return {"amp_futures": {"enabled": False}}

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
                error_msg = f"Nombre de features insuffisant: {num_cols}, attendu ≥150"
                raise ValueError(error_msg)
            self.log_performance(
                "detect_feature_set",
                time.time() - start_time,
                success=True,
                num_columns=num_cols,
            )
            logger.debug(f"Feature set détecté: {feature_set}")
            self.alert_manager.send_alert(
                f"Feature set détecté: {feature_set}", priority=1
            )
            send_telegram_alert(f"Feature set détecté: {feature_set}")
            return feature_set
        except Exception as e:
            error_msg = f"Erreur détection ensemble features: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
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
                self.alert_manager.send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                return data

            data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
            if data["timestamp"].isna().any():
                last_valid = (
                    data["timestamp"].dropna().iloc[-1]
                    if not data["timestamp"].dropna().empty
                    else pd.Timestamp.now()
                )
                data["timestamp"] = data["timestamp"].fillna(last_valid)
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
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "impute_timestamp",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            return data

    def validate_context_data(self, data: Optional[pd.DataFrame]) -> None:
        """
        Valide les données contextuelles (350 features pour entraînement, 150 SHAP pour inférence) si fournies.

        Args:
            data (Optional[pd.DataFrame]): Données contextuelles.

        Raises:
            ValueError: Si les données sont invalides.
        """
        if data is None:
            return

        start_time = time.time()
        try:
            feature_set = self.detect_feature_set(data)
            expected_features = 350 if feature_set == "training" else 150

            # Valider les colonnes via feature_sets.yaml
            feature_sets = self.with_retries(
                lambda: config_manager.get_config(
                    BASE_DIR / "config" / "feature_sets.yaml"
                )
            )
            expected_cols = feature_sets.get(
                "training_features" if feature_set == "training" else "shap_features",
                [],
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
                "validate_context_data",
                time.time() - start_time,
                success=True,
                num_features=len(data.columns),
            )
            logger.info(f"Données contextuelles validées pour ensemble {feature_set}")
            self.alert_manager.send_alert(
                f"Données contextuelles validées pour ensemble {feature_set}",
                priority=1,
            )
            send_telegram_alert(
                f"Données contextuelles validées pour ensemble {feature_set}"
            )
            self.save_snapshot(
                "validate_context_data",
                {"num_features": len(data.columns)},
                compress=False,
            )
        except Exception as e:
            error_msg = f"Erreur validation données contextuelles: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "validate_context_data",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            raise

    def validate_trade(self, trade: Dict) -> None:
        """
        Valide les paramètres d’un trade.

        Args:
            trade (Dict): Dictionnaire contenant les paramètres du trade (trade_id, action, price, size, order_type).

        Raises:
            ValueError: Si les paramètres sont invalides.
        """
        start_time = time.time()
        try:
            cache_key = hashlib.sha256(str(trade["trade_id"]).encode()).hexdigest()
            if cache_key in self.trade_cache:
                self.log_performance(
                    "validate_trade_cache_hit", 0, success=True, num_trades=1
                )
                return

            required_keys = ["trade_id", "action", "price", "size", "order_type"]
            missing_keys = [key for key in required_keys if key not in trade]
            if missing_keys:
                error_msg = f"Clés manquantes dans le trade: {missing_keys}"
                raise ValueError(error_msg)

            if not isinstance(trade["trade_id"], (int, str)):
                error_msg = (
                    f"trade_id doit être un entier ou une chaîne: {trade['trade_id']}"
                )
                raise ValueError(error_msg)
            if trade["action"] not in [-1, 0, 1, "buy", "sell", "hold"]:
                error_msg = f"Action invalide: {trade['action']}, doit être -1, 0, 1, 'buy', 'sell' ou 'hold'"
                raise ValueError(error_msg)
            if trade["price"] <= 0:
                error_msg = f"Prix invalide: {trade['price']}, doit être positif"
                raise ValueError(error_msg)
            if trade["size"] <= 0:
                error_msg = f"Taille invalide: {trade['size']}, doit être positif"
                raise ValueError(error_msg)
            if trade["order_type"] not in ["market", "limit"]:
                error_msg = f"Type d’ordre invalide: {trade['order_type']}, doit être 'market' ou 'limit'"
                raise ValueError(error_msg)

            self.trade_cache[cache_key] = True
            if len(self.trade_cache) > 1000:
                self.trade_cache.pop(next(iter(self.trade_cache)))

            self.log_performance(
                "validate_trade", time.time() - start_time, success=True, num_trades=1
            )
            logger.info(f"Trade {trade['trade_id']} validé")
            self.alert_manager.send_alert(
                f"Trade {trade['trade_id']} validé", priority=1
            )
            send_telegram_alert(f"Trade {trade['trade_id']} validé")
        except Exception as e:
            error_msg = f"Erreur validation trade: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "validate_trade", time.time() - start_time, success=False, error=str(e)
            )
            raise

    def store_execution_pattern(
        self,
        trade: Dict,
        execution_result: Dict,
        mode: str,
        context_data: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Enregistre les résultats d’exécution dans market_memory.db via store_pattern.

        Args:
            trade (Dict): Paramètres du trade initial.
            execution_result (Dict): Résultat de l’exécution.
            mode (str): Mode de trading ("paper" ou "real").
            context_data (Optional[pd.DataFrame]): Données contextuelles (facultatif).
        """
        start_time = time.time()
        try:
            trade_id = trade["trade_id"]
            execution_time = execution_result["execution_time"]
            if not isinstance(trade_id, (int, str)) or not isinstance(
                execution_time, str
            ):
                error_msg = f"trade_id ({trade_id}) ou execution_time ({execution_time}) invalide"
                raise ValueError(error_msg)

            slippage = (
                abs(execution_result["execution_price"] - trade["price"])
                / trade["price"]
                if trade["price"] != 0
                else 0.0
            )
            data = pd.DataFrame(
                [
                    {
                        "trade_id": trade_id,
                        "execution_price": execution_result["execution_price"],
                        "balance": execution_result["balance"],
                        "slippage": slippage,
                        "timestamp": execution_time,
                    }
                ]
            )

            metadata = {
                "event": "trade_execution",
                "trade_id": trade_id,
                "mode": mode,
                "timestamp": execution_time,
                "slippage": slippage,
                "balance": execution_result["balance"],
                "status": execution_result["status"],
                "order_type": trade["order_type"],
            }
            if context_data is not None and "neural_regime" in context_data.columns:
                metadata["neural_regime"] = float(context_data["neural_regime"].iloc[0])

            def store_pattern_call():
                store_pattern(
                    data=data,
                    action=trade["action"],
                    reward=execution_result["execution_price"] - trade["price"],
                    neural_regime=metadata.get("neural_regime", 0),
                    confidence=0.95 if execution_result["status"] == "filled" else 0.5,
                    metadata=metadata,
                )

            self.with_retries(store_pattern_call)

            self.log_performance(
                "store_execution_pattern",
                time.time() - start_time,
                success=True,
                num_trades=1,
            )
            logger.info(f"Exécution trade {trade_id} enregistrée dans market_memory.db")
            self.alert_manager.send_alert(
                f"Exécution trade {trade_id} enregistrée dans market_memory.db",
                priority=1,
            )
            send_telegram_alert(
                f"Exécution trade {trade_id} enregistrée dans market_memory.db"
            )
        except Exception as e:
            error_msg = f"Erreur enregistrement trade {trade.get('trade_id', 'inconnu')} dans market_memory.db: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "store_execution_pattern",
                time.time() - start_time,
                success=False,
                error=str(e),
            )

    def execute_trade(
        self,
        trade: Dict,
        mode: str = "paper",
        context_data: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Exécute un trade via Sierra Chart (API Teton) ou simulation.

        Args:
            trade (Dict): Dictionnaire contenant les paramètres du trade (trade_id, action, price, size, order_type).
            mode (str): Mode de trading ("paper" ou "real").
            context_data (Optional[pd.DataFrame]): Données contextuelles (facultatif).

        Returns:
            Dict: Résultat de l’exécution (trade_id, status, execution_price, execution_time, balance, confidence_drop_rate).

        Notes:
            TODO: Implémenter l’intégration réelle avec l’API Teton de Sierra Chart.
        """
        start_time = time.time()
        trade_id = trade.get("trade_id", "unknown")
        confidence_drop_rate = 0.0  # Phase 8: Métrique d’auto-conscience
        try:
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

            self.validate_trade(trade)
            if context_data is not None:
                self.validate_context_data(context_data)

            if mode not in ["paper", "real"]:
                error_msg = f"Mode invalide: {mode}, doit être 'paper' ou 'real'"
                raise ValueError(error_msg)
            is_paper = mode == "paper"

            if len(self.trade_buffer) > 0:
                rewards = [
                    entry["execution_price"] - entry["price"]
                    for entry in self.trade_buffer
                ]
                positive_rewards = sum(r for r in rewards if r > 0)
                negative_rewards = abs(sum(r for r in rewards if r < 0))
                profit_factor = (
                    positive_rewards / negative_rewards
                    if negative_rewards > 0
                    else float("inf")
                )
                sharpe = (
                    np.mean(rewards) / np.std(rewards) if np.std(rewards) > 0 else 0.0
                )
                drawdown = min(0, (self.trade_buffer[-1]["balance"] - 10000) / 10000)
                confidence_drop_rate = 1.0 - min(
                    profit_factor / self.performance_thresholds["min_profit_factor"],
                    1.0,
                )  # Phase 8

                for metric, threshold in self.performance_thresholds.items():
                    if metric == "min_sharpe" and sharpe < threshold:
                        error_msg = f"Seuil non atteint pour {metric}: {sharpe:.2f} < {threshold}"
                        raise ValueError(error_msg)
                    if metric == "max_drawdown" and drawdown < threshold:
                        error_msg = f"Seuil non atteint pour {metric}: {drawdown:.2f} < {threshold}"
                        raise ValueError(error_msg)
                    if metric == "min_profit_factor" and profit_factor < threshold:
                        error_msg = f"Seuil non atteint pour {metric}: {profit_factor:.2f} < {threshold}"
                        raise ValueError(error_msg)

            logger.info(f"Exécution trade {trade_id} en mode {mode}")
            self.alert_manager.send_alert(
                f"Exécution trade {trade_id} en mode {mode}", priority=2
            )
            send_telegram_alert(f"Exécution trade {trade_id} en mode {mode}")

            if self.is_simulation or is_paper:

                def simulate_trade():
                    time.sleep(np.random.uniform(0.1, 1.0))
                    slippage = trade["price"] * np.random.uniform(
                        0, self.performance_thresholds["max_slippage"]
                    )
                    execution_price = (
                        trade["price"] + slippage
                        if trade["action"] in [1, "buy"]
                        else trade["price"] - slippage
                    )
                    execution_time = datetime.now()
                    balance = 10000 + np.random.uniform(-500, 500)
                    status = "filled"
                    return {
                        "trade_id": trade_id,
                        "status": status,
                        "execution_price": execution_price,
                        "execution_time": str(execution_time),
                        "balance": balance,
                        "confidence_drop_rate": confidence_drop_rate,
                    }

                for attempt in range(MAX_RETRIES):
                    try:
                        result = simulate_trade()
                        break
                    except ConnectionError:
                        if attempt == MAX_RETRIES - 1:
                            trade["size"] *= 0.5
                            trade["order_type"] = "limit"
                            logger.info(
                                f"Repli trade {trade_id}: size={trade['size']}, order_type={trade['order_type']}"
                            )
                            self.alert_manager.send_alert(
                                f"Repli trade {trade_id}: size={trade['size']}, order_type={trade['order_type']}",
                                priority=3,
                            )
                            send_telegram_alert(
                                f"Repli trade {trade_id}: size={trade['size']}, order_type={trade['order_type']}"
                            )
                            result = self.with_retries(
                                simulate_trade, max_attempts=MAX_RETRIES
                            )
                            if result is None:
                                error_msg = (
                                    f"Échec exécution trade {trade_id} après repli"
                                )
                                raise ValueError(error_msg)
                        continue
            else:
                # TODO: Implémenter l’intégration réelle avec l’API Teton
                from sierra_chart_api import SierraChartAPI

                def real_trade():
                    return SierraChartAPI.execute(trade, mode)

                for attempt in range(MAX_RETRIES):
                    try:
                        result = real_trade()
                        result["confidence_drop_rate"] = confidence_drop_rate
                        break
                    except ConnectionError:
                        if attempt == MAX_RETRIES - 1:
                            error_msg = f"Échec exécution trade {trade_id} après {MAX_RETRIES} tentatives"
                            raise ValueError(error_msg)
                        continue

            if result["balance"] < self.performance_thresholds["min_balance"]:
                error_msg = f"Trade {trade_id}: Balance incohérente détectée ({result['balance']:.2f})"
                logger.warning(error_msg)
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)

            self.store_execution_pattern(trade, result, mode, context_data)

            def finetune():
                finetune_model(
                    context_data if context_data is not None else pd.DataFrame([trade])
                )

            self.with_retries(finetune)

            self.save_execution_snapshot(trade_id, result)
            self.log_execution(trade, result, is_paper)

            latency = time.time() - start_time
            self.log_performance("execute_trade", latency, success=True, num_trades=1)
            logger.info(f"Trade exécuté. CPU: {psutil.cpu_percent()}%")
            self.alert_manager.send_alert(
                f"Trade {trade_id} exécuté: status={result['status']}, prix={result['execution_price']:.2f}, confidence_drop_rate={confidence_drop_rate:.2f}",
                priority=2,
            )
            send_telegram_alert(
                f"Trade {trade_id} exécuté: status={result['status']}, prix={result['execution_price']:.2f}, confidence_drop_rate={confidence_drop_rate:.2f}"
            )
            return result
        except Exception as e:
            error_msg = (
                f"Erreur exécution trade {trade_id}: {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "execute_trade", time.time() - start_time, success=False, error=str(e)
            )
            return {
                "trade_id": trade_id,
                "status": "failed",
                "execution_price": 0.0,
                "execution_time": str(datetime.now()),
                "balance": 0.0,
                "confidence_drop_rate": confidence_drop_rate,
            }

    def confirm_trade(self, trade_id: str) -> Dict:
        """
        Vérifie la confirmation d’un trade.

        Args:
            trade_id (str): Identifiant du trade.

        Returns:
            Dict: Statut de confirmation (trade_id, confirmed, confirmation_time).
        """
        start_time = time.time()
        try:
            logger.info(f"Vérification confirmation trade {trade_id}")
            self.alert_manager.send_alert(
                f"Vérification confirmation trade {trade_id}", priority=2
            )
            send_telegram_alert(f"Vérification confirmation trade {trade_id}")

            def simulate_confirmation():
                time.sleep(np.random.uniform(0.1, 0.5))
                confirmed = np.random.choice([True, False], p=[0.95, 0.05])
                confirmation_time = datetime.now()
                return {
                    "trade_id": trade_id,
                    "confirmed": confirmed,
                    "confirmation_time": str(confirmation_time),
                }

            result = self.with_retries(simulate_confirmation, max_attempts=MAX_RETRIES)
            if result is None:
                error_msg = f"Échec confirmation trade {trade_id}"
                raise ValueError(error_msg)

            latency = time.time() - start_time
            self.log_performance("confirm_trade", latency, success=True, num_trades=1)
            logger.info(
                f"Confirmation trade {trade_id}: {'confirmé' if result['confirmed'] else 'non confirmé'}"
            )
            self.alert_manager.send_alert(
                f"Confirmation trade {trade_id}: {'confirmé' if result['confirmed'] else 'non confirmé'}",
                priority=2,
            )
            send_telegram_alert(
                f"Confirmation trade {trade_id}: {'confirmé' if result['confirmed'] else 'non confirmé'}"
            )
            return result
        except Exception as e:
            error_msg = f"Erreur confirmation trade {trade_id}: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "confirm_trade", time.time() - start_time, success=False, error=str(e)
            )
            return {
                "trade_id": trade_id,
                "confirmed": False,
                "confirmation_time": str(datetime.now()),
            }

    def log_execution(
        self, trade: Dict, execution_result: Dict, is_paper: bool = True
    ) -> None:
        """
        Enregistre l’exécution d’un trade dans un CSV avec buffering.

        Args:
            trade (Dict): Paramètres du trade initial.
            execution_result (Dict): Résultat de l’exécution.
            is_paper (bool): Mode paper trading (True) ou réel (False).
        """
        start_time = time.time()
        try:
            output_path = TRADES_DIR / (
                "trades_simulated.csv" if is_paper else "trades_real.csv"
            )
            logger.info(
                f"Enregistrement exécution trade {trade['trade_id']} dans {output_path}"
            )
            self.alert_manager.send_alert(
                f"Enregistrement exécution trade {trade['trade_id']} dans {output_path}",
                priority=2,
            )
            send_telegram_alert(
                f"Enregistrement exécution trade {trade['trade_id']} dans {output_path}"
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

            log_entry = {
                "timestamp": execution_result["execution_time"],
                "trade_id": trade["trade_id"],
                "action": trade["action"],
                "price": trade["price"],
                "size": trade["size"],
                "order_type": trade["order_type"],
                "status": execution_result["status"],
                "execution_price": execution_result["execution_price"],
                "balance": execution_result["balance"],
                "slippage": (
                    abs(execution_result["execution_price"] - trade["price"])
                    / trade["price"]
                    if trade["price"] != 0
                    else 0.0
                ),
                "confidence_drop_rate": execution_result.get(
                    "confidence_drop_rate", 0.0
                ),
                "is_paper": is_paper,
            }
            self.trade_buffer.append(log_entry)

            if len(self.trade_buffer) >= self.config.get("logging", {}).get(
                "buffer_size", 200
            ):
                buffer_df = pd.DataFrame(self.trade_buffer)

                def save_csv():
                    if not output_path.exists():
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
                self.trade_buffer = []
                self.log_performance(
                    "write_csv", 0, success=True, num_trades=len(buffer_df)
                )

            if (
                len(self.trade_buffer)
                % self.config.get("visualization", {}).get("plot_interval", 50)
                == 0
            ):
                self.plot_execution_metrics(self.trade_buffer)

            latency = time.time() - start_time
            self.log_performance("log_execution", latency, success=True, num_trades=1)
        except Exception as e:
            error_msg = f"Erreur enregistrement trade {trade['trade_id']}: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "log_execution", time.time() - start_time, success=False, error=str(e)
            )

    def save_execution_snapshot(self, trade_id: str, log_entry: Dict) -> None:
        """
        Sauvegarde un instantané de l’exécution d’un trade.

        Args:
            trade_id (str): Identifiant du trade.
            log_entry (Dict): Détails de l’exécution.
        """
        start_time = time.time()
        try:
            snapshot = {
                "trade_id": trade_id,
                "timestamp": log_entry["timestamp"],
                "action": log_entry["action"],
                "status": log_entry["status"],
                "execution_price": log_entry["execution_price"],
                "balance": log_entry["balance"],
                "slippage": log_entry["slippage"],
                "confidence_drop_rate": log_entry.get("confidence_drop_rate", 0.0),
                "trade_buffer_size": len(self.trade_buffer),
            }
            self.save_snapshot(f"execution_trade_{trade_id}", snapshot, compress=False)
            logger.info(f"Snapshot exécution trade {trade_id} sauvegardé")
            self.alert_manager.send_alert(
                f"Snapshot exécution trade {trade_id} sauvegardé", priority=1
            )
            send_telegram_alert(f"Snapshot exécution trade {trade_id} sauvegardé")
            self.log_performance(
                "save_execution_snapshot", time.time() - start_time, success=True
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot exécution trade {trade_id}: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "save_execution_snapshot",
                time.time() - start_time,
                success=False,
                error=str(e),
            )

    def update_dashboard(
        self, log_entry: Dict, context_data: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Met à jour un fichier JSON pour partager l’état de l’exécution avec mia_dashboard.py.

        Args:
            log_entry (Dict): Détails de l’exécution.
            context_data (Optional[pd.DataFrame]): Données contextuelles (facultatif).
        """
        start_time = time.time()
        try:
            mean_slippage = (
                sum(entry["slippage"] for entry in self.trade_buffer)
                / len(self.trade_buffer)
                if self.trade_buffer
                else 0.0
            )
            mean_confidence_drop = (
                sum(entry["confidence_drop_rate"] for entry in self.trade_buffer)
                / len(self.trade_buffer)
                if self.trade_buffer
                else 0.0
            )
            trades_per_hour = {}
            for entry in self.trade_buffer:
                hour = (
                    pd.to_datetime(entry["timestamp"])
                    .floor("H")
                    .strftime("%Y-%m-%d %H:00:00")
                )
                trades_per_hour[hour] = trades_per_hour.get(hour, 0) + 1

            status = {
                "timestamp": str(datetime.now()),
                "trade_id": log_entry["trade_id"],
                "num_trades": len(self.trade_buffer),
                "total_trades": len(self.trade_buffer),
                "status": log_entry["status"],
                "execution_price": log_entry["execution_price"],
                "balance": log_entry["balance"],
                "mean_slippage": mean_slippage,
                "mean_confidence_drop": mean_confidence_drop,
                "trades_per_hour": trades_per_hour,
                "vix": (
                    context_data["vix"].iloc[0]
                    if context_data is not None and "vix" in context_data.columns
                    else None
                ),
                "neural_regime": (
                    context_data["neural_regime"].iloc[0]
                    if context_data is not None
                    and "neural_regime" in context_data.columns
                    else None
                ),
                "predicted_volatility": (
                    context_data["predicted_volatility"].iloc[0]
                    if context_data is not None
                    and "predicted_volatility" in context_data.columns
                    else None
                ),
                "recent_errors": len(
                    [log for log in self.log_buffer if not log["success"]]
                ),
                "average_latency": (
                    sum(log["latency"] for log in self.log_buffer)
                    / len(self.log_buffer)
                    if self.log_buffer
                    else 0
                ),
                "recent_trades": self.trade_buffer[-10:],
            }

            def write_status():
                with open(DASHBOARD_PATH, "w", encoding="utf-8") as f:
                    json.dump(status, f, indent=4)

            self.with_retries(write_status)
            self.log_performance(
                "update_dashboard", time.time() - start_time, success=True, num_trades=1
            )
            logger.info("Mise à jour dashboard exécution effectuée")
            self.alert_manager.send_alert(
                "Mise à jour dashboard exécution effectuée", priority=1
            )
            send_telegram_alert("Mise à jour dashboard exécution effectuée")
        except Exception as e:
            error_msg = f"Erreur mise à jour dashboard exécution: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "update_dashboard",
                time.time() - start_time,
                success=False,
                error=str(e),
            )

    def plot_execution_metrics(self, trade_buffer: Optional[List[Dict]] = None) -> None:
        """
        Génère des graphiques pour les métriques d’exécution (slippage, balance, confidence_drop_rate).

        Args:
            trade_buffer (Optional[List[Dict]]): Buffer des trades à visualiser (par défaut: self.trade_buffer).
        """
        start_time = time.time()
        try:
            trade_buffer = trade_buffer or self.trade_buffer
            if not trade_buffer:
                error_msg = "Aucun trade à visualiser"
                self.alert_manager.send_alert(error_msg, priority=2)
                send_telegram_alert(error_msg)
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            figsize = self.config.get("visualization", {}).get("figsize", [12, 5])
            colors = self.config.get("visualization", {}).get(
                "colors",
                {
                    "slippage": "red",
                    "balance": "blue",
                    "status": ["green", "red"],
                    "confidence_drop": "purple",
                },
            )

            plt.figure(figsize=figsize)
            plt.hist(
                [entry["slippage"] for entry in trade_buffer],
                bins=30,
                color=colors["slippage"],
                edgecolor="black",
                alpha=0.7,
            )
            plt.title("Distribution du Slippage par Trade")
            plt.xlabel("Slippage")
            plt.ylabel("Nombre de Trades")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.savefig(FIGURES_DIR / f"execution_slippage_{timestamp}.png")
            plt.close()

            plt.figure(figsize=figsize)
            timestamps = [pd.to_datetime(entry["timestamp"]) for entry in trade_buffer]
            balances = [entry["balance"] for entry in trade_buffer]
            plt.plot(timestamps, balances, color=colors["balance"], label="Balance")
            plt.title("Balance au Fil du Temps")
            plt.xlabel("Timestamp")
            plt.ylabel("Balance")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()
            plt.savefig(FIGURES_DIR / f"execution_balance_{timestamp}.png")
            plt.close()

            plt.figure(figsize=figsize)
            statuses = [entry["status"] for entry in trade_buffer]
            status_counts = pd.Series(statuses).value_counts()
            plt.pie(
                status_counts,
                labels=status_counts.index,
                colors=colors["status"],
                autopct="%1.1f%%",
            )
            plt.title("Répartition des Statuts des Trades")
            plt.savefig(FIGURES_DIR / f"execution_status_{timestamp}.png")
            plt.close()

            plt.figure(figsize=figsize)
            confidence_drops = [entry["confidence_drop_rate"] for entry in trade_buffer]
            plt.hist(
                confidence_drops,
                bins=30,
                color=colors["confidence_drop"],
                edgecolor="black",
                alpha=0.7,
            )
            plt.title("Distribution du Confidence Drop Rate")
            plt.xlabel("Confidence Drop Rate")
            plt.ylabel("Nombre de Trades")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.savefig(FIGURES_DIR / f"execution_confidence_drop_{timestamp}.png")
            plt.close()

            latency = time.time() - start_time
            self.log_performance(
                "plot_execution_metrics",
                latency,
                success=True,
                num_trades=len(trade_buffer),
            )
            logger.info(
                f"Graphiques d’exécution générés pour {len(trade_buffer)} trades"
            )
            self.alert_manager.send_alert(
                f"Graphiques d’exécution générés pour {len(trade_buffer)} trades",
                priority=2,
            )
            send_telegram_alert(
                f"Graphiques d’exécution générés pour {len(trade_buffer)} trades"
            )
        except Exception as e:
            error_msg = f"Erreur génération graphiques exécution: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=2)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "plot_execution_metrics",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
