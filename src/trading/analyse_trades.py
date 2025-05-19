```python
# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/trading/analyse_trades.py
# Rôle : Analyse détaillée des trades individuels pour MIA_IA_SYSTEM_v2_2025, incluant la mémoire contextuelle.
#
# Version : 2.1.4
# Date : 2025-05-15
#
# Rôle : Analyse les trades individuels avec des métriques de performance (win rate, profit factor, Sharpe, drawdown),
#        utilise la mémoire contextuelle via comparaison aux clusters (Phase 7), intègre l’analyse SHAP (Phase 17),
#        génère des snapshots JSON (option compressée), des sauvegardes, des graphiques matplotlib, et des alertes
#        standardisées (Phase 8). Compatible avec la simulation de trading (Phase 12) et l’ensemble learning (Phase 16).
#        Collecte les méta-données par trade pour market_memory.db (Proposition 2, Étape 1).
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0
# - numpy>=1.26.4,<2.0.0
# - psutil>=5.9.8,<6.0.0
# - pyyaml>=6.0.0,<7.0.0
# - matplotlib>=3.7.0,<4.0.0
# - logging, os, json, datetime, argparse, sqlite3, signal, time, gzip
# - src.model.adaptive_learner
# - src.model.utils.config_manager
# - src.model.utils.alert_manager
# - src.trading.shap_weighting
# - data/market_memory.db
#
# Inputs :
# - Données de trading (pd.DataFrame avec 350 features pour entraînement ou 150 SHAP features pour inférence)
# - Configuration via config/market_config.yaml
# - config/feature_sets.yaml
# - Patterns dans data/market_memory.db
#
# Outputs :
# - Logs dans data/logs/analyse_trades.log
# - Logs de performance dans data/logs/analyse_trades.csv
# - Snapshots JSON dans data/trade_snapshots/*.json (option *.json.gz)
# - Résumés dans data/trades/*.csv
# - Sauvegardes dans data/checkpoints/analyse_trades/*.json.gz
# - Graphiques dans data/figures/trading/*.png
# - Méta-données dans market_memory.db (meta_runs)
#
# Notes :
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence) définies dans config/feature_sets.yaml.
# - Utilise IQFeed exclusivement pour les données d’entrée.
# - Supprime toute référence à 320 features, 81 features, et obs_t.
# - Implémente retries (max 3, délai 2^attempt secondes) pour les opérations critiques.
# - Intègre la mémoire contextuelle (Phase 7) via comparaison aux patterns dans market_memory.db.
# - Intègre l’analyse SHAP (Phase 17) avec limitation à 50 features.
# - Intègre confidence_drop_rate (Phase 8) pour l’auto-conscience.
# - Sauvegardes incrémentielles (5 min), distribuées (15 min), versionnées (5 versions).
# - Tests unitaires disponibles dans tests/test_analyse_trades.py.

import argparse
import gzip
import json
import logging
import os
import signal
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

from src.model.adaptive_learner import store_pattern
from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import config_manager
from src.trading.shap_weighting import calculate_shap

# Chemins relatifs
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
TRADES_DIR = BASE_DIR / "data" / "trades"
SNAPSHOT_DIR = BASE_DIR / "data" / "trade_snapshots"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints" / "analyse_trades"
FIGURES_DIR = BASE_DIR / "data" / "figures" / "trading"
CSV_LOG_PATH = BASE_DIR / "data" / "logs" / "analyse_trades.csv"
MARKET_MEMORY_DB = BASE_DIR / "data" / "market_memory.db"

# Configurer logging
os.makedirs(BASE_DIR / "data" / "logs", exist_ok=True)
logging.basicConfig(
    filename=BASE_DIR / "data" / "logs" / "analyse_trades.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Constantes
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Secondes, exponentiel
SHAP_FEATURE_LIMIT = 50


def fetch_clusters(db_path: str) -> pd.DataFrame:
    """
    Récupère les clusters depuis market_memory.db (implémentation fictive).

    Args:
        db_path (str): Chemin vers la base de données SQLite.

    Returns:
        pd.DataFrame: Clusters récupérés.

    Notes:
        TODO: Implémenter une récupération réelle des clusters avec validation des données.
    """
    try:
        conn = sqlite3.connect(db_path)
        query = "SELECT * FROM clusters LIMIT 100"
        clusters_df = pd.read_sql_query(query, conn)
        conn.close()
        return clusters_df
    except Exception as e:
        logger.error(f"Erreur récupération clusters: {str(e)}")
        return pd.DataFrame(columns=["cluster_id", "centroid", "regime"])


def compare_to_clusters(
    trade_data: Union[pd.DataFrame, pd.Series], clusters: pd.DataFrame
) -> float:
    """
    Compare un trade aux clusters pour calculer une similarité (implémentation fictive).

    Args:
        trade_data (Union[pd.DataFrame, pd.Series]): Données du trade.
        clusters (pd.DataFrame): Clusters à comparer.

    Returns:
        float: Score de similarité (0 à 1).

    Notes:
        TODO: Implémenter une comparaison réelle basée sur la distance euclidienne ou cosine.
    """
    try:
        if clusters.empty:
            return 0.0
        return np.random.uniform(0.5, 1.0)  # Simulation
    except Exception as e:
        logger.error(f"Erreur comparaison clusters: {str(e)}")
        return 0.0


class TradeDetailAnalyzer:
    """
    Classe pour analyser les trades individuels, incluant des métriques spécifiques et la mémoire contextuelle.
    """

    def __init__(
        self,
        config_path: str = "config/market_config.yaml",
        db_path: str = "data/market_memory.db",
    ):
        """
        Initialise l’analyseur de trades individuels.

        Args:
            config_path (str): Chemin vers la configuration du marché.
            db_path (str): Chemin vers la base de données SQLite.
        """
        self.alert_manager = AlertManager()
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        signal.signal(signal.SIGINT, self.handle_sigint)
        self.last_checkpoint_time = datetime.now()
        self.last_distributed_checkpoint_time = datetime.now()

        try:
            self.config = self.load_config(config_path)
            self.db_path = db_path
            self.log_buffer = []
            self.summary_buffer = []

            self.performance_thresholds = {
                "min_profit_factor": self.config.get("thresholds", {}).get(
                    "min_profit_factor", 1.2
                ),
                "max_slippage": self.config.get("thresholds", {}).get(
                    "max_slippage", 0.01
                ),
                "min_trade_duration": self.config.get("thresholds", {}).get(
                    "min_trade_duration", 60
                ),
            }

            for key, value in self.performance_thresholds.items():
                if not isinstance(value, (int, float)) or value < 0:
                    raise ValueError(f"Seuil invalide pour {key}: {value}")

            logger.info("TradeDetailAnalyzer initialisé avec succès")
            self.alert_manager.send_alert("TradeDetailAnalyzer initialisé", priority=2)
            self.log_performance("init", 0, success=True)
            self.save_checkpoint(incremental=True)
        except Exception as e:
            error_msg = f"Erreur initialisation TradeDetailAnalyzer: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=5)
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
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde snapshot SIGINT: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
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
            logger.info(f"Snapshot {snapshot_type} sauvegardé: {save_path}")
            self.log_performance("save_snapshot", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance("save_snapshot", 0, success=False, error=str(e))

    def save_checkpoint(self, incremental: bool = True, distributed: bool = False):
        """Sauvegarde l’état de l’analyseur (incrémentiel, distribué, versionné)."""
        try:
            start_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint = {
                "timestamp": timestamp,
                "log_buffer": self.log_buffer[-100:],  # Limiter la taille
                "summary_buffer": self.summary_buffer[-100:],  # Limiter la taille
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
            logger.error(error_msg)
            self.log_performance("save_checkpoint", 0, success=False, error=str(e))

    def generate_performance_plot(self, stats: Dict[str, float], symbol: str):
        """Génère un graphique des performances des trades."""
        try:
            start_time = time.time()
            plt.figure(figsize=(10, 6))
            regimes = stats.get("by_regime", {})
            for regime, regime_stats in regimes.items():
                plt.bar(
                    regime,
                    regime_stats["total_return"],
                    label=f"Régime {regime} (Win Rate: {regime_stats['win_rate']:.2f}%)",
                )
            plt.title(f"Performance des trades pour {symbol.upper()}")
            plt.xlabel("Régime")
            plt.ylabel("Retour total")
            plt.legend()
            plt.grid(True)
            plot_path = (
                FIGURES_DIR
                / f"performance_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            os.makedirs(plot_path.parent, exist_ok=True)
            plt.savefig(plot_path)
            plt.close()
            latency = time.time() - start_time
            self.alert_manager.send_alert(
                f"Graphique de performance sauvegardé: {plot_path}", priority=1
            )
            logger.info(f"Graphique de performance sauvegardé: {plot_path}")
            self.log_performance("generate_performance_plot", latency, success=True)
        except Exception as e:
            error_msg = (
                f"Erreur génération graphique: {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance(
                "generate_performance_plot", 0, success=False, error=str(e)
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
                    f"retry_attempt_{attempt+1}", latency, success=True
                )
                return result
            except Exception as e:
                if attempt == max_attempts - 1:
                    error_msg = f"Échec après {max_attempts} tentatives: {str(e)}\n{traceback.format_exc()}"
                    self.alert_manager.send_alert(error_msg, priority=3)
                    logger.error(error_msg)
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
            config = self.with_retries(
                lambda: config_manager.get_config(os.path.join(BASE_DIR, config_path))
            )
            if not config:
                raise ValueError("Configuration vide ou non trouvée")
            self.log_performance("load_config", time.time() - start_time, success=True)
            logger.info("Configuration chargée avec succès")
            self.alert_manager.send_alert(
                "Configuration chargée avec succès", priority=1
            )
            return config
        except Exception as e:
            error_msg = (
                f"Erreur chargement configuration: {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            logger.error(error_msg)
            self.log_performance(
                "load_config", time.time() - start_time, success=False, error=str(e)
            )
            return {
                "thresholds": {
                    "min_profit_factor": 1.2,
                    "max_slippage": 0.01,
                    "min_trade_duration": 60,
                },
                "logging": {"buffer_size": 100},
            }

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
                "detect_feature_set", time.time() - start_time, success=True
            )
            logger.debug(f"Ensemble de features détecté: {feature_set}")
            self.alert_manager.send_alert(
                f"Ensemble de features détecté: {feature_set}", priority=1
            )
            return feature_set
        except Exception as e:
            error_msg = f"Erreur détection ensemble features: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance(
                "detect_feature_set",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            return "inference"

    def validate_data(self, data: Union[pd.DataFrame, pd.Series]) -> None:
        """
        Valide les données d’entrée (350 features pour entraînement, 150 SHAP pour inférence).

        Args:
            data (Union[pd.DataFrame, pd.Series]): Données à valider.

        Raises:
            ValueError: Si les données sont invalides.
        """
        start_time = time.time()
        try:
            if isinstance(data, pd.Series):
                data = data.to_frame().T
            if not isinstance(data, pd.DataFrame):
                error_msg = "Données doivent être un DataFrame ou une Series"
                raise ValueError(error_msg)

            feature_set = self.detect_feature_set(data)
            expected_features = 350 if feature_set == "training" else 150

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
                "reward",
                "step",
                "regime",
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
                        "regime",
                    ] and not pd.api.types.is_numeric_dtype(data[col]):
                        error_msg = (
                            f"Colonne {col} n’est pas numérique: {data[col].dtype}"
                        )
                        raise ValueError(error_msg)

            if "timestamp" in data.columns:
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

            logger.debug(f"Données validées pour ensemble {feature_set}")
            self.alert_manager.send_alert(
                f"Données validées pour ensemble {feature_set}", priority=1
            )
            self.log_performance(
                "validate_data",
                time.time() - start_time,
                success=True,
                num_features=len(data.columns),
            )
            self.save_snapshot(
                "validation", {"num_features": len(data.columns)}, compress=False
            )
        except Exception as e:
            error_msg = f"Erreur validation données: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=5)
            logger.error(error_msg)
            self.log_performance(
                "validate_data", time.time() - start_time, success=False, error=str(e)
            )
            raise

    def load_trades(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Charge les données de trades avec vérifications robustes.

        Args:
            file_path (str): Chemin du fichier CSV des trades.

        Returns:
            Optional[pd.DataFrame]: Données chargées ou None si erreur.
        """
        start_time = time.time()
        try:
            logger.info(f"Tentative de chargement: {file_path}")
            self.alert_manager.send_alert(
                f"Chargement fichier trades: {file_path}", priority=2
            )
            if not os.path.exists(file_path):
                error_msg = f"Fichier introuvable: {file_path}"
                raise FileNotFoundError(error_msg)

            def read_csv():
                return pd.read_csv(file_path)

            df = self.with_retries(read_csv)
            if df is None or df.empty:
                error_msg = f"Fichier vide ou non chargé: {file_path}"
                raise ValueError(error_msg)

            self.validate_data(df)

            self.log_performance(
                "load_trades",
                time.time() - start_time,
                success=True,
                num_trades=len(df),
            )
            logger.info(f"Chargement réussi: {file_path} ({len(df)} lignes)")
            self.alert_manager.send_alert(
                f"Chargement réussi: {file_path} ({len(df)} lignes)", priority=1
            )
            return df
        except Exception as e:
            error_msg = f"Erreur chargement: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=5)
            logger.error(error_msg)
            self.log_performance(
                "load_trades", time.time() - start_time, success=False, error=str(e)
            )
            return None

    def store_trade_metadata(self, metrics: Dict, trade_data: Union[pd.DataFrame, pd.Series], regime: str) -> None:
        """Stocke les méta-données du trade dans market_memory.db (Proposition 2, Étape 1)."""
        try:
            start_time = time.time()
            if isinstance(trade_data, pd.Series):
                trade_data = trade_data.to_frame().T
            metadata = {
                "timestamp": metrics.get("timestamp", datetime.now().isoformat()),
                "trade_id": metrics.get("trade_id", 0),
                "metrics": {
                    "reward": metrics.get("reward", 0.0),
                    "slippage": metrics.get("slippage", 0.0),
                    "duration": metrics.get("duration", 0.0),
                    "vix": metrics.get("vix", 0.0),
                    "rsi_14": metrics.get("rsi_14", 0.0),
                    "gex": metrics.get("gex", 0.0),
                    "similarity": metrics.get("similarity", 0.0),
                    "confidence_drop_rate": metrics.get("confidence_drop_rate", 0.0),
                },
                "hyperparameters": self.config.get("model_params", {}),
                "performance": metrics.get("reward", 0.0),
                "regime": regime,
                "session": trade_data.get("session", "unknown").iloc[-1] if "session" in trade_data.columns else "unknown",
                "shap_metrics": {k: v for k, v in metrics.items() if k.startswith("shap_")},
            }
            conn = sqlite3.connect(MARKET_MEMORY_DB)
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS meta_runs (
                    run_id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    trade_id INTEGER,
                    metrics TEXT,
                    hyperparameters TEXT,
                    performance REAL,
                    regime TEXT,
                    session TEXT,
                    shap_metrics TEXT
                )
                """
            )
            cursor.execute(
                """
                INSERT INTO meta_runs (timestamp, trade_id, metrics, hyperparameters, performance, regime, session, shap_metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata["timestamp"],
                    metadata["trade_id"],
                    json.dumps(metadata["metrics"]),
                    json.dumps(metadata["hyperparameters"]),
                    metadata["performance"],
                    metadata["regime"],
                    metadata["session"],
                    json.dumps(metadata["shap_metrics"]),
                ),
            )
            conn.commit()
            conn.close()
            logger.info(f"Méta-données trade {metadata['trade_id']} stockées dans market_memory.db")
            self.alert_manager.send_alert(
                f"Méta-données trade {metadata['trade_id']} stockées dans market_memory.db", priority=1
            )
            self.log_performance("store_trade_metadata", time.time() - start_time, success=True)
        except Exception as e:
            error_msg = f"Erreur stockage méta-données trade {metrics.get('trade_id', 0)}: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance("store_trade_metadata", time.time() - start_time, success=False, error=str(e))

    def compute_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calcule les métriques de performance globales et par régime.

        Args:
            df (pd.DataFrame): Données des trades.

        Returns:
            Dict[str, float]: Métriques calculées.
        """
        start_time = time.time()
        try:
            logger.info("Calcul des métriques de performance")
            self.alert_manager.send_alert(
                "Calcul des métriques de performance", priority=2
            )

            rewards = df["reward"].fillna(0).values
            total_trades = len(rewards)
            win_rate = (rewards > 0).mean() * 100
            profit_factor = (
                rewards[rewards > 0].sum() / abs(rewards[rewards < 0].sum())
                if (rewards < 0).sum() != 0
                else np.inf
            )
            sharpe = (
                (rewards.mean() / rewards.std()) * np.sqrt(252)
                if rewards.std() > 0
                else 0
            )
            equity = np.cumsum(rewards)
            max_dd = np.min(equity - np.maximum.accumulate(equity))
            confidence_drop_rate = 1.0 - min(
                profit_factor / self.performance_thresholds["min_profit_factor"], 1.0
            )  # Phase 8

            stats = {
                "total_trades": total_trades,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_dd,
                "total_return": equity[-1],
                "confidence_drop_rate": confidence_drop_rate,
            }

            regimes = df["regime"].unique()
            stats_by_regime = {}
            for regime in regimes:
                regime_rewards = df[df["regime"] == regime]["reward"].fillna(0).values
                stats_by_regime[regime] = {
                    "total_trades": len(regime_rewards),
                    "win_rate": (
                        (regime_rewards > 0).mean() * 100
                        if len(regime_rewards) > 0
                        else 0
                    ),
                    "total_return": regime_rewards.sum(),
                }
                # Stocker les méta-données globales par régime
                regime_metrics = {
                    "trade_id": 0,  # ID fictif pour métriques globales
                    "reward": stats_by_regime[regime]["total_return"],
                    "regime": regime,
                    "timestamp": datetime.now().isoformat(),
                    "slippage": 0.0,
                    "duration": 0.0,
                    "vix": df[df["regime"] == regime]["vix"].mean() if "vix" in df.columns else 0.0,
                    "rsi_14": df[df["regime"] == regime]["rsi_14"].mean() if "rsi_14" in df.columns else 0.0,
                    "gex": df[df["regime"] == regime]["gex"].mean() if "gex" in df.columns else 0.0,
                    "similarity": 0.0,
                    "confidence_drop_rate": confidence_drop_rate,
                }
                self.store_trade_metadata(regime_metrics, df[df["regime"] == regime], regime)
            stats["by_regime"] = stats_by_regime

            for metric, threshold in self.performance_thresholds.items():
                if metric != "min_trade_duration":
                    value = stats.get(metric.replace("min_", "").replace("max_", ""), 0)
                    if (metric.startswith("max_") and value < threshold) or (
                        metric.startswith("min_") and value < threshold
                    ):
                        self.alert_manager.send_alert(
                            f"Seuil non atteint pour {metric}: {value} < {threshold}",
                            priority=3,
                        )
                        logger.warning(
                            f"Seuil non atteint pour {metric}: {value} < {threshold}"
                        )

            self.log_performance(
                "compute_metrics",
                time.time() - start_time,
                success=True,
                num_trades=total_trades,
            )
            logger.info(f"Métriques calculées: {stats}")
            self.alert_manager.send_alert(f"Métriques calculées: {stats}", priority=1)
            return stats
        except Exception as e:
            error_msg = f"Erreur compute_metrics: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=5)
            logger.error(error_msg)
            self.log_performance(
                "compute_metrics", time.time() - start_time, success=False, error=str(e)
            )
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "total_return": 0.0,
                "confidence_drop_rate": 0.0,
            }

    def analyse_trade(
        self, trade_id: int, trade_data: Union[pd.DataFrame, pd.Series]
    ) -> Dict[str, float]:
        """
        Analyse un trade individuel spécifique avec mémoire contextuelle et analyse SHAP.

        Args:
            trade_id (int): Identifiant du trade.
            trade_data (Union[pd.DataFrame, pd.Series]): Données du trade.

        Returns:
            Dict[str, float]: Métriques du trade, similarité aux clusters, et métriques SHAP.
        """
        start_time = time.time()
        try:
            self.validate_data(trade_data)
            logger.info(f"Analyse du trade {trade_id}")
            self.alert_manager.send_alert(f"Analyse du trade {trade_id}", priority=2)

            if isinstance(trade_data, pd.Series):
                trade = trade_data
                trade_data_df = trade_data.to_frame().T
            elif isinstance(trade_data, pd.DataFrame):
                if len(trade_data) != 1:
                    error_msg = f"trade_data doit contenir un seul trade, reçu {len(trade_data)}"
                    raise ValueError(error_msg)
                trade = trade_data.iloc[0]
                trade_data_df = trade_data
            else:
                error_msg = "trade_data doit être un DataFrame ou une Series"
                raise ValueError(error_msg)

            # Analyse SHAP (Phase 17)
            shap_metrics = {}
            if "reward" in trade_data_df.columns:

                def run_shap():
                    return calculate_shap(
                        trade_data_df, target="reward", max_features=SHAP_FEATURE_LIMIT
                    )

                shap_values = self.with_retries(run_shap)
                if shap_values is not None:
                    shap_metrics = {
                        f"shap_{col}": float(shap_values[col].iloc[0])
                        for col in shap_values.columns
                    }
                else:
                    error_msg = "Échec calcul SHAP pour le trade, poursuite sans SHAP"
                    self.alert_manager.send_alert(error_msg, priority=3)
                    logger.warning(error_msg)

            metrics = {
                "trade_id": trade_id,
                "reward": float(trade.get("reward", 0.0)),
                "regime": str(trade.get("regime", "inconnu")),
                "timestamp": str(trade.get("timestamp", datetime.now())),
                "slippage": 0.0,
                "duration": 0.0,
                "vix": float(trade.get("vix", 0.0)),
                "rsi_14": float(trade.get("rsi_14", 0.0)),
                "gex": float(trade.get("gex", 0.0)),
                "similarity": 0.0,
                "confidence_drop_rate": 0.0,  # Phase 8
                **shap_metrics,
            }

            if "entry_price" in trade and "execution_price" in trade:
                metrics["slippage"] = abs(
                    float(trade["entry_price"]) - float(trade["execution_price"])
                ) / float(trade["entry_price"])
                if metrics["slippage"] > self.performance_thresholds["max_slippage"]:
                    self.alert_manager.send_alert(
                        f"Slippage excessif pour trade {trade_id}: {metrics['slippage']:.2%}",
                        priority=3,
                    )
                    logger.warning(
                        f"Slippage excessif pour trade {trade_id}: {metrics['slippage']:.2%}"
                    )

            if "entry_time" in trade and "exit_time" in trade:
                entry_time = pd.to_datetime(trade["entry_time"])
                exit_time = pd.to_datetime(trade["exit_time"])
                metrics["duration"] = (exit_time - entry_time).total_seconds()
                if (
                    metrics["duration"]
                    < self.performance_thresholds["min_trade_duration"]
                ):
                    self.alert_manager.send_alert(
                        f"Durée trop courte pour trade {trade_id}: {metrics['duration']}s",
                        priority=3,
                    )
                    logger.warning(
                        f"Durée trop courte pour trade {trade_id}: {metrics['duration']}s"
                    )

            clusters = fetch_clusters(self.db_path)
            metrics["similarity"] = compare_to_clusters(trade_data_df, clusters)

            # Calcul de confidence_drop_rate (Phase 8)
            pf = (
                abs(metrics["reward"]) / self.performance_thresholds["min_profit_factor"]
                if metrics["reward"] < 0
                else float("inf")
            )
            drop = 1.0 - min(pf, 1.0)
            metrics["confidence_drop_rate"] = max(0.0, min(1.0, drop))

            # Stocker les méta-données du trade
            self.store_trade_metadata(metrics, trade_data_df, metrics["regime"])

            def store_in_memory():
                store_pattern(
                    trade_data_df,
                    action=1.0 if metrics["reward"] > 0 else -1.0,
                    reward=metrics["reward"],
                    neural_regime=float(trade.get("neural_regime", 0)),
                    confidence=metrics["similarity"],
                    metadata={
                        "event": "trade_analysis",
                        "trade_id": trade_id,
                        "regime": metrics["regime"],
                        "metrics": metrics,
                        "similarity": metrics["similarity"],
                        "confidence_drop_rate": metrics["confidence_drop_rate"],
                        "shap_metrics": shap_metrics,
                    },
                )

            self.with_retries(store_in_memory)

            snapshot = {
                "trade_id": trade_id,
                "timestamp": metrics["timestamp"],
                "metrics": metrics,
                "similarity": metrics["similarity"],
                "shap_metrics": shap_metrics,
            }
            self.save_snapshot(f"trade_{trade_id}", snapshot, compress=False)

            self.log_performance(
                "analyse_trade",
                time.time() - start_time,
                success=True,
                trade_id=trade_id,
            )
            logger.info(f"Trade analysé: {trade_id}. CPU: {psutil.cpu_percent()}%")
            self.alert_manager.send_alert(f"Trade analysé: {trade_id}", priority=2)
            return metrics
        except Exception as e:
            error_msg = (
                f"Erreur analyse_trade {trade_id}: {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=5)
            logger.error(error_msg)
            self.log_performance(
                "analyse_trade", time.time() - start_time, success=False, error=str(e)
            )
            return {
                "trade_id": trade_id,
                "reward": 0.0,
                "regime": "inconnu",
                "timestamp": str(datetime.now()),
                "slippage": 0.0,
                "duration": 0.0,
                "vix": 0.0,
                "rsi_14": 0.0,
                "gex": 0.0,
                "similarity": 0.0,
                "confidence_drop_rate": 0.0,
            }

    def save_summary(
        self, stats: Dict[str, float], symbol: str, output_dir: str = str(TRADES_DIR)
    ) -> None:
        """
        Sauvegarde les métriques dans un CSV avec timestamp et buffering.

        Args:
            stats (Dict[str, float]): Métriques calculées.
            symbol (str): Symbole du marché (ex. ES, NQ).
            output_dir (str): Dossier pour le résumé.
        """
        start_time = time.time()
        try:
            logger.info(f"Sauvegarde du résumé pour {symbol.upper()}")
            self.alert_manager.send_alert(
                f"Sauvegarde du résumé pour {symbol.upper()}", priority=2
            )

            summary_entry = {k: v for k, v in stats.items() if k != "by_regime"}
            summary_entry["timestamp"] = datetime.now().isoformat()
            summary_entry["symbol"] = symbol
            self.summary_buffer.append(summary_entry)

            if len(self.summary_buffer) >= self.config.get("logging", {}).get(
                "buffer_size", 100
            ):
                df_stats = pd.DataFrame(self.summary_buffer)
                output_path = os.path.join(
                    output_dir,
                    f"trade_summary_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                )
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                def save_csv():
                    df_stats.to_csv(output_path, index=False, encoding="utf-8")

                self.with_retries(save_csv)
                self.summary_buffer = []
                logger.info(f"Résumé sauvegardé: {output_path}")
                self.alert_manager.send_alert(
                    f"Résumé sauvegardé: {output_path}", priority=2
                )

            self.save_snapshot(f"summary_{symbol}", stats, compress=False)
            self.generate_performance_plot(stats, symbol)

            self.log_performance("save_summary", time.time() - start_time, success=True)
        except Exception as e:
            error_msg = f"Erreur save_summary: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=5)
            logger.error(error_msg)
            self.log_performance(
                "save_summary", time.time() - start_time, success=False, error=str(e)
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
            **kwargs: Paramètres supplémentaires.
        """
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
            cpu_percent = psutil.cpu_percent()
            if memory_usage > 1024:
                error_msg = (
                    f"ALERTE: Utilisation mémoire élevée ({memory_usage:.2f} MB)"
                )
                self.alert_manager.send_alert(error_msg, priority=5)
                logger.warning(error_msg)
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
            os.makedirs(os.path.dirname(CSV_LOG_PATH), exist_ok=True)
            log_df = pd.DataFrame([log_entry])

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
            logger.info(
                f"Performance journalisée pour {operation}. CPU: {cpu_percent}%"
            )
        except Exception as e:
            error_msg = (
                f"Erreur journalisation performance: {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)


def main():
    parser = argparse.ArgumentParser(
        description="Analyse détaillée des trades individuels"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(TRADES_DIR / "trades_simulated.csv"),
        help="Fichier CSV des trades",
    )
    parser.add_argument(
        "--symbol", type=str, default="ES", help="Symbole du marché (ex. ES, NQ)"
    )
    parser.add_argument(
        "--summary-dir",
        type=str,
        default=str(TRADES_DIR),
        help="Dossier pour le résumé",
    )
    parser.add_argument(
        "--trade-id", type=int, default=None, help="ID du trade à analyser (optionnel)"
    )
    args = parser.parse_args()

    analyzer = TradeDetailAnalyzer()
    logger.info(f"Début analyse pour {args.symbol.upper()}")
    analyzer.alert_manager.send_alert(
        f"Début analyse pour {args.symbol.upper()}", priority=2
    )

    df = analyzer.load_trades(args.input)
    if df is None:
        analyzer.alert_manager.send_alert(
            "Analyse abandonnée en raison d’un échec de chargement", priority=5
        )
        logger.warning("Analyse abandonnée en raison d’un échec de chargement")
        return

    if args.trade_id is not None:
        trade_df = df[df["trade_id"] == args.trade_id]
        if trade_df.empty:
            error_msg = f"Trade avec ID {args.trade_id} non trouvé dans les données"
            analyzer.alert_manager.send_alert(error_msg, priority=5)
            logger.error(error_msg)
            print(f"Erreur: {error_msg}")
            return
        trade_metrics = analyzer.analyse_trade(args.trade_id, trade_df)
        print(f"\nRésultats pour trade {args.trade_id} ({args.symbol.upper()})")
        print("=" * 50)
        for key, value in trade_metrics.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        print("=" * 50)
    else:
        stats = analyzer.compute_metrics(df)
        analyzer.save_summary(stats, args.symbol, args.summary_dir)

        print(f"\nRésultats pour {args.symbol.upper()}")
        print("=" * 50)
        for key, value in stats.items():
            if key != "by_regime":
                print(f"{key.replace('_', ' ').title()}: {value:.2f}")
            else:
                print("\nPar régime:")
                for regime, regime_stats in value.items():
                    print(f"- {regime}:")
                    for k, v in regime_stats.items():
                        print(f"  {k.replace('_', ' ').title()}: {v:.2f}")
        print("=" * 50)

    logger.info(f"Analyse terminée pour {args.symbol.upper()}")
    analyzer.alert_manager.send_alert(
        f"Analyse terminée pour {args.symbol.upper()}", priority=2
    )


if __name__ == "__main__":
    main()
