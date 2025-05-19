```python
# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/trading/analyse_results.py
# Rôle : Analyse globale des trades (simulés/réels) pour MIA_IA_SYSTEM_v2_2025, incluant métriques, SHAP, suggestions, rapports périodiques, et alertes multicanaux.
#
# Version : 2.1.4
# Date : 2025-05-15
#
# Rôle : Analyse les résultats des trades avec des métriques globales et par régime/session (win rate, profit factor, Sharpe, drawdown, consecutive losses),
#        intègre l’analyse SHAP incrémentale (Phase 17), utilise une pipeline neurale pour neural_regime (Phase 6), génère des rapports périodiques
#        (quotidiens à 16h05 UTC, hebdomadaires le vendredi à 16h00 UTC, mensuels le dernier jour à 00h00 UTC, tous les 500/1000 trades), et envoie des alertes
#        multicanaux (Telegram, Discord, email). Compatible avec la mémoire contextuelle (Phase 7), la simulation de trading (Phase 12), et l’ensemble learning (Phase 16).
#        Inclut des narratifs LLM via GPT-4o-mini (Proposition 1) et collecte des méta-données dans market_memory.db (Proposition 2, Étape 1).
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, matplotlib>=3.8.0,<4.0.0, psutil>=5.9.8,<6.0.0
# - pyyaml>=6.0.0,<7.0.0, sqlalchemy>=2.0.0,<3.0.0, pandera>=0.10.0,<1.0.0, dask>=2023.0.0,<2024.0.0
# - delta-spark>=2.0.0,<3.0.0, ydata-profiling>=4.0.0,<5.0.0, jinja2>=3.0.0,<4.0.0, weasyprint>=57.0,<58.0
# - plotly>=5.0.0,<6.0.0, prometheus-client>=0.14.0,<1.0.0, apscheduler>=3.9.0,<4.0.0, click>=8.0.0,<9.0.0
# - durable-rules>=2.0.0,<3.0.0, scikit-learn>=1.2.0,<2.0.0, openai>=1.0.0,<2.0.0
# - src.features.neural_pipeline, src.mind.mind, src.model.adaptive_learner
# - src.model.utils.config_manager, src.model.utils.alert_manager, src.model.utils.export_visuals
# - src.trading.shap_weighting, src.trading.trade_loader, src.trading.schema_validator
# - src.trading.metrics_computer, src.trading.shap_analyzer, src.trading.suggestion_engine
# - src.trading.report_generator, src.trading.alert_dispatcher
#
# Inputs :
# - Données de trading (pd.DataFrame avec 350 features pour entraînement ou 150 SHAP features pour inférence)
# - Configuration via config/market_config.yaml, config/feature_sets.yaml
# - Sources : data/iqfeed/*.csv, data/trades.db, data/market_memory.db
#
# Outputs :
# - Logs dans data/logs/analyse_results.log, data/logs/analyse_results.csv
# - Snapshots JSON compressés dans data/trading/analyse_snapshots/*.json.gz
# - Dashboard JSON dans data/trading/analyse_results_dashboard.json
# - Graphiques dans data/figures/trading/*.png, data/figures/trading/*.pdf
# - Résumés dans data/trades/*.csv
# - Sauvegardes dans data/checkpoints/analyse_results/*.json.gz
# - Rapports dans data/reports/*.md, data/reports/*.html
# - Audit log dans data/audit/audit.log
# - SHAP dans data/shap/YYYYMMDD.parquet
# - Méta-données dans market_memory.db (meta_runs)
#
# Notes :
# - Utilise IQFeed exclusivement via data_provider.py pour les données d’entrée.
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence) définies dans config/feature_sets.yaml.
# - Supprime toute référence à 320 features, 81 features, et obs_t.
# - Implémente retries (max 3, délai 2^attempt secondes) pour les opérations critiques.
# - Gère les sessions de trading (Londres, New York, Asie) via TradeLoader.
# - Génère des rapports périodiques (quotidiens à 16h05 UTC, hebdomadaires le vendredi à 16h00 UTC, mensuels le dernier jour à 00h00 UTC, tous les 500/1000 trades).
# - Envoie des alertes multicanaux (Telegram @MIA_IA_SYSTEM, Discord Mia IA Bot#5399, email lazardjackson5@gmail.com) avec limitation de débit.
# - Valide les données avec Pandera (trade_duration, session, prob_buy_sac, etc.).
# - Stocke SHAP incrémentalement en Parquet.
# - Génère des suggestions ML basées sur règles YAML et arbre de décision.
# - Génère des narratifs via GPT-4o-mini pour les rapports (Proposition 1).
# - Stocke les méta-données dans market_memory.db (Proposition 2, Étape 1).
# - Tests unitaires dans tests/test_analyse_results.py.

import argparse
import gzip
import hashlib
import itertools
import json
import logging
import os
import signal
import sqlite3
import time
import traceback
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import click
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandera as pa
import psutil
import sqlalchemy as sa
from apscheduler.schedulers.background import BackgroundScheduler
from delta.tables import DeltaTable
from durable.engine import ruleset
from jinja2 import Environment, FileSystemLoader
from pandera import DataFrameSchema, Column, Check
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
from ydata_profiling import ProfileReport

from src.features.neural_pipeline import NeuralPipeline
from src.mind.mind import MindEngine
from src.model.adaptive_learner import store_pattern
from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import config_manager
from src.model.utils.export_visuals import export_report
from src.trading.alert_dispatcher import AlertDispatcher
from src.trading.metrics_computer import MetricsComputer
from src.trading.report_generator import ReportGenerator
from src.trading.schema_validator import SchemaValidator
from src.trading.shap_analyzer import ShapAnalyzer
from src.trading.suggestion_engine import SuggestionEngine
from src.trading.trade_loader import TradeLoader
from src.trading.shap_weighting import calculate_shap

# Chemins relatifs
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
FIGURES_DIR = BASE_DIR / "data" / "figures" / "trading"
TRADES_DIR = BASE_DIR / "data" / "trades"
SNAPSHOT_DIR = BASE_DIR / "data" / "trading" / "analyse_snapshots"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints" / "analyse_results"
CSV_LOG_PATH = BASE_DIR / "data" / "logs" / "analyse_results.csv"
DASHBOARD_PATH = BASE_DIR / "data" / "trading" / "analyse_results_dashboard.json"
REPORTS_DIR = BASE_DIR / "data" / "reports"
AUDIT_LOG_PATH = BASE_DIR / "data" / "audit" / "audit.log"
MARKET_MEMORY_DB = BASE_DIR / "data" / "market_memory.db"

# Configurer logging
os.makedirs(BASE_DIR / "data" / "logs", exist_ok=True)
logging.basicConfig(
    filename=BASE_DIR / "data" / "logs" / "analyse_results.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Constantes
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Secondes, exponentiel
SHAP_FEATURE_LIMIT = 50
MAX_CACHE_SIZE = 100

class TradeAnalyzer:
    """
    Classe pour analyser les résultats de trading, incluant les métriques globales, par régime, et l’analyse SHAP.
    """

    def __init__(self, config_path: str = "config/market_config.yaml"):
        """
        Initialise l’analyseur de résultats de trading.

        Args:
            config_path (str): Chemin vers la configuration du marché.
        """
        self.alert_manager = AlertManager()
        SNAPSHOT_DIR.mkdir(exist_ok=True)
        FIGURES_DIR.mkdir(exist_ok=True)
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        CSV_LOG_PATH.parent.mkdir(exist_ok=True)
        signal.signal(signal.SIGINT, self.handle_sigint)
        self.last_checkpoint_time = datetime.now()
        self.last_distributed_checkpoint_time = datetime.now()

        try:
            self.config = self.load_config(config_path)
            self.mind_engine = MindEngine()
            self.log_buffer = []
            self.summary_buffer = []
            self.neural_cache = OrderedDict()

            self.performance_thresholds = {
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
                if key == "min_sharpe" and value <= 0:
                    raise ValueError(f"Seuil {key} doit être positif: {value}")
                if key == "max_drawdown" and value >= 0:
                    raise ValueError(f"Seuil {key} doit être négatif: {value}")
                if key == "min_profit_factor" and value <= 0:
                    raise ValueError(f"Seuil {key} doit être positif: {value}")

            self.use_neural_pipeline = self.config.get("analysis", {}).get(
                "use_neural_pipeline", True
            )

            logger.info("TradeAnalyzer initialisé avec succès")
            self.alert_manager.send_alert("TradeAnalyzer initialisé", priority=2)
            self.log_performance("init", 0, success=True)
            self.save_checkpoint(incremental=True)
        except Exception as e:
            error_msg = f"Erreur initialisation TradeAnalyzer: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=5)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            raise

        try:
            self.trade_loader = TradeLoader(self.config, self)
            self.schema_validator = SchemaValidator(self.config)
            self.metrics_computer = MetricsComputer(self.config)
            self.shap_analyzer = ShapAnalyzer(self.config)
            self.suggestion_engine = SuggestionEngine(self.config)
            self.report_generator = ReportGenerator(self.config)
            self.alert_dispatcher = AlertDispatcher(self.config)
            self.trade_count = 0
            REPORTS_DIR.mkdir(exist_ok=True)
            AUDIT_LOG_PATH.parent.mkdir(exist_ok=True)
            MARKET_MEMORY_DB.parent.mkdir(exist_ok=True)
            self.setup_scheduler()
            logger.info("Nouveaux composants TradeAnalyzer initialisés")
            self.alert_dispatcher.send_alert("Nouveaux composants TradeAnalyzer initialisés", priority=2)
            self.log_audit("init_components", {"components": ["trade_loader", "schema_validator", "metrics_computer", "shap_analyzer", "suggestion_engine", "report_generator", "alert_dispatcher"]})
        except Exception as e:
            error_msg = f"Erreur initialisation nouveaux composants TradeAnalyzer: {str(e)}\n{traceback.format_exc()}"
            self.alert_dispatcher.send_alert(error_msg, priority=5)
            logger.error(error_msg)
            self.log_performance("init_components", 0, success=False, error=str(e))
            raise

    def handle_sigint(self, signum, frame):
        """Gère l'arrêt propre sur SIGINT."""
        snapshot = {"timestamp": datetime.now().isoformat(), "status": "SIGINT"}
        try:
            self.save_snapshot("sigint", snapshot)
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

    def save_snapshot(self, snapshot_type: str, data: Dict) -> None:
        """Sauvegarde un instantané des résultats avec compression gzip."""
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
            with gzip.open(f"{path}.gz", "wt", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=4)
            latency = time.time() - start_time
            self.alert_manager.send_alert(
                f"Snapshot {snapshot_type} sauvegardé: {path}.gz", priority=1
            )
            logger.info(f"Snapshot {snapshot_type} sauvegardé: {path}.gz")
            self.log_performance("save_snapshot", latency, success=True)
            self.log_audit("save_snapshot", {"snapshot_type": snapshot_type, "path": str(path) + ".gz"})
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
                "log_buffer": self.log_buffer[-100:],
                "summary_buffer": self.summary_buffer[-100:],
                "neural_cache": {k: True for k in self.neural_cache},
            }
            checkpoint_path = CHECKPOINT_DIR / f"checkpoint_{timestamp}.json.gz"
            os.makedirs(checkpoint_path.parent, exist_ok=True)
            with gzip.open(checkpoint_path, "wt", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=4)

            checkpoints = sorted(CHECKPOINT_DIR.glob("checkpoint_*.json.gz"))
            if len(checkpoints) > 5:
                for old_checkpoint in checkpoints[:-5]:
                    os.remove(old_checkpoint)

            if distributed:
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
            self.log_audit("save_checkpoint", {"path": str(checkpoint_path), "incremental": incremental})
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde checkpoint: {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)
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
                self.log_audit("retry_success", {"operation": func.__name__, "attempt": attempt + 1})
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
                    self.log_audit("retry_failure", {"operation": func.__name__, "error": str(e)})
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
                lambda: config_manager.get_config(BASE_DIR / config_path)
            )
            if not config:
                raise ValueError("Configuration vide ou non trouvée")
            self.log_performance("load_config", time.time() - start_time, success=True)
            logger.info("Configuration chargée avec succès")
            self.alert_manager.send_alert(
                "Configuration chargée avec succès", priority=1
            )
            self.log_audit("load_config", {"path": config_path})
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
            self.log_audit("load_config_failure", {"path": config_path, "error": str(e)})
            return {
                "thresholds": {
                    "min_sharpe": 0.5,
                    "max_drawdown": -1000.0,
                    "min_profit_factor": 1.2,
                },
                "analysis": {"use_neural_pipeline": True},
                "logging": {"buffer_size": 100},
                "trade_sources": ["data/iqfeed/trades.csv", "data/trades.db"],
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
            self.log_audit("detect_feature_set", {"feature_set": feature_set, "num_cols": num_cols})
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
            self.log_audit("detect_feature_set_failure", {"error": str(e)})
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
                self.log_audit("impute_timestamp_failure", {"error": "missing_timestamp_column"})
                return data

            data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
            if data["timestamp"].isna().any():
                last_valid = (
                    data["timestamp"].dropna().iloc[-1]
                    if not data["timestamp"].dropna().empty
                    else pd.Timestamp.now()
                )
                self.alert_manager.send_alert(
                    f"Valeurs invalides dans 'timestamp', imputées avec {last_valid}",
                    priority=2,
                )
                data["timestamp"] = data["timestamp"].fillna(last_valid)
            self.log_performance(
                "impute_timestamp", time.time() - start_time, success=True
            )
            self.log_audit("impute_timestamp", {"num_imputed": data["timestamp"].isna().sum()})
            return data
        except Exception as e:
            error_msg = (
                f"Erreur imputation timestamp: {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance(
                "impute_timestamp",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            self.log_audit("impute_timestamp_failure", {"error": str(e)})
            return data

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
                "reward",
                "step",
                "regime",
                "timestamp",
                "iv_skew",
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

            data = self.impute_timestamp(data)

            logger.debug(f"Données validées pour ensemble {feature_set}")
            self.alert_manager.send_alert(
                f"Données validées pour ensemble {feature_set}", priority=1
            )
            self.save_snapshot("validation", {"num_features": len(data.columns)})
            self.log_audit("validate_data", {"feature_set": feature_set, "num_cols": len(data.columns)})
        except Exception as e:
            error_msg = f"Erreur validation données: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=5)
            logger.error(error_msg)
            self.log_performance(
                "validate_data", time.time() - start_time, success=False, error=str(e)
            )
            self.log_audit("validate_data_failure", {"error": str(e)})
            raise

    def _load_trades_core(self, sources: List[str], period: str = None, max_trades: int = None, advanced: bool = False) -> Optional[pd.DataFrame]:
        """Méthode factorisée pour charger et valider les trades."""
        start_time = time.time()
        try:
            logger.info(f"Tentative de chargement: {sources}")
            self.alert_manager.send_alert(
                f"Chargement fichier trades: {sources}", priority=2
            )
            if advanced:
                df = self.trade_loader.load_trades(sources, period, max_trades)
                df = self.validate_data_with_pandera(df)
            else:
                if len(sources) != 1 or not sources[0].endswith(".csv"):
                    error_msg = f"load_trades attend un fichier CSV unique, reçu: {sources}"
                    raise ValueError(error_msg)
                file_path = sources[0]
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
                "load_trades_core",
                time.time() - start_time,
                success=True,
                num_trades=len(df),
            )
            logger.info(f"Chargement réussi: {sources} ({len(df)} lignes)")
            self.alert_manager.send_alert(
                f"Chargement réussi: {sources} ({len(df)} lignes)", priority=1
            )
            self.log_audit("load_trades_core", {"sources": sources, "num_trades": len(df)})
            return df
        except Exception as e:
            error_msg = f"Erreur chargement trades: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=5)
            logger.error(error_msg)
            self.log_performance(
                "load_trades_core", time.time() - start_time, success=False, error=str(e)
            )
            self.log_audit("load_trades_core_failure", {"sources": sources, "error": str(e)})
            return None

    def load_trades(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Charge les données de trades avec vérifications robustes.

        Args:
            file_path (str): Chemin du fichier CSV des trades.

        Returns:
            Optional[pd.DataFrame]: Données chargées ou None si erreur.
        """
        return self._load_trades_core([file_path])

    def store_metadata_in_db(self, stats: Dict, df: pd.DataFrame, regime: str, session: str) -> None:
        """Stocke les méta-données dans market_memory.db (Proposition 2, Étape 1)."""
        try:
            start_time = time.time()
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "max_drawdown": stats.get("max_drawdown", 0),
                    "iv_skew": stats.get("iv_skew", 0),
                    "entry_freq": stats.get("entry_freq", 0),
                    "period": stats.get("period", "N/A"),
                    "sharpe_ratio": stats.get("sharpe_ratio", 0),
                    "win_rate": stats.get("win_rate", 0),
                },
                "hyperparameters": self.config.get("model_params", {}),
                "performance": stats.get("total_return", 0),
                "regime": regime,
                "session": session,
            }
            conn = sqlite3.connect(MARKET_MEMORY_DB)
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS meta_runs (
                    run_id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    metrics TEXT,
                    hyperparameters TEXT,
                    performance REAL,
                    regime TEXT,
                    session TEXT
                )
                """
            )
            cursor.execute(
                """
                INSERT INTO meta_runs (timestamp, metrics, hyperparameters, performance, regime, session)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata["timestamp"],
                    json.dumps(metadata["metrics"]),
                    json.dumps(metadata["hyperparameters"]),
                    metadata["performance"],
                    metadata["regime"],
                    metadata["session"],
                ),
            )
            conn.commit()
            conn.close()
            logger.info("Méta-données stockées dans market_memory.db")
            self.alert_dispatcher.send_alert("Méta-données stockées dans market_memory.db", priority=1)
            self.log_performance("store_metadata_in_db", time.time() - start_time, success=True)
            self.log_audit("store_metadata_in_db", {"regime": regime, "session": session})
        except Exception as e:
            error_msg = f"Erreur stockage méta-données: {str(e)}\n{traceback.format_exc()}"
            self.alert_dispatcher.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance("store_metadata_in_db", time.time() - start_time, success=False, error=str(e))
            self.log_audit("store_metadata_in_db_failure", {"error": str(e)})

    def generate_narrative_report(self, stats: Dict, df: pd.DataFrame, symbol: str) -> str:
        """Génère un narratif via GPT-4o-mini pour les rapports (Proposition 1)."""
        try:
            start_time = time.time()
            if not all(k in stats for k in ["max_drawdown", "iv_skew", "entry_freq", "period"]):
                error_msg = "Métriques manquantes pour le narratif: max_drawdown, iv_skew, entry_freq, period"
                self.alert_dispatcher.send_alert(error_msg, priority4200
            logger.warning(error_msg)
            return ""
        narrative = self.report_generator.generate_narrative(stats)
        logger.info(f"Narratif généré pour {symbol}")
        self.alert_dispatcher.send_alert(f"Narratif généré pour {symbol}: {narrative[:50]}...", priority=2)
        self.log_performance("generate_narrative_report", time.time() - start_time, success=True)
        self.log_audit("generate_narrative_report", {"symbol": symbol, "narrative_length": len(narrative)})
        return narrative
    except Exception as e:
        error_msg = f"Erreur génération narratif: {str(e)}\n{traceback.format_exc()}"
        self.alert_dispatcher.send_alert(error_msg, priority=5)
        logger.error(error_msg)
        self.log_performance("generate_narrative_report", time.time() - start_time, success=False, error=str(e))
        self.log_audit("generate_narrative_report_failure", {"symbol": symbol, "error": str(e)})
        return ""

    def compute_metrics(
        self, df: pd.DataFrame, neural_pipeline: Optional[NeuralPipeline] = None
    ) -> Dict[str, float]:
        """
        Calcule les métriques de performance globales et par régime, incluant neural_regime.

        Args:
            df (pd.DataFrame): Données des trades.
            neural_pipeline (NeuralPipeline, optional): Pipeline pour prédire neural_regime.

        Returns:
            Dict[str, float]: Métriques calculées.
        """
        start_time = time.time()
        try:
            logger.info("Calcul des métriques de performance")
            self.alert_manager.send_alert(
                "Calcul des métriques de performance", priority=2
            )

            stats = self.metrics_computer.compute_metrics(df)
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
            avg_win = rewards[rewards > 0].mean() if (rewards > 0).sum() > 0 else 0
            avg_loss = rewards[rewards < 0].mean() if (rewards < 0).sum() > 0 else 0
            risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

            stats.update({
                "total_trades": total_trades,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_dd,
                "total_return": equity[-1],
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "risk_reward_ratio": risk_reward,
            })

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

            if (
                self.use_neural_pipeline
                and neural_pipeline
                and all(
                    col in df.columns
                    for col in [
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
                )
            ):
                cache_key = hashlib.sha256(df.to_json().encode()).hexdigest()
                if cache_key in self.neural_cache:
                    neural_result = self.neural_cache[cache_key]
                else:
                    raw_data = df[
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
                    options_data = df[
                        ["timestamp", "gex", "oi_peak_call_near", "gamma_wall_call"]
                    ]
                    orderflow_data = df[
                        ["timestamp", "bid_size_level_1", "ask_size_level_1"]
                    ]

                    def run_pipeline():
                        return neural_pipeline.run(
                            raw_data, options_data, orderflow_data
                        )

                    neural_result = self.with_retries(run_pipeline)
                    if neural_result is None:
                        raise ValueError("Échec de l’exécution de neural_pipeline")
                    self.neural_cache[cache_key] = neural_result
                    while len(self.neural_cache) > MAX_CACHE_SIZE:
                        self.neural_cache.popitem(last=False)

                df["neural_regime"] = neural_result["regime"]
                df["predicted_volatility"] = neural_result["volatility"]

            if "neural_regime" in df.columns:
                neural_regimes = df["neural_regime"].unique()
                stats_by_neural_regime = {}
                for neural_regime in neural_regimes:
                    regime_rewards = (
                        df[df["neural_regime"] == neural_regime]["reward"]
                        .fillna(0)
                        .values
                    )
                    stats_by_neural_regime[neural_regime] = {
                        "total_trades": len(regime_rewards),
                        "win_rate": (
                            (regime_rewards > 0).mean() * 100
                            if len(regime_rewards) > 0
                            else 0
                        ),
                        "total_return": regime_rewards.sum(),
                    }
                stats["by_neural_regime"] = stats_by_neural_regime

            stats["by_regime"] = stats_by_regime

            for metric, threshold in self.performance_thresholds.items():
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

            def store_in_memory():
                store_pattern(
                    df,
                    action=1.0 if stats["total_return"] > 0 else -1.0,
                    reward=stats["total_return"],
                    neural_regime=(
                        df["neural_regime"].iloc[-1]
                        if "neural_regime" in df.columns
                        else 0
                    ),
                    confidence=(
                        max(stats["win_rate"] / 100, stats["sharpe_ratio"] / 2)
                        if stats["sharpe_ratio"] > 0
                        else 0.5
                    ),
                    metadata={
                        "event": "metrics_analysis",
                        "symbol": df.get("symbol", "unknown"),
                        "stats": stats,
                    },
                )

            self.with_retries(store_in_memory)

            regime = df["regime"].iloc[-1] if "regime" in df.columns else "unknown"
            session = df["session"].iloc[-1] if "session" in df.columns else "unknown"
            self.store_metadata_in_db(stats, df, regime, session)

            self.log_performance(
                "compute_metrics",
                time.time() - start_time,
                success=True,
                num_trades=total_trades,
            )
            logger.info(f"Métriques calculées: {stats}")
            self.alert_manager.send_alert(f"Métriques calculées: {stats}", priority=1)
            self.log_audit("compute_metrics", {"num_trades": total_trades, "metrics": list(stats.keys())})
            return stats
        except Exception as e:
            error_msg = f"Erreur compute_metrics: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=5)
            logger.error(error_msg)
            self.log_performance(
                "compute_metrics", time.time() - start_time, success=False, error=str(e)
            )
            self.log_audit("compute_metrics_failure", {"error": str(e)})
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "total_return": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "risk_reward_ratio": 0.0,
                "iv_skew": 0.0,
                "entry_freq": 0.0,
                "period": "N/A",
            }

    def plot_results(
        self,
        df: pd.DataFrame,
        symbol: str,
        output_dir: str = FIGURES_DIR,
        shap_values: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Génère des graphiques : equity curve, profit cumulé, distribution, et profit vs features SHAP.

        Args:
            df (pd.DataFrame): Données des trades.
            symbol (str): Symbole du marché (ex. ES, NQ).
            output_dir (str): Dossier pour les graphiques.
            shap_values (Optional[pd.DataFrame]): Valeurs SHAP pour les features.
        """
        start_time = time.time()
        try:
            logger.info(f"Génération des graphiques pour {symbol.upper()}")
            self.alert_dispatcher.send_alert(
                f"Génération des graphiques pour {symbol.upper()}", priority=2
            )
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            equity = np.cumsum(df["reward"].fillna(0))
            plt.figure(figsize=(12, 5))
            plt.plot(df["timestamp"], equity, label="Equity Curve", color="blue")
            for regime in df["regime"].unique():
                regime_df = df[df["regime"] == regime]
                plt.scatter(
                    regime_df["timestamp"],
                    np.cumsum(regime_df["reward"].fillna(0)),
                    label=f"Regime: {regime}",
                    alpha=0.5,
                )
            if "neural_regime" in df.columns:
                for neural_regime in df["neural_regime"].unique():
                    regime_df = df[df["neural_regime"] == neural_regime]
                    plt.plot(
                        regime_df["timestamp"],
                        np.cumsum(regime_df["reward"].fillna(0)),
                        label=f"Neural: {neural_regime}",
                        linestyle="--",
                        alpha=0.7,
                    )
            plt.title(f"Equity Curve - {symbol.upper()}")
            plt.xlabel("Timestamp")
            plt.ylabel("Equity")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()
            equity_path = output_dir / f"{symbol}_equity_{timestamp}.png"
            plt.savefig(equity_path)
            plt.close()
            logger.info(f"Equity curve sauvegardée: {equity_path}")
            self.alert_dispatcher.send_alert(
                f"Equity curve sauvegardée: {equity_path}", priority=2
            )

            plt.figure(figsize=(12, 5))
            plt.plot(df["timestamp"], equity, label="Profit Cumulé", color="orange")
            plt.title(f"Profit Cumulé - {symbol.upper()}")
            plt.xlabel("Timestamp")
            plt.ylabel("Profit Cumulé")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()
            profit_path = output_dir / f"{symbol}_cumulative_profit_{timestamp}.png"
            plt.savefig(profit_path)
            plt.close()
            logger.info(f"Profit cumulé sauvegardé: {profit_path}")
            self.alert_dispatcher.send_alert(
                f"Profit cumulé sauvegardé: {profit_path}", priority=2
            )

            plt.figure(figsize=(10, 6))
            plt.hist(
                df["reward"].fillna(0),
                bins=30,
                color="green",
                edgecolor="black",
                alpha=0.7,
            )
            plt.title(f"Distribution des Rewards - {symbol.upper()}")
            plt.xlabel("Reward par trade")
            plt.ylabel("Nombre de trades")
            plt.grid(True, linestyle="--", alpha=0.7)
            dist_path = output_dir / f"{symbol}_reward_distribution_{timestamp}.png"
            plt.savefig(dist_path)
            plt.close()
            logger.info(f"Distribution sauvegardée: {dist_path}")
            self.alert_dispatcher.send_alert(
                f"Distribution sauvegardée: {dist_path}", priority=2
            )

            if shap_values is not None and not shap_values.empty:
                for feature in shap_values.columns[:SHAP_FEATURE_LIMIT]:
                    plt.figure(figsize=(10, 6))
                    plt.scatter(
                        df["reward"], shap_values[feature], alpha=0.6, color="purple"
                    )
                    plt.title(f"Reward vs SHAP {feature} - {symbol.upper()}")
                    plt.xlabel("Reward")
                    plt.ylabel(f"SHAP Value ({feature})")
                    plt.grid(True, linestyle="--", alpha=0.7)
                    shap_path = (
                        output_dir / f"{symbol}_reward_vs_{feature}_{timestamp}.png"
                    )
                    plt.savefig(shap_path)
                    plt.close()
                    logger.info(f"Graphique SHAP sauvegardé: {shap_path}")
                    self.alert_dispatcher.send_alert(
                        f"Graphique SHAP sauvegardé: {shap_path}", priority=2
                    )

            self.log_performance(
                "plot_results",
                time.time() - start_time,
                success=True,
                num_trades=len(df),
            )
            self.log_audit("plot_results", {"symbol": symbol, "num_plots": 3 + (len(shap_values.columns[:SHAP_FEATURE_LIMIT]) if shap_values is not None else 0)})
        except Exception as e:
            error_msg = f"Erreur plot_results: {str(e)}\n{traceback.format_exc()}"
            self.alert_dispatcher.send_alert(error_msg, priority=5)
            logger.error(error_msg)
            self.log_performance(
                "plot_results", time.time() - start_time, success=False, error=str(e)
            )
            self.log_audit("plot_results_failure", {"symbol": symbol, "error": str(e)})

    def save_summary(
        self, stats: Dict[str, float], symbol: str, output_dir: str = TRADES_DIR
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
            self.alert_dispatcher.send_alert(
                f"Sauvegarde du résumé pour {symbol.upper()}", priority=2
            )

            if (
                datetime.now() - self.last_checkpoint_time
            ).total_seconds() >= 300:
                self.save_checkpoint(incremental=True)
                self.last_checkpoint_time = datetime.now()
            if (
                datetime.now() - self.last_distributed_checkpoint_time
            ).total_seconds() >= 900:
                self.save_checkpoint(incremental=False, distributed=True)
                self.last_distributed_checkpoint_time = datetime.now()

            summary_entry = {
                k: v
                for k, v in stats.items()
                if k not in ["by_regime", "by_neural_regime", "narrative"]
            }
            summary_entry["timestamp"] = datetime.now().isoformat()
            summary_entry["symbol"] = symbol
            self.summary_buffer.append(summary_entry)

            if len(self.summary_buffer) >= self.config.get("logging", {}).get(
                "buffer_size", 100
            ):
                df_stats = pd.DataFrame(self.summary_buffer)
                output_path = (
                    Path(output_dir)
                    / f"trade_summary_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )
                os.makedirs(output_path.parent, exist_ok=True)

                def save_csv():
                    df_stats.to_csv(output_path, index=False, encoding="utf-8")

                self.with_retries(save_csv)
                self.summary_buffer = []
                logger.info(f"Résumé sauvegardé: {output_path}")
                self.alert_dispatcher.send_alert(
                    f"Résumé sauvegardé: {output_path}", priority=2
                )

            self.save_snapshot(f"summary_{symbol}", stats)
            self.update_dashboard(stats, symbol)

            self.log_performance("save_summary", time.time() - start_time, success=True)
            self.log_audit("save_summary", {"symbol": symbol, "output_path": str(output_path)})
        except Exception as e:
            error_msg = f"Erreur save_summary: {str(e)}\n{traceback.format_exc()}"
            self.alert_dispatcher.send_alert(error_msg, priority=5)
            logger.error(error_msg)
            self.log_performance(
                "save_summary", time.time() - start_time, success=False, error=str(e)
            )
            self.log_audit("save_summary_failure", {"symbol": symbol, "error": str(e)})

    def save_analysis_snapshot(
        self, step: int, stats: Dict[str, float], symbol: str
    ) -> None:
        """
        Sauvegarde un instantané des métriques d’analyse.

        Args:
            step (int): Étape de l’analyse.
            stats (Dict[str, float]): Métriques calculées.
            symbol (str): Symbole du marché.
        """
        start_time = time.time()
        try:
            snapshot = {
                "step": step,
                "timestamp": str(datetime.now()),
                "symbol": symbol,
                "stats": stats,
                "summary_buffer_size": len(self.summary_buffer),
            }
            self.save_snapshot(f"analyse_step_{step:04d}", snapshot)
            logger.info(f"Snapshot analyse step {step} sauvegardé")
            self.alert_dispatcher.send_alert(
                f"Snapshot analyse step {step} sauvegardé", priority=1
            )
            self.log_performance(
                "save_analysis_snapshot",
                time.time() - start_time,
                success=True,
                step=step,
            )
            self.log_audit("save_analysis_snapshot", {"step": step, "symbol": symbol})
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot analyse: {str(e)}\n{traceback.format_exc()}"
            self.alert_dispatcher.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance(
                "save_analysis_snapshot",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            self.log_audit("save_analysis_snapshot_failure", {"step": step, "error": str(e)})

    def update_dashboard(self, stats: Dict[str, float], symbol: str) -> None:
        """
        Met à jour un fichier JSON pour partager l’état de l’analyse avec mia_dashboard.py.

        Args:
            stats (Dict[str, float]): Métriques calculées.
            symbol (str): Symbole du marché.
        """
        start_time = time.time()
        try:
            status = {
                "timestamp": str(datetime.now()),
                "symbol": symbol,
                "num_trades": stats.get("total_trades", 0),
                "win_rate": stats.get("win_rate", 0.0),
                "sharpe_ratio": stats.get("sharpe_ratio", 0.0),
                "max_drawdown": stats.get("max_drawdown", 0.0),
                "total_return": stats.get("total_return", 0.0),
                "iv_skew": stats.get("iv_skew", 0.0),
                "entry_freq": stats.get("entry_freq", 0.0),
                "period": stats.get("period", "N/A"),
                "recent_errors": len(
                    [log for log in self.log_buffer if not log["success"]]
                ),
                " Whistle: "average_latency": (
                    sum(log["latency"] for log in self.log_buffer)
                    / len(self.log_buffer)
                    if self.log_buffer
                    else 0
                ),
                "by_regime": stats.get("by_regime", {}),
                "by_neural_regime": stats.get("by_neural_regime", {}),
            }

            def write_status():
                with open(DASHBOARD_PATH, "w", encoding="utf-8") as f:
                    json.dump(status, f, indent=4)

            self.with_retries(write_status)
            logger.info("Mise à jour dashboard analyse effectuée")
            self.alert_dispatcher.send_alert(
                "Mise à jour dashboard analyse effectuée", priority=1
            )
            self.log_performance(
                "update_dashboard", time.time() - start_time, success=True
            )
            self.log_audit("update_dashboard", {"symbol": symbol})
        except Exception as e:
            error_msg = f"Erreur mise à jour dashboard analyse: {str(e)}\n{traceback.format_exc()}"
            self.alert_dispatcher.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance(
                "update_dashboard",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            self.log_audit("update_dashboard_failure", {"symbol": symbol, "error": str(e)})

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
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            if memory_usage > 1024:
                error_msg = (
                    f"ALERTE: Utilisation mémoire élevée ({memory_usage:.2f} MB)"
                )
                self.alert_dispatcher.send_alert(error_msg, priority=5)
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
            os.makedirs(CSV_LOG_PATH.parent, exist_ok=True)
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
            self.log_audit("log_performance", {"operation": operation, "success": success})
        except Exception as e:
            error_msg = (
                f"Erreur journalisation performance: {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_dispatcher.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_audit("log_performance_failure", {"operation": operation, "error": str(e)})

    def analyse_results(
        self, trades: pd.DataFrame, symbol: str = "ES"
    ) -> Dict[str, Any]:
        """
        Analyse les résultats des trades avec intégration de SHAP et visualisations.

        Args:
            trades (pd.DataFrame): Données des trades.
            symbol (str): Symbole du marché (ex. ES, NQ).

        Returns:
            Dict[str, Any]: Résultats incluant le profit total et les métriques SHAP.
        """
        start_time = time.time()
        try:
            self.validate_data(trades)
            logger.info(f"Analyse des résultats pour {symbol.upper()}")
            self.alert_dispatcher.send_alert(
                f"Analyse des résultats pour {symbol.upper()}", priority=2
            )

            shap_values = None
            if "reward" in trades.columns:

                def run_shap():
                    return calculate_shap(
                        trades, target="reward", max_features=SHAP_FEATURE_LIMIT
                    )

                shap_values = self.with_retries(run_shap)
                if shap_values is None:
                    error_msg = "Échec calcul SHAP, poursuite sans SHAP"
                    self.alert_dispatcher.send_alert(error_msg, priority=3)
                    logger.warning(error_msg)
                else:
                    self.save_snapshot(
                        "shap_analysis", {"shap_features": list(shap_values.columns)}
                    )

            neural_pipeline = None
            if self.use_neural_pipeline:
                neural_pipeline = NeuralPipeline(
                    window_size=50,
                    base_features=150,
                    config_path="config/model_params.yaml",
                )
                try:

                    def load_models():
                        neural_pipeline.load_models()

                    self.with_retries(load_models)
                    logger.info("Modèles neural_pipeline chargés")
                    self.alert_dispatcher.send_alert(
                        "Modèles neural_pipeline chargés", priority=2
                    )
                except Exception as e:
                    error_msg = f"Erreur chargement neural_pipeline: {e}, analyse sans neural_regime"
                    self.alert_dispatcher.send_alert(error_msg, priority=3)
                    logger.warning(error_msg)
                    neural_pipeline = None

            stats = self.compute_metrics(trades, neural_pipeline)
            self.plot_results(
                trades, symbol, output_dir=FIGURES_DIR, shap_values=shap_values
            )
            self.save_summary(stats, symbol)

            result = {
                "profit": stats.get("total_return", 0.0),
                "shap_metrics": (
                    shap_values.to_dict() if shap_values is not None else {}
                ),
            }

            self.log_performance(
                "analyse_results",
                time.time() - start_time,
                success=True,
                num_trades=len(trades),
            )
            logger.info(f"Analyse terminée. CPU: {psutil.cpu_percent()}%")
            self.alert_dispatcher.send_alert(
                f"Analyse terminée pour {symbol.upper()}", priority=2
            )
            self.log_audit("analyse_results", {"symbol": symbol, "num_trades": len(trades)})
            return result
        except Exception as e:
            error_msg = f"Erreur analyse_results: {str(e)}\n{traceback.format_exc()}"
            self.alert_dispatcher.send_alert(error_msg, priority=5)
            logger.error(error_msg)
            self.log_performance(
                "analyse_results", time.time() - start_time, success=False, error=str(e)
            )
            self.log_audit("analyse_results_failure", {"symbol": symbol, "error": str(e)})
            return {"profit": 0.0, "shap_metrics": {}}

    def setup_scheduler(self):
        """Configure le planificateur pour les rapports périodiques."""
        try:
            start_time = time.time()
            self.scheduler = BackgroundScheduler()
            self.scheduler.add_job(self.generate_daily_report, 'cron', hour=16, minute=5)
            self.scheduler.add_job(self.generate_weekly_report, 'cron', day_of_week='fri', hour=16, minute=0)
            self.scheduler.add_job(self.generate_monthly_report, 'cron', day=1, month='*', hour=0, minute=0)
            self.scheduler.start()
            logger.info("Planificateur de rapports périodiques configuré")
            self.alert_dispatcher.send_alert("Planificateur de rapports périodiques configuré", priority=2)
            self.log_performance("setup_scheduler", time.time() - start_time, success=True)
            self.log_audit("setup_scheduler", {"jobs": ["daily", "weekly", "monthly"]})
        except Exception as e:
            error_msg = f"Erreur configuration planificateur: {str(e)}\n{traceback.format_exc()}"
            self.alert_dispatcher.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance("setup_scheduler", 0, success=False, error=str(e))
            self.log_audit("setup_scheduler_failure", {"error": str(e)})

    def assign_session(self, timestamp: str) -> str:
        """Assigne une session de trading basée sur l’heure du timestamp."""
        try:
            start_time = time.time()
            hour = pd.to_datetime(timestamp).hour
            if 2 <= hour < 11:
                session = "London"
            elif 13 <= hour < 20:
                session = "New York"
            else:
                session = "Asian"
            self.log_performance("assign_session", time.time() - start_time, success=True)
            self.log_audit("assign_session", {"timestamp": timestamp, "session": session})
            return session
        except Exception as e:
            error_msg = f"Erreur assignation session: {str(e)}\n{traceback.format_exc()}"
            self.alert_dispatcher.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance("assign_session", 0, success=False, error=str(e))
            self.log_audit("assign_session_failure", {"timestamp": timestamp, "error": str(e)})
            return "Unknown"

    def validate_data_with_pandera(self, data: pd.DataFrame) -> pd.DataFrame:
        """Valide les données avec Pandera pour inclure les nouvelles colonnes."""
        start_time = time.time()
        try:
            df = self.schema_validator.validate(data)
            logger.info("Données validées avec Pandera")
            self.alert_dispatcher.send_alert("Données validées avec Pandera", priority=1)
            self.log_performance("validate_data_with_pandera", time.time() - start_time, success=True)
            self.log_audit("validate_data_with_pandera", {"num_cols": len(df.columns)})
            return df
        except Exception as e:
            error_msg = f"Erreur validation Pandera: {str(e)}\n{traceback.format_exc()}"
            self.alert_dispatcher.send_alert(error_msg, priority=5)
            logger.error(error_msg)
            self.log_performance("validate_data_with_pandera", time.time() - start_time, success=False, error=str(e))
            self.log_audit("validate_data_with_pandera_failure", {"error": str(e)})
            raise

    def load_trades_advanced(self, sources: List[str], period: str = None, max_trades: int = None) -> pd.DataFrame:
        """Charge les trades depuis IQFeed, SQLite, ou CSV avec gestion des sessions."""
        return self._load_trades_core(sources, period, max_trades, advanced=True)

    def generate_daily_report(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """Génère le rapport quotidien à 16h05 UTC."""
        start_time = time.time()
        try:
            df = df or self.load_trades_advanced(self.config["trade_sources"], period="00:00-23:59")
            if df is None or df.empty:
                error_msg = "Aucune donnée pour le rapport quotidien"
                self.alert_dispatcher.send_alert(error_msg, priority=5)
                logger.error(error_msg)
                return {}
            stats = self.metrics_computer.compute_metrics(df)
            shap_values = self.shap_analyzer.calculate_shap(df)
            suggestions = self.suggestion_engine.generate_suggestions(stats, df)
            narrative = self.generate_narrative_report(stats, df, symbol="ES")
            stats["narrative"] = narrative
            report = self.report_generator.save_summary(stats, shap_values, suggestions, symbol="ES")
            self.alert_dispatcher.send_alert(report["text"], priority=4)
            export_report(report["markdown"], report["figures"], BASE_DIR / f"data/figures/trading/daily_{datetime.now().strftime('%Y%m%d')}.pdf")
            regime = df["regime"].iloc[-1] if "regime" in df.columns else "unknown"
            session = df["session"].iloc[-1] if "session" in df.columns else "unknown"
            self.store_metadata_in_db(stats, df, regime, session)
            logger.info("Rapport quotidien généré")
            self.log_performance("generate_daily_report", time.time() - start_time, success=True)
            self.log_audit("generate_daily_report", {"symbol": "ES", "num_trades": len(df)})
            return {"stats": stats, "markdown": report["markdown"], "figures": report["figures"]}
        except Exception as e:
            error_msg = f"Erreur génération rapport quotidien: {str(e)}\n{traceback.format_exc()}"
            self.alert_dispatcher.send_alert(error_msg, priority=5)
            logger.error(error_msg)
            self.log_performance("generate_daily_report", time.time() - start_time, success=False, error=str(e))
            self.log_audit("generate_daily_report_failure", {"error": str(e)})
            return {}

    def generate_weekly_report(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """Génère le rapport hebdomadaire le vendredi à 16h00 UTC."""
        start_time = time.time()
        try:
            df = df or self.load_trades_advanced(self.config["trade_sources"], period="monday:00:00-friday:16:00")
            if df is None or df.empty:
                error_msg = "Aucune donnée pour le rapport hebdomadaire"
                self.alert_dispatcher.send_alert(error_msg, priority=5)
                logger.error(error_msg)
                return {}
            stats = self.metrics_computer.compute_metrics(df, rolling=[7, 30])
            shap_values = self.shap_analyzer.calculate_shap(df)
            suggestions = self.suggestion_engine.generate_suggestions(stats, df)
            narrative = self.generate_narrative_report(stats, df, symbol="ES")
            stats["narrative"] = narrative
            profile = ProfileReport(df, title="Weekly Data Quality")
            profile.to_file(BASE_DIR / f"data/reports/weekly_quality_{datetime.now().strftime('%Y%m%d')}.html")
            report = self.report_generator.save_summary(stats, shap_values, suggestions, symbol="ES")
            self.alert_dispatcher.send_alert(report["text"], priority=4)
            export_report(report["markdown"], report["figures"], BASE_DIR / f"data/figures/trading/weekly_{datetime.now().strftime('%Y%m%d')}.pdf")
            regime = df["regime"].iloc[-1] if "regime" in df.columns else "unknown"
            session = df["session"].iloc[-1] if "session" in df.columns else "unknown"
            self.store_metadata_in_db(stats, df, regime, session)
            logger.info("Rapport hebdomadaire généré")
            self.log_performance("generate_weekly_report", time.time() - start_time, success=True)
            self.log_audit("generate_weekly_report", {"symbol": "ES", "num_trades": len(df)})
            return {"stats": stats, "markdown": report["markdown"], "figures": report["figures"]}
        except Exception as e:
            error_msg = f"Erreur génération rapport hebdomadaire: {str(e)}\n{traceback.format_exc()}"
            self.alert_dispatcher.send_alert(error_msg, priority=5)
            logger.error(error_msg)
            self.log_performance("generate_weekly_report", time.time() - start_time, success=False, error=str(e))
            self.log_audit("generate_weekly_report_failure", {"error": str(e)})
            return {}

    def generate_monthly_report(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """Génère le rapport mensuel le dernier jour du mois à 00h00 UTC."""
        start_time = time.time()
        try:
            df = df or self.load_trades_advanced(self.config["trade_sources"], period="month")
            if df is None or df.empty:
                error_msg = "Aucune donnée pour le rapport mensuel"
                self.alert_dispatcher.send_alert(error_msg, priority=5)
                logger.error(error_msg)
                return {}
            stats = self.metrics_computer.compute_metrics(df, rolling=[7, 30, 90])
            shap_values = self.shap_analyzer.calculate_shap(df)
            suggestions = self.suggestion_engine.generate_suggestions(stats, df)
            narrative = self.generate_narrative_report(stats, df, symbol="ES")
            stats["narrative"] = narrative
            profile = ProfileReport(df, title="Monthly Data Quality")
            profile.to_file(BASE_DIR / f"data/reports/monthly_quality_{datetime.now().strftime('%Y%m%d')}.html")
            report = self.report_generator.save_summary(stats, shap_values, suggestions, symbol="ES")
            self.alert_dispatcher.send_alert(report["text"], priority=4)
            export_report(report["markdown"], report["figures"], BASE_DIR / f"data/figures/trading/monthly_{datetime.now().strftime('%Y%m%d')}.pdf")
            regime = df["regime"].iloc[-1] if "regime" in df.columns else "unknown"
            session = df["session"].iloc[-1] if "session" in df.columns else "unknown"
            self.store_metadata_in_db(stats, df, regime, session)
            logger.info("Rapport mensuel généré")
            self.log_performance("generate_monthly_report", time.time() - start_time, success=True)
            self.log_audit("generate_monthly_report", {"symbol": "ES", "num_trades": len(df)})
            return {"stats": stats, "markdown": report["markdown"], "figures": report["figures"]}
        except Exception as e:
            error_msg = f"Erreur génération rapport mensuel: {str(e)}\n{traceback.format_exc()}"
            self.alert_dispatcher.send_alert(error_msg, priority=5)
            logger.error(error_msg)
            self.log_performance("generate_monthly_report", time.time() - start_time, success=False, error=str(e))
            self.log_audit("generate_monthly_report_failure", {"error": str(e)})
            return {}

    def generate_trade_based_report(self, df: pd.DataFrame, threshold: int) -> Dict:
        """Génère un rapport basé sur le nombre de trades (500 ou 1000)."""
        start_time = time.time()
        try:
            df_subset = df.tail(threshold)
            stats = self.metrics_computer.compute_metrics(df_subset)
            shap_values = self.shap_analyzer.calculate_shap(df_subset) if threshold == 1000 else None
            suggestions = self.suggestion_engine.generate_suggestions(stats, df_subset)
            narrative = self.generate_narrative_report(stats, df_subset, symbol="ES")
            stats["narrative"] = narrative
            report = self.report_generator.save_summary(stats, shap_values, suggestions, symbol="ES")
            self.alert_dispatcher.send_alert(report["text"], priority=4 if threshold == 1000 else 3)
            if threshold == 1000:
                export_report(report["markdown"], report["figures"], BASE_DIR / f"data/figures/trading/trades_{threshold}_{datetime.now().strftime('%Y%m%d')}.pdf")
            regime = df_subset["regime"].iloc[-1] if "regime" in df_subset.columns else "unknown"
            session = df_subset["session"].iloc[-1] if "session" in df_subset.columns else "unknown"
            self.store_metadata_in_db(stats, df_subset, regime, session)
            logger.info(f"Rapport basé sur {threshold} trades généré")
            self.log_performance("generate_trade_based_report", time.time() - start_time, success=True)
            self.log_audit("generate_trade_based_report", {"threshold": threshold, "num_trades": len(df_subset)})
            return {"stats": stats, "markdown": report["markdown"], "figures": report["figures"]}
        except Exception as e:
            error_msg = f"Erreur génération rapport trades ({threshold}): {str(e)}\n{traceback.format_exc()}"
            self.alert_dispatcher.send_alert(error_msg, priority=5)
            logger.error(error_msg)
            self.log_performance("generate_trade_based_report", time.time() - start_time, success=False, error=str(e))
            self.log_audit("generate_trade_based_report_failure", {"threshold": threshold, "error": str(e)})
            return {}

    def save_contextual_checkpoint(self, data: Optional[pd.DataFrame], regime: str):
        """Sauvegarde l’état de l’analyseur avec tagging par régime."""
        try:
            start_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint = {
                "timestamp": timestamp,
                "regime": regime,
                "log_buffer": self.log_buffer[-100:],
                "summary_buffer": self.summary_buffer[-100:],
                "neural_cache": {k: True for k in self.neural_cache},
            }
            checkpoint_path = CHECKPOINT_DIR / f"checkpoint_{regime}_{timestamp}.json.gz"
            os.makedirs(checkpoint_path.parent, exist_ok=True)
            with gzip.open(checkpoint_path, "wt", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=4)
            if data is not None:
                data.to_csv(checkpoint_path.with_suffix(".csv"), index=False)

            checkpoints = sorted(CHECKPOINT_DIR.glob(f"checkpoint_{regime}_*.json.gz"))
            if len(checkpoints) > 5:
                for old in checkpoints[:-5]:
                    old.unlink()
                    old.with_suffix(".csv").unlink(missing_ok=True)

            latency = time.time() - start_time
            self.alert_dispatcher.send_alert(f"Checkpoint contextuel ({regime}) sauvegardé: {checkpoint_path}", priority=1)
            logger.info(f"Checkpoint contextuel ({regime}) sauvegardé: {checkpoint_path}")
            self.log_performance("save_contextual_checkpoint", latency, success=True, regime=regime)
            self.log_audit("save_contextual_checkpoint", {"regime": regime, "path": str(checkpoint_path)})
        except Exception as e:
            error_msg = f"Erreur sauvegarde checkpoint contextuel ({regime}): {str(e)}\n{traceback.format_exc()}"
            self.alert_dispatcher.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance("save_contextual_checkpoint", 0, success=False, error=str(e))
            self.log_audit("save_contextual_checkpoint_failure", {"regime": regime, "error": str(e)})

    def log_audit(self, event: str, metadata: Dict):
        """Enregistre un événement dans le journal d’audit."""
        try:
            start_time = time.time()
            audit_entry = {
                "timestamp": datetime.now().isoformat(),
                "event": event,
                "metadata": metadata,
            }
            os.makedirs(AUDIT_LOG_PATH.parent, exist_ok=True)
            with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
                json.dump(audit_entry, f)
                f.write("\n")
            self.log_performance("log_audit", time.time() - start_time, success=True)
        except Exception as e:
            error_msg = f"Erreur journalisation audit ({event}): {str(e)}\n{traceback.format_exc()}"
            self.alert_dispatcher.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance("log_audit", 0, success=False, error=str(e))

    def main_advanced(self):
        """CLI avancée pour l’analyse des trades."""
        @click.command()
        @click.option("--from-date", default="2025-05-01", help="Date de début (YYYY-MM-DD)")
        @click.option("--to-date", default="2025-05-15", help="Date de fin (YYYY-MM-DD)")
        @click.option("--market", default="ES", help="Marché (ES, MNQ)")
        @click.option("--trades", type=int, help="Nombre de trades (ex. : 500, 1000)")
        @click.option("--output", default="pdf", help="Format de sortie (pdf, html, json)")
        def report(from_date, to_date, market, trades, output):
            logger.info(f"Début analyse avancée pour {market.upper()}")
            self.alert_dispatcher.send_alert(f"Début analyse avancée pour {market.upper()}", priority=2)
            sources = self.config["trade_sources"]
            df = self.load_trades_advanced(sources, period=f"{from_date}:{to_date}", max_trades=trades)
            if df is None:
                error_msg = "Analyse abandonnée en raison d’un échec de chargement"
                self.alert_dispatcher.send_alert(error_msg, priority=5)
                logger.warning(error_msg)
                return

            result = self.analyse_results(df, market)
            print(f"\nRésultats pour {market.upper()}")
            print("=" * 50)
            print(f"Profit Total: {result['profit']:.2f}")
            for key, value in result.items():
                if key not in ["profit", "shap_metrics"]:
                    print(f"{key.replace('_', ' ').title()}: {value:.2f}")
                elif key == "shap_metrics" and result["shap_metrics"]:
                    print("\nSHAP Metrics (Top Features):")
                    for feature, values in list(result["shap_metrics"].items())[:5]:
                        print(f"- {feature}: Mean SHAP Value = {np.mean(list(values.values())):.4f}")
            print("=" * 50)
            logger.info(f"Analyse avancée terminée pour {market.upper()}")
            self.alert_dispatcher.send_alert(f"Analyse avancée terminée pour {market.upper()}", priority=2)
            self.log_audit("main_advanced", {"market": market, "from_date": from_date, "to_date": to_date, "trades": trades})

        report()

def main():
    parser = argparse.ArgumentParser(description="Analyse des résultats de trading")
    parser.add_argument(
        "--input",
        type=str,
        default=TRADES_DIR / "trades_simulated.csv",
        help="Fichier CSV des trades",
    )
    parser.add_argument(
        "--symbol", type=str, default="ES", help="Symbole du marché (ex. ES, NQ)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=FIGURES_DIR,
        help="Dossier pour les graphiques",
    )
    parser.add_argument(
        "--summary-dir", type=str, default=TRADES_DIR, help="Dossier pour le résumé"
    )
    parser.add_argument(
        "--advanced", action="store_true", help="Utiliser la CLI avancée"
    )
    args = parser.parse_args()

    analyzer = TradeAnalyzer()
    logger.info(f"Début analyse pour {args.symbol.upper()}")
    analyzer.alert_dispatcher.send_alert(
        f"Début analyse pour {args.symbol.upper()}", priority=2
    )

    if args.advanced:
        analyzer.main_advanced()
        return

    df = analyzer.load_trades(args.input)
    if df is None:
        analyzer.alert_dispatcher.send_alert(
            "Analyse abandonnée en raison d’un échec de chargement", priority=5
        )
        logger.warning("Analyse abandonnée en raison d’un échec de chargement")
        return

    result = analyzer.analyse_results(df, args.symbol)

    print(f"\nRésultats pour {args.symbol.upper()}")
    print("=" * 50)
    print(f"Profit Total: {result['profit']:.2f}")
    for key, value in result.items():
        if key not in ["profit", "shap_metrics"]:
            print(f"{key.replace('_', ' ').title()}: {value:.2f}")
        elif key == "shap_metrics" and result["shap_metrics"]:
            print("\nSHAP Metrics (Top Features):")
            for feature, values in list(result["shap_metrics"].items())[:5]:
                print(
                    f"- {feature}: Mean SHAP Value = {np.mean(list(values.values())):.4f}"
                )
    print("=" * 50)

    logger.info(f"Analyse terminée pour {args.symbol.upper()}")
    analyzer.alert_dispatcher.send_alert(
        f"Analyse terminée pour {args.symbol.upper()}", priority=2
    )
    analyzer.log_audit("main", {"symbol": args.symbol, "num_trades": len(df)})

if __name__ == "__main__":
    main()
