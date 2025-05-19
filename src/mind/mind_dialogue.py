```python
# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/mind/mind_dialogue.py
# Gère les dialogues interactifs avec MIA pour MIA_IA_SYSTEM_v2_2025.
#
# Version : 2.1.4
# Date : 2025-05-15
#
# Rôle : Gère la reconnaissance vocale, les réponses conversationnelles via NLP (OpenAI), et la journalisation des dialogues
#        avec mémoire contextuelle (méthode 7, K-means 10 clusters dans market_memory.db). Utilise IQFeed comme source de données.
#
# Dépendances :
# - speech_recognition>=3.8.0,<4.0.0, pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.8,<6.0.0,
#   sklearn>=1.2.0,<2.0.0, openai>=1.0.0,<2.0.0, matplotlib>=3.8.0,<4.0.0, seaborn>=0.13.0,<1.0.0,
#   pyyaml>=6.0.0,<7.0.0, boto3>=1.26.0,<2.0.0, queue, threading, json, re, signal, gzip, sqlite3
# - src/mind/mind.py
# - src/envs/trading_env.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/db_setup.py
#
# Inputs :
# - config/mia_config.yaml
# - config/credentials.yaml
# - config/feature_sets.yaml
# - data/features/features_latest.csv
# - data/trades/trades_simulated.csv
#
# Outputs :
# - data/logs/mind/mind_dialogue.csv
# - data/mind/mind_dialogue_snapshots/*.json.gz
# - data/checkpoints/mind_dialogue_*.json.gz
# - data/mind/mind_dialogue_dashboard.json
# - data/logs/mind_dialogue_performance.csv
# - data/figures/mind_dialogue/
# - market_memory.db (table clusters)
#
# Notes :
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence) définies dans config/feature_sets.yaml.
# - Utilise IQFeed exclusivement via data_provider.py pour les données d’entrée.
# - Supprime toute référence à 320 features, 81 features, et obs_t.
# - Implémente retries (max 3, délai 2^attempt secondes) pour les opérations critiques.
# - Logs psutil dans data/logs/mind_dialogue_performance.csv avec métriques détaillées.
# - Alertes via alert_manager.py pour priorité ≥ 4.
# - Snapshots compressés avec gzip dans data/mind/mind_dialogue_snapshots/.
# - Sauvegardes incrémentielles toutes les 5 min dans data/checkpoints/ (5 versions).
# - Sauvegardes distribuées toutes les 15 min vers S3 (configurable).
# - Tests unitaires disponibles dans tests/test_mind_dialogue.py.
# - Phases intégrées : Phase 7 (mémoire contextuelle), Phase 8 (auto-conscience via confidence_drop_rate), Phase 17 (SHAP).
# - Validation complète prévue pour juin 2025.
# - Narratif interactif via GPT-4o-mini avec métriques iv_skew, entry_freq, period (Proposition 1).

import gzip
import json
import os
import queue
import re
import signal
import sqlite3
import threading
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterator, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import speech_recognition as sr
from openai import OpenAI
from sklearn.cluster import KMeans

from src.envs.trading_env import TradingEnv
from src.mind.mind import miya_speak
from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import get_config
from src.model.utils.db_setup import get_db_connection

# Variable pour gérer l'arrêt propre
RUNNING = True

class DialogueManager:
    """
    Classe pour gérer les dialogues interactifs avec MIA, incluant la reconnaissance vocale et les réponses conversationnelles.
    """

    def __init__(
        self,
        config_path: str = "config/mia_config.yaml",
        credentials_path: str = "config/credentials.yaml",
    ):
        """
        Initialise le gestionnaire de dialogues interactifs.

        Args:
            config_path (str): Chemin vers la configuration de MIA.
            credentials_path (str): Chemin vers les identifiants (ex. : API OpenAI).
        """
        self.alert_manager = AlertManager()
        self.checkpoint_versions = []
        signal.signal(signal.SIGINT, self.handle_sigint)

        start_time = datetime.now()
        try:
            self.recognizer = sr.Recognizer()
            self.config_path = config_path
            self.credentials_path = credentials_path
            self.env = None
            self.router = None
            self.df = None
            self.log_buffer = []  # Buffer pour les écritures CSV
            self.dialogue_history = []  # Historique des dialogues
            self.query_queue = queue.Queue()  # File pour les requêtes asynchrones

            # Chemins de sortie
            self.base_dir = Path(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            )
            self.log_dir = self.base_dir / "data" / "logs" / "mind"
            self.snapshot_dir = (
                self.base_dir / "data" / "mind" / "mind_dialogue_snapshots"
            )
            self.checkpoint_dir = self.base_dir / "data" / "checkpoints"
            self.dialogue_log_path = self.log_dir / "mind_dialogue.csv"
            self.performance_log = self.log_dir / "mind_dialogue_performance.csv"
            self.dashboard_path = (
                self.base_dir / "data" / "mind" / "mind_dialogue_dashboard.json"
            )
            self.log_dir.mkdir(exist_ok=True)
            self.snapshot_dir.mkdir(exist_ok=True)
            self.checkpoint_dir.mkdir(exist_ok=True)

            # Charger la configuration
            self.config = self.load_config(config_path)

            # Charger la configuration NLP
            nlp_config = self.load_nlp_config()
            if nlp_config.get("enabled") and nlp_config.get("api_key"):
                self.nlp_client = OpenAI(api_key=nlp_config["api_key"])
            else:
                self.nlp_client = None
                warning_msg = "Client NLP non configuré, mode conversationnel limité"
                self.alert_manager.send_alert(warning_msg, priority=3)

            # Liste blanche pour les commandes
            self.command_whitelist = [
                "régime actuel",
                "valeur gex",
                "drawdown actuel",
                "analyse le drawdown",
                "résume tes stats",
                "simule position longue",
            ]

            # Lancer le thread pour traiter les requêtes asynchrones
            threading.Thread(target=self.query_worker, daemon=True).start()

            success_msg = "Gestionnaire de dialogues initialisé"
            self.alert_manager.send_alert(success_msg, priority=2)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("init_dialogue_manager", latency, success=True)
            self.save_snapshot(
                "init",
                {"config_path": config_path, "timestamp": datetime.now().isoformat()},
            )
        except Exception as e:
            error_msg = f"Erreur initialisation DialogueManager : {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=5)
            self.log_performance(
                "init_dialogue_manager", 0, success=False, error=str(e)
            )
            raise

    def handle_sigint(self, signum, frame):
        """Gère l'arrêt propre sur SIGINT."""
        global RUNNING
        snapshot = {"timestamp": datetime.now().isoformat(), "status": "SIGINT"}
        try:
            RUNNING = False
            self.save_snapshot("sigint", snapshot)
            success_msg = "Arrêt propre sur SIGINT, snapshot sauvegardé"
            self.alert_manager.send_alert(success_msg, priority=2)
            self.log_performance("handle_sigint", 0, success=True)
            exit(0)
        except Exception as e:
            error_msg = f"Erreur sauvegarde SIGINT: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            self.log_performance("handle_sigint", 0, success=False, error=str(e))
            exit(1)

    def _write_performance_log(self, log_entry: Dict) -> None:
        """Écrit une entrée de performance dans le CSV sans récursion."""
        try:
            self.log_buffer.append(log_entry)
            if len(self.log_buffer) >= self.config.get("logging", {}).get(
                "buffer_size", 100
            ):
                log_df = pd.DataFrame(self.log_buffer)
                mode = "a" if self.performance_log.exists() else "w"

                def save_log():
                    log_df.to_csv(
                        self.performance_log,
                        mode=mode,
                        index=False,
                        header=not self.performance_log.exists(),
                        encoding="utf-8",
                    )

                self.with_retries(save_log)
                self.log_buffer = []
        except Exception as e:
            error_msg = (
                f"Erreur écriture performance log: {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ):
        """
        Enregistre les performances avec psutil dans data/logs/mind_dialogue_performance.csv.

        Args:
            operation (str): Nom de l’opération.
            latency (float): Temps d’exécution en secondes.
            success (bool): Succès de l’opération.
            error (str, optional): Message d’erreur.
            **kwargs: Paramètres supplémentaires (ex. : num_dialogues, snapshot_size_mb).
        """
        start_time = datetime.now()
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
            cpu_usage = psutil.cpu_percent()
            if memory_usage > 1024:
                alert_msg = f"ALERTE: Usage mémoire élevé ({memory_usage:.2f} MB)"
                self.alert_manager.send_alert(alert_msg, priority=5)
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "success": success,
                "latency": latency,
                "error": error,
                "cpu_percent": cpu_usage,
                "memory_mb": memory_usage,
                **kwargs,
            }
            self._write_performance_log(log_entry)
        except Exception as e:
            error_msg = (
                f"Erreur journalisation performance: {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)

    def save_snapshot(
        self, snapshot_type: str, data: Dict, compress: bool = True
    ) -> None:
        """
        Sauvegarde un instantané des résultats.

        Args:
            snapshot_type (str): Type de snapshot (ex. : init, sigint).
            data (Dict): Données à sauvegarder.
            compress (bool): Compresser avec gzip (défaut : True).
        """
        try:
            start_time = datetime.now()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = {
                "timestamp": timestamp,
                "type": snapshot_type,
                "data": data,
                "buffer_size": len(self.log_buffer),
            }
            snapshot_path = (
                self.snapshot_dir / f"snapshot_{snapshot_type}_{timestamp}.json"
            )
            self.snapshot_dir.mkdir(exist_ok=True)

            def write_snapshot():
                if compress:
                    with gzip.open(f"{snapshot_path}.gz", "wt", encoding="utf-8") as f:
                        json.dump(snapshot, f, indent=4)
                else:
                    with open(snapshot_path, "w", encoding="utf-8") as f:
                        json.dump(snapshot, f, indent=4)

            self.with_retries(write_snapshot)
            save_path = f"{snapshot_path}.gz" if compress else snapshot_path
            file_size = os.path.getsize(save_path) / 1024 / 1024
            if file_size > 1.0:
                alert_msg = f"Snapshot size {file_size:.2f} MB exceeds 1 MB"
                self.alert_manager.send_alert(alert_msg, priority=3)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Snapshot {snapshot_type} sauvegardé: {save_path}"
            self.alert_manager.send_alert(success_msg, priority=1)
            self.log_performance(
                "save_snapshot", latency, success=True, snapshot_size_mb=file_size
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            self.log_performance("save_snapshot", 0, success=False, error=str(e))

    def checkpoint(self, data: pd.DataFrame) -> None:
        """
        Sauvegarde incrémentielle des dialogues toutes les 5 minutes avec versionnage (5 versions).

        Args:
            data (pd.DataFrame): Données des dialogues à sauvegarder.
        """
        try:
            start_time = datetime.now()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_data = {
                "timestamp": timestamp,
                "num_rows": len(data),
                "columns": list(data.columns),
            }
            checkpoint_path = self.checkpoint_dir / f"mind_dialogue_{timestamp}.json.gz"
            self.checkpoint_dir.mkdir(exist_ok=True)

            def write_checkpoint():
                with gzip.open(checkpoint_path, "wt", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, indent=4)
                data.to_csv(
                    checkpoint_path.with_suffix(".csv"), index=False, encoding="utf-8"
                )

            self.with_retries(write_checkpoint)
            self.checkpoint_versions.append(checkpoint_path)
            if len(self.checkpoint_versions) > 5:
                oldest = self.checkpoint_versions.pop(0)
                if os.path.exists(oldest):
                    os.remove(oldest)
                csv_oldest = oldest.with_suffix(".csv")
                if os.path.exists(csv_oldest):
                    os.remove(csv_oldest)
            file_size = os.path.getsize(checkpoint_path) / 1024 / 1024
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Checkpoint sauvegardé: {checkpoint_path}"
            self.alert_manager.send_alert(success_msg, priority=1)
            self.log_performance(
                "checkpoint",
                latency,
                success=True,
                file_size_mb=file_size,
                num_rows=len(data),
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde checkpoint: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            self.log_performance("checkpoint", 0, success=False, error=str(e))

    def with_retries(
        self, func: callable, max_attempts: int = 3, delay_base: float = 2.0
    ) -> Optional[any]:
        """
        Exécute une fonction avec retries (max 3, délai exponentiel).

        Args:
            func (callable): Fonction à exécuter.
            max_attempts (int): Nombre maximum de tentatives.
            delay_base (float): Base pour le délai exponentiel.

        Returns:
            Optional[any]: Résultat de la fonction ou None si échec.
        """
        start_time = datetime.now()
        for attempt in range(max_attempts):
            try:
                result = func()
                latency = (datetime.now() - start_time).total_seconds()
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
                    self.alert_manager.send_alert(error_msg, priority=4)
                    self.log_performance(
                        f"retry_attempt_{attempt+1}",
                        0,
                        success=False,
                        error=str(e),
                        attempt_number=attempt + 1,
                    )
                    return None
                delay = delay_base**attempt
                warning_msg = f"Tentative {attempt+1} échouée, retry après {delay}s"
                self.alert_manager.send_alert(warning_msg, priority=3)
                time.sleep(delay)

    def load_config(self, config_path: str) -> Dict:
        """
        Charge la configuration depuis mia_config.yaml.

        Args:
            config_path (str): Chemin du fichier de configuration.

        Returns:
            Dict: Configuration de MIA.
        """
        start_time = datetime.now()
        try:
            config = get_config(self.base_dir / "config" / config_path)
            if not config:
                raise ValueError("Configuration vide ou non trouvée")
            required_keys = ["logging", "s3_bucket", "s3_prefix"]
            missing_keys = [key for key in required_keys if key not in config]
            if missing_keys:
                raise ValueError(f"Clés de configuration manquantes: {missing_keys}")
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Configuration {config_path} chargée"
            self.alert_manager.send_alert(success_msg, priority=1)
            self.log_performance("load_config", latency, success=True)
            return config
        except Exception as e:
            error_msg = f"Erreur chargement config : {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=5)
            self.log_performance("load_config", 0, success=False, error=str(e))
            return {
                "logging": {"buffer_size": 100},
                "s3_bucket": None,
                "s3_prefix": "mind_dialogue/",
            }

    def load_nlp_config(self) -> Dict:
        """
        Charge la configuration NLP depuis credentials.yaml.

        Returns:
            Dict: Configuration NLP.
        """
        start_time = datetime.now()
        try:
            config = get_config(self.base_dir / self.credentials_path)
            if not config or "nlp" not in config:
                raise ValueError("Configuration NLP vide ou non trouvée")
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Configuration NLP chargée"
            self.alert_manager.send_alert(success_msg, priority=1)
            self.log_performance("load_nlp_config", latency, success=True)
            return config.get("nlp", {})
        except Exception as e:
            error_msg = (
                f"Erreur chargement config NLP : {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=5)
            self.log_performance("load_nlp_config", 0, success=False, error=str(e))
            return {}

    def detect_feature_set(self, data: pd.DataFrame) -> str:
        """
        Détecte le type d’ensemble de features basé sur le nombre de colonnes.

        Args:
            data (pd.DataFrame): Données à analyser.

        Returns:
            str: Type d’ensemble ('training', 'inference').
        """
        start_time = datetime.now()
        try:
            num_cols = len(data.columns)
            if num_cols >= 350:
                feature_set = "training"
            elif num_cols >= 150:
                feature_set = "inference"
            else:
                raise ValueError(f"Nombre de features trop bas: {num_cols} < 150")
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = (
                f"Ensemble de features détecté: {feature_set} ({num_cols} colonnes)"
            )
            self.alert_manager.send_alert(success_msg, priority=1)
            self.log_performance("detect_feature_set", latency, success=True)
            return feature_set
        except Exception as e:
            error_msg = f"Erreur détection ensemble features : {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            self.log_performance("detect_feature_set", 0, success=False, error=str(e))
            return "inference"

    def validate_data(self, data: pd.DataFrame) -> None:
        """
        Valide les données d’entrée (350/150 features) avec une validation intelligente.

        Args:
            data (pd.DataFrame): Données à valider.

        Raises:
            ValueError: Si les données sont invalides.
        """
        start_time = datetime.now()
        try:
            feature_set = self.detect_feature_set(data)
            expected_features = 350 if feature_set == "training" else 150

            if len(data.columns) != expected_features:
                raise ValueError(
                    f"Nombre de features incorrect: {len(data.columns)} != {expected_features} pour ensemble {feature_set}"
                )

            feature_sets_path = self.base_dir / "config" / "feature_sets.yaml"
            if feature_sets_path.exists() and feature_set == "inference":
                feature_config = get_config(feature_sets_path)
                shap_features = feature_config.get("inference", {}).get(
                    "shap_features", []
                )
                if len(shap_features) != 150:
                    raise ValueError(
                        f"Nombre de SHAP features incorrect dans feature_sets.yaml: {len(shap_features)} != 150"
                    )
                missing_features = [
                    col for col in shap_features if col not in data.columns
                ]
                if missing_features:
                    raise ValueError(
                        f"Features SHAP manquantes dans les données: {missing_features[:5]}"
                    )

            critical_cols = [
                "vix",
                "neural_regime",
                "predicted_volatility",
                "trade_frequency_1s",
                "close",
                "rsi_14",
                "gex",
                "drawdown",
            ]
            for col in critical_cols:
                if col in data.columns:
                    if data[col].isnull().any():
                        raise ValueError(f"Colonne {col} contient des NaN")
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        raise ValueError(
                            f"Colonne {col} n'est pas numérique: {data[col].dtype}"
                        )
            if "timestamp" in data.columns:
                latest_timestamp = pd.to_datetime(
                    data["timestamp"].iloc[-1], errors="coerce"
                )
                if pd.isna(latest_timestamp):
                    raise ValueError("Timestamp non valide")
                if latest_timestamp > datetime.now() + timedelta(
                    minutes=5
                ) or latest_timestamp < datetime.now() - timedelta(hours=24):
                    raise ValueError(f"Timestamp hors plage: {latest_timestamp}")

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Données validées pour ensemble {feature_set} ({expected_features} features)"
            self.alert_manager.send_alert(success_msg, priority=1)
            self.log_performance("validate_data", latency, success=True)
        except Exception as e:
            error_msg = (
                f"Erreur validation données : {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=5)
            self.log_performance("validate_data", 0, success=False, error=str(e))
            raise

    def collect_metrics(self) -> Iterator[Dict]:
        """
        Collecte les métriques de trading pour réponses conversationnelles, utilisant un générateur.

        Yields:
            Dict: Métriques de trading (num_trades, avg_profit, max_drawdown, regime_counts, iv_skew, entry_freq, period).
        """
        start_time = datetime.now()
        try:
            trades_file = self.base_dir / "data" / "trades" / "trades_simulated.csv"
            if not trades_file.exists():
                error_msg = "Fichier trades_simulated.csv introuvable"
                self.alert_manager.send_alert(error_msg, priority=3)
                yield {
                    "num_trades": 0,
                    "avg_profit": 0,
                    "max_drawdown": 0,
                    "regime_counts": {},
                    "iv_skew": 0.0,
                    "entry_freq": 0.0,
                    "period": "N/A",
                }
                return

            def read_trades():
                trades = pd.read_csv(trades_file, iterator=True, chunksize=1000)
                num_trades = 0
                total_reward = 0
                min_drawdown = 0
                regime_counts = {}
                iv_skew = 0.0
                entry_freq = 0.0
                period = "N/A"

                critical_cols = [
                    "bid_size_level_1",
                    "ask_size_level_1",
                    "trade_frequency_1s",
                ]
                for chunk in trades:
                    for col in critical_cols:
                        if col in chunk.columns:
                            if not pd.api.types.is_numeric_dtype(chunk[col]):
                                raise ValueError(
                                    f"Colonne {col} n'est pas numérique : {chunk[col].dtype}"
                                )
                            non_scalar = [
                                val
                                for val in chunk[col]
                                if isinstance(val, (list, dict, tuple))
                            ]
                            if non_scalar:
                                raise ValueError(
                                    f"Colonne {col} contient des valeurs non scalaires : {non_scalar[:5]}"
                                )

                    num_trades += len(chunk)
                    total_reward += (
                        chunk["reward"].sum() if "reward" in chunk.columns else 0
                    )
                    min_drawdown = min(
                        min_drawdown,
                        chunk["drawdown"].min() if "drawdown" in chunk.columns else 0,
                    )
                    if "regime" in chunk.columns:
                        chunk_counts = chunk["regime"].value_counts().to_dict()
                        for regime, count in chunk_counts.items():
                            regime_counts[regime] = regime_counts.get(regime, 0) + count
                    if "iv_skew" in chunk.columns:
                        iv_skew = chunk["iv_skew"].mean()
                    if "timestamp" in chunk.columns:
                        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce")
                        period_days = (chunk["timestamp"].max() - chunk["timestamp"].min()).days
                        entry_freq = len(chunk) / period_days if period_days > 0 else 0.0
                        period = f"{chunk['timestamp'].min().strftime('%Y-%m-%d')} → {chunk['timestamp'].max().strftime('%Y-%m-%d')}"

                return {
                    "num_trades": num_trades,
                    "avg_profit": total_reward / num_trades if num_trades > 0 else 0,
                    "max_drawdown": min_drawdown,
                    "regime_counts": regime_counts,
                    "iv_skew": iv_skew,
                    "entry_freq": entry_freq,
                    "period": period,
                }

            metrics = self.with_retries(read_trades)
            if metrics is None:
                raise ValueError("Échec collecte métriques")
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Métriques collectées: {metrics['num_trades']} trades"
            self.alert_manager.send_alert(success_msg, priority=1)
            self.log_performance(
                "collect_metrics",
                latency,
                success=True,
                num_trades=metrics["num_trades"],
            )
            yield metrics
        except Exception as e:
            error_msg = (
                f"Erreur collecte métriques : {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=5)
            yield {
                "num_trades": 0,
                "avg_profit": 0,
                "max_drawdown": 0,
                "regime_counts": {},
                "iv_skew": 0.0,
                "entry_freq": 0.0,
                "period": "N/A",
            }

    def update_context(self, df: pd.DataFrame, router, env: TradingEnv) -> None:
        """
        Met à jour les données et le contexte.

        Args:
            df (pd.DataFrame): Données de marché.
            router: Gestionnaire de régimes de marché.
            env (TradingEnv): Environnement de trading.
        """
        start_time = datetime.now()
        try:
            self.validate_data(df)
            self.df = df
            self.router = router
            self.env = env
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Contexte mis à jour"
            self.alert_manager.send_alert(success_msg, priority=2)
            self.log_performance("update_context", latency, success=True)
        except Exception as e:
            error_msg = (
                f"Erreur mise à jour contexte : {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=5)
            self.log_performance("update_context", 0, success=False, error=str(e))

    def listen_command(self) -> Optional[str]:
        """
        Écoute une commande vocale avec repli sur Sphinx en cas d’échec.

        Returns:
            Optional[str]: Commande reconnue ou None si erreur.
        """
        start_time = datetime.now()

        def listen():
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source)
                miya_speak(
                    "J’écoute, quelle est votre commande ?",
                    tag="DIALOGUE",
                    voice_profile="calm",
                    priority=2,
                )
                audio = self.recognizer.listen(source, timeout=5)
                try:
                    command = self.recognizer.recognize_google(audio, language="fr-FR")
                except sr.RequestError:
                    warning_msg = "API Google indisponible, tentative avec Sphinx"
                    self.alert_manager.send_alert(warning_msg, priority=3)
                    command = self.recognizer.recognize_sphinx(audio, language="fr-FR")
                success_msg = f"Commande reçue : {command}"
                self.alert_manager.send_alert(success_msg, priority=2)
                return command.lower()

        try:
            command = self.with_retries(listen)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("listen_command", latency, success=True)
            return command
        except Exception as e:
            error_msg = f"Erreur de reconnaissance : {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=5)
            self.log_performance("listen_command", 0, success=False, error=str(e))
            return None

    def handle_regime(self) -> str:
        """
        Gère la commande 'régime actuel'.

        Returns:
            str: Réponse générée.
        """
        start_time = datetime.now()
        try:
            regime = self.router.get_current_regime()
            response = f"Régime actuel : {regime}"
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Régime actuel récupéré: {regime}"
            self.alert_manager.send_alert(success_msg, priority=1)
            self.log_performance("handle_regime", latency, success=True)
            return response
        except Exception as e:
            error_msg = f"Erreur gestion régime : {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            self.log_performance("handle_regime", 0, success=False, error=str(e))
            return "Erreur lors de la récupération du régime"

    def handle_gex(self) -> str:
        """
        Gère la commande 'valeur gex'.

        Returns:
            str: Réponse générée.
        """
        start_time = datetime.now()
        try:
            gex = self.df["gex"].iloc[-1] if "gex" in self.df.columns else 0
            response = f"Valeur GEX : {gex:.2f}"
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Valeur GEX récupérée: {gex:.2f}"
            self.alert_manager.send_alert(success_msg, priority=1)
            self.log_performance("handle_gex", latency, success=True)
            return response
        except Exception as e:
            error_msg = f"Erreur gestion GEX : {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            self.log_performance("handle_gex", 0, success=False, error=str(e))
            return "Erreur lors de la récupération du GEX"

    def handle_drawdown(self) -> str:
        """
        Gère la commande 'drawdown actuel'.

        Returns:
            str: Réponse générée.
        """
        start_time = datetime.now()
        try:
            drawdown = (
                self.df["drawdown"].iloc[-1] if "drawdown" in self.df.columns else 0
            )
            response = f"Drawdown actuel : {drawdown:.2%}"
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Drawdown actuel récupéré: {drawdown:.2%}"
            self.alert_manager.send_alert(success_msg, priority=1)
            self.log_performance("handle_drawdown", latency, success=True)
            return response
        except Exception as e:
            error_msg = f"Erreur gestion drawdown : {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            self.log_performance("handle_drawdown", 0, success=False, error=str(e))
            return "Erreur lors de la récupération du drawdown"

    def handle_analyze_drawdown(self) -> str:
        """
        Gère la commande 'analyse le drawdown'.

        Returns:
            str: Réponse générée.
        """
        start_time = datetime.now()
        try:
            if "drawdown" in self.df.columns:
                drawdown = self.df["drawdown"].iloc[-1]
                max_drawdown = self.df["drawdown"].min()
                avg_drawdown = self.df["drawdown"].mean()
                response = f"Analyse du drawdown : actuel {drawdown:.2%}, maximum {max_drawdown:.2%}, moyen {avg_drawdown:.2%}."
            else:
                if "close" in self.df.columns:
                    equity = self.df["close"].cumsum()
                    peak = equity.cummax()
                    drawdown = (equity - peak) / peak
                    current_drawdown = drawdown.iloc[-1]
                    max_drawdown = drawdown.min()
                    avg_drawdown = drawdown.mean()
                    response = f"Analyse du drawdown calculé : actuel {current_drawdown:.2%}, maximum {max_drawdown:.2%}, moyen {avg_drawdown:.2%}."
                else:
                    response = "Données insuffisantes pour analyser le drawdown."
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Analyse du drawdown effectuée"
            self.alert_manager.send_alert(success_msg, priority=1)
            self.log_performance("handle_analyze_drawdown", latency, success=True)
            return response
        except Exception as e:
            error_msg = f"Erreur analyse drawdown : {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            self.log_performance(
                "handle_analyze_drawdown", 0, success=False, error=str(e)
            )
            return "Erreur lors de l’analyse du drawdown"

    def handle_summary(self) -> str:
        """
        Gère la commande 'résume tes stats' avec narratif interactif (Proposition 1).

        Returns:
            str: Réponse générée.
        """
        start_time = datetime.now()
        try:
            metrics = next(self.collect_metrics())
            if not all(
                k in metrics
                for k in [
                    "num_trades",
                    "avg_profit",
                    "max_drawdown",
                    "regime_counts",
                    "iv_skew",
                    "entry_freq",
                    "period",
                ]
            ):
                error_msg = "Métriques manquantes pour le narratif: num_trades, avg_profit, max_drawdown, regime_counts, iv_skew, entry_freq, period"
                self.alert_manager.send_alert(error_msg, priority=3)
                self.log_performance("handle_summary", 0, success=False, error=error_msg)
                return "Données insuffisantes pour générer un résumé."

            context = ""
            if self.dialogue_history:
                last_command = self.dialogue_history[-1].get("command", "").lower()
                if "régime" in last_command:
                    context = f"Le dernier régime mentionné était {self.router.get_current_regime()}. "
                elif "drawdown" in last_command:
                    context = f"Le drawdown a été récemment discuté, voici un résumé plus large. "

            prompt = (
                f"Tu es MIA, une IA de trading sur les E-mini S&P 500. Réponds en français de manière naturelle, engageante et concise. "
                f"{context}"
                f"Résume ces statistiques de trading : {metrics['num_trades']} trades, "
                f"profit moyen {metrics['avg_profit']:.2f} dollars, drawdown maximum {metrics['max_drawdown']:.2%}, "
                f"IV skew {metrics['iv_skew']:.2f}, fréquence d’entrées {metrics['entry_freq']:.1f}/jour, "
                f"période {metrics['period']}. "
                f"Propose une action IA en une phrase."
            )

            def summarize():
                response = self.nlp_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                )
                return response.choices[0].message.content

            if (
                self.nlp_client
                and hasattr(self.nlp_client, "chat")
                and hasattr(self.nlp_client.chat, "completions")
            ):
                response = self.with_retries(summarize)
                if response is None:
                    raise ValueError("Échec appel API NLP")
                latency = (datetime.now() - start_time).total_seconds()
                success_msg = "Résumé narratif interactif généré"
                self.alert_manager.send_alert(success_msg, priority=1)
                self.log_performance("handle_summary", latency, success=True)
                return response
            else:
                error_msg = "Analyse conversationnelle non disponible."
                self.alert_manager.send_alert(error_msg, priority=3)
                self.log_performance(
                    "handle_summary", 0, success=False, error=error_msg
                )
                return error_msg
        except Exception as e:
            error_msg = f"Erreur API NLP : {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=5)
            self.log_performance("handle_summary", 0, success=False, error=str(e))
            return error_msg

    def handle_long_position(self, command: str) -> str:
        """
        Gère la commande 'simule position longue'.

        Args:
            command (str): Commande contenant le prix.

        Returns:
            str: Réponse générée.
        """
        start_time = datetime.now()
        try:
            match = re.search(r"(\d+\.?\d*)", command)
            if match:
                price = float(match.group())
                action = 1.0  # Long
                _, reward, _, _, info = self.env.step(np.array([action]))
                response = f"Simulation à {price} : Profit estimé = {reward:.2f} $, Risque = {info.get('risk', 0):.2%}"
                latency = (datetime.now() - start_time).total_seconds()
                success_msg = f"Simulation position longue effectuée à {price}"
                self.alert_manager.send_alert(success_msg, priority=1)
                self.log_performance("handle_long_position", latency, success=True)
                return response
            else:
                error_msg = "Prix invalide, ex. : 'Simule position longue 5000'"
                self.alert_manager.send_alert(error_msg, priority=3)
                self.log_performance(
                    "handle_long_position", 0, success=False, error=error_msg
                )
                return error_msg
        except Exception as e:
            error_msg = f"Erreur simulation position longue : {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            self.log_performance("handle_long_position", 0, success=False, error=str(e))
            return error_msg

    def process_command(self, command: Optional[str]) -> None:
        """
        Traite une commande vocale ou textuelle.

        Args:
            command (Optional[str]): Commande à traiter.
        """
        start_time = datetime.now()
        try:
            if not command or not self.df or not self.router or not self.env:
                dialogue_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "command": command,
                    "response": "Contexte non initialisé ou commande vide.",
                    "cluster_id": None,
                    "priority": 3,
                }
                error_msg = dialogue_entry["response"]
                self.alert_manager.send_alert(error_msg, priority=3)
                self.log_dialogue(dialogue_entry)
                self.log_performance(
                    "process_command", 0, success=False, error=error_msg
                )
                return

            command = command.lower().strip()
            if not any(trigger in command for trigger in self.command_whitelist):
                dialogue_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "command": command,
                    "response": "Commande non reconnue, essayez 'résume tes stats' ou posez une question sur mes trades !",
                    "cluster_id": None,
                    "priority": 2,
                }
                miya_speak(
                    dialogue_entry["response"],
                    tag="DIALOGUE",
                    voice_profile="calm",
                    priority=2,
                )
                self.log_dialogue(dialogue_entry)
                latency = (datetime.now() - start_time).total_seconds()
                self.log_performance("process_command", latency, success=True)
                return

            dialogue_entry = {
                "timestamp": datetime.now().isoformat(),
                "command": command,
                "response": None,
                "context": {
                    "neural_regime": (
                        self.df["neural_regime"].iloc[-1]
                        if "neural_regime" in self.df.columns
                        else 0
                    ),
                    "strategy_params": self.config.get("strategy_params", {}),
                },
                "cluster_id": None,
                "priority": 2,
            }

            COMMAND_MAP = {
                "régime actuel": self.handle_regime,
                "valeur gex": self.handle_gex,
                "drawdown actuel": self.handle_drawdown,
                "analyse le drawdown": self.handle_analyze_drawdown,
                "résume tes stats": self.handle_summary,
                "simule position longue": lambda: self.handle_long_position(command),
            }

            response = None
            for trigger, handler in COMMAND_MAP.items():
                if trigger in command:
                    response = handler()
                    dialogue_entry["priority"] = (
                        3
                        if trigger in ["analyse le drawdown", "résume tes stats"]
                        else 2
                    )
                    break

            if response:
                dialogue_entry["response"] = response
                miya_speak(
                    response,
                    tag="DIALOGUE",
                    voice_profile="informative",
                    priority=dialogue_entry["priority"],
                )
            else:
                if (
                    self.nlp_client
                    and hasattr(self.nlp_client, "chat")
                    and hasattr(self.nlp_client.chat, "completions")
                ):
                    metrics = next(self.collect_metrics())
                    prompt = (
                        f"Tu es MIA, une IA de trading sur les E-mini S&P 500. Réponds en français de manière naturelle, "
                        f"engageante et concise à la question : '{command}'. "
                        f"Voici mes stats récentes : {metrics['num_trades']} trades, profit moyen {metrics['avg_profit']:.2f} dollars, "
                        f"drawdown max {metrics['max_drawdown']:.2%}, régimes : {metrics['regime_counts']}. "
                        f"Si la question est vague, donne une réponse générale sur mon activité de trading."
                    )

                    def call_nlp():
                        response_obj = self.nlp_client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=150,
                        )
                        return response_obj.choices[0].message.content

                    response = self.with_retries(call_nlp)
                    if response is None:
                        raise ValueError("Échec appel API NLP")
                    dialogue_entry["response"] = response
                    dialogue_entry["latency"] = (
                        datetime.now() - start_time
                    ).total_seconds()
                    dialogue_entry["priority"] = 3
                    miya_speak(
                        response,
                        tag="DIALOGUE",
                        voice_profile="informative",
                        priority=3,
                    )
                else:
                    dialogue_entry["response"] = (
                        "Commande non reconnue, essayez 'résume tes stats', 'analyse le drawdown', ou posez une question sur mes trades !"
                    )
                    miya_speak(
                        dialogue_entry["response"],
                        tag="DIALOGUE",
                        voice_profile="calm",
                        priority=2,
                    )

            self.log_dialogue(dialogue_entry)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "process_command",
                latency,
                success=True,
                num_dialogues=len(self.dialogue_history),
            )
        except Exception as e:
            error_msg = (
                f"Erreur traitement commande : {str(e)}\n{traceback.format_exc()}"
            )
            dialogue_entry = {
                "timestamp": datetime.now().isoformat(),
                "command": command,
                "response": error_msg,
                "cluster_id": None,
                "priority": 5,
            }
            self.alert_manager.send_alert(error_msg, priority=5)
            self.log_dialogue(dialogue_entry)
            self.log_performance("process_command", 0, success=False, error=str(e))

    def store_dialogue_pattern(self, dialogue_entry: Dict) -> Dict:
        """
        Stocke un pattern de dialogue dans market_memory.db avec clusterisation K-means (méthode 7).

        Args:
            dialogue_entry (Dict): Entrée de dialogue à stocker.

        Returns:
            Dict: Entrée avec cluster_id.
        """
        start_time = datetime.now()
        try:
            features = {
                "priority": dialogue_entry.get("priority", 1),
                "command_length": len(dialogue_entry.get("command", "")),
                "response_length": len(dialogue_entry.get("response", "")),
            }
            required_cols = ["priority", "command_length", "response_length"]
            missing_cols = [col for col in required_cols if col not in features]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(features)} colonnes)"
                self.alert_manager.send_alert(alert_msg, priority=3)

            df = pd.DataFrame([features])
            X = df[["priority", "command_length", "response_length"]].fillna(0).values

            if len(X) < 10:
                dialogue_entry["cluster_id"] = 0
            else:

                def run_kmeans():
                    kmeans = KMeans(n_clusters=10, random_state=42)
                    return kmeans.fit_predict(X)[0]

                dialogue_entry["cluster_id"] = self.with_retries(run_kmeans)
                if dialogue_entry["cluster_id"] is None:
                    raise ValueError("Échec clusterisation K-means")

            def store_clusters():
                conn = get_db_connection(
                    str(self.base_dir / "data" / "market_memory.db")
                )
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS clusters (
                        cluster_id INTEGER,
                        event_type TEXT,
                        features TEXT,
                        timestamp TEXT
                    )
                """
                )
                features_json = json.dumps(features)
                cursor.execute(
                    """
                    INSERT INTO clusters (cluster_id, event_type, features, timestamp)
                    VALUES (?, ?, ?, ?)
                """,
                    (
                        dialogue_entry["cluster_id"],
                        "dialogue",
                        features_json,
                        dialogue_entry["timestamp"],
                    ),
                )
                conn.commit()
                conn.close()

            self.with_retries(store_clusters)

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = (
                f"Pattern dialogue stocké, cluster_id={dialogue_entry['cluster_id']}"
            )
            self.alert_manager.send_alert(success_msg, priority=1)
            self.log_performance(
                "store_dialogue_pattern",
                latency,
                success=True,
                confidence_drop_rate=confidence_drop_rate,
            )
            self.save_snapshot(
                "store_dialogue_pattern",
                {"cluster_id": dialogue_entry["cluster_id"], "features": features},
            )
            return dialogue_entry
        except Exception as e:
            error_msg = (
                f"Erreur stockage pattern dialogue : {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            self.log_performance(
                "store_dialogue_pattern", 0, success=False, error=str(e)
            )
            dialogue_entry["cluster_id"] = 0
            return dialogue_entry

    def visualize_dialogue_patterns(self):
        """
        Génère une heatmap des clusters de dialogues dans data/figures/mind_dialogue/.
        """
        start_time = datetime.now()
        try:
            dialogues = self.dialogue_history[-100:]  # Limite à 100 dialogues
            if not dialogues:
                error_msg = "Aucun dialogue pour visualisation"
                self.alert_manager.send_alert(error_msg, priority=3)
                return

            df = pd.DataFrame(dialogues)
            if "cluster_id" not in df.columns or df["cluster_id"].isnull().all():
                error_msg = "Aucun cluster_id pour visualisation"
                self.alert_manager.send_alert(error_msg, priority=3)
                return

            pivot = df.pivot_table(
                index="cluster_id",
                columns="priority",
                values="timestamp",
                aggfunc="count",
                fill_value=0,
            )
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot, annot=True, cmap="Blues")
            plt.title("Heatmap des Clusters de Dialogues")
            figures_dir = self.base_dir / "data" / "figures" / "mind_dialogue"
            figures_dir.mkdir(exist_ok=True)
            output_path = (
                figures_dir
                / f"dialogue_clusters_{datetime.now().strftime('%Y%m%d')}.png"
            )

            def save_fig():
                plt.savefig(output_path)
                plt.close()

            self.with_retries(save_fig)

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Heatmap des clusters de dialogues générée"
            self.alert_manager.send_alert(success_msg, priority=1)
            self.log_performance("visualize_dialogue_patterns", latency, success=True)
        except Exception as e:
            error_msg = (
                f"Erreur visualisation clusters : {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            self.log_performance(
                "visualize_dialogue_patterns", 0, success=False, error=str(e)
            )

    def log_dialogue(self, dialogue_entry: Dict) -> None:
        """
        Journalise une interaction de dialogue.

        Args:
            dialogue_entry (Dict): Entrée de dialogue à journaliser.
        """
        start_time = datetime.now()
        try:
            dialogue_entry = self.store_dialogue_pattern(dialogue_entry)
            self.log_buffer.append(dialogue_entry)
            if len(self.log_buffer) >= self.config.get("logging", {}).get(
                "buffer_size", 100
            ):
                log_df = pd.DataFrame(self.log_buffer)
                mode = "a" if self.dialogue_log_path.exists() else "w"

                def save_log():
                    log_df.to_csv(
                        self.dialogue_log_path,
                        mode=mode,
                        index=False,
                        header=not self.dialogue_log_path.exists(),
                        encoding="utf-8",
                    )

                self.with_retries(save_log)
                self.checkpoint(log_df)
                self.log_buffer = []

            self.dialogue_history.append(dialogue_entry)
            self.save_dialogue_snapshot(
                len(self.dialogue_history),
                dialogue_entry["command"],
                dialogue_entry["response"],
            )
            self.update_dashboard()

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Dialogue journalisé: {dialogue_entry['command']}"
            self.alert_manager.send_alert(success_msg, priority=1)
            self.log_performance(
                "log_dialogue",
                latency,
                success=True,
                num_dialogues=len(self.dialogue_history),
            )
        except Exception as e:
            error_msg = (
                f"Erreur journalisation dialogue : {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=5)
            self.log_performance("log_dialogue", 0, success=False, error=str(e))

    def respond_to_query(self, query: str) -> str:
        """
        Répond à une requête textuelle ou vocale.

        Args:
            query (str): Requête à traiter.

        Returns:
            str: Réponse générée.
        """
        start_time = datetime.now()
        try:
            if not query:
                error_msg = "Aucune requête fournie."
                self.alert_manager.send_alert(error_msg, priority=3)
                self.log_performance(
                    "respond_to_query", 0, success=False, error=error_msg
                )
                return error_msg

            query = query.lower().strip()
            self.query_queue.put(query)

            timeout = time.time() + 10
            while time.time() < timeout:
                if (
                    self.dialogue_history
                    and self.dialogue_history[-1]["command"] == query
                ):
                    response = (
                        self.dialogue_history[-1]["response"]
                        or "Aucune réponse générée."
                    )
                    latency = (datetime.now() - start_time).total_seconds()
                    success_msg = f"Réponse générée pour la requête: {query}"
                    self.alert_manager.send_alert(success_msg, priority=1)
                    self.log_performance("respond_to_query", latency, success=True)
                    return response
            error_msg = "Timeout attente réponse"
            self.alert_manager.send_alert(error_msg, priority=3)
            self.log_performance("respond_to_query", 0, success=False, error=error_msg)
            return error_msg
        except Exception as e:
            error_msg = (
                f"Erreur traitement requête : {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=5)
            self.log_performance("respond_to_query", 0, success=False, error=str(e))
            return error_msg

    def query_worker(self):
        """
        Thread pour traiter les requêtes de manière asynchrone.
        """
        while RUNNING:
            try:
                query = self.query_queue.get(timeout=10)
                start_time = datetime.now()
                self.process_command(query)
                self.query_queue.task_done()
                latency = (datetime.now() - start_time).total_seconds()
                self.log_performance("query_worker", latency, success=True)
            except queue.Empty:
                continue
            except Exception as e:
                error_msg = f"Erreur traitement requête asynchrone : {str(e)}\n{traceback.format_exc()}"
                self.alert_manager.send_alert(error_msg, priority=5)
                self.log_performance("query_worker", 0, success=False, error=str(e))

    def save_dialogue_snapshot(self, step: int, command: str, response: str) -> None:
        """
        Sauvegarde un instantané d’une interaction de dialogue.

        Args:
            step (int): Étape du dialogue.
            command (str): Commande reçue.
            response (str): Réponse générée.
        """
        start_time = datetime.now()
        try:
            snapshot = {
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "command": command,
                "response": response,
                "dialogue_history": self.dialogue_history[-10:],
                "context": {
                    "neural_regime": (
                        self.df["neural_regime"].iloc[-1]
                        if self.df is not None and "neural_regime" in self.df.columns
                        else 0
                    ),
                    "strategy_params": self.config.get("strategy_params", {}),
                },
                "performance": {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_mb": psutil.Process().memory_info().rss / 1024 / 1024,
                },
            }
            self.save_snapshot(f"dialogue_step_{step:04d}", snapshot, compress=True)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Snapshot dialogue step {step} sauvegardé"
            self.alert_manager.send_alert(success_msg, priority=2)
            self.log_performance("save_dialogue_snapshot", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot dialogue : {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            self.log_performance(
                "save_dialogue_snapshot", 0, success=False, error=str(e)
            )

    def update_dashboard(self) -> None:
        """
        Met à jour un fichier JSON pour partager l'état des dialogues avec mia_dashboard.py.
        """
        start_time = datetime.now()
        try:
            status = {
                "timestamp": datetime.now().isoformat(),
                "num_dialogues": len(self.dialogue_history),
                "last_command": (
                    self.dialogue_history[-1]["command"]
                    if self.dialogue_history
                    else None
                ),
                "last_response": (
                    self.dialogue_history[-1]["response"]
                    if self.dialogue_history
                    else None
                ),
                "recent_dialogues": self.dialogue_history[-10:],
                "latency": (
                    self.dialogue_history[-1].get("latency", 0)
                    if self.dialogue_history
                    else 0
                ),
                "performance": {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_mb": psutil.Process().memory_info().rss / 1024 / 1024,
                },
            }

            def save_dashboard():
                with open(self.dashboard_path, "w", encoding="utf-8") as f:
                    json.dump(status, f, indent=4)

            self.with_retries(save_dashboard)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Mise à jour dashboard dialogue effectuée"
            self.alert_manager.send_alert(success_msg, priority=1)
            self.log_performance("update_dashboard", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur mise à jour dashboard dialogue : {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            self.log_performance("update_dashboard", 0, success=False, error=str(e))

if __name__ == "__main__":
    dialogue = DialogueManager()
    command = dialogue.listen_command()
    dialogue.process_command(command)
    response = dialogue.respond_to_query("résume tes stats")
    print("Réponse à la requête :", response)
