# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/scripts/run_simulate.py
# Script pour simuler le trading pour MIA_IA_SYSTEM_v2_2025.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Simule des trades basés sur des features, avec journalisation, snapshots compressés, et alertes.
#        Conforme à la Phase 8 (auto-conscience via alertes), Phase 12 (simulation de trading),
#        et Phase 16 (ensemble learning).
#
# Dépendances : json, pandas>=2.0.0, pyyaml>=6.0.0, logging, os, time, psutil>=5.9.8, matplotlib>=3.7.0,
#               datetime, typing, gzip, traceback, src.model.utils.config_manager,
#               src.model.utils.alert_manager, src.model.utils.miya_console,
#               src.utils.telegram_alert, src.utils.standard
#
# Inputs : config/es_config.yaml, data/features/features_latest_filtered.csv, config/feature_sets.yaml
#
# Outputs : data/logs/run_simulate.log, data/logs/run_simulate_performance.csv,
#           data/trades/trades_simulated.csv, data/simulation_dashboard.json,
#           data/simulation_snapshots/snapshot_*.json.gz,
#           data/figures/simulation/simulation_balance_*.png,
#           data/figures/simulation/simulation_rewards_*.png,
#           data/figures/simulation/simulation_errors_*.png
#
# Notes :
# - Gère 350 features pour l’entraînement et 150 SHAP features pour l’inférence.
# - Utilise IQFeed exclusivement via les features générées (indirectement).
# - Implémente retries (max 3, délai 2^attempt), logs psutil, snapshots JSON compressés, alertes centralisées.
# - Tests unitaires dans tests/test_run_simulate.py.

import gzip
import json
import logging
import os
import traceback
from datetime import datetime, timedelta
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import psutil
import yaml

from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import config_manager
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.utils.standard import with_retries
from src.utils.telegram_alert import send_telegram_alert

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "es_config.yaml")
FEATURES_PATH = os.path.join(
    BASE_DIR, "data", "features", "features_latest_filtered.csv"
)
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "trades", "trades_simulated.csv")
DASHBOARD_PATH = os.path.join(BASE_DIR, "data", "simulation_dashboard.json")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "simulation_snapshots")
CSV_LOG_PATH = os.path.join(BASE_DIR, "data", "logs", "run_simulate_performance.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures", "simulation")

# Configuration du logging
os.makedirs(os.path.join(BASE_DIR, "data", "logs"), exist_ok=True)
logging.basicBasicConfig(
    filename=os.path.join(BASE_DIR, "data", "logs", "run_simulate.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [Simulate] %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Configuration par défaut
DEFAULT_CONFIG = {
    "simulation": {
        "features_path": FEATURES_PATH,
        "output_path": OUTPUT_PATH,
        "capital": 10000.0,
        "trade_size": 0.1,
        "chunk_size": 10000,
        "cache_hours": 24,
        "retry_attempts": 3,
        "retry_delay": 5,
        "timeout_seconds": 3600,
        "max_cache_size": 1000,
        "num_features": 350,
        "shap_features": 150,
    }
}


class SimulationManager:
    """
    Classe pour gérer la simulation de trading avec journalisation, snapshots compressés, et alertes.
    """

    def __init__(self):
        """
        Initialise le gestionnaire de simulation.
        """
        self.log_buffer = []
        self.cache = {}
        try:
            self.config = self.load_config()
            self.buffer_size = self.config.get("logging", {}).get("buffer_size", 100)
            self.max_cache_size = self.config.get("cache", {}).get(
                "max_cache_size", 1000
            )
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            os.makedirs(FIGURES_DIR, exist_ok=True)
            miya_speak(
                "SimulationManager initialisé",
                tag="SIMULATION",
                voice_profile="calm",
                priority=2,
            )
            AlertManager().send_alert("SimulationManager initialisé", priority=1)
            logger.info("SimulationManager initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config_path": CONFIG_PATH})
        except Exception as e:
            error_msg = f"Erreur initialisation SimulationManager: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="SIMULATION", voice_profile="urgent", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            self.config = DEFAULT_CONFIG["simulation"]
            self.buffer_size = 100
            self.max_cache_size = 1000

    @with_retries(max_attempts=3, delay_base=2.0)
    def load_config(self, config_path: str = CONFIG_PATH) -> Dict:
        """
        Charge les paramètres de la simulation depuis es_config.yaml.

        Args:
            config_path (str): Chemin du fichier de configuration.

        Returns:
            Dict: Configuration de la simulation.
        """
        start_time = datetime.now()
        try:
            config = config_manager.get_config("es_config.yaml")
            if "simulation" not in config:
                raise ValueError("Clé 'simulation' manquante dans la configuration")
            required_keys = [
                "features_path",
                "output_path",
                "capital",
                "trade_size",
                "chunk_size",
                "num_features",
                "shap_features",
            ]
            missing_keys = [
                key for key in required_keys if key not in config["simulation"]
            ]
            if missing_keys:
                raise ValueError(f"Clés manquantes dans 'simulation': {missing_keys}")
            if config["simulation"]["num_features"] != 350:
                raise ValueError(
                    f"Nombre de features incorrect: {config['simulation']['num_features']} != 350"
                )
            if config["simulation"]["shap_features"] != 150:
                raise ValueError(
                    f"Nombre de SHAP features incorrect: {config['simulation']['shap_features']} != 150"
                )
            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                "Configuration simulation chargée",
                tag="SIMULATION",
                level="info",
                priority=2,
            )
            AlertManager().send_alert("Configuration simulation chargée", priority=1)
            logger.info("Configuration simulation chargée")
            self.log_performance("load_config", latency, success=True)
            self.save_snapshot("load_config", {"config_path": config_path})
            return config["simulation"]
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur chargement config: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="SIMULATION", voice_profile="urgent", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("load_config", latency, success=False, error=str(e))
            self.save_snapshot("load_config", {"error": str(e)})
            raise

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ) -> None:
        """
        Journalise les performances des opérations critiques.

        Args:
            operation (str): Nom de l’opération (ex. : simulate_trading).
            latency (float): Latence en secondes.
            success (bool): Succès de l’opération.
            error (str, optional): Message d’erreur si échec.
            **kwargs: Paramètres supplémentaires (ex. : num_trades).
        """
        try:
            memory_usage = (
                psutil.Process().memory_info().rss / 1024 / 1024
            )  # Mémoire en Mo
            cpu_percent = psutil.cpu_percent()
            if memory_usage > 1024:
                error_msg = f"ALERT: Utilisation mémoire élevée ({memory_usage:.2f} MB)"
                miya_alerts(error_msg, tag="SIMULATION", level="error", priority=5)
                AlertManager().send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
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
            error_msg = f"Erreur journalisation performance: {str(e)}"
            miya_alerts(error_msg, tag="SIMULATION", level="error", priority=3)
            AlertManager().send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def save_snapshot(self, snapshot_type: str, data: Dict) -> None:
        """
        Sauvegarde un instantané des résultats avec compression gzip.

        Args:
            snapshot_type (str): Type de snapshot (ex. : simulate_trading).
            data (Dict): Données à sauvegarder.
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
            path = os.path.join(
                SNAPSHOT_DIR, f"snapshot_{snapshot_type}_{timestamp}.json"
            )
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            with gzip.open(f"{path}.gz", "wt", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=4)
            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                f"Snapshot {snapshot_type} sauvegardé: {path}.gz",
                tag="SIMULATION",
                level="info",
                priority=1,
            )
            AlertManager().send_alert(
                f"Snapshot {snapshot_type} sauvegardé: {path}.gz", priority=1
            )
            logger.info(f"Snapshot {snapshot_type} sauvegardé: {path}.gz")
            self.log_performance("save_snapshot", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}"
            miya_alerts(error_msg, tag="SIMULATION", level="error", priority=3)
            AlertManager().send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("save_snapshot", 0, success=False, error=str(e))

    def save_dashboard_status(
        self, status: Dict, status_file: str = DASHBOARD_PATH
    ) -> None:
        """
        Sauvegarde l'état de la simulation pour mia_dashboard.py.

        Args:
            status (Dict): État de la simulation.
            status_file (str): Chemin du fichier JSON.
        """
        try:
            start_time = datetime.now()
            os.makedirs(os.path.dirname(status_file), exist_ok=True)
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4)
            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                f"État sauvegardé dans {status_file}",
                tag="SIMULATION",
                level="info",
                priority=2,
            )
            AlertManager().send_alert(f"État sauvegardé dans {status_file}", priority=1)
            logger.info(f"État sauvegardé dans {status_file}")
            self.log_performance("save_dashboard_status", latency, success=True)
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde dashboard: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="SIMULATION", voice_profile="urgent", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "save_dashboard_status", 0, success=False, error=str(e)
            )

    @with_retries(max_attempts=3, delay_base=2.0)
    def load_cache(self, output_path: str, cache_hours: int) -> Optional[pd.DataFrame]:
        """
        Charge les résultats de simulation en cache si disponibles et valides.

        Args:
            output_path (str): Chemin du fichier CSV.
            cache_hours (int): Durée de validité du cache en heures.

        Returns:
            Optional[pd.DataFrame]: Données en cache ou None.
        """
        start_time = datetime.now()
        try:
            if os.path.exists(output_path):
                df = pd.read_csv(output_path, encoding="utf-8")
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                    min_date = datetime.now() - timedelta(hours=cache_hours)
                    df = df[df["timestamp"] >= min_date]
                    if not df.empty and all(
                        col in df.columns
                        for col in ["timestamp", "action", "reward", "balance"]
                    ):
                        latency = (datetime.now() - start_time).total_seconds()
                        miya_speak(
                            f"Cache simulation chargé: {len(df)} trades",
                            tag="SIMULATION",
                            level="info",
                            priority=2,
                        )
                        AlertManager().send_alert(
                            f"Cache simulation chargé: {len(df)} trades", priority=1
                        )
                        logger.info(f"Cache simulation chargé: {len(df)} trades")
                        self.log_performance(
                            "load_cache", latency, success=True, num_trades=len(df)
                        )
                        self.save_snapshot(
                            "load_cache",
                            {"output_path": output_path, "num_trades": len(df)},
                        )
                        return df
            return None
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur chargement cache: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="SIMULATION", voice_profile="urgent", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("load_cache", latency, success=False, error=str(e))
            self.save_snapshot("load_cache", {"error": str(e)})
            return None

    def plot_simulation_metrics(
        self, df_trades: pd.DataFrame, output_dir: str = FIGURES_DIR
    ) -> None:
        """
        Génère des graphiques pour les métriques de simulation (solde, rewards).

        Args:
            df_trades (pd.DataFrame): DataFrame des trades simulés.
            output_dir (str): Répertoire pour sauvegarder les graphiques.
        """
        start_time = datetime.now()
        try:
            if df_trades.empty:
                miya_speak(
                    "Aucun trade à visualiser",
                    tag="SIMULATION",
                    level="warning",
                    priority=2,
                )
                AlertManager().send_alert("Aucun trade à visualiser", priority=2)
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(output_dir, exist_ok=True)

            # Courbe de solde
            plt.figure(figsize=(10, 6))
            plt.plot(
                df_trades["timestamp"],
                df_trades["balance"],
                label="Solde",
                color="blue",
            )
            plt.title("Solde au Fil du Temps")
            plt.xlabel("Horodatage")
            plt.ylabel("Solde")
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"simulation_balance_{timestamp}.png"))
            plt.close()

            # Distribution des rewards
            plt.figure(figsize=(10, 6))
            plt.hist(
                df_trades["reward"],
                bins=30,
                color="green",
                edgecolor="black",
                alpha=0.7,
            )
            plt.title("Distribution des Rewards")
            plt.xlabel("Reward")
            plt.ylabel("Fréquence")
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"simulation_rewards_{timestamp}.png"))
            plt.close()

            # Erreurs par type
            error_stats = {}
            for log in self.log_buffer:
                if not log["success"]:
                    error_type = (
                        log["error"].split(":")[0] if log["error"] else "Unknown"
                    )
                    error_stats[error_type] = error_stats.get(error_type, 0) + 1
            if error_stats:
                plt.figure(figsize=(10, 6))
                plt.bar(error_stats.keys(), error_stats.values(), color="red")
                plt.title("Erreurs par Type")
                plt.xlabel("Type d’Erreur")
                plt.ylabel("Nombre")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_dir, f"simulation_errors_{timestamp}.png")
                )
                plt.close()

            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                f"Graphiques de simulation générés: {output_dir}",
                tag="SIMULATION",
                level="info",
                priority=2,
            )
            AlertManager().send_alert(
                f"Graphiques de simulation générés: {output_dir}", priority=1
            )
            logger.info(f"Graphiques de simulation générés: {output_dir}")
            self.log_performance(
                "plot_simulation_metrics",
                latency,
                success=True,
                num_trades=len(df_trades),
            )
            self.save_snapshot(
                "plot_simulation_metrics",
                {"output_dir": output_dir, "num_trades": len(df_trades)},
            )
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur génération graphiques: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="SIMULATION", level="error", priority=3)
            AlertManager().send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "plot_simulation_metrics", latency, success=False, error=str(e)
            )
            self.save_snapshot("plot_simulation_metrics", {"error": str(e)})

    def validate_inputs(
        self,
        features_path: str,
        output_path: str,
        capital: float,
        trade_size: float,
        chunk_size: int,
    ) -> None:
        """
        Valide les paramètres et données d’entrée de la simulation.

        Args:
            features_path (str): Chemin des features.
            output_path (str): Chemin de sortie des trades.
            capital (float): Capital initial.
            trade_size (float): Taille des positions.
            chunk_size (int): Taille des morceaux.

        Raises:
            ValueError: Si les paramètres ou données sont invalides.
        """
        start_time = datetime.now()
        try:
            if not os.path.exists(features_path):
                error_msg = f"Fichier features introuvable: {features_path}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            if capital <= 0:
                error_msg = f"Capital invalide: {capital}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            if trade_size <= 0 or trade_size > 1:
                error_msg = f"Taille de trade invalide: {trade_size}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            if chunk_size <= 0:
                error_msg = f"Taille de chunk invalide: {chunk_size}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Charger un échantillon des features pour validation
            df_sample = pd.read_csv(features_path, nrows=10, encoding="utf-8")
            required_cols = ["timestamp", "close", "neural_regime", "rsi", "vix"]
            missing_cols = [
                col for col in required_cols if col not in df_sample.columns
            ]
            if missing_cols:
                error_msg = f"Colonnes manquantes dans {features_path}: {missing_cols}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                "Paramètres et données d’entrée validés",
                tag="SIMULATION",
                level="info",
                priority=2,
            )
            AlertManager().send_alert(
                "Paramètres et données d’entrée validés", priority=1
            )
            logger.info("Paramètres et données d’entrée validés")
            self.log_performance("validate_inputs", latency, success=True)
            self.save_snapshot(
                "validate_inputs",
                {
                    "features_path": features_path,
                    "capital": capital,
                    "trade_size": trade_size,
                },
            )
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur validation entrées: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="SIMULATION", voice_profile="urgent", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "validate_inputs", latency, success=False, error=str(e)
            )
            self.save_snapshot("validate_inputs", {"error": str(e)})
            raise

    @with_retries(max_attempts=3, delay_base=2.0)
    def simulate_trading(
        self,
        features_path: str,
        output_path: str,
        capital: float,
        trade_size: float,
        chunk_size: int,
    ) -> pd.DataFrame:
        """
        Simule des trades basés sur les features.

        Args:
            features_path (str): Chemin des features d’entrée.
            output_path (str): Chemin pour sauvegarder les trades.
            capital (float): Capital initial.
            trade_size (float): Taille des positions (fraction du capital).
            chunk_size (int): Taille des morceaux pour le traitement.

        Returns:
            pd.DataFrame: Résultats des trades simulés.
        """
        start_time = datetime.now()
        try:
            self.validate_inputs(
                features_path, output_path, capital, trade_size, chunk_size
            )
            self.config["num_features"]
            miya_speak(
                "Démarrage simulation trading",
                tag="SIMULATION",
                level="info",
                priority=3,
            )
            AlertManager().send_alert("Démarrage simulation trading", priority=2)
            logger.info("Démarrage simulation trading")

            trades = []
            balance = capital
            position = 0  # 0: neutre, 1: long, -1: short
            entry_price = 0.0

            # Charger la liste des features depuis feature_sets.yaml
            feature_sets_path = os.path.join(BASE_DIR, "config", "feature_sets.yaml")
            with open(feature_sets_path, "r", encoding="utf-8") as f:
                feature_sets = yaml.safe_load(f)
            training_features = feature_sets.get("training_features", [])[
                :350
            ]  # Limiter à 350 features
            if len(training_features) != 350:
                error_msg = f"Nombre de features incorrect dans feature_sets.yaml: {len(training_features)} != 350"
                miya_alerts(
                    error_msg, tag="SIMULATION", voice_profile="urgent", priority=5
                )
                AlertManager().send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                raise ValueError(error_msg)

            for chunk in pd.read_csv(
                features_path, chunksize=chunk_size, encoding="utf-8"
            ):
                if "timestamp" not in chunk.columns or "close" not in chunk.columns:
                    error_msg = f"Colonnes 'timestamp' ou 'close' manquantes dans {features_path}"
                    miya_alerts(
                        error_msg, tag="SIMULATION", voice_profile="urgent", priority=5
                    )
                    AlertManager().send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                if "neural_regime" not in chunk.columns:
                    chunk["neural_regime"] = "trend"
                if "rsi" not in chunk.columns:
                    chunk["rsi"] = 50.0
                if "vix" not in chunk.columns:
                    chunk["vix"] = 20.0

                critical_cols = [
                    "bid_size_level_1",
                    "ask_size_level_1",
                    "trade_frequency_1s",
                ]
                for col in critical_cols:
                    if col in chunk.columns and not pd.api.types.is_numeric_dtype(
                        chunk[col]
                    ):
                        error_msg = f"Colonne {col} non numérique: {chunk[col].dtype}"
                        miya_alerts(
                            error_msg,
                            tag="SIMULATION",
                            voice_profile="urgent",
                            priority=4,
                        )
                        AlertManager().send_alert(error_msg, priority=4)
                        send_telegram_alert(error_msg)
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    if col in chunk.columns:
                        non_scalar = [
                            val
                            for val in chunk[col]
                            if isinstance(val, (list, dict, tuple))
                        ]
                        if non_scalar:
                            error_msg = f"Colonne {col} contient des non-scalaires: {non_scalar[:5]}"
                            miya_alerts(
                                error_msg,
                                tag="SIMULATION",
                                voice_profile="urgent",
                                priority=4,
                            )
                            AlertManager().send_alert(error_msg, priority=4)
                            send_telegram_alert(error_msg)
                            logger.error(error_msg)
                            raise ValueError(error_msg)

                chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce")
                chunk = chunk.dropna(subset=["timestamp", "close"])
                chunk = chunk[chunk["close"] > 0]

                for _, row in chunk.iterrows():
                    regime = row["neural_regime"]
                    if pd.isna(regime) or not isinstance(regime, str):
                        regime = "trend"
                    else:
                        regime = regime.lower()

                    price = row["close"]
                    timestamp = row["timestamp"]
                    rsi = row["rsi"]
                    vix = row["vix"]

                    # Règles améliorées avec RSI et VIX
                    if regime == "trend" and rsi < 70 and vix < 30 and position <= 0:
                        if position == -1:
                            reward = (entry_price - price) * trade_size * capital
                            balance += reward
                            trades.append(
                                {
                                    "timestamp": timestamp,
                                    "action": "close_short",
                                    "price": price,
                                    "reward": reward,
                                    "balance": balance,
                                }
                            )
                        position = 1
                        entry_price = price
                        trades.append(
                            {
                                "timestamp": timestamp,
                                "action": "buy",
                                "price": price,
                                "reward": 0,
                                "balance": balance,
                            }
                        )
                    elif regime == "defensive" or rsi > 80 or vix > 35:
                        if position == 1:
                            reward = (price - entry_price) * trade_size * capital
                            balance += reward
                            trades.append(
                                {
                                    "timestamp": timestamp,
                                    "action": "sell",
                                    "price": price,
                                    "reward": reward,
                                    "balance": balance,
                                }
                            )
                        elif position == -1:
                            reward = (entry_price - price) * trade_size * capital
                            balance += reward
                            trades.append(
                                {
                                    "timestamp": timestamp,
                                    "action": "close_short",
                                    "price": price,
                                    "reward": reward,
                                    "balance": balance,
                                }
                            )
                        position = 0
                    elif regime == "range" and rsi > 60 and position >= 0:
                        if position == 1:
                            reward = (price - entry_price) * trade_size * capital
                            balance += reward
                            trades.append(
                                {
                                    "timestamp": timestamp,
                                    "action": "sell",
                                    "price": price,
                                    "reward": reward,
                                    "balance": balance,
                                }
                            )
                        position = -1
                        entry_price = price
                        trades.append(
                            {
                                "timestamp": timestamp,
                                "action": "short",
                                "price": price,
                                "reward": 0,
                                "balance": balance,
                            }
                        )

                self.save_snapshot(
                    "chunk",
                    {
                        "chunk_size": len(chunk),
                        "num_trades": len(trades),
                        "balance": balance,
                    },
                )

            if not trades:
                error_msg = "Aucun trade simulé"
                miya_alerts(
                    error_msg, tag="SIMULATION", voice_profile="urgent", priority=5
                )
                AlertManager().send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                raise ValueError(error_msg)

            df_trades = pd.DataFrame(trades)
            df_trades = df_trades.drop_duplicates(
                subset=["timestamp", "action"], keep="last"
            )

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df_trades.to_csv(output_path, index=False, encoding="utf-8")

            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                f"Simulation terminée: {len(df_trades)} trades, solde final {df_trades['balance'].iloc[-1]:.2f}",
                tag="SIMULATION",
                level="info",
                priority=3,
            )
            AlertManager().send_alert(
                f"Simulation terminée: {len(df_trades)} trades, solde final {df_trades['balance'].iloc[-1]:.2f}",
                priority=2,
            )
            logger.info(f"Simulation terminée: {len(df_trades)} trades")
            self.log_performance(
                "simulate_trading", latency, success=True, num_trades=len(df_trades)
            )
            self.save_snapshot(
                "simulate_trading",
                {
                    "num_trades": len(df_trades),
                    "final_balance": df_trades["balance"].iloc[-1],
                },
            )
            self.plot_simulation_metrics(df_trades)
            return df_trades
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur simulation: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="SIMULATION", voice_profile="urgent", priority=5)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "simulate_trading", latency, success=False, error=str(e)
            )
            self.save_snapshot("simulate_trading", {"error": str(e)})
            raise

    @with_retries(max_attempts=3, delay_base=2.0)
    def run_simulation(self, config: Dict) -> Dict:
        """
        Exécute la simulation de trading avec retries.

        Args:
            config (Dict): Configuration de la simulation.

        Returns:
            Dict: Statut de la simulation.
        """
        start_time = datetime.now()
        try:
            status = {
                "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "success": False,
                "trade_count": 0,
                "final_balance": 0.0,
                "errors": [],
            }
            features_path = config.get("features_path")
            output_path = config.get("output_path")
            capital = config.get("capital", 10000.0)
            trade_size = config.get("trade_size", 0.1)
            chunk_size = config.get("chunk_size", 10000)

            cached_df = self.load_cache(output_path, config.get("cache_hours", 24))
            if cached_df is not None and not cached_df.empty:
                status["success"] = True
                status["trade_count"] = len(cached_df)
                status["final_balance"] = (
                    cached_df["balance"].iloc[-1]
                    if "balance" in cached_df.columns
                    else 0.0
                )
                self.save_dashboard_status(status)
                miya_speak(
                    "Simulation ignorée, résultats chargés depuis cache",
                    tag="SIMULATION",
                    level="info",
                    priority=3,
                )
                AlertManager().send_alert(
                    "Simulation ignorée, résultats chargés depuis cache", priority=1
                )
                logger.info("Simulation ignorée, résultats chargés depuis cache")
                return status

            df = self.simulate_trading(
                features_path, output_path, capital, trade_size, chunk_size
            )
            status["success"] = True
            status["trade_count"] = len(df)
            status["final_balance"] = df["balance"].iloc[-1] if not df.empty else 0.0
            miya_speak(
                "Simulation terminée avec succès",
                tag="SIMULATION",
                level="info",
                priority=3,
            )
            AlertManager().send_alert("Simulation terminée avec succès", priority=1)
            logger.info("Simulation terminée avec succès")

            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "run_simulation",
                latency,
                success=status["success"],
                num_trades=status["trade_count"],
            )
            self.save_dashboard_status(status)
            self.save_snapshot("run_simulation", status)
            return status
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur run_simulation: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="SIMULATION", voice_profile="urgent", priority=5)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("run_simulation", latency, success=False, error=str(e))
            self.save_snapshot("run_simulation", {"error": str(e)})
            raise


def main():
    """
    Point d’entrée principal pour la simulation de trading.
    """
    try:
        manager = SimulationManager()
        config = manager.load_config()
        status = manager.run_simulation(config)
        if status["success"]:
            miya_speak(
                f"Simulation terminée: {status['trade_count']} trades, solde {status['final_balance']:.2f}",
                tag="SIMULATION",
                level="info",
                priority=3,
            )
            AlertManager().send_alert(
                f"Simulation terminée: {status['trade_count']} trades, solde {status['final_balance']:.2f}",
                priority=1,
            )
            logger.info(f"Simulation terminée: {status['trade_count']} trades")
        else:
            error_msg = "Échec simulation après retries"
            miya_alerts(error_msg, tag="SIMULATION", voice_profile="urgent", priority=5)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
    except Exception as e:
        error_msg = f"Erreur programme: {str(e)}\n{traceback.format_exc()}"
        miya_alerts(error_msg, tag="SIMULATION", voice_profile="urgent", priority=5)
        AlertManager().send_alert(error_msg, priority=4)
        send_telegram_alert(error_msg)
        logger.error(error_msg)
        exit(1)


if __name__ == "__main__":
    main()
