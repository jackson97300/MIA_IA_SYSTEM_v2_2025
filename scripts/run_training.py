# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/scripts/run_training.py
# Script pour entraîner le modèle SAC pour MIA_IA_SYSTEM_v2_2025.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Exécute l'entraînement SAC avec journalisation, snapshots compressés, et alertes.
#        Conforme à la Phase 8 (auto-conscience via alertes), Phase 10 (entraînement des politiques SAC),
#        et Phase 16 (ensemble learning).
#
# Dépendances : pandas>=2.0.0, pyyaml>=6.0.0, logging, os, torch>=2.0.0, numpy>=1.23.0, datetime, time,
#               json, psutil>=5.9.8, matplotlib>=3.7.0, typing, gzip, traceback,
#               src.envs.trading_env, src.model.adaptive_learning, src.model.utils.config_manager,
#               src.model.utils.alert_manager, src.model.utils.miya_console,
#               src.utils.telegram_alert, src.utils.standard
#
# Inputs : config/es_config.yaml, data/features/features_latest_filtered.csv, config/feature_sets.yaml
#
# Outputs : data/logs/run_training.log, data/logs/run_training_performance.csv,
#           data/models/sac_model.pth, data/training_dashboard.json,
#           data/training_snapshots/snapshot_*.json.gz,
#           data/figures/training/training_loss_*.png, data/figures/training/training_reward_*.png
#
# Notes :
# - Gère 350 features pour l’entraînement et 150 SHAP features pour l’inférence.
# - Utilise IQFeed via TradingEnv (indirectement).
# - Implémente retries (max 3, délai 2^attempt), logs psutil, snapshots JSON compressés, alertes centralisées.
# - Tests unitaires dans tests/test_run_training.py.

import gzip
import json
import logging
import os
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from src.envs.trading_env import TradingEnv
from src.model.adaptive_learning import retrain_with_recent_trades
from src.model.utils.alert_manager import AlertManager
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.utils.standard import with_retries
from src.utils.telegram_alert import send_telegram_alert

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "es_config.yaml")
FEATURES_PATH = os.path.join(
    BASE_DIR, "data", "features", "features_latest_filtered.csv"
)
MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "sac_model.pth")
DASHBOARD_PATH = os.path.join(BASE_DIR, "data", "training_dashboard.json")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "training_snapshots")
CSV_LOG_PATH = os.path.join(BASE_DIR, "data", "logs", "run_training_performance.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures", "training")

# Configuration du logging
os.makedirs(os.path.join(BASE_DIR, "data", "logs"), exist_ok=True)
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "data", "logs", "run_training.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [Training] %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Configuration par défaut
DEFAULT_CONFIG = {
    "training": {
        "features_path": FEATURES_PATH,
        "model_path": MODEL_PATH,
        "epochs": 100,
        "batch_size": 64,
        "learning_rate": 0.0003,
        "buffer_size": 10000,
        "chunk_size": 10000,
        "cache_hours": 24,
        "retry_attempts": 3,
        "retry_delay": 5,
        "timeout_seconds": 7200,
        "max_cache_size": 1000,
        "num_features": 350,
        "shap_features": 150,
    }
}


class TrainingManager:
    """
    Classe pour gérer l'entraînement SAC avec journalisation, snapshots compressés, et alertes.
    """

    def __init__(self):
        """
        Initialise le gestionnaire d'entraînement.
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
                "TrainingManager initialisé",
                tag="TRAINING",
                voice_profile="calm",
                priority=2,
            )
            AlertManager().send_alert("TrainingManager initialisé", priority=1)
            logger.info("TrainingManager initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config_path": CONFIG_PATH})
        except Exception as e:
            error_msg = f"Erreur initialisation TrainingManager: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="TRAINING", voice_profile="urgent", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            self.config = DEFAULT_CONFIG["training"]
            self.buffer_size = 100
            self.max_cache_size = 1000

    @with_retries(max_attempts=3, delay_base=2.0)
    def load_config(self, config_path: str = CONFIG_PATH) -> Dict:
        """
        Charge les paramètres de l'entraînement depuis es_config.yaml.

        Args:
            config_path (str): Chemin vers le fichier de configuration.

        Returns:
            Dict: Configuration chargée.
        """
        start_time = datetime.now()
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            if "training" not in config:
                raise ValueError("Clé 'training' manquante dans la configuration")
            required_keys = [
                "features_path",
                "model_path",
                "epochs",
                "batch_size",
                "learning_rate",
                "num_features",
                "shap_features",
            ]
            missing_keys = [
                key for key in required_keys if key not in config["training"]
            ]
            if missing_keys:
                raise ValueError(f"Clés manquantes dans 'training': {missing_keys}")
            if config["training"]["num_features"] != 350:
                raise ValueError(
                    f"Nombre de features incorrect: {config['training']['num_features']} != 350"
                )
            if config["training"]["shap_features"] != 150:
                raise ValueError(
                    f"Nombre de SHAP features incorrect: {config['training']['shap_features']} != 150"
                )
            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                "Configuration entraînement chargée",
                tag="TRAINING",
                level="info",
                priority=2,
            )
            AlertManager().send_alert("Configuration entraînement chargée", priority=1)
            logger.info("Configuration entraînement chargée")
            self.log_performance("load_config", latency, success=True)
            self.save_snapshot("load_config", {"config_path": config_path})
            return config["training"]
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur chargement config: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="TRAINING", voice_profile="urgent", priority=4)
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
            operation (str): Nom de l’opération (ex. : train_sac).
            latency (float): Latence en secondes.
            success (bool): Succès de l’opération.
            error (str, optional): Message d’erreur si échec.
            **kwargs: Paramètres supplémentaires (ex. : num_epochs).
        """
        try:
            memory_usage = (
                psutil.Process().memory_info().rss / 1024 / 1024
            )  # Mémoire en Mo
            cpu_percent = psutil.cpu_percent()
            if memory_usage > 1024:
                error_msg = f"ALERT: Utilisation mémoire élevée ({memory_usage:.2f} MB)"
                miya_alerts(error_msg, tag="TRAINING", level="error", priority=5)
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
            miya_alerts(error_msg, tag="TRAINING", level="error", priority=3)
            AlertManager().send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def save_snapshot(self, snapshot_type: str, data: Dict) -> None:
        """
        Sauvegarde un instantané des résultats avec compression gzip.

        Args:
            snapshot_type (str): Type de snapshot (ex. : train_sac).
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
                tag="TRAINING",
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
            miya_alerts(error_msg, tag="TRAINING", level="error", priority=3)
            AlertManager().send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("save_snapshot", 0, success=False, error=str(e))

    def save_dashboard_status(
        self, status: Dict, status_file: str = DASHBOARD_PATH
    ) -> None:
        """
        Sauvegarde l'état de l'entraînement pour mia_dashboard.py.

        Args:
            status (Dict): État de l'entraînement.
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
                tag="TRAINING",
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
            miya_alerts(error_msg, tag="TRAINING", voice_profile="urgent", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "save_dashboard_status", 0, success=False, error=str(e)
            )

    @with_retries(max_attempts=3, delay_base=2.0)
    def load_cache(
        self, model_path: str, cache_hours: int, expected_input_dim: int
    ) -> Optional[Dict]:
        """
        Charge le modèle en cache si disponible, valide et compatible avec expected_input_dim.

        Args:
            model_path (str): Chemin du modèle.
            cache_hours (int): Durée de validité du cache (heures).
            expected_input_dim (int): Dimension d'entrée attendue.

        Returns:
            Optional[Dict]: État du modèle chargé, ou None si invalide.
        """
        start_time = datetime.now()
        try:
            if os.path.exists(model_path):
                mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
                if (datetime.now() - mtime).total_seconds() / 3600 < cache_hours:
                    state_dict = torch.load(
                        model_path, map_location=torch.device("cpu")
                    )
                    if state_dict["actor.0.weight"].shape[1] != expected_input_dim:
                        logger.warning(
                            f"Modèle en cache incompatible: input_dim={state_dict['actor.0.weight'].shape[1]} != {expected_input_dim}"
                        )
                        miya_speak(
                            f"Modèle en cache incompatible, suppression: {model_path}",
                            tag="TRAINING",
                            level="warning",
                            priority=2,
                        )
                        AlertManager().send_alert(
                            f"Modèle en cache incompatible, suppression: {model_path}",
                            priority=2,
                        )
                        os.remove(model_path)
                        return None
                    latency = (datetime.now() - start_time).total_seconds()
                    miya_speak(
                        f"Modèle chargé depuis cache: {model_path}",
                        tag="TRAINING",
                        level="info",
                        priority=2,
                    )
                    AlertManager().send_alert(
                        f"Modèle chargé depuis cache: {model_path}", priority=1
                    )
                    logger.info(f"Modèle chargé depuis cache: {model_path}")
                    self.log_performance("load_cache", latency, success=True)
                    self.save_snapshot(
                        "load_cache",
                        {"model_path": model_path, "input_dim": expected_input_dim},
                    )
                    return state_dict
            return None
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur chargement cache modèle: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="TRAINING", voice_profile="urgent", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("load_cache", latency, success=False, error=str(e))
            self.save_snapshot("load_cache", {"error": str(e)})
            return None

    def plot_training_metrics(
        self,
        losses: List[float],
        rewards: List[float],
        epochs: List[int],
        output_dir: str = FIGURES_DIR,
    ) -> None:
        """
        Génère des graphiques pour les métriques d'entraînement (perte, récompense).

        Args:
            losses (List[float]): Historique des pertes.
            rewards (List[float]): Historique des récompenses moyennes.
            epochs (List[int]): Liste des époques.
            output_dir (str): Répertoire pour sauvegarder les graphiques.
        """
        start_time = datetime.now()
        try:
            if not losses or not rewards:
                miya_speak(
                    "Aucune donnée à visualiser",
                    tag="TRAINING",
                    level="warning",
                    priority=2,
                )
                AlertManager().send_alert("Aucune donnée à visualiser", priority=2)
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(output_dir, exist_ok=True)

            # Courbe de perte
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, losses, label="Perte", color="red")
            plt.title("Perte au Fil des Époques")
            plt.xlabel("Époque")
            plt.ylabel("Perte")
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"training_loss_{timestamp}.png"))
            plt.close()

            # Courbe de récompense
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, rewards, label="Récompense Moyenne", color="blue")
            plt.title("Récompense Moyenne au Fil des Époques")
            plt.xlabel("Époque")
            plt.ylabel("Récompense")
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"training_reward_{timestamp}.png"))
            plt.close()

            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                f"Graphiques d'entraînement générés: {output_dir}",
                tag="TRAINING",
                level="info",
                priority=2,
            )
            AlertManager().send_alert(
                f"Graphiques d'entraînement générés: {output_dir}", priority=1
            )
            logger.info(f"Graphiques d'entraînement générés: {output_dir}")
            self.log_performance(
                "plot_training_metrics", latency, success=True, num_epochs=len(epochs)
            )
            self.save_snapshot(
                "plot_training_metrics",
                {"output_dir": output_dir, "num_epochs": len(epochs)},
            )
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur génération graphiques: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="TRAINING", level="error", priority=3)
            AlertManager().send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "plot_training_metrics", latency, success=False, error=str(e)
            )
            self.save_snapshot("plot_training_metrics", {"error": str(e)})

    def validate_env(self, env: TradingEnv) -> None:
        """
        Valide l’environnement TradingEnv.

        Args:
            env (TradingEnv): Environnement de trading.

        Raises:
            ValueError: Si l’environnement est invalide.
        """
        start_time = datetime.now()
        try:
            if not isinstance(env, TradingEnv):
                error_msg = "Environnement invalide: doit être TradingEnv"
                logger.error(error_msg)
                raise ValueError(error_msg)
            if env.observation_space.shape[0] != self.config["num_features"]:
                error_msg = f"Dimension d’observation incorrecte: {env.observation_space.shape[0]} != {self.config['num_features']}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            if env.action_space.n != 3:
                error_msg = f"Nombre d’actions incorrect: {env.action_space.n} != 3"
                logger.error(error_msg)
                raise ValueError(error_msg)
            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                "Environnement TradingEnv validé",
                tag="TRAINING",
                level="info",
                priority=2,
            )
            AlertManager().send_alert("Environnement TradingEnv validé", priority=1)
            logger.info("Environnement TradingEnv validé")
            self.log_performance("validate_env", latency, success=True)
            self.save_snapshot(
                "validate_env",
                {
                    "observation_dim": env.observation_space.shape[0],
                    "action_dim": env.action_space.n,
                },
            )
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur validation environnement: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="TRAINING", voice_profile="urgent", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("validate_env", latency, success=False, error=str(e))
            self.save_snapshot("validate_env", {"error": str(e)})
            raise

    @with_retries(max_attempts=3, delay_base=2.0)
    def train_sac(
        self, features_path: str, model_path: str, config: Dict, env: TradingEnv
    ) -> Tuple[Dict, float]:
        """
        Entraîne un modèle SAC sur les features.

        Args:
            features_path (str): Chemin vers les features.
            model_path (str): Chemin pour sauvegarder le modèle.
            config (Dict): Configuration de l'entraînement.
            env (TradingEnv): Environnement de trading.

        Returns:
            Tuple[Dict, float]: État du modèle et récompense moyenne.
        """
        start_time = datetime.now()
        try:
            config["num_features"]
            miya_speak(
                "Démarrage entraînement SAC", tag="TRAINING", level="info", priority=3
            )
            AlertManager().send_alert("Démarrage entraînement SAC", priority=2)
            logger.info("Démarrage entraînement SAC")

            if not os.path.exists(features_path):
                error_msg = f"Fichier features introuvable: {features_path}"
                miya_alerts(
                    error_msg, tag="TRAINING", voice_profile="urgent", priority=5
                )
                AlertManager().send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Charger données par chunks
            dtype = {
                "timestamp": str,
                "close": float,
                "sentiment_label": str,
                "bid_size_level_1": float,
                "ask_size_level_1": float,
                "trade_frequency_1s": float,
            }
            chunk_size = config.get("chunk_size", 10000)
            chunks = pd.read_csv(
                features_path, encoding="utf-8", dtype=dtype, chunksize=chunk_size
            )
            df = pd.concat([chunk for chunk in chunks])

            if "timestamp" not in df.columns or "close" not in df.columns:
                error_msg = (
                    f"Colonnes 'timestamp' ou 'close' manquantes dans {features_path}"
                )
                miya_alerts(
                    error_msg, tag="TRAINING", voice_profile="urgent", priority=5
                )
                AlertManager().send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                raise ValueError(error_msg)

            critical_cols = [
                "bid_size_level_1",
                "ask_size_level_1",
                "trade_frequency_1s",
            ]
            for col in critical_cols:
                if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                    error_msg = f"Colonne {col} non numérique: {df[col].dtype}"
                    miya_alerts(
                        error_msg, tag="TRAINING", voice_profile="urgent", priority=4
                    )
                    AlertManager().send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                if col in df.columns:
                    non_scalar = [
                        val for val in df[col] if isinstance(val, (list, dict, tuple))
                    ]
                    if non_scalar:
                        error_msg = f"Colonne {col} contient des non-scalaires: {non_scalar[:5]}"
                        miya_alerts(
                            error_msg,
                            tag="TRAINING",
                            voice_profile="urgent",
                            priority=4,
                        )
                        AlertManager().send_alert(error_msg, priority=4)
                        send_telegram_alert(error_msg)
                        logger.error(error_msg)
                        raise ValueError(error_msg)

            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp", "close"])
            logger.info(f"Colonnes initiales du DataFrame: {list(df.columns)}")

            duplicated_cols = df.columns[df.columns.duplicated()].tolist()
            if duplicated_cols:
                logger.warning(f"Colonnes dupliquées détectées: {duplicated_cols}")
                miya_speak(
                    f"Colonnes dupliquées: {duplicated_cols}",
                    tag="TRAINING",
                    level="warning",
                    priority=2,
                )
                AlertManager().send_alert(
                    f"Colonnes dupliquées: {duplicated_cols}", priority=2
                )
                df = df.loc[:, ~df.columns.duplicated()]
                logger.info(
                    f"Colonnes après suppression des doublons: {list(df.columns)}"
                )

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
                    error_msg, tag="TRAINING", voice_profile="urgent", priority=5
                )
                AlertManager().send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                raise ValueError(error_msg)

            key_columns = ["rsi", "sentiment_label"]
            [col for col in key_columns if col in df.columns]
            missing_key_cols = [col for col in key_columns if col not in df.columns]
            if missing_key_cols:
                logger.warning(f"Colonnes clés manquantes: {missing_key_cols}")
                miya_speak(
                    f"Colonnes clés manquantes: {missing_key_cols}",
                    tag="TRAINING",
                    level="warning",
                    priority=2,
                )
                AlertManager().send_alert(
                    f"Colonnes clés manquantes: {missing_key_cols}", priority=2
                )

            training_cols = [
                col
                for col in training_features
                if col in df.columns and col != "sentiment_label"
            ]
            missing_cols = [
                col for col in training_features if col not in training_cols
            ]
            if missing_cols:
                logger.warning(
                    f"Colonnes manquantes dans training_features: {missing_cols}"
                )
                miya_speak(
                    f"Colonnes manquantes dans training_features: {missing_cols}",
                    tag="TRAINING",
                    level="warning",
                    priority=2,
                )
                AlertManager().send_alert(
                    f"Colonnes manquantes dans training_features: {missing_cols}",
                    priority=2,
                )
                for col in missing_cols:
                    df[col] = 0.0
                training_cols.extend(missing_cols)
            df = df[training_cols].fillna(0)
            df = df.reindex(columns=training_features, fill_value=0)
            logger.info(
                f"Colonnes après reindex avec training_features: {list(df.columns)}"
            )

            if df.shape[1] != len(training_features):
                error_msg = f"Nombre de colonnes incorrect: {df.shape[1]} != {len(training_features)}"
                miya_alerts(
                    error_msg, tag="TRAINING", voice_profile="urgent", priority=5
                )
                AlertManager().send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                raise ValueError(error_msg)
            unexpected_cols = set(df.columns) - set(training_features)
            if unexpected_cols:
                logger.warning(f"Colonnes inattendues: {unexpected_cols}")
                miya_speak(
                    f"Colonnes inattendues: {unexpected_cols}",
                    tag="TRAINING",
                    level="warning",
                    priority=2,
                )
                AlertManager().send_alert(
                    f"Colonnes inattendues: {unexpected_cols}", priority=2
                )

            if df.empty:
                error_msg = "Aucune donnée valide pour l'entraînement"
                miya_alerts(
                    error_msg, tag="TRAINING", voice_profile="urgent", priority=5
                )
                AlertManager().send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                raise ValueError(error_msg)

            input_dim = len(training_features)
            logger.info(f"input_dim défini à: {input_dim}")

            if os.path.exists(model_path):
                try:
                    state_dict = torch.load(
                        model_path, map_location=torch.device("cpu")
                    )
                    if (
                        state_dict.get("actor.0.weight", torch.empty(0, 0)).shape[1]
                        != input_dim
                    ):
                        os.remove(model_path)
                        logger.info(
                            f"Modèle supprimé pour incompatibilité: {state_dict['actor.0.weight'].shape[1]} != {input_dim}"
                        )
                        miya_speak(
                            f"Modèle supprimé pour incompatibilité: {input_dim}",
                            tag="TRAINING",
                            level="info",
                            priority=2,
                        )
                        AlertManager().send_alert(
                            f"Modèle supprimé pour incompatibilité: {input_dim}",
                            priority=1,
                        )
                except Exception as e:
                    logger.warning(
                        f"Impossible de charger/supprimer modèle existant: {e}"
                    )
                    miya_speak(
                        f"Impossible de charger/supprimer modèle existant: {e}",
                        tag="TRAINING",
                        level="warning",
                        priority=2,
                    )
                    AlertManager().send_alert(
                        f"Impossible de charger/supprimer modèle existant: {e}",
                        priority=2,
                    )

            action_dim = 3
            epochs = config.get("epochs", 100)
            batch_size = config.get("batch_size", 64)
            learning_rate = config.get("learning_rate", 0.0003)

            if os.path.exists(model_path):
                os.remove(model_path)
                logger.info(f"Cache de modèle supprimé: {model_path}")
                miya_speak(
                    f"Cache de modèle supprimé: {model_path}",
                    tag="TRAINING",
                    level="info",
                    priority=2,
                )
                AlertManager().send_alert(
                    f"Cache de modèle supprimé: {model_path}", priority=1
                )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = SimpleSAC(input_dim, action_dim).to(device)
            logger.info(
                f"Modèle SimpleSAC initialisé avec input_dim={input_dim} sur {device}"
            )
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()

            buffer = []
            total_rewards = []
            losses = []
            epoch_list = []

            for epoch in range(epochs):
                miya_speak(
                    f"Époque {epoch+1}/{epochs}",
                    tag="TRAINING",
                    level="info",
                    priority=2,
                )
                AlertManager().send_alert(f"Époque {epoch+1}/{epochs}", priority=1)
                logger.info(f"Époque {epoch+1}/{epochs}")

                for i in range(0, len(df), batch_size):
                    batch = df.iloc[i : i + batch_size]
                    features = batch[
                        [col for col in training_features if col in batch.columns]
                    ]
                    features = features.reindex(columns=training_features, fill_value=0)
                    states = torch.tensor(features.values, dtype=torch.float32).to(
                        device
                    )
                    prices = batch["close"].values

                    if states.shape[1] != input_dim:
                        error_msg = f"Incohérence de dimensions: states.shape[1]={states.shape[1]} != input_dim={input_dim}"
                        miya_alerts(
                            error_msg,
                            tag="TRAINING",
                            voice_profile="urgent",
                            priority=5,
                        )
                        AlertManager().send_alert(error_msg, priority=4)
                        send_telegram_alert(error_msg)
                        logger.error(error_msg)
                        raise ValueError(error_msg)

                    probs = model(states)
                    actions = torch.multinomial(probs, 1).squeeze()

                    rewards = []
                    for j in range(len(actions)):
                        action = actions[j].item()
                        price = prices[j]
                        next_price = prices[j + 1] if j + 1 < len(prices) else price
                        if action == 0:  # Buy
                            reward = (next_price - price) * 0.1
                        elif action == 1:  # Hold
                            reward = 0
                        else:  # Sell
                            reward = (price - next_price) * 0.1
                        rewards.append(reward)

                    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                    total_rewards.append(rewards.mean().item())

                    buffer.append((states, actions, rewards))
                    if len(buffer) > config.get("buffer_size", 10000):
                        buffer.pop(0)

                    if len(buffer) >= batch_size:
                        batch_states, batch_actions, batch_rewards = zip(
                            *buffer[-batch_size:]
                        )
                        batch_states = torch.cat(batch_states)
                        batch_actions = torch.cat(batch_actions)
                        batch_rewards = torch.cat(batch_rewards)

                        critic_input = torch.cat(
                            [batch_states, batch_actions.unsqueeze(1).float()], dim=1
                        )
                        q_values = model.critic(critic_input)
                        loss = criterion(q_values.squeeze(), batch_rewards)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        losses.append(loss.item())

                epoch_list.append(epoch + 1)
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(model.state_dict(), model_path)
                miya_speak(
                    f"Checkpoint sauvegardé: époque {epoch+1}",
                    tag="TRAINING",
                    level="info",
                    priority=2,
                )
                AlertManager().send_alert(
                    f"Checkpoint sauvegardé: époque {epoch+1}", priority=1
                )
                self.save_snapshot(
                    "epoch",
                    {
                        "epoch": epoch + 1,
                        "loss": losses[-1] if losses else 0.0,
                        "avg_reward": (
                            np.mean(total_rewards[-batch_size:])
                            if total_rewards
                            else 0.0
                        ),
                    },
                )
                self.plot_training_metrics(
                    losses,
                    [np.mean(total_rewards[-batch_size:]) if total_rewards else 0.0]
                    * len(losses),
                    epoch_list,
                )

            avg_reward = np.mean(total_rewards) if total_rewards else 0.0
            miya_speak(
                f"Entraînement initial terminé: récompense moyenne {avg_reward:.2f}",
                tag="TRAINING",
                level="info",
                priority=3,
            )
            AlertManager().send_alert(
                f"Entraînement initial terminé: récompense moyenne {avg_reward:.2f}",
                priority=2,
            )
            logger.info(
                f"Entraînement initial terminé: récompense moyenne {avg_reward}"
            )

            trade_log_path = os.path.join(
                BASE_DIR, "data", "trades", "trades_simulated.csv"
            )
            model = retrain_with_recent_trades(
                model,
                env,
                trade_log_path=trade_log_path,
                feature_sets_path=os.path.join(BASE_DIR, "config", "feature_sets.yaml"),
                timesteps=10000,
                min_trades=100,
                min_confidence=0.7,
                batch_size=batch_size,
            )
            torch.save(model.state_dict(), model_path)
            miya_speak(
                f"Réentraînement adaptatif terminé, modèle sauvegardé: {model_path}",
                tag="TRAINING",
                level="info",
                priority=3,
            )
            AlertManager().send_alert(
                f"Réentraînement adaptatif terminé, modèle sauvegardé: {model_path}",
                priority=2,
            )
            logger.info(f"Réentraînement adaptatif terminé: {model_path}")

            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "train_sac",
                latency,
                success=True,
                num_epochs=epochs,
                avg_reward=avg_reward,
            )
            self.save_snapshot(
                "train_sac",
                {"epochs": epochs, "avg_reward": avg_reward, "model_path": model_path},
            )
            return model.state_dict(), avg_reward
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.warning("Mémoire CUDA insuffisante, repli sur CPU")
                miya_alerts(
                    "Mémoire CUDA insuffisante, repli sur CPU",
                    tag="TRAINING",
                    voice_profile="urgent",
                    priority=4,
                )
                AlertManager().send_alert(
                    "Mémoire CUDA insuffisante, repli sur CPU", priority=4
                )
                send_telegram_alert("Mémoire CUDA insuffisante, repli sur CPU")
                device = torch.device("cpu")
                model = SimpleSAC(input_dim, action_dim).to(device)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                return self.train_sac(features_path, model_path, config, env)
            else:
                latency = (datetime.now() - start_time).total_seconds()
                error_msg = (
                    f"Erreur entraînement SAC: {str(e)}\n{traceback.format_exc()}"
                )
                miya_alerts(
                    error_msg, tag="TRAINING", voice_profile="urgent", priority=5
                )
                AlertManager().send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                self.log_performance("train_sac", latency, success=False, error=str(e))
                self.save_snapshot("train_sac", {"error": str(e)})
                raise
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur entraînement SAC: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="TRAINING", voice_profile="urgent", priority=5)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("train_sac", latency, success=False, error=str(e))
            self.save_snapshot("train_sac", {"error": str(e)})
            raise

    @with_retries(max_attempts=3, delay_base=2.0)
    def run_training(
        self, config: Dict, env: TradingEnv, live_mode: bool = False
    ) -> Dict:
        """
        Exécute l’entraînement SAC avec retries.

        Args:
            config (Dict): Configuration de l'entraînement.
            env (TradingEnv): Environnement de trading.
            live_mode (bool): Mode live (non utilisé ici).

        Returns:
            Dict: Statut de l'entraînement.
        """
        start_time = datetime.now()
        try:
            self.validate_env(env)
            status = {
                "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "success": False,
                "epochs_completed": 0,
                "avg_reward": 0.0,
                "errors": [],
            }
            features_path = config.get("features_path")
            model_path = config.get("model_path")
            epochs = config.get("epochs", 100)

            expected_input_dim = config["num_features"]
            cached_model = self.load_cache(
                model_path, config.get("cache_hours", 24), expected_input_dim
            )
            if cached_model is not None:
                status["success"] = True
                status["epochs_completed"] = epochs
                status["avg_reward"] = 0.0
                self.save_dashboard_status(status)
                miya_speak(
                    "Entraînement ignoré, modèle chargé depuis cache",
                    tag="TRAINING",
                    level="info",
                    priority=3,
                )
                AlertManager().send_alert(
                    "Entraînement ignoré, modèle chargé depuis cache", priority=1
                )
                logger.info("Entraînement ignoré, modèle chargé depuis cache")
                return status

            _, avg_reward = self.train_sac(features_path, model_path, config, env)
            status["success"] = True
            status["epochs_completed"] = epochs
            status["avg_reward"] = avg_reward
            miya_speak(
                "Entraînement terminé avec succès",
                tag="TRAINING",
                level="info",
                priority=3,
            )
            AlertManager().send_alert("Entraînement terminé avec succès", priority=1)
            logger.info("Entraînement terminé avec succès")

            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "run_training",
                latency,
                success=status["success"],
                num_epochs=status["epochs_completed"],
                avg_reward=status["avg_reward"],
            )
            self.save_dashboard_status(status)
            self.save_snapshot("run_training", status)
            return status
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur run_training: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="TRAINING", voice_profile="urgent", priority=5)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("run_training", latency, success=False, error=str(e))
            self.save_snapshot("run_training", {"error": str(e)})
            raise


class SimpleSAC(nn.Module):
    """
    Modèle SAC simplifié pour l'entraînement.
    """

    def __init__(self, input_dim: int, action_dim: int = 3):
        super(SimpleSAC, self).__init__()
        logger.info(f"Initialisation de SimpleSAC avec input_dim={input_dim}")
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor(state)


def main():
    """
    Point d’entrée principal pour l’entraînement SAC.
    """
    try:
        manager = TrainingManager()
        config = manager.load_config()
        env = TradingEnv(config_path=CONFIG_PATH)
        env.policy_type = "mlp"
        status = manager.run_training(config, env)
        if status["success"]:
            miya_speak(
                f"Entraînement terminé: {status['epochs_completed']} époques, récompense {status['avg_reward']:.2f}",
                tag="TRAINING",
                level="info",
                priority=3,
            )
            AlertManager().send_alert(
                f"Entraînement terminé: {status['epochs_completed']} époques, récompense {status['avg_reward']:.2f}",
                priority=1,
            )
            logger.info(f"Entraînement terminé: {status['epochs_completed']} époques")
        else:
            error_msg = "Échec entraînement après retries"
            miya_alerts(error_msg, tag="TRAINING", voice_profile="urgent", priority=5)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
    except Exception as e:
        error_msg = f"Erreur programme: {str(e)}\n{traceback.format_exc()}"
        miya_alerts(error_msg, tag="TRAINING", voice_profile="urgent", priority=5)
        AlertManager().send_alert(error_msg, priority=4)
        send_telegram_alert(error_msg)
        logger.error(error_msg)
        exit(1)


if __name__ == "__main__":
    main()
