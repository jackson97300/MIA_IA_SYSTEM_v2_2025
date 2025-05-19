# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/model/strategy_discovery.py
# Découvre et adapte les stratégies de trading selon le contexte de marché pour MIA_IA_SYSTEM_v2_2025.
#
# Version : 2.1.5
# Date : 2025-05-15
#
# Rôle : Génère des clusters de patterns de marché (Phase 7), optimise les paramètres de trading via CMA-ES,
#        et adapte les stratégies en fonction du contexte de marché (Phase 12). Intègre des snapshots compressés,
#        des sauvegardes, des graphiques, et des alertes standardisées (Phase 8). Compatible avec l’ensemble learning
#        pour la sélection des modèles (Phase 16). Intègre les nouvelles fonctionnalités (bid_ask_imbalance, iv_skew, 
#        iv_term_structure, trade_aggressiveness, option_skew, news_impact_score) et optimise pour le HFT.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0
# - numpy>=1.26.4,<2.0.0
# - psutil>=5.9.8,<6.0.0
# - pyyaml>=6.0.0,<7.0.0
# - sklearn>=1.2.0,<2.0.0
# - cma>=3.2.0,<4.0.0
# - matplotlib>=3.7.0,<4.0.0
# - joblib>=1.2.0,<2.0.0 (pour optimisation parallèle)
# - shap>=0.41.0,<0.42.0 (pour analyse SHAP)
# - sqlite3, logging, os, json, yaml, hashlib, datetime, time, signal, gzip
# - src.model.utils.miya_console (version 2.1.5)
# - src.model.utils.config_manager (version 2.1.5)
# - src.model.utils.alert_manager (version 2.1.5)
# - src.model.utils.obs_template (version 2.1.5)
# - src.utils.telegram_alert (version 2.1.5)
# - src.envs.trading_env (version 2.1.5)
#
# Inputs :
# - config/es_config.yaml
# - config/feature_sets.yaml
# - data/market_memory.db
# - data/features/feature_importance.csv (pour validation SHAP)
# - Données de trading (pd.DataFrame avec 350 features pour entraînement ou 150 SHAP features pour inférence)
#
# Outputs :
# - data/logs/strategy_discovery.log
# - data/logs/strategy_discovery_performance.csv
# - data/strategy_discovery/snapshots/*.json.gz
# - data/strategies/*.json
# - data/clusters.csv
# - data/strategy_discovery_dashboard.json
# - data/checkpoints/strategy_discovery/*.json.gz
# - data/figures/strategy_discovery/*.png
#
# Notes :
# - Utilise IQFeed exclusivement pour les données d’entrée.
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence) définies dans config/feature_sets.yaml.
# - Supprime toute référence à 320 features, 81 features, et obs_t.
# - Implémente retries (max 3, délai adaptatif 2^attempt secondes) pour les opérations critiques.
# - Gère SIGINT avec snapshots compressés.
# - Sauvegardes incrémentielles (5 min), distribuées (15 min), versionnées (5 versions).
# - Génère des graphiques matplotlib des clusters avec métriques SHAP.
# - Tests unitaires dans tests/test_strategy_discovery.py (couvre bid_ask_imbalance, iv_skew, etc.).
# - Optimisé pour HFT avec validation dynamique, gestion mémoire, et intégration SHAP (Phase 17).
# - Maintient PCA à 15 dimensions pour préserver 95% de variance expliquée.

import gzip
import hashlib
import json
import logging
import os
import signal
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import cma
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import shap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.envs.trading_env import TradingEnv
from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import config_manager
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.model.utils.obs_template import validate_obs_t
from src.utils.telegram_alert import send_telegram_alert

# Configuration du logging
os.makedirs(os.path.join(BASE_DIR, "data", "logs", "strategy_discovery"), exist_ok=True)
logging.basicConfig(
    filename=os.path.join(
        BASE_DIR, "data", "logs", "strategy_discovery", "strategy_discovery.log"
    ),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CLUSTERS_PATH = os.path.join(BASE_DIR, "data", "clusters.csv")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "strategy_discovery", "snapshots")
STRATEGY_DIR = os.path.join(BASE_DIR, "data", "strategies")
DASHBOARD_PATH = os.path.join(BASE_DIR, "data", "strategy_discovery_dashboard.json")
CSV_LOG_PATH = os.path.join(
    BASE_DIR, "data", "logs", "strategy_discovery", "strategy_discovery_performance.csv"
)
CHECKPOINT_DIR = os.path.join(BASE_DIR, "data", "checkpoints", "strategy_discovery")
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures", "strategy_discovery")
FEATURE_IMPORTANCE_PATH = os.path.join(BASE_DIR, "data", "features", "feature_importance.csv")

# Constantes
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Secondes, exponentiel
MEMORY_THRESHOLD = 800  # MB, seuil pour réduire max_cache_size


class StrategyDiscovery:
    """
    Classe pour découvrir et adapter des stratégies de trading selon le contexte de marché.
    """

    def __init__(
        self,
        config_path: str = "config/es_config.yaml",
        db_path: str = "data/market_memory.db",
    ):
        """
        Initialise le module de découverte de stratégies.

        Args:
            config_path (str): Chemin vers la configuration.
            db_path (str): Chemin vers la base de données SQLite.
        """
        self.alert_manager = AlertManager()
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        os.makedirs(STRATEGY_DIR, exist_ok=True)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        signal.signal(signal.SIGINT, self.handle_sigint)
        self.last_checkpoint_time = datetime.now()
        self.last_distributed_checkpoint_time = datetime.now()

        try:
            self.config = self.load_config(config_path)
            self.db_path = db_path
            self.log_buffer = []
            self.data_cache = {}  # Cache pour les données validées
            self.max_cache_size = self.config.get("cache", {}).get("max_cache_size", 1000)

            # Valider la configuration
            required_config = ["clustering", "optimization", "features", "logging"]
            missing_config = [key for key in required_config if key not in self.config]
            if missing_config:
                raise ValueError(f"Clés de configuration manquantes: {missing_config}")

            miya_speak(
                "StrategyDiscovery initialisé",
                tag="STRATEGY_DISCOVERY",
                level="info",
                priority=2,
            )
            self.alert_manager.send_alert("StrategyDiscovery initialisé", priority=2)
            send_telegram_alert("StrategyDiscovery initialisé")
            logger.info("StrategyDiscovery initialisé avec succès")
            self.log_performance("init", 0, success=True)
            self.save_checkpoint(incremental=True)
        except Exception as e:
            error_msg = f"Erreur initialisation StrategyDiscovery: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="STRATEGY_DISCOVERY", level="error", priority=5)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            raise

    def handle_sigint(self, signum, frame):
        """Gère l'arrêt propre sur SIGINT."""
        snapshot = {"timestamp": datetime.now().isoformat(), "status": "SIGINT"}
        try:
            self.save_snapshot("sigint", snapshot)
            self.save_checkpoint(incremental=True)
            miya_speak(
                "Arrêt propre sur SIGINT, snapshot et checkpoint sauvegardés",
                tag="STRATEGY_DISCOVERY",
                level="info",
                priority=2,
            )
            self.alert_manager.send_alert(
                "Arrêt propre sur SIGINT, snapshot et checkpoint sauvegardés",
                priority=2,
            )
            send_telegram_alert(
                "Arrêt propre sur SIGINT, snapshot et checkpoint sauvegardés"
            )
            logger.info("Arrêt propre sur SIGINT, snapshot et checkpoint sauvegardés")
        except Exception as e:
            error_msg = f"Erreur sauvegarde SIGINT: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="STRATEGY_DISCOVERY", level="error", priority=3)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
        exit(0)

    def load_config(self, config_path: str) -> Dict:
        """
        Charge la configuration via config_manager avec repli sur les valeurs par défaut.

        Args:
            config_path (str): Chemin vers le fichier de configuration.

        Returns:
            Dict: Configuration chargée.
        """
        try:
            config = self.with_retries(
                lambda: config_manager.get_config(os.path.join(BASE_DIR, config_path))
            )
            if not config:
                raise ValueError("Configuration vide ou non trouvée")
            return config
        except Exception as e:
            error_msg = (
                f"Erreur chargement configuration: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="STRATEGY_DISCOVERY", level="error", priority=4)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            return {
                "clustering": {"n_clusters": 10, "random_state": 42},
                "optimization": {
                    "max_iterations": 100,
                    "entry_threshold_bounds": [20.0, 80.0],
                    "exit_threshold_bounds": [20.0, 80.0],
                    "position_size_bounds": [0.1, 1.0],
                    "stop_loss_bounds": [0.5, 5.0],
                },
                "features": {"observation_dims": {"training": 350, "inference": 150}},
                "logging": {"buffer_size": 100},
                "cache": {"max_cache_size": 1000},
            }

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
                    miya_alerts(
                        error_msg, tag="STRATEGY_DISCOVERY", level="error", priority=4
                    )
                    self.alert_manager.send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    logger.error(error_msg)
                    self.log_performance(
                        f"retry_attempt_{attempt+1}", 0, success=False, error=str(e)
                    )
                    return None
                delay = delay_base ** attempt
                logger.warning(f"Tentative {attempt+1} échouée, retry après {delay}s")
                time.sleep(delay)

    def validate_data(self, data: pd.DataFrame, training_mode: bool = True) -> None:
        """
        Valide que les données contiennent les features définies dans feature_sets.yaml et passent la validation obs_t.

        Args:
            data (pd.DataFrame): Données à valider.
            training_mode (bool): Mode entraînement (350 features) ou inférence (150 SHAP features).

        Raises:
            ValueError: Si les données ne respectent pas les exigences.
        """
        start_time = time.time()
        try:
            cache_key = hashlib.sha256(data.to_json().encode()).hexdigest()
            if cache_key in self.data_cache:
                miya_speak(
                    "Données validées depuis le cache",
                    tag="STRATEGY_DISCOVERY",
                    level="info",
                    priority=1,
                )
                self.alert_manager.send_alert(
                    "Données validées depuis le cache", priority=1
                )
                return

            # Vérifier la mémoire disponible
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
            if memory_usage > MEMORY_THRESHOLD:
                self.max_cache_size = max(100, self.max_cache_size // 2)
                logger.warning(
                    f"Utilisation mémoire élevée ({memory_usage:.2f} MB), réduction max_cache_size à {self.max_cache_size}"
                )

            # Charger les features attendues
            feature_sets = self.with_retries(
                lambda: config_manager.get_config(
                    os.path.join(BASE_DIR, "config", "feature_sets.yaml")
                )
            )
            expected_cols = feature_sets.get(
                "training_features" if training_mode else "shap_features", []
            )
            expected_count = self.config.get("features", {}).get(
                "observation_dims", {"training": 350, "inference": 150}
            )["training" if training_mode else "inference"]
            if len(expected_cols) != expected_count:
                error_msg = f"Nombre de features attendu incorrect dans feature_sets.yaml: {len(expected_cols)} au lieu de {expected_count}"
                raise ValueError(error_msg)

            # Vérifier les colonnes
            missing_cols = [col for col in expected_cols if col not in data.columns]
            if missing_cols:
                error_msg = f"Colonnes manquantes dans les données: {missing_cols}"
                raise ValueError(error_msg)
            if data.shape[1] != expected_count:
                error_msg = f"Nombre de features incorrect: {data.shape[1]} au lieu de {expected_count}"
                raise ValueError(error_msg)

            # Vérifier les colonnes critiques
            critical_cols = [
                "bid_size_level_1",
                "ask_size_level_1",
                "trade_frequency_1s",
                "spread_avg_1min",
                "close",
                "predicted_volatility",
                "neural_regime",
                "bid_ask_imbalance",
                "iv_skew",
                "iv_term_structure",
                "trade_aggressiveness",
                "option_skew",
                "news_impact_score",
            ]
            for col in critical_cols:
                if col in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        error_msg = (
                            f"Colonne {col} n'est pas numérique: {data[col].dtype}"
                        )
                        raise ValueError(error_msg)
                    if data[col].isna().any():
                        error_msg = f"Colonne {col} contient des valeurs NaN"
                        raise ValueError(error_msg)
                    if (
                        data[col]
                        .apply(lambda x: isinstance(x, (list, dict, tuple)))
                        .any()
                    ):
                        error_msg = f"Colonne {col} contient des valeurs non scalaires"
                        raise ValueError(error_msg)
                    if col in [
                        "bid_size_level_1",
                        "ask_size_level_1",
                        "trade_frequency_1s",
                    ] and data[col].min() < 0:
                        error_msg = f"Colonne {col} contient des valeurs négatives: {data[col].min()}"
                        raise ValueError(error_msg)
                    if col in ["bid_ask_imbalance", "trade_aggressiveness", "option_skew"]:
                        if not (-1 <= data[col].min() and data[col].max() <= 1):
                            error_msg = f"Colonne {col} hors plage [-1, 1]: min={data[col].min()}, max={data[col].max()}"
                            raise ValueError(error_msg)
                    if col == "iv_skew" and data[col].abs().max() > 0.5:
                        error_msg = f"Colonne iv_skew hors plage raisonnable: max abs={data[col].abs().max()}"
                        raise ValueError(error_msg)

            # Validation via obs_t
            if not validate_obs_t(data, context="strategy_discovery"):
                error_msg = "Échec de la validation obs_t pour les données"
                raise ValueError(error_msg)

            # Imputation des timestamps
            if "timestamp" in data.columns:
                data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
                if data["timestamp"].isna().any():
                    last_valid = (
                        data["timestamp"].dropna().iloc[-1]
                        if not data["timestamp"].dropna().empty
                        else pd.Timestamp.now()
                    )
                    miya_speak(
                        f"Valeurs de timestamp invalides détectées, imputées avec {last_valid}",
                        tag="STRATEGY_DISCOVERY",
                        level="warning",
                        priority=2,
                    )
                    self.alert_manager.send_alert(
                        f"Valeurs de timestamp invalides détectées, imputées avec {last_valid}",
                        priority=2,
                    )
                    send_telegram_alert(
                        f"Valeurs de timestamp invalides détectées, imputées avec {last_valid}"
                    )
                    data["timestamp"] = data["timestamp"].fillna(last_valid)

            self.data_cache[cache_key] = True
            if len(self.data_cache) > self.max_cache_size:
                self.data_cache.pop(next(iter(self.data_cache)))

            miya_speak(
                f"Données validées avec {len(data.columns)} features",
                tag="STRATEGY_DISCOVERY",
                level="info",
                priority=1,
            )
            self.alert_manager.send_alert(
                f"Données validées avec {len(data.columns)} features", priority=1
            )
            send_telegram_alert(f"Données validées avec {len(data.columns)} features")
            self.log_performance(
                "validate_data",
                time.time() - start_time,
                success=True,
                num_features=data.shape[1],
            )
            self.save_snapshot("validate_data", {"num_features": data.shape[1]})
        except Exception as e:
            error_msg = f"Erreur validation données: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="STRATEGY_DISCOVERY", level="error", priority=5)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "validate_data", time.time() - start_time, success=False, error=str(e)
            )
            raise

    def save_snapshot(
        self, snapshot_type: str, data: Any, output_dir: str = SNAPSHOT_DIR
    ) -> None:
        """
        Sauvegarde un instantané des résultats avec compression gzip.

        Args:
            snapshot_type (str): Type de snapshot (ex. : clusters, strategy_params).
            data (Any): Données à sauvegarder.
            output_dir (str): Répertoire des snapshots.
        """
        try:
            start_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = {
                "timestamp": timestamp,
                "type": snapshot_type,
                "data": data,
                "buffer_size": len(self.log_buffer),
            }
            path = os.path.join(
                output_dir, f"snapshot_{snapshot_type}_{timestamp}.json"
            )
            os.makedirs(output_dir, exist_ok=True)
            with gzip.open(f"{path}.gz", "wt", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=4)
            latency = time.time() - start_time
            miya_speak(
                f"Snapshot {snapshot_type} sauvegardé: {path}.gz",
                tag="STRATEGY_DISCOVERY",
                level="info",
                priority=1,
            )
            self.alert_manager.send_alert(
                f"Snapshot {snapshot_type} sauvegardé: {path}.gz", priority=1
            )
            send_telegram_alert(f"Snapshot {snapshot_type} sauvegardé: {path}.gz")
            logger.info(f"Snapshot {snapshot_type} sauvegardé: {path}.gz")
            self.log_performance("save_snapshot", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="STRATEGY_DISCOVERY", level="error", priority=3)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("save_snapshot", 0, success=False, error=str(e))

    def save_checkpoint(self, incremental: bool = True, distributed: bool = False):
        """Sauvegarde l’état du module (incrémentiel, distribué, versionné)."""
        try:
            start_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint = {
                "timestamp": timestamp,
                "log_buffer": self.log_buffer[-100:],  # Limiter la taille
                "data_cache": {k: True for k in self.data_cache},  # Simplifié
                "max_cache_size": self.max_cache_size,
            }
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR, f"checkpoint_{timestamp}.json.gz"
            )
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            with gzip.open(checkpoint_path, "wt", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=4)

            # Gestion des versions (max 5)
            checkpoints = sorted(Path(CHECKPOINT_DIR).glob("checkpoint_*.json.gz"))
            if len(checkpoints) > 5:
                for old_checkpoint in checkpoints[:-5]:
                    os.remove(old_checkpoint)

            # Sauvegarde distribuée (simulation, ex. : AWS S3)
            if distributed:
                # TODO: Implémenter la sauvegarde vers AWS S3
                logger.info(f"Sauvegarde distribuée simulée pour {checkpoint_path}")

            latency = time.time() - start_time
            miya_speak(
                f"Checkpoint {'incrémentiel' if incremental else 'distribué'} sauvegardé: {checkpoint_path}",
                tag="STRATEGY_DISCOVERY",
                level="info",
                priority=1,
            )
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
            miya_alerts(error_msg, tag="STRATEGY_DISCOVERY", level="error", priority=3)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("save_checkpoint", 0, success=False, error=str(e))

    def generate_cluster_plot(
        self, data: pd.DataFrame, labels: np.ndarray, n_clusters: int
    ):
        """Génère un graphique des clusters en utilisant PCA avec annotations SHAP."""
        try:
            start_time = time.time()
            # Réduire les dimensions avec PCA pour visualisation (2D)
            scaler = StandardScaler()
            feature_sets = self.with_retries(
                lambda: config_manager.get_config(
                    os.path.join(BASE_DIR, "config", "feature_sets.yaml")
                )
            )
            feature_cols = feature_sets.get("training_features", [])
            pattern_data = data[feature_cols]
            pattern_scaled = scaler.fit_transform(pattern_data)
            pca = PCA(n_components=2)
            pattern_pca = pca.fit_transform(pattern_scaled)

            # Calculer les importances SHAP pour les features critiques
            explainer = shap.KernelExplainer(
                lambda x: KMeans(n_clusters=n_clusters, random_state=42).fit_predict(x),
                pattern_scaled[:100]  # Limiter pour performance
            )
            shap_values = explainer.shap_values(pattern_scaled[:100])
            shap_importance = np.abs(shap_values).mean(axis=0)
            top_features = [
                feature_cols[i] for i in np.argsort(shap_importance)[-5:]
            ]

            # Créer le graphique
            plt.figure(figsize=(12, 8))
            for i in range(n_clusters):
                cluster_points = pattern_pca[labels == i]
                plt.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    label=f"Cluster {i}",
                    alpha=0.6,
                )
            plt.title(f"Visualisation des clusters (PCA) - Top SHAP: {', '.join(top_features)}")
            plt.xlabel("Composante principale 1")
            plt.ylabel("Composante principale 2")
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(
                FIGURES_DIR, f"clusters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            os.makedirs(FIGURES_DIR, exist_ok=True)
            plt.savefig(plot_path)
            plt.close()

            latency = time.time() - start_time
            miya_speak(
                f"Graphique des clusters sauvegardé: {plot_path}",
                tag="STRATEGY_DISCOVERY",
                level="info",
                priority=1,
            )
            self.alert_manager.send_alert(
                f"Graphique des clusters sauvegardé: {plot_path}", priority=1
            )
            send_telegram_alert(f"Graphique des clusters sauvegardé: {plot_path}")
            logger.info(f"Graphique des clusters sauvegardé: {plot_path}")
            self.log_performance("generate_cluster_plot", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur génération graphique des clusters: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="STRATEGY_DISCOVERY", level="error", priority=3)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "generate_cluster_plot", 0, success=False, error=str(e)
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
                miya_alerts(
                    error_msg, tag="STRATEGY_DISCOVERY", level="error", priority=5
                )
                self.alert_manager.send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
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
            self.log_buffer.append(log_entry)
            if len(self.log_buffer) >= self.config.get("logging", {}).get(
                "buffer_size", 100
            ):
                log_df = pd.DataFrame(self.log_buffer)
                os.makedirs(os.path.dirname(CSV_LOG_PATH), exist_ok=True)

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
                self.log_buffer = []
            logger.info(
                f"Performance journalisée pour {operation}. CPU: {cpu_percent}%"
            )
        except Exception as e:
            error_msg = (
                f"Erreur journalisation performance: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="STRATEGY_DISCOVERY", level="error", priority=3)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def update_dashboard(self, snapshot_type: str, metrics: Dict, data: Any) -> None:
        """
        Met à jour le dashboard pour mia_dashboard.py.

        Args:
            snapshot_type (str): Type de snapshot (ex. : clusters, strategy_params).
            metrics (Dict): Métriques à inclure (ex. : silhouette_score, sharpe).
            data (Any): Données associées (ex. : clusters, paramètres).
        """
        try:
            start_time = time.time()
            status = {
                "timestamp": datetime.now().isoformat(),
                "snapshot_type": snapshot_type,
                "metrics": metrics,
                "data_summary": str(data)[:1000],  # Limiter la taille
                "recent_errors": len(
                    [log for log in self.log_buffer if not log["success"]]
                ),
                "average_latency": (
                    sum(log["latency"] for log in self.log_buffer)
                    / len(self.log_buffer)
                    if self.log_buffer
                    else 0
                ),
                "max_cache_size": self.max_cache_size,
            }
            os.makedirs(os.path.dirname(DASHBOARD_PATH), exist_ok=True)

            def write_status():
                with open(DASHBOARD_PATH, "w", encoding="utf-8") as f:
                    json.dump(status, f, indent=4)

            self.with_retries(write_status)
            latency = time.time() - start_time
            miya_speak(
                "Tableau de bord StrategyDiscovery mis à jour",
                tag="STRATEGY_DISCOVERY",
                level="info",
                priority=1,
            )
            self.alert_manager.send_alert(
                "Tableau de bord StrategyDiscovery mis à jour", priority=1
            )
            send_telegram_alert("Tableau de bord StrategyDiscovery mis à jour")
            logger.info("Tableau de bord StrategyDiscovery mis à jour")
            self.log_performance("update_dashboard", latency, success=True)
        except Exception as e:
            error_msg = (
                f"Erreur mise à jour dashboard: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="STRATEGY_DISCOVERY", level="error", priority=3)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("update_dashboard", 0, success=False, error=str(e))

    def generate_clusters(
        self,
        data: pd.DataFrame,
        n_clusters: int = 10,
        output_path: str = CLUSTERS_PATH,
        training_mode: bool = True,
    ) -> pd.DataFrame:
        """
        Génère des clusters à partir des patterns de marché avec PCA et K-means, et les sauvegarde dans un fichier CSV.

        Args:
            data (pd.DataFrame): Données contenant les features.
            n_clusters (int): Nombre de clusters à générer (défaut : 10).
            output_path (str): Chemin du fichier CSV pour sauvegarder les clusters.
            training_mode (bool): Mode entraînement (350 features) ou inférence (150 SHAP features).

        Returns:
            pd.DataFrame: DataFrame contenant les informations des clusters.
        """
        start_time = time.time()
        try:
            self.validate_data(data, training_mode)
            if len(data) < n_clusters:
                error_msg = (
                    f"Pas assez de patterns pour clustering: {len(data)} < {n_clusters}"
                )
                raise ValueError(error_msg)

            # Charger les features
            feature_sets = self.with_retries(
                lambda: config_manager.get_config(
                    os.path.join(BASE_DIR, "config", "feature_sets.yaml")
                )
            )
            feature_cols = feature_sets.get(
                "training_features" if training_mode else "shap_features", []
            )

            # Préparer les données pour le clustering
            pattern_data = data[feature_cols].to_numpy(dtype=np.float32)  # Optimisation mémoire
            scaler = StandardScaler()
            pattern_scaled = scaler.fit_transform(pattern_data)

            # Appliquer PCA (15 dimensions, 95% de variance)
            pca = PCA(n_components=15)
            pattern_pca = pca.fit_transform(pattern_scaled)
            explained_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components = (
                np.argmax(explained_variance >= 0.95) + 1
                if np.any(explained_variance >= 0.95)
                else 15
            )
            pca = PCA(n_components=n_components)
            pattern_pca = pca.fit_transform(pattern_scaled)

            # Appliquer K-means
            clustering_config = self.config.get("clustering", {})
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=clustering_config.get("random_state", 42),
            )
            kmeans.fit(pattern_pca)
            labels = kmeans.labels_
            centroids = scaler.inverse_transform(
                pca.inverse_transform(kmeans.cluster_centers_)
            )

            # Calculer le score de silhouette
            silhouette = (
                silhouette_score(pattern_pca, labels)
                if len(np.unique(labels)) > 1
                else 0.0
            )

            # Calculer le régime dominant par cluster
            regimes = (
                data["neural_regime"].values
                if "neural_regime" in data.columns
                else np.zeros(len(data), dtype=int)
            )
            cluster_regimes = []
            for i in range(n_clusters):
                cluster_indices = np.where(labels == i)[0]
                if len(cluster_indices) > 0:
                    cluster_regime = pd.Series(regimes[cluster_indices]).mode()[0]
                else:
                    cluster_regime = -1
                cluster_regimes.append(cluster_regime)

            # Créer le DataFrame des clusters
            clusters = []
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for i, (centroid, size, regime) in enumerate(
                zip(centroids, np.bincount(labels), cluster_regimes)
            ):
                centroid_dict = {
                    col: float(val) for col, val in zip(feature_cols, centroid)
                }
                clusters.append(
                    {
                        "cluster_id": i,
                        "timestamp": timestamp,
                        "centroid": json.dumps(centroid_dict),
                        "cluster_size": int(size),
                        "regime": int(regime),
                    }
                )

            clusters_df = pd.DataFrame(clusters)

            # Sauvegarder dans le CSV
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            def save_clusters():
                clusters_df.to_csv(output_path, index=False, encoding="utf-8")

            self.with_retries(save_clusters)

            # Synchroniser avec market_memory.db
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS clusters (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        centroid TEXT,
                        cluster_size INTEGER,
                        regime INTEGER
                    )
                """
                )
                for _, row in clusters_df.iterrows():
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO clusters (timestamp, centroid, cluster_size, regime)
                        VALUES (?, ?, ?, ?)
                        """,
                        (row["timestamp"], row["centroid"], row["cluster_size"], row["regime"]),
                    )
                conn.commit()
                conn.close()
            except sqlite3.Error as e:
                error_msg = f"Erreur SQL synchronisation clusters: {str(e)}\n{traceback.format_exc()}"
                miya_alerts(
                    error_msg, tag="STRATEGY_DISCOVERY", level="error", priority=3
                )
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                logger.error(error_msg)

            # Sauvegarder un snapshot
            self.save_snapshot("clusters", clusters_df.to_dict())

            # Mettre à jour le dashboard
            self.update_dashboard(
                "clusters",
                {
                    "silhouette_score": silhouette,
                    "n_clusters": n_clusters,
                    "n_components_pca": n_components,
                },
                clusters_df.to_dict(),
            )

            # Générer un graphique
            self.generate_cluster_plot(data, labels, n_clusters)

            latency = time.time() - start_time
            miya_speak(
                f"Clusters générés: {n_clusters} clusters, score silhouette={silhouette:.2f}, sauvegardés dans {output_path}",
                tag="STRATEGY_DISCOVERY",
                level="info",
                priority=2,
            )
            self.alert_manager.send_alert(
                f"Clusters générés: {n_clusters} clusters, score silhouette={silhouette:.2f}, sauvegardés dans {output_path}",
                priority=2,
            )
            send_telegram_alert(
                f"Clusters générés: {n_clusters} clusters, score silhouette={silhouette:.2f}, sauvegardés dans {output_path}"
            )
            logger.info(
                f"Clusters générés: {n_clusters} clusters, score silhouette={silhouette:.2f}, sauvegardés dans {output_path}"
            )
            self.log_performance("generate_clusters", latency, success=True)
            return clusters_df

        except Exception as e:
            error_msg = (
                f"Erreur génération clusters: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="STRATEGY_DISCOVERY", level="error", priority=5)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "generate_clusters",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            raise

    def optimize_strategy(
        self,
        params: Dict[str, Any],
        data: pd.DataFrame,
        env: TradingEnv,
        max_iterations: int = 100,
        training_mode: bool = True,
    ) -> Dict[str, Any]:
        """
        Optimise les paramètres d'une stratégie de trading en fonction des performances historiques.

        Args:
            params (Dict[str, Any]): Paramètres initiaux de la stratégie.
            data (pd.DataFrame): Données historiques.
            env (TradingEnv): Environnement de trading.
            max_iterations (int): Nombre maximum d'itérations.
            training_mode (bool): Mode entraînement (350 features) ou inférence (150 SHAP features).

        Returns:
            Dict[str, Any]: Paramètres optimisés.
        """
        start_time = time.time()
        try:
            self.validate_data(data, training_mode)
            required_params = [
                "entry_threshold",
                "exit_threshold",
                "position_size",
                "stop_loss",
            ]
            missing_params = [p for p in required_params if p not in params]
            if missing_params:
                error_msg = f"Paramètres manquants: {missing_params}"
                raise ValueError(error_msg)

            optimization_config = self.config.get("optimization", {})
            bounds = {
                "entry_threshold": optimization_config.get(
                    "entry_threshold_bounds", [20.0, 80.0]
                ),
                "exit_threshold": optimization_config.get(
                    "exit_threshold_bounds", [20.0, 80.0]
                ),
                "position_size": optimization_config.get(
                    "position_size_bounds", [0.1, 1.0]
                ),
                "stop_loss": optimization_config.get("stop_loss_bounds", [0.5, 5.0]),
            }

            def objective_function(param_values):
                current_params = {
                    "entry_threshold": param_values[0],
                    "exit_threshold": param_values[1],
                    "position_size": param_values[2],
                    "stop_loss": param_values[3],
                }
                env.data = data
                obs, _ = env.reset()
                iteration_rewards = []
                for step in range(len(data)):
                    action = 0.0
                    if (
                        "rsi_14" in data.columns
                        and data["rsi_14"].iloc[step]
                        > current_params["entry_threshold"]
                    ):
                        action = current_params["position_size"]
                        # Ajuster l'action avec bid_ask_imbalance et trade_aggressiveness
                        if (
                            "bid_ask_imbalance" in data.columns
                            and "trade_aggressiveness" in data.columns
                        ):
                            action *= (
                                1
                                + 0.1 * data["bid_ask_imbalance"].iloc[step]
                                + 0.1 * data["trade_aggressiveness"].iloc[step]
                            )
                    elif (
                        "rsi_14" in data.columns
                        and data["rsi_14"].iloc[step] < current_params["exit_threshold"]
                    ):
                        action = -current_params["position_size"]
                    obs, reward, done, _, info = env.step(np.array([action]))
                    # Ajuster la récompense avec news_impact_score
                    if "news_impact_score" in data.columns:
                        news_score = data["news_impact_score"].iloc[step]
                        reward *= 1.3 if news_score > 0.5 and reward > 0 else 0.7
                    iteration_rewards.append(reward)
                    if done:
                        break
                rewards = np.array(iteration_rewards)
                sharpe = rewards.mean() / rewards.std() if rewards.std() != 0 else 0.0
                return -sharpe  # CMA-ES minimise, donc négatif pour maximiser Sharpe

            initial_guess = [
                params["entry_threshold"],
                params["exit_threshold"],
                params["position_size"],
                params["stop_loss"],
            ]
            bounds_list = [bounds[p] for p in required_params]

            def run_optimization():
                optimizer = cma.CMAEvolutionStrategy(
                    initial_guess,
                    0.5,
                    {"bounds": bounds_list, "maxiter": max_iterations},
                )
                optimizer.optimize(objective_function)
                return optimizer

            optimizer = self.with_retries(run_optimization)
            optimized_values = optimizer.result.xbest
            best_sharpe = -optimizer.result.fbest

            optimized_params = {
                "entry_threshold": optimized_values[0],
                "exit_threshold": optimized_values[1],
                "position_size": optimized_values[2],
                "stop_loss": optimized_values[3],
            }

            output_path = os.path.join(
                STRATEGY_DIR,
                f"strategy_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )
            os.makedirs(STRATEGY_DIR, exist_ok=True)

            def save_params():
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(optimized_params, f, indent=4)

            self.with_retries(save_params)

            self.save_snapshot("strategy_params", optimized_params)
            self.update_dashboard(
                "strategy_params",
                {"sharpe": best_sharpe, "iterations": max_iterations},
                optimized_params,
            )

            latency = time.time() - start_time
            miya_speak(
                f"Stratégie optimisée: Sharpe={best_sharpe:.2f}, paramètres sauvegardés dans {output_path}",
                tag="STRATEGY_DISCOVERY",
                level="info",
                priority=2,
            )
            self.alert_manager.send_alert(
                f"Stratégie optimisée: Sharpe={best_sharpe:.2f}, paramètres sauvegardés dans {output_path}",
                priority=2,
            )
            send_telegram_alert(
                f"Stratégie optimisée: Sharpe={best_sharpe:.2f}, paramètres sauvegardés dans {output_path}"
            )
            logger.info(
                f"Stratégie optimisée: Sharpe={best_sharpe:.2f}, paramètres sauvegardés dans {output_path}"
            )
            self.log_performance("optimize_strategy", latency, success=True)
            return optimized_params

        except Exception as e:
            error_msg = (
                f"Erreur optimisation stratégie: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="STRATEGY_DISCOVERY", level="error", priority=5)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "optimize_strategy",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            raise

    def adapt_strategy(
        self,
        data: pd.DataFrame,
        context: Dict[str, Any],
        env: TradingEnv,
        training_mode: bool = True,
    ) -> Dict[str, Any]:
        """
        Adapte une stratégie en fonction du contexte actuel du marché.

        Args:
            data (pd.DataFrame): Données récentes.
            context (Dict[str, Any]): Contexte du marché.
            env (TradingEnv): Environnement de trading.
            training_mode (bool): Mode entraînement (350 features) ou inférence (150 SHAP features).

        Returns:
            Dict[str, Any]: Paramètres adaptés.
        """
        start_time = time.time()
        try:
            self.validate_data(data, training_mode)
            required_context = ["neural_regime", "predicted_volatility"]
            missing_context = [c for c in required_context if c not in context]
            if missing_context:
                error_msg = f"Contexte manquant: {missing_context}"
                raise ValueError(error_msg)

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

            params = {
                "entry_threshold": (
                    70.0
                    if context["neural_regime"] == 0
                    else 60.0 if context["neural_regime"] == 1 else 50.0
                ),
                "exit_threshold": (
                    30.0
                    if context["neural_regime"] == 0
                    else 40.0 if context["neural_regime"] == 1 else 50.0
                ),
                "position_size": 0.5 / max(1.0, context["predicted_volatility"]),
                "stop_loss": 2.0 * context["predicted_volatility"],
            }

            optimized_params = self.optimize_strategy(
                params, data, env, max_iterations=50, training_mode=training_mode
            )

            self.save_snapshot("adapted_strategy", optimized_params)
            self.update_dashboard(
                "adapted_strategy",
                {
                    "neural_regime": context["neural_regime"],
                    "predicted_volatility": context["predicted_volatility"],
                },
                optimized_params,
            )

            latency = time.time() - start_time
            miya_speak(
                f"Stratégie adaptée pour neural_regime={context['neural_regime']}, volatilité={context['predicted_volatility']:.2f}",
                tag="STRATEGY_DISCOVERY",
                level="info",
                priority=2,
            )
            self.alert_manager.send_alert(
                f"Stratégie adaptée pour neural_regime={context['neural_regime']}, volatilité={context['predicted_volatility']:.2f}",
                priority=2,
            )
            send_telegram_alert(
                f"Stratégie adaptée pour neural_regime={context['neural_regime']}, volatilité={context['predicted_volatility']:.2f}"
            )
            logger.info(
                f"Stratégie adaptée pour neural_regime={context['neural_regime']}, volatilité={context['predicted_volatility']:.2f}"
            )
            self.log_performance("adapt_strategy", latency, success=True)
            return optimized_params

        except Exception as e:
            error_msg = (
                f"Erreur adaptation stratégie: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="STRATEGY_DISCOVERY", level="error", priority=5)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "adapt_strategy", time.time() - start_time, success=False, error=str(e)
            )
            raise


if __name__ == "__main__":
    try:
        discovery = StrategyDiscovery()
        # Simuler des données
        feature_sets = config_manager.get_config(
            os.path.join(BASE_DIR, "config", "feature_sets.yaml")
        )
        feature_cols = feature_sets.get("training_features", [])
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-05-15 09:00", periods=100, freq="1min"
                ),
                **{
                    col: (
                        np.random.uniform(0, 1, 100)
                        if col
                        not in [
                            "neural_regime",
                            "predicted_volatility",
                            "cnn_pressure",
                            "rsi_14",
                            "bid_ask_imbalance",
                            "iv_skew",
                            "iv_term_structure",
                            "trade_aggressiveness",
                            "option_skew",
                            "news_impact_score",
                        ]
                        else (
                            np.random.randint(0, 3, 100)
                            if col == "neural_regime"
                            else (
                                np.random.uniform(0, 2, 100)
                                if col == "predicted_volatility"
                                else (
                                    np.random.uniform(-5, 5, 100)
                                    if col == "cnn_pressure"
                                    else (
                                        np.random.uniform(20, 80, 100)
                                        if col == "rsi_14"
                                        else (
                                            np.random.uniform(-1, 1, 100)
                                            if col
                                            in [
                                                "bid_ask_imbalance",
                                                "trade_aggressiveness",
                                                "option_skew",
                                            ]
                                            else (
                                                np.random.uniform(-0.5, 0.5, 100)
                                                if col == "iv_skew"
                                                else (
                                                    np.random.uniform(0, 0.05, 100)
                                                    if col == "iv_term_structure"
                                                    else np.random.uniform(
                                                        -1, 1, 100
                                                    )
                                                    if col == "news_impact_score"
                                                    else np.random.uniform(0, 1, 100)
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                    for col in feature_cols
                },
            }
        )

        env = TradingEnv(config_path="config/es_config.yaml")

        clusters_df = discovery.generate_clusters(data, n_clusters=10)
        print("Clusters générés:")
        print(clusters_df.head())

        initial_params = {
            "entry_threshold": 70.0,
            "exit_threshold": 30.0,
            "position_size": 0.5,
            "stop_loss": 2.0,
        }
        optimized_params = discovery.optimize_strategy(initial_params, data, env)
        print("Paramètres optimisés:")
        print(optimized_params)

        context = {"neural_regime": 0, "predicted_volatility": 1.5}
        adapted_params = discovery.adapt_strategy(data, context, env)
        print("Paramètres adaptés:")
        print(adapted_params)

    except Exception as e:
        error_msg = f"Erreur test principal: {str(e)}\n{traceback.format_exc()}"
        alert_manager = AlertManager()
        miya_alerts(error_msg, tag="STRATEGY_DISCOVERY", level="error", priority=5)
        alert_manager.send_alert(error_msg, priority=5)
        send_telegram_alert(error_msg)
        logger.error(error_msg)
        raise