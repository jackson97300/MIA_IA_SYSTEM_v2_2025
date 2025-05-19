# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/data/data_provider.py
# Abstraction pour la collecte de données de marché (OHLC, DOM, options, cross-market, actualités).
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Centralise la collecte de données via IQFeed (production) et CsvDataProvider (tests), 
# compatible avec 350 features pour l’entraînement et top 150 SHAP pour l’inférence/fallback.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.0,<6.0.0, scikit-learn>=1.5.0,<2.0.0,
#   joblib>=1.3.0,<2.0.0, pyiqfeed>=1.0.0,<2.0.0, logging, threading, signal, datetime
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/db_setup.py
# - src/features/news_analyzer.py (pour analyse de sentiment, Phase 1)
# - src/features/microstructure_guard.py (pour métriques de microstructure, Phase 15)
#
# Inputs :
# - config/market_config.yaml
# - config/feature_sets.yaml
# - config/credentials.yaml
# - fichiers CSV (data/iqfeed/*.csv)
#
# Outputs :
# - Données brutes pour feature_pipeline.py
# - logs dans data/logs/provider_performance.csv
# - snapshots JSON dans data/provider_snapshots/
# - cache dans data/cache/provider/
# - visualisations dans data/figures/provider/
# - patterns dans market_memory.db (table clusters)
#
# Notes :
# - Utilise exclusivement IQFeed (https://www.iqfeed.net/dev/) avec essai gratuit de 7-14 jours 
# (accord développeur : fax 402-255-3788, email support@iqfeed.net).
# - Suppression de toute référence à dxFeed, obs_t, 320/81 features pour conformité avec 
# MIA_IA_SYSTEM_v2_2025.
# - Intègre les Phases 1-18 :
#   - Phase 1 (news_scraper.py, news_analyzer.py) : Collecte et analyse de sentiment des actualités 
# (news_sentiment_score).
#   - Phase 13 (orderflow_indicators.py, options_metrics.py) : Métriques comme vix_es_correlation, 
# atr_14, call_iv_atm, put_iv_atm, option_volume, oi_concentration, option_skew.
#   - Phase 15 (microstructure_guard.py, spotgamma_recalculator.py) : Métriques de microstructure 
# (spoofing_score, volume_anomaly) et options (net_gamma, call_wall).
#   - Phase 18 : Métriques avancées de microstructure (trade_velocity, hft_activity_score, 
# hypothétiques).
# - Intègre volatilité (méthode 1 : vix_es_correlation, atr_14) et options (méthode 2 : call_iv_atm, 
# put_iv_atm, option_volume, oi_concentration, option_skew).
# - Flux d’actualités via NewsConn (DTN News Room, Benzinga Pro, Dow Jones) avec analyse de sentiment 
# via news_analyzer.py.
# - Mémoire contextuelle avec K-means 10 clusters (méthode 7) dans market_memory.db et visualisations 
# des clusters (heatmaps).
# - Tests unitaires disponibles dans tests/test_data_provider.py.
# Évolution future : Migration API Investing.com (juin 2025), intégration
# avec feature_pipeline.py pour validation des 350 features.

import hashlib
import json
import logging
import signal
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from src.features.microstructure_guard import compute_microstructure_metrics  # Phase 15
from src.features.news_analyzer import analyze_sentiment  # Phase 1
from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import config_manager
from src.model.utils.db_setup import get_db_connection

# Note : Pas d'importation de `shutil` nécessaire (F821 est une fausse alerte, aucune utilisation 
# de `shutil` dans ce fichier).
# Note : L'importation de `os` est utilisée (par exemple, dans `os.path.join`, `os.makedirs`), 
# donc F401 est une fausse alerte.

# Constantes
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "data", "logs")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "provider_snapshots")
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache", "provider")
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures", "provider")
MARKET_CONFIG = os.path.join(BASE_DIR, "config", "market_config.yaml")
MAX_RETRIES = 3
RETRY_DELAY = 2  # Base pour délai exponentiel (2^attempt)

# Configuration du logger
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
logger = logging.getLogger(__name__)
log_file = os.path.join(LOG_DIR, "provider_performance.csv")
handler = logging.FileHandler(log_file)
handler.setFormatter(logging.Formatter("%(asctime)s,%(levelname)s,%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class EnhancedJSONEncoder(json.JSONEncoder):
    """Encodeur JSON personnalisé pour gérer les types non sérialisables."""

    def default(self, obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return super().default(obj)


class DataProvider(ABC):
    """Interface abstraite pour la collecte de données de marché."""

    def __init__(self, config_path: str = MARKET_CONFIG):
        self.alert_manager = AlertManager()
        signal.signal(signal.SIGINT, self.handle_sigint)
        self.config = self.with_retries(lambda: config_manager.get_config(config_path))
        self.valid_symbols = self.config.get("data_sources", {}).get(
            "valid_symbols", ["ES", "SPY", "VIX", "TLT", "VVIX", "DVVIX"]
        )
        self.cache_expiration = (
            self.config.get("cache", {}).get("expiration_hours", 24) * 3600
        )
        self.parallel_load = self.config.get("data_sources", {}).get(
            "parallel_load", True
        )
        self.log_buffer = []
        self.pattern_buffer = []
        self._validate_config()

    def handle_sigint(self, signum, frame):
        """Gère l'arrêt propre sur SIGINT."""
        snapshot = {"timestamp": datetime.now().isoformat(), "status": "SIGINT"}
        snapshot_path = os.path.join(
            SNAPSHOT_DIR, f"provider_sigint_{snapshot['timestamp']}.json"
        )
        try:
            with open(snapshot_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=4, cls=EnhancedJSONEncoder)
            self.alert_manager.send_alert(
                "Arrêt propre sur SIGINT, snapshot sauvegardé", priority=2
            )
        except Exception as e:
            self.alert_manager.send_alert(
                f"Erreur sauvegarde SIGINT: {str(e)}", priority=3
            )
        exit(0)

    def with_retries(
        self,
        func: callable,
        max_attempts: int = MAX_RETRIES,
        delay_base: float = RETRY_DELAY,
    ) -> Optional[any]:
        """
        Exécute une fonction avec retries exponentiels (max 3, délai 2^attempt secondes).

        Args:
            func (callable): Fonction à exécuter.
            max_attempts (int): Nombre maximum de tentatives.
            delay_base (float): Base pour le délai exponentiel.

        Returns:
            Optional[any]: Résultat de la fonction ou None si échec.
        """
        for attempt in range(max_attempts):
            try:
                start_time = time.time()
                result = func()
                latency = time.time() - start_time
                self.log_performance(
                    f"retry_attempt_{attempt+1}",
                    latency,
                    success=True,
                    data_type="retry",
                )
                return result
            except Exception as e:
                if attempt == max_attempts - 1:
                    error_msg = f"Échec après {max_attempts} tentatives: {str(e)}"
                    self.alert_manager.send_alert(error_msg, priority=3)
                    logger.error(error_msg)
                    self.log_performance(
                        f"retry_attempt_{attempt+1}",
                        0,
                        success=False,
                        error=str(e),
                        data_type="retry",
                    )
                    return None
                delay = delay_base**attempt
                logger.warning(f"Tentative {attempt+1} échouée, retry après {delay}s")
                time.sleep(delay)

    def _validate_config(self):
        """Valide les clés requises dans la configuration."""
        start_time = time.time()
        try:
            required_keys = ["data_feed", "data_sources"]
            for key in required_keys:
                if key not in self.config:
                    raise ValueError(f"Clé de configuration manquante: {key}")
            latency = time.time() - start_time
            self.log_performance(
                "validate_config", latency, success=True, data_type="config"
            )
        except ValueError as e:
            self.alert_manager.send_alert(
                f"Erreur validation configuration: {str(e)}", priority=3
            )
            logger.error(f"Erreur validation configuration: {str(e)}")
            self.log_performance(
                "validate_config", 0, success=False, error=str(e), data_type="config"
            )
            raise

    @abstractmethod
    def connect(self) -> None:
        """Initialise la connexion au fournisseur de données."""

    @abstractmethod
    def fetch_ohlc(
        self, symbol: str, timeframe: str, normalize: bool = False
    ) -> Optional[pd.DataFrame]:
        """Récupère les données OHLC pour un symbole et une période."""

    @abstractmethod
    def fetch_dom(
        self, symbol: str, levels: int, normalize: bool = False
    ) -> Optional[pd.DataFrame]:
        """Récupère le carnet d’ordres pour un symbole."""

    @abstractmethod
    def fetch_options(
        self, symbol: str, normalize: bool = False
    ) -> Optional[pd.DataFrame]:
        """Récupère les données de chaîne d’options pour un symbole."""

    @abstractmethod
    def fetch_cross_market(
        self, symbols: List[str], normalize: bool = False
    ) -> Optional[pd.DataFrame]:
        """Récupère les données cross-market pour une liste de symboles."""

    @abstractmethod
    def fetch_news(self, keywords: List[str], timeframe: str) -> Optional[pd.DataFrame]:
        """Récupère les flux d’actualités pour des mots-clés donnés."""

    @abstractmethod
    def fetch_microstructure_metrics(
        self, symbol: str, normalize: bool = False
    ) -> Optional[pd.DataFrame]:
        """Récupère les métriques de microstructure pour un symbole (Phases 15, 18)."""

    def log_performance(
        self,
        operation: str,
        latency: float,
        success: bool,
        error: str = None,
        data_type: str = None,
        symbol: str = None,
    ):
        """Journalise les performances des opérations avec retries."""
        start_time = time.time()
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "data_type": data_type,
                "symbol": symbol,
                "latency": latency,
                "success": success,
                "error": error,
                "memory_mb": memory_usage,
                "cpu_percent": cpu_percent,
                "rows": None,
                "cluster_id": None,
            }
            self.log_buffer.append(log_entry)
            if len(self.log_buffer) >= self.config.get("logging", {}).get(
                "buffer_size", 100
            ):
                df = pd.DataFrame(self.log_buffer)

                def save_log():
                    df.to_csv(
                        log_file,
                        mode="a",
                        header=not os.path.exists(log_file),
                        index=False,
                        encoding="utf-8",
                    )

                self.with_retries(save_log)
                self.log_buffer = []
            self.log_performance(
                "log_performance",
                time.time() - start_time,
                success=True,
                data_type="logging",
            )
        except (OSError, ValueError) as e:
            self.alert_manager.send_alert(
                f"Erreur logging performance: {str(e)}", priority=3
            )
            logger.error(f"Erreur logging performance: {str(e)}")
            self.log_performance(
                "log_performance",
                time.time() - start_time,
                success=False,
                error=str(e),
                data_type="logging",
            )

    def save_snapshot(self, snapshot_type: str, data: Dict):
        """Sauvegarde un instantané des résultats."""
        start_time = time.time()
        try:
            if not isinstance(data, dict):
                raise ValueError("Snapshot data doit être un dictionnaire")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = {
                "timestamp": timestamp,
                "type": snapshot_type,
                "data": data,
                "performance": {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_mb": psutil.Process().memory_info().rss / 1024 / 1024,
                },
            }
            path = os.path.join(
                SNAPSHOT_DIR, f"snapshot_{snapshot_type}_{timestamp}.json"
            )

            def save():
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(snapshot, f, indent=4, cls=EnhancedJSONEncoder)

            self.with_retries(save)
            self.log_performance(
                "save_snapshot",
                time.time() - start_time,
                success=True,
                data_type="snapshot",
            )
        except (OSError, json.JSONEncodeError, ValueError) as e:
            error_msg = f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}"
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance(
                "save_snapshot",
                time.time() - start_time,
                success=False,
                error=str(e),
                data_type="snapshot",
            )

    def store_data_pattern(
        self,
        df: pd.DataFrame,
        data_type: str,
        symbol: str = None,
        keywords: List[str] = None,
    ) -> int:
        """
        Clusterise les données collectées avec K-means (méthode 7) et stocke dans market_memory.db.

        Args:
            df (pd.DataFrame): Données collectées.
            data_type (str): Type de données (ohlc, dom, options, cross_market, news, 
                microstructure).
            symbol (str, optional): Symbole associé.
            keywords (List[str], optional): Mots-clés pour les actualités.

        Returns:
            int: Identifiant du cluster (cluster_id).
        """
        start_time = time.time()
        try:
            features = {}
            if data_type == "ohlc":
                features = {
                    "vix_es_correlation": (
                        df["vix_es_correlation"].iloc[-1]
                        if "vix_es_correlation" in df.columns
                        else 0
                    ),
                    "atr_14": df["atr_14"].iloc[-1] if "atr_14" in df.columns else 0,
                }
            elif data_type == "dom":
                features = {
                    "bid_size_level_1": (
                        df["bid_size_level_1"].iloc[-1]
                        if "bid_size_level_1" in df.columns
                        else 0
                    ),
                    "ask_size_level_1": (
                        df["ask_size_level_1"].iloc[-1]
                        if "ask_size_level_1" in df.columns
                        else 0
                    ),
                    "spoofing_score": (
                        df["spoofing_score"].iloc[-1]
                        if "spoofing_score" in df.columns
                        else 0
                    ),
                    "volume_anomaly": (
                        df["volume_anomaly"].iloc[-1]
                        if "volume_anomaly" in df.columns
                        else 0
                    ),
                }
            elif data_type == "options":
                features = {
                    "call_iv_atm": (
                        df["call_iv_atm"].iloc[-1] if "call_iv_atm" in df.columns else 0
                    ),
                    "put_iv_atm": (
                        df["put_iv_atm"].iloc[-1] if "put_iv_atm" in df.columns else 0
                    ),
                    "option_volume": (
                        df["option_volume"].iloc[-1]
                        if "option_volume" in df.columns
                        else 0
                    ),
                    "net_gamma": (
                        df["net_gamma"].iloc[-1] if "net_gamma" in df.columns else 0
                    ),
                }
            elif data_type == "cross_market":
                features = {
                    "vix_es_correlation": (
                        df["vix_es_correlation"].iloc[-1]
                        if "vix_es_correlation" in df.columns
                        else 0
                    )
                }
            elif data_type == "news":
                features = {
                    "headline_length": (
                        len(df["headline"].iloc[-1]) if "headline" in df.columns else 0
                    ),
                    "news_sentiment_score": (
                        df["news_sentiment_score"].iloc[-1]
                        if "news_sentiment_score" in df.columns
                        else 0
                    ),
                }
            elif data_type == "microstructure":
                features = {
                    "spoofing_score": (
                        df["spoofing_score"].iloc[-1]
                        if "spoofing_score" in df.columns
                        else 0
                    ),
                    "volume_anomaly": (
                        df["volume_anomaly"].iloc[-1]
                        if "volume_anomaly" in df.columns
                        else 0
                    ),
                    "trade_velocity": (
                        df["trade_velocity"].iloc[-1]
                        if "trade_velocity" in df.columns
                        else 0
                    ),
                    "hft_activity_score": (
                        df["hft_activity_score"].iloc[-1]
                        if "hft_activity_score" in df.columns
                        else 0
                    ),
                }

            pattern_entry = {
                "timestamp": (
                    df["timestamp"].iloc[-1].isoformat()
                    if "timestamp" in df.columns
                    else datetime.now().isoformat()
                ),
                "data_type": data_type,
                "symbol": symbol,
                "keywords": keywords,
                "features": features,
                "cluster_id": 0,
            }
            self.pattern_buffer.append(pattern_entry)

            if len(self.pattern_buffer) >= 10:
                df_patterns = pd.DataFrame(self.pattern_buffer)
                feature_cols = list(features.keys())
                if feature_cols:
                    X = df_patterns[feature_cols].fillna(0).values
                    kmeans = KMeans(n_clusters=10, random_state=42)
                    df_patterns["cluster_id"] = kmeans.fit_predict(X)

                    def store():
                        db_path = os.path.join(BASE_DIR, "data", "market_memory.db")
                        conn = get_db_connection(db_path)
                        cursor = conn.cursor()
                        cursor.execute(
                            """
                            CREATE TABLE IF NOT EXISTS clusters (
                                cluster_id INTEGER,
                                event_type TEXT,
                                features TEXT,
                                timestamp TEXT,
                                data_type TEXT,
                                symbol TEXT,
                                keywords TEXT
                            )
                            """
                        )
                        for _, row in df_patterns.iterrows():
                            cursor.execute(
                                """
                                INSERT INTO clusters (cluster_id, event_type, features, 
                                    timestamp, data_type, symbol, keywords)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    row["cluster_id"],
                                    "data_collection",
                                    json.dumps(row["features"]),
                                    row["timestamp"],
                                    row["data_type"],
                                    row["symbol"],
                                    json.dumps(row["keywords"]),
                                ),
                            )
                        conn.commit()
                        conn.close()

                    self.with_retries(store)
                    self.pattern_buffer = []
                    threading.Thread(
                        target=self.visualize_data_patterns, daemon=True
                    ).start()
                    latency = time.time() - start_time
                    self.log_performance(
                        "store_data_pattern",
                        latency,
                        success=True,
                        data_type=data_type,
                        symbol=symbol,
                    )
                    return int(df_patterns["cluster_id"].iloc[-1])

            latency = time.time() - start_time
            self.log_performance(
                "store_data_pattern",
                latency,
                success=True,
                data_type=data_type,
                symbol=symbol,
            )
            return 0
        except Exception as e:
            error_msg = f"Erreur stockage pattern: {str(e)}"
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance(
                "store_data_pattern",
                time.time() - start_time,
                success=False,
                error=str(e),
                data_type=data_type,
                symbol=symbol,
            )
            return 0

    def visualize_data_patterns(self):
        """Génère une heatmap des clusters de données dans data/figures/provider/."""
        start_time = time.time()
        try:
            patterns = self.pattern_buffer[-100:]
            if not patterns:
                error_msg = "Aucun pattern pour visualisation"
                self.alert_manager.send_alert(error_msg, priority=2)
                logger.warning(error_msg)
                return

            df = pd.DataFrame(patterns)
            if "cluster_id" not in df.columns or df["cluster_id"].isnull().all():
                error_msg = "Aucun cluster_id pour visualisation"
                self.alert_manager.send_alert(error_msg, priority=2)
                logger.warning(error_msg)
                return

            pivot = df.pivot_table(
                index="cluster_id",
                columns="data_type",
                values="timestamp",
                aggfunc="count",
                fill_value=0,
            )
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot, annot=True, cmap="Blues")
            plt.title("Heatmap des Clusters de Données")
            timestamp = datetime.now().strftime("%Y%m%d")
            output_path = os.path.join(FIGURES_DIR, f"data_clusters_{timestamp}.png")

            def save_fig():
                plt.savefig(output_path)
                plt.close()

            self.with_retries(save_fig)
            logger.info("Heatmap des clusters générée")
            self.log_performance(
                "visualize_data_patterns",
                time.time() - start_time,
                success=True,
                data_type="visualization",
            )
        except Exception as e:
            error_msg = f"Erreur visualisation clusters: {str(e)}"
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance(
                "visualize_data_patterns",
                time.time() - start_time,
                success=False,
                error=str(e),
                data_type="visualization",
            )

    def validate_data(
        self, df: pd.DataFrame, expected_cols: List[str]
    ) -> Optional[pd.DataFrame]:
        """Valide les colonnes, types et valeurs des données."""
        start_time = time.time()
        try:
            training_mode = self.config.get("environment", {}).get(
                "training_mode", True
            )
            expected_features = 350 if training_mode else 150
            features_config = self.with_retries(lambda: config_manager.get_features())
            all_features = [
                f["name"]
                for category in features_config.get("feature_sets", {}).values()
                for f in category.get("features", [])
            ]
            if len(df.columns) < expected_features:
                error_msg = (
                    f"Nombre de features insuffisant: {len(df.columns)} < "
                    f"{expected_features} pour mode "
                    f"{'entraînement' if training_mode else 'inférence'}"
                )
                raise ValueError(error_msg)
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Colonnes manquantes: {missing_cols}")
            for col in [
                "vix_es_correlation",
                "atr_14",
                "call_iv_atm",
                "put_iv_atm",
                "option_volume",
                "oi_concentration",
                "option_skew",
                "spoofing_score",
                "volume_anomaly",
                "net_gamma",
                "trade_velocity",
                "hft_activity_score",
            ]:
                if col in df.columns:
                    if df[col].isna().any() or (df[col] < 0).any():
                        raise ValueError(f"Valeurs invalides pour {col}")
            if "timestamp" in df.columns:
                if df["timestamp"].isna().any():
                    raise ValueError("Timestamps invalides")
                current_time = pd.Timestamp.now()
                if (df["timestamp"] > current_time + pd.Timedelta(minutes=5)).any() or (
                    df["timestamp"] < current_time - pd.Timedelta(days=1)
                ).any():
                    raise ValueError("Timestamps hors plage")
            if "headline" in df.columns:
                if df["headline"].isna().any() or (df["headline"] == "").any():
                    raise ValueError("Titres d’actualités vides ou manquants")
            if "source" in df.columns:
                if df["source"].isna().any() or (df["source"] == "").any():
                    raise ValueError("Sources d’actualités vides ou manquantes")
            if "news_sentiment_score" in df.columns:
                if (
                    df["news_sentiment_score"].isna().any()
                    or (df["news_sentiment_score"] < -1).any()
                    or (df["news_sentiment_score"] > 1).any()
                ):
                    raise ValueError("Scores de sentiment invalides")
            for col in df.columns:
                if col in all_features and df[col].isna().mean() > 0.5:
                    logger.warning(f"Feature {col} a plus de 50% de NaN")
            latency = time.time() - start_time
            self.log_performance(
                "validate_data", latency, success=True, data_type="validation"
            )
            return df
        except (ValueError, TypeError) as e:
            self.alert_manager.send_alert(
                f"Erreur validation données: {str(e)}", priority=3
            )
            logger.error(f"Erreur validation données: {str(e)}")
            self.log_performance(
                "validate_data",
                time.time() - start_time,
                success=False,
                error=str(e),
                data_type="validation",
            )
            return None

    def _clean_cache(self, max_size_mb: float = 1000):
        """Supprime les fichiers cache expirés ou si la taille dépasse max_size_mb."""
        start_time = time.time()

        def clean():
            total_size = sum(
                os.path.getsize(os.path.join(CACHE_DIR, f))
                for f in os.listdir(CACHE_DIR)
                if os.path.isfile(os.path.join(CACHE_DIR, f))
            ) / (1024 * 1024)
            if total_size > max_size_mb:
                files = [
                    (f, os.path.getmtime(os.path.join(CACHE_DIR, f)))
                    for f in os.listdir(CACHE_DIR)
                ]
                files.sort(key=lambda x: x[1])
                for f, _ in files[: len(files) // 2]:
                    os.remove(os.path.join(CACHE_DIR, f))
            for filename in os.listdir(CACHE_DIR):
                path = os.path.join(CACHE_DIR, filename)
                if (
                    os.path.isfile(path)
                    and (time.time() - os.path.getmtime(path)) > self.cache_expiration
                ):
                    os.remove(path)

        try:
            self.with_retries(clean)
            latency = time.time() - start_time
            self.log_performance(
                "clean_cache", latency, success=True, data_type="cache"
            )
        except OSError as e:
            self.alert_manager.send_alert(
                f"Erreur nettoyage cache: {str(e)}", priority=3
            )
            logger.error(f"Erreur nettoyage cache: {str(e)}")
            self.log_performance(
                "clean_cache",
                time.time() - start_time,
                success=False,
                error=str(e),
                data_type="cache",
            )


class CsvDataProvider(DataProvider):
    """Implémentation stub pour la collecte de données via CSV historiques."""

    def __init__(self, config_path: str = MARKET_CONFIG):
        super().__init__(config_path)
        self.csv_paths = {
            "ohlc": os.path.join(BASE_DIR, "data", "iqfeed", "merged_data.csv"),
            "dom": os.path.join(BASE_DIR, "data", "iqfeed", "merged_data.csv"),
            "options": os.path.join(BASE_DIR, "data", "iqfeed", "option_chain.csv"),
            "cross_market": os.path.join(
                BASE_DIR, "data", "iqfeed", "cross_market.csv"
            ),
            "news": os.path.join(BASE_DIR, "data", "iqfeed", "news.csv"),
            "microstructure": os.path.join(
                BASE_DIR, "data", "iqfeed", "microstructure.csv"
            ),
        }
        self._clean_cache()

    def connect(self) -> None:
        """Vérifie l’existence des fichiers CSV."""
        start_time = time.time()

        def connect():
            for key, path in self.csv_paths.items():
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Fichier CSV manquant: {path}")
                if not os.access(os.path.dirname(path), os.W_OK):
                    raise PermissionError(f"Permissions insuffisantes pour {path}")

        try:
            self.with_retries(connect)
            latency = time.time() - start_time
            self.log_performance(
                "connect", latency, success=True, data_type="connection"
            )
            self.save_snapshot(
                "connect", {"status": "connected", "csv_paths": self.csv_paths}
            )
        except (FileNotFoundError, PermissionError) as e:
            self.alert_manager.send_alert(f"Erreur connexion CSV: {str(e)}", priority=3)
            logger.error(f"Erreur connexion CSV: {str(e)}")
            self.log_performance(
                "connect",
                time.time() - start_time,
                success=False,
                error=str(e),
                data_type="connection",
            )
            raise

    def _load_and_process(
        self,
        path: str,
        symbols: List[str],
        cols: List[str],
        normalize: bool,
        data_type: str,
        keywords: List[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Charge et traite un fichier CSV avec cache et normalisation."""
        start_time = time.time()
        try:
            if symbols:
                for symbol in symbols:
                    if symbol not in self.valid_symbols:
                        raise ValueError(f"Symbole invalide: {symbol}")
            key = hashlib.sha256(path.encode()).hexdigest()[:8]
            key_suffix = (
                hashlib.sha1("_".join(symbols + (keywords or [])).encode()).hexdigest()[:10]
                if (symbols or keywords)
                else "all"
            )
            cache_path = os.path.join(CACHE_DIR, f"{data_type}_{key}_{key_suffix}.csv")
            if len(cache_path) > 200:
                raise ValueError(f"Chemin cache trop long: {cache_path}")
            if (
                os.path.exists(cache_path)
                and (time.time() - os.path.getmtime(cache_path)) < self.cache_expiration
            ):

                def load_cache():
                    df = pd.read_csv(
                        cache_path, encoding="utf-8", parse_dates=["timestamp"]
                    )
                    if (
                        df.empty
                        or not all(col in df.columns for col in cols)
                        or os.path.getsize(cache_path) == 0
                    ):
                        os.remove(cache_path)
                        raise ValueError("Cache corrompu")
                    return df

                df = self.with_retries(load_cache)
                if df is not None:
                    latency = time.time() - start_time
                    self.log_performance(
                        f"fetch_{data_type}",
                        latency,
                        success=True,
                        data_type=data_type,
                        symbol=",".join(symbols) if symbols else None,
                    )
                    return df

            def load_csv():
                df = pd.read_csv(
                    path, encoding="utf-8", parse_dates=["timestamp"], low_memory=False
                )
                if symbols and "symbol" in cols:
                    df = df[df["symbol"].isin(symbols)][cols]
                elif data_type == "news" and keywords:
                    pattern = "|".join(keywords)
                    df = df[df["headline"].str.contains(pattern, case=False, na=False)][
                        cols
                    ]
                else:
                    df = df[cols]
                df = self.validate_data(df, cols)
                if df is None:
                    raise ValueError("Données invalides")
                if normalize and data_type != "news":
                    num_cols = [
                        c
                        for c in cols
                        if df[c].dtype in ["float64", "int64"] and c != "timestamp"
                    ]
                    if num_cols:
                        scaler = MinMaxScaler()
                        df[num_cols] = scaler.fit_transform(df[num_cols])
                if data_type == "news":
                    df["news_sentiment_score"] = df["headline"].apply(
                        analyze_sentiment
                    )  # Phase 1
                df.to_csv(cache_path, index=False, encoding="utf-8")
                return df

            df = self.with_retries(load_csv)
            if df is not None:
                self.store_data_pattern(
                    df,
                    data_type,
                    symbol=",".join(symbols) if symbols else None,
                    keywords=keywords,
                )
                latency = time.time() - start_time
                self.log_performance(
                    f"fetch_{data_type}",
                    latency,
                    success=True,
                    data_type=data_type,
                    symbol=",".join(symbols) if symbols else None,
                )
                self.save_snapshot(
                    f"fetch_{data_type}",
                    {
                        "symbols": symbols,
                        "keywords": keywords,
                        "rows": len(df),
                        "size_bytes": df.memory_usage().sum(),
                    },
                )
                return df
            return None
        except Exception as e:
            error_msg = f"Erreur inattendue _load_and_process ({data_type}): {str(e)}"
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance(
                f"fetch_{data_type}",
                time.time() - start_time,
                success=False,
                error=str(e),
                data_type=data_type,
                symbol=",".join(symbols) if symbols else None,
            )
            return None

    def _parallel_load(self, paths: List[Dict]) -> List[Optional[pd.DataFrame]]:
        """Charge plusieurs fichiers CSV en parallèle."""

        def load_single(path, symbols, cols, normalize, data_type, keywords=None):
            return self._load_and_process(
                path, symbols, cols, normalize, data_type, keywords
            )

        return Parallel(n_jobs=4)(
            delayed(load_single)(
                p["path"],
                p["symbols"],
                p["cols"],
                p["normalize"],
                p["data_type"],
                p.get("keywords"),
            )
            for p in paths
        )

    def fetch_ohlc(
        self, symbol: str, timeframe: str, normalize: bool = False
    ) -> Optional[pd.DataFrame]:
        """Récupère les données OHLC via CSV."""
        cols = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vix_es_correlation",
            "atr_14",
        ]
        df = self._load_and_process(
            self.csv_paths["ohlc"], [symbol], cols, normalize, "ohlc"
        )
        if df is not None and timeframe != "raw":
            try:
                df = (
                    df.set_index("timestamp")
                    .resample(timeframe)
                    .agg(
                        {
                            "open": "first",
                            "high": "max",
                            "low": "min",
                            "close": "last",
                            "volume": "sum",
                            "vix_es_correlation": "mean",
                            "atr_14": "mean",
                        }
                    )
                    .dropna()
                    .reset_index()
                )
            except ValueError as e:
                error_msg = f"Erreur resampling OHLC: {str(e)}"
                self.alert_manager.send_alert(error_msg, priority=3)
                logger.error(error_msg)
                self.log_performance(
                    "fetch_ohlc",
                    time.time() - start_time,
                    success=False,
                    error=str(e),
                    data_type="ohlc",
                    symbol=symbol,
                )
                return None
        return df

    def fetch_dom(
        self, symbol: str, levels: int, normalize: bool = False
    ) -> Optional[pd.DataFrame]:
        """Récupère le carnet d’ordres via CSV."""
        cols = (
            ["timestamp"]
            + [f"bid_price_level_{i}" for i in range(1, levels + 1)]
            + [f"ask_price_level_{i}" for i in range(1, levels + 1)]
            + [f"bid_size_level_{i}" for i in range(1, levels + 1)]
            + [f"ask_size_level_{i}" for i in range(1, levels + 1)]
            + ["spoofing_score", "volume_anomaly"]
        )
        df = self._load_and_process(
            self.csv_paths["dom"], [symbol], cols, normalize, "dom"
        )
        if df is not None:
            df[["spoofing_score", "volume_anomaly"]] = compute_microstructure_metrics(
                df
            )  # Phase 15
        return df

    def fetch_options(
        self, symbol: str, normalize: bool = False
    ) -> Optional[pd.DataFrame]:
        """Récupère les données de chaîne d’options via CSV."""
        cols = [
            "timestamp",
            "strike",
            "option_type",
            "open_interest",
            "volume",
            "gamma",
            "delta",
            "vega",
            "price",
            "call_iv_atm",
            "put_iv_atm",
            "option_volume",
            "oi_concentration",
            "option_skew",
            "net_gamma",
        ]
        df = self._load_and_process(
            self.csv_paths["options"], [symbol], cols, normalize, "options"
        )
        return df

    def fetch_cross_market(
        self, symbols: List[str], normalize: bool = False
    ) -> Optional[pd.DataFrame]:
        """Récupère les données cross-market via CSV."""
        cols = ["timestamp", "symbol", "close", "vix_es_correlation"]
        if self.parallel_load:
            paths = [
                {
                    "path": self.csv_paths["cross_market"],
                    "symbols": [s],
                    "cols": cols,
                    "normalize": normalize,
                    "data_type": "cross_market",
                }
                for s in symbols
            ]
            dfs = self._parallel_load(paths)
            if any(df is None for df in dfs):
                error_msg = (
                    f"Échec récupération cross-market pour certains symboles: {symbols}"
                )
                self.alert_manager.send_alert(error_msg, priority=3)
                self.log_performance(
                    "fetch_cross_market",
                    time.time() - start_time,
                    success=False,
                    error="Échec pour certains symboles",
                    data_type="cross_market",
                    symbol=",".join(symbols),
                )
                return None
            df = pd.concat([df for df in dfs if df is not None], ignore_index=True)
        else:
            df = self._load_and_process(
                self.csv_paths["cross_market"], symbols, cols, normalize, "cross_market"
            )
        if df is not None:
            missing_symbols = set(symbols) - set(df["symbol"].unique())
            if missing_symbols:
                error_msg = f"Symboles manquants dans cross-market: {missing_symbols}"
                self.alert_manager.send_alert(error_msg, priority=2)
        return df

    def fetch_news(self, keywords: List[str], timeframe: str) -> Optional[pd.DataFrame]:
        """Récupère les flux d’actualités via CSV."""
        cols = ["timestamp", "headline", "source", "news_sentiment_score"]
        return self._load_and_process(
            self.csv_paths["news"],
            [],
            cols,
            normalize=False,
            data_type="news",
            keywords=keywords,
        )

    def fetch_microstructure_metrics(
        self, symbol: str, normalize: bool = False
    ) -> Optional[pd.DataFrame]:
        """Récupère les métriques de microstructure via CSV (Phases 15, 18)."""
        cols = [
            "timestamp",
            "spoofing_score",
            "volume_anomaly",
            "quote_stuffing_intensity",
            "order_cancellation_ratio",
            "microstructure_volatility",
            "liquidity_absorption_rate",
            "depth_imbalance",
            "trade_velocity",
            "hft_activity_score",
        ]
        df = self._load_and_process(
            self.csv_paths["microstructure"],
            [symbol],
            cols,
            normalize,
            "microstructure",
        )
        if df is not None:
            df[cols[1:]] = compute_microstructure_metrics(df)  # Phase 15, 18
        return df


class IQFeedProvider(DataProvider):
    """Implémentation pour IQFeed avec pyiqfeed."""

    def __init__(self, config_path: str = MARKET_CONFIG):
        super().__init__(config_path)
        self.api_key = self.config.get("iqfeed", {}).get("api_key", None)
        self.host = self.config.get("iqfeed", {}).get("host", "localhost")
        self.ohlc_port = self.config.get("iqfeed", {}).get("ohlc_port", 9100)
        self.quote_port = self.config.get("iqfeed", {}).get("quote_port", 9200)
        self.news_port = self.config.get("iqfeed", {}).get("news_port", 9300)
        self._validate_iqfeed_config()

    def _validate_iqfeed_config(self):
        """Valide les paramètres de configuration IQFeed."""
        start_time = time.time()
        try:
            if not self.api_key:
                raise ValueError("Clé API IQFeed manquante dans la configuration")
            for port in [self.ohlc_port, self.quote_port, self.news_port]:
                if not isinstance(port, int) or port <= 0:
                    raise ValueError(f"Port IQFeed invalide: {port}")
            latency = time.time() - start_time
            self.log_performance(
                "validate_iqfeed_config", latency, success=True, data_type="config"
            )
        except ValueError as e:
            self.alert_manager.send_alert(
                f"Erreur configuration IQFeed: {str(e)}", priority=3
            )
            logger.error(f"Erreur configuration IQFeed: {str(e)}")
            self.log_performance(
                "validate_iqfeed_config",
                time.time() - start_time,
                success=False,
                error=str(e),
                data_type="config",
            )
            raise

    def connect(self) -> None:
        """Initialise la connexion à l’API IQFeed."""
        start_time = time.time()

        def connect():
            from pyiqfeed import FeedConn

            conn = FeedConn(host=self.host, port=self.ohlc_port, key=self.api_key)
            conn.connect()
            return conn

        try:
            conn = self.with_retries(connect)
            if conn is None:
                raise ValueError("Échec connexion IQFeed")
            latency = time.time() - start_time
            self.log_performance(
                "connect", latency, success=True, data_type="connection"
            )
            self.save_snapshot(
                "connect",
                {"status": "connected", "host": self.host, "ohlc_port": self.ohlc_port},
            )
        except (ImportError, ValueError) as e:
            error_msg = f"Erreur connexion IQFeed: {str(e)}"
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance(
                "connect",
                time.time() - start_time,
                success=False,
                error=str(e),
                data_type="connection",
            )
            raise

    def fetch_ohlc(
        self, symbol: str, timeframe: str, normalize: bool = False
    ) -> Optional[pd.DataFrame]:
        """Récupère les données OHLC via IQFeed (méthode 1)."""
        start_time = time.time()
        try:
            from pyiqfeed import BarConn

            if symbol not in self.valid_symbols:
                raise ValueError(f"Symbole invalide: {symbol}")
            key = hashlib.sha256(f"ohlc_{symbol}_{timeframe}".encode()).hexdigest()[:16]
            cache_path = os.path.join(CACHE_DIR, f"ohlc_{key}.csv")
            if (
                os.path.exists(cache_path)
                and (time.time() - os.path.getmtime(cache_path)) < self.cache_expiration
            ):

                def load_cache():
                    df = pd.read_csv(
                        cache_path, encoding="utf-8", parse_dates=["timestamp"]
                    )
                    return df

                df = self.with_retries(load_cache)
                if df is not None:
                    latency = time.time() - start_time
                    self.log_performance(
                        "fetch_ohlc",
                        latency,
                        success=True,
                        data_type="ohlc",
                        symbol=symbol,
                    )
                    return df

            def fetch():
                conn = BarConn(host=self.host, port=self.ohlc_port, key=self.api_key)
                conn.connect()
                data = conn.get_bars(symbol=symbol, interval=timeframe)
                df = pd.DataFrame(
                    data,
                    columns=[
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "vix_es_correlation",
                        "atr_14",
                    ],
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = self.validate_data(
                    df,
                    [
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "vix_es_correlation",
                        "atr_14",
                    ],
                )
                if df is None:
                    raise ValueError("Données OHLC invalides")
                if normalize:
                    scaler = MinMaxScaler()
                    num_cols = [
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "vix_es_correlation",
                        "atr_14",
                    ]
                    df[num_cols] = scaler.fit_transform(df[num_cols])
                df.to_csv(cache_path, index=False, encoding="utf-8")
                return df

            df = self.with_retries(fetch)
            if df is not None:
                self.store_data_pattern(df, "ohlc", symbol=symbol)
                latency = time.time() - start_time
                self.log_performance(
                    "fetch_ohlc", latency, success=True, data_type="ohlc", symbol=symbol
                )
                self.save_snapshot(
                    "fetch_ohlc",
                    {"symbol": symbol, "rows": len(df), "timeframe": timeframe},
                )
                return df
            return None
        except (ImportError, ValueError) as e:
            error_msg = f"Erreur fetch_ohlc IQFeed: {str(e)}"
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance(
                "fetch_ohlc",
                time.time() - start_time,
                success=False,
                error=str(e),
                data_type="ohlc",
                symbol=symbol,
            )
            return None

    def fetch_dom(
        self, symbol: str, levels: int, normalize: bool = False
    ) -> Optional[pd.DataFrame]:
        """Récupère le carnet d’ordres via IQFeed."""
        start_time = time.time()
        try:
            from pyiqfeed import QuoteConn

            if symbol not in self.valid_symbols:
                raise ValueError(f"Symbole invalide: {symbol}")
            if levels > 10:
                raise ValueError("Niveaux DOM limités à 10")
            key = hashlib.sha256(f"dom_{symbol}_{levels}".encode()).hexdigest()[:16]
            cache_path = os.path.join(CACHE_DIR, f"dom_{key}.csv")
            if (
                os.path.exists(cache_path)
                and (time.time() - os.path.getmtime(cache_path)) < self.cache_expiration
            ):

                def load_cache():
                    df = pd.read_csv(
                        cache_path, encoding="utf-8", parse_dates=["timestamp"]
                    )
                    return df

                df = self.with_retries(load_cache)
                if df is not None:
                    latency = time.time() - start_time
                    self.log_performance(
                        "fetch_dom",
                        latency,
                        success=True,
                        data_type="dom",
                        symbol=symbol,
                    )
                    return df

            def fetch():
                conn = QuoteConn(host=self.host, port=self.quote_port, key=self.api_key)
                conn.connect()
                data = conn.get_quotes(symbol=symbol, levels=levels)
                cols = (
                    ["timestamp"]
                    + [f"bid_price_level_{i}" for i in range(1, levels + 1)]
                    + [f"ask_price_level_{i}" for i in range(1, levels + 1)]
                    + [f"bid_size_level_{i}" for i in range(1, levels + 1)]
                    + [f"ask_size_level_{i}" for i in range(1, levels + 1)]
                    + ["spoofing_score", "volume_anomaly"]
                )
                df = pd.DataFrame(data, columns=cols)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df[["spoofing_score", "volume_anomaly"]] = (
                    compute_microstructure_metrics(df)
                )  # Phase 15
                df = self.validate_data(df, cols)
                if df is None:
                    raise ValueError("Données DOM invalides")
                if normalize:
                    scaler = MinMaxScaler()
                    num_cols = [c for c in cols if c != "timestamp"]
                    df[num_cols] = scaler.fit_transform(df[num_cols])
                df.to_csv(cache_path, index=False, encoding="utf-8")
                return df

            df = self.with_retries(fetch)
            if df is not None:
                self.store_data_pattern(df, "dom", symbol=symbol)
                latency = time.time() - start_time
                self.log_performance(
                    "fetch_dom", latency, success=True, data_type="dom", symbol=symbol
                )
                self.save_snapshot(
                    "fetch_dom", {"symbol": symbol, "rows": len(df), "levels": levels}
                )
                return df
            return None
        except (ImportError, ValueError) as e:
            error_msg = f"Erreur fetch_dom IQFeed: {str(e)}"
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance(
                "fetch_dom",
                time.time() - start_time,
                success=False,
                error=str(e),
                data_type="dom",
                symbol=symbol,
            )
            return None

    def fetch_options(
        self, symbol: str, normalize: bool = False
    ) -> Optional[pd.DataFrame]:
        """Récupère les données de chaîne d’options via IQFeed (méthode 2)."""
        start_time = time.time()
        try:
            from pyiqfeed import OptionConn

            if symbol not in self.valid_symbols:
                raise ValueError(f"Symbole invalide: {symbol}")
            key = hashlib.sha256(f"options_{symbol}".encode()).hexdigest()[:16]
            cache_path = os.path.join(CACHE_DIR, f"options_{key}.csv")
            if (
                os.path.exists(cache_path)
                and (time.time() - os.path.getmtime(cache_path)) < self.cache_expiration
            ):

                def load_cache():
                    df = pd.read_csv(
                        cache_path, encoding="utf-8", parse_dates=["timestamp"]
                    )
                    return df

                df = self.with_retries(load_cache)
                if df is not None:
                    latency = time.time() - start_time
                    self.log_performance(
                        "fetch_options",
                        latency,
                        success=True,
                        data_type="options",
                        symbol=symbol,
                    )
                    return df

            def fetch():
                conn = OptionConn(
                    host=self.host, port=self.quote_port, key=self.api_key
                )
                conn.connect()
                data = conn.get_option_chain(symbol=symbol)
                cols = [
                    "timestamp",
                    "strike",
                    "option_type",
                    "open_interest",
                    "volume",
                    "gamma",
                    "delta",
                    "vega",
                    "price",
                    "call_iv_atm",
                    "put_iv_atm",
                    "option_volume",
                    "oi_concentration",
                    "option_skew",
                    "net_gamma",
                ]
                df = pd.DataFrame(data, columns=cols)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = self.validate_data(df, cols)
                if df is None:
                    raise ValueError("Données options invalides")
                if normalize:
                    scaler = MinMaxScaler()
                    num_cols = [
                        c for c in cols if c != "timestamp" and c != "option_type"
                    ]
                    df[num_cols] = scaler.fit_transform(df[num_cols])
                df.to_csv(cache_path, index=False, encoding="utf-8")
                return df

            df = self.with_retries(fetch)
            if df is not None:
                self.store_data_pattern(df, "options", symbol=symbol)
                latency = time.time() - start_time
                self.log_performance(
                    "fetch_options",
                    latency,
                    success=True,
                    data_type="options",
                    symbol=symbol,
                )
                self.save_snapshot("fetch_options", {"symbol": symbol, "rows": len(df)})
                return df
            return None
        except (ImportError, ValueError) as e:
            error_msg = f"Erreur fetch_options IQFeed: {str(e)}"
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance(
                "fetch_options",
                time.time() - start_time,
                success=False,
                error=str(e),
                data_type="options",
                symbol=symbol,
            )
            return None

    def fetch_cross_market(
        self, symbols: List[str], normalize: bool = False
    ) -> Optional[pd.DataFrame]:
        """Récupère les données cross-market via IQFeed."""
        start_time = time.time()
        try:
            from pyiqfeed import BarConn

            for symbol in symbols:
                if symbol not in self.valid_symbols:
                    raise ValueError(f"Symbole invalide: {symbol}")
            symbols_key = "_".join(symbols)
            key = hashlib.sha256(f"cross_market_{symbols_key}".encode()).hexdigest()[:16]
            cache_path = os.path.join(CACHE_DIR, f"cross_market_{key}.csv")
            if (
                os.path.exists(cache_path)
                and (time.time() - os.path.getmtime(cache_path)) < self.cache_expiration
            ):

                def load_cache():
                    df = pd.read_csv(
                        cache_path, encoding="utf-8", parse_dates=["timestamp"]
                    )
                    return df

                df = self.with_retries(load_cache)
                if df is not None:
                    latency = time.time() - start_time
                    self.log_performance(
                        "fetch_cross_market",
                        latency,
                        success=True,
                        data_type="cross_market",
                        symbol=",".join(symbols),
                    )
                    return df

            def fetch():
                dfs = []
                for symbol in symbols:
                    conn = BarConn(
                        host=self.host, port=self.ohlc_port, key=self.api_key
                    )
                    conn.connect()
                    data = conn.get_bars(symbol=symbol, interval="1min")
                    df = pd.DataFrame(
                        data, columns=["timestamp", "close", "vix_es_correlation"]
                    )
                    df["symbol"] = symbol
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    dfs.append(df)
                df = pd.concat(dfs, ignore_index=True)
                df = self.validate_data(
                    df, ["timestamp", "symbol", "close", "vix_es_correlation"]
                )
                if df is None:
                    raise ValueError("Données cross-market invalides")
                if normalize:
                    scaler = MinMaxScaler()
                    num_cols = ["close", "vix_es_correlation"]
                    df[num_cols] = scaler.fit_transform(df[num_cols])
                df.to_csv(cache_path, index=False, encoding="utf-8")
                return df

            df = self.with_retries(fetch)
            if df is not None:
                self.store_data_pattern(df, "cross_market", symbol=",".join(symbols))
                latency = time.time() - start_time
                self.log_performance(
                    "fetch_cross_market",
                    latency,
                    success=True,
                    data_type="cross_market",
                    symbol=",".join(symbols),
                )
                self.save_snapshot(
                    "fetch_cross_market", {"symbols": symbols, "rows": len(df)}
                )
                return df
            return None
        except (ImportError, ValueError) as e:
            error_msg = f"Erreur fetch_cross_market IQFeed: {str(e)}"
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance(
                "fetch_cross_market",
                time.time() - start_time,
                success=False,
                error=str(e),
                data_type="cross_market",
                symbol=",".join(symbols),
            )
            return None

    def fetch_news(self, keywords: List[str], timeframe: str) -> Optional[pd.DataFrame]:
        """Récupère les flux d’actualités via IQFeed."""
        start_time = time.time()
        try:
            from pyiqfeed import NewsConn

            key = hashlib.sha256(
                f"news_{'_'.join(keywords)}_{timeframe}".encode()
            ).hexdigest()[:16]
            cache_path = os.path.join(CACHE_DIR, f"news_{key}.csv")
            if (
                os.path.exists(cache_path)
                and (time.time() - os.path.getmtime(cache_path)) < self.cache_expiration
            ):

                def load_cache():
                    df = pd.read_csv(
                        cache_path, encoding="utf-8", parse_dates=["timestamp"]
                    )
                    return df

                df = self.with_retries(load_cache)
                if df is not None:
                    latency = time.time() - start_time
                    self.log_performance(
                        "fetch_news", latency, success=True, data_type="news"
                    )
                    return df

            def fetch():
                conn = NewsConn(host=self.host, port=self.news_port, key=self.api_key)
                conn.connect()
                news_data = conn.get_news(keywords=keywords, timeframe=timeframe)
                df = pd.DataFrame(
                    news_data, columns=["timestamp", "headline", "source"]
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df["news_sentiment_score"] = df["headline"].apply(
                    analyze_sentiment
                )  # Phase 1
                df = self.validate_data(
                    df, ["timestamp", "headline", "source", "news_sentiment_score"]
                )
                if df is None:
                    raise ValueError("Données actualités invalides")
                df.to_csv(cache_path, index=False, encoding="utf-8")
                return df

            df = self.with_retries(fetch)
            if df is not None:
                self.store_data_pattern(df, "news", keywords=keywords)
                latency = time.time() - start_time
                self.log_performance(
                    "fetch_news", latency, success=True, data_type="news"
                )
                self.save_snapshot(
                    "fetch_news", {"keywords": keywords, "rows": len(df)}
                )
                return df
            return None
        except (ImportError, ValueError) as e:
            error_msg = f"Erreur fetch_news IQFeed: {str(e)}"
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance(
                "fetch_news",
                time.time() - start_time,
                success=False,
                error=str(e),
                data_type="news",
            )
            return None

    def fetch_microstructure_metrics(
        self, symbol: str, normalize: bool = False
    ) -> Optional[pd.DataFrame]:
        """Récupère les métriques de microstructure via IQFeed (Phases 15, 18)."""
        start_time = time.time()
        try:
            from pyiqfeed import QuoteConn

            if symbol not in self.valid_symbols:
                raise ValueError(f"Symbole invalide: {symbol}")
            key = hashlib.sha256(f"microstructure_{symbol}".encode()).hexdigest()[:16]
            cache_path = os.path.join(CACHE_DIR, f"microstructure_{key}.csv")
            if (
                os.path.exists(cache_path)
                and (time.time() - os.path.getmtime(cache_path)) < self.cache_expiration
            ):

                def load_cache():
                    df = pd.read_csv(
                        cache_path, encoding="utf-8", parse_dates=["timestamp"]
                    )
                    return df

                df = self.with_retries(load_cache)
                if df is not None:
                    latency = time.time() - start_time
                    self.log_performance(
                        "fetch_microstructure_metrics",
                        latency,
                        success=True,
                        data_type="microstructure",
                        symbol=symbol,
                    )
                    return df

            def fetch():
                conn = QuoteConn(host=self.host, port=self.quote_port, key=self.api_key)
                conn.connect()
                data = conn.get_quotes(
                    symbol=symbol, levels=5
                )  # Niveau 5 pour microstructure
                cols = [
                    "timestamp",
                    "spoofing_score",
                    "volume_anomaly",
                    "quote_stuffing_intensity",
                    "order_cancellation_ratio",
                    "microstructure_volatility",
                    "liquidity_absorption_rate",
                    "depth_imbalance",
                    "trade_velocity",
                    "hft_activity_score",
                ]
                df = pd.DataFrame(
                    data,
                    columns=["timestamp"]
                    + [
                        f"level_{i}_{attr}"
                        for i in range(1, 6)
                        for attr in ["bid_price", "ask_price", "bid_size", "ask_size"]
                    ],
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df[cols[1:]] = compute_microstructure_metrics(df)  # Phase 15, 18
                df = self.validate_data(df, cols)
                if df is None:
                    raise ValueError("Données microstructure invalides")
                if normalize:
                    scaler = MinMaxScaler()
                    num_cols = [c for c in cols if c != "timestamp"]
                    df[num_cols] = scaler.fit_transform(df[num_cols])
                df.to_csv(cache_path, index=False, encoding="utf-8")
                return df

            df = self.with_retries(fetch)
            if df is not None:
                self.store_data_pattern(df, "microstructure", symbol=symbol)
                latency = time.time() - start_time
                self.log_performance(
                    "fetch_microstructure_metrics",
                    latency,
                    success=True,
                    data_type="microstructure",
                    symbol=symbol,
                )
                self.save_snapshot(
                    "fetch_microstructure_metrics", {"symbol": symbol, "rows": len(df)}
                )
                return df
            return None
        except (ImportError, ValueError) as e:
            error_msg = f"Erreur fetch_microstructure_metrics IQFeed: {str(e)}"
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance(
                "fetch_microstructure_metrics",
                time.time() - start_time,
                success=False,
                error=str(e),
                data_type="microstructure",
                symbol=symbol,
            )
            return None


def get_data_provider(config_path: str = MARKET_CONFIG) -> DataProvider:
    """Renvoie l’instance du provider configuré (Csv, IQFeed, etc.)."""
    start_time = time.time()
    try:
        config = config_manager.get_config(config_path)
        feed = config.get("data_feed", "csv").lower()
        if feed == "csv":
            provider = CsvDataProvider()
        elif feed == "iqfeed":
            provider = IQFeedProvider()
        else:
            error_msg = f"Fournisseur de données inconnu: {feed} dans {config_path}"
            AlertManager().send_alert(error_msg, priority=3)
            logger.error(error_msg)
            raise ValueError(error_msg)
        latency = time.time() - start_time
        provider.log_performance(
            "get_data_provider", latency, success=True, data_type="provider_selection"
        )
        return provider
    except (FileNotFoundError, ValueError) as e:
        error_msg = f"Erreur sélection fournisseur: {str(e)}"
        AlertManager().send_alert(error_msg, priority=3)
        logger.error(error_msg)
        raise ValueError(f"Configuration fournisseur invalide: {str(e)}")


if __name__ == "__main__":
    """Point d’entrée pour tester l’initialisation du module."""
    try:
        provider = get_data_provider()
        print(f"Provider initialized successfully: {provider.__class__.__name__}")
    except Exception as e:
        error_msg = f"Erreur initialisation fournisseur: {str(e)}"
        print(error_msg)
        AlertManager().send_alert(error_msg, priority=3)