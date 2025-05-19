# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/scripts/run_preprocessing.py
# Script pour exécuter le pipeline de génération des features pour MIA_IA_SYSTEM_v2_2025.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Génère des features techniques et neuronales à partir de données fusionnées,
#        avec journalisation, snapshots compressés, et alertes.
#        Conforme à la Phase 7 (génération des features), Phase 8 (auto-conscience via 
# alertes), et Phase 16 (ensemble learning).
#
# Dépendances : pandas>=2.0.0, numpy>=1.23.0, pyyaml>=6.0.0, logging, json, time, 
# psutil>=5.9.8, matplotlib>=3.7.0, datetime, typing, gzip, traceback, ta, signal,
# src.api.merge_data_sources, src.features.feature_pipeline, src.features.neural_pipeline,
# src.model.utils.config_manager, src.model.utils.alert_manager,
# src.model.utils.miya_console, src.utils.telegram_alert, src.utils.standard
#
# Inputs : config/es_config.yaml, data/ibkr/ibkr_data.csv, data/news/news_data.csv, 
# config/feature_sets.yaml
#
# Outputs : data/logs/run_preprocessing.log, 
# data/logs/run_preprocessing_performance.csv,
# data/features/features_latest_filtered.csv, data/preprocessing_dashboard.json,
# data/preprocessing_snapshots/snapshot_*.json.gz,
# data/figures/preprocessing/preprocessing_status_*.png,
# data/figures/preprocessing/nan_ratios_*.png,
# data/figures/preprocessing/warmup_removed_*.png
#
# Notes :
# - Gère 350 features pour l’entraînement et 150 SHAP features pour l’inférence.
# - Utilise IQFeed exclusivement via les données IBKR et news (indirectement).
# - Implémente retries (max 3, délai 2^attempt), logs psutil, snapshots JSON 
# compressés, alertes centralisées.
# - Tests unitaires dans tests/test_run_preprocessing.py.

import gzip
import json
import logging
import signal
import traceback
from datetime import datetime, timedelta
from typing import Dict, Optional

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import psutil
import ta
import yaml

from src.api.merge_data_sources import merge_data_sources
from src.features.feature_pipeline import calculate_technical_features, feature_pipeline
from src.features.neural_pipeline import generate_neural_features
from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import config_manager
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.utils.standard import with_retries
from src.utils.telegram_alert import send_telegram_alert

matplotlib.use("Agg")  # Backend non interactif pour éviter les erreurs tkinter

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "es_config.yaml")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "features", "features_latest_filtered.csv")
DASHBOARD_PATH = os.path.join(BASE_DIR, "data", "preprocessing_dashboard.json")
NEWS_PATH = os.path.join(BASE_DIR, "data", "news", "news_data.csv")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "preprocessing_snapshots")
CSV_LOG_PATH = os.path.join(BASE_DIR, "data", "logs")
CSV_LOG_PATH = os.path.join(CSV_LOG_PATH, "run_preprocessing_performance.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures", "preprocessing")

# Configuration du logging
os.makedirs(os.path.join(BASE_DIR, "data", "logs"), exist_ok=True)
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "data", "logs", "run_preprocessing.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [Preprocessing] %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Configuration par défaut
DEFAULT_CONFIG = {
    "preprocessing": {
        "input_path": os.path.join(BASE_DIR, "data", "ibkr", "ibkr_data.csv"),
        "output_path": OUTPUT_PATH,
        "news_path": NEWS_PATH,
        "chunk_size": 10000,
        "cache_hours": 24,
        "retry_attempts": 3,
        "retry_delay": 10,
        "timeout_seconds": 1800,
        "max_timestamp": "2025-05-13 11:39:00",
        "buffer_size": 100,
        "max_cache_size": 1000,
        "num_features": 350,
        "shap_features": 150,
    }
}

# Variable pour gérer l'arrêt propre
RUNNING = True


class PreprocessingRunner:
    """
    Classe pour gérer le pipeline de génération des features.
    """

    def __init__(self):
        """
        Initialise le gestionnaire du preprocessing.
        """
        self.log_buffer = []
        self.cache = {}
        try:
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            os.makedirs(FIGURES_DIR, exist_ok=True)
            self.config = self.load_config()
            self.buffer_size = self.config.get("buffer_size", 100)
            self.max_cache_size = self.config.get("max_cache_size", 1000)
            miya_speak(
                "PreprocessingRunner initialisé",
                tag="PREPROCESS",
                voice_profile="calm",
                priority=2,
            )
            AlertManager().send_alert("PreprocessingRunner initialisé", priority=1)
            logger.info("PreprocessingRunner initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config_path": CONFIG_PATH})
        except Exception as e:
            error_msg = (
                f"Erreur initialisation PreprocessingRunner: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="PREPROCESS", voice_profile="urgent", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            self.config = DEFAULT_CONFIG["preprocessing"]
            self.buffer_size = 100
            self.max_cache_size = 1000

    @with_retries(max_attempts=3, delay_base=2.0)
    def load_config(self, config_path: str = CONFIG_PATH) -> Dict:
        """
        Charge les paramètres du pipeline de preprocessing depuis es_config.yaml.

        Args:
            config_path (str): Chemin du fichier de configuration.

        Returns:
            Dict: Configuration du preprocessing.
        """
        start_time = datetime.now()
        try:
            config = config_manager.get_config("es_config.yaml")
            if "preprocessing" not in config:
                raise ValueError("Clé 'preprocessing' manquante dans la configuration")
            required_keys = [
                "input_path",
                "output_path",
                "news_path",
                "chunk_size",
                "num_features",
                "shap_features",
            ]
            missing_keys = [
                key for key in required_keys if key not in config["preprocessing"]
            ]
            if missing_keys:
                error_msg = f"Clés manquantes dans 'preprocessing': {missing_keys}"
                raise ValueError(error_msg)
            if config["preprocessing"]["num_features"] != 350:
                error_msg = (
                    f"Nombre de features incorrect: "
                    f"{config['preprocessing']['num_features']} != 350"
                )
                raise ValueError(error_msg)
            if config["preprocessing"]["shap_features"] != 150:
                error_msg = (
                    f"Nombre de SHAP features incorrect: "
                    f"{config['preprocessing']['shap_features']} != 150"
                )
                raise ValueError(error_msg)
            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                "Configuration preprocessing chargée",
                tag="PREPROCESS",
                level="info",
                priority=2,
            )
            AlertManager().send_alert("Configuration preprocessing chargée", priority=1)
            logger.info("Configuration preprocessing chargée")
            self.log_performance("load_config", latency, success=True)
            self.save_snapshot("load_config", {"config_path": config_path})
            return config["preprocessing"]
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur chargement config: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="PREPROCESS", voice_profile="urgent", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("load_config", latency, success=False, error=str(e))
            self.save_snapshot("load_config", {"error": str(e)})
            raise

    def log_performance(
        self, 
        operation: str, 
        latency: float, 
        success: bool, 
        error: str = None, 
        **kwargs
    ) -> None:
        """
        Journalise les performances des opérations critiques.

        Args:
            operation (str): Nom de l’opération (ex. : preprocess_features).
            latency (float): Latence en secondes.
            success (bool): Succès de l’opération.
            error (str, optional): Message d’erreur si échec.
            **kwargs: Paramètres supplémentaires (ex. : num_features).
        """
        try:
            memory_usage = (
                psutil.Process().memory_info().rss / 1024 / 1024
            )  # Mémoire en Mo
            cpu_percent = psutil.cpu_percent()
            if memory_usage > 1024:
                error_msg = (
                    f"ALERT: Utilisation mémoire élevée ({memory_usage:.2f} MB)"
                )
                miya_alerts(error_msg, tag="PREPROCESS", level="error", priority=5)
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
            miya_alerts(error_msg, tag="PREPROCESS", level="error", priority=3)
            AlertManager().send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def save_snapshot(self, snapshot_type: str, data: Dict) -> None:
        """
        Sauvegarde un instantané des résultats avec compression gzip.

        Args:
            snapshot_type (str): Type de snapshot (ex. : preprocess_features).
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
                tag="PREPROCESS",
                level="info",
                priority=1,
            )
            AlertManager().send_alert(
                f"Snapshot {snapshot_type} sauvegardé: {path}.gz", priority=1
            )
            logger.info(f"Snapshot {snapshot_type} sauvegardé: {path}.gz")
            self.log_performance("save_snapshot", latency, success=True)
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="PREPROCESS", level="error", priority=3)
            AlertManager().send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("save_snapshot", 0, success=False, error=str(e))

    def save_dashboard_status(
        self, status: Dict, status_file: str = DASHBOARD_PATH
    ) -> None:
        """
        Met à jour un fichier JSON pour partager l’état avec mia_dashboard.py.

        Args:
            status (Dict): Statut du pipeline.
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
                tag="PREPROCESS",
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
            miya_alerts(error_msg, tag="PREPROCESS", voice_profile="urgent", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "save_dashboard_status", 0, success=False, error=str(e)
            )

    def plot_preprocessing_results(
        self,
        status: Dict,
        df: Optional[pd.DataFrame] = None,
        output_dir: str = FIGURES_DIR,
    ) -> None:
        """
        Génère des graphiques pour les résultats du preprocessing.

        Args:
            status (Dict): Statut du pipeline.
            df (Optional[pd.DataFrame]): DataFrame des features (pour les NaN).
            output_dir (str): Répertoire pour sauvegarder les graphiques.
        """
        start_time = datetime.now()
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(output_dir, exist_ok=True)

            # Graphique en anneau pour le statut
            labels = ["Succès", "Échec"]
            sizes = [1 if status["success"] else 0, 0 if status["success"] else 1]
            colors = [
                "green" if status["success"] else "red",
                "red" if status["success"] else "green",
            ]
            plt.figure(figsize=(8, 8))
            plt.pie(
                sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140
            )
            plt.title("Statut du Preprocessing")
            centre_circle = plt.Circle((0, 0), 0.70, fc="white")
            fig = plt.gcf()
            fig.gca().add_artist(centre_circle)
            plt.savefig(
                os.path.join(output_dir, f"preprocessing_status_{timestamp}.png")
            )
            plt.close()

            # Graphique des NaN si DataFrame fourni
            if df is not None and not df.empty:
                nan_ratios = df.isna().mean()
                plt.figure(figsize=(15, 6))
                nan_ratios[nan_ratios > 0].plot(kind="bar")
                plt.title("Ratio de NaN par Feature")
                plt.xlabel("Features")
                plt.ylabel("Ratio de NaN")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"nan_ratios_{timestamp}.png"))
                plt.close()

            # Graphique des lignes supprimées
            warmup_removed = status.get("warmup_removed", 0)
            plt.figure(figsize=(8, 6))
            plt.bar(["Lignes Supprimées"], [warmup_removed], color="orange")
            plt.title("Lignes Supprimées par Warmup")
            plt.ylabel("Nombre de Lignes")
            plt.savefig(os.path.join(output_dir, f"warmup_removed_{timestamp}.png"))
            plt.close()

            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                f"Graphiques du preprocessing générés: {output_dir}",
                tag="PREPROCESS",
                level="info",
                priority=2,
            )
            AlertManager().send_alert(
                f"Graphiques du preprocessing générés: {output_dir}", priority=1
            )
            logger.info(f"Graphiques du preprocessing générés: {output_dir}")
            self.log_performance("plot_preprocessing_results", latency, success=True)
            self.save_snapshot(
                "plot_preprocessing_results",
                {"output_dir": output_dir, "warmup_removed": warmup_removed},
            )
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur génération graphiques: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="PREPROCESS", level="error", priority=3)
            AlertManager().send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "plot_preprocessing_results", latency, success=False, error=str(e)
            )
            self.save_snapshot("plot_preprocessing_results", {"error": str(e)})

    @with_retries(max_attempts=3, delay_base=2.0)
    def load_cache(self, output_path: str, cache_hours: int) -> Optional[pd.DataFrame]:
        """
        Charge les features en cache si disponibles et valides.

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
                    if not df.empty and len(df.columns) >= self.config["num_features"]:
                        critical_cols = [
                            "bid_size_level_1",
                            "ask_size_level_1",
                            "trade_frequency_1s",
                        ]
                        for col in critical_cols:
                            if col in df.columns and not pd.api.types.is_numeric_dtype(
                                df[col]
                            ):
                                error_msg = (
                                    f"Colonne {col} non numérique dans cache: "
                                    f"{df[col].dtype}"
                                )
                                miya_alerts(
                                    error_msg,
                                    tag="PREPROCESS",
                                    voice_profile="urgent",
                                    priority=4,
                                )
                                AlertManager().send_alert(error_msg, priority=4)
                                send_telegram_alert(error_msg)
                                logger.error(error_msg)
                                return None
                            if col in df.columns:
                                non_scalar = [
                                    val
                                    for val in df[col]
                                    if isinstance(val, (list, dict, tuple))
                                ]
                                if non_scalar:
                                    error_msg = (
                                        f"Colonne {col} contient des non-scalaires "
                                        f"dans cache: {non_scalar[:5]}"
                                    )
                                    miya_alerts(
                                        error_msg,
                                        tag="PREPROCESS",
                                        voice_profile="urgent",
                                        priority=4,
                                    )
                                    AlertManager().send_alert(error_msg, priority=4)
                                    send_telegram_alert(error_msg)
                                    logger.error(error_msg)
                                    return None
                        latency = (datetime.now() - start_time).total_seconds()
                        miya_speak(
                            f"Cache chargé: {len(df)} lignes, "
                            f"{len(df.columns)} features",
                            tag="PREPROCESS",
                            level="info",
                            priority=2,
                        )
                        AlertManager().send_alert(
                            f"Cache chargé: {len(df)} lignes, "
                            f"{len(df.columns)} features",
                            priority=1,
                        )
                        logger.info(f"Cache chargé: {len(df)} lignes")
                        self.log_performance(
                            "load_cache", latency, success=True, num_rows=len(df)
                        )
                        self.save_snapshot(
                            "load_cache",
                            {"output_path": output_path, "num_rows": len(df)},
                        )
                        return df
            return None
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur chargement cache: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="PREPROCESS", voice_profile="urgent", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("load_cache", latency, success=False, error=str(e))
            self.save_snapshot("load_cache", {"error": str(e)})
            return None

    def apply_warmup_filter(
        self, 
        df: pd.DataFrame, 
        min_valid_ratio: float = 0.95, 
        min_periods: int = 10
    ) -> pd.DataFrame:
        """
        Supprime les premières lignes jusqu'à ce que les colonnes techniques aient 
        suffisamment de données valides.

        Args:
            df (pd.DataFrame): DataFrame contenant les données.
            min_valid_ratio (float): Ratio minimum de valeurs non-NaN requis pour 
            chaque colonne technique.
            min_periods (int): Nombre minimum de lignes restantes après filtrage.

        Returns:
            pd.DataFrame: DataFrame filtré avec les premières lignes supprimées.
        """
        start_time = datetime.now()
        try:
            technical_cols = [
                "rsi_7",
                "rsi_14",
                "rsi_21",
                "bollinger_upper_20",
                "bollinger_lower_20",
                "bollinger_width_20",
                "atr_14",
                "vwap",
                "delta_volume",
                "ofi_score",
                "trend_strength",
                "plus_di_14",
                "minus_di_14",
                "momentum_10",
            ]
            technical_cols = [col for col in technical_cols if col in df.columns]

            if not technical_cols:
                miya_speak(
                    "Aucune colonne technique trouvée pour le warmup",
                    tag="WARMUP",
                    level="warning",
                    priority=3,
                )
                AlertManager().send_alert(
                    "Aucune colonne technique trouvée pour le warmup", priority=2
                )
                logger.warning("Aucune colonne technique trouvée pour le warmup")
                return df

            nan_ratios = df[technical_cols].isna().mean()
            for col, ratio in nan_ratios.items():
                miya_speak(
                    f"Colonne {col} NaN ratio avant warmup: {ratio:.2f}",
                    tag="WARMUP",
                    level="info",
                    priority=2,
                )
                AlertManager().send_alert(
                    f"Colonne {col} NaN ratio avant warmup: {ratio:.2f}", priority=1
                )
                logger.info(f"Colonne {col} NaN ratio avant warmup: {ratio:.2f}")

            # Optimisation : Vérifier les NaN par blocs pour les grands datasets
            block_size = 1000
            for i in range(0, len(df), block_size):
                subset = df.iloc[i:]
                valid_ratios = []
                for col in technical_cols:
                    nan_ratio = subset[col].isna().mean()
                    if nan_ratio < 1.0:
                        valid_ratio = 1 - nan_ratio
                        valid_ratios.append(valid_ratio)
                        if valid_ratio < min_valid_ratio:
                            break
                else:
                    min_valid = min(valid_ratios) if valid_ratios else 0.0
                    if min_valid >= min_valid_ratio and len(subset) >= min_periods:
                        latency = (datetime.now() - start_time).total_seconds()
                        miya_speak(
                            f"Warmup: {i} lignes supprimées, "
                            f"min_valid_ratio={min_valid:.2f}",
                            tag="WARMUP",
                            level="info",
                            priority=2,
                        )
                        AlertManager().send_alert(
                            f"Warmup: {i} lignes supprimées, "
                            f"min_valid_ratio={min_valid:.2f}",
                            priority=1,
                        )
                        logger.info(
                            f"Warmup: {i} lignes supprimées, "
                            f"min_valid_ratio={min_valid:.2f}"
                        )
                        self.log_performance(
                            "apply_warmup_filter", 
                            latency, 
                            success=True, 
                            rows_removed=i
                        )
                        self.save_snapshot(
                            "apply_warmup_filter",
                            {"rows_removed": i, "min_valid_ratio": min_valid},
                        )
                        return subset.reset_index(drop=True)

            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                "Warmup: Aucune ligne supprimée, pas assez de données valides",
                tag="WARMUP",
                level="warning",
                priority=2,
            )
            AlertManager().send_alert(
                "Warmup: Aucune ligne supprimée, pas assez de données valides",
                priority=2,
            )
            logger.warning("Warmup: Aucune ligne supprimée")
            self.log_performance(
                "apply_warmup_filter", latency, success=True, rows_removed=0
            )
            self.save_snapshot("apply_warmup_filter", {"rows_removed": 0})
            return df
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur warmup: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="WARMUP", voice_profile="urgent", priority=5)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "apply_warmup_filter", latency, success=False, error=str(e)
            )
            self.save_snapshot("apply_warmup_filter", {"error": str(e)})
            raise

    def validate_inputs(
        self, 
        input_path: str, 
        output_path: str, 
        news_path: str, 
        chunk_size: int
    ) -> None:
        """
        Valide les paramètres et données d’entrée du preprocessing.

        Args:
            input_path (str): Chemin des données IBKR.
            output_path (str): Chemin de sortie des features.
            news_path (str): Chemin des données de news.
            chunk_size (int): Taille des morceaux.

        Raises:
            ValueError: Si les paramètres ou données sont invalides.
        """
        start_time = datetime.now()
        try:
            if not os.path.exists(input_path):
                error_msg = f"Fichier IBKR introuvable: {input_path}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            if not os.path.exists(news_path):
                error_msg = f"Fichier de news introuvable: {news_path}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            if chunk_size <= 0:
                error_msg = f"Taille de chunk invalide: {chunk_size}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Charger un échantillon des données pour validation
            df_sample = pd.read_csv(input_path, nrows=10, encoding="utf-8")
            required_cols = ["timestamp", "close"]
            missing_cols = [
                col for col in required_cols if col not in df_sample.columns
            ]
            if missing_cols:
                error_msg = f"Colonnes manquantes dans {input_path}: {missing_cols}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                "Paramètres et données d’entrée validés",
                tag="PREPROCESS",
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
                    "input_path": input_path,
                    "news_path": news_path,
                    "chunk_size": chunk_size,
                },
            )
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur validation entrées: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="PREPROCESS", voice_profile="urgent", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "validate_inputs", latency, success=False, error=str(e)
            )
            self.save_snapshot("validate_inputs", {"error": str(e)})
            raise

    @with_retries(max_attempts=3, delay_base=2.0)
    def preprocess_features(
        self, 
        input_path: str, 
        output_path: str, 
        chunk_size: int, 
        config: Dict
    ) -> pd.DataFrame:
        """
        Génère les features à partir des données fusionnées.

        Args:
            input_path (str): Chemin des données d’entrée.
            output_path (str): Chemin pour sauvegarder les features.
            chunk_size (int): Taille des morceaux pour le traitement.
            config (Dict): Configuration du preprocessing.

        Returns:
            pd.DataFrame: Features générées.
        """
        start_time = datetime.now()
        try:
            self.validate_inputs(
                input_path, output_path, config.get("news_path", NEWS_PATH), chunk_size
            )
            config["num_features"]
            miya_speak(
                "Démarrage preprocessing features",
                tag="PREPROCESS",
                level="info",
                priority=3,
            )
            AlertManager().send_alert("Démarrage preprocessing features", priority=2)
            logger.info("Démarrage preprocessing")

            # Charger la liste des features depuis feature_sets.yaml
            feature_sets_path = os.path.join(BASE_DIR, "config", "feature_sets.yaml")
            with open(feature_sets_path, "r", encoding="utf-8") as f:
                feature_sets = yaml.safe_load(f)
            training_features = feature_sets.get("training_features", [])[
                :350
            ]  # Limiter à 350 features
            if len(training_features) != 350:
                error_msg = (
                    f"Nombre de features incorrect dans feature_sets.yaml: "
                    f"{len(training_features)} != 350"
                )
                miya_alerts(
                    error_msg, tag="PREPROCESS", voice_profile="urgent", priority=5
                )
                AlertManager().send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                raise ValueError(error_msg)

            news_path = config.get("news_path", NEWS_PATH)
            merged_data = merge_data_sources(
                ibkr_path=input_path,
                news_path=news_path,
                output_path=output_path,
                config=config,
            )
            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                f"Données fusionnées: {len(merged_data)} lignes",
                tag="PREPROCESS",
                level="info",
                priority=2,
            )
            AlertManager().send_alert(
                f"Données fusionnées: {len(merged_data)} lignes", priority=1
            )
            logger.info(f"Données fusionnées: {len(merged_data)} lignes")
            self.log_performance(
                "merge_data_sources", latency, success=True, num_rows=len(merged_data)
            )
            self.save_snapshot("merge_data_sources", {"num_rows": len(merged_data)})

            critical_cols = [
                "bid_size_level_1",
                "ask_size_level_1",
                "trade_frequency_1s",
            ]
            for col in critical_cols:
                if col in merged_data.columns and not pd.api.types.is_numeric_dtype(
                    merged_data[col]
                ):
                    error_msg = (
                        f"Colonne {col} non numérique après fusion: "
                        f"{merged_data[col].dtype}"
                    )
                    miya_alerts(
                        error_msg, tag="PREPROCESS", voice_profile="urgent", priority=4
                    )
                    AlertManager().send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                if col in merged_data.columns:
                    non_scalar = [
                        val
                        for val in merged_data[col]
                        if isinstance(val, (list, dict, tuple))
                    ]
                    if non_scalar:
                        error_msg = (
                            f"Colonne {col} contient des non-scalaires: "
                            f"{non_scalar[:5]}"
                        )
                        miya_alerts(
                            error_msg,
                            tag="PREPROCESS",
                            voice_profile="urgent",
                            priority=4,
                        )
                        AlertManager().send_alert(error_msg, priority=4)
                        send_telegram_alert(error_msg)
                        logger.error(error_msg)
                        raise ValueError(error_msg)

            max_timestamp = config.get("max_timestamp", "2025-05-13 11:39:00")
            merged_data = merged_data[merged_data["timestamp"] <= max_timestamp]
            miya_speak(
                f"Données filtrées: {len(merged_data)} lignes",
                tag="PREPROCESS",
                level="info",
                priority=2,
            )
            AlertManager().send_alert(
                f"Données filtrées: {len(merged_data)} lignes", priority=1
            )
            logger.info(f"Données filtrées: {len(merged_data)} lignes")

            rsi_original = (
                merged_data["rsi"].copy() if "rsi" in merged_data.columns else None
            )
            miya_speak(
                f"Colonne rsi présente après fusion: {rsi_original is not None}",
                tag="PREPROCESS",
                level="info",
                priority=2,
            )
            AlertManager().send_alert(
                f"Colonne rsi présente après fusion: {rsi_original is not None}",
                priority=1,
            )

            missing_cols = [
                col
                for col in training_features
                if col not in merged_data.columns and col != "timestamp"
            ]
            if missing_cols:
                miya_speak(
                    f"Initialisation des colonnes manquantes: {missing_cols}",
                    tag="PREPROCESS",
                    level="warning",
                    priority=3,
                )
                AlertManager().send_alert(
                    f"Initialisation des colonnes manquantes: {missing_cols}",
                    priority=2,
                )
                logger.warning(
                    f"Initialisation des colonnes manquantes: {missing_cols}"
                )
                missing_data = pd.DataFrame(
                    0.0, index=merged_data.index, columns=missing_cols
                )
                merged_data = pd.concat([merged_data, missing_data], axis=1)

            merged_data = feature_pipeline(
                config_path=config.get(
                    "config_path", os.path.join(BASE_DIR, "config", "es_config.yaml")
                ),
                csv_file_path=input_path,
            )
            miya_speak(
                f"Features techniques calculées par feature_pipeline: "
                f"{len(merged_data.columns)} colonnes",
                tag="PREPROCESS",
                level="info",
                priority=2,
            )
            AlertManager().send_alert(
                f"Features techniques calculées par feature_pipeline: "
                f"{len(merged_data.columns)} colonnes",
                priority=1,
            )
            logger.info(
                f"Features techniques calculées par feature_pipeline: "
                f"{len(merged_data.columns)} colonnes"
            )

            if "rsi" not in merged_data.columns:
                if rsi_original is not None:
                    merged_data["rsi"] = rsi_original
                    miya_speak(
                        "Colonne rsi restaurée depuis les données fusionnées",
                        tag="PREPROCESS",
                        level="info",
                        priority=2,
                    )
                    AlertManager().send_alert(
                        "Colonne rsi restaurée depuis les données fusionnées",
                        priority=1,
                    )
                else:
                    merged_data["rsi"] = ta.momentum.RSIIndicator(
                        merged_data["close"], window=14, fillna=True
                    ).rsi()
                    miya_speak(
                        "Colonne rsi recalculée (RSI 14)",
                        tag="PREPROCESS",
                        level="info",
                        priority=2,
                    )
                    AlertManager().send_alert(
                        "Colonne rsi recalculée (RSI 14)", priority=1
                    )
                logger.info("Colonne rsi ajoutée ou restaurée")

            original_length = len(merged_data)
            merged_data = self.apply_warmup_filter(
                merged_data, min_valid_ratio=0.95, min_periods=10
            )

            merged_data = calculate_technical_features(merged_data)
            miya_speak(
                "Features techniques calculées",
                tag="PREPROCESS",
                level="info",
                priority=2,
            )
            AlertManager().send_alert("Features techniques calculées", priority=1)
            logger.info("Features techniques calculées")

            CRITICAL_TECHNICAL_COLUMNS = [
                "rsi_7",
                "rsi_14",
                "rsi_21",
                "bollinger_upper_20",
                "bollinger_lower_20",
                "bollinger_width_20",
                "atr_14",
                "vwap",
                "delta_volume",
                "ofi_score",
                "trend_strength",
                "plus_di_14",
                "minus_di_14",
                "momentum_10",
            ]
            numeric_cols = [
                col
                for col in merged_data.columns
                if col in training_features
                and col != "timestamp"
                and col != "sentiment_label"
            ]
            cols_to_impute = [
                col for col in numeric_cols if col not in CRITICAL_TECHNICAL_COLUMNS
            ]
            merged_data[cols_to_impute] = merged_data[cols_to_impute].replace(
                [float("inf"), -float("inf")], float("nan")
            )
            merged_data[cols_to_impute] = merged_data[cols_to_impute].ffill().fillna(0)
            if "sentiment_label" in merged_data.columns:
                merged_data["sentiment_label"] = merged_data["sentiment_label"].fillna(
                    "neutral"
                )

            key_columns = ["rsi", "rsi_14", "sentiment_label"]
            available_columns = [
                col for col in key_columns if col in merged_data.columns
            ]
            missing_columns = [
                col for col in key_columns if col not in merged_data.columns
            ]
            if missing_columns:
                logger.warning(
                    f"Colonnes clés manquantes avant generate_neural_features: "
                    f"{missing_columns}"
                )
                miya_speak(
                    f"Colonnes clés manquantes avant generate_neural_features: "
                    f"{missing_columns}",
                    tag="PREPROCESS",
                    level="warning",
                    priority=2,
                )
                AlertManager().send_alert(
                    f"Colonnes clés manquantes avant generate_neural_features: "
                    f"{missing_columns}",
                    priority=2,
                )
            logger.info(f"Colonnes clés présentes: {available_columns}")

            merged_data = generate_neural_features(merged_data)
            miya_speak(
                f"Features neuronales ajoutées: {len(merged_data.columns)} colonnes",
                tag="PREPROCESS",
                level="info",
                priority=2,
            )
            AlertManager().send_alert(
                f"Features neuronales ajoutées: {len(merged_data.columns)} colonnes",
                priority=1,
            )
            logger.info(
                f"Features neuronales ajoutées: {len(merged_data.columns)} colonnes"
            )

            neural_cols = [f"neural_feature_{i}" for i in range(20)] + [
                "cnn_pressure",
                "predicted_volatility",
                "neural_regime",
            ]
            for col in neural_cols:
                if col in merged_data.columns and merged_data[col].isna().any():
                    merged_data[col] = merged_data[col].interpolate(
                        method="linear", limit_direction="both"
                    )
                    if merged_data[col].isna().any():
                        mean_value = merged_data[col].mean()
                        merged_data[col] = merged_data[col].fillna(
                            mean_value if not pd.isna(mean_value) else 0
                        )
                        miya_speak(
                            f"NaN imputés à "
                            f"{mean_value if not pd.isna(mean_value) else 0} pour {col}",
                            tag="PREPROCESS",
                            level="info",
                            priority=2,
                        )
                        AlertManager().send_alert(
                            f"NaN imputés à "
                            f"{mean_value if not pd.isna(mean_value) else 0} pour {col}",
                            priority=1,
                        )

            missing_cols = [
                col
                for col in training_features
                if col not in merged_data.columns and col != "timestamp"
            ]
            if missing_cols:
                miya_speak(
                    f"Colonnes manquantes après preprocessing: {missing_cols}",
                    tag="PREPROCESS",
                    level="warning",
                    priority=3,
                )
                AlertManager().send_alert(
                    f"Colonnes manquantes après preprocessing: {missing_cols}",
                    priority=2,
                )
                logger.warning(
                    f"Colonnes manquantes après preprocessing: {missing_cols}"
                )
                missing_data = pd.DataFrame(
                    0.0, index=merged_data.index, columns=missing_cols
                )
                merged_data = pd.concat([merged_data, missing_data], axis=1)
                if len(missing_cols) > len(training_features) * 0.75:
                    error_msg = (
                        f"Trop de colonnes manquantes: "
                        f"{len(missing_cols)}/{len(training_features)}"
                    )
                    miya_alerts(
                        error_msg, tag="PREPROCESS", voice_profile="urgent", priority=5
                    )
                    AlertManager().send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    logger.error(error_msg)
                    raise ValueError(error_msg)

            duplicates = merged_data[
                merged_data.duplicated(subset=["timestamp"], keep="last")
            ]
            if not duplicates.empty:
                miya_speak(
                    f"Doublons supprimés: {len(duplicates)} lignes",
                    tag="PREPROCESS",
                    level="info",
                    priority=2,
                )
                AlertManager().send_alert(
                    f"Doublons supprimés: {len(duplicates)} lignes", priority=1
                )
                logger.info(f"Doublons supprimés: {len(duplicates)} lignes")
            merged_data = merged_data.drop_duplicates(subset=["timestamp"], keep="last")

            neural_cols = [f"neural_feature_{i}" for i in range(20)] + [
                "cnn_pressure",
                "predicted_volatility",
                "neural_regime",
            ]
            missing_neural_cols = [
                col for col in neural_cols if col not in merged_data.columns
            ]
            if missing_neural_cols:
                error_msg = f"Features neuronales manquantes: {missing_neural_cols}"
                miya_alerts(
                    error_msg, tag="PREPROCESS", voice_profile="urgent", priority=4
                )
                AlertManager().send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
            else:
                miya_speak(
                    f"Toutes les features neuronales ({len(neural_cols)}) présentes",
                    tag="PREPROCESS",
                    level="info",
                    priority=2,
                )
                AlertManager().send_alert(
                    f"Toutes les features neuronales ({len(neural_cols)}) présentes",
                    priority=1,
                )

            final_cols = list(training_features) + ["timestamp", "close"]
            dropped_cols = [col for col in merged_data.columns if col not in final_cols]
            if dropped_cols:
                miya_speak(
                    f"Colonnes supprimées: {dropped_cols}",
                    tag="PREPROCESS",
                    level="info",
                    priority=2,
                )
                AlertManager().send_alert(
                    f"Colonnes supprimées: {dropped_cols}", priority=1
                )
                logger.info(f"Colonnes supprimées: {dropped_cols}")
            merged_data = merged_data[final_cols]

            if (
                len(merged_data.columns) != len(training_features) + 2
            ):  # +2 pour timestamp et close
                error_msg = (
                    f"Nombre de colonnes incorrect: {len(merged_data.columns)} != "
                    f"{len(training_features) + 2}"
                )
                miya_alerts(
                    error_msg, tag="PREPROCESS", voice_profile="urgent", priority=5
                )
                AlertManager().send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                raise ValueError(error_msg)

            dups = merged_data.columns[merged_data.columns.duplicated()].tolist()
            if dups:
                error_msg = f"Colonnes en double détectées: {dups}"
                miya_alerts(
                    error_msg, tag="PREPROCESS", voice_profile="urgent", priority=3
                )
                AlertManager().send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                logger.warning(error_msg)
                merged_data = merged_data.loc[
                    :, ~merged_data.columns.duplicated(keep="first")
                ]

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            merged_data.to_csv(output_path, index=False, encoding="utf-8")
            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                f"Features générées: {len(merged_data)} lignes, "
                f"{len(merged_data.columns)} colonnes",
                tag="PREPROCESS",
                level="info",
                priority=3,
            )
            AlertManager().send_alert(
                f"Features générées: {len(merged_data)} lignes, "
                f"{len(merged_data.columns)} colonnes",
                priority=2,
            )
            logger.info(f"Features générées: {len(merged_data)} lignes")
            self.log_performance(
                "preprocess_features",
                latency,
                success=True,
                num_rows=len(merged_data),
                num_cols=len(merged_data.columns),
            )
            self.save_snapshot(
                "preprocess_features",
                {
                    "num_rows": len(merged_data),
                    "num_cols": len(merged_data.columns),
                    "warmup_removed": original_length - len(merged_data),
                },
            )
            self.plot_preprocessing_results(
                {
                    "success": True,
                    "feature_count": len(merged_data.columns),
                    "row_count": len(merged_data),
                    "warmup_removed": original_length - len(merged_data),
                    "errors": [],
                },
                merged_data,
            )
            config["warmup_removed"] = original_length - len(merged_data)
            return merged_data
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur preprocessing: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="PREPROCESS", voice_profile="urgent", priority=5)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "preprocess_features", latency, success=False, error=str(e)
            )
            self.save_snapshot("preprocess_features", {"error": str(e)})
            raise

    @with_retries(max_attempts=3, delay_base=2.0)
    def run_preprocessing(self) -> Dict:
        """
        Exécute le pipeline de preprocessing avec retries.

        Returns:
            Dict: Statut du pipeline.
        """
        start_time = datetime.now()
        try:
            status = {
                "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "success": False,
                "feature_count": 0,
                "row_count": 0,
                "warmup_removed": 0,
                "errors": [],
            }
            input_path = self.config.get("input_path")
            output_path = self.config.get("output_path")
            chunk_size = self.config.get("chunk_size", 10000)

            cached_df = self.load_cache(output_path, self.config.get("cache_hours", 24))
            if cached_df is not None and not cached_df.empty:
                status["success"] = True
                status["feature_count"] = len(cached_df.columns)
                status["row_count"] = len(cached_df)
                self.save_dashboard_status(status)
                self.plot_preprocessing_results(status, cached_df)
                miya_speak(
                    f"Preprocessing ignoré, features chargées depuis cache: "
                    f"{len(cached_df)} lignes",
                    tag="PREPROCESS",
                    level="info",
                    priority=3,
                )
                AlertManager().send_alert(
                    f"Preprocessing ignoré, features chargées depuis cache: "
                    f"{len(cached_df)} lignes",
                    priority=1,
                )
                logger.info(
                    f"Preprocessing ignoré, features chargées depuis cache: "
                    f"{len(cached_df)} lignes"
                )
                return status

            df = self.preprocess_features(
                input_path, output_path, chunk_size, self.config
            )
            status["success"] = True
            status["feature_count"] = len(df.columns)
            status["row_count"] = len(df)
            status["warmup_removed"] = self.config.get("warmup_removed", 0)
            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                "Pipeline preprocessing terminé avec succès",
                tag="PREPROCESS",
                level="info",
                priority=3,
            )
            AlertManager().send_alert(
                "Pipeline preprocessing terminé avec succès", priority=1
            )
            logger.info("Pipeline preprocessing terminé avec succès")
            self.log_performance("run_preprocessing", latency, success=True)
            self.save_snapshot("run_preprocessing", status)
            self.save_dashboard_status(status)
            self.plot_preprocessing_results(status, df)
            return status
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Échec preprocessing: {str(e)}\n{traceback.format_exc()}"
            )
            status["errors"].append(error_msg)
            miya_alerts(error_msg, tag="PREPROCESS", voice_profile="urgent", priority=5)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "run_preprocessing", latency, success=False, error=str(e)
            )
            self.save_snapshot("run_preprocessing", {"error": str(e)})
            self.save_dashboard_status(status)
            self.plot_preprocessing_results(status)
            raise

    def signal_handler(self, sig, frame) -> None:  # noqa: F841
        """
        Gère l'arrêt propre du pipeline (Ctrl+C).

        Args:
            sig: Signal reçu.
            frame: Frame actuel.
        """
        global RUNNING
        start_time = datetime.now()
        try:
            RUNNING = False
            miya_speak(
                "Arrêt du pipeline de preprocessing en cours...",
                tag="PREPROCESS",
                voice_profile="calm",
                priority=2,
            )
            AlertManager().send_alert(
                "Arrêt du pipeline de preprocessing en cours", priority=2
            )
            logger.info("Arrêt du pipeline initié")

            status = {
                "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "success": False,
                "feature_count": 0,
                "row_count": 0,
                "warmup_removed": 0,
                "errors": ["Pipeline arrêté manuellement"],
            }
            self.save_dashboard_status(status)
            self.save_snapshot("shutdown", {"status": status})

            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                "Pipeline arrêté proprement",
                tag="PREPROCESS",
                voice_profile="calm",
                priority=2,
            )
            AlertManager().send_alert("Pipeline arrêté proprement", priority=1)
            logger.info("Pipeline arrêté")
            self.log_performance("signal_handler", latency, success=True)
            exit(0)
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur arrêt pipeline: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="PREPROCESS", voice_profile="urgent", priority=5)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("signal_handler", latency, success=False, error=str(e))
            exit(1)

    def run(self) -> None:
        """
        Exécute le gestionnaire du pipeline de preprocessing.
        """
        try:
            signal.signal(signal.SIGINT, self.signal_handler)
            status = self.run_preprocessing()
            if status["success"]:
                miya_speak(
                    f"Preprocessing terminé: {status['feature_count']} features "
                    f"générés, {status['warmup_removed']} lignes supprimées",
                    tag="PREPROCESS",
                    level="info",
                    priority=3,
                )
                AlertManager().send_alert(
                    f"Preprocessing terminé: {status['feature_count']} features "
                    f"générés, {status['warmup_removed']} lignes supprimées",
                    priority=1,
                )
                logger.info(
                    f"Preprocessing terminé: {status['feature_count']} features "
                    f"générés"
                )
            else:
                error_msg = "Échec preprocessing après retries"
                miya_alerts(
                    error_msg, tag="PREPROCESS", voice_profile="urgent", priority=5
                )
                AlertManager().send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
        except Exception as e:
            error_msg = (
                f"Erreur programme: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="PREPROCESS", voice_profile="urgent", priority=5)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            exit(1)


if __name__ == "__main__":
    runner = PreprocessingRunner()
    runner.run()