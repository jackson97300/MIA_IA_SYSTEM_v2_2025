# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/monitoring/correlation_heatmap.py
# Rôle : Génère des heatmaps de corrélation des features pour MIA_IA_SYSTEM_v2_2025, avec focus sur les top 50 SHAP features.
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, seaborn>=0.13.0,<1.0.0, matplotlib>=3.8.0,<4.0.0, psutil>=5.9.8,<6.0.0,
#   loguru>=0.7.0,<1.0.0, numpy>=1.26.4,<2.0.0, pyyaml>=6.0.0,<7.0.0, boto3>=1.26.0,<2.0.0,
#   argparse, os, hashlib, signal, gzip
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/features/feature_pipeline.py
# - src/features/shap_weighting.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - Données de features (pd.DataFrame avec 350 features pour entraînement ou 150 SHAP features pour inférence)
# - data/features/feature_importance.csv (top 50 SHAP features)
# - config/es_config.yaml
# - config/feature_sets.yaml
#
# Outputs :
# - Heatmaps dans data/figures/heatmap/
# - Matrice de corrélations dans data/correlation_matrix.csv
# - Snapshots dans data/cache/heatmap/*.json.gz
# - Sauvegardes incrémentielles dans data/checkpoints/heatmap_*.json.gz
# - Logs dans data/logs/correlation_heatmap.log
# - Logs de performance dans data/logs/heatmap_performance.csv
#
# Notes :
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence) définies dans config/feature_sets.yaml.
# - Utilise IQFeed exclusivement via feature_pipeline.py pour les données d’entrée.
# - Supprime toute référence à 320 features, 81 features, et obs_t.
# - Implémente retries (max 3, délai 2^attempt secondes) pour les opérations critiques.
# - Intègre SHAP (méthode 17) pour les top 50 features et volatilité (méthode 1) avec vix_es_correlation et atr_14.
# - Utilise un cache local dans data/cache/heatmap/ pour réduire la charge.
# - Sauvegardes incrémentielles toutes les 5 min dans data/checkpoints/ (5 versions).
# - Sauvegardes distribuées toutes les 15 min vers S3 (configurable).
# - Tests unitaires disponibles dans tests/test_correlation_heatmap.py.
# - Phases intégrées : Phase 8 (auto-conscience via confidence_drop_rate), Phase 17 (SHAP).
# - Validation complète prévue pour juin 2025.

import argparse
import gzip
import hashlib
import json
import os
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
from loguru import logger

from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import get_config
from src.utils.telegram_alert import send_telegram_alert

# Configuration logging
logger.remove()
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOG_DIR = BASE_DIR / "data" / "logs"
FIGURES_DIR = BASE_DIR / "data" / "figures" / "heatmap"
CACHE_DIR = BASE_DIR / "data" / "cache" / "heatmap"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints"
PERF_LOG_PATH = LOG_DIR / "heatmap_performance.csv"
CORR_MATRIX_PATH = BASE_DIR / "data" / "correlation_matrix.csv"
LOG_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "correlation_heatmap.log",
    rotation="10 MB",
    level="INFO",
    encoding="utf-8",
)
logger.add(lambda msg: print(msg, end=""), level="INFO")

# Chemin par défaut
DEFAULT_DATA_PATH = BASE_DIR / "data" / "features" / "features_latest.csv"

# Seuils de performance globaux
PERFORMANCE_THRESHOLDS = {
    "min_features": 50,  # Minimum pour top 50 SHAP features
    "min_rows": 50,  # Nombre minimum de lignes pour corrélation fiable
}

# Constantes
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Secondes, exponentiel

# Variable pour gérer l'arrêt propre
RUNNING = True


class CorrelationHeatmap:
    """
    Classe pour générer des heatmaps de corrélation des features, avec focus sur les top 50 SHAP features.
    """

    def __init__(self, config_path: str = "config/es_config.yaml"):
        """
        Initialise le générateur de heatmaps.

        Args:
            config_path (str): Chemin vers le fichier de configuration.
        """
        self.alert_manager = AlertManager()
        self.checkpoint_versions = []
        signal.signal(signal.SIGINT, self.handle_sigint)

        start_time = datetime.now()
        try:
            self.config = self.load_config(config_path)
            self.log_buffer = []
            success_msg = "CorrelationHeatmap initialisé"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("init", latency, success=True)
            self.save_snapshot(
                "init",
                {"config_path": config_path, "timestamp": datetime.now().isoformat()},
            )
        except Exception as e:
            error_msg = f"Erreur initialisation CorrelationHeatmap: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            raise

    def handle_sigint(self, signum, frame):
        """Gère l'arrêt propre sur SIGINT."""
        global RUNNING
        datetime.now()
        snapshot = {"timestamp": datetime.now().isoformat(), "status": "SIGINT"}
        CACHE_DIR / f'sigint_{snapshot["timestamp"]}.json.gz'
        try:
            RUNNING = False
            self.save_snapshot("sigint", snapshot)
            success_msg = "Arrêt propre sur SIGINT, snapshot sauvegardé"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            self.log_performance("handle_sigint", 0, success=True)
            exit(0)
        except Exception as e:
            error_msg = f"Erreur sauvegarde SIGINT: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance("handle_sigint", 0, success=False, error=str(e))
            exit(1)

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
            snapshot_path = CACHE_DIR / f"snapshot_{snapshot_type}_{timestamp}.json"
            CACHE_DIR.mkdir(exist_ok=True)

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
                send_telegram_alert(alert_msg)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Snapshot {snapshot_type} sauvegardé: {save_path}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "save_snapshot", latency, success=True, snapshot_size_mb=file_size
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance("save_snapshot", 0, success=False, error=str(e))

    def checkpoint(self, data: pd.DataFrame) -> None:
        """
        Sauvegarde incrémentielle des matrices de corrélations toutes les 5 minutes avec versionnage (5 versions).

        Args:
            data (pd.DataFrame): Données de la matrice de corrélations à sauvegarder.
        """
        try:
            start_time = datetime.now()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_data = {
                "timestamp": timestamp,
                "num_rows": len(data),
                "columns": list(data.columns),
            }
            checkpoint_path = CHECKPOINT_DIR / f"heatmap_{timestamp}.json.gz"
            CHECKPOINT_DIR.mkdir(exist_ok=True)

            def write_checkpoint():
                with gzip.open(checkpoint_path, "wt", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, indent=4)
                data.to_csv(
                    checkpoint_path.replace(".json.gz", ".csv"),
                    index=True,
                    encoding="utf-8",
                )

            self.with_retries(write_checkpoint)
            self.checkpoint_versions.append(checkpoint_path)
            if len(self.checkpoint_versions) > 5:
                oldest = self.checkpoint_versions.pop(0)
                if os.path.exists(oldest):
                    os.remove(oldest)
                csv_oldest = oldest.replace(".json.gz", ".csv")
                if os.path.exists(csv_oldest):
                    os.remove(csv_oldest)
            file_size = os.path.getsize(checkpoint_path) / 1024 / 1024
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Checkpoint sauvegardé: {checkpoint_path}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "checkpoint",
                latency,
                success=True,
                file_size_mb=file_size,
                num_rows=len(data),
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde checkpoint: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance("checkpoint", 0, success=False, error=str(e))

    def cloud_backup(self, data: pd.DataFrame) -> None:
        """
        Sauvegarde distribuée des matrices de corrélations vers AWS S3 toutes les 15 minutes (configurable).

        Args:
            data (pd.DataFrame): Données de la matrice de corrélations à sauvegarder.
        """
        try:
            start_time = datetime.now()
            if not self.config.get("s3_bucket"):
                warning_msg = "S3 bucket non configuré, sauvegarde cloud ignorée"
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.config['s3_prefix']}heatmap_{timestamp}.csv.gz"
            temp_path = CHECKPOINT_DIR / f"temp_s3_{timestamp}.csv.gz"

            def write_temp():
                data.to_csv(temp_path, compression="gzip", index=True, encoding="utf-8")

            self.with_retries(write_temp)
            s3_client = boto3.client("s3")

            def upload_s3():
                s3_client.upload_file(temp_path, self.config["s3_bucket"], backup_path)

            self.with_retries(upload_s3)
            os.remove(temp_path)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Sauvegarde cloud S3 effectuée: {backup_path}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "cloud_backup", latency, success=True, num_rows=len(data)
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde cloud S3: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance("cloud_backup", 0, success=False, error=str(e))

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
                    logger.error(error_msg)
                    self.alert_manager.send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    latency = (datetime.now() - start_time).total_seconds()
                    self.log_performance(
                        f"retry_attempt_{attempt+1}",
                        latency,
                        success=False,
                        error=str(e),
                        attempt_number=attempt + 1,
                    )
                    return None
                delay = delay_base**attempt
                warning_msg = f"Tentative {attempt+1} échouée, retry après {delay}s"
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                time.sleep(delay)

    def load_config(self, config_path: str) -> Dict:
        """
        Charge la configuration avec repli sur les valeurs par défaut.

        Args:
            config_path (str): Chemin vers le fichier de configuration.

        Returns:
            Dict: Configuration chargée.
        """
        start_time = datetime.now()
        try:
            config = get_config(BASE_DIR / config_path)
            if not config:
                raise ValueError("Configuration vide ou non trouvée")
            params = config.get(
                "correlation_params",
                {
                    "significant_threshold": 0.8,
                    "s3_bucket": None,
                    "s3_prefix": "heatmap/",
                },
            )
            required_keys = ["significant_threshold", "s3_bucket", "s3_prefix"]
            missing_keys = [key for key in required_keys if key not in params]
            if missing_keys:
                raise ValueError(f"Clés de configuration manquantes: {missing_keys}")
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Configuration {config_path} chargée"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance("load_config", latency, success=True)
            return params
        except Exception as e:
            error_msg = (
                f"Erreur chargement configuration: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance("load_config", 0, success=False, error=str(e))
            return {
                "significant_threshold": 0.8,
                "s3_bucket": None,
                "s3_prefix": "heatmap/",
            }

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Charge le fichier CSV avec validation des types scalaires et focus sur SHAP features.

        Args:
            file_path (str): Chemin vers le fichier CSV.

        Returns:
            pd.DataFrame: Données filtrées sur les top 50 SHAP features et volatilité.

        Raises:
            FileNotFoundError: Si le fichier est introuvable.
            ValueError: Si les données sont vides, invalides ou contiennent des valeurs non scalaires.
        """
        start_time = datetime.now()
        try:
            logger.info(f"Tentative de chargement données: {file_path}")
            self.alert_manager.send_alert(
                f"Chargement données: {file_path}", priority=2
            )
            send_telegram_alert(f"Chargement données: {file_path}")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Fichier introuvable: {file_path}")

            def read_csv():
                return pd.read_csv(file_path)

            df = self.with_retries(read_csv)
            if df is None or df.empty:
                raise ValueError(f"Fichier vide: {file_path}")
            if len(df) < PERFORMANCE_THRESHOLDS["min_rows"]:
                raise ValueError(
                    f"Nombre de lignes insuffisant: {len(df)} < {PERFORMANCE_THRESHOLDS['min_rows']}"
                )

            shap_file = BASE_DIR / "data" / "features" / "feature_importance.csv"
            if not shap_file.exists():
                raise FileNotFoundError(f"Fichier SHAP introuvable: {shap_file}")

            def read_shap():
                return pd.read_csv(shap_file)

            shap_df = self.with_retries(read_shap)
            if shap_df is None or len(shap_df) < 50:
                raise ValueError(
                    f"Nombre de SHAP features insuffisant: {len(shap_df)} < 50"
                )

            # Validation des features SHAP dans config/feature_sets.yaml
            feature_sets_path = BASE_DIR / "config" / "feature_sets.yaml"
            if feature_sets_path.exists():
                feature_config = get_config(feature_sets_path)
                shap_features = feature_config.get("inference", {}).get(
                    "shap_features", []
                )
                if len(shap_features) != 150:
                    raise ValueError(
                        f"Nombre de SHAP features incorrect dans feature_sets.yaml: {len(shap_features)} != 150"
                    )
                missing_features = [
                    col
                    for col in shap_df["feature"].head(50)
                    if col not in shap_features
                ]
                if missing_features:
                    raise ValueError(
                        f"Top 50 SHAP features non présentes dans feature_sets.yaml: {missing_features[:5]}"
                    )

            top_features = shap_df["feature"].head(50).tolist()
            required_features = top_features + ["vix_es_correlation", "atr_14"]
            available_features = [col for col in required_features if col in df.columns]
            if len(available_features) < PERFORMANCE_THRESHOLDS["min_features"]:
                raise ValueError(
                    f"Seulement {len(available_features)} des {PERFORMANCE_THRESHOLDS['min_features']} features trouvées"
                )

            critical_cols = [
                "vix_es_correlation",
                "atr_14",
                "bid_size_level_1",
                "ask_size_level_1",
                "trade_frequency_1s",
            ]
            for col in critical_cols:
                if col in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        raise ValueError(
                            f"Colonne {col} n’est pas numérique: {df[col].dtype}"
                        )
                    non_scalar = [
                        val for val in df[col] if isinstance(val, (list, dict, tuple))
                    ]
                    if non_scalar:
                        raise ValueError(
                            f"Colonne {col} contient des valeurs non scalaires: {non_scalar[:5]}"
                        )

            df = df[available_features]
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Données chargées: {file_path} ({len(df)} lignes, {len(df.columns)} features)"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "load_data",
                latency,
                success=True,
                num_rows=len(df),
                num_features=len(df.columns),
            )
            return df
        except Exception as e:
            error_msg = f"Erreur chargement données: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance("load_data", 0, success=False, error=str(e))
            raise

    def compute_correlations(
        self, df: pd.DataFrame, method: str = "pearson"
    ) -> pd.DataFrame:
        """
        Calcule la matrice de corrélations pour les colonnes numériques.

        Args:
            df (pd.DataFrame): Données à analyser.
            method (str): Méthode de corrélation ("pearson", "spearman", "kendall").

        Returns:
            pd.DataFrame: Matrice de corrélations.
        """
        start_time = datetime.now()
        try:
            logger.info(f"Calcul des corrélations avec méthode {method}")
            self.alert_manager.send_alert(
                f"Calcul des corrélations avec méthode {method}", priority=2
            )
            send_telegram_alert(f"Calcul des corrélations avec méthode {method}")
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                raise ValueError("Aucune colonne numérique trouvée")
            numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).dropna()
            if numeric_df.empty:
                raise ValueError("Aucune donnée valide après suppression des NaN/inf")

            # Calcul du confidence_drop_rate
            required_cols = ["vix_es_correlation", "atr_14"]
            available_cols = [col for col in required_cols if col in numeric_df.columns]
            confidence_drop_rate = 1.0 - min(
                (len(available_cols) / len(required_cols)), 1.0
            )
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(available_cols)}/{len(required_cols)} colonnes)"
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)

            def compute():
                return numeric_df.corr(method=method)

            corr_matrix = self.with_retries(compute)
            if corr_matrix is None:
                raise ValueError("Échec calcul corrélations")

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Matrice de corrélations calculée ({corr_matrix.shape})"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "compute_correlations",
                latency,
                success=True,
                matrix_shape=corr_matrix.shape,
                confidence_drop_rate=confidence_drop_rate,
            )
            return corr_matrix
        except Exception as e:
            error_msg = (
                f"Erreur calcul corrélations: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance("compute_correlations", 0, success=False, error=str(e))
            raise

    def detect_significant_correlations(
        self, corr_matrix: pd.DataFrame, threshold: float
    ) -> List[Tuple[str, str, float]]:
        """
        Détecte les paires de features avec une corrélation significative.

        Args:
            corr_matrix (pd.DataFrame): Matrice de corrélations.
            threshold (float): Seuil de corrélation significative.

        Returns:
            List[Tuple[str, str, float]]: Liste des paires significatives [(feature1, feature2, valeur), ...].
        """
        start_time = datetime.now()
        try:
            logger.info(f"Détection des corrélations significatives (> {threshold})")
            self.alert_manager.send_alert(
                f"Détection des corrélations significatives (> {threshold})", priority=2
            )
            send_telegram_alert(
                f"Détection des corrélations significatives (> {threshold})"
            )
            significant_pairs = []
            for col in corr_matrix.columns:
                for row in corr_matrix.index:
                    if row < col and abs(corr_matrix.loc[row, col]) > threshold:
                        significant_pairs.append((row, col, corr_matrix.loc[row, col]))
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = (
                f"{len(significant_pairs)} corrélations significatives détectées"
            )
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "detect_significant_correlations",
                latency,
                success=True,
                num_pairs=len(significant_pairs),
            )
            return significant_pairs
        except Exception as e:
            error_msg = (
                f"Erreur détection corrélations: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance(
                "detect_significant_correlations", 0, success=False, error=str(e)
            )
            raise

    def plot_heatmap(
        self,
        df: pd.DataFrame,
        output_path: str,
        figsize: tuple = (12, 10),
        threshold: float = 0.8,
    ) -> None:
        """
        Génère et sauvegarde une heatmap des corrélations avec cache local.

        Args:
            df (pd.DataFrame): Données à analyser.
            output_path (str): Chemin pour sauvegarder la heatmap.
            figsize (tuple): Taille de la figure.
            threshold (float): Seuil de corrélation significative.
        """
        start_time = datetime.now()
        try:
            logger.info(f"Génération heatmap: {output_path}")
            self.alert_manager.send_alert(
                f"Génération heatmap: {output_path}", priority=2
            )
            send_telegram_alert(f"Génération heatmap: {output_path}")

            # Calculer le hachage des données
            data_hash = hashlib.sha256(
                pd.util.hash_pandas_object(df).tobytes()
            ).hexdigest()
            cache_file = CACHE_DIR / f"heatmap_{data_hash}.png"
            if cache_file.exists():
                success_msg = f"Heatmap récupérée depuis cache: {cache_file}"
                logger.info(success_msg)
                self.alert_manager.send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)
                latency = (datetime.now() - start_time).total_seconds()
                self.log_performance("plot_heatmap_cache_hit", latency, success=True)
                return

            corr_matrix = self.compute_correlations(df)
            plt.figure(figsize=figsize)
            sns.heatmap(
                corr_matrix,
                annot=False,
                cmap="coolwarm",
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.5},
            )
            plt.title(
                f"Heatmap des Corrélations (Top 50 SHAP + Volatilité, Seuil: {threshold})",
                pad=20,
            )
            plt.tight_layout()

            def save_fig():
                plt.savefig(output_path)
                plt.savefig(cache_file)
                plt.close()

            self.with_retries(save_fig)

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Heatmap sauvegardée: {output_path}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance("plot_heatmap", latency, success=True)
            self.save_snapshot(
                "plot_heatmap",
                {"output_path": output_path, "features": list(df.columns)},
            )
        except Exception as e:
            error_msg = f"Erreur génération heatmap: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance("plot_heatmap", 0, success=False, error=str(e))
            raise

    def save_correlation_matrix(
        self, corr_matrix: pd.DataFrame, output_csv: str
    ) -> None:
        """
        Sauvegarde la matrice de corrélations dans un CSV.

        Args:
            corr_matrix (pd.DataFrame): Matrice de corrélations.
            output_csv (str): Chemin pour sauvegarder le fichier CSV.
        """
        start_time = datetime.now()
        try:
            logger.info(f"Sauvegarde matrice corrélations: {output_csv}")
            self.alert_manager.send_alert(
                f"Sauvegarde matrice corrélations: {output_csv}", priority=2
            )
            send_telegram_alert(f"Sauvegarde matrice corrélations: {output_csv}")

            def save_csv():
                corr_matrix.to_csv(output_csv)

            self.with_retries(save_csv)

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Matrice sauvegardée: {output_csv}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance("save_correlation_matrix", latency, success=True)
            self.checkpoint(corr_matrix)
            self.cloud_backup(corr_matrix)
            self.save_snapshot(
                "save_correlation_matrix",
                {"output_csv": output_csv, "matrix_shape": corr_matrix.shape},
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde matrice: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance(
                "save_correlation_matrix", 0, success=False, error=str(e)
            )
            raise

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
            **kwargs: Paramètres supplémentaires (ex. : num_rows, matrix_shape).
        """
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
            cpu_percent = psutil.cpu_percent()
            if memory_usage > 1024:
                alert_msg = f"ALERTE: Usage mémoire élevé ({memory_usage:.2f} MB)"
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=5)
                send_telegram_alert(alert_msg)
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

                def save_log():
                    if not PERF_LOG_PATH.exists():
                        log_df.to_csv(PERF_LOG_PATH, index=False, encoding="utf-8")
                    else:
                        log_df.to_csv(
                            PERF_LOG_PATH,
                            mode="a",
                            header=False,
                            index=False,
                            encoding="utf-8",
                        )

                self.with_retries(save_log)
                self.checkpoint(log_df)
                self.cloud_backup(log_df)
                self.log_buffer = []
            logger.info(
                f"Performance journalisée pour {operation}. CPU: {cpu_percent}%"
            )
        except Exception as e:
            error_msg = (
                f"Erreur journalisation performance: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)

    def main(self):
        """
        Point d’entrée avec arguments CLI pour générer une heatmap des corrélations.
        """
        parser = argparse.ArgumentParser(
            description="Génère une heatmap des corrélations entre features"
        )
        parser.add_argument(
            "--data",
            type=str,
            default=str(DEFAULT_DATA_PATH),
            help="Fichier CSV contenant les features",
        )
        parser.add_argument(
            "--output-fig",
            type=str,
            default=str(FIGURES_DIR / "correlation_heatmap.png"),
            help="Chemin pour sauvegarder la heatmap",
        )
        parser.add_argument(
            "--output-csv",
            type=str,
            default=str(CORR_MATRIX_PATH),
            help="Chemin pour sauvegarder la matrice de corrélations",
        )
        parser.add_argument(
            "--method",
            type=str,
            choices=["pearson", "spearman", "kendall"],
            default="pearson",
            help="Méthode de corrélation (pearson, spearman, kendall)",
        )
        parser.add_argument(
            "--figsize",
            type=int,
            nargs=2,
            default=[12, 10],
            help="Taille de la figure (largeur hauteur)",
        )
        parser.add_argument(
            "--config",
            type=str,
            default="config/es_config.yaml",
            help="Fichier de configuration pour les seuils",
        )
        args = parser.parse_args()

        start_time = datetime.now()
        try:
            logger.info("Début analyse corrélations")
            self.alert_manager.send_alert("Début analyse corrélations", priority=2)
            send_telegram_alert("Début analyse corrélations")

            config = self.load_config(args.config)
            threshold = config["significant_threshold"]

            df = self.load_data(args.data)
            corr_matrix = self.compute_correlations(df, method=args.method)
            significant_pairs = self.detect_significant_correlations(
                corr_matrix, threshold
            )
            for pair in significant_pairs:
                logger.info(
                    f"Corrélation significative: {pair[0]} - {pair[1]} = {pair[2]:.3f}"
                )
                self.alert_manager.send_alert(
                    f"Corrélation significative: {pair[0]} - {pair[1]} = {pair[2]:.3f}",
                    priority=2,
                )
                send_telegram_alert(
                    f"Corrélation significative: {pair[0]} - {pair[1]} = {pair[2]:.3f}"
                )

            self.plot_heatmap(df, args.output_fig, tuple(args.figsize), threshold)
            self.save_correlation_matrix(corr_matrix, args.output_csv)

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Analyse corrélations terminée"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "main",
                latency,
                success=True,
                num_features=len(df.columns),
                num_pairs=len(significant_pairs),
            )

        except Exception as e:
            error_msg = (
                f"Erreur analyse corrélations: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance("main", 0, success=False, error=str(e))
            raise


if __name__ == "__main__":
    heatmap = CorrelationHeatmap()
    heatmap.main()
