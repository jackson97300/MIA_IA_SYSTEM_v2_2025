# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/monitoring/data_drift.py
# Rôle : Détecte les dérives dans les données entre entraînement et live pour MIA_IA_SYSTEM_v2_2025, avec focus sur SHAP features.
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, scipy>=1.10.0,<2.0.0, matplotlib>=3.8.0,<4.0.0,
#   psutil>=5.9.8,<6.0.0, loguru>=0.7.0,<1.0.0, pyyaml>=6.0.0,<7.0.0, boto3>=1.26.0,<2.0.0,
#   argparse, os, signal, gzip
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/features/feature_pipeline.py
# - src/features/shap_weighting.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - Données d’entraînement et live (pd.DataFrame avec 350 features pour entraînement ou 150 SHAP features pour inférence)
# - data/features/feature_importance.csv (top 50 SHAP features)
# - config/es_config.yaml
# - config/feature_sets.yaml
#
# Outputs :
# - Rapport de dérive dans data/logs/drift_report.csv
# - Snapshots dans data/cache/drift/*.json.gz
# - Sauvegardes incrémentielles dans data/checkpoints/drift_*.json.gz
# - Graphiques dans data/figures/drift/
# - Logs dans data/logs/data_drift.log
# - Logs de performance dans data/logs/drift_performance.csv
#
# Notes :
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence) définies dans config/feature_sets.yaml.
# - Utilise IQFeed exclusivement via feature_pipeline.py pour les données d’entrée.
# - Supprime toute référence à 320 features, 81 features, et obs_t.
# - Implémente retries (max 3, délai 2^attempt secondes) pour les opérations critiques.
# - Intègre SHAP (méthode 17) pour les top 50 features et volatilité (méthode 1) avec vix_es_correlation.
# - Alertes via alert_manager.py et Telegram pour dérives critiques (priorité ≥ 4).
# - Sauvegardes incrémentielles toutes les 5 min dans data/checkpoints/ (5 versions).
# - Sauvegardes distribuées toutes les 15 min vers S3 (configurable).
# - Tests unitaires disponibles dans tests/test_data_drift.py.
# - Phases intégrées : Phase 8 (auto-conscience via confidence_drop_rate), Phase 17 (SHAP).
# - Validation complète prévue pour juin 2025.

import argparse
import gzip
import json
import os
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import ks_2samp, ttest_ind, wasserstein_distance

from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import get_config
from src.utils.telegram_alert import send_telegram_alert

# Configuration logging
logger.remove()
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOG_DIR = BASE_DIR / "data" / "logs"
FIGURES_DIR = BASE_DIR / "data" / "figures" / "drift"
CACHE_DIR = BASE_DIR / "data" / "cache" / "drift"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints"
PERF_LOG_PATH = LOG_DIR / "drift_performance.csv"
DRIFT_REPORT_PATH = LOG_DIR / "drift_report.csv"
LOG_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
logger.add(LOG_DIR / "data_drift.log", rotation="10 MB", level="INFO", encoding="utf-8")
logger.add(lambda msg: print(msg, end=""), level="INFO")

# Seuils de performance globaux
PERFORMANCE_THRESHOLDS = {
    "min_rows": 50,  # Nombre minimum de lignes pour détection fiable
    "min_features": 50,  # Minimum pour top 50 SHAP features
}

# Constantes
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Secondes, exponentiel

# Variable pour gérer l'arrêt propre
RUNNING = True


class DataDriftDetector:
    """
    Classe pour détecter les dérives dans les données, avec focus sur SHAP features et volatilité.
    """

    def __init__(self, config_path: str = "config/es_config.yaml"):
        """
        Initialise le détecteur de dérives.

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
            success_msg = "DataDriftDetector initialisé"
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
            error_msg = f"Erreur initialisation DataDriftDetector: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            raise

    def handle_sigint(self, signum, frame):
        """Gère l'arrêt propre sur SIGINT."""
        global RUNNING
        datetime.now()
        snapshot = {"timestamp": datetime.now().isoformat(), "status": "SIGINT"}
        CACHE_DIR / f'drift_sigint_{snapshot["timestamp"]}.json.gz'
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
                logger.warning(alert_msg)
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
        Sauvegarde incrémentielle des rapports de dérive toutes les 5 minutes avec versionnage (5 versions).

        Args:
            data (pd.DataFrame): Données du rapport de dérive à sauvegarder.
        """
        try:
            start_time = datetime.now()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_data = {
                "timestamp": timestamp,
                "num_rows": len(data),
                "columns": list(data.columns),
            }
            checkpoint_path = CHECKPOINT_DIR / f"drift_{timestamp}.json.gz"
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
        Sauvegarde distribuée des rapports de dérive vers AWS S3 toutes les 15 minutes (configurable).

        Args:
            data (pd.DataFrame): Données du rapport de dérive à sauvegarder.
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
            backup_path = f"{self.config['s3_prefix']}drift_{timestamp}.csv.gz"
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
                "drift_params",
                {
                    "wass_threshold": 0.1,
                    "ks_threshold": 0.05,
                    "vix_drift_threshold": 0.5,
                    "s3_bucket": None,
                    "s3_prefix": "drift/",
                },
            )
            required_keys = [
                "wass_threshold",
                "ks_threshold",
                "vix_drift_threshold",
                "s3_bucket",
                "s3_prefix",
            ]
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
                "wass_threshold": 0.1,
                "ks_threshold": 0.05,
                "vix_drift_threshold": 0.5,
                "s3_bucket": None,
                "s3_prefix": "drift/",
            }

    def load_data(self, file_path: str, label: str) -> pd.DataFrame:
        """
        Charge un fichier CSV avec validation des types scalaires et focus sur SHAP features.

        Args:
            file_path (str): Chemin vers le fichier CSV.
            label (str): Étiquette pour identifier les données ("training" ou "live").

        Returns:
            pd.DataFrame: Données filtrées sur les top 50 SHAP features et volatilité.

        Raises:
            FileNotFoundError: Si le fichier est introuvable.
            ValueError: Si les données sont vides, invalides ou contiennent des valeurs non scalaires.
        """
        start_time = datetime.now()
        try:
            logger.info(f"Tentative de chargement données {label}: {file_path}")
            self.alert_manager.send_alert(
                f"Chargement données {label}: {file_path}", priority=2
            )
            send_telegram_alert(f"Chargement données {label}: {file_path}")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Fichier {label} introuvable: {file_path}")

            def read_csv():
                return pd.read_csv(file_path)

            df = self.with_retries(read_csv)
            if df is None or df.empty:
                raise ValueError(f"Fichier {label} vide: {file_path}")
            if len(df) < PERFORMANCE_THRESHOLDS["min_rows"]:
                raise ValueError(
                    f"Nombre de lignes insuffisant dans {label}: {len(df)} < {PERFORMANCE_THRESHOLDS['min_rows']}"
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
            required_features = top_features + ["vix_es_correlation"]
            available_features = [col for col in required_features if col in df.columns]
            if len(available_features) < PERFORMANCE_THRESHOLDS["min_features"]:
                raise ValueError(
                    f"Seulement {len(available_features)} des {PERFORMANCE_THRESHOLDS['min_features']} features trouvées dans {label}"
                )

            critical_cols = [
                "vix_es_correlation",
                "bid_size_level_1",
                "ask_size_level_1",
                "trade_frequency_1s",
            ]
            for col in critical_cols:
                if col in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        raise ValueError(
                            f"Colonne {col} n’est pas numérique dans {label}: {df[col].dtype}"
                        )
                    non_scalar = [
                        val for val in df[col] if isinstance(val, (list, dict, tuple))
                    ]
                    if non_scalar:
                        raise ValueError(
                            f"Colonne {col} contient des valeurs non scalaires dans {label}: {non_scalar[:5]}"
                        )

            df = df[available_features]
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Données {label} chargées: {file_path} ({len(df)} lignes, {len(df.columns)} features)"
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
            error_msg = (
                f"Erreur chargement données {label}: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance("load_data", 0, success=False, error=str(e))
            raise

    def compute_drift_metrics(
        self, train_df: pd.DataFrame, live_df: pd.DataFrame, features: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calcule les métriques de dérive pour chaque feature, avec alertes pour vix_es_correlation.

        Args:
            train_df (pd.DataFrame): Données d’entraînement.
            live_df (pd.DataFrame): Données live.
            features (List[str]): Liste des features à analyser.

        Returns:
            Dict[str, Dict[str, float]]: Résultats des métriques de dérive par feature.
        """
        start_time = datetime.now()
        try:
            logger.info("Calcul des métriques de dérive")
            self.alert_manager.send_alert("Calcul des métriques de dérive", priority=2)
            send_telegram_alert("Calcul des métriques de dérive")
            drift_results = {}

            # Calcul du confidence_drop_rate
            required_cols = ["vix_es_correlation"]
            available_cols = [
                col
                for col in required_cols
                if col in train_df.columns and col in live_df.columns
            ]
            confidence_drop_rate = 1.0 - min(
                (len(available_cols) / len(required_cols)), 1.0
            )
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(available_cols)}/{len(required_cols)} colonnes)"
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)

            for feature in features:
                if feature in train_df.columns and feature in live_df.columns:
                    train_data = (
                        train_df[feature].replace([np.inf, -np.inf], np.nan).dropna()
                    )
                    live_data = (
                        live_df[feature].replace([np.inf, -np.inf], np.nan).dropna()
                    )

                    if len(train_data) == 0 or len(live_data) == 0:
                        drift_results[feature] = {
                            "wasserstein": np.nan,
                            "ks_stat": np.nan,
                            "ks_pvalue": np.nan,
                            "ttest_pvalue": np.nan,
                        }
                        warning_msg = f"Données insuffisantes pour {feature} après suppression NaN/inf"
                        logger.warning(warning_msg)
                        self.alert_manager.send_alert(warning_msg, priority=3)
                        send_telegram_alert(warning_msg)
                        continue

                    wass_dist = wasserstein_distance(train_data, live_data)
                    ks_stat, ks_pvalue = ks_2samp(train_data, live_data)
                    t_stat, t_pvalue = ttest_ind(train_data, live_data, equal_var=False)

                    drift_results[feature] = {
                        "wasserstein": wass_dist,
                        "ks_stat": ks_stat,
                        "ks_pvalue": ks_pvalue,
                        "ttest_pvalue": t_pvalue,
                    }

                    if feature == "vix_es_correlation" and wass_dist > self.config.get(
                        "vix_drift_threshold", 0.5
                    ):
                        alert_msg = f"Dérive critique détectée pour {feature}: Wasserstein={wass_dist:.3f}"
                        logger.warning(alert_msg)
                        self.alert_manager.send_alert(alert_msg, priority=4)
                        send_telegram_alert(alert_msg)
                else:
                    drift_results[feature] = {
                        "wasserstein": np.nan,
                        "ks_stat": np.nan,
                        "ks_pvalue": np.nan,
                        "ttest_pvalue": np.nan,
                    }
                    warning_msg = f"Feature {feature} absente dans un dataset"
                    logger.warning(warning_msg)
                    self.alert_manager.send_alert(warning_msg, priority=3)
                    send_telegram_alert(warning_msg)

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = (
                f"Calcul des métriques terminé ({len(drift_results)} features)"
            )
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "compute_drift_metrics",
                latency,
                success=True,
                num_features=len(features),
                confidence_drop_rate=confidence_drop_rate,
            )
            return drift_results
        except Exception as e:
            error_msg = f"Erreur calcul métriques: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance(
                "compute_drift_metrics", 0, success=False, error=str(e)
            )
            raise

    def plot_drift(
        self,
        drift_results: Dict[str, Dict[str, float]],
        threshold_wass: float = 0.1,
        threshold_ks: float = 0.05,
    ) -> None:
        """
        Génère un graphique des dérives détectées.

        Args:
            drift_results (Dict[str, Dict[str, float]]): Résultats des métriques de dérive.
            threshold_wass (float): Seuil pour la distance de Wasserstein.
            threshold_ks (float): Seuil pour les p-values KS et t-test.
        """
        start_time = datetime.now()
        try:
            logger.info("Génération graphique des dérives")
            self.alert_manager.send_alert(
                "Génération graphique des dérives", priority=2
            )
            send_telegram_alert("Génération graphique des dérives")

            features = list(drift_results.keys())
            wass_values = [drift_results[f]["wasserstein"] for f in features]
            ks_pvalues = [drift_results[f]["ks_pvalue"] for f in features]
            ttest_pvalues = [drift_results[f]["ttest_pvalue"] for f in features]

            significant_drift = [
                f
                for f, d in drift_results.items()
                if (
                    d["wasserstein"] > threshold_wass
                    or d["ks_pvalue"] < threshold_ks
                    or d["ttest_pvalue"] < threshold_ks
                )
                and not np.isnan(d["wasserstein"])
            ]
            for f in significant_drift:
                alert_msg = f"Dérive significative détectée pour {f}"
                logger.info(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=2)
                send_telegram_alert(alert_msg)

            fig, ax1 = plt.subplots(figsize=(14, 8))
            ax1.bar(
                features,
                wass_values,
                color="skyblue",
                alpha=0.7,
                label="Distance Wasserstein",
            )
            ax1.set_ylabel("Distance Wasserstein", color="skyblue")
            ax1.axhline(
                y=threshold_wass,
                color="red",
                linestyle="--",
                label=f"Seuil Wasserstein ({threshold_wass})",
            )
            ax1.tick_params(axis="y", labelcolor="skyblue")
            ax1.set_xticklabels(features, rotation=90)

            ax2 = ax1.twinx()
            ax2.plot(
                features, ks_pvalues, color="orange", marker="o", label="p-value KS"
            )
            ax2.plot(
                features,
                ttest_pvalues,
                color="green",
                marker="x",
                label="p-value t-test",
            )
            ax2.set_ylabel("p-values", color="black")
            ax2.axhline(
                y=threshold_ks,
                color="purple",
                linestyle="--",
                label=f"Seuil p-value ({threshold_ks})",
            )
            ax2.tick_params(axis="y", labelcolor="black")

            plt.title("Détection de Dérive des Données (Entraînement vs Live)")
            fig.legend(
                loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes
            )
            plt.tight_layout()

            output_path = (
                FIGURES_DIR
                / f"data_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )

            def save_fig():
                plt.savefig(output_path)
                plt.close()

            self.with_retries(save_fig)

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Graphique sauvegardé: {output_path}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance("plot_drift", latency, success=True)
            self.save_snapshot(
                "plot_drift",
                {
                    "output_path": str(output_path),
                    "significant_drift": significant_drift,
                },
            )
        except Exception as e:
            error_msg = (
                f"Erreur génération graphique: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance("plot_drift", 0, success=False, error=str(e))
            raise

    def save_drift_report(
        self, drift_results: Dict[str, Dict[str, float]], output_path: str
    ) -> None:
        """
        Sauvegarde les résultats dans un CSV.

        Args:
            drift_results (Dict[str, Dict[str, float]]): Résultats des métriques de dérive.
            output_path (str): Chemin pour sauvegarder le fichier CSV.
        """
        start_time = datetime.now()
        try:
            logger.info(f"Sauvegarde rapport dérive: {output_path}")
            self.alert_manager.send_alert(
                f"Sauvegarde rapport dérive: {output_path}", priority=2
            )
            send_telegram_alert(f"Sauvegarde rapport dérive: {output_path}")

            df_report = pd.DataFrame.from_dict(drift_results, orient="index")
            df_report["timestamp"] = datetime.now().isoformat()

            def save_csv():
                if not os.path.exists(output_path):
                    df_report.to_csv(
                        output_path, index_label="feature", encoding="utf-8"
                    )
                else:
                    df_report.to_csv(
                        output_path,
                        mode="a",
                        header=False,
                        index_label="feature",
                        encoding="utf-8",
                    )

            self.with_retries(save_csv)

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Rapport sauvegardé: {output_path}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "save_drift_report",
                latency,
                success=True,
                num_features=len(drift_results),
            )
            self.checkpoint(df_report)
            self.cloud_backup(df_report)
            self.save_snapshot(
                "save_drift_report",
                {"output_path": output_path, "num_features": len(drift_results)},
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde rapport: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance("save_drift_report", 0, success=False, error=str(e))
            raise

    def main(self):
        """
        Point d’entrée avec arguments CLI pour détecter les dérives de données.
        """
        parser = argparse.ArgumentParser(
            description="Détection de dérive entre données d’entraînement et live"
        )
        parser.add_argument(
            "--train",
            type=str,
            default=str(BASE_DIR / "data" / "features" / "features_train.csv"),
            help="Fichier CSV des données d’entraînement",
        )
        parser.add_argument(
            "--live",
            type=str,
            default=str(BASE_DIR / "data" / "features" / "features_latest.csv"),
            help="Fichier CSV des données live",
        )
        parser.add_argument(
            "--output",
            type=str,
            default=str(DRIFT_REPORT_PATH),
            help="Fichier de sortie pour le rapport",
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
            logger.info("Début détection dérive")
            self.alert_manager.send_alert("Début détection dérive", priority=2)
            send_telegram_alert("Début détection dérive")

            config = self.load_config(args.config)
            wass_threshold = config["wass_threshold"]
            ks_threshold = config["ks_threshold"]

            train_df = self.load_data(args.train, "training")
            live_df = self.load_data(args.live, "live")

            shap_file = BASE_DIR / "data" / "features" / "feature_importance.csv"
            shap_df = pd.read_csv(shap_file)
            features = shap_df["feature"].head(50).tolist() + ["vix_es_correlation"]

            drift_results = self.compute_drift_metrics(train_df, live_df, features)
            self.plot_drift(drift_results, wass_threshold, ks_threshold)
            self.save_drift_report(drift_results, args.output)

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Détection dérive terminée"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "main", latency, success=True, num_features=len(features)
            )

        except Exception as e:
            error_msg = f"Erreur détection dérive: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance("main", 0, success=False, error=str(e))
            raise


if __name__ == "__main__":
    detector = DataDriftDetector()
    detector.main()
