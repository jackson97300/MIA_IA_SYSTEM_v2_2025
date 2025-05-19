```python
# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/monitoring/mia_dashboard.py
# Tableau de bord interactif Streamlit pour visualiser les performances de trading, régimes, SHAP features, et prédictions LSTM.
#
# Version : 2.1.6
# Date : 2025-05-15
#
# Rôle : Affiche les performances de trading, les régimes hybrides, les top 150 SHAP features, et les prédictions LSTM
#        (predicted_vix) avec intégration des nouvelles features bid_ask_imbalance, trade_aggressiveness, iv_skew,
#        iv_term_structure, option_skew, news_impact_score. Inclut confidence_drop_rate, validation obs_t, cache LRU,
#        métriques psutil (memory_usage_mb, cpu_usage_percent), et optimisations pour éviter la surcharge mémoire.
#
# Dépendances :
# - streamlit>=1.20.0,<2.0.0, pandas>=2.0.0,<3.0.0, plotly>=5.0.0,<6.0.0, numpy>=1.26.4,<2.0.0,
#   matplotlib>=3.8.0,<4.0.0, psutil>=5.9.8,<6.0.0, loguru>=0.7.0,<1.0.0, pyyaml>=6.0.0,<7.0.0,
#   boto3>=1.26.0,<2.0.0, os, signal, hashlib, gzip
# - src/model/router/detect_regime.py
# - src/features/neural_pipeline.py
# - src/features/shap_weighting.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - data/features/features_latest.csv (351 features pour entraînement ou 150 SHAP features pour inférence)
# - data/trades/trades_simulated.csv
# - data/features/feature_importance.csv (top 150 SHAP features)
# - data/neural_pipeline_dashboard.json (confidence_drop_rate, new_features)
# - config/market_config.yaml
# - config/feature_sets.yaml
#
# Outputs :
# - Tableau de bord Streamlit (port configurable)
# - Visualisations matplotlib dans data/figures/monitoring/
# - Snapshots dans data/cache/dashboard/*.json.gz
# - Sauvegardes incrémentielles dans data/checkpoints/dashboard_*.json.gz
# - Logs dans data/logs/mia_dashboard.log
# - Logs de performance dans data/logs/dashboard_performance.csv
# - Statut dans data/market_config_dashboard.json
#
# Notes :
# - Utilise exclusivement IQFeed via data_provider.py pour les données d’entrée.
# - Compatible avec 351 features (entraînement) et 150 SHAP features (inférence) définies dans config/feature_sets.yaml.
# - Intègre les Phases 1-18 :
#   - Phase 1 (news_scraper.py, news_analyzer.py) : Visualisation de news_impact_score.
#   - Phase 13 (orderflow_indicators.py, options_metrics.py) : Visualisation de vix_es_correlation, call_iv_atm, option_skew.
#   - Phase 15 (microstructure_guard.py, spotgamma_recalculator.py) : Visualisation de spoofing_score, volume_anomaly, net_gamma, call_wall.
#   - Phase 18 : Visualisation de trade_velocity, hft_activity_score, bid_ask_imbalance, trade_aggressiveness, iv_skew, iv_term_structure.
# - Implémente retries (max 3, délai 2^attempt secondes) pour les opérations critiques.
# - Intègre régimes hybrides (méthode 11), SHAP (méthode 17), et LSTM (méthode 12).
# - Cache local dans data/cache/dashboard/ avec gestion LRU via OrderedDict, stockant uniquement les chemins des figures.
# - Sauvegardes incrémentielles toutes les 5 min dans data/checkpoints/ (5 versions).
# - Sauvegardes distribuées toutes les 15 min vers S3 (configurable, clés AWS via ~/.aws/credentials ou variables d’environnement).
# - Alertes via alert_manager.py et Telegram pour erreurs critiques (priorité ≥ 4), conditionnées par notifications_enabled.
# - Tests unitaires disponibles dans tests/test_mia_dashboard.py, à renforcer avec mocks pour psutil, boto3, telegram_alert.
# - Corrections appliquées :
#   - Suppression du mot isolé 'one' dans handle_sigint pour corriger l’erreur de syntaxe.
#   - Changement du chemin de config par défaut de es_config.yaml à market_config.yaml.
#   - Lecture de la section 'dashboard' au lieu de 'dashboard_params' dans load_config, avec prise en compte de notifications_enabled et status_file.
#   - Suppression de regime_path du calcul de hachage car non utilisé.
#   - Utilisation de notifications_enabled pour conditionner les alertes et écriture de status_file.

import gzip
import hashlib
import json
import os
import signal
import time
import traceback
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psutil
import streamlit as st
from loguru import logger

from src.features.neural_pipeline import NeuralPipeline
from src.model.router.detect_regime import MarketRegimeDetector
from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import get_config
from src.utils.telegram_alert import send_telegram_alert

# Configuration logging
logger.remove()
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOG_DIR = BASE_DIR / "data" / "logs"
TRADES_DIR = BASE_DIR / "data" / "trades"
FIGURES_DIR = BASE_DIR / "data" / "figures" / "monitoring"
CACHE_DIR = BASE_DIR / "data" / "cache" / "dashboard"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints"
PERF_LOG_PATH = LOG_DIR / "dashboard_performance.csv"
DASHBOARD_JSON_PATH = BASE_DIR / "data" / "neural_pipeline_dashboard.json"
LOG_DIR.mkdir(exist_ok=True)
TRADES_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "mia_dashboard.log", rotation="10 MB", level="INFO", encoding="utf-8"
)
logger.add(lambda msg: print(msg, end=""), level="INFO")

# Seuils minimum
PERFORMANCE_THRESHOLDS = {
    "min_rows": 50,  # Nombre minimum de lignes
    "min_features": 50,  # Minimum pour top 50 SHAP features
}

# Constantes
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Secondes, exponentiel
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB

# Variable pour gérer l'arrêt propre
RUNNING = True

# ------------------------- CORE ------------------------- #

class MIADashboard:
    """
    Classe pour gérer le tableau de bord interactif Streamlit (logique métier).
    """

    def __init__(self, config_path: str = "config/market_config.yaml"):
        """
        Initialise le tableau de bord.

        Args:
            config_path (str): Chemin du fichier de configuration.
        """
        self.alert_manager = AlertManager()
        self.checkpoint_versions = []
        self.log_buffer = []
        self.cache = OrderedDict()  # Cache LRU avec OrderedDict
        self.dashboard_cache = {}  # Cache pour neural_pipeline_dashboard.json
        signal.signal(signal.SIGINT, self.handle_sigint)

        start_time = datetime.now()
        try:
            self.config = self.load_config(config_path)
            self.neural_pipeline = NeuralPipeline(
                window_size=50,
                base_features=150,
                config_path=str(BASE_DIR / "config" / "model_params.yaml"),
            )
            self._clean_cache()  # Nettoyer le cache à l'initialisation
            success_msg = "MIADashboard initialisé avec succès"
            logger.info(success_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(success_msg, priority=2)
                send_telegram_alert(success_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "init",
                latency,
                success=True,
                memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
                cpu_usage_percent=psutil.cpu_percent(),
            )
            self.save_snapshot(
                "init",
                {"config_path": config_path, "timestamp": datetime.now().isoformat()},
            )
        except Exception as e:
            error_msg = f"Erreur initialisation MIADashboard: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            raise

    def handle_sigint(self, signum, frame):
        """Gère l'arrêt propre sur SIGINT."""
        global RUNNING
        start_time = datetime.now()
        snapshot = {"timestamp": datetime.now().isoformat(), "status": "SIGINT"}
        try:
            RUNNING = False
            self.save_snapshot("sigint", snapshot)
            success_msg = "Arrêt propre sur SIGINT, snapshot sauvegardé"
            logger.info(success_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(success_msg, priority=2)
                send_telegram_alert(success_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "handle_sigint",
                latency,
                success=True,
                memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
                cpu_usage_percent=psutil.cpu_percent(),
            )
            exit(0)
        except Exception as e:
            error_msg = f"Erreur sauvegarde SIGINT: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
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
                if self.config.get("notifications_enabled", True):
                    self.alert_manager.send_alert(alert_msg, priority=3)
                    send_telegram_alert(alert_msg)
            latency = (datetime.now() - start_time).total_seconds()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            success_msg = f"Snapshot {snapshot_type} sauvegardé: {save_path}"
            logger.info(success_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)
            self.log_performance(
                "save_snapshot",
                latency,
                success=True,
                snapshot_size_mb=file_size,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_percent,
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
            self.log_performance("save_snapshot", 0, success=False, error=str(e))

    def checkpoint(self, data: pd.DataFrame, data_type: str = "features") -> None:
        """
        Sauvegarde incrémentielle des données du tableau de bord toutes les 5 minutes avec versionnage (5 versions).

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données ("features" ou "trades").
        """
        try:
            start_time = datetime.now()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_data = {
                "timestamp": timestamp,
                "num_rows": len(data),
                "columns": list(data.columns),
                "data_type": data_type,
            }
            checkpoint_path = (
                CHECKPOINT_DIR / f"dashboard_{data_type}_{timestamp}.json.gz"
            )
            CHECKPOINT_DIR.mkdir(exist_ok=True)

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
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            success_msg = f"Checkpoint sauvegardé: {checkpoint_path}"
            logger.info(success_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)
            self.log_performance(
                "checkpoint",
                latency,
                success=True,
                file_size_mb=file_size,
                num_rows=len(data),
                data_type=data_type,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_percent,
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde checkpoint: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
            self.log_performance(
                "checkpoint", 0, success=False, error=str(e), data_type=data_type
            )

    def cloud_backup(self, data: pd.DataFrame, data_type: str = "features") -> None:
        """
        Sauvegarde distribuée des données du tableau de bord vers AWS S3 toutes les 15 minutes (configurable).

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données ("features" ou "trades").
        """
        try:
            start_time = datetime.now()
            if not self.config.get("s3_bucket"):
                warning_msg = "S3 bucket non configuré, sauvegarde cloud ignorée. Vérifiez ~/.aws/credentials ou AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY."
                logger.warning(warning_msg)
                if self.config.get("notifications_enabled", True):
                    self.alert_manager.send_alert(warning_msg, priority=3)
                    send_telegram_alert(warning_msg)
                return
            # Vérification des clés AWS
            if not (
                os.environ.get("AWS_ACCESS_KEY_ID")
                and os.environ.get("AWS_SECRET_ACCESS_KEY")
            ) and not os.path.exists(os.path.expanduser("~/.aws/credentials")):
                warning_msg = "Clés AWS non trouvées. Configurez ~/.aws/credentials ou AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY."
                logger.warning(warning_msg)
                if self.config.get("notifications_enabled", True):
                    self.alert_manager.send_alert(warning_msg, priority=3)
                    send_telegram_alert(warning_msg)
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = (
                f"{self.config['s3_prefix']}dashboard_{data_type}_{timestamp}.csv.gz"
            )
            temp_path = CHECKPOINT_DIR / f"temp_s3_{timestamp}.csv.gz"

            def write_temp():
                data.to_csv(
                    temp_path, compression="gzip", index=False, encoding="utf-8"
                )

            self.with_retries(write_temp)
            s3_client = boto3.client("s3")

            def upload_s3():
                s3_client.upload_file(temp_path, self.config["s3_bucket"], backup_path)

            self.with_retries(upload_s3)
            os.remove(temp_path)
            latency = (datetime.now() - start_time).total_seconds()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            success_msg = f"Sauvegarde cloud S3 effectuée: {backup_path}"
            logger.info(success_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)
            self.log_performance(
                "cloud_backup",
                latency,
                success=True,
                num_rows=len(data),
                data_type=data_type,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_percent,
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde cloud S3: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
            self.log_performance(
                "cloud_backup", 0, success=False, error=str(e), data_type=data_type
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
        start_time = datetime.now()
        for attempt in range(max_attempts):
            try:
                result = func()
                latency = (datetime.now() - start_time).total_seconds()
                memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
                cpu_percent = psutil.cpu_percent()
                self.log_performance(
                    f"retry_attempt_{attempt+1}",
                    latency,
                    success=True,
                    attempt_number=attempt + 1,
                    memory_usage_mb=memory_usage,
                    cpu_usage_percent=cpu_percent,
                )
                return result
            except Exception as e:
                if attempt == max_attempts - 1:
                    error_msg = f"Échec après {max_attempts} tentatives: {str(e)}\n{traceback.format_exc()}"
                    logger.error(error_msg)
                    if self.config.get("notifications_enabled", True):
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
                if self.config.get("notifications_enabled", True):
                    self.alert_manager.send_alert(warning_msg, priority=3)
                    send_telegram_alert(warning_msg)
                time.sleep(delay)

    def _clean_cache(self, max_size_mb: float = MAX_CACHE_SIZE_MB):
        """
        Supprime les fichiers cache expirés ou si la taille dépasse max_size_mb.
        """
        start_time = datetime.now()

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
                    and (time.time() - os.path.getmtime(path)) > CACHE_EXPIRATION
                ):
                    os.remove(path)

        try:
            self.with_retries(clean)
            latency = (datetime.now() - start_time).total_seconds()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            success_msg = "Cache nettoyé avec succès"
            logger.info(success_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)
            self.log_performance(
                "clean_cache",
                latency,
                success=True,
                data_type="cache",
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_percent,
            )
        except OSError as e:
            error_msg = f"Erreur nettoyage cache: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
            self.log_performance(
                "clean_cache", latency, success=False, error=str(e), data_type="cache"
            )

    def load_config(self, config_path: str) -> Dict:
        """
        Charge la configuration avec repli sur les valeurs par défaut.

        Args:
            config_path (str): Chemin du fichier de configuration.

        Returns:
            Dict: Configuration chargée.
        """
        start_time = datetime.now()
        try:
            config = get_config(BASE_DIR / config_path)
            if not config:
                raise ValueError("Configuration vide ou non trouvée")
            params = config.get(
                "dashboard",
                {
                    "interval": 10000,
                    "max_rows": 100,
                    "s3_bucket": None,
                    "s3_prefix": "dashboard/",
                    "notifications_enabled": True,
                    "status_file": "data/market_config_dashboard.json",
                    "logging": {"buffer_size": 100},
                },
            )
            required_keys = [
                "interval",
                "max_rows",
                "s3_bucket",
                "s3_prefix",
                "notifications_enabled",
                "status_file",
            ]
            missing_keys = [key for key in required_keys if key not in params]
            if missing_keys:
                raise ValueError(f"Clés de configuration manquantes: {missing_keys}")
            latency = (datetime.now() - start_time).total_seconds()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            success_msg = f"Configuration {config_path} chargée"
            logger.info(success_msg)
            if params.get("notifications_enabled", True):
                self.alert_manager.send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)
            self.log_performance(
                "load_config",
                latency,
                success=True,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_percent,
            )
            return params
        except Exception as e:
            error_msg = (
                f"Erreur chargement configuration: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
            self.log_performance("load_config", 0, success=False, error=str(e))
            return {
                "interval": 10000,
                "max_rows": 100,
                "s3_bucket": None,
                "s3_prefix": "dashboard/",
                "notifications_enabled": True,
                "status_file": "data/market_config_dashboard.json",
                "logging": {"buffer_size": 100},
            }

    def log_performance(
        self,
        operation: str,
        latency: float,
        success: bool,
        error: str = None,
        num_rows: int = None,
        step: int = None,
        attempt_number: int = None,
        data_type: str = None,
        num_features: int = None,
        snapshot_size_mb: float = None,
        confidence_drop_rate: float = None,
        memory_usage_mb: float = None,
        cpu_usage_percent: float = None,
    ):
        """
        Journalise les performances des opérations.

        Args:
            operation (str): Nom de l'opération.
            latency (float): Temps d'exécution (secondes).
            success (bool): Succès de l'opération.
            error (str, optional): Message d'erreur.
            num_rows (int, optional): Nombre de lignes traitées.
            step (int, optional): Étape de traitement.
            attempt_number (int, optional): Numéro de tentative.
            data_type (str, optional): Type de données.
            num_features (int, optional): Nombre de features.
            snapshot_size_mb (float, optional): Taille du snapshot.
            confidence_drop_rate (float, optional): Taux de confiance.
            memory_usage_mb (float, optional): Utilisation mémoire en MB.
            cpu_usage_percent (float, optional): Utilisation CPU en %.
        """
        start_time = datetime.now()
        try:
            memory_usage = (
                memory_usage_mb or psutil.Process().memory_info().rss / 1024 / 1024
            )
            cpu_percent = cpu_usage_percent or psutil.cpu_percent()
            if memory_usage > 1024:
                alert_msg = f"ALERTE: Usage mémoire élevé ({memory_usage:.2f} MB)"
                logger.warning(alert_msg)
                if self.config.get("notifications_enabled", True):
                    self.alert_manager.send_alert(alert_msg, priority=5)
                    send_telegram_alert(alert_msg)
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "data_type": data_type,
                "latency": latency,
                "success": success,
                "error": error,
                "memory_mb": memory_usage,
                "cpu_percent": cpu_percent,
                "num_rows": num_rows,
                "step": step,
                "attempt_number": attempt_number,
                "num_features": num_features,
                "snapshot_size_mb": snapshot_size_mb,
                "confidence_drop_rate": confidence_drop_rate,
            }
            self.log_buffer.append(log_entry)
            if len(self.log_buffer) >= self.config.get("logging", {}).get(
                "buffer_size", 100
            ):
                df = pd.DataFrame(self.log_buffer)

                def save_log():
                    df.to_csv(
                        PERF_LOG_PATH,
                        mode="a",
                        header=not os.path.exists(PERF_LOG_PATH),
                        index=False,
                        encoding="utf-8",
                    )

                self.with_retries(save_log)
                self.checkpoint(df, data_type="performance_logs")
                self.cloud_backup(df, data_type="performance_logs")
                self.log_buffer = []
            latency = (datetime.now() - start_time).total_seconds()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            logger.info(
                f"Performance journalisée pour {operation}. CPU: {cpu_percent}%"
            )
            self.log_performance(
                "log_performance",
                latency,
                success=True,
                data_type="logging",
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_percent,
            )
        except Exception as e:
            error_msg = (
                f"Erreur journalisation performance: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
            self.log_performance(
                "log_performance", 0, success=False, error=str(e), data_type="logging"
            )

    def save_status(self, status_data: Dict) -> None:
        """
        Sauvegarde l'état du tableau de bord dans status_file.

        Args:
            status_data (Dict): Données de statut à sauvegarder.
        """
        try:
            start_time = datetime.now()
            status_file = self.config.get("status_file", "data/market_config_dashboard.json")
            status_path = BASE_DIR / status_file
            status_path.parent.mkdir(exist_ok=True)

            def write_status():
                with open(status_path, "w", encoding="utf-8") as f:
                    json.dump(status_data, f, indent=4)

            self.with_retries(write_status)
            latency = (datetime.now() - start_time).total_seconds()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            success_msg = f"Statut sauvegardé: {status_path}"
            logger.info(success_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)
            self.log_performance(
                "save_status",
                latency,
                success=True,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_percent,
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde statut: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
            self.log_performance("save_status", 0, success=False, error=str(e))

    def load_data(self, file_path: str, data_type: str = "features") -> pd.DataFrame:
        """
        Charge les données CSV et valide les types scalaires.

        Args:
            file_path (str): Chemin du fichier CSV.
            data_type (str): "features" ou "trades".

        Returns:
            pd.DataFrame: Données chargées.
        """
        start_time = datetime.now()
        try:
            logger.info(f"Chargement {data_type}: {file_path}")
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(
                    f"Chargement {data_type}: {file_path}", priority=2
                )
                send_telegram_alert(f"Chargement {data_type}: {file_path}")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Fichier introuvable: {file_path}")

            def read_csv():
                return pd.read_csv(file_path)

            df = self.with_retries(read_csv)
            if df is None or df.empty:
                raise ValueError(f"Fichier vide: {file_path}")
            if len(df) < PERFORMANCE_THRESHOLDS["min_rows"]:
                raise ValueError(
                    f"Trop peu de lignes: {len(df)} < {PERFORMANCE_THRESHOLDS['min_rows']}"
                )

            if data_type == "features":
                shap_file = BASE_DIR / "data" / "features" / "feature_importance.csv"
                if not shap_file.exists():
                    raise FileNotFoundError(f"Fichier SHAP introuvable: {shap_file}")
                shap_df = pd.read_csv(shap_file)
                required_cols = shap_df["feature"].head(150).tolist() + [
                    "vix_es_correlation",
                    "predicted_vix",
                    "news_impact_score",
                    "call_iv_atm",
                    "option_skew",
                    "spoofing_score",
                    "volume_anomaly",
                    "net_gamma",
                    "call_wall",
                    "trade_velocity",
                    "hft_activity_score",
                    "bid_ask_imbalance",
                    "trade_aggressiveness",
                    "iv_skew",
                    "iv_term_structure",
                    "close",
                    "volume",
                ]
                # Validation des features SHAP dans config/feature_sets.yaml
                feature_sets_path = BASE_DIR / "config" / "feature_sets.yaml"
                if feature_sets_path.exists():
                    feature_config = get_config(feature_sets_path)
                    shap_features = feature_config.get("feature_sets", {}).get(
                        "ES", {}
                    ).get("inference", [])
                    if len(shap_features) != 150:
                        raise ValueError(
                            f"Nombre de SHAP features incorrect dans feature_sets.yaml: {len(shap_features)} != 150"
                        )
                    missing_shap = [
                        col
                        for col in shap_df["feature"].head(50)
                        if col not in shap_features
                    ]
                    if missing_shap:
                        raise ValueError(
                            f"Top 50 SHAP features non présentes dans feature_sets.yaml: {missing_shap[:5]}"
                        )
                # Validation obs_t
                missing_obs_t = self.neural_pipeline.validate_obs_t(df.columns.tolist())
                if missing_obs_t:
                    warning_msg = (
                        f"Colonnes obs_t manquantes dans {data_type}: {missing_obs_t}"
                    )
                    logger.warning(warning_msg)
                    if self.config.get("notifications_enabled", True):
                        self.alert_manager.send_alert(warning_msg, priority=3)
                        send_telegram_alert(warning_msg)
            else:
                required_cols = ["reward", "timestamp"]

            available_cols = [col for col in required_cols if col in df.columns]
            confidence_drop_rate = 1.0 - min(
                (len(available_cols) / len(required_cols)), 1.0
            )
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(available_cols)}/{len(required_cols)} colonnes)"
                logger.warning(alert_msg)
                if self.config.get("notifications_enabled", True):
                    self.alert_manager.send_alert(alert_msg, priority=3)
                    send_telegram_alert(alert_msg)
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                warning_msg = f"Colonnes manquantes: {missing_cols}"
                logger.warning(warning_msg)
                if self.config.get("notifications_enabled", True):
                    self.alert_manager.send_alert(warning_msg, priority=3)
                    send_telegram_alert(warning_msg)

            critical_cols = (
                [
                    "bid_size_level_1",
                    "ask_size_level_1",
                    "trade_frequency_1s",
                    "news_impact_score",
                    "call_iv_atm",
                    "option_skew",
                    "spoofing_score",
                    "volume_anomaly",
                    "net_gamma",
                    "call_wall",
                    "trade_velocity",
                    "hft_activity_score",
                    "bid_ask_imbalance",
                    "trade_aggressiveness",
                    "iv_skew",
                    "iv_term_structure",
                    "close",
                    "volume",
                ]
                if data_type == "features"
                else ["reward"]
            )
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

            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").fillna(
                pd.Timestamp("2025-04-14")
            )
            if data_type == "features":
                df[available_cols] = (
                    df[available_cols]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(method="ffill")
                    .fillna(0)
                )
            df = df.tail(self.config["max_rows"])

            latency = (datetime.now() - start_time).total_seconds()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            success_msg = (
                f"{data_type} chargé: {len(df)} lignes, {len(df.columns)} features"
            )
            logger.info(success_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)
            self.log_performance(
                "load_data",
                latency,
                success=True,
                num_rows=len(df),
                num_features=len(df.columns),
                confidence_drop_rate=confidence_drop_rate,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_percent,
            )
            return df
        except Exception as e:
            error_msg = (
                f"Erreur chargement {data_type}: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
            self.log_performance("load_data", 0, success=False, error=str(e))
            raise

    def load_dashboard_json(self, dashboard_json_path: str) -> Dict:
        """
        Charge neural_pipeline_dashboard.json avec cache mémoire et versionnement.

        Args:
            dashboard_json_path (str): Chemin du fichier JSON.

        Returns:
            Dict: Données JSON chargées.
        """
        start_time = datetime.now()
        try:
            if not os.path.exists(dashboard_json_path):
                warning_msg = f"Fichier {dashboard_json_path} introuvable"
                logger.warning(warning_msg)
                if self.config.get("notifications_enabled", True):
                    self.alert_manager.send_alert(warning_msg, priority=3)
                    send_telegram_alert(warning_msg)
                return {}

            file_mtime = os.path.getmtime(dashboard_json_path)
            cache_key = f"dashboard_json_{hashlib.sha256(str(dashboard_json_path).encode()).hexdigest()}"
            if (
                cache_key in self.dashboard_cache
                and self.dashboard_cache[cache_key]["mtime"] == file_mtime
            ):
                latency = (datetime.now() - start_time).total_seconds()
                memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
                cpu_percent = psutil.cpu_percent()
                success_msg = f"Données JSON {dashboard_json_path} récupérées du cache"
                logger.info(success_msg)
                if self.config.get("notifications_enabled", True):
                    self.alert_manager.send_alert(success_msg, priority=1)
                    send_telegram_alert(success_msg)
                self.log_performance(
                    "load_dashboard_json_cache_hit",
                    latency,
                    success=True,
                    memory_usage_mb=memory_usage,
                    cpu_usage_percent=cpu_percent,
                )
                return self.dashboard_cache[cache_key]["data"]

            def read_json():
                with open(dashboard_json_path, "r", encoding="utf-8") as f:
                    return json.load(f)

            dashboard_data = self.with_retries(read_json)
            if dashboard_data is None:
                raise ValueError(f"Échec chargement {dashboard_json_path}")

            self.dashboard_cache[cache_key] = {
                "data": dashboard_data,
                "mtime": file_mtime,
            }
            if len(self.dashboard_cache) > 100:  # Limite raisonnable
                self.dashboard_cache.pop(next(iter(self.dashboard_cache)))

            latency = (datetime.now() - start_time).total_seconds()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            success_msg = f"Données JSON {dashboard_json_path} chargées"
            logger.info(success_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)
            self.log_performance(
                "load_dashboard_json",
                latency,
                success=True,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_percent,
            )
            return dashboard_data
        except Exception as e:
            error_msg = f"Erreur chargement JSON {dashboard_json_path}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
            self.log_performance("load_dashboard_json", 0, success=False, error=str(e))
            return {}

    def get_regime(
        self, df: pd.DataFrame, step: int, config_path: str
    ) -> Tuple[str, str, pd.Series]:
        """
        Détecte le régime du marché pour un step avec regime_probs.

        Args:
            df (pd.DataFrame): Données.
            step (int): Step à analyser.
            config_path (str): Chemin de la config.

        Returns:
            Tuple[str, str, pd.Series]: Régime, neural_regime, regime_probs.
        """
        start_time = datetime.now()
        try:
            logger.info(f"Détection régime step {step}")
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(f"Détection régime step {step}", priority=2)
                send_telegram_alert(f"Détection régime step {step}")
            detector = MarketRegimeDetector(config_path=config_path)

            def detect():
                return detector.detect(df, step)

            regime, details = self.with_retries(detect)
            if regime is None:
                raise ValueError("Échec détection régime")

            neural_regime = str(details.get("neural_regime", "N/A"))
            regime_probs = pd.Series(
                details.get(
                    "regime_probs", {"trend": 0.33, "range": 0.33, "defensive": 0.34}
                )
            )

            latency = (datetime.now() - start_time).total_seconds()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            success_msg = f"Régime: {regime}, Neural: {neural_regime}"
            logger.info(success_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)
            self.log_performance(
                "get_regime",
                latency,
                success=True,
                step=step,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_percent,
            )
            return regime, neural_regime, regime_probs
        except Exception as e:
            error_msg = f"Erreur détection régime: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
            self.log_performance("get_regime", 0, success=False, error=str(e))
            return (
                "Erreur",
                "Erreur",
                pd.Series({"trend": 0.33, "range": 0.33, "defensive": 0.34}),
            )

    def create_regime_fig(self, df: pd.DataFrame, step: int) -> go.Figure:
        """
        Crée un graphique Plotly des régimes.

        Args:
            df (pd.DataFrame): Données.
            step (int): Step actuel.

        Returns:
            go.Figure: Figure Plotly.
        """
        start_time = datetime.now()
        try:
            logger.info(f"Graphique régimes step {step}")
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(f"Graphique régimes step {step}", priority=2)
                send_telegram_alert(f"Graphique régimes step {step}")
            fig = go.Figure()
            regimes = df.get("market_regime", pd.Series(["trend"] * len(df))).replace(
                {"trend": 0, "range": 1, "defensive": 2}
            )
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=regimes,
                    mode="lines+markers",
                    name="Régime",
                    line={"color": "#2980b9"},
                )
            )
            fig.update_yaxes(
                tickvals=[0, 1, 2],
                ticktext=["Trend", "Range", "Defensive"],
                title="Régime",
            )
            fig.update_layout(
                title="Historique des Régimes",
                xaxis_title="Temps",
                height=400,
                plot_bgcolor="#ecf0f1",
                paper_bgcolor="#ffffff",
                margin={"l": 50, "r": 50, "t": 50, "b": 50},
            )
            latency = (datetime.now() - start_time).total_seconds()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            success_msg = f"Graphique régimes généré pour step {step}"
            logger.info(success_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)
            self.log_performance(
                "create_regime_fig",
                latency,
                success=True,
                step=step,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_percent,
            )
            return fig
        except Exception as e:
            error_msg = f"Erreur graphique régimes: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
            self.log_performance("create_regime_fig", 0, success=False, error=str(e))
            return go.Figure()

    def create_feature_fig(
        self, df: pd.DataFrame, selected_features: List[str]
    ) -> go.Figure:
        """
        Crée un graphique Plotly des features.

        Args:
            df (pd.DataFrame): Données.
            selected_features (List[str]): Features à afficher.

        Returns:
            go.Figure: Figure Plotly.
        """
        start_time = datetime.now()
        try:
            logger.info(f"Graphique features: {selected_features}")
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(
                    f"Graphique features: {selected_features}", priority=2
                )
                send_telegram_alert(f"Graphique features: {selected_features}")
            fig = go.Figure()
            colors = [
                "#2980b9",
                "#e74c3c",
                "#f1c40f",
                "#2ecc71",
                "#9b59b6",
                "#3498db",
                "#e67e22",
                "#1abc9c",
            ]
            for i, feature in enumerate(selected_features):
                if feature in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df["timestamp"],
                            y=df[feature],
                            mode="lines",
                            name=feature,
                            line={"color": colors[i % len(colors)]},
                        )
                    )
                else:
                    warning_msg = f"Feature {feature} non trouvée"
                    logger.warning(warning_msg)
                    if self.config.get("notifications_enabled", True):
                        self.alert_manager.send_alert(warning_msg, priority=3)
                        send_telegram_alert(warning_msg)
            fig.update_layout(
                title=f"Données: {', '.join(selected_features)}",
                xaxis_title="Temps",
                yaxis_title="Valeur",
                height=400,
                plot_bgcolor="#ecf0f1",
                paper_bgcolor="#ffffff",
                margin={"l": 50, "r": 50, "t": 50, "b": 50},
            )
            latency = (datetime.now() - start_time).total_seconds()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            success_msg = f"Graphique features généré: {selected_features}"
            logger.info(success_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)
            self.log_performance(
                "create_feature_fig",
                latency,
                success=True,
                num_features=len(selected_features),
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_percent,
            )
            return fig
        except Exception as e:
            error_msg = f"Erreur graphique features: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
            self.log_performance("create_feature_fig", 0, success=False, error=str(e))
            return go.Figure()

    def create_equity_fig(self, trades_path: str) -> go.Figure:
        """
        Crée un graphique Plotly de la courbe d'equity.

        Args:
            trades_path (str): Chemin des trades.

        Returns:
            go.Figure: Figure Plotly.
        """
        start_time = datetime.now()
        try:
            logger.info("Graphique equity")
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert("Graphique equity", priority=2)
                send_telegram_alert("Graphique equity")
            df = self.load_data(trades_path, "trades")
            equity = np.cumsum(df["reward"].fillna(0))
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=equity,
                    mode="lines",
                    name="Equity",
                    line={"color": "#2980b9"},
                )
            )
            fig.update_layout(
                title="Courbe d'Equity",
                xaxis_title="Temps",
                yaxis_title="Equity",
                height=400,
                plot_bgcolor="#ecf0f1",
                paper_bgcolor="#ffffff",
                margin={"l": 50, "r": 50, "t": 50, "b": 50},
            )
            latency = (datetime.now() - start_time).total_seconds()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            success_msg = f"Graphique equity généré"
            logger.info(success_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)
            self.log_performance(
                "create_equity_fig",
                latency,
                success=True,
                num_rows=len(df),
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_percent,
            )
            return fig
        except Exception as e:
            error_msg = f"Erreur graphique equity: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
            self.log_performance("create_equity_fig", 0, success=False, error=str(e))
            return go.Figure()

    def get_metrics_summary(self, trades_path: str) -> str:
        """
        Calcule et retourne un résumé des métriques de trading.

        Args:
            trades_path (str): Chemin des trades.

        Returns:
            str: Résumé des métriques.
        """
        start_time = datetime.now()
        try:
            logger.info("Calcul métriques")
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert("Calcul métriques", priority=2)
                send_telegram_alert("Calcul métriques")
            df = self.load_data(trades_path, "trades")
            rewards = df["reward"].fillna(0).values
            total_trades = len(rewards)
            win_rate = (rewards > 0).mean() * 100 if total_trades > 0 else 0
            total_return = rewards.sum()
            summary = (
                f"Trades: {total_trades} | Taux de succès: {win_rate:.1f}% | "
                f"Retour total: {total_return:.2f}"
            )
            latency = (datetime.now() - start_time).total_seconds()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            success_msg = f"Métriques calculées: {summary}"
            logger.info(success_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)
            self.log_performance(
                "get_metrics_summary",
                latency,
                success=True,
                num_rows=len(df),
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_percent,
            )
            return summary
        except Exception as e:
            error_msg = f"Erreur métriques: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
            self.log_performance("get_metrics_summary", 0, success=False, error=str(e))
            return "Erreur métriques"

    def plot_regime_probs(self, regime_probs: pd.Series, output_path: str) -> None:
        """
        Génère une visualisation matplotlib des probabilités de régimes.

        Args:
            regime_probs (pd.Series): Probabilités des régimes.
            output_path (str): Chemin de sauvegarde.
        """
        start_time = datetime.now()
        try:
            logger.info(f"Génération graphique regime_probs: {output_path}")
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(
                    f"Génération graphique regime_probs: {output_path}", priority=2
                )
                send_telegram_alert(f"Génération graphique regime_probs: {output_path}")

            plt.figure(figsize=(10, 6))
            plt.bar(
                regime_probs.index,
                regime_probs.values,
                color=["#2980b9", "#e74c3c", "#f1c40f"],
            )
            plt.title("Probabilités des Régimes de Marché")
            plt.xlabel("Régime")
            plt.ylabel("Probabilité")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()

            def save_fig():
                plt.savefig(output_path)
                plt.close()

            self.with_retries(save_fig)

            latency = (datetime.now() - start_time).total_seconds()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            success_msg = f"Graphique regime_probs sauvegardé: {output_path}"
            logger.info(success_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)
            self.log_performance(
                "plot_regime_probs",
                latency,
                success=True,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_percent,
            )
        except Exception as e:
            error_msg = (
                f"Erreur graphique regime_probs: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
            self.log_performance("plot_regime_probs", 0, success=False, error=str(e))

    def plot_shap_features(self, shap_data: pd.DataFrame, output_path: str) -> None:
        """
        Génère une visualisation matplotlib des importances SHAP.

        Args:
            shap_data (pd.DataFrame): Données SHAP.
            output_path (str): Chemin de sauvegarde.
        """
        start_time = datetime.now()
        try:
            logger.info(f"Génération graphique SHAP: {output_path}")
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(
                    f"Génération graphique SHAP: {output_path}", priority=2
                )
                send_telegram_alert(f"Génération graphique SHAP: {output_path}")

            plt.figure(figsize=(12, 8))
            plt.bar(shap_data["feature"], shap_data["importance"], color="#2980b9")
            plt.title("Top 150 SHAP Features Importance")
            plt.xlabel("Feature")
            plt.ylabel("Importance")
            plt.xticks(rotation=90)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()

            def save_fig():
                plt.savefig(output_path)
                plt.close()

            self.with_retries(save_fig)

            latency = (datetime.now() - start_time).total_seconds()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            success_msg = f"Graphique SHAP sauvegardé: {output_path}"
            logger.info(success_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)
            self.log_performance(
                "plot_shap_features",
                latency,
                success=True,
                num_features=len(shap_data),
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_percent,
            )
        except Exception as e:
            error_msg = f"Erreur graphique SHAP: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            if self.config.get("notifications_enabled", True):
                self.alert_manager.send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
            self.log_performance("plot_shap_features", 0, success=False, error=str(e))

# ------------------------- MAIN ------------------------- #

def main():
    dashboard = MIADashboard()
    st.set_page_config(page_title="MIA Dashboard", layout="wide")
    st.title("MIA Dashboard: Surveillez Votre Trading")

    # Chemins des fichiers
    features_path = str(BASE_DIR / "data" / "features" / "features_latest.csv")
    trades_path = str(BASE_DIR / "data" / "trades" / "trades_simulated.csv")
    shap_path = str(BASE_DIR / "data" / "features" / "feature_importance.csv")
    config_path = "config/market_config.yaml"
    dashboard_json_path = str(DASHBOARD_JSON_PATH)

    # Mise à jour du tableau de bord
    start_time = datetime.now()
    try:
        logger.info("Mise à jour dashboard")
        if dashboard.config.get("notifications_enabled", True):
            dashboard.alert_manager.send_alert("Mise à jour dashboard", priority=2)
            send_telegram_alert("Mise à jour dashboard")

        # Calculer le hachage des données
        selected_features = st.multiselect(
            "Choisissez les données à visualiser",
            options=[
                "close",
                "atr_14",
                "predicted_vix",
                "news_impact_score",
                "call_iv_atm",
                "option_skew",
                "spoofing_score",
                "volume_anomaly",
                "net_gamma",
                "call_wall",
                "trade_velocity",
                "hft_activity_score",
                "bid_ask_imbalance",
                "trade_aggressiveness",
                "iv_skew",
                "iv_term_structure",
            ],
            default=["close", "predicted_vix"],
        )
        data_hash = hashlib.sha256(
            str([features_path, trades_path, shap_path, selected_features]).encode()
        ).hexdigest()
        cache_key = f"dashboard_{data_hash}"
        if cache_key in dashboard.cache:
            dashboard.cache.move_to_end(cache_key)
            if (
                time.time() - dashboard.cache[cache_key]["timestamp"]
            ) < CACHE_EXPIRATION:
                cached_data = dashboard.cache[cache_key]["data"]
                st.subheader("Régime du Marché")
                st.write(cached_data["regime_text"])
                st.plotly_chart(go.Figure(cached_data["regime_fig"]), use_container_width=True)
                st.subheader("Probabilités des Régimes")
                st.image(cached_data["regime_probs_img"])
                st.subheader("Données de Trading et Prédictions LSTM")
                st.plotly_chart(go.Figure(cached_data["feature_fig"]), use_container_width=True)
                st.subheader("Performance du Trading")
                st.plotly_chart(go.Figure(cached_data["equity_fig"]), use_container_width=True)
                st.write(cached_data["metrics_summary"])
                st.subheader("Importance des Features SHAP")
                st.image(cached_data["shap_img"])
                st.subheader("Métriques de Confiance")
                st.write(cached_data["confidence_text"])
                latency = (datetime.now() - start_time).total_seconds()
                memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
                cpu_percent = psutil.cpu_percent()
                success_msg = "Dashboard mis à jour depuis le cache"
                logger.info(success_msg)
                if dashboard.config.get("notifications_enabled", True):
                    dashboard.alert_manager.send_alert(success_msg, priority=1)
                    send_telegram_alert(success_msg)
                dashboard.log_performance(
                    "update_dashboard_cache_hit",
                    latency,
                    success=True,
                    memory_usage_mb=memory_usage,
                    cpu_usage_percent=cpu_percent,
                )
                return

        df_features = dashboard.load_data(features_path, "features")
        dashboard.checkpoint(df_features, data_type="features")
        step = len(df_features) - 1

        # Régimes hybrides
        st.subheader("Régime du Marché")
        regime, neural_regime, regime_probs = dashboard.get_regime(
            df_features, step, config_path
        )
        regime_text = f"Régime: {regime.upper()} | Neural: {neural_regime} | Step: {step}"
        st.write(regime_text)
        regime_fig = dashboard.create_regime_fig(df_features, step)
        st.plotly_chart(regime_fig, use_container_width=True)

        # Probabilités des régimes
        st.subheader("Probabilités des Régimes")
        regime_probs_path = (
            FIGURES_DIR / f"regime_probs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        dashboard.plot_regime_probs(regime_probs, str(regime_probs_path))
        st.image(str(regime_probs_path))

        # SHAP features
        st.subheader("Importance des Features SHAP")
        def read_shap():
            return pd.read_csv(shap_path).head(150)

        shap_data = dashboard.with_retries(read_shap)
        if shap_data is None:
            raise ValueError("Échec chargement SHAP features")
        shap_fig_path = (
            FIGURES_DIR / f"shap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        dashboard.plot_shap_features(shap_data, str(shap_fig_path))
        st.image(str(shap_fig_path))

        # LSTM predicted_vix
        def predict_vix():
            result = dashboard.neural_pipeline.run(
                df_features[["timestamp", "close", "volume"]], None, None
            )
            assert isinstance(
                result, dict
            ), "Résultat de neural_pipeline.run n’est pas un dictionnaire"
            assert all(
                k in result
                for k in ["features", "volatility", "regime", "predicted_vix"]
            ), "Clés manquantes dans neural_pipeline.run"
            assert isinstance(
                result["predicted_vix"], float
            ), "predicted_vix n’est pas un flottant"
            return result["volatility"]

        predicted_vix = dashboard.with_retries(predict_vix)
        if predicted_vix is None:
            raise ValueError("Échec prédiction VIX")
        df_features["predicted_vix"] = predicted_vix[-len(df_features) :]

        # Charger neural_pipeline_dashboard.json
        st.subheader("Métriques de Confiance")
        confidence_text = "Confidence non disponible"
        dashboard_data = dashboard.load_dashboard_json(dashboard_json_path)
        if dashboard_data:
            confidence_drop_rate = dashboard_data.get("confidence_drop_rate", 0.0)
            new_features = dashboard_data.get("new_features", [])
            confidence_text = f"Confidence Drop Rate: {confidence_drop_rate:.2f} | Nouvelles Features: {', '.join(new_features)}"
        st.write(confidence_text)

        # Features
        st.subheader("Données de Trading et Prédictions LSTM")
        feature_fig = dashboard.create_feature_fig(df_features, selected_features)
        st.plotly_chart(feature_fig, use_container_width=True)

        # Performances
        st.subheader("Performance du Trading")
        equity_fig = dashboard.create_equity_fig(trades_path)
        st.plotly_chart(equity_fig, use_container_width=True)
        metrics_summary = dashboard.get_metrics_summary(trades_path)
        st.write(metrics_summary)

        # Sauvegarder dans le cache
        cache_data = {
            "regime_text": regime_text,
            "regime_fig": regime_fig,
            "feature_fig": feature_fig,
            "equity_fig": equity_fig,
            "metrics_summary": metrics_summary,
            "regime_probs_img": str(regime_probs_path),
            "shap_img": str(shap_fig_path),
            "confidence_text": confidence_text,
        }
        dashboard.cache[cache_key] = {
            "data": cache_data,
            "timestamp": time.time(),
        }
        dashboard.cache.move_to_end(cache_key)
        if len(dashboard.cache) > 1000:  # Limite raisonnable
            dashboard.cache.popitem(last=False)

        # Sauvegarder l'état du dashboard
        status_data = {
            "timestamp": datetime.now().isoformat(),
            "regime": regime,
            "neural_regime": neural_regime,
            "selected_features": selected_features,
            "num_rows": len(df_features),
            "confidence_drop_rate": confidence_drop_rate,
            "new_features": new_features,
            "metrics_summary": metrics_summary,
        }
        dashboard.save_status(status_data)

        latency = (datetime.now() - start_time).total_seconds()
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        cpu_percent = psutil.cpu_percent()
        success_msg = "Dashboard mis à jour"
        logger.info(success_msg)
        if dashboard.config.get("notifications_enabled", True):
            dashboard.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
        dashboard.log_performance(
            "update_dashboard",
            latency,
            success=True,
            num_features=len(df_features.columns),
            confidence_drop_rate=confidence_drop_rate,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_percent,
        )
        dashboard.save_snapshot(
            "update_dashboard",
            {
                "step": step,
                "regime": regime,
                "neural_regime": neural_regime,
                "selected_features": selected_features,
                "num_rows": len(df_features),
                "confidence_drop_rate": confidence_drop_rate,
                "new_features": new_features,
            },
        )

    except Exception as e:
        error_msg = f"Erreur mise à jour dashboard: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        if dashboard.config.get("notifications_enabled", True):
            dashboard.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
        latency = (datetime.now() - start_time).total_seconds()
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        cpu_percent = psutil.cpu_percent()
        dashboard.log_performance(
            "update_dashboard",
            latency,
            success=False,
            error=str(e),
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_percent,
        )
        st.error(error_msg)

if __name__ == "__main__":
    main()
