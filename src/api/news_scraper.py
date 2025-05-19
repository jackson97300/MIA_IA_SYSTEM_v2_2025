# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/api/news_scraper.py
# Collecte des nouvelles financières via NewsAPI[](https://newsapi.org/) pour news_data.csv.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Récupère les nouvelles via NewsAPI (~100 USD/mois pour premium), calcule news_impact_score
#        (méthode 5), stocke dans data/news/cache/daily_YYYYMMDD.csv, et génère news_data.csv
#        avec validation robuste, snapshots, et sauvegardes.
#
# Dépendances :
# - requests>=2.28.0,<3.0.0, pandas>=2.0.0,<3.0.0, psutil>=5.9.8,<6.0.0, pyyaml>=6.0.0,<7.0.0,
#   boto3>=1.26.0,<2.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/credentials.yaml (clés API NewsAPI)
# - config/news_config.yaml
#
# Outputs :
# - data/news/news_data.csv
# - data/news/cache/daily_YYYYMMDD.csv
# - data/news_snapshots/*.json.gz
# - data/logs/news_scraper_performance.csv
# - data/logs/news_scraper.log
# - data/checkpoints/news_data_*.json.gz
# - data/news_dashboard.json
#
# Notes :
# - Conforme à structure.txt (version 2.1.3, 2025-05-13).
# - Utilise exclusivement NewsAPI (https://newsapi.org/, ~100 USD/mois pour premium).
# - Implémente retries (max 3, délai 2^attempt), logs psutil, alertes Telegram, snapshots compressés.
# - Intègre récompenses adaptatives (méthode 5) via news_impact_score basé sur mots-clés financiers.
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des données.
# - Intègre validation SHAP (Phase 17) pour les features utilisées (ex. : news_impact_score, source).
# - Phases intégrées : Phase 1 (collecte via API), Phase 8 (auto-conscience), Phase 17 (interprétabilité SHAP).
# - Sauvegardes incrémentielles toutes les 5 min dans data/checkpoints/ avec versionnage (5 versions).
# - Sauvegardes distribuées toutes les 15 min vers S3 (configurable).
# - Tests unitaires dans tests/test_news_scraper.py.
# - Validation complète prévue pour juin 2025.
# - Évolution future : Intégration de sources alternatives (juin 2025).

import gzip
import json
import logging
import os
import signal
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import boto3
import pandas as pd
import psutil
import requests

from src.model.utils.alert_manager import send_alert
from src.model.utils.config_manager import get_config
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.utils.telegram_alert import send_telegram_alert

# Configuration du logging
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
os.makedirs(os.path.join(BASE_DIR, "data", "logs"), exist_ok=True)
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "data", "logs", "news_scraper.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Chemins et constantes
CONFIG_PATH = os.path.join(BASE_DIR, "config", "news_config.yaml")
CREDENTIALS_PATH = os.path.join(BASE_DIR, "config", "credentials.yaml")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "news", "news_data.csv")
CACHE_DIR = os.path.join(BASE_DIR, "data", "news", "cache")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "news_snapshots")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "data", "checkpoints")
PERFORMANCE_LOG = os.path.join(BASE_DIR, "data", "logs", "news_scraper_performance.csv")
DASHBOARD_PATH = os.path.join(BASE_DIR, "data", "news_dashboard.json")
FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR, "data", "features", "feature_importance.csv"
)

# Variable pour gérer l'arrêt propre
RUNNING = True


class NewsScraper:
    """Classe pour gérer la collecte des nouvelles via NewsAPI."""

    def __init__(
        self, config_path: str = CONFIG_PATH, credentials_path: str = CREDENTIALS_PATH
    ):
        """
        Initialise le scraper de nouvelles.

        Args:
            config_path (str): Chemin vers le fichier de configuration.
            credentials_path (str): Chemin vers le fichier des identifiants.
        """
        self.log_buffer = []
        self.buffer_size = 100
        self.checkpoint_versions = []
        try:
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            os.makedirs(CACHE_DIR, exist_ok=True)
            self.config = self.load_config_with_validation(config_path)
            self.credentials = self.load_config_with_validation(credentials_path)
            self.api_key = self.credentials.get("news_api_key", "")
            if not self.api_key:
                raise ValueError("Clé API NewsAPI manquante dans credentials.yaml")
            signal.signal(signal.SIGINT, self.signal_handler)
            success_msg = "NewsScraper initialisé"
            miya_speak(success_msg, tag="NEWS", voice_profile="calm", priority=2)
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("init", 0, success=True)
            self.save_snapshot(
                "init", {"config": self.config, "api_key": "masked"}, compress=True
            )
        except Exception as e:
            error_msg = (
                f"Erreur initialisation NewsScraper: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="NEWS", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            self.config = {
                "endpoint": "https://newsapi.org/v2/everything",
                "timeout": 10,
                "retry_attempts": 3,
                "retry_delay_base": 2.0,
                "cache_days": 7,
                "s3_bucket": None,
                "s3_prefix": "news_data/",
            }
            self.credentials = {"news_api_key": ""}

    def load_config_with_validation(self, config_path: str) -> Dict:
        """
        Charge et valide la configuration.

        Args:
            config_path (str): Chemin vers le fichier de configuration.

        Returns:
            Dict: Configuration validée.
        """
        start_time = datetime.now()
        try:
            config = get_config(os.path.basename(config_path))
            section = "news_scraper" if "news_config" in config_path else "credentials"
            if section not in config:
                config[section] = {}
            required_keys = (
                [
                    "endpoint",
                    "timeout",
                    "retry_attempts",
                    "retry_delay_base",
                    "cache_days",
                    "s3_bucket",
                    "s3_prefix",
                ]
                if section == "news_scraper"
                else ["news_api_key"]
            )
            missing_keys = [key for key in required_keys if key not in config[section]]
            if missing_keys:
                raise ValueError(
                    f"Clés de configuration manquantes dans {section}: {missing_keys}"
                )
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Configuration {section} chargée"
            miya_speak(success_msg, tag="NEWS", voice_profile="calm", priority=1)
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "load_config_with_validation",
                latency,
                success=True,
                config_section=section,
            )
            return config[section]
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur chargement configuration {config_path}: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="NEWS", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "load_config_with_validation", latency, success=False, error=str(e)
            )
            return (
                {
                    "endpoint": "https://newsapi.org/v2/everything",
                    "timeout": 10,
                    "retry_attempts": 3,
                    "retry_delay_base": 2.0,
                    "cache_days": 7,
                    "s3_bucket": None,
                    "s3_prefix": "news_data/",
                }
                if "news_config" in config_path
                else {"news_api_key": ""}
            )

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ) -> None:
        """
        Journalise les performances des opérations critiques.

        Args:
            operation (str): Nom de l’opération (ex. : scrape_news).
            latency (float): Latence en secondes.
            success (bool): Succès de l’opération.
            error (str, optional): Message d’erreur si échec.
            **kwargs: Paramètres supplémentaires (ex. : num_news, confidence_drop_rate).
        """
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_usage = psutil.cpu_percent()
            if memory_usage > 1024:
                alert_msg = (
                    f"ALERTE: Utilisation mémoire élevée ({memory_usage:.2f} MB)"
                )
                miya_alerts(alert_msg, tag="NEWS", voice_profile="urgent", priority=5)
                send_alert(alert_msg, priority=4)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "latency": latency,
                "success": success,
                "error": error,
                "memory_usage_mb": memory_usage,
                "cpu_usage_percent": cpu_usage,
                **kwargs,
            }
            self.log_buffer.append(log_entry)
            if len(self.log_buffer) >= self.buffer_size:
                log_df = pd.DataFrame(self.log_buffer)
                os.makedirs(os.path.dirname(PERFORMANCE_LOG), exist_ok=True)

                def write_log():
                    log_df.to_csv(
                        PERFORMANCE_LOG,
                        mode="a" if os.path.exists(PERFORMANCE_LOG) else "w",
                        header=not os.path.exists(PERFORMANCE_LOG),
                        index=False,
                        encoding="utf-8",
                    )

                self.with_retries(write_log)
                self.log_buffer = []
            logger.info(f"Performance journalisée pour {operation}. CPU: {cpu_usage}%")
        except Exception as e:
            error_msg = (
                f"Erreur journalisation performance: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="NEWS", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def save_snapshot(
        self, snapshot_type: str, data: Dict, compress: bool = True
    ) -> None:
        """
        Sauvegarde un instantané des résultats.

        Args:
            snapshot_type (str): Type de snapshot (ex. : scrape_news).
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
            snapshot_path = os.path.join(
                SNAPSHOT_DIR, f"snapshot_{snapshot_type}_{timestamp}.json"
            )
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)

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
                miya_alerts(alert_msg, tag="NEWS", voice_profile="urgent", priority=3)
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Snapshot {snapshot_type} sauvegardé: {save_path}"
            miya_speak(success_msg, tag="NEWS", voice_profile="calm", priority=1)
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "save_snapshot",
                latency,
                success=True,
                snapshot_type=snapshot_type,
                file_size_mb=file_size,
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}\n{traceback.format_exc()}"
            self.log_performance("save_snapshot", 0, success=False, error=str(e))
            miya_alerts(error_msg, tag="NEWS", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def checkpoint(self, news: pd.DataFrame) -> None:
        """
        Sauvegarde incrémentielle des nouvelles toutes les 5 minutes avec versionnage (5 versions).

        Args:
            news (pd.DataFrame): Nouvelles à sauvegarder.
        """
        try:
            start_time = datetime.now()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_data = {
                "timestamp": timestamp,
                "num_news": len(news),
                "headlines": news["headline"].tolist()[:10],
            }
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR, f"news_data_{timestamp}.json.gz"
            )
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)

            def write_checkpoint():
                with gzip.open(checkpoint_path, "wt", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, indent=4)
                news.to_csv(
                    checkpoint_path.replace(".json.gz", ".csv"),
                    index=False,
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
            miya_speak(success_msg, tag="NEWS", voice_profile="calm", priority=1)
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "checkpoint",
                latency,
                success=True,
                file_size_mb=file_size,
                num_news=len(news),
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde checkpoint: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance("checkpoint", 0, success=False, error=str(e))
            miya_alerts(error_msg, tag="NEWS", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def cloud_backup(self, news: pd.DataFrame) -> None:
        """
        Sauvegarde distribuée des nouvelles vers AWS S3 toutes les 15 minutes (configurable).

        Args:
            news (pd.DataFrame): Nouvelles à sauvegarder.
        """
        try:
            start_time = datetime.now()
            if not self.config.get("s3_bucket"):
                warning_msg = "S3 bucket non configuré, sauvegarde cloud ignorée"
                miya_speak(warning_msg, tag="NEWS", voice_profile="warning", priority=3)
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.config['s3_prefix']}news_data_{timestamp}.csv.gz"
            temp_path = os.path.join(CHECKPOINT_DIR, f"temp_s3_{timestamp}.csv.gz")

            def write_temp():
                news.to_csv(
                    temp_path, compression="gzip", index=False, encoding="utf-8"
                )

            self.with_retries(write_temp)
            s3_client = boto3.client("s3")

            def upload_s3():
                s3_client.upload_file(temp_path, self.config["s3_bucket"], backup_path)

            self.with_retries(upload_s3)
            os.remove(temp_path)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Sauvegarde cloud S3 effectuée: {backup_path}"
            miya_speak(success_msg, tag="NEWS", voice_profile="calm", priority=1)
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "cloud_backup", latency, success=True, num_news=len(news)
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde cloud S3: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance("cloud_backup", 0, success=False, error=str(e))
            miya_alerts(error_msg, tag="NEWS", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def validate_shap_features(self, features: List[str]) -> bool:
        """
        Valide que les features sont dans les top 150 SHAP (Phase 17).

        Args:
            features (List[str]): Liste des features à valider.

        Returns:
            bool: True si toutes les features sont valides, False sinon.
        """
        try:
            start_time = datetime.now()
            if not os.path.exists(FEATURE_IMPORTANCE_PATH):
                error_msg = "Fichier SHAP manquant"
                miya_alerts(error_msg, tag="NEWS", voice_profile="urgent", priority=4)
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return False
            shap_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
            if len(shap_df) < 150:
                error_msg = f"Nombre insuffisant de SHAP features: {len(shap_df)} < 150"
                miya_alerts(error_msg, tag="NEWS", voice_profile="urgent", priority=4)
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return False
            valid_features = set(shap_df["feature"].head(150))
            missing = [f for f in features if f not in valid_features]
            if missing:
                warning_msg = f"Features non incluses dans top 150 SHAP: {missing}"
                miya_alerts(
                    warning_msg, tag="NEWS", voice_profile="warning", priority=3
                )
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "SHAP features validées"
            miya_speak(success_msg, tag="NEWS", voice_profile="calm", priority=1)
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "validate_shap_features",
                latency,
                success=True,
                num_features=len(features),
            )
            self.save_snapshot(
                "validate_shap_features",
                {"num_features": len(features), "missing": missing},
                compress=True,
            )
            return len(missing) == 0
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur validation SHAP features: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="NEWS", voice_profile="urgent", priority=4)
            send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "validate_shap_features", latency, success=False, error=str(e)
            )
            return False

    def with_retries(
        self, func: callable, max_attempts: int = 3, delay_base: float = 2.0
    ) -> Any:
        """
        Exécute une fonction avec retries (max 3, délai exponentiel).

        Args:
            func (callable): Fonction à exécuter.
            max_attempts (int): Nombre maximum de tentatives (défaut : 3).
            delay_base (float): Base du délai exponentiel (défaut : 2.0).

        Returns:
            Any: Résultat de la fonction.

        Raises:
            Exception: Si toutes les tentatives échouent.
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
                    self.log_performance(
                        f"retry_attempt_{attempt+1}",
                        0,
                        success=False,
                        error=str(e),
                        attempt_number=attempt + 1,
                    )
                    miya_alerts(
                        error_msg, tag="NEWS", voice_profile="urgent", priority=4
                    )
                    send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    logger.error(error_msg)
                    raise
                delay = delay_base * (2**attempt)
                warning_msg = f"Tentative {attempt+1} échouée, retry après {delay}s"
                miya_speak(warning_msg, tag="NEWS", voice_profile="warning", priority=3)
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
                time.sleep(delay)

    def load_daily_cache(self, date: str) -> Optional[pd.DataFrame]:
        """
        Charge les données du cache quotidien pour une date donnée.

        Args:
            date (str): Date au format YYYYMMDD.

        Returns:
            Optional[pd.DataFrame]: Données en cache ou None.
        """
        try:
            start_time = datetime.now()
            cache_path = os.path.join(CACHE_DIR, f"daily_{date}.csv")
            if os.path.exists(cache_path):
                df = pd.read_csv(cache_path, encoding="utf-8")
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                    min_date = datetime.now() - timedelta(
                        days=self.config.get("cache_days", 7)
                    )
                    df = df[df["timestamp"] >= min_date]
                    if not df.empty:
                        success_msg = (
                            f"Cache quotidien chargé pour {date}: {len(df)} nouvelles"
                        )
                        miya_speak(
                            success_msg, tag="NEWS", voice_profile="calm", priority=1
                        )
                        send_alert(success_msg, priority=1)
                        send_telegram_alert(success_msg)
                        logger.info(success_msg)
                        latency = (datetime.now() - start_time).total_seconds()
                        self.log_performance(
                            "load_daily_cache", latency, success=True, num_news=len(df)
                        )
                        return df
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("load_daily_cache", latency, success=True)
            return None
        except Exception as e:
            error_msg = (
                f"Erreur chargement cache quotidien: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance("load_daily_cache", 0, success=False, error=str(e))
            miya_alerts(error_msg, tag="NEWS", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            return None

    def save_daily_cache(self, news: pd.DataFrame, date: str) -> None:
        """
        Sauvegarde les nouvelles dans le cache quotidien.

        Args:
            news (pd.DataFrame): Nouvelles à sauvegarder.
            date (str): Date au format YYYYMMDD.
        """
        try:
            start_time = datetime.now()
            cache_path = os.path.join(CACHE_DIR, f"daily_{date}.csv")
            os.makedirs(CACHE_DIR, exist_ok=True)

            def write_cache():
                news.to_csv(cache_path, index=False, encoding="utf-8")

            self.with_retries(write_cache)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = (
                f"Cache quotidien sauvegardé pour {date}: {len(news)} nouvelles"
            )
            miya_speak(success_msg, tag="NEWS", voice_profile="calm", priority=1)
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "save_daily_cache", latency, success=True, num_news=len(news)
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde cache quotidien: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance("save_daily_cache", 0, success=False, error=str(e))
            miya_alerts(error_msg, tag="NEWS", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def calculate_impact(self, title: str) -> float:
        """
        Calcule news_impact_score basé sur des mots-clés financiers (méthode 5).

        Args:
            title (str): Titre de la nouvelle.

        Returns:
            float: Score d’impact entre 0 et 1.
        """
        try:
            start_time = datetime.now()
            high_impact_keywords = {
                "fomc": 0.9,
                "cpi": 0.85,
                "nfp": 0.85,
                "gdp": 0.8,
                "interest rate": 0.9,
                "fed": 0.9,
                "ecb": 0.8,
                "inflation": 0.85,
                "unemployment": 0.8,
                "retail sales": 0.75,
            }
            medium_impact_keywords = {
                "earnings": 0.6,
                "merger": 0.55,
                "acquisition": 0.55,
                "trade": 0.5,
                "tariff": 0.5,
                "oil": 0.5,
                "stocks": 0.45,
            }
            title_lower = title.lower()
            score = 0.0
            for keyword, weight in high_impact_keywords.items():
                if keyword in title_lower:
                    score = max(score, weight)
            for keyword, weight in medium_impact_keywords.items():
                if keyword in title_lower:
                    score = max(score, weight)
            final_score = (
                score if score > 0 else 0.3
            )  # Score par défaut pour impact faible
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "calculate_impact", latency, success=True, impact_score=final_score
            )
            return final_score
        except Exception as e:
            error_msg = (
                f"Erreur calcul news_impact_score: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance("calculate_impact", 0, success=False, error=str(e))
            miya_alerts(error_msg, tag="NEWS", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            return 0.0

    def update_dashboard(self, news: pd.DataFrame) -> None:
        """
        Met à jour un fichier JSON pour partager les nouvelles avec mia_dashboard.py.

        Args:
            news (pd.DataFrame): Données des nouvelles.
        """
        try:
            start_time = datetime.now()
            if (
                news.empty
                or "timestamp" not in news.columns
                or "headline" not in news.columns
            ):
                error_msg = "Données invalides pour mise à jour dashboard"
                miya_alerts(error_msg, tag="NEWS", voice_profile="urgent", priority=4)
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return
            now = datetime.now()
            recent = news[news["timestamp"] >= now - timedelta(hours=1)]
            high_impact = recent[recent["news_impact_score"] >= 0.8]
            dashboard_data = {
                "last_update": now.strftime("%Y-%m-%d %H:%M:%S"),
                "recent_news": [
                    {
                        "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                        "headline": row["headline"],
                        "source": row["source"],
                        "news_impact_score": row["news_impact_score"],
                    }
                    for _, row in high_impact.head(5).iterrows()
                ],
                "news_count": len(recent),
            }
            os.makedirs(os.path.dirname(DASHBOARD_PATH), exist_ok=True)

            def write_dashboard():
                with open(DASHBOARD_PATH, "w", encoding="utf-8") as f:
                    json.dump(dashboard_data, f, indent=4)

            self.with_retries(write_dashboard)
            success_msg = (
                f"Dashboard mis à jour: {len(high_impact)} nouvelles à fort impact"
            )
            miya_speak(success_msg, tag="NEWS", voice_profile="calm", priority=2)
            send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            if not high_impact.empty:
                alert_msg = f"Nouvelle à fort impact: {high_impact.iloc[0]['headline']}"
                miya_speak(alert_msg, tag="NEWS", voice_profile="urgent", priority=3)
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "update_dashboard", latency, success=True, num_news=len(recent)
            )
        except Exception as e:
            error_msg = (
                f"Erreur mise à jour dashboard: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance("update_dashboard", 0, success=False, error=str(e))
            miya_alerts(error_msg, tag="NEWS", voice_profile="urgent", priority=4)
            send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def scrape_news(self) -> Optional[pd.DataFrame]:
        """
        Collecte les nouvelles via NewsAPI avec news_impact_score (méthode 5).

        Returns:
            Optional[pd.DataFrame]: Données des nouvelles ou None si échec.
        """
        try:
            start_time = datetime.now()
            required_features = ["news_impact_score", "source"]
            if not self.validate_shap_features(required_features):
                warning_msg = "Certaines features ne sont pas dans les top 150 SHAP"
                miya_speak(warning_msg, tag="NEWS", voice_profile="warning", priority=3)
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)

            # Vérifier cache quotidien
            today = datetime.now().strftime("%Y%m%d")
            cached_df = self.load_daily_cache(today)
            if cached_df is not None and not cached_df.empty:
                return cached_df

            def fetch_news():
                params = {
                    "apiKey": self.api_key,
                    "q": "ES VIX economic OR finance OR macro OR FOMC OR NFP OR CPI",
                    "from": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
                    "to": datetime.now().strftime("%Y-%m-%d"),
                    "language": "en",
                    "sortBy": "publishedAt",
                }
                response = requests.get(
                    self.config["endpoint"],
                    params=params,
                    timeout=self.config["timeout"],
                )
                response.raise_for_status()
                data = response.json()
                if data.get("status") != "ok":
                    raise ValueError("Erreur réponse NewsAPI")
                return data

            data = self.with_retries(fetch_news)
            if data is None:
                error_msg = "Aucune donnée récupérée via NewsAPI après retries"
                miya_alerts(error_msg, tag="NEWS", voice_profile="urgent", priority=4)
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return None

            articles = data.get("articles", [])
            news = []
            for article in articles:
                title = article.get("title", "")
                if not title or title.strip() == "":
                    continue
                news.append(
                    {
                        "timestamp": pd.to_datetime(article["publishedAt"]),
                        "headline": title,
                        "source": article["source"].get("name", "Unknown"),
                        "description": article.get("description", ""),
                        "news_impact_score": self.calculate_impact(title),
                    }
                )

            df = pd.DataFrame(news)
            if df.empty:
                error_msg = "Aucune nouvelle valide récupérée via NewsAPI"
                miya_alerts(error_msg, tag="NEWS", voice_profile="urgent", priority=4)
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return None

            # Validation des colonnes
            required_cols = [
                "timestamp",
                "headline",
                "source",
                "description",
                "news_impact_score",
            ]
            missing_cols = [col for col in required_cols if col not in df.columns]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(df.columns)} colonnes)"
                miya_alerts(alert_msg, tag="NEWS", voice_profile="urgent", priority=3)
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
            df = df[df["headline"].str.strip() != ""]
            df = df.drop_duplicates(subset=["timestamp", "headline"])
            df = df.fillna("")

            # Sauvegarder dans cache et output
            self.save_daily_cache(df, today)
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

            def save_output():
                df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

            self.with_retries(save_output)

            # Sauvegardes
            self.checkpoint(df)
            self.cloud_backup(df)

            # Mise à jour du dashboard
            self.update_dashboard(df)

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Nouvelles NewsAPI récupérées: {len(df)} nouvelles, Confidence drop rate: {confidence_drop_rate:.2f}"
            miya_speak(success_msg, tag="NEWS", voice_profile="calm", priority=1)
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance(
                "scrape_news",
                latency,
                success=True,
                num_news=len(df),
                confidence_drop_rate=confidence_drop_rate,
            )
            self.save_snapshot(
                "scrape_news",
                {"num_news": len(df), "confidence_drop_rate": confidence_drop_rate},
                compress=True,
            )
            return df
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur collecte nouvelles: {str(e)}\n{traceback.format_exc()}"
            self.log_performance("scrape_news", 0, success=False, error=str(e))
            miya_alerts(error_msg, tag="NEWS", voice_profile="urgent", priority=4)
            send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.save_snapshot("error_scrape_news", {"error": str(e)}, compress=True)
            return None

    def signal_handler(self, sig, frame) -> None:
        """
        Gère l'arrêt propre du service (Ctrl+C).

        Args:
            sig: Signal reçu.
            frame: Frame actuel.
        """
        global RUNNING
        start_time = datetime.now()
        try:
            RUNNING = False
            success_msg = "Arrêt du scraper de nouvelles en cours..."
            miya_speak(success_msg, tag="NEWS", voice_profile="calm", priority=2)
            send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            logger.info("Arrêt du scraper initié")

            # Sauvegarder le cache actuel si disponible
            if os.path.exists(OUTPUT_PATH):
                df = pd.read_csv(OUTPUT_PATH, encoding="utf-8")
                if not df.empty:
                    self.checkpoint(df)
                    self.save_snapshot(
                        "shutdown",
                        {
                            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                            "num_news": len(df),
                        },
                        compress=True,
                    )

            latency = (datetime.now() - start_time).total_seconds()
            success_msg = "Scraper de nouvelles arrêté proprement"
            miya_speak(success_msg, tag="NEWS", voice_profile="calm", priority=2)
            send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            self.log_performance("signal_handler", latency, success=True)
            exit(0)
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur arrêt scraper: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="NEWS", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("signal_handler", latency, success=False, error=str(e))
            exit(1)


if __name__ == "__main__":
    try:
        scraper = NewsScraper()
        data = scraper.scrape_news()
        if data is not None:
            print(f"Nouvelles récupérées: {len(data)}")
            success_msg = "Test scrape_news terminé"
            miya_speak(success_msg, tag="NEWS", voice_profile="calm", priority=1)
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
    except Exception as e:
        error_msg = f"Erreur principale: {str(e)}\n{traceback.format_exc()}"
        miya_alerts(error_msg, tag="NEWS", voice_profile="urgent", priority=5)
        send_alert(error_msg, priority=5)
        send_telegram_alert(error_msg)
        logger.error(error_msg)
