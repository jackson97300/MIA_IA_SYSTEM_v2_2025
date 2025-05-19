# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/scripts/run_pipeline.py
# Script pour orchestrer l'exécution des modules de collecte, scraping et fusion pour MIA_IA_SYSTEM_v2_2025.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Exécute de manière asynchrone les modules de collecte IBKR, scraping de news et fusion des données,
#        avec journalisation, snapshots compressés, et alertes.
#        Conforme à la Phase 6 (collecte et fusion des données), Phase 8 (auto-conscience via alertes),
#        et Phase 16 (ensemble learning).
#
# Dépendances : asyncio, yaml>=6.0.0, os, logging, signal, json, time, psutil>=5.9.8, pandas>=2.0.0,
#               matplotlib>=3.7.0, datetime, typing, gzip, traceback,
#               src.api.ibkr_fetch, src.news.news_scraper, src.api.merge_data_sources,
#               src.model.utils.config_manager, src.model.utils.alert_manager,
#               src.model.utils.miya_console, src.utils.telegram_alert, src.utils.standard
#
# Inputs : config/es_config.yaml
#
# Outputs : data/logs/run_pipeline.log, data/logs/run_pipeline_performance.csv,
#           data/pipeline_dashboard.json, data/pipeline_snapshots/snapshot_*.json.gz,
#           data/figures/pipeline/pipeline_status_*.png, data/figures/pipeline/pipeline_errors_*.png,
#           data/ibkr/ibkr_data.csv, data/news/news_data.csv, data/features/merged_data.csv
#
# Notes :
# - Utilise IQFeed exclusivement via les données IBKR (indirectement).
# - Implémente retries (max 3, délai 2^attempt), logs psutil, snapshots JSON compressés, alertes centralisées.
# - Tests unitaires dans tests/test_run_pipeline.py.

import asyncio
import gzip
import json
import logging
import os
import signal
import traceback
from datetime import datetime
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import psutil

from src.api.ibkr_fetch import main as ibkr_fetch_main
from src.api.merge_data_sources import merge_data_sources
from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import config_manager
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.news.news_scraper import scrape_news_events
from src.utils.standard import with_retries
from src.utils.telegram_alert import send_telegram_alert

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "es_config.yaml")
DASHBOARD_PATH = os.path.join(BASE_DIR, "data", "pipeline_dashboard.json")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "pipeline_snapshots")
CSV_LOG_PATH = os.path.join(BASE_DIR, "data", "logs", "run_pipeline_performance.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures", "pipeline")

# Configuration du logging
os.makedirs(os.path.join(BASE_DIR, "data", "logs"), exist_ok=True)
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "data", "logs", "run_pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [Pipeline] %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Configuration par défaut
DEFAULT_CONFIG = {
    "pipeline": {
        "ibkr_enabled": True,
        "news_enabled": True,
        "merge_enabled": True,
        "retry_attempts": 3,
        "retry_delay": 5,
        "timeout_seconds": 3600,
        "ibkr_duration": 60,
        "news_config": {
            "url": "https://www.forexfactory.com/calendar",
            "output_path": os.path.join(BASE_DIR, "data", "news", "news_data.csv"),
            "timeout": 10,
            "retry_attempts": 3,
            "retry_delay": 5,
            "cache_days": 7,
        },
        "merge_config": {
            "ibkr": os.path.join(BASE_DIR, "data", "ibkr", "ibkr_data.csv"),
            "news": os.path.join(BASE_DIR, "data", "news", "news_data.csv"),
            "merged": os.path.join(BASE_DIR, "data", "features", "merged_data.csv"),
            "chunk_size": 10000,
            "cache_hours": 24,
            "time_tolerance": "1min",
        },
        "buffer_size": 100,
        "max_cache_size": 1000,
    }
}

# Variable pour gérer l'arrêt propre
RUNNING = True


class PipelineRunner:
    """
    Classe pour orchestrer l'exécution du pipeline de collecte, scraping et fusion.
    """

    def __init__(self):
        """
        Initialise le gestionnaire du pipeline.
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
                "PipelineRunner initialisé",
                tag="PIPELINE",
                voice_profile="calm",
                priority=2,
            )
            AlertManager().send_alert("PipelineRunner initialisé", priority=1)
            logger.info("PipelineRunner initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config_path": CONFIG_PATH})
        except Exception as e:
            error_msg = f"Erreur initialisation PipelineRunner: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="PIPELINE", voice_profile="urgent", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            self.config = DEFAULT_CONFIG["pipeline"]
            self.buffer_size = 100
            self.max_cache_size = 1000

    @with_retries(max_attempts=3, delay_base=2.0)
    def load_config(self, config_path: str = CONFIG_PATH) -> Dict:
        """
        Charge les paramètres du pipeline depuis es_config.yaml.

        Args:
            config_path (str): Chemin du fichier de configuration.

        Returns:
            Dict: Configuration du pipeline.
        """
        start_time = datetime.now()
        try:
            config = config_manager.get_config("es_config.yaml")
            if "pipeline" not in config:
                raise ValueError("Clé 'pipeline' manquante dans la configuration")
            required_keys = [
                "ibkr_enabled",
                "news_enabled",
                "merge_enabled",
                "retry_attempts",
                "retry_delay",
            ]
            missing_keys = [
                key for key in required_keys if key not in config["pipeline"]
            ]
            if missing_keys:
                raise ValueError(f"Clés manquantes dans 'pipeline': {missing_keys}")
            pipeline_config = config["pipeline"]
            pipeline_config["news_config"] = pipeline_config.get(
                "news_config", DEFAULT_CONFIG["pipeline"]["news_config"]
            )
            pipeline_config["merge_config"] = pipeline_config.get(
                "merge_config", DEFAULT_CONFIG["pipeline"]["merge_config"]
            )
            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                "Configuration pipeline chargée",
                tag="PIPELINE",
                level="info",
                priority=2,
            )
            AlertManager().send_alert("Configuration pipeline chargée", priority=1)
            logger.info("Configuration pipeline chargée")
            self.log_performance("load_config", latency, success=True)
            self.save_snapshot("load_config", {"config_path": config_path})
            return pipeline_config
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur chargement config: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="PIPELINE", voice_profile="urgent", priority=4)
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
            operation (str): Nom de l’opération (ex. : run_pipeline).
            latency (float): Latence en secondes.
            success (bool): Succès de l’opération.
            error (str, optional): Message d’erreur si échec.
            **kwargs: Paramètres supplémentaires (ex. : num_events).
        """
        try:
            memory_usage = (
                psutil.Process().memory_info().rss / 1024 / 1024
            )  # Mémoire en Mo
            cpu_percent = psutil.cpu_percent()
            if memory_usage > 1024:
                error_msg = f"ALERT: Utilisation mémoire élevée ({memory_usage:.2f} MB)"
                miya_alerts(error_msg, tag="PIPELINE", level="error", priority=5)
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
            miya_alerts(error_msg, tag="PIPELINE", level="error", priority=3)
            AlertManager().send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def save_snapshot(self, snapshot_type: str, data: Dict) -> None:
        """
        Sauvegarde un instantané des résultats avec compression gzip.

        Args:
            snapshot_type (str): Type de snapshot (ex. : run_pipeline).
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
                tag="PIPELINE",
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
            miya_alerts(error_msg, tag="PIPELINE", level="error", priority=3)
            AlertManager().send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("save_snapshot", 0, success=False, error=str(e))

    def save_dashboard_status(
        self, status: Dict, status_file: str = DASHBOARD_PATH
    ) -> None:
        """
        Met à jour un fichier JSON pour partager le statut avec mia_dashboard.py.

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
                tag="PIPELINE",
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
            miya_alerts(error_msg, tag="PIPELINE", voice_profile="urgent", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "save_dashboard_status", 0, success=False, error=str(e)
            )

    def plot_pipeline_results(
        self, status: Dict, output_dir: str = FIGURES_DIR
    ) -> None:
        """
        Génère des graphiques pour les résultats du pipeline.

        Args:
            status (Dict): Statut du pipeline.
            output_dir (str): Répertoire pour sauvegarder les graphiques.
        """
        start_time = datetime.now()
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(output_dir, exist_ok=True)

            # Graphique en anneau pour les statuts des modules
            labels = ["IBKR", "News", "Merge"]
            successes = [
                1 if status["ibkr_success"] else 0,
                1 if status["news_success"] else 0,
                1 if status["merge_success"] else 0,
            ]
            colors = ["green" if s else "red" for s in successes]
            plt.figure(figsize=(8, 8))
            plt.pie(
                successes,
                labels=labels,
                colors=colors,
                autopct="%1.1f%%",
                startangle=140,
            )
            plt.title("Statut des Modules du Pipeline")
            centre_circle = plt.Circle((0, 0), 0.70, fc="white")
            fig = plt.gcf()
            fig.gca().add_artist(centre_circle)
            plt.savefig(os.path.join(output_dir, f"pipeline_status_{timestamp}.png"))
            plt.close()

            # Graphique des erreurs par tentative
            if status["errors"]:
                errors_count = len(status["errors"])
                plt.figure(figsize=(10, 6))
                plt.bar(range(1, errors_count + 1), [1] * errors_count, color="red")
                plt.title("Erreurs par Tentative")
                plt.xlabel("Tentative")
                plt.ylabel("Nombre d’Erreurs")
                plt.grid(True)
                plt.savefig(
                    os.path.join(output_dir, f"pipeline_errors_{timestamp}.png")
                )
                plt.close()

            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                f"Graphiques du pipeline générés: {output_dir}",
                tag="PIPELINE",
                level="info",
                priority=2,
            )
            AlertManager().send_alert(
                f"Graphiques du pipeline générés: {output_dir}", priority=1
            )
            logger.info(f"Graphiques du pipeline générés: {output_dir}")
            self.log_performance("plot_pipeline_results", latency, success=True)
            self.save_snapshot("plot_pipeline_results", {"output_dir": output_dir})
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur génération graphiques: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="PIPELINE", level="error", priority=3)
            AlertManager().send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "plot_pipeline_results", latency, success=False, error=str(e)
            )
            self.save_snapshot("plot_pipeline_results", {"error": str(e)})

    def validate_outputs(self, config: Dict) -> None:
        """
        Valide les fichiers générés par les modules du pipeline.

        Args:
            config (Dict): Configuration du pipeline.

        Raises:
            ValueError: Si les fichiers générés sont absents ou invalides.
        """
        start_time = datetime.now()
        try:
            if config.get("ibkr_enabled", True):
                ibkr_path = config["merge_config"]["ibkr"]
                if not os.path.exists(ibkr_path):
                    error_msg = f"Fichier IBKR introuvable: {ibkr_path}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

            if config.get("news_enabled", True):
                news_path = config["merge_config"]["news"]
                if not os.path.exists(news_path):
                    error_msg = f"Fichier news introuvable: {news_path}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

            if config.get("merge_enabled", True):
                merged_path = config["merge_config"]["merged"]
                if not os.path.exists(merged_path):
                    error_msg = f"Fichier fusionné introuvable: {merged_path}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                "Fichiers générés validés", tag="PIPELINE", level="info", priority=2
            )
            AlertManager().send_alert("Fichiers générés validés", priority=1)
            logger.info("Fichiers générés validés")
            self.log_performance("validate_outputs", latency, success=True)
            self.save_snapshot(
                "validate_outputs",
                {
                    "ibkr_path": config["merge_config"]["ibkr"],
                    "news_path": config["merge_config"]["news"],
                    "merged_path": config["merge_config"]["merged"],
                },
            )
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur validation fichiers générés: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="PIPELINE", voice_profile="urgent", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "validate_outputs", latency, success=False, error=str(e)
            )
            self.save_snapshot("validate_outputs", {"error": str(e)})
            raise

    @with_retries(max_attempts=3, delay_base=2.0)
    async def run_ibkr_fetch(self, config: Dict) -> bool:
        """
        Exécute la collecte de données IBKR.

        Args:
            config (Dict): Configuration du pipeline.

        Returns:
            bool: True si succès, False sinon.
        """
        start_time = datetime.now()
        try:
            if not config.get("ibkr_enabled", True):
                miya_speak(
                    "Collecte IBKR désactivée", tag="PIPELINE", level="info", priority=2
                )
                AlertManager().send_alert("Collecte IBKR désactivée", priority=1)
                logger.info("Collecte IBKR désactivée")
                return True

            miya_speak(
                "Lancement collecte IBKR", tag="PIPELINE", level="info", priority=2
            )
            AlertManager().send_alert("Lancement collecte IBKR", priority=1)
            logger.info("Lancement collecte IBKR")
            await ibkr_fetch_main(duration=config.get("ibkr_duration", 60))
            result = True

            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                "Collecte IBKR terminée", tag="PIPELINE", level="info", priority=2
            )
            AlertManager().send_alert("Collecte IBKR terminée", priority=1)
            logger.info("Collecte IBKR terminée")
            self.log_performance("run_ibkr_fetch", latency, success=True)
            self.save_snapshot(
                "run_ibkr_fetch", {"duration": config.get("ibkr_duration", 60)}
            )
            return result
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur collecte IBKR: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="PIPELINE", voice_profile="urgent", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("run_ibkr_fetch", latency, success=False, error=str(e))
            self.save_snapshot("run_ibkr_fetch", {"error": str(e)})
            raise

    @with_retries(max_attempts=3, delay_base=2.0)
    async def run_news_scraper(self, config: Dict) -> bool:
        """
        Exécute le scraping des nouvelles macro.

        Args:
            config (Dict): Configuration du pipeline.

        Returns:
            bool: True si succès, False sinon.
        """
        start_time = datetime.now()
        try:
            if not config.get("news_enabled", True):
                miya_speak(
                    "Scraping news désactivé", tag="PIPELINE", level="info", priority=2
                )
                AlertManager().send_alert("Scraping news désactivé", priority=1)
                logger.info("Scraping news désactivé")
                return True

            news_config = config["news_config"]
            miya_speak(
                "Lancement scraping news", tag="PIPELINE", level="info", priority=2
            )
            AlertManager().send_alert("Lancement scraping news", priority=1)
            logger.info("Lancement scraping news")
            df = scrape_news_events(
                news_config["url"], news_config["output_path"], news_config
            )
            result = True

            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                f"Scraping news terminé: {len(df)} événements",
                tag="PIPELINE",
                level="info",
                priority=2,
            )
            AlertManager().send_alert(
                f"Scraping news terminé: {len(df)} événements", priority=1
            )
            logger.info(f"Scraping news terminé: {len(df)} événements")
            self.log_performance(
                "run_news_scraper", latency, success=True, num_events=len(df)
            )
            self.save_snapshot(
                "run_news_scraper", {"num_events": len(df), "url": news_config["url"]}
            )
            return result
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur scraping news: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="PIPELINE", voice_profile="urgent", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "run_news_scraper", latency, success=False, error=str(e)
            )
            self.save_snapshot("run_news_scraper", {"error": str(e)})
            raise

    @with_retries(max_attempts=3, delay_base=2.0)
    async def run_merge_data(self, config: Dict) -> bool:
        """
        Exécute la fusion des données.

        Args:
            config (Dict): Configuration du pipeline.

        Returns:
            bool: True si succès, False sinon.
        """
        start_time = datetime.now()
        try:
            if not config.get("merge_enabled", True):
                miya_speak(
                    "Fusion désactivée", tag="PIPELINE", level="info", priority=2
                )
                AlertManager().send_alert("Fusion désactivée", priority=1)
                logger.info("Fusion désactivée")
                return True

            merge_config = config["merge_config"]
            miya_speak(
                "Lancement fusion données", tag="PIPELINE", level="info", priority=2
            )
            AlertManager().send_alert("Lancement fusion données", priority=1)
            logger.info("Lancement fusion données")
            df = merge_data_sources(
                merge_config["ibkr"],
                merge_config["news"],
                merge_config["merged"],
                merge_config,
            )
            result = True

            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                f"Fusion terminée: {len(df)} lignes",
                tag="PIPELINE",
                level="info",
                priority=2,
            )
            AlertManager().send_alert(f"Fusion terminée: {len(df)} lignes", priority=1)
            logger.info(f"Fusion terminée: {len(df)} lignes")
            self.log_performance(
                "run_merge_data", latency, success=True, num_rows=len(df)
            )
            self.save_snapshot(
                "run_merge_data",
                {"num_rows": len(df), "output_path": merge_config["merged"]},
            )
            return result
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur fusion données: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="PIPELINE", voice_profile="urgent", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("run_merge_data", latency, success=False, error=str(e))
            self.save_snapshot("run_merge_data", {"error": str(e)})
            raise

    @with_retries(max_attempts=3, delay_base=2.0)
    async def run_pipeline(self) -> Dict:
        """
        Exécute le pipeline complet avec retries.

        Returns:
            Dict: Statut du pipeline.
        """
        start_time = datetime.now()
        try:
            status = {
                "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ibkr_success": False,
                "news_success": False,
                "merge_success": False,
                "errors": [],
            }

            status["ibkr_success"] = await self.run_ibkr_fetch(self.config)
            status["news_success"] = await self.run_news_scraper(self.config)
            status["merge_success"] = await self.run_merge_data(self.config)

            if all(
                [
                    status["ibkr_success"],
                    status["news_success"],
                    status["merge_success"],
                ]
            ):
                self.validate_outputs(self.config)
                miya_speak(
                    "Pipeline exécuté avec succès",
                    tag="PIPELINE",
                    level="info",
                    priority=3,
                )
                AlertManager().send_alert("Pipeline exécuté avec succès", priority=1)
                logger.info("Pipeline exécuté avec succès")
            else:
                error_msg = "Échec d’un ou plusieurs modules du pipeline"
                status["errors"].append(error_msg)
                miya_alerts(
                    error_msg, tag="PIPELINE", voice_profile="urgent", priority=5
                )
                AlertManager().send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)

            self.save_dashboard_status(status)
            self.plot_pipeline_results(status)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "run_pipeline",
                latency,
                success=all(
                    [
                        status["ibkr_success"],
                        status["news_success"],
                        status["merge_success"],
                    ]
                ),
            )
            self.save_snapshot("run_pipeline", status)
            return status
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur pipeline: {str(e)}\n{traceback.format_exc()}"
            status["errors"].append(error_msg)
            miya_alerts(error_msg, tag="PIPELINE", voice_profile="urgent", priority=5)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("run_pipeline", latency, success=False, error=str(e))
            self.save_snapshot("run_pipeline", {"error": str(e)})
            self.save_dashboard_status(status)
            self.plot_pipeline_results(status)
            raise

    def signal_handler(self, sig, frame) -> None:
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
                "Arrêt du pipeline en cours...",
                tag="PIPELINE",
                voice_profile="calm",
                priority=2,
            )
            AlertManager().send_alert("Arrêt du pipeline en cours", priority=2)
            logger.info("Arrêt du pipeline initié")

            status = {
                "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ibkr_success": False,
                "news_success": False,
                "merge_success": False,
                "errors": ["Pipeline arrêté manuellement"],
            }
            self.save_dashboard_status(status)
            self.save_snapshot("shutdown", {"status": status})

            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                "Pipeline arrêté proprement",
                tag="PIPELINE",
                voice_profile="calm",
                priority=2,
            )
            AlertManager().send_alert("Pipeline arrêté proprement", priority=1)
            logger.info("Pipeline arrêté")
            self.log_performance("signal_handler", latency, success=True)
            exit(0)
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur arrêt pipeline: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="PIPELINE", voice_profile="urgent", priority=5)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("signal_handler", latency, success=False, error=str(e))
            exit(1)

    async def run(self) -> None:
        """
        Exécute le gestionnaire du pipeline.
        """
        try:
            signal.signal(signal.SIGINT, self.signal_handler)
            status = await self.run_pipeline()
            if not all(
                [
                    status["ibkr_success"],
                    status["news_success"],
                    status["merge_success"],
                ]
            ):
                error_msg = "Pipeline terminé avec des erreurs"
                miya_alerts(
                    error_msg, tag="PIPELINE", voice_profile="urgent", priority=5
                )
                AlertManager().send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
        except Exception as e:
            error_msg = f"Erreur pipeline: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="PIPELINE", voice_profile="urgent", priority=5)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            exit(1)


if __name__ == "__main__":
    runner = PipelineRunner()
    asyncio.run(runner.run())
