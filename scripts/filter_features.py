# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/scripts/filter_features.py
# Script pour filtrer les lignes valides de features_latest.csv jusqu'à un timestamp maximum.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Filtre les données de features en fonction d’un timestamp maximum, avec journalisation,
#        snapshots compressés, et alertes.
#        Conforme à la Phase 7 (gestion des features), Phase 8 (auto-conscience via alertes),
#        et Phase 16 (ensemble learning).
#
# Dépendances : pandas>=2.0.0, argparse, logging, psutil>=5.9.8, yaml>=6.0.0, os, datetime,
#               typing, gzip, traceback,
#               src.model.utils.config_manager, src.model.utils.alert_manager,
#               src.model.utils.miya_console, src.utils.telegram_alert, src.utils.standard
#
# Inputs : config/es_config.yaml, data(features/features_latest.csv
#
# Outputs : data/features/features_latest_filtered.csv, data/logs/filter_features_performance.csv,
#           data/filter_snapshots/snapshot_*.json.gz
#
# Notes :
# - Gère 350 features pour l’entraînement et 150 SHAP features pour l’inférence.
# - Utilise IQFeed exclusivement via les features générées (indirectement).
# - Implémente retries (max 3, délai 2^attempt), logs psutil, snapshots JSON compressés, alertes centralisées.
# - Tests unitaires dans tests/test_filter_features.py.

import argparse
import gzip
import logging
import os
import traceback
from datetime import datetime
from typing import Dict

import pandas as pd
import psutil

from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import config_manager
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.utils.standard import with_retries
from src.utils.telegram_alert import send_telegram_alert

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "es_config.yaml")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "filter_snapshots")
CSV_LOG_PATH = os.path.join(BASE_DIR, "data", "logs", "filter_features_performance.csv")

# Configuration du logging
os.makedirs(os.path.join(BASE_DIR, "data", "logs"), exist_ok=True)
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "data", "logs", "filter_features.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [FilterFeatures] %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Configuration par défaut
DEFAULT_CONFIG = {
    "filter_features": {
        "input_path": os.path.join(BASE_DIR, "data", "features", "features_latest.csv"),
        "output_path": os.path.join(
            BASE_DIR, "data", "features", "features_latest_filtered.csv"
        ),
        "max_timestamp": "2025-05-13 11:39:00",
        "retry_attempts": 3,
        "retry_delay": 5,
        "buffer_size": 100,
        "max_cache_size": 1000,
    }
}


class FeatureFilter:
    """
    Classe pour gérer le filtrage des features avec journalisation, snapshots et alertes.
    """

    def __init__(self):
        """
        Initialise le gestionnaire de filtrage.
        """
        self.log_buffer = []
        self.cache = {}
        try:
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            self.config = self.load_config()
            self.buffer_size = self.config.get("buffer_size", 100)
            self.max_cache_size = self.config.get("max_cache_size", 1000)
            miya_speak(
                "FeatureFilter initialisé",
                tag="FILTER",
                voice_profile="calm",
                priority=2,
            )
            AlertManager().send_alert("FeatureFilter initialisé", priority=1)
            logger.info("FeatureFilter initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config_path": CONFIG_PATH})
        except Exception as e:
            error_msg = f"Erreur initialisation FeatureFilter: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="FILTER", voice_profile="urgent", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            self.config = DEFAULT_CONFIG["filter_features"]
            self.buffer_size = 100
            self.max_cache_size = 1000

    @with_retries(max_attempts=3, delay_base=2.0)
    def load_config(self, config_path: str = CONFIG_PATH) -> Dict:
        """
        Charge les paramètres de filtrage depuis es_config.yaml.

        Args:
            config_path (str): Chemin du fichier de configuration.

        Returns:
            Dict: Configuration du filtrage.
        """
        start_time = datetime.now()
        try:
            config = config_manager.get_config("es_config.yaml")
            if "filter_features" not in config:
                raise ValueError(
                    "Clé 'filter_features' manquante dans la configuration"
                )
            required_keys = [
                "input_path",
                "output_path",
                "max_timestamp",
                "retry_attempts",
                "retry_delay",
            ]
            missing_keys = [
                key for key in required_keys if key not in config["filter_features"]
            ]
            if missing_keys:
                raise ValueError(
                    f"Clés manquantes dans 'filter_features': {missing_keys}"
                )
            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                "Configuration filtrage chargée", tag="FILTER", level="info", priority=2
            )
            AlertManager().send_alert("Configuration filtrage chargée", priority=1)
            logger.info("Configuration filtrage chargée")
            self.log_performance("load_config", latency, success=True)
            self.save_snapshot("load_config", {"config_path": config_path})
            return config["filter_features"]
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur chargement config: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="FILTER", voice_profile="urgent", priority=4)
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
            operation (str): Nom de l’opération (ex. : filter_features).
            latency (float): Latence en secondes.
            success (bool): Succès de l’opération.
            error (str, optional): Message d’erreur si échec.
            **kwargs: Paramètres supplémentaires (ex. : num_rows).
        """
        try:
            memory_usage = (
                psutil.Process().memory_info().rss / 1024 / 1024
            )  # Mémoire en Mo
            cpu_percent = psutil.cpu_percent()
            if memory_usage > 1024:
                error_msg = f"ALERT: Utilisation mémoire élevée ({memory_usage:.2f} MB)"
                miya_alerts(error_msg, tag="FILTER", level="error", priority=5)
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
            miya_alerts(error_msg, tag="FILTER", level="error", priority=3)
            AlertManager().send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def save_snapshot(self, snapshot_type: str, data: Dict) -> None:
        """
        Sauvegarde un instantané des résultats avec compression gzip.

        Args:
            snapshot_type (str): Type de snapshot (ex. : filter_features).
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
                tag="FILTER",
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
            miya_alerts(error_msg, tag="FILTER", level="error", priority=3)
            AlertManager().send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("save_snapshot", 0, success=False, error=str(e))

    def validate_inputs(
        self, input_path: str, output_path: str, max_timestamp: str
    ) -> None:
        """
        Valide les paramètres et données d’entrée.

        Args:
            input_path (str): Chemin du fichier d’entrée.
            output_path (str): Chemin du fichier de sortie.
            max_timestamp (str): Timestamp maximum pour le filtrage.

        Raises:
            ValueError: Si les paramètres ou données sont invalides.
        """
        start_time = datetime.now()
        try:
            if not os.path.exists(input_path):
                error_msg = f"Fichier d’entrée introuvable: {input_path}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            try:
                pd.to_datetime(max_timestamp)
            except ValueError:
                error_msg = f"Format de timestamp invalide: {max_timestamp}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Vérifier un échantillon du fichier d’entrée
            df_sample = pd.read_csv(input_path, nrows=10, encoding="utf-8")
            if "timestamp" not in df_sample.columns:
                error_msg = f"Colonne 'timestamp' manquante dans {input_path}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                "Paramètres et données d’entrée validés",
                tag="FILTER",
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
                    "output_path": output_path,
                    "max_timestamp": max_timestamp,
                },
            )
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur validation entrées: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="FILTER", voice_profile="urgent", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "validate_inputs", latency, success=False, error=str(e)
            )
            self.save_snapshot("validate_inputs", {"error": str(e)})
            raise

    @with_retries(max_attempts=3, delay_base=2.0)
    def filter_features(
        self,
        input_path: str = None,
        output_path: str = None,
        max_timestamp: str = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Filtre les lignes valides de features_latest.csv jusqu'à max_timestamp.

        Args:
            input_path (str, optional): Chemin du fichier d’entrée. Par défaut, utilise la config.
            output_path (str, optional): Chemin du fichier de sortie. Par défaut, utilise la config.
            max_timestamp (str, optional): Timestamp maximum. Par défaut, utilise la config.
            verbose (bool): Affiche un aperçu des données filtrées si True.

        Returns:
            pd.DataFrame: Données filtrées.
        """
        start_time = datetime.now()
        try:
            input_path = input_path or self.config["input_path"]
            output_path = output_path or self.config["output_path"]
            max_timestamp = max_timestamp or self.config["max_timestamp"]

            self.validate_inputs(input_path, output_path, max_timestamp)

            miya_speak(
                "Démarrage filtrage des features",
                tag="FILTER",
                level="info",
                priority=3,
            )
            AlertManager().send_alert("Démarrage filtrage des features", priority=2)
            logger.info("Démarrage filtrage des features")

            df = pd.read_csv(input_path, parse_dates=["timestamp"], encoding="utf-8")
            df_filtered = df[df["timestamp"] <= pd.to_datetime(max_timestamp)]

            if df_filtered.empty:
                error_msg = "Aucune donnée valide après filtrage"
                miya_alerts(error_msg, tag="FILTER", voice_profile="urgent", priority=5)
                AlertManager().send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                raise ValueError(error_msg)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df_filtered.to_csv(output_path, index=False, encoding="utf-8")

            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                f"Fichier filtré créé: {output_path}, {len(df_filtered)} lignes",
                tag="FILTER",
                level="info",
                priority=2,
            )
            AlertManager().send_alert(
                f"Fichier filtré créé: {output_path}, {len(df_filtered)} lignes",
                priority=1,
            )
            logger.info(
                f"Fichier filtré créé: {output_path}, {len(df_filtered)} lignes"
            )

            if verbose:
                print("Aperçu des premières lignes filtrées :")
                print(df_filtered.head())

            self.log_performance(
                "filter_features", latency, success=True, num_rows=len(df_filtered)
            )
            self.save_snapshot(
                "filter_features",
                {
                    "input_path": input_path,
                    "output_path": output_path,
                    "max_timestamp": max_timestamp,
                    "num_rows": len(df_filtered),
                },
            )

            del df  # Nettoyage mémoire
            return df_filtered
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur filtrage: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="FILTER", voice_profile="urgent", priority=5)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "filter_features", latency, success=False, error=str(e)
            )
            self.save_snapshot("filter_features", {"error": str(e)})
            raise


def main():
    """
    Point d’entrée principal pour le filtrage des features.
    """
    try:
        parser = argparse.ArgumentParser(
            description="Filtre les données de features_latest.csv"
        )
        parser.add_argument(
            "--date", type=str, help="Timestamp maximum pour le filtrage"
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Affiche un aperçu des données filtrées",
        )
        args = parser.parse_args()

        filterer = FeatureFilter()
        filterer.filter_features(max_timestamp=args.date, verbose=args.verbose)
    except Exception as e:
        error_msg = f"Erreur programme: {str(e)}\n{traceback.format_exc()}"
        miya_alerts(error_msg, tag="FILTER", voice_profile="urgent", priority=5)
        AlertManager().send_alert(error_msg, priority=4)
        send_telegram_alert(error_msg)
        logger.error(error_msg)
        exit(1)


if __name__ == "__main__":
    main()
