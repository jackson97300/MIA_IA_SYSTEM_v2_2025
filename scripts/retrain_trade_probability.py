# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/scripts/retrain_trade_probability.py
# Script pour re-entraîner TradeProbabilityPredictor périodiquement.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Relance TradeProbabilityPredictor.train() toutes les 2 semaines ou si 1000 nouveaux trades sont détectés.
#        Conforme à la Phase 8 (auto-conscience via alertes), Phase 9 (entraînement des modèles de probabilité),
#        et Phase 16 (ensemble learning).
#
# Dépendances : schedule>=1.2.0, sqlite3, psutil>=5.9.8, pandas>=2.0.0, json, yaml>=6.0.0, gzip, datetime, traceback,
#               src.model.trade_probability, src.model.utils.config_manager, src.model.utils.alert_manager,
#               src.model.utils.miya_console, src.utils.telegram_alert, src.utils.standard
#
# Inputs : config/trade_probability_config.yaml, data/market_memory.db
#
# Outputs : data/logs/retrain_trade_probability.log,
#           data/logs/trade_probability_performance.csv,
#           data/trade_probability_snapshots/snapshot_*.json.gz
#
# Notes :
# - Utilise IQFeed comme source de données via data_provider.py (indirectement).
# - Implémente retries (max 3, délai 2^attempt), logs psutil, alertes via alert_manager.py et miya_console.
# - Tests unitaires dans tests/test_retrain_trade_probability.py.

import gzip
import json
import logging
import sqlite3
import time
import traceback
from datetime import datetime
from typing import Dict

import pandas as pd
import psutil
import schedule

from src.model.trade_probability import TradeProbabilityPredictor
from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import config_manager
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.utils.standard import with_retries
from src.utils.telegram_alert import send_telegram_alert

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "trade_probability_config.yaml")
DB_PATH = os.path.join(BASE_DIR, "data", "market_memory.db")
LOG_DIR = os.path.join(BASE_DIR, "data", "logs")
CSV_LOG_PATH = os.path.join(
    BASE_DIR, "data", "logs", "trade_probability_performance.csv"
)
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "trade_probability_snapshots")

# Configuration du logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "retrain_trade_probability.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)


class TradeProbabilityRetrain:
    """
    Classe pour gérer le re-entraînement périodique de TradeProbabilityPredictor.
    """

    def __init__(self):
        """
        Initialise le gestionnaire de re-entraînement.
        """
        self.config = None
        self.log_buffer = []
        self.buffer_size = 100
        try:
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            self.validate_config()
            miya_speak(
                "TradeProbabilityRetrain initialisé",
                tag="RETRAIN_TRADE_PROBABILITY",
                voice_profile="calm",
                priority=2,
            )
            AlertManager().send_alert("TradeProbabilityRetrain initialisé", priority=1)
            logger.info("TradeProbabilityRetrain initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config_path": CONFIG_PATH})
        except Exception as e:
            error_msg = f"Erreur initialisation TradeProbabilityRetrain: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg,
                tag="RETRAIN_TRADE_PROBABILITY",
                voice_profile="urgent",
                priority=4,
            )
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            raise

    def validate_config(self):
        """
        Valide la configuration et l’environnement.

        Raises:
            FileNotFoundError: Si le fichier de configuration est introuvable.
            ValueError: Si la configuration est invalide.
        """
        start_time = datetime.now()
        try:
            if not os.path.exists(CONFIG_PATH):
                error_msg = f"Fichier de configuration introuvable : {CONFIG_PATH}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            self.config = config_manager.get_config("trade_probability_config.yaml")
            if "trade_probability" not in self.config:
                error_msg = "Clé 'trade_probability' manquante dans la configuration"
                logger.error(error_msg)
                raise ValueError(error_msg)

            required_keys = ["retrain_threshold", "retrain_frequency"]
            missing_keys = [
                key
                for key in required_keys
                if key not in self.config["trade_probability"]
            ]
            if missing_keys:
                error_msg = f"Clés manquantes dans 'trade_probability': {missing_keys}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                "Configuration validée pour le re-entraînement",
                tag="RETRAIN_TRADE_PROBABILITY",
                level="info",
                priority=2,
            )
            AlertManager().send_alert(
                "Configuration validée pour le re-entraînement", priority=1
            )
            logger.info("Configuration validée pour le re-entraînement")
            self.log_performance("validate_config", latency, success=True)
            self.save_snapshot(
                "validate_config", {"config": self.config["trade_probability"]}
            )
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur lors de la validation de la configuration : {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            miya_alerts(
                error_msg,
                tag="RETRAIN_TRADE_PROBABILITY",
                voice_profile="urgent",
                priority=4,
            )
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "validate_config", latency, success=False, error=str(e)
            )
            self.save_snapshot("validate_config", {"error": str(e)})
            raise

    def validate_db(self, db_path: str = DB_PATH) -> None:
        """
        Valide l’intégrité de la base de données market_memory.db.

        Args:
            db_path (str): Chemin de la base de données.

        Raises:
            FileNotFoundError: Si la base de données est introuvable.
            ValueError: Si la table patterns est absente ou mal formée.
        """
        start_time = datetime.now()
        try:
            if not os.path.exists(db_path):
                error_msg = f"Base de données introuvable : {db_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='patterns'"
            )
            if not cursor.fetchone():
                error_msg = "Table 'patterns' manquante dans la base de données"
                logger.error(error_msg)
                raise ValueError(error_msg)

            conn.close()
            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                "Base de données validée",
                tag="RETRAIN_TRADE_PROBABILITY",
                level="info",
                priority=2,
            )
            AlertManager().send_alert("Base de données validée", priority=1)
            logger.info("Base de données validée")
            self.log_performance("validate_db", latency, success=True)
            self.save_snapshot("validate_db", {"db_path": db_path})
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur validation base de données : {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            miya_alerts(
                error_msg,
                tag="RETRAIN_TRADE_PROBABILITY",
                voice_profile="urgent",
                priority=4,
            )
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance("validate_db", latency, success=False, error=str(e))
            self.save_snapshot("validate_db", {"error": str(e)})
            raise

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ) -> None:
        """
        Journalise les performances des opérations critiques.

        Args:
            operation (str): Nom de l’opération (ex. : retrain_job).
            latency (float): Latence en secondes.
            success (bool): Succès de l’opération.
            error (str, optional): Message d’erreur si échec.
            **kwargs: Paramètres supplémentaires (ex. : pattern_count).
        """
        try:
            memory_usage = (
                psutil.Process().memory_info().rss / 1024 / 1024
            )  # Mémoire en Mo
            cpu_percent = psutil.cpu_percent()
            if memory_usage > 1024:
                error_msg = f"ALERT: Utilisation mémoire élevée ({memory_usage:.2f} MB)"
                miya_alerts(
                    error_msg,
                    tag="RETRAIN_TRADE_PROBABILITY",
                    level="error",
                    priority=5,
                )
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
            latency = time.time() - start_time
            logger.info(
                f"Performance journalisée pour {operation}: latence={latency:.2f}s, succès={success}"
            )
        except Exception as e:
            error_msg = f"Erreur journalisation performance: {str(e)}"
            miya_alerts(
                error_msg, tag="RETRAIN_TRADE_PROBABILITY", level="error", priority=3
            )
            AlertManager().send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def save_snapshot(self, snapshot_type: str, data: Dict) -> None:
        """
        Sauvegarde un instantané des résultats avec compression gzip.

        Args:
            snapshot_type (str): Type de snapshot (ex. : retrain_job).
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
                tag="RETRAIN_TRADE_PROBABILITY",
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
            miya_alerts(
                error_msg, tag="RETRAIN_TRADE_PROBABILITY", level="error", priority=3
            )
            AlertManager().send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("save_snapshot", 0, success=False, error=str(e))

    @with_retries(max_attempts=3, delay_base=2.0)
    def count_patterns(self, db_path: str = DB_PATH) -> int:
        """
        Compte le nombre de patterns dans market_memory.db.

        Args:
            db_path (str): Chemin de la base de données.

        Returns:
            int: Nombre de patterns.
        """
        start_time = datetime.now()
        try:
            self.validate_db(db_path)
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM patterns")
            count = cursor.fetchone()[0]
            conn.close()
            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                f"Comptage patterns: {count} trouvés",
                tag="RETRAIN_TRADE_PROBABILITY",
                level="info",
                priority=2,
            )
            AlertManager().send_alert(f"Comptage patterns: {count} trouvés", priority=1)
            logger.info(f"Comptage patterns: {count} trouvés")
            self.log_performance(
                "count_patterns", latency, success=True, pattern_count=count
            )
            self.save_snapshot(
                "count_patterns", {"pattern_count": count, "db_path": db_path}
            )
            return count
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur comptage patterns: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg,
                tag="RETRAIN_TRADE_PROBABILITY",
                voice_profile="urgent",
                priority=4,
            )
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("count_patterns", latency, success=False, error=str(e))
            self.save_snapshot("count_patterns", {"error": str(e)})
            raise

    @with_retries(max_attempts=3, delay_base=2.0)
    def retrain_job(self) -> None:
        """
        Re-entraîne TradeProbabilityPredictor si nécessaire.
        """
        start_time = datetime.now()
        try:
            predictor = TradeProbabilityPredictor()
            pattern_count = self.count_patterns()
            threshold = self.config["trade_probability"]["retrain_threshold"]

            if pattern_count >= threshold:
                predictor.train()
                predictor.backtest_threshold()
                latency = (datetime.now() - start_time).total_seconds()
                miya_speak(
                    f"Re-entraînement terminé: {pattern_count} patterns",
                    tag="RETRAIN_TRADE_PROBABILITY",
                    level="info",
                    priority=3,
                )
                AlertManager().send_alert(
                    f"Re-entraînement terminé: {pattern_count} patterns", priority=2
                )
                logger.info(
                    f"Re-entraînement terminé: {pattern_count} patterns, latence={latency:.2f}s"
                )
                self.log_performance(
                    "retrain_job", latency, success=True, pattern_count=pattern_count
                )
                self.save_snapshot(
                    "retrain_job", {"pattern_count": pattern_count, "status": "success"}
                )
            else:
                latency = (datetime.now() - start_time).total_seconds()
                miya_speak(
                    f"Pas assez de nouveaux patterns: {pattern_count}/{threshold}",
                    tag="RETRAIN_TRADE_PROBABILITY",
                    level="info",
                    priority=2,
                )
                AlertManager().send_alert(
                    f"Pas assez de nouveaux patterns: {pattern_count}/{threshold}",
                    priority=1,
                )
                logger.info(
                    f"Pas assez de nouveaux patterns: {pattern_count}/{threshold}"
                )
                self.log_performance(
                    "retrain_job", latency, success=True, pattern_count=pattern_count
                )
                self.save_snapshot(
                    "retrain_job", {"pattern_count": pattern_count, "status": "skipped"}
                )
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur re-entraînement: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg,
                tag="RETRAIN_TRADE_PROBABILITY",
                voice_profile="urgent",
                priority=4,
            )
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("retrain_job", latency, success=False, error=str(e))
            self.save_snapshot("retrain_job", {"error": str(e)})
            raise

    def run(self):
        """
        Exécute le planificateur pour le re-entraînement périodique.
        """
        try:
            frequency = self.config["trade_probability"].get(
                "retrain_frequency", "biweekly"
            )
            if frequency == "biweekly":
                schedule.every(14).days.do(self.retrain_job)
            else:
                schedule.every().day.do(self.retrain_job)

            miya_speak(
                "Planificateur de re-entraînement démarré",
                tag="RETRAIN_TRADE_PROBABILITY",
                level="info",
                priority=3,
            )
            AlertManager().send_alert(
                "Planificateur de re-entraînement démarré", priority=1
            )
            logger.info("Planificateur de re-entraînement démarré")

            while True:
                schedule.run_pending()
                time.sleep(60)
        except Exception as e:
            error_msg = f"Erreur planificateur: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg,
                tag="RETRAIN_TRADE_PROBABILITY",
                voice_profile="urgent",
                priority=5,
            )
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)


if __name__ == "__main__":
    try:
        retrainer = TradeProbabilityRetrain()
        retrainer.run()
    except Exception as e:
        error_msg = f"Échec programme: {str(e)}\n{traceback.format_exc()}"
        miya_alerts(
            error_msg,
            tag="RETRAIN_TRADE_PROBABILITY",
            voice_profile="urgent",
            priority=5,
        )
        AlertManager().send_alert(error_msg, priority=4)
        send_telegram_alert(error_msg)
        logger.error(error_msg)
        exit(1)
