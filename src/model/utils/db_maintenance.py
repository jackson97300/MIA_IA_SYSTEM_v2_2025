# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/model/utils/db_maintenance.py
# Rôle : Maintient la base de données market_memory.db en purgeant les données obsolètes (plus de 30 jours)
#        dans la table clusters pour la mémoire contextuelle (méthode 7).
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Dépendances :
# - sqlite3, psutil>=5.9.8,<6.0.0, pandas>=2.0.0,<3.0.0, boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0,
#   pyyaml>=6.0.0,<7.0.0, signal
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - data/market_memory_<market>.db (table clusters avec colonnes cluster_id, event_type, features, timestamp)
# - config/market_config.yaml (chemin de la base de données)
#
# Outputs :
# - Logs dans data/logs/db_maintenance.log
# - Logs de performance dans data/logs/db_maintenance_performance.csv
# - Snapshots JSON compressés dans data/cache/db_maintenance/<market>/*.json.gz
# - Sauvegardes incrémentielles dans data/checkpoints/db_maintenance/<market>/*.json.gz
#
# Notes :
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence) via market_memory.db.
# - Utilise IQFeed exclusivement via data_provider.py pour les données stockées dans market_memory.db.
# - Supprime toute référence à 320 features, 81 features, et obs_t.
# - Implémente retries (max 3, délai 2^attempt secondes) pour les opérations SQL.
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des opérations.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.
# - Tests unitaires disponibles dans tests/test_db_maintenance.py.

import gzip
import json
import signal
import sqlite3
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
import pandas as pd
import psutil
from loguru import logger

from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import get_config
from src.utils.telegram_alert import send_telegram_alert

# Configuration logging
logger.remove()
BASE_DIR = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
LOG_DIR = BASE_DIR / "data" / "logs"
CACHE_DIR = BASE_DIR / "data" / "cache" / "db_maintenance"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints" / "db_maintenance"
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "db_maintenance.log", rotation="10 MB", level="INFO", encoding="utf-8"
)
logger.add(lambda msg: print(msg, end=""), level="INFO")

# Constantes
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Secondes, exponentiel
PERF_LOG_PATH = LOG_DIR / "db_maintenance_performance.csv"
MAX_CACHE_SIZE = 1000
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB

# Cache global pour les résultats de purge_old_data
maintenance_cache = OrderedDict()


class DBMaintenance:
    """Maintient la base de données market_memory.db en purgeant les données obsolètes."""

    def __init__(
        self,
        config_path: str = str(BASE_DIR / "config" / "market_config.yaml"),
        market: str = "ES",
    ):
        """
        Initialise le gestionnaire de maintenance de la base de données.

        Args:
            config_path (str): Chemin vers le fichier de configuration.
            market (str): Marché (ex. : ES, MNQ).
        """
        self.market = market
        self.alert_manager = AlertManager()
        self.config = get_config(config_path).get("db_maintenance", {})
        self.db_path = self.config.get(
            "db_path", str(BASE_DIR / f"data/market_memory_{market}.db")
        )
        self.snapshot_dir = CACHE_DIR / market
        self.checkpoint_dir = CHECKPOINT_DIR / market
        self.snapshot_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        signal.signal(signal.SIGINT, self.handle_sigint)
        logger.info(f"DBMaintenance initialisé pour {market}")
        self.log_performance("init", 0, success=True)

    def handle_sigint(self, signum, frame):
        """Gère l'arrêt propre sur SIGINT."""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "status": "SIGINT",
            "market": self.market,
        }
        snapshot_path = self.snapshot_dir / f'sigint_{snapshot["timestamp"]}.json.gz'
        try:
            with gzip.open(snapshot_path, "wt", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=4)
            success_msg = (
                f"Arrêt propre sur SIGINT, snapshot sauvegardé pour {self.market}"
            )
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot SIGINT pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
        exit(0)

    def log_performance(
        self,
        operation: str,
        latency: float,
        success: bool = True,
        error: str = None,
        **kwargs,
    ) -> None:
        """
        Enregistre les performances (CPU, mémoire, latence) dans db_maintenance_performance.csv.

        Args:
            operation (str): Nom de l’opération (ex. : purge_old_data).
            latency (float): Temps d’exécution en secondes.
            success (bool): Indique si l’opération a réussi.
            error (str, optional): Message d’erreur si applicable.
            **kwargs: Paramètres supplémentaires.
        """
        cache_key = f"{self.market}_{operation}_{hash(str(latency))}_{hash(str(error))}"
        if cache_key in maintenance_cache:
            return
        while len(maintenance_cache) > MAX_CACHE_SIZE:
            maintenance_cache.popitem(last=False)

        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
            cpu_percent = psutil.cpu_percent()
            confidence_drop_rate = 1.0 if success else 0.0  # Simplifié pour Phase 8
            if memory_usage > 1024:
                alert_msg = f"ALERTE: Usage mémoire élevé ({memory_usage:.2f} MB) pour {self.market}"
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
                "confidence_drop_rate": confidence_drop_rate,
                "market": self.market,
                **kwargs,
            }
            log_df = pd.DataFrame([log_entry])

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
            success_msg = (
                f"Performance journalisée pour {operation}. CPU: {cpu_percent}%"
            )
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.save_snapshot("log_performance", log_entry)
            maintenance_cache[cache_key] = True
        except Exception as e:
            error_msg = f"Erreur journalisation performance pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)

    def save_snapshot(
        self, snapshot_type: str, data: Dict, compress: bool = True
    ) -> None:
        """
        Sauvegarde un instantané JSON des résultats, compressé avec gzip.

        Args:
            snapshot_type (str): Type de snapshot (ex. : purge_old_data).
            data (Dict): Données à sauvegarder.
            compress (bool): Compresser avec gzip (défaut : True).
        """
        start_time = time.time()
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = {
                "timestamp": timestamp,
                "type": snapshot_type,
                "market": self.market,
                "data": data,
            }
            snapshot_path = (
                self.snapshot_dir / f"snapshot_{snapshot_type}_{timestamp}.json"
            )

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
                alert_msg = (
                    f"Snapshot size {file_size:.2f} MB exceeds 1 MB pour {self.market}"
                )
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
            latency = time.time() - start_time
            success_msg = (
                f"Snapshot {snapshot_type} sauvegardé pour {self.market}: {save_path}"
            )
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "save_snapshot", latency, success=True, snapshot_size_mb=file_size
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot {snapshot_type} pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance("save_snapshot", 0, success=False, error=str(e))

    def checkpoint(
        self, data: pd.DataFrame, data_type: str = "db_maintenance_state"
    ) -> None:
        """
        Sauvegarde incrémentielle des données toutes les 5 minutes avec versionnage (5 versions).

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : db_maintenance_state).
        """
        try:
            start_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_data = {
                "timestamp": timestamp,
                "num_rows": len(data),
                "columns": list(data.columns),
                "data_type": data_type,
                "market": self.market,
            }
            checkpoint_path = (
                self.checkpoint_dir / f"db_maintenance_{data_type}_{timestamp}.json.gz"
            )
            checkpoint_versions = []

            def write_checkpoint():
                with gzip.open(checkpoint_path, "wt", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, indent=4)
                data.to_csv(
                    checkpoint_path.with_suffix(".csv"), index=False, encoding="utf-8"
                )

            self.with_retries(write_checkpoint)
            checkpoint_versions.append(checkpoint_path)
            if len(checkpoint_versions) > 5:
                oldest = checkpoint_versions.pop(0)
                if oldest.exists():
                    oldest.unlink()
                csv_oldest = oldest.with_suffix(".csv")
                if csv_oldest.exists():
                    csv_oldest.unlink()
            file_size = os.path.getsize(checkpoint_path) / 1024 / 1024
            latency = time.time() - start_time
            success_msg = f"Checkpoint sauvegardé pour {self.market}: {checkpoint_path}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "checkpoint",
                latency,
                success=True,
                file_size_mb=file_size,
                num_rows=len(data),
                data_type=data_type,
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde checkpoint pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "checkpoint", 0, success=False, error=str(e), data_type=data_type
            )

    def cloud_backup(
        self, data: pd.DataFrame, data_type: str = "db_maintenance_state"
    ) -> None:
        """
        Sauvegarde distribuée des données vers AWS S3 toutes les 15 minutes.

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : db_maintenance_state).
        """
        try:
            start_time = time.time()
            config = get_config(str(BASE_DIR / "config/es_config.yaml"))
            if not config.get("s3_bucket"):
                warning_msg = f"S3 bucket non configuré, sauvegarde cloud ignorée pour {self.market}"
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{config['s3_prefix']}db_maintenance_{data_type}_{self.market}_{timestamp}.csv.gz"
            temp_path = self.checkpoint_dir / f"temp_s3_{timestamp}.csv.gz"
            temp_path.parent.mkdir(exist_ok=True)

            def write_temp():
                data.to_csv(
                    temp_path, compression="gzip", index=False, encoding="utf-8"
                )

            self.with_retries(write_temp)
            s3_client = boto3.client("s3")

            def upload_s3():
                s3_client.upload_file(str(temp_path), config["s3_bucket"], backup_path)

            self.with_retries(upload_s3)
            temp_path.unlink()
            latency = time.time() - start_time
            success_msg = (
                f"Sauvegarde cloud S3 effectuée pour {self.market}: {backup_path}"
            )
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "cloud_backup",
                latency,
                success=True,
                num_rows=len(data),
                data_type=data_type,
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde cloud S3 pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
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
        start_time = time.time()
        for attempt in range(max_attempts):
            try:
                result = func()
                latency = time.time() - start_time
                self.log_performance(
                    f"retry_attempt_{attempt+1}",
                    latency,
                    success=True,
                    attempt_number=attempt + 1,
                )
                return result
            except Exception as e:
                if attempt == max_attempts - 1:
                    error_msg = f"Échec après {max_attempts} tentatives pour {self.market}: {str(e)}\n{traceback.format_exc()}"
                    logger.error(error_msg)
                    self.alert_manager.send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    self.log_performance(
                        f"retry_attempt_{attempt+1}",
                        time.time() - start_time,
                        success=False,
                        error=str(e),
                        attempt_number=attempt + 1,
                    )
                    return None
                delay = delay_base**attempt
                warning_msg = f"Tentative {attempt+1} échouée pour {self.market}, retry après {delay}s"
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                time.sleep(delay)

    def verify_indexes(self, cursor: sqlite3.Cursor) -> bool:
        """
        Vérifie la présence des index nécessaires dans la table clusters.

        Args:
            cursor (sqlite3.Cursor): Curseur de connexion à la base de données.

        Returns:
            bool: True si les index sont présents, False sinon.
        """
        try:
            cursor.execute("PRAGMA index_list(clusters)")
            indexes = [index[1] for index in cursor.fetchall()]
            expected_indexes = ["idx_cluster_id", "idx_timestamp"]
            missing_indexes = [
                index for index in expected_indexes if index not in indexes
            ]
            if missing_indexes:
                warning_msg = f"Index manquants dans la table clusters pour {self.market}: {missing_indexes}"
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                return False
            return True
        except Exception as e:
            error_msg = f"Erreur vérification index pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            return False

    def purge_old_data(self) -> int:
        """
        Purge les données de plus de 30 jours dans la table clusters de market_memory.db.

        Returns:
            int: Nombre de lignes supprimées.
        """
        start_time = time.time()
        try:
            cache_key = f"{self.market}_{hash(self.db_path)}"
            if cache_key in maintenance_cache:
                return maintenance_cache[cache_key]
            while len(maintenance_cache) > MAX_CACHE_SIZE:
                maintenance_cache.popitem(last=False)

            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.access(db_dir, os.W_OK):
                error_msg = (
                    f"Permission d’écriture refusée pour {db_dir} pour {self.market}"
                )
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                self.log_performance(
                    "purge_old_data",
                    time.time() - start_time,
                    success=False,
                    error="Permission refusée",
                )
                return 0

            def connect_db():
                return sqlite3.connect(self.db_path)

            conn = self.with_retries(connect_db)
            if not conn:
                error_msg = f"Échec de la connexion à {self.db_path} pour {self.market}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                self.log_performance(
                    "purge_old_data",
                    time.time() - start_time,
                    success=False,
                    error="Échec de connexion",
                )
                return 0
            cursor = conn.cursor()

            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='clusters'"
            )
            if not cursor.fetchone():
                error_msg = (
                    f"Table clusters non trouvée dans market_memory_{self.market}.db"
                )
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                conn.close()
                self.log_performance(
                    "purge_old_data",
                    time.time() - start_time,
                    success=False,
                    error="Table clusters non trouvée",
                )
                return 0

            if not self.verify_indexes(cursor):
                warning_msg = (
                    f"Purge non optimisée pour {self.market}: index manquants détectés"
                )
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)

            cursor.execute(
                "SELECT COUNT(*) FROM clusters WHERE timestamp < date('now', '-30 days')"
            )
            rows_to_delete = cursor.fetchone()[0]

            def execute_purge():
                cursor.execute(
                    "DELETE FROM clusters WHERE timestamp < date('now', '-30 days')"
                )
                conn.commit()

            self.with_retries(execute_purge)

            cursor.execute(
                "SELECT COUNT(*) FROM clusters WHERE timestamp < date('now', '-30 days')"
            )
            remaining_rows = cursor.fetchone()[0]
            deleted_rows = rows_to_delete - remaining_rows

            snapshot_data = {
                "deleted_rows": deleted_rows,
                "remaining_rows": remaining_rows,
            }
            self.save_snapshot("purge_old_data", snapshot_data)
            self.checkpoint(pd.DataFrame([snapshot_data]), data_type="purge_old_data")
            self.cloud_backup(pd.DataFrame([snapshot_data]), data_type="purge_old_data")

            success_msg = f"Données obsolètes purgées pour {self.market}: {deleted_rows} lignes supprimées"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            self.log_performance(
                "purge_old_data",
                time.time() - start_time,
                success=True,
                deleted_rows=deleted_rows,
            )
            maintenance_cache[cache_key] = deleted_rows
            conn.close()
            return deleted_rows

        except Exception as e:
            error_msg = f"Erreur dans purge_old_data pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "purge_old_data", time.time() - start_time, success=False, error=str(e)
            )
            return 0


if __name__ == "__main__":
    db_maintenance = DBMaintenance()
    deleted_rows = db_maintenance.purge_old_data()
    print(f"Données purgées pour ES: {deleted_rows} lignes supprimées")
