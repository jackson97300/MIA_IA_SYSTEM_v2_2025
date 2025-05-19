# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/model/utils/pattern_analyzer.py
# Rôle : Analyse les patterns dans market_memory.db (méthode 7) (Phase 15).
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, sqlite3, psutil>=5.9.8,<6.0.0, boto3>=1.26.0,<2.0.0,
#   loguru>=0.7.0,<1.0.0, pyyaml>=6.0.0,<7.0.0
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - data/market_memory.db (base de données SQLite avec données historiques)
# - config/feature_sets.yaml (pour charger les 350 features)
#
# Outputs :
# - Logs dans data/logs/market/pattern_analyzer.log
# - Logs de performance dans data/logs/market/pattern_analyzer_performance.csv
# - Snapshots JSON compressés dans data/cache/patterns/<market>/*.json.gz
# - Sauvegardes incrémentielles dans data/checkpoints/patterns/<market>/*.json.gz
# - Patterns sauvegardés (optionnel) dans data/patterns/<market>/*.json.gz
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Intègre la méthode 7 (analyse des patterns) et la Phase 15 (apprentissage adaptatif).
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.
# - Utilisation exclusive d’IQFeed, suppression des références à dxFeed.
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des données.
# - Tests unitaires disponibles dans tests/test_pattern_analyzer.py.

import gzip
import json
import sqlite3
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import boto3
import numpy as np
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
LOG_DIR = BASE_DIR / "data" / "logs" / "market"
CACHE_DIR = BASE_DIR / "data" / "cache" / "patterns"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints" / "patterns"
PATTERN_DIR = BASE_DIR / "data" / "patterns"
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
PATTERN_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "pattern_analyzer.log", rotation="10 MB", level="INFO", encoding="utf-8"
)
logger.add(lambda msg: print(msg, end=""), level="INFO")

# Constantes
PERF_LOG_PATH = LOG_DIR / "pattern_analyzer_performance.csv"
MAX_RETRIES = 3
RETRY_DELAY = 2.0
MAX_CACHE_SIZE = 1000
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB

# Cache global pour les résultats d’analyse
pattern_cache = OrderedDict()


class PatternAnalyzer:
    """
    Classe pour analyser les patterns dans la base de données market_memory.db.
    """

    def __init__(
        self, db_path: Path = BASE_DIR / "data" / "market_memory.db", market: str = "ES"
    ):
        """
        Initialise l’analyseur de patterns.

        Args:
            db_path (Path): Chemin de la base de données SQLite.
            market (str): Marché (ex. : ES, MNQ).
        """
        self.db_path = db_path
        self.market = market
        self._ensure_db_exists()

    def log_performance(
        self,
        operation: str,
        latency: float,
        success: bool = True,
        error: str = None,
        **kwargs,
    ) -> None:
        """
        Enregistre les performances (CPU, mémoire, latence) dans pattern_analyzer_performance.csv.

        Args:
            operation (str): Nom de l’opération.
            latency (float): Temps d’exécution en secondes.
            success (bool): Indique si l’opération a réussi.
            error (str): Message d’erreur (si applicable).
        """
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
            cpu_percent = psutil.cpu_percent()
            if memory_usage > 1024:
                alert_msg = f"ALERTE: Usage mémoire élevé ({memory_usage:.2f} MB) pour {self.market}"
                logger.warning(alert_msg)
                AlertManager().send_alert(alert_msg, priority=5)
                send_telegram_alert(alert_msg)
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "latency": latency,
                "success": success,
                "error": error,
                "memory_usage_mb": memory_usage,
                "cpu_percent": cpu_percent,
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
            logger.info(
                f"Performance journalisée pour {operation}. CPU: {cpu_percent}%"
            )
        except Exception as e:
            error_msg = f"Erreur journalisation performance pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            AlertManager().send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)

    def save_snapshot(
        self, snapshot_type: str, data: Dict, compress: bool = True
    ) -> None:
        """
        Sauvegarde un instantané JSON des résultats, compressé avec gzip.

        Args:
            snapshot_type (str): Type de snapshot (ex. : pattern_analysis).
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
            snapshot_dir = CACHE_DIR / self.market
            snapshot_dir.mkdir(exist_ok=True)
            snapshot_path = snapshot_dir / f"snapshot_{snapshot_type}_{timestamp}.json"

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
                AlertManager().send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
            latency = time.time() - start_time
            success_msg = (
                f"Snapshot {snapshot_type} sauvegardé pour {self.market}: {save_path}"
            )
            logger.info(success_msg)
            AlertManager().send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "save_snapshot", latency, success=True, snapshot_size_mb=file_size
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot {snapshot_type} pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            AlertManager().send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance("save_snapshot", 0, success=False, error=str(e))

    def checkpoint(self, data: pd.DataFrame, data_type: str = "pattern_state") -> None:
        """
        Sauvegarde incrémentielle des données toutes les 5 minutes avec versionnage (5 versions).

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : pattern_state).
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
            checkpoint_dir = CHECKPOINT_DIR / self.market
            checkpoint_dir.mkdir(exist_ok=True)
            checkpoint_path = (
                checkpoint_dir / f"pattern_{data_type}_{timestamp}.json.gz"
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
            AlertManager().send_alert(success_msg, priority=1)
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
            AlertManager().send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "checkpoint", 0, success=False, error=str(e), data_type=data_type
            )

    def cloud_backup(
        self, data: pd.DataFrame, data_type: str = "pattern_state"
    ) -> None:
        """
        Sauvegarde distribuée des données vers AWS S3 toutes les 15 minutes.

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : pattern_state).
        """
        try:
            start_time = time.time()
            config = get_config(str(BASE_DIR / "config/es_config.yaml"))
            if not config.get("s3_bucket"):
                warning_msg = f"S3 bucket non configuré, sauvegarde cloud ignorée pour {self.market}"
                logger.warning(warning_msg)
                AlertManager().send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{config['s3_prefix']}pattern_{data_type}_{self.market}_{timestamp}.csv.gz"
            temp_path = CHECKPOINT_DIR / self.market / f"temp_s3_{timestamp}.csv.gz"
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
            AlertManager().send_alert(success_msg, priority=1)
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
            AlertManager().send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "cloud_backup", 0, success=False, error=str(e), data_type=data_type
            )

    def with_retries(
        self,
        func: callable,
        max_attempts: int = MAX_RETRIES,
        delay_base: float = RETRY_DELAY,
    ) -> Optional[Any]:
        """
        Exécute une fonction avec retries exponentiels (max 3, délai 2^attempt secondes).

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
                    AlertManager().send_alert(error_msg, priority=4)
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
                AlertManager().send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                time.sleep(delay)

    def _ensure_db_exists(self):
        """
        Vérifie l’existence de la base de données et crée la table si nécessaire.
        """
        start_time = time.time()
        try:
            if not self.db_path.exists():
                logger.warning(
                    f"Base de données introuvable : {self.db_path}. Création d’une nouvelle base."
                )
                self.db_path.parent.mkdir(parents=True, exist_ok=True)

            feature_sets_path = BASE_DIR / "config/feature_sets.yaml"
            features_config = get_config(feature_sets_path)
            feature_cols = [
                f.replace("'", "''")
                for f in features_config.get("training", {}).get("features", [])[:350]
            ]
            feature_definitions = ", ".join([f"{col} REAL" for col in feature_cols])

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS market_data (
                    timestamp TEXT,
                    close REAL,
                    volume INTEGER,
                    vix_close REAL,
                    bid_size_level_2 INTEGER,
                    ask_size_level_2 INTEGER,
                    {feature_definitions}
                )
            """
            )
            conn.commit()
            conn.close()
            success_msg = f"Base de données vérifiée/créée : {self.db_path}"
            logger.info(success_msg)
            AlertManager().send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "ensure_db_exists", time.time() - start_time, success=True
            )
        except Exception as e:
            error_msg = f"Erreur lors de la vérification de la base de données pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            AlertManager().send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance(
                "ensure_db_exists",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            raise

    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Valide les données chargées depuis la base de données avec confidence_drop_rate.

        Args:
            data (pd.DataFrame): Données à valider.

        Raises:
            ValueError: Si les données sont invalides.
        """
        start_time = time.time()
        try:
            feature_sets_path = BASE_DIR / "config/feature_sets.yaml"
            features_config = get_config(feature_sets_path)
            expected_cols = [
                "timestamp",
                "close",
                "volume",
                "vix_close",
                "bid_size_level_2",
                "ask_size_level_2",
            ] + features_config.get("training", {}).get("features", [])[:350]
            missing_cols = [col for col in expected_cols if col not in data.columns]
            null_count = data[expected_cols].isnull().sum().sum()
            confidence_drop_rate = (
                null_count / (len(data) * len(expected_cols))
                if (len(data) * len(expected_cols)) > 0
                else 0.0
            )
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé pour {self.market}: {confidence_drop_rate:.2f} ({null_count} valeurs nulles)"
                logger.warning(alert_msg)
                AlertManager().send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
            if missing_cols:
                warning_msg = f"Colonnes manquantes pour {self.market}: {missing_cols}"
                logger.warning(warning_msg)
                AlertManager().send_alert(warning_msg, priority=2)
                send_telegram_alert(warning_msg)
                for col in missing_cols:
                    data[col] = 0.0

            if data["timestamp"].isna().any():
                raise ValueError(f"NaN dans les timestamps pour {self.market}")
            if not data["timestamp"].is_monotonic_increasing:
                raise ValueError(f"Timestamps non croissants pour {self.market}")

            critical_cols = ["close", "volume", "bid_size_level_2", "ask_size_level_2"]
            for col in critical_cols:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    raise ValueError(
                        f"Colonne {col} non numérique pour {self.market}: {data[col].dtype}"
                    )
                if data[col].isna().any():
                    data[col] = (
                        data[col]
                        .interpolate(method="linear", limit_direction="both")
                        .fillna(0.0)
                    )
                if (
                    col in ["volume", "bid_size_level_2", "ask_size_level_2"]
                    and (data[col] <= 0).any()
                ):
                    warning_msg = f"Valeurs non positives dans {col} pour {self.market}, corrigées à 1e-6"
                    logger.warning(warning_msg)
                    AlertManager().send_alert(warning_msg, priority=4)
                    data[col] = data[col].clip(lower=1e-6)

            self.save_snapshot(
                "validate_data",
                {
                    "num_columns": len(data.columns),
                    "missing_columns": missing_cols,
                    "confidence_drop_rate": confidence_drop_rate,
                },
            )
            self.log_performance(
                "validate_data", time.time() - start_time, success=True
            )
        except Exception as e:
            error_msg = f"Erreur validation données pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            AlertManager().send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance(
                "validate_data", time.time() - start_time, success=False, error=str(e)
            )
            raise

    def analyze_patterns(
        self, window_size: int = 10, threshold_z_score: float = 2.0
    ) -> Dict[str, List[Dict]]:
        """
        Analyse les patterns récurrents dans les données historiques.

        Args:
            window_size (int): Taille de la fenêtre pour détecter les séquences.
            threshold_z_score (float): Seuil pour identifier les anomalies (en écarts-types).

        Returns:
            Dict[str, List[Dict]]: Dictionnaire contenant les patterns détectés :
                - price_spike: Séquences de pics de prix anormaux
                - volume_spike: Séquences de pics de volume anormaux

        Raises:
            ValueError: Si la base de données est vide ou inaccessible.
        """
        start_time = time.time()
        try:
            cache_key = f"{self.market}_{window_size}_{threshold_z_score}_{hash(str(self.db_path))}"
            if cache_key in pattern_cache:
                patterns = pattern_cache[cache_key]
                pattern_cache.move_to_end(cache_key)
                return patterns
            while len(pattern_cache) > MAX_CACHE_SIZE:
                pattern_cache.popitem(last=False)

            conn = sqlite3.connect(self.db_path)
            feature_sets_path = BASE_DIR / "config/feature_sets.yaml"
            features_config = get_config(feature_sets_path)
            feature_cols = features_config.get("training", {}).get("features", [])[:350]
            query_cols = [
                "timestamp",
                "close",
                "volume",
                "vix_close",
                "bid_size_level_2",
                "ask_size_level_2",
            ] + feature_cols
            query = f"SELECT {', '.join(query_cols)} FROM market_data"
            data = pd.read_sql_query(query, conn, parse_dates=["timestamp"])
            conn.close()

            if data.empty:
                error_msg = f"Base de données vide pour {self.market}: aucune donnée à analyser."
                logger.error(error_msg)
                AlertManager().send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            self._validate_data(data)
            logger.info(
                f"Données chargées depuis {self.db_path} : {len(data)} enregistrements pour {self.market}."
            )

            data["price_z_score"] = (data["close"] - data["close"].mean()) / data[
                "close"
            ].std()
            data["volume_z_score"] = (data["volume"] - data["volume"].mean()) / data[
                "volume"
            ].std()

            price_spikes = data[data["price_z_score"].abs() > threshold_z_score]
            price_patterns = []
            for idx in price_spikes.index:
                start_idx = max(0, idx - window_size // 2)
                end_idx = min(len(data), idx + window_size // 2 + 1)
                pattern_data = data.iloc[start_idx:end_idx][
                    ["timestamp", "close", "vix_close"] + feature_cols[:10]
                ]
                price_patterns.append(
                    {
                        "timestamp": data.loc[idx, "timestamp"],
                        "z_score": data.loc[idx, "price_z_score"],
                        "pattern_data": pattern_data.to_dict(orient="records"),
                    }
                )
            logger.info(
                f"Détection des pics de prix terminée : {len(price_patterns)} patterns trouvés pour {self.market}."
            )

            volume_spikes = data[data["volume_z_score"].abs() > threshold_z_score]
            volume_patterns = []
            for idx in volume_spikes.index:
                start_idx = max(0, idx - window_size // 2)
                end_idx = min(len(data), idx + window_size // 2 + 1)
                pattern_data = data.iloc[start_idx:end_idx][
                    ["timestamp", "volume", "bid_size_level_2", "ask_size_level_2"]
                    + feature_cols[:10]
                ]
                volume_patterns.append(
                    {
                        "timestamp": data.loc[idx, "timestamp"],
                        "z_score": data.loc[idx, "volume_z_score"],
                        "pattern_data": pattern_data.to_dict(orient="records"),
                    }
                )
            logger.info(
                f"Détection des pics de volume terminée : {len(volume_patterns)} patterns trouvés pour {self.market}."
            )

            patterns = {"price_spike": price_patterns, "volume_spike": volume_patterns}
            pattern_cache[cache_key] = patterns

            self.save_snapshot(
                "pattern_analysis",
                {
                    "num_price_patterns": len(price_patterns),
                    "num_volume_patterns": len(volume_patterns),
                    "num_rows": len(data),
                },
            )
            self.checkpoint(data, data_type="pattern_analysis")
            self.cloud_backup(data, data_type="pattern_analysis")

            latency = time.time() - start_time
            self.log_performance(
                "analyze_patterns",
                latency,
                success=True,
                num_patterns=len(price_patterns) + len(volume_patterns),
            )
            return patterns

        except Exception as e:
            error_msg = f"Erreur lors de l’analyse des patterns pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            AlertManager().send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance(
                "analyze_patterns",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            raise

    def save_patterns(
        self, patterns: Dict[str, List[Dict]], output_path: Optional[Path] = None
    ):
        """
        Sauvegarde les patterns détectés dans un fichier JSON compressé (facultatif).

        Args:
            patterns (Dict[str, List[Dict]]): Patterns détectés.
            output_path (Path, optional): Chemin du fichier JSON de sortie.
        """
        start_time = time.time()
        try:
            if output_path:
                output_path = (
                    PATTERN_DIR
                    / self.market
                    / output_path.relative_to(output_path.parent)
                    if not output_path.is_absolute()
                    else output_path
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)

                def write_patterns():
                    with gzip.open(f"{output_path}.gz", "wt", encoding="utf-8") as f:
                        json.dump(patterns, f, default=str, indent=2)

                self.with_retries(write_patterns)
                file_size = os.path.getsize(f"{output_path}.gz") / 1024 / 1024
                success_msg = (
                    f"Patterns sauvegardés à : {output_path}.gz pour {self.market}"
                )
                logger.info(success_msg)
                AlertManager().send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)
                self.log_performance(
                    "save_patterns",
                    time.time() - start_time,
                    success=True,
                    file_size_mb=file_size,
                )
        except Exception as e:
            error_msg = f"Erreur lors de la sauvegarde des patterns pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            AlertManager().send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance(
                "save_patterns", time.time() - start_time, success=False, error=str(e)
            )
            raise


def main():
    """
    Exemple d’utilisation pour le débogage.
    """
    try:
        feature_sets_path = BASE_DIR / "config/feature_sets.yaml"
        features_config = get_config(feature_sets_path)
        feature_cols = features_config.get("training", {}).get("features", [])[:350]

        conn = sqlite3.connect(BASE_DIR / "data" / "market_memory.db")
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start="2025-05-13 10:00:00", periods=100, freq="S"
                ).astype(str),
                "close": np.random.uniform(5000, 5100, 100),
                "volume": np.random.randint(1000, 5000, 100),
                "vix_close": np.random.uniform(20, 30, 100),
                "bid_size_level_2": np.random.randint(100, 1000, 100),
                "ask_size_level_2": np.random.randint(100, 1000, 100),
                **{col: np.random.uniform(0, 1, 100) for col in feature_cols},
            }
        )
        data.loc[50:55, "close"] += 100
        data.loc[50:55, "volume"] *= 3
        data.to_sql("market_data", conn, if_exists="replace", index=False)
        conn.close()

        for market in ["ES", "MNQ"]:
            analyzer = PatternAnalyzer(market=market)
            patterns = analyzer.analyze_patterns()
            output_path = PATTERN_DIR / market / "patterns.json"
            analyzer.save_patterns(patterns, output_path)

        print("Patterns analysés et sauvegardés avec succès pour ES et MNQ.")

    except Exception as e:
        error_msg = (
            f"Échec de l’analyse des patterns : {str(e)}\n{traceback.format_exc()}"
        )
        logger.error(error_msg)
        AlertManager().send_alert(error_msg, priority=5)
        send_telegram_alert(error_msg)
        print(error_msg)
        exit(1)


if __name__ == "__main__":
    main()
