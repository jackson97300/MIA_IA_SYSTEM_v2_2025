# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/model/utils/standard.py
# Rôle : Centralise les fonctions communes (retries, logging psutil, alertes) (Phase 15).
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, psutil>=5.9.8,<6.0.0, boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
#
# Outputs :
# - Logs dans data/logs/standard_utils.log
# - Logs de performance dans data/logs/standard_utils_performance.csv
# - Snapshots JSON compressés dans data/cache/standard/<market>/*.json.gz
# - Sauvegardes incrémentielles dans data/checkpoints/standard/<market>/*.json.gz
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Intègre la Phase 15 (centralisation des fonctions communes).
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des métriques.
# - Tests unitaires disponibles dans tests/test_standard.py.

import functools
import gzip
import json
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import boto3
import pandas as pd
import psutil
from loguru import logger

from src.model.utils.alert_manager import AlertManager
from src.utils.telegram_alert import send_telegram_alert

# Configuration logging
logger.remove()
BASE_DIR = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
LOG_DIR = BASE_DIR / "data" / "logs"
CACHE_DIR = BASE_DIR / "data" / "cache" / "standard"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints" / "standard"
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "standard_utils.log", rotation="10 MB", level="INFO", encoding="utf-8"
)
logger.add(lambda msg: print(msg, end=""), level="INFO")

# Constantes
PERF_LOG_PATH = LOG_DIR / "standard_utils_performance.csv"
MAX_RETRIES = 3
RETRY_DELAY = 1.0
MAX_CACHE_SIZE = 1000
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB

# Cache global pour les résultats de log_performance
performance_cache = OrderedDict()


def with_retries(
    max_attempts: int = MAX_RETRIES,
    delay: float = RETRY_DELAY,
    exceptions: tuple = (Exception,),
) -> Callable:
    """
    Décorateur pour réessayer une fonction en cas d'échec.

    Args:
        max_attempts (int): Nombre maximum de tentatives.
        delay (float): Délai entre les tentatives (en secondes).
        exceptions (tuple): Exceptions à capturer pour réessayer.

    Returns:
        Callable: Fonction décorée avec gestion des retries.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, market: str = "ES", **kwargs) -> Any:
            start_time = time.time()
            attempts = 0
            last_exception = None

            while attempts < max_attempts:
                try:
                    result = func(*args, **kwargs)
                    latency = time.time() - start_time
                    log_performance(
                        "retry_success",
                        latency,
                        success=True,
                        operation=func.__name__,
                        attempt_number=attempts + 1,
                        market=market,
                    )
                    logger.info(
                        f"Fonction {func.__name__} exécutée avec succès après {attempts + 1} tentative(s) pour {market}."
                    )
                    return result
                except exceptions as e:
                    attempts += 1
                    last_exception = e
                    warning_msg = f"Échec de {func.__name__} (tentative {attempts}/{max_attempts}) pour {market} : {str(e)}"
                    logger.warning(warning_msg)
                    AlertManager().send_alert(warning_msg, priority=3)
                    send_telegram_alert(warning_msg)
                    if attempts == max_attempts:
                        error_msg = f"Échec définitif de {func.__name__} après {max_attempts} tentatives pour {market} : {str(e)}"
                        logger.error(error_msg)
                        AlertManager().send_alert(error_msg, priority=4)
                        send_telegram_alert(error_msg)
                        log_performance(
                            "retry_failure",
                            time.time() - start_time,
                            success=False,
                            operation=func.__name__,
                            error=str(e),
                            attempt_number=attempts,
                            market=market,
                        )
                        raise last_exception
                    time.sleep(delay * (2 ** (attempts - 1)))

            raise last_exception

        return wrapper

    return decorator


def log_performance(
    operation: str,
    latency: float,
    success: bool = True,
    error: str = None,
    market: str = "ES",
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Journalise les performances d'une opération avec les métriques système via psutil.

    Args:
        operation (str): Nom de l'opération (ex. : feature_pipeline, trading_loop).
        latency (float): Temps d’exécution en secondes.
        success (bool): Indique si l’opération a réussi.
        error (str): Message d’erreur (si applicable).
        market (str): Marché (ex. : ES, MNQ).
        extra_metrics (Dict[str, Any], optional): Métriques supplémentaires à journaliser.
    """
    cache_key = f"{market}_{operation}_{hash(str(latency))}_{hash(str(error))}"
    if cache_key in performance_cache:
        return
    while len(performance_cache) > MAX_CACHE_SIZE:
        performance_cache.popitem(last=False)

    try:
        cpu_usage = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()
        memory_used = memory_info.used / (1024**2)  # En Mo
        memory_percent = memory_info.percent
        confidence_drop_rate = 1.0 if success else 0.0  # Simplifié pour Phase 8

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "latency_seconds": latency,
            "success": success,
            "error": error,
            "cpu_percent": cpu_usage,
            "memory_used_mb": memory_used,
            "memory_percent": memory_percent,
            "confidence_drop_rate": confidence_drop_rate,
            "market": market,
        }
        if extra_metrics:
            log_entry.update(extra_metrics)

        log_df = pd.DataFrame([log_entry])

        def save_log():
            PERF_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            if not PERF_LOG_PATH.exists():
                log_df.to_csv(PERF_LOG_PATH, index=False, encoding="utf-8")
            else:
                log_df.to_csv(
                    PERF_LOG_PATH, mode="a", header=False, index=False, encoding="utf-8"
                )

        with_retries()(save_log)(market=market)

        if memory_used > 1024:
            alert_msg = f"ALERTE: Usage mémoire élevé ({memory_used:.2f} MB) pour {market} dans {operation}"
            logger.warning(alert_msg)
            AlertManager().send_alert(alert_msg, priority=5)
            send_telegram_alert(alert_msg)

        success_msg = f"Performance journalisée pour {operation} : durée={latency:.2f}s, CPU={cpu_usage}%, Mémoire={memory_used:.2f}Mo pour {market}"
        logger.info(success_msg)
        performance_cache[cache_key] = True
        save_snapshot("log_performance", log_entry, market=market)
    except Exception as e:
        error_msg = f"Erreur lors de la journalisation des performances pour {operation} dans {market} : {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        AlertManager().send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)


def save_snapshot(
    snapshot_type: str, data: Dict, market: str = "ES", compress: bool = True
) -> None:
    """
    Sauvegarde un instantané JSON des résultats, compressé avec gzip.

    Args:
        snapshot_type (str): Type de snapshot (ex. : log_performance).
        data (Dict): Données à sauvegarder.
        market (str): Marché (ex. : ES, MNQ).
        compress (bool): Compresser avec gzip (défaut : True).
    """
    start_time = time.time()
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot = {
            "timestamp": timestamp,
            "type": snapshot_type,
            "market": market,
            "data": data,
        }
        snapshot_dir = CACHE_DIR / market
        snapshot_dir.mkdir(exist_ok=True)
        snapshot_path = snapshot_dir / f"snapshot_{snapshot_type}_{timestamp}.json"

        def write_snapshot():
            if compress:
                with gzip.open(f"{snapshot_path}.gz", "wt", encoding="utf-8") as f:
                    json.dump(snapshot, f, indent=4)
            else:
                with open(snapshot_path, "w", encoding="utf-8") as f:
                    json.dump(snapshot, f, indent=4)

        with_retries()(write_snapshot)(market=market)
        save_path = f"{snapshot_path}.gz" if compress else snapshot_path
        file_size = os.path.getsize(save_path) / 1024 / 1024
        if file_size > 1.0:
            alert_msg = f"Snapshot size {file_size:.2f} MB exceeds 1 MB pour {market}"
            logger.warning(alert_msg)
            AlertManager().send_alert(alert_msg, priority=3)
            send_telegram_alert(alert_msg)
        success_msg = f"Snapshot {snapshot_type} sauvegardé pour {market}: {save_path}"
        logger.info(success_msg)
        AlertManager().send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
        log_performance(
            "save_snapshot",
            time.time() - start_time,
            success=True,
            snapshot_size_mb=file_size,
            market=market,
        )
    except Exception as e:
        error_msg = f"Erreur sauvegarde snapshot {snapshot_type} pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        AlertManager().send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)
        log_performance(
            "save_snapshot",
            time.time() - start_time,
            success=False,
            error=str(e),
            market=market,
        )


def checkpoint(
    data: pd.DataFrame, data_type: str = "standard_state", market: str = "ES"
) -> None:
    """
    Sauvegarde incrémentielle des données toutes les 5 minutes avec versionnage (5 versions).

    Args:
        data (pd.DataFrame): Données à sauvegarder.
        data_type (str): Type de données (ex. : standard_state).
        market (str): Marché (ex. : ES, MNQ).
    """
    start_time = time.time()
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_data = {
            "timestamp": timestamp,
            "num_rows": len(data),
            "columns": list(data.columns),
            "data_type": data_type,
            "market": market,
        }
        checkpoint_dir = CHECKPOINT_DIR / market
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_path = checkpoint_dir / f"standard_{data_type}_{timestamp}.json.gz"
        checkpoint_versions = []

        def write_checkpoint():
            with gzip.open(checkpoint_path, "wt", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=4)
            data.to_csv(
                checkpoint_path.with_suffix(".csv"), index=False, encoding="utf-8"
            )

        with_retries()(write_checkpoint)(market=market)
        checkpoint_versions.append(checkpoint_path)
        if len(checkpoint_versions) > 5:
            oldest = checkpoint_versions.pop(0)
            if oldest.exists():
                oldest.unlink()
            csv_oldest = oldest.with_suffix(".csv")
            if csv_oldest.exists():
                csv_oldest.unlink()
        file_size = os.path.getsize(checkpoint_path) / 1024 / 1024
        success_msg = f"Checkpoint sauvegardé pour {market}: {checkpoint_path}"
        logger.info(success_msg)
        AlertManager().send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
        log_performance(
            "checkpoint",
            time.time() - start_time,
            success=True,
            file_size_mb=file_size,
            num_rows=len(data),
            data_type=data_type,
            market=market,
        )
    except Exception as e:
        error_msg = f"Erreur sauvegarde checkpoint pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        AlertManager().send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)
        log_performance(
            "checkpoint",
            time.time() - start_time,
            success=False,
            error=str(e),
            data_type=data_type,
            market=market,
        )


def cloud_backup(
    data: pd.DataFrame, data_type: str = "standard_state", market: str = "ES"
) -> None:
    """
    Sauvegarde distribuée des données vers AWS S3 toutes les 15 minutes.

    Args:
        data (pd.DataFrame): Données à sauvegarder.
        data_type (str): Type de données (ex. : standard_state).
        market (str): Marché (ex. : ES, MNQ).
    """
    start_time = time.time()
    try:
        from src.model.utils.config_manager import get_config

        config = get_config(str(BASE_DIR / "config/es_config.yaml"))
        if not config.get("s3_bucket"):
            warning_msg = (
                f"S3 bucket non configuré, sauvegarde cloud ignorée pour {market}"
            )
            logger.warning(warning_msg)
            AlertManager().send_alert(warning_msg, priority=3)
            send_telegram_alert(warning_msg)
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = (
            f"{config['s3_prefix']}standard_{data_type}_{market}_{timestamp}.csv.gz"
        )
        temp_path = CHECKPOINT_DIR / market / f"temp_s3_{timestamp}.csv.gz"
        temp_path.parent.mkdir(exist_ok=True)

        def write_temp():
            data.to_csv(temp_path, compression="gzip", index=False, encoding="utf-8")

        with_retries()(write_temp)(market=market)
        s3_client = boto3.client("s3")

        def upload_s3():
            s3_client.upload_file(str(temp_path), config["s3_bucket"], backup_path)

        with_retries()(upload_s3)(market=market)
        temp_path.unlink()
        success_msg = f"Sauvegarde cloud S3 effectuée pour {market}: {backup_path}"
        logger.info(success_msg)
        AlertManager().send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
        log_performance(
            "cloud_backup",
            time.time() - start_time,
            success=True,
            num_rows=len(data),
            data_type=data_type,
            market=market,
        )
    except Exception as e:
        error_msg = f"Erreur sauvegarde cloud S3 pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        AlertManager().send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)
        log_performance(
            "cloud_backup",
            time.time() - start_time,
            success=False,
            error=str(e),
            data_type=data_type,
            market=market,
        )


def main():
    """
    Exemple d’utilisation pour le débogage.
    """
    try:

        @with_retries(max_attempts=3, delay=0.5, exceptions=(ValueError,))
        def risky_operation(market: str = "ES"):
            import numpy as np

            if np.random.random() > 0.5:
                raise ValueError("Erreur simulée")
            return "Succès"

        start_time = time.time()
        for market in ["ES", "MNQ"]:
            result = risky_operation(market=market)
            log_performance(
                "risky_operation",
                time.time() - start_time,
                success=True,
                operation="risky_operation",
                result=result,
                market=market,
            )
            df = pd.DataFrame(
                {"timestamp": [datetime.now().isoformat()], "result": [result]}
            )
            checkpoint(df, data_type="test_metrics", market=market)
            cloud_backup(df, data_type="test_metrics", market=market)

        print(
            "Exemple d’utilisation des utilitaires terminé avec succès pour ES et MNQ."
        )
    except Exception as e:
        error_msg = f"Échec de l’exemple : {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        AlertManager().send_alert(error_msg, priority=5)
        send_telegram_alert(error_msg)
        print(error_msg)
        exit(1)


if __name__ == "__main__":
    main()
