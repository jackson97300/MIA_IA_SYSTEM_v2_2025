# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/backtest/backtest_lab.py
# Gère le backtesting et la simulation des stratégies de trading pour MIA_IA_SYSTEM_v2_2025.
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle :
# Simule des stratégies de trading basées sur le signal SGC (SpotGamma Composite) avec backtesting
# complet et incrémental. Intègre des récompenses adaptatives (méthode 5) basées sur news_impact_score.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, pyyaml>=6.0.0,<7.0.0, psutil>=5.9.8,<6.0.0,
#   boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
# - src/features/signal_selector.py
# - src/backtest/simulate_trades.py
# - src/features/context_aware_filter.py (implicite via feature_pipeline)
# - src/features/cross_asset_processor.py (implicite via feature_pipeline)
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/backtest_config.yaml
# - Données historiques (DataFrame avec timestamp, close, 350 features, news_impact_score)
# - data/features/spotgamma_metrics.csv (pour features comme call_wall, put_wall)
# - data/iqfeed/news_data.csv (pour features contextuelles comme news_volume_spike_1m)
#
# Outputs :
# - Résultats dans data/backtest/backtest_results.csv
# - Trades dans data/backtest/backtest_results_trades.csv
# - Logs dans data/logs/backtest_lab.log
# - Logs de performance dans data/logs/backtest_performance.csv
# - Snapshots JSON compressés dans data/cache/backtest/<market>/*.json.gz
# - Sauvegardes incrémentielles dans data/checkpoints/backtest/<market>/*.json.gz
#
# Notes :
# - Intègre récompenses adaptatives (méthode 5) avec news_impact_score.
# - Ajoute logs psutil (CPU, mémoire) dans backtest_performance.csv.
# - Implémente retries exponentiels (max 3, délai 2^attempt).
# - Tests unitaires disponibles dans tests/test_backtest_lab.py.
# - Compatible avec 350 features pour l’entraînement et 150 SHAP features pour l’inférence/fallback.
# - Utilisation exclusive d’IQFeed comme source de données.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des données.

import gzip
import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import boto3
import numpy as np
import pandas as pd
from loguru import logger

from src.backtest.simulate_trades import simulate_trades
from src.features.signal_selector import calculate_sgc
from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import get_config
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.utils.telegram_alert import send_telegram_alert

# Configuration logging
logger.remove()
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOG_DIR = BASE_DIR / "data" / "logs" / "backtest"
CACHE_DIR = BASE_DIR / "data" / "cache" / "backtest"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints" / "backtest"
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "backtest_lab.log", rotation="10 MB", level="INFO", encoding="utf-8"
)
logger.add(lambda msg: print(msg, end=""), level="INFO")

# Constantes
PERFORMANCE_LOG_PATH = LOG_DIR / "backtest_performance.csv"
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # Base pour délai exponentiel (2^attempt)
MAX_CACHE_SIZE = 1000
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB

# Cache global pour les résultats de backtest
backtest_cache = OrderedDict()


def log_performance(
    operation: str,
    latency: float,
    success: bool = True,
    error: str = None,
    market: str = "ES",
    **kwargs,
) -> None:
    """
    Enregistre les performances (CPU, mémoire, latence) dans backtest_performance.csv.

    Args:
        operation (str): Nom de l’opération.
        latency (float): Temps d’exécution en secondes.
        success (bool): Indique si l’opération a réussi.
        error (str): Message d’erreur (si applicable).
        market (str): Marché (ex. : ES, MNQ).
    """
    try:
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
        cpu_percent = psutil.cpu_percent()
        if memory_usage > 1024:
            alert_msg = (
                f"ALERTE: Usage mémoire élevé ({memory_usage:.2f} MB) pour {market}"
            )
            logger.warning(alert_msg)
            miya_alerts(alert_msg, tag="BACKTEST", priority=5)
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
            "market": market,
            **kwargs,
        }
        log_df = pd.DataFrame([log_entry])

        def save_log():
            if not PERFORMANCE_LOG_PATH.exists():
                log_df.to_csv(PERFORMANCE_LOG_PATH, index=False, encoding="utf-8")
            else:
                log_df.to_csv(
                    PERFORMANCE_LOG_PATH,
                    mode="a",
                    header=False,
                    index=False,
                    encoding="utf-8",
                )

        with_retries(save_log, market=market)
        logger.info(f"Performance journalisée pour {operation}. CPU: {cpu_percent}%")
    except Exception as e:
        error_msg = f"Erreur journalisation performance pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        miya_alerts(error_msg, tag="BACKTEST", priority=3)
        AlertManager().send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)


def save_snapshot(
    snapshot_type: str, data: Dict, market: str = "ES", compress: bool = True
) -> None:
    """
    Sauvegarde un instantané JSON des résultats, compressé avec gzip.

    Args:
        snapshot_type (str): Type de snapshot (ex. : backtest, incremental).
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

        with_retries(write_snapshot, market=market)
        save_path = f"{snapshot_path}.gz" if compress else snapshot_path
        file_size = os.path.getsize(save_path) / 1024 / 1024
        if file_size > 1.0:
            alert_msg = f"Snapshot size {file_size:.2f} MB exceeds 1 MB pour {market}"
            logger.warning(alert_msg)
            miya_alerts(alert_msg, tag="BACKTEST", priority=3)
            AlertManager().send_alert(alert_msg, priority=3)
            send_telegram_alert(alert_msg)
        latency = time.time() - start_time
        success_msg = f"Snapshot {snapshot_type} sauvegardé pour {market}: {save_path}"
        logger.info(success_msg)
        miya_speak(success_msg, tag="BACKTEST", voice_profile="calm")
        AlertManager().send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
        log_performance(
            "save_snapshot",
            latency,
            success=True,
            snapshot_size_mb=file_size,
            market=market,
        )
    except Exception as e:
        error_msg = f"Erreur sauvegarde snapshot {snapshot_type} pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        miya_alerts(error_msg, tag="BACKTEST", priority=3)
        AlertManager().send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)
        log_performance("save_snapshot", 0, success=False, error=str(e), market=market)


def checkpoint(
    data: pd.DataFrame, data_type: str = "backtest_state", market: str = "ES"
) -> None:
    """
    Sauvegarde incrémentielle des données toutes les 5 minutes avec versionnage (5 versions).

    Args:
        data (pd.DataFrame): Données à sauvegarder.
        data_type (str): Type de données (ex. : backtest_state).
        market (str): Marché (ex. : ES, MNQ).
    """
    try:
        start_time = time.time()
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
        checkpoint_path = checkpoint_dir / f"backtest_{data_type}_{timestamp}.json.gz"
        checkpoint_versions = []

        def write_checkpoint():
            with gzip.open(checkpoint_path, "wt", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=4)
            data.to_csv(
                checkpoint_path.with_suffix(".csv"), index=False, encoding="utf-8"
            )

        with_retries(write_checkpoint, market=market)
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
        success_msg = f"Checkpoint sauvegardé pour {market}: {checkpoint_path}"
        logger.info(success_msg)
        miya_speak(success_msg, tag="BACKTEST", voice_profile="calm")
        AlertManager().send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
        log_performance(
            "checkpoint",
            latency,
            success=True,
            file_size_mb=file_size,
            num_rows=len(data),
            data_type=data_type,
            market=market,
        )
    except Exception as e:
        error_msg = f"Erreur sauvegarde checkpoint pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        miya_alerts(error_msg, tag="BACKTEST", priority=3)
        AlertManager().send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)
        log_performance(
            "checkpoint",
            0,
            success=False,
            error=str(e),
            data_type=data_type,
            market=market,
        )


def cloud_backup(
    data: pd.DataFrame, data_type: str = "backtest_state", market: str = "ES"
) -> None:
    """
    Sauvegarde distribuée des données vers AWS S3 toutes les 15 minutes.

    Args:
        data (pd.DataFrame): Données à sauvegarder.
        data_type (str): Type de données (ex. : backtest_state).
        market (str): Marché (ex. : ES, MNQ).
    """
    try:
        start_time = time.time()
        config = get_config(str(BASE_DIR / "config/backtest_config.yaml"))
        if not config.get("s3_bucket"):
            warning_msg = (
                f"S3 bucket non configuré, sauvegarde cloud ignorée pour {market}"
            )
            logger.warning(warning_msg)
            miya_alerts(warning_msg, tag="BACKTEST", priority=3)
            AlertManager().send_alert(warning_msg, priority=3)
            send_telegram_alert(warning_msg)
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = (
            f"{config['s3_prefix']}backtest_{data_type}_{market}_{timestamp}.csv.gz"
        )
        temp_path = CHECKPOINT_DIR / market / f"temp_s3_{timestamp}.csv.gz"
        temp_path.parent.mkdir(exist_ok=True)

        def write_temp():
            data.to_csv(temp_path, compression="gzip", index=False, encoding="utf-8")

        with_retries(write_temp, market=market)
        s3_client = boto3.client("s3")

        def upload_s3():
            s3_client.upload_file(str(temp_path), config["s3_bucket"], backup_path)

        with_retries(upload_s3, market=market)
        temp_path.unlink()
        latency = time.time() - start_time
        success_msg = f"Sauvegarde cloud S3 effectuée pour {market}: {backup_path}"
        logger.info(success_msg)
        miya_speak(success_msg, tag="BACKTEST", voice_profile="calm")
        AlertManager().send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
        log_performance(
            "cloud_backup",
            latency,
            success=True,
            num_rows=len(data),
            data_type=data_type,
            market=market,
        )
    except Exception as e:
        error_msg = f"Erreur sauvegarde cloud S3 pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        miya_alerts(error_msg, tag="BACKTEST", priority=3)
        AlertManager().send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)
        log_performance(
            "cloud_backup",
            0,
            success=False,
            error=str(e),
            data_type=data_type,
            market=market,
        )


def with_retries(
    func: callable,
    max_attempts: int = MAX_RETRIES,
    delay_base: float = RETRY_DELAY,
    market: str = "ES",
) -> Optional[Any]:
    """
    Exécute une fonction avec retries exponentiels (max 3, délai 2^attempt secondes).

    Args:
        func (callable): Fonction à exécuter.
        max_attempts (int): Nombre maximum de tentatives.
        delay_base (float): Base pour le délai exponentiel.
        market (str): Marché (ex. : ES, MNQ).

    Returns:
        Optional[Any]: Résultat de la fonction ou None si échec.
    """
    start_time = time.time()
    for attempt in range(max_attempts):
        try:
            result = func()
            latency = time.time() - start_time
            log_performance(
                f"retry_attempt_{attempt+1}",
                latency,
                success=True,
                attempt_number=attempt + 1,
                market=market,
            )
            return result
        except Exception as e:
            if attempt == max_attempts - 1:
                error_msg = f"Échec après {max_attempts} tentatives pour {market}: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                miya_alerts(error_msg, tag="BACKTEST", priority=4)
                AlertManager().send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                log_performance(
                    f"retry_attempt_{attempt+1}",
                    time.time() - start_time,
                    success=False,
                    error=str(e),
                    attempt_number=attempt + 1,
                    market=market,
                )
                return None
            delay = delay_base**attempt
            warning_msg = (
                f"Tentative {attempt+1} échouée pour {market}, retry après {delay}s"
            )
            logger.warning(warning_msg)
            miya_speak(warning_msg, tag="BACKTEST", level="warning")
            AlertManager().send_alert(warning_msg, priority=3)
            send_telegram_alert(warning_msg)
            time.sleep(delay)


def validate_data(data: pd.DataFrame, market: str = "ES") -> bool:
    """
    Valide que les données contiennent les 350 features attendues et news_impact_score.

    Args:
        data (pd.DataFrame): Données à valider.
        market (str): Marché (ex. : ES, MNQ).

    Returns:
        bool: True si valide, False sinon.
    """
    start_time = time.time()
    try:
        feature_sets_path = BASE_DIR / "config/feature_sets.yaml"
        features_config = get_config(feature_sets_path)
        required_cols = [
            "timestamp",
            "close",
            "news_impact_score",
        ] + features_config.get("training", {}).get("features", [])[:350]
        shap_features = features_config.get("inference", {}).get("shap_features", [])[
            :150
        ]

        missing_cols = [col for col in required_cols if col not in data.columns]
        null_count = data[required_cols].isnull().sum().sum()
        confidence_drop_rate = (
            null_count / (len(data) * len(required_cols))
            if (len(data) * len(required_cols)) > 0
            else 0.0
        )
        if confidence_drop_rate > 0.5:
            alert_msg = f"Confidence_drop_rate élevé pour {market}: {confidence_drop_rate:.2f} ({null_count} valeurs nulles)"
            logger.warning(alert_msg)
            miya_alerts(alert_msg, tag="BACKTEST", priority=3)
            AlertManager().send_alert(alert_msg, priority=3)
            send_telegram_alert(alert_msg)
        if missing_cols:
            for col in missing_cols:
                data[col] = 0.5 if col == "news_impact_score" else 0.0
                warning_msg = f"Colonne {col} manquante pour {market}, imputée à {0.5 if col == 'news_impact_score' else 0}"
                logger.warning(warning_msg)
                miya_speak(warning_msg, tag="BACKTEST", level="warning")
                AlertManager().send_alert(warning_msg, priority=2)
                send_telegram_alert(warning_msg)

        if data["timestamp"].isna().any():
            error_msg = f"NaN dans les timestamps pour {market}"
            logger.error(error_msg)
            miya_alerts(error_msg, tag="BACKTEST", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            return False
        if not data["timestamp"].is_monotonic_increasing:
            error_msg = f"Timestamps non croissants pour {market}"
            logger.error(error_msg)
            miya_alerts(error_msg, tag="BACKTEST", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            return False

        critical_cols = [
            "bid_size_level_1",
            "ask_size_level_1",
            "trade_frequency_1s",
            "close",
            "volume",
            "news_impact_score",
        ]
        for col in critical_cols:
            if col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    error_msg = (
                        f"Colonne {col} non numérique pour {market}: {data[col].dtype}"
                    )
                    logger.error(error_msg)
                    miya_alerts(error_msg, tag="BACKTEST", priority=4)
                    AlertManager().send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    return False
                non_scalar = data[col].apply(
                    lambda x: isinstance(x, (list, dict, tuple))
                )
                if non_scalar.any():
                    error_msg = f"Colonne {col} contient des valeurs non scalaires pour {market}: {data[col][non_scalar].head().tolist()}"
                    logger.error(error_msg)
                    miya_alerts(error_msg, tag="BACKTEST", priority=4)
                    AlertManager().send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    return False
                if data[col].isna().any():
                    data[col] = (
                        data[col]
                        .interpolate(method="linear", limit_direction="both")
                        .fillna(
                            0.5 if col == "news_impact_score" else data[col].median()
                        )
                    )
                if col != "news_impact_score" and (data[col] <= 0).any():
                    warning_msg = f"Valeurs non positives dans {col} pour {market}, corrigées à 1e-6"
                    logger.warning(warning_msg)
                    miya_alerts(warning_msg, tag="BACKTEST", priority=4)
                    data[col] = data[col].clip(lower=1e-6)

        if len(shap_features) != 150:
            error_msg = (
                f"Attendu 150 SHAP features pour {market}, trouvé {len(shap_features)}"
            )
            logger.error(error_msg)
            miya_alerts(error_msg, tag="BACKTEST", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            return False
        missing_shap = [col for col in shap_features if col not in data.columns]
        if missing_shap:
            error_msg = f"SHAP features manquantes pour {market}: {missing_shap}"
            logger.error(error_msg)
            miya_alerts(error_msg, tag="BACKTEST", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            return False

        save_snapshot(
            "validate_data",
            {
                "num_columns": len(data.columns),
                "missing_columns": missing_cols,
                "confidence_drop_rate": confidence_drop_rate,
                "market": market,
            },
            market=market,
        )
        log_performance(
            "validate_data", time.time() - start_time, success=True, market=market
        )
        return True
    except Exception as e:
        error_msg = f"Erreur validation données pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        miya_alerts(error_msg, tag="BACKTEST", priority=4)
        AlertManager().send_alert(error_msg, priority=4)
        send_telegram_alert(error_msg)
        log_performance(
            "validate_data",
            time.time() - start_time,
            success=False,
            error=str(e),
            market=market,
        )
        return False


def load_config(
    config_path: str = str(BASE_DIR / "config/backtest_config.yaml"), market: str = "ES"
) -> Dict[str, Any]:
    """
    Charge la configuration depuis backtest_config.yaml via config_manager.

    Args:
        config_path (str): Chemin vers le fichier de configuration.
        market (str): Marché (ex. : ES, MNQ).

    Returns:
        Dict[str, Any]: Configuration chargée.
    """
    start_time = time.time()
    try:

        def load():
            config = get_config(os.path.basename(config_path))
            return config.get("backtest", {})

        config = with_retries(load, market=market)
        if config is None:
            config = {
                "initial_capital": 100000,
                "position_size_pct": 0.1,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.05,
                "transaction_cost_bps": 5,
                "slippage_pct": 0.001,
                "min_rows": 100,
                "sgc_threshold": 0.7,
            }
            warning_msg = f"Configuration par défaut utilisée pour {market}"
            logger.warning(warning_msg)
            miya_alerts(warning_msg, tag="BACKTEST", priority=3)
            AlertManager().send_alert(warning_msg, priority=3)
            send_telegram_alert(warning_msg)
        success_msg = f"Configuration {config_path} chargée pour {market}"
        logger.info(success_msg)
        miya_speak(success_msg, tag="BACKTEST", voice_profile="calm")
        AlertManager().send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
        log_performance(
            "load_config", time.time() - start_time, success=True, market=market
        )
        return config
    except Exception as e:
        error_msg = f"Erreur chargement config pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        miya_alerts(error_msg, tag="BACKTEST", priority=4)
        AlertManager().send_alert(error_msg, priority=4)
        send_telegram_alert(error_msg)
        log_performance(
            "load_config",
            time.time() - start_time,
            success=False,
            error=str(e),
            market=market,
        )
        return {
            "initial_capital": 100000,
            "position_size_pct": 0.1,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.05,
            "transaction_cost_bps": 5,
            "slippage_pct": 0.001,
            "min_rows": 100,
            "sgc_threshold": 0.7,
        }


def run_backtest(
    data: pd.DataFrame,
    config_path: str = str(BASE_DIR / "config/backtest_config.yaml"),
    output_path: str = str(BASE_DIR / "data/backtest/backtest_results.csv"),
    market: str = "ES",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Exécute un backtest sur les données historiques avec une stratégie basée sur le SGC.

    Args:
        data (pd.DataFrame): Données contenant timestamp, close, 350 features, news_impact_score.
        config_path (str): Chemin vers la configuration.
        output_path (str): Chemin pour sauvegarder les résultats.
        market (str): Marché (ex. : ES, MNQ).

    Returns:
        Tuple[pd.DataFrame, Dict[str, float]]: DataFrame des résultats et métriques de performance.
    """
    start_time = time.time()
    try:
        cache_key = f"{market}_{hash(str(data))}"
        if cache_key in backtest_cache:
            results, metrics = backtest_cache[cache_key]
            backtest_cache.move_to_end(cache_key)
            return results, metrics
        while len(backtest_cache) > MAX_CACHE_SIZE:
            backtest_cache.popitem(last=False)

        config = load_config(config_path, market=market)
        initial_capital = config.get("initial_capital", 100000)
        position_size_pct = config.get("position_size_pct", 0.1)
        stop_loss_pct = config.get("stop_loss_pct", 0.02)
        take_profit_pct = config.get("take_profit_pct", 0.05)
        transaction_cost_bps = config.get("transaction_cost_bps", 5) / 10000
        slippage_pct = config.get("slippage_pct", 0.001)
        min_rows = config.get("min_rows", 100)
        sgc_threshold = config.get("sgc_threshold", 0.7)

        if data.empty or len(data) < min_rows:
            error_msg = f"DataFrame vide ou insuffisant pour {market} ({len(data)} < {min_rows})"
            logger.error(error_msg)
            miya_alerts(error_msg, tag="BACKTEST", priority=4)
            raise ValueError(error_msg)

        # Validation des données
        if not validate_data(data, market=market):
            error_msg = f"Données invalides pour {market}"
            logger.error(error_msg)
            miya_alerts(error_msg, tag="BACKTEST", priority=4)
            raise ValueError(error_msg)

        # Appel à simulate_trades
        trades = simulate_trades(data)
        simulated_trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        # Simulation interne
        capital = initial_capital
        position = 0
        entry_price = 0
        portfolio_value = []
        trades = []
        equity_curve = []

        # Calcul du SGC
        sgc, _ = calculate_sgc(data)
        signals = sgc > sgc_threshold

        # Simulation du trading
        for i in range(1, len(data)):
            price = data["close"].iloc[i]
            news_impact = data["news_impact_score"].iloc[i]
            signal = signals.iloc[i]

            # Gestion de la position
            if position > 0:
                if price <= entry_price * (
                    1 - stop_loss_pct
                ) or price >= entry_price * (1 + take_profit_pct):
                    exit_price = price * (1 - slippage_pct)
                    proceeds = position * exit_price * (1 - transaction_cost_bps)
                    profit = (proceeds - position * entry_price) * (
                        1 + news_impact
                    )  # Récompense adaptative
                    capital += proceeds
                    trades.append(
                        {
                            "entry_time": data["timestamp"].iloc[i - 1],
                            "exit_time": data["timestamp"].iloc[i],
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "profit": profit,
                            "type": "long",
                            "news_impact_score": news_impact,
                        }
                    )
                    position = 0
                    success_msg = f"Position fermée à {exit_price:.2f}, capital: {capital:.2f} pour {market}"
                    logger.debug(success_msg)
                    miya_speak(success_msg, tag="BACKTEST", voice_profile="calm")
                    AlertManager().send_alert(success_msg, priority=1)
                    send_telegram_alert(success_msg)

            # Nouvelle position si signal d'achat
            if signal and position == 0:
                position_size = capital * position_size_pct
                position = position_size / (price * (1 + slippage_pct))
                entry_price = price * (1 + slippage_pct)
                capital -= position_size * (1 + transaction_cost_bps)
                success_msg = f"Position ouverte à {entry_price:.2f}, capital: {capital:.2f} pour {market}"
                logger.debug(success_msg)
                miya_speak(success_msg, tag="BACKTEST", voice_profile="calm")
                AlertManager().send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)

            # Mise à jour de la valeur du portefeuille
            portfolio_value.append(capital + position * price)
            equity_curve.append(
                {
                    "timestamp": data["timestamp"].iloc[i],
                    "equity": capital + position * price,
                    "capital": capital,
                    "position": position,
                    "price": price,
                }
            )

        # Fermeture de la dernière position
        if position > 0:
            exit_price = data["close"].iloc[-1] * (1 - slippage_pct)
            proceeds = position * exit_price * (1 - transaction_cost_bps)
            profit = (proceeds - position * entry_price) * (
                1 + data["news_impact_score"].iloc[-1]
            )  # Récompense adaptative
            capital += proceeds
            trades.append(
                {
                    "entry_time": data["timestamp"].iloc[-2],
                    "exit_time": data["timestamp"].iloc[-1],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "profit": profit,
                    "type": "long",
                    "news_impact_score": data["news_impact_score"].iloc[-1],
                }
            )

        # Calcul des métriques
        equity_series = pd.Series([x["equity"] for x in equity_curve])
        returns = equity_series.pct_change().fillna(0)
        total_return = (equity_series.iloc[-1] - initial_capital) / initial_capital
        annualized_return = ((1 + total_return) ** (252 / len(data))) - 1
        sharpe_ratio = (
            returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        )
        max_drawdown = (
            (equity_series.cummax() - equity_series) / equity_series.cummax()
        ).max()
        win_rate = (
            len([t for t in trades if t["profit"] > 0]) / len(trades) if trades else 0
        )

        metrics = {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "num_trades": len(trades),
        }

        # Sauvegarde des résultats
        results_df = pd.DataFrame(equity_curve)
        trades_df = pd.DataFrame(trades)

        def save_results():
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            results_df.to_csv(output_path, index=False, encoding="utf-8")
            trades_df.to_csv(
                output_path.replace(".csv", "_trades.csv"),
                index=False,
                encoding="utf-8",
            )

        with_retries(save_results, market=market)

        # Fusion avec trades simulés
        results = (
            pd.concat([results_df, simulated_trades_df], axis=0, ignore_index=True)
            if not simulated_trades_df.empty
            else results_df
        )

        # Mise en cache
        backtest_cache[cache_key] = (results, metrics)

        # Sauvegarde et alertes
        save_snapshot(
            "backtest",
            {"metrics": metrics, "num_rows": len(data), "num_trades": len(trades)},
            market=market,
        )
        checkpoint(results, data_type="backtest_results", market=market)
        cloud_backup(results, data_type="backtest_results", market=market)

        success_msg = (
            f"Backtest terminé pour {market}: Total Return: {total_return:.2%}, Sharpe: {sharpe_ratio:.2f}, "
            f"Max Drawdown: {max_drawdown:.2%}, Win Rate: {win_rate:.2%}, Trades: {len(trades)}"
        )
        logger.info(success_msg)
        miya_speak(success_msg, tag="BACKTEST", voice_profile="calm")
        AlertManager().send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
        log_performance(
            "run_backtest", time.time() - start_time, success=True, market=market
        )
        return results, metrics

    except Exception as e:
        error_msg = f"Erreur dans run_backtest pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        miya_alerts(error_msg, tag="BACKTEST", priority=4)
        AlertManager().send_alert(error_msg, priority=4)
        send_telegram_alert(error_msg)
        log_performance(
            "run_backtest",
            time.time() - start_time,
            success=False,
            error=str(e),
            market=market,
        )
        return pd.DataFrame(), {
            "total_return": 0,
            "annualized_return": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "win_rate": 0,
            "num_trades": 0,
        }


def run_incremental_backtest(
    row: pd.Series,
    buffer: pd.DataFrame,
    state: Dict[str, Any],
    config_path: str = str(BASE_DIR / "config/backtest_config.yaml"),
    market: str = "ES",
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Exécute un backtest incrémental pour une seule ligne en temps réel.

    Args:
        row (pd.Series): Ligne contenant timestamp, close, 350 features, news_impact_score.
        buffer (pd.DataFrame): Buffer des données précédentes.
        state (Dict[str, Any]): État du backtest (capital, position, etc.).
        config_path (str): Chemin vers la configuration.
        market (str): Marché (ex. : ES, MNQ).

    Returns:
        Tuple[Dict[str, float], Dict[str, Any]]: Métriques de performance et état mis à jour.
    """
    start_time = time.time()
    try:
        config = load_config(config_path, market=market)
        position_size_pct = config.get("position_size_pct", 0.1)
        stop_loss_pct = config.get("stop_loss_pct", 0.02)
        take_profit_pct = config.get("take_profit_pct", 0.05)
        transaction_cost_bps = config.get("transaction_cost_bps", 5) / 10000
        slippage_pct = config.get("slippage_pct", 0.001)
        sgc_threshold = config.get("sgc_threshold", 0.7)

        # Validation des données
        row_df = row.to_frame().T
        if not validate_data(row_df, market=market):
            error_msg = f"Données invalides pour backtest incrémental pour {market}"
            logger.error(error_msg)
            miya_alerts(error_msg, tag="BACKTEST", priority=4)
            raise ValueError(error_msg)

        # Initialisation de l’état
        if not state:
            state = {
                "capital": config.get("initial_capital", 100000),
                "position": 0,
                "entry_price": 0,
                "equity_curve": [],
                "trades": [],
            }

        capital = state["capital"]
        position = state["position"]
        entry_price = state["entry_price"]
        price = row["close"]
        news_impact = row["news_impact_score"]

        # Calcul du SGC
        buffer = pd.concat([buffer, row_df], ignore_index=True)
        sgc, _ = calculate_sgc(buffer)
        signal = sgc.iloc[-1] > sgc_threshold if not sgc.empty else False

        # Gestion de la position
        if position > 0:
            if price <= entry_price * (1 - stop_loss_pct) or price >= entry_price * (
                1 + take_profit_pct
            ):
                exit_price = price * (1 - slippage_pct)
                proceeds = position * exit_price * (1 - transaction_cost_bps)
                profit = (proceeds - position * entry_price) * (
                    1 + news_impact
                )  # Récompense adaptative
                capital += proceeds
                state["trades"].append(
                    {
                        "entry_time": (
                            buffer["timestamp"].iloc[-2]
                            if len(buffer) > 1
                            else row["timestamp"]
                        ),
                        "exit_time": row["timestamp"],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "profit": profit,
                        "type": "long",
                        "news_impact_score": news_impact,
                    }
                )
                position = 0
                state["entry_price"] = 0
                success_msg = f"Position fermée à {exit_price:.2f}, capital: {capital:.2f} pour {market}"
                logger.debug(success_msg)
                miya_speak(success_msg, tag="BACKTEST", voice_profile="calm")
                AlertManager().send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)

        # Nouvelle position si signal d'achat
        if signal and position == 0:
            position_size = capital * position_size_pct
            position = position_size / (price * (1 + slippage_pct))
            entry_price = price * (1 + slippage_pct)
            capital -= position_size * (1 + transaction_cost_bps)
            state["entry_price"] = entry_price
            success_msg = f"Position ouverte à {entry_price:.2f}, capital: {capital:.2f} pour {market}"
            logger.debug(success_msg)
            miya_speak(success_msg, tag="BACKTEST", voice_profile="calm")
            AlertManager().send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)

        # Mise à jour de l’état
        state["capital"] = capital
        state["position"] = position
        equity = capital + position * price
        state["equity_curve"].append(
            {
                "timestamp": row["timestamp"],
                "equity": equity,
                "capital": capital,
                "position": position,
                "price": price,
            }
        )

        # Calcul des métriques
        equity_series = pd.Series([x["equity"] for x in state["equity_curve"]])
        returns = equity_series.pct_change().fillna(0)
        total_return = (
            equity_series.iloc[-1] - config.get("initial_capital", 100000)
        ) / config.get("initial_capital", 100000)
        sharpe_ratio = (
            returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        )
        max_drawdown = (
            ((equity_series.cummax() - equity_series) / equity_series.cummax()).max()
            if len(equity_series) > 1
            else 0
        )
        win_rate = (
            len([t for t in state["trades"] if t["profit"] > 0]) / len(state["trades"])
            if state["trades"]
            else 0
        )

        metrics = {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "num_trades": len(state["trades"]),
        }

        # Sauvegarde
        save_snapshot(
            "incremental_backtest",
            {"metrics": metrics, "num_trades": len(state["trades"]), "market": market},
            market=market,
        )
        checkpoint(
            pd.DataFrame(state["equity_curve"]),
            data_type="incremental_equity_curve",
            market=market,
        )
        cloud_backup(
            pd.DataFrame(state["equity_curve"]),
            data_type="incremental_equity_curve",
            market=market,
        )

        log_performance(
            "run_incremental_backtest",
            time.time() - start_time,
            success=True,
            market=market,
        )
        return metrics, state

    except Exception as e:
        error_msg = f"Erreur dans run_incremental_backtest pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        miya_alerts(error_msg, tag="BACKTEST", priority=4)
        AlertManager().send_alert(error_msg, priority=4)
        send_telegram_alert(error_msg)
        log_performance(
            "run_incremental_backtest",
            time.time() - start_time,
            success=False,
            error=str(e),
            market=market,
        )
        return {
            "total_return": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "win_rate": 0,
            "num_trades": 0,
        }, state


if __name__ == "__main__":
    try:
        # Test avec données simulées
        feature_sets_path = BASE_DIR / "config/feature_sets.yaml"
        features_config = get_config(feature_sets_path)
        feature_cols = features_config.get("training", {}).get("features", [])[:350]
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-05-13 09:00", periods=1000, freq="1min"
                ),
                "close": np.random.normal(5100, 10, 1000),
                "news_impact_score": np.random.uniform(0, 1, 1000),
                **{col: np.random.uniform(0, 1, 1000) for col in feature_cols},
            }
        )
        data.to_csv(
            BASE_DIR / "data/features/features_latest_filtered.csv", index=False
        )

        for market in ["ES", "MNQ"]:
            results_df, metrics = run_backtest(data, market=market)
            print(f"Métriques du backtest pour {market}:")
            print(metrics)
            print(f"Résultats (premières 5 lignes) pour {market}:")
            print(results_df.head())
        success_msg = "Test run_backtest terminé pour ES et MNQ"
        logger.info(success_msg)
        miya_speak(success_msg, tag="BACKTEST", voice_profile="calm")
        AlertManager().send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
    except Exception as e:
        error_msg = f"Erreur test principal: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        miya_alerts(error_msg, tag="BACKTEST", priority=4)
        AlertManager().send_alert(error_msg, priority=4)
        send_telegram_alert(error_msg)
        raise
