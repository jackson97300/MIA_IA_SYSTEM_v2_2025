# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/model/utils/trading_utils.py
# Utilitaires de trading pour MIA_IA_SYSTEM_v2_2025, incluant la détection de régime, l'ajustement du risque/levier,
# et le calcul du profit (méthode 5 : récompenses adaptatives).
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle :
# Fournit des fonctions utilitaires pour le trading, supportant la méthode 5 (récompenses adaptatives) via le calcul
# du profit, la détection de régime, et l’ajustement du risque/levier. Compatible avec 350 features (entraînement)
# et 150 SHAP features (inférence), aligné avec feature_sets.yaml. Utilise IQFeed exclusivement via data_provider.py.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, pyyaml>=6.0.0,<7.0.0, psutil>=5.9.8,<6.0.0, pickle,
#   boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0
# - src/model/utils/miya_console.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/router/detect_regime.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/es_config.yaml (paramètres de trading)
# - Données de trading (pd.DataFrame avec 350 features pour l’entraînement, 150 SHAP features pour l’inférence)
# - Modèles ML (pickle files dans model/ml_models/)
#
# Outputs :
# - Logs dans data/logs/trading_utils.log
# - Logs de performance dans data/logs/trading_utils_performance.csv
# - Snapshots JSON compressés dans data/cache/trading_utils/<market>/*.json.gz
# - Sauvegardes incrémentielles dans data/checkpoints/trading_utils/<market>/*.json.gz
#
# Notes :
# - Supprime toutes les références à 320 features, 81 features, et obs_t.
# - Implémente retries (max 3, délai 2^attempt secondes) pour les appels critiques.
# - Intègre la méthode 5 via calculate_profit pour les récompenses adaptatives.
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des opérations.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.
# - Tests unitaires disponibles dans tests/test_trading_utils.py.
# - TODO : Intégration future avec API Bloomberg (juin 2025).

import gzip
import json
import pickle
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import boto3
import numpy as np
import pandas as pd
import psutil
from loguru import logger

from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import get_config
from src.model.utils.miya_console import MiyaConsole
from src.utils.telegram_alert import send_telegram_alert

# Configuration logging
logger.remove()
BASE_DIR = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
LOG_DIR = BASE_DIR / "data" / "logs"
CACHE_DIR = BASE_DIR / "data" / "cache" / "trading_utils"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints" / "trading_utils"
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "trading_utils.log", rotation="10 MB", level="INFO", encoding="utf-8"
)
logger.add(lambda msg: print(msg, end=""), level="INFO")

# Constantes
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Secondes, exponentiel
PERF_LOG_PATH = LOG_DIR / "trading_utils_performance.csv"
MAX_CACHE_SIZE = 1000
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB

# Cache global pour les modèles ML et les résultats
MODEL_CACHE = {}
FUNCTION_CACHE = OrderedDict()


def with_retries(
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
            log_performance(
                f"retry_attempt_{attempt+1}",
                latency,
                success=True,
                attempt_number=attempt + 1,
            )
            return result
        except Exception as e:
            if attempt == max_attempts - 1:
                latency = time.time() - start_time
                error_msg = f"Échec après {max_attempts} tentatives: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                AlertManager(market="ES").send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                log_performance(
                    f"retry_attempt_{attempt+1}",
                    latency,
                    success=False,
                    error=str(e),
                    attempt_number=attempt + 1,
                )
                return None
            delay = delay_base**attempt
            logger.warning(f"Tentative {attempt+1} échouée, retry après {delay}s")
            time.sleep(delay)


def log_performance(
    operation: str,
    latency: float,
    success: bool = True,
    error: str = None,
    market: str = "ES",
    **kwargs,
) -> None:
    """
    Enregistre les performances (CPU, mémoire, latence) dans trading_utils_performance.csv.

    Args:
        operation (str): Nom de l’opération.
        latency (float): Temps d’exécution en secondes.
        success (bool): Indique si l’opération a réussi.
        error (str, optional): Message d’erreur si applicable.
        market (str): Marché (ex. : ES, MNQ).
        **kwargs: Paramètres supplémentaires.
    """
    cache_key = f"{market}_{operation}_{hash(str(latency))}_{hash(str(error))}"
    if cache_key in FUNCTION_CACHE:
        return
    while len(FUNCTION_CACHE) > MAX_CACHE_SIZE:
        FUNCTION_CACHE.popitem(last=False)

    try:
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
        cpu_percent = psutil.cpu_percent()
        confidence_drop_rate = 1.0 if success else 0.0  # Simplifié pour Phase 8
        if memory_usage > 1024:
            alert_msg = (
                f"ALERTE: Usage mémoire élevé ({memory_usage:.2f} MB) pour {market}"
            )
            logger.warning(alert_msg)
            AlertManager(market=market).send_alert(alert_msg, priority=5)
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
            "market": market,
            **kwargs,
        }
        log_df = pd.DataFrame([log_entry])

        def save_log():
            if not PERF_LOG_PATH.exists():
                log_df.to_csv(PERF_LOG_PATH, index=False, encoding="utf-8")
            else:
                log_df.to_csv(
                    PERF_LOG_PATH, mode="a", header=False, index=False, encoding="utf-8"
                )

        with_retries(save_log)
        success_msg = f"Performance journalisée pour {operation}. CPU: {cpu_percent}%"
        logger.info(success_msg)
        send_telegram_alert(success_msg)
        save_snapshot("log_performance", log_entry, market=market)
        FUNCTION_CACHE[cache_key] = True
    except Exception as e:
        error_msg = f"Erreur journalisation performance pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        AlertManager(market=market).send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)


def save_snapshot(
    snapshot_type: str, data: Dict, market: str = "ES", compress: bool = True
) -> None:
    """
    Sauvegarde un instantané JSON des résultats, compressé avec gzip.

    Args:
        snapshot_type (str): Type de snapshot (ex. : calculate_profit).
        data (Dict): Données à sauvegarder.
        market (str): Marché (ex. : ES, MNQ).
        compress (bool): Compresser avec gzip (défaut : True).
    """
    start_time = time.time()
    snapshot_dir = CACHE_DIR / market
    snapshot_dir.mkdir(exist_ok=True)
    try:
        if not os.access(snapshot_dir, os.W_OK):
            error_msg = f"Permission d’écriture refusée pour {snapshot_dir}"
            logger.error(error_msg)
            AlertManager(market=market).send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot = {
            "timestamp": timestamp,
            "type": snapshot_type,
            "market": market,
            "data": data,
        }
        snapshot_path = snapshot_dir / f"snapshot_{snapshot_type}_{timestamp}.json"

        def write_snapshot():
            if compress:
                with gzip.open(f"{snapshot_path}.gz", "wt", encoding="utf-8") as f:
                    json.dump(snapshot, f, indent=4)
            else:
                with open(snapshot_path, "w", encoding="utf-8") as f:
                    json.dump(snapshot, f, indent=4)

        with_retries(write_snapshot)
        save_path = f"{snapshot_path}.gz" if compress else snapshot_path
        file_size = os.path.getsize(save_path) / 1024 / 1024
        if file_size > 1.0:
            alert_msg = f"Snapshot size {file_size:.2f} MB exceeds 1 MB pour {market}"
            logger.warning(alert_msg)
            AlertManager(market=market).send_alert(alert_msg, priority=3)
            send_telegram_alert(alert_msg)
        latency = time.time() - start_time
        success_msg = f"Snapshot {snapshot_type} sauvegardé pour {market}: {save_path}"
        logger.info(success_msg)
        send_telegram_alert(success_msg)
        log_performance(
            "save_snapshot",
            latency,
            success=True,
            market=market,
            snapshot_size_mb=file_size,
        )
    except Exception as e:
        error_msg = f"Erreur sauvegarde snapshot {snapshot_type} pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        AlertManager(market=market).send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)
        log_performance("save_snapshot", 0, success=False, error=str(e), market=market)


def checkpoint(
    data: pd.DataFrame, data_type: str = "trading_utils_state", market: str = "ES"
) -> None:
    """
    Sauvegarde incrémentielle des données toutes les 5 minutes avec versionnage (5 versions).

    Args:
        data (pd.DataFrame): Données à sauvegarder.
        data_type (str): Type de données (ex. : trading_utils_state).
        market (str): Marché (ex. : ES, MNQ).
    """
    start_time = time.time()
    checkpoint_dir = CHECKPOINT_DIR / market
    checkpoint_dir.mkdir(exist_ok=True)
    try:
        if not os.access(checkpoint_dir, os.W_OK):
            error_msg = f"Permission d’écriture refusée pour {checkpoint_dir}"
            logger.error(error_msg)
            AlertManager(market=market).send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_data = {
            "timestamp": timestamp,
            "num_rows": len(data),
            "columns": list(data.columns),
            "data_type": data_type,  # pylint: disable=line-too-long
            "market": market,
        }
        checkpoint_path = (
            checkpoint_dir / f"trading_utils_{data_type}_{timestamp}.json.gz"
        )
        checkpoint_versions = []

        def write_checkpoint():
            with gzip.open(checkpoint_path, "wt", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=4)
            data.to_csv(
                checkpoint_path.with_suffix(".csv"), index=False, encoding="utf-8"
            )

        with_retries(write_checkpoint)
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
        AlertManager(market=market).send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
        log_performance(
            "checkpoint",
            latency,
            success=True,
            market=market,
            file_size_mb=file_size,
            num_rows=len(data),
            data_type=data_type,
        )
    except Exception as e:
        error_msg = f"Erreur sauvegarde checkpoint pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        AlertManager(market=market).send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)
        log_performance(
            "checkpoint",
            0,
            success=False,
            error=str(e),
            market=market,
            data_type=data_type,
        )


def cloud_backup(
    data: pd.DataFrame, data_type: str = "trading_utils_state", market: str = "ES"
) -> None:
    """
    Sauvegarde distribuée des données vers AWS S3 toutes les 15 minutes.

    Args:
        data (pd.DataFrame): Données à sauvegarder.
        data_type (str): Type de données (ex. : trading_utils_state).
        market (str): Marché (ex. : ES, MNQ).
    """
    start_time = time.time()
    checkpoint_dir = CHECKPOINT_DIR / market
    checkpoint_dir.mkdir(exist_ok=True)
    try:
        config = get_config(str(BASE_DIR / "config/es_config.yaml"))
        if not config.get("s3_bucket"):
            warning_msg = (
                f"S3 bucket non configuré, sauvegarde cloud ignorée pour {market}"
            )
            logger.warning(warning_msg)
            AlertManager(market=market).send_alert(warning_msg, priority=3)
            send_telegram_alert(warning_msg)
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{config['s3_prefix']}trading_utils_{data_type}_{market}_{timestamp}.csv.gz"
        temp_path = checkpoint_dir / f"temp_s3_{timestamp}.csv.gz"

        def write_temp():
            data.to_csv(temp_path, compression="gzip", index=False, encoding="utf-8")

        with_retries(write_temp)
        s3_client = boto3.client("s3")

        def upload_s3():
            s3_client.upload_file(str(temp_path), config["s3_bucket"], backup_path)

        with_retries(upload_s3)
        temp_path.unlink()
        latency = time.time() - start_time
        success_msg = f"Sauvegarde cloud S3 effectuée pour {market}: {backup_path}"
        logger.info(success_msg)
        AlertManager(market=market).send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
        log_performance(
            "cloud_backup",
            latency,
            success=True,
            market=market,
            num_rows=len(data),
            data_type=data_type,
        )
    except Exception as e:
        error_msg = f"Erreur sauvegarde cloud S3 pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        AlertManager(market=market).send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)
        log_performance(
            "cloud_backup",
            0,
            success=False,
            error=str(e),
            market=market,
            data_type=data_type,
        )


def load_config_manager(
    config_path: str = str(BASE_DIR / "config" / "es_config.yaml"),
) -> Dict[str, Any]:
    """
    Charge la configuration via config_manager.

    Args:
        config_path (str): Chemin vers le fichier de configuration.

    Returns:
        Dict[str, Any]: Configuration chargée.
    """
    start_time = time.time()
    try:

        def load():
            config = get_config(config_path).get("trading_utils", {})
            return config

        config = with_retries(load)
        if not config:
            config = {
                "max_leverage": 5.0,
                "risk_factor": 1.0,
                "range_pred_threshold": 0.7,
                "order_flow_pred_threshold": 0.6,
                "atr_threshold": 2.0,
                "adx_threshold": 20,
                "batch_size": 1000,
                "transaction_cost": 2.0,
                "news_impact_threshold": 0.5,
                "vix_threshold": 20.0,
            }
            logger.warning(f"Configuration par défaut utilisée pour {config_path}")
        MiyaConsole(market="ES").miya_speak(
            f"Configuration chargée via config_manager: {config_path}",
            tag="TRADING_UTILS",
        )
        log_performance("load_config_manager", time.time() - start_time, success=True)
        save_snapshot("load_config_manager", {"config_path": config_path})
        return config
    except Exception as e:
        error_msg = f"Erreur chargement config via config_manager: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        MiyaConsole(market="ES").miya_alerts(
            error_msg, tag="TRADING_UTILS", voice_profile="urgent"
        )
        send_telegram_alert(error_msg)
        log_performance(
            "load_config_manager", time.time() - start_time, success=False, error=str(e)
        )
        return {
            "max_leverage": 5.0,
            "risk_factor": 1.0,
            "range_pred_threshold": 0.7,
            "order_flow_pred_threshold": 0.6,
            "atr_threshold": 2.0,
            "adx_threshold": 20,
            "batch_size": 1000,
            "transaction_cost": 2.0,
            "news_impact_threshold": 0.5,
            "vix_threshold": 20.0,
        }


def load_model_cached(path: str, market: str = "ES") -> Optional[Any]:
    """
    Charge un modèle ML depuis un fichier pickle avec mise en cache.

    Args:
        path (str): Chemin vers le fichier du modèle.
        market (str): Marché (ex. : ES, MNQ).

    Returns:
        Optional[Any]: Modèle chargé ou None si introuvable.
    """
    start_time = time.time()
    cache_key = f"{market}_{path}"
    try:
        if cache_key in MODEL_CACHE:
            log_performance(
                "load_model_cached",
                time.time() - start_time,
                success=True,
                source="cache",
                market=market,
            )
            return MODEL_CACHE[cache_key]
        if os.path.exists(path):

            def load_model():
                with open(path, "rb") as f:
                    return pickle.load(f)

            model = with_retries(load_model)
            if model:
                MODEL_CACHE[cache_key] = model
                MiyaConsole(market=market).miya_speak(
                    f"Modèle chargé et mis en cache: {path}", tag="TRADING_UTILS"
                )
                log_performance(
                    "load_model_cached",
                    time.time() - start_time,
                    success=True,
                    source="file",
                    market=market,
                )
                return model
        error_msg = f"Modèle non trouvé: {path}"
        logger.error(error_msg)
        MiyaConsole(market=market).miya_alerts(
            error_msg, tag="TRADING_UTILS", voice_profile="urgent"
        )
        send_telegram_alert(error_msg)
        log_performance(
            "load_model_cached",
            time.time() - start_time,
            success=False,
            error="Modèle non trouvé",
            market=market,
        )
        return None
    except Exception as e:
        error_msg = f"Erreur chargement modèle {path} pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        MiyaConsole(market=market).miya_alerts(
            error_msg, tag="TRADING_UTILS", voice_profile="urgent"
        )
        send_telegram_alert(error_msg)
        log_performance(
            "load_model_cached",
            time.time() - start_time,
            success=False,
            error=str(e),
            market=market,
        )
        return None


def detect_market_regime(
    data: pd.DataFrame, config_path: Optional[str] = None, market: str = "ES"
) -> pd.Series:
    """
    Détecte le régime de marché (trend, range, défensif) en utilisant MarketRegimeDetector.

    Args:
        data (pd.DataFrame): DataFrame contenant les données (ex. atr_14, adx_14, delta_volume).
        config_path (str, optional): Chemin vers le fichier de configuration.
        market (str): Marché (ex. : ES, MNQ).

    Returns:
        pd.Series: Série contenant les régimes ("trend", "range", "defensive").
    """
    start_time = time.time()
    cache_key = f"{market}_detect_market_regime_{hash(str(data))}"
    if cache_key in FUNCTION_CACHE:
        return FUNCTION_CACHE[cache_key]
    try:
        config = load_config_manager(config_path)
        batch_size = config.get("batch_size", 1000)
        required_cols = ["atr_14", "adx_14", "delta_volume"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            error_msg = f"Colonnes manquantes pour {market}: {missing_cols}"
            logger.error(error_msg)
            MiyaConsole(market=market).miya_alerts(
                error_msg, tag="TRADING_UTILS", voice_profile="urgent"
            )
            send_telegram_alert(error_msg)
            raise ValueError(error_msg)

        from src.model.router.detect_regime import MarketRegimeDetector

        detector = MarketRegimeDetector(
            config_path=(
                config_path
                if config_path
                else str(BASE_DIR / "config/router_config.yaml")
            )
        )
        regimes = pd.Series(index=data.index, dtype=str)

        for start_idx in range(0, len(data), batch_size):
            end_idx = min(start_idx + batch_size, len(data))
            batch_data = data.iloc[start_idx:end_idx]
            for i in batch_data.index:
                regime, _ = detector.detect(data, data.index.get_loc(i))
                regimes[i] = regime

        MiyaConsole(market=market).miya_speak(
            f"Régime de marché détecté pour {market}: {regimes.value_counts().to_dict()}",
            tag="TRADING_UTILS",
        )
        log_performance(
            "detect_market_regime",
            time.time() - start_time,
            success=True,
            num_rows=len(data),
            market=market,
        )
        save_snapshot(
            "detect_market_regime",
            {"num_rows": len(data), "regime_counts": regimes.value_counts().to_dict()},
            market=market,
        )
        FUNCTION_CACHE[cache_key] = regimes
        checkpoint(
            pd.DataFrame({"regimes": regimes, "timestamp": data.index}),
            data_type="detect_market_regime",
            market=market,
        )
        cloud_backup(
            pd.DataFrame({"regimes": regimes, "timestamp": data.index}),
            data_type="detect_market_regime",
            market=market,
        )
        return regimes

    except Exception as e:
        error_msg = f"Erreur dans detect_market_regime pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        MiyaConsole(market=market).miya_alerts(
            error_msg, tag="TRADING_UTILS", voice_profile="urgent"
        )
        send_telegram_alert(error_msg)
        log_performance(
            "detect_market_regime",
            time.time() - start_time,
            success=False,
            error=str(e),
            market=market,
        )
        return pd.Series("defensive", index=data.index)


def incremental_detect_market_regime(
    row: pd.Series,
    buffer: pd.DataFrame,
    config_path: Optional[str] = None,
    market: str = "ES",
) -> str:
    """
    Détecte le régime de marché pour une seule ligne en temps réel.

    Args:
        row (pd.Series): Ligne contenant les données (ex. atr_14, adx_14, delta_volume).
        buffer (pd.DataFrame): Buffer des données précédentes.
        config_path (str, optional): Chemin vers le fichier de configuration.
        market (str): Marché (ex. : ES, MNQ).

    Returns:
        str: Régime de marché ("trend", "range", "defensive").
    """
    start_time = time.time()
    cache_key = f"{market}_incremental_detect_market_regime_{hash(str(row))}_{hash(str(buffer))}"
    if cache_key in FUNCTION_CACHE:
        return FUNCTION_CACHE[cache_key]
    try:
        buffer = pd.concat([buffer, row.to_frame().T], ignore_index=True)
        required_cols = ["atr_14", "adx_14", "delta_volume"]
        missing_cols = [col for col in required_cols if col not in buffer.columns]
        if missing_cols:
            error_msg = f"Colonnes manquantes pour {market}: {missing_cols}"
            logger.error(error_msg)
            MiyaConsole(market=market).miya_alerts(
                error_msg, tag="TRADING_UTILS", voice_profile="urgent"
            )
            send_telegram_alert(error_msg)
            raise ValueError(error_msg)

        from src.model.router.detect_regime import MarketRegimeDetector

        detector = MarketRegimeDetector(
            config_path=(
                config_path
                if config_path
                else str(BASE_DIR / "config/router_config.yaml")
            )
        )
        regime, _ = detector.detect(buffer, len(buffer) - 1)
        MiyaConsole(market=market).miya_speak(
            f"Régime détecté pour step {len(buffer) - 1} pour {market}: {regime}",
            tag="TRADING_UTILS",
        )
        log_performance(
            "incremental_detect_market_regime",
            time.time() - start_time,
            success=True,
            market=market,
        )
        save_snapshot(
            "incremental_detect_market_regime",
            {"regime": regime, "step": len(buffer) - 1},
            market=market,
        )
        FUNCTION_CACHE[cache_key] = regime
        checkpoint(
            pd.DataFrame([{"regime": regime, "timestamp": row.name}]),
            data_type="incremental_detect_market_regime",
            market=market,
        )
        cloud_backup(
            pd.DataFrame([{"regime": regime, "timestamp": row.name}]),
            data_type="incremental_detect_market_regime",
            market=market,
        )
        return regime

    except Exception as e:
        error_msg = f"Erreur dans incremental_detect_market_regime pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        MiyaConsole(market=market).miya_alerts(
            error_msg, tag="TRADING_UTILS", voice_profile="urgent"
        )
        send_telegram_alert(error_msg)
        log_performance(
            "incremental_detect_market_regime",
            time.time() - start_time,
            success=False,
            error=str(e),
            market=market,
        )
        return "defensive"


def adjust_risk_and_leverage(
    data: pd.DataFrame,
    regime: pd.Series,
    config_path: Optional[str] = None,
    market: str = "ES",
) -> pd.DataFrame:
    """
    Ajuste le risque et le levier en fonction du régime de marché.

    Args:
        data (pd.DataFrame): DataFrame contenant les données (ex. atr_14, adx_14).
        regime (pd.Series): Série des régimes de marché ("trend", "range", "defensive").
        config_path (str, optional): Chemin vers le fichier de configuration.
        market (str): Marché (ex. : ES, MNQ).

    Returns:
        pd.DataFrame: DataFrame enrichi avec "adjusted_leverage" et "adjusted_risk".
    """
    start_time = time.time()
    cache_key = (
        f"{market}_adjust_risk_and_leverage_{hash(str(data))}_{hash(str(regime))}"
    )
    if cache_key in FUNCTION_CACHE:
        return FUNCTION_CACHE[cache_key]
    try:
        config = load_config_manager(config_path)
        max_leverage = config.get("max_leverage", 5.0)
        risk_factor = config.get("risk_factor", 1.0)

        required_cols = ["atr_14", "adx_14"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            error_msg = f"Colonnes manquantes pour {market}: {missing_cols}"
            logger.error(error_msg)
            MiyaConsole(market=market).miya_alerts(
                error_msg, tag="TRADING_UTILS", voice_profile="urgent"
            )
            send_telegram_alert(error_msg)
            raise ValueError(error_msg)

        data["adjusted_leverage"] = 1.0
        data["adjusted_risk"] = risk_factor

        trend_mask = regime == "trend"
        range_mask = regime == "range"
        defensive_mask = regime == "defensive"

        data.loc[trend_mask, "adjusted_leverage"] = np.minimum(
            max_leverage, 3.0 + (data.loc[trend_mask, "adx_14"] / 50)
        )
        data.loc[trend_mask, "adjusted_risk"] = risk_factor * 1.5
        data.loc[range_mask, "adjusted_leverage"] = 2.0
        data.loc[range_mask, "adjusted_risk"] = risk_factor * 0.8
        data.loc[defensive_mask, "adjusted_leverage"] = 1.0
        data.loc[defensive_mask, "adjusted_risk"] = risk_factor * 0.5

        MiyaConsole(market=market).miya_speak(
            f"Levier ajusté pour {market}: min={data['adjusted_leverage'].min():.2f}, max={data['adjusted_leverage'].max():.2f}",
            tag="TRADING_UTILS",
        )
        log_performance(
            "adjust_risk_and_leverage",
            time.time() - start_time,
            success=True,
            num_rows=len(data),
            market=market,
        )
        save_snapshot(
            "adjust_risk_and_leverage",
            {
                "min_leverage": data["adjusted_leverage"].min(),
                "max_leverage": data["adjusted_leverage"].max(),
                "num_rows": len(data),
            },
            market=market,
        )
        FUNCTION_CACHE[cache_key] = data
        checkpoint(
            data[["adjusted_leverage", "adjusted_risk"]],
            data_type="adjust_risk_and_leverage",
            market=market,
        )
        cloud_backup(
            data[["adjusted_leverage", "adjusted_risk"]],
            data_type="adjust_risk_and_leverage",
            market=market,
        )
        return data

    except Exception as e:
        error_msg = f"Erreur dans adjust_risk_and_leverage pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        MiyaConsole(market=market).miya_alerts(
            error_msg, tag="TRADING_UTILS", voice_profile="urgent"
        )
        send_telegram_alert(error_msg)
        log_performance(
            "adjust_risk_and_leverage",
            time.time() - start_time,
            success=False,
            error=str(e),
            market=market,
        )
        data["adjusted_leverage"] = 1.0
        data["adjusted_risk"] = 1.0
        return data


def incremental_adjust_risk_and_leverage(
    row: pd.Series, regime: str, config_path: Optional[str] = None, market: str = "ES"
) -> Tuple[float, float]:
    """
    Ajuste le risque et le levier pour une seule ligne en temps réel.

    Args:
        row (pd.Series): Ligne contenant les données (ex. atr_14, adx_14).
        regime (str): Régime de marché ("trend", "range", "defensive").
        config_path (str, optional): Chemin vers le fichier de configuration.
        market (str): Marché (ex. : ES, MNQ).

    Returns:
        Tuple[float, float]: Levier ajusté et risque ajusté.
    """
    start_time = time.time()
    cache_key = (
        f"{market}_incremental_adjust_risk_and_leverage_{hash(str(row))}_{regime}"
    )
    if cache_key in FUNCTION_CACHE:
        return FUNCTION_CACHE[cache_key]
    try:
        config = load_config_manager(config_path)
        max_leverage = config.get("max_leverage", 5.0)
        risk_factor = config.get("risk_factor", 1.0)

        required_cols = ["atr_14", "adx_14"]
        missing_cols = [col for col in required_cols if col not in row]
        if missing_cols:
            error_msg = f"Colonnes manquantes pour {market}: {missing_cols}"
            logger.error(error_msg)
            MiyaConsole(market=market).miya_alerts(
                error_msg, tag="TRADING_UTILS", voice_profile="urgent"
            )
            send_telegram_alert(error_msg)
            raise ValueError(error_msg)

        leverage = 1.0
        risk = risk_factor

        if regime == "trend":
            leverage = min(max_leverage, 3.0 + (row["adx_14"] / 50))
            risk = risk_factor * 1.5
        elif regime == "range":
            leverage = 2.0
            risk = risk_factor * 0.8
        elif regime == "defensive":
            leverage = 1.0
            risk = risk_factor * 0.5

        MiyaConsole(market=market).miya_speak(
            f"Levier ajusté pour {market}: {leverage:.2f}, Risque ajusté: {risk:.2f}",
            tag="TRADING_UTILS",
        )
        log_performance(
            "incremental_adjust_risk_and_leverage",
            time.time() - start_time,
            success=True,
            market=market,
        )
        save_snapshot(
            "incremental_adjust_risk_and_leverage",
            {"leverage": leverage, "risk": risk},
            market=market,
        )
        FUNCTION_CACHE[cache_key] = (leverage, risk)
        checkpoint(
            pd.DataFrame([{"leverage": leverage, "risk": risk, "timestamp": row.name}]),
            data_type="incremental_adjust_risk_and_leverage",
            market=market,
        )
        cloud_backup(
            pd.DataFrame([{"leverage": leverage, "risk": risk, "timestamp": row.name}]),
            data_type="incremental_adjust_risk_and_leverage",
            market=market,
        )
        return leverage, risk

    except Exception as e:
        error_msg = f"Erreur dans incremental_adjust_risk_and_leverage pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        MiyaConsole(market=market).miya_alerts(
            error_msg, tag="TRADING_UTILS", voice_profile="urgent"
        )
        send_telegram_alert(error_msg)
        log_performance(
            "incremental_adjust_risk_and_leverage",
            time.time() - start_time,
            success=False,
            error=str(e),
            market=market,
        )
        return 1.0, 1.0


def validate_trade_entry_combined(
    env: Any,
    range_model_path: str = str(BASE_DIR / "model/ml_models/range_filter.pkl"),
    order_flow_model_path: str = str(
        BASE_DIR / "model/ml_models/order_flow_filter.pkl"
    ),
    sequence_length: int = 50,
    config_path: Optional[str] = None,
    debug: bool = False,
    market: str = "ES",
) -> bool:
    """
    Valide l'entrée en trade en combinant des filtres basés sur le range et le flux d'ordres.

    Args:
        env: Environnement de trading (TradingEnv) contenant les données actuelles.
        range_model_path (str): Chemin vers le modèle de filtre range.
        order_flow_model_path (str): Chemin vers le modèle de filtre de flux d'ordres.
        sequence_length (int): Longueur de la séquence pour les prédictions.
        config_path (str, optional): Chemin vers le fichier de configuration.
        debug (bool): Si True, affiche les détails des rejets.
        market (str): Marché (ex. : ES, MNQ).

    Returns:
        bool: True si l'entrée en trade est validée, False sinon.
    """
    start_time = time.time()
    cache_key = f"{market}_validate_trade_entry_combined_{hash(str(env.data))}_{env.current_step}"
    if cache_key in FUNCTION_CACHE:
        return FUNCTION_CACHE[cache_key]
    try:
        config = load_config_manager(config_path)
        range_pred_threshold = config.get("range_pred_threshold", 0.7)
        order_flow_pred_threshold = config.get("order_flow_pred_threshold", 0.6)
        atr_threshold = config.get("atr_threshold", 2.0)
        adx_threshold = config.get("adx_threshold", 20)

        if not hasattr(env, "data") or env.data is None or env.current_step is None:
            error_msg = f"Données ou current_step non définis pour {market}"
            logger.error(error_msg)
            MiyaConsole(market=market).miya_alerts(
                error_msg, tag="TRADING_UTILS", voice_profile="urgent"
            )
            send_telegram_alert(error_msg)
            if debug:
                print("[DEBUG] Rejeté: Données ou current_step non définis")
            log_performance(
                "validate_trade_entry_combined",
                time.time() - start_time,
                success=False,
                error="Données ou current_step non définis",
                market=market,
            )
            return False

        current_step = env.current_step
        if current_step < sequence_length:
            warning_msg = f"Step {current_step} trop bas pour séquence {sequence_length} pour {market}"
            logger.warning(warning_msg)
            MiyaConsole(market=market).miya_speak(
                warning_msg, tag="TRADING_UTILS", level="warning"
            )
            send_telegram_alert(warning_msg)
            if debug:
                print(f"[DEBUG] Rejeté: Step {current_step} < {sequence_length}")
            log_performance(
                "validate_trade_entry_combined",
                time.time() - start_time,
                success=False,
                error="Step trop bas",
                market=market,
            )
            return False

        required_cols = [
            "atr_14",
            "adx_14",
            "vwap",
            "delta_volume",
            "bid_size_level_1",
            "ask_size_level_1",
        ]
        missing_cols = [col for col in required_cols if col not in env.data.columns]
        if missing_cols:
            error_msg = f"Colonnes manquantes pour {market}: {missing_cols}"
            logger.error(error_msg)
            MiyaConsole(market=market).miya_alerts(
                error_msg, tag="TRADING_UTILS", voice_profile="urgent"
            )
            send_telegram_alert(error_msg)
            if debug:
                print(f"[DEBUG] Rejeté: Colonnes manquantes {missing_cols}")
            log_performance(
                "validate_trade_entry_combined",
                time.time() - start_time,
                success=False,
                error=f"Colonnes manquantes: {missing_cols}",
                market=market,
            )
            return False

        for col in required_cols:
            env.data[col] = (
                pd.to_numeric(env.data[col], errors="coerce")
                .interpolate(method="linear")
                .fillna(method="bfill")
                .fillna(method="ffill")
            )
            if env.data[col].isna().any():
                error_msg = (
                    f"NaN persistants dans {col} après interpolation pour {market}"
                )
                logger.error(error_msg)
                MiyaConsole(market=market).miya_alerts(
                    error_msg, tag="TRADING_UTILS", priority=4
                )
                send_telegram_alert(error_msg)
                if debug:
                    print(f"[DEBUG] Rejeté: NaN persistants dans {col}")
                log_performance(
                    "validate_trade_entry_combined",
                    time.time() - start_time,
                    success=False,
                    error=f"NaN persistants dans {col}",
                    market=market,
                )
                return False
            if (env.data[col] <= 0).any():
                env.data[col] = env.data[col].clip(lower=1e-6)

        window_start = max(0, current_step - sequence_length + 1)
        window_end = current_step + 1
        data_window = env.data.iloc[window_start:window_end]
        if len(data_window) < sequence_length:
            warning_msg = (
                f"Fenêtre de données trop courte pour {market}: {len(data_window)}"
            )
            logger.warning(warning_msg)
            MiyaConsole(market=market).miya_speak(
                warning_msg, tag="TRADING_UTILS", level="warning"
            )
            send_telegram_alert(warning_msg)
            if debug:
                print(f"[DEBUG] Rejeté: Fenêtre trop courte ({len(data_window)})")
            log_performance(
                "validate_trade_entry_combined",
                time.time() - start_time,
                success=False,
                error="Fenêtre trop courte",
                market=market,
            )
            return False

        range_model = load_model_cached(range_model_path, market=market)
        order_flow_model = load_model_cached(order_flow_model_path, market=market)

        range_pred = None
        if range_model:
            range_features = (
                data_window[["atr_14", "adx_14", "vwap"]].iloc[-1].values.reshape(1, -1)
            )
            range_pred = range_model.predict_proba(range_features)[:, 1][0]
            if range_pred > range_pred_threshold:
                warning_msg = f"Trade rejeté par filtre range pour {market}: prob={range_pred:.2f}"
                logger.warning(warning_msg)
                MiyaConsole(market=market).miya_speak(
                    warning_msg, tag="TRADING_UTILS", level="warning"
                )
                send_telegram_alert(warning_msg)
                if debug:
                    print(
                        f"[DEBUG] Rejeté: Range_pred={range_pred:.2f} > {range_pred_threshold}"
                    )
                log_performance(
                    "validate_trade_entry_combined",
                    time.time() - start_time,
                    success=False,
                    error="Rejeté par filtre range",
                    market=market,
                )
                return False

        order_flow_pred = None
        if order_flow_model:
            order_flow_features = (
                data_window[["delta_volume", "bid_size_level_1", "ask_size_level_1"]]
                .iloc[-1]
                .values.reshape(1, -1)
            )
            order_flow_pred = order_flow_model.predict_proba(order_flow_features)[:, 1][
                0
            ]
            if order_flow_pred < order_flow_pred_threshold:
                warning_msg = f"Trade rejeté par filtre order flow pour {market}: prob={order_flow_pred:.2f}"
                logger.warning(warning_msg)
                MiyaConsole(market=market).miya_speak(
                    warning_msg, tag="TRADING_UTILS", level="warning"
                )
                send_telegram_alert(warning_msg)
                if debug:
                    print(
                        f"[DEBUG] Rejeté: Order_flow_pred={order_flow_pred:.2f} < {order_flow_pred_threshold}"
                    )
                log_performance(
                    "validate_trade_entry_combined",
                    time.time() - start_time,
                    success=False,
                    error="Rejeté par filtre order flow",
                    market=market,
                )
                return False

        atr_current = data_window["atr_14"].iloc[-1]
        adx_current = data_window["adx_14"].iloc[-1]
        if atr_current > atr_threshold or adx_current < adx_threshold:
            warning_msg = f"Trade rejeté pour {market}: ATR={atr_current:.2f}, ADX={adx_current:.2f}"
            logger.warning(warning_msg)
            MiyaConsole(market=market).miya_speak(
                warning_msg, tag="TRADING_UTILS", level="warning"
            )
            send_telegram_alert(warning_msg)
            if debug:
                print(
                    f"[DEBUG] Rejeté: ATR={atr_current:.2f} > {atr_threshold}, ADX={adx_current:.2f} < {adx_threshold}"
                )
            log_performance(
                "validate_trade_entry_combined",
                time.time() - start_time,
                success=False,
                error="Rejeté par ATR/ADX",
                market=market,
            )
            return False

        MiyaConsole(market=market).miya_speak(
            f"Entrée en trade validée au step {current_step} pour {market}",
            tag="TRADING_UTILS",
        )
        log_performance(
            "validate_trade_entry_combined",
            time.time() - start_time,
            success=True,
            step=current_step,
            market=market,
        )
        save_snapshot(
            "validate_trade_entry_combined",
            {
                "step": current_step,
                "range_pred": range_pred if range_pred is not None else "N/A",
                "order_flow_pred": (
                    order_flow_pred if order_flow_pred is not None else "N/A"
                ),
                "atr": atr_current,
                "adx": adx_current,
            },
            market=market,
        )
        FUNCTION_CACHE[cache_key] = True
        checkpoint(
            pd.DataFrame(
                [
                    {
                        "step": current_step,
                        "range_pred": range_pred,
                        "order_flow_pred": order_flow_pred,
                        "atr": atr_current,
                        "adx": adx_current,
                    }
                ]
            ),
            data_type="validate_trade_entry_combined",
            market=market,
        )
        cloud_backup(
            pd.DataFrame(
                [
                    {
                        "step": current_step,
                        "range_pred": range_pred,
                        "order_flow_pred": order_flow_pred,
                        "atr": atr_current,
                        "adx": adx_current,
                    }
                ]
            ),
            data_type="validate_trade_entry_combined",
            market=market,
        )
        if debug:
            print(
                f"[DEBUG] Validé: Range_pred={range_pred or 'N/A':.2f}, Order_flow_pred={order_flow_pred or 'N/A':.2f}, "
                f"ATR={atr_current:.2f}, ADX={adx_current:.2f}"
            )
        return True

    except Exception as e:
        error_msg = f"Erreur dans validate_trade_entry_combined pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        MiyaConsole(market=market).miya_alerts(
            error_msg, tag="TRADING_UTILS", voice_profile="urgent"
        )
        send_telegram_alert(error_msg)
        log_performance(
            "validate_trade_entry_combined",
            time.time() - start_time,
            success=False,
            error=str(e),
            market=market,
        )
        if debug:
            print(f"[DEBUG] Rejeté: Erreur {str(e)}")
        return False


def calculate_profit(
    trade: Dict[str, Any], config_path: Optional[str] = None, market: str = "ES"
) -> float:
    """
    Calcule le profit d’un trade pour la méthode 5 (récompenses adaptatives).

    Args:
        trade (Dict[str, Any]): Dictionnaire contenant les détails du trade (entry_price, exit_price, position_size, trade_type).
        config_path (str, optional): Chemin vers le fichier de configuration.
        market (str): Marché (ex. : ES, MNQ).

    Returns:
        float: Profit net du trade (en USD).
    """
    start_time = time.time()
    cache_key = f"{market}_calculate_profit_{hash(str(trade))}"
    if cache_key in FUNCTION_CACHE:
        return FUNCTION_CACHE[cache_key]
    try:
        config = load_config_manager(config_path)
        transaction_cost = config.get("transaction_cost", 2.0)

        required_keys = ["entry_price", "exit_price", "position_size", "trade_type"]
        missing_keys = [key for key in required_keys if key not in trade]
        if missing_keys:
            error_msg = f"Clés manquantes dans trade pour {market}: {missing_keys}"
            logger.error(error_msg)
            MiyaConsole(market=market).miya_alerts(
                error_msg, tag="TRADING_UTILS", voice_profile="urgent"
            )
            send_telegram_alert(error_msg)
            raise ValueError(error_msg)

        entry_price = float(trade["entry_price"])
        exit_price = float(trade["exit_price"])
        position_size = float(trade["position_size"])
        trade_type = trade["trade_type"].lower()

        if trade_type not in ["long", "short"]:
            error_msg = f"Type de trade invalide pour {market}: {trade_type}"
            logger.error(error_msg)
            MiyaConsole(market=market).miya_alerts(
                error_msg, tag="TRADING_UTILS", voice_profile="urgent"
            )
            send_telegram_alert(error_msg)
            raise ValueError(error_msg)

        if trade_type == "long":
            profit = (exit_price - entry_price) * position_size
        else:  # short
            profit = (entry_price - exit_price) * position_size

        profit -= transaction_cost * abs(position_size)

        news_impact = trade.get("news_impact_score", 0.0)
        vix = trade.get("vix", 20.0)
        news_threshold = config.get("news_impact_threshold", 0.5)
        vix_threshold = config.get("vix_threshold", 20.0)

        if news_impact > news_threshold:
            profit *= 1.2  # Bonus pour impact des nouvelles
        elif news_impact < -news_threshold:
            profit *= 0.8  # Pénalité pour impact négatif
        if vix > vix_threshold:
            profit *= 0.9  # Pénalité pour volatilité élevée

        MiyaConsole(market=market).miya_speak(
            f"Profit calculé pour {market}: {profit:.2f} USD pour trade {trade_type}",
            tag="TRADING_UTILS",
        )
        log_performance(
            "calculate_profit",
            time.time() - start_time,
            success=True,
            profit=profit,
            market=market,
        )
        save_snapshot(
            "calculate_profit",
            {
                "trade_type": trade_type,
                "profit": profit,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "position_size": position_size,
                "news_impact": news_impact,
                "vix": vix,
            },
            market=market,
        )
        FUNCTION_CACHE[cache_key] = profit
        checkpoint(
            pd.DataFrame(
                [
                    {
                        "trade_type": trade_type,
                        "profit": profit,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "position_size": position_size,
                        "news_impact": news_impact,
                        "vix": vix,
                    }
                ]
            ),
            data_type="calculate_profit",
            market=market,
        )
        cloud_backup(
            pd.DataFrame(
                [
                    {
                        "trade_type": trade_type,
                        "profit": profit,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "position_size": position_size,
                        "news_impact": news_impact,
                        "vix": vix,
                    }
                ]
            ),
            data_type="calculate_profit",
            market=market,
        )
        return profit

    except Exception as e:
        error_msg = f"Erreur dans calculate_profit pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        MiyaConsole(market=market).miya_alerts(
            error_msg, tag="TRADING_UTILS", voice_profile="urgent"
        )
        send_telegram_alert(error_msg)
        log_performance(
            "calculate_profit",
            time.time() - start_time,
            success=False,
            error=str(e),
            market=market,
        )
        return 0.0


if __name__ == "__main__":
    try:
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "atr_14": np.random.uniform(0.5, 2.0, 100),
                "adx_14": np.random.uniform(10, 50, 100),
                "close": np.random.uniform(4000, 4500, 100),
                "delta_volume": np.random.uniform(-1000, 1000, 100),
                "vwap": np.random.uniform(4000, 4500, 100),
                "bid_size_level_1": np.random.randint(50, 500, 100),
                "ask_size_level_1": np.random.randint(50, 500, 100),
            }
        )
        data.set_index("timestamp", inplace=True)

        regimes = detect_market_regime(data, market="ES")
        print("Régimes détectés:")
        print(regimes.value_counts())

        data = adjust_risk_and_leverage(data, regimes, market="ES")
        print("\nDonnées avec levier et risque ajustés:")
        print(data[["atr_14", "adx_14", "adjusted_leverage", "adjusted_risk"]].head())

        class MockEnv:
            def __init__(self):
                self.data = data
                self.current_step = 60

        env = MockEnv()
        is_valid = validate_trade_entry_combined(env, debug=True, market="ES")
        print(f"\nEntrée en trade validée: {is_valid}")

        trade = {
            "entry_price": 4000.0,
            "exit_price": 4050.0,
            "position_size": 10,
            "trade_type": "long",
            "news_impact_score": 0.6,
            "vix": 18.0,
        }
        profit = calculate_profit(trade, market="ES")
        print(f"\nProfit du trade: {profit:.2f} USD")

        MiyaConsole(market="ES").miya_speak(
            "Test trading_utils terminé", tag="TRADING_UTILS", voice_profile="calm"
        )
    except Exception as e:
        error_msg = f"Erreur test principal: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        MiyaConsole(market="ES").miya_alerts(
            error_msg, tag="TRADING_UTILS", voice_profile="urgent"
        )
        send_telegram_alert(error_msg)
        raise
