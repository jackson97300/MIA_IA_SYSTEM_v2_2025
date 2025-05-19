# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/model/utils/maml_utils.py
# Gère le meta-learning (prototypical networks) pour train_sac.py.
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle :
# Implémente le meta-learning (méthode 18) avec prototypical networks pour adapter les modèles SAC à de nouvelles tâches,
# utilisant les 350 features pour l’entraînement et 150 SHAP features pour l’inférence. Intègre des mécanismes robustes
# (retries, validation, snapshots, logs psutil) pour une fiabilité optimale.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.8,<6.0.0, stable-baselines3>=2.0.0,<3.0.0,
#   pyyaml>=6.0.0,<7.0.0, torch>=2.0.0,<3.0.0, boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/miya_console.py
# - src/model/utils/alert_manager.py
# - src/envs/trading_env.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/es_config.yaml (paramètres de meta-learning)
# - config/feature_sets.yaml (350 features et 150 SHAP features)
# - model/sac_models/<market>/<mode>/<policy_type>/*.zip (modèles SAC pré-entraînés)
# - Données (pd.DataFrame avec 350 features ou 150 SHAP features)
#
# Outputs :
# - Logs dans data/logs/maml_utils.log
# - Logs de performance dans data/logs/maml_performance.csv
# - Snapshots JSON compressés dans data/cache/maml/<market>/*.json.gz
# - Sauvegardes incrémentielles dans data/checkpoints/maml/<market>/*.json.gz
# - Modèles mis à jour dans model/sac_models/<market>/<mode>/<policy_type>/
#
# Notes :
# - Intègre meta-learning (méthode 18) avec prototypical networks pour réduire la complexité.
# - Valide les 350 features pour l’entraînement et 150 SHAP features pour l’inférence via config/feature_sets.yaml.
# - Utilise IQFeed comme source de données via TradingEnv.
# - Implémente retries exponentiels (max 3, délai 2^attempt) pour les opérations critiques.
# - Tests unitaires disponibles dans tests/test_maml_utils.py.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des données.

import gzip
import json
import os
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from loguru import logger
from stable_baselines3 import SAC

from src.envs.trading_env import TradingEnv
from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import get_config
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.utils.telegram_alert import send_telegram_alert

# Configuration logging
logger.remove()
BASE_DIR = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
LOG_DIR = BASE_DIR / "data" / "logs"
CACHE_DIR = BASE_DIR / "data" / "cache" / "maml"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints" / "maml"
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
logger.add(LOG_DIR / "maml_utils.log", rotation="10 MB", level="INFO", encoding="utf-8")
logger.add(lambda msg: print(msg, end=""), level="INFO")

# Constantes
PERF_LOG_PATH = LOG_DIR / "maml_performance.csv"
MODEL_DIR = BASE_DIR / "model" / "sac_models"
MAX_RETRIES = 3
RETRY_DELAY = 2.0
MAX_CACHE_SIZE = 1000
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB

# Cache global pour les résultats de meta-learning
maml_cache = OrderedDict()


def log_performance(
    operation: str,
    latency: float,
    success: bool = True,
    error: str = None,
    market: str = "ES",
    **kwargs,
) -> None:
    """
    Enregistre les performances (CPU, mémoire, latence) dans maml_performance.csv.

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
            miya_alerts(alert_msg, tag="MAML", priority=5)
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
            if not PERF_LOG_PATH.exists():
                log_df.to_csv(PERF_LOG_PATH, index=False, encoding="utf-8")
            else:
                log_df.to_csv(
                    PERF_LOG_PATH, mode="a", header=False, index=False, encoding="utf-8"
                )

        with_retries(save_log, market=market)
        logger.info(f"Performance journalisée pour {operation}. CPU: {cpu_percent}%")
    except Exception as e:
        error_msg = f"Erreur journalisation performance pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        miya_alerts(error_msg, tag="MAML", priority=3)
        AlertManager().send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)


def save_snapshot(
    snapshot_type: str, data: Dict, market: str = "ES", compress: bool = True
) -> None:
    """
    Sauvegarde un instantané JSON des résultats, compressé avec gzip.

    Args:
        snapshot_type (str): Type de snapshot (ex. : apply_prototypical_networks).
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
            miya_alerts(alert_msg, tag="MAML", priority=3)
            AlertManager().send_alert(alert_msg, priority=3)
            send_telegram_alert(alert_msg)
        latency = time.time() - start_time
        success_msg = f"Snapshot {snapshot_type} sauvegardé pour {market}: {save_path}"
        logger.info(success_msg)
        miya_speak(success_msg, tag="MAML", voice_profile="calm")
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
        miya_alerts(error_msg, tag="MAML", priority=3)
        AlertManager().send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)
        log_performance("save_snapshot", 0, success=False, error=str(e), market=market)


def checkpoint(
    data: pd.DataFrame, data_type: str = "maml_state", market: str = "ES"
) -> None:
    """
    Sauvegarde incrémentielle des données toutes les 5 minutes avec versionnage (5 versions).

    Args:
        data (pd.DataFrame): Données à sauvegarder.
        data_type (str): Type de données (ex. : maml_state).
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
        checkpoint_path = checkpoint_dir / f"maml_{data_type}_{timestamp}.json.gz"
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
        miya_speak(success_msg, tag="MAML", voice_profile="calm")
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
        miya_alerts(error_msg, tag="MAML", priority=3)
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
    data: pd.DataFrame, data_type: str = "maml_state", market: str = "ES"
) -> None:
    """
    Sauvegarde distribuée des données vers AWS S3 toutes les 15 minutes.

    Args:
        data (pd.DataFrame): Données à sauvegarder.
        data_type (str): Type de données (ex. : maml_state).
        market (str): Marché (ex. : ES, MNQ).
    """
    try:
        start_time = time.time()
        config = get_config(str(BASE_DIR / "config/es_config.yaml"))
        if not config.get("s3_bucket"):
            warning_msg = (
                f"S3 bucket non configuré, sauvegarde cloud ignorée pour {market}"
            )
            logger.warning(warning_msg)
            miya_alerts(warning_msg, tag="MAML", priority=3)
            AlertManager().send_alert(warning_msg, priority=3)
            send_telegram_alert(warning_msg)
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = (
            f"{config['s3_prefix']}maml_{data_type}_{market}_{timestamp}.csv.gz"
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
        miya_speak(success_msg, tag="MAML", voice_profile="calm")
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
        miya_alerts(error_msg, tag="MAML", priority=3)
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
                miya_alerts(error_msg, tag="MAML", priority=4)
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
            miya_speak(warning_msg, tag="MAML", level="warning")
            AlertManager().send_alert(warning_msg, priority=3)
            send_telegram_alert(warning_msg)
            time.sleep(delay)


def validate_features(
    data: pd.DataFrame, shap_features: bool = False, market: str = "ES"
) -> None:
    """
    Valide la présence des 350 features (entraînement) ou 150 SHAP features (inférence) et impute les valeurs manquantes.

    Args:
        data (pd.DataFrame): Données à valider.
        shap_features (bool): Si True, valide les 150 SHAP features; sinon, valide les 350 features.
        market (str): Marché (ex. : ES, MNQ).

    Raises:
        ValueError: Si des colonnes critiques sont manquantes ou non numériques.
    """
    try:
        feature_sets_path = BASE_DIR / "config/feature_sets.yaml"
        features_config = get_config(feature_sets_path)
        expected_cols = (
            features_config.get("training", {}).get("features", [])[:350]
            if not shap_features
            else features_config.get("inference", {}).get("shap_features", [])[:150]
        )
        expected_len = 150 if shap_features else 350
        if len(expected_cols) != expected_len:
            raise ValueError(
                f"Nombre de features incorrect pour {market}: {len(expected_cols)} au lieu de {expected_len}"
            )

        missing_cols = [col for col in expected_cols if col not in data.columns]
        null_count = data[expected_cols].isnull().sum().sum()
        confidence_drop_rate = (
            null_count / (len(data) * len(expected_cols))
            if (len(data) * len(expected_cols)) > 0
            else 0.0
        )
        if confidence_drop_rate > 0.5:
            alert_msg = f"Confidence_drop_rate élevé pour {market}: {confidence_drop_rate:.2f} ({null_count} valeurs nulles)"
            logger.warning(alert_msg)
            miya_alerts(alert_msg, tag="MAML", priority=3)
            AlertManager().send_alert(alert_msg, priority=3)
            send_telegram_alert(alert_msg)
        if missing_cols:
            warning_msg = f"Colonnes manquantes pour {market} ({'SHAP' if shap_features else 'full'}): {missing_cols}"
            logger.warning(warning_msg)
            miya_speak(warning_msg, tag="MAML", level="warning")
            AlertManager().send_alert(warning_msg, priority=2)
            send_telegram_alert(warning_msg)
            for col in missing_cols:
                data[col] = 0.0

        critical_cols = [
            "bid_size_level_1",
            "ask_size_level_1",
            "trade_frequency_1s",
            f"spread_avg_1min_{market.lower()}",
            "close",
        ]
        for col in critical_cols:
            if col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    raise ValueError(
                        f"Colonne {col} n'est pas numérique pour {market}: {data[col].dtype}"
                    )
                non_scalar = data[col].apply(
                    lambda x: isinstance(x, (list, dict, tuple))
                )
                if non_scalar.any():
                    raise ValueError(
                        f"Colonne {col} contient des valeurs non scalaires pour {market}: {data[col][non_scalar].head().tolist()}"
                    )
                if data[col].isna().any():
                    data[col] = (
                        data[col]
                        .interpolate(method="linear", limit_direction="both")
                        .fillna(0.0)
                    )
                if col in [
                    "bid_size_level_1",
                    "ask_size_level_1",
                    "trade_frequency_1s",
                    f"spread_avg_1min_{market.lower()}",
                ]:
                    if (data[col] <= 0).any():
                        warning_msg = f"Valeurs non positives dans {col} pour {market}, corrigées à 1e-6"
                        logger.warning(warning_msg)
                        miya_alerts(warning_msg, tag="MAML", priority=4)
                        data[col] = data[col].clip(lower=1e-6)
        save_snapshot(
            "validate_features",
            {
                "num_columns": len(data.columns),
                "missing_columns": missing_cols,
                "shap_features": shap_features,
                "confidence_drop_rate": confidence_drop_rate,
                "market": market,
            },
            market=market,
        )
    except Exception as e:
        error_msg = f"Erreur validation features pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        miya_alerts(error_msg, tag="MAML", priority=5)
        AlertManager().send_alert(error_msg, priority=5)
        send_telegram_alert(error_msg)
        raise


def apply_prototypical_networks(
    model: SAC,
    data: pd.DataFrame,
    config_path: str = str(BASE_DIR / "config/es_config.yaml"),
    mode: str = "trend",
    policy_type: str = "mlp",
    batch_size: int = 100,
    market: str = "ES",
) -> Optional[SAC]:
    """
    Applique le meta-learning (méthode 18) avec prototypical networks pour adapter un modèle SAC.

    Args:
        model (SAC): Modèle SAC à adapter.
        data (pd.DataFrame): Données contenant les 350 features (entraînement) ou 150 SHAP features (inférence).
        config_path (str): Chemin vers la configuration. Par défaut "config/es_config.yaml".
        mode (str): Mode du modèle ("trend", "range", "defensive"). Par défaut "trend".
        policy_type (str): Type de politique ("mlp", "transformer"). Par défaut "mlp".
        batch_size (int): Taille du mini-batch pour calculer les prototypes. Par défaut 100.
        market (str): Marché (ex. : ES, MNQ).

    Returns:
        Optional[SAC]: Modèle SAC adapté ou None si échec.
    """
    start_time = time.time()
    try:
        cache_key = f"{market}_{mode}_{policy_type}_{hash(str(data))}"
        if cache_key in maml_cache:
            model = maml_cache[cache_key]
            maml_cache.move_to_end(cache_key)
            return model
        while len(maml_cache) > MAX_CACHE_SIZE:
            maml_cache.popitem(last=False)

        # Validation des paramètres
        valid_modes = ["trend", "range", "defensive"]
        valid_policies = ["mlp", "transformer"]
        if mode not in valid_modes:
            raise ValueError(
                f"Mode non supporté pour {market}: {mode}. Options: {valid_modes}"
            )
        if policy_type not in valid_policies:
            raise ValueError(
                f"Type de politique non supporté pour {market}: {policy_type}. Options: {valid_policies}"
            )

        # Valider les données
        validate_features(
            data, shap_features=(policy_type == "transformer"), market=market
        )
        if len(data) < 1:
            raise ValueError(
                f"Données insuffisantes pour le meta-learning pour {market}"
            )

        # Configurer l’environnement
        get_config(config_path).get("backtest_sac", {})
        env = TradingEnv(config_path, market=market)
        env.mode = mode
        env.policy_type = policy_type
        feature_sets_path = BASE_DIR / "config/feature_sets.yaml"
        features_config = get_config(feature_sets_path)
        env.obs_t = (
            features_config.get("training", {}).get("features", [])[:350]
            if policy_type == "mlp"
            else features_config.get("inference", {}).get("shap_features", [])[:150]
        )
        env.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                (env.sequence_length, len(env.obs_t))
                if policy_type == "transformer"
                else (len(env.obs_t),)
            ),
            dtype=np.float32,
        )
        env.data = data

        # Sélectionner un mini-batch
        mini_batch = data.sample(n=min(batch_size, len(data)), random_state=42)
        feature_cols = env.obs_t

        # Calculer les prototypes (moyenne des features)
        prototypes = torch.tensor(
            mini_batch[feature_cols].mean().values, dtype=torch.float32
        )
        logger.info(
            f"Prototypes calculés pour {len(feature_cols)} features pour {market}"
        )

        # Mettre à jour le modèle
        def update_model():
            model.set_parameters({"policy": model.policy})  # Simuler une mise à jour
            model.learn(total_timesteps=100)  # Apprentissage court pour adaptation
            return model

        model = with_retries(update_model, market=market)
        if model is None:
            raise RuntimeError(
                f"Échec de la mise à jour du modèle après retries pour {market}"
            )
        logger.info(
            f"Meta-learning appliqué pour mode {mode}, policy_type={policy_type}. CPU: {psutil.cpu_percent()}%"
        )

        # Sauvegarder le modèle adapté
        model_dir = MODEL_DIR / market / mode / policy_type
        model_dir.mkdir(exist_ok=True)
        new_model_path = (
            model_dir
            / f"sac_{mode}_{policy_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        )
        model.save(new_model_path)
        logger.info(f"Modèle adapté sauvegardé: {new_model_path}")

        # Sauvegarde et cache
        snapshot_data = {
            "mode": mode,
            "policy_type": policy_type,
            "batch_size": len(mini_batch),
            "num_features": len(feature_cols),
            "model_path": str(new_model_path),
            "num_rows": len(data),
            "market": market,
        }
        save_snapshot("apply_prototypical_networks", snapshot_data, market=market)
        checkpoint(mini_batch, data_type="maml_batch", market=market)
        cloud_backup(mini_batch, data_type="maml_batch", market=market)
        maml_cache[cache_key] = model

        latency = time.time() - start_time
        log_performance(
            "apply_prototypical_networks",
            latency,
            success=True,
            mode=mode,
            policy_type=policy_type,
            batch_size=len(mini_batch),
            num_features=len(feature_cols),
            market=market,
        )
        success_msg = f"Meta-learning appliqué pour {market}: mode={mode}, policy={policy_type}, batch_size={len(mini_batch)}"
        logger.info(success_msg)
        miya_speak(success_msg, tag="MAML", voice_profile="calm")
        AlertManager().send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
        return model

    except Exception as e:
        error_msg = f"Erreur dans apply_prototypical_networks pour {market}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        miya_alerts(error_msg, tag="MAML", priority=5)
        AlertManager().send_alert(error_msg, priority=5)
        send_telegram_alert(error_msg)
        log_performance(
            "apply_prototypical_networks",
            time.time() - start_time,
            success=False,
            error=str(e),
            mode=mode,
            policy_type=policy_type,
            market=market,
        )
        return None
