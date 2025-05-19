# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/model/utils/obs_template.py
# Formate le vecteur d’observation (150 SHAP features) pour trading_env.py, optimisé pour l’inférence.
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle :
# Formate le vecteur d’observation pour trading_env.py, intégrant la pondération des features selon le régime (méthode 3),
# utilisant les top 150 SHAP features sélectionnées via shap_weighting.py. Documente les 150 SHAP features dans
# feature_sets.yaml. Utilise exclusivement IQFeed comme source de données.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.8,<6.0.0, pyyaml>=6.0.0,<7.0.0, matplotlib>=3.7.0,<4.0.0,
#   boto3>=1.26.0,<2.0.0, loguru>=0.7.0,<1.0.0, signal
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
# - src/features/shap_weighting.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/es_config.yaml (paramètres de configuration)
# - config/feature_sets.yaml (350 features et 150 SHAP features)
# - data/features/feature_importance.csv (SHAP features)
# - data/features/feature_importance_cache.csv (cache SHAP)
#
# Outputs :
# - data/logs/<market>/obs_template_performance.csv (logs psutil)
# - data/cache/obs_template/<market>/*.json.gz (snapshots JSON)
# - data/figures/obs_template/<market>/ (visualisations matplotlib)
# - data/features/<market>/obs_template.csv (vecteur d’observation)
# - config/feature_sets.yaml (150 SHAP features documentées)
# - data/checkpoints/obs_template/<market>/*.json.gz (sauvegardes incrémentielles)
#
# Notes :
# - Préserve toutes les fonctionnalités existantes (formatage pour SAC, notifications via MiyaConsole).
# - Utilise les top 150 SHAP features pour l’inférence, compatible avec 350 features (entraînement).
# - Supprime toutes les références à 320 features, 81 features, et obs_t.
# - Intègre un fallback SHAP (cache ou liste statique).
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des opérations.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.
# - Tests unitaires disponibles dans tests/test_obs_template.py.
# - TODO : Intégration future avec API Bloomberg (juin 2025).

import gzip
import json
import signal
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import yaml
from loguru import logger

from src.features.shap_weighting import calculate_shap_weights
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
CACHE_DIR = BASE_DIR / "data" / "cache" / "obs_template"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints" / "obs_template"
FIGURE_DIR = BASE_DIR / "data" / "figures" / "obs_template"
FEATURE_DIR = BASE_DIR / "data" / "features"
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)
FEATURE_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "obs_template.log", rotation="10 MB", level="INFO", encoding="utf-8"
)
logger.add(lambda msg: print(msg, end=""), level="INFO")

# Constantes
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Secondes, exponentiel
MAX_CACHE_SIZE = 1000
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB

# Cache global pour les résultats
OBS_CACHE = OrderedDict()


class ObsTemplate:
    """Classe pour gérer le vecteur d'observation et sa validation."""

    def __init__(
        self,
        config_path: str = str(BASE_DIR / "config" / "es_config.yaml"),
        market: str = "ES",
    ):
        """
        Initialise le gestionnaire du vecteur d’observation.

        Args:
            config_path (str): Chemin vers le fichier de configuration.
            market (str): Marché (ex. : ES, MNQ).
        """
        self.market = market
        self.log_buffer = []
        self.cache = OrderedDict()
        self.alert_manager = AlertManager(market=market)
        self.miya_console = MiyaConsole(market=market)
        try:
            self.config = get_config(config_path).get("obs_template", {})
            self.buffer_size = self.config.get("buffer_size", 100)
            self.max_cache_size = self.config.get("max_cache_size", 1000)
            signal.signal(signal.SIGINT, self.handle_sigint)
            self.miya_console.miya_speak(
                "ObsTemplate initialisé",
                tag="OBS_TEMPLATE",
                voice_profile="calm",
                priority=2,
            )
            self.alert_manager.send_alert("ObsTemplate initialisé", priority=1)
            logger.info(f"ObsTemplate initialisé pour {market}")
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config_path": config_path})
            self.update_feature_sets_yaml([])  # Initialiser feature_sets.yaml
        except Exception as e:
            error_msg = f"Erreur initialisation ObsTemplate pour {market}: {str(e)}\n{traceback.format_exc()}"
            self.miya_console.miya_alerts(
                error_msg, tag="OBS_TEMPLATE", voice_profile="urgent"
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            self.config = {
                "buffer_size": 100,
                "max_cache_size": 1000,
                "weights": {
                    "trend": {"mm_score": 1.5, "hft_score": 1.5, "breakout_score": 1.0},
                    "range": {"breakout_score": 1.5, "mm_score": 1.0, "hft_score": 1.0},
                    "defensive": {
                        "mm_score": 0.5,
                        "hft_score": 0.5,
                        "breakout_score": 0.5,
                    },
                },
            }
            self.buffer_size = 100
            self.max_cache_size = 1000

    def handle_sigint(self, signum, frame):
        """Gère l'arrêt propre sur SIGINT."""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "status": "SIGINT",
            "market": self.market,
            "cache_size": len(self.cache),
        }
        snapshot_path = (
            CACHE_DIR / self.market
        ) / f'sigint_{snapshot["timestamp"]}.json.gz'
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
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ) -> None:
        """
        Journalise les performances des opérations critiques.

        Args:
            operation (str): Nom de l’opération (ex. : create_observation).
            latency (float): Latence en secondes.
            success (bool): Succès de l’opération.
            error (str, optional): Message d’erreur si échec.
            **kwargs: Paramètres supplémentaires (ex. : num_features).
        """
        cache_key = f"{self.market}_{operation}_{hash(str(latency))}_{hash(str(error))}"
        if cache_key in OBS_CACHE:
            return
        while len(OBS_CACHE) > MAX_CACHE_SIZE:
            OBS_CACHE.popitem(last=False)

        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            confidence_drop_rate = 1.0 if success else 0.0  # Simplifié pour Phase 8
            if memory_usage > 1024:
                alert_msg = f"ALERTE: Usage mémoire élevé ({memory_usage:.2f} MB) pour {self.market}"
                self.miya_console.miya_alerts(
                    alert_msg, tag="OBS_TEMPLATE", level="error", priority=5
                )
                self.alert_manager.send_alert(alert_msg, priority=4)
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
            self.log_buffer.append(log_entry)
            if len(self.log_buffer) >= self.buffer_size:
                log_df = pd.DataFrame(self.log_buffer)
                log_path = LOG_DIR / self.market / "obs_template_performance.csv"
                log_path.parent.mkdir(exist_ok=True)

                def save_log():
                    if not log_path.exists():
                        log_df.to_csv(log_path, index=False, encoding="utf-8")
                    else:
                        log_df.to_csv(
                            log_path,
                            mode="a",
                            header=False,
                            index=False,
                            encoding="utf-8",
                        )

                self.with_retries(save_log)
                self.log_buffer = []
            success_msg = f"Performance journalisée pour {operation} pour {self.market}. CPU: {cpu_percent}%"
            logger.info(success_msg)
            send_telegram_alert(success_msg)
            OBS_CACHE[cache_key] = True
        except Exception as e:
            error_msg = f"Erreur journalisation performance pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            self.miya_console.miya_alerts(
                error_msg, tag="OBS_TEMPLATE", level="error", priority=3
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def save_snapshot(
        self, snapshot_type: str, data: Dict, compress: bool = True
    ) -> None:
        """
        Sauvegarde un instantané JSON des résultats, compressé avec gzip.

        Args:
            snapshot_type (str): Type de snapshot (ex. : create_observation).
            data (Dict): Données à sauvegarder.
            compress (bool): Compresser avec gzip (défaut : True).
        """
        start_time = time.time()
        snapshot_dir = CACHE_DIR / self.market
        snapshot_dir.mkdir(exist_ok=True)
        try:
            if not os.access(snapshot_dir, os.W_OK):
                error_msg = f"Permission d’écriture refusée pour {snapshot_dir}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = {
                "timestamp": timestamp,
                "type": snapshot_type,
                "market": self.market,
                "data": data,
                "buffer_size": len(self.log_buffer),
            }
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
        self, data: pd.DataFrame, data_type: str = "obs_template_state"
    ) -> None:
        """
        Sauvegarde incrémentielle des données toutes les 5 minutes avec versionnage (5 versions).

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : obs_template_state).
        """
        start_time = time.time()
        checkpoint_dir = CHECKPOINT_DIR / self.market
        checkpoint_dir.mkdir(exist_ok=True)
        try:
            if not os.access(checkpoint_dir, os.W_OK):
                error_msg = f"Permission d’écriture refusée pour {checkpoint_dir}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_data = {
                "timestamp": timestamp,
                "num_rows": len(data),
                "columns": list(data.columns),
                "data_type": data_type,
                "market": self.market,
            }
            checkpoint_path = (
                checkpoint_dir / f"obs_template_{data_type}_{timestamp}.json.gz"
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
        self, data: pd.DataFrame, data_type: str = "obs_template_state"
    ) -> None:
        """
        Sauvegarde distribuée des données vers AWS S3 toutes les 15 minutes.

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : obs_template_state).
        """
        start_time = time.time()
        checkpoint_dir = CHECKPOINT_DIR / self.market
        checkpoint_dir.mkdir(exist_ok=True)
        try:
            config = get_config(str(BASE_DIR / "config/es_config.yaml"))
            if not config.get("s3_bucket"):
                warning_msg = f"S3 bucket non configuré, sauvegarde cloud ignorée pour {self.market}"
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{config['s3_prefix']}obs_template_{data_type}_{self.market}_{timestamp}.csv.gz"
            temp_path = checkpoint_dir / f"temp_s3_{timestamp}.csv.gz"

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

    def update_feature_sets_yaml(self, features: List[str]) -> None:
        """
        Met à jour feature_sets.yaml avec les 150 SHAP features du vecteur d’observation.

        Args:
            features (List[str]): Liste des 150 SHAP features à inclure.
        """
        start_time = time.time()
        feature_sets_path = BASE_DIR / "config" / "feature_sets.yaml"
        try:
            if feature_sets_path.exists():
                with open(feature_sets_path, "r", encoding="utf-8") as f:
                    feature_sets = yaml.safe_load(f) or {}
            else:
                feature_sets = {
                    "metadata": {
                        "version": "2.1.4",
                        "updated": "2025-05-13",
                        "total_features": 350,
                    },
                    "dashboard_display": {
                        "enabled": True,
                        "status_file": str(
                            BASE_DIR / "data" / "feature_sets_dashboard.json"
                        ),
                        "categories": [
                            "raw_data",
                            "order_flow",
                            "trend",
                            "volatility",
                            "neural_pipeline",
                            "latent_factors",
                            "self_awareness",
                            "mia_memory",
                            "option_metrics",
                            "market_structure_signals",
                            "context_aware",
                            "cross_asset",
                            "dynamic_features",
                        ],
                    },
                    "feature_sets": {},
                }

            observation_features = (
                features
                if features
                else [
                    "rsi_14",
                    "adx_14",
                    "momentum_10",
                    "atr_14",
                    "bollinger_width_20",
                    "delta_volume",
                    "obi_score",
                    "absorption_strength",
                    "bid_ask_ratio",
                    "gex",
                    "oi_peak_call_near",
                    "oi_peak_put_near",
                    "gamma_wall",
                    "vanna_strike_near",
                    "iv_skew_10delta",
                    "max_pain_strike",
                    "gamma_velocity_near",
                    "delta_hedge_pressure",
                    "breakout_score",
                    "mm_score",
                    "hft_score",
                    "volatility_rolling_20",
                    "spy_lead_return",
                    "vix_es_correlation",
                    "bond_equity_risk_spread",
                    "event_volatility_impact",
                    "event_frequency_24h",
                    "news_impact_score",
                    "predicted_volatility",
                    "neural_regime",
                    "cnn_pressure",
                    "neural_feature_0",
                    "neural_feature_1",
                    "neural_feature_2",
                    "neural_feature_3",
                    "neural_feature_4",
                    "neural_feature_5",
                    "neural_feature_6",
                    "neural_feature_7",
                    "latent_vol_regime_vec_1",
                    "topic_vector_news_0",
                    "topic_vector_news_1",
                    "topic_vector_news_2",
                    "news_topic_stability",
                    "vanna_cliff_slope",
                    "latency_spread",
                    "order_flow_acceleration",
                    "confidence_drop_rate",
                    "sgc_entropy",
                    "key_strikes_1",
                    "key_strikes_2",
                    "key_strikes_3",
                    "key_strikes_4",
                    "key_strikes_5",
                    "net_gamma",
                    "zero_gamma",
                    "dealer_zones_count",
                    "vol_trigger",
                    "ref_px",
                    "data_release",
                    "trade_frequency_1s",
                    "neural_dynamic_feature_1",
                    "neural_dynamic_feature_2",
                    "neural_dynamic_feature_3",
                    "neural_dynamic_feature_4",
                    "neural_dynamic_feature_5",
                    "neural_dynamic_feature_6",
                    "neural_dynamic_feature_7",
                    "neural_dynamic_feature_8",
                    "neural_dynamic_feature_9",
                    "neural_dynamic_feature_10",
                    "neural_dynamic_feature_11",
                    "neural_dynamic_feature_12",
                    "neural_dynamic_feature_13",
                    "neural_dynamic_feature_14",
                    "neural_dynamic_feature_15",
                    "neural_dynamic_feature_16",
                    "neural_dynamic_feature_17",
                    "neural_dynamic_feature_18",
                    "neural_dynamic_feature_19",
                    "neural_dynamic_feature_20",
                    "neural_dynamic_feature_21",
                    "neural_dynamic_feature_22",
                    "neural_dynamic_feature_23",
                    "neural_dynamic_feature_24",
                    "neural_dynamic_feature_25",
                    "neural_dynamic_feature_26",
                    "neural_dynamic_feature_27",
                    "neural_dynamic_feature_28",
                    "neural_dynamic_feature_29",
                    "neural_dynamic_feature_30",
                    "neural_dynamic_feature_31",
                    "neural_dynamic_feature_32",
                    "neural_dynamic_feature_33",
                    "neural_dynamic_feature_34",
                    "neural_dynamic_feature_35",
                    "neural_dynamic_feature_36",
                    "neural_dynamic_feature_37",
                    "neural_dynamic_feature_38",
                    "neural_dynamic_feature_39",
                    "neural_dynamic_feature_40",
                    "neural_dynamic_feature_41",
                    "neural_dynamic_feature_42",
                    "neural_dynamic_feature_43",
                    "neural_dynamic_feature_44",
                    "neural_dynamic_feature_45",
                    "neural_dynamic_feature_46",
                    "neural_dynamic_feature_47",
                    "neural_dynamic_feature_48",
                    "neural_dynamic_feature_49",
                    "neural_dynamic_feature_50",
                ]
            )
            feature_sets["feature_sets"]["observation"] = {
                "description": f"Features du vecteur d’observation (top 150 SHAP) pour trading_env.py pour {self.market}",
                "dimensions": 150,
                "features": [
                    {
                        "name": f,
                        "type": "float",
                        "source": "calculated",
                        "status": "active",
                        "range": [-float("inf"), float("inf")],
                        "priority": "high",
                        "description": f"Feature SHAP pour le vecteur d’observation pour {self.market}",
                    }
                    for f in observation_features[:150]
                ],
            }

            feature_sets["metadata"]["updated"] = "2025-05-13"
            feature_sets["metadata"]["total_features"] = 350

            def save_yaml():
                with open(feature_sets_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(feature_sets, f, allow_unicode=True, sort_keys=False)

            self.with_retries(save_yaml)
            latency = time.time() - start_time
            success_msg = f"feature_sets.yaml mis à jour avec {len(observation_features)} features pour {self.market}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "update_feature_sets_yaml",
                latency,
                success=True,
                num_features=len(observation_features),
            )
            self.save_snapshot(
                "update_feature_sets_yaml", {"num_features": len(observation_features)}
            )
            checkpoint(
                pd.DataFrame([{"features": observation_features[:150]}]),
                data_type="update_feature_sets_yaml",
            )
            cloud_backup(
                pd.DataFrame([{"features": observation_features[:150]}]),
                data_type="update_feature_sets_yaml",
            )
        except Exception as e:
            error_msg = f"Erreur mise à jour feature_sets.yaml pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "update_feature_sets_yaml", 0, success=False, error=str(e)
            )

    def get_shap_features(self, data: pd.DataFrame, regime: str) -> List[str]:
        """
        Sélectionne les top 150 SHAP features.

        Args:
            data (pd.DataFrame): Données d'entrée avec 350 features.
            regime (str): Régime de marché (trend, range, defensive).

        Returns:
            List[str]: Liste des 150 SHAP features.
        """
        start_time = time.time()
        cache_key = (
            f"{self.market}_get_shap_features_{regime}_{hash(str(data.columns))}"
        )
        if cache_key in OBS_CACHE:
            return OBS_CACHE[cache_key]
        try:
            shap_file = FEATURE_DIR / self.market / "feature_importance.csv"
            cache_file = FEATURE_DIR / self.market / "feature_importance_cache.csv"
            if shap_file.exists():
                shap_df = pd.read_csv(shap_file)
            elif cache_file.exists():
                shap_df = pd.read_csv(cache_file)
                warning_msg = (
                    f"Utilisation du cache SHAP pour {self.market}: {cache_file}"
                )
                logger.warning(warning_msg)
                send_telegram_alert(warning_msg)
            else:
                shap_features = [
                    "rsi_14",
                    "adx_14",
                    "momentum_10",
                    "atr_14",
                    "bollinger_width_20",
                    "delta_volume",
                    "obi_score",
                    "absorption_strength",
                    "bid_ask_ratio",
                    "gex",
                    "oi_peak_call_near",
                    "oi_peak_put_near",
                    "gamma_wall",
                    "vanna_strike_near",
                    "iv_skew_10delta",
                    "max_pain_strike",
                    "gamma_velocity_near",
                    "delta_hedge_pressure",
                    "breakout_score",
                    "mm_score",
                    "hft_score",
                    "volatility_rolling_20",
                    "spy_lead_return",
                    "vix_es_correlation",
                    "bond_equity_risk_spread",
                    "event_volatility_impact",
                    "event_frequency_24h",
                    "news_impact_score",
                    "predicted_volatility",
                    "neural_regime",
                    "cnn_pressure",
                    "neural_feature_0",
                    "neural_feature_1",
                    "neural_feature_2",
                    "neural_feature_3",
                    "neural_feature_4",
                    "neural_feature_5",
                    "neural_feature_6",
                    "neural_feature_7",
                    "latent_vol_regime_vec_1",
                    "topic_vector_news_0",
                    "topic_vector_news_1",
                    "topic_vector_news_2",
                    "news_topic_stability",
                    "vanna_cliff_slope",
                    "latency_spread",
                    "order_flow_acceleration",
                    "confidence_drop_rate",
                    "sgc_entropy",
                    "key_strikes_1",
                    "key_strikes_2",
                    "key_strikes_3",
                    "key_strikes_4",
                    "key_strikes_5",
                    "net_gamma",
                    "zero_gamma",
                    "dealer_zones_count",
                    "vol_trigger",
                    "ref_px",
                    "data_release",
                    "trade_frequency_1s",
                    "neural_dynamic_feature_1",
                    "neural_dynamic_feature_2",
                    "neural_dynamic_feature_3",
                    "neural_dynamic_feature_4",
                    "neural_dynamic_feature_5",
                    "neural_dynamic_feature_6",
                    "neural_dynamic_feature_7",
                    "neural_dynamic_feature_8",
                    "neural_dynamic_feature_9",
                    "neural_dynamic_feature_10",
                    "neural_dynamic_feature_11",
                    "neural_dynamic_feature_12",
                    "neural_dynamic_feature_13",
                    "neural_dynamic_feature_14",
                    "neural_dynamic_feature_15",
                    "neural_dynamic_feature_16",
                    "neural_dynamic_feature_17",
                    "neural_dynamic_feature_18",
                    "neural_dynamic_feature_19",
                    "neural_dynamic_feature_20",
                    "neural_dynamic_feature_21",
                    "neural_dynamic_feature_22",
                    "neural_dynamic_feature_23",
                    "neural_dynamic_feature_24",
                    "neural_dynamic_feature_25",
                    "neural_dynamic_feature_26",
                    "neural_dynamic_feature_27",
                    "neural_dynamic_feature_28",
                    "neural_dynamic_feature_29",
                    "neural_dynamic_feature_30",
                    "neural_dynamic_feature_31",
                    "neural_dynamic_feature_32",
                    "neural_dynamic_feature_33",
                    "neural_dynamic_feature_34",
                    "neural_dynamic_feature_35",
                    "neural_dynamic_feature_36",
                    "neural_dynamic_feature_37",
                    "neural_dynamic_feature_38",
                    "neural_dynamic_feature_39",
                    "neural_dynamic_feature_40",
                    "neural_dynamic_feature_41",
                    "neural_dynamic_feature_42",
                    "neural_dynamic_feature_43",
                    "neural_dynamic_feature_44",
                    "neural_dynamic_feature_45",
                    "neural_dynamic_feature_46",
                    "neural_dynamic_feature_47",
                    "neural_dynamic_feature_48",
                    "neural_dynamic_feature_49",
                    "neural_dynamic_feature_50",
                ]
                warning_msg = f"SHAP non disponible pour {self.market}, utilisation de la liste statique: {len(shap_features)} features"
                logger.warning(warning_msg)
                send_telegram_alert(warning_msg)
                OBS_CACHE[cache_key] = shap_features
                return shap_features

            shap_df = calculate_shap_weights(data, regime)
            new_metrics = [
                "key_strikes_1",
                "key_strikes_2",
                "key_strikes_3",
                "key_strikes_4",
                "key_strikes_5",
                "max_pain_strike",
                "net_gamma",
                "zero_gamma",
                "dealer_zones_count",
                "vol_trigger",
                "ref_px",
                "data_release",
            ]
            for metric in new_metrics:
                if metric in shap_df["feature"].values:
                    shap_df.loc[shap_df["feature"] == metric, "importance"] *= 1.2

            shap_df = shap_df.sort_values("importance", ascending=False).head(150)
            selected_features = shap_df["feature"].tolist()
            latency = time.time() - start_time
            success_msg = f"Sélectionné {len(selected_features)} SHAP features pour régime {regime} pour {self.market}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "get_shap_features",
                latency,
                success=True,
                num_features=len(selected_features),
            )
            self.save_snapshot(
                "get_shap_features",
                {"num_features": len(selected_features), "regime": regime},
            )
            OBS_CACHE[cache_key] = selected_features
            return selected_features
        except Exception as e:
            error_msg = f"Erreur sélection SHAP features pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance("get_shap_features", 0, success=False, error=str(e))
            shap_features = [
                "rsi_14",
                "adx_14",
                "momentum_10",
                "atr_14",
                "bollinger_width_20",
                "delta_volume",
                "obi_score",
                "absorption_strength",
                "bid_ask_ratio",
                "gex",
                "oi_peak_call_near",
                "oi_peak_put_near",
                "gamma_wall",
                "vanna_strike_near",
                "iv_skew_10delta",
                "max_pain_strike",
                "gamma_velocity_near",
                "delta_hedge_pressure",
                "breakout_score",
                "mm_score",
                "hft_score",
                "volatility_rolling_20",
                "spy_lead_return",
                "vix_es_correlation",
                "bond_equity_risk_spread",
                "event_volatility_impact",
                "event_frequency_24h",
                "news_impact_score",
                "predicted_volatility",
                "neural_regime",
                "cnn_pressure",
                "neural_feature_0",
                "neural_feature_1",
                "neural_feature_2",
                "neural_feature_3",
                "neural_feature_4",
                "neural_feature_5",
                "neural_feature_6",
                "neural_feature_7",
                "latent_vol_regime_vec_1",
                "topic_vector_news_0",
                "topic_vector_news_1",
                "topic_vector_news_2",
                "news_topic_stability",
                "vanna_cliff_slope",
                "latency_spread",
                "order_flow_acceleration",
                "confidence_drop_rate",
                "sgc_entropy",
                "key_strikes_1",
                "key_strikes_2",
                "key_strikes_3",
                "key_strikes_4",
                "key_strikes_5",
                "net_gamma",
                "zero_gamma",
                "dealer_zones_count",
                "vol_trigger",
                "ref_px",
                "data_release",
                "trade_frequency_1s",
                "neural_dynamic_feature_1",
                "neural_dynamic_feature_2",
                "neural_dynamic_feature_3",
                "neural_dynamic_feature_4",
                "neural_dynamic_feature_5",
                "neural_dynamic_feature_6",
                "neural_dynamic_feature_7",
                "neural_dynamic_feature_8",
                "neural_dynamic_feature_9",
                "neural_dynamic_feature_10",
                "neural_dynamic_feature_11",
                "neural_dynamic_feature_12",
                "neural_dynamic_feature_13",
                "neural_dynamic_feature_14",
                "neural_dynamic_feature_15",
                "neural_dynamic_feature_16",
                "neural_dynamic_feature_17",
                "neural_dynamic_feature_18",
                "neural_dynamic_feature_19",
                "neural_dynamic_feature_20",
                "neural_dynamic_feature_21",
                "neural_dynamic_feature_22",
                "neural_dynamic_feature_23",
                "neural_dynamic_feature_24",
                "neural_dynamic_feature_25",
                "neural_dynamic_feature_26",
                "neural_dynamic_feature_27",
                "neural_dynamic_feature_28",
                "neural_dynamic_feature_29",
                "neural_dynamic_feature_30",
                "neural_dynamic_feature_31",
                "neural_dynamic_feature_32",
                "neural_dynamic_feature_33",
                "neural_dynamic_feature_34",
                "neural_dynamic_feature_35",
                "neural_dynamic_feature_36",
                "neural_dynamic_feature_37",
                "neural_dynamic_feature_38",
                "neural_dynamic_feature_39",
                "neural_dynamic_feature_40",
                "neural_dynamic_feature_41",
                "neural_dynamic_feature_42",
                "neural_dynamic_feature_43",
                "neural_dynamic_feature_44",
                "neural_dynamic_feature_45",
                "neural_dynamic_feature_46",
                "neural_dynamic_feature_47",
                "neural_dynamic_feature_48",
                "neural_dynamic_feature_49",
                "neural_dynamic_feature_50",
            ]
            OBS_CACHE[cache_key] = shap_features
            return shap_features

    def create_observation(
        self, data: pd.DataFrame, regime: str = "range"
    ) -> np.ndarray:
        """
        Crée un vecteur d'observation de 150 dimensions avec pondération selon le régime.

        Args:
            data (pd.DataFrame): Données d'entrée avec les 350 features.
            regime (str, optional): Régime de marché (trend, range, defensive; défaut : 'range').

        Returns:
            np.ndarray: Vecteur d'observation de 150 dimensions.
        """
        start_time = time.time()
        cache_key = f"{self.market}_create_observation_{regime}_{hash(str(data))}"
        if cache_key in OBS_CACHE:
            return OBS_CACHE[cache_key]

        def compute_observation():
            valid_regimes = ["trend", "range", "defensive"]
            if regime not in valid_regimes:
                raise ValueError(
                    f"Régime invalide pour {self.market}: {regime}, attendu: {valid_regimes}"
                )

            shap_features = self.get_shap_features(data, regime)
            shap_data = data[shap_features].copy()

            weights = self.config.get("weights", {}).get(regime, {})
            for col in shap_data.columns:
                weight = weights.get(col, 1.0)
                shap_data[col] *= weight

            if len(shap_data.columns) != 150:
                raise ValueError(
                    f"Nombre de features incorrect pour {self.market}: {len(shap_data.columns)}, attendu: 150"
                )

            for col in shap_data.columns:
                shap_data[col] = shap_data[col].fillna(0)

            obs_template_csv = FEATURE_DIR / self.market / "obs_template.csv"
            obs_template_csv.parent.mkdir(exist_ok=True)
            shap_data.to_csv(obs_template_csv, encoding="utf-8", index=False)

            self.update_feature_sets_yaml(shap_data.columns.tolist())
            return shap_data.values[0]

        try:
            observation = self.with_retries(compute_observation)
            latency = time.time() - start_time
            success_msg = f"Vecteur d'observation créé pour {self.market}: {len(observation)} features, régime {regime}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "create_observation",
                latency,
                success=True,
                num_features=len(observation),
            )
            self.save_snapshot(
                "create_observation",
                {"num_features": len(observation), "regime": regime},
            )
            self.plot_observation(observation, datetime.now().strftime("%Y%m%d_%H%M%S"))
            checkpoint(
                pd.DataFrame([observation], columns=shap_features),
                data_type="create_observation",
            )
            cloud_backup(
                pd.DataFrame([observation], columns=shap_features),
                data_type="create_observation",
            )
            OBS_CACHE[cache_key] = observation
            return observation
        except Exception as e:
            error_msg = f"Erreur dans create_observation pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.miya_console.miya_alerts(
                error_msg, tag="OBS_TEMPLATE", voice_profile="urgent"
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance("create_observation", 0, success=False, error=str(e))
            return np.zeros(150)

    def plot_observation(self, observation: np.ndarray, timestamp: str) -> None:
        """
        Génère une visualisation du vecteur d’observation.

        Args:
            observation (np.ndarray): Vecteur d’observation (150 dimensions).
            timestamp (str): Horodatage pour le nom du fichier.
        """
        start_time = time.time()
        figure_dir = FIGURE_DIR / self.market
        figure_dir.mkdir(exist_ok=True)
        try:
            if not os.access(figure_dir, os.W_OK):
                error_msg = f"Permission d’écriture refusée pour {figure_dir}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                return
            plt.figure(figsize=(10, 6))
            plt.plot(
                observation,
                label="Observation Vector (150 SHAP features)",
                color="blue",
            )
            plt.title(
                f"Observation Vector (150 SHAP Features) pour {self.market} - {timestamp}"
            )
            plt.xlabel("Feature Index")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.savefig(
                figure_dir / f"observation_{timestamp}.png",
                bbox_inches="tight",
                optimize=True,
            )
            plt.close()
            latency = time.time() - start_time
            success_msg = (
                f"Visualisation générée pour {self.market}: observation_{timestamp}.png"
            )
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance("plot_observation", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur génération visualisation pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance("plot_observation", 0, success=False, error=str(e))

    def validate_obs_template(
        self, data: Optional[pd.DataFrame] = None, policy_type: str = "mlp"
    ) -> bool:
        """
        Valide la cohérence du vecteur d’observation avec les données fournies et les seuils de performance.

        Args:
            data (Optional[pd.DataFrame]): Données à valider (défaut : None).
            policy_type (str): Type de politique ("mlp", "transformer", défaut : "mlp").

        Returns:
            bool: True si la validation réussit, False sinon.
        """
        start_time = time.time()
        try:
            if data is None:
                success_msg = f"Validation limitée à la taille du vecteur pour {self.market}, aucune donnée fournie"
                logger.info(success_msg)
                self.miya_console.miya_speak(success_msg, tag="OBS_TEMPLATE")
                self.alert_manager.send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)
                return True

            feature_sets_path = BASE_DIR / "config" / "feature_sets.yaml"
            with open(feature_sets_path, "r", encoding="utf-8") as f:
                feature_sets = yaml.safe_load(f)
            expected_features = [
                f["name"]
                for f in feature_sets.get("feature_sets", {})
                .get("observation", {})
                .get("features", [])
            ][:150]

            missing_cols = [col for col in expected_features if col not in data.columns]
            if missing_cols:
                error_msg = f"Colonnes manquantes dans les données pour {self.market}: {missing_cols}, policy_type={policy_type}"
                logger.warning(error_msg)
                self.miya_console.miya_speak(
                    error_msg, tag="OBS_TEMPLATE", level="warning"
                )
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                return False

            critical_cols = [
                "bid_size_level_1",
                "ask_size_level_1",
                "delta_volume",
                "obi_score",
                "gex",
                "event_volatility_impact",
                "spy_lead_return",
                "latent_vol_regime_vec_1",
                "key_strikes_1",
                "max_pain_strike",
                "net_gamma",
                "zero_gamma",
                "dealer_zones_count",
                "vol_trigger",
                "ref_px",
                "news_impact_score",
            ]
            for col in critical_cols:
                if col in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        error_msg = f"Colonne {col} n’est pas numérique pour {self.market}: {data[col].dtype}"
                        logger.error(error_msg)
                        self.miya_console.miya_alerts(error_msg, tag="OBS_TEMPLATE")
                        self.alert_manager.send_alert(error_msg, priority=4)
                        send_telegram_alert(error_msg)
                        return False
                    non_scalar = [
                        val for val in data[col] if isinstance(val, (list, dict, tuple))
                    ]
                    if non_scalar:
                        error_msg = f"Colonne {col} contient des valeurs non scalaires pour {self.market}: {non_scalar[:5]}"
                        logger.error(error_msg)
                        self.miya_console.miya_alerts(error_msg, tag="OBS_TEMPLATE")
                        self.alert_manager.send_alert(error_msg, priority=4)
                        send_telegram_alert(error_msg)
                        return False

            PERFORMANCE_THRESHOLDS = {
                "predicted_volatility": {"min": 0.0},
                "neural_regime": {"min": 0, "max": 2},
                "cnn_pressure": {"min": -5.0, "max": 5.0},
                "bid_size_level_1": {"min": 0.0},
                "ask_size_level_1": {"min": 0.0},
                "delta_volume": {"min": float("-inf")},
                "obi_score": {"min": -1.0, "max": 1.0},
                "gex": {"min": float("-inf")},
                "event_volatility_impact": {"min": 0.0},
                "spy_lead_return": {"min": float("-inf")},
                "latent_vol_regime_vec_1": {"min": float("-inf")},
                "topic_vector_news_0": {"min": 0.0, "max": 1.0},
                "key_strikes_1": {"min": 0.0},
                "max_pain_strike": {"min": 0.0},
                "net_gamma": {"min": -float("inf"), "max": float("inf")},
                "zero_gamma": {"min": 0.0},
                "dealer_zones_count": {"min": 0, "max": 10},
                "vol_trigger": {"min": -float("inf"), "max": float("inf")},
                "ref_px": {"min": 0.0},
                "news_impact_score": {"min": 0.0, "max": 1.0},
            }

            for feature, thresholds in PERFORMANCE_THRESHOLDS.items():
                if feature in data.columns:
                    min_val = data[feature].min()
                    max_val = data[feature].max()
                    if "min" in thresholds and min_val < thresholds["min"]:
                        error_msg = f"Seuil non atteint pour {feature} pour {self.market}: min={min_val} < {thresholds['min']}, policy_type={policy_type}"
                        logger.warning(error_msg)
                        self.miya_console.miya_speak(
                            error_msg, tag="OBS_TEMPLATE", level="warning"
                        )
                        self.alert_manager.send_alert(error_msg, priority=3)
                        send_telegram_alert(error_msg)
                        return False
                    if "max" in thresholds and max_val > thresholds["max"]:
                        error_msg = f"Seuil non atteint pour {feature} pour {self.market}: max={max_val} > {thresholds['max']}, policy_type={policy_type}"
                        logger.warning(error_msg)
                        self.miya_console.miya_speak(
                            error_msg, tag="OBS_TEMPLATE", level="warning"
                        )
                        self.alert_manager.send_alert(error_msg, priority=3)
                        send_telegram_alert(error_msg)
                        return False

            success_msg = f"Validation du vecteur d’observation réussie pour {self.market}: 150 features, policy_type={policy_type}"
            logger.info(success_msg)
            self.miya_console.miya_speak(success_msg, tag="OBS_TEMPLATE")
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            latency = time.time() - start_time
            self.log_performance("validate_obs_template", latency, success=True)
            return True
        except Exception as e:
            error_msg = f"Erreur validation vecteur d’observation pour {self.market}: {str(e)}, policy_type={policy_type}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.miya_console.miya_alerts(
                error_msg, tag="OBS_TEMPLATE", voice_profile="urgent"
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "validate_obs_template",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            return False


if __name__ == "__main__":
    try:
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                **{f"feature_{i}": np.random.uniform(0, 1, 100) for i in range(350)},
                "rsi_14": np.random.uniform(0, 100, 100),
                "obi_score": np.random.uniform(-1, 1, 100),
                "gex": np.random.uniform(-1000, 1000, 100),
                "news_impact_score": np.random.uniform(0, 1, 100),
                "predicted_volatility": np.random.uniform(0, 2, 100),
                "neural_regime": np.random.randint(0, 3, 100),
                "cnn_pressure": np.random.uniform(-5, 5, 100),
                "key_strikes_1": np.random.uniform(5000, 5200, 100),
                "max_pain_strike": np.random.uniform(5000, 5200, 100),
                "net_gamma": np.random.uniform(-1, 1, 100),
                "zero_gamma": np.random.uniform(5000, 5200, 100),
                "dealer_zones_count": np.random.randint(0, 11, 100),
                "vol_trigger": np.random.uniform(-1, 1, 100),
                "ref_px": np.random.uniform(5000, 5200, 100),
                "data_release": np.random.randint(0, 2, 100),
            }
        )

        calculator = ObsTemplate(market="ES")
        for policy_type in ["mlp", "transformer"]:
            result = calculator.validate_obs_template(data, policy_type=policy_type)
            calculator.miya_console.miya_speak(
                f"Validation du vecteur d’observation pour policy_type={policy_type}: {result}",
                tag="TEST",
            )
            calculator.alert_manager.send_alert(
                f"Validation du vecteur d’observation pour policy_type={policy_type}: {result}",
                priority=1,
            )
            print(
                f"Validation du vecteur d’observation pour policy_type={policy_type}: {result}"
            )

        observation = calculator.create_observation(data, regime="trend")
        print(f"Observation vector shape: {observation.shape}")
    except Exception as e:
        calculator.miya_console.miya_alerts(
            f"Erreur test principal: {str(e)}", tag="ALERT", voice_profile="urgent"
        )
        calculator.alert_manager.send_alert(
            f"Erreur test principal: {str(e)}", priority=4
        )
        send_telegram_alert(f"Erreur test principal: {str(e)}")
        logger.error(f"Erreur test principal: {str(e)}")
        raise
