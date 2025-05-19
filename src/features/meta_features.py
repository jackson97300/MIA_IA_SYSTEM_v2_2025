# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/features/meta_features.py
# Gère les métriques d’auto-analyse et de mémoire à partir des données IQFeed.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Calcule les métriques d’auto-analyse (ex. : confidence_drop_rate, sgc_entropy),
#        avec cache intermédiaire, logs psutil, validation SHAP (méthode 17), et compatibilité top 150 SHAP.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, scipy>=1.9.0,<2.0.0, psutil>=5.9.0,<6.0.0,
#   pyyaml>=6.0.0,<7.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
# - src/model/utils/obs_template.py
# - src/data/data_provider.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/es_config.yaml
# - data/iqfeed/merged_data.csv
# - data/features/feature_importance.csv
#
# Outputs :
# - data/features/cache/meta_features/*.csv
# - data/logs/meta_features_performance.csv
# - data/meta_features_snapshots/*.json (option *.json.gz)
#
# Notes :
# - Conforme à structure.txt (version 2.1.3, 2025-05-13).
# - Utilise IQFeed via data_provider.py, avec retries (max 3, délai 2^attempt secondes).
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des métriques.
# - Intègre validation SHAP (Phase 17) pour assurer la conformité avec les top 150 SHAP features.
# - Sauvegarde des snapshots JSON non compressés par défaut avec option de compression gzip.
# - Envoie des alertes via AlertManager et telegram_alert pour les erreurs critiques et succès.
# - Tests unitaires disponibles dans tests/test_meta_features.py.
# - Conforme à la ligne rouge des top 150 SHAP features définie dans feature_pipeline.py.

import gzip
import hashlib
import json
import logging
import os
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import psutil
import yaml
from scipy.stats import entropy

from src.model.utils.alert_manager import send_alert
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.model.utils.obs_template import obs_t
from src.utils.telegram_alert import send_telegram_alert

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data", "features", "cache", "meta_features")
PERF_LOG_PATH = os.path.join(BASE_DIR, "data", "logs", "meta_features_performance.csv")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "meta_features_snapshots")
FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR, "data", "features", "feature_importance.csv"
)

# Création des répertoires
os.makedirs(os.path.join(BASE_DIR, "data", "logs"), exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Configuration du logging
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "data", "logs", "meta_features.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Constantes
MAX_RETRIES = 3
RETRY_DELAY = 2.0


class MetaFeatures:
    """Gère les métriques d’auto-analyse avec cache, logs, et validation SHAP."""

    def __init__(
        self, config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml")
    ):
        """Initialise le générateur de métriques d’auto-analyse."""
        self.log_buffer = []
        self.cache = {}
        try:
            self.config = self.load_config(config_path)
            os.makedirs(CACHE_DIR, exist_ok=True)
            os.makedirs(os.path.dirname(PERF_LOG_PATH), exist_ok=True)
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            miya_speak(
                "MetaFeatures initialisé", tag="META", voice_profile="calm", priority=2
            )
            send_alert("MetaFeatures initialisé", priority=2)
            send_telegram_alert("MetaFeatures initialisé")
            logger.info("MetaFeatures initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config_path": config_path}, compress=False)
        except Exception as e:
            error_msg = f"Erreur initialisation MetaFeatures: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="META", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            self.config = {
                "confidence_window": 10,
                "error_window": 20,
                "entropy_window": 20,
            }

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Charge la configuration depuis es_config.yaml."""

        def load_yaml():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            if "meta_features" not in config:
                raise ValueError("Clé 'meta_features' manquante dans la configuration")
            required_keys = ["confidence_window", "error_window", "entropy_window"]
            missing_keys = [
                key for key in required_keys if key not in config["meta_features"]
            ]
            if missing_keys:
                raise ValueError(
                    f"Clés manquantes dans 'meta_features': {missing_keys}"
                )
            return config["meta_features"]

        try:
            start_time = time.time()
            config = self.with_retries(load_yaml)
            cache_key = hashlib.sha256(str(config).encode()).hexdigest()
            self.cache[cache_key] = {"config": config, "timestamp": datetime.now()}
            if len(self.cache) > self.config.get("max_cache_size", 1000):
                self.cache.pop(next(iter(self.cache)))
            latency = time.time() - start_time
            miya_speak(
                "Configuration meta_features chargée",
                tag="META",
                voice_profile="calm",
                priority=2,
            )
            send_alert("Configuration meta_features chargée", priority=2)
            send_telegram_alert("Configuration meta_features chargée")
            logger.info("Configuration meta_features chargée")
            self.log_performance("load_config", latency, success=True)
            self.save_snapshot(
                "load_config", {"config_path": config_path}, compress=False
            )
            return config
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur chargement config: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="META", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("load_config", latency, success=False, error=str(e))
            raise

    def with_retries(
        self,
        func: callable,
        max_attempts: int = MAX_RETRIES,
        delay_base: float = RETRY_DELAY,
    ) -> Any:
        """Exécute une fonction avec retries (max 3, délai exponentiel)."""
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
                    latency = time.time() - start_time
                    self.log_performance(
                        f"retry_attempt_{attempt+1}",
                        latency,
                        success=False,
                        error=str(e),
                        attempt_number=attempt + 1,
                    )
                    miya_alerts(
                        f"Échec après {max_attempts} tentatives: {str(e)}",
                        tag="META",
                        voice_profile="urgent",
                        priority=4,
                    )
                    send_alert(
                        f"Échec après {max_attempts} tentatives: {str(e)}", priority=4
                    )
                    send_telegram_alert(
                        f"Échec après {max_attempts} tentatives: {str(e)}"
                    )
                    logger.error(f"Échec après {max_attempts} tentatives: {str(e)}")
                    raise
                delay = delay_base * (2**attempt)
                miya_speak(
                    f"Tentative {attempt+1} échouée, retry après {delay}s",
                    tag="META",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(
                    f"Tentative {attempt+1} échouée, retry après {delay}s", priority=3
                )
                send_telegram_alert(
                    f"Tentative {attempt+1} échouée, retry après {delay}s"
                )
                logger.warning(f"Tentative {attempt+1} échouée, retry après {delay}s")
                time.sleep(delay)

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ) -> None:
        """Journalise les performances des opérations critiques avec psutil."""
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
            cpu_usage = psutil.cpu_percent()  # % CPU
            if memory_usage > 1024:
                miya_alerts(
                    f"ALERTE: Utilisation mémoire élevée ({memory_usage:.2f} MB)",
                    tag="META",
                    voice_profile="urgent",
                    priority=5,
                )
                send_alert(
                    f"ALERTE: Utilisation mémoire élevée ({memory_usage:.2f} MB)",
                    priority=5,
                )
                send_telegram_alert(
                    f"ALERTE: Utilisation mémoire élevée ({memory_usage:.2f} MB)"
                )
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "latency": latency,
                "success": success,
                "error": error,
                "memory_usage_mb": memory_usage,
                "cpu_usage_percent": cpu_usage,
                **kwargs,
            }
            self.log_buffer.append(log_entry)
            if len(self.log_buffer) >= self.config.get("buffer_size", 100):
                log_df = pd.DataFrame(self.log_buffer)
                os.makedirs(os.path.dirname(PERF_LOG_PATH), exist_ok=True)

                def write_log():
                    if not os.path.exists(PERF_LOG_PATH):
                        log_df.to_csv(PERF_LOG_PATH, index=False, encoding="utf-8")
                    else:
                        log_df.to_csv(
                            PERF_LOG_PATH,
                            mode="a",
                            header=False,
                            index=False,
                            encoding="utf-8",
                        )

                self.with_retries(write_log)
                self.log_buffer = []
            logger.info(f"Performance journalisée pour {operation}. CPU: {cpu_usage}%")
        except Exception as e:
            miya_alerts(
                f"Erreur journalisation performance: {str(e)}",
                tag="META",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(f"Erreur journalisation performance: {str(e)}", priority=3)
            send_telegram_alert(f"Erreur journalisation performance: {str(e)}")
            logger.error(f"Erreur journalisation performance: {str(e)}")

    def save_snapshot(
        self, snapshot_type: str, data: Dict, compress: bool = False
    ) -> None:
        """Sauvegarde un instantané des résultats avec option de compression gzip."""
        try:
            start_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = {
                "timestamp": timestamp,
                "type": snapshot_type,
                "data": data,
                "buffer_size": len(self.log_buffer),
            }
            snapshot_path = os.path.join(
                SNAPSHOT_DIR, f"snapshot_{snapshot_type}_{timestamp}.json"
            )
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)

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
                miya_alerts(
                    f"Snapshot size {file_size:.2f} MB exceeds 1 MB",
                    tag="META",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(f"Snapshot size {file_size:.2f} MB exceeds 1 MB", priority=3)
                send_telegram_alert(f"Snapshot size {file_size:.2f} MB exceeds 1 MB")
            latency = time.time() - start_time
            miya_speak(
                f"Snapshot {snapshot_type} sauvegardé: {save_path}",
                tag="META",
                voice_profile="calm",
                priority=1,
            )
            send_alert(f"Snapshot {snapshot_type} sauvegardé", priority=1)
            send_telegram_alert(f"Snapshot {snapshot_type} sauvegardé")
            logger.info(f"Snapshot {snapshot_type} sauvegardé: {save_path}")
            self.log_performance(
                "save_snapshot",
                latency,
                success=True,
                snapshot_type=snapshot_type,
                file_size_mb=file_size,
            )
        except Exception as e:
            self.log_performance("save_snapshot", 0, success=False, error=str(e))
            miya_alerts(
                f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}",
                tag="META",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}", priority=3
            )
            send_telegram_alert(f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}")
            logger.error(f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}")

    def validate_shap_features(self, features: List[str]) -> bool:
        """Valide que les features sont dans les top 150 SHAP."""
        try:
            start_time = time.time()
            if not os.path.exists(FEATURE_IMPORTANCE_PATH):
                miya_alerts(
                    "Fichier SHAP manquant",
                    tag="META",
                    voice_profile="urgent",
                    priority=4,
                )
                send_alert("Fichier SHAP manquant", priority=4)
                send_telegram_alert("Fichier SHAP manquant")
                logger.error("Fichier SHAP manquant")
                return False
            shap_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
            if len(shap_df) < 150:
                miya_alerts(
                    f"Nombre insuffisant de SHAP features: {len(shap_df)} < 150",
                    tag="META",
                    voice_profile="urgent",
                    priority=4,
                )
                send_alert(
                    f"Nombre insuffisant de SHAP features: {len(shap_df)}", priority=4
                )
                send_telegram_alert(
                    f"Nombre insuffisant de SHAP features: {len(shap_df)}"
                )
                logger.error(f"Nombre insuffisant de SHAP features: {len(shap_df)}")
                return False
            valid_features = set(shap_df["feature"].head(150)).union(obs_t)
            missing = [f for f in features if f not in valid_features]
            if missing:
                miya_alerts(
                    f"Features non incluses dans top 150 SHAP ou obs_t: {missing}",
                    tag="META",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(
                    f"Features non incluses dans top 150 SHAP ou obs_t: {missing}",
                    priority=3,
                )
                send_telegram_alert(
                    f"Features non incluses dans top 150 SHAP ou obs_t: {missing}"
                )
                logger.warning(
                    f"Features non incluses dans top 150 SHAP ou obs_t: {missing}"
                )
            latency = time.time() - start_time
            miya_speak(
                "SHAP features validées", tag="META", voice_profile="calm", priority=1
            )
            send_alert("SHAP features validées", priority=1)
            send_telegram_alert("SHAP features validées")
            logger.info("SHAP features validées")
            self.log_performance(
                "validate_shap_features",
                latency,
                success=True,
                num_features=len(features),
            )
            self.save_snapshot(
                "validate_shap_features",
                {"num_features": len(features), "missing": missing},
                compress=False,
            )
            return True
        except Exception as e:
            latency = time.time() - start_time
            miya_alerts(
                f"Erreur validation SHAP features: {str(e)}",
                tag="META",
                voice_profile="urgent",
                priority=4,
            )
            send_alert(f"Erreur validation SHAP features: {str(e)}", priority=4)
            send_telegram_alert(f"Erreur validation SHAP features: {str(e)}")
            logger.error(f"Erreur validation SHAP features: {str(e)}")
            self.log_performance(
                "validate_shap_features", latency, success=False, error=str(e)
            )
            return False

    def calculate_confidence_drop_rate(
        self, data: pd.DataFrame, window: int = 10
    ) -> pd.Series:
        """Calcule le taux de diminution de la confiance dans les prédictions."""
        cache_path = os.path.join(
            CACHE_DIR,
            f"confidence_drop_rate_{hashlib.sha256(data.to_json().encode()).hexdigest()}.csv",
        )
        try:
            start_time = time.time()
            if os.path.exists(cache_path):
                cached_data = pd.read_csv(cache_path)
                if len(cached_data) == len(data) and cached_data["timestamp"].equals(
                    data["timestamp"]
                ):
                    latency = time.time() - start_time
                    self.log_performance(
                        "calculate_confidence_drop_rate_cache", latency, success=True
                    )
                    miya_speak(
                        "Confidence drop rate chargé depuis cache",
                        tag="META",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert("Confidence drop rate chargé depuis cache", priority=1)
                    send_telegram_alert("Confidence drop rate chargé depuis cache")
                    return cached_data["confidence_drop_rate"]

            required_cols = ["confidence", "timestamp"]
            missing_cols = [col for col in required_cols if col not in data.columns]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )
            if len(data) < window:
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(required_cols) - len(missing_cols)}/{len(required_cols)} colonnes valides, {len(data)} lignes)"
                miya_alerts(alert_msg, tag="META", voice_profile="urgent", priority=3)
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            if "confidence" not in data.columns:
                raise ValueError("Colonne 'confidence' manquante")
            confidence_change = data["confidence"].diff()
            drop_rate = confidence_change.rolling(window=window, min_periods=1).mean()
            result = drop_rate.clip(-1, 1).fillna(0)

            os.makedirs(CACHE_DIR, exist_ok=True)

            def write_cache():
                cache_df = pd.DataFrame(
                    {"timestamp": data["timestamp"], "confidence_drop_rate": result}
                )
                cache_df.to_csv(cache_path, index=False, encoding="utf-8")

            self.with_retries(write_cache)
            file_size = os.path.getsize(cache_path) / 1024 / 1024
            if file_size > 1.0:
                miya_alerts(
                    f"Cache confidence_drop_rate size {file_size:.2f} MB exceeds 1 MB",
                    tag="META",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(
                    f"Cache confidence_drop_rate size {file_size:.2f} MB exceeds 1 MB",
                    priority=3,
                )
                send_telegram_alert(
                    f"Cache confidence_drop_rate size {file_size:.2f} MB exceeds 1 MB"
                )

            latency = time.time() - start_time
            self.log_performance(
                "calculate_confidence_drop_rate",
                latency,
                success=True,
                confidence_drop_rate=confidence_drop_rate,
            )
            miya_speak(
                "Confidence drop rate calculé",
                tag="META",
                voice_profile="calm",
                priority=1,
            )
            send_alert("Confidence drop rate calculé", priority=1)
            send_telegram_alert("Confidence drop rate calculé")
            logger.info("Confidence drop rate calculé")
            self.save_snapshot(
                "calculate_confidence_drop_rate", {"window": window}, compress=False
            )
            return result
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans calculate_confidence_drop_rate: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                "calculate_confidence_drop_rate", latency, success=False, error=str(e)
            )
            miya_alerts(error_msg, tag="META", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.save_snapshot(
                "calculate_confidence_drop_rate", {"error": str(e)}, compress=False
            )
            return pd.Series(0.0, index=data.index)

    def calculate_error_rolling_std(
        self, data: pd.DataFrame, window: int = 20
    ) -> pd.Series:
        """Calcule l’écart-type glissant des erreurs de prédiction."""
        cache_path = os.path.join(
            CACHE_DIR,
            f"error_rolling_std_{hashlib.sha256(data.to_json().encode()).hexdigest()}.csv",
        )
        try:
            start_time = time.time()
            if os.path.exists(cache_path):
                cached_data = pd.read_csv(cache_path)
                if len(cached_data) == len(data) and cached_data["timestamp"].equals(
                    data["timestamp"]
                ):
                    latency = time.time() - start_time
                    self.log_performance(
                        "calculate_error_rolling_std_cache", latency, success=True
                    )
                    miya_speak(
                        "Error rolling std chargé depuis cache",
                        tag="META",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert("Error rolling std chargé depuis cache", priority=1)
                    send_telegram_alert("Error rolling std chargé depuis cache")
                    return cached_data["error_rolling_std"]

            required_cols = ["close", "predicted_price", "timestamp"]
            missing_cols = [col for col in required_cols if col not in data.columns]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )
            if len(data) < window:
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(required_cols) - len(missing_cols)}/{len(required_cols)} colonnes valides, {len(data)} lignes)"
                miya_alerts(alert_msg, tag="META", voice_profile="urgent", priority=3)
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            if "close" not in data.columns or "predicted_price" not in data.columns:
                raise ValueError("Colonnes 'close' ou 'predicted_price' manquantes")
            error = data["close"] - data["predicted_price"]
            error_std = error.rolling(window=window, min_periods=1).std()
            result = error_std.fillna(0)

            os.makedirs(CACHE_DIR, exist_ok=True)

            def write_cache():
                cache_df = pd.DataFrame(
                    {"timestamp": data["timestamp"], "error_rolling_std": result}
                )
                cache_df.to_csv(cache_path, index=False, encoding="utf-8")

            self.with_retries(write_cache)
            file_size = os.path.getsize(cache_path) / 1024 / 1024
            if file_size > 1.0:
                miya_alerts(
                    f"Cache error_rolling_std size {file_size:.2f} MB exceeds 1 MB",
                    tag="META",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(
                    f"Cache error_rolling_std size {file_size:.2f} MB exceeds 1 MB",
                    priority=3,
                )
                send_telegram_alert(
                    f"Cache error_rolling_std size {file_size:.2f} MB exceeds 1 MB"
                )

            latency = time.time() - start_time
            self.log_performance(
                "calculate_error_rolling_std",
                latency,
                success=True,
                confidence_drop_rate=confidence_drop_rate,
            )
            miya_speak(
                "Error rolling std calculé",
                tag="META",
                voice_profile="calm",
                priority=1,
            )
            send_alert("Error rolling std calculé", priority=1)
            send_telegram_alert("Error rolling std calculé")
            logger.info("Error rolling std calculé")
            self.save_snapshot(
                "calculate_error_rolling_std", {"window": window}, compress=False
            )
            return result
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans calculate_error_rolling_std: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                "calculate_error_rolling_std", latency, success=False, error=str(e)
            )
            miya_alerts(error_msg, tag="META", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.save_snapshot(
                "calculate_error_rolling_std", {"error": str(e)}, compress=False
            )
            return pd.Series(0.0, index=data.index)

    def calculate_sgc_entropy(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calcule l’entropie des signaux générés par le système."""
        cache_path = os.path.join(
            CACHE_DIR,
            f"sgc_entropy_{hashlib.sha256(data.to_json().encode()).hexdigest()}.csv",
        )
        try:
            start_time = time.time()
            if os.path.exists(cache_path):
                cached_data = pd.read_csv(cache_path)
                if len(cached_data) == len(data) and cached_data["timestamp"].equals(
                    data["timestamp"]
                ):
                    latency = time.time() - start_time
                    self.log_performance(
                        "calculate_sgc_entropy_cache", latency, success=True
                    )
                    miya_speak(
                        "SGC entropy chargé depuis cache",
                        tag="META",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert("SGC entropy chargé depuis cache", priority=1)
                    send_telegram_alert("SGC entropy chargé depuis cache")
                    return cached_data["sgc_entropy"]

            required_cols = ["predicted_price", "timestamp"]
            missing_cols = [col for col in required_cols if col not in data.columns]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )
            if len(data) < window:
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(required_cols) - len(missing_cols)}/{len(required_cols)} colonnes valides, {len(data)} lignes)"
                miya_alerts(alert_msg, tag="META", voice_profile="urgent", priority=3)
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            if "predicted_price" not in data.columns:
                raise ValueError("Colonne 'predicted_price' manquante")

            def compute_entropy(window_data):
                if len(window_data) < 2 or np.std(window_data) < 1e-6:
                    return 0.0
                bins = min(10, max(2, len(window_data) // 5))
                hist, _ = np.histogram(window_data, bins=bins, density=True)
                hist = hist[hist > 0]
                return entropy(hist) if len(hist) > 0 else 0.0

            entropy_series = (
                data["predicted_price"]
                .rolling(window=window, min_periods=1)
                .apply(compute_entropy, raw=True)
            )
            result = entropy_series.fillna(0).clip(lower=0)

            os.makedirs(CACHE_DIR, exist_ok=True)

            def write_cache():
                cache_df = pd.DataFrame(
                    {"timestamp": data["timestamp"], "sgc_entropy": result}
                )
                cache_df.to_csv(cache_path, index=False, encoding="utf-8")

            self.with_retries(write_cache)
            file_size = os.path.getsize(cache_path) / 1024 / 1024
            if file_size > 1.0:
                miya_alerts(
                    f"Cache sgc_entropy size {file_size:.2f} MB exceeds 1 MB",
                    tag="META",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(
                    f"Cache sgc_entropy size {file_size:.2f} MB exceeds 1 MB",
                    priority=3,
                )
                send_telegram_alert(
                    f"Cache sgc_entropy size {file_size:.2f} MB exceeds 1 MB"
                )

            latency = time.time() - start_time
            self.log_performance(
                "calculate_sgc_entropy",
                latency,
                success=True,
                confidence_drop_rate=confidence_drop_rate,
            )
            miya_speak(
                "SGC entropy calculé", tag="META", voice_profile="calm", priority=1
            )
            send_alert("SGC entropy calculé", priority=1)
            send_telegram_alert("SGC entropy calculé")
            logger.info("SGC entropy calculé")
            self.save_snapshot(
                "calculate_sgc_entropy", {"window": window}, compress=False
            )
            return result
        except Exception as e:
            latency = time.time() - start_time
            error_msg = (
                f"Erreur dans calculate_sgc_entropy: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance(
                "calculate_sgc_entropy", latency, success=False, error=str(e)
            )
            miya_alerts(error_msg, tag="META", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.save_snapshot(
                "calculate_sgc_entropy", {"error": str(e)}, compress=False
            )
            return pd.Series(0.0, index=data.index)

    def calculate_meta_features(
        self,
        data: pd.DataFrame,
        config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml"),
    ) -> pd.DataFrame:
        """Calcule les métriques d’auto-analyse et de mémoire à partir des données IQFeed."""
        cache_path = os.path.join(
            CACHE_DIR,
            f"meta_features_{hashlib.sha256(data.to_json().encode()).hexdigest()}.csv",
        )
        try:
            start_time = time.time()
            current_time = datetime.now()
            expired_keys = [
                k
                for k, v in self.cache.items()
                if (current_time - v["timestamp"]).total_seconds()
                > self.config.get("cache_hours", 24) * 3600
            ]
            for k in expired_keys:
                self.cache.pop(k)
            if os.path.exists(cache_path):
                cached_data = pd.read_csv(cache_path)
                if len(cached_data) == len(data) and cached_data["timestamp"].equals(
                    data["timestamp"]
                ):
                    latency = time.time() - start_time
                    self.log_performance(
                        "calculate_meta_features_cache_hit", latency, success=True
                    )
                    miya_speak(
                        "Métriques d’auto-analyse chargées depuis cache",
                        tag="META",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert(
                        "Métriques d’auto-analyse chargées depuis cache", priority=1
                    )
                    send_telegram_alert(
                        "Métriques d’auto-analyse chargées depuis cache"
                    )
                    return cached_data

            config = self.load_config(config_path)
            confidence_window = config.get("confidence_window", 10)
            error_window = config.get("error_window", 20)
            entropy_window = config.get("entropy_window", 20)

            if data.empty:
                error_msg = "DataFrame vide"
                miya_alerts(error_msg, tag="META", voice_profile="urgent", priority=5)
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            required_cols = ["timestamp", "close"]
            optional_cols = ["predicted_price", "confidence"]
            missing_required_cols = [
                col for col in required_cols if col not in data.columns
            ]
            missing_optional_cols = [
                col for col in optional_cols if col not in data.columns
            ]
            confidence_drop_rate = 1.0 - min(
                (
                    len(required_cols)
                    + len(optional_cols)
                    - len(missing_required_cols)
                    - len(missing_optional_cols)
                )
                / (len(required_cols) + len(optional_cols)),
                1.0,
            )
            if len(data) < min(confidence_window, error_window, entropy_window):
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(required_cols) + len(optional_cols) - len(missing_required_cols) - len(missing_optional_cols)}/{len(required_cols) + len(optional_cols)} colonnes valides, {len(data)} lignes)"
                miya_alerts(alert_msg, tag="META", voice_profile="urgent", priority=3)
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            data = data.copy()
            for col in missing_required_cols:
                if col == "timestamp":
                    data[col] = pd.date_range(
                        start=pd.Timestamp.now(), periods=len(data), freq="1min"
                    )
                    miya_speak(
                        "Colonne timestamp manquante, création par défaut",
                        tag="META",
                        voice_profile="warning",
                        priority=2,
                    )
                    send_alert(
                        "Colonne timestamp manquante, création par défaut", priority=2
                    )
                    send_telegram_alert(
                        "Colonne timestamp manquante, création par défaut"
                    )
                else:
                    data[col] = (
                        data[col]
                        .interpolate(method="linear", limit_direction="both")
                        .fillna(data[col].median() if col in data.columns else 0)
                    )
                    miya_speak(
                        f"Colonne '{col}' manquante, imputée",
                        tag="META",
                        voice_profile="warning",
                        priority=2,
                    )
                    send_alert(f"Colonne '{col}' manquante, imputée", priority=2)
                    send_telegram_alert(f"Colonne '{col}' manquante, imputée")
            for col in missing_optional_cols:
                if col == "predicted_price":
                    data[col] = (
                        data["close"]
                        .rolling(window=5, min_periods=1)
                        .mean()
                        .fillna(data["close"])
                    )
                    miya_speak(
                        "Colonne predicted_price manquante, simulée avec moyenne glissante",
                        tag="META",
                        voice_profile="warning",
                        priority=2,
                    )
                    send_alert(
                        "Colonne predicted_price manquante, simulée avec moyenne glissante",
                        priority=2,
                    )
                    send_telegram_alert(
                        "Colonne predicted_price manquante, simulée avec moyenne glissante"
                    )
                elif col == "confidence":
                    data[col] = 0.5
                    miya_speak(
                        "Colonne confidence manquante, imputée à 0.5",
                        tag="META",
                        voice_profile="warning",
                        priority=2,
                    )
                    send_alert(
                        "Colonne confidence manquante, imputée à 0.5", priority=2
                    )
                    send_telegram_alert("Colonne confidence manquante, imputée à 0.5")

            data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
            if data["timestamp"].isna().any():
                error_msg = "NaN dans les timestamps"
                miya_alerts(error_msg, tag="META", voice_profile="urgent", priority=5)
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)
            if not data["timestamp"].is_monotonic_increasing:
                error_msg = "Timestamps non croissants"
                miya_alerts(error_msg, tag="META", voice_profile="urgent", priority=5)
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            features = [
                "confidence_drop_rate",
                "error_rolling_std",
                "sgc_entropy",
                "prediction_bias",
                "error_trend",
                "confidence_volatility",
                "signal_stability",
                "memory_retention_score",
            ]
            missing_features = [f for f in features if f not in obs_t]
            if missing_features:
                miya_alerts(
                    f"Features manquantes dans obs_t: {missing_features}",
                    tag="META",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(
                    f"Features manquantes dans obs_t: {missing_features}", priority=3
                )
                send_telegram_alert(
                    f"Features manquantes dans obs_t: {missing_features}"
                )
                logger.warning(f"Features manquantes dans obs_t: {missing_features}")
            self.validate_shap_features(features)

            for feature in features:
                data[feature] = 0.0

            data["confidence_drop_rate"] = self.calculate_confidence_drop_rate(
                data, confidence_window
            )
            data["error_rolling_std"] = self.calculate_error_rolling_std(
                data, error_window
            )
            data["sgc_entropy"] = self.calculate_sgc_entropy(data, entropy_window)

            error = data["close"] - data["predicted_price"]
            data["prediction_bias"] = error.rolling(
                window=error_window, min_periods=1
            ).mean()
            data["error_trend"] = (
                error.diff().rolling(window=error_window, min_periods=1).mean()
            )
            data["confidence_volatility"] = (
                data["confidence"]
                .rolling(window=confidence_window, min_periods=1)
                .std()
            )
            data["signal_stability"] = (
                data["predicted_price"]
                .diff()
                .rolling(window=entropy_window, min_periods=1)
                .std()
            )
            data["memory_retention_score"] = error.rolling(
                window=error_window, min_periods=1
            ).mean() / (error.rolling(window=error_window, min_periods=1).std() + 1e-6)

            for feature in features:
                if data[feature].isna().any():
                    data[feature] = data[feature].fillna(0)
                if feature == "sgc_entropy" and data[feature].min() < 0:
                    miya_alerts(
                        f"Valeurs négatives dans {feature}",
                        tag="META",
                        voice_profile="urgent",
                        priority=3,
                    )
                    send_alert(f"Valeurs négatives dans {feature}", priority=3)
                    send_telegram_alert(f"Valeurs négatives dans {feature}")
                    data[feature] = data[feature].clip(lower=0)

            os.makedirs(CACHE_DIR, exist_ok=True)

            def write_cache():
                data.to_csv(cache_path, index=False, encoding="utf-8")

            self.with_retries(write_cache)
            file_size = os.path.getsize(cache_path) / 1024 / 1024
            if file_size > 1.0:
                miya_alerts(
                    f"Cache meta_features size {file_size:.2f} MB exceeds 1 MB",
                    tag="META",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(
                    f"Cache meta_features size {file_size:.2f} MB exceeds 1 MB",
                    priority=3,
                )
                send_telegram_alert(
                    f"Cache meta_features size {file_size:.2f} MB exceeds 1 MB"
                )

            latency = time.time() - start_time
            self.log_performance(
                "calculate_meta_features",
                latency,
                success=True,
                num_rows=len(data),
                num_features=len(features),
                confidence_drop_rate=confidence_drop_rate,
            )
            miya_speak(
                "Métriques d’auto-analyse calculées",
                tag="META",
                voice_profile="calm",
                priority=1,
            )
            send_alert("Métriques d’auto-analyse calculées", priority=1)
            send_telegram_alert("Métriques d’auto-analyse calculées")
            logger.info("Métriques d’auto-analyse calculées")
            self.save_snapshot(
                "meta_features",
                {
                    "num_rows": len(data),
                    "num_features": len(features),
                    "confidence_drop_rate": confidence_drop_rate,
                },
                compress=False,
            )
            return data

        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans calculate_meta_features: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                "calculate_meta_features", latency, success=False, error=str(e)
            )
            miya_alerts(error_msg, tag="META", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.save_snapshot("meta_features", {"error": str(e)}, compress=False)
            for feature in features:
                data[feature] = 0.0
            return data

    def calculate_incremental_meta_features(
        self,
        row: pd.Series,
        buffer: pd.DataFrame,
        confidence_window: int = 10,
        error_window: int = 20,
        entropy_window: int = 20,
    ) -> pd.Series:
        """Calcule les métriques d’auto-analyse pour une seule ligne en temps réel."""
        try:
            start_time = time.time()
            features = {
                f: 0.0
                for f in [
                    "confidence_drop_rate",
                    "error_rolling_std",
                    "sgc_entropy",
                    "prediction_bias",
                    "error_trend",
                    "confidence_volatility",
                    "signal_stability",
                    "memory_retention_score",
                ]
            }

            buffer = pd.concat([buffer, row.to_frame().T], ignore_index=True)
            if len(buffer) < min(confidence_window, error_window, entropy_window):
                return pd.Series(features)

            required_cols = ["timestamp", "close", "predicted_price", "confidence"]
            missing_cols = [col for col in required_cols if col not in buffer.columns]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )
            if len(buffer) < min(confidence_window, error_window, entropy_window):
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(required_cols) - len(missing_cols)}/{len(required_cols)} colonnes valides, {len(buffer)} lignes)"
                miya_alerts(alert_msg, tag="META", voice_profile="urgent", priority=3)
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            confidence_change = buffer["confidence"].diff()
            features["confidence_drop_rate"] = confidence_change.tail(
                confidence_window
            ).mean()
            error = buffer["close"] - buffer["predicted_price"]
            features["error_rolling_std"] = error.tail(error_window).std()
            window_data = buffer["predicted_price"].tail(entropy_window)
            if len(window_data) >= 2 and np.std(window_data) >= 1e-6:
                bins = min(10, max(2, len(window_data) // 5))
                hist, _ = np.histogram(window_data, bins=bins, density=True)
                hist = hist[hist > 0]
                features["sgc_entropy"] = entropy(hist) if len(hist) > 0 else 0.0
            features["prediction_bias"] = error.tail(error_window).mean()
            features["error_trend"] = error.diff().tail(error_window).mean()
            features["confidence_volatility"] = (
                buffer["confidence"].tail(confidence_window).std()
            )
            features["signal_stability"] = (
                buffer["predicted_price"].diff().tail(entropy_window).std()
            )
            features["memory_retention_score"] = error.tail(error_window).mean() / (
                error.tail(error_window).std() + 1e-6
            )

            result = pd.Series(
                {k: v if not np.isnan(v) else 0.0 for k, v in features.items()}
            )

            latency = time.time() - start_time
            self.log_performance(
                "calculate_incremental_meta_features",
                latency,
                success=True,
                num_features=len(features),
                confidence_drop_rate=confidence_drop_rate,
            )
            miya_speak(
                "Métriques d’auto-analyse incrémentales calculées",
                tag="META",
                voice_profile="calm",
                priority=1,
            )
            send_alert("Métriques d’auto-analyse incrémentales calculées", priority=1)
            send_telegram_alert("Métriques d’auto-analyse incrémentales calculées")
            logger.info("Métriques d’auto-analyse incrémentales calculées")
            self.save_snapshot(
                "incremental_meta_features",
                {
                    "features": {k: float(v) for k, v in features.items()},
                    "confidence_drop_rate": confidence_drop_rate,
                },
                compress=False,
            )
            return result

        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans calculate_incremental_meta_features: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                "calculate_incremental_meta_features",
                latency,
                success=False,
                error=str(e),
            )
            miya_alerts(error_msg, tag="META", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.save_snapshot(
                "incremental_meta_features", {"error": str(e)}, compress=False
            )
            return pd.Series(features)


if __name__ == "__main__":
    try:
        calculator = MetaFeatures()
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "close": np.random.normal(5100, 10, 100),
                "predicted_price": np.random.normal(5100, 12, 100),
                "confidence": np.random.uniform(0.4, 0.9, 100),
            }
        )
        result = calculator.calculate_meta_features(data)
        print(
            result[
                [
                    "timestamp",
                    "confidence_drop_rate",
                    "error_rolling_std",
                    "sgc_entropy",
                    "prediction_bias",
                ]
            ].head()
        )
        miya_speak(
            "Test calculate_meta_features terminé",
            tag="TEST",
            voice_profile="calm",
            priority=1,
        )
        send_alert("Test calculate_meta_features terminé", priority=1)
        send_telegram_alert("Test calculate_meta_features terminé")
        logger.info("Test calculate_meta_features terminé")
    except Exception as e:
        miya_alerts(
            f"Erreur test: {str(e)}\n{traceback.format_exc()}",
            tag="ALERT",
            voice_profile="urgent",
            priority=3,
        )
        send_alert(f"Erreur test: {str(e)}", priority=3)
        send_telegram_alert(f"Erreur test: {str(e)}")
        logger.error(f"Erreur test: {str(e)}")
        raise
