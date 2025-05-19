# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/features/advanced_feature_generator.py
# Gère les features de microstructure et options avancées à partir des données IQFeed.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Calcule les features avancées (latency_spread, iv_atm, option_skew, gex_slope, etc.),
#        avec cache intermédiaire, logs psutil, alertes, validation SHAP (méthode 17), et compatibilité top 150 SHAP features.
#        Intègre iv_atm, option_skew, gex_slope comme métriques distinctes des 59 nouvelles features
#        de feature_pipeline.py (après suppression de volatility_skew, iv_slope_10_25, gamma_net_spread).
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.0,<6.0.0, pyyaml>=6.0.0,<7.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
# - src/model/utils/obs_template.py
# - src/data/data_provider.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/es_config.yaml
# - data/iqfeed/option_chain.csv
# - data/features/feature_importance.csv
#
# Outputs :
# - data/logs/advanced_feature_generator.log
# - data/logs/advanced_features_performance.csv
# - data/features/cache/advanced/*.csv
# - data/advanced_snapshots/*.json (option *.json.gz)
#
# Notes :
# - Conforme à structure.txt (version 2.1.3, 2025-05-13).
# - Utilise IQFeed via data_provider.py, avec retries (max 3, délai 2^attempt secondes).
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des métriques.
# - Intègre validation SHAP (Phase 17) pour assurer la conformité avec les top 150 SHAP features.
# - Sauvegarde des snapshots JSON non compressés par défaut avec option de compression gzip.
# - Envoie des alertes via AlertManager et telegram_alert pour les erreurs critiques et succès.
# - Tests unitaires disponibles dans tests/test_advanced_feature_generator.py.
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

from src.data.data_provider import get_data_provider
from src.model.utils.alert_manager import send_alert
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.model.utils.obs_template import obs_t
from src.utils.telegram_alert import send_telegram_alert

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data", "features", "cache", "advanced")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "advanced_snapshots")
PERF_LOG_PATH = os.path.join(
    BASE_DIR, "data", "logs", "advanced_features_performance.csv"
)
FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR, "data", "features", "feature_importance.csv"
)

# Création des répertoires
os.makedirs(os.path.join(BASE_DIR, "data", "logs"), exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Configuration du logging
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "data", "logs", "advanced_feature_generator.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Constantes
MAX_RETRIES = 3
RETRY_DELAY = 2.0


class AdvancedFeatureGenerator:
    """Gère le calcul des features avancées avec cache, logs, et alertes."""

    def __init__(
        self, config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml")
    ):
        """Initialise l’engine de génération des features avancées."""
        self.log_buffer = []
        self.cache = {}
        try:
            self.config = self.load_config(config_path)
            os.makedirs(CACHE_DIR, exist_ok=True)
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            os.makedirs(os.path.dirname(PERF_LOG_PATH), exist_ok=True)
            miya_speak(
                "AdvancedFeatureGenerator initialisé",
                tag="ADVANCED",
                voice_profile="calm",
                priority=2,
            )
            send_alert("AdvancedFeatureGenerator initialisé", priority=2)
            send_telegram_alert("AdvancedFeatureGenerator initialisé")
            logger.info("AdvancedFeatureGenerator initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config_path": config_path}, compress=False)
        except Exception as e:
            error_msg = f"Erreur initialisation AdvancedFeatureGenerator: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="ADVANCED", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            self.config = {
                "window_size": 5,
                "min_option_rows": 10,
                "buffer_size": 100,
                "max_cache_size": 1000,
                "cache_hours": 24,
            }

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Charge la configuration depuis es_config.yaml."""

        def load_yaml():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            if "advanced_feature_generator" not in config:
                raise ValueError(
                    "Clé 'advanced_feature_generator' manquante dans la configuration"
                )
            required_keys = ["window_size", "min_option_rows"]
            missing_keys = [
                key
                for key in required_keys
                if key not in config["advanced_feature_generator"]
            ]
            if missing_keys:
                raise ValueError(
                    f"Clés manquantes dans 'advanced_feature_generator': {missing_keys}"
                )
            return config["advanced_feature_generator"]

        try:
            start_time = time.time()
            config = self.with_retries(load_yaml)
            cache_key = hashlib.sha256(str(config).encode()).hexdigest()
            self.cache[cache_key] = {"config": config, "timestamp": datetime.now()}
            if len(self.cache) > self.config.get("max_cache_size", 1000):
                self.cache.pop(next(iter(self.cache)))
            latency = time.time() - start_time
            miya_speak(
                "Configuration advanced_feature_generator chargée",
                tag="ADVANCED",
                voice_profile="calm",
                priority=2,
            )
            send_alert("Configuration advanced_feature_generator chargée", priority=2)
            send_telegram_alert("Configuration advanced_feature_generator chargée")
            logger.info("Configuration advanced_feature_generator chargée")
            self.log_performance("load_config", latency, success=True)
            self.save_snapshot(
                "load_config", {"config_path": config_path}, compress=False
            )
            return config
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur chargement config: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="ADVANCED", voice_profile="urgent", priority=3)
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
                        tag="ADVANCED",
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
                    tag="ADVANCED",
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
                    tag="ADVANCED",
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
                tag="ADVANCED",
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
                    tag="ADVANCED",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(f"Snapshot size {file_size:.2f} MB exceeds 1 MB", priority=3)
                send_telegram_alert(f"Snapshot size {file_size:.2f} MB exceeds 1 MB")
            latency = time.time() - start_time
            miya_speak(
                f"Snapshot {snapshot_type} sauvegardé: {save_path}",
                tag="ADVANCED",
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
                tag="ADVANCED",
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
                    tag="ADVANCED",
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
                    tag="ADVANCED",
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
                    tag="ADVANCED",
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
                "SHAP features validées",
                tag="ADVANCED",
                voice_profile="calm",
                priority=1,
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
                tag="ADVANCED",
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

    def load_option_chain(
        self,
        option_chain_path: str = os.path.join(
            BASE_DIR, "data", "iqfeed", "option_chain.csv"
        ),
    ) -> pd.DataFrame:
        """Charge la chaîne d’options via data_provider.py avec retries."""
        try:
            start_time = time.time()
            required_cols = [
                "timestamp",
                "underlying_price",
                "strike",
                "option_type",
                "implied_volatility",
                "open_interest",
                "vanna",
                "gamma",
                "expiration_date",
            ]
            missing_cols = []
            confidence_drop_rate = 0.0

            def fetch_data():
                nonlocal missing_cols
                provider = get_data_provider("iqfeed")
                data = provider.fetch_option_chain(option_chain_path)
                if data.empty:
                    raise ValueError("Aucune donnée d’options chargée depuis IQFeed")
                missing_cols = [col for col in required_cols if col not in data.columns]
                confidence_drop_rate = 1.0 - min(
                    (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
                )
                for col in missing_cols:
                    data[col] = 0
                    miya_speak(
                        f"Colonne '{col}' manquante dans option_chain, imputée à 0",
                        tag="ADVANCED",
                        voice_profile="warning",
                        priority=2,
                    )
                    send_alert(
                        f"Colonne '{col}' manquante dans option_chain, imputée à 0",
                        priority=2,
                    )
                    send_telegram_alert(
                        f"Colonne '{col}' manquante dans option_chain, imputée à 0"
                    )
                data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
                if data["timestamp"].isna().any():
                    raise ValueError("NaN dans les timestamps d’option_chain")
                return data

            data = self.with_retries(fetch_data)
            latency = time.time() - start_time
            self.log_performance(
                "load_option_chain",
                latency,
                success=True,
                num_rows=len(data),
                confidence_drop_rate=confidence_drop_rate,
            )
            self.save_snapshot(
                "load_option_chain",
                {
                    "num_rows": len(data),
                    "columns": list(data.columns),
                    "confidence_drop_rate": confidence_drop_rate,
                },
                compress=False,
            )
            miya_speak(
                f"Chaîne d’options chargée: {len(data)} lignes",
                tag="ADVANCED",
                voice_profile="calm",
                priority=1,
            )
            send_alert(f"Chaîne d’options chargée: {len(data)} lignes", priority=1)
            send_telegram_alert(f"Chaîne d’options chargée: {len(data)} lignes")
            logger.info(f"Chaîne d’options chargée: {len(data)} lignes")
            return data
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur chargement option_chain via IQFeed: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                "load_option_chain", latency, success=False, error=str(e)
            )
            miya_alerts(error_msg, tag="ADVANCED", voice_profile="urgent", priority=4)
            send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            try:
                data = pd.read_csv(option_chain_path)
                missing_cols = [col for col in required_cols if col not in data.columns]
                confidence_drop_rate = 1.0 - min(
                    (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
                )
                for col in missing_cols:
                    data[col] = 0
                    miya_speak(
                        f"Colonne '{col}' manquante dans option_chain (fallback), imputée à 0",
                        tag="ADVANCED",
                        voice_profile="warning",
                        priority=2,
                    )
                    send_alert(
                        f"Colonne '{col}' manquante dans option_chain (fallback), imputée à 0",
                        priority=2,
                    )
                    send_telegram_alert(
                        f"Colonne '{col}' manquante dans option_chain (fallback), imputée à 0"
                    )
                data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
                if data["timestamp"].isna().any():
                    raise ValueError(
                        "NaN dans les timestamps d’option_chain (fallback)"
                    )
                self.log_performance(
                    "load_option_chain_fallback",
                    latency,
                    success=True,
                    num_rows=len(data),
                    confidence_drop_rate=confidence_drop_rate,
                )
                miya_speak(
                    "Fallback à pd.read_csv pour option_chain",
                    tag="ADVANCED",
                    voice_profile="warning",
                    priority=2,
                )
                send_alert("Fallback à pd.read_csv pour option_chain", priority=2)
                send_telegram_alert("Fallback à pd.read_csv pour option_chain")
                logger.info("Fallback à pd.read_csv pour option_chain")
                self.save_snapshot(
                    "load_option_chain_fallback",
                    {
                        "num_rows": len(data),
                        "columns": list(data.columns),
                        "confidence_drop_rate": confidence_drop_rate,
                    },
                    compress=False,
                )
                return data
            except Exception as fallback_e:
                error_msg = f"Erreur fallback option_chain: {str(fallback_e)}\n{traceback.format_exc()}"
                miya_alerts(
                    error_msg, tag="ADVANCED", voice_profile="urgent", priority=5
                )
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                self.save_snapshot(
                    "error_load_option_chain",
                    {"error": str(fallback_e)},
                    compress=False,
                )
                return pd.DataFrame()

    def calculate_latency_spread(self, data: pd.DataFrame) -> pd.Series:
        """Calcule la latence estimée entre les mises à jour de l'order book."""
        cache_path = os.path.join(
            CACHE_DIR,
            f"latency_{hashlib.sha256(data.to_json().encode()).hexdigest()}.csv",
        )
        try:
            start_time = time.time()
            required_cols = ["timestamp"]
            missing_cols = [col for col in required_cols if col not in data.columns]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )
            if len(data) < 2:
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(required_cols) - len(missing_cols)}/{len(required_cols)} colonnes valides, {len(data)} lignes)"
                miya_alerts(
                    alert_msg, tag="ADVANCED", voice_profile="urgent", priority=3
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            if os.path.exists(cache_path):
                cached_data = pd.read_csv(cache_path)
                if len(cached_data) == len(data) and cached_data["timestamp"].equals(
                    data["timestamp"]
                ):
                    latency = time.time() - start_time
                    self.log_performance(
                        "calculate_latency_spread_cache",
                        latency,
                        success=True,
                        confidence_drop_rate=confidence_drop_rate,
                    )
                    miya_speak(
                        "Latency spread chargé depuis cache",
                        tag="ADVANCED",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert("Latency spread chargé depuis cache", priority=1)
                    send_telegram_alert("Latency spread chargé depuis cache")
                    logger.info("Latency spread chargé depuis cache")
                    return cached_data["latency_spread"]

            data = data.copy()
            data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
            if data["timestamp"].isna().any():
                raise ValueError("NaN dans les timestamps")
            if not data["timestamp"].is_monotonic_increasing:
                raise ValueError("Timestamps non croissants")
            latency = data["timestamp"].diff().dt.total_seconds().fillna(0)
            if latency.min() < 0:
                raise ValueError("Latences négatives détectées")

            os.makedirs(CACHE_DIR, exist_ok=True)

            def write_cache():
                cache_df = pd.DataFrame(
                    {"timestamp": data["timestamp"], "latency_spread": latency}
                )
                cache_df.to_csv(cache_path, index=False, encoding="utf-8")

            self.with_retries(write_cache)
            file_size = os.path.getsize(cache_path) / 1024 / 1024
            if file_size > 1.0:
                miya_alerts(
                    f"Cache latency_spread size {file_size:.2f} MB exceeds 1 MB",
                    tag="ADVANCED",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(
                    f"Cache latency_spread size {file_size:.2f} MB exceeds 1 MB",
                    priority=3,
                )
                send_telegram_alert(
                    f"Cache latency_spread size {file_size:.2f} MB exceeds 1 MB"
                )

            latency_time = time.time() - start_time
            self.log_performance(
                "calculate_latency_spread",
                latency_time,
                success=True,
                confidence_drop_rate=confidence_drop_rate,
            )
            miya_speak(
                "Latency spread calculé",
                tag="ADVANCED",
                voice_profile="calm",
                priority=1,
            )
            send_alert("Latency spread calculé", priority=1)
            send_telegram_alert("Latency spread calculé")
            logger.info("Latency spread calculé")
            self.save_snapshot(
                "calculate_latency_spread",
                {"num_rows": len(data), "confidence_drop_rate": confidence_drop_rate},
                compress=False,
            )
            return latency
        except Exception as e:
            latency_time = time.time() - start_time
            error_msg = f"Erreur dans calculate_latency_spread: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                "calculate_latency_spread", latency_time, success=False, error=str(e)
            )
            miya_alerts(error_msg, tag="ADVANCED", voice_profile="urgent", priority=4)
            send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.save_snapshot(
                "error_latency_spread", {"error": str(e)}, compress=False
            )
            return pd.Series(0.0, index=data.index)

    def calculate_queue_position_score(self, data: pd.DataFrame) -> pd.Series:
        """Calcule un score de position dans la file d'attente des ordres basé sur la profondeur."""
        cache_path = os.path.join(
            CACHE_DIR,
            f"queue_position_score_{hashlib.sha256(data.to_json().encode()).hexdigest()}.csv",
        )
        try:
            start_time = time.time()
            required_cols = [
                "timestamp",
                "bid_size_level_1",
                "ask_size_level_1",
                "bid_size_level_2",
                "ask_size_level_2",
            ]
            missing_cols = [col for col in required_cols if col not in data.columns]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )
            if len(data) < 1:
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(required_cols) - len(missing_cols)}/{len(required_cols)} colonnes valides, {len(data)} lignes)"
                miya_alerts(
                    alert_msg, tag="ADVANCED", voice_profile="urgent", priority=3
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            if os.path.exists(cache_path):
                cached_data = pd.read_csv(cache_path)
                if len(cached_data) == len(data) and cached_data["timestamp"].equals(
                    data["timestamp"]
                ):
                    latency = time.time() - start_time
                    self.log_performance(
                        "calculate_queue_position_score_cache",
                        latency,
                        success=True,
                        confidence_drop_rate=confidence_drop_rate,
                    )
                    miya_speak(
                        "Queue position score chargé depuis cache",
                        tag="ADVANCED",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert("Queue position score chargé depuis cache", priority=1)
                    send_telegram_alert("Queue position score chargé depuis cache")
                    logger.info("Queue position score chargé depuis cache")
                    return cached_data["queue_position_score"]

            data = data.copy()
            for col in missing_cols:
                if col != "timestamp":
                    data[col] = 0
                    miya_speak(
                        f"Colonne '{col}' manquante, imputée à 0",
                        tag="ADVANCED",
                        voice_profile="warning",
                        priority=2,
                    )
                    send_alert(f"Colonne '{col}' manquante, imputée à 0", priority=2)
                    send_telegram_alert(f"Colonne '{col}' manquante, imputée à 0")

            total_depth = (
                data["bid_size_level_1"].fillna(0)
                + data["ask_size_level_1"].fillna(0)
                + data["bid_size_level_2"].fillna(0)
                + data["ask_size_level_2"].fillna(0)
            )
            imbalance = (
                data["bid_size_level_1"].fillna(0) + data["bid_size_level_2"].fillna(0)
            ) - (
                data["ask_size_level_1"].fillna(0) + data["ask_size_level_2"].fillna(0)
            )
            score = imbalance / (total_depth + 1e-6)
            score = score.fillna(0)

            os.makedirs(CACHE_DIR, exist_ok=True)

            def write_cache():
                cache_df = pd.DataFrame(
                    {"timestamp": data["timestamp"], "queue_position_score": score}
                )
                cache_df.to_csv(cache_path, index=False, encoding="utf-8")

            self.with_retries(write_cache)
            file_size = os.path.getsize(cache_path) / 1024 / 1024
            if file_size > 1.0:
                miya_alerts(
                    f"Cache queue_position_score size {file_size:.2f} MB exceeds 1 MB",
                    tag="ADVANCED",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(
                    f"Cache queue_position_score size {file_size:.2f} MB exceeds 1 MB",
                    priority=3,
                )
                send_telegram_alert(
                    f"Cache queue_position_score size {file_size:.2f} MB exceeds 1 MB"
                )

            latency = time.time() - start_time
            self.log_performance(
                "calculate_queue_position_score",
                latency,
                success=True,
                confidence_drop_rate=confidence_drop_rate,
            )
            miya_speak(
                "Queue position score calculé",
                tag="ADVANCED",
                voice_profile="calm",
                priority=1,
            )
            send_alert("Queue position score calculé", priority=1)
            send_telegram_alert("Queue position score calculé")
            logger.info("Queue position score calculé")
            self.save_snapshot(
                "calculate_queue_position_score",
                {"num_rows": len(data), "confidence_drop_rate": confidence_drop_rate},
                compress=False,
            )
            return score
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans calculate_queue_position_score: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                "calculate_queue_position_score", latency, success=False, error=str(e)
            )
            miya_alerts(error_msg, tag="ADVANCED", voice_profile="urgent", priority=4)
            send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.save_snapshot(
                "error_queue_position_score", {"error": str(e)}, compress=False
            )
            return pd.Series(0.0, index=data.index)

    def calculate_vanna_cliff_slope(
        self, option_chain: pd.DataFrame, underlying_price: float
    ) -> float:
        """Calcule la pente de la sensibilité vanna par rapport au prix sous-jacent."""
        cache_path = os.path.join(
            CACHE_DIR,
            f"vanna_cliff_slope_{hashlib.sha256(str(underlying_price).encode()).hexdigest()}.csv",
        )
        try:
            start_time = time.time()
            required_cols = ["timestamp", "strike", "vanna", "open_interest"]
            missing_cols = [
                col for col in required_cols if col not in option_chain.columns
            ]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )
            if len(option_chain) < 5:
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(required_cols) - len(missing_cols)}/{len(required_cols)} colonnes valides, {len(option_chain)} lignes)"
                miya_alerts(
                    alert_msg, tag="ADVANCED", voice_profile="urgent", priority=3
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            if os.path.exists(cache_path):
                cached_data = pd.read_csv(cache_path)
                if len(cached_data) == 1:
                    latency = time.time() - start_time
                    self.log_performance(
                        "calculate_vanna_cliff_slope_cache",
                        latency,
                        success=True,
                        confidence_drop_rate=confidence_drop_rate,
                    )
                    miya_speak(
                        "Vanna cliff slope chargé depuis cache",
                        tag="ADVANCED",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert("Vanna cliff slope chargé depuis cache", priority=1)
                    send_telegram_alert("Vanna cliff slope chargé depuis cache")
                    logger.info("Vanna cliff slope chargé depuis cache")
                    return cached_data["vanna_cliff_slope"].iloc[0]

            option_chain = option_chain.copy()
            for col in missing_cols:
                option_chain[col] = 0
                miya_speak(
                    f"Colonne '{col}' manquante dans option_chain, imputée à 0",
                    tag="ADVANCED",
                    voice_profile="warning",
                    priority=2,
                )
                send_alert(
                    f"Colonne '{col}' manquante dans option_chain, imputée à 0",
                    priority=2,
                )
                send_telegram_alert(
                    f"Colonne '{col}' manquante dans option_chain, imputée à 0"
                )

            if len(option_chain["strike"].unique()) < 5:
                return 0.0
            option_chain["vanna_exposure"] = (
                option_chain["vanna"] * option_chain["open_interest"]
            )
            vanna_by_strike = (
                option_chain.groupby("strike")["vanna_exposure"].sum().reset_index()
            )
            if len(vanna_by_strike) < 2:
                return 0.0
            vanna_by_strike = vanna_by_strike.sort_values("strike")
            delta_vanna = vanna_by_strike["vanna_exposure"].diff()
            delta_strike = vanna_by_strike["strike"].diff()
            slope = (delta_vanna / delta_strike).mean()
            slope = slope if not np.isnan(slope) else 0.0

            os.makedirs(CACHE_DIR, exist_ok=True)

            def write_cache():
                cache_df = pd.DataFrame({"vanna_cliff_slope": [slope]})
                cache_df.to_csv(cache_path, index=False, encoding="utf-8")

            self.with_retries(write_cache)
            file_size = os.path.getsize(cache_path) / 1024 / 1024
            if file_size > 1.0:
                miya_alerts(
                    f"Cache vanna_cliff_slope size {file_size:.2f} MB exceeds 1 MB",
                    tag="ADVANCED",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(
                    f"Cache vanna_cliff_slope size {file_size:.2f} MB exceeds 1 MB",
                    priority=3,
                )
                send_telegram_alert(
                    f"Cache vanna_cliff_slope size {file_size:.2f} MB exceeds 1 MB"
                )

            latency = time.time() - start_time
            self.log_performance(
                "calculate_vanna_cliff_slope",
                latency,
                success=True,
                confidence_drop_rate=confidence_drop_rate,
            )
            miya_speak(
                "Vanna cliff slope calculé",
                tag="ADVANCED",
                voice_profile="calm",
                priority=1,
            )
            send_alert("Vanna cliff slope calculé", priority=1)
            send_telegram_alert("Vanna cliff slope calculé")
            logger.info("Vanna cliff slope calculé")
            self.save_snapshot(
                "calculate_vanna_cliff_slope",
                {"slope": slope, "confidence_drop_rate": confidence_drop_rate},
                compress=False,
            )
            return slope
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans calculate_vanna_cliff_slope: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                "calculate_vanna_cliff_slope", latency, success=False, error=str(e)
            )
            miya_alerts(error_msg, tag="ADVANCED", voice_profile="urgent", priority=4)
            send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.save_snapshot(
                "error_vanna_cliff_slope", {"error": str(e)}, compress=False
            )
            return 0.0

    def calculate_order_book_pressure_slope(
        self, data: pd.DataFrame, window_size: int = 5
    ) -> pd.Series:
        """Calcule la pente de la pression de l'order book basée sur les variations des tailles bid/ask."""
        cache_path = os.path.join(
            CACHE_DIR,
            f"order_book_pressure_slope_{hashlib.sha256(data.to_json().encode()).hexdigest()}.csv",
        )
        try:
            start_time = time.time()
            required_cols = [
                "timestamp",
                "bid_size_level_1",
                "ask_size_level_1",
                "bid_size_level_2",
                "ask_size_level_2",
            ]
            missing_cols = [col for col in required_cols if col not in data.columns]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )
            if len(data) < window_size:
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(required_cols) - len(missing_cols)}/{len(required_cols)} colonnes valides, {len(data)} lignes)"
                miya_alerts(
                    alert_msg, tag="ADVANCED", voice_profile="urgent", priority=3
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            if os.path.exists(cache_path):
                cached_data = pd.read_csv(cache_path)
                if len(cached_data) == len(data) and cached_data["timestamp"].equals(
                    data["timestamp"]
                ):
                    latency = time.time() - start_time
                    self.log_performance(
                        "calculate_order_book_pressure_slope_cache",
                        latency,
                        success=True,
                        confidence_drop_rate=confidence_drop_rate,
                    )
                    miya_speak(
                        "Order book pressure slope chargé depuis cache",
                        tag="ADVANCED",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert(
                        "Order book pressure slope chargé depuis cache", priority=1
                    )
                    send_telegram_alert("Order book pressure slope chargé depuis cache")
                    logger.info("Order book pressure slope chargé depuis cache")
                    return cached_data["order_book_pressure_slope"]

            data = data.copy()
            for col in missing_cols:
                if col != "timestamp":
                    data[col] = 0
                    miya_speak(
                        f"Colonne '{col}' manquante, imputée à 0",
                        tag="ADVANCED",
                        voice_profile="warning",
                        priority=2,
                    )
                    send_alert(f"Colonne '{col}' manquante, imputée à 0", priority=2)
                    send_telegram_alert(f"Colonne '{col}' manquante, imputée à 0")

            pressure = (
                data["bid_size_level_1"].fillna(0) + data["bid_size_level_2"].fillna(0)
            ) - (
                data["ask_size_level_1"].fillna(0) + data["ask_size_level_2"].fillna(0)
            )
            slope = pressure.diff().rolling(window=window_size, min_periods=1).mean()
            slope = slope.fillna(0)

            os.makedirs(CACHE_DIR, exist_ok=True)

            def write_cache():
                cache_df = pd.DataFrame(
                    {"timestamp": data["timestamp"], "order_book_pressure_slope": slope}
                )
                cache_df.to_csv(cache_path, index=False, encoding="utf-8")

            self.with_retries(write_cache)
            file_size = os.path.getsize(cache_path) / 1024 / 1024
            if file_size > 1.0:
                miya_alerts(
                    f"Cache order_book_pressure_slope size {file_size:.2f} MB exceeds 1 MB",
                    tag="ADVANCED",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(
                    f"Cache order_book_pressure_slope size {file_size:.2f} MB exceeds 1 MB",
                    priority=3,
                )
                send_telegram_alert(
                    f"Cache order_book_pressure_slope size {file_size:.2f} MB exceeds 1 MB"
                )

            latency = time.time() - start_time
            self.log_performance(
                "calculate_order_book_pressure_slope",
                latency,
                success=True,
                confidence_drop_rate=confidence_drop_rate,
            )
            miya_speak(
                "Order book pressure slope calculé",
                tag="ADVANCED",
                voice_profile="calm",
                priority=1,
            )
            send_alert("Order book pressure slope calculé", priority=1)
            send_telegram_alert("Order book pressure slope calculé")
            logger.info("Order book pressure slope calculé")
            self.save_snapshot(
                "calculate_order_book_pressure_slope",
                {"num_rows": len(data), "confidence_drop_rate": confidence_drop_rate},
                compress=False,
            )
            return slope
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans calculate_order_book_pressure_slope: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                "calculate_order_book_pressure_slope",
                latency,
                success=False,
                error=str(e),
            )
            miya_alerts(error_msg, tag="ADVANCED", voice_profile="urgent", priority=4)
            send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.save_snapshot(
                "error_order_book_pressure_slope", {"error": str(e)}, compress=False
            )
            return pd.Series(0.0, index=data.index)

    def calculate_iv_atm(
        self, option_chain: pd.DataFrame, underlying_price: float
    ) -> float:
        """Calcule la volatilité implicite at-the-money (strike le plus proche du prix sous-jacent)."""
        cache_path = os.path.join(
            CACHE_DIR,
            f"iv_atm_{hashlib.sha256(str(underlying_price).encode()).hexdigest()}.csv",
        )
        try:
            start_time = time.time()
            required_cols = ["timestamp", "strike", "implied_volatility"]
            missing_cols = [
                col for col in required_cols if col not in option_chain.columns
            ]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )

            if len(option_chain) < 1:
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = (
                    f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} "
                    f"({len(required_cols) - len(missing_cols)}/{len(required_cols)} colonnes valides, "
                    f"{len(option_chain)} lignes)"
                )
                miya_alerts(
                    alert_msg, tag="ADVANCED", voice_profile="urgent", priority=3
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            if os.path.exists(cache_path):
                cached_data = pd.read_csv(cache_path)
                if len(cached_data) == 1:
                    latency = time.time() - start_time
                    self.log_performance(
                        "calculate_iv_atm_cache",
                        latency,
                        success=True,
                        confidence_drop_rate=confidence_drop_rate,
                    )
                    miya_speak(
                        "IV ATM chargé depuis cache",
                        tag="ADVANCED",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert("IV ATM chargé depuis cache", priority=1)
                    send_telegram_alert("IV ATM chargé depuis cache")
                    logger.info("IV ATM chargé depuis cache")
                    return cached_data["iv_atm"].iloc[0]

            # Cas où le cache n'existe pas
            option_chain = option_chain.copy()
            for col in missing_cols:
                option_chain[col] = 0
                miya_speak(
                    f"Colonne '{col}' manquante dans option_chain, imputée à 0",
                    tag="ADVANCED",
                    voice_profile="warning",
                    priority=2,
                )
                send_alert(
                    f"Colonne '{col}' manquante dans option_chain, imputée à 0",
                    priority=2,
                )
                send_telegram_alert(
                    f"Colonne '{col}' manquante dans option_chain, imputée à 0"
                )

            if option_chain.empty:
                logger.warning("Option chain vide, retourne 0.0")
                return 0.0

            # Calcul de l'IV ATM
            option_chain["strike_diff"] = abs(option_chain["strike"] - underlying_price)
            atm_strike = option_chain.loc[option_chain["strike_diff"].idxmin()]
            iv_atm = (
                atm_strike["implied_volatility"]
                if not pd.isna(atm_strike["implied_volatility"])
                else 0.0
            )

            # Sauvegarde dans le cache
            os.makedirs(CACHE_DIR, exist_ok=True)

            def write_cache():
                cache_df = pd.DataFrame({"iv_atm": [iv_atm]})
                cache_df.to_csv(cache_path, index=False, encoding="utf-8")

            self.with_retries(write_cache)

            file_size = os.path.getsize(cache_path) / 1024 / 1024
            if file_size > 1.0:
                miya_alerts(
                    f"Cache iv_atm size {file_size:.2f} MB exceeds 1 MB",
                    tag="ADVANCED",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(
                    f"Cache iv_atm size {file_size:.2f} MB exceeds 1 MB", priority=3
                )
                send_telegram_alert(
                    f"Cache iv_atm size {file_size:.2f} MB exceeds 1 MB"
                )

            latency = time.time() - start_time
            self.log_performance(
                "calculate_iv_atm",
                latency,
                success=True,
                confidence_drop_rate=confidence_drop_rate,
            )
            miya_speak(
                "IV ATM calculé", tag="ADVANCED", voice_profile="calm", priority=1
            )
            send_alert("IV ATM calculé", priority=1)
            send_telegram_alert("IV ATM calculé")
            logger.info("IV ATM calculé")
            self.save_snapshot(
                "calculate_iv_atm",
                {"iv_atm": iv_atm, "confidence_drop_rate": confidence_drop_rate},
                compress=False,
            )
            return iv_atm

        except Exception as e:
            latency = time.time() - start_time
            error_msg = (
                f"Erreur dans calculate_iv_atm: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance(
                "calculate_iv_atm", latency, success=False, error=str(e)
            )
            miya_alerts(error_msg, tag="ADVANCED", voice_profile="urgent", priority=4)
            send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.save_snapshot("error_iv_atm", {"error": str(e)}, compress=False)
            return 0.0

    def calculate_option_skew(
        self, option_chain: pd.DataFrame, underlying_price: float
    ) -> float:
        """Calcule l’asymétrie des volatilités implicites entre calls et puts pour le strike le plus proche."""
        cache_path = os.path.join(
            CACHE_DIR,
            f"option_skew_{hashlib.sha256(str(underlying_price).encode()).hexdigest()}.csv",
        )
        try:
            start_time = time.time()
            required_cols = ["timestamp", "strike", "implied_volatility", "option_type"]
            missing_cols = [
                col for col in required_cols if col not in option_chain.columns
            ]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )
            if len(option_chain) < 1:
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(required_cols) - len(missing_cols)}/{len(required_cols)} colonnes valides, {len(option_chain)} lignes)"
                miya_alerts(
                    alert_msg, tag="ADVANCED", voice_profile="urgent", priority=3
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            if os.path.exists(cache_path):
                cached_data = pd.read_csv(cache_path)
                if len(cached_data) == 1:
                    latency = time.time() - start_time
                    self.log_performance(
                        "calculate_option_skew_cache",
                        latency,
                        success=True,
                        confidence_drop_rate=confidence_drop_rate,
                    )
                    miya_speak(
                        "Option skew chargé depuis cache",
                        tag="ADVANCED",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert("Option skew chargé depuis cache", priority=1)
                    send_telegram_alert("Option skew chargé depuis cache")
                    logger.info("Option skew chargé depuis cache")
                    return cached_data["option_skew"].iloc[0]

            option_chain = option_chain.copy()
            for col in missing_cols:
                option_chain[col] = 0 if col != "option_type" else "call"
                miya_speak(
                    f"Colonne '{col}' manquante dans option_chain, imputée",
                    tag="ADVANCED",
                    voice_profile="warning",
                    priority=2,
                )
                send_alert(
                    f"Colonne '{col}' manquante dans option_chain, imputée", priority=2
                )
                send_telegram_alert(
                    f"Colonne '{col}' manquante dans option_chain, imputée"
                )

            if option_chain.empty:
                return 0.0
            option_chain["strike_diff"] = abs(option_chain["strike"] - underlying_price)
            atm_strike = option_chain["strike_diff"].idxmin()
            atm_options = option_chain[
                option_chain["strike"] == option_chain.loc[atm_strike, "strike"]
            ]
            calls = atm_options[atm_options["option_type"] == "call"]
            puts = atm_options[atm_options["option_type"] == "put"]
            iv_call = calls["implied_volatility"].mean() if not calls.empty else 0.0
            iv_put = puts["implied_volatility"].mean() if not puts.empty else 0.0
            skew = (
                iv_call - iv_put if not (np.isnan(iv_call) or np.isnan(iv_put)) else 0.0
            )

            os.makedirs(CACHE_DIR, exist_ok=True)

            def write_cache():
                cache_df = pd.DataFrame({"option_skew": [skew]})
                cache_df.to_csv(cache_path, index=False, encoding="utf-8")

            self.with_retries(write_cache)
            file_size = os.path.getsize(cache_path) / 1024 / 1024
            if file_size > 1.0:
                miya_alerts(
                    f"Cache option_skew size {file_size:.2f} MB exceeds 1 MB",
                    tag="ADVANCED",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(
                    f"Cache option_skew size {file_size:.2f} MB exceeds 1 MB",
                    priority=3,
                )
                send_telegram_alert(
                    f"Cache option_skew size {file_size:.2f} MB exceeds 1 MB"
                )

            latency = time.time() - start_time
            self.log_performance(
                "calculate_option_skew",
                latency,
                success=True,
                confidence_drop_rate=confidence_drop_rate,
            )
            miya_speak(
                "Option skew calculé", tag="ADVANCED", voice_profile="calm", priority=1
            )
            send_alert("Option skew calculé", priority=1)
            send_telegram_alert("Option skew calculé")
            logger.info("Option skew calculé")
            self.save_snapshot(
                "calculate_option_skew",
                {"skew": skew, "confidence_drop_rate": confidence_drop_rate},
                compress=False,
            )
            return skew
        except Exception as e:
            latency = time.time() - start_time
            error_msg = (
                f"Erreur dans calculate_option_skew: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance(
                "calculate_option_skew", latency, success=False, error=str(e)
            )
            miya_alerts(error_msg, tag="ADVANCED", voice_profile="urgent", priority=4)
            send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.save_snapshot("error_option_skew", {"error": str(e)}, compress=False)
            return 0.0

    def calculate_gex_slope(
        self, option_chain: pd.DataFrame, underlying_price: float
    ) -> float:
        """Calcule la pente de l’exposition gamma (GEX) par rapport au prix sous-jacent."""
        cache_path = os.path.join(
            CACHE_DIR,
            f"gex_slope_{hashlib.sha256(str(underlying_price).encode()).hexdigest()}.csv",
        )
        try:
            start_time = time.time()
            required_cols = [
                "timestamp",
                "strike",
                "gamma",
                "open_interest",
                "option_type",
            ]
            missing_cols = [
                col for col in required_cols if col not in option_chain.columns
            ]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )
            if len(option_chain) < 2:
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(required_cols) - len(missing_cols)}/{len(required_cols)} colonnes valides, {len(option_chain)} lignes)"
                miya_alerts(
                    alert_msg, tag="ADVANCED", voice_profile="urgent", priority=3
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            if os.path.exists(cache_path):
                cached_data = pd.read_csv(cache_path)
                if len(cached_data) == 1:
                    latency = time.time() - start_time
                    self.log_performance(
                        "calculate_gex_slope_cache",
                        latency,
                        success=True,
                        confidence_drop_rate=confidence_drop_rate,
                    )
                    miya_speak(
                        "GEX slope chargé depuis cache",
                        tag="ADVANCED",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert("GEX slope chargé depuis cache", priority=1)
                    send_telegram_alert("GEX slope chargé depuis cache")
                    logger.info("GEX slope chargé depuis cache")
                    return cached_data["gex_slope"].iloc[0]

            option_chain = option_chain.copy()
            for col in missing_cols:
                option_chain[col] = 0 if col != "option_type" else "call"
                miya_speak(
                    f"Colonne '{col}' manquante dans option_chain, imputée",
                    tag="ADVANCED",
                    voice_profile="warning",
                    priority=2,
                )
                send_alert(
                    f"Colonne '{col}' manquante dans option_chain, imputée", priority=2
                )
                send_telegram_alert(
                    f"Colonne '{col}' manquante dans option_chain, imputée"
                )

            if option_chain.empty:
                return 0.0
            option_chain["gex"] = (
                option_chain["gamma"]
                * option_chain["open_interest"]
                * (1 if option_chain["option_type"] == "call" else -1)
            )
            gex_by_strike = option_chain.groupby("strike")["gex"].sum().reset_index()
            if len(gex_by_strike) < 2:
                return 0.0
            gex_by_strike = gex_by_strike.sort_values("strike")
            delta_gex = gex_by_strike["gex"].diff()
            delta_strike = gex_by_strike["strike"].diff()
            slope = (delta_gex / delta_strike).mean()
            slope = slope if not np.isnan(slope) else 0.0

            os.makedirs(CACHE_DIR, exist_ok=True)

            def write_cache():
                cache_df = pd.DataFrame({"gex_slope": [slope]})
                cache_df.to_csv(cache_path, index=False, encoding="utf-8")

            self.with_retries(write_cache)
            file_size = os.path.getsize(cache_path) / 1024 / 1024
            if file_size > 1.0:
                miya_alerts(
                    f"Cache gex_slope size {file_size:.2f} MB exceeds 1 MB",
                    tag="ADVANCED",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(
                    f"Cache gex_slope size {file_size:.2f} MB exceeds 1 MB", priority=3
                )
                send_telegram_alert(
                    f"Cache gex_slope size {file_size:.2f} MB exceeds 1 MB"
                )

            latency = time.time() - start_time
            self.log_performance(
                "calculate_gex_slope",
                latency,
                success=True,
                confidence_drop_rate=confidence_drop_rate,
            )
            miya_speak(
                "GEX slope calculé", tag="ADVANCED", voice_profile="calm", priority=1
            )
            send_alert("GEX slope calculé", priority=1)
            send_telegram_alert("GEX slope calculé")
            logger.info("GEX slope calculé")
            self.save_snapshot(
                "calculate_gex_slope",
                {"slope": slope, "confidence_drop_rate": confidence_drop_rate},
                compress=False,
            )
            return slope
        except Exception as e:
            latency = time.time() - start_time
            error_msg = (
                f"Erreur dans calculate_gex_slope: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance(
                "calculate_gex_slope", latency, success=False, error=str(e)
            )
            miya_alerts(error_msg, tag="ADVANCED", voice_profile="urgent", priority=4)
            send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.save_snapshot("error_gex_slope", {"error": str(e)}, compress=False)
            return 0.0


def calculate_advanced_features(
    self,
    data: pd.DataFrame,
    option_chain_path: str = os.path.join(
        BASE_DIR, "data", "iqfeed", "option_chain.csv"
    ),
    config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml"),
) -> pd.DataFrame:
    """Calcule les features avancées de microstructure et options à partir des données IQFeed."""
    try:
        start_time = time.time()
        config = self.load_config(config_path)
        window_size = config.get("window_size", 5)
        min_option_rows = config.get("min_option_rows", 10)

        required_data_cols = [
            "timestamp",
            "close",
            "bid_size_level_1",
            "ask_size_level_1",
            "bid_size_level_2",
            "ask_size_level_2",
            "bid_price_level_1",
            "ask_price_level_1",
        ]
        missing_data_cols = [
            col for col in required_data_cols if col not in data.columns
        ]
        confidence_drop_rate = 1.0 - min(
            (len(required_data_cols) - len(missing_data_cols))
            / len(required_data_cols),
            1.0,
        )
        if len(data) < window_size:
            confidence_drop_rate = max(confidence_drop_rate, 0.5)
        if confidence_drop_rate > 0.5:
            alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(required_data_cols) - len(missing_data_cols)}/{len(required_data_cols)} colonnes valides, {len(data)} lignes)"
            miya_alerts(alert_msg, tag="ADVANCED", voice_profile="urgent", priority=3)
            send_alert(alert_msg, priority=3)
            send_telegram_alert(alert_msg)
            logger.warning(alert_msg)

        cache_path = os.path.join(
            CACHE_DIR,
            f"advanced_features_{hashlib.sha256(data.to_json().encode()).hexdigest()}.csv",
        )
        if os.path.exists(cache_path):
            cached_data = pd.read_csv(cache_path)
            if len(cached_data) == len(data) and cached_data["timestamp"].equals(
                data["timestamp"]
            ):
                latency = time.time() - start_time
                self.log_performance(
                    "calculate_advanced_features_cache_hit",
                    latency,
                    success=True,
                    confidence_drop_rate=confidence_drop_rate,
                )
                miya_speak(
                    "Features avancées chargées depuis cache",
                    tag="ADVANCED",
                    voice_profile="calm",
                    priority=1,
                )
                send_alert("Features avancées chargées depuis cache", priority=1)
                send_telegram_alert("Features avancées chargées depuis cache")
                logger.info("Features avancées chargées depuis cache")
                return cached_data

        if data.empty:
            error_msg = "DataFrame vide"
            miya_alerts(error_msg, tag="ADVANCED", voice_profile="urgent", priority=5)
            send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.save_snapshot(
                "error_advanced_features", {"error": error_msg}, compress=False
            )
            raise ValueError(error_msg)

        data = data.copy()
        for col in missing_data_cols:
            if col == "timestamp":
                data[col] = pd.date_range(
                    start=pd.Timestamp.now(), periods=len(data), freq="1min"
                )
                miya_speak(
                    "Colonne timestamp manquante, création par défaut",
                    tag="ADVANCED",
                    voice_profile="warning",
                    priority=2,
                )
                send_alert(
                    "Colonne timestamp manquante, création par défaut", priority=2
                )
                send_telegram_alert("Colonne timestamp manquante, création par défaut")
            else:
                data[col] = (
                    data["close"]
                    .interpolate(method="linear", limit_direction="both")
                    .fillna(data["close"].median())
                    if col == "close"
                    else 0
                )
                miya_speak(
                    f"Colonne '{col}' manquante, imputée",
                    tag="ADVANCED",
                    voice_profile="warning",
                    priority=2,
                )
                send_alert(f"Colonne '{col}' manquante, imputée", priority=2)
                send_telegram_alert(f"Colonne '{col}' manquante, imputée")

        option_chain = self.load_option_chain(option_chain_path)
        if len(option_chain) < min_option_rows:
            error_msg = f"Moins de {min_option_rows} lignes dans option_chain"
            miya_alerts(error_msg, tag="ADVANCED", voice_profile="urgent", priority=5)
            send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.save_snapshot(
                "error_advanced_features", {"error": error_msg}, compress=False
            )
            raise ValueError(error_msg)

        option_chain["timestamp"] = pd.to_datetime(
            option_chain["timestamp"], errors="coerce"
        )
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
        if data["timestamp"].isna().any() or option_chain["timestamp"].isna().any():
            error_msg = "NaN dans les timestamps"
            miya_alerts(error_msg, tag="ADVANCED", voice_profile="urgent", priority=5)
            send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.save_snapshot(
                "error_advanced_features", {"error": error_msg}, compress=False
            )
            raise ValueError(error_msg)

        features = [
            "latency_spread",
            "queue_position_score",
            "vanna_cliff_slope",
            "order_book_pressure_slope",
            "depth_imbalance",
            "vanna_exposure",
            "order_flow_acceleration",
            "vanna_skew",
            "microstructure_volatility",
            "liquidity_absorption_rate",
            "iv_atm",
            "option_skew",
            "gex_slope",
        ]
        self.validate_shap_features(features)
        missing_features = [f for f in features if f not in obs_t]
        if missing_features:
            miya_alerts(
                f"Features manquantes dans obs_t: {missing_features}",
                tag="ADVANCED",
                voice_profile="warning",
                priority=3,
            )
            send_alert(
                f"Features manquantes dans obs_t: {missing_features}", priority=3
            )
            send_telegram_alert(f"Features manquantes dans obs_t: {missing_features}")
            logger.warning(f"Features manquantes dans obs_t: {missing_features}")

        for feature in features:
            data[feature] = 0.0

        data["latency_spread"] = self.calculate_latency_spread(data)
        data["queue_position_score"] = self.calculate_queue_position_score(data)
        data["order_book_pressure_slope"] = self.calculate_order_book_pressure_slope(
            data, window_size
        )

        option_chain["vanna_exposure"] = (
            option_chain["vanna"] * option_chain["open_interest"]
        )
        vanna_by_timestamp = (
            option_chain.groupby("timestamp")
            .agg(
                {
                    "vanna_exposure": "sum",
                    "option_type": lambda x: pd.Series(x == "call").sum()
                    - pd.Series(x == "put").sum(),
                }
            )
            .rename(
                columns={
                    "vanna_exposure": "vanna_exposure_sum",
                    "option_type": "vanna_skew",
                }
            )
        )

        slopes = []
        iv_atm_values = []
        option_skew_values = []
        gex_slopes = []
        for timestamp in data["timestamp"]:
            current_chain = option_chain[option_chain["timestamp"] == timestamp]
            close_price = (
                data[data["timestamp"] == timestamp]["close"].iloc[0]
                if not data[data["timestamp"] == timestamp].empty
                else 0
            )
            slopes.append(self.calculate_vanna_cliff_slope(current_chain, close_price))
            iv_atm_values.append(self.calculate_iv_atm(current_chain, close_price))
            option_skew_values.append(
                self.calculate_option_skew(current_chain, close_price)
            )
            gex_slopes.append(self.calculate_gex_slope(current_chain, close_price))
        data["vanna_cliff_slope"] = pd.Series(slopes, index=data.index)
        data["iv_atm"] = pd.Series(iv_atm_values, index=data.index)
        data["option_skew"] = pd.Series(option_skew_values, index=data.index)
        data["gex_slope"] = pd.Series(gex_slopes, index=data.index)

        data = data.merge(
            vanna_by_timestamp[["vanna_exposure_sum", "vanna_skew"]],
            on="timestamp",
            how="left",
        )
        data["vanna_exposure"] = data["vanna_exposure_sum"].fillna(0)
        data["vanna_skew"] = data["vanna_skew"].fillna(0)
        data = data.drop(columns=["vanna_exposure_sum"], errors="ignore")

        total_depth = (
            data["bid_size_level_1"]
            + data["bid_size_level_2"]
            + data["ask_size_level_1"]
            + data["ask_size_level_2"]
        )
        data["depth_imbalance"] = (
            (data["bid_size_level_1"] + data["bid_size_level_2"])
            - (data["ask_size_level_1"] + data["ask_size_level_2"])
        ) / (total_depth + 1e-6)
        order_flow = data["bid_size_level_1"] + data["ask_size_level_1"]
        data["order_flow_acceleration"] = (
            order_flow.diff()
            .diff()
            .rolling(window=window_size, min_periods=1)
            .mean()
            .fillna(0)
        )
        bid_ask_spread = data["ask_price_level_1"] - data["bid_price_level_1"]
        data["microstructure_volatility"] = (
            bid_ask_spread.rolling(window=window_size * 2, min_periods=1)
            .std()
            .fillna(0)
        )
        data["liquidity_absorption_rate"] = (
            (order_flow.diff().abs() / (total_depth + 1e-6))
            .rolling(window=window_size, min_periods=1)
            .mean()
            .fillna(0)
        )

        for feature in features:
            if data[feature].isna().any():
                data[feature] = data[feature].fillna(0)
            if (
                feature
                in [
                    "latency_spread",
                    "microstructure_volatility",
                    "liquidity_absorption_rate",
                    "iv_atm",
                ]
                and data[feature].min() < 0
            ):
                miya_alerts(
                    f"Valeurs négatives dans {feature}",
                    tag="ADVANCED",
                    voice_profile="urgent",
                    priority=4,
                )
                send_alert(f"Valeurs négatives dans {feature}", priority=4)
                send_telegram_alert(f"Valeurs négatives dans {feature}")
                logger.error(f"Valeurs négatives dans {feature}")
                data[feature] = data[feature].clip(lower=0)

        os.makedirs(CACHE_DIR, exist_ok=True)

        def write_cache():
            data.to_csv(cache_path, index=False, encoding="utf-8")

        self.with_retries(write_cache)
        file_size = os.path.getsize(cache_path) / 1024 / 1024
        if file_size > 1.0:
            miya_alerts(
                f"Cache advanced_features size {file_size:.2f} MB exceeds 1 MB",
                tag="ADVANCED",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Cache advanced_features size {file_size:.2f} MB exceeds 1 MB",
                priority=3,
            )
            send_telegram_alert(
                f"Cache advanced_features size {file_size:.2f} MB exceeds 1 MB"
            )

        current_time = datetime.now()
        expired_keys = [
            k
            for k, v in self.cache.items()
            if (current_time - v["timestamp"]).total_seconds()
            > self.config.get("cache_hours", 24) * 3600
        ]
        for k in expired_keys:
            self.cache.pop(k)

        latency = time.time() - start_time
        self.log_performance(
            "calculate_advanced_features",
            latency,
            success=True,
            num_features=len(features),
            confidence_drop_rate=confidence_drop_rate,
        )
        miya_speak(
            "Features avancées calculées",
            tag="ADVANCED",
            voice_profile="calm",
            priority=1,
        )
        send_alert("Features avancées calculées", priority=1)
        send_telegram_alert("Features avancées calculées")
        logger.info("Features avancées calculées")
        self.save_snapshot(
            "advanced_features",
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
        error_msg = f"Erreur dans calculate_advanced_features: {str(e)}\n{traceback.format_exc()}"
        self.log_performance(
            "calculate_advanced_features", latency, success=False, error=str(e)
        )
        miya_alerts(error_msg, tag="ADVANCED", voice_profile="urgent", priority=5)
        send_alert(error_msg, priority=5)
        send_telegram_alert(error_msg)
        logger.error(error_msg)
        self.save_snapshot("error_advanced_features", {"error": str(e)}, compress=False)
        for feature in features:
            data[feature] = 0.0
        return data
