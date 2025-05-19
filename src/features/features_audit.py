# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/features/features_audit.py
# Audite les features pour détecter NaN, outliers, et vérifier les types, avec rapport détaillé.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, matplotlib>=3.7.0,<3.8.0, seaborn>=0.12.0,<0.13.0,
#   psutil>=5.9.0,<6.0.0, pyyaml>=6.0.0,<7.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
# - src/model/utils/obs_template.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/feature_sets.yaml
# - data/features/feature_importance.csv
# - data/iqfeed/merged_data.csv
#
# Outputs :
# - data/logs/features_audit_raw.csv
# - data/logs/features_audit_final.csv
# - data/logs/features_audit_performance.csv
# - data/audit_snapshots/*.json (option *.json.gz)
# - data/figures/
# - data/features_audit_dashboard.json
#
# Notes :
# - Conforme à structure.txt (version 2.1.3, 2025-05-13).
# - Audite les 350 features, incluant 62 nouvelles features (ex. : level_4_size_bid, vix_term_1m, oi_concentration_ratio).
# - Intègre SHAP (Phase 17) pour identifier 70 features critiques par régime (Trend, Range, Defensive).
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité de l’audit.
# - Vérifie les types pour feature_pipeline.py, valide les données IQFeed via data_provider.py.
# - Utilise exclusivement IQFeed, avec retries (max 3, délai 2^attempt secondes).
# - Sauvegarde des snapshots JSON non compressés par défaut avec option de compression gzip.
# - Enregistre logs psutil avec timestamp dans data/logs/features_audit_performance.csv.
# - Envoie des alertes via AlertManager et telegram_alert pour les erreurs critiques et succès.
# - Tests unitaires disponibles dans tests/test_features_audit.py.
# - Conforme à la ligne rouge des top 150 SHAP features définie dans feature_pipeline.py.

import gzip
import hashlib
import json
import logging
import os
import traceback
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import yaml

from src.model.utils.alert_manager import send_alert
from src.model.utils.config_manager import config_manager
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.model.utils.obs_template import obs_t
from src.utils.telegram_alert import send_telegram_alert

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Configuration du logging
os.makedirs(os.path.join(BASE_DIR, "data", "logs"), exist_ok=True)
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "data", "logs", "features_audit.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Chemins de snapshots, performance, et dashboard
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "audit_snapshots")
CSV_LOG_PATH = os.path.join(BASE_DIR, "data", "logs", "features_audit_performance.csv")
DASHBOARD_PATH = os.path.join(BASE_DIR, "data", "features_audit_dashboard.json")
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures")
FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR, "data", "features", "feature_importance.csv"
)

# Seuils de performance pour 350 features
PERFORMANCE_THRESHOLDS = {
    "max_missing_features": 5,  # Tolérance de 5 features manquantes
    "max_nan_percentage": 0.1,  # 10% max de NaN par feature
    "max_outliers_percentage": 0.05,  # 5% max d'outliers par feature
}

# Constantes
MAX_RETRIES = 3
RETRY_DELAY = 2.0


class FeaturesAudit:

    def __init__(self):
        self.log_buffer = []
        self.cache = {}
        self.config_path = os.path.join(BASE_DIR, "config", "feature_sets.yaml")
        try:
            self.config = self.load_config_with_manager_new(self.config_path)
            required_config = ["feature_sets"]
            missing_config = [key for key in required_config if key not in self.config]
            if missing_config:
                raise ValueError(f"Clés de configuration manquantes: {missing_config}")
            self.buffer_size = self.config.get("logging", {}).get("buffer_size", 100)
            self.max_cache_size = self.config.get("cache", {}).get(
                "max_cache_size", 1000
            )
            global PERFORMANCE_THRESHOLDS
            PERFORMANCE_THRESHOLDS = self.config.get(
                "performance_thresholds", PERFORMANCE_THRESHOLDS
            )
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            os.makedirs(os.path.dirname(CSV_LOG_PATH), exist_ok=True)
            os.makedirs(FIGURES_DIR, exist_ok=True)
            miya_speak(
                "FeaturesAudit initialisé",
                tag="FEATURES_AUDIT",
                voice_profile="calm",
                priority=2,
            )
            send_alert("FeaturesAudit initialisé", priority=2)
            send_telegram_alert("FeaturesAudit initialisé")
            logger.info("FeaturesAudit initialisé")
            self.log_performance("init", 0, success=True)
        except Exception as e:
            error_msg = f"Erreur initialisation FeaturesAudit: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="FEATURES_AUDIT", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            self.config = {
                "performance_thresholds": {
                    "max_missing_features": 5,
                    "max_nan_percentage": 0.1,
                    "max_outliers_percentage": 0.05,
                },
                "logging": {"buffer_size": 100},
                "cache": {"max_cache_size": 1000},
            }
            self.buffer_size = 100
            self.max_cache_size = 1000

    def load_config(self, config_path: str) -> Dict:
        """Charge la configuration via yaml."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            miya_speak(
                f"Configuration chargée via yaml depuis {config_path}",
                tag="FEATURES_AUDIT",
                voice_profile="calm",
                priority=1,
            )
            send_alert(
                f"Configuration chargée via yaml depuis {config_path}", priority=1
            )
            send_telegram_alert(f"Configuration chargée via yaml depuis {config_path}")
            logger.info(f"Configuration chargée via yaml depuis {config_path}")
            return config
        except Exception as e:
            miya_alerts(
                f"Échec yaml pour {config_path}: {str(e)}",
                tag="FEATURES_AUDIT",
                voice_profile="warning",
                priority=3,
            )
            send_alert(f"Échec yaml pour {config_path}: {str(e)}", priority=3)
            send_telegram_alert(f"Échec yaml pour {config_path}: {str(e)}")
            logger.warning(f"Échec yaml pour {config_path}: {str(e)}")
            return {
                "performance_thresholds": {
                    "max_missing_features": 5,
                    "max_nan_percentage": 0.1,
                    "max_outliers_percentage": 0.05,
                },
                "logging": {"buffer_size": 100},
                "cache": {"max_cache_size": 1000},
            }

    def load_config_with_manager_new(self, config_path: str) -> Dict:
        """Charge la configuration via config_manager."""

        def load_yaml():
            config = config_manager.get_config(os.path.basename(config_path))
            if "feature_sets" not in config:
                raise ValueError("Clé 'feature_sets' manquante dans la configuration")
            required_keys = [
                "feature_sets",
                "logging",
                "cache",
                "performance_thresholds",
            ]
            missing_keys = [key for key in required_keys if key not in config]
            if missing_keys:
                raise ValueError(
                    f"Clés manquantes dans la configuration: {missing_keys}"
                )
            return config

        try:
            start_time = time.time()
            config = self.with_retries(load_yaml)
            latency = time.time() - start_time
            miya_speak(
                "Configuration features_audit chargée via config_manager",
                tag="FEATURES_AUDIT",
                voice_profile="calm",
                priority=2,
            )
            send_alert(
                "Configuration features_audit chargée via config_manager", priority=1
            )
            send_telegram_alert(
                "Configuration features_audit chargée via config_manager"
            )
            logger.info("Configuration features_audit chargée via config_manager")
            self.log_performance("load_config_with_manager_new", latency, success=True)
            self.save_snapshot(
                "load_config_with_manager_new",
                {"config_path": config_path},
                compress=False,
            )
            return config
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur chargement config via config_manager: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="FEATURES_AUDIT", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "load_config_with_manager_new", latency, success=False, error=str(e)
            )
            return {
                "performance_thresholds": {
                    "max_missing_features": 5,
                    "max_nan_percentage": 0.1,
                    "max_outliers_percentage": 0.05,
                },
                "logging": {"buffer_size": 100},
                "cache": {"max_cache_size": 1000},
            }

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
                        tag="FEATURES_AUDIT",
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
                    tag="FEATURES_AUDIT",
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
        """Journalise les performances des opérations critiques."""
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
            cpu_usage = psutil.cpu_percent()
            if memory_usage > 1024:
                miya_alerts(
                    f"ALERTE: Utilisation mémoire élevée ({memory_usage:.2f} MB)",
                    tag="FEATURES_AUDIT",
                    voice_profile="urgent",
                    priority=5,
                )
                send_alert(
                    f"ALERTE: Utilisation mémoire élevée ({memory_usage:.2f} MB)",
                    priority=4,
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
            if len(self.log_buffer) >= self.buffer_size:
                log_df = pd.DataFrame(self.log_buffer)
                os.makedirs(os.path.dirname(CSV_LOG_PATH), exist_ok=True)

                def write_log():
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

                self.with_retries(write_log)
                self.log_buffer = []
            logger.info(f"Performance journalisée pour {operation}. CPU: {cpu_usage}%")
        except Exception as e:
            miya_alerts(
                f"Erreur journalisation performance: {str(e)}",
                tag="FEATURES_AUDIT",
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
                    tag="FEATURES_AUDIT",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(f"Snapshot size {file_size:.2f} MB exceeds 1 MB", priority=3)
                send_telegram_alert(f"Snapshot size {file_size:.2f} MB exceeds 1 MB")
            latency = time.time() - start_time
            miya_speak(
                f"Snapshot {snapshot_type} sauvegardé: {save_path}",
                tag="FEATURES_AUDIT",
                voice_profile="calm",
                priority=1,
            )
            send_alert(f"Snapshot {snapshot_type} sauvegardé: {save_path}", priority=1)
            send_telegram_alert(f"Snapshot {snapshot_type} sauvegardé: {save_path}")
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
                tag="FEATURES_AUDIT",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}", priority=3
            )
            send_telegram_alert(f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}")
            logger.error(f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}")

    def save_dashboard_status(
        self, status: Dict, status_file: str = DASHBOARD_PATH, compress: bool = False
    ) -> None:
        """Sauvegarde le statut du tableau de bord."""
        try:
            start_time = time.time()
            os.makedirs(os.path.dirname(status_file), exist_ok=True)

            def write_status():
                if compress:
                    with gzip.open(f"{status_file}.gz", "wt", encoding="utf-8") as f:
                        json.dump(status, f, indent=4)
                else:
                    with open(status_file, "w", encoding="utf-8") as f:
                        json.dump(status, f, indent=4)

            self.with_retries(write_status)
            save_path = f"{status_file}.gz" if compress else status_file
            latency = time.time() - start_time
            miya_speak(
                f"État sauvegardé dans {save_path}",
                tag="FEATURES_AUDIT",
                voice_profile="calm",
                priority=1,
            )
            send_alert(f"État sauvegardé dans {save_path}", priority=1)
            send_telegram_alert(f"État sauvegardé dans {save_path}")
            logger.info(f"État sauvegardé dans {save_path}")
            self.log_performance("save_dashboard_status", latency, success=True)
        except Exception as e:
            latency = time.time() - start_time
            error_msg = (
                f"Erreur sauvegarde dashboard: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, tag="FEATURES_AUDIT", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "save_dashboard_status", latency, success=False, error=str(e)
            )

    def parse_range_value(self, value):
        """Convertit une valeur de plage en float, gérant inf/-inf."""
        try:
            if value is None:
                return None
            if isinstance(value, str):
                val = value.lower().strip()
                if val == "inf":
                    return float("inf")
                elif val == "-inf":
                    return float("-inf")
                return float(value)
            return float(value)
        except Exception as e:
            error_msg = f"Erreur conversion range: {value} -> {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="FEATURES_AUDIT", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            raise ValueError(f"Erreur conversion range: {value} -> {e}")

    def validate_iqfeed_data(
        self, data: pd.DataFrame, expected_features: List[Dict]
    ) -> bool:
        """Valide les données IQFeed."""
        try:
            start_time = time.time()
            iqfeed_features = [
                f["name"]
                for f in expected_features
                if f.get("source") == "IQFeed"
                or f["name"]
                in [
                    "level_4_size_bid",
                    "level_4_size_ask",
                    "vix_term_1m",
                    "vix_term_3m",
                    "vix_term_6m",
                    "oi_concentration_ratio",
                ]
            ]
            missing_cols = [col for col in iqfeed_features if col not in data.columns]
            if missing_cols:
                error_msg = f"Colonnes IQFeed manquantes: {missing_cols}"
                miya_alerts(
                    error_msg, tag="FEATURES_AUDIT", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return False
            if data[iqfeed_features].isna().any().any():
                error_msg = "Valeurs NaN dans les données IQFeed"
                miya_alerts(
                    error_msg, tag="FEATURES_AUDIT", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return False
            if not np.isfinite(
                data[iqfeed_features].select_dtypes(include=[np.number]).values
            ).all():
                error_msg = "Valeurs infinies dans les données IQFeed"
                miya_alerts(
                    error_msg, tag="FEATURES_AUDIT", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return False
            latency = time.time() - start_time
            miya_speak(
                "Données IQFeed validées",
                tag="FEATURES_AUDIT",
                voice_profile="calm",
                priority=1,
            )
            send_alert("Données IQFeed validées", priority=1)
            send_telegram_alert("Données IQFeed validées")
            logger.info("Données IQFeed validées")
            self.log_performance(
                "validate_iqfeed_data",
                latency,
                success=True,
                num_features=len(iqfeed_features),
            )
            return True
        except Exception as e:
            latency = time.time() - start_time
            error_msg = (
                f"Erreur validation données IQFeed: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, tag="FEATURES_AUDIT", voice_profile="urgent", priority=4
            )
            send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "validate_iqfeed_data", latency, success=False, error=str(e)
            )
            return False

    def validate_shap_features(self, features: List[str]) -> bool:
        """Valide les features par rapport aux top 150 SHAP."""
        try:
            start_time = time.time()
            if not os.path.exists(FEATURE_IMPORTANCE_PATH):
                miya_alerts(
                    "Fichier SHAP manquant",
                    tag="FEATURES_AUDIT",
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
                    tag="FEATURES_AUDIT",
                    voice_profile="urgent",
                    priority=4,
                )
                send_alert(
                    f"Nombre insuffisant de SHAP features: {len(shap_df)} < 150",
                    priority=4,
                )
                send_telegram_alert(
                    f"Nombre insuffisant de SHAP features: {len(shap_df)} < 150"
                )
                logger.error(f"Nombre insuffisant de SHAP features: {len(shap_df)}")
                return False
            valid_features = set(shap_df["feature"].head(150)).union(obs_t)
            missing = [f for f in features if f not in valid_features]
            if missing:
                miya_alerts(
                    f"Features non incluses dans top 150 SHAP ou obs_t: {missing}",
                    tag="FEATURES_AUDIT",
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
                tag="FEATURES_AUDIT",
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
            return True
        except Exception as e:
            latency = time.time() - start_time
            miya_alerts(
                f"Erreur validation SHAP features: {str(e)}",
                tag="FEATURES_AUDIT",
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

    def identify_critical_features(
        self, regime: str, max_features: int = 70
    ) -> List[str]:
        """Identifie les 70 features critiques par régime en utilisant SHAP (Phase 17)."""
        try:
            start_time = time.time()
            if not os.path.exists(FEATURE_IMPORTANCE_PATH):
                miya_alerts(
                    "Fichier SHAP manquant",
                    tag="FEATURES_AUDIT",
                    voice_profile="urgent",
                    priority=4,
                )
                send_alert("Fichier SHAP manquant", priority=4)
                send_telegram_alert("Fichier SHAP manquant")
                logger.error("Fichier SHAP manquant")
                return []
            shap_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
            if "feature" not in shap_df.columns or "importance" not in shap_df.columns:
                miya_alerts(
                    "Format SHAP invalide",
                    tag="FEATURES_AUDIT",
                    voice_profile="urgent",
                    priority=4,
                )
                send_alert("Format SHAP invalide", priority=4)
                send_telegram_alert("Format SHAP invalide")
                logger.error("Format SHAP invalide")
                return []
            boost_factors = {"Trend": 1.5, "Range": 1.5, "Defensive": 1.5}
            momentum_features = [
                "spy_lead_return",
                "sector_leader_momentum",
                "spy_momentum_diff",
                "order_flow_acceleration",
            ]
            volatility_features = [
                "atr_14",
                "volatility_trend",
                "microstructure_volatility",
                "vix_es_correlation",
                "iv_atm",
                "option_skew",
                "vix_term_1m",
                "vix_term_3m",
                "vix_term_6m",
            ]
            risk_features = [
                "bond_equity_risk_spread",
                "vanna_cliff_slope",
                "vanna_exposure",
                "delta_exposure",
                "gex_slope",
            ]
            orderbook_features = [
                "level_4_size_bid",
                "level_4_size_ask",
                "orderbook_imbalance",
                "depth_imbalance",
            ]
            adjusted_importance = shap_df.set_index("feature")["importance"].copy()
            if regime == "Trend":
                for feature in momentum_features:
                    if feature in adjusted_importance:
                        adjusted_importance[feature] *= boost_factors["Trend"]
            elif regime == "Range":
                for feature in volatility_features:
                    if feature in adjusted_importance:
                        adjusted_importance[feature] *= boost_factors["Range"]
            elif regime == "Defensive":
                for feature in risk_features:
                    if feature in adjusted_importance:
                        adjusted_importance[feature] *= boost_factors["Defensive"]
            critical_features = (
                adjusted_importance.sort_values(ascending=False)
                .head(max_features)
                .index.tolist()
            )
            self.validate_shap_features(critical_features)
            latency = time.time() - start_time
            miya_speak(
                f"Identifié {len(critical_features)} features critiques pour {regime}",
                tag="FEATURES_AUDIT",
                voice_profile="calm",
                priority=1,
            )
            send_alert(
                f"Identifié {len(critical_features)} features critiques pour {regime}",
                priority=1,
            )
            send_telegram_alert(
                f"Identifié {len(critical_features)} features critiques pour {regime}"
            )
            logger.info(
                f"Identifié {len(critical_features)} features critiques pour {regime}"
            )
            self.log_performance(
                "identify_critical_features",
                latency,
                success=True,
                num_features=len(critical_features),
                regime=regime,
            )
            self.save_snapshot(
                "identify_critical_features",
                {
                    "num_features": len(critical_features),
                    "regime": regime,
                    "features": critical_features,
                },
                compress=False,
            )
            return critical_features
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans identify_critical_features: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="FEATURES_AUDIT", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "identify_critical_features", latency, success=False, error=str(e)
            )
            return []

    def load_feature_sets(
        self, config_path: str = os.path.join(BASE_DIR, "config", "feature_sets.yaml")
    ) -> List[Dict]:
        """Charge les ensembles de features depuis feature_sets.yaml."""
        try:
            start_time = time.time()
            config = self.load_config_with_manager_new(config_path)
            features = []
            for category, details in config["feature_sets"].items():
                for feature in details["features"]:
                    feature_dict = feature.copy()
                    feature_dict.setdefault("range", None)
                    feature_dict["category"] = category
                    features.append(feature_dict)
            neural_features = [
                {"name": f"neural_feature_{i}", "range": None, "category": "neural"}
                for i in range(8)
            ] + [
                {
                    "name": "predicted_volatility",
                    "range": [0, None],
                    "category": "neural",
                },
                {"name": "neural_regime", "range": [0, 2], "category": "neural"},
                {"name": "cnn_pressure", "range": None, "category": "neural"},
            ]
            features.extend(neural_features)
            new_features = [
                {
                    "name": "level_4_size_bid",
                    "range": [0, None],
                    "category": "orderbook",
                    "source": "IQFeed",
                },
                {
                    "name": "level_4_size_ask",
                    "range": [0, None],
                    "category": "orderbook",
                    "source": "IQFeed",
                },
                {
                    "name": "vix_term_1m",
                    "range": [0, None],
                    "category": "volatility",
                    "source": "IQFeed",
                },
                {
                    "name": "vix_term_3m",
                    "range": [0, None],
                    "category": "volatility",
                    "source": "IQFeed",
                },
                {
                    "name": "vix_term_6m",
                    "range": [0, None],
                    "category": "volatility",
                    "source": "IQFeed",
                },
                {
                    "name": "oi_concentration_ratio",
                    "range": [0, 1],
                    "category": "options",
                    "source": "IQFeed",
                },
            ]
            features.extend(new_features)
            latency = time.time() - start_time
            miya_speak(
                f"Chargé {len(features)} features attendues depuis {config_path}",
                tag="FEATURES_AUDIT",
                voice_profile="calm",
                priority=1,
            )
            send_alert(
                f"Chargé {len(features)} features attendues depuis {config_path}",
                priority=1,
            )
            send_telegram_alert(
                f"Chargé {len(features)} features attendues depuis {config_path}"
            )
            logger.info(f"Chargé {len(features)} features depuis {config_path}")
            self.log_performance(
                "load_feature_sets", latency, success=True, num_features=len(features)
            )
            self.save_snapshot(
                "load_feature_sets",
                {"num_features": len(features), "config_path": config_path},
                compress=False,
            )
            return features
        except Exception as e:
            latency = time.time() - start_time
            error_msg = (
                f"Erreur dans load_feature_sets: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, tag="FEATURES_AUDIT", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "load_feature_sets", latency, success=False, error=str(e)
            )
            raise

    def audit_momentum_features(
        self, data: pd.DataFrame, expected_features: List[Dict], stage: str
    ) -> Dict:
        """Audite les features de momentum."""
        momentum_features = [
            f["name"] for f in expected_features if f["category"] == "momentum"
        ]
        return self._audit_category_features(data, momentum_features, "momentum", stage)

    def audit_volatility_features(
        self, data: pd.DataFrame, expected_features: List[Dict], stage: str
    ) -> Dict:
        """Audite les features de volatilité."""
        volatility_features = [
            f["name"] for f in expected_features if f["category"] == "volatility"
        ]
        return self._audit_category_features(
            data, volatility_features, "volatility", stage
        )

    def audit_risk_features(
        self, data: pd.DataFrame, expected_features: List[Dict], stage: str
    ) -> Dict:
        """Audite les features de risque."""
        risk_features = [
            f["name"] for f in expected_features if f["category"] == "risk"
        ]
        return self._audit_category_features(data, risk_features, "risk", stage)

    def audit_orderbook_features(
        self, data: pd.DataFrame, expected_features: List[Dict], stage: str
    ) -> Dict:
        """Audite les features d’orderbook."""
        orderbook_features = [
            f["name"] for f in expected_features if f["category"] == "orderbook"
        ]
        return self._audit_category_features(
            data, orderbook_features, "orderbook", stage
        )

    def audit_options_features(
        self, data: pd.DataFrame, expected_features: List[Dict], stage: str
    ) -> Dict:
        """Audite les features d’options."""
        options_features = [
            f["name"] for f in expected_features if f["category"] == "options"
        ]
        return self._audit_category_features(data, options_features, "options", stage)

    def audit_neural_features(
        self, data: pd.DataFrame, expected_features: List[Dict], stage: str
    ) -> Dict:
        """Audite les features neurales."""
        neural_features = [
            f["name"] for f in expected_features if f["category"] == "neural"
        ]
        return self._audit_category_features(data, neural_features, "neural", stage)

    def _audit_category_features(
        self, data: pd.DataFrame, features: List[str], category: str, stage: str
    ) -> Dict:
        """Audite les features d’une catégorie spécifique."""
        try:
            start_time = time.time()
            available_features = [f for f in features if f in data.columns]
            missing = [f for f in features if f not in data.columns]
            nan_counts = (
                data[available_features].isna().sum()
                if available_features
                else pd.Series()
            )
            outliers = {
                col: (
                    (data[col] > data[col].mean() + 3 * data[col].std())
                    | (data[col] < data[col].mean() - 3 * data[col].std())
                ).sum()
                for col in available_features
                if data[col].dtype in ["float64", "int64"]
            }
            report = {
                "category": category,
                "missing": missing,
                "nan_counts": nan_counts.to_dict(),
                "outliers": outliers,
            }
            latency = time.time() - start_time
            self.log_performance(
                f"audit_{category}_features",
                latency,
                success=True,
                num_features=len(available_features),
            )
            self.save_snapshot(
                f"audit_{category}_features",
                {
                    "category": category,
                    "num_features": len(available_features),
                    "missing": missing,
                    "nan_counts": nan_counts.to_dict(),
                    "outliers": outliers,
                },
                compress=False,
            )
            return report
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans audit_{category}_features: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="FEATURES_AUDIT", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                f"audit_{category}_features", latency, success=False, error=str(e)
            )
            return {
                "category": category,
                "missing": [],
                "nan_counts": {},
                "outliers": {},
            }

    def audit_features(
        self,
        data: pd.DataFrame,
        expected_features: Optional[List[Dict]] = None,
        stage: str = "raw",
        output_fig_dir: str = os.path.join(BASE_DIR, "data", "figures"),
        timestamp: Optional[str] = None,
    ) -> bool:
        """Audite les features pour NaN, outliers, types, et conformité SHAP."""
        try:
            start_time = time.time()
            if data.empty:
                error_msg = "DataFrame vide"
                miya_alerts(
                    error_msg, tag="FEATURES_AUDIT", voice_profile="urgent", priority=3
                )
                send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)
            output_csv = os.path.join(
                BASE_DIR, "data", "logs", f"features_audit_{stage}.csv"
            )
            if expected_features is None:
                expected_features = self.load_feature_sets()
            feature_names = [f["name"] for f in expected_features]
            timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
            cache_key = hashlib.sha256(data.to_json().encode()).hexdigest()
            current_time = datetime.now()
            # Nettoyage du cache
            expired_keys = [
                k
                for k, v in self.cache.items()
                if (current_time - v["timestamp"]).total_seconds()
                > self.config.get("cache_hours", 24) * 3600
            ]
            for k in expired_keys:
                self.cache.pop(k)
            if len(self.cache) > self.max_cache_size:
                oldest_key = min(self.cache, key=lambda k: self.cache[k]["timestamp"])
                self.cache.pop(oldest_key)
            if cache_key in self.cache:
                miya_speak(
                    f"Audit {stage} retrieved from cache",
                    tag="FEATURES_AUDIT",
                    voice_profile="calm",
                    priority=1,
                )
                send_alert(f"Audit {stage} retrieved from cache", priority=1)
                send_telegram_alert(f"Audit {stage} retrieved from cache")
                self.log_performance("audit_features_cache_hit", 0, success=True)
                return self.cache[cache_key]["result"]
            self.validate_iqfeed_data(data, expected_features)
            data = data.copy()
            for col in data.columns:
                if col != "timestamp":
                    if data[col].dtype == object:
                        miya_speak(
                            f"Colonne {col} de type objet, conversion numérique",
                            tag="FEATURES_AUDIT",
                            voice_profile="warning",
                            priority=2,
                        )
                        send_alert(
                            f"Colonne {col} de type objet, conversion numérique",
                            priority=2,
                        )
                        send_telegram_alert(
                            f"Colonne {col} de type objet, conversion numérique"
                        )
                        try:
                            data[col] = pd.to_numeric(data[col], errors="coerce")
                        except Exception as e:
                            miya_alerts(
                                f"Échec conversion {col}: {str(e)}",
                                tag="FEATURES_AUDIT",
                                voice_profile="urgent",
                                priority=3,
                            )
                            send_alert(f"Échec conversion {col}: {str(e)}", priority=3)
                            send_telegram_alert(f"Échec conversion {col}: {str(e)}")
                            logger.warning(f"Échec conversion {col}: {str(e)}")
            missing = [col for col in feature_names if col not in data.columns]
            extra = [
                col
                for col in data.columns
                if col not in feature_names and col != "timestamp"
            ]
            available_features = [col for col in data.columns if col in feature_names]
            nan_counts = (
                data[available_features].isna().sum()
                if available_features
                else pd.Series()
            )
            outliers = {
                col: (
                    (data[col] > data[col].mean() + 3 * data[col].std())
                    | (data[col] < data[col].mean() - 3 * data[col].std())
                ).sum()
                for col in available_features
                if data[col].dtype in ["float64", "int64"]
            }
            skewness = (
                data[available_features].skew() if available_features else pd.Series()
            )
            stats = (
                data[available_features].agg(["min", "max", "median"]).to_dict()
                if available_features
                else {}
            )
            correlations = (
                data[available_features].corr()
                if available_features and len(available_features) > 1
                else pd.DataFrame()
            )
            range_violations = {}
            for feature in expected_features:
                name = feature["name"]
                if name in data.columns and feature["range"]:
                    try:
                        min_val, max_val = feature["range"]
                        min_val = self.parse_range_value(min_val)
                        max_val = self.parse_range_value(max_val)
                        if min_val is not None and max_val is not None:
                            violations = (
                                (data[name] < min_val) | (data[name] > max_val)
                            ).sum()
                        elif min_val is not None:
                            violations = (data[name] < min_val).sum()
                        elif max_val is not None:
                            violations = (data[name] > max_val).sum()
                        else:
                            violations = 0
                        if violations > 0:
                            range_violations[name] = violations
                    except Exception as e:
                        miya_alerts(
                            f"Erreur comparaison range pour {name}: {str(e)}",
                            tag="FEATURES_AUDIT",
                            voice_profile="urgent",
                            priority=3,
                        )
                        send_alert(
                            f"Erreur comparaison range pour {name}: {str(e)}",
                            priority=3,
                        )
                        send_telegram_alert(
                            f"Erreur comparaison range pour {name}: {str(e)}"
                        )
                        logger.warning(f"Erreur range pour {name}: {str(e)}")
            critical_features = {}
            for regime in ["Trend", "Range", "Defensive"]:
                critical_features[regime] = self.identify_critical_features(
                    regime, max_features=70
                )
            category_reports = {
                "momentum": self.audit_momentum_features(
                    data, expected_features, stage
                ),
                "volatility": self.audit_volatility_features(
                    data, expected_features, stage
                ),
                "risk": self.audit_risk_features(data, expected_features, stage),
                "orderbook": self.audit_orderbook_features(
                    data, expected_features, stage
                ),
                "options": self.audit_options_features(data, expected_features, stage),
                "neural": self.audit_neural_features(data, expected_features, stage),
            }
            # Calculer confidence_drop_rate (Phase 8)
            valid_features = len(available_features)
            confidence_drop_rate = 1.0 - min(
                valid_features / PERFORMANCE_THRESHOLDS["min_features"], 1.0
            )
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({valid_features}/{PERFORMANCE_THRESHOLDS['min_features']} features valides)"
                miya_alerts(
                    alert_msg, tag="FEATURES_AUDIT", voice_profile="urgent", priority=3
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)
            report = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "stage": stage,
                "missing": ",".join(missing),
                "extra": ",".join(extra),
                "nan_counts": nan_counts.to_dict(),
                "outliers": outliers,
                "skewness": skewness.to_dict(),
                "stats": stats,
                "range_violations": range_violations,
                "correlation_max": (
                    correlations.max().max() if not correlations.empty else None
                ),
                "critical_features_trend": ",".join(critical_features["Trend"]),
                "critical_features_range": ",".join(critical_features["Range"]),
                "critical_features_defensive": ",".join(critical_features["Defensive"]),
                "category_reports": category_reports,
                "confidence_drop_rate": confidence_drop_rate,
            }
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)

            def write_report():
                pd.DataFrame([report]).to_csv(
                    output_csv,
                    mode="a",
                    header=not os.path.exists(output_csv),
                    index=False,
                    encoding="utf-8",
                )

            self.with_retries(write_report)
            os.makedirs(output_fig_dir, exist_ok=True)
            if available_features:
                sample_data = (
                    data[available_features].sample(frac=min(1.0, 1000 / len(data)))
                    if len(data) > 1000
                    else data[available_features]
                )
                plt.figure(figsize=(15, 8))
                sns.heatmap(sample_data.isna(), cbar=True, cmap="viridis")
                plt.title(f"Heatmap des valeurs manquantes ({stage})")
                plt.xlabel("Features")
                plt.ylabel("Observations")
                plt.savefig(
                    os.path.join(
                        output_fig_dir,
                        f"features_audit_heatmap_{stage}_{timestamp}.png",
                    )
                )
                plt.close()
                if not correlations.empty:
                    plt.figure(figsize=(12, 10))
                    sns.heatmap(
                        correlations, cmap="coolwarm", vmin=-1, vmax=1, center=0
                    )
                    plt.title(f"Matrice de corrélation ({stage})")
                    plt.savefig(
                        os.path.join(
                            output_fig_dir,
                            f"features_audit_correlations_{stage}_{timestamp}.png",
                        )
                    )
                    plt.close()
                outlier_counts = pd.Series(outliers)
                if outlier_counts.sum() > 0:
                    plt.figure(figsize=(15, 6))
                    outlier_counts[outlier_counts > 0].plot(kind="bar")
                    plt.title(f"Outliers par feature (> 3s) ({stage})")
                    plt.xlabel("Features")
                    plt.ylabel("Nombre d'outliers")
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(
                            output_fig_dir,
                            f"features_audit_outliers_{stage}_{timestamp}.png",
                        )
                    )
                    plt.close()
            nan_percentage = (
                nan_counts / len(data) if available_features else pd.Series()
            )
            outliers_percentage = (
                pd.Series(outliers) / len(data) if available_features else pd.Series()
            )
            issues = []
            if len(missing) > PERFORMANCE_THRESHOLDS["max_missing_features"]:
                issues.append(
                    f"{len(missing)} features manquantes > {PERFORMANCE_THRESHOLDS['max_missing_features']}"
                )
            if any(nan_percentage > PERFORMANCE_THRESHOLDS["max_nan_percentage"]):
                issues.append(
                    f"NaN > {PERFORMANCE_THRESHOLDS['max_nan_percentage']*100}% dans {nan_percentage[nan_percentage > PERFORMANCE_THRESHOLDS['max_nan_percentage']].index.tolist()}"
                )
            if any(
                outliers_percentage > PERFORMANCE_THRESHOLDS["max_outliers_percentage"]
            ):
                issues.append(
                    f"Outliers > {PERFORMANCE_THRESHOLDS['max_outliers_percentage']*100}% dans {outliers_percentage[outliers_percentage > PERFORMANCE_THRESHOLDS['max_outliers_percentage']].index.tolist()}"
                )
            if issues:
                miya_alerts(
                    f"Issues détectés: {'; '.join(issues)}",
                    tag="FEATURES_AUDIT",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(f"Issues détectés: {'; '.join(issues)}", priority=3)
                send_telegram_alert(f"Issues détectés: {'; '.join(issues)}")
                logger.warning(f"Issues: {issues}")
            else:
                miya_speak(
                    f"Audit terminé sans problème majeur ({stage})",
                    tag="FEATURES_AUDIT",
                    voice_profile="calm",
                    priority=1,
                )
                send_alert(f"Audit terminé sans problème majeur ({stage})", priority=1)
                send_telegram_alert(f"Audit terminé sans problème majeur ({stage})")
            miya_speak(
                f"Audit ({stage}) - Missing: {len(missing)}, Extra: {len(extra)}, NaN: {nan_counts.sum()}, Outliers: {sum(outliers.values())}",
                tag="FEATURES_AUDIT",
                voice_profile="calm",
                priority=1,
            )
            send_alert(
                f"Audit ({stage}) - Missing: {len(missing)}, Extra: {len(extra)}, NaN: {nan_counts.sum()}, Outliers: {sum(outliers.values())}",
                priority=1,
            )
            send_telegram_alert(
                f"Audit ({stage}) - Missing: {len(missing)}, Extra: {len(extra)}, NaN: {nan_counts.sum()}, Outliers: {sum(outliers.values())}"
            )
            logger.info(
                f"Audit ({stage}) - Missing: {len(missing)}, Extra: {len(extra)}, NaN: {nan_counts.sum()}, Outliers: {sum(outliers.values())}"
            )
            if missing:
                miya_alerts(
                    f"Features manquantes: {missing}",
                    tag="FEATURES_AUDIT",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(f"Features manquantes: {missing}", priority=3)
                send_telegram_alert(f"Features manquantes: {missing}")
            if nan_counts.sum() > 0:
                miya_alerts(
                    f"NaN dans: {nan_counts[nan_counts > 0].index.tolist()}",
                    tag="FEATURES_AUDIT",
                    voice_profile="warning",
                    priority=2,
                )
                send_alert(
                    f"NaN dans: {nan_counts[nan_counts > 0].index.tolist()}", priority=2
                )
                send_telegram_alert(
                    f"NaN dans: {nan_counts[nan_counts > 0].index.tolist()}"
                )
            if range_violations:
                miya_alerts(
                    f"Violations de plages: {range_violations}",
                    tag="FEATURES_AUDIT",
                    voice_profile="warning",
                    priority=2,
                )
                send_alert(f"Violations de plages: {range_violations}", priority=2)
                send_telegram_alert(f"Violations de plages: {range_violations}")
            if not correlations.empty and correlations.max().max() > 0.9:
                miya_alerts(
                    f"Corrélation élevée: {correlations.max().max():.2f}",
                    tag="FEATURES_AUDIT",
                    voice_profile="warning",
                    priority=2,
                )
                send_alert(
                    f"Corrélation élevée: {correlations.max().max():.2f}", priority=2
                )
                send_telegram_alert(
                    f"Corrélation élevée: {correlations.max().max():.2f}"
                )
            result = len(missing) <= PERFORMANCE_THRESHOLDS["max_missing_features"]
            self.cache[cache_key] = {
                "result": result,
                "report": report,
                "timestamp": current_time,
            }
            latency = time.time() - start_time
            self.log_performance(
                "audit_features",
                latency,
                success=True,
                num_features=len(available_features),
                missing=len(missing),
                confidence_drop_rate=confidence_drop_rate,
            )
            self.save_snapshot(
                "audit_features",
                {
                    "stage": stage,
                    "num_features": len(available_features),
                    "missing": len(missing),
                    "extra": len(extra),
                    "nan_counts": nan_counts.sum(),
                    "outliers": sum(outliers.values()),
                    "critical_features_trend": len(critical_features["Trend"]),
                    "critical_features_range": len(critical_features["Range"]),
                    "critical_features_defensive": len(critical_features["Defensive"]),
                    "confidence_drop_rate": confidence_drop_rate,
                },
                compress=False,
            )
            self.save_dashboard_status(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "stage": stage,
                    "num_features": len(available_features),
                    "missing": len(missing),
                    "extra": len(extra),
                    "nan_counts": nan_counts.sum(),
                    "outliers": sum(outliers.values()),
                    "recent_errors": len(
                        [log for log in self.log_buffer if not log["success"]]
                    ),
                    "average_latency": (
                        sum(log["latency"] for log in self.log_buffer)
                        / len(self.log_buffer)
                        if self.log_buffer
                        else 0
                    ),
                    "confidence_drop_rate": confidence_drop_rate,
                },
                compress=False,
            )
            return result
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur audit_features: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="FEATURES_AUDIT", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("audit_features", latency, success=False, error=str(e))
            return False


if __name__ == "__main__":
    try:
        audit = FeaturesAudit()
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-13 10:00:00", periods=4, freq="1min"
                ),
                "close": [5100, 5105, 5110, 5108],
                "atr_14": [1.5, 2.0, 1.8, 1.7],
                "rsi_14": [70, 72, 75, 73],
                "gex": [100000, 105000, 110000, 108000],
                "oi_peak_call_near": [10000, 11000, 12000, 11500],
                "gamma_wall_call": [0.02, 0.021, 0.022, 0.02],
                "iv_atm": [0.15, 0.16, 0.14, 0.15],
                "option_skew": [0.1, 0.2, 0.1, 0.15],
                "gex_slope": [0.01, 0.02, 0.01, 0.015],
                "trade_success_prob": [0.7, 0.8, 0.75, 0.72],
                "level_4_size_bid": [1000, 1200, 1100, 1150],
                "level_4_size_ask": [1100, 1300, 1200, 1250],
                "vix_term_1m": [20, 21, 20.5, 21],
                "oi_concentration_ratio": [0.8, 0.85, 0.9, 0.87],
                **{f"neural_feature_{i}": np.random.randn(4) for i in range(8)},
                "predicted_volatility": [0.1, 0.12, 0.15, 0.13],
                "neural_regime": [0, 1, 2, 1],
                "cnn_pressure": [0.5, 0.6, 0.55, 0.52],
            }
        )
        result = audit.audit_features(data, stage="test")
        print(f"Audit réussi: {result}")
        miya_speak(
            "Test FeaturesAudit terminé",
            tag="FEATURES_AUDIT",
            voice_profile="calm",
            priority=1,
        )
        send_alert("Test FeaturesAudit terminé", priority=1)
        send_telegram_alert("Test FeaturesAudit terminé")
        logger.info("Test FeaturesAudit terminé")
    except Exception as e:
        error_msg = f"Erreur test: {str(e)}\n{traceback.format_exc()}"
        miya_alerts(error_msg, tag="FEATURES_AUDIT", voice_profile="urgent", priority=3)
        send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)
        logger.error(error_msg)
        raise
