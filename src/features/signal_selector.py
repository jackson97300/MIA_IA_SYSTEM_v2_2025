# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/trading/signal_selector.py
# Filtre et classe les signaux pour maximiser la rentabilité.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Calcule le Score Global de Confiance (SGC) avec régimes hybrides (méthode 11),
#        logs psutil, validation SHAP (méthode 17), et compatibilité top 150 SHAP.
#        Intègre données IQFeed comme source exclusive (remplace dxFeed).
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, sklearn>=1.0.2,<1.1.0, psutil>=5.9.0,<6.0.0,
#   pyyaml>=6.0.0,<7.0.0, matplotlib>=3.7.0,<3.8.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
# - src/model/utils/obs_template.py
# - src/trading/detect_regime.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/es_config.yaml
# - data/features/feature_importance.csv
# - data/iqfeed/merged_data.csv
# - data/iqfeed/option_chain.csv
#
# Outputs :
# - data/logs/signal_selector_performance.csv
# - data/signal_snapshots/*.json (option *.json.gz)
# - data/figures/signals/
#
# Notes :
# - Conforme à structure.txt (version 2.1.3, 2025-05-13).
# - Utilise retries (max 3, délai 2^attempt secondes) pour les opérations critiques.
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des signaux.
# - Intègre validation SHAP (Phase 17) pour assurer la conformité avec les top 150 SHAP features.
# - Sauvegarde des snapshots JSON non compressés par défaut avec option de compression gzip.
# - Envoie des alertes via AlertManager et telegram_alert pour les erreurs critiques et succès.
# - Tests unitaires disponibles dans tests/test_signal_selector.py.
# - Migration complète de dxFeed à IQFeed pour toutes les données (OHLC, options, order flow).
# - Évolutions futures : Intégrer analyse de sentiment des nouvelles via news_analyzer.py (prévu pour juin 2025).

import gzip
import hashlib
import json
import logging
import os
import time
import traceback
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import yaml
from sklearn.preprocessing import MinMaxScaler

from src.model.utils.alert_manager import send_alert
from src.model.utils.config_manager import config_manager
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.model.utils.obs_template import obs_t
from src.utils.telegram_alert import send_telegram_alert

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "signal_snapshots")
CSV_LOG_PATH = os.path.join(BASE_DIR, "data", "logs", "signal_selector_performance.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures", "signals")
FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR, "data", "features", "feature_importance.csv"
)

# Création des répertoires
os.makedirs(os.path.join(BASE_DIR, "data", "logs"), exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Configuration du logging
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "data", "logs", "signal_selector.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Constantes
MAX_RETRIES = 3
RETRY_DELAY = 2.0


class SignalSelector:

    def __init__(
        self, config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml")
    ):
        self.log_buffer = []
        self.cache = {}
        self.buffer = deque(maxlen=1000)
        try:
            self.config = self.load_config(config_path)
            self.buffer_size = self.config.get("buffer_size", 100)
            self.max_cache_size = self.config.get("max_cache_size", 1000)
            self.buffer_maxlen = self.config.get("buffer_maxlen", 1000)
            self.buffer = deque(maxlen=self.buffer_maxlen)
            miya_speak(
                "SignalSelector initialisé",
                tag="SIGNAL_SELECTOR",
                voice_profile="calm",
                priority=2,
            )
            send_alert("SignalSelector initialisé", priority=2)
            send_telegram_alert("SignalSelector initialisé")
            logger.info("SignalSelector initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config_path": config_path}, compress=False)
        except Exception as e:
            error_msg = f"Erreur initialisation SignalSelector: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="SIGNAL_SELECTOR", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            self.config = {
                "feature_weights": {
                    "rsi_14": 0.15,
                    "atr_14": 0.10,
                    "trend_strength": 0.10,
                    "gex": 0.15,
                    "iv_atm": 0.10,
                    "gex_slope": 0.10,
                    "pca_orderflow_1": 0.10,
                    "pca_orderflow_2": 0.10,
                    "confidence_drop_rate": 0.05,
                    "spy_lead_return": 0.05,
                    "sector_leader_correlation": 0.05,
                },
                "sgc_threshold": 0.7,
                "window_size": 20,
                "buffer_size": 100,
                "max_cache_size": 1000,
                "buffer_maxlen": 1000,
                "cache_hours": 24,
                "regime_weights": {
                    "trend": {
                        "rsi_14": 1.5,
                        "trend_strength": 1.5,
                        "gex": 1.2,
                        "pca_orderflow_1": 1.2,
                    },
                    "range": {
                        "atr_14": 1.5,
                        "iv_atm": 1.5,
                        "gex_slope": 1.2,
                        "pca_orderflow_2": 1.2,
                    },
                    "defensive": {
                        "confidence_drop_rate": 1.5,
                        "spy_lead_return": 1.2,
                        "sector_leader_correlation": 1.2,
                    },
                },
            }
            self.buffer_size = 100
            self.max_cache_size = 1000
            self.buffer_maxlen = 1000
            self.buffer = deque(maxlen=self.buffer_maxlen)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Charge la configuration via yaml."""
        start_time = time.time()
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            if "signal_selector" not in config:
                raise ValueError(
                    "Clé 'signal_selector' manquante dans la configuration"
                )
            required_keys = ["feature_weights", "sgc_threshold", "window_size"]
            missing_keys = [
                key for key in required_keys if key not in config["signal_selector"]
            ]
            if missing_keys:
                raise ValueError(
                    f"Clés manquantes dans 'signal_selector': {missing_keys}"
                )
            latency = time.time() - start_time
            cache_key = hashlib.sha256(
                str(config["signal_selector"]).encode()
            ).hexdigest()
            self.cache[cache_key] = config["signal_selector"]
            if len(self.cache) > self.max_cache_size:
                self.cache.pop(next(iter(self.cache)))
            miya_speak(
                f"Configuration signal_selector chargée: {config['signal_selector']}",
                tag="SIGNAL_SELECTOR",
                voice_profile="calm",
                priority=2,
            )
            send_alert("Configuration signal_selector chargée", priority=2)
            send_telegram_alert("Configuration signal_selector chargée")
            logger.info("Configuration signal_selector chargée")
            self.log_performance("load_config", latency, success=True)
            self.save_snapshot(
                "load_config", {"config_path": config_path}, compress=False
            )
            return config["signal_selector"]
        except FileNotFoundError as e:
            latency = time.time() - start_time
            error_msg = (
                f"Fichier config introuvable: {config_path}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, tag="SIGNAL_SELECTOR", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("load_config", latency, success=False, error=str(e))
            raise
        except yaml.YAMLError as e:
            latency = time.time() - start_time
            error_msg = f"Erreur format YAML: {config_path}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="SIGNAL_SELECTOR", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("load_config", latency, success=False, error=str(e))
            raise
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur chargement config: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="SIGNAL_SELECTOR", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("load_config", latency, success=False, error=str(e))
            return {
                "feature_weights": {
                    "rsi_14": 0.15,
                    "atr_14": 0.10,
                    "trend_strength": 0.10,
                    "gex": 0.15,
                    "iv_atm": 0.10,
                    "gex_slope": 0.10,
                    "pca_orderflow_1": 0.10,
                    "pca_orderflow_2": 0.10,
                    "confidence_drop_rate": 0.05,
                    "spy_lead_return": 0.05,
                    "sector_leader_correlation": 0.05,
                },
                "sgc_threshold": 0.7,
                "window_size": 20,
                "buffer_size": 100,
                "max_cache_size": 1000,
                "buffer_maxlen": 1000,
                "cache_hours": 24,
                "regime_weights": {
                    "trend": {
                        "rsi_14": 1.5,
                        "trend_strength": 1.5,
                        "gex": 1.2,
                        "pca_orderflow_1": 1.2,
                    },
                    "range": {
                        "atr_14": 1.5,
                        "iv_atm": 1.5,
                        "gex_slope": 1.2,
                        "pca_orderflow_2": 1.2,
                    },
                    "defensive": {
                        "confidence_drop_rate": 1.5,
                        "spy_lead_return": 1.2,
                        "sector_leader_correlation": 1.2,
                    },
                },
            }

    def load_config_with_manager(
        self, config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml")
    ) -> Dict[str, Any]:
        """Charge la configuration via config_manager."""

        def load_yaml():
            config = config_manager.get_config(os.path.basename(config_path))
            if "signal_selector" not in config:
                raise ValueError(
                    "Clé 'signal_selector' manquante dans la configuration"
                )
            required_keys = [
                "feature_weights",
                "sgc_threshold",
                "window_size",
                "regime_weights",
            ]
            missing_keys = [
                key for key in required_keys if key not in config["signal_selector"]
            ]
            if missing_keys:
                raise ValueError(
                    f"Clés manquantes dans 'signal_selector': {missing_keys}"
                )
            return config["signal_selector"]

        try:
            start_time = time.time()
            config = self.with_retries(load_yaml)
            cache_key = hashlib.sha256(str(config).encode()).hexdigest()
            self.cache[cache_key] = config
            if len(self.cache) > self.max_cache_size:
                self.cache.pop(next(iter(self.cache)))
            latency = time.time() - start_time
            miya_speak(
                "Configuration signal_selector chargée via config_manager",
                tag="SIGNAL_SELECTOR",
                voice_profile="calm",
                priority=2,
            )
            send_alert(
                "Configuration signal_selector chargée via config_manager", priority=2
            )
            send_telegram_alert(
                "Configuration signal_selector chargée via config_manager"
            )
            logger.info("Configuration signal_selector chargée via config_manager")
            self.log_performance("load_config_with_manager", latency, success=True)
            self.save_snapshot(
                "load_config_with_manager", {"config_path": config_path}, compress=False
            )
            return config
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur chargement config via config_manager: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="SIGNAL_SELECTOR", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "load_config_with_manager", latency, success=False, error=str(e)
            )
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
                        tag="SIGNAL_SELECTOR",
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
                    tag="SIGNAL_SELECTOR",
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
                    tag="SIGNAL_SELECTOR",
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
                tag="SIGNAL_SELECTOR",
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
                    tag="SIGNAL_SELECTOR",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(f"Snapshot size {file_size:.2f} MB exceeds 1 MB", priority=3)
                send_telegram_alert(f"Snapshot size {file_size:.2f} MB exceeds 1 MB")
            latency = time.time() - start_time
            miya_speak(
                f"Snapshot {snapshot_type} sauvegardé: {save_path}",
                tag="SIGNAL_SELECTOR",
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
                tag="SIGNAL_SELECTOR",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}", priority=3
            )
            send_telegram_alert(f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}")
            logger.error(f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}")

    def validate_shap_features(self, features: List[str]) -> bool:
        """Valide les features par rapport aux top 150 SHAP."""
        try:
            start_time = time.time()
            if not os.path.exists(FEATURE_IMPORTANCE_PATH):
                miya_alerts(
                    "Fichier SHAP manquant",
                    tag="SIGNAL_SELECTOR",
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
                    tag="SIGNAL_SELECTOR",
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
                    tag="SIGNAL_SELECTOR",
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
                tag="SIGNAL_SELECTOR",
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
                tag="SIGNAL_SELECTOR",
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

    def validate_iqfeed_data(self, data: pd.DataFrame) -> bool:
        """Vérifie la cohérence des données IQFeed."""
        try:
            start_time = time.time()
            required_cols = [
                "timestamp",
                "gex",
                "iv_atm",
                "gex_slope",
                "pca_orderflow_1",
                "pca_orderflow_2",
                "confidence_drop_rate",
            ]
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                error_msg = f"Colonnes IQFeed manquantes: {missing_cols}"
                miya_alerts(
                    error_msg, tag="SIGNAL_SELECTOR", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return False
            if data[required_cols].isna().any().any():
                error_msg = "Valeurs NaN dans les données IQFeed"
                miya_alerts(
                    error_msg, tag="SIGNAL_SELECTOR", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return False
            if not np.isfinite(
                data[required_cols].select_dtypes(include=[np.number]).values
            ).all():
                error_msg = "Valeurs infinies dans les données IQFeed"
                miya_alerts(
                    error_msg, tag="SIGNAL_SELECTOR", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return False
            latency = time.time() - start_time
            miya_speak(
                "Données IQFeed validées",
                tag="SIGNAL_SELECTOR",
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
                num_columns=len(required_cols),
            )
            return True
        except Exception as e:
            latency = time.time() - start_time
            error_msg = (
                f"Erreur validation données IQFeed: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, tag="SIGNAL_SELECTOR", voice_profile="urgent", priority=4
            )
            send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "validate_iqfeed_data", latency, success=False, error=str(e)
            )
            return False

    def plot_sgc(
        self, sgc: pd.Series, secondary_metrics: pd.DataFrame, timestamp: str
    ) -> None:
        """Génère des visualisations du SGC et des métriques secondaires."""
        start_time = time.time()
        try:
            timestamp_safe = timestamp.replace(":", "-")
            os.makedirs(FIGURES_DIR, exist_ok=True)

            plt.figure(figsize=(12, 6))
            plt.plot(sgc.index, sgc, label="SGC", color="blue")
            plt.plot(
                secondary_metrics.index,
                secondary_metrics["raw_sgc"],
                label="Raw SGC",
                color="green",
                alpha=0.5,
            )
            plt.plot(
                secondary_metrics.index,
                secondary_metrics["volatility_adjustment"],
                label="Volatility Adjustment",
                color="red",
                alpha=0.5,
            )
            plt.plot(
                secondary_metrics.index,
                secondary_metrics["confidence_adjustment"],
                label="Confidence Adjustment",
                color="purple",
                alpha=0.5,
            )
            plt.title(f"SGC et Métriques Secondaires - {timestamp}")
            plt.xlabel("Index")
            plt.ylabel("Valeur")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(FIGURES_DIR, f"sgc_temporal_{timestamp_safe}.png"))
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.hist(sgc, bins=20, alpha=0.7, label="SGC", color="blue")
            plt.title(f"Distribution du SGC - {timestamp}")
            plt.xlabel("SGC")
            plt.ylabel("Fréquence")
            plt.legend()
            plt.grid(True)
            plt.savefig(
                os.path.join(FIGURES_DIR, f"sgc_distribution_{timestamp_safe}.png")
            )
            plt.close()

            latency = time.time() - start_time
            miya_speak(
                f"Visualisations SGC générées: {FIGURES_DIR}",
                tag="SIGNAL_SELECTOR",
                voice_profile="calm",
                priority=2,
            )
            send_alert(f"Visualisations SGC générées: {FIGURES_DIR}", priority=2)
            send_telegram_alert(f"Visualisations SGC générées: {FIGURES_DIR}")
            logger.info(f"Visualisations SGC générées: {FIGURES_DIR}")
            self.log_performance("plot_sgc", latency, success=True)
        except Exception as e:
            latency = time.time() - start_time
            error_msg = (
                f"Erreur génération visualisations: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, tag="SIGNAL_SELECTOR", voice_profile="urgent", priority=2
            )
            send_alert(error_msg, priority=2)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("plot_sgc", latency, success=False, error=str(e))

    def normalize_and_weight(
        self,
        data: pd.DataFrame,
        feature_weights: Dict[str, float],
        regime_probs: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """Normalise et pondère les features pour le calcul du SGC."""
        start_time = time.time()
        try:
            scaler = MinMaxScaler()
            normalized = pd.DataFrame(
                scaler.fit_transform(data), columns=data.columns, index=data.index
            )

            if not ((normalized >= 0) & (normalized <= 1)).all().all():
                error_msg = "Valeurs hors bornes [0, 1] après normalisation"
                miya_alerts(
                    error_msg, tag="SIGNAL_SELECTOR", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                normalized = normalized.clip(0, 1)

            if not np.isfinite(normalized.values).all():
                error_msg = "Valeurs infinies ou NaN après normalisation"
                miya_alerts(
                    error_msg, tag="SIGNAL_SELECTOR", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                normalized = normalized.fillna(0.0)

            regime_probs = regime_probs or {
                "trend": 0.4,
                "range": 0.5,
                "defensive": 0.1,
            }
            if abs(sum(regime_probs.values()) - 1.0) > 1e-6:
                error_msg = f"Somme des probabilités de régime non égale à 1: {sum(regime_probs.values())}"
                miya_alerts(
                    error_msg, tag="SIGNAL_SELECTOR", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                regime_probs = {
                    k: v / sum(regime_probs.values()) for k, v in regime_probs.items()
                }

            weighted_sum = pd.Series(0.0, index=data.index)
            regime_weights = self.config.get("regime_weights", {})
            for col, base_weight in feature_weights.items():
                if col in normalized.columns:
                    adjusted_weight = base_weight
                    for regime, prob in regime_probs.items():
                        regime_factor = regime_weights.get(regime, {}).get(col, 1.0)
                        adjusted_weight += base_weight * prob * (regime_factor - 1.0)
                    weighted_sum += normalized[col] * adjusted_weight

            latency = time.time() - start_time
            self.log_performance(
                "normalize_and_weight", latency, success=True, num_rows=len(data)
            )
            self.save_snapshot(
                "normalize_and_weight",
                {"num_rows": len(data), "regime_probs": regime_probs},
                compress=False,
            )
            return weighted_sum
        except Exception as e:
            latency = time.time() - start_time
            error_msg = (
                f"Erreur normalisation/pondération: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, tag="SIGNAL_SELECTOR", voice_profile="urgent", priority=4
            )
            send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "normalize_and_weight", latency, success=False, error=str(e)
            )
            return pd.Series(0.0, index=data.index)

    def calculate_sgc(
        self,
        data: pd.DataFrame,
        regime_probs: Dict[str, float],
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """Calcule le Score Global de Confiance (SGC)."""
        try:
            start_time = time.time()
            config = config or self.load_config_with_manager()
            feature_weights = config.get("feature_weights", {})
            sgc_threshold = config.get("sgc_threshold", 0.7)
            config.get("window_size", 20)

            cache_key = hashlib.sha256(
                f"{json.dumps(config)}_{data.to_json()}_{json.dumps(regime_probs)}".encode()
            ).hexdigest()
            current_time = datetime.now()
            expired_keys = [
                k
                for k, v in self.cache.items()
                if (current_time - v["timestamp"]).total_seconds()
                > self.config.get("cache_hours", 24) * 3600
            ]
            for k in expired_keys:
                self.cache.pop(k)
            if cache_key in self.cache:
                miya_speak(
                    "Résultat SGC récupéré du cache",
                    tag="SIGNAL_SELECTOR",
                    voice_profile="calm",
                    priority=1,
                )
                send_alert("Résultat SGC récupéré du cache", priority=1)
                send_telegram_alert("Résultat SGC récupéré du cache")
                self.log_performance("calculate_sgc_cache_hit", 0, success=True)
                return self.cache[cache_key]["result"]

            if data.empty:
                error_msg = "DataFrame vide"
                miya_alerts(
                    error_msg, tag="SIGNAL_SELECTOR", voice_profile="urgent", priority=5
                )
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            data = data.copy()
            if not self.validate_iqfeed_data(data):
                error_msg = "Échec validation données IQFeed"
                miya_alerts(
                    error_msg, tag="SIGNAL_SELECTOR", voice_profile="urgent", priority=5
                )
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            if "timestamp" not in data.columns:
                error_msg = "Colonne 'timestamp' manquante"
                miya_alerts(
                    error_msg, tag="SIGNAL_SELECTOR", voice_profile="urgent", priority=5
                )
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
            if data["timestamp"].isna().any():
                error_msg = "NaN dans les timestamps"
                miya_alerts(
                    error_msg, tag="SIGNAL_SELECTOR", voice_profile="urgent", priority=5
                )
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)
            if not data["timestamp"].is_monotonic_increasing:
                error_msg = "Timestamps non croissants"
                miya_alerts(
                    error_msg, tag="SIGNAL_SELECTOR", voice_profile="urgent", priority=5
                )
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            expected_count = config.get("expected_count", 81)
            required_features = list(feature_weights.keys())
            if expected_count == 320:
                available_features = [
                    col for col in required_features if col in data.columns
                ]
            else:
                available_features = [
                    col
                    for col in required_features
                    if col in data.columns and col in obs_t
                ]
            self.validate_shap_features(available_features)
            missing_features = [
                col for col in required_features if col not in available_features
            ]
            if missing_features:
                miya_speak(
                    f"Features manquantes: {missing_features}, imputées à 0",
                    tag="SIGNAL_SELECTOR",
                    voice_profile="warning",
                    priority=2,
                )
                send_alert(f"Features manquantes: {missing_features}", priority=2)
                send_telegram_alert(f"Features manquantes: {missing_features}")
                logger.warning(f"Features manquantes: {missing_features}")
                for col in missing_features:
                    data[col] = 0.0

            for col in available_features:
                data[col] = pd.to_numeric(
                    data[col], errors="coerce", downcast="float32"
                )
                if data[col].isna().any():
                    data[col] = (
                        data[col]
                        .interpolate(method="linear", limit_direction="both")
                        .fillna(data[col].median())
                    )
                    if data[col].isna().sum() / len(data) > 0.1:
                        error_msg = f"Plus de 10% de NaN dans {col}"
                        miya_alerts(
                            error_msg,
                            tag="SIGNAL_SELECTOR",
                            voice_profile="urgent",
                            priority=5,
                        )
                        send_alert(error_msg, priority=5)
                        send_telegram_alert(error_msg)
                        raise ValueError(error_msg)

            raw_sgc = self.normalize_and_weight(
                data[available_features], feature_weights, regime_probs
            )

            scaler = MinMaxScaler()
            volatility_adjustment = np.clip(
                1 - scaler.fit_transform(data[["atr_14"]].fillna(0.5))[:, 0], 0.1, 1.0
            )
            confidence_adjustment = np.clip(
                scaler.fit_transform(data[["confidence_drop_rate"]].fillna(0.5))[:, 0],
                0.1,
                1.0,
            )
            sgc = raw_sgc * volatility_adjustment * confidence_adjustment

            sgc = pd.Series(sgc, index=data.index).clip(0, 1)
            sgc = sgc.where(sgc >= sgc_threshold, 0.0)

            secondary_metrics = pd.DataFrame(
                {
                    "volatility_adjustment": volatility_adjustment,
                    "confidence_adjustment": confidence_adjustment,
                    "raw_sgc": raw_sgc,
                },
                index=data.index,
            )

            sgc_mean = sgc.mean()
            if sgc_mean < 0.3:
                miya_alerts(
                    f"SGC moyen faible: {sgc_mean:.2f}",
                    tag="SIGNAL_SELECTOR",
                    voice_profile="urgent",
                    priority=4,
                )
                send_alert(f"SGC moyen faible: {sgc_mean:.2f}", priority=4)
                send_telegram_alert(f"SGC moyen faible: {sgc_mean:.2f}")
            elif sgc_mean > 0.8:
                miya_speak(
                    f"Signal fort détecté, SGC moyen: {sgc_mean:.2f}",
                    tag="SIGNAL_SELECTOR",
                    voice_profile="calm",
                    priority=2,
                )
                send_alert(
                    f"Signal fort détecté, SGC moyen: {sgc_mean:.2f}", priority=2
                )
                send_telegram_alert(f"Signal fort détecté, SGC moyen: {sgc_mean:.2f}")

            self.cache[cache_key] = {
                "result": (sgc, secondary_metrics),
                "timestamp": current_time,
            }
            if len(self.cache) > self.max_cache_size:
                oldest_key = min(self.cache, key=lambda k: self.cache[k]["timestamp"])
                self.cache.pop(oldest_key)

            latency = time.time() - start_time
            miya_speak(
                f"SGC calculé, moyenne: {sgc_mean:.2f}",
                tag="SIGNAL_SELECTOR",
                voice_profile="calm",
                priority=1,
            )
            send_alert(f"SGC calculé, moyenne: {sgc_mean:.2f}", priority=1)
            send_telegram_alert(f"SGC calculé, moyenne: {sgc_mean:.2f}")
            logger.info(
                f"SGC calculé, moyenne: {sgc_mean:.2f}, CPU: {psutil.cpu_percent()}%"
            )
            self.log_performance(
                "calculate_sgc",
                latency,
                success=True,
                num_rows=len(data),
                sgc_mean=sgc_mean,
                confidence_drop_rate=float(data["confidence_drop_rate"].mean()),
            )
            self.save_snapshot(
                "calculate_sgc",
                {
                    "sgc_mean": float(sgc_mean),
                    "num_rows": len(data),
                    "regime_probs": regime_probs,
                    "confidence_drop_rate": float(data["confidence_drop_rate"].mean()),
                },
                compress=False,
            )
            self.plot_sgc(
                sgc,
                secondary_metrics,
                data["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S"),
            )
            return sgc, secondary_metrics
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans calculate_sgc: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="SIGNAL_SELECTOR", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("calculate_sgc", latency, success=False, error=str(e))
            self.save_snapshot("calculate_sgc", {"error": str(e)}, compress=False)
            return pd.Series(0.0, index=data.index), pd.DataFrame(
                {
                    "volatility_adjustment": 0.0,
                    "confidence_adjustment": 0.0,
                    "raw_sgc": 0.0,
                },
                index=data.index,
            )

    def calculate_incremental_sgc(
        self,
        row: pd.Series,
        regime_probs: Dict[str, float],
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Calcule le SGC incrémental pour une seule ligne."""
        try:
            start_time = time.time()
            config = config or self.load_config_with_manager()
            window_size = config.get("window_size", 20)

            row = row.copy()
            if "timestamp" not in row.index:
                error_msg = "Timestamp manquant dans la ligne"
                miya_alerts(
                    error_msg, tag="SIGNAL_SELECTOR", voice_profile="urgent", priority=5
                )
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            row["timestamp"] = pd.to_datetime(row["timestamp"], errors="coerce")
            if pd.isna(row["timestamp"]):
                error_msg = "Timestamp invalide dans la ligne"
                miya_alerts(
                    error_msg, tag="SIGNAL_SELECTOR", voice_profile="urgent", priority=5
                )
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            self.buffer.append(row.to_frame().T)
            if len(self.buffer) < window_size:
                return 0.0, {
                    "volatility_adjustment": 0.0,
                    "confidence_adjustment": 0.0,
                    "raw_sgc": 0.0,
                }

            data = pd.concat(list(self.buffer), ignore_index=True).tail(window_size)
            if not self.validate_iqfeed_data(data):
                error_msg = "Échec validation données IQFeed"
                miya_alerts(
                    error_msg, tag="SIGNAL_SELECTOR", voice_profile="urgent", priority=5
                )
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            sgc_series, secondary_metrics = self.calculate_sgc(
                data, regime_probs, config
            )
            sgc = sgc_series.iloc[-1]
            metrics = {
                "volatility_adjustment": float(
                    secondary_metrics["volatility_adjustment"].iloc[-1]
                ),
                "confidence_adjustment": float(
                    secondary_metrics["confidence_adjustment"].iloc[-1]
                ),
                "raw_sgc": float(secondary_metrics["raw_sgc"].iloc[-1]),
            }

            latency = time.time() - start_time
            miya_speak(
                f"Incremental SGC calculé: {sgc:.2f}",
                tag="SIGNAL_SELECTOR",
                voice_profile="calm",
                priority=1,
            )
            send_alert(f"Incremental SGC calculé: {sgc:.2f}", priority=1)
            send_telegram_alert(f"Incremental SGC calculé: {sgc:.2f}")
            logger.info(
                f"Incremental SGC calculé: {sgc:.2f}, CPU: {psutil.cpu_percent()}%"
            )
            self.log_performance(
                "calculate_incremental_sgc",
                latency,
                success=True,
                sgc=sgc,
                confidence_drop_rate=float(row.get("confidence_drop_rate", 0.0)),
            )
            self.save_snapshot(
                "calculate_incremental_sgc",
                {
                    "sgc": sgc,
                    "metrics": metrics,
                    "regime_probs": regime_probs,
                    "confidence_drop_rate": float(row.get("confidence_drop_rate", 0.0)),
                },
                compress=False,
            )
            return sgc, metrics
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans calculate_incremental_sgc: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="SIGNAL_SELECTOR", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "calculate_incremental_sgc", latency, success=False, error=str(e)
            )
            self.save_snapshot(
                "calculate_incremental_sgc", {"error": str(e)}, compress=False
            )
            return 0.0, {
                "volatility_adjustment": 0.0,
                "confidence_adjustment": 0.0,
                "raw_sgc": 0.0,
            }


if __name__ == "__main__":
    try:
        selector = SignalSelector()
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "rsi_14": np.random.uniform(30, 70, 100),
                "atr_14": np.random.uniform(0.5, 2.0, 100),
                "trend_strength": np.random.uniform(10, 50, 100),
                "gex": np.random.uniform(-1000, 1000, 100),
                "iv_atm": np.random.uniform(0.1, 0.3, 100),
                "gex_slope": np.random.uniform(-0.05, 0.05, 100),
                "pca_orderflow_1": np.random.uniform(-1, 1, 100),
                "pca_orderflow_2": np.random.uniform(-1, 1, 100),
                "confidence_drop_rate": np.random.uniform(0, 1, 100),
                "spy_lead_return": np.random.uniform(-0.02, 0.02, 100),
                "sector_leader_correlation": np.random.uniform(-1, 1, 100),
            }
        )
        regime_probs = {"trend": 0.4, "range": 0.5, "defensive": 0.1}
        sgc, secondary_metrics = selector.calculate_sgc(data, regime_probs)
        selector.plot_sgc(sgc, secondary_metrics, "2025-04-14 09:00")
        print("SGC (premières 5 valeurs):")
        print(sgc.head())
        print("Métriques secondaires (premières 5 lignes):")
        print(secondary_metrics.head())
        miya_speak(
            "Test calculate_sgc terminé", tag="TEST", voice_profile="calm", priority=1
        )
        send_alert("Test calculate_sgc terminé", priority=1)
        send_telegram_alert("Test calculate_sgc terminé")
        logger.info("Test calculate_sgc terminé")
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
