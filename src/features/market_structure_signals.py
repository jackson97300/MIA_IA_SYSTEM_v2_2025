# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/features/market_structure_signals.py
# Gère les signaux cross-market à partir des données IQFeed.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Calcule les signaux cross-market (ex. : spy_lead_return, tlt_correlation),
#        intègre la volatilité (méthode 1) dans spy_lead_return, avec cache intermédiaire,
#        logs psutil, validation SHAP (méthode 17), et compatibilité top 150 SHAP features.
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
# - data/iqfeed/merged_data.csv
# - data/features/feature_importance.csv
#
# Outputs :
# - data/features/cache/market_signals/*.csv
# - data/logs/market_signals_performance.csv
# - data/market_signals_snapshots/*.json (option *.json.gz)
#
# Notes :
# - Conforme à structure.txt (version 2.1.3, 2025-05-13).
# - Utilise IQFeed via data_provider.py, avec retries (max 3, délai 2^attempt secondes).
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des signaux.
# - Intègre validation SHAP (Phase 17) pour assurer la conformité avec les top 150 SHAP features.
# - Sauvegarde des snapshots JSON non compressés par défaut avec option de compression gzip.
# - Envoie des alertes via AlertManager et telegram_alert pour les erreurs critiques et succès.
# - Tests unitaires disponibles dans tests/test_market_structure_signals.py.
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

import pandas as pd
import psutil
import yaml

from src.model.utils.alert_manager import send_alert
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.model.utils.obs_template import obs_t
from src.utils.telegram_alert import send_telegram_alert

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data", "features", "cache", "market_signals")
PERF_LOG_PATH = os.path.join(BASE_DIR, "data", "logs", "market_signals_performance.csv")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "market_signals_snapshots")
FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR, "data", "features", "feature_importance.csv"
)

# Création des répertoires
os.makedirs(os.path.join(BASE_DIR, "data", "logs"), exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Configuration du logging
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "data", "logs", "market_signals.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Constantes
MAX_RETRIES = 3
RETRY_DELAY = 2.0


class MarketStructureSignals:
    """Gère les signaux cross-market avec cache, logs, et volatilité conditionnelle."""

    def __init__(
        self, config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml")
    ):
        """Initialise le générateur de signaux cross-market."""
        self.log_buffer = []
        self.cache = {}
        try:
            self.config = self.load_config(config_path)
            os.makedirs(CACHE_DIR, exist_ok=True)
            os.makedirs(os.path.dirname(PERF_LOG_PATH), exist_ok=True)
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            miya_speak(
                "MarketStructureSignals initialisé",
                tag="CROSS",
                voice_profile="calm",
                priority=2,
            )
            send_alert("MarketStructureSignals initialisé", priority=2)
            send_telegram_alert("MarketStructureSignals initialisé")
            logger.info("MarketStructureSignals initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config_path": config_path}, compress=False)
        except Exception as e:
            error_msg = f"Erreur initialisation MarketStructureSignals: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="CROSS", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            self.config = {
                "lead_window": 5,
                "corr_window": 20,
                "vol_skew_window": 20,
                "vol_threshold": 0.5,
                "buffer_size": 100,
                "max_cache_size": 1000,
                "cache_hours": 24,
            }

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Charge la configuration depuis es_config.yaml."""

        def load_yaml():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            if "market_structure_signals" not in config:
                raise ValueError(
                    "Clé 'market_structure_signals' manquante dans la configuration"
                )
            required_keys = [
                "lead_window",
                "corr_window",
                "vol_skew_window",
                "vol_threshold",
            ]
            missing_keys = [
                key
                for key in required_keys
                if key not in config["market_structure_signals"]
            ]
            if missing_keys:
                raise ValueError(
                    f"Clés manquantes dans 'market_structure_signals': {missing_keys}"
                )
            return config["market_structure_signals"]

        try:
            start_time = time.time()
            config = self.with_retries(load_yaml)
            cache_key = hashlib.sha256(str(config).encode()).hexdigest()
            self.cache[cache_key] = {"config": config, "timestamp": datetime.now()}
            if len(self.cache) > self.config.get("max_cache_size", 1000):
                self.cache.pop(next(iter(self.cache)))
            latency = time.time() - start_time
            miya_speak(
                "Configuration market_structure_signals chargée",
                tag="CROSS",
                voice_profile="calm",
                priority=2,
            )
            send_alert("Configuration market_structure_signals chargée", priority=2)
            send_telegram_alert("Configuration market_structure_signals chargée")
            logger.info("Configuration market_structure_signals chargée")
            self.log_performance("load_config", latency, success=True)
            self.save_snapshot(
                "load_config", {"config_path": config_path}, compress=False
            )
            return config
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur chargement config: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="CROSS", voice_profile="urgent", priority=3)
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
                        tag="CROSS",
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
                    tag="CROSS",
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
                    tag="CROSS",
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
                tag="CROSS",
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
                    tag="CROSS",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(f"Snapshot size {file_size:.2f} MB exceeds 1 MB", priority=3)
                send_telegram_alert(f"Snapshot size {file_size:.2f} MB exceeds 1 MB")
            latency = time.time() - start_time
            miya_speak(
                f"Snapshot {snapshot_type} sauvegardé: {save_path}",
                tag="CROSS",
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
                tag="CROSS",
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
                    tag="CROSS",
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
                    tag="CROSS",
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
                    tag="CROSS",
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
                "SHAP features validées", tag="CROSS", voice_profile="calm", priority=1
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
                tag="CROSS",
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

    def calculate_spy_lead_return(
        self, data: pd.DataFrame, window: int = 5, vol_threshold: float = 0.5
    ) -> pd.Series:
        """Calcule le rendement anticipé de SPY par rapport à l'instrument principal (lead-lag), pondéré par vix_es_correlation (méthode 1)."""
        cache_path = os.path.join(
            CACHE_DIR,
            f"spy_lead_return_{hashlib.sha256(data.to_json().encode()).hexdigest()}.csv",
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
                        "calculate_spy_lead_return_cache", latency, success=True
                    )
                    miya_speak(
                        "Spy lead return chargé depuis cache",
                        tag="CROSS",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert("Spy lead return chargé depuis cache", priority=1)
                    send_telegram_alert("Spy lead return chargé depuis cache")
                    return cached_data["spy_lead_return"]

            required_cols = ["close", "spy_close", "vix_es_correlation", "timestamp"]
            missing_cols = [col for col in required_cols if col not in data.columns]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )
            if len(data) < window:
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(required_cols) - len(missing_cols)}/{len(required_cols)} colonnes valides, {len(data)} lignes)"
                miya_alerts(alert_msg, tag="CROSS", voice_profile="urgent", priority=3)
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            if (
                "close" not in data.columns
                or "spy_close" not in data.columns
                or "vix_es_correlation" not in data.columns
            ):
                raise ValueError(
                    "Colonnes 'close', 'spy_close' ou 'vix_es_correlation' manquantes"
                )

            spy_returns = data["spy_close"].pct_change()
            lead_returns = spy_returns.shift(-window)
            vol_weight = data["vix_es_correlation"].apply(
                lambda x: 1.0 if abs(x) > vol_threshold else 0.5
            )
            weighted_returns = lead_returns * vol_weight
            result = weighted_returns.clip(-0.1, 0.1).fillna(0)

            os.makedirs(CACHE_DIR, exist_ok=True)

            def write_cache():
                cache_df = pd.DataFrame(
                    {"timestamp": data["timestamp"], "spy_lead_return": result}
                )
                cache_df.to_csv(cache_path, index=False, encoding="utf-8")

            self.with_retries(write_cache)
            file_size = os.path.getsize(cache_path) / 1024 / 1024
            if file_size > 1.0:
                miya_alerts(
                    f"Cache spy_lead_return size {file_size:.2f} MB exceeds 1 MB",
                    tag="CROSS",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(
                    f"Cache spy_lead_return size {file_size:.2f} MB exceeds 1 MB",
                    priority=3,
                )
                send_telegram_alert(
                    f"Cache spy_lead_return size {file_size:.2f} MB exceeds 1 MB"
                )

            latency = time.time() - start_time
            self.log_performance(
                "calculate_spy_lead_return",
                latency,
                success=True,
                confidence_drop_rate=confidence_drop_rate,
            )
            miya_speak(
                "Spy lead return calculé", tag="CROSS", voice_profile="calm", priority=1
            )
            send_alert("Spy lead return calculé", priority=1)
            send_telegram_alert("Spy lead return calculé")
            logger.info("Spy lead return calculé")
            self.save_snapshot(
                "calculate_spy_lead_return",
                {"window": window, "vol_threshold": vol_threshold},
                compress=False,
            )
            return result
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans calculate_spy_lead_return: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                "calculate_spy_lead_return", latency, success=False, error=str(e)
            )
            miya_alerts(error_msg, tag="CROSS", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.save_snapshot(
                "calculate_spy_lead_return", {"error": str(e)}, compress=False
            )
            return pd.Series(0.0, index=data.index)

    def calculate_sector_leader_correlation(
        self, data: pd.DataFrame, window: int = 20
    ) -> pd.Series:
        """Calcule la corrélation glissante entre l'instrument principal et le secteur leader."""
        cache_path = os.path.join(
            CACHE_DIR,
            f"sector_leader_correlation_{hashlib.sha256(data.to_json().encode()).hexdigest()}.csv",
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
                        "calculate_sector_leader_correlation_cache",
                        latency,
                        success=True,
                    )
                    miya_speak(
                        "Sector leader correlation chargé depuis cache",
                        tag="CROSS",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert(
                        "Sector leader correlation chargé depuis cache", priority=1
                    )
                    send_telegram_alert("Sector leader correlation chargé depuis cache")
                    return cached_data["sector_leader_correlation"]

            required_cols = ["close", "sector_leader_close", "timestamp"]
            missing_cols = [col for col in required_cols if col not in data.columns]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )
            if len(data) < window:
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(required_cols) - len(missing_cols)}/{len(required_cols)} colonnes valides, {len(data)} lignes)"
                miya_alerts(alert_msg, tag="CROSS", voice_profile="urgent", priority=3)
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            if "close" not in data.columns or "sector_leader_close" not in data.columns:
                raise ValueError("Colonnes 'close' ou 'sector_leader_close' manquantes")
            if data["sector_leader_close"].var() < 1e-6:
                result = pd.Series(0.0, index=data.index)
            else:
                correlation = (
                    data["close"]
                    .rolling(window=window, min_periods=1)
                    .corr(data["sector_leader_close"])
                )
                result = correlation.clip(-1, 1).fillna(0)

            os.makedirs(CACHE_DIR, exist_ok=True)

            def write_cache():
                cache_df = pd.DataFrame(
                    {
                        "timestamp": data["timestamp"],
                        "sector_leader_correlation": result,
                    }
                )
                cache_df.to_csv(cache_path, index=False, encoding="utf-8")

            self.with_retries(write_cache)
            file_size = os.path.getsize(cache_path) / 1024 / 1024
            if file_size > 1.0:
                miya_alerts(
                    f"Cache sector_leader_correlation size {file_size:.2f} MB exceeds 1 MB",
                    tag="CROSS",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(
                    f"Cache sector_leader_correlation size {file_size:.2f} MB exceeds 1 MB",
                    priority=3,
                )
                send_telegram_alert(
                    f"Cache sector_leader_correlation size {file_size:.2f} MB exceeds 1 MB"
                )

            latency = time.time() - start_time
            self.log_performance(
                "calculate_sector_leader_correlation",
                latency,
                success=True,
                confidence_drop_rate=confidence_drop_rate,
            )
            miya_speak(
                "Sector leader correlation calculé",
                tag="CROSS",
                voice_profile="calm",
                priority=1,
            )
            send_alert("Sector leader correlation calculé", priority=1)
            send_telegram_alert("Sector leader correlation calculé")
            logger.info("Sector leader correlation calculé")
            self.save_snapshot(
                "calculate_sector_leader_correlation",
                {"window": window},
                compress=False,
            )
            return result
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans calculate_sector_leader_correlation: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                "calculate_sector_leader_correlation",
                latency,
                success=False,
                error=str(e),
            )
            miya_alerts(error_msg, tag="CROSS", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.save_snapshot(
                "calculate_sector_leader_correlation", {"error": str(e)}, compress=False
            )
            return pd.Series(0.0, index=data.index)

    def calculate_vol_skew_cross_index(
        self, data: pd.DataFrame, window: int = 20
    ) -> pd.Series:
        """Calcule l’asymétrie de volatilité entre l'instrument principal et un indice (ex. VIX)."""
        cache_path = os.path.join(
            CACHE_DIR,
            f"vol_skew_cross_index_{hashlib.sha256(data.to_json().encode()).hexdigest()}.csv",
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
                        "calculate_vol_skew_cross_index_cache", latency, success=True
                    )
                    miya_speak(
                        "Vol skew cross index chargé depuis cache",
                        tag="CROSS",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert("Vol skew cross index chargé depuis cache", priority=1)
                    send_telegram_alert("Vol skew cross index chargé depuis cache")
                    return cached_data["vol_skew_cross_index"]

            required_cols = ["atr_14", "vix_close", "timestamp"]
            missing_cols = [col for col in required_cols if col not in data.columns]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )
            if len(data) < window:
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(required_cols) - len(missing_cols)}/{len(required_cols)} colonnes valides, {len(data)} lignes)"
                miya_alerts(alert_msg, tag="CROSS", voice_profile="urgent", priority=3)
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            if "atr_14" not in data.columns or "vix_close" not in data.columns:
                raise ValueError("Colonnes 'atr_14' ou 'vix_close' manquantes")
            atr_normalized = (
                data["atr_14"]
                / data["atr_14"].rolling(window=window, min_periods=1).mean()
            )
            vix_normalized = (
                data["vix_close"]
                / data["vix_close"].rolling(window=window, min_periods=1).mean()
            )
            vol_skew = (atr_normalized - vix_normalized) / (
                atr_normalized + vix_normalized + 1e-6
            )
            result = vol_skew.clip(-1, 1).fillna(0)

            os.makedirs(CACHE_DIR, exist_ok=True)

            def write_cache():
                cache_df = pd.DataFrame(
                    {"timestamp": data["timestamp"], "vol_skew_cross_index": result}
                )
                cache_df.to_csv(cache_path, index=False, encoding="utf-8")

            self.with_retries(write_cache)
            file_size = os.path.getsize(cache_path) / 1024 / 1024
            if file_size > 1.0:
                miya_alerts(
                    f"Cache vol_skew_cross_index size {file_size:.2f} MB exceeds 1 MB",
                    tag="CROSS",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(
                    f"Cache vol_skew_cross_index size {file_size:.2f} MB exceeds 1 MB",
                    priority=3,
                )
                send_telegram_alert(
                    f"Cache vol_skew_cross_index size {file_size:.2f} MB exceeds 1 MB"
                )

            latency = time.time() - start_time
            self.log_performance(
                "calculate_vol_skew_cross_index",
                latency,
                success=True,
                confidence_drop_rate=confidence_drop_rate,
            )
            miya_speak(
                "Vol skew cross index calculé",
                tag="CROSS",
                voice_profile="calm",
                priority=1,
            )
            send_alert("Vol skew cross index calculé", priority=1)
            send_telegram_alert("Vol skew cross index calculé")
            logger.info("Vol skew cross index calculé")
            self.save_snapshot(
                "calculate_vol_skew_cross_index", {"window": window}, compress=False
            )
            return result
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans calculate_vol_skew_cross_index: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                "calculate_vol_skew_cross_index", latency, success=False, error=str(e)
            )
            miya_alerts(error_msg, tag="CROSS", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.save_snapshot(
                "calculate_vol_skew_cross_index", {"error": str(e)}, compress=False
            )
            return pd.Series(0.0, index=data.index)

    def calculate_bond_equity_risk_spread(self, data: pd.DataFrame) -> pd.Series:
        """Calcule l’écart de risque entre les obligations et les actions."""
        cache_path = os.path.join(
            CACHE_DIR,
            f"bond_equity_risk_spread_{hashlib.sha256(data.to_json().encode()).hexdigest()}.csv",
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
                        "calculate_bond_equity_risk_spread_cache", latency, success=True
                    )
                    miya_speak(
                        "Bond equity risk spread chargé depuis cache",
                        tag="CROSS",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert(
                        "Bond equity risk spread chargé depuis cache", priority=1
                    )
                    send_telegram_alert("Bond equity risk spread chargé depuis cache")
                    return cached_data["bond_equity_risk_spread"]

            required_cols = ["atr_14", "bond_yield", "timestamp"]
            missing_cols = [col for col in required_cols if col not in data.columns]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )
            if len(data) < 1:
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(required_cols) - len(missing_cols)}/{len(required_cols)} colonnes valides, {len(data)} lignes)"
                miya_alerts(alert_msg, tag="CROSS", voice_profile="urgent", priority=3)
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            if "atr_14" not in data.columns or "bond_yield" not in data.columns:
                raise ValueError("Colonnes 'atr_14' ou 'bond_yield' manquantes")
            risk_spread = data["atr_14"] - data["bond_yield"]
            result = risk_spread.clip(-10, 10).fillna(0)

            os.makedirs(CACHE_DIR, exist_ok=True)

            def write_cache():
                cache_df = pd.DataFrame(
                    {"timestamp": data["timestamp"], "bond_equity_risk_spread": result}
                )
                cache_df.to_csv(cache_path, index=False, encoding="utf-8")

            self.with_retries(write_cache)
            file_size = os.path.getsize(cache_path) / 1024 / 1024
            if file_size > 1.0:
                miya_alerts(
                    f"Cache bond_equity_risk_spread size {file_size:.2f} MB exceeds 1 MB",
                    tag="CROSS",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(
                    f"Cache bond_equity_risk_spread size {file_size:.2f} MB exceeds 1 MB",
                    priority=3,
                )
                send_telegram_alert(
                    f"Cache bond_equity_risk_spread size {file_size:.2f} MB exceeds 1 MB"
                )

            latency = time.time() - start_time
            self.log_performance(
                "calculate_bond_equity_risk_spread",
                latency,
                success=True,
                confidence_drop_rate=confidence_drop_rate,
            )
            miya_speak(
                "Bond equity risk spread calculé",
                tag="CROSS",
                voice_profile="calm",
                priority=1,
            )
            send_alert("Bond equity risk spread calculé", priority=1)
            send_telegram_alert("Bond equity risk spread calculé")
            logger.info("Bond equity risk spread calculé")
            self.save_snapshot("calculate_bond_equity_risk_spread", {}, compress=False)
            return result
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans calculate_bond_equity_risk_spread: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                "calculate_bond_equity_risk_spread",
                latency,
                success=False,
                error=str(e),
            )
            miya_alerts(error_msg, tag="CROSS", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.save_snapshot(
                "calculate_bond_equity_risk_spread", {"error": str(e)}, compress=False
            )
            return pd.Series(0.0, index=data.index)


def calculate_cross_market_signals(
    self,
    data: pd.DataFrame,
    config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml"),
) -> pd.DataFrame:
    """Calcule les signaux cross-market à partir des données IQFeed."""
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
        cache_path = os.path.join(
            CACHE_DIR,
            f"cross_market_signals_{hashlib.sha256(data.to_json().encode()).hexdigest()}.csv",
        )
        if os.path.exists(cache_path):
            cached_data = pd.read_csv(cache_path)
            if len(cached_data) == len(data) and cached_data["timestamp"].equals(
                data["timestamp"]
            ):
                latency = time.time() - start_time
                self.log_performance(
                    "calculate_cross_market_signals_cache_hit", latency, success=True
                )
                miya_speak(
                    "Signaux cross-market chargés depuis cache",
                    tag="CROSS",
                    voice_profile="calm",
                    priority=1,
                )
                send_alert("Signaux cross-market chargés depuis cache", priority=1)
                send_telegram_alert("Signaux cross-market chargés depuis cache")
                return cached_data

        config = self.load_config(config_path)
        lead_window = config.get("lead_window", 5)
        corr_window = config.get("corr_window", 20)
        vol_skew_window = config.get("vol_skew_window", 20)
        vol_threshold = config.get("vol_threshold", 0.5)

        if data.empty:
            error_msg = "DataFrame vide"
            miya_alerts(error_msg, tag="CROSS", voice_profile="urgent", priority=5)
            send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            raise ValueError(error_msg)

        required_cols = ["timestamp", "close"]
        optional_cols = [
            "spy_close",
            "sector_leader_close",
            "vix_close",
            "bond_yield",
            "atr_14",
            "vix_es_correlation",
        ]
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
        if len(data) < max(lead_window, corr_window, vol_skew_window):
            confidence_drop_rate = max(confidence_drop_rate, 0.5)
        if confidence_drop_rate > 0.5:
            alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(required_cols) + len(optional_cols) - len(missing_required_cols) - len(missing_optional_cols)}/{len(required_cols) + len(optional_cols)} colonnes valides, {len(data)} lignes)"
            miya_alerts(alert_msg, tag="CROSS", voice_profile="urgent", priority=3)
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
                    tag="CROSS",
                    voice_profile="warning",
                    priority=2,
                )
                send_alert(
                    "Colonne timestamp manquante, création par défaut", priority=2
                )
                send_telegram_alert("Colonne timestamp manquante, création par défaut")
            else:
                data[col] = (
                    data[col]
                    .interpolate(method="linear", limit_direction="both")
                    .fillna(data[col].median() if col in data.columns else 0)
                )
                miya_speak(
                    f"Colonne '{col}' manquante, imputée",
                    tag="CROSS",
                    voice_profile="warning",
                    priority=2,
                )
                send_alert(f"Colonne '{col}' manquante, imputée", priority=2)
                send_telegram_alert(f"Colonne '{col}' manquante, imputée")
        for col in missing_optional_cols:
            if col == "spy_close":
                data[col] = (
                    data["close"]
                    .rolling(window=5, min_periods=1)
                    .mean()
                    .fillna(data["close"])
                )
                miya_speak(
                    "Colonne spy_close manquante, simulée avec moyenne glissante",
                    tag="CROSS",
                    voice_profile="warning",
                    priority=2,
                )
                send_alert(
                    "Colonne spy_close manquante, simulée avec moyenne glissante",
                    priority=2,
                )
                send_telegram_alert(
                    "Colonne spy_close manquante, simulée avec moyenne glissante"
                )
            elif col == "sector_leader_close":
                data[col] = (
                    data["close"]
                    .rolling(window=5, min_periods=1)
                    .mean()
                    .fillna(data["close"])
                )
                miya_speak(
                    "Colonne sector_leader_close manquante, simulée avec moyenne glissante",
                    tag="CROSS",
                    voice_profile="warning",
                    priority=2,
                )
                send_alert(
                    "Colonne sector_leader_close manquante, simulée avec moyenne glissante",
                    priority=2,
                )
                send_telegram_alert(
                    "Colonne sector_leader_close manquante, simulée avec moyenne glissante"
                )
            elif col == "vix_close":
                data[col] = 20.0
                miya_speak(
                    "Colonne vix_close manquante, imputée à 20.0",
                    tag="CROSS",
                    voice_profile="warning",
                    priority=2,
                )
                send_alert("Colonne vix_close manquante, imputée à 20.0", priority=2)
                send_telegram_alert("Colonne vix_close manquante, imputée à 20.0")
            elif col == "bond_yield":
                data[col] = 3.0
                miya_speak(
                    "Colonne bond_yield manquante, imputée à 3.0",
                    tag="CROSS",
                    voice_profile="warning",
                    priority=2,
                )
                send_alert("Colonne bond_yield manquante, imputée à 3.0", priority=2)
                send_telegram_alert("Colonne bond_yield manquante, imputée à 3.0")
            elif col == "atr_14":
                data[col] = 1.0
                miya_speak(
                    "Colonne atr_14 manquante, imputée à 1.0",
                    tag="CROSS",
                    voice_profile="warning",
                    priority=2,
                )
                send_alert("Colonne atr_14 manquante, imputée à 1.0", priority=2)
                send_telegram_alert("Colonne atr_14 manquante, imputée à 1.0")
            elif col == "vix_es_correlation":
                data[col] = 0.0
                miya_speak(
                    "Colonne vix_es_correlation manquante, imputée à 0.0",
                    tag="CROSS",
                    voice_profile="warning",
                    priority=2,
                )
                send_alert(
                    "Colonne vix_es_correlation manquante, imputée à 0.0", priority=2
                )
                send_telegram_alert(
                    "Colonne vix_es_correlation manquante, imputée à 0.0"
                )

        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
        try:
            if data["timestamp"].isna().any():
                error_msg = "NaN dans les timestamps"
                miya_alerts(error_msg, tag="CROSS", voice_profile="urgent", priority=5)
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)
            if not data["timestamp"].is_monotonic_increasing:
                error_msg = "Timestamps non croissants"
                miya_alerts(error_msg, tag="CROSS", voice_profile="urgent", priority=5)
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)
        except ValueError as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans la validation des timestamps: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                "calculate_cross_market_signals", latency, success=False, error=str(e)
            )
            miya_alerts(error_msg, tag="CROSS", voice_profile="urgent", priority=5)
            send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.save_snapshot(
                "error_cross_market_signals", {"error": str(e)}, compress=False
            )
            raise

        features = [
            "spy_lead_return",
            "sector_leader_correlation",
            "vol_skew_cross_index",
            "bond_equity_risk_spread",
            "spy_momentum_diff",
            "vix_correlation",
            "sector_leader_momentum",
            "cross_index_beta",
        ]
        missing_features = [f for f in features if f not in obs_t]
        if missing_features:
            miya_alerts(
                f"Features manquantes dans obs_t: {missing_features}",
                tag="CROSS",
                voice_profile="warning",
                priority=3,
            )
            send_alert(
                f"Features manquantes dans obs_t: {missing_features}", priority=3
            )
            send_telegram_alert(f"Features manquantes dans obs_t: {missing_features}")
            logger.warning(f"Features manquantes dans obs_t: {missing_features}")
        self.validate_shap_features(features)

        for feature in features:
            data[feature] = 0.0

        data["spy_lead_return"] = self.calculate_spy_lead_return(
            data, lead_window, vol_threshold
        )
        data["sector_leader_correlation"] = self.calculate_sector_leader_correlation(
            data, corr_window
        )
        data["vol_skew_cross_index"] = self.calculate_vol_skew_cross_index(
            data, vol_skew_window
        )
        data["bond_equity_risk_spread"] = self.calculate_bond_equity_risk_spread(data)
        spy_momentum = (
            data["spy_close"]
            .pct_change()
            .rolling(window=lead_window, min_periods=1)
            .mean()
        )
        main_momentum = (
            data["close"].pct_change().rolling(window=lead_window, min_periods=1).mean()
        )
        data["spy_momentum_diff"] = spy_momentum - main_momentum
        data["vix_correlation"] = (
            data["close"]
            .rolling(window=corr_window, min_periods=1)
            .corr(data["vix_close"])
            .clip(-1, 1)
        )
        data["sector_leader_momentum"] = (
            data["sector_leader_close"]
            .pct_change()
            .rolling(window=lead_window, min_periods=1)
            .mean()
        )
        cov = (
            data["close"]
            .rolling(window=corr_window, min_periods=1)
            .cov(data["spy_close"])
        )
        var = data["spy_close"].rolling(window=corr_window, min_periods=1).var()
        data["cross_index_beta"] = (cov / (var + 1e-6)).clip(-2, 2)

        for feature in features:
            if data[feature].isna().any():
                data[feature] = data[feature].fillna(0)
            if feature in ["sector_leader_correlation", "vix_correlation"] and (
                (data[feature] < -1).any() or (data[feature] > 1).any()
            ):
                miya_alerts(
                    f"Valeurs hors limites dans {feature}",
                    tag="CROSS",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(f"Valeurs hors limites dans {feature}", priority=3)
                send_telegram_alert(f"Valeurs hors limites dans {feature}")
                data[feature] = data[feature].clip(-1, 1)

        os.makedirs(CACHE_DIR, exist_ok=True)

        def write_cache():
            data.to_csv(cache_path, index=False, encoding="utf-8")

        self.with_retries(write_cache)
        file_size = os.path.getsize(cache_path) / 1024 / 1024
        if file_size > 1.0:
            miya_alerts(
                f"Cache cross_market_signals size {file_size:.2f} MB exceeds 1 MB",
                tag="CROSS",
                voice_profile="urgent",
                priority=3,
            )
            send_alert(
                f"Cache cross_market_signals size {file_size:.2f} MB exceeds 1 MB",
                priority=3,
            )
            send_telegram_alert(
                f"Cache cross_market_signals size {file_size:.2f} MB exceeds 1 MB"
            )

        latency = time.time() - start_time
        self.log_performance(
            "calculate_cross_market_signals",
            latency,
            success=True,
            num_rows=len(data),
            num_features=len(features),
            confidence_drop_rate=confidence_drop_rate,
        )
        miya_speak(
            "Signaux cross-market calculés",
            tag="CROSS",
            voice_profile="calm",
            priority=1,
        )
        send_alert("Signaux cross-market calculés", priority=1)
        send_telegram_alert("Signaux cross-market calculés")
        logger.info("Signaux cross-market calculés")
        self.save_snapshot(
            "cross_market_signals",
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
        error_msg = f"Erreur dans calculate_cross_market_signals: {str(e)}\n{traceback.format_exc()}"
        self.log_performance(
            "calculate_cross_market_signals", latency, success=False, error=str(e)
        )
        miya_alerts(error_msg, tag="CROSS", voice_profile="urgent", priority=3)
        send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)
        logger.error(error_msg)
        self.save_snapshot("cross_market_signals", {"error": str(e)}, compress=False)
        for feature in features:
            data[feature] = 0.0
        return data
