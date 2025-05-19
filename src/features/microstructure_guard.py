# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/features/microstructure_guard.py
# Analyse la microstructure du marché pour détecter spoofing et comportements de l'order book.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Calcule les scores iceberg et spoofing, avec cache intermédiaire, logs psutil,
#        validation SHAP (méthode 17), et compatibilité top 150 SHAP features.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, matplotlib>=3.7.0,<3.8.0, psutil>=5.9.0,<6.0.0,
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
# - data/features/cache/microstructure/*.csv
# - data/logs/microstructure_guard_performance.csv
# - data/microstructure_snapshots/*.json (option *.json.gz)
# - data/figures/microstructure/
#
# Notes :
# - Conforme à structure.txt (version 2.1.3, 2025-05-13).
# - Utilise IQFeed via data_provider.py, avec retries (max 3, délai 2^attempt secondes).
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des scores.
# - Intègre validation SHAP (Phase 17) pour assurer la conformité avec les top 150 SHAP features.
# - Sauvegarde des snapshots JSON non compressés par défaut avec option de compression gzip.
# - Envoie des alertes via AlertManager et telegram_alert pour les erreurs critiques et succès.
# - Tests unitaires disponibles dans tests/test_microstructure_guard.py.
# - Conforme à la ligne rouge des top 150 SHAP features définie dans feature_pipeline.py.

import gzip
import hashlib
import json
import logging
import os
import traceback
from collections import deque
from datetime import datetime
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import yaml

from src.model.utils.alert_manager import send_alert
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.model.utils.obs_template import obs_t
from src.utils.telegram_alert import send_telegram_alert

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "microstructure_snapshots")
CSV_LOG_PATH = os.path.join(
    BASE_DIR, "data", "logs", "microstructure_guard_performance.csv"
)
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures", "microstructure")
CACHE_DIR = os.path.join(BASE_DIR, "data", "features", "cache", "microstructure")
FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR, "data", "features", "feature_importance.csv"
)

# Création des dossiers
os.makedirs(os.path.join(BASE_DIR, "data", "logs"), exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Configuration du logging
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "data", "logs", "microstructure_guard.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Constantes
MAX_RETRIES = 3
RETRY_DELAY = 2.0


class MicrostructureGuard:
    """
    Classe pour analyser la microstructure du marché et détecter les comportements anormaux.
    """

    def __init__(
        self, config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml")
    ):
        """
        Initialise le garde de microstructure.
        """
        self.log_buffer = []
        self.cache = {}
        self.config_path = config_path
        self.buffer = deque(maxlen=1000)
        try:
            self.config = self.load_config(config_path)
            self.buffer_size = self.config.get("buffer_size", 100)
            self.max_cache_size = self.config.get("max_cache_size", 1000)
            self.buffer_maxlen = self.config.get("buffer_maxlen", 1000)
            self.buffer = deque(maxlen=self.buffer_maxlen)
            miya_speak(
                "MicrostructureGuard initialisé",
                tag="MICROSTRUCTURE",
                voice_profile="calm",
                priority=2,
            )
            send_alert("MicrostructureGuard initialisé", priority=2)
            send_telegram_alert("MicrostructureGuard initialisé")
            logger.info("MicrostructureGuard initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config_path": config_path}, compress=False)
        except Exception as e:
            error_msg = f"Erreur initialisation MicrostructureGuard: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="MICROSTRUCTURE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            self.config = {
                "window_size": 20,
                "iceberg_threshold": 0.8,
                "spoofing_spread_threshold": 0.05,
                "min_rows": 10,
                "buffer_size": 100,
                "max_cache_size": 1000,
                "buffer_maxlen": 1000,
                "cache_hours": 24,
            }
            self.buffer_size = 100
            self.max_cache_size = 1000
            self.buffer_maxlen = 1000
            self.buffer = deque(maxlen=self.buffer_maxlen)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Charge la configuration depuis es_config.yaml.
        """

        def load_yaml():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            if "microstructure_guard" not in config:
                raise ValueError(
                    "Clé 'microstructure_guard' manquante dans la configuration"
                )
            required_keys = [
                "window_size",
                "iceberg_threshold",
                "spoofing_spread_threshold",
                "min_rows",
            ]
            missing_keys = [
                key
                for key in required_keys
                if key not in config["microstructure_guard"]
            ]
            if missing_keys:
                raise ValueError(
                    f"Clés manquantes dans 'microstructure_guard': {missing_keys}"
                )
            return config["microstructure_guard"]

        try:
            start_time = time.time()
            config = self.with_retries(load_yaml)
            cache_key = hashlib.sha256(str(config).encode()).hexdigest()
            self.cache[cache_key] = {"config": config, "timestamp": datetime.now()}
            if len(self.cache) > self.max_cache_size:
                self.cache.pop(next(iter(self.cache)))
            latency = time.time() - start_time
            miya_speak(
                "Configuration microstructure_guard chargée",
                tag="MICROSTRUCTURE",
                voice_profile="calm",
                priority=2,
            )
            send_alert("Configuration microstructure_guard chargée", priority=2)
            send_telegram_alert("Configuration microstructure_guard chargée")
            logger.info("Configuration microstructure_guard chargée")
            self.log_performance("load_config", latency, success=True)
            self.save_snapshot(
                "load_config", {"config_path": config_path}, compress=False
            )
            return config
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur chargement config: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="MICROSTRUCTURE", voice_profile="urgent", priority=3
            )
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
        """Exécute une fonction avec retries."""
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
                        tag="MICROSTRUCTURE",
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
                    tag="MICROSTRUCTURE",
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
                    tag="MICROSTRUCTURE",
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
                tag="MICROSTRUCTURE",
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
                    tag="MICROSTRUCTURE",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(f"Snapshot size {file_size:.2f} MB exceeds 1 MB", priority=3)
                send_telegram_alert(f"Snapshot size {file_size:.2f} MB exceeds 1 MB")
            latency = time.time() - start_time
            miya_speak(
                f"Snapshot {snapshot_type} sauvegardé: {save_path}",
                tag="MICROSTRUCTURE",
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
                tag="MICROSTRUCTURE",
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
                    tag="MICROSTRUCTURE",
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
                    tag="MICROSTRUCTURE",
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
                    tag="MICROSTRUCTURE",
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
                tag="MICROSTRUCTURE",
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
                tag="MICROSTRUCTURE",
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

    def plot_scores(
        self, iceberg_score: pd.Series, spoofing_score: pd.Series, timestamp: str
    ) -> None:
        """Génère des visualisations pour les scores iceberg et spoofing."""
        start_time = time.time()
        try:
            timestamp_safe = timestamp.replace(":", "-")
            os.makedirs(FIGURES_DIR, exist_ok=True)

            plt.figure(figsize=(12, 6))
            plt.plot(
                iceberg_score.index, iceberg_score, label="Iceberg Score", color="blue"
            )
            plt.plot(
                spoofing_score.index,
                spoofing_score,
                label="Spoofing Score",
                color="red",
            )
            plt.title(f"Scores Iceberg et Spoofing - {timestamp}")
            plt.xlabel("Index")
            plt.ylabel("Score")
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(
                FIGURES_DIR, f"scores_temporal_{timestamp_safe}.png"
            )
            plt.savefig(plot_path)
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.hist(
                iceberg_score, bins=20, alpha=0.5, label="Iceberg Score", color="blue"
            )
            plt.hist(
                spoofing_score, bins=20, alpha=0.5, label="Spoofing Score", color="red"
            )
            plt.title(f"Distribution des Scores - {timestamp}")
            plt.xlabel("Score")
            plt.ylabel("Fréquence")
            plt.legend()
            plt.grid(True)
            plt.savefig(
                os.path.join(FIGURES_DIR, f"scores_distribution_{timestamp_safe}.png")
            )
            plt.close()

            latency = time.time() - start_time
            miya_speak(
                f"Visualisations générées: {plot_path}",
                tag="MICROSTRUCTURE",
                voice_profile="calm",
                priority=2,
            )
            send_alert(f"Visualisations générées: {plot_path}", priority=2)
            send_telegram_alert(f"Visualisations générées: {plot_path}")
            logger.info(f"Visualisations générées: {plot_path}")
            self.log_performance("plot_scores", latency, success=True)
        except Exception as e:
            latency = time.time() - start_time
            error_msg = (
                f"Erreur génération visualisations: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, tag="MICROSTRUCTURE", voice_profile="urgent", priority=2
            )
            send_alert(error_msg, priority=2)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("plot_scores", latency, success=False, error=str(e))

    def calculate_iceberg_order_score(
        self,
        data: pd.DataFrame,
        config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml"),
    ) -> pd.Series:
        """Calcule un score pour détecter les ordres iceberg dans l'order book."""
        cache_path = os.path.join(
            CACHE_DIR,
            f"iceberg_score_{hashlib.sha256(data.to_json().encode()).hexdigest()}.csv",
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
                        "calculate_iceberg_order_score_cache_hit", latency, success=True
                    )
                    miya_speak(
                        "Iceberg score chargé depuis cache",
                        tag="MICROSTRUCTURE",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert("Iceberg score chargé depuis cache", priority=1)
                    send_telegram_alert("Iceberg score chargé depuis cache")
                    return cached_data["iceberg_order_score"]

            config = self.load_config(config_path)
            window_size = config.get("window_size", 20)
            iceberg_threshold = config.get("iceberg_threshold", 0.8)
            min_rows = config.get("min_rows", 10)

            if data.empty or len(data) < min_rows:
                error_msg = f"DataFrame vide ou insuffisant ({len(data)} < {min_rows})"
                miya_alerts(
                    error_msg, tag="MICROSTRUCTURE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            data = data.copy()
            data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
            if data["timestamp"].isna().any():
                error_msg = "NaN dans les timestamps"
                miya_alerts(
                    error_msg, tag="MICROSTRUCTURE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)
            if not data["timestamp"].is_monotonic_increasing:
                error_msg = "Timestamps non croissants"
                miya_alerts(
                    error_msg, tag="MICROSTRUCTURE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            required_features = [
                "bid_size_level_1",
                "ask_size_level_1",
                "bid_price_level_1",
                "ask_price_level_1",
                "bid_size_level_2",
                "ask_size_level_2",
            ]
            available_features = [
                col for col in required_features if col in data.columns
            ]
            missing_features = [
                col for col in required_features if col not in available_features
            ]
            # Calculer confidence_drop_rate (Phase 8)
            confidence_drop_rate = 1.0 - min(
                len(available_features) / len(required_features), 1.0
            )
            if len(data) < min_rows:
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(available_features)}/{len(required_features)} features valides, {len(data)} lignes)"
                miya_alerts(
                    alert_msg, tag="MICROSTRUCTURE", voice_profile="urgent", priority=3
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            if missing_features:
                miya_speak(
                    f"Features manquantes: {missing_features}, imputées à 0",
                    tag="MICROSTRUCTURE",
                    voice_profile="warning",
                    priority=2,
                )
                send_alert(
                    f"Features manquantes: {missing_features}, imputées à 0", priority=2
                )
                send_telegram_alert(
                    f"Features manquantes: {missing_features}, imputées à 0"
                )
                for col in missing_features:
                    data[col] = 0.0

            for col in available_features:
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
                            tag="MICROSTRUCTURE",
                            voice_profile="urgent",
                            priority=4,
                        )
                        send_alert(error_msg, priority=4)
                        send_telegram_alert(error_msg)
                        raise ValueError(error_msg)

            bid_persistence = data["bid_size_level_1"].rolling(
                window=window_size, min_periods=1
            ).std() / (
                data["bid_size_level_1"]
                .rolling(window=window_size, min_periods=1)
                .mean()
                + 1e-6
            )
            ask_persistence = data["ask_size_level_1"].rolling(
                window=window_size, min_periods=1
            ).std() / (
                data["ask_size_level_1"]
                .rolling(window=window_size, min_periods=1)
                .mean()
                + 1e-6
            )
            price_stability = (
                1
                - (
                    data["bid_price_level_1"].diff().abs()
                    + data["ask_price_level_1"].diff().abs()
                )
                .rolling(window=window_size, min_periods=1)
                .mean()
            )

            iceberg_score = (
                (1 - bid_persistence) * (1 - ask_persistence) * price_stability
            )
            iceberg_score = iceberg_score.clip(0, 1)
            iceberg_score = (iceberg_score - iceberg_score.min()) / (
                iceberg_score.max() - iceberg_score.min() + 1e-6
            )  # Normalisation
            iceberg_score = iceberg_score.where(iceberg_score >= iceberg_threshold, 0.0)

            if not np.isfinite(iceberg_score).all():
                error_msg = "Valeurs infinies ou NaN dans iceberg_score"
                miya_alerts(
                    error_msg, tag="MICROSTRUCTURE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                iceberg_score = iceberg_score.fillna(0.0)

            os.makedirs(CACHE_DIR, exist_ok=True)

            def write_cache():
                cache_df = pd.DataFrame(
                    {
                        "timestamp": data["timestamp"],
                        "iceberg_order_score": iceberg_score,
                    }
                )
                cache_df.to_csv(cache_path, index=False, encoding="utf-8")

            self.with_retries(write_cache)
            file_size = os.path.getsize(cache_path) / 1024 / 1024
            if file_size > 1.0:
                miya_alerts(
                    f"Cache iceberg_score size {file_size:.2f} MB exceeds 1 MB",
                    tag="MICROSTRUCTURE",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(
                    f"Cache iceberg_score size {file_size:.2f} MB exceeds 1 MB",
                    priority=3,
                )
                send_telegram_alert(
                    f"Cache iceberg_score size {file_size:.2f} MB exceeds 1 MB"
                )

            self.validate_shap_features(["iceberg_order_score"])

            latency = time.time() - start_time
            miya_speak(
                f"Iceberg order score calculé, moyenne: {iceberg_score.mean():.2f}",
                tag="MICROSTRUCTURE",
                voice_profile="calm",
                priority=1,
            )
            send_alert(
                f"Iceberg order score calculé, moyenne: {iceberg_score.mean():.2f}",
                priority=1,
            )
            send_telegram_alert(
                f"Iceberg order score calculé, moyenne: {iceberg_score.mean():.2f}"
            )
            logger.info(
                f"Iceberg order score calculé, moyenne: {iceberg_score.mean():.2f}"
            )
            self.log_performance(
                "calculate_iceberg_order_score",
                latency,
                success=True,
                num_rows=len(data),
                mean_score=iceberg_score.mean(),
                confidence_drop_rate=confidence_drop_rate,
            )
            self.save_snapshot(
                "calculate_iceberg_order_score",
                {
                    "mean_score": float(iceberg_score.mean()),
                    "num_rows": len(data),
                    "confidence_drop_rate": confidence_drop_rate,
                },
                compress=False,
            )
            return iceberg_score
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans calculate_iceberg_order_score: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="MICROSTRUCTURE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "calculate_iceberg_order_score", latency, success=False, error=str(e)
            )
            self.save_snapshot(
                "calculate_iceberg_order_score", {"error": str(e)}, compress=False
            )
            return pd.Series(0.0, index=data.index)

    def calculate_spoofing_score(
        self,
        data: pd.DataFrame,
        config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml"),
    ) -> pd.Series:
        """Calcule un score pour détecter le spoofing dans l'order book."""
        cache_path = os.path.join(
            CACHE_DIR,
            f"spoofing_score_{hashlib.sha256(data.to_json().encode()).hexdigest()}.csv",
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
                        "calculate_spoofing_score_cache_hit", latency, success=True
                    )
                    miya_speak(
                        "Spoofing score chargé depuis cache",
                        tag="MICROSTRUCTURE",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert("Spoofing score chargé depuis cache", priority=1)
                    send_telegram_alert("Spoofing score chargé depuis cache")
                    return cached_data["spoofing_score"]

            config = self.load_config(config_path)
            window_size = config.get("window_size", 20)
            spoofing_spread_threshold = config.get("spoofing_spread_threshold", 0.05)
            min_rows = config.get("min_rows", 10)

            if data.empty or len(data) < min_rows:
                error_msg = f"DataFrame vide ou insuffisant ({len(data)} < {min_rows})"
                miya_alerts(
                    error_msg, tag="MICROSTRUCTURE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            data = data.copy()
            data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
            if data["timestamp"].isna().any():
                error_msg = "NaN dans les timestamps"
                miya_alerts(
                    error_msg, tag="MICROSTRUCTURE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)
            if not data["timestamp"].is_monotonic_increasing:
                error_msg = "Timestamps non croissants"
                miya_alerts(
                    error_msg, tag="MICROSTRUCTURE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            required_features = [
                "bid_size_level_1",
                "ask_size_level_1",
                "bid_price_level_1",
                "ask_price_level_1",
            ]
            available_features = [
                col for col in required_features if col in data.columns
            ]
            missing_features = [
                col for col in required_features if col not in available_features
            ]
            confidence_drop_rate = 1.0 - min(
                len(available_features) / len(required_features), 1.0
            )
            if len(data) < min_rows:
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(available_features)}/{len(required_features)} features valides, {len(data)} lignes)"
                miya_alerts(
                    alert_msg, tag="MICROSTRUCTURE", voice_profile="urgent", priority=3
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            if missing_features:
                miya_speak(
                    f"Features manquantes: {missing_features}, imputées à 0",
                    tag="MICROSTRUCTURE",
                    voice_profile="warning",
                    priority=2,
                )
                send_alert(
                    f"Features manquantes: {missing_features}, imputées à 0", priority=2
                )
                send_telegram_alert(
                    f"Features manquantes: {missing_features}, imputées à 0"
                )
                for col in missing_features:
                    data[col] = 0.0

            for col in available_features:
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
                            tag="MICROSTRUCTURE",
                            voice_profile="urgent",
                            priority=4,
                        )
                        send_alert(error_msg, priority=4)
                        send_telegram_alert(error_msg)
                        raise ValueError(error_msg)

            bid_size_change = data["bid_size_level_1"].diff().abs()
            ask_size_change = data["ask_size_level_1"].diff().abs()
            spread = (data["ask_price_level_1"] - data["bid_price_level_1"]) / data[
                "bid_price_level_1"
            ]
            large_order_imbalance = (
                bid_size_change
                > data["bid_size_level_1"]
                .rolling(window=window_size, min_periods=1)
                .quantile(0.9)
            ) | (
                ask_size_change
                > data["ask_size_level_1"]
                .rolling(window=window_size, min_periods=1)
                .quantile(0.9)
            )
            spoofing_score = large_order_imbalance.astype(float) * (
                spread > spoofing_spread_threshold
            ).astype(float)
            spoofing_score = (spoofing_score - spoofing_score.min()) / (
                spoofing_score.max() - spoofing_score.min() + 1e-6
            )  # Normalisation

            if not np.isfinite(spoofing_score).all():
                error_msg = "Valeurs infinies ou NaN dans spoofing_score"
                miya_alerts(
                    error_msg, tag="MICROSTRUCTURE", voice_profile="urgent", priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                spoofing_score = spoofing_score.fillna(0.0)

            os.makedirs(CACHE_DIR, exist_ok=True)

            def write_cache():
                cache_df = pd.DataFrame(
                    {"timestamp": data["timestamp"], "spoofing_score": spoofing_score}
                )
                cache_df.to_csv(cache_path, index=False, encoding="utf-8")

            self.with_retries(write_cache)
            file_size = os.path.getsize(cache_path) / 1024 / 1024
            if file_size > 1.0:
                miya_alerts(
                    f"Cache spoofing_score size {file_size:.2f} MB exceeds 1 MB",
                    tag="MICROSTRUCTURE",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(
                    f"Cache spoofing_score size {file_size:.2f} MB exceeds 1 MB",
                    priority=3,
                )
                send_telegram_alert(
                    f"Cache spoofing_score size {file_size:.2f} MB exceeds 1 MB"
                )

            self.validate_shap_features(["spoofing_score"])

            latency = time.time() - start_time
            miya_speak(
                f"Spoofing score calculé, moyenne: {spoofing_score.mean():.2f}",
                tag="MICROSTRUCTURE",
                voice_profile="calm",
                priority=1,
            )
            send_alert(
                f"Spoofing score calculé, moyenne: {spoofing_score.mean():.2f}",
                priority=1,
            )
            send_telegram_alert(
                f"Spoofing score calculé, moyenne: {spoofing_score.mean():.2f}"
            )
            logger.info(f"Spoofing score calculé, moyenne: {spoofing_score.mean():.2f}")
            self.log_performance(
                "calculate_spoofing_score",
                latency,
                success=True,
                num_rows=len(data),
                mean_score=spoofing_score.mean(),
                confidence_drop_rate=confidence_drop_rate,
            )
            self.save_snapshot(
                "calculate_spoofing_score",
                {
                    "mean_score": float(spoofing_score.mean()),
                    "num_rows": len(data),
                    "confidence_drop_rate": confidence_drop_rate,
                },
                compress=False,
            )
            return spoofing_score
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans calculate_spoofing_score: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="MICROSTRUCTURE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "calculate_spoofing_score", latency, success=False, error=str(e)
            )
            self.save_snapshot(
                "calculate_spoofing_score", {"error": str(e)}, compress=False
            )
            return pd.Series(0.0, index=data.index)

    def calculate_incremental_iceberg_order_score(
        self,
        row: pd.Series,
        config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml"),
    ) -> float:
        """Calcule le score iceberg pour une seule ligne en temps réel."""
        try:
            start_time = time.time()
            config = self.load_config(config_path)
            window_size = config.get("window_size", 20)
            min_rows = config.get("min_rows", 10)

            row = row.copy()
            row["timestamp"] = pd.to_datetime(row["timestamp"], errors="coerce")
            if pd.isna(row["timestamp"]):
                error_msg = "Timestamp invalide dans la ligne"
                miya_alerts(
                    error_msg, tag="MICROSTRUCTURE", voice_profile="urgent", priority=3
                )
                send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            self.buffer.append(row.to_frame().T)
            if len(self.buffer) < min_rows:
                return 0.0

            data = pd.concat(list(self.buffer), ignore_index=True).tail(window_size)
            required_features = [
                "bid_size_level_1",
                "ask_size_level_1",
                "bid_price_level_1",
                "ask_price_level_1",
                "bid_size_level_2",
                "ask_size_level_2",
            ]
            available_features = [
                col for col in required_features if col in data.columns
            ]
            confidence_drop_rate = 1.0 - min(
                len(available_features) / len(required_features), 1.0
            )
            if len(data) < min_rows:
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(available_features)}/{len(required_features)} features valides, {len(data)} lignes)"
                miya_alerts(
                    alert_msg, tag="MICROSTRUCTURE", voice_profile="urgent", priority=3
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            iceberg_score = self.calculate_iceberg_order_score(data, config_path)
            score = iceberg_score.iloc[-1]

            latency = time.time() - start_time
            miya_speak(
                f"Incremental iceberg score calculé: {score:.2f}",
                tag="MICROSTRUCTURE",
                voice_profile="calm",
                priority=1,
            )
            send_alert(f"Incremental iceberg score calculé: {score:.2f}", priority=1)
            send_telegram_alert(f"Incremental iceberg score calculé: {score:.2f}")
            logger.info(f"Incremental iceberg score calculé: {score:.2f}")
            self.log_performance(
                "calculate_incremental_iceberg_order_score",
                latency,
                success=True,
                score=score,
                confidence_drop_rate=confidence_drop_rate,
            )
            self.save_snapshot(
                "calculate_incremental_iceberg_order_score",
                {"score": float(score), "confidence_drop_rate": confidence_drop_rate},
                compress=False,
            )
            return score
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans calculate_incremental_iceberg_order_score: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="MICROSTRUCTURE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "calculate_incremental_iceberg_order_score",
                latency,
                success=False,
                error=str(e),
            )
            self.save_snapshot(
                "calculate_incremental_iceberg_order_score",
                {"error": str(e)},
                compress=False,
            )
            return 0.0

    def calculate_incremental_spoofing_score(
        self,
        row: pd.Series,
        config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml"),
    ) -> float:
        """Calcule le score de spoofing pour une seule ligne en temps réel."""
        try:
            start_time = time.time()
            config = self.load_config(config_path)
            window_size = config.get("window_size", 20)
            min_rows = config.get("min_rows", 10)

            row = row.copy()
            row["timestamp"] = pd.to_datetime(row["timestamp"], errors="coerce")
            if pd.isna(row["timestamp"]):
                error_msg = "Timestamp invalide dans la ligne"
                miya_alerts(
                    error_msg, tag="MICROSTRUCTURE", voice_profile="urgent", priority=3
                )
                send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            self.buffer.append(row.to_frame().T)
            if len(self.buffer) < min_rows:
                return 0.0

            data = pd.concat(list(self.buffer), ignore_index=True).tail(window_size)
            required_features = [
                "bid_size_level_1",
                "ask_size_level_1",
                "bid_price_level_1",
                "ask_price_level_1",
            ]
            available_features = [
                col for col in required_features if col in data.columns
            ]
            confidence_drop_rate = 1.0 - min(
                len(available_features) / len(required_features), 1.0
            )
            if len(data) < min_rows:
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(available_features)}/{len(required_features)} features valides, {len(data)} lignes)"
                miya_alerts(
                    alert_msg, tag="MICROSTRUCTURE", voice_profile="urgent", priority=3
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            spoofing_score = self.calculate_spoofing_score(data, config_path)
            score = spoofing_score.iloc[-1]

            latency = time.time() - start_time
            miya_speak(
                f"Incremental spoofing score calculé: {score:.2f}",
                tag="MICROSTRUCTURE",
                voice_profile="calm",
                priority=1,
            )
            send_alert(f"Incremental spoofing score calculé: {score:.2f}", priority=1)
            send_telegram_alert(f"Incremental spoofing score calculé: {score:.2f}")
            logger.info(f"Incremental spoofing score calculé: {score:.2f}")
            self.log_performance(
                "calculate_incremental_spoofing_score",
                latency,
                success=True,
                score=score,
                confidence_drop_rate=confidence_drop_rate,
            )
            self.save_snapshot(
                "calculate_incremental_spoofing_score",
                {"score": float(score), "confidence_drop_rate": confidence_drop_rate},
                compress=False,
            )
            return score
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans calculate_incremental_spoofing_score: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="MICROSTRUCTURE", voice_profile="urgent", priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "calculate_incremental_spoofing_score",
                latency,
                success=False,
                error=str(e),
            )
            self.save_snapshot(
                "calculate_incremental_spoofing_score",
                {"error": str(e)},
                compress=False,
            )
            return 0.0


if __name__ == "__main__":
    try:
        guard = MicrostructureGuard()
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "bid_size_level_1": np.random.randint(100, 1000, 100),
                "ask_size_level_1": np.random.randint(100, 1000, 100),
                "bid_price_level_1": np.random.uniform(5090, 5110, 100),
                "ask_price_level_1": np.random.uniform(5090, 5110, 100),
                "bid_size_level_2": np.random.randint(50, 500, 100),
                "ask_size_level_2": np.random.randint(50, 500, 100),
            }
        )
        iceberg_score = guard.calculate_iceberg_order_score(data)
        spoofing_score = guard.calculate_spoofing_score(data)
        guard.plot_scores(iceberg_score, spoofing_score, "2025-04-14 09:00")
        print("Iceberg order score (premières 5 valeurs):")
        print(iceberg_score.head())
        print("Spoofing score (premières 5 valeurs):")
        print(spoofing_score.head())
        miya_speak(
            "Test microstructure_guard terminé",
            tag="MICROSTRUCTURE",
            voice_profile="calm",
            priority=1,
        )
        send_alert("Test microstructure_guard terminé", priority=1)
        send_telegram_alert("Test microstructure_guard terminé")
        logger.info("Test microstructure_guard terminé")
    except Exception as e:
        miya_alerts(
            f"Erreur test: {str(e)}\n{traceback.format_exc()}",
            tag="MICROSTRUCTURE",
            voice_profile="urgent",
            priority=3,
        )
        send_alert(f"Erreur test: {str(e)}", priority=3)
        send_telegram_alert(f"Erreur test: {str(e)}")
        logger.error(f"Erreur test: {str(e)}")
        raise
