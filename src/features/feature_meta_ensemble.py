# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/features/feature_meta_ensemble.py
# Réduit dynamiquement le vecteur d’observation (350 features) en fonction des conditions de marché.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Réduit dynamiquement les features en utilisant SHAP (méthode 17) et pondération par régime (méthode 3),
#        avec cache intermédiaire, logs psutil, validation SHAP (méthode 17), et compatibilité top 150 SHAP features.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, xgboost>=2.0.0,<3.0.0, shap>=0.41.0,<0.42.0,
#   psutil>=5.9.0,<6.0.0, scikit-learn>=1.0.2,<2.0.0, pyyaml>=6.0.0,<7.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
# - src/model/utils/obs_template.py
# - src/features/feature_pipeline.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/es_config.yaml
# - data/features/feature_importance.csv
# - data/iqfeed/merged_data.csv
#
# Outputs :
# - data/features/cache/ensemble/*.csv
# - data/logs/feature_meta_ensemble_performance.csv
# - data/ensemble_snapshots/*.json (option *.json.gz)
# - data/figures/ensemble/
#
# Notes :
# - Conforme à structure.txt (version 2.1.3, 2025-05-13).
# - Utilise IQFeed via feature_pipeline.py, avec retries (max 3, délai 2^attempt secondes).
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des réductions de features.
# - Intègre validation SHAP (Phase 17) pour assurer la conformité avec les top 150 SHAP features.
# - Sauvegarde des snapshots JSON non compressés par défaut avec option de compression gzip.
# - Envoie des alertes via AlertManager et telegram_alert pour les erreurs critiques et succès.
# - Tests unitaires disponibles dans tests/test_feature_meta_ensemble.py.
# - Conforme à la ligne rouge des top 150 SHAP features définie dans feature_pipeline.py.

import gzip
import hashlib
import json
import logging
import os
import traceback
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import shap
import xgboost as xgb
import yaml
from sklearn.model_selection import cross_val_score

from src.model.utils.alert_manager import send_alert
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.model.utils.obs_template import obs_t
from src.utils.telegram_alert import send_telegram_alert

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "ensemble_snapshots")
CSV_LOG_PATH = os.path.join(
    BASE_DIR, "data", "logs", "feature_meta_ensemble_performance.csv"
)
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures", "ensemble")
CACHE_DIR = os.path.join(BASE_DIR, "data", "features", "cache", "ensemble")
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
    filename=os.path.join(BASE_DIR, "data", "logs", "feature_meta_ensemble.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Constantes
MAX_RETRIES = 3
RETRY_DELAY = 2.0


class FeatureMetaEnsemble:
    """
    Classe pour réduire dynamiquement le vecteur d’observation en fonction des conditions de marché.
    """

    def __init__(
        self, config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml")
    ):
        """
        Initialise le réducteur de features.
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
                "FeatureMetaEnsemble initialisé",
                tag="OPTIMIZE",
                voice_profile="calm",
                priority=2,
            )
            send_alert("FeatureMetaEnsemble initialisé", priority=2)
            send_telegram_alert("FeatureMetaEnsemble initialisé")
            logger.info("FeatureMetaEnsemble initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config_path": config_path}, compress=False)
        except Exception as e:
            error_msg = f"Erreur initialisation FeatureMetaEnsemble: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="OPTIMIZE", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            self.config = {
                "target_dims": 81,
                "window_size": 20,
                "boost_factors": {"Trend": 1.5, "Range": 1.5, "Defensive": 1.5},
                "momentum_features": [
                    "spy_lead_return",
                    "sector_leader_momentum",
                    "spy_momentum_diff",
                    "order_flow_acceleration",
                ],
                "volatility_features": [
                    "atr_14",
                    "volatility_trend",
                    "microstructure_volatility",
                    "vix_es_correlation",
                    "iv_atm",
                    "option_skew",
                ],
                "risk_features": [
                    "bond_equity_risk_spread",
                    "vanna_cliff_slope",
                    "vanna_exposure",
                    "delta_exposure",
                    "gex_slope",
                ],
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
            if "feature_meta_ensemble" not in config:
                raise ValueError(
                    "Clé 'feature_meta_ensemble' manquante dans la configuration"
                )
            required_keys = [
                "target_dims",
                "window_size",
                "boost_factors",
                "momentum_features",
                "volatility_features",
                "risk_features",
            ]
            missing_keys = [
                key
                for key in required_keys
                if key not in config["feature_meta_ensemble"]
            ]
            if missing_keys:
                raise ValueError(
                    f"Clés manquantes dans 'feature_meta_ensemble': {missing_keys}"
                )
            return config["feature_meta_ensemble"]

        try:
            start_time = time.time()
            config = self.with_retries(load_yaml)
            cache_key = hashlib.sha256(str(config).encode()).hexdigest()
            self.cache[cache_key] = {"config": config, "timestamp": datetime.now()}
            if len(self.cache) > self.max_cache_size:
                self.cache.pop(next(iter(self.cache)))
            latency = time.time() - start_time
            miya_speak(
                "Configuration feature_meta_ensemble chargée",
                tag="OPTIMIZE",
                voice_profile="calm",
                priority=2,
            )
            send_alert("Configuration feature_meta_ensemble chargée", priority=2)
            send_telegram_alert("Configuration feature_meta_ensemble chargée")
            logger.info("Configuration feature_meta_ensemble chargée")
            self.log_performance("load_config", latency, success=True)
            self.save_snapshot(
                "load_config", {"config_path": config_path}, compress=False
            )
            return config
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur chargement config: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="OPTIMIZE", voice_profile="urgent", priority=3)
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
                        tag="OPTIMIZE",
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
                    tag="OPTIMIZE",
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
                    tag="OPTIMIZE",
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
                tag="OPTIMIZE",
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
                    tag="OPTIMIZE",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(f"Snapshot size {file_size:.2f} MB exceeds 1 MB", priority=3)
                send_telegram_alert(f"Snapshot size {file_size:.2f} MB exceeds 1 MB")
            latency = time.time() - start_time
            miya_speak(
                f"Snapshot {snapshot_type} sauvegardé: {save_path}",
                tag="OPTIMIZE",
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
                tag="OPTIMIZE",
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
                    tag="OPTIMIZE",
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
                    tag="OPTIMIZE",
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
                    tag="OPTIMIZE",
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
                tag="OPTIMIZE",
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
                tag="OPTIMIZE",
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

    def plot_shap_importance(
        self, shap_importance: pd.Series, market_condition: str, timestamp: str
    ) -> None:
        """Génère des visualisations pour l’importance SHAP des features."""
        start_time = time.time()
        try:
            timestamp_safe = timestamp.replace(":", "-")
            os.makedirs(FIGURES_DIR, exist_ok=True)

            plt.figure(figsize=(10, 6))
            shap_importance.head(10).plot(kind="bar")
            plt.title(
                f"Top 10 Features par Importance SHAP ({market_condition}) - {timestamp}"
            )
            plt.xlabel("Feature")
            plt.ylabel("Importance SHAP (normalisée)")
            plt.xticks(rotation=45, ha="right")
            plt.grid(True)
            plt.tight_layout()
            plot_path = os.path.join(
                FIGURES_DIR, f"shap_bar_{market_condition}_{timestamp_safe}.png"
            )
            plt.savefig(plot_path)
            plt.close()

            latency = time.time() - start_time
            miya_speak(
                f"Visualisations SHAP générées: {plot_path}",
                tag="OPTIMIZE",
                voice_profile="calm",
                priority=2,
            )
            send_alert(f"Visualisations SHAP générées: {plot_path}", priority=2)
            send_telegram_alert(f"Visualisations SHAP générées: {plot_path}")
            logger.info(f"Visualisations SHAP générées: {plot_path}")
            self.log_performance("plot_shap_importance", latency, success=True)
        except Exception as e:
            latency = time.time() - start_time
            error_msg = (
                f"Erreur génération visualisations: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="OPTIMIZE", voice_profile="urgent", priority=2)
            send_alert(error_msg, priority=2)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "plot_shap_importance", latency, success=False, error=str(e)
            )

    def configure_feature_set(self, expected_count: int = 350) -> list:
        """Configure le jeu de features en fonction du mode (350 ou 81 features)."""
        start_time = time.time()
        try:
            confidence_drop_rate = 0.0
            if expected_count == 350:
                feature_set = list(
                    set(
                        obs_t
                        + [
                            "atr_14",
                            "volatility_trend",
                            "spy_lead_return",
                            "sector_leader_momentum",
                            "order_flow_acceleration",
                            "microstructure_volatility",
                            "vix_es_correlation",
                            "bond_equity_risk_spread",
                            "vanna_cliff_slope",
                            "vanna_exposure",
                            "delta_exposure",
                            "iv_atm",
                            "option_skew",
                            "gex_slope",
                            "trade_success_prob",
                        ]
                        + [f"feature_{i}" for i in range(350 - len(obs_t) - 15)]
                    )
                )
                miya_speak(
                    f"Mode 350 features activé ({len(feature_set)} features)",
                    tag="OPTIMIZE",
                    voice_profile="calm",
                    priority=1,
                )
                send_alert(
                    f"Mode 350 features activé ({len(feature_set)} features)",
                    priority=1,
                )
                send_telegram_alert(
                    f"Mode 350 features activé ({len(feature_set)} features)"
                )
            else:
                feature_set = list(obs_t)
                miya_speak(
                    f"Mode 81 features activé ({len(feature_set)} features)",
                    tag="OPTIMIZE",
                    voice_profile="calm",
                    priority=1,
                )
                send_alert(
                    f"Mode 81 features activé ({len(feature_set)} features)", priority=1
                )
                send_telegram_alert(
                    f"Mode 81 features activé ({len(feature_set)} features)"
                )
            latency = time.time() - start_time
            logger.info(f"Feature set configuré: {len(feature_set)} features")
            self.log_performance(
                "configure_feature_set",
                latency,
                success=True,
                num_features=len(feature_set),
                confidence_drop_rate=confidence_drop_rate,
            )
            self.save_snapshot(
                "configure_feature_set",
                {
                    "num_features": len(feature_set),
                    "confidence_drop_rate": confidence_drop_rate,
                },
                compress=False,
            )
            return feature_set
        except Exception as e:
            latency = time.time() - start_time
            error_msg = (
                f"Erreur configuration feature set: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="OPTIMIZE", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "configure_feature_set", latency, success=False, error=str(e)
            )
            self.save_snapshot(
                "configure_feature_set", {"error": str(e)}, compress=False
            )
            return list(obs_t)

    def calculate_shap_importance(
        self, data: pd.DataFrame, target_col: str = "close"
    ) -> pd.Series:
        """Calcule l'importance des features à l'aide de SHAP, en priorité depuis feature_importance.csv."""
        start_time = time.time()
        try:
            required_cols = ["timestamp", target_col, "neural_regime"]
            missing_cols = [col for col in required_cols if col not in data.columns]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )
            if len(data) < self.config.get("window_size", 20):
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(required_cols) - len(missing_cols)}/{len(required_cols)} colonnes valides, {len(data)} lignes)"
                miya_alerts(
                    alert_msg, tag="OPTIMIZE", voice_profile="urgent", priority=3
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            if os.path.exists(FEATURE_IMPORTANCE_PATH):
                shap_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
                if "feature" in shap_df.columns and "importance" in shap_df.columns:
                    shap_importance = pd.Series(
                        shap_df["importance"].values, index=shap_df["feature"]
                    ).sort_values(ascending=False)
                    shap_importance = (shap_importance - shap_importance.min()) / (
                        shap_importance.max() - shap_importance.min() + 1e-6
                    )
                    latency = time.time() - start_time
                    miya_speak(
                        "Importance SHAP chargée depuis feature_importance.csv",
                        tag="OPTIMIZE",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert(
                        "Importance SHAP chargée depuis feature_importance.csv",
                        priority=1,
                    )
                    send_telegram_alert(
                        "Importance SHAP chargée depuis feature_importance.csv"
                    )
                    logger.info("Importance SHAP chargée depuis feature_importance.csv")
                    self.log_performance(
                        "calculate_shap_importance_cache_hit",
                        latency,
                        success=True,
                        num_features=len(shap_importance),
                        confidence_drop_rate=confidence_drop_rate,
                    )
                    return shap_importance

            cache_key = hashlib.sha256(
                f"{data.to_json()}_{target_col}".encode()
            ).hexdigest()
            if cache_key in self.cache:
                miya_speak(
                    "Importance SHAP récupérée du cache",
                    tag="OPTIMIZE",
                    voice_profile="calm",
                    priority=1,
                )
                send_alert("Importance SHAP récupérée du cache", priority=1)
                send_telegram_alert("Importance SHAP récupérée du cache")
                self.log_performance(
                    "calculate_shap_importance_cache_hit",
                    0,
                    success=True,
                    confidence_drop_rate=confidence_drop_rate,
                )
                return self.cache[cache_key]

            if target_col not in data.columns:
                error_msg = f"Colonne '{target_col}' manquante"
                miya_alerts(
                    error_msg, tag="OPTIMIZE", voice_profile="urgent", priority=3
                )
                send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            data = data.copy()
            X = data.drop(
                columns=["timestamp", target_col, "neural_regime"], errors="ignore"
            )
            y = data[target_col]

            for col in X.columns:
                if X[col].isna().any():
                    X[col] = (
                        X[col]
                        .interpolate(method="linear", limit_direction="both")
                        .fillna(X[col].median())
                    )

            model = xgb.XGBRegressor(random_state=42, n_estimators=100, max_depth=5)
            cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
            cv_r2 = cv_scores.mean()
            if cv_r2 < 0.1:
                miya_alerts(
                    f"Modèle XGBoost faible (CV R²={cv_r2:.2f})",
                    tag="OPTIMIZE",
                    priority=2,
                    voice_profile="warning",
                )
                send_alert(f"Modèle XGBoost faible (CV R²={cv_r2:.2f})", priority=2)
                send_telegram_alert(f"Modèle XGBoost faible (CV R²={cv_r2:.2f})")
                logger.warning(f"Modèle XGBoost faible (CV R²={cv_r2:.2f})")

            model.fit(X, y)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            shap_importance = pd.Series(
                np.abs(shap_values).mean(axis=0), index=X.columns
            ).sort_values(ascending=False)

            shap_importance = (shap_importance - shap_importance.min()) / (
                shap_importance.max() - shap_importance.min() + 1e-6
            )

            self.cache[cache_key] = shap_importance
            if len(self.cache) > self.max_cache_size:
                self.cache.pop(next(iter(self.cache)))

            shap_df = pd.DataFrame(
                {"feature": shap_importance.index, "importance": shap_importance.values}
            )

            def write_shap():
                shap_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False, encoding="utf-8")

            self.with_retries(write_shap)

            latency = time.time() - start_time
            miya_speak(
                f"Importance SHAP calculée pour {len(shap_importance)} features",
                tag="OPTIMIZE",
                voice_profile="calm",
                priority=1,
            )
            send_alert(
                f"Importance SHAP calculée pour {len(shap_importance)} features",
                priority=1,
            )
            send_telegram_alert(
                f"Importance SHAP calculée pour {len(shap_importance)} features"
            )
            logger.info(
                f"Importance SHAP calculée pour {len(shap_importance)} features"
            )
            self.log_performance(
                "calculate_shap_importance",
                latency,
                success=True,
                num_features=len(shap_importance),
                confidence_drop_rate=confidence_drop_rate,
            )
            self.save_snapshot(
                "calculate_shap_importance",
                {
                    "num_features": len(shap_importance),
                    "top_feature": shap_importance.index[0],
                    "confidence_drop_rate": confidence_drop_rate,
                },
                compress=False,
            )
            return shap_importance
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans calculate_shap_importance: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="OPTIMIZE", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "calculate_shap_importance", latency, success=False, error=str(e)
            )
            self.save_snapshot(
                "calculate_shap_importance", {"error": str(e)}, compress=False
            )
            return pd.Series(
                0.0,
                index=data.drop(
                    columns=["timestamp", target_col, "neural_regime"], errors="ignore"
                ).columns,
            )

    def reduce_features(
        self,
        data: pd.DataFrame,
        market_condition: str,
        target_dims: int = 81,
        boost_factors: Dict[str, float] = None,
    ) -> pd.DataFrame:
        """Réduit le vecteur d'observation en fonction des conditions de marché."""
        start_time = time.time()
        try:
            required_cols = ["timestamp", "neural_regime"]
            missing_cols = [col for col in required_cols if col not in data.columns]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )
            if len(data) < self.config.get("window_size", 20):
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(required_cols) - len(missing_cols)}/{len(required_cols)} colonnes valides, {len(data)} lignes)"
                miya_alerts(alert_msg, tag="REDUCE", voice_profile="urgent", priority=3)
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            boost_factors = boost_factors or self.config.get(
                "boost_factors", {"Trend": 1.5, "Range": 1.5, "Defensive": 1.5}
            )
            cache_key = hashlib.sha256(
                f"{data.to_json()}_{market_condition}_{target_dims}".encode()
            ).hexdigest()
            cache_path = os.path.join(CACHE_DIR, f"reduced_{cache_key}.csv")
            if os.path.exists(cache_path):
                reduced_data = pd.read_csv(cache_path)
                if len(reduced_data) == len(data) and reduced_data["timestamp"].equals(
                    data["timestamp"]
                ):
                    latency = time.time() - start_time
                    miya_speak(
                        f"Features réduites chargées depuis cache pour {market_condition}",
                        tag="REDUCE",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert(
                        f"Features réduites chargées depuis cache pour {market_condition}",
                        priority=1,
                    )
                    send_telegram_alert(
                        f"Features réduites chargées depuis cache pour {market_condition}"
                    )
                    logger.info(
                        f"Features réduites chargées depuis cache pour {market_condition}"
                    )
                    self.log_performance(
                        "reduce_features_cache_hit",
                        latency,
                        success=True,
                        num_features=len(reduced_data.columns) - 1,
                        confidence_drop_rate=confidence_drop_rate,
                    )
                    return reduced_data

            shap_importance = self.calculate_shap_importance(data)

            momentum_features = self.config.get(
                "momentum_features",
                [
                    "spy_lead_return",
                    "sector_leader_momentum",
                    "spy_momentum_diff",
                    "order_flow_acceleration",
                ],
            )
            volatility_features = self.config.get(
                "volatility_features",
                [
                    "atr_14",
                    "volatility_trend",
                    "microstructure_volatility",
                    "vix_es_correlation",
                    "iv_atm",
                    "option_skew",
                ],
            )
            risk_features = self.config.get(
                "risk_features",
                [
                    "bond_equity_risk_spread",
                    "vanna_cliff_slope",
                    "vanna_exposure",
                    "delta_exposure",
                    "gex_slope",
                ],
            )
            mandatory_features = (
                list(set(obs_t) - {"timestamp", "neural_regime"})
                if target_dims == 81
                else []
            )

            adjusted_importance = shap_importance.copy()
            if market_condition == "Trend":
                for feature in momentum_features:
                    if feature in adjusted_importance:
                        adjusted_importance[feature] *= boost_factors["Trend"]
            elif market_condition == "Range":
                for feature in volatility_features:
                    if feature in adjusted_importance:
                        adjusted_importance[feature] *= boost_factors["Range"]
            elif market_condition == "Defensive":
                for feature in risk_features:
                    if feature in adjusted_importance:
                        adjusted_importance[feature] *= boost_factors["Defensive"]

            selected_features = [
                f for f in mandatory_features if f in adjusted_importance
            ]
            remaining_dims = target_dims - len(selected_features)
            if remaining_dims > 0:
                additional_features = [
                    f for f in adjusted_importance.index if f not in selected_features
                ][:remaining_dims]
                selected_features.extend(additional_features)

            keep_cols = ["timestamp"] + selected_features
            reduced_data = data[keep_cols]

            self.validate_shap_features(selected_features)

            os.makedirs(CACHE_DIR, exist_ok=True)

            def write_cache():
                reduced_data.to_csv(cache_path, index=False, encoding="utf-8")

            self.with_retries(write_cache)
            file_size = os.path.getsize(cache_path) / 1024 / 1024
            if file_size > 1.0:
                miya_alerts(
                    f"Cache reduced_data size {file_size:.2f} MB exceeds 1 MB",
                    tag="REDUCE",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(
                    f"Cache reduced_data size {file_size:.2f} MB exceeds 1 MB",
                    priority=3,
                )
                send_telegram_alert(
                    f"Cache reduced_data size {file_size:.2f} MB exceeds 1 MB"
                )

            latency = time.time() - start_time
            miya_speak(
                f"Features sélectionnées ({len(selected_features)} dimensions) pour {market_condition}",
                tag="REDUCE",
                voice_profile="calm",
                priority=1,
            )
            send_alert(
                f"Features sélectionnées ({len(selected_features)} dimensions) pour {market_condition}",
                priority=1,
            )
            send_telegram_alert(
                f"Features sélectionnées ({len(selected_features)} dimensions) pour {market_condition}"
            )
            logger.info(f"Features sélectionnées: {selected_features}")
            self.log_performance(
                "reduce_features",
                latency,
                success=True,
                num_features=len(selected_features),
                market_condition=market_condition,
                confidence_drop_rate=confidence_drop_rate,
            )
            self.save_snapshot(
                "reduce_features",
                {
                    "num_features": len(selected_features),
                    "market_condition": market_condition,
                    "confidence_drop_rate": confidence_drop_rate,
                },
                compress=False,
            )
            self.plot_shap_importance(
                shap_importance,
                market_condition,
                data["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S"),
            )
            return reduced_data
        except Exception as e:
            latency = time.time() - start_time
            error_msg = (
                f"Erreur dans reduce_features: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="REDUCE", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "reduce_features", latency, success=False, error=str(e)
            )
            self.save_snapshot("reduce_features", {"error": str(e)}, compress=False)
            return data[["timestamp"] + list(obs_t)[:target_dims]]

    def optimize_features(
        self,
        data: pd.DataFrame,
        config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml"),
    ) -> pd.DataFrame:
        """Optimise le vecteur d'observation en réduisant dynamiquement les dimensions."""
        start_time = time.time()
        try:
            required_cols = ["timestamp", "neural_regime"]
            missing_cols = [col for col in required_cols if col not in data.columns]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )
            if len(data) < self.config.get("window_size", 20):
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(required_cols) - len(missing_cols)}/{len(required_cols)} colonnes valides, {len(data)} lignes)"
                miya_alerts(
                    alert_msg, tag="OPTIMIZE", voice_profile="urgent", priority=3
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            config = self.load_config(config_path)
            target_dims = config.get("target_dims", 81)
            config.get("window_size", 20)
            boost_factors = config.get(
                "boost_factors", {"Trend": 1.5, "Range": 1.5, "Defensive": 1.5}
            )

            cache_key = hashlib.sha256(
                f"{config_path}_{data.to_json()}".encode()
            ).hexdigest()
            cache_path = os.path.join(CACHE_DIR, f"optimized_{cache_key}.csv")
            if os.path.exists(cache_path):
                reduced_data = pd.read_csv(cache_path)
                if len(reduced_data) == len(data) and reduced_data["timestamp"].equals(
                    data["timestamp"]
                ):
                    latency = time.time() - start_time
                    miya_speak(
                        "Features réduites récupérées du cache",
                        tag="OPTIMIZE",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert("Features réduites récupérées du cache", priority=1)
                    send_telegram_alert("Features réduites récupérées du cache")
                    logger.info("Features réduites récupérées du cache")
                    self.log_performance(
                        "optimize_features_cache_hit",
                        latency,
                        success=True,
                        num_features=len(reduced_data.columns) - 1,
                        confidence_drop_rate=confidence_drop_rate,
                    )
                    return reduced_data

            if data.empty:
                error_msg = "DataFrame vide"
                miya_alerts(
                    error_msg, tag="OPTIMIZE", voice_profile="urgent", priority=3
                )
                send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            data = data.copy()
            for col in missing_cols:
                if col == "timestamp":
                    data[col] = pd.date_range(
                        start=pd.Timestamp.now(), periods=len(data), freq="1min"
                    )
                    miya_speak(
                        "Colonne timestamp manquante, création par défaut",
                        tag="OPTIMIZE",
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
                        data["neural_regime"].mode()[0]
                        if "neural_regime" in data.columns
                        and not data["neural_regime"].empty
                        else "Range"
                    )
                    miya_speak(
                        f"Colonne neural_regime manquante, imputée à {data[col].iloc[0]}",
                        tag="OPTIMIZE",
                        voice_profile="warning",
                        priority=2,
                    )
                    send_alert(
                        f"Colonne neural_regime manquante, imputée à {data[col].iloc[0]}",
                        priority=2,
                    )
                    send_telegram_alert(
                        f"Colonne neural_regime manquante, imputée à {data[col].iloc[0]}"
                    )

            data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
            if data["timestamp"].isna().any():
                error_msg = "NaN dans les timestamps"
                miya_alerts(
                    error_msg, tag="OPTIMIZE", voice_profile="urgent", priority=3
                )
                send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)
            if not data["timestamp"].is_monotonic_increasing:
                error_msg = "Timestamps non croissants"
                miya_alerts(
                    error_msg, tag="OPTIMIZE", voice_profile="urgent", priority=3
                )
                send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            for col in data.columns:
                if col != "timestamp" and data[col].isna().any():
                    data[col] = (
                        data[col]
                        .interpolate(method="linear", limit_direction="both")
                        .fillna(data[col].median())
                    )
                    if data[col].isna().sum() / len(data) > 0.1:
                        error_msg = f"Plus de 10% de NaN dans {col}"
                        miya_alerts(
                            error_msg,
                            tag="OPTIMIZE",
                            voice_profile="urgent",
                            priority=3,
                        )
                        send_alert(error_msg, priority=3)
                        send_telegram_alert(error_msg)

            reduced_data = pd.DataFrame()
            for regime in ["Trend", "Range", "Defensive"]:
                regime_data = data[data["neural_regime"] == regime]
                if not regime_data.empty:
                    reduced_regime = self.reduce_features(
                        regime_data, regime, target_dims, boost_factors
                    )
                    reduced_data = pd.concat(
                        [reduced_data, reduced_regime], ignore_index=True
                    )

            reduced_data = reduced_data.sort_values("timestamp").reset_index(drop=True)
            reduced_data["timestamp"] = data["timestamp"]

            os.makedirs(CACHE_DIR, exist_ok=True)

            def write_cache():
                reduced_data.to_csv(cache_path, index=False, encoding="utf-8")

            self.with_retries(write_cache)
            file_size = os.path.getsize(cache_path) / 1024 / 1024
            if file_size > 1.0:
                miya_alerts(
                    f"Cache optimized_data size {file_size:.2f} MB exceeds 1 MB",
                    tag="OPTIMIZE",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(
                    f"Cache optimized_data size {file_size:.2f} MB exceeds 1 MB",
                    priority=3,
                )
                send_telegram_alert(
                    f"Cache optimized_data size {file_size:.2f} MB exceeds 1 MB"
                )

            latency = time.time() - start_time
            miya_speak(
                f"Vecteur d'observation réduit à {len(reduced_data.columns)-1} dimensions",
                tag="OPTIMIZE",
                voice_profile="calm",
                priority=1,
            )
            send_alert(
                f"Vecteur d'observation réduit à {len(reduced_data.columns)-1} dimensions",
                priority=1,
            )
            send_telegram_alert(
                f"Vecteur d'observation réduit à {len(reduced_data.columns)-1} dimensions"
            )
            logger.info(
                f"Vecteur d'observation réduit à {len(reduced_data.columns)-1} dimensions"
            )
            self.log_performance(
                "optimize_features",
                latency,
                success=True,
                num_features=len(reduced_data.columns) - 1,
                confidence_drop_rate=confidence_drop_rate,
            )
            self.save_snapshot(
                "optimize_features",
                {
                    "num_features": len(reduced_data.columns) - 1,
                    "confidence_drop_rate": confidence_drop_rate,
                },
                compress=False,
            )
            return reduced_data
        except Exception as e:
            latency = time.time() - start_time
            error_msg = (
                f"Erreur dans optimize_features: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="OPTIMIZE", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "optimize_features", latency, success=False, error=str(e)
            )
            self.save_snapshot("optimize_features", {"error": str(e)}, compress=False)
            return data[["timestamp"] + list(obs_t)[:target_dims]]

    def reduce_incremental_features(
        self,
        row: pd.Series,
        market_condition: str,
        target_dims: int = 81,
        shap_importance: Optional[pd.Series] = None,
        config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml"),
    ) -> pd.Series:
        """Réduit les features pour une seule ligne en temps réel."""
        start_time = time.time()
        try:
            required_cols = ["timestamp", "neural_regime"]
            missing_cols = [col for col in required_cols if col not in row.index]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(required_cols) - len(missing_cols)}/{len(required_cols)} colonnes valides)"
                miya_alerts(
                    alert_msg, tag="OPTIMIZE", voice_profile="urgent", priority=3
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            config = self.load_config(config_path)
            window_size = config.get("window_size", 20)

            row = row.copy()
            row["timestamp"] = pd.to_datetime(row["timestamp"], errors="coerce")
            if pd.isna(row["timestamp"]):
                error_msg = "Timestamp invalide dans la ligne"
                miya_alerts(
                    error_msg, tag="OPTIMIZE", voice_profile="urgent", priority=3
                )
                send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            self.buffer.append(row.to_frame().T)
            if len(self.buffer) < window_size:
                return row[["timestamp"] + list(obs_t)[:target_dims]]

            buffer_data = pd.concat(list(self.buffer), ignore_index=True).tail(
                window_size
            )
            if shap_importance is None:
                shap_importance = self.calculate_shap_importance(buffer_data)

            reduced_data = self.reduce_features(
                buffer_data, market_condition, target_dims, config.get("boost_factors")
            )

            latency = time.time() - start_time
            miya_speak(
                f"Features réduites incrémentalement à {target_dims} dimensions pour {market_condition}",
                tag="OPTIMIZE",
                voice_profile="calm",
                priority=1,
            )
            send_alert(
                f"Features réduites incrémentalement à {target_dims} dimensions pour {market_condition}",
                priority=1,
            )
            send_telegram_alert(
                f"Features réduites incrémentalement à {target_dims} dimensions pour {market_condition}"
            )
            logger.info(
                f"Features réduites incrémentalement à {target_dims} dimensions pour {market_condition}"
            )
            self.log_performance(
                "reduce_incremental_features",
                latency,
                success=True,
                num_features=target_dims,
                market_condition=market_condition,
                confidence_drop_rate=confidence_drop_rate,
            )
            self.save_snapshot(
                "reduce_incremental_features",
                {
                    "num_features": target_dims,
                    "market_condition": market_condition,
                    "confidence_drop_rate": confidence_drop_rate,
                },
                compress=False,
            )
            return reduced_data.iloc[-1]
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans reduce_incremental_features: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="OPTIMIZE", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "reduce_incremental_features", latency, success=False, error=str(e)
            )
            self.save_snapshot(
                "reduce_incremental_features", {"error": str(e)}, compress=False
            )
            return row[["timestamp"] + list(obs_t)[:target_dims]]


if __name__ == "__main__":
    try:
        ensemble = FeatureMetaEnsemble()
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "close": np.random.normal(5100, 10, 100),
                "neural_regime": ["Trend"] * 20 + ["Range"] * 40 + ["Defensive"] * 40,
                "atr_14": np.random.uniform(0.5, 2.0, 100),
                "volatility_trend": np.random.uniform(-0.1, 0.1, 100),
                "spy_lead_return": np.random.uniform(-0.02, 0.02, 100),
                "sector_leader_momentum": np.random.uniform(-1, 1, 100),
                "order_flow_acceleration": np.random.uniform(-0.5, 0.5, 100),
                "microstructure_volatility": np.random.uniform(0, 0.1, 100),
                "vix_es_correlation": np.random.uniform(-1, 1, 100),
                "bond_equity_risk_spread": np.random.uniform(-1, 1, 100),
                "vanna_cliff_slope": np.random.uniform(-0.1, 0.1, 100),
                "vanna_exposure": np.random.uniform(-1000, 1000, 100),
                "delta_exposure": np.random.uniform(-1000, 1000, 100),
                "iv_atm": np.random.uniform(0.1, 0.3, 100),
                "option_skew": np.random.uniform(-0.5, 0.5, 100),
                "gex_slope": np.random.uniform(-0.1, 0.1, 100),
                "trade_success_prob": np.random.uniform(0, 1, 100),
            }
        )
        for i in range(350 - len(data.columns) + 1):
            data[f"feature_{i}"] = np.random.normal(0, 1, 100)

        result = ensemble.optimize_features(data)
        print(result.head())
        print(f"Nombre de dimensions après réduction : {len(result.columns)-1}")
        miya_speak(
            "Test optimize_features terminé",
            tag="TEST",
            voice_profile="calm",
            priority=1,
        )
        send_alert("Test optimize_features terminé", priority=1)
        send_telegram_alert("Test optimize_features terminé")
        logger.info("Test optimize_features terminé")
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
