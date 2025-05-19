# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/features/pca_orderflow.py
# Applique PCA sur les indicateurs d'order flow pour réduction de dimension.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Analyse PCA des features d'order flow avec pondération par régime (méthode 3),
#        logs psutil, validation SHAP (méthode 17), et compatibilité top 150 SHAP.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, sklearn>=1.0.2,<1.1.0, psutil>=5.9.0,<6.0.0,
#   pyyaml>=6.0.0,<7.0.0, matplotlib>=3.7.0,<3.8.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
# - src/model/utils/obs_template.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/es_config.yaml
# - data/features/feature_importance.csv
#
# Outputs :
# - data/features/pca_orderflow.csv
# - data/logs/pca_orderflow_performance.csv
# - data/pca_snapshots/*.json (option *.json.gz)
# - data/figures/pca/
#
# Notes :
# - Conforme à structure.txt (version 2.1.3, 2025-05-13).
# - Utilise retries (max 3, délai 2^attempt secondes) pour les opérations critiques.
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des composantes PCA.
# - Intègre validation SHAP (Phase 17) pour assurer la conformité avec les top 150 SHAP features.
# - Sauvegarde des snapshots JSON non compressés par défaut avec option de compression gzip.
# - Envoie des alertes via AlertManager et telegram_alert pour les erreurs critiques et succès.
# - Tests unitaires disponibles dans tests/test_pca_orderflow.py.
# - Conforme à la ligne rouge des top 150 SHAP features définie dans feature_pipeline.py.

import hashlib
import json
import logging
import os
import pickle
import traceback
from collections import deque
from datetime import datetime
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.model.utils.alert_manager import send_alert
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.model.utils.obs_template import obs_t
from src.utils.telegram_alert import send_telegram_alert

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "pca_snapshots")
CSV_LOG_PATH = os.path.join(BASE_DIR, "data", "logs", "pca_orderflow_performance.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures", "pca")
FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR, "data", "features", "feature_importance.csv"
)

# Création des répertoires
os.makedirs(os.path.join(BASE_DIR, "data", "logs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data", "models"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data", "features"), exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Configuration du logging
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "data", "logs", "pca_orderflow.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Constantes
MAX_RETRIES = 3
RETRY_DELAY = 2.0


class PCAOrderFlow:

    def __init__(
        self, config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml")
    ):
        self.log_buffer = []
        self.cache = {}
        self.buffer = deque(maxlen=1000)
        self.scaler = None
        self.pca = None
        try:
            self.config = self.load_config(config_path)
            self.buffer_size = self.config.get("buffer_size", 100)
            self.max_cache_size = self.config.get("max_cache_size", 1000)
            self.buffer_maxlen = self.config.get("buffer_maxlen", 1000)
            self.buffer = deque(maxlen=self.buffer_maxlen)
            self.load_models()
            miya_speak(
                "PCAOrderFlow initialisé", tag="PCA", voice_profile="calm", priority=2
            )
            send_alert("PCAOrderFlow initialisé", priority=2)
            send_telegram_alert("PCAOrderFlow initialisé")
            logger.info("PCAOrderFlow initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config_path": config_path}, compress=False)
        except Exception as e:
            error_msg = f"Erreur initialisation PCAOrderFlow: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="PCA", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            self.config = {
                "n_components": 2,
                "output_csv": os.path.join(
                    BASE_DIR, "data", "features", "pca_orderflow.csv"
                ),
                "window_size": 100,
                "model_path": os.path.join(
                    BASE_DIR, "data", "models", "pca_orderflow.pkl"
                ),
                "scaler_path": os.path.join(
                    BASE_DIR, "data", "models", "scaler_orderflow.pkl"
                ),
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
        """Charge la configuration via yaml."""

        def load_yaml():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            if "pca_orderflow" not in config:
                raise ValueError("Clé 'pca_orderflow' manquante dans la configuration")
            required_keys = [
                "n_components",
                "output_csv",
                "window_size",
                "model_path",
                "scaler_path",
            ]
            missing_keys = [
                key for key in required_keys if key not in config["pca_orderflow"]
            ]
            if missing_keys:
                raise ValueError(
                    f"Clés manquantes dans 'pca_orderflow': {missing_keys}"
                )
            return config["pca_orderflow"]

        try:
            start_time = time.time()
            config = self.with_retries(load_yaml)
            cache_key = hashlib.sha256(str(config).encode()).hexdigest()
            self.cache[cache_key] = config
            if len(self.cache) > self.max_cache_size:
                self.cache.pop(next(iter(self.cache)))
            latency = time.time() - start_time
            miya_speak(
                "Configuration pca_orderflow chargée",
                tag="PCA",
                voice_profile="calm",
                priority=2,
            )
            send_alert("Configuration pca_orderflow chargée", priority=2)
            send_telegram_alert("Configuration pca_orderflow chargée")
            logger.info("Configuration pca_orderflow chargée")
            self.log_performance("load_config", latency, success=True)
            self.save_snapshot(
                "load_config", {"config_path": config_path}, compress=False
            )
            return config
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur chargement config: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="PCA", voice_profile="urgent", priority=3)
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
                        tag="PCA",
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
                    tag="PCA",
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

    def load_models(self) -> None:
        """Charge les modèles PCA et scaler depuis le disque."""
        try:
            start_time = time.time()
            model_path = self.config.get(
                "model_path",
                os.path.join(BASE_DIR, "data", "models", "pca_orderflow.pkl"),
            )
            scaler_path = self.config.get(
                "scaler_path",
                os.path.join(BASE_DIR, "data", "models", "scaler_orderflow.pkl"),
            )
            if os.path.exists(model_path):
                self.pca = pickle.load(open(model_path, "rb"))
                miya_speak(
                    f"Modèle PCA chargé: {model_path}",
                    tag="PCA",
                    voice_profile="calm",
                    priority=1,
                )
                send_alert(f"Modèle PCA chargé: {model_path}", priority=1)
                send_telegram_alert(f"Modèle PCA chargé: {model_path}")
                logger.info(f"Modèle PCA chargé: {model_path}")
            if os.path.exists(scaler_path):
                self.scaler = pickle.load(open(scaler_path, "rb"))
                miya_speak(
                    f"Scaler chargé: {scaler_path}",
                    tag="PCA",
                    voice_profile="calm",
                    priority=1,
                )
                send_alert(f"Scaler chargé: {scaler_path}", priority=1)
                send_telegram_alert(f"Scaler chargé: {scaler_path}")
                logger.info(f"Scaler chargé: {scaler_path}")
            latency = time.time() - start_time
            self.log_performance("load_models", latency, success=True)
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur chargement modèles PCA/scaler: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="PCA", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("load_models", latency, success=False, error=str(e))
            self.pca = None
            self.scaler = None

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
                    tag="PCA",
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
                tag="PCA",
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
                    tag="PCA",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(f"Snapshot size {file_size:.2f} MB exceeds 1 MB", priority=3)
                send_telegram_alert(f"Snapshot size {file_size:.2f} MB exceeds 1 MB")
            latency = time.time() - start_time
            miya_speak(
                f"Snapshot {snapshot_type} sauvegardé: {save_path}",
                tag="PCA",
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
                tag="PCA",
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
                    tag="PCA",
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
                    tag="PCA",
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
                    tag="PCA",
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
                "SHAP features validées", tag="PCA", voice_profile="calm", priority=1
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
                tag="PCA",
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

    def plot_pca_results(
        self, pca_result: np.ndarray, timestamp: str, explained_variance: np.ndarray
    ) -> None:
        """Génère des visualisations des résultats PCA."""
        start_time = time.time()
        try:
            timestamp_safe = timestamp.replace(":", "-")
            os.makedirs(FIGURES_DIR, exist_ok=True)

            plt.figure(figsize=(8, 6))
            plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
            plt.title(f"Composantes PCA 1 et 2 - {timestamp}")
            plt.xlabel("PCA 1")
            plt.ylabel("PCA 2")
            plt.grid(True)
            plt.savefig(os.path.join(FIGURES_DIR, f"pca_scatter_{timestamp_safe}.png"))
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.plot(
                range(1, len(explained_variance) + 1), explained_variance, marker="o"
            )
            plt.title(f"Variance Expliquée par Composante - {timestamp}")
            plt.xlabel("Composante")
            plt.ylabel("Variance Expliquée")
            plt.grid(True)
            plt.savefig(os.path.join(FIGURES_DIR, f"pca_scree_{timestamp_safe}.png"))
            plt.close()

            latency = time.time() - start_time
            miya_speak(
                f"Visualisations PCA générées: {FIGURES_DIR}",
                tag="PCA",
                voice_profile="calm",
                priority=2,
            )
            send_alert(f"Visualisations PCA générées: {FIGURES_DIR}", priority=2)
            send_telegram_alert(f"Visualisations PCA générées: {FIGURES_DIR}")
            logger.info(f"Visualisations PCA générées: {FIGURES_DIR}")
            self.log_performance("plot_pca_results", latency, success=True)
        except Exception as e:
            latency = time.time() - start_time
            error_msg = (
                f"Erreur génération visualisations: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="PCA", voice_profile="urgent", priority=2)
            send_alert(error_msg, priority=2)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "plot_pca_results", latency, success=False, error=str(e)
            )

    def apply_pca_orderflow(
        self,
        data: pd.DataFrame,
        regime: str = "range",
        config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml"),
    ) -> pd.DataFrame:
        """Applique PCA sur les features d'order flow."""
        try:
            start_time = time.time()
            config = self.load_config(config_path)
            n_components = config.get("n_components", 2)
            output_csv = config.get(
                "output_csv",
                os.path.join(BASE_DIR, "data", "features", "pca_orderflow.csv"),
            )
            window_size = config.get("window_size", 100)
            model_path = config.get(
                "model_path",
                os.path.join(BASE_DIR, "data", "models", "pca_orderflow.pkl"),
            )
            scaler_path = config.get(
                "scaler_path",
                os.path.join(BASE_DIR, "data", "models", "scaler_orderflow.pkl"),
            )

            weights = {"trend": 1.5, "range": 1.0, "defensive": 0.5}
            regime_weight = weights.get(regime, 1.0)
            valid_regimes = list(weights.keys())
            if regime not in valid_regimes:
                error_msg = f"Régime invalide: {regime}, attendu: {valid_regimes}"
                miya_alerts(error_msg, tag="PCA", voice_profile="urgent", priority=5)
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            cache_key = hashlib.sha256(
                f"{config_path}_{data.to_json()}_{regime}".encode()
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
                    "Composantes PCA récupérées du cache",
                    tag="PCA",
                    voice_profile="calm",
                    priority=1,
                )
                send_alert("Composantes PCA récupérées du cache", priority=1)
                send_telegram_alert("Composantes PCA récupérées du cache")
                self.log_performance("apply_pca_orderflow_cache_hit", 0, success=True)
                return self.cache[cache_key]["result"]

            if data.empty:
                error_msg = "DataFrame vide"
                miya_alerts(error_msg, tag="PCA", voice_profile="urgent", priority=5)
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            data = data.copy()
            if "timestamp" not in data.columns:
                error_msg = "Colonne 'timestamp' manquante"
                miya_alerts(error_msg, tag="PCA", voice_profile="urgent", priority=5)
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
            if data["timestamp"].isna().any():
                error_msg = "NaN dans les timestamps"
                miya_alerts(error_msg, tag="PCA", voice_profile="urgent", priority=5)
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)
            if not data["timestamp"].is_monotonic_increasing:
                error_msg = "Timestamps non croissants"
                miya_alerts(error_msg, tag="PCA", voice_profile="urgent", priority=5)
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            orderflow_features = [
                "delta_volume",
                "obi_score",
                "vds",
                "absorption_strength",
                "bid_size_level_1",
                "ask_size_level_1",
                "bid_price_level_1",
                "ask_price_level_1",
                "bid_size_level_2",
                "ask_size_level_2",
                "bid_price_level_2",
                "ask_price_level_2",
                "trade_frequency_1s",
            ]
            expected_count = config.get("expected_count", 81)
            if expected_count == 320:
                feature_set = orderflow_features
            else:
                feature_set = [col for col in orderflow_features if col in obs_t]
            available_features = [col for col in feature_set if col in data.columns]
            missing_features = [
                col for col in feature_set if col not in available_features
            ]
            if missing_features:
                miya_speak(
                    f"Features manquantes: {missing_features}, imputées à 0",
                    tag="PCA",
                    voice_profile="warning",
                    priority=2,
                )
                send_alert(f"Features manquantes: {missing_features}", priority=2)
                send_telegram_alert(f"Features manquantes: {missing_features}")
                logger.warning(f"Features manquantes: {missing_features}")
                for col in missing_features:
                    data[col] = 0.0
                available_features = feature_set

            if not available_features:
                error_msg = "Aucune feature d'order flow disponible"
                miya_alerts(error_msg, tag="PCA", voice_profile="urgent", priority=5)
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            orderflow_data = data[available_features].tail(window_size).copy()

            for col in available_features:
                if orderflow_data[col].isna().any():
                    orderflow_data[col] = (
                        orderflow_data[col]
                        .interpolate(method="linear", limit_direction="both")
                        .fillna(orderflow_data[col].median())
                    )
                    if orderflow_data[col].isna().sum() / len(orderflow_data) > 0.1:
                        error_msg = f"Plus de 10% de NaN dans {col}"
                        miya_alerts(
                            error_msg, tag="PCA", voice_profile="urgent", priority=5
                        )
                        send_alert(error_msg, priority=5)
                        send_telegram_alert(error_msg)
                        raise ValueError(error_msg)

            if not np.isfinite(orderflow_data.values).all():
                error_msg = "Valeurs non finies détectées"
                miya_alerts(error_msg, tag="PCA", voice_profile="urgent", priority=5)
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            # Calculer confidence_drop_rate (Phase 8)
            valid_features_count = len(
                [
                    col
                    for col in available_features
                    if not orderflow_data[col].isna().all()
                ]
            )
            confidence_drop_rate = 1.0 - min(
                valid_features_count / len(feature_set), 1.0
            )
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({valid_features_count}/{len(feature_set)} features valides)"
                miya_alerts(alert_msg, tag="PCA", voice_profile="urgent", priority=3)
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            # Appliquer la pondération par régime
            orderflow_data = orderflow_data * regime_weight

            if self.scaler is None:
                self.scaler = StandardScaler()
                self.scaler.fit(orderflow_data)
                with open(scaler_path, "wb") as f:
                    pickle.dump(self.scaler, f)
            scaled_data = self.scaler.transform(orderflow_data)

            if self.pca is None:
                self.pca = PCA(n_components=n_components)
                self.pca.fit(scaled_data)
                with open(model_path, "wb") as f:
                    pickle.dump(self.pca, f)
            pca_result = self.pca.transform(scaled_data)
            explained_variance = self.pca.explained_variance_ratio_
            total_variance = explained_variance.sum()
            if total_variance < 0.5:
                miya_alerts(
                    f"Variance expliquée faible ({total_variance:.2%})",
                    tag="PCA",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(
                    f"Variance expliquée faible ({total_variance:.2%})", priority=3
                )
                send_telegram_alert(f"Variance expliquée faible ({total_variance:.2%})")
            miya_speak(
                f"Variance expliquée: {total_variance:.2%}",
                tag="PCA",
                voice_profile="calm",
                priority=1,
            )
            send_alert(f"Variance expliquée: {total_variance:.2%}", priority=1)
            send_telegram_alert(f"Variance expliquée: {total_variance:.2%}")

            scaler = MinMaxScaler()
            pca_result = scaler.fit_transform(pca_result)

            pca_features = [f"pca_orderflow_{i+1}" for i in range(n_components)]
            self.validate_shap_features(pca_features)

            for i in range(n_components):
                data[f"pca_orderflow_{i+1}"] = 0.0
                data.iloc[
                    -len(pca_result) :, data.columns.get_loc(f"pca_orderflow_{i+1}")
                ] = pca_result[:, i]

            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            pca_df = pd.DataFrame(
                pca_result, columns=pca_features, index=data.tail(window_size).index
            )
            pca_df["timestamp"] = data["timestamp"].tail(window_size)

            def write_output():
                pca_df.to_csv(output_csv, index=False, encoding="utf-8")

            self.with_retries(write_output)
            miya_speak(
                f"Résultats PCA sauvegardés: {output_csv}",
                tag="PCA",
                voice_profile="calm",
                priority=1,
            )
            send_alert(f"Résultats PCA sauvegardés: {output_csv}", priority=1)
            send_telegram_alert(f"Résultats PCA sauvegardés: {output_csv}")

            self.cache[cache_key] = {"result": data, "timestamp": current_time}
            if len(self.cache) > self.max_cache_size:
                oldest_key = min(self.cache, key=lambda k: self.cache[k]["timestamp"])
                self.cache.pop(oldest_key)

            latency = time.time() - start_time
            miya_speak(
                "Composantes PCA calculées", tag="PCA", voice_profile="calm", priority=1
            )
            send_alert("Composantes PCA calculées", priority=1)
            send_telegram_alert("Composantes PCA calculées")
            logger.info("Composantes PCA calculées")
            self.log_performance(
                "apply_pca_orderflow",
                latency,
                success=True,
                num_rows=len(data),
                total_variance=total_variance,
                confidence_drop_rate=confidence_drop_rate,
            )
            self.save_snapshot(
                "apply_pca_orderflow",
                {
                    "num_rows": len(data),
                    "total_variance": float(total_variance),
                    "regime": regime,
                    "confidence_drop_rate": confidence_drop_rate,
                },
                compress=False,
            )
            self.plot_pca_results(
                pca_result,
                data["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S"),
                explained_variance,
            )
            return data
        except Exception as e:
            latency = time.time() - start_time
            error_msg = (
                f"Erreur dans apply_pca_orderflow: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="PCA", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "apply_pca_orderflow", latency, success=False, error=str(e)
            )
            self.save_snapshot("apply_pca_orderflow", {"error": str(e)}, compress=False)
            for i in range(config.get("n_components", 2)):
                data[f"pca_orderflow_{i+1}"] = 0.0
            return data

    def apply_incremental_pca_orderflow(
        self,
        row: pd.Series,
        regime: str = "range",
        config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml"),
    ) -> pd.Series:
        """Applique PCA incrémental sur une seule ligne."""
        try:
            start_time = time.time()
            config = self.load_config(config_path)
            n_components = config.get("n_components", 2)
            window_size = config.get("window_size", 100)

            weights = {"trend": 1.5, "range": 1.0, "defensive": 0.5}
            regime_weight = weights.get(regime, 1.0)
            valid_regimes = list(weights.keys())
            if regime not in valid_regimes:
                error_msg = f"Régime invalide: {regime}, attendu: {valid_regimes}"
                miya_alerts(error_msg, tag="PCA", voice_profile="urgent", priority=5)
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            row = row.copy()
            if "timestamp" not in row.index:
                error_msg = "Timestamp manquant dans la ligne"
                miya_alerts(error_msg, tag="PCA", voice_profile="urgent", priority=5)
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            row["timestamp"] = pd.to_datetime(row["timestamp"], errors="coerce")
            if pd.isna(row["timestamp"]):
                error_msg = "Timestamp invalide dans la ligne"
                miya_alerts(error_msg, tag="PCA", voice_profile="urgent", priority=5)
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            self.buffer.append(row.to_frame().T)
            if len(self.buffer) < window_size:
                return pd.Series(
                    {f"pca_orderflow_{i+1}": 0.0 for i in range(n_components)},
                    index=row.index,
                )

            orderflow_features = [
                "delta_volume",
                "obi_score",
                "vds",
                "absorption_strength",
                "bid_size_level_1",
                "ask_size_level_1",
                "bid_price_level_1",
                "ask_price_level_1",
                "bid_size_level_2",
                "ask_size_level_2",
                "bid_price_level_2",
                "ask_price_level_2",
                "trade_frequency_1s",
            ]
            expected_count = config.get("expected_count", 81)
            if expected_count == 320:
                feature_set = orderflow_features
            else:
                feature_set = [col for col in orderflow_features if col in obs_t]
            available_features = [col for col in feature_set if col in row.index]
            missing_features = [
                col for col in feature_set if col not in available_features
            ]
            if missing_features:
                for col in missing_features:
                    row[col] = 0.0
                available_features = feature_set

            orderflow_data = pd.concat(list(self.buffer), ignore_index=True).tail(
                window_size
            )
            orderflow_data = orderflow_data[available_features]

            # Calculer confidence_drop_rate (Phase 8)
            valid_features_count = len(
                [
                    col
                    for col in available_features
                    if not orderflow_data[col].isna().all()
                ]
            )
            confidence_drop_rate = 1.0 - min(
                valid_features_count / len(feature_set), 1.0
            )
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({valid_features_count}/{len(feature_set)} features valides)"
                miya_alerts(alert_msg, tag="PCA", voice_profile="urgent", priority=3)
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            for col in available_features:
                if orderflow_data[col].isna().any():
                    orderflow_data[col] = (
                        orderflow_data[col]
                        .interpolate(method="linear", limit_direction="both")
                        .fillna(orderflow_data[col].median())
                    )

            if self.scaler is None or self.pca is None:
                error_msg = "Modèles PCA/scaler non initialisés"
                miya_alerts(error_msg, tag="PCA", voice_profile="urgent", priority=5)
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            orderflow_data = orderflow_data * regime_weight
            scaled_data = self.scaler.transform(orderflow_data)
            pca_result = self.pca.transform(scaled_data[-1:])

            scaler = MinMaxScaler()
            pca_result = scaler.fit_transform(pca_result)

            result = pd.Series(
                {f"pca_orderflow_{i+1}": pca_result[0, i] for i in range(n_components)},
                index=row.index,
            )

            latency = time.time() - start_time
            miya_speak(
                "Incremental composantes PCA calculées",
                tag="PCA",
                voice_profile="calm",
                priority=1,
            )
            send_alert("Incremental composantes PCA calculées", priority=1)
            send_telegram_alert("Incremental composantes PCA calculées")
            logger.info("Incremental composantes PCA calculées")
            self.log_performance(
                "apply_incremental_pca_orderflow",
                latency,
                success=True,
                num_rows=1,
                confidence_drop_rate=confidence_drop_rate,
            )
            self.save_snapshot(
                "apply_incremental_pca_orderflow",
                {
                    "pca_result": pca_result.tolist(),
                    "regime": regime,
                    "confidence_drop_rate": confidence_drop_rate,
                },
                compress=False,
            )
            return result
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans apply_incremental_pca_orderflow: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="PCA", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "apply_incremental_pca_orderflow", latency, success=False, error=str(e)
            )
            return pd.Series(
                {f"pca_orderflow_{i+1}": 0.0 for i in range(n_components)},
                index=row.index,
            )


if __name__ == "__main__":
    try:
        calculator = PCAOrderFlow()
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "delta_volume": np.random.uniform(-500, 500, 100),
                "obi_score": np.random.uniform(-1, 1, 100),
                "vds": np.random.uniform(-200, 200, 100),
                "absorption_strength": np.random.uniform(0, 100, 100),
                "bid_size_level_1": np.random.randint(100, 1000, 100),
                "ask_size_level_1": np.random.randint(100, 1000, 100),
                "trade_frequency_1s": np.random.randint(0, 50, 100),
                "close": np.random.normal(5100, 10, 100),
            }
        )
        enriched_data = calculator.apply_pca_orderflow(data, regime="trend")
        print(enriched_data[["timestamp", "pca_orderflow_1", "pca_orderflow_2"]].head())
        miya_speak(
            "Test apply_pca_orderflow terminé",
            tag="TEST",
            voice_profile="calm",
            priority=1,
        )
        send_alert("Test apply_pca_orderflow terminé", priority=1)
        send_telegram_alert("Test apply_pca_orderflow terminé")
        logger.info("Test apply_pca_orderflow terminé")
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
