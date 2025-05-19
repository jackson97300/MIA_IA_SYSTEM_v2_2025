# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/features/contextual_state_encoder.py
# Encode les composantes latentes (UMAP, NLP) à partir des données IQFeed.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Encode latent_vol_regime_vec, topic_vector_news, et autres composantes latentes,
#        avec mémoire contextuelle (market_memory.db), cache intermédiaire, logs psutil,
#        validation SHAP (méthode 17), et compatibilité top 150 SHAP features.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, umap-learn>=0.5.3,<0.6.0, scikit-learn>=1.0.2,<2.0.0,
#   psutil>=5.9.0,<6.0.0, pyyaml>=6.0.0,<7.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/miya_console.py
# - src/model/utils/obs_template.py
# - src/utils/database_manager.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/es_config.yaml
# - data/news/news_data.csv
# - data/iqfeed/merged_data.csv
#
# Outputs :
# - data/latent/latent_vectors.csv
# - data/news/news_topics.csv
# - data/features/cache/contextual/*.csv
# - data/logs/contextual_encoder_performance.csv
# - data/contextual_snapshots/*.json (option *.json.gz)
# - market_memory.db (table clusters)
#
# Notes :
# - Conforme à structure.txt (version 2.1.3, 2025-05-13).
# - Utilise IQFeed via feature_pipeline.py, avec retries (max 3, délai 2^attempt secondes).
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des encodages.
# - Intègre validation SHAP (Phase 17) pour assurer la conformité avec les top 150 SHAP features.
# - Sauvegarde des snapshots JSON non compressés par défaut avec option de compression gzip.
# - Envoie des alertes via AlertManager et telegram_alert pour les erreurs critiques et succès.
# - Tests unitaires disponibles dans tests/test_contextual_state_encoder.py.
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
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from src.model.utils.alert_manager import send_alert
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.model.utils.obs_template import obs_t
from src.utils.database_manager import DatabaseManager
from src.utils.telegram_alert import send_telegram_alert

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data", "features", "cache", "contextual")
PERF_LOG_PATH = os.path.join(
    BASE_DIR, "data", "logs", "contextual_encoder_performance.csv"
)
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "contextual_snapshots")
FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR, "data", "features", "feature_importance.csv"
)

# Création des répertoires
os.makedirs(os.path.join(BASE_DIR, "data", "logs"), exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Configuration du logging
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "data", "logs", "contextual_state_encoder.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Constantes
MAX_RETRIES = 3
RETRY_DELAY = 2.0


class ContextualStateEncoder:
    """Gère l'encodage des composantes latentes avec cache, logs, et mémoire contextuelle."""

    def __init__(
        self, config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml")
    ):
        """Initialise l’encodeur de composantes latentes."""
        self.log_buffer = []
        self.cache = {}
        try:
            self.config = self.load_config(config_path)
            os.makedirs(CACHE_DIR, exist_ok=True)
            os.makedirs(os.path.dirname(PERF_LOG_PATH), exist_ok=True)
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            self.db_manager = DatabaseManager(
                os.path.join(BASE_DIR, "data", "market_memory.db")
            )
            miya_speak(
                "ContextualStateEncoder initialisé",
                tag="ENCODING",
                voice_profile="calm",
                priority=2,
            )
            send_alert("ContextualStateEncoder initialisé", priority=2)
            send_telegram_alert("ContextualStateEncoder initialisé")
            logger.info("ContextualStateEncoder initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config_path": config_path}, compress=False)
        except Exception as e:
            error_msg = f"Erreur initialisation ContextualStateEncoder: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="ENCODING", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            self.config = {
                "n_topics": 3,
                "window_size": 10,
                "min_news": 10,
                "buffer_size": 100,
                "max_cache_size": 1000,
                "cache_hours": 24,
            }
            self.db_manager = DatabaseManager(
                os.path.join(BASE_DIR, "data", "market_memory.db")
            )

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Charge la configuration depuis es_config.yaml."""

        def load_yaml():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            if "contextual_state_encoder" not in config:
                raise ValueError(
                    "Clé 'contextual_state_encoder' manquante dans la configuration"
                )
            required_keys = ["n_topics", "window_size", "min_news"]
            missing_keys = [
                key
                for key in required_keys
                if key not in config["contextual_state_encoder"]
            ]
            if missing_keys:
                raise ValueError(
                    f"Clés manquantes dans 'contextual_state_encoder': {missing_keys}"
                )
            return config["contextual_state_encoder"]

        try:
            start_time = time.time()
            config = self.with_retries(load_yaml)
            cache_key = hashlib.sha256(str(config).encode()).hexdigest()
            self.cache[cache_key] = {"config": config, "timestamp": datetime.now()}
            if len(self.cache) > self.config.get("max_cache_size", 1000):
                self.cache.pop(next(iter(self.cache)))
            latency = time.time() - start_time
            miya_speak(
                "Configuration contextual_state_encoder chargée",
                tag="ENCODING",
                voice_profile="calm",
                priority=2,
            )
            send_alert("Configuration contextual_state_encoder chargée", priority=2)
            send_telegram_alert("Configuration contextual_state_encoder chargée")
            logger.info("Configuration contextual_state_encoder chargée")
            self.log_performance("load_config", latency, success=True)
            self.save_snapshot(
                "load_config", {"config_path": config_path}, compress=False
            )
            return config
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur chargement config: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="ENCODING", voice_profile="urgent", priority=3)
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
                        tag="ENCODING",
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
                    tag="ENCODING",
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
                    tag="ENCODING",
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
                tag="ENCODING",
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
                    tag="ENCODING",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(f"Snapshot size {file_size:.2f} MB exceeds 1 MB", priority=3)
                send_telegram_alert(f"Snapshot size {file_size:.2f} MB exceeds 1 MB")
            latency = time.time() - start_time
            miya_speak(
                f"Snapshot {snapshot_type} sauvegardé: {save_path}",
                tag="ENCODING",
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
                tag="ENCODING",
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
                    tag="ENCODING",
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
                    tag="ENCODING",
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
                    tag="ENCODING",
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
                tag="ENCODING",
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
                tag="ENCODING",
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

    def save_to_market_memory(
        self, data: pd.DataFrame, event_type: str, features: List[str]
    ) -> None:
        """Sauvegarde les vecteurs latents dans market_memory.db (table clusters)."""
        try:
            start_time = time.time()
            for _, row in data.iterrows():
                feature_dict = {col: row[col] for col in features}
                self.db_manager.insert_cluster(
                    cluster_id=0,
                    event_type=event_type,
                    features=json.dumps(feature_dict),
                    timestamp=str(row["timestamp"]),
                )
            latency = time.time() - start_time
            self.log_performance(
                "save_to_market_memory", latency, success=True, num_rows=len(data)
            )
            self.save_snapshot(
                "market_memory",
                {"num_rows": len(data), "event_type": event_type},
                compress=False,
            )
            miya_speak(
                f"Vecteurs sauvegardés dans market_memory.db pour {event_type}",
                tag="ENCODING",
                voice_profile="calm",
                priority=1,
            )
            send_alert(
                f"Vecteurs sauvegardés dans market_memory.db pour {event_type}",
                priority=1,
            )
            send_telegram_alert(
                f"Vecteurs sauvegardés dans market_memory.db pour {event_type}"
            )
            logger.info(f"Vecteurs sauvegardés dans market_memory.db pour {event_type}")
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde market_memory: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance(
                "save_to_market_memory", 0, success=False, error=str(e)
            )
            miya_alerts(error_msg, tag="ENCODING", voice_profile="urgent", priority=4)
            send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def encode_vol_regime(
        self,
        data: pd.DataFrame,
        latent_vectors_path: str = os.path.join(
            BASE_DIR, "data", "latent", "latent_vectors.csv"
        ),
        config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml"),
    ) -> pd.DataFrame:
        """Encode les régimes de volatilité en utilisant UMAP et sauvegarde les vecteurs latents."""
        try:
            start_time = time.time()
            config = self.load_config(config_path)
            window_size = config.get("window_size", 1000)

            required_cols = ["timestamp", "atr_14", "volatility_trend", "close"]
            missing_cols = [col for col in required_cols if col not in data.columns]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )
            if len(data) < window_size:
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(required_cols) - len(missing_cols)}/{len(required_cols)} colonnes valides, {len(data)} lignes)"
                miya_alerts(alert_msg, tag="LATENT", voice_profile="urgent", priority=3)
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            data = data.copy()
            for col in missing_cols:
                data[col] = 0
                miya_speak(
                    f"Colonne IQFeed '{col}' manquante, imputée à 0",
                    tag="LATENT",
                    voice_profile="warning",
                    priority=2,
                )
                send_alert(f"Colonne '{col}' manquante, imputée à 0", priority=2)
                send_telegram_alert(f"Colonne '{col}' manquante, imputée à 0")
                logger.warning(f"Colonne manquante: {col}")

            for col in required_cols:
                if col != "timestamp" and data[col].isna().any():
                    data[col] = data[col].interpolate(
                        method="linear", limit_direction="both"
                    )
                    if data[col].isna().any():
                        median_value = data[col].median()
                        data[col] = data[col].fillna(median_value)
                        miya_speak(
                            f"NaN persistants dans {col}, imputés à la médiane ({median_value:.2f})",
                            tag="LATENT",
                            voice_profile="warning",
                            priority=2,
                        )
                        send_alert(
                            f"NaN persistants dans {col}, imputés à la médiane ({median_value:.2f})",
                            priority=2,
                        )
                        send_telegram_alert(
                            f"NaN persistants dans {col}, imputés à la médiane ({median_value:.2f})"
                        )
                        logger.warning(
                            f"NaN persistants dans {col}, imputés à la médiane ({median_value:.2f})"
                        )
                    if data[col].isna().sum() / len(data) > 0.1:
                        miya_alerts(
                            f"Plus de 10% de NaN dans {col}",
                            tag="LATENT",
                            voice_profile="urgent",
                            priority=4,
                        )
                        send_alert(f"Plus de 10% de NaN dans {col}", priority=4)
                        send_telegram_alert(f"Plus de 10% de NaN dans {col}")
                        logger.error(f"Plus de 10% de NaN dans {col}")

            cache_path = os.path.join(
                CACHE_DIR,
                f"vol_regime_{hashlib.sha256(data.to_json().encode()).hexdigest()}.csv",
            )
            if os.path.exists(cache_path):
                cached_data = pd.read_csv(cache_path)
                if len(cached_data) == len(data) and cached_data["timestamp"].equals(
                    data["timestamp"]
                ):
                    latency = time.time() - start_time
                    self.log_performance(
                        "encode_vol_regime_cache",
                        latency,
                        success=True,
                        confidence_drop_rate=confidence_drop_rate,
                    )
                    miya_speak(
                        "Vol regime chargé depuis cache",
                        tag="LATENT",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert("Vol regime chargé depuis cache", priority=1)
                    send_telegram_alert("Vol regime chargé depuis cache")
                    logger.info("Vol regime chargé depuis cache")
                    return cached_data

            vol_features = (
                data[["atr_14", "volatility_trend", "close"]].tail(window_size).values
            )
            if len(vol_features) < 2:
                logger.warning(
                    "Données insuffisantes pour UMAP, retour de vecteurs nuls"
                )
                latent_vectors = np.zeros((len(data), 2))
            else:
                umap = UMAP(
                    n_components=2,
                    random_state=42,
                    n_neighbors=min(15, len(vol_features) - 1),
                )
                latent_vectors = umap.fit_transform(vol_features)
                if len(vol_features) < len(data):
                    latent_vectors = np.pad(
                        latent_vectors,
                        ((0, len(data) - len(vol_features)), (0, 0)),
                        mode="constant",
                    )

            latent_df = pd.DataFrame(
                {
                    "timestamp": data["timestamp"],
                    "latent_vol_regime_vec_0": latent_vectors[:, 0],
                    "latent_vol_regime_vec_1": latent_vectors[:, 1],
                }
            )

            os.makedirs(CACHE_DIR, exist_ok=True)

            def write_cache():
                latent_df.to_csv(cache_path, index=False, encoding="utf-8")

            self.with_retries(write_cache)
            file_size = os.path.getsize(cache_path) / 1024 / 1024
            if file_size > 1.0:
                miya_alerts(
                    f"Cache vol_regime size {file_size:.2f} MB exceeds 1 MB",
                    tag="LATENT",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(
                    f"Cache vol_regime size {file_size:.2f} MB exceeds 1 MB", priority=3
                )
                send_telegram_alert(
                    f"Cache vol_regime size {file_size:.2f} MB exceeds 1 MB"
                )

            self.save_to_market_memory(
                latent_df,
                "vol_regime",
                ["latent_vol_regime_vec_0", "latent_vol_regime_vec_1"],
            )

            os.makedirs(os.path.dirname(latent_vectors_path), exist_ok=True)

            def write_latent():
                latent_df.to_csv(latent_vectors_path, index=False, encoding="utf-8")

            self.with_retries(write_latent)

            self.validate_shap_features(
                ["latent_vol_regime_vec_0", "latent_vol_regime_vec_1"]
            )

            latency = time.time() - start_time
            self.log_performance(
                "encode_vol_regime",
                latency,
                success=True,
                num_rows=len(latent_df),
                confidence_drop_rate=confidence_drop_rate,
            )
            miya_speak(
                f"Vecteurs latents sauvegardés dans {latent_vectors_path}",
                tag="LATENT",
                voice_profile="calm",
                priority=1,
            )
            send_alert(
                f"Vecteurs latents sauvegardés dans {latent_vectors_path}", priority=1
            )
            send_telegram_alert(
                f"Vecteurs latents sauvegardés dans {latent_vectors_path}"
            )
            logger.info(f"Vecteurs latents sauvegardés dans {latent_vectors_path}")
            self.save_snapshot(
                "encode_vol_regime",
                {
                    "num_rows": len(latent_df),
                    "latent_vectors_path": latent_vectors_path,
                    "confidence_drop_rate": confidence_drop_rate,
                },
                compress=False,
            )
            return latent_df

        except Exception as e:
            latency = time.time() - start_time
            error_msg = (
                f"Erreur dans encode_vol_regime: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance(
                "encode_vol_regime", latency, success=False, error=str(e)
            )
            miya_alerts(error_msg, tag="LATENT", voice_profile="urgent", priority=4)
            send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            latent_df = pd.DataFrame(
                {
                    "timestamp": data["timestamp"],
                    "latent_vol_regime_vec_0": 0.0,
                    "latent_vol_regime_vec_1": 0.0,
                }
            )

            def write_latent_error():
                latent_df.to_csv(latent_vectors_path, index=False, encoding="utf-8")

            self.with_retries(write_latent_error)
            self.save_snapshot("encode_vol_regime", {"error": str(e)}, compress=False)
            return latent_df

    def encode_news_topics(
        self,
        data: pd.DataFrame,
        news_path: str = os.path.join(BASE_DIR, "data", "news", "news_data.csv"),
        news_topics_path: str = os.path.join(
            BASE_DIR, "data", "news", "news_topics.csv"
        ),
        n_topics: int = 3,
        config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml"),
    ) -> pd.DataFrame:
        """Encode les sujets des news en utilisant NLP (LDA) et sauvegarde les vecteurs de topics."""
        try:
            start_time = time.time()
            config = self.load_config(config_path)
            min_news = config.get("min_news", 10)

            required_cols = ["timestamp"]
            missing_cols = [col for col in required_cols if col not in data.columns]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )
            if len(data) < min_news:
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(required_cols) - len(missing_cols)}/{len(required_cols)} colonnes valides, {len(data)} lignes)"
                miya_alerts(alert_msg, tag="NLP", voice_profile="urgent", priority=3)
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            cache_path = os.path.join(
                CACHE_DIR,
                f"news_topics_{hashlib.sha256(data.to_json().encode()).hexdigest()}.csv",
            )
            if os.path.exists(cache_path):
                cached_data = pd.read_csv(cache_path)
                if len(cached_data) == len(data) and cached_data["timestamp"].equals(
                    data["timestamp"]
                ):
                    latency = time.time() - start_time
                    self.log_performance(
                        "encode_news_topics_cache",
                        latency,
                        success=True,
                        confidence_drop_rate=confidence_drop_rate,
                    )
                    miya_speak(
                        "News topics chargé depuis cache",
                        tag="NLP",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert("News topics chargé depuis cache", priority=1)
                    send_telegram_alert("News topics chargé depuis cache")
                    logger.info("News topics chargé depuis cache")
                    return cached_data

            if not os.path.exists(news_path):
                error_msg = f"Fichier {news_path} introuvable"
                miya_alerts(error_msg, tag="NLP", voice_profile="urgent", priority=4)
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            news_data = pd.read_csv(news_path)
            if len(news_data) < min_news:
                error_msg = f"Moins de {min_news} news disponibles dans {news_path}"
                miya_alerts(error_msg, tag="NLP", voice_profile="urgent", priority=4)
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                raise ValueError(error_msg)

            required_cols_news = ["timestamp", "news_content"]
            missing_cols_news = [
                col for col in required_cols_news if col not in news_data.columns
            ]
            for col in missing_cols_news:
                news_data[col] = "" if col == "news_content" else data["timestamp"]
                miya_speak(
                    f"Colonne '{col}' manquante dans news_data, imputée",
                    tag="NLP",
                    voice_profile="warning",
                    priority=2,
                )
                send_alert(
                    f"Colonne '{col}' manquante dans news_data, imputée", priority=2
                )
                send_telegram_alert(
                    f"Colonne '{col}' manquante dans news_data, imputée"
                )
                logger.warning(f"Colonne manquante dans news_data: {col}")

            news_data["timestamp"] = pd.to_datetime(
                news_data["timestamp"], errors="coerce"
            )
            if news_data["timestamp"].isna().any():
                error_msg = "NaN dans les timestamps de news_data"
                miya_alerts(error_msg, tag="NLP", voice_profile="urgent", priority=4)
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                raise ValueError(error_msg)

            valid_news = news_data["news_content"].str.strip().str.len() > 10
            if valid_news.sum() < min_news:
                error_msg = (
                    f"Moins de {min_news} news valides (contenu > 10 caractères)"
                )
                miya_alerts(error_msg, tag="NLP", voice_profile="urgent", priority=4)
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                raise ValueError(error_msg)

            vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
            doc_term_matrix = vectorizer.fit_transform(
                news_data["news_content"][valid_news]
            )

            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            topic_distributions = lda.fit_transform(doc_term_matrix)

            topic_cols = [f"topic_vector_news_{i}" for i in range(n_topics)]
            topics_df = pd.DataFrame(topic_distributions, columns=topic_cols)
            topics_df["timestamp"] = news_data[valid_news]["timestamp"].reset_index(
                drop=True
            )

            os.makedirs(CACHE_DIR, exist_ok=True)

            def write_cache():
                topics_df.to_csv(cache_path, index=False, encoding="utf-8")

            self.with_retries(write_cache)
            file_size = os.path.getsize(cache_path) / 1024 / 1024
            if file_size > 1.0:
                miya_alerts(
                    f"Cache news_topics size {file_size:.2f} MB exceeds 1 MB",
                    tag="NLP",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(
                    f"Cache news_topics size {file_size:.2f} MB exceeds 1 MB",
                    priority=3,
                )
                send_telegram_alert(
                    f"Cache news_topics size {file_size:.2f} MB exceeds 1 MB"
                )

            self.save_to_market_memory(topics_df, "news_topics", topic_cols)

            os.makedirs(os.path.dirname(news_topics_path), exist_ok=True)

            def write_topics():
                topics_df.to_csv(news_topics_path, index=False, encoding="utf-8")

            self.with_retries(write_topics)

            self.validate_shap_features(topic_cols)

            latency = time.time() - start_time
            self.log_performance(
                "encode_news_topics",
                latency,
                success=True,
                num_rows=len(topics_df),
                confidence_drop_rate=confidence_drop_rate,
            )
            miya_speak(
                f"Vecteurs de topics sauvegardés dans {news_topics_path}",
                tag="NLP",
                voice_profile="calm",
                priority=1,
            )
            send_alert(
                f"Vecteurs de topics sauvegardés dans {news_topics_path}", priority=1
            )
            send_telegram_alert(
                f"Vecteurs de topics sauvegardés dans {news_topics_path}"
            )
            logger.info(f"Vecteurs de topics sauvegardés dans {news_topics_path}")
            self.save_snapshot(
                "encode_news_topics",
                {
                    "num_rows": len(topics_df),
                    "news_topics_path": news_topics_path,
                    "confidence_drop_rate": confidence_drop_rate,
                },
                compress=False,
            )
            return topics_df

        except Exception as e:
            latency = time.time() - start_time
            error_msg = (
                f"Erreur dans encode_news_topics: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance(
                "encode_news_topics", latency, success=False, error=str(e)
            )
            miya_alerts(error_msg, tag="NLP", voice_profile="urgent", priority=4)
            send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            topic_cols = [f"topic_vector_news_{i}" for i in range(n_topics)]
            topics_df = pd.DataFrame(
                0.0, index=range(len(data)), columns=["timestamp"] + topic_cols
            )
            topics_df["timestamp"] = data["timestamp"]

            def write_topics_error():
                topics_df.to_csv(news_topics_path, index=False, encoding="utf-8")

            self.with_retries(write_topics_error)
            self.save_snapshot("encode_news_topics", {"error": str(e)}, compress=False)
            return topics_df

    def calculate_contextual_encodings(
        self,
        data: pd.DataFrame,
        news_path: str = os.path.join(BASE_DIR, "data", "news", "news_data.csv"),
        latent_vectors_path: str = os.path.join(
            BASE_DIR, "data", "latent", "latent_vectors.csv"
        ),
        news_topics_path: str = os.path.join(
            BASE_DIR, "data", "news", "news_topics.csv"
        ),
        n_topics: int = 3,
        config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml"),
    ) -> pd.DataFrame:
        """Calcule les composantes latentes (UMAP, NLP) et enrichit les données."""
        try:
            start_time = time.time()
            config = self.load_config(config_path)
            window_size = config.get("window_size", 10)

            required_cols = ["timestamp"]
            missing_cols = [col for col in required_cols if col not in data.columns]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )
            if len(data) < window_size:
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(required_cols) - len(missing_cols)}/{len(required_cols)} colonnes valides, {len(data)} lignes)"
                miya_alerts(
                    alert_msg, tag="ENCODING", voice_profile="urgent", priority=3
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            cache_path = os.path.join(
                CACHE_DIR,
                f"contextual_encodings_{hashlib.sha256(data.to_json().encode()).hexdigest()}.csv",
            )
            if os.path.exists(cache_path):
                cached_data = pd.read_csv(cache_path)
                if len(cached_data) == len(data) and cached_data["timestamp"].equals(
                    data["timestamp"]
                ):
                    latency = time.time() - start_time
                    self.log_performance(
                        "calculate_contextual_encodings_cache_hit",
                        latency,
                        success=True,
                        confidence_drop_rate=confidence_drop_rate,
                    )
                    miya_speak(
                        "Encodages contextuels chargés depuis cache",
                        tag="ENCODING",
                        voice_profile="calm",
                        priority=1,
                    )
                    send_alert("Encodages contextuels chargés depuis cache", priority=1)
                    send_telegram_alert("Encodages contextuels chargés depuis cache")
                    logger.info("Encodages contextuels chargés depuis cache")
                    return cached_data

            if data.empty:
                error_msg = "DataFrame vide dans calculate_contextual_encodings"
                miya_alerts(
                    error_msg, tag="ENCODING", voice_profile="urgent", priority=5
                )
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                raise ValueError(error_msg)

            data = data.copy()
            for col in missing_cols:
                default_start = pd.Timestamp.now()
                data[col] = pd.date_range(
                    start=default_start, periods=len(data), freq="1min"
                )
                miya_speak(
                    f"Colonne '{col}' manquante, création par défaut",
                    tag="ENCODING",
                    voice_profile="warning",
                    priority=2,
                )
                send_alert(
                    f"Colonne '{col}' manquante, création par défaut", priority=2
                )
                send_telegram_alert(f"Colonne '{col}' manquante, création par défaut")
                logger.warning(f"Colonne manquante: {col}")

            data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
            if data["timestamp"].isna().any():
                error_msg = "NaN dans les timestamps de data"
                miya_alerts(
                    error_msg, tag="ENCODING", voice_profile="urgent", priority=5
                )
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                raise ValueError(error_msg)
            if not data["timestamp"].is_monotonic_increasing:
                error_msg = "Timestamps non croissants dans data"
                miya_alerts(
                    error_msg, tag="ENCODING", voice_profile="urgent", priority=5
                )
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                raise ValueError(error_msg)

            expected_features = [
                "latent_vol_regime_vec_0",
                "latent_vol_regime_vec_1",
                "vol_regime_transition_prob",
                "news_topic_stability",
                "topic_vector_news_0",
                "topic_vector_news_1",
                "topic_vector_news_2",
            ]
            missing_features = [f for f in expected_features if f not in obs_t]
            if missing_features:
                miya_alerts(
                    f"Features attendues manquantes dans obs_t: {missing_features}",
                    tag="ENCODING",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(
                    f"Features attendues manquantes dans obs_t: {missing_features}",
                    priority=3,
                )
                send_telegram_alert(
                    f"Features attendues manquantes dans obs_t: {missing_features}"
                )
                logger.warning(
                    f"Features attendues manquantes dans obs_t: {missing_features}"
                )

            latent_df = self.encode_vol_regime(data, latent_vectors_path, config_path)
            topics_df = self.encode_news_topics(
                data, news_path, news_topics_path, n_topics, config_path
            )

            data = data.merge(latent_df, on="timestamp", how="left")
            data = data.merge(topics_df, on="timestamp", how="left")

            features = ["vol_regime_transition_prob", "news_topic_stability"] + [
                f"topic_vector_news_{i}" for i in range(n_topics)
            ]
            for feature in features:
                if feature not in data.columns:
                    data[feature] = 0.0

            vol_regime_vec = data[
                ["latent_vol_regime_vec_0", "latent_vol_regime_vec_1"]
            ]
            vol_regime_change = vol_regime_vec.diff().abs().sum(axis=1)
            data["vol_regime_transition_prob"] = vol_regime_change / (
                vol_regime_change.rolling(window=window_size, min_periods=1).mean()
                + 1e-6
            )

            topic_cols = [f"topic_vector_news_{i}" for i in range(n_topics)]
            topic_vec = data[topic_cols]
            topic_change = topic_vec.diff().abs().sum(axis=1)
            data["news_topic_stability"] = 1 / (
                topic_change.rolling(window=window_size, min_periods=1).mean() + 1e-6
            )

            for col in list(latent_df.columns) + list(topics_df.columns) + features:
                if col != "timestamp" and data[col].isna().any():
                    data[col] = data[col].fillna(0)

            os.makedirs(CACHE_DIR, exist_ok=True)

            def write_cache():
                data.to_csv(cache_path, index=False, encoding="utf-8")

            self.with_retries(write_cache)
            file_size = os.path.getsize(cache_path) / 1024 / 1024
            if file_size > 1.0:
                miya_alerts(
                    f"Cache contextual_encodings size {file_size:.2f} MB exceeds 1 MB",
                    tag="ENCODING",
                    voice_profile="urgent",
                    priority=3,
                )
                send_alert(
                    f"Cache contextual_encodings size {file_size:.2f} MB exceeds 1 MB",
                    priority=3,
                )
                send_telegram_alert(
                    f"Cache contextual_encodings size {file_size:.2f} MB exceeds 1 MB"
                )

            latency = time.time() - start_time
            self.log_performance(
                "calculate_contextual_encodings",
                latency,
                success=True,
                num_rows=len(data),
                confidence_drop_rate=confidence_drop_rate,
            )
            miya_speak(
                "Composantes latentes calculées : latent_vol_regime_vec, topic_vector_news et autres",
                tag="ENCODING",
                voice_profile="calm",
                priority=1,
            )
            send_alert("Composantes latentes calculées", priority=1)
            send_telegram_alert("Composantes latentes calculées")
            logger.info("Composantes latentes calculées")
            self.save_snapshot(
                "contextual_encodings",
                {
                    "num_rows": len(data),
                    "features": features,
                    "confidence_drop_rate": confidence_drop_rate,
                },
                compress=False,
            )
            return data

        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans calculate_contextual_encodings: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                "calculate_contextual_encodings", latency, success=False, error=str(e)
            )
            miya_alerts(error_msg, tag="ENCODING", voice_profile="urgent", priority=5)
            send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            features = [
                "latent_vol_regime_vec_0",
                "latent_vol_regime_vec_1",
                "vol_regime_transition_prob",
                "news_topic_stability",
            ]
            topic_cols = [f"topic_vector_news_{i}" for i in range(n_topics)]
            for col in features + topic_cols:
                data[col] = 0.0
            self.save_snapshot(
                "contextual_encodings", {"error": str(e)}, compress=False
            )
            return data


if __name__ == "__main__":
    try:
        encoder = ContextualStateEncoder()
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "atr_14": np.random.uniform(0.5, 2.0, 100),
                "volatility_trend": np.random.uniform(-0.1, 0.1, 100),
                "close": np.random.normal(5100, 10, 100),
            }
        )
        news_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "news_content": ["Market up due to tech rally"] * 20
                + ["Inflation fears rise"] * 20
                + ["Earnings season starts"] * 20
                + ["Fed rate decision looms"] * 20
                + ["Geopolitical tensions increase"] * 20,
            }
        )
        os.makedirs(os.path.join(BASE_DIR, "data", "news"), exist_ok=True)
        news_data.to_csv(
            os.path.join(BASE_DIR, "data", "news", "news_data.csv"), index=False
        )
        result = encoder.calculate_contextual_encodings(data, n_topics=3)
        print(
            result[
                [
                    "timestamp",
                    "latent_vol_regime_vec_0",
                    "latent_vol_regime_vec_1",
                    "topic_vector_news_0",
                    "vol_regime_transition_prob",
                ]
            ].head()
        )
        miya_speak(
            "Test calculate_contextual_encodings terminé",
            tag="TEST",
            voice_profile="calm",
            priority=1,
        )
        send_alert("Test calculate_contextual_encodings terminé", priority=1)
        send_telegram_alert("Test calculate_contextual_encodings terminé")
        logger.info("Test calculate_contextual_encodings terminé")
    except Exception as e:
        miya_alerts(
            f"Erreur test: {str(e)}\n{traceback.format_exc()}",
            tag="ALERT",
            voice_profile="urgent",
            priority=5,
        )
        send_alert(f"Erreur test: {str(e)}", priority=5)
        send_telegram_alert(f"Erreur test: {str(e)}")
        logger.error(f"Erreur test: {str(e)}")
        raise
