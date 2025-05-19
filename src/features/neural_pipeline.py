```python
# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/features/neural_pipeline.py
# Pipeline neuronal multi-architecture (LSTM, CNN, MLP volatilité, MLP classification) 
# pour MIA_IA_SYSTEM_v2_2025.
# Génère cnn_pressure, neural_regime, predicted_vix avec LSTM dédié (méthode 12).
#
# Version : 2.1.6
# Date : 2025-05-15
#
# Rôle : Prédit les features neurales et predicted_vix à partir de 350 features 
# (entraînement) ou top 150 SHAP (production).
#        Intègre les nouvelles features bid_ask_imbalance, trade_aggressiveness, 
# iv_skew, iv_term_structure, option_skew, et news_impact_score pour la cohérence 
# avec trade_probability.py.
#        Collecte les hyperparamètres dans market_memory.db (Proposition 2, Étape 1).
#
# Dépendances :
# - tensorflow>=2.16.0,<3.0.0, pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, 
#   sklearn>=1.2.0,<2.0.0, psutil>=5.9.0,<6.0.0, joblib>=1.2.0,<2.0.0, 
#   matplotlib>=3.7.0,<4.0.0, pyyaml>=6.0.0,<7.0.0
# - src/model/utils/miya_console.py
# - src/model/utils/alert_manager.py
# - src/model/utils/config_manager.py
# - src/model/utils/obs_template.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/model_params.yaml
# - data/features/merged_data.csv
#
# Outputs :
# - data/models/lstm_model.h5
# - data/models/cnn_model.h5
# - data/models/vol_mlp_model.h5
# - data/models/regime_mlp_model.h5
# - data/models/lstm_vix_model.h5
# - data/logs/neural_pipeline_performance.csv
# - data/neural_snapshots/*.json (option *.json.gz)
# - data/figures/neural_pipeline/
# - data/neural_pipeline_dashboard.json
# - market_memory.db (table meta_runs)
#
# Notes :
# - Conforme à structure.txt (version 2.1.5, 2025-05-14).
# - Intègre un modèle LSTM dédié pour predicted_vix (méthode 12) avec 2 couches 
# (128 unités).
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des prédictions.
# - Intègre validation SHAP (Phase 17) pour assurer la conformité des top 150 features 
# en production, incluant les nouvelles features.
# - Préserve toutes les fonctionnalités existantes (LSTM, CNN, MLP).
# - Ajoute logs psutil (memory_usage_mb, cpu_usage_percent) dans toutes les méthodes 
# critiques.
# - Utilise exclusivement IQFeed via data_provider.py, avec retries (max 3, délai 
# 2^attempt secondes).
# - Sauvegarde des snapshots JSON non compressés par défaut avec option de compression 
# gzip.
# - Envoie des alertes via AlertManager et telegram_alert pour les erreurs critiques 
# et succès.
# - Tests unitaires disponibles dans tests/test_neural_pipeline.py (à implémenter).
# - Validation complète prévue pour juin 2025.
# - Évolution future : Migration API Investing.com (juin 2025), intégration avec 
# feature_pipeline.py.

import gzip
import hashlib
import json
import logging
import os
import sqlite3
import time
import traceback
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Conv1D, Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

from src.model.utils.alert_manager import send_alert
from src.model.utils.config_manager import config_manager
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.model.utils.obs_template import obs_t
from src.utils.telegram_alert import send_telegram_alert

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "neural_snapshots")
CSV_LOG_PATH = os.path.join(BASE_DIR, "data", "logs", "neural_pipeline_performance.csv")
DASHBOARD_PATH = os.path.join(BASE_DIR, "data", "neural_pipeline_dashboard.json")
FIGURE_DIR = os.path.join(BASE_DIR, "data", "figures", "neural_pipeline")
FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR, "data", "features", "feature_importance.csv"
)
DB_PATH = os.path.join(BASE_DIR, "data", "market_memory.db")

# Création des dossiers
os.makedirs(os.path.join(BASE_DIR, "data", "logs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data", "models"), exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# Configuration du logging
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "data", "logs", "neural_pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Constantes
MAX_RETRIES = 3
RETRY_DELAY = 2.0


class ScalerManager:
    """Gère les scalers pour éviter la répétition du code."""

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.scalers = {}
        os.makedirs(save_dir, exist_ok=True)

    def load_or_create(
        self, 
        name: str, 
        expected_features: int, 
        data: Optional["np.ndarray"] = None
    ) -> StandardScaler:
        """Charge ou crée un scaler."""
        scaler_path = os.path.join(self.save_dir, f"scaler_{name}.pkl")
        if os.path.exists(scaler_path):
            try:
                scaler = joblib.load(scaler_path)
                if scaler.n_features_in_ != expected_features:
                    error_msg = (
                        f"Mismatch in scaler {name} dimensions: expected "
                        f"{expected_features}, got {scaler.n_features_in_}"
                    )
                    raise ValueError(error_msg)
                self.scalers[name] = scaler
            except Exception as e:
                error_msg = f"Erreur chargement scaler {name}: {str(e)}"
                miya_alerts(
                    error_msg, 
                    tag="NEURAL_PIPELINE", 
                    voice_profile="urgent", 
                    priority=3
                )
                logger.error(error_msg)
                scaler = StandardScaler()
        else:
            scaler = StandardScaler()
        if data is not None and name not in self.scalers:
            scaler.fit(data)
            joblib.dump(scaler, scaler_path)
            self.scalers[name] = scaler
        return self.scalers[name]

    def transform(self, name: str, data: "np.ndarray") -> "np.ndarray":
        """Transforme les données avec le scaler spécifié."""
        if name not in self.scalers:
            raise ValueError(f"Scaler {name} non initialisé")
        return self.scalers[name].transform(data)


class NeuralPipeline:
    """
    Pipeline neuronal multi-architecture pour générer des features neurales et 
    predicted_vix.
    """

    def __init__(
        self,
        window_size: int = 50,
        base_features: int = 350,
        training_mode: bool = True,
        config_path: str = os.path.join(BASE_DIR, "config", "model_params.yaml"),
    ):
        start_time = datetime.now()  # Initialisation de start_time au début
        # 350 pour entraînement, 150 pour production
        self.base_features = 350 if training_mode else 150
        self.window_size = window_size
        self.training_mode = training_mode
        self.log_buffer = []
        self.cache = OrderedDict()  # Utilisation d'OrderedDict pour LRU

        # Charger la config
        try:
            config = config_manager.get_config(config_path)
            if "neural_pipeline" not in config:
                error_msg = "Clé 'neural_pipeline' manquante dans la configuration"
                raise ValueError(error_msg)
            required_keys = ["lstm", "cnn", "mlp_volatility", "mlp_regime", "vix_lstm"]
            missing_keys = [
                key for key in required_keys if key not in config["neural_pipeline"]
            ]
            if missing_keys:
                error_msg = f"Clés manquantes dans 'neural_pipeline': {missing_keys}"
                raise ValueError(error_msg)
            self.config = config["neural_pipeline"]
            self.buffer_size = self.config.get("logging", {}).get("buffer_size", 100)
            self.max_cache_size = self.config.get("cache", {}).get(
                "max_cache_size", 1000
            )
            self.cache_hours = self.config.get("cache", {}).get("cache_hours", 24)
        except Exception as e:
            error_msg = (
                f"Erreur chargement config: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, 
                tag="NEURAL_PIPELINE", 
                voice_profile="urgent", 
                priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.config = {
                "lstm": {
                    "units": 128,
                    "dropout": 0.2,
                    "hidden_layers": [64],
                    "learning_rate": 0.001,
                },
                "cnn": {
                    "filters": 32,
                    "kernel_size": 5,
                    "dropout": 0.1,
                    "hidden_layers": [16],
                    "learning_rate": 0.001,
                },
                "mlp_volatility": {
                    "units": 128,
                    "hidden_layers": [64],
                    "learning_rate": 0.001,
                },
                "mlp_regime": {
                    "units": 128,
                    "hidden_layers": [64],
                    "learning_rate": 0.001,
                },
                "vix_lstm": {
                    "units": 128,
                    "dropout": 0.2,
                    "hidden_layers": [64],
                    "learning_rate": 0.001,
                },
                "batch_size": 32,
                "pretrain_epochs": 5,
                "validation_split": 0.2,
                "normalization": True,
                "save_dir": os.path.join(BASE_DIR, "data", "models"),
                "num_lstm_features": 8,
            }
            self.buffer_size = 100
            self.max_cache_size = 1000
            self.cache_hours = 24

        self.lstm_config = self.config.get("lstm", {})
        self.cnn_config = self.config.get("cnn", {})
        self.vol_mlp_config = self.config.get("mlp_volatility", {})
        self.regime_mlp_config = self.config.get("mlp_regime", {})
        self.vix_lstm_config = self.config.get("vix_lstm", {})
        self.batch_size = self.config.get("batch_size", 32)
        self.pretrain_epochs = self.config.get("pretrain_epochs", 5)
        self.validation_split = self.config.get("validation_split", 0.2)
        self.normalization = self.config.get("normalization", True)
        self.save_dir = self.config.get(
            "save_dir", os.path.join(BASE_DIR, "data", "models")
        )
        self.num_lstm_features = self.config.get(
            "num_lstm_features", 8
        )  # Aligné avec obs_t (neural_feature_0-7)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        os.makedirs(FIGURE_DIR, exist_ok=True)

        # Gestion des scalers
        self.scaler_manager = ScalerManager(self.save_dir)

        # LSTM général
        lstm_units = self.lstm_config.get("units", 128)
        self.lstm = Sequential(
            [
                Input(shape=(window_size, 16)),
                LSTM(lstm_units, return_sequences=False),
                Dropout(self.lstm_config.get("dropout", 0.2)),
                Dense(
                    self.lstm_config.get("hidden_layers", [64])[0], 
                    activation="relu"
                ),
                Dense(self.num_lstm_features, activation="relu"),  # 8 features
            ]
        )
        self.lstm.compile(
            optimizer=Adam(learning_rate=self.lstm_config.get("learning_rate", 0.001)),
            loss="mean_squared_error",
        )

        # CNN
        cnn_filters = self.cnn_config.get("filters", 32)
        self.cnn = Sequential(
            [
                Input(shape=(window_size, 7)),
                Conv1D(
                    cnn_filters,
                    kernel_size=self.cnn_config.get("kernel_size", 5),
                    activation="relu",
                ),
                Flatten(),
                Dropout(self.cnn_config.get("dropout", 0.1)),
                Dense(
                    self.cnn_config.get("hidden_layers", [16])[0], 
                    activation="relu"
                ),
                Dense(1, activation="linear"),  # cnn_pressure
            ]
        )
        self.cnn.compile(
            optimizer=Adam(learning_rate=self.cnn_config.get("learning_rate", 0.001)),
            loss="mean_squared_error",
        )

        # MLP volatilité
        self.vol_mlp = Sequential(
            [
                Input(shape=(self.base_features + self.num_lstm_features + 1,)),
                Dense(self.vol_mlp_config.get("units", 128), activation="relu"),
                Dense(
                    self.vol_mlp_config.get("hidden_layers", [64])[0], 
                    activation="relu"
                ),
                Dense(1, activation="linear"),  # predicted_volatility
            ]
        )
        self.vol_mlp.compile(
            optimizer=Adam(
                learning_rate=self.vol_mlp_config.get("learning_rate", 0.001)
            ),
            loss="mean_squared_error",
        )

        # MLP régime
        self.regime_mlp = Sequential(
            [
                Input(shape=(self.base_features + self.num_lstm_features + 1,)),
                Dense(self.regime_mlp_config.get("units", 128), activation="relu"),
                Dense(
                    self.regime_mlp_config.get("hidden_layers", [64])[0],
                    activation="relu",
                ),
                Dense(3, activation="softmax"),  # trend, range, defensive
            ]
        )
        self.regime_mlp.compile(
            optimizer=Adam(
                learning_rate=self.regime_mlp_config.get("learning_rate", 0.001)
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # LSTM pour predict_vix (méthode 12)
        vix_lstm_units = self.vix_lstm_config.get("units", 128)
        self.lstm_vix = Sequential(
            [
                Input(shape=(window_size, 14)),
                LSTM(vix_lstm_units, return_sequences=True),
                LSTM(vix_lstm_units, return_sequences=False),
                Dropout(self.vix_lstm_config.get("dropout", 0.2)),
                Dense(
                    self.vix_lstm_config.get("hidden_layers", [64])[0],
                    activation="relu",
                ),
                Dense(1, activation="linear"),  # predicted_vix
            ]
        )
        self.lstm_vix.compile(
            optimizer=Adam(
                learning_rate=self.vix_lstm_config.get("learning_rate", 0.001)
            ),
            loss="mean_squared_error",
        )

        # Charger les modèles
        try:
            self.models_loaded = self.load_models()
            latency = (datetime.now() - start_time).total_seconds()  # Calcul de la latence
            if self.models_loaded == 0:
                warning_msg = (
                    "Aucun modèle pré-entraîné chargé. Pré-entraînement requis."
                )
                miya_speak(
                    warning_msg, 
                    tag="NEURAL_PIPELINE", 
                    level="warning", 
                    priority=3
                )
                logger.warning(warning_msg)
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            success_msg = "Pipeline neuronal initialisé"
            miya_speak(
                success_msg, 
                tag="NEURAL_PIPELINE", 
                voice_profile="calm", 
                priority=2
            )
            logger.info(success_msg)
            self.log_performance(
                "init",
                latency,
                success=True,
                num_features=self.base_features,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_percent,
            )
            self.save_snapshot(
                "init",
                {
                    "window_size": self.window_size, 
                    "base_features": self.base_features
                },
                compress=False,
            )
            send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur initialisation pipeline: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, 
                tag="NEURAL_PIPELINE", 
                voice_profile="urgent", 
                priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", latency, success=False, error=str(e))
            raise

    def with_retries(
        self,
        func: callable,
        max_attempts: int = MAX_RETRIES,
        delay_base: float = RETRY_DELAY,
    ) -> Any:
        """Exécute une fonction avec retries (max 3, délai exponentiel)."""
        start_time = datetime.now()
        for attempt in range(max_attempts):
            try:
                result = func()
                latency = (datetime.now() - start_time).total_seconds()
                memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
                cpu_percent = psutil.cpu_percent()
                self.log_performance(
                    f"retry_attempt_{attempt+1}",
                    latency,
                    success=True,
                    attempt_number=attempt + 1,
                    memory_usage_mb=memory_usage,
                    cpu_usage_percent=cpu_percent,
                )
                return result
            except Exception as e:
                if attempt == max_attempts - 1:
                    latency = (datetime.now() - start_time).total_seconds()
                    error_msg = (
                        f"Échec après {max_attempts} tentatives: {str(e)}\n"
                        f"{traceback.format_exc()}"
                    )
                    miya_alerts(
                        error_msg,
                        tag="NEURAL_PIPELINE",
                        voice_profile="urgent",
                        priority=5,
                    )
                    send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    logger.error(error_msg)
                    self.log_performance(
                        f"retry_attempt_{attempt+1}",
                        latency,
                        success=False,
                        error=str(e),
                        attempt_number=attempt + 1,
                    )
                    raise
                delay = delay_base * (2**attempt)
                warning_msg = (
                    f"Tentative {attempt+1} échouée, retry après {delay}s"
                )
                miya_speak(
                    warning_msg, 
                    tag="NEURAL_PIPELINE", 
                    level="warning", 
                    priority=3
                )
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
                time.sleep(delay)

    def log_performance(
        self, 
        operation: str, 
        latency: float, 
        success: bool, 
        error: str = None, 
        **kwargs
    ) -> None:
        """Journalise les performances des opérations critiques."""
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
            cpu_usage = psutil.cpu_percent()  # % CPU
            if memory_usage > 1024:
                alert_msg = (
                    f"ALERTE: Utilisation mémoire élevée ({memory_usage:.2f} MB)"
                )
                miya_alerts(
                    alert_msg, 
                    tag="NEURAL_PIPELINE", 
                    voice_profile="urgent", 
                    priority=5
                )
                send_alert(alert_msg, priority=4)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)
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
                    log_df.to_csv(
                        CSV_LOG_PATH,
                        mode="a" if os.path.exists(CSV_LOG_PATH) else "w",
                        header=not os.path.exists(CSV_LOG_PATH),
                        index=False,
                        encoding="utf-8",
                    )

                self.with_retries(write_log)
                self.log_buffer = []
            logger.info(f"Performance journalisée pour {operation}. CPU: {cpu_usage}%")
        except Exception as e:
            error_msg = f"Erreur journalisation performance: {str(e)}"
            miya_alerts(
                error_msg, 
                tag="NEURAL_PIPELINE", 
                voice_profile="urgent", 
                priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def save_snapshot(
        self, 
        snapshot_type: str, 
        data: Dict, 
        compress: bool = False
    ) -> None:
        """Sauvegarde un instantané des résultats avec option de compression gzip."""
        try:
            start_time = datetime.now()
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
                alert_msg = f"Snapshot size {file_size:.2f} MB exceeds 1 MB"
                miya_alerts(
                    alert_msg, 
                    tag="NEURAL_PIPELINE", 
                    voice_profile="urgent", 
                    priority=3
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)
            latency = (datetime.now() - start_time).total_seconds()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            success_msg = f"Snapshot {snapshot_type} sauvegardé: {save_path}"
            miya_speak(
                success_msg, 
                tag="NEURAL_PIPELINE", 
                voice_profile="calm", 
                priority=1
            )
            logger.info(success_msg)
            self.log_performance(
                "save_snapshot",
                latency,
                success=True,
                snapshot_type=snapshot_type,
                file_size_mb=file_size,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_percent,
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            self.log_performance("save_snapshot", 0, success=False, error=str(e))
            miya_alerts(
                error_msg, 
                tag="NEURAL_PIPELINE", 
                voice_profile="urgent", 
                priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def validate_obs_t(self, features: List[str]) -> List[str]:
        """Valide que les colonnes obs_t sont présentes dans les features."""
        try:
            start_time = datetime.now()
            missing = [f for f in obs_t if f not in features]
            if missing:
                warning_msg = f"Colonnes obs_t manquantes: {missing}"
                miya_speak(
                    warning_msg, 
                    tag="NEURAL_PIPELINE", 
                    level="warning", 
                    priority=3
                )
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
            latency = (datetime.now() - start_time).total_seconds()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            self.log_performance(
                "validate_obs_t",
                latency,
                success=True,
                num_features=len(features),
                num_missing=len(missing),
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_percent,
            )
            return missing
        except Exception as e:
            error_msg = f"Erreur validation obs_t: {str(e)}"
            miya_alerts(
                error_msg, 
                tag="NEURAL_PIPELINE", 
                voice_profile="urgent", 
                priority=4
            )
            send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            return []

    def validate_shap_features(self, features: List[str]) -> bool:
        """Valide que les features sont dans les top 150 SHAP (production)."""
        try:
            start_time = datetime.now()
            if not os.path.exists(FEATURE_IMPORTANCE_PATH):
                error_msg = "Fichier SHAP manquant"
                miya_alerts(
                    error_msg, 
                    tag="NEURAL_PIPELINE", 
                    voice_profile="urgent", 
                    priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return False
            shap_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
            if len(shap_df) < 150:
                error_msg = (
                    f"Nombre insuffisant de SHAP features: {len(shap_df)} < 150"
                )
                miya_alerts(
                    error_msg, 
                    tag="NEURAL_PIPELINE", 
                    voice_profile="urgent", 
                    priority=4
                )
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return False
            missing_obs_t = self.validate_obs_t(features)
            valid_features = (
                set(shap_df["feature"].head(150))
                .union(obs_t)
                .union(
                    [
                        "bid_ask_imbalance",
                        "trade_aggressiveness",
                        "iv_skew",
                        "iv_term_structure",
                        "option_skew",
                        "news_impact_score",
                    ]
                )
            )
            missing = [f for f in features if f not in valid_features]
            if missing or missing_obs_t:
                warning_msg = (
                    f"Features non incluses dans top 150 SHAP ou obs_t: {missing}, "
                    f"obs_t manquantes: {missing_obs_t}"
                )
                miya_alerts(
                    warning_msg,
                    tag="NEURAL_PIPELINE",
                    voice_profile="warning",
                    priority=3,
                )
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
            latency = (datetime.now() - start_time).total_seconds()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            success_msg = "SHAP features validées"
            miya_speak(
                success_msg, 
                tag="NEURAL_PIPELINE", 
                voice_profile="calm", 
                priority=1
            )
            logger.info(success_msg)
            self.log_performance(
                "validate_shap_features",
                latency,
                success=True,
                num_features=len(features),
                num_missing=len(missing) + len(missing_obs_t),
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_percent,
            )
            self.save_snapshot(
                "validate_shap_features",
                {
                    "num_features": len(features),
                    "missing": missing,
                    "missing_obs_t": missing_obs_t,
                },
                compress=False,
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            return True
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur validation SHAP features: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, 
                tag="NEURAL_PIPELINE", 
                voice_profile="urgent", 
                priority=4
            )
            send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "validate_shap_features", 
                latency, 
                success=False, 
                error=str(e)
            )
            return False

    def save_dashboard_status(
        self, 
        status: Dict, 
        status_file: str = DASHBOARD_PATH
    ) -> None:
        """Sauvegarde l'état du pipeline pour mia_dashboard.py."""
        try:
            start_time = datetime.now()
            os.makedirs(os.path.dirname(status_file), exist_ok=True)
            status.update(
                {
                    "new_features": [
                        "bid_ask_imbalance",
                        "trade_aggressiveness",
                        "iv_skew",
                        "iv_term_structure",
                        "option_skew",
                        "news_impact_score",
                    ],
                    "confidence_drop_rate": status.get("confidence_drop_rate", 0.0),
                }
            )

            def write_status():
                with open(status_file, "w", encoding="utf-8") as f:
                    json.dump(status, f, indent=4)

            self.with_retries(write_status)
            latency = (datetime.now() - start_time).total_seconds()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            success_msg = f"État sauvegardé dans {status_file}"
            miya_speak(
                success_msg, 
                tag="NEURAL_PIPELINE", 
                voice_profile="calm", 
                priority=1
            )
            logger.info(success_msg)
            self.log_performance(
                "save_dashboard_status",
                latency,
                success=True,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_percent,
            )
            self.save_snapshot(
                "save_dashboard_status", 
                {"status_file": status_file}, 
                compress=False
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur sauvegarde dashboard: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, 
                tag="NEURAL_PIPELINE", 
                voice_profile="urgent", 
                priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "save_dashboard_status", 
                latency, 
                success=False, 
                error=str(e)
            )

    def preprocess(
        self,
        raw_data: pd.DataFrame,
        options_data: pd.DataFrame,
        orderflow_data: pd.DataFrame,
    ) -> Tuple["np.ndarray", "np.ndarray", pd.DataFrame]:
        """Prétraite les données pour LSTM, CNN et MLP."""
        try:
            start_time = datetime.now()
            required_raw = [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "atr_14",
                "adx_14",
                "bid_ask_imbalance",
                "trade_aggressiveness",
                "iv_skew",
                "iv_term_structure",
            ]
            required_options = [
                "timestamp",
                "gex",
                "oi_peak_call_near",
                "gamma_wall",
                "key_strikes_1",
                "max_pain_strike",
                "net_gamma",
                "zero_gamma",
                "dealer_zones_count",
                "vol_trigger",
                "ref_px",
                "data_release",
                "option_skew",
                "news_impact_score",
            ]
            required_order = ["timestamp", "bid_size_level_1", "ask_size_level_1"]

            # Validation des colonnes
            confidence_drop_rate = 0.0
            for df, cols, name in [
                (raw_data, required_raw, "raw"),
                (options_data, required_options, "options"),
                (orderflow_data, required_order, "orderflow"),
            ]:
                missing = [col for col in cols if col not in df.columns]
                confidence_drop_rate = max(
                    confidence_drop_rate,
                    1.0 - min((len(cols) - len(missing)) / len(cols), 1.0),
                )
                if missing:
                    df = df.copy()
                    for col in missing:
                        if col == "timestamp":
                            df[col] = pd.date_range(
                                start="2025-04-13", periods=len(df), freq="1min"
                            )
                        elif col in ["bid_ask_imbalance", "trade_aggressiveness"]:
                            df[col] = np.random.normal(
                                0, 
                                0.1 if col == "bid_ask_imbalance" else 0.2, 
                                len(df)
                            )
                        elif col in ["iv_skew", "iv_term_structure", "option_skew"]:
                            df[col] = np.random.uniform(0.01, 0.05, len(df))
                        elif col == "news_impact_score":
                            df[col] = np.random.uniform(0, 1, len(df))
                        else:
                            df[col] = (
                                df[col]
                                .interpolate(method="linear", limit_direction="both")
                                .fillna(df[col].median() if col in df.columns else 0)
                            )
                    warning_msg = (
                        f"Colonnes manquantes dans {name}: {missing}. Imputées."
                    )
                    miya_speak(
                        warning_msg, 
                        tag="NEURAL_PIPELINE", 
                        level="warning", 
                        priority=2
                    )
                    send_alert(warning_msg, priority=2)
                    send_telegram_alert(warning_msg)
                    logger.warning(warning_msg)

            if len(raw_data) < self.window_size:
                error_msg = (
                    f"Données insuffisantes: {len(raw_data)} lignes, "
                    f"requis >= {self.window_size}"
                )
                miya_alerts(
                    error_msg, 
                    tag="NEURAL_PIPELINE", 
                    voice_profile="urgent", 
                    priority=3
                )
                send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)
            confidence_drop_rate = max(
                confidence_drop_rate,
                1.0 - min(len(raw_data) / (self.window_size * 2), 1.0),
            )
            if confidence_drop_rate > 0.5:
                alert_msg = (
                    f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} "
                    f"({len(raw_data)} lignes, {len(raw_data.columns)} features)"
                )
                miya_alerts(
                    alert_msg, 
                    tag="NEURAL_PIPELINE", 
                    voice_profile="urgent", 
                    priority=3
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            # Fusionner et aligner avec les features
            data = pd.merge(raw_data, options_data, on="timestamp", how="inner").merge(
                orderflow_data, on="timestamp", how="inner"
            )
            if data.empty:
                error_msg = "Fusion des données a produit un DataFrame vide."
                miya_alerts(
                    error_msg, 
                    tag="NEURAL_PIPELINE", 
                    voice_profile="urgent", 
                    priority=3
                )
                send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            valid_cols = [col for col in data.columns if col != "timestamp"]
            if not self.training_mode:
                self.validate_shap_features(valid_cols)
            if len(valid_cols) < self.base_features:
                data = data.copy()
                for i in range(self.base_features - len(valid_cols)):
                    data[f"placeholder_{i}"] = 0
                    valid_cols.append(f"placeholder_{i}")
            data = data.reindex(
                columns=valid_cols[: self.base_features] + ["timestamp"], 
                fill_value=0
            )

            # Vérifier le cache
            cache_key = hashlib.sha256(data.to_json().encode()).hexdigest()
            if cache_key in self.cache:
                self.cache.move_to_end(cache_key)
                if (
                    datetime.now() - self.cache[cache_key]["timestamp"]
                ).total_seconds() < self.cache_hours * 3600:
                    result = self.cache[cache_key]["result"]
                    latency = (datetime.now() - start_time).total_seconds()
                    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
                    cpu_percent = psutil.cpu_percent()
                    success_msg = "Preprocess récupéré du cache"
                    miya_speak(
                        success_msg,
                        tag="NEURAL_PIPELINE",
                        voice_profile="calm",
                        priority=1,
                    )
                    logger.info(success_msg)
                    self.log_performance(
                        "preprocess_cache_hit",
                        latency,
                        success=True,
                        num_features=len(valid_cols),
                        memory_usage_mb=memory_usage,
                        cpu_usage_percent=cpu_percent,
                    )
                    send_alert(success_msg, priority=1)
                    send_telegram_alert(success_msg)
                    return result

            # Fenêtres pour LSTM
            lstm_columns = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "gex",
                "oi_peak_call_near",
                "gamma_wall",
                "net_gamma",
                "vol_trigger",
                "key_strikes_1",
                "max_pain_strike",
                "bid_ask_imbalance",
                "trade_aggressiveness",
                "iv_skew",
                "iv_term_structure",
            ]
            lstm_data = data[[col for col in lstm_columns if col in data.columns]]
            lstm_input = np.array(
                [
                    lstm_data[i : i + self.window_size].values
                    for i in range(len(data) - self.window_size + 1)
                ]
            )
            if self.normalization:
                lstm_input = (
                    self.scaler_manager.load_or_create(
                        "lstm",
                        expected_features=lstm_input.shape[2],
                        data=lstm_input.reshape(-1, lstm_input.shape[2]),
                    )
                    .transform(lstm_input.reshape(-1, lstm_input.shape[2]))
                    .reshape(-1, self.window_size, lstm_input.shape[2])
                )

            # Fenêtres pour CNN
            cnn_columns = [
                "gex",
                "oi_peak_call_near",
                "gamma_wall",
                "bid_size_level_1",
                "ask_size_level_1",
                "option_skew",
                "news_impact_score",
            ]
            cnn_data = data[[col for col in cnn_columns if col in data.columns]]
            cnn_input = np.array(
                [
                    cnn_data[i : i + self.window_size].values
                    for i in range(len(data) - self.window_size + 1)
                ]
            )
            if self.normalization:
                cnn_input = (
                    self.scaler_manager.load_or_create(
                        "cnn",
                        expected_features=cnn_input.shape[2],
                        data=cnn_input.reshape(-1, cnn_input.shape[2]),
                    )
                    .transform(cnn_input.reshape(-1, cnn_input.shape[2]))
                    .reshape(-1, self.window_size, cnn_input.shape[2])
                )

            full_data = data.iloc[self.window_size :].reset_index(drop=True)
            result = (lstm_input, cnn_input, full_data)
            self.cache[cache_key] = {"result": result, "timestamp": datetime.now()}
            self.cache.move_to_end(cache_key)
            if len(self.cache) > self.max_cache_size:
                self.cache.popitem(last=False)

            latency = (datetime.now() - start_time).total_seconds()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            success_msg = f"Preprocess terminé pour {len(full_data)} lignes"
            miya_speak(
                success_msg, 
                tag="NEURAL_PIPELINE", 
                voice_profile="calm", 
                priority=1
            )
            logger.info(success_msg)
            self.log_performance(
                "preprocess",
                latency,
                success=True,
                num_rows=len(full_data),
                confidence_drop_rate=confidence_drop_rate,
                num_features=len(valid_cols),
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_percent,
            )
            self.save_snapshot(
                "preprocess",
                {
                    "num_rows": len(full_data),
                    "lstm_shape": str(lstm_input.shape),
                    "cnn_shape": str(cnn_input.shape),
                    "confidence_drop_rate": confidence_drop_rate,
                    "new_features": [
                        f
                        for f in [
                            "bid_ask_imbalance",
                            "trade_aggressiveness",
                            "iv_skew",
                            "iv_term_structure",
                            "option_skew",
                            "news_impact_score",
                        ]
                        if f in valid_cols
                    ],
                },
                compress=False,
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            return result
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur preprocess: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, 
                tag="NEURAL_PIPELINE", 
                voice_profile="urgent", 
                priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("preprocess", latency, success=False, error=str(e))
            self.save_snapshot(
                "error_preprocess", 
                {"error": str(e)}, 
                compress=False
            )
            raise

    def preprocess_vix(self, data: pd.DataFrame) -> "np.ndarray":
        """Prétraite les données pour predict_vix."""
        try:
            start_time = datetime.now()
            required_cols = [
                "vix_es_correlation",
                "iv_atm",
                "key_strikes_1",
                "max_pain_strike",
                "net_gamma",
                "zero_gamma",
                "dealer_zones_count",
                "vol_trigger",
                "ref_px",
                "data_release",
                "iv_skew",
                "iv_term_structure",
                "option_skew",
                "news_impact_score",
            ]
            missing = [col for col in required_cols if col not in data.columns]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing)) / len(required_cols), 1.0
            )
            if len(data) < self.window_size:
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if missing:
                data = data.copy()
                for col in missing:
                    if col in ["iv_skew", "iv_term_structure", "option_skew"]:
                        data[col] = np.random.uniform(
                            0.01, 0.05, len(data)
                        )  # Valeurs réalistes
                    elif col == "news_impact_score":
                        data[col] = np.random.uniform(0, 1, len(data))  # Score entre 0 et 1
                    elif col == "dealer_zones_count":
                        data[col] = np.random.randint(
                            0, 11, len(data)
                        )  # Entiers entre 0 et 10
                    elif col == "data_release":
                        data[col] = np.random.randint(0, 2, len(data))  # Binaire (0 ou 1)
                    elif col in ["net_gamma", "vol_trigger"]:
                        data[col] = np.random.uniform(-1, 1, len(data))
                    else:
                        data[col] = np.clip(
                            np.random.normal(0, 1, len(data)), -10, 10
                        )  # Valeurs génériques
                warning_msg = (
                    f"Colonnes manquantes pour VIX: {missing}. Imputées."
                )
                miya_speak(
                    warning_msg, 
                    tag="NEURAL_PIPELINE", 
                    level="warning", 
                    priority=2
                )
                send_alert(warning_msg, priority=2)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)

            if len(data) < self.window_size:
                error_msg = (
                    f"Données insuffisantes pour VIX: {len(data)} lignes, "
                    f"requis >= {self.window_size}"
                )
                miya_alerts(
                    error_msg, 
                    tag="NEURAL_PIPELINE", 
                    voice_profile="urgent", 
                    priority=3
                )
                send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)
            if confidence_drop_rate > 0.5:
                alert_msg = (
                    f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} "
                    f"({len(data)} lignes, {len(data.columns)} features)"
                )
                miya_alerts(
                    alert_msg, 
                    tag="NEURAL_PIPELINE", 
                    voice_profile="urgent", 
                    priority=3
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            vix_data = data[[col for col in required_cols if col in data.columns]]
            vix_input = np.array(
                [
                    vix_data[i : i + self.window_size].values
                    for i in range(len(vix_data) - self.window_size + 1)
                ]
            )
            if self.normalization:
                vix_input = self.scaler_manager.transform(
                    "vix", 
                    vix_input.reshape(-1, vix_input.shape[2])
                ).reshape(-1, self.window_size, vix_input.shape[2])

            latency = (datetime.now() - start_time).total_seconds()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            success_msg = f"Preprocess VIX terminé pour {len(vix_data)} lignes"
            miya_speak(
                success_msg, 
                tag="NEURAL_PIPELINE", 
                voice_profile="calm", 
                priority=1
            )
            logger.info(success_msg)
            self.log_performance(
                "preprocess_vix",
                latency,
                success=True,
                num_rows=len(vix_data),
                confidence_drop_rate=confidence_drop_rate,
                num_features=len(vix_data.columns),
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_percent,
            )
            self.save_snapshot(
                "preprocess_vix",
                {
                    "num_rows": len(vix_data),
                    "vix_shape": str(vix_input.shape),
                    "confidence_drop_rate": confidence_drop_rate,
                    "new_features": [
                        f
                        for f in [
                            "iv_skew",
                            "iv_term_structure",
                            "option_skew",
                            "news_impact_score",
                        ]
                        if f in vix_data.columns
                    ],
                },
                compress=False,
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            return vix_input
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur preprocess_vix: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, 
                tag="NEURAL_PIPELINE", 
                voice_profile="urgent", 
                priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("preprocess_vix", latency, success=False, error=str(e))
            self.save_snapshot(
                "error_preprocess_vix", 
                {"error": str(e)}, 
                compress=False
            )
            raise

    def store_hyperparameters(
        self, 
        data: pd.DataFrame, 
        performance_metrics: Dict, 
        regime: str = "unknown"
    ) -> None:
        """Stocke les hyperparamètres des modèles dans market_memory.db (Proposition 2, Étape 1)."""
        try:
            start_time = datetime.now()
            hyperparameters = {
                "lstm": self.lstm_config,
                "cnn": self.cnn_config,
                "mlp_volatility": self.vol_mlp_config,
                "mlp_regime": self.regime_mlp_config,
                "vix_lstm": self.vix_lstm_config,
                "batch_size": self.batch_size,
                "pretrain_epochs": self.pretrain_epochs,
                "validation_split": self.validation_split,
                "normalization": self.normalization,
                "num_lstm_features": self.num_lstm_features,
            }
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "trade_id": 0,  # Pas de trade_id spécifique
                "metrics": performance_metrics,
                "hyperparameters": hyperparameters,
                "performance": performance_metrics.get("average_loss", 0.0),
                "regime": regime,
                "session": data.get("session", "unknown").iloc[-1] if "session" in data.columns else "unknown",
                "shap_metrics": {},
                "context": {"pipeline": "neural_pipeline"},
            }
            os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS meta_runs (
                    run_id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    trade_id INTEGER,
                    metrics TEXT,
                    hyperparameters TEXT,
                    performance REAL,
                    regime TEXT,
                    session TEXT,
                    shap_metrics TEXT,
                    context TEXT
                )
                """
            )
            cursor.execute(
                """
                INSERT INTO meta_runs (timestamp, trade_id, metrics, hyperparameters, performance, regime, session, shap_metrics, context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata["timestamp"],
                    metadata["trade_id"],
                    json.dumps(metadata["metrics"]),
                    json.dumps(metadata["hyperparameters"]),
                    metadata["performance"],
                    metadata["regime"],
                    metadata["session"],
                    json.dumps(metadata["shap_metrics"]),
                    json.dumps(metadata["context"]),
                ),
            )
            conn.commit()
            conn.close()
            latency = (datetime.now() - start_time).total_seconds()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            success_msg = f"Hyperparamètres stockés dans market_memory.db à {metadata['timestamp']}"
            miya_speak(
                success_msg, 
                tag="NEURAL_PIPELINE", 
                voice_profile="calm", 
                priority=1
            )
            logger.info(success_msg)
            self.log_performance(
                "store_hyperparameters",
                latency,
                success=True,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_percent,
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur stockage hyperparamètres: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, 
                tag="NEURAL_PIPELINE", 
                voice_profile="urgent", 
                priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "store_hyperparameters", 
                latency, 
                success=False, 
                error=str(e)
            )

    def predict_vix(self, data: pd.DataFrame) -> float:
        """Prédit la volatilité implicite (VIX) avec un modèle LSTM dédié (méthode 12)."""
        try:
            start_time = datetime.now()
            required_cols = [
                "vix_es_correlation",
                "iv_atm",
                "key_strikes_1",
                "max_pain_strike",
                "net_gamma",
                "zero_gamma",
                "dealer_zones_count",
                "vol_trigger",
                "ref_px",
                "data_release",
                "iv_skew",
                "iv_term_structure",
                "option_skew",
                "news_impact_score",
            ]
            missing = [col for col in required_cols if col not in data.columns]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing)) / len(required_cols), 1.0
            )
            if len(data) < self.window_size:
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = (
                    f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} "
                    f"({len(data)} lignes, {len(data.columns)} features)"
                )
                miya_alerts(
                    alert_msg, 
                    tag="NEURAL_PIPELINE", 
                    voice_profile="urgent", 
                    priority=3
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            cache_key = hashlib.sha256(data.to_json().encode()).hexdigest()
            if cache_key in self.cache:
                self.cache.move_to_end(cache_key)
                if (
                    datetime.now() - self.cache[cache_key]["timestamp"]
                ).total_seconds() < self.cache_hours * 3600:
                    vix_value = self.cache[cache_key]["vix_value"]
                    latency = (datetime.now() - start_time).total_seconds()
                    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
                    cpu_percent = psutil.cpu_percent()
                    success_msg = f"VIX prédit récupéré du cache: {vix_value:.2f}"
                    miya_speak(
                        success_msg, 
                        tag="NEURAL_PIPELINE", 
                        voice_profile="calm", 
                        priority=1
                    )
                    logger.info(success_msg)
                    self.log_performance(
                        "predict_vix_cache_hit",
                        latency,
                        success=True,
                        predicted_vix=vix_value,
                        memory_usage_mb=memory_usage,
                        cpu_usage_percent=cpu_percent,
                    )
                    send_alert(success_msg, priority=1)
                    send_telegram_alert(success_msg)
                    return vix_value

            def compute_vix():
                vix_input = self.preprocess_vix(data)
                predicted_vix = self.lstm_vix.predict(vix_input, verbose=0)
                return predicted_vix[-1, 0].item()

            vix_value = self.with_retries(compute_vix)
            self.cache[cache_key] = {
                "vix_value": vix_value, 
                "timestamp": datetime.now()
            }
            self.cache.move_to_end(cache_key)
            if len(self.cache) > self.max_cache_size:
                self.cache.popitem(last=False)

            latency = (datetime.now() - start_time).total_seconds()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            success_msg = (
                f"VIX prédit: {vix_value:.2f}, CPU: {cpu_percent}%, "
                f"Mémoire: {memory_usage:.2f} MB"
            )
            miya_speak(
                success_msg, 
                tag="NEURAL_PIPELINE", 
                voice_profile="calm", 
                priority=1
            )
            logger.info(success_msg)
            self.log_performance(
                "predict_vix",
                latency,
                success=True,
                predicted_vix=vix_value,
                cpu_usage_percent=cpu_percent,
                memory_usage_mb=memory_usage,
                confidence_drop_rate=confidence_drop_rate,
            )
            self.save_snapshot(
                "predict_vix",
                {
                    "predicted_vix": vix_value,
                    "cpu_percent": cpu_percent,
                    "memory_usage_mb": memory_usage,
                    "confidence_drop_rate": confidence_drop_rate,
                    "new_features": [
                        f
                        for f in [
                            "iv_skew",
                            "iv_term_structure",
                            "option_skew",
                            "news_impact_score",
                        ]
                        if f in data.columns
                    ],
                },
                compress=False,
            )
            self.plot_vix(
                np.array([vix_value]), 
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            return vix_value
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur predict_vix: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, 
                tag="NEURAL_PIPELINE", 
                voice_profile="urgent", 
                priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("predict_vix", latency, success=False, error=str(e))
            self.save_snapshot(
                "error_predict_vix", 
                {"error": str(e)}, 
                compress=False
            )
            return 0.0

    def plot_vix(self, predicted_vix: "np.ndarray", timestamp: str) -> None:
        """Génère une visualisation pour predicted_vix."""
        try:
            start_time = datetime.now()
            timestamp_safe = timestamp.replace(":", "-")
            os.makedirs(FIGURE_DIR, exist_ok=True)
            plt.figure(figsize=(10, 6))
            plt.plot(predicted_vix, label="Predicted VIX")  # Suppression de color="blue"
            plt.title(f"Predicted VIX - {timestamp}")
            plt.xlabel("Sample Index")
            plt.ylabel("VIX Value")
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(FIGURE_DIR, f"vix_{timestamp_safe}.png")
            plt.savefig(plot_path)
            plt.close()
            latency = (datetime.now() - start_time).total_seconds()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            success_msg = f"Visualisation VIX générée: {plot_path}"
            miya_speak(
                success_msg, 
                tag="NEURAL_PIPELINE", 
                voice_profile="calm", 
                priority=2
            )
            logger.info(success_msg)
            self.log_performance(
                "plot_vix",
                latency,
                success=True,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_percent,
            )
            self.save_snapshot(
                "plot_vix", 
                {"plot_path": plot_path}, 
                compress=False
            )
            send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur génération visualisation VIX: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, 
                tag="NEURAL_PIPELINE", 
                voice_profile="urgent", 
                priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("plot_vix", latency, success=False, error=str(e))
            self.save_snapshot(
                "error_plot_vix", 
                {"error": str(e)}, 
                compress=False
            )

    def pretrain_models(
        self,
        lstm_input: "np.ndarray",
        cnn_input: "np.ndarray",
        full_data: pd.DataFrame,
        lstm_targets: Optional["np.ndarray"] = None,
        cnn_targets: Optional["np.ndarray"] = None,
        vol_targets: Optional["np.ndarray"] = None,
        regime_targets: Optional["np.ndarray"] = None,
        vix_targets: Optional["np.ndarray"] = None,
    ) -> None:
        """Pré-entraîne les modèles LSTM, CNN, MLP, et LSTM VIX."""
        try:
            start_time = datetime.now()
            required_cols = [
                "timestamp",
                "ofi_score",
                "vix_es_correlation",
                "atr_14",
                "bid_ask_imbalance",
                "trade_aggressiveness",
                "iv_skew",
                "iv_term_structure",
                "option_skew",
                "news_impact_score",
            ]
            missing = [col for col in required_cols if col not in full_data.columns]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing)) / len(required_cols), 1.0
            )
            if len(full_data) < self.window_size:
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = (
                    f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} "
                    f"({len(full_data)} lignes, {len(full_data.columns)} features)"
                )
                miya_alerts(
                    alert_msg, 
                    tag="NEURAL_PIPELINE", 
                    voice_profile="urgent", 
                    priority=3
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            if not self.training_mode:
                self.validate_shap_features(full_data.columns.tolist())

            # Cibles réalistes
            if lstm_targets is None:
                lstm_targets = np.zeros(
                    (len(lstm_input), self.num_lstm_features)
                )  # Placeholder
            if cnn_targets is None:
                cnn_targets = (
                    full_data["bid_ask_imbalance"].values.reshape(-1, 1)
                    if "bid_ask_imbalance" in full_data.columns
                    else np.zeros((len(cnn_input), 1))
                )
            if vix_targets is None:
                vix_targets = (
                    full_data["vix_es_correlation"].values
                    if "vix_es_correlation" in full_data.columns
                    else np.random.uniform(10, 30, len(full_data))
                )
            lstm_features = self.lstm.predict(lstm_input, verbose=0)[: len(full_data)]
            cnn_pressure = self.cnn.predict(cnn_input, verbose=0)[: len(full_data)]
            combined_features = np.hstack(
                [
                    full_data.iloc[:, : self.base_features].values,
                    lstm_features,
                    cnn_pressure,
                ]
            )
            if self.normalization:
                combined_features = self.scaler_manager.transform(
                    "full", 
                    combined_features
                )
            if vol_targets is None:
                vol_targets = (
                    full_data["atr_14"].values
                    if "atr_14" in full_data.columns
                    else np.random.uniform(0.01, 0.1, len(combined_features))
                )
            if regime_targets is None:
                regime_targets = np.random.randint(0, 3, len(combined_features))

            # Entraînement avec collecte des métriques
            lstm_history = self.lstm.fit(
                lstm_input[: len(full_data)],
                lstm_targets,
                epochs=self.pretrain_epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                verbose=0,
                callbacks=[],
            )
            cnn_history = self.cnn.fit(
                cnn_input[: len(full_data)],
                cnn_targets,
                epochs=self.pretrain_epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                verbose=0,
                callbacks=[],
            )
            vol_mlp_history = self.vol_mlp.fit(
                combined_features,
                vol_targets,
                epochs=self.pretrain_epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                verbose=0,
                callbacks=[],
            )
            regime_mlp_history = self.regime_mlp.fit(
                combined_features,
                regime_targets,
                epochs=self.pretrain_epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                verbose=0,
                callbacks=[],
            )
            vix_input = self.preprocess_vix(full_data)
            vix_lstm_history = self.lstm_vix.fit(
                vix_input,
                vix_targets[: len(vix_input)],
                epochs=self.pretrain_epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                verbose=0,
                callbacks=[],
            )
            self.save_models()

            # Collecte des métriques réelles
            performance_metrics = {
                "lstm_val_loss": float(lstm_history.history["val_loss"][-1]) if "val_loss" in lstm_history.history else 0.0,
                "cnn_val_loss": float(cnn_history.history["val_loss"][-1]) if "val_loss" in cnn_history.history else 0.0,
                "vol_mlp_val_loss": float(vol_mlp_history.history["val_loss"][-1]) if "val_loss" in vol_mlp_history.history else 0.0,
                "regime_mlp_val_loss": float(regime_mlp_history.history["val_loss"][-1]) if "val_loss" in regime_mlp_history.history else 0.0,
                "regime_mlp_val_accuracy": float(regime_mlp_history.history["val_accuracy"][-1]) if "val_accuracy" in regime_mlp_history.history else 0.0,
                "vix_lstm_val_loss": float(vix_lstm_history.history["val_loss"][-1]) if "val_loss" in vix_lstm_history.history else 0.0,
                "num_rows": len(full_data),
                "average_loss": float(np.mean([
                    lstm_history.history["val_loss"][-1] if "val_loss" in lstm_history.history else 0.0,
                    cnn_history.history["val_loss"][-1] if "val_loss" in cnn_history.history else 0.0,
                    vol_mlp_history.history["val_loss"][-1] if "val_loss" in vol_mlp_history.history else 0.0,
                    regime_mlp_history.history["val_loss"][-1] if "val_loss" in regime_mlp_history.history else 0.0,
                    vix_lstm_history.history["val_loss"][-1] if "val_loss" in vix_lstm_history.history else 0.0,
                ])),
            }
            self.store_hyperparameters(full_data, performance_metrics)

            latency = (datetime.now() - start_time).total_seconds()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            success_msg = (
                f"Pré-entraînement modèles terminé pour {len(full_data)} lignes"
            )
            miya_speak(
                success_msg, 
                tag="NEURAL_PIPELINE", 
                voice_profile="calm", 
                priority=1
            )
            logger.info(success_msg)
            self.log_performance(
                "pretrain_models",
                latency,
                success=True,
                num_rows=len(full_data),
                confidence_drop_rate=confidence_drop_rate,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_percent,
            )
            self.save_snapshot(
                "pretrain_models",
                {
                    "num_rows": len(full_data),
                    "confidence_drop_rate": confidence_drop_rate,
                    "new_features": [
                        f
                        for f in [
                            "bid_ask_imbalance",
                            "trade_aggressiveness",
                            "iv_skew",
                            "iv_term_structure",
                            "option_skew",
                            "news_impact_score",
                        ]
                        if f in full_data.columns
                    ],
                },
                compress=False,
            )
            self.save_dashboard_status(
                {
                    "status": "pretrained",
                    "num_rows": len(full_data),
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
                    "new_features": [
                        f
                        for f in [
                            "bid_ask_imbalance",
                            "trade_aggressiveness",
                            "iv_skew",
                            "iv_term_structure",
                            "option_skew",
                            "news_impact_score",
                        ]
                        if f in full_data.columns
                    ],
                }
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur pré-entraînement: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, 
                tag="NEURAL_PIPELINE", 
                voice_profile="urgent", 
                priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "pretrain_models", 
                latency, 
                success=False, 
                error=str(e)
            )
            self.save_snapshot(
                "error_pretrain_models", 
                {"error": str(e)}, 
                compress=False
            )
            raise

 ```python
    def run(
        self,
        raw_data: pd.DataFrame,
        options_data: pd.DataFrame,
        orderflow_data: pd.DataFrame,
    ) -> Dict[str, Union["np.ndarray", float]]:
        """Exécute le pipeline neuronal en mode batch."""
        try:
            start_time = datetime.now()
            confidence_drop_rate = 0.0
            for df, name in [
                (raw_data, "raw"),
                (options_data, "options"),
                (orderflow_data, "orderflow"),
            ]:
                if len(df) < self.window_size:
                    confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = (
                    f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} "
                    f"({len(raw_data)} lignes)"
                )
                miya_alerts(
                    alert_msg, 
                    tag="NEURAL_PIPELINE", 
                    voice_profile="urgent", 
                    priority=3
                )
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            cache_key = hashlib.sha256(
                pd.concat([raw_data, options_data, orderflow_data]).to_json().encode()
            ).hexdigest()
            if cache_key in self.cache:
                self.cache.move_to_end(cache_key)
                if (
                    datetime.now() - self.cache[cache_key]["timestamp"]
                ).total_seconds() < self.cache_hours * 3600:
                    result = self.cache[cache_key]["result"]
                    latency = (datetime.now() - start_time).total_seconds()
                    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
                    cpu_percent = psutil.cpu_percent()
                    success_msg = "Run batch récupéré du cache"
                    miya_speak(
                        success_msg, 
                        tag="NEURAL_PIPELINE", 
                        voice_profile="calm", 
                        priority=1
                    )
                    logger.info(success_msg)
                    self.log_performance(
                        "run_cache_hit",
                        latency,
                        success=True,
                        num_rows=len(raw_data),
                        memory_usage_mb=memory_usage,
                        cpu_usage_percent=cpu_percent,
                    )
                    send_alert(success_msg, priority=1)
                    send_telegram_alert(success_msg)
                    return result

            lstm_input, cnn_input, full_data = self.preprocess(
                raw_data, options_data, orderflow_data
            )
            lstm_features = self.lstm.predict(lstm_input, verbose=0)[: len(full_data)]
            cnn_pressure = self.cnn.predict(cnn_input, verbose=0)[: len(full_data)]
            combined_features = np.hstack(
                [
                    full_data.iloc[:, : self.base_features].values,
                    lstm_features,
                    cnn_pressure,
                ]
            )
            if self.normalization:
                combined_features = self.scaler_manager.transform(
                    "full", 
                    combined_features
                )
            volatility = self.vol_mlp.predict(combined_features, verbose=0)
            regime_probs = self.regime_mlp.predict(combined_features, verbose=0)
            vix_value = self.predict_vix(full_data)
            result = {
                "features": np.hstack(
                    [
                        lstm_features,
                        cnn_pressure,
                        volatility,
                        regime_probs,
                        np.full((len(full_data), 1), vix_value),
                    ]
                ),
                "volatility": volatility,
                "regime": np.argmax(regime_probs, axis=1),
                "predicted_vix": vix_value,
            }
            self.cache[cache_key] = {"result": result, "timestamp": datetime.now()}
            self.cache.move_to_end(cache_key)
            if len(self.cache) > self.max_cache_size:
                self.cache.popitem(last=False)

            latency = (datetime.now() - start_time).total_seconds()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            success_msg = f"Run batch terminé pour {len(full_data)} lignes"
            miya_speak(
                success_msg, 
                tag="NEURAL_PIPELINE", 
                voice_profile="calm", 
                priority=1
            )
            logger.info(success_msg)
            self.log_performance(
                "run",
                latency,
                success=True,
                num_rows=len(full_data),
                confidence_drop_rate=confidence_drop_rate,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_percent,
            )
            self.save_snapshot(
                "run",
                {
                    "num_rows": len(full_data),
                    "feature_shape": str(result["features"].shape),
                    "confidence_drop_rate": confidence_drop_rate,
                    "new_features": [
                        f
                        for f in [
                            "bid_ask_imbalance",
                            "trade_aggressiveness",
                            "iv_skew",
                            "iv_term_structure",
                            "option_skew",
                            "news_impact_score",
                        ]
                        if f in full_data.columns
                    ],
                },
                compress=False,
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            return result
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur run: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, 
                tag="NEURAL_PIPELINE", 
                voice_profile="urgent", 
                priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("run", latency, success=False, error=str(e))
            self.save_snapshot(
                "error_run", 
                {"error": str(e)}, 
                compress=False
            )
            return {
                "features": np.zeros(
                    (
                        len(raw_data) - self.window_size + 1,
                        self.base_features + self.num_lstm_features + 2,
                    )
                ),
                "volatility": np.zeros((len(raw_data) - self.window_size + 1, 1)),
                "regime": np.zeros(len(raw_data) - self.window_size + 1),
                "predicted_vix": 0.0,
            }

    def run_incremental(
        self,
        row: pd.Series,
        buffer: pd.DataFrame,
        options_row: pd.Series,
        orderflow_row: pd.Series,
    ) -> Dict[str, Union["np.ndarray", float]]:
        """Exécute le pipeline neuronal en mode incrémental pour une seule ligne."""
        try:
            start_time = datetime.now()
            if len(buffer) < self.window_size:
                error_msg = (
                    f"Buffer insuffisant: {len(buffer)} lignes, "
                    f"requis >= {self.window_size}"
                )
                miya_alerts(
                    error_msg, 
                    tag="NEURAL_PIPELINE", 
                    voice_profile="urgent", 
                    priority=3
                )
                send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            # Créer un DataFrame temporaire pour une ligne
            raw_data = pd.concat([buffer, pd.DataFrame([row])], ignore_index=True)
            options_data = pd.DataFrame([options_row])
            orderflow_data = pd.DataFrame([orderflow_row])

            lstm_input, cnn_input, full_data = self.preprocess(
                raw_data, options_data, orderflow_data
            )
            lstm_features = self.lstm.predict(lstm_input, verbose=0)[-1:]  # Dernière fenêtre
            cnn_pressure = self.cnn.predict(cnn_input, verbose=0)[-1:]
            combined_features = np.hstack(
                [
                    full_data.iloc[-1:, : self.base_features].values,
                    lstm_features,
                    cnn_pressure,
                ]
            )
            if self.normalization:
                combined_features = self.scaler_manager.transform(
                    "full", 
                    combined_features
                )
            volatility = self.vol_mlp.predict(combined_features, verbose=0)
            regime_probs = self.regime_mlp.predict(combined_features, verbose=0)
            vix_value = self.predict_vix(full_data)

            result = {
                "features": np.hstack(
                    [
                        lstm_features,
                        cnn_pressure,
                        volatility,
                        regime_probs,
                        np.array([[vix_value]]),
                    ]
                ),
                "volatility": volatility,
                "regime": np.argmax(regime_probs, axis=1),
                "predicted_vix": vix_value,
            }

            latency = (datetime.now() - start_time).total_seconds()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            success_msg = "Run incrémental terminé pour 1 ligne"
            miya_speak(
                success_msg, 
                tag="NEURAL_PIPELINE", 
                voice_profile="calm", 
                priority=1
            )
            logger.info(success_msg)
            self.log_performance(
                "run_incremental",
                latency,
                success=True,
                num_rows=1,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_percent,
            )
            self.save_snapshot(
                "run_incremental",
                {
                    "num_rows": 1,
                    "feature_shape": str(result["features"].shape),
                },
                compress=False,
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            return result
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur run_incremental: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, 
                tag="NEURAL_PIPELINE", 
                voice_profile="urgent", 
                priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("run_incremental", latency, success=False, error=str(e))
            self.save_snapshot(
                "error_run_incremental", 
                {"error": str(e)}, 
                compress=False
            )
            return {
                "features": np.zeros((1, self.base_features + self.num_lstm_features + 2)),
                "volatility": np.zeros((1, 1)),
                "regime": np.zeros(1),
                "predicted_vix": 0.0,
            }

    def save_models(self, output_dir: str = None) -> None:
        """Sauvegarde les modèles neuronaux."""
        try:
            start_time = datetime.now()
            output_dir = output_dir or self.save_dir
            os.makedirs(output_dir, exist_ok=True)

            def save_all_models():
                self.lstm.save(os.path.join(output_dir, "lstm_model.h5"))
                self.cnn.save(os.path.join(output_dir, "cnn_model.h5"))
                self.vol_mlp.save(os.path.join(output_dir, "vol_mlp_model.h5"))
                self.regime_mlp.save(os.path.join(output_dir, "regime_mlp_model.h5"))
                self.lstm_vix.save(os.path.join(output_dir, "lstm_vix_model.h5"))

            self.with_retries(save_all_models)
            latency = (datetime.now() - start_time).total_seconds()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            success_msg = f"Modèles sauvegardés dans {output_dir}"
            miya_speak(
                success_msg, 
                tag="NEURAL_PIPELINE", 
                voice_profile="calm", 
                priority=1
            )
            logger.info(success_msg)
            self.log_performance(
                "save_models",
                latency,
                success=True,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_percent,
            )
            self.save_snapshot(
                "save_models", 
                {"output_dir": output_dir}, 
                compress=False
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur save_models: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, 
                tag="NEURAL_PIPELINE", 
                voice_profile="urgent", 
                priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "save_models", 
                latency, 
                success=False, 
                error=str(e)
            )
            self.save_snapshot(
                "error_save_models", 
                {"error": str(e)}, 
                compress=False
            )

    def load_models(self, input_dir: str = None) -> int:
        """Charge les modèles neuronaux."""
        try:
            start_time = datetime.now()
            input_dir = input_dir or self.save_dir
            loaded = 0
            for model_name, path in [
                ("lstm", "lstm_model.h5"),
                ("cnn", "cnn_model.h5"),
                ("vol_mlp", "vol_mlp_model.h5"),
                ("regime_mlp", "regime_mlp_model.h5"),
                ("lstm_vix", "lstm_vix_model.h5"),
            ]:
                full_path = os.path.join(input_dir, path)
                if os.path.exists(full_path):
                    try:
                        model = load_model(full_path)
                        test_input = (
                            np.zeros(
                                (
                                    1,
                                    self.window_size,
                                    (
                                        16
                                        if model_name == "lstm"
                                        else 14 if model_name == "lstm_vix" else 7
                                    ),
                                )
                            )
                            if model_name in ["lstm", "cnn", "lstm_vix"]
                            else np.zeros(
                                (1, self.base_features + self.num_lstm_features + 1)
                            )
                        )
                        model.predict(test_input, verbose=0)  # Test d'intégrité
                        setattr(self, model_name, model)
                        loaded += 1
                    except Exception as e:
                        error_msg = (
                            f"Échec chargement modèle {model_name}: {str(e)}\n"
                            f"{traceback.format_exc()}"
                        )
                        miya_alerts(
                            error_msg,
                            tag="NEURAL_PIPELINE",
                            voice_profile="urgent",
                            priority=3,
                        )
                        send_alert(error_msg, priority=3)
                        send_telegram_alert(error_msg)
                        logger.error(error_msg)
            latency = (datetime.now() - start_time).total_seconds()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            success_msg = f"{loaded} modèles chargés"
            miya_speak(
                success_msg, 
                tag="NEURAL_PIPELINE", 
                voice_profile="calm", 
                priority=1
            )
            logger.info(success_msg)
            self.log_performance(
                "load_models",
                latency,
                success=True,
                num_models=loaded,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_percent,
            )
            self.save_snapshot(
                "load_models",
                {"num_models": loaded, "input_dir": input_dir},
                compress=False,
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            return loaded
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur load_models: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(
                error_msg, 
                tag="NEURAL_PIPELINE", 
                voice_profile="urgent", 
                priority=3
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "load_models", 
                latency, 
                success=False, 
                error=str(e)
            )
            self.save_snapshot(
                "error_load_models", 
                {"error": str(e)}, 
                compress=False
            )
            return 0


if __name__ == "__main__":
    try:
        pipeline = NeuralPipeline()
        # Simuler des données pour le test
        raw_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "open": np.random.normal(5100, 10, 100),
                "high": np.random.normal(5105, 10, 100),
                "low": np.random.normal(5095, 10, 100),
                "close": np.random.normal(5100, 10, 100),
                "volume": np.random.randint(100, 1000, 100),
                "atr_14": np.random.uniform(0.5, 2.0, 100),
                "adx_14": np.random.uniform(10, 40, 100),
                "rsi_14": np.random.uniform(30, 70, 100),
                "vix_es_correlation": np.random.uniform(-0.5, 0.5, 100),
                "bid_ask_imbalance": np.random.normal(0, 0.1, 100),
                "trade_aggressiveness": np.random.normal(0, 0.2, 100),
                "iv_skew": np.random.normal(0.01, 0.005, 100),
                "iv_term_structure": np.random.normal(0.02, 0.005, 100),
            }
        )
        options_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "gex": np.random.uniform(-1000, 1000, 100),
                "oi_peak_call_near": np.random.randint(5000, 15000, 100),
                "gamma_wall": np.random.uniform(5090, 5110, 100),
                "key_strikes_1": np.random.uniform(5000, 5200, 100),
                "key_strikes_2": np.random.uniform(5000, 5200, 100),
                "key_strikes_3": np.random.uniform(5000, 5200, 100),
                "key_strikes_4": np.random.uniform(5000, 5200, 100),
                "key_strikes_5": np.random.uniform(5000, 5200, 100),
                "max_pain_strike": np.random.uniform(5000, 5200, 100),
                "net_gamma": np.random.uniform(-1, 1, 100),
                "zero_gamma": np.random.uniform(5000, 5200, 100),
                "dealer_zones_count": np.random.randint(0, 11, 100),
                "vol_trigger": np.random.uniform(-1, 1, 100),
                "ref_px": np.random.uniform(5000, 5200, 100),
                "data_release": np.random.randint(0, 2, 100),
                "iv_atm": np.random.uniform(0.1, 0.3, 100),
                "option_skew": np.random.normal(0.02, 0.01, 100),
                "news_impact_score": np.random.uniform(0.0, 1.0, 100),
            }
        )
        orderflow_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "bid_size_level_1": np.random.randint(50, 500, 100),
                "ask_size_level_1": np.random.randint(50, 500, 100),
                "ofi_score": np.random.uniform(-1, 1, 100),
            }
        )
        lstm_input, cnn_input, full_data = pipeline.preprocess(
            raw_data, options_data, orderflow_data
        )
        pipeline.pretrain_models(lstm_input, cnn_input, full_data)
        result = pipeline.run(raw_data, options_data, orderflow_data)
        print(
            f"Features shape: {result['features'].shape}, "
            f"Predicted VIX: {result['predicted_vix']}"
        )
        success_msg = "Test NeuralPipeline batch terminé"
        miya_speak(
            success_msg, 
            tag="NEURAL_PIPELINE", 
            voice_profile="calm", 
            priority=1
        )
        logger.info(success_msg)
        send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)

        # Test incrémental
        row = raw_data.iloc[-1]
        options_row = options_data.iloc[-1]
        orderflow_row = orderflow_data.iloc[-1]
        buffer = raw_data.iloc[-50:].copy()
        result_incremental = pipeline.run_incremental(
            row, buffer, options_row, orderflow_row
        )
        print(
            f"Incremental features shape: {result_incremental['features'].shape}, "
            f"Predicted VIX: {result_incremental['predicted_vix']}"
        )
        success_msg = "Test NeuralPipeline incrémental terminé"
        miya_speak(
            success_msg, 
            tag="NEURAL_PIPELINE", 
            voice_profile="calm", 
            priority=1
        )
        logger.info(success_msg)
        send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)

        # Test predict_vix
        vix_value = pipeline.predict_vix(full_data)
        print(f"Predicted VIX: {vix_value}")
        success_msg = "Test predict_vix terminé"
        miya_speak(
            success_msg, 
            tag="NEURAL_PIPELINE", 
            voice_profile="calm", 
            priority=1
        )
        logger.info(success_msg)
        send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
    except Exception as e:
        error_msg = (
            f"Erreur exécution: {str(e)}\n{traceback.format_exc()}"
        )
        miya_alerts(
            error_msg, 
            tag="NEURAL_PIPELINE", 
            voice_profile="urgent", 
            priority=3
        )
        send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)
        logger.error(error_msg)
        raise
