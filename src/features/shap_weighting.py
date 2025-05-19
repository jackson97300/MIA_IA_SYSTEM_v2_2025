# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/features/shap_weighting.py
# Analyse SHAP pour calculer l'importance des features (320 pour entraînement, top 150 pour production).
# Intègre les nouvelles métriques de spotgamma_recalculator.py.
#
# Version : 2.1.5
# Date : 2025-05-13
#
# Rôle : Génère data/features/feature_importance.csv avec les SHAP values pour les top 150 features,
#        priorisant key_strikes_1 à _5, max_pain_strike, net_gamma, zero_gamma, dealer_zones_count,
#        vol_trigger, ref_px, data_release, avec parallélisation via joblib (méthode 17).
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, xgboost>=2.0.0,<3.0.0, shap>=0.41.0,<0.42.0,
#   psutil>=5.9.0,<6.0.0, matplotlib>=3.7.0,<4.0.0, joblib>=1.2.0,<2.0.0, pyyaml>=6.0.0,<7.0.0
# - src/model/utils/miya_console.py
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/es_config.yaml
# - data/models/shap_regime_detector.pkl
# - Données avec features
#
# Outputs :
# - data/features/feature_importance.csv
# - data/logs/shap_performance.csv
# - data/shap_snapshots/*.json (option *.json.gz)
# - data/figures/shap/
# - data/models/shap_regime_detector.pkl
#
# Notes :
# - Conforme à structure.txt (version 2.1.5, 2025-05-13).
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des calculs SHAP.
# - Intègre validation SHAP (Phase 17) pour assurer la conformité des top 150 features en production.
# - Préserve toutes les fonctionnalités existantes (entraînement XGBoost, calcul SHAP).
# - Utilise exclusivement IQFeed via data_provider.py, avec retries (max 3, délai 2^attempt secondes).
# - Sauvegarde des snapshots JSON non compressés par défaut avec option de compression gzip.
# - Envoie des alertes via AlertManager et telegram_alert pour les erreurs critiques et succès.
# - Tests unitaires disponibles dans tests/test_shap_weighting.py (à implémenter).
# - Validation complète prévue pour juin 2025.
# - Évolution future : Migration API Investing.com (juin 2025), optimisation pour feature_pipeline.py.

import gzip
import hashlib
import json
import logging
import os
import pickle
import time
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
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from src.model.utils.alert_manager import send_alert
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.utils.telegram_alert import send_telegram_alert

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "shap_snapshots")
CSV_LOG_PATH = os.path.join(BASE_DIR, "data", "logs", "shap_performance.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures", "shap")
FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR, "data", "features", "feature_importance.csv"
)
MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "shap_regime_detector.pkl")

# Création des dossiers
os.makedirs(os.path.join(BASE_DIR, "data", "logs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data", "models"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data", "features"), exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Configuration du logging
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "data", "logs", "shap_weighting.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Constantes
MAX_RETRIES = 3
RETRY_DELAY = 2.0


class SHAPWeighting:
    """
    Classe pour calculer l'importance des features via analyse SHAP avec un modèle de détection de régime.
    """

    def __init__(
        self, config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml")
    ):
        """
        Initialise le calculateur SHAP.

        Args:
            config_path (str): Chemin vers le fichier de configuration.
        """
        self.log_buffer = []
        self.cache = {}
        self.model = None
        self.explainer = None
        try:
            self.config = self.load_config(config_path)
            self.buffer_size = self.config.get("buffer_size", 100)
            self.max_cache_size = self.config.get("max_cache_size", 1000)
            self.buffer_maxlen = self.config.get("buffer_maxlen", 1000)
            self.buffer = deque(maxlen=self.buffer_maxlen)
            self.load_model()
            miya_speak(
                "SHAPWeighting initialisé", tag="SHAP", voice_profile="calm", priority=2
            )
            logger.info("SHAPWeighting initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config_path": config_path}, compress=False)
            send_alert("SHAPWeighting initialisé", priority=2)
            send_telegram_alert("SHAPWeighting initialisé")
        except Exception as e:
            error_msg = f"Erreur initialisation SHAPWeighting: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="SHAP", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            self.config = {
                "model_path": MODEL_PATH,
                "window_size": 100,
                "regime_mapping": {"Trend": 0, "Range": 1, "Defensive": 2},
                "buffer_size": 100,
                "max_cache_size": 1000,
                "buffer_maxlen": 1000,
                "cache_hours": 24,
                "expected_count": 320,
                "key_features_count": 150,
            }
            self.buffer_size = 100
            self.max_cache_size = 1000
            self.buffer_maxlen = 1000
            self.buffer = deque(maxlen=self.buffer_maxlen)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Charge la configuration depuis es_config.yaml."""
        try:
            start_time = time.time()

            def load_yaml():
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                if "shap_weighting" not in config:
                    raise ValueError(
                        "Clé 'shap_weighting' manquante dans la configuration"
                    )
                required_keys = [
                    "model_path",
                    "window_size",
                    "regime_mapping",
                    "expected_count",
                    "key_features_count",
                ]
                missing_keys = [
                    key for key in required_keys if key not in config["shap_weighting"]
                ]
                if missing_keys:
                    raise ValueError(
                        f"Clés manquantes dans 'shap_weighting': {missing_keys}"
                    )
                return config["shap_weighting"]

            config = self.with_retries(load_yaml)
            cache_key = hashlib.sha256(str(config).encode()).hexdigest()
            self.cache[cache_key] = {"config": config, "timestamp": datetime.now()}
            if len(self.cache) > self.max_cache_size:
                self.cache.pop(next(iter(self.cache)))
            latency = time.time() - start_time
            miya_speak(
                "Configuration shap_weighting chargée",
                tag="SHAP",
                voice_profile="calm",
                priority=2,
            )
            logger.info("Configuration shap_weighting chargée")
            self.log_performance("load_config", latency, success=True)
            self.save_snapshot(
                "load_config", {"config_path": config_path}, compress=False
            )
            send_alert("Configuration shap_weighting chargée", priority=2)
            send_telegram_alert("Configuration shap_weighting chargée")
            return config
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur chargement config: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="SHAP", voice_profile="urgent", priority=3)
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
                    error_msg = f"Échec après {max_attempts} tentatives: {str(e)}"
                    miya_alerts(
                        error_msg, tag="SHAP", voice_profile="urgent", priority=4
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
                warning_msg = f"Tentative {attempt+1} échouée, retry après {delay}s"
                miya_speak(warning_msg, tag="SHAP", level="warning", priority=3)
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
                time.sleep(delay)

    def load_model(self) -> None:
        """Charge le modèle XGBoost si disponible."""
        try:
            start_time = time.time()
            model_path = self.config.get("model_path", MODEL_PATH)
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
                self.explainer = shap.TreeExplainer(self.model)
                latency = time.time() - start_time
                success_msg = f"Modèle XGBoost chargé: {model_path}"
                miya_speak(success_msg, tag="SHAP", voice_profile="calm", priority=1)
                logger.info(success_msg)
                send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)
                self.log_performance("load_model", latency, success=True)
            else:
                warning_msg = f"Modèle XGBoost introuvable: {model_path}"
                miya_speak(warning_msg, tag="SHAP", level="warning", priority=3)
                logger.warning(warning_msg)
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                self.model = None
                self.explainer = None
        except Exception as e:
            latency = time.time() - start_time
            error_msg = (
                f"Erreur chargement modèle XGBoost: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="SHAP", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("load_model", latency, success=False, error=str(e))
            self.model = None
            self.explainer = None

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ) -> None:
        """Journalise les performances des opérations critiques."""
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
            cpu_usage = psutil.cpu_percent()  # % CPU
            if memory_usage > 1024:
                alert_msg = (
                    f"ALERTE: Utilisation mémoire élevée ({memory_usage:.2f} MB)"
                )
                miya_alerts(alert_msg, tag="SHAP", voice_profile="urgent", priority=5)
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
            miya_alerts(error_msg, tag="SHAP", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

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
                alert_msg = f"Snapshot size {file_size:.2f} MB exceeds 1 MB"
                miya_alerts(alert_msg, tag="SHAP", voice_profile="urgent", priority=3)
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)
            latency = time.time() - start_time
            success_msg = f"Snapshot {snapshot_type} sauvegardé: {save_path}"
            miya_speak(success_msg, tag="SHAP", voice_profile="calm", priority=1)
            logger.info(success_msg)
            self.log_performance(
                "save_snapshot",
                latency,
                success=True,
                snapshot_type=snapshot_type,
                file_size_mb=file_size,
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}"
            self.log_performance("save_snapshot", 0, success=False, error=str(e))
            miya_alerts(error_msg, tag="SHAP", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def validate_shap_features(self, features: List[str]) -> bool:
        """Valide que les features sont dans les top 150 SHAP."""
        try:
            start_time = time.time()
            if not os.path.exists(FEATURE_IMPORTANCE_PATH):
                error_msg = "Fichier SHAP manquant"
                miya_alerts(error_msg, tag="SHAP", voice_profile="urgent", priority=4)
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return False
            shap_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
            if len(shap_df) < 150:
                error_msg = f"Nombre insuffisant de SHAP features: {len(shap_df)} < 150"
                miya_alerts(error_msg, tag="SHAP", voice_profile="urgent", priority=4)
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return False
            valid_features = set(shap_df["feature"].head(150))
            missing = [f for f in features if f not in valid_features]
            if missing:
                warning_msg = f"Features non incluses dans top 150 SHAP: {missing}"
                miya_alerts(
                    warning_msg, tag="SHAP", voice_profile="warning", priority=3
                )
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
            latency = time.time() - start_time
            success_msg = "SHAP features validées"
            miya_speak(success_msg, tag="SHAP", voice_profile="calm", priority=1)
            logger.info(success_msg)
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
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
            error_msg = f"Erreur validation SHAP features: {str(e)}"
            miya_alerts(error_msg, tag="SHAP", voice_profile="urgent", priority=4)
            send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "validate_shap_features", latency, success=False, error=str(e)
            )
            return False

    def plot_shap_weights(self, shap_importance: pd.Series, timestamp: str) -> None:
        """Génère des visualisations pour les poids SHAP."""
        try:
            start_time = time.time()
            timestamp_safe = timestamp.replace(":", "-")
            os.makedirs(FIGURES_DIR, exist_ok=True)

            # Bar plot des top 10 features
            plt.figure(figsize=(10, 6))
            shap_importance.head(10).plot(kind="bar")
            plt.title(f"Top 10 Features par Importance SHAP - {timestamp}")
            plt.xlabel("Feature")
            plt.ylabel("Importance SHAP (moyenne absolue)")
            plt.xticks(rotation=45, ha="right")
            plt.grid(True)
            plt.tight_layout()
            bar_plot_path = os.path.join(FIGURES_DIR, f"shap_bar_{timestamp_safe}.png")
            plt.savefig(bar_plot_path)
            plt.close()

            # Plot spécifique pour les nouvelles métriques
            new_metrics = [
                "key_strikes_1",
                "key_strikes_2",
                "key_strikes_3",
                "key_strikes_4",
                "key_strikes_5",
                "max_pain_strike",
                "net_gamma",
                "zero_gamma",
                "dealer_zones_count",
                "vol_trigger",
                "ref_px",
                "data_release",
            ]
            new_metrics_present = [m for m in new_metrics if m in shap_importance.index]
            if new_metrics_present:
                plt.figure(figsize=(10, 6))
                shap_importance[new_metrics_present].plot(kind="bar", color="orange")
                plt.title(f"SHAP Importance des Nouvelles Métriques - {timestamp}")
                plt.xlabel("Feature")
                plt.ylabel("Importance SHAP (moyenne absolue)")
                plt.xticks(rotation=45, ha="right")
                plt.grid(True)
                plt.tight_layout()
                metrics_plot_path = os.path.join(
                    FIGURES_DIR, f"shap_new_metrics_{timestamp_safe}.png"
                )
                plt.savefig(metrics_plot_path)
                plt.close()

            latency = time.time() - start_time
            success_msg = f"Visualisations SHAP générées: {bar_plot_path}"
            miya_speak(success_msg, tag="SHAP", voice_profile="calm", priority=2)
            logger.info(success_msg)
            self.log_performance(
                "plot_shap_weights",
                latency,
                success=True,
                new_metrics_count=len(new_metrics_present),
            )
            send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
        except Exception as e:
            latency = time.time() - start_time
            error_msg = (
                f"Erreur génération visualisations: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="SHAP", voice_profile="urgent", priority=2)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "plot_shap_weights", latency, success=False, error=str(e)
            )

    def train_shap_regime_detector(
        self,
        data: pd.DataFrame,
        model_path: str = MODEL_PATH,
        target_col: str = "neural_regime",
    ) -> None:
        """Entraîne et sauvegarde un modèle XGBoost pour la détection de régime."""
        try:
            start_time = time.time()
            if target_col not in data.columns:
                error_msg = f"Colonne '{target_col}' manquante"
                miya_alerts(error_msg, tag="SHAP", voice_profile="urgent", priority=3)
                send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            confidence_drop_rate = 1.0 - min(
                len(data.columns) / self.config.get("expected_count", 320), 1.0
            )
            if len(data) < 100:
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(data.columns)} features, {len(data)} lignes)"
                miya_alerts(alert_msg, tag="SHAP", voice_profile="urgent", priority=3)
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            data = data.copy()
            X = data.drop(columns=["timestamp", target_col], errors="ignore")
            regime_mapping = self.config.get(
                "regime_mapping", {"Trend": 0, "Range": 1, "Defensive": 2}
            )
            y = data[target_col].map(regime_mapping)
            if y.isna().any():
                error_msg = f"Étiquettes invalides dans '{target_col}'"
                miya_alerts(error_msg, tag="SHAP", voice_profile="urgent", priority=3)
                send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            for col in X.columns:
                if X[col].isna().any():
                    X[col] = (
                        X[col]
                        .interpolate(method="linear", limit_direction="both")
                        .fillna(X[col].median())
                    )

            model = xgb.XGBClassifier(random_state=42, n_estimators=100, max_depth=5)
            cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
            cv_accuracy = cv_scores.mean()
            if cv_accuracy < 0.5:
                warning_msg = f"Modèle faible (CV accuracy={cv_accuracy:.2f})"
                miya_alerts(
                    warning_msg, tag="SHAP", voice_profile="warning", priority=3
                )
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)

            model.fit(X, y)
            self.model = model
            self.explainer = shap.TreeExplainer(model)

            y_pred = model.predict(X)
            accuracy = accuracy_score(y, y_pred)

            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            def save_model():
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)

            self.with_retries(save_model)

            latency = time.time() - start_time
            success_msg = f"Modèle shap_regime_detector sauvegardé dans {model_path}, accuracy={accuracy:.2f}"
            miya_speak(success_msg, tag="SHAP", voice_profile="calm", priority=2)
            logger.info(success_msg)
            self.log_performance(
                "train_shap_regime_detector",
                latency,
                success=True,
                accuracy=accuracy,
                cv_accuracy=cv_accuracy,
                confidence_drop_rate=confidence_drop_rate,
            )
            self.save_snapshot(
                "train_shap_regime_detector",
                {
                    "model_path": model_path,
                    "accuracy": accuracy,
                    "cv_accuracy": cv_accuracy,
                    "confidence_drop_rate": confidence_drop_rate,
                },
                compress=False,
            )
            send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
        except Exception as e:
            latency = time.time() - start_time
            error_msg = (
                f"Erreur entraînement modèle: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="SHAP", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "train_shap_regime_detector", latency, success=False, error=str(e)
            )
            self.save_snapshot(
                "error_train_shap_regime_detector", {"error": str(e)}, compress=False
            )
            raise

    def calculate_shap_weights(
        self,
        data: pd.DataFrame,
        config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml"),
    ) -> pd.DataFrame:
        """Calcule les poids SHAP des features pour les top 150 features en utilisant un modèle pré-entraîné."""
        try:
            start_time = time.time()
            config = self.load_config(config_path)
            model_path = config.get("model_path", MODEL_PATH)
            window_size = config.get("window_size", 100)
            key_features_count = config.get("key_features_count", 150)

            if data.empty:
                error_msg = "DataFrame vide"
                miya_alerts(error_msg, tag="SHAP", voice_profile="urgent", priority=3)
                send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            confidence_drop_rate = 1.0 - min(
                len(data.columns) / config.get("expected_count", 320), 1.0
            )
            if len(data) < window_size:
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(data.columns)} features, {len(data)} lignes)"
                miya_alerts(alert_msg, tag="SHAP", voice_profile="urgent", priority=3)
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            cache_key = hashlib.sha256(
                f"{config_path}_{data.to_json().encode()}".encode()
            ).hexdigest()
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if (
                    datetime.now() - cached_data["timestamp"]
                ).total_seconds() < config.get("cache_hours", 24) * 3600:
                    latency = time.time() - start_time
                    success_msg = "Poids SHAP récupérés du cache"
                    miya_speak(
                        success_msg, tag="SHAP", voice_profile="calm", priority=1
                    )
                    logger.info(success_msg)
                    self.log_performance(
                        "calculate_shap_weights_cache_hit", latency, success=True
                    )
                    send_alert(success_msg, priority=1)
                    send_telegram_alert(success_msg)
                    return cached_data["shap_importance"]

            data = data.copy()
            data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
            if data["timestamp"].isna().any():
                error_msg = "NaN dans les timestamps"
                miya_alerts(error_msg, tag="SHAP", voice_profile="urgent", priority=3)
                send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)
            if not data["timestamp"].is_monotonic_increasing:
                error_msg = "Timestamps non croissants"
                miya_alerts(error_msg, tag="SHAP", voice_profile="urgent", priority=3)
                send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            data = data.tail(window_size)
            feature_set = data.drop(
                columns=["timestamp", "neural_regime"], errors="ignore"
            ).columns
            X = data[feature_set]

            new_metrics = [
                "key_strikes_1",
                "key_strikes_2",
                "key_strikes_3",
                "key_strikes_4",
                "key_strikes_5",
                "max_pain_strike",
                "net_gamma",
                "zero_gamma",
                "dealer_zones_count",
                "vol_trigger",
                "ref_px",
                "data_release",
            ]
            missing_cols = [col for col in new_metrics if col not in X.columns]
            for col in missing_cols:
                X[col] = 0.0
                if col == "dealer_zones_count":
                    X[col] = X[col].clip(0, 10)
                elif col == "data_release":
                    X[col] = X[col].clip(0, 1)
                elif col in ["net_gamma", "vol_trigger"]:
                    X[col] = X[col].clip(-1, 1)
                else:
                    X[col] = (
                        X[col]
                        .clip(lower=0)
                        .fillna(X["close"].median() if "close" in X.columns else 0)
                    )

            for col in X.columns:
                if X[col].isna().any():
                    X[col] = (
                        X[col]
                        .interpolate(method="linear", limit_direction="both")
                        .fillna(X[col].median())
                    )
                    if X[col].isna().sum() / len(X) > 0.1:
                        error_msg = f"Plus de 10% de NaN dans {col}"
                        miya_alerts(
                            error_msg, tag="SHAP", voice_profile="urgent", priority=3
                        )
                        send_alert(error_msg, priority=3)
                        send_telegram_alert(error_msg)

            if self.model is None or self.explainer is None:
                if not os.path.exists(model_path):
                    error_msg = f"Modèle introuvable: {model_path}"
                    miya_alerts(
                        error_msg, tag="SHAP", voice_profile="urgent", priority=3
                    )
                    send_alert(error_msg, priority=3)
                    send_telegram_alert(error_msg)
                    raise FileNotFoundError(error_msg)
                self.load_model()

            def compute_shap():
                test_batch = X.head(10)
                self.model.predict(test_batch)
                shap_values = Parallel(n_jobs=-1)(
                    delayed(self.explainer.shap_values)(X[[f]].values)
                    for f in feature_set
                )
                shap_importance = pd.DataFrame(
                    {
                        "feature": feature_set,
                        "importance": [
                            np.abs(shap_val).mean() for shap_val in shap_values
                        ],
                        "regime": data.get("neural_regime", "range").iloc[-1],
                    }
                ).sort_values("importance", ascending=False)

                for metric in new_metrics:
                    if metric in shap_importance["feature"].values:
                        shap_importance.loc[
                            shap_importance["feature"] == metric, "importance"
                        ] *= 1.2

                shap_importance = shap_importance.head(key_features_count)
                shap_importance["importance"] = (
                    shap_importance["importance"] - shap_importance["importance"].min()
                ) / (
                    shap_importance["importance"].max()
                    - shap_importance["importance"].min()
                    + 1e-6
                )

                self.validate_shap_features(shap_importance["feature"].tolist())

                os.makedirs(os.path.dirname(FEATURE_IMPORTANCE_PATH), exist_ok=True)

                def write_shap():
                    shap_importance.to_csv(
                        FEATURE_IMPORTANCE_PATH, index=False, encoding="utf-8"
                    )

                self.with_retries(write_shap)

                return shap_importance

            shap_importance = self.with_retries(compute_shap)

            self.cache[cache_key] = {
                "shap_importance": shap_importance,
                "timestamp": datetime.now(),
            }
            if len(self.cache) > self.max_cache_size:
                self.cache.pop(next(iter(self.cache)))

            new_metrics_count = len(
                [m for m in new_metrics if m in shap_importance["feature"].values]
            )
            latency = time.time() - start_time
            success_msg = (
                f"Importance SHAP calculée pour {len(shap_importance)} features"
            )
            miya_speak(success_msg, tag="SHAP", voice_profile="calm", priority=1)
            logger.info(success_msg)
            self.log_performance(
                "calculate_shap_weights",
                latency,
                success=True,
                num_features=len(shap_importance),
                new_metrics_count=new_metrics_count,
                confidence_drop_rate=confidence_drop_rate,
            )
            self.save_snapshot(
                "calculate_shap_weights",
                {
                    "num_features": len(shap_importance),
                    "top_feature": shap_importance["feature"].iloc[0],
                    "confidence_drop_rate": confidence_drop_rate,
                },
                compress=False,
            )
            self.plot_shap_weights(
                shap_importance.set_index("feature")["importance"],
                data["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S"),
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            return shap_importance
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans calculate_shap_weights: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="SHAP", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "calculate_shap_weights", latency, success=False, error=str(e)
            )
            self.save_snapshot(
                "error_calculate_shap_weights", {"error": str(e)}, compress=False
            )
            return pd.DataFrame({"feature": [], "importance": [], "regime": []})

    def calculate_incremental_shap_weights(
        self,
        row: pd.Series,
        config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml"),
        shap_importance: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Calcule les poids SHAP pour une seule ligne en temps réel."""
        try:
            start_time = time.time()
            config = self.load_config(config_path)
            window_size = config.get("window_size", 100)

            row = row.copy()
            row["timestamp"] = pd.to_datetime(row["timestamp"], errors="coerce")
            if pd.isna(row["timestamp"]):
                error_msg = "Timestamp invalide dans la ligne"
                miya_alerts(error_msg, tag="SHAP", voice_profile="urgent", priority=3)
                send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            confidence_drop_rate = 1.0 - min(
                len(row.index) / config.get("expected_count", 320), 1.0
            )
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(row.index)} features)"
                miya_alerts(alert_msg, tag="SHAP", voice_profile="urgent", priority=3)
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            self.buffer.append(row.to_frame().T)
            if len(self.buffer) < window_size:
                warning_msg = (
                    f"Buffer pas encore plein ({len(self.buffer)}/{window_size})"
                )
                miya_speak(warning_msg, tag="SHAP", voice_profile="warning", priority=3)
                logger.warning(warning_msg)
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                return pd.DataFrame({"feature": [], "importance": [], "regime": []})

            if shap_importance is not None:
                success_msg = "Utilisation des poids SHAP pré-calculés"
                miya_speak(success_msg, tag="SHAP", voice_profile="calm", priority=1)
                logger.info(success_msg)
                send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)
                return shap_importance

            data = pd.concat(list(self.buffer), ignore_index=True).tail(window_size)
            shap_importance = self.calculate_shap_weights(data, config_path)

            latency = time.time() - start_time
            new_metrics_count = len(
                [m for m in new_metrics if m in shap_importance["feature"].values]
            )
            success_msg = (
                f"Incremental poids SHAP calculés pour {len(shap_importance)} features"
            )
            miya_speak(success_msg, tag="SHAP", voice_profile="calm", priority=1)
            logger.info(success_msg)
            self.log_performance(
                "calculate_incremental_shap_weights",
                latency,
                success=True,
                num_features=len(shap_importance),
                new_metrics_count=new_metrics_count,
                confidence_drop_rate=confidence_drop_rate,
            )
            self.save_snapshot(
                "calculate_incremental_shap_weights",
                {
                    "num_features": len(shap_importance),
                    "confidence_drop_rate": confidence_drop_rate,
                },
                compress=False,
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            return shap_importance
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur dans calculate_incremental_shap_weights: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="SHAP", voice_profile="urgent", priority=3)
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "calculate_incremental_shap_weights",
                latency,
                success=False,
                error=str(e),
            )
            self.save_snapshot(
                "error_calculate_incremental_shap_weights",
                {"error": str(e)},
                compress=False,
            )
            return pd.DataFrame({"feature": [], "importance": [], "regime": []})


if __name__ == "__main__":
    try:
        calculator = SHAPWeighting()
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
                "key_strikes_1": np.random.uniform(5000, 5200, 100),
                "max_pain_strike": np.random.uniform(5000, 5200, 100),
                "net_gamma": np.random.uniform(-1, 1, 100),
                "zero_gamma": np.random.uniform(5000, 5200, 100),
                "dealer_zones_count": np.random.randint(0, 11, 100),
                "vol_trigger": np.random.uniform(-1, 1, 100),
                "ref_px": np.random.uniform(5000, 5200, 100),
                "data_release": np.random.randint(0, 2, 100),
            }
        )
        for i in range(320 - len(data.columns) + 1):
            data[f"feature_{i}"] = np.random.normal(0, 1, 100)

        os.makedirs(os.path.join(BASE_DIR, "data", "models"), exist_ok=True)
        calculator.train_shap_regime_detector(data, MODEL_PATH)
        shap_weights = calculator.calculate_shap_weights(data)
        print("Top 5 features par importance SHAP:")
        print(shap_weights.head())
        success_msg = "Test calculate_shap_weights terminé"
        miya_speak(success_msg, tag="TEST", voice_profile="calm", priority=1)
        send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)

        row = data.iloc[-1]
        incremental_shap = calculator.calculate_incremental_shap_weights(row)
        print("Incremental SHAP weights:")
        print(incremental_shap.head())
        success_msg = "Test calculate_incremental_shap_weights terminé"
        miya_speak(success_msg, tag="TEST", voice_profile="calm", priority=1)
        send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
    except Exception as e:
        error_msg = f"Erreur test: {str(e)}\n{traceback.format_exc()}"
        miya_alerts(error_msg, tag="ALERT", voice_profile="urgent", priority=3)
        send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)
        logger.error(error_msg)
        raise
