# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/features/regime_detector.py
# Détecte les régimes de marché (bull, bear, range) avec HMM pour MIA_IA_SYSTEM_v2_2025.
#
# Version: 2.1.6
# Date: 2025-05-14
#
# Rôle : Génère une feature regime_hmm à partir des données d'order flow (bid-ask imbalance, volume)
#        pour enrichir les prédictions et décisions stratégiques.
# Utilisé par : feature_pipeline.py (génération de la feature), mia_switcher.py (décisions).
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, hmmlearn>=0.2.8,<0.3.0, psutil>=5.9.8,<6.0.0
# - scikit-learn>=1.5.0,<2.0.0, joblib>=1.3.0,<2.0.0, cachetools>=5.3.0,<6.0.0
# - pydantic>=2.0.0,<3.0.0
# - src/data/feature_pipeline.py (données d'order flow)
# - src/monitoring/prometheus_metrics.py (métriques regime_hmm, regime_transition_rate, hmm_state_distribution)
# - src/model/utils/alert_manager.py (gestion des alertes)
# - src/utils/error_tracker.py (capture des erreurs)
# - src/utils/telegram_alert.py (alertes Telegram)
#
# Inputs :
# - config/regime_detector_config.yaml (constantes)
# - data/market_memory.db (données order flow via feature_pipeline.py)
#
# Outputs :
# - data/models/hmm_ES.pkl (modèle HMM sauvegardé)
# - data/logs/regime_detector_performance.csv (logs des performances)
# - data/logs/regime_detector_errors.csv (logs critiques)
# - data/logs/hmm_transitions.csv (matrice de transition HMM)
# - data/figures/monitoring/regime_vs_vix.png (heatmap trimestriel)
#
# Notes :
# - Conforme à structure.txt (version 2.1.6, 2025-05-14).
# - Supporte la suggestion 1 (features dynamiques) en enrichissant les features.
# - Intègre retries avec jitter, logs psutil, alertes via AlertManager.
# - Cache TTLCache avec TTL adaptatif pour scalping/HFT.
# - Tests unitaires dans tests/test_regime_detector.py.
# - Distinct de detect_regime.py (routage stratégique dans src/model/router/).
# - Mise à jour pour inclure min_state_duration, window_size_adaptive, cache_ttl_adaptive, convergence_threshold.
# - Corrections : gestion de orderflow_data dans generate_heatmap, cache persistant, contrôle de convergence,
#                méthode predict_series, bins VIX, histogramme Prometheus pour latence.

import os
import random
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import psutil
from cachetools import TTLCache
from hmmlearn.hmm import GaussianHMM
from loguru import logger
from pydantic import BaseModel, PositiveFloat, PositiveInt, conint, constr
from sklearn.preprocessing import StandardScaler

from src.model.utils.alert_manager import AlertManager
from src.monitoring.prometheus_metrics import Gauge, Histogram
from src.utils.error_tracker import capture_error

# Configuration logging
logger.remove()
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOG_DIR = BASE_DIR / "data" / "logs"
MODEL_DIR = BASE_DIR / "data" / "models"
FIGURES_DIR = BASE_DIR / "data" / "figures" / "monitoring"
LOG_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "regime_detector.log", rotation="10 MB", level="INFO", encoding="utf-8"
)
logger.add(
    LOG_DIR / "regime_detector_errors.csv",
    rotation="10 MB",
    level="ERROR",
    encoding="utf-8",
)
logger.add(lambda msg: print(msg, end=""), level="INFO")

# Métriques Prometheus
regime_hmm_state = Gauge(
    "regime_hmm_state",
    "État du régime de marché détecté par HMM (0: bull, 1: bear, 2: range)",
    ["market"],
)
regime_transition_rate = Gauge(
    "regime_transition_rate", "Taux de transition entre régimes par heure", ["market"]
)
hmm_state_distribution = Gauge(
    "hmm_state_distribution",
    "Répartition des états HMM (0: bull, 1: bear, 2: range)",
    ["market", "state"],
)
detect_regime_latency = Histogram(
    "regime_detector_latency_seconds",
    "Latence des appels à detect_regime en secondes",
    ["market"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
)


class RegimeConfig(BaseModel):
    """Modèle Pydantic pour valider regime_detector_config.yaml."""

    buffer_size: conint(ge=10, le=1000)
    critical_buffer_size: conint(ge=5, le=50)
    max_retries: PositiveInt
    retry_delay: PositiveFloat
    n_iterations: conint(ge=5, le=50)
    convergence_threshold: PositiveFloat
    covariance_type: constr(pattern="^(full|diag|tied|spherical)$")
    n_components: conint(ge=2, le=5)
    window_size: conint(ge=10, le=500)
    window_size_adaptive: bool
    random_state: PositiveInt
    use_random_state: bool
    min_train_rows: conint(ge=50)
    min_state_duration: conint(ge=1, le=20)
    cache_ttl_seconds: conint(ge=10, le=3600)
    cache_ttl_adaptive: bool
    prometheus_labels: Dict[str, str]


class RegimeDetector:
    """
    Détecte les régimes de marché (bull, bear, range) avec HMM.

    Attributes:
        market (str): Marché cible (ex. : 'ES').
        log_buffer (list): Buffer pour logs de performance.
        error_buffer (list): Buffer pour logs critiques.
        config (dict): Configuration validée via Pydantic.
        hmm_model (GaussianHMM): Modèle HMM entraîné.
        scaler (StandardScaler): Scaler pour données d'order flow.
        feature_cache (TTLCache): Cache des régimes calculés.
        data_buffer (deque): Buffer roulant pour données.
        alert_manager (AlertManager): Gestionnaire d'alertes.
        last_regime (int): Dernier régime détecté.
        transitions (list): Historique des transitions.
        current_state_duration (int): Durée du régime actuel.
        state_counts (dict): Répartition des états.
    """

    def __init__(self, market: str = "ES"):
        """Initialise le détecteur de régimes."""
        start_time = datetime.now()
        try:
            self.market = market
            self.log_buffer = []
            self.error_buffer = []
            self.config = None
            self.hmm_model = None
            self.scaler = StandardScaler()
            self.feature_cache = None
            self.data_buffer = None
            self.alert_manager = AlertManager()
            self.last_regime = 2  # Range par défaut
            self.transitions = []
            self.current_state_duration = 0
            self.state_counts = {i: 0 for i in range(3)}  # Pour 3 états
            from src.model.utils.config_manager import get_config

            config_raw = self.with_retries(
                lambda: get_config(BASE_DIR / "config/regime_detector_config.yaml")
            )
            self.config = RegimeConfig(**config_raw).dict()
            self.feature_cache = TTLCache(
                maxsize=100, ttl=self.config["cache_ttl_seconds"]
            )
            self.data_buffer = deque(maxlen=self.config["window_size"])
            LOG_DIR.mkdir(exist_ok=True)
            MODEL_DIR.mkdir(exist_ok=True)
            FIGURES_DIR.mkdir(exist_ok=True)
            self._load_model()
            success_msg = f"RegimeDetector initialisé pour {market} avec {self.config['n_components']} états HMM"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=2)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("init", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur initialisation RegimeDetector: {str(e)}"
            logger.error(error_msg)
            capture_error(
                e,
                context={"market": market},
                market=market,
                operation="init_regime_detector",
            )
            self.alert_manager.send_alert(error_msg, priority=5)
            self.log_performance("init", 0, success=False, error=str(e))
            raise

    def _load_model(self) -> None:
        """Charge un modèle HMM sauvegardé s'il existe."""
        model_path = MODEL_DIR / f"hmm_{self.market}.pkl"
        if model_path.exists():
            try:
                self.hmm_model = joblib.load(model_path)
                success_msg = f"Modèle HMM chargé depuis {model_path}"
                logger.info(success_msg)
                self.alert_manager.send_alert(success_msg, priority=2)
            except Exception as e:
                error_msg = f"Erreur chargement modèle HMM: {str(e)}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=4)
                self.hmm_model = None

    def with_retries(
        self, func: callable, max_attempts: int = None, delay_base: float = None
    ) -> Optional[any]:
        """Exécute une fonction avec retries exponentiels et jitter."""
        start_time = datetime.now()
        max_attempts = max_attempts or self.config["max_retries"]
        delay_base = delay_base or self.config["retry_delay"]
        for attempt in range(max_attempts):
            try:
                result = func()
                latency = (datetime.now() - start_time).total_seconds()
                self.log_performance(
                    f"retry_attempt_{attempt+1}",
                    latency,
                    success=True,
                    attempt_number=attempt + 1,
                )
                return result
            except Exception as e:
                if attempt == max_attempts - 1:
                    error_msg = f"Échec après {max_attempts} tentatives: {str(e)}"
                    logger.error(error_msg)
                    capture_error(
                        e,
                        context={"market": self.market},
                        market=self.market,
                        operation="retry_regime_detector",
                    )
                    self.alert_manager.send_alert(error_msg, priority=4)
                    self.log_performance(
                        f"retry_attempt_{attempt+1}",
                        0,
                        success=False,
                        error=str(e),
                        attempt_number=attempt + 1,
                    )
                    raise
                delay = delay_base * (2**attempt) * (1 + random.uniform(-0.2, 0.2))
                logger.warning(
                    f"Tentative {attempt+1} échouée, retry après {delay:.2f}s"
                )
                time.sleep(delay)
        return None

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ) -> None:
        """Journalise les performances des opérations critiques."""
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_usage = psutil.cpu_percent()
            if memory_usage > 1024:
                alert_msg = f"ALERTE: Usage mémoire élevé ({memory_usage:.2f} MB)"
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=5)
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
            buffer = self.error_buffer if not success and error else self.log_buffer
            buffer_size = (
                self.config["critical_buffer_size"]
                if buffer is self.error_buffer
                else self.config["buffer_size"]
            )
            if len(buffer) >= buffer_size:
                log_df = pd.DataFrame(buffer)
                log_path = LOG_DIR / (
                    "regime_detector_errors.csv"
                    if buffer is self.error_buffer
                    else "regime_detector_performance.csv"
                )

                def save_log():
                    if not log_path.exists():
                        log_df.to_csv(log_path, index=False, encoding="utf-8")
                    else:
                        log_df.to_csv(
                            log_path,
                            mode="a",
                            header=False,
                            index=False,
                            encoding="utf-8",
                        )

                self.with_retries(save_log)
                buffer.clear()
            logger.info(f"Performance journalisée pour {operation}. CPU: {cpu_usage}%")
        except Exception as e:
            error_msg = f"Erreur journalisation performance: {str(e)}"
            logger.error(error_msg)
            capture_error(
                e,
                context={"market": self.market},
                market=self.market,
                operation="log_performance",
            )
            self.alert_manager.send_alert(error_msg, priority=3)

    def update_cache_ttl(self, vix: float = None) -> None:
        """Ajuste le TTL du cache en fonction de la volatilité."""
        try:
            cache_ttl = self.config["cache_ttl_seconds"]
            if self.config["cache_ttl_adaptive"] and vix is not None and vix > 25:
                cache_ttl = 60
            if self.feature_cache is None or self.feature_cache.ttl != cache_ttl:
                self.feature_cache = TTLCache(maxsize=100, ttl=cache_ttl)
            logger.debug(f"Cache TTL ajusté à {cache_ttl} secondes")
        except Exception as e:
            error_msg = f"Erreur lors de l'ajustement du cache TTL: {str(e)}"
            logger.error(error_msg)
            capture_error(
                e,
                context={"market": self.market},
                market=self.market,
                operation="update_cache_ttl",
            )
            self.alert_manager.send_alert(error_msg, priority=3)

    def train_hmm(self, orderflow_data: pd.DataFrame) -> None:
        """
        Entraîne le modèle HMM sur les données d'order flow.

        Args:
            orderflow_data (pd.DataFrame): Données avec colonnes (bid_ask_imbalance, total_volume).
        """
        start_time = datetime.now()
        try:
            required_cols = ["bid_ask_imbalance", "total_volume"]
            missing_cols = [
                col for col in required_cols if col not in orderflow_data.columns
            ]
            if missing_cols:
                raise ValueError(
                    f"Colonnes manquantes dans orderflow_data: {missing_cols}"
                )
            if len(orderflow_data) < self.config["min_train_rows"]:
                raise ValueError(
                    f"Données insuffisantes pour entraînement HMM: {len(orderflow_data)} lignes"
                )

            # Préparer et scaler les données
            X = (
                orderflow_data[required_cols]
                .ffill()
                .fillna(orderflow_data[required_cols].median())
                .values
            )
            X_scaled = self.scaler.fit_transform(X)

            # Entraîner le modèle HMM
            self.hmm_model = GaussianHMM(
                n_components=self.config["n_components"],
                covariance_type=self.config["covariance_type"],
                n_iter=self.config["n_iterations"],
                tol=self.config["convergence_threshold"],
                random_state=(
                    self.config["random_state"]
                    if self.config["use_random_state"]
                    else None
                ),
            )
            with joblib.parallel_backend("loky"):
                self.hmm_model.fit(X_scaled)

            # Vérifier la convergence
            converged = (
                getattr(self.hmm_model, "monitor_", None)
                and self.hmm_model.monitor_.converged
            )
            log_prob = self.hmm_model.score(X_scaled)
            logger.info(
                f"Entraînement HMM - Converged: {converged}, Log-likelihood: {log_prob:.2f}"
            )

            # Sauvegarder le modèle
            model_path = MODEL_DIR / f"hmm_{self.market}.pkl"
            joblib.dump(self.hmm_model, model_path)

            # Exporter la matrice de transition
            trans_df = pd.DataFrame(
                self.hmm_model.transmat_,
                columns=[f"to_state_{i}" for i in range(self.config["n_components"])],
            )
            trans_df.index = [
                f"from_state_{i}" for i in range(self.config["n_components"])
            ]
            trans_path = LOG_DIR / "hmm_transitions.csv"
            trans_df.to_csv(trans_path, encoding="utf-8")

            # Journaliser la performance
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "train_hmm",
                latency,
                success=True,
                num_rows=len(orderflow_data),
                n_components=self.config["n_components"],
                converged=converged,
                log_likelihood=log_prob,
            )
            success_msg = f"Modèle HMM entraîné et sauvegardé pour {self.market} avec {len(orderflow_data)} lignes"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=2)
        except Exception as e:
            error_msg = f"Erreur entraînement HMM pour {self.market}: {str(e)}"
            logger.error(error_msg)
            capture_error(
                e,
                context={"market": self.market},
                market=self.market,
                operation="train_hmm",
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            self.log_performance("train_hmm", 0, success=False, error=str(e))
            raise

    def predict_series(self, orderflow_data: pd.DataFrame) -> List[int]:
        """
        Prédit les régimes pour une série complète de données.

        Args:
            orderflow_data (pd.DataFrame): Données avec colonnes (bid_ask_imbalance, total_volume).

        Returns:
            List[int]: Liste des régimes prédits (0: bull, 1: bear, 2: range).
        """
        start_time = datetime.now()
        try:
            if self.hmm_model is None:
                raise ValueError("Modèle HMM non entraîné")

            required_cols = ["bid_ask_imbalance", "total_volume"]
            missing_cols = [
                col for col in required_cols if col not in orderflow_data.columns
            ]
            if missing_cols:
                raise ValueError(
                    f"Colonnes manquantes dans orderflow_data: {missing_cols}"
                )

            data = (
                orderflow_data[required_cols]
                .ffill()
                .fillna(orderflow_data[required_cols].median())
            )
            X = data.values
            X_scaled = self.scaler.transform(X)
            regimes = self.hmm_model.predict(X_scaled).tolist()

            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "predict_series", latency, success=True, num_rows=len(orderflow_data)
            )
            return regimes
        except Exception as e:
            error_msg = f"Erreur prédiction série pour {self.market}: {str(e)}"
            logger.error(error_msg)
            capture_error(
                e,
                context={"market": self.market},
                market=self.market,
                operation="predict_series",
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            self.log_performance("predict_series", 0, success=False, error=str(e))
            return [2] * len(orderflow_data)  # Range par défaut

    def detect_regime(self, orderflow_data: pd.DataFrame, vix: float = None) -> int:
        """
        Détecte le régime de marché avec le modèle HMM.

        Args:
            orderflow_data (pd.DataFrame): Données avec colonnes (bid_ask_imbalance, total_volume).
            vix (float): Valeur du VIX pour ajustement adaptatif.

        Returns:
            int: État du régime (0: bull, 1: bear, 2: range).
        """
        start_time = datetime.now()
        try:
            if self.hmm_model is None:
                raise ValueError("Modèle HMM non entraîné")

            required_cols = ["bid_ask_imbalance", "total_volume"]
            missing_cols = [
                col for col in required_cols if col not in orderflow_data.columns
            ]
            if missing_cols:
                raise ValueError(
                    f"Colonnes manquantes dans orderflow_data: {missing_cols}"
                )

            # Ajustement adaptatif
            window_size = self.config["window_size"]
            if self.config["window_size_adaptive"] and vix is not None and vix > 25:
                window_size = 20
            self.update_cache_ttl(vix)
            self.data_buffer = deque(maxlen=window_size)

            # Préparer les données
            data = (
                orderflow_data[required_cols]
                .ffill()
                .fillna(data[required_cols].median())
            )
            X = data.tail(window_size).values
            X_scaled = self.scaler.transform(X)

            # Mettre à jour le buffer
            for row in X:
                self.data_buffer.append(row)
            np.array(self.data_buffer)

            # Vérifier le cache
            cache_key = hash(tuple(X[-1]))
            if cache_key in self.feature_cache:
                regime = self.feature_cache[cache_key]
                logger.debug(f"Cache hit pour régime: {regime}")
            else:
                # Détecter le régime
                regime = self.hmm_model.predict(X_scaled)[-1]
                # Appliquer min_state_duration
                if (
                    regime != self.last_regime
                    and self.current_state_duration < self.config["min_state_duration"]
                ):
                    regime = self.last_regime
                else:
                    self.current_state_duration = (
                        1
                        if regime != self.last_regime
                        else self.current_state_duration + 1
                    )
                    self.feature_cache[cache_key] = regime

            # Mettre à jour les transitions et distribution
            self.state_counts[regime] += 1
            total_counts = sum(self.state_counts.values())
            for state in self.state_counts:
                hmm_state_distribution.labels(market=self.market, state=str(state)).set(
                    self.state_counts[state] / total_counts if total_counts > 0 else 0
                )
            if regime != self.last_regime:
                self.transitions.append(datetime.now())
                self.last_regime = regime
            transition_rate = (
                len(self.transitions)
                / max(1, (datetime.now() - self.transitions[0]).total_seconds() / 3600)
                if self.transitions
                else 0.0
            )

            # Enregistrer les métriques Prometheus
            regime_hmm_state.labels(market=self.market).set(regime)
            regime_transition_rate.labels(market=self.market).set(transition_rate)

            # Journaliser la performance
            latency = (datetime.now() - start_time).total_seconds()
            detect_regime_latency.labels(market=self.market).observe(latency)
            self.log_performance(
                "detect_regime",
                latency,
                success=True,
                num_rows=len(orderflow_data),
                regime=regime,
                transition_rate=transition_rate,
            )

            return regime
        except Exception as e:
            error_msg = f"Erreur détection régime pour {self.market}: {str(e)}"
            logger.error(error_msg)
            capture_error(
                e,
                context={"market": self.market},
                market=self.market,
                operation="detect_regime",
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            latency = (datetime.now() - start_time).total_seconds()
            detect_regime_latency.labels(market=self.market).observe(latency)
            self.log_performance("detect_regime", latency, success=False, error=str(e))
            return 2  # Range par défaut

    def generate_heatmap(
        self, orderflow_data: pd.DataFrame, vix_data: pd.DataFrame
    ) -> None:
        """
        Génère un heatmap trimestriel régime vs VIX.

        Args:
            orderflow_data (pd.DataFrame): Données avec colonnes (bid_ask_imbalance, total_volume).
            vix_data (pd.DataFrame): Données avec colonne vix.
        """
        start_time = datetime.now()
        try:
            if "vix" not in vix_data.columns:
                raise ValueError("Colonne vix manquante dans vix_data")
            regimes = self.predict_series(orderflow_data)
            import matplotlib.pyplot as plt
            import seaborn as sns

            heatmap_data = pd.DataFrame({"regime": regimes, "vix": vix_data["vix"]})
            heatmap_data["vix_bin"] = pd.cut(
                heatmap_data["vix"], bins=[10, 15, 20, 25, 30, 40]
            )
            pivot = heatmap_data.pivot_table(
                index="regime", columns="vix_bin", aggfunc="size", fill_value=0
            )
            sns.heatmap(pivot, cmap="Blues")
            plt.savefig(
                FIGURES_DIR / f"regime_vs_vix_{datetime.now().strftime('%Y%m')}.png"
            )
            plt.close()
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("generate_heatmap", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur génération heatmap pour {self.market}: {str(e)}"
            logger.error(error_msg)
            capture_error(
                e,
                context={"market": self.market},
                market=self.market,
                operation="generate_heatmap",
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            self.log_performance("generate_heatmap", 0, success=False, error=str(e))
