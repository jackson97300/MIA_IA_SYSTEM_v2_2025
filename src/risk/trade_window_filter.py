# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/risk/trade_window_filter.py
# Rôle : Filtre les moments opportuns pour trader en fonction des conditions macro et de volatilité pour MIA_IA_SYSTEM_v2_2025.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.0,<6.0.0, pyyaml>=6.0.0,<7.0.0, logging, os, json,
#   datetime, signal, gzip
# - src/model/adaptive_learner.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
# - src/trading/shap_weighting.py
#
# Inputs :
# - Données de trading (pd.DataFrame avec 350 features pour entraînement ou 150 SHAP features pour inférence)
# - Configuration via config/market_config.yaml
# - macro_events.csv pour les événements macroéconomiques
#
# Outputs :
# - Logs dans data/logs/risk/trade_window_filter.log
# - Logs de performance dans data/logs/trade_window_performance.csv
# - Snapshots JSON dans data/risk/trade_window_snapshots/*.json (option *.json.gz)
# - Dashboard JSON dans data/risk/trade_window_dashboard.json
#
# Notes :
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence) définies dans config/feature_sets.yaml.
# - Utilise IQFeed exclusivement via data_provider.py pour les données d’entrée.
# - Supprime toute référence à 320 features, 81 features, et obs_t.
# - Intègre la volatilité (méthode 1) via vix_es_correlation.
# - Implémente retries (max 3, délai 2^attempt secondes) pour les opérations critiques.
# - Intègre confidence_drop_rate (Phase 8) pour l’auto-conscience, basé sur window_score.
# - Intègre l’analyse SHAP (Phase 17) pour évaluer l’impact des features sur window_score, limitée à 50 features.
# - Sauvegarde des snapshots JSON non compressés par défaut avec option de compression gzip.
# - Utilise AlertManager et telegram_alert pour les notifications critiques.
# - Tests unitaires disponibles dans tests/test_trade_window_filter.py.

import gzip
import json
import logging
import os
import signal
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import psutil

from src.model.adaptive_learner import store_pattern
from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import config_manager
from src.trading.shap_weighting import calculate_shap
from src.utils.telegram_alert import send_telegram_alert

# Configuration du logging
os.makedirs("data/logs/risk", exist_ok=True)
logging.basicConfig(
    filename="data/logs/risk/trade_window_filter.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Constantes
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Secondes, exponentiel
PERF_LOG_PATH = Path("data/logs/trade_window_performance.csv")
SNAPSHOT_DIR = Path("data/risk/trade_window_snapshots")
SHAP_FEATURE_LIMIT = 50


class TradeWindowFilter:
    """
    Classe pour filtrer les moments opportuns pour trader en fonction des conditions macro et de volatilité.
    """

    def __init__(
        self,
        config_path: str = "config/market_config.yaml",
        macro_events_path: str = "data/news/macro_events.csv",
    ):
        """
        Initialise le filtre de fenêtre de trading.

        Args:
            config_path (str): Chemin vers la configuration du marché.
            macro_events_path (str): Chemin vers les événements macroéconomiques.
        """
        self.alert_manager = AlertManager()
        SNAPSHOT_DIR.mkdir(exist_ok=True)
        PERF_LOG_PATH.parent.mkdir(exist_ok=True)
        signal.signal(signal.SIGINT, self.handle_sigint)

        try:
            # Charger la configuration
            config = config_manager.get_config(config_path).get(
                "trade_window_filter", {}
            )
            self.thresholds = {
                "event_timing_threshold": config.get(
                    "event_timing_threshold", 3600
                ),  # 1h en secondes
                "event_frequency_threshold": config.get(
                    "event_frequency_threshold", 3
                ),  # Max 3 événements/24h
                "vix_threshold": config.get("vix_threshold", 30),
                "vix_es_correlation_threshold": config.get(
                    "vix_es_correlation_threshold", 25.0
                ),
                "event_impact_threshold": config.get("event_impact_threshold", 0.5),
                "max_trades_per_hour": config.get("max_trades_per_hour", 10),
                "window_score_threshold": config.get("window_score_threshold", 0.7),
                "macro_score_threshold": config.get("macro_score_threshold", 0.8),
                "min_confidence": config.get(
                    "min_confidence", 0.7
                ),  # Pour confidence_drop_rate
            }
            self.score_weights = {
                "vix_weight": config.get("vix_weight", 0.3),
                "event_impact_weight": config.get("event_impact_weight", 0.3),
                "event_proximity_weight": config.get("event_proximity_weight", 0.2),
                "event_frequency_weight": config.get("event_frequency_weight", 0.1),
                "trade_frequency_weight": config.get("trade_frequency_weight", 0.1),
            }

            # Valider les seuils
            for key, value in self.thresholds.items():
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Seuil invalide pour {key}: {value}")
                if key != "event_timing_threshold" and value <= 0:
                    raise ValueError(f"Seuil {key} doit être positif: {value}")

            # Charger les événements macro
            self.macro_events_path = macro_events_path
            self.macro_events = self.load_macro_events()

            # État interne
            self.trade_timestamps = (
                []
            )  # Historique des trades pour calculer la fréquence
            self.block_events = []  # Historique des blocages
            self.window_score = 0.0  # Score de la fenêtre actuelle
            self.log_buffer = []  # Buffer pour écritures CSV
            self.last_vix = 0.0  # Pour ajuster max_trades_per_hour

            logger.info("TradeWindowFilter initialisé avec succès")
            self.alert_manager.send_alert(
                "TradeWindowFilter initialisé avec succès", priority=2
            )
            send_telegram_alert("TradeWindowFilter initialisé avec succès")
            self.log_performance("init", 0, success=True)
        except Exception as e:
            self.alert_manager.send_alert(
                f"Erreur initialisation TradeWindowFilter: {str(e)}", priority=5
            )
            send_telegram_alert(f"Erreur initialisation TradeWindowFilter: {str(e)}")
            logger.error(f"Erreur initialisation TradeWindowFilter: {str(e)}")
            self.log_performance("init", 0, success=False, error=str(e))
            raise

    def handle_sigint(self, signum, frame):
        """Gère l'arrêt propre sur SIGINT."""
        snapshot = {"timestamp": datetime.now().isoformat(), "status": "SIGINT"}
        try:
            self.save_snapshot("sigint", snapshot, compress=False)
            logger.info("Arrêt propre sur SIGINT, snapshot sauvegardé")
            self.alert_manager.send_alert(
                "Arrêt propre sur SIGINT, snapshot sauvegardé", priority=2
            )
            send_telegram_alert("Arrêt propre sur SIGINT, snapshot sauvegardé")
        except Exception as e:
            logger.error(f"Erreur sauvegarde snapshot SIGINT: {str(e)}")
            self.alert_manager.send_alert(
                f"Erreur sauvegarde snapshot SIGINT: {str(e)}", priority=3
            )
            send_telegram_alert(f"Erreur sauvegarde snapshot SIGINT: {str(e)}")
        exit(0)

    def with_retries(
        self,
        func: callable,
        max_attempts: int = MAX_RETRIES,
        delay_base: float = RETRY_DELAY_BASE,
    ) -> Optional[Any]:
        """
        Exécute une fonction avec retries exponentiels.

        Args:
            func (callable): Fonction à exécuter.
            max_attempts (int): Nombre maximum de tentatives.
            delay_base (float): Base pour le délai exponentiel.

        Returns:
            Optional[Any]: Résultat de la fonction ou None si échec.
        """
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
                    self.alert_manager.send_alert(
                        f"Échec après {max_attempts} tentatives: {str(e)}", priority=3
                    )
                    send_telegram_alert(
                        f"Échec après {max_attempts} tentatives: {str(e)}"
                    )
                    logger.error(f"Échec après {max_attempts} tentatives: {str(e)}")
                    return None
                delay = delay_base**attempt
                logger.warning(f"Tentative {attempt+1} échouée, retry après {delay}s")
                time.sleep(delay)

    def log_performance(
        self,
        operation: str,
        latency: float,
        success: bool = True,
        error: str = None,
        **kwargs,
    ) -> None:
        """
        Enregistre les performances (CPU, mémoire, latence) dans trade_window_performance.csv.

        Args:
            operation (str): Nom de l’opération.
            latency (float): Temps d’exécution en secondes.
            success (bool): Indique si l’opération a réussi.
            error (str, optional): Message d’erreur si applicable.
            **kwargs: Paramètres supplémentaires.
        """
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
            cpu_percent = psutil.cpu_percent()
            if memory_usage > 1024:
                self.alert_manager.send_alert(
                    f"ALERTE: Utilisation mémoire élevée ({memory_usage:.2f} MB)",
                    priority=5,
                )
                send_telegram_alert(
                    f"ALERTE: Utilisation mémoire élevée ({memory_usage:.2f} MB)"
                )
                logger.warning(f"Utilisation mémoire élevée: {memory_usage:.2f} MB")
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "latency": latency,
                "success": success,
                "error": error,
                "memory_usage_mb": memory_usage,
                "cpu_percent": cpu_percent,
                **kwargs,
            }
            os.makedirs(os.path.dirname(PERF_LOG_PATH), exist_ok=True)
            log_df = pd.DataFrame([log_entry])

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
            logger.info(
                f"Performance journalisée pour {operation}. CPU: {cpu_percent}%"
            )
        except Exception as e:
            self.alert_manager.send_alert(
                f"Erreur journalisation performance: {str(e)}", priority=3
            )
            send_telegram_alert(f"Erreur journalisation performance: {str(e)}")
            logger.error(f"Erreur journalisation performance: {str(e)}")

    def load_macro_events(self) -> pd.DataFrame:
        """
        Charge les événements macroéconomiques depuis macro_events.csv.

        Returns:
            pd.DataFrame: Données des événements macro.
        """
        start_time = time.time()
        try:

            def load():
                if not os.path.exists(self.macro_events_path):
                    logger.warning(
                        "Fichier macro_events.csv introuvable, création d’un DataFrame vide"
                    )
                    return pd.DataFrame(
                        columns=["timestamp", "event_type", "impact_score"]
                    )
                data = pd.read_csv(self.macro_events_path, encoding="utf-8")
                data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
                if data["timestamp"].isna().any():
                    logger.warning(
                        "Valeurs invalides dans timestamp de macro_events, remplacées par défaut"
                    )
                    data["timestamp"] = data["timestamp"].fillna(pd.Timestamp.now())
                required_cols = ["timestamp", "event_type", "impact_score"]
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    raise ValueError(
                        f"Colonnes manquantes dans macro_events.csv: {missing_cols}"
                    )
                if (
                    data["impact_score"].isnull().any()
                    or data["impact_score"].le(0).any()
                    or data["impact_score"].gt(1).any()
                ):
                    raise ValueError(
                        "impact_score invalide dans macro_events.csv: doit être 0 < impact_score ≤ 1"
                    )
                if not data["event_type"].str.isalpha().all():
                    raise ValueError(
                        "event_type invalide dans macro_events.csv: doit contenir uniquement des lettres"
                    )
                return data

            data = self.with_retries(load)
            logger.info(f"Événements macro chargés: {len(data)} lignes")
            self.alert_manager.send_alert(
                f"Événements macro chargés: {len(data)} lignes", priority=1
            )
            send_telegram_alert(f"Événements macro chargés: {len(data)} lignes")
            self.log_performance(
                "load_macro_events",
                time.time() - start_time,
                success=True,
                num_events=len(data),
            )
            self.save_snapshot(
                "load_macro_events", {"num_events": len(data)}, compress=False
            )
            return data
        except Exception as e:
            self.alert_manager.send_alert(
                f"Erreur chargement macro_events: {str(e)}", priority=4
            )
            send_telegram_alert(f"Erreur chargement macro_events: {str(e)}")
            logger.error(f"Erreur chargement macro_events: {str(e)}")
            self.log_performance(
                "load_macro_events",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            return pd.DataFrame(columns=["timestamp", "event_type", "impact_score"])

    def validate_data(self, data: pd.DataFrame) -> None:
        """
        Valide les données d’entrée (350 features pour entraînement, 150 SHAP pour inférence).

        Args:
            data (pd.DataFrame): Données à valider.

        Raises:
            ValueError: Si les données sont invalides.
        """
        start_time = time.time()
        try:
            expected_dims = self.thresholds.get(
                "observation_dims", {"training": 350, "inference": 150}
            )
            data_dim = data.shape[1]
            if data_dim not in (expected_dims["training"], expected_dims["inference"]):
                raise ValueError(
                    f"Nombre de features incorrect: {data_dim}, attendu {expected_dims['training']} ou {expected_dims['inference']}"
                )
            critical_cols = [
                "vix",
                "vix_es_correlation",
                "event_volatility_impact",
                "event_timing_proximity",
                "event_frequency_24h",
                "trade_frequency_1s",
            ]
            for col in critical_cols:
                if col in data.columns:
                    if col != "event_timing_proximity" and (
                        data[col].isnull().any() or data[col].le(0).any()
                    ):
                        raise ValueError(
                            f"Colonne {col} contient des NaN ou des valeurs non positives"
                        )
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        raise ValueError(
                            f"Colonne {col} n'est pas numérique: {data[col].dtype}"
                        )
            if "timestamp" in data.columns:
                data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
                if data["timestamp"].isna().any():
                    last_valid = (
                        data["timestamp"].dropna().iloc[-1]
                        if not data["timestamp"].dropna().empty
                        else pd.Timestamp.now()
                    )
                    data["timestamp"] = data["timestamp"].fillna(last_valid)
                    self.alert_manager.send_alert(
                        f"Valeurs timestamp invalides imputées avec {last_valid}",
                        priority=2,
                    )
                    send_telegram_alert(
                        f"Valeurs timestamp invalides imputées avec {last_valid}"
                    )
                latest_timestamp = data["timestamp"].iloc[-1]
                if not isinstance(latest_timestamp, pd.Timestamp):
                    raise ValueError(f"Timestamp non valide: {latest_timestamp}")
                if latest_timestamp > datetime.now() + timedelta(
                    minutes=5
                ) or latest_timestamp < datetime.now() - timedelta(hours=24):
                    raise ValueError(f"Timestamp hors plage: {latest_timestamp}")
            logger.debug("Données validées avec succès")
            self.log_performance(
                "validate_data",
                time.time() - start_time,
                success=True,
                num_features=data_dim,
            )
        except Exception as e:
            self.alert_manager.send_alert(
                f"Erreur validation données: {str(e)}", priority=4
            )
            send_telegram_alert(f"Erreur validation données: {str(e)}")
            logger.error(f"Erreur validation données: {str(e)}")
            self.log_performance(
                "validate_data", time.time() - start_time, success=False, error=str(e)
            )
            raise

    def save_snapshot(
        self, snapshot_type: str, data: Dict, compress: bool = False
    ) -> None:
        """
        Sauvegarde un instantané JSON des résultats, avec option de compression gzip.

        Args:
            snapshot_type (str): Type de snapshot (ex. : trade_window_filter).
            data (Dict): Données à sauvegarder.
            compress (bool): Si True, compresse en gzip.
        """
        start_time = time.time()
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = {"timestamp": timestamp, "type": snapshot_type, "data": data}
            snapshot_path = SNAPSHOT_DIR / f"snapshot_{snapshot_type}_{timestamp}.json"
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
            latency = time.time() - start_time
            self.log_performance("save_snapshot", latency, success=True)
            logger.info(f"Snapshot {snapshot_type} sauvegardé: {save_path}")
            self.alert_manager.send_alert(
                f"Snapshot {snapshot_type} sauvegardé: {save_path}", priority=1
            )
            send_telegram_alert(f"Snapshot {snapshot_type} sauvegardé: {save_path}")
        except Exception as e:
            self.log_performance("save_snapshot", 0, success=False, error=str(e))
            self.alert_manager.send_alert(
                f"Erreur sauvegarde snapshot: {str(e)}", priority=3
            )
            send_telegram_alert(f"Erreur sauvegarde snapshot: {str(e)}")
            logger.error(f"Erreur sauvegarde snapshot: {str(e)}")

    def calculate_shap_window(
        self, data: pd.DataFrame, target: str = "window_score"
    ) -> Optional[pd.DataFrame]:
        """
        Calcule les valeurs SHAP pour le score de fenêtre (Phase 17).

        Args:
            data (pd.DataFrame): Données d’entrée avec les features.
            target (str): Métrique cible pour SHAP (ex. : window_score).

        Returns:
            Optional[pd.DataFrame]: DataFrame des valeurs SHAP ou None si échec.
        """
        start_time = time.time()
        try:
            shap_values = calculate_shap(
                data, target=target, max_features=SHAP_FEATURE_LIMIT
            )
            logger.info(f"Calcul SHAP terminé pour {target}")
            self.log_performance(
                "calculate_shap_window",
                time.time() - start_time,
                success=True,
                target=target,
            )
            return shap_values
        except Exception as e:
            self.alert_manager.send_alert(
                f"Erreur calcul SHAP pour {target}: {str(e)}", priority=3
            )
            send_telegram_alert(f"Erreur calcul SHAP pour {target}: {str(e)}")
            logger.error(f"Erreur calcul SHAP pour {target}: {str(e)}")
            self.log_performance(
                "calculate_shap_window",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            return None

    def calculate_window_score(self, data: pd.DataFrame) -> float:
        """
        Calcule un score de fenêtre de trading basé sur les conditions macro et de trading.

        Args:
            data (pd.DataFrame): Données contenant les features.

        Returns:
            float: Score de la fenêtre (0 à 1, 1 étant optimal).
        """
        start_time = time.time()
        try:
            self.validate_data(data)
            current_data = data.iloc[-1]

            # Utiliser vix_es_correlation si disponible, sinon vix
            vix = current_data.get("vix_es_correlation", current_data.get("vix", 0))
            event_impact = current_data.get("event_volatility_impact", 0)
            event_proximity = current_data.get("event_timing_proximity", 7200)
            event_frequency = current_data.get("event_frequency_24h", 0)
            trade_frequency = current_data.get("trade_frequency_1s", 0)

            # Mettre à jour last_vix pour prevent_overtrading
            self.last_vix = vix

            # Calcul du score avec poids configurables
            score = (
                self.score_weights["vix_weight"]
                * (1 - min(vix / self.thresholds["vix_es_correlation_threshold"], 1.0))
                + self.score_weights["event_impact_weight"]
                * (
                    1
                    - min(event_impact / self.thresholds["event_impact_threshold"], 1.0)
                )
                + self.score_weights["event_proximity_weight"]
                * (
                    min(
                        event_proximity / self.thresholds["event_timing_threshold"], 1.0
                    )
                )
                + self.score_weights["event_frequency_weight"]
                * (
                    1
                    - min(
                        event_frequency / self.thresholds["event_frequency_threshold"],
                        1.0,
                    )
                )
                + self.score_weights["trade_frequency_weight"]
                * (1 - min(trade_frequency / 10.0, 1.0))
            )
            score = min(1.0, max(0.0, score))

            # Calculer confidence_drop_rate (Phase 8)
            confidence_drop_rate = 1.0 - min(
                score / self.thresholds["window_score_threshold"], 1.0
            )
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f}"
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            # Analyse SHAP (Phase 17)
            shap_metrics = {}
            shap_values = self.calculate_shap_window(data, target="window_score")
            if shap_values is not None:
                shap_metrics = {
                    f"shap_{col}": float(shap_values[col].iloc[-1])
                    for col in shap_values.columns
                }
            else:
                logger.warning("SHAP non calculé, métriques vides")
                self.alert_manager.send_alert(
                    "SHAP non calculé, métriques vides", priority=3
                )
                send_telegram_alert("SHAP non calculé, métriques vides")

            self.window_score = score
            logger.info(
                f"Score de fenêtre calculé: {score:.2f}, confidence_drop_rate: {confidence_drop_rate:.2f}"
            )
            self.alert_manager.send_alert(
                f"Score de fenêtre: {score:.2f}, confidence_drop_rate: {confidence_drop_rate:.2f}",
                priority=1,
            )
            send_telegram_alert(
                f"Score de fenêtre: {score:.2f}, confidence_drop_rate: {confidence_drop_rate:.2f}"
            )
            self.log_performance(
                "calculate_window_score",
                time.time() - start_time,
                success=True,
                window_score=score,
                confidence_drop_rate=confidence_drop_rate,
            )
            self.save_snapshot(
                "calculate_window_score",
                {
                    "window_score": score,
                    "confidence_drop_rate": confidence_drop_rate,
                    **shap_metrics,
                },
                compress=False,
            )
            return score
        except Exception as e:
            self.alert_manager.send_alert(
                f"Erreur calculate_window_score: {str(e)}", priority=4
            )
            send_telegram_alert(f"Erreur calculate_window_score: {str(e)}")
            logger.error(f"Erreur calculate_window_score: {str(e)}")
            self.log_performance(
                "calculate_window_score",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            return 0.0

    def block_trade(self, macro_score: float, data: pd.DataFrame, step: int) -> bool:
        """
        Vérifie si un trade doit être bloqué en fonction des conditions macro et de volatilité.

        Args:
            macro_score (float): Score macro (par exemple, confiance de la prédiction).
            data (pd.DataFrame): Données contenant les features.
            step (int): Étape du trading.

        Returns:
            bool: True si le trade est bloqué, False sinon.
        """
        start_time = time.time()
        try:
            self.validate_data(data)
            current_data = data.iloc[-1]
            timestamp = current_data["timestamp"]

            # Utiliser vix_es_correlation si disponible, sinon vix
            vix = current_data.get("vix_es_correlation", current_data.get("vix", 0))

            # Charger les événements macro récents
            recent_events = self.macro_events[
                (self.macro_events["timestamp"] >= timestamp - timedelta(hours=24))
                & (self.macro_events["timestamp"] <= timestamp + timedelta(hours=1))
            ]
            event_count_24h = len(recent_events)
            min_proximity = current_data.get(
                "event_timing_proximity", 7200
            )  # 2h par défaut

            # Ajuster le seuil selon impact_score
            event_threshold = (
                self.thresholds["event_timing_threshold"]
                * (2 if recent_events["impact_score"].max() > 0.8 else 1)
                if not recent_events.empty
                else self.thresholds["event_timing_threshold"]
            )

            # Vérifier les conditions de blocage
            is_blocked = False
            reason = "Fenêtre de trading valide"
            window_score = self.calculate_window_score(data)
            confidence_drop_rate = 1.0 - min(
                window_score / self.thresholds["window_score_threshold"], 1.0
            )

            if macro_score > self.thresholds["macro_score_threshold"]:
                reason = f"Score macro trop élevé: {macro_score:.2f} > {self.thresholds['macro_score_threshold']}"
                is_blocked = True
            elif vix > self.thresholds["vix_es_correlation_threshold"]:
                reason = f"Volatilité élevée: vix_es_correlation={vix:.2f} > {self.thresholds['vix_es_correlation_threshold']}"
                is_blocked = True
            elif min_proximity < event_threshold:
                reason = f"Événement macro imminent: {min_proximity} secondes (seuil ajusté: {event_threshold})"
                is_blocked = True
            elif event_count_24h > self.thresholds["event_frequency_threshold"]:
                reason = (
                    f"Fréquence d’événements élevée: {event_count_24h} événements/24h"
                )
                is_blocked = True
            elif window_score < self.thresholds["window_score_threshold"]:
                reason = f"Score de fenêtre trop faible: {window_score:.2f} < {self.thresholds['window_score_threshold']}"
                is_blocked = True

            # Journaliser la décision
            log_entry = {
                "timestamp": str(timestamp),
                "step": step,
                "decision": "block" if is_blocked else "allow",
                "reason": reason,
                "macro_score": macro_score,
                "vix_es_correlation": vix,
                "window_score": window_score,
                "confidence_drop_rate": confidence_drop_rate,
            }
            self.log_buffer.append(log_entry)
            if len(self.log_buffer) >= self.thresholds.get("buffer_size", 100):
                log_df = pd.DataFrame(self.log_buffer)
                log_path = "data/risk/trade_window_log.csv"
                os.makedirs(os.path.dirname(log_path), exist_ok=True)

                def write_log():
                    if not os.path.exists(log_path):
                        log_df.to_csv(log_path, index=False, encoding="utf-8")
                    else:
                        log_df.to_csv(
                            log_path,
                            mode="a",
                            header=False,
                            index=False,
                            encoding="utf-8",
                        )

                self.with_retries(write_log)
                self.log_buffer = []

            if is_blocked:
                self.block_events.append(
                    {
                        "timestamp": str(timestamp),
                        "reason": reason,
                        "macro_score": macro_score,
                        "vix_es_correlation": vix,
                        "confidence_drop_rate": confidence_drop_rate,
                    }
                )
                logger.warning(
                    f"Trade bloqué: {reason}, confidence_drop_rate: {confidence_drop_rate:.2f}"
                )
                self.alert_manager.send_alert(
                    f"Trade bloqué: {reason}, confidence_drop_rate: {confidence_drop_rate:.2f}",
                    priority=2,
                )
                send_telegram_alert(
                    f"Trade bloqué: {reason}, confidence_drop_rate: {confidence_drop_rate:.2f}"
                )

                def store_block():
                    store_pattern(
                        data,
                        action=0.0,
                        reward=0.0,
                        neural_regime="defensive",
                        confidence=macro_score,
                        metadata={
                            "event": reason.split(":")[0].lower().replace(" ", "_"),
                            "value": macro_score,
                            "vix_es_correlation": vix,
                            "confidence_drop_rate": confidence_drop_rate,
                        },
                    )

                self.with_retries(store_block)

                self.save_block_snapshot(
                    step, timestamp, reason, macro_score, data, confidence_drop_rate
                )
            else:
                logger.info(
                    f"Trade autorisé: {reason}, confidence_drop_rate: {confidence_drop_rate:.2f}"
                )
                self.alert_manager.send_alert(
                    f"Trade autorisé: {reason}, confidence_drop_rate: {confidence_drop_rate:.2f}",
                    priority=1,
                )
                send_telegram_alert(
                    f"Trade autorisé: {reason}, confidence_drop_rate: {confidence_drop_rate:.2f}"
                )

            self.log_performance(
                "block_trade",
                time.time() - start_time,
                success=True,
                is_blocked=is_blocked,
                macro_score=macro_score,
                vix_es_correlation=vix,
                confidence_drop_rate=confidence_drop_rate,
            )
            return is_blocked
        except Exception as e:
            self.alert_manager.send_alert(f"Erreur block_trade: {str(e)}", priority=4)
            send_telegram_alert(f"Erreur block_trade: {str(e)}")
            logger.error(f"Erreur block_trade: {str(e)}")
            self.log_performance(
                "block_trade", time.time() - start_time, success=False, error=str(e)
            )
            return True  # Bloquer par défaut en cas d’erreur

    def prevent_overtrading(self, last_trade_time: Optional[datetime] = None) -> bool:
        """
        Vérifie si un trade doit être bloqué pour éviter le surtrading.

        Args:
            last_trade_time (Optional[datetime]): Horodatage du dernier trade.

        Returns:
            bool: True si le trade est bloqué, False sinon.
        """
        start_time = time.time()
        try:
            if last_trade_time:
                self.trade_timestamps.append(last_trade_time)

            # Nettoyer les anciens timestamps
            self.trade_timestamps = [
                t
                for t in self.trade_timestamps
                if t > datetime.now() - timedelta(hours=1)
            ]

            # Ajuster la limite selon VIX
            trade_limit = self.thresholds["max_trades_per_hour"]
            if self.last_vix > self.thresholds["vix_es_correlation_threshold"]:
                trade_limit *= 0.5

            # Vérifier la fréquence des trades
            trade_count_1h = len(self.trade_timestamps)
            if trade_count_1h >= trade_limit:
                reason = f"Surtrading détecté: {trade_count_1h} trades/heure (limite ajustée: {trade_limit})"
                confidence_drop_rate = 1.0 - min(
                    trade_count_1h / (trade_limit * 2), 1.0
                )
                self.block_events.append(
                    {
                        "timestamp": str(datetime.now()),
                        "reason": reason,
                        "macro_score": 0.0,
                        "vix_es_correlation": self.last_vix,
                        "confidence_drop_rate": confidence_drop_rate,
                    }
                )
                logger.warning(
                    f"Trade bloqué: {reason}, confidence_drop_rate: {confidence_drop_rate:.2f}"
                )
                self.alert_manager.send_alert(
                    f"Trade bloqué: {reason}, confidence_drop_rate: {confidence_drop_rate:.2f}",
                    priority=2,
                )
                send_telegram_alert(
                    f"Trade bloqué: {reason}, confidence_drop_rate: {confidence_drop_rate:.2f}"
                )

                def store_overtrade():
                    store_pattern(
                        pd.DataFrame({"timestamp": [datetime.now()]}),
                        action=0.0,
                        reward=0.0,
                        neural_regime="defensive",
                        confidence=0.0,
                        metadata={
                            "event": "overtrade_block",
                            "trade_count": trade_count_1h,
                            "confidence_drop_rate": confidence_drop_rate,
                        },
                    )

                self.with_retries(store_overtrade)

                self.save_block_snapshot(
                    len(self.block_events),
                    pd.Timestamp.now(),
                    reason,
                    0.0,
                    confidence_drop_rate=confidence_drop_rate,
                )
                self.log_performance(
                    "prevent_overtrading",
                    time.time() - start_time,
                    success=True,
                    trade_count=trade_count_1h,
                    confidence_drop_rate=confidence_drop_rate,
                )
                return True

            self.log_performance(
                "prevent_overtrading",
                time.time() - start_time,
                success=True,
                trade_count=trade_count_1h,
            )
            return False
        except Exception as e:
            self.alert_manager.send_alert(
                f"Erreur prevent_overtrading: {str(e)}", priority=4
            )
            send_telegram_alert(f"Erreur prevent_overtrading: {str(e)}")
            logger.error(f"Erreur prevent_overtrading: {str(e)}")
            self.log_performance(
                "prevent_overtrading",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            return True  # Bloquer par défaut en cas d’erreur

    def check_trade_window(
        self,
        macro_score: float,
        data: pd.DataFrame,
        step: int,
        last_trade_time: Optional[datetime] = None,
    ) -> bool:
        """
        Vérifie si la fenêtre de trading est appropriée en combinant block_trade et prevent_overtrading.

        Args:
            macro_score (float): Score macro.
            data (pd.DataFrame): Données contenant les features.
            step (int): Étape du trading.
            last_trade_time (Optional[datetime]): Horodatage du dernier trade.

        Returns:
            bool: True si le trade est bloqué, False sinon.
        """
        start_time = time.time()
        try:
            if self.prevent_overtrading(last_trade_time):
                self.log_performance(
                    "check_trade_window",
                    time.time() - start_time,
                    success=True,
                    is_blocked=True,
                )
                return True
            is_blocked = self.block_trade(macro_score, data, step)
            self.log_performance(
                "check_trade_window",
                time.time() - start_time,
                success=True,
                is_blocked=is_blocked,
                macro_score=macro_score,
            )
            return is_blocked
        except Exception as e:
            self.alert_manager.send_alert(
                f"Erreur check_trade_window: {str(e)}", priority=4
            )
            send_telegram_alert(f"Erreur check_trade_window: {str(e)}")
            logger.error(f"Erreur check_trade_window: {str(e)}")
            self.log_performance(
                "check_trade_window",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            return True

    def save_block_snapshot(
        self,
        step: int,
        timestamp: pd.Timestamp,
        reason: str,
        macro_score: float,
        data: Optional[pd.DataFrame] = None,
        confidence_drop_rate: float = 0.0,
    ):
        """
        Sauvegarde un instantané des décisions de blocage, avec option de compression gzip.

        Args:
            step (int): Étape du trading.
            timestamp (pd.Timestamp): Horodatage.
            reason (str): Raison du blocage.
            macro_score (float): Score macro.
            data (Optional[pd.DataFrame]): Données contextuelles.
            confidence_drop_rate (float): Métrique d’auto-conscience.
        """
        start_time = time.time()
        try:
            snapshot = {
                "step": step,
                "timestamp": str(timestamp),
                "reason": reason,
                "macro_score": macro_score,
                "window_score": self.window_score,
                "confidence_drop_rate": confidence_drop_rate,
                "block_events": self.block_events[-10:],
                "features": (
                    data.to_dict(orient="records")[0]
                    if data is not None and not data.empty
                    else None
                ),
            }
            snapshot_path = SNAPSHOT_DIR / f"window_step_{step:04d}.json"

            def write_snapshot():
                with open(snapshot_path, "w", encoding="utf-8") as f:
                    json.dump(snapshot, f, indent=4)

            self.with_retries(write_snapshot)
            logger.info(f"Snapshot blocage sauvegardé: {snapshot_path}")
            self.alert_manager.send_alert(
                f"Snapshot blocage step {step} sauvegardé", priority=1
            )
            send_telegram_alert(f"Snapshot blocage step {step} sauvegardé")
            self.log_performance(
                "save_block_snapshot",
                time.time() - start_time,
                success=True,
                step=step,
                confidence_drop_rate=confidence_drop_rate,
            )
        except Exception as e:
            self.alert_manager.send_alert(
                f"Erreur sauvegarde snapshot blocage: {str(e)}", priority=3
            )
            send_telegram_alert(f"Erreur sauvegarde snapshot blocage: {str(e)}")
            logger.error(f"Erreur sauvegarde snapshot blocage: {str(e)}")
            self.log_performance(
                "save_block_snapshot",
                time.time() - start_time,
                success=False,
                error=str(e),
            )

    def save_dashboard_status(
        self, status_file: str = "data/risk/trade_window_dashboard.json"
    ):
        """
        Sauvegarde l’état du filtre pour mia_dashboard.py.

        Args:
            status_file (str): Chemin du fichier JSON.
        """
        start_time = time.time()
        try:
            confidence_drop_rate = 1.0 - min(
                self.window_score / self.thresholds["window_score_threshold"], 1.0
            )
            status = {
                "timestamp": datetime.now().isoformat(),
                "num_blocks": len(self.block_events),
                "window_score": self.window_score,
                "confidence_drop_rate": confidence_drop_rate,
                "last_block_reason": (
                    self.block_events[-1]["reason"]
                    if self.block_events
                    else "Aucun blocage"
                ),
                "block_events": self.block_events[-10:],
            }
            os.makedirs(os.path.dirname(status_file), exist_ok=True)

            def write_status():
                with open(status_file, "w", encoding="utf-8") as f:
                    json.dump(status, f, indent=4)

            self.with_retries(write_status)
            logger.info(f"Dashboard filtre sauvegardé: {status_file}")
            self.alert_manager.send_alert("Dashboard filtre mis à jour", priority=1)
            send_telegram_alert("Dashboard filtre mis à jour")
            self.log_performance(
                "save_dashboard_status",
                time.time() - start_time,
                success=True,
                confidence_drop_rate=confidence_drop_rate,
            )
        except Exception as e:
            self.alert_manager.send_alert(
                f"Erreur sauvegarde dashboard: {str(e)}", priority=3
            )
            send_telegram_alert(f"Erreur sauvegarde dashboard: {str(e)}")
            logger.error(f"Erreur sauvegarde dashboard: {str(e)}")
            self.log_performance(
                "save_dashboard_status",
                time.time() - start_time,
                success=False,
                error=str(e),
            )


if __name__ == "__main__":
    try:
        # Données simulées pour test
        data = pd.DataFrame(
            {
                "timestamp": [datetime.now()],
                "vix": [20.0],
                "vix_es_correlation": [22.0],
                "event_volatility_impact": [0.3],
                "event_timing_proximity": [1800.0],
                "event_frequency_24h": [2.0],
                "trade_frequency_1s": [8.0],
                **{
                    f"feat_{i}": [np.random.uniform(0, 1)] for i in range(344)
                },  # 350 features
            }
        )
        macro_events = pd.DataFrame(
            {
                "timestamp": [datetime.now() + timedelta(minutes=30)],
                "event_type": ["FOMC"],
                "impact_score": [0.8],
            }
        )
        os.makedirs("data/news", exist_ok=True)
        macro_events.to_csv("data/news/macro_events.csv", index=False, encoding="utf-8")

        # Test TradeWindowFilter
        filter = TradeWindowFilter()
        window_score = filter.calculate_window_score(data)
        print("Window Score:", window_score)

        macro_score = 0.8
        step = 1
        is_blocked = filter.block_trade(macro_score, data, step)
        print("Trade Blocked:", is_blocked)

        last_trade_time = datetime.now()
        is_overtrading = filter.prevent_overtrading(last_trade_time)
        print("Overtrading Prevented:", is_overtrading)

        is_window_blocked = filter.check_trade_window(
            macro_score, data, step, last_trade_time
        )
        print("Window Blocked:", is_window_blocked)

        filter.save_dashboard_status()
    except Exception as e:
        alert_manager = AlertManager()
        alert_manager.send_alert(f"Erreur test TradeWindowFilter: {str(e)}", priority=5)
        send_telegram_alert(f"Erreur test TradeWindowFilter: {str(e)}")
        logger.error(f"Erreur test TradeWindowFilter: {str(e)}")
