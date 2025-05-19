# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/risk/decision_log.py
# Rôle : Enregistre les décisions de trading pour audit et transparence dans MIA_IA_SYSTEM_v2_2025, incluant la probabilité
#        de réussite du trade pour analyse ultérieure.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.0,<6.0.0, pyyaml>=6.0.0,<7.0.0, logging, os, json,
#   datetime, signal, gzip
# - src/model/adaptive_learner.py
# - src/utils/telegram_alert.py
# - src/model/utils/miya_console.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/trade_probability.py
# - src/trading/shap_weighting.py
#
# Inputs :
# - Données de trading (pd.DataFrame avec 350 features pour entraînement ou 150 SHAP features pour inférence)
# - Configuration via config/market_config.yaml
#
# Outputs :
# - Logs dans data/logs/trading/decision_log.csv (colonnes : timestamp, trade_id, decision, signal_score, regime_probs, trade_success_prob, ...)
# - Logs dans data/logs/trading/decision_log.log
# - Logs de performance dans data/logs/decision_log_performance.csv
# - Snapshots JSON dans data/risk/decision_snapshots/*.json (option *.json.gz)
# - Dashboard JSON dans data/risk/decision_log_dashboard.json
#
# Notes :
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence) définies dans config/feature_sets.yaml.
# - Utilise IQFeed exclusivement via data_provider.py pour les données d’entrée.
# - Supprime toute référence à 320 features, 81 features, et obs_t.
# - Implémente retries (max 3, délai 2^attempt secondes) pour les opérations critiques.
# - Intègre la probabilité de réussite du trade via TradeProbabilityPredictor.
# - Intègre confidence_drop_rate (Phase 8) pour l’auto-conscience, calculé à partir de trade_success_prob.
# - Intègre l’analyse SHAP (Phase 17) dans log_decision pour les données contextuelles, limitée à 50 features.
# - Tests unitaires disponibles dans tests/test_decision_log.py.

import gzip
import json
import logging
import os
import signal
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import psutil

from src.model.adaptive_learner import store_pattern
from src.model.trade_probability import TradeProbabilityPredictor
from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import config_manager
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.trading.shap_weighting import calculate_shap
from src.utils.telegram_alert import send_telegram_alert

# Configuration du logging
os.makedirs("data/logs/trading", exist_ok=True)
logging.basicConfig(
    filename="data/logs/trading/decision_log.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Constantes
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Secondes, exponentiel
PERF_LOG_PATH = Path("data/logs/decision_log_performance.csv")
SNAPSHOT_DIR = Path("data/risk/decision_snapshots")
SHAP_FEATURE_LIMIT = 50


class DecisionLog:
    """
    Classe pour enregistrer et gérer les décisions de trading pour audit et transparence.
    """

    def __init__(self, config_path: str = "config/market_config.yaml"):
        """
        Initialise le journal des décisions.

        Args:
            config_path (str): Chemin vers la configuration du marché.
        """
        self.alert_manager = AlertManager()
        self.snapshot_dir = SNAPSHOT_DIR
        self.perf_log = PERF_LOG_PATH
        SNAPSHOT_DIR.mkdir(exist_ok=True)
        PERF_LOG_PATH.parent.mkdir(exist_ok=True)
        signal.signal(signal.SIGINT, self.handle_sigint)

        try:
            # Charger la configuration
            config = config_manager.get_config(config_path).get("decision_log", {})
            self.config = {
                "log_level": config.get("log_level", "detailed"),
                "alert_threshold": config.get("alert_threshold", 0.9),
                "context_features": config.get(
                    "context_features", ["event_volatility_impact", "vix"]
                ),
                "max_log_size": config.get("max_log_size", 10 * 1024 * 1024),  # 10 MB
                "snapshot_count": config.get("snapshot_count", 10),
                "buffer_size": config.get("buffer_size", 100),
                "observation_dims": config.get(
                    "observation_dims", {"training": 350, "inference": 150}
                ),
                "min_trade_success_prob": config.get("thresholds", {}).get(
                    "min_trade_success_prob", 0.7
                ),
            }

            # Initialiser le prédicteur de probabilité de trade
            self.trade_predictor = TradeProbabilityPredictor()

            # Chemins de sortie
            self.decision_log_path = "data/logs/trading/decision_log.csv"
            self.dashboard_path = "data/risk/decision_log_dashboard.json"

            # État interne
            self.decisions = []  # Historique des décisions en mémoire
            self.success_count = 0  # Nombre de trades réussis
            self.total_trades = 0  # Nombre total de trades
            self.decision_buffer = []  # Buffer pour écritures CSV

            logger.info("DecisionLog initialisé avec succès")
            miya_speak(
                "DecisionLog initialisé", tag="DECISION_LOG", voice_profile="calm"
            )
            self.log_performance("init", 0, success=True)
        except Exception as e:
            miya_alerts(
                f"Erreur initialisation DecisionLog : {str(e)}",
                tag="DECISION_LOG",
                priority=5,
                voice_profile="urgent",
            )
            self.alert_manager.send_alert(
                f"Erreur initialisation DecisionLog: {str(e)}", priority=5
            )
            logger.error(f"Erreur initialisation DecisionLog : {str(e)}")
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
                    f"retry_attempt_{attempt+1}", latency, success=True
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
        Enregistre les performances (CPU, mémoire, latence) dans decision_log_performance.csv.

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
            os.makedirs(os.path.dirname(self.perf_log), exist_ok=True)
            log_df = pd.DataFrame([log_entry])

            def write_log():
                if not os.path.exists(self.perf_log):
                    log_df.to_csv(self.perf_log, index=False, encoding="utf-8")
                else:
                    log_df.to_csv(
                        self.perf_log,
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

    def save_snapshot(
        self, snapshot_type: str, data: Dict, compress: bool = False
    ) -> None:
        """
        Sauvegarde un instantané JSON des résultats, avec option de compression gzip.

        Args:
            snapshot_type (str): Type de snapshot (ex. : decision_log).
            data (Dict): Données à sauvegarder.
            compress (bool): Si True, compresse en gzip.
        """
        try:
            start_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = {"timestamp": timestamp, "type": snapshot_type, "data": data}
            snapshot_path = (
                self.snapshot_dir / f"snapshot_{snapshot_type}_{timestamp}.json"
            )
            os.makedirs(self.snapshot_dir, exist_ok=True)

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

    def archive_log(self) -> None:
        """
        Archive decision_log.csv si sa taille dépasse max_log_size.
        """
        start_time = time.time()
        try:
            if os.path.exists(self.decision_log_path):
                file_size = os.path.getsize(self.decision_log_path)
                if file_size > self.config["max_log_size"]:
                    archive_path = f"data/logs/trading/decision_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

                    def archive():
                        os.rename(self.decision_log_path, archive_path)

                    self.with_retries(archive)
                    logger.info(f"Journal archivé: {archive_path}")
                    miya_speak(
                        f"Journal des décisions archivé: {archive_path}",
                        tag="DECISION_LOG",
                        voice_profile="calm",
                    )
                    self.alert_manager.send_alert(
                        f"Journal archivé: {archive_path}", priority=1
                    )
                    send_telegram_alert(f"Journal archivé: {archive_path}")
            self.log_performance("archive_log", time.time() - start_time, success=True)
        except Exception as e:
            miya_alerts(
                f"Erreur archivage journal : {str(e)}",
                tag="DECISION_LOG",
                priority=3,
                voice_profile="urgent",
            )
            self.alert_manager.send_alert(
                f"Erreur archivage journal: {str(e)}", priority=3
            )
            send_telegram_alert(f"Erreur archivage journal: {str(e)}")
            logger.error(f"Erreur archivage journal : {str(e)}")
            self.log_performance(
                "archive_log", time.time() - start_time, success=False, error=str(e)
            )

    def validate_data(self, data: Optional[pd.DataFrame] = None) -> None:
        """
        Valide les données d’entrée (350 features pour entraînement, 150 SHAP pour inférence).

        Args:
            data (Optional[pd.DataFrame]): Données à valider (facultatif).

        Raises:
            ValueError: Si les données sont invalides.
        """
        start_time = time.time()
        try:
            if data is None:
                return
            expected_dims = self.config.get(
                "observation_dims", {"training": 350, "inference": 150}
            )
            data_dim = data.shape[1]
            if data_dim not in (expected_dims["training"], expected_dims["inference"]):
                raise ValueError(
                    f"Nombre de features incorrect: {data_dim}, attendu {expected_dims['training']} ou {expected_dims['inference']}"
                )
            for col in self.config["context_features"]:
                if col in data.columns:
                    if data[col].isnull().any() or data[col].le(0).any():
                        raise ValueError(
                            f"Colonne {col} contient des NaN ou des valeurs non positives"
                        )
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        raise ValueError(
                            f"Colonne {col} n'est pas numérique: {data[col].dtype}"
                        )
            # Validation du timestamp
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
            miya_alerts(
                f"Erreur validation données : {str(e)}",
                tag="DECISION_LOG",
                priority=4,
                voice_profile="urgent",
            )
            self.alert_manager.send_alert(
                f"Erreur validation données: {str(e)}", priority=4
            )
            send_telegram_alert(f"Erreur validation données: {str(e)}")
            logger.error(f"Erreur validation données : {str(e)}")
            self.log_performance(
                "validate_data", time.time() - start_time, success=False, error=str(e)
            )
            raise

    def log_decision(
        self,
        trade_id: int,
        decision: str,
        score: float,
        reason: str,
        data: Optional[pd.DataFrame] = None,
        outcome: Optional[float] = None,
        regime_probs: Optional[List[float]] = None,
    ) -> None:
        """
        Enregistre une décision de trading, incluant confidence_drop_rate et analyse SHAP.

        Args:
            trade_id (int): Identifiant unique du trade.
            decision (str): Décision prise ("execute", "block", "hold", "reduce").
            score (float): Score de confiance ou probabilité de la décision.
            reason (str): Raison de la décision.
            data (Optional[pd.DataFrame]): Données contextuelles (350 features pour entraînement, 150 pour inférence).
            outcome (Optional[float]): Résultat du trade (profit/perte, facultatif).
            regime_probs (Optional[List[float]]): Probabilités des régimes de marché (trend, range, defensive).

        Raises:
            ValueError: Si les paramètres sont invalides.
        """
        start_time = time.time()
        try:
            # Validation des paramètres
            if not isinstance(trade_id, int) or trade_id < 0:
                raise ValueError(f"trade_id invalide: {trade_id}")
            if decision not in ["execute", "block", "hold", "reduce"]:
                raise ValueError(f"Décision invalide: {decision}")
            if not isinstance(score, (int, float)) or score < 0 or score > 1:
                raise ValueError(f"Score invalide: {score}")
            if not isinstance(reason, str) or not reason:
                raise ValueError("Raison vide ou invalide")
            if outcome is not None and (
                not isinstance(outcome, (int, float)) or pd.isna(outcome)
            ):
                raise ValueError(f"Outcome invalide: {outcome}")
            if regime_probs is not None and (
                not isinstance(regime_probs, list)
                or len(regime_probs) != 3
                or not all(
                    isinstance(p, (int, float)) and 0 <= p <= 1 for p in regime_probs
                )
            ):
                raise ValueError(f"regime_probs invalide: {regime_probs}")
            self.validate_data(data)

            # Calculer la probabilité de réussite du trade
            trade_success_prob = None
            if data is not None:

                def predict_trade_prob():
                    return self.trade_predictor.predict(data)

                trade_success_prob = self.with_retries(predict_trade_prob)
                if trade_success_prob is None:
                    self.alert_manager.send_alert(
                        "Échec calcul probabilité de réussite du trade", priority=3
                    )
                    send_telegram_alert("Échec calcul probabilité de réussite du trade")
                    logger.warning("Échec calcul probabilité de réussite du trade")
                    trade_success_prob = 0.0

            # Calculer confidence_drop_rate (Phase 8)
            confidence_drop_rate = (
                1.0
                - min(trade_success_prob / self.config["min_trade_success_prob"], 1.0)
                if trade_success_prob is not None
                else 0.0
            )

            # Analyse SHAP (Phase 17)
            shap_metrics = {}
            if data is not None and not data.empty:

                def run_shap():
                    return calculate_shap(
                        data,
                        target=(
                            "trade_success_prob"
                            if trade_success_prob is not None
                            else "signal_score"
                        ),
                        max_features=SHAP_FEATURE_LIMIT,
                    )

                shap_values = self.with_retries(run_shap)
                if shap_values is not None:
                    shap_metrics = {
                        f"shap_{col}": float(shap_values[col].iloc[-1])
                        for col in shap_values.columns
                    }
                else:
                    error_msg = "Échec calcul SHAP, poursuite sans SHAP"
                    self.alert_manager.send_alert(error_msg, priority=3)
                    send_telegram_alert(error_msg)
                    logger.warning(error_msg)

            # Construire l’entrée de log
            log_entry = {
                "trade_id": trade_id,
                "timestamp": datetime.now().isoformat(),
                "decision": decision,
                "signal_score": score,
                "reason": reason,
                "outcome": outcome if outcome is not None else None,
                "regime_probs": (
                    json.dumps(regime_probs) if regime_probs is not None else None
                ),
                "trade_success_prob": trade_success_prob,
                "confidence_drop_rate": confidence_drop_rate,
                **shap_metrics,
            }

            # Ajouter le contexte si données fournies
            if data is not None:
                for col in self.config["context_features"]:
                    if col in data.columns:
                        log_entry[col] = data[col].iloc[-1]

            # Enregistrer dans la mémoire
            self.decisions.append(log_entry)
            self.total_trades += 1
            if outcome is not None and outcome > 0:
                self.success_count += 1

            # Ajouter au buffer pour écriture CSV
            self.decision_buffer.append(log_entry)
            if len(self.decision_buffer) >= self.config["buffer_size"]:
                log_df = pd.DataFrame(self.decision_buffer)
                os.makedirs(os.path.dirname(self.decision_log_path), exist_ok=True)

                def write_csv():
                    if not os.path.exists(self.decision_log_path):
                        log_df.to_csv(
                            self.decision_log_path, index=False, encoding="utf-8"
                        )
                    else:
                        log_df.to_csv(
                            self.decision_log_path,
                            mode="a",
                            header=False,
                            index=False,
                            encoding="utf-8",
                        )

                self.with_retries(write_csv)
                self.decision_buffer = []
                self.archive_log()

            # Stocker dans market_memory.db
            metadata = {
                "decision": decision,
                "reason": reason,
                "signal_score": score,
                "outcome": outcome,
                "regime_probs": regime_probs,
                "trade_success_prob": trade_success_prob,
                "confidence_drop_rate": confidence_drop_rate,
                "shap_metrics": shap_metrics,
            }

            def store_in_memory():
                store_pattern(
                    (
                        data
                        if data is not None
                        else pd.DataFrame({"timestamp": [datetime.now()]})
                    ),
                    action=1.0 if decision == "execute" else 0.0,
                    reward=outcome if outcome is not None else 0.0,
                    neural_regime="trading_decision",
                    confidence=score,
                    metadata=metadata,
                )

            self.with_retries(store_in_memory)

            # Alerte pour les décisions critiques
            if score > self.config["alert_threshold"] and decision == "block":
                alert_message = f"Décision bloquée critique: {reason} (score={score:.2f}, prob={trade_success_prob:.2f}, confidence_drop_rate={confidence_drop_rate:.2f})"
                self.alert_manager.send_alert(alert_message, priority=5)
                send_telegram_alert(alert_message)
                miya_alerts(
                    alert_message,
                    tag="DECISION_LOG",
                    priority=5,
                    voice_profile="urgent",
                )
            if decision == "execute" and (
                score < 0.5
                or (outcome is not None and outcome < -0.1)
                or trade_success_prob < 0.5
                or confidence_drop_rate > 0.5
            ):
                alert_message = f"Décision exécutée risquée: {reason} (score={score:.2f}, outcome={outcome}, prob={trade_success_prob:.2f}, confidence_drop_rate={confidence_drop_rate:.2f})"
                self.alert_manager.send_alert(alert_message, priority=4)
                send_telegram_alert(alert_message)
                miya_alerts(
                    alert_message,
                    tag="DECISION_LOG",
                    priority=4,
                    voice_profile="urgent",
                )

            logger.info(
                f"Décision enregistrée: trade_id={trade_id}, decision={decision}, score={score:.2f}, prob={trade_success_prob:.2f}, confidence_drop_rate={confidence_drop_rate:.2f}, reason={reason}"
            )
            miya_speak(
                f"Décision enregistrée: {decision} (score={score:.2f}, prob={trade_success_prob:.2f}, confidence_drop_rate={confidence_drop_rate:.2f})",
                tag="DECISION_LOG",
                voice_profile="calm",
            )

            # Mettre à jour le dashboard
            self.save_dashboard_status()
            self.log_performance(
                "log_decision",
                time.time() - start_time,
                success=True,
                trade_id=trade_id,
                decision=decision,
                trade_success_prob=trade_success_prob,
                confidence_drop_rate=confidence_drop_rate,
            )

            # Sauvegarder un snapshot
            self.save_snapshot("log_decision", log_entry, compress=False)

        except Exception as e:
            miya_alerts(
                f"Erreur log_decision : {str(e)}",
                tag="DECISION_LOG",
                priority=4,
                voice_profile="urgent",
            )
            self.alert_manager.send_alert(f"Erreur log_decision: {str(e)}", priority=4)
            send_telegram_alert(f"Erreur log_decision: {str(e)}")
            logger.error(f"Erreur log_decision : {str(e)}")
            self.log_performance(
                "log_decision", time.time() - start_time, success=False, error=str(e)
            )
            raise

    def summarize_decisions(self) -> Dict[str, Union[int, float, Dict]]:
        """
        Résume les décisions enregistrées, incluant confidence_drop_rate.

        Returns:
            Dict[str, Union[int, float, Dict]]: Métriques (nombre de trades, taux de succès, raisons fréquentes, statistiques temporelles).
        """
        start_time = time.time()
        try:
            if not self.decisions:
                summary = {
                    "num_decisions": 0,
                    "success_rate": 0.0,
                    "execute_count": 0,
                    "block_count": 0,
                    "reason_frequency": {},
                    "hourly_decisions": {},
                    "daily_decisions": {},
                    "average_score": 0.0,
                    "score_variance": 0.0,
                    "average_outcome": 0.0,
                    "average_trade_success_prob": 0.0,
                    "average_confidence_drop_rate": 0.0,
                }
                self.log_performance(
                    "summarize_decisions", time.time() - start_time, success=True
                )
                return summary

            # Calculer les métriques
            num_decisions = len(self.decisions)
            execute_count = sum(1 for d in self.decisions if d["decision"] == "execute")
            block_count = sum(1 for d in self.decisions if d["decision"] == "block")
            success_rate = (
                self.success_count / self.total_trades if self.total_trades > 0 else 0.0
            )

            # Fréquence des raisons
            reasons = [d["reason"] for d in self.decisions]
            reason_frequency = {
                reason: reasons.count(reason) for reason in set(reasons)
            }

            # Statistiques temporelles
            df = pd.DataFrame(self.decisions)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            hourly_decisions = (
                df.groupby(df["timestamp"].dt.floor("H")).size().to_dict()
            )
            daily_decisions = df.groupby(df["timestamp"].dt.floor("D")).size().to_dict()

            # Métriques avancées
            scores = [d["signal_score"] for d in self.decisions]
            outcomes = [
                d["outcome"] for d in self.decisions if d["outcome"] is not None
            ]
            trade_probs = [
                d["trade_success_prob"]
                for d in self.decisions
                if d["trade_success_prob"] is not None
            ]
            confidence_drop_rates = [
                d["confidence_drop_rate"]
                for d in self.decisions
                if "confidence_drop_rate" in d
            ]
            average_score = float(np.mean(scores)) if scores else 0.0
            score_variance = float(np.var(scores)) if scores else 0.0
            average_outcome = float(np.mean(outcomes)) if outcomes else 0.0
            average_trade_success_prob = (
                float(np.mean(trade_probs)) if trade_probs else 0.0
            )
            average_confidence_drop_rate = (
                float(np.mean(confidence_drop_rates)) if confidence_drop_rates else 0.0
            )

            summary = {
                "num_decisions": num_decisions,
                "success_rate": success_rate,
                "execute_count": execute_count,
                "block_count": block_count,
                "reason_frequency": reason_frequency,
                "hourly_decisions": {str(k): v for k, v in hourly_decisions.items()},
                "daily_decisions": {str(k): v for k, v in daily_decisions.items()},
                "average_score": average_score,
                "score_variance": score_variance,
                "average_outcome": average_outcome,
                "average_trade_success_prob": average_trade_success_prob,
                "average_confidence_drop_rate": average_confidence_drop_rate,
            }

            logger.info(f"Résumé des décisions: {summary}")
            miya_speak(
                f"Résumé: {num_decisions} décisions, taux de succès={success_rate:.2%}, probabilité moyenne={average_trade_success_prob:.2f}, confidence_drop_rate moyen={average_confidence_drop_rate:.2f}",
                tag="DECISION_LOG",
                voice_profile="calm",
            )
            self.log_performance(
                "summarize_decisions", time.time() - start_time, success=True
            )
            self.save_snapshot("summarize_decisions", summary, compress=False)
            return summary
        except Exception as e:
            miya_alerts(
                f"Erreur summarize_decisions : {str(e)}",
                tag="DECISION_LOG",
                priority=4,
                voice_profile="urgent",
            )
            self.alert_manager.send_alert(
                f"Erreur summarize_decisions: {str(e)}", priority=4
            )
            send_telegram_alert(f"Erreur summarize_decisions: {str(e)}")
            logger.error(f"Erreur summarize_decisions : {str(e)}")
            self.log_performance(
                "summarize_decisions",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            return {
                "num_decisions": 0,
                "success_rate": 0.0,
                "execute_count": 0,
                "block_count": 0,
                "reason_frequency": {},
                "hourly_decisions": {},
                "daily_decisions": {},
                "average_score": 0.0,
                "score_variance": 0.0,
                "average_outcome": 0.0,
                "average_trade_success_prob": 0.0,
                "average_confidence_drop_rate": 0.0,
            }

    def export_decision_snapshot(
        self, step: int, data: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Sauvegarde un instantané des décisions pour analyse détaillée.

        Args:
            step (int): Étape du trading.
            data (Optional[pd.DataFrame]): Données contextuelles (facultatif).
        """
        start_time = time.time()
        try:
            snapshot = {
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "decisions": self.decisions[-self.config["snapshot_count"] :],
                "summary": self.summarize_decisions(),
                "features": (
                    data.to_dict(orient="records")[0]
                    if data is not None and not data.empty
                    else None
                ),
            }
            snapshot_path = self.snapshot_dir / f"decision_step_{step:04d}.json"

            def write_snapshot():
                with open(snapshot_path, "w", encoding="utf-8") as f:
                    json.dump(snapshot, f, indent=4)

            self.with_retries(write_snapshot)
            logger.info(f"Snapshot décision sauvegardé: {snapshot_path}")
            miya_speak(
                f"Snapshot décision step {step} sauvegardé",
                tag="DECISION_LOG",
                voice_profile="calm",
            )
            self.alert_manager.send_alert(
                f"Snapshot décision sauvegardé: {snapshot_path}", priority=1
            )
            send_telegram_alert(f"Snapshot décision sauvegardé: {snapshot_path}")
            self.log_performance(
                "export_decision_snapshot",
                time.time() - start_time,
                success=True,
                step=step,
            )
        except Exception as e:
            miya_alerts(
                f"Erreur export_decision_snapshot : {str(e)}",
                tag="DECISION_LOG",
                priority=3,
                voice_profile="urgent",
            )
            self.alert_manager.send_alert(
                f"Erreur export_decision_snapshot: {str(e)}", priority=3
            )
            send_telegram_alert(f"Erreur export_decision_snapshot: {str(e)}")
            logger.error(f"Erreur export_decision_snapshot : {str(e)}")
            self.log_performance(
                "export_decision_snapshot",
                time.time() - start_time,
                success=False,
                error=str(e),
            )

    def save_dashboard_status(self, status_file: Optional[str] = None) -> None:
        """
        Sauvegarde l’état des décisions pour mia_dashboard.py.

        Args:
            status_file (Optional[str]): Chemin du fichier JSON (par défaut: self.dashboard_path).
        """
        start_time = time.time()
        try:
            status_file = status_file or self.dashboard_path
            summary = self.summarize_decisions()
            status = {
                "timestamp": datetime.now().isoformat(),
                "num_decisions": summary["num_decisions"],
                "success_rate": summary["success_rate"],
                "execute_count": summary["execute_count"],
                "block_count": summary["block_count"],
                "last_decision": self.decisions[-1] if self.decisions else None,
                "reason_frequency": summary["reason_frequency"],
                "hourly_decisions": summary["hourly_decisions"],
                "daily_decisions": summary["daily_decisions"],
                "average_score": summary["average_score"],
                "score_variance": summary["score_variance"],
                "average_outcome": summary["average_outcome"],
                "average_trade_success_prob": summary["average_trade_success_prob"],
                "average_confidence_drop_rate": summary["average_confidence_drop_rate"],
            }
            os.makedirs(os.path.dirname(status_file), exist_ok=True)

            def write_status():
                with open(status_file, "w", encoding="utf-8") as f:
                    json.dump(status, f, indent=4)

            self.with_retries(write_status)
            logger.info(f"Dashboard décisions sauvegardé: {status_file}")
            miya_speak(
                "Dashboard décisions mis à jour",
                tag="DECISION_LOG",
                voice_profile="calm",
            )
            self.alert_manager.send_alert(
                f"Dashboard décisions sauvegardé: {status_file}", priority=1
            )
            send_telegram_alert(f"Dashboard décisions sauvegardé: {status_file}")
            self.log_performance(
                "save_dashboard_status", time.time() - start_time, success=True
            )
        except Exception as e:
            miya_alerts(
                f"Erreur sauvegarde dashboard : {str(e)}",
                tag="DECISION_LOG",
                priority=3,
                voice_profile="urgent",
            )
            self.alert_manager.send_alert(
                f"Erreur sauvegarde dashboard: {str(e)}", priority=3
            )
            send_telegram_alert(f"Erreur sauvegarde dashboard: {str(e)}")
            logger.error(f"Erreur sauvegarde dashboard : {str(e)}")
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
                "event_volatility_impact": [0.3],
                "vix": [20.0],
                **{
                    f"feat_{i}": [np.random.uniform(0, 1)] for i in range(348)
                },  # 350 features
            }
        )

        # Test DecisionLog
        decision_log = DecisionLog()
        decision_log.log_decision(
            trade_id=1,
            decision="execute",
            score=0.85,
            reason="Trade exécuté avec confiance élevée",
            data=data,
            outcome=0.02,
            regime_probs=[0.7, 0.2, 0.1],
        )
        decision_log.log_decision(
            trade_id=2,
            decision="block",
            score=0.95,
            reason="Événement macro imminent",
            data=data,
            regime_probs=[0.1, 0.3, 0.6],
        )
        decision_log.log_decision(
            trade_id=3,
            decision="hold",
            score=0.75,
            reason="Maintien de la position actuelle",
            data=data,
        )
        summary = decision_log.summarize_decisions()
        print("Résumé:", summary)
        decision_log.export_decision_snapshot(step=1, data=data)
        decision_log.save_dashboard_status()
    except Exception as e:
        miya_alerts(
            f"Erreur test DecisionLog : {str(e)}",
            tag="DECISION_LOG",
            priority=5,
            voice_profile="urgent",
        )
        alert_manager = AlertManager()
        alert_manager.send_alert(f"Erreur test DecisionLog: {str(e)}", priority=5)
        send_telegram_alert(f"Erreur test DecisionLog: {str(e)}")
        logger.error(f"Erreur test DecisionLog : {str(e)}")
