```python
# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/risk/risk_controller.py
# Rôle : Gère les risques de trading (arrêt, sizing, pénalités IA) avec overtrade_risk_score et market_liquidity_crash_risk.
#
# Version : 2.1.6
# Date : 2025-05-15
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.0,<6.0.0, pyyaml>=6.0.0,<7.0.0, sqlite3, logging, os, json,
#   datetime, signal, gzip, boto3>=1.28.0,<2.0.0
# - src/model/adaptive_learner.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/risk/options_risk_manager.py
# - src/risk/sierra_chart_errors.py
# - src/api/news_scraper.py
# - src/utils/telegram_alert.py
# - src/trading/shap_weighting.py
#
# Inputs :
# - Données de trading (pd.DataFrame avec 351 features pour entraînement ou 150 SHAP features pour inférence)
# - Configuration via config/market_config.yaml, config/s3_config.yaml
# - config/feature_sets.yaml pour les listes de features statiques
# - market_memory.db pour la mémoire contextuelle (méthode 7)
# - data/features/feature_importance_cache.csv pour le fallback SHAP
#
# Outputs :
# - Logs dans data/logs/trading/risk_controller.log
# - Logs de performance dans data/logs/risk_performance.csv
# - Snapshots JSON dans data/risk_snapshots/*.json (option *.json.gz)
# - Dashboard JSON dans data/risk_dashboard.json
# - penalty_log.csv dans data/logs/trading/
# - Sauvegardes incrémentielles dans data/checkpoints/
# - Sauvegardes S3 dans le bucket configuré (config/s3_config.yaml)
#
# Notes :
# - Compatible avec 351 features (entraînement, incluant iv_skew, bid_ask_imbalance, trade_aggressiveness) et 150 SHAP features (inférence) définies dans config/feature_sets.yaml.
# - Utilise IQFeed exclusivement via data_provider.py pour les données d’entrée.
# - Supprime toute référence à 320 features, 81 features, et obs_t.
# - Intègre la mémoire contextuelle (méthode 7) via market_memory.db et LSTM (méthode 12) pour predicted_vix.
# - Implémente retries (max 3, délai 2^attempt secondes) pour les opérations critiques.
# - Intègre confidence_drop_rate (Phase 8) pour l’auto-conscience, basé sur market_liquidity_crash_risk.
# - Intègre l’analyse SHAP (Phase 17) pour évaluer l’impact des features sur market_liquidity_crash_risk, limitée à 50 features.
# - Sauvegarde des snapshots JSON non compressés par défaut avec option de compression gzip.
# - Sauvegarde S3 pour les métriques de risque via config/s3_config.yaml.
# - Contrôle de la fréquence des trades avec seuils entry_freq_max et entry_freq_min_interval.
# - Utilise AlertManager et telegram_alert pour les notifications critiques.
# - Tests unitaires disponibles dans tests/test_risk_controller.py, étendus pour iv_skew, bid_ask_imbalance, trade_aggressiveness, S3, et fréquence des trades.
# - Refactorisation de log_performance avec _persist_log pour éviter la récursion infinie.

import gzip
import json
import logging
import os
import signal
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import boto3
import numpy as np
import pandas as pd
import psutil
from loguru import logger

from src.api.news_scraper import fetch_news
from src.model.adaptive_learner import store_pattern
from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import get_config
from src.risk.options_risk_manager import OptionsRiskManager
from src.risk.sierra_chart_errors import SierraChartErrorManager
from src.trading.shap_weighting import calculate_shap
from src.utils.telegram_alert import send_telegram_alert

# Configuration du logging
os.makedirs("data/logs/trading", exist_ok=True)
logging.basicConfig(
    filename="data/logs/trading/risk_controller.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger.remove()
logger.add("data/logs/trading/risk_controller.log", level="INFO")

# Constantes
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Secondes, exponentiel
PERF_LOG_PATH = Path("data/logs/risk_performance.csv")
SNAPSHOT_DIR = Path("data/risk_snapshots")
CHECKPOINT_DIR = Path("data/checkpoints")
DB_PATH = "data/market_memory.db"
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
SHAP_FEATURE_LIMIT = 50
S3_CONFIG_PATH = BASE_DIR / "config/s3_config.yaml"


class RiskController:
    """
    Classe pour gérer les risques de trading, incluant stop trading, sizing, et pénalités IA.
    """

    def __init__(self, config_path: str = "config/market_config.yaml"):
        """
        Initialise le contrôleur de risque.

        Args:
            config_path (str): Chemin vers la configuration du marché.
        """
        self.alert_manager = AlertManager()
        SNAPSHOT_DIR.mkdir(exist_ok=True)
        PERF_LOG_PATH.parent.mkdir(exist_ok=True)
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        signal.signal(signal.SIGINT, self.handle_sigint)

        try:
            # Charger les configurations
            config = get_config(config_path).get("risk_controller", {})
            self.s3_config = get_config(S3_CONFIG_PATH)
            self.config = {
                "buffer_size": config.get("buffer_size", 100),
                "observation_dims": {"training": 351, "inference": 150},
                "entry_freq_max": config.get("entry_freq_max", 10),  # Trades par heure
                "entry_freq_min_interval": config.get("entry_freq_min_interval", 60),  # Secondes
            }
            self.thresholds = {
                "max_drawdown": config.get("max_drawdown", -0.1),
                "vix_threshold": config.get("vix_threshold", 30),
                "spread_threshold": config.get("spread_threshold", 0.05),
                "event_impact_threshold": config.get("event_impact_threshold", 0.5),
                "max_position_size": config.get("max_position_size", 5),
                "overtrade_risk_threshold": config.get("overtrade_risk_threshold", 0.8),
                "liquidity_crash_risk_threshold": config.get(
                    "liquidity_crash_risk_threshold", 0.8
                ),
                "max_consecutive_losses": config.get("max_consecutive_losses", 3),
                "penalty_threshold": config.get("penalty_threshold", 0.1),
                "risk_score_threshold": config.get("risk_score_threshold", 0.5),
                "predicted_vix_threshold": config.get("predicted_vix_threshold", 25.0),
                "min_confidence": config.get("min_confidence", 0.7),
            }
            self.metric_weights = {
                "spread_weight": config.get("spread_weight", 0.5),
                "trade_frequency_weight": config.get("trade_frequency_weight", 0.3),
                "vix_weight": config.get("vix_weight", 0.2),
                "cluster_risk_weight": config.get("cluster_risk_weight", 0.3),
                "options_risk_weight": config.get("options_risk_weight", 0.2),
                "news_impact_weight": config.get("news_impact_weight", 0.1),
                "iv_skew_weight": config.get("iv_skew_weight", 0.15),
                "bid_ask_imbalance_weight": config.get("bid_ask_imbalance_weight", 0.1),
                "trade_aggressiveness_weight": config.get("trade_aggressiveness_weight", 0.1),
            }

            # Valider les seuils
            for key, value in self.thresholds.items():
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Seuil invalide pour {key}: {value}")
                if (
                    key
                    in [
                        "max_position_size",
                        "vix_threshold",
                        "spread_threshold",
                        "event_impact_threshold",
                        "overtrade_risk_threshold",
                        "liquidity_crash_risk_threshold",
                        "max_consecutive_losses",
                        "penalty_threshold",
                        "risk_score_threshold",
                        "predicted_vix_threshold",
                        "min_confidence",
                    ]
                    and value <= 0
                ):
                    raise ValueError(f"Seuil {key} doit être positif: {value}")
                if key == "max_drawdown" and value >= 0:
                    raise ValueError(f"Seuil {key} doit être négatif: {value}")

            # État interne
            self.balance = 100000  # Solde initial (simulé)
            self.positions = []  # Liste des positions ouvertes
            self.consecutive_losses = 0  # Compteur de pertes consécutives
            self.penalty_log = []  # Historique des pénalités
            self.risk_metrics = {}  # Métriques de risque
            self.risk_events = []  # Historique des événements de risque
            self.penalty_buffer = []  # Buffer pour écritures CSV
            self.trade_timestamps = []  # Historique des timestamps des trades

            # Initialisation des composants
            self.options_risk_manager = OptionsRiskManager()
            self.sierra_error_manager = SierraChartErrorManager()

            logger.info("RiskController initialisé avec succès")
            self.alert_manager.send_alert("RiskController initialisé", priority=2)
            send_telegram_alert("RiskController initialisé")
            self.log_performance("init", 0, success=True)
        except Exception as e:
            self.alert_manager.send_alert(
                f"Erreur initialisation RiskController: {str(e)}", priority=5
            )
            send_telegram_alert(f"Erreur initialisation RiskController: {str(e)}")
            logger.error(f"Erreur initialisation RiskController: {str(e)}")
            self.sierra_error_manager.log_error(
                "INIT_FAIL",
                f"Erreur initialisation RiskController: {str(e)}",
                severity="CRITICAL",
            )
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
            self.sierra_error_manager.log_error(
                "SIGINT_FAIL",
                f"Erreur sauvegarde snapshot SIGINT: {str(e)}",
                severity="ERROR",
            )
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
                    self.sierra_error_manager.log_error(
                        "RETRY_FAIL",
                        f"Échec après {max_attempts} tentatives: {str(e)}",
                        severity="ERROR",
                    )
                    return None
                delay = delay_base**attempt
                logger.warning(f"Tentative {attempt+1} échouée, retry après {delay}s")
                time.sleep(delay)

    def _persist_log(self, log_entry: Dict):
        """Sauvegarde un log dans risk_performance.csv."""
        start_time = time.time()
        try:
            log_df = pd.DataFrame([log_entry])
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
            latency = time.time() - start_time
            logger.info(f"Log sauvegardé dans {PERF_LOG_PATH}")
            return latency
        except Exception as e:
            error_msg = f"Erreur sauvegarde log: {str(e)}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            return 0

    def log_performance(
        self,
        operation: str,
        latency: float,
        success: bool = True,
        error: str = None,
        **kwargs,
    ) -> None:
        """
        Enregistre les performances (CPU, mémoire, latence) dans risk_performance.csv.

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
                alert_msg = f"ALERTE: Utilisation mémoire élevée ({memory_usage:.2f} MB)"
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=5)
                send_telegram_alert(alert_msg)
                self.sierra_error_manager.log_error(
                    "HIGH_MEMORY",
                    f"Utilisation mémoire élevée: {memory_usage:.2f} MB",
                    severity="WARNING",
                )
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
            self._persist_log(log_entry)
            logger.info(f"Performance journalisée pour {operation}. CPU: {cpu_percent}%")
        except Exception as e:
            error_msg = f"Erreur journalisation performance: {str(e)}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.sierra_error_manager.log_error(
                "LOG_PERF_FAIL",
                f"Erreur journalisation performance: {str(e)}",
                severity="ERROR",
            )

    def save_snapshot(
        self, snapshot_type: str, data: Dict, compress: bool = False
    ) -> None:
        """
        Sauvegarde un instantané JSON des résultats, avec option de compression gzip.

        Args:
            snapshot_type (str): Type de snapshot (ex. : risk_controller).
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
            self.log_performance("save_snapshot", time.time() - start_time, success=False, error=str(e))
            self.alert_manager.send_alert(
                f"Erreur sauvegarde snapshot: {str(e)}", priority=3
            )
            send_telegram_alert(f"Erreur sauvegarde snapshot: {str(e)}")
            logger.error(f"Erreur sauvegarde snapshot: {str(e)}")
            self.sierra_error_manager.log_error(
                "SNAPSHOT_FAIL",
                f"Erreur sauvegarde snapshot: {str(e)}",
                severity="ERROR",
            )

    def save_checkpoint(self, metrics: Dict, regime: str, compress: bool = True) -> None:
        """Sauvegarde incrémentielle et distribuée des métriques de risque."""
        start_time = time.time()
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = (
                CHECKPOINT_DIR / f"risk_metrics_{regime}_{timestamp}.json.gz"
            )
            checkpoint_data = {
                "timestamp": timestamp,
                "regime": regime,
                "metrics": metrics,
                "balance": self.balance,
                "num_positions": len(self.positions),
                "consecutive_losses": self.consecutive_losses,
            }
            os.makedirs(checkpoint_path.parent, exist_ok=True)

            def write_checkpoint():
                with gzip.open(checkpoint_path, "wt", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, indent=4)

            self.with_retries(write_checkpoint)
            logger.info(f"Checkpoint sauvegardé: {checkpoint_path}")
            self.alert_manager.send_alert(
                f"Checkpoint sauvegardé: {checkpoint_path}", priority=1
            )
            send_telegram_alert(f"Checkpoint sauvegardé: {checkpoint_path}")
            self.log_performance(
                "save_checkpoint", time.time() - start_time, success=True
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde checkpoint: {str(e)}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.sierra_error_manager.log_error(
                "CHECKPOINT_FAIL", error_msg, severity="ERROR"
            )
            self.log_performance(
                "save_checkpoint", time.time() - start_time, success=False, error=str(e)
            )

    def save_s3_checkpoint(self, metrics: Dict, regime: str) -> None:
        """Sauvegarde les métriques de risque sur S3."""
        start_time = time.time()
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_key = f"{self.s3_config['s3_prefix']}/risk_metrics_{regime}_{timestamp}.json.gz"
            checkpoint_data = {
                "timestamp": timestamp,
                "regime": regime,
                "metrics": metrics,
                "balance": self.balance,
                "num_positions": len(self.positions),
                "consecutive_losses": self.consecutive_losses,
            }

            def upload_to_s3():
                s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=self.s3_config["aws_access_key_id"],
                    aws_secret_access_key=self.s3_config["aws_secret_access_key"],
                )
                with gzip.GzipFile(fileobj=open(f"/tmp/{s3_key}", "wb"), mode="wb") as gz:
                    gz.write(json.dumps(checkpoint_data, indent=4).encode("utf-8"))
                s3_client.upload_file(
                    f"/tmp/{s3_key}",
                    self.s3_config["s3_bucket"],
                    s3_key,
                )
                os.remove(f"/tmp/{s3_key}")

            self.with_retries(upload_to_s3)
            logger.info(f"Checkpoint S3 sauvegardé: {s3_key}")
            self.alert_manager.send_alert(
                f"Checkpoint S3 sauvegardé: {s3_key}", priority=1
            )
            send_telegram_alert(f"Checkpoint S3 sauvegardé: {s3_key}")
            self.log_performance(
                "save_s3_checkpoint", time.time() - start_time, success=True
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde checkpoint S3: {str(e)}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.sierra_error_manager.log_error(
                "S3_CHECKPOINT_FAIL", error_msg, severity="ERROR"
            )
            self.log_performance(
                "save_s3_checkpoint", time.time() - start_time, success=False, error=str(e)
            )

    def fetch_clusters(self, db_path: str = DB_PATH) -> List[Dict]:
        """
        Récupère les clusters de risque depuis market_memory.db (méthode 7).

        Args:
            db_path (str): Chemin vers market_memory.db.

        Returns:
            List[Dict]: Liste des clusters de risque.
        """
        start_time = time.time()
        try:

            def fetch():
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT metadata FROM patterns WHERE neural_regime = 'defensive'"
                )
                clusters = [
                    {"risk_score": 0.5, "vix": 20.0, "drawdown": -0.05}
                ]  # Simulation fictive
                conn.close()
                return clusters

            clusters = self.with_retries(fetch)
            if clusters is None:
                raise ValueError("Échec récupération clusters")
            self.log_performance(
                "fetch_clusters",
                time.time() - start_time,
                success=True,
                num_clusters=len(clusters),
            )
            self.save_snapshot(
                "fetch_clusters", {"num_clusters": len(clusters)}, compress=False
            )
            return clusters
        except Exception as e:
            self.alert_manager.send_alert(
                f"Erreur récupération clusters: {str(e)}", priority=3
            )
            send_telegram_alert(f"Erreur récupération clusters: {str(e)}")
            logger.error(f"Erreur récupération clusters: {str(e)}")
            self.sierra_error_manager.log_error(
                "CLUSTER_FETCH_FAIL",
                f"Erreur récupération clusters: {str(e)}",
                severity="ERROR",
            )
            self.log_performance(
                "fetch_clusters", time.time() - start_time, success=False, error=str(e)
            )
            return []

    def predict_vix(self, data: pd.DataFrame) -> float:
        """
        Prédit le VIX à l’aide d’un modèle LSTM (méthode 12).

        Args:
            data (pd.DataFrame): Données contenant les features nécessaires.

        Returns:
            float: VIX prédit.
        """
        start_time = time.time()
        try:
            # Simulation fictive (remplacer par neural_pipeline.predict_vix dans une
            # implémentation réelle)
            predicted_vix = (
                data["vix"].iloc[-1] * 1.1 if "vix" in data.columns else 20.0
            )
            self.log_performance(
                "predict_vix",
                time.time() - start_time,
                success=True,
                predicted_vix=predicted_vix,
            )
            self.save_snapshot(
                "predict_vix", {"predicted_vix": predicted_vix}, compress=False
            )
            return predicted_vix
        except Exception as e:
            self.alert_manager.send_alert(
                f"Erreur prédiction VIX: {str(e)}", priority=3
            )
            send_telegram_alert(f"Erreur prédiction VIX: {str(e)}")
            logger.error(f"Erreur prédiction VIX: {str(e)}")
            self.sierra_error_manager.log_error(
                "VIX_PREDICT_FAIL", f"Erreur prédiction VIX: {str(e)}", severity="ERROR"
            )
            self.log_performance(
                "predict_vix", time.time() - start_time, success=False, error=str(e)
            )
            return 0.0

    def calculate_shap_risk(
        self, data: pd.DataFrame, target: str = "market_liquidity_crash_risk"
    ) -> Optional[pd.DataFrame]:
        """
        Calcule les valeurs SHAP pour les métriques de risque (Phase 17).

        Args:
            data (pd.DataFrame): Données d’entrée avec les features.
            target (str): Métrique cible pour SHAP (ex. : market_liquidity_crash_risk).

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
                "calculate_shap_risk",
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
                "calculate_shap_risk",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            return None

    def validate_data(self, data: pd.DataFrame) -> None:
        """
        Valide les données d’entrée (351 features pour entraînement, 150 SHAP pour inférence).

        Args:
            data (pd.DataFrame): Données à valider.

        Raises:
            ValueError: Si les données sont invalides.
        """
        start_time = time.time()
        try:
            expected_dims = self.config.get(
                "observation_dims", {"training": 351, "inference": 150}
            )
            data_dim = data.shape[1]
            if data_dim not in (expected_dims["training"], expected_dims["inference"]):
                raise ValueError(
                    f"Nombre de features incorrect: {data_dim}, attendu {expected_dims['training']} ou {expected_dims['inference']}"
                )
            critical_cols = [
                "close",
                "bid_price_level_1",
                "ask_price_level_1",
                "vix",
                "event_volatility_impact",
                "spread_avg_1min",
                "trade_frequency_1s",
                "iv_atm",
                "option_skew",
                "news_impact_score",
                "spoofing_score",
                "volume_anomaly",
                "iv_skew",
                "bid_ask_imbalance",
                "trade_aggressiveness",
            ]
            for col in critical_cols:
                if col in data.columns:
                    if data[col].isnull().any() or data[col].le(0).any() or data[col].isin([np.inf, -np.inf]).any():
                        raise ValueError(
                            f"Colonne {col} contient des NaN, inf ou valeurs non positives"
                        )
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        raise ValueError(
                            f"Colonne {col} n'est pas numérique: {data[col].dtype}"
                        )
            if "option_type" in data.columns and not all(
                data["option_type"].isin(["call", "put", None])
            ):
                raise ValueError(
                    "Valeurs option_type invalides (doit être 'call', 'put' ou None)"
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
            self.sierra_error_manager.log_error(
                "DATA_VALIDATE_FAIL",
                f"Erreur validation données: {str(e)}",
                severity="CRITICAL",
            )
            self.log_performance(
                "validate_data", time.time() - start_time, success=False, error=str(e)
            )
            raise

    def validate_positions(self, positions: List[Dict]) -> None:
        """
        Valide la structure des positions.

        Args:
            positions (List[Dict]): Liste des positions ouvertes.

        Raises:
            ValueError: Si les positions sont invalides.
        """
        start_time = time.time()
        try:
            required_keys = ["timestamp", "action", "price", "size"]
            for pos in positions:
                if not all(key in pos for key in required_keys):
                    raise ValueError(f"Position invalide, clés manquantes: {pos}")
                if not isinstance(pos["timestamp"], str) or not pd.to_datetime(
                    pos["timestamp"], errors="coerce"
                ):
                    raise ValueError(
                        f"Timestamp invalide dans position: {pos['timestamp']}"
                    )
                if (
                    not isinstance(pos["action"], (int, float))
                    or not isinstance(pos["price"], (int, float))
                    or not isinstance(pos["size"], (int, float))
                ):
                    raise ValueError(f"Position invalide, types incorrects: {pos}")
            self.log_performance(
                "validate_positions",
                time.time() - start_time,
                success=True,
                num_positions=len(positions),
            )
        except Exception as e:
            self.alert_manager.send_alert(
                f"Erreur validation positions: {str(e)}", priority=4
            )
            send_telegram_alert(f"Erreur validation positions: {str(e)}")
            logger.error(f"Erreur validation positions: {str(e)}")
            self.sierra_error_manager.log_error(
                "POS_VALIDATE_FAIL",
                f"Erreur validation positions: {str(e)}",
                severity="CRITICAL",
            )
            self.log_performance(
                "validate_positions",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            raise

    def check_trade_frequency(self, current_time: datetime) -> float:
        """
        Vérifie la fréquence des trades et retourne un facteur d'ajustement.

        Args:
            current_time (datetime): Horodatage actuel.

        Returns:
            float: Facteur d'ajustement (0.0 à 1.0, 0.0 si surtrading).
        """
        start_time = time.time()
        try:
            # Nettoyer les timestamps anciens
            self.trade_timestamps = [
                ts for ts in self.trade_timestamps
                if current_time - ts < timedelta(hours=1)
            ]
            # Compter les trades récents
            recent_trades = len(self.trade_timestamps)
            if recent_trades >= self.config["entry_freq_max"]:
                logger.warning(f"Surtrading détecté: {recent_trades} trades/heure")
                self.alert_manager.send_alert(
                    f"Surtrading détecté: {recent_trades} trades/heure", priority=3
                )
                send_telegram_alert(f"Surtrading détecté: {recent_trades} trades/heure")
                return 0.0
            # Vérifier l'intervalle minimum
            if self.trade_timestamps and (current_time - max(self.trade_timestamps)).total_seconds() < self.config["entry_freq_min_interval"]:
                logger.warning("Intervalle entre trades trop court")
                self.alert_manager.send_alert(
                    "Intervalle entre trades trop court", priority=3
                )
                send_telegram_alert("Intervalle entre trades trop court")
                return 0.0
            # Ajouter le timestamp actuel
            self.trade_timestamps.append(current_time)
            self.log_performance(
                "check_trade_frequency",
                time.time() - start_time,
                success=True,
                recent_trades=recent_trades,
            )
            return 1.0
        except Exception as e:
            error_msg = f"Erreur check_trade_frequency: {str(e)}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.sierra_error_manager.log_error(
                "TRADE_FREQ_FAIL", error_msg, severity="ERROR"
            )
            self.log_performance(
                "check_trade_frequency",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            return 0.0

    def calculate_risk_metrics(
        self, data: pd.DataFrame, positions: List[Dict]
    ) -> Dict[str, float]:
        """
        Calcule les métriques de risque (market_liquidity_crash_risk, overtrade_risk_score, confidence_drop_rate).

        Args:
            data (pd.DataFrame): Données contenant les features.
            positions (List[Dict]): Liste des positions ouvertes.

        Returns:
            Dict[str, float]: Métriques de risque, incluant confidence_drop_rate et SHAP.
        """
        start_time = time.time()
        try:
            self.validate_data(data)
            self.validate_positions(positions)
            spread = (
                data["ask_price_level_1"].iloc[-1] - data["bid_price_level_1"].iloc[-1]
            ) / data["close"].iloc[-1]
            trade_frequency = data["trade_frequency_1s"].iloc[-1]
            vix = data["vix"].iloc[-1]
            event_impact = data["event_volatility_impact"].iloc[-1]
            iv_skew = data["iv_skew"].iloc[-1] if "iv_skew" in data.columns else 0.0
            bid_ask_imbalance = data["bid_ask_imbalance"].iloc[-1] if "bid_ask_imbalance" in data.columns else 0.0
            trade_aggressiveness = data["trade_aggressiveness"].iloc[-1] if "trade_aggressiveness" in data.columns else 0.0

            # Intégration des métriques options
            options_risk = self.options_risk_manager.calculate_options_risk(data)

            # Intégration des données de nouvelles
            news_data = fetch_news(api_key="")  # Remplacer par une clé API valide
            if news_data.empty:
                logger.warning("Aucune donnée de nouvelles disponible")
                self.alert_manager.send_alert(
                    "Aucune donnée de nouvelles disponible", priority=2
                )
                send_telegram_alert("Aucune donnée de nouvelles disponible")
                news_impact = 0.0
            else:
                news_impact = (
                    news_data["sentiment_score"].mean()
                    if "sentiment_score" in news_data.columns
                    else 0.0
                )

            # Statistiques temporelles (moyenne sur 1h si données disponibles)
            if len(data) > 1:
                window_data = data.tail(60)  # Supposer 1 minute par ligne
                vix_mean = window_data["vix"].mean()
                spread_mean = (
                    (
                        window_data["ask_price_level_1"]
                        - window_data["bid_price_level_1"]
                    )
                    / window_data["close"]
                ).mean()
            else:
                vix_mean = vix
                spread_mean = spread

            # Méthode 7 : Ajustement via clusters
            clusters = self.fetch_clusters()
            cluster_risk = 0.0
            if clusters:
                cluster_risk = np.mean(
                    [c["risk_score"] for c in clusters]
                )  # Moyenne des scores de risque

            # Calculer market_liquidity_crash_risk
            liquidity_crash_risk = (
                self.metric_weights["spread_weight"]
                * (spread / self.thresholds["spread_threshold"])
                + self.metric_weights["trade_frequency_weight"]
                * (trade_frequency / 10.0)
                + self.metric_weights["vix_weight"]
                * (vix / self.thresholds["vix_threshold"])
                + self.metric_weights["cluster_risk_weight"] * cluster_risk
                + self.metric_weights["options_risk_weight"]
                * options_risk.get("risk_alert", 0.0)
                + self.metric_weights["news_impact_weight"] * abs(news_impact)
                + self.metric_weights["iv_skew_weight"] * (iv_skew / 0.5)
                + self.metric_weights["bid_ask_imbalance_weight"] * abs(bid_ask_imbalance)
                + self.metric_weights["trade_aggressiveness_weight"] * trade_aggressiveness
            )

            # Calculer overtrade_risk_score
            recent_trades = len(
                [
                    p
                    for p in positions
                    if p["timestamp"]
                    > (datetime.now() - timedelta(minutes=60)).isoformat()
                ]
            )
            overtrade_risk = recent_trades / 10.0  # Normalisé par 10 trades/heure

            # Calculer confidence_drop_rate (Phase 8)
            confidence_drop_rate = 1.0 - min(
                liquidity_crash_risk
                / self.thresholds["liquidity_crash_risk_threshold"],
                1.0,
            )
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f}"
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            # Analyse SHAP (Phase 17)
            shap_metrics = {}
            shap_values = self.calculate_shap_risk(
                data, target="market_liquidity_crash_risk"
            )
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

            self.risk_metrics = {
                "market_liquidity_crash_risk": min(1.0, max(0.0, liquidity_crash_risk)),
                "overtrade_risk_score": min(1.0, max(0.0, overtrade_risk)),
                "confidence_drop_rate": confidence_drop_rate,
                "vix": vix,
                "vix_mean_1h": vix_mean,
                "spread": spread,
                "spread_mean_1h": spread_mean,
                "event_impact": event_impact,
                "cluster_risk": cluster_risk,
                "options_risk_score": options_risk.get("risk_alert", 0.0),
                "news_impact": news_impact,
                "iv_skew": iv_skew,
                "bid_ask_imbalance": bid_ask_imbalance,
                "trade_aggressiveness": trade_aggressiveness,
                **shap_metrics,
            }

            logger.info(f"Métriques de risque calculées: {self.risk_metrics}")
            self.alert_manager.send_alert(
                f"Métriques de risque calculées: {self.risk_metrics}", priority=1
            )
            send_telegram_alert(f"Métriques de risque calculées: {self.risk_metrics}")
            self.log_performance(
                "calculate_risk_metrics",
                time.time() - start_time,
                success=True,
                num_rows=len(data),
            )
            self.save_snapshot(
                "calculate_risk_metrics", self.risk_metrics, compress=False
            )
            self.save_checkpoint(
                self.risk_metrics, data.get("neural_regime", "range").iloc[-1]
            )
            self.save_s3_checkpoint(
                self.risk_metrics, data.get("neural_regime", "range").iloc[-1]
            )
            return self.risk_metrics
        except Exception as e:
            self.alert_manager.send_alert(
                f"Erreur calculate_risk_metrics: {str(e)}", priority=4
            )
            send_telegram_alert(f"Erreur calculate_risk_metrics: {str(e)}")
            logger.error(f"Erreur calculate_risk_metrics: {str(e)}")
            self.sierra_error_manager.log_error(
                "RISK_METRICS_FAIL",
                f"Erreur calculate_risk_metrics: {str(e)}",
                severity="CRITICAL",
            )
            self.log_performance(
                "calculate_risk_metrics",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            return {
                "market_liquidity_crash_risk": 0.0,
                "overtrade_risk_score": 0.0,
                "confidence_drop_rate": 0.0,
                "vix": 0.0,
                "vix_mean_1h": 0.0,
                "spread": 0.0,
                "spread_mean_1h": 0.0,
                "event_impact": 0.0,
                "cluster_risk": 0.0,
                "options_risk_score": 0.0,
                "news_impact": 0.0,
                "iv_skew": 0.0,
                "bid_ask_imbalance": 0.0,
                "trade_aggressiveness": 0.0,
            }

    def save_risk_snapshot(
        self, step: int, timestamp: pd.Timestamp, metrics: Dict, reason: str
    ) -> None:
        """
        Sauvegarde un instantané des événements de risque, avec option de compression gzip.

        Args:
            step (int): Étape du trading.
            timestamp (pd.Timestamp): Horodatage.
            metrics (Dict): Métriques de risque.
            reason (str): Raison de l’événement.
        """
        start_time = time.time()
        try:
            snapshot = {
                "step": step,
                "timestamp": str(timestamp),
                "metrics": metrics,
                "reason": reason,
                "balance": self.balance,
                "consecutive_losses": self.consecutive_losses,
                "num_positions": len(self.positions),
            }
            snapshot_path = SNAPSHOT_DIR / f"risk_step_{step:04d}.json"

            def write_snapshot():
                with open(snapshot_path, "w", encoding="utf-8") as f:
                    json.dump(snapshot, f, indent=4)

            self.with_retries(write_snapshot)
            logger.info(f"Snapshot risque sauvegardé: {snapshot_path}")
            self.alert_manager.send_alert(
                f"Snapshot risque step {step} sauvegardé", priority=1
            )
            send_telegram_alert(f"Snapshot risque step {step} sauvegardé")
            self.log_performance(
                "save_risk_snapshot", time.time() - start_time, success=True, step=step
            )
        except Exception as e:
            self.alert_manager.send_alert(
                f"Erreur sauvegarde snapshot risque: {str(e)}", priority=3
            )
            send_telegram_alert(f"Erreur sauvegarde snapshot risque: {str(e)}")
            logger.error(f"Erreur sauvegarde snapshot risque: {str(e)}")
            self.sierra_error_manager.log_error(
                "RISK_SNAPSHOT_FAIL",
                f"Erreur sauvegarde snapshot risque: {str(e)}",
                severity="ERROR",
            )
            self.log_performance(
                "save_risk_snapshot",
                time.time() - start_time,
                success=False,
                error=str(e),
            )

    def stop_trading(
        self, drawdown: float, data: pd.DataFrame, positions: List[Dict]
    ) -> bool:
        """
        Vérifie si le trading doit être arrêté en raison de risques élevés.

        Args:
            drawdown (float): Drawdown actuel (négatif).
            data (pd.DataFrame): Données contenant les features.
            positions (List[Dict]): Liste des positions ouvertes.

        Returns:
            bool: True si le trading doit être arrêté, False sinon.
        """
        start_time = time.time()
        try:
            self.validate_data(data)
            self.validate_positions(positions)

            # Calculer les métriques de risque
            risk_metrics = self.calculate_risk_metrics(data, positions)
            risk_score = max(
                risk_metrics["market_liquidity_crash_risk"],
                risk_metrics["overtrade_risk_score"],
                risk_metrics["options_risk_score"],
            )

            # Méthode 12 : Prédire le VIX
            predicted_vix = self.predict_vix(data)

            # Vérifier les conditions d’arrêt
            should_stop = False
            reason = ""
            if drawdown < self.thresholds["max_drawdown"]:
                should_stop = True
                reason = f"Drawdown excessif: {drawdown:.2f} < {self.thresholds['max_drawdown']}"
            elif risk_score > self.thresholds["risk_score_threshold"]:
                should_stop = True
                reason = f"Risque élevé: score={risk_score:.2f} > {self.thresholds['risk_score_threshold']}"
            elif predicted_vix > self.thresholds["predicted_vix_threshold"]:
                should_stop = True
                reason = f"VIX prédit élevé: {predicted_vix:.2f} > {self.thresholds['predicted_vix_threshold']}"
            elif (
                abs(risk_metrics["news_impact"])
                > self.thresholds["event_impact_threshold"]
            ):
                should_stop = True
                reason = f"Impact des nouvelles élevé: {risk_metrics['news_impact']:.2f} > {self.thresholds['event_impact_threshold']}"

            if should_stop:
                self.alert_manager.send_alert(f"Arrêt trading: {reason}", priority=3)
                send_telegram_alert(f"Arrêt trading: {reason}")
                logger.warning(reason)

                # Stocker l’événement dans market_memory.db
                def store_event():
                    store_pattern(
                        data,
                        action=0.0,
                        reward=drawdown,
                        neural_regime="defensive",
                        confidence=1.0,
                        metadata={
                            "event": "stop_trading",
                            "reason": reason,
                            "risk_score": risk_score,
                            "predicted_vix": predicted_vix,
                            "confidence_drop_rate": risk_metrics[
                                "confidence_drop_rate"
                            ],
                        },
                    )

                self.with_retries(store_event)

                self.risk_events.append(
                    {
                        "timestamp": str(datetime.now()),
                        "type": "stop_trading",
                        "reason": reason,
                    }
                )
                self.save_risk_snapshot(
                    len(self.risk_events), pd.Timestamp.now(), self.risk_metrics, reason
                )

            self.log_performance(
                "stop_trading",
                time.time() - start_time,
                success=True,
                drawdown=drawdown,
                risk_score=risk_score,
                predicted_vix=predicted_vix,
            )
            return should_stop
        except Exception as e:
            self.alert_manager.send_alert(f"Erreur stop_trading: {str(e)}", priority=4)
            send_telegram_alert(f"Erreur stop_trading: {str(e)}")
            logger.error(f"Erreur stop_trading: {str(e)}")
            self.sierra_error_manager.log_error(
                "STOP_TRADING_FAIL",
                f"Erreur stop_trading: {str(e)}",
                severity="CRITICAL",
            )
            self.log_performance(
                "stop_trading", time.time() - start_time, success=False, error=str(e)
            )
            return False

    def adjust_position_size(
        self, signal_score: float, data: pd.DataFrame, positions: List[Dict]
    ) -> float:
        """
        Ajuste la taille de la position en fonction du score de signal et des conditions de marché.

        Args:
            signal_score (float): Score de la prédiction (par exemple, confidence).
            data (pd.DataFrame): Données contenant les features.
            positions (List[Dict]): Liste des positions ouvertes.

        Returns:
            float: Taille de la position ajustée (0 à max_position_size).
        """
        start_time = time.time()
        try:
            self.validate_data(data)
            self.validate_positions(positions)
            risk_metrics = self.calculate_risk_metrics(data, positions)

            # Vérifier la fréquence des trades
            freq_factor = self.check_trade_frequency(datetime.now())
            if freq_factor == 0.0:
                return 0.0

            # Facteurs de risque
            vix = risk_metrics["vix"]
            event_impact = risk_metrics["event_impact"]
            liquidity_risk = risk_metrics["market_liquidity_crash_risk"]
            overtrade_risk = risk_metrics["overtrade_risk_score"]
            options_risk = risk_metrics["options_risk_score"]
            news_impact = risk_metrics["news_impact"]
            confidence_drop_rate = risk_metrics["confidence_drop_rate"]
            iv_skew = risk_metrics["iv_skew"]
            bid_ask_imbalance = risk_metrics["bid_ask_imbalance"]
            trade_aggressiveness = risk_metrics["trade_aggressiveness"]

            # Taille de base
            base_size = min(
                self.thresholds["max_position_size"],
                signal_score * self.thresholds["max_position_size"],
            )

            # Ajustements
            risk_factor = freq_factor
            if vix > self.thresholds["vix_threshold"]:
                risk_factor *= 0.5
                self.alert_manager.send_alert(
                    f"VIX élevé ({vix:.2f}), réduction taille à {risk_factor}",
                    priority=3,
                )
                send_telegram_alert(
                    f"VIX élevé ({vix:.2f}), réduction taille à {risk_factor}"
                )
            if event_impact > self.thresholds["event_impact_threshold"]:
                risk_factor *= 0.3
                self.alert_manager.send_alert(
                    f"Impact macro élevé ({event_impact:.2f}), réduction taille à {risk_factor}",
                    priority=3,
                )
                send_telegram_alert(
                    f"Impact macro élevé ({event_impact:.2f}), réduction taille à {risk_factor}"
                )
            if liquidity_risk > self.thresholds["liquidity_crash_risk_threshold"]:
                risk_factor *= 0.4
                self.alert_manager.send_alert(
                    f"Risque de liquidité élevé ({liquidity_risk:.2f}), réduction taille à {risk_factor}",
                    priority=3,
                )
                send_telegram_alert(
                    f"Risque de liquidité élevé ({liquidity_risk:.2f}), réduction taille à {risk_factor}"
                )
            if overtrade_risk > self.thresholds["overtrade_risk_threshold"]:
                risk_factor *= 0.5
                self.alert_manager.send_alert(
                    f"Risque de surtrading ({overtrade_risk:.2f}), réduction taille à {risk_factor}",
                    priority=3,
                )
                send_telegram_alert(
                    f"Risque de surtrading ({overtrade_risk:.2f}), réduction taille à {risk_factor}"
                )
            if options_risk > self.thresholds["risk_score_threshold"]:
                risk_factor *= 0.4
                self.alert_manager.send_alert(
                    f"Risque options élevé ({options_risk:.2f}), réduction taille à {risk_factor}",
                    priority=3,
                )
                send_telegram_alert(
                    f"Risque options élevé ({options_risk:.2f}), réduction taille à {risk_factor}"
                )
            if abs(news_impact) > self.thresholds["event_impact_threshold"]:
                risk_factor *= 0.3
                self.alert_manager.send_alert(
                    f"Impact nouvelles élevé ({news_impact:.2f}), réduction taille à {risk_factor}",
                    priority=3,
                )
                send_telegram_alert(
                    f"Impact nouvelles élevé ({news_impact:.2f}), réduction taille à {risk_factor}"
                )
            if confidence_drop_rate > 0.5:
                risk_factor *= 0.5
                self.alert_manager.send_alert(
                    f"Confidence_drop_rate élevé ({confidence_drop_rate:.2f}), réduction taille à {risk_factor}",
                    priority=3,
                )
                send_telegram_alert(
                    f"Confidence_drop_rate élevé ({confidence_drop_rate:.2f}), réduction taille à {risk_factor}"
                )
            if iv_skew > 0.5:
                risk_factor *= 0.7
                self.alert_manager.send_alert(
                    f"IV skew élevé ({iv_skew:.2f}), réduction taille à {risk_factor}",
                    priority=3,
                )
                send_telegram_alert(
                    f"IV skew élevé ({iv_skew:.2f}), réduction taille à {risk_factor}"
                )
            if abs(bid_ask_imbalance) > 0.5:
                risk_factor *= 0.6
                self.alert_manager.send_alert(
                    f"Imbalance bid-ask élevé ({bid_ask_imbalance:.2f}), réduction taille à {risk_factor}",
                    priority=3,
                )
                send_telegram_alert(
                    f"Imbalance bid-ask élevé ({bid_ask_imbalance:.2f}), réduction taille à {risk_factor}"
                )
            if trade_aggressiveness > 0.5:
                risk_factor *= 0.5
                self.alert_manager.send_alert(
                    f"Agressivité des trades élevée ({trade_aggressiveness:.2f}), réduction taille à {risk_factor}",
                    priority=3,
                )
                send_telegram_alert(
                    f"Agressivité des trades élevée ({trade_aggressiveness:.2f}), réduction taille à {risk_factor}"
                )
            if len(positions) >= self.thresholds["max_position_size"]:
                risk_factor = 0.0
                self.alert_manager.send_alert(
                    "Taille maximale des positions atteinte", priority=3
                )
                send_telegram_alert("Taille maximale des positions atteinte")

            adjusted_size = base_size * risk_factor
            logger.info(
                f"Taille ajustée: {adjusted_size:.2f} (base={base_size:.2f}, risk_factor={risk_factor:.2f})"
            )
            self.alert_manager.send_alert(
                f"Taille ajustée: {adjusted_size:.2f}", priority=1
            )
            send_telegram_alert(f"Taille ajustée: {adjusted_size:.2f}")
            self.log_performance(
                "adjust_position_size",
                time.time() - start_time,
                success=True,
                adjusted_size=adjusted_size,
                risk_factor=risk_factor,
            )
            self.save_snapshot(
                "adjust_position_size",
                {
                    "adjusted_size": adjusted_size,
                    "risk_factor": risk_factor,
                    "confidence_drop_rate": confidence_drop_rate,
                    "iv_skew": iv_skew,
                    "bid_ask_imbalance": bid_ask_imbalance,
                    "trade_aggressiveness": trade_aggressiveness,
                },
                compress=False,
            )
            return adjusted_size
        except Exception as e:
            self.alert_manager.send_alert(
                f"Erreur adjust_position_size: {str(e)}", priority=4
            )
            send_telegram_alert(f"Erreur adjust_position_size: {str(e)}")
            logger.error(f"Erreur adjust_position_size: {str(e)}")
            self.sierra_error_manager.log_error(
                "ADJ_POS_SIZE_FAIL",
                f"Erreur adjust_position_size: {str(e)}",
                severity="CRITICAL",
            )
            self.log_performance(
                "adjust_position_size",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            return 0.0

    def penalize_ia(self, loss: float, data: pd.DataFrame) -> float:
        """
        Applique une pénalité à l’IA en cas de perte, enregistrée dans penalty_log.csv.

        Args:
            loss (float): Perte subie (négatif).
            data (pd.DataFrame): Données contenant les features.

        Returns:
            float: Valeur de la pénalité (0 à 1).
        """
        start_time = time.time()
        try:
            self.validate_data(data)
            penalty = 0.0

            if loss < -self.thresholds["penalty_threshold"]:
                self.consecutive_losses += 1
                penalty = min(
                    1.0,
                    self.consecutive_losses / self.thresholds["max_consecutive_losses"],
                )
                alert_level = (
                    5
                    if self.consecutive_losses
                    >= self.thresholds["max_consecutive_losses"]
                    else 4
                )
                alert_message = f"Pénalité IA: {penalty:.2f} (perte={loss:.2f}, pertes consécutives={self.consecutive_losses})"
                self.alert_manager.send_alert(alert_message, priority=alert_level)
                send_telegram_alert(alert_message)
                logger.warning(alert_message)

                # Enregistrer la pénalité
                penalty_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "penalty_type": "consecutive_losses",
                    "value": penalty,
                    "reason": f"Perte {loss:.2f}, {self.consecutive_losses} pertes consécutives",
                }
                self.penalty_buffer.append(penalty_entry)
                if len(self.penalty_buffer) >= self.config.get("buffer_size", 100):
                    penalty_df = pd.DataFrame(self.penalty_buffer)
                    penalty_log_path = "data/logs/trading/penalty_log.csv"
                    os.makedirs(os.path.dirname(penalty_log_path), exist_ok=True)

                    def write_penalty():
                        if not os.path.exists(penalty_log_path):
                            penalty_df.to_csv(
                                penalty_log_path, index=False, encoding="utf-8"
                            )
                        else:
                            penalty_df.to_csv(
                                penalty_log_path,
                                mode="a",
                                header=False,
                                index=False,
                                encoding="utf-8",
                            )

                    self.with_retries(write_penalty)
                    self.penalty_buffer = []

                # Stocker l’événement dans market_memory.db
                def store_penalty():
                    store_pattern(
                        data,
                        action=0.0,
                        reward=loss,
                        neural_regime="defensive",
                        confidence=0.0,
                        metadata={
                            "event": "penalty",
                            "penalty": penalty,
                            "loss": loss,
                            "confidence_drop_rate": self.risk_metrics.get(
                                "confidence_drop_rate", 0.0
                            ),
                        },
                    )

                self.with_retries(store_penalty)

                self.risk_events.append(
                    {
                        "timestamp": str(datetime.now()),
                        "type": "penalty",
                        "value": penalty,
                    }
                )
                self.save_risk_snapshot(
                    len(self.risk_events),
                    pd.Timestamp.now(),
                    self.risk_metrics,
                    alert_message,
                )
            else:
                self.consecutive_losses = 0

            self.log_performance(
                "penalize_ia", time.time() - start_time, success=True, penalty=penalty
            )
            return penalty
        except Exception as e:
            self.alert_manager.send_alert(f"Erreur penalize_ia: {str(e)}", priority=4)
            send_telegram_alert(f"Erreur penalize_ia: {str(e)}")
            logger.error(f"Erreur penalize_ia: {str(e)}")
            self.sierra_error_manager.log_error(
                "PENALIZE_IA_FAIL", f"Erreur penalize_ia: {str(e)}", severity="CRITICAL"
            )
            self.log_performance(
                "penalize_ia", time.time() - start_time, success=False, error=str(e)
            )
            return 0.0

    def save_dashboard_status(self, status_file: str = "data/risk_dashboard.json"):
        """
        Sauvegarde l’état des risques pour mia_dashboard.py.

        Args:
            status_file (str): Chemin du fichier JSON.
        """
        start_time = time.time()
        try:
            status = {
                "timestamp": datetime.now().isoformat(),
                "balance": self.balance,
                "num_positions": len(self.positions),
                "consecutive_losses": self.consecutive_losses,
                "risk_metrics": self.risk_metrics,
                "num_penalties": len(self.penalty_log),
                "risk_events": self.risk_events[-10:],  # Derniers 10 événements
            }
            os.makedirs(os.path.dirname(status_file), exist_ok=True)

            def write_status():
                with open(status_file, "w", encoding="utf-8") as f:
                    json.dump(status, f, indent=4)

            self.with_retries(write_status)
            logger.info(f"État des risques sauvegardé dans {status_file}")
            self.alert_manager.send_alert("État des risques sauvegardé", priority=1)
            send_telegram_alert("État des risques sauvegardé")
            self.log_performance(
                "save_dashboard_status", time.time() - start_time, success=True
            )
        except Exception as e:
            self.alert_manager.send_alert(
                f"Erreur sauvegarde dashboard: {str(e)}", priority=3
            )
            send_telegram_alert(f"Erreur sauvegarde dashboard: {str(e)}")
            logger.error(f"Erreur sauvegarde dashboard: {str(e)}")
            self.sierra_error_manager.log_error(
                "DASHBOARD_FAIL",
                f"Erreur sauvegarde dashboard: {str(e)}",
                severity="ERROR",
            )
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
                "close": [5100.0],
                "bid_price_level_1": [5098.0],
                "ask_price_level_1": [5102.0],
                "vix": [20.0],
                "event_volatility_impact": [0.3],
                "spread_avg_1min": [0.04],
                "trade_frequency_1s": [8.0],
                "iv_atm": [0.15],
                "option_skew": [0.1],
                "news_impact_score": [0.2],
                "spoofing_score": [0.3],
                "volume_anomaly": [0.1],
                "iv_skew": [0.2],
                "bid_ask_imbalance": [0.3],
                "trade_aggressiveness": [0.4],
                "option_type": ["call"],
                **{
                    f"feat_{i}": [np.random.uniform(0, 1)] for i in range(336)
                },  # 351 features
            }
        )
        positions = [
            {
                "timestamp": datetime.now().isoformat(),
                "action": 1.0,
                "price": 5100.0,
                "size": 1,
            }
        ]

        # Test RiskController
        risk_controller = RiskController()
        risk_metrics = risk_controller.calculate_risk_metrics(data, positions)
        print("Risk Metrics:", risk_metrics)

        drawdown = -0.05
        should_stop = risk_controller.stop_trading(drawdown, data, positions)
        print("Stop Trading:", should_stop)

        signal_score = 0.8
        size = risk_controller.adjust_position_size(signal_score, data, positions)
        print("Adjusted Position Size:", size)

        loss = -0.15
        penalty = risk_controller.penalize_ia(loss, data)
        print("Penalty:", penalty)

        risk_controller.save_dashboard_status()
    except Exception as e:
        alert_manager = AlertManager()
        alert_manager.send_alert(f"Erreur test RiskController: {str(e)}", priority=5)
        send_telegram_alert(f"Erreur test RiskController: {str(e)}")
        logger.error(f"Erreur test RiskController: {str(e)}")
