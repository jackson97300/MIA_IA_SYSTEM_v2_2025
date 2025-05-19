```python
# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/risk_management/risk_manager.py
# Gère le position sizing dynamique avec Kelly/ATR pour MIA_IA_SYSTEM_v2_2025.
#
# Version: 2.1.6
# Date: 2025-05-15
#
# Rôle : Calcule la taille des positions en fonction de l'ATR, de l'imbalance du carnet d'ordres,
#        et des nouvelles features (iv_skew, bid_ask_imbalance, trade_aggressiveness),
#        avec une limite configurable et un cache léger.
# Utilisé par : mia_switcher.py pour ajuster les tailles de position avant chaque ordre.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.8,<6.0.0, boto3>=1.28.0,<2.0.0
# - src/data/feature_pipeline.py (atr_dynamic, orderflow_imbalance, volatility_score, iv_skew, bid_ask_imbalance, trade_aggressiveness)
# - src/monitoring/prometheus_metrics.py (métriques position_size, position_size_variance)
# - src/model/utils/alert_manager.py (gestion des alertes)
# - src/utils/error_tracker.py (capture des erreurs)
#
# Inputs :
# - config/risk_manager_config.yaml (constantes)
# - config/s3_config.yaml (configuration S3)
# - data/market_memory.db (données order flow via feature_pipeline.py)
#
# Outputs :
# - data/logs/risk_manager_performance.csv (logs des performances)
# - data/risk_snapshots/*.json (snapshots locaux)
# - Sauvegardes S3 dans le bucket configuré (config/s3_config.yaml)
#
# Notes :
# - Conforme à structure.txt (version 2.1.6, 2025-05-15).
# - Supporte la suggestion 5 (profit factor) en stabilisant les rendements.
# - Intègre retries (max 3, délai 2^attempt), logs psutil, alertes via AlertManager.
# - Cache léger pour optimiser les calculs en scalping/HFT, incluant volatility_score, iv_skew, bid_ask_imbalance, trade_aggressiveness.
# - Contrôle de la fréquence des trades avec seuils entry_freq_max et entry_freq_min_interval.
# - Tests unitaires dans tests/test_risk_manager.py, étendus pour iv_skew, bid_ask_imbalance, trade_aggressiveness, S3, et fréquence des trades.
# - Refactorisation de log_performance avec _persist_log pour éviter la récursion infinie.
# - Policies Note: The official directory for routing policies is src/model/router/policies.
#   The src/model/policies directory is a residual and should be verified for removal to avoid import conflicts.

import os
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import boto3
import numpy as np
import pandas as pd
import psutil
from loguru import logger

from src.model.utils.alert_manager import AlertManager
from src.monitoring.prometheus_metrics import Gauge
from src.utils.error_tracker import capture_error

# Configuration logging
logger.remove()
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOG_DIR = BASE_DIR / "data" / "logs"
SNAPSHOT_DIR = BASE_DIR / "data" / "risk_snapshots"
S3_CONFIG_PATH = BASE_DIR / "config/s3_config.yaml"

# Métriques Prometheus
position_size_metric = Gauge(
    "position_size_percent",
    "Taille de la position en pourcentage du capital",
    ["market"],
)
position_size_variance = Gauge(
    "position_size_variance",
    "Variance des tailles de position sur une fenêtre glissante",
    ["market"],
)


class RiskManager:
    """
    Gère le position sizing dynamique avec Kelly/ATR.

    Attributes:
        market (str): Marché cible (ex. : 'ES').
        capital (float): Capital disponible pour le trading.
        log_buffer (list): Buffer pour les logs de performance.
        config (dict): Configuration chargée via config_manager.
        s3_config (dict): Configuration S3 pour les sauvegardes.
        size_cache (OrderedDict): Cache des tailles calculées.
        recent_sizes (list): Fenêtre glissante des tailles pour variance.
        alert_manager (AlertManager): Gestionnaire d'alertes.
        trade_timestamps (list): Historique des timestamps des trades.
    """

    def __init__(self, market: str = "ES", capital: float = 100000.0):
        """Initialise le gestionnaire de risques."""
        start_time = datetime.now()
        try:
            self.market = market
            self.capital = capital
            self.log_buffer = []
            self.size_cache = OrderedDict()
            self.cache_max_size = 100
            self.recent_sizes = []
            self.variance_window = 100
            self.trade_timestamps = []
            self.alert_manager = AlertManager()
            from src.model.utils.config_manager import get_config

            self.config = self.with_retries(
                lambda: get_config(BASE_DIR / "config/risk_manager_config.yaml")
            )
            self.s3_config = self.with_retries(
                lambda: get_config(S3_CONFIG_PATH)
            )
            self._validate_config()
            LOG_DIR.mkdir(exist_ok=True)
            SNAPSHOT_DIR.mkdir(exist_ok=True)
            success_msg = f"RiskManager initialisé pour {market} avec capital {capital}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=2)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("init", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur initialisation RiskManager: {str(e)}"
            logger.error(error_msg)
            capture_error(
                e,
                context={"market": market},
                market=market,
                operation="init_risk_manager",
            )
            self.alert_manager.send_alert(error_msg, priority=5)
            self.log_performance("init", 0, success=False, error=str(e))
            raise

    def _validate_config(self) -> None:
        """Valide la structure du fichier de configuration."""
        required_keys = [
            "buffer_size",
            "max_retries",
            "retry_delay",
            "kelly_fraction",
            "max_position_fraction",
            "entry_freq_max",
            "entry_freq_min_interval",
        ]
        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            error_msg = f"Clés manquantes dans risk_manager_config.yaml: {missing_keys}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def with_retries(
        self, func: callable, max_attempts: int = None, delay_base: float = None
    ) -> Optional[any]:
        """Exécute une fonction avec retries exponentiels."""
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
                        operation="retry_risk_manager",
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
                if attempt >= 1:  # Alerte seulement après le 2ᵉ échec
                    warning_msg = f"Tentative {attempt+1} échouée, retry après {delay_base * (2 ** attempt)}s"
                    logger.warning(warning_msg)
                    self.alert_manager.send_alert(warning_msg, priority=3)
                time.sleep(delay_base * (2**attempt))
        return None

    def _persist_log(self, log_entry: Dict):
        """Sauvegarde un log dans risk_manager_performance.csv."""
        start_time = time.time()
        try:
            log_df = pd.DataFrame([log_entry])
            os.makedirs(os.path.dirname(LOG_DIR / "risk_manager_performance.csv"), exist_ok=True)

            def write_log():
                log_path = LOG_DIR / "risk_manager_performance.csv"
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

            self.with_retries(write_log)
            latency = time.time() - start_time
            logger.info(f"Log sauvegardé dans {LOG_DIR / 'risk_manager_performance.csv'}")
            return latency
        except Exception as e:
            error_msg = f"Erreur sauvegarde log: {str(e)}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            return 0

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
            self.log_buffer.append(log_entry)
            if len(self.log_buffer) >= self.config["buffer_size"]:
                for entry in self.log_buffer:
                    self._persist_log(entry)
                self.log_buffer = []
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

    def save_snapshot(self, snapshot_type: str, data: Dict, compress: bool = False) -> None:
        """Sauvegarde un instantané JSON des résultats, avec option de compression gzip."""
        start_time = datetime.now()
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = {"timestamp": timestamp, "type": snapshot_type, "data": data}
            snapshot_path = SNAPSHOT_DIR / f"snapshot_{snapshot_type}_{timestamp}.json"
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)

            def write_snapshot():
                import gzip
                if compress:
                    with gzip.open(f"{snapshot_path}.gz", "wt", encoding="utf-8") as f:
                        json.dump(snapshot, f, indent=4)
                else:
                    with open(snapshot_path, "w", encoding="utf-8") as f:
                        json.dump(snapshot, f, indent=4)

            self.with_retries(write_snapshot)
            save_path = f"{snapshot_path}.gz" if compress else snapshot_path
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("save_snapshot", latency, success=True)
            logger.info(f"Snapshot {snapshot_type} sauvegardé: {save_path}")
            self.alert_manager.send_alert(
                f"Snapshot {snapshot_type} sauvegardé: {save_path}", priority=1
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot: {str(e)}"
            logger.error(error_msg)
            capture_error(
                e,
                context={"market": self.market},
                market=self.market,
                operation="save_snapshot",
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            self.log_performance("save_snapshot", time.time() - start_time.total_seconds(), success=False, error=str(e))

    def save_s3_checkpoint(self, metrics: Dict) -> None:
        """Sauvegarde les métriques de position sizing sur S3."""
        start_time = datetime.now()
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_key = f"{self.s3_config['s3_prefix']}/risk_manager_metrics_{timestamp}.json.gz"
            checkpoint_data = {
                "timestamp": timestamp,
                "market": self.market,
                "metrics": metrics,
                "capital": self.capital,
            }

            def upload_to_s3():
                import gzip
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
            latency = (datetime.now() - start_time).total_seconds()
            logger.info(f"Checkpoint S3 sauvegardé: {s3_key}")
            self.alert_manager.send_alert(
                f"Checkpoint S3 sauvegardé: {s3_key}", priority=1
            )
            self.log_performance("save_s3_checkpoint", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur sauvegarde checkpoint S3: {str(e)}"
            logger.error(error_msg)
            capture_error(
                e,
                context={"market": self.market},
                market=self.market,
                operation="save_s3_checkpoint",
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            self.log_performance("save_s3_checkpoint", time.time() - start_time.total_seconds(), success=False, error=str(e))

    def check_trade_frequency(self, current_time: datetime) -> float:
        """
        Vérifie la fréquence des trades et retourne un facteur d'ajustement.

        Args:
            current_time (datetime): Horodatage actuel.

        Returns:
            float: Facteur d'ajustement (0.0 à 1.0, 0.0 si surtrading).
        """
        start_time = datetime.now()
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
                return 0.0
            # Vérifier l'intervalle minimum
            if self.trade_timestamps and (current_time - max(self.trade_timestamps)).total_seconds() < self.config["entry_freq_min_interval"]:
                logger.warning("Intervalle entre trades trop court")
                self.alert_manager.send_alert(
                    "Intervalle entre trades trop court", priority=3
                )
                return 0.0
            # Ajouter le timestamp actuel
            self.trade_timestamps.append(current_time)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "check_trade_frequency",
                latency,
                success=True,
                recent_trades=recent_trades,
            )
            return 1.0
        except Exception as e:
            error_msg = f"Erreur check_trade_frequency: {str(e)}"
            logger.error(error_msg)
            capture_error(
                e,
                context={"market": self.market},
                market=self.market,
                operation="check_trade_frequency",
            )
            self.alert_manager.send_alert(error_msg, priority=3)
            self.log_performance(
                "check_trade_frequency",
                time.time() - start_time.total_seconds(),
                success=False,
                error=str(e),
            )
            return 0.0

    def calculate_position_size(
        self,
        atr_dynamic: float,
        orderflow_imbalance: float,
        volatility_score: float = 0.0,
        iv_skew: float = 0.0,
        bid_ask_imbalance: float = 0.0,
        trade_aggressiveness: float = 0.0,
    ) -> float:
        """
        Calcule la taille de la position avec Kelly/ATR et cache.

        Args:
            atr_dynamic (float): ATR calculé sur 1-5 min.
            orderflow_imbalance (float): Imbalance du carnet (bid_volume - ask_volume) / total_volume.
            volatility_score (float): Score de volatilité historique (0 à 1).
            iv_skew (float): Implied volatility skew (0 à 0.5 typiquement).
            bid_ask_imbalance (float): Imbalance bid-ask (-1 à 1).
            trade_aggressiveness (float): Agressivité des trades (0 à 1).

        Returns:
            float: Taille de la position en pourcentage du capital.
        """
        start_time = datetime.now()
        try:
            # Vérification des entrées
            for param, name in [
                (atr_dynamic, "atr_dynamic"),
                (orderflow_imbalance, "orderflow_imbalance"),
                (volatility_score, "volatility_score"),
                (iv_skew, "iv_skew"),
                (bid_ask_imbalance, "bid_ask_imbalance"),
                (trade_aggressiveness, "trade_aggressiveness"),
            ]:
                if pd.isna(param) or np.isinf(param):
                    raise ValueError(f"{name} contient NaN ou inf")
            if atr_dynamic <= 0:
                raise ValueError("ATR dynamique doit être positif")
            if not -1 <= orderflow_imbalance <= 1:
                raise ValueError("Orderflow imbalance doit être entre -1 et 1")
            if not 0 <= volatility_score <= 1:
                raise ValueError("Volatility score doit être entre 0 et 1")
            if not 0 <= iv_skew <= 0.5:
                raise ValueError("IV skew doit être entre 0 et 0.5")
            if not -1 <= bid_ask_imbalance <= 1:
                raise ValueError("Bid-ask imbalance doit être entre -1 et 1")
            if not 0 <= trade_aggressiveness <= 1:
                raise ValueError("Trade aggressiveness doit être entre 0 et 1")

            # Vérifier la fréquence des trades
            freq_factor = self.check_trade_frequency(datetime.now())
            if freq_factor == 0.0:
                return 0.0

            # Vérifier le cache
            cache_key = hash((atr_dynamic, orderflow_imbalance, volatility_score, iv_skew, bid_ask_imbalance, trade_aggressiveness))
            if cache_key in self.size_cache:
                size = self.size_cache[cache_key]
                self.size_cache.move_to_end(cache_key)
                logger.debug(f"Cache hit pour position size: {size}")
            else:
                # Ajustement dynamique de Kelly en fonction de la volatilité et des nouvelles features
                adjusted_kelly = self.config["kelly_fraction"] * (1 - volatility_score)
                # Formule hybride Kelly/ATR pondérée par les imbalances et l'agressivité
                size = adjusted_kelly * (1 / atr_dynamic) * (
                    1 + orderflow_imbalance + 0.5 * iv_skew + 0.3 * bid_ask_imbalance - 0.2 * trade_aggressiveness
                )
                size = min(
                    size, self.config["max_position_fraction"]
                )  # Limite configurable
                size = max(size, 0.0)  # Évite les tailles négatives
                size *= freq_factor  # Appliquer le facteur de fréquence
                # Mettre à jour le cache
                self.size_cache[cache_key] = size
                if len(self.size_cache) > self.cache_max_size:
                    self.size_cache.popitem(last=False)

            # Mettre à jour la fenêtre glissante pour la variance
            self.recent_sizes.append(size)
            if len(self.recent_sizes) > self.variance_window:
                self.recent_sizes.pop(0)
            variance = np.var(self.recent_sizes) if self.recent_sizes else 0.0

            # Enregistrer les métriques Prometheus
            position_size_metric.labels(market=self.market).set(size * 100)
            position_size_variance.labels(market=self.market).set(variance)

            # Journaliser la performance
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "calculate_position_size",
                latency,
                success=True,
                atr_dynamic=atr_dynamic,
                orderflow_imbalance=orderflow_imbalance,
                volatility_score=volatility_score,
                iv_skew=iv_skew,
                bid_ask_imbalance=bid_ask_imbalance,
                trade_aggressiveness=trade_aggressiveness,
                position_size=size,
                variance=variance,
            )

            # Alerte si la taille est proche de la limite
            if size >= 0.9 * self.config["max_position_fraction"]:
                alert_msg = f"Taille de position élevée pour {self.market}: {size*100:.2f}% du capital"
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=3)

            # Sauvegarde S3 et snapshot
            metrics = {
                "position_size": size,
                "variance": variance,
                "atr_dynamic": atr_dynamic,
                "orderflow_imbalance": orderflow_imbalance,
                "volatility_score": volatility_score,
                "iv_skew": iv_skew,
                "bid_ask_imbalance": bid_ask_imbalance,
                "trade_aggressiveness": trade_aggressiveness,
            }
            self.save_snapshot("calculate_position_size", metrics, compress=False)
            self.save_s3_checkpoint(metrics)

            return size
        except Exception as e:
            error_msg = f"Erreur calcul position size pour {self.market}: {str(e)}"
            logger.error(error_msg)
            capture_error(
                e,
                context={"market": self.market},
                market=self.market,
                operation="calculate_position_size",
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "calculate_position_size", latency, success=False, error=str(e)
            )
            return 0.0
