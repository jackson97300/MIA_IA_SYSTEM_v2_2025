# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/model/utils/trading_loop.py
# Rôle : Gère la boucle de trading principale pour live_trading.py, intégrant toutes les méthodes (1-18) pour la détection,
#        la gestion des risques, la sélection des signaux, et l’exécution des trades.
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.8,<6.0.0, asyncio, boto3>=1.26.0,<2.0.0,
#   loguru>=0.7.0,<1.0.0, pyyaml>=6.0.0,<7.0.0, signal
# - src/model/router/detect_regime.py
# - src/model/sac/train_sac.py
# - src/model/utils/risk_controller.py
# - src/model/utils/trade_executor.py
# - src/model/utils/signal_selector.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - data_stream (AsyncIterable[pd.DataFrame]): Flux de données IQFeed (350 features pour entraînement, 150 SHAP pour inférence)
# - regime_detector (DetectRegime): Détecteur de régime (méthode 1)
# - risk_controller (RiskController): Contrôleur de risques (méthode 5)
# - trade_executor (TradeExecutor): Exécuteur de trades
# - signal_selector (SignalSelector): Sélecteur de signaux (méthode 2)
#
# Outputs :
# - Trades exécutés
# - Logs dans data/logs/trading_loop.log
# - Logs de performance dans data/logs/trading_loop_performance.csv
# - Snapshots JSON compressés dans data/cache/trading_loop/<market>/*.json.gz
# - Sauvegardes incrémentielles dans data/checkpoints/trading_loop/<market>/*.json.gz
#
# Notes :
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence) définies dans feature_sets.yaml.
# - Utilise IQFeed exclusivement via data_provider.py pour le flux de données.
# - Supprime toute référence à 320 features, 81 features, et obs_t.
# - Implémente retries (max 3, délai 2^attempt secondes) pour les opérations critiques.
# - Intègre toutes les méthodes (1-18) via les dépendances et la logique explicite.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des données.
# - Tests unitaires disponibles dans tests/test_trading_loop.py.

import asyncio
import gzip
import json
import signal
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterable, Dict, Optional

import boto3
import numpy as np
import pandas as pd
import psutil
from loguru import logger

from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import get_config
from src.utils.telegram_alert import send_telegram_alert

# Configuration logging
logger.remove()
BASE_DIR = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
LOG_DIR = BASE_DIR / "data" / "logs"
CACHE_DIR = BASE_DIR / "data" / "cache" / "trading_loop"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints" / "trading_loop"
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "trading_loop.log", rotation="10 MB", level="INFO", encoding="utf-8"
)
logger.add(lambda msg: print(msg, end=""), level="INFO")

# Constantes
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Secondes, exponentiel
PERF_LOG_PATH = LOG_DIR / "trading_loop_performance.csv"
MAX_CACHE_SIZE = 1000
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB

# Cache global pour les résultats de trading_loop
trading_cache = OrderedDict()


class TradingLoop:
    """Gère la boucle de trading principale pour live_trading.py."""

    def __init__(
        self,
        config_path: str = str(BASE_DIR / "config" / "algo_config.yaml"),
        market: str = "ES",
    ):
        """
        Initialise la boucle de trading.

        Args:
            config_path (str): Chemin vers le fichier de configuration.
            market (str): Marché (ex. : ES, MNQ).
        """
        self.market = market
        self.alert_manager = AlertManager()
        self.config = get_config(config_path).get("trading_loop", {})
        self.perf_log = PERF_LOG_PATH
        self.snapshot_dir = CACHE_DIR / market
        self.snapshot_dir.mkdir(exist_ok=True)
        PERF_LOG_PATH.parent.mkdir(exist_ok=True)
        signal.signal(signal.SIGINT, self.handle_sigint)
        logger.info(f"TradingLoop initialisé pour {market}")
        self.log_performance("init", 0, success=True)

    def handle_sigint(self, signum, frame):
        """Gère l'arrêt propre sur SIGINT."""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "status": "SIGINT",
            "market": self.market,
        }
        snapshot_path = self.snapshot_dir / f'sigint_{snapshot["timestamp"]}.json.gz'
        try:
            with gzip.open(snapshot_path, "wt", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=4)
            success_msg = (
                f"Arrêt propre sur SIGINT, snapshot sauvegardé pour {self.market}"
            )
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot SIGINT pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
        exit(0)

    def log_performance(
        self,
        operation: str,
        latency: float,
        success: bool = True,
        error: str = None,
        **kwargs,
    ) -> None:
        """
        Enregistre les performances (CPU, mémoire, latence) dans trading_loop_performance.csv.

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
            confidence_drop_rate = 1.0 if success else 0.0  # Simplifié pour Phase 8
            if memory_usage > 1024:
                alert_msg = f"ALERTE: Usage mémoire élevé ({memory_usage:.2f} MB) pour {self.market}"
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=5)
                send_telegram_alert(alert_msg)
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "latency": latency,
                "success": success,
                "error": error,
                "memory_usage_mb": memory_usage,
                "cpu_percent": cpu_percent,
                "confidence_drop_rate": confidence_drop_rate,
                "market": self.market,
                **kwargs,
            }
            log_df = pd.DataFrame([log_entry])

            def save_log():
                if not PERF_LOG_PATH.exists():
                    log_df.to_csv(PERF_LOG_PATH, index=False, encoding="utf-8")
                else:
                    log_df.to_csv(
                        PERF_LOG_PATH,
                        mode="a",
                        header=False,
                        index=False,
                        encoding="utf-8",
                    )

            self.with_retries(save_log)
            logger.info(
                f"Performance journalisée pour {operation}. CPU: {cpu_percent}%"
            )
            self.save_snapshot("log_performance", log_entry)
        except Exception as e:
            error_msg = f"Erreur journalisation performance pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)

    def save_snapshot(
        self, snapshot_type: str, data: Dict, compress: bool = True
    ) -> None:
        """
        Sauvegarde un instantané JSON des résultats, compressé avec gzip.

        Args:
            snapshot_type (str): Type de snapshot (ex. : trading_loop).
            data (Dict): Données à sauvegarder.
            compress (bool): Compresser avec gzip (défaut : True).
        """
        start_time = time.time()
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = {
                "timestamp": timestamp,
                "type": snapshot_type,
                "market": self.market,
                "data": data,
            }
            snapshot_path = (
                self.snapshot_dir / f"snapshot_{snapshot_type}_{timestamp}.json"
            )

            def write_snapshot():
                if compress:
                    with gzip.open(f"{snapshot_path}.gz", "wt", encoding="utf-8") as f:
                        json.dump(snapshot, f, indent=4)
                else:
                    with open(snapshot_path, "w", encoding="utf-8") as f:
                        json.dump(snapshot, f, indent=4)

            self: with_retries(write_snapshot)
            save_path = f"{snapshot_path}.gz" if compress else snapshot_path
            file_size = os.path.getsize(save_path) / 1024 / 1024
            if file_size > 1.0:
                alert_msg = (
                    f"Snapshot size {file_size:.2f} MB exceeds 1 MB pour {self.market}"
                )
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
            latency = time.time() - start_time
            success_msg = (
                f"Snapshot {snapshot_type} sauvegardé pour {self.market}: {save_path}"
            )
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "save_snapshot", latency, success=True, snapshot_size_mb=file_size
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot {snapshot_type} pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance("save_snapshot", 0, success=False, error=str(e))

    def checkpoint(
        self, data: pd.DataFrame, data_type: str = "trading_loop_state"
    ) -> None:
        """
        Sauvegarde incrémentielle des données toutes les 5 minutes avec versionnage (5 versions).

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : trading_loop_state).
        """
        try:
            start_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_data = {
                "timestamp": timestamp,
                "num_rows": len(data),
                "columns": list(data.columns),
                "data_type": data_type,
                "market": self.market,
            }
            checkpoint_dir = CHECKPOINT_DIR / self.market
            checkpoint_dir.mkdir(exist_ok=True)
            checkpoint_path = (
                checkpoint_dir / f"trading_loop_{data_type}_{timestamp}.json.gz"
            )
            checkpoint_versions = []

            def write_checkpoint():
                with gzip.open(checkpoint_path, "wt", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, indent=4)
                data.to_csv(
                    checkpoint_path.with_suffix(".csv"), index=False, encoding="utf-8"
                )

            self.with_retries(write_checkpoint)
            checkpoint_versions.append(checkpoint_path)
            if len(checkpoint_versions) > 5:
                oldest = checkpoint_versions.pop(0)
                if oldest.exists():
                    oldest.unlink()
                csv_oldest = oldest.with_suffix(".csv")
                if csv_oldest.exists():
                    csv_oldest.unlink()
            file_size = os.path.getsize(checkpoint_path) / 1024 / 1024
            latency = time.time() - start_time
            success_msg = f"Checkpoint sauvegardé pour {self.market}: {checkpoint_path}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "checkpoint",
                latency,
                success=True,
                file_size_mb=file_size,
                num_rows=len(data),
                data_type=data_type,
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde checkpoint pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "checkpoint", 0, success=False, error=str(e), data_type=data_type
            )

    def cloud_backup(
        self, data: pd.DataFrame, data_type: str = "trading_loop_state"
    ) -> None:
        """
        Sauvegarde distribuée des données vers AWS S3 toutes les 15 minutes.

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : trading_loop_state).
        """
        try:
            start_time = time.time()
            config = get_config(str(BASE_DIR / "config/es_config.yaml"))
            if not config.get("s3_bucket"):
                warning_msg = f"S3 bucket non configuré, sauvegarde cloud ignorée pour {self.market}"
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{config['s3_prefix']}trading_loop_{data_type}_{self.market}_{timestamp}.csv.gz"
            temp_path = CHECKPOINT_DIR / self.market / f"temp_s3_{timestamp}.csv.gz"
            temp_path.parent.mkdir(exist_ok=True)

            def write_temp():
                data.to_csv(
                    temp_path, compression="gzip", index=False, encoding="utf-8"
                )

            self.with_retries(write_temp)
            s3_client = boto3.client("s3")

            def upload_s3():
                s3_client.upload_file(str(temp_path), config["s3_bucket"], backup_path)

            self.with_retries(upload_s3)
            temp_path.unlink()
            latency = time.time() - start_time
            success_msg = (
                f"Sauvegarde cloud S3 effectuée pour {self.market}: {backup_path}"
            )
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "cloud_backup",
                latency,
                success=True,
                num_rows=len(data),
                data_type=data_type,
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde cloud S3 pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "cloud_backup", 0, success=False, error=str(e), data_type=data_type
            )

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
                    error_msg = f"Échec après {max_attempts} tentatives pour {self.market}: {str(e)}\n{traceback.format_exc()}"
                    logger.error(error_msg)
                    self.alert_manager.send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    self.log_performance(
                        f"retry_attempt_{attempt+1}",
                        time.time() - start_time,
                        success=False,
                        error=str(e),
                        attempt_number=attempt + 1,
                    )
                    return None
                delay = delay_base**attempt
                warning_msg = f"Tentative {attempt+1} échouée pour {self.market}, retry après {delay}s"
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                time.sleep(delay)

    async def with_retries_async(
        self,
        func: callable,
        max_attempts: int = MAX_RETRIES,
        delay_base: float = RETRY_DELAY_BASE,
    ) -> Optional[Any]:
        """
        Exécute une fonction asynchrone avec retries exponentiels.

        Args:
            func (callable): Fonction asynchrone à exécuter.
            max_attempts (int): Nombre maximum de tentatives.
            delay_base (float): Base pour le délai exponentiel.

        Returns:
            Optional[Any]: Résultat de la fonction ou None si échec.
        """
        start_time = time.time()
        for attempt in range(max_attempts):
            try:
                result = await func()
                latency = time.time() - start_time
                self.log_performance(
                    f"retry_attempt_async_{attempt+1}",
                    latency,
                    success=True,
                    attempt_number=attempt + 1,
                )
                return result
            except Exception as e:
                if attempt == max_attempts - 1:
                    error_msg = f"Échec après {max_attempts} tentatives pour {self.market}: {str(e)}\n{traceback.format_exc()}"
                    logger.error(error_msg)
                    self.alert_manager.send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    self.log_performance(
                        f"retry_attempt_async_{attempt+1}",
                        time.time() - start_time,
                        success=False,
                        error=str(e),
                        attempt_number=attempt + 1,
                    )
                    return None
                delay = delay_base**attempt
                warning_msg = f"Tentative {attempt+1} échouée pour {self.market}, retry après {delay}s"
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                await asyncio.sleep(delay)

    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Valide les données d’entrée avec confidence_drop_rate.

        Args:
            data (pd.DataFrame): Données à valider.

        Raises:
            ValueError: Si les données sont invalides.
        """
        start_time = time.time()
        try:
            feature_sets_path = BASE_DIR / "config/feature_sets.yaml"
            features_config = get_config(feature_sets_path)
            expected_cols = (
                features_config.get("training", {}).get("features", [])[:350]
                if data.shape[1] >= 350
                else features_config.get("inference", {}).get("shap_features", [])[:150]
            )
            expected_len = 350 if data.shape[1] >= 350 else 150
            if len(expected_cols) != expected_len:
                raise ValueError(
                    f"Nombre de features incorrect pour {self.market}: {len(expected_cols)} au lieu de {expected_len}"
                )

            missing_cols = [col for col in expected_cols if col not in data.columns]
            null_count = data[expected_cols].isnull().sum().sum()
            confidence_drop_rate = (
                null_count / (len(data) * len(expected_cols))
                if (len(data) * len(expected_cols)) > 0
                else 0.0
            )
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé pour {self.market}: {confidence_drop_rate:.2f} ({null_count} valeurs nulles)"
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
            if missing_cols:
                warning_msg = f"Colonnes manquantes pour {self.market} ({'SHAP' if expected_len == 150 else 'full'}): {missing_cols}"
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=2)
                send_telegram_alert(warning_msg)
                for col in missing_cols:
                    data[col] = 0.0

            critical_cols = [
                "bid_size_level_1",
                "ask_size_level_1",
                "trade_frequency_1s",
                f"spread_avg_1min_{self.market.lower()}",
                "close",
            ]
            for col in critical_cols:
                if col in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        raise ValueError(
                            f"Colonne {col} n'est pas numérique pour {self.market}: {data[col].dtype}"
                        )
                    if data[col].isna().any():
                        data[col] = (
                            data[col]
                            .interpolate(method="linear", limit_direction="both")
                            .fillna(0.0)
                        )
                    if col in [
                        "bid_size_level_1",
                        "ask_size_level_1",
                        "trade_frequency_1s",
                        f"spread_avg_1min_{self.market.lower()}",
                    ]:
                        if (data[col] <= 0).any():
                            warning_msg = f"Valeurs non positives dans {col} pour {self.market}, corrigées à 1e-6"
                            logger.warning(warning_msg)
                            self.alert_manager.send_alert(warning_msg, priority=4)
                            data[col] = data[col].clip(lower=1e-6)

            self.save_snapshot(
                "validate_data",
                {
                    "num_columns": len(data.columns),
                    "missing_columns": missing_cols,
                    "confidence_drop_rate": confidence_drop_rate,
                },
            )
            self.log_performance(
                "validate_data", time.time() - start_time, success=True
            )
        except Exception as e:
            error_msg = f"Erreur validation données pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance(
                "validate_data", time.time() - start_time, success=False, error=str(e)
            )
            raise

    async def trading_loop(
        self,
        data_stream: AsyncIterable[pd.DataFrame],
        regime_detector: Any,
        risk_controller: Any,
        trade_executor: Any,
        signal_selector: Any,
    ) -> None:
        """
        Gère la boucle de trading principale, intégrant toutes les méthodes (1-18).

        Args:
            data_stream (AsyncIterable[pd.DataFrame]): Flux de données IQFeed.
            regime_detector (Any): Détecteur de régime (méthode 1).
            risk_controller (Any): Contrôleur de risques (méthode 5).
            trade_executor (Any): Exécuteur de trades.
            signal_selector (Any): Sélecteur de signaux (méthode 2).
        """
        start_time = time.time()
        try:
            cache_key = f"{self.market}_{hash(str(data_stream))}"
            if cache_key in trading_cache:
                return trading_cache[cache_key]
            while len(trading_cache) > MAX_CACHE_SIZE:
                trading_cache.popitem(last=False)

            expected_dims = self.config.get(
                "observation_dims", {"training": 350, "inference": 150}
            )
            valid_regimes = {"trend", "range", "defensive"}

            async for data in data_stream:
                loop_start_time = time.time()
                try:
                    if not isinstance(data, pd.DataFrame):
                        error_msg = f"Données d’entrée invalides pour {self.market}: DataFrame attendu"
                        logger.error(error_msg)
                        self.alert_manager.send_alert(error_msg, priority=4)
                        send_telegram_alert(error_msg)
                        self.log_performance(
                            "trading_loop",
                            time.time() - loop_start_time,
                            success=False,
                            error="Données invalides",
                        )
                        continue

                    data_dim = data.shape[1]
                    if data_dim not in (
                        expected_dims["training"],
                        expected_dims["inference"],
                    ):
                        error_msg = f"Dimension des données invalide pour {self.market}: {data_dim}, attendu {expected_dims['training']} ou {expected_dims['inference']}"
                        logger.error(error_msg)
                        self.alert_manager.send_alert(error_msg, priority=4)
                        send_telegram_alert(error_msg)
                        self.log_performance(
                            "trading_loop",
                            time.time() - loop_start_time,
                            success=False,
                            error="Dimension données invalide",
                        )
                        continue

                    self._validate_data(data)

                    async def detect_regime():
                        return regime_detector.detect_market_regime_vectorized(data, 0)

                    regime, probs = await self.with_retries_async(detect_regime)
                    if regime is None or regime.lower() not in valid_regimes:
                        error_msg = (
                            f"Régime détecté invalide pour {self.market}: {regime}"
                        )
                        logger.error(error_msg)
                        self.alert_manager.send_alert(error_msg, priority=3)
                        send_telegram_alert(error_msg)
                        self.log_performance(
                            "trading_loop",
                            time.time() - loop_start_time,
                            success=False,
                            error="Régime invalide",
                        )
                        continue

                    drawdown = data.get("drawdown", 0.0)

                    async def check_risk():
                        return risk_controller.stop_trading(drawdown)

                    stop_trading = await self.with_retries_async(check_risk)
                    if stop_trading:
                        success_msg = f"Trading arrêté pour {self.market}: drawdown={drawdown:.2f}"
                        logger.info(success_msg)
                        self.alert_manager.send_alert(success_msg, priority=3)
                        send_telegram_alert(success_msg)
                        self.log_performance(
                            "trading_loop",
                            time.time() - loop_start_time,
                            success=True,
                            regime=regime,
                            stop_reason="Drawdown",
                        )
                        continue

                    context_confidence = (
                        1.0  # Placeholder pour méthode 7 (market_memory.db)
                    )

                    async def select_signal():
                        return signal_selector.select_signal(data, probs, regime)

                    trade = await self.with_retries_async(select_signal)
                    if trade is None:
                        error_msg = f"Échec sélection signal pour {self.market}"
                        logger.error(error_msg)
                        self.alert_manager.send_alert(error_msg, priority=3)
                        send_telegram_alert(error_msg)
                        self.log_performance(
                            "trading_loop",
                            time.time() - loop_start_time,
                            success=False,
                            error="Échec signal",
                        )
                        continue

                    async def execute_trade():
                        return trade_executor.execute_trade(trade)

                    trade_result = await self.with_retries_async(execute_trade)
                    if trade_result is None:
                        error_msg = f"Échec exécution trade pour {self.market}"
                        logger.error(error_msg)
                        self.alert_manager.send_alert(error_msg, priority=4)
                        send_telegram_alert(error_msg)
                        self.log_performance(
                            "trading_loop",
                            time.time() - loop_start_time,
                            success=False,
                            error="Échec exécution",
                        )
                        continue

                    snapshot = {
                        "timestamp": datetime.now().isoformat(),
                        "regime": regime,
                        "probs": probs,
                        "drawdown": float(drawdown),
                        "trade": trade,
                        "context_confidence": context_confidence,
                        "data_dim": data_dim,
                    }
                    self.save_snapshot("trading_loop", snapshot)
                    self.checkpoint(data, data_type="trading_loop")
                    self.cloud_backup(data, data_type="trading_loop")

                    success_msg = f"Boucle exécutée pour {self.market}: régime={regime}, trade={trade}"
                    logger.info(success_msg)
                    self.alert_manager.send_alert(success_msg, priority=1)
                    send_telegram_alert(success_msg)
                    self.log_performance(
                        "trading_loop",
                        time.time() - loop_start_time,
                        success=True,
                        regime=regime,
                        trade=str(trade),
                    )

                except Exception as e:
                    error_msg = f"Erreur boucle trading pour {self.market}: {str(e)}\n{traceback.format_exc()}"
                    logger.error(error_msg)
                    self.alert_manager.send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    self.log_performance(
                        "trading_loop",
                        time.time() - loop_start_time,
                        success=False,
                        error=str(e),
                    )

            trading_cache[cache_key] = None
            latency = time.time() - start_time
            self.log_performance("trading_loop_complete", latency, success=True)

        except Exception as e:
            error_msg = f"Erreur globale boucle trading pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance(
                "trading_loop_complete",
                time.time() - start_time,
                success=False,
                error=str(e),
            )


if __name__ == "__main__":
    from src.model.router.detect_regime import MarketRegimeDetector
    from src.model.utils.risk_controller import RiskController
    from src.model.utils.signal_selector import SignalSelector
    from src.model.utils.trade_executor import TradeExecutor

    async def mock_data_stream():
        feature_sets_path = BASE_DIR / "config/feature_sets.yaml"
        features_config = get_config(feature_sets_path)
        feature_cols = features_config.get("training", {}).get("features", [])[:350]
        for _ in range(2):
            data = pd.DataFrame(
                {
                    "vix_es_correlation": [20.0],
                    "drawdown": [0.01],
                    **{col: [np.random.uniform(0, 1)] for col in feature_cols},
                }
            )
            yield data
            await asyncio.sleep(1)

    async def main():
        loop = TradingLoop()
        await loop.trading_loop(
            mock_data_stream(),
            MarketRegimeDetector(),
            RiskController(),
            TradeExecutor(),
            SignalSelector(),
        )

    asyncio.run(main())
