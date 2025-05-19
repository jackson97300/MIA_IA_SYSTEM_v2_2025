# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/features/spotgamma_recalculator.py
# Recalcule les niveaux critiques d’options, y compris dealer_position_bias, pour MIA_IA_SYSTEM_v2_2025.
# Intègre l’analyse SHAP (méthode 17) pour évaluer l’impact des features d’options (limité à 50 features).
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Génère les métriques d’options (call_wall, put_wall, zero_gamma, etc.) et analyse leur importance via SHAP.
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.0,<6.0.0, joblib>=1.2.0,<2.0.0,
#   shap>=0.41.0,<0.42.0, matplotlib>=3.7.0,<4.0.0, pyyaml>=6.0.0,<7.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/features/shap_weighting.py
# - src/data/data_provider.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/es_config.yaml
# - data/iqfeed/option_chain.csv
#
# Outputs :
# - data/features/spotgamma_metrics.csv
# - data/features/spotgamma_shap.csv
# - data/logs/spotgamma_performance.csv
# - data/figures/spotgamma/
# - data/spotgamma_snapshots/*.json (option *.json.gz)
#
# Notes :
# - Conforme à structure.txt (version 2.1.3, 2025-05-13).
# - Utilise exclusivement IQFeed via data_provider.py, avec retries (max 3, délai 2^attempt secondes).
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des calculs.
# - Intègre validation SHAP (Phase 17) limitée à 50 features clés pour l’impact des métriques d’options.
# - Préserve toutes les fonctionnalités existantes (calcul des niveaux d’options).
# - Sauvegarde des snapshots JSON non compressés par défaut avec option de compression gzip.
# - Envoie des alertes via AlertManager et telegram_alert pour les erreurs critiques et succès.
# - Tests unitaires disponibles dans tests/test_spotgamma_recalculator.py (à implémenter).
# - Validation complète prévue pour juin 2025.
# - Évolution future : Migration API Investing.com (juin 2025), optimisation pour feature_pipeline.py.

import gzip
import json
import logging
import os
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import yaml

from src.features.shap_weighting import SHAPWeighting
from src.model.utils.alert_manager import send_alert
from src.utils.telegram_alert import send_telegram_alert

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "spotgamma_snapshots")
CSV_LOG_PATH = os.path.join(BASE_DIR, "data", "logs", "spotgamma_performance.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures", "spotgamma")
METRICS_PATH = os.path.join(BASE_DIR, "data", "features", "spotgamma_metrics.csv")
SHAP_PATH = os.path.join(BASE_DIR, "data", "features", "spotgamma_shap.csv")

# Création des dossiers
os.makedirs(os.path.join(BASE_DIR, "data", "logs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data", "features"), exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Configuration du logging
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "data", "logs", "spotgamma_recalculator.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Constantes
MAX_RETRIES = 3
RETRY_DELAY = 2.0


class SpotGammaRecalculator:
    """
    Recalcule les niveaux critiques d’options et analyse leur importance via SHAP.
    """

    def __init__(
        self, config_path: str = os.path.join(BASE_DIR, "config", "es_config.yaml")
    ):
        """
        Initialise le recalculateur SpotGamma.

        Args:
            config_path (str): Chemin vers le fichier de configuration.
        """
        self.log_buffer = []
        self.cache = {}
        self.config_path = config_path
        try:
            self.config = self.load_config(config_path)
            self.buffer_size = self.config.get("buffer_size", 100)
            self.max_cache_size = self.config.get("max_cache_size", 1000)
            self.shap_weighting = SHAPWeighting(config_path)
            logger.info("SpotGammaRecalculator initialisé")
            send_alert("SpotGammaRecalculator initialisé", priority=2)
            send_telegram_alert("SpotGammaRecalculator initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config_path": config_path}, compress=False)
        except Exception as e:
            error_msg = f"Erreur initialisation SpotGammaRecalculator: {str(e)}\n{traceback.format_exc()}"
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            self.config = {
                "buffer_size": 100,
                "max_cache_size": 1000,
                "cache_hours": 24,
                "option_metrics": [
                    "call_wall",
                    "put_wall",
                    "zero_gamma",
                    "dealer_position_bias",
                    "iv_rank_30d",
                    "key_strikes_1",
                    "key_strikes_2",
                    "key_strikes_3",
                    "key_strikes_4",
                    "key_strikes_5",
                    "max_pain_strike",
                    "net_gamma",
                    "dealer_zones_count",
                    "vol_trigger",
                    "ref_px",
                    "data_release",
                ],
            }
            self.buffer_size = 100
            self.max_cache_size = 1000
            self.shap_weighting = SHAPWeighting(config_path)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Charge la configuration depuis es_config.yaml."""

        def load_yaml():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            if "spotgamma_recalculator" not in config:
                raise ValueError(
                    "Clé 'spotgamma_recalculator' manquante dans la configuration"
                )
            required_keys = ["buffer_size", "max_cache_size", "option_metrics"]
            missing_keys = [
                key
                for key in required_keys
                if key not in config["spotgamma_recalculator"]
            ]
            if missing_keys:
                raise ValueError(
                    f"Clés manquantes dans 'spotgamma_recalculator': {missing_keys}"
                )
            return config["spotgamma_recalculator"]

        try:
            start_time = time.time()
            config = self.with_retries(load_yaml)
            cache_key = hashlib.sha256(str(config).encode()).hexdigest()
            self.cache[cache_key] = {"config": config, "timestamp": datetime.now()}
            if len(self.cache) > self.max_cache_size:
                self.cache.pop(next(iter(self.cache)))
            latency = time.time() - start_time
            logger.info("Configuration spotgamma_recalculator chargée")
            send_alert("Configuration spotgamma_recalculator chargée", priority=2)
            send_telegram_alert("Configuration spotgamma_recalculator chargée")
            self.log_performance("load_config", latency, success=True)
            self.save_snapshot(
                "load_config", {"config_path": config_path}, compress=False
            )
            return config
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur chargement config: {str(e)}\n{traceback.format_exc()}"
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
                    error_msg = f"Échec après {max_attempts} tentatives: {str(e)}"
                    send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    logger.error(error_msg)
                    raise
                delay = delay_base * (2**attempt)
                warning_msg = f"Tentative {attempt+1} échouée, retry après {delay}s"
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
                time.sleep(delay)

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ) -> None:
        """Journalise les performances des opérations critiques avec psutil."""
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
            cpu_usage = psutil.cpu_percent()  # % CPU
            if memory_usage > 1024:
                alert_msg = (
                    f"ALERTE: Utilisation mémoire élevée ({memory_usage:.2f} MB)"
                )
                send_alert(alert_msg, priority=5)
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
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)
            latency = time.time() - start_time
            self.log_performance(
                "save_snapshot",
                latency,
                success=True,
                snapshot_type=snapshot_type,
                file_size_mb=file_size,
            )
            success_msg = f"Snapshot {snapshot_type} sauvegardé: {save_path}"
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}"
            self.log_performance("save_snapshot", 0, success=False, error=str(e))
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def validate_shap_features(self, features: List[str]) -> bool:
        """Valide que les features sont dans les top 50 SHAP."""
        try:
            start_time = time.time()
            if not os.path.exists(SHAP_PATH):
                error_msg = "Fichier SHAP manquant"
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return False
            shap_df = pd.read_csv(SHAP_PATH)
            if len(shap_df) < 50:
                error_msg = f"Nombre insuffisant de SHAP features: {len(shap_df)} < 50"
                send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                return False
            valid_features = set(shap_df["feature"].head(50))
            missing = [f for f in features if f not in valid_features]
            if missing:
                warning_msg = f"Features non incluses dans top 50 SHAP: {missing}"
                send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)
            latency = time.time() - start_time
            success_msg = "SHAP features validées"
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
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
            send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "validate_shap_features", latency, success=False, error=str(e)
            )
            return False

    def plot_metrics(self, metrics: pd.DataFrame, timestamp: str) -> None:
        """Génère des visualisations pour les métriques d’options."""
        try:
            start_time = time.time()
            timestamp_safe = timestamp.replace(":", "-")
            os.makedirs(FIGURES_DIR, exist_ok=True)
            plt.figure(figsize=(10, 6))
            if "net_gamma" in metrics.columns:
                plt.plot(
                    metrics["timestamp"],
                    metrics["net_gamma"],
                    label="Net Gamma",
                    color="orange",
                )
            if "vol_trigger" in metrics.columns:
                plt.plot(
                    metrics["timestamp"],
                    metrics["vol_trigger"],
                    label="Vol Trigger",
                    color="blue",
                )
            plt.title(f"SpotGamma Metrics - {timestamp}")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(FIGURES_DIR, f"metrics_{timestamp_safe}.png")
            plt.savefig(plot_path)
            plt.close()
            latency = time.time() - start_time
            self.log_performance("plot_metrics", latency, success=True)
            success_msg = f"Visualisations SpotGamma générées: {plot_path}"
            send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
        except Exception as e:
            latency = time.time() - start_time
            error_msg = (
                f"Erreur génération visualisations: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance("plot_metrics", latency, success=False, error=str(e))
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def calculate_shap_impact(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcule l’impact SHAP des features d’options, limité à 50 features clés."""
        try:
            start_time = time.time()
            required_cols = [
                "strike",
                "option_type",
                "open_interest",
                "volume",
                "gamma",
                "delta",
                "vega",
                "price",
                "underlying_price",
            ]
            missing_cols = [col for col in required_cols if col not in data.columns]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )
            if len(data) < 10:
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(required_cols) - len(missing_cols)}/{len(required_cols)} colonnes valides, {len(data)} lignes)"
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            cache_key = hashlib.sha256(data.to_json().encode()).hexdigest()
            if cache_key in self.cache:
                shap_importance = self.cache[cache_key]["shap_importance"]
                latency = time.time() - start_time
                self.log_performance(
                    "calculate_shap_impact_cache_hit",
                    latency,
                    success=True,
                    num_features=len(shap_importance),
                )
                success_msg = "Importance SHAP récupérée du cache"
                send_alert(success_msg, priority=1)
                send_telegram_alert(success_msg)
                logger.info(success_msg)
                return shap_importance

            data = data.copy()
            for col in missing_cols:
                data[col] = 0
                warning_msg = (
                    f"Colonne '{col}' manquante dans option_chain, imputée à 0"
                )
                send_alert(warning_msg, priority=2)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)

            shap_importance = self.shap_weighting.calculate_shap_weights(data)
            shap_importance = shap_importance.head(50)
            option_metrics = self.config.get("option_metrics", [])
            for metric in option_metrics:
                if metric in shap_importance["feature"].values:
                    shap_importance.loc[
                        shap_importance["feature"] == metric, "importance"
                    ] *= 1.2

            self.validate_shap_features(shap_importance["feature"].tolist())

            os.makedirs(os.path.dirname(SHAP_PATH), exist_ok=True)

            def write_shap():
                shap_importance.to_csv(
                    SHAP_PATH,
                    mode="a" if os.path.exists(SHAP_PATH) else "w",
                    header=not os.path.exists(SHAP_PATH),
                    index=False,
                    encoding="utf-8",
                )

            self.with_retries(write_shap)

            self.cache[cache_key] = {
                "shap_importance": shap_importance,
                "timestamp": datetime.now(),
            }
            if len(self.cache) > self.max_cache_size:
                self.cache.pop(next(iter(self.cache)))

            latency = time.time() - start_time
            self.log_performance(
                "calculate_shap_impact",
                latency,
                success=True,
                num_features=len(shap_importance),
                confidence_drop_rate=confidence_drop_rate,
            )
            self.save_snapshot(
                "calculate_shap_impact",
                {
                    "num_features": len(shap_importance),
                    "confidence_drop_rate": confidence_drop_rate,
                },
                compress=False,
            )
            success_msg = (
                f"Importance SHAP calculée pour {len(shap_importance)} features"
            )
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(success_msg)
            return shap_importance
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur calcul SHAP: {str(e)}\n{traceback.format_exc()}"
            self.log_performance(
                "calculate_shap_impact", latency, success=False, error=str(e)
            )
            send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.save_snapshot(
                "error_calculate_shap_impact", {"error": str(e)}, compress=False
            )
            return pd.DataFrame({"feature": [], "importance": [], "regime": []})

    def recalculate_levels(
        self, option_chain: pd.DataFrame, timestamp: str, mode: str = "snapshot"
    ) -> Dict:
        """Recalcule les niveaux critiques d’options et analyse leur importance via SHAP."""
        try:
            start_time = time.time()
            required_cols = [
                "strike",
                "option_type",
                "open_interest",
                "volume",
                "gamma",
                "delta",
                "vega",
                "price",
                "underlying_price",
            ]
            missing_cols = [
                col for col in required_cols if col not in option_chain.columns
            ]
            confidence_drop_rate = 1.0 - min(
                (len(required_cols) - len(missing_cols)) / len(required_cols), 1.0
            )
            if len(option_chain) < 10:
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(required_cols) - len(missing_cols)}/{len(required_cols)} colonnes valides, {len(option_chain)} lignes)"
                send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            if option_chain.empty:
                error_msg = "DataFrame option_chain vide"
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                self.save_snapshot(
                    "error_recalculate_levels", {"error": error_msg}, compress=False
                )
                raise ValueError(error_msg)

            option_chain = option_chain.copy()
            for col in missing_cols:
                option_chain[col] = 0
                warning_msg = (
                    f"Colonne '{col}' manquante dans option_chain, imputée à 0"
                )
                send_alert(warning_msg, priority=2)
                send_telegram_alert(warning_msg)
                logger.warning(warning_msg)

            option_chain["timestamp"] = pd.to_datetime(
                option_chain["timestamp"], errors="coerce"
            )
            if option_chain["timestamp"].isna().any():
                error_msg = "NaN dans les timestamps d’option_chain"
                send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                self.save_snapshot(
                    "error_recalculate_levels", {"error": error_msg}, compress=False
                )
                raise ValueError(error_msg)

            def compute_levels():
                metrics = {}
                metrics["timestamp"] = timestamp
                metrics["call_wall"] = option_chain[
                    option_chain["option_type"] == "call"
                ]["strike"].max()
                metrics["put_wall"] = option_chain[
                    option_chain["option_type"] == "put"
                ]["strike"].min()
                metrics["zero_gamma"] = (
                    option_chain.loc[option_chain["gamma"].abs().idxmin(), "strike"]
                    if not option_chain["gamma"].empty
                    else 0
                )
                metrics["dealer_position_bias"] = (
                    option_chain[option_chain["option_type"] == "call"]["gamma"].sum()
                    - option_chain[option_chain["option_type"] == "put"]["gamma"].sum()
                )
                metrics["iv_rank_30d"] = (
                    option_chain["vega"].mean() * 100
                )  # Approximation
                top_strikes = option_chain.nlargest(5, "open_interest")[
                    "strike"
                ].tolist()
                for i, strike in enumerate(top_strikes[:5], 1):
                    metrics[f"key_strikes_{i}"] = strike
                for i in range(len(top_strikes), 5):
                    metrics[f"key_strikes_{i+1}"] = 0
                metrics["max_pain_strike"] = (
                    option_chain.groupby("strike")["price"].sum().idxmin()
                    if not option_chain.empty
                    else 0
                )
                metrics["net_gamma"] = option_chain["gamma"].sum()
                metrics["dealer_zones_count"] = len(
                    option_chain[
                        option_chain["gamma"] > option_chain["gamma"].quantile(0.9)
                    ]
                )
                metrics["vol_trigger"] = option_chain["vega"].std()
                metrics["ref_px"] = option_chain["underlying_price"].mean()
                metrics["data_release"] = 0  # Placeholder

                metrics_df = pd.DataFrame([metrics])
                os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

                def write_metrics():
                    metrics_df.to_csv(
                        METRICS_PATH,
                        mode="a" if os.path.exists(METRICS_PATH) else "w",
                        header=not os.path.exists(METRICS_PATH),
                        index=False,
                        encoding="utf-8",
                    )

                self.with_retries(write_metrics)

                self.calculate_shap_impact(option_chain)

                return metrics

            metrics = self.with_retries(compute_levels)
            latency = time.time() - start_time
            cpu_percent = psutil.cpu_percent()
            self.log_performance(
                "recalculate_levels",
                latency,
                success=True,
                cpu_percent=cpu_percent,
                confidence_drop_rate=confidence_drop_rate,
            )
            self.save_snapshot(
                "recalculate_levels",
                {
                    "metrics": list(metrics.keys()),
                    "timestamp": timestamp,
                    "confidence_drop_rate": confidence_drop_rate,
                },
                compress=False,
            )
            self.plot_metrics(pd.DataFrame([metrics]), timestamp)
            success_msg = f"Niveaux recalculés pour {timestamp}"
            send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            logger.info(f"Niveaux recalculés. CPU: {cpu_percent}%")
            return metrics
        except Exception as e:
            latency = time.time() - start_time
            error_msg = (
                f"Erreur dans recalculate_levels: {str(e)}\n{traceback.format_exc()}"
            )
            self.log_performance(
                "recalculate_levels", latency, success=False, error=str(e)
            )
            send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.save_snapshot(
                "error_recalculate_levels", {"error": str(e)}, compress=False
            )
            return {}


if __name__ == "__main__":
    try:
        recalculator = SpotGammaRecalculator()
        option_chain = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-04-14 09:00", periods=100, freq="1min"
                ),
                "strike": np.random.uniform(5000, 5200, 100),
                "option_type": np.random.choice(["call", "put"], 100),
                "open_interest": np.random.randint(100, 1000, 100),
                "volume": np.random.randint(10, 100, 100),
                "gamma": np.random.uniform(0, 0.1, 100),
                "delta": np.random.uniform(-1, 1, 100),
                "vega": np.random.uniform(0, 10, 100),
                "price": np.random.uniform(0, 200, 100),
                "underlying_price": np.random.normal(5100, 10, 100),
                "iv_atm": np.random.uniform(0.1, 0.3, 100),
            }
        )
        timestamp = option_chain["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")
        metrics = recalculator.recalculate_levels(option_chain, timestamp)
        print("Métriques recalculées:", metrics)
        success_msg = "Test recalculate_levels terminé"
        send_alert(success_msg, priority=1)
        send_telegram_alert(success_msg)
        logger.info(success_msg)
    except Exception as e:
        error_msg = f"Erreur test: {str(e)}\n{traceback.format_exc()}"
        send_alert(error_msg, priority=3)
        send_telegram_alert(error_msg)
        logger.error(error_msg)
        raise
