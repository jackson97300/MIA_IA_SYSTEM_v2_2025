# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/scripts/run_system.py
# Script pour exécuter le pipeline global de MIA_IA_SYSTEM_v2_2025.
#
# Version : 2.1.5
# Date : 2025-05-14
#
# Rôle : Exécute le pipeline global collecte → features → trading (Phase 16),
#        appelant data_provider.py, feature_pipeline.py, et live_trading.py avec options CLI (--paper, --live).
#        Intègre la résolution des conflits de signaux via SignalResolver et utilise trade_success_prob comme critère
#        de décision dans LiveTrading. Conforme à la Phase 1 (collecte via IQFeed), Phase 8 (auto-conscience via alertes),
#        et Phase 16 (ensemble et transfer learning).
#
# Dépendances : logging, argparse, sys, pathlib, yaml, json, gzip, psutil>=5.9.8, pandas>=2.0.0,
#               src.data.data_provider, src.features.feature_pipeline, src.trading.live_trading,
#               src.model.utils.config_manager, src.model.utils.alert_manager, src.model.utils.miya_console,
#               src.utils.telegram_alert, src.utils.standard, src.model.trade_probability
#
# Inputs : config/es_config.yaml
#
# Outputs : data/logs/system/run_system.log, data/logs/run_system_performance.csv,
#           data/system_snapshots/snapshot_*.json.gz
#
# Notes :
# - Utilise IQFeed exclusivement via data_provider.py.
# - Gère 350 features pour l’entraînement et 150 SHAP features pour l’inférence via feature_pipeline.py.
# - Implémente retries (max 3, délai 2^attempt), logs psutil, alertes via alert_manager.py.
# - Intègre SignalResolver pour résoudre les conflits de signaux et journalise les métadonnées (normalized_score, conflict_coefficient, run_id, contributions).
# - Génère des alertes visuelles via miya_alerts pour les conflits de signaux élevés (conflict_coefficient > 0.5).
# - Utilise trade_success_prob comme critère de décision dans LiveTrading avec un seuil configurable.
# - Tests unitaires dans tests/test_run_system.py.
# - Policies Note: The official directory for routing policies is src/model/router/policies.

import argparse
import gzip
import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import psutil

from src.data.data_provider import DataProvider
from src.features.feature_pipeline import FeaturePipeline
from src.model.trade_probability import TradeProbabilityPredictor
from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import config_manager
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.trading.live_trading import LiveTrading
from src.utils.standard import with_retries
from src.utils.telegram_alert import send_telegram_alert

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "es_config.yaml")
LOG_DIR = os.path.join(BASE_DIR, "data", "logs", "system")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "system_snapshots")
CSV_LOG_PATH = os.path.join(BASE_DIR, "data", "logs", "run_system_performance.csv")

# Configuration du logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "run_system.log"),
    level=logging.INFO,
    format="%(asctime)s,%(levelname)s,%(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Configuration par défaut
DEFAULT_CONFIG = {
    "system": {
        "mode": "paper",
        "data_provider": {"source": "iqfeed"},
        "feature_pipeline": {
            "num_features": 350,
            "shap_features": 150,
            "required_features": [
                "bid_ask_imbalance",
                "trade_aggressiveness",
                "iv_skew",
                "iv_term_structure",
                "vix_es_correlation",
                "call_iv_atm",
                "option_skew",
                "news_impact_score",
            ],
        },
        "trading": {
            "max_position_size": 5,
            "reward_threshold": 0.01,
            "news_impact_threshold": 0.5,
            "trade_success_threshold": 0.6,
        },
        "retry_attempts": 3,
        "retry_delay_base": 2.0,
        "buffer_size": 100,
    }
}


class SystemRunner:
    """
    Classe pour exécuter le pipeline global collecte → features → trading.
    """

    def __init__(self, mode: str = "paper", config_path: Path = Path(CONFIG_PATH)):
        """
        Initialise le lanceur du système.

        Args:
            mode (str): Mode d'exécution (paper ou live).
            config_path (Path): Chemin du fichier de configuration YAML.
        """
        self.mode = mode
        self.config_path = config_path
        self.config = None
        self.log_buffer = []
        self.buffer_size = DEFAULT_CONFIG["system"]["buffer_size"]
        try:
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            self.validate_config()
            miya_speak(
                f"SystemRunner initialisé en mode {self.mode}",
                tag="RUN_SYSTEM",
                voice_profile="calm",
                priority=2,
            )
            AlertManager().send_alert(
                f"SystemRunner initialisé en mode {self.mode}", priority=1
            )
            logger.info(f"SystemRunner initialisé en mode {self.mode}")
            self.log_performance("init", 0, success=True)
            self.save_snapshot(
                "init", {"mode": self.mode, "config_path": str(self.config_path)}
            )
        except Exception as e:
            error_msg = f"Erreur initialisation SystemRunner: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="RUN_SYSTEM", voice_profile="urgent", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            raise

    def validate_config(self):
        """
        Valide la configuration et l'environnement.

        Raises:
            FileNotFoundError: Si le fichier de configuration est introuvable.
            ValueError: Si la configuration ou le mode est invalide.
        """
        start_time = datetime.now()
        try:
            if not self.config_path.exists():
                error_msg = f"Fichier de configuration introuvable : {self.config_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            self.config = config_manager.get_config("es_config.yaml")
            if "system" not in self.config:
                error_msg = "Clé 'system' manquante dans la configuration"
                logger.error(error_msg)
                raise ValueError(error_msg)

            required_keys = ["data_provider", "feature_pipeline", "trading"]
            missing_keys = [
                key for key in required_keys if key not in self.config["system"]
            ]
            if missing_keys:
                error_msg = f"Clés manquantes dans 'system': {missing_keys}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            if self.mode not in ["paper", "live"]:
                error_msg = f"Mode invalide : {self.mode}. Modes valides : paper, live"
                logger.error(error_msg)
                raise ValueError(error_msg)

            if self.config["system"]["data_provider"].get("source") != "iqfeed":
                error_msg = "Source de données doit être 'iqfeed'"
                logger.error(error_msg)
                raise ValueError(error_msg)

            required_features = DEFAULT_CONFIG["system"]["feature_pipeline"][
                "required_features"
            ]
            config_features = self.config["system"]["feature_pipeline"].get(
                "required_features", []
            )
            missing_features = [
                f for f in required_features if f not in config_features
            ]
            if missing_features:
                error_msg = f"Features requises manquantes dans la configuration : {missing_features}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                f"Configuration validée : mode={self.mode}",
                tag="RUN_SYSTEM",
                level="info",
                priority=2,
            )
            AlertManager().send_alert(
                f"Configuration validée : mode={self.mode}", priority=1
            )
            logger.info(
                f"Configuration validée : mode={self.mode}, config={self.config_path}"
            )
            self.log_performance("validate_config", latency, success=True)
            self.save_snapshot(
                "validate_config", {"mode": self.mode, "config": self.config["system"]}
            )
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur lors de la validation de la configuration : {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.log_performance(
                "validate_config", latency, success=False, error=str(e)
            )
            raise

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ) -> None:
        """
        Journalise les performances des opérations critiques.

        Args:
            operation (str): Nom de l’opération (ex. : run_pipeline).
            latency (float): Latence en secondes.
            success (bool): Succès de l’opération.
            error (str, optional): Message d’erreur si échec.
            **kwargs: Paramètres supplémentaires (ex. : num_records, normalized_score, conflict_coefficient).
        """
        try:
            memory_usage = (
                psutil.Process().memory_info().rss / 1024 / 1024
            )  # Mémoire en Mo
            cpu_percent = psutil.cpu_percent()
            if memory_usage > 1024:
                error_msg = f"ALERT: Utilisation mémoire élevée ({memory_usage:.2f} MB)"
                miya_alerts(error_msg, tag="RUN_SYSTEM", level="error", priority=5)
                AlertManager().send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
            log_entry = {
                "timestamp": str(datetime.now()),
                "operation": operation,
                "latency": latency,
                "success": success,
                "error": error,
                "memory_usage_mb": memory_usage,
                "cpu_percent": cpu_percent,
                **kwargs,
            }
            self.log_buffer.append(log_entry)
            if len(self.log_buffer) >= self.buffer_size:
                log_df = pd.DataFrame(self.log_buffer)
                os.makedirs(os.path.dirname(CSV_LOG_PATH), exist_ok=True)
                if not os.path.exists(CSV_LOG_PATH):
                    log_df.to_csv(CSV_LOG_PATH, index=False, encoding="utf-8")
                else:
                    log_df.to_csv(
                        CSV_LOG_PATH,
                        mode="a",
                        header=False,
                        index=False,
                        encoding="utf-8",
                    )
                self.log_buffer = []
        except Exception as e:
            error_msg = f"Erreur journalisation performance: {str(e)}"
            miya_alerts(error_msg, tag="RUN_SYSTEM", level="error", priority=3)
            AlertManager().send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def save_snapshot(self, snapshot_type: str, data: dict) -> None:
        """
        Sauvegarde un instantané des résultats avec compression gzip.

        Args:
            snapshot_type (str): Type de snapshot (ex. : run_pipeline).
            data (dict): Données à sauvegarder.
        """
        try:
            start_time = datetime.now()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = {
                "timestamp": timestamp,
                "type": snapshot_type,
                "data": data,
                "buffer_size": len(self.log_buffer),
            }
            path = os.path.join(
                SNAPSHOT_DIR, f"snapshot_{snapshot_type}_{timestamp}.json"
            )
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            with gzip.open(f"{path}.gz", "wt", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=4)
            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                f"Snapshot {snapshot_type} sauvegardé: {path}.gz",
                tag="RUN_SYSTEM",
                level="info",
                priority=1,
            )
            AlertManager().send_alert(
                f"Snapshot {snapshot_type} sauvegardé: {path}.gz", priority=1
            )
            logger.info(f"Snapshot {snapshot_type} sauvegardé: {path}.gz")
            self.log_performance("save_snapshot", latency, success=True)
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}"
            miya_alerts(error_msg, tag="RUN_SYSTEM", level="error", priority=3)
            AlertManager().send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("save_snapshot", 0, success=False, error=str(e))

    @with_retries(max_attempts=3, delay_base=2.0)
    def collect_data(self) -> pd.DataFrame:
        """
        Collecte les données via DataProvider.

        Returns:
            pd.DataFrame: Données collectées.

        Raises:
            ValueError: Si aucune donnée n’est collectée.
        """
        start_time = datetime.now()
        try:
            logger.info("Collecte des données via DataProvider...")
            data_provider = DataProvider()
            data = data_provider.get_data()
            if data.empty:
                error_msg = "Aucune donnée collectée."
                logger.error(error_msg)
                raise ValueError(error_msg)
            required_features = self.config["system"]["feature_pipeline"].get(
                "required_features", []
            )
            missing_features = [f for f in required_features if f not in data.columns]
            if missing_features:
                error_msg = f"Features manquantes dans les données collectées : {missing_features}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                f"Données collectées : {len(data)} enregistrements",
                tag="RUN_SYSTEM",
                level="info",
                priority=2,
            )
            AlertManager().send_alert(
                f"Données collectées : {len(data)} enregistrements", priority=1
            )
            logger.info(f"Données collectées : {len(data)} enregistrements.")
            self.log_performance(
                "collect_data", latency, success=True, num_records=len(data)
            )
            self.save_snapshot("collect_data", {"num_records": len(data)})
            return data
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur collecte données : {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="RUN_SYSTEM", voice_profile="urgent", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("collect_data", latency, success=False, error=str(e))
            raise

    @with_retries(max_attempts=3, delay_base=2.0)
    def generate_features(self, data: pd.DataFrame) -> dict:
        """
        Génère les features via FeaturePipeline.

        Args:
            data (pd.DataFrame): Données d’entrée.

        Returns:
            dict: Features générées.

        Raises:
            ValueError: Si aucune feature n’est générée.
        """
        start_time = datetime.now()
        try:
            logger.info("Génération des features via FeaturePipeline...")
            feature_pipeline = FeaturePipeline()
            features = feature_pipeline.generate_features(data)
            if not features:
                error_msg = "Aucune feature générée."
                logger.error(error_msg)
                raise ValueError(error_msg)
            required_features = self.config["system"]["feature_pipeline"].get(
                "required_features", []
            )
            missing_features = [f for f in required_features if f not in features]
            if missing_features:
                error_msg = f"Features manquantes dans les features générées : {missing_features}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                f"Features générées : {len(features)} métriques",
                tag="RUN_SYSTEM",
                level="info",
                priority=2,
            )
            AlertManager().send_alert(
                f"Features générées : {len(features)} métriques", priority=1
            )
            logger.info(f"Features générées : {len(features)} métriques.")
            self.log_performance(
                "generate_features", latency, success=True, num_features=len(features)
            )
            self.save_snapshot("generate_features", {"num_features": len(features)})
            return features
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur génération features : {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="RUN_SYSTEM", voice_profile="urgent", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "generate_features", latency, success=False, error=str(e)
            )
            raise

    @with_retries(max_attempts=3, delay_base=2.0)
    def execute_trading(self, features: dict) -> dict:
        """
        Exécute le trading via LiveTrading, en utilisant trade_success_prob et signal_metadata.

        Args:
            features (dict): Features générées.

        Returns:
            dict: Décision de trading.

        Raises:
            ValueError: Si aucune décision de trading n’est produite.
        """
        start_time = datetime.now()
        try:
            logger.info(f"Exécution du trading ({self.mode}) via LiveTrading...")
            trading_config = self.config["system"]["trading"]

            # Calculer trade_success_prob et signal_metadata
            predictor = TradeProbabilityPredictor(market="ES")
            data_df = pd.DataFrame([features])
            trade_success_prob = predictor.predict(data_df, market="ES")
            signal_metadata = predictor.resolve_signals(data_df)

            # Vérifier les conflits de signaux élevés
            conflict_threshold = (
                self.config["system"]
                .get("signal_resolver", {})
                .get("thresholds", {})
                .get("conflict_coefficient_alert", 0.5)
            )
            if signal_metadata.get("conflict_coefficient", 0.0) > conflict_threshold:
                conflict_msg = "Conflit de signaux détecté. Analyse requise."
                miya_alerts(
                    conflict_msg, tag="RUN_SYSTEM", voice_profile="urgent", priority=3
                )
                AlertManager().send_alert(conflict_msg, priority=3)
                send_telegram_alert(conflict_msg)
                logger.warning(conflict_msg)

            trading_config["trade_success_prob"] = trade_success_prob
            trading_config["signal_metadata"] = signal_metadata

            live_trading = LiveTrading(mode=self.mode, env_config=trading_config)
            trade_decision = live_trading.process_features(
                features,
                trade_success_prob=trade_success_prob,
                signal_metadata=signal_metadata,
            )
            if not trade_decision:
                error_msg = "Aucune décision de trading produite."
                logger.error(error_msg)
                raise ValueError(error_msg)

            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                f"Décision de trading : {trade_decision}, Probabilité de succès : {trade_success_prob:.2f}",
                tag="RUN_SYSTEM",
                level="info",
                priority=2,
            )
            AlertManager().send_alert(
                f"Décision de trading : {trade_decision}, Probabilité de succès : {trade_success_prob:.2f}",
                priority=1,
            )
            logger.info(
                f"Décision de trading : {trade_decision}, Probabilité de succès : {trade_success_prob:.2f}, Signal Metadata : {signal_metadata}"
            )
            self.log_performance(
                "execute_trading",
                latency,
                success=True,
                decision=str(trade_decision),
                trade_success_prob=trade_success_prob,
                normalized_score=signal_metadata.get("normalized_score", 0.0),
                conflict_coefficient=signal_metadata.get("conflict_coefficient", 0.0),
                run_id=signal_metadata.get("run_id", None),
            )
            self.save_snapshot(
                "execute_trading",
                {
                    "decision": trade_decision,
                    "trade_success_prob": trade_success_prob,
                    "signal_metadata": signal_metadata,
                    "contributions": signal_metadata.get("contributions", {}),
                },
            )
            return trade_decision
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur exécution trading : {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="RUN_SYSTEM", voice_profile="urgent", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "execute_trading", latency, success=False, error=str(e)
            )
            raise

    def run_pipeline(self) -> bool:
        """
        Exécute le pipeline global : collecte, génération de features, trading.

        Returns:
            bool: True si le pipeline s'exécute avec succès, False sinon.
        """
        start_time = datetime.now()
        try:
            logger.info("Démarrage du pipeline global...")
            miya_speak(
                "Démarrage du pipeline global...",
                tag="RUN_SYSTEM",
                voice_profile="calm",
                priority=3,
            )
            AlertManager().send_alert("Démarrage du pipeline global", priority=2)

            # Étape 1 : Collecte des données
            data = self.collect_data()

            # Étape 2 : Génération des features
            features = self.generate_features(data)

            # Étape 3 : Trading
            trade_decision = self.execute_trading(features)

            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                "Pipeline global exécuté avec succès",
                tag="RUN_SYSTEM",
                level="info",
                priority=3,
            )
            AlertManager().send_alert("Pipeline global exécuté avec succès", priority=1)
            logger.info("Pipeline global exécuté avec succès.")
            self.log_performance(
                "run_pipeline",
                latency,
                success=True,
                num_records=len(data),
                num_features=len(features),
                decision=str(trade_decision),
            )
            self.save_snapshot(
                "run_pipeline",
                {
                    "num_records": len(data),
                    "num_features": len(features),
                    "decision": trade_decision,
                },
            )
            return True
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur lors de l'exécution du pipeline : {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="RUN_SYSTEM", voice_profile="urgent", priority=5)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("run_pipeline", latency, success=False, error=str(e))
            self.save_snapshot("run_pipeline", {"error": str(e)})
            raise


def main():
    """
    Point d'entrée pour exécuter le pipeline avec des arguments CLI.
    """
    parser = argparse.ArgumentParser(
        description="Exécute le pipeline global de MIA_IA_SYSTEM_v2_2025."
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="Mode d'exécution (paper ou live)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=CONFIG_PATH,
        help="Chemin du fichier de configuration",
    )

    args = parser.parse_args()

    try:
        runner = SystemRunner(mode=args.mode, config_path=Path(args.config))
        runner.run_pipeline()
        print(f"Pipeline exécuté avec succès en mode {args.mode}.")
        miya_speak(
            f"Pipeline exécuté avec succès en mode {args.mode}",
            tag="RUN_SYSTEM",
            voice_profile="calm",
            priority=3,
        )
        AlertManager().send_alert(
            f"Pipeline exécuté avec succès en mode {args.mode}", priority=1
        )
    except Exception as e:
        error_msg = (
            f"Échec de l'exécution du pipeline : {str(e)}\n{traceback.format_exc()}"
        )
        print(error_msg)
        miya_alerts(error_msg, tag="RUN_SYSTEM", voice_profile="urgent", priority=5)
        AlertManager().send_alert(error_msg, priority=4)
        send_telegram_alert(error_msg)
        logger.error(error_msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
