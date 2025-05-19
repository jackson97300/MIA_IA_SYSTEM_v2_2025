# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/scripts/generate_defensive_policy.py
# Script pour générer la politique SAC pour le mode défensif de MIA_IA_SYSTEM_v2_2025.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Crée une politique SAC initiale pour le trading ES en mode défensif (Phase 10).
#        Conforme à la Phase 8 (auto-conscience via alertes),
#        Phase 10 (génération de politiques), et Phase 16 (ensemble et transfer learning via SAC).
#
# Dépendances : os, logging, pickle, pathlib, stable_baselines3>=2.0.0, numpy>=1.23.0,
#               psutil>=5.9.8, json, yaml>=6.0.0, gzip, datetime, traceback,
#               src.model.utils.config_manager, src.model.utils.alert_manager,
#               src.model.utils.miya_console, src.utils.telegram_alert, src.utils.standard,
#               src.model.trading_env
#
# Inputs : config/es_config.yaml
#
# Outputs : src/model/router/policies/defensive_policy.pkl,
#           data/logs/market/generate_defensive_policy.log,
#           data/logs/generate_defensive_policy_performance.csv,
#           data/defensive_policy_snapshots/snapshot_*.json.gz
#
# Notes :
# - Utilise TradingEnv avec 350 features pour l’entraînement et 150 SHAP features pour l’inférence.
# - Implémente retries (max 3, délai 2^attempt), logs psutil, alertes via alert_manager.py.
# - Tests unitaires dans tests/test_generate_defensive_policy.py.

import gzip
import json
import logging
import os
import pickle
import traceback
from datetime import datetime
from pathlib import Path

import psutil
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

from src.model.trading_env import TradingEnv
from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import config_manager
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.utils.standard import with_retries
from src.utils.telegram_alert import send_telegram_alert

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "es_config.yaml")
POLICY_PATH = os.path.join(
    BASE_DIR, "src", "model", "router", "policies", "defensive_policy.pkl"
)
LOG_DIR = os.path.join(BASE_DIR, "data", "logs", "market")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "defensive_policy_snapshots")
CSV_LOG_PATH = os.path.join(
    BASE_DIR, "data", "logs", "generate_defensive_policy_performance.csv"
)

# Configuration du logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "generate_defensive_policy.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)


class DefensivePolicyGenerator:
    """
    Classe pour générer et sauvegarder une politique SAC pour le mode défensif.
    """

    def __init__(self, config_path: Path = Path(CONFIG_PATH)):
        """
        Initialise le générateur de politique.

        Args:
            config_path (Path): Chemin du fichier de configuration YAML.
        """
        self.config_path = config_path
        self.config = None
        self.log_buffer = []
        self.buffer_size = 100
        try:
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            self.validate_config()
            miya_speak(
                "DefensivePolicyGenerator initialisé",
                tag="DEFENSIVE_POLICY",
                voice_profile="calm",
                priority=2,
            )
            AlertManager().send_alert("DefensivePolicyGenerator initialisé", priority=1)
            logger.info("DefensivePolicyGenerator initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config_path": str(self.config_path)})
        except Exception as e:
            error_msg = f"Erreur initialisation DefensivePolicyGenerator: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="DEFENSIVE_POLICY", voice_profile="urgent", priority=4
            )
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
            ValueError: Si la configuration est invalide.
        """
        start_time = datetime.now()
        try:
            if not self.config_path.exists():
                error_msg = f"Fichier de configuration introuvable : {self.config_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            self.config = config_manager.get_config("es_config.yaml")
            if "sac_defensive" not in self.config:
                error_msg = "Clé 'sac_defensive' manquante dans la configuration"
                logger.error(error_msg)
                raise ValueError(error_msg)

            required_keys = [
                "learning_rate",
                "buffer_size",
                "batch_size",
                "ent_coef",
                "total_timesteps",
            ]
            missing_keys = [
                key for key in required_keys if key not in self.config["sac_defensive"]
            ]
            if missing_keys:
                error_msg = f"Clés manquantes dans 'sac_defensive': {missing_keys}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Vérifier l'espace disque (minimum 10 Mo)
            disk = os.statvfs(BASE_DIR)
            free_space_mb = disk.f_bavail * disk.f_frsize / (1024 * 1024)
            if free_space_mb < 10:
                error_msg = (
                    f"Espace disque insuffisant : {free_space_mb:.2f} Mo disponibles"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                "Configuration validée pour la politique SAC",
                tag="DEFENSIVE_POLICY",
                level="info",
                priority=2,
            )
            AlertManager().send_alert(
                "Configuration validée pour la politique SAC", priority=1
            )
            logger.info("Configuration validée pour la politique SAC")
            self.log_performance("validate_config", latency, success=True)
            self.save_snapshot(
                "validate_config", {"config": self.config["sac_defensive"]}
            )
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur lors de la validation de la configuration : {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            miya_alerts(
                error_msg, tag="DEFENSIVE_POLICY", voice_profile="urgent", priority=4
            )
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "validate_config", latency, success=False, error=str(e)
            )
            self.save_snapshot("validate_config", {"error": str(e)})
            raise

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ) -> None:
        """
        Journalise les performances des opérations critiques.

        Args:
            operation (str): Nom de l’opération (ex. : create_defensive_policy).
            latency (float): Latence en secondes.
            success (bool): Succès de l’opération.
            error (str, optional): Message d’erreur si échec.
            **kwargs: Paramètres supplémentaires (ex. : timesteps).
        """
        try:
            memory_usage = (
                psutil.Process().memory_info().rss / 1024 / 1024
            )  # Mémoire en Mo
            cpu_percent = psutil.cpu_percent()
            if memory_usage > 1024:
                error_msg = f"ALERT: Utilisation mémoire élevée ({memory_usage:.2f} MB)"
                miya_alerts(
                    error_msg, tag="DEFENSIVE_POLICY", level="error", priority=5
                )
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
            miya_alerts(error_msg, tag="DEFENSIVE_POLICY", level="error", priority=3)
            AlertManager().send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def save_snapshot(self, snapshot_type: str, data: Dict) -> None:
        """
        Sauvegarde un instantané des résultats avec compression gzip.

        Args:
            snapshot_type (str): Type de snapshot (ex. : create_defensive_policy).
            data (Dict): Données à sauvegarder.
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
                tag="DEFENSIVE_POLICY",
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
            miya_alerts(error_msg, tag="DEFENSIVE_POLICY", level="error", priority=3)
            AlertManager().send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("save_snapshot", 0, success=False, error=str(e))

    @with_retries(max_attempts=3, delay_base=2.0)
    def create_defensive_policy(self) -> None:
        """
        Crée une politique SAC pour le mode défensif et la sauvegarde dans defensive_policy.pkl.
        """
        start_time = datetime.now()
        try:
            # Définir les chemins
            policy_dir = Path(POLICY_PATH).parent
            policy_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Dossier vérifié/créé : {policy_dir}")

            # Créer l’environnement TradingEnv
            env_config = self.config.get(
                "trading_env",
                {
                    "max_position_size": 5,
                    "reward_threshold": 0.01,
                    "news_impact_threshold": 0.5,
                    "obs_dimensions": 350,  # 350 features
                },
            )
            env = make_vec_env(lambda: TradingEnv(**env_config), n_envs=1)
            logger.info("Environnement TradingEnv créé pour la politique SAC.")
            miya_speak(
                "Environnement TradingEnv créé",
                tag="DEFENSIVE_POLICY",
                level="info",
                priority=2,
            )
            AlertManager().send_alert("Environnement TradingEnv créé", priority=1)

            # Configurer le modèle SAC
            sac_config = self.config["sac_defensive"]
            model = SAC(
                policy="MlpPolicy",
                env=env,
                learning_rate=sac_config["learning_rate"],
                buffer_size=sac_config["buffer_size"],
                learning_starts=sac_config.get("learning_starts", 100),
                batch_size=sac_config["batch_size"],
                tau=sac_config.get("tau", 0.005),
                gamma=sac_config.get("gamma", 0.99),
                train_freq=sac_config.get("train_freq", 1),
                gradient_steps=sac_config.get("gradient_steps", 1),
                ent_coef=sac_config["ent_coef"],
                verbose=1,
                device="auto",
            )
            logger.info(
                "Modèle SAC initialisé avec hyperparamètres pour mode défensif."
            )
            miya_speak(
                "Modèle SAC initialisé",
                tag="DEFENSIVE_POLICY",
                level="info",
                priority=2,
            )
            AlertManager().send_alert("Modèle SAC initialisé", priority=1)

            # Entraîner le modèle
            total_timesteps = sac_config["total_timesteps"]
            model.learn(total_timesteps=total_timesteps, log_interval=10)
            logger.info(f"Entraînement terminé ({total_timesteps} timesteps).")
            miya_speak(
                f"Entraînement SAC terminé ({total_timesteps} timesteps)",
                tag="DEFENSIVE_POLICY",
                level="info",
                priority=2,
            )
            AlertManager().send_alert(
                f"Entraînement SAC terminé ({total_timesteps} timesteps)", priority=1
            )

            # Sauvegarder la politique
            with open(POLICY_PATH, "wb") as f:
                pickle.dump(model.policy, f)
            logger.info(f"Politique SAC sauvegardée : {POLICY_PATH}")
            miya_speak(
                f"Politique SAC sauvegardée : {POLICY_PATH}",
                tag="DEFENSIVE_POLICY",
                level="info",
                priority=3,
            )
            AlertManager().send_alert(
                f"Politique SAC sauvegardée : {POLICY_PATH}", priority=1
            )

            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "create_defensive_policy",
                latency,
                success=True,
                timesteps=total_timesteps,
            )
            self.save_snapshot(
                "create_defensive_policy",
                {
                    "policy_path": str(POLICY_PATH),
                    "timesteps": total_timesteps,
                    "config": sac_config,
                },
            )
            print(f"Politique défensive créée avec succès à {POLICY_PATH}")
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur lors de la création de la politique : {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            miya_alerts(
                error_msg, tag="DEFENSIVE_POLICY", voice_profile="urgent", priority=5
            )
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "create_defensive_policy", latency, success=False, error=str(e)
            )
            self.save_snapshot("create_defensive_policy", {"error": str(e)})
            raise


def main():
    """
    Point d'entrée pour générer la politique SAC.
    """
    try:
        generator = DefensivePolicyGenerator()
        generator.create_defensive_policy()
    except Exception as e:
        error_msg = (
            f"Échec de la création de la politique : {str(e)}\n{traceback.format_exc()}"
        )
        print(error_msg)
        miya_alerts(
            error_msg, tag="DEFENSIVE_POLICY", voice_profile="urgent", priority=5
        )
        AlertManager().send_alert(error_msg, priority=4)
        send_telegram_alert(error_msg)
        logger.error(error_msg)
        exit(1)


if __name__ == "__main__":
    main()
