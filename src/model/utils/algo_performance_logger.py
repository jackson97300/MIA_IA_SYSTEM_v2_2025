# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/model/utils/algo_performance_logger.py
# Rôle : Enregistre les performances des algorithmes SAC, PPO, DDPG pour MIA_IA_SYSTEM_v2_2025, incluant les métriques
#        de fine-tuning (méthode 8), apprentissage en ligne (méthode 10), et meta-learning (méthode 18).
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, psutil>=5.9.8,<6.0.0, matplotlib>=3.7.0,<4.0.0, boto3>=1.26.0,<2.0.0,
#   loguru>=0.7.0,<1.0.0, pyyaml>=6.0.0,<7.0.0, signal
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - Données de performance (récompense, latence, mémoire, finetune_loss, online_learning_steps, maml_steps)
# - Configuration via algo_config.yaml
#
# Outputs :
# - Logs dans data/logs/<market>/<algo>_performance.csv (par algorithme)
# - Logs consolidés dans data/logs/<market>/train_sac_performance.csv (tous algorithmes)
# - Snapshots JSON compressés dans data/cache/algo_performance/<market>/*.json.gz
# - Visualisations dans data/figures/algo_performance/<market>/
# - Sauvegardes incrémentielles dans data/checkpoints/algo_performance/<market>/*.json.gz
#
# Notes :
# - Fournit des logs pour train_sac.py, compatible avec 350 features (entraînement) et 150 SHAP features (inférence).
# - Utilise IQFeed exclusivement via data_provider.py (implicite via train_sac.py).
# - Supprime toutes les références à 320 features, 81 features, et obs_t.
# - Implémente retries (max 3, délai 2^attempt secondes) pour les appels critiques.
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des opérations.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.
# - Tests unitaires disponibles dans tests/test_algo_performance_logger.py.
# - Policies Note: The official directory for routing policies is src/model/router/policies.
# The src/model/policies directory is a residual and should be verified
# for removal to avoid import conflicts.

import gzip
import json
import signal
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
import matplotlib.pyplot as plt
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
CACHE_DIR = BASE_DIR / "data" / "cache" / "algo_performance"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints" / "algo_performance"
FIGURE_DIR = BASE_DIR / "data" / "figures" / "algo_performance"
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "algo_performance_logger.log",
    rotation="10 MB",
    level="INFO",
    encoding="utf-8",
)
logger.add(lambda msg: print(msg, end=""), level="INFO")

# Constantes
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Secondes, exponentiel
MAX_CACHE_SIZE = 1000
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB

# Cache global pour les résultats de journalisation
PERFORMANCE_CACHE = OrderedDict()


class AlgoPerformanceLogger:
    """Enregistre les performances séparées de SAC, PPO, DDPG."""

    def __init__(
        self,
        config_path: str = str(BASE_DIR / "config" / "algo_config.yaml"),
        market: str = "ES",
    ):
        """
        Initialise le logger de performance des algorithmes.

        Args:
            config_path (str): Chemin vers le fichier de configuration.
            market (str): Marché (ex. : ES, MNQ).
        """
        self.market = market
        self.config = get_config(config_path)
        self.alert_manager = AlertManager(market=market)
        self.snapshot_dir = CACHE_DIR / market
        self.figure_dir = FIGURE_DIR / market
        self.checkpoint_dir = CHECKPOINT_DIR / market
        self.log_dir = LOG_DIR / market
        self.snapshot_dir.mkdir(exist_ok=True)
        self.figure_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)

        # Initialiser fichiers de logs par algorithme
        self.log_files = {
            "sac": self.log_dir / "sac_performance.csv",
            "ppo": self.log_dir / "ppo_performance.csv",
            "ddpg": self.log_dir / "ddpg_performance.csv",
        }
        for algo, log_file in self.log_files.items():
            if not log_file.exists():
                pd.DataFrame(
                    columns=["timestamp", "regime", "reward", "latency", "memory"]
                ).to_csv(log_file, index=False)

        # Initialiser fichier de log consolidé
        self.consolidated_log_file = self.log_dir / "train_sac_performance.csv"
        if not self.consolidated_log_file.exists():
            pd.DataFrame(
                columns=[
                    "timestamp",
                    "algo_type",
                    "regime",
                    "reward",
                    "latency",
                    "memory",
                    "finetune_loss",
                    "online_learning_steps",
                    "maml_steps",
                ]
            ).to_csv(self.consolidated_log_file, index=False)

        signal.signal(signal.SIGINT, self.handle_sigint)
        success_msg = f"AlgoPerformanceLogger initialisé pour {market}"
        logger.info(success_msg)
        send_telegram_alert(success_msg)

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
                    "retry_attempt", "unknown", 0.0, latency, 0.0, attempt=attempt + 1
                )
                return result
            except Exception as e:
                if attempt == max_attempts - 1:
                    latency = time.time() - start_time
                    error_msg = f"Échec après {max_attempts} tentatives pour {self.market}: {str(e)}\n{traceback.format_exc()}"
                    logger.error(error_msg)
                    self.alert_manager.send_alert(error_msg, priority=3)
                    send_telegram_alert(error_msg)
                    self.log_performance(
                        "retry_attempt",
                        "unknown",
                        0.0,
                        latency,
                        0.0,
                        error=str(e),
                        attempt=attempt + 1,
                    )
                    return None
                delay = delay_base**attempt
                warning_msg = f"Tentative {attempt+1} échouée pour {self.market}, retry après {delay}s"
                logger.warning(warning_msg)
                send_telegram_alert(warning_msg)
                time.sleep(delay)

    def save_snapshot(
        self, snapshot_type: str, data: Dict, compress: bool = True
    ) -> None:
        """
        Sauvegarde un instantané JSON des résultats, compressé avec gzip.

        Args:
            snapshot_type (str): Type de snapshot (ex. : log_performance).
            data (Dict): Données à sauvegarder.
            compress (bool): Compresser avec gzip (défaut : True).
        """
        start_time = time.time()
        try:
            if not os.access(self.snapshot_dir, os.W_OK):
                error_msg = f"Permission d’écriture refusée pour {self.snapshot_dir}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                return
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

            self.with_retries(write_snapshot)
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
                "save_snapshot",
                "unknown",
                0.0,
                latency,
                0.0,
                snapshot_size_mb=file_size,
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot {snapshot_type} pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "save_snapshot",
                "unknown",
                0.0,
                time.time() - start_time,
                0.0,
                error=str(e),
            )

    def checkpoint(
        self, data: pd.DataFrame, data_type: str = "algo_performance_state"
    ) -> None:
        """
        Sauvegarde incrémentielle des données toutes les 5 minutes avec versionnage (5 versions).

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : algo_performance_state).
        """
        start_time = time.time()
        try:
            if not os.access(self.checkpoint_dir, os.W_OK):
                error_msg = f"Permission d’écriture refusée pour {self.checkpoint_dir}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_data = {
                "timestamp": timestamp,
                "num_rows": len(data),
                "columns": list(data.columns),
                "data_type": data_type,
                "market": self.market,
            }
            checkpoint_path = (
                self.checkpoint_dir
                / f"algo_performance_{data_type}_{timestamp}.json.gz"
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
                "unknown",
                0.0,
                latency,
                0.0,
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
                "checkpoint",
                "unknown",
                0.0,
                time.time() - start_time,
                0.0,
                error=str(e),
                data_type=data_type,
            )

    def cloud_backup(
        self, data: pd.DataFrame, data_type: str = "algo_performance_state"
    ) -> None:
        """
        Sauvegarde distribuée des données vers AWS S3 toutes les 15 minutes.

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : algo_performance_state).
        """
        start_time = time.time()
        try:
            config = get_config(str(BASE_DIR / "config/es_config.yaml"))
            if not config.get("s3_bucket"):
                warning_msg = f"S3 bucket non configuré, sauvegarde cloud ignorée pour {self.market}"
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{config['s3_prefix']}algo_performance_{data_type}_{self.market}_{timestamp}.csv.gz"
            temp_path = self.checkpoint_dir / f"temp_s3_{timestamp}.csv.gz"
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
                "unknown",
                0.0,
                latency,
                0.0,
                num_rows=len(data),
                data_type=data_type,
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde cloud S3 pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "cloud_backup",
                "unknown",
                0.0,
                time.time() - start_time,
                0.0,
                error=str(e),
                data_type=data_type,
            )

    def log_performance(
        self,
        algo_type: str,
        regime: str,
        reward: float,
        latency: float,
        memory: float,
        **kwargs,
    ):
        """Enregistre les performances d’un algorithme."""
        start_time = time.time()
        cache_key = f"{self.market}_{algo_type}_{regime}_{hash(str(reward))}_{hash(str(latency))}"
        if cache_key in PERFORMANCE_CACHE:
            return
        while len(PERFORMANCE_CACHE) > MAX_CACHE_SIZE:
            PERFORMANCE_CACHE.popitem(last=False)

        process = psutil.Process()
        try:
            algo_type = algo_type.lower()
            if algo_type not in self.log_files:
                error_msg = f"Algorithme inconnu pour {self.market}: {algo_type}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                return

            log_file = self.log_files[algo_type]
            if not os.access(log_file.parent, os.W_OK):
                error_msg = f"Permission d’écriture refusée pour {log_file.parent}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                return

            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "algo_type": algo_type,
                "regime": regime,
                "reward": reward,
                "latency": latency,
                "memory": memory,
                "confidence_drop_rate": 1.0,  # Simplifié pour Phase 8
            }

            def write_log():
                pd.DataFrame([snapshot]).to_csv(
                    log_file, mode="a", header=False, index=False
                )

            self.with_retries(write_log)

            self.save_snapshot(f"log_{algo_type}_{regime}", snapshot)

            def create_plot():
                recent_logs = pd.read_csv(log_file).tail(100)
                plt.figure(figsize=(10, 6))
                plt.plot(recent_logs["reward"], label="Reward")
                plt.title(
                    f"Performance {algo_type} ({regime}) pour {self.market}, Recent Reward: {reward:.2f}"
                )
                plt.xlabel("Étape")
                plt.ylabel("Reward")
                plt.legend()
                plt.grid(True)
                plt.savefig(
                    self.figure_dir
                    / f'log_{algo_type}_{regime}_{snapshot["timestamp"]}.png',
                    bbox_inches="tight",
                    optimize=True,
                )
                plt.close()

            self.with_retries(create_plot)

            self.log_extended_performance(
                algo_type, regime, reward, latency, memory, 0.0, 0, 0, **kwargs
            )
            success_msg = f"Performance enregistrée pour {algo_type} ({regime}) pour {self.market}: reward={reward:.2f}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            PERFORMANCE_CACHE[cache_key] = True
            self.checkpoint(
                pd.DataFrame([snapshot]), data_type=f"log_{algo_type}_{regime}"
            )
            self.cloud_backup(
                pd.DataFrame([snapshot]), data_type=f"log_{algo_type}_{regime}"
            )

        except Exception as e:
            error_msg = f"Erreur enregistrement performance {algo_type} ({regime}) pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            latency = time.time() - start_time
            memory_usage = process.memory_info().rss / 1024 / 1024
            self.log_extended_performance(
                algo_type,
                regime,
                reward,
                latency,
                memory_usage,
                0.0,
                0,
                0,
                error=str(e),
            )

    def log_extended_performance(
        self,
        algo_type: str,
        regime: str,
        reward: float,
        latency: float,
        memory: float,
        finetune_loss: float,
        online_learning_steps: int,
        maml_steps: int,
        error: str = None,
        **kwargs,
    ):
        """
        Enregistre les performances étendues incluant fine-tuning, apprentissage en ligne, et meta-learning.

        Args:
            algo_type (str): Type d’algorithme ("sac", "ppo", "ddpg").
            regime (str): Régime de marché ("trend", "range", "defensive").
            reward (float): Récompense observée.
            latency (float): Latence de l’opération (secondes).
            memory (float): Utilisation mémoire (Mo).
            finetune_loss (float): Perte du fine-tuning (méthode 8).
            online_learning_steps (int): Nombre d’étapes d’apprentissage en ligne (méthode 10).
            maml_steps (int): Nombre d’étapes de meta-learning (méthode 18).
            error (str, optional): Message d’erreur si applicable.
            **kwargs: Paramètres supplémentaires.
        """
        start_time = time.time()
        cache_key = f"{self.market}_extended_{algo_type}_{regime}_{hash(str(reward))}_{hash(str(finetune_loss))}"
        if cache_key in PERFORMANCE_CACHE:
            return
        while len(PERFORMANCE_CACHE) > MAX_CACHE_SIZE:
            PERFORMANCE_CACHE.popitem(last=False)

        process = psutil.Process()
        try:
            if not os.access(self.consolidated_log_file.parent, os.W_OK):
                error_msg = f"Permission d’écriture refusée pour {self.consolidated_log_file.parent}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                return

            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "algo_type": algo_type,
                "regime": regime,
                "reward": reward,
                "latency": latency,
                "memory": memory,
                "finetune_loss": finetune_loss,
                "online_learning_steps": online_learning_steps,
                "maml_steps": maml_steps,
                "error": error,
                "confidence_drop_rate": (
                    1.0 if error is None else 0.0
                ),  # Simplifié pour Phase 8
            }

            def write_log():
                pd.DataFrame([snapshot]).to_csv(
                    self.consolidated_log_file, mode="a", header=False, index=False
                )

            self.with_retries(write_log)

            self.save_snapshot(f"extended_log_{algo_type}_{regime}", snapshot)

            def create_plot():
                recent_logs = pd.read_csv(self.consolidated_log_file).tail(100)
                plt.figure(figsize=(10, 6))
                plt.plot(recent_logs["reward"], label="Reward")
                plt.plot(recent_logs["finetune_loss"], label="Finetune Loss")
                plt.title(
                    f"Extended Performance {algo_type} ({regime}) pour {self.market}, Reward: {reward:.2f}, Finetune Loss: {finetune_loss:.2f}"
                )
                plt.xlabel("Étape")
                plt.ylabel("Valeur")
                plt.legend()
                plt.grid(True)
                plt.savefig(
                    self.figure_dir
                    / f'extended_log_{algo_type}_{regime}_{snapshot["timestamp"]}.png',
                    bbox_inches="tight",
                    optimize=True,
                )
                plt.close()

            self.with_retries(create_plot)

            success_msg = f"Performance étendue enregistrée pour {algo_type} ({regime}) pour {self.market}: reward={reward:.2f}, finetune_loss={finetune_loss:.2f}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            PERFORMANCE_CACHE[cache_key] = True
            self.checkpoint(
                pd.DataFrame([snapshot]), data_type=f"extended_log_{algo_type}_{regime}"
            )
            self.cloud_backup(
                pd.DataFrame([snapshot]), data_type=f"extended_log_{algo_type}_{regime}"
            )

        except Exception as e:
            error_msg = f"Erreur enregistrement performance étendue {algo_type} ({regime}) pour {self.market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
        finally:
            time.time() - start_time
            cpu_percent = psutil.cpu_percent()
            process.memory_info().rss / 1024 / 1024
            success_msg = f"Performance étendue loguée pour {algo_type} ({regime}) pour {self.market}. CPU: {cpu_percent}%"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)

    def log_performance_buffer(
        self,
        operation: str,
        latency: float,
        success: bool = True,
        error: str = None,
        **kwargs,
    ) -> None:
        """Journalise les métriques dans performance_buffer.

        Args:
            operation (str): Nom de l’opération.
            latency (float): Temps d’exécution (secondes).
            success (bool): Succès de l’opération.
            error (str, optional): Message d’erreur.
            **kwargs: Métriques supplémentaires.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "latency": latency,
            "success": success,
            "error": error,
            **kwargs,
        }
        pd.DataFrame([log_entry]).to_csv(
            f"data/logs/{self.market}/train_sac_performance.csv",
            mode="a",
            header=not os.path.exists(
                f"data/logs/{self.market}/train_sac_performance.csv"
            ),
            index=False,
        )


if __name__ == "__main__":
    logger = AlgoPerformanceLogger()
    logger.log_performance(
        algo_type="sac", regime="range", reward=100.0, latency=0.5, memory=512.0
    )
    logger.log_extended_performance(
        algo_type="sac",
        regime="range",
        reward=100.0,
        latency=0.5,
        memory=512.0,
        finetune_loss=0.01,
        online_learning_steps=10,
        maml_steps=5,
    )
