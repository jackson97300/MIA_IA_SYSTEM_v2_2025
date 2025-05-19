# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/model/utils/hyperparam_manager.py
# Rôle : Gère les hyperparamètres pour SAC, PPO, DDPG (méthodes 6, 8, 18) (Phase 14).
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, pyyaml>=6.0.0,<7.0.0, psutil>=5.9.8,<6.0.0, boto3>=1.26.0,<2.0.0,
#   loguru>=0.7.0,<1.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/algo_config.yaml (hyperparamètres pour SAC, PPO, DDPG)
#
# Outputs :
# - Logs dans data/logs/market/hyperparam_manager.log
# - Logs de performance dans data/logs/market/hyperparam_manager_performance.csv
# - Snapshots JSON compressés dans data/cache/hyperparams/<market>/*.json.gz
# - Sauvegardes incrémentielles dans data/checkpoints/hyperparams/<market>/*.json.gz
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Intègre les méthodes 6 (SAC), 8 (fine-tuning), 18 (meta-learning) et la Phase 14 (gestion des hyperparamètres).
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des hyperparamètres.
# - Tests unitaires disponibles dans tests/test_hyperparam_manager.py.

import gzip
import json
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import boto3
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
LOG_DIR = BASE_DIR / "data" / "logs" / "market"
CACHE_DIR = BASE_DIR / "data" / "cache" / "hyperparams"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints" / "hyperparams"
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "hyperparam_manager.log", rotation="10 MB", level="INFO", encoding="utf-8"
)
logger.add(lambda msg: print(msg, end=""), level="INFO")

# Constantes
PERF_LOG_PATH = LOG_DIR / "hyperparam_manager_performance.csv"
MAX_RETRIES = 3
RETRY_DELAY = 2.0
MAX_CACHE_SIZE = 1000
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB

# Cache global pour les résultats de get_hyperparams
hyperparam_cache = OrderedDict()


class HyperparamManager:
    """
    Classe pour gérer les hyperparamètres des algorithmes de reinforcement learning.
    """

    def __init__(self, config_path: Path = BASE_DIR / "config" / "algo_config.yaml"):
        """
        Initialise le gestionnaire d'hyperparamètres.

        Args:
            config_path (Path): Chemin du fichier YAML contenant les hyperparamètres.
        """
        self.config_path = config_path
        self.hyperparams = self._load_config()

    def log_performance(
        self,
        operation: str,
        latency: float,
        success: bool = True,
        error: str = None,
        market: str = "ES",
        **kwargs,
    ) -> None:
        """
        Enregistre les performances (CPU, mémoire, latence) dans hyperparam_manager_performance.csv.

        Args:
            operation (str): Nom de l’opération.
            latency (float): Temps d’exécution en secondes.
            success (bool): Indique si l’opération a réussi.
            error (str): Message d’erreur (si applicable).
            market (str): Marché (ex. : ES, MNQ).
        """
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
            cpu_percent = psutil.cpu_percent()
            confidence_drop_rate = 1.0 if success else 0.0  # Simplifié pour Phase 8
            if memory_usage > 1024:
                alert_msg = (
                    f"ALERTE: Usage mémoire élevé ({memory_usage:.2f} MB) pour {market}"
                )
                logger.warning(alert_msg)
                AlertManager().send_alert(alert_msg, priority=5)
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
                "market": market,
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

            self.with_retries(save_log, market=market)
            logger.info(
                f"Performance journalisée pour {operation}. CPU: {cpu_percent}%"
            )
            self.save_snapshot("log_performance", log_entry, market=market)
        except Exception as e:
            error_msg = f"Erreur journalisation performance pour {market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            AlertManager().send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)

    def save_snapshot(
        self, snapshot_type: str, data: Dict, market: str = "ES", compress: bool = True
    ) -> None:
        """
        Sauvegarde un instantané JSON des résultats, compressé avec gzip.

        Args:
            snapshot_type (str): Type de snapshot (ex. : hyperparam_load).
            data (Dict): Données à sauvegarder.
            market (str): Marché (ex. : ES, MNQ).
            compress (bool): Compresser avec gzip (défaut : True).
        """
        start_time = time.time()
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = {
                "timestamp": timestamp,
                "type": snapshot_type,
                "market": market,
                "data": data,
            }
            snapshot_dir = CACHE_DIR / market
            snapshot_dir.mkdir(exist_ok=True)
            snapshot_path = snapshot_dir / f"snapshot_{snapshot_type}_{timestamp}.json"

            def write_snapshot():
                if compress:
                    with gzip.open(f"{snapshot_path}.gz", "wt", encoding="utf-8") as f:
                        json.dump(snapshot, f, indent=4)
                else:
                    with open(snapshot_path, "w", encoding="utf-8") as f:
                        json.dump(snapshot, f, indent=4)

            self.with_retries(write_snapshot, market=market)
            save_path = f"{snapshot_path}.gz" if compress else snapshot_path
            file_size = os.path.getsize(save_path) / 1024 / 1024
            if file_size > 1.0:
                alert_msg = (
                    f"Snapshot size {file_size:.2f} MB exceeds 1 MB pour {market}"
                )
                logger.warning(alert_msg)
                AlertManager().send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
            latency = time.time() - start_time
            success_msg = (
                f"Snapshot {snapshot_type} sauvegardé pour {market}: {save_path}"
            )
            logger.info(success_msg)
            AlertManager().send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "save_snapshot",
                latency,
                success=True,
                snapshot_size_mb=file_size,
                market=market,
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot {snapshot_type} pour {market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            AlertManager().send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "save_snapshot", 0, success=False, error=str(e), market=market
            )

    def checkpoint(
        self,
        data: pd.DataFrame,
        data_type: str = "hyperparam_state",
        market: str = "ES",
    ) -> None:
        """
        Sauvegarde incrémentielle des données toutes les 5 minutes avec versionnage (5 versions).

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : hyperparam_state).
            market (str): Marché (ex. : ES, MNQ).
        """
        try:
            start_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_data = {
                "timestamp": timestamp,
                "num_rows": len(data),
                "columns": list(data.columns),
                "data_type": data_type,
                "market": market,
            }
            checkpoint_dir = CHECKPOINT_DIR / market
            checkpoint_dir.mkdir(exist_ok=True)
            checkpoint_path = (
                checkpoint_dir / f"hyperparam_{data_type}_{timestamp}.json.gz"
            )
            checkpoint_versions = []

            def write_checkpoint():
                with gzip.open(checkpoint_path, "wt", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, indent=4)
                data.to_csv(
                    checkpoint_path.with_suffix(".csv"), index=False, encoding="utf-8"
                )

            self.with_retries(write_checkpoint, market=market)
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
            success_msg = f"Checkpoint sauvegardé pour {market}: {checkpoint_path}"
            logger.info(success_msg)
            AlertManager().send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "checkpoint",
                latency,
                success=True,
                file_size_mb=file_size,
                num_rows=len(data),
                data_type=data_type,
                market=market,
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde checkpoint pour {market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            AlertManager().send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "checkpoint",
                0,
                success=False,
                error=str(e),
                data_type=data_type,
                market=market,
            )

    def cloud_backup(
        self,
        data: pd.DataFrame,
        data_type: str = "hyperparam_state",
        market: str = "ES",
    ) -> None:
        """
        Sauvegarde distribuée des données vers AWS S3 toutes les 15 minutes.

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : hyperparam_state).
            market (str): Marché (ex. : ES, MNQ).
        """
        try:
            start_time = time.time()
            config = get_config(str(BASE_DIR / "config/es_config.yaml"))
            if not config.get("s3_bucket"):
                warning_msg = (
                    f"S3 bucket non configuré, sauvegarde cloud ignorée pour {market}"
                )
                logger.warning(warning_msg)
                AlertManager().send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{config['s3_prefix']}hyperparam_{data_type}_{market}_{timestamp}.csv.gz"
            temp_path = CHECKPOINT_DIR / market / f"temp_s3_{timestamp}.csv.gz"
            temp_path.parent.mkdir(exist_ok=True)

            def write_temp():
                data.to_csv(
                    temp_path, compression="gzip", index=False, encoding="utf-8"
                )

            self.with_retries(write_temp, market=market)
            s3_client = boto3.client("s3")

            def upload_s3():
                s3_client.upload_file(str(temp_path), config["s3_bucket"], backup_path)

            self.with_retries(upload_s3, market=market)
            temp_path.unlink()
            latency = time.time() - start_time
            success_msg = f"Sauvegarde cloud S3 effectuée pour {market}: {backup_path}"
            logger.info(success_msg)
            AlertManager().send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "cloud_backup",
                latency,
                success=True,
                num_rows=len(data),
                data_type=data_type,
                market=market,
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde cloud S3 pour {market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            AlertManager().send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "cloud_backup",
                0,
                success=False,
                error=str(e),
                data_type=data_type,
                market=market,
            )

    def with_retries(
        self,
        func: callable,
        max_attempts: int = MAX_RETRIES,
        delay_base: float = RETRY_DELAY,
        market: str = "ES",
    ) -> Optional[Any]:
        """
        Exécute une fonction avec retries exponentiels (max 3, délai 2^attempt secondes).

        Args:
            func (callable): Fonction à exécuter.
            max_attempts (int): Nombre maximum de tentatives.
            delay_base (float): Base pour le délai exponentiel.
            market (str): Marché (ex. : ES, MNQ).

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
                    market=market,
                )
                return result
            except Exception as e:
                if attempt == max_attempts - 1:
                    error_msg = f"Échec après {max_attempts} tentatives pour {market}: {str(e)}\n{traceback.format_exc()}"
                    logger.error(error_msg)
                    AlertManager().send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    self.log_performance(
                        f"retry_attempt_{attempt+1}",
                        time.time() - start_time,
                        success=False,
                        error=str(e),
                        attempt_number=attempt + 1,
                        market=market,
                    )
                    return None
                delay = delay_base**attempt
                warning_msg = (
                    f"Tentative {attempt+1} échouée pour {market}, retry après {delay}s"
                )
                logger.warning(warning_msg)
                AlertManager().send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                time.sleep(delay)

    def _load_config(self) -> Dict:
        """
        Charge le fichier de configuration YAML via config_manager.

        Returns:
            Dict: Dictionnaire contenant les hyperparamètres.

        Raises:
            FileNotFoundError: Si le fichier de configuration est introuvable.
            yaml.YAMLError: Si le fichier YAML est mal formé.
        """
        start_time = time.time()
        try:
            config = get_config(str(self.config_path))
            if not config:
                error_msg = f"Fichier de configuration vide : {self.config_path}"
                logger.error(error_msg)
                AlertManager().send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)
            success_msg = f"Configuration chargée depuis : {self.config_path}"
            logger.info(success_msg)
            AlertManager().send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance("load_config", time.time() - start_time, success=True)
            return config
        except Exception as e:
            error_msg = f"Erreur lors du chargement de {self.config_path}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            AlertManager().send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance(
                "load_config", time.time() - start_time, success=False, error=str(e)
            )
            raise

    def _validate_hyperparams(
        self,
        hyperparams: Dict,
        model_type: str,
        regime: Optional[str],
        market: str = "ES",
    ) -> None:
        """
        Valide les hyperparamètres pour un modèle et un régime donné.

        Args:
            hyperparams (Dict): Hyperparamètres à valider.
            model_type (str): Type de modèle (sac, ppo, ddpg).
            regime (str, optional): Régime de marché (trend, range, defensive).
            market (str): Marché (ex. : ES, MNQ).

        Raises:
            ValueError: Si les hyperparamètres sont invalides.
        """
        start_time = time.time()
        try:
            required_params = [
                "learning_rate",
                "batch_size",
                "gamma",
                "ent_coef",
                "l2_lambda",
            ]
            missing_params = [
                param
                for param in required_params
                if param not in hyperparams or hyperparams[param].get("value") is None
            ]
            null_count = len(missing_params)
            total_params = len(required_params)
            confidence_drop_rate = (
                null_count / total_params if total_params > 0 else 0.0
            )
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé pour {market} dans {model_type} ({regime or 'N/A'}): {confidence_drop_rate:.2f} ({null_count} paramètres nulles)"
                logger.warning(alert_msg)
                AlertManager().send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
            if missing_params:
                error_msg = f"Hyperparamètres manquants pour {model_type} ({regime or 'N/A'}) dans {market}: {missing_params}"
                logger.error(error_msg)
                AlertManager().send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            for param, constraints in [
                ("learning_rate", (0.00001, 0.01)),
                ("batch_size", (32, 1024)),
                ("gamma", (0.9, 1.0)),
                ("ent_coef", (0.0, 0.5)),
                ("l2_lambda", (0.0, 0.1)),
            ]:
                value = hyperparams[param].get("value")
                if (
                    not isinstance(value, (int, float))
                    or value < constraints[0]
                    or value > constraints[1]
                ):
                    error_msg = f"Valeur de {param} invalide pour {model_type} ({regime or 'N/A'}) dans {market}: {value}, attendu dans [{constraints[0]}, {constraints[1]}]"
                    logger.error(error_msg)
                    AlertManager().send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    raise ValueError(error_msg)

            self.save_snapshot(
                "validate_hyperparams",
                {
                    "model_type": model_type,
                    "regime": regime or "N/A",
                    "confidence_drop_rate": confidence_drop_rate,
                    "missing_params": missing_params,
                },
                market=market,
            )
            self.log_performance(
                "validate_hyperparams",
                time.time() - start_time,
                success=True,
                model_type=model_type,
                regime=regime or "N/A",
                market=market,
            )
        except Exception as e:
            error_msg = f"Erreur validation hyperparamètres pour {model_type} ({regime or 'N/A'}) dans {market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            AlertManager().send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance(
                "validate_hyperparams",
                time.time() - start_time,
                success=False,
                error=str(e),
                model_type=model_type,
                regime=regime or "N/A",
                market=market,
            )
            raise

    def get_hyperparams(
        self, model_type: str, regime: Optional[str] = None, market: str = "ES"
    ) -> Dict:
        """
        Récupère les hyperparamètres pour un type de modèle et un régime donné.

        Args:
            model_type (str): Type de modèle (sac, ppo, ddpg).
            regime (str, optional): Régime de marché (trend, range, defensive).
            market (str): Marché (ex. : ES, MNQ).

        Returns:
            Dict: Dictionnaire des hyperparamètres pour le modèle et le régime.

        Raises:
            ValueError: Si le type de modèle ou le régime est invalide.
        """
        start_time = time.time()
        try:
            cache_key = f"{market}_{model_type}_{regime}"
            if cache_key in hyperparam_cache:
                hyperparams = hyperparam_cache[cache_key]
                hyperparam_cache.move_to_end(cache_key)
                return hyperparams
            while len(hyperparam_cache) > MAX_CACHE_SIZE:
                hyperparam_cache.popitem(last=False)

            valid_models = ["sac", "ppo", "ddpg"]
            if model_type.lower() not in valid_models:
                error_msg = f"Type de modèle invalide pour {market}: {model_type}. Modèles valides : {valid_models}"
                logger.error(error_msg)
                AlertManager().send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            valid_regimes = ["trend", "range", "defensive", None]
            if regime and regime.lower() not in valid_regimes:
                error_msg = f"Régime invalide pour {market}: {regime}. Régimes valides : {valid_regimes}"
                logger.error(error_msg)
                AlertManager().send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            model_key = model_type.lower()
            if model_key not in self.hyperparams:
                error_msg = f"Aucun hyperparamètre trouvé pour le modèle {model_key} dans {market}"
                logger.error(error_msg)
                AlertManager().send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            hyperparams = {}
            for key, value in self.hyperparams[model_key].items():
                if isinstance(value, dict) and "value" in value:
                    hyperparams[key] = value
                elif key in ["trend", "range", "defensive"]:
                    continue
                else:
                    hyperparams[key] = {"value": value}

            if regime:
                regime_key = regime.lower()
                if regime_key in self.hyperparams[model_key]:
                    for key, value in self.hyperparams[model_key][regime_key].items():
                        hyperparams[key] = (
                            value
                            if isinstance(value, dict) and "value" in value
                            else {"value": value}
                        )
                    logger.info(
                        f"Hyperparamètres spécifiques au régime {regime_key} appliqués pour {market}."
                    )
                else:
                    warning_msg = f"Aucun hyperparamètre spécifique trouvé pour le régime {regime_key} dans {market}"
                    logger.warning(warning_msg)
                    AlertManager().send_alert(warning_msg, priority=3)
                    send_telegram_alert(warning_msg)

            self._validate_hyperparams(hyperparams, model_type, regime, market=market)
            hyperparam_cache[cache_key] = hyperparams

            df = pd.DataFrame(
                [
                    {
                        "model_type": model_type,
                        "regime": regime or "N/A",
                        "num_params": len(hyperparams),
                    }
                ]
            )
            self.checkpoint(df, data_type="hyperparam_load", market=market)
            self.cloud_backup(df, data_type="hyperparam_load", market=market)

            latency = time.time() - start_time
            self.log_performance(
                "get_hyperparams",
                latency,
                success=True,
                model_type=model_type,
                regime=regime or "N/A",
                market=market,
            )
            return hyperparams

        except Exception as e:
            error_msg = f"Erreur lors de la récupération des hyperparamètres pour {model_type} ({regime or 'N/A'}) dans {market}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            AlertManager().send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance(
                "get_hyperparams",
                time.time() - start_time,
                success=False,
                error=str(e),
                model_type=model_type,
                regime=regime or "N/A",
                market=market,
            )
            raise


def main():
    """
    Exemple d’utilisation pour le débogage.
    """
    try:
        manager = HyperparamManager()
        for market in ["ES", "MNQ"]:
            for model_type, regime in [
                ("sac", "range"),
                ("ppo", "trend"),
                ("ddpg", None),
            ]:
                params = manager.get_hyperparams(model_type, regime, market=market)
                print(
                    f"Hyperparamètres {model_type} ({regime or 'N/A'}) pour {market}: {params}"
                )

    except Exception as e:
        error_msg = f"Échec de la récupération des hyperparamètres : {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        AlertManager().send_alert(error_msg, priority=5)
        send_telegram_alert(error_msg)
        print(error_msg)
        exit(1)


if __name__ == "__main__":
    main()
