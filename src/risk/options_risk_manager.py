# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/risk/options_risk_manager.py
# Rôle : Gère les risques spécifiques aux options (méthode 17) (Phase 15).
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.0,<6.0.0, logging, os, signal, gzip
# - src.model.utils.alert_manager
# - src.utils.telegram_alert
# - src.trading.shap_weighting
#
# Inputs :
# - Données de trading (pd.DataFrame avec colonnes : timestamp, strike, option_type, implied_volatility, gamma, delta, position_size)
#
# Outputs :
# - Logs dans data/logs/trading/options_risk_performance.csv (colonnes : timestamp, operation, latency, success, error, memory_usage_mb, cpu_percent)
# - Snapshots JSON dans data/risk_snapshots/*.json (option *.json.gz)
#
# Notes :
# - Conforme à structure.txt (version 2.1.3, 2025-05-13).
# - Implémente retries (max 3, délai 2^attempt secondes) pour les opérations critiques.
# - Intègre confidence_drop_rate (Phase 8) pour l’auto-conscience, basé sur risk_alert.
# - Intègre l’analyse SHAP (Phase 17) pour évaluer l’impact des features sur les métriques de risque, limitée à 50 features.
# - Sauvegarde des snapshots JSON non compressés par défaut avec option de compression gzip.
# - Utilise AlertManager et telegram_alert pour les notifications critiques.
# - Gère les interruptions SIGINT avec sauvegarde des snapshots.
# - Tests unitaires disponibles dans tests/test_options_risk_manager.py.

import gzip
import logging
import signal
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import psutil

from src.model.utils.alert_manager import AlertManager
from src.trading.shap_weighting import calculate_shap
from src.utils.telegram_alert import send_telegram_alert

# Configurer la journalisation
log_dir = Path("data/logs/trading")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=log_dir / "options_risk.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Constantes
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Secondes, exponentiel
SNAPSHOT_DIR = Path("data/risk_snapshots")
PERF_LOG_PATH = Path("data/logs/trading/options_risk_performance.csv")
SHAP_FEATURE_LIMIT = 50


class OptionsRiskManager:
    """
    Classe pour gérer les risques liés aux positions en options.
    """

    def __init__(self, risk_thresholds: Dict[str, float] = None):
        """
        Initialise le gestionnaire de risques pour les options.

        Args:
            risk_thresholds (Dict[str, float], optional): Seuils pour les métriques de risque.
        """
        self.alert_manager = AlertManager()
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        PERF_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        signal.signal(signal.SIGINT, self.handle_sigint)

        self.risk_thresholds = risk_thresholds or {
            "gamma_exposure": 1000.0,  # Seuil pour l'exposition au gamma
            "iv_sensitivity": 0.5,  # Seuil pour la sensibilité à la volatilité implicite
            "min_confidence": 0.7,  # Seuil pour confidence_drop_rate (Phase 8)
        }
        self.log_path = PERF_LOG_PATH
        logger.info("OptionsRiskManager initialisé avec succès")
        self.alert_manager.send_alert(
            "OptionsRiskManager initialisé avec succès", priority=2
        )
        send_telegram_alert("OptionsRiskManager initialisé avec succès")
        self.log_performance("init", 0, success=True)

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
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ) -> None:
        """
        Enregistre les performances dans options_risk_performance.csv.

        Args:
            operation (str): Nom de l’opération.
            latency (float): Latence en secondes.
            success (bool): Succès de l’opération.
            error (str, optional): Message d’erreur si échec.
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
            log_df = pd.DataFrame([log_entry])

            def write_log():
                if not self.log_path.exists():
                    log_df.to_csv(self.log_path, index=False, encoding="utf-8")
                else:
                    log_df.to_csv(
                        self.log_path,
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
        Sauvegarde un instantané JSON des métriques de risque.

        Args:
            snapshot_type (str): Type de snapshot (ex. : options_risk).
            data (Dict): Données à sauvegarder.
            compress (bool): Si True, compresse en gzip.
        """
        try:
            start_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = {"timestamp": timestamp, "type": snapshot_type, "data": data}
            snapshot_path = SNAPSHOT_DIR / f"snapshot_{snapshot_type}_{timestamp}.json"

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

    def calculate_shap_risk(
        self, data: pd.DataFrame, target: str = "gamma_exposure"
    ) -> Optional[pd.DataFrame]:
        """
        Calcule les valeurs SHAP pour les métriques de risque (Phase 17).

        Args:
            data (pd.DataFrame): Données d’entrée avec les features.
            target (str): Métrique cible pour SHAP (ex. : gamma_exposure, iv_sensitivity).

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

    def calculate_options_risk(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calcule les métriques de risque pour les positions en options, incluant confidence_drop_rate et SHAP.

        Args:
            data (pd.DataFrame): Données contenant les colonnes nécessaires :
                - timestamp: Horodatage
                - strike: Prix d'exercice
                - option_type: Type d'option (call ou put)
                - implied_volatility: Volatilité implicite
                - gamma: Sensibilité gamma
                - delta: Sensibilité delta
                - position_size: Taille de la position (positive pour long, négative pour short)

        Returns:
            Dict[str, pd.Series]: Dictionnaire contenant les métriques de risque :
                - gamma_exposure: Exposition nette au gamma
                - iv_sensitivity: Sensibilité à la volatilité implicite
                - risk_alert: Indicateur de dépassement des seuils (0 ou 1)
                - confidence_drop_rate: Métrique d’auto-conscience (Phase 8)
                - shap_metrics: Valeurs SHAP pour gamma_exposure (Phase 17)

        Raises:
            ValueError: Si les colonnes requises sont manquantes ou si les données sont invalides.
        """
        start_time = time.time()
        try:
            # Vérifier les colonnes requises
            required_columns = [
                "timestamp",
                "strike",
                "option_type",
                "implied_volatility",
                "gamma",
                "delta",
                "position_size",
            ]
            missing_columns = [
                col for col in required_columns if col not in data.columns
            ]
            if missing_columns:
                error_msg = f"Colonnes manquantes dans les données : {missing_columns}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            # Vérifier les données non nulles
            if data[required_columns].isnull().any().any():
                logger.warning(
                    "Valeurs NaN détectées dans les données. Remplacement par 0."
                )
                self.alert_manager.send_alert(
                    "Valeurs NaN détectées, remplacement par 0", priority=2
                )
                send_telegram_alert("Valeurs NaN détectées, remplacement par 0")
                data = data.fillna(0)

            # Validation des types de données
            if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
                data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
                if data["timestamp"].isna().any():
                    error_msg = "Timestamps invalides dans les données"
                    logger.error(error_msg)
                    self.alert_manager.send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    raise ValueError(error_msg)
            if not all(data["option_type"].isin(["call", "put"])):
                error_msg = "Valeurs option_type invalides (doit être 'call' ou 'put')"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            # Calculer l'exposition nette au gamma
            data["gamma_contribution"] = data["gamma"] * data["position_size"]
            gamma_exposure = data.groupby("timestamp")["gamma_contribution"].sum()
            gamma_exposure.name = "gamma_exposure"
            logger.info("Calcul de l'exposition au gamma terminé.")

            # Calculer la sensibilité à la volatilité implicite
            data["delta_abs_contribution"] = abs(data["delta"] * data["position_size"])
            iv_sensitivity = data.groupby("timestamp").apply(
                lambda x: x["delta_abs_contribution"].sum()
                * x["implied_volatility"].mean()
            )
            iv_sensitivity.name = "iv_sensitivity"
            iv_sensitivity = iv_sensitivity.fillna(0)
            logger.info("Calcul de la sensibilité à la volatilité implicite terminé.")

            # Détecter les dépassements des seuils de risque
            risk_alert = (
                (gamma_exposure.abs() > self.risk_thresholds["gamma_exposure"])
                | (iv_sensitivity > self.risk_thresholds["iv_sensitivity"])
            ).astype(int)
            risk_alert.name = "risk_alert"
            if risk_alert.any():
                alert_msg = "Dépassement des seuils de risque détecté"
                self.alert_manager.send_alert(alert_msg, priority=4)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)
            logger.info("Calcul des alertes de risque terminé.")

            # Calculer confidence_drop_rate (Phase 8)
            confidence_drop_rate = (
                risk_alert * (1.0 - self.risk_thresholds["min_confidence"])
            ).rename("confidence_drop_rate")
            if confidence_drop_rate.mean() > 0.5:
                alert_msg = "Confidence_drop_rate élevé détecté"
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            # Analyse SHAP (Phase 17)
            shap_metrics = {}
            shap_values = self.calculate_shap_risk(data, target="gamma_exposure")
            if shap_values is not None:
                shap_metrics = {
                    f"shap_{col}": shap_values[col] for col in shap_values.columns
                }
            else:
                logger.warning("SHAP non calculé, métriques vides")
                self.alert_manager.send_alert(
                    "SHAP non calculé, métriques vides", priority=3
                )
                send_telegram_alert("SHAP non calculé, métriques vides")

            # Retourner les métriques
            metrics = {
                "gamma_exposure": gamma_exposure,
                "iv_sensitivity": iv_sensitivity,
                "risk_alert": risk_alert,
                "confidence_drop_rate": confidence_drop_rate,
                "shap_metrics": shap_metrics,
            }
            logger.info("Métriques de risque des options générées avec succès.")
            self.log_performance(
                "calculate_options_risk",
                time.time() - start_time,
                success=True,
                num_rows=len(data),
            )

            # Sauvegarder un snapshot
            self.save_snapshot(
                "options_risk",
                {
                    "gamma_exposure": gamma_exposure.to_dict(),
                    "iv_sensitivity": iv_sensitivity.to_dict(),
                    "risk_alert": risk_alert.to_dict(),
                    "confidence_drop_rate": confidence_drop_rate.to_dict(),
                    "shap_metrics": {k: v.to_dict() for k, v in shap_metrics.items()},
                },
                compress=False,
            )

            return metrics

        except Exception as e:
            self.log_performance(
                "calculate_options_risk",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            self.alert_manager.send_alert(
                f"Erreur calcul métriques de risque: {str(e)}", priority=4
            )
            send_telegram_alert(f"Erreur calcul métriques de risque: {str(e)}")
            logger.error(f"Erreur calcul métriques de risque: {str(e)}")
            raise

    def save_risk_metrics(
        self, metrics: Dict[str, pd.Series], output_path: Optional[Path] = None
    ):
        """
        Sauvegarde les métriques de risque dans un fichier CSV.

        Args:
            metrics (Dict[str, pd.Series]): Métriques calculées.
            output_path (Path, optional): Chemin du fichier CSV de sortie.
        """
        start_time = time.time()
        try:
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df = pd.DataFrame(
                    {k: v for k, v in metrics.items() if k not in ["shap_metrics"]}
                )

                def write_csv():
                    df.to_csv(output_path, index=True, encoding="utf-8")

                self.with_retries(write_csv)
                logger.info(f"Métriques sauvegardées à : {output_path}")
                self.alert_manager.send_alert(
                    f"Métriques sauvegardées à : {output_path}", priority=1
                )
                send_telegram_alert(f"Métriques sauvegardées à : {output_path}")
            self.log_performance(
                "save_risk_metrics", time.time() - start_time, success=True
            )
        except Exception as e:
            self.log_performance(
                "save_risk_metrics",
                time.time() - start_time,
                success=False,
                error=str(e),
            )
            self.alert_manager.send_alert(
                f"Erreur sauvegarde métriques: {str(e)}", priority=3
            )
            send_telegram_alert(f"Erreur sauvegarde métriques: {str(e)}")
            logger.error(f"Erreur sauvegarde métriques: {str(e)}")
            raise


def main():
    """
    Exemple d'utilisation pour le débogage.
    """
    try:
        # Créer des données factices pour tester
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start="2025-05-13 10:00:00", periods=10, freq="H"
                ),
                "strike": [5100.0] * 10,
                "option_type": ["call"] * 5 + ["put"] * 5,
                "implied_volatility": np.random.uniform(0.1, 0.3, 10),
                "gamma": np.random.uniform(0.01, 0.05, 10),
                "delta": np.random.uniform(-0.5, 0.5, 10),
                "position_size": np.random.randint(-100, 100, 10),
            }
        )

        # Initialiser le gestionnaire de risques
        risk_manager = OptionsRiskManager()

        # Calculer les métriques de risque
        metrics = risk_manager.calculate_options_risk(data)

        # Sauvegarder les métriques
        output_path = Path("data/risk_snapshots/options_risk_metrics.csv")
        risk_manager.save_risk_metrics(metrics, output_path)

        print("Métriques de risque des options calculées et sauvegardées avec succès.")

    except Exception as e:
        print(f"Échec du calcul des métriques: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
