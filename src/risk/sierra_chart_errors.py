# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/risk/sierra_chart_errors.py
# Rôle : Gère les erreurs de l’API Teton (méthode 8) (Phase 14).
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, psutil>=5.9.0,<6.0.0, logging, csv, os, datetime, signal, gzip
# - src.model.utils.alert_manager
# - src.utils.telegram_alert
#
# Outputs :
# - Logs dans data/logs/trading/sierra_errors.log
# - Logs d’erreurs dans data/logs/trading/sierra_errors.csv (colonnes : timestamp, error_code, message, trade_id, severity, confidence_drop_rate)
# - Logs de performance dans data/logs/trading/sierra_errors_performance.csv (colonnes : timestamp, operation, latency, success, error, memory_usage_mb, cpu_percent)
# - Snapshots JSON dans data/risk_snapshots/*.json (option *.json.gz)
#
# Notes :
# - Conforme à structure.txt (version 2.1.3, 2025-05-13).
# - Implémente retries (max 3, délai 2^attempt secondes) pour les opérations critiques.
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des alertes critiques, basé sur la fréquence des erreurs critiques.
# - Sauvegarde des snapshots JSON non compressés par défaut avec option de compression gzip.
# - Utilise AlertManager et telegram_alert pour les notifications critiques.
# - Gère les interruptions SIGINT avec sauvegarde des snapshots.
# - Tests unitaires disponibles dans tests/test_sierra_chart_errors.py.

import csv
import gzip
import logging
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import psutil

from src.model.utils.alert_manager import AlertManager
from src.utils.telegram_alert import send_telegram_alert

# Configurer la journalisation
log_dir = Path("data/logs/trading")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=log_dir / "sierra_errors.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Constantes
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Secondes, exponentiel
SNAPSHOT_DIR = Path("data/risk_snapshots")
PERF_LOG_PATH = Path("data/logs/trading/sierra_errors_performance.csv")


class SierraChartErrorManager:
    """
    Classe pour gérer les erreurs de l'API Teton de Sierra Chart.
    """

    def __init__(
        self, log_csv_path: Path = Path("data/logs/trading/sierra_errors.csv")
    ):
        """
        Initialise le gestionnaire d'erreurs.

        Args:
            log_csv_path (Path): Chemin du fichier CSV pour journaliser les erreurs.
        """
        self.alert_manager = AlertManager()
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        PERF_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        signal.signal(signal.SIGINT, self.handle_sigint)

        self.log_csv_path = log_csv_path
        self.critical_error_count = 0
        self.total_error_count = 0
        self._ensure_log_file()
        logger.info("SierraChartErrorManager initialisé avec succès")
        self.alert_manager.send_alert(
            "SierraChartErrorManager initialisé avec succès", priority=2
        )
        send_telegram_alert("SierraChartErrorManager initialisé avec succès")
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
        Enregistre les performances dans sierra_errors_performance.csv.

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
        Sauvegarde un instantané JSON de l’erreur, avec option de compression gzip.

        Args:
            snapshot_type (str): Type de snapshot (ex. : sierra_error).
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

    def _ensure_log_file(self):
        """
        Crée le fichier CSV de journalisation s'il n'existe pas.
        """
        try:

            def create_log():
                self.log_csv_path.parent.mkdir(parents=True, exist_ok=True)
                if not self.log_csv_path.exists():
                    with open(
                        self.log_csv_path, "w", newline="", encoding="utf-8"
                    ) as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                "timestamp",
                                "error_code",
                                "message",
                                "trade_id",
                                "severity",
                                "confidence_drop_rate",
                            ]
                        )
                    logger.info(f"Fichier de log créé: {self.log_csv_path}")

            self.with_retries(create_log)
        except Exception as e:
            logger.error(
                f"Erreur lors de la création du fichier de log {self.log_csv_path}: {str(e)}"
            )
            self.alert_manager.send_alert(
                f"Erreur création fichier log: {str(e)}", priority=3
            )
            send_telegram_alert(f"Erreur création fichier log: {str(e)}")
            raise

    def log_error(
        self,
        error_code: str,
        message: str,
        trade_id: Optional[str] = None,
        severity: str = "INFO",
    ):
        """
        Journalise une erreur de l'API Teton, incluant confidence_drop_rate.

        Args:
            error_code (str): Code d'erreur de l'API (ex. : TETON_1001).
            message (str): Description de l'erreur.
            trade_id (str, optional): Identifiant du trade associé.
            severity (str): Niveau de gravité (INFO, WARNING, CRITICAL).

        Raises:
            ValueError: Si les paramètres sont invalides.
        """
        start_time = time.time()
        try:
            # Valider les paramètres
            if not error_code or not isinstance(error_code, str):
                error_msg = "Code d'erreur invalide"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                raise ValueError("Le code d'erreur doit être une chaîne non vide.")
            if not message or not isinstance(message, str):
                error_msg = "Message d'erreur invalide"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=3)
                send_telegram_alert(error_msg)
                raise ValueError("Le message d'erreur doit être une chaîne non vide.")
            if severity not in ["INFO", "WARNING", "CRITICAL"]:
                logger.warning(
                    f"Gravité invalide: {severity}. Utilisation de INFO par défaut."
                )
                severity = "INFO"

            # Mettre à jour les compteurs
            self.total_error_count += 1
            if severity == "CRITICAL":
                self.critical_error_count += 1

            # Calculer confidence_drop_rate (Phase 8)
            confidence_drop_rate = (
                min(1.0, self.critical_error_count / 10.0)
                if self.total_error_count > 0
                else 0.0
            )
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} (erreurs critiques: {self.critical_error_count})"
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
                logger.warning(alert_msg)

            # Créer l'entrée de log
            timestamp = datetime.now().isoformat()
            log_entry = {
                "timestamp": timestamp,
                "error_code": error_code,
                "message": message,
                "trade_id": trade_id if trade_id else "N/A",
                "severity": severity,
                "confidence_drop_rate": confidence_drop_rate,
            }

            # Journaliser dans le fichier CSV
            def write_csv():
                df = pd.DataFrame([log_entry])
                df.to_csv(
                    self.log_csv_path,
                    mode="a",
                    header=False,
                    index=False,
                    encoding="utf-8",
                )

            self.with_retries(write_csv)
            logger.info(
                f"Erreur journalisée: {error_code} - {message} (trade_id: {trade_id}, severity: {severity}, confidence_drop_rate: {confidence_drop_rate:.2f})"
            )

            # Sauvegarder un snapshot pour les erreurs critiques
            if severity == "CRITICAL":
                self.save_snapshot("sierra_error", log_entry, compress=False)
                self._send_alert(error_code, message, trade_id, confidence_drop_rate)

            self.log_performance(
                "log_error",
                time.time() - start_time,
                success=True,
                error_code=error_code,
                severity=severity,
                confidence_drop_rate=confidence_drop_rate,
            )
        except Exception as e:
            self.log_performance(
                "log_error",
                time.time() - start_time,
                success=False,
                error=str(e),
                error_code=error_code,
            )
            self.alert_manager.send_alert(
                f"Erreur journalisation erreur {error_code}: {str(e)}", priority=3
            )
            send_telegram_alert(f"Erreur journalisation erreur {error_code}: {str(e)}")
            logger.error(f"Erreur journalisation erreur {error_code}: {str(e)}")
            raise

    def _send_alert(
        self,
        error_code: str,
        message: str,
        trade_id: Optional[str],
        confidence_drop_rate: float,
    ):
        """
        Envoie une alerte pour les erreurs critiques via AlertManager et Telegram.

        Args:
            error_code (str): Code d'erreur.
            message (str): Message d'erreur.
            trade_id (str, optional): Identifiant du trade.
            confidence_drop_rate (float): Métrique d’auto-conscience.
        """
        try:
            alert_message = f"Erreur critique API Teton: {error_code} - {message} (trade_id: {trade_id or 'N/A'}, confidence_drop_rate: {confidence_drop_rate:.2f})"
            self.alert_manager.send_alert(alert_message, priority=5)
            send_telegram_alert(alert_message)
            logger.critical(alert_message)
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi de l'alerte: {str(e)}")
            self.alert_manager.send_alert(f"Erreur envoi alerte: {str(e)}", priority=3)
            send_telegram_alert(f"Erreur envoi alerte: {str(e)}")


def main():
    """
    Exemple d'utilisation pour le débogage.
    """
    try:
        error_manager = SierraChartErrorManager()

        # Simuler quelques erreurs
        error_manager.log_error(
            error_code="TETON_1001",
            message="Connexion à l'API Teton refusée",
            trade_id="TRADE_123",
            severity="CRITICAL",
        )
        error_manager.log_error(
            error_code="TETON_2002",
            message="Ordre rejeté: fonds insuffisants",
            trade_id="TRADE_124",
            severity="WARNING",
        )
        error_manager.log_error(
            error_code="TETON_3003",
            message="Délai d'exécution dépassé",
            severity="INFO",
        )

        print(f"Erreurs journalisées avec succès dans {error_manager.log_csv_path}")

    except Exception as e:
        print(f"Échec de la journalisation des erreurs: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
