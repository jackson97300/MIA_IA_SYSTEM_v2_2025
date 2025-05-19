# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/scripts/setup_env.py
# Script pour automatiser l'installation des dépendances de MIA_IA_SYSTEM_v2_2025.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Exécute pip install -r requirements.txt et valide l'environnement Python (Phase 15).
#        Conforme à la Phase 8 (auto-conscience via alertes) et Phase 15 (automatisation de l’environnement).
#
# Dépendances : logging, subprocess, sys, pkg_resources, pathlib, typing, os, psutil>=5.9.8, json, yaml>=6.0.0, gzip,
#               pandas>=2.0.0, src.model.utils.alert_manager, src.model.utils.miya_console,
#               src.utils.telegram_alert, src.utils.standard
#
# Inputs : requirements.txt
#
# Outputs : data/logs/setup_env.log, data/logs/setup_env_performance.csv,
#           data/setup_snapshots/snapshot_*.json.gz
#
# Notes :
# - Implémente retries (max 3, délai 2^attempt), logs psutil, alertes via alert_manager.py.
# - Snapshots JSON compressés avec gzip pour optimisation.
# - Tests unitaires dans tests/test_setup_env.py.

import gzip
import json
import logging
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pkg_resources
import psutil

from src.model.utils.alert_manager import AlertManager
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.utils.standard import with_retries
from src.utils.telegram_alert import send_telegram_alert

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
REQUIREMENTS_PATH = os.path.join(BASE_DIR, "requirements.txt")
LOG_DIR = os.path.join(BASE_DIR, "data", "logs")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "setup_snapshots")
CSV_LOG_PATH = os.path.join(BASE_DIR, "data", "logs", "setup_env_performance.csv")

# Configuration du logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "setup_env.log"),
    level=logging.INFO,
    format="%(asctime)s,%(levelname)s,%(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Configuration par défaut
DEFAULT_CONFIG = {
    "environment": {
        "min_python_version": "3.8",
        "retry_attempts": 3,
        "retry_delay_base": 2.0,
        "buffer_size": 100,
    }
}


class EnvironmentSetup:
    """
    Classe pour gérer l'installation et la validation de l'environnement Python.
    """

    def __init__(self, requirements_path: Path = Path(REQUIREMENTS_PATH)):
        """
        Initialise le gestionnaire d'environnement.

        Args:
            requirements_path (Path): Chemin du fichier requirements.txt.
        """
        self.requirements_path = requirements_path
        self.min_python_version = (3, 8)
        self.log_buffer = []
        self.buffer_size = DEFAULT_CONFIG["environment"]["buffer_size"]
        try:
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            miya_speak(
                "EnvironmentSetup initialisé",
                tag="SETUP_ENV",
                voice_profile="calm",
                priority=2,
            )
            AlertManager().send_alert("EnvironmentSetup initialisé", priority=1)
            logger.info("EnvironmentSetup initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot(
                "init", {"requirements_path": str(self.requirements_path)}
            )
        except Exception as e:
            error_msg = f"Erreur initialisation EnvironmentSetup: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="SETUP_ENV", voice_profile="urgent", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            raise

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ) -> None:
        """
        Journalise les performances des opérations critiques.

        Args:
            operation (str): Nom de l’opération (ex. : install_requirements).
            latency (float): Latence en secondes.
            success (bool): Succès de l’opération.
            error (str, optional): Message d’erreur si échec.
            **kwargs: Paramètres supplémentaires (ex. : num_packages).
        """
        try:
            memory_usage = (
                psutil.Process().memory_info().rss / 1024 / 1024
            )  # Mémoire en Mo
            cpu_percent = psutil.cpu_percent()
            if memory_usage > 1024:
                error_msg = f"ALERT: Utilisation mémoire élevée ({memory_usage:.2f} MB)"
                miya_alerts(error_msg, tag="SETUP_ENV", level="error", priority=5)
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
            miya_alerts(error_msg, tag="SETUP_ENV", level="error", priority=3)
            AlertManager().send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def save_snapshot(self, snapshot_type: str, data: Dict) -> None:
        """
        Sauvegarde un instantané des résultats avec compression gzip.

        Args:
            snapshot_type (str): Type de snapshot (ex. : install_requirements).
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
                tag="SETUP_ENV",
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
            miya_alerts(error_msg, tag="SETUP_ENV", level="error", priority=3)
            AlertManager().send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("save_snapshot", 0, success=False, error=str(e))

    def check_python_version(self) -> bool:
        """
        Vérifie que la version de Python est compatible.

        Returns:
            bool: True si la version est compatible, False sinon.
        """
        start_time = datetime.now()
        try:
            current_version = sys.version_info[:2]
            if current_version < self.min_python_version:
                error_msg = f"Version de Python {current_version} non supportée. Requis : {self.min_python_version}"
                logger.error(error_msg)
                miya_alerts(
                    error_msg, tag="SETUP_ENV", voice_profile="urgent", priority=4
                )
                AlertManager().send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                latency = (datetime.now() - start_time).total_seconds()
                self.log_performance(
                    "check_python_version", latency, success=False, error=error_msg
                )
                self.save_snapshot(
                    "check_python_version",
                    {"version": current_version, "required": self.min_python_version},
                )
                return False
            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                f"Version de Python vérifiée : {current_version}",
                tag="SETUP_ENV",
                level="info",
                priority=2,
            )
            AlertManager().send_alert(
                f"Version de Python vérifiée : {current_version}", priority=1
            )
            logger.info(f"Version de Python vérifiée : {current_version}")
            self.log_performance(
                "check_python_version",
                latency,
                success=True,
                version=str(current_version),
            )
            self.save_snapshot("check_python_version", {"version": current_version})
            return True
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur lors de la vérification de la version de Python : {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            miya_alerts(error_msg, tag="SETUP_ENV", voice_profile="urgent", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "check_python_version", latency, success=False, error=str(e)
            )
            self.save_snapshot("check_python_version", {"error": str(e)})
            raise

    def validate_requirements_file(self) -> bool:
        """
        Vérifie l’intégrité et la validité du fichier requirements.txt.

        Returns:
            bool: True si le fichier est valide, False sinon.
        """
        start_time = datetime.now()
        try:
            if not self.requirements_path.exists():
                error_msg = (
                    f"Fichier requirements.txt introuvable : {self.requirements_path}"
                )
                logger.error(error_msg)
                miya_alerts(
                    error_msg, tag="SETUP_ENV", voice_profile="urgent", priority=4
                )
                AlertManager().send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                latency = (datetime.now() - start_time).total_seconds()
                self.log_performance(
                    "validate_requirements_file",
                    latency,
                    success=False,
                    error=error_msg,
                )
                self.save_snapshot("validate_requirements_file", {"error": error_msg})
                return False

            with open(self.requirements_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if not lines:
                error_msg = "Fichier requirements.txt vide"
                logger.error(error_msg)
                miya_alerts(
                    error_msg, tag="SETUP_ENV", voice_profile="urgent", priority=4
                )
                AlertManager().send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                latency = (datetime.now() - start_time).total_seconds()
                self.log_performance(
                    "validate_requirements_file",
                    latency,
                    success=False,
                    error=error_msg,
                )
                self.save_snapshot("validate_requirements_file", {"error": error_msg})
                return False

            # Vérifier le format des lignes (simplifié)
            for line in lines:
                line = line.strip()
                if (
                    line
                    and not line.startswith("#")
                    and not any(op in line for op in ["==", ">=", "<=", ">", "<"])
                ):
                    error_msg = f"Ligne mal formée dans requirements.txt : {line}"
                    logger.error(error_msg)
                    miya_alerts(
                        error_msg, tag="SETUP_ENV", voice_profile="urgent", priority=4
                    )
                    AlertManager().send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    latency = (datetime.now() - start_time).total_seconds()
                    self.log_performance(
                        "validate_requirements_file",
                        latency,
                        success=False,
                        error=error_msg,
                    )
                    self.save_snapshot(
                        "validate_requirements_file", {"error": error_msg}
                    )
                    return False

            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                f"Fichier requirements.txt validé : {len(lines)} lignes",
                tag="SETUP_ENV",
                level="info",
                priority=2,
            )
            AlertManager().send_alert(
                f"Fichier requirements.txt validé : {len(lines)} lignes", priority=1
            )
            logger.info(f"Fichier requirements.txt validé : {len(lines)} lignes")
            self.log_performance(
                "validate_requirements_file",
                latency,
                success=True,
                num_lines=len(lines),
            )
            self.save_snapshot("validate_requirements_file", {"num_lines": len(lines)})
            return True
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur lors de la validation de requirements.txt : {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            miya_alerts(error_msg, tag="SETUP_ENV", voice_profile="urgent", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "validate_requirements_file", latency, success=False, error=str(e)
            )
            self.save_snapshot("validate_requirements_file", {"error": str(e)})
            raise

    @with_retries(max_attempts=3, delay_base=2.0)
    def install_requirements(self) -> bool:
        """
        Installe les dépendances depuis requirements.txt.

        Returns:
            bool: True si l'installation réussit, False sinon.
        """
        start_time = datetime.now()
        try:
            if not self.validate_requirements_file():
                error_msg = "Fichier requirements.txt invalide, installation annulée"
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info(
                f"Installation des dépendances depuis : {self.requirements_path}"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(self.requirements_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                error_msg = f"Échec de l'installation des dépendances : {result.stderr}"
                logger.error(error_msg)
                miya_alerts(
                    error_msg, tag="SETUP_ENV", voice_profile="urgent", priority=4
                )
                AlertManager().send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                latency = (datetime.now() - start_time).total_seconds()
                self.log_performance(
                    "install_requirements", latency, success=False, error=error_msg
                )
                self.save_snapshot("install_requirements", {"error": error_msg})
                return False

            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                "Dépendances installées avec succès",
                tag="SETUP_ENV",
                level="info",
                priority=3,
            )
            AlertManager().send_alert("Dépendances installées avec succès", priority=1)
            logger.info("Dépendances installées avec succès.")
            self.log_performance("install_requirements", latency, success=True)
            self.save_snapshot("install_requirements", {"status": "success"})
            return True
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur lors de l'installation des dépendances : {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            miya_alerts(error_msg, tag="SETUP_ENV", voice_profile="urgent", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "install_requirements", latency, success=False, error=str(e)
            )
            self.save_snapshot("install_requirements", {"error": str(e)})
            raise

    def verify_dependencies(self, required_packages: List[Tuple[str, str]]) -> bool:
        """
        Vérifie que les packages requis sont installés avec les versions correctes.

        Args:
            required_packages (List[Tuple[str, str]]): Liste des tuples (nom_package, version_min).

        Returns:
            bool: True si toutes les dépendances sont valides, False sinon.
        """
        start_time = datetime.now()
        try:
            missing_or_invalid = []
            for package, min_version in required_packages:
                try:
                    installed_version = pkg_resources.get_distribution(package).version
                    if pkg_resources.parse_version(
                        installed_version
                    ) < pkg_resources.parse_version(min_version):
                        missing_or_invalid.append(
                            f"{package} (installé: {installed_version}, requis: >= {min_version})"
                        )
                    else:
                        logger.info(f"{package} vérifié : version {installed_version}")
                except pkg_resources.DistributionNotFound:
                    missing_or_invalid.append(
                        f"{package} (non installé, requis: >= {min_version})"
                    )

            if missing_or_invalid:
                error_msg = f"Dépendances manquantes ou invalides : {', '.join(missing_or_invalid)}"
                logger.error(error_msg)
                miya_alerts(
                    error_msg, tag="SETUP_ENV", voice_profile="urgent", priority=4
                )
                AlertManager().send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                latency = (datetime.now() - start_time).total_seconds()
                self.log_performance(
                    "verify_dependencies",
                    latency,
                    success=False,
                    error=error_msg,
                    num_invalid=len(missing_or_invalid),
                )
                self.save_snapshot(
                    "verify_dependencies", {"missing_or_invalid": missing_or_invalid}
                )
                return False

            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                f"Toutes les dépendances vérifiées : {len(required_packages)} packages",
                tag="SETUP_ENV",
                level="info",
                priority=3,
            )
            AlertManager().send_alert(
                f"Toutes les dépendances vérifiées : {len(required_packages)} packages",
                priority=1,
            )
            logger.info(
                f"Toutes les dépendances vérifiées : {len(required_packages)} packages."
            )
            self.log_performance(
                "verify_dependencies",
                latency,
                success=True,
                num_packages=len(required_packages),
            )
            self.save_snapshot(
                "verify_dependencies", {"num_packages": len(required_packages)}
            )
            return True
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur lors de la vérification des dépendances : {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            miya_alerts(error_msg, tag="SETUP_ENV", voice_profile="urgent", priority=4)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "verify_dependencies", latency, success=False, error=str(e)
            )
            self.save_snapshot("verify_dependencies", {"error": str(e)})
            raise

    def setup_environment(self) -> bool:
        """
        Configure l'environnement complet en vérifiant Python et en installant les dépendances.

        Returns:
            bool: True si la configuration réussit, False sinon.
        """
        start_time = datetime.now()
        try:
            # Vérifier la version de Python
            if not self.check_python_version():
                return False

            # Installer les dépendances
            if not self.install_requirements():
                return False

            # Liste des dépendances critiques avec versions minimales
            required_packages = [
                ("pandas", "2.0.0"),
                ("numpy", "1.23.0"),
                ("stable-baselines3", "2.0.0"),
                ("pyiqfeed", "1.0.0"),
                ("pytest", "7.3.0"),
                ("python-dotenv", "1.0.0"),
                ("textblob", "0.17.0"),
                ("pyyaml", "6.0.0"),
                ("dash", "2.17.0"),
                ("schedule", "1.2.0"),
                ("psutil", "5.9.8"),
            ]

            # Vérifier les dépendances installées
            if not self.verify_dependencies(required_packages):
                return False

            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                "Configuration de l'environnement terminée avec succès",
                tag="SETUP_ENV",
                voice_profile="calm",
                priority=3,
            )
            AlertManager().send_alert(
                "Configuration de l'environnement terminée avec succès", priority=1
            )
            logger.info("Configuration de l'environnement terminée avec succès.")
            self.log_performance(
                "setup_environment",
                latency,
                success=True,
                num_packages=len(required_packages),
            )
            self.save_snapshot(
                "setup_environment",
                {"num_packages": len(required_packages), "status": "success"},
            )
            return True
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur lors de la configuration de l'environnement : {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            miya_alerts(error_msg, tag="SETUP_ENV", voice_profile="urgent", priority=5)
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "setup_environment", latency, success=False, error=str(e)
            )
            self.save_snapshot("setup_environment", {"error": str(e)})
            raise


def main():
    """
    Point d'entrée pour configurer l'environnement.
    """
    try:
        setup = EnvironmentSetup()
        if setup.setup_environment():
            print(
                "Environnement configuré avec succès. Prêt pour MIA_IA_SYSTEM_v2_2025."
            )
            miya_speak(
                "Environnement configuré avec succès",
                tag="SETUP_ENV",
                voice_profile="calm",
                priority=3,
            )
            AlertManager().send_alert("Environnement configuré avec succès", priority=1)
        else:
            print(
                "Échec de la configuration de l'environnement. Consultez data/logs/setup_env.log."
            )
            miya_alerts(
                "Échec de la configuration de l'environnement",
                tag="SETUP_ENV",
                voice_profile="urgent",
                priority=5,
            )
            AlertManager().send_alert(
                "Échec de la configuration de l'environnement", priority=4
            )
            send_telegram_alert("Échec de la configuration de l'environnement")
            sys.exit(1)
    except Exception as e:
        error_msg = f"Échec de la configuration : {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        miya_alerts(error_msg, tag="SETUP_ENV", voice_profile="urgent", priority=5)
        AlertManager().send_alert(error_msg, priority=4)
        send_telegram_alert(error_msg)
        logger.error(error_msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
