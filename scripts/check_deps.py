# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/scripts/check_deps.py
# Vérifie que toutes les dépendances Python sont installées pour MIA_IA_SYSTEM_v2_2025.
#
# Version : 2.1.3
# Mis à jour : 2025-05-13
#
# Rôle : Vérifie l’installation des dépendances Python requises et optionnelles, avec journalisation,
#        snapshots JSON, monitoring des performances, et génération de graphiques pour mia_dashboard.py.
#        Conforme à la Phase 1 (collecte via IQFeed dans es_config.yaml),
#        Phase 8 (auto-conscience via miya_console), et Phase 16 (ensemble learning via torch, tensorflow).
# Dépendances : importlib, logging, os, pkg_resources, yaml, psutil, hashlib, matplotlib.pyplot,
#               pandas, datetime, typing, src.model.utils.miya_console, src.utils.telegram_alert
# Inputs : config/es_config.yaml
# Outputs : Logs dans data/logs/check_deps.log, snapshots dans data/deps_snapshots/,
#           CSV dans data/logs/check_deps_performance.csv, graphiques dans data/figures/dependencies/,
#           JSON dans data/deps_dashboard.json
# Tests unitaires disponibles dans tests/test_check_deps.py.
# Note : Utilisation exclusive d'IQFeed, suppression des références à dxFeed.

import hashlib
import importlib
import logging
import os
import traceback
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import pkg_resources
import psutil
import yaml

from src.model.utils.miya_console import miya_alerts, miya_speak
from src.utils.telegram_alert import send_telegram_alert

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "es_config.yaml")
DASHBOARD_PATH = os.path.join(BASE_DIR, "data", "deps_dashboard.json")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "deps_snapshots")
CSV_LOG_PATH = os.path.join(BASE_DIR, "data", "logs", "check_deps_performance.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures", "dependencies")

# Configuration du logging
os.makedirs(os.path.join(BASE_DIR, "data", "logs"), exist_ok=True)
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "data", "logs", "check_deps.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [CheckDeps] %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Configuration par défaut
DEFAULT_CONFIG = {
    "dependencies": {
        "required": [
            {"name": "pandas", "version": ">=1.5.0"},
            {"name": "requests", "version": ">=2.25.0"},
            {"name": "beautifulsoup4", "version": ">=4.9.0", "import_name": "bs4"},
            {"name": "ib_insync", "version": ">=0.9.70"},
            {"name": "pyyaml", "version": ">=5.4.0", "import_name": "yaml"},
            {"name": "dash", "version": ">=2.0.0"},
            {"name": "plotly", "version": ">=5.0.0"},
            {"name": "gtts", "version": ">=2.2.0"},
            {"name": "sounddevice", "version": ">=0.4.0"},
            {"name": "pyttsx3", "version": ">=2.90"},
            {"name": "loguru", "version": ">=0.5.0"},
            {"name": "pytest", "version": ">=7.0.0"},
            {"name": "ta", "version": ">=0.10.0"},
            {"name": "torch", "version": ">=1.10.0"},
            {"name": "matplotlib", "version": ">=3.5.0"},
            {"name": "seaborn", "version": ">=0.11.0"},
            {"name": "numpy", "version": ">=1.21.0"},
            {"name": "scikit-learn", "version": ">=1.0.0"},
        ],
        "optional": [{"name": "tensorflow", "version": ">=2.8.0"}],
    }
}


class DependencyChecker:
    """
    Classe pour vérifier les dépendances Python avec journalisation et snapshots.
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 1  # Secondes, exponentiel

    def __init__(self):
        """
        Initialise le vérificateur de dépendances.
        """
        self.log_buffer = []
        self.cache = {}
        try:
            self.config = self.load_config()
            self.buffer_size = self.config.get("logging", {}).get("buffer_size", 100)
            self.max_cache_size = self.config.get("cache", {}).get(
                "max_cache_size", 1000
            )
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            os.makedirs(FIGURES_DIR, exist_ok=True)
            miya_speak(
                "DependencyChecker initialisé",
                tag="CHECK_DEPS",
                voice_profile="calm",
                priority=2,
            )
            logger.info("DependencyChecker initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot("init", {"config_path": CONFIG_PATH})
        except Exception as e:
            error_msg = f"Erreur initialisation DependencyChecker: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="CHECK_DEPS", voice_profile="urgent")
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            self.config = DEFAULT_CONFIG["dependencies"]
            self.buffer_size = 100
            self.max_cache_size = 1000

    def load_config(self, config_path: str = CONFIG_PATH) -> Dict:
        """
        Charge la liste des dépendances depuis es_config.yaml avec retries.

        Args:
            config_path (str): Chemin du fichier de configuration.

        Returns:
            Dict: Configuration des dépendances.
        """
        start_time = datetime.now()
        for attempt in range(self.MAX_RETRIES):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                if "dependencies" not in config:
                    raise ValueError(
                        "Clé 'dependencies' manquante dans la configuration"
                    )
                if "required" not in config["dependencies"]:
                    raise ValueError("Clé 'required' manquante dans 'dependencies'")
                latency = (datetime.now() - start_time).total_seconds()
                miya_speak(
                    f"Configuration dépendances chargée: {len(config['dependencies']['required'])} modules requis",
                    tag="CHECK_DEPS",
                    priority=2,
                )
                logger.info("Configuration dépendances chargée")
                self.log_performance("load_config", latency, success=True)
                self.save_snapshot(
                    "load_config",
                    {
                        "config_path": config_path,
                        "num_deps": len(config["dependencies"]["required"]),
                    },
                )
                return config["dependencies"]
            except (FileNotFoundError, yaml.YAMLError) as e:
                if attempt == self.MAX_RETRIES - 1:
                    latency = (datetime.now() - start_time).total_seconds()
                    error_msg = f"Échec chargement {config_path}: {str(e)}\n{traceback.format_exc()}"
                    miya_alerts(
                        error_msg, tag="CHECK_DEPS", voice_profile="urgent", priority=4
                    )
                    send_telegram_alert(error_msg)
                    logger.error(error_msg)
                    self.log_performance(
                        "load_config", latency, success=False, error=str(e)
                    )
                    raise
                time.sleep(self.RETRY_DELAY * (2**attempt))
                logger.warning(
                    f"Échec chargement {config_path}, tentative {attempt + 1}/{self.MAX_RETRIES}"
                )
            except Exception as e:
                latency = (datetime.now() - start_time).total_seconds()
                error_msg = (
                    f"Erreur chargement config: {str(e)}\n{traceback.format_exc()}"
                )
                miya_alerts(
                    error_msg, tag="CHECK_DEPS", voice_profile="urgent", priority=4
                )
                send_telegram_alert(error_msg)
                logger.error(error_msg)
                self.log_performance(
                    "load_config", latency, success=False, error=str(e)
                )
                return DEFAULT_CONFIG["dependencies"]

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ) -> None:
        """
        Journalise les performances des opérations critiques.

        Args:
            operation (str): Nom de l’opération (ex. : check_dependency).
            latency (float): Latence en secondes.
            success (bool): Succès de l’opération.
            error (str, optional): Message d’erreur si échec.
            **kwargs: Paramètres supplémentaires (ex. : num_deps).
        """
        try:
            memory_usage = (
                psutil.Process().memory_info().rss / 1024 / 1024
            )  # Mémoire en Mo
            if memory_usage > 1024:
                miya_alerts(
                    f"ALERT: Utilisation mémoire élevée ({memory_usage:.2f} MB)",
                    tag="CHECK_DEPS",
                    level="error",
                    priority=5,
                )
                send_telegram_alert(
                    f"ALERT: Utilisation mémoire élevée ({memory_usage:.2f} MB)"
                )
            log_entry = {
                "timestamp": str(datetime.now()),
                "operation": operation,
                "latency": latency,
                "success": success,
                "error": error,
                "memory_usage_mb": memory_usage,
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
            miya_alerts(
                f"Erreur journalisation performance: {str(e)}",
                tag="CHECK_DEPS",
                level="error",
                priority=3,
            )
            send_telegram_alert(f"Erreur journalisation performance: {str(e)}")
            logger.error(f"Erreur journalisation performance: {str(e)}")

    def save_snapshot(self, snapshot_type: str, data: Dict) -> None:
        """
        Sauvegarde un instantané des résultats.

        Args:
            snapshot_type (str): Type de snapshot (ex. : init, check_dependencies).
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
            with open(path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=4)
            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                f"Snapshot {snapshot_type} sauvegardé: {path}",
                tag="CHECK_DEPS",
                level="info",
                priority=1,
            )
            logger.info(f"Snapshot {snapshot_type} sauvegardé: {path}")
            self.log_performance("save_snapshot", latency, success=True)
        except Exception as e:
            self.log_performance("save_snapshot", 0, success=False, error=str(e))
            miya_alerts(
                f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}",
                tag="CHECK_DEPS",
                level="error",
                priority=3,
            )
            send_telegram_alert(f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}")
            logger.error(f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}")

    def save_dashboard_status(
        self, status: Dict, status_file: str = DASHBOARD_PATH
    ) -> None:
        """
        Sauvegarde l'état des dépendances pour mia_dashboard.py.

        Args:
            status (Dict): État des dépendances.
            status_file (str): Chemin du fichier JSON.
        """
        try:
            start_time = datetime.now()
            os.makedirs(os.path.dirname(status_file), exist_ok=True)
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4)
            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(f"État sauvegardé dans {status_file}", tag="CHECK_DEPS")
            logger.info(f"État sauvegardé dans {status_file}")
            self.log_performance("save_dashboard_status", latency, success=True)
        except Exception as e:
            self.log_performance(
                "save_dashboard_status", 0, success=False, error=str(e)
            )
            error_msg = (
                f"Erreur sauvegarde dashboard: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="CHECK_DEPS", voice_profile="urgent")
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def plot_dependency_status(
        self, results: List[Dict], output_dir: str = FIGURES_DIR
    ) -> None:
        """
        Génère un graphique pour l’état des dépendances.

        Args:
            results (List[Dict]): Statuts des dépendances.
            output_dir (str): Répertoire pour sauvegarder le graphique.
        """
        start_time = datetime.now()
        try:
            if not results:
                miya_speak(
                    "Aucun résultat à visualiser",
                    tag="CHECK_DEPS",
                    level="warning",
                    priority=2,
                )
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(output_dir, exist_ok=True)

            # Compter les statuts
            installed = len([r for r in results if r["installed"] and not r["error"]])
            missing = len([r for r in results if not r["installed"]])
            version_errors = len([r for r in results if r["installed"] and r["error"]])

            # Graphique en anneau
            labels = ["Installées", "Manquantes", "Versions Incompatibles"]
            sizes = [installed, missing, version_errors]
            colors = ["green", "red", "orange"]
            plt.figure(figsize=(8, 8))
            plt.pie(
                sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140
            )
            plt.title("État des Dépendances")
            centre_circle = plt.Circle((0, 0), 0.70, fc="white")
            fig = plt.gcf()
            fig.gca().add_artist(centre_circle)
            plt.savefig(os.path.join(output_dir, f"dependency_status_{timestamp}.png"))
            plt.close()

            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                f"Graphique des dépendances généré: {output_dir}",
                tag="CHECK_DEPS",
                level="info",
                priority=2,
            )
            logger.info(f"Graphique des dépendances généré: {output_dir}")
            self.log_performance(
                "plot_dependency_status", latency, success=True, num_deps=len(results)
            )
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur génération graphique: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="CHECK_DEPS", level="error", priority=2)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "plot_dependency_status", latency, success=False, error=str(e)
            )

    def check_dependency(self, dep: Dict) -> Dict:
        """
        Vérifie l’état d’une dépendance.

        Args:
            dep (Dict): Détails de la dépendance (nom, version, import_name).

        Returns:
            Dict: Statut de la dépendance (installée, version, erreur).
        """
        start_time = datetime.now()
        try:
            cache_key = hashlib.sha256(
                f"{dep['name']}_{dep['version']}".encode()
            ).hexdigest()
            if cache_key in self.cache:
                result = self.cache[cache_key]
                miya_speak(
                    f"Dépendance {dep['name']} récupérée du cache",
                    tag="CHECK_DEPS",
                    level="info",
                    priority=1,
                )
                self.log_performance(
                    "check_dependency_cache_hit", 0, success=True, dep_name=dep["name"]
                )
                return result

            module_name = dep.get("import_name", dep["name"])
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                result = {
                    "name": dep["name"],
                    "installed": False,
                    "version": None,
                    "required_version": dep["version"],
                    "error": f"Module {dep['name']} non installé",
                }
            else:
                installed_version = pkg_resources.get_distribution(dep["name"]).version
                required_version = dep["version"].lstrip(">=")
                if pkg_resources.parse_version(
                    installed_version
                ) < pkg_resources.parse_version(required_version):
                    result = {
                        "name": dep["name"],
                        "installed": True,
                        "version": installed_version,
                        "required_version": dep["version"],
                        "error": f"Version {installed_version} inférieure à {dep['version']}",
                    }
                else:
                    result = {
                        "name": dep["name"],
                        "installed": True,
                        "version": installed_version,
                        "required_version": dep["version"],
                        "error": None,
                    }

            self.cache[cache_key] = result
            if len(self.cache) > self.max_cache_size:
                self.cache.pop(next(iter(self.cache)))

            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "check_dependency",
                latency,
                success=result["error"] is None,
                dep_name=dep["name"],
            )
            return result
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur vérification dépendance {dep['name']}: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(error_msg, tag="CHECK_DEPS", voice_profile="urgent", priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "check_dependency",
                latency,
                success=False,
                error=str(e),
                dep_name=dep["name"],
            )
            return {
                "name": dep["name"],
                "installed": False,
                "version": None,
                "required_version": dep["version"],
                "error": str(e),
            }

    def check_all_dependencies(self, config: Dict) -> List[Dict]:
        """
        Vérifie toutes les dépendances requises et optionnelles.

        Args:
            config (Dict): Configuration des dépendances.

        Returns:
            List[Dict]: Liste des statuts des dépendances.
        """
        start_time = datetime.now()
        try:
            miya_speak("Vérification des dépendances...", tag="CHECK_DEPS", priority=3)
            logger.info("Vérification des dépendances")
            results = []

            # Vérifier les dépendances requises
            for dep in config.get("required", []):
                result = self.check_dependency(dep)
                results.append(result)
                if result["error"]:
                    miya_alerts(
                        f"Problème dépendance {result['name']}: {result['error']}",
                        tag="CHECK_DEPS",
                        voice_profile="urgent",
                        priority=4,
                    )
                    logger.error(
                        f"Problème dépendance {result['name']}: {result['error']}"
                    )
                else:
                    miya_speak(
                        f"Dépendance {result['name']} OK: version {result['version']}",
                        tag="CHECK_DEPS",
                        priority=2,
                    )
                    logger.info(
                        f"Dépendance {result['name']} OK: version {result['version']}"
                    )

            # Vérifier les dépendances optionnelles
            for dep in config.get("optional", []):
                result = self.check_dependency(dep)
                result["optional"] = True
                results.append(result)
                if result["error"]:
                    miya_speak(
                        f"Dépendance optionnelle {result['name']}: {result['error']}",
                        tag="CHECK_DEPS",
                        level="warning",
                        priority=3,
                    )
                    logger.warning(
                        f"Dépendance optionnelle {result['name']}: {result['error']}"
                    )
                else:
                    miya_speak(
                        f"Dépendance optionnelle {result['name']} OK: version {result['version']}",
                        tag="CHECK_DEPS",
                        priority=2,
                    )
                    logger.info(
                        f"Dépendance optionnelle {result['name']} OK: version {result['version']}"
                    )

            missing = [
                r["name"]
                for r in results
                if not r["installed"] and not r.get("optional", False)
            ]
            if missing:
                install_cmd = f"pip install {' '.join(missing)}"
                miya_alerts(
                    f"Dépendances requises manquantes, installez avec: {install_cmd}",
                    tag="CHECK_DEPS",
                    voice_profile="urgent",
                    priority=5,
                )
                send_telegram_alert(
                    f"Dépendances requises manquantes, installez avec: {install_cmd}"
                )
                logger.warning(f"Dépendances requises manquantes: {missing}")

            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "check_all_dependencies",
                latency,
                success=len(missing) == 0,
                num_deps=len(results),
            )
            self.save_snapshot(
                "check_all_dependencies",
                {"num_deps": len(results), "missing_count": len(missing)},
            )
            self.plot_dependency_status(results)
            self.save_dashboard_status(
                {
                    "last_check": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "dependencies": results,
                    "missing_count": len(missing),
                    "status": "complete" if len(missing) == 0 else "incomplete",
                }
            )
            return results
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"Erreur vérification dépendances: {str(e)}\n{traceback.format_exc()}"
            )
            miya_alerts(error_msg, tag="CHECK_DEPS", voice_profile="urgent", priority=5)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance(
                "check_all_dependencies", latency, success=False, error=str(e)
            )
            return []


def main():
    """
    Point d’entrée principal pour vérifier les dépendances.
    """
    try:
        checker = DependencyChecker()
        config = checker.load_config()
        results = checker.check_all_dependencies(config)
        if all(
            r["installed"] and not r["error"]
            for r in results
            if not r.get("optional", False)
        ):
            miya_speak(
                "Toutes dépendances requises installées correctement",
                tag="CHECK_DEPS",
                priority=3,
            )
            logger.info("Toutes dépendances requises installées correctement")
        else:
            miya_alerts(
                "Certaines dépendances requises nécessitent une action",
                tag="CHECK_DEPS",
                voice_profile="urgent",
                priority=5,
            )
            send_telegram_alert("Certaines dépendances requises nécessitent une action")
            logger.error("Certaines dépendances requises nécessitent une action")
    except Exception as e:
        error_msg = f"Erreur programme: {str(e)}\n{traceback.format_exc()}"
        miya_alerts(error_msg, tag="CHECK_DEPS", voice_profile="urgent", priority=5)
        send_telegram_alert(error_msg)
        logger.error(error_msg)


if __name__ == "__main__":
    main()
