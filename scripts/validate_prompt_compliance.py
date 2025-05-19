# Placeholder pour validate_prompt_compliance.py
# Rôle : Valide la conformité (absence de obs_t, dxFeed, 320/81 features, présence de retries, logs psutil, alertes) (Phase 10).
# Vérifie fichiers Python/YAML.
# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/scripts/validate_prompt_compliance.py
# Script pour valider la conformité des fichiers Python avec les exigences du prompt de MIA_IA_SYSTEM_v2_2025.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Analyse les fichiers Python du projet pour vérifier la conformité avec les exigences du prompt,
#        incluant version, date, dépendances, retries, logs psutil, snapshots JSON, alertes, Phases, et tests (Phase 15).
#        Conforme à la Phase 8 (auto-conscience via alertes) et Phase 15 (automatisation et validation).
#
# Dépendances : os, logging, pathlib, ast, psutil>=5.9.8, json, yaml>=6.0.0, gzip, datetime, traceback,
#               pandas>=2.0.0, src.model.utils.config_manager, src.model.utils.alert_manager,
#               src.model.utils.miya_console, src.utils.telegram_alert, src.utils.standard
#
# Inputs : Fichiers Python dans src/ et tests/
#
# Outputs : data/logs/validate_prompt_compliance.log,
#           data/logs/validate_prompt_compliance_performance.csv,
#           data/validation_snapshots/snapshot_*.json.gz
#
# Notes :
# - Vérifie la présence de la version 2.1.3, la date 2025-05-13, les dépendances requises,
#   l’utilisation de retries, logs psutil, snapshots JSON, alertes, Phases, et tests.
# - Assure l’absence de références à dxFeed, obs_t, 320/81 features.
# - Tests unitaires dans tests/test_validate_prompt_compliance.py.

import ast
import gzip
import json
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import psutil

from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import config_manager
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.utils.standard import with_retries
from src.utils.telegram_alert import send_telegram_alert

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "es_config.yaml")
LOG_DIR = os.path.join(BASE_DIR, "data", "logs")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "validation_snapshots")
CSV_LOG_PATH = os.path.join(
    BASE_DIR, "data", "logs", "validate_prompt_compliance_performance.csv"
)

# Configuration du logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "validate_prompt_compliance.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)


class PromptComplianceValidator:
    """
    Classe pour valider la conformité des fichiers Python avec les exigences du prompt.
    """

    def __init__(
        self,
        config_path: Path = Path(CONFIG_PATH),
        src_dir: Path = Path(BASE_DIR) / "src",
        tests_dir: Path = Path(BASE_DIR) / "tests",
    ):
        """
        Initialise le validateur de conformité.

        Args:
            config_path (Path): Chemin du fichier de configuration YAML.
            src_dir (Path): Répertoire contenant les fichiers source (src/).
            tests_dir (Path): Répertoire contenant les fichiers de test (tests/).
        """
        self.config_path = config_path
        self.src_dir = src_dir
        self.tests_dir = tests_dir
        self.config = None
        self.log_buffer = []
        self.buffer_size = 100
        try:
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            self.validate_config()
            miya_speak(
                "PromptComplianceValidator initialisé",
                tag="PROMPT_COMPLIANCE",
                voice_profile="calm",
                priority=2,
            )
            AlertManager().send_alert(
                "PromptComplianceValidator initialisé", priority=1
            )
            logger.info("PromptComplianceValidator initialisé")
            self.log_performance("init", 0, success=True)
            self.save_snapshot(
                "init",
                {
                    "config_path": str(self.config_path),
                    "src_dir": str(self.src_dir),
                    "tests_dir": str(self.tests_dir),
                },
            )
        except Exception as e:
            error_msg = f"Erreur initialisation PromptComplianceValidator: {str(e)}\n{traceback.format_exc()}"
            miya_alerts(
                error_msg, tag="PROMPT_COMPLIANCE", voice_profile="urgent", priority=4
            )
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            raise

    def validate_config(self):
        """
        Valide la configuration et les répertoires.

        Raises:
            FileNotFoundError: Si le fichier de configuration ou les répertoires sont introuvables.
        """
        start_time = datetime.now()
        try:
            if not self.config_path.exists():
                error_msg = f"Fichier de configuration introuvable : {self.config_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            if not self.src_dir.exists():
                error_msg = f"Répertoire source introuvable : {self.src_dir}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            if not self.tests_dir.exists():
                error_msg = f"Répertoire de tests introuvable : {self.tests_dir}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            self.config = config_manager.get_config("es_config.yaml")
            latency = (datetime.now() - start_time).total_seconds()
            miya_speak(
                "Configuration validée pour la validation de conformité",
                tag="PROMPT_COMPLIANCE",
                level="info",
                priority=2,
            )
            AlertManager().send_alert(
                "Configuration validée pour la validation de conformité", priority=1
            )
            logger.info("Configuration validée pour la validation de conformité")
            self.log_performance("validate_config", latency, success=True)
            self.save_snapshot(
                "validate_config", {"config_path": str(self.config_path)}
            )
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur lors de la validation de la configuration : {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            miya_alerts(
                error_msg, tag="PROMPT_COMPLIANCE", voice_profile="urgent", priority=4
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
            operation (str): Nom de l’opération (ex. : validate_file).
            latency (float): Latence en secondes.
            success (bool): Succès de l’opération.
            error (str, optional): Message d’erreur si échec.
            **kwargs: Paramètres supplémentaires (ex. : num_files).
        """
        try:
            memory_usage = (
                psutil.Process().memory_info().rss / 1024 / 1024
            )  # Mémoire en Mo
            cpu_percent = psutil.cpu_percent()
            if memory_usage > 1024:
                error_msg = f"ALERT: Utilisation mémoire élevée ({memory_usage:.2f} MB)"
                miya_alerts(
                    error_msg, tag="PROMPT_COMPLIANCE", level="error", priority=5
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
            miya_alerts(error_msg, tag="PROMPT_COMPLIANCE", level="error", priority=3)
            AlertManager().send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)

    def save_snapshot(self, snapshot_type: str, data: Dict) -> None:
        """
        Sauvegarde un instantané des résultats avec compression gzip.

        Args:
            snapshot_type (str): Type de snapshot (ex. : validate_file).
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
                tag="PROMPT_COMPLIANCE",
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
            miya_alerts(error_msg, tag="PROMPT_COMPLIANCE", level="error", priority=3)
            AlertManager().send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            logger.error(error_msg)
            self.log_performance("save_snapshot", 0, success=False, error=str(e))

    def get_python_files(self) -> List[Path]:
        """
        Récupère la liste des fichiers Python dans src/ et tests/.

        Returns:
            List[Path]: Liste des chemins des fichiers Python.
        """
        start_time = datetime.now()
        try:
            python_files = []
            for dir_path in [self.src_dir, self.tests_dir]:
                for root, _, files in os.walk(dir_path):
                    for file in files:
                        if file.endswith(".py"):
                            python_files.append(Path(root) / file)
            latency = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"{len(python_files)} fichiers Python trouvés dans {self.src_dir} et {self.tests_dir}"
            )
            self.log_performance(
                "get_python_files", latency, success=True, num_files=len(python_files)
            )
            self.save_snapshot("get_python_files", {"num_files": len(python_files)})
            return python_files
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur lors de la récupération des fichiers Python : {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            miya_alerts(
                error_msg, tag="PROMPT_COMPLIANCE", voice_profile="urgent", priority=4
            )
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "get_python_files", latency, success=False, error=str(e)
            )
            self.save_snapshot("get_python_files", {"error": str(e)})
            raise

    @with_retries(max_attempts=3, delay_base=2.0)
    def validate_file(self, file_path: Path) -> Dict:
        """
        Valide la conformité d’un fichier Python avec les exigences du prompt.

        Args:
            file_path (Path): Chemin du fichier à analyser.

        Returns:
            Dict: Résultats de la validation (conforme ou non, détails des non-conformités).
        """
        start_time = datetime.now()
        try:
            results = {"file": str(file_path), "compliant": True, "issues": []}

            # Lire le contenu du fichier
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                tree = ast.parse(content)

            # Vérifier la version et la date dans les commentaires
            if not ("Version : 2.1.3" in content and "Date : 2025-05-13" in content):
                results["issues"].append(
                    "Version ou date incorrecte (requis : 2.1.3, 2025-05-13)"
                )
                results["compliant"] = False

            # Vérifier les dépendances
            required_imports = [
                "psutil",
                "config_manager",
                "alert_manager",
                "miya_console",
                "telegram_alert",
                "standard",
            ]
            imports = [
                node.names[0].name
                for node in tree.body
                if isinstance(node, ast.ImportFrom)
                for alias in node.names
            ]
            missing_imports = [imp for imp in required_imports if imp not in imports]
            if missing_imports:
                results["issues"].append(
                    f"Imports manquants : {', '.join(missing_imports)}"
                )
                results["compliant"] = False

            # Vérifier l’utilisation de with_retries
            has_with_retries = any(
                "with_retries" in ast.unparse(node)
                for node in ast.walk(tree)
                if isinstance(node, (ast.Call, ast.Decorator))
            )
            if not has_with_retries:
                results["issues"].append("with_retries non utilisé")
                results["compliant"] = False

            # Vérifier les logs psutil
            has_psutil_logs = any(
                "psutil" in ast.unparse(node)
                and (
                    "cpu_percent" in ast.unparse(node)
                    or "memory_info" in ast.unparse(node)
                )
                for node in ast.walk(tree)
            )
            if not has_psutil_logs:
                results["issues"].append("Logs psutil non détectés")
                results["compliant"] = False

            # Vérifier les snapshots JSON
            has_snapshots = any(
                "gzip" in ast.unparse(node) and "json.dump" in ast.unparse(node)
                for node in ast.walk(tree)
            )
            if not has_snapshots:
                results["issues"].append("Snapshots JSON non détectés")
                results["compliant"] = False

            # Vérifier les alertes
            has_alerts = any(
                "miya_speak" in ast.unparse(node) or "AlertManager" in ast.unparse(node)
                for node in ast.walk(tree)
            )
            if not has_alerts:
                results["issues"].append(
                    "Alertes (miya_speak ou AlertManager) non détectées"
                )
                results["compliant"] = False

            # Vérifier les Phases
            if not any(f"Phase {i}" in content for i in [1, 8, 10, 15, 16]):
                results["issues"].append(
                    "Aucune mention des Phases (1, 8, 10, 15, 16) détectée"
                )
                results["compliant"] = False

            # Vérifier l’absence de références obsolètes
            forbidden_refs = ["dxFeed", "obs_t", "320 features", "81 features"]
            for ref in forbidden_refs:
                if ref in content:
                    results["issues"].append(f"Référence interdite détectée : {ref}")
                    results["compliant"] = False

            # Vérifier la référence aux tests unitaires
            if "Tests unitaires dans tests/test_" not in content:
                results["issues"].append(
                    "Aucune référence aux tests unitaires détectée"
                )
                results["compliant"] = False

            latency = (datetime.now() - start_time).total_seconds()
            if results["compliant"]:
                miya_speak(
                    f"Fichier {file_path} conforme",
                    tag="PROMPT_COMPLIANCE",
                    level="info",
                    priority=2,
                )
                AlertManager().send_alert(f"Fichier {file_path} conforme", priority=1)
                logger.info(f"Fichier {file_path} conforme")
            else:
                miya_alerts(
                    f"Fichier {file_path} non conforme : {', '.join(results['issues'])}",
                    tag="PROMPT_COMPLIANCE",
                    voice_profile="urgent",
                    priority=4,
                )
                AlertManager().send_alert(
                    f"Fichier {file_path} non conforme : {', '.join(results['issues'])}",
                    priority=4,
                )
                send_telegram_alert(
                    f"Fichier {file_path} non conforme : {', '.join(results['issues'])}"
                )
                logger.error(
                    f"Fichier {file_path} non conforme : {', '.join(results['issues'])}"
                )

            self.log_performance(
                "validate_file",
                latency,
                success=results["compliant"],
                issues=len(results["issues"]),
            )
            self.save_snapshot("validate_file", results)
            return results
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur lors de la validation du fichier {file_path} : {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            miya_alerts(
                error_msg, tag="PROMPT_COMPLIANCE", voice_profile="urgent", priority=4
            )
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance("validate_file", latency, success=False, error=str(e))
            self.save_snapshot(
                "validate_file", {"file": str(file_path), "error": str(e)}
            )
            raise

    def validate_all_files(self) -> Dict:
        """
        Valide tous les fichiers Python dans src/ et tests/.

        Returns:
            Dict: Rapport de conformité global.
        """
        start_time = datetime.now()
        try:
            python_files = self.get_python_files()
            report = {
                "total_files": len(python_files),
                "compliant_files": 0,
                "non_compliant_files": [],
                "results": [],
            }

            for file_path in python_files:
                result = self.validate_file(file_path)
                report["results"].append(result)
                if result["compliant"]:
                    report["compliant_files"] += 1
                else:
                    report["non_compliant_files"].append(result)

            report["compliance_rate"] = (
                report["compliant_files"] / report["total_files"]
                if report["total_files"] > 0
                else 0
            )
            latency = (datetime.now() - start_time).total_seconds()

            if report["compliance_rate"] == 1:
                miya_speak(
                    f"Tous les fichiers ({report['total_files']}) sont conformes",
                    tag="PROMPT_COMPLIANCE",
                    level="info",
                    priority=3,
                )
                AlertManager().send_alert(
                    f"Tous les fichiers ({report['total_files']}) sont conformes",
                    priority=1,
                )
                logger.info(
                    f"Tous les fichiers ({report['total_files']}) sont conformes"
                )
            else:
                miya_alerts(
                    f"{len(report['non_compliant_files'])} fichiers non conformes sur {report['total_files']}",
                    tag="PROMPT_COMPLIANCE",
                    voice_profile="urgent",
                    priority=4,
                )
                AlertManager().send_alert(
                    f"{len(report['non_compliant_files'])} fichiers non conformes sur {report['total_files']}",
                    priority=4,
                )
                send_telegram_alert(
                    f"{len(report['non_compliant_files'])} fichiers non conformes sur {report['total_files']}"
                )
                logger.error(
                    f"{len(report['non_compliant_files'])} fichiers non conformes sur {report['total_files']}"
                )

            self.log_performance(
                "validate_all_files",
                latency,
                success=report["compliance_rate"] == 1,
                num_files=report["total_files"],
                non_compliant=len(report["non_compliant_files"]),
            )
            self.save_snapshot("validate_all_files", report)
            print(
                f"Rapport de conformité : {report['compliant_files']} fichiers conformes sur {report['total_files']}"
            )
            return report
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur lors de la validation de tous les fichiers : {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            miya_alerts(
                error_msg, tag="PROMPT_COMPLIANCE", voice_profile="urgent", priority=5
            )
            AlertManager().send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance(
                "validate_all_files", latency, success=False, error=str(e)
            )
            self.save_snapshot("validate_all_files", {"error": str(e)})
            raise


def main():
    """
    Point d'entrée pour valider la conformité des fichiers Python.
    """
    try:
        validator = PromptComplianceValidator()
        report = validator.validate_all_files()
        if report["compliance_rate"] == 1:
            print("Validation terminée : tous les fichiers sont conformes.")
        else:
            print(
                f"Validation terminée : {len(report['non_compliant_files'])} fichiers non conformes."
            )
            for result in report["non_compliant_files"]:
                print(f"- {result['file']} : {', '.join(result['issues'])}")
    except Exception as e:
        error_msg = f"Échec de la validation : {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        miya_alerts(
            error_msg, tag="PROMPT_COMPLIANCE", voice_profile="urgent", priority=5
        )
        AlertManager().send_alert(error_msg, priority=4)
        send_telegram_alert(error_msg)
        logger.error(error_msg)
        exit(1)


if __name__ == "__main__":
    main()
