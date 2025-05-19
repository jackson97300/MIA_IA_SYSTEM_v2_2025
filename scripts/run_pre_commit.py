# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/scripts/run_pre_commit.py
# Script wrapper pour exécuter pre-commit avec journalisation des performances
#
# Version: 2.1.5
# Date: 2025-05-14
#
# Rôle: Exécute pre-commit run --all-files et journalise les métriques psutil (latence, mémoire, CPU)
#       dans data/logs/pre_commit_performance.csv.
#
# Utilisé par: Hook pre-commit (.pre-commit-config.yaml).
#
# Notes:
# - Conforme à structure.txt (version 2.1.5, 2025-05-14).
# - Supporte la suggestion 7 (tests unitaires, couverture 100%).
# - Intègre avec .pre FASSET_commit-config.yaml pour le linting.
# - Pas de références à dxFeed, obs_t, 320 features, ou 81 features.
# - Policies Note: The official directory for routing policies is src/model/router/policies.
# The src/model/policies directory is a residual and should be verified
# for removal to avoid import conflicts.

import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import psutil
from loguru import logger

# Configuration du logging Loguru
logger.remove()
BASE_DIR = Path(os.path.dirname(os.path.dirname(__file__)))
LOG_DIR = BASE_DIR / "data" / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(LOG_DIR / "pre_commit.log", rotation="10 MB", level="INFO", encoding="utf-8")


def run_pre_commit():
    """Exécute pre-commit et journalise les performances."""
    start_time = datetime.now()
    try:
        result = subprocess.run(
            ["pre-commit", "run", "--all-files"],
            capture_output=True,
            text=True,
            check=True,
        )
        success = result.returncode == 0
        stdout = result.stdout
        stderr = result.stderr
        logger.info(f"Pre-commit exécuté avec succès. Sortie: {stdout}")
        if stderr:
            logger.warning(f"Erreurs pre-commit: {stderr}")
    except subprocess.CalledProcessError as e:
        success = False
        stdout = e.stdout
        stderr = e.stderr
        logger.error(f"Erreur pre-commit: {str(e)}. Sortie: {stdout}. Erreur: {stderr}")
    except Exception as e:
        success = False
        stdout = stderr = ""
        logger.error(f"Erreur inattendue pre-commit: {str(e)}")

    # Journalisation des performances
    latency = (datetime.now() - start_time).total_seconds()
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "operation": "pre_commit",
        "latency": latency,
        "success": success,
        "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
        "cpu_usage_percent": psutil.cpu_percent(),
        "stdout": stdout[
            :1000
        ],  # Limiter la taille pour éviter les logs trop volumineux
        "stderr": stderr[:1000],
    }
    log_df = pd.DataFrame([log_entry])
    log_path = LOG_DIR / "pre_commit_performance.csv"
    log_df.to_csv(
        log_path, mode="a", header=not log_path.exists(), index=False, encoding="utf-8"
    )
    logger.info(f"Pre-commit journalisé. Latence: {latency}s, Succès: {success}")

    # Propager l’erreur si pre-commit a échoué
    if not success:
        raise RuntimeError(f"Pre-commit a échoué: {stderr}")


if __name__ == "__main__":
    run_pre_commit()
