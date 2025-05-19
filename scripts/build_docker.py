# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/scripts/build_docker.py
# Script pour construire l’image Docker avec journalisation des performances
#
# Version: 2.1.5
# Date: 2025-05-14
#
# Rôle: Exécute docker build et journalise les métriques psutil (latence, mémoire, CPU)
#       dans data/logs/docker_build_performance.csv.
#
# Utilisé par: Processus de construction Docker, CI/CD.
#
# Notes:
# - Conforme à structure.txt (version 2.1.5, 2025-05-14).
# - Supporte les suggestions 1 (position sizing dynamique), 4 (HMM/changepoint detection),
#   7 (Safe RL/CVaR-PPO), 8 (Distributional RL/QR-DQN), 9 (réentraînement).
# - Intègre avec Dockerfile pour construire l’image mia-ia-system:latest.
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
logger.add(
    LOG_DIR / "docker_build.log", rotation="10 MB", level="INFO", encoding="utf-8"
)


def build_docker():
    """Exécute docker build et journalise les performances."""
    start_time = datetime.now()
    try:
        result = subprocess.run(
            ["docker", "build", "-t", "mia-ia-system:latest", "."],
            capture_output=True,
            text=True,
            check=True,
        )
        success = result.returncode == 0
        stdout = result.stdout
        stderr = result.stderr
        logger.info(f"Docker build exécuté avec succès. Sortie: {stdout}")
        if stderr:
            logger.warning(f"Erreurs docker build: {stderr}")
    except subprocess.CalledProcessError as e:
        success = False
        stdout = e.stdout
        stderr = e.stderr
        logger.error(
            f"Erreur docker build: {str(e)}. Sortie: {stdout}. Erreur: {stderr}"
        )
    except Exception as e:
        success = False
        stdout = stderr = ""
        logger.error(f"Erreur inattendue docker build: {str(e)}")

    # Journalisation des performances
    latency = (datetime.now() - start_time).total_seconds()
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "operation": "docker_build",
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
    log_path = LOG_DIR / "docker_build_performance.csv"
    log_df.to_csv(
        log_path, mode="a", header=not log_path.exists(), index=False, encoding="utf-8"
    )
    logger.info(f"Docker build journalisé. Latence: {latency}s, Succès: {success}")

    # Propager l’erreur si le build a échoué
    if not success:
        raise RuntimeError(f"Docker build a échoué: {stderr}")


if __name__ == "__main__":
    build_docker()
