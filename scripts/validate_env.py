# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/scripts/validate_env.py
# Script pour valider l’environnement de développement.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie les versions des dépendances et des outils, génère un snapshot,
#        et envoie des alertes en cas d’erreur.
#        Conforme à la Phase 1 (configuration de l’environnement) et Phase 8 (auto-conscience via alertes).
#
# Dépendances : importlib.metadata, os, json, gzip, src.model.utils.alert_manager,
#               src.model.utils.miya_console, src.utils.telegram_alert
#
# Inputs : Aucun
#
# Outputs : data/devcontainer_snapshots/snapshot_validate_env_*.json.gz
#
# Notes :
# - Valide les versions spécifiques des dépendances.
# - Génère un snapshot JSON compressé pour l’état de l’environnement.

import gzip
import importlib.metadata
import json
import os
from datetime import datetime

from src.model.utils.alert_manager import AlertManager
from src.model.utils.miya_console import miya_alerts, miya_speak
from src.utils.telegram_alert import send_telegram_alert

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "devcontainer_snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


def save_snapshot(status: str, error: str = None) -> None:
    """Sauvegarde un snapshot JSON compressé."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot = {
        "timestamp": timestamp,
        "type": "validate_env",
        "data": {"status": status, "error": error},
    }
    path = os.path.join(SNAPSHOT_DIR, f"snapshot_validate_env_{timestamp}.json.gz")
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=4)
    miya_speak(
        f"Snapshot sauvegardé: {path}", tag="VALIDATE_ENV", level="info", priority=1
    )
    AlertManager().send_alert(f"Snapshot sauvegardé: {path}", priority=1)


def validate_environment() -> bool:
    """Valide les versions des dépendances et des outils."""
    try:
        required = {
            "pytest": "7.3.0",
            "flake8": "6.0.0",
            "pylint": "3.0.0",
            "mypy": "1.0.0",
            "coverage": "7.0.0",
            "pandas": "2.0.0",
            "pyyaml": "6.0.0",
            "psutil": "5.9.8",
            "matplotlib": "3.7.0",
        }
        for pkg, min_version in required.items():
            installed_version = importlib.metadata.version(pkg)
            if installed_version < min_version:
                error_msg = (
                    f"Version de {pkg} ({installed_version}) inférieure à {min_version}"
                )
                miya_alerts(
                    error_msg, tag="VALIDATE_ENV", voice_profile="urgent", priority=5
                )
                AlertManager().send_alert(error_msg, priority=4)
                send_telegram_alert(error_msg)
                save_snapshot("failure", error_msg)
                return False
            miya_speak(
                f"{pkg} version {installed_version} validée",
                tag="VALIDATE_ENV",
                level="info",
                priority=2,
            )
            AlertManager().send_alert(
                f"{pkg} version {installed_version} validée", priority=1
            )

        save_snapshot("success")
        return True
    except Exception as e:
        error_msg = f"Erreur validation environnement: {str(e)}"
        miya_alerts(error_msg, tag="VALIDATE_ENV", voice_profile="urgent", priority=5)
        AlertManager().send_alert(error_msg, priority=4)
        send_telegram_alert(error_msg)
        save_snapshot("failure", error_msg)
        return False


if __name__ == "__main__":
    if validate_environment():
        miya_speak(
            "Environnement validé avec succès",
            tag="VALIDATE_ENV",
            level="info",
            priority=3,
        )
        AlertManager().send_alert("Environnement validé avec succès", priority=1)
        exit(0)
    else:
        miya_alerts(
            "Échec validation environnement",
            tag="VALIDATE_ENV",
            voice_profile="urgent",
            priority=5,
        )
        AlertManager().send_alert("Échec validation environnement", priority=4)
        send_telegram_alert("Échec validation environnement")
        exit(1)
