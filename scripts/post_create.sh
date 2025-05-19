#!/bin/bash
# MIA_IA_SYSTEM_v2_2025/scripts/post_create.sh
# Script pour installer les dépendances avec retries et générer un snapshot.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Installe les dépendances Python et valide l’environnement avec retries et alertes.
#        Conforme à la Phase 1 (configuration de l’environnement).
#
# Dépendances : bash, pip, python3, requirements.txt, validate_env.py
#
# Inputs : requirements.txt, scripts/validate_env.py
#
# Outputs : data/devcontainer_snapshots/snapshot_post_create_*.json.gz
#
# Notes :
# - Implémente retries (max 3, délai exponentiel).
# - Génère un snapshot JSON compressé pour l’état de l’installation.

set -e

MAX_ATTEMPTS=3
BASE_DIR=$(dirname $(dirname $(realpath $0)))
SNAPSHOT_DIR="$BASE_DIR/data/devcontainer_snapshots"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Créer le répertoire des snapshots
mkdir -p "$SNAPSHOT_DIR"

# Fonction pour générer un snapshot
save_snapshot() {
    SNAPSHOT_PATH="$SNAPSHOT_DIR/snapshot_post_create_$TIMESTAMP.json.gz"
    echo "{\"timestamp\": \"$TIMESTAMP\", \"type\": \"post_create\", \"data\": {\"status\": \"$1\", \"error\": \"$2\"}}" | gzip > "$SNAPSHOT_PATH"
    echo "Snapshot sauvegardé: $SNAPSHOT_PATH"
}

# Installation avec retries
for ((attempt=1; attempt<=MAX_ATTEMPTS; attempt++)); do
    echo "Tentative $attempt/$MAX_ATTEMPTS d’installation des dépendances..."
    if pip install -r requirements.txt && pip install pytest>=7.3.0,<8.0.0 flake8>=6.0.0 pylint>=3.0.0 mypy>=1.0.0 coverage>=7.0.0; then
        echo "Installation réussie"
        save_snapshot "success" ""
        break
    else
        ERROR_MSG="Échec installation, tentative $attempt"
        echo "$ERROR_MSG"
        save_snapshot "failure" "$ERROR_MSG"
        if [ $attempt -lt $MAX_ATTEMPTS ]; then
            DELAY=$((2**attempt))
            echo "Nouvelle tentative dans $DELAY secondes..."
            sleep $DELAY
        else
            echo "Échec après $MAX_ATTEMPTS tentatives"
            exit 1
        fi
    fi
done

# Validation de l’environnement
echo "Validation de l’environnement..."
python3 scripts/validate_env.py
if [ $? -eq 0 ]; then
    echo "Environnement validé avec succès"
    save_snapshot "validation_success" ""
else
    ERROR_MSG="Échec validation environnement"
    echo "$ERROR_MSG"
    save_snapshot "validation_failure" "$ERROR_MSG"
    exit 1
fi