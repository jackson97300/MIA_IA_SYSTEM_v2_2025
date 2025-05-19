# scripts/run_checks.sh
pip install -r requirements.txt
pytest tests/ --cov=src --cov-report=xml
flake8 src/
pylint src/
mypy src/
#!/bin/bash
# MIA_IA_SYSTEM_v2_2025/run_checks.sh
# Script pour exécuter les vérifications critiques de MIA_IA_SYSTEM_v2_2025.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Orchestre l'exécution des vérifications, incluant l'installation des dépendances,
#        les tests unitaires et d'intégration (via run_all_tests.py),
#        et la validation de la conformité des fichiers (via validate_prompt_compliance.py) (Phase 15).
#        Conforme à la Phase 8 (auto-conscience via alertes) et Phase 15 (automatisation et validation).
#
# Dépendances : bash, python3 (>=3.8), src/scripts/run_all_tests.py, src/scripts/validate_prompt_compliance.py,
#               config/es_config.yaml, requirements.txt
#
# Inputs : config/es_config.yaml, requirements.txt, tests/
#
# Outputs : data/logs/run_checks.log, data/check_snapshots/snapshot_*.json.gz
#
# Notes :
# - Implémente des retries simples pour les commandes critiques.
# - S’appuie sur des scripts Python pour les snapshots JSON et les alertes.
# - Tests unitaires dans tests/test_run_checks.sh.
# - Compatible avec Linux/macOS et Windows (via Git Bash ou WSL).

# Définir le répertoire de base
BASE_DIR="$(dirname "$(dirname "$(realpath "$0")")")"
LOG_DIR="$BASE_DIR/data/logs"
LOG_FILE="$LOG_DIR/run_checks.log"
SNAPSHOT_DIR="$BASE_DIR/data/check_snapshots"
CONFIG_PATH="$BASE_DIR/config/es_config.yaml"
REQUIREMENTS_PATH="$BASE_DIR/requirements.txt"
RUN_ALL_TESTS="$BASE_DIR/src/scripts/run_all_tests.py"
VALIDATE_PROMPT="$BASE_DIR/src/scripts/validate_prompt_compliance.py"

# Créer le répertoire des logs
mkdir -p "$LOG_DIR"
mkdir -p "$SNAPSHOT_DIR"

# Fonction pour journaliser
log() {
    local level="$1"
    local message="$2"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $level - $message" >> "$LOG_FILE"
    echo "$level: $message"
}

# Fonction pour exécuter une commande avec retries
run_with_retries() {
    local cmd="$1"
    local max_attempts=3
    local attempt=1
    local delay_base=2

    while [ $attempt -le $max_attempts ]; do
        log "INFO" "Exécution de '$cmd' (tentative $attempt/$max_attempts)"
        if $cmd; then
            log "INFO" "Commande '$cmd' exécutée avec succès"
            return 0
        else
            log "ERROR" "Échec de '$cmd' (tentative $attempt/$max_attempts)"
            if [ $attempt -eq $max_attempts ]; then
                log "ERROR" "Échec définitif après $max_attempts tentatives"
                return 1
            fi
            sleep $((delay_base ** attempt))
            attempt=$((attempt + 1))
        fi
    done
}

# Fonction pour envoyer une alerte via Python
send_alert() {
    local message="$1"
    local priority="$2"
    local level="$3"
    local tag="RUN_CHECKS"
    python3 - <<EOF
from src.model.utils.alert_manager import AlertManager
from src.model.utils.miya_console import miya_speak, miya_alerts
from src.utils.telegram_alert import send_telegram_alert
message = "$message"
priority = $priority
level = "$level"
tag = "$tag"
if level == "error":
    miya_alerts(message, tag=tag, voice_profile="urgent", priority=priority)
else:
    miya_speak(message, tag=tag, level=level, priority=priority)
AlertManager().send_alert(message, priority=priority)
send_telegram_alert(message)
EOF
}

# Fonction pour sauvegarder un snapshot JSON via Python
save_snapshot() {
    local snapshot_type="$1"
    local status="$2"
    local details="$3"
    python3 - <<EOF
import json
import gzip
from datetime import datetime
import os
snapshot_type = "$snapshot_type"
status = "$status"
details = "$details"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
snapshot = {
    "timestamp": timestamp,
    "type": snapshot_type,
    "data": {"status": status, "details": details}
}
path = os.path.join("$SNAPSHOT_DIR", f"snapshot_{snapshot_type}_{timestamp}.json")
os.makedirs("$SNAPSHOT_DIR", exist_ok=True)
with gzip.open(f"{path}.gz", "wt", encoding="utf-8") as f:
    json.dump(snapshot, f, indent=4)
EOF
}

# Vérifier la présence des fichiers critiques
log "INFO" "Vérification des fichiers critiques..."
for file in "$CONFIG_PATH" "$REQUIREMENTS_PATH" "$RUN_ALL_TESTS" "$VALIDATE_PROMPT"; do
    if [ ! -f "$file" ]; then
        log "ERROR" "Fichier manquant : $file"
        send_alert "Fichier manquant : $file" 4 "error"
        save_snapshot "check_files" "failed" "Fichier manquant : $file"
        exit 1
    fi
done
log "INFO" "Tous les fichiers critiques sont présents"
send_alert "Tous les fichiers critiques sont présents" 1 "info"
save_snapshot "check_files" "success" "Tous les fichiers critiques présents"

# Vérifier et installer les dépendances
log "INFO" "Installation des dépendances..."
cmd="python3 -m pip install -r \"$REQUIREMENTS_PATH\""
if ! run_with_retries "$cmd"; then
    log "ERROR" "Échec de l’installation des dépendances"
    send_alert "Échec de l’installation des dépendances" 4 "error"
    save_snapshot "install_dependencies" "failed" "Échec de l’installation des dépendances"
    exit 1
fi
log "INFO" "Dépendances installées avec succès"
send_alert "Dépendances installées avec succès" 1 "info"
save_snapshot "install_dependencies" "success" "Dépendances installées"

# Exécuter les tests via run_all_tests.py
log "INFO" "Exécution des tests via run_all_tests.py..."
cmd="python3 \"$RUN_ALL_TESTS\""
if ! run_with_retries "$cmd"; then
    log "ERROR" "Échec de l’exécution des tests"
    send_alert "Échec de l’exécution des tests" 4 "error"
    save_snapshot "run_tests" "failed" "Échec de l’exécution des tests"
    exit 1
fi
log "INFO" "Tests exécutés avec succès"
send_alert "Tests exécutés avec succès" 1 "info"
save_snapshot "run_tests" "success" "Tests exécutés"

# Valider la conformité des fichiers via validate_prompt_compliance.py
log "INFO" "Validation de la conformité des fichiers via validate_prompt_compliance.py..."
cmd="python3 \"$VALIDATE_PROMPT\""
if ! run_with_retries "$cmd"; then
    log "ERROR" "Échec de la validation de la conformité des fichiers"
    send_alert "Échec de la validation de la conformité des fichiers" 4 "error"
    save_snapshot "validate_compliance" "failed" "Échec de la validation de la conformité"
    exit 1
fi
log "INFO" "Validation de la conformité terminée avec succès"
send_alert "Validation de la conformité terminée avec succès" 1 "info"
save_snapshot "validate_compliance" "success" "Validation de la conformité terminée"

# Rapport final
log "INFO" "Toutes les vérifications terminées avec succès"
send_alert "Toutes les vérifications terminées avec succès" 1 "info"
save_snapshot "run_checks" "success" "Toutes les vérifications terminées"
echo "Toutes les vérifications terminées avec succès"
exit 0