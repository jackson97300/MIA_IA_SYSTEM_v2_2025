#!/bin/bash
# MIA_IA_SYSTEM_v2_2025/tests/test_run_checks.sh
# Tests unitaires pour run_checks.sh.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie le comportement de run_checks.sh, incluant la vérification des fichiers,
#        l’installation des dépendances, l’exécution des tests, la validation de la conformité,
#        les retries, les logs, et les snapshots JSON.
#        Conforme à la Phase 8 (auto-conscience via alertes) et Phase 15 (automatisation et validation).
#
# Dépendances : bash, run_checks.sh, src/scripts/run_all_tests.py, src/scripts/validate_prompt_compliance.py
#
# Notes :
# - Utilise un répertoire temporaire pour simuler l’environnement.
# - Teste les cas de succès et d’échec.

# Définir le répertoire de test
TEST_DIR=$(mktemp -d)
BASE_DIR="$TEST_DIR/MIA_IA_SYSTEM_v2_2025"
LOG_DIR="$BASE_DIR/data/logs"
LOG_FILE="$LOG_DIR/run_checks.log"
SNAPSHOT_DIR="$BASE_DIR/data/check_snapshots"
CONFIG_PATH="$BASE_DIR/config/es_config.yaml"
REQUIREMENTS_PATH="$BASE_DIR/requirements.txt"
RUN_ALL_TESTS="$BASE_DIR/src/scripts/run_all_tests.py"
VALIDATE_PROMPT="$BASE_DIR/src/scripts/validate_prompt_compliance.py"

# Fonction pour initialiser l’environnement de test
setup_env() {
    mkdir -p "$BASE_DIR/config"
    mkdir -p "$BASE_DIR/src/scripts"
    mkdir -p "$BASE_DIR/data/logs"
    mkdir -p "$BASE_DIR/data/check_snapshots"

    # Créer des fichiers simulés
    echo "pytest:" > "$CONFIG_PATH"
    echo "  verbose: true" >> "$CONFIG_PATH"
    echo "pandas>=2.0.0" > "$REQUIREMENTS_PATH"
    echo "print('Tests simulés')" > "$RUN_ALL_TESTS"
    echo "print('Validation simulée')" > "$VALIDATE_PROMPT"
}

# Fonction pour nettoyer l’environnement
cleanup() {
    rm -rf "$TEST_DIR"
}

# Test 1 : Exécution réussie avec tous les fichiers présents
test_success() {
    setup_env
    cp ../run_checks.sh "$TEST_DIR"
    cd "$TEST_DIR"
    bash run_checks.sh > output.txt 2>&1
    if [ $? -eq 0 ]; then
        echo "Test 1: Exécution réussie - OK"
    else
        echo "Test 1: Exécution réussie - ÉCHEC"
        cat output.txt
        cleanup
        exit 1
    fi
    if grep -q "Toutes les vérifications terminées avec succès" "$LOG_FILE"; then
        echo "Test 1: Log de succès trouvé - OK"
    else
        echo "Test 1: Log de succès non trouvé - ÉCHEC"
        cleanup
        exit 1
    fi
    if [ -d "$SNAPSHOT_DIR" ] && ls "$SNAPSHOT_DIR"/snapshot_*.json.gz >/dev/null 2>&1; then
        echo "Test 1: Snapshots JSON trouvés - OK"
    else
        echo "Test 1: Snapshots JSON non trouvés - ÉCHEC"
        cleanup
        exit 1
    fi
    cleanup
}

# Test 2 : Échec si es_config.yaml est manquant
test_missing_config() {
    setup_env
    rm "$CONFIG_PATH"
    cp ../run_checks.sh "$TEST_DIR"
    cd "$TEST_DIR"
    bash run_checks.sh > output.txt 2>&1
    if [ $? -ne 0 ]; then
        echo "Test 2: Échec attendu (config manquant) - OK"
    else
        echo "Test 2: Échec attendu (config manquant) - ÉCHEC"
        cat output.txt
        cleanup
        exit 1
    fi
    if grep -q "Fichier manquant : $CONFIG_PATH" "$LOG_FILE"; then
        echo "Test 2: Log d’erreur trouvé - OK"
    else
        echo "Test 2: Log d’erreur non trouvé - ÉCHEC"
        cleanup
        exit 1
    fi
    cleanup
}

# Test 3 : Échec si l’installation des dépendances échoue
test_pip_failure() {
    setup_env
    echo "invalid_package" > "$REQUIREMENTS_PATH"
    cp ../run_checks.sh "$TEST_DIR"
    cd "$TEST_DIR"
    bash run_checks.sh > output.txt 2>&1
    if [ $? -ne 0 ]; then
        echo "Test 3: Échec attendu (pip échoue) - OK"
    else
        echo "Test 3: Échec attendu (pip échoue) - ÉCHEC"
        cat output.txt
        cleanup
        exit 1
    fi
    if grep -q "Échec de l’installation des dépendances" "$LOG_FILE"; then
        echo "Test 3: Log d’erreur trouvé - OK"
    else
        echo "Test 3: Log d’erreur non trouvé - ÉCHEC"
        cleanup
        exit 1
    fi
    cleanup
}

# Exécuter tous les tests
test_success
test_missing_config
test_pip_failure

echo "Tous les tests terminés avec succès"
exit 0