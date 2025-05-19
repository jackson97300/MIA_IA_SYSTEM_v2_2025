# Placeholder pour test_build.sh
# Rôle : Teste build.sh (Phase 16).
# test_build()
#!/bin/bash
# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_build.sh
#
# Tests unitaires pour le script scripts/build.sh.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie que build.sh s’exécute correctement, crée les fichiers de log attendus,
#        enregistre les métriques de performance via psutil, et envoie des alertes via alert_manager.py.
#        Conforme à la Phase 16 (ensemble et transfer learning).
#
# Dépendances :
# - bash
# - python3 (>=3.8)
# - scripts/build.sh
# - src/model/utils/alert_manager.py
#
# Notes :
# - Simule un environnement de test avec des stubs pour setup_env.py, validate_prompt_compliance.py, et run_all_tests.py.
# - Vérifie l’absence de références à dxFeed, obs_t, 320/81 features.
# - Vérifie les logs dans data/logs/build.log et data/logs/build_performance.csv.

# Définir le mode strict
set -euo pipefail

# Définir les chemins
TEST_DIR="/tmp/mia_ia_system_test"
PROJECT_ROOT="$TEST_DIR/MIA_IA_SYSTEM_v2_2025"
LOG_DIR="$PROJECT_ROOT/data/logs"
LOG_FILE="$LOG_DIR/build.log"
PERF_LOG_FILE="$LOG_DIR/build_performance.csv"
BUILD_SCRIPT="$PROJECT_ROOT/scripts/build.sh"
PYTHON_BIN="python3"

# Nettoyer l’environnement de test
rm -rf "$TEST_DIR"
mkdir -p "$PROJECT_ROOT/scripts" "$LOG_DIR"

# Créer des stubs pour les scripts Python
cat > "$PROJECT_ROOT/scripts/setup_env.py" <<EOF
#!/usr/bin/env python3
print("Stub: setup_env.py exécuté avec succès")
exit(0)
EOF

cat > "$PROJECT_ROOT/scripts/validate_prompt_compliance.py" <<EOF
#!/usr/bin/env python3
print("Stub: validate_prompt_compliance.py exécuté avec succès")
exit(0)
EOF

cat > "$PROJECT_ROOT/scripts/run_all_tests.py" <<EOF
#!/usr/bin/env python3
print("Stub: run_all_tests.py exécuté avec succès")
exit(0)
EOF

# Créer un stub pour alert_manager.py
mkdir -p "$PROJECT_ROOT/src/model/utils"
cat > "$PROJECT_ROOT/src/model/utils/alert_manager.py" <<EOF
#!/usr/bin/env python3
class AlertManager:
    def send_alert(self, message, priority):
        print(f"Stub: Alerte envoyée - Message: {message}, Priorité: {priority}")
EOF

# Copier le script build.sh à tester
cp "$PROJECT_ROOT/../scripts/build.sh" "$BUILD_SCRIPT"
chmod +x "$BUILD_SCRIPT" "$PROJECT_ROOT/scripts/setup_env.py" "$PROJECT_ROOT/scripts/validate_prompt_compliance.py" "$PROJECT_ROOT/scripts/run_all_tests.py" "$PROJECT_ROOT/src/model/utils/alert_manager.py"

# Test 1 : Vérifier que le script s’exécute sans erreur
echo "Test 1 : Exécution de build.sh..."
if ! bash "$BUILD_SCRIPT" > /tmp/build_output.log 2>&1; then
    echo "Échec : build.sh a retourné une erreur"
    cat /tmp/build_output.log
    exit 1
fi
echo "Succès : build.sh s’est exécuté sans erreur"

# Test 2 : Vérifier la création du fichier de log
echo "Test 2 : Vérification de la création de build.log..."
if [ ! -f "$LOG_FILE" ]; then
    echo "Échec : Fichier de log $LOG_FILE non créé"
    exit 1
fi
if ! grep "Build terminé avec succès" "$LOG_FILE" > /dev/null; then
    echo "Échec : Message de succès absent dans $LOG_FILE"
    exit 1
fi
echo "Succès : Fichier de log $LOG_FILE créé et contient le message de succès"

# Test 3 : Vérifier la création du fichier de performance
echo "Test 3 : Vérification de la création de build_performance.csv..."
if [ ! -f "$PERF_LOG_FILE" ]; then
    echo "Échec : Fichier de performance $PERF_LOG_FILE non créé"
    exit 1
fi
if ! grep "setup_env" "$PERF_LOG_FILE" > /dev/null; then
    echo "Échec : Métriques setup_env absentes dans $PERF_LOG_FILE"
    exit 1
fi
if ! grep "validate_prompt_compliance" "$PERF_LOG_FILE" > /dev/null; then
    echo "Échec : Métriques validate_prompt_compliance absentes dans $PERF_LOG_FILE"
    exit 1
fi
if ! grep "run_all_tests" "$PERF_LOG_FILE" > /dev/null; then
    echo "Échec : Métriques run_all_tests absentes dans $PERF_LOG_FILE"
    exit 1
fi
echo "Succès : Fichier de performance $PERF_LOG_FILE créé avec les métriques attendues"

# Test 4 : Vérifier l’absence de références obsolètes
echo "Test 4 : Vérification de l’absence de références à dxFeed, obs_t, 320/81 features..."
if grep -i "dxFeed" "$BUILD_SCRIPT" > /dev/null; then
    echo "Échec : Référence à dxFeed trouvée dans build.sh"
    exit 1
fi
if grep -i "obs_t" "$BUILD_SCRIPT" > /dev/null; then
    echo "Échec : Référence à obs_t trouvée dans build.sh"
    exit 1
fi
if grep -i "320 features" "$BUILD_SCRIPT" > /dev/null; then
    echo "Échec : Référence à 320 features trouvée dans build.sh"
    exit 1
fi
if grep -i "81 features" "$BUILD_SCRIPT" > /dev/null; then
    echo "Échec : Référence à 81 features trouvée dans build.sh"
    exit 1
fi
echo "Succès : Aucune référence obsolète trouvée"

# Test 5 : Vérifier les alertes en cas d’échec (simuler un échec)
echo "Test 5 : Vérification des alertes en cas d’échec..."
# Modifier validate_prompt_compliance.py pour simuler un échec
cat > "$PROJECT_ROOT/scripts/validate_prompt_compliance.py" <<EOF
#!/usr/bin/env python3
print("Stub: validate_prompt_compliance.py échoue")
exit(1)
EOF
chmod +x "$PROJECT_ROOT/scripts/validate_prompt_compliance.py"
if bash "$BUILD_SCRIPT" > /tmp/build_output.log 2>&1; then
    echo "Échec : build.sh n’a pas détecté l’erreur de validate_prompt_compliance.py"
    exit 1
fi
if ! grep "Alerte envoyée.*validate_prompt_compliance" /tmp/build_output.log > /dev/null; then
    echo "Échec : Alerte non envoyée pour l’échec de validate_prompt_compliance.py"
    exit 1
fi
echo "Succès : Alerte envoyée pour l’échec simulé"

# Nettoyer
rm -rf "$TEST_DIR"
echo "Tous les tests ont réussi !"
exit 0