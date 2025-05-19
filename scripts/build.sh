#!/bin/bash
# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/scripts/build.sh
#
# Script shell pour automatiser la compilation, la validation, et les tests de MIA_IA_SYSTEM_v2_2025.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Appelle setup_env.py, validate_prompt_compliance.py, et run_all_tests.py pour configurer l’environnement,
#        valider la conformité du code, et exécuter les tests unitaires. Conforme à la Phase 16 (ensemble et transfer learning).
#        Aligné sur 350 features pour l’entraînement et 150 SHAP features pour l’inférence/fallback via feature_sets.yaml.
#        Utilise exclusivement IQFeed comme source de données via data_provider.py.
#
# Dépendances :
# - python3 (>=3.8)
# - scripts/setup_env.py
# - scripts/validate_prompt_compliance.py
# - scripts/run_all_tests.py
#
# Outputs :
# - data/logs/build.log : Logs des étapes du build.
# - data/logs/build_performance.csv : Métriques de performance (CPU, mémoire) via psutil.
#
# Notes :
# - Encodage UTF-8 requis pour tous les fichiers.
# - Envoie des alertes via alert_manager.py en cas d’échec (priorités : 3=error, 4=urgent).
# - Tests unitaires disponibles dans tests/test_build.sh pour valider le script.
# - Vérifier les dépendances Python avant exécution (requirements.txt).
# - Consultez data/logs/build.log et data/logs/build_performance.csv pour les détails.

# Définir le mode strict
set -euo pipefail

# Définir les chemins
PROJECT_ROOT="D:/MIA_IA_SYSTEM_v2_2025"
LOG_DIR="$PROJECT_ROOT/data/logs"
LOG_FILE="$LOG_DIR/build.log"
PERF_LOG_FILE="$LOG_DIR/build_performance.csv"

# Vérifier que le répertoire racine existe
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "Erreur : Répertoire racine inaccessible ($PROJECT_ROOT)"
    echo "[$TIMESTAMP] Erreur : Répertoire racine inaccessible" >> "$LOG_FILE"
    $PYTHON_BIN -c "from src.model.utils.alert_manager import AlertManager; AlertManager().send_alert('Échec du build : Répertoire racine inaccessible ($PROJECT_ROOT)', priority=4)"
    exit 1
fi

# Créer le répertoire de logs s'il n'existe pas
mkdir -p "$LOG_DIR"

# Initialiser le fichier de log
TIMESTAMP=$(date -u '+%Y-%m-%d %H:%M:%S')
echo "[$TIMESTAMP] Début du build de MIA_IA_SYSTEM_v2_2025..." | tee -a "$LOG_FILE"

# Vérifier la version de Python
PYTHON_BIN="python3"
PYTHON_VERSION=$($PYTHON_BIN --version 2>&1 | grep -o '[0-9]\.[0-9]')
MIN_PYTHON_VERSION="3.8"
if [[ "$PYTHON_VERSION" < "$MIN_PYTHON_VERSION" ]]; then
    echo "Erreur : Version de Python $PYTHON_VERSION non supportée. Requis : >= $MIN_PYTHON_VERSION"
    echo "[$TIMESTAMP] Erreur : Version de Python non supportée" >> "$LOG_FILE"
    $PYTHON_BIN -c "from src.model.utils.alert_manager import AlertManager; AlertManager().send_alert('Échec du build : Version de Python $PYTHON_VERSION non supportée', priority=4)"
    exit 1
fi
echo "[$TIMESTAMP] Version de Python vérifiée : $PYTHON_VERSION" >> "$LOG_FILE"

# Configurer l'environnement
echo "[$TIMESTAMP] Configuration de l'environnement..." | tee -a "$LOG_FILE"
if ! $PYTHON_BIN "$PROJECT_ROOT/scripts/setup_env.py" >> "$LOG_FILE" 2>&1; then
    echo "Erreur : Échec de la configuration de l'environnement"
    echo "[$TIMESTAMP] Erreur : Échec de setup_env.py" >> "$LOG_FILE"
    $PYTHON_BIN -c "from src.model.utils.alert_manager import AlertManager; AlertManager().send_alert('Échec du build : Configuration de l\'environnement (setup_env.py)', priority=4)"
    exit 1
fi
echo "[$TIMESTAMP] Environnement configuré avec succès" >> "$LOG_FILE"

# Enregistrer les métriques de performance après configuration
$PYTHON_BIN -c "import psutil, csv, datetime; cpu = psutil.cpu_percent(); mem = psutil.virtual_memory().percent; \
    with open('$PERF_LOG_FILE', 'a') as f: \
    writer = csv.writer(f); writer.writerow([datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'setup_env', cpu, mem])" >> "$LOG_FILE" 2>&1
echo "[$TIMESTAMP] Métriques de performance enregistrées (setup_env)" >> "$LOG_FILE"

# Valider la conformité du code
echo "[$TIMESTAMP] Validation de la conformité du code..." | tee -a "$LOG_FILE"
if ! $PYTHON_BIN "$PROJECT_ROOT/scripts/validate_prompt_compliance.py" >> "$LOG_FILE" 2>&1; then
    echo "Erreur : Échec de la validation de la conformité"
    echo "[$TIMESTAMP] Erreur : Échec de validate_prompt_compliance.py" >> "$LOG_FILE"
    $PYTHON_BIN -c "from src.model.utils.alert_manager import AlertManager; AlertManager().send_alert('Échec du build : Validation de la conformité (validate_prompt_compliance.py)', priority=4)"
    exit 1
fi
echo "[$TIMESTAMP] Conformité validée avec succès" >> "$LOG_FILE"

# Enregistrer les métriques de performance après validation
$PYTHON_BIN -c "import psutil, csv, datetime; cpu = psutil.cpu_percent(); mem = psutil.virtual_memory().percent; \
    with open('$PERF_LOG_FILE', 'a') as f: \
    writer = csv.writer(f); writer.writerow([datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'validate_prompt_compliance', cpu, mem])" >> "$LOG_FILE" 2>&1
echo "[$TIMESTAMP] Métriques de performance enregistrées (validate_prompt_compliance)" >> "$LOG_FILE"

# Exécuter les tests
echo "[$TIMESTAMP] Exécution des tests..." | tee -a "$LOG_FILE"
if ! $PYTHON_BIN "$PROJECT_ROOT/scripts/run_all_tests.py" >> "$LOG_FILE" 2>&1; then
    echo "Erreur : Échec des tests"
    echo "[$TIMESTAMP] Erreur : Échec de run_all_tests.py" >> "$LOG_FILE"
    $PYTHON_BIN -c "from src.model.utils.alert_manager import AlertManager; AlertManager().send_alert('Échec du build : Tests unitaires (run_all_tests.py)', priority=4)"
    exit 1
fi
echo "[$TIMESTAMP] Tests exécutés avec succès" >> "$LOG_FILE"

# Enregistrer les métriques de performance après tests
$PYTHON_BIN -c "import psutil, csv, datetime; cpu = psutil.cpu_percent(); mem = psutil.virtual_memory().percent; \
    with open('$PERF_LOG_FILE', 'a') as f: \
    writer = csv.writer(f); writer.writerow([datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'run_all_tests', cpu, mem])" >> "$LOG_FILE" 2>&1
echo "[$TIMESTAMP] Métriques de performance enregistrées (run_all_tests)" >> "$LOG_FILE"

# Finalisation
echo "[$TIMESTAMP] Build terminé avec succès" | tee -a "$LOG_FILE"
echo "Build terminé avec succès. Consultez $LOG_FILE et $PERF_LOG_FILE pour les détails."
$PYTHON_BIN -c "from src.model.utils.alert_manager import AlertManager; AlertManager().send_alert('Build de MIA_IA_SYSTEM_v2_2025 terminé avec succès', priority=1)"
exit 0