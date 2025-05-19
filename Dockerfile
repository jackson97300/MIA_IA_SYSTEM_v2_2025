# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/Dockerfile
# Containerisation pour MIA_IA_SYSTEM_v2_2025
#
# Version: 2.1.5
# Date: 2025-05-14
#
# Rôle: Construit une image Docker pour déployer l’application de trading sur Kubernetes.
# Utilisé par: Helm chart (helm/mia-system/) pour orchestrer le déploiement.
#
# Notes:
# - Conforme à structure.txt (version 2.1.5, 2025-05-14).
# - Supporte les suggestions 1 (position sizing dynamique), 2 (loggers), 4 (HMM/changepoint detection),
#   5 (profit factor), 7 (Safe RL/CVaR-PPO), 8 (Distributional RL/QR-DQN), 9 (réentraînement).
# - Inclut les dépendances pour risk_manager.py, regime_detector.py, trade_probability.py.
# - Copie src/, config/, tests/, et scripts/ pour exécuter run_system.py et tests.
# - Journalise les performances du build via scripts/build_docker.py.
# - Utilise .dockerignore pour exclure les fichiers temporaires et logs.
# - Pas de références à dxFeed, obs_t, 320 features, ou 81 features.
# - Policies Note: The official directory for routing policies is src/model/router/policies.
#   The src/model/policies directory is a residual and should be verified for removal to avoid import conflicts.

# Utiliser une image Python légère
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier requirements.txt et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier les répertoires src/, config/, tests/, et scripts/
COPY src/ src/
COPY config/ config/
COPY tests/ tests/
COPY scripts/ scripts/

# Définir les variables d’environnement par défaut
ENV MARKET=ES
ENV PYTHONUNBUFFERED=1

# Exposer le port pour les métriques Prometheus
EXPOSE 8000

# Commande pour exécuter l’application
CMD ["python", "src/run_system.py", "--market", "ES"]