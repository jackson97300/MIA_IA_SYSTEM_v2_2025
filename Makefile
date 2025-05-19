# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/Makefile
# Automatisation des tâches de développement pour MIA_IA_SYSTEM_v2_2025
#
# Version: 2.1.5
# Date: 2025-05-14
#
# Rôle: Simplifie l'exécution des tâches courantes : construction de l'image Docker,
#       exécution des tests unitaires, et linting du code.
#
# Utilisé par: Développeurs, CI/CD (.github/workflows/python.yml).
#
# Notes:
# - Conforme à structure.txt (version 2.1.5, 2025-05-14).
# - Supporte les suggestions 1 (position sizing dynamique), 4 (HMM/changepoint detection),
#   7 (Safe RL/CVaR-PPO), 8 (Distributional RL/QR-DQN), 9 (réentraînement).
# - Intègre avec scripts/build_docker.py, pytest, et flake8.
# - Compatible avec requirements.txt (flake8>=7.1.0,<8.0.0, pytest>=7.3.0,<8.0.0).
# - Pas de références à dxFeed, obs_t, 320 features, ou 81 features.
# - Policies Note: The official directory for routing policies is src/model/router/policies.
#   The src/model/policies directory is a residual and should be verified for removal to avoid import conflicts.

.PHONY: build test lint all

build:
	python scripts/build_docker.py

test:
	pytest tests/

lint:
	flake8 src/ tests/ scripts/

all: lint test build