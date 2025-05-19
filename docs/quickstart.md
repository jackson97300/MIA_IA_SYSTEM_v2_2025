Guide de démarrage rapide pour MIA_IA_SYSTEM_v2_2025
Version : 2.1.5Date : 2025-05-14  
Aperçu
Ce guide fournit un aperçu concis pour configurer et exécuter le projet MIA_IA_SYSTEM_v2_2025, un système de trading avancé pour les contrats à terme E-mini S&P 500 (ES) et Micro E-mini Nasdaq-100 (MNQ), utilisant les données IQFeed. Conçu pour les opérations de 2025, le système est extensible à d’autres instruments (ex. : NQ, DAX) d’ici 2026-2027 (Phase 16). Il exploite des modèles d’apprentissage automatique (SAC, PPO, DDPG, PPO-Lagrangian, QR-DQN), des réseaux neuronaux (LSTM, CNN), et un pipeline de fonctionnalités robuste pour traiter les données de marché et exécuter des trades. Les nouveaux modules incluent le dimensionnement dynamique des positions (risk_manager.py, suggestion 1), la détection des régimes de marché (regime_detector.py, suggestion 4), Safe RL/CVaR-PPO (trade_probability.py, suggestion 7), RL Distributionnel/QR-DQN (trade_probability.py, suggestion 8), et les ensembles de politiques avec vote bayésien (trade_probability.py, suggestion 10).
Note sur les dossiers policies : Le répertoire officiel pour les politiques de routage est src/model/router/policies. Le dossier src/model/policies semble être un résidu et doit être vérifié pour suppression afin d’éviter des conflits d’importation. Consultez troubleshooting.md pour résoudre les erreurs liées à des importations ambiguës.
Conformité : Toutes les références à dxFeed, obs_t, 320 features, et 81 features ont été supprimées pour aligner avec les spécifications de MIA_IA_SYSTEM_v2_2025.
Prérequis

Système d’exploitation : Linux (préféré, ex. : Ubuntu 22.04 LTS), Windows 11 avec WSL2, ou macOS Ventura (13.0 ou supérieur). Ubuntu est recommandé pour la compatibilité avec les scripts shell et les services AWS.
Python : Version 3.10.x (3.10.12 recommandé).
Dépendances : Listées dans requirements.txt, installées via pip. Principaux packages : pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, boto3>=1.24.0,<2.0.0, loguru>=0.7.0,<1.0.0, hmmlearn>=0.2.8,<0.3.0, pydantic>=2.0.0,<3.0.0, cachetools>=5.3.0,<6.0.0, scikit-learn>=1.5.0,<2.0.0, joblib>=1.3.0,<2.0.0, stable-baselines3>=2.0.0,<3.0.0, ray[rllib]>=2.0.0,<3.0.0.
Clés API et services :
IQFeed : Compte actif et clé API (obtenir sur IQFeed).
NewsAPI : Clé pour l’analyse de sentiment des nouvelles (obtenir sur NewsAPI).
AWS CLI : Configuré pour les sauvegardes S3 (installer via pip install awscli et exécuter aws configure).
Telegram : Jeton de bot et ID de chat pour les alertes (créer via BotFather).


Matériel : Minimum 16 Go RAM, 4 cœurs CPU, 50 Go d’espace SSD libre. Recommandé : 32 Go RAM (64 Go pour PPO-Lagrangian/QR-DQN), 8 cœurs CPU, 100 Go NVMe SSD.
Réseau : Connexion Internet stable (minimum 10 Mbps, recommandé 100 Mbps). Ports ouverts pour IQFeed (9100, 9300, 9400).

Configuration étape par étape
1. Cloner le dépôt
Clonez le dépôt du projet sur votre machine locale ou dans un environnement Codespaces :
git clone https://github.com/xAI/MIA_IA_SYSTEM_v2_2025.git
cd MIA_IA_SYSTEM_v2_2025

Si le dépôt est privé, utilisez un jeton d’accès personnel ou une clé SSH :
git clone https://<votre-jeton>@github.com/xAI/MIA_IA_SYSTEM_v2_2025.git

2. Configurer l’environnement
Créez et activez un environnement virtuel Python, puis installez les dépendances :
python3 -m venv venv
source venv/bin/activate  # Ubuntu/macOS
venv\Scripts\activate     # Windows
pip install --upgrade pip
pip install -r requirements.txt

Vérifiez la version de Python :
python --version  # Doit afficher 3.10.x

3. Configurer le système
Configurez les paramètres du système dans config/es_config.yaml :

Mettez à jour les identifiants IQFeed :
data_provider:
  iqfeed:
    username: "votre_utilisateur"
    password: "votre_mot_de_passe"
    host: "127.0.0.1"
    port: 9100


Configurez la clé NewsAPI :
news_analyzer:
  newsapi_key: "votre_clé_newsapi"


Configurez le bot Telegram pour les alertes :
telegram:
  enabled: true
  bot_token: "votre_bot_token"
  chat_id: "votre_chat_id"


Configurez AWS S3 pour les sauvegardes distribuées :
s3_bucket: "votre-bucket-s3"
s3_prefix: "mia_ia_system/"


Configurez les paramètres des nouveaux modules :
risk_manager:
  atr_threshold: 100.0
  orderflow_imbalance_limit: 0.9
regime_detector:
  hmm_states: 3
  cache_ttl: 3600
trade_probability:
  model_type: "ppo_cvar"  # Options: ppo_cvar, qr_dqn, ensemble
  cvar_alpha: 0.95
  quantiles: 51
  ensemble_weights: "0.4,0.3,0.3"  # SAC, PPO, DDPG



Initialisez les configurations par défaut si nécessaire :
python src/model/utils/config_manager.py --init-config

4. Tester la connexion IQFeed
Vérifiez la connexion à IQFeed :
python src/data/data_provider.py --test-connection

Si la connexion échoue, consultez data/logs/<market>/data_provider.log et référez-vous à troubleshooting.md.
Exécution du système
1. Initialiser la base de données
Configurez la base de données de mémoire contextuelle (market_memory.db) :
mkdir -p data/<market>
python src/model/utils/db_maintenance.py --init --market ES

Remplacez <market> par ES ou MNQ.
2. Exécuter en mode paper trading
Démarrez le système en mode paper trading pour simuler des trades :
python src/run_system.py --paper --market ES

Cette commande exécute le système avec des trades simulés, utilisant trading_env.py, main_router.py, risk_manager.py, regime_detector.py, et trade_probability.py. Les journaux sont enregistrés dans data/logs/ES/, et les instantanés sont stockés dans data/cache/<module>/ES/.
3. Vérifier le fonctionnement
Vérifiez les journaux du système pour confirmer son fonctionnement :
cat data/logs/ES/run_system.log

Assurez-vous que les instantanés et les sauvegardes sont créés :
ls data/cache/trading_utils/ES/
ls data/checkpoints/db_maintenance/ES/

Exécution des tests
Exécutez les tests unitaires pour valider la configuration du système :
pytest tests/ -v

Tests clés incluent :

test_data_provider.py : Valide la connexion IQFeed.
test_obs_template.py : Vérifie le formatage correct des vecteurs d’observation.
test_trading_utils.py : Teste la détection des régimes et le calcul des profits.
test_risk_manager.py : Vérifie le dimensionnement dynamique (atr_dynamic, suggestion 1).
test_regime_detector.py : Teste la détection HMM (hmm_state_distribution, suggestion 4).
test_trade_probability.py : Vérifie les prédictions RL (cvar_loss, qr_dqn_quantiles, ensemble_weights, suggestions 7, 8, 10).

Si des tests échouent, consultez troubleshooting.md.
Prochaines étapes

Extension à d’autres instruments : La Phase 16 prévoit le support d’instruments supplémentaires (NQ, DAX) d’ici 2026-2027. Mettez à jour es_config.yaml et feature_sets.yaml pour les nouveaux marchés lorsque disponibles.
Trading en direct : Passez au trading en direct en supprimant l’option --paper :python src/run_system.py --market ES


Intégration Bloomberg : Préparez l’intégration de l’API Bloomberg (juin 2025) en surveillant les mises à jour de data_provider.py.
Documentation : Explorez les guides détaillés dans docs/ :
installation.md : Instructions complètes d’installation.
setup.md : Configuration détaillée de l’environnement.
troubleshooting.md : Solutions aux erreurs courantes.
architecture.md : Vue d’ensemble de l’architecture.
techniques.md : Techniques de trading.
methodology.md : Détails sur les méthodes (3, 5, 7, 8, 10, 18).
risk_manager.md : Guide pour risk_manager.py.
regime_detector.md : Guide pour regime_detector.py.
trade_probability.md : Guide pour trade_probability.py.



En cas de problèmes, vérifiez data/logs/<market>/ (ex. : risk_manager.log, regime_detector.log, trade_probability.log) et consultez troubleshooting.md.
Fichier mis à jour : quickstart.md
Voici le contenu mis à jour de docs/quickstart.md, transformant le placeholder en un guide complet et synthétique :

markdown

Copier
# Quickstart Guide for MIA_IA_SYSTEM_v2_2025

**Version**: 2.1.4  
**Date**: 2025-05-13  

## Overview

This guide provides a concise overview to set up and run the `MIA_IA_SYSTEM_v2_2025` project, an advanced trading system for E-mini S&P 500 (ES) and Micro E-mini Nasdaq-100 (MNQ) futures, using IQFeed data. Designed for 2025 operations, the system is extensible to additional instruments (e.g., NQ, DAX) by 2026-2027 (Phase 16). It leverages machine learning models (SAC, PPO, DDPG), neural networks (LSTM, CNN), and a robust feature pipeline to process market data and execute trades.

**Note on `policies` directories**: The official directory for routing policies is `src/model/router/policies`. The `src/model/policies` directory appears to be a residual and should be verified for removal to avoid import conflicts.

## Prerequisites

- **Operating System**: Linux (preferred, e.g., Ubuntu 22.04 LTS), Windows 11 with WSL2, or macOS Ventura (13.0 or higher). Ubuntu is recommended for compatibility with shell scripts and AWS services.
- **Python**: Version 3.10.x (3.10.12 recommended).
- **Dependencies**: Listed in `requirements.txt`, installed via `pip`. Key packages include `pandas>=2.0.0,<3.0.0`, `numpy>=1.26.4,<2.0.0`, `boto3>=1.26.0,<2.0.0`, `loguru>=0.7.0,<1.0.0`.
- **API Keys and Services**:
  - **IQFeed**: Active account and API key (obtain from [IQFeed](https://www.iqfeed.net)).
  - **NewsAPI**: Key for news sentiment analysis (obtain from [NewsAPI](https://newsapi.org)).
  - **AWS CLI**: Configured for S3 backups (install via `pip install awscli` and run `aws configure`).
  - **Telegram**: Bot token and chat ID for alerts (create via [BotFather](https://t.me/BotFather)).
- **Hardware**: Minimum 16GB RAM, 4 CPU cores, 50GB free SSD space. Recommended: 32GB RAM, 8 CPU cores, 100GB NVMe SSD.
- **Network**: Stable internet connection (minimum 10 Mbps, recommended 100 Mbps). Open ports for IQFeed (9100, 9300, 9400).

## Step-by-Step Setup

### 1. Clone the Repository

Clone the project repository to your local machine or Codespaces environment:

```bash
git clone https://github.com/xAI/MIA_IA_SYSTEM_v2_2025.git
cd MIA_IA_SYSTEM_v2_2025
If the repository is private, use a personal access token or SSH key:

bash

Copier
git clone https://<your-token>@github.com/xAI/MIA_IA_SYSTEM_v2_2025.git
2. Set Up the Environment
Create and activate a Python virtual environment, then install dependencies:

bash

Copier
python3 -m venv venv
source venv/bin/activate  # Ubuntu/macOS
venv\Scripts\activate     # Windows
pip install --upgrade pip
pip install -r requirements.txt
Verify Python version:

bash

Copier
python --version  # Should output 3.10.x
3. Configure the System
Configure the system settings in config/es_config.yaml:

Update IQFeed credentials:
yaml

Copier
data_provider:
  iqfeed:
    username: "your_username"
    password: "your_password"
    host: "127.0.0.1"
    port: 9100
Configure NewsAPI key:
yaml

Copier
news_analyzer:
  newsapi_key: "your_newsapi_key"
Set up Telegram bot for alerts:
yaml

Copier
telegram:
  enabled: true
  bot_token: "your_bot_token"
  chat_id: "your_chat_id"
Configure AWS S3 for distributed backups:
yaml

Copier
s3_bucket: "your-s3-bucket"
s3_prefix: "mia_ia_system/"
Initialize default configurations if needed:
bash

Copier
python src/model/utils/config_manager.py --init-config
4. Test the IQFeed Connection
Verify the connection to IQFeed:

bash

Copier
python src/data/data_provider.py --test-connection
If the connection fails, check data/logs/<market>/data_provider.log and refer to troubleshooting.md.

Running the System
1. Initialize the Database
Set up the contextual memory database (market_memory.db):

bash

Copier
mkdir -p data/<market>
python src/model/utils/db_maintenance.py --init --market ES
Replace <market> with ES or MNQ.

2. Run in Paper Trading Mode
Start the system in paper trading mode to simulate trades:

bash

Copier
python src/run_system.py --paper --market ES
This command runs the system with simulated trades, using trading_env.py and main_router.py. Logs are saved in data/logs/ES/, and snapshots are stored in data/cache/<module>/ES/.

3. Verify Operation
Check the system logs to ensure it’s running correctly:

bash

Copier
cat data/logs/ES/run_system.log
Verify that snapshots and backups are created:

bash

Copier
ls data/cache/trading_utils/ES/
ls data/checkpoints/db_maintenance/ES/
Running Tests
Run unit tests to validate the system setup:

bash

Copier
pytest tests/ -v
Key tests include:

test_data_provider.py: Validates IQFeed connection.
test_obs_template.py: Ensures correct observation vector formatting.
test_trading_utils.py: Tests regime detection and profit calculation.
If tests fail, refer to troubleshooting.md.

Next Steps
Extend to Other Instruments: Phase 16 plans to support additional instruments (NQ, DAX) by 2026-2027. Update es_config.yaml and feature_sets.yaml for new markets when available.
Live Trading: Transition to live trading by removing the --paper flag:
bash

Copier
python src/run_system.py --market ES
Bloomberg Integration: Prepare for Bloomberg API integration (June 2025) by monitoring updates in data_provider.py.
Documentation: Explore detailed guides in docs/:
installation.md: Full setup instructions.
troubleshooting.md: Solutions to common errors.
architecture.md: System architecture overview.
methodology.md: Details on methods (3, 5, 7, 8, 10, 18).
If issues arise, check data/logs/<market>/ and refer to troubleshooting.md.