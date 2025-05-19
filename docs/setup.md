Guide de configuration de MIA_IA_SYSTEM_v2_2025
Version : 2.1.5Date : 2025-05-14  
Aperçu
Ce guide fournit des instructions détaillées pour préparer l’environnement logiciel et matériel de MIA_IA_SYSTEM_v2_2025, un système avancé de trading algorithmique pour les contrats à terme E-mini S&P 500 (ES) et Micro E-mini Nasdaq-100 (MNQ). Le système traite les données de marché en temps réel d’IQFeed, utilisant 350 fonctionnalités pour l’entraînement et 150 fonctionnalités sélectionnées par SHAP pour l’inférence. Il intègre des modules pour le dimensionnement dynamique des positions (risk_manager.py, suggestion 1), la détection des régimes de marché (regime_detector.py, suggestion 4), le Safe RL/CVaR-PPO (trade_probability.py, suggestion 7), le RL Distributionnel/QR-DQN (trade_probability.py, suggestion 8), et les ensembles de politiques avec vote bayésien (trade_probability.py, suggestion 10). Ce document se concentre sur la préparation du système d’exploitation, de l’environnement Python, des services externes (IQFeed, AWS S3, Telegram, NewsAPI), et des prérequis matériels avant l’installation et l’exécution.
Note sur les dossiers policies : Le répertoire officiel pour les politiques de routage est src/model/router/policies. Le dossier src/model/policies semble être un résidu et doit être vérifié pour suppression afin d’éviter des conflits d’importation. Consultez troubleshooting.md pour résoudre les erreurs liées à des importations ambiguës.
Conformité : Toutes les références à dxFeed, obs_t, 320 features, et 81 features ont été supprimées pour aligner avec les spécifications de MIA_IA_SYSTEM_v2_2025.
Pour plus de détails, voir usage.md (guide d’utilisation), troubleshooting.md (dépannage), quickstart.md (démarrage rapide), installation.md (installation), architecture.md (vue d’ensemble), phases.md (phases de développement), modules.md (modules), api_reference.md (API), et roadmap.md (feuille de route).
Prérequis
Prérequis logiciels

Système d’exploitation :
Supportés : Ubuntu 22.04 LTS (recommandé), Windows 11 avec WSL2, macOS Ventura (13.0 ou supérieur).
Notes : Ubuntu est préféré pour la compatibilité avec les scripts shell et les services AWS. WSL2 est requis pour Windows afin d’exécuter les scripts basés sur Linux.


Python :
Version 3.10.x (3.10.12 recommandé).
Installer via un gestionnaire de paquets (ex. : apt sur Ubuntu, brew sur macOS) ou depuis python.org.


Dépendances :
Listées dans requirements.txt, incluant :
pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.0,<6.0.0 (pour risk_manager.py, suggestion 1).
hmmlearn>=0.2.8,<0.3.0, pydantic>=2.0.0,<3.0.0, cachetools>=5.3.0,<6.0.0, scikit-learn>=1.5.0,<2.0.0, joblib>=1.3.0,<2.0.0 (pour regime_detector.py, suggestion 4).
stable-baselines3>=2.0.0,<3.0.0 (pour trade_probability.py, suggestion 7).
ray[rllib]>=2.0.0,<3.0.0 (pour trade_probability.py, suggestion 8).
boto3>=1.24.0,<2.0.0, loguru>=0.7.0,<1.0.0, pytest>=7.3.0,<8.0.0, requests>=2.31.0,<3.0.0, sentry-sdk>=1.9.0,<2.0.0, mlflow>=2.0.0,<3.0.0, optuna>=3.0.0,<4.0.0, river>=0.8.0,<1.0.0.


Installées via pip après la configuration de l’environnement.


Services externes et clés API :
IQFeed : Compte actif et clé API depuis IQFeed. Requiert le logiciel client (version 6.2 ou supérieure).
NewsAPI : Clé API pour l’analyse de sentiment des nouvelles depuis NewsAPI.
AWS CLI : Configuré pour les sauvegardes S3. Installer via pip install awscli et exécuter aws configure.
Telegram : Jeton de bot et ID de chat pour les alertes, créés via BotFather.


Outils supplémentaires :
Git : Pour cloner le dépôt (git clone).
SQLite : Pour market_memory.db (inclus avec Python).
Telnet : Pour tester les ports IQFeed (installer via apt ou fonctionnalités Windows).



Prérequis matériels

Minimum :
CPU : 4 cœurs (Intel i5 ou équivalent).
RAM : 16 Go.
Stockage : 50 Go SSD (pour la base de données, les journaux, et les sauvegardes).


Recommandé :
CPU : 8 cœurs (Intel i9 ou AMD Ryzen 7).
RAM : 32 Go (64 Go pour trade_probability.py avec PPO-Lagrangian/QR-DQN, suggestions 7, 8).
Stockage : 100 Go NVMe SSD.
GPU : NVIDIA GPU avec CUDA 11.8 (optionnel, pour accélérer l’entraînement des modèles, ex. : train_ppo_cvar.py, train_qr_dqn.py).


Notes : Des spécifications plus élevées améliorent les performances pour le trading en temps réel et l’entraînement des modèles, notamment pour les modules intensifs comme trade_probability.py.

Prérequis réseau

Connexion : Internet stable (minimum 10 Mbps, recommandé 100 Mbps).
Ports : Ports ouverts pour IQFeed (9100, 9300, 9400).
Accès : Accès sans restriction aux API IQFeed, NewsAPI, AWS S3, et Telegram.

Configuration de l’environnement
1. Préparer le système d’exploitation

Ubuntu 22.04 LTS :
Mettre à jour le système :sudo apt update && sudo apt upgrade -y


Installer les outils essentiels :sudo apt install -y python3.10 python3.10-venv python3-pip git telnet




Windows 11 avec WSL2 :
Activer WSL2 :wsl --install -d Ubuntu-22.04


Mettre à jour Ubuntu dans WSL :sudo apt update && sudo apt upgrade -y


Installer les outils :sudo apt install -y python3.10 python3.10-venv python3-pip git telnet




macOS Ventura :
Installer Homebrew si absent :/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"


Installer les outils :brew install python@3.10 git


Installer telnet :brew install telnet





2. Installer et vérifier Python

Vérifier la version de Python :python3 --version

La sortie doit être 3.10.x. Si incorrect, installer Python 3.10 :
Ubuntu :sudo apt install -y python3.10


macOS :brew install python@3.10


Windows : Télécharger depuis python.org.


S’assurer que pip est installé :python3 -m ensurepip --upgrade
python3 -m pip install --upgrade pip



3. Cloner le dépôt

Cloner le dépôt GitHub :git clone https://github.com/xAI/MIA_IA_SYSTEM_v2_2025.git
cd MIA_IA_SYSTEM_v2_2025


Si le dépôt est privé, utiliser un jeton d’accès :
Générer un jeton sur GitHub (Settings > Developer settings > Personal access tokens).
Cloner avec le jeton :git clone https://<votre-jeton>@github.com/xAI/MIA_IA_SYSTEM_v2_2025.git





4. Configurer l’environnement virtuel

Créer un environnement virtuel :python3 -m venv venv


Activer l’environnement :source venv/bin/activate  # Ubuntu/macOS
venv\Scripts\activate     # Windows


Installer les dépendances :pip install --upgrade pip
pip install -r requirements.txt


Vérifier l’installation des dépendances clés :pip show pandas numpy psutil hmmlearn pydantic cachetools scikit-learn joblib stable-baselines3 ray

Assurez-vous que les versions correspondent à celles dans requirements.txt.

5. Configurer les services externes

IQFeed :

Installer le client IQFeed (télécharger depuis IQFeed).
Configurer les identifiants dans config/es_config.yaml :data_provider:
  iqfeed:
    username: "votre_utilisateur"
    password: "votre_mot_de_passe"
    host: "127.0.0.1"
    port: 9100


Démarrer le client IQFeed pour vérifier qu’il est en cours d’exécution :
Sur Windows : Vérifier dans le Gestionnaire des tâches.
Sur Ubuntu/macOS : Lancer via Wine si nécessaire :wine iqfeed_client.exe




Tester la connexion :python src/data/data_provider.py --test-connection




NewsAPI :

Obtenir une clé API depuis NewsAPI.
Configurer dans config/es_config.yaml :news_analyzer:
  newsapi_key: "votre_clé_newsapi"


Tester l’accès à NewsAPI :python src/model/utils/news_analyzer.py --test-api




AWS CLI :

Installer AWS CLI :pip install awscli


Configurer les identifiants AWS :aws configure

Entrer AWS Access Key ID, AWS Secret Access Key, région (us-east-1), et format de sortie (json).
Configurer S3 dans config/es_config.yaml :s3_bucket: "votre-bucket-s3"
s3_prefix: "mia_ia_system/"


Tester l’accès à S3 :aws s3 ls s3://votre-bucket-s3/mia_ia_system/




Telegram :

Créer un bot via BotFather :
Envoyer /start, puis /newbot.
Suivre les instructions pour obtenir un bot_token.


Obtenir le chat_id :
Envoyer un message au bot depuis Telegram.
Accéder à https://api.telegram.org/bot<votre_token>/getUpdates dans un navigateur pour récupérer le chat_id.


Configurer dans config/es_config.yaml :telegram:
  enabled: true
  bot_token: "votre_bot_token"
  chat_id: "votre_chat_id"


Tester l’envoi d’une alerte Telegram :python src/utils/telegram_alert.py --test




Nouveaux modules (suggestions 1, 4, 7, 8, 10) :

Configurer les paramètres spécifiques dans config/es_config.yaml :risk_manager:
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


Vérifier l’installation des dépendances spécifiques :pip show hmmlearn pydantic cachetools scikit-learn joblib stable-baselines3 ray


Tester les modules individuellement :python src/risk_management/risk_manager.py --test-config
python src/features/regime_detector.py --test-config
python src/model/trade_probability.py --test-config





6. Vérifier les prérequis

Tester la connectivité réseau :ping 8.8.8.8 -c 4  # Ubuntu/macOS
ping 8.8.8.8       # Windows


Tester les ports IQFeed :telnet 127.0.0.1 9100

Si la connexion échoue, ouvrir les ports dans le pare-feu :sudo ufw allow 9100  # Ubuntu

Sur Windows, utiliser le Pare-feu Windows pour autoriser les ports 9100, 9300, 9400.
Vérifier l’accès à AWS S3 :aws s3 ls s3://votre-bucket-s3/mia_ia_system/


Tester l’API Telegram :curl https://api.telegram.org/bot<votre_token>/getMe

La réponse doit retourner les détails du bot.
Vérifier les performances matérielles :# Ubuntu/macOS
lscpu  # Vérifier les cœurs CPU
free -m  # Vérifier la RAM
df -h  # Vérifier le stockage
# Windows
systeminfo  # Vérifier CPU, RAM, stockage

Assurez-vous que les spécifications répondent aux exigences minimales (4 cœurs, 16 Go RAM, 50 Go SSD) ou recommandées (8 cœurs, 32 Go RAM, 100 Go NVMe SSD).

7. Initialiser la base de données

Créer market_memory.db pour ES et MNQ :python src/db_maintenance.py --init --market ES
python src/db_maintenance.py --init --market MNQ


Vérifier la création de la base de données :ls data/<market>/market_memory_<market>.db

Exemple : data/ES/market_memory_ES.db.
Tester l’accès à la base de données :python src/db_maintenance.py --test-db --market ES



8. Exécuter les tests unitaires

Exécuter la suite de tests pour valider l’environnement :pytest tests/ -v


Vérifier les tests spécifiques aux nouveaux modules :pytest tests/test_risk_manager.py -v
pytest tests/test_regime_detector.py -v
pytest tests/test_trade_probability.py -v


Si des tests échouent, consulter troubleshooting.md pour résoudre les erreurs (ex. : ModuleNotFoundError, KeyError).

Notes

Conformité : Toutes les références à dxFeed, obs_t, 320 features, et 81 features ont été supprimées pour aligner avec les spécifications de MIA_IA_SYSTEM_v2_2025.
Évolutivité : Le système prend en charge ES et MNQ, avec des plans pour NQ, DAX, et les cryptomonnaies d’ici 2026-2027 (Phases 16, 17, voir phases.md).
Dépannage : Si des erreurs surviennent (ex. : ConnectionError pour IQFeed, OSError pour S3), consulter troubleshooting.md.
Prochaines étapes : Après la configuration, suivre installation.md pour installer le système, puis usage.md pour exécuter des trades.

