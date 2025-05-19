Installation de MIA_IA_SYSTEM_v2_2025
Version : 2.1.5Date : 2025-05-14  
Aperçu
Ce document fournit des instructions détaillées pour installer et configurer l’environnement de MIA_IA_SYSTEM_v2_2025, un système avancé de trading algorithmique conçu pour les contrats à terme (ES, MNQ). L’installation inclut la configuration des dépendances Python, des bases de données, des services externes (IQFeed, AWS S3, Telegram), et la validation de l’environnement via des tests unitaires. Le système utilise exclusivement IQFeed comme source de données via data_provider.py et est compatible avec 350 features pour l’entraînement et 150 SHAP features pour l’inférence, comme défini dans config/feature_sets.yaml. Les nouvelles fonctionnalités incluent le dimensionnement dynamique des positions (risk_manager.py, suggestion 1), la détection des régimes de marché (regime_detector.py, suggestion 4), Safe RL/CVaR-PPO (trade_probability.py, suggestion 7), RL Distributionnel/QR-DQN (trade_probability.py, suggestion 8), et les ensembles de politiques avec vote bayésien (trade_probability.py, suggestion 10).
Note sur les dossiers policies : Le répertoire officiel pour les politiques de routage est src/model/router/policies. Le dossier src/model/policies semble être un résidu et doit être vérifié pour suppression afin d’éviter des conflits d’importation. Consultez troubleshooting.md pour résoudre les erreurs liées à des importations ambiguës.
Conformité : Toutes les références à dxFeed, obs_t, 320 features, et 81 features ont été supprimées pour aligner avec les spécifications de MIA_IA_SYSTEM_v2_2025.
Prérequis
Avant de commencer, assurez-vous que les prérequis suivants sont remplis :
1. Système d’exploitation

Supporté : Ubuntu 22.04 LTS, Windows 11, macOS Ventura (13.0) ou supérieur.
Recommandé : Ubuntu 22.04 LTS pour une compatibilité optimale avec les dépendances et les services AWS.

2. Matériel

Minimum :
CPU : 4 cœurs (Intel i5 ou équivalent).
RAM : 16 Go.
Stockage : 50 Go SSD (pour les bases de données et les logs).


Recommandé :
CPU : 8 cœurs (Intel i9 ou AMD Ryzen 7).
RAM : 64 Go (pour PPO-Lagrangian et QR-DQN).
Stockage : 100 Go NVMe SSD.
GPU : NVIDIA GPU avec CUDA 11.8 (facultatif, pour l’entraînement accéléré des modèles neuronaux).



3. Logiciels

Python : Version 3.10.x (3.10.12 recommandé).
Gestionnaire de paquets : pip (version 23.0 ou supérieure).
Base de données : SQLite (pour market_memory.db).
Git : Pour cloner le dépôt du projet.
IQFeed : Compte actif et client IQFeed installé (version 6.2 ou supérieure).
AWS CLI : Pour les sauvegardes distribuées sur S3 (version 2.13 ou supérieure).
Telegram : Bot configuré pour les alertes (voir alert_manager.py).
Dépendances supplémentaires : hmmlearn>=0.2.8,<0.3.0, stable-baselines3>=2.0.0,<3.0.0, ray[rllib]>=2.0.0,<3.0.0 pour les nouveaux modules (voir requirements.txt).

4. Accès réseau

Connexion Internet stable (minimum 10 Mbps, recommandé 100 Mbps).
Ports ouverts pour IQFeed (par défaut : 9100, 9300, 9400).
Accès à l’API Telegram pour les alertes.

Installation de base
1. Cloner le dépôt
Clonez le dépôt du projet depuis le référentiel Git :
git clone https://github.com/xAI/MIA_IA_SYSTEM_v2_2025.git
cd MIA_IA_SYSTEM_v2_2025

Note : Si le dépôt est privé, utilisez un jeton d’accès personnel ou des identifiants SSH.
2. Créer un environnement virtuel
Créez et activez un environnement virtuel Python pour isoler les dépendances :
python3 -m venv venv
source venv/bin/activate  # Sur Ubuntu/macOS
venv\Scripts\activate     # Sur Windows

3. Installer les dépendances
Installez les dépendances listées dans requirements.txt :
pip install --upgrade pip
pip install -r requirements.txt

Les dépendances principales incluent :

pandas>=2.0.0,<3.0.0
numpy>=1.26.4,<2.0.0
psutil>=5.9.8,<6.0.0
pyyaml>=6.0.0,<7.0.0
matplotlib>=3.7.0,<4.0.0
boto3>=1.26.0,<2.0.0
loguru>=0.7.0,<1.0.0
pytest>=7.3.0,<8.0.0
twilio>=7.0.0,<8.0.0 (pour les alertes SMS)
hmmlearn>=0.2.8,<0.3.0 (pour regime_detector.py, suggestion 4)
stable-baselines3>=2.0.0,<3.0.0 (pour PPO-Lagrangian, QR-DQN, suggestions 7, 8)
ray[rllib]>=2.0.0,<3.0.0 (pour les ensembles, suggestion 10)
Autres dépendances spécifiques (voir requirements.txt).

Note : Si vous utilisez un GPU, installez les dépendances CUDA séparément (ex. : pytorch avec CUDA 11.8).
4. Configurer IQFeed
Installez le client IQFeed (disponible sur www.iqfeed.net).Configurez vos identifiants IQFeed dans config/es_config.yaml :
data_provider:
  iqfeed:
    username: "votre_utilisateur"
    password: "votre_mot_de_passe"
    host: "127.0.0.1"
    port: 9100

Testez la connexion :
python src/data/data_provider.py --test-connection

5. Configurer AWS S3
Pour les sauvegardes distribuées, configurez vos identifiants AWS :Installez AWS CLI :
pip install awscli
aws configure

Entrez vos clés d’accès AWS (AWS Access Key ID, AWS Secret Access Key, région par défaut : us-east-1).Configurez le bucket S3 dans config/es_config.yaml :
s3_bucket: "votre-bucket-s3"
s3_prefix: "mia_ia_system/"

Configuration de l’environnement
1. Configurer Telegram pour les alertes

Créez un bot Telegram via BotFather et obtenez un token.
Obtenez votre chat_id en envoyant un message au bot et en utilisant une API comme https://api.telegram.org/bot<token>/getUpdates.
Configurez les paramètres dans config/es_config.yaml :telegram:
  enabled: true
  bot_token: "votre_token"
  chat_id: "votre_chat_id"


Testez les alertes :python src/utils/telegram_alert.py --test



2. Configurer la base de données
Le système utilise market_memory.db pour stocker les données contextuelles (méthode 7). Configurez la base de données :

Créez le répertoire pour la base de données :mkdir -p data/<market>

Remplacez <market> par ES ou MNQ selon votre configuration.
Configurez le chemin dans config/es_config.yaml :db_maintenance:
  db_path: "data/<market>/market_memory_<market>.db"


Initialisez la base de données avec le schéma :python src/model/utils/db_maintenance.py --init



3. Configurer les fichiers de configuration
Les fichiers de configuration principaux sont situés dans config/ :

es_config.yaml : Paramètres pour IQFeed, AWS S3, Telegram, et autres modules.
feature_sets.yaml : Documentation des 350 fonctionnalités (entraînement) et 150 fonctionnalités SHAP (inférence).
algo_config.yaml : Paramètres pour les algorithmes SAC, PPO, DDPG.
risk_manager_config.yaml : Paramètres pour le dimensionnement dynamique (suggestion 1).
regime_detector_config.yaml : Paramètres pour la détection HMM (suggestion 4).
trade_probability_config.yaml : Paramètres pour PPO-Lagrangian, QR-DQN, et le vote bayésien (suggestions 7, 8, 10).

Vérifiez que config/es_config.yaml contient les sections nécessaires :
data_provider:
  iqfeed: { ... }
telegram: { ... }
s3_bucket: "votre-bucket-s3"
s3_prefix: "mia_ia_system/"
obs_template:
  buffer_size: 100
  max_cache_size: 1000
  weights: { ... }
db_maintenance:
  db_path: "data/<market>/market_memory_<market>.db"
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

Si feature_sets.yaml, algo_config.yaml, risk_manager_config.yaml, regime_detector_config.yaml, ou trade_probability_config.yaml sont absents, initialisez-les avec des valeurs par défaut :
python src/model/utils/config_manager.py --init-config

4. Vérifier la structure du projet
Assurez-vous que la structure des dossiers est correcte :
MIA_IA_SYSTEM_v2_2025/
├── config/
│   ├── es_config.yaml
│   ├── feature_sets.yaml
│   ├── algo_config.yaml
│   ├── risk_manager_config.yaml
│   ├── regime_detector_config.yaml
│   └── trade_probability_config.yaml
├── data/
│   ├── logs/<market>/
│   ├── cache/<module>/<market>/
│   ├── checkpoints/<module>/<market>/
│   ├── figures/<module>/<market>/
│   └── features/<market>/
├── src/
│   ├── data/
│   ├── features/
│   ├── model/
│   │   ├── envs/
│   │   ├── rl/
│   │   ├── router/
│   │   │   └── policies/  # Répertoire officiel
│   │   └── utils/
│   └── utils/
├── tests/
└── docs/

Note : Vérifiez le dossier src/model/policies. S’il existe, confirmez qu’il n’est pas utilisé et supprimez-le pour éviter les conflits avec src/model/router/policies.
Validation et tests
1. Exécuter les tests unitaires
Le projet inclut des tests unitaires dans tests/ pour valider les modules principaux. Exécutez-les pour confirmer que l’installation est correcte :
pytest tests/ -v

Les tests couvrent :

test_data_provider.py : Connexion à IQFeed.
test_shap_weighting.py : Sélection des 150 fonctionnalités SHAP.
test_obs_template.py : Formatage du vecteur d’observation.
test_trading_utils.py : Détection de régime, ajustement du risque, calcul du profit.
test_db_maintenance.py : Maintenance de market_memory.db.
test_alert_manager.py : Envoi d’alertes Telegram.
test_algo_performance_logger.py : Journalisation des performances.
test_risk_manager.py : Dimensionnement dynamique (atr_dynamic, orderflow_imbalance, suggestion 1).
test_regime_detector.py : Détection HMM (hmm_state_distribution, suggestion 4).
test_trade_probability.py : Prédictions RL (cvar_loss, qr_dqn_quantiles, ensemble_weights, suggestions 7, 8, 10).

Attendu : Tous les tests doivent passer. Si des tests échouent, consultez docs/troubleshooting.md.
2. Tester un module spécifique
Pour tester un module individuel, par exemple obs_template.py :
pytest tests/test_obs_template.py -v

Pour tester les nouveaux modules :

risk_manager.py :pytest tests/test_risk_manager.py -v


regime_detector.py :pytest tests/test_regime_detector.py -v


trade_probability.py :pytest tests/test_trade_probability.py -v



3. Vérifier les sorties
Après l’exécution des tests, vérifiez les sorties suivantes :

data/logs/<market>/ : Logs des modules (ex. : obs_template.log, risk_manager.log, regime_detector.log, trade_probability.log).
data/cache/<module>/<market>/*.json.gz : Snapshots compressés.
data/checkpoints/<module>/<market>/*.json.gz : Sauvegardes incrémentielles.
data/figures/<module>/<market>/*.png : Visualisations.
data/features/<market>/obs_template.csv : Vecteur d’observation.
data/logs/<market>/risk_manager_performance.csv : Performances du dimensionnement dynamique (suggestion 1).
data/logs/<market>/regime_detector_performance.csv : Performances de la détection HMM (suggestion 4).
data/logs/<market>/trade_probability.log : Logs des prédictions RL (suggestions 7, 8, 10).

Dépannage
Pour les erreurs courantes, consultez docs/troubleshooting.md. Voici quelques problèmes fréquents :

Erreur de connexion IQFeed

Symptôme : ConnectionError dans data_provider.py.
Solution :
Vérifiez les identifiants dans es_config.yaml.
Assurez-vous que le client IQFeed est en cours d’exécution.
Testez les ports : telnet 127.0.0.1 9100.




Dépendances manquantes

Symptôme : ModuleNotFoundError lors de l’exécution des tests.
Solution :
Réinstallez les dépendances : pip install -r requirements.txt.
Vérifiez la version de Python : python --version (doit être 3.10.x).




Permissions d’écriture

Symptôme : Erreurs liées à l’écriture dans data/logs/ ou data/checkpoints/.
Solution :
Ajustez les permissions : chmod -R 777 data/.
Exécutez le script en tant qu’administrateur sur Windows.




Tests unitaires échoués

Symptôme : Certains tests dans pytest échouent.
Solution :
Consultez les logs dans data/logs/<market>/.
Vérifiez les configurations dans es_config.yaml.
Exécutez les tests individuellement pour isoler l’erreur.
Pour les nouveaux modules, vérifiez les journaux spécifiques : risk_manager.log, regime_detector.log, trade_probability.log.





Prochaines étapes

Intégration Bloomberg : Préparez l’environnement pour l’API Bloomberg (prévue pour juin 2025) en installant les dépendances nécessaires.
Vérification des dossiers : Supprimez src/model/policies s’il n’est pas utilisé pour éviter les conflits avec src/model/router/policies.
Monitoring : Configurez des outils de surveillance (ex. : Prometheus, Grafana) pour suivre les performances du système.
Modules spécifiques : Configurez et testez les nouveaux modules :
risk_manager.py : Dimensionnement dynamique (suggestion 1).
regime_detector.py : Détection HMM (suggestion 4).
trade_probability.py : Prédictions RL (suggestions 7, 8, 10).



Pour plus de détails, consultez :

architecture.md : Vue d’ensemble de l’architecture.
feature_engineering.md : Détails sur les fonctionnalités.
troubleshooting.md : Solutions aux erreurs courantes.
api_reference.md : Référence des fonctions clés.
risk_manager.md : Guide pour risk_manager.py.
regime_detector.md : Guide pour regime_detector.py.
trade_probability.md : Guide pour trade_probability.py.

