Architecture de MIA_IA_SYSTEM_v2_2025
Version : 2.1.5Date : 2025-05-14  
Aperçu
MIA_IA_SYSTEM_v2_2025 est un système avancé de trading algorithmique conçu pour optimiser les décisions de trading sur les marchés financiers, avec un focus sur les contrats à terme (ES, MNQ). Le système intègre des techniques d’apprentissage par renforcement (SAC, PPO, DDPG, PPO-Lagrangian, QR-DQN), des modèles neuronaux profonds, et une gestion contextuelle des données via une mémoire adaptative (méthode 7). Il utilise exclusivement IQFeed comme source de données via data_provider.py et est compatible avec 350 fonctionnalités pour l’entraînement et 150 fonctionnalités SHAP pour l’inférence, comme défini dans config/feature_sets.yaml. Les nouvelles fonctionnalités incluent le dimensionnement dynamique des positions (risk_manager.py, suggestion 1), la détection des régimes de marché (regime_detector.py, suggestion 4), Safe RL/CVaR-PPO (trade_probability.py, suggestion 7), RL Distributionnel/QR-DQN (trade_probability.py, suggestion 8), et les ensembles de politiques avec vote bayésien (trade_probability.py, suggestion 10).
L’architecture est modulaire, avec des composants interconnectés pour assurer une scalabilité, une robustesse et une adaptabilité aux régimes de marché (tendance, range, défensif). Elle inclut des mécanismes de journalisation des performances, de sauvegardes incrémentielles/distribuées (AWS S3), et des alertes en temps réel via Telegram.
Note sur les dossiers policies : Le répertoire officiel pour les politiques de routage est src/model/router/policies. Le dossier src/model/policies semble être un résidu et doit être vérifié pour suppression afin d’éviter des conflits d’importation. Consultez troubleshooting.md pour résoudre les erreurs liées à des importations ambiguës.
Conformité : Toutes les références à dxFeed, obs_t, 320 features, et 81 features ont été supprimées pour aligner avec les spécifications de MIA_IA_SYSTEM_v2_2025.
Modules principaux
1. Data Provider (src/data/data_provider.py)

Rôle : Récupère les données de marché en temps réel via IQFeed.
Fonctionnalités :
Fournit des données brutes pour les 350 fonctionnalités (entraînement) et 150 fonctionnalités SHAP (inférence).
Intègre des métriques de volatilité, flux d’ordres, et données d’options (ex. : gex, max_pain_strike).


Interactions : Alimente trading_env.py, shap_weighting.py, risk_manager.py, regime_detector.py, et trade_probability.py.

2. Feature Engineering (src/features/)

Rôle : Génère et sélectionne les fonctionnalités pour l’entraînement et l’inférence.
Fichiers clés :
shap_weighting.py : Calcule les poids SHAP pour sélectionner les top 150 fonctionnalités pour l’inférence.
feature_sets.yaml : Documente les 350 fonctionnalités (entraînement) et 150 fonctionnalités SHAP (inférence).
[Ajout] risk_manager.py : Génère des fonctionnalités comme atr_dynamic et orderflow_imbalance (suggestion 1).
[Ajout] regime_detector.py : Génère hmm_state_distribution pour la détection des régimes (suggestion 4).


Méthodes :
Méthode 3 : Pondération des fonctionnalités selon le régime de marché (tendance, range, défensif).


Interactions : Fournit les fonctionnalités à obs_template.py, trading_env.py, et trade_probability.py.

3. Environnement de trading (src/model/envs/trading_env.py)

Rôle : Simule l’environnement de trading pour l’apprentissage par renforcement.
Fonctionnalités :
Utilise le vecteur d’observation formaté par obs_template.py (150 fonctionnalités SHAP).
Intègre les récompenses adaptatives (méthode 5), incluant cvar_loss pour PPO-Lagrangian (suggestion 7).


Interactions : Interface avec train_sac.py, train_ppo_cvar.py, train_qr_dqn.py, train_ensemble.py, et router/main_router.py.

4. Modèles d’apprentissage (src/model/rl/)

Rôle : Entraîne et exécute les algorithmes SAC, PPO, DDPG, PPO-Lagrangian, QR-DQN, et les ensembles.
Fichiers clés :
train_sac.py : Entraînement principal avec fine-tuning (méthode 8), apprentissage en ligne (méthode 10), et meta-learning (méthode 18).
[Ajout] train_ppo_cvar.py : Entraînement de PPO-Lagrangian pour minimiser cvar_loss (suggestion 7).
[Ajout] train_qr_dqn.py : Entraînement de QR-DQN pour qr_dqn_quantiles (suggestion 8).
[Ajout] train_ensemble.py : Entraînement des ensembles avec vote bayésien pour ensemble_weights (suggestion 10).
custom_mlp_policy.py, transformer_policy.py : Politiques personnalisées pour SAC.


Interactions : Utilise trading_env.py, trade_probability.py, et algo_performance_logger.py.

5. Routage (src/model/router/)

Rôle : Gère le routage des décisions de trading en fonction des régimes de marché.
Fichiers clés :
main_router.py : Orchestre les décisions entre les politiques.
detect_regime.py : Détecte les régimes de marché (tendance, range, défensif).
policies/ : Contient les politiques de routage (répertoire officiel, à ne pas confondre avec src/model/policies).


Interactions : Collabore avec trading_env.py, trading_utils.py, regime_detector.py, et trade_probability.py.

6. Utilitaires (src/model/utils/)

Rôle : Fournit des outils pour la gestion des performances, alertes, et observations.
Fichiers clés :
obs_template.py : Formate le vecteur d’observation (150 fonctionnalités SHAP).
trading_utils.py : Détection de régime, ajustement du risque/levier, calcul du profit (méthode 5).
algo_performance_logger.py : Enregistre les performances des algorithmes, incluant cvar_loss, qr_dqn_quantiles, et ensemble_weights.
alert_manager.py : Gère les alertes (Telegram, SMS, email).
miya_console.py : Interface de notification vocale et textuelle.
config_manager.py : Charge les configurations (ex. : es_config.yaml, risk_manager_config.yaml, regime_detector_config.yaml, trade_probability_config.yaml).


Interactions : Utilisé par tous les modules principaux.

7. Gestion des données (src/model/utils/db_maintenance.py)

Rôle : Maintient la base de données market_memory.db pour la mémoire contextuelle (méthode 7).
Fonctionnalités :
Purge des données obsolètes (>30 jours).
Sauvegardes incrémentielles et distribuées (AWS S3).
Stocke des métriques comme hmm_state_distribution (suggestion 4).


Interactions : Fournit des données historiques à trading_env.py, regime_detector.py, et trade_probability.py.

8. Gestion des risques (src/risk_management/risk_manager.py)

Rôle : Gère le dimensionnement dynamique des positions pour minimiser les risques (suggestion 1).
Fonctionnalités :
Calcule la taille des positions en fonction de atr_dynamic et orderflow_imbalance.
Ajuste le levier selon les régimes (ex. : 5x en tendance, 1x en défensif).


Interactions : Collabore avec trading_utils.py, trading_env.py, et main_router.py.

9. Détection des régimes (src/features/regime_detector.py)

Rôle : Détecte les régimes de marché (tendance, range, défensif) à l’aide d’un modèle HMM (suggestion 4).
Fonctionnalités :
Génère hmm_state_distribution pour améliorer la pondération des fonctionnalités.
Enregistre les performances dans data/logs/<market>/regime_detector_performance.csv.


Interactions : Fournit des données à obs_template.py, trading_utils.py, et trade_probability.py.

10. Prédiction des probabilités de trading (src/model/trade_probability.py)

Rôle : Prédit la probabilité de succès des trades à l’aide de modèles RL (suggestions 7, 8, 10).
Fonctionnalités :
Utilise PPO-Lagrangian pour minimiser cvar_loss (suggestion 7).
Utilise QR-DQN pour modéliser qr_dqn_quantiles (suggestion 8).
Implémente le vote bayésien pour combiner SAC, PPO, DDPG avec ensemble_weights (suggestion 10).


Interactions : Collabore avec trading_env.py, main_router.py, et algo_performance_logger.py.

Flux de données

Acquisition des données : data_provider.py récupère les données brutes via IQFeed.
Feature Engineering : shap_weighting.py sélectionne les 150 fonctionnalités SHAP, pondérées par régime (méthode 3), avec contributions de risk_manager.py (atr_dynamic, orderflow_imbalance) et regime_detector.py (hmm_state_distribution).
Formatage de l’observation : obs_template.py crée le vecteur d’observation pour trading_env.py.
Entraînement/Inférence : train_sac.py, train_ppo_cvar.py, train_qr_dqn.py, et train_ensemble.py entraînent les modèles SAC, PPO, DDPG, PPO-Lagrangian, QR-DQN, et les ensembles, avec fine-tuning (méthode 8) et meta-learning (méthode 18).
Routage des décisions : main_router.py utilise detect_regime.py et regime_detector.py pour router les décisions selon le régime, avec prédictions de trade_probability.py.
Exécution des trades : trading_utils.py ajuste le risque/levier et calcule le profit (méthode 5), en collaboration avec risk_manager.py.
Journalisation et alertes : algo_performance_logger.py enregistre les performances (cvar_loss, qr_dqn_quantiles, ensemble_weights), alert_manager.py envoie des alertes Telegram.
Maintenance des données : db_maintenance.py gère la mémoire contextuelle (méthode 7).

Diagrammes
(TODO : Inclure des diagrammes Mermaid ou ASCII pour visualiser les interactions entre modules dans une future mise à jour.)
Notes

Scalabilité : L’architecture modulaire permet d’ajouter de nouveaux algorithmes ou sources de données (ex. : API Bloomberg prévue pour juin 2025).
Robustesse : Les retries (max 3, délai 2^tentative secondes) et les sauvegardes incrémentielles/distribuées assurent la fiabilité.
Conformité : Suppression des références à 320 features, 81 features, et obs_t pour alignement avec les standards actuels.
Dépendances : Les nouveaux modules requièrent hmmlearn>=0.2.8,<0.3.0, stable-baselines3>=2.0.0,<3.0.0, et ray[rllib]>=2.0.0,<3.0.0 (voir requirements.txt).

Pour plus de détails, consultez les autres fichiers de documentation dans docs/ :

feature_engineering.md : Détails sur les fonctionnalités.
methodology.md : Explications des méthodes clés.
api_reference.md : Référence de l’API interne.
quickstart.md : Guide synthétique pour démarrer.
modules.md : Vue d’ensemble des modules.
risk_manager.md : Guide pour risk_manager.py.
regime_detector.md : Guide pour regime_detector.py.
trade_probability.md : Guide pour trade_probability.py.

Prochaines étapes

Vérifiez et supprimez le dossier src/model/policies s’il n’est pas utilisé pour éviter les conflits avec src/model/router/policies.
Préparez l’intégration de l’API Bloomberg (juin 2025).
Configurez et testez les nouveaux modules : risk_manager.py, regime_detector.py, trade_probability.py.

