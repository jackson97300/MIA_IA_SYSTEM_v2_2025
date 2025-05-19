Guide d'utilisation de MIA_IA_SYSTEM_v2_2025
Version : 2.1.5Date : 2025-05-14  
Introduction
Ce guide fournit des instructions pratiques pour utiliser MIA_IA_SYSTEM_v2_2025, un système de trading avancé pour les contrats à terme E-mini S&P 500 (ES) et Micro E-mini Nasdaq-100 (MNQ). Le système exploite des modèles d'apprentissage automatique (SAC, PPO, DDPG), des réseaux neuronaux (LSTM, CNN), et un pipeline de traitement de 350 fonctionnalités pour l'entraînement et 150 fonctionnalités sélectionnées par SHAP pour l'inférence, en utilisant les données IQFeed. Il inclut des modules avancés pour le dimensionnement des positions (risk_manager.py, suggestion 1), la détection de régimes (regime_detector.py, suggestion 4), le Safe RL/CVaR-PPO (trade_probability.py, suggestion 7), le RL Distributionnel/QR-DQN (trade_probability.py, suggestion 8), et le vote d'ensemble de politiques (trade_probability.py, suggestion 10). Ce document couvre l'exécution des trades (papier ou live), l'interaction avec l'interface vocale, la navigation dans le tableau de bord, et la surveillance des performances, permettant aux traders, développeurs, et opérateurs de maximiser les capacités du système.
Note sur les dossiers policies : Le dossier officiel pour les politiques de routage est src/model/router/policies. Le dossier src/model/policies semble être un résidu et doit être vérifié pour suppression afin d'éviter des conflits d'importation.
Intégration future : Le système se prépare à intégrer l'API Bloomberg en juin 2025, ce qui améliorera les entrées de données pour le tableau de bord et les capacités de trading (voir roadmap.md).
Pour un démarrage rapide, consultez quickstart.md. Pour la configuration et l'installation, voir setup.md et installation.md. Pour les détails des modules, reportez-vous à modules.md et api_reference.md.
Prérequis
Avant d'utiliser le système, assurez-vous qu'il est entièrement configuré comme décrit dans setup.md et installation.md. Les prérequis clés incluent :

Configuration de es_config.yaml avec les paramètres IQFeed, NewsAPI, AWS S3, et Telegram.
Initialisation de market_memory.db via db_maintenance.py.
Vérification de la connexion IQFeed et des tests du système (pytest tests/ -v).

Exécution du système
1. Mode trading papier
Lancez le système en mode trading papier pour simuler des trades sans capital réel :
python src/run_system.py --paper --market ES

Options :

--market : Spécifiez ES ou MNQ.
--log-level : Définissez le niveau de journalisation (ex. : INFO, DEBUG). Par défaut : INFO.
--model-type : Spécifiez le modèle pour trade-probability (ex. : ppo_cvar, qr_dqn, ensemble). Par défaut : ppo_cvar.

Sortie :

Journaux dans data/logs/<market>/run_system.log.
Instantanés dans data/cache/<module>/<market>/*.json.gz.
Visualisations dans data/figures/<market>/*.png.

Exemple :
python src/run_system.py --paper --market MNQ --log-level DEBUG --model-type ensemble

2. Mode trading live
Lancez le système en mode trading live (assurez-vous d'avoir une gestion des risques et une configuration de compte appropriées) :
python src/run_system.py --market ES --model-type ppo_cvar

Avertissement : Le trading live exécute de vrais trades. Vérifiez les paramètres du compte dans es_config.yaml et testez d'abord en mode papier.
Configuration :Mettez à jour les paramètres de trading dans es_config.yaml :
trading_utils:
  max_leverage: 5.0
  risk_per_trade: 0.01  # 1% du compte par trade
risk_manager:
  atr_threshold: 100.0
  orderflow_imbalance_limit: 0.9
trade_probability:
  model_type: "ppo_cvar"  # Options : ppo_cvar, qr_dqn, ensemble
  cvar_alpha: 0.95
  quantiles: 51
  ensemble_weights: "0.4,0.3,0.3"  # SAC, PPO, DDPG

Sortie :

Identique au mode papier, avec des journaux supplémentaires d'exécution des trades dans data/logs/<market>/trading_utils.log.

Exemple :
python src/run_system.py --market MNQ --model-type qr_dqn

3. Arrêt du système
Arrêtez le système proprement pour sauvegarder les instantanés sur SIGINT :
Ctrl+C

Vérifiez les instantanés sauvegardés :
ls data/cache/<module>/<market>/sigint_*.json.gz

Utilisation de l'interface vocale
L'interface vocale, alimentée par mind_dialogue.py et mind_voice.py, permet aux utilisateurs de donner des commandes et de recevoir des réponses pour les trades et les requêtes système.
1. Lancement de l'interface vocale
Lancez l'interface vocale :
python src/model/utils/mind_dialogue.py --market ES

Options :

--market : ES ou MNQ.
--voice-profile : Style de voix (ex. : calm, urgent). Par défaut : calm.

2. Commandes vocales
Commandes de trading :

"Placer un trade long sur ES avec 2 contrats" : Exécute un trade long avec la taille spécifiée.
"Clôturer toutes les positions sur MNQ" : Clôture les positions ouvertes pour MNQ.
"Définir la taille de position à 1% de risque sur ES" : Ajuste la taille des positions via risk_manager.py.

Requêtes système :

"Quel est le régime de marché actuel ?" : Retourne le régime (tendance, range, défensif) depuis regime_detector.py.
"Afficher les métriques de performance récentes" : Vocalise la récompense, la latence, et l'utilisation mémoire.
"Lister les principales fonctionnalités SHAP" : Décrit les 5 principales fonctionnalités SHAP depuis obs_template.py.
"Quelle est la perte CVaR ?" : Rapporte cvar_loss pour PPO-Lagrangian depuis trade_probability.py.
"Quels sont les quantiles QR-DQN ?" : Rapporte qr_dqn_quantiles depuis trade_probability.py.
"Afficher les poids de l'ensemble" : Rapporte ensemble_weight_sac, ensemble_weight_ppo, ensemble_weight_ddpg depuis trade_probability.py.

Exemple :

Dites : "Placer un trade papier long sur ES avec 1 contrat."
Réponse : "Trade papier placé : long 1 contrat sur ES à [prix]."

3. Alertes vocales
Les alertes sont vocalisées pour les événements critiques (ex. : utilisation mémoire élevée, erreurs de trade). Configurez les paramètres des alertes dans es_config.yaml :
alert_manager:
  voice_alerts: true
  priority_threshold: 3  # Alertes pour priorité >= 3 (erreur, urgent)

Navigation dans le tableau de bord
Le tableau de bord (mia_dashboard.py) visualise en temps réel les performances de trading, les régimes de marché, et l'importance des fonctionnalités SHAP.
1. Lancement du tableau de bord
Lancez le tableau de bord :
python src/model/utils/mia_dashboard.py --market ES

Options :

--market : ES ou MNQ.
--refresh-rate : Intervalle de mise à jour en secondes (par défaut : 60).

2. Fonctionnalités du tableau de bord
Métriques de performance :

Affiche la récompense cumulée, la latence moyenne, et l'utilisation mémoire.
Graphiques sauvegardés dans data/figures/<market>/performance_*.png.

Régimes de marché :

Visualise les transitions de régimes (tendance, range, défensif) depuis regime_detector.py.
Affiche hmm_state_distribution (suggestion 4).
Graphiques sauvegardés dans data/figures/<market>/regime_*.png.

Importance des fonctionnalités SHAP :

Affiche les 10 principales fonctionnalités SHAP (ex. : obi_score, net_gamma, atr_dynamic, orderflow_imbalance).
Graphiques sauvegardés dans data/figures/<market>/shap_*.png.

Prédictions LSTM :

Affiche les prédictions comme predicted_vix depuis neural_pipeline.py.
Graphiques sauvegardés dans data/figures/<market>/prediction_*.png.

Métriques avancées :

Perte CVaR : Visualise cvar_loss pour PPO-Lagrangian (suggestion 7).
Quantiles QR-DQN : Visualise qr_dqn_quantiles (suggestion 8).
Poids de l'ensemble : Visualise ensemble_weight_sac, ensemble_weight_ppo, ensemble_weight_ddpg (suggestion 10).
Graphiques sauvegardés dans data/figures/<market>/advanced_*.png.

3. Accès aux graphiques
Visualisez les graphiques générés :
ls data/figures/<market>/

Ouvrez les graphiques dans un visualiseur (ex. : visionneuse d'images par défaut ou navigateur).
Surveillance et maintenance
1. Vérification des journaux
Surveillez l'activité du système via les journaux :
cat data/logs/<market>/<module>.log

Journaux clés :

run_system.log : Activité générale du système.
trading_utils.log : Détails d'exécution des trades.
alert_manager.log : Notifications d'alertes.
train_sac.log : Métriques d'entraînement des modèles.
risk_manager.log : Ajustements de dimensionnement des positions (suggestion 1).
regime_detector.log : Événements de détection de régimes (suggestion 4).
trade_probability.log : Détails d'exécution des modèles (PPO-Lagrangian, QR-DQN, ensemble, suggestions 7, 8, 10).

2. Gestion des alertes
Alertes Telegram : Recevez des notifications en temps réel configurées dans es_config.yaml. Testez les alertes :
python src/utils/telegram_alert.py --test

Alertes SMS/Email : Configurez des canaux supplémentaires dans es_config.yaml :
alert_manager:
  sms_enabled: true
  email_enabled: true
  twilio_sid: "votre_twilio_sid"
  twilio_token: "votre_twilio_token"
  email_smtp: "smtp.exemple.com"

3. Surveillance des sauvegardes
Sauvegardes incrémentielles : Vérifiez les instantanés et points de contrôle :
ls data/cache/<module>/<market>/*.json.gz
ls data/checkpoints/<module>/<market>/*.json.gz

Sauvegardes distribuées (AWS S3) : Vérifiez les téléversements S3 :
aws s3 ls s3://votre-s3-bucket/mia_ia_system/

En cas d'échec des sauvegardes, consultez troubleshooting.md pour les erreurs comme OSError: Échec du téléversement vers S3.
4. Mise à jour des modèles
Réentraînez les modèles périodiquement pour s'adapter aux nouvelles conditions de marché :
python src/model/rl/train_sac.py --market ES --retrain
python src/model/rl/train_ppo_cvar.py --market ES --retrain  # Suggestion 7
python src/model/rl/train_qr_dqn.py --market ES --retrain  # Suggestion 8
python src/model/rl/train_ensemble.py --market ES --retrain  # Suggestion 10

Surveillez la progression de l'entraînement :
cat data/logs/<market>/train_*.log

Notes

Conformité : Toutes les références à dxFeed, obs_t, 320 features, et 81 features ont été supprimées pour aligner avec les spécifications de MIA_IA_SYSTEM_v2_2025.
Évolutivité : Le système prend en charge ES et MNQ, avec des plans pour NQ, DAX, et les cryptomonnaies d'ici 2026-2027 (Phases 16, 17, voir phases.md).
Intégration future : L'intégration de l'API Bloomberg (juin 2025) améliorera les entrées de données, renforçant le tableau de bord et les capacités de trading (voir roadmap.md).
Documentation : Ce guide complète :
quickstart.md : Instructions de démarrage rapide.
installation.md : Guide d'installation complet.
setup.md : Préparation de l'environnement.
architecture.md : Vue d'ensemble du système.
phases.md : Phases de développement (1-18).
troubleshooting.md : Solutions aux problèmes courants.
modules.md : Descriptions détaillées des modules.
api_reference.md : Documentation de l'API.
roadmap.md : Fonctionnalités futures et calendrier.



Prochaines étapes

Exécuter les tests : Validez la fonctionnalité du système :pytest tests/ -v


Explorer les modules : Consultez modules.md et api_reference.md pour une documentation détaillée sur risk_manager.py, regime_detector.py, trade_probability.py, et autres modules.
Approfondir la compréhension : Reportez-vous à architecture.md pour une vue d'ensemble du système et à methodology.md pour les méthodes (3, 5, 7, 8, 10, 18).
Résoudre les problèmes : Consultez troubleshooting.md pour des solutions aux problèmes (ex. : erreurs de l'interface vocale, échecs du tableau de bord).
Suivre la feuille de route : Vérifiez roadmap.md pour les fonctionnalités à venir (ex. : NQ/DAX en 2026, cryptomonnaies en 2027).

