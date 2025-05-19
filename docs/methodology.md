Méthodologie de MIA_IA_SYSTEM_v2_2025
Version : 2.1.5Date : 2025-05-14  
Aperçu
Ce document décrit les méthodes clés utilisées dans MIA_IA_SYSTEM_v2_2025, un système avancé de trading algorithmique pour les contrats à terme E-mini S&P 500 (ES) et Micro E-mini Nasdaq-100 (MNQ). Ces méthodes permettent au système d’optimiser les décisions de trading en fonction des régimes de marché (tendance, range, défensif), en s’appuyant sur des données provenant exclusivement d’IQFeed via data_provider.py. Les méthodes incluent la pondération des features, les récompenses adaptatives, la mémoire contextuelle, le fine-tuning, l’apprentissage en ligne, et le meta-learning, enrichies par le dimensionnement dynamique des positions (suggestion 1), la détection des régimes avec HMM (suggestion 4), Safe RL/CVaR-PPO (suggestion 7), RL Distributionnel/QR-DQN (suggestion 8), et les ensembles de politiques avec vote bayésien (suggestion 10). Elles sont implémentées dans divers modules, notamment trading_utils.py, obs_template.py, train_sac.py, db_maintenance.py, risk_manager.py, regime_detector.py, et trade_probability.py.
Note sur les dossiers policies : Le répertoire officiel pour les politiques de routage est src/model/router/policies. Le dossier src/model/policies semble être un résidu et doit être vérifié pour suppression afin d’éviter des conflits d’importation. Consultez troubleshooting.md pour résoudre les erreurs liées à des importations ambiguës.
Conformité : Toutes les références à dxFeed, obs_t, 320 features, et 81 features ont été supprimées pour aligner avec les spécifications de MIA_IA_SYSTEM_v2_2025.
Méthodes clés
Méthode 3 : Pondération des features selon le régime
Description : Cette méthode ajuste les poids des 150 fonctionnalités SHAP utilisées pour l’inférence en fonction du régime de marché (tendance, range, défensif), optimisant la pertinence des observations pour trading_env.py.
Implémentation :

Modules : obs_template.py (méthode create_observation), regime_detector.py (suggestion 4).
Processus :
Les poids sont définis dans config/es_config.yaml sous obs_template.weights.
Les fonctionnalités critiques (ex. : mm_score, hft_score, breakout_score, hmm_state_distribution) reçoivent des poids spécifiques :
Tendance : mm_score (1.5), hft_score (1.5), breakout_score (1.0), hmm_state_distribution (1.2).
Range : breakout_score (1.5), mm_score (1.0), hft_score (1.0), hmm_state_distribution (1.0).
Défensif : mm_score (0.5), hft_score (0.5), breakout_score (0.5), hmm_state_distribution (0.8).


Les poids sont appliqués au vecteur d’observation après la sélection des 150 fonctionnalités SHAP via shap_weighting.py.
regime_detector.py fournit hmm_state_distribution pour affiner la détection des régimes (suggestion 4).


Impact : Améliore la précision des décisions en priorisant les fonctionnalités pertinentes pour chaque régime, avec une détection plus robuste grâce à HMM.

Exemple :

Dans un marché en tendance, mm_score (mémoire contextuelle) et hmm_state_distribution (probabilité de tendance HMM) sont amplifiés pour capturer les dynamiques persistantes.
Dans un marché en range, breakout_score est priorisé pour détecter les ruptures potentielles.

Référence : Voir feature_engineering.md pour les détails des fonctionnalités, regime_detector.md pour HMM, et api_reference.md pour create_observation.
Méthode 5 : Récompenses adaptatives
Description : Cette méthode calcule des récompenses dynamiques pour les algorithmes d’apprentissage par renforcement (SAC, PPO, DDPG, PPO-Lagrangian, QR-DQN) en tenant compte des conditions de marché, comme l’impact des nouvelles, la volatilité (VIX), et les métriques de risque.
Implémentation :

Modules : trading_utils.py (méthode calculate_profit), risk_manager.py (suggestion 1).
Processus :
Le profit brut est calculé comme : (exit_price - entry_price) * position_size pour les trades long, ou (entry_price - exit_price) * position_size pour les trades short.
Les coûts de transaction sont déduits (par défaut : 2,0 USD par unité de position).
Ajustements adaptatifs :
Impact des nouvelles : Si news_impact_score > 0,5, le profit est multiplié par 1,2 (bonus). Si < -0,5, multiplié par 0,8 (pénalité).
Volatilité (VIX) : Si vix > 20, le profit est multiplié par 0,9 (pénalité pour volatilité élevée).
Risque dynamique : Si atr_dynamic > seuil (via risk_manager.py, suggestion 1), le profit est multiplié par 0,85 (pénalité). Si orderflow_imbalance > limite, multiplié par 1,1 (bonus).


Intégration de cvar_loss (suggestion 7) pour pénaliser les pertes extrêmes dans PPO-Lagrangian.


Impact : Aligne les récompenses sur les conditions de marché et les risques, favorisant les trades dans des contextes favorables tout en minimisant les pertes.

Exemple :

Un trade long avec un profit brut de 100 USD, un news_impact_score de 0,6, un VIX de 18, un atr_dynamic dans la norme, et un orderflow_imbalance favorable donnera une récompense ajustée de : 100 * 1,2 * 1,1 = 132 USD.
Si cvar_loss > 0,1, la récompense est réduite pour refléter le risque.

Référence : Voir api_reference.md pour calculate_profit, risk_manager.md pour atr_dynamic, et trade_probability.md pour cvar_loss.
Méthode 7 : Mémoire contextuelle
Description : Cette méthode utilise une base de données SQLite (market_memory.db) pour stocker et récupérer des données contextuelles, permettant au système de mémoriser les dynamiques historiques du marché.
Implémentation :

Modules : db_maintenance.py, regime_detector.py (suggestion 4).
Processus :
Stocke les fonctionnalités calculées (ex. : mm_score, event_volatility_impact, hmm_state_distribution) dans market_memory.db.
Purge les données obsolètes (>30 jours) pour optimiser l’espace.
Fournit des fonctionnalités contextuelles à trading_env.py via trading_utils.py.
Sauvegardes incrémentielles dans data/checkpoints/db_maintenance/<market>/*.json.gz (toutes les 5 minutes, 5 versions).
Sauvegardes distribuées vers AWS S3 (toutes les 15 minutes).
regime_detector.py enrichit la base avec hmm_state_distribution pour les régimes HMM (suggestion 4).


Impact : Améliore la prise de décision en intégrant des informations historiques pertinentes, renforcées par une détection précise des régimes.

Exemple :

Une augmentation soudaine de delta_volume est contextualisée par des événements passés similaires stockés dans market_memory.db, avec hmm_state_distribution indiquant un régime de tendance.

Référence : Voir architecture.md pour les interactions avec db_maintenance.py et regime_detector.md pour HMM.
Méthode 8 : Fine-tuning des modèles
Description : Cette méthode ajuste les hyperparamètres des modèles SAC, PPO, DDPG, PPO-Lagrangian, et QR-DQN pour améliorer leurs performances dans des régimes de marché spécifiques.
Implémentation :

Modules : train_sac.py, train_ppo_cvar.py, train_qr_dqn.py, trade_probability.py (suggestions 7, 8).
Processus :
Utilise des métriques de perte (finetune_loss, cvar_loss pour PPO-Lagrangian, suggestion 7) pour ajuster les poids des réseaux neuronaux.
Suit qr_dqn_quantiles pour QR-DQN (suggestion 8) afin d’optimiser la distribution des rendements.
Enregistre les performances dans data/logs/<market>/train_sac_performance.csv, train_ppo_cvar_performance.csv, train_qr_dqn_performance.csv via algo_performance_logger.py.
Applique des techniques comme l’optimisation bayésienne pour explorer l’espace des hyperparamètres.


Impact : Réduit la perte et améliore la convergence des modèles dans des conditions de marché dynamiques, avec une gestion robuste des risques extrêmes.

Exemple :

Dans un marché volatile (VIX > 20), le fine-tuning de PPO-Lagrangian réduit cvar_loss en ajustant le taux d’apprentissage, tandis que QR-DQN optimise qr_dqn_quantiles pour capturer les rendements incertains.

Référence : Voir api_reference.md pour algo_performance_logger.py, trade_probability.md pour PPO-Lagrangian et QR-DQN.
Méthode 10 : Apprentissage en ligne
Description : Cette méthode permet aux modèles d’apprendre en temps réel à partir de nouvelles données de marché, adaptant leurs politiques sans réentraînement complet.
Implémentation :

Modules : train_sac.py, train_ppo_cvar.py, train_qr_dqn.py, train_ensemble.py, trade_probability.py (suggestions 7, 8, 10).
Processus :
Mise à jour incrémentielle des poids des modèles en fonction des observations récentes, incluant cvar_loss (suggestion 7) et qr_dqn_quantiles (suggestion 8).
Intégration du vote bayésien via train_ensemble.py pour combiner SAC, PPO, DDPG avec ensemble_weights (suggestion 10).
Nombre d’étapes d’apprentissage en ligne (online_learning_steps) et ensemble_weights enregistrés dans train_sac_performance.csv, train_ppo_cvar_performance.csv, train_qr_dqn_performance.csv.
Utilise obs_template.py pour fournir des vecteurs d’observation actualisés.


Impact : Permet une adaptation rapide aux changements de régime de marché, avec une prise de décision robuste via le vote bayésien.

Exemple :

Une transition soudaine vers un régime défensif déclenche des mises à jour en ligne pour réduire le levier, avec ensemble_weights ajustés pour prioriser PPO-Lagrangian si cvar_loss est élevé.

Référence : Voir api_reference.md pour train_sac.py, train_ppo_cvar.py, train_qr_dqn.py, train_ensemble.py, et trade_probability.md.
Méthode 18 : Meta-learning
Description : Cette méthode utilise des techniques de meta-learning pour permettre aux modèles d’apprendre à apprendre, améliorant leur capacité à s’adapter à de nouveaux régimes de marché avec peu de données.
Implémentation :

Modules : train_sac.py, train_ppo_cvar.py, train_qr_dqn.py, train_ensemble.py, trade_probability.py (suggestions 7, 8, 10).
Processus :
Implémente MAML (Model-Agnostic Meta-Learning) pour initialiser les modèles avec des poids généralisés.
Intègre cvar_loss (suggestion 7), qr_dqn_quantiles (suggestion 8), et ensemble_weights (suggestion 10) pour optimiser les tâches d’apprentissage.
Nombre d’étapes de meta-learning (maml_steps) enregistré dans train_sac_performance.csv, train_ppo_cvar_performance.csv, train_qr_dqn_performance.csv.
Utilise des tâches multiples (ex. : différents régimes de marché) pour entraîner la généralisation.


Impact : Accélère l’adaptation des modèles à des conditions de marché inédites, avec une robustesse accrue via le vote bayésien.

Exemple :

Un modèle meta-entraîné peut s’adapter à un nouveau régime de marché en quelques itérations, optimisant cvar_loss pour PPO-Lagrangian et ajustant ensemble_weights pour équilibrer SAC, PPO, DDPG.

Référence : Voir api_reference.md pour algo_performance_logger.py, trade_probability.md pour PPO-Lagrangian, QR-DQN, et vote bayésien.
Intégration des méthodes
Les méthodes sont interconnectées pour former un pipeline cohérent :

Méthode 3 (pondération) génère des vecteurs d’observation optimisés pour trading_env.py, enrichis par hmm_state_distribution (suggestion 4).
Méthode 5 (récompenses) fournit des signaux de récompense adaptés aux conditions de marché, ajustés par atr_dynamic et orderflow_imbalance (suggestion 1) et cvar_loss (suggestion 7).
Méthode 7 (mémoire) enrichit les observations avec un contexte historique, incluant hmm_state_distribution.
Méthodes 8, 10, 18 (fine-tuning, apprentissage en ligne, meta-learning) ajustent les modèles en temps réel pour maximiser les performances, en intégrant cvar_loss, qr_dqn_quantiles, et ensemble_weights (suggestions 7, 8, 10).

Validation
Chaque méthode est validée via :

Tests unitaires :
tests/test_trading_utils.py : Valide calculate_profit et les ajustements de récompense.
tests/test_obs_template.py : Vérifie la pondération des fonctionnalités.
tests/test_algo_performance_logger.py : Teste l’enregistrement des performances.
tests/test_risk_manager.py : Valide atr_dynamic et orderflow_imbalance (suggestion 1).
tests/test_regime_detector.py : Vérifie hmm_state_distribution (suggestion 4).
tests/test_trade_probability.py : Teste cvar_loss, qr_dqn_quantiles, et ensemble_weights (suggestions 7, 8, 10).


Journalisation : Performances enregistrées dans data/logs/<market>/ (ex. : train_sac_performance.csv, risk_manager_performance.csv, regime_detector_performance.csv, trade_probability.log).
Alertes : Notifications via Telegram pour les erreurs critiques, incluant cvar_loss élevé ou ensemble_weights déséquilibrés, implémentées dans alert_manager.py.

Notes

Conformité : Suppression des références à dxFeed, obs_t, 320 features, et 81 features pour alignement avec les standards actuels.
Évolutivité : Les méthodes sont conçues pour s’adapter à de nouvelles sources de données (ex. : API Bloomberg, prévue pour juin 2025) et à des instruments futurs (NQ, DAX, cryptomonnaies).
Dépendances : Les nouveaux modules requièrent hmmlearn>=0.2.8,<0.3.0, stable-baselines3>=2.0.0,<3.0.0, et ray[rllib]>=2.0.0,<3.0.0 (voir requirements.txt).
Documentation : Consultez architecture.md pour l’architecture globale, feature_engineering.md pour les fonctionnalités, api_reference.md pour les fonctions, risk_manager.md, regime_detector.md, et trade_probability.md pour les nouveaux modules.

Prochaines étapes

Vérifier les méthodes : Consultez api_reference.md pour les détails des fonctions implémentant ces méthodes.
Exécuter les tests : Validez les méthodes avec :pytest tests/ -v

Tests clés : test_risk_manager.py, test_regime_detector.py, test_trade_probability.py.
Configurer le système : Suivez quickstart.md ou installation.md pour configurer le système.
Résoudre les problèmes : Référez-vous à troubleshooting.md pour les solutions aux erreurs.
Suivre la feuille de route : Consultez roadmap.md pour les jalons futurs (ex. : Bloomberg en 2025, NQ/DAX en 2026).

En cas de problèmes, vérifiez data/logs/<market>/ (ex. : risk_manager.log, regime_detector.log, trade_probability.log) et consultez troubleshooting.md.
