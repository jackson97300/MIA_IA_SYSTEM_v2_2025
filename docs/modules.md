Modules de MIA_IA_SYSTEM_v2_2025
Version : 2.1.5Date : 2025-05-14  
Aperçu
Ce document fournit une vue d’ensemble des modules principaux de MIA_IA_SYSTEM_v2_2025, un système de trading avancé pour les contrats à terme E-mini S&P 500 (ES) et Micro E-mini Nasdaq-100 (MNQ). Le système traite les données de marché d’IQFeed via un pipeline de fonctionnalités robuste (350 fonctionnalités pour l’entraînement, 150 fonctionnalités sélectionnées par SHAP pour l’inférence) et exploite des modèles d’apprentissage automatique (SAC, PPO, DDPG, PPO-Lagrangian, QR-DQN), des réseaux neuronaux (LSTM, CNN), et une architecture modulaire pour exécuter des trades. Chaque module est conçu pour une fonctionnalité spécifique, de l’acquisition de données à l’exécution des trades, et est interconnecté pour garantir l’évolutivité et la robustesse. Les nouveaux modules incluent le dimensionnement dynamique des positions (risk_manager.py, suggestion 1), la détection des régimes de marché (regime_detector.py, suggestion 4), Safe RL/CVaR-PPO (trade_probability.py, suggestion 7), RL Distributionnel/QR-DQN (trade_probability.py, suggestion 8), et les ensembles de politiques avec vote bayésien (trade_probability.py, suggestion 10).
Note sur les dossiers policies : Le répertoire officiel pour les politiques de routage est src/model/router/policies. Le dossier src/model/policies semble être un résidu et doit être vérifié pour suppression afin d’éviter des conflits d’importation. Consultez troubleshooting.md pour résoudre les erreurs liées à des importations ambiguës.
Conformité : Toutes les références à dxFeed, obs_t, 320 features, et 81 features ont été supprimées pour aligner avec les spécifications de MIA_IA_SYSTEM_v2_2025.
Modules principaux
1. Acquisition de données : src/data/data_provider.py

Rôle : Récupère les données de marché en temps réel d’IQFeed pour les marchés ES et MNQ.
Fonctionnalités :
Récupère les données OHLC, profondeur de marché (DOM), options, données inter-marchés, et nouvelles.
Supporte les flux de données multi-marchés (ES, MNQ).
Implémente des tentatives de reconnexion (maximum 3, délai 2^tentative secondes).


Méthodes associées : Aucune directement, mais fournit les données brutes pour la génération de fonctionnalités.
Interactions :
Fournit des données à feature_pipeline.py, orderflow_indicators.py, et regime_detector.py.
Enregistre l’état de la connexion dans data/logs/<market>/data_provider.log.


Référence : Voir api_reference.md pour les détails de la classe DataProvider.

2. Pipeline de fonctionnalités : src/features/feature_pipeline.py

Rôle : Traite et génère 350 fonctionnalités pour l’entraînement et 150 fonctionnalités sélectionnées par SHAP pour l’inférence.
Fonctionnalités :
Agrège les données brutes en fonctionnalités (ex. : rsi_14, delta_volume, news_impact_score, atr_dynamic, orderflow_imbalance, hmm_state_distribution).
Intègre les sorties de news_analyzer.py, orderflow_indicators.py, spotgamma_recalculator.py, et regime_detector.py.
Stocke les définitions des fonctionnalités dans config/feature_sets.yaml.


Méthodes associées : Méthode 3 (pondération des fonctionnalités par régime, via obs_template.py).
Interactions :
Reçoit les données de data_provider.py.
Fournit les fonctionnalités à obs_template.py, trading_env.py, et trade_probability.py.


Référence : Voir feature_engineering.md pour les catégories de fonctionnalités et api_reference.md pour les fonctions.

3. Métriques d’ordre et d’options : src/features/orderflow_indicators.py et options_metrics.py

Rôle : Génère des fonctionnalités spécialisées pour l’analyse des flux d’ordres et des options.
Fonctionnalités :
orderflow_indicators.py : Calcule des métriques comme obi_score, absorption_strength, vix_es_correlation, et orderflow_imbalance (suggestion 1).
options_metrics.py : Calcule des fonctionnalités liées aux options comme net_gamma, call_iv_atm, et option_skew.
Supporte la Phase 13 (flux d’ordres) et la Phase 15 (recalculs d’options).


Méthodes associées : Aucune directement, mais soutient la méthode 3 (pondération des fonctionnalités).
Interactions :
Utilise les données de data_provider.py.
Fournit les fonctionnalités à feature_pipeline.py.


Référence : Voir api_reference.md pour les fonctions spécifiques.

4. Analyse de la microstructure : src/features/microstructure_guard.py et spotgamma_recalculator.py

Rôle : Analyse la microstructure du marché et recalcule les métriques d’options.
Fonctionnalités :
microstructure_guard.py : Détecte les anomalies comme spoofing_score, volume_anomaly, bid_ask_imbalance, trade_velocity, et hft_activity_score (Phase 15).
spotgamma_recalculator.py : Génère des métriques avancées d’options comme call_wall, zero_gamma, et key_strikes_1 (Phase 15).
Supporte la Phase 18 (métriques avancées comme trade_velocity, hft_activity_score).


Méthodes associées : Aucune directement, mais améliore la pertinence des fonctionnalités pour la méthode 3.
Interactions :
Traite les données de data_provider.py.
Fournit les fonctionnalités à feature_pipeline.py.


Référence : Voir feature_engineering.md pour les fonctionnalités de microstructure.

5. Formatage des observations : src/model/utils/obs_template.py

Rôle : Formate le vecteur d’observation (150 fonctionnalités SHAP) pour trading_env.py.
Fonctionnalités :
Sélectionne 150 fonctionnalités SHAP avec shap_weighting.py.
Applique une pondération basée sur les régimes (méthode 3) pour les marchés en tendance, range, et défensif, incluant hmm_state_distribution (suggestion 4).
Enregistre les observations dans data/features/<market>/obs_template.csv.


Méthodes associées : Méthode 3 (pondération des fonctionnalités).
Interactions :
Utilise les fonctionnalités de feature_pipeline.py.
Fournit les observations à trading_env.py et trade_probability.py.


Référence : Voir api_reference.md pour la classe ObsTemplate.

6. Environnement de trading : src/model/envs/trading_env.py

Rôle : Fournit un environnement basé sur Gymnasium pour l’entraînement et la simulation des trades.
Fonctionnalités :
Simule des scénarios de trading pour ES et MNQ en utilisant des vecteurs d’observation.
Implémente des récompenses adaptatives (méthode 5) basées sur les conditions de marché, incluant cvar_loss (suggestion 7).
Supporte les modes paper trading et trading en direct.


Méthodes associées : Méthode 5 (récompenses adaptatives).
Interactions :
Utilise les observations de obs_template.py.
Interagit avec train_sac.py, train_ppo_cvar.py, train_qr_dqn.py, et main_router.py pour l’entraînement et l’exécution.


Référence : Voir api_reference.md pour la classe TradingEnv.

7. Entraînement des modèles : src/model/rl/train_sac.py, train_ppo_cvar.py, train_qr_dqn.py, train_ensemble.py

Rôle : Entraîne les modèles SAC, PPO, DDPG, PPO-Lagrangian, QR-DQN, et les ensembles pour les décisions de trading.
Fonctionnalités :
Implémente l’ajustement fin (méthode 8), l’apprentissage en ligne (méthode 10), et le méta-apprentissage (méthode 18).
Supporte PPO-Lagrangian pour minimiser cvar_loss (suggestion 7) et QR-DQN pour qr_dqn_quantiles (suggestion 8).
Implémente le vote bayésien pour combiner SAC, PPO, DDPG avec ensemble_weights (suggestion 10) via train_ensemble.py.
Enregistre les métriques de performance dans data/logs/<market>/train_sac_performance.csv, train_ppo_cvar_performance.csv, train_qr_dqn_performance.csv.


Méthodes associées : Méthodes 8, 10, 18.
Interactions :
Utilise trading_env.py pour la simulation.
Enregistre via algo_performance_logger.py.
Interagit avec trade_probability.py pour les prédictions RL.


Référence : Voir api_reference.md pour les fonctions train_sac, train_ppo_cvar, train_qr_dqn, train_ensemble, et trade_probability.md.

8. Routage et détection des régimes : src/model/router/main_router.py et detect_regime.py

Rôle : Gère le routage des décisions de trading et la détection des régimes de marché.
Fonctionnalités :
main_router.py : Route les décisions en fonction des régimes (tendance, range, défensif).
detect_regime.py : Détecte les régimes de marché à l’aide de fonctionnalités comme atr_14, adx_14, et hmm_state_distribution (suggestion 4).
Stocke les politiques de routage dans src/model/router/policies/.


Méthodes associées : Aucune directement, mais soutient la méthode 3 (pondération basée sur les régimes).
Interactions :
Utilise les régimes de trading_utils.py et regime_detector.py.
Interagit avec trading_env.py et trade_probability.py pour l’exécution.


Référence : Voir api_reference.md pour les fonctions de routage et regime_detector.md.

9. Utilitaires de trading : src/model/utils/trading_utils.py

Rôle : Fournit des utilitaires pour la détection des régimes, l’ajustement des risques, et le calcul des profits.
Fonctionnalités :
Détecte les régimes de marché (méthode 3).
Ajuste les risques et le levier en fonction des régimes, utilisant atr_dynamic (suggestion 1).
Calcule les profits avec des récompenses adaptatives (méthode 5).


Méthodes associées : Méthodes 3, 5.
Interactions :
Utilise les fonctionnalités de feature_pipeline.py.
Supporte trading_env.py, main_router.py, et risk_manager.py.


Référence : Voir api_reference.md pour detect_market_regime, adjust_risk_and_leverage, calculate_profit, et risk_manager.md.

10. Mémoire contextuelle : src/model/utils/db_maintenance.py

Rôle : Gère la base de données de mémoire contextuelle (market_memory.db) pour les schémas historiques.
Fonctionnalités :
Stocke et récupère les données de marché (méthode 7), incluant hmm_state_distribution (suggestion 4).
Supprime les données antérieures à 30 jours.
Supporte les sauvegardes incrémentielles et distribuées (AWS S3).


Méthodes associées : Méthode 7 (mémoire contextuelle).
Interactions :
Stocke les données de feature_pipeline.py et regime_detector.py.
Fournit le contexte à trading_utils.py et trade_probability.py.


Référence : Voir api_reference.md pour la classe DBMaintenance.

11. Enregistrement des performances : src/model/utils/algo_performance_logger.py

Rôle : Enregistre les métriques de performance pour les modèles SAC, PPO, DDPG, PPO-Lagrangian, QR-DQN, et les ensembles.
Fonctionnalités :
Enregistre des métriques comme la récompense, la latence, la mémoire, finetune_loss, cvar_loss (suggestion 7), qr_dqn_quantiles (suggestion 8), et ensemble_weights (suggestion 10).
Génère des visualisations dans data/figures/<market>/.
Supporte les méthodes 8, 10, 18 (ajustement fin, apprentissage en ligne, méta-apprentissage).


Méthodes associées : Méthodes 8, 10, 18.
Interactions :
Utilisé par train_sac.py, train_ppo_cvar.py, train_qr_dqn.py, et train_ensemble.py.
Enregistre les données dans data/logs/<market>/train_sac_performance.csv, train_ppo_cvar_performance.csv, train_qr_dqn_performance.csv.


Référence : Voir api_reference.md pour la classe AlgoPerformanceLogger.

12. Alertes et notifications : src/model/utils/alert_manager.py et utils/telegram_alert.py

Rôle : Gère les alertes en temps réel via Telegram, SMS, et email.
Fonctionnalités :
Envoie des alertes priorisées (info, avertissement, erreur, urgent), incluant cvar_loss, qr_dqn_quantiles, et ensemble_weights (suggestions 7, 8, 10).
Intègre Telegram via telegram_alert.py.
Enregistre les alertes dans data/logs/<market>/alert_manager.log.


Méthodes associées : Aucune directement, mais soutient tous les modules.
Interactions :
Utilisé par tous les modules pour signaler les erreurs.
Configuré via es_config.yaml.


Référence : Voir api_reference.md pour la classe AlertManager.

13. Tableau de bord : src/model/utils/mia_dashboard.py

Rôle : Visualise les performances de trading et les métriques du système en temps réel.
Fonctionnalités :
Affiche les métriques de performance, les régimes de marché (hmm_state_distribution, suggestion 4), l’importance des fonctionnalités SHAP, cvar_loss (suggestion 7), qr_dqn_quantiles (suggestion 8), et ensemble_weights (suggestion 10).
Génère des graphiques enregistrés dans data/figures/<market>/.
Supporte la visualisation multi-marchés (ES, MNQ).


Méthodes associées : Aucune directement, mais visualise les sorties de la méthode 3 (pondération des régimes).
Interactions :
Utilise les données de algo_performance_logger.py, obs_template.py, et trade_probability.py.
Interagit avec feature_pipeline.py pour l’importance des fonctionnalités.


Référence : Voir api_reference.md pour les fonctions du tableau de bord.

14. Interface vocale : src/model/utils/mind_dialogue.py et mind_voice.py

Rôle : Active les interactions vocales pour le trading et les requêtes système.
Fonctionnalités :
Traite les commandes vocales et génère des réponses.
Intègre avec alert_manager.py pour des alertes vocales, incluant cvar_loss et qr_dqn_quantiles.
Supporte les commandes multi-marchés (ES, MNQ).


Méthodes associées : Aucune directement, mais améliore l’interaction utilisateur.
Interactions :
Interface avec main_router.py pour l’exécution des trades.
Enregistre les interactions dans data/logs/<market>/mind_dialogue.log.


Référence : Voir api_reference.md pour les fonctions de l’interface vocale.

15. Gestion des risques : src/risk_management/risk_manager.py

Rôle : Gère le dimensionnement dynamique des positions pour minimiser les risques (suggestion 1).
Fonctionnalités :
Calcule la taille des positions en fonction de atr_dynamic et orderflow_imbalance.
Ajuste le levier selon les régimes (ex. : 5x en tendance, 1x en défensif).
Intègre avec trading_utils.py pour les calculs de risque/profit.


Méthodes associées : Méthode 5 (récompenses adaptatives).
Interactions :
Utilise les fonctionnalités de feature_pipeline.py et regime_detector.py.
Fournit des données à trading_env.py et main_router.py.
Enregistre les performances dans data/logs/<market>/risk_manager_performance.csv.


Référence : Voir api_reference.md pour la classe RiskManager et risk_manager.md.

16. Détection des régimes : src/features/regime_detector.py

Rôle : Détecte les régimes de marché (tendance, range, défensif) à l’aide d’un modèle HMM (suggestion 4).
Fonctionnalités :
Entraîne un modèle HMM avec n_components=3 pour générer hmm_state_distribution.
Calcule les matrices de transition enregistrées dans data/logs/<market>/hmm_transitions.csv.
Supporte la Phase 3 (pondération des régimes) et la Phase 15 (microstructure).


Méthodes associées : Méthode 3 (pondération basée sur les régimes).
Interactions :
Utilise les données de data_provider.py et feature_pipeline.py.
Fournit hmm_state_distribution à obs_template.py, trading_utils.py, et trade_probability.py.
Enregistre les performances dans data/logs/<market>/regime_detector_performance.csv.


Référence : Voir api_reference.md pour la classe RegimeDetector et regime_detector.md.

17. Prédiction des probabilités de trading : src/model/trade_probability.py

Rôle : Prédit la probabilité de succès des trades à l’aide de modèles RL (suggestions 7, 8, 10).
Fonctionnalités :
Utilise PPO-Lagrangian pour minimiser cvar_loss (suggestion 7).
Utilise QR-DQN pour modéliser qr_dqn_quantiles (suggestion 8).
Implémente le vote bayésien pour combiner SAC, PPO, DDPG avec ensemble_weights (suggestion 10).
Supporte la prise de décision en temps réel pour ES et MNQ.


Méthodes associées : Méthodes 8, 10, 18 (ajustement fin, apprentissage en ligne, méta-apprentissage).
Interactions :
Utilise les observations de obs_template.py et les régimes de regime_detector.py.
Fournit des probabilités à main_router.py et trading_env.py.
Enregistre les performances dans data/logs/<market>/trade_probability.log.


Référence : Voir api_reference.md pour la classe TradeProbabilityPredictor et trade_probability.md.

Notes

Conformité : Toutes les références à dxFeed, obs_t, 320 features, et 81 features ont été supprimées pour aligner avec les spécifications de MIA_IA_SYSTEM_v2_2025.
Évolutivité : Les modules sont conçus pour supporter des instruments supplémentaires (ex. : NQ, DAX, cryptomonnaies) d’ici 2026-2027 (Phases 16, 17).
Dépendances : Les nouveaux modules requièrent des dépendances spécifiques, telles que hmmlearn>=0.2.8,<0.3.0, stable-baselines3>=2.0.0,<3.0.0, et ray[rllib]>=2.0.0,<3.0.0 (voir requirements.txt).
Documentation : La documentation détaillée de l’API est disponible dans api_reference.md, le contexte architectural dans architecture.md, et les guides spécifiques dans risk_manager.md, regime_detector.md, trade_probability.md.
Intégration future : L’intégration de l’API Bloomberg (juin 2025) améliorera les capacités de data_provider.py.

Prochaines étapes

Explorer les modules : Consultez api_reference.md pour la documentation détaillée des fonctions et classes.
Exécuter les tests : Validez les fonctionnalités des modules avec :pytest tests/ -v

Tests clés : test_risk_manager.py, test_regime_detector.py, test_trade_probability.py.
Configurer le système : Suivez quickstart.md ou installation.md pour configurer le système.
Résoudre les problèmes : Référez-vous à troubleshooting.md pour les solutions aux erreurs courantes.
Suivre la feuille de route : Consultez roadmap.md pour les fonctionnalités à venir (ex. : NQ/DAX en 2026, cryptomonnaies en 2027).

En cas de problèmes, vérifiez data/logs/<market>/ (ex. : risk_manager.log, regime_detector.log, trade_probability.log) et consultez troubleshooting.md.
