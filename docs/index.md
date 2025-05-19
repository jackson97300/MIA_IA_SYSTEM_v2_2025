Bienvenue dans la documentation de MIA_IA_SYSTEM_v2_2025
MIA_IA_SYSTEM_v2_2025 est un système de trading automatisé avancé conçu pour les contrats à terme E-mini S&P 500 (ES) et Micro E-mini Nasdaq-100 (MNQ), doté d’une interface vocale interactive, d’un apprentissage adaptatif, et d’un tableau de bord interactif pour la surveillance en temps réel. Le système exploite des modèles d’apprentissage automatique (SAC, PPO, DDPG, PPO-Lagrangian, QR-DQN), des réseaux neuronaux (LSTM, CNN), et un pipeline de fonctionnalités robuste pour traiter les données de marché et exécuter des trades.
Version : 2.1.5Dernière mise à jour : 2025-05-14
Fonctionnalités clés

Trading automatisé : Exécute des trades basés sur des données de marché en temps réel, utilisant 350 fonctionnalités pour l’entraînement et 150 fonctionnalités SHAP pour l’inférence, avec prise en charge des multi-marchés (ES, MNQ).
Interface vocale interactive : Supporte les commandes vocales et les réponses pour les opérations de trading et les requêtes système, alimenté par mind_dialogue.py et mind_voice.py.
Apprentissage adaptatif : Apprend en continu des schémas de marché, stockés dans market_memory.db, et réentraîne les modèles avec adaptive_learning.py. Inclut le dimensionnement dynamique des positions (risk_manager.py, suggestion 1), la détection des régimes HMM (regime_detector.py, suggestion 4), Safe RL/CVaR-PPO et RL distributionnel/QR-DQN (trade_probability.py, suggestions 7, 8), et le vote bayésien pour les ensembles de politiques (trade_probability.py, suggestion 10).
Tableau de bord interactif : Visualise les performances de trading, les régimes de marché, l’importance des fonctionnalités SHAP, et les prédictions LSTM via mia_dashboard.py, avec des métriques comme hmm_state_distribution, cvar_loss, qr_dqn_quantiles, et ensemble_weights.
Architecture robuste : Implémente des tentatives exponentielles (max 3, délai 2^tentative), journalisation des performances avec psutil, alertes via alert_manager.py (Telegram, SMS, email), instantanés JSON, gestion SIGINT, sauvegardes incrémentielles (cache, base de données), et sauvegardes distribuées (AWS S3).

Notes système

Source de données : Utilise exclusivement IQFeed pour les données de marché (OHLC, DOM, options, inter-marchés, nouvelles), implémenté dans data_provider.py.
Exclusions : Toutes les références à dxFeed, obs_t, 320 features, et 81 features ont été supprimées pour assurer la conformité avec les spécifications de MIA_IA_SYSTEM_v2_2025.
Configuration : Gérée via des fichiers YAML (es_config.yaml, market_config.yaml, feature_sets.yaml, risk_manager_config.yaml, regime_detector_config.yaml, trade_probability_config.yaml) en utilisant config_manager.py.
Tests : Tests unitaires complets disponibles dans le répertoire tests/ (ex. : test_feature_pipeline.py, test_mia_dashboard.py, test_trading_utils.py, test_risk_manager.py, test_regime_detector.py, test_trade_probability.py).
Note sur les politiques : Le répertoire officiel pour les politiques de routage est src/model/router/policies. Le dossier src/model/policies semble être un résidu et doit être vérifié pour suppression afin d’éviter des conflits d’importation. Consultez troubleshooting.md pour plus de détails.

Aperçu des phases 1-18
Le système est développé en 18 phases, chacune contribuant des fonctionnalités et composants spécifiques :

Phase 1 : Collecte de nouvelles et analyse de sentiment (news_scraper.py, news_analyzer.py) pour des fonctionnalités comme news_sentiment_score et news_impact_score.
Phase 3 : Pondération des fonctionnalités par régime de marché (tendance, range, défensif) implémentée dans obs_template.py (méthode 3), améliorée par regime_detector.py avec hmm_state_distribution (suggestion 4), renforçant la pertinence des observations.
Phase 5 : Récompenses adaptatives pour l’apprentissage par renforcement (trading_utils.py, méthode 5), ajustant les profits des trades en fonction de l’impact des nouvelles, de la volatilité (VIX), et des métriques de risque comme atr_dynamic et orderflow_imbalance via risk_manager.py (suggestion 1).
Phase 7 : Gestion de la mémoire contextuelle (db_maintenance.py, méthode 7), stockant les schémas de marché historiques dans market_memory.db, enrichi par hmm_state_distribution (suggestion 4).
Phase 8 : Ajustement fin des modèles SAC, PPO, DDPG, PPO-Lagrangian, et QR-DQN (train_sac.py, train_ppo_cvar.py, train_qr_dqn.py, méthode 8), optimisant les hyperparamètres pour des régimes de marché spécifiques, avec suivi de cvar_loss et qr_dqn_quantiles (suggestions 7, 8).
Phase 10 : Apprentissage en ligne (train_sac.py, train_ensemble.py, méthode 10), permettant des mises à jour des modèles en temps réel avec des données de marché, renforcé par le vote bayésien pour combiner SAC, PPO, DDPG via trade_probability.py avec ensemble_weights (suggestion 10).
Phase 13 : Métriques d’ordre et d’options (orderflow_indicators.py, options_metrics.py) pour des fonctionnalités comme vix_es_correlation, call_iv_atm, option_skew, et orderflow_imbalance (suggestion 1).
Phase 15 : Analyse de la microstructure et recalculs d’options (microstructure_guard.py, spotgamma_recalculator.py) pour des fonctionnalités comme spoofing_score, volume_anomaly, net_gamma, call_wall, intégrant hmm_state_distribution (suggestion 4).
Phase 18 : Métriques avancées de microstructure (trade_velocity, hft_activity_score), étendant microstructure_guard.py pour des informations sur le trading à haute fréquence, avec ajustements via risk_manager.py et prédictions RL via trade_probability.py (suggestions 1, 7, 8, 10).

Pour un détail complet de toutes les phases, voir :ref:phases.
Architecture
Le système est modulaire, avec des composants clés incluant :

Acquisition de données : data_provider.py récupère les données en temps réel d’IQFeed pour les marchés (ES, MNQ).
Pipeline de fonctionnalités : feature_pipeline.py traite 350 fonctionnalités (entraînement) ou 150 fonctionnalités SHAP (inférence), définies dans feature_sets.yaml.
Environnement de trading : trading_env.py fournit un environnement basé sur Gymnasium pour l’entraînement et la simulation.
Modèles neuronaux : neural_pipeline.py implémente des modèles LSTM, CNN, et MLP pour des prédictions (ex. : predicted_vix).
Apprentissage adaptatif : adaptive_learning.py stocke les schémas de marché dans market_memory.db et réentraîne les modèles.
Routage : main_router.py et detect_regime.py gèrent le routage des décisions en fonction des régimes de marché, améliorés par regime_detector.py (suggestion 4).
Gestion des risques : risk_manager.py ajuste dynamiquement les positions avec atr_dynamic et orderflow_imbalance (suggestion 1).
Prédictions RL : trade_probability.py prédit les probabilités de succès des trades avec PPO-Lagrangian (cvar_loss, suggestion 7), QR-DQN (qr_dqn_quantiles, suggestion 8), et vote bayésien (ensemble_weights, suggestion 10).
Tableau de bord : mia_dashboard.py visualise les performances, les régimes, l’importance des fonctionnalités SHAP, et les prédictions LSTM.
Interface vocale : mind_dialogue.py et mind_voice.py permettent des interactions vocales.
Robustesse : Tentatives exponentielles (max 3, délai 2^tentative), journalisation basée sur psutil, alertes, instantanés JSON, gestion SIGINT, sauvegardes incrémentielles (data/checkpoints/<module>/<market>/*.json.gz), et sauvegardes distribuées (AWS S3).

Pour une vue d’ensemble détaillée de l’architecture, voir :ref:architecture.
Table des matières
.. toctree::   :maxdepth: 2   :caption: Contenu :
   architecture   feature_engineering   installation   methodology   api_reference   troubleshooting   quickstart   modules   phases   tests   risk_manager   regime_detector   trade_probability
Index et tables

:ref:genindex
:ref:modindex
:ref:search

.. _architecture: architecture.. _phases: phases
