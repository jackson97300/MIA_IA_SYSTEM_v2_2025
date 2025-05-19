Feature Engineering pour MIA_IA_SYSTEM_v2_2025
Version : 2.1.5Date : 2025-05-14  
Aperçu
Le module de feature engineering de MIA_IA_SYSTEM_v2_2025 est responsable de la génération, de la sélection et de la pondération des fonctionnalités utilisées pour l’entraînement (350 fonctionnalités) et l’inférence (150 fonctionnalités SHAP) dans le cadre du trading algorithmique sur les contrats à terme (ES, MNQ). Les fonctionnalités sont calculées à partir des données brutes fournies exclusivement par IQFeed via data_provider.py et sont documentées dans config/feature_sets.yaml. Ce document décrit les catégories de fonctionnalités, le processus de sélection des 150 fonctionnalités SHAP, le fallback SHAP, et la pondération selon le régime de marché (méthode 3). Les nouvelles fonctionnalités incluent le dimensionnement dynamique des positions (risk_manager.py, suggestion 1), la détection des régimes avec HMM (regime_detector.py, suggestion 4), Safe RL/CVaR-PPO (trade_probability.py, suggestion 7), RL Distributionnel/QR-DQN (trade_probability.py, suggestion 8), et les ensembles de politiques avec vote bayésien (trade_probability.py, suggestion 10).
Note sur les dossiers policies : Le répertoire officiel pour les politiques de routage est src/model/router/policies. Le dossier src/model/policies semble être un résidu et doit être vérifié pour suppression afin d’éviter des conflits d’importation. Consultez troubleshooting.md pour résoudre les erreurs liées à des importations ambiguës.
Conformité : Toutes les références à dxFeed, obs_t, 320 features, et 81 features ont été supprimées pour aligner avec les spécifications de MIA_IA_SYSTEM_v2_2025.
Catégories de fonctionnalités
Les 350 fonctionnalités utilisées pour l’entraînement sont organisées en 13 catégories, comme défini dans config/feature_sets.yaml. Chaque catégorie capture des aspects spécifiques du marché, allant des données brutes aux signaux avancés dérivés de réseaux neuronaux.



Catégorie
Description
Exemples de fonctionnalités
Nombre estimé



raw_data
Données brutes du marché (prix, volume, etc.).
close, volume, bid_size_level_1, ask_size_level_1
~30


order_flow
Métriques dérivées du flux d’ordres (imbalance, absorption).
delta_volume, obi_score, absorption_strength, orderflow_imbalance (suggestion 1)
~20


trend
Indicateurs de tendance (momentum, moyennes mobiles).
rsi_14, adx_14, momentum_10, hmm_state_distribution (suggestion 4)
~25


volatility
Mesures de volatilité (ATR, VIX).
atr_14, bollinger_width_20, volatility_rolling_20, atr_dynamic (suggestion 1)
~15


neural_pipeline
Fonctionnalités générées par des réseaux neuronaux (CNN, LSTM).
neural_feature_0, neural_feature_1, cnn_pressure
~40


latent_factors
Facteurs latents issus de l’analyse de données (PCA, autoencodeurs).
latent_vol_regime_vec_1
~10


self_awareness
Métriques de performance interne du système (confiance, entropie).
confidence_drop_rate, sgc_entropy, cvar_loss (suggestion 7), qr_dqn_quantiles (suggestion 8), ensemble_weight_sac, ensemble_weight_ppo, ensemble_weight_ddpg (suggestion 10)
~15


mia_memory
Fonctionnalités dérivées de la mémoire contextuelle (méthode 7).
mm_score
~15


option_metrics
Métriques d’options (gamma, vanna, max pain).
gex, max_pain_strike, net_gamma, zero_gamma
~30


market_structure_signals
Signaux de structure de marché (zones des dealers, déclencheurs de volatilité).
dealer_zones_count, vol_trigger, ref_px
~20


context_aware
Fonctionnalités sensibles au contexte (impact des nouvelles, événements).
news_impact_score, event_volatility_impact
~15


cross_asset
Corrélations inter-actifs (actions, obligations).
spy_lead_return, vix_es_correlation
~10


dynamic_features
Fonctionnalités dynamiques générées en temps réel (neural dynamic).
neural_dynamic_feature_1 à neural_dynamic_feature_50
~50


Total : 350 fonctionnalités pour l’entraînement, réduites à 150 fonctionnalités SHAP pour l’inférence.
Processus de sélection des 150 fonctionnalités SHAP
Les 150 fonctionnalités SHAP utilisées pour l’inférence sont sélectionnées à partir des 350 fonctionnalités d’entraînement via un processus basé sur l’importance SHAP (Shapley Additive Explanations) implémenté dans src/features/shap_weighting.py. Les étapes sont les suivantes :

Calcul des poids SHAP :

calculate_shap_weights analyse l’importance des fonctionnalités en fonction de leur contribution aux prédictions des modèles (SAC, PPO, DDPG, PPO-Lagrangian, QR-DQN).
Les poids sont stockés dans data/features/<market>/feature_importance.csv.
Un cache est maintenu dans data/features/<market>/feature_importance_cache.csv pour les cas où les données principales ne sont pas disponibles.
[Ajout] Les fonctionnalités comme atr_dynamic, orderflow_imbalance (suggestion 1), hmm_state_distribution (suggestion 4), cvar_loss, qr_dqn_quantiles, ensemble_weight_sac, ensemble_weight_ppo, ensemble_weight_ddpg (suggestions 7, 8, 10) sont évaluées pour leur contribution via risk_manager.py, regime_detector.py, et trade_probability.py.


Priorisation des métriques critiques :

Certaines métriques, notamment celles issues de spotgamma_recalculator.py (ex. : key_strikes_1, max_pain_strike, net_gamma, zero_gamma, dealer_zones_count, vol_trigger, ref_px, data_release), reçoivent un bonus d’importance (x1.2) pour refléter leur pertinence dans l’analyse de la structure du marché.
[Ajout] Les nouvelles fonctionnalités (atr_dynamic, orderflow_imbalance, hmm_state_distribution, cvar_loss, qr_dqn_quantiles, ensemble_weight_sac, ensemble_weight_ppo, ensemble_weight_ddpg) reçoivent également un bonus d’importance (x1.2) pour leur rôle dans la gestion des risques, la détection des régimes, et les prédictions RL.


Sélection des top 150 fonctionnalités :

Les fonctionnalités sont triées par importance SHAP décroissante, et les 150 premières sont sélectionnées.
La liste est validée pour garantir qu’elle inclut les métriques critiques (ex. : obi_score, gex, news_impact_score, atr_dynamic, hmm_state_distribution, cvar_loss).


Fallback :

En cas d’absence de feature_importance.csv ou feature_importance_cache.csv, une liste statique de 150 fonctionnalités est utilisée (voir obs_template.py pour la liste complète).
Cette liste est alignée avec les catégories clés (ex. : rsi_14, neural_dynamic_feature_1, key_strikes_1, hmm_state_distribution).


Documentation :

Les 150 fonctionnalités SHAP sont documentées dans config/feature_sets.yaml sous la section feature_sets.observation.
Chaque fonctionnalité est accompagnée de métadonnées (nom, type, source, statut, plage, priorité, description).



Pondération selon le régime (Méthode 3)
La méthode 3 consiste à pondérer les fonctionnalités selon le régime de marché (tendance, range, défensif) pour optimiser les décisions de trading. Les poids sont définis dans config/es_config.yaml sous obs_template.weights. Voici un aperçu :



Régime
Fonctionnalité
Poids
Raison



Tendance
mm_score
1.5
Priorité à la mémoire contextuelle pour capturer les tendances persistantes.


Tendance
hft_score
1.5
Importance des signaux à haute fréquence dans les marchés directionnels.


Tendance
breakout_score
1.0
Poids neutre pour les signaux de rupture.


Tendance
hmm_state_distribution (sugg. 4)
1.2
Priorité à la détection des régimes HMM pour confirmer les tendances.


Tendance
atr_dynamic (sugg. 1)
1.2
Importance de la volatilité dynamique pour ajuster les positions.


Range
breakout_score
1.5
Priorité aux signaux de rupture dans les marchés en consolidation.


Range
mm_score
1.0
Poids neutre pour la mémoire contextuelle.


Range
hft_score
1.0
Poids neutre pour les signaux à haute fréquence.


Range
hmm_state_distribution (sugg. 4)
1.0
Poids neutre pour la détection des régimes HMM dans les marchés stables.


Range
orderflow_imbalance (sugg. 1)
1.2
Importance des déséquilibres d’ordre pour détecter les mouvements.


Défensif
mm_score
0.5
Réduction de l’importance pour minimiser les faux signaux.


Défensif
hft_score
0.5
Réduction de l’importance dans un marché instable.


Défensif
breakout_score
0.5
Réduction de l’importance pour éviter les trades risqués.


Défensif
hmm_state_distribution (sugg. 4)
0.8
Importance réduite pour la détection des régimes dans un marché volatil.


Défensif
cvar_loss (sugg. 7)
1.0
Priorité à la gestion des pertes extrêmes pour minimiser les risques.


Ces poids sont appliqués dans obs_template.py lors de la création du vecteur d’observation pour trading_env.py.
Gestion des données

Source : Toutes les fonctionnalités sont calculées à partir des données IQFeed via data_provider.py. Aucune autre source (ex. : dxFeed) n’est utilisée.
Stockage :
Les fonctionnalités brutes et calculées sont stockées dans market_memory.db (géré par db_maintenance.py).
Les 150 fonctionnalités SHAP sont sauvegardées dans data/features/<market>/obs_template.csv.
[Ajout] Les nouvelles fonctionnalités (atr_dynamic, orderflow_imbalance, hmm_state_distribution, cvar_loss, qr_dqn_quantiles, ensemble_weight_sac, ensemble_weight_ppo, ensemble_weight_ddpg) sont stockées dans market_memory.db et journalisées dans data/logs/<market>/risk_manager_performance.csv, regime_detector_performance.csv, trade_probability.log.


Sauvegardes :
Sauvegardes incrémentielles dans data/checkpoints/obs_template/<market>/*.json.gz (toutes les 5 minutes, 5 versions).
Sauvegardes distribuées vers AWS S3 dans obs_template/ (toutes les 15 minutes).


Cache : Un cache LRU est utilisé pour les résultats de get_shap_features et create_observation (taille max : 1000).

Validation
La validation des fonctionnalités est effectuée par obs_template.py dans la méthode validate_obs_template :

Structure : Vérifie que les 150 fonctionnalités SHAP sont présentes et de type numérique.
Seuils : Contrôle les plages des métriques critiques (ex. : obi_score entre -1 et 1, dealer_zones_count entre 0 et 10, hmm_state_distribution entre 0 et 1).
Cohérence : S’assure qu’aucune valeur non scalaire (listes, dictionnaires) n’est présente.
[Ajout] Validation spécifique pour les nouvelles fonctionnalités : atr_dynamic (>0), orderflow_imbalance (entre -1 et 1), cvar_loss (≥0), qr_dqn_quantiles (entre 0 et 1), ensemble_weight_sac, ensemble_weight_ppo, ensemble_weight_ddpg (entre 0 et 1, somme=1).

Fallback Statique pour fonctionnalités SHAP
En cas d’absence du cache SHAP (data/features/feature_importance_cache.csv), le système charge une liste statique de 150 fonctionnalités SHAP depuis config/feature_sets.yaml (section feature_sets.ES.inference pour ES, feature_sets.MNQ.inference pour MNQ). Cette logique est implémentée dans feature_pipeline.py via la méthode load_shap_fallback.

Implémentation : src/features/feature_pipeline.py
Configuration : config/feature_sets.yaml
Tests : tests/test_feature_pipeline.py, tests/test_risk_manager.py, tests/test_regime_detector.py, tests/test_trade_probability.py

Notes

Conformité : Suppression des références à 320 features, 81 features, et obs_t pour alignement avec les standards actuels.
Scalabilité : Les fonctionnalités dynamiques (neural_dynamic_feature_1 à neural_dynamic_feature_50) permettent d’incorporer de nouveaux signaux sans modifier la structure.
Dépendances : Les nouveaux modules requièrent hmmlearn>=0.2.8,<0.3.0, stable-baselines3>=2.0.0,<3.0.0, et ray[rllib]>=2.0.0,<3.0.0 (voir requirements.txt).
Prochaines étapes :
Intégration de l’API Bloomberg pour enrichir les fonctionnalités cross_asset (prévue pour juin 2025).
Vérification et suppression du dossier src/model/policies s’il n’est pas utilisé.
Ajout de tests unitaires pour les nouvelles fonctionnalités dans tests/test_risk_manager.py, tests/test_regime_detector.py, tests/test_trade_probability.py.



Pour plus de détails, consultez :

architecture.md : Vue d’ensemble de l’architecture.
methodology.md : Détails sur la méthode 3 (pondération des fonctionnalités).
api_reference.md : Référence des fonctions de shap_weighting.py et obs_template.py.
risk_manager.md : Guide pour risk_manager.py.
regime_detector.md : Guide pour regime_detector.py.
trade_probability.md : Guide pour trade_probability.py.

