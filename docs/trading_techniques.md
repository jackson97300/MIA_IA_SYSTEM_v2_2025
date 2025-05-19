Techniques de Trading de MIA_IA_SYSTEM_v2_2025
Version: 2.1.5Date: 2025-05-14
Introduction
Ce document fournit une explication détaillée et claire des techniques de trading utilisées par MIA_IA_SYSTEM_v2_2025, un système avancé de trading algorithmique pour les contrats à terme E-mini S&P 500 (ES) et Micro E-mini Nasdaq-100 (MNQ). Le système traite les données en temps réel d’IQFeed via un pipeline de fonctionnalités (350 fonctionnalités pour l’entraînement, 150 fonctionnalités SHAP pour l’inférence) et utilise des modèles d’apprentissage par renforcement (SAC, PPO, DDPG, PPO-Lagrangian, QR-DQN) pour exécuter des trades. Il intègre des modules avancés pour :

Le dimensionnement dynamique des positions (risk_manager.py, suggestion 1).
La détection des régimes de marché (regime_detector.py, suggestion 4).
Le Safe RL/CVaR-PPO (trade_probability.py, suggestion 7).
Le RL Distributionnel/QR-DQN (trade_probability.py, suggestion 8).
Les ensembles de politiques avec vote bayésien (trade_probability.py, suggestion 10).

Ce guide couvre les stratégies d’entrée en position, la gestion des positions, les caractéristiques techniques des modèles, les setups de trading, les approches probabilistes, les niveaux de SpotGamma, l’analyse de la microstructure, et la robustesse du système, incluant le réentraînement et l’analyse des trades. L’objectif est de permettre aux traders, développeurs, et opérateurs de comprendre précisément comment le système trade, son processus décisionnel, et pourquoi il est robuste.
Note sur les dossiers policies : Le répertoire officiel pour les politiques de routage est src/model/router/policies. Le dossier src/model/policies semble être un résidu et doit être vérifié pour suppression afin d’éviter des conflits d’importation. Consultez troubleshooting.md pour résoudre les erreurs liées à des importations ambiguës.
Conformité : Toutes les références à dxFeed, obs_t, 320 features, et 81 features ont été supprimées pour aligner avec les spécifications de MIA_IA_SYSTEM_v2_2025.
Note sur live_trading.py : Le module live_trading.py est actuellement en version 2.1.3 (2025-05-13) et ne valide pas encore les nouvelles fonctionnalités (bid_ask_imbalance, trade_aggressiveness, iv_skew, iv_term_structure) ni la validation obs_t. Une mise à jour vers la version 2.1.5 est prévue pour assurer la compatibilité.
Pour plus de détails, voir :

usage.md (guide d’utilisation).
troubleshooting.md (dépannage).
quickstart.md (démarrage rapide).
installation.md (installation).
setup.md (configuration).
architecture.md (vue d’ensemble).
phases.md (phases de développement).
modules.md (modules).
api_reference.md (API).
roadmap.md (feuille de route).


Modèles d’apprentissage
MIA_IA_SYSTEM_v2_2025 utilise une combinaison de modèles d’apprentissage par renforcement pour optimiser les décisions de trading selon les régimes de marché (tendance, range, défensif). Les modèles incluent Soft Actor-Critic (SAC), Proximal Policy Optimization (PPO), Deep Deterministic Policy Gradient (DDPG), PPO-Lagrangian (Safe RL, suggestion 7), et QR-DQN (RL Distributionnel, suggestion 8), avec un vote bayésien pour les ensembles de politiques (suggestion 10).
1. Soft Actor-Critic (SAC)
Architecture :

Réseau de l’acteur : Perceptron multicouche (MLP) avec 3 couches cachées (256, 128, 64 unités), activation ReLU.
Réseaux des critiques : Deux réseaux Q jumeaux, chacun avec 3 couches cachées (256, 128, 64 unités), activation ReLU.
Régularisation par entropie pour l’exploration (alpha auto-ajusté).

Paramètres :

Taux d’apprentissage : 3e-4 (acteur), 1e-3 (critique).
Taille du batch : 64 (configurable dans algo_config.yaml).
Facteur de remise (γ) : 0.99.
Tampon de rejeu : 1M de transitions.

Entraînement :

Entraînement initial sur des données historiques ES/MNQ via trading_env.py.
Réentraînement hebdomadaire ou lors de changements de régime détectés par regime_detector.py (suggestion 4, ex. : hmm_state_distribution change).
Utilise la méthode 8 (fine-tuning) pour ajuster les poids selon les régimes.

Pourquoi utilisé :

Équilibre exploration et exploitation, idéal pour les marchés volatils.
Robuste aux données bruitées grâce à la régularisation par entropie.

Implémentation : src/model/rl/train_sac.py (voir api_reference.md pour train_sac).
Exemple :

En régime tendance, SAC privilégie les entrées basées sur mm_score (>1.0) et ajuste la taille des positions via risk_manager.py (suggestion 1).

2. Proximal Policy Optimization (PPO)
Architecture :

Réseau de politique : MLP avec 2 couches cachées (128, 64 unités), activation tanh.
Réseau de valeur : MLP avec 2 couches cachées (128, 64 unités), activation ReLU.
Objectif de substitution tronqué pour des mises à jour stables.

Paramètres :

Taux d’apprentissage : 2e-4.
Taille du batch : 128.
Ratio de troncature : 0.2.
Époques par mise à jour : 10.

Entraînement :

Entraîné sur trading_env.py avec des trades simulés.
Réentraînement déclenché par une dégradation des performances (ex. : sharpe_ratio < 1.0, suivi par algo_performance_logger.py).
Utilise la méthode 10 (apprentissage en ligne) pour des mises à jour en temps réel.

Pourquoi utilisé :

Stable et efficace en termes d’échantillons, adapté au trading continu.
Gère efficacement les espaces d’actions discrets et continus.

Implémentation : src/model/rl/train_sac.py (logique partagée).
Exemple :

En régime range, PPO ajuste les entrées près de key_strikes_1 avec une probabilité calculée par neural_pipeline.py.

3. Deep Deterministic Policy Gradient (DDPG)
Architecture :

Réseau de l’acteur : MLP avec 3 couches cachées (256, 128, 64 unités), activation tanh.
Réseau du critique : MLP avec 3 couches cachées (256, 128, 64 unités), activation ReLU.
Réseaux cibles avec mises à jour douces (τ = 0.005).

Paramètres :

Taux d’apprentissage : 1e-4 (acteur), 3e-4 (critique).
Taille du batch : 64.
Facteur de remise (γ) : 0.99.
Tampon de rejeu : 1M de transitions.

Entraînement :

Similaire à SAC, avec un focus sur les politiques déterministes.
Réentraînement aligné sur SAC, utilisant la méthode 18 (meta-learning) pour la généralisation.

Pourquoi utilisé :

Efficace pour les espaces d’actions continus (ex. : taille des positions via risk_manager.py).
Complète l’approche stochastique de SAC dans les régimes stables.

Implémentation : src/model/rl/train_sac.py.
Exemple :

En régime défensif, DDPG réduit la taille des positions à 0.5x levier si vol_trigger est faible.

4. PPO-Lagrangian (Safe RL, suggestion 7)
Architecture :

Extension de PPO avec une contrainte de CVaR (Conditional Value at Risk) pour minimiser les pertes extrêmes.
Réseau de politique : MLP avec 2 couches cachées (128, 64 unités), activation tanh.
Réseau de valeur : MLP avec 2 couches cachées (128, 64 unités), activation ReLU.
Multiplicateur de Lagrange pour ajuster la contrainte CVaR.

Paramètres :

Taux d’apprentissage : 2e-4.
Taille du batch : 128.
Ratio de troncature : 0.2.
CVaR Alpha : 0.95 (configurable dans es_config.yaml).

Entraînement :

Entraîné sur trading_env.py avec une fonction de coût intégrant cvar_loss.
Réentraînement déclenché par une augmentation de cvar_loss (>0.1) ou un changement de régime.
Utilise la méthode 7 (Safe RL) pour optimiser sous contraintes de risque.

Pourquoi utilisé :

Minimise les pertes extrêmes dans des conditions de marché volatiles.
Idéal pour les régimes défensifs où la préservation du capital est prioritaire.

Implémentation : src/model/rl/train_ppo_cvar.py.
Exemple :

Réduit la taille des positions si cvar_loss dépasse 0.1, en ajustant via risk_manager.py.

5. QR-DQN (RL Distributionnel, suggestion 8)
Architecture :

Réseau Q distributionnel : MLP avec 3 couches cachées (256, 128, 64 unités), activation ReLU.
Sortie : Distribution des valeurs Q sur 51 quantiles (configurable dans es_config.yaml).

Paramètres :

Taux d’apprentissage : 1e-4.
Taille du batch : 64.
Quantiles : 51.
Facteur de remise (γ) : 0.99.

Entraînement :

Entraîné sur trading_env.py with historical and simulated data.
Réentraînement déclenché par une divergence dans qr_dqn_quantiles (ex. : variance > 0.5).
Utilise la méthode 8 (RL distributionnel) pour capturer l’incertitude des rendements.

Pourquoi utilisé :

Capture la distribution complète des rendements, utile pour les régimes range avec incertitude élevée.
Améliore la prise de décision dans des scénarios à faible probabilité mais à fort impact.

Implémentation : src/model/rl/train_qr_dqn.py.
Exemple :

Privilégie les entrées près de max_pain_strike si la distribution des qr_dqn_quantiles indique un rendement attendu positif.

6. Vote bayésien (Ensembles de politiques, suggestion 10)
Architecture :

Combinaison des prédictions de SAC, PPO, et DDPG via un vote bayésien.
Poids : ensemble_weight_sac, ensemble_weight_ppo, ensemble_weight_ddpg (ex. : 0.4, 0.3, 0.3, configurables dans es_config.yaml).
Utilise une approche probabiliste pour pondérer les décisions en fonction des performances historiques.

Paramètres :

Poids d’ensemble : Somme à 1.0 (ex. : 0.4 + 0.3 + 0.3).
Seuil de confiance : 0.65 (configurable dans es_config.yaml).

Entraînement :

Les modèles individuels (SAC, PPO, DDPG) sont entraînés séparément.
Les poids d’ensemble sont ajustés dynamiquement par trade_probability.py en fonction des métriques de performance (sharpe_ratio, drawdown).

Pourquoi utilisé :

Améliore la robustesse en combinant les forces des différents modèles.
Réduit le risque d’erreurs en régime incertain en pondérant les décisions.

Implémentation : src/model/rl/train_ensemble.py.
Exemple :

Si SAC prédit une entrée longue avec 0.7 de probabilité et PPO avec 0.6, le vote bayésien ajuste la décision finale en fonction des poids (ex. : entrée si probabilité pondérée > 0.65). Par exemple, en régime tendance, si bid_ask_imbalance (>0.2) et mm_score (>1.0) renforcent la prédiction de SAC, le poids ensemble_weight_sac est augmenté temporairely, favorisant une entrée longue.


Réentraînement et robustesse
Fréquence

Réentraînement hebdomadaire ou lors d’un changement de régime détecté par regime_detector.py (suggestion 4). Les déclencheurs incluent :
VIX > 20.
event_frequency_24h > 5.
Changement dans hmm_state_distribution (ex. : probabilité d’un nouveau régime > 0.7).
Augmentation de cvar_loss (>0.1, suggestion 7).
Divergence des qr_dqn_quantiles (variance > 0.5, suggestion 8).



Processus

adaptive_learning.py met à jour market_memory.db avec de nouveaux schémas (méthode 7).
Les scripts train_sac.py, train_ppo_cvar.py, train_qr_dqn.py, et train_ensemble.py réentraînent les modèles avec des données récentes, optimisant pour finetune_loss ou cvar_loss.
Les poids d’ensemble (ensemble_weight_sac, ensemble_weight_ppo, ensemble_weight_ddpg) sont ajustés via trade_probability.py pour maximiser le sharpe_ratio (suggestion 10).
Les nouvelles fonctionnalités (bid_ask_imbalance, trade_aggressiveness, iv_skew, iv_term_structure) sont intégrées dans le réentraînement pour affiner les prédictions.

Exemple :

Réentraînement du modèle PPO-Lagrangian si cvar_loss dépasse 0.1 sur 100 trades.
Augmentation de ensemble_weight_sac à 0.5 si SAC surperforme PPO et DDPG, basée sur l’importance SHAP de trade_aggressiveness.

Implémentation : src/model/utils/adaptive_learning.py, src/model/rl/train_*.py.
Journaux : Suivi dans data/logs/<market>/train_*.log.
Optimisation du cache LRU

Les modules (neural_pipeline.py, mia_dashboard.py, feature_pipeline.py, live_trading.py) utilisent un cache LRU basé sur OrderedDict pour optimiser les performances. Par exemple :
neural_pipeline.py met en cache les prédictions LSTM/CNN (predicted_vix, cnn_pressure) pour éviter les recalculs.
mia_dashboard.py stocke les chemins des figures PNG (regime_fig_path, feature_fig_path) pour réduire la surcharge mémoire.
feature_pipeline.py met en cache les importances SHAP pour accélérer la validation.
live_trading.py utilise prediction_cache pour stocker les prédictions récentes, avec une taille maximale configurable (max_prediction_cache_size).


Le cache est évincé selon la taille (max_cache_size) ou l’âge (cache_duration, ex. : 24 heures), assurant une utilisation mémoire efficace.

Exemple :

En régime range, le cache LRU de neural_pipeline.py récupère une prédiction basée sur iv_skew en <0.1s, évitant un recalcul coûteux.

Tests unitaires

La robustesse est renforcée par des tests unitaires dans :
tests/test_neural_pipeline.py : Valide les prédictions LSTM/CNN, la validation obs_t, et le cache LRU.
tests/test_mia_dashboard.py : Teste le layout Dash, les callbacks, et l’affichage des nouvelles fonctionnalités (bid_ask_imbalance, iv_term_structure).
tests/test_feature_pipeline.py : Vérifie la génération des 350/150 fonctionnalités, incluant trade_aggressiveness et iv_skew.


Note : Les tests pour live_trading.py (tests/test_live_trading.py) sont en cours de développement pour couvrir les nouvelles fonctionnalités et la validation obs_t, en raison du décalage de version (2.1.3).

Exemple :

test_neural_pipeline.py valide que validate_obs_t détecte les colonnes manquantes comme bid_ask_imbalance, déclenchant une alerte si absente.


Entrée en position
Le système entre en position en fonction de setups probabilistes combinant les régimes de marché (détectés par regime_detector.py, suggestion 4), les niveaux de SpotGamma, les signaux de microstructure, et les probabilités basées sur les fonctionnalités SHAP.
1. Setups de trading
Setups par régime (Méthode 3, améliorée par regime_detector.py) :
Régime Tendance

Entrée : Forte dynamique (momentum_10 > 0.5) et mm_score élevé (>1.0).
Condition : hmm_state_distribution indique un état tendance (>0.7 probabilité).
Nouvelles fonctionnalités : bid_ask_imbalance (>0.2) et trade_aggressiveness (>0.3) renforcent la confiance dans les entrées haussières.
Exemple : Entrée longue si rsi_14 > 70, adx_14 > 25, bid_ask_imbalance > 0.2, et hmm_state_distribution confirme la tendance.

Régime Range

Entrée : Signaux de rupture (breakout_score > 1.5) près des niveaux clés (ex. : max_pain_strike).
Condition : hmm_state_distribution indique un état range (>0.6 probabilité).
Nouvelles fonctionnalités : iv_skew (>0.01) et iv_term_structure (>0.02) signalent des opportunités de rupture.
Exemple : Entrée longue sur une rupture au-dessus de key_strikes_1 avec obi_score > 0.3, orderflow_imbalance > 0.5, et iv_skew > 0.01.

Régime Défensif

Entrée : Signaux conservateurs (hft_score < 0.5, vol_trigger proche de 0).
Condition : hmm_state_distribution indique un état défensif (>0.7 probabilité).
Nouvelles fonctionnalités : news_impact_score (< -0.5) et spoofing_score (<0.3) confirment les conditions conservatrices.
Exemple : Entrée courte sur un rejet à zero_gamma avec news_impact_score < -0.5 et spoofing_score < 0.3.

Implémentation : src/features/regime_detector.py (suggestion 4), src/model/utils/trading_utils.py (detect_market_regime), src/model/utils/obs_template.py (méthode 3).Journaux : Suivi dans data/logs/<market>/regime_detector.log.
Niveaux de SpotGamma

Niveaux clés : max_pain_strike, zero_gamma, call_wall, key_strikes_1 à key_strikes_5.
Entrée : Forte probabilité à un support/résistance (ex. : entrée longue près de call_wall avec net_gamma > 0).
Nouvelles fonctionnalités : iv_term_structure (>0.02) indique une volatilité stable, renforçant les entrées près des niveaux clés.
Exemple : Achat lorsque le prix approche zero_gamma avec gamma_velocity_near en hausse, orderflow_imbalance > 0.5, et iv_term_structure > 0.02.

Implémentation : src/features/spotgamma_recalculator.py, intégré dans src/features/feature_pipeline.py.Journaux : Suivi dans data/logs/<market>/spotgamma_recalculator.log.
Signaux de microstructure

Signaux : spoofing_score, volume_anomaly, hft_activity_score, trade_velocity, bid_ask_imbalance, trade_aggressiveness (suggestion 3).
Entrée :
Éviter les entrées lors d’un spoofing_score élevé (>0.7).
Privilégier un trade_velocity élevé (>1.0) et bid_ask_imbalance aligné (>0.2).
trade_aggressiveness (>0.3) confirme la pression directionnelle.


Exemple : Entrée longue lorsque volume_anomaly est faible, hft_activity_score confirme la dynamique, bid_ask_imbalance > 0.2, et trade_aggressiveness > 0.3.

Implémentation : src/features/microstructure_guard.py.Journaux : Suivi dans data/logs/<market>/microstructure_guard.log.
2. Approche probabiliste
Calcul des probabilités :

Utilise neural_pipeline.py (LSTM/CNN) pour estimer la probabilité de succès d’un trade basée sur des fonctionnalités comme predicted_vix, cnn_pressure, atr_dynamic (suggestion 1), hmm_state_distribution (suggestion 4), bid_ask_imbalance, trade_aggressiveness, iv_skew, et iv_term_structure.
Seuil : Entrée uniquement si probabilité > 65% (configurable dans es_config.yaml).
Exemple : Probabilité de 70% pour une entrée longue si obi_score > 0.3, net_gamma > 0, bid_ask_imbalance > 0.2, et cvar_loss < 0.1 (suggestion 7).

Pondération des fonctionnalités :

Fonctionnalités SHAP (ex. : obi_score, net_gamma, atr_dynamic, orderflow_imbalance, trade_aggressiveness, iv_skew) pondérées par régime via regime_detector.py (suggestion 4) pour prioriser les signaux à fort impact.
Exemple : En régime tendance, poids de mm_score = 1.5 et bid_ask_imbalance = 1.3, augmentant la confiance dans l’entrée.

Implémentation : src/model/utils/obs_template.py, src/model/neural_pipeline.py.Journaux : Suivi dans data/logs/<market>/neural_pipeline.log.
3. Signaux d’entrée
Génération des signaux :

Combine les signaux de régime (regime_detector.py), les niveaux de SpotGamma (spotgamma_recalculator.py), et les données de microstructure (microstructure_guard.py).
Exemple : Entrée longue lorsque :
Régime : Tendance (hmm_state_distribution > 0.7).
SpotGamma : Prix près de call_wall.
Microstructure : spoofing_score faible (<0.3), trade_velocity élevé (>1.0), bid_ask_imbalance > 0.2, trade_aggressiveness > 0.3.
Probabilité : >70% (calculée par neural_pipeline.py).



Confirmation :

Nécessite un alignement de news_impact_score (ex. : >0.5 pour les entrées haussières).
Utilise trade_probability.py pour estimer le rapport risque/récompense via calculate_profit, tenant compte de cvar_loss (suggestion 7) et qr_dqn_quantiles (suggestion 8).

Implémentation : src/model/main_router.py, src/model/utils/trading_utils.py, src/model/trade_probability.py.Journaux : Suivi dans data/logs/<market>/trade_probability.log.

Gestion des positions
Le système gère les positions ouvertes de manière dynamique pour maximiser les rendements et minimiser les risques, en intégrant le dimensionnement dynamique des positions (risk_manager.py, suggestion 1) et les régimes de marché (regime_detector.py, suggestion 4).
1. Ajustement du risque
Levier :

Ajusté selon le régime détecté par regime_detector.py (suggestion 4) :
Tendance : Jusqu’à 5x levier (max_leverage: 5.0 dans es_config.yaml).
Range : 2-3x levier.
Défensif : 1x ou pas de levier.


Nouvelles fonctionnalités : bid_ask_imbalance (>0.9) ou iv_term_structure (>0.05) peuvent réduire le levier à 1x pour limiter les risques de liquidité.
Exemple : Réduire le levier à 1x si vix_es_correlation > 0.8, cvar_loss > 0.1 (suggestion 7), ou bid_ask_imbalance > 0.9.

Risque par trade :

Limité à 1% du compte (risk_per_trade: 0.01 dans es_config.yaml).
Ajusté dynamiquement par risk_manager.py selon atr_dynamic (ex. : réduire à 0.5% si atr_dynamic > 100), orderflow_imbalance (ex. : réduire si >0.9), et trade_aggressiveness (ex. : réduire si >0.5) (suggestion 1).
Exemple : Risque réduit à 0.3% si iv_skew > 0.02 indique une volatilité asymétrique.

Implémentation : src/risk_management/risk_manager.py, src/model/utils/trading_utils.py (adjust_risk_and_leverage).Journaux : Suivi dans data/logs/<market>/risk_manager.log.
2. Stop-loss et take-profit
Stop-loss :

Défini selon atr_14 (ex. : 2x ATR sous le prix d’entrée pour une position longue).
Stop suiveur en régime tendance : Suit à 1.5x ATR si hmm_state_distribution confirme la tendance (suggestion 4).
Nouvelles fonctionnalités : Ajusté par iv_term_structure (ex. : élargi à 3x ATR si iv_term_structure > 0.05 pour anticiper une volatilité accrue).
Exemple : Entrée longue à 5000, ATR = 10, stop-loss à 4980, ajusté à 4970 si iv_term_structure > 0.05.

Take-profit :

Défini aux niveaux clés de SpotGamma (ex. : key_strikes_2 ou call_wall).
Sorties partielles : 50% à 1:2 risque/récompense, 50% à 1:4.
Nouvelles fonctionnalités : bid_ask_imbalance (>0.2) peut déclencher une sortie partielle anticipée pour capturer les profits.
Exemple : Take-profit à 5040 (2:1) et 5080 (4:1) pour un stop-loss à 4980, avec sortie partielle à 5020 si bid_ask_imbalance > 0.2.

Implémentation : src/model/utils/trading_utils.py (calculate_profit).Journaux : Suivi dans data/logs/<market>/trading_utils.log.
3. Sorties dynamiques
Changements de régime :

Sortie si changement de régime (ex. : tendance vers défensif) détecté par regime_detector.py (suggestion 4).
Exemple : Clôture d’une position longue si adx_14 tombe sous 20 ou hmm_state_distribution indique un état défensif.

Déclencheurs de microstructure :

Sortie sur un spoofing_score élevé (>0.7), des pics de volume_anomaly, ou un bid_ask_imbalance inversé (< -0.2) (suggestion 3).
Nouvelles fonctionnalités : trade_aggressiveness (>0.5) ou iv_skew (>0.02) peuvent déclencher une sortie anticipée pour éviter les renversements.
Exemple : Sortie si hft_activity_score signale un renversement, trade_aggressiveness > 0.5 indique une vente massive, ou iv_skew > 0.02 suggère une volatilité asymétrique.

Impact des nouvelles :

Sortie si news_impact_score < -0.5 (nouvelles baissières).
Exemple : Clôture d’une position longue si news_impact_score chute à -0.6 après une annonce macroéconomique.

Implémentation : src/model/main_router.py, src/model/utils/trading_utils.py.Journaux : Suivi dans data/logs/<market>/main_router.log.

Robustesse du système
La robustesse du système est assurée par le réentraînement, l’analyse des trades, les alertes, les sauvegardes, la validation des données, le cache LRU, les tests unitaires, et la surveillance des performances, avec une intégration des nouvelles métriques comme cvar_loss, qr_dqn_quantiles, et ensemble_weights.
1. Réentraînement
Fréquence : Hebdomadaire ou lors d’un changement de régime (ex. : VIX > 20, event_frequency_24h > 5, ou hmm_state_distribution indique un nouveau régime, suggestion 4).
Processus :

adaptive_learning.py met à jour market_memory.db avec de nouveaux schémas (méthode 7).
train_sac.py, train_ppo_cvar.py, train_qr_dqn.py, et train_ensemble.py réentraînent les modèles avec des données récentes, optimisant pour finetune_loss ou cvar_loss.
Ajustement des ensemble_weights via trade_probability.py pour maximiser le sharpe_ratio (suggestion 10).

Exemple :

Réentraînement du modèle QR-DQN si la variance des qr_dqn_quantiles dépasse 0.5 sur 100 trades.
Augmentation de ensemble_weight_sac à 0.5 si SAC surperforme PPO et DDPG.

Implémentation : src/model/utils/adaptive_learning.py, src/model/rl/train_*.py.Journaux : Suivi dans data/logs/<market>/train_*.log.
2. Analyse des trades
Métriques :

Taux de réussite : Pourcentage de trades gagnants (cible : >60%).
Profit/perte moyen : Suivi par trade dans data/logs/<market>/trading_utils.log.
Drawdown : Perte maximale depuis le pic, surveillée par mia_dashboard.py.
CVaR Loss : Perte attendue dans les pires scénarios (suggestion 7, cible : <0.1).
Quantiles QR-DQN : Distribution des rendements (suggestion 8, variance cible : <0.5).
Poids d’ensemble : Performance relative des modèles SAC, PPO, DDPG (suggestion 10).
Nouvelles fonctionnalités : Analyse de l’impact de bid_ask_imbalance, trade_aggressiveness, iv_skew, et iv_term_structure via SHAP pour identifier les setups à forte probabilité.

Analyse :

Identifie les setups à forte probabilité (ex. : entrées près de zero_gamma avec obi_score > 0.3, orderflow_imbalance > 0.5, et trade_aggressiveness > 0.3).
Ajuste les poids des modèles pour favoriser les schémas gagnants (méthode 8).
Exemple : Si les trades perdants sont corrélés à un spoofing_score élevé (>0.7) ou iv_skew > 0.02, réduire la probabilité d’entrée dans ces conditions.

Implémentation : src/model/utils/algo_performance_logger.py, src/model/utils/mia_dashboard.py.Journaux : Suivi dans data/logs/<market>/algo_performance_logger.log.
3. Alertes et surveillance
Alertes :

Notifications en temps réel via Telegram, SMS, email (alert_manager.py, telegram_alert.py).
Déclenchées pour :
Utilisation mémoire élevée (>1024 MB, mesurée via psutil).
Erreurs de trade (ex. : échec d’exécution dans live_trading.py).
Changements de régime (hmm_state_distribution change, suggestion 4).
Augmentation de cvar_loss (>0.1, suggestion 7).
Divergence des qr_dqn_quantiles (>0.5, suggestion 8).
Déséquilibre des ensemble_weights (ex. : un modèle domine, suggestion 10).
confidence_drop_rate > 0.5 (métrique Phase 8).


Note : Dans live_trading.py (version 2.1.3), les alertes Telegram sont implicites via AlertManager. Une mise à jour vers 2.1.5 ajoutera des appels explicites à send_telegram_alert.

Surveillance :

Journaux dans data/logs/<market>/<module>.log (ex. : run_system.log, trade_probability.log, risk_manager.log).
Tableau de bord (mia_dashboard.py) affiche le taux de réussite, le drawdown, l’importance des fonctionnalités SHAP, cvar_loss, qr_dqn_quantiles, ensemble_weights, et les nouvelles fonctionnalités (bid_ask_imbalance, iv_term_structure).

Implémentation : src/model/utils/alert_manager.py, src/utils/telegram_alert.py, src/model/utils/mia_dashboard.py.Journaux : Suivi dans data/logs/<market>/alert_manager.log.
4. Sauvegardes et récupération
Sauvegardes incrémentielles :

Instantanés dans data/cache/<module>/<market>/*.json.gz toutes les 60 secondes.
Points de contrôle dans data/checkpoints/<module>/<market>/*.json.gz toutes les 5 minutes (5 versions).

Sauvegardes distribuées (AWS S3) :

Téléversements S3 toutes les 15 minutes (s3_bucket dans es_config.yaml).
Exemple : s3://votre-bucket/mia_ia_system/obs_template_ES_*.csv.gz.

Récupération :

Restauration depuis le dernier point de contrôle en cas de crash :python src/model/utils/db_maintenance.py --restore --market ES



Implémentation : src/model/utils/db_maintenance.py.Journaux : Suivi dans data/logs/<market>/db_maintenance.log.
5. Validation des données

Les données sont validées via validate_features (src/data/validate_data.py) dans feature_pipeline.py et neural_pipeline.py pour garantir l’intégrité des 350 fonctionnalités (entraînement) ou 150 fonctionnalités SHAP (inférence).
Validation obs_t : Introduite dans neural_pipeline.py et feature_pipeline.py pour standardiser la validation des colonnes critiques (ex. : bid_ask_imbalance, trade_aggressiveness, iv_skew). Cette validation vérifie la présence et le type des colonnes, déclenchant des alertes si des écarts sont détectés.
Note : live_trading.py (version 2.1.3) utilise validate_data sans validate_obs_t. Une mise à jour vers 2.1.5 intégrera cette validation pour assurer la cohérence.

Exemple :

Si iv_term_structure est absente dans features_latest.csv, validate_obs_t déclenche une alerte Telegram et impute la valeur médiane.

6. Tests unitaires

Les tests unitaires renforcent la robustesse en validant les fonctionnalités critiques :
tests/test_neural_pipeline.py : Teste les prédictions LSTM/CNN, validate_obs_t, et le cache LRU.
tests/test_mia_dashboard.py : Vérifie le layout Dash, les callbacks, et l’affichage des nouvelles fonctionnalités.
tests/test_feature_pipeline.py : Valide la génération des 350/150 fonctionnalités, incluant bid_ask_imbalance et iv_skew.


Note : Les tests pour live_trading.py (tests/test_live_trading.py) sont en développement pour couvrir les nouvelles fonctionnalités, les commandes vocales, et la validation obs_t, en raison du décalage de version (2.1.3).

Exemple :

test_mia_dashboard.py valide que feature-dropdown affiche trade_aggressiveness et que les figures PNG sont générées sans surcharge mémoire.

7. Cache LRU

Les modules utilisent un cache LRU basé sur OrderedDict pour optimiser les performances :
neural_pipeline.py : Cache les prédictions (predicted_vix, cnn_pressure) pour éviter les recalculs.
mia_dashboard.py : Stocke les chemins des figures PNG pour réduire la mémoire.
feature_pipeline.py : Met en cache les importances SHAP.
live_trading.py : Utilise prediction_cache pour les prédictions récentes.


Configuration : Taille maximale (max_cache_size, ex. : 1000 entrées) et durée (cache_duration, ex. : 24 heures) configurables dans es_config.yaml.
Exemple : En régime défensif, le cache LRU de live_trading.py récupère une prédiction basée sur iv_skew en <0.1s.


Faiblesses ou risques potentiels
Malgré sa robustesse, MIA_IA_SYSTEM_v2_2025 présente certaines faiblesses et risques potentiels qui pourraient affecter ses performances ou sa maintenabilité. Ces points sont identifiés pour guider les futures améliorations.
1. Couplage implicite avec les configurations

Problème : Les fichiers de configuration (feature_sets.yaml, credentials.yaml, market_config.yaml) sont chargés directement dans plusieurs modules (live_trading.py, feature_pipeline.py, neural_pipeline.py), créant un couplage implicite. Cela complique la gestion des erreurs si un fichier est manquant ou mal configuré.
Risque : Une erreur dans un fichier YAML (ex. : clé absente) peut provoquer des échecs silencieux ou des comportements imprévisibles.
Solution proposée : Encapsuler les configurations dans une classe ConfigContext (ex. : dans src/model/utils/config_manager.py) pour centraliser la validation et la gestion des erreurs.
Exemple : Une clé manquante dans feature_sets.yaml pourrait biaiser la sélection des fonctionnalités SHAP dans live_trading.py.

2. Absence de test explicite de fin de boucle de trading

Problème : live_trading.py (version 2.1.3) manque d’une méthode explicite pour tester la fin de la boucle de trading (ex. : run() ou start_live_loop()). Cela complique la validation de la terminaison propre du système en mode live.
Risque : En cas d’interruption (ex. : SIGINT), des trades pourraient rester ouverts ou des snapshots ne pas être sauvegardés.
Solution proposée : Ajouter une méthode stop_trading() dans live_trading.py pour gérer la fin de la boucle, avec des tests unitaires dans tests/test_live_trading.py.
Exemple : Sans test de fin, un crash pourrait laisser des positions ouvertes, augmentant les pertes.

3. Biais dans load_shap_fallback

Problème : La méthode load_shap_fallback dans live_trading.py utilise une liste statique de 150 fonctionnalités SHAP hardcodées si le cache et feature_sets.yaml échouent, ce qui peut biaiser les prédictions.
Risque : Les prédictions basées sur des fonctionnalités non pertinentes peuvent réduire le sharpe_ratio ou augmenter le cvar_loss.
Solution proposée : Implémenter une validation dynamique des fonctionnalités SHAP (ex. : via validate_features dans src/data/validate_data.py) avant de recourir au fallback, et alerter via Telegram si le fallback est utilisé.
Exemple : Un fallback sur des fonctionnalités non alignées avec bid_ask_imbalance pourrait ignorer des signaux de liquidité critiques.

4. Décalage de version de live_trading.py

Problème : live_trading.py est en version 2.1.3, alors que neural_pipeline.py, mia_dashboard.py, et feature_pipeline.py sont en 2.1.5. Cela entraîne un manque d’intégration des nouvelles fonctionnalités (bid_ask_imbalance, trade_aggressiveness, iv_skew, iv_term_structure) et de la validation obs_t.
Risque : Les décisions de trading peuvent ignorer des signaux critiques, réduisant l’efficacité et la cohérence avec le tableau de bord (mia_dashboard.py).
Solution proposée : Mettre à jour live_trading.py vers la version 2.1.5, en ajoutant la validation des nouvelles fonctionnalités dans validate_data et validate_obs_t.

5. Dépendances non versionnées

Problème : Les dépendances comme trade_executor.py, risk_controller.py, ou sierra_chart_errors.py ne précisent pas leur version, ce qui peut causer des incohérences si elles ne sont pas alignées avec la version 2.1.5.
Risque : Une dépendance obsolète pourrait introduire des bugs ou des performances sous-optimales.
Solution proposée : Documenter les versions des dépendances dans modules.md et vérifier leur compatibilité dans troubleshooting.md.

6. Surcharge mémoire potentielle

Problème : Bien que le cache LRU limite la mémoire, des datasets volumineux ou un grand nombre de trades simultanés peuvent augmenter l’utilisation mémoire, surtout dans live_trading.py et feature_pipeline.py.
Risque : Une utilisation mémoire >1024 MB déclenche des alertes, mais un crash pourrait survenir en cas de surcharge prolongée.
Solution proposée : Ajouter des seuils de mémoire dynamiques dans es_config.yaml et optimiser le cache LRU pour évincer les entrées plus tôt en cas de pression mémoire.

Implémentation : Les solutions proposées seront intégrées dans la roadmap (voir roadmap.md) et suivies via troubleshooting.md.

