Référence de l’API pour MIA_IA_SYSTEM_v2_2025
Version : 2.1.5Date : 2025-05-14  
Aperçu
Ce document fournit une référence de l’API interne pour les modules principaux de MIA_IA_SYSTEM_v2_2025, un système avancé de trading algorithmique pour les contrats à terme (ES, MNQ). Il décrit les classes, méthodes, et fonctions clés utilisées dans les modules tels que data_provider.py, shap_weighting.py, obs_template.py, trading_utils.py, train_sac.py, db_maintenance.py, alert_manager.py, algo_performance_logger.py, risk_manager.py, regime_detector.py, et trade_probability.py. L’API interne facilite l’interaction entre les composants du système, qui utilise exclusivement IQFeed comme source de données via data_provider.py et est compatible avec 350 fonctionnalités pour l’entraînement et 150 fonctionnalités SHAP pour l’inférence, comme défini dans config/feature_sets.yaml. Les nouvelles fonctionnalités incluent le dimensionnement dynamique des positions (risk_manager.py, suggestion 1), la détection des régimes de marché (regime_detector.py, suggestion 4), Safe RL/CVaR-PPO (trade_probability.py, suggestion 7), RL Distributionnel/QR-DQN (trade_probability.py, suggestion 8), et les ensembles de politiques avec vote bayésien (trade_probability.py, suggestion 10).
Note sur les dossiers policies : Le répertoire officiel pour les politiques de routage est src/model/router/policies. Le dossier src/model/policies semble être un résidu et doit être vérifié pour suppression afin d’éviter des conflits d’importation. Consultez troubleshooting.md pour résoudre les erreurs liées à des importations ambiguës.
Note : Cette référence concerne l’API interne du projet, et non l’API xAI. Pour les services xAI, consultez https://x.ai/api.
Conformité : Toutes les références à dxFeed, obs_t, 320 features, et 81 features ont été supprimées pour aligner avec les spécifications de MIA_IA_SYSTEM_v2_2025.
Structure de l’API
L’API est organisée par module, avec une description des classes et fonctions principales, leurs paramètres, retours, et exemples d’utilisation. Les modules sont interconnectés pour former un pipeline de trading robuste, avec des mécanismes de journalisation, d’alertes (via Telegram), et de sauvegardes (AWS S3).
Modules et fonctions clés
1. src/data/data_provider.py
Classe : DataProvider

Description : Gère la récupération des données de marché en temps réel via IQFeed.
Méthodes principales :
__init__(config_path: str = "config/es_config.yaml")
Rôle : Initialise la connexion à IQFeed.
Paramètres :
config_path : Chemin vers le fichier de configuration.


Retour : Aucun.
Exemple :from src.data.data_provider import DataProvider
provider = DataProvider()




fetch_realtime_data(market: str = "ES", interval: str = "1min") -> pd.DataFrame
Rôle : Récupère les données brutes pour un marché donné.
Paramètres :
market : Marché (ex. : "ES", "MNQ").
interval : Intervalle des données (ex. : "1min", "5min").


Retour : DataFrame avec les données brutes (ex. : close, volume, bid_size_level_1).
Exemple :data = provider.fetch_realtime_data(market="ES", interval="1min")
print(data.head())




test_connection() -> bool
Rôle : Vérifie la connexion à IQFeed.
Retour : True si la connexion est réussie, False sinon.
Exemple :if provider.test_connection():
    print("Connexion IQFeed réussie")







2. src/features/shap_weighting.py
Fonction : calculate_shap_weights(data: pd.DataFrame, regime: str) -> pd.DataFrame

Description : Calcule les poids SHAP pour sélectionner les 150 fonctionnalités les plus importantes.
Paramètres :
data : DataFrame contenant les 350 fonctionnalités d’entraînement.
regime : Régime de marché ("trend", "range", "defensive").


Retour : DataFrame avec les colonnes feature et importance.
Exemple :from src.features.shap_weighting import calculate_shap_weights
import pandas as pd
data = pd.DataFrame({f"feature_{i}": np.random.uniform(0, 1, 100) for i in range(350)})
shap_df = calculate_shap_weights(data, regime="trend")
print(shap_df.head())



3. src/model/utils/obs_template.py
Classe : ObsTemplate

Description : Formate le vecteur d’observation (150 fonctionnalités SHAP) pour trading_env.py.
Méthodes principales :
__init__(config_path: str = "config/es_config.yaml", market: str = "ES")
Rôle : Initialise le gestionnaire du vecteur d’observation.
Paramètres :
config_path : Chemin vers le fichier de configuration.
market : Marché (ex. : "ES", "MNQ").


Retour : Aucun.
Exemple :from src.model.utils.obs_template import ObsTemplate
calculator = ObsTemplate(market="ES")




get_shap_features(data: pd.DataFrame, regime: str) -> List[str]
Rôle : Sélectionne les 150 fonctionnalités SHAP.
Paramètres :
data : DataFrame avec les 350 fonctionnalités.
regime : Régime de marché ("trend", "range", "defensive").


Retour : Liste des 150 noms de fonctionnalités.
Exemple :features = calculator.get_shap_features(data, regime="trend")
print(len(features))  # 150




create_observation(data: pd.DataFrame, regime: str = "range") -> np.ndarray
Rôle : Crée un vecteur d’observation de 150 dimensions.
Paramètres :
data : DataFrame avec les 350 fonctionnalités.
regime : Régime de marché.


Retour : Vecteur NumPy de 150 dimensions.
Exemple :observation = calculator.create_observation(data, regime="trend")
print(observation.shape)  # (150,)




validate_obs_template(data: Optional[pd.DataFrame] = None, policy_type: str = "mlp") -> bool
Rôle : Valide la cohérence du vecteur d’observation.
Paramètres :
data : DataFrame à valider (facultatif).
policy_type : Type de politique ("mlp", "transformer").


Retour : True si valide, False sinon.
Exemple :is_valid = calculator.validate_obs_template(data, policy_type="mlp")
print(is_valid)  # True ou False







4. src/model/utils/trading_utils.py
Fonctions principales :

detect_market_regime(data: pd.DataFrame, config_path: Optional[str] = None, market: str = "ES") -> pd.Series
Description : Détecte le régime de marché (tendance, range, défensif).
Paramètres :
data : DataFrame avec les fonctionnalités (ex. : atr_14, adx_14).
config_path : Chemin vers le fichier de configuration (facultatif).
market : Marché (ex. : "ES", "MNQ").


Retour : Série Pandas avec les régimes.
Exemple :from src.model.utils.trading_utils import detect_market_regime
regimes = detect_market_regime(data, market="ES")
print(regimes.value_counts())




adjust_risk_and_leverage(data: pd.DataFrame, regime: pd.Series, config_path: Optional[str] = None, market: str = "ES") -> pd.DataFrame
Description : Ajuste le risque et le levier selon le régime.
Paramètres :
data : DataFrame avec les fonctionnalités.
regime : Série des régimes.
config_path : Chemin vers le fichier de configuration (facultatif).
market : Marché.


Retour : DataFrame avec les colonnes adjusted_leverage et adjusted_risk.
Exemple :adjusted_data = adjust_risk_and_leverage(data, regimes, market="ES")
print(adjusted_data[["adjusted_leverage", "adjusted_risk"]].head())




calculate_profit(trade: Dict[str, Any], config_path: Optional[str] = None, market: str = "ES") -> float
Description : Calcule le profit avec des récompenses adaptatives (méthode 5).
Paramètres :
trade : Dictionnaire avec entry_price, exit_price, position_size, trade_type, etc.
config_path : Chemin vers le fichier de configuration (facultatif).
market : Marché.


Retour : Profit net en USD.
Exemple :trade = {"entry_price": 4000.0, "exit_price": 4050.0, "position_size": 10, "trade_type": "long"}
profit = calculate_profit(trade, market="ES")
print(profit)  # Ex. : 480.0





5. src/model/rl/train_sac.py
Fonction : train_sac(env: TradingEnv, config_path: str = "config/algo_config.yaml", market: str = "ES")

Description : Entraîne un modèle SAC avec fine-tuning (méthode 8), apprentissage en ligne (méthode 10), et meta-learning (méthode 18).
Paramètres :
env : Environnement de trading (TradingEnv).
config_path : Chemin vers le fichier de configuration.
market : Marché.


Retour : Aucun (enregistre les performances dans train_sac_performance.csv).
Exemple :from src.model.envs.trading_env import TradingEnv
from src.model.rl.train_sac import train_sac
env = TradingEnv(market="ES")
train_sac(env, market="ES")



6. src/model/utils/db_maintenance.py
Classe : DBMaintenance

Description : Gère la base de données market_memory.db pour la mémoire contextuelle (méthode 7).
Méthodes principales :
__init__(db_path: str = "data/<market>/market_memory_<market>.db", market: str = "ES")
Rôle : Initialise la connexion à la base de données.
Paramètres :
db_path : Chemin vers la base de données.
market : Marché.


Retour : Aucun.
Exemple :from src.model.utils.db_maintenance import DBMaintenance
db = DBMaintenance(market="ES")




purge_old_data(max_age_days: int = 30) -> None
Rôle : Supprime les données obsolètes (>30 jours).
Paramètres :
max_age_days : Âge maximum des données à conserver.


Retour : Aucun.
Exemple :db.purge_old_data()







7. src/model/utils/alert_manager.py
Classe : AlertManager

Description : Gère l’envoi d’alertes via Telegram, SMS, et email.
Méthodes principales :
__init__(config_path: str = "config/es_config.yaml", market: str = "ES")
Rôle : Initialise le gestionnaire d’alertes.
Paramètres :
config_path : Chemin vers le fichier de configuration.
market : Marché.


Retour : Aucun.
Exemple :from src.model.utils.alert_manager import AlertManager
alert_manager = AlertManager(market="ES")




send_alert(message: str, priority: int = 1) -> bool
Rôle : Envoie une alerte selon la priorité (1=info, 2=warning, 3=error, 4=urgent).
Paramètres :
message : Message à envoyer.
priority : Niveau de priorité.


Retour : True si l’envoi réussit, False sinon.
Exemple :alert_manager.send_alert("Test alert", priority=1)







8. src/model/utils/algo_performance_logger.py
Classe : AlgoPerformanceLogger

Description : Enregistre les performances des algorithmes SAC, PPO, DDPG, PPO-Lagrangian, QR-DQN, et les ensembles.
Méthodes principales :
__init__(config_path: str = "config/algo_config.yaml", market: str = "ES")
Rôle : Initialise le logger de performances.
Paramètres :
config_path : Chemin vers le fichier de configuration.
market : Marché.


Retour : Aucun.
Exemple :from src.model.utils.algo_performance_logger import AlgoPerformanceLogger
logger = AlgoPerformanceLogger(market="ES")




log_performance(algo_type: str, regime: str, reward: float, latency: float, memory: float, **kwargs)
Rôle : Enregistre les performances de base.
Paramètres :
algo_type : Type d’algorithme ("sac", "ppo", "ddpg", "ppo_cvar", "qr_dqn", "ensemble").
regime : Régime de marché.
reward : Récompense observée.
latency : Latence (secondes).
memory : Utilisation mémoire (Mo).
**kwargs : Paramètres supplémentaires (ex. : cvar_loss, qr_dqn_quantiles, ensemble_weights).


Retour : Aucun.
Exemple :logger.log_performance(algo_type="sac", regime="range", reward=100.0, latency=0.5, memory=512.0)




log_extended_performance(algo_type: str, regime: str, reward: float, latency: float, memory: float, finetune_loss: float, online_learning_steps: int, maml_steps: int, error: str = None, **kwargs)
Rôle : Enregistre les performances étendues (méthodes 8, 10, 18).
Paramètres :
Comme ci-dessus, plus :
finetune_loss : Perte de fine-tuning.
online_learning_steps : Étapes d’apprentissage en ligne.
maml_steps : Étapes de meta-learning.
error : Message d’erreur (facultatif).
**kwargs : Paramètres supplémentaires (ex. : cvar_loss, qr_dqn_quantiles, ensemble_weights).


Retour : Aucun.
Exemple :logger.log_extended_performance(algo_type="ppo_cvar", regime="range", reward=100.0, latency=0.5, memory=512.0,
                                finetune_loss=0.01, online_learning_steps=10, maml_steps=5, cvar_loss=0.05)







9. src/risk_management/risk_manager.py
Classe : RiskManager

Description : Gère le dimensionnement dynamique des positions pour minimiser les risques (suggestion 1).
Méthodes principales :
__init__(config_path: str = "config/risk_manager_config.yaml", market: str = "ES")
Rôle : Initialise le gestionnaire de risques.
Paramètres :
config_path : Chemin vers le fichier de configuration.
market : Marché.


Retour : Aucun.
Exemple :from src.risk_management.risk_manager import RiskManager
risk_manager = RiskManager(market="ES")




calculate_position_size(data: pd.DataFrame, regime: str) -> float
Rôle : Calcule la taille des positions en fonction de atr_dynamic et orderflow_imbalance.
Paramètres :
data : DataFrame avec les fonctionnalités (ex. : atr_dynamic, orderflow_imbalance).
regime : Régime de marché ("trend", "range", "defensive").


Retour : Taille de la position (en unités).
Exemple :position_size = risk_manager.calculate_position_size(data, regime="trend")
print(position_size)  # Ex. : 10.0




adjust_leverage(data: pd.DataFrame, regime: str) -> float
Rôle : Ajuste le levier selon le régime et les métriques de risque.
Paramètres :
data : DataFrame avec les fonctionnalités.
regime : Régime de marché.


Retour : Niveau de levier ajusté.
Exemple :leverage = risk_manager.adjust_leverage(data, regime="defensive")
print(leverage)  # Ex. : 1.0







10. src/features/regime_detector.py
Classe : RegimeDetector

Description : Détecte les régimes de marché (tendance, range, défensif) à l’aide d’un modèle HMM (suggestion 4).
Méthodes principales :
__init__(config_path: str = "config/regime_detector_config.yaml", market: str = "ES")
Rôle : Initialise le détecteur de régimes.
Paramètres :
config_path : Chemin vers le fichier de configuration.
market : Marché.


Retour : Aucun.
Exemple :from src.features.regime_detector import RegimeDetector
detector = RegimeDetector(market="ES")




train_hmm(data: pd.DataFrame) -> None
Rôle : Entraîne le modèle HMM avec n_components=3 pour générer hmm_state_distribution.
Paramètres :
data : DataFrame avec les fonctionnalités (ex. : orderflow_data).


Retour : Aucun.
Exemple :detector.train_hmm(data)




detect_regime(data: pd.DataFrame) -> pd.Series
Rôle : Prédit les régimes de marché à l’aide du modèle HMM entraîné.
Paramètres :
data : DataFrame avec les fonctionnalités.


Retour : Série Pandas avec les régimes ("trend", "range", "defensive").
Exemple :regimes = detector.detect_regime(data)
print(regimes.value_counts())







11. src/model/trade_probability.py
Classe : TradeProbabilityPredictor

Description : Prédit la probabilité de succès des trades à l’aide de modèles RL (suggestions 7, 8, 10).
Méthodes principales :
__init__(config_path: str = "config/trade_probability_config.yaml", market: str = "ES")
Rôle : Initialise le prédicteur de probabilité de trading.
Paramètres :
config_path : Chemin vers le fichier de configuration.
market : Marché.


Retour : Aucun.
Exemple :from src.model.trade_probability import TradeProbabilityPredictor
predictor = TradeProbabilityPredictor(market="ES")




predict_trade_success(data: pd.DataFrame, regime: str) -> float
Rôle : Prédit la probabilité de succès d’un trade en utilisant PPO-Lagrangian, QR-DQN, ou vote bayésien.
Paramètres :
data : DataFrame avec les fonctionnalités (ex. : cvar_loss, qr_dqn_quantiles, ensemble_weights).
regime : Régime de marché.


Retour : Probabilité de succès (entre 0 et 1).
Exemple :probability = predictor.predict_trade_success(data, regime="trend")
print(probability)  # Ex. : 0.75




update_model(data: pd.DataFrame, regime: str) -> None
Rôle : Met à jour le modèle RL avec de nouvelles données (apprentissage en ligne, méthode 10).
Paramètres :
data : DataFrame avec les fonctionnalités.
regime : Régime de marché.


Retour : Aucun.
Exemple :predictor.update_model(data, regime="range")







Notes

Conformité : Suppression des références à 320 features, 81 features, et obs_t pour alignement avec les standards actuels.
Scalabilité : L’API est conçue pour supporter de nouveaux modules et sources de données (ex. : API Bloomberg, prévue pour juin 2025).
Validation : Chaque fonction est testée via des tests unitaires dans tests/ (ex. : test_obs_template.py, test_trading_utils.py, test_risk_manager.py, test_regime_detector.py, test_trade_probability.py).
Dépendances : Les nouveaux modules requièrent hmmlearn>=0.2.8,<0.3.0, stable-baselines3>=2.0.0,<3.0.0, et ray[rllib]>=2.0.0,<3.0.0 (voir requirements.txt).
Prochaines étapes : Vérification et suppression du dossier src/model/policies s’il n’est pas utilisé.

Pour plus de détails, consultez :

architecture.md : Vue d’ensemble de l’architecture.
feature_engineering.md : Détails sur les fonctionnalités utilisées par l’API.
methodology.md : Explications des méthodes implémentées (3, 5, 7, 8, 10, 18).
troubleshooting.md : Solutions aux erreurs liées à l’utilisation de l’API.
quickstart.md : Guide synthétique pour démarrer.
modules.md : Vue d’ensemble des modules.
risk_manager.md : Guide pour risk_manager.py.
regime_detector.md : Guide pour regime_detector.py.
trade_probability.md : Guide pour trade_probability.py.

