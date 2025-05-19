Guide de Mise à Jour pour MIA_IA_SYSTEM_v2_2025
Version: 2.1.4Date: 2025-05-13  
Aperçu
Ce guide fournit des instructions pour appliquer les modifications aux fichiers du projet MIA_IA_SYSTEM_v2_2025, conformément aux suggestions 1 à 9. Il inclut des étapes pour enregistrer les fichiers, tester les modifications, renvoyer les fichiers mis à jour, et intégrer la documentation avec Sphinx. Les modifications sont priorisées selon leur importance (très élevée > élevée > moyenne > basse) et respectent les standards de structure.txt (version 2.1.4, 2025-05-13).
Note sur les dossiers policies : Le répertoire officiel pour les politiques de routage est src/model/router/policies. Le dossier src/model/policies semble être un résidu et doit être vérifié pour suppression afin d’éviter toute confusion.
Instructions générales
Enregistrer le guide

Créez D:\MIA_IA_SYSTEM_v2_2025\docs\update_guide.md avec le contenu de ce fichier.
Assurez l’encodage UTF-8.

Appliquer les modifications
Pour chaque fichier modifié :

Ajoutez les sections ou méthodes indiquées sans modifier le contenu existant, sauf si explicitement requis (ex. : remplacement de obs_t).
Vérifiez les dépendances (ex. : imports nécessaires, fichiers YAML mis à jour).
Priorisez les modifications selon leur priorité :
Très élevée : Suggestions 9 (réentraînement), 7 (tests unitaires).
Élevée : Suggestion 8 (fallback SHAP).
Moyenne : Suggestions 2 (loggers), 3 (simulation configurable), 5 (profit factor).
Basse : Non applicable ici.



Tester les modifications
Après chaque ajout, exécutez les tests unitaires associés :
pytest tests/test_mia_switcher.py -v
pytest tests/test_config_manager.py -v
pytest tests/test_feature_pipeline.py -v
pytest tests/test_trade_probability.py -v
pytest tests/test_feature_sets.py -v

Vérifiez les logs générés dans data/logs/<market>/ pour confirmer que les nouveaux buffers et journaux fonctionnent :
ls D:/MIA_IA_SYSTEM_v2_2025/data/logs/ES/
cat D:/MIA_IA_SYSTEM_v2_2025/data/logs/trading/decision_log.csv

Testez le système global :
python src/run_system.py --market ES

Renvoi des fichiers mis à jour

Une fois les modifications appliquées, renvoyez les fichiers modifiés (ex. : mia_switcher.py, config_manager.py) via la conversation.
Ce guide sera mis à jour pour refléter les changements effectués et intégrer les fichiers dans le projet.

Intégration avec Sphinx
Ajoutez update_guide au toctree dans docs/index.rst :
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   architecture
   feature_engineering
   installation
   methodology
   api_reference
   troubleshooting
   quickstart
   roadmap
   modules
   phases
   setup
   usage
   trading_techniques
   tests
   update_guide

Générez la documentation :
cd D:/MIA_IA_SYSTEM_v2_2025/docs
make html

Vérifiez le rendu dans D:/MIA_IA_SYSTEM_v2_2025/docs/_build/html/index.html.
État des modifications
Les fichiers suivants ont été mis à jour ou créés dans le cadre des suggestions 1 à 9 :



Fichier
Suggestion
Priorité
Statut
Notes



config/feature_sets.yaml
1, 8
Élevée
Mis à jour
Ajout des sections ES.training, ES.inference, MNQ.training, MNQ.inference.


data/market_memory.sql
9
Très élevée
Mis à jour
Ajout de la table training_log.


data/logs/trading/decision_log.csv
2
Moyenne
Mis à jour
Ajout des colonnes timestamp, decision, reason, regime.


tests/test_feature_sets.py
8
Élevée
Créé
Tests pour feature_sets.yaml.


config/algo_config.yaml
3, 5
Moyenne
Mis à jour
Ajout de evaluation_steps et max_profit_factor.


.github/workflows/python.yml
7, 9
Très élevée
Mis à jour
Ajout des tests MiaSwitcher et job retrain.


docs/feature_engineering.md
8
Élevée
Mis à jour
Ajout de la section sur le fallback SHAP.


docs/modules.md
2, 7
Moyenne à élevée
Créé
Sections pour PerformanceLogger, SwitchLogger, MiaSwitcher.


docs/api_reference.md
2, 7, 9
Moyenne à élevée
Créé
Références pour PerformanceLogger, SwitchLogger, MiaSwitcher, TradeProbability.


Fichiers en attente de confirmation :

src/strategy/mia_switcher.py
src/model/utils/config_manager.py
Autres fichiers mentionnés dans les conversations précédentes.

Prochaines étapes

Confirmer les fichiers mis à jour : Renvoyez les fichiers modifiés (ex. : mia_switcher.py, config_manager.py) avec un message "… OK" ou précisez les ajustements nécessaires.
Créer ou mettre à jour d’autres fichiers : Si des fichiers comme scripts/retrain_trade_probability.py ou config/es_config.yaml doivent être créés ou modifiés, fournissez les détails.
Valider la documentation : Après avoir appliqué les modifications à docs/index.rst, vérifiez la documentation générée.
Compléter troubleshooting.md : Si la Partie 4 doit être validée ou complétée, partagez-la ou précisez les attentes.

Pour toute question, contactez l’équipe xAI via GitHub Issues ou le canal Telegram configuré dans src/utils/telegram_alert.py.
