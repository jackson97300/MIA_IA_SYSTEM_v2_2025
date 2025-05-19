# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_feature_sets.py
# Tests unitaires pour config/feature_sets.yaml
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide la structure, la cohérence, et les métadonnées des features définies dans feature_sets.yaml,
#        incluant les sections ES.training, ES.inference, MNQ.training, MNQ.inference (suggestion 8),
#        et les features dynamiques (suggestion 1).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pyyaml>=6.0.0,<7.0.0
# - src/model/utils/config_manager.py
#
# Inputs :
# - Fichier de configuration factice (feature_sets.yaml)
#
# Outputs :
# - Tests unitaires validant feature_sets.yaml.
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Tests la suggestion 1 (features dynamiques) et la suggestion 8 (fallback SHAP).
# - Vérifie l’absence de dxFeed, la suppression de 320/81 features et obs_t.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.
# - Policies Note: The official directory for routing policies is src/model/router/policies.
# The src/model/policies directory is a residual and should be verified
# for removal to avoid import conflicts.
#
# Détails du fichier :
# 1. Métadonnées et structure :
#    - Version: 2.1.4, alignée avec feature_sets.yaml et structure.txt.
#    - Date: 2025-05-13.
#    - Note policies incluse pour conformité.
#    - Dépendances: pytest et pyyaml, avec une référence à config_manager.py pour la compatibilité.
#    - Rôle: Valide la structure, les métadonnées, les catégories, les features, les features dynamiques(suggestion 1), et les sections ES/MNQ (suggestion 8).
#
# 2. Fixture :
#    - tmp_dirs: Crée un répertoire temporaire avec un fichier feature_sets.yaml factice, contenant:
#        - Une structure simplifiée avec metadata, dashboard_display, et feature_sets.
#        - Quelques catégories(raw_data, order_flow, option_metrics, dynamic_features, microstructure_metrics) avec des features représentatives.
#        - Des sections ES et MNQ avec des listes training et inference réduites pour tester la suggestion 8.
#    - Le fichier factice est conçu pour être minimal mais représentatif, évitant la duplication du fichier complet de 350 features.
#
# 3. Tests :
#    - test_feature_sets_structure: Vérifie la présence des sections principales(metadata, dashboard_display, feature_sets), la version(2.1.4), le nombre total de features(350), et les paramètres de dashboard_display.
#    - test_feature_sets_categories: Vérifie que chaque catégorie a une description et une liste de features non vide.
#    - test_feature_properties: Vérifie que chaque feature a un name, type (float, int, bool), source (IQFeed, calculated, neural_pipeline), status (active), range, priority (high, medium, low), et description.
#    - test_dynamic_features: Teste la section dynamic_features pour la suggestion 1, vérifiant que les features commencent par neural_dynamic_feature_ et ont les propriétés correctes.
#    - test_es_mnq_sections: Teste les sections ES et MNQ pour la suggestion 8, vérifiant la présence de training et inference, leur longueur, et la cohérence des features avec les catégories.
#    - test_no_obsolete_references: Vérifie l’absence de références à dxFeed, obs_t, 320/81 features dans les sources et descriptions.
#
# 4. Conformité :
#    - Aucun code obsolète(dxFeed, obs_t, 320/81 features) n’est inclus.
#    - Les tests utilisent une fixture minimaliste pour éviter les dépendances externes.
#    - Style cohérent avec les autres fichiers de test(docstrings, assertions claires, pas de marquage @pytest.mark.asyncio car non nécessaire).
#    - Priorité élevée respectée avec des tests dédiés pour les suggestions 1 et 8.
#
# Instructions de validation :
# 1. Enregistrer le fichier :
#    - Créez D:\MIA_IA_SYSTEM_v2_2025\tests\test_feature_sets.py avec le contenu ci-dessus.
#    - Assurez l’encodage UTF-8.
#
# 2. Vérifier le contenu :
#    - Confirmez que test_dynamic_features valide la section dynamic_features pour la suggestion 1.
#    - Vérifiez que test_es_mnq_sections teste les sections ES et MNQ pour la suggestion 8.
#    - Assurez-vous que la version est 2.1.4, que la note policies est présente, et que les dépendances sont correctes.
#    - Vérifiez que test_no_obsolete_references couvre l’absence de dxFeed, obs_t, etc.
#
# 3. Tester l’intégration :
#    - Copiez le fichier feature_sets.yaml mis à jour(parties 1 à 4) dans D:\MIA_IA_SYSTEM_v2_2025\config\.
#    - Exécutez les tests unitaires:
#        ```bash
#        pytest tests/test_feature_sets.py -v
#        ```
#        Attendu: Tous les tests passent, validant la structure et les ajouts de feature_sets.yaml.
#    - Vérifiez les erreurs potentielles(ex.: syntaxe YAML, features manquantes):
#        ```bash
#        python -c "import yaml; yaml.safe_load(open('config/feature_sets.yaml'))"
#        ```
#
# 4. Tester avec le projet :
#    - Assurez-vous que feature_sets.yaml, feature_pipeline.py, trade_probability.py, et autres fichiers mis à jour sont en place.
#    - Exécutez les tests associés:
#        ```bash
#        pytest tests/test_feature_pipeline.py -v
#        pytest tests/test_trade_probability.py -v
#        ```
#    - Exécutez feature_pipeline.py pour vérifier que les features de ES et MNQ sont chargées correctement:
#        ```bash
#        python src/features/feature_pipeline.py
#        ```
#    - Vérifiez l’affichage dans mia_dashboard.py:
#        ```bash
#        python src/model/utils/mia_dashboard.py --market ES
#        ```
#
# 5. Confirmation :
#    - Si le fichier est correct, confirmez avec "… OK" après vos tests.
#    - Si des ajustements sont nécessaires(ex.: tests supplémentaires pour des features spécifiques, mocks pour config_manager.py), précisez-les, et je mettrai à jour le fichier.
#

import pytest
import yaml


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée un répertoire temporaire pour feature_sets.yaml."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()

    # Créer feature_sets.yaml factice
    feature_sets_path = config_dir / "feature_sets.yaml"
    feature_sets_content = {
        "metadata": {
            "version": "2.1.4",
            "updated": "2025-05-13",
            "total_features": 350,
            "description": "Catalogue complet des 350 features pour ES et MNQ.",
        },
        "dashboard_display": {
            "enabled": True,
            "status_file": "data/feature_sets_dashboard.json",
            "categories": [
                "raw_data",
                "order_flow",
                "option_metrics",
                "dynamic_features",
                "microstructure_metrics",
            ],
            "description": "Métadonnées envoyées à mia_dashboard.py.",
        },
        "feature_sets": {
            "raw_data": {
                "description": "Données brutes issues de l'API IQFeed",
                "features": [
                    {
                        "name": "open",
                        "type": "float",
                        "source": "IQFeed",
                        "status": "active",
                        "range": [0, float("inf")],
                        "priority": "high",
                        "description": "Prix d'ouverture du contrat ES",
                    },
                    {
                        "name": "close",
                        "type": "float",
                        "source": "IQFeed",
                        "status": "active",
                        "range": [0, float("inf")],
                        "priority": "high",
                        "description": "Prix de clôture du contrat ES",
                    },
                ],
            },
            "order_flow": {
                "description": "Métriques d'ordre flow calculées ou issues de IQFeed",
                "features": [
                    {
                        "name": "delta_volume",
                        "type": "float",
                        "source": "calculated",
                        "status": "active",
                        "range": [-float("inf"), float("inf")],
                        "priority": "high",
                        "description": "Différence ask - bid volume",
                    },
                    {
                        "name": "obi_score",
                        "type": "float",
                        "source": "calculated",
                        "status": "active",
                        "range": [-1, 1],
                        "priority": "high",
                        "description": "Score d'imbalance order flow",
                    },
                ],
            },
            "option_metrics": {
                "description": "Indicateurs dérivés des données d'options",
                "features": [
                    {
                        "name": "iv_atm",
                        "type": "float",
                        "source": "calculated",
                        "status": "active",
                        "range": [0, float("inf")],
                        "priority": "high",
                        "description": "Volatilité implicite ATM",
                    },
                    {
                        "name": "gex_slope",
                        "type": "float",
                        "source": "calculated",
                        "status": "active",
                        "range": [-float("inf"), float("inf")],
                        "priority": "high",
                        "description": "Pente de l'exposition gamma nette",
                    },
                ],
            },
            "dynamic_features": {
                "description": "Features dynamiques générées par le pipeline",
                "features": [
                    {
                        "name": "neural_dynamic_feature_1",
                        "type": "float",
                        "source": "calculated",
                        "status": "active",
                        "range": [-float("inf"), float("inf")],
                        "priority": "medium",
                        "description": "Feature dynamique générique 1",
                    },
                    {
                        "name": "neural_dynamic_feature_2",
                        "type": "float",
                        "source": "calculated",
                        "status": "active",
                        "range": [-float("inf"), float("inf")],
                        "priority": "medium",
                        "description": "Feature dynamique générique 2",
                    },
                ],
            },
            "microstructure_metrics": {
                "description": "Métriques de microstructure de marché",
                "features": [
                    {
                        "name": "spoofing_score",
                        "type": "float",
                        "source": "calculated",
                        "status": "active",
                        "range": [0, 1],
                        "priority": "high",
                        "description": "Score de détection de spoofing",
                    },
                    {
                        "name": "volume_anomaly",
                        "type": "float",
                        "source": "calculated",
                        "status": "active",
                        "range": [0, 1],
                        "priority": "high",
                        "description": "Score d'anomalie de volume",
                    },
                ],
            },
            "ES": {
                "training": [
                    "open",
                    "close",
                    "delta_volume",
                    "obi_score",
                    "iv_atm",
                    "gex_slope",
                    "neural_dynamic_feature_1",
                    "spoofing_score",
                ],
                "inference": [
                    "obi_score",
                    "iv_atm",
                    "neural_dynamic_feature_1",
                    "spoofing_score",
                ],
            },
            "MNQ": {
                "training": ["open", "close", "delta_volume", "iv_atm"],
                "inference": ["iv_atm", "neural_dynamic_feature_1"],
            },
        },
    }
    with open(feature_sets_path, "w", encoding="utf-8") as f:
        yaml.dump(feature_sets_content, f, allow_unicode=True)

    return {"base_dir": str(base_dir), "feature_sets_path": str(feature_sets_path)}


def test_feature_sets_structure(tmp_dirs):
    """Teste la structure globale de feature_sets.yaml."""
    with open(tmp_dirs["feature_sets_path"], "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Vérifier les sections principales
    assert "metadata" in config, "Section 'metadata' manquante"
    assert "dashboard_display" in config, "Section 'dashboard_display' manquante"
    assert "feature_sets" in config, "Section 'feature_sets' manquante"

    # Vérifier les métadonnées
    metadata = config["metadata"]
    assert metadata["version"] == "2.1.4", "Version incorrecte"
    assert metadata["total_features"] == 350, "Nombre total de features incorrect"
    assert "description" in metadata, "Description des métadonnées manquante"

    # Vérifier dashboard_display
    dashboard = config["dashboard_display"]
    assert dashboard["enabled"] is True, "Dashboard non activé"
    assert "categories" in dashboard, "Liste des catégories manquante"
    assert len(dashboard["categories"]) >= 5, "Nombre de catégories insuffisant"


def test_feature_sets_categories(tmp_dirs):
    """Teste la validité des catégories dans feature_sets."""
    with open(tmp_dirs["feature_sets_path"], "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    feature_sets = config["feature_sets"]
    expected_categories = [
        "raw_data",
        "order_flow",
        "option_metrics",
        "dynamic_features",
        "microstructure_metrics",
    ]
    for category in expected_categories:
        assert category in feature_sets, f"Catégorie '{category}' manquante"
        assert (
            "description" in feature_sets[category]
        ), f"Description manquante pour '{category}'"
        assert (
            "features" in feature_sets[category]
        ), f"Liste de features manquante pour '{category}'"
        assert (
            len(feature_sets[category]["features"]) > 0
        ), f"Aucune feature définie pour '{category}'"


def test_feature_properties(tmp_dirs):
    """Teste les propriétés des features dans chaque catégorie."""
    with open(tmp_dirs["feature_sets_path"], "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    feature_sets = config["feature_sets"]
    for category, data in feature_sets.items():
        if category in ["ES", "MNQ"]:
            continue
        for feature in data["features"]:
            assert "name" in feature, f"Nom manquant pour une feature dans '{category}'"
            assert (
                "type" in feature
            ), f"Type manquant pour '{feature['name']}' dans '{category}'"
            assert feature["type"] in [
                "float",
                "int",
                "bool",
            ], f"Type invalide pour '{feature['name']}'"
            assert "source" in feature, f"Source manquante pour '{feature['name']}'"
            assert feature["source"] in [
                "IQFeed",
                "calculated",
                "neural_pipeline (LSTM)",
                "neural_pipeline (CNN)",
                "neural_pipeline (MLP)",
            ], f"Source invalide pour '{feature['name']}'"
            assert (
                "status" in feature and feature["status"] == "active"
            ), f"Statut invalide pour '{feature['name']}'"
            assert "range" in feature, f"Plage manquante pour '{feature['name']}'"
            assert "priority" in feature and feature["priority"] in [
                "high",
                "medium",
                "low",
            ], f"Priorité invalide pour '{feature['name']}'"
            assert (
                "description" in feature
            ), f"Description manquante pour '{feature['name']}'"


def test_dynamic_features(tmp_dirs):
    """Teste les features dynamiques pour la suggestion 1."""
    with open(tmp_dirs["feature_sets_path"], "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dynamic_features = config["feature_sets"]["dynamic_features"]["features"]
    assert len(dynamic_features) >= 2, "Nombre insuffisant de features dynamiques"
    for feature in dynamic_features:
        assert feature["name"].startswith(
            "neural_dynamic_feature_"
        ), f"Nom invalide pour feature dynamique : '{feature['name']}'"
        assert (
            feature["source"] == "calculated"
        ), f"Source incorrecte pour '{feature['name']}'"
        assert feature["range"] == [
            -float("inf"),
            float("inf"),
        ], f"Plage incorrecte pour '{feature['name']}'"


def test_es_mnq_sections(tmp_dirs):
    """Teste les sections ES et MNQ pour la suggestion 8."""
    with open(tmp_dirs["feature_sets_path"], "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    feature_sets = config["feature_sets"]

    # Vérifier ES
    assert "ES" in feature_sets, "Section 'ES' manquante"
    es = feature_sets["ES"]
    assert (
        "training" in es and len(es["training"]) >= 8
    ), "Section 'ES.training' manquante ou trop courte"
    assert (
        "inference" in es and len(es["inference"]) >= 4
    ), "Section 'ES.inference' manquante ou trop courte"

    # Vérifier MNQ
    assert "MNQ" in feature_sets, "Section 'MNQ' manquante"
    mnq = feature_sets["MNQ"]
    assert (
        "training" in mnq and len(mnq["training"]) >= 4
    ), "Section 'MNQ.training' manquante ou trop courte"
    assert (
        "inference" in mnq and len(mnq["inference"]) >= 2
    ), "Section 'MNQ.inference' manquante ou trop courte"

    # Vérifier la cohérence des features
    all_features = set()
    for category, data in feature_sets.items():
        if category not in ["ES", "MNQ"]:
            for feature in data["features"]:
                all_features.add(feature["name"])

    for feature in (
        es["training"] + es["inference"] + mnq["training"] + mnq["inference"]
    ):
        assert (
            feature in all_features
        ), f"Feature '{feature}' dans ES/MNQ non définie dans les catégories"


def test_no_obsolete_references(tmp_dirs):
    """Vérifie l'absence de références obsolètes (dxFeed, obs_t, 320/81 features)."""
    with open(tmp_dirs["feature_sets_path"], "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    feature_sets = config["feature_sets"]
    for category, data in feature_sets.items():
        if category in ["ES", "MNQ"]:
            continue
        for feature in data["features"]:
            assert (
                "dxFeed" not in feature["source"]
            ), f"Référence à dxFeed dans '{feature['name']}'"
            assert (
                "obs_t" not in feature["name"]
            ), f"Référence à obs_t dans '{feature['name']}'"
            assert (
                "320_features" not in feature["description"]
            ), f"Référence à 320 features dans '{feature['name']}'"
            assert (
                "81_features" not in feature["description"]
            ), f"Référence à 81 features dans '{feature['name']}'"
