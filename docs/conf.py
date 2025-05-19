# -*- coding: utf-8 -*-
#
# Fichier de configuration de la documentation de MIA_IA_SYSTEM_v2_2025
#
# Version : 2.1.5
# Date : 2025-05-14
#
# Rôle : Configure Sphinx pour générer la documentation de MIA_IA_SYSTEM_v2_2025, un système de trading avancé pour les contrats à terme ES et MNQ.
#
# Notes :
# - Supporte les fichiers Markdown (.md) et reStructuredText (.rst) dans docs/ (ex. : architecture.md, installation.md, quickstart.md, modules.md, risk_manager.md, regime_detector.md, trade_probability.md).
# - Assure la conformité avec structure.txt (version 2.1.5), supprimant les références à dxFeed, obs_t, 320 features, et 81 features.
# - Note sur les politiques : Le répertoire officiel pour les politiques de routage est src/model/router/policies. Le dossier src/model/policies est un résidu et doit être vérifié pour suppression afin d’éviter des conflits d’importation.
# - Nouveaux modules : Inclut risk_manager.py (dimensionnement dynamique, suggestion 1), regime_detector.py (détection HMM, suggestion 4), trade_probability.py (Safe RL/CVaR-PPO, RL Distributionnel/QR-DQN, vote bayésien, suggestions 7, 8, 10).
# - Dépendances : Requiert hmmlearn>=0.2.8,<0.3.0, stable-baselines3>=2.0.0,<3.0.0, ray[rllib]>=2.0.0,<3.0.0 pour les nouveaux modules.
#

import os
import sys

# Ajoute le répertoire racine du projet à sys.path pour permettre
# l’importation des modules
sys.path.insert(0, os.path.abspath("../.."))

# -- Informations du projet -----------------------------------------------------

project = "MIA_IA_SYSTEM_v2_2025"
copyright = "2025, xAI"
author = "xAI"
release = "2.1.5"  # Version complète
version = "2.1"  # Version majeur.mineur

# -- Configuration générale ---------------------------------------------------

# Extensions Sphinx
extensions = [
    "myst_parser",  # Support pour les fichiers Markdown
    "sphinx.ext.autodoc",  # Génère la documentation à partir des docstrings Python
    "sphinx.ext.napoleon",  # Support pour les styles de docstring Google/NumPy
    "sphinx.ext.viewcode",  # Ajoute des liens vers le code source
    "sphinx.ext.todo",  # Support pour les directives TODO
    "sphinx.ext.intersphinx",  # Lien vers la documentation externe
]

# Suffixes des fichiers source
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Document principal
master_doc = "index"

# Langue
language = "en"

# Motifs à ignorer
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Activer les directives TODO
todo_include_todos = True

# -- Options pour la sortie HTML -------------------------------------------------

# Thème
html_theme = "sphinx_rtd_theme"

# Options du thème
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Répertoire des fichiers statiques
html_static_path = ["_static"]

# Logo
html_logo = "_static/logo.png"  # Placeholder ; remplacez par le logo réel si disponible

# Favicon
html_favicon = "_static/favicon.ico"  # Placeholder ; remplacez si disponible

# CSS personnalisé
html_css_files = ["css/custom.css"]  # Optionnel ; créez custom.css si nécessaire

# -- Options pour MyST-Parser (Markdown) ---------------------------------------

# Activer les extensions MyST pour des fonctionnalités Markdown avancées
myst_enable_extensions = [
    "colon_fence",  # Support pour les blocs ::: délimités
    "deflist",  # Listes de définitions
    "substitution",  # Substitution de variables
    "linkify",  # Détection automatique des liens
]

# -- Options pour autodoc -----------------------------------------------------

# Extraire automatiquement les type hints
autodoc_typehints = "description"

# Inclure les membres documentés et non documentés
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": False,
    "special-members": "__init__",
    "show-inheritance": True,
}

# -- Options pour intersphinx -------------------------------------------------

# Lien vers la documentation externe (ex. : Python, NumPy)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
}

# -- Options pour la sortie HTMLHelp ---------------------------------------------

htmlhelp_basename = "MIA_IA_SYSTEM_v2_2025_doc"

# -- Options pour la sortie LaTeX ------------------------------------------------

latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": "",
}

# Document LaTeX
latex_documents = [
    (
        master_doc,
        "MIA_IA_SYSTEM_v2_2025.tex",
        "Documentation de MIA_IA_SYSTEM_v2_2025",
        "xAI",
        "manual",
    ),
]

# -- Options pour la sortie des pages de manuel ----------------------------------

man_pages = [
    (
        master_doc,
        "mia_ia_system_v2_2025",
        "Documentation de MIA_IA_SYSTEM_v2_2025",
        [author],
        1,
    )
]

# -- Options pour la sortie Texinfo ----------------------------------------------

texinfo_documents = [
    (
        master_doc,
        "MIA_IA_SYSTEM_v2_2025",
        "Documentation de MIA_IA_SYSTEM_v2_2025",
        author,
        "MIA_IA_SYSTEM_v2_2025",
        "Système de trading avancé pour les contrats à terme ES et MNQ.",
        "Divers",
    ),
]
