# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/trading/suggestion_engine.py
# Rôle : Génère des suggestions d’actions basées sur les métriques de performance et les données des trades.
#
# Version : 2.1.3
# Date : 2025-05-15
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0
# - sklearn>=1.2.0,<2.0.0
# - durable-rules>=2.0.0,<3.0.0
#
# Inputs :
# - Métriques de performance (Dict avec win_rate, sharpe_ratio, iv_skew, entry_freq, etc.)
# - Données de trading (pd.DataFrame avec reward, regime, session, etc.)
# - Configuration via config/market_config.yaml
#
# Outputs :
# - Liste de suggestions (List[str]) pour optimiser les trades
# - Logs dans data/logs/suggestion_engine.log
# - Méta-données des suggestions dans market_memory.db (Proposition 2, Étape 1)
#
# Notes :
# - Utilise des règles YAML via durable-rules pour les suggestions basées sur les métriques.
# - Intègre un arbre de décision pour identifier les features clés.
# - Stocke les suggestions dans market_memory.db pour les méta-données.
# - Compatible avec le narratif LLM (Proposition 1) pour des suggestions cohérentes.
# - Utilise l’encodage UTF-8 pour tous les fichiers.

import json
import logging
import os
import time
import sqlite3
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from durable.engine import ruleset
from sklearn.tree import DecisionTreeClassifier

# Chemins relatifs
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
MARKET_MEMORY_DB = BASE_DIR / "data" / "market_memory.db"

# Configurer logging
logging.basicConfig(
    filename=BASE_DIR / "data" / "logs" / "suggestion_engine.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

class SuggestionEngine:
    def __init__(self, config: Dict):
        self.config = config
        if "suggestions" not in config:
            logger.warning("Clé 'suggestions' manquante dans la configuration")
        if "thresholds" not in config or "max_entry_freq" not in config.get("thresholds", {}):
            logger.warning("Clé 'thresholds.max_entry_freq' manquante, utilisation de la valeur par défaut: 5.0")
        self.rules = ruleset("trading_suggestions")
        for rule in self.config.get("suggestions", []):
            try:
                self.rules.assert_rule({"name": rule["name"], "condition": rule["condition"], "action": rule["action"]})
            except Exception as e:
                logger.error(f"Erreur lors de l'ajout de la règle {rule.get('name', 'inconnue')}: {str(e)}\n{traceback.format_exc()}")

    def generate_suggestions(self, stats: Dict, df: pd.DataFrame) -> List[str]:
        """Génère des suggestions basées sur les métriques et les données des trades."""
        try:
            start_time = time.time()
            suggestions = []

            # Appliquer les règles YAML
            for rule in self.rules.get_rules():
                if eval(rule["condition"], {"stats": stats, "df": df}):
                    suggestions.append(rule["action"])

            # Suggestions basées sur l’arbre de décision
            if len(df) > 10 and "reward" in df.columns:
                model = DecisionTreeClassifier().fit(df.drop("reward", axis=1), df["reward"] > 0)
                feature_importance = pd.Series(model.feature_importances_, index=df.drop("reward", axis=1).columns)
                top_feature = feature_importance.idxmax()
                if feature_importance[top_feature] > 0.3:
                    suggestions.append(f"Augmenter la pondération de la feature {top_feature}")

            # Suggestion spécifique pour iv_skew et entry_freq (Proposition 1)
            if stats.get("iv_skew", 0) > 0.3:
                suggestions.append("Réduire l’exposition en raison d’un IV skew élevé")
            if stats.get("entry_freq", 0) > self.config.get("thresholds", {}).get("max_entry_freq", 5.0):
                suggestions.append("Diminuer la fréquence des entrées pour éviter le surtrading")

            # Stocker les suggestions dans market_memory.db (Proposition 2, Étape 1)
            self.log_suggestion_metadata(suggestions, stats)

            logger.info(f"{len(suggestions)} suggestions générées")
            return suggestions
        except Exception as e:
            error_msg = f"Erreur génération suggestions: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return []

    def log_suggestion_metadata(self, suggestions: List[str], stats: Dict) -> None:
        """Stocke les méta-données des suggestions dans market_memory.db."""
        try:
            suggestion_metadata = {
                "timestamp": datetime.now().isoformat(),
                "num_suggestions": len(suggestions),
                "suggestions": suggestions,
                "stats": {
                    "max_drawdown": stats.get("max_drawdown", 0),
                    "iv_skew": stats.get("iv_skew", 0),
                    "entry_freq": stats.get("entry_freq", 0),
                    "period": stats.get("period", "N/A"),
                },
            }
            
            conn = sqlite3.connect(MARKET_MEMORY_DB)
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS suggestion_metadata (
                    timestamp TEXT,
                    event_type TEXT,
                    metadata TEXT
                )
                """
            )
            cursor.execute(
                """
                INSERT INTO suggestion_metadata (timestamp, event_type, metadata)
                VALUES (?, ?, ?)
                """,
                (
                    suggestion_metadata["timestamp"],
                    "suggestion_analysis",
                    json.dumps(suggestion_metadata),
                ),
            )
            conn.commit()
            conn.close()
            logger.info("Méta-données des suggestions stockées dans market_memory.db")
        except Exception as e:
            error_msg = f"Erreur stockage méta-données suggestions: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)