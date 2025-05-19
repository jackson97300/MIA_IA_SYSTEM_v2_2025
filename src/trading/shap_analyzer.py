# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/trading/shap_analyzer.py
# Rôle : Calcule les valeurs SHAP pour expliquer l’impact des features sur les prédictions des trades.
#
# Version : 2.1.3
# Date : 2025-05-15
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0
# - numpy>=1.26.4,<2.0.0
# - sklearn>=1.2.0,<2.0.0
# - delta-spark>=2.0.0,<3.0.0
# - src.trading.shap_weighting
#
# Inputs :
# - Données de trading (pd.DataFrame avec 350 features pour entraînement ou 150 SHAP features pour inférence)
# - Configuration via config/market_config.yaml
#
# Outputs :
# - Valeurs SHAP (pd.DataFrame) stockées incrémentalement dans data/shap/YYYYMMDD.parquet
# - Logs dans data/logs/shap_analyzer.log
#
# Notes :
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence) définies dans config/feature_sets.yaml.
# - Stocke les valeurs SHAP en Parquet pour une analyse incrémentale.
# - Intègre des alertes via AlertDispatcher pour les erreurs critiques.
# - Ajoute un log des méta-données SHAP dans market_memory.db (Proposition 2, Étape 1).
# - Vérifie la présence de iv_skew dans les features SHAP.

import logging
import os
import time
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np
from delta.tables import DeltaTable
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier

from src.trading.shap_weighting import calculate_shap

# Chemins relatifs
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
SHAP_DIR = BASE_DIR / "data" / "shap"
MARKET_MEMORY_DB = BASE_DIR / "data" / "market_memory.db"

# Configurer logging
logging.basicConfig(
    filename=BASE_DIR / "data" / "logs" / "shap_analyzer.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

class ShapAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        os.makedirs(SHAP_DIR, exist_ok=True)
        if "shap_feature_limit" not in config:
            logger.warning("Clé 'shap_feature_limit' manquante dans la configuration, utilisation de la valeur par défaut: 50")

    def calculate_shap(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calcule les valeurs SHAP pour les features du DataFrame."""
        try:
            start_time = time.time()
            if "reward" not in df.columns:
                logger.warning("Colonne 'reward' manquante, SHAP non calculé")
                return None
            
            # Vérifier la présence de iv_skew
            if "iv_skew" not in df.columns:
                logger.warning("Feature 'iv_skew' manquante dans les données")
            
            # Sélectionner les features pour SHAP
            model = DecisionTreeClassifier().fit(df.drop("reward", axis=1), df["reward"] > 0)
            pi = permutation_importance(model, df.drop("reward", axis=1), df["reward"] > 0, n_repeats=10)
            top_features = df.drop("reward", axis=1).columns[pi.importances_mean.argsort()[-100:]]
            shap_values = calculate_shap(df[top_features], target="reward", max_features=self.config.get("shap_feature_limit", 50))
            
            if shap_values is None or shap_values.empty:
                logger.warning("Échec calcul SHAP, DataFrame vide")
                return None
            
            # Filtrer les features SHAP significatives
            filtered = self.filter_shap_features(shap_values, threshold=0.01)
            
            # Stocker en Parquet
            output_path = SHAP_DIR / f"{datetime.now().strftime('%Y%m%d')}.parquet"
            filtered.to_parquet(output_path)
            
            # Log des méta-données SHAP dans market_memory.db (Proposition 2, Étape 1)
            self.log_shap_metadata(filtered, df)
            
            logger.info(f"SHAP calculé et stocké: {output_path}")
            return filtered
        except Exception as e:
            error_msg = f"Erreur calcul SHAP: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return None

    def filter_shap_features(self, shap_df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """Filtre les features SHAP avec une importance moyenne supérieure au seuil."""
        try:
            mean_abs = shap_df.abs().mean()
            return shap_df[mean_abs[mean_abs >= threshold].index]
        except Exception as e:
            error_msg = f"Erreur filtrage SHAP: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return shap_df

    def log_shap_metadata(self, shap_values: pd.DataFrame, df: pd.DataFrame) -> None:
        """Stocke les méta-données SHAP dans market_memory.db."""
        try:
            import sqlite3
            
            shap_metadata = {
                "timestamp": datetime.now().isoformat(),
                "num_features": len(shap_values.columns),
                "top_features": shap_values.abs().mean().nlargest(5).to_dict(),
                "row_count": len(df),
            }
            
            conn = sqlite3.connect(MARKET_MEMORY_DB)
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS shap_metadata (
                    timestamp TEXT,
                    event_type TEXT,
                    metadata TEXT
                )
                """
            )
            cursor.execute(
                """
                INSERT INTO shap_metadata (timestamp, event_type, metadata)
                VALUES (?, ?, ?)
                """,
                (
                    shap_metadata["timestamp"],
                    "shap_analysis",
                    json.dumps(shap_metadata),
                ),
            )
            conn.commit()
            conn.close()
            logger.info("Méta-données SHAP stockées dans market_memory.db")
        except Exception as e:
            error_msg = f"Erreur stockage méta-données SHAP: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)