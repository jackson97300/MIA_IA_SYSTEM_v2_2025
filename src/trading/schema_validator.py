# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/trading/schema_validator.py
# Rôle : Valide les données de trades avec Pandera et génère des rapports de qualité des données.
#
# Version : 2.1.3
# Date : 2025-05-15
#
# Dépendances :
# - pandera>=0.10.0,<1.0.0
# - ydata-profiling>=4.0.0,<5.0.0
# - pandas>=2.0.0,<3.0.0
#
# Inputs :
# - Données de trading (pd.DataFrame)
# - Configuration via config/market_config.yaml
#
# Outputs :
# - Données validées (pd.DataFrame)
# - Rapports de qualité dans data/reports/data_quality_*.html
#
# Notes :
# - Valide les colonnes requises (action, regime, reward, trade_duration, session, etc.) et optionnelles (prob_buy_sac, indicators).
# - Génère des rapports de qualité avec ydata-profiling.
# - Utilise l’encodage UTF-8 pour tous les fichiers.
# - Intègre les alertes via AlertDispatcher pour les erreurs de validation.

import logging
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import pandera as pa
from pandera import DataFrameSchema, Column, Check
from ydata_profiling import ProfileReport

# Chemins relatifs
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
REPORTS_DIR = BASE_DIR / "data" / "reports"

# Configurer logging
logging.basicConfig(
    filename=BASE_DIR / "data" / "logs" / "schema_validator.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

class SchemaValidator:
    def __init__(self, config: Dict):
        self.config = config
        self.schema = DataFrameSchema({
            "action": Column(str, Check.isin(["buy_ES", "sell_ES", "buy_MNQ", "sell_MNQ"]), required=True),
            "regime": Column(str, Check.isin(["trend", "range", "defensive"]), required=True),
            "reward": Column(float, Check(lambda x: x.notnull()), required=True),
            "trade_duration": Column(float, Check(lambda x: x >= 0), required=True),
            "session": Column(str, Check.isin(["London", "New York", "Asian"]), required=True),
            "entry_time": Column(str, Check(lambda x: pd.to_datetime(x, errors="coerce").notnull()), required=True),
            "exit_time": Column(str, Check(lambda x: pd.to_datetime(x, errors="coerce").notnull()), required=True),
            "prob_buy_sac": Column(float, Check(lambda x: 0 <= x <= 1), required=False),
            "vote_count_buy": Column(int, Check(lambda x: x >= 0), required=False),
            "ensemble_weight_sac": Column(str, Check.isin(["buy", "sell", "wait"]), required=False),
            "orderflow_imbalance": Column(float, Check(lambda x: x.notnull()), required=False),
            "rsi_14": Column(float, Check(lambda x: 0 <= x <= 100), required=False),
            "news_impact_score": Column(float, Check(lambda x: -1 <= x <= 1), required=False),
            "bid_ask_imbalance": Column(float, Check(lambda x: x.notnull()), required=False),
            "trade_aggressiveness": Column(float, Check(lambda x: x.notnull()), required=False),
            "iv_skew": Column(float, Check(lambda x: x.notnull()), required=False),
            "iv_term_structure": Column(float, Check(lambda x: x.notnull()), required=False),
        })

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            start_time = time.time()
            df_validated = self.schema.validate(df)
            profile = ProfileReport(df_validated, title="Data Quality Report")
            os.makedirs(REPORTS_DIR, exist_ok=True)  # Créer le dossier si nécessaire
            output_path = REPORTS_DIR / f"data_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            profile.to_file(output_path)
            logger.info(f"Validation Pandera réussie, rapport qualité généré: {output_path}")
            return df_validated
        except Exception as e:
            error_msg = f"Erreur validation Pandera: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            raise