# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/trading/trade_loader.py
# Rôle : Charge les données de trades depuis IQFeed, SQLite, ou CSV, assigne les sessions de trading, et versionne en Parquet.
#
# Version : 2.1.3
# Date : 2025-05-15
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0
# - dask>=2023.0.0,<2024.0.0
# - sqlalchemy>=2.0.0,<3.0.0
# - src.data.trading_env
#
# Inputs :
# - Sources : data/iqfeed/*.csv, data/trades.db, data/market_memory.db
# - Configuration via config/market_config.yaml
#
# Outputs :
# - Données de trades (pd.DataFrame avec session, trade_duration)
# - Fichiers Parquet dans data/trades/*.parquet
# - Audit log dans data/audit/audit.log
#
# Notes :
# - Utilise IQFeed via data_provider.py pour les données en temps réel.
# - Assigne les sessions (Londres, New York, Asie) via TradeAnalyzer.assign_session.
# - Gère les watermarks pour éviter les duplications dans SQLite.
# - Déclenche des rapports périodiques tous les 500/1000 trades via TradeAnalyzer.

import hashlib
import os
import time
import traceback
import json
import logging
import pandas as pd
import dask.dataframe as dd
import sqlalchemy as sa
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.data.trading_env import load_trade_history

# Chemins relatifs
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
AUDIT_LOG_PATH = BASE_DIR / "data" / "audit" / "audit.log"
(TRADES_DIR := BASE_DIR / "data" / "trades").mkdir(parents=True, exist_ok=True)
AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Configurer logging
logging.basicConfig(
    filename=BASE_DIR / "data" / "logs" / "trade_loader.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

class TradeLoader:
    def __init__(self, config: Dict, analyzer: Any):
        self.config = config
        self.analyzer = analyzer
        self.engine = sa.create_engine("sqlite:///" + str(BASE_DIR / "data/trades.db"))
        self.trade_count = 0

    def load_trades(self, sources: List[str], period: str = None, max_trades: int = None, watermark: bool = True) -> pd.DataFrame:
        try:
            start_time = time.time()
            dfs = []
            for src in sources:
                if src.endswith(".csv"):
                    dfs.append(dd.read_csv(src).compute())
                elif src.endswith(".db"):
                    query = "SELECT * FROM trades"
                    if watermark:
                        last_id = pd.read_sql("SELECT MAX(trade_id) FROM trade_watermarks", self.engine).iloc[0, 0] or 0
                        query += f" WHERE trade_id > {last_id}"
                    dfs.append(pd.read_sql(query, self.engine))
                elif src == "iqfeed":
                    dfs.append(load_trade_history(period=period, max_trades=max_trades))
            if not dfs:
                raise ValueError("Aucune donnée chargée depuis les sources spécifiées")
            df = pd.concat(dfs, ignore_index=True)
            df["batch_hash"] = hashlib.sha256(df.to_json().encode()).hexdigest()
            df["session"] = df["timestamp"].apply(self.analyzer.assign_session)
            if {"entry_time", "exit_time"}.issubset(df.columns):
                df["trade_duration"] = (
                    pd.to_datetime(df["exit_time"]) - pd.to_datetime(df["entry_time"])
                ).dt.total_seconds() / 60
            else:
                df["trade_duration"] = 5.0
            output_path = TRADES_DIR / f"{df['batch_hash'].iloc[0]}.parquet"
            with open(output_path, "wb") as f:  # Utiliser un bloc with pour éviter les verrouillages
                df.to_parquet(f)
            self.trade_count += len(df)
            if self.trade_count >= 500:
                self.analyzer.generate_trade_based_report(df.tail(500), threshold=500)
                self.trade_count = self.trade_count % 500 if self.trade_count < 1000 else 0
            if self.trade_count >= 1000:
                self.analyzer.generate_trade_based_report(df.tail(1000), threshold=1000)
                self.trade_count = 0
            with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
                json.dump({"timestamp": datetime.now().isoformat(), "event": "trade_loaded", "batch_hash": df["batch_hash"].iloc[0], "num_trades": len(df)}, f)
                f.write("\n")
            self.analyzer.log_performance("load_trades", time.time() - start_time, success=True, num_trades=len(df))
            with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
                json.dump({"timestamp": datetime.now().isoformat(), "event": "load_trades", "sources": sources, "num_trades": len(df), "output_path": str(output_path)}, f)
                f.write("\n")
            return df
        except Exception as e:
            error_msg = f"Erreur chargement trades: {str(e)}\n{traceback.format_exc()}"
            self.analyzer.alert_dispatcher.send_alert(error_msg, priority=5)
            logger.error(error_msg)
            self.analyzer.log_performance("load_trades", time.time() - start_time, success=False, error=str(e))
            with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
                json.dump({"timestamp": datetime.now().isoformat(), "event": "load_trades_failure", "sources": sources, "error": str(e)}, f)
                f.write("\n")
            return pd.DataFrame()