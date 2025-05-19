import itertools
import pandas as pd
import dask.dataframe as dd
import sqlalchemy as sa
import hashlib
from pathlib import Path

class TradeLoader:
    def __init__(self, config: Dict, analyzer: 'TradeAnalyzer'):
        self.config = config
        self.analyzer = analyzer
        self.engine = sa.create_engine("sqlite:///" + str(BASE_DIR / "data/trades.db"))
        self.trade_count = 0

    def load_trades(self, sources: List[str], period: str = None, max_trades: int = None, watermark: bool = True) -> pd.DataFrame:
        dfs = []
        for src in sources:
            if src.endswith(".csv"):
                dfs.append(dd.read_csv(src).compute())
            elif src.endswith(".db"):
                query = "SELECT * FROM trades"
                if watermark:
                    last_id = pd.read_sql("SELECT MAX(trade_id) FROM trade_watermarks", self.engine).iloc[0, 0]
                    query += f" WHERE trade_id > {last_id}"
                dfs.append(pd.read_sql(query, self.engine))
        df = pd.concat(dfs, ignore_index=True)
        df["batch_hash"] = hashlib.sha256(df.to_json().encode()).hexdigest()
        df["session"] = df["timestamp"].apply(self.assign_session)
        df["trade_duration"] = (pd.to_datetime(df["exit_time"]) - pd.to_datetime(df["entry_time"])).dt.total_seconds() / 60 if "entry_time" in df and "exit_time" in df else 5.0
        df.to_parquet(BASE_DIR / f"data/trades/{df['batch_hash']}.parquet")
        self.trade_count += len(df)
        if self.trade_count >= 500:
            self.analyzer.generate_trade_based_report(df.tail(500), threshold=500)
            self.trade_count = self.trade_count % 500 if self.trade_count < 1000 else 0
        if self.trade_count >= 1000:
            self.analyzer.generate_trade_based_report(df.tail(1000), threshold=1000)
            self.trade_count = 0
        with open(AUDIT_LOG_PATH, "a") as f:
            json.dump({"timestamp": datetime.now().isoformat(), "event": "trade_loaded", "batch_hash": df["batch_hash"].iloc[0]}, f)
        return df

    def assign_session(self, timestamp: str) -> str:
        hour = pd.to_datetime(timestamp).hour
        if 2 <= hour < 11:
            return "London"
        elif 13 <= hour < 20:
            return "New York"
        return "Asian"