from pandera import DataFrameSchema, Column, Check
from ydata_profiling import ProfileReport

class SchemaValidator:
    def __init__(self, config: Dict):
        self.config = config
        self.schema = DataFrameSchema({
            "action": Column(str, Check.isin(["buy_ES", "sell_ES", "buy_MNQ", "sell_MNQ"])),
            "regime": Column(str, Check.isin(["trend", "range", "defensive"])),
            "reward": Column(float, Check(lambda x: x.notnull())),
            "trade_duration": Column(float, Check(lambda x: x >= 0)),
            "prob_buy_sac": Column(float, Check(lambda x: 0 <= x <= 1)),
            "prob_sell_sac": Column(float, Check(lambda x: 0 <= x <= 1)),
            "prob_wait_sac": Column(float, Check(lambda x: 0 <= x <= 1)),
            "vote_count_buy": Column(int, Check(lambda x: x >= 0)),
            "vote_count_sell": Column(int, Check(lambda x: x >= 0)),
            "vote_count_wait": Column(int, Check(lambda x: x >= 0)),
            "session": Column(str, Check.isin(["London", "New York", "Asian"])),
            "ensemble_weight_sac": Column(str, Check.isin(["buy", "sell", "wait"])),
            "ensemble_weight_ppo": Column(str, Check.isin(["buy", "sell", "wait"])),
            "ensemble_weight_ddpg": Column(str, Check.isin(["buy", "sell", "wait"])),
            "ensemble_weight_lstm": Column(str, Check.isin(["buy", "sell", "wait"])),
            "orderflow_imbalance": Column(float, Check(lambda x: x.notnull())),
            "rsi_14": Column(float, Check(lambda x: 0 <= x <= 100)),
            "news_impact_score": Column(float, Check(lambda x: -1 <= x <= 1)),
            "bid_ask_imbalance": Column(float, Check(lambda x: x.notnull())),
            "trade_aggressiveness": Column(float, Check(lambda x: x.notnull())),
            "iv_skew": Column(float, Check(lambda x: x.notnull())),
            "iv_term_structure": Column(float, Check(lambda x: x.notnull())),
        })

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.schema.validate(df)
        profile = ProfileReport(df, title="Data Quality")
        profile.to_file(BASE_DIR / "data/reports/data_quality.html")
        return df