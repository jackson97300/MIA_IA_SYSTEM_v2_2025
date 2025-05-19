import itertools
import numpy as np
import pandas as pd

class MetricsFactory:
    @staticmethod
    def compute_sharpe(df: pd.DataFrame) -> float:
        rewards = df["reward"].fillna(0).values
        return (rewards.mean() / rewards.std()) * np.sqrt(252) if rewards.std() > 0 else 0

    @staticmethod
    def compute_cvar(df: pd.DataFrame, alpha: float = 0.05, threshold: float = 0.1) -> float:
        losses = -df["reward"].fillna(0).values
        return np.percentile(losses, alpha * 100)

    @staticmethod
    def compute_qr_dqn(df: pd.DataFrame) -> float:
        rewards = df["reward"].fillna(0).values
        return np.var(np.percentile(rewards, [10, 50, 90]))

class MetricsComputer:
    def __init__(self, config: Dict):
        self.config = config

    def _base_metrics(self, df: pd.DataFrame) -> Dict:
        rewards = df["reward"].fillna(0).values
        consecutive_losses = max(len(list(g)) for k, g in itertools.groupby(rewards, lambda x: x < 0) if k) if len(rewards) > 0 else 0
        equity = np.cumsum(rewards)
        return {
            "total_trades": len(rewards),
            "win_rate": (rewards > 0).mean() * 100,
            "profit_factor": rewards[rewards > 0].sum() / abs(rewards[rewards < 0].sum()) if (rewards < 0).sum() != 0 else np.inf,
            "sharpe_ratio": MetricsFactory.compute_sharpe(df),
            "max_drawdown": np.min(equity - np.maximum.accumulate(equity)) if len(equity) > 0 else 0,
            "total_return": equity[-1] if len(equity) > 0 else 0,
            "avg_win": rewards[rewards > 0].mean() if (rewards > 0).sum() > 0 else 0,
            "avg_loss": rewards[rewards < 0].mean() if (rewards < 0).sum() != 0 else 0,
            "risk_reward_ratio": abs(rewards[rewards > 0].mean() / rewards[rewards < 0].mean()) if (rewards < 0).sum() != 0 else np.inf,
            "cvar_loss": MetricsFactory.compute_cvar(df),
            "qr_dqn_quantile_var": MetricsFactory.compute_qr_dqn(df),
            "avg_duration": df["trade_duration"].mean() if "trade_duration" in df else 0,
            "consecutive_losses": consecutive_losses
        }

    def compute_metrics(self, df: pd.DataFrame, neural_pipeline: Optional[NeuralPipeline] = None, rolling: List[int] = None) -> Dict:
        stats = self._base_metrics(df)
        stats["by_regime"] = {regime: self._base_metrics(df[df["regime"] == regime]) for regime in df["regime"].unique()}
        stats["by_session"] = {session: self._base_metrics(df[df["session"] == session]) for session in df["session"].unique()}
        for model in self.config["models"]:
            stats[f"vote_accuracy_{model}"] = (
                ((df[f"ensemble_weight_{model}"] == "buy") & (df["reward"] > 0)) |
                ((df[f"ensemble_weight_{model}"] == "sell") & (df["reward"] < 0))
            ).mean() * 100
        if rolling:
            for window in rolling:
                rolling_df = df[df["timestamp"] >= (datetime.now() - timedelta(days=window))]
                stats[f"rolling_{window}d"] = self._base_metrics(rolling_df)
        if neural_pipeline and all(col in df.columns for col in ["timestamp", "open", "high", "low", "close", "volume", "atr_14", "adx_14", "gex", "oi_peak_call_near", "gamma_wall_call", "bid_size_level_1", "ask_size_level_1"]):
            cache_key = hashlib.sha256(df.to_json().encode()).hexdigest()
            if cache_key in self.analyzer.neural_cache:
                neural_result = self.analyzer.neural_cache[cache_key]
            else:
                neural_result = self.with_retries(lambda: neural_pipeline.run(
                    df[["timestamp", "open", "high", "low", "close", "volume", "atr_14", "adx_14"]],
                    df[["timestamp", "gex", "oi_peak_call_near", "gamma_wall_call"]],
                    df[["timestamp", "bid_size_level_1", "ask_size_level_1"]]
                ))
                self.analyzer.neural_cache[cache_key] = neural_result
                if len(self.analyzer.neural_cache) > MAX_CACHE_SIZE:
                    self.analyzer.neural_cache.popitem(last=False)
            df["neural_regime"] = neural_result["regime"]
            df["predicted_volatility"] = neural_result["volatility"]
            stats["by_neural_regime"] = {regime: self._base_metrics(df[df["neural_regime"] == regime]) for regime in df["neural_regime"].unique()}
        return stats