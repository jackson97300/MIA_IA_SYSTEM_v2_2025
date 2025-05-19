from jinja2 import Environment, FileSystemLoader
import plotly.graph_objects as go

class ReportGenerator:
    def __init__(self, config: Dict):
        self.config = config
        self.env = Environment(loader=FileSystemLoader(BASE_DIR / "reports/templates"))

    def plot_results(self, df: pd.DataFrame, symbol: str, shap_values: Optional[pd.DataFrame] = None) -> Dict:
        figures = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Equity Curve
        equity = np.cumsum(df["reward"].fillna(0))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["timestamp"], y=equity, mode="lines", name="Equity"))
        fig.update_layout(title=f"Equity Curve - {symbol}", xaxis_title="Timestamp", yaxis_title="Equity")
        figures["equity"] = FIGURES_DIR / f"{symbol}_equity_{timestamp}.png"
        fig.write_image(figures["equity"])

        # Profits par Mode/Session
        for group in ["regime", "session"]:
            fig = go.Figure()
            for val in df[group].unique():
                group_df = df[df[group] == val]
                fig.add_trace(go.Bar(x=[val], y=[group_df["reward"].mean()], name=val))
            fig.update_layout(title=f"Profits par {group.capitalize()} - {symbol}", xaxis_title=group.capitalize(), yaxis_title="Profit Moyen")
            figures[f"profits_{group}"] = FIGURES_DIR / f"{symbol}_profits_{group}_{timestamp}.png"
            fig.write_image(figures[f"profits_{group}"])

        # SHAP Importance
        if shap_values is not None:
            fig = go.Figure()
            top_features = shap_values.abs().mean().nlargest(10).index
            fig.add_trace(go.Bar(x=top_features, y=shap_values[top_features].abs().mean()))
            fig.update_layout(title=f"Top 10 SHAP Features - {symbol}", xaxis_title="Feature", yaxis_title="Importance")
            figures["shap"]"] = FIGURES_DIR / f"{symbol}_shap_{timestamp}.png"
            fig.write_image(figures["shap"])

        # Précision des Votes
        fig = go.Figure()
        for model in self.config["models"]:
            fig.add_trace(go.Bar(x=[model], y=[stats[f"vote_accuracy_{model}"]], name=model))
        fig.update_layout(title=f"Précision des Votes par Modèle - {symbol}", xaxis_title="Modèle", yaxis_title="Précision (%)")
        figures["vote_accuracy"] = FIGURES_DIR / f"{symbol}_vote_accuracy_{timestamp}.png"
        fig.write_image(figures["vote_accuracy"])

        # Distribution des Durées
        fig = go.Figure()
        for group in ["regime", "session"]:
            for val in df[group].unique():
                group_df = df[df[group] == val]
                fig.add_trace(go.Histogram(x=group_df["trade_duration"], nbinsx=30, name=f"{group}: {val}"))
        fig.update_layout(title=f"Distribution des Durées - {symbol}", xaxis_title="Durée (min)", yaxis_title="Nombre de Trades")
        figures["durations"] = FIGURES_DIR / f"{symbol}_durations_{timestamp}.png"
        fig.write_image(figures["durations"])

        # Rolling Metrics
        fig = go.Figure()
        for window in self.config["periods"]["rolling"]:
            fig.add_trace(go.Scatter(x=df["timestamp"], y=df[f"rolling_{window}d_win_rate"], mode="lines", name=f"Win Rate {window}d"))
        fig.update_layout(title=f"Rolling Metrics - {symbol}", xaxis_title="Timestamp", yaxis_title="Win Rate (%)")
        figures["rolling"] = FIGURES_DIR / f"{symbol}_rolling_{timestamp}.png"
        fig.write_image(figures["rolling"])

        # Heatmap Mode × Modèle × Session
        heatmap_data = df.pivot_table(values="win_rate", index="regime", columns=["session", "ensemble_weight_sac"])
        fig = go.Figure(data=go.Heatmap(z=heatmap_data.values, x=heatmap_data.columns, y=heatmap_data.index))
        fig.update_layout(title=f"Win Rate Heatmap - {symbol}", xaxis_title="Session/Modèle", yaxis_title="Régime")
        figures["heatmap"] = FIGURES_DIR / f"{symbol}_heatmap_{timestamp}.png"
        fig.write_image(figures["heatmap"])

        return figures

    def save_summary(self, stats: Dict, shap_values: Optional[pd.DataFrame], suggestions: List[str], symbol: str, df: pd.DataFrame) -> Dict:
        template = self.env.get_template("report.md")
        report = template.render(
            date=datetime.now().strftime("%Y-%m-%d"),
            stats=stats,
            shap=shap_values.abs().mean().nlargest(10).to_dict() if shap_values is not None else {},
            suggestions=suggestions,
            symbol=symbol,
            median_duration=df["trade_duration"].median() if "trade_duration" in df else 0,
            percentile_95_duration=df["trade_duration"].quantile(0.95) if "trade_duration" in df else 0
        )
        with open(REPORTS_DIR / f"trade_summary_{symbol}.md", "w") as f:
            f.write(report)
        return {"text": report, "markdown": report, "figures": self.plot_results(df, symbol, shap_values)}