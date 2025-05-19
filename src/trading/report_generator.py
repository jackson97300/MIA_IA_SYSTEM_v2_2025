```python
# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/trading/report_generator.py
# Rôle : Génère des rapports visuels et textuels pour les trades, incluant des narratifs via LLM.
#
# Version : 2.1.3
# Date : 2025-05-15
#
# Dépendances :
# - jinja2>=3.0.0,<4.0.0
# - plotly>=5.0.0,<6.0.0
# - pandas>=2.0.0,<3.0.0
# - numpy>=1.26.4,<2.0.0
# - openai>=1.0.0,<2.0.0
#
# Inputs :
# - Données de trading (pd.DataFrame avec reward, regime, session, trade_duration, timestamp)
# - Métriques (Dict avec win_rate, sharpe_ratio, iv_skew, entry_freq, period, etc.)
# - Configuration via config/market_config.yaml
# - Templates dans reports/templates/
#
# Outputs :
# - Rapports Markdown dans data/reports/*.md
# - Graphiques Plotly dans data/figures/trading/*.png
# - Cache des narratifs dans data/narrative_cache/*.json
#
# Notes :
# - Génère des graphiques (equity curve, profits, SHAP, vote accuracy, durations, rolling metrics, heatmap).
# - Intègre un narratif LLM via GPT-4o-mini pour la Proposition 1.
# - Utilise un cache local pour réduire les appels API.
# - Utilise l’encodage UTF-8 pour tous les fichiers.

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from openai import OpenAI
import plotly.graph_objects as go

# Chemins relatifs
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
FIGURES_DIR = BASE_DIR / "data" / "figures" / "trading"
REPORTS_DIR = BASE_DIR / "data" / "reports"
CACHE_DIR = BASE_DIR / "data" / "narrative_cache"

# Configurer logging
logging.basicConfig(
    filename=BASE_DIR / "data" / "logs" / "report_generator.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, config: Dict):
        self.config = config
        self.env = Environment(loader=FileSystemLoader(BASE_DIR / "reports/templates"))
        self.nlp_client = OpenAI(api_key=config.get("openai_api_key", ""))
        os.makedirs(CACHE_DIR, exist_ok=True)

    def build_report_prompt(self, stats: Dict) -> str:
        """Construit un prompt dynamique pour le narratif LLM (Proposition 1)."""
        return (
            f"En français, rédige un paragraphe narratif engageant à partir de ces données : "
            f"drawdown = {stats.get('max_drawdown', 0) * 100:.1f} %, "
            f"IV skew = {stats.get('iv_skew', 0) * 100:.1f} %, "
            f"fréquence d’entrées = {stats.get('entry_freq', 0):.1f}/jour "
            f"sur la période {stats.get('period', 'N/A')}. "
            f"Propose aussi une action IA en une phrase."
        )

    def generate_narrative(self, stats: Dict) -> str:
        """Génère un narratif via GPT-4o-mini avec cache local."""
        try:
            # Créer un hash des métriques pour le cache
            metrics_key = f"{stats.get('max_drawdown', 0)}_{stats.get('iv_skew', 0)}_{stats.get('entry_freq', 0)}_{stats.get('period', 'N/A')}"
            cache_hash = hashlib.sha256(metrics_key.encode()).hexdigest()
            cache_path = CACHE_DIR / f"{cache_hash}.json"

            # Vérifier le cache
            if cache_path.exists():
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if cached["timestamp"] > (datetime.now() - timedelta(days=7)).isoformat():
                    logger.info("Narrative chargé depuis le cache")
                    return cached["narrative"]

            # Appel GPT-4o-mini
            prompt = self.build_report_prompt(stats)
            response = self.nlp_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
            ).choices[0].message.content

            # Sauvegarder dans le cache
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"timestamp": datetime.now().isoformat(), "narrative": response}, f, indent=4)
            logger.info("Narrative généré et mis en cache")
            return response
        except Exception as e:
            error_msg = f"Erreur génération narrative: {str(e)}"
            logger.error(error_msg)
            return f"Erreur lors de la génération du narratif: {str(e)}"

    def plot_results(self, df: pd.DataFrame, symbol: str, stats: Dict, shap_values: Optional[pd.DataFrame] = None) -> Dict:
        """Génère des graphiques Plotly pour le rapport."""
        try:
            figures = {}
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(FIGURES_DIR, exist_ok=True)

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
            if shap_values is not None and not shap_values.empty:
                fig = go.Figure()
                top_features = shap_values.abs().mean().nlargest(10).index
                fig.add_trace(go.Bar(x=top_features, y=shap_values[top_features].abs().mean()))
                fig.update_layout(title=f"Top 10 SHAP Features - {symbol}", xaxis_title="Feature", yaxis_title="Importance")
                figures["shap"] = FIGURES_DIR / f"{symbol}_shap_{timestamp}.png"
                fig.write_image(figures["shap"])

            # Précision des Votes (suppose vote_accuracy_{model} dans df)
            if any(f"vote_accuracy_{model}" in df.columns for model in self.config.get("models", [])):
                fig = go.Figure()
                for model in self.config.get("models", []):
                    if f"vote_accuracy_{model}" in df.columns:
                        fig.add_trace(go.Bar(x=[model], y=[df[f"vote_accuracy_{model}"].mean()], name=model))
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
            rolling_windows = self.config.get("periods", {}).get("rolling", [])
            if rolling_windows:
                fig = go.Figure()
                for window in rolling_windows:
                    if f"rolling_{window}d_win_rate" in df.columns:
                        fig.add_trace(go.Scatter(x=df["timestamp"], y=df[f"rolling_{window}d_win_rate"], mode="lines", name=f"Win Rate {window}d"))
                fig.update_layout(title=f"Rolling Metrics - {symbol}", xaxis_title="Timestamp", yaxis_title="Win Rate (%)")
                figures["rolling"] = FIGURES_DIR / f"{symbol}_rolling_{timestamp}.png"
                fig.write_image(figures["rolling"])

            # Heatmap Mode × Modèle × Session
            try:
                heatmap_data = df.pivot_table(values="reward", index="regime", columns=["session", "ensemble_weight_sac"], aggfunc="mean")
                fig = go.Figure(data=go.Heatmap(z=heatmap_data.values, x=[f"{col[0]}/{col[1]}" for col in heatmap_data.columns], y=heatmap_data.index))
                fig.update_layout(title=f"Profit Moyen Heatmap - {symbol}", xaxis_title="Session/Modèle", yaxis_title="Régime")
                figures["heatmap"] = FIGURES_DIR / f"{symbol}_heatmap_{timestamp}.png"
                fig.write_image(figures["heatmap"])
            except Exception as e:
                logger.warning(f"Erreur heatmap: {str(e)}")

            # Ajouter le narratif comme annotation
            narrative = self.generate_narrative(stats)
            fig = go.Figure()
            fig.add_annotation(
                x=0.5, y=0.5, text=narrative, showarrow=False, font=dict(size=12), align="center", xref="paper", yref="paper"
            )
            fig.update_layout(title=f"Rapport Narratif - {symbol}", width=800, height=400)
            figures["narrative"] = FIGURES_DIR / f"{symbol}_narrative_{timestamp}.png"
            fig.write_image(figures["narrative"])

            logger.info(f"Graphiques générés pour {symbol}")
            return figures
        except Exception as e:
            error_msg = f"Erreur génération graphiques: {str(e)}"
            logger.error(error_msg)
            return {}

    def save_summary(self, stats: Dict, shap_values: Optional[pd.DataFrame], suggestions: List[str], symbol: str, df: pd.DataFrame) -> Dict:
        """Sauvegarde le rapport avec narratif LLM."""
        try:
            template = self.env.get_template("report.md")
            narrative = self.generate_narrative(stats)
            report = template.render(
                date=datetime.now().strftime("%Y-%m-%d"),
                stats=stats,
                shap=shap_values.abs().mean().nlargest(10).to_dict() if shap_values is not None else {},
                suggestions=suggestions,
                symbol=symbol,
                median_duration=df["trade_duration"].median() if "trade_duration" in df else 0,
                percentile_95_duration=df["trade_duration"].quantile(0.95) if "trade_duration" in df else 0,
                narrative=narrative
            )
            output_path = REPORTS_DIR / f"trade_summary_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            os.makedirs(output_path.parent, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
            figures = self.plot_results(df, symbol, stats, shap_values)
            logger.info(f"Rapport sauvegardé: {output_path}")
            return {"text": report, "markdown": report, "figures": figures, "narrative": narrative}
        except Exception as e:
            error_msg = f"Erreur sauvegarde rapport: {str(e)}"
            logger.error(error_msg)
            return {"text": "", "markdown": "", "figures": {}, "narrative": ""}

