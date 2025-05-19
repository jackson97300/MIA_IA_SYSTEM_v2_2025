# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_context_aware_filter.py
# Tests unitaires pour ContextAwareFilter.
# Version : 2.1.2
# Date : 2025-05-07
# Rôle : Valide compute_contextual_metrics avec 19 features contextuelles (16 nouvelles + 3 existantes),
#        données IQFeed, et SHAP (méthode 17).
# Dépendances : pytest, pandas==1.5.0, numpy==1.23.0, psutil==5.9.0, matplotlib==3.7.0
# Inputs : config/es_config.yaml (mock), data/features/feature_importance.csv (mock),
#          data/events/macro_events.csv (mock), data/events/event_volatility_history.csv (mock)
# Outputs : Aucun (tests)

import os

import numpy as np
import pandas as pd
import pytest

from src.api.context_aware_filter import ContextAwareFilter


@pytest.fixture
def calculator(tmp_path):
    config_path = tmp_path / "config" / "es_config.yaml"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        f.write(
            """
        context_aware_filter:
          buffer_size: 100
          max_cache_size: 1000
          cache_hours: 24
        """
        )
    return ContextAwareFilter(config_path=str(config_path))


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-04-14 09:00", periods=100, freq="1min"),
            "close": np.random.normal(5100, 10, 100),
        }
    )


@pytest.fixture
def news_data():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-04-14 09:00", periods=100, freq="1min"),
            "sentiment_score": np.random.uniform(-1, 1, 100),
            "volume": np.random.randint(1, 10, 100),
        }
    )


@pytest.fixture
def calendar_data():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-04-14 09:00", periods=100, freq="1min"),
            "severity": np.random.uniform(0, 1, 100),
            "weight": np.random.uniform(0, 1, 100),
        }
    )


@pytest.fixture
def futures_data():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-04-14 09:00", periods=100, freq="1min"),
            "near_price": np.random.normal(5100, 10, 100),
            "far_price": np.random.normal(5120, 10, 100),
        }
    )


@pytest.fixture
def expiry_dates():
    return pd.Series(pd.date_range("2025-04-15", periods=100, freq="1d"))


@pytest.fixture
def macro_events(tmp_path):
    macro_events_path = tmp_path / "events" / "macro_events.csv"
    os.makedirs(os.path.dirname(macro_events_path), exist_ok=True)
    macro_events = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2025-04-14 08:00",
                    "2025-04-14 09:10",
                    "2025-04-14 10:00",
                    "2025-04-13 09:00",
                    "2025-04-15 09:00",
                ]
            ),
            "event_type": [
                "CPI Release",
                "FOMC Meeting",
                "Earnings Report",
                "GDP Release",
                "Jobs Report",
            ],
            "event_impact_score": [0.8, 0.9, 0.6, 0.7, 0.85],
        }
    )
    macro_events.to_csv(macro_events_path, index=False)
    return macro_events_path


@pytest.fixture
def volatility_history(tmp_path):
    volatility_history_path = tmp_path / "events" / "event_volatility_history.csv"
    os.makedirs(os.path.dirname(volatility_history_path), exist_ok=True)
    volatility_history = pd.DataFrame(
        {
            "event_type": [
                "CPI Release",
                "FOMC Meeting",
                "Earnings Report",
                "GDP Release",
                "Jobs Report",
            ],
            "volatility_impact": [0.15, 0.20, 0.10, 0.12, 0.18],
            "timestamp": pd.to_datetime(["2025-04-01"] * 5),
        }
    )
    volatility_history.to_csv(volatility_history_path, index=False)
    return volatility_history_path


@pytest.fixture
def shap_features(tmp_path):
    shap_file = tmp_path / "features" / "feature_importance.csv"
    shap_features = [
        "event_volatility_impact",
        "event_timing_proximity",
        "event_frequency_24h",
        "news_sentiment_momentum",
        "news_event_proximity",
        "macro_event_severity",
        "time_to_expiry_proximity",
        "economic_calendar_weight",
        "news_volume_spike",
        "news_volume_1h",
        "news_volume_1d",
        "news_sentiment_acceleration",
        "macro_event_momentum",
        "month_of_year_sin",
        "month_of_year_cos",
        "week_of_month_sin",
        "week_of_month_cos",
        "roll_yield_curve",
    ] + ["feature_" + str(i) for i in range(132)]
    pd.DataFrame(
        {
            "feature": shap_features,
            "importance": np.random.uniform(0, 1, len(shap_features)),
        }
    ).to_csv(shap_file, index=False)
    return shap_file


def test_compute_contextual_metrics(
    calculator,
    sample_data,
    news_data,
    calendar_data,
    futures_data,
    expiry_dates,
    macro_events,
    volatility_history,
    shap_features,
):
    result = calculator.compute_contextual_metrics(
        sample_data,
        news_data,
        calendar_data,
        futures_data,
        expiry_dates,
        macro_events,
        volatility_history,
    )
    assert isinstance(result, pd.DataFrame), "Résultat invalide"
    expected_cols = [
        "event_volatility_impact",
        "event_timing_proximity",
        "event_frequency_24h",
        "news_sentiment_momentum",
        "news_event_proximity",
        "macro_event_severity",
        "time_to_expiry_proximity",
        "economic_calendar_weight",
        "news_volume_spike",
        "news_volume_1h",
        "news_volume_1d",
        "news_sentiment_acceleration",
        "macro_event_momentum",
        "month_of_year_sin",
        "month_of_year_cos",
        "week_of_month_sin",
        "week_of_month_cos",
        "roll_yield_curve",
    ]
    assert all(col in result.columns for col in expected_cols), "Features manquantes"
    assert len(result) == len(sample_data), "Nombre de lignes incorrect"
    assert os.path.exists(CSV_LOG_PATH), "Logs de performance non générés"
    assert os.path.exists(SNAPSHOT_DIR), "Snapshots non générés"
    assert os.path.exists(FIGURES_DIR), "Figures non générées"


def test_calculate_event_volatility_impact(
    calculator, macro_events, volatility_history, shap_features
):
    macro_events = pd.read_csv(macro_events)
    volatility_history = pd.read_csv(volatility_history)
    timestamp = pd.Timestamp("2025-04-14 09:00")
    result = calculator.calculate_event_volatility_impact(
        macro_events, volatility_history, timestamp
    )
    assert isinstance(result, float), "Résultat invalide"
    assert result >= 0, "Impact de volatilité négatif"


def test_parse_news_data(calculator, news_data, shap_features):
    result = calculator.parse_news_data(news_data)
    assert isinstance(result, pd.DataFrame), "Résultat invalide"
    assert "sentiment_score" in result.columns, "sentiment_score manquant"
    assert "volume" in result.columns, "volume manquant"
