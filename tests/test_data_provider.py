# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_data_provider.py
# Tests unitaires pour src/data/data_provider.py
# Version : 2.1.3
# Date : 2025-05-09
# Rôle : Valide la collecte de données, la clusterisation, les
# visualisations, et les logs.

import os
from datetime import datetime

import pandas as pd
import pytest

from src.data.data_provider import CsvDataProvider, get_data_provider


@pytest.fixture
def csv_provider(tmp_path):
    """Fixture pour CsvDataProvider avec un fichier CSV temporaire."""
    config_path = tmp_path / "market_config.yaml"
    config = {
        "data_feed": "csv",
        "data_sources": {"valid_symbols": ["ES"], "parallel_load": False},
        "cache": {"expiration_hours": 24},
        "logging": {"buffer_size": 100},
    }
    with open(config_path, "w") as f:
        import yaml

        yaml.dump(config, f)
    return CsvDataProvider(config_path=str(config_path))


def test_csv_provider_connect(csv_provider, tmp_path):
    """Teste la connexion du CsvDataProvider."""
    csv_path = tmp_path / "data" / "iqfeed" / "merged_data.csv"
    os.makedirs(csv_path.parent, exist_ok=True)
    pd.DataFrame({"timestamp": [datetime.now()]}).to_csv(csv_path, index=False)
    csv_provider.csv_paths["ohlc"] = str(csv_path)
    csv_provider.connect()  # Ne doit pas lever d’exception


def test_csv_fetch_ohlc(csv_provider, tmp_path):
    """Teste fetch_ohlc avec un CSV."""
    csv_path = tmp_path / "data" / "iqfeed" / "merged_data.csv"
    os.makedirs(csv_path.parent, exist_ok=True)
    data = pd.DataFrame(
        {
            "timestamp": [datetime.now()],
            "open": [4000.0],
            "high": [4010.0],
            "low": [3990.0],
            "close": [4005.0],
            "volume": [1000],
            "vix_es_correlation": [0.5],
            "atr_14": [20.0],
        }
    )
    data.to_csv(csv_path, index=False)
    csv_provider.csv_paths["ohlc"] = str(csv_path)
    df = csv_provider.fetch_ohlc(symbol="ES", timeframe="raw")
    assert df is not None
    assert len(df) == 1
    assert all(col in df.columns for col in ["timestamp", "open", "vix_es_correlation"])


def test_store_data_pattern(csv_provider):
    """Teste la clusterisation et le stockage des patterns."""
    df = pd.DataFrame(
        {"timestamp": [datetime.now()], "vix_es_correlation": [0.5], "atr_14": [20.0]}
    )
    cluster_id = csv_provider.store_data_pattern(df, data_type="ohlc", symbol="ES")
    assert isinstance(cluster_id, int)
    assert cluster_id >= 0


def test_get_data_provider_csv(tmp_path):
    """Teste la sélection du CsvDataProvider."""
    config_path = tmp_path / "market_config.yaml"
    config = {"data_feed": "csv"}
    with open(config_path, "w") as f:
        import yaml

        yaml.dump(config, f)
    provider = get_data_provider(str(config_path))
    assert isinstance(provider, CsvDataProvider)
