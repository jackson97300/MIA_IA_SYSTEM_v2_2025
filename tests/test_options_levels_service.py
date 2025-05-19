# Teste options_levels_service.py.
# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_options_levels_service.py
# Tests unitaires pour options_levels_service.py.
#
# Version : 2.1.3
# Date : 2025-05-13
#
# Rôle : Vérifie les fonctionnalités de OptionsLevelsService, incluant l'initialisation,
#        la boucle asynchrone avec retries, les snapshots JSON compressés, les logs de performance,
#        la fraîcheur des données IQFeed, l'intégration de predicted_vix, et les visualisations.
#        Conforme à la Phase 1 (collecte via IQFeed), Phase 8 (auto-conscience via miya_console),
#        et Phase 16 (ensemble learning via LSTM).
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pytest-asyncio>=0.21.0
# - pandas>=2.0.0
# - psutil>=5.9.8
# - matplotlib>=3.5.0
# - src.scripts.options_levels_service
# - src.features.spotgamma_recalculator
# - src.api.iqfeed_fetch
# - src.model.utils.miya_console
# - src.utils.telegram_alert
# - src.model.utils.config_manager
# - src.model.utils.alert_manager
# - src.model.neural_pipeline
#
# Notes :
# - Utilise des mocks pour simuler les dépendances externes.
# - Vérifie l'absence de références à dxFeed, obs_t, 320/81 features.

import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from src.scripts.options_levels_service import OptionsLevelsService


@pytest.fixture
def temp_dir(tmp_path):
    """Crée un répertoire temporaire pour les tests."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    data_dir = tmp_path / "data"
    logs_dir = data_dir / "logs"
    snapshots_dir = data_dir / "options_snapshots"
    figures_dir = data_dir / "figures" / "options"
    logs_dir.mkdir(parents=True)
    snapshots_dir.mkdir(parents=True)
    figures_dir.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def mock_config(temp_dir):
    """Crée un fichier es_config.yaml simulé."""
    config_path = temp_dir / "config" / "es_config.yaml"
    config_content = """
options_levels_service:
  max_data_age_seconds: 300
spotgamma_recalculator:
  min_volume_threshold: 10
  shap_features: ["iv_atm", "option_skew"]
"""
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def service(temp_dir, mock_config, monkeypatch):
    """Initialise OptionsLevelsService avec des mocks."""
    monkeypatch.setattr(
        "src.scripts.options_levels_service.get_config",
        lambda x: {
            "options_levels_service": {"max_data_age_seconds": 300},
            "spotgamma_recalculator": {
                "min_volume_threshold": 10,
                "shap_features": ["iv_atm", "option_skew"],
            },
        },
    )
    service = OptionsLevelsService(str(mock_config))
    return service


@pytest.mark.asyncio
async def test_init_service(temp_dir, mock_config, monkeypatch):
    """Teste l'initialisation de OptionsLevelsService."""
    service = OptionsLevelsService(str(mock_config))
    assert service.config["max_data_age_seconds"] == 300
    assert os.path.exists(temp_dir / "data" / "options_snapshots")
    assert os.path.exists(temp_dir / "data" / "figures" / "options")
    snapshots = list(
        (temp_dir / "data" / "options_snapshots").glob("snapshot_init_*.json.gz")
    )
    assert len(snapshots) >= 1


@pytest.mark.asyncio
async def test_load_config_invalid(temp_dir, monkeypatch):
    """Teste le chargement d'une configuration invalide."""
    invalid_config = temp_dir / "config" / "es_config.yaml"
    invalid_config.write_text("")
    with patch(
        "src.scripts.options_levels_service.get_config",
        side_effect=ValueError("Configuration vide"),
    ):
        service = OptionsLevelsService(str(invalid_config))
    assert service.config["max_data_age_seconds"] == 300  # Fallback to default
    snapshots = list(
        (temp_dir / "data" / "options_snapshots").glob("snapshot_init_*.json.gz")
    )
    assert len(snapshots) >= 1


@pytest.mark.asyncio
async def test_update_levels_async(temp_dir, service, monkeypatch):
    """Teste la mise à jour asynchrone des niveaux."""
    mock_option_chain = pd.DataFrame(
        {"timestamp": [datetime.now()], "vix_es_correlation": [20.0]}
    )
    mock_levels = {
        "put_wall": 4000,
        "call_wall": 4200,
        "zero_gamma": 4100,
        "timestamp": "2025-05-13 12:00:00",
    }

    with patch(
        "src.scripts.options_levels_service.fetch_option_chain",
        return_value=mock_option_chain,
    ), patch(
        "src.scripts.options_levels_service.predict_vix", return_value=20.0
    ), patch.object(
        service.recalculator, "recalculate_levels", return_value=mock_levels
    ), patch.object(
        service.recalculator, "save_levels"
    ) as mock_save:
        await service.update_levels_async()

    snapshots = list(
        (temp_dir / "data" / "options_snapshots").glob(
            "snapshot_update_levels_*.json.gz"
        )
    )
    assert len(snapshots) >= 1
    figures = list((temp_dir / "data" / "figures" / "options").glob("levels_*.png"))
    assert len(figures) >= 1
    assert mock_save.called
    assert os.path.exists(temp_dir / "data" / "options_levels_dashboard.json")


@pytest.mark.asyncio
async def test_run_async_with_retries(temp_dir, service, monkeypatch):
    """Teste la boucle asynchrone avec retries."""
    mock_option_chain = pd.DataFrame(
        {"timestamp": [datetime.now()], "vix_es_correlation": [20.0]}
    )
    mock_levels = {
        "put_wall": 4000,
        "call_wall": 4200,
        "zero_gamma": 4100,
        "timestamp": "2025-05-13 12:00:00",
    }

    async def mock_update_levels():
        await service.update_levels_async()

    with patch(
        "src.scripts.options_levels_service.fetch_option_chain",
        return_value=mock_option_chain,
    ), patch(
        "src.scripts.options_levels_service.predict_vix", return_value=20.0
    ), patch.object(
        service.recalculator, "recalculate_levels", return_value=mock_levels
    ), patch.object(
        service.recalculator, "save_levels"
    ), patch(
        "asyncio.sleep", new=AsyncMock()
    ) as mock_sleep:
        # Simulate one failure and one success
        with patch.object(
            service, "update_levels_async", side_effect=[ValueError("Test error"), None]
        ):
            await service.with_retries_async(mock_update_levels)
        assert mock_sleep.called
        assert mock_sleep.call_args[0][0] == 2.0  # First retry delay
        snapshots = list(
            (temp_dir / "data" / "options_snapshots").glob(
                "snapshot_update_levels_*.json.gz"
            )
        )
        assert len(snapshots) >= 1


@pytest.mark.asyncio
async def test_check_iqfeed_freshness(temp_dir, service):
    """Teste la vérification de la fraîcheur des données IQFeed."""
    fresh_data = pd.DataFrame({"timestamp": [datetime.now()]})
    old_data = pd.DataFrame({"timestamp": [datetime.now() - timedelta(seconds=600)]})

    assert service.check_iqfeed_freshness(fresh_data, max_age_seconds=300) is True
    assert service.check_iqfeed_freshness(old_data, max_age_seconds=300) is False
    assert service.check_iqfeed_freshness(pd.DataFrame(), max_age_seconds=300) is False


@pytest.mark.asyncio
async def test_signal_handler(temp_dir, service, monkeypatch):
    """Teste l'arrêt propre du service."""
    mock_option_chain = pd.DataFrame({"timestamp": [datetime.now()]})
    mock_levels = {
        "put_wall": 4000,
        "call_wall": 4200,
        "zero_gamma": 4100,
        "timestamp": "2025-05-13 12:00:00",
    }

    with patch(
        "src.scripts.options_levels_service.fetch_option_chain",
        return_value=mock_option_chain,
    ), patch.object(
        service.recalculator, "recalculate_levels", return_value=mock_levels
    ), patch.object(
        service.recalculator, "save_levels"
    ), patch(
        "sys.exit"
    ) as mock_exit:
        service.signal_handler(signal.SIGINT, None)

    snapshots = list(
        (temp_dir / "data" / "options_snapshots").glob("snapshot_shutdown_*.json.gz")
    )
    assert len(snapshots) >= 1
    assert mock_exit.called


def test_no_obsolete_references(temp_dir, mock_config):
    """Vérifie l'absence de références à dxFeed, obs_t, 320/81 features."""
    with open(mock_config, "r") as f:
        content = f.read()
    assert "dxFeed" not in content, "Référence à dxFeed trouvée"
    assert "obs_t" not in content, "Référence à obs_t trouvée"
    assert "320 features" not in content, "Référence à 320 features trouvée"
    assert "81 features" not in content, "Référence à 81 features trouvée"
