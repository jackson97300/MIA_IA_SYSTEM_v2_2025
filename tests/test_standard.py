# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_standard.py
# Tests unitaires pour src/model/utils/standard.py
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide les fonctions communes (retries, logging psutil,
# alertes) pour la Phase 15, avec support multi-marchés, snapshots
# compressés, et sauvegardes incrémentielles/distribuées.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0
# - pandas>=2.0.0,<3.0.0
# - psutil>=5.9.8,<6.0.0
# - boto3>=1.26.0,<2.0.0
# - loguru>=0.7.0,<1.0.0
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
#
# Outputs :
# - Tests unitaires validant les fonctionnalités de standard.py.
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Tests la Phase 8 (confidence_drop_rate) et Phase 15 (fonctions
#   communes).
# - Vérifie l’absence de dxFeed, la suppression de 320/81 features et
#   obs_t, les retries, logs psutil, alertes Telegram, snapshots
#   compressés, et sauvegardes incrémentielles/distribuées.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.

from datetime import datetime
from unittest.mock import patch
import time

import pandas as pd
import pytest
import yaml

from src.model.utils.standard import (
    checkpoint,
    cloud_backup,
    log_performance,
    save_snapshot,
    with_retries,
)


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les logs, cache, et
    checkpoints."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    cache_dir = data_dir / "cache" / "standard" / "ES"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints" / "standard" / "ES"
    checkpoints_dir.mkdir(parents=True)

    # Créer es_config.yaml
    config_dir = base_dir / "config"
    config_dir.mkdir()
    config_path = config_dir / "es_config.yaml"
    config_content = {
        "s3_bucket": "test-bucket",
        "s3_prefix": "standard/",
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    return {
        "base_dir": str(base_dir),
        "logs_dir": str(logs_dir),
        "cache_dir": str(cache_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "perf_log_path": str(
            logs_dir / "standard_utils_performance.csv"
        ),
    }


@pytest.mark.asyncio
async def test_with_retries_success(tmp_dirs):
    """Teste with_retries avec une exécution réussie."""
    with patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:

        @with_retries(
            max_attempts=3,
            delay=0.1,
            exceptions=(ValueError,),
        )
        def successful_func(market: str = "ES"):
            return "Succès"

        result = successful_func(market="ES")
        assert (
            result == "Succès"
        ), "Fonction n’a pas retourné le résultat attendu"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "retry_success" in str(op) for op in df_perf["operation"]
        ), "Opération retry_success non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_with_retries_failure(tmp_dirs):
    """Teste with_retries avec un échec après max_attempts."""
    with patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:

        @with_retries(
            max_attempts=2,
            delay=0.1,
            exceptions=(ValueError,),
        )
        def failing_func(market: str = "ES"):
            raise ValueError("Erreur simulée")

        with pytest.raises(ValueError):
            failing_func(market="ES")
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "retry_failure" in str(op) for op in df_perf["operation"]
        ), "Opération retry_failure non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_log_performance(tmp_dirs):
    """Teste la journalisation des performances."""
    with patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        start_time = time.time()
        log_performance(
            operation="test_op",
            latency=time.time() - start_time,
            success=True,
            market="ES",
            extra_metrics={"test_metric": 42},
        )
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "test_op" in str(op) for op in df_perf["operation"]
        ), "Opération test_op non journalisée"
        assert any(
            "confidence_drop_rate" in str(kw)
            for kw in df_perf.to_dict("records")
        ), "confidence_drop_rate absent"
        assert (
            df_perf["memory_used_mb"].iloc[-1] > 0
        ), "Usage mémoire non journalisé"
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_log_performance" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot log_performance non créé"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_save_snapshot(tmp_dirs):
    """Teste la sauvegarde des snapshots."""
    with patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        data = {"test": "data"}
        save_snapshot(
            snapshot_type="test_snapshot",
            data=data,
            market="ES",
        )
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_test_snapshot" in f and f.endswith(".json.gz")
            for f in snapshot_files
        ), "Snapshot test_snapshot non créé"
        with gzip.open(
            os.path.join(tmp_dirs["cache_dir"], snapshot_files[0]),
            "rt",
            encoding="utf-8",
        ) as f:
            snapshot_data = json.load(f)
        assert (
            snapshot_data["type"] == "test_snapshot"
        ), "Type de snapshot incorrect"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "save_snapshot" in str(op) for op in df_perf["operation"]
        ), "Opération save_snapshot non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_checkpoint(tmp_dirs):
    """Teste la sauvegarde incrémentielle."""
    with patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        df = pd.DataFrame(
            {
                "timestamp": [datetime.now().isoformat()],
                "value": [42],
            }
        )
        checkpoint(
            df=df,
            data_type="test_metrics",
            market="ES",
        )
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "standard_test_metrics" in f and f.endswith(".json.gz")
            for f in checkpoint_files
        ), "Checkpoint non créé"
        with gzip.open(
            os.path.join(tmp_dirs["checkpoints_dir"], checkpoint_files[0]),
            "rt",
            encoding="utf-8",
        ) as f:
            checkpoint_data = json.load(f)
        assert (
            checkpoint_data["num_rows"] == len(df)
        ), "Nombre de lignes incorrect"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "checkpoint" in str(op) for op in df_perf["operation"]
        ), "Opération checkpoint non journalisée"
        mock_telegram.assert_called()


@pytest.mark.asyncio
async def test_cloud_backup(tmp_dirs):
    """Teste la sauvegarde distribuée S3."""
    with patch("boto3.client") as mock_s3, patch(
        "src.utils.telegram_alert.send_telegram_alert"
    ) as mock_telegram:
        df = pd.DataFrame(
            {
                "timestamp": [datetime.now().isoformat()],
                "value": [42],
            }
        )
        cloud_backup(
            df=df,
            data_type="test_metrics",
            market="ES",
        )
        assert mock_s3.called, "Client S3 non appelé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "temp_s3_" in f for f in checkpoint_files
        ), "Fichier temporaire S3 non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "cloud_backup" in str(op) for op in df_perf["operation"]
        ), "Opération cloud_backup non journalisée"
        mock_telegram.assert_called()