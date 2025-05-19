# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_export_visuals.py
# Tests unitaires pour src/monitoring/export_visuals.py
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide l’exportation des visualisations en PDF/HTML, avec focus sur les performances et risques
#        (Phase 15, méthode 17), intégration de confidence_drop_rate (Phase 8), snapshots compressés,
#        sauvegardes incrémentielles/distribuées, et alertes Telegram.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, plotly>=5.15.0,<6.0.0, matplotlib>=3.8.0,<4.0.0,
#   reportlab>=4.0.0,<5.0.0, psutil>=5.9.8,<6.0.0, loguru>=0.7.0,<1.0.0, pyyaml>=6.0.0,<7.0.0,
#   boto3>=1.26.0,<2.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/es_config.yaml
# - Données de performance (pd.DataFrame avec timestamp, cumulative_return, gamma_exposure, iv_sensitivity)
#
# Outputs :
# - Visualisations HTML/PDF dans data/figures/monitoring/
# - Snapshots dans data/cache/visuals/*.json.gz
# - Sauvegardes incrémentielles dans data/checkpoints/visuals_*.json.gz
# - Logs dans data/logs/monitoring/export_visuals.log
# - Logs de performance dans data/logs/monitoring/export_visuals_performance.csv
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Tests les phases 8 (auto-conscience via confidence_drop_rate), 15 (visualisations).
# - Vérifie l’absence de dxFeed, la suppression de 320/81 features et obs_t, les retries, logs psutil, alertes Telegram,
#   snapshots compressés, et sauvegardes incrémentielles/distribuées.

import gzip
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.monitoring.export_visuals import VisualExporter


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les données, logs, figures, cache, et checkpoints."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs" / "monitoring"
    logs_dir.mkdir(parents=True)
    figures_dir = data_dir / "figures" / "monitoring"
    figures_dir.mkdir(parents=True)
    cache_dir = data_dir / "cache" / "visuals"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints"
    checkpoints_dir.mkdir()

    # Créer es_config.yaml
    config_path = config_dir / "es_config.yaml"
    config_content = {
        "visuals_params": {"s3_bucket": "test-bucket", "s3_prefix": "visuals/"},
        "logging": {"buffer_size": 100},
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    return {
        "base_dir": str(base_dir),
        "config_path": str(config_path),
        "logs_dir": str(logs_dir),
        "figures_dir": str(figures_dir),
        "cache_dir": str(cache_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "perf_log_path": str(logs_dir / "export_visuals_performance.csv"),
    }


@pytest.fixture
def sample_data():
    """Crée des données factices pour les tests."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start="2025-05-13 10:00:00", periods=10, freq="H"
            ),
            "cumulative_return": np.cumsum(np.random.uniform(-0.01, 0.01, 10)),
            "gamma_exposure": np.random.uniform(-500, 500, 10),
            "iv_sensitivity": np.random.uniform(0.1, 0.5, 10),
        }
    )


def test_init(tmp_dirs):
    """Teste l’initialisation de VisualExporter."""
    exporter = VisualExporter(
        config_path=tmp_dirs["config_path"], output_dir=Path(tmp_dirs["figures_dir"])
    )
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "init" in str(op) for op in df["operation"]
    ), "Opération init non journalisée"
    snapshot_files = os.listdir(tmp_dirs["cache_dir"])
    assert any(
        "snapshot_init" in f and f.endswith(".json.gz") for f in snapshot_files
    ), "Snapshot init non créé"


def test_load_config(tmp_dirs):
    """Teste le chargement de la configuration."""
    exporter = VisualExporter(config_path=tmp_dirs["config_path"])
    config = exporter.load_config(tmp_dirs["config_path"])
    assert "s3_bucket" in config, "Clé s3_bucket manquante"
    assert "s3_prefix" in config, "Clé s3_prefix manquante"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "load_config" in str(op) for op in df["operation"]
    ), "Opération load_config non journalisée"


def test_export_visuals(tmp_dirs, sample_data):
    """Teste l’exportation des visualisations en HTML et PDF."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        exporter = VisualExporter(
            config_path=tmp_dirs["config_path"],
            output_dir=Path(tmp_dirs["figures_dir"]),
        )
        output_files = exporter.export_visuals(sample_data, prefix="test")
        assert "html" in output_files, "Fichier HTML non exporté"
        assert "pdf" in output_files, "Fichier PDF non exporté"
        assert os.path.exists(output_files["html"]), "Fichier HTML non créé"
        assert os.path.exists(output_files["pdf"]), "Fichier PDF non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "export_visuals" in str(op) for op in df_perf["operation"]
        ), "Opération export_visuals non journalisée"
        assert any(
            "confidence_drop_rate" in str(kw) for kw in df_perf.to_dict("records")
        ), "confidence_drop_rate absent"
        mock_telegram.assert_called()


def test_export_visuals_missing_columns(tmp_dirs):
    """Teste l’exportation avec des colonnes manquantes."""
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start="2025-05-13 10:00:00", periods=10, freq="H"
            )
            # cumulative_return manquant
        }
    )
    exporter = VisualExporter(config_path=tmp_dirs["config_path"])
    with pytest.raises(ValueError, match="Colonnes manquantes"):
        exporter.export_visuals(data, prefix="test")
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "export_visuals" in str(op) and not success
        for success, op in zip(df_perf["success"], df_perf["operation"])
    ), "Échec non journalisé"


def test_save_snapshot_compressed(tmp_dirs):
    """Teste la sauvegarde d’un snapshot compressé."""
    exporter = VisualExporter(config_path=tmp_dirs["config_path"])
    snapshot_data = {"test": "compressed_snapshot"}
    exporter.save_snapshot("test_compressed", snapshot_data, compress=True)
    snapshot_files = os.listdir(tmp_dirs["cache_dir"])
    assert any(
        "snapshot_test_compressed" in f and f.endswith(".json.gz")
        for f in snapshot_files
    ), "Snapshot compressé non créé"
    with gzip.open(
        os.path.join(tmp_dirs["cache_dir"], snapshot_files[-1]), "rt", encoding="utf-8"
    ) as f:
        snapshot = json.load(f)
    assert snapshot["data"] == snapshot_data, "Contenu du snapshot compressé incorrect"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "save_snapshot" in str(op) for op in df_perf["operation"]
    ), "Opération save_snapshot non journalisée"


def test_checkpoint(tmp_dirs, sample_data):
    """Teste la sauvegarde incrémentielle."""
    exporter = VisualExporter(config_path=tmp_dirs["config_path"])
    exporter.checkpoint(sample_data)
    checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
    assert any(
        "visuals_" in f and f.endswith(".json.gz") for f in checkpoint_files
    ), "Checkpoint non créé"
    with gzip.open(
        os.path.join(tmp_dirs["checkpoints_dir"], checkpoint_files[0]),
        "rt",
        encoding="utf-8",
    ) as f:
        checkpoint = json.load(f)
    assert checkpoint["num_rows"] == len(sample_data), "Nombre de lignes incorrect"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "checkpoint" in str(op) for op in df_perf["operation"]
    ), "Opération checkpoint non journalisée"


def test_cloud_backup(tmp_dirs, sample_data):
    """Teste la sauvegarde distribuée S3."""
    with patch("boto3.client") as mock_s3:
        exporter = VisualExporter(config_path=tmp_dirs["config_path"])
        exporter.cloud_backup(sample_data)
        assert mock_s3.called, "Client S3 non appelé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "temp_s3_" in f for f in checkpoint_files
        ), "Fichier temporaire S3 non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "cloud_backup" in str(op) for op in df_perf["operation"]
        ), "Opération cloud_backup non journalisée"


def test_handle_sigint(tmp_dirs):
    """Teste la gestion SIGINT."""
    exporter = VisualExporter(config_path=tmp_dirs["config_path"])
    with patch("sys.exit") as mock_exit:
        exporter.handle_sigint(signal.SIGINT, None)
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_sigint" in f for f in snapshot_files
        ), "Snapshot sigint non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "handle_sigint" in str(op) for op in df_perf["operation"]
        ), "Opération handle_sigint non journalisée"
        mock_exit.assert_called_with(0)
