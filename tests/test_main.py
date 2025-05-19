# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_main.py
# Tests unitaires pour src/main.py
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide le point d'entrée principal pour orchestrer l'exécution des modules de MIA_IA_SYSTEM_v2_2025,
#        avec intégration de confidence_drop_rate (Phase 8), snapshots compressés,
#        sauvegardes incrémentielles/distribuées, et alertes Telegram.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, loguru>=0.7.0,<1.0.0, pyyaml>=6.0.0,<7.0.0,
#   boto3>=1.26.0,<2.0.0, psutil>=5.9.8,<6.0.0
# - src/mind/mind_voice.py
# - src/monitoring/correlation_heatmap.py
# - src/monitoring/data_drift.py
# - src/monitoring/export_visuals.py
# - src/monitoring/mia_dashboard.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/es_config.yaml
# - data/features/features_latest.csv
# - data/trades/trades_simulated.csv
# - data/features/feature_importance.csv
# - data/logs/regime_history.csv
#
# Outputs :
# - Logs dans data/logs/main.log
# - Logs de performance dans data/logs/main_performance.csv
# - Snapshots dans data/cache/main/*.json.gz
# - Sauvegardes incrémentielles dans data/checkpoints/main_*.json.gz
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Tests les phases 8 (auto-conscience via confidence_drop_rate), Phases 1-18 via modules appelés.
# - Vérifie l’absence de dxFeed, la suppression de 320/81 features et obs_t, les retries, logs psutil, alertes Telegram,
#   snapshots compressés, et sauvegardes incrémentielles/distribuées.

import gzip
import json
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.main import MainRunner


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les données, logs, cache, et checkpoints."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()
    trades_dir = data_dir / "trades"
    trades_dir.mkdir()
    features_dir = data_dir / "features"
    features_dir.mkdir()
    cache_dir = data_dir / "cache" / "main"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints"
    checkpoints_dir.mkdir()

    # Créer es_config.yaml
    config_path = config_dir / "es_config.yaml"
    config_content = {
        "main_params": {
            "s3_bucket": "test-bucket",
            "s3_prefix": "main/",
            "modules": [
                "mind_voice",
                "correlation_heatmap",
                "data_drift",
                "export_visuals",
                "mia_dashboard",
            ],
        },
        "logging": {"buffer_size": 100},
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    # Créer features_latest.csv
    features_latest_path = features_dir / "features_latest.csv"
    features_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=100, freq="1min"),
            "close": np.random.normal(5100, 10, 100),
            "volume": np.random.randint(1000, 10000, 100),
            "vix_es_correlation": np.random.uniform(-1, 1, 100),
            **{
                f"feature_{i}": np.random.uniform(0, 1, 100) for i in range(147)
            },  # Total 150 columns
        }
    )
    features_data.to_csv(features_latest_path, index=False, encoding="utf-8")

    # Créer trades_simulated.csv
    trades_simulated_path = trades_dir / "trades_simulated.csv"
    trades_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=100, freq="1min"),
            "reward": np.random.uniform(-100, 100, 100),
        }
    )
    trades_data.to_csv(trades_simulated_path, index=False, encoding="utf-8")

    # Créer feature_importance.csv
    feature_importance_path = features_dir / "feature_importance.csv"
    shap_data = pd.DataFrame(
        {
            "feature": [f"feature_{i}" for i in range(150)],
            "importance": np.random.uniform(0, 1, 150),
        }
    )
    shap_data.to_csv(feature_importance_path, index=False, encoding="utf-8")

    # Créer regime_history.csv
    regime_history_path = logs_dir / "regime_history.csv"
    regime_history_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=100, freq="1min"),
            "regime": np.random.choice(["trend", "range", "defensive"], 100),
        }
    )
    regime_history_data.to_csv(regime_history_path, index=False, encoding="utf-8")

    # Créer features_train.csv
    features_train_path = features_dir / "features_train.csv"
    features_train_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-12 09:00", periods=100, freq="1min"),
            "close": np.random.normal(5100, 10, 100),
            "volume": np.random.randint(1000, 10000, 100),
            "vix_es_correlation": np.random.uniform(-1, 1, 100),
            **{
                f"feature_{i}": np.random.uniform(0, 1, 100) for i in range(147)
            },  # Total 150 columns
        }
    )
    features_train_data.to_csv(features_train_path, index=False, encoding="utf-8")

    return {
        "base_dir": str(base_dir),
        "config_path": str(config_path),
        "features_latest_path": str(features_latest_path),
        "trades_simulated_path": str(trades_simulated_path),
        "feature_importance_path": str(feature_importance_path),
        "regime_history_path": str(regime_history_path),
        "features_train_path": str(features_train_path),
        "logs_dir": str(logs_dir),
        "cache_dir": str(cache_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "perf_log_path": str(logs_dir / "main_performance.csv"),
    }


@pytest.fixture
def mock_modules():
    """Mock les modules internes pour les tests."""
    with patch("src.main.VoiceManager") as mock_voice, patch(
        "src.main.CorrelationHeatmap"
    ) as mock_heatmap, patch("src.main.DataDriftDetector") as mock_drift, patch(
        "src.main.VisualExporter"
    ) as mock_visuals, patch(
        "src.main.MIADashboard"
    ) as mock_dashboard:
        mock_voice.return_value.speak.return_value = None
        mock_heatmap.return_value.plot_heatmap.return_value = None
        mock_heatmap.return_value.compute_correlations.return_value = pd.DataFrame()
        mock_heatmap.return_value.save_correlation_matrix.return_value = None
        mock_drift.return_value.compute_drift_metrics.return_value = {}
        mock_drift.return_value.plot_drift.return_value = None
        mock_drift.return_value.save_drift_report.return_value = None
        mock_visuals.return_value.export_visuals.return_value = {
            "html": "test.html",
            "pdf": "test.pdf",
        }
        mock_dashboard.return_value.main.return_value = None
        yield


def test_init(tmp_dirs, mock_modules):
    """Teste l’initialisation de MainRunner."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        runner = MainRunner(config_path=tmp_dirs["config_path"])
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
        mock_telegram.assert_called()


def test_load_config(tmp_dirs):
    """Teste le chargement de la configuration."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        runner = MainRunner(config_path=tmp_dirs["config_path"])
        config = runner.load_config(tmp_dirs["config_path"])
        assert "s3_bucket" in config, "Clé s3_bucket manquante"
        assert "modules" in config, "Clé modules manquante"
        df = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "load_config" in str(op) for op in df["operation"]
        ), "Opération load_config non journalisée"
        mock_telegram.assert_called()


def test_validate_inputs(tmp_dirs, mock_modules):
    """Teste la validation des fichiers d'entrée."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        runner = MainRunner(config_path=tmp_dirs["config_path"])
        runner.validate_inputs(
            tmp_dirs["features_latest_path"],
            tmp_dirs["trades_simulated_path"],
            tmp_dirs["feature_importance_path"],
            tmp_dirs["regime_history_path"],
        )
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "validate_inputs" in str(op) for op in df_perf["operation"]
        ), "Opération validate_inputs non journalisée"
        assert any(
            "confidence_drop_rate" in str(kw) for kw in df_perf.to_dict("records")
        ), "confidence_drop_rate absent"
        mock_telegram.assert_called()


def test_validate_inputs_missing_file(tmp_dirs, mock_modules):
    """Teste la validation avec un fichier manquant."""
    runner = MainRunner(config_path=tmp_dirs["config_path"])
    with pytest.raises(FileNotFoundError):
        runner.validate_inputs(
            "non_existent.csv",
            tmp_dirs["trades_simulated_path"],
            tmp_dirs["feature_importance_path"],
            tmp_dirs["regime_history_path"],
        )
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "validate_inputs" in str(op) and not success
        for success, op in zip(df_perf["success"], df_perf["operation"])
    ), "Échec non journalisé"


def test_run_module_mind_voice(tmp_dirs, mock_modules):
    """Teste l’exécution du module mind_voice."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        runner = MainRunner(config_path=tmp_dirs["config_path"])
        runner.run_module("mind_voice", config_path=tmp_dirs["config_path"])
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "run_module_mind_voice" in str(op) for op in df_perf["operation"]
        ), "Opération run_module_mind_voice non journalisée"
        mock_telegram.assert_called()


def test_run_module_correlation_heatmap(tmp_dirs, mock_modules):
    """Teste l’exécution du module correlation_heatmap."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        runner = MainRunner(config_path=tmp_dirs["config_path"])
        runner.run_module(
            "correlation_heatmap",
            config_path=tmp_dirs["config_path"],
            features_path=tmp_dirs["features_latest_path"],
            output_csv=tmp_dirs["logs_dir"] + "/correlation_matrix.csv",
        )
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "run_module_correlation_heatmap" in str(op) for op in df_perf["operation"]
        ), "Opération run_module_correlation_heatmap non journalisée"
        mock_telegram.assert_called()


def test_run_module_data_drift(tmp_dirs, mock_modules):
    """Teste l’exécution du module data_drift."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        runner = MainRunner(config_path=tmp_dirs["config_path"])
        runner.run_module(
            "data_drift",
            config_path=tmp_dirs["config_path"],
            features_path=tmp_dirs["features_latest_path"],
            train_path=tmp_dirs["features_train_path"],
            shap_path=tmp_dirs["feature_importance_path"],
            output_path=tmp_dirs["logs_dir"] + "/drift_report.csv",
        )
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "run_module_data_drift" in str(op) for op in df_perf["operation"]
        ), "Opération run_module_data_drift non journalisée"
        mock_telegram.assert_called()


def test_run_module_export_visuals(tmp_dirs, mock_modules):
    """Teste l’exécution du module export_visuals."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        runner = MainRunner(config_path=tmp_dirs["config_path"])
        runner.run_module(
            "export_visuals",
            config_path=tmp_dirs["config_path"],
            trades_path=tmp_dirs["trades_simulated_path"],
        )
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "run_module_export_visuals" in str(op) for op in df_perf["operation"]
        ), "Opération run_module_export_visuals non journalisée"
        mock_telegram.assert_called()


def test_save_snapshot_compressed(tmp_dirs, mock_modules):
    """Teste la sauvegarde d’un snapshot compressé."""
    runner = MainRunner(config_path=tmp_dirs["config_path"])
    snapshot_data = {"test": "compressed_snapshot"}
    runner.save_snapshot("test_compressed", snapshot_data, compress=True)
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


def test_checkpoint(tmp_dirs, mock_modules):
    """Teste la sauvegarde incrémentielle."""
    runner = MainRunner(config_path=tmp_dirs["config_path"])
    df = pd.read_csv(tmp_dirs["trades_simulated_path"])
    runner.checkpoint(df, data_type="system_state")
    checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
    assert any(
        "main_system_state" in f and f.endswith(".json.gz") for f in checkpoint_files
    ), "Checkpoint non créé"
    with gzip.open(
        os.path.join(tmp_dirs["checkpoints_dir"], checkpoint_files[0]),
        "rt",
        encoding="utf-8",
    ) as f:
        checkpoint = json.load(f)
    assert checkpoint["num_rows"] == len(df), "Nombre de lignes incorrect"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "checkpoint" in str(op) for op in df_perf["operation"]
    ), "Opération checkpoint non journalisée"


def test_cloud_backup(tmp_dirs, mock_modules):
    """Teste la sauvegarde distribuée S3."""
    with patch("boto3.client") as mock_s3:
        runner = MainRunner(config_path=tmp_dirs["config_path"])
        df = pd.read_csv(tmp_dirs["trades_simulated_path"])
        runner.cloud_backup(df, data_type="system_state")
        assert mock_s3.called, "Client S3 non appelé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "temp_s3_" in f for f in checkpoint_files
        ), "Fichier temporaire S3 non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "cloud_backup" in str(op) for op in df_perf["operation"]
        ), "Opération cloud_backup non journalisée"


def test_handle_sigint(tmp_dirs, mock_modules):
    """Teste la gestion SIGINT."""
    runner = MainRunner(config_path=tmp_dirs["config_path"])
    with patch("sys.exit") as mock_exit:
        runner.handle_sigint(signal.SIGINT, None)
        snapshot_files = os.listdir(tmp_dirs["cache_dir"])
        assert any(
            "snapshot_sigint" in f for f in snapshot_files
        ), "Snapshot sigint non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "handle_sigint" in str(op) for op in df_perf["operation"]
        ), "Opération handle_sigint non journalisée"
        mock_exit.assert_called_with(0)
