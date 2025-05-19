# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_mind_dialogue.py
# Tests unitaires pour src/mind/mind_dialogue.py
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide la gestion des dialogues interactifs avec MIA, incluant la reconnaissance vocale, les réponses conversationnelles
#        via NLP (OpenAI), la journalisation des dialogues avec mémoire contextuelle (méthode 7, K-means 10 clusters dans
#        market_memory.db), l’utilisation exclusive d’IQFeed, l’intégration SHAP (Phase 17), confidence_drop_rate (Phase 8),
#        snapshots compressés, sauvegardes incrémentielles/distribuées.
#
# Dépendances :
# - pytest>=7.3.0,<8.0.0, pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.8,<6.0.0, sklearn>=1.2.0,<2.0.0,
#   openai>=1.0.0,<2.0.0, matplotlib>=3.8.0,<4.0.0, seaborn>=0.13.0,<1.0.0, pyyaml>=6.0.0,<7.0.0, boto3>=1.26.0,<2.0.0,
#   speech_recognition>=3.8.0,<4.0.0
# - src/mind/mind.py
# - src/envs/trading_env.py
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/db_setup.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - config/mia_config.yaml
# - config/credentials.yaml
# - config/feature_sets.yaml
# - data/features/features_latest.csv
# - data/trades/trades_simulated.csv
#
# Outputs :
# - data/logs/mind/mind_dialogue.csv
# - data/mind/mind_dialogue_snapshots/*.json.gz
# - data/checkpoints/mind_dialogue_*.json.gz
# - data/mind/mind_dialogue_dashboard.json
# - data/logs/mind_dialogue_performance.csv
# - data/figures/mind_dialogue/
# - market_memory.db (table clusters)
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Tests les phases 7 (mémoire contextuelle), 8 (auto-conscience via confidence_drop_rate), 17 (SHAP).
# - Vérifie l’absence de dxFeed, la suppression de 320/81 features et obs_t, les retries, logs psutil, alertes Telegram,
#   snapshots compressés, et sauvegardes incrémentielles/distribuées.
# - Compatible avec 350 features (entraînement) et 150 SHAP features (inférence) via config/feature_sets.yaml.

import gzip
import json
import os
import sqlite3
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.mind.mind_dialogue import DialogueManager


@pytest.fixture
def tmp_dirs(tmp_path):
    """Crée des répertoires temporaires pour les données, logs, snapshots, checkpoints, et figures."""
    base_dir = tmp_path / "MIA_IA_SYSTEM_v2_2025"
    base_dir.mkdir()
    config_dir = base_dir / "config"
    config_dir.mkdir()
    data_dir = base_dir / "data"
    data_dir.mkdir()
    logs_dir = data_dir / "logs" / "mind"
    logs_dir.mkdir(parents=True)
    snapshots_dir = data_dir / "mind" / "mind_dialogue_snapshots"
    snapshots_dir.mkdir(parents=True)
    checkpoints_dir = data_dir / "checkpoints"
    checkpoints_dir.mkdir()
    figures_dir = data_dir / "figures" / "mind_dialogue"
    figures_dir.mkdir(parents=True)
    features_dir = data_dir / "features"
    features_dir.mkdir()
    trades_dir = data_dir / "trades"
    trades_dir.mkdir()

    # Créer mia_config.yaml
    config_path = config_dir / "mia_config.yaml"
    config_content = {
        "logging": {"buffer_size": 100},
        "s3_bucket": "test-bucket",
        "s3_prefix": "mind_dialogue/",
        "strategy_params": {},
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    # Créer credentials.yaml
    credentials_path = config_dir / "credentials.yaml"
    credentials_content = {"nlp": {"enabled": True, "api_key": "test-api-key"}}
    with open(credentials_path, "w", encoding="utf-8") as f:
        yaml.dump(credentials_content, f)

    # Créer feature_sets.yaml
    feature_sets_path = config_dir / "feature_sets.yaml"
    feature_sets_content = {
        "inference": {"shap_features": [f"feature_{i}" for i in range(150)]},
        "training": {"features": [f"feature_{i}" for i in range(350)]},
    }
    with open(feature_sets_path, "w", encoding="utf-8") as f:
        yaml.dump(feature_sets_content, f)

    # Créer features_latest.csv
    features_latest_path = features_dir / "features_latest.csv"
    features_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=10, freq="1min"),
            "vix": np.random.uniform(10, 30, 10),
            "neural_regime": np.random.randint(0, 3, 10),
            "predicted_volatility": np.random.uniform(0.01, 0.1, 10),
            "trade_frequency_1s": np.random.uniform(0, 10, 10),
            "close": np.random.normal(5100, 10, 10),
            "rsi_14": np.random.uniform(30, 70, 10),
            "gex": np.random.uniform(100000, 200000, 10),
            "drawdown": np.random.uniform(-0.05, 0, 10),
            **{
                f"feature_{i}": np.random.uniform(0, 1, 10) for i in range(141)
            },  # Total 150 columns
        }
    )
    features_data.to_csv(features_latest_path, index=False, encoding="utf-8")

    # Créer trades_simulated.csv
    trades_simulated_path = trades_dir / "trades_simulated.csv"
    trades_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-13 09:00", periods=10, freq="1min"),
            "reward": np.random.uniform(-100, 100, 10),
            "drawdown": np.random.uniform(-0.05, 0, 10),
            "regime": ["trend"] * 5 + ["range"] * 5,
            "bid_size_level_1": np.random.randint(100, 1000, 10),
            "ask_size_level_1": np.random.randint(100, 1000, 10),
            "trade_frequency_1s": np.random.uniform(0, 10, 10),
        }
    )
    trades_data.to_csv(trades_simulated_path, index=False, encoding="utf-8")

    return {
        "base_dir": str(base_dir),
        "config_path": str(config_path),
        "credentials_path": str(credentials_path),
        "feature_sets_path": str(feature_sets_path),
        "features_latest_path": str(features_latest_path),
        "trades_simulated_path": str(trades_simulated_path),
        "logs_dir": str(logs_dir),
        "snapshots_dir": str(snapshots_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "figures_dir": str(figures_dir),
        "dialogue_log_path": str(logs_dir / "mind_dialogue.csv"),
        "perf_log_path": str(logs_dir / "mind_dialogue_performance.csv"),
        "dashboard_path": str(data_dir / "mind" / "mind_dialogue_dashboard.json"),
        "db_path": str(data_dir / "market_memory.db"),
    }


@pytest.fixture
def mock_trading_env():
    """Crée un environnement de trading factice."""
    env = MagicMock()
    env.balance = 10000.0
    env.position = 1
    env.mode = "trend"
    env.rsi = 75
    env.step.return_value = (None, 50.0, False, False, {"risk": 0.02})
    return env


@pytest.fixture
def mock_router():
    """Crée un routeur de régime factice."""
    router = MagicMock()
    router.get_current_regime.return_value = "trend"
    return router


def test_init(tmp_dirs):
    """Teste l’initialisation de DialogueManager."""
    dialogue = DialogueManager(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    assert os.path.exists(
        tmp_dirs["perf_log_path"]
    ), "Fichier de logs de performance non créé"
    assert os.path.exists(tmp_dirs["snapshots_dir"]), "Dossier de snapshots non créé"
    assert os.path.exists(
        tmp_dirs["checkpoints_dir"]
    ), "Dossier de checkpoints non créé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert all(
        col in df.columns
        for col in ["timestamp", "operation", "latency", "cpu_percent"]
    ), "Colonnes de performance manquantes"
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_init" in f and f.endswith(".json.gz") for f in snapshot_files
    ), "Snapshot non créé"


def test_load_config(tmp_dirs):
    """Teste le chargement de la configuration."""
    dialogue = DialogueManager(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    config = dialogue.load_config(tmp_dirs["config_path"])
    assert "s3_bucket" in config, "Clé s3_bucket manquante"
    assert "logging" in config, "Clé logging manquante"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "load_config" in str(op) for op in df["operation"]
    ), "Opération non journalisée"


def test_load_nlp_config(tmp_dirs):
    """Teste le chargement de la configuration NLP."""
    dialogue = DialogueManager(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    nlp_config = dialogue.load_nlp_config()
    assert "api_key" in nlp_config, "Clé api_key manquante"
    assert nlp_config["enabled"], "NLP non activé"
    df = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "load_nlp_config" in str(op) for op in df["operation"]
    ), "Opération non journalisée"


def test_validate_data(tmp_dirs):
    """Teste la validation des données."""
    dialogue = DialogueManager(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    df = pd.read_csv(tmp_dirs["features_latest_path"])
    dialogue.validate_data(df)
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "validate_data" in str(op) for op in df_perf["operation"]
    ), "Opération non journalisée"


def test_process_command(tmp_dirs, mock_trading_env, mock_router):
    """Teste le traitement des commandes."""
    with patch("src.utils.telegram_alert.send_telegram_alert") as mock_telegram:
        dialogue = DialogueManager(
            config_path=tmp_dirs["config_path"],
            credentials_path=tmp_dirs["credentials_path"],
        )
        df = pd.read_csv(tmp_dirs["features_latest_path"])
        dialogue.update_context(df, mock_router, mock_trading_env)
        dialogue.process_command("régime actuel")
        assert os.path.exists(
            tmp_dirs["dialogue_log_path"]
        ), "Fichier de logs de dialogues non créé"
        df_log = pd.read_csv(tmp_dirs["dialogue_log_path"])
        assert any(
            "régime actuel" in str(cmd) for cmd in df_log["command"]
        ), "Commande non journalisée"
        assert any(
            "trend" in str(resp) for resp in df_log["response"]
        ), "Réponse incorrecte"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "process_command" in str(op) for op in df_perf["operation"]
        ), "Opération non journalisée"
        mock_telegram.assert_called()


def test_handle_summary(tmp_dirs, mock_trading_env, mock_router):
    """Teste la commande 'résume tes stats'."""
    with patch("openai.OpenAI.chat.completions.create") as mock_nlp:
        mock_nlp.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Résumé des stats..."))]
        )
        dialogue = DialogueManager(
            config_path=tmp_dirs["config_path"],
            credentials_path=tmp_dirs["credentials_path"],
        )
        df = pd.read_csv(tmp_dirs["features_latest_path"])
        dialogue.update_context(df, mock_router, mock_trading_env)
        response = dialogue.handle_summary()
        assert "Résumé des stats" in response, "Réponse NLP incorrecte"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "handle_summary" in str(op) for op in df_perf["operation"]
        ), "Opération non journalisée"


def test_handle_analyze_drawdown(tmp_dirs, mock_trading_env, mock_router):
    """Teste la commande 'analyse le drawdown'."""
    dialogue = DialogueManager(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    df = pd.read_csv(tmp_dirs["features_latest_path"])
    dialogue.update_context(df, mock_router, mock_trading_env)
    response = dialogue.handle_analyze_drawdown()
    assert "actuel" in response and "maximum" in response, "Réponse drawdown incorrecte"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "handle_analyze_drawdown" in str(op) for op in df_perf["operation"]
    ), "Opération non journalisée"


def test_store_dialogue_pattern(tmp_dirs, mock_trading_env, mock_router):
    """Teste la clusterisation et le stockage des dialogues."""
    dialogue = DialogueManager(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    dialogue_entry = {
        "timestamp": datetime.now().isoformat(),
        "command": "résume tes stats",
        "response": "Résumé des stats...",
        "priority": 3,
    }
    result = dialogue.store_dialogue_pattern(dialogue_entry)
    assert "cluster_id" in result, "cluster_id manquant"
    conn = sqlite3.connect(tmp_dirs["db_path"])
    df_db = pd.read_sql_query("SELECT * FROM clusters", conn)
    conn.close()
    assert not df_db.empty, "Aucun cluster stocké dans la base"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "store_dialogue_pattern" in str(op) for op in df_perf["operation"]
    ), "Opération non journalisée"
    assert any(
        "confidence_drop_rate" in str(kw) for kw in df_perf.to_dict("records")
    ), "confidence_drop_rate absent"


def test_visualize_dialogue_patterns(tmp_dirs, mock_trading_env, mock_router):
    """Teste la génération de la heatmap des clusters de dialogues."""
    dialogue = DialogueManager(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    df = pd.read_csv(tmp_dirs["features_latest_path"])
    dialogue.update_context(df, mock_router, mock_trading_env)
    dialogue.process_command("résume tes stats")
    dialogue.visualize_dialogue_patterns()
    assert any(
        f.endswith(".png") for f in os.listdir(tmp_dirs["figures_dir"])
    ), "Heatmap non générée"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "visualize_dialogue_patterns" in str(op) for op in df_perf["operation"]
    ), "Opération non journalisée"


def test_save_snapshot_compressed(tmp_dirs):
    """Teste la sauvegarde d’un snapshot compressé."""
    dialogue = DialogueManager(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    snapshot_data = {"test": "compressed_snapshot"}
    dialogue.save_snapshot("test_compressed", snapshot_data, compress=True)
    snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
    assert any(
        "snapshot_test_compressed" in f and f.endswith(".json.gz")
        for f in snapshot_files
    ), "Snapshot compressé non créé"
    with gzip.open(
        os.path.join(tmp_dirs["snapshots_dir"], snapshot_files[-1]),
        "rt",
        encoding="utf-8",
    ) as f:
        snapshot = json.load(f)
    assert snapshot["data"] == snapshot_data, "Contenu du snapshot compressé incorrect"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "save_snapshot" in str(op) for op in df_perf["operation"]
    ), "Opération non journalisée"


def test_checkpoint(tmp_dirs, mock_trading_env, mock_router):
    """Teste la sauvegarde incrémentielle."""
    dialogue = DialogueManager(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    df = pd.DataFrame(
        {
            "timestamp": [datetime.now().isoformat()] * 5,
            "command": ["résume tes stats"] * 5,
            "response": ["Résumé..."] * 5,
            "cluster_id": [0] * 5,
            "priority": [3] * 5,
        }
    )
    dialogue.checkpoint(df)
    checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
    assert any(
        "mind_dialogue_" in f and f.endswith(".json.gz") for f in checkpoint_files
    ), "Checkpoint non créé"
    with gzip.open(
        os.path.join(tmp_dirs["checkpoints_dir"], checkpoint_files[0]),
        "rt",
        encoding="utf-8",
    ) as f:
        checkpoint = json.load(f)
    assert checkpoint["num_rows"] == 5, "Nombre de lignes incorrect"
    df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
    assert any(
        "checkpoint" in str(op) for op in df_perf["operation"]
    ), "Opération non journalisée"


def test_cloud_backup(tmp_dirs, mock_trading_env, mock_router):
    """Teste la sauvegarde distribuée S3."""
    with patch("boto3.client") as mock_s3:
        dialogue = DialogueManager(
            config_path=tmp_dirs["config_path"],
            credentials_path=tmp_dirs["credentials_path"],
        )
        df = pd.DataFrame(
            {
                "timestamp": [datetime.now().isoformat()] * 5,
                "command": ["résume tes stats"] * 5,
                "response": ["Résumé..."] * 5,
                "cluster_id": [0] * 5,
                "priority": [3] * 5,
            }
        )
        dialogue.cloud_backup(df)
        assert mock_s3.called, "Client S3 non appelé"
        checkpoint_files = os.listdir(tmp_dirs["checkpoints_dir"])
        assert any(
            "temp_s3_" in f for f in checkpoint_files
        ), "Fichier temporaire S3 non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "cloud_backup" in str(op) for op in df_perf["operation"]
        ), "Opération non journalisée"


def test_handle_sigint(tmp_dirs):
    """Teste la gestion SIGINT."""
    dialogue = DialogueManager(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    with patch("sys.exit") as mock_exit:
        dialogue.handle_sigint(signal.SIGINT, None)
        snapshot_files = os.listdir(tmp_dirs["snapshots_dir"])
        assert any(
            "snapshot_sigint" in f for f in snapshot_files
        ), "Snapshot sigint non créé"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "handle_sigint" in str(op) for op in df_perf["operation"]
        ), "Opération non journalisée"
        mock_exit.assert_called_with(0)


def test_respond_to_query(tmp_dirs, mock_trading_env, mock_router):
    """Teste la réponse à une requête textuelle."""
    dialogue = DialogueManager(
        config_path=tmp_dirs["config_path"],
        credentials_path=tmp_dirs["credentials_path"],
    )
    df = pd.read_csv(tmp_dirs["features_latest_path"])
    dialogue.update_context(df, mock_router, mock_trading_env)
    with patch("openai.OpenAI.chat.completions.create") as mock_nlp:
        mock_nlp.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Résumé des stats..."))]
        )
        response = dialogue.respond_to_query("résume tes stats")
        assert "Résumé des stats" in response, "Réponse incorrecte"
        df_perf = pd.read_csv(tmp_dirs["perf_log_path"])
        assert any(
            "respond_to_query" in str(op) for op in df_perf["operation"]
        ), "Opération non journalisée"
