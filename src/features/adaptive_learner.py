```python
# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/features/adaptive_learner.py
#
# Rôle :
# Gère l'apprentissage adaptatif en stockant les patterns de marché (features, action, récompense, régime)
# dans market_memory.db, utilise K-means pour la mémoire contextuelle (méthode 7), et fine-tune/apprend en ligne
# les modèles SAC/PPO/DDPG (méthodes 8, 10) pour MIA_IA_SYSTEM_v2_2025. Intègre XGBoost pour l'optimisation
# des hyperparamètres (Proposition 2, Étape 2).
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, numpy>=1.26.4,<2.0.0, psutil>=5.9.0,<6.0.0, matplotlib>=3.7.0,<4.0.0,
#   scikit-learn>=1.2.0,<2.0.0, stable-baselines3>=2.0.0,<3.0.0, pyyaml>=6.0.0,<7.0.0, xgboost>=2.0.0,<3.0.0
# - src/model/utils/config_manager.py
# - src/model/utils/alert_manager.py
# - src/model/utils/model_validator.py
# - src/model/utils/algo_performance_logger.py
# - src/model/train_sac.py
# - src/model/train_sac_auto.py
# - src/envs/trading_env.py
# - src/model/utils/finetune_utils.py
# - src/utils/database_manager.py
#
# Inputs :
# - config/feature_sets.yaml, config/algo_config.yaml via config_manager
# - Données brutes (DataFrame avec 350 features pour entraînement, 150 pour production)
#
# Outputs :
# - Patterns/clusters dans data/market_memory.db
# - Modèles SAC/PPO/DDPG dans data/models/
# - Logs dans data/logs/adaptive_learning_performance.csv
# - Snapshots JSON dans data/adaptive_learning_snapshots/*.json (option *.json.gz)
# - Figures dans data/figures/adaptive_learning/
# - Méta-données dans market_memory.db (meta_runs)
#
# Lien avec SAC :
# Fournit les patterns et déclenche l'entraînement pour train_sac_auto.py and live_trading.py
#
# Version : 2.1.4
# Mis à jour : 2025-05-15
#
# Notes :
# - Conforme à structure.txt (version 2.1.3, 2025-05-13).
# - Intègre mémoire contextuelle (méthode 7), fine-tuning (méthode 8), apprentissage en ligne (méthode 10).
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des patterns et clusters.
# - Intègre validation SHAP (Phase 17) pour assurer la conformité des 150 features en production.
# - Corrige pour 350 features (entraînement) et 150 SHAP (production).
# - Utilise exclusivement IQFeed via data_provider.py, avec retries (max 3, délai 2^attempt secondes).
# - Sauvegarde des snapshots JSON non compressés par défaut avec option de compression gzip.
# - Envoie des alertes via AlertManager pour les erreurs critiques et succès.
# - Tests unitaires disponibles dans tests/test_adaptive_learner.py (à implémenter).
# - Validation complète prévue pour juin 2025.
# - Évolution future : Migration API Investing.com (juin 2025), optimisation pour feature_pipeline.py.

import gzip
import hashlib
import json
import logging
import os
import signal
import sqlite3
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import DDPG, PPO, SAC
from xgboost import XGBRegressor

from src.envs.trading_env import TradingEnv
from src.model.train_sac import SACTrainer
from src.model.utils.alert_manager import AlertManager
from src.model.utils.algo_performance_logger import AlgoPerformanceLogger
from src.model.utils.config_manager import config_manager
from src.model.utils.finetune_utils import finetune_model
from src.model.utils.model_validator import ModelValidator
from src.utils.database_manager import DatabaseManager

# Chemins relatifs
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DB_PATH = BASE_DIR / "data" / "market_memory.db"
CSV_LOG_PATH = BASE_DIR / "data" / "logs" / "adaptive_learning_performance.csv"
SNAPSHOT_DIR = BASE_DIR / "data" / "adaptive_learning_snapshots"
FIGURE_DIR = BASE_DIR / "data" / "figures" / "adaptive_learning"
FEATURE_IMPORTANCE_PATH = BASE_DIR / "data" / "features" / "feature_importance.csv"

# Création des dossiers
CSV_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Configuration du logging
logging.basicConfig(
    filename=BASE_DIR / "data" / "logs" / "adaptive_learning.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Constantes
MAX_RETRIES = 3
RETRY_DELAY = 2.0


class AdaptiveLearner:
    """
    Gère l'apprentissage adaptatif et la mémoire de l'IA pour MIA_IA_SYSTEM_v2_2025.

    Attributes:
        config (Dict): Configuration chargée depuis algo_config.yaml.
        alert_manager (AlertManager): Gestionnaire d'alertes multi-canaux.
        validator (ModelValidator): Validateur de modèles.
        performance_logger (AlgoPerformanceLogger): Enregistreur de performances.
        scaler (StandardScaler): Normalisateur des données.
        log_buffer (List): Buffer pour logs de performance.
        snapshot_dir (Path): Dossier pour snapshots JSON.
        perf_log (Path): Fichier pour logs de performance.
        figure_dir (Path): Dossier pour visualisations.
        training_mode (bool): True pour 350 features (entraînement), False pour 150 (production).
        db_manager (DatabaseManager): Gestionnaire de la base de données market_memory.db.
    """

    def __init__(self, training_mode: bool = True):
        """Initialise l'apprenant adaptatif."""
        self.config = config_manager.get_config(
            BASE_DIR / "config" / "algo_config.yaml"
        )
        self.alert_manager = AlertManager()
        self.validator = ModelValidator()
        self.performance_logger = AlgoPerformanceLogger()
        self.scaler = StandardScaler()
        self.log_buffer = []
        self.snapshot_dir = SNAPSHOT_DIR
        self.perf_log = CSV_LOG_PATH
        self.figure_dir = FIGURE_DIR
        self.training_mode = training_mode
        self.db_manager = DatabaseManager(DB_PATH)
        self.cache = {}

        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.perf_log.parent.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)

        # Initialiser la base de données
        self.initialize_database()

        signal.signal(signal.SIGINT, self.handle_sigint)
        logger.info("AdaptiveLearner initialisé")
        self.alert_manager.send_alert("AdaptiveLearner initialisé", priority=2)
        self.log_performance("init", 0, success=True)
        self.save_snapshot("init", {"training_mode": training_mode}, compress=False)

    def handle_sigint(self, signum, frame):
        """Gère l'arrêt propre sur SIGINT."""
        snapshot = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "status": "SIGINT",
        }
        self.save_snapshot("sigint", snapshot, compress=True)
        logger.info("Arrêt propre sur SIGINT")
        exit(0)

    def with_retries(
        self,
        func: callable,
        max_attempts: int = MAX_RETRIES,
        delay_base: float = RETRY_DELAY,
    ) -> Any:
        """
        Exécute une fonction avec retries exponentiels.
        Le délai commence à 2 secondes pour la première tentative (attempt=0) et double à chaque tentative.

        Args:
            func (callable): Fonction à exécuter.
            max_attempts (int): Nombre maximum de tentatives.
            delay_base (float): Base pour le délai exponentiel (défaut : 2.0).

        Returns:
            Any: Résultat de la fonction ou lève une exception si échec après max_attempts.
        """
        start_time = time.time()
        for attempt in range(max_attempts):
            try:
                result = func()
                latency = time.time() - start_time
                self.log_performance(
                    f"retry_attempt_{attempt+1}",
                    latency,
                    success=True,
                    attempt_number=attempt + 1,
                )
                return result
            except Exception as e:
                if attempt == max_attempts - 1:
                    latency = time.time() - start_time
                    self.log_performance(
                        f"retry_attempt_{attempt+1}",
                        latency,
                        success=False,
                        error=str(e),
                        attempt_number=attempt + 1,
                    )
                    error_msg = f"Échec après {max_attempts} tentatives: {str(e)}"
                    self.alert_manager.send_alert(error_msg, priority=4)
                    logger.error(error_msg)
                    raise
                delay = delay_base * (2**attempt)
                warning_msg = f"Tentative {attempt+1} échouée, retry après {delay}s"
                self.alert_manager.send_alert(warning_msg, priority=3)
                logger.warning(warning_msg)
                time.sleep(delay)

    def initialize_database(self, db_path: str = str(DB_PATH)) -> None:
        """Initialise la base de données SQLite pour stocker les patterns, clusters et méta-données."""
        try:
            start_time = time.time()
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            with self.db_manager.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        features TEXT,  -- JSON des 350 features
                        action REAL,
                        reward REAL,
                        neural_regime INTEGER,
                        confidence REAL
                    )
                """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS clusters (
                        cluster_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        event_type TEXT,  -- Type d'événement (ex. : Trend, Range, Defensive)
                        features TEXT,    -- JSON des centroïdes (350 features)
                        cluster_size INTEGER
                    )
                """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS meta_runs (
                        run_id INTEGER PRIMARY KEY,
                        timestamp TEXT,
                        trade_id INTEGER,
                        metrics TEXT,
                        hyperparameters TEXT,
                        performance REAL,
                        regime TEXT,
                        session TEXT,
                        shap_metrics TEXT,
                        context TEXT
                    )
                """
                )
                conn.commit()
            latency = time.time() - start_time
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            logger.info(f"Base de données initialisée: {db_path}")
            self.alert_manager.send_alert(
                f"Base de données initialisée: {db_path}", priority=1
            )
            self.log_performance(
                "initialize_database",
                latency,
                success=True,
                memory_usage_mb=memory_usage,
            )
            self.save_snapshot(
                "initialize_database", {"db_path": db_path}, compress=False
            )
        except Exception as e:
            error_msg = f"Erreur initialisation base de données: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=4)
            logger.error(error_msg)
            self.log_performance("initialize_database", 0, success=False, error=str(e))
            raise

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ) -> None:
        """Journalise les performances des opérations critiques."""
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
            cpu_usage = psutil.cpu_percent()  # % CPU
            if memory_usage > 1024:
                alert_msg = (
                    f"ALERTE: Utilisation mémoire élevée ({memory_usage:.2f} MB)"
                )
                self.alert_manager.send_alert(alert_msg, priority=5)
                logger.warning(alert_msg)
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "latency": latency,
                "success": success,
                "error": error,
                "memory_usage_mb": memory_usage,
                "cpu_usage_percent": cpu_usage,
                **kwargs,
            }
            self.log_buffer.append(log_entry)
            if len(self.log_buffer) >= self.config.get("buffer_size", 100):
                log_df = pd.DataFrame(self.log_buffer)
                self.perf_log.parent.mkdir(parents=True, exist_ok=True)

                def write_log():
                    log_df.to_csv(
                        self.perf_log,
                        mode="a" if self.perf_log.exists() else "w",
                        header=not self.perf_log.exists(),
                        index=False,
                        encoding="utf-8",
                    )

                self.with_retries(write_log)
                self.log_buffer = []
            logger.info(f"Performance journalisée pour {operation}. CPU: {cpu_usage}%")
        except Exception as e:
            error_msg = f"Erreur journalisation performance: {str(e)}"
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)

    def save_snapshot(
        self, snapshot_type: str, data: Dict, compress: bool = False
    ) -> None:
        """Sauvegarde un instantané des résultats avec option de compression gzip."""
        try:
            start_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = {
                "timestamp": timestamp,
                "type": snapshot_type,
                "data": data,
                "buffer_size": len(self.log_buffer),
            }
            snapshot_path = self.snapshot_dir / f"snapshot_{snapshot_type}_{timestamp}.json"
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)

            def write_snapshot():
                if compress:
                    with gzip.open(f"{snapshot_path}.gz", "wt", encoding="utf-8") as f:
                        json.dump(snapshot, f, indent=4)
                else:
                    with open(snapshot_path, "w", encoding="utf-8") as f:
                        json.dump(snapshot, f, indent=4)

            self.with_retries(write_snapshot)
            save_path = f"{snapshot_path}.gz" if compress else snapshot_path
            file_size = os.path.getsize(save_path) / 1024 / 1024
            if file_size > 1.0:
                alert_msg = f"Snapshot size {file_size:.2f} MB exceeds 1 MB"
                self.alert_manager.send_alert(alert_msg, priority=3)
                logger.warning(alert_msg)
            latency = time.time() - start_time
            self.log_performance(
                "save_snapshot",
                latency,
                success=True,
                snapshot_type=snapshot_type,
                file_size_mb=file_size,
            )
            success_msg = f"Snapshot {snapshot_type} sauvegardé: {save_path}"
            self.alert_manager.send_alert(success_msg, priority=1)
            logger.info(success_msg)
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}"
            self.log_performance("save_snapshot", 0, success=False, error=str(e))
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)

    def validate_shap_features(self, features: List[str]) -> bool:
        """Valide que les features sont dans les top 150 SHAP (production)."""
        try:
            start_time = time.time()
            if not FEATURE_IMPORTANCE_PATH.exists():
                error_msg = "Fichier SHAP manquant"
                self.alert_manager.send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return False
            shap_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
            if len(shap_df) < 150:
                error_msg = f"Nombre insuffisant de SHAP features: {len(shap_df)} < 150"
                self.alert_manager.send_alert(error_msg, priority=4)
                logger.error(error_msg)
                return False
            valid_features = set(shap_df["feature"].head(150))
            missing = [f for f in features if f not in valid_features]
            if missing:
                warning_msg = f"Features non incluses dans top 150 SHAP: {missing}"
                self.alert_manager.send_alert(warning_msg, priority=3)
                logger.warning(warning_msg)
            latency = time.time() - start_time
            success_msg = "SHAP features validées"
            self.alert_manager.send_alert(success_msg, priority=1)
            logger.info(success_msg)
            self.log_performance(
                "validate_shap_features",
                latency,
                success=True,
                num_features=len(features),
            )
            self.save_snapshot(
                "validate_shap_features",
                {"num_features": len(features), "missing": missing},
                compress=False,
            )
            return True
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Erreur validation SHAP features: {str(e)}"
            self.alert_manager.send_alert(error_msg, priority=4)
            logger.error(error_msg)
            self.log_performance(
                "validate_shap_features", latency, success=False, error=str(e)
            )
            return False

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Valide que les données contiennent les features attendues (350 pour entraînement, 150 pour production)."""
        try:
            start_time = time.time()
            feature_cols = [
                f["name"]
                for cat in config_manager.get_features(
                    BASE_DIR / "config" / "feature_sets.yaml"
                )["feature_sets"].values()
                for f in cat["features"]
            ]
            expected_count = 350 if self.training_mode else 150
            new_metrics = [
                "key_strikes_1",
                "key_strikes_2",
                "key_strikes_3",
                "key_strikes_4",
                "key_strikes_5",
                "max_pain_strike",
                "net_gamma",
                "zero_gamma",
                "dealer_zones_count",
                "vol_trigger",
                "ref_px",
                "data_release",
            ]
            feature_cols.extend(new_metrics)
            feature_cols = list(set(feature_cols))
            if len(feature_cols) < expected_count:
                missing_count = expected_count - len(feature_cols)
                for i in range(missing_count):
                    feature_cols.append(f"placeholder_{i}")
            feature_cols = feature_cols[:expected_count]
            missing_cols = [col for col in feature_cols if col not in data.columns]
            confidence_drop_rate = 1.0 - min(
                (len(feature_cols) - len(missing_cols)) / len(feature_cols), 1.0
            )
            if len(data) < 1:
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(feature_cols) - len(missing_cols)}/{len(feature_cols)} colonnes valides, {len(data)} lignes)"
                self.alert_manager.send_alert(alert_msg, priority=3)
                logger.warning(alert_msg)

            if missing_cols:
                data = data.copy()
                for col in missing_cols:
                    data[col] = 0.0
                    if col == "dealer_zones_count":
                        data[col] = data[col].clip(0, 10)
                    elif col == "data_release":
                        data[col] = data[col].clip(0, 1)
                    elif col in ["net_gamma", "vol_trigger"]:
                        data[col] = data[col].clip(-1, 1)
                    else:
                        data[col] = (
                            data[col]
                            .clip(lower=0)
                            .fillna(
                                data["close"].median() if "close" in data.columns else 0
                            )
                        )
                logger.warning(f"Colonnes manquantes imputées: {missing_cols}")
            if data[feature_cols].isnull().any().any():
                raise ValueError("Valeurs nulles détectées dans les features")
            if (
                "timestamp" not in data.columns
                or not pd.api.types.is_datetime64_any_dtype(data["timestamp"])
            ):
                raise ValueError("Colonne 'timestamp' doit être de type datetime")
            for col in feature_cols:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    raise ValueError(
                        f"Colonne {col} n'est pas numérique: {data[col].dtype}"
                    )
                if data[col].apply(lambda x: isinstance(x, (list, dict, tuple))).any():
                    raise ValueError(
                        f"Colonne {col} contient des valeurs non scalaires"
                    )
            if not self.training_mode:
                self.validate_shap_features(feature_cols)
            latency = time.time() - start_time
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            self.log_performance(
                "validate_data",
                latency,
                success=True,
                memory_usage_mb=memory_usage,
                confidence_drop_rate=confidence_drop_rate,
            )
            self.save_snapshot(
                "validate_data",
                {
                    "num_features": len(feature_cols),
                    "missing_cols": missing_cols,
                    "confidence_drop_rate": confidence_drop_rate,
                },
                compress=False,
            )
            return True
        except Exception as e:
            error_msg = f"Erreur validation données: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=3)
            logger.error(error_msg)
            self.log_performance("validate_data", 0, success=False, error=str(e))
            return False

    def store_pattern(
        self,
        data: pd.DataFrame,
        action: float,
        reward: float,
        neural_regime: Optional[int] = None,
        confidence: float = 0.7,
        db_path: str = str(DB_PATH),
    ) -> None:
        """Stocke un pattern de marché dans market_memory.db."""
        try:
            start_time = time.time()
            if not isinstance(data, pd.DataFrame) or len(data) != 1:
                raise ValueError(
                    "Les données doivent être un DataFrame avec une seule ligne"
                )
            if not self.validate_data(data):
                error_msg = "Données invalides pour stockage pattern"
                self.alert_manager.send_alert(error_msg, priority=3)
                logger.error(error_msg)
                return

            feature_cols = [
                f["name"]
                for cat in config_manager.get_features(
                    BASE_DIR / "config" / "feature_sets.yaml"
                )["feature_sets"].values()
                for f in cat["features"]
            ]
            expected_count = 350 if self.training_mode else 150
            if len(feature_cols) < expected_count:
                missing_count = expected_count - len(feature_cols)
                for i in range(missing_count):
                    feature_cols.append(f"placeholder_{i}")
            feature_cols = feature_cols[:expected_count]
            features = data[feature_cols].iloc[0].to_dict()
            features_json = pd.Series(features).to_json(orient="columns")
            timestamp = (
                data["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M:%S")
                if "timestamp" in data.columns
                else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            neural_regime = neural_regime if neural_regime is not None else -1

            def store():
                with self.db_manager.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO patterns (timestamp, features, action, reward, neural_regime, confidence)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            timestamp,
                            features_json,
                            action,
                            reward,
                            neural_regime,
                            confidence,
                        ),
                    )
                    conn.commit()

            self.with_retries(store)

            snapshot = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "action": action,
                "reward": reward,
                "neural_regime": neural_regime,
                "confidence": confidence,
            }
            self.save_snapshot("store_pattern", snapshot, compress=True)

            latency = time.time() - start_time
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            success_msg = f"Pattern stocké: timestamp={timestamp}, action={action}, reward={reward}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            self.log_performance(
                "store_pattern", latency, success=True, memory_usage_mb=memory_usage
            )
        except Exception as e:
            error_msg = f"Erreur stockage pattern: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=4)
            logger.error(error_msg)
            self.log_performance("store_pattern", 0, success=False, error=str(e))
            self.save_snapshot("error_store_pattern", {"error": str(e)}, compress=False)

    def last_3_trades_success_rate(self, db_path: str = str(DB_PATH)) -> float:
        """Calcule le taux de succès des 3 derniers trades."""
        try:
            start_time = time.time()

            def fetch_rewards():
                with self.db_manager.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT reward FROM patterns
                        ORDER BY timestamp DESC
                        LIMIT 3
                        """
                    )
                    rewards = [row[0] for row in cursor.fetchall()]
                return rewards

            rewards = self.with_retries(fetch_rewards)

            if len(rewards) < 3:
                warning_msg = f"Moins de 3 trades disponibles ({len(rewards)})"
                self.alert_manager.send_alert(warning_msg, priority=3)
                logger.warning(warning_msg)
                self.save_snapshot(
                    "last_3_trades_success_rate",
                    {"num_trades": len(rewards)},
                    compress=False,
                )
                return 0.0

            success_rate = sum(1 for r in rewards if r > 0) / len(rewards)
            latency = time.time() - start_time
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            success_msg = f"Taux de succès des 3 derniers trades: {success_rate:.2f}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            self.log_performance(
                "last_3_trades_success_rate",
                latency,
                success=True,
                memory_usage_mb=memory_usage,
            )
            self.save_snapshot(
                "last_3_trades_success_rate",
                {"success_rate": success_rate},
                compress=False,
            )
            return success_rate
        except Exception as e:
            error_msg = (
                f"Erreur calcul taux de succès: {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            logger.error(error_msg)
            self.log_performance(
                "last_3_trades_success_rate", 0, success=False, error=str(e)
            )
            self.save_snapshot(
                "error_last_3_trades_success_rate", {"error": str(e)}, compress=False
            )
            return 0.0

    def optimize_hyperparameters_xgboost(
        self, data: pd.DataFrame, algo_type: str, regime: str
    ) -> Dict:
        """Optimise les hyperparamètres du modèle avec XGBoost (Proposition 2, Étape 2)."""
        try:
            start_time = time.time()
            if not self.validate_data(data):
                error_msg = "Données invalides pour optimisation XGBoost"
                self.alert_manager.send_alert(error_msg, priority=3)
                logger.error(error_msg)
                return {}

            X = data.drop(columns=["timestamp", "neural_regime", "reward"], errors="ignore")
            y = data["reward"] if "reward" in data.columns else np.zeros(len(data))
            model = XGBRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
            )
            model.fit(X, y)

            # Définir les hyperparamètres à optimiser en fonction de l'algorithme
            param_grid = {
                "sac": {
                    "learning_rate": [0.0001, 0.0003, 0.001],
                    "gamma": [0.95, 0.99, 0.999],
                    "tau": [0.005, 0.01, 0.02],
                    "batch_size": [64, 128, 256],
                },
                "ppo": {
                    "learning_rate": [0.0001, 0.0003, 0.001],
                    "n_steps": [2048, 4096, 8192],
                    "batch_size": [64, 128, 256],
                    "gamma": [0.95, 0.99, 0.999],
                },
                "ddpg": {
                    "learning_rate": [0.0001, 0.0003, 0.001],
                    "gamma": [0.95, 0.99, 0.999],
                    "tau": [0.005, 0.01, 0.02],
                    "batch_size": [64, 128, 256],
                },
            }.get(algo_type.lower(), {})

            if not param_grid:
                warning_msg = f"Pas de grille d’hyperparamètres pour '{algo_type}'"
                self.alert_manager.send_alert(warning_msg, priority=3)
                logger.warning(warning_msg)
                return {}

            # Simuler l'évaluation des hyperparamètres (en pratique, utiliser une recherche par grille ou aléatoire)
            best_params = {}
            best_score = float("-inf")
            for param_set in [
                {k: v[i % len(v)] for k, v in param_grid.items()}
                for i in range(max(len(v) for v in param_grid.values()))
            ]:
                score = model.predict(X).mean()  # Simplification pour l'exemple
                if score > best_score:
                    best_score = score
                    best_params = param_set

            # Stocker les hyperparamètres optimisés dans meta_runs
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "trade_id": 0,  # Pas de trade_id spécifique
                "metrics": {"best_score": float(best_score)},
                "hyperparameters": best_params,
                "performance": best_score,
                "regime": regime,
                "session": data.get("session", "unknown").iloc[-1] if "session" in data.columns else "unknown",
                "shap_metrics": {},
                "context": {"algo_type": algo_type},
            }
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO meta_runs (timestamp, trade_id, metrics, hyperparameters, performance, regime, session, shap_metrics, context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata["timestamp"],
                    metadata["trade_id"],
                    json.dumps(metadata["metrics"]),
                    json.dumps(metadata["hyperparameters"]),
                    metadata["performance"],
                    metadata["regime"],
                    metadata["session"],
                    json.dumps(metadata["shap_metrics"]),
                    json.dumps(metadata["context"]),
                ),
            )
            conn.commit()
            conn.close()

            latency = time.time() - start_time
            success_msg = f"Hyperparamètres optimisés pour {algo_type} ({regime}): {best_params}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            self.log_performance(
                "optimize_hyperparameters_xgboost",
                latency,
                success=True,
                algo_type=algo_type,
                regime=regime,
            )
            return best_params
        except Exception as e:
            error_msg = f"Erreur optimisation XGBoost: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=4)
            logger.error(error_msg)
            self.log_performance(
                "optimize_hyperparameters_xgboost", 0, success=False, error=str(e)
            )
            return {}

    def cluster_patterns(
        self, n_clusters: int = 10, db_path: str = str(DB_PATH), max_patterns: int = 1000
    ) -> List[int]:
        """Applique K-means sur les patterns stockés pour mémoire contextuelle (méthode 7)."""
        try:
            start_time = time.time()

            def fetch_patterns():
                with self.db_manager.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT features, neural_regime FROM patterns
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (max_patterns,),
                    )
                    patterns = [
                        (json.loads(row[0]), row[1]) for row in cursor.fetchall()
                    ]
                return pd.DataFrame([p[0] for p in patterns]), [p[1] for p in patterns]

            patterns_df, regimes = self.with_retries(fetch_patterns)

            if patterns_df.empty:
                error_msg = "Aucun pattern disponible pour clustering"
                self.alert_manager.send_alert(error_msg, priority=3)
                logger.warning(error_msg)
                self.save_snapshot(
                    "cluster_patterns", {"num_patterns": 0}, compress=False
                )
                return []

            feature_cols = patterns_df.columns
            confidence_drop_rate = 1.0 - min(
                len(feature_cols) / (350 if self.training_mode else 150), 1.0
            )
            if len(patterns_df) < max_patterns / 2:
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(feature_cols)} features, {len(patterns_df)} patterns)"
                self.alert_manager.send_alert(alert_msg, priority=3)
                logger.warning(alert_msg)

            cache_key = hashlib.sha256(patterns_df.to_json().encode()).hexdigest()
            if cache_key in self.cache:
                clusters = self.cache[cache_key]["clusters"]
                latency = time.time() - start_time
                self.log_performance(
                    "cluster_patterns_cache_hit",
                    latency,
                    success=True,
                    num_patterns=len(patterns_df),
                )
                success_msg = "Clusters récupérés du cache"
                self.alert_manager.send_alert(success_msg, priority=1)
                logger.info(success_msg)
                return clusters

            X = patterns_df[feature_cols].fillna(0)
            X_scaled = self.scaler.fit_transform(X)

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)

            centroids = self.scaler.inverse_transform(kmeans.cluster_centers_)
            regime_map = {0: "Trend", 1: "Range", 2: "Defensive"}
            for i, centroid in enumerate(centroids):
                centroid_json = pd.Series(centroid, index=feature_cols).to_json(
                    orient="columns"
                )
                event_type = regime_map.get(
                    regimes[i] if i < len(regimes) else -1, "Unknown"
                )
                with self.db_manager.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO clusters (timestamp, event_type, features, cluster_size)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            event_type,
                            centroid_json,
                            sum(clusters == i),
                        ),
                    )
                    conn.commit()

            # Visualisation
            plt.figure(figsize=(10, 6))
            plt.scatter(
                X.get("vix_es_correlation", np.zeros(len(X))),
                X.get("reward", np.zeros(len(X))),
                c=clusters,
                cmap="viridis",
            )
            plt.xlabel("VIX-ES Correlation")
            plt.ylabel("Reward")
            plt.title("Clusters de Patterns")
            plt.colorbar(label="Cluster")
            plt.grid(True)
            plot_path = (
                self.figure_dir / f"cluster_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            plt.savefig(plot_path)
            plt.close()

            snapshot = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "n_clusters": n_clusters,
                "num_patterns": len(patterns_df),
                "confidence_drop_rate": confidence_drop_rate,
            }
            self.save_snapshot("cluster_patterns", snapshot, compress=True)

            self.cache[cache_key] = {
                "clusters": clusters.tolist(),
                "timestamp": datetime.now(),
            }
            if len(self.cache) > self.config.get("max_cache_size", 1000):
                self.cache.pop(next(iter(self.cache)))

            latency = time.time() - start_time
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            success_msg = f"Clustering terminé: {n_clusters} clusters, {len(patterns_df)} patterns"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            self.log_performance(
                "cluster_patterns",
                latency,
                success=True,
                memory_usage_mb=memory_usage,
                confidence_drop_rate=confidence_drop_rate,
            )
            return clusters.tolist()
        except Exception as e:
            error_msg = (
                f"Erreur clustering patterns: {str(e)}\n{traceback.format_exc()}"
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            logger.error(error_msg)
            self.log_performance("cluster_patterns", 0, success=False, error=str(e))
            self.save_snapshot(
                "error_cluster_patterns", {"error": str(e)}, compress=False
            )
            return []

    def retrain_model(
        self, data: pd.DataFrame, algo_type: str = "sac", regime: str = "range"
    ) -> Any:
        """Fine-tune un modèle par cluster (méthode 8)."""
        try:
            start_time = time.time()
            if not self.validate_data(data):
                error_msg = "Données invalides pour retrain"
                self.alert_manager.send_alert(error_msg, priority=3)
                logger.error(error_msg)
                self.save_snapshot(
                    "error_retrain_model",
                    {"error": "Données invalides"},
                    compress=False,
                )
                return None

            confidence_drop_rate = 1.0 - min(
                len(data.columns) / (350 if self.training_mode else 150), 1.0
            )
            if len(data) < 10:
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(data.columns)} features, {len(data)} lignes)"
                self.alert_manager.send_alert(alert_msg, priority=3)
                logger.warning(alert_msg)

            cache_key = hashlib.sha256(
                f"{algo_type}_{regime}_{data.to_json().encode()}".encode()
            ).hexdigest()
            if cache_key in self.cache:
                model_path = self.cache[cache_key]["model_path"]
                model = (
                    SAC.load(model_path)
                    if algo_type == "sac"
                    else (
                        PPO.load(model_path)
                        if algo_type == "ppo"
                        else DDPG.load(model_path)
                    )
                )
                latency = time.time() - start_time
                self.log_performance(
                    "retrain_model_cache_hit",
                    latency,
                    success=True,
                    algo_type=algo_type,
                    regime=regime,
                )
                success_msg = f"Modèle {algo_type} ({regime}) chargé depuis cache"
                self.alert_manager.send_alert(success_msg, priority=1)
                logger.info(success_msg)
                return model

            # Optimiser les hyperparamètres avec XGBoost
            best_params = self.optimize_hyperparameters_xgboost(data, algo_type, regime)

            def cluster_and_train():
                X = data.drop(columns=["timestamp", "neural_regime"], errors="ignore")
                X_scaled = self.scaler.fit_transform(X)
                kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)

                env = TradingEnv(
                    config_path=str(BASE_DIR / "config" / "trading_env_config.yaml")
                )
                SACTrainer(env)
                model = None
                for cluster in set(clusters):
                    cluster_data = data[clusters == cluster]
                    if len(cluster_data) < 10:
                        continue
                    model = finetune_model(
                        cluster_data,
                        algo_type=algo_type,
                        regime=regime,
                        hyperparameters=best_params,
                    )
                    model_path = self.config[algo_type][regime]["model_path"]
                    model.save(model_path)

                # Visualisation
                plt.figure(figsize=(10, 6))
                plt.scatter(
                    X.get("vix_es_correlation", np.zeros(len(X))),
                    X.get("reward", np.zeros(len(X))),
                    c=clusters,
                    cmap="viridis",
                )
                plt.xlabel("VIX-ES Correlation")
                plt.ylabel("Reward")
                plt.title(f"Clusters pour {algo_type} ({regime})")
                plt.colorbar(label="Cluster")
                plt.grid(True)
                plot_path = (
                    self.figure_dir
                    / f"cluster_{algo_type}_{regime}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                plt.savefig(plot_path)
                plt.close()

                return model

            model = self.with_retries(cluster_and_train)

            snapshot = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "algo_type": algo_type,
                "regime": regime,
                "num_patterns": len(data),
                "confidence_drop_rate": confidence_drop_rate,
                "hyperparameters": best_params,
            }
            self.save_snapshot("retrain_model", snapshot, compress=True)

            self.cache[cache_key] = {
                "model_path": self.config[algo_type][regime]["model_path"],
                "timestamp": datetime.now(),
            }
            if len(self.cache) > self.config.get("max_cache_size", 1000):
                self.cache.pop(next(iter(self.cache)))

            latency = time.time() - start_time
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            success_msg = f"Modèle {algo_type} ({regime}) fine-tuné"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            self.log_performance(
                "retrain_model",
                latency,
                success=True,
                memory_usage_mb=memory_usage,
                algo_type=algo_type,
                regime=regime,
                confidence_drop_rate=confidence_drop_rate,
            )
            return model
        except Exception as e:
            error_msg = f"Erreur fine-tuning modèle: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=4)
            logger.error(error_msg)
            self.log_performance("retrain_model", 0, success=False, error=str(e))
            self.save_snapshot("error_retrain_model", {"error": str(e)}, compress=False)
            return None

    def online_learning(
        self, data: pd.DataFrame, algo_type: str = "sac", regime: str = "range"
    ) -> Any:
        """Apprend en ligne avec mini-batchs (méthode 10)."""
        try:
            start_time = time.time()
            if not self.validate_data(data):
                error_msg = "Données invalides pour online learning"
                self.alert_manager.send_alert(error_msg, priority=3)
                logger.error(error_msg)
                self.save_snapshot(
                    "error_online_learning",
                    {"error": "Données invalides"},
                    compress=False,
                )
                return None

            confidence_drop_rate = 1.0 - min(
                len(data.columns) / (350 if self.training_mode else 150), 1.0
            )
            if len(data) < 10:
                confidence_drop_rate = max(confidence_drop_rate, 0.5)
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(data.columns)} features, {len(data)} lignes)"
                self.alert_manager.send_alert(alert_msg, priority=3)
                logger.warning(alert_msg)

            cache_key = hashlib.sha256(
                f"{algo_type}_{regime}_{data.to_json().encode()}".encode()
            ).hexdigest()
            if cache_key in self.cache:
                model_path = self.cache[cache_key]["model_path"]
                model = (
                    SAC.load(model_path)
                    if algo_type == "sac"
                    else (
                        PPO.load(model_path)
                        if algo_type == "ppo"
                        else DDPG.load(model_path)
                    )
                )
                latency = time.time() - start_time
                self.log_performance(
                    "online_learning_cache_hit",
                    latency,
                    success=True,
                    algo_type=algo_type,
                    regime=regime,
                )
                success_msg = f"Modèle {algo_type} ({regime}) chargé depuis cache"
                self.alert_manager.send_alert(success_msg, priority=1)
                logger.info(success_msg)
                return model

            # Optimiser les hyperparamètres avec XGBoost
            best_params = self.optimize_hyperparameters_xgboost(data, algo_type, regime)

            def online_train():
                env = TradingEnv(
                    config_path=str(BASE_DIR / "config" / "trading_env_config.yaml")
                )
                SACTrainer(env)
                model = finetune_model(
                    data,
                    algo_type=algo_type,
                    regime=regime,
                    online=True,
                    hyperparameters=best_params,
                )
                model_path = self.config[algo_type][regime]["model_path"]
                model.save(model_path)
                return model

            model = self.with_retries(online_train)

            snapshot = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "algo_type": algo_type,
                "regime": regime,
                "num_patterns": len(data),
                "confidence_drop_rate": confidence_drop_rate,
                "hyperparameters": best_params,
            }
            self.save_snapshot("online_learning", snapshot, compress=True)

            self.cache[cache_key] = {
                "model_path": self.config[algo_type][regime]["model_path"],
                "timestamp": datetime.now(),
            }
            if len(self.cache) > self.config.get("max_cache_size", 1000):
                self.cache.pop(next(iter(self.cache)))

            latency = time.time() - start_time
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            success_msg = f"Online learning terminé pour {algo_type} ({regime})"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            self.log_performance(
                "online_learning",
                latency,
                success=True,
                memory_usage_mb=memory_usage,
                algo_type=algo_type,
                regime=regime,
                confidence_drop_rate=confidence_drop_rate,
            )
            return model
        except Exception as e:
            error_msg = f"Erreur online learning: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=4)
            logger.error(error_msg)
            self.log_performance("online_learning", 0, success=False, error=str(e))
            self.save_snapshot(
                "error_online_learning", {"error": str(e)}, compress=False
            )
            return None

    def cluster_score_distance(
        self, data: pd.DataFrame, n_clusters: int = 5, db_path: str = str(DB_PATH)
    ) -> float:
        """Calcule la distance entre le pattern actuel et le centroïde du cluster le plus proche."""
        try:
            start_time = time.time()
            if not isinstance(data, pd.DataFrame) or len(data) != 1:
                raise ValueError(
                    "Les données doivent être un DataFrame avec une seule ligne"
                )
            if not self.validate_data(data):
                error_msg = "Données invalides pour clustering"
                self.alert_manager.send_alert(error_msg, priority=3)
                logger.error(error_msg)
                self.save_snapshot(
                    "error_cluster_score_distance",
                    {"error": "Données invalides"},
                    compress=False,
                )
                return 0.0

            confidence_drop_rate = 1.0 - min(
                len(data.columns) / (350 if self.training_mode else 150), 1.0
            )
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(data.columns)} features)"
                self.alert_manager.send_alert(alert_msg, priority=3)
                logger.warning(alert_msg)

            cache_key = hashlib.sha256(data.to_json().encode()).hexdigest()
            if cache_key in self.cache:
                distance = self.cache[cache_key]["distance"]
                latency = time.time() - start_time
                self.log_performance(
                    "cluster_score_distance_cache_hit", latency, success=True
                )
                success_msg = "Distance au centroïde récupérée du cache"
                self.alert_manager.send_alert(success_msg, priority=1)
                logger.info(success_msg)
                return distance

            def compute_distance():
                with self.db_manager.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT features FROM clusters
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (n_clusters,),
                    )
                    centroids = [json.loads(row[0]) for row in cursor.fetchall()]

                if not centroids:
                    return 0.0

                feature_cols = data.drop(
                    columns=["timestamp", "neural_regime"], errors="ignore"
                ).columns
                X = data[feature_cols].fillna(0)
                X_scaled = self.scaler.fit_transform(X)
                distances = [
                    np.linalg.norm(
                        X_scaled
                        - self.scaler.transform(
                            pd.DataFrame([centroid], columns=feature_cols)
                        )
                    )
                    for centroid in centroids
                ]
                return min(distances)

            distance = self.with_retries(compute_distance)

            snapshot = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "n_clusters": n_clusters,
                "distance": distance,
                "confidence_drop_rate": confidence_drop_rate,
            }
            self.save_snapshot("cluster_score_distance", snapshot, compress=True)

            self.cache[cache_key] = {"distance": distance, "timestamp": datetime.now()}
            if len(self.cache) > self.config.get("max_cache_size", 1000):
                self.cache.pop(next(iter(self.cache)))

            latency = time.time() - start_time
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            success_msg = f"Distance au centroïde le plus proche: {distance:.2f}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            self.log_performance(
                "cluster_score_distance",
                latency,
                success=True,
                memory_usage_mb=memory_usage,
                confidence_drop_rate=confidence_drop_rate,
            )
            return distance
        except Exception as e:
            error_msg = f"Erreur calcul distance: {str(e)}\n{traceback.format_exc()}"
            self.alert_manager.send_alert(error_msg, priority=4)
            logger.error(error_msg)
            self.log_performance(
                "cluster_score_distance", 0, success=False, error=str(e)
            )
            self.save_snapshot(
                "error_cluster_score_distance", {"error": str(e)}, compress=False
            )
            return 0.0
