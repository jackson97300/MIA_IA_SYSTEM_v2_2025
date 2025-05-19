# src/data/data_lake.py
# Gestion du data lake S3 pour MIA_IA_SYSTEM_v2_2025
#
# Version: 2.2.2
# Date: 2025-05-14
#
# Rôle: Stocke et récupère les données dans un data lake S3 structuré (raw/processed/presentation) avec chiffrement AES256.
# Utilisé par: feature_pipeline.py, trade_probability.py, validate_data.py.
#
# Notes:
# - Conforme à structure.txt (version 2.2.2, 2025-05-14).
# - Supporte les suggestions 1 (position sizing dynamique), 2 (coûts de transaction), 3 (microstructure),
#   4 (HMM/changepoint), 9 (surface de volatilité), 10 (ensembles de politiques).
# - Journalise les performances avec psutil dans data/logs/data_lake_performance.csv.
# - Lit les configurations depuis config/es_config.yaml (s3_bucket, s3_encryption).
# - Utilise src/utils/secret_manager.py pour les clés AWS.
# - Mise à jour pour inclure le stockage explicite des nouvelles features (atr_dynamic, orderflow_imbalance,
#   slippage_estimate, bid_ask_imbalance, trade_aggressiveness, regime_hmm, iv_skew, iv_term_structure)
#   et renforcer la gestion des erreurs et alertes.

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import boto3
import pandas as pd
import psutil
from botocore.exceptions import ClientError
from prometheus_client import Counter

from src.data.validate_data import load_feature_expectations
from src.model.utils.alert_manager import AlertManager
from src.utils.error_tracker import capture_error
from src.utils.secret_manager import get_secret

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOG_DIR = BASE_DIR / "data" / "logs"
LOG_DIR.mkdir(exist_ok=True)
PERF_LOG = LOG_DIR / "data_lake_performance.csv"

logging.basicConfig(
    filename=LOG_DIR / "data_lake.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)

# Compteur Prometheus
data_lake_stores = Counter(
    "data_lake_stores_total",
    "Nombre total d’opérations de stockage dans le data lake",
    ["market", "layer"],
)


class DataLake:
    def __init__(self, bucket: str, encryption: str = "AES256", market: str = "ES"):
        """Initialise le client S3 avec les identifiants AWS et vérifie le bucket."""
        try:
            self.s3 = boto3.client("s3", **get_secret("aws_credentials"))
            self.bucket = bucket
            self.encryption = encryption
            self.market = market
            self.alert_manager = AlertManager()
            self.s3.head_bucket(Bucket=bucket)
            logging.info(
                f"Client S3 initialisé pour {bucket} avec chiffrement {encryption}"
            )
            self.alert_manager.send_alert(
                f"Client S3 initialisé pour {bucket}", priority=2
            )
        except ClientError as e:
            error_code = e.response["Error"].get("Code", "Unknown")
            error_msg = f"Erreur S3 ({error_code}): {e}"
            logging.error(error_msg)
            capture_error(
                e, context={"market": market}, market=market, operation="init_data_lake"
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            raise ValueError(f"Échec de l’initialisation S3 ({error_code}): {e}")

    def store(
        self, data: pd.DataFrame, layer: str, key: str, market: Optional[str] = None
    ) -> None:
        """Stocke un DataFrame dans S3 avec chiffrement."""
        mkt = market or self.market
        if layer not in ["raw", "processed", "presentation"]:
            error_msg = (
                f"Couche invalide: {layer} (doit être raw, processed ou presentation)"
            )
            logging.error(error_msg)
            raise ValueError(error_msg)

        start_time = datetime.now()
        try:
            s3_key = f"{layer}/{mkt}/{key}"
            self.s3.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=data.to_csv(index=False, encoding="utf-8"),
                ServerSideEncryption=self.encryption,
            )
            data_lake_stores.labels(market=mkt, layer=layer).inc()
            latency = (datetime.now() - start_time).total_seconds()
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            pd.DataFrame(
                [
                    {
                        "timestamp": datetime.now().isoformat(),
                        "market": mkt,
                        "category": "store",
                        "cpu": psutil.cpu_percent(),
                        "memory_mb": memory_mb,
                        "latency_s": latency,
                    }
                ]
            ).to_csv(PERF_LOG, mode="a", index=False, header=not PERF_LOG.exists())
            logging.info(f"Stockage réussi: {s3_key}")
            self.alert_manager.send_alert(f"Données stockées dans {s3_key}", priority=2)
        except ClientError as e:
            error_code = e.response["Error"].get("Code", "Unknown")
            error_msg = f"Erreur stockage S3 ({error_code}): {e}"
            logging.error(error_msg)
            capture_error(e, context={"market": mkt}, market=mkt, operation="store")
            self.alert_manager.send_alert(error_msg, priority=4)
            raise ValueError(f"Échec du stockage dans S3 ({error_code}): {e}")
        except Exception as e:
            error_msg = f"Erreur stockage: {e}"
            logging.error(error_msg)
            capture_error(e, context={"market": mkt}, market=mkt, operation="store")
            self.alert_manager.send_alert(error_msg, priority=4)
            raise

    def retrieve(
        self, layer: str, key: str, market: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Récupère un DataFrame depuis S3."""
        mkt = market or self.market
        start_time = datetime.now()
        try:
            s3_key = f"{layer}/{mkt}/{key}"
            response = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
            df = pd.read_csv(response["Body"])
            latency = (datetime.now() - start_time).total_seconds()
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            pd.DataFrame(
                [
                    {
                        "timestamp": datetime.now().isoformat(),
                        "market": mkt,
                        "category": "retrieve",
                        "cpu": psutil.cpu_percent(),
                        "memory_mb": memory_mb,
                        "latency_s": latency,
                    }
                ]
            ).to_csv(PERF_LOG, mode="a", index=False, header=not PERF_LOG.exists())
            logging.info(f"Récupération réussie: {s3_key}")
            self.alert_manager.send_alert(
                f"Données récupérées depuis {s3_key}", priority=2
            )
            return df
        except ClientError as e:
            error_code = e.response["Error"].get("Code", "Unknown")
            error_msg = f"Erreur récupération S3 ({error_code}): {e}"
            logging.warning(error_msg)
            capture_error(e, context={"market": mkt}, market=mkt, operation="retrieve")
            self.alert_manager.send_alert(error_msg, priority=4)
            return None
        except Exception as e:
            error_msg = f"Erreur récupération: {e}"
            logging.error(error_msg)
            capture_error(e, context={"market": mkt}, market=mkt, operation="retrieve")
            self.alert_manager.send_alert(error_msg, priority=4)
            return None

    def store_features(self, data: pd.DataFrame):
        """Stocke les features dans S3 sous processed/features/."""
        start_time = datetime.now()
        try:
            # Liste des features à stocker, incluant les nouvelles
            required_features = [
                "atr_dynamic",
                "orderflow_imbalance",
                "slippage_estimate",
                "bid_ask_imbalance",
                "trade_aggressiveness",
                "regime_hmm",
                "iv_skew",
                "iv_term_structure",
            ]
            # Inclure les features existantes définies dans feature_sets.yaml
            expected_features = list(load_feature_expectations().keys())
            features = list(set(required_features + expected_features))

            # Vérifier les features manquantes
            missing = [f for f in features if f not in data.columns]
            if missing:
                error_msg = f"Features manquantes: {missing}"
                logging.error(error_msg)
                raise ValueError(error_msg)

            df = data[features].copy()
            ts = start_time.strftime("%Y%m%d_%H%M%S")
            key = f"features_{ts}.csv"
            self.store(df, layer="processed", key=key)

            latency = (datetime.now() - start_time).total_seconds()
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            pd.DataFrame(
                [
                    {
                        "timestamp": datetime.now().isoformat(),
                        "market": self.market,
                        "category": "features",
                        "cpu": psutil.cpu_percent(),
                        "memory_mb": memory_mb,
                        "latency_s": latency,
                    }
                ]
            ).to_csv(PERF_LOG, mode="a", index=False, header=not PERF_LOG.exists())
            logging.info(f"Features stockées pour {self.market}")
            self.alert_manager.send_alert(
                f"Features stockées pour {self.market}", priority=2
            )
        except Exception as e:
            error_msg = f"Erreur stockage features: {e}"
            logging.error(error_msg)
            capture_error(
                e,
                context={"market": self.market},
                market=self.market,
                operation="store_features",
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            raise

    def store_model_results(self, model_results: Dict):
        """Stocke les résultats des modèles dans S3 sous processed/models/."""
        start_time = datetime.now()
        try:
            ts = start_time.strftime("%Y%m%d_%H%M%S")
            key = f"model_results_{ts}.json"
            self.s3.put_object(
                Bucket=self.bucket,
                Key=f"processed/{self.market}/{key}",
                Body=json.dumps(model_results),
                ServerSideEncryption=self.encryption,
            )
            data_lake_stores.labels(market=self.market, layer="processed").inc()
            latency = (datetime.now() - start_time).total_seconds()
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            pd.DataFrame(
                [
                    {
                        "timestamp": datetime.now().isoformat(),
                        "market": self.market,
                        "category": "models",
                        "cpu": psutil.cpu_percent(),
                        "memory_mb": memory_mb,
                        "latency_s": latency,
                    }
                ]
            ).to_csv(PERF_LOG, mode="a", index=False, header=not PERF_LOG.exists())
            logging.info(f"Résultats modèles stockés pour {self.market}")
            self.alert_manager.send_alert(
                f"Résultats modèles stockés pour {self.market}", priority=2
            )
        except Exception as e:
            error_msg = f"Erreur stockage résultats modèles: {e}"
            logging.error(error_msg)
            capture_error(
                e,
                context={"market": self.market},
                market=self.market,
                operation="store_model_results",
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            raise

    def retrieve_shap_fallback(self, key: str) -> Optional[pd.DataFrame]:
        """Récupère les données SHAP de fallback depuis S3."""
        start_time = datetime.now()
        try:
            response = self.s3.get_object(
                Bucket=self.bucket, Key=f"fallback/shap/{key}"
            )
            df = pd.read_csv(response["Body"])
            latency = (datetime.now() - start_time).total_seconds()
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            pd.DataFrame(
                [
                    {
                        "timestamp": datetime.now().isoformat(),
                        "market": self.market,
                        "category": "shap_fallback",
                        "cpu": psutil.cpu_percent(),
                        "memory_mb": memory_mb,
                        "latency_s": latency,
                    }
                ]
            ).to_csv(PERF_LOG, mode="a", index=False, header=not PERF_LOG.exists())
            logging.info(f"SHAP fallback récupéré: {key}")
            self.alert_manager.send_alert(f"SHAP fallback récupéré: {key}", priority=2)
            return df
        except ClientError as e:
            error_code = e.response["Error"].get("Code", "Unknown")
            error_msg = f"Erreur SHAP fallback S3 ({error_code}): {e}"
            logging.warning(error_msg)
            capture_error(
                e,
                context={"market": self.market},
                market=self.market,
                operation="retrieve_shap_fallback",
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            return None
        except Exception as e:
            error_msg = f"Erreur récupération SHAP fallback: {e}"
            logging.error(error_msg)
            capture_error(
                e,
                context={"market": self.market},
                market=self.market,
                operation="retrieve_shap_fallback",
            )
            self.alert_manager.send_alert(error_msg, priority=4)
            return None
