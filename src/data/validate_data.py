# src/data/validate_data.py
# Validation des features avec Great Expectations pour MIA_IA_SYSTEM_v2_2025
#
# Version: 2.1.5
# Date: 2025-05-13
#
# Rôle: Valide les schémas et plages des features avant leur utilisation dans le pipeline de trading.
# Utilisé par: feature_pipeline.py, mia_switcher.py.
#
# Notes:
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Supporte les suggestions 1 (position sizing dynamique), 2 (coûts de transaction), 3 (microstructure),
#   4 (HMM/changepoint), 9 (surface de volatilité).
# - Lit les attentes depuis config/feature_sets.yaml (expected_range).
# - Journalise les erreurs dans data/logs/validation_errors.csv.
# - Journalise les performances avec psutil dans data/logs/validate_data_performance.csv.
# - Pas de références à dxFeed, obs_t, 320 features, ou 81 features.
# - Policies Note: The official directory for routing policies is src/model/router/policies.
# The src/model/policies directory is a residual and should be verified
# for removal to avoid import conflicts.

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Union

import pandas as pd
import psutil
import yaml
from great_expectations.dataset import PandasDataset
from prometheus_client import Counter

from src.model.utils.alert_manager import AlertManager
from src.utils.error_tracker import capture_error

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOG_DIR = BASE_DIR / "data" / "logs"
LOG_DIR.mkdir(exist_ok=True)
FEATURE_LOG = LOG_DIR / "validate_data_performance.csv"
FEATURE_AUDIT = BASE_DIR / "data" / "features" / "features_audit_final.csv"
CONFIG_PATH = BASE_DIR / "config" / "feature_sets.yaml"
MODEL_CONFIG_PATH = BASE_DIR / "config" / "model_params.yaml"

logging.basicConfig(
    filename=LOG_DIR / "validate_data.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)

validation_errors = Counter(
    "validation_errors_total",
    "Nombre total d’erreurs de validation des features",
    ["market"],
)

alert_manager = AlertManager()


def load_feature_expectations() -> Dict[str, Dict]:
    try:
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        if not isinstance(config.get("feature_sets", {}), dict):
            raise ValueError("Format invalide dans feature_sets.yaml")

        expectations = {}
        sections = ["features"]
        if "range" in config.get("feature_sets", {}):
            sections.append("range/features")

        for section in sections:
            parts = section.split("/")
            features = config["feature_sets"]
            for p in parts:
                features = features.get(p, [])
            for feature in features:
                name = feature["name"]
                if "expected_range" in feature:
                    expectations[name] = {
                        "min": feature["expected_range"][0],
                        "max": feature["expected_range"][1],
                    }
        return expectations
    except Exception as e:
        logging.error(f"Erreur chargement feature_sets.yaml: {e}")
        raise ValueError(f"Échec chargement des attentes: {e}")


def load_hmm_components() -> int:
    try:
        with open(MODEL_CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        return config.get("hmm", {}).get("n_components", 3)
    except Exception as e:
        logging.warning(f"Fallback n_components = 3: {e}")
        alert_manager.send_alert(
            f"Impossible de charger n_components, fallback = 3: {e}", priority=3
        )
        return 3


def validate_features(
    data: pd.DataFrame, market: str = "ES", return_dict: bool = False
) -> Union[bool, Dict]:
    start_time = datetime.now()
    try:
        dataset = PandasDataset(data)
        expectations = load_feature_expectations()

        dataset.expect_table_columns_to_match_set(
            column_set=list(expectations.keys()), exact_match=False
        )
        dataset.expect_table_row_count_to_be_between(min_value=1, max_value=10000)

        for feature, bounds in expectations.items():
            if feature in data.columns:
                dataset.expect_column_values_to_be_between(
                    column=feature,
                    min_value=bounds["min"],
                    max_value=bounds["max"],
                    mostly=0.95,
                )
                dataset.expect_column_values_to_be_of_type(
                    column=feature, type_="float64"
                )

        rules = {
            "atr_dynamic": (0, 100),
            "orderflow_imbalance": (-1, 1),
            "slippage_estimate": (0, 1000),
            "bid_ask_imbalance": (-1, 1),
            "trade_aggressiveness": (0, 1),
            "iv_skew": (0, 1),
            "iv_term_structure": (-1, 1),
        }
        for f, (mi, ma) in rules.items():
            if f in data.columns:
                dataset.expect_column_values_to_be_between(
                    column=f, min_value=mi, max_value=ma, mostly=0.95
                )

        n_components = load_hmm_components()
        if "regime_hmm" in data.columns:
            dataset.expect_column_values_to_be_in_set(
                "regime_hmm", value_set=list(range(n_components))
            )

        result = dataset.validate()
        latency = (datetime.now() - start_time).total_seconds()
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024

        perf_entry = pd.DataFrame(
            [
                {
                    "timestamp": datetime.now().isoformat(),
                    "market": market,
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_usage_mb": memory_usage,
                    "latency_s": latency,
                }
            ]
        )
        if FEATURE_LOG.exists():
            perf_entry.to_csv(FEATURE_LOG, mode="a", index=False, header=False)
        else:
            perf_entry.to_csv(FEATURE_LOG, index=False)

        if not result["success"]:
            validation_errors.labels(market=market).inc()
            errors = [
                r["expectation_config"]["kwargs"].get("column", "unknown")
                + ": "
                + str(r["exception_info"]["exception_message"])
                for r in result["results"]
                if not r["success"]
            ]
            logging.error(f"Validation échouée pour {market}: {errors}")
            pd.DataFrame(
                [
                    {
                        "timestamp": datetime.now().isoformat(),
                        "market": market,
                        "errors": str(errors),
                    }
                ]
            ).to_csv(
                LOG_DIR / "validation_errors.csv", mode="a", index=False, header=False
            )
            alert_manager.send_alert(
                f"Échec de validation des features pour {market}", priority=3
            )
            return (
                {"success": False, "valid": False, "errors": errors}
                if return_dict
                else False
            )

        pd.DataFrame(result["results"]).to_csv(
            FEATURE_AUDIT, index=False, encoding="utf-8"
        )
        alert_manager.send_alert(f"Validation réussie pour {market}", priority=2)
        logging.info(f"Validation réussie pour {market}")
        return {"success": True, "valid": True, "errors": []} if return_dict else True

    except Exception as e:
        logging.error(f"Erreur validate_features: {e}")
        capture_error(
            e, context={"market": market}, market=market, operation="validate_features"
        )
        alert_manager.send_alert(
            f"Erreur critique validation features: {e}", priority=4
        )
        validation_errors.labels(market=market).inc()
        return (
            {"success": False, "valid": False, "errors": [str(e)]}
            if return_dict
            else False
        )
