# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/model/utils/config_manager.py
# Rôle : Charge, valide et distribue les configurations YAML pour tous les modules, 
# standardisant sur 350 features, avec support multi-environnements, retries, snapshots 
# JSON, et encapsulation via ConfigContext.
#
# Version : 2.1.5
# Date : 2025-05-14
#
# Dépendances :
# - pyyaml>=6.0.0,<7.0.0, psutil>=5.9.8,<6.0.0, boto3>=1.26.0,<2.0.0, 
#   loguru>=0.7.0,<1.0.0
# - pydantic>=2.0.0,<3.0.0
# - src/model/utils/alert_manager.py
# - src/utils/telegram_alert.py
#
# Inputs :
# - Fichiers YAML dans config/ ou config/envs/{env}/ (ex. : feature_sets.yaml, 
#   credentials.yaml)
#
# Outputs :
# - Configurations validées via get_config(), get_features(), snapshots JSON compressés 
#   dans data/cache/config/<market>/*.json.gz
# - Sauvegardes incrémentielles dans data/checkpoints/config/<market>/*.json.gz
# - Logs dans data/logs/config_errors.log
# - Logs de performance dans data/logs/config_performance.csv
#
# Lien avec SAC :
# - Fournit les 350 features validées pour l'entraînement SAC dans train_sac.py
#
# Notes :
# - Utilisation exclusive d’IQFeed, suppression des références à dxFeed.
# - Support de trade_probability_rl.yaml pour TradeProbabilityPredictor (corrigé de 
#   trade_probability_rl.py).
# - Ajout de validations pour les nouvelles features (bid_ask_imbalance, 
#   trade_aggressiveness, iv_skew, iv_term_structure, option_skew, news_impact_score).
# - Intégration de validate_obs_t pour standardiser les colonnes critiques.
# - Encapsulation des configurations dans ConfigContext pour réduire le couplage 
#   implicite.
# - Conforme à la Phase 1 (collecte via IQFeed), Phase 8 (auto-conscience via 
#   validations cognitives), et Phase 16 (ensemble et transfer learning via validations 
#   avancées).
# - Tests unitaires renforcés dans tests/test_config_manager.py pour couvrir les 
#   nouvelles validations.
# - Supporte multi-marchés (ES, MNQ) avec chemins spécifiques.

# Note : L'erreur F824 concernant `global RUNNING` est une fausse alerte. 
# Aucune déclaration ou utilisation de `RUNNING` dans ce fichier.

import gzip
import json
import os
import time
import traceback
from collections import OrderedDict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import pandas as pd
import psutil
import yaml
from loguru import logger
from pydantic import (
    BaseModel,
    PositiveFloat,
    PositiveInt,
    ValidationError,
    conint,
    constr,
)

from src.model.utils.alert_manager import AlertManager
from src.utils.telegram_alert import send_telegram_alert

# Configuration logging
logger.remove()
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
BASE_DIR = Path(os.path.dirname(parent_dir))
LOG_DIR = BASE_DIR / "data" / "logs"
CACHE_DIR = BASE_DIR / "data" / "cache" / "config"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints" / "config"
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "config_errors.log", 
    rotation="10 MB", 
    level="INFO", 
    encoding="utf-8"
)
logger.add(lambda msg: print(msg, end=""), level="INFO")

# Constantes
PERF_LOG_PATH = LOG_DIR / "config_performance.csv"
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # Secondes, exponentiel
MAX_CACHE_SIZE = 1000
CACHE_EXPIRATION = 24 * 3600  # 24 heures en secondes
MAX_CACHE_SIZE_MB = 1000  # Taille max du cache en MB
CRITICAL_FEATURES = [
    "rsi_14",
    "ofi_score",
    "vix_es_correlation",
    "atr_dynamic",
    "orderflow_imbalance",
    "bid_ask_imbalance",
    "trade_aggressiveness",
    "iv_skew",
    "iv_term_structure",
    "option_skew",
    "news_impact_score",
    "spoofing_score",
    "volume_anomaly",
]

# Cache global pour les résultats de validation
config_cache = OrderedDict()


class RiskManagerConfig(BaseModel):
    """Modèle Pydantic pour valider risk_manager_config.yaml."""

    buffer_size: conint(ge=10, le=1000)
    critical_buffer_size: conint(ge=5, le=50)
    max_retries: PositiveInt
    retry_delay: PositiveFloat
    kelly_fraction: PositiveFloat
    max_position_fraction: PositiveFloat
    atr_threshold: PositiveFloat
    orderflow_imbalance_limit: PositiveFloat


class RegimeConfig(BaseModel):
    """Modèle Pydantic pour valider regime_detector_config.yaml."""

    buffer_size: conint(ge=10, le=1000)
    critical_buffer_size: conint(ge=5, le=50)
    max_retries: PositiveInt
    retry_delay: PositiveFloat
    n_iterations: conint(ge=5, le=50)
    convergence_threshold: PositiveFloat
    covariance_type: constr(pattern="^(full|diag|tied|spherical)$")
    n_components: conint(ge=2, le=5)
    window_size: conint(ge=10, le=500)
    window_size_adaptive: bool
    random_state: PositiveInt
    use_random_state: bool
    min_train_rows: conint(ge=50)
    min_state_duration: conint(ge=1, le=20)
    cache_ttl_seconds: conint(ge=10, le=3600)
    cache_ttl_adaptive: bool
    prometheus_labels: Dict[str, str]


class TradeProbabilityRLConfig(BaseModel):
    """Modèle Pydantic pour valider trade_probability_rl.yaml."""

    cvar_alpha: PositiveFloat
    quantiles: conint(ge=10, le=100)
    ensemble_weights: List[PositiveFloat]
    confidence_threshold: PositiveFloat
    retrain_interval: constr(pattern="^(daily|weekly|biweekly)$")


class ConfigContext:
    """Encapsule les configurations pour réduire le couplage implicite avec les 
    fichiers YAML."""

    def __init__(self, env: str = "prod", fallback: bool = False):
        self.env = env
        self.fallback = fallback
        self.configs: Dict[str, Any] = {}
        self.alert_manager = AlertManager()
        self.load_all_configs()

    def load_all_configs(self):
        """Charge et valide tous les fichiers de configuration."""
        start_time = time.time()
        for config_file in ConfigManager.CONFIG_FILES:
            config_path = Path(
                f"config/envs/{self.env}/{config_file}"
                if self.env
                else f"config/{config_file}"
            )
            try:
                config = self._load_config_with_retries(config_path)
                self._validate_config(config_file, config, market="ES")
                self.configs[config_file] = config
                logger.info(f"Chargé et validé : {config_file}")
                self.save_snapshot(
                    "config_load",
                    {"file": config_file, "status": "success"},
                    market="ES",
                )
            except Exception as e:
                error_snapshot = {
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "file": str(config_path),
                    "error": str(e).replace(
                        self.configs.get("credentials.yaml", {})
                        .get("iqfeed", {})
                        .get("api_key", ""),
                        "***",
                    ),
                }
                self.save_snapshot("config_error", error_snapshot, market="ES")
                error_msg = (
                    f"Erreur dans {config_file} : {str(e)}\n"
                    f"{traceback.format_exc()}"
                )
                logger.error(error_msg)
                self.alert_manager.send_alert(
                    f"Erreur de config : {config_file} - {str(e)}",
                    channel="telegram",
                    priority=config.get("metadata", {}).get("alert_priority", 4),
                )
                if self.fallback:
                    logger.warning(f"Fallback activé, poursuite sans {config_file}")
                    continue
                raise
            finally:
                self.log_performance(
                    "load_config",
                    time.time() - start_time,
                    success=True,
                    config_file=config_file,
                    market="ES",
                )

    def _load_config_with_retries(self, config_path: Path) -> Dict[str, Any]:
        """Charge un fichier YAML avec retries."""
        for attempt in range(MAX_RETRIES):
            try:
                return self._load_config(str(config_path))
            except (FileNotFoundError, yaml.YAMLError):
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(RETRY_DELAY * (2**attempt))
                logger.warning(
                    f"Échec chargement {config_path}, tentative "
                    f"{attempt + 1}/{MAX_RETRIES}"
                )
        return {}

    @lru_cache(maxsize=10)
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Charge un fichier YAML avec cache."""
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Fichier introuvable : {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def reload_config(self, config_file: str, market: str = "ES"):
        """Recharge un fichier de configuration spécifique, invalidant le cache."""
        if config_file not in ConfigManager.CONFIG_FILES:
            raise ValueError(f"Fichier inconnu : {config_file}")
        config_path = Path(
            f"config/envs/{self.env}/{config_file}"
            if self.env
            else f"config/{config_file}"
        )
        try:
            self._load_config.cache_clear()
            config = self._load_config_with_retries(config_path)
            self._validate_config(config_file, config, market=market)
            self.configs[config_file] = config
            logger.info(f"Rechargé et validé : {config_file}")
            self.save_snapshot(
                "config_reload",
                {"file": config_file, "status": "success"},
                market=market,
            )
        except Exception as e:
            error_msg = (
                f"Échec rechargement {config_file} pour {market} : {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, channel="telegram", priority=4)
            self.save_snapshot(
                "config_reload_error",
                {"file": config_file, "error": str(e)},
                market=market,
            )
            raise

    def _validate_config(
        self, config_file: str, config: Dict[str, Any], market: str = "ES"
    ):
        """Valide un fichier de configuration selon des règles spécifiques."""
        cache_key = f"{market}_{config_file}_{hash(str(config))}"
        if cache_key in config_cache:
            return
        while len(config_cache) > MAX_CACHE_SIZE:
            config_cache.popitem(last=False)

        start_time = time.time()
        if not config:
            error_msg = f"{config_file} est vide ou mal formé pour {market}"
            raise ValueError(error_msg)

        # Validation générale
        if "metadata" not in config or "version" not in config["metadata"]:
            error_msg = f"{config_file} : metadata.version manquant pour {market}"
            raise ValueError(error_msg)
        if "alert_priority" not in config.get("metadata", {}):
            config["metadata"]["alert_priority"] = 4  # Par défaut

        # Validation des chemins
        for key in [
            "input_path",
            "output_path",
            "log_dir",
            "data.raw",
            "data.features",
            "data.processed",
        ]:
            path = config
            for part in key.split("."):
                path = (
                    path.get(part, {}).get("value", "")
                    if isinstance(path, dict)
                    else ""
                )
            if path and not Path(path).parent.exists():
                Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Validation spécifique pour feature_sets.yaml
        if config_file == "feature_sets.yaml":
            total_features = config.get("metadata", {}).get("total_features", 0)
            feature_count = sum(
                len(category.get("features", []))
                for category in config.get("feature_sets", {}).values()
            )
            null_count = sum(
                len([f for f in category.get("features", []) if f.get("name") is None])
                for category in config.get("feature_sets", {}).values()
            )
            confidence_drop_rate = (
                null_count / feature_count if feature_count > 0 else 0.0
            )
            if confidence_drop_rate > 0.5:
                alert_msg = (
                    f"Confidence_drop_rate élevé pour {market} dans {config_file}: "
                    f"{confidence_drop_rate:.2f} ({null_count} features nulles)"
                )
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
            if total_features != 350 or total_features != feature_count:
                error_msg = (
                    f"{config_file} : total_features={total_features}, mais "
                    f"{feature_count} features listées, attendu 350 pour {market}"
                )
                raise ValueError(error_msg)
            shap_features = config.get("fallback_features", {}).get("features", [])
            if len(shap_features) != 150:
                error_msg = (
                    f"{config_file} : fallback_features doit contenir 150 features, "
                    f"trouvé {len(shap_features)} pour {market}"
                )
                raise ValueError(error_msg)
            # Validation des nouvelles features
            training_features = config.get("ES", {}).get("training", [])
            inference_features = config.get("ES", {}).get("inference", [])
            missing_critical = [
                f for f in CRITICAL_FEATURES if f not in training_features
            ]
            if missing_critical:
                error_msg = (
                    f"{config_file} : Features critiques manquantes dans training "
                    f"pour {market}: {missing_critical}"
                )
                raise ValueError(error_msg)
            missing_shap = [f for f in CRITICAL_FEATURES if f not in inference_features]
            if missing_shap:
                alert_msg = (
                    f"{config_file} : Features critiques manquantes dans inference "
                    f"pour {market}: {missing_shap}"
                )
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)

        # Validation spécifique pour risk_manager_config.yaml
        elif config_file == "risk_manager_config.yaml":
            try:
                RiskManagerConfig(**config)
            except ValidationError as e:
                error_msg = (
                    f"{config_file} : Validation Pydantic échouée pour {market}: "
                    f"{str(e)}"
                )
                raise ValueError(error_msg)
            if not (0.01 <= config.get("kelly_fraction", 0) <= 1.0):
                error_msg = (
                    f"{config_file} : kelly_fraction hors plage [0.01, 1.0] "
                    f"pour {market}"
                )
                raise ValueError(error_msg)
            if not (0.01 <= config.get("max_position_fraction", 0) <= 0.5):
                error_msg = (
                    f"{config_file} : max_position_fraction hors plage [0.01, 0.5] "
                    f"pour {market}"
                )
                raise ValueError(error_msg)
            if not (50.0 <= config.get("atr_threshold", 0) <= 500.0):
                error_msg = (
                    f"{config_file} : atr_threshold hors plage [50.0, 500.0] "
                    f"pour {market}"
                )
                raise ValueError(error_msg)
            if not (0.5 <= config.get("orderflow_imbalance_limit", 0) <= 1.0):
                error_msg = (
                    f"{config_file} : orderflow_imbalance_limit hors plage [0.5, 1.0] "
                    f"pour {market}"
                )
                raise ValueError(error_msg)

        # Validation spécifique pour regime_detector_config.yaml
        elif config_file == "regime_detector_config.yaml":
            try:
                RegimeConfig(**config)
            except ValidationError as e:
                error_msg = (
                    f"{config_file} : Validation Pydantic échouée pour {market}: "
                    f"{str(e)}"
                )
                raise ValueError(error_msg)
            if not (50 <= config.get("min_train_rows", 0) <= 1000):
                error_msg = (
                    f"{config_file} : min_train_rows hors plage [50, 1000] "
                    f"pour {market}"
                )
                raise ValueError(error_msg)
            if not (1 <= config.get("min_state_duration", 0) <= 20):
                error_msg = (
                    f"{config_file} : min_state_duration hors plage [1, 20] "
                    f"pour {market}"
                )
                raise ValueError(error_msg)
            if not (10 <= config.get("cache_ttl_seconds", 0) <= 3600):
                error_msg = (
                    f"{config_file} : cache_ttl_seconds hors plage [10, 3600] "
                    f"pour {market}"
                )
                raise ValueError(error_msg)

        # Validation spécifique pour trade_probability_rl.yaml
        elif config_file == "trade_probability_rl.yaml":
            try:
                TradeProbabilityRLConfig(**config)
            except ValidationError as e:
                error_msg = (
                    f"{config_file} : Validation Pydantic échouée pour {market}: "
                    f"{str(e)}"
                )
                raise ValueError(error_msg)
            if not (0.9 <= config.get("cvar_alpha", 0) <= 1.0):
                error_msg = (
                    f"{config_file} : cvar_alpha hors plage [0.9, 1.0] "
                    f"pour {market}"
                )
                raise ValueError(error_msg)
            if sum(config.get("ensemble_weights", [])) != 1.0:
                error_msg = (
                    f"{config_file} : ensemble_weights doit sommer à 1.0 "
                    f"pour {market}"
                )
                raise ValueError(error_msg)
            if not (0.5 <= config.get("confidence_threshold", 0) <= 1.0):
                error_msg = (
                    f"{config_file} : confidence_threshold hors plage [0.5, 1.0] "
                    f"pour {market}"
                )
                raise ValueError(error_msg)

        # Validations spécifiques pour chaque fichier (inchangées)
        elif config_file == "es_config.yaml":
            valid_symbols = ["ES", "MNQ"] if market in ["ES", "MNQ"] else ["ES"]
            if (
                config.get("market", {}).get("symbol", {}).get("value")
                not in valid_symbols
            ):
                error_msg = (
                    f"{config_file} : market.symbol.value doit être dans "
                    f"{valid_symbols} pour {market}"
                )
                raise ValueError(error_msg)
            if not (1 <= config.get("preprocessing", {}).get("depth_level", 0) <= 10):
                error_msg = (
                    f"{config_file} : depth_level hors plage [1, 10] "
                    f"pour {market}"
                )
                raise ValueError(error_msg)
            if (
                config.get("spotgamma_recalculator", {}).get("min_volume_threshold", 0)
                <= 0
            ):
                error_msg = (
                    f"{config_file} : min_volume_threshold doit être > 0 "
                    f"pour {market}"
                )
                raise ValueError(error_msg)
            input_path = config.get("preprocessing", {}).get("input_path", {}).get(
                "value", ""
            )
            if input_path and not Path(input_path).parent.exists():
                error_msg = f"{config_file} : input_path invalide pour {market}"
                raise ValueError(error_msg)
            if (
                config.get("spotgamma_recalculator", {}).get("price_proximity_range", 0)
                <= 0
            ):
                error_msg = (
                    f"{config_file} : price_proximity_range doit être > 0 "
                    f"pour {market}"
                )
                raise ValueError(error_msg)
            shap_features = config.get("spotgamma_recalculator", {}).get(
                "shap_features", []
            )
            if shap_features != ["iv_atm", "option_skew"]:
                error_msg = (
                    f"{config_file} : shap_features doit être ['iv_atm', "
                    f"'option_skew'] pour {market}"
                )
                raise ValueError(error_msg)

        elif config_file == "trading_env_config.yaml":
            max_position_size = (
                config.get("environment", {})
                .get("max_position_size", {})
                .get("value", 0)
            )
            if not (1 <= max_position_size <= 10):
                error_msg = (
                    f"{config_file} : max_position_size hors plage [1, 10] "
                    f"pour {market}"
                )
                raise ValueError(error_msg)
            obs_dims = (
                config.get("observation", {})
                .get("obs_dimensions", {})
                .get("value", 0)
            )
            if obs_dims != 350:
                error_msg = (
                    f"{config_file} : obs_dimensions={obs_dims}, attendu 350 "
                    f"pour {market}"
                )
                raise ValueError(error_msg)
            reward_scaling = (
                config.get("reward", {}).get("reward_scaling", {}).get("value", 0)
            )
            if not (0.1 <= reward_scaling <= 10.0):
                error_msg = (
                    f"{config_file} : reward_scaling hors plage [0.1, 10.0] "
                    f"pour {market}"
                )
                raise ValueError(error_msg)

        elif config_file == "mia_config.yaml":
            language = config.get("mia", {}).get("language", {}).get("value")
            if language not in ["fr", "en", "es"]:
                error_msg = f"{config_file} : language non valide pour {market}"
                raise ValueError(error_msg)
            log_dir = config.get("logging", {}).get("log_dir", {}).get("value", "")
            if log_dir and not Path(log_dir).exists():
                Path(log_dir).mkdir(parents=True, exist_ok=True)
            max_message_length = (
                config.get("mia", {}).get("max_message_length", {}).get("value", 0)
            )
            if not (50 <= max_message_length <= 500):
                error_msg = (
                    f"{config_file} : max_message_length hors plage [50, 500] "
                    f"pour {market}"
                )
                raise ValueError(error_msg)
            interactive_mode = config.get("mia", {}).get("interactive_mode", {})
            if interactive_mode.get("enabled", False) and interactive_mode.get(
                "language"
            ) != language:
                error_msg = (
                    f"{config_file} : interactive_mode.language doit correspondre à "
                    f"mia.language pour {market}"
                )
                raise ValueError(error_msg)

        elif config_file == "credentials.yaml":
            for api in ["iqfeed", "investing_com", "newsdata_io", "nlp"]:
                api_config = config.get(api, {})
                if "api_key" in api_config:
                    api_key = api_config.get("api_key", "")
                    if api_key in ["to_be_defined", None, ""]:
                        error_msg = (
                            f"{config_file} : {api}.api_key non défini ou vide "
                            f"pour {market}"
                        )
                        raise ValueError(error_msg)
                elif api == "newsdata_io":
                    api_keys = api_config.get("api_keys", [])
                    if not api_keys:
                        error_msg = (
                            f"{config_file} : newsdata_io.api_keys vide "
                            f"pour {market}"
                        )
                        raise ValueError(error_msg)
                    if any(key in ["to_be_defined", None, ""] for key in api_keys):
                        error_msg = (
                            f"{config_file} : newsdata_io.api_keys contient une clé "
                            f"vide pour {market}"
                        )
                        raise ValueError(error_msg)
            if any("api_key" in str(v).lower() for v in config.values()):
                alert_msg = (
                    f"{config_file} : Clés API détectées, éviter logging en clair "
                    f"pour {market}"
                )
                logger.warning(alert_msg)

        elif config_file == "market_config.yaml":
            valid_symbols = ["ES", "MNQ"] if market in ["ES", "MNQ"] else ["ES"]
            symbol_value = config.get("market", {}).get("symbol", {}).get("value")
            if symbol_value not in valid_symbols:
                error_msg = (
                    f"{config_file} : market.symbol.value doit être dans "
                    f"{valid_symbols} pour {market}"
                )
                raise ValueError(error_msg)
            max_drawdown = config.get("risk", {}).get("max_drawdown", {}).get(
                "value", 0
            )
            if not (-1.0 <= max_drawdown <= 0.0):
                error_msg = (
                    f"{config_file} : max_drawdown hors plage [-1.0, 0.0] "
                    f"pour {market}"
                )
                raise ValueError(error_msg)
            default_rrr = config.get("risk", {}).get("default_rrr", {}).get("value", 0)
            if not (1.0 <= default_rrr <= 5.0):
                error_msg = (
                    f"{config_file} : default_rrr hors plage [1.0, 5.0] "
                    f"pour {market}"
                )
                raise ValueError(error_msg)

        elif config_file == "model_params.yaml":
            learning_rate = (
                config.get("default", {}).get("learning_rate", {}).get("value", 0)
            )
            if not (0.00001 <= learning_rate <= 0.01):
                error_msg = (
                    f"{config_file} : learning_rate hors plage [0.00001, 0.01] "
                    f"pour {market}"
                )
                raise ValueError(error_msg)
            base_features = (
                config.get("neural_pipeline", {})
                .get("base_features", {})
                .get("value", 0)
            )
            if base_features != 350:
                error_msg = (
                    f"{config_file} : base_features={base_features}, attendu 350 "
                    f"pour {market}"
                )
                raise ValueError(error_msg)
            buffer_size = (
                config.get("default", {}).get("buffer_size", {}).get("value", 0)
            )
            if not (10000 <= buffer_size <= 1000000):
                error_msg = (
                    f"{config_file} : buffer_size hors plage [10000, 1000000] "
                    f"pour {market}"
                )
                raise ValueError(error_msg)
            news_impact_threshold = (
                config.get("default", {})
                .get("news_impact_threshold", {})
                .get("value", 0)
            )
            if not (0.0 <= news_impact_threshold <= 1.0):
                error_msg = (
                    f"{config_file} : news_impact_threshold hors plage [0.0, 1.0] "
                    f"pour {market}"
                )
                raise ValueError(error_msg)
            vix_threshold = (
                config.get("default", {}).get("vix_threshold", {}).get("value", 0)
            )
            if not (10.0 <= vix_threshold <= 50.0):
                error_msg = (
                    f"{config_file} : vix_threshold hors plage [10.0, 50.0] "
                    f"pour {market}"
                )
                raise ValueError(error_msg)

        elif config_file == "router_config.yaml":
            atr_threshold = (
                config.get("trend", {}).get("atr_threshold", {}).get("value", 0)
            )
            if not (0.5 <= atr_threshold <= 5.0):
                error_msg = (
                    f"{config_file} : atr_threshold hors plage [0.5, 5.0] "
                    f"pour {market}"
                )
                raise ValueError(error_msg)
            vwap_slope_threshold = (
                config.get("range", {})
                .get("vwap_slope_threshold", {})
                .get("value", 0)
            )
            if not (0.0 <= vwap_slope_threshold <= 0.05):
                error_msg = (
                    f"{config_file} : vwap_slope_threshold hors plage [0.0, 0.05] "
                    f"pour {market}"
                )
                raise ValueError(error_msg)
            volatility_spike_threshold = (
                config.get("defensive", {})
                .get("volatility_spike_threshold", {})
                .get("value", 0)
            )
            if not (1.0 <= volatility_spike_threshold <= 5.0):
                error_msg = (
                    f"{config_file} : volatility_spike_threshold hors plage [1.0, 5.0] "
                    f"pour {market}"
                )
                raise ValueError(error_msg)

        elif config_file == "iqfeed_config.yaml":
            symbols = config.get("symbols", {}).get("value", [])
            if market.lower() not in symbols:
                error_msg = (
                    f"{config_file} : symbols doit inclure {market} pour {market}"
                )
                raise ValueError(error_msg)
            retry_attempts = (
                config.get("connection", {})
                .get("retry_attempts", {})
                .get("value", 0)
            )
            if not (1 <= retry_attempts <= 10):
                error_msg = (
                    f"{config_file} : retry_attempts hors plage [1, 10] "
                    f"pour {market}"
                )
                raise ValueError(error_msg)
            host = config.get("iqfeed", {}).get("host", {}).get("value", "")
            if host != "127.0.0.1":
                error_msg = (
                    f"{config_file} : iqfeed.host doit être '127.0.0.1' "
                    f"pour {market}"
                )
                raise ValueError(error_msg)
            port = config.get("iqfeed", {}).get("port", {}).get("value", 0)
            if not (9000 <= port <= 9200):
                error_msg = (
                    f"{config_file} : iqfeed.port hors plage [9000, 9200] "
                    f"pour {market}"
                )
                raise ValueError(error_msg)
            dom_depth = config.get("data_types", {}).get("dom", {}).get("depth", 0)
            if not (1 <= dom_depth <= 10):
                error_msg = (
                    f"{config_file} : dom.depth hors plage [1, 10] pour {market}"
                )
                raise ValueError(error_msg)

        elif config_file == "algo_config.yaml":
            for algo in ["sac", "ppo", "ddpg"]:
                for regime in ["range", "trend", "defensive"]:
                    params = config.get(algo, {}).get(regime, {})
                    learning_rate = params.get("learning_rate", {}).get("value", 0)
                    if not (0.00001 <= learning_rate <= 0.01):
                        error_msg = (
                            f"{config_file} : {algo}.{regime}.learning_rate hors "
                            f"plage [0.00001, 0.01] pour {market}"
                        )
                        raise ValueError(error_msg)
                    gamma = params.get("gamma", {}).get("value", 0)
                    if not (0.9 <= gamma <= 1.0):
                        error_msg = (
                            f"{config_file} : {algo}.{regime}.gamma hors plage "
                            f"[0.9, 1.0] pour {market}"
                        )
                        raise ValueError(error_msg)
                    ent_coef = params.get("ent_coef", {}).get("value", 0)
                    if not (0 <= ent_coef <= 0.5):
                        error_msg = (
                            f"{config_file} : {algo}.{regime}.ent_coef hors plage "
                            f"[0, 0.5] pour {market}"
                        )
                        raise ValueError(error_msg)
                    l2_lambda = params.get("l2_lambda", {}).get("value", 0)
                    if not (0 <= l2_lambda <= 0.1):
                        error_msg = (
                            f"{config_file} : {algo}.{regime}.l2_lambda hors plage "
                            f"[0, 0.1] pour {market}"
                        )
                        raise ValueError(error_msg)

        elif config_file == "alert_config.yaml":
            for channel in ["telegram", "sms", "email"]:
                channel_config = config.get(channel, {})
                if channel_config.get("enabled", False):
                    required_keys = (
                        ["bot_token", "chat_id"]
                        if channel == "telegram"
                        else (
                            ["account_sid", "auth_token", "from_number", "to_number"]
                            if channel == "sms"
                            else ["sender_email", "sender_password", "receiver_email"]
                        )
                    )
                    for key in required_keys:
                        if not channel_config.get(key, ""):
                            error_msg = (
                                f"{config_file} : {channel}.{key} non défini "
                                f"pour {market}"
                            )
                            raise ValueError(error_msg)
                priority = channel_config.get("priority", 0)
                if priority not in [1, 2, 3, 4]:
                    error_msg = (
                        f"{config_file} : {channel}.priority doit être dans "
                        f"[1, 2, 3, 4] pour {market}"
                    )
                    raise ValueError(error_msg)
            thresholds = config.get("alert_thresholds", {})
            volatility_spike_threshold = thresholds.get(
                "volatility_spike_threshold", {}
            ).get("value", 0)
            if not (1.0 <= volatility_spike_threshold <= 5.0):
                error_msg = (
                    f"{config_file} : volatility_spike_threshold hors plage "
                    f"[1.0, 5.0] pour {market}"
                )
                raise ValueError(error_msg)
            oi_sweep_alert_threshold = thresholds.get(
                "oi_sweep_alert_threshold", {}
            ).get("value", 0)
            if not (0.0 <= oi_sweep_alert_threshold <= 1.0):
                error_msg = (
                    f"{config_file} : oi_sweep_alert_threshold hors plage "
                    f"[0.0, 1.0] pour {market}"
                )
                raise ValueError(error_msg)
            macro_event_severity_alert = thresholds.get(
                "macro_event_severity_alert", {}
            ).get("value", 0)
            if not (0.0 <= macro_event_severity_alert <= 1.0):
                error_msg = (
                    f"{config_file} : macro_event_severity_alert hors plage "
                    f"[0.0, 1.0] pour {market}"
                )
                raise ValueError(error_msg)

        elif config_file == "trade_probability_config.yaml":
            buffer_size = (
                config.get("trade_probability", {})
                .get("buffer_size", {})
                .get("value", 0)
            )
            if not (50 <= buffer_size <= 1000):
                error_msg = (
                    f"{config_file} : buffer_size hors plage [50, 1000] "
                    f"pour {market}"
                )
                raise ValueError(error_msg)
            min_trade_success_prob = (
                config.get("trade_probability", {})
                .get("min_trade_success_prob", {})
                .get("value", 0)
            )
            if not (0.5 <= min_trade_success_prob <= 1.0):
                error_msg = (
                    f"{config_file} : min_trade_success_prob hors plage [0.5, 1.0] "
                    f"pour {market}"
                )
                raise ValueError(error_msg)
            retrain_frequency = (
                config.get("trade_probability", {})
                .get("retrain_frequency", {})
                .get("value", "")
            )
            if retrain_frequency not in ["weekly", "biweekly", "daily"]:
                error_msg = (
                    f"{config_file} : retrain_frequency doit être 'weekly', "
                    f"'biweekly', ou 'daily' pour {market}"
                )
                raise ValueError(error_msg)
            retrain_threshold = (
                config.get("trade_probability", {})
                .get("retrain_threshold", {})
                .get("value", 0)
            )
            if not (500 <= retrain_threshold <= 5000):
                error_msg = (
                    f"{config_file} : retrain_threshold hors plage [500, 5000] "
                    f"pour {market}"
                )
                raise ValueError(error_msg)

        config_cache[cache_key] = True
        self.save_snapshot(
            "config_validation",
            {
                "file": config_file,
                "status": "success",
                "confidence_drop_rate": (
                    confidence_drop_rate if config_file == "feature_sets.yaml" else 0.0
                ),
            },
            market=market,
        )
        self.log_performance(
            "validate_config",
            time.time() - start_time,
            success=True,
            config_file=config_file,
            market=market,
        )

    def get_config(
        self, path: str = None, mock_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Charge la configuration depuis un fichier YAML ou un mock."""
        if mock_config is not None:
            return mock_config
        config_file = os.path.basename(path)
        if config_file not in self.configs:
            self.reload_config(config_file)
        return self.configs.get(config_file, {})

    def get_features(self, context: str) -> List[str]:
        """Charge la liste des features pour le contexte spécifié."""
        config = self.get_config("config/feature_sets.yaml")
        if context == "training":
            return config.get("ES", {}).get("training", [])
        elif context == "inference":
            return config.get("ES", {}).get("inference", [])
        error_msg = f"Contexte inconnu : {context}"
        raise ValueError(error_msg)

    def get_iqfeed_config(self) -> Dict[str, Any]:
        """Retourne la configuration iqfeed_config.yaml."""
        return self.get_config("config/iqfeed_config.yaml")

    def get_credentials(self) -> Dict[str, Any]:
        """Retourne la configuration credentials.yaml (sécurisée)."""
        return {
            k: {sk: "***" if "api_key" in sk.lower() else sv for sk, sv in v.items()}
            for k, v in self.get_config("config/credentials.yaml").items()
        }

    def validate_obs_t(
        self, features: List[str], context: str = "training"
    ) -> List[str]:
        """Valide les features par rapport à un modèle obs_t standardisé."""
        missing_features = [f for f in CRITICAL_FEATURES if f not in features]
        if missing_features:
            alert_msg = (
                f"Validation obs_t échouée pour {context}: features manquantes "
                f"{missing_features}"
            )
            logger.warning(alert_msg)
            self.alert_manager.send_alert(alert_msg, priority=3)
            send_telegram_alert(alert_msg)
        return missing_features

    def finalize_config(self):
        """Finalise les configurations en validant l'intégrité globale."""
        for config_file in self.configs:
            self._validate_config(config_file, self.configs[config_file], market="ES")
        logger.info("Configurations finalisées avec succès")
        self.save_snapshot(
            "config_finalization",
            {"status": "success", "num_configs": len(self.configs)},
            market="ES",
        )

    def log_performance(
        self,
        operation: str,
        latency: float,
        success: bool = True,
        error: str = None,
        market: str = "ES",
        **kwargs,
    ) -> None:
        """
        Enregistre les performances (CPU, mémoire, latence) dans config_performance.csv.

        Args:
            operation (str): Nom de l’opération.
            latency (float): Temps d’exécution en secondes.
            success (bool): Indique si l’opération a réussi.
            error (str): Message d’erreur (si applicable).
            market (str): Marché (ex. : ES, MNQ).
        """
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
            cpu_percent = psutil.cpu_percent()
            if memory_usage > 1024:
                alert_msg = (
                    f"ALERTE: Usage mémoire élevé ({memory_usage:.2f} MB) "
                    f"pour {market}"
                )
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=5)
                send_telegram_alert(alert_msg)
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "latency": latency,
                "success": success,
                "error": error,
                "memory_usage_mb": memory_usage,
                "cpu_percent": cpu_percent,
                "market": market,
                **kwargs,
            }
            log_df = pd.DataFrame([log_entry])

            def save_log():
                if not PERF_LOG_PATH.exists():
                    log_df.to_csv(PERF_LOG_PATH, index=False, encoding="utf-8")
                else:
                    log_df.to_csv(
                        PERF_LOG_PATH,
                        mode="a",
                        header=False,
                        index=False,
                        encoding="utf-8",
                    )

            self.with_retries(save_log, market=market)
            logger.info(
                f"Performance journalisée pour {operation}. CPU: {cpu_percent}%"
            )
        except Exception as e:
            error_msg = (
                f"Erreur journalisation performance pour {market}: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)

    def save_snapshot(
        self, 
        snapshot_type: str, 
        data: Dict, 
        market: str = "ES", 
        compress: bool = True
    ) -> None:
        """
        Sauvegarde un instantané JSON des résultats, compressé avec gzip.

        Args:
            snapshot_type (str): Type de snapshot (ex. : config_validation).
            data (Dict): Données à sauvegarder.
            market (str): Marché (ex. : ES, MNQ).
            compress (bool): Compresser avec gzip (défaut : True).
        """
        start_time = time.time()
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = {
                "timestamp": timestamp,
                "type": snapshot_type,
                "market": market,
                "data": data,
            }
            snapshot_dir = CACHE_DIR / market
            snapshot_dir.mkdir(exist_ok=True)
            snapshot_path = snapshot_dir / f"snapshot_{snapshot_type}_{timestamp}.json"

            def write_snapshot():
                if compress:
                    with gzip.open(f"{snapshot_path}.gz", "wt", encoding="utf-8") as f:
                        json.dump(snapshot, f, indent=4)
                else:
                    with open(snapshot_path, "w", encoding="utf-8") as f:
                        json.dump(snapshot, f, indent=4)

            self.with_retries(write_snapshot, market=market)
            save_path = f"{snapshot_path}.gz" if compress else snapshot_path
            file_size = os.path.getsize(save_path) / 1024 / 1024
            if file_size > 1.0:
                alert_msg = (
                    f"Snapshot size {file_size:.2f} MB exceeds 1 MB pour {market}"
                )
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
            latency = time.time() - start_time
            success_msg = (
                f"Snapshot {snapshot_type} sauvegardé pour {market}: {save_path}"
            )
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "save_snapshot",
                latency,
                success=True,
                snapshot_size_mb=file_size,
                market=market,
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde snapshot {snapshot_type} pour {market}: "
                f"{str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "save_snapshot", 
                0, 
                success=False, 
                error=str(e), 
                market=market
            )

    def checkpoint(
        self, 
        data: pd.DataFrame, 
        data_type: str = "config_state", 
        market: str = "ES"
    ) -> None:
        """
        Sauvegarde incrémentielle des données toutes les 5 minutes avec versionnage 
        (5 versions).

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : config_state).
            market (str): Marché (ex. : ES, MNQ).
        """
        try:
            start_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_data = {
                "timestamp": timestamp,
                "num_rows": len(data),
                "columns": list(data.columns),
                "data_type": data_type,
                "market": market,
            }
            checkpoint_dir = CHECKPOINT_DIR / market
            checkpoint_dir.mkdir(exist_ok=True)
            checkpoint_path = (
                checkpoint_dir / f"config_{data_type}_{timestamp}.json.gz"
            )
            checkpoint_versions = []

            def write_checkpoint():
                with gzip.open(checkpoint_path, "wt", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, indent=4)
                csv_path = checkpoint_path.with_suffix(".csv")
                data.to_csv(csv_path, index=False, encoding="utf-8")

            self.with_retries(write_checkpoint, market=market)
            checkpoint_versions.append(checkpoint_path)
            if len(checkpoint_versions) > 5:
                oldest = checkpoint_versions.pop(0)
                if oldest.exists():
                    oldest.unlink()
                csv_oldest = oldest.with_suffix(".csv")
                if csv_oldest.exists():
                    csv_oldest.unlink()
            file_size = os.path.getsize(checkpoint_path) / 1024 / 1024
            latency = time.time() - start_time
            success_msg = (
                f"Checkpoint sauvegardé pour {market}: {checkpoint_path}"
            )
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "checkpoint",
                latency,
                success=True,
                file_size_mb=file_size,
                num_rows=len(data),
                data_type=data_type,
                market=market,
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde checkpoint pour {market}: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "checkpoint",
                0,
                success=False,
                error=str(e),
                data_type=data_type,
                market=market,
            )

    def cloud_backup(
        self, 
        data: pd.DataFrame, 
        data_type: str = "config_state", 
        market: str = "ES"
    ) -> None:
        """
        Sauvegarde distribuée des données vers AWS S3 toutes les 15 minutes.

        Args:
            data (pd.DataFrame): Données à sauvegarder.
            data_type (str): Type de données (ex. : config_state).
            market (str): Marché (ex. : ES, MNQ).
        """
        try:
            start_time = time.time()
            config = self.get_config("config/es_config.yaml")
            if not config.get("s3_bucket") or not (
                os.environ.get("AWS_ACCESS_KEY_ID")
                or os.path.exists(os.path.expanduser("~/.aws/credentials"))
            ):
                warning_msg = (
                    f"S3 bucket ou clés AWS non configurés, sauvegarde cloud ignorée "
                    f"pour {market}"
                )
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = (
                f"{config['s3_prefix']}config_{data_type}_{market}_{timestamp}.csv.gz"
            )
            temp_path = CHECKPOINT_DIR / market / f"temp_s3_{timestamp}.csv.gz"
            temp_path.parent.mkdir(exist_ok=True)

            def write_temp():
                data.to_csv(
                    temp_path, 
                    compression="gzip", 
                    index=False, 
                    encoding="utf-8"
                )

            self.with_retries(write_temp, market=market)
            s3_client = boto3.client("s3")

            def upload_s3():
                s3_client.upload_file(str(temp_path), config["s3_bucket"], backup_path)

            self.with_retries(upload_s3, market=market)
            temp_path.unlink()
            latency = time.time() - start_time
            success_msg = (
                f"Sauvegarde cloud S3 effectuée pour {market}: {backup_path}"
            )
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "cloud_backup",
                latency,
                success=True,
                num_rows=len(data),
                data_type=data_type,
                market=market,
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde cloud S3 pour {market}: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance(
                "cloud_backup",
                0,
                success=False,
                error=str(e),
                data_type=data_type,
                market=market,
            )

    def with_retries(
        self,
        func: callable,
        max_attempts: int = MAX_RETRIES,
        delay_base: float = RETRY_DELAY,
        market: str = "ES",
    ) -> Optional[Any]:
        """
        Exécute une fonction avec retries exponentiels (max 3, délai exponentiel).

        Args:
            func (callable): Fonction à exécuter.
            max_attempts (int): Nombre maximum de tentatives.
            delay_base (float): Base pour le délai exponentiel.
            market (str): Marché (ex. : ES, MNQ).

        Returns:
            Optional[Any]: Résultat de la fonction ou None si échec.
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
                    market=market,
                )
                return result
            except Exception as e:
                if attempt == max_attempts - 1:
                    error_msg = (
                        f"Échec après {max_attempts} tentatives pour {market}: "
                        f"{str(e)}\n{traceback.format_exc()}"
                    )
                    logger.error(error_msg)
                    self.alert_manager.send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    self.log_performance(
                        f"retry_attempt_{attempt+1}",
                        time.time() - start_time,
                        success=False,
                        error=str(e),
                        attempt_number=attempt + 1,
                        market=market,
                    )
                    return None
                delay = delay_base**attempt
                warning_msg = (
                    f"Tentative {attempt+1} échouée pour {market}, "
                    f"retry après {delay}s"
                )
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                time.sleep(delay)


class ConfigManager:
    """Gère le chargement, la validation, et la distribution des configurations avec 
    retries, snapshots JSON, et monitoring des performances."""

    CONFIG_FILES = [
        "es_config.yaml",
        "trading_env_config.yaml",
        "mia_config.yaml",
        "credentials.yaml",
        "feature_sets.yaml",
        "market_config.yaml",
        "model_params.yaml",
        "router_config.yaml",
        "iqfeed_config.yaml",
        "algo_config.yaml",
        "alert_config.yaml",
        "trade_probability_config.yaml",
        "risk_manager_config.yaml",
        "regime_detector_config.yaml",
        "trade_probability_rl.yaml",  # Corrigé de trade_probability_rl.py
    ]

    def __init__(self, env: str = "prod", fallback: bool = False):
        """Initialise avec support multi-environnements et fallback optionnel."""
        self.context = ConfigContext(env, fallback)

    def get_config(
        self, path: str = None, mock_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Charge la configuration depuis un fichier YAML ou un mock."""
        return self.context.get_config(path, mock_config)

    def get_features(self, context: str) -> List[str]:
        """Charge la liste des features pour le contexte spécifié."""
        return self.context.get_features(context)

    def get_iqfeed_config(self) -> Dict[str, Any]:
        """Retourne la configuration iqfeed_config.yaml."""
        return self.context.get_iqfeed_config()

    def get_credentials(self) -> Dict[str, Any]:
        """Retourne la configuration credentials.yaml (sécurisée)."""
        return self.context.get_credentials()

    def reload_config(self, config_file: str, market: str = "ES"):
        """Recharge un fichier de configuration spécifique."""
        self.context.reload_config(config_file, market)

    def validate_obs_t(
        self, features: List[str], context: str = "training"
    ) -> List[str]:
        """Valide les features par rapport à un modèle obs_t standardisé."""
        return self.context.validate_obs_t(features, context)

    def finalize_config(self):
        """Finalise les configurations en validant l'intégrité globale."""
        self.context.finalize_config()


# Instance globale
config_manager = ConfigManager()