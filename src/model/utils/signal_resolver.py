"""
signal_resolver.py - Module pour résoudre les conflits de signaux dans MIA_IA_SYSTEM_v2_2025.

Version : 2.1.5
Date : 2025-05-14

Rôle : Calcule un score agrégé à partir de signaux potentiellement contradictoires (ex. : régime de marché, microstructure, nouvelles, RL) en utilisant des poids configurables. Supporte la normalisation optionnelle, la persistance des scores dans market_memory.db, l'exportation en CSV, le calcul de l'entropie, et la détection des conflits élevés via un coefficient de conflit.

Dépendances :
- pandas, numpy : Manipulation des données.
- loguru : Journalisation.
- scipy.stats : Calcul de l'entropie.
- alert_manager : Envoi d'alertes en cas de scores anormaux.
- config_manager : Chargement des configurations depuis es_config.yaml.
- db_maintenance : Persistance des scores dans market_memory.db.

Conformité : Aucune référence à dxFeed, obs_t, 320 features, ou 81 features.
Note sur les policies : Le répertoire officiel pour les politiques de routage est src/model/router/policies.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import entropy

from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import load_config


class SignalResolver:
    """
    Classe pour résoudre les conflits entre signaux dans MIA_IA_SYSTEM_v2_2025.
    """

    def __init__(self, config_path: str = "config/es_config.yaml", market: str = "ES"):
        """
        Initialise le résolveur de signaux.

        Args:
            config_path (str): Chemin vers le fichier de configuration.
            market (str): Marché cible (ex. : "ES", "MNQ").
        """
        self.config_path = config_path
        self.market = market
        self.config = load_config(config_path)
        self.default_weights = self.config.get("signal_resolver", {}).get(
            "default_weights", {}
        )
        self.alert_manager = AlertManager(config_path=config_path, market=market)
        self.log_path = Path(f"data/logs/{market}/signal_resolver.log")
        self.csv_path = Path(f"data/logs/{market}/conflict_scores.csv")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(self.log_path, rotation="10 MB", retention="7 days", level="INFO")
        self.db_path = Path(f"data/{market}/market_memory_{market}.db")
        self.csv_buffer = []
        self.csv_buffer_limit = 100  # Écrire après 100 enregistrements
        self._initialize_db()
        logger.info(f"SignalResolver initialisé pour le marché {market}")

    def _initialize_db(self) -> None:
        """
        Initialise la table conflict_scores dans market_memory.db.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conflict_scores (
                        timestamp TEXT,
                        market TEXT,
                        score REAL,
                        contributions_json TEXT,
                        run_id TEXT
                    )
                """
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(
                f"Erreur lors de l'initialisation de la table conflict_scores : {str(e)}"
            )
            self.alert_manager.send_alert(
                f"Erreur DB SignalResolver : {str(e)}", priority=4
            )
            raise

    def _log_to_db(self, score: float, contributions: Dict, run_id: str) -> None:
        """
        Enregistre le score, les contributions, et l'UUID du run dans market_memory.db.

        Args:
            score (float): Score agrégé.
            contributions (Dict): Contributions des signaux.
            run_id (str): Identifiant unique du run.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                contributions_json = json.dumps(contributions)
                cursor.execute(
                    """
                    INSERT INTO conflict_scores (timestamp, market, score, contributions_json, run_id)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (timestamp, self.market, score, contributions_json, run_id),
                )
                conn.commit()
                logger.info(
                    f"Score {score:.2f} enregistré dans market_memory.db avec run_id {run_id}"
                )
        except sqlite3.Error as e:
            logger.error(
                f"Erreur lors de l'enregistrement dans market_memory.db : {str(e)}"
            )
            self.alert_manager.send_alert(
                f"Erreur DB SignalResolver : {str(e)}", priority=4
            )

    def _export_to_csv(self, score: float, contributions: Dict, run_id: str) -> None:
        """
        Ajoute un enregistrement au buffer CSV et écrit dans le fichier si nécessaire.

        Args:
            score (float): Score agrégé.
            contributions (Dict): Contributions des signaux.
            run_id (str): Identifiant unique du run.
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            record = {
                "timestamp": timestamp,
                "market": self.market,
                "score": score,
                "contributions_json": json.dumps(contributions),
                "run_id": run_id,
            }
            self.csv_buffer.append(record)

            if len(self.csv_buffer) >= self.csv_buffer_limit:
                df = pd.DataFrame(self.csv_buffer)
                mode = "a" if self.csv_path.exists() else "w"
                header = not self.csv_path.exists()
                df.to_csv(self.csv_path, mode=mode, header=header, index=False)
                logger.info(
                    f"Exporté {len(self.csv_buffer)} enregistrements dans {self.csv_path}"
                )
                self.csv_buffer = []
        except Exception as e:
            logger.error(f"Erreur lors de l'exportation en CSV : {str(e)}")
            self.alert_manager.send_alert(
                f"Erreur CSV SignalResolver : {str(e)}", priority=4
            )

    def resolve_conflict(
        self,
        signals: Dict[str, float],
        weights: Optional[Dict[str, float]] = None,
        normalize: bool = False,
        persist_to_db: bool = False,
        export_to_csv: bool = False,
        run_id: Optional[str] = None,
        score_type: str = "final",
        mode_debug: bool = False,
    ) -> Tuple[float, Dict]:
        """
        Résout les conflits entre signaux en calculant un score agrégé pondéré.

        Args:
            signals (Dict[str, float]): Dictionnaire des signaux (ex. : {"regime_trend": 1, "news_score_positive": 0}).
            weights (Optional[Dict[str, float]]): Dictionnaire des poids (ex. : {"regime_trend": 2.0}). Si None, utilise les poids par défaut.
            normalize (bool): Si True, normalise le score par la somme des poids.
            persist_to_db (bool): Si True, enregistre le score et les contributions dans market_memory.db.
            export_to_csv (bool): Si True, exporte le score et les contributions dans un fichier CSV.
            run_id (Optional[str]): Identifiant unique du run. Si None, génère un UUID.
            score_type (str): Type de score calculé (ex. : "final", "intermédiaire"). Par défaut : "final".
            mode_debug (bool): Si True, active les logs de débogage détaillés. Par défaut : False.

        Returns:
            Tuple[float, Dict]: Score agrégé (normalisé si demandé) et métadonnées des contributions.

        Raises:
            ValueError: Si les signaux ou les poids sont invalides.
        """
        try:
            # Validation des signaux
            for name, value in signals.items():
                if not isinstance(value, (int, float)) or value < -1 or value > 1:
                    raise ValueError(
                        f"Signal {name} doit être un nombre entre -1 et 1, reçu : {value}"
                    )

            # Utilisation des poids par défaut si non spécifiés
            weights = weights or self.default_weights
            weights = {
                k: v
                for k, v in weights.items()
                if isinstance(v, (int, float)) and v >= 0
            }

            # Génération de l'UUID du run si non fourni
            run_id = run_id or str(uuid.uuid4())

            # Calcul du score agrégé
            score = 0.0
            contributions = {}
            for name, value in signals.items():
                weight = weights.get(name, 1.0)
                contribution = value * weight
                score += contribution
                contributions[name] = {
                    "value": value,
                    "weight": weight,
                    "contribution": contribution,
                }

            # Normalisation optionnelle
            normalized_score = score
            total_weight = sum(weights.get(name, 1.0) for name in signals)
            if normalize and total_weight > 0:
                normalized_score = score / total_weight
            elif normalize:
                logger.warning("Somme des poids nulle, normalisation ignorée")

            # Calcul de l'entropie pour la cohérence
            contributions_normalized = [
                abs(c["contribution"]) / total_weight if total_weight > 0 else 0
                for c in contributions.values()
            ]
            contributions_normalized = [
                max(1e-10, c) for c in contributions_normalized
            ]  # Éviter log(0)
            score_entropy = (
                entropy(contributions_normalized, base=2)
                if contributions_normalized
                else 0.0
            )

            # Calcul du coefficient de conflit
            total_contribution = sum(
                abs(c["contribution"]) for c in contributions.values()
            )
            max_contribution = (
                max(abs(c["contribution"]) for c in contributions.values())
                if contributions
                else 1e-10
            )
            conflict_coefficient = (
                1.0 - (max_contribution / total_contribution)
                if total_contribution > 0
                else 0.0
            )

            # Métadonnées enrichies
            metadata = {
                "score": score,
                "normalized_score": normalized_score if normalize else None,
                "entropy": score_entropy,
                "conflict_coefficient": conflict_coefficient,
                "score_type": score_type,
                "contributions": contributions,
                "run_id": run_id,
            }

            # Journalisation
            logger.info(
                f"Score brut : {score:.2f}, Score normalisé : {normalized_score:.2f}, Entropie : {score_entropy:.2f}, "
                f"Coefficient de conflit : {conflict_coefficient:.4f}, Type : {score_type}, Run ID : {run_id}"
            )
            if mode_debug:
                logger.debug(f"[DEBUG] Signaux bruts : {signals}")
                logger.debug(f"[DEBUG] Poids utilisés : {weights}")
                logger.debug(
                    f"[DEBUG] Contributions normalisées : {contributions_normalized}"
                )
                logger.debug(
                    f"[DEBUG] Coefficient de conflit : {conflict_coefficient:.4f}"
                )

            # Persistance dans la base de données
            if persist_to_db:
                self._log_to_db(score, contributions, run_id)

            # Exportation en CSV
            if export_to_csv:
                self._export_to_csv(score, contributions, run_id)

            # Alertes en cas de score anormal ou entropie/conflit élevé
            if score < 0:
                self.alert_manager.send_alert(
                    f"Score de conflit négatif détecté : {score:.2f}, Contributions : {contributions}, Run ID : {run_id}",
                    priority=3,
                )
            elif score < 1.0:
                self.alert_manager.send_alert(
                    f"Score de conflit faible : {score:.2f}, Contributions : {contributions}, Run ID : {run_id}",
                    priority=2,
                )
            if score_entropy > 0.5:
                self.alert_manager.send_alert(
                    f"Entropie élevée détectée : {score_entropy:.2f}, Signaux dispersés, Contributions : {contributions}, Run ID : {run_id}",
                    priority=2,
                )
            if conflict_coefficient > 0.5:
                self.alert_manager.send_alert(
                    f"Conflit élevé détecté : {conflict_coefficient:.2f}, Signaux ambigus, Run ID : {run_id}",
                    priority=2,
                )

            return normalized_score if normalize else score, metadata

        except Exception as e:
            logger.error(
                f"Erreur lors de la résolution des conflits : {str(e)}, Run ID : {run_id}"
            )
            self.alert_manager.send_alert(
                f"Erreur SignalResolver : {str(e)}, Run ID : {run_id}", priority=4
            )
            raise

    def update_weights(self, performance_metrics: Dict[str, float]) -> None:
        """
        Met à jour les poids par défaut en fonction des performances historiques (option future).

        Args:
            performance_metrics (Dict[str, float]): Métriques de performance des signaux (ex. : {"regime_trend": 0.6}).

        Note : Non implémenté dans la version initiale. À développer avec meta-learning (méthode 18).
        """
        # TODO : Implémenter l'ajustement adaptatif des poids
        logger.warning(
            "update_weights non implémenté. À développer avec meta-learning."
        )


# Exemple d'utilisation
if __name__ == "__main__":
    import numpy as np

    resolver = SignalResolver(market="ES")
    signals = {
        "regime_trend": 1 if np.random.rand() > 0.3 else 0,
        "microstructure_bullish": 1 if np.random.rand() > 0.8 else 0,
        "news_score_positive": 1 if np.random.rand() > 0.5 else 0,
        "qr_dqn_positive": 1 if np.random.rand() > 0.5 else 0,
    }
    weights = {
        "regime_trend": 2.0,
        "qr_dqn_positive": 1.5,
        "microstructure_bullish": 1.0,
        "news_score_positive": 0.5,
    }
    score, metadata = resolver.resolve_conflict(
        signals,
        weights,
        normalize=True,
        persist_to_db=True,
        export_to_csv=True,
        run_id="test_run_123",
        score_type="intermédiaire",
        mode_debug=True,
    )
    print(f"Score final : {score:.2f}")
    print(f"Métadonnées : {metadata}")
