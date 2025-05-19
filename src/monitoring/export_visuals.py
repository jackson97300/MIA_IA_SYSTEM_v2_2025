# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/src/monitoring/export_visuals.py
# Module pour exporter les visualisations en PDF/HTML dans MIA_IA_SYSTEM_v2_2025.
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Génère et sauvegarde des graphiques de performance et de risque (Phase 15, méthode 17).
#        Conforme à structure.txt (version 2.1.4, 2025-05-13).
#
# Dépendances :
# - pandas>=2.0.0,<3.0.0, plotly>=5.15.0,<6.0.0, matplotlib>=3.8.0,<4.0.0, reportlab>=4.0.0,<5.0.0,
#   psutil>=5.9.8,<6.0.0, loguru>=0.7.0,<1.0.0, pyyaml>=6.0.0,<7.0.0, boto3>=1.26.0,<2.0.0,
#   os, tempfile, signal, gzip
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
# - Compatible avec les données de performance et de risque (Phase 15, méthode 17).
# - Supprime toute référence à 320 features, 81 features, et obs_t.
# - Implémente retries (max 3, délai 2^attempt secondes) pour les opérations critiques.
# - Intègre confidence_drop_rate (Phase 8) pour évaluer la fiabilité des données.
# - Alertes via alert_manager.py et Telegram pour erreurs critiques (priorité ≥ 4).
# - Sauvegardes incrémentielles toutes les 5 min dans data/checkpoints/ (5 versions).
# - Sauvegardes distribuées toutes les 15 min vers S3 (configurable).
# - Tests unitaires disponibles dans tests/test_export_visuals.py.
# - Validation complète prévue pour juin 2025.

import gzip
import json
import os
import signal
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.io as pio
import psutil
from loguru import logger
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from src.model.utils.alert_manager import AlertManager
from src.model.utils.config_manager import get_config
from src.utils.telegram_alert import send_telegram_alert

# Configuration logging
logger.remove()
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOG_DIR = BASE_DIR / "data" / "logs" / "monitoring"
FIGURES_DIR = BASE_DIR / "data" / "figures" / "monitoring"
CACHE_DIR = BASE_DIR / "data" / "cache" / "visuals"
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints"
PERF_LOG_PATH = LOG_DIR / "export_visuals_performance.csv"
LOG_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "export_visuals.log", rotation="10 MB", level="INFO", encoding="utf-8"
)
logger.add(lambda msg: print(msg, end=""), level="INFO")

# Constantes
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Secondes, exponentiel

# Variable pour gérer l'arrêt propre
RUNNING = True


class VisualExporter:
    """
    Classe pour générer et exporter des visualisations en PDF/HTML.
    """

    def __init__(
        self, config_path: str = "config/es_config.yaml", output_dir: Path = FIGURES_DIR
    ):
        """
        Initialise l'exportateur de visualisations.

        Args:
            config_path (str): Chemin vers le fichier de configuration.
            output_dir (Path): Répertoire pour sauvegarder les visualisations.
        """
        self.alert_manager = AlertManager()
        self.output_dir = output_dir
        self.checkpoint_versions = []
        signal.signal(signal.SIGINT, self.handle_sigint)

        start_time = datetime.now()
        try:
            self.config = self.load_config(config_path)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.log_buffer = []
            success_msg = (
                f"VisualExporter initialisé, répertoire de sortie: {self.output_dir}"
            )
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance("init", latency, success=True)
            self.save_snapshot(
                "init",
                {
                    "config_path": config_path,
                    "output_dir": str(self.output_dir),
                    "timestamp": datetime.now().isoformat(),
                },
            )
        except Exception as e:
            error_msg = f"Erreur initialisation VisualExporter: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance("init", 0, success=False, error=str(e))
            raise

    def handle_sigint(self, signum, frame):
        """Gère l'arrêt propre sur SIGINT."""
        global RUNNING
        datetime.now()
        snapshot = {"timestamp": datetime.now().isoformat(), "status": "SIGINT"}
        CACHE_DIR / f'visuals_sigint_{snapshot["timestamp"]}.json.gz'
        try:
            RUNNING = False
            self.save_snapshot("sigint", snapshot)
            success_msg = "Arrêt propre sur SIGINT, snapshot sauvegardé"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=2)
            send_telegram_alert(success_msg)
            self.log_performance("handle_sigint", 0, success=True)
            exit(0)
        except Exception as e:
            error_msg = f"Erreur sauvegarde SIGINT: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance("handle_sigint", 0, success=False, error=str(e))
            exit(1)

    def save_snapshot(
        self, snapshot_type: str, data: Dict, compress: bool = True
    ) -> None:
        """
        Sauvegarde un instantané des résultats.

        Args:
            snapshot_type (str): Type de snapshot (ex. : init, sigint).
            data (Dict): Données à sauvegarder.
            compress (bool): Compresser avec gzip (défaut : True).
        """
        try:
            start_time = datetime.now()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = {
                "timestamp": timestamp,
                "type": snapshot_type,
                "data": data,
                "buffer_size": len(self.log_buffer),
            }
            snapshot_path = CACHE_DIR / f"snapshot_{snapshot_type}_{timestamp}.json"
            CACHE_DIR.mkdir(exist_ok=True)

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
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Snapshot {snapshot_type} sauvegardé: {save_path}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "save_snapshot", latency, success=True, snapshot_size_mb=file_size
            )
        except Exception as e:
            error_msg = f"Erreur sauvegarde snapshot {snapshot_type}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance("save_snapshot", 0, success=False, error=str(e))

    def checkpoint(self, data: pd.DataFrame) -> None:
        """
        Sauvegarde incrémentielle des données des visualisations toutes les 5 minutes avec versionnage (5 versions).

        Args:
            data (pd.DataFrame): Données des visualisations à sauvegarder.
        """
        try:
            start_time = datetime.now()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_data = {
                "timestamp": timestamp,
                "num_rows": len(data),
                "columns": list(data.columns),
            }
            checkpoint_path = CHECKPOINT_DIR / f"visuals_{timestamp}.json.gz"
            CHECKPOINT_DIR.mkdir(exist_ok=True)

            def write_checkpoint():
                with gzip.open(checkpoint_path, "wt", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, indent=4)
                data.to_csv(
                    checkpoint_path.replace(".json.gz", ".csv"),
                    index=False,
                    encoding="utf-8",
                )

            self.with_retries(write_checkpoint)
            self.checkpoint_versions.append(checkpoint_path)
            if len(self.checkpoint_versions) > 5:
                oldest = self.checkpoint_versions.pop(0)
                if os.path.exists(oldest):
                    os.remove(oldest)
                csv_oldest = oldest.replace(".json.gz", ".csv")
                if os.path.exists(csv_oldest):
                    os.remove(csv_oldest)
            file_size = os.path.getsize(checkpoint_path) / 1024 / 1024
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Checkpoint sauvegardé: {checkpoint_path}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "checkpoint",
                latency,
                success=True,
                file_size_mb=file_size,
                num_rows=len(data),
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde checkpoint: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance("checkpoint", 0, success=False, error=str(e))

    def cloud_backup(self, data: pd.DataFrame) -> None:
        """
        Sauvegarde distribuée des données des visualisations vers AWS S3 toutes les 15 minutes (configurable).

        Args:
            data (pd.DataFrame): Données des visualisations à sauvegarder.
        """
        try:
            start_time = datetime.now()
            if not self.config.get("s3_bucket"):
                warning_msg = "S3 bucket non configuré, sauvegarde cloud ignorée"
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.config['s3_prefix']}visuals_{timestamp}.csv.gz"
            temp_path = CHECKPOINT_DIR / f"temp_s3_{timestamp}.csv.gz"

            def write_temp():
                data.to_csv(
                    temp_path, compression="gzip", index=False, encoding="utf-8"
                )

            self.with_retries(write_temp)
            s3_client = boto3.client("s3")

            def upload_s3():
                s3_client.upload_file(temp_path, self.config["s3_bucket"], backup_path)

            self.with_retries(upload_s3)
            os.remove(temp_path)
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Sauvegarde cloud S3 effectuée: {backup_path}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance(
                "cloud_backup", latency, success=True, num_rows=len(data)
            )
        except Exception as e:
            error_msg = (
                f"Erreur sauvegarde cloud S3: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)
            self.log_performance("cloud_backup", 0, success=False, error=str(e))

    def with_retries(
        self,
        func: callable,
        max_attempts: int = MAX_RETRIES,
        delay_base: float = RETRY_DELAY_BASE,
    ) -> Optional[Any]:
        """
        Exécute une fonction avec retries exponentiels.

        Args:
            func (callable): Fonction à exécuter.
            max_attempts (int): Nombre maximum de tentatives.
            delay_base (float): Base pour le délai exponentiel.

        Returns:
            Optional[Any]: Résultat de la fonction ou None si échec.
        """
        start_time = datetime.now()
        for attempt in range(max_attempts):
            try:
                result = func()
                latency = (datetime.now() - start_time).total_seconds()
                self.log_performance(
                    f"retry_attempt_{attempt+1}",
                    latency,
                    success=True,
                    attempt_number=attempt + 1,
                )
                return result
            except Exception as e:
                if attempt == max_attempts - 1:
                    error_msg = f"Échec après {max_attempts} tentatives: {str(e)}\n{traceback.format_exc()}"
                    logger.error(error_msg)
                    self.alert_manager.send_alert(error_msg, priority=4)
                    send_telegram_alert(error_msg)
                    latency = (datetime.now() - start_time).total_seconds()
                    self.log_performance(
                        f"retry_attempt_{attempt+1}",
                        latency,
                        success=False,
                        error=str(e),
                        attempt_number=attempt + 1,
                    )
                    return None
                delay = delay_base**attempt
                warning_msg = f"Tentative {attempt+1} échouée, retry après {delay}s"
                logger.warning(warning_msg)
                self.alert_manager.send_alert(warning_msg, priority=3)
                send_telegram_alert(warning_msg)
                time.sleep(delay)

    def load_config(self, config_path: str) -> Dict:
        """
        Charge la configuration avec repli sur les valeurs par défaut.

        Args:
            config_path (str): Chemin vers le fichier de configuration.

        Returns:
            Dict: Configuration chargée.
        """
        start_time = datetime.now()
        try:
            config = get_config(BASE_DIR / config_path)
            if not config:
                raise ValueError("Configuration vide ou non trouvée")
            params = config.get(
                "visuals_params", {"s3_bucket": None, "s3_prefix": "visuals/"}
            )
            required_keys = ["s3_bucket", "s3_prefix"]
            missing_keys = [key for key in required_keys if key not in params]
            if missing_keys:
                raise ValueError(f"Clés de configuration manquantes: {missing_keys}")
            latency = (datetime.now() - start_time).total_seconds()
            success_msg = f"Configuration {config_path} chargée"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)
            self.log_performance("load_config", latency, success=True)
            return params
        except Exception as e:
            error_msg = (
                f"Erreur chargement configuration: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=4)
            send_telegram_alert(error_msg)
            self.log_performance("load_config", 0, success=False, error=str(e))
            return {"s3_bucket": None, "s3_prefix": "visuals/"}

    def log_performance(
        self, operation: str, latency: float, success: bool, error: str = None, **kwargs
    ) -> None:
        """
        Journalise les performances des opérations critiques.

        Args:
            operation (str): Nom de l’opération.
            latency (float): Latence en secondes.
            success (bool): Succès de l’opération.
            error (str, optional): Message d’erreur si échec.
            **kwargs: Paramètres supplémentaires (ex. : num_rows, attempt_number).
        """
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Mo
            cpu_percent = psutil.cpu_percent()
            if memory_usage > 1024:
                alert_msg = f"ALERTE: Usage mémoire élevé ({memory_usage:.2f} MB)"
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
                **kwargs,
            }
            self.log_buffer.append(log_entry)
            if len(self.log_buffer) >= self.config.get("logging", {}).get(
                "buffer_size", 100
            ):
                log_df = pd.DataFrame(self.log_buffer)

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

                self.with_retries(save_log)
                self.checkpoint(log_df)
                self.cloud_backup(log_df)
                self.log_buffer = []
            logger.info(
                f"Performance journalisée pour {operation}. CPU: {cpu_percent}%"
            )
        except Exception as e:
            error_msg = (
                f"Erreur journalisation performance: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=3)
            send_telegram_alert(error_msg)

    def export_visuals(
        self, data: pd.DataFrame, prefix: str = "visuals"
    ) -> Dict[str, Path]:
        """
        Génère et exporte des visualisations à partir des données.

        Args:
            data (pd.DataFrame): Données contenant les colonnes nécessaires :
                - timestamp: Horodatage
                - cumulative_return: Rendement cumulé
                - gamma_exposure: Exposition au gamma (optionnel)
                - iv_sensitivity: Sensibilité à la volatilité implicite (optionnel)
            prefix (str): Préfixe pour les noms de fichiers exportés.

        Returns:
            Dict[str, Path]: Dictionnaire des chemins des fichiers exportés (html, pdf).

        Raises:
            ValueError: Si les colonnes requises sont manquantes.
        """
        start_time = datetime.now()
        try:
            # Vérifier les colonnes requises
            required_columns = ["timestamp", "cumulative_return"]
            missing_columns = [
                col for col in required_columns if col not in data.columns
            ]
            confidence_drop_rate = 1.0 - min(
                (len(required_columns) - len(missing_columns)) / len(required_columns),
                1.0,
            )
            if confidence_drop_rate > 0.5:
                alert_msg = f"Confidence_drop_rate élevé: {confidence_drop_rate:.2f} ({len(required_columns) - len(missing_columns)}/{len(required_columns)} colonnes)"
                logger.warning(alert_msg)
                self.alert_manager.send_alert(alert_msg, priority=3)
                send_telegram_alert(alert_msg)
            if missing_columns:
                error_msg = f"Colonnes manquantes dans les données: {missing_columns}"
                logger.error(error_msg)
                self.alert_manager.send_alert(error_msg, priority=5)
                send_telegram_alert(error_msg)
                raise ValueError(error_msg)

            # Générer les visualisations
            output_files = {}

            # Visualisation HTML avec Plotly
            html_path = self.output_dir / f"{prefix}_performance.html"
            fig = px.line(
                data,
                x="timestamp",
                y="cumulative_return",
                title="Rendement Cumulé",
                labels={"cumulative_return": "Rendement Cumulé", "timestamp": "Date"},
            )

            # Ajouter des courbes pour gamma_exposure et iv_sensitivity si présentes
            if "gamma_exposure" in data.columns:
                fig.add_scatter(
                    x=data["timestamp"],
                    y=data["gamma_exposure"],
                    name="Exposition au Gamma",
                    yaxis="y2",
                )
                fig.update_layout(
                    yaxis2=dict(
                        title="Exposition au Gamma", overlaying="y", side="right"
                    )
                )
            if "iv_sensitivity" in data.columns:
                fig.add_scatter(
                    x=data["timestamp"],
                    y=data["iv_sensitivity"],
                    name="Sensibilité IV",
                    yaxis="y3",
                )
                fig.update_layout(
                    yaxis3=dict(
                        title="Sensibilité IV",
                        overlaying="y",
                        side="right",
                        anchor="free",
                        position=0.95,
                    )
                )

            def save_html():
                pio.write_html(fig, html_path)

            self.with_retries(save_html)
            output_files["html"] = html_path
            success_msg = f"Visualisation HTML exportée: {html_path}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)

            # Visualisation PDF avec Matplotlib et ReportLab
            pdf_path = self.output_dir / f"{prefix}_performance.pdf"
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_png = Path(temp_file.name)

                plt.figure(figsize=(8, 6))
                plt.plot(
                    data["timestamp"],
                    data["cumulative_return"],
                    label="Rendement Cumulé",
                    color="blue",
                )
                plt.title("Rendement Cumulé")
                plt.xlabel("Date")
                plt.ylabel("Rendement Cumulé")

                if "gamma_exposure" in data.columns or "iv_sensitivity" in data.columns:
                    ax2 = plt.gca().twinx()
                    if "gamma_exposure" in data.columns:
                        ax2.plot(
                            data["timestamp"],
                            data["gamma_exposure"],
                            label="Exposition au Gamma",
                            color="orange",
                        )
                        ax2.set_ylabel("Exposition au Gamma", color="orange")
                    if "iv_sensitivity" in data.columns:
                        ax2.plot(
                            data["timestamp"],
                            data["iv_sensitivity"],
                            label="Sensibilité IV",
                            color="green",
                        )
                        ax2.set_ylabel("Sensibilité IV", color="green")
                    ax2.legend(loc="upper right")

                plt.legend(loc="upper left")
                plt.tight_layout()

                def save_png():
                    plt.savefig(temp_png, format="png")
                    plt.close()

                self.with_retries(save_png)

                def save_pdf():
                    c = canvas.Canvas(str(pdf_path), pagesize=letter)
                    c.drawImage(str(temp_png), 50, 50, width=500, height=400)
                    c.showPage()
                    c.save()

                self.with_retries(save_pdf)

                os.unlink(temp_png)

            output_files["pdf"] = pdf_path
            success_msg = f"Visualisation PDF exportée: {pdf_path}"
            logger.info(success_msg)
            self.alert_manager.send_alert(success_msg, priority=1)
            send_telegram_alert(success_msg)

            latency = (datetime.now() - start_time).total_seconds()
            self.log_performance(
                "export_visuals",
                latency,
                success=True,
                num_rows=len(data),
                confidence_drop_rate=confidence_drop_rate,
            )
            self.checkpoint(data)
            self.cloud_backup(data)
            self.save_snapshot(
                "export_visuals",
                {
                    "prefix": prefix,
                    "output_files": {k: str(v) for k, v in output_files.items()},
                    "num_rows": len(data),
                },
            )
            return output_files

        except Exception as e:
            error_msg = f"Erreur lors de l'exportation des visualisations: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.alert_manager.send_alert(error_msg, priority=5)
            send_telegram_alert(error_msg)
            self.log_performance("export_visuals", 0, success=False, error=str(e))
            raise


def main():
    """
    Exemple d'utilisation pour le débogage.
    """
    try:
        # Créer des données factices pour tester
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start="2025-05-13 10:00:00", periods=10, freq="H"
                ),
                "cumulative_return": np.cumsum(np.random.uniform(-0.01, 0.01, 10)),
                "gamma_exposure": np.random.uniform(-500, 500, 10),
                "iv_sensitivity": np.random.uniform(0.1, 0.5, 10),
            }
        )

        # Initialiser l'exportateur
        exporter = VisualExporter()

        # Exporter les visualisations
        output_files = exporter.export_visuals(data, prefix="test")

        success_msg = f"Visualisations exportées avec succès: {output_files}"
        logger.info(success_msg)
        print(success_msg)

    except Exception as e:
        error_msg = f"Échec de l'exportation des visualisations: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        print(error_msg)
        exit(1)


if __name__ == "__main__":
    main()
