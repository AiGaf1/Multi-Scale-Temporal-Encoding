import os
import logging
from dataclasses import asdict

from pytorch_lightning.loggers import CSVLogger
from datetime import datetime

from config import ExperimentConfig


def setup_logger(name: str = None, level=logging.INFO) -> logging.Logger:
    """Set up and return a logger instance with a consistent format."""
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-2s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
        logger.propagate = False
    return logger

def setup_run_logging(config: ExperimentConfig, model_name: str) -> tuple[CSVLogger, str]:
    """Log metrics to outputs/<name>/<timestamp>/metrics.csv; checkpoints and predictions go to the same folder."""
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_logger = CSVLogger(save_dir="outputs", name=config.name, version=version)

    raw = asdict(config)
    hparams = {f"model.{k}": v for k, v in raw.pop("model").items()}
    hparams.update(raw)
    hparams["model_name"] = model_name
    csv_logger.log_hyperparams(hparams)

    return csv_logger, csv_logger.log_dir

def load_comparisons(scenario: str, logger) -> list[tuple[str, str]]:
    """Load comparison pairs from file."""
    comps_file = f"data/Aalto/raw/{scenario}/{scenario}_comparisons.txt"

    with open(comps_file, "r") as f:
        comparisons = eval(f.readline())

    logger.info(f"Loaded {len(comparisons)} comparison pairs")
    return comparisons

def save_predictions(similarities: list[float], scenario: str, run_dir: str, logger) -> None:
    """Save similarity scores to file."""
    os.makedirs(run_dir, exist_ok=True)
    output_file = os.path.join(run_dir, f'{scenario}_predictions.txt')
    with open(output_file, "w") as f:
        f.write(str(similarities))
    logger.info(f"Predictions saved to {output_file}")
