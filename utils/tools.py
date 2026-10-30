import os
import logging
from dataclasses import asdict

from pytorch_lightning.loggers import WandbLogger
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

def setup_wandb_logging(
    config: ExperimentConfig,
    model_name: str,
    run_id: str = None,
) -> tuple[WandbLogger, str]:
    """Setup W&B logging with appropriate tags and versioning.

    When *run_id* is provided (sweep mode) the logger attaches to the
    already-active W&B run instead of creating a new one.
    """
    tags = [
        f"scenario_{config.scenario}",
        f"window_size_{config.window_size}",
        f"epochs_{config.epochs}",
        f"trigperiods_{config.model.n_periods}",
        f"model_{model_name}"
    ]

    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f'{config.epochs}_{config.scenario}'

    # save_dir is the parent of where WandbLogger writes {project}/{version}/
    # so we point it one level up ("outputs") and let the logger create the rest
    save_dir = "outputs"
    run_dir = os.path.join(save_dir, config.name, version)

    if run_id:
        # Attach to the sweep-managed run — do not call wandb.init() again
        wandb_logger = WandbLogger(
            project=config.name,
            name=experiment_name,
            id=run_id,
            resume="allow",
            save_dir=save_dir,
            log_model=True,
            tags=tags,
        )
    else:
        wandb_logger = WandbLogger(
            project=config.name,
            name=experiment_name,
            version=version,
            save_dir=save_dir,
            log_model=True,
            tags=tags,
        )

    if not run_id:
        # In sweep mode the agent already owns the config — skip to avoid locked-key warnings
        raw = asdict(config)
        hparams = {f"model.{k}": v for k, v in raw.pop("model").items()}
        hparams.update(raw)
        hparams["model_name"] = model_name
        wandb_logger.log_hyperparams(hparams)

    return wandb_logger, run_dir

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
