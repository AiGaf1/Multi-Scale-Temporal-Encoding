# 1. Standard Library
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")
warnings.filterwarnings("ignore", message=".*dynamic_axes.*dynamo.*")
warnings.filterwarnings("ignore", message=".*dtype.*align.*", module="numpy")
warnings.filterwarnings("ignore", message=".*bf16-mixed is not supported by the model summary.*")

import logging
logging.getLogger("torch._inductor.autotune_process").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)

# 2. Third-Party Libraries
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import Logger
from pytorch_lightning.callbacks import ModelSummary
import torch
# 3. Local Modules (Project Specific)
import config as conf
from data.Aalto.dataset import KeystrokeDataModule
from models.lit_model import KeystrokeLitModel
from models.factory import build_model
from utils.callbacks import create_callbacks
from utils.losses import build_loss
from utils.tools import (
    load_comparisons,
    save_predictions,
    setup_logger,
    setup_run_logging
)
from utils.metrics import compute_distances4comps, normalize_distances

logger = setup_logger("train")

def initialize_environment(config: conf.ExperimentConfig) -> None:
    pl.seed_everything(config.seed, workers=True, verbose=False)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
    logger.info(f"Environment initialized | Seed: {config.seed} | Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

def create_data_module(
        file_path: str,
        predict_file_path: str = None,
        config: conf.ExperimentConfig = None,
) -> KeystrokeDataModule:
    """Create and setup the data module for training and prediction."""
    logger.info(f"Loading data from {file_path}")
    raw_data = np.load(file_path, allow_pickle=True).item()

    dm = KeystrokeDataModule(
        raw_data=raw_data,
        predict_file_path=predict_file_path,
        window_size=config.window_size,
        samples_per_batch_train=config.samples_per_batch_train,
        samples_per_batch_val=config.samples_per_batch_val,
        batches_per_epoch_train=config.batches_per_epoch_train,
        batches_per_epoch_val=config.batches_per_epoch_val,
        train_val_division=config.train_split,
        seed=config.seed,
        min_session_length=config.min_session_length,
        min_sessions_per_user=config.min_sessions_per_user,
        clip_percentile_lo=config.clip_percentile_lo,
        clip_percentile_hi=config.clip_percentile_hi,
        use_mste=config.use_mste,
    )
    dm.setup(None)
    return dm

def create_trainer(
        epochs: int,
        run_logger: Logger = None,
        callbacks: list[pl.Callback] = None,
) -> pl.Trainer:
    """Create PyTorch Lightning trainer with optimal settings."""
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {accelerator}")

    return pl.Trainer(
        max_epochs=epochs,
        logger=run_logger,
        callbacks=callbacks,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        accelerator=accelerator,
        devices=1,
        deterministic=True,
        log_every_n_steps=256, #per epoch
        num_sanity_val_steps=1,
        enable_model_summary=True,
    )

def run_predictions(
        trainer: pl.Trainer,
        model: KeystrokeLitModel,
        dm: KeystrokeDataModule,
        checkpoint_path: str = "best"
) -> dict[str, torch.Tensor]:
    """Run predictions and collect embeddings."""
    logger.info("Running predictions...")

    predictions = trainer.predict(
        model,
        datamodule=dm,
        ckpt_path=checkpoint_path
    )

    # Aggregate embeddings from all batches
    embeddings = {}
    for batch_predictions in predictions:
        for session_id, embedding in batch_predictions:
            embeddings[session_id] = embedding

    logger.info(f"Collected {len(embeddings)} session embeddings")
    return embeddings

def run_experiment(config: conf.ExperimentConfig) -> None:
    initialize_environment(config)

    # 1. Data & Model Setup
    dm = create_data_module(config.file_path, config.predict_file_path, config)

    loss_fn = build_loss(config)
    nn_model = build_model(config, dm.min_max)
    logger.info(f"Model: {nn_model.__class__.__name__} | model_type={config.model_type} | use_mste={config.use_mste}")
    lit_model = KeystrokeLitModel(nn_model, loss_fn, t_0=config.t_0, lr=config.lr)

    # 3. Training Infrastructure
    run_logger, run_dir = setup_run_logging(config, model_name=nn_model.__class__.__name__)
    callbacks = create_callbacks(config.scenario, run_dir)
    callbacks.append(ModelSummary(max_depth=4))
    trainer = create_trainer(config.epochs, run_logger, callbacks)

    # 4. Execution
    trainer.fit(lit_model, datamodule=dm)

    # 5. Score the official test comparisons with the best checkpoint
    best_model_path = callbacks[0].best_model_path
    logger.info(f"Best checkpoint: {best_model_path}")
    embeddings = run_predictions(trainer, lit_model, dm, best_model_path)
    comparisons = load_comparisons(config.scenario, logger)
    distances = compute_distances4comps(embeddings, comparisons)
    similarities = normalize_distances(distances)
    save_predictions(similarities, config.scenario, run_dir, logger)


def parse_args() -> argparse.Namespace:
    """Defaults reproduce the paper's best configuration (LSTM, MSTE with N = 32, SupConLoss)."""
    p = argparse.ArgumentParser(description="Train and evaluate MSTE keystroke verification models.")
    p.add_argument("--scenario", nargs="+", default=["desktop", "mobile"], choices=["desktop", "mobile"])
    p.add_argument("--model", default="lstm", choices=list(conf.MODEL_CONFIGS))
    p.add_argument("--loss", default="supcon", choices=["supcon", "triplet"])
    p.add_argument("--n-periods", type=int, default=32, help="number of MSTE frequencies N")
    p.add_argument("--no-mste", action="store_true", help="outlier clipping instead of MSTE")
    p.add_argument("--fixed-frequencies", action="store_true", help="MSTE with fixed (non-learnable) frequencies")
    p.add_argument("--window-size", type=int, default=50, help="maximum keystrokes per session")
    p.add_argument("--epochs", type=int, default=1000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for scenario in args.scenario:
        run_experiment(conf.ExperimentConfig(
            scenario=scenario,
            model_type=args.model,
            model=conf.MODEL_CONFIGS[args.model](n_periods=args.n_periods),
            loss_type=args.loss,
            use_mste=not args.no_mste,
            learnable_mste=not args.fixed_frequencies,
            window_size=args.window_size,
            epochs=args.epochs,
        ))
