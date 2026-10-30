"""Training: the Lightning module, its callbacks, and console / run logging."""
import logging
import os
from dataclasses import asdict
from datetime import datetime

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from mste.config import ExperimentConfig
from mste.metrics import compute_eer
from typing import Any, Dict, Tuple, Optional


class VerificationModel(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: Optional[torch.nn.Module],
        t_0: float = 2500,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.save_hyperparameters(ignore=['model', 'loss_fn'])
        self._eer_outputs: Dict[str, list] = {"train": [], "val": []}

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.model(x, mask)

    def _step_common(
        self, batch: Tuple[Tuple[torch.Tensor, torch.Tensor],
            torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], stage: str
    ) -> Dict[str, Any]:
        """
        batch = ((x1_features, x1_keys), (x2_features, x2_keys)), labels, (u1, u2)
        """
        (x1, mask1), (x2, mask2), labels, (_, _), (u1, u2) = batch

        z1 = self.model(x1, mask1)
        z2 = self.model(x2, mask2)

        embeddings = torch.cat([z1, z2], dim=0)
        user_idx = torch.cat([u1, u2], dim=0)

        loss = self.loss_fn(embeddings, user_idx)

        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=x1.shape[0])

        self._eer_outputs[stage].append((
            z1.detach().float().cpu(),
            z2.detach().float().cpu(),
            labels.detach().float().cpu(),
        ))

        return loss

    def _compute_epoch_eer(self, stage: str):
        outputs = self._eer_outputs[stage]
        if not outputs:
            return
        z1 = torch.cat([o[0] for o in outputs])
        z2 = torch.cat([o[1] for o in outputs])
        labels = torch.cat([o[2] for o in outputs])
        eer = compute_eer(z1, z2, labels)
        self.log(f"{stage}/eer", eer, prog_bar=True)
        self._eer_outputs[stage].clear()

    def training_step(self, batch, batch_idx):
        return self._step_common(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step_common(batch, "val")

    def on_train_epoch_end(self):
        self._compute_epoch_eer("train")

    def on_validation_epoch_end(self):
        self._compute_epoch_eer("val")

    def predict_step(self, batch, batch_idx):
        fix_sessions, masks, session_ids = batch

        embeddings = self.forward(fix_sessions, mask=masks)

        # Return list of (session_id, embedding) tuples
        return [(sid, emb.cpu()) for sid, emb in zip(session_ids, embeddings)]

    def configure_optimizers(self):
        param_groups = [
            {"params": self.model.parameters(), "lr": self.hparams.lr},
        ]

        optimizer = torch.optim.AdamW(param_groups, betas=(0.95, 0.999))

        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.hparams.t_0,
            eta_min=self.hparams.lr / 100
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


def create_callbacks(scenario: str, run_dir: str = None) -> list[pl.Callback]:
    """Create training callbacks for checkpointing and monitoring."""
    dirpath = os.path.join(run_dir, "checkpoints") if run_dir else None
    checkpoint_cb = ModelCheckpoint(
        dirpath=dirpath,
        monitor="val/eer",
        mode="min",
        filename=f'{scenario}' + "-{epoch:02d}-{val/eer:.2f}",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False
    )

    lr_monitor = LearningRateMonitor(logging_interval=None)

    return [checkpoint_cb, lr_monitor]


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
