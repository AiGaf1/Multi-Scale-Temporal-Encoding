"""Metric-learning losses: SupCon and triplet."""
from pytorch_metric_learning import distances
from pytorch_metric_learning.losses import SupConLoss, TripletMarginLoss
from torch import nn

from mste import config as conf


def build_loss(config: conf.ExperimentConfig) -> nn.Module:
    """Instantiate the loss function specified by config.loss_type."""
    if config.loss_type == "supcon":
        return SupConLoss(temperature=config.loss_temperature)
    elif config.loss_type == "triplet":
        return TripletMarginLoss(
            margin=0.25,
            distance=distances.CosineSimilarity(),
        )
    else:
        raise ValueError(
            f"Unknown loss_type: '{config.loss_type}'. "
            f"Expected 'supcon' or 'triplet'."
        )
