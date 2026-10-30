"""Embedding distances, benchmark similarity scores and the equal error rate (EER)."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve


def norm_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    return F.normalize(embeddings, dim=-1, p=2)


def score_comparisons(embeddings, comps):
    """Euclidean distance between L2-normalised embeddings for each comparison pair."""
    distances = {}
    for comp in comps:
        e1, e2 = norm_embeddings(embeddings[comp[0]]), norm_embeddings(embeddings[comp[1]])
        distances[str(comp)] = F.pairwise_distance(e1, e2).item()
    return distances

def distances_to_similarities(distances):
    """
    Convert distances to similarity in [0, 1]
    """
    distances_list = torch.tensor(list(distances.values()), dtype=torch.float32)
    max_dist = distances_list.max()
    similarities = 1 - distances_list / max_dist
    return similarities.tolist()


def compute_eer(pred1, pred2, labels):
    if not (torch.isfinite(pred1).all() and torch.isfinite(pred2).all()):
        print("Warning: NaN or Inf detected in embeddings")
    dists = torch.nn.functional.pairwise_distance(
        norm_embeddings(pred1), norm_embeddings(pred2)
    ).numpy()
    fpr, tpr, threshold = roc_curve(labels, dists, pos_label=1)
    fnr = 1 - tpr
    EER1 = fpr[np.nanargmin(np.abs(fnr - fpr))]
    EER2 = fnr[np.nanargmin(np.abs(fnr - fpr))]
    return np.mean([EER1, EER2]) * 100
