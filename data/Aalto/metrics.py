import torch
import numpy as np
from sklearn.metrics import roc_curve
from utils.metrics import norm_embeddings

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