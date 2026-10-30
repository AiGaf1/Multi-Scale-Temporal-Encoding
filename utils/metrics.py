import torch
import torch.nn.functional as F


def norm_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    return F.normalize(embeddings, dim=-1, p=2)


def compute_distances4comps(embeddings, comps):
    """Euclidean distance between L2-normalised embeddings for each comparison pair."""
    distances = {}
    for comp in comps:
        e1, e2 = norm_embeddings(embeddings[comp[0]]), norm_embeddings(embeddings[comp[1]])
        distances[str(comp)] = F.pairwise_distance(e1, e2).item()
    return distances

def normalize_distances(distances):
    """
    Convert distances to similarity in [0, 1]
    """
    distances_list = torch.tensor(list(distances.values()), dtype=torch.float32)
    max_dist = distances_list.max()
    similarities = 1 - distances_list / max_dist
    return similarities.tolist()