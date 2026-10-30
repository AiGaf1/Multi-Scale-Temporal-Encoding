import torch.nn as nn
import torch

import math


# ---- Time2Vec (Kazemi et al., 2019) ----------------------------------------

def _t2v(tau, f, w, b, w0, b0):
    v1 = f(torch.matmul(tau, w) + b)   # periodic: (..., out_features-1)
    v2 = torch.matmul(tau, w0) + b0    # linear:   (..., 1)
    return torch.cat([v1, v2], dim=-1) # (..., out_features)


class _SineActivation(nn.Module):
    """Time2Vec for a single scalar input (in_features=1)."""
    def __init__(self, out_features: int):
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(1, 1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w  = nn.Parameter(torch.randn(1, out_features - 1))
        self.b  = nn.Parameter(torch.randn(out_features - 1))

    def forward(self, tau):   # tau: (..., 1)
        return _t2v(tau, torch.sin, self.w, self.b, self.w0, self.b0)


class Time2VecFeatures(nn.Module):
    """Applies Time2Vec independently to each timing feature (HT, FT).
    out_features=64 per feature × 2 features = 128-dim, matching MSTE(N=32)."""
    def __init__(self, feature_dict: dict, num_features: int):
        super().__init__()
        self.input_dim = len(feature_dict)
        self.encoders = nn.ModuleList([
            _SineActivation(out_features=num_features) for _ in range(self.input_dim)
        ])
        self.d_out = self.input_dim * num_features

    def forward(self, x):
        # x: (B, L, M)
        outs = [self.encoders[i](x[..., i:i+1]) for i in range(self.input_dim)]
        return torch.cat(outs, dim=-1)   # (B, L, M * num_features)

class LearnableFourierFeatures(nn.Module):
    def __init__(self, feature_dict: dict, num_features: int, learnable: bool = True,
                 sin_only: bool = False, init_at_max: bool = False):
        """
        feature_dict: dict where key=feature_name, value={"min": float, "max": float}
        num_features: number of Fourier frequencies per feature
        learnable: if True, scales are learned via sigmoid (init 0.5); if False, fixed to 1
        sin_only: if True, output only sin (not [sin, cos]) — halves output dim
        init_at_max: if True, initialize single period at max_f (lowest frequency);
                     useful for dimensionality-matched baselines with num_features=1
        """
        super().__init__()

        self.feature_names = list(feature_dict.keys())
        self.num_features = num_features
        self.input_dim = len(self.feature_names)
        self.learnable = learnable
        self.sin_only = sin_only

        # Initialize a frequency matrix (M x D)
        period_list = []
        for feat in self.feature_names:
            min_f, max_f = feature_dict[feat]["min"], feature_dict[feat]["max"]
            if init_at_max:
                period = torch.full((num_features,), max_f)
            else:
                period = torch.logspace(math.log10(min_f), math.log10(max_f), steps=num_features)
            period_list.append(period)
        period_init = torch.stack(period_list, dim=0)  # (M, D)

        freq = 2 * torch.pi / period_init
        self.register_buffer("freq", freq)

        if learnable:
            # sigmoid(0) = 0.5, so each frequency starts at 0.5 * its log-spaced initial value
            self.scales_raw = nn.Parameter(torch.zeros(self.input_dim, self.num_features))
        else:
            self.register_buffer("scales", torch.ones(self.input_dim, self.num_features))

        components_per_freq = 1 if sin_only else 2
        self.d_out = components_per_freq * self.input_dim * num_features

    def get_scales(self) -> torch.Tensor:
        return torch.sigmoid(self.scales_raw) if self.learnable else self.scales

    def forward(self, x):
        """
        x: (B, L, M)
        return: (B, L, M*D) if sin_only else (B, L, M*2*D)
        """
        x = x.unsqueeze(-1)                    # (B, L, M, 1)
        scales = self.get_scales()
        proj = x * self.freq * scales          # (B, L, M, D)

        if self.sin_only:
            return proj.sin().flatten(start_dim=-2)   # (B, L, M*D)

        fourier = torch.stack(
            [proj.sin(), proj.cos()],
            dim=-1                             # (B, L, M, D, 2)
        )
        return fourier.flatten(start_dim=-3)
    