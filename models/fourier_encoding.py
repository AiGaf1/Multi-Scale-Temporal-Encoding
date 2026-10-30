import torch.nn as nn
import torch

import math


class LearnableFourierFeatures(nn.Module):
    def __init__(self, feature_dict: dict, num_features: int, learnable: bool = True):
        """
        feature_dict: dict where key=feature_name, value={"min": float, "max": float}
        num_features: number of Fourier frequencies per feature
        learnable: if True, scales are learned via sigmoid (init 0.5); if False, fixed to 1
        """
        super().__init__()

        self.feature_names = list(feature_dict.keys())
        self.num_features = num_features
        self.input_dim = len(self.feature_names)
        self.learnable = learnable

        # Initialize a frequency matrix (M x D)
        period_list = []
        for feat in self.feature_names:
            min_f, max_f = feature_dict[feat]["min"], feature_dict[feat]["max"]
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

        self.d_out = 2 * self.input_dim * num_features

    def get_scales(self) -> torch.Tensor:
        return torch.sigmoid(self.scales_raw) if self.learnable else self.scales

    def forward(self, x):
        """
        x: (B, L, M)
        return: (B, L, M*2*D)
        """
        x = x.unsqueeze(-1)                    # (B, L, M, 1)
        scales = self.get_scales()
        proj = x * self.freq * scales          # (B, L, M, D)

        fourier = torch.stack(
            [proj.sin(), proj.cos()],
            dim=-1                             # (B, L, M, D, 2)
        )
        return fourier.flatten(start_dim=-3)
    