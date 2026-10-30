from dataclasses import dataclass, field
from typing import Union


@dataclass
class TransformerConfig:
    num_layers: int = 5
    num_heads: int = 2
    ff_dim: int = 256
    dropout: float = 0.1
    n_periods: int = 16
    use_pos_enc: bool = True


@dataclass
class CNNConfig:
    n_periods: int = 16
    hidden_size: int = 128


@dataclass
class LSTMConfig:
    num_layers: int = 2
    dropout: float = 0.1
    n_periods: int = 16
    hidden_size: int = 128


@dataclass
class ExperimentConfig:
    name: str = "Keystroke-XAI"

    # Data
    window_size: int = 50
    min_session_length: int = 5
    min_sessions_per_user: int = 2
    train_split: float = 0.80

    # Model
    model_type: str = "transformer"  # "transformer" | "cnn" | "lstm"
    model: Union[TransformerConfig, CNNConfig, LSTMConfig] = field(default_factory=TransformerConfig)
    hidden_size: int = 128
    output_size: int = 256
    use_mste: bool = True
    learnable_mste: bool = True

    # Loss
    loss_type: str = "supcon"  # "supcon" | "triplet"
    loss_temperature: float = 0.1

    # Training
    epochs: int = 1000
    lr: float = 1e-3
    t_0: int = 2500
    batches_per_epoch_train: int = 256
    batches_per_epoch_val: int = 64
    samples_per_batch_train: int = 512
    samples_per_batch_val: int = 512

    # Preprocessing
    clip_percentile_lo: float = 0.01   # lower percentile for timing feature clipping
    clip_percentile_hi: float = 99.99   # upper percentile for timing feature clipping

    # Data path
    scenario: str = 'desktop'  # 'desktop' | 'mobile'
    file_path: str = ''
    predict_file_path: str = ''

    seed: int = 42

# Base config — only override what differs from ExperimentConfig defaults.
SWEEP_BASE = ExperimentConfig(
    model=TransformerConfig(),
)

# W&B native sweep config — passed to run_sweep_agent() in train.py.
WANDB_SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "val/eer", "goal": "minimize"},
    "parameters": {
        "loss_type":        {"values": ["supcon"]},
        "model_type":       {"values": ["lstm"]},
        "loss_temperature": {"values": [0.07]},
        "use_mste":         {"values": [True]},
        "model.n_periods":  {"values": [32]},
        "scenario":         {"values": ["mobile", "desktop"]},
    },
}