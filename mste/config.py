from dataclasses import dataclass, field
from typing import Union


@dataclass
class TransformerConfig:
    num_layers: int = 5
    num_heads: int = 2
    ff_dim: int = 256
    dropout: float = 0.1
    n_periods: int = 32
    use_pos_enc: bool = True


@dataclass
class CNNConfig:
    n_periods: int = 32
    hidden_size: int = 128


@dataclass
class LSTMConfig:
    num_layers: int = 2
    dropout: float = 0.1
    n_periods: int = 32
    hidden_size: int = 128


@dataclass
class ExperimentConfig:
    name: str = "MSTE"

    # Data
    window_size: int = 50
    min_session_length: int = 5
    min_sessions_per_user: int = 2
    train_split: float = 0.80

    # Model
    model_type: str = "lstm"  # "transformer" | "cnn" | "lstm"
    model: Union[TransformerConfig, CNNConfig, LSTMConfig] = field(default_factory=LSTMConfig)
    hidden_size: int = 128
    output_size: int = 256
    use_mste: bool = True
    learnable_mste: bool = True

    # Loss
    loss_type: str = "supcon"  # "supcon" | "triplet"
    loss_temperature: float = 0.07

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

    # Data path (derived from the scenario unless given)
    scenario: str = 'desktop'  # 'desktop' | 'mobile'
    file_path: str = ''
    predict_file_path: str = ''

    seed: int = 42

    def __post_init__(self):
        data_dir = f'data/raw/{self.scenario}'
        if not self.file_path:
            self.file_path = f'{data_dir}/{self.scenario}_dev_set.npy'
        if not self.predict_file_path:
            self.predict_file_path = f'{data_dir}/{self.scenario}_test_sessions.npy'


MODEL_CONFIGS = {
    "transformer": TransformerConfig,
    "cnn": CNNConfig,
    "lstm": LSTMConfig,
}
