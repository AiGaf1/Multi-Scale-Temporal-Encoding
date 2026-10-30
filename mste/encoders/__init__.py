"""Sequence encoders (LSTM, CNN, Transformer) that turn MSTE-encoded keystrokes into embeddings."""
import torch.nn as nn
from mste import config as conf
from mste.encoders.transformer import KeystrokeTransformer
from mste.encoders.cnn import KeystrokeCNN
from mste.encoders.lstm import KeystrokeLSTM


def build_encoder(config: conf.ExperimentConfig, periods_dict) -> nn.Module:
    """Instantiate the model specified by config.model_type."""
    if config.model_type == "transformer":
        cfg: conf.TransformerConfig = config.model
        return KeystrokeTransformer(
            periods_dict=periods_dict,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            window_size=config.window_size,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            ff_dim=cfg.ff_dim,
            dropout=cfg.dropout,
            n_periods=cfg.n_periods,
            use_pos_enc=cfg.use_pos_enc,
            use_mste=config.use_mste,
            learnable_mste=config.learnable_mste,
        )
    elif config.model_type == "cnn":
        cfg: conf.CNNConfig = config.model
        return KeystrokeCNN(
            periods_dict=periods_dict,
            hidden_size=cfg.hidden_size,
            output_size=config.output_size,
            n_periods=cfg.n_periods,
            use_mste=config.use_mste,
            learnable_mste=config.learnable_mste,
        )
    elif config.model_type == "lstm":
        cfg: conf.LSTMConfig = config.model
        return KeystrokeLSTM(
            periods_dict=periods_dict,
            hidden_size=cfg.hidden_size,
            output_size=config.output_size,
            n_periods=cfg.n_periods,
            use_mste=config.use_mste,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            learnable_mste=config.learnable_mste,
        )
    else:
        raise ValueError(
            f"Unknown model_type: '{config.model_type}'. Expected 'transformer', 'cnn', or 'lstm'."
        )
