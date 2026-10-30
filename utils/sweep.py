from datetime import datetime
from dataclasses import replace
from typing import Callable

import wandb

import config as conf
from utils.tools import setup_logger

logger = setup_logger("sweep")

_ABBREV = {
    "use_pos_enc": "pos", "num_layers": "L",
    "loss_type": "loss", "loss_temperature": "τ", "scenario": "sc", "num_heads": "H",
    "n_periods": "N", "learnable_mste": "lrn",
}


_MODEL_CFG_CLS = {
    "transformer": conf.TransformerConfig,
    "cnn":         conf.CNNConfig,
    "lstm":        conf.LSTMConfig,
}

def _cfg_from_wandb(wc: dict) -> conf.ExperimentConfig:
    import dataclasses
    model_overrides = {k.split("model.", 1)[1]: v for k, v in wc.items() if k.startswith("model.")}
    top_overrides   = {k: v for k, v in wc.items() if not k.startswith("model.")}
    model_type = top_overrides.get("model_type", conf.SWEEP_BASE.model_type)
    model_cls  = _MODEL_CFG_CLS[model_type]
    valid_fields = {f.name for f in dataclasses.fields(model_cls)}
    model_cfg = model_cls(**{k: v for k, v in model_overrides.items() if k in valid_fields})
    scenario = top_overrides.get("scenario", conf.SWEEP_BASE.scenario)
    data_dir = f"data/Aalto/raw/{scenario}"
    file_path = f"{data_dir}/{scenario}_dev_set.npy"
    predict_file_path = f"{data_dir}/{scenario}_test_sessions.npy"
    # keep min_session_length in sync with window_size so short sessions aren't zero-padded
    if "window_size" in top_overrides and "min_session_length" not in top_overrides:
        top_overrides["min_session_length"] = top_overrides["window_size"]
    return replace(conf.SWEEP_BASE, model=model_cfg, file_path=file_path, predict_file_path=predict_file_path, **top_overrides)


def _run_name(wc: dict) -> str:
    parts = [f"{_ABBREV.get(k.split('.')[-1], k.split('.')[-1])}={int(v) if isinstance(v, bool) else v}"
             for k, v in wc.items()]
    return "_".join(parts)


def run_sweep_agent(
    run_experiment: Callable,
    sweep_config: dict = conf.WANDB_SWEEP_CONFIG,
) -> None:
    named_config = {**sweep_config, "name": datetime.now().strftime("%Y%m%d_%H%M")}
    sweep_id = wandb.sweep(named_config, project=conf.SWEEP_BASE.name)
    logger.info(f"Sweep created: {sweep_id}")

    def _agent_fn():
        run = wandb.init()
        wc = dict(run.config)
        run.name = _run_name(wc) + "_" + datetime.now().strftime("%Y%m%d_%H%M")
        run_experiment(_cfg_from_wandb(wc), sweep_run_id=run.id)

    wandb.agent(sweep_id, function=_agent_fn)
