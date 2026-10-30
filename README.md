# Multi-Scale Temporal Encoding (MSTE) for Keystroke Verification

Code for the paper **"Multi-Scale Temporal Encoding for Keystroke Biometric Verification Toward Continuous Authentication"**
(A. Gaffarov, F. Ajmi, A. B. Karami, B. Zouari — CRiSIS 2026, Springer LNCS).

MSTE maps each keystroke's hold time (HT) and flight time (FT) to sinusoidal features
`[sin(ω_j t), cos(ω_j t)]` for `j = 1..N`, where the frequencies `ω_j` start log-spaced over the
observed timing range and are rescaled by a learnable sigmoid gate. The encoding is
concatenated with a 16-dimensional key-code embedding and fed to a sequence encoder
(LSTM, Transformer, or CNN) trained with a metric-learning loss.

On the official KVC-onGoing test set, LSTM + MSTE + SupCon loss reaches a global EER of
**1.18%** (desktop) and **2.32%** (mobile).

## Repository layout

```
config.py                 experiment configuration and W&B sweep grid
train.py                  training, validation and test-set prediction
models/fourier_encoding.py  MSTE (LearnableFourierFeatures)
models/{lstm,transformer,cnn}.py  sequence encoders
models/factory.py         builds the encoder selected in config.py
models/lit_model.py       PyTorch Lightning module
utils/losses.py           SupCon and triplet losses
data/Aalto/               KVC-onGoing data loading and preprocessing
plot_mste_explanation.py  reproduces the MSTE illustration (Fig. 3 of the paper)
```

## Setup

```bash
pip install -r requirements.txt
```

The paper's experiments used Python 3, PyTorch 2.7.0 and PyTorch Lightning 2.5.1 on a single
NVIDIA RTX 4080 Super GPU.

## Data

The experiments use the KVC-onGoing benchmark (desktop and mobile), hosted on CodaLab.
Download the data from the challenge page and place the files as:

```
data/Aalto/raw/desktop/desktop_dev_set.npy
data/Aalto/raw/desktop/desktop_test_sessions.npy
data/Aalto/raw/desktop/desktop_comparisons.txt
data/Aalto/raw/mobile/mobile_dev_set.npy
data/Aalto/raw/mobile/mobile_test_sessions.npy
data/Aalto/raw/mobile/mobile_comparisons.txt
```

The data is not redistributed in this repository.

## Training

Training runs as a [Weights & Biases](https://wandb.ai) grid sweep defined by
`WANDB_SWEEP_CONFIG` in `config.py`. Put your API key in a `.env` file:

```
WANDB_API_KEY=<your key>
```

then run

```bash
python train.py
```

The default sweep trains the paper's best configuration (LSTM, MSTE with `N = 32`,
SupCon loss with temperature 0.07, 50-keystroke windows) on both desktop and mobile.
After training, the best checkpoint scores the official test comparisons and writes
`<scenario>_predictions.txt` to the run directory, ready for submission to the
KVC-onGoing evaluation server.

### Ablations

Change the sweep grid in `config.py` to reproduce the ablations:

| Ablation | Setting |
|---|---|
| Encoder | `model_type`: `"lstm"`, `"transformer"`, `"cnn"` |
| MSTE vs. outlier clipping | `use_mste`: `True` / `False` |
| Learnable vs. fixed frequencies | `learnable_mste`: `True` / `False` |
| Number of frequencies | `model.n_periods`: e.g. `2 … 128` |
| Loss | `loss_type`: `"supcon"` / `"triplet"` |
| Window length | `window_size`: e.g. `30`, `40`, `50` |

## Citation

```bibtex
@inproceedings{gaffarov2026mste,
  title     = {Multi-Scale Temporal Encoding for Keystroke Biometric Verification Toward Continuous Authentication},
  author    = {Gaffarov, Aidar and Ajmi, Faiza and Karami, Abir B. and Zouari, Belhassen},
  booktitle = {Risks and Security of Internet and Systems (CRiSIS 2026)},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
