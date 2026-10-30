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
train.py                    train and evaluate (paper configuration by default)
mste/
  config.py                 experiment configuration (paper defaults)
  encoding.py               MSTE, the multi-scale temporal encoding
  encoders/                 LSTM, CNN and Transformer sequence encoders
  data.py                   KVC-onGoing feature extraction and datasets
  losses.py                 SupCon and triplet losses
  training.py               Lightning module, checkpointing and logging
  metrics.py                embedding distances and EER
  benchmark.py              official test comparisons in, prediction scores out
figures/
  fig2_timing_distributions.py
  fig3_mste_components.py
  fig5_eer_vs_n.py
data/raw/                   KVC-onGoing data (not included, see below)
outputs/                    runs, checkpoints and figures (created when you run the code)
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
data/raw/desktop/desktop_dev_set.npy
data/raw/desktop/desktop_test_sessions.npy
data/raw/desktop/desktop_comparisons.txt
data/raw/mobile/mobile_dev_set.npy
data/raw/mobile/mobile_test_sessions.npy
data/raw/mobile/mobile_comparisons.txt
```

The data is not redistributed in this repository.

## Training

```bash
python train.py
```

trains the paper's best configuration (LSTM, MSTE with `N = 32`, SupCon loss with
temperature 0.07, up to 50 keystrokes per session) on desktop and then mobile. Each run
writes to `outputs/MSTE/<timestamp>/`: training/validation metrics in
`metrics.csv`, the best checkpoint in `checkpoints/`, and `<scenario>_predictions.txt`,
the scores for the official test comparisons, ready for submission to the KVC-onGoing
evaluation server.

### Figures

All figure scripts write a PDF and a PNG to `outputs/` and typeset their text with LaTeX
(the `newtx` fonts used by the paper), so a LaTeX installation is required.

```bash
python figures/fig2_timing_distributions.py                          # Fig. 2
python figures/fig3_mste_components.py <path/to/checkpoint.ckpt> --scenario desktop   # Fig. 3
python figures/fig5_eer_vs_n.py                                      # Fig. 5
```

Fig. 2 reads the development sets from `data/raw/`; Fig. 5 uses the final
train/validation EERs of the N-ablation runs, listed in the script.

### Ablations

Command-line options reproduce the ablations (the defaults are the paper's configuration):

| Ablation | Option |
|---|---|
| Encoder | `--model lstm`, `--model transformer`, `--model cnn` |
| MSTE vs. outlier clipping | `--no-mste` |
| Learnable vs. fixed frequencies | `--fixed-frequencies` |
| Number of frequencies | `--n-periods N`, e.g. `2 … 128` |
| Loss | `--loss supcon`, `--loss triplet` |
| Window length | `--window-size 30`, `40`, `50` |
| Device | `--scenario desktop`, `--scenario mobile` |

For example, `python train.py --model cnn --no-mste --scenario desktop` trains the CNN
with outlier clipping on desktop data.

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
