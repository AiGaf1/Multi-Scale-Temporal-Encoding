"""KVC-onGoing data: timing-feature extraction, padding, and the training / test datasets."""
import random

import pytorch_lightning as pl
from torch.utils.data import DataLoader

import numpy as np

from typing import Dict, List, Tuple, Any
import torch

def extract_timing_features(session):
    """
    Extract keystroke dynamics features from a session.
    session: np.ndarray of shape (N, 3) with columns [press_time, release_time, key_code]
    """

    press = session[:, 0].astype(np.float64) / 1000.0
    release = session[:, 1].astype(np.float64) / 1000.0
    ascii_f = session[:, 2].astype(int)

    valid_mask = (ascii_f >= 0) & (ascii_f <= 255)
    press = press[valid_mask]
    release = release[valid_mask]
    ascii_f = ascii_f[valid_mask]

    hold = release - press
    flight_time = np.zeros_like(press)
    flight_time[1:] = press[1:] - release[:-1]

    return np.column_stack([hold, flight_time, ascii_f])

def pad_session(session, sequence_length, start_zero=True):
    T, D = session.shape
    mask = np.zeros(sequence_length, dtype=np.bool_)

    if T >= sequence_length:
        start_idx = 0 if start_zero else np.random.randint(0, T - sequence_length + 1)
        mask[:] = True
        return session[start_idx:start_idx + sequence_length], mask

    padded = np.zeros((sequence_length, D), dtype=session.dtype)
    padded[:T] = session
    mask[:T] = 1 #True
    return padded, mask


def compute_timing_ranges(dataset, precomputed: bool = False,
                            clip_percentile_lo: float = 0.01,
                            clip_percentile_hi: float = 99.99):
    """
    dataset: dictionary {user_id: {session_id: np.ndarray}}
    Returns: dict with (min, max) per feature after percentile clipping.
    """
    hold_values = []
    flight_values = []

    for user_sessions in dataset.values():
        for session in user_sessions.values():
            feats = session if precomputed else extract_timing_features(session)
            hold_values.append(feats[:, 0])
            flight_values.append(feats[:, 1])

    clip_dict = {}
    for name, values in (('hold', hold_values), ('flight', flight_values)):
        col = np.concatenate(values)
        pos_vals = col[col > 0]
        lo = float(np.percentile(pos_vals, clip_percentile_lo))
        hi = float(np.percentile(col,      clip_percentile_hi))
        min_val = max(lo, 1e-6)  # guard against zero/negative after clipping
        max_val = hi

        clip_dict[name] = {"min": min_val, "max": max_val}

        print(f"{name}: min = {min_val:.6f}, max = {max_val:.6f}  "
              f"(clipped [{clip_percentile_lo}, {clip_percentile_hi}]%)")

    return clip_dict


SessionData = np.ndarray
UserSessionDict = Dict[Any, Dict[Any, SessionData]]
UserPairs = List[Tuple[Any, Any, Any, Any, Tuple]]

class KeystrokeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        raw_data: np.array,
        predict_file_path: str = None,
        window_size: int,
        samples_per_batch_train: int,
        samples_per_batch_val: int,
        batches_per_epoch_train: int,
        batches_per_epoch_val: int,
        num_workers_train: int = 8,
        num_workers_val: int = 4,
        persistent_workers: bool = True,
        train_val_division: float = 0.8,
        seed: int = 42,
        min_session_length: int = 5,
        min_sessions_per_user: int = 2,
        precomputed_features: bool = False,
        clip_percentile_lo: float = 0.05,
        clip_percentile_hi: float = 99.9,
        use_mste: bool = True,
    ):
        super().__init__()
        self.raw_data = raw_data
        self.predict_file_path = predict_file_path
        self.sequence_length = window_size
        self.samples_per_batch_train = samples_per_batch_train
        self.samples_per_batch_val = samples_per_batch_val
        self.batches_per_epoch_train = batches_per_epoch_train
        self.batches_per_epoch_val = batches_per_epoch_val
        self.num_workers_train = num_workers_train
        self.num_workers_val = num_workers_val
        self.persistent_workers = persistent_workers
        self.train_val_division = train_val_division
        self.seed = seed
        self.clip_percentile_lo = clip_percentile_lo
        self.clip_percentile_hi = clip_percentile_hi
        self.use_mste = use_mste

        self._train_users = None
        self._val_users = None
        self.train_data = None
        self.val_data = None
        self.ds_train = None
        self.ds_val = None
        self.ds_predict = None
        self.min_session_length = min_session_length
        self.min_sessions_per_user = min_sessions_per_user
        self.precomputed_features = precomputed_features

    @staticmethod
    def _filter_data(
            data: UserSessionDict,
            min_session_length: int,
            min_sessions_per_user: int
    ) -> UserSessionDict:
        """
        Filter out:
        1. Sessions shorter than min_session_length
        2. Users with fewer than min_sessions_per_user sessions

        Args:
            data: Nested dict {user_id: {session_id: session_data}}
            min_session_length: Minimum keystrokes per session
            min_sessions_per_user: Minimum sessions per user

        Returns:
            Filtered data dictionary
        """
        # Deep copy to avoid modifying original
        filtered = {u: dict(sessions) for u, sessions in data.items()}

        # Remove short sessions
        for user_id in list(filtered.keys()):
            sessions = filtered[user_id]
            filtered[user_id] = {
                sid: sess for sid, sess in sessions.items()
                if len(sess) >= min_session_length
            }

        # Remove users with too few sessions
        filtered = {
            user_id: sessions for user_id, sessions in filtered.items()
            if len(sessions) >= min_sessions_per_user
        }

        return filtered

    def setup(self, stage: str|None = None) -> None:
        if stage in ("fit", None):
            # Split by users
            self.data = self._filter_data(self.raw_data, self.min_session_length, self.min_sessions_per_user)
            users = list(self.data.keys())
            random.Random(self.seed).shuffle(users)

            n_total = len(users)
            n_train = int(n_total * self.train_val_division)

            self._train_users = users[:n_train]
            self._val_users = users[n_train:]

            print('Train num users', len(self._train_users))
            print('Validation num users', len(self._val_users))

            self.train_data = {u: self.data[u] for u in self._train_users}
            self.val_data = {u: self.data[u] for u in self._val_users}

            self.min_max = compute_timing_ranges(
                self.train_data,
                precomputed=self.precomputed_features,
                clip_percentile_lo=self.clip_percentile_lo,
                clip_percentile_hi=self.clip_percentile_hi,
            )

            clip_min_max = self.min_max if not self.use_mste else None

            self.ds_train = PairDataset(
                self.train_data,
                window_size=self.sequence_length,
                samples_considered_per_epoch=self.batches_per_epoch_train * self.samples_per_batch_train,
                precomputed_features=self.precomputed_features,
                min_max=clip_min_max,
            )

            self.ds_val = PairDataset(
                self.val_data,
                window_size=self.sequence_length,
                samples_considered_per_epoch=self.batches_per_epoch_val * self.samples_per_batch_val,
                precomputed_features=self.precomputed_features,
                deterministic=True,
                min_max=clip_min_max,
            )

        if stage == "predict":
            pred_data = np.load(self.predict_file_path, allow_pickle=True).item()
            self.ds_predict = TestSessionDataset(pred_data, self.sequence_length)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_train,
            batch_size=self.samples_per_batch_train,
            num_workers=self.num_workers_train,
            persistent_workers=self.persistent_workers and self.num_workers_train > 0,
            pin_memory=True,
            prefetch_factor=4 if self.num_workers_train > 0 else None,
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_val,
            batch_size=self.samples_per_batch_val,
            num_workers=self.num_workers_val,
            persistent_workers=self.persistent_workers and self.num_workers_val > 0,
            pin_memory=True,
            prefetch_factor=4 if self.num_workers_val > 0 else None,
            shuffle=False
        )

    def predict_dataloader(self):
        return DataLoader(
            self.ds_predict,
            batch_size=self.samples_per_batch_val,
            num_workers=self.num_workers_val,
            persistent_workers=self.persistent_workers and self.num_workers_val > 0,
            shuffle=False
        )


class PairDataset:
    def __init__(self, dataset, window_size, samples_considered_per_epoch, precomputed_features: bool = False, deterministic: bool = False, min_max: dict = None):
        self.len = samples_considered_per_epoch
        self.deterministic = deterministic

        # Pre-compute features and fixed-length padded tensors once at startup
        self.data: dict[Any, dict[Any, tuple[torch.Tensor, torch.Tensor]]] = {}
        for user, sessions in dataset.items():
            self.data[user] = {}
            for sid, sess in sessions.items():
                feat = sess if precomputed_features else extract_timing_features(sess)
                if min_max is not None:
                    feat = feat.copy()
                    feat[:, 0] = np.clip(feat[:, 0], min_max['hold']['min'],   min_max['hold']['max'])
                    feat[:, 1] = np.clip(feat[:, 1], min_max['flight']['min'], min_max['flight']['max'])
                fixed, mask = pad_session(feat, window_size)
                self.data[user][sid] = (torch.from_numpy(fixed).float(), torch.from_numpy(mask))

        self.user_keys: list = list(self.data.keys())
        self.session_keys: dict[Any, list] = {
            user: list(sids.keys()) for user, sids in self.data.items()
        }
    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx):
        rng = random.Random(idx) if self.deterministic else random
        user_idx_1, user_key_1 = self._random_user(rng)
        sid_1 = self._random_session(user_key_1, rng=rng)
        session_1, mask_1 = self.data[user_key_1][sid_1]

        same_user = rng.random() < 0.5
        label = 0 if same_user else 1

        if same_user:
            user_idx_2, user_key_2 = user_idx_1, user_key_1
            sid_2 = self._random_session(user_key_2, exclude=sid_1, rng=rng)
        else:
            user_idx_2, user_key_2 = self._random_different_user(user_idx_1, rng=rng)
            sid_2 = self._random_session(user_key_2, rng=rng)

        session_2, mask_2 = self.data[user_key_2][sid_2]

        return (
            (session_1, mask_1),
            (session_2, mask_2),
            label,
            (user_key_1, user_key_2),
            (torch.tensor(user_idx_1), torch.tensor(user_idx_2)),
        )

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _random_user(self, rng=random) -> tuple[int, Any]:
        idx = rng.randrange(len(self.user_keys))
        return idx, self.user_keys[idx]

    def _random_different_user(self, exclude_idx: int, rng=random) -> tuple[int, Any]:
        idx = exclude_idx
        while idx == exclude_idx:
            idx = rng.randrange(len(self.user_keys))
        return idx, self.user_keys[idx]

    def _random_session(self, user_id, exclude=None, rng=random) -> Any:
        sessions = self.session_keys[user_id]
        if exclude is not None:
            sessions = [s for s in sessions if s != exclude]
        return rng.choice(sessions)

class TestSessionDataset:
    def __init__(self, dataset, sequence_length):
        self.sequence_length = sequence_length
        self.data = dataset
        self.samples = []

        for session_id in dataset.keys():
            self.samples.append(session_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        session_id = self.samples[index]
        session = self.data[session_id]

        prep_session = extract_timing_features(session)
        fix_session, mask = pad_session(
            prep_session,
            self.sequence_length,
            start_zero=True
        )
        return fix_session, mask, session_id
