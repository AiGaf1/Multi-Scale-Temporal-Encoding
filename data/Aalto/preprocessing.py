import numpy as np


def extract_features_classic(session):
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

def get_session_fixed_length_zero_pad_with_mask(session, sequence_length, start_zero=True):
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


def compute_feature_min_max(dataset, precomputed: bool = False,
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
            feats = session if precomputed else extract_features_classic(session)
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
