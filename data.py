"""Data loading, preprocessing, handcrafted features and subject-wise splits.

Two data sources share one interface and return (X, y, patient_ids):

* ``synthetic``: generated ECG-like beats (PQRST built from Gaussians). It needs no
  download and has KNOWN morphology per class, so it is useful for testing the
  pipeline and for checking that SHAP highlights the right part of the beat.
  It is NOT a substitute for real data in your write-up.
* ``mitbih``: MIT-BIH Arrhythmia Database via ``wfdb`` (downloaded from PhysioNet).

Beats are 180 samples at 360 Hz (250 ms either side of the R peak), so the R peak
is always at index ``HALF``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import butter, detrend, sosfiltfilt
from scipy.stats import kurtosis, skew
from sklearn.model_selection import GroupShuffleSplit

FS = 360
HALF = 90
BEAT_LEN = 2 * HALF
CLASS_NAMES = ["N (normal)", "S (supraventricular)", "V (ventricular)"]
SHORT = ["N", "S", "V"]

# Approximate beat regions (sample index, R peak at HALF=90). Contiguous partition.
REGIONS = {"P/PR (0-68)": (0, 68), "QRS (68-112)": (68, 112), "ST/T (112-180)": (112, 180)}

# De Chazal et al. inter-patient split of MIT-BIH (paced records excluded).
MITBIH_DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
              201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
MITBIH_DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
              213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
# AAMI grouping of MIT-BIH beat annotations (F and Q classes are dropped here).
AAMI = {"N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
        "A": 1, "a": 1, "J": 1, "S": 1,
        "V": 2, "E": 2}


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def bandpass(sig: np.ndarray, fs: float, lo: float = 0.5, hi: float = 40.0) -> np.ndarray:
    sos = butter(3, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, sig)


def preprocess_beats(X: np.ndarray) -> np.ndarray:
    """Per-beat linear detrend + z-score (same for every dataset)."""
    X = detrend(X, axis=1, type="linear")
    X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
    return X.astype(np.float32)


# --------------------------------------------------------------------------- #
# Synthetic beats
# --------------------------------------------------------------------------- #
def _g(t, mu, amp, sigma):
    return amp * np.exp(-0.5 * ((t - mu) / sigma) ** 2)


def make_synthetic(n_patients: int = 30, beats_per_patient: int = 400, seed: int = 0,
                   class_probs=(0.8, 0.1, 0.1)):
    """Synthetic beats with patient-specific morphology.

    N: normal P-QRS-T.  S: absent / weak / early P wave, narrow QRS.
    V: no P wave, wide tall QRS, inverted T wave.
    Patient-level gain, noise, baseline wander and wave shapes make a
    subject-wise split meaningfully harder than a random split.
    """
    rng = np.random.default_rng(seed)
    t0 = (np.arange(BEAT_LEN) - HALF) / FS
    n = n_patients * beats_per_patient
    X = np.empty((n, BEAT_LEN), dtype=np.float64)
    y = np.empty(n, dtype=np.int64)
    pid = np.empty(n, dtype=np.int64)
    k = 0
    for p in range(n_patients):
        gain = rng.uniform(0.7, 1.4)
        noise = rng.uniform(0.02, 0.07)
        wander = rng.uniform(0.05, 0.25)
        qrs_w = rng.uniform(0.010, 0.015)
        p_amp = rng.uniform(0.12, 0.25)
        t_amp = rng.uniform(0.20, 0.40)
        for _ in range(beats_per_patient):
            c = int(rng.choice(3, p=class_probs))
            t = t0 - rng.uniform(-0.006, 0.006)  # small R-peak jitter
            if c in (0, 1):
                if c == 0:
                    P = _g(t, -0.16 + rng.normal(0, 0.008), p_amp * rng.uniform(0.85, 1.15), 0.025)
                else:
                    scale = rng.uniform(0.0, 0.3) if rng.random() < 0.75 else rng.uniform(0.4, 0.7)
                    P = _g(t, rng.uniform(-0.12, -0.07), p_amp * scale, 0.025)
                Q = _g(t, -0.03, -0.12, 0.010)
                R = _g(t, 0.0, rng.uniform(0.9, 1.1), qrs_w)
                S = _g(t, 0.03, -0.22, 0.012)
                T = _g(t, 0.20 + rng.normal(0, 0.01), t_amp, 0.045)
            else:
                w = qrs_w * rng.uniform(2.2, 3.2)
                P = _g(t, -0.16, p_amp * rng.uniform(0.0, 0.1), 0.025)
                Q = 0.0
                R = _g(t, 0.0, rng.uniform(1.1, 1.5), w)
                S = _g(t, 0.045, -rng.uniform(0.3, 0.6), w * 1.2)
                T = _g(t, 0.18, -t_amp * rng.uniform(0.8, 1.5), 0.06)
            beat = gain * (P + Q + R + S + T)
            beat = beat + wander * np.sin(2 * np.pi * rng.uniform(0.2, 0.5) * t + rng.uniform(0, 2 * np.pi))
            beat = beat + rng.normal(0, noise, BEAT_LEN)
            X[k], y[k], pid[k] = beat, c, p
            k += 1
    return preprocess_beats(X), y, pid


# --------------------------------------------------------------------------- #
# MIT-BIH
# --------------------------------------------------------------------------- #
def load_mitbih(data_dir: str = "data/mitdb", download: bool = True):
    import wfdb  # imported lazily so the synthetic path works without it

    data_dir = Path(data_dir)
    if download and not (data_dir / "100.dat").exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        wfdb.dl_database("mitdb", str(data_dir))
    beats, labels, pids = [], [], []
    for rec in MITBIH_DS1 + MITBIH_DS2:
        path = str(data_dir / str(rec))
        r = wfdb.rdrecord(path)
        if "MLII" not in r.sig_name:
            continue
        sig = bandpass(r.p_signal[:, r.sig_name.index("MLII")], r.fs)
        ann = wfdb.rdann(path, "atr")
        for s, sym in zip(ann.sample, ann.symbol):
            if sym in AAMI and s - HALF >= 0 and s + HALF <= len(sig):
                beats.append(sig[s - HALF:s + HALF])
                labels.append(AAMI[sym])
                pids.append(rec)
    return preprocess_beats(np.asarray(beats)), np.asarray(labels), np.asarray(pids)


def load_data(name: str, **kw):
    if name == "synthetic":
        return make_synthetic(**kw)
    if name == "mitbih":
        return load_mitbih(**kw)
    raise ValueError(f"unknown dataset {name!r}")


# --------------------------------------------------------------------------- #
# Subject-wise splits
# --------------------------------------------------------------------------- #
def make_splits(pids: np.ndarray, dataset: str, seed: int = 0):
    """Train / val / test indices with NO patient shared between sets.

    MIT-BIH: test = DS2 (fixed inter-patient protocol); val is carved out of DS1 by patient.
    Synthetic: random patients.
    """
    pids = np.asarray(pids)
    if dataset == "mitbih":
        test_mask = np.isin(pids, MITBIH_DS2)
        trainval, test = np.where(~test_mask)[0], np.where(test_mask)[0]
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        trainval, test = next(gss.split(pids, groups=pids))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, va = next(gss.split(trainval, groups=pids[trainval]))
    train, val = trainval[tr], trainval[va]
    sets = [set(pids[i]) for i in (train, val, test)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), "patient leakage!"
    return train, val, test


# --------------------------------------------------------------------------- #
# Handcrafted features for the baseline (and TreeSHAP)
# --------------------------------------------------------------------------- #
FEATURE_NAMES = [
    "r_amp", "qrs_width", "q_depth", "s_depth", "qrs_energy",
    "p_amp", "p_energy", "t_amp", "t_energy",
    "skew", "kurtosis", "pow_0_5Hz", "pow_5_15Hz", "pow_15_40Hz",
]


def extract_features(X: np.ndarray) -> np.ndarray:
    n = len(X)
    idx = np.arange(n)
    peak = X[:, HALF - 5:HALF + 5].max(1)
    qrs_win = X[:, HALF - 30:HALF + 30]
    t_seg = X[:, 130:175]
    spec = np.abs(np.fft.rfft(X, axis=1)) ** 2
    freqs = np.fft.rfftfreq(X.shape[1], 1 / FS)
    total = spec.sum(1) + 1e-8

    def band(lo, hi):
        return spec[:, (freqs >= lo) & (freqs < hi)].sum(1) / total

    feats = [
        peak,
        (qrs_win > 0.5 * peak[:, None]).sum(1),
        X[:, HALF - 30:HALF].min(1),
        X[:, HALF:HALF + 30].min(1),
        (X[:, 70:112] ** 2).sum(1),
        X[:, 15:55].max(1),
        (X[:, 15:55] ** 2).sum(1),
        t_seg[idx, np.abs(t_seg).argmax(1)],
        (t_seg ** 2).sum(1),
        skew(X, axis=1),
        kurtosis(X, axis=1),
        band(0, 5), band(5, 15), band(15, 40),
    ]
    return np.stack(feats, axis=1).astype(np.float32)
