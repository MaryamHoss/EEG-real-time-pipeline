"""
EEG-BCI (PhysioNet) preprocessing, epoching, and CNN-ready tensors for MNE.

Per MNE's EEGBCI notes, runs **6, 10, and 14** are motor-imagery **hands vs feet**
(same paradigm repeated). Annotations use T0/T1/T2; we epoch on T1/T2.

For **left-hand vs right-hand** motor imagery, use runs **4, 8, 12** instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

# --------------------------------
#Configuration & Setup
# --------------------------------

@dataclass(frozen=True)
class PreprocConfig:
    """Settings that offline preprocessing and streaming should share."""

    l_freq: float = 8.0 # Low cut-off: removes slow drifts/sweat artifacts
    h_freq: float = 35.0 # High cut-off: removes high-frequency noise, keeps Mu and Beta waves (motor imagery)
    notch_freq: float | None = 60.0 # Specific filter for US electrical hum (60Hz)
    baseline: tuple[float, float] = (-0.2, 0.0) # Baseline correction: subtracts the mean of the baseline period from the data
    tmin: float = -0.2 # Epoch start time: -0.2s before the cue (cue at 0s)
    tmax: float = 4.0 # Epoch end time: 4s after the cue
    event_id: Mapping[str, int] = field( # Mapping of event labels to integer codes (T1=0 left, T2=1 right)
        default_factory=lambda: {"T1": 0, "T2": 1},
    )

# --------------------------------
#Data Loading
# --------------------------------

def load_concatenated_eegbci(
    subject: int,
    runs: Sequence[int],
    *,
    data_path: Path | str | None = None,
) -> "mne.io.BaseRaw":
    import mne
    from mne.datasets import eegbci
    from mne.io import read_raw_edf

    path = Path(data_path).expanduser() if data_path is not None else Path.home() / "mne_data"
    paths = eegbci.load_data(
        subject,
        runs,
        path=path,
        update_path=False,
        verbose="warning",
    ) # Loads the data from the PhysioNet website and returns a list of raw objects
    raws = [read_raw_edf(p, preload=True, verbose="warning") for p in paths]
    for r in raws: # Standardizes the channels to the same reference (average)
        eegbci.standardize(r)
    raw = mne.concatenate_raws(raws) # Concatenates the raw objects into a single raw object
    return raw

# --------------------------------
#Data Preprocessing, Cleaning the Signal
# --------------------------------
def preprocess_raw(
    raw: "mne.io.BaseRaw",
    config: PreprocConfig | None = None,
) -> "mne.io.BaseRaw":
    """
    Band-pass, optional notch, EEG channel selection, average reference.
    Operates on a copy of ``raw``.
    """
    import mne

    cfg = config or PreprocConfig()
    raw = raw.copy()
    # 1. Keep only EEG sensors (remove stimulus channels or non-brain sensors)
    raw.pick(picks="eeg", exclude="bads")

    # 2. Re-referencing: Subtract the average of all sensors from each sensor.
    # This cancels out noise that is common to the whole head.    
    raw.set_eeg_reference("average", projection=False, verbose="warning")

    # 3. Band-pass Filter: Only allow signals between 8Hz and 35Hz.
    raw.filter(
        cfg.l_freq,
        cfg.h_freq,
        fir_design="firwin",
        skip_by_annotation="edge",
        verbose="warning",
    )
    # 4. Notch Filter: Remove 60Hz electrical hum.
    if cfg.notch_freq is not None:
        raw.notch_filter(freqs=cfg.notch_freq, verbose="warning")
    return raw

# --------------------------------
#Data Epoching
#We don't want the whole 10-minute recording; we want the specific 
#4-second blocks where the user was imagining movement.
# --------------------------------
def make_epochs(
    raw: "mne.io.BaseRaw",
    config: PreprocConfig | None = None,
) -> "mne.Epochs":
    """
    Build fixed-length epochs time-locked to T1/T2 imagery cues.
    """
    import mne
    from mne import Epochs

    cfg = config or PreprocConfig()
    # Find the timestamps where "T1" or "T2" annotations appear in the data
    events, event_id = mne.events_from_annotations(raw, event_id=dict(cfg.event_id))
    # Select only the EEG channels (exclude bad channels)
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    # Create epochs: a fixed-length time-locked to T1/T2 imagery cues.
    epochs = Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=cfg.tmin,
        tmax=cfg.tmax,
        baseline=cfg.baseline,
        picks=picks,
        preload=True,
        reject_by_annotation=True,
        verbose="warning",
    )
    return epochs

# --------------------------------
#Data Conversion to NumPy Arrays and then tensors to pass to the EEGNET
# --------------------------------
def epochs_to_numpy(epochs: "mne.Epochs") -> tuple[np.ndarray, np.ndarray]:
    """
    Return X of shape (n_epochs, n_channels, n_times) and y of shape (n_epochs,).

    y values match the event codes from PreprocConfig.event_id (defaults: 0=T1, 1=T2).
    """
    data = epochs.get_data() # Convert to NumPy array: (Trials, Channels, Time)
    y = epochs.events[:, 2].astype(np.int64) # Extract the event codes (0=T1, 1=T2)
    return data, y


def zscore_time(x: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """
    Per-channel z-score over the last axis (time). Accepts (channels, times)` or
    (epochs, channels, times).
    Normalize data so mean=0 and standard deviation=1.
    """
    if x.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array; got shape {x.shape}")
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return ((x - mean) / std).astype(np.float32)


def to_cnn_tensors(
    X: np.ndarray,
    y: np.ndarray,
    *,
    as_torch: bool = True,
):
    """
    Normalize per epoch per channel (z-score over time) — common before small CNNs on EEG.

    Returns (X_out, y): X_out is float32 with shape (n_epochs, n_channels, n_times).
    """
    if X.ndim != 3:
        raise ValueError(f"X must be (epochs, chans, times); got {X.shape}")
    Xn = zscore_time(X)
    y = y.astype(np.int64)
    if as_torch:
        import torch

        return torch.from_numpy(Xn), torch.from_numpy(y)
    return Xn, y


def preproc_params_for_streaming(config: PreprocConfig | None = None) -> dict:
    """
    Parameters to mirror in a streaming path (same bands and epoch window in samples).

    Online you typically apply the same IIR/FIR design on overlapping windows rather
    than calling ``raw.filter`` on the full recording.
    """
    cfg = config or PreprocConfig()
    return {
        "l_freq": cfg.l_freq,
        "h_freq": cfg.h_freq,
        "notch_freq": cfg.notch_freq,
        "baseline_sec": cfg.baseline,
        "epoch_tmin_sec": cfg.tmin,
        "epoch_tmax_sec": cfg.tmax,
        "event_labels": dict(cfg.event_id),
    }
