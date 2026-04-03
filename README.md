# EEG real-time pipeline (EEGBCI + EEGNet)

End-to-end workflow for **PhysioNet EEG Motor Movement/Imagery** data (via MNE): download standardized runs, **preprocess** and **epoch** left/right **motor imagery**, train a compact **EEGNet** classifier, and run a **simulated real-time neurofeedback** loop (matplotlib bars, no hardware).

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On first run, MNE downloads EEGBCI files under `%USERPROFILE%\mne_data` (non-interactive; no config prompt).

## Pipeline overview

| Step | Script | Output |
|------|--------|--------|
| Preprocess + cache tensors | `prepare_motor_imagery_cnn.py` | `data_cache/mi_subject1_leftright_runs4812.pt` |
| Train EEGNet | `train_eegnet.py` | `data_cache/eegnet_mi.pt` |
| Simulated neurofeedback | `simulate_neurofeedback.py` | Live figure (or `--headless`) |

**Left vs right hand** motor imagery uses EEGBCI runs **`4, 8, 12`** (see MNE dataset table). Runs **`6, 10, 14`** are **hands vs feet** imagery, not left/right.

Events **T1** and **T2** are mapped to labels **0** and **1** (saved as `left_hand` / `right_hand` in the `.pt` bundle).

## Preprocessing (library)

`preprocessing.py` exposes:

- `PreprocConfig` — bandpass (default 8–35 Hz), notch (60 Hz), epoching, `event_id` for T1/T2
- `load_concatenated_eegbci` — fetch runs, `standardize` channels, concatenate
- `preprocess_raw` — EEG picks, average reference, filter, notch
- `make_epochs` — annotations → `mne.Epochs`
- `epochs_to_numpy`, `to_cnn_tensors`, `zscore_time` — NumPy / Torch with per-channel z-score over time
- `preproc_params_for_streaming` — dict of settings to mirror online

The example script crops epochs to **0–4 s** after the cue (641 samples at **160 Hz**).

## Train EEGNet

```powershell
python prepare_motor_imagery_cnn.py
python train_eegnet.py --epochs 15 --data data_cache/mi_subject1_leftright_runs4812.pt --out data_cache/eegnet_mi.pt
```

CLI options include `--batch-size`, `--lr`, `--weight-decay`, `--val-frac`, `--seed`.

## Simulated neurofeedback

Replays trials with a **growing window** from cue onset, runs the trained net every **`--hop-ms`**, and updates **left/right probability** bars.

```powershell
python simulate_neurofeedback.py
```

Useful flags:

- `--speed` — wall-clock multiplier (higher = faster replay)
- `--hop-ms` — update interval (default 80 ms)
- `--max-trials` — number of trials to replay
- `--causal-norm` — z-score only on data received so far (more “online”; may differ from offline training)
- `--headless` — no GUI (non-interactive backend)

This is **not** a Lab Streaming Layer (LSL) client; it replays preprocessed EEGBCI data to mimic decoder timing and feedback.

## Project layout

- `preprocessing.py` — MNE preprocessing and tensor helpers
- `prepare_motor_imagery_cnn.py` — build and save cached MI tensors
- `eegnet.py` — PyTorch EEGNet `(B, C, T)` → logits
- `train_eegnet.py` — short supervised training loop
- `simulate_neurofeedback.py` — replay + visualization
- `data_cache/` — generated tensors and checkpoints (gitignored except you may add small samples deliberately)

## Notes

- Trial counts per subject/run set are modest; expect noisy validation metrics unless you add subjects, runs, or cross-validation.
- For production streaming, replicate **filtering, reference, and normalization** to match offline training (see `preproc_params_for_streaming`).
