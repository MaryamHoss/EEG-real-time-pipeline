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
| LSL-style sim (2 terminals) | `lsl_replay_outlet.py` + `lsl_neurofeedback_client.py` | Real LSL inlets/outlets on the LAN |

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

CLI options include `--batch-size`, `--lr`, `--weight-decay`, `--val-frac`, `--seed`, **`--balanced-loss`**, and **`--no-stratify`** (stratified split is on by default). The saved checkpoint is the **epoch with highest `val_acc`** (ties keep the earlier epoch); metadata includes `best_val_acc` and `best_epoch`.

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

## Simulated Lab Streaming Layer (LSL)

Two-process setup similar to a lab: one program **publishes** streams (like an amplifier + task PC), another **subscribes** and runs the decoder.

| Role | Script |
|------|--------|
| Outlet (EEG + markers) | `lsl_replay_outlet.py` |
| Inlet (EEGNet neurofeedback) | `lsl_neurofeedback_client.py` |

Streams are named **`EEGBCI_MI_Replay`** (EEG, `float32`, nominal rate) and **`EEGBCI_MI_Markers`** (irregular `int32`: **1 = T1**, **2 = T2**). Identifiers are centralized in `lsl_config.py`.

**Terminal 1 — outlet** (start first; use `--loop` while developing the client):

```powershell
python lsl_replay_outlet.py --speed 4 --loop
```

Optional: `--tmax 60` replays only the first *N* seconds of recording (markers may be sparse at the very beginning).

**Terminal 2 — client**:

```powershell
python lsl_neurofeedback_client.py --checkpoint data_cache/eegnet_mi.pt
```

Flags: `--hop-ms`, `--speed`, `--resolve-timeout` (default 30 s), `--headless`, `--wall-max` (auto-exit for tests).

**Dependency:** `pylsl` (see `requirements.txt`). On some systems you must install **liblsl** separately or set **`PYLSL_LIB`**; the PyPI wheel often works on Windows.

**Caveat:** This replays **already filtered / referenced** EEG (same as `preprocess_raw`), not raw hardware volts. A full hardware twin would stream raw data and run your filters on the client.

**If the figure stays frozen / “Not responding”:** The main loop must **yield to Matplotlib every iteration**; the client does that with **`--gui-pause`** (default **0.02** s). The decoder also **only resamples a short slice of the buffer** around each trial (not the full 60 s) so **real-time outlet `--speed 1`** does not stall the UI. Trial timing uses **LSL sample timestamps**, not wall clock minus marker time. Increase **`--gui-pause`** if needed; the client’s **`--speed`** is decoder refresh rate, **not** the outlet replay speed (set that on `lsl_replay_outlet.py`).

## Project layout

- `preprocessing.py` — MNE preprocessing and tensor helpers
- `prepare_motor_imagery_cnn.py` — build and save cached MI tensors
- `eegnet.py` — PyTorch EEGNet `(B, C, T)` → logits
- `train_eegnet.py` — short supervised training loop
- `simulate_neurofeedback.py` — replay + visualization (no LSL)
- `lsl_config.py`, `lsl_replay_outlet.py`, `lsl_neurofeedback_client.py` — LSL-style simulator
- `data_cache/` — generated tensors and checkpoints (gitignored except you may add small samples deliberately)

## Notes

- Trial counts per subject/run set are modest; expect noisy validation metrics unless you add subjects, runs, or cross-validation.
- For production streaming, replicate **filtering, reference, and normalization** to match offline training (see `preproc_params_for_streaming`).

### Left vs right: why one class can look easier

Motor-imagery decoding is **not symmetric** in practice. **Event-related desynchronization (ERD)** in sensorimotor rhythms often **ramps up over ~1–2 s after the cue**, so probabilities can be **more stable late in the 4 s window**—what you see in the LSL client is normal. **Right-hand imagery** is also often **noisier or less lateralized** than left in some subjects and montages (C3/C4 SNR, volume conduction, dominance). None of that implies a bug in the labels.

**Training:** `train_eegnet.py` now uses a **stratified** train/val split by default and prints **`val_acc_left` / `val_acc_right`** each epoch so you can see class-specific behavior. Try **`--balanced-loss`** if the model systematically favors one class. For serious evaluation, use **k-fold CV** and more runs/subjects.

