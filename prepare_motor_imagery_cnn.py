"""
End-to-end: EEGBCI **left hand vs right hand** motor-imagery runs (PhysioNet / MNE: 4, 8, 12).

Event codes: **T1 → left fist imagery**, **T2 → right fist imagery** (mapped to y=0 and y=1).
For **hands vs feet** imagery instead, use runs ``[6, 10, 14]``.
"""

from __future__ import annotations

from pathlib import Path

from preprocessing import (
    PreprocConfig,
    epochs_to_numpy,
    load_concatenated_eegbci,
    make_epochs,
    preprocess_raw,
    preproc_params_for_streaming,
    to_cnn_tensors,
)

def main() -> None:
    subject = 1
    runs = [4, 8, 12]  # motor imagery: left vs right hand (not 6/10/14 = hands vs feet)
    cfg = PreprocConfig()

    #download the brainwaves, remove the noise (filtering), and chop the data into trials
    raw = load_concatenated_eegbci(subject, runs)
    raw = preprocess_raw(raw, cfg)
    epochs = make_epochs(raw, cfg)

    # Decoder / streaming window: keep only post-cue segment (baseline already applied).
    #-0.2 to 0 was used for baseline correction, so we only want the 4 seconds after the cue
    epochs.crop(tmin=0.0, tmax=4.0)

    X, y = epochs_to_numpy(epochs)
    X_t, y_t = to_cnn_tensors(X, y, as_torch=True)

    print("Epochs:", len(epochs))
    print("X shape (epochs, channels, times):", tuple(X_t.shape))
    print("y shape:", tuple(y_t.shape), "classes:", sorted(set(y_t.tolist())))
    print("Sampling frequency (Hz):", epochs.info["sfreq"])
    print("Streaming mirror params:", preproc_params_for_streaming(cfg))

    out = Path("data_cache")
    out.mkdir(exist_ok=True)
    import torch

    torch.save(
        {
            "X": X_t,
            "y": y_t,
            "sfreq": epochs.info["sfreq"],
            "class_names": {0: "left_hand", 1: "right_hand"},
            "runs": list(runs),
        },
        out / "mi_subject1_leftright_runs4812.pt",
    )


if __name__ == "__main__":
    main()
