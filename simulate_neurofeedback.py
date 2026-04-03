"""
Simulated real-time neurofeedback: replay each trial with a growing window from cue onset,
run EEGNet each hop, and update a simple left/right probability bar (no LSL hardware).

Uses the same continuous pipeline as ``prepare_motor_imagery_cnn.py`` so channel order and
filtering match training when you trained on tensors built from that script.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from eegnet import EEGNet
from preprocessing import PreprocConfig, load_concatenated_eegbci, make_epochs, preprocess_raw, zscore_time


def main() -> None:
    p = argparse.ArgumentParser(description="Simulated real-time MI neurofeedback (EEGNet).")
    p.add_argument("--checkpoint", type=Path, default=Path("data_cache/eegnet_mi.pt"))
    p.add_argument("--subject", type=int, default=1)
    p.add_argument("--speed", type=float, default=8.0, help="Wall-clock multiplier (higher = faster replay).")
    p.add_argument("--hop-ms", type=float, default=80.0, help="Inference update interval in milliseconds.")
    p.add_argument("--max-trials", type=int, default=6, help="Number of trials to replay (in order).")
    p.add_argument("--pause-between", type=float, default=0.4, help="Seconds between trials.")
    p.add_argument(
        "--causal-norm",
        action="store_true",
        help="Z-score using only samples received so far (more 'online'; may mismatch offline training).",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        help="Non-interactive run (no GUI); use for CI or SSH.",
    )
    args = p.parse_args()

    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    n_ch = ckpt["n_channels"]
    n_times = ckpt["n_times"]
    class_names = ckpt.get("class_names", {0: "left_hand", 1: "right_hand"})
    left_name = class_names.get(0, "left")
    right_name = class_names.get(1, "right")

    model = EEGNet(n_ch, n_times, n_classes=ckpt.get("n_classes", 2)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    cfg = PreprocConfig()
    runs = [4, 8, 12]
    raw = load_concatenated_eegbci(args.subject, runs)
    raw = preprocess_raw(raw, cfg)
    epochs = make_epochs(raw, cfg)
    epochs.crop(tmin=0.0, tmax=4.0)
    sfreq = float(epochs.info["sfreq"])
    hop = max(1, int(round(sfreq * args.hop_ms / 1000.0)))

    volt_trials = epochs.get_data()  # (n_trials, n_ch, n_times)
    labels = epochs.events[:, 2]
    n_show = min(args.max_trials, len(volt_trials))

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 4))
    bar_left = ax.barh([0], [0.5], height=0.35, color="#2a9d8f", label=left_name)
    bar_right = ax.barh([1], [0.5], height=0.35, color="#e76f51", label=right_name)
    ax.set_xlim(0, 1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([left_name, right_name])
    ax.set_xlabel("Estimated probability")
    ax.set_title("Simulated real-time neurofeedback (EEGNet)")
    status = fig.text(0.5, 0.02, "", ha="center", fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)

    for ti in range(n_show):  # Loop through each trial
        trial = volt_trials[ti] # Get the trial data
        true_y = int(labels[ti]) # Get the true label for the trial
        truth_txt = f"ground truth: {class_names.get(true_y, true_y)}" # Display the true label
        # Inner loop: Move forward through time in 80ms increments
        for n in range(hop, n_times + 1, hop):
            window = np.zeros_like(trial)
            window[:, :n] = trial[:, :n] # Only show the AI the data "received so far"
            # Normalization (Crucial for AI stability)
            # If we're using "causal norm" (more online), only normalize what we've seen so far
            # If not, normalize the entire window
            # Normalization makes sure the data has a mean of 0 and a standard deviation of 1
            # This helps the AI learn more effectively
            if args.causal_norm:
                w = np.zeros_like(trial, dtype=np.float32)
                seg = window[:, :n].astype(np.float32)
                mu = seg.mean(axis=-1, keepdims=True)
                sig = np.maximum(seg.std(axis=-1, keepdims=True), 1e-6)
                w[:, :n] = (seg - mu) / sig
                x = w[np.newaxis, ...]
            else:
                x = zscore_time(window)[None, ...]
            with torch.no_grad():
                logits = model(torch.from_numpy(x).to(device))
                # Softmax turns raw numbers into percentages (e.g., 0.80 Left, 0.20 Right)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            # Update the bars to show the estimated probabilities
            bar_left[0].set_width(float(probs[0]))
            bar_right[0].set_width(float(probs[1]))
            t_sec = n / sfreq # Calculate the time in seconds
            # Display the time and the true label
            status.set_text(f"trial {ti + 1}/{n_show}  t={t_sec:.2f}s / {n_times / sfreq:.2f}s  {truth_txt}") 
            #refresh the plot to show the new probabilities
            fig.canvas.draw_idle()
            fig.canvas.flush_events() #process the events (like mouse clicks)
            delay = (hop / sfreq) / max(args.speed, 0.01) #calculate the delay between updates
            time.sleep(delay) #wait for the delay before the next update

        time.sleep(args.pause_between) #wait for the pause between trials

    plt.ioff()
    if args.headless:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    import sys

    if "--headless" in sys.argv:
        import matplotlib

        matplotlib.use("Agg")
    main()
