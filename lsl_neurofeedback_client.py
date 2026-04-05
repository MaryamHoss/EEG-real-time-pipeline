"""
LSL **inlet** client: resolve EEG + marker streams (from ``lsl_replay_outlet.py``), buffer
samples in LSL time, and run EEGNet neurofeedback after each T1/T2 marker.

Run **after** the outlet has started (same PC is fine).

This mirrors a typical lab layout: one process owns the amplifier stream, another owns the
decoder / feedback UI.
"""

from __future__ import annotations

import argparse
import bisect
import time
from pathlib import Path

import numpy as np
import torch

from eegnet import EEGNet
from lsl_config import EEG_STREAM_NAME, MARKER_STREAM_NAME, MARKER_T1, MARKER_T2
from preprocessing import zscore_time


def _pretty_label(raw: str) -> str:
    return str(raw).replace("_", " ").strip().title()


class _SampleBuffer:
    """Growing buffer of (timestamp, sample row); trims old data by wall span.
    LSL sends data in bursts. We need a place to store the last 60 seconds of data so we can "look back"
     in time when a marker arrives.
    """

    def __init__(self, n_ch: int, keep_sec: float = 60.0) -> None:
        self.n_ch = n_ch
        self.keep_sec = keep_sec # 60 seconds of data
        self._t: list[float] = [] # List of timestamps
        self._x: list[np.ndarray] = [] # A list to hold the actual brainwave voltage values.

    # This function is called when new data comes in from the LSL stream.
    # It adds the new data to the buffer.
    def extend(self, timestamps: list[float], samples: list[list[float]]) -> None:
        if not timestamps: #If no new data arrived, do nothing.
            return 
        for t, row in zip(timestamps, samples, strict=True): #Loop through the new data
            self._t.append(float(t))
            self._x.append(np.asarray(row, dtype=np.float32))
        self._trim()
    # deletes data older than 60 seconds so the computer doesn't run out of RAM
    def _trim(self) -> None:
        if not self._t: #If buffer is empty, stop.
            return
        t_hi = self._t[-1] #Find the timestamp of the very latest sample.
        t_lo = t_hi - self.keep_sec #Find the timestamp of the oldest sample from the cutoff time (60 sec)
        i0 = 0
        while i0 < len(self._t) and self._t[i0] < t_lo: #Loop through the timestamps and find the index of the oldest sample
            i0 += 1
        if i0: #If we found an oldest sample, delete all samples older than it.
            self._t = self._t[i0:]
            self._x = self._x[i0:]

    #Network data is messy. Sometimes samples arrive slightly late or early.
    #Neural Networks (EEGNet) require a perfectly steady rhythm (e.g., exactly 160.00 Hz).
    #This function looks at the messy timestamps and "re-aligns" them to a perfect grid so the AI sees a clean signal.
    def nearest_on_grid(self, t0: float, n: int, sfreq: float) -> np.ndarray | None:
        """Return (n_ch, n) by nearest-neighbour to regular grid t0 + arange(n)/sfreq."""
        if n <= 0 or not self._t:
            return None
        # Only stack samples in [t0, t_end] — stacking the full 60 s buffer every hop freezes the UI at real-time speed.
        dt = 1.0 / float(sfreq)
        margin = 4.0 * dt
        t_window_lo = t0 - margin
        t_window_hi = t0 + (n - 1) * dt + margin
        lo = bisect.bisect_left(self._t, t_window_lo)
        hi = bisect.bisect_right(self._t, t_window_hi)
        if hi <= lo:
            return None
        t_arr = np.asarray(self._t[lo:hi], dtype=np.float64)
        x_arr = np.stack(self._x[lo:hi], axis=0)
        grid = t0 + np.arange(n, dtype=np.float64) / float(sfreq)
        idx = np.searchsorted(t_arr, grid, side="left")
        idx = np.clip(idx, 0, len(t_arr) - 1)
        idx_prev = np.clip(idx - 1, 0, len(t_arr) - 1)
        use_prev = np.abs(t_arr[idx_prev] - grid) <= np.abs(t_arr[idx] - grid)
        pick = np.where(use_prev, idx_prev, idx)
        return x_arr[pick].T.astype(np.float32)


def _resolve_inlet(name: str, timeout: float): # This function finds the LSL stream we want to use
    import pylsl

    deadline = time.perf_counter() + timeout # Set a deadline for the search
    while time.perf_counter() < deadline:
        wait = min(1.0, deadline - time.perf_counter())
        if wait <= 0:
            break
        for s in pylsl.resolve_streams(wait_time=wait): # Search for the stream
            if s.name() == name:
                # Smaller buffer → less burst CPU when catching up; 90 s >> one 4 s trial + margin.
                return pylsl.StreamInlet(s, max_buflen=90, max_chunklen=0)
        time.sleep(0.05)
    raise RuntimeError(
        f"No LSL stream named {name!r} within {timeout:.1f}s. Start lsl_replay_outlet.py first."
    )


def main() -> None:
    p = argparse.ArgumentParser(description="LSL neurofeedback client (EEGNet).")
    p.add_argument("--checkpoint", type=Path, default=Path("data_cache/eegnet_mi.pt"))
    p.add_argument("--resolve-timeout", type=float, default=30.0)
    p.add_argument("--hop-ms", type=float, default=80.0)
    p.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="How often to re-run the decoder vs wall clock (higher = more frequent updates). "
        "Not the LSL outlet replay rate — that is set on lsl_replay_outlet.py --speed.",
    )
    p.add_argument("--headless", action="store_true")
    p.add_argument("--wall-max", type=float, default=None, help="Exit after N wall seconds (headless smoke test).")
    p.add_argument(
        "--gui-pause",
        type=float,
        default=0.02,
        help="Seconds to yield to Matplotlib each loop (prevents 'Not Responding' on Windows).",
    )
    args = p.parse_args()

    if args.headless:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pylsl

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    n_ch = ckpt["n_channels"]
    n_times = ckpt["n_times"]
    class_names = ckpt.get("class_names", {0: "left_hand", 1: "right_hand"})

    model = EEGNet(n_ch, n_times, n_classes=ckpt.get("n_classes", 2)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Resolving LSL streams '{EEG_STREAM_NAME}' and '{MARKER_STREAM_NAME}'...")
    inlet_eeg = _resolve_inlet(EEG_STREAM_NAME, args.resolve_timeout)
    inlet_mk = _resolve_inlet(MARKER_STREAM_NAME, args.resolve_timeout)
    info = inlet_eeg.info()
    sfreq = float(info.nominal_srate()) #Find out how fast the "headset" is sending data 
    if abs(sfreq) < 1e-6:
        raise RuntimeError("EEG stream has invalid nominal sampling rate.")
    hop_sec = max(args.hop_ms / 1000.0, 1.0 / sfreq) #This is the time between each inference.
    speed = max(args.speed, 0.01) #This is the speed of the UI vs the wall clock.

    buf = _SampleBuffer(n_ch)
    left_name = class_names.get(0, "left")
    right_name = class_names.get(1, "right")
    left_pretty = _pretty_label(left_name)
    right_pretty = _pretty_label(right_name)

    plt.ion()
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    fig.patch.set_facecolor("#fafafa")
    # Highlight row that matches PhysioNet cue (T1 = left, T2 = right in this pipeline).
    span_left = ax.axhspan(
        -0.28,
        0.28,
        facecolor="#c8ebd4",
        edgecolor="#1b7f6a",
        linewidth=2.5,
        zorder=0,
        visible=False,
        alpha=0.85,
    )
    span_right = ax.axhspan(
        0.72,
        1.28,
        facecolor="#f5d0c5",
        edgecolor="#c44c32",
        linewidth=2.5,
        zorder=0,
        visible=False,
        alpha=0.85,
    )
    bar_left = ax.barh([0], [0.5], height=0.35, color="#2a9d8f", label=left_pretty, zorder=2)
    bar_right = ax.barh([1], [0.5], height=0.35, color="#e76f51", label=right_pretty, zorder=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.45, 1.45)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([f"{left_pretty}\n(model output)", f"{right_pretty}\n(model output)"], fontsize=10)
    ax.set_xlabel("Model probability (softmax)")
    ax.set_title("LSL neurofeedback (EEGNet)", fontsize=11, pad=8)
    truth_title = fig.text(
        0.5,
        0.94,
        "",
        ha="center",
        va="top",
        fontsize=17,
        fontweight="bold",
        color="#1a1a1a",
        transform=fig.transFigure,
    )
    truth_sub = fig.text(
        0.5,
        0.875,
        "",
        ha="center",
        va="top",
        fontsize=11,
        color="#333333",
        transform=fig.transFigure,
    )
    pred_line = fig.text(
        0.5,
        0.815,
        "",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="semibold",
        color="#444444",
        transform=fig.transFigure,
    )
    status = fig.text(0.5, 0.03, "", ha="center", fontsize=9, color="#555555", transform=fig.transFigure)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.72, bottom=0.12)

    def set_truth_visual(label: int | None, cue_name: str = "") -> None:
        if label is None:
            span_left.set_visible(False)
            span_right.set_visible(False)
            truth_title.set_text("Ground truth: —")
            truth_title.set_color("#666666")
            truth_sub.set_text("Waiting for a T1 / T2 marker on the LSL stream.")
            pred_line.set_text("")
            return
        if label == 0:
            span_left.set_visible(True)
            span_right.set_visible(False)
            truth_title.set_text(f"GROUND TRUTH: {left_pretty.upper()}")
            truth_title.set_color("#0d5c4a")
        else:
            span_left.set_visible(False)
            span_right.set_visible(True)
            truth_title.set_text(f"GROUND TRUTH: {right_pretty.upper()}")
            truth_title.set_color("#9b2c1a")
        truth_sub.set_text(
            f"Cue on stream: {cue_name}  (T1 → {left_pretty}, T2 → {right_pretty})"
        )

    trial_t0: float | None = None
    trial_label: int | None = None
    trial_name = ""
    # Wall-clock pacing for redraws (do not use local_clock() vs marker time — stream
    # timestamps can run ahead of wall clock when the outlet uses --speed > 1).
    next_infer_wall = 0.0
    wall_start = time.perf_counter()

    print("Listening... (start the replay outlet if you have not yet).")
    set_truth_visual(None)
    status.set_text("Buffering LSL: waiting for EEG chunks and T1/T2 markers.")
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.05)

    try:
        while True:
            if args.wall_max is not None and (time.perf_counter() - wall_start) > args.wall_max:
                break

            samples, ts = inlet_eeg.pull_chunk(timeout=0.0, max_samples=1024)
            buf.extend(ts, samples)

            ms, mt = inlet_mk.pull_chunk(timeout=0.0, max_samples=256)
            if ms:
                for row, tm in zip(ms, mt, strict=True):
                    code = int(row[0])
                    if code == MARKER_T1:
                        trial_t0 = tm
                        trial_label = 0
                        trial_name = str(class_names.get(0, "T1"))
                        next_infer_wall = time.perf_counter()
                        set_truth_visual(0, cue_name="T1")
                    elif code == MARKER_T2:
                        trial_t0 = tm
                        trial_label = 1
                        trial_name = str(class_names.get(1, "T2"))
                        next_infer_wall = time.perf_counter()
                        set_truth_visual(1, cue_name="T2")

            wall_now = time.perf_counter()
            if trial_t0 is not None and wall_now >= next_infer_wall:
                next_infer_wall = wall_now + hop_sec / speed
                elapsed = 0.0
                if buf._t:
                    # Stream timestamps from the outlet — not wall clock (see module docstring).
                    elapsed = max(0.0, buf._t[-1] - trial_t0)
                n_have = min(n_times, int(np.floor(elapsed * sfreq)))
                n_have = max(0, min(n_have, n_times))
                if n_have > 0:
                    grid_part = buf.nearest_on_grid(trial_t0, n_have, sfreq)
                    if grid_part is not None and grid_part.shape[1] > 0:
                        n_use = min(grid_part.shape[1], n_have)
                        window = np.zeros((n_ch, n_times), dtype=np.float32)
                        window[:, :n_use] = grid_part[:, :n_use]
                        x = zscore_time(window)[None, ...]
                        with torch.no_grad():
                            logits = model(torch.from_numpy(x).to(device))
                            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                        bar_left[0].set_width(float(probs[0]))
                        bar_right[0].set_width(float(probs[1]))
                        p0, p1 = float(probs[0]), float(probs[1])
                        guess = 0 if p0 >= p1 else 1
                        guess_name = left_pretty if guess == 0 else right_pretty
                        match = trial_label is not None and guess == trial_label
                        match_txt = "matches ground truth" if match else "does not match ground truth"
                        pred_line.set_text(
                            f"Model favors: {guess_name}  "
                            f"(left {p0:.0%} · right {p1:.0%})  — {match_txt}"
                        )
                        pred_line.set_color("#1a6b2d" if match else "#8a4b00")
                        status.set_text(
                            f"Trial time: {elapsed:.2f}s / {n_times / sfreq:.2f}s  ·  "
                            f"Highlighted row = correct answer for this trial"
                        )
                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()

                if elapsed >= (n_times / sfreq) + 0.25:
                    trial_t0 = None
                    trial_label = None
                    trial_name = ""
                    set_truth_visual(None)
                    status.set_text("Trial done. Waiting for next marker...")
                    fig.canvas.draw_idle()
                    plt.pause(0.001)

            if not samples and not ms:
                time.sleep(0.002)

            # Always yield to the GUI: during an active trial we otherwise spin on pull_chunk
            # with no pause, which freezes the window (Windows reports "Not responding").
            if not args.headless and args.gui_pause > 0:
                plt.pause(args.gui_pause)

    except KeyboardInterrupt:
        print("stopped.")

    plt.ioff()
    if args.headless:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
