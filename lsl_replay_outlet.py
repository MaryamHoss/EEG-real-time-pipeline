"""
Simulated LSL **outlet** (amplifier + markers): streams preprocessed EEGBCI EEG at nominal
rate and irregular marker samples at T1/T2 onsets.

Run this in one terminal, then ``lsl_neurofeedback_client.py`` in another (same machine is fine;
LSL uses the local network stack).

Requires: ``pip install pylsl`` (bundles liblsl on many Windows installs).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from lsl_config import (
    EEG_SOURCE_ID,
    EEG_STREAM_NAME,
    MARKER_SOURCE_ID,
    MARKER_STREAM_NAME,
    MARKER_T1,
    MARKER_T2,
)
from preprocessing import PreprocConfig, load_concatenated_eegbci, preprocess_raw


def _build_eeg_stream_info(n_channels: int, sfreq: float, ch_names: list[str]):
    import pylsl
# Create the base stream info
# Name, type, number of channels, sample rate, data type, source ID
    info = pylsl.StreamInfo(
        EEG_STREAM_NAME, # "EEGBCI_MI_Replay"
        "EEG", # Type of data (standard category)
        n_channels, # Number of channels (64 channels)
        sfreq, # Sample rate (160 Hz)
        pylsl.cf_float32, # Data type (float32)
        EEG_SOURCE_ID, # # Unique ID for this specific "headset" (eegbci_mi_replay_eeg_v1)
    )
    # Append the channel information to the stream info
    # Each channel has a label and a type
    # This appends the actual channel names (C3, C4, Cz, etc.) 
    channels = info.desc().append_child("channels")
    for lab in ch_names:
        c = channels.append_child("channel")
        c.append_child_value("label", str(lab))
    return info


def _build_marker_stream_info():
    import pylsl

    return pylsl.StreamInfo(
        MARKER_STREAM_NAME, # "EEGBCI_MI_Markers"
        "Markers", # Type of data (standard category)
        1, # Number of channels (1 or 2)
        pylsl.IRREGULAR_RATE, # We don't send markers on a rhythm; only when needed
        pylsl.cf_int32, # Data type (int32)
        MARKER_SOURCE_ID, # # Unique ID for this specific "headset" (eegbci_mi_replay_markers_v1)
    )


def main() -> None:
    p = argparse.ArgumentParser(description="LSL outlet: replay preprocessed EEGBCI MI EEG + markers.")
    p.add_argument("--subject", type=int, default=1)
    p.add_argument("--runs", type=int, nargs="+", default=[4, 8, 12])
    p.add_argument("--speed", type=float, default=4.0, help=">1 = faster than real time.")
    p.add_argument("--chunk", type=int, default=32, help="Samples per push_chunk.")
    p.add_argument(
        "--tmax",
        type=float,
        default=None,
        help="Replay only the first N seconds of the concatenated recording (default: all).",
    )
    p.add_argument(
        "--loop",
        action="store_true",
        help="Repeat the replay indefinitely (good while tuning the client).",
    )
    args = p.parse_args()

    import pylsl

    cfg = PreprocConfig()
    raw = load_concatenated_eegbci(args.subject, args.runs)
    raw = preprocess_raw(raw, cfg)
    data = raw.get_data()  # volts, (n_ch, n_samp)
    n_ch, n_samp = data.shape
    sfreq = float(raw.info["sfreq"])
    ch_names = raw.ch_names

    if args.tmax is not None:
        n_samp = min(n_samp, int(args.tmax * sfreq))

    markers: list[tuple[int, int]] = []
    ann = raw.annotations # Get markers from the file
    for desc, onset, _dur in zip(ann.description, ann.onset, ann.duration, strict=True):
        if desc not in cfg.event_id: # If the marker is not in the event ID, skip it
            continue
        sidx = int(round(float(onset) * sfreq)) # Convert the onset time to a sample index
        if 0 <= sidx < n_samp: # If the sample index is within the data, add the marker
            code = MARKER_T1 if desc == "T1" else MARKER_T2 # If the marker is T1, add 1, if it's T2, add 2
            markers.append((sidx, code))
    markers.sort(key=lambda x: x[0]) # Sort the markers by the sample index

    eeg_info = _build_eeg_stream_info(n_ch, sfreq, ch_names)
    mk_info = _build_marker_stream_info()
    outlet_eeg = pylsl.StreamOutlet(eeg_info, chunk_size=max(1, args.chunk)) # Create the EEG stream outlet
    outlet_mk = pylsl.StreamOutlet(mk_info, chunk_size=0) # Create the marker stream outlet

    speed = max(args.speed, 0.01) # Set the speed to the maximum of the speed argument or 0.01
    chunk = max(1, args.chunk) # Set the chunk to the maximum of 1 or the chunk argument

    print(
        f"LSL replay: {n_ch} ch @ {sfreq:.1f} Hz, {n_samp} samples, {len(markers)} T1/T2 markers. "
        f"Streams '{EEG_STREAM_NAME}' + '{MARKER_STREAM_NAME}'. Speed={speed}x. "
        f"Loop={'on' if args.loop else 'off'}. Ctrl+C to stop."
    )

    try:
        while True:
            base = pylsl.local_clock() # The "Master Clock"
            mi = 0 # Marker index counter (to keep track of which marker we're sending)
            # Loop through the data in chunks of 32 samples (chunk)
            for global_i in range(0, n_samp, chunk):
                end = min(global_i + chunk, n_samp) # The end of the current chunk
                # Loop through the markers and send them as they come
                while mi < len(markers) and markers[mi][0] < end:
                    sidx, code = markers[mi] # Get the sample index and code for the current marker
                    ts = base + sidx / sfreq # Convert the sample index to a timestamp
                    outlet_mk.push_sample([code], timestamp=ts) # Send the marker to the marker stream outlet
                    mi += 1 # Increment the marker index counter
                # Get the data for the current chunk
                slab = data[:, global_i:end].astype(np.float32, copy=False) # Convert the data to a float32 array
                ts0 = base + global_i / sfreq # Convert the chunk start time to a timestamp
                # pylsl: first array dim = n_samples; shape (n_samples, n_channels).
                # Push to LSL: We flip the data (.T) to match LSL's format (n_channels, n_samples)
                outlet_eeg.push_chunk(np.ascontiguousarray(slab.T), timestamp=ts0) # Send the data to the EEG stream outlet
                time.sleep((end - global_i) / sfreq / speed) #Wait: This keeps the loop running at the correct speed
            if not args.loop:
                break
    except KeyboardInterrupt:
        print("stopped.")


if __name__ == "__main__":
    main()
