"""Shared Lab Streaming Layer stream identifiers for the EEGBCI replay simulator."""

from __future__ import annotations

# Resolve inlets with: pylsl.resolve_byprop("name", ...)
EEG_STREAM_NAME = "EEGBCI_MI_Replay"
MARKER_STREAM_NAME = "EEGBCI_MI_Markers"

EEG_SOURCE_ID = "eegbci_mi_replay_eeg_v1"
MARKER_SOURCE_ID = "eegbci_mi_replay_markers_v1"

# Marker channel: int32 trial cue codes (match annotations T1/T2)
MARKER_T1 = 1
MARKER_T2 = 2
