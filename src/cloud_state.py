"""

ARM KAZR ARSCL cloud-state construction and run segmentation.

Created: October 21, 2026
Author: Alec P.

This file turns a single QC-applied KAZR dataset into two reusable artifacts:

1. ``construct_cloud_state(ds)`` — a timestamp-level ``pandas.DataFrame``
   with cloud state, layer count, lowest base, highest top, and KAZR
   availability.

2. ``segment_runs(cloud_state_df)`` — a run-level ``pandas.DataFrame`` of
   maximal consecutive same-known-state segments, with sample-supported and
   first-to-last duration conventions plus boundary-censoring flags.

Input: 
--------------
``construct_cloud_state`` expects a QC-applied dataset produced by
``src.qc.apply_qc(ds, load=True)``. That guarantees ``kazr_available`` is
present. If you have only an interim file and no radar
features are needed, call ``apply_qc`` once for the state column. This keeps
a single input contract instead of supporting two.

Cloud-state rules:
----------------
The primary cloud state is defined from the ARM cloud-layer product, *not*
from reflectivity presence:

    cloud_state =  1   at least one ``cloud_layer_base_height`` >= 0
                =  0   every ``cloud_layer_base_height`` == -1
                = -1   neither condition holds (unknown / needs review)

A ``-1`` in the layer-base fields is the documented clear-sky sentinel and is
*not* a missing value. ``NaN`` or unexpected sentinels therefore fall into
``unknown`` on purpose. ``cloud_layer_top_height`` is used only to compute
``highest_cloud_top_m`` and never defines cloud state.

Run definition:
--------------
A run is a maximal consecutive sequence of timestamps sharing the same known
state (0 or 1). An unknown observation, a change in known state, or a time
step other than ``expected_step_seconds`` breaks a run. Runs adjacent to an
unknown observation or a time gap, or touching the start/end of the analyzed
series, are flagged as censored rather than dropped: their observed duration
is a lower bound on the true duration.

"""

###################
##Library Imports##
###################

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

#########
##Setup##
#########

DEFAULT_STEP_SECONDS = 4.0
CLOUD_STATE_LABELS = {1: "Cloudy", 0: "Clear", -1: "Unknown"}

_REQUIRED_CLOUD_VARIABLES = {
    "time",
    "cloud_layer_base_height",
    "cloud_layer_top_height",
    "kazr_available",
}

####################
##Helper Functions##
####################

#A function to check for required variables in an xarray Dataset
def _require_variables(ds: xr.Dataset, required: Iterable[str]) -> None:
    """Raise a clear error if an expected variable is absent."""
    missing = sorted(set(required) - set(ds.variables))
    if missing:
        raise ValueError(
            "Input dataset is missing required variables: " + ", ".join(missing)
        )


########################
##Cloud State Function##
########################

def construct_cloud_state(ds: xr.Dataset) -> pd.DataFrame:
    """Build a timestamp-level cloud-state DataFrame from a QC-applied dataset.

    Parameters:
    ----------
    ds : xr.Dataset
        A QC-applied ``xarray.Dataset`` (output of
        ``src.qc.apply_qc(ds, load=True)``) containing at least ``time``,
        ``cloud_layer_base_height``, ``cloud_layer_top_height``, and
        ``kazr_available``.

    Returns:
    -------
    pandas.DataFrame
        One row per timestamp, indexed by time (name ``"time"``). Columns:

        - ``cloud_state``        : int8 {1 = Cloudy, 0 = Clear, -1 = Unknown}
        - ``n_cloud_layers``     : int8, count of nonnegative layer bases
        - ``lowest_cloud_base_m``: float32, min nonnegative layer base
                                   (NaN when clear)
        - ``highest_cloud_top_m``: float32, max valid layer top
                                   (NaN when clear)
        - ``kazr_available``     : bool
        - ``cloud_state_label``  : str
    """
    _require_variables(ds, _REQUIRED_CLOUD_VARIABLES) #Call helper function

    base = ds["cloud_layer_base_height"]
    top = ds["cloud_layer_top_height"]

    # Layer-level masks
    valid_base = base >= 0 # Mask for valid cloud layer bases
    has_cloud = valid_base.any(dim="layer") # True if any valid cloud layer exists
    known_clear = (base == -1).all(dim="layer") # True if all layers are explicitly clear

    # -1 is the deliberate "unknown" state; clear and cloudy are assigned 0/1.
    state = xr.where(has_cloud, 1, xr.where(known_clear, 0, -1)).astype(np.int8) # Cloud state: 1=cloudy, 0=clear, -1=unknown
    n_layers = valid_base.sum(dim="layer").astype(np.int8) # Number of valid cloud layers

    # NaN where no valid base. For a fully clear timestamp, valid_base is all false and this reduces to NaN
    lowest_base = base.where(valid_base).min(dim="layer", skipna=True)

    # Only tops from layers that have a valid base count toward the top.
    valid_top = valid_base & (top >= 0)
    highest_top = top.where(valid_top).max(dim="layer", skipna=True)

    kazr_available = np.asarray( # Convert KAZR availability to boolean array
        ds["kazr_available"].fillna(0).values, dtype=bool
    )

    index = pd.DatetimeIndex(ds["time"].values, name="time") # Create a pandas DatetimeIndex from the time coordinate
    state_values = state.values # Extract the cloud state values as a NumPy array
    frame = pd.DataFrame(
        {
            "cloud_state": state_values,
            "n_cloud_layers": n_layers.values,
            "lowest_cloud_base_m": lowest_base.values,
            "highest_cloud_top_m": highest_top.values,
            "kazr_available": kazr_available,
        },
        index=index, # Use the previously created DatetimeIndex as the DataFrame index
    )
    frame["cloud_state_label"] = [ # Map cloud state integers to human-readable labels
        CLOUD_STATE_LABELS[int(s)] for s in state_values
    ]
    return frame[
        [
            "cloud_state",
            "n_cloud_layers",
            "lowest_cloud_base_m",
            "highest_cloud_top_m",
            "kazr_available",
            "cloud_state_label",
        ]
    ]

#Here, we will define a function to concatenate multiple daily frames. 
def concat_cloud_state_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate daily cloud-state frames into one sorted frame.

    Intended as a building block in ``build_dataset.py``. Raises on duplicate
    timestamps, which would indicate that adjacent daily files overlap or
    that the source list was not deduplicated.
    """
    frames = list(frames)
    if not frames:
        raise ValueError("No cloud-state frames to concatenate.")
    out = pd.concat(frames).sort_index() # Concatenate the frames and sort by the index (time)
    if out.index.duplicated().any(): # Check for duplicate timestamps after concatenation
        raise ValueError(
            "Duplicate timestamps found after concatenation. "
            "Check the input file list for overlapping days."
        )
    return out

####################
##Run Segmentation##
####################

def segment_runs(cloud_state_df: pd.DataFrame, *, expected_step_seconds: float = DEFAULT_STEP_SECONDS, ) -> pd.DataFrame:
    """Segment a timestamp-level cloud-state frame into known-state runs.

    Parameters:
    ----------
    cloud_state_df
        Output of :func:`construct_cloud_state`.
    expected_step_seconds
        Nominal sampling interval used to detect gaps. Defaults to 4 seconds (KAZR native cadence). Probably should use 20 sec.

    Returns:
    -------
    pandas.DataFrame
        One row per known-state run, indexed by a 0-based ``run_id``.
        Columns:

        - ``cloud_state``, ``state_label``
        - ``start_time``, ``end_time``
        - ``n_timestamps``
        - ``sample_duration_seconds``, ``sample_duration_minutes``
        - ``timestamp_span_seconds``
        - ``left_censored``, ``right_censored``, ``boundary_censored``
    """
    #Quick checks
    if "cloud_state" not in cloud_state_df.columns:
        raise ValueError("cloud_state_df must have a 'cloud_state' column.")
    if not isinstance(cloud_state_df.index, pd.DatetimeIndex):
        raise ValueError("cloud_state_df must be indexed by a DatetimeIndex.")

    state_values = cloud_state_df["cloud_state"].to_numpy(dtype=np.int8)
    time_index = cloud_state_df.index
    n = len(state_values)

    if n == 0:
        return _empty_run_frame() #Call helper function to deal with this

    deltas = time_index.to_series().diff().dt.total_seconds().to_numpy() # Compute time differences between consecutive timestamps

    # A new run starts when the state changes, a time gap occurs, or either side of the boundary is unknown. Unknown observations always break.
    new_run = np.ones(n, dtype=bool)
    if n > 1:
        new_run[1:] = (
            (state_values[1:] != state_values[:-1]) # State change triggers a new run
            | (deltas[1:] != expected_step_seconds) # Time gap triggers a new run
            | (state_values[1:] == -1) # Unknown observation triggers a new run
            | (state_values[:-1] == -1) # Previous unknown observation triggers a new run
        )

    run_starts_all = np.flatnonzero(new_run) # Indices where new runs start
    run_ends_all = np.r_[run_starts_all[1:] - 1, n - 1] # Indices where runs end

    # Keep only clear/cloudy runs; unknown observations remain boundaries but are not events.
    run_is_known = np.isin(state_values[run_starts_all], [0, 1]) # Boolean array indicating which runs start with a known state
    starts = run_starts_all[run_is_known] # Filtered start indices for known runs
    ends = run_ends_all[run_is_known] # Filtered end indices for known runs
    states = state_values[starts] # State values for the filtered runs

    if len(states) == 0:
        return _empty_run_frame() #Again a function to help with this

    start_times = time_index[starts]
    end_times = time_index[ends]
    n_timestamps = ends - starts + 1

    # Boundary censoring: a run is censored on the left if it touches the
    # start of the frame, or if the previous observation is unknown or
    # separated from this run by a time gap.
    left_censored = starts == 0
    left_other = ~left_censored # Indices where the run is not left-censored
    if left_other.any():
        prev_idx = starts[left_other] - 1
        left_censored[left_other] = (
            (state_values[prev_idx] == -1)
            | (deltas[starts[left_other]] != expected_step_seconds)
        )

    right_censored = ends == n - 1
    right_other = ~right_censored # Indices where the run is not right-censored
    if right_other.any():
        next_idx = ends[right_other] + 1
        right_censored[right_other] = (
            (state_values[next_idx] == -1)
            | (deltas[next_idx] != expected_step_seconds)
        )

    frame = pd.DataFrame( #return the run frame as a DataFrame
        {
            "cloud_state": states,
            "state_label": [CLOUD_STATE_LABELS[int(s)] for s in states],
            "start_time": start_times,
            "end_time": end_times,
            "n_timestamps": n_timestamps,
            "sample_duration_seconds": n_timestamps * expected_step_seconds,
            "sample_duration_minutes":
                n_timestamps * expected_step_seconds / 60.0,
            "timestamp_span_seconds":
                (end_times - start_times).total_seconds(),
            "left_censored": left_censored,
            "right_censored": right_censored,
        }
    )
    frame["boundary_censored"] = (
        frame["left_censored"] | frame["right_censored"]
    )
    frame.index.name = "run_id"
    return frame

#Now a function to summarize the durations of runs by cloud state
def summarize_run_durations(run_frame: pd.DataFrame, *, complete_only: bool = False, ) -> pd.DataFrame:
    """
    Summarize run durations by state.

    Reproduces the tables printed in nb 04, section 5 and nb 05, section 7. Uses the
    sample-supported duration convention (``n_timestamps * expected_step_seconds``).

    Parameters
    ----------
    run_frame
        Output of :func:`segment_runs`.
    complete_only
        If True, exclude runs adjacent to an edge, gap, or unknown state
        before summarizing.
    """

    frame = run_frame
    if complete_only:
        frame = frame.loc[~frame["boundary_censored"]] # Exclude runs that are boundary-censored if complete_only is True

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "n_runs", "mean_minutes", "median_minutes",
                "p25_minutes", "p75_minutes", "p90_minutes",
                "maximum_minutes",
            ]
        )

    rows = []
    for label, group in frame.groupby("state_label"):
        durations = group["sample_duration_minutes"]
        rows.append(
            {
                "state": label,
                "n_runs": int(len(group)),
                "mean_minutes": float(durations.mean()),
                "median_minutes": float(durations.median()),
                "p25_minutes": float(durations.quantile(0.25)),
                "p75_minutes": float(durations.quantile(0.75)),
                "p90_minutes": float(durations.quantile(0.90)),
                "maximum_minutes": float(durations.max()),
            }
        )
    return pd.DataFrame(rows).set_index("state")


#Then a quick helper function to create an empty run frame
def _empty_run_frame() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            col: pd.Series(dtype=t)
            for col, t in [
                ("cloud_state", "int8"),
                ("state_label", "object"),
                ("start_time", "datetime64[ns]"),
                ("end_time", "datetime64[ns]"),
                ("n_timestamps", "int64"),
                ("sample_duration_seconds", "float64"),
                ("sample_duration_minutes", "float64"),
                ("timestamp_span_seconds", "float64"),
                ("left_censored", "bool"),
                ("right_censored", "bool"),
                ("boundary_censored", "bool"),
            ]
        }
    )
    df.index.name = "run_id"
    return df