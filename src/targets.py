# Module for defining and computing annual cloud targets based on aggregated cloud states.
# This module was created 10/23/2026, Author: Alec P.

#These functions and definitions are illistrated in notebooks/07_annual_target_sketch.ipynb

#Here, we will build supervised labels from the bin-state series, while also defining the train/validation/test splits. 

"""
Future-outcome targets defined on a regular bin-state series.

We have a dependancy chain: 
`apply_qc` → `construct_cloud_state` → `aggregate_cloud_state` → `build_target` → `chronological_split`

"""

###################
##Library Imports##
###################

from __future__ import annotations

import numpy as np
import pandas as pd

#Here, we convert a time horizon into a number of bins(60). 
def horizon_to_bins(horizon: str, interval: str) -> int:
    """e.g. horizon_to_bins('20min', '20s') -> 60."""
    ratio = pd.Timedelta(horizon) / pd.Timedelta(interval) # Compute how many bins fit into the horizon
    if ratio < 1 or ratio != int(ratio):
        raise ValueError("horizon must be a positive multiple of interval")
    return int(ratio)

#Just a check function to validate the series and horizon
def _check(s: pd.Series, h: int) -> None:
    if h < 1:
        raise ValueError("h must be >= 1") # Horizon must be at least one bin long
    if not isinstance(s.index, pd.DatetimeIndex) or not s.index.is_monotonic_increasing:
        raise ValueError("state must have a sorted DatetimeIndex")
    if s.index.to_series().diff().dropna().nunique() > 1:
        raise ValueError("state index must be a regular grid (shift assumes it)")

#Nice helper function to count True values in a sliding window of length h
def _window_counts(ind: np.ndarray, h: int) -> np.ndarray: #Just like in our notebook
    """Count True values in indices (i, i+h], truncated at the series end."""
    cs = np.r_[0, np.cumsum(ind, dtype=np.int64)] # Cumulative sum with a leading zero for easier window sum calculation
    i = np.arange(len(ind))
    return cs[np.minimum(i + h + 1, len(ind))] - cs[i + 1]

#Helper function to construct the DataFrame for the target building functions
def _frame(s, cur, tgt, valid) -> pd.DataFrame:
    return pd.DataFrame({"current_state": cur, "target": tgt,
                         "target_valid": valid}, index=s.index)


##################
##Main Functions##
##################

#Here, PointState Target. 
def build_point_state_target(s: pd.Series, h: int) -> pd.DataFrame:
    """Among cloudy bins: is bin t+h cloudy? Valid only if t+h is known."""
    _check(s, h) #Call check function
    cur = s.to_numpy()
    fut = s.shift(-h).to_numpy() # Future state at t+h
    valid = (cur == 1) & np.isin(fut, [0, 1]) # Only cloudy bins with known future state are valid
    tgt = np.where(valid, fut, -1).astype(np.int8) # Target is future state if valid, otherwise -1
    return _frame(s, cur, tgt, valid)


#Here, Persistence Target. Main target. 
def build_persistence_target(s: pd.Series, h: int) -> pd.DataFrame:
    """
    Among cloudy bins: is every bin in (t, t+h] cloudy?

    0 if any known clear bin is in the window (unknowns cannot undo it);
    1 if all h bins are cloudy; otherwise invalid (unknown or off the end).
    """
    _check(s, h) #Validate
    v = s.to_numpy()
    n = len(v)
    any_clear = _window_counts(v == 0, h) > 0  # Any known clear bin in the window
    all_cloudy = (np.arange(n) + h < n) & (_window_counts(v == 1, h) == h) # All bins in the window are cloudy
    valid = (v == 1) & (any_clear | all_cloudy) # Only cloudy bins with a definitive window are valid
    tgt = np.where(valid, np.where(any_clear, 0, 1), -1).astype(np.int8) # Target is 0 if any clear, 1 if all cloudy, -1 if invalid
    return _frame(s, v, tgt, valid)


TARGET_BUILDERS = {"point": build_point_state_target,
                   "persistence": build_persistence_target}

#Helper function to pick a builder by name
def build_target(s: pd.Series, h: int, kind: str = "persistence") -> pd.DataFrame:
    return TARGET_BUILDERS[kind](s, h)

#Here, a function to return train/val/test masks.
def chronological_split(index: pd.DatetimeIndex, bounds: dict, h: int, interval: str) -> dict:
    """
    Boolean masks for train/val/test with an embargo of h bins.

    `bounds` = {"val_start": ..., "test_start": ...}. Half-open intervals;
    the last h bins before each boundary are dropped so target windows
    cannot reach into the next fold.
    """
    gap = pd.Timedelta(interval) * h # Calculate the embargo gap in time units
    vs, ts = pd.Timestamp(bounds["val_start"]), pd.Timestamp(bounds["test_start"])
    return {
        "train": index < vs - gap,
        "val": (index >= vs) & (index < ts - gap), # Validation set with embargo applied
        "test": index >= ts,
    }