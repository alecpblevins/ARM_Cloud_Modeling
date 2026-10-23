# Aggregation functions for cloud persistence analysis
# This module was created 10/23/2026, Author: Alec P.

#These functions and definitions are illistrated in notebooks/07_annual_target_sketch.ipynb

#This will turn native 4-second cloud-state series into bins, by default, 20 seconds each. 

"""
Aggregate native 4 s cloud-state series into fixed time bins.
"""

###################
##Library Imports##
###################

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_STEP_SECONDS = 4


##################
##Main Functions##
##################

#Here, this function aggregates the cloud-state series into fixed time bins and computes per-bin statistics.
def aggregate_to_bins(state: pd.Series, interval: str = "20s", step: int = DEFAULT_STEP_SECONDS) -> pd.DataFrame:

    """
    Per-bin counts and cloud fraction (cloudy / known), left-labelled.

    `state` is int {1 cloudy, 0 clear, -1 unknown} on a DatetimeIndex.
    Bins are anchored at the start of day so they never straddle midnight.

    Returns a DataFrame with per-bin statistics:
    - n_observed: number of observations in the bin
    - n_known: number of known (clear or cloudy) observations
    - n_cloudy: number of cloudy observations
    - cloud_fraction: fraction of known observations that are cloudy
    - fully_observed: whether the bin has the expected number of observations
    - fully_known: whether all observations in the bin are known

    """
    interval_s = pd.Timedelta(interval).total_seconds() # Convert interval to seconds
    if interval_s % step:
        raise ValueError("interval must be a multiple of the native step")
    
    kw = dict(origin="start_day", label="left", closed="left") #Same as before, ensures bins are left-labelled and anchored at the start of the day
    known = state.isin([0, 1]).astype(np.int8).resample(interval, **kw).sum() # Count of known observations per bin
    cloudy = state.eq(1).astype(np.int8).resample(interval, **kw).sum() # Count of cloudy observations per bin
    obs = pd.Series(1, index=state.index, dtype=np.int8).resample(interval, **kw).sum()
    expected = int(interval_s // step)
    return pd.DataFrame({
        "n_observed": obs,
        "n_known": known,
        "n_cloudy": cloudy,
        "cloud_fraction": cloudy / known.replace(0, np.nan),
        "fully_observed": obs.eq(expected),
        "fully_known": known.eq(obs) & obs.gt(0),
    })

#Here, this function determines the cloud state of each bin based on the per-bin statistics.
def bin_state(bins: pd.DataFrame) -> pd.Series:
    """
    1 cloudy, 0 clear, -1 unknown (incomplete, unknown, or 50/50 tie).

    """
    ok = bins["fully_observed"] & bins["fully_known"] # Bins that are fully observed and fully known
    s = pd.Series(-1, index=bins.index, dtype=np.int8, name="bin_state") # Initialize all bins as unknown (-1)
    s[ok & bins["cloud_fraction"].gt(0.5)] = 1 # Set bins with cloud fraction > 0.5 as cloudy
    s[ok & bins["cloud_fraction"].lt(0.5)] = 0 # Set bins with cloud fraction < 0.5 as clear
    return s

#Finally a helper function to aggregate the cloud state into bins and determine the bin states
def aggregate_cloud_state(state: pd.Series, interval: str = "20s", step: int = DEFAULT_STEP_SECONDS):
    """
    Convenience: return (bins, bin_state).
    
    """
    bins = aggregate_to_bins(state, interval, step)
    return bins, bin_state(bins)