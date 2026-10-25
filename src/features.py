#This module will be fore past-only predictors. 
# This module was created 10/25/2026, Author: Alec P.

"""
Past-only predictors built on the bin-state series.

IMPORTANT:
Every feature at bin t uses bins <= t only. Bin t covers [t, t+interval), and
the target window starts at bin t+1, so bin t is the last information the
model is allowed to see.

So far, this assumes the following decisions: 
1. Target is persistence: is every bin in the next 20 min cloudy? 
2. It is restricted to currently cloudy bins
3. Unknown outcomes are dropped
4. A known clear bin in the window gives 0
5. Split into past-only features, no lookahead allowed

"""

##################
##Library Import##
##################

from __future__ import annotations

import numpy as np
import pandas as pd

from src.targets import horizon_to_bins #Import horizon conversion function from targets module

DEFAULT_WINDOWS = ("1min", "5min", "20min")

##################
##Main Functions##
##################

#This helper function will aggregate cloud structure information over a specified time interval.
def aggregate_structure(frame4: pd.DataFrame, interval: str = "20s") -> pd.DataFrame:
    """
    Per-bin cloud-structure summaries from the native construct_cloud_state frame.

    Needs the columns n_cloud_layers, lowest_cloud_base_m, highest_cloud_top_m
    and kazr_available (notebook 07 currently keeps only cloud_state).
    """

    r = frame4.resample(interval, origin="start_day", label="left", closed="left") # Resample the frame to the specified interval, aligning to the start of the day
    out = pd.DataFrame({
        "n_layers_mean": r["n_cloud_layers"].mean(),
        "lowest_base_m": r["lowest_cloud_base_m"].min(),
        "highest_top_m": r["highest_cloud_top_m"].max(),
        "kazr_avail_frac": r["kazr_available"].mean(),
    })
    out["vertical_span_m"] = out["highest_top_m"] - out["lowest_base_m"]
    return out


#Helper function to compute run lengths of consecutive bins in the current state
# Consecutive bins are counted until a change occurs or an unknown bin is encountered.
def _run_length(s: pd.Series) -> pd.Series:
    """
    Consecutive bins in the current state, ending at t (unknown bins break runs).
    """

    grp = (s != s.shift()).cumsum() # Group consecutive identical values together
    return s.groupby(grp).cumcount() + 1

#Main feature-building function, constructs features for each bin based on cloud fraction, run lengths, time of day, and optional cloud structure information.
def build_features(bins: pd.DataFrame, s: pd.Series, interval: str = "20s", windows=DEFAULT_WINDOWS, structure: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    One row per bin; NaN where a feature's past window is incomplete or unknown.
    """

    cf = bins["cloud_fraction"] # Cloud fraction for each bin
    known = s.isin([0, 1]) # Identify bins with known states (0 or 1)
    prev_known = known & known.shift(fill_value=False) # Identify bins where the previous state is also known
    flip = ((s != s.shift()) & prev_known).astype(float) # Identify state flips where both current and previous states are known

    X = pd.DataFrame(index=bins.index)
    X["cloud_fraction"] = cf
    X["run_length_min"] = _run_length(s) * pd.Timedelta(interval).total_seconds() / 60 # Run length of current state in minutes
    X["hour_sin"] = np.sin(2 * np.pi * X.index.hour / 24) # Sine of hour of day
    X["hour_cos"] = np.cos(2 * np.pi * X.index.hour / 24) # Cosine of hour of day

    for w in windows: # Compute rolling features for each window
        n = horizon_to_bins(w, interval)
        X[f"cf_mean_{w}"] = cf.rolling(n, min_periods=n).mean()
        X[f"cf_min_{w}"] = cf.rolling(n, min_periods=n).min()
        X[f"cf_change_{w}"] = cf - cf.shift(n)
        X[f"n_flips_{w}"] = flip.rolling(n, min_periods=n).sum()

    if structure is not None: # Include cloud structure features if available, will add in later
        S = structure.reindex(bins.index)
        for c in ["n_layers_mean", "lowest_base_m", "highest_top_m", # Cloud structure columns to include
                  "vertical_span_m", "kazr_avail_frac"]:
            X[c] = S[c]
        n5 = horizon_to_bins("5min", interval) # Number of bins corresponding to 5 minutes
        X["lowest_base_change_5min"] = S["lowest_base_m"] - S["lowest_base_m"].shift(n5)
        X["highest_top_change_5min"] = S["highest_top_m"] - S["highest_top_m"].shift(n5)

    X["features_valid"] = X.drop(columns="features_valid", errors="ignore").notna().all(axis=1)
    return X

# Function to check that features do not use future information (no lookahead)
def check_no_lookahead(bins, s, cut, **kw) -> bool:
    """
    Features at times <= cut must be identical when all later data is removed.
    """
    full = build_features(bins, s, **kw).loc[:cut] # Features computed using the full dataset up to the cut time
    trunc = build_features(bins.loc[:cut], s.loc[:cut], **kw) # Features computed using only data up to the cut time
    return bool(np.allclose(full.drop(columns="features_valid"),
                            trunc.drop(columns="features_valid"), equal_nan=True))