#This module will compile all modules, constructing a one table dataset and QA report
# This module was created 10/27/2026, Author: Alec P.

"""
Build the modeling table: interim ARM files -> one table of cloudy-bin rows.

Pipeline: apply_qc (in memory) -> construct_cloud_state -> concat days ->
aggregate_cloud_state -> build_features (past only) -> build_target (future
only) -> row mask -> chronological folds -> table + QA report.

Run from the project root:
    python -m src.build_dataset --val-start 2024-09-01 --test-start 2024-11-01

"""

###################
##Library Imports##
###################

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import xarray as xr

##################
##Module Imports##
##################

from src.aggregation import aggregate_cloud_state
from src.cloud_state import concat_cloud_state_frames, construct_cloud_state
from src.features import aggregate_structure, build_features, check_no_lookahead
from src.qc import apply_qc
from src.targets import build_target, chronological_split, horizon_to_bins

#################################
##Path and Variable Definitions##
#################################

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INTERIM = (PROJECT_ROOT / "ARM_DATA" / "2024-2025" / "interim"
                   / "sgparsclkazr1kolliasC1.c0")
DEFAULT_OUT = PROJECT_ROOT / "ARM_DATA" / "2024-2025" / "modeling"

# Required for a row to be kept. Rolling and change features may stay NaN:
#dropping them removes young clouds, which dissolve more often (nb 08, sec. 9).
CORE_FEATURES = ["cloud_fraction", "run_length_min", "hour_sin", "hour_cos",
                 "n_layers_mean", "lowest_base_m", "highest_top_m",
                 "vertical_span_m", "kazr_avail_frac"]

#############
##Functions##
#############

#Here, a function to list all dates within a range for which interim files exist.
def list_dates(interim_dir: Path, start: str, end: str) -> list[str]:
    """Contiguous YYYYMMDD strings; every file must exist (regular grid needed)."""
    dates = [d.strftime("%Y%m%d") for d in pd.date_range(start, end, freq="D")] # Generate all dates in the range as YYYYMMDD strings
    missing = [d for d in dates if not (interim_dir / f"kazr_{d}.nc").is_file()] # Check for missing interim files
    if missing:
        raise FileNotFoundError(f"{len(missing)} interim files missing, e.g. {missing[:3]}")
    return dates

#Here, a function to load and concatenate state frames for the given dates.
def load_state_frame(interim_dir: Path, dates: list[str]) -> pd.DataFrame:
    """QC in memory, then one 4 s state frame per day, concatenated."""
    frames = []
    for i, d in enumerate(dates, 1):
        with xr.open_dataset(interim_dir / f"kazr_{d}.nc") as ds: #Open the interim file for the current date
            frames.append(construct_cloud_state(apply_qc(ds)) #Apply QC, then construct cloud state, append
                          .drop(columns="cloud_state_label")) #Drop the cloud state label column
        if i % 30 == 0 or i == len(dates):
            print(f"  loaded {i}/{len(dates)} days") #Load progress for every 30 days or last day
    return concat_cloud_state_frames(frames) #Concatenate all daily state frames into a single DataFrame


def build_table(frame4: pd.DataFrame, interval: str = "20s", horizon: str = "20min", target_kind: str = "persistence", required: str = "core", check: bool = True) -> tuple[pd.DataFrame, int]:
    """One row per bin with features, target and validity flags (all bins kept)."""
    h = horizon_to_bins(horizon, interval) #Convert the horizon to the number of bins based on the interval
    bins, s = aggregate_cloud_state(frame4["cloud_state"], interval) #Aggregate the cloud state into bins and summary statistics
    structure = aggregate_structure(frame4, interval) #Aggregate the structure of the cloud state
    X = build_features(bins, s, interval=interval, structure=structure) #Build features from the aggregated bins and structure

    if check: #Perform a lookahead check to ensure features do not change when future data is removed (no overlap)
        cut = s.index[len(s) // 2]
        if not check_no_lookahead(bins, s, cut, interval=interval, structure=structure):
            raise AssertionError("features changed when future data was removed")

    T = build_target(s, h, target_kind) #Build the target variable based on the summary statistics, horizon, and target kind
    tbl = X.drop(columns="features_valid").join(
        T[["current_state", "target", "target_valid"]]) #Combine the features and target into a single table
    req = (CORE_FEATURES if required == "core"
           else [c for c in X.columns if c != "features_valid"]) #Determine the required features based on the 'required' parameter
    tbl["features_ok"] = tbl[req].notna().all(axis=1) #Check if all required features are present
    tbl["row_ok"] = tbl["target_valid"] & tbl["features_ok"] #Check if the row is valid based on target and features
    return tbl, h #Return the final table and the horizon in bins

# Assign folds for cross-validation based on chronological order with an embargo period
# 'val_start' and 'test_start' define the start dates for the validation and test sets
# 'h' is the horizon in bins, 'interval' is the binning interval
def assign_folds(tbl: pd.DataFrame, val_start: str, test_start: str, h: int, interval: str) -> pd.DataFrame:
    masks = chronological_split(tbl.index, {"val_start": val_start, "test_start": test_start}, h, interval) #Get boolean masks for each fold based on chronological split with embargo
    tbl["fold"] = "embargo"
    for name, m in masks.items(): #Assign the fold name to the corresponding rows based on the boolean mask
        tbl.loc[m, "fold"] = name
    return tbl

# Generate a QA report summarizing the status of the dataset
# 'tbl' is the dataset table, 'config' is the configuration dictionary
# Returns a dictionary containing various QA metrics and statistics
def qa_report(tbl: pd.DataFrame, config: dict) -> dict:
    cloudy = tbl["current_state"] == 1
    lost = tbl[tbl["target_valid"] & ~tbl["features_ok"]] #Rows where the target is valid but required features are missing
    ok = tbl[tbl["row_ok"]] #Rows that are considered okay for analysis
    report = {
        "config": config,
        "n_bins": int(len(tbl)),
        "waterfall": {
            "not_cloudy": int((~cloudy).sum()),
            "cloudy_target_undefined": int((cloudy & ~tbl["target_valid"]).sum()),
            "cloudy_target_ok_features_missing": int(len(lost)),
            "final_rows": int(len(ok)),
        },
        "lost_to_features": {"dissolved": int((lost["target"] == 0).sum()),
                             "persisted": int((lost["target"] == 1).sum())}, #Breakdown of lost rows by target value
        "p_persisted": float(ok["target"].mean()) if len(ok) else None,
        "p_persisted_by_month": ok.groupby(ok.index.month)["target"].mean().round(4).to_dict(), #Proportion of persisted targets by month
        "nan_rate": ok.drop(columns=["target", "fold"], errors="ignore") #Develop NaN rate for each column
                      .isna().mean().round(4).to_dict(),
    }
    if "fold" in ok:
        report["folds"] = {
            k: {"n": int(len(g)), "n_dissolved": int((g["target"] == 0).sum()),
                "p_persisted": float(g["target"].mean())}
            for k, g in ok.groupby("fold")}
    return report

#################
##Main Function##
#################

def main() -> None:

    # Parse command-line arguments and set up configuration
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interim-dir", type=Path, default=DEFAULT_INTERIM) # Directory containing interim data files
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT) # Directory to write output files
    p.add_argument("--start", default="20240101") # Start date for data processing
    p.add_argument("--end", default="20241231") # End date for data processing
    p.add_argument("--interval", default="20s") # Time interval for data aggregation
    p.add_argument("--horizon", default="20min") # Prediction horizon
    p.add_argument("--target", choices=["persistence", "point"], default="persistence")
    p.add_argument("--required", choices=["core", "all"], default="core",
                   help="'all' reproduces the strict filter from notebook 08") #Default to core features, we can add in the additional features later
    p.add_argument("--val-start") # Start date for validation fold
    p.add_argument("--test-start") # Start date for test fold
    p.add_argument("--format", choices=["parquet", "csv"], default="parquet") # Output file format
    args = p.parse_args()

    # Convert Path objects to strings in the configuration dictionary
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    dates = list_dates(args.interim_dir, args.start, args.end) # List of dates to process based on start and end arguments
    print(f"Building from {len(dates)} days ({dates[0]} to {dates[-1]})")

    # Load the state frame for the specified dates
    frame4 = load_state_frame(args.interim_dir, dates)

    # Build the modeling table and compute the horizon
    tbl, h = build_table(frame4, args.interval, args.horizon, args.target, args.required)

    if args.val_start and args.test_start:
        tbl = assign_folds(tbl, args.val_start, args.test_start, h, args.interval) # Assign validation and test folds based on the provided start dates
    elif args.val_start or args.test_start:
        p.error("--val-start and --test-start must be given together")

    report = qa_report(tbl, config) # Generate a QA report for the modeling table
    final = (tbl[tbl["row_ok"]]
             .drop(columns=["features_ok", "row_ok", "target_valid"]))
    final = final.astype({c: "float32" for c in final.select_dtypes("float64").columns})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"modeling_{args.target}_{args.horizon}_{args.interval}"
    out = args.out_dir / f"{stem}.{args.format}" # Output file path for the final dataset
    final.to_parquet(out) if args.format == "parquet" else final.to_csv(out) # Write the final dataset to the specified output format
    (args.out_dir / f"{stem}_qa.json").write_text(json.dumps(report, indent=2)) # Write the QA report to a JSON file
    print(f"Wrote {len(final):,} rows -> {out}")
    print(json.dumps(report["waterfall"], indent=2))


if __name__ == "__main__":
    main()

