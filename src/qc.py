"""
ARM KAZRARSCL QC pipeline.

Created: October 14, 2026
Author: Alec P.

This script applies conservative quality-control masks to interim ARM KAZR files; 
This module turns one interim dily file into a QC-stage file. It preserves all timestams and cloud-layer variables, 
but masks radar measurements that are unavilable, fail their ARM QC test, or are not classified as hydrometeor-only detections. 

Notes:
This pipeline has two main jobs: 
1. apply_qc(ds): Takes one dataset already loaded, decodes instrument availability, constructs masks from the QC flags and clutter classification, 
masks selected measurement fields, and retuns the modified dataset. 
2. Finds daily files, selects the dates to process, calls the helper function, writes outputs, and reports sucesses or errors. 

By default, output files are written to `ARM_DATA/2024-2025/qc/sgparsclkazr1kolliasC1.c0/`

Command-line usage supports processing either a specified date
range or all available observations.
"""

###################
##Library Imports##
###################

#from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr

######################
##Path Configuration##
######################

#qc.py is located in <project-root>/src/qc.py.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "ARM_DATA"
    / "2024-2025"
    / "interim"
    / "sgparsclkazr1kolliasC1.c0"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "ARM_DATA"
    / "2024-2025"
    / "qc"
    / "sgparsclkazr1kolliasC1.c0"
)

########################
##Variable Definitions##
########################

# Measurement variable -> its corresponding ARM QC flag.
MEASUREMENT_QC_VARIABLES = {
    "reflectivity_best_estimate": "qc_reflectivity_best_estimate",
    "mean_doppler_velocity": "qc_mean_doppler_velocity",
    "spectral_width": "qc_spectral_width",
}

####ASSUMPTION####
#Right now, we will preserve SNR without applying any QC mask, as there is no dedicated SNR QC variable. 

REQUIRED_VARIABLES = {
    "instrument_availability_flag",
    "reflectivity_clutter_flag",
    *MEASUREMENT_QC_VARIABLES.keys(),
    *MEASUREMENT_QC_VARIABLES.values(),
}

####################
##Helper Functions##
####################

# Helper function to ensure required variables are present in the dataset.
def _require_variables(ds: xr.Dataset, required: Iterable[str]) -> None:
    """Raise a useful error if an expected interim variable is absent."""
    missing = sorted(set(required) - set(ds.variables))
    if missing:
        raise ValueError(
            "Input dataset is missing required QC variables: " + ", ".join(missing)
        )

def _normalize_missing_value_encoding(ds: xr.Dataset) -> None:
    """Align conflicting NetCDF fill-value metadata before writing.

    Some interim variables retain ARM's ``missing_value`` attribute (for
    example, -9999.0) while their xarray encoding has ``_FillValue=NaN``.
    Xarray refuses to serialize that conflict. When both are present and
    disagree, use the declared ``missing_value`` as the output ``_FillValue``.
    NaNs in memory will then be encoded using that marker and decoded back to
    NaNs by xarray when the file is reopened.
    """
    for name in ds.variables:
        variable = ds[name]

        # xarray (v2022.03+) stores these in .encoding after decoding.
        fill_value = variable.encoding.get("_FillValue")
        missing_value = variable.encoding.get("missing_value")

        # Be defensive: some code paths (e.g. freshly constructed variables)
        # may still leave missing_value in attrs.
        if missing_value is None:
            missing_value = variable.attrs.get("missing_value")

        if fill_value is None or missing_value is None:
            continue

        try:
            values_match = np.array_equal(
                np.asarray(fill_value),
                np.asarray(missing_value),
                equal_nan=True,
            )
        except TypeError:
            values_match = np.array_equal(
                np.asarray(fill_value), np.asarray(missing_value)
            )

        if not values_match:
            variable.encoding["_FillValue"] = missing_value

#########################
##MAIN QC RULE FUNCTION##
#########################

def apply_qc(ds: xr.Dataset) -> xr.Dataset:
    """Return a QC-masked view of an interim KAZR dataset.

    Rules used in this initial implementation:
    - KAZR availability is decoded from bit mask 1 of
      ``instrument_availability_flag``.
    - A radar measurement passes its ARM measurement QC when its QC flag is 0.
    - Only clutter flag 1 (hydrometeor-only detection) is retained for radar
      measurement features. Mixed hydrometeor/clutter (2), clutter-only (3),
      no significant detection (0), and bad/missing clutter categories are
      masked out of the measurement fields.
    - SNR has no dedicated QC flag in the interim variable set, so it 
    is not explicitly masked based on QC in this initial implementation.
    - Cloud-layer base/top heights and their -1 clear-sky sentinel values are
      preserved unchanged. Timestamps are never removed or resampled.

    The supplied dataset is not modified in place. Original QC flags and
    clutter categories are retained so the masking remains auditable.
    """
    _require_variables(ds, REQUIRED_VARIABLES)

    # The input availability variable is time-level (one value per timestamp).
    #Fill missing availability flags with 0 so unknown availability does not accidentally count as available.
    #Therefore, check if KAZR is actually available: 
    availability = ds["instrument_availability_flag"].fillna(0).astype(np.int64)
    kazr_available = (availability & 1) != 0

    #This is a time-height mask. It is only the common KAZR-available and hydrometeor-only condition
    #each field also has to pass its own QC flag.
    clutter = ds["reflectivity_clutter_flag"]
    kazr_hydrometeor_only = kazr_available & (clutter == 1) #Which heights qualify under the clutter rule? 

    #Start with a copy, original variables/flags are retained, and only 
    #the measurement variables plus the two diagnostic masks are added or replaced in the returned dataset.
    out = ds.copy(deep=False)
    out.attrs = dict(ds.attrs)

    out["kazr_available"] = kazr_available.astype(np.int8)
    out["kazr_available"].attrs = {
        "long_name": "KAZR instrument availability decoded from the availability bit field",
        "units": "1",
        "flag_values": np.array([0, 1], dtype=np.int8),
        "flag_meanings": "unavailable available",
        "comment": "Derived from bit mask 1 of instrument_availability_flag.",
    }

    out["qc_kazr_hydrometeor_only_mask"] = kazr_hydrometeor_only.astype(np.int8)
    out["qc_kazr_hydrometeor_only_mask"].attrs = {
        "long_name": "KAZR available and reflectivity clutter flag indicates hydrometeor-only detection",
        "units": "1",
        "flag_values": np.array([0, 1], dtype=np.int8),
        "flag_meanings": "condition_not_met condition_met",
        "comment": (
            "A value of 1 means KAZR is available and reflectivity_clutter_flag == 1. "
            "The associated measurement QC flag must also equal 0 for a measurement "
            "to be retained. This mask is not a cloud-state label."
        ),
    }

    # Apply measurement-specific QC and the common availability/clutter mask.
    for variable, qc_variable in MEASUREMENT_QC_VARIABLES.items():
        qc_pass = ds[qc_variable].notnull() & (ds[qc_variable] == 0)
        valid = kazr_hydrometeor_only & qc_pass #A measurement must pass both the shared mask and its own QC test. 
        cleaned = ds[variable].where(valid)
        cleaned.attrs = dict(ds[variable].attrs)
        cleaned.attrs["qc_applied"] = (
            f"Masked unless KAZR available, {qc_variable} == 0, and "
            "reflectivity_clutter_flag == 1 (hydrometeor-only)."
        )
        out[variable] = cleaned

    out.attrs["qc_pipeline"] = "Initial QC mask implemented in src/qc.py"
    out.attrs["qc_policy"] = (
        "Preserve timestamps and cloud-layer state. Mask radar measurements unless "
        "KAZR is available, the relevant QC flag equals 0, and the clutter flag "
        "indicates hydrometeor-only detection (flag 1)."
    )
    return out

# Helper functions for file handling and processing
def _date_from_filename(path: Path):
    """Extract YYYYMMDD from names such as kazr_20240101.nc."""
    match = re.search(r"(\d{8})", path.name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError:
        return None

# Process daily files with QC applied
def process_daily_file(
    input_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False, #Implement overwrite option for existing output files
) -> str:
    """Apply QC to one file and write it to output_path.

    Returns ``"written"`` or ``"skipped"``. A temporary file is used so an
    interrupted write does not leave a partial final output file.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        return "skipped"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")

    try:
        with xr.open_dataset(input_path) as source:
            qc_ds = apply_qc(source) #APPLY!
            _normalize_missing_value_encoding(qc_ds) #Ensure consistent missing value encoding before writing
            qc_ds.to_netcdf(temporary_path) #Write the QC-applied dataset to a temporary file
        temporary_path.replace(output_path) #Replace the temporary file with the final output file
    except Exception:
        # Clean up a partial temporary file, but leave any pre-existing final
        # file untouched when overwrite=True and a write fails.
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    return "written"

# Command-line interface helpers
def parse_date_arg(value: str):
    """Parse a CLI date written as YYYYMMDD."""
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date {value!r}; expected YYYYMMDD, e.g. 20240101"
        ) from exc

########
##MAIN##
########

def main(argv: list[str] | None = None) -> int:

    #Parser Args
    parser = argparse.ArgumentParser(
        description="Apply QC masks to daily interim ARM KAZR NetCDF files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory of interim kazr_YYYYMMDD.nc files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for QC files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--start", type=parse_date_arg, help="First date to process, inclusive (YYYYMMDD).")
    parser.add_argument("--end", type=parse_date_arg, help="Last date to process, inclusive (YYYYMMDD).")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output files that already exist.",
    )
    args = parser.parse_args(argv)

    #Add in safety arguement checks
    if args.start and args.end and args.start > args.end:
        parser.error("--start must be earlier than or equal to --end.")
    if not args.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.input_dir}")

    #Load files
    files = sorted(args.input_dir.glob("kazr_*.nc"))
    if args.start or args.end:
        selected = []
        for path in files:
            file_date = _date_from_filename(path)
            if file_date is None:
                print(f"[WARN] Could not read date from filename; excluding {path.name}")
                continue
            if args.start and file_date < args.start:
                continue
            if args.end and file_date > args.end:
                continue
            selected.append(path)
        files = selected

    if not files:
        parser.error(f"No matching kazr_*.nc files found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = skipped = failed = 0

    #Process each file
    for index, input_path in enumerate(files, start=1):
        output_path = args.output_dir / input_path.name
        print(f"[{index:03d}/{len(files):03d}] {input_path.name}", flush=True)
        try:
            status = process_daily_file(
                input_path, output_path, overwrite=args.overwrite
            )
            if status == "written":
                written += 1
                print(f"    Wrote {output_path}", flush=True)
            else:
                skipped += 1
                print(f"    Skipped existing output: {output_path}", flush=True)
        except Exception as exc:  # continue so one corrupt file does not halt the batch
            failed += 1
            print(f"    ERROR processing {input_path.name}: {exc}", file=sys.stderr, flush=True)

    print(
        f"\nQC batch complete: {written} written, {skipped} skipped, {failed} failed."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())