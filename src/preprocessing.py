"""
ARM KAZRARSCL preprocessing pipeline.

Created: September 27, 2026
Author: Alec P.

This script selects relevant KAZR variables from the raw KAZRARSCL files, 
and compiles them into a reduced interim dataset for further preprocessing, 
EDA, feature engineering, and model development.

Command-line usage supports processing either a specified date
range or all available observations.
"""

#WARNING: This script is not 100% robust and may require manual verification of the output.

###################
##Library Imports##
###################

from pathlib import Path
import argparse

import pandas as pd
import xarray as xr

#####################
##Feature Selection##
#####################

KAZR_VARIABLES = [
    # Radar measurements
    "reflectivity_best_estimate",
    "mean_doppler_velocity",
    "spectral_width",
    "signal_to_noise_ratio",

    # Measurement quality
    "qc_reflectivity_best_estimate",
    "qc_mean_doppler_velocity",
    "qc_spectral_width",
    "reflectivity_clutter_flag",

    # Cloud structure
    "cloud_base_best_estimate",
    "cloud_layer_base_height",
    "cloud_layer_top_height",
    "cloud_mask_mplzwang",
    "radar_first_top",

    # Instrument / retrieval information
    "instrument_availability_flag",
    "radar_mode_flag",
    "cloud_source_flag",
]

###############
##Parser Args##
###############

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compile ARM KAZRARSCL files into a reduced dataset "
            "for cloud-persistence analysis. (Raw -> Interim)"
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("C:\\Users\\alecp\\MyPrograms\\STA 496\\ARM_Cloud_Modeling\\ARM_DATA\\2024-2025\\Raw\\sgparsclkazr1kolliasC1.c0"),
        help="Directory containing raw KAZR NetCDF files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("C:\\Users\\alecp\\MyPrograms\\STA 496\\ARM_Cloud_Modeling\\ARM_DATA\\2024-2025\\interim\\sgparsclkazr1kolliasC1.c0"),
        help="Directory for compiled KAZR files.",
    )

    parser.add_argument(
        "--start",
        type=str,
        help="First date to process in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--end",
        type=str,
        help="Last date to process in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all available KAZR files.",
    )

    parser.add_argument( #NEW
    "--overwrite",
    action="store_true",
    help="Overwrite compiled files that already exist.",
)

    return parser.parse_args()


###############################
##Processing Helper Functions##
###############################

def select_kazr_variables(ds):
    #This function selects only the relevant KAZR variables from the dataset.
    missing = [
        var for var in KAZR_VARIABLES
        if var not in ds.data_vars
    ]

    if missing:
        raise ValueError(
            f"Missing expected KAZR variables: {missing}"
        )

    return ds[KAZR_VARIABLES]


def get_file_date(file):
    """
    Extract date from an ARM filename such as:
    sgparsclkazr1kolliasC1.c0.20240101.000000.nc
    """
    date_string = file.name.split(".")[-3]

    return pd.to_datetime(
        date_string,
        format="%Y%m%d",
    )

def select_files(files, start=None, end=None):
    """
    Select files within the specified date range.

    Parameters:
    - files: list of Path objects representing ARM files.
    - start: start date as a string in YYYY-MM-DD format or None.
    - end: end date as a string in YYYY-MM-DD format or None.

    Returns:
    - List of Path objects within the date range.
    """
    selected = []

    start = pd.Timestamp(start) if start else None
    end = pd.Timestamp(end) if end else None

    for file in files:
        date = get_file_date(file)

        if start is not None and date < start:
            continue

        if end is not None and date > end:
            continue

        selected.append(file)

    return selected


def compile_kazr_file(input_path, output_dir):
    #This function compiles a single KAZR file by selecting the relevant variables and saving to the output directory.
    output_dir.mkdir(parents=True, exist_ok=True)

    date = get_file_date(input_path)
    date_string = date.strftime("%Y%m%d")

    output_path = output_dir / f"kazr_{date_string}.nc"

    with xr.open_dataset(input_path) as ds:
        subset = select_kazr_variables(ds)
        subset.to_netcdf(output_path)

    return output_path

########
##Main##
########

def main():
    args = parse_args()

    files = sorted(args.input_dir.glob("*.nc")) #Get all NetCDF files in the input directory

    if not files:  #Raise an error if no NetCDF files are found
        raise FileNotFoundError(
            f"No NetCDF files found in {args.input_dir}"
        )

    if args.all: #Select all files if the --all flag is provided
        selected_files = files
    else:
        if args.start is None and args.end is None:  #Ensure date boundary is specified if --all is not used
            raise ValueError(
                "Specify --start/--end or use --all."
            )

        selected_files = select_files( #Select files within the specified date range
            files,
            start=args.start,
            end=args.end,
        )

    if not selected_files:  #Raise an error if no files matched the requested date range
        raise ValueError(
            "No files matched the requested date range."
        )

    print()
    print("KAZR Compilation")
    print("----------------")
    print(f"Available files: {len(files)}")  #Print the total number of available files
    print(f"Selected files:  {len(selected_files)}")  #Print the number of files selected for processing
    print(
        f"Date range:      "  #Print the date range of the selected files
        f"{get_file_date(selected_files[0]).date()} "
        f"to {get_file_date(selected_files[-1]).date()}"
    )
    print(f"Output:          {args.output_dir}")  #Print the output directory where compiled files will be saved
    print()

    for i, file in enumerate(selected_files, start=1):  #Process each selected file and compile it

        date_string = get_file_date(file).strftime("%Y%m%d") #Grab date
        output_path = args.output_dir / f"kazr_{date_string}.nc"

        if output_path.exists() and not args.overwrite:  #Skip processing if the output file already exists and overwrite is not specified
            print(
                f"[{i:03d}/{len(selected_files):03d}] "
                f"Skipping {file.name} (already compiled)"
            )
            continue

        print(
            f"[{i:03d}/{len(selected_files):03d}] "
            f"Processing {file.name}"
        )

        compile_kazr_file(
            file,
            args.output_dir,
        )

    print()
    print(
        f"Finished compiling "
        f"{len(selected_files)} KAZR files."
    )


if __name__ == "__main__":
    main()