#This script was created 10/15/2026. 
#This file contains a quick check of the qc.py pipeline, comparing output against the original. 


######################
##Path Configuration##
######################

from pathlib import Path
import xarray as xr

interim_dir = Path(
    "C:/Users/alecp/MyPrograms/STA 496/ARM_Cloud_Modeling/"
    "ARM_DATA/2024-2025/interim/sgparsclkazr1kolliasC1.c0"
)
qc_dir = Path(
    "C:/Users/alecp/MyPrograms/STA 496/ARM_Cloud_Modeling/"
    "ARM_DATA/2024-2025/qc/sgparsclkazr1kolliasC1.c0"
)

for interim_file in sorted(interim_dir.glob("kazr_*.nc")):
    qc_file = qc_dir / interim_file.name
    if not qc_file.exists():
        print(f"[SKIP] no QC counterpart for {interim_file.name}")
        continue

    with xr.open_dataset(interim_file) as raw, xr.open_dataset(qc_file) as qc:
        print(f"--- {interim_file.name} ---")
        print("Dimensions unchanged:", raw.sizes == qc.sizes)
        print("Timestamps unchanged:", raw["time"].equals(qc["time"]))
        print(
            "Cloud-layer bases unchanged:",
            raw["cloud_layer_base_height"].equals(qc["cloud_layer_base_height"]),
        )
        print("KAZR availability:", qc["kazr_available"].mean().item())
        print(
            "Valid reflectivity fraction:",
            qc["reflectivity_best_estimate"].notnull().mean().item(),
        )
        cloudy = (qc["cloud_layer_base_height"] >= 0).any(dim="layer")
        print("Cloudy timestamp fraction:", cloudy.mean().item())


#Mask check 
for date in ["20240101", "20240102", "20240103"]:
    raw_file = interim_dir / f"kazr_{date}.nc"
    qc_file = qc_dir / f"kazr_{date}.nc"

    with xr.open_dataset(raw_file) as raw, xr.open_dataset(qc_file) as qc:
        expected = (
            raw["reflectivity_best_estimate"].notnull()
            & (qc["kazr_available"] == 1)
            & (qc["qc_reflectivity_best_estimate"] == 0)
            & (qc["reflectivity_clutter_flag"] == 1)
        )

        observed = (
            qc["reflectivity_best_estimate"].notnull()
        )

        print(
            date,
            "Mask matches:",
            bool((expected == observed).all().item())
        )


#Jan 3rd investigation - Why so low reflectivity? 
import numpy as np
import pandas as pd
import xarray as xr

for date in ["20240101", "20240102", "20240103"]:
    file = interim_dir / f"kazr_{date}.nc"

    with xr.open_dataset(file) as ds:
        refl = ds["reflectivity_best_estimate"]
        qc = ds["qc_reflectivity_best_estimate"]
        clutter = ds["reflectivity_clutter_flag"]

        # Is a reflectivity value present?
        has_refl = refl.notnull()

        # Does the measurement pass its own QC test?
        qc_pass = qc.notnull() & (qc == 0)

        # Is KAZR available at this timestamp?
        availability = (
            ds["instrument_availability_flag"]
            .fillna(0)
            .astype(np.int64)
        )
        kazr_available = (availability & 1) != 0

        # Successive filtering stages
        raw_present = has_refl
        passed_qc = has_refl & qc_pass
        available_and_qc = passed_qc & kazr_available
        current_mask = available_and_qc & (clutter == 1)

        stages = {
            "Raw reflectivity present": raw_present,
            "Present + QC pass": passed_qc,
            "Present + QC pass + KAZR available": available_and_qc,
            "Current mask (also clutter == 1)": current_mask,
        }

        total_cells = refl.size

        print(f"\n{'=' * 55}")
        print(date)
        print(f"Total time-height cells: {total_cells:,}")

        for label, mask in stages.items():
            n = int(mask.sum().item())
            pct = 100 * n / total_cells
            print(f"{label:42s} {n:>10,} ({pct:7.4f}%)")

        candidate_count = int(available_and_qc.sum().item())
        retained_count = int(current_mask.sum().item())

        if candidate_count:
            retained_pct = 100 * retained_count / candidate_count
            print(
                f"Retained after clutter filter: "
                f"{retained_pct:.2f}% of eligible candidates"
            )

        # Which clutter categories occur among measurements
        # that passed reflectivity QC and the availability check?
        print("\nClutter breakdown before clutter filtering:")

        for flag, name in [
            (0, "No significant detection"),
            (1, "Hydrometeor only"),
            (2, "Hydrometeor + clutter"),
            (3, "Clutter only"),
            (9, "Bad data"),
            (10, "Missing data"),
        ]:
            category = available_and_qc & (clutter == flag)
            n = int(category.sum().item())
            pct = 100 * n / candidate_count if candidate_count else 0

            print(f"  {flag}: {name:29s} {n:>10,} ({pct:7.2f}%)")




#####Another diagnostic: Investigate low reflectivity on Jan 3rd

file = interim_dir / "kazr_20240103.nc"

with xr.open_dataset(file) as ds:
    reflectivity = ds["reflectivity_best_estimate"]

    # Does any height have a nonmissing reflectivity value?
    has_radar_return = reflectivity.notnull().any(dim="height")

    # Does the cloud-layer product report any cloud layer?
    cloudy = (
        ds["cloud_layer_base_height"] >= 0
    ).any(dim="layer")

    print("Reflectivity variable metadata:")
    print("Attributes:", reflectivity.attrs)
    print("Encoding:", reflectivity.encoding)

    print("\nReflectivity summary:")
    print("Nonmissing cells:", int(reflectivity.notnull().sum().item()))
    print("Timestamps with any radar return:",
          int(has_radar_return.sum().item()))
    print("Cloudy timestamps:", int(cloudy.sum().item()))

    # Compare cloud state with the presence of radar returns.
    comparison = pd.crosstab(
        pd.Series(cloudy.values, name="Cloudy"),
        pd.Series(has_radar_return.values, name="Radar return"),
        margins=True
    )
    print("\nCloud state versus radar returns:")
    print(comparison)

    # Summarize how the data vary throughout the day.
    hourly = pd.DataFrame(
        {
            "reflectivity_cells": (
                reflectivity.notnull().sum(dim="height").values
            ),
            "timestamps_with_radar": has_radar_return.values,
            "cloud_fraction": cloudy.values,
        },
        index=pd.DatetimeIndex(ds.time.values),
    )

    hourly_summary = hourly.resample("1h").agg(
        {
            "reflectivity_cells": "sum",
            "timestamps_with_radar": "sum",
            "cloud_fraction": "mean",
        }
    )

    print("\nHourly summary:")
    print(hourly_summary)


    ####Verifying intrium against raw

raw_file = (
    "C:/Users/alecp/MyPrograms/STA 496/ARM_Cloud_Modeling/ARM_DATA/2024-2025/Raw/sgparsclkazr1kolliasC1.c0/sgparsclkazr1kolliasC1.c0.20240103.000000.nc"
)

interim_file = interim_dir / "kazr_20240103.nc"

variables = [
    "reflectivity_best_estimate",
    "qc_reflectivity_best_estimate",
    "mean_doppler_velocity",
    "spectral_width",
    "signal_to_noise_ratio",
]

with xr.open_dataset(raw_file) as raw, xr.open_dataset(interim_file) as interim:
    print("Raw dimensions:", dict(raw.sizes))
    print("Interim dimensions:", dict(interim.sizes))

    print("\nNonmissing measurement counts:")

    for variable in variables:
        if variable not in raw or variable not in interim:
            print(f"{variable}: missing from one dataset")
            continue

        raw_count = int(raw[variable].notnull().sum().item())
        interim_count = int(interim[variable].notnull().sum().item())

        print(
            f"{variable:35s} "
            f"raw={raw_count:>10,}  "
            f"interim={interim_count:>10,}  "
            f"match={raw_count == interim_count}"
        )