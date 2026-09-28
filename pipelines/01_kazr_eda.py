#Welcome. This file was created 9/25/26. 
#This file contains exploratory data analysis for the KAZR dataset.
#That includes exploration of the structure, temporal resolution, quality, and cloud observations contained in the ARM KAZRARSCL dataset. 
#In doing this analysis, we aim to understand how to process a larger set of KAZR data, and generate a clean subset. 

#WARNING
#This script is intended for line-by-line exploratory execution and may be difficult to follow as a standard script. 
#Use with caution for automated processing.

'''
Goals: 
1. Understand the dataset structure and dimensions. 
2. Examine temporal and vertical resolution of the data. 
3. Identify missing and quality-controlled observations. 
4. Visualize radar reflectivity and cloud structure. 
5. Understand how cloud presence and cloud events can be defined. 
6. Identify candidate features for cloud-persistence prediction. 
'''

###################
##Library Imports##
###################

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


####################
##Path Definitions##
####################

DATA_DIR = Path("C:\\Users\\alecp\\MyPrograms\\STA 496\\ARM_Cloud_Modeling\\ARM_DATA\\2024-2025\\Raw\\sgparsclkazr1kolliasC1.c0")
FIGURE_DIR = Path("C:\\Users\\alecp\\MyPrograms\\STA 496\\ARM_Cloud_Modeling\\figures")

kazr_files = sorted(DATA_DIR.glob("*.nc"))

print(f"Found {len(kazr_files)} KAZR files.")

for file in kazr_files[:5]:
    print(file.name)

##########################
##File 0 - Initial Check##
##########################

sample_file = kazr_files[0]
ds = xr.open_dataset(sample_file)
print(f"Opening sample file: {sample_file.name}")
print(ds)

#Dimensions of the dataset
print("Dimensions:")
for dim, size in ds.sizes.items():
    print(f"  {dim}: {size:,}")

#Time range of the dataset
print("\nTime range:")
print("  Start:", ds.time.values[0])
print("  End:  ", ds.time.values[-1])

#Number of variables in the dataset
print("\nNumber of variables:", len(ds.data_vars))

#Coordinate inspection
print("Coordinates:")
print(ds.coords)

print(ds["time"])
print()
print(ds["height"])
print()
print(ds["height"].attrs)

#Height inspection
height_diff = ds.height.diff("height")

print("Minimum height:", ds.height.min().item())
print("Maximum height:", ds.height.max().item())
print("Number of levels:", ds.sizes["height"])
print("Typical spacing:", np.median(height_diff.values))
print("Units:", ds.height.attrs.get("units"))


############################
##Variable Inventory Build##
############################

variable_info = []

for name, da in ds.data_vars.items():
    variable_info.append({
        "variable": name,
        "dimensions": str(da.dims),
        "shape": str(da.shape),
        "units": da.attrs.get("units", ""),
        "description": da.attrs.get("long_name", "")
    })

variable_df = pd.DataFrame(variable_info)

print("Variable Inventory:")
print(variable_df)


#############
##EDA Plots##
#############

#Here, we will produce a three-hour reflectivity 
start = pd.Timestamp(ds.time.values[0])
end = start + pd.Timedelta(hours=3)

ds_3h = ds.sel(time=slice(start, end))

#Plot the three-hour reflectivity data
fig, ax = plt.subplots(figsize=(14, 6))

ds_3h["reflectivity_best_estimate"].plot(
    ax=ax,
    x="time",
    y="height",
    cmap="viridis",
)

ax.set_title("KAZR Best-Estimate Reflectivity")
ax.set_xlabel("Time")
ax.set_ylabel(
    f"Height ({ds.height.attrs.get('units', '')})"
)

plt.tight_layout()
plt.savefig(FIGURE_DIR / "three_hour_reflectivity.png")
plt.show()


#Printing attrubutes for interpreting: 
print("Cloud layer base:")
for key, value in ds["cloud_layer_base_height"].attrs.items():
    print(f"  {key}: {value}")

print("\nCloud layer top:")
for key, value in ds["cloud_layer_top_height"].attrs.items():
    print(f"  {key}: {value}")

print("\nCloud mask:")
for key, value in ds["cloud_mask_mplzwang"].attrs.items():
    print(f"  {key}: {value}")

print("\nClutter flag:")
for key, value in ds["reflectivity_clutter_flag"].attrs.items():
    print(f"  {key}: {value}")

#Overlaying the derived cloud boundaries on the reflectivity plot

cloud_base = ds_3h["cloud_layer_base_height"].where(
    ds_3h["cloud_layer_base_height"] >= 0
)

cloud_top = ds_3h["cloud_layer_top_height"].where(
    ds_3h["cloud_layer_top_height"] >= 0
)

fig, ax = plt.subplots(figsize=(14, 6))

ds_3h["reflectivity_best_estimate"].plot(
    ax=ax,
    x="time",
    y="height",
    cmap="viridis",
)

for layer in ds.layer.values:
    ax.plot(
        ds_3h.time,
        cloud_base.sel(layer=layer),
        linewidth=0.8,
    )

    ax.plot(
        ds_3h.time,
        cloud_top.sel(layer=layer),
        linewidth=0.8,
    )

ax.set_title("KAZR Reflectivity with ARM Cloud-Layer Boundaries")
ax.set_xlabel("Time")
ax.set_ylabel("Height above ground level (m)")

plt.tight_layout()
plt.savefig(FIGURE_DIR / "three_hour_reflectivity_with_cloud_boundaries_filtered.png")
plt.show()

#################
##EDA Questions##
#################

#What constituses a cloudy observation? At each timestamp, does ARM report at least one hydrometeor layer? 
cloudy = (ds["cloud_layer_base_height"] >= 0).any(dim="layer")
print(cloudy)

n_total = cloudy.size
n_cloudy = cloudy.sum().item()
n_clear = n_total - n_cloudy

print(f"Total observations:  {n_total:,}")
print(f"Cloudy observations: {n_cloudy:,} ({n_cloudy/n_total:.1%})")
print(f"Clear observations:  {n_clear:,} ({n_clear/n_total:.1%})")

fig, ax = plt.subplots(figsize=(14, 3))
ax.step(
    ds.time,
    cloudy.astype(int),
    where="post"
)

ax.set_yticks([0, 1])
ax.set_yticklabels(["Clear", "Cloudy"])
ax.set_xlabel("Time")
ax.set_title("ARM Cloud Presence")

plt.tight_layout()
plt.savefig(FIGURE_DIR / "cloud_presence.png")
plt.show()

#How many layers does ARM report simultaneously? 
n_layers = (ds["cloud_layer_base_height"] >= 0).sum(dim="layer")

values, counts = np.unique(n_layers.values, return_counts=True)

print("Number of cloud layers:")
for value, count in zip(values, counts):
    print(
        f"  {value}: {count:,} observations "
        f"({count / n_layers.size:.1%})"
    )

#What if we calculate the fraction of profiles classified as cloudy
cloud_fraction_5min = cloudy.resample(time="5min").mean()
print(cloud_fraction_5min)

print(cloud_fraction_5min.to_series().describe())

fig, ax = plt.subplots(figsize=(14, 4))

ax.plot(
    cloud_fraction_5min.time,
    cloud_fraction_5min,
)

ax.set_ylim(-0.05, 1.05)
ax.set_ylabel("Cloud fraction")
ax.set_xlabel("Time")
ax.set_title("5-Minute Cloud Fraction")

plt.tight_layout()
plt.savefig(FIGURE_DIR / "cloud_fraction_5min.png")
plt.show()

cloudy_values = cloudy.values.astype(int)

transitions = np.diff(cloudy_values)

n_clear_to_cloudy = np.sum(transitions == 1)
n_cloudy_to_clear = np.sum(transitions == -1)

print(f"Clear → Cloudy transitions: {n_clear_to_cloudy:,}")
print(f"Cloudy → Clear transitions: {n_cloudy_to_clear:,}")
print(f"Total state changes:        {np.sum(transitions != 0):,}")

#######################
##Run-length Analysis##
#######################

def get_run_lengths(binary_array, sample_seconds=4):
    values = np.asarray(binary_array).astype(int)

    change_points = np.where(np.diff(values) != 0)[0] + 1
    starts = np.r_[0, change_points]
    ends = np.r_[change_points, len(values)]

    return pd.DataFrame({
        "state": values[starts],
        "n_observations": ends - starts,
        "duration_seconds": (ends - starts) * sample_seconds,
    })


runs = get_run_lengths(cloudy.values)

runs["state"] = runs["state"].map({
    0: "Clear",
    1: "Cloudy",
})

print(runs)
print()
print(runs.groupby("state")["duration_seconds"].describe())

runs["duration_minutes"] = runs["duration_seconds"] / 60

print(
    runs[
        ["state", "n_observations", "duration_minutes"]
    ].to_string(index=False)
)


#Daily summary function
def summarize_kazr_day(ds):
    """
    Calculate basic cloud statistics for one KAZR daily file.
    """

    cloudy = (
        ds["cloud_layer_base_height"] >= 0
    ).any(dim="layer")

    n_total = cloudy.size
    n_cloudy = cloudy.sum().item()
    n_clear = n_total - n_cloudy

    transitions = np.diff(cloudy.values.astype(int))

    n_transitions = np.sum(transitions != 0)

    return {
        "date": pd.Timestamp(ds.time.values[0]).date(),
        "n_observations": n_total,
        "cloudy_fraction": n_cloudy / n_total,
        "clear_fraction": n_clear / n_total,
        "n_transitions": n_transitions,
    }

#Function text on data 0
summary = summarize_kazr_day(ds)
print(summary)

############
##Year EDA##
############

kazr_files = sorted(DATA_DIR.glob("*.nc"))

print(f"Found {len(kazr_files)} files")
print(kazr_files[0].name)
print(kazr_files[-1].name)

#Processing
daily_summaries = []

for i, file in enumerate(kazr_files):

    print(f"Processing {i + 1}/{len(kazr_files)}: {file.name}")

    with xr.open_dataset(file) as ds_day:
        summary = summarize_kazr_day(ds_day)
        daily_summaries.append(summary)

daily_summary_df = pd.DataFrame(daily_summaries)
print(daily_summary_df.head())
daily_summary_df.describe()

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(
    daily_summary_df["date"],
    daily_summary_df["cloudy_fraction"],
)

ax.set_xlabel("Date")
ax.set_ylabel("Fraction of observations classified as cloudy")
ax.set_title("Daily Cloud Fraction — ARM SGP KAZR")

plt.tight_layout()
plt.savefig(FIGURE_DIR / "daily_cloud_fraction.png")
plt.show()