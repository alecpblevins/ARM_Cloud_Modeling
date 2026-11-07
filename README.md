# Statistical Modeling and Prediction of Cloud Persistence Using ARM Radar Observations

**Alec Blevins** \
Fall 2026

**Question:** At the ARM Southern Great Plains (SGP) site, given that the sky is cloudy right now, will it stay cloudy for the *whole* next 20 minutes? 

This repository turns one year (2024) of ARM KAZR radar cloud products into a modeling table, builds past-only features and a future-only target, trains and evaluates models on a strictly later period, and documents every decision along the way. 

> **New here?** Read this README, then open `notebooks/12_results.ipynb`. It is a short guided tour of the pipeline and results, and it runs without the raw data! 

---

## 1. Data

| Item | Value |
|---|---|
| Product | ARM KAZR ARSCL (`sgparsclkazr1kolliasC1.c0`), Ka-band ARM Zenith Radar |
| Site | SGP Central Facility (C1), Lamont, Oklahoma |
| Period | 2024-01-01 to 2024-12-31 (366 daily files) |
| Native resolution | 4 seconds (21,600 samples/day), 596 height gates, up to 10 cloud layers |

The primary cloud state comes from the **cloud-layer product** (`cloud_layer_base_height`), not from raw reflectivity. A base of `-1` is ARM's documented clear-sky sentinel. 

Raw data is not in this repository (roughly 12.3 GB). Download raw data from the ARM Data Discovery portal, and place the files as described in section 4. 

---

## 2. Pipeline at a glance (FIX)

```
raw .c0 files ──► preprocessing ──► qc ──► cloud_state ──► aggregation ──► 

features ─┐
 (ARM, daily)     interim .nc      masks    4 s state      20 s bins       (past only) ├─► build_dataset ──► modeling ──► V2_results/
targets ────┘     parquet + QA     HGB, logit     json/csv/parquet
(future only)

```

| Stage | Module | Input | Output |
|---|---|---|---|
| 1. Select variables | `src/preprocessing.py` | raw daily KAZR files | `interim/kazr_YYYYMMDD.nc` |
| 2. Quality control | `src/qc.py` | interim files | QC-masked daily files |
| 3. Cloud state | `src/cloud_state.py` | one QC'd day | 4 s state table, run segments |
| 4. Aggregate | `src/aggregation.py` | 4 s state | 20 s bins |
| 5. Features | `src/features.py` | bins | 23 past-only features |
| 6. Target | `src/targets.py` | bins | persistence label |
| 7. Build dataset | `src/build_dataset.py` | interim files | modeling parquet + QA report |
| 8. Model | `src/modeling.py` | modeling parquet | results in `V2_results/` |

Key design rules used throughout: 
- **Past-only features, future-only target.** A feature at time *t* never sees data after *t*'s bin. This is tested directly and confirmed. 
- **Unknown is never bridged.** Missing or unreliable bins are left unknown and excluded, never guessed. 
- **Chronological splits with an embargo.** The model is trained on the past and scored on the future. 
- **Test set scored once.** The threshold is chosen on validation and frozen. 

---

## 3. The source modules (`src/`)
Run everything from the project root with `python -m src.<module>`.

### `preprocessing.py`: variable selection
Reduces each raw daily file to the variables the project needs (`KAZR_VARIABLES`) so later steps are fast and the data is small enough to handle. 
- `select_kazr_variables(ds)`: keeps only the needed variables. 
- `get_file_date(file)` / `select_files(files, start, end)`: parse the date from a filename and filter a date range. 
- `compile_kazr_file(input_path, output_dir)`: reduces one file and writes `kazr_YYYYMMDD.nc`. 
- `main()`: CLI with `--input-dir`, `--output-dir`, `--start`, `--end`, `--all`, `--overwrite`. 

### `qc.py`: conservative quality control
Keeps every timestamp and cloud-layer variable, but **masks radar measurements** that are unavailable, fail their ARM QC flag, or are not classified as hydrometeor-only detections (for example clutter). Also decodes whether the radar was available. 
- `apply_qc(ds)`: decodes availability, builds masks from the QC flags and clutter classification, masks the measurement fields, and returns the dataset. 
- `_normalize_missing_value_encoding(ds)`, `_require_variables(ds, requried)`: input checks and cleanup. 
- `process_daily_file(...)`: QC for one file, written to disk. 
- `main()`: CLI with `--input-dir`, `--output-dir`, `--start`, `--end`, `--overwrite`. 

### `cloud_state.py`: from layers to a cloud-state series
Defines the central object of the project: a state at every 4 second sample. 
| State | Rule |
|---|---|
| Cloudy (1) | at least one `cloud_layer_base_height >= 0` |
| Clear (0) | every layer base equals `-1` |
| Unknown (-1) | neither (NaN, unexpected values, downtime) |

- `construct_cloud_state(ds)`: returns a time-indexed frame with state, layer count, lowest base, highest top, and radar availability. 
- `concat_cloud_state_frames(frames)`: joins days into one continuous series so midnight is handled. 
- `segment_runs(df)`: splits the series into maximal runs of the same known state, with duration conventions and flags for runs cut by the record edge. 
- `summarize_run_durations(run_frame)`: summary statistics of run durations. 

### `aggregation.py`: 4 second samples to 20 second bins 
A 20 second bin holds five native samples. The bin is Cloudy, Clear, or Unknown. 
- `aggregate_to_bins(state, interval, step)`: counts cloudy, clear, and unknown samples per left-labelled bin (`origin = "start_day"`). 
- `bin_state(bins)`: turns counts into a bin state (unknown bins stay unknown). 
- `aggregate_cloud_state(state, ...)`: the wrapper that returns the bin counts and the bin state. 

### `features.py`: 23 past-only features
Every feature uses data up to the end of the current bin. 

| Group | Features |
|---|---|
| Cloud fraction and age | `cloud_fraction`, `run_length_min` (minutes cloudy so far) |
| Short-term history | for 1, 5 and 20 min: `cf_mean`, `cf_min`, `cf_change`, `n_flips` (12 features) |
| Vertical structure | `n_layers_mean`, `lowest_base_m`, `highest_top_m`, `vertical_span_m`, `lowest_base_change_5min`, `highest_top_change_5min` |
| Context | `hour_sin`, `hour_cos`, `kazr_avail_frac` |

- `aggregate_structure(frame4, interval)`: aggregates layer-height information to bins. 
- `build_features(bins, s, interval, windows, structure)`: builds all features using trailing windows only. 
- `check_no_lookahead(bins, s, cut, ...)`: truncates the data after a cut time and confirms features up to the cut are unchanged. 

### `targets.py`: what we predict
For every cloudy bin *t*, look at the next *h* bins. 
| Window (t, t+h] | Persistence label |
|---|---|
| every bin cloudy | 1 (persisted) |
| at least one known clear bin | 0 (dissolved, the positive class in modeling) |
| otherwise | invalid (dropped) |

- `horizon_to_bins(horizon, interval)`: "20min" at 20 seconds gives 60 bins. 
- `build_persistence_target(s, h)`: the target used in the project. 
- `build_point_state_target(s, h)`: an alternative that only asks whether the state at *t+h* is cloudy (used as a comparison in notebooks). 
- `build_target(s, h, kind)`: dispatches through `TARGET_BUILDERS`. 
- `chronological_split(index, bounds, h, interval)`: assigns train/val/test with an *h*-bin embargo before each boundary. 

### `build_dataset.py`: the orchestrator
Runs QC in memory, builds state, bins, features and targets, applies the row mask, assigns folds, and writes the table plus a QA report. 
- `list_dates`, `load_state_frame`: find and load interim days. 
- `build_table(frame4, interval, horizon, target_kind, required, check)`: the full chain, including the lookahead check. 
- `assign_folds(tbl, val_start, test_start, h, interval)`: chronological folds. 
- `qa_report(tbl, config)`: row counts, class balance, missingness, fold sizes, saved as `_qa.json`. 
- `main()`: CLI. Main options: `--horizon 20min`, `--interval 20s`, `--target persistence`, `--requried core|all`, `--val-start`, `--test-start`. 

`--requried core` (default) keeps rows when the essential features are present; `all` requires every feature and was found to bias the data (it drops far more dissolving rows than persisting ones). 

### `modeling.py`: models & evaluation
- `load_dataset(path)`: reads the table and returns the feature list and folds. 
- `add_lead_time(df, interval)`: minutes until the cloud dissolves (for analysis only). 
- `make_hgb()`, `make_logit()`: histogram gradient boosting, and standardised logistic regression. 
- `best_f1_threshold(y, p)`: threshold chosen on validation to maximise F1. 
- `metrics(y, p, thr)`: PR-AUC, ROC-AUC, Brier, precision, recall, F1. 
- `operating_points(y, p, thr_source, recalls)`: thresholds for target recall levels. 
- `day_block_bootstrap(index, y, p)`: resamples whole days to give a PR-AUC interval. 
- `rolling_origin(df, features, horizon, months)`: month-by-month train-then-score check. 
- `run(...)` / `main()`: runs everything and writes `results.json`, `predictions.parquet`, `rolling_origin.csv`. 

`GROUP_RULES` assigns every feature to one of five groups for the importance analysis. 

---

## 4. Reproducing results

```bash
pip install -r requirements.txt
```

Expected layout (not tracked by git):

```
ARM_DATA/2024-2025/
├── Raw/sgparsclkazr1kolliasC1.c0/          # downloaded .nc files
├── interim/sgparsclkazr1kolliasC1.c0/      # created by preprocessing
├── qc/sgparsclkazr1kolliasC1.c0/           # created by qc (optional, for notebooks)
└── modeling/                               # created by build_dataset
```

```bash
python -m src.preprocessing --all
python -m src.qc #optional 
python -m src.build_dataset --val-start 2024-09-01 --test-start 2024-11-01
python -m src.modeling
```

`src.modeling` writes to `ARM_DATA/.../modeling/results/`. The committed copies live in `V2_results/`. Seeds are fixed (`SEED = 0`). 

**Without the raw data:** open `notebooks/12_results.ipynb`. It reads only `V2_results/`. 

---

## 5. Notebook timeline
The notebooks were written in order, each one motivating a source module. 

| # | Notebook | What it does | Leads to |
|---|---|---|---|
| 01 | `01_data_preparation` | Data source, raw file structure, variable selection | `preprocessing.py` |
| 02 | `02_qc_investigation` | Separates instrument unavailable, invalid measurement, and no detection; designs QC rules | `qc.py` |
| 03 | `03_cloud_state` | Builds the cloud state from layer bases on three days; checks sentinels and time axis | `cloud_state.py` |
| 04 | `04_temporal_dynamics` | Cloudy/clear run durations and aggregation intervals on three days | choice of 20 s bins |
| 05 | `05_annual_temporal_dynamics` | Repeats 04 over all of 2024; audits missing days | confirms the 20 s choice |
| 06 | `06_target_sketch` | Validates `cloud_state.py` against 03/04; sketches target definitions and horizons | target design |
| 07 | `07_annual_target_sketch` | Full-year class balance by target and horizon; picks persistence at 20 min | `aggregation.py`, `targets.py` |
| 08 | `08_build_dataset` | Walks each module on a few days including edge cases, then a full-year run | `features.py`, `build_dataset.py` |
| 09 | `09_modeling` | Baselines, feature ladder, HGB, threshold, test scoring, rolling origin, error analysis | `modeling.py` |
| 10 | `10_analysis` | Operating points, lead time, old-cloud analysis, permutation importance by group | interpretation |
| 11 | `11_radar_profiles` | Side study: whether radar reflectivity features add skill (small, uncertain gain; excluded) | future work |
| 12 | `12_results` | Guided tour of the pipeline and the final results | this README's companion |

Figures are saved in `figures/`. 

---

## 6. Repository map

```
src/                pipeline modules (section 3)
notebooks/          exploration, validation, and results (section 5)
V2_results/         small final outputs: results.json, rolling_origin.csv, predictions parquet
figures/            saved figures
requirements.txt
```

## 7. Evaluation conventions
- **Positive class = dissolved.** A moment is positive if the cloud disappears within the horizon. 
- **PR-AUC** is the main metric: how well dissolving moments are ranked above persisting ones, with precision counted. Its no-skill value equals the dissolved rate, so compare it against that. 
- **ROC-AUC** is more stable across periods with different base rates. 
- **Threshold** converts a proabillity into an alarm. It is chosen on validation and applied unchanged to test. 
- **Embargo** is a gap of one horizon (20 minutes) before each split boundary, so no training target looks into the next fold. 
- **Day-block bootstrap** resamples whole days (not rows), because neighbouring rows are nearly identical. It gives the 95% interval on test PR-AUC. 
- **Rolling origin** trains on all months before *m* and scores month *m*, for April to December. It checks stability across the year. 
- **Run length** is minutes the current cloud has been continuously present (cloud age). 
- **Lead time** is minutes until a dissolving cloud actually dissolves (analysis only). 

---

## 8. Results & limitations
Results are in `notebooks/12_results.ipynb` (summary at the end). 

**Headline:** on the held-out November to December test fold, the gradient boosting model reaches PR-AUC 0.646 [0.612, 0.680] against a no-skill level of 0.183 and baselines of 0.588 (cloud age only) and 0.606 (logistic). 

**Limitations:** one year and one site; the target is persistence of a state product, not of a tracked cloud; the interval reflects sampling noise only; run length is censored at record edges; importance shows reliance, not causation. 

## 9. Future Work: *How long will a cloud last?*
The current model answers a conditional question: given that a cloud is present now, will it dissolve within the next 20 minutes? A natural next step is to ask the question from the other side: *given that a cloudy period has just started, how long will it last?* This turns project into a **time-to-event (survival) problem**. 

The existing pipeline already provides most of what is needed. `segment_runs` in `cloud_state.py` splits the record into maximal cloudy runs with their durations, and flags the runs that were cut off by the edge of the record or by unknown data. Those runs are **right-censored**: we only know they lasted *at least* that long, so they cannot simply be dropped or treated as complete without biasing durations short. 

Therefore, the dataset would change from one row per 20-second bin to one row per run. Each row would hold information available at the start of the run (time of day, month, the length of the preceding clear gap, the previous run's duration, early vertical structure) and the run's duration as the outcome. Radar observations could also be used here. 

If time allows, a first version could estimate Kaplan-Meier survival curves of run duration, split by season or time of day. A second could fit a Cox proportional-hazards (evaluating the effect of one or more predictors on the time-to-event) or Weibull model (to model time-to-failure data), evaluated chronologically with the concordance index and calibration of the predicted survival curve. 

However, the two models would be connected, through: the 20 minute dissolution probability for a cloud of age *a* is already a slice of the same survival curve, so the survival estimates provide an independent check on this existing model. 