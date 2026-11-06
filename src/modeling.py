#This module will freeze only what notebooks 09 and 10 settled: the 23 feature HGB at a 20 min horizon, with the threshold chosen on val and test scored once. 
#This module was created on created 11/5/2025 by Alec P.

"""
Fit and evaluate the cloud-persistence model.

Task: among cloudy 20 s bins, predict whether the cloud dissolves (any known clear bin)
within the horizon. Positive class = dissolved (target == 0).

Pipeline: load modeling parquet 
-> baselines (no-skill, -run_length, logistic) 
-> HGB
-> threshold chosen on val only 
-> test scored once with day-block bootstrap CI
-> rolling-origin by month 
-> write results JSON, predictions parquet, rolling CSV.

Usage:
    python -m src.modeling
    python -m src.modeling --data path/to/modeling_persistence_20min_20s.parquet
"""

###################
##Library Imports##
###################

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

##############
##Path Setup##
##############

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = (PROJECT_ROOT / "ARM_DATA" / "2024-2025" / "modeling"
                / "modeling_persistence_20min_20s.parquet")

NON_FEATURES = {"current_state", "target", "fold", "y"}
SEED = 0
OLD_CLOUD_MIN = 60   # "old cloud" threshold in minutes of run length

# Every feature belongs to exactly one group (used for importance and the logistic ladder).
GROUP_RULES = {
    "run_length": lambda c: c == "run_length_min",
    "cloud_fraction": lambda c: c.startswith(("cloud_fraction", "cf_", "n_flips")),
    "structure": lambda c: c in {"n_layers_mean", "lowest_base_m", "highest_top_m", "vertical_span_m",
                                 "lowest_base_change_5min", "highest_top_change_5min"},
    "availability": lambda c: c == "kazr_avail_frac",
    "time_of_day": lambda c: c in {"hour_sin", "hour_cos"},
}

#############
##Load data##
#############

#Helper function for loading and processing dataset
def load_dataset(path: Path):
    """Return (df, features). y = 1 means dissolved."""
    df = pd.read_parquet(path).sort_index() #Grab df

    #Catch required columns missing from the dataframe
    required = {"current_state", "target", "fold"}
    if not required <= set(df.columns):
        raise ValueError(f"missing columns {required - set(df.columns)}; rebuild with --val-start/--test-start")
    if not df.index.is_unique:
        raise ValueError("index must be unique")
    assert (df.current_state == 1).all(), "table must contain cloudy bins only"

    df["y"] = (df.target == 0).astype(int) #Create binary target column: 1 if dissolved, 0 otherwise
    features = [c for c in df.columns if c not in NON_FEATURES] #Select feature columns by excluding non-features
    groups = {g: [c for c in features if rule(c)] for g, rule in GROUP_RULES.items()} #Group features according to GROUP_RULES
    assert sorted(sum(groups.values(), [])) == sorted(features), "every feature must be in exactly one group"
    return df, features, groups

#Function to add lead time column to the dataframe
def add_lead_time(df: pd.DataFrame, interval: str = "20s") -> pd.DataFrame:
    """Minutes until the cloudy run containing each row ends, read from gaps in the index.
    Approximate where the first non-cloudy bin is unknown rather than clear (notebook 10, sec. 3)."""
    step = pd.Timedelta(interval) #Convert interval string to Timedelta object
    s = df.index.to_series()
    run_id = (s.diff() != step).cumsum() #Identify contiguous runs of cloudy bins based on the index difference
    out = df.copy()
    out["lead_min"] = ((s.groupby(run_id).transform("max") - s) + step).dt.total_seconds() / 60 #Calculate lead time in minutes for each row
    return out

#############
##Metrics####
#############

def best_f1_threshold(y, p) -> float:
    """Probability cutoff maximising F1. Choose on validation only."""
    pr, rc, th = precision_recall_curve(y, p) #Compute precision-recall curve
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None) #compute F1 score for each threshold
    return float(th[np.argmax(f1)]) #Return max

#Metrics function to compute various evaluation metrics
def metrics(y, p, thr=None) -> dict:
    out = dict(n=int(len(y)), rate=float(np.mean(y)), #Dictionary to store evaluation metrics
               pr_auc=float(average_precision_score(y, p)),
               roc_auc=float(roc_auc_score(y, p)),
               brier=float(brier_score_loss(y, np.clip(p, 0, 1))))
    if thr is not None: #If a threshold is provided, compute additional metrics based on the threshold
        yhat = (p >= thr).astype(int) #Predicted labels based on the threshold
        out |= dict(thr=float(thr), #Store threshold and corresponding metrics
                    precision=float(precision_score(y, yhat, zero_division=0)),
                    recall=float(recall_score(y, yhat)),
                    f1=float(f1_score(y, yhat)))
    return out

#Function to compute operating points for given target recalls
def operating_points(y, p, thr_source, recalls=(0.5, 0.6, 0.7, 0.8, 0.9)) -> list[dict]:
    """Thresholds chosen on the source fold (val) to reach a target recall, evaluated on y, p.
    y: true labels for evaluation
    p: predicted probabilities for evaluation
    thr_source: tuple of (y_source, p_source) used to determine thresholds
    recalls: target recall levels to compute operating points for
    Returns a list of dictionaries with target recall, threshold, recall, precision, and alarm rate."""

    y_s, p_s = thr_source #Unpack source fold labels and predictions
    _, rc, th = precision_recall_curve(y_s, p_s) #Compute precision-recall curve on source fold
    rows = []
    for r in recalls:
        t = float(th[np.flatnonzero(rc[:-1] >= r)[-1]]) #Select threshold achieving target recall on source fold
        yhat = p >= t #Predicted labels based on threshold
        tp = int((yhat & (y == 1)).sum()) #True positives
        rows.append(dict(target_recall=r, thr=t, recall=tp / y.sum(),
                         precision=tp / max(int(yhat.sum()), 1), alarm_rate=float(yhat.mean()))) #Store operating point metrics
    return rows

#Function to perform day-block bootstrap resampling for PR-AUC
def day_block_bootstrap(index: pd.DatetimeIndex, y, p, n_boot: int = 200, seed: int = SEED) -> list[float]:
    """PR-AUC resampled by whole days: neighbouring bins are not independent."""
    codes, _ = pd.factorize(index.normalize()) #Factorize dates to identify unique days
    order = np.argsort(codes, kind="stable") #Sort indices by day codes
    groups = np.split(order, np.flatnonzero(np.diff(codes[order])) + 1) #Split indices into groups by day
    rng = np.random.default_rng(seed) 
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) #Sample days with replacement
        out.append(average_precision_score(y[idx], p[idx])) #Compute PR-AUC for resampled indices
    return out #Return list of resampled PR-AUC scores

#############
##Models#####
#############

#HGB Model function
def make_hgb(max_iter: int = 300, early_stopping: bool = True) -> HistGradientBoostingClassifier:
    # early stopping holds out a random 10% of train; autocorrelation makes this slightly optimistic
    return HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=max_iter,
                                          early_stopping=early_stopping, validation_fraction=0.1,
                                          random_state=SEED)

#Logistic Regression Model function
def make_logit():
    # imputer medians and scaler are fit on the training data only
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                         StandardScaler(), LogisticRegression(max_iter=1000))

#############
##Evaluation#
#############

#Function to perform rolling-origin evaluation on time-series data
def rolling_origin(df, features, horizon: pd.Timedelta, months=range(4, 13)) -> pd.DataFrame: #Months April - December, can't start earlier than April. 
    """Train on everything before month m (minus an embargo), score month m. HGB without early stopping."""
    rows = []
    for m in months:
        start = pd.Timestamp(f"{df.index[0].year}-{m:02d}-01") #Start of the evaluation month
        end = start + pd.offsets.MonthBegin(1) #End of the evaluation month
        a = df[df.index < start - horizon] #Training data before the evaluation month (minus embargo)
        b = df[(df.index >= start) & (df.index < end)] #Evaluation data for the current month
        if len(a) == 0 or len(b) == 0 or b.y.nunique() < 2:
            continue #Skip if there is no training or evaluation data, or if the evaluation data has less than 2 classes
        model = make_hgb(max_iter=200, early_stopping=False).fit(a[features], a.y) #Fit HGB model on training data
        rows.append(dict(month=m, **metrics(b.y.values, model.predict_proba(b[features])[:, 1]))) #Compute metrics on evaluation data
    return pd.DataFrame(rows)

#RUN FUNCTION: defining a function to run the modeling pipeline on the dataset
def run(data: Path, out_dir: Path, horizon: str = "20min", interval: str = "20s") -> dict:
    df, features, groups = load_dataset(data) #Load dataset and extract features and groups
    df = add_lead_time(df, interval) #Add lead time feature to the dataset
    tr, va, te = (df[df.fold == k] for k in ["train", "val", "test"]) #Split dataset into train, val, and test folds
    if min(len(tr), len(va), len(te)) == 0:
        raise ValueError("train, val and test folds must all be non-empty") #Ensure all folds are non-empty
    print(f"{len(features)} features | train {len(tr):,} val {len(va):,} test {len(te):,}") #Print dataset summary

    # 1) no-learning baselines (all scored on val; test numbers are reported once below)
    base = {"always_persist_accuracy": {"val": float(1 - va.y.mean()), "test": float(1 - te.y.mean())}}
    score_rl = lambda d: -d.run_length_min.values            # higher score = younger cloud = more likely dissolved

    # 2) models fit on train only
    hgb = make_hgb().fit(tr[features], tr.y) #Fit!
    logit = make_logit().fit(tr[features], tr.y) #Fit logistic regression model on training data
    p = {name: {k: m.predict_proba(d[features])[:, 1] for k, d in [("val", va), ("test", te)]} #Predict probabilities for val and test sets
         for name, m in [("hgb", hgb), ("logit", logit)]}

    # 3) thresholds from val only, then test scored once
    thr = {name: best_f1_threshold(va.y.values, p[name]["val"]) for name in p} #Determine best F1 thresholds based on validation set
    results = {"config": {"data": str(data), "horizon": horizon, "interval": interval, "seed": SEED, #Configuration details for the run
                          "n_features": len(features), "features": features},
               "baselines": base, "val": {}, "test": {}}
    for name in p: #Compute metrics for each model on validation and test sets
        results["val"][name] = metrics(va.y.values, p[name]["val"], thr[name]) #Compute metrics for validation set
        results["test"][name] = metrics(te.y.values, p[name]["test"], thr[name]) #Compute metrics for test set
    results["val"]["-run_length"] = {k: v for k, v in metrics(va.y.values, score_rl(va)).items() if k != "brier"} #Compute run length metrics for validation set
    results["test"]["-run_length"] = {k: v for k, v in metrics(te.y.values, score_rl(te)).items() if k != "brier"} #Compute run length metrics for test set
    results["test"]["no_skill"] = dict(pr_auc=float(te.y.mean()), roc_auc=0.5,
                                       brier=float(te.y.mean() * (1 - te.y.mean()))) #No-skill baseline for test set

    ci = np.percentile(day_block_bootstrap(te.index, te.y.values, p["hgb"]["test"]), [2.5, 97.5]) #Compute 95% CI for PR AUC using day-block bootstrap
    results["test"]["hgb"]["pr_auc_ci95"] = [float(ci[0]), float(ci[1])] #Store CI in results
    results["operating_points_test"] = operating_points(te.y.values, p["hgb"]["test"],
                                                        (va.y.values, p["hgb"]["val"])) #Compute operating points for test set based on HGB model``

    # 4) breakdown by cloud age on test (descriptive, not used to choose anything)
    age = pd.cut(te.run_length_min, [0, 1, 5, 20, 60, np.inf])
    results["test_by_run_length"] = {
        str(k): dict(n=int(len(g)), rate=float(g.y.mean()), #Compute metrics for each run length group
                     pr_auc=float(average_precision_score(g.y, p["hgb"]["test"][te.index.get_indexer(g.index)])) if g.y.nunique() > 1 else None,
                     recall=float(((p["hgb"]["test"][te.index.get_indexer(g.index)] >= thr["hgb"]) & (g.y == 1)).sum() / max(g.y.sum(), 1)))
        for k, g in te.groupby(age, observed=True)}

    # 5) rolling origin (consistency check; Nov-Dec overlap the test fold, so do not tune on it)
    ro = rolling_origin(df, features, pd.Timedelta(horizon)) #Compute rolling origin evaluation for the specified horizon

    # outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = pd.concat([d[["fold", "y", "run_length_min", "lead_min"]].assign(p_hgb=p["hgb"][k], p_logit=p["logit"][k])
                      for k, d in [("val", va), ("test", te)]]) #Concatenate predictions for validation and test sets
    pred.to_parquet(out_dir / "predictions.parquet") #Save concatenated predictions to parquet file
    ro.to_csv(out_dir / "rolling_origin.csv", index=False) #Save rolling origin evaluation to CSV file
    (out_dir / "results.json").write_text(json.dumps(results, indent=2)) #Save results to JSON file
    print(json.dumps({k: results["test"][k] for k in ["hgb", "logit"]}, indent=2))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_DATA.parent / "results")
    ap.add_argument("--horizon", default="20min")
    ap.add_argument("--interval", default="20s")
    args = ap.parse_args()
    run(args.data, args.out_dir, args.horizon, args.interval)


if __name__ == "__main__":
    main()