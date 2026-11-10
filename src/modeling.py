"""
modeling.py
===========
The shared modelling machinery: scaffold-grouped splits, the model zoo, and
the evaluation loop. Imported by 05_models.py and 06_selectivity.py so that
Models A, B and C are judged by *identical* code -- which is the only way the
comparison between them means anything.

Key design decisions, all quoted in the report's methods section:

  * Splitting is GROUPED BY BEMIS-MURCKO SCAFFOLD, never random. A random
    split leaks close analogs across the fence and inflates every metric.
  * Every metric is reported as mean +/- SD over N_SPLITS independent
    scaffold splits, because a single split of ~1000 compounds has a
    standard error large enough to reverse model rankings.
  * Every model is a Pipeline ending in the estimator, with imputation and
    scaling fitted INSIDE the training fold. Fitting a scaler on all the data
    before splitting leaks test-set information.
  * A DummyRegressor (predict the training mean) is always included. A model
    that cannot beat it has learned nothing, and on scaffold splits that
    happens more often than the literature admits.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from common import SEED, N_SPLITS, TEST_SIZE


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------


def scaffold_splits(groups, n_splits: int = N_SPLITS,
                    test_size: float = TEST_SIZE, seed: int = SEED):
    """Return a list of (train_idx, test_idx) with no scaffold on both sides.

    GroupShuffleSplit holds out whole GROUPS, so `test_size` is the fraction
    of scaffolds, not of compounds. The realised compound fraction drifts a
    little from TEST_SIZE because scaffolds differ in size; the evaluation
    loop prints the realised number so it can be reported honestly.
    """
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size,
                            random_state=seed)
    dummy = np.zeros(len(groups))
    return list(gss.split(dummy, groups=groups))


def random_splits(n: int, n_splits: int = N_SPLITS,
                  test_size: float = TEST_SIZE, seed: int = SEED):
    """Random splits, used ONLY as the deliberately-optimistic comparison."""
    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    return list(ss.split(np.zeros(n)))


# --------------------------------------------------------------------------
# Model zoo
# --------------------------------------------------------------------------


def regressors(seed: int = SEED) -> dict[str, Pipeline]:
    """Five regressors plus a dummy, each wrapped in the same preprocessing.

    Hyper-parameters are the scikit-learn defaults except where noted. They
    are NOT tuned per feature set: tuning inside this loop without a nested
    cross-validation would leak the test fold, and a nested CV over six models
    and three feature sets is beyond the compute budget of a class project.
    This is stated as a limitation in the report rather than hidden.
    """
    pre = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    return {
        "Dummy (mean)": Pipeline(pre + [("m", DummyRegressor(strategy="mean"))]),
        "Ridge": Pipeline(pre + [("m", Ridge(alpha=1.0, random_state=None))]),
        "k-NN (k=5)": Pipeline(pre + [("m", KNeighborsRegressor(n_neighbors=5))]),
        "SVR (RBF)": Pipeline(pre + [("m", SVR(C=10.0, gamma="scale",
                                               epsilon=0.1))]),
        "Random forest": Pipeline(pre + [("m", RandomForestRegressor(
            n_estimators=500, min_samples_leaf=1, n_jobs=-1,
            random_state=seed))]),
        "Gradient boosting": Pipeline(pre + [("m", HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.06, random_state=seed))]),
    }


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    resid = y_true - y_pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))

    # R^2 against the TEST-set mean. On a scaffold split this is routinely
    # negative, which is informative: it means the model does worse than
    # predicting the average, i.e. it has not generalised to new chemistry.
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Spearman is undefined when either vector is constant, which is exactly
    # what the dummy regressor produces. Return NaN quietly rather than
    # letting scipy warn on every fold.
    rho = float("nan")
    if len(y_true) > 2 and np.std(y_pred) > 0 and np.std(y_true) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho = float(spearmanr(y_true, y_pred).statistic)

    return {
        "R2": r2,
        "RMSE": float(np.sqrt(np.mean(resid**2))),
        "MAE": float(np.mean(np.abs(resid))),
        "Spearman": rho,
    }


def evaluate(X: np.ndarray, y: np.ndarray, splits,
             models: dict[str, Pipeline] | None = None,
             label: str = "") -> tuple[pd.DataFrame, dict]:
    """Fit every model on every split; return a summary table and predictions.

    The returned `oof` dict maps model name -> (y_true, y_pred) pooled across
    all test folds, which is what the observed-vs-predicted figure plots.
    Because the splits are independent shuffles rather than a partition, a
    compound can appear in more than one test fold; the pooled scatter is
    therefore a visualisation, not a basis for computing a headline metric.
    All headline numbers come from the per-split table.
    """
    models = models or regressors()
    rows, oof = [], {}

    for name, model in models.items():
        per_split, trues, preds = [], [], []
        for train_idx, test_idx in splits:
            m = model
            try:
                from sklearn.base import clone
                m = clone(model)
            except Exception:
                pass
            m.fit(X[train_idx], y[train_idx])
            p = m.predict(X[test_idx])
            per_split.append(_metrics(y[test_idx], p))
            trues.append(y[test_idx])
            preds.append(p)

        agg = {"model": name}
        for key in ("R2", "RMSE", "MAE", "Spearman"):
            vals = np.array([s[key] for s in per_split], dtype=float)
            if np.all(np.isnan(vals)):        # e.g. Spearman for the dummy
                agg[f"{key}_mean"] = float("nan")
                agg[f"{key}_sd"] = float("nan")
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                agg[f"{key}_mean"] = float(np.nanmean(vals))
                agg[f"{key}_sd"] = (float(np.nanstd(vals, ddof=1))
                                    if np.sum(~np.isnan(vals)) > 1 else 0.0)
        rows.append(agg)
        oof[name] = (np.concatenate(trues), np.concatenate(preds))

        if label:
            print(f"    {name:<20} R2 = {agg['R2_mean']:+.3f} +/- {agg['R2_sd']:.3f}"
                  f"   RMSE = {agg['RMSE_mean']:.3f}")

    return pd.DataFrame(rows), oof


def y_scramble(X: np.ndarray, y: np.ndarray, splits, model: Pipeline,
               n_rounds: int = 20, seed: int = SEED) -> dict[str, float]:
    """Refit on randomly permuted labels. R^2 should collapse to ~0.

    If a scrambled model still scores well, the pipeline is leaking -- almost
    always because preprocessing saw the test fold, or because duplicate
    structures survived curation. This is the cheapest real check in QSAR and
    is almost never reported.
    """
    rng = np.random.default_rng(seed)
    scores = []
    from sklearn.base import clone

    for _ in range(n_rounds):
        y_perm = rng.permutation(y)
        for train_idx, test_idx in splits[:1]:        # one split per round
            m = clone(model)
            m.fit(X[train_idx], y_perm[train_idx])
            scores.append(_metrics(y_perm[test_idx], m.predict(X[test_idx]))["R2"])

    arr = np.array(scores, dtype=float)
    return {
        "mean_R2": float(np.nanmean(arr)),
        "sd_R2": float(np.nanstd(arr, ddof=1)),
        "max_R2": float(np.nanmax(arr)),
        "n_rounds": int(n_rounds),
    }


def best_row(table: pd.DataFrame) -> pd.Series:
    """Highest mean R^2 among the real (non-dummy) models."""
    real = table[~table["model"].str.startswith("Dummy")]
    return real.loc[real["R2_mean"].idxmax()]
