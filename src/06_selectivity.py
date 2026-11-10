"""
06_selectivity.py
=================
Model C and the central question of the project.

    Model C   :  delta = pHER2 - pEGFR   <-  features     (direct)
    Baseline  :  delta = A(x) - B(x)                      (difference of two
                                                           potency models)

Does a model trained directly on the selectivity difference beat the naive
approach of training two potency models and subtracting them? And how do both
compare against the ceiling imposed by measurement noise?

The noise argument, stated up front
-----------------------------------
delta is a difference of two independent measurements. Variances add:

    s_delta^2 = s_HER2^2 + s_EGFR^2   ->   s_delta ~= sqrt(2) * s_single

01_explore_her2.py measured s_single ~= 0.63 log units from compounds assayed
in multiple independent HER2 studies. So delta carries ~0.89 log units of
noise -- while its DYNAMIC RANGE is typically narrower than that of pHER2,
because most ErbB inhibitors are somewhat active on both targets.

Noise up, signal down. The prediction this script tests is that the ceiling
R^2 for selectivity is far below the ~0.69 available for potency, and that
both Model C and the baseline will land near that lower ceiling. If they do,
the finding is not "our model is bad" -- it is "selectivity prediction is
data-limited, and here is the number that proves it."

LEAKAGE WARNING, handled explicitly below
-----------------------------------------
The baseline needs Models A and B, but they must never see a scaffold that
appears in the selectivity test fold. For each split we therefore retrain A
on all HER2 compounds whose scaffold is NOT in the test fold, and likewise B.
Skipping this step is the single easiest way to produce a beautiful,
meaningless baseline.

Inputs
------
  data/processed/matched_set.csv, features_matched.npz, matched_with_scaffold.csv
  data/processed/her2_with_scaffold.csv, features_her2.npz
  data/processed/egfr_with_scaffold.csv, features_egfr.npz

Outputs
-------
  results/tables/table8_model_c_selectivity.csv
  results/tables/table9_c_vs_baseline.csv
  results/tables/table10_selectivity_drivers.csv
  results/figures/fig9_selectivity_models.png
  results/figures/fig10_applicability_domain.png

Usage
-----
  python src/06_selectivity.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import (PROC, FIGS, TABS, MIN_COMPOUNDS, MORGAN_RADIUS, MORGAN_BITS,
                    SEED, INK, ACCENT, GREEN, MUTED, rule)
from modeling import scaffold_splits, regressors, evaluate, y_scramble, best_row, _metrics

try:
    from rdkit import Chem, RDLogger, DataStructs
    from rdkit.Chem import rdFingerprintGenerator, Draw
    RDLogger.DisableLog("rdApp.*")
    HAVE_RDKIT = True
except ImportError:
    HAVE_RDKIT = False


def load(name: str):
    npz = os.path.join(PROC, f"features_{name}.npz")
    csv = os.path.join(PROC, f"{name}_with_scaffold.csv")
    if not (os.path.exists(npz) and os.path.exists(csv)):
        return None, None
    df = pd.read_csv(csv)
    store = np.load(npz, allow_pickle=True)
    names = [str(n) for n in store["feature_names"]]
    return df, {n: np.asarray(store[n], dtype=float) for n in names}


# --------------------------------------------------------------------------
# Model C
# --------------------------------------------------------------------------


def run_model_c(df: pd.DataFrame, feats: dict, splits) -> tuple[pd.DataFrame, dict]:
    rule("MODEL C -- direct selectivity regression")
    y = df["delta"].to_numpy(dtype=float)

    print(f"compounds        : {len(df):,}")
    print(f"unique scaffolds : {df['scaffold'].nunique():,}")
    print(f"delta range      : {y.min():+.2f} to {y.max():+.2f}  (SD {y.std(ddof=1):.3f})")

    tables, oof = [], {}
    for fname, X in feats.items():
        print(f"\n  --- {fname} ({X.shape[1]} features) ---")
        tab, o = evaluate(X, y, splits, label=fname)
        tab.insert(0, "features", fname)
        tables.append(tab)
        for mname, pair in o.items():
            oof[(fname, mname)] = pair

    table = pd.concat(tables, ignore_index=True)
    table.insert(0, "target", "delta")
    table.round(4).to_csv(os.path.join(TABS, "table8_model_c_selectivity.csv"),
                          index=False)
    print("\nwrote results/tables/table8_model_c_selectivity.csv")

    bst = best_row(table)
    print(f"\nBEST Model C: {bst['model']} on {bst['features']}")
    print(f"  R2   = {bst['R2_mean']:+.3f} +/- {bst['R2_sd']:.3f}")
    print(f"  RMSE = {bst['RMSE_mean']:.3f} log units")
    return table, {"best": bst, "oof": oof, "y": y}


# --------------------------------------------------------------------------
# The leakage-free A - B baseline
# --------------------------------------------------------------------------


def run_baseline(matched: pd.DataFrame, m_feats: dict,
                 her2: pd.DataFrame, h_feats: dict,
                 egfr: pd.DataFrame, e_feats: dict,
                 splits, feature_set: str, model_name: str) -> dict:
    rule("BASELINE -- difference of two independently trained potency models")
    print(f"using {model_name} on {feature_set} for both A and B\n")

    y_true_all, y_pred_all, per_split = [], [], []

    for k, (train_idx, test_idx) in enumerate(splits, start=1):
        test_scaffolds = set(matched["scaffold"].to_numpy()[test_idx])

        # --- Model A: HER2, excluding every scaffold in the test fold -----
        h_mask = ~her2["scaffold"].isin(test_scaffolds).to_numpy()
        h_mask &= her2["pHER2"].notna().to_numpy()
        Xa = h_feats[feature_set][h_mask]
        ya = her2.loc[h_mask, "pHER2"].to_numpy(dtype=float)

        # --- Model B: EGFR, same exclusion --------------------------------
        e_mask = ~egfr["scaffold"].isin(test_scaffolds).to_numpy()
        e_mask &= egfr["pEGFR"].notna().to_numpy()
        Xb = e_feats[feature_set][e_mask]
        yb = egfr.loc[e_mask, "pEGFR"].to_numpy(dtype=float)

        from sklearn.base import clone
        ma = clone(regressors()[model_name]).fit(Xa, ya)
        mb = clone(regressors()[model_name]).fit(Xb, yb)

        Xt = m_feats[feature_set][test_idx]
        pred = ma.predict(Xt) - mb.predict(Xt)
        truth = matched["delta"].to_numpy(dtype=float)[test_idx]

        per_split.append(_metrics(truth, pred))
        y_true_all.append(truth)
        y_pred_all.append(pred)
        print(f"  split {k}: A trained on {h_mask.sum():,}, "
              f"B on {e_mask.sum():,}, tested on {len(test_idx):,}"
              f"   R2 = {per_split[-1]['R2']:+.3f}")

    agg = {}
    for key in ("R2", "RMSE", "MAE", "Spearman"):
        vals = np.array([s[key] for s in per_split], dtype=float)
        agg[f"{key}_mean"] = float(np.nanmean(vals))
        agg[f"{key}_sd"] = float(np.nanstd(vals, ddof=1))

    print(f"\nBaseline  R2 = {agg['R2_mean']:+.3f} +/- {agg['R2_sd']:.3f}"
          f"   RMSE = {agg['RMSE_mean']:.3f}")
    return {"agg": agg, "oof": (np.concatenate(y_true_all), np.concatenate(y_pred_all))}


# --------------------------------------------------------------------------
# Applicability domain
# --------------------------------------------------------------------------


def applicability_domain(df: pd.DataFrame, splits, y: np.ndarray,
                         feats: dict, feature_set: str, model_name: str) -> pd.DataFrame:
    """Relate prediction error to distance from the training set.

    For each test compound we compute the maximum Tanimoto similarity to any
    training compound. A trustworthy QSAR model shows error rising as that
    similarity falls; if it does not, the model is extrapolating blindly and
    its predictions on novel chemistry mean nothing.
    """
    rule("APPLICABILITY DOMAIN")
    if not HAVE_RDKIT:
        print("RDKit unavailable; skipping.")
        return pd.DataFrame()

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=2048)
    smiles = df["canonical_smiles"].tolist()
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None
           for s in smiles]

    from sklearn.base import clone
    X = feats[feature_set]
    rows = []

    for train_idx, test_idx in splits:
        m = clone(regressors()[model_name]).fit(X[train_idx], y[train_idx])
        pred = m.predict(X[test_idx])
        train_fps = [fps[i] for i in train_idx if fps[i] is not None]
        for j, idx in enumerate(test_idx):
            if fps[idx] is None or not train_fps:
                continue
            sim = max(DataStructs.BulkTanimotoSimilarity(fps[idx], train_fps))
            rows.append({"similarity": sim,
                         "abs_error": abs(y[idx] - pred[j]),
                         "observed": y[idx], "predicted": pred[j]})

    ad = pd.DataFrame(rows)
    if ad.empty:
        return ad

    bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.01]
    labels = ["<0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", ">0.7"]
    ad["bin"] = pd.cut(ad["similarity"], bins=bins, labels=labels, right=False)

    summary = (ad.groupby("bin", observed=True)
               .agg(n=("abs_error", "size"),
                    mean_abs_error=("abs_error", "mean"),
                    median_abs_error=("abs_error", "median"))
               .reset_index())
    print(summary.round(3).to_string(index=False))
    print("\nIf mean_abs_error falls monotonically as similarity rises, the")
    print("model knows what it does not know, and a similarity cut-off can be")
    print("used as a confidence filter when screening new compounds.")
    return ad


# --------------------------------------------------------------------------
# Which substructures drive selectivity
# --------------------------------------------------------------------------


def selectivity_drivers(df: pd.DataFrame, feats: dict, y: np.ndarray) -> None:
    """Fit a random forest on the full matched set and read off what it used.

    Two views:
      * descriptor importances -- interpretable physicochemistry
      * Morgan bit importances -- mapped back to the substructure they encode
    """
    rule("WHAT DRIVES SELECTIVITY")
    from sklearn.base import clone

    rows = []

    # --- physicochemical view --------------------------------------------
    lip = [k for k in feats if k.startswith("Lipinski")]
    if lip:
        key = lip[0]
        m = clone(regressors()["Random forest"]).fit(feats[key], y)
        imp = m[-1].feature_importances_
        # Column order is fixed by SIMPLE in 04_features.py; read it back from
        # the sidecar when present so the two files cannot drift apart.
        sidecar = os.path.join(PROC, "features_matched_columns.json")
        if os.path.exists(sidecar):
            names = json.load(open(sidecar))[key]
        else:
            names = ["MolWt", "cLogP", "TPSA", "HBD", "HBA", "RotB",
                     "AromaticRings", "FractionCSP3"][: len(imp)]
        order = np.argsort(imp)[::-1]
        print("\nphysicochemical drivers of HER2/EGFR selectivity:")
        for i in order:
            # Sign of the correlation tells you the direction; importance
            # alone does not.
            corr = float(np.corrcoef(feats[key][:, i], y)[0, 1])
            direction = "-> HER2-selective" if corr > 0 else "-> EGFR-selective"
            print(f"  {names[i]:<15} importance {imp[i]:.3f}   r = {corr:+.2f}  {direction}")
            rows.append({"view": "descriptor", "feature": names[i],
                         "importance": round(float(imp[i]), 4),
                         "correlation_with_delta": round(corr, 4)})

    # --- substructure view ------------------------------------------------
    fp_keys = [k for k in feats if k.startswith("Morgan")]
    if fp_keys and HAVE_RDKIT:
        key = fp_keys[0]
        X = feats[key]
        m = clone(regressors()["Random forest"]).fit(X, y)
        imp = m[-1].feature_importances_
        top_bits = np.argsort(imp)[::-1][:15]

        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
        smiles = pd.read_csv(os.path.join(PROC, "matched_with_scaffold.csv"))[
            "canonical_smiles"].tolist()

        print("\ntop Morgan bits (substructures) and the direction they push:")
        for b in top_bits:
            on = X[:, b] > 0
            if on.sum() < 5:
                continue
            diff = float(y[on].mean() - y[~on].mean())
            direction = "HER2-selective" if diff > 0 else "EGFR-selective"
            # Find a molecule containing this bit, so the substructure can be
            # drawn and shown in the report rather than quoted as a number.
            example = ""
            for s in np.array(smiles)[on][:40]:
                mol = Chem.MolFromSmiles(s)
                if mol is None:
                    continue
                ao = rdFingerprintGenerator.AdditionalOutput()
                ao.AllocateBitInfoMap()
                gen.GetFingerprint(mol, additionalOutput=ao)
                info = ao.GetBitInfoMap()
                if int(b) in info:
                    atom_idx, rad = info[int(b)][0]
                    env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom_idx) \
                        if rad > 0 else []
                    amap: dict = {}
                    sub = Chem.PathToSubmol(mol, env, atomMap=amap) if env else None
                    example = Chem.MolToSmiles(sub) if sub else s
                    break
            print(f"  bit_{b:<5} imp {imp[b]:.4f}  present in {on.sum():>4}  "
                  f"mean delta shift {diff:+.2f}  {direction:<16} {example}")
            rows.append({"view": "morgan_bit", "feature": f"bit_{b}",
                         "importance": round(float(imp[b]), 4),
                         "n_compounds_with_bit": int(on.sum()),
                         "mean_delta_shift": round(diff, 4),
                         "direction": direction,
                         "example_substructure": example})

    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(TABS, "table10_selectivity_drivers.csv"), index=False)
        print("\nwrote results/tables/table10_selectivity_drivers.csv")


# --------------------------------------------------------------------------


def main() -> None:
    matched, m_feats = load("matched")
    if matched is None:
        sys.exit("No matched set. Run 02_fetch_egfr.py, 03_curate.py, 04_features.py first.")
    if "delta" not in matched.columns:
        sys.exit("matched_with_scaffold.csv has no `delta` column.")

    if len(matched) < MIN_COMPOUNDS:
        print("!" * 72)
        print(f"Only {len(matched)} matched compounds. Model C will be unstable;")
        print("report it with that caveat, or use the fallback in the README.")
        print("!" * 72)

    splits = scaffold_splits(matched["scaffold"].to_numpy())
    table_c, ctx_c = run_model_c(matched, m_feats, splits)
    y = ctx_c["y"]

    # Use the configuration that won on HER2, so A, B and C are comparable.
    cfg_path = os.path.join(PROC, "model_a_best.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        feature_set, model_name = cfg["features"], cfg["model"]
    else:
        feature_set = str(ctx_c["best"]["features"])
        model_name = str(ctx_c["best"]["model"])

    her2, h_feats = load("her2")
    egfr, e_feats = load("egfr")

    base = None
    if her2 is not None and egfr is not None and feature_set in h_feats:
        base = run_baseline(matched, m_feats, her2, h_feats, egfr, e_feats,
                            splits, feature_set, model_name)

    # ---- the comparison --------------------------------------------------
    rule("MODEL C vs BASELINE vs CEILING")

    s_single = 0.630
    nf = os.path.join(TABS, "table3_noise_floor.csv")
    if os.path.exists(nf):
        s = pd.read_csv(nf).set_index("quantity")["value"]
        s_single = float(s.get("pooled within-compound SD (log units)", s_single))
    s_delta = float(np.sqrt(2) * s_single)
    sd_delta = float(np.std(y, ddof=1))
    ceiling = 1.0 - (s_delta**2) / (sd_delta**2)

    bst = ctx_c["best"]
    comparison = [
        {"approach": "Model C (direct on delta)",
         "R2_mean": round(float(bst["R2_mean"]), 4),
         "R2_sd": round(float(bst["R2_sd"]), 4),
         "RMSE_mean": round(float(bst["RMSE_mean"]), 4)},
    ]
    if base:
        comparison.append(
            {"approach": "Baseline A(x) - B(x)",
             "R2_mean": round(base["agg"]["R2_mean"], 4),
             "R2_sd": round(base["agg"]["R2_sd"], 4),
             "RMSE_mean": round(base["agg"]["RMSE_mean"], 4)})
    comparison.append(
        {"approach": "Noise ceiling", "R2_mean": round(ceiling, 4),
         "R2_sd": 0.0, "RMSE_mean": round(s_delta, 4)})

    comp = pd.DataFrame(comparison)
    print(comp.to_string(index=False))
    comp.to_csv(os.path.join(TABS, "table9_c_vs_baseline.csv"), index=False)

    print(f"\nSD of delta                : {sd_delta:.3f} log units")
    print(f"propagated measurement noise: {s_delta:.3f} log units")
    print(f"ceiling R^2 for selectivity : {ceiling:.3f}")
    print(f"(for comparison, potency had a ceiling of ~0.69)")

    if base:
        gap = float(bst["R2_mean"]) - base["agg"]["R2_mean"]
        pooled_sd = float(np.hypot(bst["R2_sd"], base["agg"]["R2_sd"]))
        print(f"\nModel C minus baseline: {gap:+.3f} R2 units "
              f"(pooled SD {pooled_sd:.3f})")
        if abs(gap) < pooled_sd:
            print("==> The difference is within noise. A dedicated selectivity")
            print("    model does NOT beat subtracting two potency models here.")
        elif gap > 0:
            print("==> Direct modelling of delta wins: training on the difference")
            print("    lets the model exploit cancellation of shared chemistry.")
        else:
            print("==> Subtracting two potency models wins, most likely because")
            print("    A and B each train on far more compounds than C.")

    # ---- y-scrambling for Model C ---------------------------------------
    rule("SANITY CHECK -- y-scrambling Model C")
    scr = y_scramble(m_feats[str(bst["features"])], y, splits,
                     regressors()[str(bst["model"])], n_rounds=20)
    print(f"scrambled R2: {scr['mean_R2']:+.3f} +/- {scr['sd_R2']:.3f} "
          f"(max {scr['max_R2']:+.3f})")

    # ---- applicability domain -------------------------------------------
    ad = applicability_domain(matched, splits, y, m_feats, feature_set, model_name)

    # ---- drivers ---------------------------------------------------------
    selectivity_drivers(matched, m_feats, y)

    # ---- figures ---------------------------------------------------------
    rule("FIGURES")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4))

    yt, yp = ctx_c["oof"][(bst["features"], bst["model"])]
    ax1.scatter(yt, yp, s=14, color=INK, alpha=0.45, edgecolors="none",
                label="Model C")
    if base:
        bt, bp = base["oof"]
        ax1.scatter(bt, bp, s=14, color=GREEN, alpha=0.35, edgecolors="none",
                    label="A $-$ B baseline")
    lo = float(min(yt.min(), yp.min())) - 0.4
    hi = float(max(yt.max(), yp.max())) + 0.4
    ax1.plot([lo, hi], [lo, hi], color=ACCENT, lw=1.1, ls="--")
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("Observed $\\Delta$  [log units]")
    ax1.set_ylabel("Predicted $\\Delta$  [log units]")
    ax1.set_title("(a) Selectivity predictions", loc="left")
    ax1.legend(fontsize=7.5, loc="upper left")

    names = [c["approach"] for c in comparison]
    vals = [c["R2_mean"] for c in comparison]
    errs = [c["R2_sd"] for c in comparison]
    cols = [INK, GREEN, ACCENT][: len(names)]
    ax2.barh(np.arange(len(names))[::-1], vals, xerr=errs, color=cols,
             height=0.55, capsize=3, error_kw={"elinewidth": 0.9})
    ax2.set_yticks(np.arange(len(names))[::-1])
    ax2.set_yticklabels(names, fontsize=8)
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlabel("$R^2$ (scaffold split)")
    ax2.set_title("(b) Against the noise ceiling", loc="left")
    ax2.grid(axis="y", visible=False)

    fig.suptitle("Figure 9. Can selectivity be predicted?", x=0.02, ha="left",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(FIGS, "fig9_selectivity_models.png"))
    plt.close(fig)
    print("wrote results/figures/fig9_selectivity_models.png")

    if not ad.empty:
        fig, ax = plt.subplots(figsize=(4.8, 3.3))
        grouped = ad.groupby("bin", observed=True)["abs_error"]
        labels = [str(b) for b in grouped.groups]
        data = [grouped.get_group(b).to_numpy() for b in grouped.groups]
        # `labels=` was renamed `tick_labels=` in matplotlib 3.9; set the tick
        # labels afterwards instead, which works on every version.
        bp = ax.boxplot(data, showfliers=False, widths=0.6,
                        patch_artist=True, medianprops={"color": ACCENT, "lw": 1.3})
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=8)
        for patch in bp["boxes"]:
            patch.set_facecolor(INK)
            patch.set_alpha(0.45)
            patch.set_edgecolor(INK)
        counts = [len(d) for d in data]
        for i, n in enumerate(counts, start=1):
            ax.text(i, ax.get_ylim()[1] * 0.96, f"n={n}", ha="center", fontsize=6.5,
                    color=MUTED)
        ax.set_xlabel("Max Tanimoto similarity to any training compound")
        ax.set_ylabel("|error| in $\\Delta$  [log units]")
        ax.set_title("Figure 10. Applicability domain of the selectivity model",
                     loc="left", fontsize=9.5)
        fig.savefig(os.path.join(FIGS, "fig10_applicability_domain.png"))
        plt.close(fig)
        print("wrote results/figures/fig10_applicability_domain.png")


if __name__ == "__main__":
    main()
