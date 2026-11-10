"""
05_models.py
============
Models A and B: predict absolute potency from structure.

    Model A :  pHER2  <- features
    Model B :  pEGFR  <- features

Six regressors x three feature sets x five scaffold splits, for each target.
Plus the three sanity checks that separate an honest QSAR report from an
optimistic one:

    1. Dummy baseline      -- can the model beat predicting the mean?
    2. Random-split contrast-- how much of the headline number is analog leakage?
    3. y-scrambling        -- does the pipeline score well on noise?

Inputs
------
  data/processed/features_her2.npz, her2_with_scaffold.csv
  data/processed/features_egfr.npz, egfr_with_scaffold.csv   (optional)

Outputs
-------
  results/tables/table5_model_a_her2.csv
  results/tables/table6_model_b_egfr.csv
  results/tables/table7_sanity_checks.csv
  results/figures/fig7_model_comparison.png
  results/figures/fig8_observed_vs_predicted.png
  data/processed/model_a_best.json   (feature set + model name, used by 06)

Usage
-----
  python src/05_models.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import (PROC, FIGS, TABS, N_SPLITS, MIN_COMPOUNDS,
                    INK, ACCENT, GREEN, MUTED, SERIES, rule, ic50_um)
from modeling import (scaffold_splits, random_splits, regressors, evaluate,
                      y_scramble, best_row)


def load(name: str):
    """Return (table, {feature_set_name: ndarray}) or (None, None)."""
    npz = os.path.join(PROC, f"features_{name}.npz")
    csv = os.path.join(PROC, f"{name}_with_scaffold.csv")
    if not (os.path.exists(npz) and os.path.exists(csv)):
        return None, None
    df = pd.read_csv(csv)
    store = np.load(npz, allow_pickle=True)
    names = [str(n) for n in store["feature_names"]]
    feats = {n: store[n] for n in names}
    return df, feats


def run_target(name: str, target_col: str, title: str,
               table_path: str) -> tuple[pd.DataFrame, dict] | tuple[None, None]:
    df, feats = load(name)
    if df is None:
        print(f"\nskipping {title}: run 04_features.py first")
        return None, None
    if target_col not in df.columns:
        print(f"\nskipping {title}: no column {target_col}")
        return None, None

    rule(title)

    keep = df[target_col].notna().to_numpy()
    df = df[keep].reset_index(drop=True)
    y = df[target_col].to_numpy(dtype=float)
    groups = df["scaffold"].to_numpy()

    print(f"compounds        : {len(df):,}")
    print(f"unique scaffolds : {df['scaffold'].nunique():,}")
    print(f"{target_col} range : {y.min():.2f} to {y.max():.2f}  (SD {y.std(ddof=1):.3f})")

    if len(df) < MIN_COMPOUNDS:
        print(f"\nWARNING: fewer than {MIN_COMPOUNDS} compounds; results will be unstable.")

    splits = scaffold_splits(groups)
    sizes = [len(te) for _, te in splits]
    print(f"\n{N_SPLITS} scaffold splits, test sets of "
          f"{np.mean(sizes):.0f} +/- {np.std(sizes):.0f} compounds "
          f"({np.mean(sizes)/len(df):.0%} of the data)")

    all_tables, all_oof = [], {}
    for fname, X in feats.items():
        X = np.asarray(X, dtype=float)[keep]
        print(f"\n  --- {fname}  ({X.shape[1]} features) ---")
        tab, oof = evaluate(X, y, splits, label=fname)
        tab.insert(0, "features", fname)
        all_tables.append(tab)
        for mname, pair in oof.items():
            all_oof[(fname, mname)] = pair

    table = pd.concat(all_tables, ignore_index=True)
    table.insert(0, "target", target_col)
    table.round(4).to_csv(table_path, index=False)
    print(f"\nwrote {os.path.relpath(table_path, os.path.dirname(os.path.dirname(table_path)))}")

    bst = best_row(table)
    print(f"\nBEST: {bst['model']} on {bst['features']}")
    print(f"      R2   = {bst['R2_mean']:+.3f} +/- {bst['R2_sd']:.3f}")
    print(f"      RMSE = {bst['RMSE_mean']:.3f} +/- {bst['RMSE_sd']:.3f} log units")
    print(f"      that is a factor of {10**bst['RMSE_mean']:.1f} in IC50")

    return table, {"best": bst, "oof": all_oof, "df": df, "feats": feats,
                   "keep": keep, "y": y, "groups": groups, "splits": splits}


def sanity_checks(ctx: dict, label: str) -> list[dict]:
    """Random-split contrast and y-scrambling for the best (model, features)."""
    rule(f"SANITY CHECKS -- {label}")
    bst = ctx["best"]
    X = np.asarray(ctx["feats"][bst["features"]], dtype=float)[ctx["keep"]]
    y, groups = ctx["y"], ctx["groups"]
    model = regressors()[bst["model"]]

    # -- 1. random split, the deliberately optimistic number ---------------
    rnd = random_splits(len(y))
    rnd_tab, _ = evaluate(X, y, rnd, models={bst["model"]: model})
    r_rand = float(rnd_tab["R2_mean"].iloc[0])
    r_scaf = float(bst["R2_mean"])
    print(f"random split  R2 = {r_rand:+.3f}")
    print(f"scaffold split R2 = {r_scaf:+.3f}")
    print(f"optimism from analog leakage = {r_rand - r_scaf:+.3f} R2 units")

    # -- 2. y-scrambling ---------------------------------------------------
    scr = y_scramble(X, y, ctx["splits"], model, n_rounds=20)
    print(f"\ny-scrambled R2 over {scr['n_rounds']} rounds: "
          f"{scr['mean_R2']:+.3f} +/- {scr['sd_R2']:.3f}  (max {scr['max_R2']:+.3f})")
    verdict = "PASS" if scr["max_R2"] < max(0.15, r_scaf * 0.5) else "FAIL -- investigate leakage"
    print(f"verdict: {verdict}")

    return [
        {"target": label, "check": "scaffold-split R2", "value": round(r_scaf, 4)},
        {"target": label, "check": "random-split R2", "value": round(r_rand, 4)},
        {"target": label, "check": "optimism (random - scaffold)",
         "value": round(r_rand - r_scaf, 4)},
        {"target": label, "check": "y-scrambled mean R2", "value": round(scr["mean_R2"], 4)},
        {"target": label, "check": "y-scrambled max R2", "value": round(scr["max_R2"], 4)},
        {"target": label, "check": "y-scramble verdict", "value": verdict},
    ]


def figure_model_comparison(tables: dict[str, pd.DataFrame]) -> None:
    """Grouped bars: mean R^2 per model per feature set, with SD error bars."""
    panels = [(k, v) for k, v in tables.items() if v is not None]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.4),
                             squeeze=False)
    for ax, (label, tab) in zip(axes[0], panels):
        fsets = list(dict.fromkeys(tab["features"]))
        models = list(dict.fromkeys(tab["model"]))
        width = 0.8 / len(fsets)
        xs = np.arange(len(models))
        for i, fs in enumerate(fsets):
            sub = tab[tab["features"] == fs].set_index("model").reindex(models)
            ax.bar(xs + i * width - 0.4 + width / 2, sub["R2_mean"], width,
                   yerr=sub["R2_sd"], capsize=2, label=fs.split(" (")[0],
                   color=SERIES[i % len(SERIES)],
                   error_kw={"elinewidth": 0.8, "capthick": 0.8})
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels([m.split(" (")[0] for m in models], rotation=38,
                           ha="right", fontsize=7.5)
        ax.set_ylabel("$R^2$ (scaffold split, mean $\\pm$ SD)")
        ax.set_title(label, loc="left")
        ax.legend(fontsize=7)
    fig.suptitle("Figure 7. Model and descriptor comparison", x=0.02, ha="left",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(FIGS, "fig7_model_comparison.png"))
    plt.close(fig)
    print("wrote results/figures/fig7_model_comparison.png")


def figure_obs_vs_pred(ctxs: dict[str, dict], ceiling: float | None = None) -> None:
    panels = [(k, v) for k, v in ctxs.items() if v is not None]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.7 * len(panels), 3.5),
                             squeeze=False)
    for ax, (label, ctx) in zip(axes[0], panels):
        bst = ctx["best"]
        yt, yp = ctx["oof"][(bst["features"], bst["model"])]
        ax.scatter(yt, yp, s=12, color=INK, alpha=0.4, edgecolors="none")
        lo = float(min(yt.min(), yp.min())) - 0.3
        hi = float(max(yt.max(), yp.max())) + 0.3
        ax.plot([lo, hi], [lo, hi], color=ACCENT, lw=1.1, ls="--")
        ax.plot([lo, hi], [lo + 1, hi + 1], color=MUTED, lw=0.8, ls=":")
        ax.plot([lo, hi], [lo - 1, hi - 1], color=MUTED, lw=0.8, ls=":")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Observed  [-log$_{10}$(M)]")
        ax.set_ylabel("Predicted  [-log$_{10}$(M)]")
        txt = (f"{bst['model']} / {bst['features'].split(' (')[0]}\n"
               f"$R^2$ = {bst['R2_mean']:+.2f} $\\pm$ {bst['R2_sd']:.2f}\n"
               f"RMSE = {bst['RMSE_mean']:.2f}")
        if ceiling is not None:
            txt += f"\nceiling $R^2$ $\\approx$ {ceiling:.2f}"
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", fontsize=7.5)
        ax.set_title(label, loc="left")
    fig.suptitle("Figure 8. Observed vs predicted, pooled held-out scaffolds",
                 x=0.02, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(FIGS, "fig8_observed_vs_predicted.png"))
    plt.close(fig)
    print("wrote results/figures/fig8_observed_vs_predicted.png")


def main() -> None:
    tab_a, ctx_a = run_target("her2", "pHER2", "MODEL A -- HER2 potency",
                              os.path.join(TABS, "table5_model_a_her2.csv"))
    tab_b, ctx_b = run_target("egfr", "pEGFR", "MODEL B -- EGFR potency",
                              os.path.join(TABS, "table6_model_b_egfr.csv"))

    checks: list[dict] = []
    if ctx_a:
        checks += sanity_checks(ctx_a, "HER2")
    if ctx_b:
        checks += sanity_checks(ctx_b, "EGFR")
    if checks:
        pd.DataFrame(checks).to_csv(
            os.path.join(TABS, "table7_sanity_checks.csv"), index=False)
        print("\nwrote results/tables/table7_sanity_checks.csv")

    rule("FIGURES")
    figure_model_comparison({"Model A: HER2": tab_a, "Model B: EGFR": tab_b})

    ceiling = None
    nf = os.path.join(TABS, "table3_noise_floor.csv")
    if os.path.exists(nf):
        s = pd.read_csv(nf).set_index("quantity")["value"]
        ceiling = float(s.get("ceiling R^2", float("nan")))
    figure_obs_vs_pred({"Model A: HER2": ctx_a, "Model B: EGFR": ctx_b}, ceiling)

    # Record the winning configuration so 06_selectivity.py uses the same one.
    if ctx_a:
        bst = ctx_a["best"]
        with open(os.path.join(PROC, "model_a_best.json"), "w") as fh:
            json.dump({"features": str(bst["features"]),
                       "model": str(bst["model"]),
                       "R2_mean": float(bst["R2_mean"]),
                       "RMSE_mean": float(bst["RMSE_mean"])}, fh, indent=2)
        print("wrote data/processed/model_a_best.json")

    # A chemist-readable summary of what the error actually means.
    if ctx_a:
        rmse = float(ctx_a["best"]["RMSE_mean"])
        rule("WHAT THE ERROR MEANS")
        print(f"RMSE of {rmse:.2f} log units means a compound whose true IC50 is")
        for p in (6.0, 7.0, 8.0):
            print(f"  {ic50_um(p):>7.3f} uM  is predicted somewhere in "
                  f"{ic50_um(p + rmse):.3f} - {ic50_um(p - rmse):.3f} uM")


if __name__ == "__main__":
    main()
