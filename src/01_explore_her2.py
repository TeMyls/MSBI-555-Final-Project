"""
01_explore_her2.py
==================
Curation waterfall, endpoint characterisation and an EMPIRICAL NOISE-FLOOR
estimate for the HER2 (ERBB2) bioactivity set downloaded from PubChem.

Why this script exists
----------------------
Before fitting any QSAR model you need to know two things:

  1. How many usable measurements survive curation, and what you threw away.
  2. How reproducible the measurements are *between independent assays*.

(2) is the one almost every QSAR report skips. If the same compound, measured
in two different papers, disagrees by 0.5 log units, then no model can ever
achieve an RMSE below ~0.5 log units -- not because the model is bad, but
because the labels themselves carry that much noise. That number is the
ceiling on your R^2, and it is the single most useful piece of context for
interpreting every model result that follows.

This script deliberately uses NO RDKit. Compound identity comes from
PUBCHEM_CID, which is PubChem's own structure-based identifier, so replicate
detection is exact without any cheminformatics toolkit. Structure-dependent
work (fingerprints, scaffolds) happens in 03_features.py.

Inputs
------
  data/raw/combined_pubchem_data.csv
  data/raw/combined_pubchem_data_assay_catalogue.csv

Outputs
-------
  data/processed/her2_measurements.csv    one row per curated measurement
  data/processed/her2_compounds.csv       one row per compound (model input)
  results/tables/*.csv                    numbers quoted in the report
  results/figures/*.png                   figures 1-4 of the report

Usage
-----
  python src/01_explore_her2.py
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "data", "raw")
PROC = os.path.join(HERE, "data", "processed")
FIGS = os.path.join(HERE, "results", "figures")
TABS = os.path.join(HERE, "results", "tables")

for d in (PROC, FIGS, TABS):
    os.makedirs(d, exist_ok=True)

# Endpoints we are willing to pool into a single "potency" column. All four are
# concentrations at which a defined effect occurs, all are log-transformable,
# and all are routinely pooled in ChEMBL-derived QSAR sets. We keep the
# original type in a column so the choice can be revisited (and so we can test
# whether pooling hurt -- see the sensitivity check at the end).
POTENCY_TYPES = ["IC50", "Ki", "Kd", "EC50"]

# Only nanomolar. A handful of rows are ug/mL, which cannot be converted to a
# molar concentration without the molecular weight; rather than guess, we drop
# them and log the loss.
POTENCY_UNITS = "nM"

# Matplotlib: a clean, journal-style default. The instructor requires figures
# be generated with a plotting library rather than an LLM, and requires axis
# labels, units and captions on every one.
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
    }
)

INK = "#1b3a5c"
ACCENT = "#c1543a"
MUTED = "#8a9bab"


def rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# --------------------------------------------------------------------------
# 1. Load
# --------------------------------------------------------------------------

rule("1. LOADING")

raw = pd.read_csv(os.path.join(RAW, "combined_pubchem_data.csv"))
cat = pd.read_csv(os.path.join(RAW, "combined_pubchem_data_assay_catalogue.csv"))

print(f"raw measurement rows        : {len(raw):,}")
print(f"assays in catalogue         : {len(cat):,}")
print(f"assays flagged keep=True    : {int(cat['keep'].sum()):,}")
print(f"distinct AIDs in data file  : {raw['AID'].nunique():,}")

# Sanity check: confirm the data file really does contain only the assays we
# intended to keep. If this ever prints a non-zero number, the download and the
# catalogue have drifted apart and everything downstream is suspect.
keep_aids = set(cat.loc[cat["keep"] == True, "AID"])  # noqa: E712
stray = set(raw["AID"].unique()) - keep_aids
print(f"AIDs present but not kept   : {len(stray)}  (expected 0)")

# --------------------------------------------------------------------------
# 2. Curation waterfall
# --------------------------------------------------------------------------
# The rule we follow at every step: never silently drop a row. Count it, name
# the reason, and carry the count into the figure.

rule("2. CURATION WATERFALL")

waterfall: list[tuple[str, int, int]] = []  # (label, rows, unique compounds)


def record(label: str, frame: pd.DataFrame) -> None:
    waterfall.append((label, len(frame), frame["PUBCHEM_CID"].nunique()))
    print(f"{label:<44} rows={len(frame):>6,}  compounds={frame['PUBCHEM_CID'].nunique():>6,}")


df = raw.copy()
record("0. all downloaded rows", df)

# -- 2.1 structure present -------------------------------------------------
# No SMILES means nothing to featurise. No CID means we cannot establish
# compound identity for replicate detection.
df = df[df["PUBCHEM_EXT_DATASOURCE_SMILES"].notna()]
record("1. has a depositor SMILES", df)

df = df[df["PUBCHEM_CID"].notna()]
df["PUBCHEM_CID"] = df["PUBCHEM_CID"].astype("int64")
record("2. has a PubChem CID", df)

# -- 2.2 the endpoint must be a potency ------------------------------------
# "Inhibition" (a % at a single concentration), "Selectivity" (a ratio),
# "k_on"/"k_off" (kinetics) and "Residual activity" are all real measurements,
# but they are not concentrations and cannot join a pIC50 column.
df = df[df["Standard Type"].isin(POTENCY_TYPES)]
record("3. endpoint is IC50/Ki/Kd/EC50", df)

df = df[df["Standard Units"] == POTENCY_UNITS]
record("4. units are nM", df)

# -- 2.3 the qualifier trap ------------------------------------------------
# A value reported as "> 10000 nM" means the compound was inactive up to the
# highest concentration tested. Its true IC50 is somewhere above 10 uM -- it is
# NOT 10 uM. Treating it as an exact value would compress the inactive tail of
# the distribution and flatter every model. We separate it out, quantify it,
# and exclude it from the regression target.
censored = df[df["Standard Relation"] != "="].copy()
df = df[df["Standard Relation"] == "="]
record("5. exact values only (relation '=')", df)

print(f"\ncensored rows set aside     : {len(censored):,}")
print(censored["Standard Relation"].value_counts().to_string())

# -- 2.4 flagged values ----------------------------------------------------
# PubChem itself flags values it considers implausible for the assay type.
flagged = int(df["Data Validity Comment"].notna().sum())
print(f"\nrows carrying a PubChem validity comment: {flagged}")
print(df.loc[df["Data Validity Comment"].notna(), "Data Validity Comment"].value_counts().to_string())
df = df[df["Data Validity Comment"].isna()]
record("6. no PubChem validity flag", df)

# -- 2.5 physically sensible range -----------------------------------------
# Values at or below zero cannot be log-transformed. We also cap at 1 M: a
# reported IC50 above 1 M is not a measurement, it is an artefact of a
# curve fit that never reached inhibition.
df = df[(df["Standard Value"] > 0) & (df["Standard Value"] <= 1e9)]
record("7. 0 < value <= 1 M", df)

# -- 2.6 the target variable ----------------------------------------------
# pActivity = -log10(molar concentration). nM -> M is a factor of 1e-9.
df = df.copy()
df["pActivity"] = -np.log10(df["Standard Value"] * 1e-9)

print("\npActivity distribution (one row per measurement):")
print(df["pActivity"].describe().to_string())

wf = pd.DataFrame(waterfall, columns=["step", "rows", "compounds"])
wf.to_csv(os.path.join(TABS, "table1_curation_waterfall.csv"), index=False)

# --------------------------------------------------------------------------
# 3. What kind of measurements are these?
# --------------------------------------------------------------------------

rule("3. ENDPOINT COMPOSITION")

comp = (
    df.groupby("Standard Type")
    .agg(
        measurements=("pActivity", "size"),
        compounds=("PUBCHEM_CID", "nunique"),
        median_pActivity=("pActivity", "median"),
        sd=("pActivity", "std"),
    )
    .sort_values("measurements", ascending=False)
)
print(comp.round(2).to_string())
comp.round(3).to_csv(os.path.join(TABS, "table2_endpoint_composition.csv"))

# How concentrated is the data in a few large assays? If 80% of measurements
# come from 10 papers, the "dataset" is really ten congeneric series and the
# scaffold split matters enormously.
per_assay = df.groupby("AID").size().sort_values(ascending=False)
share_top10 = per_assay.head(10).sum() / len(df)
print(f"\nassays contributing data      : {len(per_assay)}")
print(f"median measurements per assay : {per_assay.median():.0f}")
print(f"largest assay                 : {per_assay.max()} measurements (AID {per_assay.idxmax()})")
print(f"share of data in top 10 assays: {share_top10:.1%}")

# --------------------------------------------------------------------------
# 4. THE NOISE FLOOR
# --------------------------------------------------------------------------
# This is the centrepiece of the script.
#
# Some compounds were measured more than once, in assays deposited by
# different groups under different conditions. Those repeats are independent
# estimates of the same underlying quantity. Their disagreement is measurement
# noise -- irreducible by any model.
#
# We estimate the standard deviation of a SINGLE measurement using the pooled
# within-compound variance:
#
#     s_rep^2 = sum_g sum_i (x_gi - xbar_g)^2 / sum_g (n_g - 1)
#
# where g indexes compounds with n_g >= 2 replicates. This is the standard
# one-way ANOVA within-group ("error") mean square, and it is unbiased for the
# replicate variance provided the repeats are genuinely independent.
#
# The ceiling on model performance then follows. A model predicting the TRUE
# value perfectly would still be scored against a noisy observation, so
#
#     R^2_max = 1 - s_rep^2 / s_total^2
#
# where s_total^2 is the variance of the observed compound-level values.

rule("4. EMPIRICAL NOISE FLOOR (replicate analysis)")

# Only count repeats that come from DIFFERENT assays. Two rows from the same
# AID may be the same experiment reported twice, which would understate noise.
rep = (
    df.groupby("PUBCHEM_CID")
    .agg(
        n_measurements=("pActivity", "size"),
        n_assays=("AID", "nunique"),
        mean_pActivity=("pActivity", "mean"),
        sd_pActivity=("pActivity", "std"),
        min_pActivity=("pActivity", "min"),
        max_pActivity=("pActivity", "max"),
    )
    .reset_index()
)
rep["range_pActivity"] = rep["max_pActivity"] - rep["min_pActivity"]

multi = rep[rep["n_assays"] >= 2].copy()

print(f"compounds with >=1 measurement       : {len(rep):,}")
print(f"compounds measured in >=2 assays     : {len(multi):,}")
print(f"compounds measured in >=3 assays     : {(rep['n_assays'] >= 3).sum():,}")

if len(multi) >= 10:
    # Pooled within-compound variance, restricted to cross-assay repeats.
    sub = df[df["PUBCHEM_CID"].isin(multi["PUBCHEM_CID"])]
    grp = sub.groupby("PUBCHEM_CID")["pActivity"]
    ss_within = float(((sub["pActivity"] - grp.transform("mean")) ** 2).sum())
    dof = int(grp.size().sum() - grp.ngroups)
    s_rep = float(np.sqrt(ss_within / dof))

    # Total variance is computed on compound-level means over ALL compounds,
    # because that is what the model will actually be scored against.
    compound_mean = rep["mean_pActivity"]
    s_total = float(compound_mean.std(ddof=1))

    r2_max = 1.0 - (s_rep**2) / (s_total**2)

    # Median absolute difference between two independent measurements of the
    # same compound -- a robust, assumption-free companion to s_rep.
    pair_abs = multi["range_pActivity"]

    print(f"\npooled within-compound SD (s_rep)    : {s_rep:.3f} log units  (dof={dof})")
    print(f"compound-level SD (s_total)          : {s_total:.3f} log units")
    print(f"median cross-assay range             : {pair_abs.median():.3f} log units")
    print(f"90th pct cross-assay range           : {pair_abs.quantile(0.90):.3f} log units")
    print(f"compounds disagreeing by >1 log unit : {(pair_abs > 1).sum()} / {len(multi)}"
          f"  ({(pair_abs > 1).mean():.1%})")
    print(f"\n==> CEILING ON MODEL PERFORMANCE")
    print(f"    best achievable R^2  ~= {r2_max:.3f}")
    print(f"    best achievable RMSE ~= {s_rep:.3f} log units")
    print("\n    Any model reporting R^2 above this on a random split is")
    print("    almost certainly leaking analogs between train and test.")

    noise = pd.DataFrame(
        {
            "quantity": [
                "compounds with cross-assay repeats",
                "pooled within-compound SD (log units)",
                "compound-level SD (log units)",
                "median cross-assay range (log units)",
                "90th percentile cross-assay range",
                "fraction disagreeing by >1 log unit",
                "ceiling R^2",
                "ceiling RMSE (log units)",
            ],
            "value": [
                len(multi),
                round(s_rep, 4),
                round(s_total, 4),
                round(float(pair_abs.median()), 4),
                round(float(pair_abs.quantile(0.90)), 4),
                round(float((pair_abs > 1).mean()), 4),
                round(r2_max, 4),
                round(s_rep, 4),
            ],
        }
    )
    noise.to_csv(os.path.join(TABS, "table3_noise_floor.csv"), index=False)
else:
    s_rep = s_total = r2_max = float("nan")
    print("Too few cross-assay repeats for a stable estimate.")

# --------------------------------------------------------------------------
# 5. Compound-level table (the actual model input)
# --------------------------------------------------------------------------
# One row per compound. Where a compound has several measurements we take the
# MEDIAN, which is robust to a single badly-fitted curve, and we keep the
# spread so that downstream code can optionally down-weight or exclude
# compounds whose measurements disagree wildly.

rule("5. COMPOUND-LEVEL TABLE")

# Pick one representative SMILES per CID (they should be identical or trivial
# variants, since the CID is structure-derived; we take the most common).
smiles_per_cid = (
    df.groupby("PUBCHEM_CID")["PUBCHEM_EXT_DATASOURCE_SMILES"]
    .agg(lambda s: s.value_counts().index[0])
    .rename("smiles")
)

compounds = (
    df.groupby("PUBCHEM_CID")
    .agg(
        pActivity=("pActivity", "median"),
        pActivity_mean=("pActivity", "mean"),
        pActivity_sd=("pActivity", "std"),
        n_measurements=("pActivity", "size"),
        n_assays=("AID", "nunique"),
        endpoint_types=("Standard Type", lambda s: "|".join(sorted(set(s)))),
    )
    .join(smiles_per_cid)
    .reset_index()
)

print(f"compounds in model-ready table : {len(compounds):,}")
print(f"  measured once                : {(compounds['n_measurements'] == 1).sum():,}")
print(f"  measured 2+ times            : {(compounds['n_measurements'] >= 2).sum():,}")
print(f"\npActivity range                : "
      f"{compounds['pActivity'].min():.2f} to {compounds['pActivity'].max():.2f}")
print(f"pActivity SD                   : {compounds['pActivity'].std():.3f}")

df.to_csv(os.path.join(PROC, "her2_measurements.csv"), index=False)
compounds.to_csv(os.path.join(PROC, "her2_compounds.csv"), index=False)
censored.to_csv(os.path.join(PROC, "her2_censored.csv"), index=False)
print(f"\nwrote data/processed/her2_measurements.csv  ({len(df):,} rows)")
print(f"wrote data/processed/her2_compounds.csv     ({len(compounds):,} rows)")
print(f"wrote data/processed/her2_censored.csv      ({len(censored):,} rows)")

# --------------------------------------------------------------------------
# 6. Figures
# --------------------------------------------------------------------------

rule("6. FIGURES")

# --- Figure 1: curation waterfall ----------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.2))
labels = [s for s, _, _ in waterfall]
rows = [r for _, r, _ in waterfall]
ypos = np.arange(len(labels))[::-1]
ax.barh(ypos, rows, color=INK, height=0.62)
for y, r in zip(ypos, rows):
    ax.text(r + max(rows) * 0.012, y, f"{r:,}", va="center", fontsize=8, color=INK)
ax.set_yticks(ypos)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Measurements remaining (rows)")
ax.set_xlim(0, max(rows) * 1.13)
ax.set_title("Figure 1. Curation waterfall, HER2 (ERBB2) PubChem set", loc="left")
ax.grid(axis="y", visible=False)
fig.savefig(os.path.join(FIGS, "fig1_curation_waterfall.png"))
plt.close(fig)
print("fig1_curation_waterfall.png")

# --- Figure 2: potency distribution + censored values ---------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))

ax1.hist(df["pActivity"], bins=40, color=INK, alpha=0.85, edgecolor="white", linewidth=0.4)
ax1.axvline(df["pActivity"].median(), color=ACCENT, lw=1.4, ls="--",
            label=f"median {df['pActivity'].median():.2f}")
ax1.set_xlabel("pActivity  [-log$_{10}$(M)]")
ax1.set_ylabel("Measurements")
ax1.set_title("(a) Exact potency values", loc="left")
ax1.legend(fontsize=8)

# Censored values, shown on the same axis so the reader can see which part of
# the range we are discarding. Almost all ">" values sit in the weak tail.
if len(censored):
    cv = censored.copy()
    cv = cv[(cv["Standard Value"] > 0) & (cv["Standard Units"] == "nM")]
    cv["pActivity"] = -np.log10(cv["Standard Value"] * 1e-9)
    gt = cv[cv["Standard Relation"].isin([">", ">="])]["pActivity"]
    lt = cv[cv["Standard Relation"].isin(["<", "<="])]["pActivity"]
    ax2.hist([gt, lt], bins=28, stacked=True, color=[MUTED, ACCENT],
             edgecolor="white", linewidth=0.4,
             label=[f"'>' n={len(gt)} (inactive)", f"'<' n={len(lt)} (ultra-potent)"])
    ax2.legend(fontsize=7.5)
ax2.set_xlabel("Reported bound  [-log$_{10}$(M)]")
ax2.set_ylabel("Measurements")
ax2.set_title("(b) Censored values excluded", loc="left")

fig.suptitle("Figure 2. Potency distribution and the qualifier trap", x=0.02, ha="left", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(FIGS, "fig2_potency_distribution.png"))
plt.close(fig)
print("fig2_potency_distribution.png")

# --- Figure 3: the noise floor -------------------------------------------
if len(multi) >= 10:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # (a) min vs max across independent assays. Points far from the diagonal
    #     are compounds whose two measurements disagree.
    ax1.scatter(multi["min_pActivity"], multi["max_pActivity"], s=14,
                color=INK, alpha=0.55, edgecolors="none")
    lo = float(min(multi["min_pActivity"].min(), multi["max_pActivity"].min())) - 0.3
    hi = float(max(multi["min_pActivity"].max(), multi["max_pActivity"].max())) + 0.3
    ax1.plot([lo, hi], [lo, hi], color=ACCENT, lw=1.0, ls="--", label="perfect agreement")
    ax1.plot([lo, hi], [lo + 1, hi + 1], color=MUTED, lw=0.9, ls=":", label="1 log unit apart")
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_xlabel("Lowest reported pActivity")
    ax1.set_ylabel("Highest reported pActivity")
    ax1.set_title(f"(a) Cross-assay agreement (n={len(multi)})", loc="left")
    ax1.legend(fontsize=7.5, loc="lower right")
    ax1.set_aspect("equal", adjustable="box")

    # (b) distribution of disagreement, with the derived performance ceiling.
    ax2.hist(multi["range_pActivity"], bins=30, color=INK, alpha=0.85,
             edgecolor="white", linewidth=0.4)
    ax2.axvline(s_rep, color=ACCENT, lw=1.4, ls="--",
                label=f"$s_{{rep}}$ = {s_rep:.2f}")
    ax2.set_xlabel("Spread between independent measurements [log units]")
    ax2.set_ylabel("Compounds")
    ax2.set_title("(b) Measurement noise", loc="left")
    ax2.legend(fontsize=8)
    ax2.text(0.97, 0.62, f"ceiling $R^2 \\approx$ {r2_max:.2f}",
             transform=ax2.transAxes, ha="right", fontsize=8.5, color=ACCENT)

    fig.suptitle("Figure 3. The experimental noise floor sets the ceiling on any model",
                 x=0.02, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(FIGS, "fig3_noise_floor.png"))
    plt.close(fig)
    print("fig3_noise_floor.png")

# --- Figure 4: dataset structure -----------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))

counts = compounds["n_measurements"].value_counts().sort_index()
counts = counts[counts.index <= 8]
ax1.bar(counts.index.astype(str), counts.values, color=INK, width=0.68)
ax1.set_xlabel("Measurements per compound")
ax1.set_ylabel("Compounds")
ax1.set_title("(a) Replicate structure", loc="left")

ranked = per_assay.values
ax2.plot(np.arange(1, len(ranked) + 1), np.cumsum(ranked) / ranked.sum(),
         color=INK, lw=1.6)
ax2.axhline(0.5, color=MUTED, lw=0.9, ls=":")
ax2.set_xlabel("Assays, ranked by size")
ax2.set_ylabel("Cumulative share of measurements")
ax2.set_ylim(0, 1.02)
ax2.set_title("(b) Data concentration across assays", loc="left")

fig.suptitle("Figure 4. Structure of the assembled HER2 set", x=0.02, ha="left", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(os.path.join(FIGS, "fig4_dataset_structure.png"))
plt.close(fig)
print("fig4_dataset_structure.png")

rule("DONE")
print("Tables -> results/tables/   Figures -> results/figures/")
