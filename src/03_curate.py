"""
03_curate.py
============
Standardise structures for BOTH targets, build the two potency columns, and
assemble the matched set on which the selectivity model is trained.

NEEDS RDKit.

Standardisation (identical for both targets -- this matters, because any
difference in how the two sides are canonicalised would break the join and
silently shrink the matched set):

    largest organic fragment  ->  neutralise charges  ->  canonical SMILES

Then, per target, one row per compound: the MEDIAN pActivity across all
measurements, which is robust to a single badly fitted dose-response curve.

Finally the join. A compound present in both tables contributes:

    delta = pActivity(HER2) - pActivity(EGFR)

Positive delta = HER2-selective. Negative = EGFR-selective. Around zero =
dual inhibitor.

Inputs
------
  data/raw/combined_pubchem_data.csv            (HER2, supplied)
  data/raw/combined_pubchem_data_assay_catalogue.csv
  data/raw/egfr_pubchem_data.csv                (from 02_fetch_egfr.py)

Outputs
-------
  data/processed/her2_curated.csv
  data/processed/egfr_curated.csv
  data/processed/matched_set.csv
  results/tables/table4_matched_set_summary.csv
  results/figures/fig5_selectivity_landscape.png

Usage
-----
  python src/03_curate.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import (RAW, PROC, FIGS, TABS, POTENCY_TYPES, POTENCY_UNITS,
                    MIN_COMPOUNDS, INK, ACCENT, MUTED, GREEN, rule, p_activity)

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")        # suppress per-molecule parse warnings
except ImportError:
    sys.exit("RDKit is required.  pip install rdkit    (or run in Google Colab)")


# --------------------------------------------------------------------------
# Structure standardisation
# --------------------------------------------------------------------------


def largest_fragment(mol):
    """Return the fragment with the most heavy atoms.

    A SMILES containing a '.' is a salt or a mixture, e.g.
    'CC(=O)[O-].[Na+]' is sodium acetate. The assay measured the drug, not the
    counter-ion, so we keep the biggest organic piece. Ties go to the first,
    which is arbitrary but deterministic.
    """
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) == 1:
        return mol
    return max(frags, key=lambda m: m.GetNumHeavyAtoms())


def neutralise(mol):
    """Strip formal charges where chemistry allows.

    The same compound is often deposited once as a free base and once as a
    hydrochloride. Unless we neutralise, those become two different canonical
    SMILES, the replicate is missed, and the compound appears in both the
    training and the test set of a scaffold split.

    The pattern matches any atom with a non-zero formal charge that is not
    part of a charge-separated group that must stay (e.g. a nitro group,
    where N is formally +1 and O is -1 and neutralising would be wrong).
    """
    pattern = Chem.MolFromSmarts(
        "[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]"
    )
    at_matches = mol.GetSubstructMatches(pattern)
    for (idx,) in at_matches:
        atom = mol.GetAtomWithIdx(idx)
        charge = atom.GetFormalCharge()
        hcount = atom.GetTotalNumHs()
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(hcount - charge)
        atom.UpdatePropertyCache(strict=False)
    return mol


def standardise(smiles: str) -> str | None:
    """SMILES -> canonical SMILES of the neutral parent, or None if unusable."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = largest_fragment(mol)
        mol = neutralise(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    # Reject anything with no carbon (inorganic salts leak through otherwise)
    # and anything implausibly large for a kinase inhibitor QSAR set --
    # peptides and antibodies-as-SMILES do appear in PubChem assay tables.
    if not any(a.GetSymbol() == "C" for a in mol.GetAtoms()):
        return None
    if mol.GetNumHeavyAtoms() < 6 or mol.GetNumHeavyAtoms() > 100:
        return None
    if Descriptors.MolWt(mol) > 1200:
        return None

    return Chem.MolToSmiles(mol, isomericSmiles=True)


# --------------------------------------------------------------------------
# Per-target curation
# --------------------------------------------------------------------------


def curate(raw: pd.DataFrame, label: str,
           keep_aids: set | None = None) -> pd.DataFrame:
    """Apply the full curation chain to one target's raw measurement table."""
    rule(f"CURATING {label}")

    df = raw.copy()
    log: list[tuple[str, int]] = [("downloaded rows", len(df))]

    def step(name, frame):
        log.append((name, len(frame)))
        return frame

    if keep_aids is not None:
        df = step("assay passed target triage", df[df["AID"].isin(keep_aids)])

    df = step("has depositor SMILES", df[df["PUBCHEM_EXT_DATASOURCE_SMILES"].notna()])
    df = step("endpoint is a potency", df[df["Standard Type"].isin(POTENCY_TYPES)])
    df = step("units are nM", df[df["Standard Units"] == POTENCY_UNITS])
    df = step("exact value (relation '=')", df[df["Standard Relation"] == "="])
    df = step("no PubChem validity flag", df[df["Data Validity Comment"].isna()])
    df = step("0 < value <= 1 M",
              df[(df["Standard Value"] > 0) & (df["Standard Value"] <= 1e9)])

    # Structure standardisation is the expensive step, so it runs last, on the
    # smallest surviving table.
    df = df.copy()
    df["canonical_smiles"] = df["PUBCHEM_EXT_DATASOURCE_SMILES"].map(standardise)
    df = step("structure parsed and standardised", df[df["canonical_smiles"].notna()])

    df["pActivity"] = p_activity(df["Standard Value"])

    for name, n in log:
        print(f"  {name:<36} {n:>6,}")

    # One row per compound.
    out = (
        df.groupby("canonical_smiles")
        .agg(
            **{
                f"p{label}": ("pActivity", "median"),
                f"p{label}_sd": ("pActivity", "std"),
                f"n_{label}": ("pActivity", "size"),
                f"n_assays_{label}": ("AID", "nunique"),
            }
        )
        .reset_index()
    )
    print(f"\n  -> {len(out):,} unique compounds")
    print(f"     p{label}: {out[f'p{label}'].min():.2f} to {out[f'p{label}'].max():.2f}"
          f"  (median {out[f'p{label}'].median():.2f}, SD {out[f'p{label}'].std():.2f})")
    return out


# --------------------------------------------------------------------------


def main() -> None:
    # ---- HER2 ------------------------------------------------------------
    her2_raw = pd.read_csv(os.path.join(RAW, "combined_pubchem_data.csv"))
    cat = pd.read_csv(os.path.join(RAW, "combined_pubchem_data_assay_catalogue.csv"))
    keep = set(cat.loc[cat["keep"] == True, "AID"])  # noqa: E712
    her2 = curate(her2_raw, "HER2", keep_aids=keep)
    her2.to_csv(os.path.join(PROC, "her2_curated.csv"), index=False)

    # ---- EGFR ------------------------------------------------------------
    egfr_path = os.path.join(RAW, "egfr_pubchem_data.csv")
    if not os.path.exists(egfr_path):
        print("\n" + "!" * 72)
        print("No EGFR data found at data/raw/egfr_pubchem_data.csv")
        print("Run:  python src/02_fetch_egfr.py")
        print("Only the HER2 table has been written; the matched set is skipped.")
        print("!" * 72)
        return

    egfr_raw = pd.read_csv(egfr_path, low_memory=False)
    egfr_cat = pd.read_csv(os.path.join(RAW, "egfr_assay_catalogue.csv"))
    egfr_keep = set(egfr_cat.loc[egfr_cat["keep"] == True, "AID"])  # noqa: E712
    egfr = curate(egfr_raw, "EGFR", keep_aids=egfr_keep)
    egfr.to_csv(os.path.join(PROC, "egfr_curated.csv"), index=False)

    # ---- the matched set -------------------------------------------------
    rule("MATCHED SET")

    matched = her2.merge(egfr, on="canonical_smiles", how="inner")
    matched["delta"] = matched["pHER2"] - matched["pEGFR"]

    print(f"HER2 compounds          : {len(her2):,}")
    print(f"EGFR compounds          : {len(egfr):,}")
    print(f"measured against BOTH   : {len(matched):,}")

    if len(matched) == 0:
        print("\nNo overlap. Check that both catalogues kept the right assays.")
        return

    print(f"\ndelta = pHER2 - pEGFR")
    print(matched["delta"].describe().round(3).to_string())

    her2_sel = (matched["delta"] >= 1).sum()
    egfr_sel = (matched["delta"] <= -1).sum()
    dual = ((matched["delta"] > -1) & (matched["delta"] < 1)).sum()
    print(f"\nHER2-selective (delta >= +1) : {her2_sel:,}  ({her2_sel/len(matched):.1%})")
    print(f"dual (-1 < delta < +1)       : {dual:,}  ({dual/len(matched):.1%})")
    print(f"EGFR-selective (delta <= -1) : {egfr_sel:,}  ({egfr_sel/len(matched):.1%})")

    # ---- the ceiling on the SELECTIVITY model ----------------------------
    # delta is a difference of two independent noisy measurements, so its
    # noise variance is the SUM of theirs:
    #
    #     s_delta^2 = s_HER2^2 + s_EGFR^2
    #
    # With the HER2 replicate SD of ~0.63 log units measured in
    # 01_explore_her2.py, and assuming EGFR is comparable, s_delta ~= 0.89.
    # This is the quantitative prediction the project tests.
    s_her2 = 0.630          # measured, see results/tables/table3_noise_floor.csv
    noise_path = os.path.join(TABS, "table3_noise_floor.csv")
    if os.path.exists(noise_path):
        nf = pd.read_csv(noise_path).set_index("quantity")["value"]
        s_her2 = float(nf.get("pooled within-compound SD (log units)", s_her2))

    s_delta = float(np.sqrt(2) * s_her2)
    sd_delta = float(matched["delta"].std(ddof=1))
    ceiling_delta = 1.0 - (s_delta**2) / (sd_delta**2)

    print(f"\n--- predicted ceiling for the SELECTIVITY model ---")
    print(f"single-target replicate SD   s      = {s_her2:.3f} log units")
    print(f"propagated noise on delta    s*sqrt2= {s_delta:.3f} log units")
    print(f"observed SD of delta                = {sd_delta:.3f} log units")
    print(f"ceiling R^2 for delta               = {ceiling_delta:.3f}")
    print(f"(compare: ceiling R^2 for pHER2     = ~0.69)")
    if ceiling_delta < 0.4:
        print("\n==> Selectivity is intrinsically HARDER to predict than potency:")
        print("    the noise doubles while the signal does not.")

    matched.to_csv(os.path.join(PROC, "matched_set.csv"), index=False)
    print(f"\nwrote data/processed/matched_set.csv ({len(matched):,} compounds)")

    pd.DataFrame(
        {
            "quantity": [
                "HER2 compounds", "EGFR compounds", "matched compounds",
                "HER2-selective (delta>=+1)", "dual (|delta|<1)",
                "EGFR-selective (delta<=-1)",
                "SD of delta", "propagated noise SD on delta",
                "ceiling R^2 for delta",
            ],
            "value": [
                len(her2), len(egfr), len(matched),
                int(her2_sel), int(dual), int(egfr_sel),
                round(sd_delta, 4), round(s_delta, 4), round(ceiling_delta, 4),
            ],
        }
    ).to_csv(os.path.join(TABS, "table4_matched_set_summary.csv"), index=False)

    if len(matched) < MIN_COMPOUNDS:
        print("\n" + "!" * 72)
        print(f"WARNING: only {len(matched)} matched compounds (want >= {MIN_COMPOUNDS}).")
        print("A scaffold-split selectivity model will be unstable. Consider the")
        print("fallback described in the README before reporting Model C.")
        print("!" * 72)

    # ---- Figure 5 --------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    # (a) the selectivity landscape. The diagonal is "equally potent on both".
    colour = np.where(matched["delta"] >= 1, ACCENT,
                      np.where(matched["delta"] <= -1, GREEN, MUTED))
    ax1.scatter(matched["pEGFR"], matched["pHER2"], s=16, c=colour,
                alpha=0.6, edgecolors="none")
    lo = float(min(matched["pEGFR"].min(), matched["pHER2"].min())) - 0.3
    hi = float(max(matched["pEGFR"].max(), matched["pHER2"].max())) + 0.3
    ax1.plot([lo, hi], [lo, hi], color=INK, lw=1.0, ls="--", label="equipotent")
    ax1.plot([lo, hi], [lo + 1, hi + 1], color=INK, lw=0.8, ls=":", alpha=0.6,
             label="10x selective")
    ax1.plot([lo, hi], [lo - 1, hi - 1], color=INK, lw=0.8, ls=":", alpha=0.6)
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("pEGFR  [-log$_{10}$(M)]")
    ax1.set_ylabel("pHER2  [-log$_{10}$(M)]")
    ax1.set_title(f"(a) Selectivity landscape (n={len(matched)})", loc="left")
    ax1.legend(fontsize=7.5, loc="lower right")

    # (b) distribution of delta, with the propagated noise width for scale.
    ax2.hist(matched["delta"], bins=34, color=INK, alpha=0.85,
             edgecolor="white", linewidth=0.4)
    ax2.axvline(0, color=MUTED, lw=1.0, ls="-")
    ax2.axvspan(-s_delta, s_delta, color=ACCENT, alpha=0.14,
                label=f"$\\pm s_\\Delta$ = {s_delta:.2f} (noise)")
    ax2.set_xlabel("$\\Delta$ = pHER2 $-$ pEGFR  [log units]")
    ax2.set_ylabel("Compounds")
    ax2.set_title("(b) Selectivity distribution vs noise", loc="left")
    ax2.legend(fontsize=8)

    fig.suptitle("Figure 5. HER2 / EGFR selectivity in the matched set",
                 x=0.02, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(FIGS, "fig5_selectivity_landscape.png"))
    plt.close(fig)
    print("wrote results/figures/fig5_selectivity_landscape.png")


if __name__ == "__main__":
    main()
