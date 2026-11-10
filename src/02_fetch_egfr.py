"""
02_fetch_egfr.py
================
Download the EGFR (ERBB1) bioactivity set from PubChem, mirroring exactly the
procedure used for HER2 in the Part 1 workshop (sections 3, 5.2 and 6.2).

NEEDS INTERNET. Run this once; afterwards every other script works offline.

What it does
------------
  1. Ask PubChem which assays are linked to Gene ID 1956 (human EGFR).
  2. Pull a one-line summary for each AID (name, target, counts).
  3. Auto-triage the assay list into keep / drop with explicit, inspectable
     rules, and write it to a catalogue CSV.
  4. Download the data table for every kept AID and concatenate.

Step 3 is the step that matters, and the auto-triage is a FIRST PASS ONLY.
Open the catalogue, read the `keep` and `drop_reason` columns, and override
anything the rules got wrong before running step 4. The rules are deliberately
conservative: when in doubt they drop, because a mis-kept cell-viability assay
silently corrupts the endpoint while a mis-dropped one only costs data.

Why these rules
---------------
  * Drop MUTANT EGFR assays. The catalogue for HER2 already shows a large
    block of `L747-T751del`, `E746-A750del`, `S752-I759del` and `L858R`
    entries. Those are oncogenic kinase-domain mutants with materially
    different inhibitor pharmacology -- gefitinib is ~100x more potent on
    L858R than on wild type. Pooling them into one "EGFR" endpoint would make
    the HER2/EGFR difference meaningless.
  * Drop CELL PROLIFERATION / VIABILITY assays. They measure whether cells
    died, which depends on permeability, efflux and off-target toxicity as
    well as on target engagement. The HER2 side of this project is biochemical
    kinase inhibition, so the EGFR side must be too, or the selectivity
    difference confounds assay format with biology.
  * Drop SELECTIVITY and RATIO endpoints. They are already differences; using
    them as an absolute potency would double-count.

Usage
-----
  python src/02_fetch_egfr.py
  python src/02_fetch_egfr.py --catalogue-only   # stop after step 3 to review
  python src/02_fetch_egfr.py --download-only    # step 4, after you edited it
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import time

import pandas as pd
import requests

from common import RAW, rule

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

EGFR_GENE_ID = 1956          # human EGFR / ERBB1 / HER1
HER2_GENE_ID = 2064          # for reference; HER2 was downloaded in Part 1

CATALOGUE = os.path.join(RAW, "egfr_assay_catalogue.csv")
DATAFILE = os.path.join(RAW, "egfr_pubchem_data.csv")

# PubChem asks for no more than 5 requests per second. 0.25 s is polite and
# keeps a 250-assay download to a few minutes.
PAUSE = 0.25


# --------------------------------------------------------------------------
# HTTP with retries
# --------------------------------------------------------------------------


def pubchem_get(url: str, retries: int = 4, timeout: int = 60) -> str | None:
    """GET a PUG REST URL, retrying on 5xx and transient network errors.

    Returns the response body, or None if the resource genuinely does not
    exist (404) or every retry failed. Callers must handle None.
    """
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                time.sleep(PAUSE)
                return r.text
            if r.status_code == 404:
                return None                      # nothing there; do not retry
            # 503 means PubChem is throttling us: back off and try again.
            wait = 2 ** attempt
            print(f"    HTTP {r.status_code}, retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
        except requests.RequestException as exc:
            wait = 2 ** attempt
            print(f"    {type(exc).__name__}, retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    return None


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


# --------------------------------------------------------------------------
# Steps 1-2: which assays exist, and what are they
# --------------------------------------------------------------------------


def list_aids(gene_id: int) -> list[int]:
    rule(f"1. ASSAYS LINKED TO GENE {gene_id}")
    txt = pubchem_get(f"{PUG}/assay/target/geneid/{gene_id}/aids/TXT")
    if not txt:
        raise SystemExit(f"PubChem returned nothing for gene {gene_id}.")
    aids = [int(a) for a in txt.split() if a.strip().isdigit()]
    print(f"{len(aids):,} AIDs linked to gene {gene_id}")
    return aids


def summarise(aids: list[int]) -> pd.DataFrame:
    """Fetch name / target / activity counts for each AID, 100 at a time."""
    rule("2. ASSAY SUMMARIES")
    rows = []
    for i, block in enumerate(chunked(aids, 100), start=1):
        ids = ",".join(str(a) for a in block)
        txt = pubchem_get(f"{PUG}/assay/aid/{ids}/summary/JSON")
        if not txt:
            print(f"  block {i}: no response, skipped", file=sys.stderr)
            continue
        try:
            data = pd.read_json(io.StringIO(txt))
        except ValueError:
            import json

            data = json.loads(txt)
            summaries = data.get("AssaySummaries", {}).get("AssaySummary", [])
            data = pd.DataFrame(summaries)
        if "AssaySummaries" in data.columns:
            data = pd.DataFrame(data["AssaySummaries"]["AssaySummary"])
        rows.append(data)
        print(f"  block {i}: {len(data)} summaries")

    if not rows:
        raise SystemExit("No assay summaries retrieved.")

    df = pd.concat(rows, ignore_index=True)

    # The summary schema nests the target list. Flatten it to a single string
    # so the triage rules below can regex over it.
    def flatten_target(t) -> str:
        if isinstance(t, list):
            names = []
            for item in t:
                if isinstance(item, dict):
                    names.append(str(item.get("Name", "")))
                else:
                    names.append(str(item))
            return " | ".join(n for n in names if n)
        if isinstance(t, dict):
            return str(t.get("Name", ""))
        return "" if pd.isna(t) else str(t)

    out = pd.DataFrame(
        {
            "AID": df.get("AID"),
            "assay_name": df.get("Name", "").astype(str),
            "target": df.get("Target", "").map(flatten_target) if "Target" in df else "",
            "n_active": df.get("ActiveCount", pd.Series(dtype="float")),
            "n_tested": df.get("TotalCount", pd.Series(dtype="float")),
        }
    )
    out = out.dropna(subset=["AID"])
    out["AID"] = out["AID"].astype(int)
    return out.drop_duplicates(subset="AID").reset_index(drop=True)


# --------------------------------------------------------------------------
# Step 3: triage
# --------------------------------------------------------------------------

# Each rule is (regex, reason). Applied to assay_name + target, case-insensitive.
DROP_RULES: list[tuple[str, str]] = [
    (r"\b(del|ins)\b|"
     r"[LTEGSV]\d{3}[A-Z]\b|"
     r"\b\d{3}-[A-Z]\d{3}\b|"
     r"\bmutant\b|\bmutation\b|T790M|L858R|L861Q|G719|C797",
     "mutant EGFR"),
    (r"proliferation|viability|cytotox|growth inhibition|\bGI50\b|\bCC50\b|anti-?tumou?r",
     "cell proliferation / viability"),
    (r"selectivity|\bratio\b|fold[- ]selective|protective index",
     "selectivity or ratio endpoint"),
    (r"\bsiRNA\b|\bRNAi\b|knockdown",
     "genetic screen, not a small-molecule assay"),
    (r"\bnavigating the kinome\b|profiling assay|panel of \d+ kinases",
     "broad profiling panel"),
]

# A kept assay must look like EGFR AND like a biochemical potency measurement.
EGFR_TARGET = r"epidermal growth factor receptor|\bEGFR\b|\berbB-?1\b|\bHER-?1\b"
POTENCY_HINT = (
    r"inhibit|IC50|Ki\b|Kd\b|EC50|autophosphorylation|kinase activity|"
    r"binding affinity|phosphorylation"
)


def triage(cat: pd.DataFrame) -> pd.DataFrame:
    rule("3. TRIAGE")
    text = (cat["assay_name"].fillna("") + " || " + cat["target"].fillna(""))

    cat = cat.copy()
    cat["keep"] = True
    cat["drop_reason"] = ""

    for pattern, reason in DROP_RULES:
        hit = text.str.contains(pattern, case=False, regex=True, na=False) & cat["keep"]
        cat.loc[hit, "keep"] = False
        cat.loc[hit, "drop_reason"] = reason

    # Must actually be EGFR.
    not_egfr = ~text.str.contains(EGFR_TARGET, case=False, regex=True, na=False) & cat["keep"]
    cat.loc[not_egfr, ["keep", "drop_reason"]] = [False, "target is not wild-type EGFR"]

    # Must look like a potency measurement.
    no_potency = ~text.str.contains(POTENCY_HINT, case=False, regex=True, na=False) & cat["keep"]
    cat.loc[no_potency, ["keep", "drop_reason"]] = [False, "not a potency endpoint"]

    print(f"kept   : {int(cat['keep'].sum()):,}")
    print(f"dropped: {int((~cat['keep']).sum()):,}")
    print("\nreasons for dropping:")
    print(cat.loc[~cat["keep"], "drop_reason"].value_counts().to_string())
    return cat


# --------------------------------------------------------------------------
# Step 4: download the kept data tables
# --------------------------------------------------------------------------


def download(cat: pd.DataFrame) -> pd.DataFrame:
    rule("4. DOWNLOADING DATA TABLES")
    keep = cat.loc[cat["keep"] == True, "AID"].tolist()  # noqa: E712
    print(f"{len(keep):,} assays to download\n")

    frames, failed = [], []
    for i, aid in enumerate(keep, start=1):
        txt = pubchem_get(f"{PUG}/assay/aid/{aid}/CSV")
        if not txt:
            failed.append(aid)
            continue
        try:
            tab = pd.read_csv(io.StringIO(txt), low_memory=False)
        except Exception:
            failed.append(aid)
            continue

        # The first rows of a PubChem data table are the RESULT_TYPE /
        # RESULT_DESCR / RESULT_UNIT metadata rows, not data (Part 1, 5.2).
        # They are identifiable because PUBCHEM_RESULT_TAG is not an integer.
        tag = pd.to_numeric(tab.get("PUBCHEM_RESULT_TAG"), errors="coerce")
        tab = tab[tag.notna()]

        if len(tab):
            tab["AID"] = aid
            frames.append(tab)
        if i % 25 == 0 or i == len(keep):
            print(f"  {i}/{len(keep)} assays, {sum(len(f) for f in frames):,} rows so far")

    if failed:
        print(f"\n{len(failed)} assays failed to download: {failed[:20]}")

    if not frames:
        raise SystemExit("Nothing downloaded.")

    data = pd.concat(frames, ignore_index=True)
    print(f"\ntotal rows: {len(data):,}")
    if "PUBCHEM_CID" in data.columns:
        print(f"unique CIDs: {data['PUBCHEM_CID'].nunique():,}")
    return data


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalogue-only", action="store_true",
                    help="build the assay catalogue and stop, so you can review it")
    ap.add_argument("--download-only", action="store_true",
                    help="skip straight to downloading, using the catalogue on disk")
    ap.add_argument("--gene", type=int, default=EGFR_GENE_ID)
    args = ap.parse_args()

    if not args.download_only:
        aids = list_aids(args.gene)
        cat = triage(summarise(aids))
        cat.to_csv(CATALOGUE, index=False)
        print(f"\nwrote {CATALOGUE}")
        print("\n>>> REVIEW THIS FILE before downloading. Check the `keep` column,")
        print(">>> especially rows whose drop_reason is 'mutant EGFR'. Then run:")
        print(">>>     python src/02_fetch_egfr.py --download-only")
        if args.catalogue_only:
            return
    else:
        if not os.path.exists(CATALOGUE):
            raise SystemExit(f"No catalogue at {CATALOGUE}. Run without --download-only first.")
        cat = pd.read_csv(CATALOGUE)

    data = download(cat)
    data.to_csv(DATAFILE, index=False)
    print(f"wrote {DATAFILE}")


if __name__ == "__main__":
    main()
