"""
common.py
=========
Paths, plotting defaults and small helpers shared by every script in src/.

Keeping these in one place means the figures in the report all look alike and
that no script has to guess where the data lives.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "data", "raw")
PROC = os.path.join(HERE, "data", "processed")
FIGS = os.path.join(HERE, "results", "figures")
TABS = os.path.join(HERE, "results", "tables")

for _d in (RAW, PROC, FIGS, TABS):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
# Every split, every model, every shuffle derives from this one number. It is
# quoted in the README so a grader can reproduce the exact tables.
SEED = 20261109

# --------------------------------------------------------------------------
# Modelling constants (quoted verbatim in the report's methods section)
# --------------------------------------------------------------------------

POTENCY_TYPES = ["IC50", "Ki", "Kd", "EC50"]
POTENCY_UNITS = "nM"

MORGAN_RADIUS = 2
MORGAN_BITS = 1024

N_SPLITS = 5          # scaffold splits, for mean +/- SD on every metric
TEST_SIZE = 0.25      # fraction of SCAFFOLD GROUPS held out
MIN_COMPOUNDS = 150   # below this a scaffold-split model is not worth fitting

# --------------------------------------------------------------------------
# Plot style
# --------------------------------------------------------------------------

INK = "#1b3a5c"
ACCENT = "#c1543a"
MUTED = "#8a9bab"
GREEN = "#3f7d58"
SERIES = [INK, ACCENT, GREEN, MUTED, "#7b5aa6", "#b8893a"]

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


def rule(title: str) -> None:
    """Print a section banner so long script logs stay readable."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def p_activity(value_nm) -> np.ndarray:
    """nM concentration -> pActivity = -log10(molar). Vectorised."""
    v = np.asarray(value_nm, dtype=float)
    return -np.log10(v * 1e-9)


def ic50_um(p) -> np.ndarray:
    """pActivity -> IC50 in micromolar, for reporting to a chemist."""
    return 10.0 ** (-np.asarray(p, dtype=float)) * 1e6
