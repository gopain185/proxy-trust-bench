"""The benchmark grid.

    python -m proxytrust.run [--quick]

Three sets of four proxies (their true correlations rho with the latent quality), two levels of shared proxy error c,
three leaderboard reliabilities r, five phase-1 sizes and two phase-1 designs; every cell is one season repeated `reps` times with its own seed. Writes
results/runs.csv (one row per repetition) and results/summary.csv (one row per cell).
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from .sim import population_spearman, season

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = {
    "spread": [0.4, 0.55, 0.7, 0.85],
    "close": [0.6, 0.65, 0.7, 0.75],
    "equal": [0.7, 0.7, 0.7, 0.7],
}
CS = [0.0, 0.8]
RS = [0.5, 0.7, 0.9]
N1 = [5, 10, 15, 20, 40]
DESIGNS = ["random", "top"]
N2, POOL = 5, 300


def run(reps: int = 1000, n1s=N1) -> pd.DataFrame:
    frames = []
    for i, (name, c, r, n1, design) in enumerate(itertools.product(SCENARIOS, CS, RS, n1s, DESIGNS)):
        rng = np.random.default_rng(1000 + i)
        out = season(rng, reps, SCENARIOS[name], r, n1, N2, POOL, design, c)
        df = pd.DataFrame(out)
        df.insert(0, "rep", np.arange(reps))
        for col, val in [("design", design), ("n1", n1), ("r", r), ("c", c), ("scenario", name)]:
            df.insert(0, col, val)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["scenario", "c", "r", "n1", "design"]
    g = df.groupby(keys, sort=False)
    s = g.agg(hit=("hit", "mean"), rho1_sel=("rho1_sel", "mean"), rho1_best=("rho1_best", "mean"),
              rho2_sel=("rho2_sel", "mean"), q1=("q1_mean", "mean"),
              q2_select=("q2_select", "mean"), q2_mean=("q2_mean", "mean"), q2_oracle=("q2_oracle", "mean")).reset_index()
    s["rho_true_sel"] = [np.mean([population_spearman(SCENARIOS[sc][k] * r) for k in grp.sel])
                         for (sc, _, r, _, _), grp in g]
    s["optimism"] = s.rho1_sel - s.rho_true_sel
    s["regret_select"] = s.q2_oracle - s.q2_select
    s["regret_mean"] = s.q2_oracle - s.q2_mean
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    df = run(reps=100, n1s=[5, 20]) if a.quick else run()
    suffix = "_quick" if a.quick else ""
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / f"runs{suffix}.csv", index=False, float_format="%.6g")
    s = summarise(df)
    s.to_csv(ROOT / "results" / f"summary{suffix}.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 250)
    print(s.round(3).to_string())
