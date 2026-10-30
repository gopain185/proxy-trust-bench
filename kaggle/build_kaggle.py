"""Builds the Kaggle dataset and notebook for this project.

    python kaggle/build_kaggle.py [--public] [--notebook-only]

writes kaggle/dataset/ (the per-season results of the full grid, the summary table and the datasheet) and
kaggle/notebook/ (a CPU notebook that embeds the package source and reruns the grid).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "kaggle"
DATASET_ID = "locdunhat/proxy-selection-simulations"
NOTEBOOK_ID = "locdunhat/which-offline-score-tracks-the-lb"
TITLE = "Which Offline Score Tracks the LB?"


def build_dataset(public: bool) -> Path:
    out = OUT / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    for name in ["runs.csv", "summary.csv"]:
        (out / name).write_bytes((ROOT / "results" / name).read_bytes())
    datasheet = (OUT / "DATASHEET.md").read_text(encoding="utf-8")
    (out / "DATASHEET.md").write_text(datasheet, encoding="utf-8")
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "Proxy Selection Simulations",
        "id": DATASET_ID,
        "subtitle": "Picking an offline score by its LB correlation: 180,000 simulated seasons",
        "description": datasheet,
        "isPrivate": not public,
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": ["benchmark", "simulations"],
    }, indent=2), encoding="utf-8")
    return out


def _sources() -> dict[str, str]:
    return {f.relative_to(ROOT).as_posix(): f.read_text(encoding="utf-8") for f in sorted((ROOT / "proxytrust").glob("*.py"))}


SETUP = """

import os, sys
ROOT = "/kaggle/working/proxy-trust-bench" if os.path.isdir("/kaggle/working") else os.path.abspath("proxy-trust-bench")
for path, text in SOURCES.items():
    os.makedirs(os.path.dirname(os.path.join(ROOT, path)), exist_ok=True)
    with open(os.path.join(ROOT, path), "w", encoding="utf-8") as f:
        f.write(text)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
"""

RUN = """import time
import pandas as pd
from proxytrust import run as bench
pd.set_option("display.width", 250); pd.set_option("display.precision", 3)
t = time.time()
df = bench.run()
print(len(df), "seasons in", round(time.time() - t), "s")
summary = bench.summarise(df)
r7 = summary[summary.r == 0.7]
"""

HIT = """r7[r7.scenario != "equal"].pivot_table(index=["scenario", "c", "design"], columns="n1", values="hit").round(2)
"""

OPT = """r7[r7.design == "random"].pivot_table(index=["scenario", "c"], columns="n1", values=["rho1_sel", "rho_true_sel"]).round(2)
"""

COLLAPSE = """print("phase-2 Spearman of the selected proxy, over the whole grid:", round(float(summary.rho2_sel.min()), 2), "to", round(float(summary.rho2_sel.max()), 2))
x = df[(df.scenario == "close") & (df.c == 0.8) & (df.r == 0.7) & (df.n1 == 10) & (df.design == "top")]
print("close proxies, c = 0.8, r = 0.7, 10 top submissions: median phase-2 Spearman", round(x.rho2_sel.median(), 2),
      "| share at or below 0.1:", round((x.rho2_sel <= 0.1).mean(), 2))
r7.pivot_table(index=["scenario", "c", "design"], columns="n1", values="rho2_sel").round(2)
"""

REGRET = """r7.pivot_table(index=["scenario", "c", "design"], columns="n1", values=["regret_select", "regret_mean"]).round(3)
"""

COMPARE = """import glob
hits = glob.glob("/kaggle/input/**/summary.csv", recursive=True)
if hits:
    saved = pd.read_csv(hits[0])
    key = ["scenario", "c", "r", "n1", "design"]
    m = summary.merge(saved, on=key, suffixes=("", "_saved"))
    print("cells matched:", len(m), "largest difference in mean phase-2 quality:", float((m.q2_select - m.q2_select_saved).abs().max()))
else:
    print("dataset not attached")
"""


def _cells() -> list[tuple[str, str]]:
    return [
        ("markdown", (OUT / "notebook_intro.md").read_text(encoding="utf-8")),
        ("code", "SOURCES = " + json.dumps(_sources(), indent=0) + SETUP),
        ("markdown", "## 1. The grid"),
        ("code", RUN),
        ("markdown", "## 2. How often the best-correlated proxy is truly the best (r = 0.7)"),
        ("code", HIT),
        ("markdown", "## 3. The winner's observed correlation against its true correlation (random phase 1, r = 0.7)"),
        ("code", OPT),
        ("markdown", "## 4. The collapse on the next five submissions"),
        ("code", COLLAPSE),
        ("markdown", "## 5. What it costs: true quality lost against the truly best proxy (r = 0.7; negative = better)"),
        ("code", REGRET),
        ("markdown", "## 6. Against the saved runs"),
        ("code", COMPARE),
        ("markdown", (OUT / "notebook_findings.md").read_text(encoding="utf-8")),
    ]


def build_notebook(public: bool) -> Path:
    out = OUT / "notebook"
    out.mkdir(parents=True, exist_ok=True)
    cells = []
    for i, (kind, src) in enumerate(_cells()):
        c = {"cell_type": kind, "id": f"cell-{i}", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            c.update(outputs=[], execution_count=None)
        cells.append(c)
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (out / "proxy-trust-bench.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": NOTEBOOK_ID, "title": TITLE, "code_file": "proxy-trust-bench.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": not public, "enable_gpu": False, "enable_internet": False,
        "dataset_sources": [DATASET_ID], "competition_sources": [], "kernel_sources": [],
        "keywords": ["benchmark"],
    }, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    public = "--public" in sys.argv
    if "--notebook-only" not in sys.argv:
        print("wrote", build_dataset(public))
    print("wrote", build_notebook(public))
