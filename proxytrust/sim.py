"""One simulated competition season, vectorised over repetitions.

A pool of N configurations has a latent true quality q ~ N(0, 1). Each offline proxy k (an OOF score, a cut of it, a
holdout) is p_k = rho_k * q + sqrt(1 - rho_k^2) * (sqrt(c) * s + sqrt(1 - c) * e_k), and a public leaderboard score
is L = r * q + sqrt(1 - r^2) * u, with independent standard normal s, e_k and u. rho_k is how well proxy k tracks the
truth, r how well the LB does (corr(p_k, L) = rho_k * r), and c how much of their error the proxies share: proxies
computed from the same out-of-fold predictions share most of it.

Season: submit n1 configurations (phase 1), measure every proxy's Spearman correlation with the LB on them, then rank
the remaining pool by a strategy and submit its top n2 (phase 2).

Strategies for phase 2:
    select  the proxy with the highest phase-1 Spearman (ties: the lowest index)
    mean    the mean of the proxies' standardised scores, no calibration at all
    oracle  the single proxy with the highest true rho_k (unknown in practice; with independent errors the mean
            of all proxies can beat it)

Phase-1 designs:
    random  n1 configurations drawn at random from the pool
    top     the top n1 by the mean of the proxies, which is how submissions are usually chosen
"""

from __future__ import annotations

import numpy as np


def ranks(x: np.ndarray) -> np.ndarray:
    """Ranks along the last axis (0 .. n-1); continuous data, so ties have probability zero."""
    return np.argsort(np.argsort(x, axis=-1), axis=-1).astype(float)


def spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Spearman correlation of two (reps, n) arrays."""
    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean(-1, keepdims=True)
    rb -= rb.mean(-1, keepdims=True)
    return (ra * rb).sum(-1) / np.sqrt((ra ** 2).sum(-1) * (rb ** 2).sum(-1))


def population_spearman(rho: float) -> float:
    """Spearman correlation of a bivariate normal with Pearson correlation rho."""
    return 6 / np.pi * np.arcsin(rho / 2)


def draw(rng: np.random.Generator, reps: int, n_pool: int, rhos: np.ndarray, r: float, c: float = 0.0):
    q = rng.standard_normal((reps, n_pool))
    e = np.sqrt(c) * rng.standard_normal((reps, n_pool)) + np.sqrt(1 - c) * rng.standard_normal((len(rhos), reps, n_pool))
    p = rhos[:, None, None] * q + np.sqrt(1 - rhos[:, None, None] ** 2) * e
    lb = r * q + np.sqrt(1 - r ** 2) * rng.standard_normal((reps, n_pool))
    return q, p, lb


def season(rng: np.random.Generator, reps: int, rhos, r: float, n1: int, n2: int = 5, n_pool: int = 300,
           design: str = "random", c: float = 0.0) -> dict[str, np.ndarray]:
    rhos = np.asarray(rhos, float)
    q, p, lb = draw(rng, reps, n_pool, rhos, r, c)
    z = (p - p.mean(-1, keepdims=True)) / p.std(-1, keepdims=True)
    mean_score = z.mean(0)
    rows = np.arange(reps)[:, None]

    if design == "random":
        first = np.argsort(rng.random((reps, n_pool)), axis=-1)[:, :n1]
    elif design == "top":
        first = np.argsort(-mean_score, axis=-1)[:, :n1]
    else:
        raise ValueError(design)
    taken = np.zeros((reps, n_pool), bool)
    taken[rows, first] = True

    rho1 = np.stack([spearman(p[k][rows, first], lb[rows, first]) for k in range(len(rhos))])  # (P, reps)
    sel = rho1.argmax(0)
    best = int(rhos.argmax())

    def phase2(score: np.ndarray) -> np.ndarray:
        s = np.where(taken, -np.inf, score)
        return np.argsort(-s, axis=-1)[:, :n2]

    picks = {
        "select": phase2(p[sel, rows[:, 0]]),
        "mean": phase2(mean_score),
        "oracle": phase2(p[best]),
    }
    out = {
        "sel": sel,
        # whether the chosen proxy is a truly best one; undefined when every proxy is equally good
        "hit": np.isclose(rhos[sel], rhos.max()) if np.ptp(rhos) > 0 else np.full(reps, np.nan),
        "rho1_sel": rho1[sel, np.arange(reps)],
        "rho1_best": rho1[best],
        "q1_mean": q[rows, first].mean(-1),
    }
    for name, idx in picks.items():
        out[f"q2_{name}"] = q[rows, idx].mean(-1)
        out[f"q2max_{name}"] = q[rows, idx].max(-1)
    second = picks["select"]
    out["rho2_sel"] = spearman(p[sel, rows[:, 0]][rows, second], lb[rows, second]) if n2 > 2 else np.full(reps, np.nan)
    return out
