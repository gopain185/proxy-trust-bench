# proxy-trust-bench

You have several offline scores for each candidate (the full out-of-fold score, a cut of it, a holdout) and a
handful of leaderboard results. One proxy correlated best with the leaderboard over your last ten submissions:
should you rank the next candidates by it? This repository simulates that decision with known ground truth.

- **Kaggle notebook:** [Which Offline Score Tracks the LB? Ten Points Are Not Enough](https://www.kaggle.com/code/locdunhat/which-offline-score-tracks-the-lb)
- **Kaggle dataset:** [Proxy Selection Simulations](https://www.kaggle.com/datasets/locdunhat/proxy-selection-simulations) (CC0)

## The model

A pool of 300 configurations has a latent true quality `q ~ N(0, 1)`. Proxy `k` is

    p_k = rho_k * q + sqrt(1 - rho_k^2) * (sqrt(c) * s + sqrt(1 - c) * e_k)

and the public leaderboard is `L = r * q + sqrt(1 - r^2) * u`, with independent standard normal `s`, `e_k`, `u`.
`rho_k` is how well proxy `k` tracks the truth, `r` how well the leaderboard does, and `c` how much error the
proxies share (cuts of the same out-of-fold predictions share most of it).

A season has two phases:

1. Submit `n1` configurations, either drawn at random (`random`) or the top `n1` by the mean of the proxies (`top`,
   which is how submissions are usually chosen), and measure each proxy's Spearman correlation with the leaderboard.
2. Rank the rest of the pool and submit the top 5, by one of three strategies:
   - `select`: the proxy with the best phase-1 correlation;
   - `mean`: the mean of the standardised proxies, with no calibration;
   - `oracle`: the single proxy with the highest true `rho_k`.

Grid:

| setting | values |
| --- | --- |
| proxy sets (`rho`) | `spread` 0.40/0.55/0.70/0.85; `close` 0.60/0.65/0.70/0.75; `equal` 0.70 × 4 |
| shared error `c` | 0, 0.8 |
| leaderboard reliability `r` | 0.5, 0.7, 0.9 |
| phase-1 size `n1` | 5, 10, 15, 20, 40 |
| phase-1 design | `random`, `top` |

1,000 seasons per cell, which gives 180,000 rows.

## Findings

Unless stated otherwise, the numbers are for `r = 0.7` and `n1 = 10`.

1. **Ten points rarely identify the best proxy.** The proxy with the best phase-1 correlation is truly the best:
   - 28–41% of the time when the proxies are close (chance is 25%);
   - 43–64% of the time when they are well separated.

   At `n1 = 40` the hit rates are 39–60% for close proxies and 69–94% for well-separated ones.
2. **The winning correlation is inflated.** With a random phase 1, the winner's Spearman correlation is about 0.65,
   while its true correlation is about 0.47. That is the same inflation as when all four proxies are identical and
   the winner is pure noise. Averaged over the grid, the inflation is +0.21 at `n1 = 5`, +0.13 at 10 and +0.05 at 40.
   When phase 1 is the top of the pool, range restriction pulls every observed correlation down (to about 0.40–0.49).
3. **The correlation always collapses in phase 2, and this says nothing about the proxy.**
   - On the five configurations a proxy ranks highest, its own Spearman with the leaderboard averages −0.02 to 0.27
     across the whole grid, including when the proxy is the truly best one.
   - The five have similar true quality, so leaderboard noise decides their order. In the `close`, `c = 0.8`, `top`
     cell, the median phase-2 correlation is 0.10.
   - A proxy that "stopped working" on the next five submissions may be exactly as good as before.
4. **Averaging versus selecting depends on shared error.**
   - With independent proxy errors (`c = 0`), the mean of the proxies beats even the oracle single proxy, by up to
     0.47 standard deviations of true quality.
   - With shared error (`c = 0.8`) and proxies of unequal quality, the mean loses to the best single proxy: by 0.43
     (`spread`, random phase 1) to 0.69 (`spread`, top phase 1).
   - In that case selection pays off once `n1` reaches about 20. Its regret at `n1 = 20` is 0.09–0.14, against 0.23–0.30
     at `n1 = 10`.
5. **Equal proxies cost nothing to select among.** When the proxies are equally good, choosing by the best
   correlation loses nothing. You just learn nothing either.

What to do with ten leaderboard points: do not switch proxies on the best correlation alone. Prefer changes that
improve every reasonable proxy at once. Do not read a low correlation among your top few submissions as evidence
against the proxy that chose them.

## Usage

    python -m proxytrust.run            # full grid, writes results/runs.csv and results/summary.csv (under a minute)
    python -m proxytrust.run --quick    # 100 seasons, n1 in {5, 20}
    python -m pytest -q tests
    python kaggle/build_kaggle.py [--public] [--notebook-only]

Requires numpy and pandas; the tests use pytest.

## Limitations

- The model is Gaussian and linear, and each proxy has one fixed correlation with the truth. Real proxies can be
  biased in a direction that matters, for example an out-of-fold score that flatters one kind of component. That
  kind of bias is what a calibration against the leaderboard is meant to detect, and it is not modelled here.
- The leaderboard noise is independent across submissions. A real public leaderboard is one fixed split, so its
  errors are correlated across submissions that are similar.
- Phase 2 always submits 5 configurations.

## License

MIT for the code; the generated data is CC0.
