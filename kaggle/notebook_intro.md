# Which Offline Score Tracks the LB? Ten Points Are Not Enough

You have several offline scores for each candidate (the full out-of-fold score, a cut of it, a holdout) and ten
leaderboard results. One of them correlated best with the leaderboard. Should you rank your next submissions by it?

This notebook simulates that decision with a known truth. A pool of 300 configurations has a latent quality `q`.
Proxy `k` is `p_k = rho_k q + sqrt(1 - rho_k^2) (sqrt(c) s + sqrt(1 - c) e_k)`: `rho_k` says how well it tracks the
truth, and `c` how much error the proxies share. The public LB is `L = r q + sqrt(1 - r^2) u`.

A season has two phases. In phase 1 you submit `n1` configurations, chosen at random or as the top of the pool, and
measure each proxy's Spearman correlation with the LB. In phase 2 you submit the top 5 of the rest, ranked by one of:
- the proxy that won phase 1 (`select`);
- the mean of all proxies (`mean`);
- the truly best single proxy (`oracle`).

The grid has 3 proxy sets, 2 levels of shared error, 3 LB reliabilities, 5 phase-1 sizes and 2 phase-1 designs, with
1,000 seasons per cell. It runs in under a minute on CPU.

I wrote this after it happened to me in a competition. A cut of my out-of-fold score correlated 0.82 (Spearman) with
the LB over ten submissions. I picked the next five by it, and on those five the correlation was 0.05.
