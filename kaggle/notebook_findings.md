## What to take from it

- **Ten points rarely identify the best proxy.**
  - When the proxies are close in quality, the winner of phase 1 is truly the best 28–41% of the time (chance: 25%).
  - When they are well separated, it is truly the best 43–64% of the time.
- **The winning correlation is inflated.** With 10 random points, the winner's observed Spearman is about 0.65 against
  a true 0.47. Four identical proxies show exactly the same inflation, so a high winning correlation is no evidence
  that the proxy is better.
- **The collapse in phase 2 is expected.** On the five configurations a proxy ranks highest, its correlation with the
  LB averages between −0.02 and 0.27 everywhere in the grid, including when it is the truly best proxy. Those five
  have similar true quality, so LB noise decides their order.
- **Average or select depending on how much error the proxies share.**
  - With independent errors, averaging the proxies beats even the best single one.
  - With shared errors (cuts of one out-of-fold matrix), averaging dilutes the good proxy with the bad ones. There,
    selecting by correlation pays off once you have about 20 LB points.
- **In practice,** prefer changes that improve every reasonable proxy at once. Do not read a low correlation among
  your top few submissions as evidence against the proxy that chose them.

Limitations: Gaussian, linear proxies without directional bias; LB noise independent across submissions (a real public
split is fixed); phase 2 always submits 5 configurations.

Code (MIT) and the full results (CC0) are in the attached dataset.
