"""Feasible sets, weight parameterisations, and the objective being optimised.

The single most important property of this module is that *every* allocator
optimises the same objective over the same feasible set. In the first version of
this project the convex baseline was allowed to short (bounds ``[-0.5, 0.5]``)
while the Gaussian-process search was long-only, so the reported performance gap
between them confounded the optimiser with its search space. :class:`FeasibleSet`
exists to make that mistake impossible to repeat: an allocator takes one, and
every weight vector it returns is checked against it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "FeasibleSet",
    "neg_sharpe",
    "project_to_feasible",
    "simplex_from_logits",
    "turnover",
]


@dataclass(frozen=True)
class FeasibleSet:
    """The budget-constrained box ``{w : sum(w) = 1, min_weight <= w <= max_weight}``.

    Long-only with a per-asset cap is the default because it is the constraint
    set a real mandate would impose, and because unconstrained mean-variance on
    estimated moments produces the extreme long/short positions that make the
    in-sample optimum meaningless out of sample.
    """

    n_assets: int
    min_weight: float = 0.0
    max_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.n_assets < 1:
            raise ValueError("n_assets must be positive")
        if self.min_weight > self.max_weight:
            raise ValueError("min_weight must not exceed max_weight")
        # The budget constraint has to be satisfiable inside the box.
        if self.n_assets * self.max_weight < 1.0 - 1e-12:
            raise ValueError(
                f"max_weight={self.max_weight} across {self.n_assets} assets cannot "
                f"sum to 1 (upper bound {self.n_assets * self.max_weight:.3f})"
            )
        if self.n_assets * self.min_weight > 1.0 + 1e-12:
            raise ValueError(
                f"min_weight={self.min_weight} across {self.n_assets} assets forces "
                f"a sum of at least {self.n_assets * self.min_weight:.3f} > 1"
            )

    @property
    def bounds(self) -> list[tuple[float, float]]:
        """Per-asset ``(lo, hi)`` pairs, in the form ``scipy.optimize`` wants."""
        return [(self.min_weight, self.max_weight)] * self.n_assets

    def contains(self, weights, tol: float = 1e-6) -> bool:
        w = np.asarray(weights, dtype=float)
        if w.shape != (self.n_assets,) or not np.isfinite(w).all():
            return False
        return (
            abs(w.sum() - 1.0) <= tol
            and (w >= self.min_weight - tol).all()
            and (w <= self.max_weight + tol).all()
        )

    def equal_weight(self) -> np.ndarray:
        """The 1/N point, which is always feasible given the checks above."""
        return np.repeat(1.0 / self.n_assets, self.n_assets)


def project_to_feasible(v, feasible: FeasibleSet) -> np.ndarray:
    """Euclidean projection of ``v`` onto ``feasible``.

    Projecting onto ``{w : sum(w) = 1, lo <= w <= hi}`` reduces to finding the
    single dual variable ``theta`` with ``sum(clip(v - theta, lo, hi)) == 1``.
    The left-hand side is non-increasing in ``theta``, so a bisection converges
    reliably; the bracket below is wide enough to straddle the root for any ``v``.
    """
    v = np.asarray(v, dtype=float)
    if v.shape != (feasible.n_assets,):
        raise ValueError(f"expected {feasible.n_assets} weights, got {v.shape}")

    lo, hi = feasible.min_weight, feasible.max_weight
    theta_lo = float(v.min() - hi)  # everything clips to hi -> sum = n*hi >= 1
    theta_hi = float(v.max() - lo)  # everything clips to lo -> sum = n*lo <= 1

    for _ in range(200):
        theta = 0.5 * (theta_lo + theta_hi)
        total = np.clip(v - theta, lo, hi).sum()
        if abs(total - 1.0) < 1e-12:
            break
        if total > 1.0:
            theta_lo = theta
        else:
            theta_hi = theta

    w = np.clip(v - 0.5 * (theta_lo + theta_hi), lo, hi)
    # Bisection leaves an O(1e-13) budget error; rescale the slack away.
    return w + (1.0 - w.sum()) / feasible.n_assets


def simplex_from_logits(logits, feasible: FeasibleSet) -> np.ndarray:
    """Map an unconstrained vector in R^n onto ``feasible``.

    Gaussian-process search wants a box to sample from, not a simplex. Softmax
    turns the box into budget-feasible weights, and the projection then enforces
    the per-asset cap (softmax alone cannot). Doing it this way -- rather than
    the original approach of sampling in ``[0, 1]^n`` and normalising afterwards
    -- means the GP sees a smooth surjection onto the same set SLSQP searches,
    instead of a distorted one that over-weights the interior of the box.
    """
    z = np.asarray(logits, dtype=float)
    if z.shape != (feasible.n_assets,):
        raise ValueError(f"expected {feasible.n_assets} logits, got {z.shape}")
    e = np.exp(z - z.max())  # shift for numerical stability
    w = e / e.sum()
    if feasible.min_weight == 0.0 and feasible.max_weight >= 1.0:
        return w
    return project_to_feasible(w, feasible)


def turnover(weights, prev_weights=None) -> float:
    """Total traded notional as a fraction of portfolio value.

    Both sides of the trade are counted, so rotating an entire portfolio scores
    2.0 (sell 100%, buy 100%). Transaction costs elsewhere are charged against
    this quantity, which makes ``cost_bps`` a per-unit-traded cost.
    """
    w = np.asarray(weights, dtype=float)
    prev = np.zeros_like(w) if prev_weights is None else np.asarray(prev_weights, float)
    return float(np.abs(w - prev).sum())


def neg_sharpe(
    weights,
    mu,
    cov,
    frequency: int = 52,
    risk_free: float = 0.0,
    prev_weights=None,
    cost_bps: float = 0.0,
    rebalances_per_year: float = 4.0,
) -> float:
    """Negative annualised Sharpe ratio of ``weights`` under the belief ``(mu, cov)``.

    ``mu`` and ``cov`` are per-period moments; ``frequency`` annualises them.
    When ``cost_bps`` is non-zero the expected cost of trading into ``weights``
    is amortised over a year at ``rebalances_per_year`` and charged against the
    numerator, so a cost-aware allocator will hold a position it would otherwise
    trade out of.
    """
    w = np.asarray(weights, dtype=float)
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)

    variance = float(w @ cov @ w)
    if not np.isfinite(variance) or variance <= 0.0:
        return 1e6  # infeasible/degenerate: make it unattractive, don't crash

    ann_return = frequency * float(w @ mu)
    ann_vol = np.sqrt(frequency * variance)

    if cost_bps:
        drag = turnover(w, prev_weights) * (cost_bps / 1e4) * rebalances_per_year
        ann_return -= drag

    return -(ann_return - risk_free) / ann_vol
