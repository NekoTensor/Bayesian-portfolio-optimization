"""Allocators: belief -> weights, over a feasible set the caller supplies.

The comparison this project is built around is between :func:`max_sharpe_convex`
and :func:`max_sharpe_gp`. They optimise the *identical* objective over the
*identical* feasible set and differ only in search method -- SLSQP exploits the
smoothness of the Sharpe surface, ``gp_minimize`` treats it as a black box. Any
performance difference is therefore attributable to search, which is the only
way the "does Bayesian optimisation help?" question can be honestly asked.

``search_budget`` is exposed on both so that thoroughness can be equalised too:
comparing dozens of GP evaluations against a single SLSQP start would confound
the method with the effort spent on it.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .estimators import Belief
from .objectives import FeasibleSet, neg_sharpe, project_to_feasible, simplex_from_logits

__all__ = [
    "ALLOCATORS",
    "Allocator",
    "equal_weight",
    "max_sharpe_convex",
    "max_sharpe_gp",
    "min_variance",
]

#: ``(belief, feasible, prev_weights) -> weights``
Allocator = Callable[[Belief, FeasibleSet, "np.ndarray | None"], np.ndarray]


def _finalise(w: np.ndarray, feasible: FeasibleSet) -> np.ndarray:
    """Return the nearest admissible portfolio to ``w``.

    Projection rather than clip-then-renormalise. Renormalising after clipping is
    the pattern that reintroduces cap violations -- clipping ``[0.5, 0.5, 0]`` to a
    40% cap gives ``[0.4, 0.4, 0]``, and dividing by the new sum puts it straight
    back to ``[0.5, 0.5, 0]``. The projection has no such fixed point.
    """
    w = np.asarray(w, dtype=float)
    if w.shape != (feasible.n_assets,) or not np.isfinite(w).all():
        return feasible.equal_weight()
    return project_to_feasible(w, feasible)


def equal_weight(belief, feasible: FeasibleSet, prev_weights=None) -> np.ndarray:
    """1/N. It ignores the belief entirely, which is the point.

    DeMiguel, Garlappi & Uppal (2009) found no optimised policy in their sample
    consistently beat 1/N out of sample once estimation error was accounted for.
    Any study of portfolio optimisation that omits this baseline is not measuring
    what it claims to measure.
    """
    return feasible.equal_weight()


def min_variance(belief: Belief, feasible: FeasibleSet, prev_weights=None) -> np.ndarray:
    """Minimum-variance portfolio: uses ``Sigma`` but ignores ``mu`` completely.

    A useful diagnostic. Expected returns are far harder to estimate than
    covariances, so if this beats the max-Sharpe allocators out of sample, the
    honest conclusion is that the ``mu`` estimates carry no signal.
    """
    from scipy.optimize import minimize

    cov = np.asarray(belief.cov, dtype=float)
    n = feasible.n_assets
    result = minimize(
        lambda w: float(w @ cov @ w),
        feasible.equal_weight(),
        jac=lambda w: 2.0 * cov @ w,
        method="SLSQP",
        bounds=feasible.bounds,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    return _finalise(result.x if result.success else feasible.equal_weight(), feasible)


def max_sharpe_convex(
    belief: Belief,
    feasible: FeasibleSet,
    prev_weights=None,
    frequency: int = 52,
    cost_bps: float = 0.0,
    rebalances_per_year: float = 4.0,
    search_budget: int = 5,
    random_state: int = 0,
) -> np.ndarray:
    """Sequential least-squares maximisation of the Sharpe ratio.

    Multi-start because the Sharpe surface over a capped simplex is not concave
    in general; ``search_budget`` starts are drawn from a Dirichlet so the
    restarts spread across the simplex rather than clustering near 1/N.
    """
    from scipy.optimize import minimize

    def objective(w: np.ndarray) -> float:
        return neg_sharpe(
            w, belief.mu, belief.cov, frequency,
            prev_weights=prev_weights, cost_bps=cost_bps,
            rebalances_per_year=rebalances_per_year,
        )

    rng = np.random.default_rng(random_state)
    starts = [feasible.equal_weight()]
    starts += [rng.dirichlet(np.ones(feasible.n_assets)) for _ in range(max(0, search_budget - 1))]

    best, best_value = None, np.inf
    for x0 in starts:
        result = minimize(
            objective,
            project_to_feasible(x0, feasible),
            method="SLSQP",
            bounds=feasible.bounds,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if result.success and result.fun < best_value:
            best, best_value = result.x, float(result.fun)

    return _finalise(best if best is not None else feasible.equal_weight(), feasible)


def max_sharpe_gp(
    belief: Belief,
    feasible: FeasibleSet,
    prev_weights=None,
    frequency: int = 52,
    cost_bps: float = 0.0,
    rebalances_per_year: float = 4.0,
    search_budget: int = 60,
    random_state: int = 42,
    logit_bound: float = 4.0,
) -> np.ndarray:
    """Gaussian-process (Bayesian) optimisation of the same Sharpe objective.

    ``gp_minimize`` fits a GP surrogate to the objective and picks each next
    evaluation by maximising Expected Improvement. It searches in unconstrained
    logit space; :func:`~portfolio.objectives.simplex_from_logits` maps that onto
    the same feasible set SLSQP searches directly.

    The original version of this project searched ``[0, 1]^n`` and normalised
    afterwards, which is not a uniform parameterisation of the simplex and
    quietly biased the search toward the interior. ``logit_bound`` caps the box
    at +/-4, enough to reach weights near 0 and near the cap without wasting
    evaluations in saturated regions where softmax is flat.
    """
    from skopt import gp_minimize
    from skopt.space import Real

    def objective(logits) -> float:
        return neg_sharpe(
            simplex_from_logits(logits, feasible),
            belief.mu, belief.cov, frequency,
            prev_weights=prev_weights, cost_bps=cost_bps,
            rebalances_per_year=rebalances_per_year,
        )

    space = [Real(-logit_bound, logit_bound) for _ in range(feasible.n_assets)]
    result = gp_minimize(
        objective,
        space,
        n_calls=search_budget,
        n_initial_points=min(10, max(5, search_budget // 5)),
        acq_func="EI",
        random_state=random_state,
        noise=1e-10,  # the objective is deterministic given the belief
    )
    return _finalise(simplex_from_logits(result.x, feasible), feasible)


#: Registry used by the experiment scripts so the run is declarative.
ALLOCATORS = {
    "equal_weight": equal_weight,
    "min_variance": min_variance,
    "convex": max_sharpe_convex,
    "gp_bayesopt": max_sharpe_gp,
}
