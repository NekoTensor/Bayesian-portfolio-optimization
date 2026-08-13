"""Beliefs about next period: estimators of the mean vector and covariance matrix.

This is the half of the problem the original version of this project ignored.
It called itself Bayesian on the strength of using ``gp_minimize`` -- a
*black-box optimisation* technique -- while estimating ``mu`` and ``Sigma`` with
plain sample moments. Bayesian *statistics* enters portfolio choice here, in how
you form beliefs from a short, noisy training window, not in how you search the
weight space afterwards.

The distinction matters empirically, not just terminologically. With ``N`` assets
and ``T`` observations the sample mean has estimation error of order
``sqrt(N/T)``, and mean-variance optimisation is an error-*maximising* operation:
it loads onto whichever asset's mean was most overestimated. Shrinking the
inputs attacks the problem at its source; searching the weight space harder does
not.

Every estimator here is a pure function of its training window, which is what
makes the walk-forward harness in :mod:`portfolio.backtest` free of look-ahead.

References
----------
Ledoit & Wolf (2004), "Honey, I Shrunk the Sample Covariance Matrix", JPM 30(4).
Jorion (1986), "Bayes-Stein Estimation for Portfolio Analysis", JFQA 21(3).
Black & Litterman (1992), "Global Portfolio Optimization", FAJ 48(5).
Gelman et al., *Bayesian Data Analysis* (3rd ed.), ch. 3 for the NIW posterior.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

__all__ = [
    "Belief",
    "ESTIMATORS",
    "bayes_stein",
    "black_litterman",
    "ledoit_wolf",
    "niw_posterior_predictive",
    "sample_moments",
]


class Belief(NamedTuple):
    """A per-period predictive mean and covariance for the next holding period."""

    mu: np.ndarray
    cov: np.ndarray


def _as_matrix(train_returns) -> np.ndarray:
    x = (
        train_returns.to_numpy(dtype=float)
        if isinstance(train_returns, (pd.DataFrame, pd.Series))
        else np.asarray(train_returns, dtype=float)
    )
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D (T, N) return panel, got shape {x.shape}")
    t, n = x.shape
    if t <= n:
        raise ValueError(
            f"training window has T={t} observations for N={n} assets; the sample "
            f"covariance is singular unless T > N"
        )
    if not np.isfinite(x).all():
        raise ValueError("training window contains non-finite returns")
    return x


def _nearest_psd(cov: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Symmetrise and clip eigenvalues so downstream Cholesky/inverse calls hold.

    Shrinkage targets are PSD by construction, but floating-point asymmetry of
    order 1e-18 is enough to make ``scipy`` refuse a solve, so this runs on every
    returned covariance.
    """
    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    if (eigvals >= floor).all():
        return cov
    return eigvecs @ np.diag(np.maximum(eigvals, floor)) @ eigvecs.T


def sample_moments(train_returns) -> Belief:
    """Plain sample mean and covariance -- the maximum-likelihood baseline.

    Included precisely because it is the thing every other estimator has to beat.
    """
    x = _as_matrix(train_returns)
    return Belief(mu=x.mean(axis=0), cov=_nearest_psd(np.cov(x, rowvar=False, ddof=1)))


def ledoit_wolf(train_returns) -> Belief:
    """Ledoit-Wolf shrinkage of the covariance toward constant correlation.

    The estimator is ``delta * F + (1 - delta) * S``, where ``F`` imposes a single
    average correlation on every pair and ``delta`` is chosen analytically to
    minimise expected squared Frobenius error. The constant-correlation target is
    used rather than scaled identity because equities share a dominant market
    factor: pretending assets are uncorrelated is a worse prior than pretending
    they are equally correlated.

    The mean is left at its sample value, which isolates the effect of covariance
    shrinkage alone. :func:`bayes_stein` is the counterpart that shrinks the mean.
    """
    x = _as_matrix(train_returns)
    t, n = x.shape
    dev = x - x.mean(axis=0)

    # The shrinkage-intensity algebra below is derived for the 1/T (biased)
    # covariance, so use it throughout and restore the 1/(T-1) convention at the
    # very end to stay consistent with the other estimators.
    sample = (dev.T @ dev) / t
    var = np.diag(sample).copy()
    std = np.sqrt(var)
    outer_std = np.outer(std, std)

    # Constant-correlation target: every pair gets the average correlation.
    r_bar = float((sample / outer_std).sum() - n) / (n * (n - 1))
    target = r_bar * outer_std
    np.fill_diagonal(target, var)

    # pi: summed asymptotic variance of the sample covariance entries,
    #     pi_ij = (1/T) sum_t (dev_ti dev_tj - s_ij)^2.
    dev_sq = dev**2
    phi = (dev_sq.T @ dev_sq) / t - sample**2
    pi_hat = float(phi.sum())

    # rho: asymptotic covariance between the sample entries and the target's own
    #      estimation error. theta_ii,ij = (1/T) sum_t dev_ti^3 dev_tj - s_ii s_ij.
    third = ((dev**3).T @ dev) / t
    theta = third - var[:, None] * sample  # theta[i, j] = theta_ii,ij
    # theta_jj,ij is the same quantity with i and j swapped, i.e. theta.T.
    ratio = np.sqrt(np.outer(1.0 / var, var))  # ratio[i, j] = sqrt(s_jj / s_ii)
    cross = 0.5 * r_bar * (ratio * theta + ratio.T * theta.T)
    np.fill_diagonal(cross, 0.0)
    rho_hat = float(np.trace(phi) + cross.sum())

    # gamma: squared Frobenius distance between target and sample.
    gamma_hat = float(((target - sample) ** 2).sum())

    if gamma_hat <= 0.0:
        delta = 0.0  # target already equals the sample; nothing to shrink toward
    else:
        delta = float(np.clip((pi_hat - rho_hat) / gamma_hat / t, 0.0, 1.0))

    shrunk = (delta * target + (1.0 - delta) * sample) * t / (t - 1.0)
    return Belief(mu=x.mean(axis=0), cov=_nearest_psd(shrunk))


def bayes_stein(train_returns) -> Belief:
    """Jorion's (1986) Bayes-Stein shrinkage of the mean toward the GMV return.

    The sample mean is inadmissible in dimension >= 3 under quadratic loss, and in
    portfolio choice its errors are amplified rather than averaged out. Jorion
    shrinks every asset's mean toward ``mu_gmv``, the expected return of the
    global minimum-variance portfolio, with an intensity that is estimated from
    how dispersed the sample means are relative to their own standard errors:
    tightly clustered means get shrunk hard, genuinely dispersed ones do not.

    The covariance is also inflated to reflect that ``mu`` is estimated rather
    than known -- the second term below is the parameter-uncertainty premium that
    plug-in mean-variance ignores entirely.
    """
    x = _as_matrix(train_returns)
    t, n = x.shape
    mu_hat = x.mean(axis=0)

    # Jorion's bias-adjusted covariance for the precision used in shrinking.
    sample = np.cov(x, rowvar=False, ddof=1)
    scale = (t - 1.0) / (t - n - 2.0) if t - n - 2.0 > 0 else 1.0
    sigma = _nearest_psd(scale * sample)
    sigma_inv = np.linalg.pinv(sigma)

    ones = np.ones(n)
    denom = float(ones @ sigma_inv @ ones)
    mu_gmv = float(ones @ sigma_inv @ mu_hat) / denom

    diff = mu_hat - mu_gmv * ones
    quad = float(diff @ sigma_inv @ diff)
    # lambda is the precision of the prior relative to the data; large dispersion
    # (big quad) means the data disagree with the prior, so shrink less.
    lam = (n + 2.0) / quad if quad > 0 else np.inf

    weight = lam / (t + lam) if np.isfinite(lam) else 1.0
    mu_bs = (1.0 - weight) * mu_hat + weight * mu_gmv * ones

    cov_bs = sigma * (1.0 + 1.0 / (t + lam)) if np.isfinite(lam) else sigma
    if np.isfinite(lam):
        cov_bs = cov_bs + (lam / (t * (t + 1.0 + lam))) * np.outer(ones, ones) / denom

    return Belief(mu=mu_bs, cov=_nearest_psd(cov_bs))


def niw_posterior_predictive(
    train_returns,
    kappa_0: float = 1.0,
    nu_extra: float = 2.0,
    prior_corr: float | None = None,
) -> Belief:
    """Posterior predictive moments under a Normal-Inverse-Wishart prior.

    This is the fully Bayesian treatment the project's title always implied:
    place a conjugate prior on ``(mu, Sigma)``, condition on the training window,
    and hand the allocator the *predictive* distribution of next period's return
    rather than a point estimate. The predictive is multivariate-t, so its
    covariance is strictly wider than the plug-in sample covariance -- fat tails
    and parameter uncertainty both make it out, which is exactly the conservatism
    a plug-in optimiser lacks.

    Parameters
    ----------
    kappa_0
        Prior observations backing the mean. ``1.0`` is deliberately weak.
    nu_extra
        Degrees of freedom above the ``N + 1`` minimum for a proper prior.
    prior_corr
        Correlation imposed by the prior scale matrix. ``None`` uses the training
        window's average pairwise correlation, which keeps the estimator a pure
        function of past data.

    Notes
    -----
    The prior is centred on a common mean across assets (the grand mean of the
    training window). That encodes "absent evidence, assets earn the same" --
    the same no-free-lunch stance that makes 1/N hard to beat.
    """
    x = _as_matrix(train_returns)
    t, n = x.shape
    x_bar = x.mean(axis=0)

    # --- Prior, built only from the training window -------------------------
    mu_0 = np.repeat(x_bar.mean(), n)
    var = x.var(axis=0, ddof=1)
    std = np.sqrt(var)
    if prior_corr is None:
        corr = np.corrcoef(x, rowvar=False)
        prior_corr = float((corr.sum() - n) / (n * (n - 1)))
    prior_corr = float(np.clip(prior_corr, -1.0 / (n - 1) + 1e-6, 1.0 - 1e-6))

    scale_0 = prior_corr * np.outer(std, std)
    np.fill_diagonal(scale_0, var)
    nu_0 = n + 1.0 + nu_extra

    # An inverse-Wishart with scale Psi and nu degrees of freedom has mean
    # Psi / (nu - p - 1), so this scaling is what makes the prior *centred* on
    # scale_0 rather than on a multiple of it. Setting psi_0 = scale_0 * nu_0
    # instead -- the easy mistake -- inflates the prior covariance by a factor of
    # nu_0 / nu_extra, which for seven assets is 5x, and the predictive
    # distribution inherits that inflation as spurious conservatism.
    psi_0 = _nearest_psd(scale_0) * (nu_0 - n - 1.0)

    # --- Posterior ----------------------------------------------------------
    kappa_n = kappa_0 + t
    nu_n = nu_0 + t
    mu_n = (kappa_0 * mu_0 + t * x_bar) / kappa_n

    dev = x - x_bar
    scatter = dev.T @ dev
    shift = x_bar - mu_0
    psi_n = psi_0 + scatter + (kappa_0 * t / kappa_n) * np.outer(shift, shift)

    # --- Posterior predictive: multivariate-t with dof = nu_n - n + 1 -------
    dof = nu_n - n + 1.0
    scale = psi_n * (kappa_n + 1.0) / (kappa_n * dof)
    if dof <= 2.0:
        raise ValueError(f"predictive dof={dof:.2f} <= 2; covariance undefined")
    cov = scale * dof / (dof - 2.0)

    return Belief(mu=mu_n, cov=_nearest_psd(cov))


def black_litterman(
    train_returns,
    tau: float = 0.05,
    risk_aversion: float | None = None,
    view_lookback: int = 26,
    view_fraction: float = 0.3,
) -> Belief:
    """Black-Litterman posterior: equilibrium prior blended with a momentum view.

    Reverse-optimisation turns a reference portfolio into the expected returns
    that would make it optimal (``Pi = delta * Sigma * w_ref``); that becomes the
    prior. A cross-sectional momentum view -- recent winners out-earn recent
    losers over the next period -- is then blended in with uncertainty
    proportional to the view's own variance.

    The reference portfolio is equal-weight because this panel carries no market
    capitalisations. That is a documented approximation, not a market portfolio:
    with true caps this would be the CAPM equilibrium, and with 1/N it is instead
    "the returns that would justify holding everything equally".
    """
    x = _as_matrix(train_returns)
    t, n = x.shape
    if view_lookback >= t:
        raise ValueError(f"view_lookback={view_lookback} exceeds window length {t}")

    sigma = ledoit_wolf(x).cov  # shrunk prior covariance; sample cov is too noisy here
    w_ref = np.repeat(1.0 / n, n)

    if risk_aversion is None:
        # delta = market Sharpe / market vol, implied by the reference portfolio.
        ref_returns = x @ w_ref
        ref_var = float(ref_returns.var(ddof=1))
        risk_aversion = float(ref_returns.mean() / ref_var) if ref_var > 0 else 1.0
        risk_aversion = float(np.clip(risk_aversion, 0.5, 10.0))

    pi = risk_aversion * sigma @ w_ref

    # --- Momentum view, formed only from the tail of the training window ----
    recent = x[-view_lookback:]
    momentum = (1.0 + recent).prod(axis=0) - 1.0
    k = max(1, int(round(view_fraction * n)))
    order = np.argsort(momentum)
    losers, winners = order[:k], order[-k:]

    p = np.zeros((1, n))
    p[0, winners] = 1.0 / k
    p[0, losers] = -1.0 / k
    # View magnitude: the realised spread, damped toward zero because momentum
    # decays. Half is a standing assumption, not a fitted parameter.
    q = np.array([0.5 * (momentum[winners].mean() - momentum[losers].mean()) / view_lookback])

    tau_sigma = tau * sigma
    omega = np.diag(np.diag(p @ tau_sigma @ p.T))
    if omega[0, 0] <= 0:
        return Belief(mu=pi, cov=_nearest_psd(sigma))

    tau_sigma_inv = np.linalg.pinv(tau_sigma)
    omega_inv = np.linalg.pinv(omega)

    posterior_prec = tau_sigma_inv + p.T @ omega_inv @ p
    posterior_cov_mu = np.linalg.pinv(posterior_prec)
    mu_bl = posterior_cov_mu @ (tau_sigma_inv @ pi + p.T @ omega_inv @ q)

    # Uncertainty in mu adds to the return covariance the allocator faces.
    return Belief(mu=mu_bl, cov=_nearest_psd(sigma + posterior_cov_mu))


#: Registry used by the experiment scripts so the run is declarative.
ESTIMATORS = {
    "sample": sample_moments,
    "ledoit_wolf": ledoit_wolf,
    "bayes_stein": bayes_stein,
    "niw_predictive": niw_posterior_predictive,
    "black_litterman": black_litterman,
}
