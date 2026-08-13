"""Figures for the walk-forward study.

The design rule throughout: the reader's eye should land on the *uncertainty*,
not on the ranking. A bar chart of Sharpe ratios would imply the ordering is
meaningful; the interval plot below shows that it mostly is not, which is the
actual finding.

Colour is used sparingly -- three hues carry identity (benchmark, best optimised
strategy, everything else is recessive grey), because a sixteen-colour legend
communicates nothing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Validated categorical slots plus chrome/ink roles. The first three clear the
# all-pairs separation gates and carry identity in the line/interval charts; the
# full eight are used only where the pairlist is adjacent (the stacked weights
# chart), which is the case they were ordered to satisfy.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CATEGORICAL = [BLUE, ORANGE, AQUA, "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
RECESSIVE = "#c9c8c2"

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


def _style() -> None:
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "600",
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.dpi": 130,
        "savefig.bbox": "tight",
    })


def _despine(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig, path: Path) -> None:
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print(f"  figure -> {path.name}")


def plot_equity_curves(results, metrics, outdir: Path, benchmark: str) -> None:
    """Growth of $1 out of sample, log scale so equal ratios look equal."""
    best = metrics.index[0]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    for name, result in results.items():
        if name in (benchmark, best):
            continue
        ax.plot(result.equity_curve, color=RECESSIVE, linewidth=1.0, zorder=1)

    for name, colour in ((benchmark, BLUE), (best, ORANGE)):
        curve = results[name].equity_curve
        ax.plot(curve, color=colour, linewidth=2.0, zorder=3, solid_capstyle="round")
        ax.annotate(
            f"  {name}  ({curve.iloc[-1]:.2f}x)",
            xy=(curve.index[-1], curve.iloc[-1]),
            color=colour, fontsize=9, fontweight="600", va="center",
        )

    ax.plot([], [], color=RECESSIVE, linewidth=1.0,
            label=f"other optimised strategies (n={len(results) - 2})")
    ax.set_yscale("log")
    ax.set_yticks([1, 1.5, 2, 3, 4])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylabel("growth of $1 (log scale)")
    ax.set_title("Out-of-sample equity curves, net of transaction costs")
    ax.margins(x=0.14)
    ax.legend(loc="upper left")
    _despine(ax)
    _save(fig, outdir / "equity_curves.png")


def plot_sharpe_intervals(inference, outdir: Path, benchmark: str) -> None:
    """The headline figure: Sharpe point estimates with 95% bootstrap intervals.

    Whether the intervals overlap the benchmark's is the entire result. Ranking
    strategies without these bars is how a 0.3 Sharpe gap gets reported as a
    35% improvement.
    """
    frame = inference.sort_values("sharpe")
    y = np.arange(len(frame))

    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(frame) + 2.0))

    bench_sharpe = float(inference.loc[benchmark, "sharpe"])
    ax.axvline(bench_sharpe, color=BLUE, linewidth=1.5, linestyle="--", zorder=1,
               label=f"1/N benchmark ({bench_sharpe:.2f})")

    for i, (name, row) in enumerate(frame.iterrows()):
        colour = BLUE if name == benchmark else (ORANGE if i == len(frame) - 1 else INK_MUTED)
        ax.plot([row["sharpe_ci_low"], row["sharpe_ci_high"]], [i, i],
                color=colour, linewidth=2.0, solid_capstyle="round", zorder=2)
        ax.scatter([row["sharpe"]], [i], color=colour, s=42, zorder=3,
                   edgecolor=SURFACE, linewidth=1.2)

    ax.set_yticks(y)
    ax.set_yticklabels(frame.index, fontsize=9)
    for tick, name in zip(ax.get_yticklabels(), frame.index):
        if name == benchmark:
            tick.set_color(BLUE)
            tick.set_fontweight("600")

    ax.set_yticks(y)
    ax.set_xlabel("annualised Sharpe ratio (95% stationary-bootstrap interval)")

    # Derive the headline from the data rather than asserting it: if a future
    # run does separate a strategy from the benchmark, the title must say so.
    spans = (
        (frame["sharpe_ci_low"] <= bench_sharpe) & (frame["sharpe_ci_high"] >= bench_sharpe)
    )
    others = spans.drop(index=benchmark, errors="ignore")
    if others.all():
        title = "Every strategy's Sharpe interval contains the benchmark's estimate"
    else:
        title = (
            f"{(~others).sum()} of {len(others)} strategies have intervals "
            f"excluding the benchmark"
        )
    ax.set_title(title)

    # Leave room for the interval ends; the default limits clip them at the spine.
    ax.set_xlim(frame["sharpe_ci_low"].min() - 0.08, frame["sharpe_ci_high"].max() + 0.08)
    ax.set_ylim(-0.8, len(frame) - 0.2)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    _despine(ax)
    _save(fig, outdir / "sharpe_intervals.png")


def plot_estimator_optimiser_grid(metrics, outdir: Path, benchmark: str) -> None:
    """The factorial view: does moving down (estimator) or across (optimiser) matter more?"""
    split = [name.split("+") for name in metrics.index if "+" in name]
    estimators = sorted({e for e, _ in split})
    allocators = ["min_variance", "convex", "gp_bayesopt"]

    grid = pd.DataFrame(index=estimators, columns=allocators, dtype=float)
    for est in estimators:
        for alloc in allocators:
            key = f"{est}+{alloc}"
            if key in metrics.index:
                grid.loc[est, alloc] = metrics.loc[key, "sharpe"]

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    image = ax.imshow(grid.to_numpy(dtype=float), cmap=SEQUENTIAL, aspect="auto")

    for i in range(len(estimators)):
        for j in range(len(allocators)):
            value = grid.iloc[i, j]
            if np.isfinite(value):
                # Ink flips on dark cells so the label always clears contrast.
                normalised = (value - np.nanmin(grid.to_numpy(dtype=float))) / (
                    np.nanmax(grid.to_numpy(dtype=float)) - np.nanmin(grid.to_numpy(dtype=float)) + 1e-12
                )
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=10,
                        fontweight="600", color="#ffffff" if normalised > 0.55 else INK)

    ax.set_xticks(range(len(allocators)), allocators)
    ax.set_yticks(range(len(estimators)), estimators)
    ax.set_title("Out-of-sample Sharpe by belief (rows) and optimiser (columns)")
    ax.grid(visible=False)

    bench = metrics.loc[benchmark, "sharpe"]
    ax.set_xlabel(f"1/N benchmark = {bench:.3f}", color=INK_SECONDARY)
    fig.colorbar(image, ax=ax, label="Sharpe", fraction=0.046, pad=0.03)
    _save(fig, outdir / "estimator_optimiser_grid.png")


def plot_rolling_sharpe(results, metrics, outdir: Path, benchmark: str, window: int = 104) -> None:
    """Two-year rolling Sharpe: is any edge persistent, or concentrated in one stretch?"""
    best = metrics.index[0]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))

    for name, colour in ((benchmark, BLUE), (best, ORANGE)):
        r = results[name].net_returns
        rolling = r.rolling(window).mean() / r.rolling(window).std(ddof=1) * np.sqrt(52)
        ax.plot(rolling.dropna(), color=colour, linewidth=2.0, label=name,
                solid_capstyle="round")

    ax.axhline(0.0, color=BASELINE, linewidth=1.0)
    ax.set_ylabel(f"rolling {window // 52}-year Sharpe")
    ax.set_title("Rolling Sharpe: the ranking flips repeatedly")
    ax.legend(loc="best")
    _despine(ax)
    _save(fig, outdir / "rolling_sharpe.png")


def plot_weights(results, metrics, outdir: Path) -> None:
    """Allocation of the best strategy over time -- shows concentration and churn."""
    best = metrics.index[0]
    weights = results[best].weights
    fig, ax = plt.subplots(figsize=(9.5, 4.4))

    # Assets are categorical identity, not magnitude, so they take the fixed
    # categorical hue order rather than a sequential ramp. The order is the
    # CVD-safety mechanism -- it clears the adjacent-pair separation gates, which
    # is the relevant pairlist for stacked bands. The surface-coloured edge keeps
    # a visible gap between neighbouring segments.
    colours = [CATEGORICAL[i % len(CATEGORICAL)] for i in range(weights.shape[1])]
    ax.stackplot(weights.index, weights.to_numpy().T, labels=list(weights.columns),
                 colors=colours, edgecolor=SURFACE, linewidth=1.2)

    ax.set_ylim(0, 1)
    ax.set_ylabel("portfolio weight")
    ax.set_title(f"Allocation through time: {best}")
    ax.legend(loc="upper center", ncol=min(7, weights.shape[1]), bbox_to_anchor=(0.5, -0.08))
    ax.grid(visible=False)
    _despine(ax)
    _save(fig, outdir / "weights_best.png")


def make_all_figures(results, metrics, inference, outdir: Path, benchmark: str) -> None:
    _style()
    outdir.mkdir(parents=True, exist_ok=True)
    print("\ngenerating figures ...")
    plot_equity_curves(results, metrics, outdir, benchmark)
    plot_sharpe_intervals(inference, outdir, benchmark)
    plot_estimator_optimiser_grid(metrics, outdir, benchmark)
    plot_rolling_sharpe(results, metrics, outdir, benchmark)
    plot_weights(results, metrics, outdir)
