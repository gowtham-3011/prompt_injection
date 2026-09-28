"""Plots for the evaluation report.

Chart rules followed here:
  * One measure per axis, never a second y-scale.
  * Proportions are shown with their Wilson confidence intervals. A bar without an
    interval invites over-reading a difference that the sample size cannot support.
  * Categorical colors are assigned in a fixed order, never cycled, and the two-series
    pair used here is validated for colour-vision deficiency and contrast.
  * The ROC uses a LOG x-axis: the operating region for a firewall is FPR <= 1%, and a
    linear axis compresses that whole region into the leftmost pixel column.
  * Values are labelled directly where it aids reading, rather than on every mark.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: scripts must work without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Validated categorical slots (light surface): blue, orange.
SERIES = ["#2a78d6", "#eb6834"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#dcdbd6"


def _style(ax) -> None:
    """Recessive axes and grid: the data should be the most prominent thing."""
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_SOFT)
    ax.yaxis.label.set_color(INK_SOFT)


def asr_before_after(
    categories: list[str],
    before: list[tuple[float, float, float]],   # (rate, ci_low, ci_high)
    after: list[tuple[float, float, float]],
    path: Path,
    before_label: str = "no firewall",
    after_label: str = "firewall on",
    title: str = "Attack success rate by category",
) -> Path:
    """Grouped bars: ASR per category, undefended vs defended, with CIs."""
    x = np.arange(len(categories))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.6))

    for offset, data, color, label in (
        (-width / 2, before, SERIES[0], before_label),
        (width / 2, after, SERIES[1], after_label),
    ):
        rates = [d[0] for d in data]
        lower = [max(0.0, d[0] - d[1]) for d in data]
        upper = [max(0.0, d[2] - d[0]) for d in data]
        ax.bar(x + offset, rates, width, label=label, color=color, edgecolor=SURFACE,
               linewidth=2, zorder=3)
        ax.errorbar(x + offset, rates, yerr=[lower, upper], fmt="none",
                    ecolor=INK_SOFT, elinewidth=1.2, capsize=3, zorder=4)
        # Label above the interval's upper bound, so the number never sits on a whisker.
        for xi, rate, point in zip(x + offset, rates, data):
            ax.text(xi, point[2] + 0.02, f"{rate:.0%}", ha="center", va="bottom",
                    fontsize=8, color=INK_SOFT)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=9)
    ax.set_ylabel("attack success rate")
    ax.set_ylim(0, 1.18)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0, 1.01, 0.25)])
    ax.set_title(title, fontsize=12, pad=14, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, loc="upper right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def ablation_by_layer(
    configs: list[str],
    asr: list[tuple[float, float, float]],
    benign: list[tuple[float, float, float]],
    path: Path,
    title: str = "Ablation: attack success vs benign task success",
) -> Path:
    """Two bars per configuration. ASR alone would hide a firewall that breaks the app."""
    y = np.arange(len(configs))
    height = 0.36
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(configs) + 2.2))

    for offset, data, color, label in (
        (height / 2, asr, SERIES[0], "attack success rate (lower is better)"),
        (-height / 2, benign, SERIES[1], "benign task success (higher is better)"),
    ):
        rates = [d[0] for d in data]
        lower = [max(0.0, d[0] - d[1]) for d in data]
        upper = [max(0.0, d[2] - d[0]) for d in data]
        ax.barh(y + offset, rates, height, label=label, color=color, edgecolor=SURFACE,
                linewidth=2, zorder=3)
        ax.errorbar(rates, y + offset, xerr=[lower, upper], fmt="none",
                    ecolor=INK_SOFT, elinewidth=1.2, capsize=3, zorder=4)
        # Label past the interval's upper bound, so the number never sits on a whisker.
        for yi, rate, point in zip(y + offset, rates, data):
            ax.text(point[2] + 0.02, yi, f"{rate:.0%}", va="center", fontsize=8, color=INK_SOFT)

    ax.set_yticks(y)
    ax.set_yticklabels(configs, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.22)
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_xticklabels([f"{v:.0%}" for v in np.arange(0, 1.01, 0.25)])
    ax.set_title(title, fontsize=12, pad=14, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, loc="lower right")
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.7)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def roc_log_fpr(
    curves: dict[str, tuple[list[float], list[float]]],  # name -> (fpr, tpr)
    path: Path,
    operating_points: dict[str, tuple[float, float]] | None = None,
    title: str = "Detector ROC (log-scale FPR)",
) -> Path:
    """ROC on a log FPR axis, which is where a firewall actually operates."""
    fig, ax = plt.subplots(figsize=(6.4, 5))
    for i, (name, (fpr, tpr)) in enumerate(curves.items()):
        # A zero FPR cannot be drawn on a log axis; clamp to half the smallest
        # representable rate so the curve starts at the left edge honestly.
        floor = 1e-4
        fpr_plot = [max(f, floor) for f in fpr]
        ax.plot(fpr_plot, tpr, linewidth=2, color=SERIES[i % len(SERIES)], label=name, zorder=3)
    if operating_points:
        for i, (name, (fpr, tpr)) in enumerate(operating_points.items()):
            ax.scatter([max(fpr, 1e-4)], [tpr], s=64, color=SERIES[i % len(SERIES)],
                       edgecolor=SURFACE, linewidth=2, zorder=5)
            ax.annotate(f"chosen operating point\n({fpr:.1%} FPR, {tpr:.0%} TPR)",
                        (max(fpr, 1e-4), tpr), xytext=(14, -26), textcoords="offset points",
                        fontsize=8, color=INK_SOFT, va="top")
    ax.axvline(0.01, color=GRID, linestyle="--", linewidth=1.2, zorder=2)
    ax.text(0.01, 1.04, "1% FPR budget", fontsize=8, color=INK_SOFT, ha="center")
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("false positive rate (log scale)")
    ax.set_ylabel("true positive rate")
    ax.set_title(title, fontsize=12, pad=14, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, loc="lower right")
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def latency_distribution(
    samples: dict[str, list[float]],
    path: Path,
    title: str = "Per-request latency added by the firewall",
) -> Path:
    """Horizontal box plot: distribution, not just a mean, since tail latency is the risk."""
    names = list(samples)
    data = [samples[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 0.7 * len(names) + 2))
    box = ax.boxplot(data, vert=False, patch_artist=True, widths=0.5,
                     medianprops={"color": INK, "linewidth": 1.6},
                     whiskerprops={"color": INK_SOFT}, capprops={"color": INK_SOFT},
                     flierprops={"markeredgecolor": INK_SOFT, "marker": ".", "markersize": 4})
    # One color for every box: the y-axis labels already carry identity, so a second hue
    # here would imply a grouping that does not exist.
    for patch in box["boxes"]:
        patch.set_facecolor(SERIES[0])
        patch.set_alpha(0.85)
        patch.set_edgecolor(SURFACE)
        patch.set_linewidth(2)
    for i, values in enumerate(data, start=1):
        if values:
            p95 = sorted(values)[int(0.95 * (len(values) - 1))]
            ax.text(p95, i + 0.32, f"p95 {p95:.1f} ms", fontsize=8, color=INK_SOFT)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("milliseconds")
    ax.set_title(title, fontsize=12, pad=14, loc="left")
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.7)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def roc_points(scores: list[float], labels: list[int]) -> tuple[list[float], list[float]]:
    """Compute an ROC curve without sklearn, so plots work with the light install."""
    pairs = sorted(zip(scores, labels), key=lambda p: -p[0])
    n_pos = sum(labels) or 1
    n_neg = len(labels) - sum(labels) or 1
    tp = fp = 0
    fpr, tpr = [0.0], [0.0]
    for score, label in pairs:
        tp += label
        fp += 1 - label
        fpr.append(fp / n_neg)
        tpr.append(tp / n_pos)
    return fpr, tpr
