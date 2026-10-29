"""Plotting helpers for ``01_data_exploration.ipynb``.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, PercentFormatter

from data_pipeline import RESULTS_EDA_DIR

PALETTE = sns.color_palette("colorblind")
NEGATIVE_COLOR, POSITIVE_COLOR = PALETTE[0], PALETTE[1]


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
    plt.rcParams.update({"axes.titleweight": "bold", "figure.dpi": 110})


def _finish(fig, filename: str | None, rect=None) -> None:
    fig.tight_layout(rect=rect) if rect else fig.tight_layout()
    if filename:
        RESULTS_EDA_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(RESULTS_EDA_DIR / filename, dpi=200, bbox_inches="tight")
    plt.show()


def plot_class_distribution(y: pd.Series, class_labels, title: str, filename: str | None = None) -> None:
    """Bar chart of class counts annotated with count and percentage."""
    counts = y.value_counts().sort_index()
    labels = [class_labels[int(k)] for k in counts.index]
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=PALETTE[: len(counts)])
    for bar, n in zip(bars, counts.values):
        ax.annotate(
            f"{n:,}\n({n / total:.1%})",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
        )
    ax.set_ylim(0, counts.max() * 1.2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlabel("Target class")
    ax.set_ylabel("Number of records")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    _finish(fig, filename)


def plot_correlation_heatmap(corr: pd.DataFrame, title: str, filename: str | None = None,
                             cbar_label: str = "Spearman ρ") -> None:
    """Lower-triangle heatmap with a diverging colormap centred on zero."""
    n = len(corr)
    side = max(6.0, 0.5 * n + 2.5)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(side * 1.1, side))
    sns.heatmap(
        corr, mask=mask, cmap="vlag", vmin=-1, vmax=1, center=0, square=True,
        annot=n <= 12, fmt=".2f", annot_kws={"size": 9},
        linewidths=0.5 if n <= 12 else 0, cbar_kws={"label": cbar_label, "shrink": 0.8}, ax=ax,
    )
    ax.grid(False)
    ax.set_title(title)
    _finish(fig, filename)


def plot_numeric_by_class(df: pd.DataFrame, columns, target: str, class_labels, title: str,
                          filename: str | None = None,
                          clip_quantiles: tuple[float, float] = (0.005, 0.995)) -> None:
    """Class-conditional histograms (density-normalised per class).

    * The x-range is clipped to ``clip_quantiles`` so extreme outliers do not
      flatten the plot; each panel says how many values fall outside the view.
    * Integer-valued columns with few distinct values are binned per value
      (avoids empty-bin banding); other columns use 40 bins.
    * If a single value holds more than half the rows (zero-inflated counts),
      the y-axis is logarithmic so the rest of the distribution stays visible.
    """
    columns = list(columns)
    ncols = min(3, len(columns))
    nrows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.5 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    labels = df[target].map(lambda v: class_labels[int(v)])
    palette = dict(zip(class_labels, PALETTE[:2]))

    for i, (ax, col) in enumerate(zip(axes_flat, columns)):
        series = df[col]
        low, high = series.quantile(list(clip_quantiles)).to_numpy()
        inside = series.between(low, high)
        plot_df = pd.DataFrame({col: series[inside], "class": labels[inside]})

        integer_valued = bool(np.allclose(series, np.round(series)))
        discrete = integer_valued and series.nunique() <= 100
        bin_kwargs = {"discrete": True} if discrete else {"bins": 40}
        sns.histplot(
            data=plot_df, x=col, hue="class", hue_order=list(class_labels), palette=palette,
            stat="density", common_norm=False, element="step", alpha=0.35,
            legend=False, ax=ax, **bin_kwargs,
        )

        heavy_mode = series.value_counts(normalize=True).iloc[0] > 0.5
        if heavy_mode:
            ax.set_yscale("log")
        note = ", log y-axis" if heavy_mode else ""
        ax.set_title(f"{col}\n({int((~inside).sum()):,} values outside view{note})", fontsize=10)
        ax.set_xlabel(col)
        ax.set_ylabel("Density" if i % ncols == 0 else "")

    for ax in axes_flat[len(columns):]:
        ax.set_visible(False)

    handles = [Patch(facecolor=palette[label], alpha=0.6, label=label) for label in class_labels]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    fig.suptitle(title, fontweight="bold")
    _finish(fig, filename, rect=(0, 0.07, 1, 0.95))


def plot_boxplots_with_bounds(df: pd.DataFrame, range_rules: dict, title: str,
                              filename: str | None = None) -> None:
    """Raw-scale boxplots (symlog x-axis) with the validity bounds overlaid."""
    columns = list(range_rules)
    fig, axes = plt.subplots(len(columns), 1, figsize=(8, 1.6 * len(columns) + 0.8), squeeze=False)

    for ax, col in zip(axes.ravel(), columns):
        low, high = range_rules[col]
        sns.boxplot(x=df[col], ax=ax, color=PALETTE[0], fliersize=2, linewidth=1)
        ax.axvline(low, color="crimson", ls="--", lw=1.2, label="validity bound")
        ax.axvline(high, color="crimson", ls="--", lw=1.2)
        ax.set_xscale("symlog", linthresh=10)
        n_out = int((~df[col].between(low, high)).sum())
        ax.set_title(f"{col}: {n_out:,} rows outside [{low:g}, {high:g}]", loc="left", fontsize=10)
        ax.set_xlabel(f"{col} (symlog scale)")
        ax.set_yticks([])
    axes[0, 0].legend(loc="upper right")
    fig.suptitle(title, fontweight="bold")
    _finish(fig, filename)


def plot_target_association(assoc: pd.Series, title: str, filename: str | None = None,
                            xlabel: str = "Spearman ρ with target") -> None:
    """Horizontal bars of signed feature–target association."""
    assoc = assoc.sort_values()
    colors = [POSITIVE_COLOR if v > 0 else NEGATIVE_COLOR for v in assoc.values]

    fig, ax = plt.subplots(figsize=(7, 0.32 * len(assoc) + 1.6))
    ax.barh(assoc.index, assoc.values, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    for i, value in enumerate(assoc.values):
        ax.text(value + (0.005 if value >= 0 else -0.005), i, f"{value:+.2f}",
                va="center", ha="left" if value >= 0 else "right", fontsize=8)
    ax.margins(x=0.15)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    _finish(fig, filename)


def plot_target_rate_by_level(df: pd.DataFrame, columns, target: str, title: str,
                              filename: str | None = None, overall: float | None = None) -> None:
    """Positive-class rate for each level of each feature, vs the overall rate."""
    columns = list(columns)
    ncols = min(4, len(columns))
    nrows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.0 * nrows),
                             sharey=True, squeeze=False)
    axes_flat = axes.ravel()

    for i, (ax, col) in enumerate(zip(axes_flat, columns)):
        rate = df.groupby(col)[target].mean()
        ax.bar(rate.index.astype(str), rate.values, color=PALETTE[0])
        if overall is not None:
            ax.axhline(overall, color="gray", ls="--", lw=1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("level")
        ax.set_ylabel("Positive-class rate" if i % ncols == 0 else "")
        ax.grid(axis="x", visible=False)
    for ax in axes_flat[len(columns):]:
        ax.set_visible(False)
    if overall is not None:
        fig.legend(handles=[Line2D([0], [0], color="gray", ls="--", label=f"overall rate ({overall:.1%})")],
                   loc="lower center", frameon=False)
    fig.suptitle(title, fontweight="bold")
    _finish(fig, filename, rect=(0, 0.06, 1, 0.95))
