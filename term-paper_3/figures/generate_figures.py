"""Generate thesis figures for term-paper_3.

The script is intentionally self-contained and path-relative so the figures can
be regenerated from any checkout of the repository.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "EXP-007"
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

DPI = 300

COLORS = {
    "S1": "#4E79A7",
    "S2+R": "#F28E2B",
    "S3+R": "#59A14F",
    "S7": "#E15759",
    "S2": "#9C755F",
    "S3": "#B07AA1",
    "S3-legacy": "#BAB0AC",
}

HEADLINE = ["S1", "S2+R", "S3+R", "S7"]
ACTIVE = ["S1", "S2+R", "S3+R", "S7", "S2", "S3", "S3-legacy"]
DISPLAY = {
    "S1": "Base-RAG",
    "S2+R": "RAFT-RAG",
    "S3+R": "CLM-RAG",
    "S7": "Merge-RAG",
    "S2": "RAFT-Closed",
    "S3": "CLM-Closed",
    "S3-legacy": "D2L-Closed",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
    }
)


def clean_axes(ax, grid_axis: str | None = "y") -> None:
    """Apply a restrained publication-style axis treatment."""

    if grid_axis:
        ax.grid(axis=grid_axis, color="#E3E3E3", linewidth=0.8)
        ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


def system_row(
    ax,
    y: float,
    label: str,
    adapter: str,
    color: str,
    *,
    posthoc: bool = False,
) -> None:
    box = mpatches.FancyBboxPatch(
        (7.25, y - 0.30),
        2.25,
        0.60,
        boxstyle="round,pad=0.08,rounding_size=0.08",
        facecolor=color,
        edgecolor="#333333",
        linewidth=1.0,
        alpha=0.93,
        hatch="//" if posthoc else None,
    )
    ax.add_patch(box)
    ax.text(
        8.38,
        y + 0.08,
        label,
        ha="center",
        va="center",
        color="white",
        fontsize=8.9,
        fontweight="bold",
    )
    ax.text(
        8.38,
        y - 0.13,
        adapter,
        ha="center",
        va="center",
        color="white",
        fontsize=6.9,
    )
    ax.annotate(
        "",
        xy=(7.12, y),
        xytext=(6.58, y),
        arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#555555"),
    )
    ax.annotate(
        "",
        xy=(10.35, y),
        xytext=(9.62, y),
        arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#555555"),
    )
    ax.text(10.48, y, "answer", ha="left", va="center", fontsize=8.0, color="#555555")


def fig01_system_schematic() -> None:
    """System overview with fixed retrieval internals made explicit."""

    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    ax.set_xlim(0, 11.3)
    ax.set_ylim(0, 5.8)
    ax.axis("off")

    # Query and fixed retrieval stack.
    ax.text(0.35, 4.75, "Question", fontsize=9.0, fontweight="bold", ha="left")
    ax.annotate(
        "",
        xy=(1.35, 4.55),
        xytext=(0.85, 4.55),
        arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#555555"),
    )

    retrieval = mpatches.FancyBboxPatch(
        (1.35, 2.55),
        5.05,
        2.65,
        boxstyle="round,pad=0.12,rounding_size=0.10",
        facecolor="#F6F7F9",
        edgecolor="#4D4D4D",
        linewidth=1.0,
    )
    ax.add_patch(retrieval)
    ax.text(
        3.88,
        5.00,
        "Fixed retrieval backbone",
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        color="#333333",
    )

    stages = ["chunk", "hybrid", "RRF", "rerank", "top-3"]
    xs = np.linspace(1.95, 5.80, len(stages))
    for i, (top, x) in enumerate(zip(stages, xs)):
        stage = mpatches.FancyBboxPatch(
            (x - 0.41, 3.55),
            0.82,
            0.74,
            boxstyle="round,pad=0.06,rounding_size=0.06",
            facecolor="#FFFFFF",
            edgecolor="#8E8E8E",
            linewidth=0.9,
        )
        ax.add_patch(stage)
        ax.text(x, 3.88, top, ha="center", va="center", fontsize=7.3, fontweight="bold")
        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.44, 3.92),
                xytext=(x + 0.44, 3.92),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#777777"),
            )

    ax.text(
        3.88,
        3.35,
        "chunks -> dense/BM25 -> RRF -> rerank -> top-3 evidence",
        ha="center",
        va="top",
        fontsize=6.5,
        color="#555555",
    )

    ax.text(
        3.88,
        2.95,
        "Same evidence path for all retrieval-aware systems",
        ha="center",
        va="center",
        fontsize=7.8,
        color="#555555",
    )

    ax.annotate(
        "",
        xy=(7.1, 3.85),
        xytext=(6.40, 3.85),
        arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#555555"),
    )

    system_row(ax, 4.55, "Base-RAG", "base generator", COLORS["S1"])
    system_row(ax, 3.75, "RAFT-RAG", "RAFT adapter", COLORS["S2+R"])
    system_row(ax, 2.95, "CLM-RAG", "CLM adapter", COLORS["S3+R"])
    system_row(ax, 2.15, "Merge-RAG", "merged adapter", COLORS["S7"], posthoc=True)

    # No-retrieval controls.
    ax.text(0.35, 1.10, "Question", fontsize=8.0, ha="left", color="#666666")
    ax.annotate(
        "",
        xy=(7.0, 1.10),
        xytext=(1.15, 1.10),
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#888888"),
    )
    ctrl = mpatches.FancyBboxPatch(
        (7.25, 0.78),
        2.25,
        0.64,
        boxstyle="round,pad=0.08,rounding_size=0.08",
        facecolor="#D6D6D6",
        edgecolor="#777777",
        linewidth=0.9,
    )
    ax.add_patch(ctrl)
    ax.text(
        8.38,
        1.17,
        "Closed controls",
        ha="center",
        va="center",
        fontsize=7.9,
        fontweight="bold",
    )
    ax.text(
        8.38,
        0.95,
        "RAFT / CLM / D2L",
        ha="center",
        va="center",
        fontsize=6.8,
        color="#555555",
    )
    ax.annotate(
        "",
        xy=(10.35, 1.10),
        xytext=(9.62, 1.10),
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#888888"),
    )
    ax.text(10.48, 1.10, "answer", ha="left", va="center", fontsize=8.0, color="#777777")

    ax.text(8.38, 5.28, "Retrieval-aware systems", ha="center", fontsize=8.2, color="#333333")
    ax.text(8.38, 0.42, "Retrieval-free controls", ha="center", fontsize=8.2, color="#777777")

    save(fig, "fig01_system_schematic.png")


def fig02_delta_bars() -> None:
    """Delta-to-baseline view of the headline adaptation trade-off."""

    df = pd.read_csv(DATA / "consolidated_results.csv").set_index("system")
    systems = ["S2+R", "S3+R", "S7"]
    metrics = [("q_main", "Q_main"), ("s_det", "S_det"), ("s_asst", "S_asst")]

    records = []
    for system in systems:
        for metric, label in metrics:
            records.append(
                {
                    "system": system,
                    "metric": label,
                    "delta": df.loc[system, metric] - df.loc["S1", metric],
                }
            )
    plot_df = pd.DataFrame.from_records(records)

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    y = np.arange(len(metrics))
    offsets = {"S2+R": 0.22, "S3+R": 0.0, "S7": -0.22}
    height = 0.18

    for system in systems:
        vals = [
            plot_df[(plot_df["system"] == system) & (plot_df["metric"] == label)]["delta"].iloc[0]
            for _, label in metrics
        ]
        bars = ax.barh(
            y + offsets[system],
            vals,
            height,
            label=DISPLAY[system],
            color=COLORS[system],
            edgecolor="white",
            linewidth=0.5,
            hatch="//" if system == "S7" else None,
            alpha=0.92,
        )
        for bar, val in zip(bars, vals):
            x = val + (0.004 if val >= 0 else -0.004)
            ha = "left" if val >= 0 else "right"
            ax.text(
                x,
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}",
                ha=ha,
                va="center",
                fontsize=7.7,
                color="#222222",
            )

    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([label for _, label in metrics])
    ax.set_xlabel("Change relative to Base-RAG")
    ax.set_title("Where Adaptation Improves Over the Fixed RAG Baseline")
    ax.set_xlim(-0.035, 0.100)
    ax.invert_yaxis()
    clean_axes(ax, "x")
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        handlelength=1.7,
        columnspacing=1.6,
    )

    save(fig, "fig02_delta_bars.png")


def fig03_judge_criteria() -> None:
    """Free-text judge profile as a line plot instead of crowded bars."""

    df = pd.read_csv(DATA / "judge_criteria_profile.csv")
    criteria = ["correctness", "completeness", "grounding", "calibration", "clarity"]
    labels = ["Correct.", "Complete.", "Grounded", "Calibr.", "Clear"]
    x = np.arange(len(criteria))

    fig, ax = plt.subplots(figsize=(6.7, 3.6))
    for system in HEADLINE:
        vals = (
            df[df["system"] == system]
            .set_index("criterion")
            .loc[criteria, "mean_score"]
            .to_numpy()
        )
        ax.plot(
            x,
            vals,
            marker="o",
            markersize=5.0,
            linewidth=1.8 if system == "S3+R" else 1.3,
            color=COLORS[system],
            label=DISPLAY[system],
            alpha=0.95,
            linestyle="--" if system == "S7" else "-",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean judge score")
    ax.set_ylim(0.45, 1.03)
    ax.set_title("Free-Text Judge Profile")
    clean_axes(ax, "y")
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.30))
    ax.text(
        2.15,
        0.97,
        "CLM adapter leads\non judged prose",
        fontsize=7.6,
        color=COLORS["S3+R"],
        ha="left",
        va="top",
    )

    save(fig, "fig03_judge_criteria.png")


def fig04_per_type_heatmap() -> None:
    """Raw per-type scores, with column winners outlined."""

    df = pd.read_csv(DATA / "per_type_breakdown.csv")
    type_n = {
        "boolean": 12,
        "number": 7,
        "name": 8,
        "names": 5,
        "date": 5,
        "free_text": 13,
    }
    types_order = ["boolean", "number", "name", "names", "date", "free_text"]
    systems = HEADLINE

    matrix = np.array(
        [
            [
                df[(df["system"] == system) & (df["answer_type"] == answer_type)][
                    "score"
                ].iloc[0]
                for answer_type in types_order
            ]
            for system in systems
        ]
    )

    fig, ax = plt.subplots(figsize=(6.9, 3.2))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(len(types_order)))
    ax.set_xticklabels([f"{t.replace('_', '-')}\n(n={type_n[t]})" for t in types_order])
    ax.set_yticks(np.arange(len(systems)))
    ax.set_yticklabels([DISPLAY[system] for system in systems], fontweight="bold")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if val >= 0.68 else "#222222"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7.8, color=color)

    for j in range(matrix.shape[1]):
        best = np.argwhere(np.isclose(matrix[:, j], matrix[:, j].max())).flatten()
        for i in best:
            ax.add_patch(
                mpatches.Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="#222222",
                    linewidth=1.5,
                )
            )

    ax.set_title("Per-Type Score Profile")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, shrink=0.84, pad=0.02)
    cbar.set_label("Score", rotation=270, labelpad=12)
    cbar.ax.tick_params(labelsize=8)

    ax.text(
        -0.48,
        4.08,
        "outlined cells mark the best system per answer type",
        fontsize=7.3,
        color="#666666",
        ha="left",
    )

    save(fig, "fig04_per_type_heatmap.png")


def fig05_singledoc_multidoc() -> None:
    """Single vs multi-document performance as a gap/slope chart."""

    df = pd.read_csv(DATA / "question_score_summary.csv")
    df = df[df["system"].isin(HEADLINE)]
    groups = df.groupby(["system", "is_multi_doc"])["score_mean"].mean()

    single = {system: groups.loc[(system, False)] for system in HEADLINE}
    multi = {system: groups.loc[(system, True)] for system in HEADLINE}

    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    x = np.array([0, 1])

    left_offsets = {"S1": -0.010, "S2+R": -0.026, "S3+R": 0.026, "S7": 0.010}
    right_offsets = {"S1": -0.015, "S2+R": 0.000, "S3+R": 0.015, "S7": 0.000}

    for system in HEADLINE:
        vals = [single[system], multi[system]]
        ax.plot(
            x,
            vals,
            marker="o",
            markersize=5.5,
            linewidth=1.9,
            color=COLORS[system],
            label=system,
            linestyle="--" if system == "S7" else "-",
        )
        ax.text(
            -0.05,
            vals[0] + left_offsets[system],
            f"{DISPLAY[system]}  {vals[0]:.3f}",
            ha="right",
            va="center",
            fontsize=7.5,
            color=COLORS[system],
            fontweight="bold" if system == "S7" else "normal",
        )
        ax.text(
            1.05,
            vals[1] + right_offsets[system],
            f"{vals[1]:.3f}",
            ha="left",
            va="center",
            fontsize=7.5,
            color=COLORS[system],
            fontweight="bold" if system == "S7" else "normal",
        )

    ax.set_xlim(-0.35, 1.25)
    ax.set_ylim(0.25, 0.78)
    ax.set_xticks(x)
    ax.set_xticklabels(["Single-document\n(n=42)", "Multi-document\n(n=8)"])
    ax.set_ylabel("Q_main")
    ax.set_title("Multi-Document Questions Expose the Largest Gap")
    clean_axes(ax, "y")

    ax.text(
        0.52,
        0.66,
        "RAFT-RAG and Merge-RAG retain\nmore multi-doc quality",
        ha="center",
        va="center",
        fontsize=7.7,
        color="#444444",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor="#DDDDDD"),
    )

    save(fig, "fig05_singledoc_multidoc.png")


if __name__ == "__main__":
    print(f"Generating figures from {DATA}")
    missing = [path for path in [DATA / "consolidated_results.csv"] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected input files: {missing}")
    fig01_system_schematic()
    fig02_delta_bars()
    fig03_judge_criteria()
    fig04_per_type_heatmap()
    fig05_singledoc_multidoc()
    print(f"All figures saved to {OUT}")
