"""Generate thesis figures for term-paper_3.

The script is intentionally self-contained and path-relative so the figures can
be regenerated from any checkout of the repository.
"""

from __future__ import annotations

import json
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
DOC_SCOPE = ROOT / "results" / "EXP-006" / "single_vs_multi_doc.csv"
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

SEED_QMAIN_SOURCES = {
    "S2+R": ROOT / "results" / "EXP-003" / "aggregate_summary.json",
    "S3+R": ROOT / "results" / "EXP-004b" / "aggregate_summary.json",
    "S7": ROOT / "results" / "EXP-010" / "alpha_0.5" / "aggregate_summary.json",
    "S2": ROOT / "results" / "EXP-003b" / "aggregate_summary.json",
    "S3": ROOT / "results" / "EXP-004_clm" / "aggregate_summary.json",
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


def load_seed_qmain() -> pd.DataFrame:
    """Load per-seed Q_main values from the experiment aggregate summaries."""

    rows = []
    for system, path in SEED_QMAIN_SOURCES.items():
        data = json.loads(path.read_text())
        for item in data["seed_results"]:
            rows.append(
                {
                    "system": system,
                    "seed": int(item["seed"]),
                    "q_main": float(item["q_main"]),
                }
            )
    return pd.DataFrame.from_records(rows)


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
    """Free-text judge profile as grouped bars."""

    df = pd.read_csv(DATA / "judge_criteria_profile.csv")
    criteria = ["correctness", "completeness", "grounding", "calibration", "clarity"]
    labels = ["Correctness", "Completeness", "Grounding", "Calibration", "Clarity"]
    x = np.arange(len(criteria))
    width = 0.18

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for i, system in enumerate(HEADLINE):
        vals = (
            df[df["system"] == system]
            .set_index("criterion")
            .loc[criteria, "mean_score"]
            .to_numpy()
        )
        offset = (i - (len(HEADLINE) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            color=COLORS[system],
            label=DISPLAY[system],
            alpha=0.92,
            edgecolor="#333333",
            linewidth=0.35,
            hatch="//" if system == "S7" else None,
        )
        if system == "S3+R":
            for bar in bars:
                bar.set_linewidth(0.65)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean judge score")
    ax.set_ylim(0.45, 1.03)
    ax.set_title("Free-Text Judge Profile")
    clean_axes(ax, "y")
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.30))
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

    save(fig, "fig04_per_type_heatmap.png")


def fig05_singledoc_multidoc() -> None:
    """Single vs multi-document performance as a gap/slope chart."""

    df = pd.read_csv(DOC_SCOPE).set_index("system")
    single = {system: df.loc[system, "single_doc_q_main"] for system in HEADLINE}
    multi = {system: df.loc[system, "multi_doc_q_main"] for system in HEADLINE}

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


def figB1_error_overlap() -> None:
    """Failure-overlap Jaccard matrix for headline systems."""

    df = pd.read_csv(DATA / "error_overlap_jaccard.csv", index_col=0)
    systems = HEADLINE
    matrix = df.loc[systems, systems].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    im = ax.imshow(matrix, cmap="magma", vmin=0.40, vmax=1.00)

    ax.set_xticks(np.arange(len(systems)))
    ax.set_yticks(np.arange(len(systems)))
    ax.set_xticklabels([DISPLAY[system] for system in systems], rotation=25, ha="right")
    ax.set_yticklabels([DISPLAY[system] for system in systems])
    ax.set_xlabel("System B")
    ax.set_ylabel("System A")
    ax.set_title("Failure-Overlap Jaccard Among Headline Systems")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            rgba = im.cmap(im.norm(val))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "#222222" if luminance > 0.58 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8.2, color=color)

    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.03)
    cbar.set_label("Jaccard over missed questions", rotation=270, labelpad=16)
    cbar.ax.tick_params(labelsize=8)

    ax.text(
        0.0,
        -0.24,
        "Higher values mean that two systems fail on more of the same questions.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        color="#555555",
    )

    save(fig, "figB1_error_overlap_heatmap.png")


def figB2_seed_stability() -> None:
    """Grouped per-seed Q_main bars for trained systems."""

    seed_df = load_seed_qmain()
    systems = ["S2+R", "S3+R", "S7", "S2", "S3"]
    seeds = [42, 123, 777]
    x = np.arange(len(systems))
    width = 0.22
    seed_colors = {
        42: "#C7D8ED",
        123: "#7FA6CF",
        777: "#315F8C",
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for idx, seed in enumerate(seeds):
        vals = [
            seed_df[(seed_df["system"] == system) & (seed_df["seed"] == seed)][
                "q_main"
            ].iloc[0]
            for system in systems
        ]
        offset = (idx - 1) * width
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            color=seed_colors[seed],
            edgecolor="#333333",
            linewidth=0.45,
            label=f"seed {seed}",
        )
        for bar, val in zip(bars, vals):
            if val < 0.35:
                label_y = val + 0.010
                label_va = "bottom"
                text_color = "#222222"
            else:
                label_y = val - 0.012
                label_va = "top"
                text_color = "white" if seed == 777 else "#222222"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f"{val:.3f}",
                ha="center",
                va=label_va,
                fontsize=7.0,
                rotation=90,
                color=text_color,
            )

    base = pd.read_csv(DATA / "consolidated_results.csv").set_index("system").loc["S1", "q_main"]
    ax.axhline(base, color=COLORS["S1"], linewidth=1.2, linestyle="--")
    ax.text(
        len(systems) - 0.05,
        base + 0.008,
        "Base-RAG baseline",
        ha="right",
        va="bottom",
        fontsize=7.6,
        color=COLORS["S1"],
    )

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[system] for system in systems], rotation=18, ha="right")
    ax.set_ylabel("Q_main")
    ax.set_ylim(0.15, 0.82)
    ax.set_title("Seed-Level Stability of Trained Systems")
    clean_axes(ax, "y")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.24))

    save(fig, "figB2_seed_stability.png")


def figB3_pairwise_win_rates() -> None:
    """Pairwise win-rate heatmap for headline systems."""

    df = pd.read_csv(DATA / "pairwise_win_rate.csv")
    systems = HEADLINE
    matrix = np.full((len(systems), len(systems)), np.nan)
    counts = np.zeros((len(systems), len(systems)), dtype=int)
    ties = np.zeros((len(systems), len(systems)), dtype=int)

    for i, row_system in enumerate(systems):
        for j, col_system in enumerate(systems):
            row = df[(df["system_a"] == row_system) & (df["system_b"] == col_system)].iloc[0]
            if row_system != col_system:
                matrix[i, j] = float(row["win_rate_a_over_b"])
            counts[i, j] = int(row["win_count"])
            ties[i, j] = int(row["tie_count"])

    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    cmap = plt.colormaps["YlGn"].copy()
    cmap.set_bad(color="#F2F2F2")
    im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=0.30)

    ax.set_xticks(np.arange(len(systems)))
    ax.set_yticks(np.arange(len(systems)))
    ax.set_xticklabels([DISPLAY[system] for system in systems], rotation=25, ha="right")
    ax.set_yticklabels([DISPLAY[system] for system in systems])
    ax.set_xlabel("Column system")
    ax.set_ylabel("Row system")
    ax.set_title("Pairwise Win Rates Among Headline Systems")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i == j:
                text = "-"
                color = "#777777"
            else:
                text = f"{matrix[i, j]:.2f}\n({counts[i, j]}/50)"
                color = "white" if matrix[i, j] >= 0.22 else "#222222"
            ax.text(j, i, text, ha="center", va="center", fontsize=8.0, color=color)

    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.03)
    cbar.set_label("Fraction of questions won", rotation=270, labelpad=16)
    cbar.ax.tick_params(labelsize=8)

    ax.text(
        0.0,
        -0.24,
        "Cell = row system scores higher than column system; ties are not wins.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        color="#555555",
    )

    save(fig, "figB3_pairwise_win_rates.png")


if __name__ == "__main__":
    print(f"Generating figures from {DATA}")
    missing = [
        path
        for path in [
            DATA / "consolidated_results.csv",
            DATA / "error_overlap_jaccard.csv",
            DATA / "pairwise_win_rate.csv",
            DOC_SCOPE,
            *SEED_QMAIN_SOURCES.values(),
        ]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing expected input files: {missing}")
    fig01_system_schematic()
    fig02_delta_bars()
    fig03_judge_criteria()
    fig04_per_type_heatmap()
    fig05_singledoc_multidoc()
    figB1_error_overlap()
    figB2_seed_stability()
    figB3_pairwise_win_rates()
    print(f"All figures saved to {OUT}")
