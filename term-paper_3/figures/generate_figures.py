"""Generate all 5 thesis figures for the term paper."""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
DATA = Path("/home/xeliaray/Projects/Term-Paper/results/EXP-007")
OUT  = Path("/home/xeliaray/Projects/Term-Paper/term-paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────
COLORS = {
    "S1": "#4C78A8", "S2+R": "#F58518", "S3+R": "#54A24B",
    "S7": "#E45756", "S2": "#9D755D", "S3": "#B279A2",
    "S3-legacy": "#AAAAAA",
}
HEADLINE = ["S1", "S2+R", "S3+R", "S7"]
DPI = 300

def apply_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#FAFAFA")
    ax.grid(axis="y", alpha=0.22, linestyle="--", color="#777777")
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(labelsize=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "figure.facecolor": "white",
})


# ══════════════════════════════════════════════════════════════════════
# Figure 1 — System Overview Schematic
# ══════════════════════════════════════════════════════════════════════
def fig01_system_schematic():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_facecolor("white")

    arr_kw = dict(arrowstyle="-|>", color="#555555", lw=1.5,
                  connectionstyle="arc3,rad=0.0")

    def box(x, y, w, h, text, color, fontsize=9.5):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.15",
            facecolor=color, edgecolor="#333333", linewidth=1.2, alpha=0.85,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=arr_kw)

    # ── Shared retrieval block ──
    box(0.3, 2.3, 2.0, 1.4, "Retrieval\nBackbone\n(fixed)", "#6B7B8D", 10)

    # ── Query ──
    ax.text(1.3, 5.3, "Query", ha="center", va="center", fontsize=11,
            fontweight="bold", color="#333333")
    arrow(1.3, 5.0, 1.3, 3.75)

    # ── Arrows from retrieval to generators ──
    arrow(2.35, 3.5, 4.0, 4.85)
    arrow(2.35, 3.2, 4.0, 3.85)
    arrow(2.35, 2.9, 4.0, 2.85)
    arrow(2.35, 2.6, 4.0, 1.85)

    # ── Generator boxes (with system label inside) ──
    box(4.0, 4.5, 3.0, 0.7, "S1 — Base Generator", COLORS["S1"], 9.5)
    box(4.0, 3.5, 3.0, 0.7, "S2+R — RAFT Adapter", COLORS["S2+R"], 9.5)
    box(4.0, 2.5, 3.0, 0.7, "S3+R — CLM Adapter", COLORS["S3+R"], 9.5)
    box(4.0, 1.5, 3.0, 0.7, "S7 — Merged Adapter", COLORS["S7"], 9.5)

    # ── Arrows to answer ──
    for y in [4.85, 3.85, 2.85, 1.85]:
        arrow(7.05, y, 8.0, y)

    # ── Answer labels ──
    for y in [4.85, 3.85, 2.85, 1.85]:
        ax.text(8.3, y, "Answer", fontsize=9, va="center", color="#555555")

    # ── Controls (no retrieval) ──
    ax.text(1.3, 0.7, "Query", ha="center", fontsize=9, color="#888888")
    arrow(1.3, 0.55, 4.0, 0.55)
    box(4.0, 0.2, 3.0, 0.7, "S2, S3 — No Retrieval", "#AAAAAA", 9)
    arrow(7.05, 0.55, 8.0, 0.55)
    ax.text(8.3, 0.55, "Answer", fontsize=9, va="center", color="#888888")

    # ── Section labels ──
    ax.text(5.4, 5.6, "Retrieval-aware systems", fontsize=11,
            ha="center", fontstyle="italic", color="#333333")
    ax.text(5.4, -0.15, "No-retrieval controls", fontsize=10,
            ha="center", fontstyle="italic", color="#888888")

    fig.tight_layout()
    fig.savefig(OUT / "fig01_system_schematic.png", dpi=DPI,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  fig01 done")


# ══════════════════════════════════════════════════════════════════════
# Figure 2 — Improvement over S1 Baseline (Δ bars)
# ══════════════════════════════════════════════════════════════════════
def fig02_delta_bars():
    df = pd.read_csv(DATA / "consolidated_results.csv")
    s1 = df[df["system"] == "S1"].iloc[0]

    systems = ["S2+R", "S3+R", "S7"]
    metrics = ["q_main", "s_det", "s_asst"]
    labels  = ["ΔQ_main", "ΔS_det", "ΔS_asst"]

    deltas = {}
    for sys in systems:
        row = df[df["system"] == sys].iloc[0]
        deltas[sys] = [row[m] - s1[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.22
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for i, sys in enumerate(systems):
        bars = ax.bar(x + (i - 1) * width, deltas[sys], width,
                      label=sys, color=COLORS[sys], edgecolor="white",
                      linewidth=0.5, alpha=0.88)
        for bar, val in zip(bars, deltas[sys]):
            yoff = 0.003 if val >= 0 else -0.008
            ax.text(bar.get_x() + bar.get_width() / 2, val + yoff,
                    f"{val:+.3f}", ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=8.5, fontweight="bold")

    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    apply_style(ax, title="Improvement over S1 Baseline",
                ylabel="Δ Score")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.set_ylim(-0.06, 0.12)

    fig.tight_layout()
    fig.savefig(OUT / "fig02_delta_bars.png", dpi=DPI,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  fig02 done")


# ══════════════════════════════════════════════════════════════════════
# Figure 3 — Judge Criteria Profile (headline only)
# ══════════════════════════════════════════════════════════════════════
def fig03_judge_criteria():
    df = pd.read_csv(DATA / "judge_criteria_profile.csv")
    df = df[df["system"].isin(HEADLINE)]

    criteria = ["correctness", "completeness", "grounding",
                "calibration", "clarity"]
    systems = HEADLINE

    x = np.arange(len(criteria))
    width = 0.18
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    for i, sys in enumerate(systems):
        sdf = df[df["system"] == sys].set_index("criterion")
        vals = [sdf.loc[c, "mean_score"] for c in criteria]
        ax.bar(x + (i - 1.5) * width, vals, width,
               label=sys, color=COLORS[sys], edgecolor="white",
               linewidth=0.5, alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in criteria], fontsize=10)
    apply_style(ax, title="Judge Criteria Profile (Free-Text Questions)",
                ylabel="Mean Score")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper left", frameon=False, fontsize=10, ncol=2)

    fig.tight_layout()
    fig.savefig(OUT / "fig03_judge_criteria.png", dpi=DPI,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  fig03 done")


# ══════════════════════════════════════════════════════════════════════
# Figure 4 — Per-Type Score Heatmap (headline only, with n)
# ══════════════════════════════════════════════════════════════════════
def fig04_per_type_heatmap():
    df = pd.read_csv(DATA / "per_type_breakdown.csv")
    df = df[df["system"].isin(HEADLINE)]

    type_n = {"boolean": 12, "number": 7, "name": 8,
              "names": 5, "date": 5, "free_text": 13}
    types_order = list(type_n.keys())
    systems = HEADLINE

    matrix = []
    for sys in systems:
        sdf = df[df["system"] == sys].set_index("answer_type")
        matrix.append([sdf.loc[t, "score"] for t in types_order])
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto",
                   vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(types_order)))
    xlabels = [f"{t}\n(n={type_n[t]})" for t in types_order]
    ax.set_xticklabels(xlabels, fontsize=9.5)
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels(systems, fontsize=10, fontweight="bold")

    for i in range(len(systems)):
        for j in range(len(types_order)):
            val = matrix[i, j]
            color = "white" if val > 0.6 else "#333333"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=9)
    ax.set_title("Score by Answer Type", fontsize=13,
                 fontweight="bold", pad=10)

    fig.tight_layout()
    fig.savefig(OUT / "fig04_per_type_heatmap.png", dpi=DPI,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  fig04 done")


# ══════════════════════════════════════════════════════════════════════
# Figure 5 — Single-doc vs Multi-doc
# ══════════════════════════════════════════════════════════════════════
def fig05_singledoc_multidoc():
    df = pd.read_csv(DATA / "question_score_summary.csv")
    df = df[df["system"].isin(HEADLINE)]

    groups = df.groupby(["system", "is_multi_doc"])["score_mean"].mean()
    systems = HEADLINE

    single = [groups.get((s, False), 0) for s in systems]
    multi  = [groups.get((s, True), 0)  for s in systems]

    x = np.arange(len(systems))
    width = 0.32
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    bars1 = ax.bar(x - width / 2, single, width, label="Single-doc (n=42)",
                   color=[COLORS[s] for s in systems], edgecolor="white",
                   linewidth=0.5, alpha=0.88)
    bars2 = ax.bar(x + width / 2, multi, width, label="Multi-doc (n=8)",
                   color=[COLORS[s] for s in systems], edgecolor="white",
                   linewidth=0.5, alpha=0.45, hatch="///")

    for bar, val in zip(bars1, single):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8.5,
                fontweight="bold")
    for bar, val in zip(bars2, multi):
        ax.text(bar.get_x() + bar.get_width() / 2, val / 2,
                f"{val:.3f}", ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=11, fontweight="bold")
    apply_style(ax, title="Single-Document vs. Multi-Document Performance",
                ylabel="Q_main")
    ax.set_ylim(0, 0.85)
    ax.legend(loc="upper left", frameon=False, fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT / "fig05_singledoc_multidoc.png", dpi=DPI,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  fig05 done")


# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures...")
    fig01_system_schematic()
    fig02_delta_bars()
    fig03_judge_criteria()
    fig04_per_type_heatmap()
    fig05_singledoc_multidoc()
    print("All figures saved to:", OUT)
