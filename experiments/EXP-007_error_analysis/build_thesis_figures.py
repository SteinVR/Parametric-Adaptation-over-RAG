"""Build a compact thesis-ready figure pack for EXP-007 outputs."""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config as _base_cfg  # noqa: F401, E402

_spec = importlib.util.spec_from_file_location("exp007_config", Path(__file__).with_name("config.py"))
if _spec is None or _spec.loader is None:
    raise RuntimeError("Failed to load EXP-007 config")
exp_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp_cfg)


SYSTEM_ORDER: list[str] = list(exp_cfg.ALL_SYSTEMS)
HEADLINE_ORDER: list[str] = list(exp_cfg.HEADLINE_SYSTEMS)
ANSWER_TYPE_ORDER: list[str] = list(exp_cfg.ANSWER_TYPES)
DIFFICULTY_ORDER: list[str] = list(exp_cfg.DIFFICULTY_LEVELS)

SYSTEM_COLORS: dict[str, str] = {
    "S1": "#4C78A8",
    "S2+R": "#F58518",
    "S3+R": "#54A24B",
    "S7": "#E45756",
    "S2": "#9D755D",
    "S3": "#B279A2",
}

CLASS_MARKERS: dict[str, str] = {
    "Headline": "o",
    "Post-hoc": "D",
    "Control": "s",
}

TYPE_LABELS: dict[str, str] = {
    "boolean": "boolean",
    "number": "number",
    "name": "name",
    "names": "names",
    "date": "date",
    "free_text": "free_text",
}

DIFFICULTY_LABELS: dict[str, str] = {
    "easy": "легкий",
    "medium": "средний",
    "hard": "сложный",
}

FIGURE_NAMES: dict[str, str] = {
    "fig01": "fig01_quality_latency_tradeoff.png",
    "fig02": "fig02_pareto_quality_cost.png",
    "fig03": "fig03_per_type_heatmap.png",
    "fig04": "fig04_headline_pairwise_winrate.png",
    "fig05": "fig05_headline_difficulty_profile.png",
}


@dataclass(frozen=True, slots=True)
class Inputs:
    consolidated: pd.DataFrame
    per_type: pd.DataFrame
    pairwise: pd.DataFrame
    difficulty: pd.DataFrame


def _apply_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.facecolor": "#FAFAFA",
            "figure.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "grid.color": "#777777",
            "legend.frameon": False,
        }
    )


def _ordered(df: pd.DataFrame, column: str, order: Iterable[str]) -> pd.DataFrame:
    ordered_df = df.copy()
    ordered_df[column] = pd.Categorical(ordered_df[column], categories=list(order), ordered=True)
    return ordered_df.sort_values(column).reset_index(drop=True)


def _load_inputs() -> Inputs:
    consolidated = pd.read_csv(exp_cfg.CONSOLIDATED_RESULTS_CSV)
    per_type = pd.read_csv(exp_cfg.PER_TYPE_BREAKDOWN_CSV)
    pairwise = pd.read_csv(exp_cfg.PAIRWISE_WIN_RATE_CSV)
    difficulty = pd.read_csv(exp_cfg.DIFFICULTY_PROFILE_CSV)

    consolidated = _ordered(consolidated, "system", SYSTEM_ORDER)
    per_type = _ordered(per_type, "system", SYSTEM_ORDER)
    pairwise = _ordered(pairwise, "system_a", SYSTEM_ORDER)
    difficulty = _ordered(difficulty, "system", SYSTEM_ORDER)
    return Inputs(consolidated=consolidated, per_type=per_type, pairwise=pairwise, difficulty=difficulty)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _compute_pareto(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[["system", "offline_cost_seconds", "q_main"]].copy()
    subset["offline_cost_seconds"] = subset["offline_cost_seconds"].fillna(0.0)

    flags: list[bool] = []
    for _, row in subset.iterrows():
        dominated = False
        for _, other in subset.iterrows():
            if other["system"] == row["system"]:
                continue
            no_worse_cost = float(other["offline_cost_seconds"]) <= float(row["offline_cost_seconds"])
            no_worse_quality = float(other["q_main"]) >= float(row["q_main"])
            strictly_better = (
                float(other["offline_cost_seconds"]) < float(row["offline_cost_seconds"])
                or float(other["q_main"]) > float(row["q_main"])
            )
            if no_worse_cost and no_worse_quality and strictly_better:
                dominated = True
                break
        flags.append(not dominated)

    subset["is_pareto"] = flags
    return subset


def _plot_quality_latency_tradeoff(consolidated: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    plot_df = consolidated.copy()
    marker_sizes = 120 + 5.0 * np.sqrt(np.maximum(plot_df["offline_cost_seconds"].fillna(0.0), 0.0))

    for idx, row in plot_df.iterrows():
        system = str(row["system"])
        marker = CLASS_MARKERS.get(str(row["class"]), "o")
        ax.scatter(
            float(row["latency_median_ms"]),
            float(row["q_main"]),
            s=float(marker_sizes.iloc[idx]),
            marker=marker,
            color=SYSTEM_COLORS[system],
            edgecolor="#1A1A1A",
            linewidth=0.8,
            zorder=3,
        )

        q_std = row.get("q_main_std")
        if pd.notna(q_std):
            ax.errorbar(
                float(row["latency_median_ms"]),
                float(row["q_main"]),
                yerr=float(q_std),
                fmt="none",
                ecolor="#333333",
                capsize=3,
                linewidth=1.0,
                zorder=2,
            )

        ax.annotate(
            system,
            (float(row["latency_median_ms"]), float(row["q_main"])),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=10,
            color="#111111",
        )

    ax.set_title("Q_main и латентность (размер маркера = оффлайн-стоимость)")
    ax.set_xlabel("Медианная латентность, мс")
    ax.set_ylabel("Q_main")
    ax.set_ylim(0.15, 0.78)

    legend_items = [
        Line2D([0], [0], marker="o", color="w", label="headline", markerfacecolor="#666666", markersize=7),
        Line2D([0], [0], marker="D", color="w", label="post-hoc", markerfacecolor="#666666", markersize=7),
        Line2D([0], [0], marker="s", color="w", label="control", markerfacecolor="#666666", markersize=7),
    ]
    ax.legend(handles=legend_items, loc="lower left")

    output_path = output_dir / FIGURE_NAMES["fig01"]
    _save(fig, output_path)
    return output_path


def _plot_pareto_quality_cost(consolidated: pd.DataFrame, output_dir: Path) -> tuple[Path, pd.DataFrame]:
    pareto = _compute_pareto(consolidated)
    pareto["cost_plot"] = pareto["offline_cost_seconds"].fillna(0.0) + 1.0

    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    dominated = pareto[~pareto["is_pareto"]]
    front = pareto[pareto["is_pareto"]].sort_values("cost_plot")

    ax.scatter(
        dominated["cost_plot"],
        dominated["q_main"],
        s=135,
        color="#CDCDCD",
        edgecolor="#6A6A6A",
        linewidth=0.6,
        zorder=1,
        label="Доминируемые",
    )

    for _, row in front.iterrows():
        system = str(row["system"])
        ax.scatter(
            float(row["cost_plot"]),
            float(row["q_main"]),
            s=180,
            color=SYSTEM_COLORS[system],
            edgecolor="#1A1A1A",
            linewidth=0.9,
            zorder=3,
        )

    for _, row in pareto.iterrows():
        ax.annotate(
            str(row["system"]),
            (float(row["cost_plot"]), float(row["q_main"])),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=10,
        )

    if len(front) > 1:
        ax.plot(front["cost_plot"], front["q_main"], color="#1A1A1A", linewidth=1.6, zorder=2, label="Линия Парето")

    ax.set_xscale("log")
    ax.set_title("Парето-фронт: качество и оффлайн-стоимость")
    ax.set_xlabel("Оффлайн-стоимость, с (логарифм, +1)")
    ax.set_ylabel("Q_main")
    ax.set_ylim(0.15, 0.78)
    ax.legend(loc="lower right")

    output_path = output_dir / FIGURE_NAMES["fig02"]
    _save(fig, output_path)
    return output_path, pareto


def _plot_per_type_heatmap(per_type: pd.DataFrame, output_dir: Path) -> Path:
    matrix = per_type.pivot(index="answer_type", columns="system", values="score")
    matrix = matrix.reindex(index=ANSWER_TYPE_ORDER, columns=SYSTEM_ORDER)
    data = matrix.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    im = ax.imshow(data, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_title("Профиль качества по типам ответов")
    ax.set_xticks(np.arange(len(SYSTEM_ORDER)))
    ax.set_xticklabels(SYSTEM_ORDER)
    ax.set_yticks(np.arange(len(ANSWER_TYPE_ORDER)))
    ax.set_yticklabels([TYPE_LABELS[t] for t in ANSWER_TYPE_ORDER])

    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            value = data[row_idx, col_idx]
            if math.isnan(value):
                label = "N/A"
                color = "#1A1A1A"
            else:
                label = f"{value:.2f}"
                color = "#FFFFFF" if value > 0.62 else "#1A1A1A"
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=9, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Оценка")

    output_path = output_dir / FIGURE_NAMES["fig03"]
    _save(fig, output_path)
    return output_path


def _plot_headline_pairwise_heatmap(pairwise: pd.DataFrame, output_dir: Path) -> Path:
    headline = pairwise[
        pairwise["system_a"].isin(HEADLINE_ORDER) & pairwise["system_b"].isin(HEADLINE_ORDER)
    ].copy()
    matrix = headline.pivot(index="system_a", columns="system_b", values="win_rate_a_over_b")
    matrix = matrix.reindex(index=HEADLINE_ORDER, columns=HEADLINE_ORDER)
    data = matrix.to_numpy(dtype=float)

    norm = colors.TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    fig, ax = plt.subplots(figsize=(7.8, 6.4))
    im = ax.imshow(data, cmap="RdYlGn", norm=norm, aspect="equal")

    ax.set_title("Матрица попарных побед (headline-системы)")
    ax.set_xticks(np.arange(len(HEADLINE_ORDER)))
    ax.set_xticklabels(HEADLINE_ORDER)
    ax.set_yticks(np.arange(len(HEADLINE_ORDER)))
    ax.set_yticklabels(HEADLINE_ORDER)

    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            value = data[row_idx, col_idx]
            label = "N/A" if math.isnan(value) else f"{value * 100:.0f}%"
            color = "#111111" if math.isnan(value) or value < 0.65 else "#FFFFFF"
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=10, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Доля побед системы строки над системой столбца")

    output_path = output_dir / FIGURE_NAMES["fig04"]
    _save(fig, output_path)
    return output_path


def _plot_headline_difficulty_profile(difficulty: pd.DataFrame, output_dir: Path) -> Path:
    headline = difficulty[difficulty["system"].isin(HEADLINE_ORDER)].copy()
    headline["difficulty"] = pd.Categorical(headline["difficulty"], categories=DIFFICULTY_ORDER, ordered=True)
    headline = headline.sort_values(["system", "difficulty"]).reset_index(drop=True)

    x = np.arange(len(DIFFICULTY_ORDER))
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    for system in HEADLINE_ORDER:
        subset = headline[headline["system"] == system].set_index("difficulty")
        values = [float(subset.loc[level, "mean_score"]) for level in DIFFICULTY_ORDER]
        ax.plot(
            x,
            values,
            marker="o",
            linewidth=2.0,
            markersize=6,
            color=SYSTEM_COLORS[system],
            label=system,
        )

    ax.set_title("Качество по уровням сложности (headline-системы)")
    ax.set_xlabel("Сложность")
    ax.set_ylabel("Средний score по вопросам")
    ax.set_xticks(x)
    ax.set_xticklabels([DIFFICULTY_LABELS[level] for level in DIFFICULTY_ORDER])
    ax.set_ylim(0.2, 0.82)
    ax.legend(ncol=2, loc="lower left")

    output_path = output_dir / FIGURE_NAMES["fig05"]
    _save(fig, output_path)
    return output_path


def _build_caption_md(
    output_dir: Path,
    consolidated: pd.DataFrame,
    per_type: pd.DataFrame,
    pairwise: pd.DataFrame,
    difficulty: pd.DataFrame,
    pareto_df: pd.DataFrame,
) -> Path:
    leader = consolidated.sort_values("q_main", ascending=False).iloc[0]
    hard = difficulty[difficulty["difficulty"] == "hard"].sort_values("mean_score", ascending=False).iloc[0]

    free_text = per_type[per_type["answer_type"] == "free_text"].sort_values("score", ascending=False).iloc[0]

    s7_over_s2r = pairwise[
        (pairwise["system_a"] == "S7") & (pairwise["system_b"] == "S2+R")
    ]["win_rate_a_over_b"].iloc[0]

    pareto_systems = pareto_df[pareto_df["is_pareto"]].sort_values("q_main", ascending=False)["system"].tolist()

    lines = [
        "# Thesis-ready Figures (EXP-007 refresh)",
        "",
        "S6 (Naive RAG) исключен. В консолидацию включены S1, S2+R, S3+R, S7, S2, S3.",
        "",
        "## Список фигур и готовые подписи",
        "",
        f"1. `{FIGURE_NAMES['fig01']}`",
        "   Подпись: Зависимость Q_main от медианной латентности; размер маркера пропорционален оффлайн-стоимости.",
        f"   Вывод: лучший глобальный Q_main у {leader['system']} ({float(leader['q_main']):.3f}) при сопоставимом online-режиме по задержке.",
        "",
        f"2. `{FIGURE_NAMES['fig02']}`",
        "   Подпись: Парето-фронт в координатах (оффлайн-стоимость, Q_main) с отделением доминируемых систем от оптимальных кандидатов.",
        f"   Вывод: множество Парето = {', '.join(pareto_systems)}.",
        "",
        f"3. `{FIGURE_NAMES['fig03']}`",
        "   Подпись: Тепловая карта качества по типам ответов и системам.",
        f"   Вывод: лидер на free_text — {free_text['system']} ({float(free_text['score']):.3f}); по детерминированным типам преимущества распределены более фрагментированно.",
        "",
        f"4. `{FIGURE_NAMES['fig04']}`",
        "   Подпись: Матрица долей попарных побед между headline-системами по среднему score вопроса.",
        f"   Вывод: S7 побеждает S2+R на {float(s7_over_s2r) * 100:.0f}% вопросов (при большом числе ничьих, что отражает узкий, но устойчивый отрыв).",
        "",
        f"5. `{FIGURE_NAMES['fig05']}`",
        "   Подпись: Профиль среднего качества по уровням сложности для headline-систем.",
        f"   Вывод: на сложных вопросах лидер — {hard['system']} ({float(hard['mean_score']):.3f}).",
        "",
        "## Рекомендация по порядку вставки",
        "",
        "Рекомендуемый порядок в главе: Figure 2 -> Figure 1 -> Figure 3 -> Figure 4 -> Figure 5.",
        "Такая последовательность ведет от глобального компромисса внедрения к детальной диагностике поведения систем.",
    ]

    captions_path = output_dir / "CAPTIONS_RU.md"
    captions_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return captions_path


def main() -> None:
    _apply_style()
    output_dir = exp_cfg.FIGURES_DIR / "thesis_ready"
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = _load_inputs()

    _plot_quality_latency_tradeoff(inputs.consolidated, output_dir)
    _, pareto_df = _plot_pareto_quality_cost(inputs.consolidated, output_dir)
    _plot_per_type_heatmap(inputs.per_type, output_dir)
    _plot_headline_pairwise_heatmap(inputs.pairwise, output_dir)
    _plot_headline_difficulty_profile(inputs.difficulty, output_dir)
    captions_path = _build_caption_md(
        output_dir=output_dir,
        consolidated=inputs.consolidated,
        per_type=inputs.per_type,
        pairwise=inputs.pairwise,
        difficulty=inputs.difficulty,
        pareto_df=pareto_df,
    )

    print(f"Thesis-ready figures generated in: {output_dir}")
    print(f"Captions file: {captions_path}")


if __name__ == "__main__":
    main()
