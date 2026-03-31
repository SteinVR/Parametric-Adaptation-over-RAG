"""
EXP-006: Main comparison refresh (with EXP-010 / S7, without S6).
Analysis-only script: consumes existing experiment artifacts and regenerates
comparison tables, deltas, and report outputs.
"""

from __future__ import annotations

import csv
import importlib.util
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg  # noqa: E402
from src.data.io import load_goldset, load_json, save_json  # noqa: E402

logging.basicConfig(format=cfg.LOG_FORMAT, datefmt=cfg.LOG_DATE_FORMAT, level=logging.INFO)
log = logging.getLogger(__name__)

_spec = importlib.util.spec_from_file_location("exp006_config", EXPERIMENT_DIR / "config.py")
exp_cfg = importlib.util.module_from_spec(_spec)
if _spec is None or _spec.loader is None:
    raise RuntimeError("Failed to load EXP-006 config")
_spec.loader.exec_module(exp_cfg)


@dataclass(slots=True)
class SystemResult:
    system_id: str
    system_class: str
    q_main: float
    q_main_std: float
    s_det: float
    s_det_std: float
    s_asst: float
    s_asst_std: float
    g: float | None
    g_std: float | None
    ttft_median_ms: float | None
    ttft_median_std: float | None
    ttft_p95_ms: float | None
    ttft_p95_std: float | None
    latency_median_ms: float | None
    latency_median_std: float | None
    latency_p95_ms: float | None
    latency_p95_std: float | None
    peak_infer_vram_mb: float | None
    peak_infer_vram_std: float | None
    peak_train_vram_mb: float | None
    peak_train_vram_std: float | None
    offline_cost_seconds: float | None
    offline_cost_std: float | None
    malformed_rate: float | None
    malformed_rate_std: float | None
    breakdown: dict[str, float]


def _extract_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if "mean" in value:
            value = value["mean"]
        else:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return mean(values), (stdev(values) if len(values) > 1 else 0.0)


def _collect_seed_metric(seed_reports: list[dict[str, Any]], key: str) -> tuple[float | None, float | None]:
    values = [float(r[key]) for r in seed_reports if r.get(key) is not None]
    return _mean_std(values)


def _extract_breakdown_scores(raw_breakdown: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for answer_type in exp_cfg.ANSWER_TYPES:
        type_metrics = raw_breakdown.get(answer_type)
        if not isinstance(type_metrics, dict):
            continue
        metric_key = "s_asst_mean" if answer_type == "free_text" else "s_det_mean"
        value = _extract_float(type_metrics.get(metric_key))
        if value is not None:
            scores[answer_type] = value
    return scores


def _load_seed_reports(results_dir: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for seed in exp_cfg.SEEDS:
        report_path = results_dir / f"seed_{seed}" / "eval_report.json"
        if report_path.exists():
            reports.append(load_json(report_path))
    if not reports:
        raise FileNotFoundError(f"No seed eval_report.json files in {results_dir}")
    return reports


def _load_seed_eval_results(results_dir: Path) -> list[list[dict[str, Any]]]:
    eval_runs: list[list[dict[str, Any]]] = []
    for seed in exp_cfg.SEEDS:
        eval_path = results_dir / f"seed_{seed}" / "eval_results.json"
        if eval_path.exists():
            eval_runs.append(load_json(eval_path))
    if not eval_runs:
        raise FileNotFoundError(f"No seed eval_results.json files in {results_dir}")
    return eval_runs


def _load_single_run(report_path: Path, system_id: str, system_class: str) -> SystemResult:
    report = load_json(report_path)
    systems_metrics = load_json(exp_cfg.S1_SYSTEMS_METRICS) if exp_cfg.S1_SYSTEMS_METRICS.exists() else {}
    breakdown = _extract_breakdown_scores(report.get("breakdown_by_type", {}))

    return SystemResult(
        system_id=system_id,
        system_class=system_class,
        q_main=float(report["q_main"]),
        q_main_std=0.0,
        s_det=float(report["s_det"]),
        s_det_std=0.0,
        s_asst=float(report["s_asst"]),
        s_asst_std=0.0,
        g=_extract_float(report.get("grounding_f_beta")),
        g_std=0.0 if report.get("grounding_f_beta") is not None else None,
        ttft_median_ms=_extract_float(report.get("ttft_median_ms")),
        ttft_median_std=0.0,
        ttft_p95_ms=_extract_float(report.get("ttft_p95_ms")),
        ttft_p95_std=0.0,
        latency_median_ms=_extract_float(report.get("latency_median_ms")),
        latency_median_std=0.0,
        latency_p95_ms=_extract_float(report.get("latency_p95_ms")),
        latency_p95_std=0.0,
        peak_infer_vram_mb=_extract_float(systems_metrics.get("peak_vram_mb")),
        peak_infer_vram_std=0.0 if systems_metrics.get("peak_vram_mb") is not None else None,
        peak_train_vram_mb=None,
        peak_train_vram_std=None,
        offline_cost_seconds=0.0,
        offline_cost_std=0.0,
        malformed_rate=_extract_float(report.get("malformed_rate")),
        malformed_rate_std=0.0,
        breakdown=breakdown,
    )


def _load_aggregate(agg_path: Path, results_dir: Path, system_id: str, system_class: str) -> SystemResult:
    aggregate = load_json(agg_path)
    seed_reports = _load_seed_reports(results_dir)
    breakdown = _extract_breakdown_scores(aggregate.get("breakdown_by_type", {}))

    ttft_median, ttft_median_std = _collect_seed_metric(seed_reports, "ttft_median_ms")
    ttft_p95, ttft_p95_std = _collect_seed_metric(seed_reports, "ttft_p95_ms")
    latency_median, latency_median_std = _collect_seed_metric(seed_reports, "latency_median_ms")
    latency_p95, latency_p95_std = _collect_seed_metric(seed_reports, "latency_p95_ms")
    malformed_rate, malformed_rate_std = _collect_seed_metric(seed_reports, "malformed_rate")

    g_data = aggregate.get("G_f_beta")
    g_mean = _extract_float(g_data)
    g_std = None
    if isinstance(g_data, dict):
        g_std = _extract_float(g_data.get("std"))

    peak_infer = aggregate.get("peak_infer_vram_mb")
    peak_train = aggregate.get("peak_train_vram_mb")
    offline_cost = aggregate.get("train_time_seconds")

    return SystemResult(
        system_id=system_id,
        system_class=system_class,
        q_main=float(aggregate["Q_main"]["mean"]),
        q_main_std=float(aggregate["Q_main"]["std"]),
        s_det=float(aggregate["S_det"]["mean"]),
        s_det_std=float(aggregate["S_det"]["std"]),
        s_asst=float(aggregate["S_asst"]["mean"]),
        s_asst_std=float(aggregate["S_asst"]["std"]),
        g=g_mean,
        g_std=g_std,
        ttft_median_ms=ttft_median,
        ttft_median_std=ttft_median_std,
        ttft_p95_ms=ttft_p95,
        ttft_p95_std=ttft_p95_std,
        latency_median_ms=latency_median,
        latency_median_std=latency_median_std,
        latency_p95_ms=latency_p95,
        latency_p95_std=latency_p95_std,
        peak_infer_vram_mb=_extract_float(peak_infer),
        peak_infer_vram_std=_extract_float(peak_infer.get("std") if isinstance(peak_infer, dict) else None),
        peak_train_vram_mb=_extract_float(peak_train),
        peak_train_vram_std=_extract_float(peak_train.get("std") if isinstance(peak_train, dict) else None),
        offline_cost_seconds=_extract_float(offline_cost),
        offline_cost_std=_extract_float(offline_cost.get("std") if isinstance(offline_cost, dict) else None),
        malformed_rate=malformed_rate,
        malformed_rate_std=malformed_rate_std,
        breakdown=breakdown,
    )


def collect_all() -> dict[str, SystemResult]:
    results: dict[str, SystemResult] = {}
    results["S1"] = _load_single_run(exp_cfg.S1_EVAL_REPORT, "S1", "Headline")
    results["S2+R"] = _load_aggregate(exp_cfg.S2R_AGGREGATE, exp_cfg.S2R_RESULTS_DIR, "S2+R", "Headline")
    results["S3+R"] = _load_aggregate(exp_cfg.S3R_AGGREGATE, exp_cfg.S3R_RESULTS_DIR, "S3+R", "Headline")
    results["S7"] = _load_aggregate(exp_cfg.S7_AGGREGATE, exp_cfg.S7_RESULTS_DIR, "S7", "Post-hoc")
    results["S2"] = _load_aggregate(exp_cfg.S2_AGGREGATE, exp_cfg.S2_RESULTS_DIR, "S2", "Control")
    results["S3"] = _load_aggregate(exp_cfg.S3_AGGREGATE, exp_cfg.S3_RESULTS_DIR, "S3", "Control")
    log.info("Collected %d systems", len(results))
    return results


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def write_main_results_csv(results: dict[str, SystemResult]) -> None:
    exp_cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with exp_cfg.MAIN_RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "system",
                "class",
                "q_main",
                "q_main_std",
                "s_det",
                "s_det_std",
                "s_asst",
                "s_asst_std",
                "g",
                "g_std",
                "ttft_median_ms",
                "ttft_median_std",
                "ttft_p95_ms",
                "ttft_p95_std",
                "latency_median_ms",
                "latency_median_std",
                "latency_p95_ms",
                "latency_p95_std",
                "peak_infer_vram_mb",
                "peak_infer_vram_std",
                "peak_train_vram_mb",
                "peak_train_vram_std",
                "offline_cost_seconds",
                "offline_cost_std",
                "malformed_rate",
                "malformed_rate_std",
            ]
        )
        for system_id in exp_cfg.SYSTEMS_ORDER:
            result = results[system_id]
            writer.writerow(
                [
                    result.system_id,
                    result.system_class,
                    _fmt(result.q_main),
                    _fmt(result.q_main_std),
                    _fmt(result.s_det),
                    _fmt(result.s_det_std),
                    _fmt(result.s_asst),
                    _fmt(result.s_asst_std),
                    _fmt(result.g),
                    _fmt(result.g_std),
                    _fmt(result.ttft_median_ms),
                    _fmt(result.ttft_median_std),
                    _fmt(result.ttft_p95_ms),
                    _fmt(result.ttft_p95_std),
                    _fmt(result.latency_median_ms),
                    _fmt(result.latency_median_std),
                    _fmt(result.latency_p95_ms),
                    _fmt(result.latency_p95_std),
                    _fmt(result.peak_infer_vram_mb),
                    _fmt(result.peak_infer_vram_std),
                    _fmt(result.peak_train_vram_mb),
                    _fmt(result.peak_train_vram_std),
                    _fmt(result.offline_cost_seconds),
                    _fmt(result.offline_cost_std),
                    _fmt(result.malformed_rate),
                    _fmt(result.malformed_rate_std),
                ]
            )
    log.info("Wrote %s", exp_cfg.MAIN_RESULTS_CSV)


def write_per_type_breakdown(results: dict[str, SystemResult]) -> None:
    with exp_cfg.PER_TYPE_BREAKDOWN_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["answer_type"] + exp_cfg.SYSTEMS_ORDER)
        for answer_type in exp_cfg.ANSWER_TYPES:
            row = [answer_type]
            for system_id in exp_cfg.SYSTEMS_ORDER:
                row.append(_fmt(results[system_id].breakdown.get(answer_type)))
            writer.writerow(row)
    log.info("Wrote %s", exp_cfg.PER_TYPE_BREAKDOWN_CSV)


def _load_multi_doc_question_sets() -> tuple[set[str], set[str]]:
    references = load_goldset(exp_cfg.GOLDSET_PATH)
    refs_by_id = {ref["question_id"]: ref for ref in references}
    split = load_json(exp_cfg.SPLIT_PATH)

    single_doc_ids: set[str] = set()
    multi_doc_ids: set[str] = set()
    for question_id in split["eval"]:
        ref = refs_by_id[question_id]
        doc_ids = {
            item.get("doc_id")
            for item in ref.get("gold_retrieval", [])
            if isinstance(item, dict) and item.get("doc_id")
        }
        if len(doc_ids) > 1:
            multi_doc_ids.add(question_id)
        else:
            single_doc_ids.add(question_id)
    return single_doc_ids, multi_doc_ids


def _subset_q_main(eval_results: list[dict[str, Any]], question_ids: set[str]) -> float:
    det_values: list[float] = []
    asst_values: list[float] = []

    for row in eval_results:
        if row.get("question_id") not in question_ids:
            continue
        if row.get("answer_type") == "free_text":
            if row.get("s_asst") is not None:
                asst_values.append(float(row["s_asst"]))
        else:
            if row.get("s_det") is not None:
                det_values.append(float(row["s_det"]))

    if det_values and asst_values:
        return (
            exp_cfg.Q_MAIN_WEIGHTS["S_det"] * mean(det_values)
            + exp_cfg.Q_MAIN_WEIGHTS["S_asst"] * mean(asst_values)
        )
    if det_values:
        return mean(det_values)
    if asst_values:
        return mean(asst_values)
    return 0.0


def _collect_eval_runs_by_system() -> dict[str, list[list[dict[str, Any]]]]:
    return {
        "S1": [load_json(exp_cfg.S1_EVAL_RESULTS)],
        "S2+R": _load_seed_eval_results(exp_cfg.S2R_RESULTS_DIR),
        "S3+R": _load_seed_eval_results(exp_cfg.S3R_RESULTS_DIR),
        "S7": _load_seed_eval_results(exp_cfg.S7_RESULTS_DIR),
        "S2": _load_seed_eval_results(exp_cfg.S2_RESULTS_DIR),
        "S3": _load_seed_eval_results(exp_cfg.S3_RESULTS_DIR),
    }


def write_single_multi_breakdown() -> dict[str, dict[str, float]]:
    single_doc_ids, multi_doc_ids = _load_multi_doc_question_sets()
    eval_runs = _collect_eval_runs_by_system()

    summary: dict[str, dict[str, float]] = {}
    with exp_cfg.SINGLE_MULTI_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "system",
                "single_doc_q_main",
                "single_doc_std",
                "multi_doc_q_main",
                "multi_doc_std",
                "delta_multi_minus_single",
            ]
        )

        for system_id in exp_cfg.SYSTEMS_ORDER:
            single_scores = [_subset_q_main(run, single_doc_ids) for run in eval_runs[system_id]]
            multi_scores = [_subset_q_main(run, multi_doc_ids) for run in eval_runs[system_id]]
            single_mean, single_std = _mean_std(single_scores)
            multi_mean, multi_std = _mean_std(multi_scores)

            delta = None
            if single_mean is not None and multi_mean is not None:
                delta = multi_mean - single_mean

            summary[system_id] = {
                "single_doc_q_main": float(single_mean or 0.0),
                "single_doc_std": float(single_std or 0.0),
                "multi_doc_q_main": float(multi_mean or 0.0),
                "multi_doc_std": float(multi_std or 0.0),
                "delta_multi_minus_single": float(delta or 0.0),
            }

            writer.writerow(
                [
                    system_id,
                    _fmt(single_mean),
                    _fmt(single_std),
                    _fmt(multi_mean),
                    _fmt(multi_std),
                    _fmt(delta),
                ]
            )

    log.info("Wrote %s", exp_cfg.SINGLE_MULTI_CSV)
    return summary


def compute_deltas(results: dict[str, SystemResult]) -> dict[str, dict[str, float]]:
    pairs = [
        ("S2+R_vs_S1", "S2+R", "S1"),
        ("S3+R_vs_S1", "S3+R", "S1"),
        ("S7_vs_S1", "S7", "S1"),
        ("S2+R_vs_S3+R", "S2+R", "S3+R"),
        ("S7_vs_S2+R", "S7", "S2+R"),
        ("S7_vs_S3+R", "S7", "S3+R"),
        ("S2+R_vs_S2", "S2+R", "S2"),
        ("S3+R_vs_S3", "S3+R", "S3"),
    ]
    deltas: dict[str, dict[str, float]] = {}
    for label, a, b in pairs:
        ra, rb = results[a], results[b]
        deltas[label] = {
            "Q_main": ra.q_main - rb.q_main,
            "S_det": ra.s_det - rb.s_det,
            "S_asst": ra.s_asst - rb.s_asst,
        }
    save_json(deltas, exp_cfg.DELTAS_JSON)
    log.info("Wrote %s (%d pairs)", exp_cfg.DELTAS_JSON, len(deltas))
    return deltas


def write_gradient_plot(results: dict[str, SystemResult]) -> None:
    ordered = sorted(exp_cfg.SYSTEMS_ORDER, key=lambda s: results[s].q_main, reverse=True)
    values = [results[s].q_main for s in ordered]
    colors = [
        "#0E7C86" if results[s].system_class == "Headline" else "#8A5A44"
        for s in ordered
    ]
    colors = ["#2C6E49" if s == "S7" else color for s, color in zip(ordered, colors, strict=True)]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
    bars = ax.bar(ordered, values, color=colors, edgecolor="#222222", linewidth=1.0)
    ax.set_title("EXP-006: Q_main Ranking (S6 Excluded)")
    ax.set_ylabel("Q_main")
    ax.set_ylim(0.0, max(values) + 0.08)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.005,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(exp_cfg.GRADIENT_PLOT_PATH)
    plt.close(fig)
    log.info("Wrote %s", exp_cfg.GRADIENT_PLOT_PATH)


def _fmt_with_std(value: float, std: float) -> str:
    if std > 0:
        return f"{value:.4f} ± {std:.4f}"
    return f"{value:.4f}"


def _fmt_opt(value: float | None, std: float | None = None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    if std is not None and std > 0:
        return f"{value:.{digits}f} ± {std:.{digits}f}"
    return f"{value:.{digits}f}"


def write_report(
    results: dict[str, SystemResult],
    deltas: dict[str, dict[str, float]],
    single_multi: dict[str, dict[str, float]],
) -> None:
    lines: list[str] = [
        "# EXP-006: Main Comparison",
        "",
        f"**Date:** {datetime.now(timezone.utc).date().isoformat()}",
        "",
        "## Scope",
        "",
        "- Systems: S1, S2+R, S3+R, S7 (post-hoc from EXP-010), S2, S3.",
        "- S6 (Naive RAG) is intentionally excluded from EXP-006 outputs.",
        "",
        "## Table 1: Unified System Metrics",
        "",
        "| System | Class | Q_main | S_det | S_asst | G | TTFT median (ms) | Latency median (ms) | Peak infer VRAM (MB) | Offline cost (s) |",
        "|--------|-------|--------|-------|--------|---|------------------|---------------------|----------------------|------------------|",
    ]

    for system_id in exp_cfg.SYSTEMS_ORDER:
        result = results[system_id]
        lines.append(
            "| {sid} | {cls} | {q} | {sdet} | {sasst} | {g} | {ttft} | {lat} | {vram} | {cost} |".format(
                sid=result.system_id,
                cls=result.system_class,
                q=_fmt_with_std(result.q_main, result.q_main_std),
                sdet=_fmt_with_std(result.s_det, result.s_det_std),
                sasst=_fmt_with_std(result.s_asst, result.s_asst_std),
                g=_fmt_opt(result.g, result.g_std, 4),
                ttft=_fmt_opt(result.ttft_median_ms, result.ttft_median_std, 1),
                lat=_fmt_opt(result.latency_median_ms, result.latency_median_std, 1),
                vram=_fmt_opt(result.peak_infer_vram_mb, result.peak_infer_vram_std, 1),
                cost=_fmt_opt(result.offline_cost_seconds, result.offline_cost_std, 1),
            )
        )

    lines.extend([
        "",
        "## Table 2: Per-Type Score (S_det for deterministic, S_asst for free_text)",
        "",
        "| Answer type | " + " | ".join(exp_cfg.SYSTEMS_ORDER) + " |",
        "|---|" + "---|" * len(exp_cfg.SYSTEMS_ORDER),
    ])

    for answer_type in exp_cfg.ANSWER_TYPES:
        row = [answer_type]
        for system_id in exp_cfg.SYSTEMS_ORDER:
            row.append(_fmt(results[system_id].breakdown.get(answer_type)))
        lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "## Table 3: Single-Doc vs Multi-Doc Q_main",
        "",
        "| System | Single-doc | Multi-doc | Δ (multi - single) |",
        "|---|---:|---:|---:|",
    ])

    for system_id in exp_cfg.SYSTEMS_ORDER:
        sm = single_multi[system_id]
        lines.append(
            f"| {system_id} | {sm['single_doc_q_main']:.4f} ± {sm['single_doc_std']:.4f} | "
            f"{sm['multi_doc_q_main']:.4f} ± {sm['multi_doc_std']:.4f} | "
            f"{sm['delta_multi_minus_single']:+.4f} |"
        )

    lines.extend([
        "",
        "## Key Deltas",
        "",
    ])
    for label, delta in deltas.items():
        lines.append(
            f"- **{label}**: Q_main={delta['Q_main']:+.4f}, "
            f"S_det={delta['S_det']:+.4f}, S_asst={delta['S_asst']:+.4f}"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- S7 (adapter merge from EXP-010) is strongest by Q_main: {results['S7'].q_main:.4f}.",
        f"- S2+R vs S3+R remains trade-off shaped: ΔQ_main={deltas['S2+R_vs_S3+R']['Q_main']:+.4f}, "
        f"ΔS_det={deltas['S2+R_vs_S3+R']['S_det']:+.4f}, "
        f"ΔS_asst={deltas['S2+R_vs_S3+R']['S_asst']:+.4f}.",
        f"- Retrieval contribution stays dominant: "
        f"S2→S2+R {deltas['S2+R_vs_S2']['Q_main']:+.4f}, "
        f"S3→S3+R {deltas['S3+R_vs_S3']['Q_main']:+.4f}.",
        "",
        "## Artifacts",
        "",
        f"- `{exp_cfg.MAIN_RESULTS_CSV}`",
        f"- `{exp_cfg.PER_TYPE_BREAKDOWN_CSV}`",
        f"- `{exp_cfg.SINGLE_MULTI_CSV}`",
        f"- `{exp_cfg.DELTAS_JSON}`",
        f"- `{exp_cfg.GRADIENT_PLOT_PATH}`",
    ])

    exp_cfg.REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Wrote %s", exp_cfg.REPORT_PATH)


def main() -> None:
    log.info("=== EXP-006 refresh start (with S7 / without S6) ===")
    results = collect_all()
    write_main_results_csv(results)
    write_per_type_breakdown(results)
    single_multi = write_single_multi_breakdown()
    deltas = compute_deltas(results)
    write_gradient_plot(results)
    write_report(results, deltas, single_multi)
    log.info("=== EXP-006 refresh done ===")


if __name__ == "__main__":
    main()
