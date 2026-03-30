"""Aggregation helpers for multi-seed experiment summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from src.evaluation.schemas import EvalReport


@dataclass(frozen=True, slots=True)
class SeedEvalSummary:
    """Compact summary for one EXP-003 seed run."""

    seed: int
    q_main: float
    s_det: float
    s_asst: float
    grounding_f_beta: float | None
    train_time_seconds: float
    peak_train_vram_mb: float | None
    peak_infer_vram_mb: float | None
    breakdown_by_type: dict[str, dict[str, float]]


def aggregate_seed_summaries(
    summaries: list[SeedEvalSummary],
    s1_report_path: Path,
) -> dict[str, Any]:
    """Aggregate mean/std metrics and deltas versus S1."""

    if not summaries:
        raise ValueError("Expected at least one seed summary")

    s1_report = json.loads(s1_report_path.read_text(encoding="utf-8"))
    metrics = {
        "Q_main": [item.q_main for item in summaries],
        "S_det": [item.s_det for item in summaries],
        "S_asst": [item.s_asst for item in summaries],
        "G_f_beta": [item.grounding_f_beta for item in summaries if item.grounding_f_beta is not None],
        "train_time_seconds": [item.train_time_seconds for item in summaries],
        "peak_train_vram_mb": [item.peak_train_vram_mb for item in summaries if item.peak_train_vram_mb is not None],
        "peak_infer_vram_mb": [item.peak_infer_vram_mb for item in summaries if item.peak_infer_vram_mb is not None],
    }

    aggregate = {
        metric_name: {
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
        }
        for metric_name, values in metrics.items()
        if values
    }
    delta_vs_s1: dict[str, float] = {
        "Q_main": aggregate["Q_main"]["mean"] - float(s1_report["q_main"]),
        "S_det": aggregate["S_det"]["mean"] - float(s1_report["s_det"]),
        "S_asst": aggregate["S_asst"]["mean"] - float(s1_report["s_asst"]),
    }
    if "G_f_beta" in aggregate and s1_report.get("grounding_f_beta") is not None:
        delta_vs_s1["G_f_beta"] = aggregate["G_f_beta"]["mean"] - float(s1_report["grounding_f_beta"])
    aggregate["delta_vs_s1"] = delta_vs_s1
    aggregate["seed_results"] = [asdict(summary) for summary in summaries]
    aggregate["breakdown_by_type"] = _aggregate_breakdown(summaries)
    return aggregate


def save_seed_aggregate(summary: dict[str, Any], output_dir: Path) -> None:
    """Persist aggregate summary as JSON and CSV."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_path = output_dir / "aggregate_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "mean", "std"])
        for metric_name, metric_summary in summary.items():
            if not isinstance(metric_summary, dict):
                continue
            if "mean" not in metric_summary:
                continue
            writer.writerow([metric_name, metric_summary["mean"], metric_summary["std"]])


def eval_report_to_summary(
    *,
    seed: int,
    report: EvalReport,
    train_time_seconds: float,
    peak_train_vram_mb: float | None,
    peak_infer_vram_mb: float | None,
) -> SeedEvalSummary:
    """Convert an EvalReport into the summary schema used for aggregation."""

    return SeedEvalSummary(
        seed=seed,
        q_main=report.q_main,
        s_det=report.s_det,
        s_asst=report.s_asst,
        grounding_f_beta=report.grounding_f_beta,
        train_time_seconds=train_time_seconds,
        peak_train_vram_mb=peak_train_vram_mb,
        peak_infer_vram_mb=peak_infer_vram_mb,
        breakdown_by_type=report.breakdown_by_type,
    )


def _aggregate_breakdown(
    summaries: list[SeedEvalSummary],
) -> dict[str, dict[str, dict[str, float]]]:
    per_type_metric_values: dict[str, dict[str, list[float]]] = {}
    for summary in summaries:
        for answer_type, metrics in summary.breakdown_by_type.items():
            type_metrics = per_type_metric_values.setdefault(answer_type, {})
            for metric_name, metric_value in metrics.items():
                type_metrics.setdefault(metric_name, []).append(float(metric_value))

    return {
        answer_type: {
            metric_name: {
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
            }
            for metric_name, values in metrics.items()
        }
        for answer_type, metrics in per_type_metric_values.items()
    }
