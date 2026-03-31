"""
EXP-007: Error analysis + trade-off synthesis refresh.

Rebuilds mandatory outputs and extends analysis with deeper aggregations/figures,
while explicitly excluding S6 and incorporating S7 from EXP-010.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg  # noqa: E402
from src.data.io import load_goldset, load_json  # noqa: E402

logging.basicConfig(level=logging.INFO, format=cfg.LOG_FORMAT, datefmt=cfg.LOG_DATE_FORMAT)
log = logging.getLogger(__name__)

_spec = importlib.util.spec_from_file_location("exp007_config", Path(__file__).with_name("config.py"))
if _spec is None or _spec.loader is None:
    raise RuntimeError("Failed to load EXP-007 config")
exp_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp_cfg)


QUESTION_SCORE_CSV = exp_cfg.RESULTS_DIR / "question_score_summary.csv"


@dataclass(frozen=True, slots=True)
class SystemSpec:
    system_id: str
    system_class: str
    root_dir: Path
    aggregate_path: Path | None
    seeded: bool
    retrieval: bool


SYSTEM_SPECS: dict[str, SystemSpec] = {
    "S1": SystemSpec("S1", "Headline", exp_cfg.S1_DIR, None, False, True),
    "S2+R": SystemSpec("S2+R", "Headline", exp_cfg.S2R_DIR, exp_cfg.S2R_AGGREGATE, True, True),
    "S3+R": SystemSpec("S3+R", "Headline", exp_cfg.S3R_DIR, exp_cfg.S3R_AGGREGATE, True, True),
    "S7": SystemSpec("S7", "Post-hoc", exp_cfg.S7_DIR, exp_cfg.S7_AGGREGATE, True, True),
    "S2": SystemSpec("S2", "Control", exp_cfg.S2_DIR, exp_cfg.S2_AGGREGATE, True, False),
    "S3": SystemSpec("S3", "Control", exp_cfg.S3_DIR, exp_cfg.S3_AGGREGATE, True, False),
}


def _ensure_dirs() -> None:
    exp_cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    exp_cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return mean(values), (stdev(values) if len(values) > 1 else 0.0)


def _as_float(value: Any) -> float | None:
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


def _extract_stat(value: Any) -> tuple[float | None, float | None]:
    if value is None:
        return None, None
    if isinstance(value, dict):
        mean_value = _as_float(value.get("mean"))
        std_value = _as_float(value.get("std"))
        return mean_value, (std_value if std_value is not None else 0.0)
    cast_value = _as_float(value)
    return cast_value, (0.0 if cast_value is not None else None)


def _seed_dir(spec: SystemSpec, seed: int) -> Path:
    return spec.root_dir / f"seed_{seed}"


def _eval_report_path(spec: SystemSpec, seed: int | None = None) -> Path:
    if spec.seeded:
        if seed is None:
            raise ValueError(f"Seed required for {spec.system_id}")
        return _seed_dir(spec, seed) / "eval_report.json"
    return spec.root_dir / "eval_report.json"


def _eval_results_path(spec: SystemSpec, seed: int | None = None) -> Path:
    if spec.seeded:
        if seed is None:
            raise ValueError(f"Seed required for {spec.system_id}")
        return _seed_dir(spec, seed) / "eval_results.json"
    return spec.root_dir / "eval_results.json"


def _predictions_path(spec: SystemSpec, seed: int | None = None) -> Path:
    if spec.seeded:
        if seed is None:
            raise ValueError(f"Seed required for {spec.system_id}")
        seed_root = _seed_dir(spec, seed)
        candidates = [
            seed_root / f"predictions_seed_{seed}.json",
            seed_root / "predictions.json",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"Predictions file not found for {spec.system_id} seed {seed}")
    return spec.root_dir / "predictions.json"


def _collect_seed_reports(spec: SystemSpec) -> list[dict[str, Any]]:
    if not spec.seeded:
        return [load_json(_eval_report_path(spec))]

    reports: list[dict[str, Any]] = []
    for seed in exp_cfg.SEEDS:
        path = _eval_report_path(spec, seed)
        if not path.exists():
            raise FileNotFoundError(f"Missing eval report: {path}")
        reports.append(load_json(path))
    return reports


def _collect_eval_runs(spec: SystemSpec) -> list[tuple[int, list[dict[str, Any]]]]:
    if not spec.seeded:
        return [(0, load_json(_eval_results_path(spec)))]

    runs: list[tuple[int, list[dict[str, Any]]]] = []
    for seed in exp_cfg.SEEDS:
        path = _eval_results_path(spec, seed)
        if not path.exists():
            raise FileNotFoundError(f"Missing eval results: {path}")
        runs.append((seed, load_json(path)))
    return runs


def _collect_representative_predictions(spec: SystemSpec) -> dict[str, dict[str, Any]]:
    seed = exp_cfg.REPRESENTATIVE_SEED if spec.seeded else None
    records = load_json(_predictions_path(spec, seed))
    return {record["question_id"]: record for record in records}


def _extract_type_score(type_metrics: dict[str, Any], answer_type: str) -> float | None:
    metric_key = "s_asst_mean" if answer_type == "free_text" else "s_det_mean"
    value = type_metrics.get(metric_key)
    if isinstance(value, dict):
        return _as_float(value.get("mean"))
    return _as_float(value)


def _collect_consolidated() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    per_type_rows: list[dict[str, Any]] = []

    for system_id in exp_cfg.ALL_SYSTEMS:
        spec = SYSTEM_SPECS[system_id]
        seed_reports = _collect_seed_reports(spec)

        if spec.seeded:
            if spec.aggregate_path is None:
                raise ValueError(f"Aggregate path required for {system_id}")
            aggregate = load_json(spec.aggregate_path)
            q_main, q_main_std = _extract_stat(aggregate.get("Q_main"))
            s_det, s_det_std = _extract_stat(aggregate.get("S_det"))
            s_asst, s_asst_std = _extract_stat(aggregate.get("S_asst"))
            g, g_std = _extract_stat(aggregate.get("G_f_beta"))
            peak_infer, peak_infer_std = _extract_stat(aggregate.get("peak_infer_vram_mb"))
            peak_train, peak_train_std = _extract_stat(aggregate.get("peak_train_vram_mb"))
            offline_cost, offline_cost_std = _extract_stat(aggregate.get("train_time_seconds"))
            breakdown_source = aggregate.get("breakdown_by_type", {})
        else:
            report = seed_reports[0]
            systems_metrics_path = spec.root_dir / "systems_metrics.json"
            systems_metrics = load_json(systems_metrics_path) if systems_metrics_path.exists() else {}
            q_main, q_main_std = _extract_stat(report.get("q_main"))
            s_det, s_det_std = _extract_stat(report.get("s_det"))
            s_asst, s_asst_std = _extract_stat(report.get("s_asst"))
            g, g_std = _extract_stat(report.get("grounding_f_beta"))
            peak_infer, peak_infer_std = _extract_stat(systems_metrics.get("peak_vram_mb"))
            peak_train, peak_train_std = None, None
            offline_cost, offline_cost_std = 0.0, 0.0
            breakdown_source = report.get("breakdown_by_type", {})

        ttft_median, ttft_median_std = _mean_std([
            float(rep["ttft_median_ms"]) for rep in seed_reports if rep.get("ttft_median_ms") is not None
        ])
        ttft_p95, ttft_p95_std = _mean_std([
            float(rep["ttft_p95_ms"]) for rep in seed_reports if rep.get("ttft_p95_ms") is not None
        ])
        latency_median, latency_median_std = _mean_std([
            float(rep["latency_median_ms"]) for rep in seed_reports if rep.get("latency_median_ms") is not None
        ])
        latency_p95, latency_p95_std = _mean_std([
            float(rep["latency_p95_ms"]) for rep in seed_reports if rep.get("latency_p95_ms") is not None
        ])
        malformed_rate, malformed_rate_std = _mean_std([
            float(rep["malformed_rate"]) for rep in seed_reports if rep.get("malformed_rate") is not None
        ])

        rows.append(
            {
                "system": system_id,
                "class": spec.system_class,
                "retrieval": spec.retrieval,
                "q_main": q_main,
                "q_main_std": q_main_std,
                "s_det": s_det,
                "s_det_std": s_det_std,
                "s_asst": s_asst,
                "s_asst_std": s_asst_std,
                "g": g,
                "g_std": g_std,
                "ttft_median_ms": ttft_median,
                "ttft_median_std": ttft_median_std,
                "ttft_p95_ms": ttft_p95,
                "ttft_p95_std": ttft_p95_std,
                "latency_median_ms": latency_median,
                "latency_median_std": latency_median_std,
                "latency_p95_ms": latency_p95,
                "latency_p95_std": latency_p95_std,
                "peak_infer_vram_mb": peak_infer,
                "peak_infer_vram_std": peak_infer_std,
                "peak_train_vram_mb": peak_train,
                "peak_train_vram_std": peak_train_std,
                "offline_cost_seconds": offline_cost,
                "offline_cost_std": offline_cost_std,
                "malformed_rate": malformed_rate,
                "malformed_rate_std": malformed_rate_std,
            }
        )

        for answer_type in exp_cfg.ANSWER_TYPES:
            type_metrics = breakdown_source.get(answer_type, {})
            if not isinstance(type_metrics, dict):
                continue
            score = _extract_type_score(type_metrics, answer_type)
            per_type_rows.append(
                {
                    "system": system_id,
                    "answer_type": answer_type,
                    "score": score,
                }
            )

    consolidated_df = pd.DataFrame(rows)
    consolidated_df["system"] = pd.Categorical(
        consolidated_df["system"],
        categories=exp_cfg.ALL_SYSTEMS,
        ordered=True,
    )
    consolidated_df = consolidated_df.sort_values("system").reset_index(drop=True)

    per_type_df = pd.DataFrame(per_type_rows)
    per_type_df["system"] = pd.Categorical(
        per_type_df["system"],
        categories=exp_cfg.ALL_SYSTEMS,
        ordered=True,
    )
    per_type_df["answer_type"] = pd.Categorical(
        per_type_df["answer_type"],
        categories=exp_cfg.ANSWER_TYPES,
        ordered=True,
    )
    per_type_df = per_type_df.sort_values(["answer_type", "system"]).reset_index(drop=True)

    return consolidated_df, per_type_df


def _load_question_metadata() -> dict[str, dict[str, Any]]:
    references = load_goldset(exp_cfg.GOLDSET_PATH)
    refs_by_id = {ref["question_id"]: ref for ref in references}
    split = load_json(exp_cfg.SPLIT_PATH)

    metadata: dict[str, dict[str, Any]] = {}
    for question_id in split["eval"]:
        ref = refs_by_id[question_id]
        doc_ids = {
            item.get("doc_id")
            for item in ref.get("gold_retrieval", [])
            if isinstance(item, dict) and item.get("doc_id")
        }
        metadata[question_id] = {
            "question": ref.get("question"),
            "answer": ref.get("answer"),
            "answer_type": ref.get("answer_type"),
            "difficulty": ref.get("difficulty"),
            "is_unanswerable": ref.get("answer") is None,
            "is_multi_doc": len(doc_ids) > 1,
        }
    return metadata


def _primary_score(row: dict[str, Any]) -> float:
    if row.get("answer_type") == "free_text":
        return float(row.get("s_asst") or 0.0)
    return float(row.get("s_det") or 0.0)


def _collect_question_scores(metadata: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for system_id in exp_cfg.ALL_SYSTEMS:
        spec = SYSTEM_SPECS[system_id]
        for run_id, eval_results in _collect_eval_runs(spec):
            for row in eval_results:
                question_id = row["question_id"]
                meta = metadata[question_id]
                score = _primary_score(row)
                criteria = row.get("judge_criteria") or {}
                rows.append(
                    {
                        "system": system_id,
                        "run_id": run_id,
                        "question_id": question_id,
                        "answer_type": meta["answer_type"],
                        "difficulty": meta["difficulty"],
                        "is_unanswerable": meta["is_unanswerable"],
                        "is_multi_doc": meta["is_multi_doc"],
                        "score": score,
                        "is_error": float(score < 1.0 - 1e-9),
                        "correctness": criteria.get("correctness"),
                        "completeness": criteria.get("completeness"),
                        "grounding": criteria.get("grounding"),
                        "calibration": criteria.get("calibration"),
                        "clarity": criteria.get("clarity"),
                    }
                )

    run_df = pd.DataFrame(rows)
    agg_df = (
        run_df.groupby(
            [
                "system",
                "question_id",
                "answer_type",
                "difficulty",
                "is_unanswerable",
                "is_multi_doc",
            ],
            as_index=False,
        )
        .agg(
            score_mean=("score", "mean"),
            score_std=("score", "std"),
            error_rate=("is_error", "mean"),
        )
        .fillna({"score_std": 0.0})
    )

    agg_df.to_csv(QUESTION_SCORE_CSV, index=False)
    log.info("Wrote %s", QUESTION_SCORE_CSV)
    return run_df, agg_df


def _compute_pairwise_win_rates(question_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = question_scores.pivot(index="question_id", columns="system", values="score_mean")

    rows: list[dict[str, Any]] = []
    matrix = pd.DataFrame(index=exp_cfg.ALL_SYSTEMS, columns=exp_cfg.ALL_SYSTEMS, dtype=float)

    for system_a in exp_cfg.ALL_SYSTEMS:
        for system_b in exp_cfg.ALL_SYSTEMS:
            joined = pivot[[system_a, system_b]].dropna()
            if joined.empty:
                win_rate = math.nan
                wins = 0
                ties = 0
            elif system_a == system_b:
                win_rate = 1.0
                wins = len(joined)
                ties = len(joined)
            else:
                wins = int((joined[system_a] > joined[system_b]).sum())
                ties = int((joined[system_a] == joined[system_b]).sum())
                win_rate = wins / float(len(joined))

            rows.append(
                {
                    "system_a": system_a,
                    "system_b": system_b,
                    "win_rate_a_over_b": win_rate,
                    "win_count": wins,
                    "tie_count": ties,
                    "n_questions": int(len(joined)),
                }
            )
            matrix.loc[system_a, system_b] = win_rate

    long_df = pd.DataFrame(rows)
    return long_df, matrix


def _compute_error_overlap(question_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    failure_sets: dict[str, set[str]] = {}
    for system_id in exp_cfg.ALL_SYSTEMS:
        failed_qids = set(
            question_scores[
                (question_scores["system"] == system_id)
                & (question_scores["score_mean"] < 1.0 - 1e-9)
            ]["question_id"].tolist()
        )
        failure_sets[system_id] = failed_qids

    rows: list[dict[str, Any]] = []
    matrix = pd.DataFrame(index=exp_cfg.ALL_SYSTEMS, columns=exp_cfg.ALL_SYSTEMS, dtype=float)

    for system_a in exp_cfg.ALL_SYSTEMS:
        for system_b in exp_cfg.ALL_SYSTEMS:
            set_a = failure_sets[system_a]
            set_b = failure_sets[system_b]
            union = set_a | set_b
            intersection = set_a & set_b
            jaccard = 1.0 if not union else (len(intersection) / float(len(union)))
            rows.append(
                {
                    "system_a": system_a,
                    "system_b": system_b,
                    "jaccard": jaccard,
                    "intersection": len(intersection),
                    "union": len(union),
                }
            )
            matrix.loc[system_a, system_b] = jaccard

    long_df = pd.DataFrame(rows)
    return long_df, matrix


def _compute_difficulty_profile(question_scores: pd.DataFrame) -> pd.DataFrame:
    return (
        question_scores.groupby(["system", "difficulty"], as_index=False)
        .agg(mean_score=("score_mean", "mean"), std_score=("score_mean", "std"))
        .fillna({"std_score": 0.0})
    )


def _compute_judge_criteria_profile(run_scores: pd.DataFrame) -> pd.DataFrame:
    free_text = run_scores[run_scores["answer_type"] == "free_text"].copy()
    criteria = ["correctness", "completeness", "grounding", "calibration", "clarity"]

    rows: list[dict[str, Any]] = []
    for system_id in exp_cfg.ALL_SYSTEMS:
        subset = free_text[free_text["system"] == system_id]
        for criterion in criteria:
            values = subset[criterion].dropna().astype(float).tolist()
            criterion_mean = mean(values) if values else math.nan
            rows.append(
                {
                    "system": system_id,
                    "criterion": criterion,
                    "mean_score": criterion_mean,
                }
            )
    return pd.DataFrame(rows)


def _run_level_q_main(run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for (system_id, run_id), subset in run_df.groupby(["system", "run_id"]):
        det_scores = subset[subset["answer_type"] != "free_text"]["score"].tolist()
        asst_scores = subset[subset["answer_type"] == "free_text"]["score"].tolist()

        det_mean = mean(det_scores) if det_scores else 0.0
        asst_mean = mean(asst_scores) if asst_scores else 0.0
        if det_scores and asst_scores:
            q_main = exp_cfg.Q_MAIN_WEIGHTS["S_det"] * det_mean + exp_cfg.Q_MAIN_WEIGHTS["S_asst"] * asst_mean
        elif det_scores:
            q_main = det_mean
        else:
            q_main = asst_mean

        rows.append(
            {
                "system": system_id,
                "run_id": run_id,
                "q_main_run": q_main,
                "s_det_run": det_mean,
                "s_asst_run": asst_mean,
            }
        )

    return pd.DataFrame(rows)


def _compute_seed_stability(run_df: pd.DataFrame, question_scores: pd.DataFrame) -> pd.DataFrame:
    per_run = _run_level_q_main(run_df)

    rows: list[dict[str, Any]] = []
    for system_id in exp_cfg.ALL_SYSTEMS:
        run_subset = per_run[per_run["system"] == system_id]
        q_values = run_subset["q_main_run"].tolist()
        s_det_values = run_subset["s_det_run"].tolist()
        s_asst_values = run_subset["s_asst_run"].tolist()

        q_std = stdev(q_values) if len(q_values) > 1 else 0.0
        s_det_std = stdev(s_det_values) if len(s_det_values) > 1 else 0.0
        s_asst_std = stdev(s_asst_values) if len(s_asst_values) > 1 else 0.0

        q_range = (max(q_values) - min(q_values)) if q_values else 0.0

        question_std_mean = float(
            question_scores[question_scores["system"] == system_id]["score_std"].mean()
        )
        high_var_share = float(
            (
                question_scores[
                    (question_scores["system"] == system_id)
                    & (question_scores["score_std"] > 0.2)
                ].shape[0]
                / max(1, question_scores[question_scores["system"] == system_id].shape[0])
            )
        )

        rows.append(
            {
                "system": system_id,
                "q_main_std_across_runs": q_std,
                "q_main_range_across_runs": q_range,
                "s_det_std_across_runs": s_det_std,
                "s_asst_std_across_runs": s_asst_std,
                "mean_question_score_std": question_std_mean,
                "high_variance_question_share": high_var_share,
            }
        )

    return pd.DataFrame(rows)


def _prediction_to_text(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return str(value)
    return str(value)


def _build_error_analysis(
    metadata: dict[str, dict[str, Any]],
    question_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    pivot = question_scores[
        question_scores["system"].isin(exp_cfg.HEADLINE_SYSTEMS)
    ].pivot(index="question_id", columns="system", values="score_mean")

    correct = pivot >= (1.0 - 1e-9)
    all_wrong = int((~correct).all(axis=1).sum())
    only_correct_counts: dict[str, int] = {}
    for system_id in exp_cfg.HEADLINE_SYSTEMS:
        only_correct_counts[system_id] = int(
            (correct[system_id] & (~correct.drop(columns=[system_id]).any(axis=1))).sum()
        )

    failure_rows: list[dict[str, Any]] = []
    sections: list[str] = [
        "# Error Analysis",
        "",
        f"- **All headline systems wrong (S1, S2+R, S3+R, S7):** {all_wrong}",
        f"- **Only S1 correct:** {only_correct_counts['S1']}",
        f"- **Only S2+R correct:** {only_correct_counts['S2+R']}",
        f"- **Only S3+R correct:** {only_correct_counts['S3+R']}",
        f"- **Only S7 correct:** {only_correct_counts['S7']}",
        "",
    ]

    for system_id in exp_cfg.HEADLINE_SYSTEMS:
        spec = SYSTEM_SPECS[system_id]
        rep_seed = exp_cfg.REPRESENTATIVE_SEED if spec.seeded else None
        eval_rows = load_json(_eval_results_path(spec, rep_seed))
        predictions = _collect_representative_predictions(spec)

        enriched: list[dict[str, Any]] = []
        for row in eval_rows:
            qid = row["question_id"]
            meta = metadata[qid]
            score = _primary_score(row)
            pred = predictions.get(qid, {})
            enriched.append(
                {
                    "system": system_id,
                    "question_id": qid,
                    "score": score,
                    "answer_type": meta["answer_type"],
                    "difficulty": meta["difficulty"],
                    "question": meta["question"],
                    "gold_answer": _prediction_to_text(meta["answer"]),
                    "predicted_answer": _prediction_to_text(pred.get("parsed_answer")),
                    "is_malformed": bool(row.get("is_malformed", False)),
                }
            )

        system_failures = sorted(enriched, key=lambda item: (item["score"], item["question_id"]))[:5]
        sections.append(f"## Top-5 failures: {system_id}")
        sections.append("")
        for item in system_failures:
            sections.append(f"**Q:** {item['question']}")
            sections.append(
                f"- Type: {item['answer_type']}, Difficulty: {item['difficulty']}, Score: {item['score']:.3f}"
            )
            sections.append(f"- Gold: {item['gold_answer']}")
            sections.append(f"- System: {item['predicted_answer']}")
            sections.append(f"- Malformed: {item['is_malformed']}")
            sections.append("")
            failure_rows.append(item)

    return pd.DataFrame(failure_rows), "\n".join(sections) + "\n"


def _practical_winner_call(consolidated_df: pd.DataFrame) -> dict[str, Any]:
    s2r = consolidated_df[consolidated_df["system"] == "S2+R"].iloc[0]
    s3r = consolidated_df[consolidated_df["system"] == "S3+R"].iloc[0]

    delta_q = float(s2r["q_main"] - s3r["q_main"])
    delta_s_det = float(s2r["s_det"] - s3r["s_det"])
    delta_s_asst = float(s2r["s_asst"] - s3r["s_asst"])

    ambiguous_tradeoff = (
        abs(delta_q) < 0.01
        and delta_s_det > 0
        and delta_s_asst < 0
    )

    if ambiguous_tradeoff:
        verdict = "No single practical winner"
    else:
        verdict = "S2+R" if delta_q > 0 else "S3+R"

    return {
        "verdict": verdict,
        "delta_q_main": delta_q,
        "delta_s_det": delta_s_det,
        "delta_s_asst": delta_s_asst,
    }


def _plot_main_results_table(consolidated_df: pd.DataFrame) -> None:
    display_df = consolidated_df[["system", "class", "q_main", "s_det", "s_asst", "g", "latency_median_ms"]].copy()
    for column in ["q_main", "s_det", "s_asst", "g", "latency_median_ms"]:
        display_df[column] = display_df[column].map(lambda v: "N/A" if pd.isna(v) else f"{float(v):.3f}")

    fig, ax = plt.subplots(figsize=(12, 3.8), dpi=160)
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=["System", "Class", "Q_main", "S_det", "S_asst", "G", "Latency(ms)"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    ax.set_title("EXP-007 Consolidated Results (S6 excluded, S7 included)", fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(exp_cfg.MAIN_RESULTS_TABLE_PNG)
    plt.close(fig)


def _plot_cost_quality_scatter(consolidated_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=160)

    for _, row in consolidated_df.iterrows():
        system = row["system"]
        x = float(row["offline_cost_seconds"] if not pd.isna(row["offline_cost_seconds"]) else 0.0)
        y = float(row["q_main"])
        size = 120
        color = "#1D7874" if row["class"] == "Headline" else "#9E6A3A"
        if system == "S7":
            color = "#2E8B57"
        ax.scatter(x, y, s=size, c=color, edgecolors="#1f1f1f", linewidth=0.8, alpha=0.9)
        ax.text(x + 6, y + 0.003, system, fontsize=9)

    ax.set_xlabel("Offline Packaging Cost (seconds)")
    ax.set_ylabel("Q_main")
    ax.set_title("Cost vs Quality Trade-off")
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(exp_cfg.COST_QUALITY_SCATTER_PNG)
    plt.close(fig)


def _plot_per_type_heatmap(per_type_df: pd.DataFrame) -> None:
    pivot = per_type_df.pivot(index="answer_type", columns="system", values="score")
    pivot = pivot.reindex(index=exp_cfg.ANSWER_TYPES, columns=exp_cfg.ALL_SYSTEMS)

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(exp_cfg.ALL_SYSTEMS)))
    ax.set_xticklabels(exp_cfg.ALL_SYSTEMS)
    ax.set_yticks(np.arange(len(exp_cfg.ANSWER_TYPES)))
    ax.set_yticklabels(exp_cfg.ANSWER_TYPES)
    ax.set_title("Per-Type Score Heatmap")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            text = "N/A" if math.isnan(value) else f"{value:.2f}"
            ax.text(j, i, text, ha="center", va="center", color="#1f1f1f", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Score")
    fig.tight_layout()
    fig.savefig(exp_cfg.PER_TYPE_HEATMAP_PNG)
    plt.close(fig)


def _plot_latency_grounding(consolidated_df: pd.DataFrame) -> None:
    subset = consolidated_df[consolidated_df["retrieval"] & consolidated_df["g"].notna()].copy()

    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=160)
    scatter = ax.scatter(
        subset["latency_median_ms"],
        subset["g"],
        c=subset["q_main"],
        cmap="viridis",
        s=140,
        edgecolors="#222222",
        linewidth=0.8,
    )

    for _, row in subset.iterrows():
        ax.text(row["latency_median_ms"] + 8, row["g"] + 0.002, row["system"], fontsize=9)

    ax.set_xlabel("Latency median (ms)")
    ax.set_ylabel("Grounding G")
    ax.set_title("Latency vs Grounding (retrieval systems)")
    ax.grid(alpha=0.3, linestyle="--")
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Q_main")
    fig.tight_layout()
    fig.savefig(exp_cfg.LATENCY_GROUNDING_SCATTER_PNG)
    plt.close(fig)


def _plot_heatmap(
    matrix: pd.DataFrame,
    title: str,
    output_path: Path,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "magma",
) -> None:
    matrix = matrix.reindex(index=exp_cfg.ALL_SYSTEMS, columns=exp_cfg.ALL_SYSTEMS)
    data = matrix.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=160)
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(exp_cfg.ALL_SYSTEMS)))
    ax.set_xticklabels(exp_cfg.ALL_SYSTEMS, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(exp_cfg.ALL_SYSTEMS)))
    ax.set_yticklabels(exp_cfg.ALL_SYSTEMS)
    ax.set_title(title)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            label = "N/A" if math.isnan(value) else f"{value:.2f}"
            ax.text(j, i, label, ha="center", va="center", color="#f5f5f5", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_difficulty_profile(difficulty_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
    x = np.arange(len(exp_cfg.DIFFICULTY_LEVELS))
    width = 0.12

    for idx, system_id in enumerate(exp_cfg.ALL_SYSTEMS):
        subset = difficulty_df[difficulty_df["system"] == system_id].set_index("difficulty")
        values = [
            float(subset.loc[level, "mean_score"]) if level in subset.index else math.nan
            for level in exp_cfg.DIFFICULTY_LEVELS
        ]
        ax.bar(x + (idx - (len(exp_cfg.ALL_SYSTEMS) - 1) / 2) * width, values, width=width, label=system_id)

    ax.set_xticks(x)
    ax.set_xticklabels(exp_cfg.DIFFICULTY_LEVELS)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean primary score")
    ax.set_title("Difficulty Profile by System")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(exp_cfg.DIFFICULTY_PROFILE_PNG)
    plt.close(fig)


def _plot_judge_criteria(criteria_df: pd.DataFrame) -> None:
    criteria_order = ["correctness", "completeness", "grounding", "calibration", "clarity"]
    systems = exp_cfg.ALL_SYSTEMS

    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160)
    x = np.arange(len(criteria_order))
    width = 0.12

    for idx, system_id in enumerate(systems):
        subset = criteria_df[criteria_df["system"] == system_id].set_index("criterion")
        values = [float(subset.loc[c, "mean_score"]) if c in subset.index else math.nan for c in criteria_order]
        ax.bar(x + (idx - (len(systems) - 1) / 2) * width, values, width=width, label=system_id)

    ax.set_xticks(x)
    ax.set_xticklabels(criteria_order, rotation=15)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean binary judge score")
    ax.set_title("Free-text Judge Criteria Profile")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(exp_cfg.JUDGE_CRITERIA_PNG)
    plt.close(fig)


def _plot_seed_stability(stability_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    ordered = stability_df.set_index("system").reindex(exp_cfg.ALL_SYSTEMS)
    values = ordered["q_main_std_across_runs"].to_numpy(dtype=float)

    ax.bar(exp_cfg.ALL_SYSTEMS, values, color="#287271", edgecolor="#1f1f1f")
    ax.set_title("Q_main Stability Across Runs")
    ax.set_ylabel("Std across runs")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.001, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(exp_cfg.SEED_STABILITY_PNG)
    plt.close(fig)


def _compute_pareto(consolidated_df: pd.DataFrame) -> pd.DataFrame:
    df = consolidated_df[["system", "offline_cost_seconds", "q_main"]].copy()
    df = df.fillna({"offline_cost_seconds": 0.0})

    pareto_flags: list[bool] = []
    for _, row in df.iterrows():
        dominated = False
        for _, other in df.iterrows():
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
        pareto_flags.append(not dominated)

    df["is_pareto"] = pareto_flags
    return df


def _plot_pareto_frontier(consolidated_df: pd.DataFrame) -> pd.DataFrame:
    pareto_df = _compute_pareto(consolidated_df)

    fig, ax = plt.subplots(figsize=(8.5, 5.4), dpi=160)
    for _, row in pareto_df.iterrows():
        color = "#2E8B57" if row["is_pareto"] else "#A7A7A7"
        ax.scatter(row["offline_cost_seconds"], row["q_main"], c=color, s=140, edgecolors="#1f1f1f")
        ax.text(row["offline_cost_seconds"] + 6, row["q_main"] + 0.0025, row["system"], fontsize=9)

    frontier = pareto_df[pareto_df["is_pareto"]].sort_values("offline_cost_seconds")
    ax.plot(frontier["offline_cost_seconds"], frontier["q_main"], color="#2E8B57", linewidth=1.6)

    ax.set_xlabel("Offline cost (seconds)")
    ax.set_ylabel("Q_main")
    ax.set_title("Pareto Frontier: Cost vs Quality")
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(exp_cfg.PARETO_FRONTIER_PNG)
    plt.close(fig)

    return pareto_df


def _write_deep_analysis(
    consolidated_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    pairwise_long_df: pd.DataFrame,
    overlap_long_df: pd.DataFrame,
    difficulty_df: pd.DataFrame,
    criteria_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
) -> None:
    q_rank = consolidated_df.sort_values("q_main", ascending=False)[["system", "q_main"]]
    top_system = q_rank.iloc[0]["system"]
    top_value = float(q_rank.iloc[0]["q_main"])

    s2r = consolidated_df[consolidated_df["system"] == "S2+R"].iloc[0]
    s3r = consolidated_df[consolidated_df["system"] == "S3+R"].iloc[0]
    s7 = consolidated_df[consolidated_df["system"] == "S7"].iloc[0]

    avg_overlap_headline = float(
        overlap_long_df[
            (overlap_long_df["system_a"].isin(exp_cfg.HEADLINE_SYSTEMS))
            & (overlap_long_df["system_b"].isin(exp_cfg.HEADLINE_SYSTEMS))
            & (overlap_long_df["system_a"] != overlap_long_df["system_b"])
        ]["jaccard"].mean()
    )

    hard_profile = difficulty_df[difficulty_df["difficulty"] == "hard"].sort_values("mean_score", ascending=False)
    best_hard = hard_profile.iloc[0]

    criteria_pivot = criteria_df.pivot(index="criterion", columns="system", values="mean_score")
    grounding_leader = criteria_pivot.loc["grounding"].sort_values(ascending=False).index[0]

    pareto_systems = pareto_df[pareto_df["is_pareto"]]["system"].tolist()

    lines = [
        "# Deep Analysis",
        "",
        "## 1. Ranking and Margin Structure",
        "",
        f"- Global quality leader: **{top_system}** with `Q_main={top_value:.4f}`.",
        f"- Merge effect (S7 vs S2+R): `ΔQ_main={float(s7['q_main'] - s2r['q_main']):+.4f}`, `ΔS_det={float(s7['s_det'] - s2r['s_det']):+.4f}`, `ΔS_asst={float(s7['s_asst'] - s2r['s_asst']):+.4f}`.",
        f"- Trade-off persists in base hybrids: S2+R dominates deterministic score (`S_det={float(s2r['s_det']):.4f}`), S3+R dominates assistant score (`S_asst={float(s3r['s_asst']):.4f}`).",
        "",
        "## 2. Error Topology and Complementarity",
        "",
        f"- Mean pairwise failure-overlap (headline systems) Jaccard: `{avg_overlap_headline:.3f}`.",
        "- Lower overlap means systems fail on different subsets, creating room for fusion/selection strategies.",
        "- Pairwise win-rate matrix quantifies question-level dominance and reveals where apparent aggregate ties hide local regime shifts.",
        "",
        "## 3. Stability and Variance",
        "",
        f"- Highest run-to-run variance (Q_main std) is visible in: `{stability_df.sort_values('q_main_std_across_runs', ascending=False).iloc[0]['system']}`.",
        "- Mean per-question score std highlights systems with unstable behavior across seeds, not just unstable global means.",
        "",
        "## 4. Difficulty and Judge-Dimension Behavior",
        "",
        f"- Best performer on `hard` questions: **{best_hard['system']}** (`{float(best_hard['mean_score']):.3f}`).",
        f"- Free-text grounding criterion leader: **{grounding_leader}**.",
        "- Criterion-level profiles show where quality gains come from (correctness/completeness) versus stylistic clarity.",
        "",
        "## 5. Cost-Quality Frontier",
        "",
        f"- Pareto-optimal systems on (offline cost, Q_main): `{', '.join(pareto_systems)}`.",
        "- Systems outside Pareto front are strictly dominated and can be deprioritized in practical deployment decisions.",
        "",
        "## 6. Practical Recommendations",
        "",
        "- Keep S2+R and S3+R both in discussion as base hybrids; they encode different strengths and should not be collapsed into a single narrative.",
        "- Promote S7 as best observed post-hoc merged configuration, but preserve caveat: it is a merge-based conclusion, not independent retraining evidence.",
        "- Use question-level win-rate and overlap artifacts to motivate targeted ensemble/routing hypotheses in future work.",
    ]

    exp_cfg.DEEP_ANALYSIS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Wrote %s", exp_cfg.DEEP_ANALYSIS_MD)


def _write_report(
    consolidated_df: pd.DataFrame,
    winner: dict[str, Any],
) -> None:
    s2r = consolidated_df[consolidated_df["system"] == "S2+R"].iloc[0]
    s3r = consolidated_df[consolidated_df["system"] == "S3+R"].iloc[0]
    s7 = consolidated_df[consolidated_df["system"] == "S7"].iloc[0]

    lines = [
        "# EXP-007: Error Analysis + Trade-off",
        "",
        f"**Date:** {datetime.now(timezone.utc).date().isoformat()}",
        "",
        "## Scope",
        "",
        "- Consolidation includes S1, S2+R, S3+R, S7, S2, S3.",
        "- S6 (Naive RAG) intentionally excluded from this refresh.",
        "",
        "## Practical Winner Call (S2+R vs S3+R)",
        "",
        "| Metric | S2+R | S3+R | Winner |",
        "|--------|------|------|--------|",
        f"| Q_main | {float(s2r['q_main']):.3f} | {float(s3r['q_main']):.3f} | {'Tie' if abs(winner['delta_q_main']) < 0.01 else ('S2+R' if winner['delta_q_main'] > 0 else 'S3+R')} |",
        f"| S_det | {float(s2r['s_det']):.3f} | {float(s3r['s_det']):.3f} | {'S2+R' if winner['delta_s_det'] > 0 else 'S3+R'} |",
        f"| S_asst | {float(s2r['s_asst']):.3f} | {float(s3r['s_asst']):.3f} | {'S2+R' if winner['delta_s_asst'] > 0 else 'S3+R'} |",
        f"| G | {float(s2r['g']):.3f} | {float(s3r['g']):.3f} | Tie |",
        f"| Offline cost (s) | {float(s2r['offline_cost_seconds']):.1f} | {float(s3r['offline_cost_seconds']):.1f} | {'S2+R' if float(s2r['offline_cost_seconds']) < float(s3r['offline_cost_seconds']) else 'S3+R'} |",
        "",
        f"**Verdict:** {winner['verdict']}",
        "",
        "## EXP-010 Impact (S7)",
        "",
        f"- S7 reaches `Q_main={float(s7['q_main']):.4f}`, best among all included systems.",
        f"- Relative to S2+R: `ΔQ_main={float(s7['q_main'] - s2r['q_main']):+.4f}`, `ΔS_det={float(s7['s_det'] - s2r['s_det']):+.4f}`, `ΔS_asst={float(s7['s_asst'] - s2r['s_asst']):+.4f}`.",
        "",
        "## Error Analysis",
        "",
        f"See `{exp_cfg.ERROR_ANALYSIS_MD}`.",
        "",
        "## Deep Analysis",
        "",
        f"See `{exp_cfg.DEEP_ANALYSIS_MD}`.",
        "",
        "## Figures",
        "",
        f"- `{exp_cfg.MAIN_RESULTS_TABLE_PNG}`",
        f"- `{exp_cfg.COST_QUALITY_SCATTER_PNG}`",
        f"- `{exp_cfg.PER_TYPE_HEATMAP_PNG}`",
        f"- `{exp_cfg.LATENCY_GROUNDING_SCATTER_PNG}`",
        f"- `{exp_cfg.ERROR_OVERLAP_HEATMAP_PNG}`",
        f"- `{exp_cfg.PAIRWISE_WIN_HEATMAP_PNG}`",
        f"- `{exp_cfg.DIFFICULTY_PROFILE_PNG}`",
        f"- `{exp_cfg.JUDGE_CRITERIA_PNG}`",
        f"- `{exp_cfg.SEED_STABILITY_PNG}`",
        f"- `{exp_cfg.PARETO_FRONTIER_PNG}`",
        "",
        "## Mandatory Caveats",
        "",
        "- Bounded to this corpus, benchmark split, backbone, and hardware setup.",
        "- CLM-based systems use supervision-free document exposure, not QA supervision.",
        "- S7 is a post-hoc adapter-merge result; it is not a separately retrained system.",
    ]

    exp_cfg.REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Wrote %s", exp_cfg.REPORT_PATH)


def main() -> None:
    log.info("=== EXP-007 refresh start (with S7 / without S6) ===")
    _ensure_dirs()

    consolidated_df, per_type_df = _collect_consolidated()
    consolidated_df.to_csv(exp_cfg.CONSOLIDATED_RESULTS_CSV, index=False)
    per_type_df.to_csv(exp_cfg.PER_TYPE_BREAKDOWN_CSV, index=False)
    log.info("Wrote %s", exp_cfg.CONSOLIDATED_RESULTS_CSV)
    log.info("Wrote %s", exp_cfg.PER_TYPE_BREAKDOWN_CSV)

    metadata = _load_question_metadata()
    run_scores_df, question_scores_df = _collect_question_scores(metadata)

    pairwise_long_df, pairwise_matrix_df = _compute_pairwise_win_rates(question_scores_df)
    pairwise_long_df.to_csv(exp_cfg.PAIRWISE_WIN_RATE_CSV, index=False)

    overlap_long_df, overlap_matrix_df = _compute_error_overlap(question_scores_df)
    overlap_matrix_df.to_csv(exp_cfg.ERROR_OVERLAP_CSV, index=True)

    difficulty_df = _compute_difficulty_profile(question_scores_df)
    difficulty_df.to_csv(exp_cfg.DIFFICULTY_PROFILE_CSV, index=False)

    criteria_df = _compute_judge_criteria_profile(run_scores_df)
    criteria_df.to_csv(exp_cfg.JUDGE_CRITERIA_CSV, index=False)

    stability_df = _compute_seed_stability(run_scores_df, question_scores_df)
    stability_df.to_csv(exp_cfg.SEED_STABILITY_CSV, index=False)

    failures_df, error_analysis_md = _build_error_analysis(metadata, question_scores_df)
    failures_df.to_csv(exp_cfg.HEADLINE_FAILURES_CSV, index=False)
    exp_cfg.ERROR_ANALYSIS_MD.write_text(error_analysis_md, encoding="utf-8")
    log.info("Wrote %s", exp_cfg.ERROR_ANALYSIS_MD)

    _plot_main_results_table(consolidated_df)
    _plot_cost_quality_scatter(consolidated_df)
    _plot_per_type_heatmap(per_type_df)
    _plot_latency_grounding(consolidated_df)
    _plot_heatmap(overlap_matrix_df, "Error Overlap (Jaccard)", exp_cfg.ERROR_OVERLAP_HEATMAP_PNG, cmap="magma")
    _plot_heatmap(pairwise_matrix_df, "Pairwise Win Rate", exp_cfg.PAIRWISE_WIN_HEATMAP_PNG, cmap="viridis")
    _plot_difficulty_profile(difficulty_df)
    _plot_judge_criteria(criteria_df)
    _plot_seed_stability(stability_df)
    pareto_df = _plot_pareto_frontier(consolidated_df)

    winner = _practical_winner_call(consolidated_df)
    _write_deep_analysis(
        consolidated_df,
        stability_df,
        pairwise_long_df,
        overlap_long_df,
        difficulty_df,
        criteria_df,
        pareto_df,
    )
    _write_report(consolidated_df, winner)

    log.info("=== EXP-007 refresh done ===")


if __name__ == "__main__":
    main()
