#!/usr/bin/env python3
"""Paired stratified bootstrap intervals for the headline RQ1 contrasts.

The analysis pairs systems by question ID and resamples deterministic and
free-text questions separately so that every bootstrap replicate preserves the
definition Q_main = 0.7 * S_det + 0.3 * S_asst. For trained systems, each
question's score is averaged across the three training seeds before resampling.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results" / "EXP-007" / "rq1_paired_bootstrap.json"
Q_MAIN_WEIGHTS = {"deterministic": 0.7, "free_text": 0.3}

SYSTEM_REPORTS = {
    "Base-RAG": [ROOT / "results" / "EXP-002" / "eval_report.json"],
    "RAFT-RAG": [
        ROOT / "results" / "EXP-003" / f"seed_{seed}" / "eval_report.json"
        for seed in (42, 123, 777)
    ],
    "CLM-RAG": [
        ROOT / "results" / "EXP-004b" / f"seed_{seed}" / "eval_report.json"
        for seed in (42, 123, 777)
    ],
}

CONTRASTS = (
    ("RAFT-RAG", "Base-RAG"),
    ("CLM-RAG", "Base-RAG"),
    ("RAFT-RAG", "CLM-RAG"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20_260_825)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def percentile(values: list[float], probability: float) -> float:
    """Return a linearly interpolated percentile for sorted finite values."""
    if not values:
        raise ValueError("Cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def score_question(item: dict[str, Any]) -> float:
    s_det = item.get("s_det")
    s_asst = item.get("s_asst")
    if (s_det is None) == (s_asst is None):
        raise ValueError(
            f"Question {item.get('question_id')} must have exactly one applicable score"
        )
    score = float(s_det if s_det is not None else s_asst)
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"Question {item.get('question_id')} has score {score}")
    return score


def q_main(scores: dict[str, float], answer_types: dict[str, str]) -> float:
    deterministic = [
        score for qid, score in scores.items() if answer_types[qid] != "free_text"
    ]
    free_text = [
        score for qid, score in scores.items() if answer_types[qid] == "free_text"
    ]
    return (
        Q_MAIN_WEIGHTS["deterministic"] * statistics.fmean(deterministic)
        + Q_MAIN_WEIGHTS["free_text"] * statistics.fmean(free_text)
    )


def load_run(
    path: Path,
) -> tuple[dict[str, float], dict[str, str], float]:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = report.get("question_scores")
    if not isinstance(rows, list) or len(rows) != 50:
        raise ValueError(f"Expected 50 question scores in {path}, found {len(rows or [])}")

    scores: dict[str, float] = {}
    answer_types: dict[str, str] = {}
    for item in rows:
        qid = str(item["question_id"])
        if qid in scores:
            raise ValueError(f"Duplicate question ID {qid} in {path}")
        scores[qid] = score_question(item)
        answer_types[qid] = str(item["answer_type"])

    recomputed = q_main(scores, answer_types)
    reported = float(report["q_main"])
    if not math.isclose(recomputed, reported, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"Q_main mismatch in {path}: reported {reported}, recomputed {recomputed}"
        )
    return scores, answer_types, reported


def load_profiles() -> tuple[
    dict[str, dict[str, float]],
    dict[str, str],
    dict[str, list[float]],
]:
    profiles: dict[str, dict[str, float]] = {}
    reference_types: dict[str, str] | None = None
    run_q_main: dict[str, list[float]] = {}

    for system, paths in SYSTEM_REPORTS.items():
        run_scores: list[dict[str, float]] = []
        run_q_main[system] = []
        for path in paths:
            scores, answer_types, reported_q_main = load_run(path)
            if reference_types is None:
                reference_types = answer_types
            elif answer_types != reference_types:
                raise ValueError(f"Question IDs or answer types differ in {path}")
            run_scores.append(scores)
            run_q_main[system].append(reported_q_main)

        assert reference_types is not None
        profiles[system] = {
            qid: statistics.fmean(run[qid] for run in run_scores)
            for qid in reference_types
        }

    return profiles, reference_types or {}, run_q_main


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def run_analysis(replicates: int, random_seed: int) -> dict[str, Any]:
    if replicates < 1_000:
        raise ValueError("Use at least 1,000 bootstrap replicates")

    profiles, answer_types, run_q_main = load_profiles()
    deterministic_ids = sorted(
        qid for qid, answer_type in answer_types.items() if answer_type != "free_text"
    )
    free_text_ids = sorted(
        qid for qid, answer_type in answer_types.items() if answer_type == "free_text"
    )
    if (len(deterministic_ids), len(free_text_ids)) != (37, 13):
        raise ValueError(
            "Expected 37 deterministic and 13 free-text evaluation questions, found "
            f"{len(deterministic_ids)} and {len(free_text_ids)}"
        )

    differences: dict[tuple[str, str], tuple[list[float], list[float]]] = {}
    for system_a, system_b in CONTRASTS:
        differences[(system_a, system_b)] = (
            [profiles[system_a][qid] - profiles[system_b][qid] for qid in deterministic_ids],
            [profiles[system_a][qid] - profiles[system_b][qid] for qid in free_text_ids],
        )

    bootstrap_values = {contrast: [] for contrast in CONTRASTS}
    rng = random.Random(random_seed)
    for _ in range(replicates):
        deterministic_sample = [
            rng.randrange(len(deterministic_ids)) for _ in deterministic_ids
        ]
        free_text_sample = [rng.randrange(len(free_text_ids)) for _ in free_text_ids]
        for contrast in CONTRASTS:
            deterministic_diff, free_text_diff = differences[contrast]
            delta = (
                Q_MAIN_WEIGHTS["deterministic"]
                * statistics.fmean(deterministic_diff[index] for index in deterministic_sample)
                + Q_MAIN_WEIGHTS["free_text"]
                * statistics.fmean(free_text_diff[index] for index in free_text_sample)
            )
            bootstrap_values[contrast].append(delta)

    contrasts: list[dict[str, Any]] = []
    for system_a, system_b in CONTRASTS:
        deterministic_diff, free_text_diff = differences[(system_a, system_b)]
        observed = (
            Q_MAIN_WEIGHTS["deterministic"] * statistics.fmean(deterministic_diff)
            + Q_MAIN_WEIGHTS["free_text"] * statistics.fmean(free_text_diff)
        )
        samples = bootstrap_values[(system_a, system_b)]
        low = percentile(samples, 0.025)
        high = percentile(samples, 0.975)
        contrasts.append(
            {
                "system_a": system_a,
                "system_b": system_b,
                "delta_q_main": round(observed, 12),
                "ci_95": [round(low, 12), round(high, 12)],
                "bootstrap_standard_error": round(statistics.stdev(samples), 12),
                "includes_zero": low <= 0.0 <= high,
            }
        )

    return {
        "analysis": "RQ1 paired stratified bootstrap",
        "metric": {
            "name": "Q_main",
            "definition": "0.7 * mean deterministic score + 0.3 * mean free-text score",
        },
        "question_counts": {
            "total": len(answer_types),
            "deterministic": len(deterministic_ids),
            "free_text": len(free_text_ids),
        },
        "bootstrap": {
            "replicates": replicates,
            "confidence_level": 0.95,
            "interval": "percentile",
            "random_seed": random_seed,
            "pairing_key": "question_id",
            "strata": ["deterministic", "free_text"],
            "trained_system_score": "per-question arithmetic mean across seeds 42, 123, and 777",
        },
        "inferential_scope": (
            "Question-level sampling uncertainty within the fixed 50-question holdout, "
            "conditional on the observed training runs; it does not estimate transfer to "
            "an independent benchmark or additional training seeds."
        ),
        "source_reports": {
            system: [relative(path) for path in paths]
            for system, paths in SYSTEM_REPORTS.items()
        },
        "run_q_main": {
            system: [round(value, 12) for value in values]
            for system, values in run_q_main.items()
        },
        "system_q_main": {
            system: round(q_main(profile, answer_types), 12)
            for system, profile in profiles.items()
        },
        "contrasts": contrasts,
    }


def main() -> None:
    args = parse_args()
    result = run_analysis(args.replicates, args.random_seed)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    for contrast in result["contrasts"]:
        low, high = contrast["ci_95"]
        print(
            f"{contrast['system_a']} - {contrast['system_b']}: "
            f"{contrast['delta_q_main']:+.3f} "
            f"(95% CI [{low:+.3f}, {high:+.3f}])"
        )


if __name__ == "__main__":
    main()
