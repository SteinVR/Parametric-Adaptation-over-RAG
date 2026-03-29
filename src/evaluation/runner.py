"""EvalRunner: the reusable evaluation orchestrator.

Every experiment calls:
    runner = EvalRunner(goldset_path, split_path)
    report = runner.evaluate(predictions, system_id="S1", experiment_id="EXP-002")
    runner.save_report(report, output_dir)
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from statistics import mean, median
from typing import Any

from src.data.io import load_goldset, load_json, save_json
from src.evaluation.deterministic import score_deterministic
from src.evaluation.grounding import compute_grounding, expand_gold_retrieval
from src.evaluation.judge import LLMJudge
from src.evaluation.schemas import EvalReport, PageRef, Prediction, QuestionScore

logger = logging.getLogger(__name__)


class EvalRunner:
    """Scores predictions against goldset references."""

    def __init__(
        self,
        goldset_path: Path,
        split_path: Path,
        judge_model: str = "gpt-5.4-mini",
        judge_reasoning: str = "medium",
        grounding_beta: float = 2.5,
        q_main_weights: dict[str, float] | None = None,
    ) -> None:
        self.grounding_beta = grounding_beta
        weights = q_main_weights or {"S_det": 0.7, "S_asst": 0.3}
        self.w_det = weights["S_det"]
        self.w_asst = weights["S_asst"]

        # Load goldset as {question_id: reference}
        refs = load_goldset(goldset_path)
        self.refs_by_id: dict[str, dict[str, Any]] = {r["question_id"]: r for r in refs}

        # Load split
        split_data = load_json(split_path)
        self.splits: dict[str, list[str]] = split_data

        # Lazy-init judge (only if free_text questions exist)
        self._judge: LLMJudge | None = None
        self._judge_model = judge_model
        self._judge_reasoning = judge_reasoning

    @property
    def judge(self) -> LLMJudge:
        if self._judge is None:
            self._judge = LLMJudge(model=self._judge_model, reasoning=self._judge_reasoning)
        return self._judge

    def evaluate(
        self,
        predictions: list[Prediction],
        system_id: str,
        experiment_id: str,
        split: str = "eval",
        compute_grounding_flag: bool = True,
    ) -> EvalReport:
        """Score predictions on the specified split."""
        split_ids = set(self.splits.get(split, []))
        preds_by_id = {p.question_id: p for p in predictions}

        question_scores: list[QuestionScore] = []
        det_scores: list[float] = []
        asst_scores: list[float] = []
        grounding_scores: list[float] = []

        # Only score questions we have predictions for (supports subset mode)
        scored_ids = split_ids & set(preds_by_id.keys())
        if len(scored_ids) < len(split_ids):
            logger.info("Scoring %d/%d split questions (subset mode)", len(scored_ids), len(split_ids))

        for qid in sorted(scored_ids):
            ref = self.refs_by_id.get(qid)
            pred = preds_by_id.get(qid)
            if ref is None or pred is None:
                continue

            answer_type = ref["answer_type"]
            is_unanswerable = ref["answer"] is None

            # Deterministic scoring (all types except free_text)
            s_det: float | None = None
            if answer_type != "free_text":
                if pred.is_malformed:
                    s_det = 0.0
                else:
                    s_det = score_deterministic(
                        pred.parsed_answer, ref["answer"], answer_type, is_unanswerable
                    )
                det_scores.append(s_det)

            # LLM judge scoring (free_text only)
            s_asst: float | None = None
            judge_criteria: dict[str, int] | None = None
            if answer_type == "free_text":
                if pred.is_malformed:
                    judge_criteria = {k: 0 for k in ["correctness", "completeness", "grounding", "calibration", "clarity"]}
                else:
                    judge_criteria = self.judge.score(
                        question=ref["question"],
                        reference_answer=str(ref["answer"]),
                        system_response=pred.raw_output,
                    )
                s_asst = mean(judge_criteria.values())
                asst_scores.append(s_asst)

            # Grounding
            g_prec: float | None = None
            g_rec: float | None = None
            g_fbeta: float | None = None
            if compute_grounding_flag:
                gold_pages = expand_gold_retrieval(ref.get("gold_retrieval", []))
                g_prec, g_rec, g_fbeta = compute_grounding(
                    pred.predicted_pages, gold_pages, self.grounding_beta
                )
                grounding_scores.append(g_fbeta)

            question_scores.append(
                QuestionScore(
                    question_id=qid,
                    answer_type=answer_type,
                    difficulty=ref.get("difficulty", "unknown"),
                    is_unanswerable=is_unanswerable,
                    s_det=s_det,
                    s_asst=s_asst,
                    judge_criteria=judge_criteria,
                    grounding_precision=g_prec,
                    grounding_recall=g_rec,
                    grounding_f_beta=g_fbeta,
                    is_malformed=pred.is_malformed,
                )
            )

        # Aggregates
        agg_s_det = mean(det_scores) if det_scores else 0.0
        agg_s_asst = mean(asst_scores) if asst_scores else 0.0
        q_main = self.w_det * agg_s_det + self.w_asst * agg_s_asst
        agg_grounding = mean(grounding_scores) if grounding_scores else None

        # Breakdown by answer_type
        breakdown = self._compute_breakdown(question_scores)

        # Systems metrics
        ttfts = [p.ttft_ms for p in predictions if p.ttft_ms is not None and p.question_id in split_ids]
        latencies = [p.latency_ms for p in predictions if p.latency_ms is not None and p.question_id in split_ids]
        malformed_count = sum(1 for qs in question_scores if qs.is_malformed)
        total = len(question_scores) or 1

        return EvalReport(
            system_id=system_id,
            experiment_id=experiment_id,
            question_scores=question_scores,
            q_main=q_main,
            s_det=agg_s_det,
            s_asst=agg_s_asst,
            grounding_f_beta=agg_grounding,
            breakdown_by_type=breakdown,
            ttft_median_ms=median(ttfts) if ttfts else None,
            ttft_p95_ms=_percentile(ttfts, 0.95) if ttfts else None,
            latency_median_ms=median(latencies) if latencies else None,
            latency_p95_ms=_percentile(latencies, 0.95) if latencies else None,
            malformed_rate=malformed_count / total,
        )

    def save_report(self, report: EvalReport, output_dir: Path) -> None:
        """Save eval_results.json and eval_summary.csv."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Per-question results
        save_json(
            [qs.model_dump() for qs in report.question_scores],
            output_dir / "eval_results.json",
        )

        # Summary CSV
        rows: list[dict[str, Any]] = []
        # Aggregate row
        rows.append({
            "scope": "aggregate",
            "Q_main": f"{report.q_main:.4f}",
            "S_det": f"{report.s_det:.4f}",
            "S_asst": f"{report.s_asst:.4f}",
            "G_f_beta": f"{report.grounding_f_beta:.4f}" if report.grounding_f_beta is not None else "N/A",
            "TTFT_median_ms": f"{report.ttft_median_ms:.1f}" if report.ttft_median_ms else "N/A",
            "latency_median_ms": f"{report.latency_median_ms:.1f}" if report.latency_median_ms else "N/A",
            "malformed_rate": f"{report.malformed_rate:.2%}",
        })
        # Per answer_type rows
        for atype, metrics in report.breakdown_by_type.items():
            row: dict[str, Any] = {"scope": atype}
            for k, v in metrics.items():
                row[k] = f"{v:.4f}" if isinstance(v, float) else str(v)
            rows.append(row)

        csv_path = output_dir / "eval_summary.csv"
        if rows:
            fieldnames = list(rows[0].keys())
            # Collect all keys from all rows
            for row in rows:
                for k in row:
                    if k not in fieldnames:
                        fieldnames.append(k)
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)

        # Full report JSON
        save_json(report.model_dump(), output_dir / "eval_report.json")

        logger.info("Eval report saved to %s", output_dir)

    @staticmethod
    def _compute_breakdown(scores: list[QuestionScore]) -> dict[str, dict[str, float]]:
        """Compute metrics breakdown by answer_type."""
        by_type: dict[str, list[QuestionScore]] = {}
        for qs in scores:
            by_type.setdefault(qs.answer_type, []).append(qs)

        breakdown: dict[str, dict[str, float]] = {}
        for atype, group in by_type.items():
            metrics: dict[str, float] = {"count": len(group)}
            if atype == "free_text":
                vals = [qs.s_asst for qs in group if qs.s_asst is not None]
                metrics["s_asst_mean"] = mean(vals) if vals else 0.0
            else:
                vals = [qs.s_det for qs in group if qs.s_det is not None]
                metrics["s_det_mean"] = mean(vals) if vals else 0.0

            g_vals = [qs.grounding_f_beta for qs in group if qs.grounding_f_beta is not None]
            if g_vals:
                metrics["grounding_f_beta_mean"] = mean(g_vals)

            malformed = sum(1 for qs in group if qs.is_malformed)
            metrics["malformed_rate"] = malformed / len(group) if group else 0.0

            breakdown[atype] = metrics

        # Unanswerable cross-cutting breakdown (per SPEC-evaluation.md)
        unanswerable = [qs for qs in scores if qs.is_unanswerable]
        if unanswerable:
            u_det = [qs.s_det for qs in unanswerable if qs.s_det is not None]
            breakdown["_unanswerable"] = {
                "count": len(unanswerable),
                "s_det_mean": mean(u_det) if u_det else 0.0,
                "malformed_rate": sum(1 for qs in unanswerable if qs.is_malformed) / len(unanswerable),
            }

        return breakdown


def _percentile(values: list[float], p: float) -> float:
    """Compute p-th percentile (0-1 scale)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(p * (len(sorted_vals) - 1))
    return sorted_vals[idx]
