"""EXP-004b: S3+R CLM + Retrieval (Headline, Architecture v9.0)."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import config as cfg
import torch
from src.data.io import load_goldset, load_json, save_json
from src.evaluation.runner import EvalRunner
from src.evaluation.s2_runner import prepare_s2_eval_samples, run_s2_generation
from src.evaluation.seed_stats import (
    aggregate_seed_summaries,
    eval_report_to_summary,
    save_seed_aggregate,
)

logging.basicConfig(
    level=logging.INFO,
    format=cfg.LOG_FORMAT,
    datefmt=cfg.LOG_DATE_FORMAT,
)
logger = logging.getLogger(__name__)

_EXP_CONFIG_SPEC = importlib.util.spec_from_file_location(
    "exp004b_config",
    Path(__file__).with_name("config.py"),
)
if _EXP_CONFIG_SPEC is None or _EXP_CONFIG_SPEC.loader is None:
    raise RuntimeError("Failed to load experiment config")
exp_cfg = importlib.util.module_from_spec(_EXP_CONFIG_SPEC)
_EXP_CONFIG_SPEC.loader.exec_module(exp_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-004b: S3+R CLM + retrieval")
    parser.add_argument("--smoke", action="store_true", help="Smoke test with 2 questions")
    parser.add_argument("--seed", type=int, default=None, help="Run a single seed")
    args = parser.parse_args()

    selected_seeds = _resolve_seeds(args.seed, args.smoke)
    logger.info(
        "=== EXP-004b start === mode=%s, seeds=%s",
        "smoke" if args.smoke else "full", selected_seeds,
    )

    # Verify adapters exist
    for seed in selected_seeds:
        adapter_dir = exp_cfg.CLM_MODELS_DIR / f"seed_{seed}"
        if not adapter_dir.exists():
            raise FileNotFoundError(f"CLM adapter not found: {adapter_dir} — run EXP-004 first")

    # ── Phase 1: Retrieval (cached once) ──
    logger.info("── Phase 1: Running S1 retrieval pipeline on eval questions ──")
    eval_refs = _load_eval_refs(limit=exp_cfg.SMOKE_EVAL_QUESTIONS if args.smoke else None)
    logger.info("Eval refs: %d questions", len(eval_refs))

    eval_samples = prepare_s2_eval_samples(
        refs=eval_refs,
        corpus_dir=cfg.CORPUS_DIR,
        index_output_dir=exp_cfg.INDEX_OUTPUT_DIR,
    )
    logger.info("Retrieval complete: %d cached samples", len(eval_samples))
    _free_gpu()

    # ── Phase 2: Inference per seed ──
    eval_runner = EvalRunner(
        goldset_path=cfg.GOLDSET_PATH,
        split_path=cfg.DATA_SPLITS / "split_v1.json",
        judge_model=cfg.JUDGE_MODEL,
        judge_reasoning=cfg.JUDGE_REASONING,
        grounding_beta=cfg.GROUNDING_BETA,
        q_main_weights=cfg.Q_MAIN_WEIGHTS,
    )

    seed_summaries = []
    for seed in selected_seeds:
        adapter_dir = exp_cfg.CLM_MODELS_DIR / f"seed_{seed}"
        result_dir = exp_cfg.RESULTS_DIR / ("smoke" if args.smoke else f"seed_{seed}")

        _free_gpu()
        logger.info("── Inference seed %d → adapter=%s ──", seed, adapter_dir)
        predictions, peak_infer_vram_mb = run_s2_generation(
            model_name=cfg.BACKBONE_MODEL,
            adapter_dir=adapter_dir,
            eval_samples=eval_samples,
            max_new_tokens=exp_cfg.MAX_NEW_TOKENS,
        )
        logger.info(
            "Seed %d inference done: %d predictions, peak VRAM %.1f MB",
            seed, len(predictions), peak_infer_vram_mb or 0,
        )

        save_json(
            [p.model_dump() for p in predictions],
            result_dir / f"predictions_seed_{seed}.json",
        )

        logger.info("Scoring seed %d with judge...", seed)
        report = eval_runner.evaluate(
            predictions=predictions,
            system_id="S3+R",
            experiment_id=exp_cfg.EXPERIMENT_ID,
            split="eval",
            compute_grounding_flag=True,
        )
        eval_runner.save_report(report, result_dir)
        logger.info(
            "Seed %d scored: Q_main=%.4f, S_det=%.4f, S_asst=%.4f, G=%.4f",
            seed, report.q_main, report.s_det, report.s_asst,
            report.grounding_f_beta or 0,
        )

        save_json(
            {"peak_infer_vram_mb": peak_infer_vram_mb},
            result_dir / "systems_metrics.json",
        )

        # CLM is inference-only; use EXP-004 training metrics
        exp004_metrics_path = (
            cfg.RESULTS_DIR / "EXP-004_clm" / f"seed_{seed}" / "training_metrics.json"
        )
        if exp004_metrics_path.exists():
            saved = load_json(exp004_metrics_path)
            train_time_seconds = float(saved["train_time_seconds"])
            peak_train_vram_mb = float(saved["peak_train_vram_mb"])
        else:
            train_time_seconds = 0.0
            peak_train_vram_mb = None

        seed_summaries.append(
            eval_report_to_summary(
                seed=seed,
                report=report,
                train_time_seconds=train_time_seconds,
                peak_train_vram_mb=peak_train_vram_mb,
                peak_infer_vram_mb=peak_infer_vram_mb,
            )
        )
        _free_gpu()

    # ── Aggregation ──
    if seed_summaries:
        aggregate_summary = aggregate_seed_summaries(
            seed_summaries,
            s1_report_path=exp_cfg.EXP002_REPORT_PATH,
        )
        # Delta vs S2+R
        if exp_cfg.EXP003_AGGREGATE_PATH.exists():
            s2r_agg = load_json(exp_cfg.EXP003_AGGREGATE_PATH)
            aggregate_summary["delta_vs_s2r"] = {
                "Q_main": aggregate_summary["Q_main"]["mean"] - s2r_agg["Q_main"]["mean"],
                "S_det": aggregate_summary["S_det"]["mean"] - s2r_agg["S_det"]["mean"],
                "S_asst": aggregate_summary["S_asst"]["mean"] - s2r_agg["S_asst"]["mean"],
            }
        # Delta vs S3 (CLM no retrieval)
        if exp_cfg.EXP004_CLM_AGGREGATE_PATH.exists():
            s3_agg = load_json(exp_cfg.EXP004_CLM_AGGREGATE_PATH)
            aggregate_summary["delta_vs_s3"] = {
                "Q_main": aggregate_summary["Q_main"]["mean"] - s3_agg["Q_main"]["mean"],
                "S_det": aggregate_summary["S_det"]["mean"] - s3_agg["S_det"]["mean"],
                "S_asst": aggregate_summary["S_asst"]["mean"] - s3_agg["S_asst"]["mean"],
            }

        if args.smoke:
            save_seed_aggregate(aggregate_summary, exp_cfg.RESULTS_DIR / "smoke")
        else:
            save_seed_aggregate(aggregate_summary, exp_cfg.RESULTS_DIR)
            _write_report(aggregate_summary)


def _load_eval_refs(limit: int | None) -> list[dict]:
    refs = load_goldset(cfg.GOLDSET_PATH)
    refs_by_id = {ref["question_id"]: ref for ref in refs}
    split = load_json(cfg.DATA_SPLITS / "split_v1.json")
    eval_refs = [refs_by_id[qid] for qid in split["eval"]]
    return eval_refs[:limit] if limit is not None else eval_refs


def _resolve_seeds(seed: int | None, smoke: bool) -> list[int]:
    if seed is not None:
        return [seed]
    if smoke:
        return [cfg.DEFAULT_SEED]
    return list(exp_cfg.TRAIN_SEEDS)


def _write_report(summary: dict) -> None:
    seed_results = summary["seed_results"]
    delta_s1 = summary["delta_vs_s1"]
    delta_s2r = summary.get("delta_vs_s2r")
    delta_s3 = summary.get("delta_vs_s3")

    lines = [
        "# Experiment Report: EXP-004b - S3+R CLM + Retrieval (Headline)",
        "",
        f"**Date:** {datetime.now(timezone.utc).date().isoformat()}",
        "**Status:** Completed",
        "",
        "## 1. Goal",
        "",
        "- Evaluate CLM adapter as retrieval-conditioned generator inside S1 RAG pipeline.",
        "- Symmetric comparison with S2+R: same retrieval, same PEFT arch, different training signal.",
        "- No new training — inference-only using CLM adapters from EXP-004.",
        "",
        "## 2. Setup",
        "",
        f"- Seeds: {', '.join(str(item['seed']) for item in seed_results)}",
        f"- Backbone: `{cfg.BACKBONE_MODEL}` + CLM adapter",
        "- Retrieval: S1 pipeline (Qdrant hybrid + reranker + evidence compression)",
        "- Prompt: RAG template (retrieved context + question)",
        "",
        "## 3. Results",
        "",
    ]
    for metric_name in ("Q_main", "S_det", "S_asst"):
        metric = summary.get(metric_name)
        if metric is None:
            continue
        lines.append(f"- {metric_name}: {metric['mean']:.4f} ± {metric['std']:.4f}")
    g = summary.get("grounding_f_beta")
    if g:
        lines.append(f"- G (F_β=2.5): {g['mean']:.4f} ± {g['std']:.4f}")

    lines.extend([
        "",
        "## 4. Per-Seed Summary",
        "",
        "| Seed | Q_main | S_det | S_asst | G | Peak infer VRAM MB |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for item in seed_results:
        infer_vram = item["peak_infer_vram_mb"]
        g_val = item.get("grounding_f_beta")
        lines.append(
            "| {seed} | {q_main:.4f} | {s_det:.4f} | {s_asst:.4f} | {g} | {vram} |".format(
                seed=item["seed"],
                q_main=item["q_main"],
                s_det=item["s_det"],
                s_asst=item["s_asst"],
                g=f"{g_val:.4f}" if g_val is not None else "N/A",
                vram=f"{infer_vram:.1f}" if infer_vram is not None else "N/A",
            )
        )

    lines.extend([
        "",
        "## 5. Delta vs S1 (RAG baseline, no adapter)",
        "",
        f"- Q_main delta: {delta_s1['Q_main']:+.4f}",
        f"- S_det delta: {delta_s1['S_det']:+.4f}",
        f"- S_asst delta: {delta_s1['S_asst']:+.4f}",
    ])

    if delta_s2r:
        lines.extend([
            "",
            "## 6. Delta vs S2+R (RAFT + retrieval)",
            "",
            f"- Q_main delta: {delta_s2r['Q_main']:+.4f}",
            f"- S_det delta: {delta_s2r['S_det']:+.4f}",
            f"- S_asst delta: {delta_s2r['S_asst']:+.4f}",
        ])

    if delta_s3:
        lines.extend([
            "",
            "## 7. Delta vs S3 (CLM no retrieval — retrieval contribution)",
            "",
            f"- Q_main delta: {delta_s3['Q_main']:+.4f}",
            f"- S_det delta: {delta_s3['S_det']:+.4f}",
            f"- S_asst delta: {delta_s3['S_asst']:+.4f}",
            "",
            "Positive delta = retrieval helps CLM system.",
        ])

    lines.extend([
        "",
        "## 8. Breakdown By Answer Type",
        "",
    ])
    for answer_type, metrics in sorted(summary["breakdown_by_type"].items()):
        lines.append(f"### {answer_type}")
        lines.append("")
        for metric_name, stats in sorted(metrics.items()):
            lines.append(f"- {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        lines.append("")

    lines.extend([
        "## 9. Artifacts",
        "",
        f"- Aggregate summary: `{exp_cfg.RESULTS_DIR / 'aggregate_summary.json'}`",
        f"- Seed outputs: `{exp_cfg.RESULTS_DIR}`",
        f"- CLM adapters: `{exp_cfg.CLM_MODELS_DIR}`",
    ])
    exp_cfg.REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Report written to %s", exp_cfg.REPORT_PATH)


def _free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
