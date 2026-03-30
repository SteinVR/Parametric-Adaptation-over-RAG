"""EXP-003b: S2 QLoRA closed-book baseline (Axis 1)."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
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
from src.data.closed_book import build_closed_book_examples, save_closed_book_jsonl
from src.data.io import load_goldset, load_json, save_json
from src.data.raft import RaftExample, load_raft_jsonl
from src.evaluation.runner import EvalRunner
from src.evaluation.s2_closed_runner import run_s2_closed_generation
from src.evaluation.seed_stats import (
    aggregate_seed_summaries,
    eval_report_to_summary,
    save_seed_aggregate,
)
from src.training.qlora import QloraTrainingConfig, train_qlora_adapter

logging.basicConfig(
    level=logging.INFO,
    format=cfg.LOG_FORMAT,
    datefmt=cfg.LOG_DATE_FORMAT,
)
logger = logging.getLogger(__name__)
_EXP_CONFIG_SPEC = importlib.util.spec_from_file_location(
    "exp003b_config",
    Path(__file__).with_name("config.py"),
)
if _EXP_CONFIG_SPEC is None or _EXP_CONFIG_SPEC.loader is None:
    raise RuntimeError("Failed to load experiment config")
exp_cfg = importlib.util.module_from_spec(_EXP_CONFIG_SPEC)
_EXP_CONFIG_SPEC.loader.exec_module(exp_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-003b: S2 QLoRA closed-book")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny end-to-end smoke flow")
    parser.add_argument("--seed", type=int, default=None, help="Run a single training seed")
    parser.add_argument("--all-seeds", action="store_true", help="Run all configured seeds")
    parser.add_argument("--skip-train", action="store_true", help="Skip adapter training")
    parser.add_argument("--skip-eval", action="store_true", help="Skip eval after training")
    args = parser.parse_args()

    selected_seeds = _resolve_seeds(args.seed, args.all_seeds, args.smoke)
    examples = _build_or_load_dataset()
    if args.smoke:
        examples = examples[: exp_cfg.SMOKE_TRAIN_EXAMPLES]

    per_seed_paths: dict[int, dict[str, object]] = {}
    seed_summaries = []
    for seed in selected_seeds:
        adapter_dir = exp_cfg.MODELS_DIR / ("smoke" if args.smoke else f"seed_{seed}")
        result_dir = exp_cfg.RESULTS_DIR / ("smoke" if args.smoke else f"seed_{seed}")
        train_result = None
        if not args.skip_train:
            training_config = QloraTrainingConfig(
                model_name=cfg.BACKBONE_MODEL,
                output_dir=adapter_dir,
                seed=seed,
                rank=cfg.QLORA_RANK,
                alpha=cfg.QLORA_ALPHA,
                dropout=cfg.QLORA_DROPOUT,
                target_modules=tuple(cfg.QLORA_TARGET_MODULES),
                learning_rate=cfg.QLORA_LR,
                max_seq_length=cfg.QLORA_MAX_SEQ_LEN,
                per_device_batch_size=exp_cfg.TRAIN_MICRO_BATCH_SIZE,
                gradient_accumulation_steps=exp_cfg.GRADIENT_ACCUMULATION_STEPS,
                epochs=1 if args.smoke else cfg.QLORA_EPOCHS,
                warmup_ratio=cfg.QLORA_WARMUP_RATIO,
                weight_decay=cfg.QLORA_WEIGHT_DECAY,
                max_steps=1 if args.smoke else -1,
            )
            train_result = train_qlora_adapter(examples, training_config)
            save_json(
                {
                    "seed": seed,
                    "train_time_seconds": train_result.train_time_seconds,
                    "peak_train_vram_mb": train_result.peak_vram_mb,
                },
                result_dir / "training_metrics.json",
            )
            _free_gpu()
        per_seed_paths[seed] = {
            "adapter_dir": adapter_dir,
            "result_dir": result_dir,
            "train_result": train_result,
        }

        if args.skip_eval:
            continue

    if args.skip_eval:
        return

    _free_gpu()
    eval_refs = _load_eval_refs(limit=exp_cfg.SMOKE_EVAL_QUESTIONS if args.smoke else None)
    eval_runner = EvalRunner(
        goldset_path=cfg.GOLDSET_PATH,
        split_path=cfg.DATA_SPLITS / "split_v1.json",
        judge_model=cfg.JUDGE_MODEL,
        judge_reasoning=cfg.JUDGE_REASONING,
        grounding_beta=cfg.GROUNDING_BETA,
        q_main_weights=cfg.Q_MAIN_WEIGHTS,
    )

    for seed in selected_seeds:
        adapter_dir = per_seed_paths[seed]["adapter_dir"]
        result_dir = per_seed_paths[seed]["result_dir"]
        train_result = per_seed_paths[seed]["train_result"]

        _free_gpu()
        predictions, peak_infer_vram_mb = run_s2_closed_generation(
            model_name=cfg.BACKBONE_MODEL,
            adapter_dir=adapter_dir,
            eval_refs=eval_refs,
            max_new_tokens=exp_cfg.MAX_NEW_TOKENS,
        )
        save_json([p.model_dump() for p in predictions], result_dir / "predictions.json")
        report = eval_runner.evaluate(
            predictions=predictions,
            system_id="S2",
            experiment_id=exp_cfg.EXPERIMENT_ID,
            split="eval",
            compute_grounding_flag=False,
        )
        eval_runner.save_report(report, result_dir)
        save_json(
            {"peak_infer_vram_mb": peak_infer_vram_mb},
            result_dir / "systems_metrics.json",
        )
        if train_result is not None:
            train_time_seconds = train_result.train_time_seconds
            peak_train_vram_mb = train_result.peak_vram_mb
        else:
            saved_metrics_path = result_dir / "training_metrics.json"
            if saved_metrics_path.exists():
                saved = load_json(saved_metrics_path)
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

    if seed_summaries:
        aggregate_summary = aggregate_seed_summaries(
            seed_summaries,
            s1_report_path=exp_cfg.EXP002_REPORT_PATH,
        )
        # Add delta vs S2+R if available
        if exp_cfg.EXP003_AGGREGATE_PATH.exists():
            s2r_agg = load_json(exp_cfg.EXP003_AGGREGATE_PATH)
            aggregate_summary["delta_vs_s2r"] = {
                "Q_main": aggregate_summary["Q_main"]["mean"] - s2r_agg["Q_main"]["mean"],
                "S_det": aggregate_summary["S_det"]["mean"] - s2r_agg["S_det"]["mean"],
                "S_asst": aggregate_summary["S_asst"]["mean"] - s2r_agg["S_asst"]["mean"],
            }
        if args.smoke:
            save_seed_aggregate(aggregate_summary, exp_cfg.RESULTS_DIR / "smoke")
        else:
            save_seed_aggregate(aggregate_summary, exp_cfg.RESULTS_DIR)
            _write_report(aggregate_summary)


def _build_or_load_dataset() -> list[RaftExample]:
    expected_count = cfg.S2_TRAIN_SIZE
    if exp_cfg.CLOSED_BOOK_DATASET_PATH.exists():
        logger.info("Loading frozen closed-book dataset from %s", exp_cfg.CLOSED_BOOK_DATASET_PATH)
        examples = load_raft_jsonl(exp_cfg.CLOSED_BOOK_DATASET_PATH)
        if len(examples) != expected_count:
            raise ValueError(f"Expected {expected_count} examples, found {len(examples)}")
        return examples

    examples = build_closed_book_examples(
        goldset_path=cfg.GOLDSET_PATH,
        split_path=cfg.DATA_SPLITS / "split_v1.json",
    )
    save_closed_book_jsonl(examples, exp_cfg.CLOSED_BOOK_DATASET_PATH)
    return examples


def _load_eval_refs(limit: int | None) -> list[dict]:
    refs = load_goldset(cfg.GOLDSET_PATH)
    refs_by_id = {ref["question_id"]: ref for ref in refs}
    split = load_json(cfg.DATA_SPLITS / "split_v1.json")
    eval_refs = [refs_by_id[qid] for qid in split["eval"]]
    return eval_refs[:limit] if limit is not None else eval_refs


def _resolve_seeds(seed: int | None, all_seeds: bool, smoke: bool) -> list[int]:
    if seed is not None:
        return [seed]
    if smoke:
        return [cfg.DEFAULT_SEED]
    if all_seeds:
        return list(exp_cfg.TRAIN_SEEDS)
    return list(exp_cfg.TRAIN_SEEDS)


def _write_report(summary: dict) -> None:
    seed_results = summary["seed_results"]
    delta_s1 = summary["delta_vs_s1"]
    delta_s2r = summary.get("delta_vs_s2r")

    lines = [
        "# Experiment Report: EXP-003b - S2 QLoRA Closed-Book (Axis 1)",
        "",
        f"**Date:** {datetime.now(timezone.utc).date().isoformat()}",
        "**Status:** Completed",
        "",
        "## 1. Hypothesis",
        "",
        "- S2 closed-book tests whether supervised QA-pair training injects document knowledge into a 2B model.",
        "- Expected to underperform S1 (RAG) — model cannot memorize 176 pages from 150 QA pairs.",
        "",
        "## 2. Setup",
        "",
        f"- Seeds: {', '.join(str(item['seed']) for item in seed_results)}",
        f"- Dataset: `{exp_cfg.CLOSED_BOOK_DATASET_PATH}`",
        f"- Backbone: `{cfg.BACKBONE_MODEL}`",
        "- **No retrieval at inference.**",
        "",
        "## 3. Results",
        "",
    ]
    for metric_name in ("Q_main", "S_det", "S_asst"):
        metric = summary.get(metric_name)
        if metric is None:
            continue
        lines.append(f"- {metric_name}: {metric['mean']:.4f} ± {metric['std']:.4f}")
    lines.extend([
        "",
        "## 4. Per-seed Summary",
        "",
        "| Seed | Q_main | S_det | S_asst | Train sec | Peak train VRAM MB | Peak infer VRAM MB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for item in seed_results:
        train_vram = item["peak_train_vram_mb"]
        infer_vram = item["peak_infer_vram_mb"]
        lines.append(
            "| {seed} | {q_main:.4f} | {s_det:.4f} | {s_asst:.4f} | {train_time:.1f} | {train_vram} | {infer_vram} |".format(
                seed=item["seed"],
                q_main=item["q_main"],
                s_det=item["s_det"],
                s_asst=item["s_asst"],
                train_time=item["train_time_seconds"],
                train_vram=f"{train_vram:.1f}" if train_vram is not None else "N/A",
                infer_vram=f"{infer_vram:.1f}" if infer_vram is not None else "N/A",
            )
        )
    lines.extend([
        "",
        "## 5. Delta vs S1 (Axis 1)",
        "",
        f"- Q_main delta: {delta_s1['Q_main']:.4f}",
        f"- S_det delta: {delta_s1['S_det']:.4f}",
        f"- S_asst delta: {delta_s1['S_asst']:.4f}",
    ])
    if delta_s2r:
        lines.extend([
            "",
            "## 6. Delta vs S2+R (Axis 2: retrieval contribution)",
            "",
            f"- Q_main delta: {delta_s2r['Q_main']:.4f}",
            f"- S_det delta: {delta_s2r['S_det']:.4f}",
            f"- S_asst delta: {delta_s2r['S_asst']:.4f}",
            "",
            "Negative delta = S2 closed-book worse than S2+R → retrieval helps.",
        ])
    lines.extend([
        "",
        "## 7. Breakdown By Answer Type",
        "",
    ])
    for answer_type, metrics in sorted(summary["breakdown_by_type"].items()):
        lines.append(f"### {answer_type}")
        lines.append("")
        for metric_name, stats in sorted(metrics.items()):
            lines.append(f"- {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        lines.append("")
    lines.extend([
        "## 8. Artifacts",
        "",
        f"- Aggregate summary: `{exp_cfg.RESULTS_DIR / 'aggregate_summary.json'}`",
        f"- Seed outputs: `{exp_cfg.RESULTS_DIR}`",
        f"- Adapters: `{exp_cfg.MODELS_DIR}`",
    ])
    exp_cfg.REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
