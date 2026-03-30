"""EXP-004: S3 CLM continued pretraining (Architecture v9.0)."""

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
from src.d2l.corpus import load_frozen_corpus_documents
from src.d2l.runner import run_d2l_no_retrieval_generation
from src.data.io import load_goldset, load_json, save_json
from src.evaluation.runner import EvalRunner
from src.evaluation.seed_stats import (
    aggregate_seed_summaries,
    eval_report_to_summary,
    save_seed_aggregate,
)
from src.training.clm import train_clm_adapter
from src.training.qlora import QloraTrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format=cfg.LOG_FORMAT,
    datefmt=cfg.LOG_DATE_FORMAT,
)
logger = logging.getLogger(__name__)

_EXP_CONFIG_SPEC = importlib.util.spec_from_file_location(
    "exp004_config",
    Path(__file__).with_name("config.py"),
)
if _EXP_CONFIG_SPEC is None or _EXP_CONFIG_SPEC.loader is None:
    raise RuntimeError("Failed to load experiment config")
exp_cfg = importlib.util.module_from_spec(_EXP_CONFIG_SPEC)
_EXP_CONFIG_SPEC.loader.exec_module(exp_cfg)


# %% ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-004: S3 CLM continued pretraining")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny end-to-end smoke flow")
    parser.add_argument("--seed", type=int, default=None, help="Run a single training seed")
    parser.add_argument("--all-seeds", action="store_true", help="Run all configured seeds")
    parser.add_argument("--skip-train", action="store_true", help="Skip adapter training")
    parser.add_argument("--skip-eval", action="store_true", help="Skip eval after training")
    args = parser.parse_args()

    selected_seeds = _resolve_seeds(args.seed, args.all_seeds, args.smoke)
    logger.info(
        "=== EXP-004 CLM start === mode=%s, seeds=%s, skip_train=%s, skip_eval=%s",
        "smoke" if args.smoke else "full", selected_seeds, args.skip_train, args.skip_eval,
    )

    corpus_texts = _load_corpus_texts()
    logger.info(
        "Loaded %d documents, total %d chars, per-doc sizes: %s",
        len(corpus_texts),
        sum(len(t) for t in corpus_texts),
        [len(t) for t in corpus_texts],
    )

    # ── Training phase ──
    per_seed_paths: dict[int, dict[str, object]] = {}
    for seed in selected_seeds:
        adapter_dir = exp_cfg.MODELS_DIR / ("smoke" if args.smoke else f"seed_{seed}")
        result_dir = exp_cfg.RESULTS_DIR / ("smoke" if args.smoke else f"seed_{seed}")
        train_result = None

        if not args.skip_train:
            logger.info("── Training seed %d → adapter_dir=%s", seed, adapter_dir)
            training_config = QloraTrainingConfig(
                model_name=cfg.BACKBONE_MODEL,
                output_dir=adapter_dir,
                seed=seed,
                rank=cfg.QLORA_RANK,
                alpha=cfg.QLORA_ALPHA,
                dropout=cfg.QLORA_DROPOUT,
                target_modules=tuple(cfg.QLORA_TARGET_MODULES),
                learning_rate=exp_cfg.CLM_LR,
                max_seq_length=exp_cfg.CLM_MAX_SEQ_LENGTH,
                per_device_batch_size=exp_cfg.TRAIN_MICRO_BATCH_SIZE,
                gradient_accumulation_steps=exp_cfg.GRADIENT_ACCUMULATION_STEPS,
                epochs=1 if args.smoke else exp_cfg.CLM_EPOCHS,
                warmup_ratio=exp_cfg.CLM_WARMUP_RATIO,
                weight_decay=cfg.QLORA_WEIGHT_DECAY,
                max_steps=1 if args.smoke else -1,
            )
            train_result = train_clm_adapter(corpus_texts, training_config)
            logger.info(
                "Seed %d trained: %.1fs, peak VRAM %.1f MB, adapter at %s",
                seed, train_result.train_time_seconds,
                train_result.peak_vram_mb or 0, train_result.adapter_dir,
            )
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
        return

    # ── Evaluation phase ──
    logger.info("── Entering evaluation phase ──")
    _free_gpu()
    eval_refs = _load_eval_refs(limit=exp_cfg.SMOKE_EVAL_QUESTIONS if args.smoke else None)
    logger.info("Eval refs loaded: %d questions", len(eval_refs))
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
        adapter_dir = per_seed_paths[seed]["adapter_dir"]
        result_dir = per_seed_paths[seed]["result_dir"]
        train_result = per_seed_paths[seed]["train_result"]

        _free_gpu()
        logger.info("── Inference seed %d → adapter=%s", seed, adapter_dir)
        predictions, peak_infer_vram_mb = run_d2l_no_retrieval_generation(
            model_name=cfg.BACKBONE_MODEL,
            adapter_dir=adapter_dir,
            eval_refs=eval_refs,
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
            system_id="S3",
            experiment_id=exp_cfg.EXPERIMENT_ID,
            split="eval",
            compute_grounding_flag=False,
        )
        eval_runner.save_report(report, result_dir)
        logger.info(
            "Seed %d scored: Q_main=%.4f, S_det=%.4f, S_asst=%.4f",
            seed, report.q_main, report.s_det, report.s_asst,
        )
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
        # Delta vs S2 closed-book
        if exp_cfg.EXP003B_AGGREGATE_PATH.exists():
            s2_agg = load_json(exp_cfg.EXP003B_AGGREGATE_PATH)
            aggregate_summary["delta_vs_s2"] = {
                "Q_main": aggregate_summary["Q_main"]["mean"] - s2_agg["Q_main"]["mean"],
                "S_det": aggregate_summary["S_det"]["mean"] - s2_agg["S_det"]["mean"],
                "S_asst": aggregate_summary["S_asst"]["mean"] - s2_agg["S_asst"]["mean"],
            }

        if args.smoke:
            save_seed_aggregate(aggregate_summary, exp_cfg.RESULTS_DIR / "smoke")
        else:
            save_seed_aggregate(aggregate_summary, exp_cfg.RESULTS_DIR)
            _write_report(aggregate_summary)


# %% ── Helpers ───────────────────────────────────────────────────────────────

def _load_corpus_texts() -> list[str]:
    documents = load_frozen_corpus_documents(
        corpus_dir=cfg.CORPUS_DIR,
        goldset_path=cfg.GOLDSET_PATH,
    )
    return [doc.full_text for doc in documents]


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
    return list(exp_cfg.TRAIN_SEEDS)


def _write_report(summary: dict) -> None:
    seed_results = summary["seed_results"]
    delta_s1 = summary["delta_vs_s1"]
    delta_s2r = summary.get("delta_vs_s2r")
    delta_s2 = summary.get("delta_vs_s2")

    lines = [
        "# Experiment Report: EXP-004 - S3 CLM Continued Pretraining",
        "",
        f"**Date:** {datetime.now(timezone.utc).date().isoformat()}",
        "**Status:** Completed",
        "",
        "## 1. Goal",
        "",
        "- Train QLoRA adapter on corpus document text with causal LM loss (next-token prediction).",
        "- Evaluate as pure parametric control (no retrieval).",
        "- Compare with S1 (RAG), S2+R (RAFT), and S2 (closed-book).",
        "",
        "## 2. Historical Note",
        "",
        "- EXP-004 originally used Doc-to-LoRA (D2L) hypernetwork. D2L was non-viable:",
        "  documents exceeded hypernetwork context, chunk workaround yielded Q_main=0.210.",
        "- Architecture pivoted to CLM in v9.0. D2L results archived.",
        "",
        "## 3. Setup",
        "",
        f"- Seeds: {', '.join(str(item['seed']) for item in seed_results)}",
        f"- Backbone: `{cfg.BACKBONE_MODEL}`",
        f"- Training data: 8 documents (~115K tokens), causal LM loss",
        f"- PEFT: QLoRA rank={cfg.QLORA_RANK}, alpha={cfg.QLORA_ALPHA}, target q_proj+v_proj",
        "- **No retrieval at inference.**",
        "",
        "## 4. Results",
        "",
    ]
    for metric_name in ("Q_main", "S_det", "S_asst"):
        metric = summary.get(metric_name)
        if metric is None:
            continue
        lines.append(f"- {metric_name}: {metric['mean']:.4f} ± {metric['std']:.4f}")

    lines.extend([
        "",
        "## 5. Per-Seed Summary",
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
        "## 6. Delta vs S1 (RAG baseline)",
        "",
        f"- Q_main delta: {delta_s1['Q_main']:+.4f}",
        f"- S_det delta: {delta_s1['S_det']:+.4f}",
        f"- S_asst delta: {delta_s1['S_asst']:+.4f}",
    ])

    if delta_s2r:
        lines.extend([
            "",
            "## 7. Delta vs S2+R (RAFT + retrieval)",
            "",
            f"- Q_main delta: {delta_s2r['Q_main']:+.4f}",
            f"- S_det delta: {delta_s2r['S_det']:+.4f}",
            f"- S_asst delta: {delta_s2r['S_asst']:+.4f}",
        ])

    if delta_s2:
        lines.extend([
            "",
            "## 8. Delta vs S2 (closed-book)",
            "",
            f"- Q_main delta: {delta_s2['Q_main']:+.4f}",
            f"- S_det delta: {delta_s2['S_det']:+.4f}",
            f"- S_asst delta: {delta_s2['S_asst']:+.4f}",
            "",
            "Positive delta = CLM pretraining outperforms supervised closed-book on this metric.",
        ])

    lines.extend([
        "",
        "## 9. Breakdown By Answer Type",
        "",
    ])
    for answer_type, metrics in sorted(summary["breakdown_by_type"].items()):
        lines.append(f"### {answer_type}")
        lines.append("")
        for metric_name, stats in sorted(metrics.items()):
            lines.append(f"- {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        lines.append("")

    lines.extend([
        "## 10. Artifacts",
        "",
        f"- Aggregate summary: `{exp_cfg.RESULTS_DIR / 'aggregate_summary.json'}`",
        f"- Seed outputs: `{exp_cfg.RESULTS_DIR}`",
        f"- Adapters: `{exp_cfg.MODELS_DIR}`",
    ])
    exp_cfg.REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Report written to %s", exp_cfg.REPORT_PATH)


def _free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# %% ── Entry ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
