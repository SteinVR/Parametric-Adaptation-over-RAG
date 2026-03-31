"""EXP-010: Adapter Merge — CLM + RAFT linear interpolation, eval-only."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import config as cfg
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.data.io import load_goldset, load_json, save_json
from src.evaluation.runner import EvalRunner
from src.evaluation.s2_runner import prepare_s2_eval_samples, RetrievedEvalSample
from src.evaluation.schemas import Prediction
from src.evaluation.seed_stats import (
    aggregate_seed_summaries,
    eval_report_to_summary,
    save_seed_aggregate,
)
from src.generation.loader import unload_model
from src.generation.pipeline import GenerationPipeline
from src.generation.prompt import format_context_from_chunks

logging.basicConfig(
    level=logging.INFO,
    format=cfg.LOG_FORMAT,
    datefmt=cfg.LOG_DATE_FORMAT,
)
logger = logging.getLogger(__name__)

_EXP_CONFIG_SPEC = importlib.util.spec_from_file_location(
    "exp010_config",
    Path(__file__).with_name("config.py"),
)
if _EXP_CONFIG_SPEC is None or _EXP_CONFIG_SPEC.loader is None:
    raise RuntimeError("Failed to load experiment config")
exp_cfg = importlib.util.module_from_spec(_EXP_CONFIG_SPEC)
_EXP_CONFIG_SPEC.loader.exec_module(exp_cfg)


# ── Merged adapter loader ──────────────────────────────────────────────


def load_backbone_with_merged_adapter(
    *,
    model_name: str,
    clm_adapter_dir: Path,
    raft_adapter_dir: Path,
    alpha: float,
    device_map: str = "auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load quantized backbone and attach a linear merge of CLM + RAFT adapters.

    merged = alpha * CLM + (1 - alpha) * RAFT
    """
    if not torch.cuda.is_available():
        raise RuntimeError("4-bit inference requires CUDA with bitsandbytes support.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    # Load first adapter (CLM)
    model = PeftModel.from_pretrained(base_model, clm_adapter_dir, adapter_name="clm")
    # Load second adapter (RAFT)
    model.load_adapter(raft_adapter_dir, adapter_name="raft")
    # Linear merge
    model.add_weighted_adapter(
        adapters=["clm", "raft"],
        weights=[alpha, 1.0 - alpha],
        combination_type="linear",
        adapter_name="merged",
    )
    model.set_adapter("merged")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(
        "Loaded merged adapter: %.2f * CLM(%s) + %.2f * RAFT(%s)",
        alpha, clm_adapter_dir.name, 1.0 - alpha, raft_adapter_dir.name,
    )
    return model, tokenizer


# ── Generation with merged adapter ─────────────────────────────────────


def run_merged_generation(
    *,
    model_name: str,
    clm_adapter_dir: Path,
    raft_adapter_dir: Path,
    alpha: float,
    eval_samples: list[RetrievedEvalSample],
    max_new_tokens: int = 256,
) -> tuple[list[Prediction], float | None]:
    """Generate predictions using the merged adapter."""
    model, tokenizer = load_backbone_with_merged_adapter(
        model_name=model_name,
        clm_adapter_dir=clm_adapter_dir,
        raft_adapter_dir=raft_adapter_dir,
        alpha=alpha,
    )
    pipeline = GenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        do_sample=False,
        max_retries=1,
    )

    predictions: list[Prediction] = []
    peak_vram_mb = 0.0
    for sample in eval_samples:
        prediction = pipeline.generate_answer(
            question=sample.question,
            answer_type=sample.answer_type,
            context=sample.context,
            question_id=sample.question_id,
        )
        prediction.predicted_pages = sample.predicted_pages
        predictions.append(prediction)
        if torch.cuda.is_available():
            peak_vram_mb = max(
                peak_vram_mb,
                torch.cuda.max_memory_allocated() / 1024 / 1024,
            )

    unload_model(model)
    return predictions, (peak_vram_mb if torch.cuda.is_available() else None)


# ── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-010: Adapter Merge (CLM + RAFT)")
    parser.add_argument("--smoke", action="store_true", help="Smoke test with 2 questions")
    parser.add_argument("--seed", type=int, default=None, help="Run a single seed")
    parser.add_argument("--alpha", type=float, default=None, help="Override merge alpha")
    args = parser.parse_args()

    alpha = args.alpha if args.alpha is not None else exp_cfg.MERGE_ALPHA
    if not (0.0 <= alpha <= 1.0) or math.isnan(alpha):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    selected_seeds = _resolve_seeds(args.seed, args.smoke)
    alpha_tag = f"alpha_{round(alpha, 4)}"
    logger.info(
        "=== EXP-010 start === mode=%s, seeds=%s, alpha=%s",
        "smoke" if args.smoke else "full", selected_seeds, alpha_tag,
    )

    # Verify adapters exist and configs are compatible (every seed)
    for seed in selected_seeds:
        clm_dir = exp_cfg.CLM_MODELS_DIR / f"seed_{seed}"
        raft_dir = exp_cfg.RAFT_MODELS_DIR / f"seed_{seed}"
        if not clm_dir.exists():
            raise FileNotFoundError(f"CLM adapter not found: {clm_dir}")
        if not raft_dir.exists():
            raise FileNotFoundError(f"RAFT adapter not found: {raft_dir}")
        _verify_adapter_compatibility(clm_dir, raft_dir)

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
        clm_dir = exp_cfg.CLM_MODELS_DIR / f"seed_{seed}"
        raft_dir = exp_cfg.RAFT_MODELS_DIR / f"seed_{seed}"
        result_dir = exp_cfg.RESULTS_DIR / alpha_tag / ("smoke" if args.smoke else f"seed_{seed}")

        _free_gpu()
        logger.info(
            "── Inference seed %d → CLM=%s, RAFT=%s, α=%.2f ──",
            seed, clm_dir.name, raft_dir.name, alpha,
        )
        predictions, peak_infer_vram_mb = run_merged_generation(
            model_name=cfg.BACKBONE_MODEL,
            clm_adapter_dir=clm_dir,
            raft_adapter_dir=raft_dir,
            alpha=alpha,
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
            system_id="S7_merged",
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
            {"peak_infer_vram_mb": peak_infer_vram_mb, "merge_alpha": alpha},
            result_dir / "systems_metrics.json",
        )

        seed_summaries.append(
            eval_report_to_summary(
                seed=seed,
                report=report,
                train_time_seconds=0.0,  # No training — merge only
                peak_train_vram_mb=None,
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
        aggregate_summary["merge_alpha"] = alpha

        # Delta vs S2+R (RAFT)
        if exp_cfg.EXP003_AGGREGATE_PATH.exists():
            s2r_agg = load_json(exp_cfg.EXP003_AGGREGATE_PATH)
            aggregate_summary["delta_vs_s2r"] = {
                "Q_main": aggregate_summary["Q_main"]["mean"] - s2r_agg["Q_main"]["mean"],
                "S_det": aggregate_summary["S_det"]["mean"] - s2r_agg["S_det"]["mean"],
                "S_asst": aggregate_summary["S_asst"]["mean"] - s2r_agg["S_asst"]["mean"],
            }
        # Delta vs S3+R (CLM)
        if exp_cfg.EXP004B_AGGREGATE_PATH.exists():
            s3r_agg = load_json(exp_cfg.EXP004B_AGGREGATE_PATH)
            aggregate_summary["delta_vs_s3r"] = {
                "Q_main": aggregate_summary["Q_main"]["mean"] - s3r_agg["Q_main"]["mean"],
                "S_det": aggregate_summary["S_det"]["mean"] - s3r_agg["S_det"]["mean"],
                "S_asst": aggregate_summary["S_asst"]["mean"] - s3r_agg["S_asst"]["mean"],
            }

        agg_dir = exp_cfg.RESULTS_DIR / alpha_tag
        if args.smoke:
            save_seed_aggregate(aggregate_summary, agg_dir / "smoke")
        else:
            save_seed_aggregate(aggregate_summary, agg_dir)
            _write_report(aggregate_summary, alpha, agg_dir)


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


def _write_report(summary: dict, alpha: float, agg_dir: Path) -> None:
    seed_results = summary["seed_results"]
    delta_s1 = summary["delta_vs_s1"]
    delta_s2r = summary.get("delta_vs_s2r")
    delta_s3r = summary.get("delta_vs_s3r")

    lines = [
        "# Experiment Report: EXP-010 — Adapter Merge (CLM + RAFT)",
        "",
        f"**Date:** {datetime.now(timezone.utc).date().isoformat()}",
        "**Status:** Completed",
        "",
        "## 1. Goal",
        "",
        "Test whether a linear merge of CLM and RAFT adapters combines their strengths",
        "(CLM's S_asst advantage + RAFT's S_det advantage) without retraining.",
        "",
        "## 2. Setup",
        "",
        f"- Merge: α * CLM + (1−α) * RAFT, α={alpha}",
        f"- Seeds: {', '.join(str(item['seed']) for item in seed_results)} (same-seed matching)",
        f"- Backbone: `{cfg.BACKBONE_MODEL}` + merged adapter",
        "- Retrieval: S1 pipeline (Qdrant hybrid + reranker + evidence compression)",
        "- No training — linear interpolation of existing adapter weights",
        "",
        "## 3. Results",
        "",
    ]
    for metric_name in ("Q_main", "S_det", "S_asst"):
        metric = summary.get(metric_name)
        if metric is None:
            continue
        lines.append(f"- {metric_name}: {metric['mean']:.4f} ± {metric['std']:.4f}")
    g = summary.get("G_f_beta")
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
        "## 5. Deltas",
        "",
        "### vs S1 (RAG, no adapter)",
        "",
        f"- Q_main: {delta_s1['Q_main']:+.4f}",
        f"- S_det: {delta_s1['S_det']:+.4f}",
        f"- S_asst: {delta_s1['S_asst']:+.4f}",
    ])

    if delta_s2r:
        lines.extend([
            "",
            "### vs S2+R (RAFT + retrieval)",
            "",
            f"- Q_main: {delta_s2r['Q_main']:+.4f}",
            f"- S_det: {delta_s2r['S_det']:+.4f}",
            f"- S_asst: {delta_s2r['S_asst']:+.4f}",
        ])

    if delta_s3r:
        lines.extend([
            "",
            "### vs S3+R (CLM + retrieval)",
            "",
            f"- Q_main: {delta_s3r['Q_main']:+.4f}",
            f"- S_det: {delta_s3r['S_det']:+.4f}",
            f"- S_asst: {delta_s3r['S_asst']:+.4f}",
        ])

    lines.extend([
        "",
        "## 6. Interpretation",
        "",
        "TODO: Fill after results.",
        "",
        "## 7. Artifacts",
        "",
        f"- Aggregate summary: `{agg_dir / 'aggregate_summary.json'}`",
        f"- Seed outputs: `{agg_dir}`",
    ])
    exp_cfg.REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Report written to %s", exp_cfg.REPORT_PATH)


def _verify_adapter_compatibility(clm_dir: Path, raft_dir: Path) -> None:
    """Check that CLM and RAFT adapters share rank, target modules, and backbone."""
    clm_cfg = json.loads((clm_dir / "adapter_config.json").read_text())
    raft_cfg = json.loads((raft_dir / "adapter_config.json").read_text())
    checks = ["r", "target_modules", "base_model_name_or_path"]
    mismatches = [
        f"  {k}: CLM={clm_cfg.get(k)} vs RAFT={raft_cfg.get(k)}"
        for k in checks
        if clm_cfg.get(k) != raft_cfg.get(k)
    ]
    if mismatches:
        raise ValueError(
            "Adapter configs are incompatible:\n" + "\n".join(mismatches)
        )
    logger.info("Adapter compatibility OK (rank=%s, targets=%s)", clm_cfg["r"], clm_cfg["target_modules"])


def _free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
