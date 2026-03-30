"""EXP-004: S3 Doc-to-LoRA monolithic packaging and evaluation."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import config as cfg
import torch
from peft import LoraConfig
from src.data.io import load_goldset, load_json, save_json
from src.d2l.adapter_io import load_peft_lora_state_dict, save_peft_lora_adapter
from src.d2l.checkpoint import load_d2l_model, resolve_checkpoint_file
from src.d2l.corpus import CorpusDocument, load_frozen_corpus_documents
from src.d2l.packaging import (
    DocAdapterGenerationResult,
    average_lora_state_dicts,
    generate_document_adapter,
)
from src.d2l.runner import run_d2l_no_retrieval_generation
from src.d2l.sanity import select_doc_train_refs, score_deterministic_subset
from src.evaluation.runner import EvalRunner
from src.generation.loader import unload_model

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EXP-004: S3 Doc-to-LoRA monolithic packaging"
    )
    parser.add_argument("--smoke", action="store_true", help="Run a reduced smoke flow")
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip D2L adapter generation and reuse existing adapters",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip eval after building the monolithic adapter",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs and report blockers without loading the D2L checkpoint",
    )
    args = parser.parse_args()

    validation = _validate_inputs()
    if args.validate_only:
        report_status = (
            "Blocked"
            if str(validation["checkpoint_status"]).startswith("missing")
            else "Validation only"
        )
        save_json(validation, exp_cfg.RESULTS_DIR / "validation.json")
        _write_report(
            validation=validation,
            doc_generation_results=[],
            doc_sanity_results=[],
            merge_summary=None,
            eval_report=None,
            eval_artifacts=None,
            status_label=report_status,
            blocker_message=(
                validation.get("checkpoint_status")
                if report_status == "Blocked"
                else None
            ),
        )
        logger.info("Validate-only run completed")
        return

    if not args.skip_generate:
        checkpoint_file = resolve_checkpoint_file(exp_cfg.DOC2LORA_CHECKPOINT_ROOT)
        logger.info("Using Doc-to-LoRA checkpoint: %s", checkpoint_file)
    else:
        logger.info("Skipping generation; will reuse existing per-document adapters")

    refs = load_goldset(cfg.GOLDSET_PATH)
    refs_by_id = {ref["question_id"]: ref for ref in refs}
    split = load_json(cfg.DATA_SPLITS / "split_v1.json")
    train_ids = list(split[exp_cfg.TRAIN_SPLIT_NAME])
    eval_ids = list(split["eval"])
    if args.smoke:
        eval_ids = eval_ids[: exp_cfg.SMOKE_EVAL_QUESTIONS]

    train_refs = [refs_by_id[qid] for qid in train_ids]
    eval_refs = [refs_by_id[qid] for qid in eval_ids]

    documents = load_frozen_corpus_documents(
        corpus_dir=cfg.CORPUS_DIR,
        goldset_path=cfg.GOLDSET_PATH,
    )
    if args.smoke:
        documents = documents[: exp_cfg.SMOKE_DOCS]
        logger.info("Smoke mode: packaging %d documents", len(documents))

    if args.skip_generate:
        doc_generation_results = []
        logger.warning("Skipping adapter generation and expecting existing adapters")
    else:
        _free_gpu()
        doc_generation_results = _generate_doc_adapters(
            documents=documents,
        )

    doc_sanity_results = _run_doc_sanity_checks(
        documents=documents,
        train_refs=train_refs,
    )

    merge_summary = _merge_doc_adapters(documents=documents)

    eval_report = None
    eval_artifacts = None
    if not args.skip_eval:
        _free_gpu()
        eval_report, eval_artifacts = _run_monolithic_eval(eval_refs=eval_refs)

    _write_report(
        validation=validation,
        doc_generation_results=doc_generation_results,
        doc_sanity_results=doc_sanity_results,
        merge_summary=merge_summary,
        eval_report=eval_report,
        eval_artifacts=eval_artifacts,
        status_label="Completed",
        blocker_message=None,
    )


def _validate_inputs() -> dict[str, str | int | bool]:
    documents = load_frozen_corpus_documents(
        corpus_dir=cfg.CORPUS_DIR,
        goldset_path=cfg.GOLDSET_PATH,
    )
    refs = load_goldset(cfg.GOLDSET_PATH)
    split = load_json(cfg.DATA_SPLITS / "split_v1.json")
    checkpoint_exists = exp_cfg.DOC2LORA_CHECKPOINT_FILE.exists()
    checkpoint_status = (
        "present"
        if checkpoint_exists
        else f"missing: {exp_cfg.DOC2LORA_CHECKPOINT_FILE}"
    )
    validation = {
        "document_count": len(documents),
        "goldset_count": len(refs),
        "train_count": len(split[exp_cfg.TRAIN_SPLIT_NAME]),
        "eval_count": len(split["eval"]),
        "checkpoint_exists": checkpoint_exists,
        "checkpoint_status": checkpoint_status,
    }
    logger.info(
        "Validation: %d docs, %d refs, checkpoint=%s",
        validation["document_count"],
        validation["goldset_count"],
        validation["checkpoint_status"],
    )
    return validation


def _generate_doc_adapters(
    *,
    documents: Sequence[CorpusDocument],
) -> list[dict[str, object]]:
    model = load_d2l_model(exp_cfg.DOC2LORA_CHECKPOINT_ROOT)
    results: list[dict[str, object]] = []
    try:
        for document in documents:
            output_dir = exp_cfg.MODELS_DIR / f"doc{document.doc_index}"
            generation_result = generate_document_adapter(
                model=model,
                document=document,
                output_dir=output_dir,
            )
            row = _doc_generation_row(generation_result)
            save_json(
                row,
                exp_cfg.RESULTS_DIR / "generation" / f"doc{document.doc_index}.json",
            )
            results.append(row)
    finally:
        unload_model(model)
    save_json(results, exp_cfg.RESULTS_DIR / "document_generation.json")
    return results


def _run_doc_sanity_checks(
    *,
    documents: Sequence[CorpusDocument],
    train_refs: list[dict[str, Any]],
) -> list[dict[str, object]]:
    sanity_rows: list[dict[str, object]] = []
    for document in documents:
        selected_refs = select_doc_train_refs(
            train_refs=train_refs,
            doc_id=document.doc_id,
        )
        if not selected_refs:
            logger.warning(
                "No deterministic single-document train refs found for %s",
                document.doc_name,
            )
            sanity_rows.append(_empty_doc_sanity_row(document))
            continue

        adapter_dir = exp_cfg.MODELS_DIR / f"doc{document.doc_index}"
        predictions, peak_vram_mb = run_d2l_no_retrieval_generation(
            model_name=cfg.BACKBONE_MODEL,
            adapter_dir=adapter_dir,
            eval_refs=selected_refs,
            max_new_tokens=exp_cfg.MAX_NEW_TOKENS,
        )
        score = score_deterministic_subset(
            predictions=predictions,
            refs=selected_refs,
        )
        sanity_rows.append(
            {
                "doc_index": document.doc_index,
                "doc_name": document.doc_name,
                "doc_id": document.doc_id,
                "question_count": score.question_count,
                "deterministic_count": score.deterministic_count,
                "s_det": score.s_det,
                "malformed_count": score.malformed_count,
                "peak_vram_mb": peak_vram_mb,
            }
        )
    save_json(sanity_rows, exp_cfg.RESULTS_DIR / "doc_sanity.json")
    return sanity_rows


def _merge_doc_adapters(
    *,
    documents: Sequence[CorpusDocument],
) -> dict[str, object]:
    state_dicts: list[dict[str, torch.Tensor]] = []
    adapter_dirs: list[str] = []
    for document in documents:
        adapter_dir = exp_cfg.MODELS_DIR / f"doc{document.doc_index}"
        adapter_dirs.append(str(adapter_dir))
        state_dicts.append(load_peft_lora_state_dict(adapter_dir))

    start = time.perf_counter()
    merged_state_dict = average_lora_state_dicts(state_dicts)
    merge_seconds = time.perf_counter() - start

    reference_adapter_dir = exp_cfg.MODELS_DIR / "doc1"
    save_json(
        {
            "merge_seconds": merge_seconds,
            "source_adapter_dirs": adapter_dirs,
            "tensor_count": len(merged_state_dict),
            "reference_config_path": str(reference_adapter_dir / "adapter_config.json"),
        },
        exp_cfg.RESULTS_DIR / "merge_summary.json",
    )

    # Reuse the same PEFT adapter metadata layout as a per-doc adapter.
    peft_config = LoraConfig.from_pretrained(reference_adapter_dir)

    save_peft_lora_adapter(
        adapter_dir=exp_cfg.MONOLITHIC_MODEL_DIR,
        state_dict=merged_state_dict,
        peft_config=peft_config,
        base_model_name=cfg.BACKBONE_MODEL,
    )
    return {
        "merge_seconds": merge_seconds,
        "source_adapter_dirs": adapter_dirs,
        "tensor_count": len(merged_state_dict),
        "output_dir": str(exp_cfg.MONOLITHIC_MODEL_DIR),
    }


def _run_monolithic_eval(
    *,
    eval_refs: list[dict[str, Any]],
) -> tuple[dict[str, object], dict[str, object]]:
    predictions, peak_infer_vram_mb = run_d2l_no_retrieval_generation(
        model_name=cfg.BACKBONE_MODEL,
        adapter_dir=exp_cfg.MONOLITHIC_MODEL_DIR,
        eval_refs=eval_refs,
        max_new_tokens=exp_cfg.MAX_NEW_TOKENS,
    )
    save_json(
        [prediction.model_dump() for prediction in predictions],
        exp_cfg.RESULTS_DIR / "predictions.json",
    )

    eval_runner = EvalRunner(
        goldset_path=cfg.GOLDSET_PATH,
        split_path=cfg.DATA_SPLITS / "split_v1.json",
        judge_model=cfg.JUDGE_MODEL,
        judge_reasoning=cfg.JUDGE_REASONING,
        grounding_beta=cfg.GROUNDING_BETA,
        q_main_weights=cfg.Q_MAIN_WEIGHTS,
    )
    report = eval_runner.evaluate(
        predictions=predictions,
        system_id="S3",
        experiment_id=exp_cfg.EXPERIMENT_ID,
        split="eval",
        compute_grounding_flag=False,
    )
    eval_runner.save_report(report, exp_cfg.RESULTS_DIR)
    systems_metrics = {
        "peak_infer_vram_mb": peak_infer_vram_mb,
        "ttft_median_ms": report.ttft_median_ms,
        "ttft_p95_ms": report.ttft_p95_ms,
        "latency_median_ms": report.latency_median_ms,
        "latency_p95_ms": report.latency_p95_ms,
        "malformed_rate": report.malformed_rate,
    }
    save_json(systems_metrics, exp_cfg.RESULTS_DIR / "systems_metrics.json")
    return report.model_dump(), systems_metrics


def _write_report(
    *,
    validation: dict[str, str | int | bool],
    doc_generation_results: list[dict[str, object]],
    doc_sanity_results: list[dict[str, object]],
    merge_summary: dict[str, object] | None,
    eval_report: dict[str, object] | None,
    eval_artifacts: dict[str, object] | None,
    status_label: str,
    blocker_message: str | None,
) -> None:
    lines: list[str] = [
        "# Experiment Report: EXP-004 - S3 Doc-to-LoRA Monolithic",
        "",
        f"**Date:** {datetime.now(timezone.utc).date().isoformat()}",
        f"**Status:** {status_label}",
        "",
        "## 1. Goal",
        "",
        "- Generate 8 per-document Doc-to-LoRA adapters, validate each on its own train-document deterministic subset, merge them by simple average, and evaluate the monolithic adapter without retrieval.",
        "",
        "## 2. Validation",
        "",
        f"- Corpus documents: {validation['document_count']}",
        f"- Goldset references: {validation['goldset_count']}",
        f"- S2-train questions: {validation['train_count']}",
        f"- Eval questions: {validation['eval_count']}",
        f"- Checkpoint status: {validation['checkpoint_status']}",
        "",
    ]

    if blocker_message:
        lines.extend(
            [
                "## 3. Blocker",
                "",
                f"- {blocker_message}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## 3. Per-Document Packaging",
                "",
                "| Doc | Pages | Words | Gen sec | Peak VRAM MB | Adapter bytes |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in doc_generation_results:
            lines.append(
                "| {doc_index} | {page_count} | {word_count} | {generation_seconds:.1f} | {peak_vram_mb} | {adapter_file_size_bytes} |".format(
                    doc_index=row["doc_index"],
                    page_count=row["page_count"],
                    word_count=row["word_count"],
                    generation_seconds=float(row["generation_seconds"]),
                    peak_vram_mb=(
                        f"{float(row['peak_vram_mb']):.1f}"
                        if row.get("peak_vram_mb") is not None
                        else "N/A"
                    ),
                    adapter_file_size_bytes=row["adapter_file_size_bytes"],
                )
            )
        lines.extend(
            [
                "",
                "## 4. Sanity Check",
                "",
                "| Doc | Train refs | Deterministic refs | S_det | Malformed |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in doc_sanity_results:
            lines.append(
                "| {doc_index} | {question_count} | {deterministic_count} | {s_det:.4f} | {malformed_count} |".format(
                    doc_index=row["doc_index"],
                    question_count=row["question_count"],
                    deterministic_count=row["deterministic_count"],
                    s_det=float(row["s_det"]),
                    malformed_count=row["malformed_count"],
                )
            )
        lines.extend(
            [
                "",
                "## 5. Merge Summary",
                "",
            ]
        )
        if merge_summary is not None:
            lines.append(f"- Merge seconds: {float(merge_summary['merge_seconds']):.1f}")
            lines.append(f"- Source adapters: {len(merge_summary['source_adapter_dirs'])}")
            lines.append(f"- Output dir: `{merge_summary['output_dir']}`")
        else:
            lines.append("- Merge summary unavailable.")

        lines.extend(
            [
                "",
                "## 6. Eval",
                "",
            ]
        )
        if eval_report is not None:
            lines.extend(
                [
                    "| Metric | Value |",
                    "| --- | ---: |",
                    f"| Q_main | {float(eval_report['q_main']):.4f} |",
                    f"| S_det | {float(eval_report['s_det']):.4f} |",
                    f"| S_asst | {float(eval_report['s_asst']):.4f} |",
                    f"| Grounding F_beta | {eval_report['grounding_f_beta'] if eval_report.get('grounding_f_beta') is not None else 'N/A'} |",
                ]
            )
        else:
            lines.append("- Eval not run.")

        if eval_artifacts is not None:
            lines.extend(
                [
                    "",
                    "## 7. System Metrics",
                    "",
                    f"- Peak infer VRAM MB: {eval_artifacts['peak_infer_vram_mb']}",
                    f"- TTFT median ms: {eval_artifacts['ttft_median_ms']}",
                    f"- Latency median ms: {eval_artifacts['latency_median_ms']}",
                ]
            )

    lines.extend(
        [
            "",
            "## 8. Artifacts",
            "",
            f"- Generation records: `{exp_cfg.RESULTS_DIR / 'document_generation.json'}`",
            f"- Sanity results: `{exp_cfg.RESULTS_DIR / 'doc_sanity.json'}`",
            f"- Merge summary: `{exp_cfg.RESULTS_DIR / 'merge_summary.json'}`",
            f"- Validation summary: `{exp_cfg.RESULTS_DIR / 'validation.json'}`",
            f"- Predictions: `{exp_cfg.RESULTS_DIR / 'predictions.json'}`",
            f"- Eval outputs: `{exp_cfg.RESULTS_DIR}`",
            f"- Adapters: `{exp_cfg.MODELS_DIR}`",
        ]
    )
    exp_cfg.REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _doc_generation_row(result: DocAdapterGenerationResult) -> dict[str, object]:
    return {
        "doc_index": result.doc_index,
        "doc_name": result.doc_name,
        "doc_id": result.doc_id,
        "page_count": result.page_count,
        "word_count": result.word_count,
        "char_count": result.char_count,
        "generation_seconds": result.generation_seconds,
        "peak_vram_mb": result.peak_vram_mb,
        "adapter_dir": str(result.adapter.adapter_dir),
        "adapter_file_size_bytes": result.adapter.byte_size,
        "tensor_count": result.adapter.tensor_count,
    }


def _empty_doc_sanity_row(document: CorpusDocument) -> dict[str, object]:
    return {
        "doc_index": document.doc_index,
        "doc_name": document.doc_name,
        "doc_id": document.doc_id,
        "question_count": 0,
        "deterministic_count": 0,
        "s_det": 0.0,
        "malformed_count": 0,
    }


def _free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
