#!/usr/bin/env python3
"""Build a self-contained comparison report for the paper's figures.

The script reads the frozen experiment outputs and the existing figure PNGs,
then emits the canonical Data Analytics ``artifact.json`` used by the portable
HTML report builder. It does not modify the paper or the existing figures.
"""

from __future__ import annotations

import base64
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = ROOT / "assets" / "figures"
RESULTS = ROOT / "results"
SOURCE_DATA = HERE / "source-data"

DISPLAY = {
    "S1": "Base-RAG",
    "S2+R": "RAFT-RAG",
    "S3+R": "CLM-RAG",
    "S7": "Merge-RAG",
    "S2": "RAFT-Closed",
    "S3": "CLM-Closed",
}

CURRENT_FIGURES = [
    (
        "fig01_system_schematic.png",
        "Figure 1. System overview schematic",
        "Показывает общий фиксированный retrieval backbone и различия между генераторами. Схема уже выполняет свою задачу и её разумно сохранить.",
    ),
    (
        "fig02_delta_bars.png",
        "Figure 2. Improvement over Base-RAG",
        "Показывает направление изменений Q_main, S_det и S_asst, но не показывает неопределённость вокруг небольших дельт.",
    ),
    (
        "fig03_judge_criteria.png",
        "Figure 3. Free-text judge profile",
        "Разлагает free-text score на пять критериев. Усечённая ось у абсолютных столбцов визуально усиливает различия, поэтому этот вариант лучше заменить или перенести в приложение.",
    ),
    (
        "fig04_per_type_heatmap.png",
        "Figure 4. Per-type score heatmap",
        "Показывает абсолютные оценки по типам, но во многом повторяет Table 3. Delta-версия ниже делает специализацию адаптеров заметнее.",
    ),
    (
        "fig05_singledoc_multidoc.png",
        "Figure 5. Single-document vs multi-document",
        "Хорошо показывает падение на multi-document вопросах, но n=8 и seed variation нужно держать рядом с интерпретацией.",
    ),
]

REPORTS = {
    "Base-RAG": [RESULTS / "EXP-002" / "eval_report.json"],
    "RAFT-RAG": [
        RESULTS / "EXP-003" / f"seed_{seed}" / "eval_report.json"
        for seed in (42, 123, 777)
    ],
    "CLM-RAG": [
        RESULTS / "EXP-004b" / f"seed_{seed}" / "eval_report.json"
        for seed in (42, 123, 777)
    ],
    "Merge-RAG": [
        RESULTS / "EXP-010" / "alpha_0.5" / f"seed_{seed}" / "eval_report.json"
        for seed in (42, 123, 777)
    ],
}

SEED_SUMMARIES = {
    "RAFT-RAG": RESULTS / "EXP-003" / "aggregate_summary.json",
    "CLM-RAG": RESULTS / "EXP-004b" / "aggregate_summary.json",
    "Merge-RAG": RESULTS / "EXP-010" / "alpha_0.5" / "aggregate_summary.json",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def question_score(item: dict) -> float:
    score = item.get("s_det")
    if score is None:
        score = item.get("s_asst")
    if score is None:
        raise ValueError(f"Missing question score for {item.get('question_id')}")
    return float(score)


def load_question_profiles() -> tuple[dict, dict, dict]:
    profiles: dict[str, dict[str, float]] = {}
    metadata: dict[str, dict] = {}
    coverage: dict[str, str] = {}

    for system, paths in REPORTS.items():
        per_question: dict[str, list[float]] = defaultdict(list)
        for path in paths:
            report = json.loads(path.read_text(encoding="utf-8"))
            for item in report["question_scores"]:
                qid = item["question_id"]
                per_question[qid].append(question_score(item))
                metadata.setdefault(
                    qid,
                    {
                        "answer_type": item["answer_type"],
                        "is_unanswerable": bool(item["is_unanswerable"]),
                    },
                )
                if system == "Base-RAG" and not item["is_unanswerable"]:
                    recall = float(item["grounding_recall"])
                    coverage[qid] = (
                        "Full (n=30)"
                        if recall == 1.0
                        else "None (n=12)"
                        if recall == 0.0
                        else "Partial (n=5)"
                    )
        profiles[system] = {
            qid: float(np.mean(scores)) for qid, scores in per_question.items()
        }

    base_ids = set(profiles["Base-RAG"])
    for system, system_scores in profiles.items():
        if set(system_scores) != base_ids:
            raise ValueError(f"Question ids differ for {system}")
    return profiles, metadata, coverage


def bootstrap_deltas(
    profiles: dict[str, dict[str, float]], metadata: dict[str, dict]
) -> tuple[list[dict], dict[str, tuple[float, float, float]]]:
    rng = np.random.default_rng(20260804)
    det_ids = np.array(
        [qid for qid, meta in metadata.items() if meta["answer_type"] != "free_text"]
    )
    asst_ids = np.array(
        [qid for qid, meta in metadata.items() if meta["answer_type"] == "free_text"]
    )
    rows: list[dict] = []
    summaries: dict[str, tuple[float, float, float]] = {}

    for system in ("RAFT-RAG", "CLM-RAG", "Merge-RAG"):
        deltas: list[float] = []
        for replicate in range(500):
            det_sample = rng.choice(det_ids, size=len(det_ids), replace=True)
            asst_sample = rng.choice(asst_ids, size=len(asst_ids), replace=True)

            base_det = float(np.mean([profiles["Base-RAG"][qid] for qid in det_sample]))
            base_asst = float(np.mean([profiles["Base-RAG"][qid] for qid in asst_sample]))
            system_det = float(np.mean([profiles[system][qid] for qid in det_sample]))
            system_asst = float(np.mean([profiles[system][qid] for qid in asst_sample]))
            delta = 0.7 * (system_det - base_det) + 0.3 * (system_asst - base_asst)
            deltas.append(delta)
            rows.append(
                {
                    "system": system,
                    "replicate": replicate + 1,
                    "delta_q_main": round(delta, 6),
                }
            )

        low, median, high = np.quantile(deltas, [0.025, 0.5, 0.975])
        summaries[system] = (float(low), float(median), float(high))
    return rows, summaries


def seed_rows() -> list[dict]:
    by_system: dict[str, dict[int, float]] = {}
    for system, path in SEED_SUMMARIES.items():
        summary = json.loads(path.read_text(encoding="utf-8"))
        by_system[system] = {
            int(item["seed"]): float(item["q_main"])
            for item in summary["seed_results"]
        }

    rows = []
    for system in ("RAFT-RAG", "CLM-RAG", "Merge-RAG"):
        for seed in (42, 123, 777):
            rows.append(
                {
                    "system": system,
                    "seed": f"Seed {seed}",
                    "q_main": round(by_system[system][seed], 4),
                    "baseline": 0.6425,
                }
            )
    return rows


def retrieval_rows(main_results: list[dict[str, str]]) -> list[dict]:
    rows_by_system = {row["system"]: row for row in main_results}
    return [
        {
            "paradigm": "RAFT",
            "retrieval_state": "With retrieval",
            "q_main": round(float(rows_by_system["S2+R"]["q_main"]), 4),
        },
        {
            "paradigm": "CLM",
            "retrieval_state": "With retrieval",
            "q_main": round(float(rows_by_system["S3+R"]["q_main"]), 4),
        },
        {
            "paradigm": "RAFT",
            "retrieval_state": "Without retrieval",
            "q_main": round(float(rows_by_system["S2"]["q_main"]), 4),
        },
        {
            "paradigm": "CLM",
            "retrieval_state": "Without retrieval",
            "q_main": round(float(rows_by_system["S3"]["q_main"]), 4),
        },
    ]


def evidence_rows(
    profiles: dict[str, dict[str, float]], coverage: dict[str, str]
) -> list[dict]:
    order = ("Full (n=30)", "Partial (n=5)", "None (n=12)")
    rows: list[dict] = []
    for group in order:
        ids = [qid for qid, label in coverage.items() if label == group]
        for system in ("Base-RAG", "RAFT-RAG", "CLM-RAG", "Merge-RAG"):
            rows.append(
                {
                    "coverage": group,
                    "system": system,
                    "mean_score": round(
                        float(np.mean([profiles[system][qid] for qid in ids])), 4
                    ),
                    "n_questions": len(ids),
                }
            )
    return rows


def type_delta_rows() -> list[dict]:
    raw = read_csv(RESULTS / "EXP-007" / "per_type_breakdown.csv")
    values = {(row["system"], row["answer_type"]): float(row["score"]) for row in raw}
    counts = {
        "boolean": 12,
        "number": 7,
        "name": 8,
        "names": 5,
        "date": 5,
        "free_text": 13,
    }
    labels = {
        "boolean": "Boolean (n=12)",
        "number": "Number (n=7)",
        "name": "Name (n=8)",
        "names": "Names (n=5)",
        "date": "Date (n=5)",
        "free_text": "Free-text (n=13)",
    }
    rows = []
    for system in ("S2+R", "S3+R", "S7"):
        for answer_type in counts:
            rows.append(
                {
                    "system": DISPLAY[system],
                    "answer_type": labels[answer_type],
                    "n": counts[answer_type],
                    "delta_vs_base": round(
                        values[(system, answer_type)] - values[("S1", answer_type)], 4
                    ),
                }
            )
    return rows


def win_tie_loss_rows() -> list[dict]:
    raw = read_csv(RESULTS / "EXP-007" / "pairwise_win_rate.csv")
    lookup = {(row["system_a"], row["system_b"]): row for row in raw}
    rows = []
    for system in ("S2+R", "S3+R", "S7"):
        row = lookup[(system, "S1")]
        reverse = lookup[("S1", system)]
        outcomes = (
            ("Adapted system wins", int(row["win_count"])),
            ("Ties", int(row["tie_count"])),
            ("Base-RAG wins", int(reverse["win_count"])),
        )
        for outcome, count in outcomes:
            rows.append(
                {
                    "comparison": f"{DISPLAY[system]} vs Base-RAG",
                    "outcome": outcome,
                    "count": count,
                    "share": count / int(row["n_questions"]),
                    "n_questions": int(row["n_questions"]),
                }
            )
    return rows


def encoded_png(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def current_figures_html() -> str:
    figures = []
    for filename, title, caption in CURRENT_FIGURES:
        data = encoded_png(ASSETS / filename)
        figures.append(
            f"""
            <figure>
              <div class="figure-number">{title}</div>
              <img src="data:image/png;base64,{data}" alt="{title}">
              <figcaption>{caption}</figcaption>
            </figure>
            """
        )
    return f"""
    <style>
      :root {{ color-scheme: light; }}
      body {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; color: #252a31; }}
      .stack {{ display: grid; grid-template-columns: minmax(0, 1fr); gap: 28px; }}
      figure {{ margin: 0; padding: 20px; border: 1px solid #dfe3e8; border-radius: 14px; background: #fff; }}
      .figure-number {{ margin: 0 0 12px; font-size: 16px; font-weight: 700; color: #1f2933; }}
      img {{ display: block; width: 100%; border-radius: 8px; background: #fff; }}
      figcaption {{ margin-top: 14px; font-size: 14px; line-height: 1.55; color: #56616f; }}
      @media (max-width: 720px) {{ figure {{ padding: 12px; }} .stack {{ gap: 18px; }} }}
    </style>
    <div class="stack">{''.join(figures)}</div>
    """


def write_derived_sources(datasets: dict[str, list[dict]]) -> None:
    """Write compact chart-ready extracts next to the report for provenance."""

    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    filenames = {
        "bootstrap_deltas": "bootstrap-deltas.csv",
        "seed_qmain": "seed-qmain.csv",
        "retrieval_contribution": "retrieval-contribution.csv",
        "evidence_coverage": "evidence-coverage.csv",
        "type_deltas": "type-deltas.csv",
        "win_tie_loss": "win-tie-loss.csv",
    }
    for dataset_id, filename in filenames.items():
        rows = datasets[dataset_id]
        if not rows:
            raise ValueError(f"Derived dataset {dataset_id} is empty")
        with (SOURCE_DATA / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    figure_notes = [
        "# Existing figures",
        "",
        "Embedded without modification from the following project files:",
        "",
    ]
    figure_notes.extend(f"- `assets/figures/{item[0]}`" for item in CURRENT_FIGURES)
    figure_notes.extend(
        [
            "",
            "The comparison report embeds these PNG bytes directly so the delivered HTML remains self-contained.",
        ]
    )
    (SOURCE_DATA / "existing-figures.md").write_text(
        "\n".join(figure_notes) + "\n", encoding="utf-8"
    )


def source_specs() -> list[dict]:
    return [
        {
            "id": "existing-figures",
            "label": "Existing thesis figure PNGs",
            "path": "source-data/existing-figures.md",
        },
        {
            "id": "bootstrap-source",
            "label": "Paired stratified bootstrap deltas derived from per-question reports",
            "path": "source-data/bootstrap-deltas.csv",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": "SELECT * FROM read_csv_auto('source-data/bootstrap-deltas.csv');",
                "description": "Read the reviewed paired-bootstrap delta extract.",
                "tables_used": ["source-data/bootstrap-deltas.csv"],
                "filters": ["500 replicates per adapted retrieval-aware system"],
                "metric_definitions": ["delta_q_main is paired against Base-RAG after stratified resampling of deterministic and free-text questions."],
            },
        },
        {
            "id": "seed-source",
            "label": "Seed-level Q_main values from aggregate experiment summaries",
            "path": "source-data/seed-qmain.csv",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": "SELECT * FROM read_csv_auto('source-data/seed-qmain.csv');",
                "description": "Read the reviewed seed-level Q_main extract.",
                "tables_used": ["source-data/seed-qmain.csv"],
                "metric_definitions": ["Q_main = 0.7 × S_det + 0.3 × S_asst."],
            },
        },
        {
            "id": "retrieval-source",
            "label": "Matched retrieval on/off comparison derived from EXP-006",
            "path": "source-data/retrieval-contribution.csv",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": "SELECT * FROM read_csv_auto('source-data/retrieval-contribution.csv');",
                "description": "Read matched RAFT and CLM Q_main values with and without retrieval.",
                "tables_used": ["source-data/retrieval-contribution.csv"],
                "metric_definitions": ["Q_main = 0.7 × S_det + 0.3 × S_asst."],
            },
        },
        {
            "id": "coverage-source",
            "label": "Question scores grouped by shared gold-page coverage",
            "path": "source-data/evidence-coverage.csv",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": "SELECT * FROM read_csv_auto('source-data/evidence-coverage.csv');",
                "description": "Read mean per-question scores by shared gold-page retrieval coverage.",
                "tables_used": ["source-data/evidence-coverage.csv"],
                "filters": ["47 answerable questions; three unanswerable questions excluded"],
                "metric_definitions": ["Full coverage has grounding_recall=1, none has grounding_recall=0, and partial is between 0 and 1."],
            },
        },
        {
            "id": "type-source",
            "label": "Per-answer-type score deltas derived from EXP-007",
            "path": "source-data/type-deltas.csv",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": "SELECT * FROM read_csv_auto('source-data/type-deltas.csv');",
                "description": "Read per-answer-type score changes relative to Base-RAG.",
                "tables_used": ["source-data/type-deltas.csv"],
                "metric_definitions": ["delta_vs_base is the system's original per-type score minus Base-RAG's score for the same answer type."],
            },
        },
        {
            "id": "outcome-source",
            "label": "Question-level wins, ties, and losses derived from EXP-007",
            "path": "source-data/win-tie-loss.csv",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": "SELECT * FROM read_csv_auto('source-data/win-tie-loss.csv');",
                "description": "Read question-level pairwise outcomes against Base-RAG.",
                "tables_used": ["source-data/win-tie-loss.csv"],
                "filters": ["50 held-out questions per comparison"],
            },
        },
    ]


def chart_specs() -> list[dict]:
    return [
        {
            "id": "bootstrap-delta",
            "title": "Bootstrap distribution of ΔQ_main vs Base-RAG",
            "subtitle": "500 paired stratified resamples of the 50-question evaluation set; trained systems are averaged over three seeds",
            "showDescription": True,
            "intent": "distribution",
            "question": "How uncertain are the observed aggregate gains over Base-RAG?",
            "rationale": "A box plot shows the sampling distribution and whether plausible deltas include zero.",
            "comparisonContext": {
                "baseline": "Base-RAG",
                "grain": "bootstrap replicate",
                "unit": "Q_main delta",
            },
            "type": "boxPlot",
            "dataset": "bootstrap_deltas",
            "sourceId": "bootstrap-source",
            "encodings": {
                "x": {"field": "system", "type": "nominal", "label": "System"},
                "y": {
                    "field": "delta_q_main",
                    "type": "quantitative",
                    "label": "ΔQ_main vs Base-RAG",
                    "format": "number",
                },
            },
            "valueFormat": "number",
            "referenceLines": [
                {"axis": "y", "value": 0, "label": "No change", "color": "neutral", "lineStyle": "dashed"}
            ],
            "palette": {"kind": "sequential", "name": "blue"},
            "layout": "full",
            "surface": {"surface": "card", "showControls": True, "viewMode": "visualization"},
        },
        {
            "id": "seed-qmain",
            "title": "Q_main across training seeds",
            "subtitle": "Three trained seeds per system; dashed reference is the fixed Base-RAG score of 0.643",
            "showDescription": True,
            "intent": "comparison",
            "question": "How much does training stochasticity move each reported system score?",
            "rationale": "Grouped bars keep the three observed seeds visible without presenting n=3 as a confidence interval.",
            "comparisonContext": {"baseline": "Base-RAG = 0.6425", "grain": "training seed", "unit": "Q_main"},
            "type": "bar",
            "dataset": "seed_qmain",
            "sourceId": "seed-source",
            "encodings": {
                "x": {"field": "system", "type": "nominal", "label": "System"},
                "y": {
                    "field": "q_main",
                    "type": "quantitative",
                    "label": "Q_main",
                    "format": "number",
                },
                "color": {"field": "seed", "type": "nominal", "label": "Training seed"},
            },
            "combinationRationale": "Color separates the three observed training seeds within each system group.",
            "palette": {"kind": "categorical"},
            "settings": {"groupMode": "grouped", "showValues": True, "sort": "custom"},
            "referenceLines": [
                {"axis": "y", "value": 0.6425, "label": "Base-RAG", "color": "neutral", "lineStyle": "dashed"}
            ],
            "layout": "full",
        },
        {
            "id": "retrieval-contribution",
            "title": "Q_main with and without retrieval",
            "subtitle": "Matched RAFT and CLM adapters; the retrieval-aware variants use the shared fixed evidence pipeline",
            "showDescription": True,
            "intent": "comparison",
            "question": "Does parametric adaptation replace retrieved evidence?",
            "rationale": "Grouped zero-based bars make the on/off retrieval contrast explicit for both training paradigms.",
            "comparisonContext": {"grain": "training paradigm", "unit": "Q_main"},
            "type": "bar",
            "dataset": "retrieval_contribution",
            "sourceId": "retrieval-source",
            "encodings": {
                "x": {"field": "paradigm", "type": "nominal", "label": "Adaptation paradigm"},
                "y": {
                    "field": "q_main",
                    "type": "quantitative",
                    "label": "Q_main",
                    "format": "number",
                },
                "color": {"field": "retrieval_state", "type": "nominal", "label": "Evidence access"},
            },
            "combinationRationale": "Color separates matched retrieval-on and retrieval-off variants within each training paradigm.",
            "palette": {"kind": "categorical"},
            "settings": {"groupMode": "grouped", "showValues": True, "sort": "custom"},
            "layout": "full",
        },
        {
            "id": "evidence-coverage",
            "title": "Mean question score by gold-evidence coverage",
            "subtitle": "47 answerable questions; coverage is measured from the shared retrieved evidence, so group membership is identical across systems",
            "showDescription": True,
            "intent": "comparison",
            "question": "Where do adapter gains appear when retrieval fully, partially, or never covers the gold pages?",
            "rationale": "Grouped bars compare generator behavior within the same evidence-availability regimes.",
            "comparisonContext": {
                "denominator": "answerable questions only",
                "grain": "question",
                "unit": "mean per-question score",
            },
            "type": "bar",
            "dataset": "evidence_coverage",
            "sourceId": "coverage-source",
            "encodings": {
                "x": {"field": "coverage", "type": "ordinal", "label": "Gold-page coverage"},
                "y": {
                    "field": "mean_score",
                    "type": "quantitative",
                    "label": "Mean question score",
                    "format": "number",
                },
                "color": {"field": "system", "type": "nominal", "label": "System"},
                "tooltip": [
                    {"field": "n_questions", "type": "quantitative", "label": "Questions"}
                ],
            },
            "combinationRationale": "Color identifies the four generators compared within the same evidence-coverage regime.",
            "palette": {"kind": "categorical"},
            "settings": {"groupMode": "grouped", "showValues": True, "sort": "custom"},
            "layout": "full",
        },
        {
            "id": "type-deltas",
            "title": "Score change by answer type vs Base-RAG",
            "subtitle": "Positive cells favor the adapted system; deterministic and free-text columns retain their original scoring definitions",
            "showDescription": True,
            "intent": "comparison",
            "question": "Which answer types improve or regress under each adaptation recipe?",
            "rationale": "Grouped horizontal bars keep zero visible and make specialization and regressions easier to read than the duplicated absolute-score table.",
            "comparisonContext": {"baseline": "Base-RAG", "grain": "system × answer type", "unit": "score delta"},
            "type": "horizontalBar",
            "dataset": "type_deltas",
            "sourceId": "type-source",
            "encodings": {
                "x": {"field": "answer_type", "type": "ordinal", "label": "Answer type"},
                "y": {"field": "delta_vs_base", "type": "quantitative", "label": "Δ score vs Base-RAG"},
                "color": {"field": "system", "type": "nominal", "label": "System"},
                "tooltip": [
                    {"field": "delta_vs_base", "type": "quantitative", "label": "Δ vs Base-RAG"},
                    {"field": "n", "type": "quantitative", "label": "Questions"},
                ],
            },
            "combinationRationale": "Color identifies the adapted system within each answer-type group.",
            "palette": {"kind": "categorical"},
            "settings": {"groupMode": "grouped", "showValues": True, "sort": "custom"},
            "labels": {"values": "all"},
            "layout": "full",
        },
        {
            "id": "win-tie-loss",
            "title": "Question-level outcomes vs Base-RAG",
            "subtitle": "50 held-out questions per comparison; ties are shown explicitly rather than disappearing from pairwise win rates",
            "showDescription": True,
            "intent": "composition",
            "question": "Are aggregate gains broad across questions or concentrated in a small number of wins?",
            "rationale": "A 100% stacked horizontal bar exposes the dominant tie share and avoids misreading asymmetric pairwise win-rate cells.",
            "comparisonContext": {"denominator": "50 evaluation questions", "grain": "system comparison", "unit": "question count"},
            "type": "horizontalStackedBar100",
            "dataset": "win_tie_loss",
            "sourceId": "outcome-source",
            "encodings": {
                "x": {"field": "comparison", "type": "nominal", "label": "Comparison"},
                "y": {
                    "field": "count",
                    "type": "quantitative",
                    "label": "Questions",
                    "format": "number",
                },
                "color": {"field": "outcome", "type": "nominal", "label": "Outcome"},
                "tooltip": [
                    {"field": "share", "type": "quantitative", "label": "Share", "format": "percent"},
                    {"field": "n_questions", "type": "quantitative", "label": "Questions in comparison"},
                ],
            },
            "combinationRationale": "Color separates adapted-system wins, ties, and Base-RAG wins within each comparison.",
            "palette": {"kind": "categorical"},
            "settings": {"groupMode": "stacked100", "showValues": True, "sort": "custom"},
            "layout": "full",
        },
    ]


def build_artifact() -> dict:
    generated_at = datetime.now(timezone.utc).isoformat()
    main_results = read_csv(RESULTS / "EXP-006" / "main_results.csv")
    profiles, metadata, coverage = load_question_profiles()
    bootstrap_rows, bootstrap_summary = bootstrap_deltas(profiles, metadata)

    interval_lines = []
    for system in ("RAFT-RAG", "CLM-RAG", "Merge-RAG"):
        low, median, high = bootstrap_summary[system]
        interval_lines.append(
            f"- **{system}:** median ΔQ_main `{median:+.3f}`, paired bootstrap interval `[{low:+.3f}, {high:+.3f}]`."
        )

    datasets = {
        "bootstrap_deltas": bootstrap_rows,
        "seed_qmain": seed_rows(),
        "retrieval_contribution": retrieval_rows(main_results),
        "evidence_coverage": evidence_rows(profiles, coverage),
        "type_deltas": type_delta_rows(),
        "win_tie_loss": win_tie_loss_rows(),
    }
    write_derived_sources(datasets)

    sources = source_specs()
    charts = chart_specs()
    blocks = [
        {
            "id": "title",
            "type": "markdown",
            "body": "# Figure Review: Current and Proposed Visuals",
            "layout": "full",
        },
        {
            "id": "technical-summary",
            "type": "markdown",
            "body": (
                "## Короткий вывод\n\n"
                "Текущий набор уже хорошо показывает архитектуру и профиль RAFT/CLM, но слабо отвечает на два вопроса: насколько устойчивы небольшие aggregate gains и где заканчивается retrieval failure и начинается generation failure. Ниже сначала показаны исходные рисунки без изменений, затем — более диагностичный набор на тех же результатах."
            ),
            "layout": "full",
        },
        {
            "id": "current-section",
            "type": "markdown",
            "body": (
                "## 1. Что есть сейчас\n\n"
                "Пять рисунков из текущего draft собраны в исходном виде. Под каждым дана короткая оценка его аналитической роли; сами PNG не изменены."
            ),
            "layout": "full",
        },
        {
            "id": "current-figures",
            "type": "html",
            "body": current_figures_html(),
            "sourceId": "existing-figures",
            "layout": "full",
        },
        {
            "id": "proposed-section",
            "type": "markdown",
            "body": (
                "## 2. Предлагаемый набор\n\n"
                "Новые графики не добавляют новые эксперименты: они переиспользуют сохранённые system-level и per-question outputs. Их задача — сделать неопределённость, вклад retrieval и локальность улучшений видимыми прямо в основном narrative."
            ),
            "layout": "full",
        },
        {
            "id": "bootstrap-note",
            "type": "markdown",
            "body": (
                "### Aggregate gains с sampling uncertainty\n\n"
                "Box plot ниже показывает paired bootstrap distribution, а не variation между тремя training seeds. Интервалы описывают чувствительность результата к составу 50-question holdout и не заменяют независимый benchmark.\n\n"
                + "\n".join(interval_lines)
            ),
            "sourceId": "bootstrap-source",
            "layout": "full",
        },
        {"id": "bootstrap-chart", "type": "chart", "chartId": "bootstrap-delta", "layout": "full"},
        {
            "id": "seed-note",
            "type": "markdown",
            "body": (
                "### Training stochasticity остаётся отдельной осью\n\n"
                "Три исходных seed value показаны напрямую. Это не confidence interval: график нужен, чтобы не смешивать training variance с uncertainty по вопросам. Merge-RAG заметно чувствительнее к pairing исходных adapters."
            ),
            "sourceId": "seed-source",
            "layout": "full",
        },
        {"id": "seed-chart", "type": "chart", "chartId": "seed-qmain", "layout": "full"},
        {
            "id": "retrieval-note",
            "type": "markdown",
            "body": (
                "### Retrieval остаётся основным механизмом памяти\n\n"
                "Matched on/off comparison отвечает на RQ2 напрямую: RAFT и CLM adapters без evidence теряют примерно 0.41 и 0.48 Q_main соответственно. Этот эффект намного крупнее generator-side gains над Base-RAG."
            ),
            "sourceId": "retrieval-source",
            "layout": "full",
        },
        {"id": "retrieval-chart", "type": "chart", "chartId": "retrieval-contribution", "layout": "full"},
        {
            "id": "coverage-note",
            "type": "markdown",
            "body": (
                "### Fixed retrieval не означает одинаковую сложность evidence access\n\n"
                "Группы определены только по gold-page recall общего retriever. Full coverage содержит 30 answerable questions, partial — 5, none — 12. Средний per-question score здесь диагностический: он объединяет S_det и S_asst на уровне вопроса и не равен Q_main."
            ),
            "sourceId": "coverage-source",
            "layout": "full",
        },
        {"id": "coverage-chart", "type": "chart", "chartId": "evidence-coverage", "layout": "full"},
        {
            "id": "type-note",
            "type": "markdown",
            "body": (
                "### Профиль адаптации лучше читать как изменение относительно baseline\n\n"
                "Delta chart сохраняет sample sizes и сразу отделяет улучшения от регрессий. Это делает главный trade-off яснее: RAFT преимущественно помогает структурированному извлечению, CLM — free-text, а multi-name остаётся слабым местом."
            ),
            "sourceId": "type-source",
            "layout": "full",
        },
        {"id": "type-chart", "type": "chart", "chartId": "type-deltas", "layout": "full"},
        {
            "id": "outcome-note",
            "type": "markdown",
            "body": (
                "### Aggregate improvement не распределён равномерно по вопросам\n\n"
                "В каждой паре большинство вопросов заканчиваются ничьей. Win/tie/loss composition поэтому объясняет умеренный масштаб эффекта лучше, чем матрица win rates, где ties визуально исчезают."
            ),
            "sourceId": "outcome-source",
            "layout": "full",
        },
        {"id": "outcome-chart", "type": "chart", "chartId": "win-tie-loss", "layout": "full"},
        {
            "id": "scope-method",
            "type": "markdown",
            "body": (
                "## 3. Scope, method и ограничения\n\n"
                "**Scope.** Использованы те же 50 held-out questions, те же три training seeds и те же сохранённые predictions, что и в draft. Ни один model run не добавлен и основной Markdown не изменён.\n\n"
                "**Method.** Bootstrap стратифицирован по deterministic/free-text subset, чтобы при resampling сохранить определение Q_main. Evidence coverage взят из page-level grounding_recall Base-RAG и применён ко всем retrieval-aware systems, поскольку retrieved evidence фиксирован.\n\n"
                "**Limitations.** Bootstrap характеризует устойчивость внутри текущего holdout, а не переносимость на другую юрисдикцию или независимый question set. Partial-coverage group содержит только пять вопросов; её нужно читать как диагностическую, а не как самостоятельный benchmark."
            ),
            "layout": "full",
        },
        {
            "id": "recommendation",
            "type": "markdown",
            "body": (
                "## 4. Рекомендуемая замена в статье\n\n"
                "1. Оставить system schematic.\n"
                "2. Заменить текущие aggregate delta bars на bootstrap effect plot, а seed variability держать рядом или в appendix.\n"
                "3. Добавить retrieval on/off как прямой ответ на RQ2.\n"
                "4. Заменить absolute per-type heatmap на grouped delta chart.\n"
                "5. Оставить single-vs-multi figure, но добавить seed points/error bars в финальной статической версии.\n"
                "6. Перенести judge profile в appendix и заменить pairwise matrix на win/tie/loss composition."
            ),
            "layout": "full",
        },
        {
            "id": "further-questions",
            "type": "markdown",
            "body": (
                "## 5. Что ещё стоит проверить\n\n"
                "Главный открытый вопрос — сохраняются ли bootstrap intervals и evidence-coverage patterns на независимом legal QA holdout. Если нет нового набора вопросов, текущие графики следует трактовать как более честную визуализацию имеющихся результатов, а не как дополнительное подтверждение generalization."
            ),
            "layout": "full",
        },
    ]

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Figure Review: Current and Proposed Visuals",
            "description": "Side-by-side review of the current thesis figures and proposed diagnostic replacements.",
            "generatedAt": generated_at,
            "sources": sources,
            "charts": charts,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": datasets,
        },
        "sources": sources,
        "package_info": {
            "generator": "term-paper_3/visualization-comparison/build_report.py",
            "note": "Generated from frozen local experiment outputs; no model reruns.",
        },
    }


def main() -> None:
    artifact = build_artifact()
    output = HERE / "artifact.json"
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Artifact size: {output.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
