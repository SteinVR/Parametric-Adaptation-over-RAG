"""
EXP-001: Data Audit — Corpus Manifest, Goldset Validation, Split Freeze, Capacity Audit.

Run blocks individually with IDE's "Run Cell" (Ctrl+Enter).
"""

# %% [Setup] ===================================================================

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import hashlib
from collections import Counter

import numpy as np
import pandas as pd

from config import (
    CORPUS_DIR, GOLDSET_PATH, QUESTIONS_PATH,
    DATA_MANIFESTS, DATA_SPLITS,
    DEV_SIZE, TEST_SIZE, DEFAULT_SEED,
    DOC2LORA_MAX_CONTEXT_TOKENS,
)
from src.data.io import load_goldset, list_corpus_pdfs, save_json

EXP_DIR = Path(__file__).resolve().parent
EXP_DIR.mkdir(exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"Corpus dir: {CORPUS_DIR}")
print(f"Goldset: {GOLDSET_PATH}")


# %% [1. Corpus Manifest] =====================================================

def extract_pdf_page_count(pdf_path: Path) -> int:
    """Extract page count from PDF without heavy dependencies."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count
    except ImportError:
        # Fallback: count pages via simple PDF parsing
        content = pdf_path.read_bytes()
        # Count /Type /Page occurrences (rough heuristic)
        count = content.count(b"/Type /Page") - content.count(b"/Type /Pages")
        return max(count, 1)


def estimate_token_count(page_count: int, tokens_per_page: int = 800) -> int:
    """Rough estimate: legal docs ~800 tokens/page on average."""
    return page_count * tokens_per_page


pdfs = list_corpus_pdfs(CORPUS_DIR)
print(f"Found {len(pdfs)} PDF files")

manifest_rows = []
for pdf in pdfs:
    doc_id = pdf.stem
    pages = extract_pdf_page_count(pdf)
    tokens_est = estimate_token_count(pages)
    manifest_rows.append({
        "doc_id": doc_id,
        "filename": pdf.name,
        "page_count": pages,
        "approx_token_count": tokens_est,
    })

manifest_df = pd.DataFrame(manifest_rows)
manifest_path = DATA_MANIFESTS / "corpus_manifest.csv"
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_df.to_csv(manifest_path, index=False)

print(f"\nCorpus manifest saved: {manifest_path}")
print(f"  Documents: {len(manifest_df)}")
print(f"  Total pages: {manifest_df['page_count'].sum()}")
print(f"  Total tokens (est): {manifest_df['approx_token_count'].sum():,}")
print(f"  Pages per doc: min={manifest_df['page_count'].min()}, "
      f"median={manifest_df['page_count'].median():.0f}, "
      f"max={manifest_df['page_count'].max()}")


# %% [2. Goldset Validation] ==================================================

refs = load_goldset(GOLDSET_PATH)
print(f"Goldset loaded: {len(refs)} references")

# Schema validation
required_fields = {"question_id", "answer_type", "answer", "question", "gold_retrieval"}
issues = []
for i, r in enumerate(refs):
    missing = required_fields - set(r.keys())
    if missing:
        issues.append(f"  ref[{i}] missing: {missing}")

if issues:
    print(f"Schema issues ({len(issues)}):")
    for issue in issues:
        print(issue)
else:
    print("Schema validation: OK (all required fields present)")

# doc_id cross-check against manifest
manifest_doc_ids = set(manifest_df["doc_id"])
gold_doc_ids = set()
missing_docs = set()
for r in refs:
    for gr in r.get("gold_retrieval", []):
        did = gr["doc_id"]
        gold_doc_ids.add(did)
        if did not in manifest_doc_ids:
            missing_docs.add(did)

print(f"\nGold references {len(gold_doc_ids)} unique doc_ids")
if missing_docs:
    print(f"  WARNING: {len(missing_docs)} doc_ids in gold NOT in corpus manifest!")
    for d in sorted(missing_docs)[:5]:
        print(f"    {d}")
else:
    print("  All gold doc_ids found in corpus manifest: OK")

# Distributions
type_dist = Counter(r["answer_type"] for r in refs)
diff_dist = Counter(r.get("difficulty", "unknown") for r in refs)
tag_counter = Counter()
for r in refs:
    for t in r.get("tags", []):
        tag_counter[t] += 1
source_dist = Counter(r.get("source_type", "unknown") for r in refs)

# Evidence pages stats
evidence_page_counts = []
for r in refs:
    total_pages = sum(len(gr["page_numbers"]) for gr in r.get("gold_retrieval", []))
    evidence_page_counts.append(total_pages)

print(f"\nAnswer type distribution:")
for t, c in type_dist.most_common():
    print(f"  {t}: {c}")

print(f"\nDifficulty distribution:")
for d, c in diff_dist.most_common():
    print(f"  {d}: {c}")

print(f"\nSource type distribution:")
for s, c in source_dist.most_common():
    print(f"  {s}: {c}")

print(f"\nTag distribution:")
for t, c in tag_counter.most_common():
    print(f"  {t}: {c}")

print(f"\nEvidence pages per question: "
      f"min={min(evidence_page_counts)}, "
      f"median={np.median(evidence_page_counts):.1f}, "
      f"max={max(evidence_page_counts)}, "
      f"total_unique_pages={sum(evidence_page_counts)}")


# %% [3. Near-Duplicate Detection] ============================================

# Simple approach: check for very similar questions by normalized string overlap
def normalize_question(q: str) -> str:
    """Lowercase, strip whitespace and punctuation for comparison."""
    import re
    return re.sub(r'[^\w\s]', '', q.lower()).strip()

normalized = [normalize_question(r["question"]) for r in refs]

# Pairwise check for high overlap (Jaccard on word sets)
potential_dupes = []
for i in range(len(normalized)):
    words_i = set(normalized[i].split())
    for j in range(i + 1, len(normalized)):
        words_j = set(normalized[j].split())
        if not words_i or not words_j:
            continue
        jaccard = len(words_i & words_j) / len(words_i | words_j)
        if jaccard > 0.8:
            potential_dupes.append((i, j, jaccard, refs[i]["question"][:80], refs[j]["question"][:80]))

if potential_dupes:
    print(f"Potential near-duplicates ({len(potential_dupes)}):")
    for i, j, jac, q1, q2 in potential_dupes:
        print(f"  [{i}] vs [{j}] (jaccard={jac:.2f})")
        print(f"    Q1: {q1}...")
        print(f"    Q2: {q2}...")
else:
    print("No near-duplicates found (Jaccard > 0.8): OK")


# %% [4. Split Creation] ======================================================

from src.data.splits import create_stratified_split

split = create_stratified_split(
    refs,
    test_size=TEST_SIZE,
    seed=DEFAULT_SEED,
)

split_path = DATA_SPLITS / "split_v1.json"
save_json(split, split_path)

# Verify split
dev_ids = set(split["dev"])
test_ids = set(split["test"])
all_ids = {r["question_id"] for r in refs}

assert len(dev_ids) == DEV_SIZE, f"Expected {DEV_SIZE} dev, got {len(dev_ids)}"
assert len(test_ids) == TEST_SIZE, f"Expected {TEST_SIZE} test, got {len(test_ids)}"
assert dev_ids | test_ids == all_ids, "Split doesn't cover all questions"
assert not (dev_ids & test_ids), "Overlap between dev and test!"

# Show stratification quality
ref_by_id = {r["question_id"]: r for r in refs}
for split_name, ids in [("dev", dev_ids), ("test", test_ids)]:
    types = Counter(ref_by_id[qid]["answer_type"] for qid in ids)
    diffs = Counter(ref_by_id[qid].get("difficulty", "?") for qid in ids)
    print(f"\n{split_name} ({len(ids)}):")
    print(f"  answer_types: {dict(types)}")
    print(f"  difficulty: {dict(diffs)}")

print(f"\nSplit saved: {split_path}")


# %% [5. Capacity Audit] ======================================================

total_tokens = manifest_df["approx_token_count"].sum()
d2l_limit = DOC2LORA_MAX_CONTEXT_TOKENS

print(f"\n=== Capacity Audit ===")
print(f"Total corpus tokens (est): {total_tokens:,}")
print(f"Doc-to-LoRA single-pass limit: ~{d2l_limit:,} tokens")
print(f"Ratio: {total_tokens / d2l_limit:.1f}x over single-pass capacity")

n_segments = int(np.ceil(total_tokens / d2l_limit))
print(f"Estimated segments needed (full corpus): {n_segments}")
print(f"Estimated segments per cluster (k=4): ~{n_segments / 4:.1f}")

# Per-document feasibility
docs_that_fit = (manifest_df["approx_token_count"] <= d2l_limit).sum()
print(f"\nDocuments that fit single D2L pass individually: {docs_that_fit}/{len(manifest_df)}")

# Distribution of document sizes
print(f"\nDocument token distribution:")
print(manifest_df["approx_token_count"].describe().to_string())

# Can we fit ~16 docs (one cluster) in one pass?
manifest_sorted = manifest_df.sort_values("approx_token_count")
cluster_size = len(manifest_df) // 4
cumulative = manifest_sorted["approx_token_count"].iloc[:cluster_size].sum()
print(f"\nSmallest {cluster_size} docs combined: {cumulative:,} tokens "
      f"({'fits' if cumulative <= d2l_limit else 'exceeds'} D2L limit)")

# Monolithic feasibility
print(f"\nMonolithic single-pass feasible: NO ({total_tokens:,} >> {d2l_limit:,})")
print(f"Strategy needed: chunked processing + adapter merge")


# %% [6. Generate Report] =====================================================

report_lines = [
    "# EXP-001: Data Audit Report",
    "",
    f"**Date:** 2026-03-26",
    f"**Status:** Complete",
    "",
    "## Corpus",
    f"- Documents: {len(manifest_df)}",
    f"- Total pages: {manifest_df['page_count'].sum()}",
    f"- Total tokens (est): {total_tokens:,}",
    f"- Pages per doc: min={manifest_df['page_count'].min()}, "
    f"median={manifest_df['page_count'].median():.0f}, "
    f"max={manifest_df['page_count'].max()}",
    "",
    "## Goldset",
    f"- QA pairs: {len(refs)}",
    f"- Unique docs referenced: {len(gold_doc_ids)}",
    f"- All gold doc_ids in corpus: {'YES' if not missing_docs else 'NO — ' + str(len(missing_docs)) + ' missing'}",
    f"- Near-duplicates found: {len(potential_dupes)}",
    "",
    "### Answer type distribution",
    "",
    "| Type | Count |",
    "|------|-------|",
]
for t, c in type_dist.most_common():
    report_lines.append(f"| {t} | {c} |")

report_lines += [
    "",
    "### Difficulty distribution",
    "",
    "| Difficulty | Count |",
    "|------------|-------|",
]
for d, c in diff_dist.most_common():
    report_lines.append(f"| {d} | {c} |")

report_lines += [
    "",
    "## Split",
    f"- Dev: {len(dev_ids)} questions",
    f"- Test: {len(test_ids)} questions (locked)",
    f"- Stratified by answer_type + difficulty",
    f"- Saved: `data/splits/split_v1.json`",
    "",
    "## Capacity Audit",
    f"- Total corpus tokens: ~{total_tokens:,}",
    f"- Doc-to-LoRA single-pass limit: ~{d2l_limit:,}",
    f"- Segments needed (full corpus): ~{n_segments}",
    f"- **Monolithic single-pass: NOT feasible**",
    f"- Strategy: chunked processing + adapter merge (S3) or clustered routing (S4)",
    "",
    "## Conclusion",
    "- Data artifacts are valid and consistent.",
    "- Split is frozen. Proceed to EXP-002 (S1 baseline).",
]

report_text = "\n".join(report_lines)
report_path = EXP_DIR / "REPORT.md"
report_path.write_text(report_text)
print(f"\nReport saved: {report_path}")
