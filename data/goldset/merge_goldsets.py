"""
Merge two goldset batches into a single unified benchmark.
Produces:
  - goldset.benchmark.json (combined benchmark)
  - goldset.questions.json (questions-only list)
  - goldset_audit.json (quality audit report)
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

GS1_PATH = PROJECT_ROOT / "data/corpus4-100/dev-gold-corpus4-100-v1.benchmark.json"
GS2_PATH = PROJECT_ROOT / "data/corpus4_2-100/dev-gold-corpus4_2-100-v1.benchmark.json"
OUT_DIR = PROJECT_ROOT / "data/goldset"

DOC_NAMES = {
    "302a0bd8d67775e8dc5960ecec7879be566300d8b32c4b0153ba15ebdb279425": "doc1_general_partnership_law",
    "04be93255ec4b88e6e6c65c8e9335e10729723c2637452f3ad66d5b3dbe87513": "doc2_crs_regulations",
    "3f8a5ea0e051ba3af7a02da340c911fe0970ebece6c852c3e61a10c00cac6d6e": "doc3_techteryx_v_aria",
    "ad76dc7093851d116c8878802b815608adceed54f6b5195ae04ffec28ae25d32": "doc4_bond_v_tr88house",
    "536bbce854b9406cc22697e04fcdabd645e030c0e55b918252643b00e0b2b25f": "doc5_personal_property_law",
    "3fa59589a91bf4913703ba0eedd08faa128948285b8a9d085bd7422248abe6c5": "doc6_securities_regulations",
    "5d3df6d69fac3ef91e13ac835b43a35e9e434fbc7e72ea5c01e288d69b66e6a2": "doc7_ozias_v_obadiah",
    "437568a801115019fe8278385c0484bdf07ab86f9a499ecaba2b7969b37c764b": "doc8_lxt_v_sir_realestate",
}

# Load
with open(GS1_PATH) as f:
    gs1 = json.load(f)
with open(GS2_PATH) as f:
    gs2 = json.load(f)

refs1 = gs1["references"]
refs2 = gs2["references"]
all_refs = refs1 + refs2

# Validate no ID collision
ids = [r["question_id"] for r in all_refs]
assert len(ids) == len(set(ids)), "Question ID collision!"

# Near-duplicate detection
def normalize_words(q: str) -> set[str]:
    return set(re.sub(r"[^\w\s]", "", q.lower()).strip().split())

near_dupe_groups = []
seen = set()
for i in range(len(all_refs)):
    if i in seen:
        continue
    words_i = normalize_words(all_refs[i]["question"])
    group = [i]
    for j in range(i + 1, len(all_refs)):
        if j in seen:
            continue
        words_j = normalize_words(all_refs[j]["question"])
        if words_i and words_j:
            jaccard = len(words_i & words_j) / len(words_i | words_j)
            if jaccard > 0.75:
                group.append(j)
                seen.add(j)
    if len(group) > 1:
        seen.update(group)
        near_dupe_groups.append([all_refs[idx]["question_id"] for idx in group])

# Build combined benchmark
combined = {
    "benchmark_id": "goldset-200-v1",
    "title": "Combined 8-Document Goldset (200 QA)",
    "description": f"Merged from {gs1['benchmark_id']} and {gs2['benchmark_id']}. "
                   f"8 DIFC documents, {len(all_refs)} questions.",
    "corpus_documents": DOC_NAMES,
    "references": all_refs,
}

# Build questions-only list
questions = [
    {"id": r["question_id"], "question": r["question"], "answer_type": r["answer_type"]}
    for r in all_refs
]

# Build audit
audit = {
    "total_questions": len(all_refs),
    "answer_type_distribution": dict(Counter(r["answer_type"] for r in all_refs).most_common()),
    "difficulty_distribution": dict(Counter(r.get("difficulty", "unknown") for r in all_refs).most_common()),
    "document_coverage": {},
    "multi_document_questions": sum(1 for r in all_refs if len({gr["doc_id"] for gr in r.get("gold_retrieval", [])}) > 1),
    "null_unanswerable": sum(1 for r in all_refs if r["answer"] is None),
    "near_duplicate_groups": near_dupe_groups,
    "cross_batch_multi_doc": 0,
}

# Document coverage
doc_counter = Counter()
for r in all_refs:
    for gr in r.get("gold_retrieval", []):
        doc_counter[gr["doc_id"]] += 1
audit["document_coverage"] = {
    DOC_NAMES.get(d, d): c for d, c in doc_counter.most_common()
}

# Save
with open(OUT_DIR / "goldset.benchmark.json", "w") as f:
    json.dump(combined, f, indent=2, ensure_ascii=False)

with open(OUT_DIR / "goldset.questions.json", "w") as f:
    json.dump(questions, f, indent=2, ensure_ascii=False)

with open(OUT_DIR / "goldset_audit.json", "w") as f:
    json.dump(audit, f, indent=2, ensure_ascii=False)

print(f"Merged: {len(all_refs)} questions")
print(f"Saved to: {OUT_DIR}")
print(f"Near-duplicate groups: {len(near_dupe_groups)}")
print(f"Multi-doc: {audit['multi_document_questions']}")
print(f"Null: {audit['null_unanswerable']}")
