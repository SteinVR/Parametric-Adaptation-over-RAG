"""
Split generation utilities.
"""

import re
import random
from typing import Any


def find_near_duplicate_groups(
    refs: list[dict[str, Any]],
    jaccard_threshold: float = 0.8,
) -> list[list[str]]:
    """Find groups of near-duplicate questions by word-level Jaccard similarity.

    Returns list of groups (each group is a list of question_ids that should stay together).
    """
    def normalize(q: str) -> set[str]:
        return set(re.sub(r'[^\w\s]', '', q.lower()).strip().split())

    word_sets = [(r["question_id"], normalize(r["question"])) for r in refs]

    # Union-Find for grouping
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(word_sets)):
        qid_i, words_i = word_sets[i]
        for j in range(i + 1, len(word_sets)):
            qid_j, words_j = word_sets[j]
            if not words_i or not words_j:
                continue
            jaccard = len(words_i & words_j) / len(words_i | words_j)
            if jaccard > jaccard_threshold:
                union(qid_i, qid_j)

    # Collect groups
    groups_map: dict[str, list[str]] = {}
    all_ids = [r["question_id"] for r in refs]
    for qid in all_ids:
        root = find(qid)
        groups_map.setdefault(root, []).append(qid)

    # Return only groups with >1 member
    return [g for g in groups_map.values() if len(g) > 1]


def create_stratified_split(
    refs: list[dict[str, Any]],
    test_size: int = 30,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Create stratified dev/test split on question_id.

    Groups near-duplicates so they always land in the same split.
    Stratifies by (answer_type, difficulty).

    Returns dict with 'dev' and 'test' lists of question_ids.
    """
    rng = random.Random(seed)

    # Find near-duplicate groups
    dupe_groups = find_near_duplicate_groups(refs)
    grouped_ids: set[str] = set()
    group_map: dict[str, list[str]] = {}  # representative_id -> all ids in group
    for group in dupe_groups:
        rep = group[0]
        group_map[rep] = group
        grouped_ids.update(group)

    ref_by_id = {r["question_id"]: r for r in refs}

    # Build allocation units: either a group (treated as one unit) or a singleton
    units: list[tuple[str, list[str]]] = []  # (stratum_key_str, [question_ids])
    seen: set[str] = set()
    for r in refs:
        qid = r["question_id"]
        if qid in seen:
            continue
        if qid in group_map:
            ids = group_map[qid]
            seen.update(ids)
            key = (r["answer_type"], r.get("difficulty", "unknown"))
            units.append((f"{key[0]}_{key[1]}", ids))
        elif qid not in grouped_ids:
            seen.add(qid)
            key = (r["answer_type"], r.get("difficulty", "unknown"))
            units.append((f"{key[0]}_{key[1]}", [qid]))

    # Group units by stratum
    strata: dict[str, list[list[str]]] = {}
    for stratum_key, ids in units:
        strata.setdefault(stratum_key, []).append(ids)

    for unit_list in strata.values():
        rng.shuffle(unit_list)

    total_questions = sum(len(ids) for _, ids in units)
    test_ratio = test_size / total_questions

    test_ids: list[str] = []
    dev_ids: list[str] = []

    for key in sorted(strata.keys()):
        unit_list = strata[key]
        stratum_total = sum(len(u) for u in unit_list)
        target_test = max(1, round(stratum_total * test_ratio))

        current_test = 0
        for unit in unit_list:
            if current_test < target_test and current_test + len(unit) <= target_test + 1:
                test_ids.extend(unit)
                current_test += len(unit)
            else:
                dev_ids.extend(unit)

    # Adjust to exact test_size
    rng.shuffle(dev_ids)
    while len(test_ids) > test_size:
        moved = test_ids.pop()
        dev_ids.append(moved)
    while len(test_ids) < test_size:
        moved = dev_ids.pop()
        test_ids.append(moved)

    rng.shuffle(test_ids)
    rng.shuffle(dev_ids)

    return {
        "dev": dev_ids,
        "test": test_ids,
        "near_duplicate_groups": [[qid for qid in g] for g in dupe_groups],
    }
