# Spec Cross-Check Report

**Reviewer:** Victor (apm-code-reviewer)
**Date:** 2026-03-28
**Scope:** SPEC_EXP-002..008 vs ARCHITECTURE.md, SPEC-systems.md, SPEC-data.md, SPEC-evaluation.md

---

## Summary Verdict

**CHANGES REQUIRED** — 6 issues found (2 P1, 3 P2, 1 P3). No P0 blocking bugs. Core logic is sound and internally consistent across most specs. Issues are concentrated around: a wave-assignment mismatch, a missing dependency on EXP-001, a TBD that must be resolved, a missing grounding metric in EXP-008, and two minor ambiguities.

---

## Issue Catalogue

### P1 — High Impact

---

**[P1-A] EXP-003/004 Wave Assignment Contradicts TASKS.md and Wave Semantics**

- **File:** `SPEC_EXP-003.md` line 3, `SPEC_EXP-004.md` line 3
- **Problem:** Both specs declare `Wave: 2`. TASKS.md correctly lists EXP-003 and EXP-004 in Wave 2. However, ARCHITECTURE.md §8 (Experiment Phases) lists EXP-003 as dependent on EXP-002, and EXP-004 as dependent on EXP-002 — which is Wave 1. Waves execute sequentially (CLAUDE.md), so Wave 2 tasks can only begin after Wave 1 completes. This is actually correct. **The real problem** is that EXP-003 spec says `Depends on: EXP-002 (prompt template, chunking)` — meaning EXP-003 cannot run in parallel with EXP-002 even within a wave transition, but it also can't run in parallel with EXP-004 because EXP-004 blocks EXP-005a/b, which EXP-003 does not. This is fine. However: **EXP-003 step 5 says** "S2 receives retrieved chunks (same retriever as S1)." This implies S2 at inference time uses the S1 retriever — but SPEC-systems.md S2 section says "no grounding — no retrieval at inference." These two are directly contradictory.
- **Impact:** If EXP-003 step 5 is taken literally, S2 gets retrieval context at inference, making it a hybrid system rather than a purely supervised parametric one. This would invalidate the paradigm comparison and contaminate S2's results.
- **Fix:** Resolve the contradiction. SPEC-systems.md is more authoritative here. EXP-003 step 5 should be rewritten: "Evaluation: S2 generates answer from question only (no retrieved chunks), using the trained adapter. Evaluate on dev set."

---

**[P1-B] EXP-002 Missing Dependency on EXP-001**

- **File:** `SPEC_EXP-002.md` line 3
- **Problem:** `Depends on: EXP-001 (split)` — this is listed correctly. However, EXP-002 says "Baseline metrics on 160 dev questions" in the Output section, which presupposes the 160/40 split produced by EXP-001. This part is fine. The issue is that **SPEC_EXP-001.md does not exist** in `memory_bank/specs/`, but TASKS.md references `./specs/SPEC_EXP-001.md`. EXP-001 is marked as complete `[x]` in TASKS.md, so the missing spec file is not a blocker, but it means the dependency chain is not fully documented. If EXP-001 is complete, its outputs (`data/goldset/`, `data/splits/split_v1.json`) must be present for all downstream specs to be executable.
- **Impact:** Anyone picking up EXP-002 has no spec to verify EXP-001 completion criteria against. Weak but real documentation gap.
- **Fix:** Either create a minimal retrospective SPEC_EXP-001.md (inputs, outputs, frozen artifacts) or add a note in EXP-002's Depends-on line: "EXP-001 complete — artifacts at `data/splits/split_v1.json`, `data/goldset/goldset.benchmark.json`."

---

### P2 — Medium Impact

---

**[P2-A] S3 Merge Strategy Marked TBD — Must Be Resolved Before EXP-005b**

- **File:** `SPEC_EXP-004.md` (Merge Strategy Candidates section), `SPEC_EXP-005b.md` line 12 ("same strategy selected in EXP-004")
- **Problem:** EXP-004 correctly marks merge strategy as "TBD at EXP-004 feasibility." This is appropriate since it's an experimental decision. However, SPEC_EXP-005b hard-depends on that decision being made: "merge its 2 doc adapters (same strategy selected in EXP-004)." The TBD is correctly scoped — it will be decided during EXP-004 execution. **However**, SPEC_EXP-004 has no explicit output artifact for the selected merge strategy (no mention of a frozen config file or decision log). EXP-005b will inherit an undocumented decision.
- **Impact:** If the merge strategy decision is not written down as a frozen output of EXP-004, EXP-005b has no authoritative source to reference. Reproducibility risk.
- **Fix:** Add to EXP-004 Output section: "Frozen merge strategy documented in `experiments/EXP-004/REPORT.md` (section: Frozen Decisions) and referenced as the canonical strategy for S4 cluster merges."

---

**[P2-B] EXP-006 Wave Assignment Inconsistent with ARCHITECTURE.md**

- **File:** `SPEC_EXP-006.md` line 3
- **Problem:** EXP-006 declares `Wave: 3`. TASKS.md also places it in Wave 3 (Routing + Comparison). ARCHITECTURE.md §8 shows EXP-006 with goal "Main comparison S1-S4 on dev." This is consistent. However, EXP-006 `Depends on: EXP-002..005b` — EXP-005a and EXP-005b are also Wave 3. Within a wave, tasks execute in parallel (CLAUDE.md). This means EXP-006 **cannot** run in parallel with EXP-005a/005b even though they share Wave 3. EXP-006 must wait for both 005a and 005b to complete. Placing EXP-006 in the same wave as its blockers is a structural inconsistency: it implies it could be parallelized, but it cannot.
- **Impact:** A subagent executing Wave 3 in parallel would incorrectly attempt to start EXP-006 simultaneously with EXP-005a/005b, which would fail because EXP-006 needs their outputs.
- **Fix:** Move EXP-006 to Wave 4 (or create a Wave 3b). Alternatively, document explicitly in EXP-006: "Runs after EXP-005a and EXP-005b complete; cannot be parallelized with them." TASKS.md should be updated accordingly. Since EXP-007 currently blocks on EXP-006, this cascade would mean Wave 4 contains EXP-006, EXP-007, EXP-008 — or a new intermediate wave is introduced.

---

**[P2-C] EXP-008 Missing Grounding Metric (G) in Pipeline**

- **File:** `SPEC_EXP-008.md` line 10
- **Problem:** EXP-008 pipeline step 2 says "Score with same metrics as EXP-006." EXP-006 does not include grounding G (it covers S1-S4, and only S1 and S5 have grounding per SPEC-evaluation.md §Retrieval-Aware Metrics). However, EXP-008 also evaluates S5 (line 9 lists "S1, S2, S3, S4-doc, S4-cluster, S5"). S5 requires grounding G, but the reference to "same metrics as EXP-006" would omit it, since EXP-006 doesn't score S5 and doesn't mention G explicitly in its metrics section.
- **Impact:** Final paper tables (the explicit goal of EXP-008) would be missing G for S5 on the test set if this ambiguity is not resolved.
- **Fix:** EXP-008 step 2 should read: "Score with same metrics as EXP-006, plus G (F_β=2.5) for S1 and S5." Or reference SPEC-evaluation.md directly.

---

### P3 — Low Impact / Clarity

---

**[P3-A] EXP-005a Generation Step Omits Context — Ambiguous vs SPEC-systems.md**

- **File:** `SPEC_EXP-005a.md` line 13
- **Problem:** Step 4 says "selected adapter + question (no retrieved chunks) → answer." SPEC-systems.md S4-doc section does not explicitly state whether S4-doc operates with or without retrieval — it describes only the routing mechanism. The parenthetical "(no retrieved chunks)" in EXP-005a is the only place this is stated. This is likely intentional (S4-doc is parametric, not hybrid), but since it contradicts the parallel ambiguity in EXP-003 (see P1-A), the lack of an explicit statement in SPEC-systems.md S4 section creates a gap.
- **Impact:** Low — EXP-005a is self-consistent. The gap is in SPEC-systems.md not being explicit about inference-time context for S3/S4.
- **Fix:** Add one line to SPEC-systems.md S3 and S4 sections: "Inference: no retrieved context — adapter parameters only."

---

## Factual Consistency Check

All numeric constants checked:

| Constant | ARCHITECTURE | SPEC-data | SPEC-eval | EXP specs | Status |
|----------|-------------|-----------|-----------|-----------|--------|
| 200 QA pairs | Yes | Yes | Yes | EXP-006 (160 dev implied) | OK |
| 160 dev / 40 test | Yes | Yes | — | EXP-002,003,006,007,008 | OK |
| 8 docs | Yes | Yes | — | EXP-002,004,005a,005b | OK |
| 3 seeds (S2) | Yes | Yes | — | EXP-003 | OK |
| k=4 clusters | Yes | — | — | EXP-005b | OK |
| F_β=2.5 (G) | Yes | — | Yes | EXP-002,007 | OK |
| gpt-5.4-mini judge | Yes | — | Yes | Not mentioned in EXP specs | OK (inherited) |
| 4-bit NF4 | Yes | Yes | — | EXP-003 | OK |

No factual numeric contradictions found.

---

## Dependency Chain Verification

```
EXP-001 (done) → EXP-002 (Wave 1)
EXP-002 → EXP-003 (Wave 2), EXP-004 (Wave 2)
EXP-004 → EXP-005a (Wave 3), EXP-005b (Wave 3)
EXP-002 → EXP-006 (Wave 3) [also needs 003, 005a, 005b — WAVE CONFLICT: see P2-B]
EXP-006 → EXP-007 (Wave 4)
EXP-007 → EXP-008 (Wave 4)
```

No circular dependencies. The chain is acyclic. The only structural issue is EXP-006 in Wave 3 (P2-B).

---

## System ID Consistency Check

| ID | ARCHITECTURE | SPEC-systems | EXP specs | Status |
|----|-------------|--------------|-----------|--------|
| S1 | Yes | Yes | EXP-002,006,007,008 | OK |
| S2 | Yes | Yes | EXP-003,006,007,008 | OK |
| S3 | Yes | Yes | EXP-004,006,008 | OK |
| S4-doc | Yes | Yes | EXP-005a,006,008 | OK |
| S4-cluster | Yes | Yes | EXP-005b,006,008 | OK |
| S5 / S5a / S5b | Yes | Yes | EXP-007,008 | OK |

No ID inconsistencies. "S4" (without suffix) appears in ARCHITECTURE.md §3 hypotheses but is used as an umbrella term, not a distinct system ID — acceptable.

---

## TBD Resolution Status

| TBD | Location | Status |
|-----|----------|--------|
| Top-k for retrieval | SPEC-systems S1, EXP-002 Key Decisions | Appropriately deferred to EXP-002 execution |
| Chunking strategy | EXP-002 Key Decisions | Appropriately deferred |
| Embedding model | EXP-002 Key Decisions | Appropriately deferred (shared constraint documented) |
| Distractor policy | SPEC-data, EXP-003 Key Decisions | Appropriately deferred to EXP-003 |
| Merge strategy | EXP-004 | Appropriately deferred — but output artifact missing (see P2-A) |
| D2L generation parameters | EXP-004 | Appropriately deferred |

No TBDs that should already be resolved given current project state (EXP-001 complete, EXP-002 not started).

---

## Final Verdict

**CHANGES REQUIRED**

Priority order for fixes:
1. **P1-A** (EXP-003 step 5): resolve S2 inference contradiction immediately — this affects the validity of the paradigm comparison.
2. **P2-B** (EXP-006 wave placement): fix before subagent execution of Wave 3 to avoid parallel launch errors.
3. **P2-C** (EXP-008 missing G for S5): fix before EXP-008 execution.
4. **P2-A** (EXP-004 merge strategy output artifact): fix before EXP-005b starts.
5. **P1-B** (missing SPEC_EXP-001.md): low operational risk since EXP-001 is complete; can be addressed as documentation cleanup.
6. **P3-A** (SPEC-systems.md S3/S4 inference context): minor clarity fix, low priority.
