# Spec Critical Review Report

**Reviewer:** Victor (apm-code-reviewer)
**Date:** 2026-03-28
**Scope:** SPEC_EXP-002..008 vs ARCHITECTURE.md, SPEC-systems.md, SPEC-evaluation.md, config.py
**Check categories:** Contradictions, Vagueness, Logic, Missing pieces, config.py mismatches

---

## Verdict: CHANGES REQUIRED

14 findings (2 P0, 4 P1, 5 P2, 3 P3). Two P0 blockers must be resolved before any scoring run.

---

## P0 — Blockers

### [P0-1] S2 trains on dev questions, then is benchmarked on those same dev questions

**Files:** SPEC_EXP-003.md lines 9-25; SPEC_EXP-006.md pipeline step 3; ARCHITECTURE.md §4

RAFT dataset is built from "160 dev questions." S2 is evaluated on those same 160 dev questions in EXP-003 Metrics and EXP-006. The 160/40 split exists precisely to prevent this (ARCHITECTURE.md §4: "Split needed for S2 leakage prevention"). Every S2 dev metric is inflated by training-set evaluation.

**Impact:** S2 results in EXP-006 cross-paradigm comparison are invalid. The paper's central comparative claim cannot be supported.

**Fix:** Option A (preferred): train S2 on all 160 dev, report S2 only in EXP-008 on 40 locked test questions. Exclude S2 from EXP-006 dev table or label its numbers as "train-set performance." Option B: split dev into 120/40, train on 120. Option A preserves training data.

---

### [P0-2] Judge prompt for gpt-5.4-mini is not defined anywhere

**Files:** SPEC-evaluation.md lines 33-48; config.py lines 99-100

The rubric (5 binary criteria) exists in prose. The actual system prompt, user message template with {question}/{reference_answer}/{system_answer} slots, output format instruction, and JSON parsing schema are absent from all files. config.py has JUDGE_MODEL and JUDGE_REASONING but no prompt reference.

**Impact:** S_asst is not reproducible. Contributes 30% of Q_main for all 53 free_text questions. The 10% manual audit has no baseline.

**Fix:** Write verbatim judge prompt into SPEC-evaluation.md §Free-Text Scoring. Add JUDGE_PROMPT_PATH to config.py. Required before any S_asst score is recorded.

---

## P1 — High Impact

### [P1-1] RAFT distractor-only examples produce broken training signal for deterministic types

**File:** SPEC_EXP-003.md lines 19-23

32 examples (20% of training) have no gold context; model must produce correct factual answers from "memorized knowledge." A 2B model with 128 training examples cannot recall DIFC-specific dates, case numbers, or names. The original RAFT paper does not remove the gold chunk — it tests robustness to distractors while gold is present. This deviation trains the model to hallucinate or refuse on out-of-distribution retrieval failures.

**Impact:** 20% of training data has inverted learning signal for deterministic types. S2 adapter quality degraded.

**Fix:** Remove distractor-only examples (100% oracle format). If retained, restrict to free_text type only where null/refusal is a valid signal.

---

### [P1-2] No-retrieval prompt has no null instruction — 9 null questions will systematically score 0 for S3/S4

**Files:** SPEC_EXP-004.md lines 43-52; SPEC-evaluation.md lines 22-26

Prompt says "Answer the question based on your knowledge." S_det for null type: both null → 1.0, one null → 0.0. S1 RAG prompt explicitly says "respond with null." S3/S4 have no equivalent instruction. The model will hallucinate rather than return null. This is a systematic prompt engineering bias, not a capability difference.

**Impact:** S3/S4 score 0 on all 9 null questions by design. Comparison against S1 is unfair on this question type.

**Fix:** Add "If you do not know the answer, respond with null." to the no-retrieval prompt in EXP-004. Propagate to EXP-005a/005b.

---

### [P1-3] Grounding metric G scope contradicts across 4 documents

**Files:** SPEC-evaluation.md line 54; SPEC_EXP-003.md line 63; SPEC_EXP-008.md line 10; SPEC_EXP-006.md Table 1

SPEC-evaluation.md: G applies to S1 and S5 only.
EXP-003 Metrics: explicitly lists G for S2.
EXP-008 step 2: G for S1, S2, S5.
EXP-006 Table 1: G column for all systems without qualification.
Three documents give three different answers.

**Impact:** Final comparison table will have inconsistent G columns. Some reports compute G for S2, others don't.

**Fix:** S2 uses retrieval at inference → G is computable for S2. Decide: include S2 in G scoring everywhere. Update SPEC-evaluation.md to say "S1, S2, S5." Remove G from S3/S4 everywhere and mark N/A.

---

### [P1-4] S2 hyperparameter sweep: SPEC-systems.md says sweep; EXP-003 and config.py say fixed values

**Files:** SPEC-systems.md lines 24-29; SPEC_EXP-003.md lines 35-48; config.py lines 61-70

SPEC-systems.md: Alpha sweep [16,32,64], LR sweep [5e-5,1e-4,2e-4,4e-4], early stopping on dev Q_main.
EXP-003: "No hyperparameter sweep. Standard QLoRA defaults."
config.py: QLORA_ALPHA=32, QLORA_LR=2e-4 (single values).

**Impact:** Direct contradiction across three documents. Implementer following SPEC-systems.md will build sweep infrastructure that EXP-003 cancels.

**Fix:** Strike the sweep from SPEC-systems.md. Add: "Sweep deferred; fixed defaults used per EXP-003. Acknowledged as limitation."

---

## P2 — Medium Impact

### [P2-1] EXP-004 merge fallback threshold miscalibrated — nearly never triggers

**File:** SPEC_EXP-004.md lines 31-34

Fallback at "Q_main < 50% of mean per-doc Q_main." If per-doc Q_main = 0.40, threshold = 0.20. Random baseline for goldset type distribution ≈ 0.35-0.40. The "Q_main < random baseline" viability criterion means S3 is declared viable at any score above chance — including a merged adapter that adds zero value over the base model.

**Impact:** S3 nearly always declared viable regardless of merge quality. Fallback to weighted average may never trigger.

**Fix:** Redefine threshold as "monolithic Q_main < S1 dev Q_main × 0.6." Compute explicit random baseline from goldset type distribution and include in EXP-004 as a constant.

---

### [P2-2] Page-level grounding G not guaranteed computable — chunk metadata survival unverified

**Files:** SPEC_EXP-002.md lines 11-12; SPEC-evaluation.md lines 54-64

EXP-002 uses a "hierarchical page → chunk → subchunk system (frozen block, integrated as-is)." G requires (doc_id, page_number) on every chunk in the FAISS index. Whether this metadata survives the chunker is not verified. "Frozen block, integrated as-is" means no inspection of output schema.

**Impact:** If page metadata is dropped, G silently returns 0 for all questions — looks like retrieval failure, is actually instrumentation failure.

**Fix:** Add explicit data contract to EXP-002 Step 2: "Every chunk/subchunk in FAISS must carry doc_id and page_number. Verify before indexing." Add an assertion in the indexing code.

---

### [P2-3] EXP-006 Wave 3 alongside its own blockers EXP-005a/005b — parallel execution fails

**Files:** SPEC_EXP-006.md line 3; SPEC_EXP-005a.md line 3; SPEC_EXP-005b.md line 3

All three declare Wave 3. CLAUDE.md: tasks within a wave execute in parallel. EXP-006 depends on both EXP-005a and EXP-005b. A parallel executor will launch all three simultaneously; EXP-006 will immediately fail on missing inputs.

**Impact:** Automated or subagent execution of Wave 3 will crash EXP-006.

**Fix:** Move EXP-006 to Wave 4, EXP-007 to Wave 5, EXP-008 to Wave 6. Update all spec headers and TASKS.md.

---

### [P2-4] RAFT distractor count mismatch: spec says 3 for distractor-only, config.py constant = 2

**Files:** SPEC_EXP-003.md lines 15-23; config.py line 73

Oracle examples: [gold_chunk, distractor_1, distractor_2] — 2 distractors.
Distractor-only examples: [distractor_1, distractor_2, distractor_3] — 3 distractors.
RAFT_N_DISTRACTORS = 2 — single constant for both cases.

**Impact:** Distractor-only examples built with 2 distractors instead of 3. Training data does not match spec.

**Fix:** Unify at 2 distractors for both example types (update spec), or add two separate constants to config.py.

---

### [P2-5] Best adapter selection in EXP-006 uses S2 dev Q_main contaminated by P0-1

**File:** SPEC_EXP-006.md lines 29-31

The best adapter for S5 is "rank by dev Q_main." If S2 trained on dev, its dev Q_main is inflated → S2 likely selected → S5 built on contaminated foundation → S5 dev results also inflated.

**Impact:** S5 may be inadvertently unfair. Cascades from P0-1 if not resolved.

**Fix:** Resolves automatically if P0-1 is fixed. Add explicit note: "Best adapter selection excludes S2 if S2 was trained on full dev set."

---

## P3 — Low Impact

### [P3-1] SPEC-systems.md S1 still says "k TBD at EXP-002" — already frozen at 5

**Files:** SPEC-systems.md line 12; SPEC_EXP-002.md Frozen Decisions; config.py line 79

EXP-002 and config.py freeze k=5. SPEC-systems.md still shows "k TBD at EXP-002."

**Fix:** Update SPEC-systems.md S1: "top-k retrieval, k=5 (frozen at EXP-002)."

---

### [P3-2] S5b HyDE via D2L adapter — undefined behavior not analyzed in spec

**File:** SPEC_EXP-007.md lines 17-28

Best adapter may be a D2L adapter targeting MLP layers, optimized for short factual answers. HyDE requires generating passage-length hypothetical text whose embedding is close to source document chunks. D2L adapters are not designed for this generation task.

**Impact:** If best adapter is D2L, S5b HyDE retrieval quality may be worse than raw question embedding. This would appear as S5b degradation but is a methodological artifact.

**Fix:** Add analysis commitment in EXP-007: "If best adapter is D2L, measure HyDE Recall@5 vs baseline. If HyDE does not improve retrieval, report as negative result and use S5a as headline S5."

---

### [P3-3] G column in EXP-008 test table: blank vs N/A not specified for S3/S4

**File:** SPEC_EXP-008.md line 10

EXP-008 says G for S1, S2, S5 — correct. But the output CSV spec does not say how to represent missing G for S3/S4. Blank cell vs "N/A" vs absent column are ambiguous.

**Fix:** Add to EXP-008 output spec: "G column: 'N/A' string for S3, S4-doc, S4-cluster; numeric for S1, S2, S5."

---

## Pre-execution checklist by experiment

**Before EXP-002:** P0-2 (judge prompt), P1-3 (G scope decision), P1-4 (sweep contradiction), P3-1 (k TBD)
**Before EXP-003:** P0-1 (S2 leakage strategy), P1-1 (distractor-only fix), P2-4 (distractor count)
**Before EXP-004:** P1-2 (null instruction in no-retrieval prompt)
**Before EXP-006:** P2-3 (wave placement), P2-5 (cascades from P0-1)
**Before EXP-007:** P3-2 (HyDE analysis note)
**Before EXP-008:** P2-1 (threshold documented in EXP-004 report), P3-3 (N/A in output spec)

