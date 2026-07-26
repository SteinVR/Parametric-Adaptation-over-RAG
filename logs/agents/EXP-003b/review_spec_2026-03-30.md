# Agent Log — Victor (Code Reviewer)
**Task:** Spec review of SPEC_EXP-003b.md
**Date:** 2026-03-30
**Status:** CHANGES REQUIRED

---

## Files Reviewed

- `memory_bank/specs/SPEC_EXP-003b.md` (subject)
- `memory_bank/specs/SPEC_EXP-003.md` (peer comparison)
- `memory_bank/ARCHITECTURE.md` v7.0
- `memory_bank/SPEC-systems.md`
- `memory_bank/SPEC-evaluation.md`

---

## Verification Verdict: CHANGES REQUIRED

Two P1 issues require fixes before implementation. No P0 blockers. Spec is otherwise well-structured and internally consistent.

---

## Findings (Ranked by Severity)

### P1-A — Batch size mismatch with ARCHITECTURE.md frozen decision

**File:** `SPEC_EXP-003b.md` line 64 vs `SPEC_EXP-003.md` line 69

EXP-003b states: `Batch size | 1 (gradient accumulation 4 → effective 4)`
EXP-003 states: `Batch size | 4 ( to effective 8 if needed)`
ARCHITECTURE.md §4 Frozen Decisions does not pin batch size — it pins QLoRA 4-bit NF4 only. So this is not a frozen-decision violation in the strictest sense. However, the two specs are **inconsistent with each other** when they claim to use "identical" hyperparams.

EXP-003b line 69 says: *"Hyperparams intentionally identical to S2+R for fair comparison."* This statement is false: batch size 1/effective-4 (EXP-003b) is not the same as batch size 4/effective-up-to-8 (EXP-003). The discrepancy is likely intentional (closed-book prompts are shorter → fit in larger batches; or VRAM constraints differ because there is no retrieved context in the training input). Either way, the spec must reconcile this: either acknowledge the difference and justify it, or correct the claim of identity.

**Impact:** Implementer will be confused about which value to use. The claim of identical hyperparams for fair comparison is weakened if effective batch size differs.

**Recommended fix:** Either (a) set EXP-003b batch size = 4 / effective 8 to match EXP-003 and remove the discrepancy, or (b) update the prose to say "Learning rate, optimizer, scheduler, rank, alpha, dropout, epochs are identical; batch size differs (1/eff-4 vs 4/eff-8) because closed-book inputs are shorter — learning dynamics remain comparable." A brief justification is sufficient.

---

### P1-B — Output section missing per-seed `predictions.json` path

**File:** `SPEC_EXP-003b.md` §Output (lines 93–97)

The Output section lists:
- closed_book_train.jsonl
- 3 adapter checkpoints
- Per-seed eval results: `results/EXP-003b/seed_{42,123,777}/`
- aggregate_summary.json
- REPORT.md

The DoD (line 103) requires "each `predictions.json` has 50 entries." The Output section does not specify that `predictions.json` lives inside each `results/EXP-003b/seed_{42,123,777}/` directory. EXP-003's Output section also omits this (it only lists the RAFT dataset, adapter checkpoints, and REPORT.md — no per-seed results path at all), but EXP-003b partially fixed this by listing the seed result directories. The fix is to be explicit: `results/EXP-003b/seed_{42,123,777}/predictions.json`.

**Impact:** Minor ambiguity for the implementer — low risk but the DoD references the file without the Output section anchoring its location.

**Recommended fix:** Change the Output entry to: `results/EXP-003b/seed_{42,123,777}/predictions.json (50 entries each)`.

---

### P2-A — `answer_type_instruction` placeholder undefined in prompt template

**File:** `SPEC_EXP-003b.md` lines 31–36

The closed-book prompt template references `{answer_type_instruction}` but does not define where these instructions come from beyond "the same as `src/generation/prompt.py::ANSWER_TYPE_INSTRUCTIONS`." Since this is the first spec to define a closed-book prompt, the reference is somewhat implicit. EXP-003 (RAFT) inherits the prompt from EXP-002's existing RAG prompt; EXP-003b has no such predecessor.

**Impact:** Negligible in practice because `src/generation/prompt.py` exists and the implementer can read it. But the spec is not self-contained on this point.

**Recommended fix:** Either inline the answer_type_instruction values in the spec (6 lines), or add a footnote like "Values from `ANSWER_TYPE_INSTRUCTIONS` in `src/generation/prompt.py` — boolean: 'true or false', number: 'a number', etc."

---

### P2-B — Delta computation requires EXP-003 results but dependency is only informally noted

**File:** `SPEC_EXP-003b.md` line 3 (header), line 9, lines 88–89

The spec correctly notes EXP-003 as a dependency and lists "Delta vs S2+R (EXP-003)" as a metric. The DoD (line 107) also includes it. This is logically sound. However the dependency header says "EXP-003 (shared QLoRA config, delta computation)" — the dependency is primarily on EXP-003's *results being available*, not just its config. If EXP-003 and EXP-003b run in parallel (both Wave 2), delta computation cannot happen until both finish.

**Impact:** Wave 2 parallelism is valid for training; delta reporting must be deferred. The spec does not warn the implementer that the delta is a post-hoc reporting step, not part of the EXP-003b run itself.

**Recommended fix:** Add one sentence to the Metrics section: "Delta vs S2+R is computed after both EXP-003 and EXP-003b complete; it is a reporting step, not a blocker for EXP-003b eval."

---

## Deliberate Differences vs EXP-003 — Justified or Not

| Difference | Justified |
|-----------|-----------|
| No gold chunk / distractor section | Yes — closed-book by design |
| No Qdrant / reranker at inference | Yes — closed-book by design |
| G metric absent | Yes — correct, no retrieval |
| Prompt template different (no context field) | Yes — core design point |
| Output path `models/qlora_closed/` vs `models/qlora/` | Yes — avoids collision |
| Batch size 1/eff-4 vs 4/eff-8 | NOT JUSTIFIED in spec (see P1-A) |
| EXP-003 Output section omits per-seed path; EXP-003b adds it | EXP-003b is better here; EXP-003 should be backfilled similarly |

---

## Items Confirmed Correct

- G metric correctly excluded (no retrieval at inference — consistent with ARCHITECTURE.md §5, SPEC-evaluation.md line 78, SPEC-systems.md S2 definition).
- 3-seed protocol (42, 123, 777) matches ARCHITECTURE.md frozen decision and SPEC-systems.md S2.
- Split references `data/splits/split_v1.json` — correct.
- DoD covers: dataset, adapter weights, 50-question eval, judge scoring, mean±std, both deltas, VRAM/time, answer_type breakdown, git commit, REPORT.md. All deliverables mentioned in the body are covered.
- Wave 2 placement consistent with ARCHITECTURE.md §8 experiment phases table.
- Blocks EXP-006 and EXP-007 — correct per dependency chain (EXP-006 needs all Axis 1 results; EXP-007 needs best Axis 1 adapter).
- Closed-book prompt has no "context" or "retrieved" language — correct.
- Adapter config (rank 32, alpha 32, dropout 0.05, q_proj+v_proj, lr 2e-4, paged AdamW 8-bit, cosine, 3 epochs) matches SPEC-systems.md S2 definition exactly.
- Answer formatting rules are consistent with parser expectations.
- No dev/test contamination risk: 150-train / 50-eval split is frozen and not touched.

---

## Gate Decision: CHANGES REQUIRED

Fix P1-A (batch size reconciliation) and P1-B (explicit predictions.json path in Output) before implementation begins. P2 items are low-risk and can be resolved alongside the P1 fixes.
