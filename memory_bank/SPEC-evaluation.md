# SPEC: Evaluation Protocol

> Detail spec for metrics, scoring rules, judge config. Parent: `memory_bank/ARCHITECTURE.md`

---

## Primary Metric

`Q_main = 0.7 × S_det + 0.3 × S_asst`

Applied to active and reported systems: S1, S2+R, S3+R, S7, S2, S3, S3-legacy (D2L).

---

## Deterministic Scoring (S_det)

Per-question score ∈ {0, 1} (or continuous for set-based types):

| answer_type | Scoring rule |
|-------------|-------------|
| `number` | Exact numeric match, ±1% tolerance |
| `boolean` | Exact match (true/false) |
| `name` | Normalized exact string match (lowercased, stripped) |
| `names` | Jaccard similarity over normalized string sets |
| `date` | Exact ISO 8601 match (YYYY-MM-DD) |
| unanswerable | Gold answer is `null` → expected system response is `[]` (empty). Both `[]` → 1.0; one `[]` one non-empty → 0.0 |

`S_det` = mean of per-question deterministic scores across all deterministic questions.

---

## Free-Text Scoring (S_asst)

**Judge model:** gpt-5.4-mini (OpenAI API), medium reasoning, version-pinned at experiment start.

**Rubric:** 5 binary criteria per question:

1. **Correctness** — key information present, no factual errors
2. **Completeness** — all aspects addressed, key points covered
3. **Grounding** — statements supported by context, no unsupported claims
4. **Confidence calibration** — uncertainty expressed when appropriate
5. **Clarity & relevance** — clear, concise, on-topic

Each criterion → {0, 1}. Per-question S_asst = mean of 5 criteria.
Submission S_asst = mean across all free_text questions.

**Judge prompt (frozen):**

```
System: You are an impartial judge evaluating a legal QA system's response.
Score each criterion as 1 (met) or 0 (not met). Return ONLY a JSON object.

User:
Question: {question}
Reference answer: {reference_answer}
System response: {system_response}

Criteria:
1. correctness: Does the response contain the key information from the reference and no factual errors?
2. completeness: Does the response address all aspects of the question?
3. grounding: Is every claim supported by plausible legal reasoning (no hallucinated specifics)?
4. calibration: Does the response appropriately express uncertainty when information is missing?
5. clarity: Is the answer clear, concise, and directly addresses the question?

Return JSON: {"correctness": 0|1, "completeness": 0|1, "grounding": 0|1, "calibration": 0|1, "clarity": 0|1}
```

**Judge rules:**
- Same prompt for all systems
- No self-judging (judge ≠ evaluated model)
- Parse JSON response; if malformed, retry once, then score 0 for all criteria
- Manual audit on ~10% of judged answers before final conclusions

---

## Retrieval-Aware Metrics

**Systems with retrieval:** S1, S2+R, S3+R, S7.

Controls S2, S3, S3-legacy (D2L): `G = N/A` (no retrieval).

**Grounding:**
`G = F_β(β=2.5)` on `(doc_id, page_number)` pairs.

**Building predicted set P:** deduplicated union of all `(doc_id, page_number)` from **final evidence chunks** — i.e. after the full retrieval pipeline completes (search → rerank → evidence compression → page lifting).

- precision = |P ∩ G_ref| / |P|
- recall = |P ∩ G_ref| / |G_ref|
- Both empty → 1.0; one empty → 0.0

**Supplementary:** Recall@k on raw pre-compression candidates (optional, for retrieval stack analysis).

---

## Systems Metrics (All)

| Metric | Unit | Scope |
|--------|------|-------|
| TTFT | ms (median, p95) | All |
| End-to-end latency | ms (median, p95) | All |
| Peak VRAM (inference) | MB | All |
| Peak VRAM (training) | MB | S2, S3 |
| Offline packaging cost | seconds | All (index build / training time / merge time if applicable) |
| Malformed output rate | % | All (parser fails to extract valid answer → `_malformed_` marker, scored as 0) |
| Storage footprint | MB | Adapter / index size |

---

## Reporting Format

Every result reported as:
1. **Aggregate** Q_main, S_det, S_asst, G (where applicable), latency, VRAM
2. **By answer_type** breakdown: number, boolean, name, names, date, free_text (6 types). Unanswerable is a cross-cutting flag (`is_unanswerable`), not a separate answer_type — report separately.

Tables saved as CSV in experiment artifacts + rendered in REPORT.md.

---

## Interpretation Guidelines

| Outcome | Interpretation |
|---------|---------------|
| S2+R or S3+R best on Q_main | Parametric adaptation adds value on top of RAG (H1 confirmed) |
| S2+R beats S3+R | Supervised RAFT adapter > supervision-free CLM adapter on this benchmark; goldset worth the cost |
| S3+R beats S2+R | Supervision-free CLM adapter viable without QA supervision; practical win |
| S7 beats both S2+R and S3+R | Post-hoc adapter interpolation captures complementary strengths without retraining |
| S1 beats all augmented systems | Retrieval engineering dominates over adaptation on this corpus; valid finding |
| All close | Main result is trade-off analysis, not a winner |

**Archived note:** S6 is not part of the active evaluation protocol and should not appear in default thesis tables unless explicitly requested.

**Mandatory caveats in all conclusions:**
- Bounded to this corpus, goldset, backbone, hardware
- CLM continued pretraining uses only corpus text, no QA supervision
- S7 is post-hoc merge (eval-only), not a retrained pipeline
- D2L appears only as legacy negative control (`S3-legacy`)
