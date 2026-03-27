# SPEC: Evaluation Protocol

> Detail spec for metrics, scoring rules, judge config. Parent: `memory_bank/ARCHITECTURE.md`

---

## Primary Metric

`Q_main = 0.7 × S_det + 0.3 × S_asst`

Applied universally to S1-S5.

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
| `null` | Both null → 1.0; one null → 0.0 |

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

**Judge rules:**
- Same prompt for all systems
- No self-judging (judge ≠ evaluated model)
- Manual audit on ~10% of judged answers before final conclusions

---

## Retrieval-Aware Metrics (S1, S5 only)

**Grounding:**
`G = F_β(β=2.5)` on `(doc_id, page_number)` pairs.

- precision = |P ∩ G_ref| / |P|
- recall = |P ∩ G_ref| / |G_ref|
- Both empty → 1.0; one empty → 0.0

**Supplementary:** Recall@k, evidence hit-rate (optional, for analysis).

---

## Systems Metrics (All)

| Metric | Unit | Scope |
|--------|------|-------|
| TTFT | ms (median, p95) | All |
| End-to-end latency | ms (median, p95) | All |
| Peak VRAM (inference) | MB | All |
| Peak VRAM (training/generation) | MB | S2-S4 |
| Offline packaging cost | seconds | All (index build / train time / adapter gen time) |
| Malformed output rate | % | All |
| Storage footprint | MB | Adapter / index size |

---

## Reporting Format

Every result reported as:
1. **Aggregate** Q_main, S_det, S_asst, G (where applicable), latency, VRAM
2. **By answer_type** breakdown: number, boolean, name, names, date, free_text, null

Tables saved as CSV in experiment artifacts + rendered in REPORT.md.

---

## Interpretation Guidelines

| Outcome | Interpretation |
|---------|---------------|
| S5 best on Q_main | Hybrid approach justified; discuss grounding + latency cost |
| S2 beats S3/S4 | Supervised adaptation > hypernetwork on this benchmark; goldset worth the cost |
| S3/S4 beats S2 | Hypernetwork viable without QA supervision; practical win |
| S4 beats S3 | Routing + sharding > monolithic merge; capacity bottleneck confirmed |
| S3 beats S4 | Routing overhead / clustering noise hurts; monolithic sufficient |
| S1 beats all parametric | Parametric injection not justified on small corpus; retrieval sufficient |
| All close | Main result is the cost/quality trade-off analysis, not a winner |

**Mandatory caveats in all conclusions:**
- Bounded to this corpus, goldset, backbone, hardware
- Doc-to-LoRA hypernetwork is pre-trained, not project-trained
- S2's advantage on format is expected, not evidence of corpus internalization
