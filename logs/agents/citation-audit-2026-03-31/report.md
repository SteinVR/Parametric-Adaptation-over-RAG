# Citation Audit Report
**Task:** Citation correctness audit + suggested additions for Term_Paper.md
**Date:** 2026-03-31
**Agent:** Victor (Code Reviewer)
**Scope:** All 10 references verified against live arXiv pages; Literature.md reviewed for additions

---

## Part 1: Citation Correctness

### Findings

| Reference (as cited) | Status | Issue |
|---|---|---|
| Dettmers et al. (2023) — QLoRA | CORRECT | Authors, title, year, URL all match arXiv 2305.14314 |
| Guha et al. (2023) — LegalBench | MINOR | 6th author listed as "Narayanan, A." but arXiv shows "Narayana, Aditya" (not "Narayanan") — one-letter typo; rest of truncated author list correct |
| Han et al. (2024) — PEFT Survey | CORRECT | Authors (Zeyu Han, Chao Gao, Jinyang Liu, Jeff Zhang, Sai Qian Zhang), title, year match. Note: 4th author listed as "Zhang, J." which is ambiguous (Jeff Zhang), but not incorrect |
| Hu et al. (2022) — LoRA | CORRECT | arXiv posted 2021-06-17, but ICLR 2022 publication year is used — correct for a conference citation |
| Lee et al. (2024) — Merging LoRAs | **WRONG** | **Author list is entirely fabricated.** Paper cites "Lee, M., Lee, S., & Song, M." — actual authors are Akshara Prabhakar, Yuanzhi Li, Karthik Narasimhan, Sham Kakade, Eran Malach, Samy Jelassi. Actual title is "LoRA Soups: Merging LoRAs for Practical Skill Composition Tasks" (missing "LoRA Soups:" prefix). Should be cited as Prabhakar et al. (2024). |
| Lewis et al. (2020) — RAG | CORRECT | All 12 authors match. Title, year, URL correct. |
| Pipitone et al. (2024) — LegalBench-RAG | **WRONG** | **Author list is wrong.** Paper cites "Pipitone, A., Tikhonova, M., & Ferraro, A." — actual authors are **Nicholas Pipitone** and **Ghita Houir Alami** (2 authors only). No Tikhonova or Ferraro. First name also wrong: "A." vs "Nicholas". Should be cited as Pipitone & Alami (2024). |
| Seshadri (2025) — LLM-as-Judge | **WRONG** | **Wrong first author and wrong attribution.** Paper cites "Seshadri, M." as sole/first author; actual first author is **Anu Pradhan**. Full author list: Pradhan, Ortan, Verma, Seshadri. Seshadri is the last of 4 authors. In-text citation "Seshadri et al., 2025" is doubly wrong: wrong lead author, and cites as single author when there are 4. Should be "Pradhan et al., 2025". Full title: "LLM-as-a-Judge: Rapid Evaluation of Legal Document Recommendation for Retrieval-Augmented Generation". |
| Sukhbaatar et al. (2026) — Doc-to-LoRA | **WRONG** | **Author list is entirely wrong.** Paper cites "Sukhbaatar, S., Grave, E., Bojanowski, P., & Joulin, A." — actual authors are **Rujikorn Charakorn, Edoardo Cetin, Shinnosuke Uesaka, Robert Tjarko Lange**. Sukhbaatar does not appear. Should be cited as Charakorn et al. (2026). |
| Zhang et al. (2024) — RAFT | CORRECT | Authors (Tianjun Zhang, Shishir G. Patil, Naman Jain, Sheng Shen, Matei Zaharia, Ion Stoica, Joseph E. Gonzalez), title, URL match. |

### Summary of errors

- **3 critical errors** (completely wrong author lists): Lee et al., Pipitone et al., Sukhbaatar et al., Seshadri
- **1 minor error**: Guha et al. ("Narayanan" vs "Narayana")

---

## Part 2: Suggested Additions from Literature.md

The paper currently has 10 references. Below are suggested additions ranked by value.

### Rank 1 — LRAGE (arXiv 2504.01840)
**Section:** 2.4 (Research Gap and Positioning) or 3.3 (Fixed Retrieval Backbone)
**Sentence:** After mentioning LegalBench-RAG, add: "LRAGE (2025) extends this with a holistic factorial evaluation that isolates the contribution of each RAG component — corpus, retriever, reranker, generator, metrics — providing a methodological framework that informs our own design choice of holding the retrieval stack constant across all systems."
**Rationale:** Directly addresses this paper's core methodological claim (fixing retrieval to isolate generator training signal). Factually strengthens the design justification rather than padding context.

### Rank 2 — LoRA-LEGO (arXiv 2409.16167)
**Section:** 6.1 (Answer to RQ1), alongside the existing Lee et al. citation on adapter merging
**Sentence:** Add after the Lee et al. reference: "More granular rank-wise clustering approaches (LoRA-LEGO; 2024) suggest that structured module composition can further improve over simple linear interpolation — a direction the S7 result motivates for future work."
**Rationale:** S7 is the paper's most surprising result. Citing a more advanced merging scheme (rank-wise clustering) directly contextualizes the gap between the paper's simple 0.5×0.5 merge and what is possible. Adds specificity to the future-work direction already mentioned.

### Rank 3 — Legal RAG Bench (deeplearn.org/arxiv/708587)
**Section:** 2.4 (Research Gap and Positioning)
**Sentence:** After mentioning LegalBench-RAG, add it alongside LRAGE as a third legal RAG benchmark reference.
**Rationale:** The paper currently has two legal QA/RAG benchmarks in §2.4. A third corroborates the claim that "benchmarks such as X and Y have evaluated LLM capabilities for legal reasoning." Keeps the review of the positioning landscape complete without inflating the section.

### Rank 4 — S-LoRA (arXiv 2311.03285)
**Section:** 6.4 (Limitations), under "Single backbone"
**Sentence:** "Consumer-hardware single-GPU deployment is also contrasted with emerging serving infrastructure for LoRA adapters (S-LoRA; 2023) designed to serve thousands of concurrent adapters on shared GPU resources — a very different operational regime from this study."
**Rationale:** The paper's §6.4 could use one concrete external reference to ground the "serving" limitation. S-LoRA is in Literature.md and directly relevant to the deployment-scope caveat. One sentence, one reference, real value.

### NOT recommended (would be padding)

- LoRA variants (LoRA+, LoRA-FA, LoTR): the paper uses a standard QLoRA configuration and does not investigate rank or LR scheduling. No citation point exists in the text.
- LoftQ / IR-QLoRA / quantization improvements: the paper does not discuss quantization quality degradation as a finding; citing QLoRA improvement work has no anchor in the text.
- Orthogonality in adapter merging (arXiv 2510.03262): interesting theoretically, but S7 is already flagged as exploratory and post-hoc — adding a 2025 critique of orthogonality assumptions would require more discussion than the paper provides for S7.

---

## Summary

| Metric | Count |
|---|---|
| Current references | 10 |
| Critical author errors | 4 (Lee, Pipitone, Seshadri, Sukhbaatar) |
| Minor author errors | 1 (Guha — "Narayanan" spelling) |
| Recommended additions | 2 (essential: LRAGE, LoRA-LEGO); 2 optional (Legal RAG Bench, S-LoRA) |
| Recommended final count | 12–14 |

The four critical errors must be fixed before submission; they are not marginal — three of them involve author lists with no overlap with the actual authors of the cited papers. The Seshadri error additionally produces a wrong in-text citation key.
