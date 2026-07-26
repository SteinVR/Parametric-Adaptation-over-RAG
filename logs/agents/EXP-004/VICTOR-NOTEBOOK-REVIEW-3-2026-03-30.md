# Victor — Notebook Review Pass 3 (PDF-handling follow-up)
**Task:** EXP-004 Colab notebook review — PDF fix verification
**Date:** 2026-03-30
**Reviewer:** Victor (Code Reviewer)
**File reviewed:** notebooks/generate_d2l_adapters.ipynb
**Cross-reference:** src/d2l/corpus.py, src/d2l/adapter_io.py, src/d2l/packaging.py

---

## Scope

Follow-up review after the PDF fix (corpus files are PDFs, not .txt). Verify:
1. All previous fixes still intact
2. PDF handling correct and matching local pipeline
3. Upload cell filters .pdf correctly
4. Sort order / adapter numbering matches local pipeline
5. No new issues introduced

---

## Checklist results

### 1. Previous fixes — all confirmed intact

| Fix | Location | Status |
|-----|----------|--------|
| HF login before any download | Cell 1 `login()` | Present |
| `--no-deps` | Cell 2 pip install | Present |
| `snapshot_download` Gemma pre-cache | Cell 3 | Present |
| Monkey-patch + try/finally restore | Cell 5 | Present |
| Zip recursive extraction + tmp cleanup | Cell 4 | Present |
| `gc.collect()` + `torch.cuda.empty_cache()` per doc | Cell 6 | Present |
| `assert len(corpus_files) > 0` | Cell 4 and Cell 6 | Present |

### 2. PDF handling

- `pymupdf` in deps list: present (cell-2).
- `fitz.open` with try/finally `doc.close()`: present (cell-6, `extract_pdf_text`).
- `page.get_text("text").strip()`: matches `corpus.py:extract_pdf_page_texts` line 67.
- Join logic: notebook uses `"\n\n".join(p for p in pages if p).strip()` where pages
  are already `.strip()`'d at collection time. Local `render_document_text` (corpus.py:76)
  uses `"\n\n".join(text.strip() for text in page_texts if text.strip()).strip()`.
  Both are semantically identical — double-stripping an already-stripped string is a no-op.

### 3. Upload filtering

- Zip path: `glob.glob("_tmp_corpus/**/*.pdf", recursive=True)` — correct.
- Direct upload path: `elif fname.endswith(".pdf")` — correct.
- Final list: `os.listdir("corpus")` filtered to `.pdf` — correct.

### 4. Sort order and adapter numbering

- Notebook cell-6: `sorted(corpus_dir.glob("*.pdf"))` using `pathlib.Path.glob`.
- Local corpus.py line 39: `sorted(corpus_dir.glob("*.pdf"))` using `pathlib.Path.glob`.
- Both use Path lexicographic sort on the same filenames — order is identical.
- Both use `enumerate(..., start=1)` — `doc{idx}` mapping is identical.
- Local `main_exp.py` uses `document.doc_index` which is set by `load_frozen_corpus_documents`
  with the same `enumerate(pdf_paths, start=1)` — mapping is stable end-to-end.

### 5. New issues found

#### P1-A: `model.peft_config["default"]` — potential TypeError

Cell-6 line: `peft_config = model.peft_config["default"]`

Local `packaging.py` line 130 reads: `peft_config = model.peft_config` (no key access).

In PEFT, `model.peft_config` on a `PeftModel` is a dict keyed by adapter name (e.g.
`"default"`). However, `ModulatedPretrainedModel` wraps the base model differently —
the local pipeline reads it without subscript, implying it may already return a plain
`LoraConfig` at that access point. If `model.peft_config` returns a plain `LoraConfig`,
`["default"]` will raise `TypeError: 'LoraConfig' object is not subscriptable` and the
entire adapter generation loop will crash before any adapter is saved.

Fix: mirror local code — use `peft_config = model.peft_config` and guard with
`if isinstance(peft_config, dict): peft_config = list(peft_config.values())[0]`.

#### P1-B: Missing `_normalize_state_dict_for_peft_save` — silent wrong lora_B orientation

Cell-6 saves the adapter state dict directly via `save_safetensors_file({k: v.detach().cpu().contiguous() ...})`.

The local `save_peft_lora_adapter` (adapter_io.py lines 46-50) first passes the dict
through `_normalize_state_dict_for_peft_save`, which detects D2L's internal `(r, d_out)`
lora_B orientation and transposes to PEFT's expected `(d_out, r)` convention
(adapter_io.py lines 88-98).

The notebook skips this step. If the D2L checkpoint's `generated_lora_to_state_dict`
emits lora_B in `(r, d_out)` orientation (the local code already accounts for this),
adapters downloaded from Colab will load into PEFT with transposed lora_B matrices.
PEFT will not raise an error — it will silently load and run with wrong matrix shapes,
producing garbage inference output from every adapter.

Fix: inline the normalization before saving, or replace the manual save block with the
equivalent of `save_peft_lora_adapter` logic (deepcopy config, set
`base_model_name_or_path` and `inference_mode=True`, apply normalization, then
`save_safetensors_file`).

---

## Gate Decision

**CHANGES REQUIRED**

| # | Severity | File | Finding |
|---|----------|------|---------|
| 1 | P1 | notebooks/generate_d2l_adapters.ipynb cell-6 | `model.peft_config["default"]` — TypeError if peft_config is a plain LoraConfig |
| 2 | P1 | notebooks/generate_d2l_adapters.ipynb cell-6 | No lora_B orientation normalization — silently wrong adapters if D2L uses (r, d_out) |

PDF extraction, sort order, adapter numbering, and all previous fixes: PASS.
Both P1 findings require fixes before the notebook can be safely used in Colab.
