# Victor Code Review — EXP-004 Notebook Final Pass
**Date:** 2026-03-30
**Reviewer:** Victor (apm-code-reviewer)
**Scope:** `notebooks/generate_d2l_adapters.ipynb` — final verification pass
**Cross-reference:** `src/d2l/corpus.py`, `src/d2l/adapter_io.py`, `src/d2l/packaging.py`, `src/d2l/checkpoint.py`

---

## Verdict: PASS

All previously-required fixes are confirmed present and correct. No P0 or P1 bugs found.

---

## All-Fixes Confirmation

| Fix | Location | Status |
|---|---|---|
| HF login | cell-1 | PRESENT |
| `--no-deps` + `transformers==4.51.3` version pin | cell-2 | PRESENT |
| `snapshot_download` pre-caches Gemma | cell-3 | PRESENT |
| Monkey-patch with `try/finally` restore | cell-5 | PRESENT |
| T4-only / 8 GB warning comment | cell-5 | PRESENT |
| zip + individual PDF upload handling | cell-4 | PRESENT |
| `gc.collect()` + `torch.cuda.empty_cache()` after model load | cell-5 | PRESENT |
| `assert len(corpus_files) > 0` (cell-4 and cell-6) | both cells | PRESENT |
| zip root produces `doc1/doc2/...` directly (not nested) | cell-7 | PRESENT |
| PDF extraction via PyMuPDF `fitz.open` with `try/finally` | cell-6 | PRESENT |
| `peft_config` dict/object unwrap guard | cell-6 | PRESENT |
| `normalize_lora_b_orientation` defined and called before save | cell-6 | PRESENT |

---

## Cross-Reference Analysis

### Text extraction — notebook vs `src/d2l/corpus.py`

Notebook `extract_pdf_text` (cell-6): `fitz.open`, `page.get_text("text").strip()` per page, `"\n\n".join(p for p in pages if p).strip()`.

`corpus.py` `extract_pdf_page_texts` + `render_document_text`: same method, `"\n\n".join(text.strip() for text in page_texts if text.strip()).strip()`.

Since notebook strips at extraction time and filters on truthiness, and `corpus.py` strips again before joining (no-op on already-stripped strings) — these are **functionally identical**. PASS.

### lora_B normalization — notebook vs `src/d2l/adapter_io.py`

Notebook `normalize_lora_b_orientation` and `adapter_io.py` `_normalize_state_dict_for_peft_save` apply identical logic:
- iterate `.lora_B.` keys
- skip if ndim != 2
- find paired `lora_A.weight`
- read rank from `lora_A.shape[0]`
- transpose if `value.shape[0] == rank`

**Identical.** PASS.

### peft_config access — notebook vs `src/d2l/packaging.py`

Notebook guards with `if isinstance(peft_config, dict)` before accessing `.target_modules`. `packaging.py` accesses `.target_modules` directly. The notebook is *more* defensive; no regression risk. PASS.

### Sort / enumerate order

Notebook cell-6: `sorted(corpus_dir.glob("*.pdf"))` with `enumerate(..., start=1)`.
`corpus.py`: `sorted(corpus_dir.glob("*.pdf"))` with `enumerate(..., start=1)`.
Identical. `doc{idx}` mapping will match local pipeline adapter directories. PASS.

### Monkey-patch scope divergence (intentional)

`checkpoint.py` patches both `load_state_dict` and `_init_model`. Notebook only patches `load_state_dict`. This is intentional: T4 has 16 GB VRAM, so the full `_init_model` GPU load is acceptable; only the *redundant second* call needs to be blocked. The comment in cell-5 documents this explicitly. PASS.

---

## Remaining Observations (non-blocking)

### P2 — `total_bytes` lacks `if path.is_file()` guard

**Cell-6:**
```python
total_bytes = sum(f.stat().st_size for f in adapter_dir.iterdir())
```
**Canonical (`adapter_io.py` line 53):**
```python
byte_size = sum(path.stat().st_size for path in adapter_dir.iterdir() if path.is_file())
```
Impact: cosmetic — affects only the print statement. If a future PEFT version creates a subdirectory inside the adapter dir, size is slightly inflated. Adapter artifacts are unaffected.

### P3 — No `del state_dict` after `from_state_dict`

`checkpoint.py` line 93 explicitly `del state_dict` after loading to release the reference. The notebook keeps the variable alive until cell-5 scope ends. Since `_lean_load_state_dict` pops all keys during load, the dict is empty by then — actual retained memory is negligible.

### P3 — Per-document loop has no `try/finally` for `model.reset()`

Production `packaging.py` wraps per-document internalization in `try/finally: model.reset()`. Notebook calls `model.reset()` at the bottom of the loop body. If a non-fatal exception occurs mid-loop, `model.reset()` is skipped. In Colab, OOM restarts the kernel anyway; the only risk is a non-OOM exception leaving the model in a dirty state for subsequent documents.

---

## Gate Decision

**APPROVE.** All previously-flagged P1 issues are resolved. No new P0/P1 issues found. Three remaining observations (P2/P3) are cosmetic or defensive-coding items that do not affect adapter correctness, the generated safetensors files, or the Colab execution path for the 8-document corpus.
