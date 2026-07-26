# Victor — Notebook Review Log
**Task:** Review `notebooks/generate_d2l_adapters.ipynb` for Colab T4 failures
**Date:** 2026-03-30
**Reviewer:** Victor (apm-code-reviewer)
**Status:** CHANGES REQUIRED

---

## Files read
- `/home/xeliaray/Projects/Term-Paper/notebooks/generate_d2l_adapters.ipynb`
- `/home/xeliaray/Projects/Term-Paper/src/d2l/checkpoint.py`
- `/home/xeliaray/Projects/Term-Paper/src/d2l/adapter_io.py`
- `/home/xeliaray/Projects/Term-Paper/src/d2l/packaging.py`
- `/home/xeliaray/Projects/Term-Paper/external/doc-to-lora/pyproject.toml`
- `/home/xeliaray/Projects/Term-Paper/external/doc-to-lora/src/ctx_to_lora/modeling/hypernet.py` (lines 460-560, 600-632)
- `/home/xeliaray/Projects/Term-Paper/external/doc-to-lora/src/ctx_to_lora/utils.py` (lines 200-260)

---

## Findings

### P0 — Will crash

**P0-A: Double `_init_model()` — guaranteed OOM on Colab T4**
- `from_state_dict` at line 497 calls `cls(...)` which hits `__init__` → `_init_model()`.
  Then at line 498 it calls `model.load_state_dict(state_dict)` which calls `_init_model()` again (hypernet.py line 618).
- Each `_init_model()` allocates: a new `HyperLoRA` (perceiver + heads) on GPU + a full second Gemma-2-2b `ctx_encoder` loaded via `get_model` with `device_map=auto`.
- The ctx_encoder loads in full precision (no quantization in `ctx_model_kwargs` — line 543-547 of `_init_model` provides no `quantization_config`). That is ~5 GB fp32.
- First call: ~5 GB ctx_encoder + hypernet on 16 GB T4, fine. Second call allocates another one before the first can be garbage-collected. Peak is ~10+ GB ctx_encoders simultaneously plus the 4-bit base model (~1.5 GB). Will OOM before `load_state_dict` returns.
- The monkey-patch in `src/d2l/checkpoint.py` (`_lean_load_state_dict`) is documented and working — it is simply absent from the notebook.
- **Fix:** Apply the same monkey-patch before calling `from_state_dict`, identical to `checkpoint.py` lines 79-91.

**P0-B: `_init_model` hardcodes `local_files_only=True` for ctx_encoder — will crash on Colab**
- `_init_model` builds `ctx_model_kwargs` with `"local_files_only": True` (hypernet.py line 545).
- The ctx_encoder model name defaults to `self.base_model.config.name_or_path` = `google/gemma-2-2b-it`.
- On a fresh Colab instance that model is not cached. `local_files_only=True` will raise `OSError: Couldn't find a model file...` before anything runs.
- The base model loads fine because the notebook passes `local_files_only=False` through `base_model_kwargs`, but `_init_model` constructs its own kwargs dict for the ctx_encoder and ignores whatever was passed in.
- Workaround is to pre-download (or pre-cache) Gemma-2-2b-it before calling `from_state_dict`, so the cache exists and `local_files_only=True` succeeds. The notebook currently has no such step.
- **Fix:** Add a cell that calls `snapshot_download("google/gemma-2-2b-it", ...)` or does a dummy `AutoModel.from_pretrained("google/gemma-2-2b-it", ...)` load before D2L instantiation, so the local cache is populated. (The monkey-patch from P0-A also sidesteps the second `_init_model` call, but the first call in `__init__` still hits this.)

**P0-C: `get_lora_module_names` return shape is incompatible with how the notebook indexes it**
- The library's `get_lora_module_names` (utils.py lines 229-247) returns:
  `dict[target_module -> list[list[str]]]` — indexed as `module_names[target_module][layer_idx]`.
  The inner list has length `len(layer_indices)` but is indexed by positional rank, NOT by raw layer index.
  That is: `module_names["q_proj"][0]` = names for the first element of `layer_indices`, not for layer 0.
- However `generated_lora_to_state_dict` (utils.py lines 206-226) also iterates `layer_indices` as a sorted list and accesses `module_names[target_module][layer_idx]` using the raw index value. If `layer_indices = [10, 14, 18, ...]` then `module_names[...][10]` will be an out-of-range list access (the list has length = count of layer indices, not 28).
- The local code in `packaging.py` calls the same function with the same indexing — this is a pre-existing upstream bug in D2L that the notebook inherits. Depending on actual `layer_indices` values it may throw `IndexError` or silently return empty lists causing missing tensors.
- **Fix:** Verify at runtime that adapters contain the expected number of tensors; if the upstream indexing is indeed broken the workaround used in the project's own code must be confirmed.

---

### P1 — Likely problem

**P1-A: `pip install git+https://github.com/SakanaAI/doc-to-lora.git` pulls deepspeed and vllm**
- `pyproject.toml` lists `deepspeed==0.17.1` and `vllm==0.8.5.post1` as hard dependencies.
- On Colab, `vllm` requires compilation of CUDA extensions and takes 5-10 minutes; `deepspeed` also compiles ops. Both may fail on the free T4 image depending on CUDA/GCC versions.
- The local project deliberately excludes D2L from `pyproject.toml` with `uv pip install --no-deps -e ./external/doc-to-lora` precisely to avoid this.
- The notebook uses a plain `pip install git+...` with no `--no-deps`, so it will pull all dependencies including these heavy ones.
- **Fix:** Use `pip install --no-deps git+https://github.com/SakanaAI/doc-to-lora.git` and install only the runtime-required subset manually: `transformers`, `peft`, `accelerate`, `bitsandbytes`, `safetensors`, `einops`, `opt-einsum`.

**P1-B: Gemma-2-2b-it requires accepting Google's license on HuggingFace**
- `google/gemma-2-2b-it` is a gated model. Downloading it requires an HF account with the license accepted and an HF token.
- The notebook has no `huggingface-cli login` step and no `HF_TOKEN` env variable setup.
- Without a valid token the download will fail with a 401/403.
- **Fix:** Add a cell with `from huggingface_hub import login; login(token="...")` or set `os.environ["HF_TOKEN"] = "..."` before checkpoint loading. The D2L checkpoint download itself (`SakanaAI/doc-to-lora`) is likely public, but the base model is not.

**P1-C: `corpus.zip` extraction may produce incorrect paths**
- Cell 3 extracts to `"corpus"` but the zip may contain a top-level directory. For example if the user runs `zip -r corpus.zip data/corpus/` the extracted structure will be `corpus/data/corpus/*.txt`, not `corpus/*.txt`.
- The `os.listdir("corpus")` filter for `.txt` will find nothing, and the notebook will silently proceed with 0 documents.
- **Fix:** Print the full tree after extraction so the user can see the actual layout. Or use `glob("corpus/**/*.txt", recursive=True)` to tolerate nested paths.

**P1-D: Non-zip uploads are silently dropped**
- In cell 3, if the user uploads individual `.txt` files (not a zip), the code saves them via `files.upload()` but the `else` branch does `os.makedirs("corpus", exist_ok=True)` without placing the uploaded file there. `files.upload()` saves files to the current directory (`/content/`), not to `corpus/`. The subsequent `os.listdir("corpus")` will find nothing.
- **Fix:** In the `else` branch, move/copy each uploaded file into `corpus/`. Concretely: `import shutil; shutil.copy(fname, f"corpus/{fname}")`.

---

### P2 — Minor issues

**P2-A: No HF checkpoint auth check — unclear whether `SakanaAI/doc-to-lora` is public**
- Cell 2 runs `huggingface-cli download SakanaAI/doc-to-lora` with no token. If the repo is private or gated this fails silently in the `--include` filter. Worth adding `--token $HF_TOKEN` defensively.

**P2-B: `module_names` is computed once, before the loop, but `model.reset()` may invalidate internal state**
- The notebook calls `model.reset()` after each document and computes `module_names` before the loop (cell 5). `model.reset()` sets `self.generated_loras = None` and possibly patches forward again. The `module_names` dict contains static string keys derived from the PEFT model state dict and should remain valid across resets. Low risk but worth confirming.

**P2-C: `torch.cuda.empty_cache()` without `gc.collect()` before each document**
- `empty_cache()` alone does not release Python-side tensor references. Adding `gc.collect()` before `torch.cuda.empty_cache()` in the loop is the established pattern (already used in `checkpoint.py`). On 16 GB T4 this matters less than on 8 GB, but is still good practice.

**P2-D: No progress guard for 0 documents found**
- If `corpus_files` is empty the loop exits silently with "Done! All 0 adapters saved." A guard `assert len(corpus_files) == 8` (or at minimum a warning) would surface the P1-C/P1-D failures early.

---

## Verdict
**CHANGES REQUIRED**

P0-A (double `_init_model`) and P0-B (`local_files_only=True` for ctx_encoder) will each independently crash the notebook on a fresh Colab T4 before any adapter is produced. P1-A (vllm/deepspeed pull) will make installation take 10+ minutes or fail outright. P1-B (gated Gemma model) will fail the download. These four issues together make the notebook non-runnable as written.
