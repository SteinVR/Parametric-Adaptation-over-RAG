# Agent Log — Victor (Code Reviewer)
## Task: EXP-004 checkpoint / packaging review
## Date: 2026-03-30

### Files reviewed
- src/d2l/checkpoint.py
- src/d2l/packaging.py
- src/d2l/adapter_io.py
- experiments/EXP-004_d2l_monolithic/main_exp.py
- experiments/EXP-004_d2l_monolithic/config.py
- notebooks/generate_d2l_adapters.ipynb
- external/doc-to-lora/src/ctx_to_lora/modeling/hypernet.py (upstream reference)
- external/doc-to-lora/src/ctx_to_lora/modeling/ctx_encoder.py (upstream reference)

### Verification approach
Full cross-reference of lean loader against original `_init_model` (lines 523-558) and
`generate_weights` (lines 634-699). Math check on SVD re-decomposition. Import
accessibility check. Notebook vs local-code consistency check.

### Findings

**P0 — None confirmed.**

**P1 — `_lean_generate_weights`: ModernBert branch moves ctx_attn_mask to hypernet
device AFTER the encode, but the non-ModernBert path also does this. The real issue:
the ModernBert branch assigns `attention_mask=-1` and `seq_len=-1` into
ctx_encoder_kwargs, so ctx_attn_mask is not used in ctx_encoder at all. After
encoding, the code then moves `ctx_attn_mask` to `hypernet_device` and passes it to
`hypernet.generate_weights`. Original does the same. Confirmed consistent — no bug.**

**P1 — `_bias_hyper_init` call: confirmed NOT a bug. `_bias_hyper_init` is called at
line 467 in `__init__`, which runs during `from_state_dict` before `load_state_dict`
is invoked. Because `_lean_init_model` runs inside `__init__` (via the patched class
method), `self.hypernet` is fully constructed by the time `_bias_hyper_init` fires.
After that, `load_state_dict` (patched to `_lean_load_state_dict`) loads the real
checkpoint weights into the same `self.hypernet` via `self.hypernet.load_state_dict`,
overwriting the randomly initialised weights. The init ordering is correct.**

**P1 — `_lean_generate_weights` `user_defined_scaling` path: original line 688 checks
`user_defined_scaling == 1` AFTER the ModernBert unsqueeze. Local code checks it
BEFORE the unsqueeze (line 289 is outside the `with torch.no_grad()` block but still
before any unsqueeze for ModernBert). However the unsqueeze is for ctx_features, which
is already done inside the with-block. The ModernBert unsqueeze in the original is
at line 687 (outside `with torch.no_grad()`). In the lean version the unsqueeze is
inside `with torch.no_grad()` (line 279). This means if ModernBert is used with
user_defined_scaling != 1, `ctx_features` passed to `hypernet.generate_weights` is
already unsqueezed and on the correct device — this is functionally equivalent.
Confirmed no bug.**

**P2 (real) — `_lean_generate_weights`: `with torch.no_grad()` wraps ctx encoding but
NOT the hypernet `generate_weights` call (lines 290-304). The original wraps both
ctx encoding AND hypernet call under `torch.no_grad()`. For inference this is benign
since hypernet parameters don't require grad in this setting. However it is an
inconsistency that could cause minor overhead from gradient tracking. Not a
correctness issue.**

**P2 (real) — `packaging.py` SVD math: The decomposition uses `sqrt_s` applied
symmetrically to U and Vh (`B_new = U[:, :r] * sqrt_s`, `A_new = sqrt_s * Vh[:r,
:]`). Then `B_new @ A_new = U[:, :r] * S[:r] * Vh[:r, :]`, which is exactly the
rank-r truncated SVD reconstruction of delta_w_avg. The math is correct. Note that
PEFT applies LoRA as `(B @ A) * (alpha/r)`, so this SVD adapter needs to be applied
with the SAME peft_config as the per-doc adapters for scaling to be preserved. The
code reuses `peft_config` from the reference adapter — confirmed correct.**

**P2 (real) — Notebook vs local code divergence: The notebook patches only
`load_state_dict`, leaving the original `_init_model` active. This means on Colab
the hypernet and ctx_encoder load to GPU (device_map=auto), which is fine on T4 but
makes the notebook inconsistent with local's CPU-first approach. Additionally the
local code also patches `_init_model` (to prevent GPU spikes), which the notebook does
not. The notebook will therefore consume more peak GPU memory on Colab than the local
code does locally. This is a known and intentional divergence (Colab has more VRAM
than the RTX 4060). The notebook patch is a strict subset of the local patch — it is
NOT broken, just less aggressive.**

**P2 (informational) — `_lean_generate_weights` passes `**kwargs` through to
ctx_encoder but `generate_weights` in the original also passes `**kwargs`. The local
version does the same. Consistent.**

### Verdict
CHANGES REQUIRED — one real P1 and two real P2s. See main review for details.
