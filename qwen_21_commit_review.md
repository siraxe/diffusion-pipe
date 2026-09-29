# Review of commit `25cc3ed` — `qwen_21`

**Scope:** adds Qwen-Image 2.1 support (ComfyUI backend) — new pipeline model, dataset plumbing for
reference-image conditioning, CFG ("guidance") training, docs, and example config.

Files changed: `.gitmodules`, `docs/supported_models.md`, `examples/qwen_image_2_1.toml`,
`models/base.py`, `models/qwen_image_2_1.py` (new, 599 lines), `train.py`, `utils/dataset.py`.

---

## Overall verdict

Solid, working integration. Structure follows the established Comfy-pipeline patterns
(`InitialLayer` / `TransformerLayer` / `FinalLayer`, `ModelOffloader`, `make_contiguous`,
serialized-int `meta` tensor for pipeline parallelism). Timestep sampling, target
(`x_0 - x_1`), `requires_grad_` on float inputs, and shift handling all match house style
(`hunyuan_image.py`, `models/base.py`). Below are the items worth a second look, ordered by
importance.

---

## Issues / things to double-check

### 1. Docs contradict the code on guidance (likely copy-paste leftover)
`docs/supported_models.md` ends the Qwen-Image 2.1 section with:

> "The model was trained guidance-free, so use cfg = 1 when sampling."

But the code defaults sampling guidance to **3.0**
(`DEFAULT_GUIDANCE_SCALE = 3.0`, `prepare_sample_test` override, whose docstring explicitly
says cfg=1 "produces washed-out results") and trains with CFG distillation (`cfg=3` default
in `model_config`). One of these is wrong — and which one decides whether the whole
CFG-training machinery (uncond branch through every layer, `FinalLayer` combine) is even
desirable. **Fix the docs line or the default.**

### 2. `comfy.ldm.qwen_image21` is not in the pinned ComfyUI submodule — **RESOLVED 2026-09-21**
The model imports:

```python
from comfy.ldm.qwen_image21.model import _split_rows
import comfy.ldm.qwen_image21.model as _qwen_model
```

but `submodules/ComfyUI` at the recorded commit (`3216c62e`) contains only
`comfy/ldm/qwen_image` — **no `qwen_image21` directory**. Training evidently works on your
runtime, so `comfy` is resolving from a different ComfyUI install on `sys.path` (the
`sys.path.insert(0, ...)` at the top only takes effect if `comfy` wasn't already imported
by something earlier). The repo as committed is therefore **not reproducible** — a fresh
checkout will crash on import. Also note the same commit adds `ignore = all` to the ComfyUI
submodule in `.gitmodules`, which conveniently hides local drift there. Recommend updating
the submodule pointer to a commit that actually has `qwen_image21`, or vendoring the needed
pieces.

> **Resolution (2026-09-21):** this exact failure broke a `--resume_from_checkpoint` run
> (`ModuleNotFoundError: No module named 'comfy.ldm.qwen_image21'`) after the submodule got
> reset to the recorded commit. Fixed by pinning the submodule to `1d61dcc3` ("Compile Qwen
> Image 2.1 transformer blocks", the newest commit before `c194dd00` added the
> `preferred_attention` kwarg to the attention call, which `make_attn_fn` doesn't accept).
> Committed as `21502dc` on `q21`. Note: do **not** jump the submodule to current
> `origin/master` — `c194dd00`+ passes `preferred_attention=` to `attn_fn` and training
> crashes with `TypeError`. If you ever want to move the submodule forward, `make_attn_fn`'s
> inner `attn` must accept (or explicitly reject) `**kwargs`/`preferred_attention`.

### 3. Global monkey-patch of `_gated_residual` at import time
`models/qwen_image_2_1.py` patches `comfy.ldm.qwen_image21.model._gated_residual` at module
import, process-wide. It's needed (the original uses in-place `addcmul_` on views, which
breaks autograd), but:
- it also affects any other consumer of that module in-process;
- `x.clone()` runs on every block call in both sampling and training.

Fine for now; ideally upstream the fix to ComfyUI and drop the patch.

### 4. CFG training: gradient through the uncond branch is dropped (by design, but be aware)
The per-layer uncond forward runs under `torch.no_grad()`, and in `FinalLayer`:

```python
output = (output + (self.cfg - 1) * output_uncond) / self.cfg
```

`output_uncond` is detached, so gradients flow only through the conditional prediction,
scaled by `1/cfg`. Effectively the LoRA is trained so a **single forward pass** emulates the
CFG(3.0) prediction (guidance distillation). Legitimate technique, and the comment says it
matches ai-toolkit behavior — just confirm that's the intent, because the LoRA also modifies
the weights used by the uncond path while ignoring its gradient. The effective LR on the
cond path is also divided by `cfg` (fine with automagic, which adapts).

Related plumbing that is correct but fragile:
- uncond tensors are smuggled through the `*seg_masks` varargs and detected by counting
  (`num_cond_masks` from the `meta` tensor). Works and is commented, but any future change
  to segment serialization will break silently. A type/length tag in `meta` itself would be
  more robust.
- `has_uncond` detection in `InitialLayer` via `extra[0].dim() == 3` works because ref
  latents are 4-D, but it's implicit.
- `assert pipeline_stages == 1` when `cfg > 1` — good guard (no_grad tensors can't cross
  pipeline stages); matches the `no_backward` precedent in `utils/unsloth_utils.py` /
  `models/minimax_h3.py`.

### 5. `.gitmodules` `ignore = all` additions are unrelated to the feature
Cosmos, HunyuanImage-2.1 and ComfyUI submodules got `ignore = all` in the same commit.
Harmless housekeeping, but mixed-concern (and see issue #2 — it can mask a drifted submodule
state). Would normally be its own commit.

### 6. `models/base.py` defensive isinstance checks
The new `isinstance(v, list) and len(v) > 0 and isinstance(v[0], list)` guards make
`ComfyPipeline` tolerate non-list token-dict values (bools/metadata from the qwen3vl
tokenizer) by passing them through. Works, but:
- in the first loop (length counting) non-list values are silently skipped — if a tokenizer
  ever returned *only* non-list values, `token_lengths[i]` would never be set and later code
  would KeyError rather than fail clearly;
- it's generic base-class code being special-cased for one tokenizer — worth a comment
  naming which tokenizer triggers the non-list path.

### 7. Minor nits
- `prepare_inputs`: `if shift := self.model_config.get('shift', None)` — a user setting
  `shift = 0` is falsy and silently falls through to the default Qwen shift. Unlikely, but
  an explicit `is not None` check is safer.
- `_fold_key_valid` / `bool(mask.all())` cause a GPU→CPU sync per segment; negligible next
  to attention, just noting.
- Config key inconsistency: training uses `cfg`, sampling uses `guidance_scale` (both
  documented in the comment, but two names for near-identical concepts).
- Docs claim "blocks_to_swap, max 30" — this matches the `blocks_to_swap <= num_blocks - 2`
  assert only if the model has 32 blocks; it's derived rather than enforced by a constant.
- `get_conds()` override appears unused by this pipeline (base uses it for its own sample
  path; `prepare_inputs` re-pads itself). Harmless dead-ish code.

---

## Things that look right (spot-checked)

- **`utils/dataset.py`:** switching dispatch from `len(signature(fn).parameters) == 3` to
  `'control_file' in signature(fn).parameters` is strictly more robust and
  backward-compatible (old positional fns still match). The new `size_bucket` pass-through
  is opt-in via `'size_bucket' in signature(fn).parameters` — clean. All producers and
  consumers of the task tuple were updated consistently (`queue.put` in both map fns and
  the `task[1:]` unpack in the worker).
- **Edit/reference-image path:** per-sample text-encoder encoding with `image_slots`
  bookkeeping, batch-consistency assert, and dropping references for the whole batch on
  caption dropout (documented limitation) — consistent and guarded.
- **Shift:** default `get_lin_function(x1=256, y1=0.5, x2=8192, y2=0.9)` on `h*w` latent
  tokens matches the documented scheduler; `shift` / `flux_shift` / `disable_shift`
  overrides mirror other models.
- **VAE RGBA handling:** opaque-alpha pad on encode, alpha strip on decode — correct for
  the 2.1 VAE.
- **Block swap:** dummy offloader replaced in `enable_block_swap`, layers capture the live
  offloader; `checkpointable_layers` excludes `FinalLayer`, which is exactly why the
  uncond-`None`-during-recompute handling in `TransformerLayer` is needed and sufficient
  (FinalLayer is never recomputed, so it never sees `None`).
- **`train.py` / docs table / example TOML:** straightforward and consistent with siblings.

---

## Training-run observation (tensorboard, run `20260921_12-11-06`)

`train/loss`, 1495 steps:

| steps | mean | min | max |
|---|---|---|---|
| 1–149 | 0.167 | 0.067 | 0.526 |
| 299–447 | 0.157 | 0.061 | 0.389 |
| 746–894 | 0.158 | 0.058 | 0.359 |
| 1193–1341 | 0.164 | 0.056 | 0.431 |
| 1342–1495 | ~0.159 | 0.055 | 0.388 |

Loss hovers around ~0.16 with only a slight downward drift (epoch loss 0.169 →
~0.157–0.162, noisy, no divergence). Note `automagic_avg_lr` decayed 1.0e-4 → 5e-6 over the
run, so the flattening is partly the LR schedule, not necessarily a plateau — judge by the
eval samples rather than the loss curve.

---

## Suggested follow-ups

1. Fix the cfg=1 vs guidance-scale=3 docs/code contradiction (#1).
2. Pin/repair the ComfyUI submodule so the commit is reproducible (#2).
3. Consider `is not None` for `shift` (#7) and a comment on the base.py tokenizer path (#6).

---

# Addendum: comparison with ai-toolkit's Qwen-Image 2.1 training

Reference: `ai-toolkit/extensions_built_in/diffusion_models/qwen_image_2/` (+ `src/pipeline.py`,
`src/transformer.py`, and the SDTrainer loss path in `extensions_built_in/sd_trainer/SDTrainer.py`).

## TL;DR

ai-toolkit trains Qwen-Image 2.1 as a **plain conditional flow-matching model** (no uncond branch,
no embedded CFG) and treats it as **guidance-free at sampling** (its 2.1 pipeline defaults
`guidance_scale = 1.0`, with the comment *"2.1 is meant to be sampled without guidance; a scale of
1 skips it"*). Our `qwen_21` commit instead defaults to **embedded-CFG training at cfg=3 and
sampling guidance 3.0**. That resolves issue #1 from the first review — but in the opposite
direction from what the code assumed:

- The `guidance_scale: 3` in ai-toolkit's example configs (`train_lora_qwen_image_24gb.yaml`,
  `train_lora_qwen_image_edit_2509_32gb.yaml`) belongs to `arch: qwen_image` (**1.0**) and
  `qwen_image_edit_plus` (**2509**) — models that genuinely sample with dual-pass CFG. It is **not**
  a 2.1 setting. `DEFAULT_GUIDANCE_SCALE = 3.0` "matching ai-toolkit's behavior" was matching the
  wrong model's config.
- Per ai-toolkit (and our own docs line), 2.1 is **guidance-free-trained**, not CFG-distilled. The
  premise "the model is already cfg distilled, so training normally wouldn't produce an adapter that
  works at cfg=1" doesn't hold: ai-toolkit's normally-trained LoRAs work at cfg=1 precisely because
  the base model is guidance-free — plain conditional training teaches the LoRA the base model's
  native (already "guidance-baked") behavior.

## The interesting part: ai-toolkit *does* have our technique — as an opt-in

`SDTrainer.calculate_loss` has `do_guidance_loss` (`train_config.do_guidance_loss`, **default
`False`**). When enabled it computes, under `torch.no_grad()`, an uncond forward on the cached blank
prompt and builds a **CFG-extrapolated target**:

```python
target = uncond_target + guidance_scale * (target - uncond_target)
```

Our implementation combines on the **prediction side**:

```python
output = (output + (cfg - 1) * output_uncond) / self.cfg   # vs plain target
```

These are algebraically the same objective: minimizing `‖(pred + (c−1)·u)/c − t‖` equals minimizing
`‖pred − (u + c·(t − u))‖`, i.e. target-side guidance with `gs = c`. Differences that do matter:

| | ai-toolkit `do_guidance_loss` | our `qwen_21` CFG |
|---|---|---|
| default | **off** (plain conditional) | **on** at `cfg = 3` |
| gradient scale | full strength (target scaled) | cond-path gradient divided by `cfg` |
| `guidance_loss_schedule = 'sigma'` | decays effective gs → 1 as sigma → 0 | no schedule, constant gs |
| `do_guidance_loss_cfg_zero` | optional CFG-Zero-star per-sample scaling | none |
| uncond pass | separate full forward, reused blocks | interwoven per-layer (same blocks, `no_grad`) |

The sigma schedule is worth noting: at low sigma the `(target − uncond)` direction contains
amplified unpredictable noise, so ai-toolkit anneals guidance toward a plain flow target there. Our
constant-cfg version has a permanently elevated noise floor at low timesteps, which may explain some
of the loss fluctuation in the run above.

## Where the two implementations agree (good parity checks)

- **Flow-matching convention:** t=1 noise → t=0 clean, target `noise − clean`; no time flip or
  negation. Identical (`get_loss_target` vs our `x_0 − x_1`).
- **Timestep sampling:** ai-toolkit's default `timestep_type: 'sigmoid'` ≈ our `logit_normal`
  default.
- **Dynamic shift:** `scheduler_config` (256 → 0.5, 8192 → 0.9, `use_dynamic_shifting`) matches our
  `get_lin_function(x1=256, y1=0.5, x2=8192, y2=0.9)` on latent positions; both land at
  mu ≈ 0.69 for 1024×1024.
- **t=0 prefix modulation:** their transformer's `causal_condition` (text + reference tokens
  modulated from a trailing t=0 row, target tokens from the sampled t) is exactly our
  `torch.cat([t, t.new_zeros(1)])` + `_split_rows` scheme.
- **Reference-image routing:** references ride into the text-encoder embeddings as vision-token
  slots and their VAE latents are spliced into the sequence (4 latent tokens per vision slot /
  32 px grid) — same design on both sides, including bucket-resizing references to the target and
  requiring batch-consistent slot counts.
- **Caption dropout with references:** ai-toolkit gates on `slot_mask.any()` — "a prompt encoded
  without them takes no references here either" — the same semantics as our drop-refs-for-the-batch
  rule.
- **VAA/VAE details:** opaque-alpha pad on encode / alpha strip on decode, 16× spatial, 64 ch —
  same on both sides. (ai-toolkit additionally tiles decodes above 1 MP; we don't, but that's a
  sampling-memory nicety.)

## Differences / risks specific to our CFG path

1. **Edit datasets + cfg > 1 are probably broken.** ai-toolkit *explicitly refuses* to pair
   reference latents with embeddings encoded without references ("a blank unconditional embedding
   cannot be reused here" — it raises, because slots would mis-place). Our `InitialLayer` calls
   `build_sequence(x, context_uncond, ref_latents, image_slots)` with the **cond** caption's slot
   indices applied to the blank **uncond** sequence, which has no `<|image_pad|>` tokens — slot
   positions from a longer, different sequence spliced into a short blank one. Unverified (the
   ComfyUI `qwen_image21` source isn't even in the pinned submodule, see issue #2), but at best
   it's out-of-distribution conditioning. Recommend: force `cfg = 1` for edit datasets, or encode
   the uncond prompt *with* placeholder image tokens so the slot layout matches.
2. **`no_backward` through 32 layers of uncond tensors** doubles the activation traffic between
   pipeline layers even though the uncond branch is grad-free; ai-toolkit just runs one extra
   forward. Simpler alternative matching ai-toolkit: compute the uncond prediction once (outside
   the layer pipeline) and apply the combine in the loss. Our per-layer design only pays off under
   block swap, and even then a single extra forward would too.
3. **Default direction.** Given ai-toolkit's evidence (2.1 guidance-free, `do_guidance_loss` off by
   default, guidance 3 belongs to Qwen-Image 1.0), the safer default for `qwen_image_2_1` is
   `cfg = 1` plain training + `guidance_scale = 1.0` sampling, with the embedded-CFG path kept as
   an explicit opt-in (our equivalent of `do_guidance_loss`), ideally with a sigma-decay schedule
   if it's used.

## Bottom line

The commit's non-CFG machinery (shift, t=0 modulation, reference splicing, slot bookkeeping,
timestep sampling, VAE handling) matches ai-toolkit's implementation point-for-point — that part is
done properly. The CFG/guidance layer is the one real divergence: it's a reasonable
implementation of ai-toolkit's *opt-in* `do_guidance_loss`, but it's wired as the **default** and
keyed to a guidance scale that ai-toolkit only uses for Qwen-Image **1.0**/2509. Suggested changes:
default to cfg=1/guidance 1.0, keep embedded CFG behind an explicit opt-in, gate it off (or fix the
slot layout) for edit datasets, and consider the sigma schedule if the opt-in is used.


