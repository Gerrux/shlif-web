# Panorama inference perf — Phase 1 (contained fixes)

## Problem

Production runs on a single-GPU VM (NVIDIA L4, `docker-compose.yml` with no
`SHLIF_FORCE_CPU` override — GPU auto-detects; the dev override in
`docker-compose.override.yml` forces CPU-only locally). Priority is **latency
of a single panorama analysis**, not multi-job throughput.

Reading `backend/app/pipeline/panorama.py`, `backend/app/shlif/ore_unet.py`,
`backend/app/shlif/talc_unet.py` and `backend/app/shlif/uncertainty.py` turned
up concrete resource under-utilisation in the tiled panorama path:

1. **`ore_unet_mask`** slices each panorama tile (e.g. 2048×2048) into 512×512
   sub-crops and runs them through the U-Net **one at a time** — a Python
   `for` loop issuing a batch-of-1 forward pass per crop, with `.cpu().numpy()`
   forcing a CUDA sync every iteration. On an L4 (24GB VRAM, Ada tensor
   cores) this is a large amount of idle GPU time and avoidable launch/sync
   overhead per ore-bearing tile.
2. **No mixed precision / TF32.** Both U-Nets run plain fp32. L4's tensor
   cores get meaningful speedup from fp16 autocast and TF32 matmul with
   negligible accuracy impact for this task.
3. **`ensemble_uncertainty`** (`uncertainty.py::ensemble_phase_labels`) runs
   the classical `segment_phases` **5 times sequentially** (one per
   photometric perturbation) for **every non-empty tile**, regardless of
   whether the U-Net or classical path handles the actual ore/matrix
   decision. This is pure CPU work and is the most CPU-bound of the loop.
   It never uses more than one core even when many are idle.
4. **`JobRunner(max_workers=1)`** (`backend/app/runtime.py`) serialises all
   jobs process-wide — a trivial, low-risk fix even though throughput isn't
   the stated priority.

## Goal

Reduce single-panorama wall-clock time by fixing the above, **without**
changing `_run_panorama`'s external behaviour, call structure, or existing
test contracts. This is Phase 1 of a two-phase plan; Phase 2 (deeper
cross-tile GPU batching + process-pool parallelisation of the whole per-tile
CPU stage) is deferred until Phase 1 is measured on the actual L4 VM and
found insufficient. Phase 2 is **out of scope** for this spec.

## Architecture

No structural change to `_run_panorama`'s tile loop: tiles are still visited
in the same order, the same accumulators (`color_num`, `weight_den`,
`talc_disp`, `records`, `low_conf_zones`, counts) are updated the same way,
and stitching is untouched. Only the **internals** of three call sites
change:

- `ore_unet_mask(rgb, model, device, tile=512)` — same signature, same
  per-tile call site in `panorama.py`, same return (bool HxW mask). Internals
  go from a sequential per-crop loop to one batched (or chunked-batched)
  forward pass.
- `ore_unet_mask` / `talc_unet_mask` forward passes — wrapped in
  `torch.autocast` on CUDA only; CPU path unaffected.
- `ensemble_phase_labels(rgb, cfg, perturbations)` — same signature and
  return (a `(K, H, W)` label stack), internals go from a sequential `for`
  loop to a persistent thread-pool `map`.
- `backend/app/runtime.py` — one-line `max_workers` bump.

Because none of these change what gets computed (same crops, same
perturbations, same order of accumulation into the stitched output), the
existing tests (`test_panorama_unet_gate.py`, `test_panorama.py`,
`test_tiling_feather.py`, `test_panorama_aggregate.py`) keep passing
unmodified — the panorama-level contract (`ore_unet_mask` invoked once per
non-empty tile with that tile's full `rgb`) is preserved.

## Components

### 1. `ore_unet.py::ore_unet_mask` — batch the under-tile crops

Replace the nested `for y ... for x ...` loop (currently: one
`copyMakeBorder` + `wb_clahe` + normalise + single-image forward pass +
`.cpu().numpy()` per crop) with:

1. Build the list of `(y, x)` crop origins exactly as today (`range(0, H,
   tile)` × `range(0, W, tile)`).
2. For each crop: same `copyMakeBorder` (BORDER_REFLECT to `tile`×`tile`) +
   `wb_clahe` + ImageNet-normalise + transpose to `(C, H, W)` — this stays a
   plain Python loop (cheap CPU work, cv2-bound; not the bottleneck).
3. Stack all crops into one `(N, C, tile, tile)` tensor, move to device once.
4. Run the forward pass in chunks of `batch_size` (new parameter, default
   32) under `torch.inference_mode()` — a tile with more crops than
   `batch_size` still does more than one forward call, but far fewer than
   today's one-per-crop, and each call is a real batch instead of size 1.
5. Concatenate `argmax(1)` results, scatter back into the `(H, W)` output
   mask using the same `[:ch, :cw]` crop-back logic as today.

`batch_size` is a plain function default (consistent with the existing
`tile: int = 512` parameter), not a new config block — YAGNI; a typical
panorama tile (2048/512 = 4×4 = 16 crops) fits in a single batch on an L4
comfortably, so the chunking is a safety net for unusually large tiles, not
the common case.

### 2. Mixed precision + TF32 (`ore_unet.py`, `talc_unet.py`)

- In `build_ore_unet` / `build_talc_unet`, right after confirming
  `dev.startswith("cuda")`: set `torch.backends.cuda.matmul.allow_tf32 = True`
  and `torch.backends.cudnn.allow_tf32 = True`. Process-wide, idempotent,
  safe to set from both loaders.
- In `ore_unet_mask` / `talc_unet_mask`, wrap the forward pass in
  `torch.autocast(device_type="cuda", dtype=torch.float16)` when
  `device` starts with `"cuda"`; plain fp32 forward otherwise (CPU path
  unchanged — `torch.autocast` on CPU wouldn't help here and we keep the
  CPU fallback byte-for-byte as it is today).

### 3. `uncertainty.py::ensemble_phase_labels` — parallel perturbations

The 5 perturbations (`_PERTURBATIONS`) are independent: each does
`_perturb` (pure numpy) → `preprocess` (cv2 white-balance/CLAHE) →
`segment_phases` (skimage Lab conversion + multi-Otsu + cv2 morphology/CC).
All of that is large-array cv2/numpy/skimage work that releases the GIL, so:

- Add a lazily-created, module-level, persistent `ThreadPoolExecutor` (not
  re-created per call — this runs on every non-empty tile, potentially
  thousands of times per gigapixel panorama) sized
  `min(len(_PERTURBATIONS), os.cpu_count() or 1)`.
- `ensemble_phase_labels` submits the 5 `(gamma, gain)` jobs to the pool and
  stacks results in the original perturbation order (order matters for
  reproducibility of `labels`, even though the aggregation in
  `confidence_map`/`entropy_map` is order-independent — keep it stable
  anyway for the equivalence test).
- If measurement on the real L4 VM shows GIL contention still limiting
  gains, swapping the executor to `ProcessPoolExecutor` is a contained,
  same-call-site follow-up (noted here, not implemented in Phase 1).

### 4. `backend/app/runtime.py` — job concurrency

`JobRunner(self.store)` → `JobRunner(self.store, max_workers=2)`. Orthogonal,
low-risk, cheap; not the focus (priority is single-panorama latency) but
removes an obvious, free-standing bottleneck.

## Numeric equivalence

Tile traversal order, crop-to-output-pixel mapping, and the stitching math
are all unchanged — batching a forward pass does not change which crop
produces which output pixels. The only source of numeric drift is fp16
autocast (rounding vs fp32), which is expected for a perf change and is
very unlikely to flip an `argmax` decision in practice. This is a
perf-only change; no behavioural/verdict change is intended.

## Error handling

- `build_ore_unet` / `build_talc_unet` keep their existing guarded-`None`
  contract exactly (missing checkpoint, missing torch/smp, or a load
  failure → `None` → caller falls back to the classical path). Batching only
  changes the internals of the "happy path" once a model is loaded.
- The `ensemble_phase_labels` thread pool: if pool creation or submission
  raises (unexpected — thread pools essentially don't fail to construct),
  there is no special fallback; this mirrors today's behaviour where a
  failure in `segment_phases` propagates as an exception either way.

## Testing

Existing tests unchanged and must keep passing:
`test_panorama_unet_gate.py`, `test_panorama.py`, `test_tiling_feather.py`,
`test_panorama_aggregate.py`.

New tests:

- `ore_unet_mask` batched-vs-sequential equivalence: a deterministic stub
  model (e.g. one that returns a fixed/known argmax pattern per crop
  position) run over a synthetic multi-crop tile (e.g. 1024×1024 with
  `tile=512` → 2×2=4 crops), asserting the batched output matches a
  hand-computed per-crop-reference result.
- `ensemble_phase_labels` parallel-vs-sequential equivalence: same `rgb`
  input run through both the threaded and a reference sequential
  implementation, asserting identical `(K, H, W)` label stacks (same order).
- `JobRunner` max_workers bump: no new test needed (trivial constructor
  argument, already exercised implicitly by existing job tests if any; a
  targeted test asserting `max_workers == 2` is optional/low value).

## Rollout / measurement

This is a perf-only change validated by:

1. `pytest` (existing suite green + the two new equivalence tests).
2. Manual timing on the actual L4 VM (not reproducible in this sandbox —
   no GPU/torch here): compare `_run_panorama`'s `seconds` field
   before/after on a representative panorama, plus `nvidia-smi` GPU
   utilisation during the run, to confirm the GPU is actually kept busier
   (not just "should be faster in theory").
3. If GPU utilisation still stays low or CPU is still the bottleneck after
   Phase 1 lands, Phase 2 (cross-tile GPU batching, process-pool over the
   full per-tile CPU stage) is the documented next step — not designed
   here.
