# Segment-Phases Magnetite Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix magnetite recall in `backend/app/shlif/segment.py::segment_phases` by removing the `not_olive` gate (built on an inverted, empirically-refuted hue assumption), and re-validate the RF sort classifier whose features depend on it.

**Architecture:** `segment_phases` currently requires a magnetite candidate to satisfy `mid & neutral & not_olive` — the `not_olive` clause (`b >= green_b_min`) assumes olive/warm hue belongs to matrix, but LumenStone ground truth and the project's own labelled images show the opposite (olive = sulfide; true magnetite skews cool/negative-b) and the absolute threshold doesn't transfer across images with different lighting. Removing `not_olive` is a strict widening of the magnetite mask (nothing that was magnetite before stops being magnetite), verified against the one existing test that exercises a magnetite fixture. The already-shipped `ensemble_uncertainty` mechanism needs no changes — it re-runs the (now-corrected) `segment_phases` under perturbation and will keep flagging genuinely ambiguous magnetite/sulfide pixels automatically. Because `features.py::extract_features` derives `magn_frac` and sulfide-mask texture from `segment_phases`, the RF sort classifier must be re-extracted and re-trained against the new segmentation, using `sumple_dataset` in the sibling `hakaton_nornikel` checkout as a **read-only** data source (that repo's changes are explicitly out of scope for this plan — do not modify anything under `/home/claude/hakaton_nornikel`).

**Tech Stack:** Python 3.12, scikit-image (Lab colour segmentation), scikit-learn (RandomForest), pytest.

## Global Constraints

- **Do not modify anything under `/home/claude/hakaton_nornikel`** — that repo currently has unrelated uncommitted work from a different session (`scripts/measure_domain.py`, `.deploy_check.txt`). Only *read* `/home/claude/hakaton_nornikel/sumple_dataset/` (an external, read-only data source referenced by absolute path); never write into that directory tree.
- Removing `not_olive` must not remove any pixel that was previously classified magnetite — verify this is a strict widening (see Task 2's reasoning) and confirm via the full test suite, especially `backend/tests/test_panorama_assemble.py` (its magnetite fixture uses a flat neutral grey `(60,60,60)` with `b≈0`, which already passes `not_olive` today — should be unaffected, but run it explicitly to confirm).
- Run tests with `cd backend && .venv/bin/pytest -q` (absolute venv path).
- Feature extraction over the full 1178-image `sumple_dataset` takes roughly **40 minutes** (verified: 6 images in 12.6s). Any step that runs the full dataset MUST use `run_in_background: true` on the Bash call (or `nohup ... &` and poll) — it will exceed the 10-minute foreground Bash timeout otherwise.
- The retrain script's classifier pickle must be produced using **this repo's own `backend/.venv`** (whatever scikit-learn version is installed there — 1.9.0 in this checkout), not `hakaton_nornikel`'s separate venv, so the train-time and serve-time scikit-learn versions match within this repo (there is no hard version pin in `pyproject.toml` today — do not add one as part of this plan).

---

### Task 1: Add the classifier retraining script and capture the pre-fix baseline

**Files:**
- Create: `backend/scripts/retrain_classifier.py`
- Test: none (this is a one-off operational script, not application code; its correctness is demonstrated by the baseline numbers it prints, captured in this task's commit message)

**Interfaces:**
- Consumes: `app.shlif.imageio.list_class_images(root) -> list[tuple[Path, str]]`, `app.shlif.classify.build_dataset(pairs, cfg, verbose=bool) -> Dataset`, `app.shlif.classify.cross_validate(ds, n_splits, seed) -> dict` (keys: `f1_macro`, `f1_per_class`, `labels`, `confusion`, `auc_ovr_macro`), `app.shlif.classify.train_full(ds, seed) -> (clf, importances)`, `app.shlif.config.load_config() -> Config` — all already exist, unchanged, verified working end-to-end against `/home/claude/hakaton_nornikel/sumple_dataset` (1178 images found, `build_dataset` on a 6-image sample produces a `(6, 32)` feature matrix in ~12.6s).
- Produces: `backend/scripts/retrain_classifier.py`, invoked as `.venv/bin/python scripts/retrain_classifier.py --root <path> --per-class N --cache <path> --out <path>` (all flags optional, defaults below). Prints `f1_macro`, per-class F1, `auc_ovr_macro` to stdout and pickles `{"clf", "feature_names", "classes"}` to `--out`. Task 3 re-runs this same script after the segment.py fix and diffs the printed numbers against this task's baseline.

- [ ] **Step 1: Create the retraining script**

Create `backend/scripts/retrain_classifier.py`:

```python
"""Re-extract features from sumple_dataset and retrain the ore-sort RF classifier.

sumple_dataset is NOT part of this repo (gitignored elsewhere, and here it is read
directly from the sibling hakaton_nornikel checkout via --root). This script only
reads from --root; it never writes there.

    # sanity check (fast, ~2 min):
    .venv/bin/python scripts/retrain_classifier.py --per-class 40 --cache /tmp/feat_sample.npz --out /tmp/clf_sample.pkl
    # full run (~40 min, run in background):
    .venv/bin/python scripts/retrain_classifier.py --cache /tmp/features_full.npz --out /tmp/classifier_new.pkl
"""

from __future__ import annotations

import argparse
import pickle
import random
import time
from pathlib import Path

import numpy as np

from app.shlif.classify import Dataset, build_dataset, cross_validate, train_full
from app.shlif.config import load_config
from app.shlif.imageio import list_class_images


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/claude/hakaton_nornikel/sumple_dataset",
                    help="read-only path to the labelled close-up dataset")
    ap.add_argument("--per-class", type=int, default=0, help="0 = use all images")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="/tmp/features_full.npz")
    ap.add_argument("--out", default="/tmp/classifier_new.pkl")
    args = ap.parse_args()

    cfg = load_config()
    rng = random.Random(args.seed)

    pairs = list_class_images(args.root)
    if args.per_class > 0:
        by: dict[str, list] = {}
        for p, l in pairs:
            by.setdefault(l, []).append((p, l))
        pairs = []
        for l, items in by.items():
            pairs += rng.sample(items, min(args.per_class, len(items)))
        rng.shuffle(pairs)

    print(f"extracting features for {len(pairs)} images...")
    t0 = time.time()
    ds = build_dataset(pairs, cfg)
    print(f"features done in {time.time() - t0:.0f}s | matrix {ds.X.shape}")

    Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.cache, X=ds.X, y=ds.y, feature_names=ds.feature_names)

    rep = cross_validate(ds, n_splits=args.folds, seed=args.seed)
    print("\n=== Stratified %d-fold CV ===" % args.folds)
    print(f"labels        : {rep['labels']}")
    print(f"macro-F1      : {rep['f1_macro']:.3f}")
    print(f"per-class F1  : " + ", ".join(f"{l}={f:.3f}" for l, f in zip(rep['labels'], rep['f1_per_class'])))
    print(f"macro-AUC OvR : {rep.get('auc_ovr_macro')}")
    print("confusion (rows=true, cols=pred):")
    print(f"    {rep['labels']}")
    for label, row in zip(rep["labels"], rep["confusion"]):
        print(f"    {label:10} {row}")

    clf, importances = train_full(ds, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"clf": clf, "feature_names": ds.feature_names, "classes": list(clf.classes_)}, f)
    print(f"\nsaved {args.out} | classes {list(clf.classes_)} | {len(ds.feature_names)} features | n={len(ds.y)}")
    print("top importances:", [(n, round(v, 3)) for n, v in importances[:8]])


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sanity-check the script on a small sample**

Run (from `backend/`): `.venv/bin/python scripts/retrain_classifier.py --per-class 40 --cache /tmp/feat_sample.npz --out /tmp/clf_sample.pkl`
Expected: completes in ~2 minutes, prints `macro-F1` and `macro-AUC OvR` as plausible numbers in `[0, 1]` (not `nan`, not erroring), and reports `saved /tmp/clf_sample.pkl`.

- [ ] **Step 3: Run the full pre-fix baseline in the background**

Run with `run_in_background: true` (from `backend/`):
`.venv/bin/python scripts/retrain_classifier.py --cache /tmp/features_baseline.npz --out /tmp/classifier_baseline.pkl`
Expected: takes roughly 40 minutes. When it completes, record the printed `macro-F1`, per-class F1, and `macro-AUC OvR` — these are the **pre-fix baseline** Task 3 will compare against. (Reference point only, not an assertion: prior runs of this same feature pipeline on this dataset reported macro-F1 around 0.74 and macro-AUC around 0.90–0.94 — a wildly different number, e.g. near 0 or 1, indicates something is broken, not a real baseline.)

- [ ] **Step 4: Commit**

```bash
git add backend/scripts/retrain_classifier.py
git commit -m "$(cat <<'EOF'
feat(scripts): add classifier retraining script, capture pre-fix baseline

Re-extracts features from sumple_dataset (read-only, sibling hakaton_nornikel
checkout) and retrains the RF sort classifier -- needed because Task 2's
segment_phases fix changes magn_frac/sulfide-texture features it depends on.
Baseline captured pre-fix: macro-F1 <value>, macro-AUC <value> (see commit
body / task report for the full printed CV output) -- Task 3 compares against
this after the fix.
EOF
)"
```
(Fill in the actual printed values from Step 3 in the commit message body before committing.)

---

### Task 2: Remove the `not_olive` gate from `segment_phases`

**Files:**
- Modify: `backend/app/shlif/segment.py:1-12` (module docstring), `:86-92` (magnetite gate)
- Modify: `backend/app/config/default.yaml:10-19` (`segment:` block)
- Test: `backend/tests/test_segment.py` (new)

**Interfaces:**
- Consumes: nothing new.
- Produces: `segment_phases`'s magnetite mask is now `mid & neutral` (no third clause). `cfg.segment.green_b_min` no longer exists — confirm nothing else reads it (already checked: only `segment.py` and `config/default.yaml` reference it in this repo).

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_segment.py`:

```python
"""segment_phases's magnetite gate must not reject true magnetite for being
cool-toned (negative Lab b) -- the `not_olive` assumption it used to encode
(olive/warm hue = matrix) is backwards for this material: LumenStone ground
truth and the project's own labelled images show olive = sulfide, and real
magnetite skews cool (negative b), not warm."""
import numpy as np

from app.shlif.config import load_config
from app.shlif.segment import segment_phases

CFG = load_config()


def test_cool_toned_mid_grey_patch_is_magnetite_not_matrix():
    rng = np.random.default_rng(0)
    img = rng.integers(15, 35, (256, 256, 3)).astype(np.uint8)  # dark matrix background
    img[40:110, 40:110] = (220, 220, 220)     # bright neutral block -> sulfide
    img[150:210, 150:210] = (100, 108, 116)   # cool-toned mid-grey -> should be magnetite
                                                # (L=45.2, a=-1.3, b=-5.5, chroma=5.6 --
                                                # b is below the old green_b_min=-4.0 floor,
                                                # which used to misclassify this as matrix)

    seg = segment_phases(img, CFG.segment)

    assert seg.magnetite[150:210, 150:210].mean() > 0.9
    assert seg.sulfide[40:110, 40:110].mean() > 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/pytest -q tests/test_segment.py -v`
Expected: FAIL — `assert 0.0 > 0.9` (verified during planning: the patch is currently classified 100% matrix, 0% magnetite).

- [ ] **Step 3: Remove the `not_olive` gate**

In `backend/app/shlif/segment.py`, change:

```python
    # magnetite: mid reflectance, neutral (low chroma), and not the olive matrix.
    # olive/gangue is green-yellow -> excluded by requiring low chroma AND b above
    # a small floor (so we don't pick strongly-coloured matrix as "grey").
    mid = (L >= dark_t) & ~sulfide
    neutral = chroma <= float(cfg.chroma_max)
    not_olive = b >= float(cfg.green_b_min)
    magnetite = mid & neutral & not_olive
```

to:

```python
    # magnetite: mid reflectance, neutral (low chroma). No hue/warmth gate --
    # a `not_olive` filter here used to assume olive hue = matrix, but that's
    # backwards for this material (olive = sulfide; real magnetite skews cool,
    # negative Lab b) and an absolute b-channel floor doesn't transfer across
    # images with different lighting/white-balance anyway. Genuinely ambiguous
    # magnetite/sulfide pixels are caught by uncertainty.py's perturbation
    # ensemble instead of forced here.
    mid = (L >= dark_t) & ~sulfide
    neutral = chroma <= float(cfg.chroma_max)
    magnetite = mid & neutral
```

Also update the module docstring at the top of the file, which currently reads:

```python
"""Three-phase reflectance segmentation: sulfide / magnetite / matrix.

Method (all thresholds adaptive per image, tunable via config):
  * Work in CIE-Lab. L = reflectance, chroma = sqrt(a^2+b^2), b = warmth.
  * Sulfide is the *brightest* phase in both bright close-ups and dark panoramas
    -> take the top band of a 3-level Otsu on L.
  * Magnetite is mid-reflectance and *neutral* (low chroma) -> mid L band with
    small chroma, excluding the olive (green-warm) matrix by its chroma/hue.
  * Everything else is matrix.

Returns an integer label map using the constants in :mod:`shlif.phases`.
"""
```

Change the third bullet to remove the now-inaccurate hue-exclusion claim:

```python
"""Three-phase reflectance segmentation: sulfide / magnetite / matrix.

Method (all thresholds adaptive per image, tunable via config):
  * Work in CIE-Lab. L = reflectance, chroma = sqrt(a^2+b^2), b = warmth.
  * Sulfide is the *brightest* phase in both bright close-ups and dark panoramas
    -> take the top band of a 3-level Otsu on L.
  * Magnetite is mid-reflectance and *neutral* (low chroma) -> mid L band with
    small chroma. No hue/warmth gate (see segment_phases for why).
  * Everything else is matrix.

Returns an integer label map using the constants in :mod:`shlif.phases`.
"""
```

- [ ] **Step 4: Remove the now-unused `green_b_min` config field**

In `backend/app/config/default.yaml`, change:

```yaml
segment:
  # Сегментация трёх фаз по отражательной способности + цвету.
  # Сульфид — самая яркая фаза; магнетит — серый средней яркости, нейтральный;
  # матрица — тёмная (панорамы) либо оливковая (крупные планы).
  min_ore_area_px: 24       # удаляем связные компоненты мельче, px
  close_radius: 2           # морфологическое закрытие масок
  bright_percentile: 99.0   # запасной порог сульфида, если multiotsu не сработал
  chroma_max: 18.0          # магнетит нейтрален: хрома (Lab) не выше этого
  green_b_min: -4.0         # оливковую матрицу (зелёно-жёлтую) не считаем магнетитом
  sulfide_min_L: 12.0       # абсолютный пол яркости сульфида (Lab L, 0..100)
```

to:

```yaml
segment:
  # Сегментация трёх фаз по отражательной способности + цвету.
  # Сульфид — самая яркая фаза; магнетит — серый средней яркости, нейтральный;
  # матрица — тёмная (панорамы) либо оливковая (крупные планы).
  min_ore_area_px: 24       # удаляем связные компоненты мельче, px
  close_radius: 2           # морфологическое закрытие масок
  bright_percentile: 99.0   # запасной порог сульфида, если multiotsu не сработал
  chroma_max: 18.0          # магнетит нейтрален: хрома (Lab) не выше этого
  sulfide_min_L: 12.0       # абсолютный пол яркости сульфида (Lab L, 0..100)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && .venv/bin/pytest -q tests/test_segment.py -v`
Expected: PASS (verified during planning: after removing `not_olive`, the same patch is 100% magnetite, 0% sulfide/matrix).

- [ ] **Step 6: Run the full backend suite, with explicit attention to `test_panorama_assemble.py`**

Run: `cd backend && .venv/bin/pytest -q tests/test_panorama_assemble.py -v`
Expected: PASS. Its magnetite fixture is a flat `(60,60,60)` grey (`a=0, b=0`), which already satisfied the old `not_olive` gate trivially — removing that gate cannot un-classify it (magnetite is a strict superset of before: every pixel that passed `mid & neutral & not_olive` still passes `mid & neutral`).

Then run: `cd backend && .venv/bin/pytest -q`
Expected: all passing, no regressions.

- [ ] **Step 7: Commit**

```bash
git add backend/app/shlif/segment.py backend/app/config/default.yaml backend/tests/test_segment.py
git commit -m "$(cat <<'EOF'
fix(shlif): remove the not_olive gate from segment_phases

Confirmed on three independent sources that its assumption (olive/warm hue =
matrix) is backwards: LumenStone ground truth (magnetite recall was 12.7%,
71.5% of true magnetite misclassified as matrix -- 57.1% of that loss
attributable to this one gate), the project's own labelled images (a domain
expert confirmed olive = sulfide, not matrix, and a confirmed-magnetite blob
under an unusual lighting cast reads olive), and a threshold sweep showing
the gate barely reduces matrix false-positives (37.3%->43.6% across the whole
range) while gutting magnetite recall (12.0%->69.0%). Removing it is a strict
widening of the magnetite mask; genuinely ambiguous magnetite/sulfide pixels
are already caught by uncertainty.py's perturbation ensemble with no changes
needed there.
EOF
)"
```

---

### Task 3: Retrain the classifier against the fix, compare to baseline, and deploy

**Files:**
- Replace (NOT git-tracked — `backend/models/` is gitignored, see `.gitignore:2`; this is a local/deployment artifact update, not a commit): `backend/models/classifier.pkl`
- Modify: `backend/app/shlif/VENDORED.md`
- Modify: `README.md` (only if the F1/AUC numbers change meaningfully from what's currently documented)

**Interfaces:**
- Consumes: `backend/scripts/retrain_classifier.py` from Task 1 (unchanged), the fixed `segment_phases` from Task 2 (via `features.py`, transitively).
- Produces: an updated `backend/models/classifier.pkl` bundle (same `{"clf", "feature_names", "classes"}` shape as before — no consumer-visible interface change, `loader.py::load_classifier()` reads it identically).

- [ ] **Step 1: Run the full post-fix retrain in the background**

Run with `run_in_background: true` (from `backend/`):
`.venv/bin/python scripts/retrain_classifier.py --cache /tmp/features_postfix.npz --out /tmp/classifier_postfix.pkl`
Expected: ~40 minutes (same as Task 1's baseline run, now against the fixed `segment_phases`).

- [ ] **Step 2: Compare against Task 1's baseline**

Compare the printed `macro-F1` and `macro-AUC OvR` from this run against Task 1's Step 3 baseline. The design's tolerance (from `docs/superpowers/specs/2026-07-04-segment-phases-magnetite-fix-design.md`): the fix must not meaningfully regress these numbers — small movement either direction is acceptable (precedent: a prior dataset-dedup change *raised* these numbers rather than lowering them, so a neutral-to-positive shift is plausible, not just a risk to guard against). If `macro-F1` or `macro-AUC OvR` drops by more than ~0.03 from baseline, STOP and report BLOCKED rather than deploying — that would mean the new `magn_frac`/texture features are actively worse for sort classification, which needs human judgment on whether to proceed anyway (the segmentation fix is evidence-backed independent of this) or investigate further.

- [ ] **Step 3: Deploy the new classifier pickle locally**

```bash
cp /tmp/classifier_postfix.pkl backend/models/classifier.pkl
```

`backend/models/` is gitignored (`.gitignore:2` — model artifacts are never committed, see
`README.md`'s "Models" section) — this `cp` makes the local worktree's tests reflect the new
classifier, but it is **not** part of any git commit in this task. Shipping this artifact to a
running deployment (VM/container) is a separate, manual deployment step outside this plan's
scope (same as how `unet_talc.pt` updates have been handled previously — `scp` + service
restart, not git).

- [ ] **Step 4: Run the full backend suite**

Run: `cd backend && .venv/bin/pytest -q`
Expected: all passing. (`test_panorama.py`/`test_panorama_unet_gate.py`/etc. that depend on `models/classifier.pkl` being loadable should still pass — the pickle shape is unchanged, only the fitted weights differ.)

- [ ] **Step 5: Update `VENDORED.md`**

In `backend/app/shlif/VENDORED.md`, under the existing `## Divergence from origin (2026-07-04)` section, add a bullet after the last existing one:

```markdown
- `segment.py` — removed the `not_olive` gate from `segment_phases`'s magnetite
  criterion (was `mid & neutral & not_olive`, now `mid & neutral`). Confirmed on
  LumenStone ground truth + the project's own labelled images that the gate's
  assumption (olive hue = matrix) is backwards (olive = sulfide) and the
  underlying absolute Lab b-channel threshold doesn't transfer across images
  with different lighting. `config/default.yaml`'s `segment.green_b_min` is
  removed accordingly. `backend/models/classifier.pkl` (not git-tracked --
  `backend/models/` is gitignored) was re-extracted and retrained locally
  against the new segmentation (see `backend/scripts/retrain_classifier.py`);
  shipping that artifact to a running deployment is a separate manual step.
  Not yet ported back to origin `hakaton_nornikel` (that repo had unrelated
  in-progress work at the time of this fix — port forward when convenient).
```

- [ ] **Step 6: Update `README.md` if the classifier metrics changed**

Read the current line in `README.md` (`| \`classifier.pkl\` | The ore-sort card (RandomForest, F1 0.84 / AUC 0.92) on close-ups **and** the section verdict on panoramas | ... |`). If Step 2's new macro-F1/macro-AUC differ from `0.84`/`0.92` by more than rounding (±0.01), update this line to the new values; otherwise leave it unchanged.

- [ ] **Step 7: Commit**

`backend/models/classifier.pkl` is gitignored — do NOT attempt to `git add` it (it will be
silently skipped or, depending on git version, error/warn on an explicit ignored-path add; either
way it is not supposed to enter git). Only the tracked doc files change:

```bash
git add backend/app/shlif/VENDORED.md README.md
git commit -m "$(cat <<'EOF'
docs(shlif): record classifier retrain against the fixed segment_phases

Re-extracted features and retrained the RF sort classifier on the full
sumple_dataset (1178 images) using the corrected magnetite gate from the
previous commit. macro-F1 <baseline> -> <postfix>, macro-AUC <baseline> ->
<postfix> (see Task 1/Task 3 reports for full CV output) -- no meaningful
regression from the pre-fix baseline. The retrained backend/models/classifier.pkl
itself is gitignored (not part of this commit); shipping it to a running
deployment is a separate manual step.
EOF
)"
```
(Fill in the actual before/after values in the commit message before committing. If Step 6 changed
`README.md`'s documented F1/AUC, that file's diff is the only visible sign this task ran, besides
the commit body.)

---

## Non-goals (explicitly out of scope for this plan)

- **Any change to `/home/claude/hakaton_nornikel`** — that repo has unrelated uncommitted work from a concurrent session right now. Porting this fix back there (segment.py, config, WORKLOG.md, CLAUDE.md updates, `scripts/eval_lumenstone.py` re-run for an external-validation number) is deferred to a separate pass once that repo's main is confirmed clear.
- **Wiring `unet_s2.pt` (with or without hue-augmentation retraining)** — empirically tested and rejected (see the design doc): severe domain-transfer failure on the project's own images despite strong LumenStone-only metrics.
- **Removing or retuning the `neutral` (chroma) gate** — no evidence it's harmful (only 0.1% of magnetite loss attributable to it in the gate-by-gate breakdown); leave it as-is.
- **Any change to `uncertainty.py`** — it needs no changes; it transparently benefits from the corrected `segment_phases` it already re-runs under perturbation.
- **Any change to talc detection/classification** — unrelated to this bug.
