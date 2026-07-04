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
