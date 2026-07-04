"""Intergrowth / ore-class classification: build features, cross-validate, train.

Outputs class *probabilities* (predict_proba) so AUC is computable, as the
organiser requires. Validation is stratified K-fold on the labelled close-ups —
a balanced sample across classes, which is what they recommend (panoramas are an
unlabelled qualitative test set, not a metric target).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import extract_features
from .imageio import load_rgb
from .metrics import classification_report


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    paths: list[str]


def build_dataset(pairs, cfg, max_pixels: int = 12_000_000, verbose: bool = True) -> Dataset:
    """Extract features for [(path, label), ...]."""
    rows, labels, names, feat_names = [], [], [], None
    for i, (path, label) in enumerate(pairs):
        rgb = load_rgb(path, max_pixels=max_pixels)
        feats = extract_features(rgb, cfg, name=Path(path).name)
        if feat_names is None:
            feat_names = sorted(feats)
        rows.append([feats[k] for k in feat_names])
        labels.append(label)
        names.append(str(path))
        if verbose and (i + 1) % 50 == 0:
            print(f"  features {i + 1}/{len(pairs)}")
    return Dataset(np.array(rows, float), np.array(labels), feat_names, names)


def cross_validate(ds: Dataset, n_splits: int = 5, seed: int = 0) -> dict:
    """Stratified K-fold; report macro-F1, per-class F1, macro-AUC, confusion."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold

    labels = sorted(set(ds.y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_pred = np.empty_like(ds.y)
    y_score = np.zeros((len(ds.y), len(labels)))
    for tr, te in skf.split(ds.X, ds.y):
        clf = RandomForestClassifier(
            n_estimators=400, max_depth=None, class_weight="balanced",
            n_jobs=-1, random_state=seed,
        )
        clf.fit(ds.X[tr], ds.y[tr])
        y_pred[te] = clf.predict(ds.X[te])
        proba = clf.predict_proba(ds.X[te])
        for j, c in enumerate(clf.classes_):
            y_score[te, labels.index(c)] = proba[:, j]
    y_true_idx = np.array([labels.index(v) for v in ds.y])
    report = classification_report(ds.y, y_pred, y_score=y_score, labels=labels)
    report["y_score_labels"] = labels
    report["_y_true_idx"] = y_true_idx.tolist()
    return report


def train_full(ds: Dataset, seed: int = 0):
    """Fit a final classifier on all data (for inference / feature importance)."""
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(
        n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=seed
    )
    clf.fit(ds.X, ds.y)
    importances = sorted(
        zip(ds.feature_names, clf.feature_importances_), key=lambda t: -t[1]
    )
    return clf, importances
