"""Evaluation metrics.

Per the organiser: segmentation is judged by IoU and Hausdorff distance,
classification by F1 and AUC. These are for our own validation and reporting —
there is no automated grader in the hackathon.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection-over-Union (Jaccard) of two boolean masks."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0  # both empty -> perfect agreement
    return float(np.logical_and(pred, gt).sum()) / float(union)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(pred, gt).sum()) / float(denom)


def hausdorff(pred: np.ndarray, gt: np.ndarray) -> float:
    """Symmetric Hausdorff distance between mask boundaries (pixels).

    Inf if exactly one mask is empty; 0.0 if both empty.
    """
    from skimage.metrics import hausdorff_distance

    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return float("inf")
    return float(hausdorff_distance(pred, gt))


def hausdorff95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th-percentile Hausdorff (HD95) — robust to a few outlier pixels."""
    from scipy.spatial import cKDTree
    from skimage.segmentation import find_boundaries

    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return float("inf")
    pb = np.argwhere(find_boundaries(pred, mode="inner"))
    gb = np.argwhere(find_boundaries(gt, mode="inner"))
    if len(pb) == 0 or len(gb) == 0:
        return float("inf")
    d_pg = cKDTree(gb).query(pb)[0]
    d_gp = cKDTree(pb).query(gb)[0]
    return float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))


def segmentation_report(pred: np.ndarray, gt: np.ndarray) -> dict:
    """All segmentation metrics for one mask pair."""
    return {
        "iou": iou(pred, gt),
        "dice": dice(pred, gt),
        "hausdorff": hausdorff(pred, gt),
        "hausdorff95": hausdorff95(pred, gt),
    }


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def classification_report(y_true, y_pred, y_score=None, labels=None) -> dict:
    """Macro-F1, per-class F1, confusion matrix, and (if scores given) OvR AUC."""
    from sklearn.metrics import confusion_matrix, f1_score

    out = {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=labels)),
        "f1_per_class": f1_score(y_true, y_pred, average=None, labels=labels).tolist(),
        "labels": list(labels) if labels is not None else sorted(set(y_true)),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if y_score is not None:
        from sklearn.metrics import roc_auc_score

        try:
            out["auc_ovr_macro"] = float(
                roc_auc_score(y_true, y_score, multi_class="ovr", average="macro", labels=labels)
            )
        except ValueError as exc:
            out["auc_ovr_macro"] = None
            out["auc_error"] = str(exc)
    return out
