"""
evaluate.py  -  measures the trained model on UNSEEN images.

Put test images in:   dataset/test/good/   dataset/test/defective/   (photos of pens NOT used in train/val)
Run:                  python evaluate.py
Creates in static/evaluation/ : metrics.json, confusion_matrix.png, training_curves.png (if history.json exists)
"""
import json
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import TEST_DIR, EVAL_DIR, CLASS_NAMES, GOOD_CLASS_INDEX, MODEL_DIR, IMG_SIZE
from inference import get_model

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_test_set():
    X, y, names = [], [], []
    for idx, cls in enumerate(CLASS_NAMES):
        folder = os.path.join(TEST_DIR, cls)
        if not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(EXTS):
                img = cv2.imread(os.path.join(folder, f))
                if img is None:
                    continue
                rgb = cv2.cvtColor(cv2.resize(img, IMG_SIZE), cv2.COLOR_BGR2RGB)
                X.append(rgb.astype("float32")); y.append(idx); names.append(f"{cls}/{f}")
    return np.array(X), np.array(y), names


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def main():
    X, y, names = load_test_set()
    if len(X) == 0:
        raise SystemExit(f"No test images found in {TEST_DIR}/<class>/ . Add unseen pen photos first.")
    probs = get_model().predict(X, batch_size=16, verbose=1)
    pred = probs.argmax(axis=1)

    n = len(CLASS_NAMES)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y, pred):
        cm[t, p] += 1

    per_class = {}
    for i, c in enumerate(CLASS_NAMES):
        tp, fp, fn = cm[i, i], cm[:, i].sum() - cm[i, i], cm[i, :].sum() - cm[i, i]
        p, r, f = prf(tp, fp, fn)
        per_class[c] = {"precision": p, "recall": r, "f1": f, "support": int(cm[i, :].sum())}

    # Binary view: defective = positive class (what the factory cares about)
    true_def = y != GOOD_CLASS_INDEX
    pred_def = pred != GOOD_CLASS_INDEX
    tp = int((true_def & pred_def).sum()); fp = int((~true_def & pred_def).sum())
    fn = int((true_def & ~pred_def).sum()); tn = int((~true_def & ~pred_def).sum())
    bp, br, bf = prf(tp, fp, fn)

    metrics = {
        "images_tested": int(len(y)),
        "accuracy": round(float((pred == y).mean()), 4),
        "binary_defective_vs_good": {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
                                     "precision": bp, "recall": br, "f1": bf,
                                     "accuracy": round((tp + tn) / len(y), 4)},
        "per_class": per_class,
        "macro_f1": round(float(np.mean([v["f1"] for v in per_class.values()])), 4),
        "confusion_matrix": cm.tolist(),
        "classes": CLASS_NAMES,
        "misclassified": [names[i] for i in np.where(pred != y)[0]],
    }
    with open(os.path.join(EVAL_DIR, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    # confusion matrix image
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right"); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(EVAL_DIR, "confusion_matrix.png"), dpi=150); plt.close(fig)

    # training curves (needs history.json written by train.py patch)
    hist_path = os.path.join(MODEL_DIR, "history.json")
    if os.path.exists(hist_path):
        h = json.load(open(hist_path))
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        for ax, key, title in ((axs[0], "accuracy", "Accuracy"), (axs[1], "loss", "Loss")):
            ax.plot(h.get(key, []), label="train")
            ax.plot(h.get("val_" + key, []), label="validation")
            ax.set_title(title); ax.set_xlabel("epoch"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(EVAL_DIR, "training_curves.png"), dpi=150); plt.close(fig)

    print("\n========== RESULTS ==========")
    print(f"Images tested : {len(y)}")
    print(f"Accuracy      : {metrics['accuracy']*100:.2f}%")
    print(f"Defective  precision {bp:.3f}  recall {br:.3f}  F1 {bf:.3f}")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(cm)
    print("Saved to", EVAL_DIR)


if __name__ == "__main__":
    main()
