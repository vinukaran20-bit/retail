"""
inference.py
Single place that loads the model and inspects one image.
Used by app.py (upload + webcam), camera_live.py and evaluate.py.
Verdicts: GOOD, DEFECTIVE, NEEDS REVIEW (confidence below REVIEW_THRESHOLD).
"""
import os
import threading
import uuid

import cv2
import numpy as np
import tensorflow as tf

from config import (
    MODEL_PATH, IMG_SIZE, CLASS_NAMES, GOOD_CLASS_INDEX, RESULT_DIR,
    DEFECT_EXPLANATIONS, REVIEW_THRESHOLD,
)
from gradcam import make_gradcam_heatmap, overlay_heatmap, heatmap_region_description

_model = None
_lock = threading.Lock()


def model_ready() -> bool:
    return os.path.exists(MODEL_PATH)


def get_model():
    global _model
    if _model is None:
        try:
            _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        except Exception:
            _model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
    return _model


def to_batch(img_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, IMG_SIZE)
    return np.expand_dims(rgb.astype("float32"), 0)   # raw 0-255, model normalises internally


def inspect_bgr(img_bgr: np.ndarray, save_overlay: bool = True) -> dict:
    """Runs the model on a BGR image (as read by OpenCV)."""
    batch = to_batch(img_bgr)
    with _lock:
        model = get_model()
        probs = model.predict(batch, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    p_good = float(probs[GOOD_CLASS_INDEX])
    p_def = 1.0 - p_good
    is_def = p_def >= 0.5
    confidence = max(p_good, p_def)

    defect_type = "-"
    if is_def and len(CLASS_NAMES) > 2:
        masked = probs.copy()
        masked[GOOD_CLASS_INDEX] = -1
        defect_type = CLASS_NAMES[int(np.argmax(masked))]
    elif is_def:
        defect_type = "defective"

    if confidence < REVIEW_THRESHOLD:
        verdict = "NEEDS REVIEW"
    else:
        verdict = "DEFECTIVE" if is_def else "GOOD"

    overlay_path, region = None, ""
    if save_overlay:
        try:
            with _lock:
                heatmap = make_gradcam_heatmap(batch, model, pred_idx)
            region = heatmap_region_description(heatmap)
            h, w = img_bgr.shape[:2]
            scale = 640 / max(h, w)
            shown = cv2.resize(img_bgr, (int(w * scale), int(h * scale))) if scale < 1 else img_bgr
            overlay = overlay_heatmap(shown, heatmap)
            os.makedirs(RESULT_DIR, exist_ok=True)
            overlay_path = os.path.join(RESULT_DIR, f"{uuid.uuid4().hex}_gradcam.jpg")
            cv2.imwrite(overlay_path, overlay)
        except Exception as exc:  # Grad-CAM must never break an inspection
            print("Grad-CAM failed:", exc)
            overlay_path = None

    if verdict == "DEFECTIVE":
        key = defect_type if defect_type in DEFECT_EXPLANATIONS else "defective"
        reason = DEFECT_EXPLANATIONS.get(key, "A visual defect was detected.")
        if region:
            reason += f" The model focused mostly on the {region}."
    elif verdict == "GOOD":
        reason = DEFECT_EXPLANATIONS.get("good", "No visual defects were detected.")
    else:
        reason = (f"The model is only {confidence*100:.1f}% sure (limit {REVIEW_THRESHOLD*100:.0f}%). "
                  "Please inspect this item manually.")

    return {
        "verdict": verdict,
        "defect_type": defect_type,
        "confidence": round(confidence * 100, 1),
        "reason": reason,
        "class_probabilities": {c: float(p) for c, p in zip(CLASS_NAMES, probs)},
        "gradcam_overlay_path": overlay_path,
    }
