"""
predict.py
Loads the trained model and runs inference on a single product image.

Usage:
    python predict.py path/to/image.jpg

Returns a structured result: verdict (GOOD/DEFECTIVE), defect type,
confidence score, plain-language reason, and a saved Grad-CAM overlay
image showing WHERE the defect was detected.
"""

import os
import sys
import numpy as np
import cv2
import tensorflow as tf

from config import (
    MODEL_PATH, IMG_SIZE, CLASS_NAMES, GOOD_CLASS_INDEX,
    DEFECT_EXPLANATIONS, RESULT_DIR,
)
from gradcam import make_gradcam_heatmap, overlay_heatmap, heatmap_region_description

_model_cache = {"model": None}


def load_model() -> tf.keras.Model:
    if _model_cache["model"] is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. Run train.py first."
            )
        _model_cache["model"] = tf.keras.models.load_model(MODEL_PATH)
    return _model_cache["model"]


def preprocess_image(image_path: str):
    """Loads an image, resizes/normalizes it (report 4.2 Input Design)."""
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    bgr = cv2.resize(bgr, IMG_SIZE)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    batch = np.expand_dims(rgb.astype("float32"), axis=0)  # normalization done inside model
    return bgr, batch


def build_reason(class_name: str, confidence: float, region: str) -> str:
    base_reason = DEFECT_EXPLANATIONS.get(class_name, "Visual inconsistency detected.")
    if class_name == "good":
        return f"{base_reason} (confidence: {confidence*100:.1f}%)"
    return (
        f"{base_reason} The model focused on the {region}, "
        f"with {confidence*100:.1f}% confidence."
    )


def predict(image_path: str, save_overlay: bool = True) -> dict:
    model = load_model()
    original_bgr, batch = preprocess_image(image_path)

    predictions = model.predict(batch, verbose=0)[0]
    class_index = int(np.argmax(predictions))
    confidence = float(predictions[class_index])
    class_name = CLASS_NAMES[class_index]
    is_defective = class_index != GOOD_CLASS_INDEX

    heatmap = make_gradcam_heatmap(batch, model, class_index)
    region = heatmap_region_description(heatmap)
    reason = build_reason(class_name, confidence, region)

    overlay_path = None
    if save_overlay:
        os.makedirs(RESULT_DIR, exist_ok=True)
        overlay = overlay_heatmap(original_bgr, heatmap)
        fname = os.path.splitext(os.path.basename(image_path))[0] + "_gradcam.jpg"
        overlay_path = os.path.join(RESULT_DIR, fname)
        cv2.imwrite(overlay_path, overlay)

    # Full probability breakdown, useful for the "Reports & Analytics" screen
    class_probabilities = {
        CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))
    }

    return {
        "verdict": "DEFECTIVE" if is_defective else "GOOD",
        "defect_type": None if not is_defective else class_name,
        "confidence": round(confidence, 4),
        "reason": reason,
        "class_probabilities": class_probabilities,
        "gradcam_overlay_path": overlay_path,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    result = predict(sys.argv[1])
    print("\n--- Inspection Result ---")
    print(f"Verdict     : {result['verdict']}")
    print(f"Defect Type : {result['defect_type']}")
    print(f"Confidence  : {result['confidence']*100:.2f}%")
    print(f"Reason      : {result['reason']}")
    print(f"Grad-CAM    : {result['gradcam_overlay_path']}")