"""
gradcam.py
Grad-CAM implementation: produces a heatmap showing WHICH region of the image
drove the model's decision. This is the visual "reason" behind a defective
verdict (report Chapter 4: Result Analysis -> defect type + confidence).
"""

import numpy as np
import tensorflow as tf
import cv2


def find_last_conv_layer(model: tf.keras.Model):
    """Locates the last convolutional layer inside the MobileNetV2 backbone."""
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            backbone = layer
            break
    if backbone is None:
        raise ValueError("Could not locate the backbone sub-model for Grad-CAM.")

    conv_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.SeparableConv2D,
        tf.keras.layers.DepthwiseConv2D,
    )
    for layer in reversed(backbone.layers):
        if isinstance(layer, conv_types):
            return backbone.name, layer.name

    raise ValueError("No convolutional layer found for Grad-CAM.")


def make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model, class_index: int) -> np.ndarray:
    """
    img_array: RAW batch of shape (1, H, W, 3), NOT yet preprocessed.
    Returns a normalized heatmap (H, W) in range [0, 1].

    Works around Keras 3's stricter graph-connectivity rules by building a
    small model from the backbone's OWN input (fully self-contained), then
    manually replaying the remaining top layers (gap -> dense -> predictions)
    as plain eager calls inside the same GradientTape.
    """
    backbone_name, conv_layer_name = find_last_conv_layer(model)
    backbone = model.get_layer(backbone_name)

    # Fully self-contained sub-model: backbone's own input -> (conv output, backbone output)
    grad_backbone = tf.keras.models.Model(
        inputs=backbone.input,
        outputs=[backbone.get_layer(conv_layer_name).output, backbone.output],
    )

    gap_layer = model.get_layer("gap")
    dense_layer = model.get_layer("dense")
    predictions_layer = model.get_layer("predictions")

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_tensor)
        conv_output, backbone_output = grad_backbone(preprocessed)
        tape.watch(conv_output)
        x = gap_layer(backbone_output)
        x = dense_layer(x)
        predictions = predictions_layer(x)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(original_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlays the Grad-CAM heatmap on the original image (BGR, uint8)."""
    heatmap_resized = cv2.resize(heatmap, (original_bgr.shape[1], original_bgr.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, alpha, original_bgr, 1 - alpha, 0)
    return overlay


def heatmap_region_description(heatmap: np.ndarray) -> str:
    """Turns the peak location of the heatmap into a plain-language region description."""
    h, w = heatmap.shape
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    vertical = "top" if y < h / 3 else ("bottom" if y > 2 * h / 3 else "middle")
    horizontal = "left" if x < w / 3 else ("right" if x > 2 * w / 3 else "center")
    if vertical == "middle" and horizontal == "center":
        return "center of the product"
    return f"{vertical}-{horizontal} area of the product"