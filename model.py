"""
model.py
Defines the CNN (transfer-learning MobileNetV2) used for defect classification,
as described in Chapter 4.4 (Algorithm) of the project report.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from config import IMG_SIZE, CLASS_NAMES


def build_model(num_classes: int = len(CLASS_NAMES), fine_tune: bool = False) -> tf.keras.Model:
    """
    Builds a CNN classifier on top of a MobileNetV2 backbone (pretrained on ImageNet).

    Layers match the report's described pipeline:
      Convolution (feature extraction, via MobileNetV2) -> Pooling -> Fully Connected
      -> ReLU (hidden dense layer) -> Softmax (output layer)
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = fine_tune  # frozen by default for faster/first-pass training

    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=fine_tune)
    x = layers.GlobalAveragePooling2D(name="gap")(x)          # Pooling layer
    x = layers.Dense(128, activation="relu")(x)                # Fully connected + ReLU
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)  # Softmax

    model = models.Model(inputs, outputs, name="retail_vqc_cnn")
    return model, base_model


def compile_model(model: tf.keras.Model, lr: float):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),  # Adam optimizer (report 4.4.7)
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model