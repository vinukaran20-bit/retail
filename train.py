"""
train.py
Trains the CNN on your product image dataset.

Expected folder layout (already created for you under dataset/train/):
    dataset/train/good/*.jpg
    dataset/train/crack/*.jpg
    dataset/train/scratch/*.jpg
    dataset/train/dent/*.jpg
    dataset/train/discoloration/*.jpg
    dataset/train/missing_label/*.jpg
    dataset/train/damaged_packaging/*.jpg

Drop your labeled images into these folders (rename/add folders in config.py
if your defect categories differ), then run:

    python train.py
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from config import (
    DATASET_DIR, VAL_DIR, MODEL_DIR, MODEL_PATH,
    IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, CLASS_NAMES,
)
from model import build_model, compile_model


def get_generators():
    # Preprocessing / Data Augmentation (report 4.4.2): resize, normalize, rotate, flip, zoom
    train_datagen = ImageDataGenerator(
        rescale=None,  # normalization is done inside the model (preprocess_input)
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=(0.85, 1.15),
        validation_split=0.0 if os.path.isdir(VAL_DIR) and os.listdir(VAL_DIR) else 0.2,
    )

    common_kwargs = dict(
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
    )

    if os.path.isdir(VAL_DIR) and os.listdir(VAL_DIR):
        train_gen = train_datagen.flow_from_directory(DATASET_DIR, subset=None, **common_kwargs)
        val_datagen = ImageDataGenerator()
        val_gen = val_datagen.flow_from_directory(VAL_DIR, **common_kwargs)
    else:
        train_gen = train_datagen.flow_from_directory(DATASET_DIR, subset="training", **common_kwargs)
        val_gen = train_datagen.flow_from_directory(DATASET_DIR, subset="validation", **common_kwargs)

    return train_gen, val_gen


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_gen, val_gen = get_generators()

    if train_gen.samples == 0:
        raise RuntimeError(
            "No training images found. Add images under dataset/train/<class_name>/ "
            f"for each class in {CLASS_NAMES} before running train.py."
        )

    model, base_model = build_model()
    compile_model(model, LEARNING_RATE)
    model.summary()

    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]

    # Stage 1: train the classification head with the backbone frozen
    h1=model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Stage 2 (optional fine-tuning): unfreeze the backbone and train a few more epochs
    # at a lower learning rate for higher accuracy once you have enough data (100+ images/class).
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    compile_model(model, LEARNING_RATE / 10)

   
    h2=model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=4,
        callbacks=callbacks,
    )

    
    import json
    history = {k: [float(v) for v in h1.history[k] + h2.history.get(k, [])] for k in h1.history}
    with open(os.path.join(MODEL_DIR, "history.json"), "w") as fh:
        json.dump(history, fh)


    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()