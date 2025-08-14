import tensorflow as tf
from prepare_data import create_datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load data
train_ds, val_ds, test_ds, class_names = create_datasets()
NUM_CLASSES = len(class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# Build model
def build_model(num_classes: int, fine_tune_at: int = 100):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])

    inputs = tf.keras.layers.Input(shape=(48, 48, 1))
    x = tf.keras.layers.Rescaling(1. / 255)(inputs) 
    x = data_augmentation(x)

    x = tf.keras.layers.Conv2D(3, (3, 3), padding="same")(x)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(48, 48, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # freeze at first stage

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.fine_tune_at = fine_tune_at  
    model.base_model = base_model
    return model


model = build_model(NUM_CLASSES)

# Phase‑1 training 
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("emotion_best.keras", save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6),
]

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

initial_epochs = 15
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=initial_epochs,
    callbacks=callbacks,
)

# Fine‑tuning
model.base_model.trainable = True
for layer in model.base_model.layers[: model.fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

fine_tune_epochs = 15
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=initial_epochs + fine_tune_epochs,
    initial_epoch=history.epoch[-1],
    callbacks=callbacks,
)

# Save final model
model.save("emotion_model_v3_finetuned.keras")
print("\nModel saved to emotion_model_v3_finetuned.keras")

# Evaluate 
print("\nEvaluating on hold‑out test set …")
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.3f}")

print("Generating confusion matrix …")
y_true = np.concatenate([y.numpy() for _, y in test_ds])
y_pred = np.argmax(model.predict(test_ds), axis=1)
cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()

# Plot learning curves
acc = history.history["accuracy"] + history_fine.history["accuracy"]
val_acc = history.history["val_accuracy"] + history_fine.history["val_accuracy"]
loss = history.history["loss"] + history_fine.history["loss"]
val_loss = history.history["val_loss"] + history_fine.history["val_loss"]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(acc, label="Train Acc")
plt.plot(val_acc, label="Val Acc")
plt.axvline(initial_epochs - 1, color="gray", linestyle="--")
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(loss, label="Train Loss")
plt.plot(val_loss, label="Val Loss")
plt.axvline(initial_epochs - 1, color="gray", linestyle="--")
plt.legend()
plt.title("Loss")
plt.show()
