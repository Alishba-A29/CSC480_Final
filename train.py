import tensorflow as tf
from prepare_data import create_datasets
import matplotlib.pyplot as plt

train_ds, val_ds, test_ds, class_names = create_datasets()
NUM_CLASSES = len(class_names)

def build_model(num_classes):
    """
    Construct a model based on MobileNetV2.
    """
    inputs = tf.keras.layers.Input(shape=(48, 48, 1))

    x = tf.keras.layers.Conv2D(3, (3, 3), padding='same')(inputs)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(48, 48, 3),
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False

    x = base_model(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x) 
    x = tf.keras.layers.Dropout(0.2)(x)            

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    
    return model

model = build_model(NUM_CLASSES)
model.summary()


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),   
    metrics=['accuracy']                                 
)



EPOCHS = 10 

print("\n--- Starting Training ---")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)
print("--- Training Finished ---")


model.save('emotion_model.keras')
print("\nModel saved to emotion_model.keras")


# --- visualization ---
acc = history.history['accuracy']
val_acc = history.history['validation_accuracy']
loss = history.history['loss']
val_loss = history.history['validation_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()