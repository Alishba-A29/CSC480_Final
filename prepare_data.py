import tensorflow as tf
import os

IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 32

def create_datasets(data_dir='data/archive'):
    print("Creating datasets from image folders...")

    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,   
        subset="training",    
        seed=123,              
        image_size=(IMG_HEIGHT, IMG_WIDTH), 
        batch_size=BATCH_SIZE,
        color_mode='grayscale' 
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir, 
        validation_split=0.2,
        subset="validation",   
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        color_mode='grayscale'
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        color_mode='grayscale'
    )
    class_names = train_dataset.class_names
    print(f"Class names found: {class_names}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    
    print("Datasets created and optimized successfully!")

    return train_dataset, validation_dataset, test_dataset, class_names

if __name__ == '__main__':
    train_ds, val_ds, test_ds, classes = create_datasets()
    print(f"\nTraining dataset: {train_ds}")
    print(f"Validation dataset: {val_ds}")
    print(f"Test dataset: {test_ds}")