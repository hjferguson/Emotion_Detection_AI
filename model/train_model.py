# Harlan Ferguson 101133838
# Initial notes: Ran model for the first time. 10 epoch. Got 83% acc on train set. But 51% accuracy on test set.
# This is a sign of overfitting.
# Second round, I introduced a validation set. I also added early stopping to prevent overfitting. No improvement.
# Third round, implementing data augmentation. No improvement. 

import os
import tensorflow as tf
from keras import layers, models, callbacks

data_dir = os.path.join(os.pardir, 'data', 'archive')
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

BATCH_SIZE = 32
IMG_HEIGHT = 48
IMG_WIDTH = 48

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    label_mode='categorical'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    label_mode='categorical'
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    label_mode='categorical'
)

AUTOTUNE = tf.data.AUTOTUNE #this is used to optimize loading the data into the model.

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = models.Sequential([
    data_augmentation,  # Adding data augmentation to the model
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(7, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#if validation set isn't improving, stop training

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

epochs = 40
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[early_stopping]
)

test_loss, test_acc = model.evaluate(test_ds)
print('\nTest accuracy:', test_acc)
