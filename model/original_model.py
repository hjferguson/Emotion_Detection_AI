# Harlan Ferguson 101133838

# Initial notes: Ran model for the first time. 10 epoch. Got 83% acc on train set. But 51% accuracy on test set.
# This is a sign of overfitting.
# Second round, I introduced a validation set. I also added early stopping to prevent overfitting. No improvement.
# Third round, implementing data augmentation. No improvement. 
#4th attempt, a simpler model. No improvement.
#5th attempt, increased complexity of model. No improvement.
#6th attempt, used a model with 4 convolutional layers, slight improvement.
#7. adding rescaling to the new model improves accuracy slightly. Probably because 48x48 pretty pixelated whereas 255 is more clear.
#8. after multiple tests, attempting to augment the data always resulted in a lower accuracy. 
#I removed "disgusted" from the dataset because it was underrepresented. This improved accuracy slightly
#With hours of different attempts, I'm going to settle at 57-59%. From research, it looks like experts can only get to around 75% with this dataset.

#ok its not an issue with the input image, because I can put an obviously happy face infront of the camera
#and still get an awful prediciton. Happy should be index 2, but im only getting index 0.
#Raw prediction: [[0.35343826 0.06493568 0.07987152 0.14573923 0.34574193 0.0102734 ]]

import os
import tensorflow as tf
from keras import layers, models, callbacks, optimizers
import matplotlib.pyplot as plt

data_dir = os.path.join(os.pardir, 'data', 'archive')
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

BATCH_SIZE = 32
IMG_HEIGHT = 48
IMG_WIDTH = 48

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


#model without aug, recaling, 55% acc
#adding rescaling brought accuracy up to 59%
model = models.Sequential([
    
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.1),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.1),
    
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.1),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),

    layers.Dense(6, activation='softmax')
])

model.compile(optimizer=optimizers.RMSprop(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#if validation set isn't improving, stop training
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

epochs = 15
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[early_stopping]
)

test_loss, test_acc = model.evaluate(test_ds)
print('\nTest accuracy:', test_acc)

model.save('model.keras')

plt.figure(figsize=(10, 4))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('training_validation_accuracy.jpg')  
#plt.show()  

plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_validation_loss.jpg') 
#plt.show()  

plt.figure(figsize=(10, 4))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['loss'], label='Training Loss', linestyle='--')
plt.title('Training Accuracy vs. Training Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('training_accuracy_vs_loss.jpg')  
#plt.show()  

plt.figure(figsize=(10, 4))
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
plt.title('Validation Accuracy vs. Validation Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('validation_accuracy_vs_loss.jpg') 
#plt.show() 
