from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization, Input
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

model = Sequential([
    Input(shape=(256, 256, 1)),
    
    Conv2D(32, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(256, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(512, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(6, activation='softmax')
])



model.compile(optimizer = Adam(learning_rate=0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 12.,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range=0.2,
                                   zoom_range = 0.15,
                                   horizontal_flip = True,
                                   brightness_range=[0.8,1.2])

val_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(
    'HandGestureDataset/train',
    target_size=(256, 256),
    color_mode='grayscale',
    batch_size=8,
    classes=['NONE', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'],
    class_mode='categorical',
    shuffle=True
)


val_set = val_datagen.flow_from_directory('HandGestureDataset/train',
                                                 target_size = (256, 256),
                                                 color_mode = 'grayscale',
                                                 batch_size = 8,
                                                 classes = ['NONE', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'],
                                                 class_mode = 'categorical')

class_labels = np.unique(training_set.classes)
class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=training_set.classes)
class_weight_dict = dict(enumerate(class_weights))


callback_list = [
    EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights=True),
    ModelCheckpoint(filepath="model.weights.h5", monitor = 'val_loss', save_best_only = True, verbose = 1)]

model.fit(
    training_set,
    steps_per_epoch=len(training_set),
    epochs=15,
    validation_data=val_set,
    validation_steps=len(val_set),
    class_weight=class_weight_dict,
    callbacks=callback_list
)

#Model Save
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.weights.h5")
print("Saved model to disk")
