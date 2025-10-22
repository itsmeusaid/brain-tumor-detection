# %% [markdown]
# **Imports**

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report

# %% [markdown]
# **Paths & Parameters**o

# %%
train_dir = r"D:\Intern base\Brain\Training"
test_dir  = r"D:\Intern base\Brain\Testing"
img_width, img_height = 224, 224
batch_size = 32
num_epochs = 30
num_classes = 4
model_path = 'my_brain_tumor_mobilenetv2.h5'
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# %% [markdown]
# **Data Generators with Augmentation**

# %%
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.25,
    width_shift_range=0.18,
    height_shift_range=0.18,
    shear_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.2]
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='categorical')
test_data = test_datagen.flow_from_directory(
    test_dir, target_size=(img_width, img_height), batch_size=batch_size,
    class_mode='categorical', shuffle=False)

# %% [markdown]
# **Model: MobileNetV2 Transfer Learning**

# %%
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(img_width, img_height, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# %% [markdown]
# **Callbacks and Training**

# %%
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
model_ckpt = ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True)
history = model.fit(
    train_data, epochs=num_epochs, validation_data=test_data,
    callbacks=[early_stop, reduce_lr, model_ckpt]
)

# %% [markdown]
# **Evaluation & Visuals**

# %%
loss, accuracy = model.evaluate(test_data)
print(f'Test Accuracy: {accuracy*100:.2f}%')
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Accuracy')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Loss')
plt.legend()
plt.show()

Y_pred = model.predict(test_data)
y_pred = np.argmax(Y_pred, axis=1)
print("Classification Report")
print(classification_report(test_data.classes, y_pred, target_names=test_data.class_indices.keys()))