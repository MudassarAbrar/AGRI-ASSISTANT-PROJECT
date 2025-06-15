import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set paths
DATA_DIR = r"C:\Users\Mudassir\OneDrive\Desktop\AGRI-ASSISTANT\data\PlantVillage"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ✅ Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# ✅ No augmentation for validation, only rescale
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# ✅ Training data
train_data = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# ✅ Validation data
val_data = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# ✅ Number of classes
num_classes = len(train_data.class_indices)

# ✅ Optional: save class labels to use later in prediction app
import json
with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)

print("✅ Data loaded successfully.")
print("📦 Classes:", train_data.class_indices)
print("🔢 Number of classes:", num_classes)
