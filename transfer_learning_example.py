import os

import pandas as pd
from icecream import ic
from keras import layers
from keras import models
from keras.applications import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator

# Define paths
csv_path = "./bald_people.csv"
output_model_path = "transfer_learning_model.keras"
images_path = "./images/"
num_classes = 6

# Read CSV file
df = pd.read_csv(csv_path)

df["images"] = df["images"].str.replace("images/", "")

# Create ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode="nearest",
)

# Create data generators
train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory='./images/',
    x_col="images",
    y_col="type",
    target_size=(224, 224),
    batch_size=42,
    class_mode="categorical",
    subset="training",
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory='./images/',
    x_col="images",
    y_col="type",
    target_size=(224, 224),
    batch_size=42,
    class_mode="categorical",
    subset="validation",
)

# Load pre-trained MobileNetV2 model
base_model = MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3), classes=num_classes
)

# Freeze the weights of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Create a custom model for classification
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(num_classes, activation="softmax"))

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
)

ic(history.history["accuracy"][-1])

# Save the model
model.save(output_model_path)
