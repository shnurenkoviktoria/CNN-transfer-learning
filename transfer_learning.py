import os
import pandas as pd
from keras import layers, models
from keras.applications import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator

# Define paths
CSV_PATH = "./bald_people.csv"
OUTPUT_MODEL_PATH = "transfer_learning_model.keras"
IMAGES_PATH = "./images/"
NUM_CLASSES = 6
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 42
EPOCHS = 20

# Read CSV file
df = pd.read_csv(CSV_PATH)
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
def create_data_generator(subset):
    return datagen.flow_from_dataframe(
        dataframe=df,
        directory=IMAGES_PATH,
        x_col="images",
        y_col="type",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset=subset,
    )


train_generator = create_data_generator("training")
validation_generator = create_data_generator("validation")

# Load pre-trained MobileNetV2 model
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
    classes=NUM_CLASSES,
)

# Freeze the weights of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Create a custom model for classification
model = models.Sequential(
    [
        base_model,
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
)

print(history.history["accuracy"][-1])

# Save the model
model.save("transfer_learning_model.keras")
