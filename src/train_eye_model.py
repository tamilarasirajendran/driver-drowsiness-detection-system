# I trained a binary eye classification model using MobileNetV2. I used only Open and Closed eye images, 
# applied data augmentation,monitored training with callbacks, and saved the best model along with training graphs.
# Import Libraries
# This code trains a binary classification model (eye open vs closed) using MobileNetV2 architecture.
import os
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from model_building import build_binary_mobilenetv2

# SETTINGS
# Defines image size, batch size, number of epochs, dataset directories, and output directories
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

TRAIN_DIR = "dataset_split/train"
VAL_DIR = "dataset_split/val"

MODELS_DIR = "models"
ASSETS_DIR = "assets"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# DATA GENERATOR
# Applies data augmentation techniques: rotation, zoom, brightness adjustment, and horizontal flip for training data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

# ONLY EYE CLASSES
# Loads only the "Closed" and "Open" classes from the training and validation directories, resizes images, 
# and applies data augmentation for training data
train_data = train_datagen.flow_from_directory( #This loads only the eye classes: Closed and Open from training data
    TRAIN_DIR,
    classes=["Closed", "Open"],
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_data = val_datagen.flow_from_directory( #This loads validation images for only the two eye classes
    VAL_DIR,
    classes=["Closed", "Open"],
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# MODEL
# Builds a binary classification model using MobileNetV2 architecture, compiles it with Adam optimizer 
# and categorical crossentropy loss
model = build_binary_mobilenetv2()

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# CALLBACKS
# Stops training early if validation loss does not improve
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
# Saves the best model automatically during training
    ModelCheckpoint(
        filepath=os.path.join(MODELS_DIR, "eye_model.keras"),
        monitor="val_loss",
        save_best_only=True
    )
]

# TRAIN
# Trains the eye model using training and validation data
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# SAVE PLOTS
# Plots training and validation accuracy and loss curves, and saves the plot as an image file
plt.figure(figsize=(10,4))

# Accuracy(Plots training and validation accuracy)
plt.subplot(1,2,1)

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])

plt.title("Eye Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend(["Train","Validation"])

# Loss
plt.subplot(1,2,2)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

plt.title("Eye Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend(["Train","Validation"])

plt.tight_layout()

plt.savefig(
    os.path.join(ASSETS_DIR, "eye_model_history.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Eye model training completed.")