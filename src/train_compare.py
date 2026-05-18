# This code trains and compares two deep learning models: Custom CNN and MobileNetV2. It preprocesses the dataset, 
# applies data augmentation, trains both models,saves the best models, plots accuracy/loss graphs, 
# and compares model performance.

# Import Libraries
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

#Imports the custom CNN model and MobileNetV2 model from another file
from model_building import build_custom_cnn, build_mobilenetv2

# SETTINGS
# Defines image size, batch size, number of epochs, dataset directories, and output directories
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

TRAIN_DIR = "dataset_split/train"
VAL_DIR = "dataset_split/val"
TEST_DIR = "dataset_split/test"

MODELS_DIR = "models"
ASSETS_DIR = "assets"

# Creates folders for saving trained models and outputs if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# DATA GENERATORS
# Applies data augmentation techniques: rotation, zoom, brightness adjustment, and horizontal flip for training data
def create_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True
    )

# Validation & Test Preprocessing
# Only normalization is applied for validation and testing
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return train_gen, val_gen, test_gen

# COMPILE
# This compiles the model with Adam optimizer, categorical crossentropy loss, and accuracy metric
def compile_model(model): #Compiles the model before training.
    model.compile(
        optimizer=Adam(learning_rate=1e-4), # Uses Adam optimizer for efficient learning
        loss="categorical_crossentropy", # Used for multi-class classification
        metrics=["accuracy"] #Tracks model accuracy during training
    )
    return model

# PLOT HISTORY
# This function plots training and validation accuracy and loss curves, and saves the plot as an image file
def save_history_plot(history, title_prefix, file_prefix):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title(f"{title_prefix} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Validation"])

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title(f"{title_prefix} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train", "Validation"])

    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, f"{file_prefix}_history.png"), dpi=300, bbox_inches="tight")
    plt.close()

# TRAIN ONE MODEL
# This function takes a model, training and validation generators, and a model name, compiles the model, 
# trains it with early stopping and model checkpointing, and returns the trained model and its training history
def train_model(model, train_gen, val_gen, model_name):
    model = compile_model(model)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True), #Stops training automatically if validation loss does not improve
        ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, f"{model_name}.keras"),
            monitor="val_loss",
            save_best_only=True
        )
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    return model, history

# MAIN
# This is the main function that orchestrates the training of both models, saves their performance plots, 
# and stores a comparison of their best validation accuracy and loss in a JSON file.
def main():
    train_gen, val_gen, test_gen = create_generators()

    # 1) Custom CNN(Builds and trains the custom CNN model)
    print("\n===== Training Custom CNN =====")
    custom_cnn = build_custom_cnn(num_classes=4)
    custom_cnn, history_cnn = train_model(custom_cnn, train_gen, val_gen, "custom_cnn")
    save_history_plot(history_cnn, "Custom CNN", "custom_cnn")

    # 2) MobileNetV2(Builds and trains the MobileNetV2 model)
    print("\n===== Training MobileNetV2 =====")
    mobilenet = build_mobilenetv2(num_classes=4)
    mobilenet, history_mnet = train_model(mobilenet, train_gen, val_gen, "mobilenetv2")
    save_history_plot(history_mnet, "MobileNetV2", "mobilenetv2")

    # Save comparison results(Stores performance comparison between both models)
    results = {
        "custom_cnn": {
            "best_val_accuracy": float(max(history_cnn.history["val_accuracy"])),
            "best_val_loss": float(min(history_cnn.history["val_loss"]))
        },
        "mobilenetv2": {
            "best_val_accuracy": float(max(history_mnet.history["val_accuracy"])),
            "best_val_loss": float(min(history_mnet.history["val_loss"]))
        }
    }

    with open(os.path.join(ASSETS_DIR, "model_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n===== Comparison Result =====")
    print(json.dumps(results, indent=2))
    print("\nTraining completed for both models.")


if __name__ == "__main__":
    main()