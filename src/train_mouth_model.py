
import os
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from model_building import build_binary_mobilenetv2

# =====================================================
# SETTINGS
# =====================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

TRAIN_DIR = "dataset_split/train"
VAL_DIR = "dataset_split/val"

MODELS_DIR = "models"
ASSETS_DIR = "assets"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# =====================================================
# DATA GENERATOR
# =====================================================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

# =====================================================
# ONLY MOUTH CLASSES
# =====================================================
train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    classes=["no_yawn", "yawn"],
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    classes=["no_yawn", "yawn"],
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# =====================================================
# MODEL
# =====================================================
model = build_binary_mobilenetv2()

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =====================================================
# CALLBACKS
# =====================================================
callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),

    ModelCheckpoint(
        filepath=os.path.join(MODELS_DIR, "mouth_model.keras"),
        monitor="val_loss",
        save_best_only=True
    )
]

# =====================================================
# TRAIN
# =====================================================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =====================================================
# SAVE PLOTS
# =====================================================
plt.figure(figsize=(10,4))

# Accuracy
plt.subplot(1,2,1)

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])

plt.title("Mouth Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend(["Train","Validation"])

# Loss
plt.subplot(1,2,2)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

plt.title("Mouth Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend(["Train","Validation"])

plt.tight_layout()

plt.savefig(
    os.path.join(ASSETS_DIR, "mouth_model_history.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Mouth model training completed.")