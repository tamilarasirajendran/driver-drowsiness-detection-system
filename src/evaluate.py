
# Import Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model #already trained model
from tensorflow.keras.preprocessing.image import ImageDataGenerator #for image preprocess

#Used for evaluation metrics(prediction correctness check and precision, recall, f1-score calculate)
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# SETTINGS
#Saved trained model location.

MODEL_PATH = "models/mobilenetv2.keras"

TEST_DIR = "dataset_split/test" #Test dataset path

ASSETS_DIR = "assets" # Folder to save output files 

IMG_SIZE = (224, 224) # Resizes all images into 224x224 size
BATCH_SIZE = 32 #Processes 32 images at one time

os.makedirs(ASSETS_DIR, exist_ok=True)


# LOAD MODEL
# Loads the trained MobileNetV2 model
model = load_model(MODEL_PATH)


# TEST DATA
# Normalizes image pixel values
test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

#Loads test images from directory.
test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)


# EVALUATE
# Evaluates model performance
loss, accuracy = model.evaluate(
    test_data,
    verbose=1
)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%") #Prints test accuracy
print(f"Test Loss: {loss:.4f}") #Prints model loss


# PREDICTIONS
# Generates predictions for test images
predictions = model.predict(
    test_data,
    verbose=1
)

#Gets predicted class label
y_pred = np.argmax(predictions, axis=1)

#Gets actual class labels
y_true = test_data.classes

#Gets class names
class_names = list(
    test_data.class_indices.keys()
)


# CONFUSION MATRIX
# Creates confusion matrix
cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(8,6)) #Sets graph size

#Displays confusion matrix heatmap
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label") #Sets x-axis label
plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(
        ASSETS_DIR,
        "confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# CLASSIFICATION REPORT
# Generates classification report
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names
)

print("\nClassification Report:\n")

print(report)

# Save report
with open(
    os.path.join(
        ASSETS_DIR,
        "classification_report.txt"
    ),
    "w"
) as f:

    f.write(report)

print("\nEvaluation completed successfully.")