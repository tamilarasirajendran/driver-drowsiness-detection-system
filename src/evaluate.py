
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# =====================================================
# SETTINGS
# =====================================================
MODEL_PATH = "models/mobilenetv2.keras"

TEST_DIR = "dataset_split/test"

ASSETS_DIR = "assets"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

os.makedirs(ASSETS_DIR, exist_ok=True)

# =====================================================
# LOAD MODEL
# =====================================================
model = load_model(MODEL_PATH)

# =====================================================
# TEST DATA
# =====================================================
test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# =====================================================
# EVALUATE
# =====================================================
loss, accuracy = model.evaluate(
    test_data,
    verbose=1
)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

# =====================================================
# PREDICTIONS
# =====================================================
predictions = model.predict(
    test_data,
    verbose=1
)

y_pred = np.argmax(predictions, axis=1)

y_true = test_data.classes

class_names = list(
    test_data.class_indices.keys()
)

# =====================================================
# CONFUSION MATRIX
# =====================================================
cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
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

# =====================================================
# CLASSIFICATION REPORT
# =====================================================
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