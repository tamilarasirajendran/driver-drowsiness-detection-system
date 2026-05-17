
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# =====================================================
# SETTINGS
# =====================================================
MODEL_PATH = "models/mobilenetv2.keras"   # change if needed
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
# LOAD TEST DATA
# =====================================================
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# =====================================================
# PREDICTIONS
# =====================================================
predictions = model.predict(test_data, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes
class_names = list(test_data.class_indices.keys())

# =====================================================
# CLASSIFICATION REPORT
# =====================================================
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names
)

print("\n===== Classification Report =====\n")
print(report)

with open(os.path.join(ASSETS_DIR, "performance_report.txt"), "w") as f:
    f.write(report)

# =====================================================
# CONFUSION MATRIX
# =====================================================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
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
plt.title("Performance Analysis - Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "performance_confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.show()

# =====================================================
# CLASS-WISE ACCURACY
# =====================================================
class_correct = {cls: 0 for cls in class_names}
class_total = {cls: 0 for cls in class_names}

for i, true_label in enumerate(y_true):
    true_class = class_names[true_label]
    class_total[true_class] += 1
    if y_pred[i] == true_label:
        class_correct[true_class] += 1

class_acc = {}
for cls in class_names:
    if class_total[cls] > 0:
        class_acc[cls] = class_correct[cls] / class_total[cls]
    else:
        class_acc[cls] = 0

print("\n===== Class-wise Accuracy =====\n")
for cls, acc in class_acc.items():
    print(f"{cls}: {acc * 100:.2f}%")

# Bar plot
plt.figure(figsize=(8, 5))
plt.bar(class_acc.keys(), class_acc.values())
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Class-wise Accuracy Comparison")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "classwise_accuracy.png"), dpi=300, bbox_inches="tight")
plt.show()

# =====================================================
# ERROR CASES
# =====================================================
errors = np.where(y_pred != y_true)[0]

print("\n===== Error Cases =====\n")
print(f"Total errors: {len(errors)}")

for i in errors[:5]:
    img_path = test_data.filepaths[i]
    actual = class_names[y_true[i]]
    predicted = class_names[y_pred[i]]
    print(f"Actual: {actual} | Predicted: {predicted} | Image: {img_path}")

print("\nPerformance analysis completed successfully.")