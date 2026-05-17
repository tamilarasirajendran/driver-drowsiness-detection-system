
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# =====================================================
# SETTINGS
# =====================================================
MODEL_PATH = "models/mobilenetv2.keras"   # change if needed
TEST_DIR = "dataset_split/test"
ASSETS_DIR = "assets"

CLASS_NAMES = ["Closed", "Open", "no_yawn", "yawn"]
FRAMES_PER_INTERVAL = 10

os.makedirs(ASSETS_DIR, exist_ok=True)

# =====================================================
# FATIGUE MAPPING
# =====================================================
def get_fatigue_level(label):
    if label in ["Open", "no_yawn"]:
        return 0  # Alert
    elif label == "yawn":
        return 1  # Mild Fatigue
    elif label == "Closed":
        return 2  # Severe Fatigue
    else:
        return -1

# =====================================================
# LOAD MODEL
# =====================================================
model = load_model(MODEL_PATH)

# =====================================================
# COLLECT SEQUENTIAL PREDICTIONS
# =====================================================
fatigue_levels = []

for class_folder in os.listdir(TEST_DIR):
    folder_path = os.path.join(TEST_DIR, class_folder)

    if not os.path.isdir(folder_path):
        continue

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        try:
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_array, verbose=0)
            pred_index = np.argmax(pred)
            predicted_label = CLASS_NAMES[pred_index]

            stage = get_fatigue_level(predicted_label)
            if stage != -1:
                fatigue_levels.append(stage)

        except Exception as e:
            print(f"Skipping {img_path} -> {e}")

# =====================================================
# GROUP INTO TIME INTERVALS
# =====================================================
fatigue_over_time = []

for i in range(0, len(fatigue_levels), FRAMES_PER_INTERVAL):
    interval = fatigue_levels[i:i + FRAMES_PER_INTERVAL]

    if len(interval) > 0:
        avg_fatigue = np.mean(interval)
        fatigue_over_time.append(avg_fatigue)

# =====================================================
# PLOT FATIGUE PROGRESSION
# =====================================================
time_axis = list(range(len(fatigue_over_time)))

plt.figure(figsize=(10, 5))
plt.plot(time_axis, fatigue_over_time, marker='o')

plt.title("Driver Fatigue Progression Curve")
plt.xlabel("Time Interval")
plt.ylabel("Fatigue Level")
plt.yticks([0, 1, 2], ["Alert", "Mild Fatigue", "Severe Fatigue"])
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "fatigue_progression.png"), dpi=300, bbox_inches="tight")
plt.show()

# =====================================================
# FIND TRANSITION POINTS
# =====================================================
print("\n===== Fatigue Transition Analysis =====")

for i in range(1, len(fatigue_over_time)):
    prev_level = fatigue_over_time[i - 1]
    curr_level = fatigue_over_time[i]

    if curr_level > prev_level:
        print(f"Fatigue increased at interval {i}: {prev_level:.2f} -> {curr_level:.2f}")

print("\nFatigue progression curve saved to assets/fatigue_progression.png")