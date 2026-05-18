
# I loaded the trained model, predicted each test image, converted predictions into fatigue levels, 
# grouped them over time, plotted a fatigue progression curve, and detected increases in fatigue

#I imported libraries for file handling, numerical operations, plotting, 
# model loading, and image preprocessing
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# I defined the model path, test dataset folder, and output folder
MODEL_PATH = "models/mobilenetv2.keras"   # change if needed
TEST_DIR = "dataset_split/test"
ASSETS_DIR = "assets"

# I defined the class labels used by the model and the number of frames per time interval
CLASS_NAMES = ["Closed", "Open", "no_yawn", "yawn"]
FRAMES_PER_INTERVAL = 10

# I created the assets folder if it does not already exist
os.makedirs(ASSETS_DIR, exist_ok=True)

# FATIGUE MAPPING
# I created a function to convert prediction labels into fatigue levels
def get_fatigue_level(label):
    
    if label in ["Open", "no_yawn"]: # If the label is Open or no_yawn, I return 0, which means Alert
        return 0  # Alert
    elif label == "yawn":  # If the label is yawn, I return 1, which means Mild Fatigue
        return 1  # Mild Fatigue
    elif label == "Closed":  #If the label is Closed, I return 2, which means Severe Fatigue
        return 2  # Severe Fatigue 
    else:
        return -1 

# LOAD MODEL
# I loaded the trained MobileNetV2 model from the saved file
model = load_model(MODEL_PATH)

# COLLECT SEQUENTIAL PREDICTIONS
# I created a list to store fatigue levels
fatigue_levels = []

for class_folder in os.listdir(TEST_DIR):  # I looped through each folder inside the test directory
    folder_path = os.path.join(TEST_DIR, class_folder) # I built the full path for the current folder.

    if not os.path.isdir(folder_path):  #I skipped anything that is not a folder
        continue

    for img_name in os.listdir(folder_path):  #I looped through each image in that folder
        img_path = os.path.join(folder_path, img_name) #I created the full image path

        try:
            img = image.load_img(img_path, target_size=(224, 224)) #I loaded the image and resized it to 224x224
            img_array = image.img_to_array(img) / 255.0 # I converted the image to an array and normalized pixel values
            img_array = np.expand_dims(img_array, axis=0) # I added a batch dimension for model prediction

            pred = model.predict(img_array, verbose=0) # I predicted the class for the image using the model
            pred_index = np.argmax(pred)
            predicted_label = CLASS_NAMES[pred_index] # I converted the prediction index into the class label

            stage = get_fatigue_level(predicted_label) # I converted the predicted label into a fatigue stage
            if stage != -1:
                fatigue_levels.append(stage) # I stored only valid fatigue levels

        except Exception as e:
            print(f"Skipping {img_path} -> {e}") # I skipped images that caused errors


# GROUP INTO TIME INTERVALS
# I created a list to store average fatigue values over time
fatigue_over_time = []

# I grouped fatigue values into chunks of 10 frames
for i in range(0, len(fatigue_levels), FRAMES_PER_INTERVAL):
    interval = fatigue_levels[i:i + FRAMES_PER_INTERVAL]  # I took one interval of fatigue values

    if len(interval) > 0:  # I calculated the average fatigue for each interval and stored it
        avg_fatigue = np.mean(interval)
        fatigue_over_time.append(avg_fatigue)


# PLOT FATIGUE PROGRESSION
# I created x-axis values for the graph
time_axis = list(range(len(fatigue_over_time)))

plt.figure(figsize=(10, 5)) # I set the figure size
plt.plot(time_axis, fatigue_over_time, marker='o')

plt.title("Driver Fatigue Progression Curve") 
plt.xlabel("Time Interval")
plt.ylabel("Fatigue Level")
plt.yticks([0, 1, 2], ["Alert", "Mild Fatigue", "Severe Fatigue"])
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "fatigue_progression.png"), dpi=300, bbox_inches="tight")
plt.show()


# FIND TRANSITION POINTS
# I analyzed the fatigue progression curve to find points where fatigue increased from one interval to the next
print("\n===== Fatigue Transition Analysis =====")

for i in range(1, len(fatigue_over_time)): #I checked fatigue changes from one interval to the next.
    prev_level = fatigue_over_time[i - 1]
    curr_level = fatigue_over_time[i]

    if curr_level > prev_level:
        print(f"Fatigue increased at interval {i}: {prev_level:.2f} -> {curr_level:.2f}")

print("\nFatigue progression curve saved to assets/fatigue_progression.png")