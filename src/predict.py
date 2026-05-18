# This code loads a trained model, takes one test image, predicts the class, calculates confidence, 
# converts the prediction into a fatigue stage, and prints the final driver state.

# import necessary libraries for model loading, image processing, and fatigue logic
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from fatigue_logic import get_fatigue

# SETTINGS
# Defines the path to the trained model, the image to predict, and class names
MODEL_PATH = "models/mobilenetv2.keras"   # or "models/model.keras"
IMAGE_PATH = r"D:\Tamil Files\Project\driver_drowsiness\dataset\Open\_23.jpg"

CLASS_NAMES = ["Closed", "Open", "no_yawn", "yawn"]

# LOAD MODEL
# This loads the trained MobileNetV2 model from the saved file
model = load_model(MODEL_PATH)

# LOAD IMAGE
# This loads the image, resizes it to 224x224, converts it to an array, normalizes pixel values, and adds a batch dimension
img = image.load_img(IMAGE_PATH, target_size=(224, 224))
img_array = image.img_to_array(img) #converts it into an array
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0) #adds batch dimension for prediction

# PREDICT
# This gives the prediction probabilities for the image
prediction = model.predict(img_array, verbose=0)

# This selects the class with the highest probability
pred_index = np.argmax(prediction)
predicted_label = CLASS_NAMES[pred_index] #converts the predicted index into the class label
confidence = prediction[0][pred_index] * 100 #calculates the confidence percentage of the prediction

# FATIGUE LOGIC
# This converts the predicted label into a fatigue stage using the get_fatigue function
stage_id, stage_name = get_fatigue(predicted_label)

# OUTPUT
# This prints the predicted class label, confidence percentage, fatigue level ID, and fatigue stage name
print("Model Prediction:", predicted_label)
print(f"Confidence: {confidence:.2f}%")
print("Fatigue Level ID:", stage_id)
print("Driver State:", stage_name)