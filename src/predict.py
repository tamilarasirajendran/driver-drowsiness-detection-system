
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from fatigue_logic import get_fatigue

# =====================================================
# SETTINGS
# =====================================================
MODEL_PATH = "models/mobilenetv2.keras"   # or "models/model.keras"
IMAGE_PATH = r"D:\Tamil Files\Project\driver_drowsiness\dataset\Open\_23.jpg"

CLASS_NAMES = ["Closed", "Open", "no_yawn", "yawn"]

# =====================================================
# LOAD MODEL
# =====================================================
model = load_model(MODEL_PATH)

# =====================================================
# LOAD IMAGE
# =====================================================
img = image.load_img(IMAGE_PATH, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# =====================================================
# PREDICT
# =====================================================
prediction = model.predict(img_array, verbose=0)

pred_index = np.argmax(prediction)
predicted_label = CLASS_NAMES[pred_index]
confidence = prediction[0][pred_index] * 100

# =====================================================
# FATIGUE LOGIC
# =====================================================
stage_id, stage_name = get_fatigue(predicted_label)

# =====================================================
# OUTPUT
# =====================================================
print("Model Prediction:", predicted_label)
print(f"Confidence: {confidence:.2f}%")
print("Fatigue Level ID:", stage_id)
print("Driver State:", stage_name)