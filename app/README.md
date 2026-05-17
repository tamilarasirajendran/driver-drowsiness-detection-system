
# 🚗 Driver Drowsiness Detection System

## 📌 Project Overview

This project is a deep learning-based Driver Drowsiness Detection System developed using TensorFlow, Keras, OpenCV, and Streamlit.

The system detects driver fatigue by analyzing:

* Eye closure
* Yawning activity

The model predicts four classes:

* Closed
* Open
* no_yawn
* yawn

The predicted result is then converted into real-world fatigue stages:

* Alert
* Mild Fatigue
* Severe Fatigue

---

# 🎯 Objectives

The main objectives of this project are:

* Detect driver drowsiness using deep learning
* Compare Custom CNN and MobileNetV2 models
* Build separate eye-state and mouth-state models
* Perform fatigue progression analysis
* Create a real-time Streamlit application

---

# 🛠 Technologies Used

| Technology         | Purpose              |
| ------------------ | -------------------- |
| Python             | Programming Language |
| TensorFlow / Keras | Deep Learning        |
| OpenCV             | Webcam Processing    |
| NumPy              | Numerical Operations |
| Matplotlib         | Visualization        |
| Seaborn            | Confusion Matrix     |
| Streamlit          | Web Application      |
| Scikit-learn       | Evaluation Metrics   |

---

# 📂 Project Structure

```text
driver_drowsiness/
│
├── app/
│   └── streamlit_app.py
│
├── assets/
│   ├── confusion_matrix.png
│   ├── fatigue_progression.png
│   ├── classwise_accuracy.png
│   └── performance_report.txt
│
├── dataset/
│   ├── Closed/
│   ├── Open/
│   ├── no_yawn/
│   └── yawn/
│
├── dataset_split/
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
│   ├── custom_cnn.keras
│   ├── mobilenetv2.keras
│   ├── eye_model.keras
│   └── mouth_model.keras
│
├── src/
│   ├── data_preprocessing.py
│   ├── model_building.py
│   ├── train_compare.py
│   ├── train_eye_model.py
│   ├── train_mouth_model.py
│   ├── evaluate.py
│   ├── fatigue_logic.py
│   ├── predict.py
│   ├── fatigue_progression.py
│   └── performance_analysis.py
│
├── requirements.txt
└── README.md
```

---

# 📊 Dataset

The dataset contains four classes:

| Class   | Description         |
| ------- | ------------------- |
| Closed  | Closed eye images   |
| Open    | Open eye images     |
| no_yawn | Normal mouth images |
| yawn    | Yawning images      |

Dataset split:

* 70% Training
* 15% Validation
* 15% Testing

---

# 🧠 Model Architecture

## 1️⃣ Custom CNN

A custom Convolutional Neural Network was built using:

* Conv2D
* MaxPooling2D
* Flatten
* Dense
* Dropout

Purpose:

* Compare performance with pretrained models

---

## 2️⃣ MobileNetV2

MobileNetV2 transfer learning model was used because:

* Lightweight architecture
* Faster training
* Better real-time performance
* Suitable for webcam applications

Fine-tuning was applied to improve performance.

---

# 👁 Eye State Model

A separate eye-state model was trained using:

* Closed
* Open

Purpose:

* Specialized eye fatigue detection

---

# 👄 Mouth State Model

A separate mouth-state model was trained using:

* no_yawn
* yawn

Purpose:

* Specialized yawning detection

---

# 🔄 Fatigue Logic

The four-class output was converted into real-world fatigue stages.

| Model Prediction | Fatigue Stage  |
| ---------------- | -------------- |
| Open             | Alert          |
| no_yawn          | Alert          |
| yawn             | Mild Fatigue   |
| Closed           | Severe Fatigue |

---

# 📈 Evaluation Metrics

The following evaluation metrics were used:

* Accuracy
* Loss
* Precision
* Recall
* F1-score
* Confusion Matrix

The project achieved approximately:

```text
Test Accuracy: 86% - 90%
```

---

# 📉 Fatigue Progression Analysis

A fatigue progression curve was generated to monitor:

* Fatigue increase over time
* Transition from alert to drowsy state

This simulates real-world driver monitoring.

---

# 📊 Performance Analysis

Performance analysis includes:

* Class-wise accuracy
* Confusion matrix
* Error case analysis
* Misclassification analysis

Limitations observed:

* Low lighting conditions
* Side face angles
* Motion blur
* Occlusion and glasses

---

# 💻 Streamlit Application

The Streamlit application provides:

## Dashboard

* Project overview
* Accuracy
* Model details

## Image Detection

* Upload image
* Predict fatigue level

## Live Webcam Detection

* Real-time fatigue monitoring
* Driver alert detection

---

# ▶️ How to Run

## 1️⃣ Create Virtual Environment

```bash
python -m venv venv
```

## 2️⃣ Activate Environment

```bash
venv\Scripts\activate
```

## 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

## 4️⃣ Run Dataset Split

```bash
py src/data_preprocessing.py
```

## 5️⃣ Train Models

```bash
py src/train_compare.py
```

```bash
py src/train_eye_model.py
```

```bash
py src/train_mouth_model.py
```

## 6️⃣ Evaluate Model

```bash
py src/evaluate.py
```

## 7️⃣ Run Streamlit App

```bash
python -m streamlit run app/streamlit_app.py
```

---

# ✅ Results

* Real-time driver monitoring achieved
* Fatigue stage detection implemented
* Streamlit UI successfully developed
* Multiple model comparison completed
* Performance analysis completed

---

# 🚀 Future Improvements

Possible future enhancements:

* Audio alert system
* Head pose estimation
* Night vision support
* Better low-light detection
* Cloud deployment
* Mobile application support

---

# 👨‍💻 Author

Tamilarasi Rajendran

Driver Drowsiness Detection System using Deep Learning and Streamlit.
