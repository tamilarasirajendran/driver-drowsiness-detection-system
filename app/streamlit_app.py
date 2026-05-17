
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide"
)

# Change this if needed
MODEL_PATH = "models/mobilenetv2.keras"
CLASS_NAMES = ["Closed", "Open", "no_yawn", "yawn"]

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)

    if os.path.exists("models/model.keras"):
        return tf.keras.models.load_model("models/model.keras")

    if os.path.exists("models/model.h5"):
        return tf.keras.models.load_model("models/model.h5")

    raise FileNotFoundError("No saved model found in models/")

model = load_model()

# =====================================================
# STYLE
# =====================================================
st.markdown("""
<style>
.title {
    font-size: 40px;
    font-weight: 800;
    color: #1f4e79;
}
.subtitle {
    font-size: 16px;
    color: #555;
    margin-bottom: 10px;
}
.card {
    background: #f7f9fc;
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #e5eaf2;
    box-shadow: 0 4px 15px rgba(0,0,0,0.04);
}
.good {
    background: #e8f7ee;
    color: #1e7e34;
    padding: 12px 14px;
    border-radius: 12px;
    font-weight: 700;
}
.warn {
    background: #fff3cd;
    color: #856404;
    padding: 12px 14px;
    border-radius: 12px;
    font-weight: 700;
}
.bad {
    background: #f8d7da;
    color: #842029;
    padding: 12px 14px;
    border-radius: 12px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HELPERS
# =====================================================
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_result(label, confidence):
    if label == "Closed":
        return f"⚠️ Severe Fatigue ({confidence:.2f}%)", "bad"
    elif label == "yawn":
        return f"😴 Mild Fatigue ({confidence:.2f}%)", "warn"
    else:
        return f"✅ Alert Driver ({confidence:.2f}%)", "good"

def show_probability_chart(pred):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(CLASS_NAMES, pred[0])
    ax.set_title("Prediction Probability")
    ax.set_ylabel("Confidence")
    plt.xticks(rotation=15)
    plt.tight_layout()
    return fig

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("🚗 Driver Monitor")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "📷 Image Detection", "🎥 Live Webcam"]
)

st.sidebar.markdown("---")
st.sidebar.write("Model Accuracy: **90.37%**")
st.sidebar.write("Model: **MobileNetV2**")

# =====================================================
# DASHBOARD
# =====================================================
if page == "🏠 Dashboard":
    st.markdown('<div class="title">🚗 Driver Drowsiness Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Deep learning-based eye closure and yawning detection.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card"><h4>Test Accuracy</h4><h2>90.37%</h2></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><h4>Classes</h4><h2>4</h2></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card"><h4>Model</h4><h2>MobileNetV2</h2></div>', unsafe_allow_html=True)

    st.markdown("### What this project does")
    st.write(
        "This project detects driver fatigue using facial image classification. "
        "It predicts one of four classes: Closed, Open, no_yawn, and yawn. "
        "Then it converts that prediction into a simple fatigue result like Alert, Mild Fatigue, or Severe Fatigue."
    )

# =====================================================
# IMAGE DETECTION
# =====================================================
elif page == "📷 Image Detection":
    st.markdown('<div class="title">📷 Image Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload one image and get the result instantly.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Driver Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Uploaded Image", use_container_width=True)

        img_input = preprocess_image(img)
        pred = model.predict(img_input, verbose=0)
        idx = int(np.argmax(pred))
        label = CLASS_NAMES[idx]
        confidence = float(pred[0][idx] * 100)

        msg, kind = get_result(label, confidence)

        with col2:
            st.subheader("Prediction Result")
            if kind == "bad":
                st.markdown(f'<div class="bad">{msg}</div>', unsafe_allow_html=True)
            elif kind == "warn":
                st.markdown(f'<div class="warn">{msg}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="good">{msg}</div>', unsafe_allow_html=True)

            st.write(f"Predicted Class: **{label}**")
            st.write(f"Confidence: **{confidence:.2f}%**")

            st.pyplot(show_probability_chart(pred), use_container_width=True)

# =====================================================
# LIVE WEBCAM
# =====================================================
elif page == "🎥 Live Webcam":
    st.markdown('<div class="title">🎥 Live Webcam Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Start the camera for live driver monitoring.</div>', unsafe_allow_html=True)

    start = st.button("▶ Start Camera")
    stop = st.button("⏹ Stop Camera")

    frame_placeholder = st.empty()
    result_placeholder = st.empty()

    if "running" not in st.session_state:
        st.session_state.running = False

    if start:
        st.session_state.running = True

    if stop:
        st.session_state.running = False

    if st.session_state.running:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Camera not opened.")
        else:
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.error("Could not read camera frame.")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img_input = preprocess_image(img)

                pred = model.predict(img_input, verbose=0)
                idx = int(np.argmax(pred))
                label = CLASS_NAMES[idx]
                confidence = float(pred[0][idx] * 100)

                msg, kind = get_result(label, confidence)

                if kind == "bad":
                    result_placeholder.markdown(f'<div class="bad">{msg}</div>', unsafe_allow_html=True)
                elif kind == "warn":
                    result_placeholder.markdown(f'<div class="warn">{msg}</div>', unsafe_allow_html=True)
                else:
                    result_placeholder.markdown(f'<div class="good">{msg}</div>', unsafe_allow_html=True)

                cv2.putText(
                    frame_rgb,
                    f"{label} ({confidence:.1f}%)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                if stop:
                    st.session_state.running = False
                    break

            cap.release()