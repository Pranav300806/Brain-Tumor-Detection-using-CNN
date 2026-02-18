import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # hide TF warnings
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

# ---------------- CSS ----------------
st.markdown("""
<style>

/* Make page wide */
.main .block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

/* Background */
.stApp {
    background: linear-gradient(-45deg, #141E30, #243B55, #1c1c3c, #2b5876);
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
    color: white;
}
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Title */
.big-title {
    font-size: 65px;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(to right, #00f5ff, #ff00c8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sub-text {
    font-size: 22px;
    text-align: center;
    margin-bottom: 40px;
}

/* CARD STYLE (REAL FIX) */
[data-testid="stVerticalBlock"] > div:has(div.stFileUploader),
[data-testid="stVerticalBlock"] > div:has(div.stImage),
[data-testid="stVerticalBlock"] > div:has(div.stProgress) {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(14px);
    padding: 35px;
    border-radius: 25px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.4);
}

.result-box {
    padding: 30px;
    border-radius: 20px;
    font-size: 26px;
    font-weight: bold;
    text-align: center;
    margin-top: 30px;
}

.footer {
    text-align: center;
    margin-top: 70px;
    color: #cccccc;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/brain_tumor_model.keras")

model = load_model()

# ---------------- TITLE ----------------
st.markdown("<div class='big-title'>🧠 Brain Tumor Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>AI Powered MRI Scan Analysis</div>", unsafe_allow_html=True)
st.write("")

# ---------------- LAYOUT (REAL CARDS) ----------------
container = st.container()
col1, col2 = container.columns([1.2,1], gap="large")

with col1:
    st.markdown("### 📤 Upload MRI Scan")
    uploaded_file = st.file_uploader("Choose MRI image", type=["jpg","jpeg","png"])

with col2:
    st.markdown("### 📊 Prediction Result")
    result_placeholder = st.empty()

# ---------------- PREPROCESS (FIXED SIZE) ----------------
def preprocess_image(image):
    image = np.array(image)
    
    # ⭐ Resize to EXACT training size
    image = cv2.resize(image, (128,128))
    
    # Normalize
    image = image / 255.0
    
    # ⭐ Add batch dimension (DO NOT FLATTEN)
    image = np.expand_dims(image, axis=0)
    
    return image


# ---------------- PREDICTION ----------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    processed = preprocess_image(image)

    with st.spinner("🔍 Analyzing MRI Scan..."):
        prediction = model.predict(processed)

        # ⭐ MULTI-CLASS FIX
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        class_names = [
            "Glioma Tumor",
            "Meningioma Tumor",
            "Pituitary Tumor",
            "No Tumor"
        ]

        result = class_names[predicted_class]

    with col2:
        # Tumor vs No tumor color logic
        if result == "No Tumor":
            color = "linear-gradient(to right,#00c853,#00e676)"
            emoji = "✅"
        else:
            color = "linear-gradient(to right,#ff416c,#ff4b2b)"
            emoji = "⚠"

        result_placeholder.markdown(
            f"<div class='result-box' style='background:{color};'>"
            f"{emoji} {result}"
            f"</div>",
            unsafe_allow_html=True
        )

        st.progress(float(confidence))
        st.markdown(
            f"<center>Confidence: {confidence*100:.2f}%</center>",
            unsafe_allow_html=True
        )


# ---------------- FOOTER ----------------
st.markdown("<div class='footer'>Developed for AI Medical Image Analysis Project</div>", unsafe_allow_html=True)
