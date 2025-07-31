import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pyttsx3

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# Speak function
def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except:
        st.warning("🔇 Could not play audio. Check pyttsx3 installation.")

# Streamlit page config
st.set_page_config(page_title="🦋 Insect Detector", layout="centered")
st.markdown("""
    <style>
        .main {
            background: linear-gradient(to right, #74ebd5, #acb6e5);
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
            backdrop-filter: blur(8.5px);
            color: #000000;
        }
        .title {
            font-size: 2.5rem;
            font-weight: 800;
            color: #000000;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">🦋 Real-time Insect Detection App</div>', unsafe_allow_html=True)
st.markdown("Detect insects via webcam, image or video using an AI model.")

# Load model
@st.cache_resource
def load_model_cached():
    return load_model("insect_model_mobilenetv2.keras")

model = load_model_cached()
class_names = ['Butterfly', 'Dragonfly', 'Grasshopper', 'Ladybird', 'Mosquito']
IMG_SIZE = 150

def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0]
    class_id = np.argmax(prediction)
    confidence = prediction[class_id]
    return class_names[class_id], confidence

def display_prediction(label, confidence):
    st.success(f"🧠 Predicted: **{label}** with {confidence*100:.2f}% confidence")

# ----------- Webcam Input -----------
if st.checkbox("📷 Use Webcam"):
    st.warning("Press 'q' in the webcam window to quit.")
    cap = cv2.VideoCapture(0)
    stop_button = st.button("⛔ Stop Webcam")

    stframe = st.empty()
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        label, conf = predict(pil_img)

        cv2.putText(frame, f"{label} ({conf*100:.2f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        stframe.image(frame, channels="BGR", caption="Webcam Feed")
        if conf > 0.8:
            display_prediction(label, conf)
            speak(f"{label} detected")

    cap.release()

# ----------- Image Upload -----------
uploaded_image = st.file_uploader("📁 Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    label, confidence = predict(image)
    display_prediction(label, confidence)
    speak(f"{label} detected")

# ----------- Video Upload -----------
uploaded_video = st.file_uploader("📁 Upload a Video", type=["mp4", "avi", "mov"])
if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    predictions = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        label, conf = predict(pil_img)
        predictions.append((label, round(conf, 2)))
        cv2.putText(frame, f"{label} ({conf*100:.2f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        stframe.image(frame, channels="BGR")

        if frame_count % 30 == 0 and conf > 0.8:
            speak(f"{label} detected")

    cap.release()
    os.unlink(tfile.name)

    st.subheader("📊 Prediction Summary from Video")
    if predictions:
        unique_counts = {}
        for label, conf in predictions:
            unique_counts[label] = unique_counts.get(label, 0) + 1
        for label, count in unique_counts.items():
            st.write(f"🪲 {label}: {count} frames")

st.markdown("</div>", unsafe_allow_html=True)
