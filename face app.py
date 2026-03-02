import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# --- 1. SETTING UP THE "SAFE" MODEL LOADING ---
# We must define 'tf' globally for the Lambda layer in the saved model
import builtins
builtins.tf = tf

@st.cache_resource # Prevents reloading the model on every button click
def load_emotion_model():
    try:
        # custom_objects handles the 'tf' NameError we saw earlier
        model = tf.keras.models.load_model(
            'facial_emotion_v1_2026.keras', 
            custom_objects={'tf': tf}, 
            safe_mode=False
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_emotion_model()

# --- 2. PREPROCESSING FUNCTION ---
def preprocess_image(img):
    # Convert PIL to OpenCv
    img = np.array(img.convert('RGB'))
    # Convert to Grayscale (as per our model training)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Resize to 48x48
    resized = cv2.resize(gray, (48, 48))
    # Add batch and channel dimensions: (1, 48, 48, 1)
    reshaped = np.reshape(resized, (1, 48, 48, 1))
    # Rescale pixels (Matches training preprocessing)
    return reshaped / 255.0

# --- 3. STREAMLIT UI ---
st.title("🎭 Facial Emotion Recognizer")
st.write("Upload a photo of a face to detect the emotion.")

# Sidebar info
st.sidebar.title("Model Details")
st.sidebar.info("Model: MobileNetV2 Transfer Learning\n\nClasses: 7 Emotions")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button('Predict Emotion'):
        with st.spinner('Analyzing...'):
            # Preprocess
            processed_img = preprocess_image(image)
            
            # Predict
            prediction = model.predict(processed_img)
            
            # Class Names (ensure order matches your training)
            class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
            
            result = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            
            # Show Results
            st.success(f"Prediction: **{result}**")
            st.progress(int(confidence))
            st.write(f"Confidence: {confidence:.2f}%")

            # Create a bar chart for all emotions
            chart_data = dict(zip(class_names, prediction[0]))
            st.bar_chart(chart_data)