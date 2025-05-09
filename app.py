import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
from PIL import Image


MODEL_PATH = "mango_leaf_model_3.h5"
model = load_model(MODEL_PATH)


CLASS_NAMES = ["Anthracnose", "Bacterial Canker", "Cutting Weevil", "Die Back", "Gall Midge", "Healthy", "Powdery Mildew", "Sooty Mould"]

def predict_disease(img):
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    return predicted_class, confidence

# Streamlit UI
st.title("🍃 Mango Leaf Disease Detection")
st.write("Upload an image of a mango leaf to detect its disease.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("Predict"):
        prediction, confidence = predict_disease(image)
        st.success(f"Prediction: **{prediction}**")
        st.info(f"Confidence: **{confidence:.2f}%**")
