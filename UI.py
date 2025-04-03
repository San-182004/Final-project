import streamlit as st
from PIL import Image
import numpy as np
from tensorflow import keras

# Streamlit UI
st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{Image.png}") no-repeat center center fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
st.title("Brain Tumor and Alzheimer's Disease detection using Python")
st.write("Upload an image, of appropriate filetype(jpg,jpeg,png).")

# Model Selection
model_option = st.radio("Choose a model:", ("Alzheimer Detection Model", "Brain Tumor Detection Model"))

# Load the selected model
if model_option == "Alzheimer Detection Model":
    model = keras.models.load_model("Alzheimer_detection_model.h5")
    class_names = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
else:
    model = keras.models.load_model("Brain_Tumor_Model.h5")  # Path to Model B
    class_names = ["Glioma", "Healthy", "Meningioma", "Pituitary"]  # Replace with actual class names

# Image Upload
uploaded_image = st.file_uploader("Upload file here:", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open and display the image
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    image = image.resize((128, 128))  # Adjust if needed
    image_array = np.array(image) / 255.0  # Normalizing as you used Rescaling(1./255)
    image_array = np.expand_dims(image, axis=0)

    # Make Predictions
    predictions = model.predict(image_array)
    st.write(predictions)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class] * 100

    # Display the prediction
    st.write("### Prediction:")
    st.write(f"**Result:** {class_names[predicted_class]} with confidence {confidence:.2f}%")
