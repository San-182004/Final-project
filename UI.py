import streamlit as st
from PIL import Image
import numpy as np
from tensorflow import keras

# Streamlit UI
st.title("Image Recognition App")
st.write("Upload an image, and I'll tell you what it is!")

# Model Selection
model_option = st.radio("Choose a model:", ("Alzheimer Detection Model", "Model B"))

# Load the selected model
if model_option == "Alzheimer Detection Model":
    model = keras.models.load_model("Alzheimer_detection_model.h5")
    class_names = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
else:
    model = keras.models.load_model("Model_B.h5")  # Path to Model B
    class_names = ["Class A", "Class B", "Class C", "Class D"]  # Replace with actual class names

# Image Upload
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

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
