import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
from keras.applications.resnet import preprocess_input, decode_predictions

# Correct model loading
# Model Selection
model_option = st.radio("Choose a model:", ("Alzheimer Detection Model", "Model B"))

# Load the selected model
if model_option == "Alzheimer Detection Model":
    model = load_model("Alzheimer_detection_model.h5")
    class_names = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
else:
    model = load_model("Model_B.h5")  # Replace with the correct path for Model B
    class_names = ["Class A", "Class B", "Class C"]  # Replace with Model B's class names
# Streamlit UI
st.title("Image Recognition App")
st.write("Upload an image, and I'll tell you what it is!")

# Image Upload
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open and display the image
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    image = image.resize((128, 128))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    # Make Predictions
    predictions = model.predict(image_array)
    st.write(predictions)

    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class] * 100

    # Display the prediction
    st.write("### Prediction:")
    st.write(f"**Result:** {class_names[predicted_class]} with confidence {confidence:.2f}%")
