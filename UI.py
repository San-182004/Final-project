import streamlit as st
from PIL import Image
import numpy as np
!pip install keras
from keras.models import load_model
from keras.applications.resnet import preprocess_input, decode_predictions

model = load_model("Models/Alzheimer_detection_model.h5", compile=False)

# Streamlit UI
st.title("Image Recognition App")

st.write("Upload an image, and I'll tell you what it is!")

# Image Upload
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open and display the image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image for MobileNetV2
    image = image.resize((128, 128))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    # Make Predictions
    predictions = model.predict(image_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Display the predictions
    st.write("### Predictions:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        st.write(f"{i + 1}. **{label}** ({score * 100:.2f}%)")
