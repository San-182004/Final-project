import streamlit as st
from PIL import Image
import numpy as np
from pickle
from keras.applications.resnet import preprocess_input, decode_predictions

model = pickle.load("Models/Alzheimer_detection_model.h5", compile=False)

# Streamlit UI
st.title("ALzheimer's Detection")

st.write("Upload the image in the designated space and in supported formats.")

uploaded_image = st.file_uploader("Upload the image(jpg,jpeg,png): ", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    image = image.resize((128, 128))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    predictions = model.predict(image_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    st.write("### Predictions:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        st.write(f"{i + 1}. **{label}** ({score * 100:.2f}%)")
