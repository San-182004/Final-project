import streamlit as st
from PIL import Image
import numpy as np
from tensorflow import keras

# Streamlit UI
st.image("Image.jpg", use_column_width=True)
st.title("Brain Tumor and Alzheimer's Disease detection using Python")
st.write("This application is a dynamic detection model that classifies the MRI scans uploaded by the users, for Brain Tumor and Alzheimer's disease.")
st.write("Upload an image, of appropriate filetype(jpg,jpeg,png).")

# Model Selection
model_option = st.radio("Choose a model:", ("Alzheimer Detection Model", "Brain Tumor Detection Model"))

# Load the selected model
if model_option == "Alzheimer Detection Model":
    model = keras.models.load_model("Alzheimer_detection_model.h5")
    class_names = ["Mild Demented", "Moderate Demented", "Non-Demented", "Very-Mild Demented"]
else:
    model = keras.models.load_model("Brain_Tumor_Model.h5")  # Path to Model B
    class_names = ["Glioma", "Healthy", "Meningioma", "Pituitary"]  # Replace with actual class names

if model_option == "Alzheimer Detection Model":
    st.write("""\t\tAlzheimer's disease is classified into different stages based on cognitive decline and daily functioning. 
    Here’s a brief summary of the key categories: 
        \n1.Non-Demented (No Dementia) – Individuals show no signs of cognitive impairment. Memory, reasoning, and daily activities are unaffected.
        \n2.Very Mildly Demented (Early Stage) – Slight memory lapses occur, but they do not interfere significantly with daily life. Individuals may occasionally forget names or misplace objects.
        \n3.Mildly Demented (Mild Alzheimer's) – Noticeable memory loss, difficulty with problem-solving, and challenges in managing finances or planning. Individuals may struggle to recall recent events but can still perform basic self-care.
        \n4.Moderate Demented (Moderate Alzheimer's) – Significant cognitive decline affects language, problem-solving, and personal care. Patients may become confused about time and place, need assistance with daily tasks, and experience personality changes.
        \nEach stage progressively worsens, with early detection and intervention playing a key role in managing symptoms.""")
else:
    st.write("""**A brain tumor is an abnormal growth of cells in or around the brain. It can be benign (non-cancerous) or malignant (cancerous). Brain tumors can develop from brain tissue itself (primary tumors) or spread from other parts of the body (secondary or metastatic tumors). Depending on their size and location, they may affect brain function, causing symptoms like headaches, seizures, vision problems, and difficulty with movement or cognition.
        \n\n1.Glioma:
        \n\t-A type of tumor that starts in the glial cells of the brain or spinal cord.
        \n\t-Can be malignant (cancerous) or benign.
        \n\t-Common subtypes include astrocytomas, oligodendrogliomas, and glioblastomas.
        \n\t\tSymptoms: Headaches, seizures, memory loss, and neurological deficits.
        \n\n2.Pituitary Tumor:
        \n\t-Forms in the pituitary gland, affecting hormone production.
        \n\t-Mostly benign but can disrupt bodily functions by overproducing or underproducing hormones.
        \n\t\tSymptoms: Vision problems, headaches, fatigue, and hormonal imbalances.
        \n\n3.Meningioma:
        \n\t-A tumor that arises from the meninges (the membranes covering the brain and spinal cord).
        \n\t-Usually benign but can grow and cause pressure-related symptoms.
        \n\t\tSymptoms: Headaches, seizures, vision problems, and cognitive changes.
        \n\4.Healthy Brain:
        \n\t-A normal brain with no abnormal growths or masses.
        \n\t-Functions optimally without neurological deficits or symptoms caused by tumors.
        \n\nTreatment options include surgery, radiation, chemotherapy, and targeted therapies.""")


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
    st.write(f"{class_names[predicted_class]} with confidence rate of {confidence:.2f}%")

