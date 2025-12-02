import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st


working_dir = os.path.dirname(os.path.abspath(__file__))

# Safe path creation
model_path = os.path.join(working_dir, "plant_model.h5")
class_indices_path = os.path.join(working_dir, "class_indices.json")

# Load model and classes
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(class_indices_path))


# Load & preprocess image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array


# Predict image class
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Streamlit App
st.title('Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Save the uploaded file temporarily
            temp_path = os.path.join(working_dir, "temp_image.jpg")
            image.save(temp_path)

            # Predict
            prediction = predict_image_class(model, temp_path, class_indices)
            st.success(f'Prediction: {prediction}')
