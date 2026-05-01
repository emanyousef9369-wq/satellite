import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.models import load_model

model = load_model("final_model.keras", compile=False)

classes = ['cloudy', 'fire', 'floods', 'normal']

st.title("🌍 Satellite Image Classification App")
st.write("Upload a satellite image and get prediction")


uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

def preprocess(img):
    img = img.resize((64, 64))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = preprocess(image)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    
    st.success(f"Prediction: {classes[class_index]}")
