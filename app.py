import numpy as np 
import pandas as pd 
import streamlit as st
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2

model=load_model('cat_dog_classifier.keras')
class_names = ['Cat', 'Dog']

st.title('CAT OR DOG?')
st.header('This is a Cat-Dog Classification App')
st.subheader('upload an image of a cat or dog and test the limits of AI')
uploaded_file=st.file_uploader('Upload an image',type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    temp_file_path = os.path.join("temp_image.jpg")
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    img = cv2.imread(temp_file_path)
    # Display the uploaded image
    st.image(img, caption='Uploaded Image', use_container_width=False,width=400)
    
    # Preprocess the image for the model
    
    # Preprocess the image using OpenCV
    test_img=cv2.resize(img,(256,256))  # Resize to match model input size

    test_input=test_img.reshape((1,256,256,3))
    # Make prediction
    prediction = model.predict(test_input)

    if prediction[0][0]==0.0 :
        st.header("this is a cat")
    elif prediction[0][0]==1.0:
        st.header("this is a dog")


    os.remove(temp_file_path)
    
