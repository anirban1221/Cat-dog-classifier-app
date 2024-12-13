import numpy as np 
import streamlit as st
import os
import tensorflow as tf
from PIL import Image
import cv2

# Load the TensorFlow Lite model
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="cat_dog_classifier_quantized.tflite")
    interpreter.allocate_tensors()
    return interpreter

# Function to preprocess the image for the model
def preprocess_image(img):
    test_img = cv2.resize(img, (256, 256))  # Resize to match model input size
    test_input = test_img.astype(np.float32) / 255.0  # Normalize pixel values
    return np.expand_dims(test_input, axis=0)  # Add batch dimension

# Function to make predictions with the TensorFlow Lite model
def predict_with_tflite(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    predictions = interpreter.get_tensor(output_details[0]['index'])
    return predictions

# Streamlit app
st.title('CAT OR DOG?')
st.header('This is a Cat-Dog Classification App')
st.subheader('Upload an image of a cat or dog and test the limits of AI')

uploaded_file = st.file_uploader('Upload an image', type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    temp_file_path = os.path.join("temp_image.jpg")
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    img = cv2.imread(temp_file_path)
    # Display the uploaded image
    st.image(img, caption='Uploaded Image', use_container_width=False, width=400)
    
    # Preprocess the image for the model
    test_input = preprocess_image(img)

    # Load the TensorFlow Lite model
    interpreter = load_tflite_model()

    # Make prediction
    prediction = predict_with_tflite(interpreter, test_input)

    # Interpret the prediction
    if prediction[0][0] < 0.5:
        st.header("This is a Cat")
    else:
        st.header("This is a Dog")

    # Remove the temporary image file
    os.remove(temp_file_path)
