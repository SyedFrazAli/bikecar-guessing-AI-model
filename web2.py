import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load the model
model = load_model('model.keras')

def predict_image(image):
    # Preprocess the image
    image = cv2.resize(image, (64, 64))
    image = image / 255.
    image = np.expand_dims(image, axis=0)

    # Make the prediction
    prediction = model.predict(image)
    
    return 'Car' if prediction < 0.5 else 'Bike'

st.title('Car or Bike Image guessing AI model')
st.write("Developed with ❤ by Your Name")
# Upload the image file
file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])

if file is not None:
    # Convert the file to an opencv image
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display the image
    st.image(image, use_column_width=True)

    # Make a prediction
    prediction = predict_image(image)

    # Show the prediction
    st.write("This image is most likely a ", prediction)

