import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model('model.keras')

def predict_image(image):

    image = cv2.resize(image, (64, 64))
    image = image / 255.
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    
    return 'Car' if prediction < 0.5 else 'Bike'

st.title('Car or Bike Image guessing AI model')
st.write("Developed by Syed M Fraz Ali")
st.write("“Warning: This AI has a one-track mind! It’s either ‘Car’ or ‘Bike’. Any other photo will cause a serious case of identity crisis!”")
file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])

if file is not None:

    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, use_column_width=True)
    prediction = predict_image(image)
    st.write("This image is most likely a ", prediction)