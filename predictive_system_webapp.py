# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 20:35:42 2023

@author: sss
"""


import numpy as np
import pickle
import streamlit as st
from PIL import Image


# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# creating a function for Prediction

def diabetes_prediction(image):
    
    # preprocess the image
    img = Image.open(image)
    # convert to grayscale and resize
    img = img.convert('L').resize((64, 64))
    # convert to numpy array
    input_data_as_numpy_array = np.asarray(img)
    # flatten the array
    input_data_as_numpy_array = input_data_as_numpy_array.flatten()
    
    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
  
def main():
    
    
    # giving a title
    st.title('Diabetes Prediction Web App')
    
    
    # getting the input data from the user
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    if uploaded_file is not None:
        diagnosis = diabetes_prediction(uploaded_file)
        
        
    st.success(diagnosis)
    
if __name__ == '__main__':
    main()
    
    
    
