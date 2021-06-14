######################
# Import libraries
######################

import pandas as pd
import streamlit as st
import altair as alt
from PIL import Image
import pickle

# loading the trained model
pickle_in = open('sentiment_classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

# loading the vectorizer
vectorizer_in = open('vectorizer.pickle', 'rb')
vectorizer = pickle.load(vectorizer_in)

######################
# Page Title
######################

st.write("""
# Review Classification Web App
This app classifies Reviews into POSITIVE, NEGATIVE or NEUTRAL!
***
""")

image = Image.open('nlp-logo.jpg')

st.image(image, use_column_width=True)

st.write("""
***
""")

######################
# Input Text Box
######################

#st.sidebar.header('Enter DNA sequence')
st.header('Enter Reviews here')

review_input = "This book was really helpful\nI did not enjoy the book\nI've been here since the beginning and have yet to go away unfulfilled"

#review = st.sidebar.text_area("Review input", review_input, height=250)
review = st.text_area("Review input", review_input, height=250)
review = review.lower().splitlines()
#review = review[1:] # Skips the review name (first line)
# review = ''.join(review) # Concatenates list to string

st.write("""
***
""")

## Prints the input review
st.header('INPUT (Review Query)')
review

st.write("""
***
""")

# when 'Predict' is clicked, make the prediction and store it
pred = classifier.predict(vectorizer.transform(review))
if st.button('Predict'):
 # Review Classification
 st.header('OUTPUT (Review Classification)')
 ### Printing the Classification
 st.text_area("Prediction", pred, height=20)
else:
 pass

