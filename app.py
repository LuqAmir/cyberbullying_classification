# app.py

import streamlit as st
import pickle

# Load your model and vectorizer
model_path = "models/model.pkl"
vectorizer_path = "models/vectorizer.pkl"

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# App Title
st.title("Text Classification App")

# User Input
user_input = st.text_area("Enter text for classification:")

if st.button("Classify"):
    if user_input.strip():
        # Transform input and make a prediction
        input_vectorized = vectorizer.transform([user_input])
        prediction = model.predict(input_vectorized)[0]

        # Display the result
        st.success(f"The predicted label is: **{prediction}**")
    else:
        st.warning("Please enter some text to classify!")
