import streamlit as st
import seaborn as sns
import pickle
import numpy as np

# Load all models and their vectorizers
models = {
    "LinearSVC": {
        "model": "models/LinearSVC.pkl",
        "vectorizer": "vectorizer/LinearSVC_vectorizer.pkl"
    },
    "KNeighborsClassifier": {
        "model": "models/KNeighborsClassifier.pkl",
        "vectorizer": "vectorizer/KNeighborsClassifier_vectorizer.pkl"
    },
    "Logistic Regression": {
        "model": "models/logistic_regression.pkl",
        "vectorizer": "vectorizer/logistic_regression_vectorizer.pkl"
    },
    "MultinomialNB": {
        "model": "models/MultinomialNB.pkl",
        "vectorizer": "vectorizer/MultinomialNB_vectorizer.pkl"
    }
}

# App Title
st.title("Text Classification App")

# User Input
user_input = st.text_area("Enter text for classification:")

# Model Selection
model_name = st.selectbox("Choose a model:", list(models.keys()))

if st.button("Classify"):
    if user_input.strip():
        try:
            # Load the selected model and vectorizer
            with open(models[model_name]["model"], "rb") as model_file:
                model = pickle.load(model_file)
            with open(models[model_name]["vectorizer"], "rb") as vectorizer_file:
                vectorizer = pickle.load(vectorizer_file)

            # Transform input and make a prediction
            input_vectorized = vectorizer.transform([user_input])
            prediction = model.predict(input_vectorized)[0]

            # Display the result
            if prediction == 0:
                st.success("The predicted label is: Not Cyberbully")
            else:
                st.error("The predicted label is: Cyberbully")

            # Display probabilities or decision function
            if hasattr(model, "decision_function"):
                probabilities = model.decision_function(input_vectorized)
                st.write(f"Decision function output: {probabilities[0]}")
                if probabilities[0] > 0:
                    st.write("The input text is likely cyberbullying.")
                else:
                    st.write("The input text is not cyberbullying.")
            elif hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_vectorized)
                st.write(f"Probabilities: {probabilities[0]}")
                if probabilities[0][1] > 0.5:
                    st.write("The input text is likely cyberbullying.")
                else:
                    st.write("The input text is not cyberbullying.")

            # Clear unnecessary code for confusion matrix (irrelevant here)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter some text to classify!")
