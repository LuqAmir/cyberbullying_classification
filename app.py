import streamlit as st
import pickle

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
            st.success(f"The predicted label is: **{prediction}**")

            # Display probabilities or decision function
            if hasattr(model, "decision_function"):  # For models like SVC, LogisticRegression
                probabilities = model.decision_function(input_vectorized)
                st.write(f"Decision function output: {probabilities}")
            elif hasattr(model, "predict_proba"):  # For models like KNN, MultinomialNB
                probabilities = model.predict_proba(input_vectorized)
                st.write(f"Probabilities: {probabilities}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter some text to classify!")
