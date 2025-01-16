import streamlit as st
import pickle

# Load all models and vectorizer
models = {
    "LinearSVC": "models/LinearSVC.pkl",
    "KNeighborsClassifier": "models/KNeighborsClassifier.pkl",
    "Logistic Regression": "models/logistic_regression.pkl",
    "MultinomialNB": "models/MultinomialNB.pkl"
}
vectorizer_path = "vectorizer/vectorizer.pkl"

# Load the vectorizer
with open(vectorizer_path, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# App Title
st.title("Text Classification App")

# User Input
user_input = st.text_area("Enter text for classification:")

# Model Selection
model_name = st.selectbox("Choose a model:", list(models.keys()))

if st.button("Classify"):
    if user_input.strip():
        # Load the selected model
        with open(models[model_name], "rb") as model_file:
            model = pickle.load(model_file)
        
        # Transform input and make a prediction
        input_vectorized = vectorizer.transform([user_input])
        prediction = model.predict(input_vectorized)[0]
        
        # Display the result
        st.success(f"The predicted label is: **{prediction}**")
        
        # Display the probability for cyberbullying or not (if applicable)
        if hasattr(model, "decision_function"):  # For models like SVC, LogisticRegression
            probabilities = model.decision_function(input_vectorized)
            st.write(f"Decision function output: {probabilities}")
        elif hasattr(model, "predict_proba"):  # For models like KNN, MultinomialNB
            probabilities = model.predict_proba(input_vectorized)
            st.write(f"Probabilities: {probabilities}")
    else:
        st.warning("Please enter some text to classify!")
