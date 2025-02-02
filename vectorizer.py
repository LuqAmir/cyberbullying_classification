import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Load dataset
data = pd.read_csv("cleaned_data.csv")

# Assuming 'Processed_Comment' column contains text
if 'cleaned_comments' not in data.columns:
    raise ValueError("Column 'cleaned_comments' not found in data.csv!")

# Fit vectorizer
vectorizer = CountVectorizer()
vectorizer.fit(data['cleaned_comments'])

# Create and fit the vectorizer on the training data
count_vector = CountVectorizer()
count_vector.fit(X_train)  # Fit on the training data

# Save vectorizer to a .pkl file
with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("Vectorizer saved successfully as 'vectorizer.pkl'!")
