import pandas as pd
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("IMDB Dataset(1).csv")

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub("<.*?>", "", text)  # Remove HTML tags
    text = re.sub("\\d+", "", text)  # Remove numbers
    return text

df["cleaned_review"] = df["review"].apply(clean_text)

# Convert labels to binary
df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned_review"], df["sentiment"], test_size=0.2, random_state=42
)

# Text vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model training
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# User input for sentiment prediction
def predict_sentiment():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    
    review = input("Enter a movie review: ")
    review_cleaned = clean_text(review)
    review_vec = vectorizer.transform([review_cleaned])
    prediction = model.predict(review_vec)[0]
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    print(f"Predicted Sentiment: {sentiment}")

# Run user input function
if __name__ == "__main__":
    predict_sentiment()