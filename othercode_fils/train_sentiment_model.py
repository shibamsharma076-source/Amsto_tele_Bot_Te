# train_sentiment_model.py

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression # Good for text classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def generate_sentiment_dataset(file_name='sentiment_dataset.csv'):
    """
    Generates a simple CSV dataset for sentiment analysis.
    """
    data = {
        'text': [
            "I love this product! It's amazing.",
            "This is terrible, I'm so disappointed.",
            "The service was okay, nothing special.",
            "Feeling great today, everything is perfect.",
            "This movie was boring and slow.",
            "It's a neutral statement.",
            "Fantastic experience, highly recommend!",
            "Worst day ever, everything went wrong.",
            "The weather is cloudy.",
            "I am very happy with the results."
        ],
        'sentiment': [
            "positive",
            "negative",
            "neutral",
            "positive",
            "negative",
            "neutral",
            "positive",
            "negative",
            "neutral",
            "positive"
        ]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(file_name, index=False)
    print(f"Generated '{file_name}' successfully.")

def train_sentiment_model():
    """
    Trains a sentiment classification model and saves it.
    """
    dataset_file = 'sentiment_dataset.csv'
    generate_sentiment_dataset(dataset_file) # Ensure dataset exists

    df = pd.read_csv(dataset_file)

    # Train the model
    vectorizer = TfidfVectorizer(max_features=1000) # Limit features to avoid overfitting on small data
    X = vectorizer.fit_transform(df['text'])
    y = df['sentiment']

    # Split data for validation (optional)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifier = LogisticRegression(max_iter=1000) # Increased max_iter for convergence
    classifier.fit(X_train, y_train)

    # Evaluate the model
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Sentiment model trained successfully with an accuracy of: {accuracy:.2f}")

    # Save the trained model and vectorizer
    with open('sentiment_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("Saved 'sentiment_vectorizer.pkl'")

    with open('sentiment_classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    print("Saved 'sentiment_classifier.pkl'")

    print("\nLocal sentiment classification model training complete.")
    print("You can now run the Flask application.")

if __name__ == '__main__':
    train_sentiment_model()
    print("Please ensure 'pandas' and 'scikit-learn' are installed: pip install pandas scikit-learn")
