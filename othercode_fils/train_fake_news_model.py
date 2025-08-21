# train_fake_news_model.py

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def generate_fake_news_dataset(file_name='fake_news_dataset.csv'):
    """
    Generates a simple CSV dataset for fake news detection.
    """
    data = {
        'text': [
            "Breaking: Aliens landed in New York, world leaders in panic!",
            "Study finds coffee consumption linked to longer life.",
            "URGENT: New virus spreads globally, no cure found yet.",
            "Local council approves new park development project.",
            "Shocking: Celebrity spotted with three heads!",
            "Scientists discover new species of deep-sea fish.",
            "This article claims the moon is made of cheese.",
            "Official report confirms economic growth in Q3.",
            "Clickbait: You won't believe what this politician said!",
            "Researchers publish findings on climate change impacts."
        ],
        'label': [
            "fake",
            "real",
            "fake",
            "real",
            "fake",
            "real",
            "fake",
            "real",
            "fake",
            "real"
        ],
        'reason': [
            "Highly improbable event, typical of sensationalist fake news.",
            "Based on scientific studies, common news topic.",
            "Often used in sensationalist headlines, check official health organizations.",
            "Typical local government news, verifiable.",
            "Absurd claim, clear sign of fake news.",
            "Common scientific discovery, verifiable through scientific journals.",
            "Clearly a false claim, lacks any scientific basis.",
            "Official economic reports are verifiable from government sources.",
            "Uses sensational language to attract clicks, common fake news tactic.",
            "Verifiable through academic publications and scientific consensus."
        ]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(file_name, index=False)
    print(f"Generated '{file_name}' successfully.")

def train_fake_news_model():
    """
    Trains a fake news classification model and saves it.
    """
    dataset_file = 'fake_news_dataset.csv'
    generate_fake_news_dataset(dataset_file) # Ensure dataset exists

    df = pd.read_csv(dataset_file)

    # Train the model
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['text'])
    y = df['label']

    # Split data for validation (optional)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_train)

    # Save the trained model and vectorizer
    with open('fake_news_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("Saved 'fake_news_vectorizer.pkl'")

    with open('fake_news_classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    print("Saved 'fake_news_classifier.pkl'")

    print("\nLocal fake news classification model training complete.")
    print("You can now run the Flask application.")

if __name__ == '__main__':
    train_fake_news_model()
    print("Please ensure 'pandas' and 'scikit-learn' are installed: pip install pandas scikit-learn")
