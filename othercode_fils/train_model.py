# train_model.py

import sys
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def train_model(csv_file_path, text_col='text', label_col='label'):
    """
    Trains a spam classification model using a provided CSV file
    and saves the model and vectorizer to .pkl files.
    This version allows for custom column names for the text and labels.
    """
    try:
        # Attempt to load the dataset with different encodings
        try:
            df = pd.read_csv(csv_file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(csv_file_path, encoding='latin-1')

        print(f"Loaded dataset from '{csv_file_path}'.")

        # --- IMPORTANT CHANGE: Check for user-specified columns ---
        if text_col not in df.columns or label_col not in df.columns:
            error_msg = (
                f"Error: CSV file must contain '{text_col}' and '{label_col}' columns.\n"
                f"Found columns: {list(df.columns)}\n"
                f"Please provide the correct column names using --text-col and --label-col arguments."
            )
            raise ValueError(error_msg)

        # Handle 'ham' labels, converting them to 'not spam'
        df[label_col] = df[label_col].replace('ham', 'not spam')
        print("Converted 'ham' labels to 'not spam' in the dataset.")

        # Train the model
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df[text_col])
        y = df[label_col]

        classifier = MultinomialNB()
        classifier.fit(X, y)

        # Save the updated model and vectorizer
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        print("Saved 'vectorizer.pkl'")

        with open('classifier.pkl', 'wb') as f:
            pickle.dump(classifier, f)
        print("Saved 'classifier.pkl'")

        print("\nModel training and update complete.")
        print("Remember to restart the Flask application to use the new model.")

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during training: {e}")
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python train_model.py <path_to_your_csv_file> [--text-col <text_column_name>] [--label-col <label_column_name>]")
        sys.exit(1)

    csv_path = sys.argv[1]
    text_column_name = 'text'
    label_column_name = 'label'

    # Parse command-line arguments for custom column names
    if '--text-col' in sys.argv:
        text_column_name = sys.argv[sys.argv.index('--text-col') + 1]
    if '--label-col' in sys.argv:
        label_column_name = sys.argv[sys.argv.index('--label-col') + 1]

    train_model(csv_path, text_col=text_column_name, label_col=label_column_name)
