# train_disease_model.py

import sys
import pandas as pd
import pickle
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def clean_and_uniquify_column_names(df):
    """
    Cleans column names and ensures they are unique within the dataframe.
    """
    cleaned_cols = []
    seen_cols = {}
    
    for col in df.columns:
        # Step 1: Robust cleaning using a regular expression
        cleaned_name = re.sub(r'[^a-zA-Z0-9_]+', '_', str(col)).strip('_').lower()
        
        # Step 2: Ensure uniqueness by adding a suffix if a duplicate is found
        if cleaned_name in seen_cols:
            seen_cols[cleaned_name] += 1
            cleaned_name = f"{cleaned_name}_{seen_cols[cleaned_name]}"
        else:
            seen_cols[cleaned_name] = 1
        
        cleaned_cols.append(cleaned_name)
    
    df.columns = cleaned_cols
    return df

def load_and_preprocess_file(file_path):
    """
    Loads a single CSV or XLSX file and cleans its column names,
    ensuring they are unique.
    """
    try:
        if file_path.endswith('.csv'):
            try:
                df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin-1')
            print(f"Loaded CSV file: '{file_path}'")
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
            print(f"Loaded XLSX file: '{file_path}'")
        else:
            raise ValueError("Unsupported file format. Please provide .csv or .xlsx files.")

        # Clean and make column names unique for this specific dataframe
        df = clean_and_uniquify_column_names(df)
        print(f"Cleaned and uniquified columns for '{file_path}': {list(df.columns)}")
        
        return df

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while loading '{file_path}': {e}")
        sys.exit(1)

def train_model(file_paths):
    """
    Trains a disease prediction model using a list of CSV/XLSX files.
    """
    if not file_paths:
        print("No files provided for training.")
        sys.exit(1)

    try:
        # Load the first file to initialize the combined DataFrame
        first_df = load_and_preprocess_file(file_paths[0])
        all_symptom_columns = [col for col in first_df.columns if col != 'prognosis']
        combined_df = first_df.reindex(columns=all_symptom_columns + ['prognosis'], fill_value=0)
        
        # Process remaining files one by one and append to the combined DataFrame
        for file_path in file_paths[1:]:
            df = load_and_preprocess_file(file_path)
            
            # Update the unified list of all symptom columns
            current_symptom_cols = [col for col in df.columns if col != 'prognosis']
            all_symptom_columns = sorted(list(set(all_symptom_columns) | set(current_symptom_cols)))
            
            # Re-index the current dataframe to match the unified column list
            df_reindexed = df.reindex(columns=all_symptom_columns + ['prognosis'], fill_value=0)
            
            # Concatenate the current dataframe with the combined one
            combined_df = pd.concat([combined_df, df_reindexed], ignore_index=True)
            print(f"Appended data from '{file_path}'. Current total rows: {len(combined_df)}")

    except MemoryError:
        print("\n-------------------------------------------------------------")
        print("ERROR: Your dataset is too large to fit in memory (RAM).")
        print("Please consider using a smaller sample of your data for training.")
        print("-------------------------------------------------------------")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred during data combination: {e}")
        sys.exit(1)

    print(f"\nCombined data from all files. Total rows: {len(combined_df)}")

    # Check if the 'prognosis' column is present
    if 'prognosis' not in combined_df.columns:
        raise ValueError("Combined dataset must contain a 'prognosis' column.")

    # All columns except 'prognosis' are symptoms
    X = combined_df[all_symptom_columns]
    y = combined_df['prognosis']

    # Handle potential NaNs in the data by filling with 0 (important for re-indexing)
    X = X.fillna(0)
    
    print(f"Number of unique symptoms (features): {len(all_symptom_columns)}")
    print(f"Number of diseases (targets): {len(y.unique())}")

    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- IMPORTANT CHANGE: Reduced n_estimators and added MemoryError handling ---
    try:
        # Train a RandomForestClassifier model with fewer estimators
        classifier = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1) # n_jobs=-1 uses all available cores
        classifier.fit(X_train, y_train)
    except MemoryError:
        print("\n-------------------------------------------------------------")
        print("ERROR: Not enough memory to train the RandomForestClassifier.")
        print("Try reducing 'n_estimators' further (e.g., to 10 or 20),")
        print("or use a smaller sample of your data for training.")
        print("-------------------------------------------------------------")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during model training: {e}")
        sys.exit(1)

    # Evaluate the model
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained successfully with an accuracy of: {accuracy:.2f}")

    # Save the trained model and the list of symptom columns
    with open('disease_classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    print("Saved 'disease_classifier.pkl'")
    
    with open('symptom_columns.pkl', 'wb') as f:
        pickle.dump(all_symptom_columns, f)
    print("Saved 'symptom_columns.pkl'")

    print("\nModel training and update complete.")
    print("Remember to restart the Flask application to use the new model.")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python train_disease_model.py <file1.csv> <file2.xlsx> ...")
        sys.exit(1)

    file_paths = sys.argv[1:]
    train_model(file_paths)
