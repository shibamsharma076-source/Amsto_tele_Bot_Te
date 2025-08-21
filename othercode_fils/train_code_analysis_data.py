# train_code_analysis_data.py

import pandas as pd
import os

def generate_code_analysis_dataset(file_name='code_analysis_dataset.csv'):
    """
    Generates a simple CSV dataset for code analysis scenarios.
    This acts as a lookup table for basic code analysis queries.
    """
    data = {
        'query_type': [
            "detect_language", "detect_language", "detect_language",
            "fix_code", "fix_code", "fix_code",
            "analyze_code", "analyze_code"
        ],
        'code_snippet_pattern': [
            "import os", "function", "<!DOCTYPE html>",
            "print('Hello')", "for i in range(5):", "const x = 10",
            "def factorial(n):", "SELECT * FROM users"
        ],
        'code_language': [
            "Python", "JavaScript", "HTML",
            "Python", "Python", "JavaScript",
            "Python", "SQL"
        ],
        'issue_or_analysis_focus': [
            "Language Detection", "Language Detection", "Language Detection",
            "Missing parenthesis for print", "Incorrect indentation", "Variable declaration",
            "Function purpose", "Database query analysis"
        ],
        'local_response': [
            "This looks like Python code.",
            "This looks like JavaScript code.",
            "This looks like HTML code.",
            "To fix 'print(\\'Hello\\')', you need to add parentheses: `print('Hello')`.",
            "If 'for i in range(5):' is causing an error, check the indentation of the lines below it.",
            "The `const` keyword in JavaScript declares a constant variable.",
            "The `factorial` function calculates the factorial of a number.",
            "This SQL query selects all columns from the 'users' table."
        ]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(file_name, index=False)
    print(f"Generated '{file_name}' successfully.")

if __name__ == '__main__':
    generate_code_analysis_dataset()
    print("Please ensure 'pandas' is installed: pip install pandas")
