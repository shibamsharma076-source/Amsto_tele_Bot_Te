# python recipe_dataset_generator_csv.py
import os
import json
import requests
import time
import pandas as pd
from urllib.parse import quote_plus
from typing import Dict, Any, List
from dotenv import load_dotenv
# --- Configuration ---
# You would need to replace this with a real API key from your environment variables.
# This is for demonstration purposes.
API_KEY = os.getenv("GEMINI_API_KEY", "")
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={API_KEY}"
OUTPUT_CSV_NAME = "recipes_dataset.csv"
# This list is used to seed the recipe generation. You would need to expand this list
# or use a more sophisticated method to generate a million unique prompts.
RECIPE_PROMPTS = [
    "A simple weeknight pasta dish.",
    "A vegetarian curry from India.",
    "A dessert suitable for a dinner party.",
    "A quick and easy chicken dish.",
    "A classic French soup.",
    "A seafood dish inspired by Mediterranean cuisine.",
    "A Mexican street food.",
    "A hearty breakfast recipe with eggs.",
    "A light and refreshing salad.",
    "A baked good using chocolate.",
    "A spicy Thai noodle dish.",
    "A classic Italian lasagna.",
    "A rustic French bread recipe.",
    "A vibrant smoothie for breakfast.",
    "A simple and healthy stir-fry.",
]

def generate_recipe_with_api(prompt: str) -> Dict[str, Any] | None:
    """
    Uses the Gemini API to generate a structured recipe in JSON format.
    
    Args:
        prompt (str): A natural language prompt for the recipe.

    Returns:
        Dict[str, Any] | None: A dictionary containing the recipe data, or None if the API call fails.
    """
    print(f"Generating recipe for: {prompt}")
    
    generation_config = {
        "responseMimeType": "application/json",
        "responseSchema": {
            "type": "OBJECT",
            "properties": {
                "recipe_name": {"type": "STRING"},
                "description": {"type": "STRING"},
                "ingredients": {"type": "ARRAY", "items": {"type": "STRING"}},
                "instructions": {"type": "ARRAY", "items": {"type": "STRING"}}
            },
            "propertyOrdering": ["recipe_name", "description", "ingredients", "instructions"]
        }
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"Provide a recipe based on the prompt: '{prompt}'. "
                                f"The response should be a JSON object with the keys: "
                                f"'recipe_name', 'description', 'ingredients' (list of strings), and "
                                f"'instructions' (list of strings). Do not include any other text."
                    }
                ]
            }
        ],
        "generationConfig": generation_config
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        
        api_response_text = response.json()['candidates'][0]['content']['parts'][0]['text']
        recipe_data = json.loads(api_response_text)
        return recipe_data
        
    except (requests.exceptions.RequestException, KeyError, json.JSONDecodeError) as e:
        print(f"Error generating recipe for '{prompt}': {e}")
        return None

def get_image_url(recipe_name: str) -> str:
    """
    Generates a placeholder image URL based on the recipe name.
    """
    search_query = quote_plus(recipe_name)
    return f"https://placehold.co/600x400/E868A7/FFFFFF?text={search_query}"

def get_video_url(recipe_name: str) -> str:
    """
    Generates a YouTube search URL for the recipe.
    """
    search_query = quote_plus(f"how to make {recipe_name}")
    return f"https://www.youtube.com/results?search_query={search_query}"

def save_recipes_to_csv(recipes: List[Dict[str, Any]]):
    """
    Converts the list of recipe dictionaries into a pandas DataFrame and saves it to a CSV.
    
    Args:
        recipes (List[Dict[str, Any]]): The list of recipe dictionaries to save.
    """
    if not recipes:
        print("No recipes to save. Exiting.")
        return
        
    print(f"Saving {len(recipes)} recipes to {OUTPUT_CSV_NAME}...")
    df = pd.DataFrame(recipes)
    # Convert lists in ingredients and instructions columns to strings
    df['ingredients'] = df['ingredients'].apply(lambda x: json.dumps(x))
    df['instructions'] = df['instructions'].apply(lambda x: json.dumps(x))
    df.to_csv(OUTPUT_CSV_NAME, index=False)
    print("CSV file saved successfully.")

def main():
    """
    Main function to orchestrate the dataset generation process.
    """
    all_recipes = []
    
    # Loop to generate recipes. The number of iterations determines the dataset size.
    # We will run this for a small number of recipes for demonstration.
    # To generate 100,000+ recipes, this loop would need to run for that many iterations
    # with a very diverse set of prompts.
    num_recipes_to_generate = 50 # Change this to a much larger number for a bigger dataset
    
    for i in range(num_recipes_to_generate):
        # A simple way to get more unique recipes is to use the prompts repeatedly
        # or generate new prompts dynamically.
        prompt = f"A recipe for a unique dish, inspired by one of these categories: {', '.join(RECIPE_PROMPTS)}"
        recipe_data = generate_recipe_with_api(prompt)
        
        if recipe_data:
            recipe_data['image_url'] = get_image_url(recipe_data['recipe_name'])
            recipe_data['video_url'] = get_video_url(recipe_data['recipe_name'])
            all_recipes.append(recipe_data)
            
            # Pause to avoid hitting API rate limits. This is crucial.
            time.sleep(2)
            
    # Save all the collected recipes to a single CSV file.
    save_recipes_to_csv(all_recipes)

if __name__ == "__main__":
    main()

































# # python recipe_dataset_generator.py

# import sqlite3
# import json
# import requests
# import time
# from urllib.parse import quote_plus
# from typing import Dict, Any, List

# # --- Configuration ---
# # You would need to replace this with a real API key from your environment variables.
# # This is for demonstration purposes.
# API_KEY = ""
# API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={API_KEY}"
# DB_NAME = "recipes.db"
# # This list is used to seed the recipe generation. You would expand this list
# # or use a more sophisticated method to generate a million unique prompts.
# RECIPE_PROMPTS = [
#     "A simple weeknight pasta dish.",
#     "A vegetarian curry from India.",
#     "A dessert suitable for a dinner party.",
#     "A quick and easy chicken dish.",
#     "A classic French soup.",
#     "A seafood dish inspired by Mediterranean cuisine.",
#     "A Mexican street food.",
#     "A hearty breakfast recipe with eggs.",
#     "A light and refreshing salad.",
#     "A baked good using chocolate.",
# ]

# def initialize_database():
#     """
#     Initializes the SQLite database and creates the recipes table if it doesn't exist.
#     """
#     conn = sqlite3.connect(DB_NAME)
#     cursor = conn.cursor()
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS recipes (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             recipe_name TEXT NOT NULL,
#             description TEXT,
#             ingredients TEXT,
#             instructions TEXT,
#             image_url TEXT,
#             video_url TEXT,
#             source TEXT
#         )
#     """)
#     conn.commit()
#     conn.close()

# def generate_recipe_with_api(prompt: str) -> Dict[str, Any] | None:
#     """
#     Uses the Gemini API to generate a structured recipe in JSON format.
    
#     Args:
#         prompt (str): A natural language prompt for the recipe.

#     Returns:
#         Dict[str, Any] | None: A dictionary containing the recipe data, or None if the API call fails.
#     """
#     print(f"Generating recipe for: {prompt}")
    
#     # Define the structure for the API's response using a generation config.
#     generation_config = {
#         "responseMimeType": "application/json",
#         "responseSchema": {
#             "type": "OBJECT",
#             "properties": {
#                 "recipe_name": {"type": "STRING"},
#                 "description": {"type": "STRING"},
#                 "ingredients": {"type": "ARRAY", "items": {"type": "STRING"}},
#                 "instructions": {"type": "ARRAY", "items": {"type": "STRING"}}
#             },
#             "propertyOrdering": ["recipe_name", "description", "ingredients", "instructions"]
#         }
#     }

#     payload = {
#         "contents": [
#             {
#                 "parts": [
#                     {
#                         "text": f"Provide a recipe based on the prompt: '{prompt}'. "
#                                 f"The response should be a JSON object with the keys: "
#                                 f"'recipe_name', 'description', 'ingredients' (list of strings), and "
#                                 f"'instructions' (list of strings). Do not include any other text."
#                     }
#                 ]
#             }
#         ],
#         "generationConfig": generation_config
#     }
    
#     try:
#         response = requests.post(API_URL, json=payload, timeout=30)
#         response.raise_for_status()
        
#         # Parse the JSON response from the API
#         api_response_text = response.json()['candidates'][0]['content']['parts'][0]['text']
#         recipe_data = json.loads(api_response_text)
#         return recipe_data
        
#     except (requests.exceptions.RequestException, KeyError, json.JSONDecodeError) as e:
#         print(f"Error generating recipe for '{prompt}': {e}")
#         return None

# def get_image_url(recipe_name: str) -> str:
#     """
#     Generates a placeholder image URL based on the recipe name.
#     In a real application, this would be a link to a real hosted image.
#     """
#     # Use urllib.parse.quote_plus to safely format the text for a URL
#     search_query = quote_plus(recipe_name)
#     # This uses a free service for generating image placeholders with text
#     return f"https://placehold.co/600x400/E868A7/FFFFFF?text={search_query}"

# def get_video_url(recipe_name: str) -> str:
#     """
#     Generates a YouTube search URL for the recipe.
#     In a real application, you would use the YouTube Data API to find a specific video.
#     """
#     search_query = quote_plus(f"how to make {recipe_name}")
#     return f"https://www.youtube.com/results?search_query={search_query}"

# def insert_recipe_into_db(recipe: Dict[str, Any]):
#     """
#     Inserts a single recipe dictionary into the database.
    
#     Args:
#         recipe (Dict[str, Any]): The recipe dictionary to insert.
#     """
#     conn = sqlite3.connect(DB_NAME)
#     cursor = conn.cursor()
#     try:
#         cursor.execute("""
#             INSERT INTO recipes (recipe_name, description, ingredients, instructions, image_url, video_url, source)
#             VALUES (?, ?, ?, ?, ?, ?, ?)
#         """, (
#             recipe['recipe_name'],
#             recipe['description'],
#             json.dumps(recipe['ingredients']), # Store lists as JSON strings
#             json.dumps(recipe['instructions']),
#             recipe['image_url'],
#             recipe['video_url'],
#             'Gemini API'
#         ))
#         conn.commit()
#         print(f"Successfully inserted recipe: {recipe['recipe_name']}")
#     except sqlite3.Error as e:
#         print(f"Database error while inserting recipe {recipe['recipe_name']}: {e}")
#     finally:
#         conn.close()

# def main():
#     """
#     Main function to orchestrate the dataset generation process.
#     """
#     initialize_database()
    
#     # Loop to generate recipes. The number of iterations determines the dataset size.
#     # We will run this for a small number of recipes for demonstration.
#     # To generate 1,000,000 recipes, this loop would need to run for that many iterations
#     # with a very diverse set of prompts.
#     for prompt in RECIPE_PROMPTS:
#         # Generate the core recipe content using the API
#         recipe_data = generate_recipe_with_api(prompt)
        
#         if recipe_data:
#             # Add image and video URLs to the recipe data
#             recipe_data['image_url'] = get_image_url(recipe_data['recipe_name'])
#             recipe_data['video_url'] = get_video_url(recipe_data['recipe_name'])
            
#             # Insert the complete recipe into the database
#             insert_recipe_into_db(recipe_data)
            
#             # Pause to avoid hitting API rate limits
#             time.sleep(1)
            
#     print("\nDataset generation process complete.")
#     print(f"You can now query the '{DB_NAME}' database.")

# if __name__ == "__main__":
#     main()
