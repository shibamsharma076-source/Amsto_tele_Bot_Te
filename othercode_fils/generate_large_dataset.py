import pandas as pd
import itertools
from tqdm import tqdm

# Define programming languages
languages = ["python", "javascript", "java", "c++", "go", "ruby"]

# Define a variety of actions for the prompts
actions = [
    "sort", "reverse", "find the maximum in", "find the minimum in",
    "remove duplicates from", "check if an element exists in",
    "merge", "calculate the sum of", "calculate the average of",
    "filter even numbers from", "filter odd numbers from"
]

# Define different data types to apply the actions to
data_types = ["list", "array", "string", "dictionary", "set"]

# Define the template for the code snippets
# This will be a simple function definition, as the actual implementation would be complex
code_templates = {
    "python": "def {action_snake_case}({data_type_var}):\n    # code to {action} the {data_type}\n    return result",
    "javascript": "function {action_camel_case}({data_type_var}) {{\n    // code to {action} the {data_type}\n    return result;\n}}",
    "java": "public static Object {action_camel_case}({data_type_var}) {{\n    // code to {action} the {data_type}\n    return result;\n}}",
    "c++": "auto {action_snake_case}({data_type_var}) {{\n    // code to {action} the {data_type}\n    return result;\n}}",
    "go": "func {action_camel_case}({data_type_var}) interface{} {{\n    // code to {action} the {data_type}\n    return result\n}}",
    "ruby": "def {action_snake_case}({data_type_var})\n    # code to {action} the {data_type}\n    return result\nend"
}

# Generate the dataset using itertools.product for all combinations
dataset = []
total_combinations = len(languages) * len(actions) * len(data_types)

print(f"Generating a dataset with approximately {total_combinations} entries...")
for lang, action, data_type in tqdm(itertools.product(languages, actions, data_types), total=total_combinations):
    # Create the prompt string
    prompt = f"Write a function to {action} a {data_type} in {lang}"
    
    # Simple formatting for function names and variable names
    action_snake_case = action.replace(" ", "_").replace("the_maximum_in", "max").replace("the_minimum_in", "min")
    
    # Using a more robust way to create camelCase
    action_camel_case = "".join(word.capitalize() for word in action.split())
    if action_camel_case:
        action_camel_case = action_camel_case[0].lower() + action_camel_case[1:]
    
    data_type_var = data_type.lower().replace(" ", "")

    # Create a dictionary of all arguments for formatting
    format_args = {
        "action": action,
        "action_snake_case": action_snake_case,
        "action_camel_case": action_camel_case,
        "data_type": data_type,
        "data_type_var": data_type_var
    }

    # Get the correct code template for the language and fill it in
    try:
        code_snippet = code_templates.get(lang).format(**format_args)
        dataset.append({
            "prompt": prompt,
            "language": lang,
            "code_snippet": code_snippet
        })
    except IndexError as e:
        print(f"Error processing template for language '{lang}' with action '{action}' and data type '{data_type}'.")
        print(f"Template: {code_templates.get(lang)}")
        print(f"Format Args: {format_args}")
        print(f"Error: {e}")

# Create a DataFrame and save it to a CSV file
df = pd.DataFrame(dataset)
output_filename = 'code_generation_dataset.csv'
df.to_csv(output_filename, index=False)

print(f"\nSuccessfully created a dataset with {len(df)} entries.")
print(f"File saved as '{output_filename}'.")
