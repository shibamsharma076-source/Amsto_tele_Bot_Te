import pandas as pd
import random
from tqdm import tqdm

# Define the number of entries to generate, between 10,000 and 10,500
num_entries = random.randint(10000, 10500)

# Define a more comprehensive list of job roles and difficulties
job_roles = [
    'Software Engineer', 'Data Scientist', 'DevOps Engineer', 
    'Product Manager', 'UX Designer', 'Data Analyst', 
    'Cybersecurity Analyst', 'Backend Developer', 'Frontend Developer'
]
difficulties = ['easy', 'medium', 'hard']

# Create a list to store the data
data = []

print(f"Generating a dataset of approximately {num_entries} interview questions...")
# Use tqdm for a progress bar
for i in tqdm(range(num_entries)):
    role = random.choice(job_roles)
    difficulty = random.choice(difficulties)
    
    # Generate a unique question and answer for each entry
    question = f"Question {i+1} for a {role} role at {difficulty} difficulty."
    answer = f"This is a sample answer to question {i+1}, which covers key concepts for a {role} at the {difficulty} level. It is designed to test fundamental knowledge and problem-solving skills."
    
    data.append({
        'job_role': role,
        'difficulty': difficulty,
        'question': question,
        'answer': answer
    })

# Create a Pandas DataFrame from the generated data
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
output_filename = 'interview_qa_dataset.csv'
df.to_csv(output_filename, index=False)

print(f"\nSuccessfully created an interview Q&A dataset with {len(df)} entries.")
print(f"File saved as '{output_filename}'.")
