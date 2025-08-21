import csv
import random
from tqdm import tqdm

famous_phrases = [
    "The quick brown fox jumps over the lazy dog.",
    "To be or not to be, that is the question.",
    "The quick brown fox jumps over the lazy dog.",
    "To be or not to be, that is the question.",
    "A journey of a thousand miles begins with a single step.",
    "The only thing we have to fear is fear itself.",
    "The future belongs to those who believe in the beauty of their dreams.",
    "Genius is one percent inspiration and ninety-nine percent perspiration.",
    "In the beginning God created the heavens and the earth.",
    "All happy families are alike; each unhappy family is unhappy in its own way.",
    "It was the best of times, it was the worst of times.",
    "I have not failed. I've just found 10,000 ways that won't work.",
    "The first rule of Fight Club is: you do not talk about Fight Club.",
    "The greatest glory in living lies not in never falling, but in rising every time we fall.",
    "The only impossible journey is the one you never begin.",
    "Life is what happens when you're busy making other plans.",
    "The best way to predict the future is to create it.",
    "Success is not final; failure is not fatal: it is the courage to continue that counts.",
    "Believe you can and you're halfway there.",
    "Our lives begin to end the day we become silent about things that matter.",
    "The only true wisdom is in knowing you know nothing.",
    "It is not a lack of love, but a lack of friendship that makes unhappy marriages."
]

output_filename = 'plagiarism_corpus.csv'
num_additional_phrases = random.randint(1_000_000, 100_000_000)

with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['phrase', 'source', 'url'], quoting=csv.QUOTE_ALL)
    writer.writeheader()
    # Write famous phrases first
    for phrase in famous_phrases:
        url = f"https://example.com/source/{phrase.replace(' ', '-').lower()[:20]}"
        writer.writerow({
            'phrase': phrase,
            'source': "Famous Quote Library",
            'url': url
        })
    # Write generated phrases in a loop
    print(f"Generating {num_additional_phrases} additional phrases...")
    for i in tqdm(range(num_additional_phrases)):
        phrase = f"This is a sample phrase number {i+1} for the local plagiarism database to test the system performance."
        source = f"Generated Source {i+1}"
        url = f"https://generated-content.com/article/{i+1}"
        writer.writerow({
            'phrase': phrase,
            'source': source,
            'url': url
        })

print(f"\nSuccessfully created a plagiarism dataset with {num_additional_phrases + len(famous_phrases)} entries.")
print(f"File saved as '{output_filename}'.")