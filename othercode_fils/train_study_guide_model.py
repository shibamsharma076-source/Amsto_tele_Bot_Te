# train_study_guide_model.py

import pandas as pd
import os

def generate_study_guide_dataset(file_name='study_guide_dataset.csv'):
    """
    Generates a simple CSV dataset for study guidance based on job roles.
    """
    data = {
        'job_role': [
            "Software Engineer",
            "Data Scientist",
            "Machine Learning Engineer",
            "Web Developer",
            "Cybersecurity Analyst",
            "Product Manager",
            "UX Designer"
        ],
        'study_guidance': [
            "Focus on Data Structures & Algorithms, Object-Oriented Programming, and System Design. Practice coding challenges daily.",
            "Master Statistics, Python (Pandas, NumPy, Scikit-learn), Machine Learning algorithms, and data visualization tools (Matplotlib, Seaborn). SQL is crucial.",
            "Deep dive into Machine Learning (Deep Learning, NLP, Computer Vision), MLOps, Python, and frameworks like TensorFlow/PyTorch. Strong math background is a plus.",
            "Learn HTML, CSS, JavaScript for frontend. For backend, choose Python (Django/Flask), Node.js (Express), or Ruby on Rails. Understand databases (SQL/NoSQL).",
            "Study network security, ethical hacking, cryptography, and security frameworks (NIST, ISO 27001). Certifications like CompTIA Security+, CEH are valuable.",
            "Develop strong communication, leadership, and analytical skills. Understand market research, product lifecycle, and agile methodologies. Business acumen is key.",
            "Learn user research, wireframing, prototyping (Figma, Sketch, Adobe XD), usability testing, and information architecture. Empathy is vital."
        ],
        'application_process': [
            "Prepare a strong resume highlighting projects. Practice behavioral and technical interview questions. Network on LinkedIn. Apply through company career pages and job boards.",
            "Showcase a portfolio of data science projects. Tailor your resume to job descriptions. Prepare for case studies and statistical/ML concept interviews.",
            "Build end-to-end ML projects. Focus on algorithm implementation and deployment. Be ready for deep technical discussions on models and infrastructure.",
            "Create a diverse portfolio of web projects. Understand version control (Git). Be prepared to discuss frameworks and responsive design.",
            "Highlight certifications and practical experience (e.g., CTF participation). Prepare for scenario-based questions and technical assessments.",
            "Demonstrate problem-solving and strategic thinking. Prepare for product sense and execution interviews. Showcase leadership and collaboration.",
            "Present a strong portfolio of design projects. Be ready to discuss your design process, user-centered approach, and tools."
        ],
        'universities': [
            "Stanford, MIT, UC Berkeley, Carnegie Mellon, University of Waterloo",
            "Stanford, CMU, UC Berkeley, Columbia, Georgia Tech",
            "Stanford, CMU, MIT, Georgia Tech, University of Washington",
            "Any reputable CS program, specialized bootcamps, online courses (Coursera, Udemy)",
            "Purdue, Carnegie Mellon, Georgia Tech, University of Maryland",
            "Harvard Business School, Stanford GSB, Wharton, INSEAD",
            "Carnegie Mellon, Georgia Tech, University of Washington, Pratt Institute"
        ]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(file_name, index=False)
    print(f"Generated '{file_name}' successfully.")

if __name__ == '__main__':
    generate_study_guide_dataset()
    print("Please ensure 'pandas' is installed: pip install pandas")
