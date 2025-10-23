import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        api_key = "YOUR_API_KEY" 
        if api_key == "YOUR_API_KEY":
            raise ValueError("API key not found. Please set the GOOGLE_API_KEY environment variable or replace 'YOUR_API_KEY' in the script.")
    
    genai.configure(api_key=api_key)

except ValueError as e:
    print(e)
    exit()

def list_generative_models():
    print("✨ Available Models: ✨")
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"- {model.name}")

if __name__ == "__main__":
    list_generative_models()