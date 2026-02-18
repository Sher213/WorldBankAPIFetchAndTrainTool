import requests
import urllib.parse
from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

client = genai.Client()

def extract_abstract(abstract_field):
    """
    Handles messy abstracts returned by the API.
    Can be a string, a dict with 'cdata', or a list.
    """
    if isinstance(abstract_field, str):
        return abstract_field
    elif isinstance(abstract_field, dict):
        return abstract_field.get("cdata")
    elif isinstance(abstract_field, list):
        for item in abstract_field:
            text = extract_abstract(item)
            if text:
                return text
    return None

def main():
    while True:
        term = input("Enter a search topic: ").strip()
        city = input("Enter a city (optional): ").strip()
        encoded_term = urllib.parse.quote(f"{term} {city}" if city else term)

        url = f"https://search.worldbank.org/api/v3/wds?format=json&qterm={encoded_term}&fl=abstracts"
        
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()

        documents = data.get("documents")
        if not documents:
            print(f"No abstracts found for '{term}'")
            return

        results_json = {}

        print(f"\nFound {len(documents)} documents:\n")
        for doc_id, doc in documents.items():
            results_json[doc_id] = doc.get("abstracts")
            print(f"Document ID: {doc_id}")
            print(f"Abstract:\n{doc.get('abstracts')}\n")
            print("-" * 40 + "\n")

        ai_question = input("Enter a question for the AI about these abstracts: ").strip()

        response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=f"{ai_question}\n\nHere are the abstracts:\n{results_json}",
        )

        print(response.text)

if __name__ == "__main__":
    main()