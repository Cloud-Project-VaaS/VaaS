# pip install google-generativeai
# export GEMINI_API_KEY='your api key'

import json
import os
import sys
from pathlib import Path

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai package not found.")
    print("Install it with: pip install google-generativeai")
    sys.exit(1)


def enrich_text(text, api_key, prompt_type="title"):
    """
    Enrich text using Gemini 2.5 Flash model.
    
    Args:
        text: The text to enrich
        api_key: Google API key
        prompt_type: Either "title" or "body"
    
    Returns:
        Enriched text
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    if prompt_type == "title":
        prompt = f"""Refactor the following issue title to be clear and concise. Keep it short (under 100 characters). Return ONLY the refactored title without any explanations or additional text:

{text}"""
    else:
        prompt = f"""Refactor the following issue body to be clear, well-structured, and professional. Remove redundant information but keep all important technical details. Format it properly with sections if needed. Return ONLY the refactored body without any explanations, introductions, or additional remarks:

{text}"""
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error enriching text: {e}")
        return text


def enrich_data(input_file, output_file, api_key):
    """
    Process JSON file and enrich section titles and issue bodies.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        api_key: Google API key
    """
    # Read input JSON
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in '{input_file}'.")
        sys.exit(1)
    
    # Process data
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        # If it's a dict, try to find items in common keys
        items = data.get('items') or data.get('data') or [data]
    else:
        items = [data]
    
    enriched_count = 0
    
    for item in items:
        # Only enrich if item type is "Issue"
        if item.get('item_type') == 'Issue' or 'pull_request' not in item:
            # Enrich title
            if 'title' in item:
                print(f"Enriching issue title: {item['title'][:50]}...")
                item['title'] = enrich_text(item['title'], api_key, "title")
                enriched_count += 1
            
            # Enrich body
            if 'body' in item:
                print(f"Enriching issue body...")
                item['body'] = enrich_text(item['body'], api_key, "body")
                enriched_count += 1
    
    # Save enriched data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nEnrichment complete!")
    print(f"Enriched {enriched_count} fields")
    print(f"Output saved to: {output_file}")


def main():
    """Main entry point."""
    # Get API key from environment variable
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Set it with: export GEMINI_API_KEY='your-api-key'")
        sys.exit(1)
    
    # Default file paths
    input_file = 'input.json'
    output_file = 'enriched_output.json'
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print("Starting enrichment process...\n")
    
    enrich_data(input_file, output_file, api_key)


if __name__ == '__main__':
    main()