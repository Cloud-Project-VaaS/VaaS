#!/usr/bin/env python3
"""
Script to classify GitHub issues and PRs as spam or not spam using Gemini 2.5 Pro.

Features:
•  Uses Gemini 2.5 Pro for classification
•  Adds a "spam" key to each item with values:
◦  true - if spam
◦  false - if not spam
◦  "uncertain" - if the model can't confidently classify
•  Considers Dependabot and automated changes as legitimate (not spam)
•  Shows progress and summary statistics
•  Default output: spam_classified.json
"""

import json
import os
import sys
import time

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai package not found.")
    print("Install it with: pip install google-generativeai")
    sys.exit(1)


def classify_spam(item, api_key):
    """
    Classify an item as spam or not spam using Gemini 2.5 Pro.
    
    Args:
        item: Dictionary containing title, body, and other fields
        api_key: Google API key
    
    Returns:
        True for spam, False for not spam, "uncertain" if can't classify
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Prepare context for classification
    title = item.get('title', '')
    body = item.get('body', '')
    item_type = item.get('item_type', 'Unknown')
    
    prompt = f"""You are a spam classifier for GitHub issues and pull requests.

Analyze the following {item_type} and determine if it is spam.

Spam indicators include:
- Promotional content or advertisements
- Unrelated links or marketing
- Gibberish or nonsensical content
- Obvious bot-generated spam
- Malicious links or phishing attempts

Legitimate content includes:
- Bug reports
- Feature requests
- Valid pull requests
- Technical discussions
- Dependency updates (like Dependabot)
- Automated but legitimate changes (like code formatting)

Title: {title}

Body: {body[:1000]}

Respond with ONLY one of these exact words:
- "spam" if this is spam
- "not_spam" if this is legitimate
- "uncertain" if you cannot confidently determine

Your response:"""
    
    try:
        response = model.generate_content(prompt)
        result = response.text.strip().lower()
        
        # Parse the response
        if 'spam' in result and 'not_spam' not in result:
            return True
        elif 'not_spam' in result or 'not spam' in result:
            return False
        elif 'uncertain' in result:
            return "uncertain"
        else:
            # If response is unclear, mark as uncertain
            return "uncertain"
    except Exception as e:
        print(f"Error classifying item {item.get('number', 'unknown')}: {e}")
        return "uncertain"


def process_items(input_file, output_file, api_key):
    """
    Process JSON file and classify items as spam or not spam.
    
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
        items = data.get('items') or data.get('data') or [data]
    else:
        items = [data]
    
    classified_count = 0
    spam_count = 0
    uncertain_count = 0
    
    for idx, item in enumerate(items, 1):
        print(f"[{idx}/{len(items)}] Classifying {item.get('item_type', 'item')} #{item.get('number', 'N/A')}...")
        
        # Classify the item
        spam_result = classify_spam(item, api_key)
        item['spam'] = spam_result
        
        classified_count += 1
        if spam_result == True:
            spam_count += 1
        elif spam_result == "uncertain":
            uncertain_count += 1
        
        # Rate limiting: 2 requests per minute = 30 seconds between requests
        if idx < len(items):  # Don't wait after the last item
            print(f"  Waiting 30 seconds (rate limit: 2 req/min)...")
            time.sleep(6)
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nClassification complete!")
    print(f"Total items: {classified_count}")
    print(f"Spam: {spam_count}")
    print(f"Not spam: {classified_count - spam_count - uncertain_count}")
    print(f"Uncertain: {uncertain_count}")
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
    output_file = 'spam_classified.json'
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print("Starting spam classification...\n")
    
    process_items(input_file, output_file, api_key)


if __name__ == '__main__':
    main()