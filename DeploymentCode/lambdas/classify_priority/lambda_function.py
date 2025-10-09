import json
import os
from typing import Dict, Any, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- Model Loading (happens once at cold start) ---
# It's assumed your models are baked into the container image at this path
base_path = "./models" 
priority_classes = ['high', 'medium', 'low']
models = {p: AutoModelForSequenceClassification.from_pretrained(os.path.join(base_path, p)) for p in priority_classes}
tokenizers = {p: AutoTokenizer.from_pretrained(os.path.join(base_path, p)) for p in priority_classes}
# --- End Model Loading ---

def get_priority_prediction(text: str) -> Dict[str, Any]:
    """Calculates scores for a single text and returns the top priority and scores."""
    scores = {p: get_priority_score(text, p) for p in priority_classes}
    predicted_priority = max(scores, key=scores.get)
    return {'priority': predicted_priority, 'scores': scores}

def get_priority_score(text: str, priority: str) -> float:
    """Gets a single score for a given priority class."""
    tokenizer = tokenizers[priority]
    model = models[priority]
    # Increased max_length might be needed for large issues, adjust if necessary
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs[0][1].item()

def lambda_handler(event: List[Dict[str, Any]], context: Any) -> List[Dict[str, Any]]:
    # 'event' is now a LIST of issue objects
    issues_list = event
    results_list = []

    # Loop through each issue in the batch
    for issue in issues_list:
        issue_text = f"{issue.get('title', '')}\n\n{issue.get('body', '')}"
        
        # Get the priority prediction for the current issue
        prediction = get_priority_prediction(issue_text)
        
        # Add the issue_id to the result for matching later
        prediction['issue_id'] = issue.get('id')
        results_list.append(prediction)
    
    # Return a LIST of results
    return results_list