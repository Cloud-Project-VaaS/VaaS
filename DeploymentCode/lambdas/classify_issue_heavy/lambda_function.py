import json
from typing import Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Model loading remains the same (happens once at cold start)
SAVE_MODEL_PATH = "./model"
tokenizer = AutoTokenizer.from_pretrained(SAVE_MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(SAVE_MODEL_PATH)
model.eval()

def _predict_probabilities(text: str) -> Dict[str, Any]:
    """Runs prediction and returns a dictionary with category and scores."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]

    index_to_category = {0: "Bug", 1: "Enhancement", 2: "Question"}
    scores: Dict[str, float] = {}
    for i in range(probs.shape[0]):
        category = index_to_category.get(i, f"Label_{i}")
        scores[category] = float(probs[i].item())
    
    top_category = max(scores.items(), key=lambda kv: kv[1])[0]

    return {
        "category": top_category,
        "scores": scores,
    }

def lambda_handler(event: List[Dict[str, Any]], context: Any) -> List[Dict[str, Any]]:
    # 'event' is now a LIST of issue objects
    issues_list = event
    results_list = []

    # Loop through each issue in the batch
    for issue in issues_list:
        title = (issue.get("title") or "").strip()
        body = (issue.get("body") or "").strip()
        text_content = f"{title}\n\n{body}".strip()

        if not text_content:
            prediction = {"category": "Question", "scores": {"Bug": 0.0, "Enhancement": 0.0, "Question": 1.0}}
        else:
            prediction = _predict_probabilities(text_content)

        # Add the issue_id to the result for matching later
        prediction["issue_id"] = issue.get("id")
        results_list.append(prediction)
    
    # Return a LIST of results
    return results_list