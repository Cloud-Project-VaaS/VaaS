import json
from typing import Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# IMPORTANT: Load the model from the local './model' directory inside the container
SAVE_MODEL_PATH = "./model"
tokenizer = AutoTokenizer.from_pretrained(SAVE_MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(SAVE_MODEL_PATH)
model.eval()

def _predict_probabilities(text: str) -> Dict[str, float]:
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]

    index_to_category = {0: "Bug", 1: "Enhancement", 2: "Question"}
    scores: Dict[str, float] = {}
    for i in range(probs.shape[0]):
        category = index_to_category.get(i, f"Label_{i}")
        scores[category] = float(probs[i].item())
    return scores

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    title = (event.get("title") or "").strip()
    body = (event.get("body") or "").strip()
    text_content = f"{title}\n\n{body}".strip()

    if not text_content:
        return {"category": "Question", "scores": {"Bug": 0.0, "Enhancement": 0.0, "Question": 1.0}}

    scores = _predict_probabilities(text_content)
    top_category = max(scores.items(), key=lambda kv: kv[1])[0]

    return {
        "category": top_category,
        "scores": scores,
    }