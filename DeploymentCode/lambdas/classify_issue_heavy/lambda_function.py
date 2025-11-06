import json
import os
import time
from typing import Dict, Any, List
from datetime import datetime, timezone

# --- AWS SDK (Boto3) ---
# This is included in the base Lambda container image,
# but it's good practice to add 'boto3' to your requirements.txt
import boto3

# --- Your Custom Model Logic (UNCHANGED) ---
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Model loading remains the same (happens once at cold start)
# This path is relative to the root of your container
SAVE_MODEL_PATH = "./model"
print(f"Loading tokenizer from {SAVE_MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(SAVE_MODEL_PATH)
print(f"Loading model from {SAVE_MODEL_PATH}...")
model = AutoModelForSequenceClassification.from_pretrained(SAVE_MODEL_PATH)
model.eval()
print("Model loaded successfully.")

# --- AWS Clients (Global) ---
dynamodb = boto3.resource('dynamodb')
eventbridge_client = boto3.client('events')

# --- Configuration (from Environment Variables) ---
ISSUES_TABLE_NAME = os.environ.get("ISSUES_TABLE_NAME")


def _predict_probabilities(text: str) -> Dict[str, Any]:
    """
    Runs prediction and returns a dictionary with category and scores.
    (This is your original, unchanged function)
    """
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

# --- NEW: Event-Driven Lambda Handler ---
def lambda_handler(event: Dict[str, Any], context: Any):
    """
    Main Lambda handler. Triggered by EventBridge with a BATCH of *enriched* issues.
    """
    print("Classify Type event received:")
    print(json.dumps(event))

    if not ISSUES_TABLE_NAME:
        print("FATAL: ISSUES_TABLE_NAME environment variable is not set.")
        return {'statusCode': 500, 'body': 'Internal configuration error'}

    # 1. Parse the event
    try:
        # This rule will listen for 'issue.batch.enriched'
        if event.get("source") != "github.issues" or event.get("detail-type") != "issue.batch.enriched":
            print(f"Ignoring event from unknown source: {event.get('source')}")
            return
            
        repo_name = event['detail'].get('repo_name')
        
        # This list only contains NON-SPAM, ENRICHED issues
        issues_list = event['detail'].get('issues', [])
        
        if not repo_name or not issues_list:
            print("Event detail is missing repo_name or issues list. Nothing to do.")
            return

    except KeyError as e:
        print(f"Error parsing event: {e}")
        return
        
    print(f"Starting issue type classification for {len(issues_list)} issues from repo: {repo_name}...")
    
    table = dynamodb.Table(ISSUES_TABLE_NAME)
    
    # Use DynamoDB Batch Writer for maximum efficiency
    with table.batch_writer() as batch:
        for issue in issues_list:
            try:
                # 2. Extract data for this issue
                issue_id = issue['issue_id']
                # Use the ENRICHED title/body if available, fall back to original
                title = issue.get('enriched_title', issue.get('title', '')).strip()
                body = issue.get('enriched_body', issue.get('body', '')).strip()
                
                text_content = f"{title}\n\n{body}".strip()
                
                # 3. Classify the issue using your model
                if not text_content:
                    prediction = {"category": "Question", "scores": {"Bug": 0.0, "Enhancement": 0.0, "Question": 1.0}}
                else:
                    prediction = _predict_probabilities(text_content)
                
                issue_type = prediction.get("category", "Question") # Get the top category
                print(f"  - Result for {repo_name}#{issue_id}: {issue_type}")

                # 4. Update the item in DynamoDB
                batch.update_item(
                    Key={
                        'repo_name': repo_name,
                        'issue_id': issue_id
                    },
                    UpdateExpression="SET issue_type = :it, issue_type_scores = :its, last_updated_pipeline = :lu",
                    ExpressionAttributeValues={
                        ':it': issue_type,
                        ':its': json.dumps(prediction.get("scores", {})), # Store all scores as a JSON string
                        ':lu': datetime.now(timezone.utc).isoformat()
                    }
                )
                
                # Pass the classification to the next event
                issue['issue_type'] = issue_type

            except Exception as e:
                print(f"ERROR: Failed to process issue {issue.get('issue_id')}: {e}")
                continue # Skip this issue and continue with the next
    
    print(f"Successfully processed and updated {len(issues_list)} issues.")
    
    # --- 5. SEND EVENT FOR NEXT LAMBDA IN THE CHAIN (e.g., classify_priority) ---
    print(f"Sending {len(issues_list)} classified issues to the next step...")
    try:
        eventbridge_client.put_events(
            Entries=[
                {
                    'Source': 'github.issues',
                    'DetailType': 'issue.batch.classified', # This is the NEW event type
                    'EventBusName': 'default',
                    'Detail': json.dumps({
                        'repo_name': repo_name,
                        'issues': issues_list # Pass the classified issues
                    })
                }
            ]
        )
    except Exception as e:
        print(f"FATAL: Failed to send 'issue.batch.classified' event: {e}")

    return {'statusCode': 200, 'body': 'Batch classification complete.'}