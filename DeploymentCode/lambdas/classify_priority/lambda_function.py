import json
import os
import time
from typing import Dict, Any, List
from datetime import datetime, timezone

# --- AWS SDK (Boto3) ---
# This should be in your container's requirements.txt
import boto3

# --- Your Custom Model Logic (UNCHANGED) ---
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Model loading remains the same (happens once at cold start)
print("Loading priority classification models...")
base_path = "./models" 
priority_classes = ['high', 'medium', 'low']
models = {p: AutoModelForSequenceClassification.from_pretrained(os.path.join(base_path, p)) for p in priority_classes}
tokenizers = {p: AutoTokenizer.from_pretrained(os.path.join(base_path, p)) for p in priority_classes}
print("Priority models loaded successfully.")
# --- End Model Loading ---

# --- AWS Clients (Global) ---
dynamodb = boto3.resource('dynamodb')
eventbridge_client = boto3.client('events')

# --- Configuration (from Environment Variables) ---
ISSUES_TABLE_NAME = os.environ.get("ISSUES_TABLE_NAME")


# --- Your Prediction Logic (UNCHANGED) ---
def get_priority_prediction(text: str) -> Dict[str, Any]:
    """Calculates scores for a single text and returns the top priority and scores."""
    scores = {p: get_priority_score(text, p) for p in priority_classes}
    predicted_priority = max(scores, key=scores.get)
    return {'priority': predicted_priority, 'scores': scores}

def get_priority_score(text: str, priority: str) -> float:
    """Gets a single score for a given priority class."""
    tokenizer = tokenizers[priority]
    model = models[priority]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs[0][1].item()
# --- End Your Prediction Logic ---


# --- NEW: Event-Driven Lambda Handler ---
def lambda_handler(event: Dict[str, Any], context: Any):
    """
    Main Lambda handler. Triggered by EventBridge with a BATCH of *classified* issues.
    """
    print("Classify Priority event received:")
    print(json.dumps(event))

    if not ISSUES_TABLE_NAME:
        print("FATAL: ISSUES_TABLE_NAME environment variable is not set.")
        return {'statusCode': 500, 'body': 'Internal configuration error'}

    # 1. Parse the event
    try:
        # This rule will listen for 'issue.batch.classified'
        if event.get("source") != "github.issues" or event.get("detail-type") != "issue.batch.classified":
            print(f"Ignoring event from unknown source: {event.get('source')}")
            return
            
        repo_name = event['detail'].get('repo_name')
        
        # This list only contains NON-SPAM, ENRICHED, CLASSIFIED issues
        issues_list = event['detail'].get('issues', [])
        
        if not repo_name or not issues_list:
            print("Event detail is missing repo_name or issues list. Nothing to do.")
            return

    except KeyError as e:
        print(f"Error parsing event: {e}")
        return
        
    print(f"Starting priority classification for {len(issues_list)} issues from repo: {repo_name}...")
    
    table = dynamodb.Table(ISSUES_TABLE_NAME)
    
    # Use DynamoDB Batch Writer for maximum efficiency
    with table.batch_writer() as batch:
        for issue in issues_list:
            try:
                # 2. Extract data for this issue
                issue_id = issue['issue_id']
                # Use the ENRICHED title/body
                title = issue.get('enriched_title', issue.get('title', '')).strip()
                body = issue.get('enriched_body', issue.get('body', '')).strip()
                
                text_content = f"{title}\n\n{body}".strip()
                
                # 3. Classify the priority using your model
                if not text_content:
                    prediction = {"priority": "low", "scores": {"high": 0.0, "medium": 0.0, "low": 1.0}}
                else:
                    prediction = get_priority_prediction(text_content)
                
                priority = prediction.get("priority", "low") # Get the top category
                print(f"  - Result for {repo_name}#{issue_id}: {priority}")

                # 4. Update the item in DynamoDB
                batch.update_item(
                    Key={
                        'repo_name': repo_name,
                        'issue_id': issue_id
                    },
                    UpdateExpression="SET priority = :p, priority_scores = :ps, last_updated_pipeline = :lu",
                    ExpressionAttributeValues={
                        ':p': priority,
                        ':ps': json.dumps(prediction.get("scores", {})), # Store all scores as a JSON string
                        ':lu': datetime.now(timezone.utc).isoformat()
                    }
                )
                
                # Pass the classification to the next event
                issue['priority'] = priority

            except Exception as e:
                print(f"ERROR: Failed to process issue {issue.get('issue_id')}: {e}")
                continue # Skip this issue and continue with the next
    
    print(f"Successfully processed and updated {len(issues_list)} issues.")
    
    # --- 5. SEND EVENT FOR NEXT LAMBDA IN THE CHAIN (e.g., classify_assignee) ---
    print(f"Sending {len(issues_list)} prioritized issues to the next step...")
    try:
        eventbridge_client.put_events(
            Entries=[
                {
                    'Source': 'github.issues',
                    'DetailType': 'issue.batch.prioritized', # This is the NEW event type
                    'EventBusName': 'default',
                    'Detail': json.dumps({
                        'repo_name': repo_name,
                        'issues': issues_list # Pass the prioritized issues
                    })
                }
            ]
        )
    except Exception as e:
        print(f"FATAL: Failed to send 'issue.batch.prioritized' event: {e}")

    return {'statusCode': 200, 'body': 'Batch priority classification complete.'}