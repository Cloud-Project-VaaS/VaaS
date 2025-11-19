import json
import os
import boto3
import math
from typing import Dict, Any, List
from datetime import datetime, timezone
from botocore.exceptions import ClientError

# --- AWS Clients ---
dynamodb = boto3.resource('dynamodb')
eventbridge_client = boto3.client('events')
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

# --- Configuration ---
ISSUES_TABLE_NAME = os.environ.get("ISSUES_TABLE_NAME")
MODEL_ID = "mistral.mistral-7b-instruct-v0:2"

# --- LLM Logic (From your test.py) ---
FEW_SHOT = """<s>[INST]
You are a GitHub issue classifier. Classify the issue into one of exactly: "bug", "enhancement", or "question".
Also classify the issue priority into exactly one of: "high", "medium", or "low".

Return **only** a single JSON object (no surrounding text, no explanation, no markdown) with exactly these keys:
- category: "bug" | "enhancement" | "question"
- priority: "high" | "medium" | "low"
- confidence_category: float between 0.0 and 1.0 (approximate)
- confidence_priority: float between 0.0 and 1.0 (approximate)

Examples:

Title: "App crashes when opening settings"
Body: "Steps to reproduce: open app -> Settings -> crash. Stack trace: NullPointer..."
JSON:
{"category":"bug","priority":"high","confidence_category":0.95,"confidence_priority":0.90}

Title: "Add dark mode support"
Body: "Dark mode would help reduce eye strain; propose CSS + toggle in settings."
JSON:
{"category":"enhancement","priority":"medium","confidence_category":0.92,"confidence_priority":0.75}

Title: "How to configure API keys?"
Body: "I'm new. How do I add my API key to the CLI and where is it stored?"
JSON:
{"category":"question","priority":"low","confidence_category":0.90,"confidence_priority":0.70}

Now classify the following issue.
[/INST]"""

def build_prompt(title: str, body: str) -> str:
    return FEW_SHOT + f"\n<INST>\nTitle: {title}\nBody: {body}\nJSON:\n"

def _read_response(response) -> dict:
    raw = response.get("body")
    if raw is None:
        raise RuntimeError("No body in Bedrock response.")
    raw_bytes = raw.read()
    try:
        return json.loads(raw_bytes)
    except Exception:
        try:
            return json.loads(raw_bytes.decode("utf-8", errors="ignore"))
        except Exception as e:
            raise RuntimeError(f"Failed to decode model response: {e}")

def _find_json_substring(s: str):
    for start in range(len(s)):
        if s[start] != "{":
            continue
        depth = 0
        for i in range(start, len(s)):
            ch = s[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    substr = s[start:i+1]
                    try:
                        return json.loads(substr)
                    except Exception:
                        break 
    raise ValueError("No valid JSON object found in text")

def extract_json_from_model_text(s: str) -> dict:
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        return _find_json_substring(s)

def _normalize_and_validate(obj: dict) -> dict:
    required = ["category", "priority", "confidence_category", "confidence_priority"]
    for k in required:
        if k not in obj:
            # Fallbacks if LLM misses a key
            if k == "category": obj[k] = "question"
            if k == "priority": obj[k] = "low"
            if "confidence" in k: obj[k] = 0.5

    category = str(obj["category"]).strip().lower()
    if category not in ("bug", "enhancement", "question"):
        category = "question" # Safe fallback

    priority = str(obj["priority"]).strip().lower()
    if priority not in ("high", "medium", "low"):
        priority = "low" # Safe fallback

    return {
        "category": category,
        "priority": priority,
        "confidence_category": float(obj.get("confidence_category", 0.5)),
        "confidence_priority": float(obj.get("confidence_priority", 0.5)),
    }

def invoke_bedrock_classification(title: str, body: str) -> dict:
    prompt = build_prompt(title, body) + "</INST>"
    native_request = {
        "prompt": prompt,
        "max_tokens": 150,
        "temperature": 0.0,
    }

    try:
        response = bedrock_client.invoke_model(modelId=MODEL_ID, body=json.dumps(native_request))
        resp_obj = _read_response(response)
        
        # Extract text (Mistral output format)
        text_output = ""
        outputs = resp_obj.get("outputs")
        if isinstance(outputs, list) and outputs:
            text_output = outputs[0].get("text", "")
        
        parsed = extract_json_from_model_text(text_output)
        return _normalize_and_validate(parsed)

    except Exception as e:
        print(f"LLM Invocation/Parsing Failed: {e}")
        # Return a safe default so the pipeline doesn't break
        return {
            "category": "question", 
            "priority": "low", 
            "confidence_category": 0.0, 
            "confidence_priority": 0.0
        }

# --- Main Lambda Handler ---
def lambda_handler(event: Dict[str, Any], context: Any):
    print("Event received:", json.dumps(event))

    if not ISSUES_TABLE_NAME:
        print("FATAL: ISSUES_TABLE_NAME environment variable is not set.")
        return {'statusCode': 500, 'body': 'Internal configuration error'}

    # 1. Parse Event (Expects issue.batch.enriched)
    try:
        detail = event.get('detail', {})
        repo_name = detail.get('repo_name')
        issues_list = detail.get('issues', [])
        
        if not repo_name or not issues_list:
            print("Event missing repo_name or issues list. Exiting.")
            return {'statusCode': 200, 'body': 'No issues to process.'}
            
    except Exception as e:
        print(f"Error parsing event: {e}")
        return {'statusCode': 400, 'body': 'Bad Request'}
        
    print(f"Processing {len(issues_list)} issues for repo: {repo_name}")
    table = dynamodb.Table(ISSUES_TABLE_NAME)
    processed_issues = []
    
    for issue in issues_list:
        try:
            issue_id = issue['issue_id']
            # Use enriched data if available
            title = issue.get('enriched_title', issue.get('title', ''))
            body = issue.get('enriched_body', issue.get('body', ''))
            
            # 2. Call Bedrock (LLM)
            print(f"Classifying issue #{issue_id}...")
            result = invoke_bedrock_classification(title, body)
            print(f"Result #{issue_id}: {result}")

            # 3. Update DynamoDB
            # We update both Category and Priority at once
            table.update_item(
                Key={
                    'repo_name': repo_name,
                    'issue_id': issue_id
                },
                UpdateExpression="SET issue_type = :cat, priority = :prio, issue_type_conf = :cc, priority_conf = :pc, last_updated_pipeline = :lu",
                ExpressionAttributeValues={
                    ':cat': result['category'],
                    ':prio': result['priority'],
                    ':cc': boto3.dynamodb.types.Decimal(str(result['confidence_category'])),
                    ':pc': boto3.dynamodb.types.Decimal(str(result['confidence_priority'])),
                    ':lu': datetime.now(timezone.utc).isoformat()
                }
            )
            
            # Update issue object for the next event
            issue['issue_type'] = result['category']
            issue['priority'] = result['priority']
            processed_issues.append(issue)

        except Exception as e:
            print(f"Error processing issue #{issue.get('issue_id')}: {e}")
            continue

    # 4. Trigger Next Lambda (trigger_component)
    if processed_issues:
        print(f"Sending {len(processed_issues)} issues to next step: trigger_component")
        try:
            eventbridge_client.put_events(
                Entries=[
                    {
                        'Source': 'github.issues',
                        'DetailType': 'issue.batch.type_and_priority_classified',
                        'EventBusName': 'default',
                        'Detail': json.dumps({
                            'repo_name': repo_name,
                            'issues': processed_issues
                        })
                    }
                ]
            )

        except Exception as e:
            print(f"Failed to send event: {e}")
            raise e

    return {'statusCode': 200, 'body': 'Classification complete.'}