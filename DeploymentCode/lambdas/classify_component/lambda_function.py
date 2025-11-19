import json
import os
import boto3
from typing import Dict, Any, List
from datetime import datetime, timezone
from decimal import Decimal

# --- AWS Clients ---
dynamodb = boto3.resource('dynamodb')
eventbridge_client = boto3.client('events')
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1") # Keeping us-east-1 for Bedrock availability

# --- Configuration ---
ISSUES_TABLE_NAME = os.environ.get("ISSUES_TABLE_NAME")
MODEL_ID = "mistral.mistral-7b-instruct-v0:2"

# --- LLM Logic: Component Classification ---
FEW_SHOT_COMPONENT = """<s>[INST]
You are a Senior Technical Architect. Classify the GitHub issue into exactly ONE of the following components:
- "frontend" (UI, CSS, React, Views, client-side)
- "backend" (API, Logic, Server, Controllers, Models)
- "database" (SQL, Schemas, Migrations, Data Integrity)
- "devops" (CI/CD, Docker, AWS, Infrastructure, Deployment)
- "mobile" (iOS, Android, React Native)
- "documentation" (Readme, Guides, typos)
- "security" (Auth, Vulnerabilities, Permissions)

Return **only** a single JSON object with these keys:
- component: string (one of the above)
- confidence: float (0.0 to 1.0)

Examples:
Title: "Button misalignment on mobile view"
Body: "The submit button overlaps with the footer on iPhone 12."
JSON:
{"component": "frontend", "confidence": 0.95}

Title: "API returns 500 error on login"
Body: "The /auth/login endpoint fails when password contains special characters."
JSON:
{"component": "backend", "confidence": 0.90}

Now classify this issue.
[/INST]"""

def build_prompt(title: str, body: str) -> str:
    return FEW_SHOT_COMPONENT + f"\n<INST>\nTitle: {title}\nBody: {body}\nJSON:\n"

def extract_json_from_text(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            pass
    return {"component": "backend", "confidence": 0.0}

def invoke_bedrock_component(title: str, body: str) -> dict:
    prompt = build_prompt(title, body) + "</INST>"
    native_request = {
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.0,
    }

    try:
        response = bedrock_client.invoke_model(modelId=MODEL_ID, body=json.dumps(native_request))
        raw_body = response['body'].read().decode('utf-8')
        resp_obj = json.loads(raw_body)
        
        text_output = ""
        outputs = resp_obj.get("outputs")
        if isinstance(outputs, list) and outputs:
            text_output = outputs[0].get("text", "")
        
        result = extract_json_from_text(text_output)
        
        valid_components = ["frontend", "backend", "database", "devops", "mobile", "documentation", "security"]
        comp = str(result.get("component", "backend")).lower()
        if comp not in valid_components:
            comp = "backend" 
            
        return {
            "component": comp,
            "confidence": float(result.get("confidence", 0.5))
        }

    except Exception as e:
        print(f"LLM Error: {e}")
        return {"component": "backend", "confidence": 0.0}

# --- Main Lambda Handler ---
def lambda_handler(event: Dict[str, Any], context: Any):
    print("Event received (Component Classifier):", json.dumps(event))

    # [CRITICAL FIX 1] MATCH THE SCREENSHOT
    # Your EventBridge rule sends 'issue.batch.type_and_priority_classified'.
    # We must accept that specific event.
    incoming_type = event.get("detail-type")
    
    if incoming_type != "issue.batch.type_and_priority_classified":
        # [CRITICAL FIX 2] LOOP PREVENTION
        # If this function somehow hears its OWN output ('issue.batch.component_classified'), stop immediately.
        if incoming_type == "issue.batch.component_classified":
             print("Loop prevention: Ignoring my own output event.")
             return {'statusCode': 200, 'body': 'Loop stopped.'}
             
        print(f"Skipping: Expected 'issue.batch.type_and_priority_classified', got '{incoming_type}'")
        return {'statusCode': 200, 'body': 'Skipped.'}

    if not ISSUES_TABLE_NAME:
        return {'statusCode': 500, 'body': 'Configuration Error: ISSUES_TABLE_NAME missing'}

    try:
        detail = event.get('detail', {})
        repo_name = detail.get('repo_name')
        issues_list = detail.get('issues', [])
        
        if not repo_name or not issues_list:
            print("No issues to process.")
            return {'statusCode': 200, 'body': 'Skipped'}
            
    except Exception:
        return {'statusCode': 400, 'body': 'Bad Event Format'}
        
    print(f"Classifying components for {len(issues_list)} issues in {repo_name}")
    table = dynamodb.Table(ISSUES_TABLE_NAME)
    processed_issues = []
    
    for issue in issues_list:
        try:
            issue_id = issue['issue_id']
            title = issue.get('enriched_title', issue.get('title', ''))
            body = issue.get('enriched_body', issue.get('body', ''))
            
            # Call Bedrock
            print(f"Classifying component for issue #{issue_id}...")
            result = invoke_bedrock_component(title, body)
            
            # Update DynamoDB
            table.update_item(
                Key={'repo_name': repo_name, 'issue_id': issue_id},
                UpdateExpression="SET component = :c, component_conf = :cc, last_updated_pipeline = :lu",
                ExpressionAttributeValues={
                    ':c': result['component'],
                    ':cc': Decimal(str(result['confidence'])),
                    ':lu': datetime.now(timezone.utc).isoformat()
                }
            )
            
            # Pass along data to next step
            issue['component'] = result['component']
            processed_issues.append(issue)

        except Exception as e:
            print(f"Error processing issue #{issue.get('issue_id')}: {e}")
            continue

    # Trigger Final Step: classify_assignee
    if processed_issues:
        print(f"Triggering classify_assignee for {len(processed_issues)} issues...")
        try:
            eventbridge_client.put_events(
                Entries=[
                    {
                        'Source': 'github.issues',
                        'DetailType': 'issue.batch.component_classified', 
                        'EventBusName': 'default',
                        'Detail': json.dumps({
                            'repo_name': repo_name,
                            'issues': processed_issues
                        })
                    }
                ]
            )
        except Exception as e:
            print(f"Failed to trigger next step: {e}")
            raise e

    return {'statusCode': 200, 'body': 'Component classification complete.'}