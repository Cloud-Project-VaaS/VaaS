import json
import boto3
import os
import re
import time
from datetime import datetime, timezone

# --- AWS Clients (Global) ---
bedrock_client = boto3.client('bedrock-runtime')
dynamodb = boto3.resource('dynamodb')
eventbridge_client = boto3.client('events') # To trigger the next step

# --- Configuration (from Environment Variables) ---
ISSUES_TABLE_NAME = os.environ.get("ISSUES_TABLE_NAME")
# Use the DeepSeek model as requested
LLM_MODEL_ID = "deepseek.v3-v1:0" 

def enrich_text_with_bedrock(text, prompt_type="title"):
    """
    Enrich text using the DeepSeek model on Bedrock.
    
    Args:
        text: The text to enrich
        prompt_type: Either "title" or "body"
    
    Returns:
        Enriched text
    """
    
    if not text or not text.strip():
        return text # Return empty or original text if no content

    if prompt_type == "title":
        # This prompt is highly constrained to return ONLY the title
        system_prompt = "You are an expert technical editor. Refactor the following issue title to be clear and concise. Keep it under 100 characters. Return ONLY the refactored title, with no other text, pre-amble, or explanations."
        user_prompt = f"Original Title: {text}"
    else:
        # This prompt is for refactoring the body
        system_prompt = "You are an expert technical editor. Refactor the following issue body to be clear, well-structured, and professional. Remove redundant information but keep all important technical details. Format it properly with sections (like '### Steps to Reproduce'). Return ONLY the refactored body, with no other text, pre-amble, or explanations."
        user_prompt = f"Original Body: {text[:15000]}" # Truncate very long bodies
    
    # --- DeepSeek Specific Payload ---
    body_obj = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 2048, # Allow for a longer body
        "temperature": 0.1,
        "top_p": 0.9
    }

    try:
        response = bedrock_client.invoke_model(
            modelId=LLM_MODEL_ID,
            body=json.dumps(body_obj),
            contentType="application/json",
            accept="application/json"
        )
        
        response_body = json.loads(response.get('body').read())
        
        # DeepSeek's response format
        result_text = response_body.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        
        if not result_text:
            # If model returns empty, just return the original text
            return text
            
        return result_text

    except Exception as e:
        print(f"Error calling Bedrock (DeepSeek) model: {e}. Returning original text.")
        return text # Return original text on failure

def lambda_handler(event, context):
    """
    Main Lambda handler. Triggered by EventBridge with a BATCH of *non-spam* issues.
    """
    print("Metadata Enrichment event received:")
    print(json.dumps(event))
    
    if not ISSUES_TABLE_NAME:
        print("FATAL: ISSUES_TABLE_NAME environment variable is not set.")
        return {'statusCode': 500, 'body': 'Internal configuration error'}

    # 1. Parse the event
    try:
        # This rule will listen for 'issue.batch.spam_classified'
        if event.get("source") != "github.issues" or event.get("detail-type") != "issue.batch.spam_classified":
            print(f"Ignoring event from unknown source: {event.get('source')}")
            return
            
        repo_name = event['detail'].get('repo_name')
        
        # This list only contains NON-SPAM issues, pre-filtered by 'classify_spam'
        issues_list = event['detail'].get('issues', [])
        
        if not repo_name or not issues_list:
            print("Event detail is missing repo_name or issues list. Nothing to do.")
            return

    except KeyError as e:
        print(f"Error parsing event: {e}")
        return
        
    print(f"Starting metadata enrichment for {len(issues_list)} issues from repo: {repo_name}...")
    
    table = dynamodb.Table(ISSUES_TABLE_NAME)
    
    # Use DynamoDB Batch Writer for maximum efficiency
    with table.batch_writer() as batch:
        for issue in issues_list:
            try:
                # 2. Extract data for this issue
                issue_id = issue['issue_id']
                original_title = issue.get('title', '')
                original_body = issue.get('body', '')
                
                # 3. Enrich the text
                print(f"  - Enriching {repo_name}#{issue_id}...")
                enriched_title = enrich_text_with_bedrock(original_title, "title")
                # Simple rate limit to not overwhelm Bedrock
                time.sleep(1) 
                
                enriched_body = enrich_text_with_bedrock(original_body, "body")
                # Simple rate limit to not overwhelm Bedrock
                time.sleep(1) 

                # 4. Update the item in DynamoDB
                batch.update_item(
                    Key={
                        'repo_name': repo_name,
                        'issue_id': issue_id
                    },
                    UpdateExpression="SET enriched_title = :et, enriched_body = :eb, last_updated_pipeline = :lu",
                    ExpressionAttributeValues={
                        ':et': enriched_title,
                        ':eb': enriched_body,
                        ':lu': datetime.now(timezone.utc).isoformat()
                    }
                )
                
                # We'll pass the enriched data to the next event
                issue['title'] = enriched_title
                issue['body'] = enriched_body

            except Exception as e:
                print(f"ERROR: Failed to process issue {issue.get('issue_id')}: {e}")
                continue # Skip this issue and continue with the next
    
    print(f"Successfully processed and updated {len(issues_list)} issues.")
    
    # --- 5. SEND EVENT FOR NEXT LAMBDA IN THE CHAIN (e.g., classify_type) ---
    print(f"Sending {len(issues_list)} enriched issues to the next step...")
    try:
        eventbridge_client.put_events(
            Entries=[
                {
                    'Source': 'github.issues',
                    'DetailType': 'issue.batch.enriched', # This is the NEW event type
                    'EventBusName': 'default',
                    'Detail': json.dumps({
                        'repo_name': repo_name,
                        'issues': issues_list # Pass the enriched issues
                    })
                }
            ]
        )
    except Exception as e:
        print(f"FATAL: Failed to send 'issue.batch.enriched' event: {e}")
        # We don't fail the whole function, just log the error

    return {'statusCode': 200, 'body': 'Batch enrichment complete.'}