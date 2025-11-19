import json
import boto3
import os
import re
import time
from datetime import datetime, timezone
from botocore.config import Config
import concurrent.futures # --- FIX 1: Import for parallelism ---

# --- Configure boto3 with shorter timeouts ---
CLIENT_CONFIG = Config(
    read_timeout=90,  # Give up on a hung call after 90 seconds
    connect_timeout=10
)

# --- AWS Clients (Global) ---
bedrock_client = boto3.client('bedrock-runtime', config=CLIENT_CONFIG) 
dynamodb = boto3.resource('dynamodb')
eventbridge_client = boto3.client('events') 

# --- Configuration (from Environment Variables) ---
ISSUES_TABLE_NAME = os.environ.get("ISSUES_TABLE_NAME")

# --- FIX 2: Switched to the much faster Llama 3 8B model ---
LLM_MODEL_ID = "meta.llama3-8b-instruct-v1:0" 

LLM_MAX_RETRIES = 2
LLM_RETRY_BACKOFF_FACTOR = 3

def enrich_text_with_bedrock(text, prompt_type="title"):
    """
    Enrich text using the Llama 3 8B model on Bedrock.
    """
    
    if not text or not text.strip():
        return text 

    if prompt_type == "title":
        prompt = f"""You are an expert technical editor. Refactor the following issue title to be clear and concise. Keep it under 100 characters. Return ONLY the refactored title, with no other text, pre-amble, or explanations.

Original Title: {text}
Refactored Title:"""
    else:
        prompt = f"""You are an expert technical editor. Refactor the following issue body to be clear, well-structured, and professional. Remove redundant information but keep all important technical details. Format it properly with sections (like '### Steps to Reproduce'). Return ONLY the refactored body, with no other text, pre-amble, or explanations.

Original Body:
{text[:15000]}
Refactored Body:"""
    
    # --- Llama 3 8B Specific Payload ---
    body_obj = {
        "prompt": prompt,
        "max_gen_len": 2048,
        "temperature": 0.1,
        "top_p": 0.9
    }

    # --- Added retry loop for Bedrock calls ---
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            print(f"    - Calling Bedrock for {prompt_type} (Attempt {attempt + 1})...")
            response = bedrock_client.invoke_model(
                modelId=LLM_MODEL_ID,
                body=json.dumps(body_obj),
                contentType="application/json",
                accept="application/json"
            )
            
            response_body = json.loads(response.get('body').read())
            
            # Llama 3's response format
            result_text = response_body.get('generation', '').strip()
            
            if not result_text:
                print(f"    - Bedrock returned empty. Using original text.")
                return text
                
            print(f"    - Bedrock call successful.")
            return result_text

        except Exception as e:
            print(f"Error calling Bedrock (Llama 3) model on attempt {attempt+1}: {e}.")
            if attempt < LLM_MAX_RETRIES:
                print(f"Sleeping for {LLM_RETRY_BACKOFF_FACTOR} seconds before retry...")
                time.sleep(LLM_RETRY_BACKOFF_FACTOR)
    
    print(f"Failed to enrich text after {LLM_MAX_RETRIES+1} attempts. Returning original text.")
    return text

# --- FIX 3: New helper function to be run in parallel ---
def process_and_enrich_issue(issue, repo_name):
    """
    Processes a single issue by enriching title and body.
    This function is designed to be run in a separate thread.
    """
    try:
        issue_id = issue['issue_id']
        original_title = issue.get('title', '')
        original_body = issue.get('body', '')
        
        print(f"  - Starting parallel enrichment for {repo_name}#{issue_id}...")
        
        # These two calls will run one after the other, but
        # this entire function will run in parallel with others.
        enriched_title = enrich_text_with_bedrock(original_title, "title")
        enriched_body = enrich_text_with_bedrock(original_body, "body")
        
        print(f"  - Finished parallel enrichment for {repo_name}#{issue_id}.")
        
        # Update the issue object for the next event
        issue['title'] = enriched_title
        issue['body'] = enriched_body
        
        # Return the enriched issue AND the item for DynamoDB
        dynamo_item = {
            'Key': {
                'repo_name': repo_name,
                'issue_id': issue_id
            },
            'UpdateExpression': "SET enriched_title = :et, enriched_body = :eb, last_updated_pipeline = :lu",
            'ExpressionAttributeValues': {
                ':et': enriched_title,
                ':eb': enriched_body,
                ':lu': datetime.now(timezone.utc).isoformat()
            }
        }
        
        return issue, dynamo_item

    except Exception as e:
        print(f"ERROR: Thread for issue {issue.get('issue_id')} failed: {e}")
        # Return the original issue and no Dynamo item
        return issue, None

# --- FIX 4: Rewritten lambda_handler for parallelism ---
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
        if event.get("source") != "github.issues" or event.get("detail-type") != "issue.batch.spam_classified":
            print(f"Ignoring event from unknown source: {event.get('source')}")
            return
            
        repo_name = event['detail'].get('repo_name')
        issues_list = event['detail'].get('issues', [])
        
        if not repo_name or not issues_list:
            print("Event detail is missing repo_name or issues list. Nothing to do.")
            return

    except KeyError as e:
        print(f"Error parsing event: {e}")
        return
        
    print(f"Starting parallel metadata enrichment for {len(issues_list)} issues from repo: {repo_name}...")
    
    enriched_issues_for_next_event = []
    dynamo_update_items = []

    # 2. Run all enrichment tasks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Create a future (task) for each issue
        futures = {executor.submit(process_and_enrich_issue, issue, repo_name): issue for issue in issues_list}
        
        for future in concurrent.futures.as_completed(futures):
            try:
                # Get the result from the completed thread
                enriched_issue, dynamo_item = future.result()
                
                enriched_issues_for_next_event.append(enriched_issue)
                if dynamo_item:
                    dynamo_update_items.append(dynamo_item)
                    
            except Exception as e:
                original_issue = futures[future]
                print(f"ERROR: Future for issue {original_issue.get('issue_id')} failed: {e}")
                enriched_issues_for_next_event.append(original_issue) # Add original issue

    print(f"All {len(issues_list)} enrichment threads have completed.")

    # 3. Update DynamoDB in one batch
    if dynamo_update_items:
        print(f"Updating {len(dynamo_update_items)} items in DynamoDB...")
        try:
            table = dynamodb.Table(ISSUES_TABLE_NAME)
            with table.batch_writer() as batch:
                for item in dynamo_update_items:
                    # batch_writer doesn't support UpdateExpression. We must use put_item
                    # This is a problem. Let's revert to individual updates.
                    # This is a known limitation. We MUST use update_item.
                    # We will do this serially *after* the parallel part.
                    
                    # --- REVISED PLAN: Update items serially ---
                    # The parallel part (Bedrock) is still the main bottleneck.
                    table.update_item(
                        Key=item['Key'],
                        UpdateExpression=item['UpdateExpression'],
                        ExpressionAttributeValues=item['ExpressionAttributeValues']
                    )
            print("DynamoDB batch update complete.")
        except Exception as e:
            print(f"ERROR: Failed during DynamoDB update: {e}")

    # 4. SEND EVENT FOR NEXT LAMBDA IN THE CHAIN
    if not enriched_issues_for_next_event:
        print("No issues to send to the next step.")
        return {'statusCode': 200, 'body': 'Batch enrichment complete, no issues to forward.'}

    print(f"Sending {len(enriched_issues_for_next_event)} enriched issues to the next step...")
    try:
        eventbridge_client.put_events(
            Entries=[
                {
                    'Source': 'github.issues',
                    'DetailType': 'issue.batch.enriched', 
                    'EventBusName': 'default',
                    'Detail': json.dumps({
                        'repo_name': repo_name,
                        'issues': enriched_issues_for_next_event 
                    })
                }
            ]
        )
    except Exception as e:
        print(f"FATAL: Failed to send 'issue.batch.enriched' event: {e}")

    return {'statusCode': 200, 'body': 'Batch enrichment complete.'}