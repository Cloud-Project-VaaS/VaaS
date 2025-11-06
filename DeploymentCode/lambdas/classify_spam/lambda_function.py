import json
import boto3
import os
import re
import time

# --- AWS Clients (Global) ---
# We get the region from the AWS_REGION environment variable, which is set by default
bedrock_client = boto3.client('bedrock-runtime')
dynamodb = boto3.resource('dynamodb')

# --- Configuration (from Environment Variables) ---
ISSUES_TABLE_NAME = os.environ.get("ISSUES_TABLE_NAME")
LLM_MODEL_ID = "meta.llama3-8b-instruct-v1:0" # Llama 3 8B is perfect for this

def classify_spam_with_bedrock(title, body, author, item_type="issue"):
    """
    Classifies an item as spam or not_spam using Llama 3 8B.
    
    **This is tuned to be "safe" - it will default to "not_spam" if unsure.**
    
    Returns one of: "spam", "not_spam"
    """
    
    # Pre-check: Dependabot is never spam
    if "dependabot[bot]" in author:
        return "not_spam"

    # Clean the body for the prompt
    body_clean = re.sub(r'<!--.*?-->', '', body, flags=re.DOTALL) # Remove HTML comments
    body_clean = re.sub(r'\s+', ' ', body_clean).strip() # Consolidate whitespace
    
    # --- Llama 3 Prompt Format ---
    # This prompt is "biased" towards "not_spam" as requested.
    system_prompt = """You are an expert GitHub spam classifier. Analyze the following GitHub item.
Respond with ONLY one of these two exact words: "spam" or "not_spam".
You must default to "not_spam" if you are at all uncertain."""
    
    user_prompt = f"""Indicators of spam:
- Promotional content or advertisements
- Unrelated links or marketing
- Gibberish or nonsensical content
- Malicious links or phishing attempts

Legitimate content (not spam):
- Bug reports
- Feature requests
- Valid pull requests
- Technical discussions
- Dependency updates or automated CI messages

Here is the {item_type} to classify:
Title: {title}
Author: {author}
Body: {body_clean[:2000]}

Decision (default to "not_spam" if unsure):"""

    # --- Llama 3 Specific Payload ---
    # Note the special <|begin_of_text|> and <|eot_id|> tokens.
    prompt_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    try:
        body_obj = {
            "prompt": prompt_template,
            "max_gen_len": 10,
            "temperature": 0.0,
            "top_p": 0.9,
        }
        
        response = bedrock_client.invoke_model(
            modelId=LLM_MODEL_ID,
            body=json.dumps(body_obj),
            contentType="application/json",
            accept="application/json"
        )
        
        response_body = json.loads(response.get('body').read())
        result_text = response_body.get('generation', 'not_spam').strip().lower()

        # Clean the response. Default to "not_spam"
        if 'spam' in result_text and 'not_spam' not in result_text:
            return "spam"
        else:
            return "not_spam" # Default to not_spam if response is unclear or "not_spam"

    except Exception as e:
        print(f"Error calling Bedrock model: {e}. Defaulting to 'not_spam'.")
        return "not_spam"

def lambda_handler(event, context):
    """
    Main Lambda handler. Triggered by EventBridge with a BATCH of new issues.
    """
    print("Spam Classification event received:")
    print(json.dumps(event))
    
    if not ISSUES_TABLE_NAME:
        print("FATAL: ISSUES_TABLE_NAME environment variable is not set.")
        return {'statusCode': 500, 'body': 'Internal configuration error'}

    # 1. Parse the event
    # This assumes the event is from our 'fetch_and_classify_issues' function
    try:
        # We will set up this source/detail-type in our 'fetcher' Lambda
        if event.get("source") != "github.issues" or event.get("detail-type") != "issue.batch.new":
            print(f"Ignoring event from unknown source: {event.get('source')}")
            return
            
        repo_name = event['detail'].get('repo_name')
        issues_list = event['detail'].get('issues', [])
        
        if not repo_name or not issues_list:
            print("Error: Event detail is missing repo_name or issues list.")
            return

    except KeyError as e:
        print(f"Error parsing event: {e}")
        return
        
    print(f"Starting spam classification for {len(issues_list)} issues from repo: {repo_name}...")
    
    table = dynamodb.Table(ISSUES_TABLE_NAME)
    
    # Use DynamoDB Batch Writer for maximum efficiency
    with table.batch_writer() as batch:
        for issue in issues_list:
            try:
                # 2. Extract data for this issue
                issue_id = issue['issue_id']
                title = issue.get('title', '')
                body = issue.get('body', '')
                author = issue.get('author_login', '')
                
                # 3. Classify the issue
                spam_result = classify_spam_with_bedrock(title, body, author)
                print(f"  - Result for {repo_name}#{issue_id}: {spam_result}")

                # 4. Update the item in DynamoDB
                # This assumes 'fetcher' created the item, and we are just updating it.
                batch.update_item(
                    Key={
                        'repo_name': repo_name,
                        'issue_id': issue_id
                    },
                    UpdateExpression="SET is_spam = :val",
                    ExpressionAttributeValues={
                        ':val': spam_result
                    }
                )
                
                # Simple rate limit to not overwhelm Bedrock
                time.sleep(0.5) # 2 requests per second

            except Exception as e:
                print(f"ERROR: Failed to process issue {issue.get('issue_id')}: {e}")
                continue # Skip this issue and continue with the next
        
    print(f"Successfully processed and updated {len(issues_list)} issues.")
    return {'statusCode': 200, 'body': 'Batch spam classification complete.'}