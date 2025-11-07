import json
import boto3
import os
import re
import time
import requests
import jwt # PyJWT
from datetime import datetime, timezone, timedelta

# --- AWS Clients (Global) ---
bedrock_client = boto3.client('bedrock-runtime')
dynamodb = boto3.resource('dynamodb')
eventbridge_client = boto3.client('events')
secrets_client = boto3.client('secretsmanager')

# --- Configuration (from Environment Variables) ---
ISSUES_TABLE_NAME = os.environ.get("ISSUES_TABLE_NAME")
SECRET_ARN = os.environ.get("SECRET_ARN") # <-- NEW: Must be set in this Lambda
LLM_MODEL_ID = "meta.llama3-8b-instruct-v1:0"

# --- GitHub Auth Functions (Copied from fetch_and_classify_issues) ---
def load_secrets():
    """Loads APP_ID and PRIVATE_KEY from AWS Secrets Manager."""
    print("Loading secrets from Secrets Manager...")
    try:
        secret_response = secrets_client.get_secret_value(SecretId=SECRET_ARN)
        secrets = json.loads(secret_response['SecretString'])
        app_id = secrets.get('APP_ID')
        private_key_pem = secrets.get('PRIVATE_KEY')
        if not app_id or not private_key_pem:
            raise KeyError("APP_ID or PRIVATE_KEY not found in secret.")
        return app_id, private_key_pem
    except Exception as e:
        print(f"FATAL: Error loading secrets: {e}")
        raise

def create_app_jwt(private_key_pem, app_id):
    """Creates a JSON Web Token (JWT) for GitHub App authentication."""
    try:
        payload = {
            'iat': int(time.time()),
            'exp': int(time.time()) + (10 * 60), # 10 minute expiration
            'iss': app_id
        }
        return jwt.encode(payload, private_key_pem, algorithm='RS256')
    except Exception as e:
        print(f"Error creating JWT: {e}")
        raise

def get_installation_access_token(installation_id, private_key_pem, app_id):
    """Gets a temporary access token for a specific installation."""
    app_jwt = create_app_jwt(private_key_pem, app_id)
    headers = {
        "Authorization": f"Bearer {app_jwt}",
        "Accept": "application/vnd.github.v3+json",
    }
    url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
    
    try:
        response = requests.post(url, headers=headers)
        response.raise_for_status()
        token_data = response.json()
        if 'token' not in token_data:
            raise ValueError("Error: 'token' not found in response.")
        return token_data['token']
    except requests.exceptions.RequestException as e:
        print(f"Error getting installation token: {e}")
        raise
# --- End Auth Functions ---

def classify_spam_with_bedrock(title, body, author, item_type="issue"):
    """
    Classifies an item as spam or not_spam using Llama 3 8B.
    Defaults to "not_spam" if unsure.
    """
    
    if "dependabot[bot]" in author:
        return "not_spam"

    body_clean = re.sub(r'<!--.*?-->', '', body, flags=re.DOTALL)
    body_clean = re.sub(r'\s+', ' ', body_clean).strip()
    
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

        if 'spam' in result_text and 'not_spam' not in result_text:
            return "spam"
        else:
            return "not_spam"

    except Exception as e:
        print(f"Error calling Bedrock model: {e}. Defaulting to 'not_spam'.")
        return "not_spam"

def update_github_spam_issue(repo_name, issue_id, token):
    """
    Updates a GitHub issue to add a 'spam' label and close it.
    """
    print(f"Updating GitHub issue {repo_name}#{issue_id} as spam...")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    url = f"https://api.github.com/repos/{repo_name}/issues/{issue_id}"
    payload = {
        "labels": ["spam"], # Adds "spam" label (and replaces others)
        "state": "closed"    # Closes the issue
    }
    
    try:
        # Use requests.patch to update the issue
        response = requests.patch(url, headers=headers, json=payload)
        response.raise_for_status()
        print(f"Successfully closed and labeled {repo_name}#{issue_id} as spam.")
    except requests.exceptions.RequestException as e:
        print(f"WARNING: Failed to update GitHub issue {repo_name}#{issue_id}: {e}")
        # Log the error but don't fail the entire function

def lambda_handler(event, context):
    """
    Main Lambda handler. Triggered by EventBridge with a BATCH of new issues.
    """
    print("Spam Classification event received:")
    print(json.dumps(event))
    
    if not ISSUES_TABLE_NAME or not SECRET_ARN:
        print("FATAL: Environment variables ISSUES_TABLE_NAME or SECRET_ARN are not set.")
        return {'statusCode': 500, 'body': 'Internal configuration error'}

    try:
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
    non_spam_issues = []
    spam_issues_to_label = []
    
    # --- 1. Get Secrets (once) and Auth Token (if needed) ---
    try:
        app_id, private_key_pem = load_secrets()
        # Get the install_id from the *first issue*
        # We assume all issues in a batch are from the same installation
        install_id = issues_list[0].get('installation_id')
        if not install_id:
            print("FATAL: installation_id missing from event payload. Cannot update spam issues.")
            # We will still process, but won't be able to label spam
            token = None
        else:
            token = get_installation_access_token(install_id, private_key_pem, app_id)
            
    except Exception as e:
        print(f"FATAL: Could not get GitHub auth token: {e}. Spam issues will not be labeled.")
        token = None

    # --- 2. Classify Issues and Update DynamoDB ---
    for issue in issues_list:
        try:
            issue_id = issue['issue_id']
            title = issue.get('title', '')
            body = issue.get('body', '')
            author = issue.get('author_login', '')
            
            # Classify the issue
            spam_result = classify_spam_with_bedrock(title, body, author)
            print(f"  - Result for {repo_name}#{issue_id}: {spam_result}")

            # Update DynamoDB with the result
            # (Fixed: No batch_writer, call table.update_item directly)
            table.update_item(
                Key={
                    'repo_name': repo_name,
                    'issue_id': issue_id
                },
                UpdateExpression="SET is_spam = :val, pipeline_step = :step",
                ExpressionAttributeValues={
                    ':val': spam_result,
                    ':step': 'spam_classified'
                }
            )
            
            if spam_result == "not_spam":
                non_spam_issues.append(issue)
            elif token: # It's spam AND we have a token
                spam_issues_to_label.append(issue)
            
        except Exception as e:
            print(f"ERROR: Failed to process issue {issue.get('issue_id')}: {e}")
            continue
    
    # --- 3. Update Spam Issues on GitHub ---
    if spam_issues_to_label:
        print(f"Updating {len(spam_issues_to_label)} spam issues on GitHub...")
        for issue in spam_issues_to_label:
            update_github_spam_issue(repo_name, issue['issue_id'], token)
            time.sleep(0.5) # Small delay to avoid secondary rate limits

    # --- 4. Send Event for Non-Spam Issues ---
    if not non_spam_issues:
        print("No non-spam issues to send to the next pipeline step.")
    else:
        print(f"Sending {len(non_spam_issues)} non-spam issues to 'issue.batch.spam_classified' event...")
        try:
            eventbridge_client.put_events(
                Entries=[
                    {
                        'Source': 'github.issues',
                        'DetailType': 'issue.batch.spam_classified',
                        'EventBusName': 'default',
                        'Detail': json.dumps({
                            'repo_name': repo_name,
                            'issues': non_spam_issues # Pass only non-spam
                        })
                    }
                ]
            )
        except Exception as e:
            print(f"FATAL: Failed to send EventBridge event: {e}")
    
    print(f"Spam classification step complete for {repo_name}.")
    return {'statusCode': 200, 'body': 'Batch spam classification complete.'}