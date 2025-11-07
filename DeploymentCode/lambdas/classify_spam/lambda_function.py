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
SECRET_ARN = os.environ.get("SECRET_ARN")
LLM_MODEL_ID = "meta.llama3-8b-instruct-v1:0"

# --- GitHub Auth Functions ---
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


# --- REFINED PROMPT LOGIC ---
def classify_spam_with_bedrock(title, body, author, item_type="issue"):
    """
    Classifies an item as spam or not_spam using Llama 3 8B.
    Defaults to "not_spam" if unsure.
    """
    
    if "dependabot[bot]" in author:
        return "not_spam"

    body_clean = re.sub(r'<!--.*?-->', '', body, flags=re.DOTALL)
    body_clean = re.sub(r'\s+', ' ', body_clean).strip()
    
    # This system prompt now uses a few-shot, chain-of-thought approach.
    system_prompt = """You are an expert GitHub spam classifier. Your task is to analyze a GitHub issue and determine if it is spam.
You must first think step-by-step about the content in <thinking> tags.
Then, you MUST provide a final decision inside <decision> tags.
The decision must be ONLY one of these two exact words: "spam" or "not_spam".
Default to "not_spam" if you are at all uncertain.

Here are your examples:

---
<example>
Title: Free Crypto!
Author: crypto-bot-123
Body: Get your free tokens now! http://example.com/free
<thinking>The title is a clear promotion for cryptocurrency. The author "crypto-bot-123" looks like a bot. The body contains a promotional link. This is definitely spam.</thinking>
<decision>spam</decision>
</example>
---
<example>
Title: check this out
Author: user-1982734
Body: great site for... http://example.com also... http://example.com/other
<thinking>The title is vague and low-effort. The author is a random user with a default name. The body is just a list of unrelated links. This is spam.</thinking>
<decision>spam</decision>
</example>
---
<example>
Title: Crash on login page
Author: real-user-dev
Body: I get a 500 error when I try to log in from our marketing site at http://example.com/login. Here is the Sentry error log: [link]
<thinking>The title is a clear, specific bug report. The author looks legitimate. The body contains links, but they are *context* for the bug (the login page and an error log), not promotion. This is a real issue.</thinking>
<decision>not_spam</decision>
</example>
---
<example>
Title: Bump requests from 2.2.1 to 2.3.0
Author: dependabot[bot]
Body: Bumps [requests] from 2.2.1 to 2.3.0. - [Release notes] - [Changelog]
<thinking>The author is dependabot[bot]. This is an automated dependency update, which is a common and legitimate repository event. This is not spam.</thinking>
<decision>not_spam</decision>
</example>
---
"""
    
    user_prompt = f"""Now, classify the following issue:
Title: {title}
Author: {author}
Body: {body_clean[:2000]}"""

    prompt_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    try:
        body_obj = {
            "prompt": prompt_template,
            "max_gen_len": 512, # Increased length to allow for thinking
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
        full_response_text = response_body.get('generation', '')

        # --- NEW PARSING LOGIC ---
        # Use regex to find the decision inside the <decision> tags
        match = re.search(r'<decision>(spam|not_spam)</decision>', full_response_text)
        
        if match:
            # The match will be 'spam' or 'not_spam'
            result_text = match.group(1).strip().lower()
            if result_text == 'spam':
                return "spam"
        
        # Default to not_spam if no match or if match is 'not_spam'
        return "not_spam"

    except Exception as e:
        print(f"Error calling Bedrock model: {e}. Defaulting to 'not_spam'.")
        return "not_spam"

# --- END REFINED PROMPT LOGIC ---


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
        response = requests.patch(url, headers=headers, json=payload)
        response.raise_for_status()
        print(f"Successfully closed and labeled {repo_name}#{issue_id} as spam.")
    except requests.exceptions.RequestException as e:
        print(f"WARNING: Failed to update GitHub issue {repo_name}#{issue_id}: {e}")

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
        install_id = issues_list[0].get('installation_id')
        if not install_id:
            print("FATAL: installation_id missing from event payload. Cannot update spam issues.")
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
            
            # --- THIS FUNCTION IS NOW UPDATED ---
            spam_result = classify_spam_with_bedrock(title, body, author)
            print(f"  - Result for {repo_name}#{issue_id}: {spam_result}")

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
            elif token:
                spam_issues_to_label.append(issue)
            
        except Exception as e:
            print(f"ERROR: Failed to process issue {issue.get('issue_id')}: {e}")
            continue
    
    # --- 3. Update Spam Issues on GitHub ---
    if spam_issues_to_label:
        print(f"Updating {len(spam_issues_to_label)} spam issues on GitHub...")
        for issue in spam_issues_to_label:
            update_github_spam_issue(repo_name, issue['issue_id'], token)
            time.sleep(0.5) 

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
                            'issues': non_spam_issues
                        })
                    }
                ]
            )
        except Exception as e:
            print(f"FATAL: Failed to send EventBridge event: {e}")
    
    print(f"Spam classification step complete for {repo_name}.")
    return {'statusCode': 200, 'body': 'Batch spam classification complete.'}