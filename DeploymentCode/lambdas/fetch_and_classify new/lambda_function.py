import json
import boto3
import os
import requests
import jwt
import time
from datetime import datetime, timedelta, timezone

# --- AWS Clients (Global) ---
secrets_client = boto3.client('secretsmanager')
dynamodb = boto3.resource('dynamodb')
eventbridge_client = boto3.client('events')

# --- Environment Variables ---
SECRET_ARN = os.environ.get("SECRET_ARN")
INSTALLATIONS_TABLE_NAME = os.environ.get("INSTALLATIONS_TABLE_NAME")
ISSUES_TABLE_NAME = os.environ.get("ISSUES_TABLE_NAME")

# --- GitHub Auth Functions (Copied from expertise_scanner) ---
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
    """Creates a short-lived JWT (10 min) to authenticate as the GitHub App."""
    try:
        payload = {
            'iat': int(time.time()) - 60,
            'exp': int(time.time()) + (10 * 60),
            'iss': app_id
        }
        return jwt.encode(payload, private_key_pem, algorithm='RS256')
    except Exception as e:
        print(f"Error encoding JWT: {e}")
        raise

def get_installation_access_token(installation_id, private_key_pem, app_id):
    """Exchanges the App JWT for a temporary installation token."""
    print(f"Getting token for installation {installation_id}...")
    try:
        app_jwt = create_app_jwt(private_key_pem, app_id)
        headers = {
            "Authorization": f"Bearer {app_jwt}",
            "Accept": "application/vnd.github.v3+json"
        }
        url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
        response = requests.post(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        print(f"Successfully generated new token for {installation_id}.")
        return data.get('token')
    except Exception as e:
        print(f"Failed to get token for installation {installation_id}. Error: {e}")
        raise

def get_all_installed_repos():
    """
    Scans the installations table to get a list of all repos,
    grouped by installation_id.
    """
    print(f"Fetching all installed repos from {INSTALLATIONS_TABLE_NAME}...")
    table = dynamodb.Table(INSTALLATIONS_TABLE_NAME)
    try:
        response = table.scan()
        items = response.get('Items', [])
        while 'LastEvaluatedKey' in response:
            response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
            items.extend(response.get('Items', []))
        
        # Group repos by installation_id
        install_repo_map = {}
        for item in items:
            install_id = item['installation_id']
            repo_name = item['repo_name']
            if install_id not in install_repo_map:
                install_repo_map[install_id] = []
            install_repo_map[install_id].append(repo_name)
            
        print(f"Found {len(items)} repos across {len(install_repo_map)} installations.")
        return install_repo_map
        
    except Exception as e:
        print(f"Error scanning DynamoDB table {INSTALLATIONS_TABLE_NAME}: {e}")
        return {}

def fetch_new_issues(repo_name, auth_token):
    """Fetches all open issues from a repo created in the last 70 minutes."""
    print(f"  - Fetching new issues for {repo_name}...")
    
    # Go back 70 minutes to provide a 10-min overlap with the hourly run
    since_time = (datetime.now(timezone.utc) - timedelta(minutes=70)).isoformat()
    
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    url = f"https://api.github.com/repos/{repo_name}/issues"
    params = {
        "state": "open",
        "since": since_time,
        "per_page": 100
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        issues = response.json()
        
        # Filter out Pull Requests (they also appear in the 'issues' endpoint)
        new_issues = [iss for iss in issues if 'pull_request' not in iss]
        print(f"  - Found {len(new_issues)} new issues for {repo_name}.")
        return new_issues
        
    except Exception as e:
        print(f"  - Error fetching issues for {repo_name}: {e}")
        return []

def save_issues_and_send_event(repo_name, issues_list):
    """
    Saves a batch of new issues to DynamoDB and sends an
    EventBridge event to trigger the classification pipeline.
    """
    if not issues_list:
        print(f"No new issues to save or send for {repo_name}.")
        return

    table = dynamodb.Table(ISSUES_TABLE_NAME)
    issues_for_event = []

    print(f"Saving {len(issues_list)} issues to {ISSUES_TABLE_NAME}...")
    with table.batch_writer() as batch:
        for issue in issues_list:
            issue_id = issue['id']
            # Create the item to be stored
            item_to_store = {
                'installation_id': install_id,
                'repo_name': repo_name,
                'issue_id': issue_id,
                'status': 'open',
                'title': issue.get('title', ''),
                'body': issue.get('body', ''),
                'author_login': issue.get('user', {}).get('login', ''),
                'created_at_github': issue.get('created_at'),
                'last_updated_pipeline': datetime.now(timezone.utc).isoformat()
                # We will add other fields (is_spam, priority, etc.) later
            }
            batch.put_item(Item=item_to_store)
            
            # Add a minimal version to the event payload
            issues_for_event.append({
                'issue_id': issue_id,
                'title': item_to_store['title'],
                'body': item_to_store['body'],
                'author_login': item_to_store['author_login']
            })

    # Send ONE event to EventBridge for this ENTIRE batch
    print(f"Sending 'issue.batch.new' event for {repo_name} to EventBridge...")
    try:
        eventbridge_client.put_events(
            Entries=[
                {
                    'Source': 'github.issues',
                    'DetailType': 'issue.batch.new',
                    'EventBusName': 'default', # Using the default bus is fine
                    'Detail': json.dumps({
                        'repo_name': repo_name,
                        'issues': issues_for_event
                    })
                }
            ]
        )
    except Exception as e:
        print(f"FATAL: Failed to send EventBridge event: {e}")
        raise

def lambda_handler(event, context):
    """
    Main handler, triggered by a 1-hour schedule.
    Fetches all new issues for all installed repos.
    """
    print("Starting hourly issue fetch job...")
    
    if not SECRET_ARN or not INSTALLATIONS_TABLE_NAME or not ISSUES_TABLE_NAME:
        print("FATAL: Environment variables not set.")
        return

    # 1. Get secrets
    app_id, private_key_pem = load_secrets()
    
    # 2. Get all repos grouped by installation
    install_repo_map = get_all_installed_repos()

    # 3. Process each installation
    for install_id, repos in install_repo_map.items():
        try:
            # 4. Get ONE token for this installation
            token = get_installation_access_token(install_id, private_key_pem, app_id)
            
            # 5. Process all repos for this installation
            for repo_name in repos:
                new_issues = fetch_new_issues(repo_name, token)
                save_issues_and_send_event(repo_name, new_issues)
                
        except Exception as e:
            print(f"ERROR: Failed to process installation {install_id}. Skipping. Error: {e}")
            continue # Skip to the next installation

    print("Hourly issue fetch job complete.")
    return {'statusCode': 200, 'body': 'Job complete.'}