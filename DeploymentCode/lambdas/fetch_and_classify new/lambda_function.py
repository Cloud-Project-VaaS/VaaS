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
    """Creates a JSON Web Token (JWT) for GitHub App authentication."""
    try:
        payload = {
            'iat': int(time.time()),
            'exp': int(time.time()) + (10 * 60), # 10 minute expiration
            'iss': app_id
        }
        # --- FIX: Changed 'RS260' to 'RS256' ---
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
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        token_data = response.json()
        if 'token' not in token_data:
            raise ValueError("Error: 'token' not found in response.")
        return token_data['token']
    except requests.exceptions.RequestException as e:
        print(f"Error getting installation token: {e}")
        raise

# --- Core Logic ---

def get_all_installed_repos():
    """
    Scans the 'github-installations' table to get all repos,
    grouped by their installation_id.
    """
    print(f"Fetching all installed repos from {INSTALLATIONS_TABLE_NAME}...")
    table = dynamodb.Table(INSTALLATIONS_TABLE_NAME)
    
    install_repo_map = {} # {install_id: [repo1, repo2]}
    
    try:
        response = table.scan(
            ProjectionExpression="installation_id, repo_name"
        )
        items = response.get('Items', [])
        
        while 'LastEvaluatedKey' in response:
            response = table.scan(
                ExclusiveStartKey=response['LastEvaluatedKey'],
                ProjectionExpression="installation_id, repo_name"
            )
            items.extend(response.get('Items', []))

        for item in items:
            install_id = int(item['installation_id']) # Ensure it's an int
            repo_name = item['repo_name']
            if install_id not in install_repo_map:
                install_repo_map[install_id] = []
            install_repo_map[install_id].append(repo_name)

        print(f"Found {len(items)} repos across {len(install_repo_map)} installations.")
        return install_repo_map

    except Exception as e:
        print(f"FATAL: Error scanning DynamoDB table {INSTALLATIONS_TABLE_NAME}: {e}")
        raise

def fetch_new_issues(repo_name, token):
    """
    Fetches all new issues for a single repo created in the last 70 minutes.
    """
    print(f"Fetching new issues for {repo_name}...")
    
    # We look back 70 minutes to ensure we don't miss anything from the 1-hour schedule
    since_time = (datetime.now(timezone.utc) - timedelta(minutes=70)).isoformat()
    # since_time = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

    
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    # We query for issues *created* since the last run.
    # We also explicitly ask for 'open' state.
    params = {
        'since': since_time,
        'state': 'open',
        'per_page': 100
    }
    url = f"https://api.github.com/repos/{repo_name}/issues"
    
    new_issues = []
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        issues_data = response.json()
        
        if not issues_data:
            print(f"  - No new issues found for {repo_name}.")
            return []

        for issue in issues_data:
            # We only care about issues, not pull requests
            if 'pull_request' in issue:
                continue
                
            # Only process issues *created* since the 'since' time.
            # The 'since' param is for *updates*, so we must double-check creation.
            created_at = datetime.fromisoformat(issue['created_at'])
            if created_at < (datetime.now(timezone.utc) - timedelta(minutes=70)):
            # if created_at < (datetime.now(timezone.utc) - timedelta(days=7)):
                continue

            # This is a brand new issue. Let's process it.
            new_issues.append({
                'issue_id': issue['number'],
                'title': issue.get('title', ''),
                'body': issue.get('body', ''),
                'author_login': issue.get('user', {}).get('login', '')
            })
        
        print(f"  - Found {len(new_issues)} new issue(s) for {repo_name}.")
        return new_issues

    except requests.exceptions.RequestException as e:
        print(f"Error fetching issues for {repo_name}: {e}")
        return [] # Return empty list on error, continue with next repo
    except Exception as e:
        print(f"Unexpected error processing issues for {repo_name}: {e}")
        return []

def save_issues_and_send_event(repo_name, issues_list, install_id): # <-- UPGRADED
    """
    Saves a batch of new issues to DynamoDB and sends an EventBridge event.
    """
    if not issues_list:
        return # Nothing to do

    table = dynamodb.Table(ISSUES_TABLE_NAME)
    
    # This list will be sent to EventBridge
    event_issues_payload = []
    
    print(f"Saving {len(issues_list)} issues to DynamoDB for {repo_name}...")
    
    try:
        with table.batch_writer() as batch:
            for issue in issues_list:
                
                # --- UPGRADE: Add install_id to the issue object ---
                # This is critical for the spam function to be able to get a token
                issue['installation_id'] = install_id
                
                item_to_store = {
                    'repo_name': repo_name, # Partition Key
                    'issue_id': issue['issue_id'], # Sort Key
                    'title': issue['title'],
                    'body': issue['body'],
                    'author_login': issue['author_login'],
                    'status': 'new', # We set the initial status
                    'pipeline_step': 'received',
                    'created_at_github': datetime.now(timezone.utc).isoformat(),
                    'last_updated_pipeline': datetime.now(timezone.utc).isoformat()
                }
                batch.put_item(Item=item_to_store)
                
                # Add the *full* issue payload (now with install_id) to the event list
                event_issues_payload.append(issue)

    except Exception as e:
        print(f"ERROR: Failed to batch write to DynamoDB: {e}")
        # If DB write fails, we should not send the event
        return

    # If DB save was successful, send the event
    print(f"Sending 'issue.batch.new' event for {repo_name} to EventBridge...")
    try:
        eventbridge_client.put_events(
            Entries=[
                {
                    'Source': 'github.issues',
                    'DetailType': 'issue.batch.new',
                    'EventBusName': 'default',
                    'Detail': json.dumps({
                        'repo_name': repo_name,
                        'issues': event_issues_payload # Send the list with install_id
                    })
                }
            ]
        )
    except Exception as e:
        print(f"FATAL: Failed to send EventBridge event: {e}")
        # This is bad, the next step in the pipeline will fail to trigger
        # We might want to add retry logic here in a future version
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
    
    try:
        # 1. Get secrets
        app_id, private_key_pem = load_secrets()
        
        # 2. Get all repos grouped by installation
        install_repo_map = get_all_installed_repos()
    
        if not install_repo_map:
            print("No installations found in DynamoDB. Job complete.")
            return

        # 3. Process each installation
        for install_id, repos in install_repo_map.items():
            try:
                # 4. Get ONE token for this installation
                print(f"Processing installation {install_id} for {len(repos)} repo(s)...")
                token = get_installation_access_token(install_id, private_key_pem, app_id)
                
                # 5. Process all repos for this installation
                for repo_name in repos:
                    new_issues = fetch_new_issues(repo_name, token)
                    # --- UPGRADE: Pass install_id to the save/send function ---
                    save_issues_and_send_event(repo_name, new_issues, install_id)
                    
            except Exception as e:
                print(f"ERROR: Failed to process installation {install_id}. Skipping. Error: {e}")
                continue # Go to the next installation

        print("Hourly issue fetch job complete.")
        return {'statusCode': 200, 'body': 'Job complete.'}

    except Exception as e:
        print(f"FATAL: Unhandled exception in lambda_handler: {e}")
        import traceback
        traceback.print_exc()
        return {'statusCode': 500, 'body': 'Internal server error'}