import json
import boto3
import os
import requests
import jwt
import time
from datetime import datetime, timedelta, timezone
from boto3.dynamodb.conditions import Key

# --- AWS Clients (Global) ---
secrets_client = boto3.client('secretsmanager')
dynamodb = boto3.resource('dynamodb')
eventbridge_client = boto3.client('events')

# --- Environment Variables ---
SECRET_ARN = os.environ.get("SECRET_ARN")
INSTALLATIONS_TABLE_NAME = os.environ.get("INSTALLATIONS_TABLE_NAME")
ISSUES_TABLE_NAME = os.environ.get("ISSUES_TABLE_NAME")

# --- GitHub Auth Functions (Reused) ---
def load_secrets():
    """Loads APP_ID and PRIVATE_KEY from AWS Secrets Manager."""
    try:
        secret_response = secrets_client.get_secret_value(SecretId=SECRET_ARN)
        secrets = json.loads(secret_response['SecretString'])
        return secrets.get('APP_ID'), secrets.get('PRIVATE_KEY')
    except Exception as e:
        print(f"FATAL: Error loading secrets: {e}")
        raise

def create_app_jwt(private_key_pem, app_id):
    try:
        payload = {
            'iat': int(time.time()),
            'exp': int(time.time()) + (10 * 60),
            'iss': app_id
        }
        return jwt.encode(payload, private_key_pem, algorithm='RS256')
    except Exception as e:
        print(f"Error creating JWT: {e}")
        raise

def get_installation_access_token(installation_id, private_key_pem, app_id):
    app_jwt = create_app_jwt(private_key_pem, app_id)
    headers = {"Authorization": f"Bearer {app_jwt}", "Accept": "application/vnd.github.v3+json"}
    url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
    try:
        response = requests.post(url, headers=headers)
        response.raise_for_status()
        return response.json()['token']
    except Exception as e:
        print(f"Error getting installation token: {e}")
        raise

# --- New Helper: Get Installation ID for a specific Repo ---
def get_installation_id_for_repo(repo_name):
    """
    Finds the installation_id for a given repo using the GSI.
    """
    print(f"Looking up installation_id for {repo_name}...")
    table = dynamodb.Table(INSTALLATIONS_TABLE_NAME)
    
    try:
        # We MUST use the GSI because 'repo_name' is the Sort Key, not Partition Key
        response = table.query(
            IndexName='repo_name-index', 
            KeyConditionExpression=Key('repo_name').eq(repo_name)
        )
        if not response['Items']:
            print(f"No installation found for {repo_name}")
            return None
            
        # Return the first match (should be unique per repo)
        return int(response['Items'][0]['installation_id'])
    except Exception as e:
        print(f"Error querying GSI: {e}")
        raise

# --- New Helper: Filter Existing Issues ---
def filter_new_issues_only(repo_name, fetched_issues):
    """
    Checks DynamoDB to see if issues already exist.
    Returns only the issues that are NOT in the table.
    """
    if not fetched_issues:
        return []
        
    table = dynamodb.Table(ISSUES_TABLE_NAME)
    new_unique_issues = []
    
    print(f"Checking {len(fetched_issues)} issues against DynamoDB to prevent duplicates...")
    
    # We check in batches of 100 (DynamoDB limit)
    # But since manual scans are usually small, we'll do a simple loop for now.
    # For production with huge scans, use batch_get_item.
    
    # Optimally, we construct keys for BatchGetItem
    keys_to_check = [{'repo_name': repo_name, 'issue_id': i['issue_id']} for i in fetched_issues]
    
    # Using batch_get_item to check existence efficiently
    try:
        found_keys = set()
        # Process in chunks of 100
        for i in range(0, len(keys_to_check), 100):
            batch_keys = keys_to_check[i:i + 100]
            response = dynamodb.batch_get_item(
                RequestItems={
                    ISSUES_TABLE_NAME: {
                        'Keys': batch_keys,
                        'ProjectionExpression': 'issue_id' # We only need to know if it exists
                    }
                }
            )
            for item in response.get('Responses', {}).get(ISSUES_TABLE_NAME, []):
                found_keys.add(int(item['issue_id']))
        
        # Filter the list
        for issue in fetched_issues:
            if issue['issue_id'] not in found_keys:
                new_unique_issues.append(issue)
            else:
                print(f"  - Skipping Issue #{issue['issue_id']} (Already exists in DB)")
                
    except Exception as e:
        print(f"Error checking for duplicates: {e}")
        # On error, fail safe: return all (might cause reprocessing, but better than data loss)
        return fetched_issues

    print(f"Filtered result: {len(new_unique_issues)} new issues found out of {len(fetched_issues)} fetched.")
    return new_unique_issues

def fetch_issues_manual(repo_name, token, hours_back):
    """
    Fetches issues created in the last X hours.
    """
    print(f"Fetching issues for {repo_name} from the last {hours_back} hours...")
    
    since_time = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).isoformat()
    
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"}
    params = {'since': since_time, 'state': 'open', 'per_page': 100}
    url = f"https://api.github.com/repos/{repo_name}/issues"
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        issues_data = response.json()
        
        cleaned_issues = []
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        
        for issue in issues_data:
            if 'pull_request' in issue: continue
            
            # Double check creation time (API 'since' includes updates)
            created_at = datetime.fromisoformat(issue['created_at'].replace('Z', '+00:00'))
            if created_at < cutoff_time: continue

            cleaned_issues.append({
                'issue_id': issue['number'],
                'title': issue.get('title', ''),
                'body': issue.get('body', ''),
                'author_login': issue.get('user', {}).get('login', '')
            })
            
        return cleaned_issues
    except Exception as e:
        print(f"Error fetching issues: {e}")
        return []

def save_issues_and_send_event(repo_name, issues_list, install_id):
    if not issues_list: return

    table = dynamodb.Table(ISSUES_TABLE_NAME)
    event_issues_payload = []
    
    print(f"Saving {len(issues_list)} NEW issues to DynamoDB...")
    
    try:
        with table.batch_writer() as batch:
            for issue in issues_list:
                issue['installation_id'] = install_id
                item_to_store = {
                    'repo_name': repo_name,
                    'issue_id': issue['issue_id'],
                    'title': issue['title'],
                    'body': issue['body'],
                    'author_login': issue['author_login'],
                    'status': 'new',
                    'pipeline_step': 'received',
                    'created_at_github': datetime.now(timezone.utc).isoformat(),
                    'last_updated_pipeline': datetime.now(timezone.utc).isoformat()
                }
                batch.put_item(Item=item_to_store)
                event_issues_payload.append(issue)

        # Trigger Pipeline
        print(f"Triggering 'issue.batch.new' for {len(event_issues_payload)} issues...")
        eventbridge_client.put_events(
            Entries=[{
                'Source': 'github.issues',
                'DetailType': 'issue.batch.new',
                'EventBusName': 'default',
                'Detail': json.dumps({'repo_name': repo_name, 'issues': event_issues_payload})
            }]
        )
    except Exception as e:
        print(f"Error in save/send: {e}")
        raise

def lambda_handler(event, context):
    """
    Payload: {"repo_name": "owner/repo", "hours_back": 24}
    """
    print("Manual Trigger Event:", json.dumps(event))
    
    repo_name = event.get('repo_name')
    hours_back = event.get('hours_back', 24) # Default to 24 if missing
    
    if not repo_name:
        return {'statusCode': 400, 'body': 'Missing repo_name'}

    try:
        # 1. Load Secrets
        app_id, private_key_pem = load_secrets()
        
        # 2. Get Installation ID (Specific to this repo)
        install_id = get_installation_id_for_repo(repo_name)
        if not install_id:
            return {'statusCode': 404, 'body': f'App not installed on {repo_name}'}
            
        # 3. Get Token
        token = get_installation_access_token(install_id, private_key_pem, app_id)
        
        # 4. Fetch Issues from GitHub
        fetched_issues = fetch_issues_manual(repo_name, token, hours_back)
        
        # 5. Filter (Remove issues already in DynamoDB)
        new_issues = filter_new_issues_only(repo_name, fetched_issues)
        
        if not new_issues:
            print("No new issues found to process.")
            return {'statusCode': 200, 'body': 'No new issues found.'}
            
        # 6. Save & Trigger Pipeline
        save_issues_and_send_event(repo_name, new_issues, install_id)
        
        return {'statusCode': 200, 'body': f'Triggered pipeline for {len(new_issues)} issues.'}

    except Exception as e:
        print(f"FATAL: {e}")
        import traceback
        traceback.print_exc()
        return {'statusCode': 500, 'body': f'Error: {str(e)}'}