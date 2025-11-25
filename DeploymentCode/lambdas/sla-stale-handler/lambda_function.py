import json
import boto3
import os
import requests
import jwt
import time
from datetime import datetime, timedelta, timezone
from boto3.dynamodb.conditions import Key

# --- CONFIGURATION ---
SECRET_ARN = os.environ.get("SECRET_ARN")
INSTALLATIONS_TABLE_NAME = os.environ.get("INSTALLATIONS_TABLE_NAME", "github-installations")
ISSUES_TABLE_NAME = os.environ.get("ISSUES_TABLE_NAME", "IssuesTrackingTable")

# --- TIMING CONFIGURATION (HOURS) ---
# Updated thresholds as requested
STALE_HOURS = 168  # 7 Days

SLA_THRESHOLDS = {
    'high': 54,      # ~2.25 Days
    'critical': 54,
    'p0': 54,
    'medium': 150,   # ~6.25 Days
    'p1': 150,
    'low': 360,      # 15 Days
    'p2': 360,
    'default': 360   # Fallback
}

# --- AWS CLIENTS ---
secrets_client = boto3.client('secretsmanager')
dynamodb = boto3.resource('dynamodb')

# --- AUTH HELPERS ---
def load_secrets():
    try:
        secret_response = secrets_client.get_secret_value(SecretId=SECRET_ARN)
        secrets = json.loads(secret_response['SecretString'])
        return secrets.get('APP_ID'), secrets.get('PRIVATE_KEY')
    except Exception as e:
        print(f"FATAL: Error loading secrets: {e}")
        raise

def create_app_jwt(private_key_pem, app_id):
    now = int(time.time())
    payload = {
        'iat': now - 60,
        'exp': now + (10 * 60),
        'iss': app_id
    }
    return jwt.encode(payload, private_key_pem, algorithm='RS256')

def get_installation_access_token(installation_id, private_key_pem, app_id):
    app_jwt = create_app_jwt(private_key_pem, app_id)
    headers = {"Authorization": f"Bearer {app_jwt}", "Accept": "application/vnd.github.v3+json"}
    url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
    response = requests.post(url, headers=headers)
    response.raise_for_status()
    return response.json()['token']

# --- CORE LOGIC ---

def get_all_installations():
    table = dynamodb.Table(INSTALLATIONS_TABLE_NAME)
    try:
        response = table.scan()
        data = response.get('Items', [])
        while 'LastEvaluatedKey' in response:
            response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
            data.extend(response.get('Items', []))
        return data
    except Exception as e:
        print(f"Error fetching installations: {e}")
        return []

def get_open_db_issues(repo_name):
    table = dynamodb.Table(ISSUES_TABLE_NAME)
    open_issues = []
    try:
        response = table.query(KeyConditionExpression=Key('repo_name').eq(repo_name))
        items = response.get('Items', [])
        for item in items:
            if item.get('status') == 'open':
                open_issues.append(item)
        return open_issues
    except Exception as e:
        print(f"Error fetching issues for {repo_name}: {e}")
        return []

def post_comment(repo_name, issue_number, body, token):
    """Posts a comment to the GitHub issue."""
    url = f"https://api.github.com/repos/{repo_name}/issues/{issue_number}/comments"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"}
    try:
        requests.post(url, headers=headers, json={"body": body})
        print(f"  -> Commented on {repo_name}#{issue_number}")
    except Exception as e:
        print(f"  -> Failed to comment: {e}")

def add_labels(repo_name, issue_number, labels, token):
    """Adds labels to the GitHub issue."""
    url = f"https://api.github.com/repos/{repo_name}/issues/{issue_number}/labels"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"}
    try:
        requests.post(url, headers=headers, json={"labels": labels})
        print(f"  -> Added labels {labels} to {repo_name}#{issue_number}")
    except Exception as e:
        print(f"  -> Failed to add labels: {e}")

def check_and_update_issue(repo_name, db_issue, token):
    issue_number = db_issue['issue_id']
    
    # Normalize priority to find threshold
    raw_priority = str(db_issue.get('priority', 'default')).lower()
    sla_limit_hours = SLA_THRESHOLDS.get('default')
    
    # Identify tier for logic branching
    priority_tier = 'low' # Default
    
    for key, val in SLA_THRESHOLDS.items():
        if key in raw_priority:
            sla_limit_hours = val
            if key in ['high', 'critical', 'p0']: priority_tier = 'high'
            elif key in ['medium', 'p1']: priority_tier = 'medium'
            else: priority_tier = 'low'
            break

    url = f"https://api.github.com/repos/{repo_name}/issues/{issue_number}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"}
    
    try:
        # 1. Fetch Live Status from GitHub
        response = requests.get(url, headers=headers)
        if response.status_code == 404: return
        
        gh_issue = response.json()
        gh_state = gh_issue['state']
        gh_updated_at = datetime.fromisoformat(gh_issue['updated_at'].replace('Z', '+00:00'))
        
        # GitHub 'updated_at' changes on ANY activity (comments, labels, etc.)
        now = datetime.now(timezone.utc)
        hours_inactive = (now - gh_updated_at).total_seconds() / 3600
        
        current_labels = [l['name'] for l in gh_issue.get('labels', [])]
        assignees = [a['login'] for a in gh_issue.get('assignees', [])]
        
        table = dynamodb.Table(ISSUES_TABLE_NAME)

        # --- CHECK 1: SYNC CLOSED STATUS ---
        if gh_state == 'closed':
            print(f"Syncing CLOSED status for {repo_name}#{issue_number}")
            table.update_item(
                Key={'repo_name': repo_name, 'issue_id': int(issue_number)},
                UpdateExpression="SET #s = :s, last_updated_pipeline = :l",
                ExpressionAttributeNames={'#s': 'status'},
                ExpressionAttributeValues={':s': 'closed', ':l': datetime.now(timezone.utc).isoformat()}
            )
            return

        # --- CHECK 2: STALE LOGIC ---
        if hours_inactive > STALE_HOURS:
            # Strict check to prevent duplicate labeling
            if "Stale" not in current_labels:
                print(f"!! Marking {repo_name}#{issue_number} as STALE ({int(hours_inactive)} hrs inactive).")
                add_labels(repo_name, issue_number, ["Stale"], token)
        
        # --- CHECK 3: SLA LOGIC ---
        if hours_inactive > sla_limit_hours:
            # STRICT CHECK: If label exists, do NOTHING. This prevents double replies.
            if "SLA Breached" in current_labels:
                return

            print(f"!! SLA Breach detected for {repo_name}#{issue_number} (Priority: {raw_priority})")
            
            # Step A: Always Label first
            add_labels(repo_name, issue_number, ["SLA Breached"], token)
            
            # Step B: Conditional Commenting based on Tier
            # Only ping for High and Medium priorities
            if priority_tier in ['high', 'medium']:
                if assignees:
                    pings = " ".join([f"@{u}" for u in assignees])
                    msg = (f"{pings} This issue has exceeded the **{raw_priority.title()}** priority SLA "
                           f"of {sla_limit_hours} hours. Please provide an update.")
                    post_comment(repo_name, issue_number, msg, token)
                else:
                    # Unassigned but high priority - maybe post a general alert?
                    # For now, sticking to label only as requested if unassigned.
                    print(f"  -> Unassigned High/Medium issue. Applied label only.")
            else:
                # Low priority: Tag only (Label applied in Step A), no comment
                print(f"  -> Low priority SLA breach. Applied label only (no ping).")

    except Exception as e:
        print(f"Error checking issue {repo_name}#{issue_number}: {e}")

def process_repo(repo_name, install_id, app_id, private_key):
    print(f"Scanning repo: {repo_name}...")
    open_issues = get_open_db_issues(repo_name)
    if not open_issues: return

    token = get_installation_access_token(install_id, private_key, app_id)
    for issue in open_issues:
        check_and_update_issue(repo_name, issue, token)

def lambda_handler(event, context):
    print("Starting SLA/Stale Check Job...")
    try:
        app_id, private_key = load_secrets()
        installations = get_all_installations()
        
        for install in installations:
            repo_name = install.get('repo_name')
            install_id = int(install.get('installation_id'))
            if repo_name and install_id:
                process_repo(repo_name, install_id, app_id, private_key)
                
        return {'statusCode': 200, 'body': "SLA check complete."}
    except Exception as e:
        print(f"FATAL: {e}")
        return {'statusCode': 500, 'body': str(e)}