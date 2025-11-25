import json
import boto3
import os
import httpx
import re
import sys
import time
import jwt  # PyJWT
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict
from botocore.config import Config

# --- Configuration ---
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "ap-south-1")
SECRET_ARN = os.environ.get("SECRET_ARN")
INSTALLATIONS_TABLE_NAME = os.environ.get("INSTALLATIONS_TABLE_NAME", "github-installations")
USER_AVAILABILITY_TABLE_NAME = os.environ.get("USER_AVAILABILITY_TABLE_NAME", "UserAvailability")
EXPERTISE_TABLE_NAME = os.environ.get("EXPERTISE_TABLE_NAME", "RepoExpertise")

# Switch to Mistral for reliability
LLM_MODEL_ID = "mistral.mistral-7b-instruct-v0:2"
DAYS_TO_SCAN = 90
MIN_ACTIVITY_THRESHOLD = 5
DEFAULT_START_TIME = "09:00"
DEFAULT_END_TIME = "17:00"

# --- AWS Clients ---
boto_config = Config(connect_timeout=10, read_timeout=60)
bedrock_client = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION, config=boto_config)
secrets_client = boto3.client('secretsmanager')
dynamodb = boto3.resource('dynamodb')

# --- Global Cache ---
APP_ID = None
PRIVATE_KEY = None
_token_cache: Dict[int, Tuple[str, datetime]] = {}

# --- Auth Helpers ---
def load_secrets():
    global APP_ID, PRIVATE_KEY
    if APP_ID and PRIVATE_KEY: return
    try:
        secret_response = secrets_client.get_secret_value(SecretId=SECRET_ARN)
        secrets = json.loads(secret_response['SecretString'])
        APP_ID = secrets.get('APP_ID')
        PRIVATE_KEY = secrets.get('PRIVATE_KEY')
    except Exception as e:
        print(f"Error loading secrets: {e}")
        raise

def create_app_jwt() -> str:
    if not APP_ID: load_secrets()
    payload = {'iat': int(time.time()), 'exp': int(time.time()) + (10 * 60), 'iss': APP_ID}
    return jwt.encode(payload, PRIVATE_KEY, algorithm='RS256')

def get_installation_access_token(installation_id: int) -> str:
    if installation_id in _token_cache:
        token, expiry = _token_cache[installation_id]
        if expiry > datetime.now(timezone.utc): return token

    app_jwt = create_app_jwt()
    headers = {"Authorization": f"Bearer {app_jwt}", "Accept": "application/vnd.github.v3+json"}
    url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
    
    response = httpx.post(url, headers=headers, timeout=10.0)
    response.raise_for_status()
    token = response.json()['token']
    _token_cache[installation_id] = (token, datetime.now(timezone.utc) + timedelta(minutes=55))
    return token

# --- Data Sources ---

def get_contributors_for_repo(repo_name: str) -> List[str]:
    """Fetch list of users from RepoExpertise table."""
    table = dynamodb.Table(EXPERTISE_TABLE_NAME)
    try:
        response = table.get_item(Key={'repo_name': repo_name})
        if 'Item' not in response: return []
        profiles = response['Item'].get('expertise_profiles', {})
        return list(profiles.keys())
    except Exception as e:
        print(f"Error fetching contributors for {repo_name}: {e}")
        return []

def get_all_installed_repos() -> Dict[str, int]:
    """Scans github-installations for all repos."""
    table = dynamodb.Table(INSTALLATIONS_TABLE_NAME)
    repo_map = {}
    try:
        response = table.scan()
        items = response.get('Items', [])
        while 'LastEvaluatedKey' in response:
            response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
            items.extend(response.get('Items', []))
        for item in items:
            repo_map[item['repo_name']] = int(item['installation_id'])
        return repo_map
    except Exception as e:
        print(f"Error scanning installations: {e}")
        return {}

# --- Async GitHub Fetching ---

async def fetch_user_timestamps(repo: str, user: str, since: str, client: httpx.AsyncClient) -> List[str]:
    timestamps = []
    
    # 1. Commits
    try:
        commits_url = f"https://api.github.com/repos/{repo}/commits"
        params = {"author": user, "since": since, "per_page": 100}
        # Just get page 1 to save time/tokens, usually enough for availability check
        resp = await client.get(commits_url, params=params)
        if resp.status_code == 200:
            for c in resp.json():
                timestamps.append(c['commit']['author']['date'])
    except Exception: pass

    # 2. Activity (Issues/PRs)
    try:
        # Search issues/PRs involved in
        query = f"repo:{repo} involves:{user} updated:>{since}"
        search_url = "https://api.github.com/search/issues"
        resp = await client.get(search_url, params={"q": query, "per_page": 100})
        if resp.status_code == 200:
            for item in resp.json().get('items', []):
                timestamps.append(item['updated_at'])
                if item.get('closed_at'): timestamps.append(item['closed_at'])
    except Exception: pass
    
    return timestamps

# --- LLM Analysis ---

def analyze_activity_with_llm(user_handle: str, all_timestamps: List[str]) -> Dict[str, Any]:
    activity_points = []
    for ts_str in all_timestamps:
        try:
            dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00')).astimezone(timezone.utc)
            activity_points.append(f"({dt.weekday()}, {dt.hour})")
        except: continue

    if len(activity_points) < MIN_ACTIVITY_THRESHOLD:
        return {
            "inferred_start_time_utc": DEFAULT_START_TIME,
            "inferred_end_time_utc": DEFAULT_END_TIME,
            "status": "default_low_activity"
        }

    # --- MISTRAL PROMPT ---
    data_str = ", ".join(activity_points[:300]) # Limit size
    
    system_prompt = """You are an expert data analyst. Analyze the list of (day_of_week, hour_of_day) tuples representing a user's UTC activity.
Infer their primary 8-hour working window in UTC.
- Day 0=Monday, 6=Sunday.
- Ignore outliers.
- Return ONLY a JSON object with keys "inferred_start_time_utc" (HH:MM) and "inferred_end_time_utc" (HH:MM)."""
    
    prompt_text = f"<s>[INST] {system_prompt} \n\n Data: [{data_str}] [/INST]"
    
    body = {
        "prompt": prompt_text,
        "max_tokens": 200,
        "temperature": 0.1
    }

    try:
        response = bedrock_client.invoke_model(
            modelId=LLM_MODEL_ID, body=json.dumps(body),
            contentType="application/json", accept="application/json"
        )
        r_body = json.loads(response.get('body').read())
        text = r_body.get('outputs')[0].get('text', '')
        
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        print(f"LLM Error for {user_handle}: {e}")

    return {
        "inferred_start_time_utc": DEFAULT_START_TIME,
        "inferred_end_time_utc": DEFAULT_END_TIME,
        "status": "error_fallback"
    }

# --- Orchestration ---

async def process_repo(repo_name: str, install_id: int, since: str):
    print(f"Scanning repo: {repo_name}")
    users = get_contributors_for_repo(repo_name)
    if not users:
        print(f"No users found in RepoExpertise for {repo_name}")
        return

    token = get_installation_access_token(install_id)
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"}
    
    async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
        tasks = []
        for user in users:
            tasks.append(fetch_user_timestamps(repo_name, user, since, client))
        
        # Gather timestamps
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze and Save
        table = dynamodb.Table(USER_AVAILABILITY_TABLE_NAME)
        with table.batch_writer() as batch:
            for i, user in enumerate(users):
                timestamps = results[i]
                if isinstance(timestamps, list) and timestamps:
                    print(f"  - Analyzing {user} ({len(timestamps)} events)...")
                    # Sync call to LLM (Bedrock is fast enough)
                    window = analyze_activity_with_llm(user, timestamps)
                    
                    item = {
                        'user_handle': user,
                        'inferred_start_time_utc': window.get('inferred_start_time_utc', DEFAULT_START_TIME),
                        'inferred_end_time_utc': window.get('inferred_end_time_utc', DEFAULT_END_TIME),
                        'last_updated': datetime.now(timezone.utc).isoformat(),
                        'status': 'active'
                    }
                    batch.put_item(Item=item)
                else:
                    print(f"  - Skipping {user} (No recent activity)")

def lambda_handler(event, context):
    print("Event:", json.dumps(event))
    
    # Check Mode: Manual (Dashboard) or Scheduled (Cron)
    target_repo = event.get('repo_name')
    manual_trigger = event.get('task') == 'infer_availability'
    
    all_repos = get_all_installed_repos()
    since = (datetime.now(timezone.utc) - timedelta(days=DAYS_TO_SCAN)).isoformat()
    
    if manual_trigger and target_repo:
        # Mode A: Manual Single Repo
        if target_repo not in all_repos:
            return {'statusCode': 404, 'body': 'Repo not found/installed'}
        
        install_id = all_repos[target_repo]
        asyncio.run(process_repo(target_repo, install_id, since))
        return {'statusCode': 200, 'body': f'Inferred availability for {target_repo}'}
    
    else:
        # Mode B: Scheduled Full Scan
        print("Running full scheduled scan...")
        for repo_name, install_id in all_repos.items():
            try:
                asyncio.run(process_repo(repo_name, install_id, since))
            except Exception as e:
                print(f"Failed to process {repo_name}: {e}")
        return {'statusCode': 200, 'body': 'Full scan complete'}