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
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict

# --- Configuration (from Environment Variables) ---
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "ap-south-1")
SECRET_ARN = os.environ.get("SECRET_ARN")
INSTALLATIONS_TABLE_NAME = os.environ.get("INSTALLATIONS_TABLE_NAME")
AVAILABILITY_TABLE_NAME = os.environ.get("USER_AVAILABILITY_TABLE_NAME") # Our new table

LLM_MODEL_ID = "deepseek.v3-v1:0"
DAYS_TO_SCAN = 90
MIN_ACTIVITY_THRESHOLD = 10  # Min data points to trigger LLM
DEFAULT_START_TIME = "09:00"
DEFAULT_END_TIME = "17:00"

# --- PASTE YOUR JSON CONTENT HERE ---
# We paste the content of example_team_structure.json here
# This makes the Lambda self-contained and removes the need for an S3 lookup.
TEAM_STRUCTURE_JSON = """
{
  "active_teams": {
    "ml_compiler_team": [
      "Venkat6871", "pschuh", "ezhulenev", "jcai19", "hhb", "chsigg", 
      "akuegel", "thomasjoerg", "felixwqp", "GleasonK", "majiddadashi", 
      "pifon2a", "mrguenther", "ghpvnist", "subhankarshah"
    ],
    "gpu_performance_team": [
      "Venkat6871", "gaikwadrahul8", "pschuh", "ezhulenev", "gbaned", 
      "chsigg", "akuegel", "thomasjoerg", "felixwqp", "metaflow"
    ],
    "build_infrastructure_team": [
      "quoctruong", "gbaned", "meteorcloudy", "majnemer", "jparkerh"
    ],
    "platform_compatibility_team": [
      "gaikwadrahul8", "penpornk", "qukhan", "majnemer", "yangustc07"
    ],
    "quantization_optimization_team": ["Venkat6871", "majiddadashi"],
    "testing_documentation_team": ["ILCSFNO", "SandSnip3r", "ghpvnist"]
  },
  "inferred_hierarchy": {
    "leads": ["Venkat6871", "pschuh", "ezhulenev", "gbaned"],
    "engineers": [
      "gaikwadrahul8", "quoctruong", "jcai19", "penpornk", "hhb", "chsigg",
      "akuegel", "thomasjoerg", "felixwqp", "GleasonK", "majiddadashi", 
      "SandSnip3r", "pifon2a", "mrguenther", "meteorcloudy", "ghpvnist", 
      "qukhan", "majnemer", "subhankarshah", "jparkerh", "yangustc07", 
      "metaflow", "ILCSFNO"
    ]
  }
}
"""

# --- AWS Clients (Global) ---
try:
    bedrock_client = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION)
    secrets_client = boto3.client('secretsmanager')
    dynamodb = boto3.resource('dynamodb')
except Exception as e:
    print(f"Error: Could not initialize boto3 clients: {e}")
    # This will fail the Lambda cold start, which is intended.

# --- Global Cache for Secrets/Tokens ---
APP_ID = None
PRIVATE_KEY = None
_token_cache: Dict[int, Tuple[str, datetime]] = {} # Cache for installation_id -> (token, expiry)

# --- GitHub App Authentication (Re-used from our other Lambdas) ---
def load_secrets():
    """Loads App ID and Private Key from Secrets Manager."""
    global APP_ID, PRIVATE_KEY
    if APP_ID and PRIVATE_KEY:
        return
    try:
        print("Loading secrets from Secrets Manager...")
        secret_response = secrets_client.get_secret_value(SecretId=SECRET_ARN)
        secrets = json.loads(secret_response['SecretString'])
        APP_ID = secrets.get('APP_ID')
        PRIVATE_KEY = secrets.get('PRIVATE_KEY')
        if not APP_ID or not PRIVATE_KEY:
            raise KeyError("APP_ID or PRIVATE_KEY not found in secret.")
    except Exception as e:
        print(f"Error loading secrets: {e}")
        raise

def create_app_jwt() -> str:
    """Creates a JSON Web Token (JWT) for GitHub App authentication."""
    if not APP_ID or not PRIVATE_KEY:
        load_secrets()
    try:
        payload = {'iat': int(time.time()), 'exp': int(time.time()) + (10 * 60), 'iss': APP_ID}
        return jwt.encode(payload, PRIVATE_KEY, algorithm='RS256')
    except Exception as e:
        print(f"Error creating JWT: {e}")
        raise

def get_installation_access_token(installation_id: int) -> str:
    """Gets a temporary access token for a specific installation, using a cache."""
    # Check cache first
    if installation_id in _token_cache:
        token, expiry = _token_cache[installation_id]
        if expiry > datetime.now(timezone.utc):
            # print(f"Using cached token for installation {installation_id}") # Too noisy
            return token

    print(f"Generating new token for installation {installation_id}...")
    app_jwt = create_app_jwt()
    try:
        headers = {"Authorization": f"Bearer {app_jwt}", "Accept": "application/vnd.github.v3+json"}
        url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
        # Use a synchronous httpx call here for simplicity in the auth function
        response = httpx.post(url, headers=headers, timeout=10.0)
        response.raise_for_status()
        
        token_data = response.json()
        if 'token' not in token_data:
            raise ValueError("Error: 'token' not found in response.")
        
        new_token = token_data['token']
        # Cache the token (it expires in 1 hour, we'll cache for 55 mins)
        new_expiry = datetime.now(timezone.utc) + timedelta(minutes=55)
        _token_cache[installation_id] = (new_token, new_expiry)
        
        return new_token
    except httpx.HTTPStatusError as http_err:
        print(f"HTTP error getting installation token: {http_err} - {http_err.response.text}")
        raise
    except Exception as e:
        print(f"Error getting installation token: {e}")
        raise

# --- 1. Load Data Sources ---
def get_all_users_from_config() -> List[str]:
    """
    Loads the pasted JSON string and returns a unique list of user handles.
    """
    print("Loading team structure from pasted JSON config...")
    try:
        team_data = json.loads(TEAM_STRUCTURE_JSON)
    except json.JSONDecodeError:
        print("Error: Could not parse the pasted TEAM_STRUCTURE_JSON. Is it valid JSON?")
        return []

    user_set: Set[str] = set()
    user_set.update(team_data.get('inferred_hierarchy', {}).get('leads', []))
    user_set.update(team_data.get('inferred_hierarchy', {}).get('engineers', []))
    for team_name, members in team_data.get('active_teams', {}).items():
        user_set.update(members)
        
    all_users = sorted(list(user_set))
    print(f"Loaded {len(all_users)} unique users from config.")
    return all_users

def get_all_installed_repos() -> Dict[str, int]:
    """
    Scans the github-installations table.
    Returns a dict of { "repo_name": installation_id }
    """
    print(f"Fetching all installed repos from {INSTALLATIONS_TABLE_NAME}...")
    table = dynamodb.Table(INSTALLATIONS_TABLE_NAME)
    repo_map = {}
    
    try:
        response = table.scan()
        items = response.get('Items', [])
        
        # Handle pagination
        while 'LastEvaluatedKey' in response:
            response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
            items.extend(response.get('Items', []))

        for item in items:
            if 'repo_name' in item and 'installation_id' in item:
                repo_map[item['repo_name']] = int(item['installation_id'])
                
        print(f"Found {len(repo_map)} installed repos.")
        return repo_map
        
    except Exception as e:
        print(f"Error scanning {INSTALLATIONS_TABLE_NAME}: {e}")
        return {}

# --- 2. Fetch GitHub Activity (Production Version) ---
async def fetch_user_commits(
    repo: str, user_handle: str, since_date_iso: str, client: httpx.AsyncClient
) -> List[str]:
    """ Fetches all commit timestamps for a user. """
    commit_timestamps = []
    page = 1
    url = f"https://api.github.com/repos/{repo}/commits"
    params = {"author": user_handle, "since": since_date_iso, "per_page": 100}
    
    print(f"  - Fetching commits for '{user_handle}' in '{repo}' (Page {page})...")
    
    try:
        while True:
            params['page'] = page
            response = await client.get(url, params=params)
            
            if response.status_code in [404, 422]:
                print(f"  - No commits found for '{user_handle}' in '{repo}'.")
                break
            
            response.raise_for_status()
            data = response.json()
            if not data:
                break 
                
            for commit in data:
                date_str = commit.get('commit', {}).get('author', {}).get('date')
                if date_str:
                    commit_timestamps.append(date_str)
            
            if "next" not in response.links:
                break
            page += 1
            await asyncio.sleep(0.2) # Be nice to the API

    except httpx.HTTPStatusError as e:
        print(f"  - HTTP error fetching commits for {user_handle}: {e}")
    except Exception as e:
        print(f"  - UNEXPECTED ERROR fetching commits: {e}")
        
    return commit_timestamps

async def fetch_user_activity(
    repo: str, user_handle: str, since_date_iso: str, client: httpx.AsyncClient
) -> List[str]:
    """ Fetches PRs created and Issues closed by the user. (Your new logic) """
    activity_timestamps = []
    
    try:
        # Search for PRs created
        query = f"repo:{repo} author:{user_handle} is:pr created:>{since_date_iso}"
        search_url = "https://api.github.com/search/issues"
        params = {"q": query, "per_page": 100}
        
        response = await client.get(search_url, params=params)
        response.raise_for_status()
        data = response.json()
        for item in data.get('items', []):
            activity_timestamps.append(item['created_at'])

        # Add issues closed
        await asyncio.sleep(0.5) # Avoid secondary rate limits
        query = f"repo:{repo} closed-by:{user_handle} is:issue closed:>{since_date_iso}"
        params = {"q": query, "per_page": 100}
        response = await client.get(search_url, params=params)
        response.raise_for_status()
        data = response.json()
        for item in data.get('items', []):
            activity_timestamps.append(item['closed_at'])

        print(f"  - Fetched {len(activity_timestamps)} PR/Issue events for '{user_handle}' in '{repo}'.")

    except httpx.HTTPStatusError as e:
        print(f"  - HTTP error fetching activity for {user_handle}: {e}")
    except Exception as e:
        print(f"  - UNEXPECTED ERROR fetching activity: {e}")
        
    return activity_timestamps

# --- 3. Process Data and Call LLM ---
def analyze_activity_with_llm(
    user_handle: str, 
    all_timestamps: List[str]
) -> Dict[str, Any]:
    """
    Converts timestamps to UTC day/hour and sends to DeepSeek.
    Returns a dict with the inferred window and status.
    """
    print(f"  - Analyzing {len(all_timestamps)} total activity events for '{user_handle}'...")
    
    activity_points = []
    for ts_str in all_timestamps:
        try:
            utc_dt = datetime.fromisoformat(ts_str.rstrip('Z')).replace(tzinfo=timezone.utc)
            day_of_week = utc_dt.weekday()  # Monday is 0, Sunday is 6
            hour_of_day = utc_dt.hour      # Hour in UTC (0-23)
            activity_points.append(f"({day_of_week}, {hour_of_day})")
        except Exception:
            continue
            
    # --- This is our "Low Activity" logic ---
    if len(activity_points) < MIN_ACTIVITY_THRESHOLD:
        print(f"  - Not enough activity ({len(activity_points)} < {MIN_ACTIVITY_THRESHOLD}). Assigning default window.")
        return {
            "inferred_start_time_utc": DEFAULT_START_TIME,
            "inferred_end_time_utc": DEFAULT_END_TIME,
            "status": "default_low_activity",
            "data_points_analyzed": len(activity_points)
        }
        
    activity_data_str = ", ".join(activity_points)
    
    system_prompt = f"""You are an expert data analyst. Your task is to analyze a list of (day_of_week, hour_of_day) tuples representing a user's commit/PR/issue activity *in UTC*.
From this data, infer their primary 8-hour working window *in UTC*.
- Focus on the *densest* clusters of activity.
- Ignore sparse, outlier activity (e.g., a single commit at 3 AM UTC).
- 'day_of_week' 0 is Monday, 6 is Sunday.
- You MUST return ONLY a single, valid JSON object with two keys:
  - "inferred_start_time_utc": The start of the 8-hour window (e.g., "09:00").
  - "inferred_end_time_utc": The end of the 8-hour window (e.g., "17:00").
"""
    user_prompt = f"""Here is the user's activity data (day_of_week, hour_of_day) in UTC for the last 90 days:
[{activity_data_str}]

Return ONLY the JSON object.
"""
    
    body_obj = {"messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "max_tokens": 200, "temperature": 0.0, "top_p": 0.9}
    
    print(f"  - Sending {len(activity_points)} data points to DeepSeek...")
    
    try:
        response = bedrock_client.invoke_model(
            modelId=LLM_MODEL_ID, body=json.dumps(body_obj),
            contentType="application/json", accept="application/json"
        )
        response_body = json.loads(response.get('body').read())
        result_text = response_body.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        
        match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if match:
            decision = json.loads(match.group(0))
            if 'inferred_start_time_utc' in decision and 'inferred_end_time_utc' in decision:
                decision["status"] = "inferred_llm"
                decision["data_points_analyzed"] = len(activity_points)
                return decision  # Success!
        
        print(f"  - Error: LLM returned invalid JSON: {result_text}. Assigning default window.")
        
    except Exception as e:
        print(f"  - Error calling Bedrock (DeepSeek): {e}. Assigning default window.")

    # Fallback for LLM errors
    return {
        "inferred_start_time_utc": DEFAULT_START_TIME,
        "inferred_end_time_utc": DEFAULT_END_TIME,
        "status": "default_llm_error",
        "data_points_analyzed": len(activity_points)
    }

# --- 4. Main Orchestration ---
async def fetch_and_analyze_user(
    user_handle: str, 
    repos_map: Dict[str, int], 
    since_date_iso: str
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Orchestrates all API calls for a single user across all repos they have access to.
    """
    print(f"\n--- Processing user: {user_handle} ---")
    all_timestamps = []
    
    # We must group repos by their installation_id to use tokens efficiently
    installs_to_repos: Dict[int, List[str]] = defaultdict(list)
    for repo, install_id in repos_map.items():
        installs_to_repos[install_id].append(repo)
    
    # Process one installation at a time for this user
    for install_id, repos in installs_to_repos.items():
        try:
            token = get_installation_access_token(install_id)
            headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github.v3+json",
                "X-GitHub-Api-Version": "2022-11-28"
            }
            
            async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=30.0) as client:
                tasks = []
                for repo_name in repos:
                    tasks.append(fetch_user_commits(repo_name, user_handle, since_date_iso, client))
                    tasks.append(fetch_user_activity(repo_name, user_handle, since_date_iso, client))
                
                # Run all tasks for this installation concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect all timestamps
                for res in results:
                    if isinstance(res, list):
                        all_timestamps.extend(res)
                    elif isinstance(res, Exception):
                        print(f"  - A fetch task failed: {res}")
        
        except Exception as e:
            print(f"  - Could not get token for install {install_id}. Skipping {len(repos)} repos. Error: {e}")
            
    # Analyze the collected data for this user
    if not all_timestamps:
        print(f"  - No activity found for {user_handle} across any repos.")
        return user_handle, None

    final_window = analyze_activity_with_llm(user_handle, all_timestamps)
    return user_handle, final_window


async def main_orchestrator():
    """
    Main async function to run the whole pipeline.
    """
    start_time = time.time()
    
    # 1. Load users and repos
    all_users = get_all_users_from_config()
    repos_map = get_all_installed_repos()
    
    if not all_users or not repos_map:
        print("No users or no installed repos found. Exiting.")
        return

    since_date = datetime.now(timezone.utc) - timedelta(days=DAYS_TO_SCAN)
    since_date_iso = since_date.isoformat()
    
    final_availability_map = {}
    
    # 2. Process all users
    # We run users one-by-one to avoid primary rate limits
    # and to simplify the token logic.
    for user_handle in all_users:
        handle, window = await fetch_and_analyze_user(
            user_handle,
            repos_map,
            since_date_iso
        )
        if window:
            final_availability_map[handle] = window

    # 3. Save to DynamoDB
    print("\n\n" + "="*50)
    print("--- FINAL INFERRED AVAILABILITY MAP (UTC) ---")
    print(json.dumps(final_availability_map, indent=2))
    
    if not AVAILABILITY_TABLE_NAME:
        print("Warning: USER_AVAILABILITY_TABLE_NAME not set. Skipping DynamoDB save.")
    else:
        print(f"Saving results to DynamoDB table: {AVAILABILITY_TABLE_NAME}...")
        try:
            table = dynamodb.Table(AVAILABILITY_TABLE_NAME)
            
            with table.batch_writer() as batch:
                for user_handle, data in final_availability_map.items():
                    item = {
                        'user_handle': user_handle, # Partition Key
                        'inferred_start_time_utc': data['inferred_start_time_utc'],
                        'inferred_end_time_utc': data['inferred_end_time_utc'],
                        'status': data['status'],
                        'data_points_analyzed': data['data_points_analyzed'],
                        'last_updated_utc': datetime.now(timezone.utc).isoformat()
                    }
                    batch.put_item(Item=item)
            print(f"Successfully saved {len(final_availability_map)} user profiles to DynamoDB.")
            
        except Exception as e:
            print(f"ERROR: Failed to save to DynamoDB: {e}")

    print(f"--- Total execution time: {time.time() - start_time:.2f} seconds ---")


# --- Lambda Handler Entrypoint ---
def lambda_handler(event, context):
    """
    AWS Lambda entrypoint. Triggered by EventBridge schedule.
    """
    print("Starting scheduled job: Calculate User Availability")
    
    # Check for required environment variables
    if not all([SECRET_ARN, INSTALLATIONS_TABLE_NAME, AVAILABILITY_TABLE_NAME]):
        print("FATAL: Missing one or more environment variables:")
        print(f"  SECRET_ARN: {'SET' if SECRET_ARN else 'MISSING'}")
        print(f"  INSTALLATIONS_TABLE_NAME: {'SET' if INSTALLATIONS_TABLE_NAME else 'MISSING'}")
        print(f"  USER_AVAILABILITY_TABLE_NAME: {'SET' if AVAILABILITY_TABLE_NAME else 'MISSING'}")
        return {"statusCode": 500, "body": "Configuration error."}

    try:
        # We need to import asyncio for the Lambda runtime
        import asyncio
        asyncio.run(main_orchestrator())
        return {"statusCode": 200, "body": "Availability calculation complete."}
    except Exception as e:
        print(f"FATAL error in lambda_handler: {e}")
        return {"statusCode": 500, "body": f"An error occurred: {e}"}