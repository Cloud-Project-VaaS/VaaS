import json
import boto3
import requests
from datetime import datetime, timedelta, timezone
import time
import asyncio
import httpx
import re
import os
from botocore.config import Config
import jwt # PyJWT
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

# --- 1. ENVIRONMENT & CONFIG ---
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "mistral.mistral-7b-instruct-v0:2") 
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "ap-south-1") 
SECRET_ARN = os.environ.get('SECRET_ARN') # <-- THIS MUST BE SET
# --- NEW: Add table name as environment variable ---
INSTALLATIONS_TABLE_NAME = os.environ.get("INSTALLATIONS_TABLE_NAME", "github-installations")
EXPERTISE_TABLE_NAME = os.environ.get("EXPERTISE_TABLE_NAME", "RepoExpertise")

# --- 2. TUNING PARAMETERS ---
# CHANGE: Increased threshold to 5 as requested (Total commits + PRs + Issues > 5)
ACTIVITY_THRESHOLD = 5 
MAX_COMMITS_TO_FETCH_DETAILS = 30
MAX_LLM_CONCURRENCY = 5
MAX_BATCH_CONTRIBUTIONS = 25
MAX_BATCH_SIZE = 3
DAYS_TO_SCAN = 30 # How many days back to scan activity

# --- 3. AWS CLIENTS (Global for Lambda re-use) ---
connect_timeout = 10
read_timeout = 60
boto_config = Config(
    connect_timeout=connect_timeout,
    read_timeout=read_timeout
)
bedrock_client = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION, config=boto_config)
secrets_client = boto3.client('secretsmanager')
dynamodb = boto3.resource('dynamodb')

# --- 4. GITHUB APP AUTHENTICATION ---

def load_secrets():
    """Loads APP_ID and PRIVATE_KEY from AWS Secrets Manager."""
    if not SECRET_ARN:
        raise ValueError("Error: SECRET_ARN environment variable is not set.")
    try:
        secret_response = secrets_client.get_secret_value(SecretId=SECRET_ARN)
        secrets = json.loads(secret_response['SecretString'])
        APP_ID = secrets.get('APP_ID')
        PRIVATE_KEY = secrets.get('PRIVATE_KEY')
        if not APP_ID or not PRIVATE_KEY:
            raise KeyError("APP_ID or PRIVATE_KEY not found in secret.")
        
        # Convert APP_ID from string to int if necessary, as GH API expects int
        return str(APP_ID), PRIVATE_KEY
    except Exception as e:
        print(f"Error getting installation token: {e}")
        raise

def create_app_jwt(private_key_pem, app_id):
    """Creates a short-lived JWT (10 min) to authenticate as the GitHub App."""
    now = int(time.time())
    payload = {
        'iat': now - 60,       # Issued at time (60s in the past)
        'exp': now + (10 * 60),  # Expiration time (10 minutes)
        'iss': app_id          # Issuer (your App ID)
    }
    
    try:
        private_key = serialization.load_pem_private_key(
            private_key_pem.encode(),
            password=None,
            backend=default_backend()
        )
        return jwt.encode(payload, private_key, algorithm='RS256')
    except Exception as e:
        print(f"Error encoding JWT: {e}")
        raise

def get_installation_access_token(installation_id, private_key_pem, app_id):
    """Exchanges the App JWT for a temporary installation token."""
    
    app_jwt = create_app_jwt(private_key_pem, app_id)
    
    headers = {
        "Authorization": f"Bearer {app_jwt}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
    
    try:
        response = requests.post(url, headers=headers)
        response.raise_for_status() # Will raise error if auth fails
        
        data = response.json()
        print("Successfully generated new installation access token.")
        return data.get('token')
    except requests.exceptions.RequestException as e:
        print(f"Error getting installation token: {e}")
        if e.response is not None:
            print(f"Response body: {e.response.text}")
        raise

# --- 5. LAMBDA HANDLER (THE MAIN ENTRYPOINT) ---

def lambda_handler(event, context):
    
    print("Event Received:")
    print(json.dumps(event))
    
    try:
        # --- ROUTER: Check what triggered the function ---
        source = event.get('source')
        detail_type = event.get('detail-type')
        
        # --- TRIGGER 1: On-Install event from our first Lambda ---
        if source == "github.webhook.handler" and detail_type == "repository.added":
            print("Processing a 'repository.added' event.")
            repo_details = event['detail'] # This is a dict
            installation_id = repo_details.get('installation_id')
            repo_name = repo_details.get('repo_name')
            
            if not installation_id or not repo_name:
                print("Error: Event detail is missing installation_id or repo_name.")
                return {'statusCode': 400, 'body': 'Missing key data in event detail.'}

            print(f"Successfully parsed event for installation ID: {installation_id}, Repo: {repo_name}")

            # Load Secrets & Authenticate
            print("Loading secrets from Secrets Manager...")
            APP_ID, PRIVATE_KEY = load_secrets()
            
            print("Generating JWT and getting installation token...")
            install_token = get_installation_access_token(installation_id, PRIVATE_KEY, APP_ID)

            # Run the Main Analysis Logic for ONE repo
            print(f"Starting async main analysis for {repo_name}...")
            asyncio.run(main(repo_name, install_token, DAYS_TO_SCAN))
            
            message = f"Successfully processed {repo_name} and saved results to DynamoDB."
            print(message)
            return {'statusCode': 200, 'body': message}

        # --- TRIGGER 2: Scheduled 7-day event ---
        elif source == "aws.events" and detail_type == "Scheduled Event":
            print("Processing a 'Scheduled Event' for a full weekly scan.")
            
            # Run the full scan logic
            asyncio.run(run_full_scan())
            
            message = "Successfully completed full scheduled scan."
            print(message)
            return {'statusCode': 200, 'body': message}

        # --- Default: Unknown trigger ---
        else:
            print(f"Ignoring event from source: {source} and detail-type: {detail_type}")
            return {'statusCode': 200, 'body': 'Event ignored.'}

    except Exception as e:
        print(f"Error processing event: {e}")
        import traceback
        traceback.print_exc()
        return {
            'statusCode': 500,
            'body': json.dumps(f"Internal server error: {str(e)}")
        }

# --- Full Scan Orchestrator ---
def get_all_installed_repos():
    """Scans the DynamoDB table to get all installation/repo pairs."""
    table = dynamodb.Table(INSTALLATIONS_TABLE_NAME)
    repos_by_install_id = {}
    
    try:
        # Use scan for simplicity. For >1MB tables, pagination would be needed.
        response = table.scan()
        items = response.get('Items', [])
        
        # Handle pagination if the table grows
        while 'LastEvaluatedKey' in response:
            print("Paginating DynamoDB scan...")
            response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
            items.extend(response.get('Items', []))

        print(f"Found {len(items)} items in {INSTALLATIONS_TABLE_NAME} table.")
        
        # Group repos by installation_id to minimize auth calls
        for item in items:
            install_id = item.get('installation_id')
            repo_name = item.get('repo_name')
            if install_id and repo_name:
                if install_id not in repos_by_install_id:
                    repos_by_install_id[install_id] = []
                repos_by_install_id[install_id].append(repo_name)
        
        return repos_by_install_id

    except Exception as e:
        print(f"Error scanning DynamoDB table: {e}")
        return {}

async def run_full_scan():
    """
    Orchestrates a full scan of all repos in the DynamoDB table.
    Gets one token per installation and scans all its repos.
    """
    print("Loading secrets for full scan...")
    APP_ID, PRIVATE_KEY = load_secrets()
    
    print("Fetching all installed repos from DynamoDB...")
    repos_by_install_id = get_all_installed_repos()
    
    if not repos_by_install_id:
        print("No repos found in DynamoDB. Exiting scan.")
        return

    tasks = []
    print(f"Found {len(repos_by_install_id)} unique installations. Generating tokens and creating tasks...")
    
    for install_id, repo_list in repos_by_install_id.items():
        try:
            print(f"Getting token for installation {install_id}...")
            token = get_installation_access_token(install_id, PRIVATE_KEY, APP_ID)
            
            # Create a task for each repo under this installation
            for repo_name in repo_list:
                print(f"  - Queuing analysis for {repo_name}")
                tasks.append(main(repo_name, token, DAYS_TO_SCAN))
                
        except Exception as e:
            print(f"Failed to get token for installation {install_id}. Skipping {len(repo_list)} repos. Error: {e}")
            
    if not tasks:
        print("No valid tasks to run. Exiting scan.")
        return
        
    print(f"--- Starting concurrent analysis for {len(tasks)} total repositories ---")
    await asyncio.gather(*tasks)
    print(f"--- Finished concurrent analysis for all repositories ---")

# --- 6. PREPROCESSING & CLEANING ---
def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'', '', text, flags=re.DOTALL)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 7. GITHUB DATA FETCHING ---

def _run_search_query(search_query, headers):
    all_items = []
    url = "https://api.github.com/search/issues"
    params = {'q': search_query, 'per_page': 100, 'page': 1}
    
    print(f"  - Running search for: {search_query}")

    while True:
        if params['page'] > 10:
            print(f"  - Reached page 10 (1000 results). Stopping search.")
            break
            
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            all_items.extend(data.get('items', []))
            
            if len(all_items) >= data.get('total_count', 0) or len(data.get('items', [])) == 0:
                break
            
            params['page'] += 1
            time.sleep(1) # Be nice to the API

        except requests.exceptions.RequestException as e:
            print(f"Warning: Failed to fetch search results. Error: {e}")
            break
            
    return all_items

def get_github_activity(repo_name, install_token, days_to_scan):
    auth_headers = {
        "Authorization": f"Bearer {install_token}", 
        "Accept": "application/vnd.github.v3+json"
    }
    
    if '/' not in repo_name:
        raise ValueError("Invalid repo_name format. Expected 'owner/repo'.")
        
    since_date = (datetime.now(timezone.utc) - timedelta(days=days_to_scan)).replace(microsecond=0).isoformat()
    
    print(f"Fetching data for {repo_name} (last {days_to_scan} days) using Search API...")
    
    pr_query = f"repo:{repo_name} is:pr is:merged updated:>{since_date}"
    pulls = _run_search_query(pr_query, auth_headers)
    
    issue_query = f"repo:{repo_name} is:issue is:closed updated:>{since_date}"
    issues = _run_search_query(issue_query, auth_headers)
    
    user_activity_map = {}

    def add_compressed_activity(username, item, item_type, role):
        if not username or "bot" in username.lower():
            return
            
        cleaned_title = clean_text(item.get('title'))
        
        if "spam" in cleaned_title:
            return
            
        if username not in user_activity_map:
            user_activity_map[username] = []
        
        compressed_item = {
            "title": cleaned_title,
            "type": item_type,
            "role": role,
            "labels": [clean_text(label.get('name')) for label in item.get('labels', [])]
        }
        user_activity_map[username].append(compressed_item)

    for item in issues:
        for assignee in item.get('assignees', []):
            assignee_username = assignee.get('login') if assignee else None
            add_compressed_activity(assignee_username, item, "issue", "assignee")
            
    for pr in pulls:
        pr_creator_username = pr.get('user', {}).get('login')
        add_compressed_activity(pr_creator_username, pr, "pull_request", "creator")
        
        for assignee in pr.get('assignees', []):
            assignee_username = assignee.get('login') if assignee else None
            add_compressed_activity(assignee_username, pr, "pull_request", "assignee")

    # --- UPDATED SECTION: Fetch all contributors and their commit counts ---
    print(f"Fetching full contributor list for {repo_name} as fallback and for commit counts...")
    user_commit_counts = {}
    
    try:
        contrib_url = f"https://api.github.com/repos/{repo_name}/contributors"
        # We ask for anon=false to ensure we only get logged-in users
        params = {'anon': 'false', 'per_page': 100}
        
        response = requests.get(contrib_url, headers=auth_headers, params=params)
        response.raise_for_status()
        contributors = response.json()
        
        for contributor in contributors:
            username = contributor.get('login')
            if not username or "bot" in username.lower():
                continue
            
            # CHANGE: Capture commit count
            user_commit_counts[username] = contributor.get('contributions', 0)
            
            # If they are not already in our map from PRs/Issues, add them with empty list
            if username not in user_activity_map:
                user_activity_map[username] = [] 
                
        print(f"Found {len(contributors)} total contributors.")
        
    except requests.exceptions.RequestException as e:
        print(f"Warning: Could not fetch contributor list. Error: {e}")
    # --- END OF UPDATED SECTION ---

    print(f"Fetched and compressed PR/Issue activity for {len(user_activity_map)} contributors.")
    # CHANGE: Return both maps
    return user_activity_map, user_commit_counts

# --- 8. ASYNC PROFILE & COMMIT FETCHING ---
async def fetch_repo_languages(client, repo_name, headers):
    print(f"Fetching repo languages for {repo_name}...")
    url = f"https://api.github.com/repos/{repo_name}/languages"
    try:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        languages = response.json()
        
        total_bytes = sum(languages.values())
        if total_bytes == 0:
            return {}
        lang_percent = {
            lang: round((bytes / total_bytes) * 100, 2)
            for lang, bytes in languages.items()
        }
        print("Repo languages fetched.")
        return lang_percent
    except Exception as e:
        print(f"Warning: Could not fetch repo languages. Error: {e}")
        return {}

async def fetch_one_profile(client, username, headers):
    url = f"https://api.github.com/users/{username}"
    try:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        profile_data = response.json()
        return username, {
            "name": profile_data.get('name'),
            "bio": clean_text(profile_data.get('bio'))
        }
    except httpx.HTTPStatusError:
        return username, {}
    except Exception:
        return username, {}

async def fetch_one_users_commits(client, username, repo_name, since_date, headers):
    url = f"https://api.github.com/repos/{repo_name}/commits"
    params = {'author': username, 'since': since_date, 'per_page': 100}
    
    try:
        response = await client.get(url, headers=headers, params=params)
        response.raise_for_status()
        commits_data = response.json()
        return username, [
            (c.get('sha'), c.get('commit', {}).get('message', ''))
            for c in commits_data
        ]
    except Exception:
        return username, []

async def fetch_one_commit_detail(client, sha, repo_name, headers):
    url = f"https://api.github.com/repos/{repo_name}/commits/{sha}"
    try:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        commit_detail = response.json()
        files = commit_detail.get('files', [])
        return [f.get('filename') for f in files if f.get('filename')]
    except Exception:
        return []

async def fetch_commit_details_concurrently(commit_shas, repo_name, headers):
    if not commit_shas:
        return []
        
    all_files = set()
    async with httpx.AsyncClient() as client:
        tasks = [fetch_one_commit_detail(client, sha, repo_name, headers) for sha in commit_shas]
        results = await asyncio.gather(*tasks)
    
    for file_list in results:
        all_files.update(file_list)
        
    return list(all_files)

# --- 9. PYTHON-BASED FILTERING & SORTING (UPDATED) ---
def preprocess_and_filter(user_activity_map, user_commit_counts):
    print("Preprocessing, filtering, and sorting users...")
    
    final_profiles = {}
    active_users_with_data = []

    for username, activity_list in user_activity_map.items():
        # Get count of PRs + Issues
        issue_pr_count = len(activity_list)
        # Get count of Commits (from contributors API)
        commit_count = user_commit_counts.get(username, 0)
        
        total_activity = issue_pr_count + commit_count
        
        # --- UPDATED FILTER (Total Commits + PRs + Issues > Threshold) ---
        if total_activity <= ACTIVITY_THRESHOLD:
            continue
        # --- END OF FILTER ---
        
        if total_activity > 20:
            role = "Core Maintainer"
        else:
            role = "Active Contributor" 
        
        final_profiles[username] = {
            "username": username,
            "inferred_role": role,
            "profile_summary": {},
            "repo_context": {},
            "activity_summary": {
                "total_contributions": total_activity,
                "commits_count": commit_count, # Added this for clarity
                "issues_closed": len([item for item in activity_list if item['type'] == 'issue']),
                "prs_merged": len([item for item in activity_list if item['type'] == 'pull_request'])
            },
            "contributions": activity_list,
            "commit_summary": {},
            "technical_skills": [],
            "contribution_types": [],
            "confidence": "Low"
        }
        
        active_users_with_data.append(username)

    active_users_with_data.sort(
        key=lambda u: final_profiles[u]['activity_summary']['total_contributions'], 
        reverse=True
    )
    
    print(f"Filtered {len(user_activity_map)} total users. Found {len(active_users_with_data)} active contributors (>{ACTIVITY_THRESHOLD} contributions).")
    return final_profiles, active_users_with_data

# --- 10. COMMIT COMPRESSION & DYNAMIC BATCHING ---
def compress_commit_data(commit_messages, file_paths):
    # Clean all messages
    cleaned_messages = [clean_text(msg) for msg in commit_messages]
    
    # Still extract keywords for high-level context
    keywords = set()
    keyword_regex = re.compile(
        r'\b(fix|feat|refactor|docs|test|style|chore|build|ci|perf|revert|autotuner|gpu|cuda|tpu|keras|model)\b', 
        re.IGNORECASE
    )
    for msg in cleaned_messages:
        matches = keyword_regex.findall(msg)
        keywords.update(m.lower() for m in matches)
    
    dirs = set()
    extensions = set()
    for path in file_paths:
        if path and '/' in path:
            dirs.add(os.path.dirname(path) + '/')
        if path:
            ext = os.path.splitext(path)[1]
            if ext:
                extensions.add(ext)

    return {
        "keywords": list(keywords),
        "file_paths": file_paths, # The LLM will see specific files touched
        "common_dirs": list(dirs),
        "common_extensions": list(extensions),
        "recent_commit_messages": cleaned_messages 
    }

def create_dynamic_batches(sorted_active_users, all_profiles, all_user_commits_data):
    print(f"Creating dynamic batches (Max Weight: {MAX_BATCH_CONTRIBUTIONS}, Max Size: {MAX_BATCH_SIZE})...")
    batches = []
    
    active_user_list = list(sorted_active_users) 
    
    batch_num = 1
    while active_user_list:
        current_batch_data = {}
        current_batch_weight = 0
        
        username = active_user_list.pop(0)
        user_profile = all_profiles[username]
        # Use simple issue/pr count for weight, or total. Total is safer for batching.
        count = user_profile['activity_summary']['total_contributions']
        
        commit_messages, file_paths = all_user_commits_data.get(username, ([], []))
        commit_summary = compress_commit_data(commit_messages, file_paths)
        
        current_batch_data[username] = {
            "username": username,
            "profile_summary": user_profile["profile_summary"],
            "repo_context": user_profile["repo_context"],
            "activity_summary": user_profile["activity_summary"],
            "contributions": user_profile["contributions"],
            "commit_summary": commit_summary
        }
        current_batch_weight += count
        
        # Iterate backwards to safely remove items while looping
        for i in range(len(active_user_list) - 1, -1, -1):
            username_light = active_user_list[i]
            user_profile_light = all_profiles[username_light]
            count_light = user_profile_light['activity_summary']['total_contributions']
            
            if (len(current_batch_data) < MAX_BATCH_SIZE) and \
               (current_batch_weight + count_light <= MAX_BATCH_CONTRIBUTIONS):
                
                commit_messages_light, file_paths_light = all_user_commits_data.get(username_light, ([], []))
                commit_summary_light = compress_commit_data(commit_messages_light, file_paths_light)
                
                current_batch_data[username_light] = {
                    "username": username_light,
                    "profile_summary": user_profile_light["profile_summary"],
                    "repo_context": user_profile_light["repo_context"],
                    "activity_summary": user_profile_light["activity_summary"],
                    "contributions": user_profile_light["contributions"],
                    "commit_summary": commit_summary_light
                }
                current_batch_weight += count_light
                active_user_list.pop(i)
        
        print(f"  - Batch {batch_num}: {list(current_batch_data.keys())} (Weight: {current_batch_weight})")
        batches.append(current_batch_data)
        batch_num += 1

    print(f"Created {len(batches)} batches from {len(sorted_active_users)} users.")
    return batches

# --- 11. PARALLEL LLM CALLS ---

def load_prompt(filename):
    """Loads a prompt from a JSON file."""
    try:
        # Prompts must be in the same directory as the lambda_function.py
        with open(filename, "r") as f:
            prompt_data = json.load(f)
            if "system_prompt_template" in prompt_data and isinstance(prompt_data["system_prompt_template"], list):
                prompt_data["system_prompt_template"] = "\n".join(prompt_data["system_prompt_template"])
            if "system_prompt" in prompt_data and isinstance(prompt_data["system_prompt"], list):
                prompt_data["system_prompt"] = "\n".join(prompt_data["system_prompt"])
            return prompt_data
            
    except FileNotFoundError:
        print(f"FATAL ERROR: Prompt file '{filename}' not found. Exiting.")
        raise
    except json.JSONDecodeError:
        print(f"FATAL ERROR: Prompt file '{filename}' is not valid JSON. Exiting.")
        raise

def invoke_llm_batch(batch_data, prompt_template_str):
    num_users = len(batch_data)
    usernames = list(batch_data.keys())
    
    try:
        system_prompt = prompt_template_str.format(num_users=num_users)
    except KeyError:
        system_prompt = prompt_template_str
    
    user_prompt = json.dumps(batch_data, indent=2)
    
    body_obj = {}
    if "mistral" in LLM_MODEL_ID:
        # Mistral Instruct format: <s>[INST] Instruction\n\nInput [/INST]
        final_prompt = f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"
        body_obj = {
            "prompt": final_prompt,
            "max_tokens": 4096,
            "temperature": 0.1,
            "top_p": 0.9
        }
    elif "anthropic.claude" in LLM_MODEL_ID:
        messages = [{"role": "user", "content": user_prompt}]
        body_obj = {
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.1,
            "top_p": 0.9,
            "system": system_prompt,
            "anthropic_version": "bedrock-2023-05-31"
        }
    elif "openai.gpt-oss-120b" in LLM_MODEL_ID or "deepseek" in LLM_MODEL_ID:
        body_obj = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 4096,
            "temperature": 0.1,
            "top_p": 0.9
        }
    else:
        raise ValueError(f"Unsupported model ID: {LLM_MODEL_ID}. Please update 'invoke_llm_batch'.")

    raw_output = ""
    try:
        response = bedrock_client.invoke_model(
            modelId=LLM_MODEL_ID,
            body=json.dumps(body_obj),
            contentType="application/json",
            accept="application/json"
        )
        response_body = json.loads(response.get('body').read())
        
        raw_output = ""
        if "mistral" in LLM_MODEL_ID:
            raw_output = response_body.get('outputs', [{}])[0].get('text', '')
        elif "anthropic.claude" in LLM_MODEL_ID:
            raw_output = response_body.get('content', [{}])[0].get('text', '')
        elif "openai.gpt-oss-120b" in LLM_MODEL_ID or "deepseek" in LLM_MODEL_ID:
            raw_output = response_body.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        start_index = raw_output.find('{')
        end_index = raw_output.rfind('}')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_output = raw_output[start_index:end_index+1]
            parsed_json = json.loads(json_output)
            print(f"  - [LLM Pass 1] Successfully processed batch for: {', '.join(usernames)}")
            return parsed_json
        else:
            raise json.JSONDecodeError("No JSON object found", raw_output, 0)
            
    except Exception as e:
        print(f"ERROR: LLM (Pass 1 Batch) failed for users {usernames}. Error: {e}")
        print(f"Raw output from LLM: {raw_output}")
        return {}


def generate_team_structure_analysis(active_profiles_data, system_prompt):
    print(f"\nSending {len(active_profiles_data)} active profiles to Bedrock LLM (Pass 2)...")
    
    user_prompt = json.dumps(active_profiles_data, indent=2)
    
    body_obj = {}
    if "mistral" in LLM_MODEL_ID:
        # Mistral Instruct format
        final_prompt = f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"
        body_obj = {
            "prompt": final_prompt,
            "max_tokens": 4096,
            "temperature": 0.1,
            "top_p": 0.9
        }
    elif "anthropic.claude" in LLM_MODEL_ID:
        messages = [{"role": "user", "content": user_prompt}]
        body_obj = {
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.1,
            "top_p": 0.9,
            "system": system_prompt,
            "anthropic_version": "bedrock-2023-05-31"
        }
    elif "openai.gpt-oss-120b" in LLM_MODEL_ID or "deepseek" in LLM_MODEL_ID:
        body_obj = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 4096,
            "temperature": 0.1,
            "top_p": 0.9
        }
    else:
        raise ValueError(f"Unsupported model ID: {LLM_MODEL_ID}. Please update 'generate_team_structure_analysis'.")
    
    raw_output = ""
    try:
        response = bedrock_client.invoke_model(
            modelId=LLM_MODEL_ID,
            body=json.dumps(body_obj),
            contentType="application/json",
            accept="application/json"
        )
        response_body = json.loads(response.get('body').read())
        
        raw_output = ""
        if "mistral" in LLM_MODEL_ID:
            raw_output = response_body.get('outputs', [{}])[0].get('text', '')
        elif "anthropic.claude" in LLM_MODEL_ID:
            raw_output = response_body.get('content', [{}])[0].get('text', '')
        elif "openai.gpt-oss-120b" in LLM_MODEL_ID or "deepseek" in LLM_MODEL_ID:
            raw_output = response_body.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        start_index = raw_output.find('{')
        end_index = raw_output.rfind('}')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_output = raw_output[start_index:end_index+1]
            parsed_json = json.loads(json_output)
            print("  - [LLM Pass 2] Successfully processed team structure.")
            return parsed_json
        else:
            raise json.JSONDecodeError("No JSON object found", raw_output, 0)
            
    except Exception as e:
        print(f"ERROR: LLM (Pass 2) failed. Error: {e}")
        print(f"Raw output from LLM: {raw_output}")
        return {}


# --- 12. MAIN ASYNC ORCHESTRATOR ---

async def main(repo_name, install_token, days_to_scan):
    """The main async function to run the full analysis."""
    try:
        start_time = time.time()
        
        # --- Load Prompts ---
        pass_1_prompt_data = load_prompt("pass_1_prompt.json")
        pass_2_prompt_data = load_prompt("pass_2_prompt.json")
        
        pass_1_prompt_template = pass_1_prompt_data["system_prompt_template"]
        pass_2_prompt = pass_2_prompt_data["system_prompt"]
        
        llm_semaphore = asyncio.Semaphore(MAX_LLM_CONCURRENCY)

        async def invoke_llm_batch_wrapper(batch, prompt_template):
            async with llm_semaphore:
                return await asyncio.to_thread(invoke_llm_batch, batch, prompt_template)
        
        # --- PASS 0: Fetch & Preprocess (Python) ---
        # CHANGE: unpack both user_activity and commit_counts
        user_activity_map, user_commit_counts = get_github_activity(repo_name, install_token, days_to_scan)
        
        if not user_activity_map:
            print("No activity found. Exiting.")
            # Create an empty item in DynamoDB so we know we've scanned it
            final_output_item = {
                'repo_name': repo_name,
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'expertise_profiles': {},
                'team_structure': {}
            }
            table = dynamodb.Table(EXPERTISE_TABLE_NAME)
            table.put_item(Item=final_output_item)
            print(f"Saved empty item for '{repo_name}' to DynamoDB.")
            return

        # --- PASS 0.1: Filter & Sort (Python) ---
        # CHANGE: Pass both maps to filter function
        final_profiles, sorted_active_users = preprocess_and_filter(user_activity_map, user_commit_counts)
        
        if not final_profiles:
            print(f"No active contributors found with >{ACTIVITY_THRESHOLD} contributions. Exiting.")
            # Create an empty item in DynamoDB
            final_output_item = {
                'repo_name': repo_name,
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'expertise_profiles': {},
                'team_structure': {}
            }
            table = dynamodb.Table(EXPERTISE_TABLE_NAME)
            table.put_item(Item=final_output_item)
            print(f"Saved empty item for '{repo_name}' to DynamoDB as no active users were found.")
            return
            
        # --- PASS 0.2: Fetch All Data for ALL active users (Async) ---
        auth_headers = {
            "Authorization": f"Bearer {install_token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        since_date = (datetime.now(timezone.utc) - timedelta(days=days_to_scan)).isoformat()
        
        all_user_commits_data = {}

        async with httpx.AsyncClient() as client:
            # 1. Fetch languages, profiles, and commit lists concurrently
            lang_task = fetch_repo_languages(client, repo_name, auth_headers)
            
            profile_tasks = [
                fetch_one_profile(client, username, auth_headers) 
                for username in sorted_active_users
            ]
            commit_list_tasks = [
                fetch_one_users_commits(client, username, repo_name, since_date, auth_headers)
                for username in sorted_active_users
            ]
            
            print(f"Fetching repo languages, {len(profile_tasks)} profiles, and {len(commit_list_tasks)} commit lists...")
            
            repo_languages, profile_results, commit_list_results = await asyncio.gather(
                lang_task,
                asyncio.gather(*profile_tasks),
                asyncio.gather(*commit_list_tasks)
            )
            
            # --- Process results ---
            for username in final_profiles:
                final_profiles[username]["repo_context"]["languages"] = repo_languages
            
            for username, profile_data in profile_results:
                if username in final_profiles:
                    final_profiles[username]["profile_summary"] = profile_data
            
            user_commit_messages = {}
            commit_detail_tasks = []
            
            for username, commit_list in commit_list_results:
                 if username in final_profiles:
                    shas = [sha for sha, msg in commit_list[:MAX_COMMITS_TO_FETCH_DETAILS] if sha]
                    messages = [msg for sha, msg in commit_list]
                    user_commit_messages[username] = messages
                    
                    task = fetch_commit_details_concurrently(shas, repo_name, auth_headers)
                    commit_detail_tasks.append((username, task))
            
            print(f"Fetching file path details for {len(commit_detail_tasks)} users...")
            detail_results = await asyncio.gather(*[task for username, task in commit_detail_tasks])
            
            user_file_paths = {}
            for (username, task), files in zip(commit_detail_tasks, detail_results):
                user_file_paths[username] = files
            
            for username in sorted_active_users:
                messages = user_commit_messages.get(username, [])
                files = user_file_paths.get(username, [])
                all_user_commits_data[username] = (messages, files)

        print("All data fetching complete.")
        
        # --- PASS 0.3: Dynamic Batching (Python) ---
        batches = create_dynamic_batches(
            sorted_active_users, 
            final_profiles, 
            all_user_commits_data
        )

        # --- PASS 1: Parallel LLM Calls (Async) ---
        print(f"\nStarting Pass 1: Running all LLM batches in parallel (max {MAX_LLM_CONCURRENCY})...")
        
        llm_tasks = [
            invoke_llm_batch_wrapper(batch, pass_1_prompt_template)
            for batch in batches
        ]
        
        batch_results = await asyncio.gather(*llm_tasks)
        print("All Pass 1 LLM batches complete.")

        # --- PASS 1.1: Assemble LLM Results (Python) ---
        for_output_pass_1 = {}
        for result in batch_results:
            if result:
                for username, inferences in result.items():
                    if username in final_profiles:
                        base_profile = final_profiles[username]
                        
                        for_output_pass_1[username] = {
                            "username": base_profile["username"],
                            "inferred_role": base_profile["inferred_role"],
                            "profile_summary": base_profile["profile_summary"],
                            "activity_summary": base_profile["activity_summary"],
                            "technical_skills": inferences.get("technical_skills", []),
                            "contribution_types": inferences.get("contribution_types", []),
                            "confidence": inferences.get("confidence", "Low")
                        }
                        # Update the main profile dict for the next pass
                        final_profiles[username].update(inferences)
        
        # --- PASS 2: Team Structure (Python + 1 LLM Call) (UPDATED) ---
        print("\n" + "="*50)
        print("Performing second-level team structure analysis (Pass 2)...")
        print("="*50)
        
        # --- HIERARCHY FIX ---
        print(f"Sending {len(for_output_pass_1)} active profiles to Pass 2...")

        if not for_output_pass_1:
            print("No active profiles with skills found after Pass 1. Skipping Pass 2.")
            llm_team_analysis = {} # Default to empty analysis
        else:
            llm_team_analysis = await asyncio.to_thread(
                generate_team_structure_analysis, 
                for_output_pass_1, 
                pass_2_prompt
            )
        # --- END HIERARCHY FIX ---
        
        final_team_structure = {
            "active_teams": llm_team_analysis.get("active_teams", {}),
            "inferred_hierarchy": llm_team_analysis.get("inferred_hierarchy", {})
        }
        
        # --- PASS 3: Assemble Final Output and Save to DynamoDB ---
        print("Assembling final item for DynamoDB...")
        
        final_output_item = {
            'repo_name': repo_name, # The Primary Key
            'last_updated': datetime.now(timezone.utc).isoformat(),
            'expertise_profiles': for_output_pass_1,
            'team_structure': final_team_structure
        }
        
        try:
            # Assumes a DynamoDB table named 'RepoExpertise' exists
            table = dynamodb.Table(EXPERTISE_TABLE_NAME)
            table.put_item(Item=final_output_item)
            print(f"Successfully saved expertise map for '{repo_name}' to DynamoDB.")
            
        except Exception as e:
            print(f"ERROR: Failed to save to DynamoDB. Error: {e}")
            raise e

        end_time = time.time()
        print(f"\n--- Total execution time: {end_time - start_time:.2f} seconds ---")

    except Exception as e:
        print(f"\nAn unexpected error occurred in main: {e}")
        import traceback
        traceback.print_exc()
        # Ensure we don't accidentally return a success to Lambda
        raise e