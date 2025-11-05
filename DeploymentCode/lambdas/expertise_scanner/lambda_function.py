import json
import boto3
import requests
# import getpass # --- DELETED ---
from datetime import datetime, timedelta, timezone
import time
import asyncio
import httpx
import re
import os
from botocore.config import Config

# --- NEW: Imports for GitHub App Auth ---
import jwt # PyJWT
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

# --- 1. CONFIGURE THESE VALUES ---
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "deepseek.v3-v1:0") 
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "ap-south-1") 

# --- 2. TUNING PARAMETERS ---
ACTIVITY_THRESHOLD = 2
MAX_COMMITS_TO_FETCH_DETAILS = 30
MAX_LLM_CONCURRENCY = 5
MAX_BATCH_CONTRIBUTIONS = 25
MAX_BATCH_SIZE = 3

# --- 3. AWS CLIENTS (Global for Lambda re-use) ---
connect_timeout = 10
read_timeout = 60
boto_config = Config(
    connect_timeout=connect_timeout,
    read_timeout=read_timeout
)
bedrock_client = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION, config=boto_config)
# --- NEW: Clients for Secrets Manager and DynamoDB ---
secrets_client = boto3.client('secretsmanager')
dynamodb = boto3.resource('dynamodb')


# --- NEW: LAMBDA HANDLER (Main Entrypoint) ---
def lambda_handler(event, context):
    
    print("Event Received:")
    print(json.dumps(event)) # Log the full event for debugging
    
    # 1. Parse the event from EventBridge
    # The data we sent is in the 'detail' field.
    try:
        source = event.get('source')
        detail_type = event.get('detail-type')
        
        # Check if this is the event we expect
        if source == "github.webhook.handler" and detail_type == "repository.added":
            print("Processing a 'repository.added' event.")
            
            # The 'detail' field is a JSON string, so we need to load it
            repo_details = event['detail']
            
            installation_id = repo_details.get('installation_id')
            repo_name = repo_details.get('repo_name')
            
            if not installation_id or not repo_name:
                print("Error: Event detail is missing installation_id or repo_name.")
                return {'statusCode': 400, 'body': 'Missing key data in event detail.'}

            print(f"Successfully parsed event for installation ID: {installation_id}")
            print(f"Repository to scan: {repo_name}")

            # --- YOUR MAIN LOGIC GOES HERE ---
            # 2. Authenticate with the installation_id
            # 3. Get the list of contributors for repo_name
            # 4. Run your two-pass prompt analysis on them
            # 5. Save the results (e.g., to S3 or another DynamoDB table)
            
            # For now, we'll just return a success message
            message = f"Successfully received event for repo: {repo_name}"
            return {'statusCode': 200, 'body': message}

        else:
            # This will handle other events, like the scheduled one later
            print(f"Ignoring event from source: {source} and detail-type: {detail_type}")
            return {'statusCode': 200, 'body': 'Event ignored.'}

    except Exception as e:
        print(f"Error processing event: {e}")
        # Log the error and return a 500
        return {
            'statusCode': 500,
            'body': json.dumps(f"Internal server error: {str(e)}")
        }


# --- NEW: AUTHENTICATION FUNCTIONS (Replaces PAT) ---

# --- MODIFIED: Function now accepts app_id ---
def create_app_jwt(private_key_pem, app_id):
    """Creates a short-lived JWT (10 min) to authenticate as the GitHub App."""
    
    # --- DELETED: Hardcoded APP_ID ---
    # APP_ID = "YOUR_GITHUB_APP_ID" 
    
    now = int(time.time())
    payload = {
        'iat': now - 60,       # Issued at time (60s in the past)
        'exp': now + (10 * 60),  # Expiration time (10 minutes)
        'iss': app_id          # --- MODIFIED: Use app_id from argument
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

# --- MODIFIED: Function now accepts app_id ---
def get_installation_access_token(installation_id, private_key_pem, app_id):
    """Exchanges the App JWT for a temporary installation token."""
    
    # --- MODIFIED: Pass app_id to create_app_jwt ---
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

# --- 4. PREPROCESSING & CLEANING ---
# (All your helper functions from here are UNCHANGED, except 'main')

def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 5. GITHUB DATA FETCHING ---

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
            time.sleep(1)

        except requests.exceptions.RequestException as e:
            print(f"Warning: Failed to fetch search results. Error: {e}")
            break
            
    return all_items

def get_github_activity(repo_name, pat, days_to_scan=30):
    # --- MODIFIED: 'pat' is now the installation token ---
    auth_headers = {
        "Authorization": f"Bearer {pat}", 
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
        if not username or "bot" in username:
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

    print(f"Fetched and compressed PR/Issue activity for {len(user_activity_map)} contributors.")
    return user_activity_map

# --- 6. ASYNC PROFILE & COMMIT FETCHING (REFACTORED) ---
# (No changes in this section)
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
    async with httpx.AsyncClient(http2=True) as client:
        tasks = [fetch_one_commit_detail(client, sha, repo_name, headers) for sha in commit_shas]
        results = await asyncio.gather(*tasks)
    
    for file_list in results:
        all_files.update(file_list)
        
    return list(all_files)

# --- 7. PYTHON-BASED FILTERING & SORTING ---
# (No changes in this section)
def preprocess_and_filter(user_activity_map):
    print("Preprocessing, filtering, and sorting users...")
    
    final_profiles = {}
    active_users_with_data = []

    for username, activity_list in user_activity_map.items():
        count = len(activity_list)
        
        if count <= ACTIVITY_THRESHOLD:
            continue
            
        if count > 15:
            role = "Core Maintainer"
        else:
            role = "Active Contributor"
        
        final_profiles[username] = {
            "username": username,
            "inferred_role": role,
            "profile_summary": {},
            "repo_context": {},
            "activity_summary": {
                "total_contributions": count,
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
    
    print(f"Filtered {len(user_activity_map)} users. Found {len(active_users_with_data)} active contributors (>{ACTIVITY_THRESHOLD} contributions).")
    return final_profiles, active_users_with_data

# --- 8. COMMIT COMPRESSION & DYNAMIC BATCHING ---
# (No changes in this section)
def compress_commit_data(commit_messages, file_paths):
    cleaned_messages = [clean_text(msg) for msg in commit_messages]
    
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
        "file_paths": file_paths,
        "common_dirs": list(dirs),
        "common_extensions": list(extensions),
        "message_samples": cleaned_messages[:3]
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
        
        for i in range(len(active_user_list) - 1, -1, -i):
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

# --- 9. PARALLEL LLM CALLS ---
# (No changes in this section, it already reads prompts from files)

def load_prompt(filename):
    """Loads a prompt from a JSON file."""
    try:
        # --- MODIFIED: Prompts must be in the same dir as lambda_function.py ---
        with open(filename, "r") as f:
            prompt_data = json.load(f)
            if "system_prompt_template" in prompt_data and isinstance(prompt_data["system_prompt_template"], list):
                prompt_data["system_prompt_template"] = "\n".join(prompt_data["system_prompt_template"])
            if "system_prompt" in prompt_data and isinstance(prompt_data["system_prompt"], list):
                prompt_data["system_prompt"] = "\n".join(prompt_data["system_prompt"])
            return prompt_data
            
    except FileNotFoundError:
        print(f"FATAL ERROR: Prompt file '{filename}' not found. Exiting.")
        # This will fail the Lambda, which is correct.
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
    if "anthropic.claude" in LLM_MODEL_ID:
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
        if "anthropic.claude" in LLM_MODEL_ID:
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


def generate_team_structure_analysis(active_profiles, system_prompt):
    print(f"\nSending {len(active_profiles)} active profiles to Bedrock LLM (Pass 2)...")
    
    user_prompt = json.dumps(active_profiles, indent=2)
    
    body_obj = {}
    if "anthropic.claude" in LLM_MODEL_ID:
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
        if "anthropic.claude" in LLM_MODEL_ID:
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


# --- 10. MAIN ASYNC ORCHESTRATOR (REFACTORED) ---

# --- MODIFIED: 'main' now accepts args from lambda_handler ---
async def main(repo_name, pat, days_to_scan=30):
    try:
        # --- DELETED: All 'input()' and 'getpass()' calls ---
        
        start_time = time.time()
        
        # --- Load Prompts (unchanged) ---
        pass_1_prompt_data = load_prompt("pass_1_prompt.json")
        pass_2_prompt_data = load_prompt("pass_2_prompt.json")
        
        pass_1_prompt_template = "\n".join(pass_1_prompt_data["system_prompt_template"])
        pass_2_prompt = "\n".join(pass_2_prompt_data["system_prompt"])
        
        llm_semaphore = asyncio.Semaphore(MAX_LLM_CONCURRENCY)

        async def invoke_llm_batch_wrapper(batch, prompt_template):
            async with llm_semaphore:
                return await asyncio.to_thread(invoke_llm_batch, batch, prompt_template)
        
        # --- PASS 0: Fetch & Preprocess (Python) ---
        # --- MODIFIED: Uses args passed to 'main' ---
        user_activity_map = get_github_activity(repo_name, pat, days_to_scan=days_to_scan)
        
        if not user_activity_map:
            print("No activity found. Exiting.")
            return # This will end the Lambda run successfully

        # --- PASS 0.1: Filter & Sort (Python) ---
        final_profiles, sorted_active_users = preprocess_and_filter(user_activity_map)
        
        if not final_profiles:
            print(f"No active contributors found with >{ACTIVITY_THRESHOLD} contributions. Exiting.")
            return # This will end the Lambda run successfully
            
        # --- PASS 0.2: Fetch All Data for ALL active users (Async) ---
        # --- MODIFIED: Uses 'pat' token passed to 'main' ---
        auth_headers = {
            "Authorization": f"Bearer {pat}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        since_date = (datetime.now(timezone.utc) - timedelta(days=days_to_scan)).isoformat()
        
        all_user_commits_data = {}

        async with httpx.AsyncClient(http2=True) as client:
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
            
            # --- Process results (unchanged) ---
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
                        final_profiles[username].update(inferences)

        # --- DELETED: All 'print()' and 'open().write()' calls for expertise_profiles.json ---
        
        # --- PASS 2: Team Structure (Python + 1 LLM Call) ---
        print("\n" + "="*50)
        print("Performing second-level team structure analysis (Pass 2)...")
        print("="*50)
        
        active_profiles_for_llm = {}
        for username, profile in final_profiles.items():
            if "technical_skills" in profile and profile["technical_skills"]:
                active_profiles_for_llm[username] = {
                    "inferred_role": profile["inferred_role"],
                    "technical_skills": profile["technical_skills"],
                    "contribution_types": profile["contribution_types"]
                }
        
        llm_team_analysis = await asyncio.to_thread(
            generate_team_structure_analysis, 
            active_profiles_for_llm,
            pass_2_prompt
        )
        
        final_team_structure = {
            "active_teams": llm_team_analysis.get("active_teams", {}),
            "inferred_hierarchy": llm_team_analysis.get("inferred_hierarchy", {})
        }
        
        # --- DELETED: All 'print()' and 'open().write()' calls for team_structure.json ---

        # --- NEW: Assemble Final Output and Save to DynamoDB ---
        print("Assembling final item for DynamoDB...")
        
        final_output_item = {
            'repo_name': repo_name, # The Primary Key
            'last_updated': datetime.now(timezone.utc).isoformat(),
            'expertise_profiles': for_output_pass_1,
            'team_structure': final_team_structure
        }
        
        try:
            # !! ACTION REQUIRED: Create a DynamoDB table named 'RepoExpertise' !!
            # The global 'dynamodb' resource was created at the top
            table = dynamodb.Table('RepoExpertise') # <-- YOUR NEW TABLE
            
            table.put_item(Item=final_output_item)
            
            print(f"Successfully saved expertise map for '{repo_name}' to DynamoDB.")
            
        except Exception as e:
            print(f"ERROR: Failed to save to DynamoDB. Error: {e}")
            # Re-raise the exception so the Lambda fails and we know about it
            raise e

        end_time = time.time()
        print(f"\n--- Total execution time: {end_time - start_time:.2f} seconds ---")
        # No return needed from main()

    except Exception as e:
        print(f"\nAn unexpected error occurred in main: {e}")
        import traceback
        traceback.print_exc()
        # Re-raise to fail the Lambda
        raise

# --- DELETED: The old entrypoint ---
# if __name__ == "__main__":
#     asyncio.run(main())

