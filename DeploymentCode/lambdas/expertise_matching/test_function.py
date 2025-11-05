import json
import boto3
import requests
import getpass
from datetime import datetime, timedelta, timezone
import time
import asyncio  # For concurrent I/O
import httpx    # For concurrent profile fetching
import re       # For cleaning text
import os       # For environment variables
from botocore.config import Config # --- NEW: To add timeouts ---

# --- 1. CONFIGURE THESE VALUES ---
# (You can also set these as environment variables)
# --- MODIFIED: Switched to DeepSeek Coder ---
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "deepseek.v3-v1:0") 
# --- MODIFIED: Reverted region back to your preference ---
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "ap-south-1") 

# --- 2. TUNING PARAMETERS ---
ACTIVITY_THRESHOLD = 2 # Min contributions to be "active" (<= 2 is inactive)
MAX_COMMITS_TO_FETCH_DETAILS = 30 # Max commit SHAs to get file paths for
MAX_LLM_CONCURRENCY = 5 # Max number of parallel LLM calls

# --- Smart Batching (Your suggestion) ---
# --- MODIFIED: Further reduced batch sizes to prevent hangs ---
MAX_BATCH_CONTRIBUTIONS = 25 # Max total contributions in a single LLM call
MAX_BATCH_SIZE = 3           # Max number of users in a single LLM call

# --- 3. AWS CLIENT (Global for Lambda re-use) ---
# --- MODIFIED: Added a 60-second read timeout ---
connect_timeout = 10
read_timeout = 60
boto_config = Config(
    connect_timeout=connect_timeout,
    read_timeout=read_timeout
)
bedrock_client = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION, config=boto_config)

# --- 4. PREPROCESSING & CLEANING ---

def clean_text(text):
    """
    Cleans text by:
    - Lowercasing
    - Removing emojis and all non-ASCII characters
    - Removing HTML comments
    - Collapsing whitespace
    """
    if not text:
        return ""
    text = text.lower()
    # Remove HTML comments first
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = re.sub(r'[^\x00-\x7F]+', '', text) # Remove emojis/non-ASCII
    text = re.sub(r'\s+', ' ', text).strip()  # Collapse whitespace
    return text

# --- 5. GITHUB DATA FETCHING ---

def _run_search_query(search_query, headers):
    """
    Helper function to run a GitHub Search API query and handle its pagination.
    Returns a list of all 'items' found.
    """
    all_items = []
    url = "https://api.github.com/search/issues"
    params = {'q': search_query, 'per_page': 100, 'page': 1}
    
    print(f"  - Running search for: {search_query}")

    while True:
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            all_items.extend(data.get('items', []))
            
            # Search API pagination
            if len(all_items) >= data.get('total_count', 0) or len(data.get('items', [])) == 0:
                break
            
            params['page'] += 1
            # Add a small delay to respect rate limits on paginated search
            time.sleep(1)

        except requests.exceptions.RequestException as e:
            print(f"Warning: Failed to fetch search results. Error: {e}")
            break
            
    return all_items

def get_github_activity(repo_name, pat, days_to_scan=30):
    """
    Fetches and PREPROCESSES all activity (Issues & PRs).
    It now returns a 'user_activity_map' with *compressed, clean* data.
    """
    auth_headers = {
        "Authorization": f"Bearer {pat}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    if '/' not in repo_name:
        raise ValueError("Invalid repo_name format. Expected 'owner/repo'.")
        
    since_date = (datetime.now(timezone.utc) - timedelta(days=days_to_scan)).isoformat()
    
    print(f"Fetching data for {repo_name} (last {days_to_scan} days) using Search API...")
    
    # --- 1. Fetch all raw data ---
    pr_query = f"repo:{repo_name} is:pr is:merged updated:>{since_date}"
    pulls = _run_search_query(pr_query, auth_headers)
    
    issue_query = f"repo:{repo_name} is:issue is:closed updated:>{since_date}"
    issues = _run_search_query(issue_query, auth_headers)
    
    # --- 2. Process and Compress (Your suggestion) ---
    user_activity_map = {}

    def add_compressed_activity(username, item, item_type, role):
        if not username or "bot" in username:
            return
            
        # --- COMPRESSION & CLEANING ---
        cleaned_title = clean_text(item.get('title'))
        
        # --- Filter out spam/junk issues ---
        if "spam" in cleaned_title:
            return
            
        if username not in user_activity_map:
            user_activity_map[username] = []
        
        # --- NEW PAYLOAD (no body) ---
        compressed_item = {
            "title": cleaned_title,
            "type": item_type,
            "role": role,
            "labels": [clean_text(label.get('name')) for label in item.get('labels', [])]
        }
        user_activity_map[username].append(compressed_item)

    # Process issues
    for item in issues:
        for assignee in item.get('assignees', []):
            assignee_username = assignee.get('login') if assignee else None
            add_compressed_activity(assignee_username, item, "issue", "assignee")
            
    # Process Pull Requests
    for pr in pulls:
        pr_creator_username = pr.get('user', {}).get('login')
        add_compressed_activity(pr_creator_username, pr, "pull_request", "creator")
        
        for assignee in pr.get('assignees', []):
            assignee_username = assignee.get('login') if assignee else None
            add_compressed_activity(assignee_username, pr, "pull_request", "assignee")

    print(f"Fetched and compressed PR/Issue activity for {len(user_activity_map)} contributors.")
    return user_activity_map

# --- 6. ASYNC PROFILE & COMMIT FETCHING (REFACTORED) ---

async def fetch_repo_languages(client, repo_name, headers):
    """Fetches the language breakdown for the repo."""
    print(f"Fetching repo languages for {repo_name}...")
    url = f"https://api.github.com/repos/{repo_name}/languages"
    try:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        languages = response.json()
        
        # Convert to percentages
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
    """Helper for fetching a single profile asynchronously."""
    # print(f"Fetching profile for {username}...") # Too noisy for prod
    url = f"https://api.github.com/users/{username}"
    try:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        profile_data = response.json()
        # Return only the data we need
        return username, {
            "name": profile_data.get('name'),
            "bio": clean_text(profile_data.get('bio'))
        }
    except httpx.HTTPStatusError:
        # print(f"Warning: Could not fetch profile for {username}.")
        return username, {}
    except Exception:
        # print(f"Warning: Could not fetch profile for {username}.")
        return username, {}

async def fetch_one_users_commits(client, username, repo_name, since_date, headers):
    """
    Helper for fetching a single user's commits.
    Returns a list of (SHA, commit_message) tuples.
    """
    # print(f"Fetching commit list for {username}...") # Too noisy for prod
    url = f"https://api.github.com/repos/{repo_name}/commits"
    # We filter by author (username) and date, get max 100
    params = {'author': username, 'since': since_date, 'per_page': 100}
    
    try:
        response = await client.get(url, headers=headers, params=params)
        response.raise_for_status()
        commits_data = response.json()
        # Return (sha, message) tuples
        return username, [
            (c.get('sha'), c.get('commit', {}).get('message', ''))
            for c in commits_data
        ]
    except Exception:
        # print(f"Warning: Could not fetch commits for {username}.")
        return username, []

async def fetch_one_commit_detail(client, sha, repo_name, headers):
    """Fetches the details for a single commit SHA to get file paths."""
    url = f"https://api.github.com/repos/{repo_name}/commits/{sha}"
    try:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        commit_detail = response.json()
        files = commit_detail.get('files', [])
        # Return just the filenames
        return [f.get('filename') for f in files if f.get('filename')]
    except Exception:
        # Fail silently on 404 or other errors
        return []

async def fetch_commit_details_concurrently(commit_shas, repo_name, headers):
    """
    Fetches details for a list of commit SHAs in parallel to get file paths.
    """
    if not commit_shas:
        return []
        
    # print(f"Fetching file paths for {len(commit_shas)} commits...") # Too noisy
    all_files = set()
    async with httpx.AsyncClient(http2=True) as client:
        tasks = [fetch_one_commit_detail(client, sha, repo_name, headers) for sha in commit_shas]
        results = await asyncio.gather(*tasks)
    
    # Flatten the list of lists and unique-ify
    for file_list in results:
        all_files.update(file_list)
        
    # print(f"Found {len(all_files)} unique changed files from {len(commit_shas)} commits.")
    return list(all_files)

# --- 7. PYTHON-BASED FILTERING & SORTING ---

def preprocess_and_filter(user_activity_map):
    """
    Applies deterministic rules to all users.
    FILTERS OUT inactive users completely.
    Returns a dict of ONLY active profiles AND a sorted list of active users.
    """
    print("Preprocessing, filtering, and sorting users...")
    
    final_profiles = {}
    active_users_with_data = []

    for username, activity_list in user_activity_map.items():
        count = len(activity_list)
        
        # --- Skip user if they are inactive (<= 2) ---
        if count <= ACTIVITY_THRESHOLD:
            continue
            
        # --- User is ACTIVE if we reach this point ---
        # Rule-based role assignment
        if count > 15:
            role = "Core Maintainer"
        else:
            role = "Active Contributor"
        
        # Create a base profile for the ACTIVE user
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
            # These will be filled by the LLM
            "technical_skills": [],
            "contribution_types": [],
            "confidence": "Low"
        }
        
        # Add them to the list for profile fetching and LLM analysis
        active_users_with_data.append(username)

    # Sort active users by contribution count (descending)
    active_users_with_data.sort(
        key=lambda u: final_profiles[u]['activity_summary']['total_contributions'], 
        reverse=True
    )
    
    print(f"Filtered {len(user_activity_map)} users. Found {len(active_users_with_data)} active contributors (>{ACTIVITY_THRESHOLD} contributions).")
    return final_profiles, active_users_with_data

# --- 8. COMMIT COMPRESSION & DYNAMIC BATCHING ---

def compress_commit_data(commit_messages, file_paths):
    """
    Takes commit messages and file paths and compresses them into
    a signal-rich object for the LLM.
    """
    cleaned_messages = [clean_text(msg) for msg in commit_messages]
    
    # Extract keywords from messages
    keywords = set()
    keyword_regex = re.compile(
        r'\b(fix|feat|refactor|docs|test|style|chore|build|ci|perf|revert|autotuner|gpu|cuda|tpu|keras|model)\b', 
        re.IGNORECASE
    )
    for msg in cleaned_messages:
        matches = keyword_regex.findall(msg)
        keywords.update(m.lower() for m in matches)
    
    # Process file paths
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
        "message_samples": cleaned_messages[:3] # --- MODIFIED: Further reduced samples
    }

def create_dynamic_batches(sorted_active_users, all_profiles, all_user_commits_data):
    """
    Packs users into batches based on "weight" (contribution count).
    """
    print(f"Creating dynamic batches (Max Weight: {MAX_BATCH_CONTRIBUTIONS}, Max Size: {MAX_BATCH_SIZE})...")
    batches = []
    
    # We iterate through a copy, so we can pop from the original
    active_user_list = list(sorted_active_users) 
    
    batch_num = 1
    while active_user_list:
        current_batch_data = {}
        current_batch_weight = 0
        
        # Always add the heaviest remaining user to start a new batch
        username = active_user_list.pop(0)
        user_profile = all_profiles[username]
        count = user_profile['activity_summary']['total_contributions']
        
        # Compress the commit data for this user
        commit_messages, file_paths = all_user_commits_data.get(username, ([], []))
        commit_summary = compress_commit_data(commit_messages, file_paths)
        
        # Build the LLM-ready payload
        current_batch_data[username] = {
            "username": username,
            "profile_summary": user_profile["profile_summary"],
            "repo_context": user_profile["repo_context"],
            "activity_summary": user_profile["activity_summary"],
            "contributions": user_profile["contributions"],
            "commit_summary": commit_summary
        }
        current_batch_weight += count
        
        # Try to add more (lighter) users to this batch
        for i in range(len(active_user_list) - 1, -1, -1):
            username_light = active_user_list[i]
            user_profile_light = all_profiles[username_light]
            count_light = user_profile_light['activity_summary']['total_contributions']
            
            # Check if adding this user exceeds limits
            if (len(current_batch_data) < MAX_BATCH_SIZE) and \
               (current_batch_weight + count_light <= MAX_BATCH_CONTRIBUTIONS):
                
                # Add user to batch
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
                
                # Remove them from the main list
                active_user_list.pop(i)
        
        # --- NEW: Added print statement for batch composition ---
        print(f"  - Batch {batch_num}: {list(current_batch_data.keys())} (Weight: {current_batch_weight})")
        batches.append(current_batch_data)
        batch_num += 1

    print(f"Created {len(batches)} batches from {len(sorted_active_users)} users.")
    return batches

# --- 9. PARALLEL LLM CALLS ---

def load_prompt(filename):
    """Loads a prompt from a JSON file."""
    try:
        with open(filename, "r") as f:
            prompt_data = json.load(f)
            # Handle new list-based format
            if "system_prompt_template" in prompt_data and isinstance(prompt_data["system_prompt_template"], list):
                prompt_data["system_prompt_template"] = "\n".join(prompt_data["system_prompt_template"])
            if "system_prompt" in prompt_data and isinstance(prompt_data["system_prompt"], list):
                prompt_data["system_prompt"] = "\n".join(prompt_data["system_prompt"])
            return prompt_data
            
    except FileNotFoundError:
        print(f"FATAL ERROR: Prompt file '{filename}' not found. Exiting.")
        exit(1)
    except json.JSONDecodeError:
        print(f"FATAL ERROR: Prompt file '{filename}' is not valid JSON. Exiting.")
        exit(1)


def invoke_llm_batch(batch_data, prompt_template_str):
    """
    Sends a BATCH of users to the LLM for skill/contribution inference.
    This is a BLOCKING function, to be run in a thread.
    """
    num_users = len(batch_data)
    usernames = list(batch_data.keys())
    # --- MODIFIED: Silenced this print statement ---
    # print(f"  - [LLM Pass 1] Sending batch of {num_users} users: {', '.join(usernames)}")
    
    try:
        system_prompt = prompt_template_str.format(num_users=num_users)
    except KeyError:
        system_prompt = prompt_template_str
    
    user_prompt = json.dumps(batch_data, indent=2)
    
    # --- MODIFIED: Handle different model formats ---
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
    # --- END MODIFIED ---

    raw_output = "" # Initialize raw_output here
    try:
        response = bedrock_client.invoke_model(
            modelId=LLM_MODEL_ID,
            body=json.dumps(body_obj),
            contentType="application/json",
            accept="application/json"
        )
        response_body = json.loads(response.get('body').read())
        
        # --- MODIFIED: Handle different model responses ---
        raw_output = ""
        if "anthropic.claude" in LLM_MODEL_ID:
            raw_output = response_body.get('content', [{}])[0].get('text', '')
        elif "openai.gpt-oss-120b" in LLM_MODEL_ID or "deepseek" in LLM_MODEL_ID:
            raw_output = response_body.get('choices', [{}])[0].get('message', {}).get('content', '')
        # --- END MODIFIED ---
        
        # --- MODIFIED: Reverted to robust find/rfind to strip reasoning tags ---
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
        # --- MODIFIED: Print the raw output on failure ---
        print(f"Raw output from LLM: {raw_output}")
        return {}


def generate_team_structure_analysis(active_profiles, system_prompt):
    """
    Sends ONLY active profiles to the LLM to infer team structure and summary.
    This is a BLOCKING function.
    """
    print(f"\nSending {len(active_profiles)} active profiles to Bedrock LLM (Pass 2)...")
    
    user_prompt = json.dumps(active_profiles, indent=2)
    
    # --- MODIFIED: Handle different model formats ---
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
    
    raw_output = "" # Initialize raw_output here
    try:
        response = bedrock_client.invoke_model(
            modelId=LLM_MODEL_ID,
            body=json.dumps(body_obj),
            contentType="application/json",
            accept="application/json"
        )
        response_body = json.loads(response.get('body').read())
        
        # --- MODIFIED: Handle different model responses ---
        raw_output = ""
        if "anthropic.claude" in LLM_MODEL_ID:
            raw_output = response_body.get('content', [{}])[0].get('text', '')
        elif "openai.gpt-oss-120b" in LLM_MODEL_ID or "deepseek" in LLM_MODEL_ID:
            raw_output = response_body.get('choices', [{}])[0].get('message', {}).get('content', '')
        # --- END MODIFIED ---
        
        # --- MODIFIED: Reverted to robust find/rfind to strip reasoning tags ---
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
        # --- MODIFIED: Print the raw output on failure ---
        print(f"Raw output from LLM: {raw_output}")
        return {} # Return empty dict on failure


# --- 10. MAIN ASYNC ORCHESTRATOR (REFACTORED) ---
async def main():
    try:
        repo_name = input("Enter the public repository name (e.g., docker/compose): ")
        pat = getpass.getpass("Enter your GitHub Personal Access Token (PAT): ")
        days_input = input("Enter scan period in days (default: 30): ")
        days_to_scan = int(days_input) if days_input.isdigit() else 30
        
        start_time = time.time()
        
        # --- Load Prompts ---
        pass_1_prompt_data = load_prompt("pass_1_prompt.json")
        pass_2_prompt_data = load_prompt("pass_2_prompt.json")
        
        # Join the new array format into a single string
        pass_1_prompt_template = "\n".join(pass_1_prompt_data["system_prompt_template"])
        pass_2_prompt = "\n".join(pass_2_prompt_data["system_prompt"])
        
        # --- NEW: Semaphore to limit LLM concurrency ---
        llm_semaphore = asyncio.Semaphore(MAX_LLM_CONCURRENCY)

        async def invoke_llm_batch_wrapper(batch, prompt_template):
            """Acquires semaphore before running the blocking LLM call."""
            async with llm_semaphore:
                return await asyncio.to_thread(invoke_llm_batch, batch, prompt_template)
        
        # --- PASS 0: Fetch & Preprocess (Python) ---
        user_activity_map = get_github_activity(repo_name, pat, days_to_scan=days_to_scan)
        
        if not user_activity_map:
            print("No activity found. Exiting.")
            return

        # --- PASS 0.1: Filter & Sort (Python) ---
        final_profiles, sorted_active_users = preprocess_and_filter(user_activity_map)
        
        if not final_profiles:
            print(f"No active contributors found with >{ACTIVITY_THRESHOLD} contributions. Exiting.")
            return
            
        # --- PASS 0.2: Fetch All Data for ALL active users (Async) ---
        auth_headers = {
            "Authorization": f"Bearer {pat}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        since_date = (datetime.now(timezone.utc) - timedelta(days=days_to_scan)).isoformat()
        
        all_user_commits_data = {} # Will store {username: (messages, file_paths)}

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
            
            # --- Process results ---
            
            # Add languages to all profiles
            for username in final_profiles:
                final_profiles[username]["repo_context"]["languages"] = repo_languages
            
            # Add profile data
            for username, profile_data in profile_results:
                if username in final_profiles:
                    final_profiles[username]["profile_summary"] = profile_data
            
            # 2. Extract SHAs and messages
            user_commit_messages = {}
            commit_detail_tasks = []
            
            for username, commit_list in commit_list_results:
                 if username in final_profiles:
                    shas = [sha for sha, msg in commit_list[:MAX_COMMITS_TO_FETCH_DETAILS] if sha]
                    messages = [msg for sha, msg in commit_list]
                    user_commit_messages[username] = messages
                    
                    # Create a task to fetch this user's commit details
                    task = fetch_commit_details_concurrently(shas, repo_name, auth_headers)
                    commit_detail_tasks.append((username, task))
            
            # 3. Fetch commit details (file paths) concurrently
            print(f"Fetching file path details for {len(commit_detail_tasks)} users...")
            detail_results = await asyncio.gather(*[task for username, task in commit_detail_tasks])
            
            # Map results back to user
            user_file_paths = {}
            for (username, task), files in zip(commit_detail_tasks, detail_results):
                user_file_paths[username] = files
            
            # Now, populate the all_user_commits_data correctly
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
        # MODIFIED: Create a clean output dict
        for_output_pass_1 = {}
        for result in batch_results:
            if result:
                for username, inferences in result.items():
                    if username in final_profiles:
                        # Get the base profile
                        base_profile = final_profiles[username]
                        
                        # Create a new, clean profile for the output
                        for_output_pass_1[username] = {
                            "username": base_profile["username"],
                            "inferred_role": base_profile["inferred_role"],
                            "profile_summary": base_profile["profile_summary"],
                            "activity_summary": base_profile["activity_summary"],
                            # Add the inferred data from the LLM
                            "technical_skills": inferences.get("technical_skills", []),
                            "contribution_types": inferences.get("contribution_types", []),
                            "confidence": inferences.get("confidence", "Low")
                        }
                        # Also update final_profiles for Pass 2
                        final_profiles[username].update(inferences)

        print("\n" + "="*50)
        print("GENERATED EXPERTISE PROFILES (PASS 1):")
        print("="*50)
        # --- MODIFIED: Silenced this print statement ---
        # print(json.dumps(for_output_pass_1, indent=2))
        
        with open("expertise_profiles.json", "w") as f:
            json.dump(for_output_pass_1, f, indent=2)
        print("\nSuccessfully saved profiles to 'expertise_profiles.json'")
        
        
        # --- PASS 2: Team Structure (Python + 1 LLM Call) ---
        print("\n" + "="*50)
        print("Performing second-level team structure analysis (Pass 2)...")
        print("="*50)
        
        # --- 2.1: Build deterministic parts in Python ---
        # We only need to send a subset of data to Pass 2
        active_profiles_for_llm = {}
        for username, profile in final_profiles.items():
            # --- MODIFIED: Check if profile has skills before adding
            if "technical_skills" in profile and profile["technical_skills"]:
                active_profiles_for_llm[username] = {
                    "inferred_role": profile["inferred_role"],
                    "technical_skills": profile["technical_skills"],
                    "contribution_types": profile["contribution_types"]
                }
        
        # --- MODIFIED: Removed Python-based hierarchy ---
        
        # --- 2.2: Call LLM for creative parts ONLY ---
        llm_team_analysis = await asyncio.to_thread(
            generate_team_structure_analysis, 
            active_profiles_for_llm,
            pass_2_prompt
        )
        
        # --- 2.3: Assemble final structure ---
        # --- MODIFIED: Get hierarchy from LLM ---
        final_team_structure = {
            "active_teams": llm_team_analysis.get("active_teams", {}),
            "inferred_hierarchy": llm_team_analysis.get("inferred_hierarchy", {})
        }

        print("\n" + "="*50)
        print("GENERATED TEAM STRUCTURE (PASS 2):")
        print("="*50)
        print(json.dumps(final_team_structure, indent=2))
        
        with open("team_structure.json", "w") as f:
            json.dump(final_team_structure, f, indent=2)
        print(f"\nSuccessfully saved team structure to 'team_structure.json'")

        end_time = time.time()
        print(f"\n--- Total execution time: {end_time - start_time:.2f} seconds ---")

    except Exception as e:
        print(f"\nAn unexpected error occurred in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # This runs the main async function
    asyncio.run(main())

