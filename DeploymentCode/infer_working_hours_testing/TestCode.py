import json
import boto3
import os
import httpx
import re
import sys
import time
import getpass  # For securely getting the PAT
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict

# --- Configuration ---
BEDROCK_REGION = "ap-south-1"  # As you specified
LLM_MODEL_ID = "deepseek.v3-v1:0"  # As you suggested
DAYS_TO_SCAN = 90
MIN_ACTIVITY_THRESHOLD = 10 # Min data points to trigger LLM
TEAM_CONFIG_FILE = "team_structure.json" # The file you provided
DEFAULT_START_TIME = "09:00"
DEFAULT_END_TIME = "17:00"

# --- AWS Clients (for Bedrock) ---
try:
    bedrock_client = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION)
except Exception as e:
    print(f"Error: Could not initialize boto3 client in region '{BEDROCK_REGION}'.")
    print("Please ensure your AWS CLI is configured and you have Bedrock access.")
    print(f"Error details: {e}")
    sys.exit(1)


def get_team_config_from_file(file_path: str) -> List[str]:
    """
    Loads example_team_structure.json and returns a unique list of user handles.
    """
    print(f"Loading team structure from {file_path}...")
    try:
        with open(file_path, 'r') as f:
            team_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found. Please make sure it's in the same directory.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not parse '{file_path}'. Is it valid JSON?")
        return []

    user_set: Set[str] = set()
    
    # Add users from hierarchy
    user_set.update(team_data.get('inferred_hierarchy', {}).get('leads', []))
    user_set.update(team_data.get('inferred_hierarchy', {}).get('engineers', []))
    
    # Add users from active_teams (in case hierarchy is incomplete)
    for team_name, members in team_data.get('active_teams', {}).items():
        user_set.update(members)
        
    all_users = sorted(list(user_set))
    print(f"Loaded {len(all_users)} unique users from config.")
    return all_users


def fetch_user_commits(
    repo: str, user_handle: str, since_date_iso: str, client: httpx.Client
) -> List[str]:
    """
    Fetches all commit timestamps (in UTC) for a user in a specific repo.
    """
    commit_timestamps = []
    page = 1
    url = f"https://api.github.com/repos/{repo}/commits"
    params = { "author": user_handle, "since": since_date_iso, "per_page": 100 }
    
    print(f"  - Fetching commits for '{user_handle}' (Page {page})...", end='', flush=True)
    
    try:
        while True:
            if page > 1:
                 print(f"  - Fetching commits for '{user_handle}' (Page {page})...", end='', flush=True)
                 
            params['page'] = page
            response = client.get(url, params=params)
            
            if response.status_code == 401:
                print("\n  - ERROR: 401 Unauthorized. Is your GITHUB_PAT valid and does it have 'repo' scope?")
                return []
            if response.status_code in [404, 422]:
                print(f" Done. (User '{user_handle}' not found or repo invalid).")
                return []

            response.raise_for_status()
            data = response.json()
            if not data:
                print(f" Done. (Found 0 commits on this page)")
                break 
                
            for commit in data:
                date_str = commit.get('commit', {}).get('author', {}).get('date')
                if date_str:
                    commit_timestamps.append(date_str)
            
            print(f" Done. (Found {len(data)}, Total: {len(commit_timestamps)})")
            
            if "next" not in response.links:
                break
            page += 1
            time.sleep(0.5) # API rate limit courtesy

    except httpx.HTTPStatusError as e:
        print(f"\n  - ERROR fetching commits for {user_handle}: {e}")
    except Exception as e:
        print(f"\n  - UNEXPECTED ERROR: {e}")
        
    return commit_timestamps

def fetch_user_activity(
    repo: str, user_handle: str, since_date_iso: str, client: httpx.Client
) -> List[str]:
    """
    Fetches PRs created and Issues closed by the user.
    """
    activity_timestamps = []
    print(f"  - Fetching PRs/Issues for '{user_handle}'...", end='', flush=True)
    
    try:
        # Search for PRs created AND Issues closed by the user in the last 90 days
        query = f"repo:{repo} author:{user_handle} is:pr created:>{since_date_iso}"
        search_url = "https://api.github.com/search/issues"
        params = {"q": query, "per_page": 100}
        
        response = client.get(search_url, params=params)
        response.raise_for_status()
        data = response.json()
        for item in data.get('items', []):
            activity_timestamps.append(item['created_at'])

        # Add issues closed
        query = f"repo:{repo} closed-by:{user_handle} is:issue closed:>{since_date_iso}"
        params = {"q": query, "per_page": 100}
        response = client.get(search_url, params=params)
        response.raise_for_status()
        data = response.json()
        for item in data.get('items', []):
            activity_timestamps.append(item['closed_at'])

        print(f" Done. (Found {len(activity_timestamps)} PR/Issue events)")
        
    except httpx.HTTPStatusError as e:
        print(f"\n  - ERROR fetching activity for {user_handle}: {e}")
    except Exception as e:
        print(f"\n  - UNEXPECTED ERROR: {e}")
        
    return activity_timestamps


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
    
    body_obj = { "messages": [ {"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt} ], "max_tokens": 200, "temperature": 0.0, "top_p": 0.9 }
    
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


def main_local_test():
    """Main function to run the local test."""
    
    print("--- Starting Local Availability Calculator ---")
    
    # 1. Get user inputs (as you requested)
    pat = getpass.getpass("Enter your GitHub PAT (press Enter if repo is public): ")
    repo_name = input(f"Enter the repo name (e.g., 'tensorflow/tensorflow'): ")

    if not repo_name:
        print("Error: Repo name cannot be empty.")
        sys.exit(1)
        
    # 2. Load users from local JSON
    all_users = get_team_config_from_file(TEAM_CONFIG_FILE)
    if not all_users:
        print(f"No users found in {TEAM_CONFIG_FILE}. Exiting.")
        sys.exit(1)
        
    since_date = datetime.now(timezone.utc) - timedelta(days=DAYS_TO_SCAN)
    since_date_iso = since_date.isoformat()
    
    headers = { "Accept": "application/vnd.github.v3+json", "X-GitHub-Api-Version": "2022-11-28" }
    if pat:
        headers["Authorization"] = f"Bearer {pat}"
    
    final_availability_map = {}

    with httpx.Client(headers=headers, follow_redirects=True, timeout=30.0) as client:
        # 5. Process each user
        for i, user_handle in enumerate(all_users):
            print(f"\n--- Processing user {i+1}/{len(all_users)}: {user_handle} ---")
            
            all_timestamps = []
            
            # A. Fetch commits
            all_timestamps.extend(fetch_user_commits(repo_name, user_handle, since_date_iso, client))
            
            # B. Fetch PRs and Issues (Your new logic)
            all_timestamps.extend(fetch_user_activity(repo_name, user_handle, since_date_iso, client))
            
            if not all_timestamps:
                print(f"  - No activity found for {user_handle} in the last {DAYS_TO_SCAN} days.")
                continue
                
            # C. Analyze with LLM (or use default)
            inferred_window = analyze_activity_with_llm(
                user_handle,
                all_timestamps
            )
            
            if inferred_window:
                print(f"  - SUCCESS: {inferred_window['status']} window for {user_handle}: {inferred_window['inferred_start_time_utc']} - {inferred_window['inferred_end_time_utc']} UTC")
                final_availability_map[user_handle] = inferred_window
            else:
                # This case should ideally not be hit, but as a fallback
                print(f"  - FAILED: Could not infer window for {user_handle}.")
                
    print("\n\n" + "="*50)
    print("--- FINAL INFERRED AVAILABILITY MAP (UTC) ---")
    print(json.dumps(final_availability_map, indent=2))
    print("="*50)


if __name__ == "__main__":
    main_local_test()