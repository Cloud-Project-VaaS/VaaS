import json
import boto3
import os
import requests
import jwt  # PyJWT
import time
import httpx
import re
import yaml  # PyYAML
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal

# --- AWS Clients (Global) ---
bedrock_client = boto3.client('bedrock-runtime')
dynamodb = boto3.resource('dynamodb')
secrets_client = boto3.client('secretsmanager')
eventbridge_client = boto3.client('events') # We don't send an event *from* this, but good to have

# --- Configuration (from Environment Variables) ---
ISSUES_TABLE_NAME = os.environ.get("ISSUES_TABLE_NAME")
EXPERTISE_TABLE_NAME = os.environ.get("EXPERTISE_TABLE_NAME")
USER_AVAILABILITY_TABLE_NAME = os.environ.get("USER_AVAILABILITY_TABLE_NAME") 
SECRET_ARN = os.environ.get('SECRET_ARN')

LLM_MODEL_ID = "deepseek.v3-v1:0"  # Sticking with DeepSeek
LLM_MAX_RETRIES = 2
LLM_RETRY_BACKOFF_FACTOR = 0.5

# --- Config Files (Pasted from your .yaml files) ---
# This makes the Lambda self-contained.
TEAM_CONFIG_DATA = """
version: 1
teams:
  - name: "backend"
    timezone: "Asia/Kolkata"
    working_hours_local: "10:00-18:00" # Team default
    weekends: ["SAT", "SUN"]
    holidays:
      - "2025-01-26"
      - "2025-08-15"
      - "2025-10-02"
    weekend_coverage:
      enabled: true
      working_hours_local: "10:00-16:00"
    members:
      - handle: "@anil"
        role: "lead"
        jira_account_id: "<anil_account_id>" # Placeholder
        availability:
          timezone: "Asia/Kolkata"
          working_hours_local: "09:00-17:00"
          pto: []
      - handle: "@bhavna"
        role: "reviewer"
        jira_account_id: "<bhavna_account_id>" # Placeholder
        availability:
          timezone: "Asia/Kolkata"
          working_hours_local: "10:30-18:30"
          pto: ["2025-10-14..2025-10-20"]

  - name: "frontend"
    timezone: "America/Los_Angeles"
    working_hours_local: "10:00-18:00"
    weekends: ["SAT", "SUN"]
    holidays:
      - "2025-07-04"
      - "2025-11-27"
    weekend_coverage:
      enabled: true
      working_hours_local: "09:00-13:00"
    members:
      - handle: "@dave"
        role: "lead"
        jira_account_id: "<dave_account_id>" # Placeholder
        availability:
          timezone: "America/Los_Angeles"
          working_hours_local: "08:00-16:00"
          pto: []
      - handle: "@erin"
        role: "reviewer"
        jira_account_id: "<erin_account_id>" # Placeholder
        availability:
          timezone: "America/Los_Angeles"
          working_hours_local: "09:30-17:30"
          pto: []

  - name: "ml"
    timezone: "Europe/Berlin"
    working_hours_local: "09:00-17:00"
    weekends: ["SAT", "SUN"]
    holidays:
      - "2025-10-03"
      - "2025-12-25"
    weekend_coverage:
      enabled: false # This team does not work weekends
    members:
      - handle: "@felix"
        role: "lead"
        jira_account_id: "<felix_account_id>" # Placeholder
        availability:
          timezone: "Europe/Berlin"
          working_hours_local: "08:00-16:00"
          pto: []
      - handle: "@giulia"
        role: "reviewer"
        jira_account_id: "<giulia_account_id>" # Placeholder
        availability:
          timezone: "Europe/Berlin"
          working_hours_local: "09:30-17:30"
          pto: []

global_escalation_team:
  name: "escalation-duty"
  timezone: "UTC"
  working_hours_local: "00:00-23:59" # 24/7
  weekends: []
  holidays: []
  members:
    - handle: "@zoe"
      role: "lead"
      jira_account_id: "<zoe_account_id>" # Placeholder
      availability:
        timezone: "UTC"
        working_hours_local: "00:00-23:59"
        pto: []
    - handle: "@yuki"
      role: "reviewer" 
      jira_account_id: "<yuki_account_id>" # Placeholder
      availability:
        timezone: "UTC"
        working_hours_local: "00:00-23:59"
        pto: []
"""

SLA_CONFIG_DATA = """
version: 1

policies:
  business_time:
    mode: "team"
    weekends: [] 
    holidays_global: []
  pause_labels: ["on-hold", "waiting-for-customer", "blocked"]
  max_reassignments_per_item: 3
  enforce_mode: "on" 
  comment_on_actions: true

# ===== ISSUE SLAs =====
issues:
  - id: "issue-bug-high"
    name: "Bug — High"
    match: 
      all: ["type:bug", "priority:high"]
    targets:
      time_to_first_response: "8 business_hours"
    escalation:
      steps:
        - at_percent: 70
          action: "notify"
          reason: "High priority bug SLA at 70%. Breach imminent."

  - id: "issue-bug-medium"
    name: "Bug — Medium"
    match:
      all: ["type:bug", "priority:medium"]
    targets:
      time_to_first_response: "1 business_day"
    escalation:
      steps:
        - at_percent: 100
          action: "reassign"
          target: "role:lead" 
          reason: "Medium bug breached. Escalating to team lead."

  - id: "issue-bug-low"
    name: "Bug — Low"
    match:
      all: ["type:bug", "priority:low"]
    targets:
      time_to_first_response: "3 business_days"
    escalation:
      steps:
        - at_percent: 100
          action: "notify"
          reason: "Low bug breached. Sending reminder to reviewer."

  - id: "issue-enhancement-high"
    name: "Enhancement — High"
    match:
      all: ["type:enhancement", "priority:high"]
    targets:
      time_to_first_response: "1 business_day"
    escalation:
      steps:
        - at_percent: 100
          action: "reassign"
          reason: "High priority enhancement breached. Escalating to team lead."
          
  - id: "issue-enhancement-medium"
    name: "Enhancement — Medium"
    match:
      all: ["type:enhancement", "priority:medium"]
    targets:
      time_to_first_response: "3 business_days"
    escalation:
      steps:
        - at_percent: 100
          action: "reassign"
          reason: "Medium enhancement breached. Assigning to team lead."

  - id: "issue-enhancement-low"
    name: "Enhancement — Low"
    match:
      all: ["type:enhancement", "priority:low"]
    targets:
      time_to_first_response: "5 business_days"
    escalation:
      steps:
        - at_percent: 100
          action: "notify"
          reason: "Low enhancement breached. Sending reminder to reviewer."

  - id: "issue-question-high"
    name: "Question — High"
    match:
      all: ["type:question", "priority:high"]
    targets:
      time_to_first_response: "1 business_day"
    escalation:
      steps:
        - at_percent: 100
          action: "reassign"
          reason: "High priority question breached. Escalating to team lead."

  - id: "issue-question-medium"
    name: "Question — Medium"
    match:
      all: ["type:question", "priority:medium"]
    targets:
      time_to_first_response: "3 business_days"
    escalation:
      steps:
        - at_percent: 100
          action: "reassign"
          reason: "Medium question breached. Assigning to team lead."

  - id: "issue-question-low"
    name: "Question — Low"
    match:
      all: ["type:question", "priority:low"]
    targets:
      time_to_first_response: "7 business_days"
    escalation:
      steps:
        - at_percent: 100
          action: "notify"
          reason: "Low question breached. Sending reminder to reviewer."
"""
# --- End Config Files ---


# --- Global Variables (Loaded once at cold start) ---
try:
    # We load the config from the strings above
    TEAM_CONFIG = yaml.safe_load(TEAM_CONFIG_DATA)
    SLA_CONFIG = yaml.safe_load(SLA_CONFIG_DATA)
    
    # Build a fast-lookup map for teams
    # handle (no @) -> {team_name, role, timezone, working_hours_local}
    TEAM_LOOKUP_MAP: Dict[str, Dict] = {}
    for team in TEAM_CONFIG.get('teams', []):
        for member in team.get('members', []):
            handle = member['handle'].lstrip('@') # Clean handle
            TEAM_LOOKUP_MAP[handle] = {
                "team_name": team['name'],
                "role": member['role'],
                "timezone": member.get('availability', {}).get('timezone', 'UTC'),
                "working_hours_local": member.get('availability', {}).get('working_hours_local', '09:00-17:00')
            }
    
except yaml.YAMLError as e:
    print(f"FATAL: Failed to parse YAML config strings: {e}")
    TEAM_CONFIG = {}
    SLA_CONFIG = {}
    TEAM_LOOKUP_MAP = {}

# Global cache for secrets
APP_ID = None
PRIVATE_KEY = None

# --- GitHub App Authentication ---
def load_secrets():
    """Loads App ID and Private Key from Secrets Manager."""
    global APP_ID, PRIVATE_KEY
    if APP_ID and PRIVATE_KEY:
        return

    if not SECRET_ARN:
        raise ValueError("Error: SECRET_ARN environment variable is not set.")
    
    try:
        print("Loading secrets from Secrets Manager...")
        secret_response = secrets_client.get_secret_value(SecretId=SECRET_ARN)
        secrets = json.loads(secret_response['SecretString'])
        APP_ID = secrets.get('APP_ID')
        PRIVATE_KEY = secrets.get('PRIVATE_KEY')
        if not APP_ID or not PRIVATE_KEY:
            raise KeyError("APP_ID or PRIVATE_KEY not found in secret.")
        print("Secrets loaded successfully.")
    except Exception as e:
        print(f"Error loading secrets: {e}")
        raise

def create_app_jwt() -> str:
    """Creates a JSON Web Token (JWT) for GitHub App authentication."""
    if not APP_ID or not PRIVATE_KEY:
        load_secrets()
        
    try:
        payload = {
            'iat': int(time.time()),
            'exp': int(time.time()) + (10 * 60),
            'iss': APP_ID
        }
        return jwt.encode(payload, PRIVATE_KEY, algorithm='RS256')
    except Exception as e:
        print(f"Error creating JWT: {e}")
        raise

def get_installation_access_token(installation_id: int) -> str:
    """Gets a temporary access token for a specific installation."""
    app_jwt = create_app_jwt()
    try:
        headers = {
            "Authorization": f"Bearer {app_jwt}",
            "Accept": "application/vnd.github.v3+json",
        }
        url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
        
        # Use httpx for consistency
        with httpx.Client() as client:
            response = client.post(url, headers=headers)
            response.raise_for_status()
        
        token_data = response.json()
        if 'token' not in token_data:
            raise ValueError("Error: 'token' not found in response.")
        
        print(f"Successfully generated new installation access token for {installation_id}.")
        return token_data['token']
    except httpx.HTTPStatusError as http_err:
        print(f"HTTP error getting installation token: {http_err} - {http_err.response.text}")
        raise
    except Exception as e:
        print(f"Error getting installation token: {e}")
        raise

# --- Core Logic: Data Fetching ---

def get_team_and_availability_context() -> str:
    """
    Parses team config and merges with INFERRED availability from UserAvailability table.
    """
    context = "Available Teams and Members (with Inferred UTC Availability):\n"
    
    if not USER_AVAILABILITY_TABLE_NAME:
        print("Warning: USER_AVAILABILITY_TABLE_NAME not set. Using default config hours.")
        # Fallback to just the config file if the table isn't set
        for handle, member in TEAM_LOOKUP_MAP.items():
            context += (
                f"  - Member: @{handle} "
                f"(Team: {member['team_name']}, "
                f"Role: {member['role']}, "
                f"Availability: {member['working_hours_local']} {member['timezone']}) (DEFAULT)\n"
            )
        return context

    try:
        table = dynamodb.Table(USER_AVAILABILITY_TABLE_NAME)
        # Use batch_get_item for efficiency
        keys_to_get = [{'user_handle': handle} for handle in TEAM_LOOKUP_MAP.keys()]
        
        if not keys_to_get:
            return "No team members configured.\n"
            
        response = dynamodb.batch_get_item(RequestItems={USER_AVAILABILITY_TABLE_NAME: {'Keys': keys_to_get}})
        availability_data = {item['user_handle']: item for item in response.get('Responses', {}).get(USER_AVAILABILITY_TABLE_NAME, [])}
        
        # Merge config data with inferred availability
        for handle, member_config in TEAM_LOOKUP_MAP.items():
            avail = availability_data.get(handle)
            if avail and 'inferred_start_time_utc' in avail:
                # Use inferred data
                start = avail.get('inferred_start_time_utc', 'N/A')
                end = avail.get('inferred_end_time_utc', 'N/A')
                avail_str = f"{start}-{end} UTC (Inferred)"
            else:
                # Use default config data
                avail_str = f"{member_config['working_hours_local']} {member_config['timezone']} (Default)"

            context += (
                f"  - Member: @{handle} "
                f"(Team: {member_config['team_name']}, "
                f"Role: {member_config['role']}, "
                f"Availability: {avail_str})\n"
            )
        return context
        
    except Exception as e:
        print(f"Error fetching from UserAvailability table: {e}. Returning default hours.")
        # Fallback to default if DDB call fails
        default_context = "Available Teams and Members (using DEFAULT hours due to error):\n"
        for handle, member in TEAM_LOOKUP_MAP.items():
            default_context += (
                f"  - Member: @{handle} "
                f"(Team: {member['team_name']}, "
                f"Role: {member['role']}, "
                f"Availability: {member['working_hours_local']} {member['timezone']}) (DEFAULT)\n"
            )
        return default_context


def get_expertise_context(repo_name: str) -> str:
    """
    Fetches the expert profiles from the RepoExpertise DynamoDB table.
    """
    if not EXPERTISE_TABLE_NAME:
        print("Warning: EXPERTISE_TABLE_NAME not set. Cannot fetch expertise.")
        return "Expertise data is not available.\n"

    try:
        table = dynamodb.Table(EXPERTISE_TABLE_NAME)
        response = table.get_item(Key={'repo_name': repo_name})
        
        if 'Item' not in response:
            print(f"No expertise data found for repo: {repo_name}")
            return "Expertise data is not available for this repo.\n"
            
        item = response['Item']
        profiles = item.get('expertise_profiles', [])
        
        if not profiles:
            return "No expertise profiles found for this repo.\n"

        # Format for LLM prompt
        context = "Ranked Expertise Profiles (from repo analysis):\n"
        for profile in profiles:
            login = profile.get('login', 'unknown')
            summary = profile.get('summary', 'no summary')
            context += f"- {login}: {summary}\n"
            
        return context

    except Exception as e:
        print(f"Error fetching from RepoExpertise table: {e}")
        return f"Error fetching expertise data: {e}\n"

def get_workload_context(candidate_handles: List[str]) -> str:
    """
    Fetches the current open issue count for each candidate from the IssuesTrackingTable.
    This relies on the GSI 'by_assignee_and_status'
    """
    if not ISSUES_TABLE_NAME:
        print("Warning: ISSUES_TABLE_NAME not set. Cannot fetch workload.")
        return "Workload data is not available.\n"

    try:
        table = dynamodb.Table(ISSUES_TABLE_NAME)
        context = "Current Candidate Workload (open issues assigned):\n"
        
        for handle_at in candidate_handles:
            # handle_at is like "@anil"
            
            # This is a GSI query.
            response = table.query(
                IndexName='by_assignee_and_status',
                KeyConditionExpression="current_assignee = :h AND #s = :s",
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues={":h": handle_at, ":s": "open"}
            )
            count = response.get('Count', 0)
            context += f"- {handle_at}: {count} open issues\n"
            
        print(f"Workload context generated.")
        return context
    except Exception as e:
        print(f"CRITICAL ERROR fetching workload from IssuesTrackingTable (Is GSI 'by_assignee_and_status' created?): {e}")
        return f"Error fetching workload data. GSI may be missing or building.\n"

def get_sla_due_date(issue_type: str, priority: str) -> Tuple[str, str]:
    """
    Finds the matching SLA policy and returns the target time and escalation action.
    """
    try:
        policies = SLA_CONFIG.get('issues', [])
        for policy in policies:
            match = policy.get('match', {})
            all_match = match.get('all', [])
            
            # Normalize inputs
            type_match = f"type:{issue_type.lower()}" in all_match
            priority_match = f"priority:{priority.lower()}" in all_match
            
            if type_match and priority_match:
                target_time = policy.get('targets', {}).get('time_to_first_response', '5 business_days')
                escalation_action = "notify" # Default
                
                for step in policy.get('escalation', {}).get('steps', []):
                    if step.get('action') == 'reassign':
                        escalation_action = f"reassign to {step.get('target', 'lead')}"
                        break
                
                return target_time, escalation_action

    except Exception as e:
        print(f"Error parsing SLA config: {e}")
        
    return "5 business_days", "notify" # Default fallback


def run_assignment_llm(issue: Dict, context: str) -> Optional[Dict]:
    """
    Calls the Bedrock LLM to get the final assignment decision.
    """
    print(f"Running assignment LLM for issue {issue['issue_id']}...")
    
    system_prompt = """You are a senior engineering manager. Your job is to assign a new GitHub issue to the best possible person.
You will be given context about the issue, the team, member expertise, and current workload.
You must return ONLY a single, valid JSON object with the following keys:
- "assignee_handle": The GitHub handle of the best person (e.g., "@anil"). MUST be one of the handles from the 'Available Teams' list.
- "labels_to_add": A list of strings for GitHub labels (e.g., ["Bug", "P1"]).
- "reasoning": A brief, professional justification for your choice.
- "confidence": A float from 0.0 to 1.0.

Use this reasoning process:
1.  **Expertise:** Use the 'Ranked Expertise Profiles' to find the person with the most relevant skills. This is the most important factor.
2.  **Workload:** Use the 'Current Candidate Workload'. If the best expert is overloaded (e.g., > 5 open issues), strongly consider the #2 expert.
3.  **Availability/Role:** Use the 'Available Teams' info. Favor 'reviewers' for most tasks and 'leads' for high-priority or complex tasks.
4.  **Labels:** The labels to add are *always* the issue `type` and `priority` (e.g., "Bug", "High").
5.  **Fallback:** If no expert is a good fit or all are overloaded, assign to the 'lead' of the most relevant team.
"""
    
    # Use the enriched title/body if they exist, otherwise fall back to original
    title = issue.get('enriched_title', issue.get('title', 'No Title'))
    body = issue.get('enriched_body', issue.get('body', 'No Body'))
    
    user_prompt = f"""--- CONTEXT ---
{context}

--- NEW ISSUE TO ASSIGN ---
Repo: {issue['repo_name']}
Issue ID: {issue['issue_id']}
Title: {title}
Body: {body[:2000]}
Type: {issue.get('issue_type', 'unknown')}
Priority: {issue.get('priority', 'unknown')}

--- YOUR DECISION ---
Return ONLY the JSON object.
"""
    
    body_obj = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.1,
        "top_p": 0.9
    }
    
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            response = bedrock_client.invoke_model(
                modelId=LLM_MODEL_ID,
                body=json.dumps(body_obj),
                contentType="application/json",
                accept="application/json"
            )
            response_body = json.loads(response.get('body').read())
            result_text = response_body.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
            
            # Find and parse the JSON block
            match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if match:
                decision = json.loads(match.group(0))
                # Validate key fields
                handle = decision.get('assignee_handle', '').lstrip('@')
                if handle and handle in TEAM_LOOKUP_MAP and 'labels_to_add' in decision:
                    decision['assignee_handle'] = f"@{handle}" # Standardize with @
                    return decision
                else:
                    print(f"Warning: LLM returned invalid handle '@{handle}' (not in TEAM_LOOKUP_MAP) or missing keys. Retrying.")
            else:
                print(f"Warning: LLM returned invalid JSON: {result_text}. Retrying.")
            
        except Exception as e:
            print(f"Error calling Bedrock (DeepSeek) on attempt {attempt+1}: {e}")
            if attempt < LLM_MAX_RETRIES:
                time.sleep(LLM_RETRY_BACKOFF_FACTOR)
            
    print(f"Failed to get valid assignment from LLM after {LLM_MAX_RETRIES+1} attempts.")
    return None

def update_github_and_dynamodb(
    issue: Dict,
    decision: Dict,
    install_token: str,
    table: Any
):
    """
    The final step. Updates both GitHub and DynamoDB with the decision.
    """
    repo_name = issue['repo_name']
    issue_id = issue['issue_id']
    assignee_handle_at = decision['assignee_handle'] # e.g., "@anil"
    assignee_handle_clean = assignee_handle_at.lstrip('@') # e.g., "anil"
    labels_to_add = decision.get('labels_to_add', [])
    
    print(f"Attempting to update GitHub for {repo_name}#{issue_id}...")
    
    # 1. Update GitHub API
    try:
        headers = {
            "Authorization": f"Bearer {install_token}",
            "Accept": "application/vnd.github.v3+json",
        }
        api_url = f"https://api.github.com/repos/{repo_name}/issues/{issue_id}"
        
        payload = {
            "assignees": [assignee_handle_clean],
            "labels": labels_to_add
        }
        
        # Use httpx for this call
        with httpx.Client() as client:
            response = client.patch(api_url, headers=headers, json=payload)
            response.raise_for_status()
        
        print(f"Successfully updated GitHub issue: assigned to {assignee_handle_at}, added labels {labels_to_add}")

    except Exception as e:
        print(f"WARNING: Failed to update GitHub issue {repo_name}#{issue_id}: {e}. Check app permissions.")
        # We still continue, to at least save our decision in DynamoDB

    # 2. Update DynamoDB
    print(f"Updating DynamoDB for {repo_name}#{issue_id}...")
    try:
        # Get SLA info
        sla_target, sla_escalation = get_sla_due_date(
            issue.get('issue_type', 'unknown'),
            issue.get('priority', 'unknown')
        )
        
        table.update_item(
            Key={
                'repo_name': repo_name,
                'issue_id': issue_id
            },
            UpdateExpression=(
                "SET current_assignee = :a, assignment_reason = :r, "
                "assignment_confidence = :c, assignment_time = :at, "
                "github_labels = :l, sla_target = :st, "
                "sla_escalation_policy = :se, #s = :s, "
                "last_updated_pipeline = :lu"
            ),
            ExpressionAttributeNames={"#s": "status"}, # 'status' is a reserved word
            ExpressionAttributeValues={
                ':a': assignee_handle_at, # Store with '@'
                ':r': decision.get('reasoning', 'N/A'),
                ':c': Decimal(str(decision.get('confidence', 0.0))), # Use Decimal
                ':at': datetime.now(timezone.utc).isoformat(),
                ':l': labels_to_add,
                ':st': sla_target,
                ':se': sla_escalation,
                ':s': 'open', # Explicitly set status to open
                ':lu': datetime.now(timezone.utc).isoformat()
            }
        )
        print("Successfully updated DynamoDB.")
        
    except Exception as e:
        print(f"ERROR: Failed to update DynamoDB for {repo_name}#{issue_id}: {e}")
        # Log the error but don't fail the whole batch

# --- Main Handler ---

def lambda_handler(event, context):
    """
    Main Lambda handler. Triggered by EventBridge with a BATCH of *prioritized* issues.
    """
    print("Classify Assignee event received:")
    print(json.dumps(event))
    
    # --- 1. Validate Environment ---
    if not all([ISSUES_TABLE_NAME, EXPERTISE_TABLE_NAME, USER_AVAILABILITY_TABLE_NAME, SECRET_ARN, TEAM_CONFIG, SLA_CONFIG]):
        print("FATAL: Missing one or more environment variables or failed to parse YAML config.")
        return {'statusCode': 500, 'body': 'Internal configuration error'}
        
    # --- 2. Parse Event ---
    try:
        if event.get("source") != "github.issues" or event.get("detail-type") != "issue.batch.prioritized":
            print(f"Ignoring event from unknown source: {event.get('source')}")
            return
            
        repo_name = event['detail'].get('repo_name')
        issues_list = event['detail'].get('issues', [])
        
        if not repo_name or not issues_list:
            print("Event detail is missing repo_name or issues list. Nothing to do.")
            return

    except KeyError as e:
        print(f"Error parsing event: {e}")
        return
        
    print(f"Starting assignment for {len(issues_list)} issues from repo: {repo_name}...")
    
    # --- 3. Prepare Context (We do this ONCE for the whole batch) ---
    try:
        print(f"Fetching context for repo: {repo_name}...")
        
        # We need an installation_id to authenticate. We fixed this in fetch_and_classify_issues.
        # All issues in this batch *must* have the same installation_id.
        first_issue = issues_list[0]
        installation_id = first_issue.get('installation_id')
        if not installation_id:
            # We must get the installation_id from the repo name, as it was lost.
            print(f"Installation_id missing from payload. Fetching from 'github-installations' table...")
            install_table = dynamodb.Table(os.environ.get("INSTALLATIONS_TABLE_NAME", "github-installations"))
            # This is an inefficient query, but it's a critical fallback.
            # A better way is to ensure fetch_and_classify_issues adds it.
            response = install_table.query(
                IndexName='repo_name-index', # Assumes a GSI. If not, this fails.
                KeyConditionExpression="repo_name = :r",
                ExpressionAttributeValues={":r": repo_name}
            )
            if not response.get('Items'):
                 print(f"FATAL: Could not find installation_id for repo {repo_name}.")
                 return
            installation_id = int(response['Items'][0]['installation_id'])
            print(f"Found installation_id: {installation_id}")

            
        # 1. Get GitHub Auth Token (one token for the whole batch)
        install_token = get_installation_access_token(installation_id)
        
        # 2. Get Team/Availability Context
        team_context = get_team_and_availability_context()
        
        # 3. Get Expertise Context
        expertise_context = get_expertise_context(repo_name)
        
        # 4. Get Workload Context (for all *potential* assignees)
        all_handles = [f"@{h}" for h in TEAM_LOOKUP_MAP.keys()]
        workload_context = get_workload_context(all_handles)
        
        # 5. Combine all context for the LLM
        final_context = f"{team_context}\n{expertise_context}\n{workload_context}"
        print("Successfully generated all context.")
        
    except Exception as e:
        print(f"FATAL: Failed to prepare context for {repo_name}. Stopping. Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 4. Process Each Issue in the Batch ---
    table = dynamodb.Table(ISSUES_TABLE_NAME)
    
    for issue in issues_list:
        try:
            # 1. Get LLM Assignment
            decision = run_assignment_llm(issue, final_context)
            
            if not decision:
                # TODO: Implement a fallback (e.g., assign to team lead)
                print(f"Warning: LLM failed to provide assignment for {issue['issue_id']}. Skipping.")
                continue

            # 2. Update GitHub and DynamoDB
            update_github_and_dynamodb(issue, decision, install_token, table)

        except Exception as e:
            print(f"ERROR: Failed to process issue {issue.get('issue_id')}: {e}")
            continue # Skip this issue and continue
            
    print(f"Successfully processed batch for {repo_name}.")
    return {'statusCode': 200, 'body': 'Batch assignment complete.'}