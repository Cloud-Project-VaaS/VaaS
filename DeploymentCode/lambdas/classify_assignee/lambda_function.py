import json
import boto3
import os
import requests
import jwt  # PyJWT
import time
import httpx
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
from botocore.config import Config

# --- AWS Clients Configuration ---
CLIENT_CONFIG = Config(
    read_timeout=90,
    connect_timeout=10
)

# --- AWS Clients (Global) ---
# [FIX: Removed region_name="us-east-1" to use native region]
bedrock_client = boto3.client('bedrock-runtime', config=CLIENT_CONFIG)
dynamodb = boto3.resource('dynamodb')
secrets_client = boto3.client('secretsmanager')
eventbridge_client = boto3.client('events')

# --- Configuration (from Environment Variables) ---
ISSUES_TABLE_NAME = os.environ.get("ISSUES_TABLE_NAME")
EXPERTISE_TABLE_NAME = os.environ.get("EXPERTISE_TABLE_NAME")
USER_AVAILABILITY_TABLE_NAME = os.environ.get("USER_AVAILABILITY_TABLE_NAME") 
SECRET_ARN = os.environ.get('SECRET_ARN')
INSTALLATIONS_TABLE_NAME = os.environ.get("INSTALLATIONS_TABLE_NAME", "github-installations")

# [FIX: Using your specified DeepSeek model ID]
LLM_MODEL_ID = "deepseek.v3-v1:0"
LLM_MAX_RETRIES = 2
LLM_RETRY_BACKOFF_FACTOR = 5

# --- Global Variables (Loaded once at cold start) ---
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

def get_availability_context(candidate_handles: List[str]) -> str:
    """
    Parses INFERRED availability from UserAvailability table for a specific list of candidates.
    """
    context = "Available Team Members (with Inferred UTC Availability):\n"
    
    if not USER_AVAILABILITY_TABLE_NAME:
        print("Warning: USER_AVAILABILITY_TABLE_NAME not set. Cannot fetch availability.")
        return "Availability data is not available.\n"
    
    if not candidate_handles:
        return "No candidates found to check availability.\n"

    try:
        table = dynamodb.Table(USER_AVAILABILITY_TABLE_NAME)
        handles_no_at = [h.lstrip('@') for h in candidate_handles]
        keys_to_get = [{'user_handle': handle} for handle in handles_no_at]
        
        response = dynamodb.batch_get_item(RequestItems={USER_AVAILABILITY_TABLE_NAME: {'Keys': keys_to_get}})
        availability_data = {item['user_handle']: item for item in response.get('Responses', {}).get(USER_AVAILABILITY_TABLE_NAME, [])}

        for handle in handles_no_at:
            avail = availability_data.get(handle)
            if avail and 'inferred_start_time_utc' in avail:
                start = avail.get('inferred_start_time_utc', 'N/A')
                end = avail.get('inferred_end_time_utc', 'N/A')
                avail_str = f"{start}-{end} UTC (Inferred)"
            else:
                avail_str = f"09:00-17:00 UTC (Default)"

            context += (
                f"  - Member: @{handle} "
                f"(Availability: {avail_str})\n"
            )
        return context
        
    except Exception as e:
        print(f"Error fetching from UserAvailability table: {e}. Returning default hours.")
        default_context = "Available Team Members (using DEFAULT hours due to error):\n"
        for handle in candidate_handles:
            default_context += (
                f"  - Member: {handle} "
                f"(Availability: 09:00-17:00 UTC) (DEFAULT)\n"
            )
        return default_context


def get_expertise_context(repo_name: str) -> Tuple[str, List[str]]:
    """
    Fetches the expert profiles from the RepoExpertise DynamoDB table.
    Returns: (Context String for LLM, List of candidate handles with '@')
    """
    if not EXPERTISE_TABLE_NAME:
        print("Warning: EXPERTISE_TABLE_NAME not set. Cannot fetch expertise.")
        return "Expertise data is not available.\n", []

    candidate_handles: List[str] = []
    try:
        table = dynamodb.Table(EXPERTISE_TABLE_NAME)
        response = table.get_item(Key={'repo_name': repo_name})
        
        if 'Item' not in response:
            print(f"No expertise data found for repo: {repo_name}")
            return "Expertise data is not available for this repo.\n", []
            
        item = response['Item']
        profiles = item.get('expertise_profiles', {})
        
        if not profiles:
            return "No expertise profiles found for this repo.\n", []

        # Format for LLM prompt
        context = "Ranked Expertise Profiles (from repo analysis):\n"
        
        for login, profile in profiles.items():
            handle_at = f"@{login}"
            
            # Create a summary of skills and contribution types for the context
            skills = ", ".join(profile.get('technical_skills', ['N/A']))
            contribs = ", ".join(profile.get('contribution_types', ['N/A']))
            summary = f"Role: {profile.get('inferred_role', 'N/A')}. Skills: {skills}. Focus: {contribs}."
            
            context += f"- {handle_at}: {summary}\n"
            candidate_handles.append(handle_at)
            
        return context, candidate_handles

    except Exception as e:
        print(f"Error fetching from RepoExpertise table: {e}")
        import traceback
        traceback.print_exc() # Print full error
        return f"Error fetching expertise data: {e}\n", []

def get_workload_context(candidate_handles: List[str]) -> str:
    """
    Fetches the current open issue count for each candidate from the IssuesTrackingTable.
    """
    if not ISSUES_TABLE_NAME:
        print("Warning: ISSUES_TABLE_NAME not set. Cannot fetch workload.")
        return "Workload data is not available.\n"
        
    if not candidate_handles:
        return "No candidates found to check workload.\n"

    try:
        table = dynamodb.Table(ISSUES_TABLE_NAME)
        context = "Current Candidate Workload (open issues assigned):\n"
        
        for handle_at in candidate_handles:
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
        print(f"CRITICAL ERROR fetching workload from IssuesTrackingTable: {e}")
        return f"Error fetching workload data. GSI may be missing or building.\n"

def run_assignment_llm(issue: Dict, context: str, candidate_handles: List[str]) -> Optional[Dict]:
    """
    Calls the Bedrock LLM to get the final assignment decision.
    """
    print(f"Running assignment LLM for issue {issue['issue_id']}...")
    
    candidate_list_str = ", ".join([f"'{h}'" for h in candidate_handles])
    
    # [CHANGE 1 START] - Updated System Prompt to mention Component
    system_prompt = f"""You are a senior engineering manager. Your job is to assign a new GitHub issue to the best possible person from a specific list of candidates.
You will be given context about the issue (including its Component), the candidates' expertise, their availability, and current workload.
You must return ONLY a single, valid JSON object with the following keys:
- "assignee_handle": The GitHub handle of the best person (e.g., "@anil"). MUST be one of the handles from this *exact* list of candidates: [{candidate_list_str}]
- "labels_to_add": A list of strings for GitHub labels. This MUST include the issue Type, Priority, and Component (e.g., "Bug", "High", "Frontend") as SEPARATE strings.
- "reasoning": A brief, professional justification for your choice.
- "confidence": A float from 0.0 to 1.0.

Use this reasoning process:
1.  **Expertise:** Match the candidate's skills to the issue's **Component** (e.g., Frontend/Backend). This is the most important factor.
2.  **Workload:** Use the 'Current Candidate Workload'. If the best expert is overloaded (e.g., > 5 open issues), strongly consider the #2 expert.
3.  **Availability:** Use the 'Available Team Members' info to see who is online. Prefer users who are within their inferred UTC window.
4.  **Labels:** always return the Type, Priority, and Component as separate labels in the 'labels_to_add' list.
5.  **Fallback:** If no expert is a good fit or all are overloaded, you MUST still pick the *best available* person from the candidate list. Do not make up a user.
"""
    # [CHANGE 1 END]

    title = issue.get('enriched_title', issue.get('title', 'No Title'))
    body = issue.get('enriched_body', issue.get('body', 'No Body'))
    
    # [CHANGE 2 START] - Added Component to User Prompt
    user_prompt = f"""--- CONTEXT ---
{context}

--- NEW ISSUE TO ASSIGN ---
Repo: {issue['repo_name']}
Issue ID: {issue['issue_id']}
Title: {title}
Body: {body[:2000]}
Type: {issue.get('issue_type', 'unknown')}
Priority: {issue.get('priority', 'unknown')}
Component: {issue.get('component', 'unknown')} <--- CRITICAL

--- CANDIDATE LIST ---
You MUST assign this issue to one of: [{candidate_list_str}]

--- YOUR DECISION ---
Return ONLY the JSON object.
"""
    # [CHANGE 2 END]
    
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
            print(f"Calling Bedrock... (Attempt {attempt + 1}/{LLM_MAX_RETRIES + 1})")
            response = bedrock_client.invoke_model(
                modelId=LLM_MODEL_ID,
                body=json.dumps(body_obj),
                contentType="application/json",
                accept="application/json"
            )
            response_body = json.loads(response.get('body').read())
            result_text = response_body.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
            
            match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if match:
                decision = json.loads(match.group(0))
                handle_at = decision.get('assignee_handle')
                if handle_at and handle_at in candidate_handles and 'labels_to_add' in decision:
                    decision['assignee_handle'] = handle_at
                    return decision
                else:
                    print(f"Warning: LLM returned invalid handle '{handle_at}' (not in candidate list) or missing keys. Retrying.")
            else:
                print(f"Warning: LLM returned invalid JSON: {result_text}. Retrying.")
            
        except Exception as e:
            print(f"Error calling Bedrock (DeepSeek) on attempt {attempt+1}: {e}")
            if attempt < LLM_MAX_RETRIES:
                print(f"Sleeping for {LLM_RETRY_BACKOFF_FACTOR} seconds before retry...")
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
    assignee_handle_at = decision['assignee_handle']
    assignee_handle_clean = assignee_handle_at.lstrip('@')
    
    # [CHANGE 3 START] - Prepare labels separately
    # We collect all 3 critical labels: Type, Priority, Component
    # We filter out any that are None, empty, or 'unknown'
    pipeline_labels = []
    
    if issue.get('issue_type') and issue.get('issue_type').lower() != 'unknown':
        pipeline_labels.append(issue.get('issue_type'))
        
    if issue.get('priority') and issue.get('priority').lower() != 'unknown':
        pipeline_labels.append(issue.get('priority'))
        
    if issue.get('component') and issue.get('component').lower() != 'unknown':
        pipeline_labels.append(issue.get('component'))
        
    # Merge with any extra labels LLM might have suggested, ensuring uniqueness
    llm_labels = decision.get('labels_to_add', [])
    final_labels = list(set(pipeline_labels + llm_labels))
    
    # Ensure we don't send an empty label if something slipped through
    final_labels = [l for l in final_labels if l]
    # [CHANGE 3 END]
    
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
            "labels": final_labels
        }
        
        with httpx.Client() as client:
            response = client.patch(api_url, headers=headers, json=payload)
            response.raise_for_status()
        
        print(f"Successfully updated GitHub issue: assigned to {assignee_handle_at}, labels: {final_labels}")

    except Exception as e:
        print(f"WARNING: Failed to update GitHub issue {repo_name}#{issue_id}: {e}. Check app permissions.")

    # 2. Update DynamoDB
    print(f"Updating DynamoDB for {repo_name}#{issue_id}...")
    try:
        table.update_item(
            Key={
                'repo_name': repo_name,
                'issue_id': issue_id
            },
            UpdateExpression=(
                "SET current_assignee = :a, assignment_reason = :r, "
                "assignment_confidence = :c, assignment_time = :at, "
                "github_labels = :l, #s = :s, "
                "last_updated_pipeline = :lu"
            ),
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={
                ':a': assignee_handle_at,
                ':r': decision.get('reasoning', 'N/A'),
                ':c': Decimal(str(decision.get('confidence', 0.0))),
                ':at': datetime.now(timezone.utc).isoformat(),
                ':l': final_labels,
                ':s': 'open',
                ':lu': datetime.now(timezone.utc).isoformat()
            }
        )
        print("Successfully updated DynamoDB.")
        
    except Exception as e:
        print(f"ERROR: Failed to update DynamoDB for {repo_name}#{issue_id}: {e}")

def update_github_labels_only(
    issue: Dict,
    install_token: str,
    table: Any
):
    """
    Updates GitHub with labels ONLY. Used when no assignees are available.
    Also updates DynamoDB with the labels and status.
    """
    repo_name = issue['repo_name']
    issue_id = issue['issue_id']
    
    # [CHANGE 4 START] - Prepare labels separately here too
    labels_to_add = []
    
    if issue.get('issue_type') and issue.get('issue_type').lower() != 'unknown':
        labels_to_add.append(issue.get('issue_type'))
        
    if issue.get('priority') and issue.get('priority').lower() != 'unknown':
        labels_to_add.append(issue.get('priority'))
        
    if issue.get('component') and issue.get('component').lower() != 'unknown':
        labels_to_add.append(issue.get('component'))
    # [CHANGE 4 END]
    
    if not labels_to_add:
        print(f"No valid labels found for {repo_name}#{issue_id}. Nothing to update.")
        return

    print(f"Attempting to update GitHub labels for {repo_name}#{issue_id}...")
    
    # 1. Update GitHub API
    try:
        headers = {
            "Authorization": f"Bearer {install_token}",
            "Accept": "application/vnd.github.v3+json",
        }
        api_url = f"https://api.github.com/repos/{repo_name}/issues/{issue_id}"
        
        payload = {
            "labels": labels_to_add
            # NO "assignees" key
        }
        
        with httpx.Client() as client:
            response = client.patch(api_url, headers=headers, json=payload)
            response.raise_for_status()
        
        print(f"Successfully updated GitHub issue: added labels {labels_to_add}")

    except Exception as e:
        print(f"WARNING: Failed to update GitHub issue {repo_name}#{issue_id} with labels: {e}.")

    # 2. Update DynamoDB
    print(f"Updating DynamoDB for {repo_name}#{issue_id} (labels only)...")
    try:
        table.update_item(
            Key={
                'repo_name': repo_name,
                'issue_id': issue_id
            },
            UpdateExpression=(
                "SET assignment_reason = :r, "
                "github_labels = :l, #s = :s, "
                "last_updated_pipeline = :lu"
            ),
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={
                ':r': 'Pipeline completed. No candidates found in expertise map for assignment.',
                ':l': labels_to_add,
                ':s': 'open', # Still open, just not assigned
                ':lu': datetime.now(timezone.utc).isoformat()
            }
        )
        print("Successfully updated DynamoDB (labels only).")
        
    except Exception as e:
        print(f"ERROR: Failed to update DynamoDB for {repo_name}#{issue_id}: {e}")

# --- Main Handler ---
def lambda_handler(event, context):
    """
    Main Lambda handler. Triggered by EventBridge with a BATCH of *prioritized* issues.
    """
    print("Classify Assignee event received:")
    print(json.dumps(event))

    # --- 1. Load Configs and Validate Environment ---
    try:
        if not all([ISSUES_TABLE_NAME, EXPERTISE_TABLE_NAME, USER_AVAILABILITY_TABLE_NAME, SECRET_ARN, INSTALLATIONS_TABLE_NAME]):
            raise EnvironmentError("Missing one or more required environment variables.")

    except Exception as e:
        print(f"FATAL: {e}")
        return {'statusCode': 500, 'body': 'Internal configuration error'}
        
    # --- 2. Parse Event ---
    try:
        # [CHANGE 5 START] - Updated Event Type Check
        if event.get("source") != "github.issues" or event.get("detail-type") != "issue.batch.component_classified":
            print(f"Ignoring event from unknown source: {event.get('source')} or type: {event.get('detail-type')}")
            return
        # [CHANGE 5 END]
            
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
        
        first_issue = issues_list[0]
        installation_id = first_issue.get('installation_id')
        
        if not installation_id:
            print(f"Installation_id missing from payload. Fetching from 'github-installations' table...")
            install_table = dynamodb.Table(INSTALLATIONS_TABLE_NAME)
            
            try:
                # Note: This GSI is specified in the context doc.
                response = install_table.query(
                    IndexName='repo_name-index', 
                    KeyConditionExpression="repo_name = :r",
                    ExpressionAttributeValues={":r": repo_name}
                )
                if not response.get('Items'):
                     print(f"FATAL: Could not find installation_id for repo {repo_name}.")
                     return
                installation_id = int(response['Items'][0]['installation_id'])
                print(f"Found installation_id: {installation_id}")
            except Exception as e:
                print(f"CRITICAL ERROR: Could not query 'github-installations' by repo_name. Did you create the GSI 'repo_name-index'?")
                print(f"Error: {e}")
                raise

        install_token = get_installation_access_token(installation_id)
        expertise_context, candidate_handles = get_expertise_context(repo_name)
        
        if not candidate_handles:
            print(f"WARNING: No candidates found in RepoExpertise for {repo_name}. Will proceed to apply labels only.")
        
        availability_context = get_availability_context(candidate_handles)
        workload_context = get_workload_context(candidate_handles)
        
        final_context = f"{availability_context}\n{expertise_context}\n{workload_context}"
        print(f"Successfully generated all context.")
        
    except Exception as e:
        print(f"FATAL: Failed to prepare context for {repo_name}. Stopping. Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 4. Process Each Issue in the Batch ---
    table = dynamodb.Table(ISSUES_TABLE_NAME)
    
    for issue in issues_list:
        try:
            issue['repo_name'] = repo_name
            
            if candidate_handles:
                # --- Path A: Full Assignment (Original Logic) ---
                decision = run_assignment_llm(issue, final_context, candidate_handles)
                
                if not decision:
                    print(f"Warning: LLM failed to provide assignment for {issue['issue_id']}. Skipping.")
                    continue

                # Update GitHub and DynamoDB
                update_github_and_dynamodb(issue, decision, install_token, table)
                
            else:
                # --- Path B: Labels Only (New Logic) ---
                print(f"No candidates for {issue['issue_id']}. Updating labels only.")
                update_github_labels_only(issue, install_token, table)

        except Exception as e:
            print(f"ERROR: Failed to process issue {issue.get('issue_id')}: {e}")
            import traceback
            traceback.print_exc() # Print full error for debugging
            continue
    
    # [CRITICAL FIX] Loop stopped by removing event emission here
    print(f"Successfully processed batch for {repo_name}.")
    return {'statusCode': 200, 'body': 'Batch assignment complete.'}