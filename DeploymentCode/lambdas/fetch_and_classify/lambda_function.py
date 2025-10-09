import json
import os
import boto3
import requests
from datetime import datetime, timedelta

# Initialize AWS clients outside the handler for reuse
s3 = boto3.client('s3')
lambda_client = boto3.client('lambda')

# Get the S3 bucket name from an environment variable for security
S3_BUCKET = os.environ.get('S3_BUCKET_NAME')

def lambda_handler(event, context):
    """
    Main orchestrator function.
    - Takes repo_name, github_pat, and hours_since as input.
    - Checks S3 for existing metadata; if not found, scans and stores it.
    - Fetches recent GitHub issues based on the time window.
    - Invokes classifier Lambdas for each issue.
    - Returns a JSON object with the aggregated results.
    """
    repo_name = event.get('repo_name')
    github_pat = event.get('github_pat')
    # Default to 1 hour if 'hours_since' is not provided
    hours_since = int(event.get('hours_since', 1))

    if not repo_name or not github_pat:
        return {'statusCode': 400, 'body': json.dumps('Error: Missing repo_name or github_pat in the input.')}
    if not S3_BUCKET:
         return {'statusCode': 500, 'body': json.dumps('Error: S3_BUCKET_NAME environment variable is not set.')}


    # --- 1. One-Time Metadata Scan Logic ---
    # Create a unique key for the metadata file based on the repo name
    metadata_key = f"{repo_name.replace('/', '-')}-metadata.json"
    
    try:
        # Check if the metadata file already exists in S3 without downloading it
        s3.head_object(Bucket=S3_BUCKET, Key=metadata_key)
        print(f"Metadata for {repo_name} already exists. Skipping scan.")
    except s3.exceptions.ClientError as e:
        # If the file is not found (error code 404), perform the scan
        if e.response['Error']['Code'] == '404':
            print(f"Metadata for {repo_name} not found. Performing one-time scan.")
            
            headers = {"Authorization": f"token {github_pat}"}
            api_url = f"https://api.github.com/repos/{repo_name}"
            response = requests.get(api_url, headers=headers)
            
            if response.status_code != 200:
                 return {'statusCode': response.status_code, 'body': json.dumps(f"Failed to fetch repo data: {response.text}")}
            
            repo_data = response.json()
            
            # Store the fetched metadata in S3
            s3.put_object(
                Bucket=S3_BUCKET,
                Key=metadata_key,
                Body=json.dumps(repo_data, indent=2),
                ContentType='application/json'
            )
            print(f"Successfully scanned and stored metadata in S3.")

    # --- 2. Fetch Recent Issues ---
    headers = {"Authorization": f"token {github_pat}"}
    since_time = (datetime.utcnow() - timedelta(hours=hours_since)).isoformat()
    issues_url = f"https://api.github.com/repos/{repo_name}/issues?since={since_time}"
    
    response = requests.get(issues_url, headers=headers)
    issues = response.json()
    classified_results = []
    
    # --- 3. Invoke Classifiers for Each Issue ---
    # This is the NEW code block to paste inside the 'for' loop
    for issue in issues:
        issue_payload = json.dumps(issue)

        # --- Check if it's an Issue or a Pull Request ---
        item_type = "Pull Request" if "pull_request" in issue else "Issue"

        # 1. Invoke the heavy classifier for the type
        heavy_type_res = lambda_client.invoke(FunctionName='classify_issue_heavy', Payload=issue_payload)
        heavy_type_payload = json.loads(heavy_type_res['Payload'].read())

        # 2. Keep using the simple classifiers for priority and assignee
        priority_res = lambda_client.invoke(FunctionName='classify_priority', Payload=issue_payload)
        assignee_res = lambda_client.invoke(FunctionName='classify_assignee', Payload=issue_payload)

        priority_payload = json.loads(priority_res['Payload'].read())
        assignee_payload = json.loads(assignee_res['Payload'].read())

        # 3. Aggregate the results, now including the 'item_type'
        classified_results.append({
            "issue_title": issue.get('title'),
            "url": issue.get('html_url'),
            "item_type": item_type,  # <-- ADDED THIS NEW FIELD
            "classification": {
                "type": heavy_type_payload.get('category'),
                "priority": priority_payload.get('priority'),
                "assignee": assignee_payload.get('assignee')
            },
            "scores": heavy_type_payload.get('scores')
        })

    # Return the final list of classified issues
    return {
        'statusCode': 200,
        'body': json.dumps(classified_results, indent=2)
    }