import json
import boto3
import requests
from datetime import datetime, timedelta

lambda_client = boto3.client('lambda')
S3_BUCKET = "cloudproject-deploymentcode-metadata"

def get_results_from_payload(response):
    """
    Helper function to correctly parse the payload from a Lambda invocation.
    It handles cases where the invoked function returns a direct JSON object/list
    or an API Gateway-style proxy response with a stringified 'body'.
    """
    payload = json.loads(response['Payload'].read())
    # Check if the payload is a dictionary containing a 'body' key
    if isinstance(payload, dict) and 'body' in payload:
        # If so, parse the body string to get the actual results
        return json.loads(payload['body'])
    else:
        # Otherwise, the payload is the result itself
        return payload

def lambda_handler(event, context):
    repo_name = event.get('repo_name')
    github_pat = event.get('github_pat')
    hours_since = int(event.get('hours_since', 1))

    if not repo_name or not github_pat:
        return {'statusCode': 400, 'body': json.dumps('Error: Missing repo_name or github_pat')}

    headers = {"Authorization": f"token {github_pat}"}
    since_time = (datetime.utcnow() - timedelta(hours=hours_since)).isoformat()
    issues_url = f"https://api.github.com/repos/{repo_name}/issues?since={since_time}"
    response = requests.get(issues_url, headers=headers)
    issues = response.json()

    if not issues:
        return {'statusCode': 200, 'body': json.dumps([])}

    payload = json.dumps(issues)
    type_response = lambda_client.invoke(FunctionName='classify-issue-heavy', Payload=payload)
    priority_response = lambda_client.invoke(FunctionName='classify_priority', Payload=payload)
    assignee_response = lambda_client.invoke(FunctionName='classify_assignee', Payload=payload)
    
    # Use the new helper function to safely extract results
    type_results = get_results_from_payload(type_response)
    priority_results = get_results_from_payload(priority_response)
    assignee_results = get_results_from_payload(assignee_response)

    # --- Merge All Results ---
    merged_data = {}
    for issue in issues:
        issue_id = issue.get('id')
        merged_data[issue_id] = {
            "issue_title": issue.get('title'),
            "url": issue.get('html_url'),
            "item_type": "Pull Request" if "pull_request" in issue else "Issue",
            "classification": {}
        }
    
    for result in type_results:
        issue_id = result.get('issue_id')
        if issue_id in merged_data:
            merged_data[issue_id]['classification']['type'] = result.get('category')
            merged_data[issue_id]['type_scores'] = result.get('scores')
            
    for result in priority_results:
        issue_id = result.get('issue_id')
        if issue_id in merged_data:
            merged_data[issue_id]['classification']['priority'] = result.get('priority')
            merged_data[issue_id]['priority_scores'] = result.get('scores')

    for result in assignee_results:
        issue_id = result.get('issue_id')
        if issue_id in merged_data:
            merged_data[issue_id]['classification']['assignee'] = result.get('assignee')
    
    final_results = list(merged_data.values())

    return {
        'statusCode': 200,
        'body': json.dumps(final_results, indent=2)
    }