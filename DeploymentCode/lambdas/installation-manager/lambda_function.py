import json
import boto3
import hmac
import hashlib
import os
from datetime import datetime

# Initialize clients
secrets_client = boto3.client('secretsmanager')
dynamodb = boto3.resource('dynamodb')
eventbridge_client = boto3.client('eventbridge')  # <-- NEW
table = dynamodb.Table('github-installations')

# Load the Secret ARN from an Environment Variable
SECRET_ARN = os.environ.get('SECRET_ARN') 
if not SECRET_ARN:
    raise ValueError("Error: SECRET_ARN environment variable is not set.")

# Load the Webhook Secret from Secrets Manager
secret = secrets_client.get_secret_value(SecretId=SECRET_ARN)
WEBHOOK_SECRET = json.loads(secret['SecretString'])['WEBHOOK_SECRET']


def verify_signature(event):
    """Verifies the GitHub webhook signature."""
    signature = event['headers'].get('x-hub-signature-256') or event['headers'].get('X-Hub-Signature-256')
    if not signature:
        raise Exception("Signature not found")

    sha_name, signature_hash = signature.split('=', 1)
    if sha_name != 'sha256':
        raise Exception("Signature format not supported")

    mac = hmac.new(WEBHOOK_SECRET.encode(), msg=event['body'].encode(), digestmod=hashlib.sha256)
    return hmac.compare_digest(mac.hexdigest(), signature_hash)


def lambda_handler(event, context):
    
    # 1. Verify the signature
    try:
        if not verify_signature(event):
            return {'statusCode': 401, 'body': 'Invalid signature'}
    except Exception as e:
        print(f"Signature verification failed: {e}")
        return {'statusCode': 401, 'body': str(e)}
        
    # 2. Parse the payload
    try:
        payload = json.loads(event['body'])
        action = payload.get('action')
        installation_id = payload.get('installation', {}).get('id')
        
        if not installation_id:
            return {'statusCode': 200, 'body': 'No installation ID, nothing to do.'}
            
        # 3. Add or Remove from DynamoDB
        
        if action == 'created' or action == 'added':
            repos_list = payload.get('repositories_added', payload.get('repositories', []))
            for repo in repos_list:
                repo_name = repo.get('full_name')
                if repo_name:
                    print(f"Adding repo: {repo_name} for install: {installation_id}")
                    # 1. Save to DynamoDB (your existing code)
                    table.put_item(
                        Item={
                            'installation_id': installation_id,
                            'repo_name': repo_name,
                            'created_at': datetime.utcnow().isoformat()
                        }
                    )
                    
                    # 2. Send event to EventBridge (THE NEW PART) <-- NEW
                    print(f"Sending 'repo.added' event for {repo_name} to EventBridge")
                    eventbridge_client.put_events(
                        Entries=[
                            {
                                'Source': 'github.webhook.handler',
                                'DetailType': 'repository.added',
                                'EventBusName': 'github-app-events', # <-- Use the bus name from Step 1
                                'Detail': json.dumps({
                                    'installation_id': installation_id,
                                    'repo_name': repo_name
                                })
                            }
                        ]
                    )

        elif action == 'deleted' or action == 'removed':
            repos_list = payload.get('repositories_removed', payload.get('repositories', []))
            for repo in repos_list:
                repo_name = repo.get('full_name')
                if repo_name:
                    print(f"Removing repo: {repo_name} for install: {installation_id}")
                    table.delete_item(
                        Key={
                            'installation_id': installation_id,
                            'repo_name': repo_name
                        }
                    )
        
        else:
            print(f"Ignoring action: {action}")
            
        return {'statusCode': 200, 'body': 'Webhook processed successfully'}

    except Exception as e:
        print(f"Error processing webhook: {e}")
        return {'statusCode': 500, 'body': 'Internal server error'}