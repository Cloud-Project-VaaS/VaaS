import json
import boto3
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# IMPORTANT: Change 'ap-south-1' if you used a different AWS region
lambda_client = boto3.client('lambda', region_name='ap-south-1') 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run_classification():
    repo_name = request.form['repo_name']
    github_pat = request.form['github_pat']
    hours_since = request.form['hours_since']

    lambda_payload = {
        "repo_name": repo_name,
        "github_pat": github_pat,
        "hours_since": hours_since
    }

    try:
        response = lambda_client.invoke(
            FunctionName='fetch_and_classify_issues',
            InvocationType='RequestResponse',
            Payload=json.dumps(lambda_payload)
        )
        
        response_payload = json.load(response['Payload'])

        if 'body' in response_payload:
            results = json.loads(response_payload['body'])
            return jsonify(results)
        else:
            return jsonify(response_payload), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
