from flask import Flask, render_template, request, jsonify
import boto3
import time
import os
from threading import Lock

# --- Configuration ---

# 1. Set your secret password here
YOUR_PASSWORD = "L0nkWhlbFnEHK8ddSyasHJK2UFy3uPkv" 

# 2. Name of the Lambda function to trigger
#    We'll get this from an environment variable, but set a default.
LAMBDA_FUNCTION_NAME = os.environ.get("LAMBDA_FUNCTION_NAME", "fetch-and-classify-issues")

# 3. Cooldown (in seconds)
COOLDOWN_SECONDS = 300  # 5 minutes

# --- Global state for cooldown ---
app = Flask(__name__)
last_run_time = 0
time_lock = Lock()

# --- Boto3 Client ---
# Boto3 will automatically use the IAM Role from the EC2 instance.
# No keys needed!
lambda_client = boto3.client('lambda', region_name="ap-south-1") # Use your Lambda's region

@app.route('/')
def index():
    """Serves the main HTML page."""
    # Note: Flask looks for 'index.html' in a folder named 'templates'
    return render_template('index.html')

@app.route('/trigger-pipeline', methods=['POST'])
def trigger_pipeline():
    """API endpoint that the HTML button calls."""
    global last_run_time
    
    data = request.json
    password = data.get('password')

    # 1. Check Password
    if not password or password != YOUR_PASSWORD:
        return jsonify({"success": False, "message": "Invalid Password."}), 401

    # 2. Check Cooldown (Thread-safe)
    with time_lock:
        current_time = time.time()
        time_since_last_run = current_time - last_run_time
        
        if time_since_last_run < COOLDOWN_SECONDS:
            remaining_time = int(COOLDOWN_SECONDS - time_since_last_run)
            return jsonify({
                "success": False, 
                "message": f"Please wait. Cooldown active for {remaining_time} more seconds."
            }), 429 # 429 Too Many Requests

        # 3. If cooldown is over, update the last run time
        last_run_time = current_time

    # 4. Trigger the Lambda
    try:
        print(f"Triggering Lambda: {LAMBDA_FUNCTION_NAME}")
        
        # We use 'Event' for an asynchronous (non-blocking) invocation.
        # This is CRITICAL. The webpage won't hang.
        lambda_client.invoke(
            FunctionName=LAMBDA_FUNCTION_NAME,
            InvocationType='Event' 
        )
        
        print("Lambda triggered successfully.")
        return jsonify({
            "success": True, 
            "message": f"Pipeline triggered! The 5-minute cooldown has started."
        })

    except Exception as e:
        print(f"Error triggering Lambda: {e}")
        # If it fails, reset the cooldown so they can try again
        with time_lock:
            last_run_time = 0
        return jsonify({"success": False, "message": f"Error triggering Lambda: {e}"}), 500

if __name__ == '__main__':
    # Make sure to run this with: flask run --host=0.0.0.0
    app.run()