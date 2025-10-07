import json

def lambda_handler(event, context):
    # 'event' is the full JSON object for a GitHub issue
    title = event.get('title', '').lower()
    body = event.get('body', '').lower()
    text_content = f"{title} {body}"
    
    if any(k in text_content for k in ["error", "bug", "crash", "exception", "fail"]):
        return {"category": "Bug"}
    elif any(k in text_content for k in ["feature", "add", "implement", "support", "enhancement"]):
        return {"category": "Feature Request"}
    elif any(k in text_content for k in ["spam", "buy", "cheap"]):
        return {"category": "Spam"}
        
    return {"category": "Other"}