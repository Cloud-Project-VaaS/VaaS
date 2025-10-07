import json

def lambda_handler(event, context):
    title = event.get('title', '').lower()
    body = event.get('body', '').lower()
    text_content = f"{title} {body}"
    
    if any(k in text_content for k in ["urgent", "critical", "production", "down"]):
        return {"priority": "High"}
    elif any(k in text_content for k in ["minor", "typo", "cosmetic"]):
        return {"priority": "Low"}
        
    return {"priority": "Medium"}