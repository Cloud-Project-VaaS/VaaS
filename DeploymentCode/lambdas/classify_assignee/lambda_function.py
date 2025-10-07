import json

def lambda_handler(event, context):
    title = event.get('title', '').lower()
    body = event.get('body', '').lower()
    text_content = f"{title} {body}"
    
    if any(k in text_content for k in ["ui", "frontend", "css", "button"]):
        return {"assignee": "frontend-team"}
    elif any(k in text_content for k in ["backend", "database", "server", "api"]):
        return {"assignee": "backend-team"}
    elif any(k in text_content for k in ["docs", "documentation", "readme"]):
        return {"assignee": "docs-team"}
        
    return {"assignee": "unassigned"}