import json
from typing import Dict, Any, List

def handler(event: List[Dict[str, Any]], context: Any) -> List[Dict[str, Any]]:
    # 'event' is now a LIST of issue objects
    issues_list = event
    results_list = []
    
    for issue in issues_list:
        title = (issue.get('title') or "").lower()
        body = (issue.get('body') or "").lower()
        text_content = f"{title} {body}"
        assignee = "unassigned"

        if any(k in text_content for k in ["ui", "frontend", "css", "button"]):
            assignee = "frontend-team"
        elif any(k in text_content for k in ["backend", "database", "server", "api"]):
            assignee = "backend-team"
        elif any(k in text_content for k in ["docs", "documentation", "readme"]):
            assignee = "docs-team"
        
        # Add the issue_id to the result for matching
        results_list.append({
            'issue_id': issue.get('id'),
            'assignee': assignee
        })
        
    # Return a LIST of results
    return results_list