import json
from datetime import datetime, timezone
from pathlib import Path

# --- CONFIGURATION ---
STALE_AFTER_DAYS = 60
CLOSE_AFTER_DAYS = 7
EXEMPT_LABELS = {"pinned", "security", "good first issue"}
EXEMPT_ASSIGNEES = set()


def parse_time(timestr):
    """Convert GitHub timestamp to datetime."""
    return datetime.strptime(timestr, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def last_activity(issue):
    """Return the latest activity timestamp for an issue."""
    updated = parse_time(issue["updated_at"])
    if issue.get("comments") and isinstance(issue["comments"], list):
        times = [parse_time(c["created_at"]) for c in issue["comments"]]
        return max([updated] + times)
    return updated


def has_exempt_label(issue):
    labels = {l["name"].lower() for l in issue.get("labels", [])}
    return not EXEMPT_LABELS.isdisjoint(labels)


def has_exempt_assignee(issue):
    assignees = {a["login"].lower() for a in issue.get("assignees", [])}
    return not EXEMPT_ASSIGNEES.isdisjoint(assignees)


def is_stale(issue, now):
    if issue["state"] != "open":
        return False, "closed"
    if issue.get("locked", False):
        return False, "locked"
    if has_exempt_label(issue):
        return False, "exempt_label"
    if has_exempt_assignee(issue):
        return False, "exempt_assignee"

    days_inactive = (now - last_activity(issue)).days
    if days_inactive > STALE_AFTER_DAYS:
        return True, f"inactive {days_inactive}d"
    return False, f"active {days_inactive}d"


def main(path):
    data = json.loads(Path(path).read_text())
    now = datetime.now(timezone.utc)

    stale_issues = []
    for issue in data:
        stale, reason = is_stale(issue, now)
        if stale:
            stale_issues.append({
                "number": issue["number"],
                "title": issue["title"],
                "reason": reason,
                "last_activity": str(last_activity(issue))
            })

    print(f"Total issues: {len(data)}")
    print(f"Stale issues: {len(stale_issues)}\n")

    for i in stale_issues:
        print(f"#{i['number']}: {i['title']}\n  -> {i['reason']} (last activity {i['last_activity']})\n")


if __name__ == "__main__":
    # Example: python stale_identifier.py short_microsoft_vs_code.json
    import sys
    if len(sys.argv) < 2:
        print("Usage: python stale_identifier.py <issues.json>")
        sys.exit(1)
    main(sys.argv[1])
