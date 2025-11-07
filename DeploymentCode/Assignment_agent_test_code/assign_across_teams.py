#!/usr/bin/env python3

"""
assign_across_teams.py

Assigns open, unassigned DynamoDB issues to the best-fit member across all
configured teams using a Bedrock LLM.

This script scans a DynamoDB table for items where `assignee_handle` is not set.
It loads a team configuration from a YAML file, prepares a detailed payload
for each issue (including all team/member details and SLA-derived hours),
and sends batches of these payloads to an AWS Bedrock model.

The LLM returns a JSON object with the best assignee (team, role, handle,
reason, confidence). The script validates this recommendation and falls back to
a deterministic (earliest start time) assignment if validation fails or the
LLM call errors. Finally, it idempotently updates the DynamoDB item with the
assignment.

---
🚀 How to Run Locally
---

1.  **Install Dependencies:**
    ```bash
    pip install boto3 pyyaml
    ```

2.  **Create `team_config.yml`:**
    (Place in the same directory or set TEAM_CONFIG_PATH)
    ```yaml
    teams:
      - name: "backend"
        members:
          - handle: "@anil"
            role: "lead"
            working_hours_utc: "09:00-17:00"
          - handle: "@bhavna"
            role: "reviewer"
            working_hours_utc: "10:30-18:30"
      - name: "frontend"
        members:
          - handle: "@dave"
            role: "lead"
            working_hours_utc: "08:00-16:00"
    ```

3.  **Set Environment Variables:**
    ```bash
    # Required:
    export AWS_REGION="us-east-1"

    # Required for Bedrock:
    export LLM_MODEL_ID="anthropic.claude-3-sonnet-20240229-v1:0" # Or deepseek, etc.
    export BEDROCK_REGION="us-east-1" # Region where model is available

    # Optional:
    export DDB_TABLE="IssuesTrackingTable"
    export TEAM_CONFIG_PATH="./team_config.yml"
    export BATCH_SIZE="10"
    export DRY_RUN="1" # "1" to log actions without writing to DDB
    export LOG_LEVEL="INFO"
    ```

4.  **Run the script:**
    ```bash
    python assign_across_teams.py
    ```

5.  **Run with CLI overrides:**
    ```bash
    # Process only 5 issues from 'my-org/my-repo' in dry-run
    python assign_across_teams.py --repo "my-org/my-repo" --limit 5 --dry-run
    ```

---
📦 IAM Permissions Required
---

The IAM role executing this script needs:
-   `dynamodb:Scan`: To find unassigned issues.
-   `dynamodb:UpdateItem`: To write the assignment back.
-   `bedrock:InvokeModel`: To call the Bedrock LLM for assignments.
"""

import os
import sys
import json
import logging
import time
import argparse
from datetime import datetime, timezone, time as dt_time
from decimal import Decimal
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Generator,
    TypedDict,
    Tuple,
    Set,
    Literal,
)

# Third-party imports
try:
    import yaml
except ImportError:
    print(
        "Error: 'pyyaml' library not found. Please install it: pip install pyyaml",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    import boto3
    from botocore.exceptions import ClientError
    from boto3.dynamodb.conditions import Attr
except ImportError:
    print(
        "Error: 'boto3' library not found. Please install it: pip install boto3",
        file=sys.stderr,
    )
    sys.exit(1)


# --- Configuration ---

# Load from environment variables
AWS_REGION = os.environ.get("AWS_REGION")
DDB_TABLE = os.environ.get("DDB_TABLE", "IssuesTrackingTable")
TEAM_CONFIG_PATH = os.environ.get("TEAM_CONFIG_PATH", "./team_config.yml")
LLM_MODEL_ID = os.environ.get(
    "LLM_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0"
)
BEDROCK_REGION = os.environ.get("BEDROCK_REGION") or AWS_REGION
DEFAULT_BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "10"))
DEFAULT_DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# Max chars of issue body to send to LLM
MAX_BODY_CHARS = 800

# LLM retry settings
LLM_MAX_RETRIES = 2
LLM_RETRY_BACKOFF_FACTOR = 0.5  # Seconds

# --- Logging Setup ---

log = logging.getLogger(__name__)
log.setLevel(LOG_LEVEL)
if not log.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", "%Y-m-d %H:M:%S"
    )
    handler.setFormatter(formatter)
    log.addHandler(handler)


# --- Type Definitions ---

SlaUnit = Literal["hours", "days"]


class TeamMember(TypedDict):
    """Stores configuration for a single team member."""

    handle: str
    role: str
    working_hours_utc: str
    # Computed fields
    start_time_utc: dt_time
    daily_hours: float


class Team(TypedDict):
    """Stores configuration for a single team."""

    name: str
    members: List[TeamMember]


class Issue(TypedDict):
    """Represents a DynamoDB issue item."""

    repo_name: str
    issue_id: str
    title: str
    body: Optional[str]
    classification: str
    priority: str
    sla_target: str
    assignment_time: Optional[str]


class LlmDecision(TypedDict):
    """Expected output structure from the LLM for a single issue."""

    team_name: str
    role: str
    person_handle: str
    reason: str
    confidence: float


class FallbackAssignee(TypedDict):
    """Stores pre-calculated fallback assignee details."""

    handle: str
    role: str
    team_name: str


class FallbackMap(TypedDict):
    """Container for pre-calculated fallback assignees."""

    lead: FallbackAssignee
    reviewer: FallbackAssignee


class RunStats(TypedDict):
    """Counters for the final summary report."""

    scanned: int
    batched: int
    assigned_llm: int
    assigned_fallback: int
    updated: int
    skipped_idempotent: int
    errors: int


# --- Bedrock LLM Client & Prompt ---

# Global client, initialized in main()
bedrock_client: Optional[boto3.client] = None

# System prompt for the LLM
PROMPT_TEMPLATE = """You are a precise triage assistant. For each input issue, choose exactly ONE best contributor from ALL provided teams.

Return ONLY a single valid JSON object. No prose, no Markdown. The JSON keys must match the input keys (e.g., "backend#123"). Each value MUST be an object with EXACT fields:
- "team_name": string (must match one of the provided teams)
- "role": "lead" or "reviewer" (must match the chosen member's role)
- "person_handle": string (must match a member handle in the chosen team)
- "reason": string (≤ 30 words, concise, practical justification)
- "confidence": number in [0.0, 1.0]

Decision rules:
1) Use issue fields (classification, priority, title, body) and SLA fields:
   - If "sla_target" is in business_hours: consider "sla_hours_base".
   - If "sla_target" is in business_days: consider each member's "member_daily_hours" and "target_hours_if_assigned".
2) For priority high/P1, favor leads unless a reviewer is clearly better.
3) For medium/low (P2/P3), prefer reviewers; escalate to leads only if needed for speed/fit.
4) Consider working_hours_utc. Prefer earlier daily start when rapid response is valuable.
5) Always output a valid choice even if uncertain; lower the confidence if unsure.
6) Keep output minimal and strictly follow the required JSON schema.

Output: a SINGLE JSON object mapping each input key → decision object.
"""


def invoke_llm_batch(
    batch_data: Dict[str, Any], prompt_template_str: str
) -> Dict[str, Any]:
    """
    Sends a BATCH of issues to the LLM for assignment.

    This is a blocking function that includes retry logic for transient
    Bedrock errors.

    Args:
        batch_data (Dict[str, Any]): The dictionary of issues to process,
            where keys are "repo#id" and values are the issue payloads.
        prompt_template_str (str): The system prompt to send to the LLM.

    Returns:
        Dict[str, Any]: The parsed JSON response from the LLM, mapping
        issue keys to LlmDecision objects. Returns {} on total failure.

    Raises:
        ValueError: If the configured LLM_MODEL_ID is not supported.
    """
    num_issues = len(batch_data)
    issue_keys = list(batch_data.keys())
    if not bedrock_client:
        log.error("Bedrock client is not initialized.")
        return {}

    try:
        system_prompt = prompt_template_str.format(num_users=num_issues)
    except KeyError:
        system_prompt = prompt_template_str

    user_prompt = json.dumps(batch_data, indent=2)
    body_obj = {}

    # Construct the request body based on the model provider
    if "anthropic.claude" in LLM_MODEL_ID:
        messages = [{"role": "user", "content": user_prompt}]
        body_obj = {
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.1,
            "top_p": 0.9,
            "system": system_prompt,
            "anthropic_version": "bedrock-2023-05-31",
        }
    elif "openai.gpt-oss-120b" in LLM_MODEL_ID or "deepseek" in LLM_MODEL_ID:
        body_obj = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 4096,
            "temperature": 0.1,
            "top_p": 0.9,
        }
    else:
        raise ValueError(
            f"Unsupported model ID: {LLM_MODEL_ID}. "
            "Please update 'invoke_llm_batch'."
        )

    raw_output = ""
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            response = bedrock_client.invoke_model(
                modelId=LLM_MODEL_ID,
                body=json.dumps(body_obj),
                contentType="application/json",
                accept="application/json",
            )
            response_body = json.loads(response.get("body").read())
            raw_output = ""

            if "anthropic.claude" in LLM_MODEL_ID:
                raw_output = response_body.get("content", [{}])[0].get("text", "")
            elif "openai.gpt-oss-120b" in LLM_MODEL_ID or "deepseek" in LLM_MODEL_ID:
                raw_output = (
                    response_body.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )

            # Find and parse the JSON block
            start_index = raw_output.find("{")
            end_index = raw_output.rfind("}")
            if start_index != -1 and end_index != -1 and end_index > start_index:
                json_output = raw_output[start_index : end_index + 1]
                parsed_json = json.loads(json_output)
                return parsed_json
            else:
                raise json.JSONDecodeError("No JSON object found", raw_output, 0)

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == "ThrottlingException" and attempt < LLM_MAX_RETRIES:
                sleep_time = (2**attempt) * LLM_RETRY_BACKOFF_FACTOR
                log.warning(
                    "LLM ThrottlingException for batch. Retrying in %.1fs...",
                    sleep_time,
                )
                time.sleep(sleep_time)
            else:
                log.error(
                    f"ERROR: LLM call failed for batch {issue_keys}. Error: {e}"
                )
                log.debug(f"Raw output from LLM: {raw_output}")
                return {}  # Non-transient error or retries exhausted
        except Exception as e:
            log.error(
                f"ERROR: LLM (Pass 1 Batch) failed for issues {issue_keys}. Error: {e}"
            )
            log.debug(f"Raw output from LLM: {raw_output}")
            return {}  # Catch-all for JSON parsing, etc.

    log.error(f"ERROR: LLM call failed for batch {issue_keys} after all retries.")
    return {}


# --- Helper Functions ---


def parse_working_hours(hours_str: str) -> Tuple[dt_time, float]:
    """
    Parses a "HH:MM-HH:MM" string into a start time and duration.

    Args:
        hours_str (str): The working hours string (e.g., "09:00-17:00").

    Returns:
        Tuple[dt_time, float]: A tuple containing the
        `datetime.time` object for the start time and a float for the
        total daily hours.

    Raises:
        ValueError: If the string format is invalid.
    """
    try:
        start_str, end_str = hours_str.split("-")
        start_time = datetime.strptime(start_str, "%H:%M").time()
        end_time = datetime.strptime(end_str, "%H:%M").time()

        # Calculate duration in hours
        today = datetime.now(timezone.utc).date()
        start_dt = datetime.combine(today, start_time)
        end_dt = datetime.combine(today, end_time)
        duration_seconds = (end_dt - start_dt).total_seconds()
        daily_hours = duration_seconds / 3600.0

        if daily_hours <= 0:
            # Handle overnight shifts, though not in spec
            daily_hours += 24.0

        return start_time, daily_hours
    except Exception as e:
        log.error(f"Invalid working_hours_utc format '{hours_str}': {e}")
        raise ValueError(f"Invalid working_hours_utc format: {hours_str}")


def load_team_config(config_path: str) -> Tuple[List[Team], FallbackMap]:
    """
    Loads and validates team config YAML, computing hours and fallbacks.

    This function reads the YAML, parses member working hours, and
    pre-calculates the deterministic fallback assignees (earliest
    start time) for both 'lead' and 'reviewer' roles.

    Args:
        config_path (str): The file path to the team_config.yml.

    Returns:
        Tuple[List[Team], FallbackMap]: A tuple containing the list of
        validated team data and the map of fallback assignees.

    Raises:
        SystemExit: If the config file is not found or is invalid.
    """
    log.info(f"Loading team config from {config_path}...")
    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        log.critical(f"FATAL: Team config file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        log.critical(f"FATAL: Error parsing team config YAML: {e}")
        sys.exit(1)

    if "teams" not in config_data or not isinstance(config_data["teams"], list):
        log.critical(
            "FATAL: Team config must have a top-level 'teams' list."
        )
        sys.exit(1)

    all_teams: List[Team] = config_data["teams"]
    all_members: List[Tuple[TeamMember, str]] = []  # (member, team_name)
    validated_teams: List[Team] = []

    try:
        for team in all_teams:
            validated_members: List[TeamMember] = []
            for member in team.get("members", []):
                start_time, daily_hours = parse_working_hours(
                    member["working_hours_utc"]
                )
                member["start_time_utc"] = start_time
                member["daily_hours"] = daily_hours
                validated_members.append(member)
                all_members.append((member, team["name"]))
            team["members"] = validated_members
            validated_teams.append(team)
    except (KeyError, ValueError) as e:
        log.critical(f"FATAL: Invalid team config structure: {e}")
        sys.exit(1)

    # Pre-calculate fallback assignees
    if not all_members:
        log.critical("FATAL: Team config has no members defined.")
        sys.exit(1)

    # Sort all members by start time (earliest first)
    all_members.sort(key=lambda m: m[0]["start_time_utc"])

    fallback_lead = next(
        (m for m in all_members if m[0]["role"] == "lead"), None
    )
    fallback_reviewer = next(
        (m for m in all_members if m[0]["role"] == "reviewer"), None
    )

    if not fallback_lead:
        log.warning("No 'lead' found for fallback; using earliest member.")
        fallback_lead = all_members[0]
    if not fallback_reviewer:
        log.warning("No 'reviewer' found for fallback; using earliest member.")
        fallback_reviewer = all_members[0]

    fallback_map: FallbackMap = {
        "lead": {
            "handle": fallback_lead[0]["handle"],
            "role": fallback_lead[0]["role"],
            "team_name": fallback_lead[1],
        },
        "reviewer": {
            "handle": fallback_reviewer[0]["handle"],
            "role": fallback_reviewer[0]["role"],
            "team_name": fallback_reviewer[1],
        },
    }

    log.info(f"Loaded {len(validated_teams)} teams and {len(all_members)} members.")
    log.info(
        f"Fallback map: Lead -> {fallback_map['lead']['handle']}, "
        f"Reviewer -> {fallback_map['reviewer']['handle']}"
    )
    return validated_teams, fallback_map


def parse_sla_target(sla_str: str) -> Tuple[Optional[SlaUnit], Optional[int]]:
    """
    Parses an SLA string into a unit and value.

    Args:
        sla_str (str): The SLA target string (e.g., "8 business_hours",
            "3 business_days", or "1 business_day").

    Returns:
        Tuple[Optional[SlaUnit], Optional[int]]: A tuple of
        (unit, value). Returns (None, None) if parsing fails.

    Example:
        >>> parse_sla_target("8 business_hours")
        ('hours', 8)
        >>> parse_sla_target("1 business_day")
        ('days', 1)
    """
    if not sla_str or not isinstance(sla_str, str):
        return None, None
    try:
        parts = sla_str.lower().split()
        if len(parts) != 2:
            return None, None

        value = int(parts[0])
        unit_str = parts[1]

        if "hour" in unit_str:
            return "hours", value
        if "day" in unit_str:
            return "days", value

        return None, None
    except Exception:
        log.warning(f"Could not parse SLA string: {sla_str}")
        return None, None


def build_llm_payload(
    issue: Issue, all_teams_data: List[Team]
) -> Dict[str, Any]:
    """
    Builds the JSON payload for a single issue to be sent to the LLM.

    This includes raw issue data and the computed team/member data,
    including member-specific SLA target hours.

    Args:
        issue (Issue): The raw issue data from DynamoDB.
        all_teams_data (List[Team]): The fully-loaded team configuration.

    Returns:
        Dict[str, Any]: The complete payload for this issue.
    """
    sla_unit, sla_value = parse_sla_target(issue.get("sla_target", ""))

    payload = {
        "issue_id": issue["issue_id"],
        "repo_name": issue["repo_name"],
        "title": issue["title"],
        "body": (issue.get("body") or "")[:MAX_BODY_CHARS],
        "classification": issue.get("classification"),
        "priority": issue.get("priority"),
        "sla_target": issue.get("sla_target"),
        "sla_hours_base": sla_value if sla_unit == "hours" else None,
        "teams": [],
    }

    # Build the member-specific data
    teams_payload = []
    for team in all_teams_data:
        members_payload = []
        for member in team["members"]:
            target_hours = None
            if sla_unit == "days" and sla_value is not None:
                target_hours = sla_value * member["daily_hours"]

            members_payload.append(
                {
                    "handle": member["handle"],
                    "role": member["role"],
                    "working_hours_utc": member["working_hours_utc"],
                    "member_daily_hours": member["daily_hours"],
                    "target_hours_if_assigned": target_hours,
                }
            )
        teams_payload.append({"name": team["name"], "members": members_payload})

    payload["teams"] = teams_payload
    return payload


def get_fallback_decision(
    priority: str, fallback_map: FallbackMap
) -> Tuple[LlmDecision, str]:
    """
    Gets a deterministic fallback assignment based on priority.

    Args:
        priority (str): The issue priority (e.g., "high", "p1").
        fallback_map (FallbackMap): The pre-calculated fallback map.

    Returns:
        Tuple[LlmDecision, str]: A tuple containing the
        LlmDecision object and the assignment method ("fallback").
    """
    priority_low = (priority or "").lower()
    if priority_low in {"high", "p1"}:
        fb = fallback_map["lead"]
        reason = "fallback deterministic: lead-for-high"
    else:
        fb = fallback_map["reviewer"]
        reason = "fallback deterministic: earliest-reviewer"

    decision = LlmDecision(
        team_name=fb["team_name"],
        role=fb["role"],
        person_handle=fb["handle"],
        reason=reason,
        confidence=0.15,
    )
    return decision, "fallback"


def validate_decision(
    decision: Dict[str, Any], team_map: Dict[str, Dict[str, TeamMember]]
) -> Optional[LlmDecision]:
    """
    Validates the structure and content of an LLM decision.

    Checks for presence of keys, correct types, and valid team/member
    handles and roles.

    Args:
        decision (Dict[str, Any]): The raw decision object from the LLM.
        team_map (Dict[str, Dict[str, TeamMember]]): A lookup map of
            team_name -> handle -> member_data for fast validation.

    Returns:
        Optional[LlmDecision]: A type-casted LlmDecision if valid,
        otherwise None.
    """
    try:
        # Basic structure validation
        if not all(
            k in decision
            for k in [
                "team_name",
                "role",
                "person_handle",
                "reason",
                "confidence",
            ]
        ):
            log.warning(f"Validation fail: Missing keys. Got: {decision}")
            return None

        team_name = str(decision["team_name"])
        role = str(decision["role"])
        handle = str(decision["person_handle"])
        confidence = float(decision["confidence"])

        # Content validation
        if team_name not in team_map:
            log.warning(f"Validation fail: Team '{team_name}' not found.")
            return None

        if handle not in team_map[team_name]:
            log.warning(
                f"Validation fail: Handle '{handle}' not in team '{team_name}'."
            )
            return None

        member = team_map[team_name][handle]
        if member["role"] != role:
            log.warning(
                f"Validation fail: Role mismatch for '{handle}'. "
                f"LLM said '{role}', config says '{member['role']}'."
            )
            return None

        if not (0.0 <= confidence <= 1.0):
            log.warning(
                f"Validation fail: Confidence {confidence} not in [0.0, 1.0]."
            )
            confidence = max(0.0, min(1.0, confidence))  # Clamp it

        return LlmDecision(
            team_name=team_name,
            role=role,
            person_handle=handle,
            reason=str(decision["reason"])[:200],  # Truncate reason
            confidence=confidence,
        )

    except (ValueError, TypeError, KeyError) as e:
        log.warning(f"Validation fail: Exception processing decision '{decision}': {e}")
        return None


def update_ddb_item(
    ddb_table: Any,
    issue_key: Dict[str, str],
    decision: LlmDecision,
    method: str,
    dry_run: bool,
    force: bool,
) -> bool:
    """
    Writes the assignment decision back to DynamoDB.

    Uses an idempotent, conditional update unless `force` is true.

    Args:
        ddb_table (Any): The boto3 DynamoDB Table resource.
        issue_key (Dict[str, str]): The PK/SK key for the item
            (e.g., {"repo_name": "...", "issue_id": "..."}).
        decision (LlmDecision): The validated decision object.
        method (str): The assignment method ("llm" or "fallback").
        dry_run (bool): If True, skips the DDB write and only logs.
        force (bool): If True, overwrites an existing assignment.

    Returns:
        bool: True if the update was successful (or in dry-run),
        False if it was skipped (idempotency) or failed.
    """
    now_iso = datetime.now(timezone.utc).isoformat()

    log_prefix = "[DRYRUN]" if dry_run else "[UPDATE]"
    log.info(
        f"{log_prefix} ASSIGN repo={issue_key['repo_name']} "
        f"id={issue_key['issue_id']} team={decision['team_name']} "
        f"handle={decision['person_handle']} role={decision['role']} "
        f"method={method} conf={decision['confidence']:.2f}"
    )

    if dry_run:
        return True

    # Idempotency: only update if assignee_handle does not exist,
    # unless --force is used.
    condition = Attr("assignee_handle").not_exists()
    if force:
        condition = None  # No condition, will overwrite

    try:
        # Use Decimal for number types in DynamoDB
        confidence_decimal = Decimal(str(decision["confidence"]))

        update_args = {
            "Key": issue_key,
            "UpdateExpression": (
                "SET assigned_team = :team, assigned_role = :role, "
                "assignee_handle = :handle, assignment_reason = :reason, "
                "assignment_confidence = :conf, assigned_at = :now"
            ),
            "ExpressionAttributeValues": {
                ":team": decision["team_name"],
                ":role": decision["role"],
                ":handle": decision["person_handle"],
                ":reason": decision["reason"],
                ":conf": confidence_decimal,
                ":now": now_iso,
            },
        }
        if condition:
            update_args["ConditionExpression"] = condition

        ddb_table.update_item(**update_args)
        return True

    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            log.warning(
                f"SKIP (idempotent) repo={issue_key['repo_name']} "
                f"id={issue_key['issue_id']}. Item is already assigned."
            )
            return False  # Skipped, not updated
        else:
            log.error(
                f"ERROR updating DDB for {issue_key['issue_id']}: {e}"
            )
            return False  # Error
    except Exception as e:
        log.error(
            f"ERROR (unhandled) updating DDB for {issue_key['issue_id']}: {e}"
        )
        return False  # Error


# --- Main Orchestration ---


def scan_unassigned_issues(
    ddb_table: Any, repo: Optional[str], limit: Optional[int]
) -> Generator[Issue, None, None]:
    """
    Scans DynamoDB for open, unassigned issues.

    Uses a FilterExpression for `attribute_not_exists(assignee_handle)`.
    Supports filtering by a single repo and limiting total items.

    Args:
        ddb_table (Any): The boto3 DynamoDB Table resource.
        repo (Optional[str]): If provided, only scan this repo (PK).
        limit (Optional[int]): If provided, stop scanning after
            this many items.

    Yields:
        Generator[Issue, None, None]: Yields one issue dictionary
        at a time.
    """
    scan_args = {
        "FilterExpression": Attr("assignee_handle").not_exists(),
    }

    if repo:
        log.info(f"Scanning for unassigned issues in repo: {repo}")
        scan_args["FilterExpression"] = scan_args["FilterExpression"] & Attr(
            "repo_name"
        ).eq(repo)
    else:
        log.info("Scanning all repos for unassigned issues...")

    item_count = 0
    pagination_key = None
    while True:
        if pagination_key:
            scan_args["ExclusiveStartKey"] = pagination_key

        try:
            response = ddb_table.scan(**scan_args)
        except ClientError as e:
            log.error(f"Failed to scan DynamoDB table: {e}")
            break

        items = response.get("Items", [])
        for item in items:
            if limit is not None and item_count >= limit:
                log.info(f"Reached scan limit of {limit} items.")
                return  # Stop the generator

            # Basic validation of required fields for processing
            if not all(
                k in item
                for k in [
                    "repo_name",
                    "issue_id",
                    "title",
                    "classification",
                    "priority",
                ]
            ):
                log.warning(
                    f"Skipping item {item.get('repo_name')}/"
                    f"{item.get('issue_id')} due to missing required fields."
                )
                continue

            yield item
            item_count += 1

        pagination_key = response.get("LastEvaluatedKey")
        if not pagination_key:
            break  # No more items


def process_issues(
    ddb_table: Any,
    all_teams_data: List[Team],
    fallback_map: FallbackMap,
    cli_args: argparse.Namespace,
) -> RunStats:
    """
    Main orchestration function.

    Scans for issues, builds batches, calls LLM, validates, falls back,
    and updates DynamoDB.

    Args:
        ddb_table (Any): The boto3 DynamoDB Table resource.
        all_teams_data (List[Team]): The loaded team configuration.
        fallback_map (FallbackMap): The pre-calculated fallback map.
        cli_args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        RunStats: A dictionary containing counts of all operations.
    """
    stats = RunStats(
        scanned=0,
        batched=0,
        assigned_llm=0,
        assigned_fallback=0,
        updated=0,
        skipped_idempotent=0,
        errors=0,
    )

    # Create a quick-lookup map for validation
    # team_name -> handle -> member_data
    team_map: Dict[str, Dict[str, TeamMember]] = {
        team["name"]: {member["handle"]: member for member in team["members"]}
        for team in all_teams_data
    }

    issue_batch: List[Issue] = []
    llm_payload_batch: Dict[str, Any] = {}

    try:
        for issue in scan_unassigned_issues(
            ddb_table, cli_args.repo, cli_args.limit
        ):
            stats["scanned"] += 1
            issue_batch.append(issue)
            batch_key = f"{issue['repo_name']}#{issue['issue_id']}"
            llm_payload_batch[batch_key] = build_llm_payload(
                issue, all_teams_data
            )

            if len(issue_batch) >= cli_args.batch_size:
                stats["batched"] += 1
                log.info(
                    f"Processing batch {stats['batched']} "
                    f"(size {len(issue_batch)})..."
                )
                # Process the full batch
                process_batch(
                    issue_batch,
                    llm_payload_batch,
                    ddb_table,
                    team_map,
                    fallback_map,
                    stats,
                    cli_args,
                )
                # Reset batches
                issue_batch = []
                llm_payload_batch = {}

        # Process any remaining partial batch
        if issue_batch:
            stats["batched"] += 1
            log.info(
                f"Processing final batch {stats['batched']} "
                f"(size {len(issue_batch)})..."
            )
            process_batch(
                issue_batch,
                llm_payload_batch,
                ddb_table,
                team_map,
                fallback_map,
                stats,
                cli_args,
            )

    except Exception as e:
        log.critical(f"Unhandled error in main processing loop: {e}")
        stats["errors"] += len(issue_batch) or 1  # Penalize current batch

    return stats


def process_batch(
    issue_batch: List[Issue],
    llm_payload_batch: Dict[str, Any],
    ddb_table: Any,
    team_map: Dict[str, Dict[str, TeamMember]],
    fallback_map: FallbackMap,
    stats: RunStats,
    cli_args: argparse.Namespace,
) -> None:
    """
    Processes a single batch of issues against the LLM.

    Modifies the `stats` dictionary in place.

    Args:
        issue_batch (List[Issue]): List of raw issue objects in the batch.
        llm_payload_batch (Dict[str, Any]): Map of "repo#id" to the
            fully-formed LLM payload.
        ddb_table (Any): The boto3 DynamoDB Table resource.
        team_map (Dict[str, Dict[str, TeamMember]]): Validation lookup map.
        fallback_map (FallbackMap): Pre-calculated fallback assignees.
        stats (RunStats): The global statistics dictionary (modified in-place).
        cli_args (argparse.Namespace): Parsed command-line arguments.
    """
    # 1. Call LLM
    llm_results = invoke_llm_batch(llm_payload_batch, PROMPT_TEMPLATE)

    # 2. Iterate issues, validate, fallback, and update
    for issue in issue_batch:
        batch_key = f"{issue['repo_name']}#{issue['issue_id']}"
        decision: Optional[LlmDecision] = None
        method = "llm"

        # 3. Validate LLM decision
        if batch_key in llm_results:
            decision = validate_decision(llm_results[batch_key], team_map)

        # 4. Fallback if needed
        if not decision:
            method = "fallback"
            stats["assigned_fallback"] += 1
            decision, method = get_fallback_decision(
                issue["priority"], fallback_map
            )
        else:
            stats["assigned_llm"] += 1

        # 5. Update DynamoDB
        issue_key = {
            "repo_name": issue["repo_name"],
            "issue_id": issue["issue_id"],
        }
        try:
            updated = update_ddb_item(
                ddb_table,
                issue_key,
                decision,
                method,
                cli_args.dry_run,
                cli_args.force,
            )
            if updated:
                stats["updated"] += 1
            else:
                # Not updated (e.g., idempotent skip)
                stats["skipped_idempotent"] += 1

        except Exception as e:
            log.error(
                f"ERROR: Failed to update DDB for {batch_key}: {e}",
                exc_info=True,
            )
            stats["errors"] += 1


def main() -> None:
    """
    Main script entrypoint.

    Parses CLI args, initializes clients, loads config, and
    starts the issue processing loop.
    """
    parser = argparse.ArgumentParser(
        description="Assign DynamoDB issues using Bedrock LLM.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=DEFAULT_DRY_RUN,
        help="Log actions without writing to DynamoDB. "
        "(Default: env DRY_RUN or False)",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=os.environ.get("REPO"),
        help="Only process issues for this specific repo (e.g., 'org/name'). "
        "(Default: env REPO or all repos)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.environ.get("LIMIT", "0")) or None,
        help="Stop after processing this many items. "
        "(Default: env LIMIT or unlimited)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of issues to send to LLM in one batch. "
        f"(Default: env BATCH_SIZE or {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force update DynamoDB items even if 'assignee_handle' "
        "already exists.",
    )
    args = parser.parse_args()

    # --- Initializations ---
    log.info("--- Starting LLM Assignee Worker ---")
    log.info(
        f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}"
        f"{' (FORCED)' if args.force else ''}"
    )

    if not AWS_REGION:
        log.critical("FATAL: AWS_REGION environment variable is not set.")
        sys.exit(1)
    if not BEDROCK_REGION:
        log.critical("FATAL: BEDROCK_REGION environment variable is not set.")
        sys.exit(1)

    # 1. Load Team Config (and calculate fallbacks)
    try:
        all_teams_data, fallback_map = load_team_config(TEAM_CONFIG_PATH)
    except Exception as e:
        log.critical(f"Failed to load configuration: {e}", exc_info=True)
        sys.exit(1)

    # 2. Initialize AWS Clients
    global bedrock_client
    try:
        dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
        ddb_table = dynamodb.Table(DDB_TABLE)
        ddb_table.load()  # Check for table existence
        log.info(f"Connected to DynamoDB table: {DDB_TABLE}")

        bedrock_client = boto3.client(
            "bedrock-runtime", region_name=BEDROCK_REGION
        )
        log.info(f"Connected to Bedrock runtime in: {BEDROCK_REGION}")
    except ClientError as e:
        log.critical(
            f"FATAL: Failed to initialize AWS clients. Check credentials/region. {e}"
        )
        sys.exit(1)
    except Exception as e:
        log.critical(f"FATAL: Unexpected error initializing: {e}")
        sys.exit(1)

    # 3. Run processing
    start_time = time.monotonic()
    stats = process_issues(ddb_table, all_teams_data, fallback_map, args)
    elapsed = time.monotonic() - start_time

    # 4. Print Summary
    summary = {
        "status": "success" if stats["errors"] == 0 else "partial_failure",
        "dry_run": args.dry_run,
        "force": args.force,
        "duration_seconds": round(elapsed, 2),
        "config": {
            "repo_filter": args.repo,
            "limit": args.limit,
            "batch_size": args.batch_size,
            "llm_model_id": LLM_MODEL_ID,
        },
        "stats": stats,
    }
    log.info("--- Worker Run Summary ---")
    print(json.dumps(summary, indent=2))

    if stats["errors"] > 0:
        log.error("Run finished with errors.")
        sys.exit(1)
    log.info("Run finished successfully.")


if __name__ == "__main__":
    main()