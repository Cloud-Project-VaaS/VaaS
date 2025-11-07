#!/usr/bin/env python3

"""
sla_breach_checker.py

A production-ready Python worker to find and process SLA breaches in a DynamoDB table.

This script queries a DynamoDB table (via a GSI) for items whose `due_at`
timestamp is in the past. It then updates those items to record the breach
and, if necessary, re-assigns them.

It supports two entrypoints:
1.  `lambda_handler(event, context)`: For deployment as an AWS Lambda function.
2.  `main()`: For local execution or running via a cron job.

---
🚀 How to Run Locally
---

1.  **Set Environment Variables:**
    ```bash
    # Required:
    export AWS_REGION="us-east-1"
    export DDB_TABLE="IssuesTrackingTable"

    # Option 1: Per-Repo Query (Recommended)
    export REPOS="my-org/repo-one,my-org/repo-two"
    export GSI_NAME="repo_due_at" # GSI with PK=repo_name, SK=due_at

    # Option 2: Global Query (Alternative)
    # (Do not set REPOS if using this)
    # export GLOBAL_GSI_NAME="open_issues_by_due_date"
    # export GLOBAL_GSI_PK_NAME="open_status"
    # export GLOBAL_GSI_PK_VALUE="OPEN"

    # Optional:
    export DRY_RUN="1" # Set to "1" to log actions without writing to DDB
    ```

2.  **Install Dependencies:**
    ```bash
    pip install boto3
    ```

3.  **Run the script:**
    ```bash
    python sla_breach_checker.py
    ```

4.  **Run with CLI overrides:**
    ```bash
    # Override env vars with CLI flags
    python sla_breach_checker.py --repos "my-org/cli-repo" --dry-run
    ```

---
📦 How to Deploy to AWS Lambda
---

1.  **Create Lambda Function:**
    * Create a new Lambda function using the Python 3.10 (or newer) runtime.
    * Upload this `.py` file (or include it in your deployment package).
    * Set the Handler to `sla_breach_checker.lambda_handler`.

2.  **Set Environment Variables (in Lambda Configuration):**
    * `AWS_REGION`: e.g., `us-east-1`
    * `DDB_TABLE`: e.g., `IssuesTrackingTable`
    * `REPOS`: e.g., `my-org/repo-one,my-org/repo-two`
    * `GSI_NAME`: e.g., `repo_due_at`
    * (Or set `GLOBAL_GSI_...` variables instead of `REPOS`/`GSI_NAME`)

3.  **Configure IAM Role:**
    The Lambda's execution role must have the following permissions:
    ```json
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "dynamodb:Query",
                    "dynamodb:UpdateItem"
                ],
                "Resource": [
                    "arn:aws:dynamodb:REGION:ACCOUNT_ID:table/IssuesTrackingTable",
                    "arn:aws:dynamodb:REGION:ACCOUNT_ID:table/IssuesTrackingTable/index/repo_due_at",
                    "arn:aws:dynamodb:REGION:ACCOUNT_ID:table/IssuesTrackingTable/index/GLOBAL_GSI_NAME"
                ]
            }
        ]
    }
    ```
    *(Adjust ARNs and index names as needed)*

4.  **Set up Scheduler (Amazon EventBridge):**
    * Create a new EventBridge (or CloudWatch Events) rule.
    * Select "Schedule".
    * Set the schedule expression: `rate(30 minutes)`
    * Set the Target to your new Lambda function.
"""

import os
import sys
import json
import logging
import time
import math
import argparse
from datetime import datetime, timezone
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Generator,
    TypedDict,
    Set,
    Tuple,
)

# Third-party imports
try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print(
        "Error: 'boto3' library not found. Please install it: pip install boto3",
        file=sys.stderr,
    )
    sys.exit(1)

# --- Configuration ---

# Set up logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
if not log.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    log.addHandler(handler)


class Config(TypedDict):
    """Strongly-typed configuration dictionary."""

    aws_region: str
    ddb_table: str
    dry_run: bool
    # Per-repo query config
    repos: List[str]
    gsi_name: str
    # Global query config
    global_gsi_name: str
    global_gsi_pk_name: str
    global_gsi_pk_value: str


# --- DynamoDB & Retry Logic ---

MAX_RETRIES = 3
TRANSIENT_ERRORS = {"ProvisionedThroughputExceededException", "ThrottlingException"}


def _ddb_retry_wrapper(func: Any) -> Any:
    """Decorator to handle transient DynamoDB errors with exponential backoff."""

    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code")
                if error_code in TRANSIENT_ERRORS:
                    if attempt < MAX_RETRIES - 1:
                        sleep_time = (2**attempt) * 0.5  # 0.5s, 1s, 2s
                        log.warning(
                            "Transient DDB error '%s'. Retrying in %.1fs... "
                            "(Attempt %d/%d)",
                            error_code,
                            sleep_time,
                            attempt + 1,
                            MAX_RETRIES,
                        )
                        time.sleep(sleep_time)
                    else:
                        log.error(
                            "Failed after %d attempts due to '%s'.",
                            MAX_RETRIES,
                            error_code,
                        )
                        raise
                else:
                    # Not a transient error, re-raise immediately
                    raise
        return None  # Should be unreachable

    return wrapper


@_ddb_retry_wrapper
def ddb_query(table: Any, **kwargs) -> Dict[str, Any]:
    """Retriable boto3 table.query()."""
    return table.query(**kwargs)


@_ddb_retry_wrapper
def ddb_update_item(table: Any, **kwargs) -> Dict[str, Any]:
    """Retriable boto3 table.update_item()."""
    return table.update_item(**kwargs)


# --- Core Processing Logic ---


def get_config(cli_args: Optional[argparse.Namespace] = None) -> Config:
    """
    Fetches and validates configuration from environment variables and
    optional CLI arguments.
    """
    config: Config = {
        "aws_region": os.environ.get("AWS_REGION"),
        "ddb_table": os.environ.get("DDB_TABLE", "IssuesTrackingTable"),
        "gsi_name": os.environ.get("GSI_NAME", "repo_due_at"),
        "dry_run": os.environ.get("DRY_RUN", "0") == "1",
        "repos": os.environ.get("REPOS", "").split(","),
        "global_gsi_name": os.environ.get("GLOBAL_GSI_NAME", ""),
        "global_gsi_pk_name": os.environ.get("GLOBAL_GSI_PK_NAME", ""),
        "global_gsi_pk_value": os.environ.get("GLOBAL_GSI_PK_VALUE", ""),
    }

    # Clean up empty repo list
    config["repos"] = [r.strip() for r in config["repos"] if r.strip()]

    # Apply CLI overrides if provided
    if cli_args:
        if cli_args.dry_run:
            config["dry_run"] = True
        if cli_args.repos:
            config["repos"] = [r.strip() for r in cli_args.repos.split(",") if r.strip()]

    # --- Validation ---
    if not config["aws_region"]:
        log.error("Missing required environment variable: AWS_REGION")
        sys.exit(1)

    has_repos = bool(config["repos"] and config["gsi_name"])
    has_global_gsi = bool(
        config["global_gsi_name"]
        and config["global_gsi_pk_name"]
        and config["global_gsi_pk_value"]
    )

    if not has_repos and not has_global_gsi:
        log.error(
            "Configuration error: Must provide either REPOS and GSI_NAME, "
            "or GLOBAL_GSI_NAME, GLOBAL_GSI_PK_NAME, and GLOBAL_GSI_PK_VALUE "
            "to avoid a full table scan."
        )
        sys.exit(1)

    if has_repos and has_global_gsi:
        log.warning(
            "Both REPOS and GLOBAL_GSI... settings found. "
            "Prioritizing REPOS query."
        )
        config["global_gsi_name"] = ""  # Disable global query

    log.info("Configuration loaded. Dry run: %s", config["dry_run"])
    if has_repos:
        log.info(
            "Query mode: Per-repo (GSI: %s) for %d repo(s).",
            config["gsi_name"],
            len(config["repos"]),
        )
    else:
        log.info(
            "Query mode: Global (GSI: %s, PK: %s=%s).",
            config["global_gsi_name"],
            config["global_gsi_pk_name"],
            config["global_gsi_pk_value"],
        )

    return config


def format_breach_message(item: Dict[str, Any]) -> str:
    """
    Composes the breach message, using a template if available or a fallback.
    """
    template = item.get("breach_template") or item.get("escalation_message")
    placeholders = {
        "sla_id": item.get("sla_id", "unknown_sla"),
        "issue_id": item.get("issue_id", "unknown_id"),
        "classification": item.get("classification", "unknown"),
        "priority": item.get("priority", "unknown"),
        "due_at": item.get("due_at", "unknown_due_at"),
        "current_assignee": item.get("current_assignee", "unknown_assignee"),
    }

    if template:
        try:
            # Simple, safe interpolation
            msg = template
            for key, value in placeholders.items():
                msg = msg.replace(f"{{{key}}}", str(value))
            return msg
        except Exception as e:
            log.warning(
                "Failed to interpolate breach_template for issue %s: %s",
                placeholders["issue_id"],
                e,
            )
            # Fall through to default message

    # Fallback message
    return (
        f"SLA breached for {placeholders['sla_id']} on "
        f"{placeholders['classification']} {placeholders['issue_id']} "
        f"(priority {placeholders['priority']}). "
        f"due_at={placeholders['due_at']}."
    )


def query_breached_items(
    config: Config, ddb_table: Any, now_iso: str
) -> Generator[Dict[str, Any], None, None]:
    """
    Queries DynamoDB for all breached items and yields them one by one.
    Handles both per-repo and global GSI query logic and pagination.
    """
    query_targets: List[Dict[str, Any]] = []

    if config["repos"]:
        # --- Per-Repo Query Mode ---
        for repo in config["repos"]:
            query_targets.append(
                {
                    "IndexName": config["gsi_name"],
                    "KeyConditionExpression": "#pk = :pk_val AND #sk <= :sk_val",
                    "ExpressionAttributeNames": {"#pk": "repo_name", "#sk": "due_at"},
                    "ExpressionAttributeValues": {
                        ":pk_val": repo,
                        ":sk_val": now_iso,
                    },
                }
            )
    else:
        # --- Global Query Mode ---
        query_targets.append(
            {
                "IndexName": config["global_gsi_name"],
                "KeyConditionExpression": "#pk = :pk_val AND #sk <= :sk_val",
                "ExpressionAttributeNames": {
                    "#pk": config["global_gsi_pk_name"],
                    "#sk": "due_at",
                },
                "ExpressionAttributeValues": {
                    ":pk_val": config["global_gsi_pk_value"],
                    ":sk_val": now_iso,
                },
            }
        )

    # --- Execute Queries ---
    total_scanned = 0
    for query_args in query_targets:
        log.info(
            "Querying GSI '%s' with PK %s...",
            query_args["IndexName"],
            query_args["ExpressionAttributeValues"][":pk_val"],
        )
        pagination_key = "start"
        while pagination_key:
            if pagination_key != "start":
                query_args["ExclusiveStartKey"] = pagination_key

            try:
                response = ddb_query(ddb_table, **query_args)
            except ClientError as e:
                log.error(
                    "Failed to query DynamoDB for %s: %s",
                    query_args["ExpressionAttributeValues"][":pk_val"],
                    e,
                )
                break  # Stop processing this query target

            items = response.get("Items", [])
            total_scanned += response.get("ScannedCount", 0)

            for item in items:
                yield item

            pagination_key = response.get("LastEvaluatedKey")

    log.info("Total items scanned across all queries: %d", total_scanned)


def process_item(
    item: Dict[str, Any], config: Config, ddb_table: Any
) -> Tuple[str, Optional[str]]:
    """
    Processes a single breached item.
    Returns (status, error_message)
    Status: "updated", "skipped", "dryrun", "error"
    """
    # 1. Validate required fields
    required_fields: Set[str] = {
        "repo_name",
        "issue_id",
        "due_at",
        "assignee_role",
        "current_assignee",
    }
    missing_fields = required_fields - set(item.keys())
    if missing_fields:
        msg = f"Item missing required fields: {missing_fields}"
        log.warning(
            "SKIP repo=%s id=%s. %s",
            item.get("repo_name"),
            item.get("issue_id"),
            msg,
        )
        return "skipped", msg

    repo = item["repo_name"]
    issue_id = item["issue_id"]
    role = item["assignee_role"]

    # 2. Build breach message
    breach_message = format_breach_message(item)
    new_assignee = None

    # 3. Determine update action
    update_expression_parts = ["SET #bm = :bm_val"]
    expression_attr_names = {"#bm": "breach_message"}
    expression_attr_values = {":bm_val": breach_message}
    # Conditional write: only update if breach_message is new or different
    condition_expression = "attribute_not_exists(#bm) OR #bm <> :bm_val"

    if role == "reviewer":
        new_assignee = "role:lead"
        update_expression_parts.append("#na = :na_val")
        expression_attr_names["#na"] = "new_assignee"
        expression_attr_values[":na_val"] = new_assignee

    update_expression = " ".join(update_expression_parts)

    # 4. Log and (maybe) execute update
    log_msg = (
        f"BREACH repo={repo} id={issue_id} assignee_role={role} "
        f"new_assignee={new_assignee or '-'} "
        f'msg="{breach_message[:200]}{"..." if len(breach_message) > 200 else ""}"'
    )

    if config["dry_run"]:
        log.info("[DRYRUN] %s", log_msg)
        return "dryrun", None

    try:
        ddb_update_item(
            ddb_table,
            Key={"repo_name": repo, "issue_id": issue_id},
            UpdateExpression=update_expression,
            ConditionExpression=condition_expression,
            ExpressionAttributeNames=expression_attr_names,
            ExpressionAttributeValues=expression_attr_values,
        )
        log.info("[UPDATE] %s", log_msg)
        return "updated", None
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            # This is not an error; it's successful idempotency.
            log.info(
                "SKIP (idempotent) repo=%s id=%s. Breach message already set.",
                repo,
                issue_id,
            )
            return "skipped", "Idempotent skip"
        else:
            msg = f"Failed to update item: {e}"
            log.error("ERROR repo=%s id=%s. %s", repo, issue_id, msg)
            return "error", msg
    except Exception as e:
        msg = f"Unhandled exception during update: {e}"
        log.error("ERROR repo=%s id=%s. %s", repo, issue_id, msg)
        return "error", msg


def process_breaches(config: Config) -> Dict[str, Any]:
    """
    Main business logic: finds and processes all breached items.
    """
    log.info("Starting SLA breach check...")
    start_time = time.monotonic()

    try:
        dynamodb = boto3.resource("dynamodb", region_name=config["aws_region"])
        ddb_table = dynamodb.Table(config["ddb_table"])
        # Check if table exists (lightweight check)
        ddb_table.load()
    except ClientError as e:
        log.error(
            "Failed to connect to DynamoDB table '%s' in region '%s': %s",
            config["ddb_table"],
            config["aws_region"],
            e,
        )
        return {"error": "DynamoDB connection failed", "status": "failed"}

    now_iso = datetime.now(timezone.utc).isoformat()
    counters = {"processed": 0, "updated": 0, "skipped": 0, "errors": 0, "dryrun": 0}

    try:
        for item in query_breached_items(config, ddb_table, now_iso):
            counters["processed"] += 1
            try:
                status, _ = process_item(item, config, ddb_table)
                counters[status] += 1
            except Exception as e:
                # Catch-all for safety in the loop
                log.error(
                    "Unhandled error processing item %s: %s",
                    item.get("issue_id"),
                    e,
                )
                counters["errors"] += 1

    except Exception as e:
        log.critical("Fatal error during item query/iteration: %s", e)
        counters["errors"] += 1

    elapsed = time.monotonic() - start_time
    summary = {
        "status": "success" if counters["errors"] == 0 else "partial_failure",
        "dry_run": config["dry_run"],
        "duration_seconds": round(elapsed, 2),
        "counts": counters,
    }

    log.info(
        "--- Breach Check Summary (%.2fs) ---",
        summary["duration_seconds"],
    )
    log.info("Processed: %d", counters["processed"])
    if config["dry_run"]:
        log.info("Dryrun (would update): %d", counters["dryrun"])
    else:
        log.info("Updated:   %d", counters["updated"])
    log.info("Skipped:   %d", counters["skipped"])
    log.info("Errors:    %d", counters["errors"])
    log.info("------------------------------------")

    return summary


# --- Entrypoints ---


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda entrypoint.
    """
    log.info("Lambda handler invoked (Event: %s)", event.get("source", "Unknown"))
    try:
        config = get_config()
        summary = process_breaches(config)
        return summary
    except Exception as e:
        log.critical("Unhandled exception in lambda_handler: %s", e)
        return {"status": "failed", "error": str(e)}


def main() -> None:
    """
    Local script entrypoint.
    """
    parser = argparse.ArgumentParser(
        description="Run SLA breach checker.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run in dry-run mode. (Overrides DRY_RUN env var)",
    )
    parser.add_argument(
        "--repos",
        type=str,
        help="Comma-separated list of repos. (Overrides REPOS env var)",
    )

    args = parser.parse_args()

    try:
        config = get_config(cli_args=args)
        summary = process_breaches(config)
        if summary.get("status") != "success":
            sys.exit(1)
    except SystemExit:
        # Raised by get_config() on validation error
        sys.exit(1)
    except Exception as e:
        log.critical("Unhandled exception in main: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()