import os
import json
from datetime import datetime


LOGS_DIR = "logs"
LOG_FILE = os.path.join(LOGS_DIR, "system_logs.json")


# Create logs folder if it does not exist
os.makedirs(LOGS_DIR, exist_ok=True)


def log_event(agent, action, result=None, error=None):
    """
    Save a structured JSON log entry.
    """

    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent": agent,
        "action": action,
        "result": result,
        "error": error
    }

    # If file does not exist → create empty list
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            json.dump([], f)

    # Read existing logs
    with open(LOG_FILE, "r") as f:
        logs = json.load(f)

    # Add new log
    logs.append(log_entry)

    # Save updated logs
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)

    print(f"[LOG] {agent} → {action}")