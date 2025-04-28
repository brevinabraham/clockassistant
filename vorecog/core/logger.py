import json
from vorecog.configs.config import LOG_PATH

def save_log(entry):
    if LOG_PATH.exists():
        with open(LOG_PATH, "r") as f:
            logs = json.load(f)
    else:
        logs = []
    logs.append(entry)
    if len(logs) > 500:
        logs = logs[-500:]
    json.dump(logs, open(LOG_PATH, "w"), indent=2)
