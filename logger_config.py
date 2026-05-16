import datetime
import json
import os

LOG_FILE = "system_logs.txt"

def log_event(event_type, query, details, execution_time=0):
    """
    event_type: 'ROUTER', 'SCRAPER', 'MEMORY', 'SYSTEM'
    details: Dict ya string jisme result/error ho
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = {
        "time": timestamp,
        "type": event_type,
        "query": query,
        "details": details,
        "latency_sec": round(execution_time, 2)
    }
    
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def get_stats():
    """Ek summary nikalne ke liye (Optional)"""
    if not os.path.exists(LOG_FILE):
        return "No logs found."
    
    with open(LOG_FILE, "r") as f:
        lines = f.readlines()
        return f"Total Interactions logged: {len(lines)}"