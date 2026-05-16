import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# Firebase Config
API_KEY    = os.getenv("FIREBASE_API_KEY")
PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID")
FOUNDER_ID = os.getenv("FOUNDER_USER_ID", "piyush_ceo")

BASE_URL = f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}/databases/(default)/documents"

def save_interaction(user_id, message, response, intent="CHIT_CHAT"):
    """Save Q&A to Firestore."""
    url = f"{BASE_URL}/chats?key={API_KEY}"
    
    data = {
        "fields": {
            "user_id":      {"stringValue": str(user_id)},
            "user_message": {"stringValue": str(message)},
            "ai_response":  {"stringValue": str(response)},
            "intent":       {"stringValue": str(intent)},
            "timestamp":    {"timestampValue": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        }
    }
    
    try:
        requests.post(url, json=data, timeout=5)
    except Exception as e:
        print(f"⚠️ Firebase save error: {e}")

def get_recent_context(user_id, limit=5):
    """Fetch last N messages for context (Replacement for semantic search)."""
    url = f"{BASE_URL}:runQuery?key={API_KEY}"
    
    query = {
        "structuredQuery": {
            "from": [{"collectionId": "chats"}],
            "where": {
                "fieldFilter": {
                    "field": {"fieldPath": "user_id"},
                    "op": "EQUAL",
                    "value": {"stringValue": user_id}
                }
            },
            "orderBy": [
                {"field": {"fieldPath": "timestamp"}, "direction": "DESCENDING"}
            ],
            "limit": limit
        }
    }
    
    try:
        resp = requests.post(url, json=query, timeout=5)
        if resp.status_code != 200:
            return ""
        
        results = resp.json()
        context_parts = []
        # Firestore runQuery returns a list of results
        for res in results:
            if "document" in res:
                fields = res["document"]["fields"]
                q = fields["user_message"]["stringValue"]
                a = fields["ai_response"]["stringValue"]
                context_parts.append(f"User: {q}\nAI: {a}")
        
        return "\n---\n".join(reversed(context_parts))
    except Exception as e:
        print(f"⚠️ Firebase fetch error: {e}")
        return ""

def add_to_knowledge(content, source="manual"):
    """Store permanent knowledge in Firestore."""
    url = f"{BASE_URL}/knowledge_base?key={API_KEY}"
    data = {
        "fields": {
            "content": {"stringValue": content},
            "source":  {"stringValue": source},
            "added_at": {"timestampValue": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        }
    }
    try:
        requests.post(url, json=data, timeout=5)
    except Exception as e:
        print(f"⚠️ Firebase KB error: {e}")