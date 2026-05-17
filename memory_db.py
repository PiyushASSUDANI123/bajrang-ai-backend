import os
import time
import json
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Founder Identity
FOUNDER_ID = os.getenv("FOUNDER_USER_ID", "piyush_ceo")

# Firebase Admin Init (Secure JSON String loading)
firebase_raw_data = os.getenv("FIREBASE_JSON")

if firebase_raw_data:
    try:
        cred_dict = json.loads(firebase_raw_data)
        cred = credentials.Certificate(cred_dict)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        print("✅ Firebase securely loaded from Secrets!")
        db = firestore.client()
    except Exception as e:
        print(f"❌ Failed to initialize Firebase Admin: {e}")
        db = None
else:
    print("❌ FIREBASE_JSON secret is missing!")
    db = None

def save_interaction(user_id, message, response, intent="CHIT_CHAT"):
    """Save Q&A to Firestore using Admin SDK."""
    if not db:
        return
    try:
        doc_ref = db.collection("chats").document()
        doc_ref.set({
            "user_id": str(user_id),
            "user_message": str(message),
            "ai_response": str(response),
            "intent": str(intent),
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        print(f"✅ Firebase saved [user: {user_id}]")
    except Exception as e:
        print(f"⚠️ Firebase save error: {e}")

def get_recent_context(user_id, limit=5):
    """Fetch last N messages for context."""
    if not db:
        return ""
    try:
        chats_ref = db.collection("chats")
        query = chats_ref.where("user_id", "==", str(user_id)).order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit)
        results = query.stream()
        
        context_parts = []
        for doc in results:
            data = doc.to_dict()
            q = data.get("user_message", "")
            a = data.get("ai_response", "")
            context_parts.append(f"User: {q}\nAI: {a}")
            
        return "\n---\n".join(reversed(context_parts))
    except Exception as e:
        print(f"⚠️ Firebase fetch error: {e}")
        return ""

def add_to_knowledge(content, source="manual"):
    """Store permanent knowledge in Firestore."""
    if not db:
        return
    try:
        doc_ref = db.collection("knowledge_base").document()
        doc_ref.set({
            "content": str(content),
            "source": str(source),
            "added_at": firestore.SERVER_TIMESTAMP
        })
    except Exception as e:
        print(f"⚠️ Firebase KB error: {e}")

def save_feedback(user_id, chat_id, feedback_type, feedback_text, last_user_msg, last_ai_msg):
    """Store user feedback."""
    if not db:
        return
    try:
        doc_ref = db.collection("feedbacks").document()
        doc_ref.set({
            "user_id": str(user_id),
            "chat_id": str(chat_id),
            "feedback_type": str(feedback_type),
            "feedback_text": str(feedback_text),
            "last_user_message": str(last_user_msg),
            "last_ai_message": str(last_ai_msg),
            "timestamp": firestore.SERVER_TIMESTAMP
        })
    except Exception as e:
        print(f"⚠️ Firebase feedback save error: {e}")

def save_shared_chat(messages: list, title: str) -> str:
    """Save full chat history for public link sharing and return doc ID."""
    if not db:
        raise ValueError("Firebase is not initialized")
    try:
        # Strip oversized base64 components if necessary or store raw
        doc_ref = db.collection("shared_chats").document()
        doc_ref.set({
            "title": str(title),
            "messages": messages,
            "shared_at": firestore.SERVER_TIMESTAMP
        })
        return doc_ref.id
    except Exception as e:
        print(f"⚠️ Firebase save shared chat error: {e}")
        raise e

def get_shared_chat(shared_id: str) -> dict:
    """Fetch shared chat history by unique ID."""
    if not db:
        raise ValueError("Firebase is not initialized")
    try:
        doc_ref = db.collection("shared_chats").document(str(shared_id))
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            # Convert timestamp to ISO string for API compatibility
            if "shared_at" in data and data["shared_at"]:
                data["shared_at"] = data["shared_at"].isoformat()
            return data
        return None
    except Exception as e:
        print(f"⚠️ Firebase fetch shared chat error: {e}")
        raise e