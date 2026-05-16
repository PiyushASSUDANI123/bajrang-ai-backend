from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from master import stream_bajrang_response
from image_analyzer import analyze_image_stream, encode_file_to_base64
import requests
import asyncio
import time
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="BAJRANG AI CORE")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    history: list = []
    user_id: str = "guest"
    image_data: str = ""   # base64 data URI or image URL (optional)
    mode: str = "default"  # active AI mode


def save_to_firebase_bg(user_message, ai_response, intent, user_id):
    """Fire-and-forget Firebase save — runs in thread, never blocks API."""
    api_key = os.getenv("FIREBASE_API_KEY")
    project_id = os.getenv("FIREBASE_PROJECT_ID")
    url = f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/(default)/documents/chats?key={api_key}"

    data = {
        "fields": {
            "user_id":      {"stringValue": str(user_id)},
            "user_message": {"stringValue": str(user_message)},
            "ai_response":  {"stringValue": str(ai_response)},
            "intent":       {"stringValue": str(intent)},
            "timestamp":    {"stringValue": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        }
    }

    try:
        response = requests.post(url, json=data, timeout=5)
        if response.status_code == 200:
            print(f"✅ Firebase saved [user: {user_id}]")
        else:
            print(f"⚠️ Firebase {response.status_code}: {response.text[:100]}")
    except Exception as e:
        print(f"⚠️ Firebase error: {e}")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    print(f"📥 [REQUEST] Mode: {request.mode} | User: {request.user_id} | Msg: {request.message[:50]}...")
    accumulated = []

    async def event_generator():
        # ── Vision path: image attached ───────────────────────────────────
        if request.image_data:
            is_founder = request.user_id == os.getenv("FOUNDER_USER_ID", "piyush_ceo")
            async for chunk in analyze_image_stream(
                image_data=request.image_data,
                user_question=request.message,
                conversation_history=request.history,
                is_founder=is_founder
            ):
                if chunk != "data: [DONE]\n\n":
                    token = chunk[6:].strip()
                    accumulated.append(token)
                yield chunk

            full_response = "".join(accumulated).replace('\\n', '\n')
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, save_to_firebase_bg,
                request.message, full_response, "VISION", request.user_id)
            return

        # ── Text path: normal chat ──────────────────────────────────────
        async for chunk in stream_bajrang_response(
            request.message,
            request.history,
            user_id=request.user_id,
            mode=request.mode
        ):
            if chunk != "data: [DONE]\n\n":
                # Collect for Firebase save
                token = chunk[6:].strip()
                accumulated.append(token)
            yield chunk

        # After stream done, save to Firebase in background thread
        full_response = "".join(accumulated).replace('\\n', '\n')
        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            None,
            save_to_firebase_bg,
            request.message, full_response, "STREAMED", request.user_id
        )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Nginx buffering disable
        }
    )


# Legacy non-streaming endpoint (fallback)
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        full_text = ""
        async for chunk in stream_bajrang_response(
            request.message, 
            request.history, 
            request.user_id,
            mode=request.mode
        ):
            if chunk.startswith("data: ") and "[DONE]" not in chunk:
                full_text += chunk[6:].replace('\\n', '\n')

        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, save_to_firebase_bg, request.message, full_text, "CHAT", request.user_id)
        return {"response": full_text, "intent": "CHAT"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


_start_time = time.time()

@app.get("/health")
async def health():
    uptime_s = int(time.time() - _start_time)
    h, m, s = uptime_s // 3600, (uptime_s % 3600) // 60, uptime_s % 60
    return {
        "status":  "online",
        "engine":  "BAJRANG AI",
        "version": "3.0-groq-streaming",
        "model":   "llama-3.3-70b-versatile",
        "vision":  "llama-4-scout-17b",
        "uptime":  f"{h:02d}:{m:02d}:{s:02d}",
        "features": ["streaming", "web_search", "agents", "rag", "vision"]
    }


@app.post("/analyze/image")
async def analyze_image_upload(
    file: UploadFile = File(...),
    question: str = Form(default=""),
    user_id: str = Form(default="guest")
):
    """
    Multipart image upload endpoint.
    Frontend can POST an actual image file here.
    """
    content_type = file.content_type or "image/jpeg"
    supported = {"image/jpeg", "image/png", "image/gif", "image/webp"}

    if content_type not in supported:
        raise HTTPException(status_code=400,
            detail=f"Unsupported type '{content_type}'. Use JPG/PNG/GIF/WebP.")

    raw_bytes = await file.read()
    if len(raw_bytes) > 20 * 1024 * 1024:  # 20MB cap
        raise HTTPException(status_code=413, detail="Image too large. Max 20MB.")

    data_uri = encode_file_to_base64(raw_bytes, content_type)
    is_founder = user_id == os.getenv("FOUNDER_USER_ID", "piyush_ceo")

    async def stream():
        async for chunk in analyze_image_stream(
            image_data=data_uri,
            user_question=question or "Analyze this image in detail.",
            is_founder=is_founder
        ):
            yield chunk

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.get("/stats")
async def stats():
    return {
        "engine":      "BAJRANG AI v3.0",
        "provider":    "Groq Cloud",
        "model":       "llama-3.1-8b-instant",
        "capabilities": {
            "streaming":   True,
            "web_search":  True,
            "agents":      True,
            "rag":         True,
            "code_runner": True,
        },
        "uptime_seconds": int(time.time() - _start_time)
    }


if __name__ == "__main__":
    import uvicorn
    import os
    # Render port environment variable se uthayega, nahi toh default 8000
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)