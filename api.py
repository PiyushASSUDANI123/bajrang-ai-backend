from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from master import stream_altair_response, groq_client
from image_analyzer import analyze_image_stream, encode_file_to_base64
from memory_db import save_interaction, save_feedback, save_shared_chat, get_shared_chat, get_all_chats
import requests
import asyncio
import time
import os
from dotenv import load_dotenv
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

# Initialize Rate Limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="TIFLO AI CORE")
app.state.limiter = limiter

# Rate limit exception handler - security first!
@app.exception_handler(RateLimitExceeded)
async def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded):
    print("🚨 [SECURITY RATE LIMIT] Rate limit triggered for remote address:", get_remote_address(request))
    return JSONResponse(
        status_code=429,
        content={"error": "Speed kam kar bhai, Tiflo AI thak raha hai."}
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://tiflo.in",
        "https://www.tiflo.in",
        "https://mancho.pages.dev", 
        "http://localhost:5500", 
        "http://127.0.0.1:5500"
    ],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FeedbackRequest(BaseModel):
    user_id: str
    chat_id: str
    feedback_type: str
    feedback_text: str = ""
    last_user_message: str = ""
    last_ai_message: str = ""

class ShareChatRequest(BaseModel):
    messages: list
    title: str = "Shared Chat"

class ChatRequest(BaseModel):
    message: str
    history: list = []
    user_id: str = "guest"
    user_email: str = ""     # Verified email from Clerk — used for God Mode
    image_data: str = ""   # base64 data URI or image URL (optional)
    mode: str = "default"  # active AI mode
    use_openrouter: bool = False
    is_incognito: bool = False
    location: str = ""


def get_client_ip(request: Request) -> str:
    x_forwarded_for = request.headers.get("x-forwarded-for")
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()
    cf_connecting_ip = request.headers.get("cf-connecting-ip")
    if cf_connecting_ip:
        return cf_connecting_ip.strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else "unknown"


def save_to_firebase_bg(user_message, ai_response, intent, user_id, user_email="", user_ip="", user_location="", model="lite", mode="default"):
    """Fire-and-forget Firebase save — runs in thread, never blocks API."""
    save_interaction(user_id, user_message, ai_response, intent, user_email, user_ip, user_location, model, mode)


def save_feedback_to_firebase_bg(user_id, chat_id, feedback_type, feedback_text, last_user_msg, last_ai_msg):
    save_feedback(user_id, chat_id, feedback_type, feedback_text, last_user_msg, last_ai_msg)


@app.post("/chat/stream")
@limiter.limit("100/minute")
async def chat_stream(request: Request, chat_req: ChatRequest):
    print(f"📥 [REQUEST] Mode: {chat_req.mode} | User: {chat_req.user_id} | Msg: {chat_req.message[:50]}... {'🕵️ [INCOGNITO]' if chat_req.is_incognito else ''}")
    accumulated = []
    client_ip = get_client_ip(request)

    async def event_generator():
        # ── Vision path: image attached ───────────────────────────────────
        if chat_req.image_data:
            is_founder = chat_req.user_id == os.getenv("FOUNDER_USER_ID", "piyush_ceo")
            async for chunk in analyze_image_stream(
                image_data=chat_req.image_data,
                user_question=chat_req.message,
                conversation_history=chat_req.history,
                is_founder=is_founder
            ):
                if chunk != "data: [DONE]\n\n":
                    token = chunk[6:].strip()
                    accumulated.append(token)
                yield chunk

            full_response = "".join(accumulated).replace('\\n', '\n')
            if not chat_req.is_incognito:
                loop = asyncio.get_event_loop()
                loop.run_in_executor(None, save_to_firebase_bg,
                    chat_req.message, full_response, "VISION", chat_req.user_id,
                    chat_req.user_email, client_ip, chat_req.location, "vision", chat_req.mode)
            return

        # ── Text path: normal chat ──────────────────────────────────────
        async for chunk in stream_altair_response(
            chat_req.message,
            chat_req.history,
            user_id=chat_req.user_id,
            user_email=chat_req.user_email,
            mode=chat_req.mode,
            use_openrouter=chat_req.use_openrouter,
            location=chat_req.location
        ):
            if chunk != "data: [DONE]\n\n":
                # Collect for Firebase save
                token = chunk[6:].strip()
                accumulated.append(token)
            yield chunk

        # After stream done, save to Firebase in background thread
        full_response = "".join(accumulated).replace('\\n', '\n')
        if not chat_req.is_incognito:
            loop = asyncio.get_event_loop()
            loop.run_in_executor(
                None,
                save_to_firebase_bg,
                chat_req.message, full_response, "STREAMED", chat_req.user_id,
                chat_req.user_email, client_ip, chat_req.location,
                "pro" if chat_req.use_openrouter else "lite", chat_req.mode
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
async def chat(req: Request, request: ChatRequest):
    try:
        full_text = ""
        client_ip = get_client_ip(req)
        async for chunk in stream_altair_response(
            request.message, 
            request.history, 
            request.user_id,
            user_email=request.user_email,
            mode=request.mode,
            use_openrouter=request.use_openrouter
        ):
            if chunk.startswith("data: ") and "[DONE]" not in chunk:
                full_text += chunk[6:].replace('\\n', '\n')

        if not request.is_incognito:
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, save_to_firebase_bg, request.message, full_text, "CHAT", request.user_id,
                                 request.user_email, client_ip, request.location, "pro" if request.use_openrouter else "lite", request.mode)
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
        "engine":  "TIFLO AI",
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


class MemoryExtractRequest(BaseModel):
    user_message: str
    ai_response: str
    current_profile: dict = {}
    current_facts: list = []

@app.post("/memory/extract")
async def extract_memory(request: MemoryExtractRequest):
    import json
    from groq import Groq
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=groq_api_key)
    
    system_prompt = """You are a highly precise user-fact extractor. Output ONLY a valid JSON object.
Analyze the latest user message and AI response, extract key personal facts or preferences about the user, and merge/update them with the current profile and facts list.

RULES:
- Clean up duplicate or contradictory facts.
- Do not extract speculative facts. Only extract definite personal details (e.g. name, role, background, interests, key milestones).
- Keep the JSON format EXACTLY like:
{
  "user_profile": {"name": "...", "role": "...", "background": "...", "interests": "..."},
  "key_facts": ["fact 1", "fact 2"]
}
"""
    user_prompt = f"""Current Profile: {json.dumps(request.current_profile)}
Current Facts: {json.dumps(request.current_facts)}

Latest Turn:
User: {request.user_message}
AI: {request.ai_response}
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content.strip())
        print(f"🧠 [LOCAL MEMORY EXTRACT] Profile: {result.get('user_profile')}")
        return result
    except Exception as e:
        print(f"⚠️ Memory extraction error: {e}")
        return {"user_profile": request.current_profile, "key_facts": request.current_facts}


@app.get("/stats")
async def stats():
    return {
        "engine":      "TIFLO AI v3.0",
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


@app.post("/voice/transcribe")
async def voice_transcribe(file: UploadFile = File(...)):
    """
    Receives raw wav audio from browser Web Audio API and transcribes it in real-time
    using Groq's lightning-fast Whisper Large V3 engine.
    """
    try:
        contents = await file.read()
        print(f"🎙️ Voice Engine: Received {len(contents)} bytes of audio. Forwarding to Groq Whisper...")
        
        # Call Groq audio transcription
        transcription = groq_client.audio.transcriptions.create(
            file=(file.filename, contents),
            model="whisper-large-v3",
            response_format="verbose_json"
        )
        print(f"🎙️ Voice Engine: Transcription success -> '{transcription.text}'")
        return {"text": transcription.text}
    except Exception as e:
        print(f"⚠️ Voice Engine Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/feedback")
async def chat_feedback(req: FeedbackRequest):
    print(f"📥 [FEEDBACK] Type: {req.feedback_type} | User: {req.user_id} | Text: {req.feedback_text[:50]}...")
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, save_feedback_to_firebase_bg, 
        req.user_id, req.chat_id, req.feedback_type, req.feedback_text, req.last_user_message, req.last_ai_message)
    return {"status": "success", "message": "Feedback submitted successfully"}


@app.post("/chat/share")
async def chat_share(req: ShareChatRequest):
    print(f"📥 [SHARE CHAT] Creating public link for '{req.title}' ({len(req.messages)} msgs)...")
    try:
        shared_id = save_shared_chat(req.messages, req.title)
        return {"status": "success", "shared_id": shared_id}
    except Exception as e:
        print(f"⚠️ Share Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/share/{shared_id}")
async def chat_share_get(shared_id: str):
    print(f"📤 [SHARE CHAT] Fetching public link '{shared_id}'...")
    try:
        chat_data = get_shared_chat(shared_id)
        if not chat_data:
            raise HTTPException(status_code=404, detail="Shared chat not found")
        return {"status": "success", "chat": chat_data}
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"⚠️ Fetch Shared Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/chats")
async def admin_chats_get(req: Request):
    auth_email = req.headers.get("X-User-Email", "")
    if auth_email != "piyushassudani96@gmail.com":
        raise HTTPException(status_code=403, detail="Forbidden: Admin access only.")
    
    print(f"🔑 [ADMIN] Secure fetching all conversation records for {auth_email}...")
    try:
        chats_data = get_all_chats()
        return {"status": "success", "chats": chats_data}
    except Exception as e:
        print(f"⚠️ Admin Chat Fetch Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    import os
    # Hugging Face Spaces default port is 7860
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)