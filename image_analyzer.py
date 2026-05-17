"""
image_analyzer.py — Tiflo AI Vision Engine & Multi-Modal Parser
=============================================================
Supports: base64 images, non-standard formats (HEIC, BMP, TIFF),
video keyframe frame extraction (MP4, MOV, WebM), PDF text extraction, 
and raw text files. Uses llama-4-scout and llama-3.3-70b via Groq.
"""

import os
import re
import base64
import time
import io
import tempfile
import asyncio
import cv2
from PIL import Image
import pillow_heif
from pypdf import PdfReader
try:
    import fitz
    PYMUPDF_AVAILABLE = True
    print("✅ PyMuPDF (fitz) loaded successfully for PDF extraction")
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("⚠️ PyMuPDF not found. Falling back to pypdf.")
from groq import Groq
from dotenv import load_dotenv
from typing import AsyncGenerator

# Register HEIF/HEIC support for PIL
pillow_heif.register_heif_opener()

load_dotenv()
_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Models
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TEXT_MODEL = "llama-3.3-70b-versatile"


def preprocess_image_data(data_uri: str) -> tuple[bool, str, str]:
    """
    If the image format is non-standard (HEIC, BMP, TIFF, ICO, etc.),
    convert it to standard JPEG using PIL / pillow-heif and return the standard data_uri.
    Returns: (success, mime_type, final_data_uri)
    """
    if not data_uri.startswith("data:"):
        return True, "image/jpeg", data_uri
        
    pattern = r'^data:([^;]+);base64,(.+)$'
    match = re.match(pattern, data_uri)
    if not match:
        return False, "", data_uri
        
    mime_type = match.group(1).lower()
    b64_data = match.group(2)
    
    # Standard format supported by Groq
    if mime_type in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
        return True, mime_type, data_uri
        
    try:
        raw_bytes = base64.b64decode(b64_data)
        image = Image.open(io.BytesIO(raw_bytes))
        
        # Convert RGBA/P modes to RGB for JPEG
        if image.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode == "RGBA":
                background.paste(image, mask=image.split()[3]) # alpha channel mask
            else:
                background.paste(image)
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")
            
        out_buf = io.BytesIO()
        image.save(out_buf, format="JPEG", quality=85)
        converted_b64 = base64.b64encode(out_buf.getvalue()).decode('utf-8')
        
        new_uri = f"data:image/jpeg;base64,{converted_b64}"
        print(f"🔄 Converted non-standard image type '{mime_type}' to image/jpeg")
        return True, "image/jpeg", new_uri
    except Exception as e:
        print(f"⚠️ Failed to convert image '{mime_type}': {e}")
        return False, mime_type, data_uri


def extract_video_keyframes(b64_data: str, num_keyframes: int = 4) -> list[str]:
    """
    Decodes the video base64, saves to a temp file, extracts evenly-spaced keyframes,
    resizes them for speed, encodes as JPEG base64, and returns them.
    """
    try:
        raw_bytes = base64.b64decode(b64_data)
    except Exception as e:
        print(f"⚠️ Failed to decode base64 video bytes: {e}")
        return []
        
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, f"temp_video_{int(time.time())}.mp4")
    
    try:
        with open(temp_file_path, "wb") as f:
            f.write(raw_bytes)
    except Exception as e:
        print(f"⚠️ Failed to write temp video file: {e}")
        return []
        
    keyframes_base64 = []
    try:
        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            print("⚠️ Failed to open video file using OpenCV")
            return []
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print("⚠️ Video has invalid frame count")
            return []
            
        # Distribute frames: e.g. at 10%, 40%, 70%, 90%
        pcts = [0.1, 0.4, 0.7, 0.9]
        frame_indices = [int(total_frames * p) for p in pcts]
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Keep payload small and fast
                h, w = frame.shape[:2]
                max_dim = 640
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                    
                success, buffer = cv2.imencode(".jpg", frame)
                if success:
                    b64_frame = base64.b64encode(buffer).decode('utf-8')
                    keyframes_base64.append(f"data:image/jpeg;base64,{b64_frame}")
                    
        cap.release()
    except Exception as e:
        print(f"⚠️ Error during keyframe extraction: {e}")
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass
                
    return keyframes_base64


def extract_pdf_text(b64_data: str) -> str:
    """
    Decodes the PDF base64 and extracts text from all pages using PyMuPDF (fitz) or pypdf fallback.
    """
    try:
        raw_bytes = base64.b64decode(b64_data)
        text_parts = []
        
        if PYMUPDF_AVAILABLE:
            print("📄 PyMuPDF (fitz) Engine: Extracting PDF text...")
            doc = fitz.open(stream=io.BytesIO(raw_bytes), filetype="pdf")
            for i, page in enumerate(doc):
                text = page.get_text()
                if text:
                    text_parts.append(f"--- PAGE {i+1} ---\n{text}")
        else:
            print("📄 pypdf Engine: Extracting PDF text...")
            reader = PdfReader(io.BytesIO(raw_bytes))
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    text_parts.append(f"--- PAGE {i+1} ---\n{text}")
                    
        return "\n\n".join(text_parts)[:15000] # Cap to 15k chars for prompt safety
    except Exception as e:
        print(f"⚠️ Failed to extract text from PDF: {e}")
        return f"Error parsing PDF file: {e}"


async def analyze_image_stream(
    image_data: str,
    user_question: str = "",
    conversation_history: list = None,
    is_founder: bool = False
) -> AsyncGenerator[str, None]:
    """
    Unified stream engine to analyze images, videos, text files, and PDFs.
    """
    start = time.time()
    persona = "TIFLO AI — your founder's trusted vision engine" if is_founder else "TIFLO AI"

    # 1. Check if public URL or base64 Data URI
    if not image_data.startswith("data:"):
        # Public image URL path
        mime_type = "image/jpeg"
        is_video = False
        is_pdf = False
        is_text = False
        final_uri = image_data
    else:
        # Data URI path
        pattern = r'^data:([^;]+);base64,(.+)$'
        match = re.match(pattern, image_data)
        if not match:
            yield "data: ⚠️ Invalid attachment file format.\n\n"
            yield "data: [DONE]\n\n"
            return
            
        mime_type = match.group(1).lower()
        b64_data = match.group(2)
        
        is_video = mime_type.startswith("video/")
        is_pdf = mime_type == "application/pdf"
        is_text = mime_type.startswith("text/")
        
        final_uri = image_data

    # 2. Routing logic
    # ── CASE A: Video ──────────────────────────────────────────────────────────
    if is_video:
        yield "data: ⚙️ Extracting video frames...\n\n"
        keyframes = await asyncio.to_thread(extract_video_keyframes, b64_data)
        if not keyframes:
            yield "data: ⚠️ Failed to extract keyframes from the video. Please verify the video format.\n\n"
            yield "data: [DONE]\n\n"
            return
            
        yield f"data: ⚙️ Analyzing {len(keyframes)} video scenes...\n\n"
        
        user_prompt_text = (
            f"{user_question or 'Describe this video.'}\n\n"
            "[System: The user uploaded a video. Above are keyframes extracted from the video chronologically. "
            "Describe the overall video content, the progression of the scenes, any text visible in the frames, and answer the user's question.]"
        )
        
        image_contents = [{"type": "image_url", "image_url": {"url": kf}} for kf in keyframes]
        messages = [
            {
                "role": "system",
                "content": f"You are {persona}, a cutting-edge multimodal intelligence. You analyze video frame sequences with high precision."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt_text},
                    *image_contents
                ]
            }
        ]
        
        or_key = os.getenv("OPENROUTER_API_KEY", "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")
        headers = {
            "Authorization": f"Bearer {or_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://triflo.in",
            "X-Title": "Tiflo AI"
        }
        
        payload = {
            "model": "meta-llama/llama-3.2-11b-vision-instruct",
            "messages": messages,
            "stream": True,
            "temperature": 0.2
        }
        
        try:
            import aiohttp
            import json
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                ) as resp:
                    if resp.status == 200:
                        async for line in resp.content:
                            if line:
                                decoded_line = line.decode('utf-8', errors='ignore').strip()
                                if decoded_line.startswith("data: "):
                                    data_str = decoded_line[6:].strip()
                                    if data_str == "[DONE]":
                                        break
                                    try:
                                        data_json = json.loads(data_str)
                                        token = data_json['choices'][0]['delta'].get('content', '')
                                        if token:
                                            yield f"data: {token.replace('\n', '\\n')}\n\n"
                                    except Exception:
                                        pass
                    else:
                        err_text = await resp.text()
                        yield f"data: ⚠️ OpenRouter Video Vision Error {resp.status}: {err_text[:200]}\\n\\n\n\n"
            print(f"✅ Video analyzed in {round(time.time() - start, 2)}s via OpenRouter Llama 3.2 Vision")
        except Exception as e:
            yield f"data: ⚠️ Video analysis failed: {str(e)[:100]}\n\n"
            
        yield "data: [DONE]\n\n"
        return

    # ── CASE B: PDF Document ──────────────────────────────────────────────────
    elif is_pdf:
        yield "data: ⚙️ Parsing PDF text...\n\n"
        pdf_text = await asyncio.to_thread(extract_pdf_text, b64_data)
        if not pdf_text.strip():
            yield "data: ⚠️ PDF parsed but no text was extracted.\n\n"
            yield "data: [DONE]\n\n"
            return
            
        yield "data: ⚙️ Analyzing document...\n\n"
        
        system_content = f"""You are {persona}, an elite analyst. 
Analyze the provided document context carefully and answer the user's questions with absolute accuracy.
Structure your findings beautifully with bullet points and bold terms.
"""
        user_prompt_text = f"""USER QUESTION: {user_question or 'Summarize this PDF document.'}

=== PDF CONTENT ===
{pdf_text}
"""
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt_text}
        ]
        
        try:
            stream = _groq.chat.completions.create(
                model=TEXT_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=1500,
                stream=True
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                if token:
                    yield f"data: {token.replace('\n', '\\n')}\n\n"
            print(f"✅ PDF analyzed in {round(time.time() - start, 2)}s")
        except Exception as e:
            yield f"data: ⚠️ PDF analysis failed: {str(e)[:100]}\n\n"
            
        yield "data: [DONE]\n\n"
        return

    # ── CASE C: Text File ─────────────────────────────────────────────────────
    elif is_text:
        yield "data: ⚙️ Reading text file...\n\n"
        try:
            raw_text = base64.b64decode(b64_data).decode('utf-8', errors='ignore')
        except Exception as e:
            yield f"data: ⚠️ Failed to decode text file: {str(e)}\n\n"
            yield "data: [DONE]\n\n"
            return
            
        yield "data: ⚙️ Analyzing text content...\n\n"
        system_content = f"You are {persona}, a highly advanced text analytics engine."
        user_prompt_text = f"""USER QUESTION: {user_question or 'Summarize this text file.'}

=== FILE CONTENT ===
{raw_text[:15000]}
"""
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt_text}
        ]
        
        try:
            stream = _groq.chat.completions.create(
                model=TEXT_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=1500,
                stream=True
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                if token:
                    yield f"data: {token.replace('\n', '\\n')}\n\n"
            print(f"✅ Text file analyzed in {round(time.time() - start, 2)}s")
        except Exception as e:
            yield f"data: ⚠️ Text analysis failed: {str(e)[:100]}\n\n"
            
        yield "data: [DONE]\n\n"
        return

    # ── CASE D: Image ─────────────────────────────────────────────────────────
    else:
        # Standardize and preprocess image (including conversions of HEIC/BMP/TIFF)
        success, final_mime, final_uri = await asyncio.to_thread(preprocess_image_data, final_uri)
        if not success:
            yield f"data: ⚠️ Unsupported or corrupted image format. Please upload JPEG, PNG, WEBP, GIF, HEIC, or BMP.\n\n"
            yield "data: [DONE]\n\n"
            return
            
        image_content = {"type": "image_url", "image_url": {"url": final_uri}}
        if not user_question.strip():
            user_question = (
                "Analyze this image in detail. Describe what you see, "
                "extract any text, identify key elements, and give insights."
            )
            
        system_content = f"""You are {persona}, a powerful multimodal AI.
Analyze images with surgical precision.

FORMAT:
- Lead with the most important observation in **bold**.
- Use bullet points for multiple findings.
- Extract ALL visible text (OCR) if present — put it in a code block.
- Flag anything unusual, important, or worth noting.
- Be direct. No filler. No "I see an image of..."
- Match the user's language/vibe (Hindi, English, Hinglish — as they wrote).
"""
        messages = [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_question},
                    image_content
                ]
            }
        ]
        
        or_key = os.getenv("OPENROUTER_API_KEY", "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")
        headers = {
            "Authorization": f"Bearer {or_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://triflo.in",
            "X-Title": "Tiflo AI"
        }
        
        payload = {
            "model": "meta-llama/llama-3.2-11b-vision-instruct",
            "messages": messages,
            "stream": True,
            "temperature": 0.2
        }
        
        try:
            import aiohttp
            import json
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                ) as resp:
                    if resp.status == 200:
                        async for line in resp.content:
                            if line:
                                decoded_line = line.decode('utf-8', errors='ignore').strip()
                                if decoded_line.startswith("data: "):
                                    data_str = decoded_line[6:].strip()
                                    if data_str == "[DONE]":
                                        break
                                    try:
                                        data_json = json.loads(data_str)
                                        token = data_json['choices'][0]['delta'].get('content', '')
                                        if token:
                                            yield f"data: {token.replace('\n', '\\n')}\n\n"
                                    except Exception:
                                        pass
                    else:
                        err_text = await resp.text()
                        yield f"data: ⚠️ OpenRouter Vision Error {resp.status}: {err_text[:200]}\\n\\n\n\n"
            print(f"✅ Image analyzed in {round(time.time() - start, 2)}s using OpenRouter Llama 3.2 Vision")
        except Exception as e:
            yield f"data: ⚠️ Image analysis failed: {str(e)[:100]}\n\n"
            
        yield "data: [DONE]\n\n"


def encode_file_to_base64(file_bytes: bytes, mime_type: str) -> str:
    """Convert raw file bytes to base64 data URI."""
    b64 = base64.b64encode(file_bytes).decode('utf-8')
    return f"data:{mime_type};base64,{b64}"
