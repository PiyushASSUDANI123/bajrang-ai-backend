"""
image_analyzer.py — Bajrang AI Vision Engine
=============================================
Uses Groq's llama-4-scout vision model to analyze images.
Supports: base64 images, image URLs, screenshots, documents.
"""

import os
import re
import base64
import time
from groq import Groq
from dotenv import load_dotenv
from typing import AsyncGenerator

load_dotenv()

_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Groq's fastest vision model
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

SUPPORTED_FORMATS = {"image/jpeg", "image/png", "image/gif", "image/webp"}


def validate_base64_image(data_uri: str) -> tuple[bool, str, str]:
    """
    Validate and extract mime type from base64 data URI.
    Returns: (is_valid, mime_type, clean_base64)
    """
    pattern = r'^data:(image/[a-zA-Z+]+);base64,(.+)$'
    match = re.match(pattern, data_uri)
    if not match:
        return False, "", ""
    mime_type = match.group(1)
    b64_data = match.group(2)
    if mime_type not in SUPPORTED_FORMATS:
        return False, mime_type, ""
    return True, mime_type, b64_data


async def analyze_image_stream(
    image_data: str,
    user_question: str = "",
    conversation_history: list = None,
    is_founder: bool = False
) -> AsyncGenerator[str, None]:
    """
    Stream image analysis response token by token.

    Args:
        image_data:  base64 data URI (data:image/jpeg;base64,...)
                     OR a public image URL (https://...)
        user_question: what the user asked about the image
        conversation_history: previous messages for context
        is_founder: whether to use founder persona
    """
    start = time.time()

    # Build the image content part
    if image_data.startswith("data:"):
        is_valid, mime_type, _ = validate_base64_image(image_data)
        if not is_valid:
            yield f"data: ⚠️ Unsupported image format. Use JPG, PNG, GIF, or WebP.\n\n"
            yield "data: [DONE]\n\n"
            return
        image_content = {"type": "image_url", "image_url": {"url": image_data}}
    elif image_data.startswith("http"):
        image_content = {"type": "image_url", "image_url": {"url": image_data}}
    else:
        yield "data: ⚠️ Invalid image data. Send a base64 data URI or a public URL.\n\n"
        yield "data: [DONE]\n\n"
        return

    # Build the question text
    if not user_question.strip():
        user_question = (
            "Analyze this image in detail. Describe what you see, "
            "extract any text, identify key elements, and give insights."
        )

    persona = "BAJRANG AI — your founder's trusted vision engine" if is_founder else "BAJRANG AI"

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

    try:
        stream = _groq.chat.completions.create(
            model=VISION_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
            stream=True
        )

        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                safe = token.replace('\n', '\\n')
                yield f"data: {safe}\n\n"

        elapsed = round(time.time() - start, 2)
        print(f"✅ [VISION] Image analyzed in {elapsed}s using {VISION_MODEL}")

    except Exception as e:
        print(f"⚠️ Vision error: {e}")
        yield f"data: ⚠️ Image analysis failed: {str(e)[:100]}\n\n"

    yield "data: [DONE]\n\n"


def encode_file_to_base64(file_bytes: bytes, mime_type: str) -> str:
    """Convert raw file bytes to base64 data URI."""
    b64 = base64.b64encode(file_bytes).decode('utf-8')
    return f"data:{mime_type};base64,{b64}"
