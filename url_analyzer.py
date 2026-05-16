"""
url_analyzer.py — Bajrang AI URL / Link Analyzer
==================================================
Fetches any URL, extracts clean content, and sends to Groq for analysis.
Handles: articles, product pages, GitHub repos, YouTube, news, docs, etc.
"""

import os
import re
import time
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_groq   = Groq(api_key=os.getenv("GROQ_API_KEY"))
_MODEL  = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/124.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
}


def extract_urls(text: str) -> list:
    """Extract all URLs from a block of text."""
    pattern = r'https?://[^\s<>"\')\]`]+'
    return list(set(re.findall(pattern, text)))


async def fetch_url_content(url: str, timeout: int = 10) -> dict:
    """
    Async fetch + clean content from a URL.
    Returns: { url, title, description, content, word_count, status }
    """
    result = {
        "url": url,
        "title": "",
        "description": "",
        "content": "",
        "word_count": 0,
        "status": "ok",
        "error": None
    }

    try:
        t = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(headers=HEADERS, timeout=t) as session:
            async with session.get(url, ssl=False, allow_redirects=True) as resp:
                if resp.status >= 400:
                    result["status"] = "error"
                    result["error"] = f"HTTP {resp.status}"
                    return result

                content_type = resp.headers.get("Content-Type", "")
                if "text/html" not in content_type and "text/plain" not in content_type:
                    result["status"] = "unsupported"
                    result["error"] = f"Content-Type not supported: {content_type}"
                    return result

                html = await resp.text(errors="ignore")

    except asyncio.TimeoutError:
        result["status"] = "timeout"
        result["error"] = "Request timed out"
        return result
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        return result

    # ── Parse HTML ────────────────────────────────────────────
    soup = BeautifulSoup(html, "html.parser")

    # Title
    title_tag = soup.find("title")
    result["title"] = title_tag.get_text(strip=True) if title_tag else ""

    # Meta description
    meta_desc = soup.find("meta", {"name": re.compile("description", re.I)})
    if meta_desc:
        result["description"] = meta_desc.get("content", "")[:300]

    # Remove noise tags
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "aside", "form", "noscript", "iframe", "svg",
                     "button", "input", "select", "textarea", "ads",
                     "[class*='cookie']", "[class*='popup']"]):
        tag.decompose()

    # Priority content zones
    content_zones = (
        soup.find("article") or
        soup.find("main") or
        soup.find(id=re.compile(r"(content|main|article|post|body)", re.I)) or
        soup.find(class_=re.compile(r"(content|article|post|entry|text)", re.I)) or
        soup.find("body")
    )

    if content_zones:
        paras = content_zones.find_all(["p", "h1", "h2", "h3", "h4", "li", "td", "blockquote"])
        text_parts = []
        for p in paras:
            t = p.get_text(separator=" ", strip=True)
            if len(t) > 20:  # Filter tiny fragments
                text_parts.append(t)
        content = "\n".join(text_parts)
    else:
        content = soup.get_text(separator="\n", strip=True)

    # Truncate to 6000 chars (enough context for LLM without overflow)
    result["content"] = content[:6000]
    result["word_count"] = len(content.split())

    return result


async def analyze_url(url: str, user_question: str = "") -> str:
    """
    Main function: fetch URL + analyze with Groq.
    user_question: optional specific question about the URL content.
    Returns: AI-generated analysis as a string.
    """
    print(f"🔗 Analyzing URL: {url}")
    t0 = time.time()

    # Fetch content
    page = await fetch_url_content(url)

    if page["status"] != "ok":
        return (
            f"**Could not fetch the URL.**\n\n"
            f"- **URL:** `{url}`\n"
            f"- **Error:** {page['error']}\n\n"
            f"Please check if the link is accessible and try again."
        )

    if not page["content"].strip():
        return (
            f"**Page fetched but no readable content found.**\n\n"
            f"- **URL:** `{url}`\n"
            f"- **Title:** {page['title'] or 'Unknown'}\n\n"
            f"The page may require JavaScript or login to display content."
        )

    fetch_time = round(time.time() - t0, 2)
    print(f"✅ Fetched {page['word_count']} words in {fetch_time}s — {page['title'][:60]}")

    # Build analysis prompt
    specific_q = f"\n\nUser's specific question: {user_question}" if user_question else ""

    prompt = f"""You are BAJRANG AI analyzing a webpage for the user.

PAGE DETAILS:
- URL: {url}
- Title: {page['title']}
- Description: {page['description']}
- Word count: ~{page['word_count']} words

PAGE CONTENT (first 6000 chars):
{page['content']}
{specific_q}

TASK: Provide a comprehensive analysis of this page. Structure your response as:

## 📄 Page Overview
Brief description of what this page/article is about.

## 🔑 Key Points
The most important information from the page (bullet points).

## 📊 Details
Any important data, stats, or specifics mentioned.

## 💡 Summary
1-2 sentence takeaway.

FORMAT RULES:
- Use **bold** for important terms/numbers.
- Be direct and specific — no filler phrases.
- If it's code/GitHub: explain the project, tech stack, and purpose.
- If it's news/article: give key facts and what it means.
- If it's product/shop: highlight key specs, price, pros/cons.
- If it's docs: explain what it does and how to use it.
"""

    try:
        response = _groq.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        analysis = response.choices[0].message.content
        total_time = round(time.time() - t0, 2)
        print(f"✅ URL analyzed in {total_time}s total")
        return analysis

    except Exception as e:
        print(f"⚠️ Groq error during URL analysis: {e}")
        return f"**Fetched the page but failed to analyze it.**\n\nError: {e}"


async def analyze_multiple_urls(urls: list, user_question: str = "") -> str:
    """Analyze multiple URLs in parallel (max 3)."""
    urls = urls[:3]  # Limit to 3 to avoid overload
    tasks = [analyze_url(u, user_question) for u in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    output = []
    for url, result in zip(urls, results):
        if isinstance(result, Exception):
            output.append(f"**{url}** — Error: {result}")
        else:
            output.append(f"---\n### 🔗 {url}\n{result}")

    return "\n\n".join(output)


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://en.wikipedia.org/wiki/Artificial_intelligence"
    result = asyncio.run(analyze_url(url))
    print(result)
