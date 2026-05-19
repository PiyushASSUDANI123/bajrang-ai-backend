import asyncio
import os
import aiohttp
import time
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
_GROQ_MODEL = "llama-3.1-8b-instant"


# ============================================================
# REAL-TIME DATA ENGINE
# Strategy: Search → Scrape 5 URLs parallel → Feed to LLM
# Fallback: If all scrapers fail, use DuckDuckGo snippets directly
# ============================================================

def search_internet(query, max_results=5):
    """Multi-backend search with automatic fallback."""
    print(f"🌍 Live Search: '{query}'...")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if results:
                print(f"✅ Found {len(results)} results via DDGS.")
                return results
    except Exception as e:
        print(f"⚠️ DDGS search failed: {e}")

    # Fallback: Try news search
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=max_results))
            if results:
                print(f"✅ Found {len(results)} news results.")
                return results
    except Exception as e:
        print(f"⚠️ News search also failed: {e}")

    return []


def extract_ddgs_snippets(search_results):
    """
    Emergency fallback: Use DuckDuckGo search snippets directly
    when scraping fails. No URLs needed.
    """
    snippets = ""
    for i, r in enumerate(search_results[:5]):
        title = r.get('title', '')
        body = r.get('body', r.get('excerpt', ''))
        url = r.get('href', r.get('url', ''))
        if body:
            snippets += f"\n[Source {i+1}] {title} ({url}):\n{body}\n"
    return snippets


async def fetch_and_scrape(session, url, idx):
    """Async scraper — extracts clean text from a URL."""
    print(f"🚀 Agent {idx} → {url}")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        timeout = aiohttp.ClientTimeout(total=6)

        async with session.get(url, headers=headers, timeout=timeout, ssl=False) as response:
            if response.status != 200:
                return ""
            html = await response.text(errors='ignore')
            soup = BeautifulSoup(html, 'html.parser')

            # Remove noise
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form']):
                tag.decompose()

            # Priority: article > main > body paragraphs
            article = soup.find('article') or soup.find('main') or soup.find('body')
            paragraphs = article.find_all('p') if article else soup.find_all('p')

            text = " ".join(p.get_text(separator=' ', strip=True) for p in paragraphs if len(p.get_text()) > 40)

            print(f"✅ Agent {idx} returned {len(text)} chars")
            return f"\n[Source {idx}] ({url}):\n{text[:2000]}\n"

    except asyncio.CancelledError:
        return ""
    except Exception as e:
        print(f"⚠️ Agent {idx} failed: {type(e).__name__}")
        return ""


async def ask_live_ai_parallel(question, search_query):
    """
    2-phase fast engine:
    Phase 1 → DuckDuckGo snippets (always ready, 0.3s)
    Phase 2 → Parallel URL scraping for depth (3.5s cap)
    Snippets = guaranteed fast. Scraping = bonus richness.
    """
    start_time = time.time()

    search_results = search_internet(search_query)
    if not search_results:
        return "❌ Could not reach the internet. Please check your network connection."

    # Phase 1: Snippets available immediately
    snippet_context = extract_ddgs_snippets(search_results)
    print(f"⚡ Snippets ready in {round(time.time()-start_time, 2)}s")

    # Phase 2: Parallel scraping — 3.5s cap (reduced from 5s)
    scraped_context = ""
    urls = [r.get('href') or r.get('url', '') for r in search_results
            if r.get('href') or r.get('url')]

    if urls:
        async with aiohttp.ClientSession() as session:
            tasks = [asyncio.create_task(fetch_and_scrape(session, u, i+1))
                     for i, u in enumerate(urls[:4])]
            done, pending = await asyncio.wait(tasks, timeout=3.5)
            for t in pending: t.cancel()
            for t in done:
                try:
                    res = t.result()
                    if res: scraped_context += res
                except Exception: pass

    # Merge: scraped (richer) + snippets (guaranteed)
    combined_context = (
        scraped_context + "\n\n--- Quick Summaries ---\n" + snippet_context
        if scraped_context else snippet_context
    )

    print(f"⏱️ Context ready in {round(time.time()-start_time, 2)}s ({len(combined_context)} chars)")

    if not combined_context.strip():
        return "I couldn't retrieve live data for this query. Please try again."

    print(f"\n🧠 Feeding {len(combined_context)} chars of context to Llama3...\n")

    final_prompt = f"""You are TIFLO AI — a real-time intelligence engine.
Today: {time.strftime('%d %B %Y')}, {time.strftime('%H:%M')} IST

Using ONLY the live data below, answer the user's question.
Each source is tagged [Source 1], [Source 2], etc. in the data.

CITATION RULES (CRITICAL):
- When you state a fact from a source, add its number inline in brackets: e.g. "the price rose to $45,000 [1]" or "according to reports [2]..."
- Use [1], [2], [3] etc. matching the source numbers in the data below.
- Multiple sources for one fact: write [1][2]
- Do NOT add a Sources or References section at the end. Citations will be rendered separately as cards.

FORMAT RULES (strictly follow):
- Lead with the direct answer in **bold**.
- Use bullet points for multiple facts.
- Use ## headers to separate sections if long.
- Numbers and stats: always highlight in **bold**.
- Keep it concise. No rambling.

BANNED PHRASES (never say these):
- "as of my knowledge cutoff"
- "I don't have real-time access"
- "Insert Date"
- "I recommend checking"
- Any disclaimer about not having live data

=== LIVE DATA ===
{combined_context}
=================

User Question: {question}

Answer (with inline [1][2] citations, no sources section at end):"""

    try:
        response = _groq.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[{'role': 'user', 'content': final_prompt}],
            temperature=0.1
        )
        ai_text = response.choices[0].message.content

        # Build structured source list — returned separately, NOT appended to text.
        # master.py will emit these as __SOURCES__: SSE event for frontend card rendering.
        sources = []
        for i, r in enumerate(search_results[:4]):
            title = r.get('title') or 'Web Result'
            url = r.get('href') or r.get('url', '')
            if url:
                sources.append({"id": i + 1, "title": title, "url": url})

        total_time = round(time.time() - start_time, 2)
        print(f"✅ Real-time response ready in {total_time}s")
        return {"text": ai_text, "sources": sources}

    except Exception as e:
        print(f"⚠️ LLM Error: {e}")
        return {"text": "There was an internal error communicating with Groq.", "sources": []}


async def generate_followups(question: str, answer: str) -> list:
    """
    Generates 3 short, relevant follow-up questions from a Q&A pair.
    Runs after the main response — result is sent as __FOLLOWUPS__: SSE event.
    """
    prompt = f"""Based on this Q&A, generate exactly 3 short, specific follow-up questions the user might want to ask next.

User asked: {question}
AI answered: {answer[:600]}

Rules:
- Each question must be self-contained and specific
- Keep each question under 12 words
- Make them genuinely useful next steps or deeper dives
- Output ONLY a JSON object like: {{"questions": ["Q1", "Q2", "Q3"]}}"""

    try:
        import json
        response = _groq.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150,
            response_format={"type": "json_object"}
        )
        raw = response.choices[0].message.content
        parsed = json.loads(raw)
        questions = parsed.get("questions", [])
        return [str(q) for q in questions[:3]]
    except Exception as e:
        print(f"\u26a0\ufe0f Follow-ups generation failed: {e}")
        return []


if __name__ == "__main__":
    q = "What is the weather today in Mumbai?"
    query = "Mumbai weather today current temperature"
    asyncio.run(ask_live_ai_parallel(q, query))