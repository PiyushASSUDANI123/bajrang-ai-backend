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

    final_prompt = f"""You are BAJRANG AI — a real-time intelligence engine.
Today: {time.strftime('%d %B %Y')}, {time.strftime('%H:%M')} IST

Using ONLY the live data below, answer the user's question.

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

Answer (organized, direct, no disclaimers):"""

    try:
        response = _groq.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[{'role': 'user', 'content': final_prompt}],
            temperature=0.1
        )
        ai_text = response.choices[0].message.content
        total_time = round(time.time() - start_time, 2)
        print(f"✅ Real-time response ready in {total_time}s")
        return ai_text

    except Exception as e:
        print(f"⚠️ LLM Error: {e}")
        return "There was an internal error communicating with Groq."


if __name__ == "__main__":
    q = "What is the weather today in Mumbai?"
    query = "Mumbai weather today current temperature"
    asyncio.run(ask_live_ai_parallel(q, query))