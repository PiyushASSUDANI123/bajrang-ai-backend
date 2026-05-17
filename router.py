import os
import re
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = "llama-3.1-8b-instant"

URL_PATTERN = re.compile(r'https?://[^\s<>"\')\]`]+', re.IGNORECASE)

# Keywords that ALWAYS trigger WEB_SEARCH — no LLM needed for these
WEB_SEARCH_KEYWORDS = [
    'news', 'khabar', 'latest', 'today', 'aaj', 'current',
    'stock', 'score', 'match', 'ipl', 'cricket', 'football',
    'who won', 'live', 'abhi', 'right now', 'is now',
    'kya hua', 'what happened', 'breaking',
]

# Keywords that trigger TEACH (Guru Mode) — only for explicit teaching requests
TEACH_KEYWORDS = [
    'explain', 'samjhao', 'samjha', 'sikha', 'sikhao', 'padha',
    'concept', 'chapter', 'theory', 'nahi samajh', 'samajh nahi',
    'mujhe batao', 'easy mein', 'simple mein', 'desi mein',
    'backbencher', 'guru mode', 'trick', 'shortcut', 'formula',
    'definition', 'derive', 'proof', 'solve karo', 'step by step',
    'basics', 'introduction to', 'beginner', 'fundamentals',
]

# Keywords that route to AGENT (Autonomous Agent with custom tools)
AGENT_KEYWORDS = [
    'calculate', 'run code', 'execute', 'solve math', 'math',
    'crypto', 'bitcoin', 'btc', 'eth', 'solana', 'doge', 'rate', 'price',
    'weather', 'mausam', 'temperature', 'rain', 'barish'
]

# Keywords for GREETINGS / CHIT_CHAT fast-path
GREETING_KEYWORDS = [
    'hey', 'hi', 'hello', 'hola', 'namaste', 'kem cho', 'kaise ho',
    'sup', 'yo', 'hiii', 'heyy', 'hellooo', 'gm', 'gn', 'good morning',
    'good night', 'good evening', 'wassup', 'kya haal', 'kya chal raha',
]

def _url_check(user_prompt: str):
    """Fast URL detector — if a URL is the ONLY thing present, route to URL_ANALYSIS."""
    urls = URL_PATTERN.findall(user_prompt)
    if urls:
        # If it is a raw URL (only the URL, allowing slight surrounding whitespace/quotes/brackets)
        cleaned = user_prompt.strip().strip('"\'[]()<>`')
        if cleaned == urls[0]:
            print(f"🔗 Raw URL detected: {urls[0]} → URL_ANALYSIS")
            return {
                "intent": "URL_ANALYSIS",
                "reason": f"URL found: {urls[0]}",
                "urls": urls,
                "search_query": ""
            }
    return None


def _keyword_check(user_prompt):
    """Fast keyword-based router — bypasses LLM for obvious web/teach/agent queries."""
    lower = user_prompt.lower()

    # Check TEACH first (more specific intent)
    for kw in TEACH_KEYWORDS:
        if kw in lower:
            print(f"📚 Keyword match '{kw}' → TEACH (Guru Mode)")
            return {
                "intent": "TEACH",
                "reason": f"Keyword '{kw}' matched",
                "search_query": ""
            }

    # Check AGENT (Autonomous loop weather, crypto, math, code)
    for kw in AGENT_KEYWORDS:
        if kw in lower:
            print(f"🔧 Keyword match '{kw}' → AGENT (Autonomous Agent)")
            return {
                "intent": "AGENT",
                "reason": f"Keyword '{kw}' matched",
                "search_query": ""
            }

    # Then check WEB_SEARCH
    for kw in WEB_SEARCH_KEYWORDS:
        if kw in lower:
            print(f"⚡ Keyword match '{kw}' → WEB_SEARCH (no LLM needed)")
            return {
                "intent": "WEB_SEARCH",
                "reason": f"Keyword '{kw}' matched",
                "search_query": user_prompt
            }
    return None


def _greeting_check(user_prompt):
    """Fast greeting detector — bypasses LLM for common greetings."""
    lower = user_prompt.lower().strip()
    if lower in GREETING_KEYWORDS:
        print(f"👋 Greeting detected → CHIT_CHAT (instant)")
        return {
            "intent": "CHIT_CHAT",
            "reason": "greeting",
            "search_query": ""
        }
    return None


def ai_router(user_prompt, conversation_history=[]):
    # Clean the IndexedDB database context if it exists
    cleaned_prompt = user_prompt
    if "User Query:" in user_prompt:
        cleaned_prompt = user_prompt.split("User Query:", 1)[1].strip()
        
    # Fastest path: Greeting check using cleaned prompt
    greet_result = _greeting_check(cleaned_prompt)
    if greet_result:
        return greet_result

    # Fastest path: URL detection
    url_result = _url_check(cleaned_prompt)
    if url_result:
        return url_result

    # Fast path: keyword check
    kw_result = _keyword_check(cleaned_prompt)
    if kw_result:
        # Override search query with clean prompt for search stability
        kw_result["search_query"] = cleaned_prompt
        return kw_result

    # Slow path: Groq LLM classification
    context_snippet = ""
    for msg in conversation_history[-4:]:
        context_snippet += f"{msg['role'].upper()}: {msg['content']}\n"

    system_prompt = f"""You are a high-speed intent classifier. Output ONLY raw JSON.

INTENT RULES:
1. LOCAL_DB     → Questions about Piyush Assudani, Assudani Group, personal business data, Tiflo AI itself.
2. WEB_SEARCH   → News, scores, current events, general web search.
3. AGENT        → Math calculations, running/debugging code, executing Python, checking crypto rates (e.g. BTC, ETH), fetching weather using tools.
4. URL_ANALYSIS → When the user explicitly asks to read, analyze, scrape, summarize, or extract information from one or more provided URLs.
5. CHIT_CHAT    → Greetings, opinions, general conversation, writing help, explanations (no math/code execution needed).

RULES:
- Fix typos automatically in search_query.
- Resolve pronouns using context.
- When in doubt between WEB_SEARCH and CHIT_CHAT → choose WEB_SEARCH.
- When user asks to CALCULATE, RUN CODE, check CRYPTO RATES, or check WEATHER → always choose AGENT.
- When user provides a URL AND asks a specific question about it, choose URL_ANALYSIS.
- Output ONLY JSON, no markdown, no explanation.

EXAMPLES:
{{"intent": "WEB_SEARCH", "reason": "weather query", "search_query": "Balotra weather today temperature"}}
{{"intent": "WEB_SEARCH", "reason": "latest news", "search_query": "India news today"}}
{{"intent": "CHIT_CHAT", "reason": "greeting", "search_query": ""}}
{{"intent": "LOCAL_DB", "reason": "founder info", "search_query": ""}}
{{"intent": "AGENT", "reason": "math calculation", "search_query": ""}}
{{"intent": "AGENT", "reason": "code execution", "search_query": ""}}
{{"intent": "URL_ANALYSIS", "reason": "summarize link", "search_query": ""}}

CONTEXT:
{context_snippet}
"""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': cleaned_prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        raw_output = response.choices[0].message.content.strip()
        clean_json = re.sub(r'^```json\s*|```$', '', raw_output, flags=re.MULTILINE)
        result = json.loads(clean_json)
        print(f"🎯 Router → {result.get('intent')} | Reason: {result.get('reason')}")
        return result

    except Exception as e:
        print(f"⚠️ Router Error: {e}. Defaulting to CHIT_CHAT.")
        return {"intent": "CHIT_CHAT", "reason": "Error", "search_query": cleaned_prompt}