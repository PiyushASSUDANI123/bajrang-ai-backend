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
    'weather', 'mausam', 'temperature', 'rain', 'barish',
    'news', 'khabar', 'latest', 'today', 'aaj', 'current',
    'price', 'rate', 'stock', 'crypto', 'bitcoin',
    'score', 'match', 'ipl', 'cricket', 'football',
    'who won', 'live', 'abhi', 'right now', 'is now',
    'kya hua', 'what happened', 'breaking',
]

# Keywords that trigger TEACH (Guru Mode)
TEACH_KEYWORDS = [
    'explain', 'samjhao', 'samjha', 'sikha', 'sikhao', 'padha',
    'what is', 'kya hai', 'kaise kaam', 'how does', 'how do',
    'concept', 'chapter', 'theory', 'nahi samajh', 'samajh nahi',
    'mujhe batao', 'easy mein', 'simple mein', 'desi mein',
    'backbencher', 'guru mode', 'trick', 'shortcut', 'formula',
    'definition', 'derive', 'proof', 'solve karo', 'step by step',
    'basics', 'introduction to', 'beginner', 'fundamentals',
]

# Keywords that trigger AGENT (tool use)
AGENT_KEYWORDS = [
    'calculate', 'compute', 'math', 'solve', '=?', 'what is', 'how much is',
    'run this', 'execute', 'code', 'python', 'run code', 'output of',
    'what does this code', 'debug', 'error in code',
]


def _url_check(user_prompt: str):
    """Fast URL detector — if a URL is present, route to URL_ANALYSIS."""
    urls = URL_PATTERN.findall(user_prompt)
    if urls:
        print(f"🔗 URL detected: {urls[0]} → URL_ANALYSIS")
        return {
            "intent": "URL_ANALYSIS",
            "reason": f"URL found: {urls[0]}",
            "urls": urls,
            "search_query": ""
        }
    return None


def _keyword_check(user_prompt):
    """Fast keyword-based router — bypasses LLM for obvious web/teach queries."""
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


def ai_router(user_prompt, conversation_history=[]):
    print(f"🧠 Router: Classifying intent...")

    # Fastest path: URL detection
    url_result = _url_check(user_prompt)
    if url_result:
        return url_result

    # Fast path: keyword check
    kw_result = _keyword_check(user_prompt)
    if kw_result:
        return kw_result

    # Slow path: Groq LLM classification
    context_snippet = ""
    for msg in conversation_history[-4:]:
        context_snippet += f"{msg['role'].upper()}: {msg['content']}\n"

    system_prompt = f"""You are a high-speed intent classifier. Output ONLY raw JSON.

INTENT RULES:
1. LOCAL_DB    → Questions about Piyush Assudani, Assudani Group, personal business data, Bajrang AI itself.
2. WEB_SEARCH  → Weather, news, scores, prices, current events, tech specs, anything time-sensitive.
3. AGENT       → Math calculations, running/debugging code, executing Python, computing formulas.
4. CHIT_CHAT   → Greetings, opinions, general conversation, writing help, explanations (no math/code execution needed).

RULES:
- Fix typos automatically in search_query.
- Resolve pronouns using context.
- When in doubt between WEB_SEARCH and CHIT_CHAT → choose WEB_SEARCH.
- When user asks to CALCULATE or RUN CODE → always choose AGENT.
- Output ONLY JSON, no markdown, no explanation.

EXAMPLES:
{{"intent": "WEB_SEARCH", "reason": "weather query", "search_query": "Balotra weather today temperature"}}
{{"intent": "WEB_SEARCH", "reason": "latest news", "search_query": "India news today"}}
{{"intent": "CHIT_CHAT", "reason": "greeting", "search_query": ""}}
{{"intent": "LOCAL_DB", "reason": "founder info", "search_query": ""}}
{{"intent": "AGENT", "reason": "math calculation", "search_query": ""}}
{{"intent": "AGENT", "reason": "code execution", "search_query": ""}}

CONTEXT:
{context_snippet}
"""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
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
        return {"intent": "CHIT_CHAT", "reason": "Error", "search_query": user_prompt}