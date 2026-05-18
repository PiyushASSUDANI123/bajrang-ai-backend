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
    'latest news', 'kya hua aaj', 'aaj ki khabar',
    'today news', 'current news', 'breaking news',
    'who won', 'live score', 'ipl score', 'cricket score', 'match score',
    'stock price', 'share price', 'abhi kya chal raha',
    'right now happening', 'what is happening now',
]

# Loose single-word triggers for WEB_SEARCH (only if standalone or at start of message)
WEB_SEARCH_LOOSE = [
    'news', 'khabar', 'latest', 'score', 'live',
]

# Keywords that trigger TEACH (Guru Mode) — only for explicit teaching requests
TEACH_KEYWORDS = [
    'explain me', 'explain karo', 'explain kar', 'mujhe samjhao', 'samjhao', 'samjha do',
    'sikhao', 'mujhe sikha', 'padha do', 'padha', 'sikha',
    'concept explain', 'concept kya hai', 'theory kya hai',
    'nahi samajh aaya', 'samajh nahi aaya', 'mujhe nahi pata',
    'easy mein batao', 'simple mein batao', 'desi mein samjhao',
    'backbencher mode', 'guru mode', 'step by step explain',
    'solve karo step', 'solve step by step',
    'kaise karte hain', 'basics samjhao', 'introduction to', 'fundamentals of',
    'beginner guide', 'for beginners',
]

# Keywords that route to AGENT (Autonomous Agent with custom tools)
AGENT_KEYWORDS = [
    'calculate', 'run code', 'execute code', 'solve math', 'compute',
    'run this', 'execute this',
    'crypto price', 'bitcoin price', 'btc price', 'eth price', 'solana price', 'doge price',
    'current price of', 'rate of bitcoin', 'rate of btc',
    'weather in', 'mausam in', 'temperature in', 'will it rain in',
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
    lower = user_prompt.lower().strip()
    word_count = len(lower.split())

    # Short messages (≤3 words) skip aggressive keyword matching — let LLM decide.
    # This prevents "what's the math" or "nice trick" from mis-routing.
    if word_count <= 3:
        return None

    # Check TEACH first (phrase-level matching — more specific intent)
    for kw in TEACH_KEYWORDS:
        if kw in lower:
            print(f"📚 Keyword match '{kw}' → TEACH (Guru Mode)")
            return {
                "intent": "TEACH",
                "reason": f"Keyword '{kw}' matched",
                "search_query": ""
            }

    # Check AGENT (explicit action + subject phrases)
    for kw in AGENT_KEYWORDS:
        if kw in lower:
            print(f"🔧 Keyword match '{kw}' → AGENT (Autonomous Agent)")
            return {
                "intent": "AGENT",
                "reason": f"Keyword '{kw}' matched",
                "search_query": ""
            }

    # Check WEB_SEARCH — phrase-level first
    for kw in WEB_SEARCH_KEYWORDS:
        if kw in lower:
            print(f"⚡ Keyword match '{kw}' → WEB_SEARCH (no LLM needed)")
            return {
                "intent": "WEB_SEARCH",
                "reason": f"Keyword '{kw}' matched",
                "search_query": user_prompt
            }

    # Loose single-word WEB_SEARCH triggers — only if the word is standalone or at start
    words = lower.split()
    for kw in WEB_SEARCH_LOOSE:
        if words[0] == kw or (len(words) > 1 and words[1] == kw):
            print(f"⚡ Loose keyword '{kw}' at start → WEB_SEARCH")
            return {
                "intent": "WEB_SEARCH",
                "reason": f"Loose keyword '{kw}' matched at start",
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

    system_prompt = f"""You are a precise intent classifier for an AI assistant. Output ONLY raw JSON. No markdown, no explanation.

INTENT DEFINITIONS — Read carefully before classifying:

1. CHIT_CHAT (DEFAULT):
   - Any general question, explanation request, concept query, opinion, advice, writing help, coding help, language/grammar, life advice, creative request, or casual conversation.
   - This is the MOST COMMON intent. Use it for ANY message that does not clearly fit the others below.
   - Examples: "what is Python?", "explain gravity", "write me a poem", "kya lagta hai?", "mujhe code samjhao", "what should I do?", "difference between X and Y", "how does X work?"

2. WEB_SEARCH:
   - ONLY for real-time / live information that changes daily: breaking news, live scores, today's stock prices, current events.
   - Do NOT use for general knowledge questions. "What is the capital of France?" is CHIT_CHAT, not WEB_SEARCH.
   - Examples: "aaj IPL mein kaun jeeta?", "latest news about AI", "current BTC price"

3. AGENT:
   - ONLY for explicit tool use: executing code, calculating a math expression, checking live crypto/stock price, real-time weather.
   - Do NOT use for explaining concepts. "How does a for loop work?" is CHIT_CHAT, not AGENT.
   - Examples: "calculate 234 * 567", "run this Python code", "BTC price abhi kya hai", "Delhi weather right now"

4. LOCAL_DB:
   - Questions specifically about Piyush Assudani, Assudani Group, Tiflo AI's own features/team/history.
   - Examples: "who is Piyush?", "tell me about Tiflo AI", "Assudani Group kya hai?"

5. URL_ANALYSIS:
   - ONLY when the user provides a URL AND asks something about its content.
   - Examples: "summarize this link: https://...", "read this article and explain"

CRITICAL RULES:
- CHIT_CHAT is the DEFAULT. When in doubt → CHIT_CHAT.
- Do NOT choose WEB_SEARCH for general knowledge, explanations, or "how does X work" questions.
- Do NOT choose AGENT unless the user is clearly asking to RUN or CALCULATE something right now.
- Fix typos in search_query. Resolve pronouns using context.

EXAMPLES:
{"intent": "CHIT_CHAT", "reason": "general question", "search_query": ""}
{"intent": "CHIT_CHAT", "reason": "explanation request", "search_query": ""}
{"intent": "CHIT_CHAT", "reason": "coding help", "search_query": ""}
{"intent": "CHIT_CHAT", "reason": "opinion/advice", "search_query": ""}
{"intent": "WEB_SEARCH", "reason": "live score query", "search_query": "IPL 2025 today match score"}
{"intent": "WEB_SEARCH", "reason": "breaking news", "search_query": "India news today"}
{"intent": "AGENT", "reason": "math calculation", "search_query": ""}
{"intent": "AGENT", "reason": "live crypto price", "search_query": ""}
{"intent": "LOCAL_DB", "reason": "founder info", "search_query": ""}
{"intent": "URL_ANALYSIS", "reason": "user provided URL with question", "search_query": ""}

RECENT CONTEXT (last few turns):
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