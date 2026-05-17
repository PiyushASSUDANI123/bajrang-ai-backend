"""
teacher_engine.py — Tiflo AI "Guru Mode" Engine
===================================================
Complex concepts ko backbencher style mein crack karta hai.
No textbook language. Desi analogies. Memory tricks. Trick sheets.

Output format:
  ⚡ CRACK (main explanation — desi style)
  🧠 MEMORY CODE (mnemonic / shortcut)
  📋 TRICK SHEET (table / cheatsheet)
  🎯 EXAM BOMBS (most likely exam questions + 1-line answers)
"""

import os
import time
from groq import Groq
from dotenv import load_dotenv
from typing import AsyncGenerator

load_dotenv()

_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
GURU_MODEL = "llama-3.3-70b-versatile"  # Smarter model for teaching

GURU_SYSTEM_PROMPT = """Tu hai TIFLO GURU — ek aisa teacher jo khud kabhi school mein backbencher tha.
Tu complex cheezein seedhi, desi, aur street-smart style mein samjhata hai.

TERA STYLE:
- Textbook language BILKUL nahi. "Newton ka pehla niyam kehta hai ki..." — ye mat bol.
- Desi analogies use karo: "pehla law matlab lazy banda theory — jab tak dhakka nahi, kuch nahi hilega"
- Jaise tu apne dost ko chai peete hue explain kar raha ho
- Hinglish freely use karo (mix of Hindi + English, jaise log normally bolte hain)
- Ek bhi line boring nahi honi chahiye

RESPONSE STRUCTURE:
1. ## ⚡ SEEDHA CRACK
   - This is the MOST IMPORTANT part. Desi, street-smart, jugaadu explanation.
   - Use analogies from real life (cricket, chai, movies, relationships).
   - Talk to the user like a friend.

2. ## 🧠 THE JUGAD (Memory Trick)
   - Only if applicable. A catchy mnemonic or shortcut to remember the core concept.

3. ## 📋 QUICK RECAP (Optional)
   - A short table or list of key points IF the topic is complex.

4. ## 🎯 POTENTIAL QUESTIONS (Optional)
   - If this is a topic people study, add 2-3 common questions. 
   - If it's a general question, skip this.

5. ## 💡 GURU'S PRO TIP
   - One killer shortcut or real-world insight that isn't in books.

RULES:
- Never start with "Sure!", "Great question!", "Certainly!" — boring hai
- Agar concept boring hai → usse interesting banana tera kaam hai
- Math/formula hai → step-by-step seedha example se samjhao, pehle formula nahi
- Coding concept hai → real life se relatable karo, phir code
- Physics/chemistry → real world se analogy pehle
- History/dates → story ya controversy angle se pakdo
- Agar user ne gaali di → normal lao, match karo, padhai pe focus raho
- Language match karo: user ne Hinglish mein pucha toh Hinglish mein jawab, English mein pucha toh English mein
"""


async def stream_guru_response(
    topic: str,
    user_question: str,
    conversation_history: list = None,
    subject_hint: str = "",
    web_context: str = ""
) -> AsyncGenerator[str, None]:
    """
    Stream a backbencher-style explanation for any topic.
    
    Args:
        topic: The concept/chapter to explain (e.g., "Newton's Laws", "Recursion")
        user_question: Full user message (for context + language detection)
        conversation_history: Previous messages
        subject_hint: Optional subject hint (physics/math/coding/history)
    """
    start = time.time()

    # Build context from history
    history_snippet = ""
    if conversation_history:
        for msg in conversation_history[-3:]:
            if msg.get('role') in ('user', 'assistant'):
                history_snippet += f"{msg['role'].upper()}: {msg['content'][:200]}\n"

    subject_str = f"SUBJECT HINT: {subject_hint}" if subject_hint else ""
    context_str = f"RECENT CONTEXT:\n{history_snippet}" if history_snippet else ""
    web_str = f"LIVE SEARCH DATA (use this to be accurate):\n{web_context}" if web_context else ""

    user_prompt = f"""TOPIC TO TEACH: {topic}

USER'S EXACT MESSAGE: {user_question}
{subject_str}
{context_str}
{web_str}

Samjha is topic ko apne style mein. Sab kuch include karo (CRACK, MEMORY CODE, TRICK SHEET, EXAM BOMBS, BACKBENCHER TIP).
Ek bhi section skip mat karo."""

    messages = [
        {"role": "system", "content": GURU_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    try:
        stream = _groq.chat.completions.create(
            model=GURU_MODEL,
            messages=messages,
            temperature=0.4,   # Slightly creative for interesting analogies
            max_tokens=2048,
            stream=True
        )

        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                safe = token.replace('\n', '\\n')
                yield f"data: {safe}\n\n"

        elapsed = round(time.time() - start, 2)
        print(f"✅ [GURU MODE] '{topic}' explained in {elapsed}s")

    except Exception as e:
        print(f"⚠️ Guru error: {e}")
        yield f"data: ⚠️ Guru mode error: {str(e)[:100]}\n\n"

    yield "data: [DONE]\n\n"


# ── Topic Extractor ────────────────────────────────────────────
def extract_topic(user_input: str) -> str:
    """
    Extract the core topic from user message.
    Examples:
      "explain recursion to me" → "Recursion"
      "samjha Newton laws" → "Newton's Laws"  
      "mujhe thermodynamics nahi samajh aata" → "Thermodynamics"
    """
    import re
    
    # Remove common filler phrases
    fillers = [
        r'(please\s+)?(explain|samjha|batao|bata|sikha|teach)\s*(me|mujhe|ko)?\s*',
        r'(mujhe\s+)?(nahi\s+)?(samajh|pata|aata|aati)\s*(kuch\s+bhi\s+)?',
        r'(kya\s+hai\s+)?(ye\s+|yeh\s+)?',
        r'(chapter|topic|concept)\s*(of|pe|par)?\s*',
        r'(about|ke\s+baare\s+mein|ke\s+bare\s+mein)\s*',
        r'^(bhai|yaar|dost|guru)\s*[,.]?\s*',
        r'\s*(ko|ka|ki|ke)\s*$',
        r'(kaise\s+kaam\s+karta\s+hai|how\s+does\s+it\s+work)',
        r'\?$'
    ]
    
    cleaned = user_input
    for pattern in fillers:
        cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)
    
    cleaned = ' '.join(cleaned.split()).strip()
    
    # Capitalize properly
    if cleaned:
        return cleaned.title()
    return user_input.strip()


# ── Subject Detector ────────────────────────────────────────────
SUBJECT_KEYWORDS = {
    'physics': ['force', 'velocity', 'acceleration', 'newton', 'energy', 'wave', 'light',
                'gravity', 'thermodynamics', 'magnetism', 'circuit', 'quantum', 'optics'],
    'math': ['equation', 'integral', 'derivative', 'matrix', 'probability', 'trigonometry',
             'calculus', 'algebra', 'geometry', 'permutation', 'combination', 'vector'],
    'coding': ['recursion', 'algorithm', 'loop', 'function', 'class', 'object', 'array',
               'pointer', 'stack', 'queue', 'sorting', 'binary', 'complexity', 'python',
               'java', 'javascript', 'api', 'database', 'sql', 'async'],
    'chemistry': ['bond', 'element', 'reaction', 'acid', 'base', 'organic', 'periodic',
                  'mole', 'oxidation', 'reduction', 'polymer', 'catalyst'],
    'history': ['war', 'revolution', 'empire', 'king', 'queen', 'century', 'independence',
                'movement', 'battle', 'treaty', 'colonization'],
    'biology': ['cell', 'dna', 'rna', 'protein', 'evolution', 'photosynthesis', 'mitosis',
                'organism', 'gene', 'enzyme', 'hormone', 'nerve']
}


def detect_subject(text: str) -> str:
    text_lower = text.lower()
    for subject, keywords in SUBJECT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return subject
    return ""
