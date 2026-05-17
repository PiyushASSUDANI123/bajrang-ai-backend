"""
brain_loader.py — Tiflo AI Knowledge Base Indexer
=====================================================
Loads structured data (text files, FAQs, company docs) into ChromaDB.
Run this once to "teach" Tiflo. Re-run whenever you update the data.

Usage:
    python brain_loader.py
"""

import os
import hashlib
import time
from rag_engine import add_to_knowledge_base, get_memory_stats

# ── Tiflo's Core Knowledge (Hard-coded facts) ──────────────
TIFLO_CORE_KNOWLEDGE = [
    # Identity
    "Tiflo AI is a proprietary intelligence engine built by Piyush Assudani, CEO of The Assudani Group. It is NOT ChatGPT, NOT Claude, NOT Gemini. It is an independent AI system.",
    "Tiflo AI was built to serve as the primary AI interface for The Assudani Group and its clients.",

    # About Piyush
    "Piyush Assudani is a 16-year-old tech entrepreneur and the CEO of Assudani Developers and The Assudani Group. He studies in Class 12 at Delhi Public School, Balotra, Rajasthan.",
    "Piyush Assudani's tech stack includes: MacBook Air M4, Python, FastAPI, Flutter, Firebase, HTML, CSS, JavaScript. He specializes in building premium web and mobile applications.",
    "Piyush Assudani's projects include: Atteni (bus attendance app), Nupost (festival poster maker), PyPocket (Python IDE), Tiflo AI, Loyalto, Rainbow E-Smart School portal, SparxYouth platform.",

    # The Assudani Group
    "The Assudani Group / Assudani Developers is a tech agency specializing in scalable Flutter applications, dynamic web platforms, and AI-powered products. Founded and led by Piyush Assudani.",
    "The Assudani Group recently hit a milestone of Rs 45,000 in turnover. The focus is on minimalist aesthetics, glassmorphism, and Apple-style premium UI/UX.",

    # Tiflo AI Technical Details
    "Tiflo AI's backend is built with FastAPI (Python). It uses Groq API for LLM inference with the LLaMA 3.1 model. The frontend is a single-file HTML/CSS/JS premium interface.",
    "Tiflo AI uses ChromaDB as its vector database for RAG (Retrieval Augmented Generation). It has sentence-transformers for semantic embeddings.",
    "Tiflo AI has two access levels: Guest (public access, general AI) and Founder (piyush_ceo — full access to private knowledge, business data, no restrictions).",
    "Tiflo AI's frontend features: real-time streaming, voice input, chat history (local storage), markdown rendering with syntax highlighting, export chat, suggestion chips.",

    # Design Philosophy
    "Tiflo AI's design follows the OpenClaw Design System V4. Key colors: pure black background (#000000), indigo-purple accent gradient (#6366f1 to #a855f7), Space Grotesk and Cabinet Grotesk fonts.",
    "The design philosophy of The Assudani Group: Premium over functional. Every product should feel like an Apple product — minimal, fast, beautiful.",

    # FAQs about Tiflo AI
    "What is Tiflo AI? Tiflo AI is an advanced AI assistant built by Piyush Assudani. It can answer questions, search the web in real-time, write code, and remember past conversations.",
    "How is Tiflo AI different from ChatGPT? Tiflo AI is a custom AI product built by The Assudani Group. It has real-time web search, private/public memory segmentation, and is specifically designed for Indian users with Hinglish support.",
    "Tiflo AI supports Hinglish — a mix of Hindi and English — naturally. It can respond in Hindi, English, or Hinglish depending on what the user uses.",
    "Tiflo AI Lite vs Pro: Lite is the free tier. Pro is coming soon with advanced capabilities, faster responses, and priority access.",
]

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list:
    """Split long text into overlapping chunks for better RAG retrieval."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def load_text_file(filepath: str, source_name: str):
    """Load a text file and add it to the knowledge base in chunks."""
    if not os.path.exists(filepath):
        print(f"⚠️ File not found: {filepath}")
        return 0

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return 0

    chunks = chunk_text(content)
    count = 0
    for i, chunk in enumerate(chunks):
        # Use content hash as ID to avoid duplicates
        doc_id = f"{source_name}_chunk_{hashlib.md5(chunk.encode()).hexdigest()[:10]}"
        add_to_knowledge_base(chunk, source=source_name, doc_id=doc_id)
        count += 1

    return count

def load_all():
    """Main loader — indexes everything into Tiflo's brain."""
    print("\n" + "="*50)
    print("🧠 TIFLO AI — BRAIN LOADER")
    print("="*50)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    total = 0

    # 1. Load hard-coded core knowledge
    print(f"📚 Loading {len(TIFLO_CORE_KNOWLEDGE)} core knowledge facts...")
    for i, fact in enumerate(TIFLO_CORE_KNOWLEDGE):
        doc_id = f"core_{hashlib.md5(fact.encode()).hexdigest()[:10]}"
        add_to_knowledge_base(fact, source="Tiflo Core Knowledge", doc_id=doc_id)
        total += 1

    # 2. Load company_data.txt
    print("\n📄 Loading company_data.txt...")
    count = load_text_file(
        os.path.join(os.path.dirname(__file__), "company_data.txt"),
        "Company Data"
    )
    total += count
    print(f"   → {count} chunks loaded")

    # 3. Load any additional .txt files in a 'knowledge/' folder
    knowledge_dir = os.path.join(os.path.dirname(__file__), "knowledge")
    if os.path.exists(knowledge_dir):
        for fname in os.listdir(knowledge_dir):
            if fname.endswith(".txt") or fname.endswith(".md"):
                fpath = os.path.join(knowledge_dir, fname)
                source_name = fname.replace(".txt", "").replace(".md", "").replace("_", " ").title()
                count = load_text_file(fpath, source_name)
                total += count
                print(f"   → {fname}: {count} chunks")

    # Stats
    stats = get_memory_stats()
    print(f"\n{'='*50}")
    print(f"✅ Brain loading complete!")
    print(f"   Total documents added: {total}")
    print(f"   Knowledge Base total:  {stats['knowledge_base']}")
    print(f"   Conversations stored:  {stats['conversations']}")
    print(f"   Founder memory:        {stats['private_founder']}")
    print("="*50 + "\n")


if __name__ == "__main__":
    load_all()
