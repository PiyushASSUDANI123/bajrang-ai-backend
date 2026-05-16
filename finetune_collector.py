"""
finetune_collector.py — Bajrang AI Fine-tune Data Pipeline
===========================================================
Collects high-quality Q&A pairs from ChromaDB conversations,
formats them as JSONL for fine-tuning on Together AI / OpenPipe / HuggingFace.

Usage:
    python finetune_collector.py              # Export all data
    python finetune_collector.py --min-len 50 # Only long responses

Output: finetune_data.jsonl (OpenAI chat format — works everywhere)
"""

import json
import os
import argparse
import chromadb
from datetime import datetime

DB_PATH      = "./ai_memory_db"
OUTPUT_FILE  = "finetune_data.jsonl"
SYSTEM_MSG   = "You are BAJRANG AI — a raw, direct, no-filter intelligence engine built by Piyush Assudani, CEO of The Assudani Group."


def extract_qa(doc: str):
    """Parse 'Q: ...\nA: ...' format from stored conversations."""
    try:
        parts = doc.split("\nA:", 1)
        if len(parts) != 2:
            return None, None
        question = parts[0].replace("Q:", "").strip()
        answer   = parts[1].strip()
        return question, answer
    except Exception:
        return None, None


def export(min_response_len: int = 30, max_records: int = 5000):
    client = chromadb.PersistentClient(path=DB_PATH)

    records = 0
    skipped = 0
    seen    = set()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for col_name in ["rag_conversations", "rag_private"]:
            try:
                col = client.get_collection(col_name)
            except Exception:
                continue

            total = col.count()
            print(f"📦 {col_name}: {total} documents found")

            if total == 0:
                continue

            # Fetch all (ChromaDB limit = 10000)
            result = col.get(include=["documents", "metadatas"], limit=min(total, 10000))

            for doc, meta in zip(result["documents"], result["metadatas"]):
                if records >= max_records:
                    break

                question, answer = extract_qa(doc)
                if not question or not answer:
                    skipped += 1
                    continue

                if len(answer) < min_response_len:
                    skipped += 1
                    continue

                # Deduplicate
                key = question.lower()[:80]
                if key in seen:
                    skipped += 1
                    continue
                seen.add(key)

                # Write in OpenAI fine-tune format
                sample = {
                    "messages": [
                        {"role": "system",    "content": SYSTEM_MSG},
                        {"role": "user",      "content": question},
                        {"role": "assistant", "content": answer}
                    ]
                }
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                records += 1

    print(f"\n✅ Fine-tune export complete!")
    print(f"   Records exported: {records}")
    print(f"   Records skipped:  {skipped}")
    print(f"   Output file:      {OUTPUT_FILE}")
    print(f"\n📤 Next steps to fine-tune:")
    print(f"   1. Together AI: https://api.together.xyz/finetune")
    print(f"   2. OpenPipe:    https://openpipe.ai")
    print(f"   3. HuggingFace: Use trl library with finetune_data.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Bajrang AI fine-tune data")
    parser.add_argument("--min-len",  type=int, default=30,   help="Minimum response length")
    parser.add_argument("--max",      type=int, default=5000, help="Max records to export")
    args = parser.parse_args()
    export(min_response_len=args.min_len, max_records=args.max)
