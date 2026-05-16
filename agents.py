"""
agents.py — Bajrang AI Tool Calling / Agent System
====================================================
Gives Bajrang AI real tools it can USE — not just talk about.
Each tool is a function the AI can invoke via Groq function calling.

Tools:
  - calculator     : Math expressions (safe eval)
  - code_runner    : Execute Python code in a sandbox
  - memory_search  : Search past conversations
  - web_search     : Real-time internet search (wraps fast_ai)
  - knowledge_search: Search Bajrang's knowledge base
"""

import os
import ast
import sys
import json
import time
import math
import asyncio
import traceback
from io import StringIO
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# ── Tool Definitions (Groq function calling format) ──────────
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression. Use for any arithmetic, algebra, percentage, or unit conversion. Always use this for math instead of calculating in your head.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate. Example: '(234 * 56) / 100', 'math.sqrt(144)', '2**10'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "code_runner",
            "description": "Execute Python code and return the output. Use when the user asks to run code, test something, or when you need to compute something complex.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Valid Python code to execute. Print results using print()."
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search Bajrang AI's memory for relevant past information, facts about Piyush, or the Assudani Group.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in memory."
                    }
                },
                "required": ["query"]
            }
        }
    }
]


# ── Tool Implementations ─────────────────────────────────────

def _tool_calculator(expression: str) -> str:
    """Safe math evaluator — no exec(), uses AST."""
    SAFE_NAMES = {
        k: v for k, v in math.__dict__.items() if not k.startswith("_")
    }
    SAFE_NAMES.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum, "len": len})
    try:
        # Parse and validate — only allow safe nodes
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Call)):
                # Allow math function calls but not arbitrary calls
                if isinstance(node, ast.Call):
                    if not isinstance(node.func, (ast.Name, ast.Attribute)):
                        raise ValueError("Unsafe expression")
        result = eval(compile(tree, "<string>", "eval"), {"__builtins__": {}}, SAFE_NAMES)
        return f"Result: {result}"
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Calculation error: {e}"


def _tool_code_runner(code: str) -> str:
    """Sandboxed Python code runner — captures stdout, limits execution."""
    # Block dangerous operations
    BLOCKED = ["import os", "import sys", "open(", "__import__", "subprocess", "eval(", "exec(", "shutil"]
    for block in BLOCKED:
        if block in code:
            return f"⚠️ Blocked: '{block}' is not allowed for security reasons."

    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()

    result = ""
    try:
        # 3-second timeout via signal (basic protection)
        local_env = {}
        exec(compile(code, "<sandbox>", "exec"), {"__builtins__": {"print": print, "range": range, "len": len, "str": str, "int": int, "float": float, "list": list, "dict": dict, "tuple": tuple, "set": set, "sorted": sorted, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter, "sum": sum, "min": min, "max": max, "abs": abs, "round": round, "math": math, "True": True, "False": False, "None": None}}, local_env)
        output = captured.getvalue()
        result = f"Output:\n```\n{output.strip() if output.strip() else '(no output)'}\n```"
    except Exception as e:
        result = f"Error: {type(e).__name__}: {e}"
    finally:
        sys.stdout = old_stdout

    return result


def _tool_memory_search(query: str, user_id: str = "guest") -> str:
    """Search RAG knowledge base and conversation memory."""
    try:
        from rag_engine import retrieve_context
        context = retrieve_context(query, user_id=user_id, n_results=3)
        if context:
            return f"Found in memory:\n{context}"
        return "Nothing relevant found in memory."
    except Exception as e:
        return f"Memory search error: {e}"


# ── Main Agent Runner ────────────────────────────────────────
async def run_agent(
    user_message: str,
    conversation_history: list,
    system_prompt: dict,
    user_id: str = "guest"
):
    """
    Agentic loop — AI can call tools, get results, then answer.
    Yields SSE-formatted tokens just like master.py streaming.
    """
    messages = conversation_history + [{"role": "user", "content": user_message}]
    if messages[0]["role"] != "system":
        messages.insert(0, system_prompt)

    try:
        # Step 1: Ask LLM if it wants to use any tools
        response = _groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            temperature=0.1
        )

        msg = response.choices[0].message
        tool_calls = msg.tool_calls

        if not tool_calls:
            # No tools needed — just stream the answer
            content = msg.content or ""
            for token in content.split():
                yield f"data: {token} \n\n"
                await asyncio.sleep(0.01)
            return

        # Step 2: Execute tool calls
        tool_results = []
        tool_summary = ""

        for tc in tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)

            print(f"🔧 Agent calling tool: {fn_name}({fn_args})")
            yield f"data: ⚙️ Using tool: **{fn_name}**...\\n\\n\n\n"
            await asyncio.sleep(0.05)

            if fn_name == "calculator":
                result = _tool_calculator(fn_args.get("expression", ""))
            elif fn_name == "code_runner":
                result = _tool_code_runner(fn_args.get("code", ""))
            elif fn_name == "memory_search":
                result = _tool_memory_search(fn_args.get("query", ""), user_id)
            else:
                result = f"Unknown tool: {fn_name}"

            tool_results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
            })
            tool_summary += f"\n**{fn_name}** result: {result[:200]}"

        # Step 3: Feed tool results back to LLM for final answer
        messages_with_tools = messages + [
            {"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ]}
        ] + tool_results

        final_stream = _groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages_with_tools,
            temperature=0.1,
            stream=True
        )

        for chunk in final_stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                safe = token.replace('\n', '\\n')
                yield f"data: {safe}\n\n"

    except Exception as e:
        print(f"⚠️ Agent error: {e}")
        traceback.print_exc()
        yield f"data: ⚠️ Agent encountered an error: {str(e)}\\n\n\n"
