"""Patch script — adds AGENT intent block to master.py"""
path = "/Users/piyush/Documents/BRAHMA AI/BACKEND/master.py"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# The exact target string to find and replace
OLD = '''        yield f"data: {safe}\\n\\n"

    action_latency = time.time() - t_action'''

NEW = '''        yield f"data: {safe}\\n\\n"

    elif intent == 'AGENT':
        print(f"AGENT mode: {user_input[:60]}")
        async for chunk in run_agent(
            user_message=user_input,
            conversation_history=temp_history,
            system_prompt=active_prompt,
            user_id=user_id
        ):
            if chunk.startswith("data: ") and "[DONE]" not in chunk:
                full_response += chunk[6:].replace("\\\\n", "\\n").rstrip()
            yield chunk

    action_latency = time.time() - t_action'''

if OLD in content:
    content = content.replace(OLD, NEW, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS: AGENT block added to master.py")
else:
    print("ERROR: Target string not found. Check master.py manually.")
    # Show surrounding context for debugging
    idx = content.find("action_latency = time.time() - t_action")
    if idx >= 0:
        print("Context around action_latency:")
        print(repr(content[idx-100:idx+50]))
