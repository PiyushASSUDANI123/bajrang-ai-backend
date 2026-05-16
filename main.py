import ollama

def chat_with_local_ai():
    print("Sending prompt to local AI... (Tera M4 ab thoda heat hoga)\n")
    
    # Yeh API call internet par nahi, tere localhost par ja rahi hai
    response = ollama.chat(model='llama3', messages=[
        {
            'role': 'user',
            'content': 'Explain the difference between a list and a tuple in Python in one strict sentence.'
        }
    ])
    
    print("--- AI RESPONSE ---")
    print(response['message']['content'])
    print("-------------------")

if __name__ == "__main__":
    chat_with_local_ai()