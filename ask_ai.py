import chromadb
import ollama

# 1. Purane database se connect kar
client = chromadb.PersistentClient(path="./my_vector_db")
collection = client.get_collection(name="personal_knowledge")

def ask_my_ai(question):
    print(f"User Query: {question}")
    print("Database mein answer search kar raha hoon...\n")
    
    # 2. Sawal ko vector mein convert kar
    query_embedding = ollama.embeddings(
        model="nomic-embed-text", 
        prompt=question
    )["embedding"]
    
    # 3. Database mein sabse closely matching 2 facts dhoondh
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2 # Top 2 results
    )
    
    # Retrieved facts ko ek string mein jod
    retrieved_context = "\n".join(results['documents'][0])
    
    # 4. LLM ke liye final prompt taiyaar kar
    # Hum model ko strictly bol rahe hain ki sirf is context ke basis par jawab de
    final_prompt = f"""
    You are a highly intelligent and accurate AI assistant.
    Use the following pieces of context to answer the user's question. 
    If the answer is not in the context, just say that you don't know, don't try to make up an answer.
    
    Context:
    {retrieved_context}
    
    Question: {question}
    
    Answer:
    """
    
    print("AI prompt process kar raha hai...\n")
    
    # 5. Llama3 ko final prompt bhej
    response = ollama.chat(model='llama3', messages=[
        {
            'role': 'user',
            'content': final_prompt
        }
    ])
    
    print("--- AI RESPONSE ---")
    print(response['message']['content'])
    print("-------------------")

if __name__ == "__main__":
    # Ab AI se apne baare mein sawal pooch
    my_question = "Who is Piyush and what is the current turnover of his business?"
    ask_my_ai(my_question)