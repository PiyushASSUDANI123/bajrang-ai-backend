import requests
from bs4 import BeautifulSoup
from ddgs import DDGS  # Purana package name hata diya
import ollama

def search_internet(query, max_results=3):
    print(f"🌍 Live Internet par search kar raha hoon: '{query}'...")
    try:
        with DDGS() as ddgs:
            # backend="lite" use kiya hai taaki DuckDuckGo teri script ko bot samajh kar block na kare
            results = list(ddgs.text(query, max_results=max_results, backend="lite"))
        return results
    except Exception as e:
        print(f"⚠️ Search failed: {e}")
        return []

def scrape_website(url):
    print(f"🕸️ Scraping text from: {url}")
    try:
        # Browser jaisa pretend karne ke liye headers
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'}
        # Timeout zaroori hai warna ek kharab website tere poore code ko hang kar degi
        response = requests.get(url, headers=headers, timeout=5)
        
        # HTML content ko BeautifulSoup se parse kar
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Sirf paragraph <p> tags nikal (Ads, headers aur menus ignore kar)
        paragraphs = soup.find_all('p')
        text = " ".join([p.text for p in paragraphs])
        
        # M4 ki RAM aur LLM ki context limit bachane ke liye sirf pehle 1500 characters le rahe hain
        return text[:1500] 
    except Exception as e:
        print(f"⚠️ Website scrape nahi ho payi ({url}): {e}")
        return ""

def ask_live_ai(question, search_query):
    # 1. Search (Human question ko keywords mein convert karke search bhej rahe hain)
    search_results = search_internet(search_query)
    
    if not search_results:
        print("❌ Kuch nahi mila internet par. Bot bypass fail ho gaya ya internet nahi chal raha.")
        return

    # Debugging: Pata toh chale kahan se data utha raha hai
    print(f"\n🔗 Top Links Found: {[res['href'] for res in search_results]}\n")

    # 2. Scrape and Combine Data
    combined_context = ""
    for idx, result in enumerate(search_results):
        url = result['href']
        scraped_text = scrape_website(url)
        if scraped_text.strip():
            combined_context += f"\n[Source {idx+1}] ({url}):\n{scraped_text}\n"

    if not combined_context.strip():
        print("❌ Links mile par websites ne text read karne se block kar diya.")
        return

    print("\n🧠 Data mil gaya. Tera M4 Mac ab Llama 3 ko data feed kar raha hai...\n")

    # 3. Prompt Engineering (The Context Stuffing)
    final_prompt = f"""
    You are a real-time AI assistant (like Perplexity AI). 
    I will provide you with live data scraped from the internet. 
    Read the context below and answer the user's question accurately. 
    If the answer is not in the context, do not make it up. Just say you don't know based on the current context.
    Always mention the source numbers in your answer like [Source 1].
    
    Live Context:
    {combined_context}
    
    User Question: {question}
    
    Answer:
    """

    # 4. Generate Answer using Local LLM
    try:
        response = ollama.chat(model='llama3', messages=[
            {'role': 'user', 'content': final_prompt}
        ])
        
        print("\n================ LIVE AI RESPONSE ================")
        print(response['message']['content'])
        print("==================================================\n")
    except Exception as e:
        print(f"⚠️ LLM Inference mein error: {e}")

if __name__ == "__main__":
    # Human readable question
    user_question = "What are the latest features of the Apple M4 chip announced recently?"
    # Search engine optimized keywords
    optimized_query = "Apple M4 chip specs features announcement tech news 2024"
    
    ask_live_ai(user_question, optimized_query)