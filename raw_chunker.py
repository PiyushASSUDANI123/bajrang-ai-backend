def load_and_chunk_text(file_path, chunk_size=30, overlap=10):
    # 1. File ko padh
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            raw_text = file.read()
    except FileNotFoundError:
        return "Bhai, pehle company_data.txt file toh bana le."

    # 2. Text ko words mein tod
    words = raw_text.split()
    chunks = []
    
    # 3. Raw Chunking Logic (Yeh hai tera core foundation)
    start = 0
    while start < len(words):
        # End index nikal
        end = start + chunk_size
        
        # Words ko wapas string mein jod
        chunk_text = " ".join(words[start:end])
        chunks.append(chunk_text)
        
        # Agle chunk ke liye aage badh, par 'overlap' jitna peeche reh kar
        start += (chunk_size - overlap)
        
    return chunks

if __name__ == "__main__":
    print("Reading and chunking raw document...\n")
    
    # Hum 30 words ka chunk bana rahe hain, jisme 10 words overlap honge
    my_chunks = load_and_chunk_text("company_data.txt", chunk_size=30, overlap=10)
    
    print(f"Total chunks created: {len(my_chunks)}\n")
    
    # Chunks ko print karke dekh
    for i, chunk in enumerate(my_chunks):
        print(f"--- Chunk {i+1} ---")
        print(chunk)
        print("------------------\n")