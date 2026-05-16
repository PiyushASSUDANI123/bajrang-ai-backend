import chromadb
import ollama

# 1. ChromaDB ko tere folder mein ek local database banane ko bol
client = chromadb.PersistentClient(path="./my_vector_db")

# 2. Ek 'collection' bana (jaise SQL mein table hoti hai)
collection = client.get_or_create_collection(name="personal_knowledge")

# 3. Tera custom data (jo hum AI ko sikhana chahte hain)
# Tu yahan Nupost, Atteni, ya apne skills ke baare mein kuch bhi daal sakta hai
my_facts = [
    "Piyush Assudani is a 16-year-old developer and entrepreneur.",
    "Piyush is the founder and CEO of Assudani Developers and owns the brand Loyalto.",
    "Assudani Developers recently reached a business turnover milestone of 45000 rupees.",
    "Piyush is currently in Class 12 and studies at Delhi Public School in Balotra."
]

print("Converting text to vectors and saving to database... (Math ho raha hai backend mein)")

# 4. Har fact ko vector mein convert karke database mein save kar
for i, fact in enumerate(my_facts):
    # Embedding model text ko math coordinates mein convert kar raha hai
    response = ollama.embeddings(model="nomic-embed-text", prompt=fact)
    embedding = response["embedding"]
    
    # Vector aur original text ko ChromaDB mein store kar
    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[fact]
    )

print("Database successfully created aur data save ho gaya!")