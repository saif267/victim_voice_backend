import os
import pickle
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from langdetect import detect
from fastapi.middleware.cors import CORSMiddleware


# Load API Key
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    print("CRITICAL ERROR: OPENAI_API_KEY not found. Check .env file.")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load Database
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.pkl"

print("Loading Knowledge Base...")
try:
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        chunks_metadata = pickle.load(f)
    print("✅ Knowledge Base Loaded.")
except Exception as e:
    print(f"❌ Error loading database: {e}")
    print("Run 'ingest.py' first!")
    index = None
    chunks_metadata = []

app = FastAPI()
# Allow requests from your Flutter app (Android emulator + phones)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # For dev, allow everything
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    query: str

class Answer(BaseModel):
    original_query: str
    legal_search_terms: str  # We will show you what the bot "thought"
    answer: str

def get_openai_embedding(text):
    response = client.embeddings.create(
        input=text.replace("\n", " "),
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# --- NEW: THE JUNIOR LAWYER FUNCTION ---
def refine_query_to_legal_terms(user_query):
    """
    Translates a layman's story into legal keywords for better searching.
    """
    system_prompt = """
    You are a legal search assistant for the Pakistan Penal Code.
    The user will describe a crime in simple words.
    Your job is to translate this description into LEGAL KEYWORDS and POTENTIAL SECTIONS.
    
    Do not answer the question. Just output a list of relevant legal terms and sections that we should search for in the database.
    
    Example:
    User: "Someone stole my car at gunpoint."
    Output: "Robbery Section 390 392, Dacoity, Theft, Extortion, putting person in fear of death Section 386 387"
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except:
        return user_query # If it fails, just use the original query

@app.post("/ask", response_model=Answer)
async def ask(user_query: Query):
    if index is None:
        raise HTTPException(status_code=503, detail="System not ready. Run ingest.py.")

    original_query = user_query.query
    processed_query = original_query

    # 1. Language Detection (Urdu -> English)
    try:
        if detect(original_query) == 'ur':
            print("Urdu detected. Translating...")
            trans_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "Translate to English:"}, 
                          {"role": "user", "content": original_query}]
            )
            processed_query = trans_response.choices[0].message.content.strip()
    except:
        pass 

    # 2. CONVERT LAYMAN STORY TO LEGAL TERMS (The Magic Step)
    print(f"Original English Query: {processed_query}")
    legal_search_query = refine_query_to_legal_terms(processed_query)
    print(f"--- Generated Search Terms: {legal_search_query} ---")

    # 3. Search Vector DB using the LEGAL TERMS (Not the story)
    query_vector = np.array([get_openai_embedding(legal_search_query)]).astype('float32')
    
    # k=10 to get enough context
    distances, indices = index.search(query_vector, k=10) 

    # 4. Retrieve Context
    context_text = ""
    print("\n--- DEBUG: WHAT THE BOT FOUND ---")
    for idx in indices[0]:
        if idx < len(chunks_metadata):
            chunk = chunks_metadata[idx]['text']
            source = chunks_metadata[idx]['source']
            # Print snippets to terminal to verify
            print(f"Found in {source}: {chunk[:50]}...") 
            context_text += f"Source: {source}\nContent: {chunk}\n\n"
    print("-----------------------------------\n")

    # 5. Generate Answer
    system_prompt = """
    You are 'Victim Voice', a legal assistant for Pakistan. 
    Answer the user's question based ONLY on the provided Context.
    
    1. Identify the crime described (e.g., Robbery, Theft, Extortion).
    2. Cite the specific Section number from the context (e.g., Section 392).
    3. Explain the punishment mentioned in the text.
    
    If the answer is truly not in the Context, say "I cannot find relevant legal information."
    
    Always end with: 
    "Disclaimer: This is for informational purposes only and is not legal advice."
    """
    
    user_message = f"Context:\n{context_text}\n\nUser Question: {processed_query}"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0
    )

    return Answer(
        original_query=original_query,
        legal_search_terms=legal_search_query,
        answer=response.choices[0].message.content
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)