import fitz  # PyMuPDF
import sqlite3
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
import openai
from dotenv import load_dotenv
import os


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def setup_database(db_path='rag_database.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # Add an 'embedding' column to store the text chunk embeddings
    c.execute('''
        CREATE TABLE IF NOT EXISTS text_chunks (
            id INTEGER PRIMARY KEY,
            chunk TEXT,
            embedding TEXT
        )
    ''')
    conn.commit()
    conn.close()
    
model = SentenceTransformer('all-MiniLM-L6-v2')

def insert_chunks_to_database(chunks, db_path='rag_database.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    for chunk in chunks:
        # Generate embedding for the chunk
        embedding = model.encode(chunk, convert_to_tensor=True)
        # Convert the embedding tensor to a list for storage
        embedding_str = ','.join(map(str, embedding.tolist()))
        # Insert both the chunk and its embedding into the database
        c.execute("INSERT INTO text_chunks (chunk, embedding) VALUES (?, ?)", (chunk, embedding_str))
    conn.commit()
    conn.close()

def search_database(query, db_path='rag_database.db', top_k=5):
    # Encode the query to get its embedding
    query_embedding = model.encode(query, convert_to_tensor=True).tolist()

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT id, embedding FROM text_chunks")
    results = c.fetchall()
    conn.close()

    # Calculate cosine similarity between query embedding and each stored embedding
    similarities = []
    for id, embedding_str in results:
        stored_embedding = np.fromstring(embedding_str, sep=',')
        cos_sim = np.dot(query_embedding, stored_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding))
        similarities.append((cos_sim, id))

    # Sort by similarity
    similarities.sort(reverse=True)
    top_k_ids = [id for _, id in similarities[:top_k]]

    # Fetch the top K chunks based on their IDs
    top_k_chunks = []
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    for id in top_k_ids:
        c.execute("SELECT chunk FROM text_chunks WHERE id = ?", (id,))
        top_k_chunks.append(c.fetchone()[0])
    conn.close()

    return top_k_chunks

def generate_response_with_context(prompt, context_chunks, max_tokens=150):
    # Craft the prompt with context for actionable wilderness survival tips
    prompt_intro = "Provide practical wilderness survival tips based on the following key points. Focus on actionable advice:"
    combined_prompt = f"{prompt_intro} {prompt} {' '.join(context_chunks)}"

    load_dotenv()
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system", "content": "You are a helpful wilderness survival assistant.",
                "role": "user", "content": combined_prompt,
            }
        ]
    )

    return response.choices[0].message.content