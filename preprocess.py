import fitz  # PyMuPDF
import sqlite3
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


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
    c.execute('''CREATE TABLE IF NOT EXISTS text_chunks
                 (id INTEGER PRIMARY KEY, chunk TEXT)''')
    conn.commit()
    conn.close()
    
def insert_chunks_to_database(chunks, db_path='rag_database.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    for chunk in chunks:
        c.execute("INSERT INTO text_chunks (chunk) VALUES (?)", (chunk,))
    conn.commit()
    conn.close()
    
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def search_database(query, db_path='rag_database.db', top_k=5):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT id, chunk FROM text_chunks")
    results = c.fetchall()
    conn.close()

    query_embedding = sentence_model.encode([query])[0]
    scores = []

    for id, chunk in results:
        chunk_embedding = sentence_model.encode([chunk])[0]
        score = np.dot(query_embedding, chunk_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding))
        scores.append((score, id, chunk))

    scores.sort(reverse=True, key=lambda x: x[0])
    return [chunk for score, id, chunk in scores[:top_k]]

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

def generate_response_with_context(user_question, context_chunks, max_length=150):
    # Engineering the prompt to direct the model's output
    prompt_engineering_intro = "Provide practical advice based on the following information. Focus on actionable advice:"
    
    # Combining the engineered prompt with the user's question and context
    # This helps in guiding the model to generate the type of response we're seeking
    combined_prompt = f"{prompt_engineering_intro} {user_question} {' '.join(context_chunks)}"
    
    input_ids = tokenizer.encode(combined_prompt, return_tensors='pt')
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        temperature=0.9,  # Adjust to control creativity
        top_k=50,  # Controls diversity
        top_p=0.95,  # Nucleus sampling for randomness
        repetition_penalty=1.2,  # Penalize repetition
        num_return_sequences=1  # Generate one sequence
    )
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return generated_text