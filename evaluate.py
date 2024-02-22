from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from typing import List, Tuple
from processing import search_database, generate_response_with_context
from dotenv import load_dotenv
from openai import OpenAI
import openai
import os

# Define the functions to generate responses using RAG 
def generate_rag_response(prompt: str) -> str:
    db_path = 'rag_database.db'  # Ensure this is correctly pointing to your database
    context_chunks = search_database(prompt, db_path)

    if not context_chunks:  # In case no relevant chunks were found
        return "Sorry, I don't have enough information to answer that question."
    
    response = generate_response_with_context(prompt, context_chunks)
    return response

# Define the function to generate responses using ChatGPT
def generate_chatgpt_response(prompt: str) -> str:
    load_dotenv()
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system", "content": "You are a helpful wilderness survival assistant."
            },
            {
                "role": "user", "content": prompt
            }
        ]
    )
    return response.choices[0].message['content']

# Define the function to evaluate the models
def evaluate_models(prompts: List[str], reference_answers: List[List[str]]) -> Tuple[float, float]:
    rag_responses = [generate_rag_response(prompt) for prompt in prompts]
    chatgpt_responses = [generate_chatgpt_response(prompt) for prompt in prompts]

    # Prepare data for BLEU evaluation
    rag_references = [[ref.split() for ref in refs] for refs in reference_answers]  # Tokenized reference answers
    rag_candidates = [response.split() for response in rag_responses]  # Tokenized generated responses

    # Calculate BLEU scores
    bleu_rag = corpus_bleu(rag_references, rag_candidates)
    bleu_chatgpt = corpus_bleu(rag_references, [response.split() for response in chatgpt_responses])

    # Prepare ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    # Calculate ROUGE scores
    rouge_scores_rag = [scorer.score(refs[0], response) for refs, response in zip(reference_answers, rag_responses)]
    rouge_scores_chatgpt = [scorer.score(refs[0], response) for refs, response in zip(reference_answers, chatgpt_responses)]

    # Average ROUGE scores
    avg_rouge_rag = {key: sum([score[key].fmeasure for score in rouge_scores_rag]) / len(rouge_scores_rag) for key in rouge_scores_rag[0]}
    avg_rouge_chatgpt = {key: sum([score[key].fmeasure for score in rouge_scores_chatgpt]) / len(rouge_scores_chatgpt) for key in rouge_scores_chatgpt[0]}

    return (bleu_rag, bleu_chatgpt), (avg_rouge_rag, avg_rouge_chatgpt)

# Usage
prompts = ["Your prompts here"]
reference_answers = [["Your reference answers here"]]
bleu_scores, rouge_scores = evaluate_models(prompts, reference_answers)
print("BLEU Scores:", bleu_scores)
print("ROUGE Scores:", rouge_scores)
