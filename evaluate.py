from rouge_score import rouge_scorer
from typing import List, Tuple
from processing import search_database, generate_response_with_context
from dotenv import load_dotenv
from openai import OpenAI
import openai
import os

# Define the functions to generate responses using RAG 
def generate_rag_response(prompt: str) -> str:
    db_path = 'rag_database.db'  
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
    return response.choices[0].message.content

# Define the function to evaluate the models
def evaluate_models(prompts: List[str], reference_answers: List[List[str]]) -> Tuple[float, float]:
    rag_responses = [generate_rag_response(prompt) for prompt in prompts]
    chatgpt_responses = [generate_chatgpt_response(prompt) for prompt in prompts]

    # Prepare ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    # Calculate ROUGE scores
    rouge_scores_rag = [scorer.score(refs[0], response) for refs, response in zip(reference_answers, rag_responses)]
    rouge_scores_chatgpt = [scorer.score(refs[0], response) for refs, response in zip(reference_answers, chatgpt_responses)]

    # Average ROUGE scores
    avg_rouge_rag = {key: sum([score[key].fmeasure for score in rouge_scores_rag]) / len(rouge_scores_rag) for key in rouge_scores_rag[0]}
    avg_rouge_chatgpt = {key: sum([score[key].fmeasure for score in rouge_scores_chatgpt]) / len(rouge_scores_chatgpt) for key in rouge_scores_chatgpt[0]}

    return avg_rouge_rag, avg_rouge_chatgpt

# Usage
prompts = [
    "How do I know if I am having an appendicitis?",
    "How do I know if I am having a heat stroke and what should I do?",
    "How should a woman give birth in the wild?",
    "What should you do if someone has a seizure?",
    "What should I do for a bad burn?"
]

reference_answers = [
    [
        "Have the person cough and see if this causes sharp pain in the belly. "
        "Slowly but forcefully, press on the abdomen a little above the left groin until it hurts a little. "
        "Then quickly remove the hand."
        "If a very sharp pain occurs when the hand is removed, appendicitis is likely. "
        "If no rebound pain occurs above the left groin, try the same test above the right groin."
    ],
    [
        "Signs: The skin is red, very hot, and dry. High fever, sometimes more than 42°C, and a rapid heartbeat. Often he is unconscious. "
        "Treatment: The body temperature must be lowered immediately. Put the person in the shade. Soak him with cold water (ice water if possible) and fan him. Continue until the fever drops. Seek medical help."
    ],
    [
        "The mom must lie down on her back, put two pillows under her bottom, bring her knees up to her chest, grab her knees, and push hard with each contraction. "
        "After the baby is born, place her or him on the mother's chest and tummy, skin to skin, and cover both with towels. "
        "If the baby is not crying, rub her back firmly."
    ],
    [
        "Only move them if they're in danger. Cushion their head if they're on the ground. Loosen any tight clothing around their neck, such as a collar or tie, to aid breathing. Turn them on to their side after their convulsions stop"
    ],
    [
        "Sterilize a little Vaseline by heating it until it boils. Let it cool and spread it on a piece of sterile gauze. Then put the gauze on the burn loosely so it does not put pressure on the wound. "
        "If there is no Vaseline, leave the burn uncovered. Never smear on grease or butter. "
        "If signs of infection appear—pus, bad smell, fever, or swollen lymph nodes—apply compresses of warm salt water (1 teaspoon salt to 1 liter water) 3 times a day. "
        "Boil both the water and cloth before use. With great care, remove the dead skin and flesh. "
        "You can spread on a little antibiotic ointment such as Neosporin. In severe cases, consider taking an antibiotic such as dicloxacillin, clindamycin, or ciprofloxacin."
    ]
]
avg_rouge_rag, avg_rouge_chatgpt = evaluate_models(prompts, reference_answers)
print("RAG ROUGE Scores:", avg_rouge_rag)
print("ChatGPT ROUGE Scores:", avg_rouge_chatgpt)
