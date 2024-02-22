# Wilderness Medical Self-Survival Guide

This project uses Retrieval Augmented Generation to assist individuals with wilderness survival by providing medical advice. It includes a web interface for easy access to medical information, a script for processing PDFs to populate a database, and an evaluation script to compare the performance of two different language models.

## Project Structure

- `app.py`: A Streamlit web application that provides a user interface for asking medical questions and receiving advice.
- `evaluate.py`: A script that evaluates the performance of the custom Retrieval-Augmented Generation (RAG) system against OpenAI's ChatGPT model using BLEU and ROUGE metrics.
- `main.py`: The main script that processes a directory of PDFs to extract text, chunk it, and insert it into a SQLite database.
- `processing.py`: Contains all the processing functions, including PDF text extraction, text chunking, database operations, and response generation.
- `requirements.txt`: Dependencies required to run.
- `pdfs` directory: contains the pdfs used to retrieve the information.

## Features

- Extract information from PDF documents and store it in a structured database.
- Generate context-aware responses to user queries using the RAG system.
- Evaluate the performance of the RAG system in comparison to ChatGPT.
- Provide a user-friendly web interface to interact with the system.

## Setup

1. Clone the repository to your local machine.
2. Install the required dependencies by running `make install`.
3. Obtain an OpenAI API key and store it in a `.env` file in the root directory. The `.env` file should look like this:
```
# Once you add your API key below, make sure to not share it with anyone! The API key should remain private.
OPENAI_API_KEY=abc123
```
4. Run the `main.py` script to process PDFs and populate the database.
5. Start the web application by running `streamlit run app.py`.

## Usage

### Web Interface

To use the web interface, run `streamlit run app.py` and navigate to the URL provided by Streamlit. You can then ask medical questions related to wilderness survival, and the system will provide you with advice based on the information stored in the database.

### Model Evaluation

To evaluate the models, run `evaluate.py`. The script will output the performance scores of the RAG system and ChatGPT, providing insights into their effectiveness in generating relevant and accurate responses. The evaluation is based on the quality of responses these models generate in response to a series of prompts related to wilderness survival. Each time `evaluate.py` is run, new scores are calculated based on the output of the RAG model and ChatGPT4. 

#### Metrics for Scoring

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) Scores: These metrics compare the overlap of n-grams between the generated text and the reference texts. The `rouge1` score measures the overlap of individual words or unigrams, while `rougeL` score accounts for pairs of words or bigrams, which can indicate the fluency and coherence of the generated responses. 

#### Performance

<img width="644" alt="Screenshot 2024-02-21 at 11 27 23 PM" src="https://github.com/tommymmcguire/IA2_590/assets/141086024/b0f3b3a2-69d0-46ee-8996-31b231d49ea0">


