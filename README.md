# Wilderness Medical Self-Survival Guide

This project uses Retrieval Augmented Generation to assist individuals with wilderness survival by providing medical advice. It includes a web interface for easy access to medical information, a script for processing PDFs to populate a database, and an evaluation script to compare the performance of two different language models.

## Project Constraints and Methodology

In developing this application, specific constraints were adhered to ensure a custom approach:

- **No Use of RAG Libraries**: The implementation does not rely on any existing libraries specifically designed for RAG tasks, such as LangChain or others. All pipelines were written from scratch in Python.

- **Text Data Extraction**: While the extraction of text data from PDFs is performed using existing parsers like `pdfplumber` or `PyMuPDF` (as seen in `processing.py`), no specialized library was used for the RAG functionality.

- **Chunking Algorithm**: Includes custom-written code for chunking the text data into manageable pieces, which is essential for the retrieval process and for providing context to the language model.

- **Database Storage**: The system uses `sqlite3` database to store and retrieve data. The application handles the storage and extraction of chunked text and embeddings directly through SQL commands.

- **Semantic Search and Retrieval**: The retrieval process is manually implemented. The application computes the semantic similarity (cosine similarity) between the query and stored text embeddings to identify the most relevant text chunks without relying on a library to perform this task.

- **Language Model Prompting**: The application integrates the relevant context into prompts for a language model (LLM), ChatGPT-4, ensuring that the generated advice is pertinent to the user's query.

- **User Interface Integration**: The generated response from the LLM is then presented through a user-friendly interface, allowing for interactive querying and response display.

## Detailed Functionality

- `processing.py`: Handles the extraction of text from PDFs, chunking of text, storage of text chunks and embeddings in a SQLite database, and performs semantic search to retrieve relevant text chunks for user queries. Embeddings were created using `thenlper/gte-small` found on Hugging Face.

- `main.py`: Serves as the entry point for processing PDFs to populate the database with text chunks and their corresponding embeddings.

- `app.py`: A Streamlit web application that provides the front-end for users to input medical questions and receive advice.

- `evaluate.py`: Contains the evaluation framework that compares the custom RAG system's performance with OpenAI's ChatGPT-4 model alone, using ROUGE.

- `requirements.txt`: Includes the dependencies required to run the application.

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

<img width="1440" alt="Screenshot 2024-02-22 at 1 01 00 AM" src="https://github.com/tommymmcguire/IA2_590/assets/141086024/ebb225a8-5f9c-46c2-b7e3-14eaca1ee09b">


### Model Evaluation

To evaluate the models, run `evaluate.py`. The script will output the performance scores of the RAG system and ChatGPT, providing insights into their effectiveness in generating relevant and accurate responses. The evaluation is based on the quality of responses these models generate in response to a series of prompts related to wilderness survival. Each time `evaluate.py` is run, new scores are calculated based on the output of the RAG model and ChatGPT4. 

#### Metrics for Scoring

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) Scores: These metrics compare the overlap of n-grams between the generated text and the reference texts. The `rouge1` score measures the overlap of individual words or unigrams, while `rougeL` score accounts for pairs of words or bigrams, which can indicate the fluency and coherence of the generated responses. 

#### Performance

<img width="605" alt="Screenshot 2024-02-22 at 12 47 13 AM" src="https://github.com/tommymmcguire/IA2_590/assets/141086024/4ccbd2f0-186a-4dad-822c-e457b1d68dd1">

**Walk Through Youtube Video**
[YouTube](https://youtu.be/2JvB30ilFV8)


