import os
def main():
    db_path = 'rag_database.db'
    setup_database(db_path)
    pdf_directory = './pdfs'
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text, chunk_size=500)  # Adjust chunk_size as needed
        insert_chunks_to_database(chunks, db_path)

    print("Finished processing PDFs and populating the database.")
    db_path = 'rag_database.db'  # Ensure this is correctly pointing to your database
    user_question = input("Please ask a question about wilderness survival: ")
    context_chunks = search_database(user_question, db_path)

    if not context_chunks:  # In case no relevant chunks were found
        print("Sorry, I don't have enough information to answer that question.")
        return

    response = generate_response_with_context(user_question, context_chunks)
    print("\nResponse:\n", response)

if __name__ == "__main__":
    main()