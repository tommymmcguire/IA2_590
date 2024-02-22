import streamlit as st
from processing import search_database, generate_response_with_context

# Assuming the database setup and PDF processing is already done
db_path = 'rag_database.db'  # Path to your database

st.title('Wilderness Medical Self-Surivial Guide')

# Input for user question
user_question = st.text_input("Please ask any medical question:")

if user_question:
    # Search the database for relevant context chunks based on the user's question
    context_chunks = search_database(user_question, db_path)

    if context_chunks:  # If relevant chunks were found
        response = generate_response_with_context(user_question, context_chunks)
        st.write("Response:", response)
    else:  # In case no relevant chunks were found
        st.error("Sorry, I don't have enough information to answer that question.")