from rag_pipeline import answer_query, retrieve_docs, llm_model
import streamlit as st
import os
from vectorDataBase import uploadPDF, loadPDF, create_chunks, FAISS_DB_PATH, FAISS, get_embeddings_model

UPLOAD_FOLDER = "uploaded_pdfs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

uploadFile = st.file_uploader("Upload the PDF",
                              type="pdf",
                              accept_multiple_files=False
                              )

user_query = st.text_area("Enter your prompt: ", height=150, placeholder="Ask me anything.")
ask_question = st.button("Ask the AI Lawyer")

if ask_question:
    if uploadFile:
        st.chat_message("user").write()
        
        # Save the uploaded PDF in a designated folder
        pdf_path = os.path.join(UPLOAD_FOLDER, "current_document.pdf")
        with open(pdf_path, "wb") as f:
            f.write(uploadFile.getbuffer())
        
        # Load the newly uploaded PDF
        documents = loadPDF(pdf_path)
        
        # Process the PDF into text chunks
        text_chunks = create_chunks(documents)
        
        # Update the FAISS database with the new document
        faiss_db = FAISS.from_documents(text_chunks, get_embeddings_model("deepseek-r1:7b"))
        faiss_db.save_local(FAISS_DB_PATH)
        
        # Reload the FAISS index to use the newly updated database
        faiss_db = FAISS.load_local(FAISS_DB_PATH, get_embeddings_model("deepseek-r1:7b"), allow_dangerous_deserialization=True)
        
        retrieved_docs = faiss_db.similarity_search(user_query)  # Retrieve documents from the updated FAISS DB
        response = answer_query(documents=retrieved_docs, model=llm_model, query=user_query)
        
        st.chat_message("AI Lawyer").write(response)
    else:
        st.error("Kindly upload a valid PDF file.")
