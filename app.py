import streamlit as st
import PyPDF2
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from docx import Document  # For handling Word files

# Initialize OpenAI API
api_key = "f614cad0fabe42bd8f287a921066b771"  # Replace with your API key
base_url = "https://api.aimlapi.com/v1"
api = OpenAI(api_key=api_key, base_url=base_url)

# Initialize Sentence Transformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS Index
dimension = 384  # Embedding dimension of the model
index = faiss.IndexFlatL2(dimension)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from Word file
def extract_text_from_word(word_file):
    document = Document(word_file)
    text = ""
    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"
    return text

# Function to extract text from Text file
def extract_text_from_txt(txt_file):
    return txt_file.read().decode("utf-8")

# Function to chunk text
def chunk_text(text, max_length=500):
    words = text.split()
    chunks = []
    chunk = []
    for word in words:
        if len(" ".join(chunk)) + len(word) <= max_length:
            chunk.append(word)
        else:
            chunks.append(" ".join(chunk))
            chunk = [word]
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

# Function to embed text and add to FAISS index
def embed_and_store(chunks):
    embeddings = embedding_model.encode(chunks)
    index.add(embeddings)

# Query handling
def query_llm(prompt):
    completion = api.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=256,
    )
    return completion.choices[0].message.content

# Streamlit App
st.title("RAG-based Document Query App")
uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx", "txt"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()

    # Handle file based on type
    if file_type == "pdf":
        text = extract_text_from_pdf(uploaded_file)
        st.write("PDF Extracted Successfully!")
    elif file_type == "docx":
        text = extract_text_from_word(uploaded_file)
        st.write("Word Document Extracted Successfully!")
    elif file_type == "txt":
        text = extract_text_from_txt(uploaded_file)
        st.write("Text File Extracted Successfully!")
    else:
        st.error("Unsupported file type. Please upload a PDF, Word document, or Text file.")
        text = ""

    # Process the text if extraction was successful
    if text:
        # Chunk and embed text
        chunks = chunk_text(text)
        embed_and_store(chunks)
        st.write(f"{len(chunks)} chunks added to the FAISS index.")

        # Query Interface
        user_query = st.text_input("Ask a question about the document:")
        if user_query:
            # Embed query and search FAISS
            query_embedding = embedding_model.encode([user_query])
            distances, indices = index.search(query_embedding, k=5)  # Top 5 results
            relevant_chunks = [chunks[i] for i in indices[0]]

            # Combine chunks for context
            context = " ".join(relevant_chunks)
            final_prompt = f"Context: {context}\n\nQuestion: {user_query}"

            # Get response from OpenAI API
            response = query_llm(final_prompt)
            st.write("### Answer")
            st.write(response)
