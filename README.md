## RAG-Based Document Query App

# Overview

This project is a Retrieval-Augmented Generation (RAG) based document query system built using Streamlit, FAISS, Sentence Transformers, and OpenAI API. It allows users to upload a document (PDF, DOCX, TXT) and perform semantic search-based querying on the extracted text.

# Features

Supports PDF, Word (.docx), and Text (.txt) file uploads.

Extracts text from uploaded documents.

Chunks and embeds text using Sentence Transformers.

Stores embeddings in a FAISS vector index for efficient retrieval.

Queries FAISS for semantic search and retrieves relevant document chunks.

Uses OpenAI's API to generate responses based on retrieved context.

# Installation

Install dependencies using:

pip install streamlit PyPDF2 faiss-cpu sentence-transformers openai python-docx

# Usage

Run the Streamlit app:

streamlit run app.py

# How It Works

Upload a document (PDF, Word, or TXT).

The system extracts and preprocesses the text.

Text is chunked and embedded using Sentence Transformers.

FAISS stores embeddings for fast retrieval.

User enters a query, which is also embedded and searched in FAISS.

The most relevant chunks are retrieved and passed to OpenAI's API for generating an answer.

# Query Example

User: "What is the main topic of this document?"

System: "The document discusses the effects of climate change on agriculture..."

# Future Improvements

Implement multi-file support for cross-document search.

Add metadata-based filtering for enhanced retrieval.

Integrate custom language models for offline processing.
