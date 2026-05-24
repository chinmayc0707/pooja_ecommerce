# RAG Module (`flask_app/rag`)

This directory contains the Retrieval-Augmented Generation (RAG) module used to power semantic product searches within the application.

## Overview

The RAG module integrates Langchain and Pinecone vector databases to provide an intelligent search experience. It takes natural language queries, converts them into embeddings, and retrieves the most relevant products from the catalog.

## Files

- `__init__.py`: Module initialization.
- `embeddings.py`: Handles connection to the embedding provider (e.g., HuggingFace, OpenAI) to generate vector representations of text.
- `indexer.py`: Responsible for chunking and formatting product descriptions to upload to Pinecone.
- `rag_engine.py`: The core engine that processes user queries, retrieves similar vector matches from Pinecone, and formats the output for the application.

## Setup

Ensure you have configured `PINECONE_API_KEY` and `PINECONE_INDEX_NAME` in your `.env` file before running any RAG functionality or `setup_pinecone.py`.
