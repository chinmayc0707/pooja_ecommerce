# Pooja Store Ecommerce

A comprehensive ecommerce application for a Pooja store, built with Flask and Python.
This application features a full shopping experience, including product cataloging, user authentication (with JWT and session management), an admin dashboard, a shopping cart, and a unique Retrieval-Augmented Generation (RAG) powered search capability using Pinecone vector embeddings.

## Features

- **User Authentication:** Secure login and registration utilizing JWT.
- **Product Management:** Browse, view details, and manage inventory (Admin only).
- **Shopping Cart:** Add, update, and remove products.
- **Smart Search (RAG):** AI-powered semantic search integration using Langchain and Pinecone vector database to help users find products easily.
- **Admin Dashboard:** Access control for administrative actions.

## Prerequisites

- Python 3.8+
- Pinecone Account (for vector database)
- Optional: API keys (OpenAI, Google, etc.) depending on the chosen embedding model.

## Installation Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/chinmayc0707/pooja_ecommerce.git
   cd pooja_ecommerce
   ```

2. **Create a virtual environment (recommended):**
   Create a standard python virtual environment, e.g. using the venv module.
   Activate the virtual environment.

3. **Install the dependencies:**
   Navigate into the `flask_app` directory (where `requirements.txt` is located) and install:
   ```bash
   cd flask_app
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the `flask_app` directory with the following required keys:
   ```env
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_INDEX_NAME=pooja-store
   ```

5. **Initialize the Vector Database:**
   Run the Pinecone setup script to create the index and upload initial product embeddings:
   ```bash
   python setup_pinecone.py
   ```

6. **Run the Application:**
   Start the Flask development server:
   ```bash
   flask run
   # Or using python
   python app.py
   ```
   The application will be accessible at `http://127.0.0.1:5000/`.
