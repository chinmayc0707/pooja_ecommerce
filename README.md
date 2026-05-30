# Flask Application (`flask_app`)

This directory contains the core application logic, database models, and routes for the Pooja Store ecommerce application.

## Structure

- `app.py`: The main entry point for the Flask application. It configures the app, sets up the SQLite database using Flask-SQLAlchemy, and defines all the HTTP routes.
- `pooja_store.db`: SQLite database file (created automatically on first run) containing user and product data.
- `requirements.txt`: Python package dependencies.
- `setup_pinecone.py`: Script to initialize the Pinecone vector index for semantic search functionality.
- `rag/`: Submodule handling the Retrieval-Augmented Generation functionality.
- `templates/`: Contains HTML templates for rendering the frontend views.
- `static/`: Contains static assets like images, CSS, and JS (uploads go here).

## Database Configuration

The application utilizes SQLite by default. Ensure that `pooja_store.db` is read/write accessible by the application.

## Running the App

```bash
cd flask_app
python app.py
```

## Supabase Storage

To upload product images to Supabase instead of the local filesystem, add the following variables to your `.env` file:

```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_api_key
SUPABASE_BUCKET=uploads
```

Ensure the specified bucket (`uploads` by default) exists in your Supabase storage and is public.
