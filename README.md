# Flask Application (`flask_app`)

This directory contains the core application logic, database models, and routes for the Pooja Store ecommerce application.

## Structure

- `app.py`: The main entry point for the Flask application. It configures the app, sets up the SQLite database using Flask-SQLAlchemy, and defines all the HTTP routes.
- `pooja_store.db`: SQLite database file (created automatically on first run) containing user and product data.
- `requirements.txt`: Python package dependencies.
- `setup_pinecone.py`: Script to initialize the Pinecone vector index for semantic search functionality.
- `supabase_schema.sql`: SQL for the Supabase `accounts` table used by login/register flows.
- `rag/`: Submodule handling the Retrieval-Augmented Generation functionality.
- `templates/`: Contains HTML templates for rendering the frontend views.
- `static/`: Contains static assets like images, CSS, and JS. Admin product uploads are stored in Supabase Storage when Supabase credentials are configured.

## Database Configuration

Products and settings still use the local SQLite database. On hosted WSGI servers such as Render, the app initializes those SQLite tables before the first request. Login/register/account credential records use the Supabase `accounts` table by default, so missing Supabase credentials will fail account reads/writes instead of silently saving credentials to SQLite.

Before running with Supabase logins, execute `supabase_schema.sql` in the Supabase SQL editor. Use a server-side key that is allowed to read and write the `accounts` table; do not expose that key in browser code.

For local-only testing, set `ACCOUNT_STORE=local` or run with Flask `TESTING=True` to use the SQLite `user` table instead.

Admin panel image uploads are written to the Supabase Storage bucket named by `SUPABASE_STORAGE_BUCKET` or `SUPABASE_BUCKET`. If neither is set, the app uses `product-images`. The app will try to create that bucket as public the first time an image is uploaded.

## Running the App

```bash
cd flask_app
python app.py
```
