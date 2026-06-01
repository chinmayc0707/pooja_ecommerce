# Flask Application (`flask_app`)

This directory contains the core application logic, database models, and routes for the Pooja Store ecommerce application.

## Structure

- `app.py`: The main entry point for the Flask application. It configures Supabase-backed account, product, and settings storage and defines all HTTP routes.
- `requirements.txt`: Python package dependencies.
- `setup_pinecone.py`: Script to initialize the Pinecone vector index for semantic search functionality.
- `supabase_schema.sql`: SQL for the Supabase `accounts`, `products`, and `settings` tables used by the app.
- `rag/`: Submodule handling the Retrieval-Augmented Generation functionality.
- `templates/`: Contains HTML templates for rendering the frontend views.
- `static/`: Contains static assets like images, CSS, and JS. Admin product uploads are stored in Supabase Storage when Supabase credentials are configured.

## Database Configuration

Accounts, products, and settings use Supabase by default. The app does not create or depend on a SQLite `.db` file.

Before running on Render, execute `supabase_schema.sql` in the Supabase SQL editor. Use a server-side key that is allowed to read and write the `accounts`, `products`, and `settings` tables; do not expose that key in browser code.

For local-only testing, set `ACCOUNT_STORE=local` and `DATA_STORE=local`, or run with Flask `TESTING=True`, to use SQLite instead.

Admin panel image uploads are written to the Supabase Storage bucket named by `SUPABASE_STORAGE_BUCKET` or `SUPABASE_BUCKET`. If neither is set, the app uses `product-images`. The app will try to create that bucket as public the first time an image is uploaded.

## Running the App

```bash
cd flask_app
python app.py
```
