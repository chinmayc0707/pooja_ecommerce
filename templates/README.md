# Templates (`flask_app/templates`)

This directory contains the HTML templates for the Pooja Store frontend. The application uses Jinja2 templating (via Flask) to dynamically render these pages with data from the backend.

## Available Templates

- `index.html`: The main landing page displaying the product catalog and search interface (including the RAG AI search capability).
- `admin.html`: The dashboard interface for site administrators to manage products (add, edit, delete) and view inventory.
- `cart.html`: The shopping cart page where users can view their selected items, update quantities, and proceed to checkout.
- `login.html`: The user authentication page for existing users to log into their accounts.
- `register.html`: The sign-up page for new users to create an account.
- `product_detail.html`: A detailed view page for individual products, showing larger images, full descriptions, and add-to-cart options.

## Styling and Assets

While the structure is defined here in the HTML files, associated CSS styling and client-side JavaScript (if any) are typically served from the `static/` directory.
