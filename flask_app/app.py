import os
import jwt
import datetime
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, make_response, flash, session, jsonify, abort
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

# Define absolute paths
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, 'templates')
db_path = os.path.join(base_dir, 'pooja_store.db')
UPLOAD_FOLDER = os.path.join(base_dir, 'static', 'uploads')
PRODUCT_IMAGE_PLACEHOLDER = 'product-placeholder.svg'

STARTER_PRODUCT_IMAGE_FILES = {
    "Pure Brass Diya": "ChatGPT_Image_May_27_2026_01_18_55_PM.png",
    "Organic Agarbatti": "ChatGPT_Image_May_27_2026_01_27_26_PM.png",
    "Puja Thali Set": "ChatGPT_Image_May_27_2026_01_22_05_PM.png",
    "Sandalwood Powder": "ChatGPT_Image_May_27_2026_01_22_55_PM.png",
    "Copper Kalash": "ChatGPT_Image_May_27_2026_01_26_05_PM.png",
    "Premium Camphor": "ChatGPT_Image_May_27_2026_01_24_00_PM.png",
}

app = Flask(__name__, template_folder=template_dir)
app.secret_key = os.environ.get('SECRET_KEY')
JWT_SECRET = os.environ.get('JWT_SECRET')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    is_admin = db.Column(db.Boolean, nullable=False, default=False)

class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    price = db.Column(db.Float, nullable=False)
    stock = db.Column(db.Integer, nullable=False)
    description = db.Column(db.Text, nullable=True)
    image_url = db.Column(db.String(500), nullable=True) # Stores either URL or local path

# Database Initialization
def _ensure_user_schema():
    """Add role support for existing SQLite databases created before is_admin."""
    with db.engine.begin() as connection:
        columns = connection.execute(text('PRAGMA table_info("user")')).mappings().all()
        if columns and 'is_admin' not in {column['name'] for column in columns}:
            connection.execute(text('ALTER TABLE "user" ADD COLUMN is_admin BOOLEAN NOT NULL DEFAULT 0'))


def init_db():
    with app.app_context():
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        db.create_all()
        _ensure_user_schema()

        admin = User.query.filter_by(username='admin').first()
        if admin:
            admin.is_admin = True
        elif not User.query.filter_by(is_admin=True).first():
            admin = User(username='admin', password='admin', is_admin=True)
            db.session.add(admin)
        db.session.commit()
        
        if not Product.query.first():
            starter_products = [
                Product(name="Pure Brass Diya", category="Brass Items", price=499.0, stock=45, description="Hand-polished traditional brass lamps.", image_url=_starter_product_image_url("Pure Brass Diya", "https://images.unsplash.com/photo-1609505848667-755547521471?auto=format&fit=crop&q=80&w=1000")),
                Product(name="Organic Agarbatti", category="Incense", price=199.0, stock=120, description="Naturally scented incense sticks.", image_url=_starter_product_image_url("Organic Agarbatti", "https://images.unsplash.com/photo-1602928321679-56077325677c?auto=format&fit=crop&q=80&w=1000")),
                Product(name="Puja Thali Set", category="Brass Items", price=1299.0, stock=12, description="All-in-one elegant brass thali set.", image_url=_starter_product_image_url("Puja Thali Set", "https://images.unsplash.com/photo-1561489573-316527703983?auto=format&fit=crop&q=80&w=1000")),
                Product(name="Sandalwood Powder", category="Fragrance", price=250.0, stock=60, description="Premium grade naturally sourced sandalwood.", image_url=_starter_product_image_url("Sandalwood Powder", "https://images.unsplash.com/photo-159543B95956D-C9D7A6E1A")),
                Product(name="Copper Kalash", category="Brass Items", price=750.0, stock=20, description="Pure copper vessel for ritual offerings.", image_url=_starter_product_image_url("Copper Kalash", "https://images.unsplash.com/photo-1609505848667-755547521471?auto=format&fit=crop&q=80&w=1000")),
                Product(name="Premium Camphor", category="Fragrance", price=150.0, stock=100, description="Pure smokeless camphor crystals.", image_url=_starter_product_image_url("Premium Camphor", "https://images.unsplash.com/photo-1609505848667-755547521471?auto=format&fit=crop&q=80&w=1000")),
            ]
            db.session.add_all(starter_products)
        _refresh_starter_product_images()
        db.session.commit()

def get_current_user():
    token = request.cookies.get('auth_token')
    if not token:
        return None
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user_id = data.get('user_id')
        if user_id:
            return db.session.get(User, user_id)
        username = data.get('user')
        if username:
            return User.query.filter_by(username=username).first()
    except jwt.PyJWTError:
        return None
    return None


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not get_current_user():
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user:
            return redirect(url_for('login'))
        if not user.is_admin:
            abort(403)
        return f(*args, **kwargs)
    return decorated


@app.context_processor
def inject_globals():
    return dict(
        current_user=get_current_user(),
        catalog_image_url=_catalog_image_url,
        image_placeholder_url=_image_placeholder_url,
    )


def _image_placeholder_url():
    return url_for('static', filename=PRODUCT_IMAGE_PLACEHOLDER)


def _starter_product_image_url(product_name, fallback_url=None):
    filename = STARTER_PRODUCT_IMAGE_FILES.get(product_name)
    if filename and os.path.exists(os.path.join(UPLOAD_FOLDER, filename)):
        return f"static/uploads/{filename}"
    return fallback_url


def _refresh_starter_product_images():
    starter_names = list(STARTER_PRODUCT_IMAGE_FILES.keys())
    for product in Product.query.filter(Product.name.in_(starter_names)).all():
        current_image = (product.image_url or '').strip()
        local_image = _starter_product_image_url(product.name)
        if local_image and (not current_image or current_image.startswith('https://images.unsplash.com/')):
            product.image_url = local_image


def _catalog_image_url(image_url):
    if not image_url:
        return None

    image_url = image_url.strip()
    if image_url.startswith(('http://', 'https://', '/')):
        return image_url

    normalized = image_url.replace('\\', '/')
    if normalized.startswith('static/'):
        normalized = normalized[len('static/'):]

    return url_for('static', filename=normalized)


@app.route('/')
def index():
    all_products = Product.query.all()
    categorized_products = {}
    for product in all_products:
        if product.category not in categorized_products:
            categorized_products[product.category] = []
        categorized_products[product.category].append(product)

    hero_slides_by_category = []
    for category, products in categorized_products.items():
        slides = []
        for product in products:
            image = _catalog_image_url(product.image_url)
            if not image:
                continue

            slides.append({
                'name': product.name,
                'image': image,
                'url': url_for('product_detail', product_id=product.id),
            })

        if slides:
            hero_slides_by_category.append({
                'category': category,
                'slides': slides,
            })

    hero_initial_image = ''
    if hero_slides_by_category:
        hero_initial_image = hero_slides_by_category[0]['slides'][0]['image']
    
    cart = session.get('cart', [])
    cart_count = len(cart)
    
    return render_template(
        'index.html',
        categorized_products=categorized_products,
        hero_slides_by_category=hero_slides_by_category,
        hero_initial_image=hero_initial_image,
        cart_count=cart_count,
    )

@app.route('/product/<int:product_id>')
def product_detail(product_id):
    product = Product.query.get_or_404(product_id)
    return render_template('product_detail.html', product=product)

@app.route('/add_to_cart/<int:product_id>', methods=['POST'])
def add_to_cart(product_id):
    cart = session.get('cart', [])
    cart.append(product_id)
    session['cart'] = cart
    session.modified = True
    return jsonify({"success": True, "cart_count": len(cart)})


@app.route('/api/cart_count')
def api_cart_count():
    cart = session.get('cart', [])
    return jsonify({"cart_count": len(cart)})

@app.route('/cart')
def view_cart():
    cart_ids = session.get('cart', [])
    products = Product.query.filter(Product.id.in_(cart_ids)).all()
    
    cart_items = []
    for p in products:
        count = cart_ids.count(p.id)
        cart_items.append({
            'product': p,
            'quantity': count,
            'subtotal': p.price * count
        })
    
    total = sum(item['subtotal'] for item in cart_items)
    return render_template('cart.html', items=cart_items, total=total)

@app.route('/remove_from_cart/<int:product_id>')
def remove_from_cart(product_id):
    if 'cart' in session:
        cart = session['cart']
        # remove all occurrences
        cart = [item for item in cart if item != product_id]
        session['cart'] = cart
        session.modified = True
    return redirect(url_for('view_cart'))

@app.route('/increase_quantity/<int:product_id>')
def increase_quantity(product_id):
    if 'cart' in session:
        cart = session['cart']
        cart.append(product_id)
        session['cart'] = cart
        session.modified = True
    return redirect(url_for('view_cart'))

@app.route('/decrease_quantity/<int:product_id>')
def decrease_quantity(product_id):
    if 'cart' in session:
        cart = session['cart']
        if product_id in cart:
            cart.remove(product_id)
            session['cart'] = cart
            session.modified = True
    return redirect(url_for('view_cart'))


@app.route('/clear_cart')
def clear_cart():
    session.pop('cart', None)
    return redirect(url_for('view_cart'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if User.query.filter_by(username=username).first():
            return render_template('register.html', error="Username already exists")
        new_user = User(username=username, password=password, is_admin=False)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            token = jwt.encode({
                'user_id': user.id,
                'user': username,
                'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
            }, JWT_SECRET, algorithm="HS256")
            redirect_endpoint = 'admin' if user.is_admin else 'index'
            response = make_response(redirect(url_for(redirect_endpoint)))
            response.set_cookie('auth_token', token, httponly=True, samesite='Lax', secure=not app.debug, max_age=24 * 3600)
            return response
        else:
            return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/admin')
@admin_required
def admin():
    products = Product.query.all()
    categories = db.session.query(Product.category).distinct().all()
    category_list = [c[0] for c in categories]
    user = get_current_user()
    return render_template('admin.html', products=products, admin_user=user.username, categories=category_list)

@app.route('/add-product', methods=['POST'])
@admin_required
def add_product():
    name = request.form.get('name')
    category = request.form.get('category')
    price = float(request.form.get('price'))
    stock = int(request.form.get('stock'))
    description = request.form.get('description')
    
    image_path = None
    image_url = request.form.get('image_url')
    image_file = request.files.get('image_file')
    
    if image_file and image_file.filename != '':
        filename = secure_filename(image_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(file_path)
        image_path = f"static/uploads/{filename}"
    elif image_url:
        image_path = image_url
        
    new_product = Product(name=name, category=category, price=price, stock=stock, description=description, image_url=image_path)
    db.session.add(new_product)
    db.session.commit()
    # Sync new product to Pinecone
    _sync_product_to_pinecone(new_product)
    return redirect(url_for('admin'))

@app.route('/edit-product/<int:id>', methods=['POST'])
@admin_required
def edit_product(id):
    product = Product.query.get_or_404(id)
    product.name = request.form.get('name')
    product.category = request.form.get('category')
    product.price = float(request.form.get('price'))
    product.stock = int(request.form.get('stock'))
    product.description = request.form.get('description')
    
    image_url = request.form.get('image_url')
    image_file = request.files.get('image_file')
    
    if image_file and image_file.filename != '':
        filename = secure_filename(image_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(file_path)
        product.image_url = f"static/uploads/{filename}"
    elif image_url:
        product.image_url = image_url
        
    db.session.commit()
    # Sync updated product to Pinecone
    _sync_product_to_pinecone(product)
    return redirect(url_for('admin'))

@app.route('/delete-product/<int:id>', methods=['POST'])
@admin_required
def delete_product(id):
    product = Product.query.get_or_404(id)
    product_id = product.id
    db.session.delete(product)
    db.session.commit()
    # Remove deleted product from Pinecone
    _delete_product_from_pinecone(product_id)
    return redirect(url_for('admin'))

@app.route('/change-password', methods=['POST'])
@admin_required
def change_password():
    new_username = request.form.get('new_username')
    new_password = request.form.get('new_password')
    user = get_current_user()
    if user and new_username and new_password:
        existing_user = User.query.filter(User.username == new_username, User.id != user.id).first()
        if existing_user:
            return render_template(
                'admin.html',
                products=Product.query.all(),
                admin_user=user.username,
                categories=[c[0] for c in db.session.query(Product.category).distinct().all()],
                error="Username already exists"
            ), 400
        user.username = new_username
        user.password = new_password
        db.session.commit()
        new_token = jwt.encode({
            'user_id': user.id,
            'user': new_username,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }, JWT_SECRET, algorithm="HS256")
        response = make_response(redirect(url_for('admin')))
        response.set_cookie('auth_token', new_token, httponly=True, samesite='Lax', secure=not app.debug, max_age=24 * 3600)
        return response
    return redirect(url_for('admin'))

@app.route('/logout')
def logout():
    response = make_response(redirect(url_for('index')))
    response.set_cookie('auth_token', '', expires=0, httponly=True, samesite='Lax', secure=not app.debug)
    return response

# ─── RAG helper functions ─────────────────────────────────────────────────────
def _sync_product_to_pinecone(product):
    """Upsert a single product vector. Silently skips if Pinecone not configured."""
    if not os.environ.get('PINECONE_API_KEY'):
        return
    try:
        from rag.indexer import index_single_product
        index_single_product(product)
    except Exception as e:
        app.logger.warning(f'Pinecone sync failed for product {product.id}: {e}')


def _delete_product_from_pinecone(product_id: int):
    """Delete a product vector. Silently skips if Pinecone not configured."""
    if not os.environ.get('PINECONE_API_KEY'):
        return
    try:
        from rag.indexer import delete_product_vector
        delete_product_vector(product_id)
    except Exception as e:
        app.logger.warning(f'Pinecone delete failed for product {product_id}: {e}')


# ─── Chat (RAG) endpoint ───────────────────────────────────────────────────────
@app.route('/chat', methods=['POST'])
def chat():
    """
    POST /chat
    Body: { "question": str, "history": [{"role": "user"|"assistant", "content": str}] }
    Returns: { "answer": str, "sources": [{"product_id", "name", "price", "category", "url"}] }
    """
    if not os.environ.get('PINECONE_API_KEY'):
        return jsonify({'error': 'RAG not configured. Please set up your .env file.'}), 503

    data = request.get_json(force=True)
    question = (data.get('question') or '').strip()
    history = data.get('history', [])

    if not question:
        return jsonify({'error': 'No question provided.'}), 400

    try:
        from rag.rag_engine import ask
        result = ask(question, history)
        return jsonify(result)
    except Exception as e:
        app.logger.error(f'RAG chat error: {e}')
        return jsonify({'error': 'Something went wrong with the AI assistant. Please try again.'}), 500


# ─── Admin: manual reindex ─────────────────────────────────────────────────────
@app.route('/admin/reindex', methods=['POST'])
@admin_required
def admin_reindex():
    """Reindex all products into Pinecone. Triggered by admin panel button."""
    if not os.environ.get('PINECONE_API_KEY'):
        return jsonify({'error': 'PINECONE_API_KEY not configured.'}), 503
    try:
        from rag.indexer import index_all_products
        count = index_all_products(app, Product)
        return jsonify({'success': True, 'indexed': count})
    except Exception as e:
        app.logger.error(f'Reindex error: {e}')
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    init_db()
    app.run(debug=True)
