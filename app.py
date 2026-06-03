import os
import json
import jwt
import uuid
import mimetypes
import datetime
from dataclasses import dataclass
from functools import wraps
from urllib import error, parse, request as urlrequest

from flask import Flask, render_template, request, redirect, url_for, make_response, flash, session, jsonify, abort, g
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from dotenv import load_dotenv

load_dotenv()

# Define absolute paths
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, 'templates')
UPLOAD_FOLDER = os.path.join(base_dir, 'static', 'uploads')
PRODUCT_IMAGE_PLACEHOLDER = 'product-placeholder.svg'
MAX_IMAGE_UPLOAD_BYTES = int(os.environ.get('MAX_IMAGE_UPLOAD_MB', '8')) * 1024 * 1024

STARTER_PRODUCT_IMAGE_FILES = {
    "Pure Brass Diya": "ChatGPT_Image_May_27_2026_01_18_55_PM.png",
    "Organic Agarbatti": "ChatGPT_Image_May_27_2026_01_27_26_PM.png",
    "Puja Thali Set": "ChatGPT_Image_May_27_2026_01_22_05_PM.png",
    "Sandalwood Powder": "ChatGPT_Image_May_27_2026_01_22_55_PM.png",
    "Copper Kalash": "ChatGPT_Image_May_27_2026_01_26_05_PM.png",
    "Premium Camphor": "ChatGPT_Image_May_27_2026_01_24_00_PM.png",
}

app = Flask(__name__, template_folder=template_dir)
app.secret_key = os.environ.get('SECRET_KEY', 'test-secret')
JWT_SECRET = os.environ.get('JWT_SECRET', 'test-secret')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_IMAGE_UPLOAD_BYTES

SUPABASE_URL = os.environ.get('SUPABASE_URL', '').rstrip('/')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', '')
SUPABASE_STORAGE_BUCKET = (
    os.environ.get('SUPABASE_STORAGE_BUCKET')
    or os.environ.get('SUPABASE_BUCKET')
    or 'product-images'
)
SUPABASE_ACCOUNTS_TABLE = os.environ.get('SUPABASE_ACCOUNTS_TABLE', 'accounts')
SUPABASE_PRODUCTS_TABLE = os.environ.get('SUPABASE_PRODUCTS_TABLE', 'products')
SUPABASE_SETTINGS_TABLE = os.environ.get('SUPABASE_SETTINGS_TABLE', 'settings')
ACCOUNT_STORE = os.environ.get('ACCOUNT_STORE', 'supabase').strip().lower()
DATA_STORE = os.environ.get('DATA_STORE', 'supabase').strip().lower()
LOCAL_STORE_NAMES = {'sqlite', 'sqlalchemy', 'local'}
AUTO_SYNC_PINECONE = os.environ.get('AUTO_SYNC_PINECONE', '').strip().lower() in {'1', 'true', 'yes', 'on'}
PASSWORD_HASH_METHOD = 'pbkdf2:sha256'

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('SQLALCHEMY_DATABASE_URI') or 'sqlite:///:memory:'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['ACCOUNT_STORE'] = ACCOUNT_STORE
app.config['DATA_STORE'] = DATA_STORE
db = SQLAlchemy(app)

# --- Models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, nullable=False, default=False)

class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    price = db.Column(db.Float, nullable=False)
    stock = db.Column(db.Integer, nullable=False)
    description = db.Column(db.Text, nullable=True)
    image_url = db.Column(db.String(500), nullable=True) # Stores either URL or local path

class Setting(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(50), unique=True, nullable=False)
    value = db.Column(db.String(500), nullable=False)


@dataclass
class AccountUser:
    id: object
    username: str
    password: str = ''
    is_admin: bool = False


@dataclass
class ProductRecord:
    id: object
    name: str
    category: str
    price: float
    stock: int
    description: str = ''
    image_url: str = ''


@dataclass
class SettingRecord:
    id: object
    key: str
    value: str = ''


class SupabaseError(RuntimeError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class SupabaseSetupError(SupabaseError):
    pass


_supabase_bucket_checked = False
_database_initialized = False


def _supabase_configured():
    return bool(SUPABASE_URL and SUPABASE_KEY)


def _use_sqlalchemy_accounts():
    return app.config.get('TESTING') or app.config.get('ACCOUNT_STORE') in LOCAL_STORE_NAMES


def _use_supabase_accounts():
    return not _use_sqlalchemy_accounts()


def _use_sqlalchemy_data():
    return app.config.get('TESTING') or app.config.get('DATA_STORE') in LOCAL_STORE_NAMES


def _supabase_headers(json_body=True, extra_headers=None):
    headers = {
        'apikey': SUPABASE_KEY,
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'Accept': 'application/json',
    }
    if json_body:
        headers['Content-Type'] = 'application/json'
    if extra_headers:
        headers.update(extra_headers)
    return headers


def _supabase_request(method, path, payload=None, query=None, data=None, headers=None, expected=(200, 201, 204)):
    if not _supabase_configured():
        raise SupabaseError('Supabase is not configured. Set SUPABASE_URL and SUPABASE_KEY in .env.')

    path = '/' + path.lstrip('/')
    url = f'{SUPABASE_URL}{path}'
    if query:
        url = f'{url}?{parse.urlencode(query, doseq=True)}'

    body = data
    request_headers = headers or _supabase_headers()
    if payload is not None:
        body = json.dumps(payload).encode('utf-8')

    req = urlrequest.Request(url, data=body, headers=request_headers, method=method)
    try:
        with urlrequest.urlopen(req, timeout=20) as response:
            response_body = response.read()
            if response.status not in expected:
                raise SupabaseError(
                    f'Supabase request failed with status {response.status}.',
                    status=response.status,
                )
            if not response_body:
                return None
            return json.loads(response_body.decode('utf-8'))
    except error.HTTPError as exc:
        details = exc.read().decode('utf-8', errors='replace')
        known_tables = [SUPABASE_ACCOUNTS_TABLE, SUPABASE_PRODUCTS_TABLE, SUPABASE_SETTINGS_TABLE]
        missing_table = next((table for table in known_tables if table in details), None)
        if exc.code == 404 and 'PGRST205' in details and missing_table:
            raise SupabaseSetupError(
                f"Supabase table '{missing_table}' is missing. Run supabase_schema.sql in the Supabase SQL editor, "
                "then restart the Flask app.",
                status=exc.code,
            ) from exc
        raise SupabaseError(f'Supabase request failed with status {exc.code}: {details}', status=exc.code) from exc
    except error.URLError as exc:
        raise SupabaseError(f'Unable to reach Supabase: {exc.reason}') from exc


def _table_path(table_name):
    return f'/rest/v1/{parse.quote(table_name, safe="")}'


def _accounts_path():
    return _table_path(SUPABASE_ACCOUNTS_TABLE)


def _products_path():
    return _table_path(SUPABASE_PRODUCTS_TABLE)


def _settings_path():
    return _table_path(SUPABASE_SETTINGS_TABLE)


def _hash_password(password):
    return generate_password_hash(password or '', method=PASSWORD_HASH_METHOD, salt_length=16)


def _looks_like_password_hash(value):
    return isinstance(value, str) and value.startswith(('pbkdf2:', 'scrypt:', 'argon2:'))


def _password_matches(stored_password, supplied_password):
    stored_password = stored_password or ''
    supplied_password = supplied_password or ''
    if _looks_like_password_hash(stored_password):
        return check_password_hash(stored_password, supplied_password)
    return stored_password == supplied_password


def _account_from_supabase_row(row):
    if not row:
        return None
    return AccountUser(
        id=row.get('id'),
        username=row.get('username', ''),
        password=row.get('password_hash') or row.get('password') or '',
        is_admin=bool(row.get('is_admin')),
    )


def _get_supabase_account_by_username(username):
    rows = _supabase_request(
        'GET',
        _accounts_path(),
        query={
            'select': 'id,username,password_hash,is_admin',
            'username': f'eq.{username}',
            'limit': '1',
        },
        expected=(200,),
    ) or []
    return _account_from_supabase_row(rows[0]) if rows else None


def _get_supabase_account_by_id(user_id):
    rows = _supabase_request(
        'GET',
        _accounts_path(),
        query={
            'select': 'id,username,password_hash,is_admin',
            'id': f'eq.{user_id}',
            'limit': '1',
        },
        expected=(200,),
    ) or []
    return _account_from_supabase_row(rows[0]) if rows else None


def _create_supabase_account(username, password, is_admin=False):
    rows = _supabase_request(
        'POST',
        _accounts_path(),
        payload={
            'username': username,
            'password_hash': _hash_password(password),
            'is_admin': bool(is_admin),
        },
        headers=_supabase_headers(extra_headers={'Prefer': 'return=representation'}),
        expected=(200, 201),
    ) or []
    return _account_from_supabase_row(rows[0]) if rows else _get_supabase_account_by_username(username)


def _update_supabase_account(user_id, username=None, password=None, is_admin=None):
    payload = {}
    if username is not None:
        payload['username'] = username
    if password is not None:
        payload['password_hash'] = _hash_password(password)
    if is_admin is not None:
        payload['is_admin'] = bool(is_admin)
    if not payload:
        return _get_supabase_account_by_id(user_id)

    rows = _supabase_request(
        'PATCH',
        _accounts_path(),
        payload=payload,
        query={'id': f'eq.{user_id}'},
        headers=_supabase_headers(extra_headers={'Prefer': 'return=representation'}),
        expected=(200, 204),
    ) or []
    return _account_from_supabase_row(rows[0]) if rows else _get_supabase_account_by_id(user_id)


def _get_account_by_username(username):
    if not username:
        return None
    if _use_sqlalchemy_accounts():
        return User.query.filter_by(username=username).first()
    return _get_supabase_account_by_username(username)


def _get_account_by_id(user_id):
    if not user_id:
        return None
    if _use_sqlalchemy_accounts():
        return db.session.get(User, user_id)
    return _get_supabase_account_by_id(user_id)


def _create_account(username, password, is_admin=False):
    if _use_sqlalchemy_accounts():
        new_user = User(username=username, password=_hash_password(password), is_admin=bool(is_admin))
        db.session.add(new_user)
        db.session.commit()
        return new_user
    return _create_supabase_account(username, password, is_admin)


def _update_account_credentials(user, username=None, password=None):
    if _use_sqlalchemy_accounts():
        if username is not None:
            user.username = username
        if password is not None:
            user.password = _hash_password(password)
        db.session.commit()
        return user
    return _update_supabase_account(user.id, username=username, password=password)


def _authenticate_account(username, password):
    user = _get_account_by_username(username)
    if not user or not _password_matches(getattr(user, 'password', ''), password):
        return None

    if not _looks_like_password_hash(getattr(user, 'password', '')):
        user = _update_account_credentials(user, password=password)
    return user


def _ensure_admin_account():
    admin = _get_account_by_username('admin')
    if admin:
        if _use_sqlalchemy_accounts():
            admin.is_admin = True
            if not _looks_like_password_hash(admin.password):
                admin.password = _hash_password(admin.password)
            db.session.commit()
        elif not admin.is_admin:
            _update_supabase_account(admin.id, is_admin=True)
        return

    if _use_sqlalchemy_accounts() and User.query.filter_by(is_admin=True).first():
        return
    _create_account('admin', 'admin', is_admin=True)


def _product_select_columns():
    return 'id,name,category,price,stock,description,image_url'


def _product_from_supabase_row(row):
    if not row:
        return None
    return ProductRecord(
        id=row.get('id'),
        name=row.get('name') or '',
        category=row.get('category') or '',
        price=float(row.get('price') or 0),
        stock=int(row.get('stock') or 0),
        description=row.get('description') or '',
        image_url=row.get('image_url') or '',
    )


def _setting_from_supabase_row(row):
    if not row:
        return None
    return SettingRecord(
        id=row.get('id'),
        key=row.get('key') or '',
        value=row.get('value') or '',
    )


def _get_all_products():
    if _use_sqlalchemy_data():
        return Product.query.order_by(Product.id).all()

    rows = _supabase_request(
        'GET',
        _products_path(),
        query={
            'select': _product_select_columns(),
            'order': 'id.asc',
        },
        expected=(200,),
    ) or []
    return [_product_from_supabase_row(row) for row in rows]


def _get_product_by_id(product_id):
    if not product_id:
        return None
    if _use_sqlalchemy_data():
        return db.session.get(Product, product_id)

    rows = _supabase_request(
        'GET',
        _products_path(),
        query={
            'select': _product_select_columns(),
            'id': f'eq.{product_id}',
            'limit': '1',
        },
        expected=(200,),
    ) or []
    return _product_from_supabase_row(rows[0]) if rows else None


def _get_product_or_404(product_id):
    product = _get_product_by_id(product_id)
    if not product:
        abort(404)
    return product


def _get_products_by_ids(product_ids):
    unique_ids = []
    for raw_id in product_ids:
        try:
            product_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if product_id not in unique_ids:
            unique_ids.append(product_id)

    if not unique_ids:
        return []

    if _use_sqlalchemy_data():
        return Product.query.filter(Product.id.in_(unique_ids)).all()

    rows = _supabase_request(
        'GET',
        _products_path(),
        query={
            'select': _product_select_columns(),
            'id': f'in.({",".join(str(product_id) for product_id in unique_ids)})',
        },
        expected=(200,),
    ) or []
    return [_product_from_supabase_row(row) for row in rows]


def _create_product_record(name, category, price, stock, description=None, image_url=None):
    payload = {
        'name': name,
        'category': category,
        'price': float(price),
        'stock': int(stock),
        'description': description,
        'image_url': image_url,
    }

    if _use_sqlalchemy_data():
        product = Product(**payload)
        db.session.add(product)
        db.session.commit()
        return product

    rows = _supabase_request(
        'POST',
        _products_path(),
        payload=payload,
        headers=_supabase_headers(extra_headers={'Prefer': 'return=representation'}),
        expected=(200, 201),
    ) or []
    return _product_from_supabase_row(rows[0]) if rows else None


def _update_product_record(product_id, **fields):
    payload = {}
    for key in ('name', 'category', 'price', 'stock', 'description', 'image_url'):
        if key in fields:
            payload[key] = fields[key]
    if 'price' in payload:
        payload['price'] = float(payload['price'])
    if 'stock' in payload:
        payload['stock'] = int(payload['stock'])

    if _use_sqlalchemy_data():
        product = _get_product_or_404(product_id)
        for key, value in payload.items():
            setattr(product, key, value)
        db.session.commit()
        return product

    if not payload:
        return _get_product_by_id(product_id)

    rows = _supabase_request(
        'PATCH',
        _products_path(),
        payload=payload,
        query={'id': f'eq.{product_id}'},
        headers=_supabase_headers(extra_headers={'Prefer': 'return=representation'}),
        expected=(200, 204),
    ) or []
    return _product_from_supabase_row(rows[0]) if rows else _get_product_by_id(product_id)


def _delete_product_record(product_id):
    product = _get_product_or_404(product_id)

    if _use_sqlalchemy_data():
        db.session.delete(product)
        db.session.commit()
        return product

    _supabase_request(
        'DELETE',
        _products_path(),
        query={'id': f'eq.{product_id}'},
        expected=(200, 204),
    )
    return product


def _get_product_categories(products=None):
    products = products if products is not None else _get_all_products()
    return sorted({product.category for product in products if product.category})


def _get_setting_record(key):
    if _use_sqlalchemy_data():
        return Setting.query.filter_by(key=key).first()

    rows = _supabase_request(
        'GET',
        _settings_path(),
        query={
            'select': 'id,key,value',
            'key': f'eq.{key}',
            'limit': '1',
        },
        expected=(200,),
    ) or []
    return _setting_from_supabase_row(rows[0]) if rows else None


def _get_setting_value(key, default=''):
    setting = _get_setting_record(key)
    return setting.value if setting else default


def _set_setting_value(key, value):
    value = value or ''
    if _use_sqlalchemy_data():
        setting = Setting.query.filter_by(key=key).first()
        if not setting:
            setting = Setting(key=key, value=value)
            db.session.add(setting)
        else:
            setting.value = value
        db.session.commit()
        return setting

    existing = _get_setting_record(key)
    if existing:
        rows = _supabase_request(
            'PATCH',
            _settings_path(),
            payload={'value': value},
            query={'key': f'eq.{key}'},
            headers=_supabase_headers(extra_headers={'Prefer': 'return=representation'}),
            expected=(200, 204),
        ) or []
    else:
        rows = _supabase_request(
            'POST',
            _settings_path(),
            payload={'key': key, 'value': value},
            headers=_supabase_headers(extra_headers={'Prefer': 'return=representation'}),
            expected=(200, 201),
        ) or []
    return _setting_from_supabase_row(rows[0]) if rows else _get_setting_record(key)


def _ensure_setting_value(key, default=''):
    if not _get_setting_record(key):
        _set_setting_value(key, default)


def _starter_product_payloads():
    return [
        {'name': "Pure Brass Diya", 'category': "Brass Items", 'price': 499.0, 'stock': 45, 'description': "Hand-polished traditional brass lamps.", 'image_url': _starter_product_image_url("Pure Brass Diya", "https://images.unsplash.com/photo-1609505848667-755547521471?auto=format&fit=crop&q=80&w=1000")},
        {'name': "Organic Agarbatti", 'category': "Incense", 'price': 199.0, 'stock': 120, 'description': "Naturally scented incense sticks.", 'image_url': _starter_product_image_url("Organic Agarbatti", "https://images.unsplash.com/photo-1602928321679-56077325677c?auto=format&fit=crop&q=80&w=1000")},
        {'name': "Puja Thali Set", 'category': "Brass Items", 'price': 1299.0, 'stock': 12, 'description': "All-in-one elegant brass thali set.", 'image_url': _starter_product_image_url("Puja Thali Set", "https://images.unsplash.com/photo-1561489573-316527703983?auto=format&fit=crop&q=80&w=1000")},
        {'name': "Sandalwood Powder", 'category': "Fragrance", 'price': 250.0, 'stock': 60, 'description': "Premium grade naturally sourced sandalwood.", 'image_url': _starter_product_image_url("Sandalwood Powder", "https://images.unsplash.com/photo-159543B95956D-C9D7A6E1A")},
        {'name': "Copper Kalash", 'category': "Brass Items", 'price': 750.0, 'stock': 20, 'description': "Pure copper vessel for ritual offerings.", 'image_url': _starter_product_image_url("Copper Kalash", "https://images.unsplash.com/photo-1609505848667-755547521471?auto=format&fit=crop&q=80&w=1000")},
        {'name': "Premium Camphor", 'category': "Fragrance", 'price': 150.0, 'stock': 100, 'description': "Pure smokeless camphor crystals.", 'image_url': _starter_product_image_url("Premium Camphor", "https://images.unsplash.com/photo-1609505848667-755547521471?auto=format&fit=crop&q=80&w=1000")},
    ]


def _ensure_starter_products():
    if _get_all_products():
        _refresh_starter_product_images()
        return

    for payload in _starter_product_payloads():
        _create_product_record(**payload)
    _refresh_starter_product_images()


def _ensure_supabase_bucket():
    global _supabase_bucket_checked
    if _supabase_bucket_checked:
        return

    bucket = parse.quote(SUPABASE_STORAGE_BUCKET, safe='')
    try:
        bucket_info = _supabase_request('GET', f'/storage/v1/bucket/{bucket}', expected=(200,))
        if bucket_info and not bucket_info.get('public'):
            _supabase_request('PUT', f'/storage/v1/bucket/{bucket}', payload={'public': True}, expected=(200,))
    except SupabaseError as exc:
        if exc.status != 404:
            raise
        try:
            _supabase_request(
                'POST',
                '/storage/v1/bucket',
                payload={
                    'id': SUPABASE_STORAGE_BUCKET,
                    'name': SUPABASE_STORAGE_BUCKET,
                    'public': True,
                },
                expected=(200, 201),
            )
        except SupabaseError as create_exc:
            if create_exc.status != 409:
                raise
    _supabase_bucket_checked = True


def _supabase_public_storage_url(object_path):
    bucket = parse.quote(SUPABASE_STORAGE_BUCKET, safe='')
    encoded_path = parse.quote(object_path, safe='/')
    return f'{SUPABASE_URL}/storage/v1/object/public/{bucket}/{encoded_path}'


def _upload_file_to_supabase_storage(image_file):
    _ensure_supabase_bucket()

    original_name = secure_filename(image_file.filename or '')
    if not original_name:
        original_name = 'product-image'
    _, extension = os.path.splitext(original_name)
    object_path = (
        'products/'
        f'{datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")}-'
        f'{uuid.uuid4().hex}{extension.lower()}'
    )
    content_type = image_file.mimetype or mimetypes.guess_type(original_name)[0] or 'application/octet-stream'
    file_bytes = image_file.read()
    if hasattr(image_file.stream, 'seek'):
        image_file.stream.seek(0)

    bucket = parse.quote(SUPABASE_STORAGE_BUCKET, safe='')
    encoded_path = parse.quote(object_path, safe='/')
    _supabase_request(
        'POST',
        f'/storage/v1/object/{bucket}/{encoded_path}',
        data=file_bytes,
        headers=_supabase_headers(
            json_body=False,
            extra_headers={
                'Content-Type': content_type,
                'x-upsert': 'true',
            },
        ),
        expected=(200, 201),
    )
    return _supabase_public_storage_url(object_path)


def _save_product_image_upload(image_file):
    if not image_file or not image_file.filename:
        raise SupabaseError('No image file was selected.')
    if image_file.mimetype and not image_file.mimetype.startswith('image/'):
        raise SupabaseError('Uploaded file must be an image.')

    if _supabase_configured() and not app.config.get('TESTING'):
        return _upload_file_to_supabase_storage(image_file)

    filename = secure_filename(image_file.filename)
    if not filename:
        raise SupabaseError('Uploaded image filename is invalid.')
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(file_path)
    return f"static/uploads/{filename}"


def _admin_redirect(message=None, error=None):
    if error:
        return redirect(url_for('admin', error=error))
    if message:
        return redirect(url_for('admin', success=message))
    return redirect(url_for('admin'))


def _json_error(message, status=500):
    return jsonify({'error': message}), status


@app.errorhandler(SupabaseError)
def handle_supabase_error(exc):
    if request.path == '/chat':
        return _json_error(str(exc), exc.status or 500)
    raise exc


@app.errorhandler(Exception)
def handle_chat_exception(exc):
    if isinstance(exc, HTTPException):
        return exc
    if request.path == '/chat':
        app.logger.exception('Unhandled chat request error')
        return _json_error('The AI assistant failed before it could answer. Check the server logs for details.', 500)
    raise exc


@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(exc):
    if request.path in {'/add-product'} or request.path.startswith('/edit-product/'):
        max_mb = MAX_IMAGE_UPLOAD_BYTES // (1024 * 1024)
        return _render_admin(error=f"Image upload failed: file is larger than {max_mb} MB.", status=413)
    return exc


def _render_admin(error=None, success=None, status=200):
    products = _get_all_products()
    category_list = _get_product_categories(products)
    user = get_current_user()
    upi_id = _get_setting_value('upi_id')
    error = error or request.args.get('error')
    success = success or request.args.get('success')
    return render_template(
        'admin.html',
        products=products,
        admin_user=user.username if user else '',
        categories=category_list,
        upi_id=upi_id,
        error=error,
        success=success,
    ), status


# Database Initialization
def _ensure_user_schema():
    """Add role support for existing SQLite databases created before is_admin."""
    with db.engine.begin() as connection:
        columns = connection.execute(text('PRAGMA table_info("user")')).mappings().all()
        if columns and 'is_admin' not in {column['name'] for column in columns}:
            connection.execute(text('ALTER TABLE "user" ADD COLUMN is_admin BOOLEAN NOT NULL DEFAULT 0'))


def init_db():
    global _database_initialized
    with app.app_context():
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        if _use_sqlalchemy_accounts() or _use_sqlalchemy_data():
            db.create_all()
        if _use_sqlalchemy_accounts():
            _ensure_user_schema()
        _ensure_admin_account()
        _ensure_setting_value('upi_id', '')
        _ensure_starter_products()
        _database_initialized = True


@app.before_request
def _ensure_database_initialized():
    if not _database_initialized:
        init_db()

def get_current_user():
    if hasattr(g, '_current_user'):
        return g._current_user

    token = request.cookies.get('auth_token')
    if not token:
        g._current_user = None
        return None
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user_id = data.get('user_id')
        if user_id:
            g._current_user = _get_account_by_id(user_id)
            return g._current_user
        username = data.get('user')
        if username:
            g._current_user = _get_account_by_username(username)
            return g._current_user
    except (jwt.PyJWTError, SupabaseError):
        g._current_user = None
        return None
    g._current_user = None
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
    starter_names = set(STARTER_PRODUCT_IMAGE_FILES.keys())
    for product in _get_all_products():
        if product.name not in starter_names:
            continue
        current_image = (product.image_url or '').strip()
        local_image = _starter_product_image_url(product.name)
        if local_image and (not current_image or current_image.startswith('https://images.unsplash.com/')):
            _update_product_record(product.id, image_url=local_image)


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
    all_products = _get_all_products()
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
    product = _get_product_or_404(product_id)
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
    products = _get_products_by_ids(cart_ids)
    
    cart_items = []
    for p in products:
        count = cart_ids.count(p.id)
        cart_items.append({
            'product': p,
            'quantity': count,
            'subtotal': p.price * count
        })
    
    total = sum(item['subtotal'] for item in cart_items)
    upi_id = _get_setting_value('upi_id')
    return render_template('cart.html', items=cart_items, total=total, upi_id=upi_id)

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
        try:
            if _get_account_by_username(username):
                return render_template('register.html', error="Username already exists")
            _create_account(username, password, is_admin=False)
        except SupabaseError as exc:
            app.logger.error(f'Supabase registration failed: {exc}')
            return render_template('register.html', error=f"Account registration failed: {exc}"), 500
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        try:
            user = _authenticate_account(username, password)
        except SupabaseError as exc:
            app.logger.error(f'Supabase login failed: {exc}')
            return render_template('login.html', error=f"Login failed: {exc}"), 500
        if user:
            token = jwt.encode({
                'user_id': user.id,
                'user': user.username,
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
    return _render_admin()

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
        try:
            image_path = _save_product_image_upload(image_file)
        except Exception as exc:
            app.logger.exception('Product image upload failed')
            return _render_admin(
                error=f"Image upload failed: {exc}",
                status=500,
            )
    elif image_url:
        image_path = image_url
        
    new_product = _create_product_record(
        name=name,
        category=category,
        price=price,
        stock=stock,
        description=description,
        image_url=image_path,
    )
    # Sync new product to Pinecone
    _sync_product_to_pinecone(new_product)
    return _admin_redirect("Product saved successfully.")

@app.route('/edit-product/<int:id>', methods=['POST'])
@admin_required
def edit_product(id):
    product = _get_product_or_404(id)
    image_path = product.image_url
    
    image_url = request.form.get('image_url')
    image_file = request.files.get('image_file')
    
    if image_file and image_file.filename != '':
        try:
            image_path = _save_product_image_upload(image_file)
        except Exception as exc:
            app.logger.exception('Product image upload failed')
            return _render_admin(
                error=f"Image upload failed: {exc}",
                status=500,
            )
    elif image_url:
        image_path = image_url

    product = _update_product_record(
        id,
        name=request.form.get('name'),
        category=request.form.get('category'),
        price=float(request.form.get('price')),
        stock=int(request.form.get('stock')),
        description=request.form.get('description'),
        image_url=image_path,
    )
    # Sync updated product to Pinecone
    _sync_product_to_pinecone(product)
    return _admin_redirect("Product updated successfully.")

@app.route('/delete-product/<int:id>', methods=['POST'])
@admin_required
def delete_product(id):
    product = _delete_product_record(id)
    product_id = product.id
    # Remove deleted product from Pinecone
    _delete_product_from_pinecone(product_id)
    return redirect(url_for('admin'))

@app.route('/update-upi-id', methods=['POST'])
@admin_required
def update_upi_id():
    new_upi_id = request.form.get('upi_id')
    if new_upi_id is not None:
        _set_setting_value('upi_id', new_upi_id)
    return redirect(url_for('admin'))

@app.route('/change-password', methods=['POST'])
@admin_required
def change_password():
    new_username = request.form.get('new_username')
    new_password = request.form.get('new_password')
    user = get_current_user()
    if user and new_username and new_password:
        existing_user = _get_account_by_username(new_username)
        if existing_user and str(existing_user.id) != str(user.id):
            return _render_admin(error="Username already exists", status=400)
        user = _update_account_credentials(user, username=new_username, password=new_password)
        new_token = jwt.encode({
            'user_id': user.id,
            'user': new_username,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }, JWT_SECRET, algorithm="HS256")
        response = make_response(redirect(url_for('admin')))
        response.set_cookie('auth_token', new_token, httponly=True, samesite='Lax', secure=not app.debug, max_age=24 * 3600)
        return response
    return redirect(url_for('admin'))


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))
    if user.is_admin:
        return redirect(url_for('admin'))

    error = None
    success = None

    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')

        if not _password_matches(getattr(user, 'password', ''), current_password):
            error = "Incorrect current password"
        else:
            _update_account_credentials(user, password=new_password)
            success = "Password updated successfully"

    return render_template('profile.html', error=error, success=success)

@app.route('/logout')

def logout():
    response = make_response(redirect(url_for('index')))
    response.set_cookie('auth_token', '', expires=0, httponly=True, samesite='Lax', secure=not app.debug)
    return response

# ─── RAG helper functions ─────────────────────────────────────────────────────
def _sync_product_to_pinecone(product):
    """Upsert a single product vector. Silently skips if Pinecone not configured."""
    if not AUTO_SYNC_PINECONE:
        return
    if not os.environ.get('PINECONE_API_KEY'):
        return
    try:
        from rag.indexer import index_single_product
        index_single_product(product)
    except Exception as e:
        app.logger.warning(f'Pinecone sync failed for product {product.id}: {e}')


def _delete_product_from_pinecone(product_id: int):
    """Delete a product vector. Silently skips if Pinecone not configured."""
    if not AUTO_SYNC_PINECONE:
        return
    if not os.environ.get('PINECONE_API_KEY'):
        return
    try:
        from rag.indexer import delete_product_vector
        delete_product_vector(product_id)
    except Exception as e:
        app.logger.warning(f'Pinecone delete failed for product {product_id}: {e}')


def _missing_ai_config():
    missing = []
    if not os.environ.get('PINECONE_API_KEY'):
        missing.append('PINECONE_API_KEY')

    provider = os.environ.get('LLM_PROVIDER', 'openai').lower()
    provider_keys = {
        'openai': 'OPENAI_API_KEY',
        'openrouter': 'OPENROUTER_API_KEY',
        'google': 'GOOGLE_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'groq': 'GROQ_API_KEY',
    }
    llm_key = provider_keys.get(provider)
    if not llm_key:
        missing.append('valid LLM_PROVIDER')
    elif not os.environ.get(llm_key):
        missing.append(llm_key)

    embedding_provider = os.environ.get('EMBEDDING_PROVIDER', '').lower()
    embedding_keys = {
        'pinecone': 'PINECONE_API_KEY',
        'openai': 'OPENAI_API_KEY',
        'google': 'GOOGLE_API_KEY',
    }
    emb_key = embedding_keys.get(embedding_provider)
    if emb_key and not os.environ.get(emb_key):
        missing.append(emb_key)

    return list(dict.fromkeys(missing))


# ─── Chat (RAG) endpoint ───────────────────────────────────────────────────────
@app.route('/chat', methods=['POST'])
def chat():
    """
    POST /chat
    Body: { "question": str, "history": [{"role": "user"|"assistant", "content": str}] }
    Returns: { "answer": str, "sources": [{"product_id", "name", "price", "category", "url"}] }
    """
    missing_config = _missing_ai_config()
    if missing_config:
        return _json_error(
            f"AI assistant is not configured. Missing: {', '.join(missing_config)}.",
            503,
        )

    data = request.get_json(force=True)
    question = (data.get('question') or '').strip()
    history = data.get('history', [])

    if not question:
        return jsonify({'error': 'No question provided.'}), 400

    try:
        from rag.rag_engine import ask
        result = ask(question, history)
        return jsonify(result)
    except ImportError as e:
        app.logger.error(f'RAG import error: {e}')
        return jsonify({'error': f'AI dependency missing: {e}'}), 503
    except MemoryError:
        app.logger.error('RAG out of memory loading model')
        return jsonify({'error': 'Server ran out of memory loading the AI model.'}), 503
    except Exception as e:
        app.logger.error(f'RAG chat error ({type(e).__name__}): {e}')
        return jsonify({'error': f'AI assistant error ({type(e).__name__}): {e}'}), 500


# ─── Chat health check (diagnose Render issues) ───────────────────────────────
@app.route('/chat/health')
def chat_health():
    """Diagnostic endpoint to test each component of the AI pipeline."""
    checks = {}

    # 1. Check env vars
    checks['env_vars'] = {
        'PINECONE_API_KEY': bool(os.environ.get('PINECONE_API_KEY')),
        'LLM_PROVIDER': os.environ.get('LLM_PROVIDER', '(not set, defaults to openai)'),
        'OPENROUTER_API_KEY': bool(os.environ.get('OPENROUTER_API_KEY')),
        'EMBEDDING_PROVIDER': os.environ.get('EMBEDDING_PROVIDER', '(not set, auto-detect)'),
    }

    # 2. Check embedding model
    try:
        from rag.embeddings import get_embedding_provider, get_embeddings, EMBED_MODEL
        provider = get_embedding_provider()
        checks['embedding_provider'] = provider
        emb = get_embeddings()
        checks['embed_model'] = EMBED_MODEL
        test_vec = emb.embed_query("test")
        checks['embedding_model'] = f'OK (dim={len(test_vec)})'
    except Exception as e:
        checks['embedding_model'] = f'FAILED: {type(e).__name__}: {e}'


    # 3. Check Pinecone connection
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
        index_name = os.environ.get('PINECONE_INDEX_NAME', 'pooja-store')
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        checks['pinecone'] = f'OK (vectors={stats.get("total_vector_count", "?")})'
    except Exception as e:
        checks['pinecone'] = f'FAILED: {type(e).__name__}: {e}'

    # 4. Check LLM connectivity (quick test)
    try:
        from rag.rag_engine import _get_llm
        llm = _get_llm()
        checks['llm'] = f'OK ({type(llm).__name__})'
    except Exception as e:
        checks['llm'] = f'FAILED: {type(e).__name__}: {e}'

    all_ok = all(
        isinstance(v, str) and v.startswith('OK')
        for k, v in checks.items()
        if k not in ('env_vars', 'embedding_provider')
    )
    checks['overall'] = 'HEALTHY' if all_ok else 'UNHEALTHY'
    return jsonify(checks), 200 if all_ok else 503


# ─── Admin: manual reindex ─────────────────────────────────────────────────────
@app.route('/admin/reindex', methods=['POST'])
@admin_required
def admin_reindex():
    """Reindex all products into Pinecone. Triggered by admin panel button."""
    if not os.environ.get('PINECONE_API_KEY'):
        return jsonify({'error': 'PINECONE_API_KEY not configured.'}), 503
    try:
        from rag.indexer import index_all_products
        count = index_all_products(products=_get_all_products())
        return jsonify({'success': True, 'indexed': count})
    except Exception as e:
        app.logger.error(f'Reindex error: {e}')
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    try:
        init_db()
    except SupabaseSetupError as exc:
        raise SystemExit(str(exc)) from exc
    app.run(debug=True)

