import importlib
import io
import sys
from pathlib import Path

import pytest
from werkzeug.datastructures import FileStorage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

app_module = importlib.import_module("app")
from app import Product, app, db, init_db


@pytest.fixture()
def client(tmp_path):
    app.config.update(
        TESTING=True,
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{tmp_path / 'test.db'}",
    )
    with app.app_context():
        db.drop_all()
        init_db()

    with app.test_client() as test_client:
        yield test_client

    with app.app_context():
        db.session.remove()
        db.drop_all()


def login(client, username, password):
    return client.post(
        "/login",
        data={"username": username, "password": password},
        follow_redirects=False,
    )


def test_registered_user_is_not_allowed_into_admin(client):
    client.post("/register", data={"username": "buyer", "password": "secret"})

    login_response = login(client, "buyer", "secret")
    assert login_response.status_code == 302
    assert login_response.headers["Location"].endswith("/")

    with app.app_context():
        product_count = Product.query.count()

    assert client.get("/admin").status_code == 403
    assert client.post("/admin/reindex").status_code == 403
    assert client.post(
        "/add-product",
        data={
            "name": "Blocked Product",
            "category": "Blocked",
            "price": "1.00",
            "stock": "1",
            "description": "Should not be created",
        },
    ).status_code == 403

    with app.app_context():
        assert Product.query.count() == product_count


def test_seed_admin_still_has_admin_access(client):
    login_response = login(client, "admin", "admin")
    assert login_response.status_code == 302
    assert login_response.headers["Location"].endswith("/admin")

    admin_response = client.get("/admin")
    assert admin_response.status_code == 200
    assert b"Welcome, admin" in admin_response.data


def test_local_product_images_render_with_static_urls(client):
    with app.app_context():
        product = Product.query.first()
        product.image_url = r"static\uploads\local-product.png"
        db.session.commit()
        product_id = product.id

    home_response = client.get("/")
    detail_response = client.get(f"/product/{product_id}")

    assert b'src="/static/uploads/local-product.png"' in home_response.data
    assert b'src="/static/uploads/local-product.png"' in detail_response.data
    assert b"/static/product-placeholder.svg" in home_response.data
    assert b"/static/product-placeholder.svg" in detail_response.data


def test_configured_admin_upload_uses_supabase_storage(monkeypatch):
    uploaded = {}

    def fake_upload(file_storage):
        uploaded["filename"] = file_storage.filename
        uploaded["body"] = file_storage.read()
        return "https://example.supabase.co/storage/v1/object/public/product-images/products/item.png"

    monkeypatch.setitem(app.config, "TESTING", False)
    monkeypatch.setattr(app_module, "_supabase_configured", lambda: True)
    monkeypatch.setattr(app_module, "_upload_file_to_supabase_storage", fake_upload)

    image_file = FileStorage(
        stream=io.BytesIO(b"image-bytes"),
        filename="item.png",
        content_type="image/png",
    )

    image_url = app_module._save_product_image_upload(image_file)

    assert image_url.startswith("https://example.supabase.co/storage/v1/object/public/")
    assert uploaded == {"filename": "item.png", "body": b"image-bytes"}


def test_admin_add_product_file_upload_stores_supabase_url(client, monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    monkeypatch.setattr(
        app_module,
        "_save_product_image_upload",
        lambda file_storage: "https://example.supabase.co/storage/v1/object/public/product-images/products/item.png",
    )

    login(client, "admin", "admin")
    response = client.post(
        "/add-product",
        data={
            "name": "Supabase Image Product",
            "category": "Uploads",
            "price": "25.00",
            "stock": "3",
            "description": "Stored in Supabase",
            "image_file": (io.BytesIO(b"image-bytes"), "item.png"),
        },
        content_type="multipart/form-data",
        follow_redirects=False,
    )

    assert response.status_code == 302
    assert "success=Product+saved+successfully" in response.headers["Location"]

    with app.app_context():
        product = Product.query.filter_by(name="Supabase Image Product").one()
        assert product.image_url == "https://example.supabase.co/storage/v1/object/public/product-images/products/item.png"


def test_admin_edit_product_file_upload_stores_supabase_url(client, monkeypatch):
    monkeypatch.setattr(
        app_module,
        "_save_product_image_upload",
        lambda file_storage: "https://example.supabase.co/storage/v1/object/public/product-images/products/updated.png",
    )

    with app.app_context():
        product_id = Product.query.first().id

    login(client, "admin", "admin")
    response = client.post(
        f"/edit-product/{product_id}",
        data={
            "name": "Updated Image Product",
            "category": "Uploads",
            "price": "55.00",
            "stock": "8",
            "description": "Updated upload",
            "image_file": (io.BytesIO(b"updated-image-bytes"), "updated.png"),
        },
        content_type="multipart/form-data",
        follow_redirects=False,
    )

    assert response.status_code == 302
    assert "success=Product+updated+successfully" in response.headers["Location"]

    with app.app_context():
        product = db.session.get(Product, product_id)
        assert product.image_url == "https://example.supabase.co/storage/v1/object/public/product-images/products/updated.png"


def test_admin_upload_failure_is_rendered_for_add_and_edit(client, monkeypatch):
    def fail_upload(file_storage):
        raise app_module.SupabaseError("storage rejected upload")

    monkeypatch.setattr(app_module, "_save_product_image_upload", fail_upload)

    with app.app_context():
        product_id = Product.query.first().id

    login(client, "admin", "admin")
    add_response = client.post(
        "/add-product",
        data={
            "name": "Rejected Image Product",
            "category": "Uploads",
            "price": "25.00",
            "stock": "3",
            "description": "Should not save",
            "image_file": (io.BytesIO(b"bad-image"), "bad.png"),
        },
        content_type="multipart/form-data",
    )
    edit_response = client.post(
        f"/edit-product/{product_id}",
        data={
            "name": "Rejected Edit",
            "category": "Uploads",
            "price": "25.00",
            "stock": "3",
            "description": "Should not update",
            "image_file": (io.BytesIO(b"bad-image"), "bad.png"),
        },
        content_type="multipart/form-data",
    )

    assert add_response.status_code == 500
    assert edit_response.status_code == 500
    assert b"Image upload failed: storage rejected upload" in add_response.data
    assert b"Image upload failed: storage rejected upload" in edit_response.data


def test_product_save_does_not_sync_pinecone_unless_enabled(monkeypatch):
    called = {}

    def fake_index(product):
        called["product"] = product

    monkeypatch.setattr(app_module, "AUTO_SYNC_PINECONE", False)
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setattr("rag.indexer.index_single_product", fake_index)

    app_module._sync_product_to_pinecone(app_module.ProductRecord(
        id=1,
        name="Test Product",
        category="Test",
        price=1.0,
        stock=1,
    ))

    assert called == {}


def test_register_uses_supabase_account_store_when_enabled(client, monkeypatch):
    created = {}

    def fake_create(username, password, is_admin=False):
        created["username"] = username
        created["password"] = password
        created["is_admin"] = is_admin
        return app_module.AccountUser(id=20, username=username, is_admin=is_admin)

    monkeypatch.setattr(app_module, "_use_sqlalchemy_accounts", lambda: False)
    monkeypatch.setattr(app_module, "_get_supabase_account_by_username", lambda username: None)
    monkeypatch.setattr(app_module, "_create_supabase_account", fake_create)

    response = client.post(
        "/register",
        data={"username": "supabase_buyer", "password": "secret"},
        follow_redirects=False,
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/login")
    assert created == {
        "username": "supabase_buyer",
        "password": "secret",
        "is_admin": False,
    }


def test_login_reads_supabase_account_store_when_enabled(client, monkeypatch):
    password_hash = app_module._hash_password("secret")

    monkeypatch.setattr(app_module, "_use_sqlalchemy_accounts", lambda: False)
    monkeypatch.setattr(
        app_module,
        "_get_supabase_account_by_username",
        lambda username: app_module.AccountUser(
            id=21,
            username=username,
            password=password_hash,
            is_admin=False,
        ),
    )

    response = login(client, "supabase_buyer", "secret")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/")
    assert "auth_token=" in response.headers["Set-Cookie"]


def test_supabase_account_store_does_not_fallback_to_sqlalchemy(monkeypatch):
    monkeypatch.setitem(app.config, "TESTING", False)
    monkeypatch.setitem(app.config, "ACCOUNT_STORE", "supabase")
    monkeypatch.setattr(app_module, "SUPABASE_URL", "")
    monkeypatch.setattr(app_module, "SUPABASE_KEY", "")

    with pytest.raises(app_module.SupabaseError, match="Supabase is not configured"):
        app_module._create_account("no_fallback", "secret")


def test_product_and_setting_helpers_use_supabase_store(monkeypatch):
    calls = []

    def fake_supabase_request(method, path, payload=None, query=None, **kwargs):
        calls.append({"method": method, "path": path, "payload": payload, "query": query})
        if path.endswith("/products") and method == "POST":
            return [{"id": 50, **payload}]
        if path.endswith("/products") and method == "GET":
            return [{
                "id": 50,
                "name": "Supabase Product",
                "category": "Remote",
                "price": 25.0,
                "stock": 5,
                "description": "Stored remotely",
                "image_url": "https://example.com/item.png",
            }]
        if path.endswith("/settings") and method == "GET":
            return []
        if path.endswith("/settings") and method == "POST":
            return [{"id": 7, **payload}]
        raise AssertionError(f"Unexpected request: {method} {path}")

    monkeypatch.setitem(app.config, "TESTING", False)
    monkeypatch.setitem(app.config, "DATA_STORE", "supabase")
    monkeypatch.setattr(app_module, "_supabase_request", fake_supabase_request)

    product = app_module._create_product_record(
        name="Supabase Product",
        category="Remote",
        price=25.0,
        stock=5,
        description="Stored remotely",
        image_url="https://example.com/item.png",
    )
    setting = app_module._set_setting_value("upi_id", "merchant@upi")
    products = app_module._get_all_products()

    assert isinstance(product, app_module.ProductRecord)
    assert isinstance(setting, app_module.SettingRecord)
    assert products[0].name == "Supabase Product"
    assert ("POST", "/rest/v1/products") in [(call["method"], call["path"]) for call in calls]
    assert ("POST", "/rest/v1/settings") in [(call["method"], call["path"]) for call in calls]


def test_chat_reports_missing_ai_config_as_json(client, monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    response = client.post("/chat", json={"question": "Which diya should I buy?"})

    assert response.status_code == 503
    assert response.is_json
    assert response.get_json()["error"] == (
        "AI assistant is not configured. Missing: PINECONE_API_KEY, OPENROUTER_API_KEY."
    )


def test_first_request_initializes_database_for_wsgi_import(tmp_path):
    app.config.update(
        TESTING=True,
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{tmp_path / 'render.db'}",
    )
    app_module._database_initialized = False

    with app.app_context():
        db.session.remove()
        db.drop_all()

    response = app.test_client().get("/")

    assert response.status_code == 200
    with app.app_context():
        assert Product.query.count() > 0
