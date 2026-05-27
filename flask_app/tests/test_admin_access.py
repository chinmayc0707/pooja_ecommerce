import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
