import re

with open('flask_app/app.py', 'r') as f:
    content = f.read()

content = content.replace("JWT_SECRET = os.environ.get('JWT_SECRET')", "JWT_SECRET = os.environ.get('JWT_SECRET', 'test-secret')")
content = content.replace("app.secret_key = os.environ.get('SECRET_KEY')", "app.secret_key = os.environ.get('SECRET_KEY', 'test-secret')")

with open('flask_app/app.py', 'w') as f:
    f.write(content)
