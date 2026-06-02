"""
gunicorn.conf.py
────────────────
Gunicorn configuration for Render deployment.

Gunicorn auto-detects this file in the project root, so these settings
apply even if the start command is just `gunicorn app:app`.
"""

import os

# Bind to PORT env var (set by Render) or default to 10000
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"

# Workers: keep low to conserve memory on Render free/starter tier
workers = 2

# CRITICAL: timeout must be long enough for the OpenRouter free-tier LLM
# to respond. Free models (e.g. gemma-4-31b-it:free) can take 30-90 seconds.
# Default gunicorn timeout is 30s which causes WORKER TIMEOUT kills.
timeout = 180

# Graceful timeout before SIGKILL after SIGTERM
graceful_timeout = 30

# Keep-alive connections
keepalive = 5

# Preload the app to share memory between workers and catch import errors
# at boot instead of at first request
preload_app = True

# Log level
loglevel = "info"

# Access log format
accesslog = "-"
errorlog = "-"
