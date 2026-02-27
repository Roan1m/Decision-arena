#!/usr/bin/env python3
"""Vercel serverless entrypoint for Decision Arena Studio.

Vercel looks for api/index.py and expects a WSGI-compatible `app` object.
All routes are rewritten here via vercel.json.
"""

import sys
import os

# Ensure the project root is on the path so web_app imports work.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from web_app import app  # noqa: E402  (import after sys.path patch)

# Vercel needs the WSGI callable to be named `app` at module level.
__all__ = ["app"]
