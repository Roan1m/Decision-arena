#!/usr/bin/env python3
"""Vercel/WSGI entrypoint for Decision Arena Studio."""

from web_app import app


if __name__ == "__main__":
    app.run()
