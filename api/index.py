"""
api/index.py
============
Vercel Serverless Entrypoint for ATLAS Problem 1 + Problem 2 MONITOR.
Provides standard WSGI application (`app`) and BaseHTTPRequestHandler (`handler`).
Pure Python standard library — zero external frameworks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from server import dispatch_request, AtlasRequestHandler


def app(environ, start_response):
    """
    WSGI entrypoint for Vercel Python runtime.
    """
    method = environ.get("REQUEST_METHOD", "GET").upper()

    # Vercel rewrites pass the matched path in HTTP_X_MATCHED_PATH or HTTP_X_FORWARDED_URI
    matched_path = environ.get("HTTP_X_MATCHED_PATH", "")
    forwarded_uri = environ.get("HTTP_X_FORWARDED_URI", "").split("?")[0]
    path_info = environ.get("PATH_INFO", "/")

    # Determine requested path
    raw_path = matched_path or forwarded_uri or path_info
    if raw_path in ("/api/index.py", "/api/index", ""):
        raw_path = "/"

    query_str = environ.get("QUERY_STRING", "")
    content_len = int(environ.get("CONTENT_LENGTH", 0) or 0)
    body_bytes = environ["wsgi.input"].read(content_len) if content_len > 0 else b""

    status_code, resp_headers, resp_body = dispatch_request(
        method=method,
        path=raw_path,
        query_str=query_str,
        body_bytes=body_bytes,
    )

    status_phrases = {
        200: "200 OK",
        201: "201 Created",
        204: "204 No Content",
        400: "400 Bad Request",
        404: "404 Not Found",
        405: "405 Method Not Allowed",
        500: "500 Internal Server Error",
    }
    status_str = status_phrases.get(status_code, f"{status_code} Status")

    start_response(status_str, list(resp_headers.items()))
    return [resp_body]


# Fallback class handler if Vercel uses BaseHTTPRequestHandler
class handler(AtlasRequestHandler):
    pass
